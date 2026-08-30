"""Confidence-guided propagation for a globally coherent directed fur field.

The input field is already oriented by the multiview/global-sign stages.  This
pass repairs the remaining continuous seams without introducing semantic,
species, region, or view-specific rules:

1. A max-confidence watershed propagates joint axis/sign confidence over the
   canonical surface graph.
2. Only watersheds whose severe-conflict repair density is above the sample's
   own graph-wide defect density are accepted.
3. A monotone local pass removes residual negative edges from stronger to
   weaker roots while preserving the accepted watershed anchors.

Every accepted operation strictly improves a graph-continuity quantity and is
guarded against creating a new severe (at least 135 degree) edge.
"""

from __future__ import annotations

import hashlib
import heapq
import math

import numpy as np
import torch
import torch.nn.functional as F

from anigroom.flow.direction_geometry import parallel_transport_vectors
from anigroom.flow.global_sign_orientation import SEVERE_DOT_THRESHOLD
from anigroom.flow.view_cluster_refinement import CONFIDENCE_DECAY


EPS = 1.0e-8
CHANGE_DOT_THRESHOLD = 1.0 - 1.0e-7

__all__ = [
    "CHANGE_DOT_THRESHOLD",
    "refine_confidence_guided_directed_flow",
]


def _unit(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, dim=-1, eps=EPS)


def _distribution(value: torch.Tensor) -> dict[str, float | int]:
    finite = value[torch.isfinite(value)].detach().cpu().to(dtype=torch.float64)
    if finite.numel() == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean()),
        "p50": float(torch.quantile(finite, 0.50)),
        "p90": float(torch.quantile(finite, 0.90)),
        "p95": float(torch.quantile(finite, 0.95)),
        "max": float(finite.max()),
    }


def _edge_dots(
    direction: torch.Tensor,
    normals: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
) -> torch.Tensor:
    if edge_u.numel() == 0:
        return direction.new_empty((0,))
    transported = parallel_transport_vectors(
        direction[edge_v], normals[edge_v], normals[edge_u]
    )
    return (direction[edge_u] * transported).sum(dim=-1).clamp(-1.0, 1.0)


def _canonical_edges(
    *,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    canonical_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    pairs: dict[tuple[int, int], tuple[int, int]] = {}
    positive_input_count = int(edge_u.size)
    for first_value, second_value in zip(edge_u.tolist(), edge_v.tolist(), strict=True):
        first = int(first_value)
        second = int(second_value)
        if first == second:
            continue
        if int(canonical_rank[first]) > int(canonical_rank[second]):
            first, second = second, first
        key = (int(canonical_rank[first]), int(canonical_rank[second]))
        pairs[key] = (first, second)
    ordered = [pairs[key] for key in sorted(pairs)]
    return (
        np.asarray([pair[0] for pair in ordered], dtype=np.int64),
        np.asarray([pair[1] for pair in ordered], dtype=np.int64),
        positive_input_count,
    )


def _build_adjacency(
    *,
    root_count: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    canonical_rank: np.ndarray,
) -> list[list[tuple[int, int]]]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(root_count)]
    for edge_id, (first, second) in enumerate(
        zip(edge_u.tolist(), edge_v.tolist(), strict=True)
    ):
        adjacency[int(first)].append((edge_id, int(second)))
        adjacency[int(second)].append((edge_id, int(first)))
    for row in adjacency:
        row.sort(key=lambda item: (int(canonical_rank[item[1]]), int(item[0])))
    return adjacency


def _build_confidence_watershed(
    *,
    trust: np.ndarray,
    adjacency: list[list[tuple[int, int]]],
    canonical_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_count = int(trust.size)
    propagated = np.asarray(trust, dtype=np.float64).copy()
    owner = np.arange(root_count, dtype=np.int64)
    parent = np.arange(root_count, dtype=np.int64)
    version = np.zeros(root_count, dtype=np.int64)
    heap: list[tuple[float, int, int, int, int]] = []
    serial = 0
    for root_id in range(root_count):
        if propagated[root_id] <= 0.0:
            continue
        serial += 1
        heapq.heappush(
            heap,
            (
                -float(propagated[root_id]),
                int(canonical_rank[root_id]),
                serial,
                root_id,
                0,
            ),
        )

    while heap:
        negative_score, _, _, root_id, queued_version = heapq.heappop(heap)
        if queued_version != int(version[root_id]):
            continue
        current_score = -float(negative_score)
        if abs(current_score - float(propagated[root_id])) > 1.0e-15:
            continue
        for _, neighbor in adjacency[root_id]:
            proposal = current_score * CONFIDENCE_DECAY
            proposal_owner = int(owner[root_id])
            better = proposal > float(propagated[neighbor]) + 1.0e-15
            tied = abs(proposal - float(propagated[neighbor])) <= 1.0e-15
            owner_is_canonical = tied and (
                int(canonical_rank[proposal_owner])
                < int(canonical_rank[int(owner[neighbor])])
            )
            if not (better or owner_is_canonical):
                continue
            propagated[neighbor] = proposal
            owner[neighbor] = proposal_owner
            parent[neighbor] = root_id
            version[neighbor] += 1
            serial += 1
            heapq.heappush(
                heap,
                (
                    -proposal,
                    int(canonical_rank[neighbor]),
                    serial,
                    neighbor,
                    int(version[neighbor]),
                ),
            )
    return owner, parent, propagated


def _root_ids_hash(root_ids: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(root_ids.astype(np.int64, copy=False)).tobytes(order="C")
    ).hexdigest()


def _validate_inputs(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    observed: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    field_confidence: torch.Tensor,
    unary_normalized_margin: torch.Tensor,
    unary_vote_coherence: torch.Tensor,
    canonical_rank: torch.Tensor,
) -> tuple[int, torch.device, torch.dtype]:
    tensors = {
        "direction": direction,
        "normals": normals,
        "observed": observed,
        "edge_u": edge_u,
        "edge_v": edge_v,
        "field_confidence": field_confidence,
        "unary_normalized_margin": unary_normalized_margin,
        "unary_vote_coherence": unary_vote_coherence,
        "canonical_rank": canonical_rank,
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all confidence-guided direction inputs must be torch.Tensor values")
    if direction.ndim != 2 or direction.shape[-1] != 3:
        raise ValueError("direction must have shape [N, 3]")
    root_count = int(direction.shape[0])
    if root_count <= 0:
        raise ValueError("direction must contain at least one root")
    if normals.shape != (root_count, 3):
        raise ValueError("normals must have shape [N, 3]")
    for name, value in (
        ("observed", observed),
        ("field_confidence", field_confidence),
        ("unary_normalized_margin", unary_normalized_margin),
        ("unary_vote_coherence", unary_vote_coherence),
        ("canonical_rank", canonical_rank),
    ):
        if value.shape != (root_count,):
            raise ValueError(f"{name} must have shape [N]")
    if edge_u.ndim != 1 or edge_v.shape != edge_u.shape:
        raise ValueError("edge_u and edge_v must have the same shape [E]")
    device = direction.device
    if any(value.device != device for value in tensors.values()):
        raise ValueError("all confidence-guided direction inputs must share one device")
    if not direction.is_floating_point() or not normals.is_floating_point():
        raise TypeError("direction and normals must be floating-point tensors")
    for name, value in (
        ("field_confidence", field_confidence),
        ("unary_normalized_margin", unary_normalized_margin),
        ("unary_vote_coherence", unary_vote_coherence),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
    if observed.dtype != torch.bool:
        raise TypeError("observed must be a boolean tensor")
    for name, value in (("edge_u", edge_u), ("edge_v", edge_v), ("canonical_rank", canonical_rank)):
        if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
            raise TypeError(f"{name} must be an integer tensor")
    for name, value in tensors.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    if edge_u.numel() > 0 and bool(
        ((edge_u < 0) | (edge_u >= root_count) | (edge_v < 0) | (edge_v >= root_count)).any()
    ):
        raise ValueError("edge endpoints contain an out-of-range root index")
    if canonical_rank.numel() > 0 and int(torch.unique(canonical_rank).numel()) != root_count:
        raise ValueError("canonical_rank must contain unique values")
    if not bool((torch.linalg.vector_norm(direction, dim=-1) > EPS).all()):
        raise ValueError("direction must contain nonzero vectors")
    if not bool((torch.linalg.vector_norm(normals, dim=-1) > EPS).all()):
        raise ValueError("normals must contain nonzero vectors")
    dtype = torch.float64 if direction.dtype == torch.float64 else torch.float32
    return root_count, device, dtype


def refine_confidence_guided_directed_flow(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    observed: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    field_confidence: torch.Tensor,
    unary_normalized_margin: torch.Tensor,
    unary_vote_coherence: torch.Tensor,
    canonical_rank: torch.Tensor,
) -> dict[str, object]:
    """Repair directed-flow seams by propagating only stronger joint confidence.

    Joint confidence is the product of the trusted axial-field confidence, the
    global-sign normalized unary margin, and the global-sign vote coherence.
    The product is deliberately not thresholded: confidence decay and a
    canonical max-product watershed decide which source is more trustworthy at
    every root.
    """

    root_count, output_device, computation_dtype = _validate_inputs(
        direction=direction,
        normals=normals,
        observed=observed,
        edge_u=edge_u,
        edge_v=edge_v,
        field_confidence=field_confidence,
        unary_normalized_margin=unary_normalized_margin,
        unary_vote_coherence=unary_vote_coherence,
        canonical_rank=canonical_rank,
    )

    initial_direction = _unit(
        direction.detach().to(device="cpu", dtype=computation_dtype)
    )
    current_direction = initial_direction.clone()
    normal = _unit(normals.detach().to(device="cpu", dtype=computation_dtype))
    observed_cpu = observed.detach().to(device="cpu", dtype=torch.bool)
    rank_np = canonical_rank.detach().cpu().numpy().astype(np.int64, copy=False)
    edge_u_np, edge_v_np, input_edge_count = _canonical_edges(
        edge_u=edge_u.detach().cpu().numpy().astype(np.int64, copy=False),
        edge_v=edge_v.detach().cpu().numpy().astype(np.int64, copy=False),
        canonical_rank=rank_np,
    )
    canonical_edge_u = torch.from_numpy(edge_u_np)
    canonical_edge_v = torch.from_numpy(edge_v_np)
    adjacency = _build_adjacency(
        root_count=root_count,
        edge_u=edge_u_np,
        edge_v=edge_v_np,
        canonical_rank=rank_np,
    )
    edge_observed = observed_cpu[canonical_edge_u] & observed_cpu[canonical_edge_v]

    trust_np = np.clip(
        field_confidence.detach().cpu().numpy().astype(np.float64, copy=False)
        * unary_normalized_margin.detach().cpu().numpy().astype(np.float64, copy=False)
        * unary_vote_coherence.detach().cpu().numpy().astype(np.float64, copy=False),
        0.0,
        None,
    )
    trust = torch.from_numpy(trust_np).to(dtype=computation_dtype)
    owner, parent, propagated_score = _build_confidence_watershed(
        trust=trust_np,
        adjacency=adjacency,
        canonical_rank=rank_np,
    )

    grouped: dict[int, list[int]] = {}
    for root_id, owner_id in enumerate(owner.tolist()):
        grouped.setdefault(int(owner_id), []).append(root_id)
    grouped_items = sorted(
        grouped.items(), key=lambda item: int(rank_np[int(item[0])])
    )
    basins = [
        np.asarray(
            sorted(members, key=lambda root: int(rank_np[int(root)])),
            dtype=np.int64,
        )
        for _, members in grouped_items
    ]
    basin_owner = np.asarray([int(item[0]) for item in grouped_items], dtype=np.int64)
    root_to_basin = np.empty(root_count, dtype=np.int64)
    basin_key = np.empty(len(basins), dtype=np.int64)
    for basin_index, members in enumerate(basins):
        root_to_basin[members] = basin_index
        basin_key[basin_index] = min(int(rank_np[root]) for root in members.tolist())

    def synthesize_basin(active_members: np.ndarray) -> torch.Tensor:
        candidate = current_direction.clone()
        order = sorted(
            active_members.tolist(),
            key=lambda root: (-float(propagated_score[root]), int(rank_np[root])),
        )
        for root_id in order:
            if int(parent[root_id]) == root_id:
                continue
            higher_neighbors = [
                neighbor
                for _, neighbor in adjacency[root_id]
                if propagated_score[neighbor] > propagated_score[root_id] + 1.0e-15
                and int(owner[neighbor]) == int(owner[root_id])
            ]
            if not higher_neighbors:
                higher_neighbors = [int(parent[root_id])]
            higher_neighbors.sort(
                key=lambda neighbor: (
                    -float(propagated_score[neighbor]),
                    int(rank_np[neighbor]),
                )
            )
            neighbor_ids = torch.tensor(higher_neighbors, dtype=torch.long)
            transported = parallel_transport_vectors(
                candidate[neighbor_ids],
                normal[neighbor_ids],
                normal[root_id][None].expand(neighbor_ids.numel(), -1),
            )
            anchor = transported[0]
            aligned = (transported * anchor[None]).sum(dim=-1) >= 0.0
            transported = transported[aligned]
            neighbor_ids = neighbor_ids[aligned]
            if neighbor_ids.numel() == 0:
                continue
            weights = torch.from_numpy(
                propagated_score[neighbor_ids.numpy()].astype(np.float64)
            ).to(dtype=computation_dtype)
            reference = (weights[:, None] * transported).sum(dim=0)
            if float(torch.linalg.vector_norm(reference)) <= EPS:
                continue
            reference = _unit(reference[None])[0]
            propagated = current_direction.new_tensor(
                float(propagated_score[root_id])
            )
            numerator = (
                trust[root_id] * initial_direction[root_id]
                + propagated * reference
            )
            if float(torch.linalg.vector_norm(numerator)) <= EPS:
                continue
            value = _unit(numerator[None])[0]
            if float((value * normal[root_id]).sum()) < 0.0:
                value = _unit(
                    (
                        value
                        - 2.0 * (value * normal[root_id]).sum() * normal[root_id]
                    )[None]
                )[0]
            candidate[root_id] = value
        return candidate[torch.from_numpy(active_members)]

    initial_edge_dots = _edge_dots(
        initial_direction, normal, canonical_edge_u, canonical_edge_v
    )
    current_edge_dots = initial_edge_dots.clone()
    initial_graph_severe_count = int(
        (initial_edge_dots <= SEVERE_DOT_THRESHOLD).sum()
    )
    baseline_graph_severe_per_root = (
        float(initial_graph_severe_count) / float(root_count)
    )
    accepted_basin_indices: set[int] = set()
    basin_trace: list[dict[str, float | int | str]] = []

    while True:
        observed_negative = edge_observed & (current_edge_dots < 0.0)
        if not bool(observed_negative.any()):
            break
        negative_roots = torch.unique(
            torch.cat(
                (
                    canonical_edge_u[observed_negative],
                    canonical_edge_v[observed_negative],
                )
            )
        ).tolist()
        candidate_basins = sorted(
            {
                int(root_to_basin[int(root)])
                for root in negative_roots
                if int(root_to_basin[int(root)]) not in accepted_basin_indices
            },
            key=lambda basin_index: int(basin_key[basin_index]),
        )
        proposals: list[
            tuple[
                tuple[float, ...],
                int,
                np.ndarray,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                dict[str, float | int | str],
            ]
        ] = []
        for basin_index in candidate_basins:
            active_members = basins[basin_index]
            if active_members.size <= 1:
                continue
            incident_edge_ids_np = np.asarray(
                sorted(
                    {
                        edge_id
                        for root in active_members.tolist()
                        for edge_id, _ in adjacency[int(root)]
                    }
                ),
                dtype=np.int64,
            )
            incident_edge_ids = torch.from_numpy(incident_edge_ids_np)
            active_tensor = torch.from_numpy(active_members)
            trial_direction = current_direction.clone()
            trial_direction[active_tensor] = synthesize_basin(active_members)
            trial_dots = _edge_dots(
                trial_direction,
                normal,
                canonical_edge_u[incident_edge_ids],
                canonical_edge_v[incident_edge_ids],
            )
            current_incident = current_edge_dots[incident_edge_ids]
            net_negative_reduction = int((current_incident < 0.0).sum()) - int(
                (trial_dots < 0.0).sum()
            )
            if net_negative_reduction <= 0:
                continue
            severe_reduction = int(
                (current_incident <= SEVERE_DOT_THRESHOLD).sum()
            ) - int((trial_dots <= SEVERE_DOT_THRESHOLD).sum())
            if severe_reduction <= 0:
                continue
            hinge_improvement = float(
                torch.relu(-current_incident).sum()
                - torch.relu(-trial_dots).sum()
            )
            if not math.isfinite(hinge_improvement) or hinge_improvement <= 0.0:
                continue
            if bool(
                (
                    (current_incident > SEVERE_DOT_THRESHOLD)
                    & (trial_dots <= SEVERE_DOT_THRESHOLD)
                ).any()
            ):
                continue
            changed_dot = (
                current_direction[active_tensor] * trial_direction[active_tensor]
            ).sum(dim=-1).clamp(-1.0, 1.0)
            changed_members = changed_dot < CHANGE_DOT_THRESHOLD
            changed_count = int(changed_members.sum())
            if changed_count <= 0:
                continue
            repair_density = float(severe_reduction) / float(changed_count)
            if not (repair_density > baseline_graph_severe_per_root):
                continue
            changed_ids = active_members[changed_members.detach().cpu().numpy()]
            record: dict[str, float | int | str] = {
                "step": len(basin_trace) + 1,
                "basin_index": int(basin_index),
                "owner_root": int(basin_owner[basin_index]),
                "owner_canonical_rank": int(rank_np[int(basin_owner[basin_index])]),
                "basin_root_count": int(active_members.size),
                "changed_root_count": changed_count,
                "changed_root_ids_sha256": _root_ids_hash(changed_ids),
                "incident_severe_reduction": severe_reduction,
                "incident_severe_reduction_per_changed_root": repair_density,
                "baseline_graph_severe_per_root": baseline_graph_severe_per_root,
                "net_incident_negative_reduction": net_negative_reduction,
                "incident_hinge_improvement": hinge_improvement,
            }
            score = (
                float(severe_reduction),
                float(net_negative_reduction),
                float(hinge_improvement),
                float(-changed_count),
                float(-basin_key[basin_index]),
            )
            proposals.append(
                (
                    score,
                    basin_index,
                    active_members,
                    trial_direction[active_tensor],
                    incident_edge_ids,
                    trial_dots,
                    record,
                )
            )
        if not proposals:
            break
        (
            _,
            accepted_basin,
            accepted_members,
            accepted_values,
            accepted_edge_ids,
            accepted_dots,
            accepted_record,
        ) = max(proposals, key=lambda item: item[0])
        current_direction[torch.from_numpy(accepted_members)] = accepted_values
        current_edge_dots[accepted_edge_ids] = accepted_dots
        accepted_basin_indices.add(int(accepted_basin))
        basin_trace.append(accepted_record)

    basin_direction = current_direction.clone()
    basin_edge_dots = current_edge_dots.clone()
    basin_changed = (
        (initial_direction * basin_direction).sum(dim=-1).clamp(-1.0, 1.0)
        < CHANGE_DOT_THRESHOLD
    )
    protected_root = np.zeros(root_count, dtype=np.bool_)
    for basin_index in accepted_basin_indices:
        protected_root[int(basin_owner[basin_index])] = True

    local_trace: list[dict[str, float | int]] = []
    root_version = np.zeros(root_count, dtype=np.int64)
    local_update_count = np.zeros(root_count, dtype=np.int64)
    local_heap: list[
        tuple[
            float,
            float,
            float,
            int,
            int,
            int,
            int,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            dict[str, float | int],
        ]
    ] = []
    local_serial = 0

    def evaluate_local(
        root_id: int,
    ) -> tuple[
        tuple[float, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, float | int],
    ] | None:
        if bool(protected_root[root_id]):
            return None
        row = adjacency[root_id]
        if not row:
            return None
        incident_edge_ids = torch.tensor(
            [edge_id for edge_id, _ in row], dtype=torch.long
        )
        if not bool((current_edge_dots[incident_edge_ids] < 0.0).any()):
            return None
        neighbor_ids = torch.tensor(
            [neighbor for _, neighbor in row], dtype=torch.long
        )
        stronger = CONFIDENCE_DECAY * trust[neighbor_ids] > trust[root_id]
        if not bool(stronger.any()):
            return None
        selected_neighbors = neighbor_ids[stronger]
        selected_trust = trust[selected_neighbors]
        transported = parallel_transport_vectors(
            current_direction[selected_neighbors],
            normal[selected_neighbors],
            normal[root_id][None].expand(selected_neighbors.numel(), -1),
        )
        strongest_index = max(
            range(int(selected_neighbors.numel())),
            key=lambda index: (
                float(selected_trust[index]),
                -int(rank_np[int(selected_neighbors[index])]),
            ),
        )
        strongest_neighbor = int(selected_neighbors[strongest_index])
        anchor = transported[strongest_index]
        aligned = (transported * anchor[None]).sum(dim=-1) >= 0.0
        selected_neighbors = selected_neighbors[aligned]
        selected_trust = selected_trust[aligned]
        transported = transported[aligned]
        if selected_neighbors.numel() == 0:
            return None
        reference = (selected_trust[:, None] * transported).sum(dim=0)
        if float(torch.linalg.vector_norm(reference)) <= EPS:
            return None
        reference = _unit(reference[None])[0]
        propagated_confidence = CONFIDENCE_DECAY * selected_trust.max()
        numerator = (
            trust[root_id] * current_direction[root_id]
            + propagated_confidence * reference
        )
        if not bool(torch.isfinite(numerator).all()) or float(
            torch.linalg.vector_norm(numerator)
        ) <= EPS:
            return None
        candidate = _unit(numerator[None])[0]
        if float((candidate * normal[root_id]).sum()) < 0.0:
            candidate = _unit(
                (
                    candidate
                    - 2.0
                    * (candidate * normal[root_id]).sum()
                    * normal[root_id]
                )[None]
            )[0]

        current_incident = current_edge_dots[incident_edge_ids]
        trial_incident = current_incident.clone()
        root_is_u = canonical_edge_u[incident_edge_ids] == root_id
        if bool(root_is_u.any()):
            neighbor_for_u = canonical_edge_v[incident_edge_ids[root_is_u]]
            transported_neighbor = parallel_transport_vectors(
                current_direction[neighbor_for_u],
                normal[neighbor_for_u],
                normal[root_id][None].expand(neighbor_for_u.numel(), -1),
            )
            trial_incident[root_is_u] = (
                candidate[None] * transported_neighbor
            ).sum(dim=-1).clamp(-1.0, 1.0)
        root_is_v = ~root_is_u
        if bool(root_is_v.any()):
            neighbor_for_v = canonical_edge_u[incident_edge_ids[root_is_v]]
            transported_candidate = parallel_transport_vectors(
                candidate[None].expand(neighbor_for_v.numel(), -1),
                normal[root_id][None].expand(neighbor_for_v.numel(), -1),
                normal[neighbor_for_v],
            )
            trial_incident[root_is_v] = (
                current_direction[neighbor_for_v] * transported_candidate
            ).sum(dim=-1).clamp(-1.0, 1.0)
        net_negative_reduction = int((current_incident < 0.0).sum()) - int(
            (trial_incident < 0.0).sum()
        )
        if net_negative_reduction <= 0:
            return None
        hinge_improvement = float(
            torch.relu(-current_incident).sum()
            - torch.relu(-trial_incident).sum()
        )
        if not math.isfinite(hinge_improvement) or hinge_improvement <= 0.0:
            return None
        if bool(
            (
                (current_incident > SEVERE_DOT_THRESHOLD)
                & (trial_incident <= SEVERE_DOT_THRESHOLD)
            ).any()
        ):
            return None
        angular_change = float(
            torch.rad2deg(
                torch.acos(
                    (current_direction[root_id] * candidate).sum().clamp(-1.0, 1.0)
                )
            )
        )
        trust_advantage = float(propagated_confidence - trust[root_id])
        record: dict[str, float | int] = {
            "step": len(local_trace) + 1,
            "root_id": root_id,
            "canonical_rank": int(rank_np[root_id]),
            "trust": float(trust[root_id]),
            "stronger_neighbor_count": int(selected_neighbors.numel()),
            "strongest_neighbor_id": strongest_neighbor,
            "strongest_neighbor_trust": float(selected_trust.max()),
            "propagated_confidence": float(propagated_confidence),
            "net_incident_negative_reduction": net_negative_reduction,
            "incident_hinge_improvement": hinge_improvement,
            "angular_change_degrees": angular_change,
        }
        score = (
            float(net_negative_reduction),
            float(hinge_improvement),
            trust_advantage,
            float(-rank_np[root_id]),
        )
        return score, candidate, incident_edge_ids, trial_incident, record

    def enqueue_local(root_id: int) -> None:
        nonlocal local_serial
        proposal = evaluate_local(root_id)
        if proposal is None:
            return
        score, candidate, incident_edge_ids, trial_incident, record = proposal
        local_serial += 1
        heapq.heappush(
            local_heap,
            (
                -score[0],
                -score[1],
                -score[2],
                int(rank_np[root_id]),
                local_serial,
                root_id,
                int(root_version[root_id]),
                candidate,
                incident_edge_ids,
                trial_incident,
                record,
            ),
        )

    observed_negative = edge_observed & (current_edge_dots < 0.0)
    if bool(observed_negative.any()):
        initial_local_roots = torch.unique(
            torch.cat(
                (
                    canonical_edge_u[observed_negative],
                    canonical_edge_v[observed_negative],
                )
            )
        ).tolist()
        for root_id in sorted(
            (int(value) for value in initial_local_roots),
            key=lambda value: int(rank_np[value]),
        ):
            enqueue_local(root_id)

    while local_heap:
        (
            _,
            _,
            _,
            _,
            _,
            root_id,
            queued_version,
            _,
            _,
            _,
            _,
        ) = heapq.heappop(local_heap)
        if queued_version != int(root_version[root_id]):
            continue
        refreshed = evaluate_local(root_id)
        if refreshed is None:
            continue
        _, candidate, incident_edge_ids, trial_incident, record = refreshed
        current_direction[root_id] = candidate
        current_edge_dots[incident_edge_ids] = trial_incident
        local_update_count[root_id] += 1
        record["step"] = len(local_trace) + 1
        local_trace.append(record)
        affected_roots = {root_id}
        affected_roots.update(neighbor for _, neighbor in adjacency[root_id])
        for affected_root in affected_roots:
            root_version[affected_root] += 1
        for affected_root in sorted(
            affected_roots, key=lambda value: int(rank_np[value])
        ):
            enqueue_local(int(affected_root))

    final_edge_dots = _edge_dots(
        current_direction, normal, canonical_edge_u, canonical_edge_v
    )
    torch.testing.assert_close(
        final_edge_dots, current_edge_dots, atol=2.0e-6, rtol=2.0e-6
    )
    initial_negative = initial_edge_dots < 0.0
    basin_negative = basin_edge_dots < 0.0
    final_negative = final_edge_dots < 0.0
    initial_severe = initial_edge_dots <= SEVERE_DOT_THRESHOLD
    basin_severe = basin_edge_dots <= SEVERE_DOT_THRESHOLD
    final_severe = final_edge_dots <= SEVERE_DOT_THRESHOLD
    new_severe = (~initial_severe) & final_severe
    if bool(new_severe.any()):
        raise RuntimeError(
            "confidence-guided direction propagation introduced a new severe edge"
        )
    final_changed = (
        (initial_direction * current_direction).sum(dim=-1).clamp(-1.0, 1.0)
        < CHANGE_DOT_THRESHOLD
    )
    local_changed = (
        (basin_direction * current_direction).sum(dim=-1).clamp(-1.0, 1.0)
        < CHANGE_DOT_THRESHOLD
    )
    changed_angles = torch.rad2deg(
        torch.acos(
            (initial_direction * current_direction)
            .sum(dim=-1)
            .clamp(-1.0, 1.0)
        )
    )
    observed_initial_negative = edge_observed & initial_negative
    observed_basin_negative = edge_observed & basin_negative
    observed_final_negative = edge_observed & final_negative
    observed_initial_severe = edge_observed & initial_severe
    observed_basin_severe = edge_observed & basin_severe
    observed_final_severe = edge_observed & final_severe
    protected_tensor = torch.from_numpy(protected_root)

    report: dict[str, object] = {
        "schema": "anigroom.confidence_guided_directed_flow.v1",
        "root_count": root_count,
        "observed_root_count": int(observed_cpu.sum()),
        "graph": {
            "input_edge_count": input_edge_count,
            "canonical_unique_edge_count": int(canonical_edge_u.numel()),
            "observed_edge_count": int(edge_observed.sum()),
            "canonical_order": "endpoint pairs sorted by canonical root rank",
        },
        "confidence": {
            "definition": (
                "axis_view_cluster_final_confidence * "
                "global_unary_normalized_margin * global_unary_vote_coherence"
            ),
            "decay": float(CONFIDENCE_DECAY),
            "statistics": _distribution(trust),
        },
        "watershed": {
            "basin_count": len(basins),
            "accepted_basin_count": len(basin_trace),
            "accepted_trace": basin_trace,
            "baseline_graph_severe_per_root": baseline_graph_severe_per_root,
            "acceptance_rule": (
                "strict severe reduction; strict negative and hinge reduction; "
                "zero new severe edges; severe reduction per changed root above "
                "the input graph severe-edge count per root"
            ),
            "changed_root_count": int(basin_changed.sum()),
            "protected_owner_count": int(protected_tensor.sum()),
        },
        "local_cleanup": {
            "accepted_update_count": len(local_trace),
            "updated_root_count": int((torch.from_numpy(local_update_count) > 0).sum()),
            "changed_root_count": int(local_changed.sum()),
            "accepted_trace": local_trace,
            "termination": "no remaining monotone stronger-confidence proposal",
            "acceptance_rule": (
                "strict incident negative-count and hinge reduction; zero new "
                "severe edges; source confidence after decay exceeds target confidence"
            ),
        },
        "counts": {
            "all_edges": {
                "initial_negative": int(initial_negative.sum()),
                "post_watershed_negative": int(basin_negative.sum()),
                "final_negative": int(final_negative.sum()),
                "initial_severe": int(initial_severe.sum()),
                "post_watershed_severe": int(basin_severe.sum()),
                "final_severe": int(final_severe.sum()),
                "new_severe": int(new_severe.sum()),
            },
            "observed_edges": {
                "initial_negative": int(observed_initial_negative.sum()),
                "post_watershed_negative": int(observed_basin_negative.sum()),
                "final_negative": int(observed_final_negative.sum()),
                "initial_severe": int(observed_initial_severe.sum()),
                "post_watershed_severe": int(observed_basin_severe.sum()),
                "final_severe": int(observed_final_severe.sum()),
                "new_severe": int((edge_observed & new_severe).sum()),
            },
            "changed_roots": int(final_changed.sum()),
        },
        "changed_angle_degrees": _distribution(changed_angles[final_changed]),
        "zero_new_severe_verification": {
            "new_severe_edge_count": int(new_severe.sum()),
            "passed": not bool(new_severe.any()),
        },
    }

    return {
        "direction": current_direction.to(
            device=output_device, dtype=direction.dtype
        ),
        "watershed_direction": basin_direction.to(
            device=output_device, dtype=direction.dtype
        ),
        "joint_confidence": trust.to(
            device=output_device, dtype=direction.dtype
        ),
        "watershed_owner": torch.from_numpy(owner).to(device=output_device),
        "watershed_parent": torch.from_numpy(parent).to(device=output_device),
        "watershed_propagated_confidence": torch.from_numpy(
            propagated_score
        ).to(device=output_device, dtype=direction.dtype),
        "watershed_changed_mask": basin_changed.to(device=output_device),
        "local_changed_mask": local_changed.to(device=output_device),
        "changed_mask": final_changed.to(device=output_device),
        "protected_owner_mask": protected_tensor.to(device=output_device),
        "local_update_count": torch.from_numpy(local_update_count).to(
            device=output_device
        ),
        "edge_u": canonical_edge_u.to(device=output_device),
        "edge_v": canonical_edge_v.to(device=output_device),
        "initial_edge_dots": initial_edge_dots.to(
            device=output_device, dtype=direction.dtype
        ),
        "watershed_edge_dots": basin_edge_dots.to(
            device=output_device, dtype=direction.dtype
        ),
        "final_edge_dots": final_edge_dots.to(
            device=output_device, dtype=direction.dtype
        ),
        "new_severe_edge_mask": new_severe.to(device=output_device),
        "report": report,
    }

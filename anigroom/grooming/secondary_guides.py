"""Parent-conditioned secondary guide roots and residual interpolation.

The primary guides own the low-frequency groom.  A denser secondary layer
stores only zero-centered geometry residuals.  Secondary roots are distributed
inside primary-guide surface cells, so their density follows the primary guide
layout instead of reverting to a global mesh-area distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from anigroom.flow.direction_geometry import parallel_transport_vectors
from anigroom.mesh_roots import (
    SurfaceRoots,
    TriangleMesh,
    sample_surface_candidates,
)
from anigroom.surface_interpolation import (
    LocalSurfaceSupport,
    SurfaceFieldInterpolator,
    SurfaceSupport,
    interpolate_physical,
    local_surface_weights,
)

from .geometry_residuals import (
    DecodedGeometryResiduals,
    RenderGeometryResidualField,
    local_components_to_world,
    vector_to_local_components,
)


EPS = 1.0e-8


@dataclass(frozen=True)
class SecondarySurfaceRoots:
    """Fixed secondary roots, grouped by their primary surface parent."""

    roots: SurfaceRoots
    parent_ids: np.ndarray
    report: dict[str, float | int]


@dataclass(frozen=True)
class InterpolatedGeometryResiduals:
    """Residual coordinates sampled from secondary guides at query roots."""

    raw: dict[str, torch.Tensor]
    decoded: DecodedGeometryResiduals


def _seeded_local_fps(
    points: np.ndarray,
    seed_point: np.ndarray,
    count: int,
) -> tuple[np.ndarray, float]:
    """Select local candidates by FPS with the primary guide as a fixed seed."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("local FPS points must have shape [N, 3]")
    if seed_point.shape != (3,):
        raise ValueError("local FPS seed_point must have shape [3]")
    if count < 0 or count > int(points.shape[0]):
        raise ValueError("local FPS count is outside the candidate range")

    min_distance = np.sum((points - seed_point[None, :]) ** 2, axis=1)
    selected = np.empty((count,), dtype=np.int64)
    available = np.ones((int(points.shape[0]),), dtype=bool)
    for index in range(count):
        score = np.where(available, min_distance, -np.inf)
        candidate = int(np.argmax(score))
        if not np.isfinite(score[candidate]):
            raise RuntimeError("local FPS exhausted its candidate cell")
        selected[index] = candidate
        available[candidate] = False
        distance = np.sum((points - points[candidate : candidate + 1]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)

    next_score = float(np.max(np.where(available, min_distance, -np.inf)))
    return selected, next_score


@torch.no_grad()
def initialize_parent_conditioned_secondary_roots(
    mesh: TriangleMesh,
    primary_roots: SurfaceRoots,
    primary_interpolator: SurfaceFieldInterpolator,
    secondary_root_count: int,
    *,
    candidate_multiplier: float,
    seed: int,
    device: torch.device | str,
) -> SecondarySurfaceRoots:
    """Create balanced local FPS children around every primary guide.

    Every primary guide is retained as one exact secondary anchor.  Remaining
    roots are sampled from dense area-uniform mesh candidates, assigned to the
    nearest topology-valid primary guide, and selected by local FPS.  All
    primary cells receive the same base number of children; any remainder is
    assigned to cells with the largest uncovered radius.
    """

    primary_count = int(primary_roots.points.shape[0])
    secondary_root_count = int(secondary_root_count)
    if primary_count <= 0:
        raise ValueError("secondary guides require primary guides")
    if secondary_root_count < primary_count:
        raise ValueError(
            "secondary_root_count must be at least the primary guide count"
        )
    if float(candidate_multiplier) <= 0.0:
        raise ValueError("candidate_multiplier must be positive")

    extra_count = secondary_root_count - primary_count
    base_extra, remainder = divmod(extra_count, primary_count)
    candidate_count = max(
        secondary_root_count,
        int(np.ceil(float(secondary_root_count) * float(candidate_multiplier))),
    )
    candidates = sample_surface_candidates(mesh, candidate_count, int(seed))
    target_device = torch.device(device)
    candidate_points = torch.as_tensor(
        candidates.points,
        device=target_device,
        dtype=torch.float32,
    )
    candidate_faces = torch.as_tensor(
        candidates.face_ids,
        device=target_device,
        dtype=torch.long,
    )
    candidate_support = primary_interpolator.build_support(
        candidate_points,
        candidate_faces,
    )
    owner = candidate_support.indices[:, 0].detach().cpu().numpy().astype(np.int64)
    cell_counts = np.bincount(owner, minlength=primary_count)
    minimum_cell_count = int(cell_counts.min())
    if minimum_cell_count < base_extra:
        deficient = np.flatnonzero(cell_counts < base_extra)
        raise RuntimeError(
            "secondary candidate pool cannot provide balanced primary cells: "
            f"required={base_extra}, deficient_cells={int(deficient.size)}, "
            f"minimum_candidates={minimum_cell_count}; "
            "increase SECONDARY_GUIDE_CANDIDATE_MULTIPLIER"
        )

    cell_order = np.argsort(owner, kind="stable")
    cell_offsets = np.concatenate([[0], np.cumsum(cell_counts)])
    selected_by_parent: list[np.ndarray] = []
    next_scores = np.full((primary_count,), -np.inf, dtype=np.float64)
    for parent_id in range(primary_count):
        local_ids = cell_order[cell_offsets[parent_id] : cell_offsets[parent_id + 1]]
        local_points = candidates.points[local_ids]
        local_selected, next_score = _seeded_local_fps(
            local_points,
            primary_roots.points[parent_id],
            base_extra,
        )
        selected_by_parent.append(local_ids[local_selected])
        if int(local_ids.size) > base_extra:
            next_scores[parent_id] = next_score

    if remainder > 0:
        eligible = np.flatnonzero(np.isfinite(next_scores))
        if int(eligible.size) < remainder:
            raise RuntimeError(
                "secondary candidate pool cannot allocate the balanced remainder: "
                f"required={remainder}, eligible_cells={int(eligible.size)}; "
                "increase SECONDARY_GUIDE_CANDIDATE_MULTIPLIER"
            )
        # Stable tie handling makes the placement deterministic across devices.
        order = np.lexsort((eligible, -next_scores[eligible]))
        extra_parents = eligible[order[:remainder]]
        for parent_id in extra_parents.tolist():
            local_ids = cell_order[cell_offsets[parent_id] : cell_offsets[parent_id + 1]]
            local_selected, _ = _seeded_local_fps(
                candidates.points[local_ids],
                primary_roots.points[parent_id],
                base_extra + 1,
            )
            selected_by_parent[parent_id] = local_ids[local_selected]

    points: list[np.ndarray] = []
    face_ids: list[np.ndarray] = []
    barycentric: list[np.ndarray] = []
    parent_ids: list[np.ndarray] = []
    selected_candidate_ids: list[np.ndarray] = []
    children_per_parent = np.empty((primary_count,), dtype=np.int64)
    for parent_id, candidate_ids in enumerate(selected_by_parent):
        points.append(primary_roots.points[parent_id : parent_id + 1])
        face_ids.append(primary_roots.face_ids[parent_id : parent_id + 1])
        barycentric.append(primary_roots.barycentric[parent_id : parent_id + 1])
        parent_ids.append(np.asarray([parent_id], dtype=np.int64))
        selected_candidate_ids.append(np.asarray([-1], dtype=np.int64))
        if candidate_ids.size:
            points.append(candidates.points[candidate_ids])
            face_ids.append(candidates.face_ids[candidate_ids])
            barycentric.append(candidates.barycentric[candidate_ids])
            parent_ids.append(
                np.full((int(candidate_ids.size),), parent_id, dtype=np.int64)
            )
            selected_candidate_ids.append(candidate_ids.astype(np.int64, copy=False))
        children_per_parent[parent_id] = 1 + int(candidate_ids.size)

    roots = SurfaceRoots(
        points=np.concatenate(points, axis=0).astype(np.float32, copy=False),
        face_ids=np.concatenate(face_ids, axis=0).astype(np.int64, copy=False),
        barycentric=np.concatenate(barycentric, axis=0).astype(np.float32, copy=False),
        selected_candidate_ids=np.concatenate(selected_candidate_ids, axis=0),
        candidate_count=int(candidate_count),
    )
    parents = np.concatenate(parent_ids, axis=0)
    if int(roots.points.shape[0]) != secondary_root_count:
        raise RuntimeError(
            f"secondary root count mismatch: {roots.points.shape[0]} != {secondary_root_count}"
        )
    if not np.array_equal(
        np.bincount(parents, minlength=primary_count),
        children_per_parent,
    ):
        raise RuntimeError("secondary parent grouping is inconsistent")

    selected_from_pool = roots.selected_candidate_ids >= 0
    selected_distance = np.linalg.norm(
        roots.points[selected_from_pool]
        - primary_roots.points[parents[selected_from_pool]],
        axis=-1,
    )
    report: dict[str, float | int] = {
        "primary_root_count": primary_count,
        "secondary_root_count": secondary_root_count,
        "candidate_count": int(candidate_count),
        "base_children_per_parent": int(base_extra + 1),
        "extra_parent_count": int(remainder),
        "children_per_parent_min": int(children_per_parent.min()),
        "children_per_parent_max": int(children_per_parent.max()),
        "candidate_cell_min": int(cell_counts.min()),
        "candidate_cell_max": int(cell_counts.max()),
        "selected_parent_distance_mean": float(selected_distance.mean())
        if selected_distance.size
        else 0.0,
        "selected_parent_distance_max": float(selected_distance.max(initial=0.0)),
        "support_fallback_query_count": int(
            candidate_support.report.get("fallback_query_count", 0)
        ),
    }
    return SecondarySurfaceRoots(roots=roots, parent_ids=parents, report=report)


@torch.no_grad()
def build_parent_conditioned_query_support(
    query_points: torch.Tensor,
    query_primary_support: SurfaceSupport,
    secondary_points: torch.Tensor,
    secondary_parent_ids: torch.Tensor,
    *,
    neighbor_count: int,
    chunk_size: int = 32768,
) -> LocalSurfaceSupport:
    """Select secondary neighbors only from topology-valid primary support."""

    if query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError("query_points must have shape [Q, 3]")
    if secondary_points.ndim != 2 or secondary_points.shape[-1] != 3:
        raise ValueError("secondary_points must have shape [S, 3]")
    if secondary_parent_ids.shape != (secondary_points.shape[0],):
        raise ValueError("secondary_parent_ids must have shape [S]")
    if int(query_primary_support.query_count) != int(query_points.shape[0]):
        raise ValueError("primary support query count mismatch")
    if int(neighbor_count) <= 0:
        raise ValueError("neighbor_count must be positive")

    device = query_points.device
    parent_ids = secondary_parent_ids.to(device=device, dtype=torch.long)
    primary_support = query_primary_support.indices.to(device=device, dtype=torch.long)
    primary_count = int(
        torch.maximum(parent_ids.max(), primary_support.max()).detach().cpu()
    ) + 1
    order = torch.argsort(parent_ids, stable=True)
    sorted_parent = parent_ids[order]
    counts = torch.bincount(parent_ids, minlength=primary_count)
    max_children = int(counts.max().detach().cpu())
    offsets = torch.cumsum(counts, dim=0) - counts
    slots = torch.arange(parent_ids.shape[0], device=device) - torch.repeat_interleave(
        offsets,
        counts,
    )
    buckets = torch.full(
        (primary_count, max_children),
        -1,
        device=device,
        dtype=torch.long,
    )
    buckets[sorted_parent, slots] = order

    candidate_width = int(primary_support.shape[1]) * max_children
    k = min(int(neighbor_count), int(secondary_points.shape[0]))
    output = torch.empty((int(query_points.shape[0]), k), device=device, dtype=torch.long)
    minimum_candidate_count = candidate_width
    for begin in range(0, int(query_points.shape[0]), int(chunk_size)):
        end = min(begin + int(chunk_size), int(query_points.shape[0]))
        candidates = buckets[primary_support[begin:end]].reshape(end - begin, candidate_width)
        valid = candidates >= 0
        valid_count = valid.sum(dim=1)
        minimum_candidate_count = min(
            minimum_candidate_count,
            int(valid_count.min().detach().cpu()),
        )
        if bool((valid_count < k).any()):
            raise RuntimeError(
                "secondary interpolation has fewer topology-valid candidates "
                f"than K={k}; minimum={int(valid_count.min().detach().cpu())}"
            )
        safe = candidates.clamp_min(0)
        delta = query_points[begin:end, None, :] - secondary_points[safe]
        distance_sq = (delta * delta).sum(dim=-1)
        distance_sq.masked_fill_(~valid, torch.inf)
        slots = torch.topk(
            distance_sq,
            k=k,
            dim=1,
            largest=False,
            sorted=True,
        ).indices
        output[begin:end] = torch.gather(candidates, 1, slots)

    return LocalSurfaceSupport(
        indices=output,
        report={
            "query_count": int(query_points.shape[0]),
            "source_count": int(secondary_points.shape[0]),
            "neighbor_count": int(k),
            "primary_support_width": int(primary_support.shape[1]),
            "children_per_parent_max": int(max_children),
            "candidate_count_min": int(minimum_candidate_count),
        },
    )


def interpolate_secondary_geometry_residuals(
    field: RenderGeometryResidualField,
    source_normals: torch.Tensor,
    source_tangents: torch.Tensor,
    source_bitangents: torch.Tensor,
    query_points: torch.Tensor,
    query_normals: torch.Tensor,
    query_tangents: torch.Tensor,
    query_bitangents: torch.Tensor,
    source_points: torch.Tensor,
    support: LocalSurfaceSupport,
) -> InterpolatedGeometryResiduals:
    """Interpolate zero-centered residual coordinates into query frames."""

    weights = local_surface_weights(query_points, source_points, support)
    raw = {
        name: interpolate_physical(parameter, support.indices, weights)
        for name, parameter in field.named_parameters()
        if name != "direction_local_raw"
    }
    source_decoded = field.decode()
    source_world = local_components_to_world(
        source_decoded.direction_local,
        source_normals,
        source_tangents,
        source_bitangents,
        normalize=False,
    )
    gathered_world = source_world[support.indices]
    gathered_normals = source_normals[support.indices]
    target_normals = query_normals[:, None, :].expand_as(gathered_normals)
    gathered_magnitude = torch.linalg.norm(
        gathered_world,
        dim=-1,
        keepdim=True,
    )
    transported = parallel_transport_vectors(
        gathered_world,
        gathered_normals,
        target_normals,
    ) * gathered_magnitude
    world_residual = (transported * weights[..., None]).sum(dim=1)
    direction_local = vector_to_local_components(
        world_residual,
        query_normals,
        query_tangents,
        query_bitangents,
    )
    # Direction residuals are vectors, not unit axes.  Parallel transport and
    # weighted averaging can legitimately produce a local component outside
    # [-1, 1], so an atanh round-trip would silently clamp its magnitude.
    raw["direction_local_raw"] = direction_local
    decoded = DecodedGeometryResiduals(
        length=torch.tanh(raw["length_raw"]),
        root_width_log_ratio=torch.asinh(raw["root_width_raw"]),
        tip_width_logit_delta=torch.asinh(raw["tip_width_ratio_raw"]),
        width_taper_log_ratio=torch.asinh(raw["width_taper_raw"]),
        curl_radius=torch.tanh(raw["curl_radius_raw"]),
        frizz=torch.tanh(raw["frizz_raw"]),
        child_radius_log_ratio=torch.asinh(raw["child_radius_raw"]),
        clump_strength=torch.tanh(raw["clump_strength_raw"]),
        direction_local=direction_local,
    )
    return InterpolatedGeometryResiduals(raw=raw, decoded=decoded)

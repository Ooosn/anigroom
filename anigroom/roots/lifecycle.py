"""Root movement, densification, and pruning primitives.

This module is independent of the training loop.  It consumes root-level
statistics accumulated by rendering/backpropagation and proposes structure
updates.  The trainer remains responsible for collecting the statistics and
for rebuilding optimizers after insertion/pruning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


EPS = 1e-8


@dataclass
class RootLifecycleState:
    """Root positions and surface coordinates."""

    points: torch.Tensor
    face_ids: torch.Tensor
    barycentric: torch.Tensor

    def validate(self) -> None:
        if self.points.ndim != 2 or self.points.shape[-1] != 3:
            raise ValueError("points must have shape [R, 3]")
        if self.face_ids.shape != (self.points.shape[0],):
            raise ValueError("face_ids must have shape [R]")
        if self.barycentric.shape != (self.points.shape[0], 3):
            raise ValueError("barycentric must have shape [R, 3]")


@dataclass
class RootStats:
    """Accumulated root-level evidence over a structure-update window."""

    root_grad_abs_sum: torch.Tensor
    gaussian_grad_abs_sum: torch.Tensor
    gaussian_contrib_sum: torch.Tensor
    visible_count: torch.Tensor
    residual_sum: torch.Tensor | None = None
    opacity_mean: torch.Tensor | None = None

    def validate(self) -> None:
        root_count = self.root_grad_abs_sum.shape[0]
        for name, value in [
            ("gaussian_grad_abs_sum", self.gaussian_grad_abs_sum),
            ("gaussian_contrib_sum", self.gaussian_contrib_sum),
            ("visible_count", self.visible_count),
        ]:
            if value.shape[0] != root_count:
                raise ValueError(f"{name} has mismatched root dimension")
        if self.residual_sum is not None and self.residual_sum.shape[0] != root_count:
            raise ValueError("residual_sum has mismatched root dimension")
        if self.opacity_mean is not None and self.opacity_mean.shape[0] != root_count:
            raise ValueError("opacity_mean has mismatched root dimension")

    @property
    def device(self) -> torch.device:
        return self.root_grad_abs_sum.device

    @property
    def root_count(self) -> int:
        return int(self.root_grad_abs_sum.shape[0])


@dataclass(frozen=True)
class DensifyConfig:
    grad_threshold: float = 0.25
    visibility_threshold: float = 1.0
    residual_threshold: float = 0.0
    max_new_roots: int = 1024
    children_per_parent: int = 2
    barycentric_step: float = 0.08
    replace_parent: bool = True
    neighbor_count: int = 12
    candidate_rings: int = 3
    candidate_face_count: int = 32
    min_child_distance: float = 0.0


@dataclass(frozen=True)
class PruneConfig:
    min_visible_count: float = 1.0
    min_contribution: float = 1e-6
    min_opacity: float = 0.0
    max_prune_fraction: float = 0.10


@dataclass
class RootStructureUpdate:
    parent_indices: torch.Tensor
    child_parent_indices: torch.Tensor
    new_face_ids: torch.Tensor
    new_barycentric: torch.Tensor
    prune_mask: torch.Tensor
    scores: dict[str, torch.Tensor]


def normalized_root_need(stats: RootStats) -> dict[str, torch.Tensor]:
    """Compute thresholdable root evidence without percentile normalization."""

    stats.validate()
    denom = stats.gaussian_contrib_sum.clamp_min(EPS)
    gaussian_grad = stats.gaussian_grad_abs_sum / denom
    root_grad = stats.root_grad_abs_sum / stats.visible_count.clamp_min(1.0)
    visibility = stats.visible_count
    if stats.residual_sum is None:
        residual = torch.zeros_like(root_grad)
    else:
        residual = stats.residual_sum / stats.visible_count.clamp_min(1.0)
    need = gaussian_grad + root_grad + residual
    return {
        "need": need.reshape(-1),
        "gaussian_grad": gaussian_grad.reshape(-1),
        "root_grad": root_grad.reshape(-1),
        "visibility": visibility.reshape(-1),
        "residual": residual.reshape(-1),
    }


def select_densify_parents(stats: RootStats, config: DensifyConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scores = normalized_root_need(stats)
    need = scores["need"]
    valid = scores["visibility"] >= float(config.visibility_threshold)
    valid = valid & (need >= float(config.grad_threshold))
    if float(config.residual_threshold) > 0.0:
        valid = valid & (scores["residual"] >= float(config.residual_threshold))
    candidates = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if candidates.numel() == 0:
        return candidates, scores
    order = torch.argsort(need[candidates], descending=True)
    limit = max(0, int(config.max_new_roots) // max(1, int(config.children_per_parent)))
    parents = candidates[order[:limit]]
    return parents, scores


def _barycentric_offsets(device: torch.device, dtype: torch.dtype, step: float) -> torch.Tensor:
    offsets = torch.tensor(
        [
            [step, -step, 0.0],
            [step, 0.0, -step],
            [-step, step, 0.0],
            [0.0, step, -step],
            [-step, 0.0, step],
            [0.0, -step, step],
        ],
        device=device,
        dtype=dtype,
    )
    return offsets


def _surface_barycentric_templates(device: torch.device, dtype: torch.dtype, candidate_rings: int) -> torch.Tensor:
    """Deterministic face samples used for child placement candidates."""

    center = torch.tensor([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], device=device, dtype=dtype)
    coarse = torch.tensor(
        [
            [0.58, 0.21, 0.21],
            [0.21, 0.58, 0.21],
            [0.21, 0.21, 0.58],
            [0.46, 0.46, 0.08],
            [0.46, 0.08, 0.46],
            [0.08, 0.46, 0.46],
        ],
        device=device,
        dtype=dtype,
    )
    if int(candidate_rings) <= 1:
        return torch.cat([center, coarse[:3]], dim=0)
    if int(candidate_rings) == 2:
        return torch.cat([center, coarse], dim=0)
    fine = torch.tensor(
        [
            [0.72, 0.14, 0.14],
            [0.14, 0.72, 0.14],
            [0.14, 0.14, 0.72],
            [0.36, 0.36, 0.28],
            [0.36, 0.28, 0.36],
            [0.28, 0.36, 0.36],
        ],
        device=device,
        dtype=dtype,
    )
    return torch.cat([center, coarse, fine], dim=0)


def _build_face_adjacency(faces: torch.Tensor) -> list[list[int]]:
    """Build triangle face adjacency through shared edges.

    This is used only at structure-update time.  It deliberately follows mesh
    topology instead of falling back to nearest face centers, so child roots stay
    on a local surface neighborhood.
    """

    faces_cpu = faces.detach().cpu().long().tolist()
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(faces_cpu):
        edges = ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        for a, b in edges:
            key = (a, b) if a <= b else (b, a)
            edge_to_faces.setdefault(key, []).append(face_id)

    adjacency: list[set[int]] = [set() for _ in faces_cpu]
    for linked_faces in edge_to_faces.values():
        if len(linked_faces) < 2:
            continue
        for src in linked_faces:
            for dst in linked_faces:
                if src != dst:
                    adjacency[src].add(dst)
    return [sorted(items) for items in adjacency]


def _topology_face_neighborhoods(
    parent_faces: torch.Tensor,
    faces: torch.Tensor,
    *,
    candidate_rings: int,
    candidate_face_count: int,
) -> torch.Tensor:
    """Return padded topology-ring face neighborhoods for parent faces."""

    if parent_faces.numel() == 0:
        return parent_faces.new_empty((0, 0))
    adjacency = _build_face_adjacency(faces)
    limit = max(1, int(candidate_face_count))
    max_ring = max(0, int(candidate_rings))
    rows: list[list[int]] = []
    for face_value in parent_faces.detach().cpu().long().tolist():
        visited = {int(face_value)}
        frontier = [int(face_value)]
        ordered = [int(face_value)]
        for _ in range(max_ring):
            next_frontier: list[int] = []
            for face_id in frontier:
                for neighbor in adjacency[face_id]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
                    ordered.append(neighbor)
                    if len(ordered) >= limit:
                        break
                if len(ordered) >= limit:
                    break
            frontier = next_frontier
            if not frontier or len(ordered) >= limit:
                break
        if len(ordered) < limit:
            ordered.extend([ordered[-1]] * (limit - len(ordered)))
        rows.append(ordered[:limit])
    return torch.tensor(rows, device=parent_faces.device, dtype=parent_faces.dtype)


def _multi_face_child_candidates(
    state: RootLifecycleState,
    parent_indices: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    candidate_face_count: int,
    candidate_rings: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate nearby cross-face child candidates around each parent root."""

    parent_count = int(parent_indices.numel())
    if parent_count == 0:
        empty_faces = state.face_ids.new_empty((0, 0))
        empty_bary = state.barycentric.new_empty((0, 0, 3))
        empty_points = state.points.new_empty((0, 0, 3))
        return empty_faces, empty_bary, empty_points

    templates = _surface_barycentric_templates(state.points.device, state.points.dtype, int(candidate_rings))
    face_ids = _topology_face_neighborhoods(
        state.face_ids[parent_indices],
        faces,
        candidate_rings=int(candidate_rings),
        candidate_face_count=max(1, int(candidate_face_count)),
    )
    tri = vertices[faces[face_ids]]
    candidate_points = (tri[:, :, None] * templates[None, None, :, :, None]).sum(dim=3)
    flat_faces = face_ids[:, :, None].expand(-1, -1, templates.shape[0]).reshape(parent_count, -1)
    flat_bary = templates[None, None].expand(parent_count, face_ids.shape[1], -1, -1).reshape(parent_count, -1, 3)
    flat_points = candidate_points.reshape(parent_count, -1, 3)
    return flat_faces, flat_bary, flat_points


def propose_split_children(
    state: RootLifecycleState,
    parent_indices: torch.Tensor,
    children_per_parent: int,
    barycentric_step: float,
    *,
    vertices: torch.Tensor | None = None,
    faces: torch.Tensor | None = None,
    neighbor_count: int = 12,
    candidate_rings: int = 3,
    candidate_face_count: int = 32,
    min_child_distance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place split children by choosing locally emptier surface candidates.

    For each selected parent, this routine samples candidate positions on
    nearby mesh faces and keeps the locally emptiest candidates in 3D.  This is
    a split operation: the returned child_parent_indices records which old
    parent each child should inherit/interpolate from before that parent is
    removed.
    """

    state.validate()
    if parent_indices.numel() == 0:
        return (
            state.face_ids.new_empty((0,)),
            state.face_ids.new_empty((0,)),
            state.barycentric.new_empty((0, 3)),
        )
    children_per_parent = max(1, int(children_per_parent))
    all_bary = []
    all_faces = []
    all_parents = []
    if vertices is None or faces is None:
        raise ValueError("vertices and faces are required for formal root split; no fallback child placement is allowed")
    candidate_faces, candidate_bary, candidate_points = _multi_face_child_candidates(
        state,
        parent_indices,
        vertices,
        faces,
        candidate_face_count=int(candidate_face_count),
        candidate_rings=int(candidate_rings),
    )
    nearest_count = max(1, min(int(neighbor_count), int(state.points.shape[0])))
    flat_candidates = candidate_points.reshape(-1, 3)
    local_distance_flat = torch.empty((flat_candidates.shape[0],), device=state.points.device, dtype=state.points.dtype)
    # Candidate count is parent_count * candidate_face_count * template_count.
    # A full candidate-to-root distance matrix can reach multiple GB during
    # densification, so compute the kNN distance in chunks.
    candidate_chunk = 1024
    for begin in range(0, int(flat_candidates.shape[0]), candidate_chunk):
        end = min(begin + candidate_chunk, int(flat_candidates.shape[0]))
        dist = torch.cdist(flat_candidates[begin:end], state.points)
        nearest_values = torch.topk(dist, k=nearest_count, largest=False, dim=-1).values
        local_distance_flat[begin:end] = nearest_values[:, -1]
    local_distance = local_distance_flat.view(candidate_points.shape[0], candidate_points.shape[1])
    if float(min_child_distance) > 0.0:
        closest_flat = torch.empty((flat_candidates.shape[0],), device=state.points.device, dtype=state.points.dtype)
        for begin in range(0, int(flat_candidates.shape[0]), candidate_chunk):
            end = min(begin + candidate_chunk, int(flat_candidates.shape[0]))
            dist = torch.cdist(flat_candidates[begin:end], state.points)
            closest_flat[begin:end] = torch.min(dist, dim=-1).values
        too_close = closest_flat.view(candidate_points.shape[0], candidate_points.shape[1]) < float(min_child_distance)
        local_distance = local_distance.masked_fill(too_close, -torch.inf)
    selected_ids = torch.topk(local_distance, k=min(children_per_parent, candidate_bary.shape[1]), largest=True, dim=-1).indices
    selected = torch.gather(candidate_bary, 1, selected_ids[:, :, None].expand(-1, -1, 3))
    selected_faces = torch.gather(candidate_faces, 1, selected_ids)
    for child_idx in range(selected.shape[1]):
        all_bary.append(selected[:, child_idx])
        all_faces.append(selected_faces[:, child_idx])
        all_parents.append(parent_indices)
    return torch.cat(all_parents, dim=0), torch.cat(all_faces, dim=0), torch.cat(all_bary, dim=0)


def select_prune_mask(stats: RootStats, config: PruneConfig) -> torch.Tensor:
    stats.validate()
    contribution = stats.gaussian_contrib_sum.reshape(-1)
    visible = stats.visible_count.reshape(-1)
    prune = (visible < float(config.min_visible_count)) | (contribution < float(config.min_contribution))
    if stats.opacity_mean is not None and float(config.min_opacity) > 0.0:
        prune = prune | (stats.opacity_mean.reshape(-1) < float(config.min_opacity))
    if not prune.any():
        return prune
    max_count = int(max(0, round(float(config.max_prune_fraction) * stats.root_count)))
    if max_count <= 0:
        return torch.zeros_like(prune)
    ids = torch.nonzero(prune, as_tuple=False).reshape(-1)
    if ids.numel() <= max_count:
        return prune
    rank_score = contribution[ids] + 1e-6 * visible[ids]
    keep_ids = ids[torch.argsort(rank_score)[:max_count]]
    limited = torch.zeros_like(prune)
    limited[keep_ids] = True
    return limited


def propose_structure_update(
    state: RootLifecycleState,
    stats: RootStats,
    densify: DensifyConfig,
    prune: PruneConfig,
    *,
    vertices: torch.Tensor | None = None,
    faces: torch.Tensor | None = None,
) -> RootStructureUpdate:
    parents, scores = select_densify_parents(stats, densify)
    child_parent_indices, new_face_ids, new_barycentric = propose_split_children(
        state,
        parents,
        densify.children_per_parent,
        densify.barycentric_step,
        vertices=vertices,
        faces=faces,
        neighbor_count=densify.neighbor_count,
        candidate_rings=densify.candidate_rings,
        candidate_face_count=densify.candidate_face_count,
        min_child_distance=densify.min_child_distance,
    )
    prune_mask = select_prune_mask(stats, prune)
    if densify.replace_parent and parents.numel() > 0:
        prune_mask = prune_mask.clone()
        prune_mask[parents] = True
    return RootStructureUpdate(
        parent_indices=parents,
        child_parent_indices=child_parent_indices,
        new_face_ids=new_face_ids,
        new_barycentric=new_barycentric,
        prune_mask=prune_mask,
        scores=scores,
    )


def barycentric_to_points(vertices: torch.Tensor, faces: torch.Tensor, face_ids: torch.Tensor, barycentric: torch.Tensor) -> torch.Tensor:
    tri = vertices[faces[face_ids]]
    return (tri * barycentric[:, :, None]).sum(dim=1)


def apply_structure_update(
    state: RootLifecycleState,
    update: RootStructureUpdate,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> RootLifecycleState:
    """Apply insertions first, then prune on the expanded root set."""

    state.validate()
    points = state.points
    face_ids = state.face_ids
    bary = state.barycentric
    if update.new_barycentric.numel() > 0:
        new_points = barycentric_to_points(vertices, faces, update.new_face_ids, update.new_barycentric)
        points = torch.cat([points, new_points], dim=0)
        face_ids = torch.cat([face_ids, update.new_face_ids], dim=0)
        bary = torch.cat([bary, update.new_barycentric], dim=0)
    if update.prune_mask.numel() > 0 and update.prune_mask.any():
        expanded_prune = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        expanded_prune[: update.prune_mask.shape[0]] = update.prune_mask.to(device=points.device)
        keep = ~expanded_prune
        points = points[keep]
        face_ids = face_ids[keep]
        bary = bary[keep]
    return RootLifecycleState(points=points, face_ids=face_ids, barycentric=bary)


def interpolate_child_attributes(
    attributes: torch.Tensor,
    state: RootLifecycleState,
    update: RootStructureUpdate,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    neighbor_count: int = 8,
    parent_weight: float = 3.0,
) -> torch.Tensor:
    """Initialize child root attributes from parent + local root neighbors.

    Call this before applying the prune mask.  Parents remain available for
    interpolation, then can be removed by ``apply_attribute_update``.
    """

    state.validate()
    if attributes.shape[0] != state.points.shape[0]:
        raise ValueError("attributes must have one row per old root")
    if update.new_barycentric.numel() == 0:
        return attributes.new_empty((0, *attributes.shape[1:]))
    flat = attributes.reshape(attributes.shape[0], -1)
    child_points = barycentric_to_points(vertices, faces, update.new_face_ids, update.new_barycentric)
    dist = torch.cdist(child_points, state.points)
    k = max(1, min(int(neighbor_count), int(state.points.shape[0])))
    knn = torch.topk(dist, k=k, largest=False, dim=-1).indices
    parent_ids = update.child_parent_indices.reshape(-1, 1)
    ids = torch.cat([parent_ids, knn], dim=1)
    gathered_dist = torch.gather(dist, 1, ids).clamp_min(EPS)
    weights = 1.0 / gathered_dist.square()
    weights[:, 0] = weights[:, 0] * float(parent_weight)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(EPS)
    values = flat[ids]
    child = (values * weights[:, :, None]).sum(dim=1)
    return child.reshape((child.shape[0], *attributes.shape[1:]))


def apply_attribute_update(
    attributes: torch.Tensor,
    update: RootStructureUpdate,
    child_attributes: torch.Tensor,
) -> torch.Tensor:
    """Append interpolated child attributes, then remove pruned old roots."""

    if child_attributes.shape[0] != update.new_barycentric.shape[0]:
        raise ValueError("child_attributes must match new root count")
    out = attributes
    if child_attributes.numel() > 0:
        out = torch.cat([out, child_attributes.to(device=out.device, dtype=out.dtype)], dim=0)
    if update.prune_mask.numel() > 0 and update.prune_mask.any():
        expanded_prune = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
        expanded_prune[: update.prune_mask.shape[0]] = update.prune_mask.to(device=out.device)
        out = out[~expanded_prune]
    return out

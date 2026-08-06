from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.roots.lifecycle import (
    DensifyConfig,
    FaceAdjacencyIndex,
    PruneConfig,
    RootLifecycleState,
    RootStats,
    _multi_face_child_candidates,
    propose_split_children,
    propose_structure_update,
)


def _plane_state(grid: int = 7) -> tuple[RootLifecycleState, torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, grid)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    vertices = torch.stack([xx.flatten(), yy.flatten(), 0.07 * torch.sin(xx.flatten())], dim=-1)
    face_rows: list[list[int]] = []
    for row in range(grid - 1):
        for column in range(grid - 1):
            v0 = row * grid + column
            v1 = v0 + 1
            v2 = v0 + grid
            v3 = v2 + 1
            face_rows.extend(([v0, v1, v3], [v0, v3, v2]))
    faces = torch.tensor(face_rows, dtype=torch.long)
    face_ids = torch.arange(faces.shape[0], dtype=torch.long)
    barycentric = torch.tensor([0.23, 0.31, 0.46]).expand(faces.shape[0], -1).clone()
    points = (vertices[faces[face_ids]] * barycentric[:, :, None]).sum(dim=1)
    return RootLifecycleState(points=points, face_ids=face_ids, barycentric=barycentric), vertices, faces


def test_face_adjacency_index_preserves_structure_update() -> None:
    state, vertices, faces = _plane_state()
    root_count = int(state.points.shape[0])
    evidence = torch.linspace(0.0, 1.0, root_count)[:, None]
    stats = RootStats(
        root_grad_abs_sum=torch.zeros_like(evidence),
        gaussian_grad_abs_sum=evidence,
        gaussian_contrib_sum=torch.ones_like(evidence),
        visible_count=torch.full_like(evidence, 12.0),
    )
    densify = DensifyConfig(
        grad_threshold=0.55,
        parent_selection_mode="evidence_local_max",
        max_new_roots=0,
        children_per_parent=2,
        neighbor_count=5,
        candidate_rings=3,
        candidate_face_count=16,
        min_child_distance=0.01,
    )
    prune = PruneConfig(max_prune_fraction=0.0)
    reference = propose_structure_update(
        state,
        stats,
        densify,
        prune,
        vertices=vertices,
        faces=faces,
    )
    accelerated = propose_structure_update(
        state,
        stats,
        densify,
        prune,
        vertices=vertices,
        faces=faces,
        face_adjacency_index=FaceAdjacencyIndex.from_faces(faces),
    )
    assert torch.equal(accelerated.parent_indices, reference.parent_indices)
    assert torch.equal(accelerated.child_parent_indices, reference.child_parent_indices)
    assert torch.equal(accelerated.new_face_ids, reference.new_face_ids)
    assert torch.equal(accelerated.new_barycentric, reference.new_barycentric)
    assert torch.equal(accelerated.prune_mask, reference.prune_mask)


def test_fused_minimum_distance_matches_independent_two_pass_reference() -> None:
    state, vertices, faces = _plane_state()
    parents = torch.tensor([8, 19, 31, 46], dtype=torch.long)
    candidate_faces, candidate_barycentric, candidate_points = _multi_face_child_candidates(
        state,
        parents,
        vertices,
        faces,
        candidate_face_count=18,
        candidate_rings=3,
    )
    flat_candidates = candidate_points.reshape(-1, 3)
    distances = torch.cdist(flat_candidates, state.points)
    nearest = torch.topk(distances, k=6, largest=False, dim=-1).values[:, -1]
    closest = torch.min(distances, dim=-1).values
    scores = nearest.view(candidate_points.shape[:2])
    scores = scores.masked_fill(
        closest.view(candidate_points.shape[:2]) < 0.035,
        -torch.inf,
    )
    selected_ids = torch.topk(scores, k=2, largest=True, dim=-1).indices
    expected_barycentric = torch.gather(
        candidate_barycentric,
        1,
        selected_ids[:, :, None].expand(-1, -1, 3),
    )
    expected_faces = torch.gather(candidate_faces, 1, selected_ids)
    expected_parents = torch.cat([parents, parents], dim=0)
    expected_faces = torch.cat([expected_faces[:, 0], expected_faces[:, 1]], dim=0)
    expected_barycentric = torch.cat(
        [expected_barycentric[:, 0], expected_barycentric[:, 1]],
        dim=0,
    )

    actual_parents, actual_faces, actual_barycentric = propose_split_children(
        state,
        parents,
        2,
        0.08,
        vertices=vertices,
        faces=faces,
        neighbor_count=6,
        candidate_rings=3,
        candidate_face_count=18,
        min_child_distance=0.035,
        face_adjacency_index=FaceAdjacencyIndex.from_faces(faces),
    )
    assert torch.equal(actual_parents, expected_parents)
    assert torch.equal(actual_faces, expected_faces)
    assert torch.equal(actual_barycentric, expected_barycentric)

"""Unit check for topology-local root densification selection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.roots import DensifyConfig, PruneConfig, RootLifecycleState, RootStats, propose_structure_update  # noqa: E402


def make_state() -> tuple[RootLifecycleState, torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
        ],
        dtype=torch.long,
    )
    face_ids = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    barycentric = torch.tensor(
        [
            [0.50, 0.25, 0.25],
            [0.25, 0.50, 0.25],
            [0.25, 0.25, 0.50],
            [0.34, 0.33, 0.33],
        ],
        dtype=torch.float32,
    )
    points = (vertices[faces[face_ids]] * barycentric[:, :, None]).sum(dim=1)
    return RootLifecycleState(points=points, face_ids=face_ids, barycentric=barycentric), vertices, faces


def make_stats() -> RootStats:
    need = torch.tensor([[0.80], [1.00], [0.70], [0.90]], dtype=torch.float32)
    visible = torch.ones_like(need) * 10.0
    contribution = torch.ones_like(need)
    return RootStats(
        root_grad_abs_sum=torch.zeros_like(need),
        gaussian_grad_abs_sum=need,
        gaussian_contrib_sum=contribution,
        visible_count=visible,
        gaussian_sample_count=torch.ones_like(need),
        residual_sum=torch.zeros_like(need),
        opacity_mean=torch.ones_like(need),
    )


def main() -> None:
    state, vertices, faces = make_state()
    update = propose_structure_update(
        state,
        make_stats(),
        DensifyConfig(
            parent_selection_mode="evidence_local_max",
            score_mode="raw",
            grad_threshold=0.5,
            visibility_threshold=1.0,
            max_new_roots=4,
            children_per_parent=2,
            neighbor_count=2,
            candidate_rings=1,
            candidate_face_count=4,
            min_child_distance=0.0,
        ),
        PruneConfig(max_prune_fraction=0.0),
        vertices=vertices,
        faces=faces,
    )
    parents = update.parent_indices.detach().cpu().tolist()
    assert parents == [1, 3], f"expected local maxima roots [1, 3], got {parents}"
    assert int(update.new_face_ids.numel()) == 4
    assert bool(update.prune_mask[1].item())
    assert bool(update.prune_mask[3].item())
    assert not bool(update.prune_mask[0].item())
    assert not bool(update.prune_mask[2].item())
    print("root lifecycle local-max selection passed")


if __name__ == "__main__":
    main()

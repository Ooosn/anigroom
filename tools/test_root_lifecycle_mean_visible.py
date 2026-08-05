"""Unit check for R005 clean geometric lifecycle evidence."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.roots import DensifyConfig, PruneConfig, RootLifecycleState, RootStats, propose_structure_update  # noqa: E402


def main() -> None:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.5, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    face_ids = torch.tensor([0, 1], dtype=torch.long)
    barycentric = torch.tensor([[0.4, 0.3, 0.3], [0.3, 0.4, 0.3]], dtype=torch.float32)
    points = (vertices[faces[face_ids]] * barycentric[:, :, None]).sum(dim=1)
    state = RootLifecycleState(points=points, face_ids=face_ids, barycentric=barycentric)

    visible = torch.tensor([[10.0], [10.0]], dtype=torch.float32)
    stats = RootStats(
        root_grad_abs_sum=torch.tensor([[0.0], [1000.0]], dtype=torch.float32),
        gaussian_grad_abs_sum=torch.tensor([[10.0], [1000.0]], dtype=torch.float32),
        gaussian_contrib_sum=torch.tensor([[100.0], [1.0]], dtype=torch.float32),
        visible_count=visible,
        gaussian_sample_count=torch.ones_like(visible),
        residual_sum=torch.tensor([[0.0], [1000.0]], dtype=torch.float32),
        opacity_mean=torch.ones_like(visible),
        gaussian_mean_grad_abs_sum=torch.tensor([[10.0], [1.0]], dtype=torch.float32),
    )

    update = propose_structure_update(
        state,
        stats,
        DensifyConfig(
            parent_selection_mode="evidence_local_max",
            score_mode="mean_visible",
            grad_threshold=0.05,
            visibility_threshold=1.0,
            max_new_roots=2,
            children_per_parent=2,
            candidate_rings=0,
            candidate_face_count=1,
            min_child_distance=0.0,
        ),
        PruneConfig(max_prune_fraction=0.0),
        vertices=vertices,
        faces=faces,
    )
    parents = update.parent_indices.detach().cpu().tolist()
    assert parents == [0], f"mean_visible evidence should select root 0, got {parents}"
    scores = update.scores
    assert torch.isclose(scores["need"][0], torch.tensor(1.0))
    assert torch.isclose(scores["need"][1], torch.tensor(0.1))
    uncapped_update = propose_structure_update(
        state,
        stats,
        DensifyConfig(
            parent_selection_mode="evidence_local_max",
            score_mode="mean_visible",
            grad_threshold=0.05,
            visibility_threshold=1.0,
            max_new_roots=0,
            children_per_parent=2,
            candidate_rings=0,
            candidate_face_count=1,
            min_child_distance=0.0,
        ),
        PruneConfig(max_prune_fraction=0.0),
        vertices=vertices,
        faces=faces,
    )
    uncapped_parents = uncapped_update.parent_indices.detach().cpu().tolist()
    assert uncapped_parents == [0, 1], f"uncapped parent budget should keep all candidates, got {uncapped_parents}"
    assert float(uncapped_update.scores["parent_budget"].item()) == -1.0
    assert float(uncapped_update.scores["budget_saturated"].item()) == 0.0
    print("root lifecycle mean-visible evidence passed")


if __name__ == "__main__":
    main()

"""Check root split-replace and child attribute interpolation.

This is a CPU structural test for the densification module. It verifies that
children are initialized while parents are still present, then parents are
removed from both root state and per-root attributes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.roots import (
    DensifyConfig,
    PruneConfig,
    RootLifecycleState,
    RootStats,
    apply_attribute_update,
    apply_structure_update,
    interpolate_child_attributes,
    propose_structure_update,
)


def make_plane_state(grid: int) -> tuple[torch.Tensor, torch.Tensor, RootLifecycleState]:
    xs = torch.linspace(-1.0, 1.0, grid)
    ys = torch.linspace(-1.0, 1.0, grid)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    vertices = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(grid * grid)], dim=1)
    faces = []
    for y in range(grid - 1):
        for x in range(grid - 1):
            v0 = y * grid + x
            v1 = v0 + 1
            v2 = v0 + grid
            v3 = v2 + 1
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    faces_t = torch.tensor(faces, dtype=torch.long)
    points = []
    face_ids = []
    bary = []
    for face_id, face in enumerate(faces[: min(len(faces), grid * grid)]):
        tri = vertices[torch.tensor(face)]
        bc = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        points.append((tri * bc[:, None]).sum(dim=0))
        face_ids.append(face_id)
        bary.append(bc)
    state = RootLifecycleState(
        points=torch.stack(points, dim=0),
        face_ids=torch.tensor(face_ids, dtype=torch.long),
        barycentric=torch.stack(bary, dim=0),
    )
    return vertices, faces_t, state


def make_stats(state: RootLifecycleState, parent_ids: torch.Tensor) -> RootStats:
    root_count = state.points.shape[0]
    root_grad = torch.zeros(root_count, 1)
    gaussian_grad = torch.zeros(root_count, 1)
    contribution = torch.ones(root_count, 1)
    visible = torch.ones(root_count, 1) * 8.0
    root_grad[parent_ids] = 0.08
    gaussian_grad[parent_ids] = 0.10
    residual = torch.zeros(root_count, 1)
    return RootStats(
        root_grad_abs_sum=root_grad,
        gaussian_grad_abs_sum=gaussian_grad,
        gaussian_contrib_sum=contribution,
        visible_count=visible,
        residual_sum=residual,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="D:/petsgaussianhair/_downloads/root_split_interpolation_20260623.json")
    args = parser.parse_args()

    vertices, faces, state = make_plane_state(grid=8)
    parent_ids = torch.tensor([5, 14, 23], dtype=torch.long)
    stats = make_stats(state, parent_ids)
    update = propose_structure_update(
        state,
        stats,
        DensifyConfig(
            grad_threshold=0.05,
            max_new_roots=6,
            children_per_parent=2,
            barycentric_step=0.12,
            replace_parent=True,
            neighbor_count=8,
            candidate_rings=3,
        ),
        PruneConfig(max_prune_fraction=0.0),
        vertices=vertices,
        faces=faces,
    )
    attributes = torch.stack(
        [
            torch.arange(state.points.shape[0], dtype=torch.float32),
            state.points[:, 0],
            state.points[:, 1],
            torch.sin(torch.arange(state.points.shape[0], dtype=torch.float32)),
        ],
        dim=1,
    )
    child_attributes = interpolate_child_attributes(attributes, state, update, vertices, faces, neighbor_count=8)
    after_state = apply_structure_update(state, update, vertices, faces)
    after_attributes = apply_attribute_update(attributes, update, child_attributes)

    expected_after = int(state.points.shape[0] + update.new_barycentric.shape[0] - update.prune_mask.sum().item())
    parent_attr_values = set(float(v) for v in attributes[parent_ids, 0].tolist())
    after_exact_parent_values = [v for v in after_attributes[:, 0].tolist() if float(v) in parent_attr_values]
    summary = {
        "old_root_count": int(state.points.shape[0]),
        "selected_parent_count": int(update.parent_indices.numel()),
        "new_root_count": int(update.new_barycentric.shape[0]),
        "pruned_old_root_count": int(update.prune_mask.sum().item()),
        "after_root_count": int(after_state.points.shape[0]),
        "after_attribute_count": int(after_attributes.shape[0]),
        "expected_after_count": expected_after,
        "child_attribute_count": int(child_attributes.shape[0]),
        "child_attribute_min": float(child_attributes.min().item()),
        "child_attribute_max": float(child_attributes.max().item()),
        "exact_parent_attribute_values_left": after_exact_parent_values,
    }
    assert update.parent_indices.numel() == parent_ids.numel(), summary
    assert update.new_barycentric.shape[0] == parent_ids.numel() * 2, summary
    assert update.prune_mask[parent_ids].all(), summary
    assert after_state.points.shape[0] == expected_after, summary
    assert after_attributes.shape[0] == expected_after, summary
    assert not after_exact_parent_values, summary
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

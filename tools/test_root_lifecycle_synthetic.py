"""Synthetic checks for root move/densify/prune logic.

The scene is intentionally simple: a rectangular mesh surface with FPS roots.
One spatial region receives high synthetic gradient/contribution demand and
another region is invisible/low-contribution.  The test verifies that
threshold-based densification and pruning select the expected regions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.mesh_roots import (
    TriangleMesh,
    initialize_surface_roots_fps,
)
from anigroom.roots import (
    DensifyConfig,
    PruneConfig,
    RootLifecycleState,
    RootStats,
    apply_structure_update,
    propose_structure_update,
)


def make_plane_mesh(grid: int = 48) -> TriangleMesh:
    xs = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    vertices = []
    for y in ys:
        for x in xs:
            vertices.append([x, y, 0.0])
    faces = []
    for iy in range(grid - 1):
        for ix in range(grid - 1):
            v00 = iy * grid + ix
            v10 = v00 + 1
            v01 = v00 + grid
            v11 = v01 + 1
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    return TriangleMesh(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64))


def make_synthetic_stats(points: torch.Tensor) -> RootStats:
    x = points[:, 0:1]
    y = points[:, 1:2]
    high_need = torch.exp(-((x + 0.45).square() / 0.06 + (y - 0.10).square() / 0.20))
    low_region = (x > 0.45) & (y < -0.25)
    visible = torch.full_like(high_need, 8.0)
    visible = torch.where(low_region, torch.zeros_like(visible), visible)
    contribution = torch.full_like(high_need, 80.0)
    contribution = torch.where(low_region, torch.zeros_like(contribution), contribution)
    gaussian_grad = contribution * (0.02 + 0.95 * high_need)
    root_grad = visible * (0.01 + 0.55 * high_need)
    residual = visible * (0.02 + 0.60 * high_need)
    opacity = torch.where(low_region, torch.full_like(high_need, 0.03), torch.full_like(high_need, 0.70))
    return RootStats(
        root_grad_abs_sum=root_grad,
        gaussian_grad_abs_sum=gaussian_grad,
        gaussian_contrib_sum=contribution,
        visible_count=visible,
        residual_sum=residual,
        opacity_mean=opacity,
    )


def save_visualization(
    output: Path,
    before: RootLifecycleState,
    after: RootLifecycleState,
    update,
    stats: RootStats,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    size = 900
    margin = 70
    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image)

    def to_xy(points: torch.Tensor) -> np.ndarray:
        arr = points.detach().cpu().numpy()
        px = margin + (arr[:, 0] + 1.0) * 0.5 * (size - 2 * margin)
        py = margin + (1.0 - (arr[:, 1] + 1.0) * 0.5) * (size - 2 * margin)
        return np.stack([px, py], axis=1)

    draw.rectangle((margin, margin, size - margin, size - margin), outline=(180, 180, 180), width=2)
    draw.ellipse((margin + 50, margin + 170, margin + 310, margin + 500), outline=(255, 180, 0), width=4)
    draw.text((margin + 55, margin + 145), "high synthetic demand", fill=(180, 100, 0))
    draw.rectangle((size - margin - 250, size - margin - 250, size - margin - 40, size - margin - 40), outline=(180, 0, 0), width=4)
    draw.text((size - margin - 250, size - margin - 280), "low visibility / prune", fill=(180, 0, 0))

    before_xy = to_xy(before.points)
    for x, y in before_xy:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(90, 120, 190))

    parent_xy = to_xy(before.points[update.parent_indices]) if update.parent_indices.numel() else np.zeros((0, 2))
    for x, y in parent_xy:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 140, 0), width=2)

    new_count = int(update.new_barycentric.shape[0])
    if new_count:
        new_points = after.points[-new_count:]
        new_xy = to_xy(new_points)
        for x, y in new_xy:
            draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=(0, 170, 80))

    prune_ids = torch.nonzero(update.prune_mask, as_tuple=False).reshape(-1)
    prune_xy = to_xy(before.points[prune_ids]) if prune_ids.numel() else np.zeros((0, 2))
    for x, y in prune_xy:
        draw.line((x - 5, y - 5, x + 5, y + 5), fill=(200, 0, 0), width=2)
        draw.line((x - 5, y + 5, x + 5, y - 5), fill=(200, 0, 0), width=2)

    draw.text((20, 20), "blue: original roots | orange: densify parents | green: inserted | red x: pruned", fill=(0, 0, 0))
    image.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="D:/petsgaussianhair/_downloads/root_lifecycle_synthetic_20260623")
    parser.add_argument("--root-count", type=int, default=512)
    parser.add_argument("--candidate-multiplier", type=float, default=12.0)
    parser.add_argument("--densify-threshold", type=float, default=0.42)
    parser.add_argument("--max-new-roots", type=int, default=80)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = make_plane_mesh()
    roots = initialize_surface_roots_fps(
        mesh,
        args.root_count,
        candidate_multiplier=args.candidate_multiplier,
        seed=args.seed,
        fps_device="cpu",
    )
    points = torch.from_numpy(roots.points)
    state = RootLifecycleState(
        points=points,
        face_ids=torch.from_numpy(roots.face_ids),
        barycentric=torch.from_numpy(roots.barycentric),
    )
    vertices = torch.from_numpy(mesh.vertices)
    faces = torch.from_numpy(mesh.faces)
    stats = make_synthetic_stats(points)
    densify_cfg = DensifyConfig(
        grad_threshold=args.densify_threshold,
        visibility_threshold=1.0,
        max_new_roots=args.max_new_roots,
        children_per_parent=args.children_per_parent,
        barycentric_step=0.10,
    )
    prune_cfg = PruneConfig(
        min_visible_count=1.0,
        min_contribution=1e-5,
        min_opacity=0.08,
        max_prune_fraction=0.15,
    )
    update = propose_structure_update(state, stats, densify_cfg, prune_cfg, vertices=vertices, faces=faces)
    after = apply_structure_update(state, update, vertices, faces)
    save_visualization(output_dir / "root_lifecycle_synthetic.jpg", state, after, update, stats)

    parent_points = state.points[update.parent_indices] if update.parent_indices.numel() else state.points[:0]
    prune_ids = torch.nonzero(update.prune_mask, as_tuple=False).reshape(-1)
    prune_points = state.points[prune_ids] if prune_ids.numel() else state.points[:0]
    summary = {
        "root_count_before": int(state.points.shape[0]),
        "densify_parent_count": int(update.parent_indices.numel()),
        "new_root_count": int(update.new_barycentric.shape[0]),
        "prune_count": int(update.prune_mask.sum().item()),
        "root_count_after": int(after.points.shape[0]),
        "parent_x_mean": float(parent_points[:, 0].mean().item()) if parent_points.numel() else None,
        "parent_y_mean": float(parent_points[:, 1].mean().item()) if parent_points.numel() else None,
        "prune_x_mean": float(prune_points[:, 0].mean().item()) if prune_points.numel() else None,
        "prune_y_mean": float(prune_points[:, 1].mean().item()) if prune_points.numel() else None,
        "output_image": str(output_dir / "root_lifecycle_synthetic.jpg"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

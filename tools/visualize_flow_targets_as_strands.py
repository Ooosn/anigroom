from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.mesh_roots import TriangleMesh, read_obj_mesh  # noqa: E402
from anigroom.projection import render_mesh_depth  # noqa: E402
from anigroom.projection.mesh_visibility import project_points, sample_depth_nearest  # noqa: E402


EPS = 1.0e-8


def load_camera_tensors(data_root: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    intr = np.load(data_root / "cameras_intr.npy").astype(np.float32)
    extr = np.load(data_root / "cameras_extr.npy").astype(np.float32)
    return torch.from_numpy(extr).to(device=device), torch.from_numpy(intr[:, :3, :3]).to(device=device)


def project_directions(points: torch.Tensor, directions: torch.Tensor, viewmat: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    rot = viewmat[:3, :3]
    trans = viewmat[:3, 3]
    cam = points @ rot.T + trans.view(1, 3)
    dirs_cam = directions @ rot.T
    z = cam[:, 2].clamp_min(1.0e-6)
    du = k[0, 0] * (dirs_cam[:, 0] * z - cam[:, 0] * dirs_cam[:, 2]) / z.square()
    dv = k[1, 1] * (dirs_cam[:, 1] * z - cam[:, 1] * dirs_cam[:, 2]) / z.square()
    return torch.stack([du, dv], dim=-1)


def sample_shell_visibility(
    points: torch.Tensor,
    normals: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    mesh_depth: torch.Tensor,
    *,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    local_depth_kernel: int,
    front_normal_z: float | None,
) -> dict[str, torch.Tensor]:
    height, width = int(mesh_depth.shape[0]), int(mesh_depth.shape[1])
    xy, depth, _ = project_points(points, viewmat, k)
    in_frame = (
        (depth > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= height - 1)
    )
    sampled_depth = sample_depth_nearest(mesh_depth, xy, kernel_size=int(local_depth_kernel))
    tolerance = float(depth_abs_tolerance) + depth.abs() * float(depth_rel_tolerance)
    depth_visible = in_frame & torch.isfinite(sampled_depth) & (depth <= sampled_depth + tolerance)
    normal_cam = normals @ viewmat[:3, :3].T
    if front_normal_z is None:
        front_facing = torch.ones_like(depth_visible, dtype=torch.bool)
    else:
        front_facing = normal_cam[:, 2] <= float(front_normal_z)
    visible = depth_visible & front_facing
    return {
        "xy": xy,
        "depth": depth,
        "mesh_depth": sampled_depth,
        "visible": visible,
        "in_frame": in_frame,
        "depth_visible": depth_visible,
        "front_facing": front_facing,
    }


def draw_strands(
    base: Image.Image,
    xy0: torch.Tensor,
    screen_dirs: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    *,
    screen_length_px: float,
    max_roots: int,
    width: int,
    color: tuple[int, int, int],
) -> Image.Image:
    canvas = base.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    xy1 = xy0 + float(screen_length_px) * screen_dirs
    ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if ids.numel() > int(max_roots):
        _, order = torch.topk(weights[ids], k=int(max_roots), largest=True)
        ids = ids[order]
    xy0_np = xy0[ids].detach().cpu().numpy()
    xy1_np = xy1[ids].detach().cpu().numpy()
    w_np = weights[ids].detach().cpu().numpy()
    for (x0, y0), (x1, y1), w in zip(xy0_np, xy1_np, w_np):
        alpha = int(40 + 190 * float(np.clip(w, 0.0, 1.0)))
        draw.line((float(x0), float(y0), float(x1), float(y1)), fill=(*color, alpha), width=int(width))
    return canvas


def draw_direction_arrows(
    base: Image.Image,
    xy0: torch.Tensor,
    screen_dirs: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    *,
    screen_length_px: float,
    max_roots: int,
    width: int,
    color: tuple[int, int, int],
) -> Image.Image:
    canvas = base.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    xy1 = xy0 + float(screen_length_px) * screen_dirs
    ids = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if ids.numel() > int(max_roots):
        _, order = torch.topk(weights[ids], k=int(max_roots), largest=True)
        ids = ids[order]
    xy0_np = xy0[ids].detach().cpu().numpy()
    xy1_np = xy1[ids].detach().cpu().numpy()
    dir_np = screen_dirs[ids].detach().cpu().numpy()
    w_np = weights[ids].detach().cpu().numpy()
    for (x0, y0), (x1, y1), (dx, dy), w in zip(xy0_np, xy1_np, dir_np, w_np):
        alpha = int(55 + 200 * float(np.clip(w, 0.0, 1.0)))
        line_width = max(1, int(width))
        draw.line((float(x0), float(y0), float(x1), float(y1)), fill=(*color, alpha), width=line_width)
        side = np.asarray([-dy, dx], dtype=np.float32)
        tip = np.asarray([x1, y1], dtype=np.float32)
        direction = np.asarray([dx, dy], dtype=np.float32)
        head_len = max(4.0, float(screen_length_px) * 0.22)
        head_width = max(2.0, float(screen_length_px) * 0.10)
        p1 = tip - head_len * direction + head_width * side
        p2 = tip - head_len * direction - head_width * side
        draw.polygon(
            [(float(tip[0]), float(tip[1])), (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))],
            fill=(*color, alpha),
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical full-view shell-root strand visualization for fused 3D flow targets.")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk"))
    parser.add_argument("--mesh-path", type=Path, default=Path("D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--direction-output-dir", type=Path)
    parser.add_argument("--view", type=int, default=9)
    parser.add_argument("--scale", type=float, default=1.28)
    parser.add_argument("--translation", default="0,0.32,0.02")
    parser.add_argument("--depth-abs-tolerance", type=float, default=0.03)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.01)
    parser.add_argument("--local-depth-kernel", type=int, default=7)
    parser.add_argument("--front-normal-z", type=float, default=0.15)
    parser.add_argument("--screen-length-px", type=float, default=34.0)
    parser.add_argument("--arrow-length-px", type=float, default=24.0)
    parser.add_argument("--anchor-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--root-mode", choices=["shell"], default="shell")
    parser.add_argument("--max-roots", type=int, default=0, help="Maximum roots to draw. 0 draws every valid root.")
    parser.add_argument("--line-width", type=int, default=1)
    parser.add_argument(
        "--diagnostic-score-npz",
        type=Path,
        help="Optional per-root diagnostic NPZ aligned with the clean-flow target.",
    )
    parser.add_argument("--diagnostic-score-key", default="local_axis_crossing")
    parser.add_argument("--diagnostic-score-quantile", type=float, default=0.95)
    args = parser.parse_args()
    if int(args.max_roots) != 0:
        raise ValueError("canonical strand visualization must draw all roots; use --max-roots 0")

    target = np.load(args.target)
    roots = torch.from_numpy(target["root_points"].astype(np.float32)).cuda()
    normals = torch.from_numpy(target["root_normals"].astype(np.float32)).cuda()
    if "shell_h" not in target:
        raise KeyError("target does not contain shell_h; canonical visualization uses shell-root strands")
    shell_h = torch.from_numpy(target["shell_h"].astype(np.float32)).cuda()
    roots = roots + shell_h[:, None] * normals
    dirs = torch.from_numpy(target["cleaned_directed_flow3d"].astype(np.float32)).cuda()
    dirs = torch.nn.functional.normalize(dirs, dim=-1, eps=1.0e-8)
    if "weight" in target:
        weights_np = target["weight"].astype(np.float32)
    else:
        weights_np = np.ones((roots.shape[0],), dtype=np.float32)
    if weights_np.max() > 0:
        weights_np = weights_np / np.percentile(weights_np[weights_np > 0], 95).clip(min=1.0e-6)
    weights = torch.from_numpy(np.clip(weights_np, 0.0, 1.0).astype(np.float32)).cuda()
    observed_np = target["observed"].astype(bool) if "observed" in target else np.ones((roots.shape[0],), dtype=bool)
    observed = torch.from_numpy(observed_np).cuda()
    if "direction_anchor" in target:
        direction_anchor = torch.from_numpy(target["direction_anchor"].astype(bool)).cuda()
    else:
        direction_anchor = torch.ones((roots.shape[0],), dtype=torch.bool, device="cuda")
    if "direction_anchor_confidence" in target:
        anchor_confidence = torch.from_numpy(target["direction_anchor_confidence"].astype(np.float32)).cuda()
    else:
        anchor_confidence = torch.ones((roots.shape[0],), dtype=torch.float32, device="cuda")

    viewmats, ks = load_camera_tensors(args.data_root, torch.device("cuda"))
    viewmat = viewmats[int(args.view)]
    k = ks[int(args.view)]
    gt = Image.open(args.data_root / "images" / f"img_{int(args.view):04d}.png").convert("RGB")
    white = Image.new("RGB", gt.size, (255, 255, 255))
    width, height = gt.size

    translation = np.asarray([float(v) for v in str(args.translation).split(",")], dtype=np.float32)
    raw_mesh = read_obj_mesh(args.mesh_path)
    mesh = TriangleMesh(
        vertices=(raw_mesh.vertices.astype(np.float32) * float(args.scale) + translation[None]).astype(np.float32),
        faces=raw_mesh.faces,
    )
    mesh_depth = render_mesh_depth(mesh, viewmat, k, width, height, device=torch.device("cuda"))
    visibility = sample_shell_visibility(
        roots,
        normals,
        viewmat,
        k,
        mesh_depth.depth,
        depth_abs_tolerance=float(args.depth_abs_tolerance),
        depth_rel_tolerance=float(args.depth_rel_tolerance),
        local_depth_kernel=int(args.local_depth_kernel),
        front_normal_z=float(args.front_normal_z),
    )
    screen_dirs = project_directions(roots, dirs, viewmat, k)
    screen_norm = torch.linalg.norm(screen_dirs, dim=-1)
    screen_dirs = F.normalize(screen_dirs, dim=-1, eps=1.0e-8)
    valid = (
        observed
        & visibility["visible"]
        & torch.isfinite(visibility["xy"]).all(dim=-1)
        & torch.isfinite(screen_dirs).all(dim=-1)
        & (screen_norm > 1.0e-6)
    )
    anchor_valid = valid & direction_anchor & (anchor_confidence >= float(args.anchor_confidence_threshold))
    visible_count = int(valid.sum().detach().cpu())
    anchor_visible_count = int(anchor_valid.sum().detach().cpu())
    observed_count = int(observed.sum().detach().cpu())
    depth_visible_count = int(visibility["depth_visible"].sum().detach().cpu())
    front_facing_count = int(visibility["front_facing"].sum().detach().cpu())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    direction_output_dir = args.direction_output_dir if args.direction_output_dir is not None else args.output_dir
    direction_output_dir.mkdir(parents=True, exist_ok=True)
    max_roots = int(args.max_roots) if int(args.max_roots) > 0 else int(roots.shape[0])
    arrows_on_gt = draw_direction_arrows(
        gt,
        visibility["xy"],
        screen_dirs,
        weights,
        valid,
        screen_length_px=float(args.arrow_length_px),
        max_roots=max_roots,
        width=int(args.line_width),
        color=(245, 40, 120),
    )
    anchor_arrows_on_gt = draw_direction_arrows(
        gt,
        visibility["xy"],
        screen_dirs,
        weights,
        anchor_valid,
        screen_length_px=float(args.arrow_length_px),
        max_roots=max_roots,
        width=int(args.line_width),
        color=(245, 40, 120),
    )
    on_gt = draw_strands(
        gt,
        visibility["xy"],
        screen_dirs,
        weights,
        valid,
        screen_length_px=float(args.screen_length_px),
        max_roots=max_roots,
        width=int(args.line_width),
        color=(235, 35, 115),
    )
    on_white = draw_strands(
        white,
        visibility["xy"],
        screen_dirs,
        weights,
        valid,
        screen_length_px=float(args.screen_length_px),
        max_roots=max_roots,
        width=max(1, int(args.line_width)),
        color=(25, 25, 25),
    )
    arrows_on_gt.save(direction_output_dir / f"view{int(args.view):02d}_shell_cleaned_3d_arrows_overlay.png")
    anchor_arrows_on_gt.save(direction_output_dir / f"view{int(args.view):02d}_shell_anchor_3d_arrows_overlay.png")
    diagnostic_report = None
    if args.diagnostic_score_npz is not None:
        diagnostic = np.load(args.diagnostic_score_npz)
        if args.diagnostic_score_key not in diagnostic:
            raise KeyError(
                f"diagnostic score '{args.diagnostic_score_key}' is missing from {args.diagnostic_score_npz}"
            )
        score_np = np.asarray(diagnostic[args.diagnostic_score_key], dtype=np.float32).reshape(-1)
        if int(score_np.shape[0]) != int(roots.shape[0]):
            raise ValueError(
                f"diagnostic score count {score_np.shape[0]} does not match root count {int(roots.shape[0])}"
            )
        finite_score = np.isfinite(score_np)
        threshold = float(np.quantile(score_np[finite_score], float(args.diagnostic_score_quantile)))
        score_lo = float(np.quantile(score_np[finite_score], 0.50))
        score_hi = float(np.quantile(score_np[finite_score], 0.99))
        score_normalized_np = np.clip((score_np - score_lo) / max(score_hi - score_lo, EPS), 0.0, 1.0)
        score = torch.from_numpy(score_normalized_np.astype(np.float32)).to(device=roots.device)
        high_score = torch.from_numpy((finite_score & (score_np >= threshold)).astype(np.bool_)).to(device=roots.device)
        diagnostic_overlay = draw_direction_arrows(
            gt,
            visibility["xy"],
            screen_dirs,
            torch.full_like(weights, 0.35),
            valid,
            screen_length_px=float(args.arrow_length_px),
            max_roots=max_roots,
            width=max(1, int(args.line_width)),
            color=(70, 150, 205),
        )
        diagnostic_overlay = draw_direction_arrows(
            diagnostic_overlay,
            visibility["xy"],
            screen_dirs,
            score,
            valid & high_score,
            screen_length_px=float(args.arrow_length_px),
            max_roots=max_roots,
            width=max(2, int(args.line_width) + 1),
            color=(235, 40, 40),
        )
        diagnostic_overlay.save(
            direction_output_dir
            / f"view{int(args.view):02d}_{args.diagnostic_score_key}_top{int(round((1.0 - float(args.diagnostic_score_quantile)) * 100.0)):02d}_overlay.png"
        )
        diagnostic_report = {
            "path": str(args.diagnostic_score_npz),
            "key": str(args.diagnostic_score_key),
            "quantile": float(args.diagnostic_score_quantile),
            "threshold": threshold,
            "high_score_root_count": int(high_score.sum().detach().cpu()),
        }
    on_gt.save(args.output_dir / f"view{int(args.view):02d}_target_strands_on_gt.png")
    on_white.save(args.output_dir / f"view{int(args.view):02d}_target_strands_white.png")
    report = {
        "view": int(args.view),
        "target": str(args.target),
        "visible_draw_count": visible_count,
        "anchor_visible_draw_count": anchor_visible_count,
        "observed_count": observed_count,
        "depth_visible_count": depth_visible_count,
        "front_facing_count": front_facing_count,
        "arrow_draw_count": visible_count,
        "anchor_arrow_draw_count": anchor_visible_count,
        "strand_draw_count": visible_count,
        "arrow_length_px": float(args.arrow_length_px),
        "strand_length_px": float(args.screen_length_px),
        "anchor_confidence_threshold": float(args.anchor_confidence_threshold),
        "max_roots": int(args.max_roots),
    }
    if diagnostic_report is not None:
        report["diagnostic_score"] = diagnostic_report
    (args.output_dir / f"view{int(args.view):02d}_visualization_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()

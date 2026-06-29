from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_white_tiger_stage1 as stage1  # noqa: E402


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    arr = image.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_root_point_overlay(
    path: Path,
    base_image: torch.Tensor,
    roots: torch.Tensor,
    root_mask: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    *,
    color: tuple[int, int, int, int],
    radius: int = 2,
    max_points: int = 20000,
) -> dict[str, int]:
    selected = torch.nonzero(root_mask.detach().reshape(-1), as_tuple=False).reshape(-1)
    selected_count = int(selected.numel())
    canvas = tensor_image_to_pil(base_image).convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    if selected_count == 0:
        canvas.save(path)
        return {"selected_root_count": 0, "drawn_root_count": 0}
    if selected_count > max_points:
        order = torch.linspace(0, selected_count - 1, steps=max_points, device=selected.device).long()
        selected = selected[order]
    xy, depth = stage1.project_points(roots[selected], viewmat, k)
    valid = (
        (depth > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= height - 1)
    )
    pts = xy[valid].detach().cpu().numpy()
    for x, y in pts:
        draw.ellipse((float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius), fill=color)
    canvas.save(path)
    return {"selected_root_count": selected_count, "drawn_root_count": int(valid.sum().detach().cpu())}


@torch.no_grad()
def save_gaussian_root_overlay(
    path: Path,
    base_image: torch.Tensor,
    gaussians,
    root_mask: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    *,
    color: tuple[float, float, float],
    opacity_scale: float = 0.80,
) -> dict[str, int]:
    root_ids = gaussians.root_indices.long().reshape(-1)
    gaussian_mask = root_mask[root_ids]
    selected_count = int(gaussian_mask.sum().detach().cpu())
    selected_root_count = int(root_mask.sum().detach().cpu())
    if selected_count == 0:
        stage1.save_image(path, base_image)
        return {"selected_root_count": selected_root_count, "selected_gaussian_count": 0}

    color_tensor = gaussians.means.new_tensor(color).view(1, 3)
    selected = replace(
        gaussians,
        means=gaussians.means[gaussian_mask],
        directions=gaussians.directions[gaussian_mask],
        quats=gaussians.quats[gaussian_mask],
        scales=gaussians.scales[gaussian_mask],
        colors=torch.ones((selected_count, 3), device=gaussians.means.device, dtype=gaussians.means.dtype) * color_tensor,
        opacities=(gaussians.opacities[gaussian_mask].reshape(-1) * float(opacity_scale)).clamp(0.0, 0.85),
        root_indices=gaussians.root_indices[gaussian_mask],
        segment_indices=gaussians.segment_indices[gaussian_mask],
    )
    overlay_rgb, overlay_alpha, _ = stage1.rasterization(
        selected.means,
        selected.quats,
        selected.scales,
        selected.opacities.reshape(-1),
        selected.colors,
        viewmat.view(1, 4, 4),
        k.view(1, 3, 3),
        width,
        height,
        packed=False,
        backgrounds=torch.zeros((1, 3), device=gaussians.means.device, dtype=gaussians.means.dtype),
        rasterize_mode="antialiased",
    )
    overlay = overlay_rgb[0]
    alpha = overlay_alpha[0].clamp(0.0, 0.85)
    out = (base_image * (1.0 - alpha) + overlay * alpha).clamp(0.0, 1.0)
    stage1.save_image(path, out)
    return {"selected_root_count": selected_root_count, "selected_gaussian_count": selected_count}


def summarize_tensor(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    q = torch.quantile(flat, torch.tensor([0.10, 0.50, 0.75, 0.90, 0.95, 0.98], device=flat.device))
    return {
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "min": float(flat.min().cpu()),
        "q10": float(q[0].cpu()),
        "q50": float(q[1].cpu()),
        "q75": float(q[2].cpu()),
        "q90": float(q[3].cpu()),
        "q95": float(q[4].cpu()),
        "q98": float(q[5].cpu()),
        "max": float(flat.max().cpu()),
    }


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing config next to checkpoint: {config_path}")
    config = stage1.Stage1Config(**json.loads(config_path.read_text(encoding="utf-8")))
    state = checkpoint["model"]
    mesh_path = stage1.resolve_project_path(config.mesh_path)
    mesh = stage1.read_obj_mesh(mesh_path)
    normals = stage1.face_normals_np(mesh)
    face_ids = state["face_ids"].detach().cpu().numpy()
    bary = state["bary_initial"].detach().cpu().numpy()
    guide_face_ids = None
    guide_bary = None
    if "guide_face_ids" in state and int(state["guide_face_ids"].numel()) > 0:
        guide_face_ids = state["guide_face_ids"].detach().cpu().numpy()
        guide_bary = state["guide_barycentric"].detach().cpu().numpy()
    model = stage1.WhiteTigerStage1Model(
        mesh,
        normals,
        face_ids,
        bary,
        stage1.dense_groom_ranges(),
        device,
        init_scale=config.init_mesh_scale,
        init_translation=tuple(config.init_mesh_translation),
        init_groom_length=getattr(config, "init_groom_length", 0.060),
        init_guide_length=getattr(config, "init_guide_length", 0.060),
        max_child_count=config.child_count,
        local_child_color_support=config.local_child_color_support,
        local_child_opacity_support=config.local_child_opacity_support,
        local_child_color_scale=config.local_child_color_scale,
        local_child_opacity_scale=config.local_child_opacity_scale,
        guide_face_ids=guide_face_ids,
        guide_barycentric=guide_bary,
        guide_interpolation_k=config.guide_interpolation_k,
        guide_controls_flow=getattr(config, "guide_controls_flow", False),
        guide_length_residual_scale=getattr(config, "guide_length_residual_scale", 0.0),
        guide_bend_residual_scale=getattr(config, "guide_bend_residual_scale", 0.0),
        guide_flow_residual_scale=getattr(config, "guide_flow_residual_scale", 1.0),
        guide_width_residual_scale=getattr(config, "guide_width_residual_scale", 1.0),
        guide_flow_strength_residual_scale=getattr(config, "guide_flow_strength_residual_scale", 1.0),
        guide_lift_residual_scale=getattr(config, "guide_lift_residual_scale", 1.0),
        guide_stiffness_residual_scale=getattr(config, "guide_stiffness_residual_scale", 1.0),
        guide_child_radius_residual_scale=getattr(config, "guide_child_radius_residual_scale", 1.0),
        guide_clump_residual_scale=getattr(config, "guide_clump_residual_scale", 1.0),
        guide_curl_residual_scale=getattr(config, "guide_curl_residual_scale", 1.0),
        guide_frizz_residual_scale=getattr(config, "guide_frizz_residual_scale", 1.0),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, checkpoint


@torch.no_grad()
def run_diagnostics(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostics require CUDA because they use the same gsplat/nvdiffrast path as training")
    device = torch.device("cuda")
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, checkpoint = load_checkpoint_model(checkpoint_path, device)
    data_root = stage1.resolve_project_path(config.data_root)
    mesh_path = stage1.resolve_project_path(config.mesh_path)
    report = stage1.build_stage1_input_report(data_root, mesh_path, test_stride=config.test_stride)
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    image_paths = stage1.list_images(Path(report.image_dir))
    mask_paths = stage1.list_images(Path(report.mask_dir))
    viewmats, ks = stage1.load_camera_tensors(data_root, device)
    width, height = config.expected_width, config.expected_height
    view_index = int(args.view)

    import nvdiffrast.torch as dr

    mesh_depth_ctx = dr.RasterizeCudaContext(device=device)
    target = stage1.load_image(image_paths[view_index], device)
    mask = stage1.load_mask(mask_paths[view_index], device)
    mesh_color = stage1.sample_backing_color(config, device, train=False)
    scene_bg = stage1.scene_background_color(config, device)
    mesh_depth = stage1.render_model_mesh_depth(model, viewmats[view_index], ks[view_index], width, height, device=device, ctx=mesh_depth_ctx)
    backing_image = stage1.make_mesh_backing_image(mesh_depth, mesh_color, scene_bg)
    pred, alpha, gaussians, roots_local, render_stats, info = stage1.render_view(
        model,
        viewmats[view_index],
        ks[view_index],
        width,
        height,
        config,
        background=mesh_color,
        mesh_depth=mesh_depth,
        backing_image=backing_image,
    )
    target_eval = stage1.composite_target(target, mask, backing_image)
    metric = stage1.MetricComputer(compute_lpips=False).to(device)
    raw_metrics = metric.image_metrics(pred, target)
    composite_metrics = metric.image_metrics(pred, target_eval)

    stage1.save_image(output_dir / "view09_pred.png", pred)
    stage1.save_image(output_dir / "view09_target_composite.png", target_eval)
    stage1.save_image(output_dir / "view09_absdiff_x4.png", torch.abs(pred - target_eval).mul(4.0).clamp(0.0, 1.0))
    stage1.save_image(output_dir / "view09_alpha.png", alpha.expand(-1, -1, 3))
    stage1.save_image(output_dir / "view09_raw_fur.png", info["raw_fur_image"].clamp(0.0, 1.0))

    roots, _, roots_local_now = model.roots_and_normals()
    groom = model.apply_guide_controls(model.groom.decode(), roots_local_now)
    length = groom.length.reshape(-1).float()
    width_root = groom.root_width.reshape(-1).float()
    opacity = groom.opacity.reshape(-1).float()
    flow_strength = groom.flow_strength.reshape(-1).float()
    clump = groom.clump_strength.reshape(-1).float()
    child_radius = groom.child_radius.reshape(-1).float()
    curl = groom.curl_radius.reshape(-1).float()
    frizz = groom.frizz.reshape(-1).float()
    luma = (
        0.2126 * groom.root_color[:, 0]
        + 0.7152 * groom.root_color[:, 1]
        + 0.0722 * groom.root_color[:, 2]
    ).reshape(-1).float()
    root_ids = gaussians.root_indices.long().reshape(-1)
    gaussian_count = torch.bincount(root_ids, minlength=int(length.numel())).float()

    gaussian_xy, gaussian_depth = stage1.project_points(gaussians.means, viewmats[view_index], ks[view_index])
    gaussian_visible = (
        (gaussian_depth > 1.0e-6)
        & (gaussian_xy[:, 0] >= 0.0)
        & (gaussian_xy[:, 0] <= width - 1)
        & (gaussian_xy[:, 1] >= 0.0)
        & (gaussian_xy[:, 1] <= height - 1)
    )
    root_count = int(length.numel())
    inf = torch.full((root_count,), float("inf"), device=device, dtype=gaussian_xy.dtype)
    neg_inf = torch.full((root_count,), float("-inf"), device=device, dtype=gaussian_xy.dtype)
    visible_root_ids = root_ids[gaussian_visible]
    visible_xy = gaussian_xy[gaussian_visible]
    if int(visible_root_ids.numel()) > 0:
        min_x = inf.clone().scatter_reduce_(0, visible_root_ids, visible_xy[:, 0], reduce="amin", include_self=True)
        max_x = neg_inf.clone().scatter_reduce_(0, visible_root_ids, visible_xy[:, 0], reduce="amax", include_self=True)
        min_y = inf.clone().scatter_reduce_(0, visible_root_ids, visible_xy[:, 1], reduce="amin", include_self=True)
        max_y = neg_inf.clone().scatter_reduce_(0, visible_root_ids, visible_xy[:, 1], reduce="amax", include_self=True)
        visible_gaussian_count = torch.bincount(visible_root_ids, minlength=root_count).float()
        screen_diag = torch.sqrt((max_x - min_x).clamp_min(0.0).square() + (max_y - min_y).clamp_min(0.0).square())
        screen_diag = torch.where(torch.isfinite(screen_diag), screen_diag, torch.zeros_like(screen_diag))
    else:
        visible_gaussian_count = torch.zeros((root_count,), device=device, dtype=gaussian_xy.dtype)
        screen_diag = torch.zeros((root_count,), device=device, dtype=gaussian_xy.dtype)
    screen_stroke_score = screen_diag * width_root * opacity
    half_axis = gaussians.directions * gaussians.scales[:, :1]
    axis_a, depth_a = stage1.project_points(gaussians.means - half_axis, viewmats[view_index], ks[view_index])
    axis_b, depth_b = stage1.project_points(gaussians.means + half_axis, viewmats[view_index], ks[view_index])
    axis_in_frame = (
        (depth_a > 1.0e-6)
        & (depth_b > 1.0e-6)
        & (axis_a[:, 0] >= 0.0)
        & (axis_a[:, 0] <= width - 1)
        & (axis_a[:, 1] >= 0.0)
        & (axis_a[:, 1] <= height - 1)
        & (axis_b[:, 0] >= 0.0)
        & (axis_b[:, 0] <= width - 1)
        & (axis_b[:, 1] >= 0.0)
        & (axis_b[:, 1] <= height - 1)
    )
    gaussian_axis_px = torch.linalg.norm(axis_b - axis_a, dim=-1)
    gaussian_axis_px = torch.where(axis_in_frame, gaussian_axis_px, torch.zeros_like(gaussian_axis_px))
    root_axis_max = torch.zeros((root_count,), device=device, dtype=gaussian_axis_px.dtype)
    if int(root_ids.numel()) > 0:
        root_axis_max.scatter_reduce_(0, root_ids, gaussian_axis_px, reduce="amax", include_self=True)
    axis_brush_score = root_axis_max * width_root * opacity

    q = {
        "length": summarize_tensor(length),
        "root_width": summarize_tensor(width_root),
        "opacity": summarize_tensor(opacity),
        "flow_strength": summarize_tensor(flow_strength),
        "clump_strength": summarize_tensor(clump),
        "child_radius": summarize_tensor(child_radius),
        "curl_radius": summarize_tensor(curl),
        "frizz": summarize_tensor(frizz),
        "root_luma": summarize_tensor(luma),
        "gaussian_count_per_root": summarize_tensor(gaussian_count),
        "visible_gaussian_count_per_root": summarize_tensor(visible_gaussian_count),
        "screen_bbox_diag_px": summarize_tensor(screen_diag),
        "screen_stroke_score": summarize_tensor(screen_stroke_score),
        "root_major_axis_px": summarize_tensor(root_axis_max),
        "axis_brush_score": summarize_tensor(axis_brush_score),
    }

    masks: dict[str, torch.Tensor] = {
        "overlength_gt_0080": length > 0.080,
        "length_p95": length >= q["length"]["q95"],
        "width_p95": width_root >= q["root_width"]["q95"],
        "opacity_p95": opacity >= q["opacity"]["q95"],
        "high_flow_p95": flow_strength >= q["flow_strength"]["q95"],
        "high_gaussian_count_p95": gaussian_count >= q["gaussian_count_per_root"]["q95"],
        "dark_luma_le_038": luma <= 0.38,
        "dark_overlength": (luma <= 0.38) & (length > 0.080),
        "dark_high_capacity": (luma <= 0.38)
        & (length >= q["length"]["q90"])
        & (width_root >= q["root_width"]["q90"])
        & (opacity >= q["opacity"]["q90"]),
        "overlength_low_clump": (length > 0.080) & (clump <= q["clump_strength"]["q10"]),
        "overlength_high_child_radius": (length > 0.080) & (child_radius >= q["child_radius"]["q90"]),
        "overlength_high_curl": (length > 0.080) & (curl >= q["curl_radius"]["q90"]),
        "overlength_high_frizz": (length > 0.080) & (frizz >= q["frizz"]["q90"]),
        "screen_diag_p95": screen_diag >= q["screen_bbox_diag_px"]["q95"],
        "dark_screen_stroke": (luma <= 0.38)
        & (screen_diag >= q["screen_bbox_diag_px"]["q90"])
        & (screen_stroke_score >= q["screen_stroke_score"]["q90"]),
        "visible_screen_stroke_p95": screen_stroke_score >= q["screen_stroke_score"]["q95"],
        "major_axis_p95": root_axis_max >= q["root_major_axis_px"]["q95"],
        "axis_brush_p95": axis_brush_score >= q["axis_brush_score"]["q95"],
        "dark_axis_brush": (luma <= 0.38)
        & (root_axis_max >= q["root_major_axis_px"]["q90"])
        & (axis_brush_score >= q["axis_brush_score"]["q90"]),
        "visible_bbox_not_axis": (screen_stroke_score >= q["screen_stroke_score"]["q95"])
        & (root_axis_max < q["root_major_axis_px"]["q75"]),
    }

    colors = {
        "overlength_gt_0080": (1.0, 0.20, 0.02),
        "length_p95": (1.0, 0.0, 0.0),
        "width_p95": (1.0, 0.85, 0.0),
        "opacity_p95": (0.0, 0.85, 1.0),
        "high_flow_p95": (0.25, 1.0, 0.25),
        "high_gaussian_count_p95": (0.9, 0.1, 1.0),
        "dark_luma_le_038": (0.1, 0.1, 1.0),
        "dark_overlength": (1.0, 0.0, 0.75),
        "dark_high_capacity": (1.0, 0.05, 0.02),
        "overlength_low_clump": (0.3, 0.9, 1.0),
        "overlength_high_child_radius": (0.7, 0.25, 1.0),
        "overlength_high_curl": (1.0, 0.5, 0.0),
        "overlength_high_frizz": (0.0, 1.0, 0.65),
        "screen_diag_p95": (0.0, 0.75, 1.0),
        "dark_screen_stroke": (1.0, 0.0, 0.0),
        "visible_screen_stroke_p95": (0.25, 0.0, 1.0),
        "major_axis_p95": (0.0, 1.0, 0.85),
        "axis_brush_p95": (0.9, 0.0, 1.0),
        "dark_axis_brush": (1.0, 0.25, 0.0),
        "visible_bbox_not_axis": (0.0, 0.45, 1.0),
    }

    category_stats = {}
    for name, mask_tensor in masks.items():
        color = colors[name]
        category_dir = output_dir / name
        category_dir.mkdir(parents=True, exist_ok=True)
        overlay_stats = save_gaussian_root_overlay(
            category_dir / "gaussian_overlay.png",
            pred,
            gaussians,
            mask_tensor,
            viewmats[view_index],
            ks[view_index],
            width,
            height,
            color=color,
        )
        point_stats = save_root_point_overlay(
            category_dir / "root_points.png",
            pred,
            roots,
            mask_tensor,
            viewmats[view_index],
            ks[view_index],
            width,
            height,
            color=(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 210),
        )
        idx = torch.nonzero(mask_tensor, as_tuple=False).reshape(-1)
        if int(idx.numel()) > 0:
            category_stats[name] = {
                **overlay_stats,
                **point_stats,
                "length_mean": float(length[idx].mean().cpu()),
                "length_max": float(length[idx].max().cpu()),
                "width_mean": float(width_root[idx].mean().cpu()),
                "opacity_mean": float(opacity[idx].mean().cpu()),
                "luma_mean": float(luma[idx].mean().cpu()),
                "flow_strength_mean": float(flow_strength[idx].mean().cpu()),
                "clump_strength_mean": float(clump[idx].mean().cpu()),
                "child_radius_mean": float(child_radius[idx].mean().cpu()),
                "curl_radius_mean": float(curl[idx].mean().cpu()),
                "frizz_mean": float(frizz[idx].mean().cpu()),
                "gaussian_count_mean": float(gaussian_count[idx].mean().cpu()),
                "visible_gaussian_count_mean": float(visible_gaussian_count[idx].mean().cpu()),
                "screen_bbox_diag_px_mean": float(screen_diag[idx].mean().cpu()),
                "screen_bbox_diag_px_max": float(screen_diag[idx].max().cpu()),
                "screen_stroke_score_mean": float(screen_stroke_score[idx].mean().cpu()),
                "root_major_axis_px_mean": float(root_axis_max[idx].mean().cpu()),
                "root_major_axis_px_max": float(root_axis_max[idx].max().cpu()),
                "axis_brush_score_mean": float(axis_brush_score[idx].mean().cpu()),
            }
        else:
            category_stats[name] = {**overlay_stats, **point_stats}

    summary = {
        "checkpoint": str(checkpoint_path),
        "iteration": int(checkpoint.get("iteration", -1)),
        "view": view_index,
        "metrics": {
            "raw_psnr": float(raw_metrics["psnr"].detach().cpu()),
            "raw_ssim": float(raw_metrics["ssim"].detach().cpu()),
            "composite_psnr": float(composite_metrics["psnr"].detach().cpu()),
            "composite_ssim": float(composite_metrics["ssim"].detach().cpu()),
        },
        "render_stats": render_stats,
        "groom_stats": q,
        "categories": category_stats,
    }
    (output_dir / "streak_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "metrics": summary["metrics"], "categories": category_stats}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=r"D:\petsgaussianhair\_downloads\white_tiger_view09_stage1_strict_length_split_1200_20260627\checkpoint_001200.pt",
    )
    parser.add_argument("--view", type=int, default=9)
    parser.add_argument(
        "--output-dir",
        default=r"D:\petsgaussianhair\_downloads\white_tiger_strict_length_streak_diagnostics_20260627",
    )
    run_diagnostics(parser.parse_args())


if __name__ == "__main__":
    main()

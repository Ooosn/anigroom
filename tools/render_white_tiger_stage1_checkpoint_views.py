from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.data.white_tiger import build_stage1_input_report, list_images  # noqa: E402
from tools.train_white_tiger_stage1 import (  # noqa: E402
    build_stage1_model_from_checkpoint,
    composite_target,
    depth_to_image,
    load_camera_tensors,
    load_image,
    load_training_checkpoint,
    load_mask,
    make_mesh_backing_image,
    render_model_mesh_depth,
    render_view,
    resolve_project_path,
    sample_backing_color,
    save_image,
    scene_background_color,
    stage1_config_from_checkpoint_mapping,
)


def parse_view_ids(value: str) -> list[int]:
    ids: list[int] = []
    for part in value.replace(",", " ").split():
        if part.strip():
            ids.append(int(part))
    if not ids:
        raise ValueError("at least one view id is required")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Render selected full-resolution views from a White Tiger Stage1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view-ids", default="0 5 9 14 18 21 27 32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint view rendering")
    device = torch.device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_training_checkpoint(checkpoint_path)
    config = stage1_config_from_checkpoint_mapping(checkpoint["config"])
    model = build_stage1_model_from_checkpoint(checkpoint, config, device)

    data_root = resolve_project_path(config.data_root)
    mesh_path = resolve_project_path(config.mesh_path)
    report = build_stage1_input_report(data_root, mesh_path, test_stride=config.test_stride)
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    image_paths = list_images(Path(report.image_dir))
    mask_paths = list_images(Path(report.mask_dir))
    viewmats, ks = load_camera_tensors(data_root, device)
    width, height = int(config.expected_width), int(config.expected_height)
    view_ids = parse_view_ids(args.view_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_depth_ctx = None
    if config.mesh_depth_clipping or config.mesh_backing_compositing:
        import nvdiffrast.torch as dr

        mesh_depth_ctx = dr.RasterizeCudaContext(device=device)

    mesh_color = sample_backing_color(config, device, train=False)
    scene_bg = scene_background_color(config, device)
    residual_enabled = model.gaussian_rgb_residual is not None
    residual_multiplier = float(model.gaussian_rgb_residual_multiplier)
    residual_active = residual_enabled and abs(residual_multiplier) > 0.0
    shape_detail_multiplier = float(model.shape_detail_multiplier)
    shape_detail_active = (
        abs(shape_detail_multiplier) > 0.0
        and float(model.shape_curl_scale) > 0.0
    )
    local_render_color_enabled = model.child_color_delta_raw is not None
    guide_view_sh_enabled = model.guide_view_sh is not None
    guide_view_sh_stats = (
        model.guide_view_sh.stats() if guide_view_sh_enabled else None
    )
    guide_view_sh_active = bool(
        guide_view_sh_stats is not None
        and float(guide_view_sh_stats["active_fraction"]) > 0.0
    )
    local_render_color_stats = None
    if local_render_color_enabled:
        local_render_color = (
            torch.tanh(model.child_color_delta_raw.detach())
            * float(model.local_child_color_scale)
        )
        local_render_color_stats = {
            "abs_mean": float(local_render_color.abs().mean().cpu()),
            "rms": float(torch.sqrt(local_render_color.square().mean()).cpu()),
            "abs_max": float(local_render_color.abs().max().cpu()),
        }
        del local_render_color
    records = []
    with torch.no_grad():
        for idx in view_ids:
            if idx < 0 or idx >= len(image_paths):
                raise RuntimeError(f"view id out of range: {idx}, available={len(image_paths)}")
            target = load_image(image_paths[idx], device)
            mask = load_mask(mask_paths[idx], device)
            mesh_depth = render_model_mesh_depth(model, viewmats[idx], ks[idx], width, height, device=device, ctx=mesh_depth_ctx)
            backing = make_mesh_backing_image(mesh_depth, mesh_color, scene_bg)
            pred, alpha, _, _, stats, _ = render_view(
                model,
                viewmats[idx],
                ks[idx],
                width,
                height,
                config,
                background=mesh_color,
                mesh_depth=mesh_depth,
                backing_image=backing,
                view_index=idx,
            )
            target_eval = composite_target(target, mask, backing)
            raw_diff = torch.abs(pred - target) * 4.0
            composite_diff = torch.abs(pred - target_eval) * 4.0
            save_image(output_dir / f"view_{idx:02d}_pred.png", pred)
            save_image(output_dir / f"view_{idx:02d}_gt.png", target)
            save_image(output_dir / f"view_{idx:02d}_target_eval.png", target_eval)
            save_image(output_dir / f"view_{idx:02d}_alpha.png", alpha)
            save_image(output_dir / f"view_{idx:02d}_mesh_depth.png", depth_to_image(mesh_depth.depth))
            save_image(output_dir / f"view_{idx:02d}_raw_diff_x4.png", raw_diff)
            save_image(output_dir / f"view_{idx:02d}_composite_diff_x4.png", composite_diff)
            raw_mse = torch.mean((pred - target).square()).clamp_min(1.0e-12)
            comp_mse = torch.mean((pred - target_eval).square()).clamp_min(1.0e-12)
            record = {
                "view": int(idx),
                "pred": f"view_{idx:02d}_pred.png",
                "gt": f"view_{idx:02d}_gt.png",
                "raw_psnr": float((-10.0 * torch.log10(raw_mse)).detach().cpu()),
                "composite_psnr": float((-10.0 * torch.log10(comp_mse)).detach().cpu()),
                "stats": stats,
            }
            without_guide_view_sh = None
            if guide_view_sh_active:
                saved_guide_view_sh = model.guide_view_sh.raw.detach().clone()
                model.guide_view_sh.raw.zero_()
                try:
                    without_guide_view_sh, _, _, _, without_guide_view_sh_stats, _ = render_view(
                        model,
                        viewmats[idx],
                        ks[idx],
                        width,
                        height,
                        config,
                        background=mesh_color,
                        mesh_depth=mesh_depth,
                        backing_image=backing,
                        view_index=idx,
                    )
                finally:
                    model.guide_view_sh.raw.copy_(saved_guide_view_sh)
                without_sh_mse = torch.mean(
                    (without_guide_view_sh - target_eval).square()
                ).clamp_min(1.0e-12)
                without_sh_psnr = -10.0 * torch.log10(without_sh_mse)
                guide_sh_delta = pred - without_guide_view_sh
                save_image(
                    output_dir / f"view_{idx:02d}_pred_without_guide_view_sh.png",
                    without_guide_view_sh,
                )
                save_image(
                    output_dir / f"view_{idx:02d}_guide_view_sh_abs_x4.png",
                    (guide_sh_delta.abs() * 4.0).clamp(0.0, 1.0),
                )
                record.update(
                    {
                        "pred_without_guide_view_sh": f"view_{idx:02d}_pred_without_guide_view_sh.png",
                        "guide_view_sh_abs_x4": f"view_{idx:02d}_guide_view_sh_abs_x4.png",
                        "composite_psnr_without_guide_view_sh": float(
                            without_sh_psnr.detach().cpu()
                        ),
                        "composite_psnr_gain_from_guide_view_sh": float(
                            ((-10.0 * torch.log10(comp_mse)) - without_sh_psnr)
                            .detach()
                            .cpu()
                        ),
                        "guide_view_sh_image_abs_mean": float(
                            guide_sh_delta.abs().mean().detach().cpu()
                        ),
                        "without_guide_view_sh_stats": without_guide_view_sh_stats,
                    }
                )
                del (
                    saved_guide_view_sh,
                    without_sh_mse,
                    without_sh_psnr,
                    guide_sh_delta,
                )
            base_pred = None
            residual_abs = None
            residual_signed = None
            if residual_active:
                model.gaussian_rgb_residual_multiplier = 0.0
                try:
                    base_pred, _, _, _, base_stats, _ = render_view(
                        model,
                        viewmats[idx],
                        ks[idx],
                        width,
                        height,
                        config,
                        background=mesh_color,
                        mesh_depth=mesh_depth,
                        backing_image=backing,
                    )
                finally:
                    model.gaussian_rgb_residual_multiplier = residual_multiplier
                residual_delta = pred - base_pred
                residual_abs = (residual_delta.abs() * 4.0).clamp(0.0, 1.0)
                residual_signed = (0.5 + residual_delta * 4.0).clamp(0.0, 1.0)
                base_mse = torch.mean((base_pred - target_eval).square()).clamp_min(1.0e-12)
                base_psnr = -10.0 * torch.log10(base_mse)
                save_image(output_dir / f"view_{idx:02d}_pred_without_gaussian_rgb_residual.png", base_pred)
                save_image(output_dir / f"view_{idx:02d}_gaussian_rgb_residual_abs_x4.png", residual_abs)
                save_image(output_dir / f"view_{idx:02d}_gaussian_rgb_residual_signed_x4.png", residual_signed)
                record.update(
                    {
                        "pred_without_gaussian_rgb_residual": f"view_{idx:02d}_pred_without_gaussian_rgb_residual.png",
                        "gaussian_rgb_residual_abs_x4": f"view_{idx:02d}_gaussian_rgb_residual_abs_x4.png",
                        "gaussian_rgb_residual_signed_x4": f"view_{idx:02d}_gaussian_rgb_residual_signed_x4.png",
                        "composite_psnr_without_gaussian_rgb_residual": float(base_psnr.detach().cpu()),
                        "composite_psnr_gain_from_gaussian_rgb_residual": float(
                            ((-10.0 * torch.log10(comp_mse)) - base_psnr).detach().cpu()
                        ),
                        "gaussian_rgb_residual_image_abs_mean": float(residual_delta.abs().mean().detach().cpu()),
                        "base_stats": base_stats,
                    }
                )
                del residual_delta, base_mse, base_psnr
            without_shape_detail = None
            without_shape_or_residual = None
            if shape_detail_active:
                model.shape_detail_multiplier = 0.0
                try:
                    without_shape_detail, _, _, _, without_shape_stats, _ = render_view(
                        model,
                        viewmats[idx],
                        ks[idx],
                        width,
                        height,
                        config,
                        background=mesh_color,
                        mesh_depth=mesh_depth,
                        backing_image=backing,
                    )
                    if residual_active:
                        model.gaussian_rgb_residual_multiplier = 0.0
                        try:
                            without_shape_or_residual, _, _, _, both_off_stats, _ = render_view(
                                model,
                                viewmats[idx],
                                ks[idx],
                                width,
                                height,
                                config,
                                background=mesh_color,
                                mesh_depth=mesh_depth,
                                backing_image=backing,
                            )
                        finally:
                            model.gaussian_rgb_residual_multiplier = residual_multiplier
                    else:
                        without_shape_or_residual = without_shape_detail
                        both_off_stats = without_shape_stats
                finally:
                    model.shape_detail_multiplier = shape_detail_multiplier

                without_shape_mse = torch.mean(
                    (without_shape_detail - target_eval).square()
                ).clamp_min(1.0e-12)
                both_off_mse = torch.mean(
                    (without_shape_or_residual - target_eval).square()
                ).clamp_min(1.0e-12)
                shape_delta = pred - without_shape_detail
                save_image(
                    output_dir / f"view_{idx:02d}_pred_without_shape_detail.png",
                    without_shape_detail,
                )
                save_image(
                    output_dir / f"view_{idx:02d}_shape_detail_abs_x4.png",
                    (shape_delta.abs() * 4.0).clamp(0.0, 1.0),
                )
                save_image(
                    output_dir / f"view_{idx:02d}_pred_without_shape_detail_or_gaussian_rgb_residual.png",
                    without_shape_or_residual,
                )
                record.update(
                    {
                        "pred_without_shape_detail": f"view_{idx:02d}_pred_without_shape_detail.png",
                        "shape_detail_abs_x4": f"view_{idx:02d}_shape_detail_abs_x4.png",
                        "pred_without_shape_detail_or_gaussian_rgb_residual": (
                            f"view_{idx:02d}_pred_without_shape_detail_or_gaussian_rgb_residual.png"
                        ),
                        "composite_psnr_without_shape_detail": float(
                            (-10.0 * torch.log10(without_shape_mse)).cpu()
                        ),
                        "composite_psnr_without_shape_detail_or_gaussian_rgb_residual": float(
                            (-10.0 * torch.log10(both_off_mse)).cpu()
                        ),
                        "shape_detail_image_abs_mean": float(shape_delta.abs().mean().cpu()),
                        "without_shape_detail_stats": without_shape_stats,
                        "without_shape_detail_or_gaussian_rgb_residual_stats": both_off_stats,
                    }
                )
                del without_shape_mse, both_off_mse, shape_delta
            without_local_render_color = None
            root_tip_only = None
            if local_render_color_enabled:
                saved_local_render_color = model.child_color_delta_raw.detach().clone()
                model.child_color_delta_raw.zero_()
                try:
                    without_local_render_color, _, _, _, without_local_stats, _ = render_view(
                        model,
                        viewmats[idx],
                        ks[idx],
                        width,
                        height,
                        config,
                        background=mesh_color,
                        mesh_depth=mesh_depth,
                        backing_image=backing,
                    )
                    if residual_active:
                        model.gaussian_rgb_residual_multiplier = 0.0
                        try:
                            root_tip_only, _, _, _, root_tip_stats, _ = render_view(
                                model,
                                viewmats[idx],
                                ks[idx],
                                width,
                                height,
                                config,
                                background=mesh_color,
                                mesh_depth=mesh_depth,
                                backing_image=backing,
                            )
                        finally:
                            model.gaussian_rgb_residual_multiplier = residual_multiplier
                    else:
                        root_tip_only = without_local_render_color
                        root_tip_stats = without_local_stats
                finally:
                    model.child_color_delta_raw.copy_(saved_local_render_color)

                without_local_mse = torch.mean(
                    (without_local_render_color - target_eval).square()
                ).clamp_min(1.0e-12)
                root_tip_mse = torch.mean((root_tip_only - target_eval).square()).clamp_min(1.0e-12)
                local_render_delta = pred - without_local_render_color
                save_image(
                    output_dir / f"view_{idx:02d}_pred_without_local_render_color.png",
                    without_local_render_color,
                )
                save_image(
                    output_dir / f"view_{idx:02d}_pred_root_tip_only.png",
                    root_tip_only,
                )
                save_image(
                    output_dir / f"view_{idx:02d}_local_render_color_abs_x4.png",
                    (local_render_delta.abs() * 4.0).clamp(0.0, 1.0),
                )
                record.update(
                    {
                        "pred_without_local_render_color": (
                            f"view_{idx:02d}_pred_without_local_render_color.png"
                        ),
                        "pred_root_tip_only": f"view_{idx:02d}_pred_root_tip_only.png",
                        "local_render_color_abs_x4": (
                            f"view_{idx:02d}_local_render_color_abs_x4.png"
                        ),
                        "composite_psnr_without_local_render_color": float(
                            (-10.0 * torch.log10(without_local_mse)).cpu()
                        ),
                        "composite_psnr_root_tip_only": float(
                            (-10.0 * torch.log10(root_tip_mse)).cpu()
                        ),
                        "without_local_render_color_stats": without_local_stats,
                        "root_tip_only_stats": root_tip_stats,
                    }
                )
                del (
                    saved_local_render_color,
                    without_local_mse,
                    root_tip_mse,
                    local_render_delta,
                )
            records.append(record)
            del target, mask, mesh_depth, backing, pred, alpha, target_eval, raw_diff, composite_diff
            if base_pred is not None:
                del base_pred, residual_abs, residual_signed
            if without_guide_view_sh is not None:
                del without_guide_view_sh
            if without_local_render_color is not None:
                del without_local_render_color, root_tip_only
            if without_shape_detail is not None:
                del without_shape_detail, without_shape_or_residual
            torch.cuda.empty_cache()

    summary = {
        "checkpoint": str(checkpoint_path),
        "iteration": int(checkpoint.get("iteration", -1)),
        "view_ids": view_ids,
        "width": width,
        "height": height,
        "guide_residual_multiplier": float(model.guide_residual_multiplier),
        "guide_coverage_residual_multiplier": float(model.guide_coverage_residual_multiplier),
        "shape_detail_multiplier": float(model.shape_detail_multiplier),
        "shape_detail_active": bool(shape_detail_active),
        "gaussian_rgb_residual": (
            model.gaussian_rgb_residual.stats(multiplier=residual_multiplier)
            if residual_enabled
            else None
        ),
        "guide_view_sh": guide_view_sh_stats,
        "local_render_color": local_render_color_stats,
        "records": records,
    }
    (output_dir / "render_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

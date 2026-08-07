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
            records.append(
                {
                    "view": int(idx),
                    "pred": f"view_{idx:02d}_pred.png",
                    "gt": f"view_{idx:02d}_gt.png",
                    "raw_psnr": float((-10.0 * torch.log10(raw_mse)).detach().cpu()),
                    "composite_psnr": float((-10.0 * torch.log10(comp_mse)).detach().cpu()),
                    "stats": stats,
                }
            )
            del target, mask, mesh_depth, backing, pred, alpha, target_eval, raw_diff, composite_diff
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
        "records": records,
    }
    (output_dir / "render_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

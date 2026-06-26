"""Validate grooming parameters through gsplat rendering.

This is a local module test, not the formal Stage 1 trainer.  It checks:
1. explicit grooming parameters generate plausible strands;
2. strand segments convert to gsplat-compatible Gaussian parameters;
3. rendered supervision can backpropagate into grooming parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.baselines.plain_gsplat import load_camera_tensors
from anigroom.data.white_tiger import build_stage1_input_report
from anigroom.grooming import (
    GroomParameterField,
    adaptive_resample_strands,
    build_strands,
    make_tangent_frames,
    strands_to_gaussians,
)
from anigroom.mesh_roots import read_obj_mesh


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0], mode="L")
    return Image.fromarray(arr, mode="RGB")


def crop_foreground(image: Image.Image, margin: int = 18) -> Image.Image:
    arr = np.asarray(image.convert("RGB"))
    mask = np.any(arr < 248, axis=-1)
    if not mask.any():
        return image
    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, image.width)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, image.height)
    return image.crop((x0, y0, x1, y1))


def make_contact_sheet(paths: list[Path], output: Path, tile_width: int = 384, crop: bool = True) -> None:
    if not paths:
        return
    thumbs: list[tuple[str, Image.Image]] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if crop:
            image = crop_foreground(image)
        image.thumbnail((tile_width, tile_width), Image.Resampling.LANCZOS)
        thumbs.append((path.stem, image.copy()))
    cols = min(3, len(thumbs))
    label_h = 26
    tile_h = max(image.height for _, image in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * tile_width, rows * (tile_h + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = col * tile_width + (tile_width - image.width) // 2
        y = row * (tile_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((col * tile_width + 8, y + tile_h + 5), label, fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def parse_views(text: str, fallback: list[int]) -> list[int]:
    if text.strip():
        return [int(part) for part in text.split(",") if part.strip()]
    return fallback[:3]


def load_root_subset(root_path: Path, mesh_path: Path, root_count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(root_path)
    points = data["root_positions"].astype(np.float32)
    face_ids = data["face_ids"].astype(np.int64)
    if points.shape[0] < root_count:
        raise ValueError(f"root file has {points.shape[0]} roots, requested {root_count}")
    points = points[:root_count]
    face_ids = face_ids[:root_count]
    mesh = read_obj_mesh(mesh_path)
    tri = mesh.vertices[mesh.faces[face_ids]]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = normals / np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    return (
        torch.from_numpy(points).to(device=device),
        torch.from_numpy(normals.astype(np.float32)).to(device=device),
    )


def scaled_cameras(data_root: Path, device: torch.device, render_width: int, render_height: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    report = build_stage1_input_report(
        data_root,
        Path("D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj"),
    )
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    orig_width, orig_height = report.image_size or (0, 0)
    if orig_width <= 0 or orig_height <= 0:
        raise RuntimeError("invalid source image size")
    viewmats, ks = load_camera_tensors(data_root, device)
    ks = ks.clone()
    sx = float(render_width) / float(orig_width)
    sy = float(render_height) / float(orig_height)
    ks[:, 0, 0] *= sx
    ks[:, 1, 1] *= sy
    ks[:, 0, 2] *= sx
    ks[:, 1, 2] *= sy
    return viewmats, ks, orig_width, orig_height


def apply_teacher_pattern(field: GroomParameterField, roots: torch.Tensor) -> None:
    """Create a visible but still grooming-like target parameter field."""

    with torch.no_grad():
        x = roots[:, [0]]
        y = roots[:, [1]]
        z = roots[:, [2]]
        length_wave = 0.55 * torch.sin(8.0 * z + 2.5 * x) + 0.35 * torch.cos(5.0 * y)
        stripe = torch.sigmoid(4.0 * (torch.sin(26.0 * z + 8.0 * x) - 0.30))
        head = torch.sigmoid(10.0 * (x - torch.quantile(x, 0.70)))
        field.length_raw.add_(1.10 * length_wave + 0.70 * head)
        field.flow_xy[:, 0:1].add_(0.40 * torch.sin(7.0 * z))
        field.flow_xy[:, 1:2].add_(0.35 * torch.cos(6.0 * x))
        field.lift_raw.add_(0.40 * head)
        field.sag_raw.add_(0.55 * torch.sigmoid(6.0 * (0.15 - y)))
        field.bend_raw.add_(0.60 * torch.sin(9.0 * z))
        field.root_width_raw.add_(0.55 * stripe + 0.25 * head)
        field.tip_width_ratio_raw.add_(0.35 * torch.cos(6.0 * z))
        dark = stripe.expand(-1, 3)
        field.root_color_raw.add_(-2.20 * dark + 0.25 * head.expand(-1, 3))
        field.tip_color_raw.add_(-1.80 * dark)
        field.opacity_raw.add_(0.45 + 0.25 * head)


def render_groom(
    field: GroomParameterField,
    roots: torch.Tensor,
    normals: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    view_indices: list[int],
    width: int,
    height: int,
    samples: int,
    min_segments: int,
    max_segments: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float | int]]:
    from gsplat.rendering import rasterization

    tangents, bitangents = make_tangent_frames(normals)
    groom = field.decode()
    strands, widths, colors, opacities = build_strands(roots, normals, tangents, bitangents, groom, samples=samples)
    resampled = adaptive_resample_strands(strands, widths, colors, opacities, groom.length, min_segments, max_segments)
    gaussians = strands_to_gaussians(
        resampled.strands,
        resampled.widths,
        resampled.colors,
        resampled.opacities,
        resampled.segment_mask,
    )
    background = torch.ones((1, 3), device=roots.device)
    images: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    for idx in view_indices:
        render, alpha, _ = rasterization(
            gaussians.means,
            gaussians.quats,
            gaussians.scales,
            gaussians.opacities.reshape(-1),
            gaussians.colors,
            viewmats[idx : idx + 1],
            ks[idx : idx + 1],
            width,
            height,
            packed=False,
            backgrounds=background,
        )
        images.append(render[0].clamp(0.0, 1.0))
        alphas.append(alpha[0].clamp(0.0, 1.0))
    stats = {
        **resampled.stats,
        "gaussian_count": int(gaussians.means.shape[0]),
        "strand_count": int(roots.shape[0]),
        "sample_count": int(samples),
    }
    return images, alphas, stats


def decoded_error(student: GroomParameterField, teacher: GroomParameterField) -> dict[str, float]:
    s = student.decode()
    t = teacher.decode()
    return {
        "length_l1": float((s.length - t.length).abs().mean().detach().cpu()),
        "root_width_l1": float((s.root_width - t.root_width).abs().mean().detach().cpu()),
        "tip_width_l1": float((s.tip_width - t.tip_width).abs().mean().detach().cpu()),
        "flow_xy_l1": float((s.flow_xy - t.flow_xy).abs().mean().detach().cpu()),
        "root_color_l1": float((s.root_color - t.root_color).abs().mean().detach().cpu()),
        "tip_color_l1": float((s.tip_color - t.tip_color).abs().mean().detach().cpu()),
        "opacity_l1": float((s.opacity - t.opacity).abs().mean().detach().cpu()),
    }


def save_controlled_edit_panel(
    output_dir: Path,
    roots: torch.Tensor,
    normals: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    view_indices: list[int],
    width: int,
    height: int,
    samples: int,
    min_segments: int,
    max_segments: int,
) -> list[Path]:
    """Render direct grooming edits for visual sanity checks."""

    edit_specs = []
    base = GroomParameterField(int(roots.shape[0]), device=roots.device)
    edit_specs.append(("edit_default", base))

    longer = GroomParameterField(int(roots.shape[0]), device=roots.device)
    with torch.no_grad():
        longer.length_raw.add_(3.10)
        longer.root_width_raw.add_(1.65)
        longer.tip_width_ratio_raw.add_(0.55)
        longer.opacity_raw.add_(0.80)
    edit_specs.append(("edit_longer_wider", longer))

    brushed = GroomParameterField(int(roots.shape[0]), device=roots.device)
    with torch.no_grad():
        brushed.flow_xy[:, 0:1].add_(1.90)
        brushed.flow_xy[:, 1:2].add_(1.45)
        brushed.bend_raw.add_(2.20)
        brushed.sag_raw.add_(1.80)
        brushed.stiffness_raw.sub_(1.25)
        brushed.length_raw.add_(1.25)
        brushed.opacity_raw.add_(0.45)
    edit_specs.append(("edit_brushed_bent", brushed))

    dark = GroomParameterField(int(roots.shape[0]), device=roots.device)
    with torch.no_grad():
        z = roots[:, [2]]
        stripe = torch.sigmoid(5.0 * (torch.sin(30.0 * z) - 0.15)).expand(-1, 3)
        dark.root_color_raw.add_(-2.4 * stripe)
        dark.tip_color_raw.add_(-2.1 * stripe)
        dark.length_raw.add_(0.35 * stripe[:, :1])
    edit_specs.append(("edit_dark_stripes", dark))

    written: list[Path] = []
    for label, field in edit_specs:
        images, _, _ = render_groom(
            field,
            roots,
            normals,
            viewmats,
            ks,
            view_indices,
            width,
            height,
            samples,
            min_segments,
            max_segments,
        )
        for view_idx, image in zip(view_indices, images):
            path = output_dir / f"{label}_view_{view_idx:04d}.png"
            tensor_to_image(image).save(path)
            written.append(path)
    make_contact_sheet(written, output_dir / "controlled_edit_sheet.jpg")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk")
    parser.add_argument("--mesh-path", default="D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj")
    parser.add_argument("--root-path", default="D:/petsgaussianhair/_downloads/root_init_20260623/white_tiger_roots_2048_fps.npz")
    parser.add_argument("--output-dir", default="D:/petsgaussianhair/_downloads/grooming_gsplat_roundtrip_20260623")
    parser.add_argument("--root-count", type=int, default=512)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=13)
    parser.add_argument("--views", default="")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this gsplat roundtrip test requires CUDA")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    mesh_path = Path(args.mesh_path)
    report = build_stage1_input_report(data_root, mesh_path)
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    view_indices = parse_views(args.views, report.train_indices)
    viewmats, ks, orig_width, orig_height = scaled_cameras(data_root, device, args.width, args.height)
    roots, normals = load_root_subset(Path(args.root_path), mesh_path, args.root_count, device)

    teacher = GroomParameterField(args.root_count, device=device)
    apply_teacher_pattern(teacher, roots)
    student = GroomParameterField(args.root_count, device=device)
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    with torch.no_grad():
        target_images, target_alphas, target_stats = render_groom(
            teacher,
            roots,
            normals,
            viewmats,
            ks,
            view_indices,
            args.width,
            args.height,
            args.samples,
            args.min_segments,
            args.max_segments,
        )
        init_images, init_alphas, init_stats = render_groom(
            student,
            roots,
            normals,
            viewmats,
            ks,
            view_indices,
            args.width,
            args.height,
            args.samples,
            args.min_segments,
            args.max_segments,
        )

    initial_param_error = decoded_error(student, teacher)
    history = []
    for iteration in range(1, args.iterations + 1):
        pred_images, pred_alphas, render_stats = render_groom(
            student,
            roots,
            normals,
            viewmats,
            ks,
            view_indices,
            args.width,
            args.height,
            args.samples,
            args.min_segments,
            args.max_segments,
        )
        image_loss = torch.stack([F.mse_loss(pred, target) for pred, target in zip(pred_images, target_images)]).mean()
        alpha_loss = torch.stack([F.mse_loss(pred, target) for pred, target in zip(pred_alphas, target_alphas)]).mean()
        loss = image_loss + 0.2 * alpha_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(student.length_raw.grad.detach().norm().cpu()) if student.length_raw.grad is not None else 0.0
        optimizer.step()
        if iteration == 1 or iteration % 10 == 0 or iteration == args.iterations:
            history.append(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach().cpu()),
                    "image_mse": float(image_loss.detach().cpu()),
                    "alpha_mse": float(alpha_loss.detach().cpu()),
                    "length_grad_norm": grad_norm,
                    **render_stats,
                }
            )
            print(json.dumps(history[-1]), flush=True)

    with torch.no_grad():
        final_images, final_alphas, final_stats = render_groom(
            student,
            roots,
            normals,
            viewmats,
            ks,
            view_indices,
            args.width,
            args.height,
            args.samples,
            args.min_segments,
            args.max_segments,
        )
    final_param_error = decoded_error(student, teacher)

    written: list[Path] = []
    for label, images in [
        ("teacher", target_images),
        ("student_init", init_images),
        ("student_final", final_images),
    ]:
        for view, image in zip(view_indices, images):
            path = output_dir / f"{label}_view_{view:04d}.png"
            tensor_to_image(image).save(path)
            written.append(path)
    for label, alphas in [
        ("teacher_alpha", target_alphas),
        ("student_init_alpha", init_alphas),
        ("student_final_alpha", final_alphas),
    ]:
        for view, alpha in zip(view_indices, alphas):
            path = output_dir / f"{label}_view_{view:04d}.png"
            tensor_to_image(alpha).save(path)

    edit_paths = save_controlled_edit_panel(
        output_dir,
        roots,
        normals,
        viewmats,
        ks,
        view_indices[:3],
        args.width,
        args.height,
        args.samples,
        args.min_segments,
        args.max_segments,
    )
    make_contact_sheet(written, output_dir / "contact_sheet.jpg")
    metrics = {
        "data_root": str(data_root),
        "mesh_path": str(mesh_path),
        "root_path": str(args.root_path),
        "root_count": args.root_count,
        "views": view_indices,
        "source_image_size": [orig_width, orig_height],
        "render_size": [args.width, args.height],
        "target_stats": target_stats,
        "init_stats": init_stats,
        "final_stats": final_stats,
        "initial_param_error": initial_param_error,
        "final_param_error": final_param_error,
        "history": history,
        "controlled_edit_images": [str(path) for path in edit_paths],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "contact_sheet": str(output_dir / "contact_sheet.jpg"), "metrics": str(output_dir / "metrics.json")}))


if __name__ == "__main__":
    main()

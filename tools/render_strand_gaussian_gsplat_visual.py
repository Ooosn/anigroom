"""Render a groom-strand reference against AniGroom's 3D Gaussian conversion.

The input contract is Blender-like curve strands: [N, S, 3] polylines plus
optional widths/colors/opacities. The output is a side-by-side image showing:

1. reference polyline rasterization from the same camera
2. gsplat rendering after continuous strand-segment Gaussian conversion
3. absolute difference

This is a visual test for "does the 3DGS representation still look like the
groom strands?", not a training or densification test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import adaptive_resample_strands, resample_strands_to_segment_budgets, strands_to_gaussians


def make_groom_fixture(strand_count: int, samples: int, width: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = int(math.sqrt(strand_count))
    cols = int(math.ceil(strand_count / max(rows, 1)))
    roots = []
    for r in range(rows):
        for c in range(cols):
            if len(roots) >= strand_count:
                break
            x = -0.45 + 0.90 * (c / max(cols - 1, 1))
            y = -0.28 + 0.56 * (r / max(rows - 1, 1))
            roots.append((x, y))
    roots_t = torch.tensor(roots, dtype=torch.float32)
    t = torch.linspace(0.0, 1.0, samples, dtype=torch.float32).view(1, samples, 1)
    phase = (roots_t[:, [0]] * 5.0 + roots_t[:, [1]] * 3.0).view(-1, 1, 1)
    flow = torch.tensor([0.20, 0.62, 0.18], dtype=torch.float32).view(1, 1, 3)
    side = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).view(1, 1, 3)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).view(1, 1, 3)
    root_xyz = torch.cat(
        [
            roots_t[:, [0]],
            roots_t[:, [1]],
            torch.full((roots_t.shape[0], 1), 2.2),
        ],
        dim=-1,
    ).view(-1, 1, 3)
    curl = 0.055 * torch.sin(2.5 * math.pi * t + phase)
    droop = -0.06 * t.square()
    strands = root_xyz + 0.42 * t * flow + curl * side + droop * up

    widths = (float(width) * (1.0 - 0.68 * t)).expand(strands.shape[0], -1, -1).contiguous()
    colors = torch.ones((*strands.shape[:2], 3), dtype=torch.float32)
    stripe = ((torch.arange(strands.shape[0]).view(-1, 1, 1) % 5) == 0).float()
    colors = colors * (0.92 - 0.48 * stripe)
    colors[..., 1] *= 0.96
    colors[..., 2] *= 0.86
    opacities = torch.ones((*strands.shape[:2], 1), dtype=torch.float32) * 0.82
    return strands, widths, colors.clamp(0.0, 1.0), opacities


def load_npz(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(path)
    key = "strands" if "strands" in data else "points"
    if key not in data:
        raise ValueError("npz must contain strands or points")
    strands = torch.tensor(data[key], dtype=torch.float32)
    n, s, _ = strands.shape
    widths = torch.tensor(data["widths"], dtype=torch.float32) if "widths" in data else torch.full((n, s, 1), 0.003)
    colors = torch.tensor(data["colors"], dtype=torch.float32) if "colors" in data else torch.ones((n, s, 3)) * 0.9
    opacities = torch.tensor(data["opacities"], dtype=torch.float32) if "opacities" in data else torch.ones((n, s, 1)) * 0.8
    if widths.ndim == 2:
        widths = widths[..., None]
    if opacities.ndim == 2:
        opacities = opacities[..., None]
    return strands, widths, colors, opacities


def project(points: torch.Tensor, fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
    z = points[..., 2].clamp_min(1e-6)
    x = fx * points[..., 0] / z + cx
    y = fy * (-points[..., 1]) / z + cy
    return torch.stack([x, y], dim=-1)


def render_polyline_reference(
    strands: torch.Tensor,
    widths: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    image_width: int,
    image_height: int,
    fx: float,
    fy: float,
    background: tuple[int, int, int],
) -> Image.Image:
    img = Image.new("RGB", (image_width, image_height), background)
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = image_width * 0.5, image_height * 0.5
    xy = project(strands, fx, fy, cx, cy).detach().cpu().numpy()
    widths_px = (0.55 * fx * widths[..., 0] / strands[..., 2].clamp_min(1e-6)).detach().cpu().numpy()
    cols = colors.detach().cpu().numpy()
    alphas = opacities[..., 0].detach().cpu().numpy()
    for sid in range(strands.shape[0]):
        for i in range(strands.shape[1] - 1):
            p0 = tuple(xy[sid, i])
            p1 = tuple(xy[sid, i + 1])
            color = tuple(int(v * 255.0) for v in cols[sid, i].clip(0, 1))
            alpha = int(float(alphas[sid, i].clip(0, 1)) * 230)
            width = max(1, int(round(widths_px[sid, i])))
            draw.line([p0, p1], fill=(*color, alpha), width=width)
    return img


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    arr = image.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render strand reference vs gsplat strand-Gaussian output.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("D:/petsgaussianhair/_downloads/strand_gsplat_visual"))
    parser.add_argument("--strand-count", type=int, default=120)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--segments", type=int, default=0)
    parser.add_argument("--adaptive-segments", action="store_true")
    parser.add_argument("--min-segments", type=int, default=12)
    parser.add_argument("--max-segments", type=int, default=64)
    parser.add_argument("--hair-width", type=float, default=0.006)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--focal", type=float, default=1450.0)
    parser.add_argument("--length-overlap", type=float, default=1.18)
    parser.add_argument("--reference-image", type=Path, default=None, help="Optional external reference render, e.g. Blender groom render.")
    args = parser.parse_args()

    from gsplat.rendering import rasterization

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat visual validation")

    if args.input is None:
        strands, widths, colors, opacities = make_groom_fixture(args.strand_count, args.samples, args.hair_width)
        source = "built_in_groom_fixture"
    else:
        strands, widths, colors, opacities = load_npz(args.input)
        source = str(args.input)

    if args.adaptive_segments:
        lengths = torch.linalg.norm(strands[:, 1:, :] - strands[:, :-1, :], dim=-1).sum(dim=-1, keepdim=True)
        resampled = adaptive_resample_strands(
            strands,
            widths,
            colors,
            opacities,
            lengths,
            min_segments=args.min_segments,
            max_segments=args.max_segments,
        )
        segment_count = int(resampled.segment_counts.max().detach().cpu())
        segment_mode = "adaptive"
    else:
        segment_count = int(args.segments) if args.segments > 0 else int(strands.shape[1] - 1)
        counts = torch.full((strands.shape[0],), segment_count, dtype=torch.long)
        resampled = resample_strands_to_segment_budgets(strands, widths, colors, opacities, counts)
        segment_mode = "fixed"
    gaussians = strands_to_gaussians(
        resampled.strands,
        resampled.widths,
        resampled.colors,
        resampled.opacities,
        resampled.segment_mask,
        length_overlap=args.length_overlap,
    )

    device = torch.device("cuda")
    means = gaussians.means.to(device)
    quats = gaussians.quats.to(device)
    scales = gaussians.scales.to(device)
    opacity = gaussians.opacities.reshape(-1).to(device)
    color = gaussians.colors.to(device)
    viewmat = torch.eye(4, device=device).view(1, 4, 4)
    k = torch.tensor(
        [[args.focal, 0.0, args.image_width * 0.5], [0.0, args.focal, args.image_height * 0.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 3)
    bg_float = torch.tensor([[0.70, 0.72, 0.74]], device=device)
    render, alpha, _ = rasterization(
        means,
        quats,
        scales,
        opacity,
        color,
        viewmat,
        k,
        args.image_width,
        args.image_height,
        packed=False,
        backgrounds=bg_float,
        rasterize_mode="antialiased",
    )
    gsplat_img = tensor_image_to_pil(render[0])
    if args.reference_image is None:
        ref_img = render_polyline_reference(
            resampled.strands,
            resampled.widths,
            resampled.colors,
            resampled.opacities,
            args.image_width,
            args.image_height,
            args.focal,
            args.focal,
            background=(179, 184, 189),
        )
        ref_label = "Polyline groom reference"
    else:
        loaded = Image.open(args.reference_image).convert("RGBA").resize((args.image_width, args.image_height), Image.Resampling.LANCZOS)
        bg = Image.new("RGBA", loaded.size, (179, 184, 189, 255))
        ref_img = Image.alpha_composite(bg, loaded).convert("RGB")
        ref_label = "Blender groom reference"
    ref_arr = np.asarray(ref_img).astype(np.float32) / 255.0
    gs_arr = np.asarray(gsplat_img).astype(np.float32) / 255.0
    diff = np.abs(ref_arr - gs_arr)
    diff_img = Image.fromarray(np.clip(diff * 3.0 * 255.0, 0, 255).astype(np.uint8), mode="RGB")

    label_h = 34
    sheet = Image.new("RGB", (args.image_width * 3, args.image_height + label_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate([(ref_label, ref_img), ("AniGroom 3DGS from same exported strands", gsplat_img), ("3x absolute difference", diff_img)]):
        x = i * args.image_width
        sheet.paste(img, (x, label_h))
        draw.text((x + 16, 8), label, fill=(0, 0, 0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = args.output_dir / ("blender_reference_composited.png" if args.reference_image is not None else "polyline_reference.png")
    gs_path = args.output_dir / "anigroom_gsplat.png"
    diff_path = args.output_dir / "difference_3x.png"
    sheet_path = args.output_dir / "strand_gsplat_side_by_side.png"
    ref_img.save(ref_path)
    gsplat_img.save(gs_path)
    diff_img.save(diff_path)
    sheet.save(sheet_path)

    report = {
        "source": source,
        "strand_count": int(strands.shape[0]),
        "source_samples": int(strands.shape[1]),
        "segment_mode": segment_mode,
        "segment_count": int(segment_count),
        "segment_min": int(resampled.segment_counts.min().detach().cpu()),
        "segment_mean": float(resampled.segment_counts.float().mean().detach().cpu()),
        "segment_max": int(resampled.segment_counts.max().detach().cpu()),
        "gaussian_count": int(gaussians.means.shape[0]),
        "mean_abs_pixel_diff": float(diff.mean()),
        "alpha_mean": float(alpha.mean().detach().cpu()),
        "reference": str(ref_path),
        "gsplat": str(gs_path),
        "difference": str(diff_path),
        "side_by_side": str(sheet_path),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

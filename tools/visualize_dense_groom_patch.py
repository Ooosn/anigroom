from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import (  # noqa: E402
    GroomParameterField,
    GroomRanges,
    adaptive_resample_strands,
    build_strands,
    expand_child_strands,
    make_tangent_frames,
    strands_to_gaussians,
)


def inv_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def set_range(raw: torch.Tensor, value: torch.Tensor | float, bounds: tuple[float, float]) -> None:
    lo, hi = bounds
    v = torch.as_tensor(value, device=raw.device, dtype=raw.dtype)
    rel = (v - lo) / max(hi - lo, 1e-8)
    raw.copy_(inv_sigmoid(rel).expand_as(raw))


def set_color(raw: torch.Tensor, value: torch.Tensor) -> None:
    raw.copy_(inv_sigmoid(value).expand_as(raw))


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_dense_roots(device: torch.device, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(-0.70, 0.70, cols, device=device)
    ys = torch.linspace(-0.45, 0.45, rows, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    # Deterministic sub-grid jitter avoids a synthetic comb while keeping an even surface-root distribution.
    jx = 0.0065 * torch.sin(37.0 * gx + 19.0 * gy)
    jy = 0.0065 * torch.cos(23.0 * gx - 31.0 * gy)
    x = gx + jx
    y = gy + jy
    z = torch.full_like(x, 2.45)
    roots = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)
    normals = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3).expand_as(roots).contiguous()
    return roots, normals


def make_field(name: str, roots: torch.Tensor, ranges: GroomRanges) -> GroomParameterField:
    device = roots.device
    field = GroomParameterField(int(roots.shape[0]), ranges=ranges, device=device)
    x = roots[:, [0]]
    y = roots[:, [1]]
    stripe = (torch.sin(18.0 * x + 10.5 * y) > 0.36).float()
    soft_stripe = torch.sigmoid(7.0 * (torch.sin(18.0 * x + 10.5 * y) - 0.30))
    length_noise = 0.015 * torch.sin(16.0 * x - 11.0 * y)
    flow_angle = -0.35 + 0.35 * torch.sin(4.0 * y) + 0.18 * torch.sin(8.0 * x)
    flow_x = torch.cos(flow_angle)
    flow_y = torch.sin(flow_angle)

    white_root = torch.tensor([0.90, 0.86, 0.70], device=device).view(1, 3)
    white_tip = torch.tensor([1.00, 0.96, 0.78], device=device).view(1, 3)
    dark_root = torch.tensor([0.075, 0.055, 0.035], device=device).view(1, 3)
    dark_tip = torch.tensor([0.16, 0.12, 0.07], device=device).view(1, 3)
    root_color = white_root * (1.0 - soft_stripe) + dark_root * soft_stripe
    tip_color = white_tip * (1.0 - soft_stripe) + dark_tip * soft_stripe

    with torch.no_grad():
        set_range(field.length_raw, 0.118 + 0.018 * torch.sin(16.0 * x - 11.0 * y), ranges.length)
        set_range(field.root_width_raw, 0.00020 + 0.00006 * stripe, ranges.root_width)
        set_range(field.tip_width_ratio_raw, 0.060 + 0.020 * stripe, ranges.tip_width_ratio)
        set_range(field.width_taper_raw, 1.95, ranges.width_taper)
        field.flow_xy[:, 0:1].copy_(flow_x)
        field.flow_xy[:, 1:2].copy_(flow_y)
        set_range(field.flow_strength_raw, 0.96, ranges.flow_strength)
        set_range(field.lift_raw, 0.030, ranges.lift)
        set_range(field.sag_raw, 0.28, ranges.sag)
        set_range(field.stiffness_raw, 0.68, ranges.stiffness)
        set_range(field.curl_radius_raw, 0.00055, ranges.curl_radius)
        set_range(field.curl_frequency_raw, 0.45, ranges.curl_frequency)
        set_range(field.frizz_raw, 0.00075, ranges.frizz)
        set_range(field.child_radius_raw, 0.0032 + 0.0014 * soft_stripe, ranges.child_radius)
        set_range(field.clump_strength_raw, 0.42 + 0.18 * soft_stripe, ranges.clump_strength)
        set_color(field.root_color_raw, root_color)
        set_color(field.tip_color_raw, tip_color)
        set_range(field.opacity_raw, 0.82, ranges.opacity)
        set_range(field.tip_opacity_ratio_raw, 0.40, ranges.tip_opacity_ratio)

        if name == "longer":
            set_range(field.length_raw, 0.172 + 0.022 * torch.sin(10.0 * x), ranges.length)
            set_range(field.sag_raw, 0.42, ranges.sag)
            set_range(field.stiffness_raw, 0.50, ranges.stiffness)
            set_range(field.child_radius_raw, 0.0048 + 0.0018 * soft_stripe, ranges.child_radius)
        elif name == "taper":
            set_range(field.root_width_raw, 0.00042 + 0.00010 * stripe, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.030, ranges.tip_width_ratio)
            set_range(field.width_taper_raw, 2.35, ranges.width_taper)
        elif name == "curled":
            set_range(field.length_raw, 0.155 + 0.017 * torch.sin(12.0 * x), ranges.length)
            set_range(field.curl_radius_raw, 0.012 + 0.004 * soft_stripe, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 2.90 + 0.35 * torch.sin(7.0 * y), ranges.curl_frequency)
            field.curl_phase.copy_(8.0 * x + 3.0 * y)
            set_range(field.root_width_raw, 0.00018 + 0.00005 * stripe, ranges.root_width)
            set_range(field.child_radius_raw, 0.0048 + 0.0017 * soft_stripe, ranges.child_radius)
        elif name == "brushed_color":
            field.flow_xy[:, 0:1].copy_(torch.cos(flow_angle - 0.75))
            field.flow_xy[:, 1:2].copy_(torch.sin(flow_angle - 0.75))
            set_range(field.length_raw, 0.145 + 0.024 * torch.sin(8.0 * y), ranges.length)
            set_range(field.frizz_raw, 0.0050, ranges.frizz)
            root_color = torch.tensor([0.04, 0.035, 0.025], device=device).view(1, 3) * soft_stripe + white_root * (1.0 - soft_stripe)
            tip_color = torch.tensor([0.88, 0.76, 0.36], device=device).view(1, 3) * soft_stripe + white_tip * (1.0 - soft_stripe)
            set_color(field.root_color_raw, root_color)
            set_color(field.tip_color_raw, tip_color)
    return field


def render_field(
    field: GroomParameterField,
    roots: torch.Tensor,
    normals: torch.Tensor,
    width: int,
    height: int,
    focal: float,
    samples: int,
    min_segments: int,
    max_segments: int,
    child_count: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    from gsplat.rendering import rasterization

    tangents, bitangents = make_tangent_frames(normals)
    groom = field.decode()
    strands, widths, colors, opacities = build_strands(roots, normals, tangents, bitangents, groom, samples=samples)
    strands, widths, colors, opacities, root_ids = expand_child_strands(
        strands,
        widths,
        colors,
        opacities,
        normals,
        groom.child_radius,
        groom.clump_strength,
        child_count=child_count,
    )
    child_lengths = groom.length[root_ids]
    resampled = adaptive_resample_strands(strands, widths, colors, opacities, child_lengths, min_segments, max_segments)
    gaussians = strands_to_gaussians(
        resampled.strands,
        resampled.widths,
        resampled.colors,
        resampled.opacities,
        resampled.segment_mask,
        length_overlap=1.55,
    )

    viewmat = torch.eye(4, device=roots.device).view(1, 4, 4)
    k = torch.tensor(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        device=roots.device,
        dtype=roots.dtype,
    ).view(1, 3, 3)
    background = torch.tensor([[0.70, 0.72, 0.74]], device=roots.device, dtype=roots.dtype)
    render, alpha, _ = rasterization(
        gaussians.means,
        gaussians.quats,
        gaussians.scales,
        gaussians.opacities.reshape(-1),
        gaussians.colors,
        viewmat,
        k,
        width,
        height,
        packed=False,
        backgrounds=background,
        rasterize_mode="antialiased",
    )
    stats = {
        **resampled.stats,
        "guide_count": int(roots.shape[0]),
        "strand_count": int(strands.shape[0]),
        "gaussian_count": int(gaussians.means.shape[0]),
        "length_mean": float(groom.length.mean().detach().cpu()),
        "root_width_mean": float(groom.root_width.mean().detach().cpu()),
        "tip_width_mean": float(groom.tip_width.mean().detach().cpu()),
        "curl_radius_mean": float(groom.curl_radius.mean().detach().cpu()),
        "frizz_mean": float(groom.frizz.mean().detach().cpu()),
        "child_count": int(child_count),
        "child_radius_mean": float(groom.child_radius.mean().detach().cpu()),
        "clump_strength_mean": float(groom.clump_strength.mean().detach().cpu()),
        "alpha_mean": float(alpha.mean().detach().cpu()),
    }
    return render[0].clamp(0.0, 1.0), stats


def to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def make_sheet(paths: list[Path], labels: list[str], stats: dict[str, dict[str, float | int]], out_path: Path) -> None:
    font_title = load_font(28)
    font_small = load_font(18)
    tile_w, tile_h = 720, 430
    pad = 28
    header = 72
    cols = 2
    rows = math.ceil(len(paths) / cols)
    sheet = Image.new("RGB", (cols * (tile_w + pad) + pad, rows * (tile_h + header + pad) + pad), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(zip(paths, labels)):
        row, col = divmod(i, cols)
        x = pad + col * (tile_w + pad)
        y = pad + row * (tile_h + header + pad)
        draw.text((x, y), label, fill=(20, 20, 20), font=font_title)
        s = stats[label]
        desc = (
            f"{s['guide_count']} guides -> {s['strand_count']} strands / {s['gaussian_count']} G, "
            f"seg {s['adaptive_min_segments']}-{s['adaptive_max_segments']}, "
            f"len {s['length_mean']:.3f}, w {s['root_width_mean']:.4f}->{s['tip_width_mean']:.4f}"
        )
        draw.text((x, y + 36), desc, fill=(65, 65, 65), font=font_small)
        img = Image.open(path).convert("RGB")
        crop = img.crop((int(img.width * 0.12), int(img.height * 0.15), int(img.width * 0.88), int(img.height * 0.88)))
        crop.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_w, tile_h), (179, 184, 189))
        canvas.paste(crop, ((tile_w - crop.width) // 2, (tile_h - crop.height) // 2))
        sheet.paste(canvas, (x, y + header))
        draw.rectangle((x, y + header, x + tile_w, y + header + tile_h), outline=(205, 205, 205), width=2)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\petsgaussianhair\_downloads\dense_groom_patch_formal"))
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--child-count", type=int, default=4)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--focal", type=float, default=2300.0)
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--min-segments", type=int, default=18)
    parser.add_argument("--max-segments", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; dense groom validation must use gsplat")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranges = GroomRanges(
        length=(0.040, 0.220),
        root_width=(0.000035, 0.00075),
        tip_width_ratio=(0.012, 0.30),
        width_taper=(0.55, 3.20),
        lift=(0.000, 0.080),
        sag=(0.0, 0.75),
        stiffness=(0.05, 0.98),
        curl_radius=(0.0, 0.026),
        curl_frequency=(0.0, 5.5),
        frizz=(0.0, 0.010),
        child_radius=(0.0, 0.012),
        clump_strength=(0.0, 1.0),
        opacity=(0.05, 0.98),
        tip_opacity_ratio=(0.08, 0.90),
    )

    roots, normals = make_dense_roots(device, args.rows, args.cols)
    labels = ["dense_base", "longer", "taper", "curled", "brushed_color"]
    paths: list[Path] = []
    stats: dict[str, dict[str, float | int]] = {}
    for label in labels:
        field = make_field(label if label != "dense_base" else "base", roots, ranges)
        image, stat = render_field(
            field,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.min_segments,
            args.max_segments,
            args.child_count,
        )
        path = args.output_dir / f"{label}.png"
        to_pil(image).save(path)
        paths.append(path)
        stats[label] = stat

    report = {"stats": stats, "output_dir": str(args.output_dir)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_sheet(paths, labels, stats, args.output_dir / "dense_groom_patch_sheet.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

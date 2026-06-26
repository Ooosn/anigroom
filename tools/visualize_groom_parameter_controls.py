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
    make_tangent_frames,
    strands_to_gaussians,
)


def logit(values: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    values = values.clamp(eps, 1.0 - eps)
    return torch.log(values / (1.0 - values))


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_roots(device: torch.device, rows: int = 4, cols: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(-0.48, 0.48, cols, device=device)
    ys = torch.linspace(-0.30, 0.30, rows, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    roots = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.full((rows * cols,), 2.35, device=device)], dim=-1)
    normals = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 3).expand_as(roots).contiguous()
    return roots, normals


def field_with_pattern(name: str, root_count: int, roots: torch.Tensor, device: torch.device) -> GroomParameterField:
    ranges = GroomRanges(
        length=(0.055, 0.210),
        root_width=(0.00010, 0.00155),
        tip_width_ratio=(0.025, 0.42),
        curl_radius=(0.0, 0.045),
        frizz=(0.0, 0.020),
    )
    field = GroomParameterField(root_count, ranges=ranges, device=device)
    x = roots[:, [0]]
    y = roots[:, [1]]
    phase = 8.0 * x + 3.5 * y
    with torch.no_grad():
        field.length_raw.fill_(0.0)
        field.root_width_raw.fill_(-0.55)
        field.tip_width_ratio_raw.fill_(-1.15)
        field.opacity_raw.fill_(1.65)
        field.flow_xy[:, 0:1].fill_(0.05)
        field.flow_xy[:, 1:2].fill_(1.05)
        if name == "base":
            pass
        elif name == "long_bent":
            field.length_raw.add_(1.65)
            field.sag_raw.add_(2.2)
            field.bend_raw.add_(1.4 * torch.sin(5.5 * x))
            field.stiffness_raw.sub_(1.8)
            field.flow_xy[:, 0:1].add_(0.7)
        elif name == "root_tip_taper":
            field.root_width_raw.add_(1.35)
            field.tip_width_ratio_raw.sub_(2.15)
            field.width_taper_raw.add_(1.6)
            field.opacity_raw.add_(0.5)
        elif name == "curl":
            field.length_raw.add_(1.20)
            field.curl_radius_raw.add_(4.0)
            field.curl_frequency_raw.add_(2.6)
            field.curl_phase.copy_(phase)
            field.root_width_raw.add_(0.8)
            field.tip_width_ratio_raw.add_(0.1)
        elif name == "frizz":
            field.length_raw.add_(0.85)
            field.frizz_raw.add_(4.3)
            field.curl_radius_raw.add_(2.2)
            field.curl_frequency_raw.add_(1.8)
            field.curl_phase.copy_(1.7 * phase)
            field.stiffness_raw.sub_(2.2)
        elif name == "root_tip_color_alpha":
            root_color = torch.tensor([0.09, 0.07, 0.045], device=device).view(1, 3)
            tip_color = torch.tensor([1.00, 0.86, 0.45], device=device).view(1, 3)
            field.root_color_raw.copy_(logit(root_color).expand(root_count, -1))
            field.tip_color_raw.copy_(logit(tip_color).expand(root_count, -1))
            field.opacity_raw.add_(0.9)
            field.tip_opacity_ratio_raw.sub_(1.7)
            field.length_raw.add_(0.55)
        else:
            raise ValueError(f"unknown pattern: {name}")
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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    from gsplat.rendering import rasterization

    tangents, bitangents = make_tangent_frames(normals)
    groom = field.decode()
    strands, strand_widths, colors, opacities = build_strands(roots, normals, tangents, bitangents, groom, samples=samples)
    resampled = adaptive_resample_strands(
        strands,
        strand_widths,
        colors,
        opacities,
        groom.length,
        min_segments=min_segments,
        max_segments=max_segments,
    )
    gaussians = strands_to_gaussians(
        resampled.strands,
        resampled.widths,
        resampled.colors,
        resampled.opacities,
        resampled.segment_mask,
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
        "gaussian_count": int(gaussians.means.shape[0]),
        "strand_count": int(roots.shape[0]),
        "root_width_mean": float(groom.root_width.mean().detach().cpu()),
        "tip_width_mean": float(groom.tip_width.mean().detach().cpu()),
        "length_mean": float(groom.length.mean().detach().cpu()),
        "curl_radius_mean": float(groom.curl_radius.mean().detach().cpu()),
        "curl_frequency_mean": float(groom.curl_frequency.mean().detach().cpu()),
        "frizz_mean": float(groom.frizz.mean().detach().cpu()),
    }
    return render[0].clamp(0.0, 1.0), alpha[0].clamp(0.0, 1.0), stats


def to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def crop_focus(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((int(0.18 * w), int(0.18 * h), int(0.84 * w), int(0.84 * h)))


def make_sheet(image_paths: list[Path], labels: list[str], stats: dict[str, dict[str, float | int]], out_path: Path) -> None:
    font_title = load_font(30)
    font_small = load_font(19)
    cols = 3
    tile_w, tile_h = 620, 360
    pad = 24
    header = 70
    rows = math.ceil(len(image_paths) / cols)
    sheet = Image.new("RGB", (cols * (tile_w + pad) + pad, rows * (tile_h + header + pad) + pad), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        row, col = divmod(i, cols)
        x = pad + col * (tile_w + pad)
        y = pad + row * (tile_h + header + pad)
        draw.text((x, y), label, fill=(20, 20, 20), font=font_title)
        s = stats[label]
        desc = (
            f"{s['strand_count']} strands / {s['gaussian_count']} G, "
            f"seg {s['adaptive_min_segments']}-{s['adaptive_max_segments']}, "
            f"len {s['length_mean']:.3f}, w {s['root_width_mean']:.4f}->{s['tip_width_mean']:.4f}"
        )
        draw.text((x, y + 38), desc, fill=(80, 80, 80), font=font_small)
        img = crop_focus(Image.open(path).convert("RGB"))
        img.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_w, tile_h), (179, 184, 189))
        canvas.paste(img, ((tile_w - img.width) // 2, (tile_h - img.height) // 2))
        sheet.paste(canvas, (x, y + header))
        draw.rectangle((x, y + header, x + tile_w, y + header + tile_h), outline=(205, 205, 205), width=2)
    sheet.save(out_path)


def gradient_report(device: torch.device) -> dict[str, float]:
    roots, normals = make_roots(device, rows=5, cols=7)
    field = field_with_pattern("curl", int(roots.shape[0]), roots, device)
    image, alpha, _ = render_field(field, roots, normals, 480, 320, 760.0, 36, 12, 56)
    loss = ((image - 0.55) ** 2).mean() + 0.15 * alpha.mean()
    loss.backward()
    names = [
        "length_raw",
        "root_width_raw",
        "tip_width_ratio_raw",
        "width_taper_raw",
        "flow_xy",
        "lift_raw",
        "bend_raw",
        "sag_raw",
        "stiffness_raw",
        "curl_radius_raw",
        "curl_frequency_raw",
        "curl_phase",
        "frizz_raw",
        "root_color_raw",
        "tip_color_raw",
        "opacity_raw",
        "tip_opacity_ratio_raw",
    ]
    report: dict[str, float] = {}
    for name in names:
        param = getattr(field, name)
        if param.grad is None:
            report[name] = 0.0
        else:
            report[name] = float(param.grad.detach().abs().mean().cpu())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\petsgaussianhair\_downloads\groom_parameter_controls_formal"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--focal", type=float, default=2175.0)
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--min-segments", type=int, default=10)
    parser.add_argument("--max-segments", type=int, default=48)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this validation must use gsplat, not a fake renderer")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roots, normals = make_roots(device)
    labels = ["base", "long_bent", "root_tip_taper", "curl", "frizz", "root_tip_color_alpha"]
    image_paths: list[Path] = []
    stats: dict[str, dict[str, float | int]] = {}
    for label in labels:
        field = field_with_pattern(label, int(roots.shape[0]), roots, device)
        image, _, stat = render_field(
            field,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.min_segments,
            args.max_segments,
        )
        path = args.output_dir / f"{label}.png"
        to_pil(image).save(path)
        image_paths.append(path)
        stats[label] = stat

    grad = gradient_report(device)
    report = {"stats": stats, "gradient_abs_mean": grad}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_sheet(image_paths, labels, stats, args.output_dir / "groom_parameter_controls_sheet.png")
    print(json.dumps({"output_dir": str(args.output_dir), **report}, indent=2))


if __name__ == "__main__":
    main()

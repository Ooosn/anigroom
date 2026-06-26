from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import (  # noqa: E402
    GroomParameterField,
    GroomRanges,
    expand_child_strands,
    make_tangent_frames,
    resample_strands_to_segment_budgets,
    strand_segment_budgets,
    build_strands,
    strands_to_gaussians,
)


@dataclass
class RenderPack:
    image: torch.Tensor
    alpha: torch.Tensor
    orientation: torch.Tensor
    orientation_conf: torch.Tensor
    stats: dict[str, float | int]


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


def make_roots(device: torch.device, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(-0.68, 0.68, cols, device=device)
    ys = torch.linspace(-0.42, 0.42, rows, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    x = gx + 0.007 * torch.sin(29.0 * gx + 17.0 * gy)
    y = gy + 0.007 * torch.cos(31.0 * gx - 13.0 * gy)
    roots = torch.stack([x.reshape(-1), y.reshape(-1), torch.full((rows * cols,), 2.45, device=device)], dim=-1)
    normals = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3).expand_as(roots).contiguous()
    return roots, normals


def make_ranges() -> GroomRanges:
    return GroomRanges(
        length=(0.030, 0.165),
        root_width=(0.00008, 0.00120),
        tip_width_ratio=(0.030, 0.45),
        width_taper=(0.55, 3.20),
        lift=(0.000, 0.100),
        sag=(0.0, 0.75),
        stiffness=(0.05, 0.98),
        curl_radius=(0.0, 0.030),
        curl_frequency=(0.0, 5.5),
        frizz=(0.0, 0.014),
        child_radius=(0.0, 0.014),
        clump_strength=(0.0, 1.0),
        opacity=(0.05, 0.98),
        tip_opacity_ratio=(0.08, 0.90),
    )


def make_teacher(root_count: int, roots: torch.Tensor, ranges: GroomRanges) -> GroomParameterField:
    device = roots.device
    field = GroomParameterField(root_count, ranges=ranges, device=device)
    x = roots[:, [0]]
    y = roots[:, [1]]
    stripe = torch.sigmoid(8.0 * (torch.sin(17.0 * x + 9.0 * y) - 0.30))
    flow_angle = -0.45 + 0.22 * torch.sin(5.0 * y) + 0.18 * torch.cos(7.0 * x)
    with torch.no_grad():
        set_range(field.length_raw, 0.082 + 0.018 * torch.sin(7.0 * x - 5.0 * y), ranges.length)
        set_range(field.root_width_raw, 0.00050 + 0.00020 * stripe, ranges.root_width)
        set_range(field.tip_width_ratio_raw, 0.12 + 0.04 * stripe, ranges.tip_width_ratio)
        set_range(field.width_taper_raw, 1.85, ranges.width_taper)
        field.flow_xy[:, 0:1].copy_(torch.cos(flow_angle))
        field.flow_xy[:, 1:2].copy_(torch.sin(flow_angle))
        set_range(field.flow_strength_raw, 0.98, ranges.flow_strength)
        set_range(field.lift_raw, 0.050, ranges.lift)
        set_range(field.sag_raw, 0.24, ranges.sag)
        set_range(field.stiffness_raw, 0.70, ranges.stiffness)
        set_range(field.curl_radius_raw, 0.004 + 0.006 * stripe, ranges.curl_radius)
        set_range(field.curl_frequency_raw, 0.95 + 0.85 * stripe, ranges.curl_frequency)
        field.curl_phase.copy_(7.0 * x + 4.0 * y)
        set_range(field.frizz_raw, 0.0025 + 0.0020 * stripe, ranges.frizz)
        set_range(field.child_radius_raw, 0.0045 + 0.0025 * stripe, ranges.child_radius)
        set_range(field.clump_strength_raw, 0.52 + 0.20 * stripe, ranges.clump_strength)
        root_color = torch.tensor([0.96, 0.90, 0.66], device=device).view(1, 3) * (1.0 - stripe) + torch.tensor(
            [0.055, 0.040, 0.025], device=device
        ).view(1, 3) * stripe
        tip_color = torch.tensor([1.00, 0.95, 0.76], device=device).view(1, 3) * (1.0 - stripe) + torch.tensor(
            [0.14, 0.10, 0.055], device=device
        ).view(1, 3) * stripe
        set_color(field.root_color_raw, root_color)
        set_color(field.tip_color_raw, tip_color)
        set_range(field.opacity_raw, 0.82, ranges.opacity)
        set_range(field.tip_opacity_ratio_raw, 0.46, ranges.tip_opacity_ratio)
    return field


def build_render_inputs(
    field: GroomParameterField,
    roots: torch.Tensor,
    normals: torch.Tensor,
    samples: int,
    child_count: int,
    segment_counts: torch.Tensor | None,
    min_segments: int,
    max_segments: int,
) -> tuple[object, torch.Tensor, dict[str, float | int]]:
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
    if segment_counts is None:
        counts, stats = strand_segment_budgets(strands.detach(), child_lengths.detach(), min_segments, max_segments)
    else:
        counts = segment_counts.to(device=roots.device)
        stats = {
            "adaptive_mean_segments": float(counts.float().mean().detach().cpu()),
            "adaptive_min_segments": int(counts.min().detach().cpu()),
            "adaptive_max_segments": int(counts.max().detach().cpu()),
        }
    resampled = resample_strands_to_segment_budgets(strands, widths, colors, opacities, counts)
    stats.update(resampled.stats)
    gaussians = strands_to_gaussians(
        resampled.strands,
        resampled.widths,
        resampled.colors,
        resampled.opacities,
        resampled.segment_mask,
        strand_root_indices=root_ids,
        length_overlap=1.45,
    )
    stats.update(
        {
            "guide_count": int(roots.shape[0]),
            "strand_count": int(strands.shape[0]),
            "gaussian_count": int(gaussians.means.shape[0]),
            "length_mean": float(groom.length.mean().detach().cpu()),
            "child_radius_mean": float(groom.child_radius.mean().detach().cpu()),
            "clump_strength_mean": float(groom.clump_strength.mean().detach().cpu()),
            "curl_radius_mean": float(groom.curl_radius.mean().detach().cpu()),
        }
    )
    return gaussians, counts.detach(), stats


def gaussian_screen_orientation_colors(gaussians: object, focal: float) -> torch.Tensor:
    """Encode projected strand tangent orientation as double-angle colors."""

    means = gaussians.means
    directions = gaussians.directions
    z = means[:, 2:3].clamp_min(1e-4)
    du = float(focal) * (directions[:, 0:1] * z - means[:, 0:1] * directions[:, 2:3]) / z.square()
    dv = float(focal) * (directions[:, 1:2] * z - means[:, 1:2] * directions[:, 2:3]) / z.square()
    screen_dir = F.normalize(torch.cat([du, dv], dim=-1), dim=-1, eps=1e-8)
    cos2 = screen_dir[:, 0:1].square() - screen_dir[:, 1:2].square()
    sin2 = 2.0 * screen_dir[:, 0:1] * screen_dir[:, 1:2]
    return torch.cat([0.5 + 0.5 * cos2, 0.5 + 0.5 * sin2, torch.zeros_like(cos2)], dim=-1)


def decode_orientation_render(flow_image: torch.Tensor, flow_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = flow_alpha[0].clamp(0.0, 1.0)
    avg = flow_image[0][..., :2] / alpha.clamp_min(1e-4)
    orientation = F.normalize(2.0 * avg - 1.0, dim=-1, eps=1e-8)
    return orientation, alpha


def render(
    field: GroomParameterField,
    roots: torch.Tensor,
    normals: torch.Tensor,
    width: int,
    height: int,
    focal: float,
    samples: int,
    child_count: int,
    segment_counts: torch.Tensor | None,
    min_segments: int,
    max_segments: int,
) -> RenderPack:
    from gsplat.rendering import rasterization

    gaussians, counts, stats = build_render_inputs(
        field,
        roots,
        normals,
        samples,
        child_count,
        segment_counts,
        min_segments,
        max_segments,
    )
    viewmat = torch.eye(4, device=roots.device).view(1, 4, 4)
    k = torch.tensor(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        device=roots.device,
        dtype=roots.dtype,
    ).view(1, 3, 3)
    background = torch.tensor([[0.70, 0.72, 0.74]], device=roots.device, dtype=roots.dtype)
    image, alpha, _ = rasterization(
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
    orient_colors = gaussian_screen_orientation_colors(gaussians, focal)
    orient_image, orient_alpha, _ = rasterization(
        gaussians.means,
        gaussians.quats,
        gaussians.scales,
        gaussians.opacities.reshape(-1),
        orient_colors,
        viewmat,
        k,
        width,
        height,
        packed=False,
        backgrounds=torch.zeros_like(background),
        rasterize_mode="antialiased",
    )
    orientation, orientation_conf = decode_orientation_render(orient_image, orient_alpha)
    stats["segment_count_shape"] = int(counts.shape[0])
    return RenderPack(
        image=image[0].clamp(0.0, 1.0),
        alpha=alpha[0].clamp(0.0, 1.0),
        orientation=orientation,
        orientation_conf=orientation_conf,
        stats=stats,
    )


def orientation_flow_loss(pred: RenderPack, target: RenderPack) -> torch.Tensor:
    target_conf = target.orientation_conf.detach().clamp(0.0, 1.0)
    pred_visible = (pred.orientation_conf.detach() > 0.02).to(dtype=target_conf.dtype)
    weight = target_conf * pred_visible
    denom = weight.sum().clamp_min(1.0)
    dot = (pred.orientation * target.orientation.detach()).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    return ((1.0 - dot) * weight).sum() / denom


def orientation_detail_loss(pred: RenderPack, target: RenderPack) -> torch.Tensor:
    target_conf = target.orientation_conf.detach().clamp(0.0, 1.0)
    pred_visible = (pred.orientation_conf.detach() > 0.02).to(dtype=target_conf.dtype)
    weight = target_conf * pred_visible
    p = pred.orientation
    t = target.orientation.detach()

    dx_weight = torch.minimum(weight[:, 1:], weight[:, :-1])
    dy_weight = torch.minimum(weight[1:, :], weight[:-1, :])
    dx = (p[:, 1:] - p[:, :-1]) - (t[:, 1:] - t[:, :-1])
    dy = (p[1:, :] - p[:-1, :]) - (t[1:, :] - t[:-1, :])
    dx_loss = (dx.abs() * dx_weight).sum() / dx_weight.sum().clamp_min(1.0)
    dy_loss = (dy.abs() * dy_weight).sum() / dy_weight.sum().clamp_min(1.0)
    return 0.5 * (dx_loss + dy_loss)


def loss_components(
    pred: RenderPack,
    target: RenderPack,
    orientation_weight: float,
    orientation_detail_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    rgb_l1 = (pred.image - target.image).abs().mean()
    rgb_mse = F.mse_loss(pred.image, target.image)
    alpha_l1 = (pred.alpha - target.alpha).abs().mean()
    orient = orientation_flow_loss(pred, target)
    orient_detail = orientation_detail_loss(pred, target)
    total = (
        rgb_l1
        + 0.35 * rgb_mse
        + 0.20 * alpha_l1
        + float(orientation_weight) * orient
        + float(orientation_detail_weight) * orient_detail
    )
    return {
        "total": total,
        "rgb_l1": rgb_l1,
        "rgb_mse": rgb_mse,
        "alpha_l1": alpha_l1,
        "orientation": orient,
        "orientation_detail": orient_detail,
    }


def image_loss(
    pred: RenderPack,
    target: RenderPack,
    orientation_weight: float = 0.0,
    orientation_detail_weight: float = 0.0,
) -> torch.Tensor:
    return loss_components(pred, target, orientation_weight, orientation_detail_weight)["total"]


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).detach().clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).cpu())


def to_pil(image: torch.Tensor) -> Image.Image:
    arr = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def orientation_to_pil(orientation: torch.Tensor, confidence: torch.Tensor) -> Image.Image:
    vec = orientation.detach().clamp(-1.0, 1.0).cpu()
    conf = confidence.detach().clamp(0.0, 1.0).cpu()
    rgb = torch.zeros((*vec.shape[:2], 3), dtype=torch.float32)
    rgb[..., 0:2] = 0.5 + 0.5 * vec
    rgb[..., 2:3] = 0.25 + 0.75 * conf
    rgb = rgb * conf + (1.0 - conf) * 0.72
    arr = (rgb.clamp(0.0, 1.0).numpy() * 255.0 + 0.5).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


def make_sheet(paths: list[Path], labels: list[str], out_path: Path) -> None:
    font_title = load_font(28)
    tile_w, tile_h = 620, 390
    pad = 28
    header = 46
    sheet = Image.new("RGB", (len(paths) * (tile_w + pad) + pad, tile_h + header + 2 * pad), (248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(zip(paths, labels)):
        x = pad + i * (tile_w + pad)
        y = pad
        draw.text((x, y), label, fill=(20, 20, 20), font=font_title)
        img = Image.open(path).convert("RGB")
        crop = img.crop((int(img.width * 0.10), int(img.height * 0.14), int(img.width * 0.90), int(img.height * 0.88)))
        crop.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile_w, tile_h), (179, 184, 189))
        canvas.paste(crop, ((tile_w - crop.width) // 2, (tile_h - crop.height) // 2))
        sheet.paste(canvas, (x, y + header))
        draw.rectangle((x, y + header, x + tile_w, y + header + tile_h), outline=(205, 205, 205), width=2)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\petsgaussianhair\_downloads\groom_layer_teacher_student"))
    parser.add_argument("--rows", type=int, default=18)
    parser.add_argument("--cols", type=int, default=28)
    parser.add_argument("--child-count", type=int, default=3)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--focal", type=float, default=1320.0)
    parser.add_argument("--samples", type=int, default=56)
    parser.add_argument("--min-segments", type=int, default=16)
    parser.add_argument("--max-segments", type=int, default=56)
    parser.add_argument("--iterations", type=int, default=220)
    parser.add_argument("--segment-warmup", type=int, default=60)
    parser.add_argument("--segment-refresh", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--orientation-weight", type=float, default=0.20)
    parser.add_argument("--orientation-detail-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this training validation must use gsplat")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranges = make_ranges()
    roots, normals = make_roots(device, args.rows, args.cols)
    teacher = make_teacher(int(roots.shape[0]), roots, ranges).eval()
    student = GroomParameterField(int(roots.shape[0]), ranges=ranges, device=device)
    with torch.no_grad():
        set_range(student.length_raw, 0.064, ranges.length)
        set_range(student.root_width_raw, 0.00042, ranges.root_width)
        set_range(student.tip_width_ratio_raw, 0.11, ranges.tip_width_ratio)
        set_range(student.opacity_raw, 0.74, ranges.opacity)
        set_range(student.child_radius_raw, 0.0025, ranges.child_radius)
        set_range(student.clump_strength_raw, 0.30, ranges.clump_strength)

    with torch.no_grad():
        target = render(
            teacher,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            None,
            args.min_segments,
            args.max_segments,
        )
        initial = render(
            student,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            None,
            args.min_segments,
            args.max_segments,
        )

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    segment_counts: torch.Tensor | None = None
    history: list[dict[str, float | int]] = []
    for iteration in range(1, args.iterations + 1):
        if iteration == 1:
            segment_counts = None
        elif iteration >= args.segment_warmup and (iteration - args.segment_warmup) % args.segment_refresh == 0:
            with torch.no_grad():
                _, segment_counts, _ = build_render_inputs(
                    student,
                    roots,
                    normals,
                    args.samples,
                    args.child_count,
                    None,
                    args.min_segments,
                    args.max_segments,
                )

        optimizer.zero_grad(set_to_none=True)
        pred = render(
            student,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            segment_counts,
            args.min_segments,
            args.max_segments,
        )
        components = loss_components(pred, target, args.orientation_weight, args.orientation_detail_weight)
        loss = components["total"]
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % 20 == 0 or iteration == args.iterations:
            history.append(
                {
                    "iter": iteration,
                    "loss": float(loss.detach().cpu()),
                    "rgb_l1": float(components["rgb_l1"].detach().cpu()),
                    "rgb_mse": float(components["rgb_mse"].detach().cpu()),
                    "alpha_l1": float(components["alpha_l1"].detach().cpu()),
                    "orientation_loss": float(components["orientation"].detach().cpu()),
                    "orientation_detail_loss": float(components["orientation_detail"].detach().cpu()),
                    "psnr": psnr(pred.image, target.image),
                    "gaussian_count": int(pred.stats["gaussian_count"]),
                    "segment_min": int(pred.stats["adaptive_min_segments"]),
                    "segment_max": int(pred.stats["adaptive_max_segments"]),
                }
            )
            print(json.dumps(history[-1]), flush=True)

    with torch.no_grad():
        final = render(
            student,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            segment_counts,
            args.min_segments,
            args.max_segments,
        )
        diff = (final.image - target.image).abs().mul(3.0).clamp(0.0, 1.0)

    paths = [
        args.output_dir / "target.png",
        args.output_dir / "initial.png",
        args.output_dir / "final.png",
        args.output_dir / "diff_x3.png",
    ]
    for path, image in zip(paths, [target.image, initial.image, final.image, diff]):
        to_pil(image).save(path)
    make_sheet(paths, ["teacher target", "student initial", "student final", "final error x3"], args.output_dir / "training_sheet.png")

    orient_paths = [
        args.output_dir / "target_orientation.png",
        args.output_dir / "initial_orientation.png",
        args.output_dir / "final_orientation.png",
    ]
    orientation_to_pil(target.orientation, target.orientation_conf).save(orient_paths[0])
    orientation_to_pil(initial.orientation, initial.orientation_conf).save(orient_paths[1])
    orientation_to_pil(final.orientation, final.orientation_conf).save(orient_paths[2])
    make_sheet(orient_paths, ["teacher orientation", "initial orientation", "final orientation"], args.output_dir / "orientation_sheet.png")

    grad_report: dict[str, float] = {}
    for name, param in student.named_parameters():
        if param.grad is not None:
            grad_report[name] = float(param.grad.detach().abs().mean().cpu())

    report = {
        "target_stats": target.stats,
        "initial_stats": initial.stats,
        "final_stats": final.stats,
        "history": history,
        "initial_psnr": psnr(initial.image, target.image),
        "final_psnr": psnr(final.image, target.image),
        "initial_loss": float(image_loss(initial, target, args.orientation_weight, args.orientation_detail_weight).detach().cpu()),
        "final_loss": float(image_loss(final, target, args.orientation_weight, args.orientation_detail_weight).detach().cpu()),
        "initial_orientation_loss": float(orientation_flow_loss(initial, target).detach().cpu()),
        "final_orientation_loss": float(orientation_flow_loss(final, target).detach().cpu()),
        "initial_orientation_detail_loss": float(orientation_detail_loss(initial, target).detach().cpu()),
        "final_orientation_detail_loss": float(orientation_detail_loss(final, target).detach().cpu()),
        "orientation_weight": float(args.orientation_weight),
        "orientation_detail_weight": float(args.orientation_detail_weight),
        "gradient_abs_mean_last_iter": grad_report,
        "segment_schedule": {
            "warmup": args.segment_warmup,
            "refresh": args.segment_refresh,
            "detached_counts": True,
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "initial_psnr": report["initial_psnr"], "final_psnr": report["final_psnr"]}, indent=2))


if __name__ == "__main__":
    main()

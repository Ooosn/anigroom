from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_white_tiger_stage1 as stage1  # noqa: E402
from anigroom.flow.clean_flow import groom_direction_3d  # noqa: E402


def _tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    arr = image.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _read_base(path: Path, width: int, height: int) -> Image.Image:
    if path.exists():
        img = Image.open(path).convert("RGB")
        if img.size == (width, height):
            return img
        img.thumbnail((width, height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (width, height), (255, 255, 255))
        tile.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
        return tile
    return Image.new("RGB", (width, height), (255, 255, 255))


def _turbo(x: np.ndarray) -> np.ndarray:
    """Small polynomial Turbo approximation, returns RGB in [0, 1]."""

    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    r = 0.13572138 + 4.61539260 * x - 42.66032258 * x**2 + 132.13108234 * x**3 - 152.94239396 * x**4 + 59.28637943 * x**5
    g = 0.09140261 + 2.19418839 * x + 4.84296658 * x**2 - 14.18503333 * x**3 + 4.27729857 * x**4 + 2.82956604 * x**5
    b = 0.10667330 + 12.64194608 * x - 60.58204836 * x**2 + 110.36276771 * x**3 - 89.90310912 * x**4 + 27.34824973 * x**5
    return np.stack([np.clip(r, 0.0, 1.0), np.clip(g, 0.0, 1.0), np.clip(b, 0.0, 1.0)], axis=-1)


def _signed_colormap(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), -1.0, 1.0)
    neg = np.stack([0.12 + 0.35 * (x + 1.0), 0.30 + 0.50 * (x + 1.0), np.ones_like(x)], axis=-1)
    pos = np.stack([np.ones_like(x), 0.25 + 0.55 * (1.0 - x), 0.18 + 0.35 * (1.0 - x)], axis=-1)
    neutral = np.ones((*x.shape, 3), dtype=np.float32) * 0.92
    return np.where(x[..., None] < -0.02, neg, np.where(x[..., None] > 0.02, pos, neutral))


def _normalize_values(values: np.ndarray, lo: float | None = None, hi: float | None = None) -> tuple[np.ndarray, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32), 0.0, 1.0
    if lo is None:
        lo = float(np.quantile(finite, 0.02))
    if hi is None:
        hi = float(np.quantile(finite, 0.98))
    if abs(hi - lo) < 1.0e-12:
        hi = lo + 1.0
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32), float(lo), float(hi)


def summarize_attribute_values(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("attribute statistics require at least one finite value")
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "p98": float(np.quantile(finite, 0.98)),
        "max": float(np.max(finite)),
    }


EFFECTIVE_ROOT_OPACITY_ATTRIBUTE_NAMES = (
    "root_opacity",
    "tip_opacity",
    "tip_opacity_ratio",
)
EFFECTIVE_OPACITY_RATIO_EPS = 1.0e-8


def effective_root_opacity_attributes(
    root_opacity: np.ndarray,
    tip_opacity: np.ndarray,
    *,
    eps: float = EFFECTIVE_OPACITY_RATIO_EPS,
) -> dict[str, np.ndarray]:
    """Return validated effective root opacity attributes for visualization."""

    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"opacity ratio eps must be finite and positive, got {eps}")
    root = np.asarray(root_opacity, dtype=np.float64).reshape(-1)
    tip = np.asarray(tip_opacity, dtype=np.float64).reshape(-1)
    if root.shape != tip.shape:
        raise ValueError(
            "root_opacity and tip_opacity must have matching shapes: "
            f"{root.shape} != {tip.shape}"
        )
    if not np.isfinite(root).all() or not np.isfinite(tip).all():
        raise ValueError("root_opacity and tip_opacity must be finite")
    if np.any((root < 0.0) | (root > 1.0)):
        raise ValueError("root_opacity must lie in [0, 1]")
    if np.any((tip < 0.0) | (tip > 1.0)):
        raise ValueError("tip_opacity must lie in [0, 1]")
    if np.any(tip > root):
        raise ValueError("tip_opacity must not exceed root_opacity")
    ratio = tip / np.maximum(root, float(eps))
    if not np.isfinite(ratio).all() or np.any((ratio < 0.0) | (ratio > 1.0)):
        raise ValueError("tip_opacity_ratio must be finite and lie in [0, 1]")
    return {
        "root_opacity": root.astype(np.float32),
        "tip_opacity": tip.astype(np.float32),
        "tip_opacity_ratio": ratio.astype(np.float32),
    }


def effective_root_attribute_map_paths(
    output_dir: Path,
    view: int,
) -> dict[str, Path]:
    return {
        name: output_dir / f"view{int(view):02d}_{name}.png"
        for name in EFFECTIVE_ROOT_OPACITY_ATTRIBUTE_NAMES
    }


def project_primary_guide_curl_turns(
    guide_points_local: torch.Tensor,
    guide_curl_turns: torch.Tensor,
    *,
    log_scale: torch.Tensor,
    translation: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    mesh_depth: torch.Tensor | None = None,
    mesh_depth_kernel: int = 1,
    mesh_depth_abs_tolerance: float = 0.0,
    mesh_depth_rel_tolerance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project primary-guide points and apply the root map visibility rules."""

    if guide_points_local.ndim != 2 or guide_points_local.shape[-1] != 3:
        raise ValueError(
            "guide_points_local must have shape [guide_count, 3], got "
            f"{tuple(guide_points_local.shape)}"
        )
    points_world = (
        guide_points_local * torch.exp(log_scale).reshape(1, 1)
        + translation.reshape(1, 3)
    )
    values = guide_curl_turns.reshape(-1)
    if values.shape != (points_world.shape[0],):
        raise ValueError(
            "guide_curl_turns must have one value per guide: "
            f"{tuple(values.shape)} != {(int(points_world.shape[0]),)}"
        )
    xy, depth = stage1.project_points(points_world, viewmat, k)
    visible = (
        (depth > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < int(width))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < int(height))
    )
    if mesh_depth is not None:
        sampled_mesh_depth = stage1.sample_depth_nearest(
            mesh_depth,
            xy,
            kernel_size=int(mesh_depth_kernel),
        )
        tolerance = float(mesh_depth_abs_tolerance) + depth.abs() * float(
            mesh_depth_rel_tolerance
        )
        visible = visible & torch.isfinite(sampled_mesh_depth) & (
            depth <= sampled_mesh_depth + tolerance
        )
    return points_world, xy, values, visible


def _overlay_points(
    base: Image.Image,
    xy: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    out_path: Path,
    radius: int = 2,
    signed: bool = False,
    lo: float | None = None,
    hi: float | None = None,
) -> dict[str, float]:
    width, height = base.size
    finite = np.isfinite(values)
    xy = xy[finite]
    values = values[finite]
    in_frame = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    xy = xy[in_frame]
    values = values[in_frame]
    if signed:
        max_abs = max(float(np.quantile(np.abs(values), 0.98)) if values.size else 1.0, 1.0e-6)
        colors = _signed_colormap(np.clip(values / max_abs, -1.0, 1.0))
        value_lo, value_hi = -max_abs, max_abs
    else:
        normed, value_lo, value_hi = _normalize_values(values, lo=lo, hi=hi)
        colors = _turbo(normed)

    canvas = np.asarray(base).astype(np.float32) / 255.0
    overlay = canvas.copy()
    alpha = np.zeros((height, width), dtype=np.float32)
    xi = np.round(xy[:, 0]).astype(np.int32)
    yi = np.round(xy[:, 1]).astype(np.int32)
    colors = colors.astype(np.float32)
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            if ox * ox + oy * oy > radius * radius:
                continue
            xx = xi + ox
            yy = yi + oy
            valid = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            if not np.any(valid):
                continue
            overlay[yy[valid], xx[valid]] = colors[valid]
            alpha[yy[valid], xx[valid]] = np.maximum(alpha[yy[valid], xx[valid]], 0.78)
    out = canvas * (1.0 - alpha[..., None]) + overlay * alpha[..., None]
    img = Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    font_title, font_small = _fonts()
    draw.rectangle((0, 0, width, 78), fill=(255, 255, 255))
    draw.text((18, 10), title, fill=(20, 20, 20), font=font_title)
    draw.text((18, 47), f"visible roots: {len(values):,}    range: {value_lo:.5g} .. {value_hi:.5g}", fill=(65, 65, 65), font=font_small)
    _draw_colorbar(draw, width - 430, 22, 360, 26, signed=signed, lo=value_lo, hi=value_hi, font=font_small)
    img.save(out_path)
    return {"visible_roots": int(len(values)), "lo": float(value_lo), "hi": float(value_hi)}


def _fonts():
    try:
        return (
            ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 30),
            ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 19),
        )
    except Exception:
        font = ImageFont.load_default()
        return font, font


def _draw_colorbar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, *, signed: bool, lo: float, hi: float, font) -> None:
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    if signed:
        colors = _signed_colormap(xs * 2.0 - 1.0)
    else:
        colors = _turbo(xs)
    for i, c in enumerate(colors):
        rgb = tuple((c * 255.0 + 0.5).astype(np.uint8).tolist())
        draw.line((x + i, y, x + i, y + h), fill=rgb)
    draw.rectangle((x, y, x + w, y + h), outline=(20, 20, 20), width=1)
    draw.text((x, y + h + 2), f"{lo:.4g}", fill=(30, 30, 30), font=font)
    draw.text((x + w - 72, y + h + 2), f"{hi:.4g}", fill=(30, 30, 30), font=font)


def _save_flow_arrows(
    base: Image.Image,
    xy: np.ndarray,
    xy2: np.ndarray,
    values: np.ndarray,
    out_path: Path,
    *,
    title: str,
    stride: int = 34,
) -> None:
    width, height = base.size
    canvas = base.convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    font_title, font_small = _fonts()
    draw.rectangle((0, 0, width, 78), fill=(255, 255, 255, 245))
    draw.text((18, 10), title, fill=(20, 20, 20), font=font_title)
    draw.text((18, 47), "arrows are projected 3D hair direction at visible roots; color follows length", fill=(65, 65, 65), font=font_small)

    finite = np.isfinite(xy).all(axis=1) & np.isfinite(xy2).all(axis=1)
    xy = xy[finite]
    xy2 = xy2[finite]
    values = values[finite]
    in_frame = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    xy = xy[in_frame]
    xy2 = xy2[in_frame]
    values = values[in_frame]
    normed, _, _ = _normalize_values(values)
    colors = _turbo(normed)
    bins: dict[tuple[int, int], int] = {}
    for i, p in enumerate(xy):
        key = (int(p[0] // stride), int(p[1] // stride))
        if key not in bins:
            bins[key] = i
    for i in bins.values():
        x0, y0 = xy[i]
        vec = xy2[i] - xy[i]
        mag = float(np.linalg.norm(vec))
        if mag < 2.0:
            continue
        vec = vec / mag * min(max(mag, 14.0), 30.0)
        x1, y1 = x0 + vec[0], y0 + vec[1]
        rgb = tuple((colors[i] * 255.0 + 0.5).astype(np.uint8).tolist()) + (230,)
        draw.line((float(x0), float(y0), float(x1), float(y1)), fill=rgb, width=2)
        angle = math.atan2(vec[1], vec[0])
        ah = 5.0
        for da in (2.55, -2.55):
            xh = x1 + ah * math.cos(angle + da)
            yh = y1 + ah * math.sin(angle + da)
            draw.line((float(x1), float(y1), float(xh), float(yh)), fill=rgb, width=2)
    canvas.convert("RGB").save(out_path)


def _make_contact_sheet(paths: list[tuple[str, Path]], out_path: Path) -> None:
    cols = 3
    thumb_w, thumb_h = 960, 540
    label_h = 52
    pad = 24
    top_h = 92
    rows = math.ceil(len(paths) / cols)
    width = cols * thumb_w + (cols + 1) * pad
    height = top_h + rows * (thumb_h + label_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (246, 246, 244))
    draw = ImageDraw.Draw(sheet)
    font_title, font_small = _fonts()
    draw.text((pad, 18), "White tiger groom attributes: 3D direction, length, brush curve, curl", fill=(20, 20, 20), font=font_title)
    draw.text((pad, 54), "Generated from checkpoint root parameters, projected onto view09 with mesh-depth visibility.", fill=(70, 70, 70), font=font_small)
    for idx, (label, path) in enumerate(paths):
        r, c = divmod(idx, cols)
        x = pad + c * (thumb_w + pad)
        y = top_h + r * (thumb_h + label_h + pad)
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, thumb_h), (255, 255, 255))
        tile.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline=(195, 195, 195), width=1)
        draw.text((x + 10, y + thumb_h + 10), label, fill=(25, 25, 25), font=font_small)
    sheet.save(out_path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize effective groom attributes from a white tiger Stage1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--view", type=int, default=9)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-image", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, _ = stage1.load_stage1_checkpoint_model(checkpoint_path, device)
    data_root = stage1.resolve_project_path(config.data_root)
    viewmats, ks = stage1.load_camera_tensors(data_root, device)
    width, height = int(config.expected_width), int(config.expected_height)
    viewmat, k = viewmats[int(args.view)], ks[int(args.view)]

    roots, normals, roots_local = model.roots_and_normals()
    groom = model.apply_guide_controls(model.groom.decode(), roots_local)
    tangents, bitangents = model.tangent_frames(normals)
    direction = groom_direction_3d(groom, normals, tangents, bitangents)
    xy, depth = stage1.project_points(roots, viewmat, k)

    if torch.cuda.is_available():
        import nvdiffrast.torch as dr

        ctx = dr.RasterizeCudaContext(device=device)
        mesh_depth = stage1.render_model_mesh_depth(model, viewmat, k, width, height, device=device, ctx=ctx)
        sampled_mesh_depth = stage1.sample_depth_nearest(mesh_depth.depth, xy, kernel_size=int(config.mesh_depth_local_kernel))
        tolerance = float(config.mesh_depth_abs_tolerance) + depth.abs() * float(config.mesh_depth_rel_tolerance)
        visible = torch.isfinite(sampled_mesh_depth) & (depth > 1.0e-6) & (depth <= sampled_mesh_depth + tolerance)
    else:
        visible = depth > 1.0e-6
    visible = visible & (xy[:, 0] >= 0.0) & (xy[:, 0] < width) & (xy[:, 1] >= 0.0) & (xy[:, 1] < height)

    xy2, depth2 = stage1.project_points(roots + direction * 0.055, viewmat, k)
    visible = visible & (depth2 > 1.0e-6)
    ids = torch.nonzero(visible, as_tuple=False).reshape(-1)
    xy_np = xy[ids].detach().cpu().numpy()
    xy2_np = xy2[ids].detach().cpu().numpy()
    direction_np = direction[ids].detach().cpu().numpy()
    normal_np = normals[ids].detach().cpu().numpy()

    if not model.guide_enabled():
        raise RuntimeError(
            "primary-guide signed curl-turn visualization requires primary guides"
        )
    (
        _guide_points_world,
        guide_xy,
        guide_turns,
        guide_visible,
    ) = project_primary_guide_curl_turns(
        model.guide_points_local,
        model.guide_curl_turns_raw,
        log_scale=model.log_scale,
        translation=model.translation,
        viewmat=viewmat,
        k=k,
        width=width,
        height=height,
        mesh_depth=(mesh_depth.depth if torch.cuda.is_available() else None),
        mesh_depth_kernel=int(config.mesh_depth_local_kernel),
        mesh_depth_abs_tolerance=float(config.mesh_depth_abs_tolerance),
        mesh_depth_rel_tolerance=float(config.mesh_depth_rel_tolerance),
    )
    guide_ids = torch.nonzero(guide_visible, as_tuple=False).reshape(-1)
    guide_xy_np = guide_xy[guide_ids].detach().cpu().numpy()
    guide_turns_all_np = guide_turns.detach().cpu().numpy()
    guide_turns_visible_np = guide_turns[guide_ids].detach().cpu().numpy()

    opacity_attributes_all = effective_root_opacity_attributes(
        groom.root_opacity.detach().cpu().numpy(),
        groom.tip_opacity.detach().cpu().numpy(),
    )
    visible_root_indices = ids.detach().cpu().numpy()
    opacity_attributes = {
        name: values[visible_root_indices]
        for name, values in opacity_attributes_all.items()
    }

    if args.base_image:
        base_path = Path(args.base_image)
    else:
        default_base = checkpoint_path.parent / f"diagnostics_006000_view{int(args.view):02d}" / f"view{int(args.view):02d}_pred.png"
        base_path = default_base
    base = _read_base(base_path, width, height)

    guide_map_path = output_dir / f"view{int(args.view):02d}_primary_guide_curl_turns.png"
    _overlay_points(
        base,
        guide_xy_np,
        guide_turns_visible_np,
        title=f"view{int(args.view):02d} primary guide curl turns",
        out_path=guide_map_path,
        signed=True,
    )

    values = {
        "length": groom.length.reshape(-1)[ids].detach().cpu().numpy(),
        "brush_stiffness": groom.brush_stiffness.reshape(-1)[ids].detach().cpu().numpy(),
        "curl_radius_ratio": groom.curl_radius_ratio.reshape(-1)[ids].detach().cpu().numpy(),
        "curl_radius": (groom.length * groom.curl_radius_ratio).reshape(-1)[ids].detach().cpu().numpy(),
        "curl_turns": groom.curl_turns.reshape(-1)[ids].detach().cpu().numpy(),
        "curl_amount_radius_x_abs_turns": ((groom.length * groom.curl_radius_ratio).reshape(-1)[ids] * groom.curl_turns.reshape(-1)[ids].abs()).detach().cpu().numpy(),
        "direction_local_tangent_x": groom.direction_local[:, 0][ids].detach().cpu().numpy(),
        "direction_local_tangent_y": groom.direction_local[:, 1][ids].detach().cpu().numpy(),
        "direction_local_outward": groom.direction_local[:, 2][ids].detach().cpu().numpy(),
        "normal_component_dot_dir_normal": np.sum(direction_np * normal_np, axis=-1),
        **opacity_attributes,
    }

    outputs: list[tuple[str, Path]] = []
    signed_attributes = {"curl_turns"}
    magnitude_attributes = {
        "curl_radius_ratio",
        "curl_radius",
        "curl_amount_radius_x_abs_turns",
        *EFFECTIVE_ROOT_OPACITY_ATTRIBUTE_NAMES,
    }
    opacity_map_paths = effective_root_attribute_map_paths(output_dir, int(args.view))
    flow_path = output_dir / f"view{int(args.view):02d}_flow_arrows_3d.png"
    _save_flow_arrows(base, xy_np, xy2_np, values["length"], flow_path, title=f"view{int(args.view):02d} projected 3D hair flow")
    outputs.append(("3D flow arrows", flow_path))

    for name, value in values.items():
        path = opacity_map_paths.get(
            name,
            output_dir / f"view{int(args.view):02d}_{name}.png",
        )
        title = f"view{int(args.view):02d} {name.replace('_', ' ')}"
        if name in magnitude_attributes and np.any(value < 0.0):
            raise RuntimeError(f"magnitude attribute became negative: {name}")
        _overlay_points(
            base,
            xy_np,
            value,
            title=title,
            out_path=path,
            signed=name in signed_attributes,
        )
        outputs.append((name.replace("_", " "), path))

    stats = {
        name: summarize_attribute_values(value)
        for name, value in values.items()
    }
    guide_visible_stats = (
        summarize_attribute_values(guide_turns_visible_np)
        if guide_turns_visible_np.size
        else None
    )
    (output_dir / f"view{int(args.view):02d}_groom_attribute_stats.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "view": int(args.view),
                "visible_root_count": int(ids.numel()),
                "stats": stats,
                "effective_root_opacity": {
                    "ratio_eps": EFFECTIVE_OPACITY_RATIO_EPS,
                    "maps": {
                        name: str(opacity_map_paths[name])
                        for name in EFFECTIVE_ROOT_OPACITY_ATTRIBUTE_NAMES
                    },
                    "stats": {
                        name: stats[name]
                        for name in EFFECTIVE_ROOT_OPACITY_ATTRIBUTE_NAMES
                    },
                },
                "primary_guide": {
                    "guide_count": int(guide_turns_all_np.size),
                    "visible_guide_count": int(guide_ids.numel()),
                    "curl_turns_map": str(guide_map_path),
                    "curl_turns_stats": {
                        "all_guides": summarize_attribute_values(guide_turns_all_np),
                        "visible_guides": guide_visible_stats,
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _make_contact_sheet(outputs, output_dir / f"view{int(args.view):02d}_groom_attribute_contact.png")
    print(json.dumps({"output_dir": str(output_dir), "visible_root_count": int(ids.numel())}, indent=2))


if __name__ == "__main__":
    main()

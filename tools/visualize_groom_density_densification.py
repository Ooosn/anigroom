"""Visualize hair density and current root densification behavior.

This is a diagnostic figure generator. It treats density strictly as root
sampling density, not as an extra trainable groom attribute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
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
from anigroom.roots import (
    DensifyConfig,
    PruneConfig,
    RootLifecycleState,
    RootStats,
    apply_attribute_update,
    apply_structure_update,
    interpolate_child_attributes,
    propose_structure_update,
)
from anigroom.roots.lifecycle import barycentric_to_points


def _tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _crop_foreground(image: Image.Image, background: tuple[int, int, int] = (184, 189, 194), margin: int = 24) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.int16)
    bg = np.asarray(background, dtype=np.int16).reshape(1, 1, 3)
    mask = np.abs(arr - bg).sum(axis=-1) > 18
    if not mask.any():
        return image.convert("RGB")
    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, image.width)
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, image.height)
    return image.crop((x0, y0, x1, y1)).convert("RGB")


def _label_tile(image: Image.Image, label: str, tile_size: tuple[int, int] = (520, 420), crop: bool = True) -> Image.Image:
    image = image.convert("RGB")
    if crop:
        image = _crop_foreground(image)
    image.thumbnail(tile_size, Image.Resampling.LANCZOS)
    header = 34
    out = Image.new("RGB", (tile_size[0], tile_size[1] + header), (245, 245, 245))
    out.paste(image, ((tile_size[0] - image.width) // 2, header + (tile_size[1] - image.height) // 2))
    draw = ImageDraw.Draw(out)
    draw.text((10, 9), label, fill=(20, 20, 20))
    return out


def _make_row(images: list[Image.Image], labels: list[str], output: Path, *, crop: bool = True) -> None:
    tiles = [_label_tile(im, label, crop=crop) for im, label in zip(images, labels)]
    w = sum(im.width for im in tiles)
    h = max(im.height for im in tiles)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im in tiles:
        sheet.paste(im, (x, 0))
        x += im.width
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def load_roots(root_path: Path, mesh_path: Path, count: int, device: torch.device):
    data = np.load(root_path)
    face_ids = torch.from_numpy(data["face_ids"][:count].astype(np.int64)).to(device=device)
    bary = torch.from_numpy(data["barycentric"][:count].astype(np.float32)).to(device=device)
    mesh = read_obj_mesh(mesh_path)
    vertices = torch.from_numpy(mesh.vertices).to(device=device)
    faces = torch.from_numpy(mesh.faces).to(device=device)
    points = barycentric_to_points(vertices, faces, face_ids, bary)
    normals = face_normals(vertices, faces, face_ids)
    return vertices, faces, points, normals, face_ids, bary


def face_normals(vertices: torch.Tensor, faces: torch.Tensor, face_ids: torch.Tensor) -> torch.Tensor:
    tri = vertices[faces[face_ids]]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp_min(1e-8)


def scaled_cameras(data_root: Path, mesh_path: Path, device: torch.device, width: int, height: int):
    report = build_stage1_input_report(data_root, mesh_path)
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    orig_w, orig_h = report.image_size or (0, 0)
    viewmats, ks = load_camera_tensors(data_root, device)
    ks = ks.clone()
    ks[:, 0, 0] *= float(width) / float(orig_w)
    ks[:, 1, 1] *= float(height) / float(orig_h)
    ks[:, 0, 2] *= float(width) / float(orig_w)
    ks[:, 1, 2] *= float(height) / float(orig_h)
    return report, viewmats, ks


def apply_visual_groom(field: GroomParameterField, roots: torch.Tensor) -> None:
    """Create a high-contrast white-tiger groom style for diagnostics."""

    with torch.no_grad():
        x = roots[:, [0]]
        y = roots[:, [1]]
        z = roots[:, [2]]
        body = (z - z.min()) / (z.max() - z.min()).clamp_min(1e-6)
        side = (y - y.min()) / (y.max() - y.min()).clamp_min(1e-6)
        stripe = torch.sigmoid(7.0 * (torch.sin(42.0 * body + 5.0 * torch.sin(9.0 * side)) - 0.18))
        long_band = torch.sigmoid(12.0 * (0.62 - body)) * torch.sigmoid(12.0 * (body - 0.18))

        field.length_raw.add_(1.1 + 1.9 * long_band + 0.45 * torch.sin(10.0 * body))
        field.root_width_raw.add_(1.35)
        field.tip_width_ratio_raw.add_(0.55)
        field.opacity_raw.add_(1.15)
        field.flow_xy[:, 0:1].add_(1.0 + 0.7 * torch.sin(6.0 * body))
        field.flow_xy[:, 1:2].add_(0.55 * torch.cos(9.0 * side))
        field.bend_raw.add_(0.95 * torch.sin(8.0 * body))
        field.sag_raw.add_(0.8 * torch.sigmoid(8.0 * (0.45 - side)))
        field.stiffness_raw.sub_(0.45)
        dark = stripe.expand(-1, 3)
        field.root_color_raw.add_(-3.2 * dark)
        field.tip_color_raw.add_(-2.7 * dark)


def make_visual_field(root_count: int, roots: torch.Tensor) -> GroomParameterField:
    field = GroomParameterField(root_count, device=roots.device)
    apply_visual_groom(field, roots.detach())
    return field


def render_hair(
    roots: torch.Tensor,
    normals: torch.Tensor,
    field: GroomParameterField,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    samples: int,
    min_segments: int,
    max_segments: int,
    background: tuple[float, float, float],
    retain_grad: bool = False,
):
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
    if retain_grad:
        gaussians.means.retain_grad()
        gaussians.scales.retain_grad()
        gaussians.opacities.retain_grad()
    bg = torch.tensor(background, device=roots.device, dtype=roots.dtype).view(1, 3)
    render, alpha, info = rasterization(
        gaussians.means,
        gaussians.quats,
        gaussians.scales,
        gaussians.opacities.reshape(-1),
        gaussians.colors,
        viewmat[None],
        k[None],
        width,
        height,
        packed=False,
        backgrounds=bg,
    )
    return render[0].clamp(0.0, 1.0), alpha[0].clamp(0.0, 1.0), gaussians, info, resampled.stats


def root_stats_from_gsplat(points: torch.Tensor, gaussians, info: dict, residual: torch.Tensor) -> RootStats:
    root_count = int(points.shape[0])
    root_ids = gaussians.root_indices.long()
    mean_grad = gaussians.means.grad.detach().abs().sum(dim=-1, keepdim=True)
    scale_grad = gaussians.scales.grad.detach().abs().sum(dim=-1, keepdim=True) if gaussians.scales.grad is not None else torch.zeros_like(mean_grad)
    gaussian_grad = mean_grad + 0.25 * scale_grad
    gaussian_grad_sum = torch.zeros(root_count, 1, device=points.device)
    gaussian_grad_sum.scatter_add_(0, root_ids[:, None], gaussian_grad)
    radii = info.get("radii") if isinstance(info, dict) else None
    if radii is None:
        visible_g = torch.ones_like(mean_grad)
    else:
        visible_g = (radii.reshape(-1) > 0).float().view(-1, 1)
        if visible_g.shape[0] != mean_grad.shape[0]:
            visible_g = torch.ones_like(mean_grad)
    contrib_g = visible_g * gaussians.opacities.detach().reshape(-1, 1).clamp_min(1e-6)
    contrib_sum = torch.zeros(root_count, 1, device=points.device)
    contrib_sum.scatter_add_(0, root_ids[:, None], contrib_g)
    visible_count = torch.zeros(root_count, 1, device=points.device)
    visible_count.scatter_add_(0, root_ids[:, None], visible_g)
    root_grad = points.grad.detach().abs().sum(dim=-1, keepdim=True) if points.grad is not None else torch.zeros_like(contrib_sum)
    return RootStats(
        root_grad_abs_sum=root_grad,
        gaussian_grad_abs_sum=gaussian_grad_sum,
        gaussian_contrib_sum=contrib_sum,
        visible_count=visible_count,
        residual_sum=torch.full_like(contrib_sum, float(residual.detach().cpu())),
    )


def propose_densified_roots(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    field: GroomParameterField,
    face_ids: torch.Tensor,
    bary: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    args: argparse.Namespace,
):
    logits = torch.nn.Parameter(torch.log(bary.clamp_min(1e-5)))
    bary_opt = torch.softmax(logits, dim=-1)
    root_points = barycentric_to_points(vertices, faces, face_ids, bary_opt)
    root_points.retain_grad()
    render, alpha, gaussians, info, _ = render_hair(
        root_points,
        normals,
        field,
        viewmat,
        k,
        width,
        height,
        args.samples,
        args.min_segments,
        args.max_segments,
        tuple(args.background),
        retain_grad=True,
    )
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=points.device),
        torch.linspace(-1.0, 1.0, width, device=points.device),
        indexing="ij",
    )
    target = render.detach().clone()
    target_region = torch.exp(-((xx + 0.36) ** 2 / 0.035 + (yy - 0.08) ** 2 / 0.050)).unsqueeze(-1)
    target = (target * (1.0 - 0.45 * target_region)).clamp(0.0, 1.0)
    loss = ((render - target) ** 2).mean() + 0.15 * ((alpha - alpha.detach()) ** 2).mean()
    loss.backward()
    stats = root_stats_from_gsplat(root_points, gaussians, info, loss)
    state = RootLifecycleState(points=root_points.detach(), face_ids=face_ids.detach(), barycentric=bary_opt.detach())
    update = propose_structure_update(
        state,
        stats,
        DensifyConfig(
            grad_threshold=args.grad_threshold,
            visibility_threshold=1.0,
            max_new_roots=args.max_new_roots,
            children_per_parent=args.children_per_parent,
            barycentric_step=args.barycentric_step,
            replace_parent=True,
            neighbor_count=args.neighbor_count,
            candidate_rings=args.candidate_rings,
            candidate_face_count=args.candidate_face_count,
        ),
        PruneConfig(max_prune_fraction=0.0),
        vertices=vertices,
        faces=faces,
    )
    after = apply_structure_update(state, update, vertices, faces)
    after_normals = face_normals(vertices, faces, after.face_ids)
    after_field = apply_field_update(field, state, update, vertices, faces, args.attribute_neighbor_count)
    return update, after, after_normals, after_field, float(loss.detach().cpu())


def raw_parameter_names(field: GroomParameterField) -> list[str]:
    return [name for name, _ in field.named_parameters()]


def apply_field_update(
    field: GroomParameterField,
    state: RootLifecycleState,
    update,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attribute_neighbor_count: int,
) -> GroomParameterField:
    new_count = int(state.points.shape[0] + update.new_barycentric.shape[0] - update.prune_mask.sum().item())
    out = GroomParameterField(new_count, ranges=field.ranges, device=state.points.device)
    for name in raw_parameter_names(field):
        old_value = getattr(field, name).detach()
        child_value = interpolate_child_attributes(
            old_value,
            state,
            update,
            vertices,
            faces,
            neighbor_count=attribute_neighbor_count,
            parent_weight=3.0,
        )
        new_value = apply_attribute_update(old_value, update, child_value)
        setattr(out, name, nn.Parameter(new_value.clone()))
    out.root_count = new_count
    return out


def draw_root_overlay(points: torch.Tensor, update, after: RootLifecycleState, output: Path) -> Image.Image:
    size = 760
    margin = 60
    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    pts = points.detach().cpu().numpy()
    all_pts = after.points.detach().cpu().numpy()
    new_pts = all_pts[pts.shape[0] :]

    def project(arr: np.ndarray):
        x = margin + (arr[:, 0] - pts[:, 0].min()) / max(np.ptp(pts[:, 0]), 1e-6) * (size - 2 * margin)
        y = margin + (1.0 - (arr[:, 2] - pts[:, 2].min()) / max(np.ptp(pts[:, 2]), 1e-6)) * (size - 2 * margin)
        return x, y

    x, y = project(pts)
    parents = set(int(v) for v in update.parent_indices.detach().cpu().reshape(-1).tolist())
    for i, (px, py) in enumerate(zip(x, y)):
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(75, 120, 190))
        if i in parents:
            draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=(245, 130, 20), width=2)
    if new_pts.size:
        nx, ny = project(new_pts)
        for px, py in zip(nx, ny):
            draw.line((px - 6, py - 6, px + 6, py + 6), fill=(20, 160, 80), width=2)
            draw.line((px - 6, py + 6, px + 6, py - 6), fill=(20, 160, 80), width=2)
    draw.text((18, 18), "blue: existing roots | orange: selected parents | green x: inserted roots", fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk")
    parser.add_argument("--mesh-path", default="D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj")
    parser.add_argument("--root-path", default="D:/petsgaussianhair/_downloads/root_init_20260623/white_tiger_roots_2048_fps.npz")
    parser.add_argument("--output-dir", default="D:/petsgaussianhair/_downloads/groom_density_densification_20260623")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--view-index", type=int, default=-1)
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=17)
    parser.add_argument("--grad-threshold", type=float, default=3e-5)
    parser.add_argument("--max-new-roots", type=int, default=240)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--barycentric-step", type=float, default=0.075)
    parser.add_argument("--neighbor-count", type=int, default=14)
    parser.add_argument("--candidate-rings", type=int, default=4)
    parser.add_argument("--candidate-face-count", type=int, default=36)
    parser.add_argument("--attribute-neighbor-count", type=int, default=8)
    parser.add_argument("--background", nargs=3, type=float, default=(0.72, 0.74, 0.76))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat rendering")
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh_path)
    data_root = Path(args.data_root)
    report, viewmats, ks = scaled_cameras(data_root, mesh_path, device, args.width, args.height)
    view_idx = int(args.view_index if args.view_index >= 0 else report.train_indices[0])
    viewmat = viewmats[view_idx]
    k = ks[view_idx]

    density_counts = [256, 768, 2048]
    density_images: list[Image.Image] = []
    density_labels: list[str] = []
    density_stats: dict[str, dict[str, float | int]] = {}
    for count in density_counts:
        _, _, roots, normals, _, _ = load_roots(Path(args.root_path), mesh_path, count, device)
        field = make_visual_field(count, roots)
        render, _, gaussians, _, stats = render_hair(
            roots,
            normals,
            field,
            viewmat,
            k,
            args.width,
            args.height,
            args.samples,
            args.min_segments,
            args.max_segments,
            tuple(args.background),
        )
        density_images.append(_tensor_to_image(render))
        density_labels.append(f"{count} FPS roots / {int(gaussians.means.shape[0])} Gaussians")
        density_stats[str(count)] = {"gaussian_count": int(gaussians.means.shape[0]), **stats}
    _make_row(density_images, density_labels, output_dir / "hair_root_density_sweep.jpg")

    vertices, faces, roots, normals, face_ids, bary = load_roots(Path(args.root_path), mesh_path, 768, device)
    before_render, _, before_gaussians, _, before_stats = render_hair(
        roots,
        normals,
        before_field := make_visual_field(int(roots.shape[0]), roots),
        viewmat,
        k,
        args.width,
        args.height,
        args.samples,
        args.min_segments,
        args.max_segments,
        tuple(args.background),
    )
    update, after, after_normals, after_field, densify_loss = propose_densified_roots(
        vertices,
        faces,
        roots,
        normals,
        before_field,
        face_ids,
        bary,
        viewmat,
        k,
        args.width,
        args.height,
        args,
    )
    after_render, _, after_gaussians, _, after_stats = render_hair(
        after.points,
        after_normals,
        after_field,
        viewmat,
        k,
        args.width,
        args.height,
        args.samples,
        args.min_segments,
        args.max_segments,
        tuple(args.background),
    )
    overlay = draw_root_overlay(roots, update, after, output_dir / "densification_root_overlay.jpg")
    _make_row(
        [_tensor_to_image(before_render), _tensor_to_image(after_render), overlay.resize((args.width, args.height), Image.Resampling.LANCZOS)],
        [
            f"before: 768 roots / {int(before_gaussians.means.shape[0])} Gaussians",
            f"after current densify: {int(after.points.shape[0])} roots / {int(after_gaussians.means.shape[0])} Gaussians",
            "root overlay",
        ],
        output_dir / "hair_densification_before_after.jpg",
    )
    summary = {
        "output_dir": str(output_dir),
        "view_index": view_idx,
        "density_sweep": density_stats,
        "densify": {
            "before_root_count": int(roots.shape[0]),
            "after_root_count": int(after.points.shape[0]),
            "selected_parent_count": int(update.parent_indices.numel()),
            "new_root_count": int(update.new_barycentric.shape[0]),
            "before_gaussian_count": int(before_gaussians.means.shape[0]),
            "after_gaussian_count": int(after_gaussians.means.shape[0]),
            "diagnostic_loss": densify_loss,
            "before_stats": before_stats,
            "after_stats": after_stats,
        },
        "figures": {
            "density_sweep": str(output_dir / "hair_root_density_sweep.jpg"),
            "densification": str(output_dir / "hair_densification_before_after.jpg"),
            "overlay": str(output_dir / "densification_root_overlay.jpg"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

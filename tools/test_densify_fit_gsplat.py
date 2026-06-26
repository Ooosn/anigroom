"""Train a groom field before and after root densification.

The target is rendered from a denser teacher root set.  A lower-density student
is optimized, then split-replace densification is applied from real gsplat
gradients, and training continues with the interpolated child groom attributes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anigroom.grooming import GroomParameterField
from anigroom.roots import DensifyConfig, PruneConfig, RootLifecycleState, RootStats, apply_attribute_update, apply_structure_update, interpolate_child_attributes, propose_structure_update
from anigroom.roots.lifecycle import barycentric_to_points
from visualize_groom_density_densification import (
    apply_visual_groom,
    face_normals,
    load_roots,
    render_hair,
    scaled_cameras,
    _tensor_to_image,
)


def make_field(root_count: int, roots: torch.Tensor) -> GroomParameterField:
    field = GroomParameterField(root_count, device=roots.device)
    apply_visual_groom(field, roots.detach())
    return field


def clone_field(field: GroomParameterField) -> GroomParameterField:
    out = GroomParameterField(field.root_count, ranges=field.ranges, device=next(field.parameters()).device)
    for name, param in field.named_parameters():
        setattr(out, name, torch.nn.Parameter(param.detach().clone()))
    out.root_count = field.root_count
    return out


def render_views(field, roots, normals, viewmats, ks, view_indices, args):
    images = []
    alphas = []
    last_gaussians = None
    last_info = None
    last_stats = None
    for view_idx in view_indices:
        image, alpha, gaussians, info, stats = render_hair(
            roots,
            normals,
            field,
            viewmats[view_idx],
            ks[view_idx],
            args.width,
            args.height,
            args.samples,
            args.min_segments,
            args.max_segments,
            tuple(args.background),
        )
        images.append(image)
        alphas.append(alpha)
        last_gaussians = gaussians
        last_info = info
        last_stats = stats
    return images, alphas, last_gaussians, last_info, last_stats


def image_loss(pred_images, pred_alphas, target_images, target_alphas):
    rgb = torch.stack([F.mse_loss(pred, target) for pred, target in zip(pred_images, target_images)]).mean()
    alpha = torch.stack([F.mse_loss(pred, target) for pred, target in zip(pred_alphas, target_alphas)]).mean()
    return rgb + 0.2 * alpha, rgb, alpha


def optimize(field, roots, normals, viewmats, ks, view_indices, target_images, target_alphas, args, steps, lr, phase):
    optimizer = torch.optim.Adam(field.parameters(), lr=lr)
    history = []
    for step in range(1, int(steps) + 1):
        pred_images, pred_alphas, gaussians, _, _ = render_views(field, roots, normals, viewmats, ks, view_indices, args)
        loss, rgb, alpha = image_loss(pred_images, pred_alphas, target_images, target_alphas)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, int(args.log_every)) == 0 or step == int(steps):
            rec = {
                "phase": phase,
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "rgb_mse": float(rgb.detach().cpu()),
                "alpha_mse": float(alpha.detach().cpu()),
                "root_count": int(roots.shape[0]),
                "gaussian_count": int(gaussians.means.shape[0]) if gaussians is not None else -1,
            }
            history.append(rec)
            print(json.dumps(rec), flush=True)
    return history


def root_stats_from_last_render(root_points, field, normals, viewmats, ks, view_indices, target_images, target_alphas, args):
    """Collect real gradient stats from the current training loss."""

    root_points = root_points.detach().clone().requires_grad_(True)
    pred_images = []
    pred_alphas = []
    infos = []
    gaussians_last = None
    for view_idx in view_indices:
        image, alpha, gaussians, info, _ = render_hair(
            root_points,
            normals,
            field,
            viewmats[view_idx],
            ks[view_idx],
            args.width,
            args.height,
            args.samples,
            args.min_segments,
            args.max_segments,
            tuple(args.background),
            retain_grad=True,
        )
        pred_images.append(image)
        pred_alphas.append(alpha)
        infos.append(info)
        gaussians_last = gaussians
    loss, _, _ = image_loss(pred_images, pred_alphas, target_images, target_alphas)
    loss.backward()
    if gaussians_last is None or gaussians_last.means.grad is None:
        raise RuntimeError("failed to collect gaussian gradients")
    root_count = int(root_points.shape[0])
    root_ids = gaussians_last.root_indices.long()
    mean_grad = gaussians_last.means.grad.detach().abs().sum(dim=-1, keepdim=True)
    scale_grad = gaussians_last.scales.grad.detach().abs().sum(dim=-1, keepdim=True) if gaussians_last.scales.grad is not None else torch.zeros_like(mean_grad)
    gaussian_grad = mean_grad + 0.25 * scale_grad
    gaussian_grad_sum = torch.zeros(root_count, 1, device=root_points.device)
    gaussian_grad_sum.scatter_add_(0, root_ids[:, None], gaussian_grad)
    visible_g = torch.zeros_like(mean_grad)
    for info in infos:
        radii = info.get("radii") if isinstance(info, dict) else None
        if radii is None or radii.numel() != mean_grad.numel():
            visible_g += 1.0
        else:
            visible_g += (radii.reshape(-1) > 0).float().view(-1, 1)
    contrib_g = visible_g * gaussians_last.opacities.detach().reshape(-1, 1).clamp_min(1e-6)
    contrib_sum = torch.zeros(root_count, 1, device=root_points.device)
    contrib_sum.scatter_add_(0, root_ids[:, None], contrib_g)
    visible_count = torch.zeros(root_count, 1, device=root_points.device)
    visible_count.scatter_add_(0, root_ids[:, None], visible_g)
    root_grad = root_points.grad.detach().abs().sum(dim=-1, keepdim=True)
    return RootStats(
        root_grad_abs_sum=root_grad,
        gaussian_grad_abs_sum=gaussian_grad_sum,
        gaussian_contrib_sum=contrib_sum,
        visible_count=visible_count,
        residual_sum=torch.full_like(contrib_sum, float(loss.detach().cpu())),
    ), float(loss.detach().cpu())


def apply_field_update(field, state, update, vertices, faces, attr_neighbors):
    old_params = dict(field.named_parameters())
    new_count = int(state.points.shape[0] + update.new_barycentric.shape[0] - update.prune_mask.sum().item())
    new_field = GroomParameterField(new_count, ranges=field.ranges, device=state.points.device)
    for name, param in old_params.items():
        old_value = param.detach()
        child_value = interpolate_child_attributes(old_value, state, update, vertices, faces, neighbor_count=attr_neighbors, parent_weight=3.0)
        new_value = apply_attribute_update(old_value, update, child_value)
        setattr(new_field, name, torch.nn.Parameter(new_value.clone()))
    new_field.root_count = new_count
    return new_field


def save_fit_sheet(output_dir, target, before, control, after):
    images = [
        ("teacher high-density target", _tensor_to_image(target)),
        ("768 roots before split", _tensor_to_image(before)),
        ("768 roots continued", _tensor_to_image(control)),
        ("split roots continued", _tensor_to_image(after)),
    ]
    tile_w, tile_h = 460, 300
    header = 34
    sheet = Image.new("RGB", (tile_w * len(images), tile_h + header), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(images):
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = i * tile_w + (tile_w - image.width) // 2
        y = header + (tile_h - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((i * tile_w + 10, 10), label, fill=(20, 20, 20))
    path = output_dir / "densify_fit_sheet.jpg"
    sheet.save(path, quality=95)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk")
    parser.add_argument("--mesh-path", default="D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj")
    parser.add_argument("--root-path", default="D:/petsgaussianhair/_downloads/root_init_20260623/white_tiger_roots_2048_fps.npz")
    parser.add_argument("--output-dir", default="D:/petsgaussianhair/_downloads/densify_fit_gsplat_20260623")
    parser.add_argument("--student-roots", type=int, default=768)
    parser.add_argument("--teacher-roots", type=int, default=2048)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--views", default="25")
    parser.add_argument("--pre-steps", type=int, default=80)
    parser.add_argument("--post-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=15)
    parser.add_argument("--grad-threshold", type=float, default=2.0e-5)
    parser.add_argument("--max-new-roots", type=int, default=240)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--candidate-face-count", type=int, default=48)
    parser.add_argument("--candidate-rings", type=int, default=4)
    parser.add_argument("--neighbor-count", type=int, default=16)
    parser.add_argument("--attribute-neighbor-count", type=int, default=8)
    parser.add_argument("--background", nargs=3, type=float, default=(0.72, 0.74, 0.76))
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = Path(args.mesh_path)
    root_path = Path(args.root_path)
    report, viewmats, ks = scaled_cameras(Path(args.data_root), mesh_path, device, args.width, args.height)
    view_indices = [int(v) for v in args.views.split(",") if v.strip()]
    if not view_indices:
        view_indices = report.train_indices[:1]

    vertices, faces, student_roots, student_normals, face_ids, bary = load_roots(root_path, mesh_path, args.student_roots, device)
    _, _, teacher_roots, teacher_normals, _, _ = load_roots(root_path, mesh_path, args.teacher_roots, device)
    teacher_field = make_field(args.teacher_roots, teacher_roots)
    with torch.no_grad():
        target_images, target_alphas, _, _, _ = render_views(teacher_field, teacher_roots, teacher_normals, viewmats, ks, view_indices, args)

    field = make_field(args.student_roots, student_roots)
    with torch.no_grad():
        before_train_images, before_train_alphas, _, _, _ = render_views(field, student_roots, student_normals, viewmats, ks, view_indices, args)
        initial_loss, initial_rgb, initial_alpha = image_loss(before_train_images, before_train_alphas, target_images, target_alphas)
    history = [
        {
            "phase": "initial",
            "step": 0,
            "loss": float(initial_loss.detach().cpu()),
            "rgb_mse": float(initial_rgb.detach().cpu()),
            "alpha_mse": float(initial_alpha.detach().cpu()),
            "root_count": int(student_roots.shape[0]),
        }
    ]
    history.extend(optimize(field, student_roots, student_normals, viewmats, ks, view_indices, target_images, target_alphas, args, args.pre_steps, args.lr, "pre"))
    with torch.no_grad():
        pre_images, pre_alphas, _, _, _ = render_views(field, student_roots, student_normals, viewmats, ks, view_indices, args)
        pre_loss, pre_rgb, pre_alpha = image_loss(pre_images, pre_alphas, target_images, target_alphas)
    control_field = clone_field(field)
    control_history = optimize(
        control_field,
        student_roots,
        student_normals,
        viewmats,
        ks,
        view_indices,
        target_images,
        target_alphas,
        args,
        args.post_steps,
        args.lr * 0.8,
        "control_no_densify",
    )
    history.extend(control_history)
    with torch.no_grad():
        control_images, control_alphas, _, _, _ = render_views(control_field, student_roots, student_normals, viewmats, ks, view_indices, args)
        control_loss, control_rgb, control_alpha = image_loss(control_images, control_alphas, target_images, target_alphas)

    stats, densify_probe_loss = root_stats_from_last_render(student_roots, field, student_normals, viewmats, ks, view_indices, target_images, target_alphas, args)
    state = RootLifecycleState(points=student_roots.detach(), face_ids=face_ids.detach(), barycentric=bary.detach())
    update = propose_structure_update(
        state,
        stats,
        DensifyConfig(
            grad_threshold=args.grad_threshold,
            max_new_roots=args.max_new_roots,
            children_per_parent=args.children_per_parent,
            replace_parent=True,
            neighbor_count=args.neighbor_count,
            candidate_rings=args.candidate_rings,
            candidate_face_count=args.candidate_face_count,
        ),
        PruneConfig(max_prune_fraction=0.0),
        vertices=vertices,
        faces=faces,
    )
    next_state = apply_structure_update(state, update, vertices, faces)
    next_normals = face_normals(vertices, faces, next_state.face_ids)
    field = apply_field_update(field, state, update, vertices, faces, args.attribute_neighbor_count)
    with torch.no_grad():
        immediate_images, immediate_alphas, _, _, _ = render_views(field, next_state.points, next_normals, viewmats, ks, view_indices, args)
        immediate_loss, immediate_rgb, immediate_alpha = image_loss(immediate_images, immediate_alphas, target_images, target_alphas)
    history.append(
        {
            "phase": "after_densify_before_fit",
            "step": 0,
            "loss": float(immediate_loss.detach().cpu()),
            "rgb_mse": float(immediate_rgb.detach().cpu()),
            "alpha_mse": float(immediate_alpha.detach().cpu()),
            "root_count": int(next_state.points.shape[0]),
            "selected_parent_count": int(update.parent_indices.numel()),
            "new_root_count": int(update.new_barycentric.shape[0]),
            "probe_loss": densify_probe_loss,
        }
    )
    history.extend(optimize(field, next_state.points, next_normals, viewmats, ks, view_indices, target_images, target_alphas, args, args.post_steps, args.lr * 0.8, "post"))
    with torch.no_grad():
        final_images, final_alphas, _, _, _ = render_views(field, next_state.points, next_normals, viewmats, ks, view_indices, args)
        final_loss, final_rgb, final_alpha = image_loss(final_images, final_alphas, target_images, target_alphas)

    sheet = save_fit_sheet(output_dir, target_images[0], pre_images[0], control_images[0], final_images[0])
    summary = {
        "output_dir": str(output_dir),
        "views": view_indices,
        "student_roots_initial": int(student_roots.shape[0]),
        "student_roots_after": int(next_state.points.shape[0]),
        "teacher_roots": int(teacher_roots.shape[0]),
        "selected_parent_count": int(update.parent_indices.numel()),
        "new_root_count": int(update.new_barycentric.shape[0]),
        "initial_loss": float(initial_loss.detach().cpu()),
        "pre_loss": float(pre_loss.detach().cpu()),
        "after_densify_before_fit_loss": float(immediate_loss.detach().cpu()),
        "control_no_densify_final_loss": float(control_loss.detach().cpu()),
        "control_no_densify_rgb_mse": float(control_rgb.detach().cpu()),
        "control_no_densify_alpha_mse": float(control_alpha.detach().cpu()),
        "final_loss": float(final_loss.detach().cpu()),
        "final_rgb_mse": float(final_rgb.detach().cpu()),
        "final_alpha_mse": float(final_alpha.detach().cpu()),
        "history": history,
        "fit_sheet": str(sheet),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Validate root densification signals from actual gsplat hair gradients.

This test uses the real grooming -> strand -> Gaussian -> gsplat path.  A
teacher groom is edited in one spatial region, then a student groom renders
against that teacher target.  After backpropagation we collect:

* natural root-position gradient;
* per-Gaussian mean gradient aggregated by ``root_indices``;
* per-root visibility from gsplat radii when available.

Those statistics are fed into the root lifecycle module to verify that parent
selection follows actual hair-rendering gradients, not synthetic placeholders.
"""

from __future__ import annotations

import argparse
import json
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
from anigroom.roots import DensifyConfig, PruneConfig, RootLifecycleState, RootStats, propose_structure_update
from anigroom.roots.lifecycle import barycentric_to_points


def load_root_data(root_path: Path, mesh_path: Path, root_count: int, device: torch.device):
    data = np.load(root_path)
    face_ids = torch.from_numpy(data["face_ids"][:root_count].astype(np.int64)).to(device=device)
    bary = torch.from_numpy(data["barycentric"][:root_count].astype(np.float32)).to(device=device)
    mesh = read_obj_mesh(mesh_path)
    vertices = torch.from_numpy(mesh.vertices).to(device=device)
    faces = torch.from_numpy(mesh.faces).to(device=device)
    tri = vertices[faces[face_ids]]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    normals = F.normalize(normals, dim=-1)
    return vertices, faces, face_ids, bary, normals


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


def apply_regional_teacher(field: GroomParameterField, roots: torch.Tensor) -> torch.Tensor:
    """Make one region require clearly higher local fur capacity."""

    with torch.no_grad():
        x = roots[:, [0]]
        y = roots[:, [1]]
        z = roots[:, [2]]
        center = torch.tensor([[-0.32, 0.08, 0.02]], device=roots.device, dtype=roots.dtype)
        dist2 = ((roots - center) ** 2).sum(dim=-1, keepdim=True)
        demand = torch.exp(-dist2 / 0.035)
        stripe = torch.sigmoid(5.0 * (torch.sin(34.0 * z + 5.0 * x) - 0.15))
        field.length_raw.add_(2.4 * demand)
        field.root_width_raw.add_(1.5 * demand)
        field.tip_width_ratio_raw.add_(0.8 * demand)
        field.opacity_raw.add_(1.1 * demand)
        field.flow_xy[:, 0:1].add_(1.1 * demand)
        field.flow_xy[:, 1:2].add_(0.9 * demand)
        field.bend_raw.add_(1.5 * demand)
        field.sag_raw.add_(1.0 * demand)
        field.root_color_raw.add_(-2.0 * demand * stripe.expand(-1, 3))
        field.tip_color_raw.add_(-1.8 * demand * stripe.expand(-1, 3))
    return demand.detach()


def render_with_gaussians(
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
    retain_gaussian_grad: bool,
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
    if retain_gaussian_grad:
        gaussians.means.retain_grad()
        gaussians.scales.retain_grad()
        gaussians.opacities.retain_grad()
    images = []
    alphas = []
    infos = []
    background = torch.ones((1, 3), device=roots.device)
    for idx in view_indices:
        render, alpha, info = rasterization(
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
        infos.append(info)
    return images, alphas, gaussians, infos, resampled.stats


def aggregate_root_stats(
    root_points: torch.Tensor,
    gaussians,
    infos: list[dict],
    loss_residual: torch.Tensor,
) -> RootStats:
    root_count = int(root_points.shape[0])
    device = root_points.device
    gaussian_count = int(gaussians.means.shape[0])
    root_ids = gaussians.root_indices.long()
    if gaussians.means.grad is None:
        raise RuntimeError("gaussian means grad was not retained")
    mean_grad = gaussians.means.grad.detach().abs().sum(dim=-1, keepdim=True)
    scale_grad = torch.zeros_like(mean_grad)
    if gaussians.scales.grad is not None:
        scale_grad = gaussians.scales.grad.detach().abs().sum(dim=-1, keepdim=True)
    gaussian_grad = mean_grad + 0.25 * scale_grad
    gaussian_grad_sum = torch.zeros(root_count, 1, device=device)
    gaussian_grad_sum.scatter_add_(0, root_ids[:, None], gaussian_grad)

    visible_g = torch.zeros(gaussian_count, 1, device=device)
    for info in infos:
        radii = info.get("radii") if isinstance(info, dict) else None
        if radii is None:
            visible_g += 1.0
            continue
        radii_t = radii.detach()
        if radii_t.ndim == 2:
            visible = (radii_t[0] > 0).float().view(-1, 1)
        else:
            visible = (radii_t.reshape(-1) > 0).float().view(-1, 1)
        if visible.shape[0] == gaussian_count:
            visible_g += visible
        else:
            visible_g += 1.0
    contrib_g = visible_g * gaussians.opacities.detach().reshape(-1, 1).clamp_min(1e-6)
    contrib_sum = torch.zeros(root_count, 1, device=device)
    contrib_sum.scatter_add_(0, root_ids[:, None], contrib_g)
    visible_count = torch.zeros(root_count, 1, device=device)
    visible_count.scatter_add_(0, root_ids[:, None], (visible_g > 0).float())
    root_grad = torch.zeros(root_count, 1, device=device)
    if root_points.grad is not None:
        root_grad = root_points.grad.detach().abs().sum(dim=-1, keepdim=True)
    residual = torch.full((root_count, 1), float(loss_residual.detach().cpu()), device=device)
    return RootStats(
        root_grad_abs_sum=root_grad,
        gaussian_grad_abs_sum=gaussian_grad_sum,
        gaussian_contrib_sum=contrib_sum,
        visible_count=visible_count,
        residual_sum=residual,
        opacity_mean=None,
    )


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_debug(
    output: Path,
    roots: torch.Tensor,
    demand: torch.Tensor,
    parent_ids: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    new_roots: torch.Tensor | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tensor_to_image(target).save(output / "teacher_target.png")
    tensor_to_image(pred).save(output / "student_pred.png")
    size = 900
    margin = 70
    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    arr = roots.detach().cpu().numpy()
    d = demand.detach().cpu().numpy().reshape(-1)
    px = margin + (arr[:, 0] - arr[:, 0].min()) / max(arr[:, 0].ptp(), 1e-6) * (size - 2 * margin)
    py = margin + (1.0 - (arr[:, 1] - arr[:, 1].min()) / max(arr[:, 1].ptp(), 1e-6)) * (size - 2 * margin)
    parent_set = set(int(x) for x in parent_ids.detach().cpu().reshape(-1).tolist())
    for i, (x, y) in enumerate(zip(px, py)):
        val = int(220 * float(d[i]))
        color = (80 + val, 120, 210 - min(val, 160))
        r = 2
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        if i in parent_set:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 120, 0), width=2)
    if new_roots is not None and new_roots.numel() > 0:
        new_arr = new_roots.detach().cpu().numpy()
        new_px = margin + (new_arr[:, 0] - arr[:, 0].min()) / max(arr[:, 0].ptp(), 1e-6) * (size - 2 * margin)
        new_py = margin + (1.0 - (new_arr[:, 1] - arr[:, 1].min()) / max(arr[:, 1].ptp(), 1e-6)) * (size - 2 * margin)
        for x, y in zip(new_px, new_py):
            draw.line((x - 5, y - 5, x + 5, y + 5), fill=(20, 170, 80), width=2)
            draw.line((x - 5, y + 5, x + 5, y - 5), fill=(20, 170, 80), width=2)
    draw.text(
        (20, 20),
        "blue/red: teacher demand | orange: densify parents | green x: proposed new roots",
        fill=(0, 0, 0),
    )
    image.save(output / "gradient_parent_selection.jpg", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk")
    parser.add_argument("--mesh-path", default="D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj")
    parser.add_argument("--root-path", default="D:/petsgaussianhair/_downloads/root_init_20260623/white_tiger_roots_2048_fps.npz")
    parser.add_argument("--output-dir", default="D:/petsgaussianhair/_downloads/root_gsplat_gradient_densify_20260623")
    parser.add_argument("--root-count", type=int, default=768)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=13)
    parser.add_argument("--grad-threshold", type=float, default=1e-7)
    parser.add_argument("--max-new-roots", type=int, default=80)
    parser.add_argument("--view-count", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gsplat gradient densify test")
    device = torch.device("cuda")
    output = Path(args.output_dir)
    mesh_path = Path(args.mesh_path)
    data_root = Path(args.data_root)
    vertices, faces, face_ids, bary0, normals = load_root_data(Path(args.root_path), mesh_path, args.root_count, device)
    report, viewmats, ks = scaled_cameras(data_root, mesh_path, device, args.width, args.height)
    view_indices = report.train_indices[: args.view_count]

    bary_logits = torch.nn.Parameter(torch.log(bary0.clamp_min(1e-5)))
    root_bary = torch.softmax(bary_logits, dim=-1)
    root_points = barycentric_to_points(vertices, faces, face_ids, root_bary)
    root_points.retain_grad()

    teacher = GroomParameterField(args.root_count, device=device)
    demand = apply_regional_teacher(teacher, root_points.detach())
    student = GroomParameterField(args.root_count, device=device)

    with torch.no_grad():
        target_images, target_alphas, _, _, _ = render_with_gaussians(
            teacher,
            root_points.detach(),
            normals,
            viewmats,
            ks,
            view_indices,
            args.width,
            args.height,
            args.samples,
            args.min_segments,
            args.max_segments,
            retain_gaussian_grad=False,
        )
    pred_images, pred_alphas, gaussians, infos, render_stats = render_with_gaussians(
        student,
        root_points,
        normals,
        viewmats,
        ks,
        view_indices,
        args.width,
        args.height,
        args.samples,
        args.min_segments,
        args.max_segments,
        retain_gaussian_grad=True,
    )
    image_loss = torch.stack([F.mse_loss(pred, target) for pred, target in zip(pred_images, target_images)]).mean()
    alpha_loss = torch.stack([F.mse_loss(pred, target) for pred, target in zip(pred_alphas, target_alphas)]).mean()
    loss = image_loss + 0.2 * alpha_loss
    loss.backward()
    root_stats = aggregate_root_stats(root_points, gaussians, infos, image_loss)
    state = RootLifecycleState(points=root_points.detach(), face_ids=face_ids.detach(), barycentric=root_bary.detach())
    update = propose_structure_update(
        state,
        root_stats,
        DensifyConfig(
            grad_threshold=args.grad_threshold,
            visibility_threshold=1.0,
            max_new_roots=args.max_new_roots,
            children_per_parent=2,
            barycentric_step=0.08,
        ),
        PruneConfig(max_prune_fraction=0.0),
        vertices=vertices,
        faces=faces,
    )
    new_points = None
    if update.new_barycentric.numel() > 0:
        new_points = barycentric_to_points(vertices, faces, update.new_face_ids, update.new_barycentric)
    save_debug(output, root_points.detach(), demand, update.parent_indices, target_images[0], pred_images[0], new_points)
    parent_demand = demand[update.parent_indices].mean() if update.parent_indices.numel() else demand.new_tensor(0.0)
    all_need = update.scores["need"].detach()
    parent_need = all_need[update.parent_indices].mean() if update.parent_indices.numel() else all_need.new_tensor(0.0)
    summary = {
        "loss": float(loss.detach().cpu()),
        "image_mse": float(image_loss.detach().cpu()),
        "alpha_mse": float(alpha_loss.detach().cpu()),
        "root_count": int(args.root_count),
        "gaussian_count": int(gaussians.means.shape[0]),
        "selected_parent_count": int(update.parent_indices.numel()),
        "new_root_count": int(update.new_barycentric.shape[0]),
        "mean_demand_all": float(demand.mean().cpu()),
        "mean_demand_selected": float(parent_demand.detach().cpu()),
        "mean_need_all": float(all_need.mean().cpu()),
        "mean_need_selected": float(parent_need.detach().cpu()),
        "root_grad_mean": float(root_stats.root_grad_abs_sum.mean().detach().cpu()),
        "gaussian_grad_mean": float(root_stats.gaussian_grad_abs_sum.mean().detach().cpu()),
        "gaussian_contrib_mean": float(root_stats.gaussian_contrib_sum.mean().detach().cpu()),
        "render_stats": render_stats,
        "output_dir": str(output),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

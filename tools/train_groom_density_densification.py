"""Diagnostic validation for root densification on differentiable groom fields.

This is not the white-tiger training pipeline.  It stress-tests whether a
sparse projected groom can grow roots in high-density regions while preserving
the already validated strand-to-Gaussian optimization behavior.
"""

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
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from anigroom.grooming import GroomParameterField, GroomRanges  # noqa: E402
from train_dense_groom_patch_teacher_student import (  # noqa: E402
    dense_ranges,
    make_optimizer,
    make_student_like_dense,
)
from train_groom_layer_teacher_student import (  # noqa: E402
    RenderPack,
    build_render_inputs,
    gaussian_screen_orientation_colors,
    decode_orientation_render,
    loss_components,
    make_sheet,
    orientation_to_pil,
    psnr,
    set_range,
    to_pil,
)
from train_groom_style_stress_suite import enrich_teacher_style, initialize_student  # noqa: E402


ROOT_X_BOUNDS = (-0.72, 0.72)
ROOT_Y_BOUNDS = (-0.47, 0.47)


@dataclass
class SignalAccum:
    pressure: torch.Tensor
    contribution: torch.Tensor
    steps: int = 0


@dataclass
class RenderWithGaussians:
    pack: RenderPack
    gaussians: object
    segment_counts: torch.Tensor


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def make_grid_roots(
    device: torch.device,
    rows: int,
    cols: int,
    *,
    extent_x: float = 0.70,
    extent_y: float = 0.45,
    jitter: float = 0.0065,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.linspace(-extent_x, extent_x, cols, device=device)
    ys = torch.linspace(-extent_y, extent_y, rows, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    x = gx + jitter * torch.sin(37.0 * gx + 19.0 * gy)
    y = gy + jitter * torch.cos(23.0 * gx - 31.0 * gy)
    z = torch.full_like(x, 2.45)
    roots = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)
    normals = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3).expand_as(roots).contiguous()
    return roots, normals


def density_probability(roots: torch.Tensor, mode: str) -> torch.Tensor:
    x = roots[:, 0:1]
    y = roots[:, 1:2]
    if mode == "animal_density":
        stripe = torch.sigmoid(8.0 * (torch.sin(13.0 * x + 8.0 * y) - 0.08))
        ridge = torch.exp(-9.0 * (y - 0.08 * torch.sin(4.0 * x)).square())
        patch = torch.sigmoid(12.0 * (torch.sin(7.0 * x) * torch.cos(5.0 * y) - 0.10))
        prob = 0.12 + 0.52 * stripe + 0.38 * ridge + 0.20 * patch
        return prob.clamp(0.04, 0.98).reshape(-1)
    elif mode == "matted_density":
        mat = torch.sigmoid(10.0 * (torch.sin(9.0 * x - 7.0 * y) + 0.25 * torch.sin(25.0 * x) - 0.10))
        prob = 0.10 + 0.78 * mat
        return prob.clamp(0.04, 0.98).reshape(-1)
    elif mode == "prune_islands":
        stripe = torch.sigmoid(9.0 * (torch.sin(12.0 * x + 7.0 * y) - 0.10))
        island_a = torch.exp(-18.0 * ((x + 0.32).square() + 1.35 * (y - 0.12).square()))
        island_b = torch.exp(-16.0 * ((x - 0.28).square() + 1.10 * (y + 0.16).square()))
        island_c = torch.exp(-26.0 * ((x - 0.02).square() + 1.70 * (y - 0.30).square()))
        support = torch.clamp(island_a + island_b + 0.75 * island_c, 0.0, 1.0)
        prob = support * (0.18 + 0.78 * stripe)
        return prob.clamp(0.0, 0.98).reshape(-1)
    elif mode == "realistic_mixed":
        stripe = torch.sigmoid(9.0 * (torch.sin(17.5 * x + 7.0 * y + 0.4 * torch.sin(3.0 * y)) - 0.10))
        head = torch.sigmoid(14.0 * (-x - 0.34 + 0.08 * torch.sin(5.0 * y)))
        ridge = torch.exp(-18.0 * (y - 0.10 * torch.sin(3.5 * x)).square())
        flank = torch.sigmoid(10.0 * (x + 0.04)) * torch.sigmoid(10.0 * (0.32 - torch.abs(y + 0.02)))
        tangle = torch.exp(-24.0 * ((x - 0.30).square() + 1.55 * (y + 0.18).square()))
        sparse_belly = torch.sigmoid(16.0 * (-y - 0.30 + 0.04 * torch.sin(5.0 * x)))
        prob = 0.18 + 0.24 * stripe + 0.20 * head + 0.30 * ridge + 0.22 * flank + 0.28 * tangle - 0.12 * sparse_belly
        return prob.clamp(0.05, 0.98).reshape(-1)
    else:
        raise ValueError(f"unknown density mode: {mode}")


def root_detail_evidence(roots: torch.Tensor, mode: str) -> torch.Tensor:
    """Synthetic detail prior used only for controlled densification experiments.

    In real Stage 1 this role is played by image detail/head/boundary maps.
    Here we compute an analytic equivalent from the teacher field so the
    lifecycle test can answer whether detail-biased parent selection helps.
    """

    x = roots[:, 0:1]
    y = roots[:, 1:2]
    if mode == "animal_density":
        stripe_phase = torch.sin(13.0 * x + 8.0 * y)
        stripe_edge = torch.exp(-18.0 * (stripe_phase - 0.08).square())
        ridge = torch.exp(-12.0 * (y - 0.08 * torch.sin(4.0 * x)).square())
        patch_edge = torch.exp(-16.0 * (torch.sin(7.0 * x) * torch.cos(5.0 * y) - 0.10).square())
        detail = 0.58 * stripe_edge + 0.26 * ridge + 0.16 * patch_edge
    elif mode == "matted_density":
        mat_phase = torch.sin(9.0 * x - 7.0 * y) + 0.25 * torch.sin(25.0 * x)
        detail = torch.exp(-12.0 * (mat_phase - 0.10).square())
    elif mode == "prune_islands":
        support = density_probability(roots, mode).view(-1, 1)
        detail = torch.sqrt(support.clamp(0.0, 1.0))
    elif mode == "realistic_mixed":
        stripe_phase = torch.sin(17.5 * x + 7.0 * y + 0.4 * torch.sin(3.0 * y))
        stripe_edge = torch.exp(-16.0 * (stripe_phase - 0.10).square())
        ridge = torch.exp(-16.0 * (y - 0.10 * torch.sin(3.5 * x)).square())
        tangle = torch.exp(-22.0 * ((x - 0.30).square() + 1.55 * (y + 0.18).square()))
        head_boundary = torch.exp(-30.0 * (-x - 0.34 + 0.08 * torch.sin(5.0 * y)).square())
        detail = 0.36 * stripe_edge + 0.26 * ridge + 0.24 * tangle + 0.14 * head_boundary
    else:
        raise ValueError(f"unknown density mode: {mode}")
    detail = detail.reshape(-1)
    detail = detail / detail.max().clamp_min(1e-6)
    return detail.clamp(0.0, 1.0)


def deterministic_keep_mask(roots: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
    x = roots[:, 0]
    y = roots[:, 1]
    z = roots[:, 2]
    h = torch.sin(91.17 * x + 57.31 * y + 13.13 * z) * 43758.5453
    u = h - torch.floor(h)
    return u < probability


def make_density_teacher_roots(
    device: torch.device,
    rows: int,
    cols: int,
    density_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    roots, normals = make_grid_roots(device, rows, cols)
    prob = density_probability(roots, density_mode)
    keep = deterministic_keep_mask(roots, prob)
    if int(keep.sum().item()) < 16:
        raise RuntimeError("density target kept too few roots")
    return roots[keep].contiguous(), normals[keep].contiguous(), prob[keep].contiguous()


def render_with_gaussians(
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
) -> RenderWithGaussians:
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
    if gaussians.means.requires_grad:
        gaussians.means.retain_grad()

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
    pack = RenderPack(
        image=image[0].clamp(0.0, 1.0),
        alpha=alpha[0].clamp(0.0, 1.0),
        orientation=orientation,
        orientation_conf=orientation_conf,
        stats=stats,
    )
    return RenderWithGaussians(pack=pack, gaussians=gaussians, segment_counts=counts.detach())


def reset_accum(root_count: int, device: torch.device) -> SignalAccum:
    return SignalAccum(
        pressure=torch.zeros(root_count, device=device),
        contribution=torch.zeros(root_count, device=device),
        steps=0,
    )


def accumulate_root_signals(accum: SignalAccum, gaussians: object) -> None:
    if gaussians.means.grad is None:
        raise RuntimeError("Gaussian means gradients are missing; cannot densify from root pressure")
    root_ids = gaussians.root_indices.detach().long()
    if root_ids.numel() == 0:
        return
    grad_norm = gaussians.means.grad.detach().norm(dim=-1)
    contrib = gaussians.opacities.detach().reshape(-1).clamp_min(0.0)
    weighted_pressure = grad_norm * contrib.clamp_min(1e-4).sqrt()
    accum.pressure.scatter_add_(0, root_ids, weighted_pressure)
    accum.contribution.scatter_add_(0, root_ids, contrib)
    accum.steps += 1


def root_signal_scores(accum: SignalAccum) -> tuple[torch.Tensor, torch.Tensor]:
    if accum.steps <= 0:
        raise RuntimeError("no accumulated root signals")
    contribution = accum.contribution / float(accum.steps)
    pressure = accum.pressure / accum.contribution.clamp_min(1e-6)
    return pressure, contribution


def roots_from_xy(root_xy: torch.Tensor, z_value: float) -> torch.Tensor:
    z = torch.full((root_xy.shape[0], 1), float(z_value), device=root_xy.device, dtype=root_xy.dtype)
    return torch.cat([root_xy, z], dim=-1)


def clamp_root_xy_(root_xy: torch.Tensor) -> None:
    with torch.no_grad():
        root_xy[:, 0].clamp_(ROOT_X_BOUNDS[0], ROOT_X_BOUNDS[1])
        root_xy[:, 1].clamp_(ROOT_Y_BOUNDS[0], ROOT_Y_BOUNDS[1])


def copy_field_with_rows(
    old: GroomParameterField,
    row_values: dict[str, torch.Tensor],
    ranges: GroomRanges,
    device: torch.device,
) -> GroomParameterField:
    first = next(iter(row_values.values()))
    new = GroomParameterField(int(first.shape[0]), ranges=ranges, device=device)
    with torch.no_grad():
        new_params = dict(new.named_parameters())
        for name, value in row_values.items():
            if name not in new_params:
                raise KeyError(f"unknown groom parameter: {name}")
            if new_params[name].shape != value.shape:
                raise ValueError(f"shape mismatch for {name}: {tuple(new_params[name].shape)} vs {tuple(value.shape)}")
            new_params[name].copy_(value.to(device=device, dtype=new_params[name].dtype))
    return new


def interpolate_param_rows(field: GroomParameterField, parent_idx: torch.Tensor, neighbor_idx: torch.Tensor) -> dict[str, torch.Tensor]:
    params = dict(field.named_parameters())
    child_rows: dict[str, torch.Tensor] = {}
    for name, param in params.items():
        parent = param.detach()[parent_idx]
        neigh = param.detach()[neighbor_idx].mean(dim=1)
        if name == "curl_phase":
            neighbor_angles = param.detach()[neighbor_idx]
            parent_vec = torch.stack([torch.cos(parent), torch.sin(parent)], dim=-1)
            neigh_vec = torch.stack([torch.cos(neighbor_angles), torch.sin(neighbor_angles)], dim=-1).mean(dim=1)
            vec = 0.78 * parent_vec + 0.22 * neigh_vec
            child_rows[name] = torch.atan2(vec[..., 1], vec[..., 0])
        else:
            child_rows[name] = 0.78 * parent + 0.22 * neigh
    return child_rows


def split_roots(
    field: GroomParameterField,
    roots: torch.Tensor,
    normals: torch.Tensor,
    split_idx: torch.Tensor,
    ranges: GroomRanges,
    *,
    neighbor_count: int,
    split_scale: float,
) -> tuple[GroomParameterField, torch.Tensor, torch.Tensor]:
    if split_idx.numel() == 0:
        return field, roots, normals
    device = roots.device
    root_count = int(roots.shape[0])
    split_idx = split_idx.detach().long().unique(sorted=True)
    keep_mask = torch.ones(root_count, dtype=torch.bool, device=device)
    keep_mask[split_idx] = False
    keep_idx = torch.where(keep_mask)[0]

    xy = roots[:, :2]
    parent_xy = xy[split_idx]
    dist = torch.cdist(parent_xy, xy)
    dist[:, split_idx] = float("inf")
    k = min(max(1, int(neighbor_count)), max(1, root_count - 1))
    nn_dist, nn_idx = torch.topk(dist, k=k, largest=False, dim=1)
    local_radius = nn_dist[:, : min(4, k)].mean(dim=1, keepdim=True).clamp_min(0.004)

    groom = field.decode()
    flow = F.normalize(groom.flow_xy.detach()[split_idx], dim=-1, eps=1e-8)
    side = F.normalize(torch.stack([-flow[:, 1], flow[:, 0]], dim=-1), dim=-1, eps=1e-8)
    offsets = side * (local_radius * float(split_scale))
    parent_roots = roots[split_idx]
    child_a = parent_roots.clone()
    child_b = parent_roots.clone()
    child_a[:, :2] = child_a[:, :2] + offsets
    child_b[:, :2] = child_b[:, :2] - offsets
    child_a[:, 0].clamp_(ROOT_X_BOUNDS[0], ROOT_X_BOUNDS[1])
    child_a[:, 1].clamp_(ROOT_Y_BOUNDS[0], ROOT_Y_BOUNDS[1])
    child_b[:, 0].clamp_(ROOT_X_BOUNDS[0], ROOT_X_BOUNDS[1])
    child_b[:, 1].clamp_(ROOT_Y_BOUNDS[0], ROOT_Y_BOUNDS[1])

    row_values: dict[str, torch.Tensor] = {}
    params = dict(field.named_parameters())
    child_rows = interpolate_param_rows(field, split_idx, nn_idx)
    for name, param in params.items():
        kept = param.detach()[keep_idx]
        children = torch.cat([child_rows[name], child_rows[name]], dim=0)
        row_values[name] = torch.cat([kept, children], dim=0)

    new_roots = torch.cat([roots.detach()[keep_idx], child_a.detach(), child_b.detach()], dim=0).contiguous()
    new_normals = torch.cat([normals.detach()[keep_idx], normals.detach()[split_idx], normals.detach()[split_idx]], dim=0).contiguous()
    new_field = copy_field_with_rows(field, row_values, ranges, device)
    return new_field, new_roots, new_normals


def prune_roots(
    field: GroomParameterField,
    roots: torch.Tensor,
    normals: torch.Tensor,
    prune_mask: torch.Tensor,
    ranges: GroomRanges,
    min_roots: int,
) -> tuple[GroomParameterField, torch.Tensor, torch.Tensor, int]:
    root_count = int(roots.shape[0])
    if prune_mask.numel() != root_count:
        raise ValueError("prune mask size mismatch")
    if root_count - int(prune_mask.sum().item()) < int(min_roots):
        allowed = max(0, root_count - int(min_roots))
        if allowed <= 0:
            prune_mask = torch.zeros_like(prune_mask)
        else:
            prune_idx = torch.where(prune_mask)[0][:allowed]
            new_mask = torch.zeros_like(prune_mask)
            new_mask[prune_idx] = True
            prune_mask = new_mask
    removed = int(prune_mask.sum().item())
    if removed == 0:
        return field, roots, normals, 0
    keep_idx = torch.where(~prune_mask)[0]
    row_values = {name: param.detach()[keep_idx] for name, param in field.named_parameters()}
    new_field = copy_field_with_rows(field, row_values, ranges, roots.device)
    return new_field, roots.detach()[keep_idx].contiguous(), normals.detach()[keep_idx].contiguous(), removed


def make_optimizer_for(
    field: GroomParameterField,
    args: argparse.Namespace,
    root_xy: torch.nn.Parameter | None = None,
) -> torch.optim.Optimizer:
    optimizer = make_optimizer(field, args.lr, args.curl_lr_scale, args.flow_lr_scale)
    if root_xy is not None and float(args.root_lr) > 0.0:
        optimizer.add_param_group({"params": [root_xy], "lr": float(args.root_lr)})
    return optimizer


def local_flow_coherence_loss(field: GroomParameterField, roots: torch.Tensor, neighbor_count: int = 6) -> torch.Tensor:
    root_count = int(roots.shape[0])
    if root_count <= 2:
        return roots.new_tensor(0.0)
    k = min(max(1, int(neighbor_count)), root_count - 1)
    xy = roots[:, :2].detach()
    dist = torch.cdist(xy, xy)
    dist.fill_diagonal_(float("inf"))
    nn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices
    flow = F.normalize(field.decode().flow_xy, dim=-1, eps=1e-8)
    center = flow[:, None, :].expand(-1, k, -1)
    neigh = flow[nn_idx]
    return (1.0 - (center * neigh).sum(dim=-1).clamp(-1.0, 1.0)).mean()


def _normalize_range(value: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    lo, hi = bounds
    return ((value - lo) / max(hi - lo, 1e-8)).clamp(0.0, 1.0)


def local_groom_smoothness_loss(
    field: GroomParameterField,
    roots: torch.Tensor,
    *,
    neighbor_count: int = 6,
    detail_weights: torch.Tensor | None = None,
    detail_protection: float = 0.0,
) -> torch.Tensor:
    root_count = int(roots.shape[0])
    if root_count <= 2:
        return roots.new_tensor(0.0)
    k = min(max(1, int(neighbor_count)), root_count - 1)
    xy = roots[:, :2].detach()
    dist = torch.cdist(xy, xy)
    dist.fill_diagonal_(float("inf"))
    nn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices

    groom = field.decode()
    ranges = field.ranges
    flow = F.normalize(groom.flow_xy, dim=-1, eps=1e-8)
    channels = torch.cat(
        [
            _normalize_range(groom.length, ranges.length),
            _normalize_range(groom.root_width, ranges.root_width),
            _normalize_range(groom.tip_width / groom.root_width.clamp_min(1e-8), ranges.tip_width_ratio),
            _normalize_range(groom.width_taper, ranges.width_taper),
            0.5 + 0.5 * flow,
            _normalize_range(groom.flow_strength, ranges.flow_strength),
            _normalize_range(groom.lift, ranges.lift),
            0.5 + 0.5 * groom.bend,
            _normalize_range(groom.sag, ranges.sag),
            _normalize_range(groom.stiffness, ranges.stiffness),
            _normalize_range(groom.curl_radius, ranges.curl_radius),
            _normalize_range(groom.curl_frequency, ranges.curl_frequency),
            _normalize_range(groom.frizz, ranges.frizz),
            _normalize_range(groom.child_radius, ranges.child_radius),
            _normalize_range(groom.clump_strength, ranges.clump_strength),
            _normalize_range(groom.opacity, ranges.opacity),
            groom.root_color,
            groom.tip_color,
        ],
        dim=-1,
    )
    center = channels[:, None, :].expand(-1, k, -1)
    neigh = channels[nn_idx]
    diff = (center - neigh).square().mean(dim=-1)
    if detail_weights is not None and float(detail_protection) > 0.0:
        detail = detail_weights.to(device=roots.device, dtype=roots.dtype).reshape(-1).clamp(0.0, 1.0)
        edge_detail = torch.maximum(detail[:, None], detail[nn_idx])
        edge_weight = (1.0 - float(detail_protection) * edge_detail).clamp(0.10, 1.0)
        diff = diff * edge_weight
    return diff.mean()


def make_root_plot(
    path: Path,
    teacher_roots: torch.Tensor,
    initial_roots: torch.Tensor,
    final_roots: torch.Tensor,
    final_weights: torch.Tensor | None = None,
) -> None:
    width, height = 1200, 760
    pad = 64
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    font = load_font(22)
    title = "Root distribution: target density / initial sparse / final densified"
    if final_weights is not None:
        title += " (final marker strength = contribution)"
    draw.text((pad, 22), title, fill=(20, 20, 20), font=font)
    draw.rectangle((pad, pad, width - pad, height - pad), outline=(205, 205, 205), width=2)

    def project(points: torch.Tensor) -> list[tuple[float, float]]:
        xy = points.detach().cpu()
        x = (xy[:, 0] + 0.72) / 1.44
        y = 1.0 - (xy[:, 1] + 0.47) / 0.94
        return [(pad + float(px) * (width - 2 * pad), pad + float(py) * (height - 2 * pad)) for px, py in zip(x, y)]

    for x, y in project(teacher_roots):
        draw.ellipse((x - 1.1, y - 1.1, x + 1.1, y + 1.1), fill=(35, 35, 35))
    for x, y in project(initial_roots):
        draw.ellipse((x - 2.6, y - 2.6, x + 2.6, y + 2.6), outline=(30, 110, 230), width=1)
    final_xy = project(final_roots)
    if final_weights is None:
        for x, y in final_xy:
            draw.ellipse((x - 1.8, y - 1.8, x + 1.8, y + 1.8), outline=(220, 60, 40), width=1)
    else:
        w = final_weights.detach().reshape(-1).cpu()
        if w.numel() != len(final_xy):
            raise ValueError("final_weights must match final_roots")
        positive = w[w > 0]
        if positive.numel() > 0:
            scale = torch.quantile(positive, 0.92).clamp_min(1e-6)
        else:
            scale = torch.tensor(1.0)
        strength = (w / scale).clamp(0.0, 1.0)
        for (x, y), value in zip(final_xy, strength.tolist()):
            if value < 0.08:
                draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), outline=(175, 175, 175), width=1)
            else:
                radius = 1.4 + 2.0 * value
                red = int(150 + 90 * value)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(red, 42, 30), width=2)
    draw.text((pad, height - 46), f"target={teacher_roots.shape[0]}  initial={initial_roots.shape[0]}  final={final_roots.shape[0]}", fill=(20, 20, 20), font=font)
    image.save(path)


def root_contribution_from_gaussians(gaussians: object, root_count: int) -> torch.Tensor:
    root_ids = gaussians.root_indices.detach().long()
    contribution = torch.zeros(root_count, device=gaussians.opacities.device, dtype=gaussians.opacities.dtype)
    if root_ids.numel() == 0:
        return contribution
    contribution.scatter_add_(0, root_ids, gaussians.opacities.detach().reshape(-1).clamp_min(0.0))
    return contribution


def save_case_images(output_dir: Path, target: RenderPack, initial: RenderPack, final: RenderPack) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    diff = (final.image - target.image).abs().mul(3.0).clamp(0.0, 1.0)
    image_paths = [
        output_dir / "target.png",
        output_dir / "initial.png",
        output_dir / "final.png",
        output_dir / "diff_x3.png",
    ]
    for path, image in zip(image_paths, [target.image, initial.image, final.image, diff]):
        to_pil(image).save(path)
    make_sheet(image_paths, ["target", "initial", "final", "error x3"], output_dir / "training_sheet.png")

    orient_paths = [
        output_dir / "target_orientation.png",
        output_dir / "initial_orientation.png",
        output_dir / "final_orientation.png",
    ]
    orientation_to_pil(target.orientation, target.orientation_conf).save(orient_paths[0])
    orientation_to_pil(initial.orientation, initial.orientation_conf).save(orient_paths[1])
    orientation_to_pil(final.orientation, final.orientation_conf).save(orient_paths[2])
    make_sheet(orient_paths, ["target orientation", "initial orientation", "final orientation"], output_dir / "orientation_sheet.png")


def jsonable_config(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\petsgaussianhair\_downloads\groom_density_densification"))
    parser.add_argument("--density-mode", choices=["animal_density", "matted_density", "prune_islands", "realistic_mixed"], default="animal_density")
    parser.add_argument("--teacher-style", default="mixed_animal")
    parser.add_argument(
        "--init-mode",
        choices=["generic", "projected_flow", "projected_shape", "projected_curl", "projected_curve"],
        default="projected_curve",
    )
    parser.add_argument("--teacher-rows", type=int, default=64)
    parser.add_argument("--teacher-cols", type=int, default=96)
    parser.add_argument("--student-rows", type=int, default=18)
    parser.add_argument("--student-cols", type=int, default=28)
    parser.add_argument("--child-count", type=int, default=4)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=810)
    parser.add_argument("--focal", type=float, default=1725.0)
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--min-segments", type=int, default=18)
    parser.add_argument("--max-segments", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=640)
    parser.add_argument("--segment-warmup", type=int, default=60)
    parser.add_argument("--segment-refresh", type=int, default=40)
    parser.add_argument("--log-interval", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.014)
    parser.add_argument("--root-lr", type=float, default=0.0015)
    parser.add_argument("--orientation-weight", type=float, default=0.45)
    parser.add_argument("--orientation-detail-weight", type=float, default=0.35)
    parser.add_argument("--flow-coherence-weight", type=float, default=1.0)
    parser.add_argument("--groom-smooth-weight", type=float, default=0.0)
    parser.add_argument("--smooth-neighbor-count", type=int, default=6)
    parser.add_argument("--smooth-detail-protection", type=float, default=0.0)
    parser.add_argument("--curl-lr-scale", type=float, default=8.0)
    parser.add_argument("--flow-lr-scale", type=float, default=0.35)
    parser.add_argument("--densify-warmup", type=int, default=160)
    parser.add_argument("--densify-interval", type=int, default=120)
    parser.add_argument("--densify-until", type=int, default=520)
    parser.add_argument("--densify-score-threshold", type=float, default=2.5e-5)
    parser.add_argument("--densify-min-contribution", type=float, default=0.45)
    parser.add_argument("--densify-detail-bias", type=float, default=0.0)
    parser.add_argument("--max-splits-per-event", type=int, default=160)
    parser.add_argument("--split-neighbor-count", type=int, default=10)
    parser.add_argument("--split-scale", type=float, default=0.36)
    parser.add_argument("--prune-start", type=int, default=320)
    parser.add_argument("--prune-interval", type=int, default=160)
    parser.add_argument("--prune-min-contribution", type=float, default=0.08)
    parser.add_argument("--prune-opacity-threshold", type=float, default=0.12)
    parser.add_argument("--prune-pressure-threshold", type=float, default=1.0e-5)
    parser.add_argument("--min-roots", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; densification validation must use gsplat")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ranges = dense_ranges()
    teacher_roots, teacher_normals, _ = make_density_teacher_roots(
        device,
        args.teacher_rows,
        args.teacher_cols,
        args.density_mode,
    )
    teacher = enrich_teacher_style(args.teacher_style, teacher_roots, ranges).eval()
    student_roots, student_normals = make_grid_roots(device, args.student_rows, args.student_cols)
    initial_roots_for_plot = student_roots.detach().clone()
    root_z = float(student_roots[0, 2].detach().cpu())
    root_xy = torch.nn.Parameter(student_roots[:, :2].detach().clone())
    teacher_at_student = enrich_teacher_style(args.teacher_style, student_roots, ranges).eval()
    student = initialize_student(args.init_mode, student_roots, ranges, teacher_at_student)

    with torch.no_grad():
        target = render_with_gaussians(
            teacher,
            teacher_roots,
            teacher_normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            None,
            args.min_segments,
            args.max_segments,
        ).pack
        initial = render_with_gaussians(
            student,
            student_roots,
            student_normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            None,
            args.min_segments,
            args.max_segments,
        ).pack

    optimizer = make_optimizer_for(student, args, root_xy)
    segment_counts: torch.Tensor | None = None
    accum = reset_accum(int(student_roots.shape[0]), device)
    history: list[dict[str, float | int]] = []
    lifecycle: list[dict[str, float | int]] = []

    for iteration in range(1, args.iterations + 1):
        if iteration == 1:
            segment_counts = None
        elif iteration >= args.segment_warmup and (iteration - args.segment_warmup) % args.segment_refresh == 0:
            with torch.no_grad():
                _, segment_counts, _ = build_render_inputs(
                    student,
                    student_roots,
                    student_normals,
                    args.samples,
                    args.child_count,
                    None,
                    args.min_segments,
                    args.max_segments,
                )

        optimizer.zero_grad(set_to_none=True)
        student_roots = roots_from_xy(root_xy, root_z)
        rendered = render_with_gaussians(
            student,
            student_roots,
            student_normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            segment_counts,
            args.min_segments,
            args.max_segments,
        )
        comps = loss_components(rendered.pack, target, args.orientation_weight, args.orientation_detail_weight)
        detail_weights_for_roots = None
        if float(args.smooth_detail_protection) > 0.0:
            detail_weights_for_roots = root_detail_evidence(student_roots.detach(), args.density_mode)
        reg = local_flow_coherence_loss(student, student_roots, neighbor_count=args.smooth_neighbor_count)
        groom_smooth = local_groom_smoothness_loss(
            student,
            student_roots,
            neighbor_count=args.smooth_neighbor_count,
            detail_weights=detail_weights_for_roots,
            detail_protection=args.smooth_detail_protection,
        )
        loss = comps["total"] + float(args.flow_coherence_weight) * reg + float(args.groom_smooth_weight) * groom_smooth
        loss.backward()
        accumulate_root_signals(accum, rendered.gaussians)
        optimizer.step()
        if float(args.root_lr) > 0.0:
            clamp_root_xy_(root_xy)

        if iteration % args.log_interval == 0 or iteration == 1:
            entry = {
                "iter": iteration,
                "loss": float(loss.detach().cpu()),
                "psnr": psnr(rendered.pack.image, target.image),
                "rgb_l1": float(comps["rgb_l1"].detach().cpu()),
                "alpha_l1": float(comps["alpha_l1"].detach().cpu()),
                "orientation": float(comps["orientation"].detach().cpu()),
                "orientation_detail": float(comps["orientation_detail"].detach().cpu()),
                "flow_coherence": float(reg.detach().cpu()),
                "groom_smooth": float(groom_smooth.detach().cpu()),
                "root_count": int(student_roots.shape[0]),
                "gaussian_count": int(rendered.pack.stats["gaussian_count"]),
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)

        should_densify = (
            iteration >= args.densify_warmup
            and iteration <= args.densify_until
            and (iteration - args.densify_warmup) % args.densify_interval == 0
        )
        should_prune = iteration >= args.prune_start and (iteration - args.prune_start) % args.prune_interval == 0
        if should_densify or should_prune:
            pressure, contribution = root_signal_scores(accum)
            pressure_cpu = pressure.detach().cpu()
            contrib_cpu = contribution.detach().cpu()
            signal_report = {
                "iter": iteration,
                "pressure_mean": float(pressure_cpu.mean()),
                "pressure_max": float(pressure_cpu.max()),
                "contribution_mean": float(contrib_cpu.mean()),
                "contribution_max": float(contrib_cpu.max()),
                "roots_before": int(student_roots.shape[0]),
            }

            did_change = False
            densified = False
            if should_densify:
                student_roots = roots_from_xy(root_xy, root_z).detach()
                eligible = (pressure > float(args.densify_score_threshold)) & (contribution > float(args.densify_min_contribution))
                split_idx = torch.where(eligible)[0]
                if split_idx.numel() > int(args.max_splits_per_event):
                    score = pressure[split_idx]
                    if float(args.densify_detail_bias) > 0.0:
                        detail = root_detail_evidence(student_roots, args.density_mode)
                        score = score * (1.0 + float(args.densify_detail_bias) * detail[split_idx])
                        signal_report["detail_bias_mean"] = float(detail[split_idx].detach().mean().cpu())
                    keep = torch.topk(score, k=int(args.max_splits_per_event), largest=True).indices
                    split_idx = split_idx[keep]
                split_idx = split_idx.detach().long().unique(sorted=True)
                old_root_count = int(student_roots.shape[0])
                keep_mask = torch.ones(old_root_count, dtype=torch.bool, device=device)
                keep_mask[split_idx] = False
                keep_idx = torch.where(keep_mask)[0]
                student, student_roots, student_normals = split_roots(
                    student,
                    student_roots,
                    student_normals,
                    split_idx,
                    ranges,
                    neighbor_count=args.split_neighbor_count,
                    split_scale=args.split_scale,
                )
                if split_idx.numel() > 0:
                    pressure = torch.cat([pressure[keep_idx], pressure[split_idx], pressure[split_idx]], dim=0)
                    contribution = torch.cat([contribution[keep_idx], contribution[split_idx], contribution[split_idx]], dim=0)
                signal_report["splits"] = int(split_idx.numel())
                densified = split_idx.numel() > 0
                did_change = did_change or densified

            if should_prune:
                if not densified:
                    student_roots = roots_from_xy(root_xy, root_z).detach()
                opacity = student.decode().opacity.detach().reshape(-1)
                prune_mask = (
                    (contribution < float(args.prune_min_contribution))
                    & (pressure < float(args.prune_pressure_threshold))
                    & (opacity < float(args.prune_opacity_threshold))
                )
                student, student_roots, student_normals, removed = prune_roots(
                    student,
                    student_roots,
                    student_normals,
                    prune_mask,
                    ranges,
                    min_roots=args.min_roots,
                )
                signal_report["pruned"] = int(removed)
                did_change = did_change or removed > 0

            signal_report["roots_after"] = int(student_roots.shape[0])
            lifecycle.append(signal_report)
            print(json.dumps({"lifecycle": signal_report}), flush=True)

            if did_change:
                root_xy = torch.nn.Parameter(student_roots[:, :2].detach().clone())
                optimizer = make_optimizer_for(student, args, root_xy)
                segment_counts = None
            accum = reset_accum(int(student_roots.shape[0]), device)

    with torch.no_grad():
        student_roots = roots_from_xy(root_xy, root_z)
        final_render = render_with_gaussians(
            student,
            student_roots,
            student_normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            None,
            args.min_segments,
            args.max_segments,
        )
        final = final_render.pack
        final_contribution = root_contribution_from_gaussians(final_render.gaussians, int(student_roots.shape[0])).detach()
    save_case_images(args.output_dir, target, initial, final)
    make_root_plot(args.output_dir / "root_distribution.png", teacher_roots, initial_roots_for_plot, student_roots)
    make_root_plot(
        args.output_dir / "effective_root_distribution.png",
        teacher_roots,
        initial_roots_for_plot,
        student_roots,
        final_weights=final_contribution,
    )
    torch.save(
        {
            "teacher_roots": teacher_roots.detach().cpu(),
            "initial_roots": initial_roots_for_plot.detach().cpu(),
            "final_roots": student_roots.detach().cpu(),
            "final_contribution": final_contribution.detach().cpu(),
            "lifecycle": lifecycle,
            "history": history,
            "config": jsonable_config(args),
        },
        args.output_dir / "root_lifecycle_debug.pt",
    )

    summary = {
        "purpose": "diagnostic densification validation; not a production training recipe",
        "target_roots": int(teacher_roots.shape[0]),
        "initial_roots": int(initial_roots_for_plot.shape[0]),
        "final_roots": int(student_roots.shape[0]),
        "initial_psnr": psnr(initial.image, target.image),
        "final_psnr": psnr(final.image, target.image),
        "history": history,
        "lifecycle": lifecycle,
        "config": jsonable_config(args),
        "outputs": {
            "training_sheet": str(args.output_dir / "training_sheet.png"),
            "orientation_sheet": str(args.output_dir / "orientation_sheet.png"),
            "root_distribution": str(args.output_dir / "root_distribution.png"),
            "effective_root_distribution": str(args.output_dir / "effective_root_distribution.png"),
            "root_lifecycle_debug": str(args.output_dir / "root_lifecycle_debug.pt"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

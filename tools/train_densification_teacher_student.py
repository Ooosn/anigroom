"""Formal teacher-student benchmark for AniGroom root densification/pruning.

This script validates the structure-update module without white-tiger image
supervision.  Targets are rendered by dense teacher grooms.  Students start
from sparse mesh roots and must recover the teacher render through real
grooming -> Gaussian -> gsplat rendering, with optional multi-round
densification/pruning driven by accumulated gradients and visibility.

No fallback path is allowed: missing gsplat radii, missing gradients, or invalid
root/attribute alignment raises an error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
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
    GroomRanges,
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
    RootStatsWindow,
    apply_attribute_update,
    apply_structure_update,
    interpolate_child_attributes,
    propose_structure_update,
)
from anigroom.roots.lifecycle import barycentric_to_points


@dataclass(frozen=True)
class RunConfig:
    data_root: str
    mesh_path: str
    root_path: str
    output_dir: str
    scenarios: tuple[str, ...]
    modes: tuple[str, ...]
    teacher_roots: int
    student_roots: int
    max_roots: int
    width: int
    height: int
    views: tuple[int, ...]
    iterations: int
    eval_every: int
    log_every: int
    warmup: int
    densify_interval: int
    prune_start: int
    prune_interval: int
    grad_threshold: float
    visibility_threshold: float
    min_contribution: float
    min_opacity: float
    max_new_roots: int
    max_prune_fraction: float
    children_per_parent: int
    candidate_rings: int
    candidate_face_count: int
    neighbor_count: int
    attribute_neighbor_count: int
    min_child_distance: float
    samples: int
    min_segments: int
    max_segments: int
    lr_field: float
    lr_root: float
    alpha_weight: float
    seed: int
    target_only: bool


def parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def parse_csv_strings(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def clean_groom_ranges() -> GroomRanges:
    """Ranges for clean high-contrast teacher-student strand targets."""

    return GroomRanges(
        length=(0.015, 0.110),
        root_width=(0.00018, 0.0024),
        tip_width_ratio=(0.05, 0.42),
        flow_strength=(0.05, 1.10),
        lift=(0.04, 0.55),
        sag=(0.00, 0.85),
        stiffness=(0.05, 0.98),
        opacity=(0.72, 0.99),
    )


def make_groom_field(root_count: int, device: torch.device) -> GroomParameterField:
    return GroomParameterField(
        root_count,
        ranges=clean_groom_ranges(),
        init_root_color=(0.98, 0.82, 0.42),
        init_tip_color=(1.00, 0.93, 0.58),
        device=device,
    )


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0], mode="L")
    return Image.fromarray(arr, mode="RGB")


def save_contact_sheet(items: list[tuple[str, Image.Image]], output: Path, tile_size: tuple[int, int] = (360, 260)) -> None:
    if not items:
        return
    label_h = 28
    cols = len(items)
    sheet = Image.new("RGB", (cols * tile_size[0], tile_size[1] + label_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(items):
        image = image.convert("RGB")
        image.thumbnail(tile_size, Image.Resampling.LANCZOS)
        x0 = idx * tile_size[0]
        sheet.paste(image, (x0 + (tile_size[0] - image.width) // 2, label_h + (tile_size[1] - image.height) // 2))
        draw.text((x0 + 8, 8), label, fill=(20, 20, 20))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)


def face_normals(vertices: torch.Tensor, faces: torch.Tensor, face_ids: torch.Tensor) -> torch.Tensor:
    tri = vertices[faces[face_ids]]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return F.normalize(normals, dim=-1)


def load_roots(root_path: Path, mesh_path: Path, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(root_path)
    if "face_ids" not in data or "barycentric" not in data:
        raise RuntimeError(f"root file must contain face_ids and barycentric: {root_path}")
    if data["face_ids"].shape[0] < count:
        raise RuntimeError(f"root file has {data['face_ids'].shape[0]} roots, requested {count}")
    mesh = read_obj_mesh(mesh_path)
    vertices = torch.from_numpy(mesh.vertices).to(device=device)
    faces = torch.from_numpy(mesh.faces).to(device=device)
    face_ids = torch.from_numpy(data["face_ids"][:count].astype(np.int64)).to(device=device)
    bary = torch.from_numpy(data["barycentric"][:count].astype(np.float32)).to(device=device)
    points = barycentric_to_points(vertices, faces, face_ids, bary)
    return vertices, faces, face_ids, bary, points


def scaled_cameras(
    data_root: Path,
    mesh_path: Path,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    report = build_stage1_input_report(data_root, mesh_path)
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    source_w, source_h = report.image_size or (0, 0)
    if source_w <= 0 or source_h <= 0:
        raise RuntimeError("invalid source image size")
    target_w = int(width) if int(width) > 0 else int(source_w)
    target_h = int(height) if int(height) > 0 else int(source_h)
    viewmats, ks = load_camera_tensors(data_root, device)
    ks = ks.clone()
    sx = float(target_w) / float(source_w)
    sy = float(target_h) / float(source_h)
    ks[:, 0, 0] *= sx
    ks[:, 1, 1] *= sy
    ks[:, 0, 2] *= sx
    ks[:, 1, 2] *= sy
    return viewmats, ks, target_w, target_h


def apply_scenario(field: GroomParameterField, roots: torch.Tensor, scenario: str) -> None:
    """Create dense teacher targets with different capacity demands."""

    with torch.no_grad():
        x = roots[:, [0]]
        y = roots[:, [1]]
        z = roots[:, [2]]
        xn = (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)
        yn = (y - y.min()) / (y.max() - y.min()).clamp_min(1e-6)
        zn = (z - z.min()) / (z.max() - z.min()).clamp_min(1e-6)
        stripe = torch.sigmoid(7.0 * (torch.sin(36.0 * zn + 5.0 * torch.sin(8.0 * yn)) - 0.15))

        if scenario == "density_detail":
            detail = torch.exp(-((xn - 0.58).square() + 0.65 * (yn - 0.55).square() + 0.45 * (zn - 0.45).square()) / 0.035)
            field.length_raw.add_(1.20 + 2.40 * detail)
            field.root_width_raw.add_(1.20 + 1.10 * detail)
            field.tip_width_ratio_raw.add_(0.50)
            field.opacity_raw.add_(0.90 + 0.85 * detail)
            field.bend_raw.add_(0.60 * detail)
        elif scenario == "flow_bend":
            wave = torch.sin(12.0 * zn + 6.0 * yn)
            field.length_raw.add_(1.30 + 0.55 * wave)
            field.flow_xy[:, 0:1].add_(1.15 * torch.sin(9.0 * zn))
            field.flow_xy[:, 1:2].add_(1.35 * torch.cos(8.0 * xn))
            field.bend_raw.add_(2.20 * torch.sin(10.0 * zn))
            field.sag_raw.add_(1.10 * torch.sigmoid(7.0 * (0.50 - yn)))
            field.stiffness_raw.sub_(0.85)
            field.opacity_raw.add_(0.70)
        elif scenario == "prune_hole":
            fur_region = torch.sigmoid(18.0 * (xn - 0.18)) * torch.sigmoid(18.0 * (0.92 - xn))
            fur_region = fur_region * torch.sigmoid(16.0 * (yn - 0.10))
            no_fur = 1.0 - fur_region
            field.length_raw.add_(1.45 * fur_region - 4.20 * no_fur)
            field.root_width_raw.add_(1.10 * fur_region - 2.80 * no_fur)
            field.opacity_raw.add_(1.30 * fur_region - 5.50 * no_fur)
            field.flow_xy[:, 0:1].add_(0.80 * fur_region)
            field.sag_raw.add_(0.50 * fur_region)
        elif scenario == "mixed":
            detail = torch.exp(-((xn - 0.62).square() + (yn - 0.42).square() + 0.55 * (zn - 0.38).square()) / 0.028)
            flow = torch.sin(13.0 * zn + 4.0 * yn)
            no_fur = torch.sigmoid(20.0 * (0.12 - yn)) + torch.sigmoid(20.0 * (xn - 0.94))
            no_fur = no_fur.clamp(0.0, 1.0)
            field.length_raw.add_(1.25 + 1.90 * detail + 0.45 * flow - 3.50 * no_fur)
            field.root_width_raw.add_(1.15 + 0.90 * detail - 2.30 * no_fur)
            field.opacity_raw.add_(1.00 + 0.70 * detail - 4.50 * no_fur)
            field.flow_xy[:, 0:1].add_(1.05 * torch.sin(10.0 * zn))
            field.flow_xy[:, 1:2].add_(1.00 * torch.cos(7.0 * xn))
            field.bend_raw.add_(1.70 * flow)
            field.sag_raw.add_(0.95 * torch.sigmoid(8.0 * (0.52 - yn)))
            field.stiffness_raw.sub_(0.65)
        else:
            raise ValueError(f"unknown scenario: {scenario}")

        dark = stripe.expand(-1, 3)
        field.root_color_raw.add_(-2.35 * dark)
        field.tip_color_raw.add_(-2.00 * dark)


def render_groom(
    field: GroomParameterField,
    root_points: torch.Tensor,
    normals: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    view_indices: list[int],
    width: int,
    height: int,
    samples: int,
    min_segments: int,
    max_segments: int,
    *,
    retain_grad: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor], object, list[dict], dict[str, float | int]]:
    from gsplat.rendering import rasterization

    tangents, bitangents = make_tangent_frames(normals)
    groom = field.decode()
    strands, widths, colors, opacities = build_strands(root_points, normals, tangents, bitangents, groom, samples=samples)
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
    background = torch.tensor([[0.07, 0.08, 0.09]], device=root_points.device, dtype=root_points.dtype)
    images: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    infos: list[dict] = []
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
        if not isinstance(info, dict) or "radii" not in info:
            raise RuntimeError("gsplat did not return radii; formal visibility accumulation cannot proceed")
        images.append(render[0].clamp(0.0, 1.0))
        alphas.append(alpha[0].clamp(0.0, 1.0))
        infos.append(info)
    stats = {
        **resampled.stats,
        "gaussian_count": int(gaussians.means.shape[0]),
        "root_count": int(root_points.shape[0]),
    }
    return images, alphas, gaussians, infos, stats


def image_loss(
    pred_images: list[torch.Tensor],
    pred_alphas: list[torch.Tensor],
    target_images: list[torch.Tensor],
    target_alphas: list[torch.Tensor],
    alpha_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = torch.stack([F.mse_loss(pred, target) for pred, target in zip(pred_images, target_images)]).mean()
    alpha = torch.stack([F.mse_loss(pred, target) for pred, target in zip(pred_alphas, target_alphas)]).mean()
    return rgb + float(alpha_weight) * alpha, rgb, alpha


def psnr_from_mse(mse: float) -> float:
    return float(-10.0 * math.log10(max(float(mse), 1e-12)))


def make_optimizer(field: GroomParameterField, bary_logits: torch.nn.Parameter, cfg: RunConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {"params": list(field.parameters()), "lr": cfg.lr_field},
            {"params": [bary_logits], "lr": cfg.lr_root},
        ]
    )


def field_with_updated_attributes(
    field: GroomParameterField,
    state: RootLifecycleState,
    update,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    neighbor_count: int,
) -> GroomParameterField:
    new_count = int(state.points.shape[0] + update.new_barycentric.shape[0] - update.prune_mask.sum().item())
    out = GroomParameterField(
        new_count,
        ranges=field.ranges,
        init_root_color=(0.98, 0.82, 0.42),
        init_tip_color=(1.00, 0.93, 0.58),
        device=state.points.device,
    )
    for name, param in field.named_parameters():
        child = interpolate_child_attributes(param.detach(), state, update, vertices, faces, neighbor_count=neighbor_count, parent_weight=3.0)
        updated = apply_attribute_update(param.detach(), update, child)
        setattr(out, name, torch.nn.Parameter(updated.clone()))
    out.root_count = new_count
    return out


def state_from_logits(vertices: torch.Tensor, faces: torch.Tensor, face_ids: torch.Tensor, bary_logits: torch.Tensor) -> tuple[RootLifecycleState, torch.Tensor]:
    bary = torch.softmax(bary_logits, dim=-1)
    points = barycentric_to_points(vertices, faces, face_ids, bary)
    return RootLifecycleState(points=points, face_ids=face_ids, barycentric=bary), points


def save_root_overlay(points: torch.Tensor, output: Path, title: str) -> None:
    arr = points.detach().cpu().numpy()
    if arr.size == 0:
        return
    size = (920, 520)
    margin = 28
    image = Image.new("RGB", size, (246, 246, 246))
    draw = ImageDraw.Draw(image)
    x = arr[:, 2]
    y = arr[:, 1]
    px = margin + (x - x.min()) / max(float(np.ptp(x)), 1e-6) * (size[0] - 2 * margin)
    py = margin + (1.0 - (y - y.min()) / max(float(np.ptp(y)), 1e-6)) * (size[1] - 2 * margin)
    for xx, yy in zip(px, py):
        draw.ellipse((xx - 1.3, yy - 1.3, xx + 1.3, yy + 1.3), fill=(20, 80, 210))
    draw.text((12, 10), title, fill=(20, 20, 20))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)


def evaluate_render(
    field: GroomParameterField,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    face_ids: torch.Tensor,
    bary_logits: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    view_indices: list[int],
    targets: tuple[list[torch.Tensor], list[torch.Tensor]],
    cfg: RunConfig,
) -> dict[str, float]:
    with torch.no_grad():
        bary = torch.softmax(bary_logits, dim=-1)
        points = barycentric_to_points(vertices, faces, face_ids, bary)
        normals = face_normals(vertices, faces, face_ids)
        images, alphas, _, _, stats = render_groom(
            field,
            points,
            normals,
            viewmats,
            ks,
            view_indices,
            cfg.width,
            cfg.height,
            cfg.samples,
            cfg.min_segments,
            cfg.max_segments,
            retain_grad=False,
        )
        loss, rgb, alpha = image_loss(images, alphas, targets[0], targets[1], cfg.alpha_weight)
    return {
        "loss": float(loss.detach().cpu()),
        "rgb_mse": float(rgb.detach().cpu()),
        "alpha_mse": float(alpha.detach().cpu()),
        "psnr": psnr_from_mse(float(rgb.detach().cpu())),
        "root_count": int(points.shape[0]),
        "gaussian_count": int(stats["gaussian_count"]),
    }


def train_one_mode(
    *,
    scenario: str,
    mode: str,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    root_face_ids_all: torch.Tensor,
    root_bary_all: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    target_images: list[torch.Tensor],
    target_alphas: list[torch.Tensor],
    cfg: RunConfig,
    output_dir: Path,
) -> dict[str, object]:
    if mode not in {"sparse_fixed", "densify_prune", "dense_fixed"}:
        raise ValueError(f"unknown mode: {mode}")
    root_count = cfg.teacher_roots if mode == "dense_fixed" else cfg.student_roots
    face_ids = root_face_ids_all[:root_count].detach().clone()
    bary_logits = torch.nn.Parameter(torch.log(root_bary_all[:root_count].detach().clone().clamp_min(1e-6)))
    field = make_groom_field(root_count, vertices.device)
    optimizer = make_optimizer(field, bary_logits, cfg)
    accumulator = RootStatsWindow(root_count, vertices.device)
    log_path = output_dir / f"{scenario}_{mode}_metrics.jsonl"
    event_path = output_dir / f"{scenario}_{mode}_events.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    view_indices = list(cfg.views)
    targets = (target_images, target_alphas)
    start_time = time.time()
    history: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    with log_path.open("w", encoding="utf-8") as log_file, event_path.open("w", encoding="utf-8") as event_file:
        for iteration in range(1, cfg.iterations + 1):
            view_idx = view_indices[(iteration - 1) % len(view_indices)]
            state, root_points = state_from_logits(vertices, faces, face_ids, bary_logits)
            root_points.retain_grad()
            normals = face_normals(vertices, faces, face_ids)
            pred_images, pred_alphas, gaussians, infos, render_stats = render_groom(
                field,
                root_points,
                normals,
                viewmats,
                ks,
                [view_idx],
                cfg.width,
                cfg.height,
                cfg.samples,
                cfg.min_segments,
                cfg.max_segments,
                retain_grad=True,
            )
            target_pos = view_indices.index(view_idx)
            loss, rgb, alpha = image_loss(pred_images, pred_alphas, [target_images[target_pos]], [target_alphas[target_pos]], cfg.alpha_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if mode == "densify_prune":
                accumulator.add(root_points=root_points, gaussians=gaussians, infos=infos, residual_loss=None)
            optimizer.step()

            if iteration == 1 or iteration % cfg.log_every == 0 or iteration == cfg.iterations:
                record = {
                    "scenario": scenario,
                    "mode": mode,
                    "iteration": iteration,
                    "loss": float(loss.detach().cpu()),
                    "rgb_mse": float(rgb.detach().cpu()),
                    "alpha_mse": float(alpha.detach().cpu()),
                    "psnr": psnr_from_mse(float(rgb.detach().cpu())),
                    "root_count": int(face_ids.shape[0]),
                    "gaussian_count": int(render_stats["gaussian_count"]),
                    "elapsed_sec": float(time.time() - start_time),
                }
                history.append(record)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record), flush=True)

            allow_structure = (
                mode == "densify_prune"
                and iteration >= cfg.warmup
                and iteration % cfg.densify_interval == 0
            )
            if allow_structure:
                stats = accumulator.to_stats()
                stats_summary = asdict(accumulator.summary())
                remaining_capacity = max(0, int(cfg.max_roots) - int(face_ids.shape[0]))
                max_new = min(int(cfg.max_new_roots), remaining_capacity)
                prune_fraction = cfg.max_prune_fraction if iteration >= cfg.prune_start and iteration % cfg.prune_interval == 0 else 0.0
                state_eval, _ = state_from_logits(vertices, faces, face_ids, bary_logits.detach())
                update = propose_structure_update(
                    state_eval,
                    stats,
                    DensifyConfig(
                        grad_threshold=cfg.grad_threshold,
                        visibility_threshold=cfg.visibility_threshold,
                        max_new_roots=max_new,
                        children_per_parent=cfg.children_per_parent,
                        replace_parent=True,
                        neighbor_count=cfg.neighbor_count,
                        candidate_rings=cfg.candidate_rings,
                        candidate_face_count=cfg.candidate_face_count,
                        min_child_distance=cfg.min_child_distance,
                    ),
                    PruneConfig(
                        min_visible_count=cfg.visibility_threshold,
                        min_contribution=cfg.min_contribution,
                        min_opacity=cfg.min_opacity,
                        max_prune_fraction=prune_fraction,
                    ),
                    vertices=vertices,
                    faces=faces,
                )
                before_count = int(face_ids.shape[0])
                parent_replace_mask = torch.zeros_like(update.prune_mask)
                if update.parent_indices.numel() > 0:
                    parent_replace_mask[update.parent_indices] = True
                independent_prune_mask = update.prune_mask & ~parent_replace_mask
                if update.new_barycentric.shape[0] > 0 or update.prune_mask.any():
                    field = field_with_updated_attributes(field, state_eval, update, vertices, faces, cfg.attribute_neighbor_count)
                    next_state = apply_structure_update(state_eval, update, vertices, faces)
                    face_ids = next_state.face_ids.detach().clone()
                    bary_logits = torch.nn.Parameter(torch.log(next_state.barycentric.detach().clone().clamp_min(1e-6)))
                    optimizer = make_optimizer(field, bary_logits, cfg)
                    accumulator = RootStatsWindow(int(face_ids.shape[0]), vertices.device)
                else:
                    accumulator.reset()
                event = {
                    "scenario": scenario,
                    "mode": mode,
                    "iteration": iteration,
                    "before_root_count": before_count,
                    "after_root_count": int(face_ids.shape[0]),
                    "selected_parent_count": int(update.parent_indices.numel()),
                    "new_root_count": int(update.new_barycentric.shape[0]),
                    "prune_count": int(update.prune_mask.sum().item()),
                    "parent_replace_prune_count": int((update.prune_mask & parent_replace_mask).sum().item()),
                    "independent_prune_count": int(independent_prune_mask.sum().item()),
                    "stats": stats_summary,
                    "need_quantiles": {
                        "q50": float(torch.quantile(update.scores["need"].detach(), 0.50).cpu()),
                        "q90": float(torch.quantile(update.scores["need"].detach(), 0.90).cpu()),
                        "q95": float(torch.quantile(update.scores["need"].detach(), 0.95).cpu()),
                        "q99": float(torch.quantile(update.scores["need"].detach(), 0.99).cpu()),
                        "max": float(update.scores["need"].detach().max().cpu()),
                    },
                    "selected_need": {
                        "min": float(update.scores["need"][update.parent_indices].detach().min().cpu()) if update.parent_indices.numel() else None,
                        "mean": float(update.scores["need"][update.parent_indices].detach().mean().cpu()) if update.parent_indices.numel() else None,
                        "max": float(update.scores["need"][update.parent_indices].detach().max().cpu()) if update.parent_indices.numel() else None,
                    },
                    "candidate_count": int(((update.scores["visibility"] >= cfg.visibility_threshold) & (update.scores["need"] >= cfg.grad_threshold)).sum().detach().cpu()),
                    "grad_threshold": cfg.grad_threshold,
                    "visibility_threshold": cfg.visibility_threshold,
                    "min_contribution": cfg.min_contribution,
                    "min_opacity": cfg.min_opacity,
                }
                events.append(event)
                event_file.write(json.dumps(event) + "\n")
                event_file.flush()
                print(json.dumps(event), flush=True)

    final_metrics = evaluate_render(field, vertices, faces, face_ids, bary_logits, viewmats, ks, view_indices, targets, cfg)
    final_state, _ = state_from_logits(vertices, faces, face_ids, bary_logits.detach())
    save_root_overlay(final_state.points, output_dir / f"{scenario}_{mode}_roots.jpg", f"{scenario} {mode}: {final_metrics['root_count']} roots")
    with torch.no_grad():
        normals = face_normals(vertices, faces, face_ids)
        images, _, _, _, _ = render_groom(
            field,
            final_state.points,
            normals,
            viewmats,
            ks,
            [view_indices[0]],
            cfg.width,
            cfg.height,
            cfg.samples,
            cfg.min_segments,
            cfg.max_segments,
            retain_grad=False,
        )
    save_contact_sheet(
        [
            ("target", tensor_to_image(target_images[0])),
            (mode, tensor_to_image(images[0])),
        ],
        output_dir / f"{scenario}_{mode}_render.jpg",
    )
    return {
        "scenario": scenario,
        "mode": mode,
        "final": final_metrics,
        "history": history,
        "events": events,
    }


def run(cfg: RunConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal densification teacher-student training")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda")
    mesh_path = Path(cfg.mesh_path)
    vertices, faces, root_face_ids_all, root_bary_all, teacher_points = load_roots(Path(cfg.root_path), mesh_path, cfg.teacher_roots, device)
    viewmats, ks, target_w, target_h = scaled_cameras(Path(cfg.data_root), mesh_path, cfg.width, cfg.height, device)
    cfg = replace(cfg, width=target_w, height=target_h)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
    if not cfg.views:
        raise RuntimeError("at least one view index is required")
    if max(cfg.views) >= viewmats.shape[0] or min(cfg.views) < 0:
        raise RuntimeError(f"view indices out of range: {cfg.views}")
    view_indices = list(cfg.views)

    summary: dict[str, object] = {"config": asdict(cfg), "runs": []}
    for scenario in cfg.scenarios:
        teacher_field = make_groom_field(cfg.teacher_roots, device)
        apply_scenario(teacher_field, teacher_points.detach(), scenario)
        teacher_normals = face_normals(vertices, faces, root_face_ids_all[: cfg.teacher_roots])
        with torch.no_grad():
            target_images, target_alphas, _, _, _ = render_groom(
                teacher_field,
                teacher_points,
                teacher_normals,
                viewmats,
                ks,
                view_indices,
                cfg.width,
                cfg.height,
                cfg.samples,
                cfg.min_segments,
                cfg.max_segments,
                retain_grad=False,
            )
        save_contact_sheet(
            [(f"target view {idx}", tensor_to_image(img)) for idx, img in zip(view_indices[:3], target_images[:3])],
            output_dir / f"{scenario}_teacher_targets.jpg",
        )
        for idx, img in zip(view_indices, target_images):
            tensor_to_image(img).save(output_dir / f"{scenario}_teacher_view_{idx:03d}.png")
        if cfg.target_only:
            continue
        for mode in cfg.modes:
            result = train_one_mode(
                scenario=scenario,
                mode=mode,
                vertices=vertices,
                faces=faces,
                root_face_ids_all=root_face_ids_all,
                root_bary_all=root_bary_all,
                viewmats=viewmats,
                ks=ks,
                target_images=target_images,
                target_alphas=target_alphas,
                cfg=cfg,
                output_dir=output_dir,
            )
            summary["runs"].append(result)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk")
    parser.add_argument("--mesh-path", default="D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj")
    parser.add_argument("--root-path", default="D:/petsgaussianhair/_downloads/root_init_20260623/white_tiger_roots_2048_fps.npz")
    parser.add_argument("--output-dir", default="D:/petsgaussianhair/_downloads/densification_teacher_student_formal")
    parser.add_argument("--scenarios", default="density_detail,flow_bend,prune_hole,mixed")
    parser.add_argument("--modes", default="sparse_fixed,densify_prune,dense_fixed")
    parser.add_argument("--teacher-roots", type=int, default=1024)
    parser.add_argument("--student-roots", type=int, default=128)
    parser.add_argument("--max-roots", type=int, default=768)
    parser.add_argument("--width", type=int, default=0, help="render width; <=0 uses the native image width")
    parser.add_argument("--height", type=int, default=0, help="render height; <=0 uses the native image height")
    parser.add_argument("--views", default="6,14,22,30")
    parser.add_argument("--iterations", type=int, default=1600)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--densify-interval", type=int, default=100)
    parser.add_argument("--prune-start", type=int, default=500)
    parser.add_argument("--prune-interval", type=int, default=200)
    parser.add_argument("--grad-threshold", type=float, default=5.0e-4)
    parser.add_argument("--visibility-threshold", type=float, default=4.0)
    parser.add_argument("--min-contribution", type=float, default=1.0e-4)
    parser.add_argument("--min-opacity", type=float, default=0.085)
    parser.add_argument("--max-new-roots", type=int, default=128)
    parser.add_argument("--max-prune-fraction", type=float, default=0.08)
    parser.add_argument("--children-per-parent", type=int, default=2)
    parser.add_argument("--candidate-rings", type=int, default=3)
    parser.add_argument("--candidate-face-count", type=int, default=36)
    parser.add_argument("--neighbor-count", type=int, default=16)
    parser.add_argument("--attribute-neighbor-count", type=int, default=8)
    parser.add_argument("--min-child-distance", type=float, default=0.0025)
    parser.add_argument("--samples", type=int, default=18)
    parser.add_argument("--min-segments", type=int, default=4)
    parser.add_argument("--max-segments", type=int, default=16)
    parser.add_argument("--lr-field", type=float, default=0.020)
    parser.add_argument("--lr-root", type=float, default=0.0015)
    parser.add_argument("--alpha-weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--target-only", action="store_true", help="render teacher targets and exit without student training")
    args = parser.parse_args()
    return RunConfig(
        data_root=args.data_root,
        mesh_path=args.mesh_path,
        root_path=args.root_path,
        output_dir=args.output_dir,
        scenarios=parse_csv_strings(args.scenarios),
        modes=parse_csv_strings(args.modes),
        teacher_roots=args.teacher_roots,
        student_roots=args.student_roots,
        max_roots=args.max_roots,
        width=args.width,
        height=args.height,
        views=parse_csv_ints(args.views),
        iterations=args.iterations,
        eval_every=args.eval_every,
        log_every=args.log_every,
        warmup=args.warmup,
        densify_interval=args.densify_interval,
        prune_start=args.prune_start,
        prune_interval=args.prune_interval,
        grad_threshold=args.grad_threshold,
        visibility_threshold=args.visibility_threshold,
        min_contribution=args.min_contribution,
        min_opacity=args.min_opacity,
        max_new_roots=args.max_new_roots,
        max_prune_fraction=args.max_prune_fraction,
        children_per_parent=args.children_per_parent,
        candidate_rings=args.candidate_rings,
        candidate_face_count=args.candidate_face_count,
        neighbor_count=args.neighbor_count,
        attribute_neighbor_count=args.attribute_neighbor_count,
        min_child_distance=args.min_child_distance,
        samples=args.samples,
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        lr_field=args.lr_field,
        lr_root=args.lr_root,
        alpha_weight=args.alpha_weight,
        seed=args.seed,
        target_only=bool(args.target_only),
    )


def main() -> None:
    summary = run(parse_args())
    print(json.dumps({"summary_path": summary["config"]["output_dir"] + "/summary.json"}, indent=2))


if __name__ == "__main__":
    main()

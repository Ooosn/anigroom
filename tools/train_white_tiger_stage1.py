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
from gsplat.rendering import rasterization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.data.white_tiger import build_stage1_input_report, list_images  # noqa: E402
from anigroom.data.alignment import apply_alignment_to_namespace, load_alignment_config  # noqa: E402
from anigroom.evaluation.metrics import MetricComputer  # noqa: E402
from anigroom.grooming import (  # noqa: E402
    GroomParameterField,
    GroomRanges,
    build_strands,
    expand_child_strands,
    make_tangent_frames,
    resample_strands_to_segment_budgets,
    strand_segment_budgets,
    strands_to_gaussians,
)
from anigroom.mesh_roots import (  # noqa: E402
    TriangleMesh,
    initialize_surface_roots_fps,
    read_obj_mesh,
    validate_surface_roots,
)
from anigroom.projection import (  # noqa: E402
    MeshDepthResult,
    render_mesh_depth,
    render_mesh_depth_from_tensors,
    sample_depth_nearest,
    sample_mesh_visible_points,
)
from anigroom.roots.lifecycle import (  # noqa: E402
    DensifyConfig,
    PruneConfig,
    RootLifecycleState,
    apply_attribute_update,
    apply_structure_update,
    interpolate_child_attributes,
    propose_structure_update,
)
from anigroom.roots.statistics import RootStatsWindow  # noqa: E402


EPS = 1.0e-8


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def inv_sigmoid(x: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def set_range(raw: torch.Tensor, value: torch.Tensor | float, bounds: tuple[float, float]) -> None:
    lo, hi = bounds
    v = torch.as_tensor(value, device=raw.device, dtype=raw.dtype)
    rel = (v - lo) / max(hi - lo, EPS)
    raw.copy_(inv_sigmoid(rel).expand_as(raw))


def set_color(raw: torch.Tensor, value: torch.Tensor) -> None:
    raw.copy_(inv_sigmoid(value).expand_as(raw))


def dense_groom_ranges() -> GroomRanges:
    return GroomRanges(
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


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device)


def load_mask(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr[..., None]).to(device=device)


def load_scalar_map(path: Path, size: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if not data.files:
            raise ValueError(f"empty npz map: {path}")
        arr = data[data.files[0]].astype(np.float32)
    else:
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"map resolution mismatch for {path}: {image.size} != {size}")
            arr = np.asarray(image.convert("L"), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != (size[1], size[0]):
        raise ValueError(f"map shape mismatch for {path}: {arr.shape} != {(size[1], size[0])}")
    return arr


def load_orientation(path: Path, conf_path: Path, size: tuple[int, int], bins: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    raw_angle = load_scalar_map(path, size)
    if raw_angle.max() <= 1.0 and raw_angle.min() >= 0.0:
        angle = raw_angle * math.pi
    elif raw_angle.max() <= float(bins) + 1.0:
        angle = raw_angle / max(float(bins), 1.0) * math.pi
    else:
        angle = raw_angle / 255.0 * math.pi
    orientation = np.stack([np.cos(angle), np.sin(angle)], axis=-1).astype(np.float32)

    conf_raw = load_scalar_map(conf_path, size).astype(np.float32)
    if conf_path.suffix.lower() in {".npy", ".npz"}:
        var = np.maximum(conf_raw / (math.pi**2), 0.0)
        confidence = 1.0 / (var * var + 1.0e-7)
        finite = np.isfinite(confidence)
        if finite.any():
            norm = max(float(np.quantile(confidence[finite], 0.95)), 1.0e-6)
            confidence = np.clip(confidence / norm, 0.0, 1.0)
        else:
            confidence = np.zeros_like(conf_raw, dtype=np.float32)
    else:
        confidence = conf_raw
        if confidence.max() > 1.5:
            confidence = confidence / 255.0
        confidence = np.clip(confidence, 0.0, 1.0)
    return (
        torch.from_numpy(orientation).to(device=device),
        torch.from_numpy(confidence[..., None].astype(np.float32)).to(device=device),
    )


def load_camera_tensors(data_root: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    intr = np.load(data_root / "cameras_intr.npy").astype(np.float32)
    extr = np.load(data_root / "cameras_extr.npy").astype(np.float32)
    return torch.from_numpy(extr).to(device=device), torch.from_numpy(intr[:, :3, :3]).to(device=device)


def face_normals_np(mesh: TriangleMesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norms, EPS)
    return normals.astype(np.float32)


def save_image(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(path)


def depth_to_image(depth: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(depth)
    if not bool(finite.any()):
        return torch.zeros((*depth.shape, 1), device=depth.device)
    values = depth[finite]
    lo = torch.quantile(values, 0.02)
    hi = torch.quantile(values, 0.98)
    norm = (depth - lo) / (hi - lo).clamp_min(EPS)
    norm = torch.where(finite, norm.clamp(0.0, 1.0), torch.zeros_like(norm))
    return norm[..., None]


@torch.no_grad()
def save_clip_overlay(
    path: Path,
    base_image: torch.Tensor,
    means: torch.Tensor,
    keep_mask: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    *,
    behind_mesh_mask: torch.Tensor | None = None,
    max_points: int = 12000,
    mode: str = "both",
) -> None:
    if mode not in {"both", "kept", "clipped"}:
        raise ValueError(f"Unknown clip overlay mode: {mode}")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (base_image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    canvas = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    xy, depth = project_points(means, viewmat, k)
    height, width = int(base_image.shape[0]), int(base_image.shape[1])
    valid = (
        (depth > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= height - 1)
    )

    def draw_subset(mask: torch.Tensor, color: tuple[int, int, int, int], radius: int) -> None:
        ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if ids.numel() == 0:
            return
        if ids.numel() > int(max_points):
            step = max(1, int(math.ceil(ids.numel() / int(max_points))))
            ids = ids[::step]
        pts = xy[ids].detach().cpu().numpy()
        for x, y in pts:
            draw.ellipse((float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius), fill=color)

    if behind_mesh_mask is None:
        behind_mesh_mask = torch.zeros_like(keep_mask)

    if mode in {"both", "kept"}:
        draw_subset(valid & keep_mask, (40, 220, 80, 110), 1)
    if mode in {"both", "clipped"}:
        draw_subset(valid & behind_mesh_mask, (255, 40, 20, 185), 2)
    label = {
        "both": "green=kept Gaussians, red=depth-clipped Gaussians",
        "kept": "kept Gaussians only",
        "clipped": "depth-clipped Gaussians only",
    }[mode]
    draw.rectangle((10, 10, 610, 42), fill=(255, 255, 255, 220))
    draw.text((18, 17), label, fill=(0, 0, 0, 255))
    canvas.save(path)


def project_points(points: torch.Tensor, viewmat: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot = viewmat[:3, :3]
    trans = viewmat[:3, 3]
    cam = points @ rot.T + trans.view(1, 3)
    z = cam[:, 2].clamp_min(1.0e-6)
    x = k[0, 0] * (cam[:, 0] / z) + k[0, 2]
    y = k[1, 1] * (cam[:, 1] / z) + k[1, 2]
    return torch.stack([x, y], dim=-1), cam[:, 2]


def project_directions(points: torch.Tensor, directions: torch.Tensor, viewmat: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    rot = viewmat[:3, :3]
    trans = viewmat[:3, 3]
    cam = points @ rot.T + trans.view(1, 3)
    dirs_cam = directions @ rot.T
    z = cam[:, 2].clamp_min(1.0e-6)
    du = k[0, 0] * (dirs_cam[:, 0] * z - cam[:, 0] * dirs_cam[:, 2]) / z.square()
    dv = k[1, 1] * (dirs_cam[:, 1] * z - cam[:, 1] * dirs_cam[:, 2]) / z.square()
    return torch.stack([du, dv], dim=-1)


def bilinear_sample(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    height, width = int(image.shape[0]), int(image.shape[1])
    grid_x = (xy[:, 0] / max(width - 1, 1)) * 2.0 - 1.0
    grid_y = (xy[:, 1] / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).T


def mask_edge_confidence(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if int(kernel_size) <= 1:
        return mask
    pad = int(kernel_size) // 2
    m = mask[..., 0] if mask.ndim == 3 else mask
    eroded = -F.max_pool2d(-m[None, None], kernel_size=int(kernel_size), stride=1, padding=pad)[0, 0]
    return eroded.clamp(0.0, 1.0)[..., None]


def view_angle_weight(normals: torch.Tensor, viewmat: torch.Tensor, power: float) -> torch.Tensor:
    normal_cam = normals @ viewmat[:3, :3].T
    weight = (-normal_cam[:, 2]).clamp(0.0, 1.0)
    if float(power) != 1.0:
        weight = weight.pow(float(power))
    return weight


def double_angle_orientation(orientation: torch.Tensor) -> torch.Tensor:
    direction = F.normalize(orientation, dim=-1, eps=1.0e-8)
    x, y = direction[..., 0:1], direction[..., 1:2]
    return torch.cat([x.square() - y.square(), 2.0 * x * y], dim=-1)


def gaussian_screen_orientation_colors(
    means: torch.Tensor,
    directions: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    screen = project_directions(means, directions, viewmat, k)
    screen_dir = F.normalize(screen, dim=-1, eps=1.0e-8)
    cos2 = screen_dir[:, 0:1].square() - screen_dir[:, 1:2].square()
    sin2 = 2.0 * screen_dir[:, 0:1] * screen_dir[:, 1:2]
    return torch.cat([0.5 + 0.5 * cos2, 0.5 + 0.5 * sin2, torch.zeros_like(cos2)], dim=-1)


def render_orientation_map(
    gaussians,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    background = torch.zeros((1, 3), device=gaussians.means.device, dtype=gaussians.means.dtype)
    orient_colors = gaussian_screen_orientation_colors(gaussians.means, gaussians.directions, viewmat, k)
    orient_image, orient_alpha, _ = rasterization(
        gaussians.means,
        gaussians.quats,
        gaussians.scales,
        gaussians.opacities.reshape(-1),
        orient_colors,
        viewmat.view(1, 4, 4),
        k.view(1, 3, 3),
        width,
        height,
        packed=False,
        backgrounds=background,
        rasterize_mode="antialiased",
    )
    alpha = orient_alpha[0].clamp(0.0, 1.0)
    avg = orient_image[0][..., :2] / alpha.clamp_min(1.0e-4)
    orientation = F.normalize(2.0 * avg - 1.0, dim=-1, eps=1.0e-8)
    return orientation, alpha


def orientation_map_losses(
    pred_orientation: torch.Tensor,
    pred_confidence: torch.Tensor,
    target_orientation: torch.Tensor,
    target_confidence: torch.Tensor,
    min_confidence: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    target_double = double_angle_orientation(target_orientation).detach()
    target_conf = target_confidence.detach().clamp(0.0, 1.0)
    pred_visible = (pred_confidence.detach() > 0.02).to(dtype=target_conf.dtype)
    valid_target = (target_conf >= float(min_confidence)).to(dtype=target_conf.dtype)
    weight = target_conf * pred_visible * valid_target
    dot = (pred_orientation * target_double).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    orient_loss = ((1.0 - dot) * weight).sum() / weight.sum().clamp_min(1.0)

    dx_weight = torch.minimum(weight[:, 1:], weight[:, :-1])
    dy_weight = torch.minimum(weight[1:, :], weight[:-1, :])
    dx = (pred_orientation[:, 1:] - pred_orientation[:, :-1]) - (target_double[:, 1:] - target_double[:, :-1])
    dy = (pred_orientation[1:, :] - pred_orientation[:-1, :]) - (target_double[1:, :] - target_double[:-1, :])
    dx_loss = (dx.abs() * dx_weight).sum() / dx_weight.sum().clamp_min(1.0)
    dy_loss = (dy.abs() * dy_weight).sum() / dy_weight.sum().clamp_min(1.0)
    detail_loss = 0.5 * (dx_loss + dy_loss)
    return orient_loss, detail_loss, {
        "orientation_loss": float(orient_loss.detach().cpu()),
        "orientation_detail_loss": float(detail_loss.detach().cpu()),
        "orientation_weight_sum": float(weight.detach().sum().cpu()),
        "orientation_valid_pixels": int((weight.detach() > 0.0).sum().cpu()),
        "orientation_pred_visible_pixels": int((pred_visible.detach() > 0.0).sum().cpu()),
    }


def build_knn_edges(points: torch.Tensor, k: int, chunk_size: int = 2048) -> torch.Tensor:
    if k <= 0 or points.shape[0] < 2:
        return torch.empty((0, 2), dtype=torch.long, device=points.device)
    pts = points.detach()
    root_count = int(pts.shape[0])
    edges = []
    for begin in range(0, root_count, int(chunk_size)):
        end = min(begin + int(chunk_size), root_count)
        dist = torch.cdist(pts[begin:end], pts)
        local_ids = torch.arange(begin, end, device=points.device)
        dist[torch.arange(end - begin, device=points.device), local_ids] = torch.inf
        nn = torch.topk(dist, k=min(k, root_count - 1), dim=1, largest=False).indices
        src = local_ids[:, None].expand_as(nn)
        edges.append(torch.stack([src.reshape(-1), nn.reshape(-1)], dim=-1))
    return torch.cat(edges, dim=0)


def interpolate_unobserved_root_values(
    roots: torch.Tensor,
    values: torch.Tensor,
    observed: torch.Tensor,
    confidence: torch.Tensor,
    *,
    neighbor_count: int = 8,
    chunk_size: int = 2048,
    normalize_vectors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[0] != roots.shape[0] or observed.shape[0] != roots.shape[0]:
        raise RuntimeError("root interpolation shape mismatch")
    if bool(observed.all()):
        return values, observed.clone()
    if not bool(observed.any()):
        return values, observed.clone()

    filled = values.clone()
    obs_idx = torch.nonzero(observed, as_tuple=False).reshape(-1)
    miss_idx = torch.nonzero(~observed, as_tuple=False).reshape(-1)
    obs_roots = roots[obs_idx].detach()
    obs_values = values[obs_idx]
    obs_conf = confidence[obs_idx].reshape(-1).clamp_min(1.0e-4)
    k = min(int(neighbor_count), int(obs_idx.numel()))
    for begin in range(0, int(miss_idx.numel()), int(chunk_size)):
        ids = miss_idx[begin : begin + int(chunk_size)]
        dist = torch.cdist(roots[ids].detach(), obs_roots)
        nn_dist, nn_local = torch.topk(dist, k=k, dim=1, largest=False)
        weights = (1.0 / nn_dist.clamp_min(1.0e-6)) * obs_conf[nn_local]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        interp = (obs_values[nn_local] * weights[..., None]).sum(dim=1)
        if normalize_vectors:
            interp = F.normalize(interp, dim=-1, eps=1.0e-8)
        filled[ids] = interp
    interpolated = observed.clone()
    interpolated[miss_idx] = True
    return filled, interpolated


def root_graph_smoothness(
    field: GroomParameterField,
    edges: torch.Tensor,
    observation_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    if edges.numel() == 0:
        return next(field.parameters()).new_tensor(0.0)
    groom = field.decode()
    src, dst = edges[:, 0], edges[:, 1]
    if observation_confidence is None:
        edge_weight = groom.length.new_ones((edges.shape[0],))
    else:
        conf = observation_confidence.detach().reshape(-1).clamp(0.0, 1.0)
        edge_weight = 0.25 + (1.0 - torch.minimum(conf[src], conf[dst]))

    def weighted_mean(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 1:
            value = value.mean(dim=tuple(range(1, value.ndim)))
        return (value * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)

    terms = [
        4.0 * weighted_mean((groom.length[src] - groom.length[dst]).square()),
        2.0 * weighted_mean((torch.log(groom.root_width[src].clamp_min(1.0e-6)) - torch.log(groom.root_width[dst].clamp_min(1.0e-6))).square()),
        0.8 * weighted_mean((torch.log(groom.tip_width[src].clamp_min(1.0e-6)) - torch.log(groom.tip_width[dst].clamp_min(1.0e-6))).square()),
        0.4 * weighted_mean((groom.width_taper[src] - groom.width_taper[dst]).square()),
        1.0 * weighted_mean((F.normalize(groom.flow_xy[src], dim=-1) - F.normalize(groom.flow_xy[dst], dim=-1)).square()),
        1.2 * weighted_mean((groom.flow_strength[src] - groom.flow_strength[dst]).square()),
        1.0 * weighted_mean((groom.lift[src] - groom.lift[dst]).square()),
        1.0 * weighted_mean((groom.bend[src] - groom.bend[dst]).square()),
        0.8 * weighted_mean((groom.stiffness[src] - groom.stiffness[dst]).square()),
        0.6 * weighted_mean((groom.curl_radius[src] - groom.curl_radius[dst]).square()),
        0.35 * weighted_mean((groom.curl_frequency[src] - groom.curl_frequency[dst]).square()),
        0.25
        * weighted_mean(
            (torch.cos(groom.curl_phase[src]) - torch.cos(groom.curl_phase[dst])).square()
            + (torch.sin(groom.curl_phase[src]) - torch.sin(groom.curl_phase[dst])).square()
        ),
        0.4 * weighted_mean((groom.frizz[src] - groom.frizz[dst]).square()),
        0.8 * weighted_mean((groom.child_radius[src] - groom.child_radius[dst]).square()),
        0.8 * weighted_mean((groom.clump_strength[src] - groom.clump_strength[dst]).square()),
        0.25 * weighted_mean((groom.root_color[src] - groom.root_color[dst]).square()),
        0.15 * weighted_mean((groom.tip_color[src] - groom.tip_color[dst]).square()),
        0.5 * weighted_mean((groom.opacity[src] - groom.opacity[dst]).square()),
        0.25 * weighted_mean((groom.tip_opacity[src] - groom.tip_opacity[dst]).square()),
    ]
    return torch.stack(terms).sum()


def groom_shape_prior(field: GroomParameterField) -> torch.Tensor:
    """Keep short-fur geometry from using long/noisy strands as paint strokes."""
    groom = field.decode()
    ranges = field.ranges

    def norm(value: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return (value - float(lo)) / max(float(hi) - float(lo), EPS)

    length_excess = torch.relu((groom.length - 0.075) / max(float(ranges.length[1] - ranges.length[0]), EPS))
    curl_n = norm(groom.curl_radius, ranges.curl_radius)
    frizz_n = norm(groom.frizz, ranges.frizz)
    child_n = norm(groom.child_radius, ranges.child_radius)
    bend_n = groom.bend
    lift_excess = torch.relu((groom.lift - 0.040) / max(float(ranges.lift[1] - ranges.lift[0]), EPS))

    return (
        2.0 * length_excess.square().mean()
        + 0.8 * curl_n.square().mean()
        + 1.2 * frizz_n.square().mean()
        + 0.8 * child_n.square().mean()
        + 0.20 * bend_n.square().mean()
        + 0.25 * lift_excess.square().mean()
    )


def strand_shape_consistency_loss(
    strands: torch.Tensor,
    edges: torch.Tensor,
    observation_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Local NeuralFur-style shape consistency on neighboring guide strands."""
    if edges.numel() == 0 or strands.shape[0] < 2 or strands.shape[1] < 4:
        return strands.new_tensor(0.0)
    src, dst = edges[:, 0], edges[:, 1]
    dirs = F.normalize(strands[:, 1:] - strands[:, :-1], dim=-1, eps=1.0e-8)
    curvature = dirs[:, 1:] - dirs[:, :-1]
    if observation_confidence is None:
        edge_weight = strands.new_ones((edges.shape[0], 1, 1))
    else:
        conf = observation_confidence.detach().reshape(-1).clamp(0.0, 1.0)
        edge_weight = (0.35 + (1.0 - torch.minimum(conf[src], conf[dst]))).view(-1, 1, 1)
    direction_diff = (dirs[src] - dirs[dst]).square()
    curvature_diff = (curvature[src] - curvature[dst]).square()
    direction_term = (direction_diff * edge_weight).sum() / (edge_weight.sum().clamp_min(1.0) * direction_diff.shape[1] * direction_diff.shape[2])
    curvature_term = (curvature_diff * edge_weight).sum() / (edge_weight.sum().clamp_min(1.0) * curvature_diff.shape[1] * curvature_diff.shape[2])
    return 0.65 * direction_term + 0.35 * curvature_term


@torch.no_grad()
def groom_parameter_stats(field: GroomParameterField) -> dict[str, dict[str, float]]:
    groom = field.decode()

    def summarize(value: torch.Tensor) -> dict[str, float]:
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        q = torch.quantile(flat, torch.tensor([0.05, 0.50, 0.95], device=flat.device))
        return {
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()),
            "min": float(flat.min().cpu()),
            "p05": float(q[0].cpu()),
            "p50": float(q[1].cpu()),
            "p95": float(q[2].cpu()),
            "max": float(flat.max().cpu()),
        }

    return {
        "length": summarize(groom.length),
        "root_width": summarize(groom.root_width),
        "tip_width": summarize(groom.tip_width),
        "width_taper": summarize(groom.width_taper),
        "flow_strength": summarize(groom.flow_strength),
        "lift": summarize(groom.lift),
        "bend": summarize(groom.bend),
        "stiffness": summarize(groom.stiffness),
        "curl_radius": summarize(groom.curl_radius),
        "curl_frequency": summarize(groom.curl_frequency),
        "frizz": summarize(groom.frizz),
        "child_radius": summarize(groom.child_radius),
        "clump_strength": summarize(groom.clump_strength),
        "opacity": summarize(groom.opacity),
        "tip_opacity": summarize(groom.tip_opacity),
    }


@dataclass(frozen=True)
class Stage1Config:
    data_root: str
    mesh_path: str
    output_dir: str
    root_count: int = 10000
    candidate_multiplier: float = 10.0
    iterations: int = 30000
    eval_every: int = 1000
    save_every: int = 5000
    test_stride: int = 6
    train_views: str = ""
    test_views: str = ""
    seed: int = 13
    expected_width: int = 1920
    expected_height: int = 1080
    init_mesh_scale: float = 1.28
    init_mesh_translation: tuple[float, float, float] = (0.0, 0.32, 0.02)
    samples: int = 48
    min_segments: int = 6
    max_segments: int = 22
    child_count: int = 8
    projected_init_views: int = 24
    projected_init_min_confidence: float = 0.08
    projected_init_depth_abs_tolerance: float = 0.03
    projected_init_depth_rel_tolerance: float = 0.01
    projected_init_local_depth_kernel: int = 7
    projected_init_front_normal_z: float = 0.15
    projected_init_mask_edge_kernel: int = 9
    projected_init_view_angle_power: float = 1.0
    lr_groom: float = 1.4e-2
    lr_high_frequency_shape_scale: float = 1.0
    lr_color: float = 2.0e-2
    lr_root: float = 7.5e-4
    lr_calibration: float = 5.0e-4
    rgb_weight: float = 1.0
    random_backing_loss_weight: float = 0.25
    mask_weight: float = 0.15
    orientation_weight: float = 0.08
    orientation_detail_weight: float = 0.02
    smooth_weight: float = 0.04
    strand_shape_smooth_weight: float = 0.0
    shape_prior_weight: float = 0.0
    root_move_reg_weight: float = 0.003
    orientation_min_confidence: float = 0.08
    compute_lpips: bool = False
    white_background: bool = True
    random_backing_color: bool = True
    backing_color_min: float = 0.05
    backing_color_max: float = 0.85
    mesh_depth_clipping: bool = True
    mesh_depth_abs_tolerance: float = 0.018
    mesh_depth_rel_tolerance: float = 0.004
    mesh_depth_local_kernel: int = 1
    mesh_backing_compositing: bool = True
    use_gravity_sag: bool = False
    densify_warmup: int = 500
    densify_interval: int = 100
    densify_until: int = 12000
    densify_score_threshold: float = 2.5e-5
    densify_min_contribution: float = 0.45
    max_splits_per_event: int = 256
    split_children_per_parent: int = 2
    split_neighbor_count: int = 12
    split_candidate_rings: int = 3
    split_candidate_face_count: int = 32
    split_min_child_distance: float = 0.0
    prune_start: int = 999999
    prune_interval: int = 100
    prune_min_contribution: float = 0.08
    prune_min_opacity: float = 0.0
    prune_max_fraction: float = 0.05
    resume_checkpoint: str = ""


class WhiteTigerStage1Model(torch.nn.Module):
    def __init__(
        self,
        mesh: TriangleMesh,
        face_normals: np.ndarray,
        face_ids: np.ndarray,
        barycentric: np.ndarray,
        ranges: GroomRanges,
        device: torch.device,
        init_scale: float = 1.25,
        init_translation: tuple[float, float, float] = (0.0, 0.32, 0.0),
    ) -> None:
        super().__init__()
        self.register_buffer("vertices", torch.from_numpy(mesh.vertices).to(device=device))
        self.register_buffer("faces", torch.from_numpy(mesh.faces).to(device=device, dtype=torch.long))
        self.register_buffer("face_ids", torch.from_numpy(face_ids).to(device=device, dtype=torch.long))
        self.register_buffer("face_normals", torch.from_numpy(face_normals).to(device=device))
        self.register_buffer("bary_initial", torch.from_numpy(barycentric).to(device=device))
        tri = self.vertices[self.faces[self.face_ids]]
        self.register_buffer("anchor_local", (tri * self.bary_initial[:, :, None]).sum(dim=1))
        self.register_buffer("root_observation_confidence", torch.zeros((int(face_ids.shape[0]),), device=device))
        self.bary_logits = torch.nn.Parameter(torch.log(self.bary_initial.clamp_min(1.0e-5)))
        self.groom = GroomParameterField(int(face_ids.shape[0]), ranges=ranges, device=device)
        self.log_scale = torch.nn.Parameter(torch.tensor([math.log(float(init_scale))], device=device))
        self.translation = torch.nn.Parameter(torch.tensor(init_translation, device=device, dtype=torch.float32))
        self.initialize_default_groom()

    def initialize_default_groom(self) -> None:
        ranges = self.groom.ranges
        with torch.no_grad():
            set_range(self.groom.length_raw, 0.060, ranges.length)
            set_range(self.groom.root_width_raw, 0.00016, ranges.root_width)
            set_range(self.groom.tip_width_ratio_raw, 0.070, ranges.tip_width_ratio)
            set_range(self.groom.width_taper_raw, 1.80, ranges.width_taper)
            self.groom.flow_xy[:, 0:1].fill_(0.92)
            self.groom.flow_xy[:, 1:2].fill_(-0.12)
            set_range(self.groom.flow_strength_raw, 0.86, ranges.flow_strength)
            set_range(self.groom.lift_raw, 0.018, ranges.lift)
            set_range(self.groom.sag_raw, 0.0, ranges.sag)
            set_range(self.groom.stiffness_raw, 0.72, ranges.stiffness)
            set_range(self.groom.curl_radius_raw, 0.0030, ranges.curl_radius)
            set_range(self.groom.curl_frequency_raw, 1.20, ranges.curl_frequency)
            set_range(self.groom.frizz_raw, 0.0008, ranges.frizz)
            set_range(self.groom.child_radius_raw, 0.0028, ranges.child_radius)
            set_range(self.groom.clump_strength_raw, 0.25, ranges.clump_strength)
            set_range(self.groom.opacity_raw, 0.74, ranges.opacity)
            set_range(self.groom.tip_opacity_ratio_raw, 0.45, ranges.tip_opacity_ratio)
            root_color = torch.tensor([0.88, 0.88, 0.82], device=self.bary_logits.device).view(1, 3)
            tip_color = torch.tensor([0.98, 0.96, 0.88], device=self.bary_logits.device).view(1, 3)
            set_color(self.groom.root_color_raw, root_color)
            set_color(self.groom.tip_color_raw, tip_color)

    def roots_and_normals(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tri = self.vertices[self.faces[self.face_ids]]
        bary = torch.softmax(self.bary_logits, dim=-1)
        roots_local = (tri * bary[:, :, None]).sum(dim=1)
        roots = roots_local * torch.exp(self.log_scale).view(1, 1) + self.translation.view(1, 3)
        normals = F.normalize(self.face_normals[self.face_ids], dim=-1, eps=1.0e-8)
        return roots, normals, roots_local

    def render_parameters(self, samples: int, child_count: int, min_segments: int, max_segments: int):
        roots, normals, roots_local = self.roots_and_normals()
        tangents, bitangents = make_tangent_frames(normals)
        groom = self.groom.decode()
        strands, widths, colors, opacities = build_strands(
            roots,
            normals,
            tangents,
            bitangents,
            groom,
            samples=samples,
            use_gravity_sag=False,
        )
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
        counts, count_stats = strand_segment_budgets(
            strands.detach(),
            child_lengths.detach(),
            min_segments,
            max_segments,
            length_bounds=self.groom.ranges.length,
        )
        resampled = resample_strands_to_segment_budgets(strands, widths, colors, opacities, counts)
        gaussians = strands_to_gaussians(
            resampled.strands,
            resampled.widths,
            resampled.colors,
            resampled.opacities,
            resampled.segment_mask,
            strand_root_indices=root_ids,
            length_overlap=1.45,
        )
        stats = {
            **count_stats,
            **resampled.stats,
            "root_count": int(roots.shape[0]),
            "gaussian_count": int(gaussians.means.shape[0]),
            "scale": float(torch.exp(self.log_scale.detach()).cpu()),
            "translation_norm": float(torch.linalg.norm(self.translation.detach()).cpu()),
        }
        return gaussians, roots, roots_local, stats

    def guide_strands_for_loss(self, samples: int) -> torch.Tensor:
        roots, normals, _ = self.roots_and_normals()
        tangents, bitangents = make_tangent_frames(normals)
        groom = self.groom.decode()
        strands, _, _, _ = build_strands(
            roots,
            normals,
            tangents,
            bitangents,
            groom,
            samples=max(int(samples), 4),
            use_gravity_sag=False,
        )
        return strands

    def lifecycle_state(self) -> RootLifecycleState:
        _, _, roots_local = self.roots_and_normals()
        return RootLifecycleState(
            points=roots_local.detach(),
            face_ids=self.face_ids.detach().clone(),
            barycentric=torch.softmax(self.bary_logits.detach(), dim=-1),
        )

    def apply_structure_update(self, update) -> dict[str, int]:
        old_state = self.lifecycle_state()
        old_count = int(old_state.points.shape[0])
        if update.new_barycentric.numel() == 0 and not bool(update.prune_mask.any()):
            return {"old_root_count": old_count, "root_count_after": old_count}

        ranges = self.groom.ranges
        device = self.vertices.device
        old_params = {name: param.detach() for name, param in self.groom.named_parameters()}
        new_values: dict[str, torch.Tensor] = {}
        for name, values in old_params.items():
            if update.new_barycentric.numel() == 0:
                child = values.new_empty((0, *values.shape[1:]))
            elif name == "curl_phase":
                child_cos = interpolate_child_attributes(
                    torch.cos(values),
                    old_state,
                    update,
                    self.vertices,
                    self.faces,
                    neighbor_count=8,
                    parent_weight=3.0,
                )
                child_sin = interpolate_child_attributes(
                    torch.sin(values),
                    old_state,
                    update,
                    self.vertices,
                    self.faces,
                    neighbor_count=8,
                    parent_weight=3.0,
                )
                child = torch.atan2(child_sin, child_cos)
            else:
                child = interpolate_child_attributes(
                    values,
                    old_state,
                    update,
                    self.vertices,
                    self.faces,
                    neighbor_count=8,
                    parent_weight=3.0,
                )
            new_values[name] = apply_attribute_update(values, update, child)

        new_state = apply_structure_update(old_state, update, self.vertices, self.faces)
        new_count = int(new_state.points.shape[0])
        new_groom = GroomParameterField(new_count, ranges=ranges, device=device)
        with torch.no_grad():
            new_params = dict(new_groom.named_parameters())
            for name, value in new_values.items():
                if name not in new_params:
                    raise KeyError(f"unknown groom parameter during structure update: {name}")
                if new_params[name].shape != value.shape:
                    raise RuntimeError(f"groom parameter shape mismatch for {name}: {tuple(new_params[name].shape)} != {tuple(value.shape)}")
                new_params[name].copy_(value.to(device=device, dtype=new_params[name].dtype))

        self.face_ids = new_state.face_ids.detach().long()
        self.bary_initial = new_state.barycentric.detach()
        self.anchor_local = new_state.points.detach()
        self.bary_logits = torch.nn.Parameter(torch.log(self.bary_initial.clamp_min(1.0e-5)))
        self.groom = new_groom
        old_conf = self.root_observation_confidence.detach()
        child_conf = (
            interpolate_child_attributes(
                old_conf[:, None],
                old_state,
                update,
                self.vertices,
                self.faces,
                neighbor_count=8,
                parent_weight=3.0,
            ).reshape(-1)
            if update.new_barycentric.numel() > 0
            else old_conf.new_empty((0,))
        )
        self.root_observation_confidence = apply_attribute_update(old_conf, update, child_conf).detach().clamp(0.0, 1.0)
        return {"old_root_count": old_count, "root_count_after": new_count}


@torch.no_grad()
def initialize_groom_from_projections(
    model: WhiteTigerStage1Model,
    image_paths: list[Path],
    mask_paths: list[Path],
    angle_paths: list[Path],
    conf_paths: list[Path],
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    train_indices: list[int],
    width: int,
    height: int,
    config: Stage1Config,
    device: torch.device,
) -> dict[str, float | int]:
    if config.projected_init_views <= 0:
        return {"projected_init_view_count": 0}
    roots, normals, _ = model.roots_and_normals()
    tangents, bitangents = make_tangent_frames(normals)
    root_count = int(roots.shape[0])
    color_sum = torch.zeros((root_count, 3), device=device)
    flow_sum = torch.zeros((root_count, 2), device=device)
    weight_sum = torch.zeros((root_count, 1), device=device)
    chosen = train_indices[: max(1, min(int(config.projected_init_views), len(train_indices)))]
    mesh_for_visibility = TriangleMesh(
        vertices=(
            model.vertices.detach().cpu().numpy() * float(torch.exp(model.log_scale.detach()).cpu())
            + model.translation.detach().cpu().numpy().reshape(1, 3)
        ).astype(np.float32),
        faces=model.faces.detach().cpu().numpy().astype(np.int32),
    )

    default_flow = F.normalize(model.groom.flow_xy.detach(), dim=-1)
    for idx in chosen:
        image = load_image(image_paths[idx], device)
        mask = load_mask(mask_paths[idx], device)
        mask_conf = mask_edge_confidence(mask, config.projected_init_mask_edge_kernel)
        target_orientation, target_conf = load_orientation(angle_paths[idx], conf_paths[idx], (width, height), bins=180, device=device)
        target_conf = target_conf * mask * mask_conf
        mesh_depth = render_mesh_depth(mesh_for_visibility, viewmats[idx], ks[idx], width, height, device=device)
        root_vis = sample_mesh_visible_points(
            roots,
            normals,
            viewmats[idx],
            ks[idx],
            mesh_depth.depth,
            depth_abs_tolerance=config.projected_init_depth_abs_tolerance,
            depth_rel_tolerance=config.projected_init_depth_rel_tolerance,
            local_depth_kernel=config.projected_init_local_depth_kernel,
            front_normal_z=config.projected_init_front_normal_z,
        )
        sampled_mask = bilinear_sample(mask, root_vis.xy)[:, 0]
        sampled_conf = bilinear_sample(target_conf, root_vis.xy)[:, 0]
        angle_weight = view_angle_weight(normals, viewmats[idx], config.projected_init_view_angle_power)
        weight = (sampled_mask * sampled_conf * angle_weight * root_vis.visible.to(sampled_mask.dtype)).clamp(0.0, 1.0)
        good = weight >= float(config.projected_init_min_confidence)
        if not bool(good.any()):
            continue
        sampled_color = bilinear_sample(image, root_vis.xy).clamp(0.0, 1.0)
        sampled_ori = F.normalize(bilinear_sample(target_orientation, root_vis.xy), dim=-1, eps=1.0e-8)
        screen_t = project_directions(roots, tangents, viewmats[idx], ks[idx])
        screen_b = project_directions(roots, bitangents, viewmats[idx], ks[idx])
        screen_default = F.normalize(default_flow[:, [0]] * screen_t + default_flow[:, [1]] * screen_b, dim=-1, eps=1.0e-8)
        signed_ori = torch.where((sampled_ori * screen_default).sum(dim=-1, keepdim=True) < 0.0, -sampled_ori, sampled_ori)
        basis = torch.stack([screen_t, screen_b], dim=-1)
        coeff = torch.linalg.pinv(basis) @ signed_ori[:, :, None]
        coeff = coeff.squeeze(-1)
        coeff = F.normalize(coeff, dim=-1, eps=1.0e-8)
        w = weight[:, None]
        color_sum[good] += sampled_color[good] * w[good]
        flow_sum[good] += coeff[good] * w[good]
        weight_sum[good] += w[good]

    observed = weight_sum[:, 0] > 0.0
    if bool(observed.any()):
        groom = model.groom.decode()
        root_conf = weight_sum[:, 0]
        conf_norm = torch.quantile(root_conf[observed], 0.95).clamp_min(1.0e-6)
        root_conf = (root_conf / conf_norm).clamp(0.0, 1.0)
        colors = groom.root_color.detach().clone()
        flows = F.normalize(groom.flow_xy.detach().clone(), dim=-1, eps=1.0e-8)
        colors[observed] = (color_sum[observed] / weight_sum[observed].clamp_min(EPS)).clamp(0.02, 0.98)
        flows[observed] = F.normalize(flow_sum[observed] / weight_sum[observed].clamp_min(EPS), dim=-1, eps=1.0e-8)
        colors, filled_color = interpolate_unobserved_root_values(
            roots,
            colors,
            observed,
            root_conf,
            neighbor_count=8,
            normalize_vectors=False,
        )
        flows, filled_flow = interpolate_unobserved_root_values(
            roots,
            flows,
            observed,
            root_conf,
            neighbor_count=8,
            normalize_vectors=True,
        )
        filled = filled_color & filled_flow
        model.groom.flow_xy[filled] = flows[filled]
        model.groom.root_color_raw[filled] = inv_sigmoid(colors[filled].clamp(0.02, 0.98))
        model.groom.tip_color_raw[filled] = inv_sigmoid((0.88 * colors[filled] + 0.12).clamp(0.02, 0.98))
        model.root_observation_confidence = root_conf.detach()
    else:
        filled = observed
    return {
        "projected_init_view_count": int(len(chosen)),
        "projected_init_observed_roots": int(observed.sum().detach().cpu()),
        "projected_init_observed_fraction": float(observed.float().mean().detach().cpu()),
        "projected_init_interpolated_roots": int((filled & ~observed).sum().detach().cpu()),
        "projected_init_filled_fraction": float(filled.float().mean().detach().cpu()),
    }


def sample_backing_color(config: Stage1Config, device: torch.device, *, train: bool) -> torch.Tensor:
    if train and config.random_backing_color:
        lo = float(config.backing_color_min)
        hi = float(config.backing_color_max)
        return torch.empty((3,), device=device).uniform_(lo, hi)
    if config.white_background:
        return torch.ones((3,), device=device)
    return torch.zeros((3,), device=device)


def scene_background_color(config: Stage1Config, device: torch.device) -> torch.Tensor:
    if config.white_background:
        return torch.ones((3,), device=device)
    return torch.zeros((3,), device=device)


def make_mesh_backing_image(
    mesh_depth: MeshDepthResult,
    mesh_color: torch.Tensor,
    scene_background: torch.Tensor,
) -> torch.Tensor:
    mesh_rgb = mesh_color.view(1, 1, 3).expand((*mesh_depth.depth.shape, 3))
    bg_rgb = scene_background.view(1, 1, 3).expand_as(mesh_rgb)
    return torch.where(mesh_depth.valid[..., None], mesh_rgb, bg_rgb)


def composite_target(target: torch.Tensor, mask: torch.Tensor, backing: torch.Tensor) -> torch.Tensor:
    if backing.ndim == 1:
        backing = backing.view(1, 1, 3)
    return target * mask + backing * (1.0 - mask)


def render_model_mesh_depth(
    model: WhiteTigerStage1Model,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    *,
    device: torch.device,
    ctx=None,
) -> MeshDepthResult:
    with torch.no_grad():
        scale = torch.exp(model.log_scale.detach()).view(1, 1)
        vertices = (model.vertices.detach() * scale + model.translation.detach().view(1, 3)).contiguous()
        faces = model.faces.detach().to(dtype=torch.int32).contiguous()
        return render_mesh_depth_from_tensors(vertices, faces, viewmat, k, width, height, device=device, ctx=ctx)


def mesh_depth_clip_gaussians(
    gaussians,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    mesh_depth: MeshDepthResult,
    config: Stage1Config,
) -> tuple[object, torch.Tensor, dict[str, float | int], dict[str, torch.Tensor]]:
    if int(config.mesh_depth_local_kernel) != 1:
        raise RuntimeError("formal mesh-depth clipping must use exact per-pixel depth; set mesh_depth_local_kernel=1")
    gaussian_xy, gaussian_depth = project_points(gaussians.means, viewmat, k)
    height, width = int(mesh_depth.depth.shape[0]), int(mesh_depth.depth.shape[1])
    in_frame = (
        (gaussian_depth > 1.0e-6)
        & (gaussian_xy[:, 0] >= 0.0)
        & (gaussian_xy[:, 0] <= width - 1)
        & (gaussian_xy[:, 1] >= 0.0)
        & (gaussian_xy[:, 1] <= height - 1)
    )
    sampled_mesh_depth = sample_depth_nearest(
        mesh_depth.depth,
        gaussian_xy,
        kernel_size=int(config.mesh_depth_local_kernel),
    )
    tolerance = float(config.mesh_depth_abs_tolerance) + gaussian_depth.abs() * float(config.mesh_depth_rel_tolerance)
    behind_mesh = in_frame & torch.isfinite(sampled_mesh_depth) & (gaussian_depth > sampled_mesh_depth + tolerance)
    keep = ~behind_mesh
    if not bool(keep.any()):
        raise RuntimeError("mesh-depth clipping removed every Gaussian; check camera/mesh alignment")
    clipped = replace(
        gaussians,
        means=gaussians.means[keep],
        directions=gaussians.directions[keep],
        quats=gaussians.quats[keep],
        scales=gaussians.scales[keep],
        colors=gaussians.colors[keep],
        opacities=gaussians.opacities[keep],
        root_indices=gaussians.root_indices[keep],
        segment_indices=gaussians.segment_indices[keep],
    )
    stats = {
        "preclip_gaussian_count": int(gaussians.means.shape[0]),
        "clipped_gaussian_count": int((~keep).sum().detach().cpu()),
        "behind_mesh_gaussian_count": int(behind_mesh.sum().detach().cpu()),
        "kept_gaussian_count": int(keep.sum().detach().cpu()),
        "clip_keep_fraction": float(keep.float().mean().detach().cpu()),
    }
    masks = {
        "behind_mesh_mask": behind_mesh.detach(),
    }
    return clipped, keep, stats, masks


def render_view(
    model: WhiteTigerStage1Model,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    config: Stage1Config,
    *,
    background: torch.Tensor,
    mesh_depth: MeshDepthResult | None = None,
    backing_image: torch.Tensor | None = None,
    retain_lifecycle_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, object, torch.Tensor, dict[str, float | int], dict]:
    gaussians, roots, roots_local, stats = model.render_parameters(config.samples, config.child_count, config.min_segments, config.max_segments)
    preclip_gaussians = gaussians
    if config.mesh_depth_clipping:
        if mesh_depth is None:
            raise RuntimeError("mesh_depth_clipping is enabled but render_view received no mesh_depth")
        gaussians, keep_mask, clip_stats, clip_masks = mesh_depth_clip_gaussians(gaussians, viewmat, k, mesh_depth, config)
        stats = {**stats, **clip_stats}
    else:
        keep_mask = torch.ones((gaussians.means.shape[0],), device=gaussians.means.device, dtype=torch.bool)
        clip_masks = {
            "behind_mesh_mask": torch.zeros_like(keep_mask),
        }
        stats = {
            **stats,
            "preclip_gaussian_count": int(gaussians.means.shape[0]),
            "clipped_gaussian_count": 0,
            "kept_gaussian_count": int(gaussians.means.shape[0]),
            "clip_keep_fraction": 1.0,
            "behind_mesh_gaussian_count": 0,
            "no_mesh_depth_gaussian_count": 0,
        }
    if retain_lifecycle_grad:
        roots_local.retain_grad()
        gaussians.means.retain_grad()
        gaussians.scales.retain_grad()
    if config.mesh_backing_compositing:
        if backing_image is None:
            raise RuntimeError("mesh_backing_compositing is enabled but render_view received no backing_image")
        raster_background = torch.zeros((1, 3), device=background.device, dtype=background.dtype)
    else:
        raster_background = background.view(1, 3)
    image, alpha, info = rasterization(
        gaussians.means,
        gaussians.quats,
        gaussians.scales,
        gaussians.opacities.reshape(-1),
        gaussians.colors,
        viewmat.view(1, 4, 4),
        k.view(1, 3, 3),
        width,
        height,
        packed=False,
        backgrounds=raster_background,
        rasterize_mode="antialiased",
    )
    raw_image = image
    if config.mesh_backing_compositing:
        image = image + (1.0 - alpha) * backing_image.view(1, height, width, 3)
    stats = {**stats, "visible_gaussian_count": int((info["radii"] > 0).sum().detach().cpu())}
    info["mesh_depth_keep_mask"] = keep_mask.detach()
    info["mesh_depth_behind_mesh_mask"] = clip_masks["behind_mesh_mask"].detach()
    info["preclip_means"] = preclip_gaussians.means.detach()
    info["raw_fur_image"] = raw_image[0]
    return image[0].clamp(0.0, 1.0), alpha[0].clamp(0.0, 1.0), gaussians, roots_local, stats, info


@torch.no_grad()
def evaluate(
    model: WhiteTigerStage1Model,
    image_paths: list[Path],
    mask_paths: list[Path],
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    indices: list[int],
    width: int,
    height: int,
    config: Stage1Config,
    metric_computer: MetricComputer,
    device: torch.device,
    mesh_depth_ctx=None,
) -> dict[str, float]:
    raw_psnrs, raw_ssims = [], []
    composite_psnrs, composite_ssims = [], []
    mask_l1s = []
    mesh_color = sample_backing_color(config, device, train=False)
    scene_bg = scene_background_color(config, device)
    for idx in indices:
        target = load_image(image_paths[idx], device)
        mask = load_mask(mask_paths[idx], device)
        mesh_depth = render_model_mesh_depth(model, viewmats[idx], ks[idx], width, height, device=device, ctx=mesh_depth_ctx)
        backing_image = make_mesh_backing_image(mesh_depth, mesh_color, scene_bg)
        pred, alpha, _, _, _, _ = render_view(
            model,
            viewmats[idx],
            ks[idx],
            width,
            height,
            config,
            background=mesh_color,
            mesh_depth=mesh_depth,
            backing_image=backing_image,
        )
        target_eval = composite_target(target, mask, backing_image)
        raw_metrics = metric_computer.image_metrics(pred, target)
        composite_metrics = metric_computer.image_metrics(pred, target_eval)
        raw_psnrs.append(raw_metrics["psnr"].detach())
        raw_ssims.append(raw_metrics["ssim"].detach())
        composite_psnrs.append(composite_metrics["psnr"].detach())
        composite_ssims.append(composite_metrics["ssim"].detach())
        mask_l1s.append(torch.mean(torch.abs(alpha - mask)).detach())
    if not raw_psnrs:
        return {
            "psnr": 0.0,
            "ssim": 0.0,
            "composite_psnr": 0.0,
            "composite_ssim": 0.0,
            "mask_l1": 0.0,
            "view_count": 0.0,
        }
    return {
        "psnr": float(torch.stack(raw_psnrs).mean().cpu()),
        "ssim": float(torch.stack(raw_ssims).mean().cpu()),
        "composite_psnr": float(torch.stack(composite_psnrs).mean().cpu()),
        "composite_ssim": float(torch.stack(composite_ssims).mean().cpu()),
        "mask_l1": float(torch.stack(mask_l1s).mean().cpu()),
        "view_count": float(len(raw_psnrs)),
    }


def make_stage1_optimizer(model: WhiteTigerStage1Model, config: Stage1Config) -> torch.optim.Optimizer:
    high_frequency_lr = config.lr_groom * float(config.lr_high_frequency_shape_scale)
    return torch.optim.Adam(
        [
            {"params": [model.bary_logits], "lr": config.lr_root},
            {"params": [model.log_scale, model.translation], "lr": config.lr_calibration},
            {
                "params": [
                    model.groom.length_raw,
                    model.groom.root_width_raw,
                    model.groom.tip_width_ratio_raw,
                    model.groom.width_taper_raw,
                    model.groom.flow_xy,
                    model.groom.flow_strength_raw,
                    model.groom.lift_raw,
                    model.groom.stiffness_raw,
                    model.groom.opacity_raw,
                    model.groom.tip_opacity_ratio_raw,
                ],
                "lr": config.lr_groom,
            },
            {
                "params": [
                    model.groom.bend_raw,
                    model.groom.curl_radius_raw,
                    model.groom.curl_frequency_raw,
                    model.groom.curl_phase,
                    model.groom.frizz_raw,
                    model.groom.child_radius_raw,
                    model.groom.clump_strength_raw,
                ],
                "lr": high_frequency_lr,
            },
            {"params": [model.groom.root_color_raw, model.groom.tip_color_raw], "lr": config.lr_color},
        ]
    )


def rebuild_graph_edges(model: WhiteTigerStage1Model, k: int = 8) -> torch.Tensor:
    with torch.no_grad():
        roots, _, _ = model.roots_and_normals()
        return build_knn_edges(roots, k=k)


def parse_index_override(text: str, default: list[int]) -> list[int]:
    if not text.strip():
        return list(default)
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("view override is empty after parsing")
    return values


def train_white_tiger_stage1(config: Stage1Config) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("White Tiger Stage 1 requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data_root = resolve_project_path(config.data_root)
    mesh_path = resolve_project_path(config.mesh_path)
    output_dir = resolve_project_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_stage1_input_report(data_root, mesh_path, test_stride=config.test_stride)
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    if tuple(report.image_size or ()) != (config.expected_width, config.expected_height):
        raise RuntimeError(f"expected native {config.expected_width}x{config.expected_height}, got {report.image_size}")
    (output_dir / "stage1_inputs.json").write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    image_paths = list_images(Path(report.image_dir))
    mask_paths = list_images(Path(report.mask_dir))
    angle_paths = sorted((Path(report.orientation_root) / "angles").glob("*.png"))
    conf_paths = sorted((Path(report.orientation_root) / "vars").glob("*.npy"))
    if len(angle_paths) != report.image_count or len(conf_paths) != report.image_count:
        raise RuntimeError("orientation map count mismatch")

    mesh = read_obj_mesh(mesh_path)
    surface_roots = initialize_surface_roots_fps(
        mesh,
        config.root_count,
        candidate_multiplier=config.candidate_multiplier,
        seed=config.seed,
        fps_device=device,
    )
    root_report = validate_surface_roots(mesh, surface_roots)
    (output_dir / "root_init_report.json").write_text(json.dumps(root_report, indent=2) + "\n", encoding="utf-8")

    normals = face_normals_np(mesh)
    model = WhiteTigerStage1Model(
        mesh,
        normals,
        surface_roots.face_ids,
        surface_roots.barycentric,
        dense_groom_ranges(),
        device,
        init_scale=config.init_mesh_scale,
        init_translation=config.init_mesh_translation,
    )
    viewmats, ks = load_camera_tensors(data_root, device)
    width, height = config.expected_width, config.expected_height

    init_report = initialize_groom_from_projections(
        model,
        image_paths,
        mask_paths,
        angle_paths,
        conf_paths,
        viewmats,
        ks,
        report.train_indices,
        width,
        height,
        config,
        device,
    )
    (output_dir / "projected_init_report.json").write_text(json.dumps(init_report, indent=2) + "\n", encoding="utf-8")

    start_iteration = 0
    if config.resume_checkpoint:
        checkpoint_path = resolve_project_path(config.resume_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
        start_iteration = int(checkpoint.get("iteration", 0))

    graph_edges = rebuild_graph_edges(model, k=8)
    (output_dir / "root_graph.json").write_text(
        json.dumps({"edge_count": int(graph_edges.shape[0]), "knn": 8}, indent=2) + "\n",
        encoding="utf-8",
    )

    metric_computer = MetricComputer(compute_lpips=config.compute_lpips).to(device)
    mesh_depth_ctx = None
    if config.mesh_depth_clipping or config.mesh_backing_compositing:
        import nvdiffrast.torch as dr

        mesh_depth_ctx = dr.RasterizeCudaContext(device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    train_indices = parse_index_override(config.train_views, report.train_indices)
    test_indices = parse_index_override(config.test_views, report.test_indices)
    if start_iteration > 0 and len(train_indices) > 0:
        torch.randint(len(train_indices), (int(start_iteration),), generator=generator)
    optimizer = make_stage1_optimizer(model, config)
    root_accum = RootStatsWindow(int(model.face_ids.shape[0]), device)
    lifecycle_history: list[dict[str, float | int]] = []

    log_path = output_dir / "metrics.jsonl"
    start = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        for iteration in range(start_iteration + 1, config.iterations + 1):
            idx = int(train_indices[int(torch.randint(len(train_indices), (1,), generator=generator))])
            target = load_image(image_paths[idx], device)
            mask = load_mask(mask_paths[idx], device)
            target_orientation, target_conf = load_orientation(angle_paths[idx], conf_paths[idx], (width, height), bins=180, device=device)
            flow_loss_mask = mask * mask_edge_confidence(mask, config.projected_init_mask_edge_kernel)
            target_conf = target_conf * flow_loss_mask
            mesh_color = sample_backing_color(config, device, train=True)
            scene_bg = mesh_color if config.random_backing_color else scene_background_color(config, device)
            mesh_depth = render_model_mesh_depth(model, viewmats[idx], ks[idx], width, height, device=device, ctx=mesh_depth_ctx)
            backing_image = make_mesh_backing_image(mesh_depth, mesh_color, scene_bg)
            target_with_backing = composite_target(target, mask, backing_image)

            pred, alpha, gaussians, roots_local_for_grad, render_stats, render_info = render_view(
                model,
                viewmats[idx],
                ks[idx],
                width,
                height,
                config,
                background=mesh_color,
                mesh_depth=mesh_depth,
                backing_image=backing_image,
                retain_lifecycle_grad=True,
            )
            fixed_bg = scene_background_color(config, device).view(1, 1, 3)
            pred_fixed = render_info["raw_fur_image"] + (1.0 - alpha) * fixed_bg
            target_fixed = composite_target(target, mask, fixed_bg)
            rgb_weight = (0.25 + 1.75 * mask).detach()
            fixed_rgb_loss = (torch.abs(pred_fixed - target_fixed) * rgb_weight).sum() / torch.clamp(rgb_weight.sum() * 3.0, min=1.0)
            random_backing_loss = (
                torch.abs((pred - pred_fixed) - (target_with_backing - target_fixed)) * rgb_weight
            ).sum() / torch.clamp(rgb_weight.sum() * 3.0, min=1.0)
            rgb_loss = fixed_rgb_loss + float(config.random_backing_loss_weight) * random_backing_loss
            mask_loss = torch.mean(torch.abs(alpha - mask))
            pred_orientation, pred_orientation_conf = render_orientation_map(
                gaussians,
                viewmats[idx],
                ks[idx],
                width,
                height,
            )
            orient_loss, orient_detail_loss, orient_stats = orientation_map_losses(
                pred_orientation,
                pred_orientation_conf,
                target_orientation,
                target_conf,
                config.orientation_min_confidence,
            )
            smooth_loss = root_graph_smoothness(model.groom, graph_edges, model.root_observation_confidence)
            strand_shape_loss = strand_shape_consistency_loss(
                model.guide_strands_for_loss(min(config.samples, 32)),
                graph_edges,
                model.root_observation_confidence,
            )
            shape_prior_loss = groom_shape_prior(model.groom)
            _, _, roots_local = model.roots_and_normals()
            root_move_loss = torch.mean((roots_local - model.anchor_local).square())
            loss = (
                config.rgb_weight * rgb_loss
                + config.mask_weight * mask_loss
                + config.orientation_weight * orient_loss
                + config.orientation_detail_weight * orient_detail_loss
                + config.smooth_weight * smooth_loss
                + config.strand_shape_smooth_weight * strand_shape_loss
                + config.shape_prior_weight * shape_prior_loss
                + config.root_move_reg_weight * root_move_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            root_accum.add(root_points=roots_local_for_grad, gaussians=gaussians, infos=[render_info])
            optimizer.step()

            should_densify = (
                iteration >= config.densify_warmup
                and iteration <= config.densify_until
                and config.densify_interval > 0
                and (iteration - config.densify_warmup) % config.densify_interval == 0
            )
            should_prune = (
                iteration >= config.prune_start
                and config.prune_interval > 0
                and (iteration - config.prune_start) % config.prune_interval == 0
            )
            if should_densify or should_prune:
                stats = root_accum.to_stats()
                root_count_before = int(model.face_ids.shape[0])
                densify_cfg = DensifyConfig(
                    grad_threshold=float(config.densify_score_threshold) if should_densify else float("inf"),
                    visibility_threshold=1.0,
                    max_new_roots=int(config.max_splits_per_event) * int(config.split_children_per_parent),
                    children_per_parent=int(config.split_children_per_parent),
                    replace_parent=True,
                    neighbor_count=int(config.split_neighbor_count),
                    candidate_rings=int(config.split_candidate_rings),
                    candidate_face_count=int(config.split_candidate_face_count),
                    min_child_distance=float(config.split_min_child_distance),
                )
                prune_cfg = PruneConfig(
                    min_visible_count=1.0 if should_prune else -1.0,
                    min_contribution=float(config.prune_min_contribution) if should_prune else -1.0,
                    min_opacity=float(config.prune_min_opacity) if should_prune else 0.0,
                    max_prune_fraction=float(config.prune_max_fraction) if should_prune else 0.0,
                )
                update = propose_structure_update(
                    model.lifecycle_state(),
                    stats,
                    densify_cfg,
                    prune_cfg,
                    vertices=model.vertices,
                    faces=model.faces,
                )
                if should_densify and float(config.densify_min_contribution) > 0.0 and update.parent_indices.numel() > 0:
                    contribution = stats.gaussian_contrib_sum.reshape(-1)
                    keep_parent = contribution[update.parent_indices] >= float(config.densify_min_contribution)
                    if not bool(keep_parent.all()):
                        original_parents = update.parent_indices
                        kept_parents = update.parent_indices[keep_parent]
                        child_keep = torch.isin(update.child_parent_indices, kept_parents)
                        update.parent_indices = kept_parents
                        update.child_parent_indices = update.child_parent_indices[child_keep]
                        update.new_face_ids = update.new_face_ids[child_keep]
                        update.new_barycentric = update.new_barycentric[child_keep]
                        new_prune = torch.zeros_like(update.prune_mask)
                        if should_prune:
                            new_prune |= update.prune_mask
                            new_prune[original_parents] = False
                        new_prune[kept_parents] = True
                        update.prune_mask = new_prune
                changed = update.new_barycentric.numel() > 0 or bool(update.prune_mask.any())
                lifecycle_record = {
                    "iteration": iteration,
                    "root_count_before": root_count_before,
                    "selected_parent_count": int(update.parent_indices.numel()),
                    "inserted_child_count": int(update.new_barycentric.shape[0]),
                    "prune_count": int(update.prune_mask.sum().detach().cpu()),
                }
                if changed:
                    result = model.apply_structure_update(update)
                    graph_edges = rebuild_graph_edges(model, k=8)
                    optimizer = make_stage1_optimizer(model, config)
                    lifecycle_record.update(result)
                else:
                    lifecycle_record["root_count_after"] = root_count_before
                lifecycle_history.append(lifecycle_record)
                log.write(json.dumps({"lifecycle": lifecycle_record}) + "\n")
                log.flush()
                print(json.dumps({"lifecycle": lifecycle_record}), flush=True)
                root_accum = RootStatsWindow(int(model.face_ids.shape[0]), device)

            if iteration == 1 or iteration % config.eval_every == 0 or iteration == config.iterations:
                train_eval = evaluate(model, image_paths, mask_paths, viewmats, ks, train_indices, width, height, config, metric_computer, device, mesh_depth_ctx=mesh_depth_ctx)
                test_eval = evaluate(model, image_paths, mask_paths, viewmats, ks, test_indices, width, height, config, metric_computer, device, mesh_depth_ctx=mesh_depth_ctx)
                record = {
                    "iteration": iteration,
                    "elapsed_sec": round(time.time() - start, 3),
                    "loss": float(loss.detach().cpu()),
                    "rgb_l1": float(rgb_loss.detach().cpu()),
                    "fixed_rgb_l1": float(fixed_rgb_loss.detach().cpu()),
                    "random_backing_l1": float(random_backing_loss.detach().cpu()),
                    "mask_l1": float(mask_loss.detach().cpu()),
                    "orientation_loss": float(orient_loss.detach().cpu()),
                    "orientation_detail_loss": float(orient_detail_loss.detach().cpu()),
                    "smooth_loss": float(smooth_loss.detach().cpu()),
                    "strand_shape_smooth_loss": float(strand_shape_loss.detach().cpu()),
                    "shape_prior_loss": float(shape_prior_loss.detach().cpu()),
                    "root_move_loss": float(root_move_loss.detach().cpu()),
                    "train": train_eval,
                    "test": test_eval,
                    "render": render_stats,
                    "groom": groom_parameter_stats(model.groom),
                    "orientation": orient_stats,
                    "max_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                eval_dir = output_dir / f"iter_{iteration:06d}"
                save_image(eval_dir / f"view_{idx:02d}_train_pred.png", pred)
                save_image(eval_dir / f"view_{idx:02d}_train_pred_fixed_bg.png", pred_fixed)
                save_image(eval_dir / f"view_{idx:02d}_train_alpha.png", alpha)
                save_image(eval_dir / f"view_{idx:02d}_mesh_depth.png", depth_to_image(mesh_depth.depth))
                save_image(eval_dir / f"view_{idx:02d}_mesh_valid.png", mesh_depth.valid[..., None].float())
                save_image(eval_dir / f"view_{idx:02d}_backing.png", backing_image)
                save_image(eval_dir / f"view_{idx:02d}_target_with_backing.png", target_with_backing)
                save_image(eval_dir / f"view_{idx:02d}_flow_loss_mask.png", target_conf)
                save_image(eval_dir / f"view_{idx:02d}_pred_orientation_conf.png", pred_orientation_conf)
                save_image(eval_dir / f"view_{idx:02d}_pred_orientation.png", torch.cat([0.5 + 0.5 * pred_orientation, torch.full_like(pred_orientation[..., :1], 0.5)], dim=-1))
                target_double_vis = double_angle_orientation(target_orientation)
                save_image(eval_dir / f"view_{idx:02d}_target_orientation.png", torch.cat([0.5 + 0.5 * target_double_vis, target_conf.clamp(0.0, 1.0)], dim=-1))
                save_image(eval_dir / f"view_{idx:02d}_raw_diff.png", torch.abs(pred - target) * 4.0)
                save_image(eval_dir / f"view_{idx:02d}_composite_diff.png", torch.abs(pred - target_with_backing) * 4.0)
                save_clip_overlay(
                    eval_dir / f"view_{idx:02d}_clipped_visibility_overlay.png",
                    target_with_backing,
                    render_info["preclip_means"],
                    render_info["mesh_depth_keep_mask"],
                    viewmats[idx],
                    ks[idx],
                    behind_mesh_mask=render_info["mesh_depth_behind_mesh_mask"],
                )
                save_clip_overlay(
                    eval_dir / f"view_{idx:02d}_kept_gaussians_overlay.png",
                    target_with_backing,
                    render_info["preclip_means"],
                    render_info["mesh_depth_keep_mask"],
                    viewmats[idx],
                    ks[idx],
                    behind_mesh_mask=render_info["mesh_depth_behind_mesh_mask"],
                    mode="kept",
                )
                save_clip_overlay(
                    eval_dir / f"view_{idx:02d}_depth_clipped_gaussians_overlay.png",
                    target_with_backing,
                    render_info["preclip_means"],
                    render_info["mesh_depth_keep_mask"],
                    viewmats[idx],
                    ks[idx],
                    behind_mesh_mask=render_info["mesh_depth_behind_mesh_mask"],
                    mode="clipped",
                )
                diag_idx = int(test_indices[0] if test_indices else idx)
                diag_target = load_image(image_paths[diag_idx], device)
                diag_mask = load_mask(mask_paths[diag_idx], device)
                diag_mesh_color = sample_backing_color(config, device, train=False)
                diag_scene_bg = scene_background_color(config, device)
                diag_mesh_depth = render_model_mesh_depth(
                    model,
                    viewmats[diag_idx],
                    ks[diag_idx],
                    width,
                    height,
                    device=device,
                    ctx=mesh_depth_ctx,
                )
                diag_backing = make_mesh_backing_image(diag_mesh_depth, diag_mesh_color, diag_scene_bg)
                diag_pred, diag_alpha, _, _, _, diag_info = render_view(
                    model,
                    viewmats[diag_idx],
                    ks[diag_idx],
                    width,
                    height,
                    config,
                    background=diag_mesh_color,
                    mesh_depth=diag_mesh_depth,
                    backing_image=diag_backing,
                )
                diag_target_eval = composite_target(diag_target, diag_mask, diag_backing)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_gt.png", diag_target)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_target.png", diag_target_eval)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_pred.png", diag_pred)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_alpha.png", diag_alpha)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_raw_diff_x4.png", torch.abs(diag_pred - diag_target) * 4.0)
                save_image(
                    eval_dir / f"view_{diag_idx:02d}_eval_composite_diff_x4.png",
                    torch.abs(diag_pred - diag_target_eval) * 4.0,
                )
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_mesh_depth.png", depth_to_image(diag_mesh_depth.depth))
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_mesh_valid.png", diag_mesh_depth.valid[..., None].float())
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_backing.png", diag_backing)
                save_clip_overlay(
                    eval_dir / f"view_{diag_idx:02d}_eval_clipped_visibility_overlay.png",
                    diag_target_eval,
                    diag_info["preclip_means"],
                    diag_info["mesh_depth_keep_mask"],
                    viewmats[diag_idx],
                    ks[diag_idx],
                    behind_mesh_mask=diag_info["mesh_depth_behind_mesh_mask"],
                )
                save_clip_overlay(
                    eval_dir / f"view_{diag_idx:02d}_eval_kept_gaussians_overlay.png",
                    diag_target_eval,
                    diag_info["preclip_means"],
                    diag_info["mesh_depth_keep_mask"],
                    viewmats[diag_idx],
                    ks[diag_idx],
                    behind_mesh_mask=diag_info["mesh_depth_behind_mesh_mask"],
                    mode="kept",
                )
                save_clip_overlay(
                    eval_dir / f"view_{diag_idx:02d}_eval_depth_clipped_gaussians_overlay.png",
                    diag_target_eval,
                    diag_info["preclip_means"],
                    diag_info["mesh_depth_keep_mask"],
                    viewmats[diag_idx],
                    ks[diag_idx],
                    behind_mesh_mask=diag_info["mesh_depth_behind_mesh_mask"],
                    mode="clipped",
                )

            if config.save_every > 0 and (iteration % config.save_every == 0 or iteration == config.iterations):
                torch.save(
                    {
                        "iteration": iteration,
                        "config": asdict(config),
                        "model": model.state_dict(),
                        "lifecycle_history": lifecycle_history,
                    },
                    output_dir / f"checkpoint_{iteration:06d}.pt",
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train clean AniGroom White Tiger Stage 1.")
    parser.add_argument("--alignment-config", default="configs/white_tiger_mesh_alignment.json")
    parser.add_argument("--data-root", default="data/neuralfur_work/whiteTiger_processed/roaringwalk")
    parser.add_argument("--mesh-path", default="data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj")
    parser.add_argument("--output-dir", default="outputs/white_tiger_stage1")
    parser.add_argument("--root-count", type=int, default=10000)
    parser.add_argument("--candidate-multiplier", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--test-stride", type=int, default=6)
    parser.add_argument("--train-views", default="")
    parser.add_argument("--test-views", default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--expected-width", type=int, default=1920)
    parser.add_argument("--expected-height", type=int, default=1080)
    parser.add_argument("--init-mesh-scale", type=float, default=1.28)
    parser.add_argument("--init-mesh-translation", type=float, nargs=3, default=[0.0, 0.32, 0.02])
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--min-segments", type=int, default=6)
    parser.add_argument("--max-segments", type=int, default=22)
    parser.add_argument("--child-count", type=int, default=8)
    parser.add_argument("--projected-init-views", type=int, default=24)
    parser.add_argument("--projected-init-min-confidence", type=float, default=0.08)
    parser.add_argument("--projected-init-depth-abs-tolerance", type=float, default=0.03)
    parser.add_argument("--projected-init-depth-rel-tolerance", type=float, default=0.01)
    parser.add_argument("--projected-init-local-depth-kernel", type=int, default=7)
    parser.add_argument("--projected-init-front-normal-z", type=float, default=0.15)
    parser.add_argument("--projected-init-mask-edge-kernel", type=int, default=9)
    parser.add_argument("--projected-init-view-angle-power", type=float, default=1.0)
    parser.add_argument("--lr-groom", type=float, default=1.4e-2)
    parser.add_argument("--lr-high-frequency-shape-scale", type=float, default=0.35)
    parser.add_argument("--lr-color", type=float, default=2.0e-2)
    parser.add_argument("--lr-root", type=float, default=7.5e-4)
    parser.add_argument("--lr-calibration", type=float, default=5.0e-4)
    parser.add_argument("--rgb-weight", type=float, default=1.0)
    parser.add_argument("--random-backing-loss-weight", type=float, default=0.25)
    parser.add_argument("--mask-weight", type=float, default=0.15)
    parser.add_argument("--orientation-weight", type=float, default=0.08)
    parser.add_argument("--orientation-detail-weight", type=float, default=0.02)
    parser.add_argument("--smooth-weight", type=float, default=0.04)
    parser.add_argument("--strand-shape-smooth-weight", type=float, default=0.0)
    parser.add_argument("--shape-prior-weight", type=float, default=0.015)
    parser.add_argument("--root-move-reg-weight", type=float, default=0.003)
    parser.add_argument("--orientation-min-confidence", type=float, default=0.08)
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--black-background", action="store_true")
    parser.add_argument("--disable-random-backing-color", action="store_true")
    parser.add_argument("--backing-color-min", type=float, default=0.05)
    parser.add_argument("--backing-color-max", type=float, default=0.85)
    parser.add_argument("--disable-mesh-depth-clipping", action="store_true")
    parser.add_argument("--mesh-depth-abs-tolerance", type=float, default=0.018)
    parser.add_argument("--mesh-depth-rel-tolerance", type=float, default=0.004)
    parser.add_argument("--mesh-depth-local-kernel", type=int, default=1)
    parser.add_argument("--disable-mesh-backing-compositing", action="store_true")
    parser.add_argument("--densify-warmup", type=int, required=True)
    parser.add_argument("--densify-interval", type=int, required=True)
    parser.add_argument("--densify-until", type=int, required=True)
    parser.add_argument("--densify-score-threshold", type=float, required=True)
    parser.add_argument("--densify-min-contribution", type=float, required=True)
    parser.add_argument("--max-splits-per-event", type=int, required=True)
    parser.add_argument("--split-children-per-parent", type=int, required=True)
    parser.add_argument("--split-neighbor-count", type=int, required=True)
    parser.add_argument("--split-candidate-rings", type=int, required=True)
    parser.add_argument("--split-candidate-face-count", type=int, required=True)
    parser.add_argument("--split-min-child-distance", type=float, required=True)
    parser.add_argument("--prune-start", type=int, required=True)
    parser.add_argument("--prune-interval", type=int, required=True)
    parser.add_argument("--prune-min-contribution", type=float, required=True)
    parser.add_argument("--prune-min-opacity", type=float, required=True)
    parser.add_argument("--prune-max-fraction", type=float, required=True)
    parser.add_argument("--resume-checkpoint", default="")
    return parser


def config_from_args(args: argparse.Namespace) -> Stage1Config:
    config = Stage1Config(
        data_root=args.data_root,
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        root_count=args.root_count,
        candidate_multiplier=args.candidate_multiplier,
        iterations=args.iterations,
        eval_every=args.eval_every,
        save_every=args.save_every,
        test_stride=args.test_stride,
        train_views=args.train_views,
        test_views=args.test_views,
        seed=args.seed,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        init_mesh_scale=args.init_mesh_scale,
        init_mesh_translation=tuple(float(v) for v in args.init_mesh_translation),
        samples=args.samples,
        min_segments=args.min_segments,
        max_segments=args.max_segments,
        child_count=args.child_count,
        projected_init_views=args.projected_init_views,
        projected_init_min_confidence=args.projected_init_min_confidence,
        projected_init_depth_abs_tolerance=args.projected_init_depth_abs_tolerance,
        projected_init_depth_rel_tolerance=args.projected_init_depth_rel_tolerance,
        projected_init_local_depth_kernel=args.projected_init_local_depth_kernel,
        projected_init_front_normal_z=args.projected_init_front_normal_z,
        projected_init_mask_edge_kernel=args.projected_init_mask_edge_kernel,
        projected_init_view_angle_power=args.projected_init_view_angle_power,
        lr_groom=args.lr_groom,
        lr_high_frequency_shape_scale=args.lr_high_frequency_shape_scale,
        lr_color=args.lr_color,
        lr_root=args.lr_root,
        lr_calibration=args.lr_calibration,
        rgb_weight=args.rgb_weight,
        random_backing_loss_weight=args.random_backing_loss_weight,
        mask_weight=args.mask_weight,
        orientation_weight=args.orientation_weight,
        orientation_detail_weight=args.orientation_detail_weight,
        smooth_weight=args.smooth_weight,
        strand_shape_smooth_weight=args.strand_shape_smooth_weight,
        shape_prior_weight=args.shape_prior_weight,
        root_move_reg_weight=args.root_move_reg_weight,
        orientation_min_confidence=args.orientation_min_confidence,
        compute_lpips=args.compute_lpips,
        white_background=not args.black_background,
        random_backing_color=not args.disable_random_backing_color,
        backing_color_min=args.backing_color_min,
        backing_color_max=args.backing_color_max,
        mesh_depth_clipping=not args.disable_mesh_depth_clipping,
        mesh_depth_abs_tolerance=args.mesh_depth_abs_tolerance,
        mesh_depth_rel_tolerance=args.mesh_depth_rel_tolerance,
        mesh_depth_local_kernel=args.mesh_depth_local_kernel,
        mesh_backing_compositing=not args.disable_mesh_backing_compositing,
        densify_warmup=args.densify_warmup,
        densify_interval=args.densify_interval,
        densify_until=args.densify_until,
        densify_score_threshold=args.densify_score_threshold,
        densify_min_contribution=args.densify_min_contribution,
        max_splits_per_event=args.max_splits_per_event,
        split_children_per_parent=args.split_children_per_parent,
        split_neighbor_count=args.split_neighbor_count,
        split_candidate_rings=args.split_candidate_rings,
        split_candidate_face_count=args.split_candidate_face_count,
        split_min_child_distance=args.split_min_child_distance,
        prune_start=args.prune_start,
        prune_interval=args.prune_interval,
        prune_min_contribution=args.prune_min_contribution,
        prune_min_opacity=args.prune_min_opacity,
        prune_max_fraction=args.prune_max_fraction,
        resume_checkpoint=args.resume_checkpoint,
    )
    return config


def main() -> None:
    args = build_arg_parser().parse_args()
    apply_alignment_to_namespace(args, load_alignment_config(args.alignment_config), include_uv=False)
    train_white_tiger_stage1(config_from_args(args))


if __name__ == "__main__":
    main()

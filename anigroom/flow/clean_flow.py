from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .direction_geometry import parallel_transport_vectors


EPS = 1.0e-8


@dataclass(frozen=True)
class CleanFlowTargets:
    """Visibility-cleaned 3D groom directions sampled from external line evidence."""

    points: torch.Tensor
    normals: torch.Tensor
    directions: torch.Tensor
    confidence: torch.Tensor
    anchor_confidence: torch.Tensor
    lambda_values: torch.Tensor
    shell_height: torch.Tensor
    raw_shell_height: torch.Tensor
    local_spacing: torch.Tensor
    observed: torch.Tensor
    anchor: torch.Tensor
    source_path: str


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, dim=-1, eps=EPS)


def load_clean_flow_targets(path: str | Path, device: torch.device | str | None = None) -> CleanFlowTargets:
    """Load the formal cleaned-flow NPZ produced by the multiview flow fusion tool."""

    source = Path(path)
    data = np.load(source)
    if "root_points" not in data:
        raise RuntimeError(f"clean-flow target is missing root_points: {source}")
    if "root_normals" not in data:
        raise RuntimeError(f"clean-flow target is missing root_normals: {source}")
    if "cleaned_directed_flow3d" in data:
        directions_np = data["cleaned_directed_flow3d"]
    elif "directed_flow3d" in data:
        directions_np = data["directed_flow3d"]
    elif "flow3d" in data:
        directions_np = data["flow3d"]
    else:
        raise RuntimeError(f"clean-flow target is missing a 3D direction field: {source}")

    dev = torch.device(device) if device is not None else None
    points = torch.from_numpy(data["root_points"].astype(np.float32)).to(device=dev)
    normals = _normalize(torch.from_numpy(data["root_normals"].astype(np.float32)).to(device=dev))
    directions = _normalize(torch.from_numpy(directions_np.astype(np.float32)).to(device=dev))

    if "shell_confidence" in data and data["shell_confidence"].ndim == 1:
        confidence_np = data["shell_confidence"].astype(np.float32)
    elif "weight" in data:
        weight = data["weight"].astype(np.float32)
        scale = np.quantile(weight[weight > 0.0], 0.95) if np.any(weight > 0.0) else 1.0
        confidence_np = np.clip(weight / max(float(scale), EPS), 0.0, 1.0)
    elif "direction_anchor_confidence" in data:
        confidence_np = data["direction_anchor_confidence"].astype(np.float32)
    else:
        confidence_np = np.ones((points.shape[0],), dtype=np.float32)

    if "direction_anchor_confidence" in data:
        anchor_confidence_np = data["direction_anchor_confidence"].astype(np.float32)
    else:
        anchor_confidence_np = confidence_np.copy()

    if "observed" in data:
        observed_np = data["observed"].astype(bool)
    else:
        observed_np = np.isfinite(directions_np).all(axis=-1)

    if "direction_anchor" in data:
        anchor_np = data["direction_anchor"].astype(bool)
    elif "direction_anchor_confidence" in data:
        anchor_np = anchor_confidence_np > 0.0
    else:
        anchor_np = observed_np

    if "cleaned_direction_lambda" in data:
        lambda_np = data["cleaned_direction_lambda"].astype(np.float32)
    elif "direction_lambda" in data:
        lambda_np = data["direction_lambda"].astype(np.float32)
    else:
        lambda_np = np.zeros((points.shape[0],), dtype=np.float32)

    shell_height_np = data["shell_h"].astype(np.float32) if "shell_h" in data else np.zeros((points.shape[0],), dtype=np.float32)
    raw_shell_height_np = data["raw_shell_h"].astype(np.float32) if "raw_shell_h" in data else shell_height_np.copy()
    local_spacing_np = data["local_spacing"].astype(np.float32) if "local_spacing" in data else np.ones((points.shape[0],), dtype=np.float32)

    confidence = torch.from_numpy(confidence_np).to(device=dev).reshape(-1).clamp(0.0, 1.0)
    anchor_confidence = torch.from_numpy(anchor_confidence_np).to(device=dev).reshape(-1).clamp(0.0, 1.0)
    observed = torch.from_numpy(observed_np).to(device=dev).reshape(-1).bool()
    anchor = torch.from_numpy(anchor_np).to(device=dev).reshape(-1).bool() & observed & (anchor_confidence > 0.0)
    lambda_values = torch.from_numpy(lambda_np).to(device=dev).reshape(-1)
    shell_height = torch.from_numpy(shell_height_np).to(device=dev).reshape(-1).clamp_min(0.0)
    raw_shell_height = torch.from_numpy(raw_shell_height_np).to(device=dev).reshape(-1).clamp_min(0.0)
    local_spacing = torch.from_numpy(local_spacing_np).to(device=dev).reshape(-1).clamp_min(EPS)

    if points.ndim != 2 or points.shape[-1] != 3:
        raise RuntimeError(f"clean-flow root_points must be [N, 3], got {tuple(points.shape)}")
    if directions.shape != points.shape:
        raise RuntimeError(f"clean-flow directions must match points, got {tuple(directions.shape)} vs {tuple(points.shape)}")
    if normals.shape != points.shape:
        raise RuntimeError(f"clean-flow normals must match points, got {tuple(normals.shape)} vs {tuple(points.shape)}")
    if confidence.shape[0] != points.shape[0]:
        raise RuntimeError("clean-flow confidence length does not match root_points")
    if anchor_confidence.shape[0] != points.shape[0]:
        raise RuntimeError("clean-flow anchor confidence length does not match root_points")
    if shell_height.shape[0] != points.shape[0]:
        raise RuntimeError("clean-flow shell_h length does not match root_points")

    return CleanFlowTargets(
        points=points,
        normals=normals,
        directions=directions,
        confidence=confidence,
        anchor_confidence=anchor_confidence,
        lambda_values=lambda_values,
        shell_height=shell_height,
        raw_shell_height=raw_shell_height,
        local_spacing=local_spacing,
        observed=observed,
        anchor=anchor,
        source_path=str(source),
    )


@torch.no_grad()
def sample_clean_flow_targets(
    targets: CleanFlowTargets,
    query_points: torch.Tensor,
    query_normals: torch.Tensor,
    *,
    k: int = 8,
    chunk_size: int = 2048,
    confidence_floor: float = 0.0,
    anchor_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Interpolate scalar targets and a surface-aware directed field onto mesh roots."""

    query_points = query_points.to(device=targets.points.device, dtype=targets.points.dtype)
    query_normals = _normalize(
        query_normals.to(device=targets.points.device, dtype=targets.points.dtype)
    )
    if query_points.shape != query_normals.shape or query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError(
            f"query_points and query_normals must both be [N, 3], got "
            f"{tuple(query_points.shape)} and {tuple(query_normals.shape)}"
        )
    source_confidence_all = targets.anchor_confidence if anchor_only else targets.confidence
    source_mask = targets.anchor if anchor_only else targets.observed
    source_mask = source_mask & (source_confidence_all >= float(confidence_floor))
    source_ids = torch.nonzero(source_mask, as_tuple=False).reshape(-1)
    if source_ids.numel() == 0:
        empty_conf = torch.zeros((query_points.shape[0],), device=query_points.device, dtype=query_points.dtype)
        return {
            "direction": torch.zeros_like(query_points),
            "confidence": empty_conf,
            "lambda": empty_conf,
            "shell_height": empty_conf,
            "raw_shell_height": empty_conf,
            "local_spacing": torch.ones_like(empty_conf),
            "valid": torch.zeros_like(empty_conf, dtype=torch.bool),
            "nearest_distance": torch.full_like(empty_conf, float("inf")),
        }

    source_points = targets.points[source_ids]
    source_normals = targets.normals[source_ids]
    source_dirs = targets.directions[source_ids]
    source_conf = source_confidence_all[source_ids].to(dtype=query_points.dtype)
    source_lambda = targets.lambda_values[source_ids].to(dtype=query_points.dtype)
    source_shell_height = targets.shell_height[source_ids].to(dtype=query_points.dtype)
    source_raw_shell_height = targets.raw_shell_height[source_ids].to(dtype=query_points.dtype)
    source_local_spacing = targets.local_spacing[source_ids].to(dtype=query_points.dtype)
    k_eff = max(1, min(int(k), int(source_ids.numel())))

    directions = []
    confidence = []
    lambdas = []
    shell_heights = []
    raw_shell_heights = []
    local_spacings = []
    nearest = []
    for begin in range(0, int(query_points.shape[0]), int(chunk_size)):
        q = query_points[begin : begin + int(chunk_size)]
        q_normals = query_normals[begin : begin + int(chunk_size)]
        dist = torch.cdist(q, source_points)
        values, ids = torch.topk(dist, k=k_eff, dim=-1, largest=False)
        weights = source_conf[ids] / values.clamp_min(1.0e-6).square()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)

        normal_dot = (q_normals @ source_normals.transpose(0, 1)).clamp(-1.0, 1.0)
        normal_compatibility = (0.5 * (normal_dot + 1.0)).square()
        surface_cost = dist / normal_compatibility.sqrt().clamp_min(EPS)
        direction_cost, direction_ids = torch.topk(surface_cost, k=k_eff, dim=-1, largest=False)
        del direction_cost
        direction_distances = dist.gather(1, direction_ids)
        direction_compatibility = normal_compatibility.gather(1, direction_ids)
        direction_weights = (
            source_conf[direction_ids]
            * direction_compatibility
            / direction_distances.clamp_min(1.0e-6).square()
        )
        direction_weight_sum = direction_weights.sum(dim=1, keepdim=True)
        if bool((direction_weight_sum <= EPS).any()):
            raise RuntimeError("surface-aware clean-flow interpolation produced an empty direction neighborhood")
        direction_weights = direction_weights / direction_weight_sum
        source_normal_neighbors = source_normals[direction_ids]
        target_normal_neighbors = q_normals[:, None, :].expand_as(source_normal_neighbors)
        transported_directions = parallel_transport_vectors(
            source_dirs[direction_ids],
            source_normal_neighbors,
            target_normal_neighbors,
        )
        direction_sum = (transported_directions * direction_weights[..., None]).sum(dim=1)
        if bool((torch.linalg.vector_norm(direction_sum, dim=-1) <= EPS).any()):
            raise RuntimeError("surface-aware clean-flow interpolation produced a cancelled direction")
        directions.append(_normalize(direction_sum))
        spacing_value = (source_local_spacing[ids] * weights).sum(dim=1).clamp_min(EPS)
        distance_decay = torch.exp(-((values[:, 0] / (3.0 * spacing_value.clamp_min(EPS))) ** 2))
        confidence.append(((source_conf[ids] * weights).sum(dim=1) * distance_decay).clamp(0.0, 1.0))
        lambdas.append((source_lambda[ids] * weights).sum(dim=1))
        shell_heights.append((source_shell_height[ids] * weights).sum(dim=1).clamp_min(0.0))
        raw_shell_heights.append((source_raw_shell_height[ids] * weights).sum(dim=1).clamp_min(0.0))
        local_spacings.append(spacing_value)
        nearest.append(values[:, 0])

    direction = torch.cat(directions, dim=0)
    conf = torch.cat(confidence, dim=0)
    lambda_values = torch.cat(lambdas, dim=0)
    shell_height = torch.cat(shell_heights, dim=0)
    raw_shell_height = torch.cat(raw_shell_heights, dim=0)
    local_spacing = torch.cat(local_spacings, dim=0)
    nearest_distance = torch.cat(nearest, dim=0)
    return {
        "direction": direction,
        "confidence": conf,
        "lambda": lambda_values,
        "shell_height": shell_height,
        "raw_shell_height": raw_shell_height,
        "local_spacing": local_spacing,
        "valid": conf > 0.0,
        "nearest_distance": nearest_distance,
    }


def groom_direction_3d(groom, normals: torch.Tensor, tangents: torch.Tensor, bitangents: torch.Tensor) -> torch.Tensor:
    """Full initial hair direction implied by lift plus local brushed flow."""

    return controls_direction_3d(groom.flow_xy, groom.flow_strength, groom.lift, normals, tangents, bitangents)


def controls_direction_3d(
    flow_xy: torch.Tensor,
    flow_strength: torch.Tensor,
    lift: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
) -> torch.Tensor:
    """Full 3D direction implied by explicit flow, flow strength, and lift controls."""

    flow_xy = _normalize(flow_xy)
    tangent_flow = _normalize(flow_xy[:, [0]] * tangents + flow_xy[:, [1]] * bitangents)
    return _normalize(lift * normals + flow_strength * tangent_flow)


def direction_to_local_controls(
    directions: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    lambda_values: torch.Tensor,
    *,
    lift_min: float,
    lift_max: float,
    lambda_low: torch.Tensor | float | None = None,
    lambda_high: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 3D clean-flow directions into local flow_xy plus a lift value."""

    directions = _normalize(directions)
    normals = _normalize(normals)
    tangents = _normalize(tangents)
    bitangents = _normalize(bitangents)
    tangent_part = directions - (directions * normals).sum(dim=-1, keepdim=True) * normals
    flow_xy = torch.stack(
        [
            (tangent_part * tangents).sum(dim=-1),
            (tangent_part * bitangents).sum(dim=-1),
        ],
        dim=-1,
    )
    flow_xy = _normalize(flow_xy)

    if lambda_low is None or lambda_high is None:
        valid_lambda = lambda_values[torch.isfinite(lambda_values)]
        if valid_lambda.numel() >= 4:
            lambda_low = torch.quantile(valid_lambda, 0.10)
            lambda_high = torch.quantile(valid_lambda, 0.90)
        else:
            lambda_low = lambda_values.new_tensor(0.0)
            lambda_high = lambda_values.new_tensor(1.0)
    lo = torch.as_tensor(lambda_low, device=lambda_values.device, dtype=lambda_values.dtype)
    hi = torch.as_tensor(lambda_high, device=lambda_values.device, dtype=lambda_values.dtype)
    alpha = ((lambda_values - lo) / (hi - lo).clamp_min(EPS)).clamp(0.0, 1.0)
    lift = float(lift_min) + (float(lift_max) - float(lift_min)) * alpha
    return flow_xy, lift.reshape(-1, 1)


def direction_to_flow_lift_strength(
    directions: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    *,
    lift_bounds: tuple[float, float],
    flow_strength_bounds: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a 3D direction to local controls while preserving its 3D angle.

    ``direction_to_local_controls`` is useful when a scalar lift proxy is
    supplied externally.  For formal clean-flow initialization we already have a
    3D direction, so lift and tangential strength must be solved together;
    otherwise high-lift hairs get flattened into the tangent plane.
    """

    directions = _normalize(directions)
    normals = _normalize(normals)
    tangents = _normalize(tangents)
    bitangents = _normalize(bitangents)
    normal_component = (directions * normals).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    tangent_part = directions - normal_component * normals
    tangent_component = torch.linalg.norm(tangent_part, dim=-1, keepdim=True).clamp_min(EPS)
    tangent_dir = tangent_part / tangent_component
    flow_xy = torch.stack(
        [
            (tangent_dir * tangents).sum(dim=-1),
            (tangent_dir * bitangents).sum(dim=-1),
        ],
        dim=-1,
    )
    flow_xy = _normalize(flow_xy)

    lift_lo, lift_hi = float(lift_bounds[0]), float(lift_bounds[1])
    flow_lo, flow_hi = float(flow_strength_bounds[0]), float(flow_strength_bounds[1])
    scale_from_lift = lift_hi / normal_component.clamp_min(1.0e-4)
    scale_from_flow = flow_hi / tangent_component.clamp_min(1.0e-4)
    scale = torch.minimum(scale_from_lift, scale_from_flow)
    lift = (scale * normal_component).clamp(lift_lo, lift_hi)
    flow_strength = (scale * tangent_component).clamp(flow_lo, flow_hi)
    return flow_xy, lift, flow_strength


def clean_flow_anchor_loss(
    predicted_direction: torch.Tensor,
    target_direction: torch.Tensor,
    confidence: torch.Tensor,
    *,
    min_confidence: float = 0.0,
) -> torch.Tensor:
    """Directed 3D anchor loss for cleaned flow targets."""

    confidence = confidence.reshape(-1).to(device=predicted_direction.device, dtype=predicted_direction.dtype)
    mask = confidence >= float(min_confidence)
    if not bool(mask.any()):
        return predicted_direction.sum() * 0.0
    pred = _normalize(predicted_direction[mask])
    target = _normalize(target_direction.to(device=predicted_direction.device, dtype=predicted_direction.dtype)[mask])
    weight = confidence[mask]
    loss = 1.0 - (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return (loss * weight).sum() / weight.sum().clamp_min(EPS)


def clean_flow_smoothness_loss(
    directions: torch.Tensor,
    edges: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mesh-neighborhood smoothing on actual 3D hair direction, not local 2D flow."""

    if edges.numel() == 0:
        return directions.sum() * 0.0
    src, dst = edges[:, 0], edges[:, 1]
    diff = _normalize(directions[src]) - _normalize(directions[dst])
    value = diff.square().mean(dim=-1)
    if confidence is None:
        return value.mean()
    conf = confidence.reshape(-1).to(device=directions.device, dtype=directions.dtype).clamp(0.0, 1.0)
    edge_weight = 0.25 + 0.75 * torch.minimum(conf[src], conf[dst])
    return (value * edge_weight).sum() / edge_weight.sum().clamp_min(EPS)

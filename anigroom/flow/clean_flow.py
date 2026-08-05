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
    """Decode the groom's direct local 3D direction into world space."""

    local = _normalize(groom.direction_local)
    return _normalize(
        local[..., 0:1] * tangents
        + local[..., 1:2] * bitangents
        + local[..., 2:3] * normals
    )


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
    *,
    normals: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mesh-neighborhood smoothing on actual 3D hair direction.

    When normals are supplied, neighbor directions are parallel transported to
    the source tangent plane before comparison. This keeps curvature of the
    carrier surface from being mistaken for a discontinuity in the groom.
    """

    if edges.numel() == 0:
        return directions.sum() * 0.0
    src, dst = edges[:, 0], edges[:, 1]
    if normals is None:
        neighbor_direction = _normalize(directions[dst])
    else:
        neighbor_direction = parallel_transport_vectors(
            directions[dst],
            normals[dst],
            normals[src],
        )
    diff = _normalize(directions[src]) - neighbor_direction
    value = diff.square().mean(dim=-1)
    if confidence is None:
        return value.mean()
    conf = confidence.reshape(-1).to(device=directions.device, dtype=directions.dtype).clamp(0.0, 1.0)
    # Reliable directions are already constrained by the anchor loss.  The
    # smoothness term should propagate that field through uncertain regions,
    # rather than spending most of its weight on already-observed edges.
    edge_weight = 1.0 + (1.0 - torch.minimum(conf[src], conf[dst]))
    return (value * edge_weight).sum() / edge_weight.sum().clamp_min(EPS)

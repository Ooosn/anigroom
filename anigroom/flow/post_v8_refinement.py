"""Automatic post-V8 multiview refit and confidence propagation.

The V8 field is treated as the initial complete directed field.  Each outer
cycle has exactly two owners:

1. a per-view-confidence-weighted refit changes only the tangent-plane angle;
2. the accepted V8 joint-confidence propagation repairs weak surface regions.

V6/V7 axes, signs, ratios, and confidence values are not recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from .confidence_guided_direction import refine_confidence_guided_directed_flow
from .direction_geometry import parallel_transport_vectors


EPS = 1.0e-8


@dataclass(frozen=True)
class PostV8RefinementConfig:
    ba_iterations: int = 120
    ba_learning_rate: float = 0.03
    ba_relative_tolerance: float = 1.0e-7
    ba_patience: int = 20
    outer_max_cycles: int = 8
    outer_relative_tolerance: float = 1.0e-4
    outer_change_p95_tolerance_deg: float = 0.10
    acceptance_tolerance: float = 1.0e-7
    backtracking_steps: int = 6

    def __post_init__(self) -> None:
        integer_values = {
            "ba_iterations": self.ba_iterations,
            "ba_patience": self.ba_patience,
            "outer_max_cycles": self.outer_max_cycles,
            "backtracking_steps": self.backtracking_steps,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        positive_values = {
            "ba_learning_rate": self.ba_learning_rate,
            "ba_relative_tolerance": self.ba_relative_tolerance,
            "outer_relative_tolerance": self.outer_relative_tolerance,
            "outer_change_p95_tolerance_deg": self.outer_change_p95_tolerance_deg,
            "acceptance_tolerance": self.acceptance_tolerance,
        }
        for name, value in positive_values.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class TangentAngleRefitResult:
    direction: torch.Tensor
    tangent_coordinate: torch.Tensor
    initial_data_energy: float
    final_data_energy: float
    history: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class PostV8RefinementResult:
    direction: torch.Tensor
    cycle_directions: tuple[torch.Tensor, ...]
    report: dict[str, Any]


def _unit(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim < 2 or value.shape[-1] != 3:
        raise ValueError(f"{name} must end in dimension 3")
    if value.is_complex() or not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a real floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    magnitude = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if bool((magnitude <= EPS).any()):
        raise ValueError(f"{name} contains a zero vector")
    return value / magnitude


def _validate_core_shapes(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    projection_points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[int, int]:
    if direction.ndim != 2 or direction.shape[1] != 3:
        raise ValueError("direction must have shape [N, 3]")
    n = int(direction.shape[0])
    for name, value in (
        ("normals", normals),
        ("projection_points", projection_points),
    ):
        if value.shape != (n, 3):
            raise ValueError(f"{name} must have shape [N, 3]")
    if per_view_axes.ndim != 3 or per_view_axes.shape[1:] != (n, 3):
        raise ValueError("per_view_axes must have shape [V, N, 3]")
    v = int(per_view_axes.shape[0])
    if per_view_weights.shape != (v, n):
        raise ValueError("per_view_weights must have shape [V, N]")
    if viewmats.shape != (v, 4, 4):
        raise ValueError("viewmats must have shape [V, 4, 4]")
    if intrinsics.shape != (v, 3, 3):
        raise ValueError("intrinsics must have shape [V, 3, 3]")
    tensors = (
        direction,
        normals,
        projection_points,
        per_view_axes,
        per_view_weights,
        viewmats,
        intrinsics,
    )
    if any(value.device != direction.device for value in tensors):
        raise ValueError("all refit tensors must share one device")
    if any(
        value.is_complex() or not torch.is_floating_point(value) for value in tensors
    ):
        raise TypeError("all refit tensors must be real floating-point tensors")
    if any(value.dtype != direction.dtype for value in tensors):
        raise TypeError("all refit tensors must share one dtype")
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("refit tensors contain non-finite values")
    if bool((per_view_weights < 0.0).any()):
        raise ValueError("per_view_weights must be non-negative")
    return n, v


def _tangent_frame(
    direction: torch.Tensor,
    normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normal = _unit(normals, name="normals")
    current = _unit(direction, name="direction")
    normal_component = (current * normal).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    tangent = current - normal_component * normal
    tangent_magnitude = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
    axis_id = torch.argmin(normal.abs(), dim=-1)
    helper = F.one_hot(axis_id, num_classes=3).to(
        device=normal.device,
        dtype=normal.dtype,
    )
    fallback = _unit(torch.cross(normal, helper, dim=-1), name="fallback tangent")
    tangent_axis = torch.where(
        tangent_magnitude > 1.0e-7,
        tangent / tangent_magnitude.clamp_min(EPS),
        fallback,
    )
    bitangent = _unit(
        torch.cross(normal, tangent_axis, dim=-1),
        name="bitangent",
    )
    return normal_component, tangent_magnitude, tangent_axis, bitangent


def compose_tangent_angle(
    *,
    normals: torch.Tensor,
    frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tangent_coordinate: torch.Tensor,
) -> torch.Tensor:
    normal = _unit(normals, name="normals")
    normal_component, tangent_magnitude, tangent_axis, bitangent = frame
    if tangent_coordinate.shape != (normal.shape[0],):
        raise ValueError("tangent_coordinate must have shape [N]")
    rotated_axis = _unit(
        tangent_axis + tangent_coordinate[:, None] * bitangent,
        name="rotated tangent axis",
    )
    return _unit(
        normal_component * normal + tangent_magnitude * rotated_axis,
        name="composed direction",
    )


def project_direction_differentials(
    *,
    points: torch.Tensor,
    directions: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation = viewmats[:, :3, :3]
    translation = viewmats[:, :3, 3]
    camera_points = torch.einsum("vij,nj->vni", rotation, points) + translation[:, None]
    camera_direction = torch.einsum("vij,nj->vni", rotation, directions)
    depth = camera_points[..., 2]
    denominator = depth.square().clamp_min(EPS)
    fx = intrinsics[:, 0, 0][:, None]
    fy = intrinsics[:, 1, 1][:, None]
    screen = torch.stack(
        (
            fx
            * (
                camera_direction[..., 0] * depth
                - camera_points[..., 0] * camera_direction[..., 2]
            )
            / denominator,
            fy
            * (
                camera_direction[..., 1] * depth
                - camera_points[..., 1] * camera_direction[..., 2]
            )
            / denominator,
        ),
        dim=-1,
    )
    magnitude = torch.linalg.vector_norm(screen, dim=-1)
    return screen, magnitude, depth


def multiview_axial_energy(
    *,
    direction: torch.Tensor,
    projection_points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    candidate_screen, candidate_magnitude, candidate_depth = (
        project_direction_differentials(
            points=projection_points,
            directions=direction,
            viewmats=viewmats,
            intrinsics=intrinsics,
        )
    )
    evidence_screen, evidence_magnitude, evidence_depth = _project_per_view_evidence(
        points=projection_points,
        per_view_axes=per_view_axes,
        viewmats=viewmats,
        intrinsics=intrinsics,
    )
    valid = (
        (per_view_weights > 0.0)
        & (candidate_magnitude > 1.0e-6)
        & (evidence_magnitude > 1.0e-6)
        & (candidate_depth > 1.0e-6)
        & (evidence_depth > 1.0e-6)
    )
    weight = torch.where(valid, per_view_weights, torch.zeros_like(per_view_weights))
    candidate_unit = candidate_screen / candidate_magnitude[..., None].clamp_min(EPS)
    evidence_unit = evidence_screen / evidence_magnitude[..., None].clamp_min(EPS)
    agreement = (candidate_unit * evidence_unit).sum(dim=-1).clamp(-1.0, 1.0)
    residual = 1.0 - agreement.square()
    return (weight * residual).sum() / weight.sum().clamp_min(EPS)


def _project_per_view_evidence(
    *,
    points: torch.Tensor,
    per_view_axes: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotation = viewmats[:, :3, :3]
    translation = viewmats[:, :3, 3]
    camera_points = torch.einsum("vij,nj->vni", rotation, points) + translation[:, None]
    camera_direction = torch.einsum("vij,vnj->vni", rotation, per_view_axes)
    depth = camera_points[..., 2]
    denominator = depth.square().clamp_min(EPS)
    fx = intrinsics[:, 0, 0][:, None]
    fy = intrinsics[:, 1, 1][:, None]
    screen = torch.stack(
        (
            fx
            * (
                camera_direction[..., 0] * depth
                - camera_points[..., 0] * camera_direction[..., 2]
            )
            / denominator,
            fy
            * (
                camera_direction[..., 1] * depth
                - camera_points[..., 1] * camera_direction[..., 2]
            )
            / denominator,
        ),
        dim=-1,
    )
    magnitude = torch.linalg.vector_norm(screen, dim=-1)
    return screen, magnitude, depth


def refit_tangent_angles(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    projection_points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    config: PostV8RefinementConfig,
) -> TangentAngleRefitResult:
    n, _ = _validate_core_shapes(
        direction=direction,
        normals=normals,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
    )
    current = _unit(direction, name="direction")
    normal = _unit(normals, name="normals")
    frame = _tangent_frame(current, normal)
    coordinate = torch.nn.Parameter(current.new_zeros((n,)))
    optimizer = torch.optim.Adam([coordinate], lr=float(config.ba_learning_rate))
    initial_energy = multiview_axial_energy(
        direction=current,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
    )
    best_energy = float(initial_energy.detach().cpu())
    best_coordinate = coordinate.detach().clone()
    no_improvement = 0
    history: list[dict[str, float | int]] = []
    for iteration in range(int(config.ba_iterations)):
        candidate = compose_tangent_angle(
            normals=normal,
            frame=frame,
            tangent_coordinate=coordinate,
        )
        loss = multiview_axial_energy(
            direction=candidate,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=per_view_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite BA loss at iteration {iteration}")
        value = float(loss.detach().cpu())
        relative_gain = (best_energy - value) / max(abs(best_energy), EPS)
        if value < best_energy and relative_gain > float(config.ba_relative_tolerance):
            best_energy = value
            best_coordinate = coordinate.detach().clone()
            no_improvement = 0
        else:
            no_improvement += 1
        if iteration == 0 or (iteration + 1) % 10 == 0:
            history.append({"iteration": iteration + 1, "data_energy": value})
        if no_improvement >= int(config.ba_patience):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if coordinate.grad is None or not bool(torch.isfinite(coordinate.grad).all()):
            raise RuntimeError(f"invalid BA gradient at iteration {iteration}")
        optimizer.step()
    final_direction = compose_tangent_angle(
        normals=normal,
        frame=frame,
        tangent_coordinate=best_coordinate,
    )
    final_energy = multiview_axial_energy(
        direction=final_direction,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
    )
    return TangentAngleRefitResult(
        direction=final_direction.detach(),
        tangent_coordinate=best_coordinate.detach(),
        initial_data_energy=float(initial_energy.detach().cpu()),
        final_data_energy=float(final_energy.detach().cpu()),
        history=tuple(history),
    )


def surface_connection_energy(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
) -> torch.Tensor:
    transported = parallel_transport_vectors(
        direction[edge_v],
        normals[edge_v],
        normals[edge_u],
    )
    edge_dot = (direction[edge_u] * transported).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - edge_dot).mean()


def _change_statistics(before: torch.Tensor, after: torch.Tensor) -> dict[str, float]:
    angle = torch.rad2deg(
        torch.acos((before * after).sum(dim=-1).clamp(-1.0, 1.0))
    )
    return {
        "mean_deg": float(angle.mean().cpu()),
        "p50_deg": float(torch.quantile(angle, 0.50).cpu()),
        "p95_deg": float(torch.quantile(angle, 0.95).cpu()),
        "p99_deg": float(torch.quantile(angle, 0.99).cpu()),
        "max_deg": float(angle.max().cpu()),
    }


def run_post_v8_refinement(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    projection_points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    observed: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    field_confidence: torch.Tensor,
    unary_normalized_margin: torch.Tensor,
    unary_vote_coherence: torch.Tensor,
    canonical_rank: torch.Tensor,
    config: PostV8RefinementConfig | None = None,
) -> PostV8RefinementResult:
    cfg = PostV8RefinementConfig() if config is None else config
    current = _unit(direction, name="direction").detach()
    normal = _unit(normals, name="normals")
    current_data = multiview_axial_energy(
        direction=current,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
    )
    current_surface = surface_connection_energy(
        direction=current,
        normals=normal,
        edge_u=edge_u,
        edge_v=edge_v,
    )
    cycles: list[dict[str, Any]] = []
    outputs: list[torch.Tensor] = []
    stop_reason = "maximum_cycles"
    for cycle in range(1, int(cfg.outer_max_cycles) + 1):
        ba = refit_tangent_angles(
            direction=current,
            normals=normal,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=per_view_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
            config=cfg,
        )
        frame = _tangent_frame(current, normal)
        accepted: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        for step in range(int(cfg.backtracking_steps)):
            scale = 0.5**step
            ba_candidate = compose_tangent_angle(
                normals=normal,
                frame=frame,
                tangent_coordinate=ba.tangent_coordinate * scale,
            ).detach()
            propagated = refine_confidence_guided_directed_flow(
                direction=ba_candidate,
                normals=normal,
                observed=observed,
                edge_u=edge_u,
                edge_v=edge_v,
                field_confidence=field_confidence,
                unary_normalized_margin=unary_normalized_margin,
                unary_vote_coherence=unary_vote_coherence,
                canonical_rank=canonical_rank,
            )
            candidate = _unit(
                propagated["direction"],
                name="propagated direction",
            ).to(device=current.device, dtype=current.dtype)
            data_energy = multiview_axial_energy(
                direction=candidate,
                projection_points=projection_points,
                per_view_axes=per_view_axes,
                per_view_weights=per_view_weights,
                viewmats=viewmats,
                intrinsics=intrinsics,
            )
            surface_energy = surface_connection_energy(
                direction=candidate,
                normals=normal,
                edge_u=edge_u,
                edge_v=edge_v,
            )
            data_ok = bool(
                data_energy <= current_data + float(cfg.acceptance_tolerance)
            )
            surface_ok = bool(
                surface_energy <= current_surface + float(cfg.acceptance_tolerance)
            )
            strict = bool(
                data_energy < current_data - float(cfg.acceptance_tolerance)
                or surface_energy < current_surface - float(cfg.acceptance_tolerance)
            )
            attempt = {
                "scale": scale,
                "data_energy": float(data_energy.cpu()),
                "surface_energy": float(surface_energy.cpu()),
                "data_nonincreasing": data_ok,
                "surface_nonincreasing": surface_ok,
                "strict_improvement": strict,
            }
            attempts.append(attempt)
            if data_ok and surface_ok and strict:
                accepted = {
                    "direction": candidate.detach(),
                    "data_energy": data_energy.detach(),
                    "surface_energy": surface_energy.detach(),
                    "scale": scale,
                    "propagation_report": propagated["report"],
                }
                break
        if accepted is None:
            stop_reason = "no_pareto_acceptable_cycle"
            cycles.append(
                {
                    "cycle": cycle,
                    "accepted": False,
                    "ba_initial_data_energy": ba.initial_data_energy,
                    "ba_final_data_energy": ba.final_data_energy,
                    "attempts": attempts,
                }
            )
            break
        candidate = accepted["direction"]
        change = _change_statistics(current, candidate)
        previous_data = float(current_data.cpu())
        previous_surface = float(current_surface.cpu())
        current_data = accepted["data_energy"]
        current_surface = accepted["surface_energy"]
        current = candidate
        outputs.append(current.detach().clone())
        relative_data = (previous_data - float(current_data.cpu())) / max(
            abs(previous_data), EPS
        )
        relative_surface = (previous_surface - float(current_surface.cpu())) / max(
            abs(previous_surface), EPS
        )
        cycles.append(
            {
                "cycle": cycle,
                "accepted": True,
                "accepted_scale": accepted["scale"],
                "ba_initial_data_energy": ba.initial_data_energy,
                "ba_final_data_energy": ba.final_data_energy,
                "data_energy": float(current_data.cpu()),
                "surface_energy": float(current_surface.cpu()),
                "relative_data_improvement": relative_data,
                "relative_surface_improvement": relative_surface,
                "change": change,
                "attempts": attempts,
                "propagation": accepted["propagation_report"],
            }
        )
        if (
            change["p95_deg"] <= float(cfg.outer_change_p95_tolerance_deg)
            and relative_data <= float(cfg.outer_relative_tolerance)
            and relative_surface <= float(cfg.outer_relative_tolerance)
        ):
            stop_reason = "fixed_point_tolerance"
            break
    report = {
        "schema": "anigroom.post_v8_refinement.v1",
        "status": "complete",
        "confidence_contract": {
            "ba": "axis_view_cluster_selected_direct_weight [V,N]",
            "propagation": (
                "axis_view_cluster_final_confidence * "
                "axis_view_cluster_global_unary_normalized_margin * "
                "axis_view_cluster_global_unary_vote_coherence [N]"
            ),
            "confidence_recomputed": False,
        },
        "cycles": cycles,
        "accepted_cycle_count": len(outputs),
        "stop_reason": stop_reason,
        "final_data_energy": float(current_data.cpu()),
        "final_surface_energy": float(current_surface.cpu()),
        "config": {
            name: getattr(cfg, name)
            for name in PostV8RefinementConfig.__dataclass_fields__
        },
    }
    return PostV8RefinementResult(
        direction=current.detach(),
        cycle_directions=tuple(outputs),
        report=report,
    )

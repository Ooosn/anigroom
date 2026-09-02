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
    refit_mode: str = "theta_only"
    rho_epsilon: float = 1.0e-6
    lift_delta_smooth_weight: float = 0.0
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
        if self.refit_mode not in {
            "theta_only",
            "theta_observable",
            "rho_only",
            "joint",
            "joint_observable",
        }:
            raise ValueError(
                "refit_mode must be theta_only, theta_observable, rho_only, joint, "
                "or joint_observable"
            )
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
            "rho_epsilon": self.rho_epsilon,
            "ba_learning_rate": self.ba_learning_rate,
            "ba_relative_tolerance": self.ba_relative_tolerance,
            "outer_relative_tolerance": self.outer_relative_tolerance,
            "outer_change_p95_tolerance_deg": self.outer_change_p95_tolerance_deg,
            "acceptance_tolerance": self.acceptance_tolerance,
        }
        for name, value in positive_values.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.lift_delta_smooth_weight)) or float(
            self.lift_delta_smooth_weight
        ) < 0.0:
            raise ValueError("lift_delta_smooth_weight must be finite and non-negative")


@dataclass(frozen=True)
class DirectionRefitResult:
    direction: torch.Tensor
    tangent_coordinate: torch.Tensor
    lift_coordinate: torch.Tensor
    initial_data_energy: float
    final_data_energy: float
    initial_optimization_energy: float
    final_optimization_energy: float
    observability_report: dict[str, float | int]
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


def compose_direction_parameters(
    *,
    normals: torch.Tensor,
    frame: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tangent_coordinate: torch.Tensor,
    lift_coordinate: torch.Tensor,
    rho_epsilon: float,
) -> torch.Tensor:
    """Decode separate tangent-angle and log-lift coordinates."""

    normal = _unit(normals, name="normals")
    normal_component, tangent_magnitude, tangent_axis, bitangent = frame
    n = int(normal.shape[0])
    if tangent_coordinate.shape != (n,) or lift_coordinate.shape != (n,):
        raise ValueError("direction coordinates must both have shape [N]")
    rotated_axis = _unit(
        tangent_axis + tangent_coordinate[:, None] * bitangent,
        name="rotated tangent axis",
    )
    baseline_rho = (
        normal_component.clamp_min(0.0) / tangent_magnitude.clamp_min(EPS)
    )[:, 0]
    rho = (
        (baseline_rho + float(rho_epsilon))
        * torch.exp(lift_coordinate.clamp(-20.0, 20.0))
        - float(rho_epsilon)
    ).clamp_min(0.0)
    return _unit(
        rotated_axis + rho[:, None] * normal,
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


@torch.no_grad()
def parameter_observability(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    projection_points: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    rho_epsilon: float,
    finite_difference_step: float = 1.0e-3,
) -> dict[str, torch.Tensor]:
    """Measure screen-angle sensitivity to tangent angle and log lift."""

    current = _unit(direction, name="direction")
    normal = _unit(normals, name="normals")
    frame = _tangent_frame(current, normal)
    n = int(current.shape[0])
    step = float(finite_difference_step)
    zero = current.new_zeros((n,))

    def projected(
        tangent_coordinate: torch.Tensor,
        lift_coordinate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate = compose_direction_parameters(
            normals=normal,
            frame=frame,
            tangent_coordinate=tangent_coordinate,
            lift_coordinate=lift_coordinate,
            rho_epsilon=float(rho_epsilon),
        )
        screen, magnitude, depth = project_direction_differentials(
            points=projection_points,
            directions=candidate,
            viewmats=viewmats,
            intrinsics=intrinsics,
        )
        return screen / magnitude[..., None].clamp_min(EPS), magnitude, depth

    theta_plus, theta_plus_mag, theta_plus_depth = projected(
        zero + math.tan(step),
        zero,
    )
    theta_minus, theta_minus_mag, theta_minus_depth = projected(
        zero - math.tan(step),
        zero,
    )
    lift_plus, lift_plus_mag, lift_plus_depth = projected(zero, zero + step)
    lift_minus, lift_minus_mag, lift_minus_depth = projected(zero, zero - step)

    def signed_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
        dot = (first * second).sum(dim=-1)
        return torch.atan2(cross, dot)

    theta_jacobian = signed_angle(theta_minus, theta_plus) / (2.0 * step)
    lift_jacobian = signed_angle(lift_minus, lift_plus) / (2.0 * step)
    valid = (
        (theta_plus_mag > 1.0e-6)
        & (theta_minus_mag > 1.0e-6)
        & (lift_plus_mag > 1.0e-6)
        & (lift_minus_mag > 1.0e-6)
        & (theta_plus_depth > 1.0e-6)
        & (theta_minus_depth > 1.0e-6)
        & (lift_plus_depth > 1.0e-6)
        & (lift_minus_depth > 1.0e-6)
    )
    theta_information = theta_jacobian.square()
    lift_information = lift_jacobian.square()
    denominator = theta_information + lift_information
    theta_fraction = torch.where(
        valid & (denominator > EPS),
        theta_information / denominator.clamp_min(EPS),
        torch.zeros_like(denominator),
    )
    lift_fraction = torch.where(
        valid & (denominator > EPS),
        lift_information / denominator.clamp_min(EPS),
        torch.zeros_like(denominator),
    )
    return {
        "theta_jacobian": theta_jacobian,
        "lift_jacobian": lift_jacobian,
        "theta_fraction": theta_fraction,
        "lift_fraction": lift_fraction,
        "valid": valid,
    }


def refit_direction_parameters(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    projection_points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    config: PostV8RefinementConfig,
    edge_u: torch.Tensor | None = None,
    edge_v: torch.Tensor | None = None,
) -> DirectionRefitResult:
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
    if float(config.lift_delta_smooth_weight) > 0.0:
        if edge_u is None or edge_v is None:
            raise ValueError("lift delta smoothing requires edge_u and edge_v")
        if edge_u.shape != edge_v.shape or edge_u.ndim != 1:
            raise ValueError("edge_u and edge_v must have matching shape [E]")
        if edge_u.device != current.device or edge_v.device != current.device:
            raise ValueError("lift smoothing edges must share the refit device")
    optimization_weights = per_view_weights
    theta_optimization_weights = per_view_weights
    lift_optimization_weights = per_view_weights
    observability_report: dict[str, float | int] = {
        "enabled": 0,
        "positive_input_pair_count": int((per_view_weights > 0.0).sum().cpu()),
    }
    if config.refit_mode in {"theta_observable", "joint_observable"}:
        observability = parameter_observability(
            direction=current,
            normals=normal,
            projection_points=projection_points,
            viewmats=viewmats,
            intrinsics=intrinsics,
            rho_epsilon=float(config.rho_epsilon),
        )
        theta_fraction = observability["theta_fraction"].to(
            device=current.device,
            dtype=current.dtype,
        )
        lift_fraction = observability["lift_fraction"].to(
            device=current.device,
            dtype=current.dtype,
        )
        theta_optimization_weights = per_view_weights * theta_fraction
        lift_optimization_weights = per_view_weights * lift_fraction
        if config.refit_mode == "theta_observable":
            optimization_weights = theta_optimization_weights
        positive = per_view_weights > 0.0
        selected_fraction = theta_fraction[positive]
        observability_report = {
            "enabled": 1,
            "positive_input_pair_count": int(positive.sum().cpu()),
            "positive_effective_pair_count": int(
                (theta_optimization_weights > 0.0).sum().cpu()
            ),
            "theta_fraction_mean": float(selected_fraction.mean().cpu()),
            "theta_fraction_p50": float(
                torch.quantile(selected_fraction, 0.50).cpu()
            ),
            "theta_fraction_p95": float(
                torch.quantile(selected_fraction, 0.95).cpu()
            ),
            "theta_fraction_min": float(selected_fraction.min().cpu()),
            "theta_fraction_max": float(selected_fraction.max().cpu()),
            "lift_fraction_mean": float(lift_fraction[positive].mean().cpu()),
            "gradient_partitioned": int(config.refit_mode == "joint_observable"),
        }
    tangent_coordinate = torch.nn.Parameter(current.new_zeros((n,)))
    lift_coordinate = torch.nn.Parameter(current.new_zeros((n,)))
    trainable = []
    if config.refit_mode in {
        "theta_only",
        "theta_observable",
        "joint",
        "joint_observable",
    }:
        trainable.append(tangent_coordinate)
    if config.refit_mode in {"rho_only", "joint", "joint_observable"}:
        trainable.append(lift_coordinate)
    optimizer = torch.optim.Adam(trainable, lr=float(config.ba_learning_rate))
    initial_energy = multiview_axial_energy(
        direction=current,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
    )
    if config.refit_mode == "joint_observable":
        initial_optimization_energy = multiview_axial_energy(
            direction=current,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=theta_optimization_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
        ) + multiview_axial_energy(
            direction=current,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=lift_optimization_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
        )
    else:
        initial_optimization_energy = multiview_axial_energy(
            direction=current,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=optimization_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
        )
    best_energy = float(initial_optimization_energy.detach().cpu())
    best_tangent_coordinate = tangent_coordinate.detach().clone()
    best_lift_coordinate = lift_coordinate.detach().clone()
    no_improvement = 0
    history: list[dict[str, float | int]] = []
    for iteration in range(int(config.ba_iterations)):
        if config.refit_mode == "joint_observable":
            theta_candidate = compose_direction_parameters(
                normals=normal,
                frame=frame,
                tangent_coordinate=tangent_coordinate,
                lift_coordinate=lift_coordinate.detach(),
                rho_epsilon=float(config.rho_epsilon),
            )
            lift_candidate = compose_direction_parameters(
                normals=normal,
                frame=frame,
                tangent_coordinate=tangent_coordinate.detach(),
                lift_coordinate=lift_coordinate,
                rho_epsilon=float(config.rho_epsilon),
            )
            loss = multiview_axial_energy(
                direction=theta_candidate,
                projection_points=projection_points,
                per_view_axes=per_view_axes,
                per_view_weights=theta_optimization_weights,
                viewmats=viewmats,
                intrinsics=intrinsics,
            ) + multiview_axial_energy(
                direction=lift_candidate,
                projection_points=projection_points,
                per_view_axes=per_view_axes,
                per_view_weights=lift_optimization_weights,
                viewmats=viewmats,
                intrinsics=intrinsics,
            )
        else:
            candidate = compose_direction_parameters(
                normals=normal,
                frame=frame,
                tangent_coordinate=tangent_coordinate,
                lift_coordinate=lift_coordinate,
                rho_epsilon=float(config.rho_epsilon),
            )
            loss = multiview_axial_energy(
                direction=candidate,
                projection_points=projection_points,
                per_view_axes=per_view_axes,
                per_view_weights=optimization_weights,
                viewmats=viewmats,
                intrinsics=intrinsics,
            )
        if float(config.lift_delta_smooth_weight) > 0.0:
            lift_delta_smooth = (
                lift_coordinate[edge_u] - lift_coordinate[edge_v]
            ).square().mean()
            loss = loss + float(config.lift_delta_smooth_weight) * lift_delta_smooth
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite BA loss at iteration {iteration}")
        value = float(loss.detach().cpu())
        relative_gain = (best_energy - value) / max(abs(best_energy), EPS)
        if value < best_energy and relative_gain > float(config.ba_relative_tolerance):
            best_energy = value
            best_tangent_coordinate = tangent_coordinate.detach().clone()
            best_lift_coordinate = lift_coordinate.detach().clone()
            no_improvement = 0
        else:
            no_improvement += 1
        if iteration == 0 or (iteration + 1) % 10 == 0:
            history.append({"iteration": iteration + 1, "data_energy": value})
        if no_improvement >= int(config.ba_patience):
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in trainable:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"invalid BA gradient at iteration {iteration}")
        optimizer.step()
    final_direction = compose_direction_parameters(
        normals=normal,
        frame=frame,
        tangent_coordinate=best_tangent_coordinate,
        lift_coordinate=best_lift_coordinate,
        rho_epsilon=float(config.rho_epsilon),
    )
    final_energy = multiview_axial_energy(
        direction=final_direction,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
    )
    if config.refit_mode == "joint_observable":
        final_optimization_energy = multiview_axial_energy(
            direction=final_direction,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=theta_optimization_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
        ) + multiview_axial_energy(
            direction=final_direction,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=lift_optimization_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
        )
    else:
        final_optimization_energy = multiview_axial_energy(
            direction=final_direction,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=optimization_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
        )
    if float(config.lift_delta_smooth_weight) > 0.0:
        final_optimization_energy = final_optimization_energy + float(
            config.lift_delta_smooth_weight
        ) * (
            best_lift_coordinate[edge_u] - best_lift_coordinate[edge_v]
        ).square().mean()
    return DirectionRefitResult(
        direction=final_direction.detach(),
        tangent_coordinate=best_tangent_coordinate.detach(),
        lift_coordinate=best_lift_coordinate.detach(),
        initial_data_energy=float(initial_energy.detach().cpu()),
        final_data_energy=float(final_energy.detach().cpu()),
        initial_optimization_energy=float(initial_optimization_energy.detach().cpu()),
        final_optimization_energy=float(final_optimization_energy.detach().cpu()),
        observability_report=observability_report,
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


def tangent_connection_energy(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
) -> torch.Tensor:
    normal = _unit(normals, name="normals")
    unit_direction = _unit(direction, name="direction")
    normal_component = (unit_direction * normal).sum(dim=-1, keepdim=True)
    tangent = unit_direction - normal_component * normal
    tangent_magnitude = torch.linalg.vector_norm(tangent, dim=-1)
    tangent_axis = F.normalize(tangent, dim=-1, eps=EPS)
    transported = parallel_transport_vectors(
        tangent_axis[edge_v],
        normal[edge_v],
        normal[edge_u],
    )
    edge_dot = (tangent_axis[edge_u] * transported).sum(dim=-1).clamp(-1.0, 1.0)
    weight = tangent_magnitude[edge_u] * tangent_magnitude[edge_v]
    return (weight * (1.0 - edge_dot)).sum() / weight.sum().clamp_min(EPS)


def lift_connection_energy(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
) -> torch.Tensor:
    normal = _unit(normals, name="normals")
    unit_direction = _unit(direction, name="direction")
    normal_component = (unit_direction * normal).sum(dim=-1).clamp_min(0.0)
    tangent = unit_direction - normal_component[:, None] * normal
    tangent_magnitude = torch.linalg.vector_norm(tangent, dim=-1)
    rho = normal_component / tangent_magnitude.clamp_min(EPS)
    log_lift = torch.log1p(rho)
    return (log_lift[edge_u] - log_lift[edge_v]).square().mean()


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
    current_tangent = tangent_connection_energy(
        direction=current,
        normals=normal,
        edge_u=edge_u,
        edge_v=edge_v,
    )
    current_lift = lift_connection_energy(
        direction=current,
        normals=normal,
        edge_u=edge_u,
        edge_v=edge_v,
    )
    initial_energy = {
        "data": float(current_data.cpu()),
        "surface": float(current_surface.cpu()),
        "tangent": float(current_tangent.cpu()),
        "lift": float(current_lift.cpu()),
    }
    cycles: list[dict[str, Any]] = []
    outputs: list[torch.Tensor] = []
    stop_reason = "maximum_cycles"
    for cycle in range(1, int(cfg.outer_max_cycles) + 1):
        ba = refit_direction_parameters(
            direction=current,
            normals=normal,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=per_view_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
            config=cfg,
            edge_u=edge_u,
            edge_v=edge_v,
        )
        frame = _tangent_frame(current, normal)
        accepted: dict[str, Any] | None = None
        attempts: list[dict[str, Any]] = []
        for step in range(int(cfg.backtracking_steps)):
            scale = 0.5**step
            ba_candidate = compose_direction_parameters(
                normals=normal,
                frame=frame,
                tangent_coordinate=ba.tangent_coordinate * scale,
                lift_coordinate=ba.lift_coordinate * scale,
                rho_epsilon=float(cfg.rho_epsilon),
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
            tangent_energy = tangent_connection_energy(
                direction=candidate,
                normals=normal,
                edge_u=edge_u,
                edge_v=edge_v,
            )
            lift_energy = lift_connection_energy(
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
            tangent_ok = bool(
                tangent_energy <= current_tangent + float(cfg.acceptance_tolerance)
            )
            lift_ok = bool(
                lift_energy <= current_lift + float(cfg.acceptance_tolerance)
            )
            strict = bool(
                data_energy < current_data - float(cfg.acceptance_tolerance)
                or surface_energy < current_surface - float(cfg.acceptance_tolerance)
                or tangent_energy < current_tangent - float(cfg.acceptance_tolerance)
                or lift_energy < current_lift - float(cfg.acceptance_tolerance)
            )
            attempt = {
                "scale": scale,
                "data_energy": float(data_energy.cpu()),
                "surface_energy": float(surface_energy.cpu()),
                "tangent_energy": float(tangent_energy.cpu()),
                "lift_energy": float(lift_energy.cpu()),
                "data_nonincreasing": data_ok,
                "surface_nonincreasing": surface_ok,
                "tangent_nonincreasing": tangent_ok,
                "lift_nonincreasing": lift_ok,
                "strict_improvement": strict,
            }
            attempts.append(attempt)
            if data_ok and surface_ok and tangent_ok and lift_ok and strict:
                accepted = {
                    "direction": candidate.detach(),
                    "data_energy": data_energy.detach(),
                    "surface_energy": surface_energy.detach(),
                    "tangent_energy": tangent_energy.detach(),
                    "lift_energy": lift_energy.detach(),
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
                    "ba_initial_optimization_energy": ba.initial_optimization_energy,
                    "ba_final_optimization_energy": ba.final_optimization_energy,
                    "ba_observability": ba.observability_report,
                    "attempts": attempts,
                }
            )
            break
        candidate = accepted["direction"]
        change = _change_statistics(current, candidate)
        previous_data = float(current_data.cpu())
        previous_surface = float(current_surface.cpu())
        previous_tangent = float(current_tangent.cpu())
        previous_lift = float(current_lift.cpu())
        current_data = accepted["data_energy"]
        current_surface = accepted["surface_energy"]
        current_tangent = accepted["tangent_energy"]
        current_lift = accepted["lift_energy"]
        current = candidate
        outputs.append(current.detach().clone())
        relative_data = (previous_data - float(current_data.cpu())) / max(
            abs(previous_data), EPS
        )
        relative_surface = (previous_surface - float(current_surface.cpu())) / max(
            abs(previous_surface), EPS
        )
        relative_tangent = (previous_tangent - float(current_tangent.cpu())) / max(
            abs(previous_tangent), EPS
        )
        relative_lift = (previous_lift - float(current_lift.cpu())) / max(
            abs(previous_lift), EPS
        )
        cycles.append(
            {
                "cycle": cycle,
                "accepted": True,
                "accepted_scale": accepted["scale"],
                "ba_initial_data_energy": ba.initial_data_energy,
                "ba_final_data_energy": ba.final_data_energy,
                "ba_initial_optimization_energy": ba.initial_optimization_energy,
                "ba_final_optimization_energy": ba.final_optimization_energy,
                "ba_observability": ba.observability_report,
                "data_energy": float(current_data.cpu()),
                "surface_energy": float(current_surface.cpu()),
                "tangent_energy": float(current_tangent.cpu()),
                "lift_energy": float(current_lift.cpu()),
                "relative_data_improvement": relative_data,
                "relative_surface_improvement": relative_surface,
                "relative_tangent_improvement": relative_tangent,
                "relative_lift_improvement": relative_lift,
                "change": change,
                "attempts": attempts,
                "propagation": accepted["propagation_report"],
            }
        )
        if (
            change["p95_deg"] <= float(cfg.outer_change_p95_tolerance_deg)
            and relative_data <= float(cfg.outer_relative_tolerance)
            and relative_surface <= float(cfg.outer_relative_tolerance)
            and relative_tangent <= float(cfg.outer_relative_tolerance)
            and relative_lift <= float(cfg.outer_relative_tolerance)
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
        "initial_energy": initial_energy,
        "final_data_energy": float(current_data.cpu()),
        "final_surface_energy": float(current_surface.cpu()),
        "final_tangent_energy": float(current_tangent.cpu()),
        "final_lift_energy": float(current_lift.cpu()),
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

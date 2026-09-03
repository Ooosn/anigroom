"""Global, geometry-only refinement of a complete post-V8 direction field.

The input is already a complete directed field.  This module optimizes one
tangent coordinate and one log-lift coordinate per root against the frozen
multiview direct-vector evidence.  Spatial energies are used only as a
dimensionless objective regularizer; the final candidate is still checked
against every unnormalized field metric before it can replace the baseline.

There are deliberately no confidence owners, protected roots, region rules,
view rules, or image-coordinate rules in this pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

from .direction_geometry import parallel_transport_vectors
from .global_sign_orientation import SEVERE_DOT_THRESHOLD
from .post_v8_refinement import (
    _tangent_frame,
    _validate_core_shapes,
    compose_direction_parameters,
    lift_connection_energy,
    multiview_axial_energy,
    surface_connection_energy,
    tangent_connection_energy,
)


EPS = 1.0e-8
RHO_EPSILON = 1.0e-6


@dataclass(frozen=True)
class GlobalDirectionFieldRefinementConfig:
    """Settings for the bounded global direction-field refiner.

    ``relative_tolerance`` controls the optimizer's best-point improvement
    test.  ``acceptance_tolerance`` is an absolute tolerance applied to all
    final metric gates, including the edge-angle percentiles in degrees.
    """

    smooth_weight: float = 1.0
    orientation_barrier_weight: float = 0.0
    iterations: int = 120
    learning_rate: float = 0.03
    patience: int = 20
    relative_tolerance: float = 1.0e-7
    backtracking_steps: int = 6
    acceptance_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        integer_values = {
            "iterations": self.iterations,
            "patience": self.patience,
            "backtracking_steps": self.backtracking_steps,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")

        nonnegative_values = {
            "smooth_weight": self.smooth_weight,
            "orientation_barrier_weight": self.orientation_barrier_weight,
            "relative_tolerance": self.relative_tolerance,
            "acceptance_tolerance": self.acceptance_tolerance,
        }
        for name, value in nonnegative_values.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        if not math.isfinite(float(self.learning_rate)) or float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be finite and positive")


@dataclass(frozen=True)
class GlobalDirectionFieldRefinementResult:
    """Output field, applied coordinates, and an auditable refinement report."""

    direction: torch.Tensor
    tangent_coordinate: torch.Tensor
    lift_coordinate: torch.Tensor
    report: dict[str, Any]

    @property
    def final_direction(self) -> torch.Tensor:
        """Alias useful to callers that name the output explicitly."""

        return self.direction


def _unit(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")
    if value.is_complex() or not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a real floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    magnitude = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if bool((magnitude <= EPS).any()):
        raise ValueError(f"{name} contains a zero vector")
    return value / magnitude


def _validate_graph(
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    root_count: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(edge_u, torch.Tensor) or not isinstance(edge_v, torch.Tensor):
        raise TypeError("edge_u and edge_v must be torch.Tensor values")
    if edge_u.ndim != 1 or edge_v.shape != edge_u.shape:
        raise ValueError("edge_u and edge_v must have matching shape [E]")
    if edge_u.is_floating_point() or edge_u.is_complex() or edge_u.dtype == torch.bool:
        raise TypeError("edge_u must be an integer tensor")
    if edge_v.is_floating_point() or edge_v.is_complex() or edge_v.dtype == torch.bool:
        raise TypeError("edge_v must be an integer tensor")
    if edge_u.device != device or edge_v.device != device:
        raise ValueError("edge_u and edge_v must share the direction device")
    if edge_u.numel() <= 0:
        raise ValueError("the direction graph must contain at least one edge")
    if bool(
        ((edge_u < 0) | (edge_u >= root_count) | (edge_v < 0) | (edge_v >= root_count)).any()
    ):
        raise ValueError("edge endpoints contain an out-of-range root index")
    return edge_u.to(dtype=torch.long), edge_v.to(dtype=torch.long)


def _validate_inputs(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    projection_points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    tensors = {
        "direction": direction,
        "normals": normals,
        "projection_points": projection_points,
        "per_view_axes": per_view_axes,
        "per_view_weights": per_view_weights,
        "viewmats": viewmats,
        "intrinsics": intrinsics,
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all direction-field inputs must be torch.Tensor values")
    root_count, _ = _validate_core_shapes(
        direction=direction,
        normals=normals,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
    )
    if root_count <= 0:
        raise ValueError("direction must contain at least one root")
    if int(per_view_axes.shape[0]) <= 0:
        raise ValueError("per_view_axes must contain at least one view")
    edge_u, edge_v = _validate_graph(
        edge_u=edge_u,
        edge_v=edge_v,
        root_count=root_count,
        device=direction.device,
    )
    direction_magnitude = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    # The public contract supplies unit directions.  Keep that tensor exactly
    # for a rejected solve; normalize only malformed-but-nonzero input so the
    # geometry helpers still receive a valid field.
    if bool(
        torch.allclose(
            direction_magnitude,
            torch.ones_like(direction_magnitude),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
    ):
        baseline = direction.detach().clone()
    else:
        baseline = _unit(direction, name="direction").detach()
    normal = _unit(normals, name="normals").detach()
    return baseline, normal, edge_u, edge_v, per_view_weights, root_count


def _edge_angle_metrics(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    transported = parallel_transport_vectors(
        direction[edge_v],
        normals[edge_v],
        normals[edge_u],
    )
    edge_dot = (direction[edge_u] * transported).sum(dim=-1).clamp(-1.0, 1.0)
    return edge_dot, torch.rad2deg(torch.acos(edge_dot))


def _orientation_barrier_components(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    baseline_nonnegative_edge: torch.Tensor,
    baseline_nonsevere_edge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize threshold crossings only on edges clean at the baseline."""

    candidate_dot, _ = _edge_angle_metrics(
        direction=direction,
        normals=normals,
        edge_u=edge_u,
        edge_v=edge_v,
    )

    def squared_mean_plus_squared_max(
        violation: torch.Tensor,
        eligible: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(eligible.any()):
            return violation.new_zeros(())
        selected_squared = violation[eligible].square()
        return selected_squared.mean() + selected_squared.max()

    negative_violation = torch.relu(-candidate_dot)
    severe_violation = torch.relu(float(SEVERE_DOT_THRESHOLD) - candidate_dot)
    negative_barrier = squared_mean_plus_squared_max(
        negative_violation,
        baseline_nonnegative_edge,
    )
    severe_barrier = squared_mean_plus_squared_max(
        severe_violation,
        baseline_nonsevere_edge,
    )
    return negative_barrier, severe_barrier, negative_barrier + severe_barrier


def _scatter_edge_mask_to_roots(
    *,
    edge_mask: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    root_count: int,
) -> torch.Tensor:
    """Mark every root incident to at least one selected edge."""

    incident_count = torch.zeros(
        (root_count,),
        dtype=torch.int64,
        device=edge_mask.device,
    )
    selected = edge_mask.to(dtype=incident_count.dtype)
    incident_count.scatter_add_(0, edge_u, selected)
    incident_count.scatter_add_(0, edge_v, selected)
    return incident_count > 0


@torch.no_grad()
def _evaluate_metrics(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    projection_points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
) -> tuple[
    dict[str, float | int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    unit_direction = _unit(direction, name="candidate direction")
    edge_dot, edge_angle = _edge_angle_metrics(
        direction=unit_direction,
        normals=normals,
        edge_u=edge_u,
        edge_v=edge_v,
    )
    negative_edge = edge_dot < 0.0
    severe_edge = edge_dot <= float(SEVERE_DOT_THRESHOLD)
    negative_root = _scatter_edge_mask_to_roots(
        edge_mask=negative_edge,
        edge_u=edge_u,
        edge_v=edge_v,
        root_count=int(unit_direction.shape[0]),
    )
    severe_root = _scatter_edge_mask_to_roots(
        edge_mask=severe_edge,
        edge_u=edge_u,
        edge_v=edge_v,
        root_count=int(unit_direction.shape[0]),
    )
    top1_count = max(1, int(math.ceil(0.01 * int(edge_angle.numel()))))
    edge_top1pct_cvar = torch.topk(
        edge_angle,
        k=top1_count,
        largest=True,
        sorted=False,
    ).values.mean()
    metrics: dict[str, float | int] = {
        "data": float(
            multiview_axial_energy(
                direction=unit_direction,
                projection_points=projection_points,
                per_view_axes=per_view_axes,
                per_view_weights=per_view_weights,
                viewmats=viewmats,
                intrinsics=intrinsics,
            ).cpu()
        ),
        "surface": float(
            surface_connection_energy(
                direction=unit_direction,
                normals=normals,
                edge_u=edge_u,
                edge_v=edge_v,
            ).cpu()
        ),
        "tangent": float(
            tangent_connection_energy(
                direction=unit_direction,
                normals=normals,
                edge_u=edge_u,
                edge_v=edge_v,
            ).cpu()
        ),
        "lift": float(
            lift_connection_energy(
                direction=unit_direction,
                normals=normals,
                edge_u=edge_u,
                edge_v=edge_v,
            ).cpu()
        ),
        "edge_p95_deg": float(torch.quantile(edge_angle, 0.95).cpu()),
        "edge_p99_deg": float(torch.quantile(edge_angle, 0.99).cpu()),
        "edge_top1pct_cvar_deg": float(edge_top1pct_cvar.cpu()),
        "edge_max_deg": float(edge_angle.max().cpu()),
        "negative_edge_count": int(negative_edge.sum().cpu()),
        "severe_edge_count": int(severe_edge.sum().cpu()),
        "negative_root_count": int(negative_root.sum().cpu()),
        "severe_root_count": int(severe_root.sum().cpu()),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError("direction-field metrics are non-finite")
    return metrics, negative_edge, severe_edge, negative_root, severe_root


def _objective(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    projection_points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    baseline_data: torch.Tensor,
    baseline_surface: torch.Tensor,
    baseline_tangent: torch.Tensor,
    baseline_lift: torch.Tensor,
    baseline_nonnegative_edge: torch.Tensor,
    baseline_nonsevere_edge: torch.Tensor,
    smooth_weight: float,
    orientation_barrier_weight: float,
) -> torch.Tensor:
    """Evaluate the dimensionless objective on the complete candidate field."""

    data = multiview_axial_energy(
        direction=direction,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
    )
    surface = surface_connection_energy(
        direction=direction,
        normals=normals,
        edge_u=edge_u,
        edge_v=edge_v,
    )
    tangent = tangent_connection_energy(
        direction=direction,
        normals=normals,
        edge_u=edge_u,
        edge_v=edge_v,
    )
    lift = lift_connection_energy(
        direction=direction,
        normals=normals,
        edge_u=edge_u,
        edge_v=edge_v,
    )
    smooth = torch.stack(
        (
            surface / baseline_surface.detach().clamp_min(EPS),
            tangent / baseline_tangent.detach().clamp_min(EPS),
            lift / baseline_lift.detach().clamp_min(EPS),
        )
    ).mean()
    _, _, orientation_barrier = _orientation_barrier_components(
        direction=direction,
        normals=normals,
        edge_u=edge_u,
        edge_v=edge_v,
        baseline_nonnegative_edge=baseline_nonnegative_edge,
        baseline_nonsevere_edge=baseline_nonsevere_edge,
    )
    return (
        data / baseline_data.detach().clamp_min(EPS)
        + float(smooth_weight) * smooth
        + float(orientation_barrier_weight) * orientation_barrier
    )


def _direction_change_statistics(
    before: torch.Tensor,
    after: torch.Tensor,
) -> dict[str, float]:
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


def _history_entry(
    *,
    iteration: int,
    objective: torch.Tensor,
    reference_direction: torch.Tensor,
    direction: torch.Tensor,
    metrics: dict[str, float | int],
    tangent_coordinate: torch.Tensor,
    lift_coordinate: torch.Tensor,
    negative_orientation_barrier: torch.Tensor,
    severe_orientation_barrier: torch.Tensor,
    orientation_barrier: torch.Tensor,
) -> dict[str, float | int]:
    return {
        "iteration": int(iteration),
        "objective": float(objective.detach().cpu()),
        "data": float(metrics["data"]),
        "surface": float(metrics["surface"]),
        "tangent": float(metrics["tangent"]),
        "lift": float(metrics["lift"]),
        "edge_p95_deg": float(metrics["edge_p95_deg"]),
        "edge_p99_deg": float(metrics["edge_p99_deg"]),
        "edge_top1pct_cvar_deg": float(metrics["edge_top1pct_cvar_deg"]),
        "edge_max_deg": float(metrics["edge_max_deg"]),
        "negative_edge_count": int(metrics["negative_edge_count"]),
        "severe_edge_count": int(metrics["severe_edge_count"]),
        "negative_root_count": int(metrics["negative_root_count"]),
        "severe_root_count": int(metrics["severe_root_count"]),
        "negative_orientation_barrier": float(
            negative_orientation_barrier.detach().cpu()
        ),
        "severe_orientation_barrier": float(severe_orientation_barrier.detach().cpu()),
        "orientation_barrier": float(orientation_barrier.detach().cpu()),
        "tangent_coordinate_rms": float(
            torch.sqrt(tangent_coordinate.detach().square().mean()).cpu()
        ),
        "lift_coordinate_rms": float(
            torch.sqrt(lift_coordinate.detach().square().mean()).cpu()
        ),
        "direction_change_p95_deg": float(
            _direction_change_statistics(reference_direction, direction)["p95_deg"]
        ),
    }


def _gate_candidate(
    *,
    baseline_metrics: dict[str, float | int],
    candidate_metrics: dict[str, float | int],
    baseline_negative_edge: torch.Tensor,
    candidate_negative_edge: torch.Tensor,
    baseline_severe_edge: torch.Tensor,
    candidate_severe_edge: torch.Tensor,
    baseline_negative_root: torch.Tensor,
    candidate_negative_root: torch.Tensor,
    baseline_severe_root: torch.Tensor,
    candidate_severe_root: torch.Tensor,
    tolerance: float,
) -> dict[str, Any]:
    continuous_metric_names = (
        "data",
        "surface",
        "tangent",
        "lift",
        "edge_p95_deg",
        "edge_p99_deg",
        "edge_top1pct_cvar_deg",
        "edge_max_deg",
    )
    count_metric_names = (
        "negative_edge_count",
        "severe_edge_count",
        "negative_root_count",
        "severe_root_count",
    )
    nonincreasing = {
        name: float(candidate_metrics[name]) <= float(baseline_metrics[name]) + tolerance
        for name in continuous_metric_names
    }
    strict = {
        name: float(candidate_metrics[name]) < float(baseline_metrics[name]) - tolerance
        for name in continuous_metric_names
    }
    for name in count_metric_names:
        nonincreasing[name] = int(candidate_metrics[name]) <= int(baseline_metrics[name])
        strict[name] = int(candidate_metrics[name]) < int(baseline_metrics[name])

    new_negative_edge = (~baseline_negative_edge) & candidate_negative_edge
    new_severe_edge = (~baseline_severe_edge) & candidate_severe_edge
    new_negative_root = (~baseline_negative_root) & candidate_negative_root
    new_severe_root = (~baseline_severe_root) & candidate_severe_root
    passed = bool(
        all(nonincreasing.values())
        and any(strict.values())
        and not bool(new_negative_root.any())
        and not bool(new_severe_root.any())
    )
    return {
        "passed": passed,
        "nonincreasing": nonincreasing,
        "strict_improvement": strict,
        "any_strict_improvement": bool(any(strict.values())),
        "negative_edge_count_nonincreasing": nonincreasing["negative_edge_count"],
        "severe_edge_count_nonincreasing": nonincreasing["severe_edge_count"],
        "negative_root_count_nonincreasing": nonincreasing["negative_root_count"],
        "severe_root_count_nonincreasing": nonincreasing["severe_root_count"],
        "new_negative_edge_count": int(new_negative_edge.sum().cpu()),
        "new_severe_edge_count": int(new_severe_edge.sum().cpu()),
        "no_new_negative_edges": not bool(new_negative_edge.any()),
        "no_new_severe_edges": not bool(new_severe_edge.any()),
        "new_negative_root_count": int(new_negative_root.sum().cpu()),
        "new_severe_root_count": int(new_severe_root.sum().cpu()),
        "no_new_negative_roots": not bool(new_negative_root.any()),
        "no_new_severe_roots": not bool(new_severe_root.any()),
        "edge_identity_gate_enforced": False,
        "root_support_gate_enforced": True,
    }


def refine_global_direction_field(
    *,
    direction: torch.Tensor,
    normals: torch.Tensor,
    projection_points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    config: GlobalDirectionFieldRefinementConfig | None = None,
) -> GlobalDirectionFieldRefinementResult:
    """Refine a complete direction field and accept only a global Pareto step.

    The optimizer always starts from zero coordinates in the tangent/log-lift
    chart of ``direction``.  It is not itself an acceptance decision: the
    best optimizer point is applied at deterministic powers-of-two scales from
    the baseline, and the first scale satisfying every final-field gate is
    selected.
    """

    cfg = GlobalDirectionFieldRefinementConfig() if config is None else config
    if not isinstance(cfg, GlobalDirectionFieldRefinementConfig):
        raise TypeError("config must be GlobalDirectionFieldRefinementConfig")
    baseline, normal, edge_u, edge_v, per_view_weights, root_count = _validate_inputs(
        direction=direction,
        normals=normals,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
        edge_u=edge_u,
        edge_v=edge_v,
    )

    (
        baseline_metrics,
        baseline_negative_edge,
        baseline_severe_edge,
        baseline_negative_root,
        baseline_severe_root,
    ) = _evaluate_metrics(
        direction=baseline,
        normals=normal,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
        edge_u=edge_u,
        edge_v=edge_v,
    )
    frame = _tangent_frame(baseline, normal)
    tangent_coordinate = torch.nn.Parameter(baseline.new_zeros((root_count,)))
    lift_coordinate = torch.nn.Parameter(baseline.new_zeros((root_count,)))
    optimizer = torch.optim.Adam(
        [tangent_coordinate, lift_coordinate],
        lr=float(cfg.learning_rate),
    )

    baseline_data = baseline.new_tensor(float(baseline_metrics["data"]))
    baseline_surface = baseline.new_tensor(float(baseline_metrics["surface"]))
    baseline_tangent = baseline.new_tensor(float(baseline_metrics["tangent"]))
    baseline_lift = baseline.new_tensor(float(baseline_metrics["lift"]))
    baseline_nonnegative_edge = ~baseline_negative_edge
    baseline_nonsevere_edge = ~baseline_severe_edge

    def compose_current() -> torch.Tensor:
        return compose_direction_parameters(
            normals=normal,
            frame=frame,
            tangent_coordinate=tangent_coordinate,
            lift_coordinate=lift_coordinate,
            rho_epsilon=RHO_EPSILON,
        )

    with torch.no_grad():
        initial_candidate = compose_direction_parameters(
            normals=normal,
            frame=frame,
            tangent_coordinate=torch.zeros_like(tangent_coordinate),
            lift_coordinate=torch.zeros_like(lift_coordinate),
            rho_epsilon=RHO_EPSILON,
        )
        initial_objective = _objective(
            direction=initial_candidate,
            normals=normal,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=per_view_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
            edge_u=edge_u,
            edge_v=edge_v,
            baseline_data=baseline_data,
            baseline_surface=baseline_surface,
            baseline_tangent=baseline_tangent,
            baseline_lift=baseline_lift,
            baseline_nonnegative_edge=baseline_nonnegative_edge,
            baseline_nonsevere_edge=baseline_nonsevere_edge,
            smooth_weight=float(cfg.smooth_weight),
            orientation_barrier_weight=float(cfg.orientation_barrier_weight),
        )
        (
            initial_negative_barrier,
            initial_severe_barrier,
            initial_orientation_barrier,
        ) = _orientation_barrier_components(
            direction=initial_candidate,
            normals=normal,
            edge_u=edge_u,
            edge_v=edge_v,
            baseline_nonnegative_edge=baseline_nonnegative_edge,
            baseline_nonsevere_edge=baseline_nonsevere_edge,
        )
        initial_candidate_metrics, _, _, _, _ = _evaluate_metrics(
            direction=initial_candidate,
            normals=normal,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=per_view_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
            edge_u=edge_u,
            edge_v=edge_v,
        )

    best_objective = float(initial_objective.cpu())
    best_tangent = torch.zeros_like(tangent_coordinate)
    best_lift = torch.zeros_like(lift_coordinate)
    best_barrier = {
        "negative": float(initial_negative_barrier.cpu()),
        "severe": float(initial_severe_barrier.cpu()),
        "total": float(initial_orientation_barrier.cpu()),
    }
    history: list[dict[str, float | int]] = [
        _history_entry(
            iteration=0,
            objective=initial_objective,
            reference_direction=baseline,
            direction=initial_candidate,
            metrics=initial_candidate_metrics,
            tangent_coordinate=best_tangent,
            lift_coordinate=best_lift,
            negative_orientation_barrier=initial_negative_barrier,
            severe_orientation_barrier=initial_severe_barrier,
            orientation_barrier=initial_orientation_barrier,
        )
    ]
    no_improvement = 0
    optimizer_stop_reason = "iteration_limit"

    for iteration in range(1, int(cfg.iterations) + 1):
        candidate = compose_current()
        objective = _objective(
            direction=candidate,
            normals=normal,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=per_view_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
            edge_u=edge_u,
            edge_v=edge_v,
            baseline_data=baseline_data,
            baseline_surface=baseline_surface,
            baseline_tangent=baseline_tangent,
            baseline_lift=baseline_lift,
            baseline_nonnegative_edge=baseline_nonnegative_edge,
            baseline_nonsevere_edge=baseline_nonsevere_edge,
            smooth_weight=float(cfg.smooth_weight),
            orientation_barrier_weight=float(cfg.orientation_barrier_weight),
        )
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"non-finite global direction objective at iteration {iteration}")
        optimizer.zero_grad(set_to_none=True)
        if not objective.requires_grad:
            optimizer_stop_reason = "zero_gradient"
            break
        objective.backward()
        for parameter in (tangent_coordinate, lift_coordinate):
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            elif not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(
                    f"non-finite global direction gradient at iteration {iteration}"
                )
        optimizer.step()

        with torch.no_grad():
            updated_candidate = compose_direction_parameters(
                normals=normal,
                frame=frame,
                tangent_coordinate=tangent_coordinate,
                lift_coordinate=lift_coordinate,
                rho_epsilon=RHO_EPSILON,
            )
            updated_objective = _objective(
                direction=updated_candidate,
                normals=normal,
                projection_points=projection_points,
                per_view_axes=per_view_axes,
                per_view_weights=per_view_weights,
                viewmats=viewmats,
                intrinsics=intrinsics,
                edge_u=edge_u,
                edge_v=edge_v,
                baseline_data=baseline_data,
                baseline_surface=baseline_surface,
                baseline_tangent=baseline_tangent,
                baseline_lift=baseline_lift,
                baseline_nonnegative_edge=baseline_nonnegative_edge,
                baseline_nonsevere_edge=baseline_nonsevere_edge,
                smooth_weight=float(cfg.smooth_weight),
                orientation_barrier_weight=float(cfg.orientation_barrier_weight),
            )
            (
                updated_negative_barrier,
                updated_severe_barrier,
                updated_orientation_barrier,
            ) = _orientation_barrier_components(
                direction=updated_candidate,
                normals=normal,
                edge_u=edge_u,
                edge_v=edge_v,
                baseline_nonnegative_edge=baseline_nonnegative_edge,
                baseline_nonsevere_edge=baseline_nonsevere_edge,
            )
            updated_metrics, _, _, _, _ = _evaluate_metrics(
                direction=updated_candidate,
                normals=normal,
                projection_points=projection_points,
                per_view_axes=per_view_axes,
                per_view_weights=per_view_weights,
                viewmats=viewmats,
                intrinsics=intrinsics,
                edge_u=edge_u,
                edge_v=edge_v,
            )
            history.append(
                _history_entry(
                    iteration=iteration,
                    objective=updated_objective,
                    reference_direction=baseline,
                    direction=updated_candidate,
                    metrics=updated_metrics,
                    tangent_coordinate=tangent_coordinate,
                    lift_coordinate=lift_coordinate,
                    negative_orientation_barrier=updated_negative_barrier,
                    severe_orientation_barrier=updated_severe_barrier,
                    orientation_barrier=updated_orientation_barrier,
                )
            )

            updated_value = float(updated_objective.cpu())
            relative_gain = (best_objective - updated_value) / max(abs(best_objective), EPS)
            if updated_value < best_objective and relative_gain > float(cfg.relative_tolerance):
                best_objective = updated_value
                best_tangent = tangent_coordinate.detach().clone()
                best_lift = lift_coordinate.detach().clone()
                best_barrier = {
                    "negative": float(updated_negative_barrier.cpu()),
                    "severe": float(updated_severe_barrier.cpu()),
                    "total": float(updated_orientation_barrier.cpu()),
                }
                no_improvement = 0
            else:
                no_improvement += 1
            if no_improvement >= int(cfg.patience):
                optimizer_stop_reason = "patience"
                break

    attempts: list[dict[str, Any]] = []
    accepted_scale: float | None = None
    accepted_direction = baseline
    accepted_tangent = torch.zeros_like(best_tangent)
    accepted_lift = torch.zeros_like(best_lift)
    final_metrics = baseline_metrics
    acceptance_report: dict[str, Any] | None = None
    for step in range(int(cfg.backtracking_steps)):
        scale = 0.5**step
        with torch.no_grad():
            candidate = compose_direction_parameters(
                normals=normal,
                frame=frame,
                tangent_coordinate=best_tangent * scale,
                lift_coordinate=best_lift * scale,
                rho_epsilon=RHO_EPSILON,
            ).detach()
        (
            candidate_metrics,
            candidate_negative_edge,
            candidate_severe_edge,
            candidate_negative_root,
            candidate_severe_root,
        ) = _evaluate_metrics(
            direction=candidate,
            normals=normal,
            projection_points=projection_points,
            per_view_axes=per_view_axes,
            per_view_weights=per_view_weights,
            viewmats=viewmats,
            intrinsics=intrinsics,
            edge_u=edge_u,
            edge_v=edge_v,
        )
        (
            candidate_negative_barrier,
            candidate_severe_barrier,
            candidate_orientation_barrier,
        ) = _orientation_barrier_components(
            direction=candidate,
            normals=normal,
            edge_u=edge_u,
            edge_v=edge_v,
            baseline_nonnegative_edge=baseline_nonnegative_edge,
            baseline_nonsevere_edge=baseline_nonsevere_edge,
        )
        gate = _gate_candidate(
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            baseline_negative_edge=baseline_negative_edge,
            candidate_negative_edge=candidate_negative_edge,
            baseline_severe_edge=baseline_severe_edge,
            candidate_severe_edge=candidate_severe_edge,
            baseline_negative_root=baseline_negative_root,
            candidate_negative_root=candidate_negative_root,
            baseline_severe_root=baseline_severe_root,
            candidate_severe_root=candidate_severe_root,
            tolerance=float(cfg.acceptance_tolerance),
        )
        attempt = {
            "scale": float(scale),
            "metrics": candidate_metrics,
            "orientation_barrier": {
                "negative": float(candidate_negative_barrier.cpu()),
                "severe": float(candidate_severe_barrier.cpu()),
                "total": float(candidate_orientation_barrier.cpu()),
            },
            "gate": gate,
        }
        attempts.append(attempt)
        if bool(gate["passed"]):
            accepted_scale = float(scale)
            accepted_direction = candidate
            accepted_tangent = (best_tangent * scale).detach().clone()
            accepted_lift = (best_lift * scale).detach().clone()
            final_metrics = candidate_metrics
            acceptance_report = gate
            break

    if accepted_scale is None:
        stop_reason = "no_acceptable_scale"
        final_direction = baseline.detach().clone()
        final_tangent = torch.zeros_like(best_tangent)
        final_lift = torch.zeros_like(best_lift)
        final_metrics = baseline_metrics
    else:
        stop_reason = "accepted"
        final_direction = accepted_direction.detach().clone()
        final_tangent = accepted_tangent
        final_lift = accepted_lift

    with torch.no_grad():
        (
            final_negative_barrier,
            final_severe_barrier,
            final_orientation_barrier,
        ) = _orientation_barrier_components(
            direction=final_direction,
            normals=normal,
            edge_u=edge_u,
            edge_v=edge_v,
            baseline_nonnegative_edge=baseline_nonnegative_edge,
            baseline_nonsevere_edge=baseline_nonsevere_edge,
        )
    change = _direction_change_statistics(baseline, final_direction)
    final_acceptance = (
        acceptance_report
        if acceptance_report is not None
        else {
            "passed": False,
            "reason": "no_acceptable_scale",
            "negative_edge_count_nonincreasing": True,
            "severe_edge_count_nonincreasing": True,
            "negative_root_count_nonincreasing": True,
            "severe_root_count_nonincreasing": True,
            "new_negative_edge_count": 0,
            "new_severe_edge_count": 0,
            "no_new_negative_edges": True,
            "no_new_severe_edges": True,
            "new_negative_root_count": 0,
            "new_severe_root_count": 0,
            "no_new_negative_roots": True,
            "no_new_severe_roots": True,
            "edge_identity_gate_enforced": False,
            "root_support_gate_enforced": True,
        }
    )
    report: dict[str, Any] = {
        "schema": "anigroom.global_direction_field_refinement.v1",
        "status": "complete",
        "stop_reason": stop_reason,
        "accepted": accepted_scale is not None,
        "accepted_scale": accepted_scale,
        "initial_metrics": baseline_metrics,
        "final_metrics": final_metrics,
        "optimizer": {
            "best_objective": float(best_objective),
            "stop_reason": optimizer_stop_reason,
            "history": history,
        },
        "optimizer_history": history,
        "backtracking_attempts": attempts,
        "acceptance": final_acceptance,
        "direction_change": change,
        "direction_change_stats": change,
        "config": asdict(cfg),
        "orientation_barrier": {
            "weight": float(cfg.orientation_barrier_weight),
            "eligible_nonnegative_edge_count": int(
                baseline_nonnegative_edge.sum().cpu()
            ),
            "eligible_nonsevere_edge_count": int(baseline_nonsevere_edge.sum().cpu()),
            "formula": "mean(violation^2) + max(violation^2)",
            "initial": {
                "negative": float(initial_negative_barrier.cpu()),
                "severe": float(initial_severe_barrier.cpu()),
                "total": float(initial_orientation_barrier.cpu()),
            },
            "best_optimizer": best_barrier,
            "final": {
                "negative": float(final_negative_barrier.cpu()),
                "severe": float(final_severe_barrier.cpu()),
                "total": float(final_orientation_barrier.cpu()),
            },
        },
        "severe_edge_threshold": float(SEVERE_DOT_THRESHOLD),
        "initial_negative_edge_count": int(baseline_metrics["negative_edge_count"]),
        "final_negative_edge_count": int(final_metrics["negative_edge_count"]),
        "initial_severe_edge_count": int(baseline_metrics["severe_edge_count"]),
        "final_severe_edge_count": int(final_metrics["severe_edge_count"]),
        "initial_negative_root_count": int(baseline_metrics["negative_root_count"]),
        "final_negative_root_count": int(final_metrics["negative_root_count"]),
        "initial_severe_root_count": int(baseline_metrics["severe_root_count"]),
        "final_severe_root_count": int(final_metrics["severe_root_count"]),
        "new_negative_edge_count": int(final_acceptance["new_negative_edge_count"]),
        "new_severe_edge_count": int(final_acceptance["new_severe_edge_count"]),
        "new_negative_root_count": int(final_acceptance["new_negative_root_count"]),
        "new_severe_root_count": int(final_acceptance["new_severe_root_count"]),
        "no_new_negative_edges": bool(final_acceptance["no_new_negative_edges"]),
        "no_new_severe_edges": bool(final_acceptance["no_new_severe_edges"]),
        "no_new_negative_roots": bool(final_acceptance["no_new_negative_roots"]),
        "no_new_severe_roots": bool(final_acceptance["no_new_severe_roots"]),
        "edge_identity_gate_enforced": False,
        "root_support_gate_enforced": True,
        "best_optimizer_coordinate_norms": {
            "tangent_l2": float(torch.linalg.vector_norm(best_tangent).cpu()),
            "lift_l2": float(torch.linalg.vector_norm(best_lift).cpu()),
        },
    }
    if accepted_scale is None:
        report["no_acceptable_scale"] = True
    return GlobalDirectionFieldRefinementResult(
        direction=final_direction,
        tangent_coordinate=final_tangent,
        lift_coordinate=final_lift,
        report=report,
    )


run_global_direction_field_refinement = refine_global_direction_field


__all__ = [
    "EPS",
    "GlobalDirectionFieldRefinementConfig",
    "GlobalDirectionFieldRefinementResult",
    "refine_global_direction_field",
    "run_global_direction_field_refinement",
]

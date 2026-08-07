from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.flow.direction_geometry import parallel_transport_vector_field
from anigroom.grooming.geometry_residuals import local_components_to_world
from anigroom.mesh_roots import TriangleMesh, sample_surface_candidates
from anigroom.surface_interpolation import local_surface_weights
from train_white_tiger_stage1 import (
    build_stage1_model_from_checkpoint,
    load_training_checkpoint,
    stage1_config_from_checkpoint_mapping,
)


EPS = 1.0e-12


def seeded_local_fps(
    points: np.ndarray,
    seed_point: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    if count > int(points.shape[0]):
        raise RuntimeError("area-proportional placement exhausted a primary cell")
    minimum_distance = np.sum((points - seed_point[None, :]) ** 2, axis=1)
    available = np.ones((int(points.shape[0]),), dtype=bool)
    selected = np.empty((count,), dtype=np.int64)
    for index in range(count):
        score = np.where(available, minimum_distance, -np.inf)
        candidate = int(np.argmax(score))
        selected[index] = candidate
        available[candidate] = False
        distance = np.sum((points - points[candidate : candidate + 1]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return selected


@torch.no_grad()
def area_proportional_secondary_topology(
    model,
    *,
    secondary_root_count: int,
    candidate_multiplier: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    primary_count = int(model.guide_points_local.shape[0])
    secondary_root_count = int(secondary_root_count)
    if secondary_root_count < primary_count:
        raise ValueError("secondary root count must retain every primary guide anchor")
    extra_count = secondary_root_count - primary_count
    candidate_count = max(
        secondary_root_count,
        int(np.ceil(float(secondary_root_count) * float(candidate_multiplier))),
    )
    mesh = TriangleMesh(
        vertices=model.vertices.detach().cpu().numpy(),
        faces=model.faces.detach().cpu().numpy(),
    )
    candidates = sample_surface_candidates(mesh, candidate_count, int(seed))
    candidate_points = torch.as_tensor(
        candidates.points,
        device=model.vertices.device,
        dtype=model.vertices.dtype,
    )
    candidate_faces = torch.as_tensor(
        candidates.face_ids,
        device=model.vertices.device,
        dtype=torch.long,
    )
    candidate_support = model.guide_surface_interpolator().build_support(
        candidate_points,
        candidate_faces,
    )
    owner = candidate_support.indices[:, 0].cpu().numpy().astype(np.int64)
    cell_counts = np.bincount(owner, minlength=primary_count)
    raw_quota = float(extra_count) * cell_counts.astype(np.float64) / float(cell_counts.sum())
    allocation = np.floor(raw_quota).astype(np.int64)
    if np.any(allocation > cell_counts):
        raise RuntimeError("area-proportional allocation exceeds candidate capacity")
    remainder = int(extra_count - allocation.sum())
    if remainder > 0:
        fractional = raw_quota - allocation
        eligible = np.flatnonzero(allocation < cell_counts)
        order = np.lexsort((eligible, -fractional[eligible]))
        allocation[eligible[order[:remainder]]] += 1
    if int(allocation.sum()) != extra_count:
        raise RuntimeError("area-proportional allocation did not preserve root count")

    primary_points = model.guide_points_local.detach().cpu().numpy()
    primary_faces = model.guide_face_ids.detach().cpu().numpy()
    primary_barycentric = model.guide_barycentric.detach().cpu().numpy()
    cell_order = np.argsort(owner, kind="stable")
    cell_offsets = np.concatenate([[0], np.cumsum(cell_counts)])
    output_faces: list[np.ndarray] = []
    output_barycentric: list[np.ndarray] = []
    output_parent: list[np.ndarray] = []
    selected_distance: list[np.ndarray] = []
    for parent_id in range(primary_count):
        output_faces.append(primary_faces[parent_id : parent_id + 1])
        output_barycentric.append(primary_barycentric[parent_id : parent_id + 1])
        output_parent.append(np.asarray([parent_id], dtype=np.int64))
        local_count = int(allocation[parent_id])
        if local_count <= 0:
            continue
        local_ids = cell_order[cell_offsets[parent_id] : cell_offsets[parent_id + 1]]
        local_selection = seeded_local_fps(
            candidates.points[local_ids],
            primary_points[parent_id],
            local_count,
        )
        selected = local_ids[local_selection]
        output_faces.append(candidates.face_ids[selected])
        output_barycentric.append(candidates.barycentric[selected])
        output_parent.append(np.full((local_count,), parent_id, dtype=np.int64))
        selected_distance.append(
            np.linalg.norm(
                candidates.points[selected] - primary_points[parent_id : parent_id + 1],
                axis=-1,
            )
        )
    face_ids = np.concatenate(output_faces).astype(np.int64, copy=False)
    barycentric = np.concatenate(output_barycentric).astype(np.float32, copy=False)
    parent_ids = np.concatenate(output_parent).astype(np.int64, copy=False)
    distances = np.concatenate(selected_distance) if selected_distance else np.empty((0,))
    if int(face_ids.shape[0]) != secondary_root_count:
        raise RuntimeError("area-proportional placement produced the wrong root count")
    return face_ids, barycentric, parent_ids, {
        "mode": "area_proportional_primary_cells",
        "candidate_count": int(candidate_count),
        "candidate_cell_count": {
            "min": int(cell_counts.min()),
            "median": float(np.median(cell_counts)),
            "max": int(cell_counts.max()),
        },
        "children_per_primary": {
            "min": int((allocation + 1).min()),
            "median": float(np.median(allocation + 1)),
            "max": int((allocation + 1).max()),
        },
        "selected_parent_distance": {
            "mean": float(distances.mean()) if distances.size else 0.0,
            "max": float(distances.max(initial=0.0)),
        },
        "support_fallback_query_count": int(
            candidate_support.report.get("fallback_query_count", 0)
        ),
    }


def quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().reshape(-1).float()
    if flat.numel() == 0:
        return {"min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    levels = flat.new_tensor([0.0, 0.05, 0.5, 0.95, 1.0])
    result = torch.quantile(flat, levels).cpu().tolist()
    return dict(zip(("min", "p05", "p50", "p95", "max"), map(float, result)))


def interpolation_forward(
    source: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return (source[indices] * weights[..., None]).sum(dim=1)


def interpolation_adjoint(
    query: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    source_count: int,
) -> torch.Tensor:
    contribution = weights[..., None] * query[:, None, :]
    result = query.new_zeros((source_count, query.shape[-1]))
    result.index_add_(0, indices.reshape(-1), contribution.reshape(-1, query.shape[-1]))
    return result


def conjugate_gradient_least_squares(
    target: torch.Tensor,
    source_count: int,
    diagonal: torch.Tensor,
    forward: Callable[[torch.Tensor], torch.Tensor],
    adjoint: Callable[[torch.Tensor], torch.Tensor],
    *,
    iterations: int,
    tolerance: float,
    coupled_channels: bool,
    label: str,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    ridge = diagonal.mean().clamp_min(EPS) * 1.0e-7
    preconditioner = (diagonal + ridge).clamp_min(EPS)[:, None]

    def normal_operator(value: torch.Tensor) -> torch.Tensor:
        return adjoint(forward(value)) + ridge * value

    right_hand_side = adjoint(target)
    estimate = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    preconditioned = residual / preconditioner
    search = preconditioned.clone()

    def inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        product = left * right
        if coupled_channels:
            return product.sum().reshape(1)
        return product.sum(dim=0)

    rhs_norm = inner(right_hand_side, right_hand_side).clamp_min(EPS)
    residual_dot = inner(residual, preconditioned)
    relative = torch.sqrt(inner(residual, residual) / rhs_norm).max()
    completed = 0
    for step in range(1, int(iterations) + 1):
        operator_search = normal_operator(search)
        denominator = inner(search, operator_search).clamp_min(EPS)
        alpha = residual_dot / denominator
        if coupled_channels:
            estimate = estimate + alpha[0] * search
            residual = residual - alpha[0] * operator_search
        else:
            estimate = estimate + search * alpha[None, :]
            residual = residual - operator_search * alpha[None, :]
        relative = torch.sqrt(inner(residual, residual) / rhs_norm).max()
        completed = step
        if step == 1 or step % 10 == 0:
            print(
                json.dumps(
                    {
                        "audit_progress": label,
                        "cg_iteration": step,
                        "relative_normal_residual": float(relative.detach().cpu()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if float(relative.detach().cpu()) <= float(tolerance):
            break
        next_preconditioned = residual / preconditioner
        next_dot = inner(residual, next_preconditioned)
        beta = next_dot / residual_dot.clamp_min(EPS)
        if coupled_channels:
            search = next_preconditioned + beta[0] * search
        else:
            search = next_preconditioned + search * beta[None, :]
        preconditioned = next_preconditioned
        residual_dot = next_dot
    return estimate, {
        "iterations": int(completed),
        "relative_normal_residual": float(relative.detach().cpu()),
        "ridge": float(ridge.detach().cpu()),
    }


def field_fit_metrics(
    target: torch.Tensor,
    prediction: torch.Tensor,
    names: list[str],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for channel, name in enumerate(names):
        expected = target[:, channel]
        actual = prediction[:, channel]
        error = actual - expected
        centered = expected - expected.mean()
        squared_error = error.square().sum()
        centered_energy = centered.square().sum()
        rms = expected.square().mean().sqrt()
        rmse = error.square().mean().sqrt()
        report[name] = {
            "target_rms": float(rms.detach().cpu()),
            "rmse": float(rmse.detach().cpu()),
            "relative_rmse_to_rms": float((rmse / rms.clamp_min(EPS)).detach().cpu()),
            "explained_variance": float(
                (1.0 - squared_error / centered_energy.clamp_min(EPS)).detach().cpu()
            ),
            "absolute_error_p95": float(
                torch.quantile(error.abs().float(), 0.95).detach().cpu()
            ),
        }
    return report


def audit_support(
    query_points: torch.Tensor,
    source_points: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    parent_ids: torch.Tensor,
    primary_owner: torch.Tensor,
) -> dict[str, object]:
    distances = torch.linalg.norm(
        query_points[:, None, :] - source_points[indices],
        dim=-1,
    )
    effective_count = weights.square().sum(dim=1).clamp_min(EPS).reciprocal()
    entropy_count = torch.exp(
        -(weights.clamp_min(EPS) * weights.clamp_min(EPS).log()).sum(dim=1)
    )
    source_mass = weights.new_zeros((source_points.shape[0],))
    source_mass.index_add_(0, indices.reshape(-1), weights.reshape(-1))
    source_hits = torch.bincount(
        indices.reshape(-1),
        minlength=source_points.shape[0],
    )

    primary_count = int(parent_ids.max().detach().cpu()) + 1
    secondary_per_primary = torch.bincount(parent_ids, minlength=primary_count)
    render_per_primary = torch.bincount(primary_owner, minlength=primary_count)
    render_per_secondary_budget = (
        render_per_primary.float() / secondary_per_primary.float().clamp_min(1.0)
    )
    return {
        "query_count": int(query_points.shape[0]),
        "source_count": int(source_points.shape[0]),
        "neighbor_count": int(indices.shape[1]),
        "nearest_distance": quantiles(distances[:, 0]),
        "farthest_support_distance": quantiles(distances[:, -1]),
        "weighted_distance": quantiles((distances * weights).sum(dim=1)),
        "effective_support_count": quantiles(effective_count),
        "entropy_support_count": quantiles(entropy_count),
        "source_interpolation_mass": quantiles(source_mass),
        "source_support_hits": quantiles(source_hits.float()),
        "zero_mass_source_count": int((source_mass <= 0.0).sum().detach().cpu()),
        "secondary_per_primary": quantiles(secondary_per_primary.float()),
        "render_per_primary": quantiles(render_per_primary.float()),
        "render_per_secondary_budget": quantiles(render_per_secondary_budget),
    }


@torch.no_grad()
def run_audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    checkpoint = load_training_checkpoint(args.target_checkpoint)
    config_mapping = checkpoint.get("config")
    if not isinstance(config_mapping, dict):
        raise RuntimeError("target checkpoint has no embedded config")
    config = stage1_config_from_checkpoint_mapping(config_mapping)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("target checkpoint has no model state")
    if config.geometry_residual_domain != "render":
        raise RuntimeError("target checkpoint must contain render-root residuals")
    # R043 predates the persistent secondary-guide topology buffers. Their
    # semantically exact value in a render-domain checkpoint is an empty tensor.
    # Migrate only this known old schema in the read-only copy used by the audit;
    # the formal checkpoint loader and checkpoint file remain unchanged.
    migrated_state = dict(state)
    migrated_state.setdefault("secondary_guide_face_ids", torch.empty((0,), dtype=torch.long))
    migrated_state.setdefault("secondary_guide_barycentric", torch.empty((0, 3)))
    migrated_state.setdefault("secondary_guide_points_local", torch.empty((0, 3)))
    migrated_state.setdefault("secondary_guide_parent_ids", torch.empty((0,), dtype=torch.long))
    migrated_checkpoint = dict(checkpoint)
    migrated_checkpoint["model"] = migrated_state
    model = build_stage1_model_from_checkpoint(migrated_checkpoint, config, device)
    if model.geometry_residual_domain != "render":
        raise RuntimeError("target checkpoint must contain render-root residuals")
    target_field = model.render_geometry_residual
    if target_field is None:
        raise RuntimeError("target checkpoint has no render-root residual field")

    basis_checkpoint = load_training_checkpoint(args.basis_checkpoint)
    basis_state = basis_checkpoint.get("model")
    if not isinstance(basis_state, dict):
        raise RuntimeError("basis checkpoint has no model state")
    required = ["guide_face_ids", "guide_barycentric"]
    if args.placement == "checkpoint":
        required.extend(
            [
                "secondary_guide_face_ids",
                "secondary_guide_barycentric",
                "secondary_guide_parent_ids",
            ]
        )
    missing = [name for name in required if name not in basis_state]
    if missing:
        raise RuntimeError("basis checkpoint is missing: " + ", ".join(missing))
    if not torch.equal(
        model.guide_face_ids.cpu(),
        basis_state["guide_face_ids"].cpu(),
    ) or not torch.allclose(
        model.guide_barycentric.cpu(),
        basis_state["guide_barycentric"].cpu(),
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("target and basis checkpoints do not share primary guides")

    target_scalars = {
        name: parameter.detach().clone()
        for name, parameter in target_field.named_parameters()
        if name != "direction_local_raw"
        and float(parameter.detach().abs().max().cpu()) > 1.0e-10
    }
    scalar_names = sorted(target_scalars)
    scalar_target = torch.cat([target_scalars[name] for name in scalar_names], dim=1)
    target_decoded = target_field.decode()
    _, query_normals, query_points = model.roots_and_normals()
    query_tangents, query_bitangents = model.tangent_frames(query_normals)
    direction_target = local_components_to_world(
        target_decoded.direction_local,
        query_normals,
        query_tangents,
        query_bitangents,
        normalize=False,
    )

    model.geometry_residual_domain = "secondary_guide"
    model.secondary_guide_interpolation_k = max(args.interpolation_k)
    if args.placement == "checkpoint":
        secondary_face_ids = basis_state["secondary_guide_face_ids"]
        secondary_barycentric = basis_state["secondary_guide_barycentric"]
        secondary_parent_ids = basis_state["secondary_guide_parent_ids"]
        placement_report: dict[str, object] = {
            "mode": "checkpoint",
            "secondary_root_count": int(secondary_face_ids.shape[0]),
        }
    else:
        (
            secondary_face_ids,
            secondary_barycentric,
            secondary_parent_ids,
            placement_report,
        ) = area_proportional_secondary_topology(
            model,
            secondary_root_count=args.secondary_root_count,
            candidate_multiplier=args.candidate_multiplier,
            seed=int(getattr(config, "seed", 29)) + 29,
        )
        placement_report["secondary_root_count"] = int(secondary_face_ids.shape[0])
    model.attach_secondary_guides(
        secondary_face_ids,
        secondary_barycentric,
        secondary_parent_ids,
    )
    source_points = model.secondary_guide_points_local
    parent_ids = model.secondary_guide_parent_ids
    source_normals, _, _ = model.tangent_frames_for_face_ids(
        model.secondary_guide_face_ids
    )
    primary_owner = model.guide_interpolation_support().indices[:, 0]

    result: dict[str, object] = {
        "target_checkpoint": str(args.target_checkpoint),
        "basis_checkpoint": str(args.basis_checkpoint),
        "target_iteration": int(checkpoint.get("iteration", -1)),
        "target_render_root_count": int(query_points.shape[0]),
        "secondary_root_count": int(source_points.shape[0]),
        "placement": placement_report,
        "target_geometry_domain": str(config.geometry_residual_domain),
        "audits": {},
    }
    for neighbor_count in args.interpolation_k:
        model.secondary_guide_interpolation_k = int(neighbor_count)
        model.rebuild_secondary_render_support()
        support = model.secondary_render_support()
        indices = support.indices
        weights = local_surface_weights(query_points, source_points, support)
        source_count = int(source_points.shape[0])
        diagonal = weights.new_zeros((source_count,))
        diagonal.index_add_(0, indices.reshape(-1), weights.square().reshape(-1))

        scalar_forward = lambda value: interpolation_forward(value, indices, weights)
        scalar_adjoint = lambda value: interpolation_adjoint(
            value,
            indices,
            weights,
            source_count,
        )
        scalar_solution, scalar_solver = conjugate_gradient_least_squares(
            scalar_target,
            source_count,
            diagonal,
            scalar_forward,
            scalar_adjoint,
            iterations=args.cg_iterations,
            tolerance=args.cg_tolerance,
            coupled_channels=False,
            label=f"K{neighbor_count}_scalar",
        )
        scalar_prediction = scalar_forward(scalar_solution)

        gathered_source_normals = source_normals[indices]
        gathered_query_normals = query_normals[:, None, :].expand_as(
            gathered_source_normals
        )

        def direction_forward(value: torch.Tensor) -> torch.Tensor:
            transported = parallel_transport_vector_field(
                value[indices],
                gathered_source_normals,
                gathered_query_normals,
            )
            return (transported * weights[..., None]).sum(dim=1)

        def direction_adjoint(value: torch.Tensor) -> torch.Tensor:
            transported = parallel_transport_vector_field(
                value[:, None, :].expand_as(gathered_query_normals),
                gathered_query_normals,
                gathered_source_normals,
            )
            contribution = transported * weights[..., None]
            output = value.new_zeros((source_count, 3))
            output.index_add_(0, indices.reshape(-1), contribution.reshape(-1, 3))
            return output

        probe_source = torch.randn(
            (source_count, 3),
            device=device,
            dtype=query_points.dtype,
        )
        probe_query = torch.randn_like(direction_target)
        left = (direction_forward(probe_source) * probe_query).sum()
        right = (probe_source * direction_adjoint(probe_query)).sum()
        adjoint_relative_error = (left - right).abs() / torch.maximum(
            left.abs(),
            right.abs(),
        ).clamp_min(EPS)
        if float(adjoint_relative_error.detach().cpu()) > 1.0e-4:
            raise RuntimeError(
                "direction transport adjoint check failed: "
                f"{float(adjoint_relative_error.detach().cpu())}"
            )

        direction_solution, direction_solver = conjugate_gradient_least_squares(
            direction_target,
            source_count,
            diagonal,
            direction_forward,
            direction_adjoint,
            iterations=args.cg_iterations,
            tolerance=args.cg_tolerance,
            coupled_channels=True,
            label=f"K{neighbor_count}_direction",
        )
        direction_prediction = direction_forward(direction_solution)
        direction_error = direction_prediction - direction_target
        target_energy = direction_target.square().sum()
        direction_report = {
            "target_rms": float(direction_target.square().mean().sqrt().cpu()),
            "rmse": float(direction_error.square().mean().sqrt().cpu()),
            "relative_rmse_to_rms": float(
                (direction_error.square().sum() / target_energy.clamp_min(EPS))
                .sqrt()
                .cpu()
            ),
            "vector_error_p95": float(
                torch.quantile(direction_error.norm(dim=-1).float(), 0.95).cpu()
            ),
            "explained_energy": float(
                (1.0 - direction_error.square().sum() / target_energy.clamp_min(EPS)).cpu()
            ),
            "adjoint_relative_error": float(adjoint_relative_error.cpu()),
        }
        result["audits"][f"K{neighbor_count}"] = {
            "support": audit_support(
                query_points,
                source_points,
                indices,
                weights,
                parent_ids,
                primary_owner,
            ),
            "scalar_solver": scalar_solver,
            "scalar_fit": field_fit_metrics(
                scalar_target,
                scalar_prediction,
                scalar_names,
            ),
            "direction_solver": direction_solver,
            "direction_fit": direction_report,
        }
        print(
            json.dumps(
                {
                    "audit_complete": f"K{neighbor_count}",
                    "scalar_fit": result["audits"][f"K{neighbor_count}"]["scalar_fit"],
                    "direction_fit": direction_report,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the upper-bound fit of a fixed secondary-guide basis."
    )
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--basis-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interpolation-k", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--placement",
        choices=("checkpoint", "area_proportional"),
        default="checkpoint",
    )
    parser.add_argument("--secondary-root-count", type=int, default=20_000)
    parser.add_argument("--candidate-multiplier", type=float, default=16.0)
    parser.add_argument("--cg-iterations", type=int, default=80)
    parser.add_argument("--cg-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.interpolation_k or any(value <= 0 for value in args.interpolation_k):
        raise ValueError("interpolation K values must be positive")
    if args.secondary_root_count <= 0:
        raise ValueError("secondary root count must be positive")
    if args.candidate_multiplier < 1.0:
        raise ValueError("candidate multiplier must be at least one")
    result = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"audit_report": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

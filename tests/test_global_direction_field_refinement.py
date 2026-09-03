from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from anigroom.flow.direction_geometry import parallel_transport_vectors
from anigroom.flow.global_direction_field_refinement import (
    GlobalDirectionFieldRefinementConfig,
    _gate_candidate,
    refine_global_direction_field,
)


DTYPE = torch.float64


def _camera() -> tuple[torch.Tensor, torch.Tensor]:
    viewmats = torch.eye(4, dtype=DTYPE)[None]
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]],
        dtype=DTYPE,
    )
    return viewmats, intrinsics


def _curved_chain_problem() -> dict[str, torch.Tensor]:
    # A short curve on a surface whose normals bend in the x-z plane.  The
    # transported y direction is coherent, while the middle root is a
    # tangent outlier and has no selected direct-vector weight.
    bend = torch.tensor([-0.30, 0.0, 0.30], dtype=DTYPE)
    normals = torch.stack(
        (torch.sin(bend), torch.zeros_like(bend), torch.cos(bend)), dim=-1
    )
    coherent = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    expected = coherent.expand(3, 3).clone()
    outlier = F.normalize(torch.linalg.cross(normals[1], coherent), dim=-1)
    direction = expected.clone()
    direction[1] = outlier
    viewmats, intrinsics = _camera()
    weak_direct = F.normalize(
        coherent.expand(3, 3).clone() + 0.15 * F.normalize(
            torch.linalg.cross(normals, coherent.expand(3, 3)), dim=-1
        ),
        dim=-1,
    )
    return {
        "direction": direction,
        "normals": normals,
        "projection_points": torch.tensor(
            [[-0.8, 0.0, 5.0], [0.0, 0.0, 5.1], [0.8, 0.0, 5.0]],
            dtype=DTYPE,
        ),
        "per_view_axes": weak_direct[None].clone(),
        "per_view_weights": torch.tensor([[1.0, 0.0, 1.0]], dtype=DTYPE),
        "viewmats": viewmats,
        "intrinsics": intrinsics,
        "edge_u": torch.tensor([0, 1], dtype=torch.long),
        "edge_v": torch.tensor([1, 2], dtype=torch.long),
    }


def test_curved_chain_repairs_no_data_middle_and_passes_all_gates() -> None:
    problem = _curved_chain_problem()
    baseline = problem["direction"]
    result = refine_global_direction_field(
        **problem,
        config=GlobalDirectionFieldRefinementConfig(
            smooth_weight=1.0,
            orientation_barrier_weight=0.0,
            iterations=160,
            learning_rate=0.03,
            patience=30,
            relative_tolerance=1.0e-9,
            backtracking_steps=8,
            acceptance_tolerance=1.0e-8,
        ),
    )

    expected = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    assert result.report["accepted"] is True
    assert result.report["accepted_scale"] is not None
    assert result.report["stop_reason"] == "accepted"
    assert float((result.direction[1] * expected).sum()) > float(
        (baseline[1] * expected).sum()
    )
    assert result.report["final_metrics"]["surface"] < result.report["initial_metrics"][
        "surface"
    ]
    assert result.report["final_metrics"]["tangent"] < result.report["initial_metrics"][
        "tangent"
    ]
    assert result.report["final_metrics"]["edge_p95_deg"] <= result.report[
        "initial_metrics"
    ]["edge_p95_deg"] + 1.0e-8
    assert result.report["final_metrics"]["edge_p99_deg"] <= result.report[
        "initial_metrics"
    ]["edge_p99_deg"] + 1.0e-8
    acceptance = result.report["acceptance"]
    assert all(acceptance["nonincreasing"].values())
    assert acceptance["new_negative_root_count"] == 0
    assert acceptance["new_severe_root_count"] == 0
    assert acceptance["no_new_negative_roots"] is True
    assert acceptance["no_new_severe_roots"] is True
    assert acceptance["edge_identity_gate_enforced"] is False
    assert acceptance["root_support_gate_enforced"] is True
    assert acceptance["any_strict_improvement"] is True
    assert result.report["orientation_barrier"]["weight"] == 0.0
    assert all(
        "orientation_barrier" in entry
        and "negative_orientation_barrier" in entry
        and "severe_orientation_barrier" in entry
        for entry in result.report["optimizer_history"]
    )
    for name in (
        "edge_top1pct_cvar_deg",
        "edge_max_deg",
        "negative_edge_count",
        "severe_edge_count",
        "negative_root_count",
        "severe_root_count",
    ):
        assert name in result.report["initial_metrics"]
        assert name in result.report["final_metrics"]


def test_zero_smooth_weight_with_zero_data_is_deterministic_no_change() -> None:
    problem = _curved_chain_problem()
    problem["per_view_axes"].zero_()
    problem["per_view_weights"].zero_()
    config = GlobalDirectionFieldRefinementConfig(
        smooth_weight=0.0,
        iterations=25,
        learning_rate=0.04,
        patience=5,
        relative_tolerance=1.0e-9,
        backtracking_steps=5,
        acceptance_tolerance=1.0e-8,
    )
    first = refine_global_direction_field(**problem, config=config)
    second = refine_global_direction_field(**problem, config=config)

    torch.testing.assert_close(first.direction, problem["direction"])
    torch.testing.assert_close(second.direction, problem["direction"])
    torch.testing.assert_close(first.direction, second.direction, atol=0.0, rtol=0.0)
    assert first.report["stop_reason"] == "no_acceptable_scale"
    assert first.report["no_acceptable_scale"] is True
    assert first.report["accepted_scale"] is None
    assert first.report["initial_metrics"] == first.report["final_metrics"]
    assert first.report["optimizer_history"] == second.report["optimizer_history"]


def test_already_fixed_point_returns_baseline_without_acceptance() -> None:
    problem = _curved_chain_problem()
    coherent = torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
    problem["direction"] = coherent.expand(3, 3).clone()
    problem["per_view_axes"] = problem["direction"][None].clone()
    problem["per_view_weights"] = torch.ones((1, 3), dtype=DTYPE)
    result = refine_global_direction_field(
        **problem,
        config=GlobalDirectionFieldRefinementConfig(
            iterations=30,
            patience=5,
            backtracking_steps=4,
        ),
    )

    torch.testing.assert_close(result.direction, problem["direction"])
    assert result.report["accepted"] is False
    assert result.report["stop_reason"] == "no_acceptable_scale"
    assert result.report["accepted_scale"] is None


def test_strong_orientation_barrier_avoids_optimizer_created_bad_root() -> None:
    def tangent_direction(angle_deg: float) -> list[float]:
        angle = math.radians(angle_deg)
        return [math.cos(angle), math.sin(angle), 0.0]

    viewmats, intrinsics = _camera()
    problem = {
        "direction": torch.tensor(
            [tangent_direction(angle) for angle in (0.0, 120.0, 205.0)],
            dtype=DTYPE,
        ),
        "normals": torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=DTYPE),
        "projection_points": torch.tensor(
            [[-1.0, 0.0, 5.0], [0.0, 0.0, 5.0], [1.0, 0.0, 5.0]],
            dtype=DTYPE,
        ),
        "per_view_axes": torch.tensor(
            [[tangent_direction(angle) for angle in (20.0, 120.0, 195.0)]],
            dtype=DTYPE,
        ),
        "per_view_weights": torch.tensor([[1.0, 0.0, 1.0]], dtype=DTYPE),
        "viewmats": viewmats,
        "intrinsics": intrinsics,
        # Duplicate support on the existing 120-degree conflict makes the
        # unconstrained optimum move its negative edge onto the clean root.
        "edge_u": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_v": torch.tensor([1, 1, 2], dtype=torch.long),
    }
    common = {
        "smooth_weight": 2.0,
        "iterations": 100,
        "learning_rate": 0.02,
        "patience": 30,
        "relative_tolerance": 1.0e-10,
        "backtracking_steps": 1,
        "acceptance_tolerance": 1.0e-8,
    }
    without_barrier = refine_global_direction_field(
        **problem,
        config=GlobalDirectionFieldRefinementConfig(
            **common,
            orientation_barrier_weight=0.0,
        ),
    )
    with_barrier = refine_global_direction_field(
        **problem,
        config=GlobalDirectionFieldRefinementConfig(
            **common,
            orientation_barrier_weight=1000.0,
        ),
    )

    unconstrained_attempt = without_barrier.report["backtracking_attempts"][0]
    assert unconstrained_attempt["metrics"]["surface"] < without_barrier.report[
        "initial_metrics"
    ]["surface"]
    assert unconstrained_attempt["gate"]["new_negative_root_count"] == 1
    assert unconstrained_attempt["orientation_barrier"]["negative"] > 0.0
    assert without_barrier.report["accepted"] is False

    assert with_barrier.report["accepted"] is True
    assert with_barrier.report["accepted_scale"] == 1.0
    assert with_barrier.report["acceptance"]["new_negative_root_count"] == 0
    assert with_barrier.report["acceptance"]["new_severe_root_count"] == 0
    assert with_barrier.report["final_metrics"]["surface"] < with_barrier.report[
        "initial_metrics"
    ]["surface"]
    assert with_barrier.report["orientation_barrier"]["weight"] == 1000.0
    assert with_barrier.report["orientation_barrier"]["final"]["total"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("smooth_weight", -1.0),
        ("orientation_barrier_weight", -1.0),
        ("orientation_barrier_weight", float("inf")),
        ("learning_rate", 0.0),
        ("relative_tolerance", -1.0),
        ("acceptance_tolerance", float("nan")),
        ("iterations", 0),
        ("patience", True),
        ("backtracking_steps", 1.5),
    ],
)
def test_config_validation(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        GlobalDirectionFieldRefinementConfig(**{field: value})


def test_input_validation_rejects_bad_graph_and_weight_shape() -> None:
    problem = _curved_chain_problem()
    bad_graph = dict(problem)
    bad_graph["edge_u"] = torch.tensor([0, 3], dtype=torch.long)
    with pytest.raises(ValueError, match="out-of-range"):
        refine_global_direction_field(
            **bad_graph,
            config=GlobalDirectionFieldRefinementConfig(),
        )

    bad_weights = dict(problem)
    bad_weights["per_view_weights"] = torch.ones((1, 2), dtype=DTYPE)
    with pytest.raises(ValueError, match="per_view_weights"):
        refine_global_direction_field(
            **bad_weights,
            config=GlobalDirectionFieldRefinementConfig(),
        )


def test_edge_identity_can_move_inside_existing_bad_root_support() -> None:
    baseline_metrics = {
        "data": 1.0,
        "surface": 1.0,
        "tangent": 1.0,
        "lift": 1.0,
        "edge_p95_deg": 160.0,
        "edge_p99_deg": 168.0,
        "edge_top1pct_cvar_deg": 170.0,
        "edge_max_deg": 170.0,
        "negative_edge_count": 2,
        "severe_edge_count": 2,
        "negative_root_count": 3,
        "severe_root_count": 3,
    }
    candidate_metrics = {
        "data": 0.5,
        "surface": 0.5,
        "tangent": 0.5,
        "lift": 0.5,
        "edge_p95_deg": 140.0,
        "edge_p99_deg": 148.0,
        "edge_top1pct_cvar_deg": 150.0,
        "edge_max_deg": 150.0,
        "negative_edge_count": 1,
        "severe_edge_count": 1,
        "negative_root_count": 2,
        "severe_root_count": 2,
    }
    gate = _gate_candidate(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_negative_edge=torch.tensor([True, True, False]),
        candidate_negative_edge=torch.tensor([False, False, True]),
        baseline_severe_edge=torch.tensor([True, True, False]),
        candidate_severe_edge=torch.tensor([False, False, True]),
        baseline_negative_root=torch.tensor([True, True, True]),
        candidate_negative_root=torch.tensor([False, True, True]),
        baseline_severe_root=torch.tensor([True, True, True]),
        candidate_severe_root=torch.tensor([False, True, True]),
        tolerance=1.0e-8,
    )

    assert gate["passed"] is True
    assert all(gate["nonincreasing"].values())
    assert gate["new_negative_edge_count"] == 1
    assert gate["new_severe_edge_count"] == 1
    assert gate["no_new_negative_edges"] is False
    assert gate["no_new_severe_edges"] is False
    assert gate["new_negative_root_count"] == 0
    assert gate["new_severe_root_count"] == 0
    assert gate["no_new_negative_roots"] is True
    assert gate["no_new_severe_roots"] is True


def test_new_bad_root_fails_even_when_tail_and_counts_do_not_increase() -> None:
    baseline_metrics = {
        "data": 1.0,
        "surface": 1.0,
        "tangent": 1.0,
        "lift": 1.0,
        "edge_p95_deg": 160.0,
        "edge_p99_deg": 168.0,
        "edge_top1pct_cvar_deg": 170.0,
        "edge_max_deg": 170.0,
        "negative_edge_count": 1,
        "severe_edge_count": 1,
        "negative_root_count": 2,
        "severe_root_count": 2,
    }
    candidate_metrics = {
        "data": 0.5,
        "surface": 0.5,
        "tangent": 0.5,
        "lift": 0.5,
        "edge_p95_deg": 140.0,
        "edge_p99_deg": 148.0,
        "edge_top1pct_cvar_deg": 150.0,
        "edge_max_deg": 150.0,
        "negative_edge_count": 1,
        "severe_edge_count": 1,
        "negative_root_count": 2,
        "severe_root_count": 2,
    }
    gate = _gate_candidate(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_negative_edge=torch.tensor([True, False]),
        candidate_negative_edge=torch.tensor([False, True]),
        baseline_severe_edge=torch.tensor([True, False]),
        candidate_severe_edge=torch.tensor([False, True]),
        baseline_negative_root=torch.tensor([True, True, False]),
        candidate_negative_root=torch.tensor([False, True, True]),
        baseline_severe_root=torch.tensor([True, True, False]),
        candidate_severe_root=torch.tensor([False, True, True]),
        tolerance=1.0e-8,
    )

    assert all(gate["nonincreasing"].values())
    assert gate["new_negative_edge_count"] == 1
    assert gate["new_severe_edge_count"] == 1
    assert gate["new_negative_root_count"] == 1
    assert gate["new_severe_root_count"] == 1
    assert gate["no_new_negative_roots"] is False
    assert gate["no_new_severe_roots"] is False
    assert gate["passed"] is False


def test_transport_reference_is_curved_not_euclidean() -> None:
    problem = _curved_chain_problem()
    transported = parallel_transport_vectors(
        problem["direction"][0:1],
        problem["normals"][0:1],
        problem["normals"][1:2],
    )[0]
    assert math.isclose(float(transported[1]), 1.0, rel_tol=0.0, abs_tol=1.0e-12)

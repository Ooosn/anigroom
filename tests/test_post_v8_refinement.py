from __future__ import annotations

import torch
import torch.nn.functional as F

import anigroom.flow.post_v8_refinement as module
from anigroom.flow.post_v8_refinement import (
    PostV8RefinementConfig,
    refit_tangent_angles,
    run_post_v8_refinement,
)


DTYPE = torch.float64


def _camera() -> tuple[torch.Tensor, torch.Tensor]:
    viewmat = torch.eye(4, dtype=DTYPE)[None]
    intrinsic = torch.tensor(
        [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]],
        dtype=DTYPE,
    )
    return viewmat, intrinsic


def _problem(angle_deg: float = 30.0) -> dict[str, torch.Tensor]:
    angle = torch.deg2rad(torch.tensor(angle_deg, dtype=DTYPE))
    evidence = torch.tensor(
        [float(torch.cos(angle)), float(torch.sin(angle)), 0.0],
        dtype=DTYPE,
    )
    viewmats, intrinsics = _camera()
    return {
        "direction": torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=DTYPE),
        "normals": torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=DTYPE),
        "projection_points": torch.tensor(
            [[-0.5, 0.0, 5.0], [0.5, 0.0, 5.0]],
            dtype=DTYPE,
        ),
        "per_view_axes": evidence.reshape(1, 1, 3).expand(1, 2, 3).clone(),
        "per_view_weights": torch.ones((1, 2), dtype=DTYPE),
        "viewmats": viewmats,
        "intrinsics": intrinsics,
    }


def _config(**overrides) -> PostV8RefinementConfig:
    values = {
        "ba_iterations": 180,
        "ba_learning_rate": 0.04,
        "ba_relative_tolerance": 1.0e-9,
        "ba_patience": 40,
        "outer_max_cycles": 3,
        "outer_relative_tolerance": 1.0e-5,
        "outer_change_p95_tolerance_deg": 0.05,
        "acceptance_tolerance": 1.0e-8,
        "backtracking_steps": 4,
    }
    values.update(overrides)
    return PostV8RefinementConfig(**values)


def test_tangent_refit_reduces_multiview_error_and_preserves_lift() -> None:
    problem = _problem()
    result = refit_tangent_angles(**problem, config=_config())

    assert result.final_data_energy < result.initial_data_energy * 0.01
    initial_normal = (problem["direction"] * problem["normals"]).sum(dim=-1)
    final_normal = (result.direction * problem["normals"]).sum(dim=-1)
    torch.testing.assert_close(final_normal, initial_normal, atol=1.0e-10, rtol=0.0)
    assert bool(((result.direction * problem["direction"]).sum(dim=-1) > 0.0).all())


def test_zero_per_view_confidence_keeps_direction_unchanged() -> None:
    problem = _problem()
    problem["per_view_weights"].zero_()
    result = refit_tangent_angles(**problem, config=_config(ba_iterations=20, ba_patience=3))

    torch.testing.assert_close(result.direction, problem["direction"])
    torch.testing.assert_close(
        result.tangent_coordinate,
        torch.zeros_like(result.tangent_coordinate),
    )


def test_automatic_loop_accepts_data_improvement_and_then_stops(monkeypatch) -> None:
    problem = _problem(angle_deg=20.0)

    def identity_propagation(**kwargs):
        return {"direction": kwargs["direction"], "report": {"mode": "identity-test"}}

    monkeypatch.setattr(
        module,
        "refine_confidence_guided_directed_flow",
        identity_propagation,
    )
    result = run_post_v8_refinement(
        **problem,
        observed=torch.ones((2,), dtype=torch.bool),
        edge_u=torch.tensor([0], dtype=torch.long),
        edge_v=torch.tensor([1], dtype=torch.long),
        field_confidence=torch.ones((2,), dtype=DTYPE),
        unary_normalized_margin=torch.ones((2,), dtype=DTYPE),
        unary_vote_coherence=torch.ones((2,), dtype=DTYPE),
        canonical_rank=torch.arange(2, dtype=torch.long),
        config=_config(),
    )

    assert result.report["accepted_cycle_count"] >= 1
    assert result.report["final_data_energy"] < result.report["cycles"][0][
        "ba_initial_data_energy"
    ]
    expected = F.normalize(problem["per_view_axes"][0], dim=-1)
    agreement = (result.direction * expected).sum(dim=-1).abs()
    assert bool((agreement > 0.99).all())

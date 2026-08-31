from __future__ import annotations

import numpy as np
import pytest
import torch

from anigroom.surface_interpolation import (
    SurfaceFieldInterpolator,
    SurfaceSupport,
    adaptive_wendland_c2_weights,
    interpolate_physical,
)


def test_wendland_c2_matches_hand_computed_row() -> None:
    distances = torch.tensor([[0.25, 0.50, 1.00, 1.50]], dtype=torch.float64)

    weights = adaptive_wendland_c2_weights(distances, active_neighbor_count=2)

    # phi(1/4) = 81/128, phi(1/2) = 3/16, so the normalized values are
    # (81/128) / (105/128) = 27/35 and (3/16) / (105/128) = 8/35.
    expected = torch.tensor([[27.0 / 35.0, 8.0 / 35.0, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(weights, expected, rtol=0.0, atol=1.0e-14)


def test_wendland_weights_are_finite_nonnegative_and_zero_at_boundary() -> None:
    distances = torch.tensor([[0.10, 0.20, 0.40, 0.70, 2.00]], dtype=torch.float32)

    weights = adaptive_wendland_c2_weights(distances, active_neighbor_count=2)

    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0.0).all())
    assert bool((weights[:, 2:] == 0.0).all())
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones((1,), dtype=weights.dtype),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_legacy_surface_interpolator_weights_remain_inverse_square() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    interpolator = SurfaceFieldInterpolator(
        vertices=vertices,
        faces=faces,
        source_points=torch.tensor(
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.0], [0.70, 0.10, 0.0]],
            dtype=torch.float32,
        ),
        source_face_ids=torch.zeros((3,), dtype=torch.long),
        neighbor_count=2,
        device="cpu",
    )
    query_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    query_face_ids = torch.zeros((1,), dtype=torch.long)
    support = SurfaceSupport(
        indices=torch.tensor([[0, 1]], dtype=torch.long),
        vertex_path_distances=torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0e-6, 1.0e-6, 1.0e-6]]],
            dtype=torch.float32,
        ),
        report={},
    )

    actual = interpolator.weights(query_points, query_face_ids, support)
    distances = interpolator.distances(query_points, query_face_ids, support)
    expected_raw = distances.clamp_min(1.0e-6).pow(-2.0)
    expected = expected_raw / expected_raw.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual[0, 0] < 1.0
    assert actual[0, 1] > 0.0


def test_wendland_reproduces_constants_and_stays_in_source_convex_hull() -> None:
    distances = torch.tensor(
        [[0.10, 0.20, 0.40, 0.80], [0.15, 0.35, 0.50, 0.90]],
        dtype=torch.float64,
    )
    support_indices = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    weights = adaptive_wendland_c2_weights(
        distances,
        active_neighbor_count=2,
        support_indices=support_indices,
    )

    constant = torch.full((4,), 3.75, dtype=torch.float64)
    constant_field = interpolate_physical(constant, support_indices, weights)
    torch.testing.assert_close(constant_field, torch.full((2,), 3.75, dtype=torch.float64))

    source_values = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    field = interpolate_physical(source_values, support_indices, weights)
    assert bool(torch.isfinite(field).all())
    assert bool((field >= source_values.min()).all())
    assert bool((field <= source_values.max()).all())
    assert bool((field > 0.0).all())


def test_wendland_weights_are_invariant_to_uniform_spatial_scale() -> None:
    distances = torch.tensor(
        [[0.10, 0.22, 0.46, 0.95], [0.07, 0.31, 0.52, 1.10]],
        dtype=torch.float64,
    )
    base = adaptive_wendland_c2_weights(distances, active_neighbor_count=2)
    scaled_small = adaptive_wendland_c2_weights(distances * 1.0e-3, active_neighbor_count=2)
    scaled_large = adaptive_wendland_c2_weights(distances * 37.0, active_neighbor_count=2)

    torch.testing.assert_close(scaled_small, base, rtol=1.0e-12, atol=1.0e-14)
    torch.testing.assert_close(scaled_large, base, rtol=1.0e-12, atol=1.0e-14)


def _scattered_distances(query_x: torch.Tensor) -> torch.Tensor:
    source_x = torch.tensor([-0.20, -1.00, 1.00, 3.00], dtype=query_x.dtype, device=query_x.device)
    return (query_x.reshape(-1, 1) - source_x).abs()


def _hard_truncated_inverse_square(
    distances: torch.Tensor,
    source_values: torch.Tensor,
    active_neighbor_count: int,
) -> torch.Tensor:
    selected = torch.argsort(distances, stable=True)[:active_neighbor_count]
    selected_distances = distances[selected]
    inverse_square = selected_distances.pow(-2.0)
    inverse_square = inverse_square / inverse_square.sum()
    return (source_values[selected] * inverse_square).sum()


def test_support_swap_is_continuous_when_boundary_values_differ() -> None:
    query_x = torch.tensor([[-1.0e-4], [0.0], [1.0e-4]], dtype=torch.float64)
    distances = torch.cat([_scattered_distances(query_x[index]) for index in range(3)], dim=0)
    support_indices = torch.arange(4, dtype=torch.long).expand(3, -1)
    source_values = torch.tensor([2.0, -5.0, 11.0, 100.0], dtype=torch.float64)

    weights = adaptive_wendland_c2_weights(
        distances,
        active_neighbor_count=2,
        support_indices=support_indices,
    )
    field = interpolate_physical(source_values, support_indices, weights)

    # The boundary source changes from source 2 on the left to source 1 on
    # the right. Both are exactly zero at their entering/leaving boundary.
    assert weights[0, 2] == 0.0
    assert weights[2, 1] == 0.0
    assert bool((weights[1, 1:3] == 0.0).all())
    assert abs(float(field[0] - field[1])) < 1.0e-6
    assert abs(float(field[2] - field[1])) < 1.0e-6

    hard_left = _hard_truncated_inverse_square(distances[0], source_values, 2)
    hard_right = _hard_truncated_inverse_square(distances[2], source_values, 2)
    assert abs(float(hard_left - hard_right)) > 0.10


def test_support_swap_has_finite_query_gradients_on_both_sides() -> None:
    source_values = torch.tensor([2.0, -5.0, 11.0, 100.0], dtype=torch.float64)
    support_indices = torch.arange(4, dtype=torch.long).reshape(1, -1)

    for query_value in (-1.0e-3, 1.0e-3):
        query_x = torch.tensor([[query_value]], dtype=torch.float64, requires_grad=True)
        distances = _scattered_distances(query_x)
        weights = adaptive_wendland_c2_weights(
            distances,
            active_neighbor_count=2,
            support_indices=support_indices,
        )
        field = interpolate_physical(source_values, support_indices, weights).sum()
        (gradient,) = torch.autograd.grad(field, query_x)

        assert field.requires_grad
        assert bool(torch.isfinite(gradient).all())


def test_invalid_distance_and_support_states_fail_explicitly() -> None:
    valid = torch.tensor([[0.10, 0.20, 0.40]], dtype=torch.float64)

    with pytest.raises(ValueError, match="shape"):
        adaptive_wendland_c2_weights(valid.reshape(-1), active_neighbor_count=2)
    with pytest.raises(TypeError, match="floating-point"):
        adaptive_wendland_c2_weights(valid.to(torch.int64), active_neighbor_count=2)
    with pytest.raises(ValueError, match="finite"):
        adaptive_wendland_c2_weights(
            torch.tensor([[0.10, float("nan"), 0.40]], dtype=torch.float64),
            active_neighbor_count=2,
        )
    with pytest.raises(ValueError, match="finite"):
        adaptive_wendland_c2_weights(
            torch.tensor([[0.10, float("inf"), 0.40]], dtype=torch.float64),
            active_neighbor_count=2,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        adaptive_wendland_c2_weights(
            torch.tensor([[0.10, -0.20, 0.40]], dtype=torch.float64),
            active_neighbor_count=2,
        )
    with pytest.raises(ValueError, match="integer"):
        adaptive_wendland_c2_weights(valid, active_neighbor_count=2.0)
    with pytest.raises(ValueError, match="positive"):
        adaptive_wendland_c2_weights(valid, active_neighbor_count=0)
    with pytest.raises(ValueError, match="positive"):
        adaptive_wendland_c2_weights(valid, active_neighbor_count=-1)
    with pytest.raises(ValueError, match="integer"):
        adaptive_wendland_c2_weights(valid, active_neighbor_count=True)
    with pytest.raises(ValueError, match=r"K \+ 1"):
        adaptive_wendland_c2_weights(
            torch.tensor([[0.10, 0.20]], dtype=torch.float64),
            active_neighbor_count=2,
        )
    with pytest.raises(ValueError, match="shape"):
        adaptive_wendland_c2_weights(
            valid,
            active_neighbor_count=2,
            support_indices=torch.zeros((1, 2), dtype=torch.long),
        )
    with pytest.raises(TypeError, match="integer"):
        adaptive_wendland_c2_weights(
            valid,
            active_neighbor_count=2,
            support_indices=torch.zeros((1, 3), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="unique"):
        adaptive_wendland_c2_weights(
            valid,
            active_neighbor_count=2,
            support_indices=torch.tensor([[0, 1, 1]], dtype=torch.long),
        )
    with pytest.raises(ValueError, match="radius"):
        adaptive_wendland_c2_weights(
            torch.zeros((1, 3), dtype=torch.float64),
            active_neighbor_count=2,
        )
    with pytest.raises(RuntimeError, match="denominator"):
        adaptive_wendland_c2_weights(
            torch.ones((1, 3), dtype=torch.float64),
            active_neighbor_count=2,
        )


def _disconnected_sheet_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [1.0, 1.0, 0.01],
            [0.0, 1.0, 0.01],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
        dtype=np.int64,
    )
    source_points = torch.tensor(
        [
            [0.15, 0.10, 0.0],
            [0.75, 0.15, 0.0],
            [0.85, 0.70, 0.0],
            [0.15, 0.75, 0.0],
            [0.15, 0.10, 0.01],
            [0.75, 0.15, 0.01],
            [0.85, 0.70, 0.01],
            [0.15, 0.75, 0.01],
        ],
        dtype=torch.float32,
    )
    source_face_ids = torch.tensor([0, 0, 0, 1, 2, 2, 2, 3], dtype=torch.long)
    query_points = torch.tensor(
        [
            [0.30, 0.20, 0.0],
            [0.70, 0.60, 0.0],
            [0.30, 0.20, 0.01],
            [0.70, 0.60, 0.01],
        ],
        dtype=torch.float32,
    )
    query_face_ids = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    query_sheet = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return (
        vertices,
        faces,
        source_points,
        source_face_ids,
        query_points,
        query_face_ids,
        query_sheet,
    )


def test_surface_interpolator_support_stays_on_disconnected_sheets() -> None:
    (
        vertices,
        faces,
        source_points,
        source_face_ids,
        query_points,
        query_face_ids,
        query_sheet,
    ) = _disconnected_sheet_fixture()
    interpolator = SurfaceFieldInterpolator(
        vertices=vertices,
        faces=faces,
        source_points=source_points,
        source_face_ids=source_face_ids,
        neighbor_count=3,
        device="cpu",
    )

    support = interpolator.build_support(query_points, query_face_ids)
    distances = interpolator.distances(query_points, query_face_ids, support)
    weights = adaptive_wendland_c2_weights(
        distances,
        active_neighbor_count=2,
        support_indices=support.indices,
    )
    source_sheet = torch.arange(source_points.shape[0]) // 4

    assert bool((source_sheet[support.indices] == query_sheet[:, None]).all())
    assert bool(torch.isfinite(weights).all())
    assert bool((weights >= 0.0).all())
    assert bool((weights.sum(dim=1) > 0.0).all())
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones((query_points.shape[0],), dtype=weights.dtype),
        rtol=0.0,
        atol=1.0e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_wendland_numerical_parity() -> None:
    distances = torch.tensor(
        [[0.05, 0.12, 0.25, 0.60, 1.20], [0.03, 0.20, 0.21, 0.80, 1.00]],
        dtype=torch.float64,
    )
    support_indices = torch.arange(5, dtype=torch.long).expand(2, -1)

    cpu_weights = adaptive_wendland_c2_weights(
        distances,
        active_neighbor_count=3,
        support_indices=support_indices,
    )
    cuda_weights = adaptive_wendland_c2_weights(
        distances.cuda(),
        active_neighbor_count=3,
        support_indices=support_indices.cuda(),
    )

    torch.testing.assert_close(cuda_weights.cpu(), cpu_weights, rtol=1.0e-12, atol=1.0e-14)

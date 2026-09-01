from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from anigroom.grooming.guide_attribute_gaussian_field import (
    GuideAttributeGaussianField,
    GuideGaussianFieldConfig,
    c2_gaussian_taper,
    initialize_guide_gaussian_binding,
)


def _config(**overrides: object) -> GuideGaussianFieldConfig:
    values: dict[str, object] = {
        "neighbor_count": 1,
        "support_sigma": 3.0,
        "taper_start_sigma": 2.5,
        "min_scale_ratio": 2.0 / 3.0,
        "max_scale_ratio": 1.5,
        "min_denominator": 1.0e-8,
    }
    values.update(overrides)
    return GuideGaussianFieldConfig(**values)


def _boundary_geometry() -> tuple[np.ndarray, np.ndarray]:
    guide_points = np.asarray(
        [[-4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    query_points = np.asarray(
        [
            [-3.99, 0.0, 0.0],
            [-4.0, 0.0, 0.0],
            [-4.01, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return guide_points, query_points


def _make_binding(
    *,
    guide_points: np.ndarray | None = None,
    query_points: np.ndarray | None = None,
    config: GuideGaussianFieldConfig | None = None,
) -> object:
    if guide_points is None or query_points is None:
        guide_points, query_points = _boundary_geometry()
    return initialize_guide_gaussian_binding(
        guide_points,
        query_points,
        _config() if config is None else config,
        device="cpu",
        dtype=torch.float64,
    )


def test_config_boundaries_and_json_are_strict() -> None:
    config = _config(
        neighbor_count=1,
        support_sigma=4.0,
        taper_start_sigma=1.0e-12,
        min_scale_ratio=1.0,
        max_scale_ratio=1.0,
        min_denominator=1.0e-15,
    )
    json_dict = config.to_json_dict()
    assert json.loads(json.dumps(json_dict)) == json_dict
    assert json_dict == {
        "neighbor_count": 1,
        "support_sigma": 4.0,
        "taper_start_sigma": 1.0e-12,
        "min_scale_ratio": 1.0,
        "max_scale_ratio": 1.0,
        "min_denominator": 1.0e-15,
    }


@pytest.mark.parametrize(
    ("field_name", "value", "exception", "message"),
    [
        ("neighbor_count", 0, ValueError, "at least one"),
        ("neighbor_count", True, TypeError, "integer"),
        ("neighbor_count", 1.5, TypeError, "integer"),
        ("support_sigma", 0.0, ValueError, "positive"),
        ("support_sigma", float("nan"), ValueError, "finite"),
        ("taper_start_sigma", 0.0, ValueError, "0 < taper_start_sigma"),
        ("taper_start_sigma", 3.0, ValueError, "0 < taper_start_sigma"),
        ("min_scale_ratio", 0.0, ValueError, "scale ratios"),
        ("max_scale_ratio", 0.9, ValueError, "scale ratios"),
        ("min_denominator", 0.0, ValueError, "positive"),
        ("min_denominator", float("inf"), ValueError, "finite"),
    ],
)
def test_config_rejects_invalid_boundaries(
    field_name: str,
    value: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        _config(**{field_name: value})


def test_binding_reference_sigma_csr_and_determinism() -> None:
    guide_points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    query_points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    config = _config(neighbor_count=2, support_sigma=2.0, taper_start_sigma=1.5)
    first = _make_binding(
        guide_points=guide_points,
        query_points=query_points,
        config=config,
    )
    second = _make_binding(
        guide_points=guide_points,
        query_points=query_points,
        config=config,
    )

    expected_spacing = np.asarray([3.0, 2.0, 3.0, 4.0, 7.0], dtype=np.float64)
    torch.testing.assert_close(
        first.reference_sigma,
        torch.as_tensor(expected_spacing / 2.0, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    query_ids = first.query_ids.cpu().numpy()
    guide_ids = first.guide_ids.cpu().numpy()
    row_ptr = first.row_ptr.cpu().numpy()
    assert np.array_equal(
        query_ids,
        np.repeat(np.arange(query_points.shape[0]), np.diff(row_ptr)),
    )
    for query_id in range(query_points.shape[0]):
        row_guides = guide_ids[row_ptr[query_id] : row_ptr[query_id + 1]]
        assert np.all(row_guides[:-1] < row_guides[1:])
    assert len(set(zip(query_ids.tolist(), guide_ids.tolist()))) == len(query_ids)
    assert first.report["all_queries_covered"] is True
    assert first.report["stable_pair_order"] is True

    for name in (
        "guide_points",
        "query_points",
        "reference_sigma",
        "row_ptr",
        "guide_ids",
        "query_ids",
    ):
        torch.testing.assert_close(
            getattr(first, name),
            getattr(second, name),
            rtol=0.0,
            atol=0.0,
        )
    assert first.report == second.report


@pytest.mark.parametrize(
    ("guide_points", "query_points", "exception", "message"),
    [
        (
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            ValueError,
            "duplicate",
        ),
        (
            np.empty((0, 3), dtype=np.float64),
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            ValueError,
            "must not be empty",
        ),
        (
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            ValueError,
            "must not be empty",
        ),
        (
            np.asarray([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            ValueError,
            "non-finite",
        ),
        (
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray([[np.inf, 0.0, 0.0]], dtype=np.float64),
            ValueError,
            "non-finite",
        ),
    ],
)
def test_binding_rejects_bad_geometry(
    guide_points: np.ndarray,
    query_points: np.ndarray,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        initialize_guide_gaussian_binding(
            guide_points,
            query_points,
            _config(),
            device="cpu",
            dtype=torch.float64,
        )


def test_binding_rejects_insufficient_guides_and_uncovered_query() -> None:
    with pytest.raises(ValueError, match="more points than neighbor_count"):
        initialize_guide_gaussian_binding(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            _config(neighbor_count=2),
            device="cpu",
            dtype=torch.float64,
        )

    guide_points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    query_points = np.asarray(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    with pytest.raises(RuntimeError, match="queries uncovered"):
        initialize_guide_gaussian_binding(
            guide_points,
            query_points,
            _config(max_scale_ratio=1.0),
            device="cpu",
            dtype=torch.float64,
        )


def test_duplicate_query_points_are_distinct_rows_with_identical_weights() -> None:
    guide_points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    query_points = np.asarray(
        [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        dtype=np.float64,
    )
    binding = initialize_guide_gaussian_binding(
        guide_points,
        query_points,
        _config(),
        device="cpu",
        dtype=torch.float64,
    )
    field = GuideAttributeGaussianField(binding)
    values = torch.tensor([0.2, 0.7, 1.1], dtype=torch.float64)
    output = field(values)
    torch.testing.assert_close(output[0], output[1], rtol=0.0, atol=0.0)
    first_ids = binding.guide_ids[binding.row_ptr[0] : binding.row_ptr[1]]
    second_ids = binding.guide_ids[binding.row_ptr[1] : binding.row_ptr[2]]
    torch.testing.assert_close(first_ids, second_ids, rtol=0.0, atol=0.0)


def test_field_uses_buffers_only_for_geometry_and_has_spd_initial_covariance() -> None:
    binding = _make_binding()
    field = GuideAttributeGaussianField(binding)

    assert field.guide_points.device.type == "cpu"
    assert set(dict(field.named_parameters())) == {
        "raw_scale_coordinate",
        "raw_quaternion",
    }
    assert set(dict(field.named_buffers())) == {
        "guide_points",
        "query_points",
        "reference_sigma",
        "row_ptr",
        "guide_ids",
        "query_ids",
    }
    for name, buffer in field.named_buffers():
        assert buffer.requires_grad is False, name

    torch.testing.assert_close(
        field.decoded_scales(),
        binding.reference_sigma[:, None].expand(-1, 3),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        field.rotation_matrices(),
        torch.eye(3, dtype=torch.float64).expand(field.guide_count, 3, 3),
        rtol=0.0,
        atol=0.0,
    )
    covariance = field.covariance_matrices()
    torch.testing.assert_close(covariance, covariance.transpose(-1, -2))
    assert bool((torch.linalg.eigvalsh(covariance) > 0.0).all())


def test_explicit_field_config_must_match_binding_config() -> None:
    binding_config = _config()
    binding = _make_binding(config=binding_config)
    with pytest.raises(ValueError, match="exactly match"):
        GuideAttributeGaussianField(
            binding,
            replace(binding_config, support_sigma=4.0),
        )


def test_raw_scale_coordinates_stay_bounded_with_finite_gradients() -> None:
    field = GuideAttributeGaussianField(_make_binding())
    for coordinate in (100.0, -100.0):
        with torch.no_grad():
            field.raw_scale_coordinate.fill_(coordinate)
        scales = field.decoded_scales()
        ratios = scales / field.reference_sigma[:, None]
        assert bool((ratios >= field.config.min_scale_ratio).all())
        assert bool((ratios <= field.config.max_scale_ratio).all())
        assert bool(torch.isfinite(scales).all())
        gradient = torch.autograd.grad(scales.sum(), field.raw_scale_coordinate)[0]
        assert bool(torch.isfinite(gradient).all())


def test_c2_taper_values_and_first_second_derivatives_are_continuous() -> None:
    start, end = 2.5, 3.0
    epsilon = 1.0e-7
    rho = torch.tensor(
        [start - epsilon, start, start + epsilon, end - epsilon, end, end + epsilon],
        dtype=torch.float64,
        requires_grad=True,
    )
    values = c2_gaussian_taper(rho, start=start, end=end)
    first = torch.autograd.grad(values.sum(), rho, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), rho)[0]

    torch.testing.assert_close(
        values,
        torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        values[[1, 4]],
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    for derivative in (first, second):
        assert bool(torch.isfinite(derivative).all())
        torch.testing.assert_close(derivative[[1, 4]], torch.zeros(2, dtype=torch.float64), atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(first[[0, 2]], torch.zeros(2, dtype=torch.float64), atol=1.0e-9, rtol=0.0)
    torch.testing.assert_close(first[[3, 5]], torch.zeros(2, dtype=torch.float64), atol=1.0e-9, rtol=0.0)
    torch.testing.assert_close(second[[0, 2]], torch.zeros(2, dtype=torch.float64), atol=1.0e-4, rtol=0.0)
    torch.testing.assert_close(second[[3, 5]], torch.zeros(2, dtype=torch.float64), atol=1.0e-4, rtol=0.0)

    outside = c2_gaussian_taper(
        torch.tensor([end, end + 1.0, 100.0], dtype=torch.float64),
        start=start,
        end=end,
    )
    torch.testing.assert_close(outside, torch.zeros_like(outside), rtol=0.0, atol=0.0)


def test_c2_taper_is_nonnegative_and_monotone_in_float32() -> None:
    rho = torch.linspace(2.5, 3.0, 10001, dtype=torch.float32)
    values = c2_gaussian_taper(rho)
    assert bool(torch.isfinite(values).all())
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0
    assert bool((torch.diff(values) <= 0.0).all())


def test_weights_are_finite_normalized_and_channelwise() -> None:
    field = GuideAttributeGaussianField(_make_binding())
    weights = field.evaluate_weights()
    assert bool(torch.isfinite(weights.raw).all())
    assert bool(torch.isfinite(weights.normalized).all())
    assert bool((weights.raw >= 0.0).all())
    assert bool((weights.normalized >= 0.0).all())
    row_sums = torch.zeros(field.query_count, dtype=torch.float64)
    row_sums.index_add_(0, weights.query_ids, weights.normalized)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=0.0, atol=1.0e-12)

    constant = torch.full((field.guide_count,), 2.75, dtype=torch.float64)
    constant_output = field(constant)
    torch.testing.assert_close(constant_output, torch.full((field.query_count,), 2.75, dtype=torch.float64), rtol=0.0, atol=1.0e-12)

    multi_channel = torch.arange(field.guide_count * 3, dtype=torch.float64).reshape(field.guide_count, 3)
    multi_output = field(multi_channel)
    for channel in range(3):
        torch.testing.assert_close(multi_output[:, channel], field(multi_channel[:, channel]), rtol=0.0, atol=1.0e-12)


def test_support_boundary_value_and_query_gradient_are_continuous() -> None:
    binding = _make_binding()
    field = GuideAttributeGaussianField(binding)
    guide_values = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    row_guides = binding.guide_ids[binding.row_ptr[0] : binding.row_ptr[1]].tolist()
    assert 0 in row_guides and 1 in row_guides

    def value_and_gradient(x: float) -> tuple[torch.Tensor, torch.Tensor]:
        query = binding.query_points.detach().clone()
        query[0, 0] = x
        query.requires_grad_(True)
        output = field(guide_values, query)
        gradient = torch.autograd.grad(output[0], query)[0][0, 0]
        return output[0], gradient

    inside, inside_gradient = value_and_gradient(-4.0 + 1.0e-4)
    boundary, boundary_gradient = value_and_gradient(-4.0)
    outside, outside_gradient = value_and_gradient(-4.0 - 1.0e-4)
    torch.testing.assert_close(inside, boundary, rtol=0.0, atol=1.0e-8)
    torch.testing.assert_close(outside, boundary, rtol=0.0, atol=1.0e-8)
    torch.testing.assert_close(inside_gradient, boundary_gradient, rtol=0.0, atol=1.0e-5)
    torch.testing.assert_close(outside_gradient, boundary_gradient, rtol=0.0, atol=1.0e-5)


def test_exact_guide_query_overlap_preserves_values_and_backward_finiteness() -> None:
    guide_points = np.asarray(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    binding = _make_binding(
        guide_points=guide_points,
        query_points=guide_points.copy(),
        config=_config(max_scale_ratio=1.0),
    )
    field = GuideAttributeGaussianField(binding)
    guide_values = torch.tensor([0.2, 0.4, 0.8, 1.6], dtype=torch.float64, requires_grad=True)
    query_points = binding.query_points.detach().clone().requires_grad_(True)
    output = field(guide_values, query_points)
    torch.testing.assert_close(output, guide_values, rtol=0.0, atol=1.0e-12)

    output.square().sum().backward()
    for tensor in (
        guide_values,
        query_points,
        field.raw_scale_coordinate,
        field.raw_quaternion,
    ):
        assert tensor.grad is not None
        assert bool(torch.isfinite(tensor.grad).all())


def test_anisotropic_scale_produces_nonzero_finite_quaternion_gradient() -> None:
    binding = _make_binding()
    field = GuideAttributeGaussianField(binding)
    with torch.no_grad():
        field.raw_scale_coordinate[0].copy_(torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64))
        field.raw_quaternion[0].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    query_points = binding.query_points.detach().clone()
    query_points[4] = torch.tensor([0.5, 0.7, 0.2], dtype=torch.float64)
    weights = field.evaluate_weights(query_points)
    pair = (weights.query_ids == 4) & (weights.guide_ids == 0)
    assert int(pair.sum()) == 1
    assert float(weights.raw[pair]) > 0.0
    weights.raw[pair].sum().backward()
    quaternion_gradient = field.raw_quaternion.grad[0]
    assert bool(torch.isfinite(quaternion_gradient).all())
    assert float(torch.linalg.vector_norm(quaternion_gradient)) > 1.0e-8


def test_dynamic_query_keeps_gradient_and_does_not_mutate_fixed_buffers() -> None:
    field = GuideAttributeGaussianField(_make_binding())
    before = {name: buffer.detach().clone() for name, buffer in field.named_buffers()}
    query_points = field.query_points.detach().clone().requires_grad_(True)
    guide_values = torch.arange(field.guide_count, dtype=torch.float64)
    output = field(guide_values, query_points)
    output.square().sum().backward()

    assert query_points.grad is not None
    assert bool(torch.isfinite(query_points.grad).all())
    for name, buffer in field.named_buffers():
        assert torch.equal(buffer, before[name]), name

from __future__ import annotations

import pytest
import torch

from anigroom import rbf_partition_of_unity as rbf_core
from anigroom.rbf_partition_of_unity import (
    IllConditionedRBFSystemError,
    PartitionCoverageError,
    SingularRBFSystemError,
    blend_partition_of_unity,
    build_augmented_system,
    evaluate_local_interpolant,
    local_cardinal_weights,
    local_kernel_values,
    normalize_partition_of_unity_weights,
    raw_partition_of_unity_weights,
    solve_augmented_system,
    validate_augmented_system,
    wendland_c2,
)


def _deterministic_patch(
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    count: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260904)
    points = (torch.rand((count, 3), generator=generator, dtype=dtype) - 0.5) * 0.8
    points = points.to(device=device)
    radius = torch.tensor(1.0, dtype=dtype, device=device)
    return points, radius


def test_wendland_c2_matches_hand_values_and_is_flat_zero_at_boundary() -> None:
    t = torch.tensor([0.0, 0.25, 0.5, 1.0, 1.25], dtype=torch.float64)
    actual = wendland_c2(t)
    expected = torch.tensor(
        [1.0, 81.0 / 128.0, 3.0 / 16.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert bool((actual[t >= 1.0] == 0.0).all())

    boundary = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    value = wendland_c2(boundary)
    (gradient,) = torch.autograd.grad(value, boundary)
    assert value == 0.0
    assert gradient == 0.0


def test_raw_partition_weights_use_distinct_patch_radii_columnwise() -> None:
    distances = torch.tensor(
        [[0.5, 0.5], [0.75, 1.5], [1.0, 2.0]],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.0, 2.0], dtype=torch.float64)
    actual = raw_partition_of_unity_weights(distances, radii)
    expected = torch.tensor(
        [
            [3.0 / 16.0, 81.0 / 128.0],
            [1.0 / 64.0, 1.0 / 64.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_random_3d_patch_interpolates_every_local_node_exactly() -> None:
    points, radius = _deterministic_patch()
    values = torch.linspace(-1.5, 2.0, points.shape[0], dtype=torch.float64)
    system = build_augmented_system(points, radius)
    report = validate_augmented_system(system)
    solution = solve_augmented_system(system, values)
    actual = evaluate_local_interpolant(points, points, radius, solution)

    assert report.rank == points.shape[0] + 1
    assert report.full_rank is True
    assert report.within_condition_limit is True
    torch.testing.assert_close(actual, values, rtol=0.0, atol=1.0e-10)


def test_float32_local_solve_and_multichannel_evaluation_are_supported() -> None:
    points, radius = _deterministic_patch(dtype=torch.float32, count=6)
    values = torch.stack(
        (
            torch.linspace(-1.0, 1.0, points.shape[0]),
            torch.linspace(2.0, 3.0, points.shape[0]),
        ),
        dim=1,
    ).to(dtype=torch.float32)
    system = build_augmented_system(points, radius)
    solution = solve_augmented_system(system, values)
    actual = evaluate_local_interpolant(points, points, radius, solution)
    torch.testing.assert_close(actual, values, rtol=0.0, atol=2.0e-5)


def test_constant_reproduction_inside_and_outside_source_convex_hull() -> None:
    points, radius = _deterministic_patch(count=7)
    constant = 3.75
    values = torch.full((points.shape[0],), constant, dtype=torch.float64)
    system = build_augmented_system(points, radius)
    solution = solve_augmented_system(system, values)
    queries = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, -0.1, 0.15],
            [3.0, -2.5, 4.0],
            [-5.0, 1.5, -2.0],
        ],
        dtype=torch.float64,
    )
    actual = evaluate_local_interpolant(queries, points, radius, solution)
    torch.testing.assert_close(
        actual,
        torch.full((queries.shape[0],), constant, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-10,
    )


def test_batched_multichannel_values_and_autograd_match_finite_difference() -> None:
    points, radius = _deterministic_patch(count=6)
    system = build_augmented_system(points, radius)
    values = torch.linspace(
        -0.7,
        1.4,
        2 * points.shape[0] * 3,
        dtype=torch.float64,
    ).reshape(2, points.shape[0], 3)
    values.requires_grad_()
    query = torch.tensor(
        [[0.13, -0.08, 0.17]],
        dtype=torch.float64,
        requires_grad=True,
    )
    solution = solve_augmented_system(system, values)
    output = evaluate_local_interpolant(query, points, radius, solution)
    assert output.shape == (2, 1, 3)
    loss = output.square().sum()
    value_gradient, query_gradient = torch.autograd.grad(loss, (values, query))
    assert bool(torch.isfinite(value_gradient).all())
    assert bool(torch.isfinite(query_gradient).all())

    scalar_values = torch.linspace(-0.4, 0.9, points.shape[0], dtype=torch.float64)
    scalar_solution = solve_augmented_system(system, scalar_values)
    finite_difference_epsilon = 1.0e-6
    plus = query.detach().clone()
    minus = query.detach().clone()
    plus[0, 0] += finite_difference_epsilon
    minus[0, 0] -= finite_difference_epsilon
    finite_difference = (
        evaluate_local_interpolant(plus, points, radius, scalar_solution)
        - evaluate_local_interpolant(minus, points, radius, scalar_solution)
    ) / (2.0 * finite_difference_epsilon)
    query_for_gradient = query.detach().clone().requires_grad_(True)
    evaluated = evaluate_local_interpolant(
        query_for_gradient,
        points,
        radius,
        scalar_solution,
    ).sum()
    (analytic_gradient,) = torch.autograd.grad(evaluated, query_for_gradient)
    torch.testing.assert_close(
        analytic_gradient[0, 0],
        finite_difference[0],
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_cardinal_weights_sum_one_reproduce_direct_solve_and_value_gradient() -> None:
    points, radius = _deterministic_patch(count=7)
    system = build_augmented_system(points, radius)
    queries = torch.tensor(
        [[0.1, 0.2, -0.1], [2.5, -1.0, 0.75]],
        dtype=torch.float64,
    )
    cardinal = local_cardinal_weights(
        queries,
        points,
        radius,
        augmented_system=system,
    )
    torch.testing.assert_close(
        cardinal.sum(dim=1),
        torch.ones((queries.shape[0],), dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-10,
    )
    values = torch.linspace(-2.0, 1.0, points.shape[0], dtype=torch.float64)
    direct = evaluate_local_interpolant(
        queries,
        points,
        radius,
        solve_augmented_system(system, values),
    )
    torch.testing.assert_close(cardinal @ values, direct, rtol=0.0, atol=1.0e-10)

    differentiable_values = values.clone().requires_grad_(True)
    output = evaluate_local_interpolant(
        queries[:1],
        points,
        radius,
        solve_augmented_system(system, differentiable_values),
    ).sum()
    (gradient,) = torch.autograd.grad(output, differentiable_values)
    torch.testing.assert_close(gradient, cardinal[0], rtol=0.0, atol=1.0e-10)


def test_partition_of_unity_shared_site_exactness_fixture() -> None:
    dtype = torch.float64
    radius = torch.tensor(1.0, dtype=dtype)
    shared = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
    patch_a = torch.cat(
        (
            shared,
            torch.tensor(
                [[0.35, 0.05, 0.1], [-0.2, 0.3, 0.05], [0.1, -0.25, 0.3]],
                dtype=dtype,
            ),
        ),
        dim=0,
    )
    patch_b = torch.cat(
        (
            shared,
            torch.tensor(
                [[-0.3, -0.1, 0.2], [0.25, 0.25, -0.1], [0.05, -0.35, -0.2]],
                dtype=dtype,
            ),
        ),
        dim=0,
    )
    values_a = torch.tensor([5.0, 1.0, -2.0, 4.0], dtype=dtype)
    values_b = torch.tensor([5.0, 20.0, -7.0, 3.0], dtype=dtype)
    local_a = evaluate_local_interpolant(
        shared,
        patch_a,
        radius,
        solve_augmented_system(build_augmented_system(patch_a, radius), values_a),
    )
    local_b = evaluate_local_interpolant(
        shared,
        patch_b,
        radius,
        solve_augmented_system(build_augmented_system(patch_b, radius), values_b),
    )
    local_values = torch.stack((local_a, local_b), dim=1)
    raw_weights = raw_partition_of_unity_weights(
        torch.tensor([[0.2, 0.65]], dtype=dtype),
        torch.tensor([1.0, 0.8], dtype=dtype),
    )
    global_value = blend_partition_of_unity(local_values, raw_weights)
    torch.testing.assert_close(global_value, torch.tensor([5.0], dtype=dtype), atol=1.0e-10, rtol=0.0)


def test_patch_entry_exit_is_continuous_and_zero_denominator_is_refused() -> None:
    dtype = torch.float64
    radius = torch.tensor(1.0, dtype=dtype)
    epsilon = 1.0e-4
    topology_distance = torch.tensor(
        [[0.0, 1.0 - epsilon], [0.0, 1.0], [0.0, 1.0 + epsilon]],
        dtype=dtype,
    )
    raw = raw_partition_of_unity_weights(
        topology_distance,
        torch.tensor([1.0, 1.0], dtype=dtype),
    )
    assert raw[0, 1] > 0.0
    assert raw[1, 1] == 0.0
    assert raw[2, 1] == 0.0
    local_values = torch.tensor(
        [[2.0, 10.0], [2.0, 10.0], [2.0, 10.0]],
        dtype=dtype,
    )
    blended = blend_partition_of_unity(local_values, raw)
    assert abs(float(blended[0] - blended[1])) < 1.0e-12
    assert blended[1] == 2.0
    assert blended[2] == 2.0

    with pytest.raises(PartitionCoverageError, match="zero denominator"):
        normalize_partition_of_unity_weights(torch.zeros((2, 3), dtype=dtype))
    with pytest.raises(PartitionCoverageError, match="zero denominator"):
        blend_partition_of_unity(
            torch.ones((1, 2), dtype=dtype),
            torch.zeros((1, 2), dtype=dtype),
        )


def test_folded_sheet_patch_excludes_topologically_opposite_sources() -> None:
    dtype = torch.float64
    sheet_zero = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.3, 0.3, 0.0]],
        dtype=dtype,
    )
    opposite_sheet = sheet_zero + torch.tensor([0.0, 0.0, 0.01], dtype=dtype)
    global_positions = torch.cat((sheet_zero, opposite_sheet), dim=0)
    global_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0], dtype=dtype)
    topology_certified_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    local_sources = global_positions[topology_certified_ids]
    local_values = global_values[topology_certified_ids]
    radius = torch.tensor(1.0, dtype=dtype)
    solution = solve_augmented_system(
        build_augmented_system(local_sources, radius),
        local_values,
    )
    actual = evaluate_local_interpolant(
        sheet_zero[:1],
        local_sources,
        radius,
        solution,
    )
    assert not bool(torch.isin(topology_certified_ids, torch.arange(4, 8)).any())
    torch.testing.assert_close(actual, torch.tensor([1.0], dtype=dtype), rtol=0.0, atol=1.0e-10)


def test_solution_and_precomputed_cardinal_system_reject_patch_mismatch() -> None:
    points, radius = _deterministic_patch(count=6)
    values = torch.linspace(-1.0, 1.0, points.shape[0], dtype=torch.float64)
    system = build_augmented_system(points, radius)
    solution = solve_augmented_system(system, values)
    assert solution.augmented_system is system
    query = torch.tensor([[0.1, 0.0, -0.1]], dtype=torch.float64)
    changed_geometry = points.clone()
    changed_geometry[0, 0] += 0.125
    changed_radius = torch.tensor(1.25, dtype=torch.float64)

    with pytest.raises(ValueError, match="does not match"):
        evaluate_local_interpolant(
            query,
            changed_geometry,
            radius,
            solution,
        )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_local_interpolant(
            query,
            points,
            changed_radius,
            solution,
        )
    with pytest.raises(ValueError, match="does not match"):
        local_cardinal_weights(
            query,
            changed_geometry,
            radius,
            augmented_system=system,
        )
    with pytest.raises(ValueError, match="does not match"):
        local_cardinal_weights(
            query,
            points,
            changed_radius,
            augmented_system=system,
        )


def test_strict_singular_ill_conditioned_nonfinite_shape_type_and_radius_failures() -> None:
    points, radius = _deterministic_patch(count=5)
    duplicate = points.clone()
    duplicate[1] = duplicate[0]
    with pytest.raises(SingularRBFSystemError):
        build_augmented_system(duplicate, radius)

    system = build_augmented_system(points, radius)
    with pytest.raises(IllConditionedRBFSystemError):
        validate_augmented_system(system, max_condition_number=1.0)
    nonfinite_system = system.clone()
    nonfinite_system[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        validate_augmented_system(nonfinite_system)
    with pytest.raises(ValueError, match="finite"):
        build_augmented_system(
            torch.tensor([[0.0, 0.0, float("nan")]], dtype=torch.float64),
            radius,
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        build_augmented_system(points, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        build_augmented_system(points, torch.tensor(0.0, dtype=torch.float64))
    with pytest.raises(ValueError, match="scalar"):
        build_augmented_system(points, torch.ones((1,), dtype=torch.float64))
    with pytest.raises(TypeError, match="float32 or torch.float64"):
        build_augmented_system(points.to(torch.float16), radius.to(torch.float16))
    with pytest.raises(ValueError, match="shape"):
        build_augmented_system(points[:, :2], radius)
    with pytest.raises(TypeError, match="dtype"):
        solve_augmented_system(system, torch.ones((5,), dtype=torch.float32))
    with pytest.raises(ValueError, match="shape"):
        solve_augmented_system(system, torch.ones((4,), dtype=torch.float64))
    with pytest.raises(ValueError, match="nonnegative"):
        raw_partition_of_unity_weights(
            torch.tensor([[0.0, -0.1]], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match=r"shape \[P\]"):
        raw_partition_of_unity_weights(
            torch.zeros((2, 3), dtype=torch.float64),
            torch.tensor(1.0, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match=r"matching P=3"):
        raw_partition_of_unity_weights(
            torch.zeros((2, 3), dtype=torch.float64),
            torch.ones((2,), dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="finite"):
        raw_partition_of_unity_weights(
            torch.zeros((1, 2), dtype=torch.float64),
            torch.tensor([1.0, float("nan")], dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="positive"):
        raw_partition_of_unity_weights(
            torch.zeros((1, 2), dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
    with pytest.raises(TypeError, match="dtype"):
        raw_partition_of_unity_weights(
            torch.zeros((1, 2), dtype=torch.float64),
            torch.ones((2,), dtype=torch.float32),
        )
    with pytest.raises(ValueError, match="matching"):
        blend_partition_of_unity(
            torch.ones((2, 3), dtype=torch.float64),
            torch.ones((2, 2), dtype=torch.float64),
        )


def test_torch_out_of_memory_is_never_wrapped_as_rbf_domain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points, radius = _deterministic_patch(count=5)
    system = build_augmented_system(points, radius)
    values = torch.ones((5,), dtype=torch.float64)

    def raise_oom(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise torch.OutOfMemoryError("injected OOM")

    with monkeypatch.context() as context:
        context.setattr(rbf_core.torch.linalg, "matrix_rank", raise_oom)
        with pytest.raises(torch.OutOfMemoryError, match="injected OOM"):
            validate_augmented_system(system)
    with monkeypatch.context() as context:
        context.setattr(rbf_core.torch.linalg, "solve", raise_oom)
        with pytest.raises(torch.OutOfMemoryError, match="injected OOM"):
            solve_augmented_system(system, values)
        with pytest.raises(torch.OutOfMemoryError, match="injected OOM"):
            local_cardinal_weights(
                points,
                points,
                radius,
                augmented_system=system,
            )


def test_torch_backend_runtime_error_propagates_without_domain_relabel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points, radius = _deterministic_patch(count=5)
    system = build_augmented_system(points, radius)

    def raise_backend(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("injected backend synchronization failure")

    monkeypatch.setattr(rbf_core.torch.linalg, "solve", raise_backend)
    with pytest.raises(RuntimeError, match="backend synchronization"):
        solve_augmented_system(system, torch.ones((5,), dtype=torch.float64))
    with pytest.raises(RuntimeError, match="backend synchronization"):
        local_cardinal_weights(
            points,
            points,
            radius,
            augmented_system=system,
        )


def test_device_mismatch_fails_when_cuda_exists() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    points, _ = _deterministic_patch()
    cuda_radius = torch.tensor(1.0, dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="device"):
        build_augmented_system(points, cuda_radius)
    with pytest.raises(ValueError, match="device"):
        raw_partition_of_unity_weights(
            torch.zeros((1, 2), dtype=torch.float64),
            torch.ones((2,), dtype=torch.float64, device="cuda"),
        )


def test_cpu_cuda_parity_when_cuda_exists() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    points, radius = _deterministic_patch(count=6)
    values = torch.linspace(-1.0, 2.0, points.shape[0], dtype=torch.float64)
    queries = torch.tensor(
        [[0.1, -0.2, 0.15], [1.75, -0.4, 0.2]],
        dtype=torch.float64,
    )
    cpu_system = build_augmented_system(points, radius)
    cpu_solution = solve_augmented_system(cpu_system, values)
    cpu_output = evaluate_local_interpolant(queries, points, radius, cpu_solution)
    cpu_cardinal = local_cardinal_weights(
        queries,
        points,
        radius,
        augmented_system=cpu_system,
    )
    topology_distances = torch.tensor(
        [[0.2, 0.6], [0.75, 1.2]],
        dtype=torch.float64,
    )
    topology_radii = torch.tensor([1.0, 1.5], dtype=torch.float64)
    cpu_raw_pu = raw_partition_of_unity_weights(
        topology_distances,
        topology_radii,
    )

    cuda_points = points.cuda()
    cuda_radius = radius.cuda()
    cuda_values = values.cuda()
    cuda_queries = queries.cuda()
    cuda_system = build_augmented_system(cuda_points, cuda_radius)
    cuda_solution = solve_augmented_system(cuda_system, cuda_values)
    cuda_output = evaluate_local_interpolant(
        cuda_queries,
        cuda_points,
        cuda_radius,
        cuda_solution,
    ).cpu()
    cuda_cardinal = local_cardinal_weights(
        cuda_queries,
        cuda_points,
        cuda_radius,
        augmented_system=cuda_system,
    ).cpu()
    cuda_raw_pu = raw_partition_of_unity_weights(
        topology_distances.cuda(),
        topology_radii.cuda(),
    ).cpu()
    torch.testing.assert_close(cuda_output, cpu_output, rtol=1.0e-9, atol=1.0e-10)
    torch.testing.assert_close(cuda_cardinal, cpu_cardinal, rtol=1.0e-9, atol=1.0e-10)
    torch.testing.assert_close(cuda_raw_pu, cpu_raw_pu, rtol=1.0e-12, atol=1.0e-12)

import torch
import torch.nn.functional as F

from anigroom.grooming import (
    GroomParameterField,
    build_brush_centerline,
    strand_segment_budgets,
)


def brush_inputs() -> tuple[torch.Tensor, ...]:
    roots = torch.zeros((1, 3), dtype=torch.float64)
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    directions = F.normalize(
        torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float64),
        dim=-1,
    )
    lengths = torch.tensor([[2.0]], dtype=torch.float64)
    return roots, normals, directions, lengths


def straight_samples(
    roots: torch.Tensor,
    directions: torch.Tensor,
    lengths: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, samples, dtype=roots.dtype).view(1, samples, 1)
    return roots[:, None] + t * (lengths * F.normalize(directions, dim=-1))[:, None]


def test_zero_brush_strength_is_the_exact_root_to_tip_segment() -> None:
    roots, normals, directions, lengths = brush_inputs()
    points = build_brush_centerline(
        roots,
        normals,
        directions,
        lengths,
        torch.zeros((1, 1), dtype=torch.float64),
        samples=17,
    )
    torch.testing.assert_close(
        points,
        straight_samples(roots, directions, lengths, 17),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_direction_difference_explicitly_scales_stiffness() -> None:
    roots, normals, directions, lengths = brush_inputs()
    stiffness = torch.tensor([[0.7]], dtype=torch.float64)
    points = build_brush_centerline(
        roots, normals, directions, lengths, stiffness, samples=9
    )

    directions = F.normalize(directions, dim=-1)
    delta = lengths * directions
    normal_delta = (delta * normals).sum(dim=-1, keepdim=True) * normals
    direction_tangent = directions - (
        directions * normals
    ).sum(dim=-1, keepdim=True) * normals
    direction_difference = torch.linalg.vector_norm(
        direction_tangent, dim=-1, keepdim=True
    )
    effective_stiffness = stiffness * direction_difference
    tip = roots + delta
    straight_control = 0.5 * (roots + tip)
    corner_control = roots + normal_delta
    control = straight_control + effective_stiffness * (
        corner_control - straight_control
    )
    t = torch.linspace(0.0, 1.0, 9, dtype=roots.dtype).view(1, 9, 1)
    expected = (
        (1.0 - t).square() * roots[:, None]
        + 2.0 * (1.0 - t) * t * control[:, None]
        + t.square() * tip[:, None]
    )
    torch.testing.assert_close(points, expected, atol=1.0e-12, rtol=1.0e-12)


def test_normal_aligned_direction_is_naturally_straight_without_a_branch() -> None:
    roots, normals, _, lengths = brush_inputs()
    points = build_brush_centerline(
        roots,
        normals,
        normals,
        lengths,
        torch.ones((1, 1), dtype=torch.float64),
        samples=17,
    )
    torch.testing.assert_close(
        points,
        straight_samples(roots, normals, lengths, 17),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_direction_closer_to_normal_has_less_curve_at_equal_stiffness() -> None:
    roots, normals, _, lengths = brush_inputs()
    near = F.normalize(torch.tensor([[0.12, 0.0, 1.0]], dtype=torch.float64), dim=-1)
    far = F.normalize(torch.tensor([[1.0, 0.0, 0.35]], dtype=torch.float64), dim=-1)
    stiffness = torch.ones((1, 1), dtype=torch.float64)
    near_curve = build_brush_centerline(roots, normals, near, lengths, stiffness, 33)
    far_curve = build_brush_centerline(roots, normals, far, lengths, stiffness, 33)
    near_error = torch.linalg.vector_norm(
        near_curve - straight_samples(roots, near, lengths, 33), dim=-1
    ).max()
    far_error = torch.linalg.vector_norm(
        far_curve - straight_samples(roots, far, lengths, 33), dim=-1
    ).max()
    assert float(near_error) < float(far_error)


def test_brush_curve_is_one_quadratic_turn_with_fixed_endpoints() -> None:
    roots, normals, directions, lengths = brush_inputs()
    points = build_brush_centerline(
        roots,
        normals,
        directions,
        lengths,
        torch.ones((1, 1), dtype=torch.float64),
        samples=65,
    )
    torch.testing.assert_close(points[:, 0], roots, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(
        points[:, -1], roots + lengths * directions, atol=1.0e-12, rtol=0.0
    )
    second_difference = points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]
    torch.testing.assert_close(
        second_difference,
        second_difference[:, :1].expand_as(second_difference),
        atol=2.0e-12,
        rtol=2.0e-10,
    )


def test_brush_curve_gradients_reach_length_direction_and_stiffness() -> None:
    roots, normals, directions, lengths = brush_inputs()
    directions = directions.detach().requires_grad_(True)
    lengths = lengths.detach().requires_grad_(True)
    stiffness = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
    points = build_brush_centerline(
        roots, normals, directions, lengths, stiffness, samples=13
    )
    weights = torch.linspace(0.2, 1.3, points.numel(), dtype=points.dtype).reshape_as(points)
    (points * weights).sum().backward()
    for value in (directions.grad, lengths.grad, stiffness.grad):
        assert value is not None
        assert bool(torch.isfinite(value).all())
        assert bool((value.abs() > 0.0).any())


def test_legacy_bend_is_absent_from_the_groom_schema() -> None:
    field = GroomParameterField(2, init_length=0.02)
    assert not hasattr(field, "bend_raw")
    assert "bend" not in field.decode().__dataclass_fields__


def test_final_brush_curve_drives_adaptive_segment_allocation() -> None:
    roots, normals, directions, lengths = brush_inputs()
    straight = build_brush_centerline(
        roots, normals, directions, lengths, torch.zeros((1, 1), dtype=torch.float64), 65
    )
    brushed = build_brush_centerline(
        roots, normals, directions, lengths, torch.ones((1, 1), dtype=torch.float64), 65
    )
    straight_budget, _ = strand_segment_budgets(
        straight,
        lengths,
        min_segments=4,
        length_origin=1.0,
        segments_per_unit_length=4.0,
        segments_per_unit_complexity=20.0,
    )
    brushed_budget, _ = strand_segment_budgets(
        brushed,
        lengths,
        min_segments=4,
        length_origin=1.0,
        segments_per_unit_length=4.0,
        segments_per_unit_complexity=20.0,
    )
    assert int(brushed_budget) > int(straight_budget)

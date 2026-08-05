from dataclasses import replace

import torch
import torch.nn.functional as F

from anigroom.grooming import (
    GroomParameterField,
    build_brush_centerline,
    build_strands,
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
    t = torch.linspace(0.0, 1.0, 17, dtype=torch.float64).view(1, 17, 1)
    expected = roots[:, None] + t * (lengths * directions)[:, None]
    torch.testing.assert_close(points, expected, atol=1.0e-12, rtol=1.0e-12)


def test_brush_strength_preserves_endpoints_and_delays_tangent_motion() -> None:
    roots, normals, directions, lengths = brush_inputs()
    straight = build_brush_centerline(
        roots, normals, directions, lengths, torch.zeros((1, 1), dtype=torch.float64), 9
    )
    brushed = build_brush_centerline(
        roots, normals, directions, lengths, torch.ones((1, 1), dtype=torch.float64), 9
    )
    tip = roots + lengths * directions
    torch.testing.assert_close(brushed[:, 0], roots)
    torch.testing.assert_close(brushed[:, -1], tip)
    assert float(brushed[0, 1, 2]) > float(straight[0, 1, 2])
    assert float(brushed[0, 1, 0]) < float(straight[0, 1, 0])


def test_brush_curve_gradients_reach_length_direction_and_strength() -> None:
    roots, normals, directions, lengths = brush_inputs()
    directions = directions.detach().requires_grad_(True)
    lengths = lengths.detach().requires_grad_(True)
    strength = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
    points = build_brush_centerline(
        roots, normals, directions, lengths, strength, samples=13
    )
    weights = torch.linspace(0.2, 1.3, points.numel(), dtype=points.dtype).reshape_as(points)
    (points * weights).sum().backward()
    for value in (directions.grad, lengths.grad, strength.grad):
        assert value is not None
        assert bool(torch.isfinite(value).all())
        assert bool((value.abs() > 0.0).any())


def test_bend_is_a_smooth_interior_deformation_with_fixed_endpoints() -> None:
    roots, normals, directions, lengths = brush_inputs()
    tangents = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    bitangents = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    field = GroomParameterField(1, init_length=2.0).to(dtype=torch.float64)
    groom = replace(
        field.decode(),
        length=lengths,
        direction_local=directions,
        brush_curve_strength=torch.tensor([[0.6]], dtype=torch.float64),
        curl_radius=torch.zeros((1, 1), dtype=torch.float64),
        frizz=torch.zeros((1, 1), dtype=torch.float64),
    )
    straight, *_ = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        replace(groom, bend=torch.zeros((1, 1), dtype=torch.float64)),
        samples=1001,
    )
    bent, *_ = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        replace(groom, bend=torch.tensor([[0.7]], dtype=torch.float64)),
        samples=1001,
    )
    torch.testing.assert_close(bent[:, 0], straight[:, 0], atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close(bent[:, -1], straight[:, -1], atol=1.0e-12, rtol=0.0)
    assert float((bent - straight)[0, 500, 1]) > 0.0
    torch.testing.assert_close(
        F.normalize(bent[:, 1] - bent[:, 0], dim=-1),
        F.normalize(straight[:, 1] - straight[:, 0], dim=-1),
        atol=2.0e-2,
        rtol=2.0e-2,
    )
    torch.testing.assert_close(
        F.normalize(bent[:, -1] - bent[:, -2], dim=-1),
        F.normalize(straight[:, -1] - straight[:, -2], dim=-1),
        atol=2.0e-2,
        rtol=2.0e-2,
    )


def test_bend_is_not_saturated_by_a_legacy_tanh_domain() -> None:
    field = GroomParameterField(2, init_length=0.02)
    with torch.no_grad():
        field.bend_raw.copy_(torch.tensor([[-3.0], [4.0]]))
    torch.testing.assert_close(field.decode().bend, field.bend_raw)


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

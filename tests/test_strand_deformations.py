from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

from anigroom.grooming.strand_deformations import (
    backbone_transverse_frame,
    backbone_transverse_frames,
    curl_backbone,
    deform_backbone,
    frizz_backbone,
)
from anigroom.grooming.strand_gaussians import (
    GroomParameterField,
    build_brush_centerline,
    build_strands,
    make_tangent_frames,
)


def canonical_backbone(
    *,
    length: float = 0.03,
    samples: int = 129,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    roots = torch.zeros((1, 3), dtype=torch.float64)
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    directions = torch.nn.functional.normalize(
        torch.tensor([[0.82, 0.0, 0.57]], dtype=torch.float64),
        dim=-1,
    )
    tangents = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    backbone = build_brush_centerline(
        roots,
        normals,
        directions,
        torch.tensor([[length]], dtype=torch.float64),
        torch.tensor([[0.65]], dtype=torch.float64),
        samples,
    )
    return backbone, normals, directions, tangents


def chord_progress(points: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return ((points - points[:, :1]) * direction[:, None]).sum(dim=-1)


def test_retired_advanced_geometry_checkpoint_schema_is_rejected() -> None:
    field = GroomParameterField(7)
    retired_state = dict(field.state_dict())
    retired_state["curl_frequency_raw"] = retired_state.pop("curl_turns_raw")

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        GroomParameterField(7).load_state_dict(retired_state, strict=True)


def test_r059_absolute_shape_checkpoint_schema_is_rejected() -> None:
    field = GroomParameterField(7)
    r059_state = dict(field.state_dict())
    r059_state["curl_radius_raw"] = r059_state.pop("curl_radius_ratio_raw")

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        GroomParameterField(7).load_state_dict(r059_state, strict=True)


def test_default_advanced_geometry_is_neutral() -> None:
    groom = GroomParameterField(7).decode()
    assert float(groom.curl_radius_ratio.max()) < 5.0e-7


def test_r067_groom_state_and_deformation_signature_are_frizz_free() -> None:
    field = GroomParameterField(7)
    state_names = set(field.state_dict())
    state_names.update(name for name, _ in field.named_parameters())
    state_names.update(name for name, _ in field.named_buffers())
    assert not any("frizz" in name.lower() for name in state_names)
    assert not any("frizz" in name.lower() for name in field.decode().__dict__)
    assert "frizz" not in inspect.signature(deform_backbone).parameters
    assert "frizz" not in inspect.signature(build_strands).parameters


def test_transverse_frame_is_orthonormal_when_direction_matches_normal() -> None:
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    directions = normals.clone()
    tangents = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)

    axis, side, outward = backbone_transverse_frame(normals, directions, tangents)

    for value in (axis, side, outward):
        torch.testing.assert_close(
            torch.linalg.norm(value, dim=-1),
            torch.ones(1, dtype=torch.float64),
        )
    torch.testing.assert_close((axis * side).sum(dim=-1), torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close((axis * outward).sum(dim=-1), torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close((side * outward).sum(dim=-1), torch.zeros(1, dtype=torch.float64))


def test_local_frames_follow_brush_backbone_without_twist() -> None:
    backbone, normals, directions, tangents = canonical_backbone()
    curve_tangent, side, outward = backbone_transverse_frames(
        backbone,
        normals,
        directions,
        tangents,
    )

    ones = torch.ones_like(curve_tangent[..., 0])
    zeros = torch.zeros_like(ones)
    for value in (curve_tangent, side, outward):
        torch.testing.assert_close(torch.linalg.norm(value, dim=-1), ones)
    torch.testing.assert_close((curve_tangent * side).sum(dim=-1), zeros, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close((curve_tangent * outward).sum(dim=-1), zeros, atol=1.0e-12, rtol=0.0)
    torch.testing.assert_close((side * outward).sum(dim=-1), zeros, atol=1.0e-12, rtol=0.0)
    # The normal-to-direction brush curve is planar, so its no-twist side axis
    # must remain constant along the strand.
    torch.testing.assert_close(side, side[:, :1].expand_as(side), atol=1.0e-12, rtol=0.0)


def test_curl_preserves_root_and_root_tangent_but_may_move_tip() -> None:
    backbone, normals, directions, tangents = canonical_backbone(samples=513)
    shaped = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.tensor([[0.004]], dtype=torch.float64),
        curl_turns=torch.tensor([[1.7]], dtype=torch.float64),
        curl_phase=torch.tensor([[0.7]], dtype=torch.float64),
    )

    torch.testing.assert_close(shaped[:, 0], backbone[:, 0], atol=0.0, rtol=0.0)
    assert float(torch.linalg.norm(shaped[:, -1] - backbone[:, -1])) > 1.0e-4

    base_first = backbone[:, 1] - backbone[:, 0]
    shaped_first = shaped[:, 1] - shaped[:, 0]
    root_tangent_cosine = torch.nn.functional.cosine_similarity(base_first, shaped_first, dim=-1)
    assert float(root_tangent_cosine.min()) > 0.99999


def test_detail_offsets_are_transverse_to_the_base_backbone() -> None:
    backbone, normals, directions, tangents = canonical_backbone()
    curve_tangent, _, _ = backbone_transverse_frames(
        backbone,
        normals,
        directions,
        tangents,
    )
    shaped = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.tensor([[0.003]], dtype=torch.float64),
        curl_turns=torch.tensor([[1.4]], dtype=torch.float64),
        curl_phase=torch.tensor([[0.2]], dtype=torch.float64),
    )
    longitudinal_offset = ((shaped - backbone) * curve_tangent).sum(dim=-1)
    torch.testing.assert_close(
        longitudinal_offset,
        torch.zeros_like(longitudinal_offset),
        atol=2.0e-12,
        rtol=0.0,
    )


def test_zero_curl_is_exact_identity() -> None:
    backbone, normals, directions, tangents = canonical_backbone()
    shaped = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.zeros((1, 1), dtype=torch.float64),
        curl_turns=torch.tensor([[5.0]], dtype=torch.float64),
        curl_phase=torch.tensor([[2.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(shaped, backbone, atol=0.0, rtol=0.0)


def test_zero_curl_turns_cannot_duplicate_the_brush_bend() -> None:
    backbone, normals, directions, tangents = canonical_backbone()
    shaped = curl_backbone(
        backbone,
        normals,
        directions,
        tangents,
        radius=torch.tensor([[0.01]], dtype=torch.float64),
        turns=torch.zeros((1, 1), dtype=torch.float64),
        phase=torch.tensor([[1.2]], dtype=torch.float64),
    )
    torch.testing.assert_close(shaped, backbone, atol=2.0e-18, rtol=0.0)


def test_zero_turn_jacobian_is_nonzero_with_positive_radius() -> None:
    backbone, normals, directions, tangents = canonical_backbone(samples=33)
    _, side, _ = backbone_transverse_frames(
        backbone,
        normals,
        directions,
        tangents,
    )
    radius = torch.tensor([[0.004]], dtype=torch.float64)
    turns = torch.zeros((1, 1), dtype=torch.float64, requires_grad=True)
    shaped = curl_backbone(
        backbone,
        normals,
        directions,
        tangents,
        radius=radius,
        turns=turns,
        phase=torch.zeros_like(turns),
    )
    tip_side_projection = (shaped[:, -1] * side[:, -1]).sum()
    gradient = torch.autograd.grad(tip_side_projection, turns)[0]
    torch.testing.assert_close(
        gradient,
        2.0 * torch.pi * radius,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_signed_turns_reverse_side_handedness_without_clamping() -> None:
    backbone, normals, directions, tangents = canonical_backbone(samples=129)
    _, side, outward = backbone_transverse_frames(
        backbone,
        normals,
        directions,
        tangents,
    )
    positive = curl_backbone(
        backbone,
        normals,
        directions,
        tangents,
        radius=torch.tensor([[0.004]], dtype=torch.float64),
        turns=torch.tensor([[1.25]], dtype=torch.float64),
        phase=torch.zeros((1, 1), dtype=torch.float64),
    )
    negative = curl_backbone(
        backbone,
        normals,
        directions,
        tangents,
        radius=torch.tensor([[0.004]], dtype=torch.float64),
        turns=torch.tensor([[-1.25]], dtype=torch.float64),
        phase=torch.zeros((1, 1), dtype=torch.float64),
    )
    positive_offset = positive - backbone
    negative_offset = negative - backbone
    torch.testing.assert_close(
        (negative_offset * side).sum(dim=-1),
        -(positive_offset * side).sum(dim=-1),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        (negative_offset * outward).sum(dim=-1),
        (positive_offset * outward).sum(dim=-1),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_curl_uses_both_transverse_axes() -> None:
    backbone, normals, directions, tangents = canonical_backbone()
    _, side, outward = backbone_transverse_frames(
        backbone,
        normals,
        directions,
        tangents,
    )
    shaped = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.tensor([[0.01]], dtype=torch.float64),
        curl_turns=torch.tensor([[2.5]], dtype=torch.float64),
        curl_phase=torch.tensor([[0.3]], dtype=torch.float64),
    )
    displacement = shaped - backbone
    side_displacement = (displacement * side).sum(dim=-1)
    outward_displacement = (displacement * outward).sum(dim=-1)
    assert float(side_displacement.abs().max()) > 0.0
    assert float(outward_displacement.abs().max()) > 0.0
    assert float(outward_displacement.min()) < 0.0
    assert float(outward_displacement.max()) > 0.0


def test_moderate_detail_does_not_create_axial_foldback() -> None:
    backbone, normals, directions, tangents = canonical_backbone(length=0.04)
    shaped = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.tensor([[0.002]], dtype=torch.float64),
        curl_turns=torch.tensor([[1.25]], dtype=torch.float64),
        curl_phase=torch.tensor([[0.4]], dtype=torch.float64),
    )
    progress = torch.diff(chord_progress(shaped, directions), dim=1)
    assert bool((progress > 0.0).all())


def test_extreme_curl_remains_finite_and_forms_a_real_coil() -> None:
    backbone, normals, directions, tangents = canonical_backbone(length=0.015, samples=257)
    shaped = curl_backbone(
        backbone,
        normals,
        directions,
        tangents,
        radius=torch.tensor([[0.008]], dtype=torch.float64),
        turns=torch.tensor([[4.0]], dtype=torch.float64),
        phase=torch.tensor([[1.4]], dtype=torch.float64),
    )
    segment_lengths = torch.linalg.norm(torch.diff(shaped, dim=1), dim=-1)
    backbone_length = torch.linalg.norm(torch.diff(backbone, dim=1), dim=-1).sum()
    shaped_length = segment_lengths.sum()
    assert bool(torch.isfinite(shaped).all())
    assert float(segment_lengths.min()) > 0.0
    assert float(shaped_length / backbone_length) > 2.0


def test_physical_shape_is_scale_equivariant() -> None:
    backbone, normals, directions, tangents = canonical_backbone(length=0.025)
    shaped = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.tensor([[0.004]], dtype=torch.float64),
        curl_turns=torch.tensor([[1.75]], dtype=torch.float64),
        curl_phase=torch.tensor([[0.2]], dtype=torch.float64),
    )

    scale = 3.0
    scaled = deform_backbone(
        backbone * scale,
        normals,
        directions,
        tangents,
        curl_radius=torch.tensor([[0.004 * scale]], dtype=torch.float64),
        curl_turns=torch.tensor([[1.75]], dtype=torch.float64),
        curl_phase=torch.tensor([[0.2]], dtype=torch.float64),
    )
    torch.testing.assert_close(scaled, shaped * scale, atol=2.0e-11, rtol=1.0e-10)


def test_groom_shape_ratios_are_scale_equivariant() -> None:
    dtype = torch.float64
    roots = torch.zeros((2, 3), dtype=dtype)
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype).expand_as(roots)
    tangents, bitangents = make_tangent_frames(normals)
    base = GroomParameterField(2).decode()
    lengths = torch.tensor([[0.025], [0.075]], dtype=dtype)
    groom = replace(
        base,
        length=lengths,
        root_width=torch.full((2, 1), 0.0004, dtype=dtype),
        tip_width=torch.full((2, 1), 0.00008, dtype=dtype),
        width_taper=torch.full((2, 1), 1.4, dtype=dtype),
        direction_local=torch.tensor(
            [[0.82, 0.0, 0.57], [0.82, 0.0, 0.57]],
            dtype=dtype,
        ),
        brush_stiffness=torch.full((2, 1), 0.65, dtype=dtype),
        curl_radius_ratio=torch.full((2, 1), 0.16, dtype=dtype),
        curl_turns=torch.full((2, 1), 1.75, dtype=dtype),
        curl_phase=torch.full((2, 1), 0.2, dtype=dtype),
        child_radius=torch.zeros((2, 1), dtype=dtype),
        clump_strength=torch.zeros((2, 1), dtype=dtype),
        root_color=torch.zeros((2, 3), dtype=dtype),
        tip_color=torch.zeros((2, 3), dtype=dtype),
        root_opacity=torch.ones((2, 1), dtype=dtype),
        tip_opacity=torch.ones((2, 1), dtype=dtype),
        opacity=torch.ones((2, 1), dtype=dtype),
    )
    strands, _, _, _ = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=129,
    )

    normalized = (strands - roots[:, None]) / lengths[:, None]
    torch.testing.assert_close(
        normalized[0],
        normalized[1],
        atol=2.0e-11,
        rtol=1.0e-10,
    )
    torch.testing.assert_close(
        strands[1],
        strands[0] * 3.0,
        atol=2.0e-11,
        rtol=1.0e-10,
    )


def test_groom_shape_ratio_geometry_has_finite_nonzero_gradients() -> None:
    dtype = torch.float64
    roots = torch.zeros((1, 3), dtype=dtype)
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
    tangents, bitangents = make_tangent_frames(normals)
    base = GroomParameterField(1).decode()
    length = torch.tensor([[0.04]], dtype=dtype, requires_grad=True)
    curl_ratio = torch.tensor([[0.10]], dtype=dtype, requires_grad=True)
    groom = replace(
        base,
        length=length,
        root_width=torch.full((1, 1), 0.0004, dtype=dtype),
        tip_width=torch.full((1, 1), 0.00008, dtype=dtype),
        width_taper=torch.full((1, 1), 1.4, dtype=dtype),
        direction_local=torch.tensor([[0.82, 0.0, 0.57]], dtype=dtype),
        brush_stiffness=torch.full((1, 1), 0.65, dtype=dtype),
        curl_radius_ratio=curl_ratio,
        curl_turns=torch.full((1, 1), 1.6, dtype=dtype),
        curl_phase=torch.full((1, 1), 0.3, dtype=dtype),
        child_radius=torch.zeros((1, 1), dtype=dtype),
        clump_strength=torch.zeros((1, 1), dtype=dtype),
        root_color=torch.zeros((1, 3), dtype=dtype),
        tip_color=torch.zeros((1, 3), dtype=dtype),
        root_opacity=torch.ones((1, 1), dtype=dtype),
        tip_opacity=torch.ones((1, 1), dtype=dtype),
        opacity=torch.ones((1, 1), dtype=dtype),
    )
    strands, _, _, _ = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=129,
    )
    weights = torch.linspace(
        0.2,
        1.3,
        strands.numel(),
        dtype=dtype,
    ).reshape_as(strands)
    (strands * weights).sum().backward()

    for gradient in (length.grad, curl_ratio.grad):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.abs().sum()) > 0.0


def test_standalone_frizz_is_deterministic_and_differentiable() -> None:
    backbone, normals, directions, tangents = canonical_backbone(samples=65)
    amplitude = torch.tensor([[0.002]], dtype=torch.float64, requires_grad=True)
    seed_phase = torch.tensor([[0.9]], dtype=torch.float64)
    first = frizz_backbone(
        backbone,
        normals,
        directions,
        tangents,
        amplitude=amplitude,
        seed_phase=seed_phase,
    )
    second = frizz_backbone(
        backbone,
        normals,
        directions,
        tangents,
        amplitude=amplitude,
        seed_phase=seed_phase,
    )
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first[:, 0], backbone[:, 0], atol=0.0, rtol=0.0)
    weights = torch.linspace(0.2, 1.3, first.numel(), dtype=first.dtype).reshape_as(first)
    (first * weights).sum().backward()
    assert amplitude.grad is not None
    assert bool(torch.isfinite(amplitude.grad).all())
    assert float(amplitude.grad.abs().sum()) > 0.0


def test_standalone_frizz_is_stable_across_sampling_density() -> None:
    low, normals, directions, tangents = canonical_backbone(samples=65)
    high, _, _, _ = canonical_backbone(samples=129)
    kwargs = {
        "amplitude": torch.tensor([[0.002]], dtype=torch.float64),
        "seed_phase": torch.tensor([[0.9]], dtype=torch.float64),
    }
    low_frizz = frizz_backbone(low, normals, directions, tangents, **kwargs)
    high_frizz = frizz_backbone(high, normals, directions, tangents, **kwargs)
    torch.testing.assert_close(high_frizz[:, ::2], low_frizz, atol=5.0e-6, rtol=0.0)


def test_curl_geometry_has_finite_nonzero_gradients() -> None:
    backbone, normals, directions, tangents = canonical_backbone()
    backbone = backbone.detach().requires_grad_(True)
    radius = torch.tensor([[0.004]], dtype=torch.float64, requires_grad=True)
    turns = torch.tensor([[1.4]], dtype=torch.float64, requires_grad=True)
    curl_phase = torch.tensor([[0.6]], dtype=torch.float64, requires_grad=True)

    points = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=radius,
        curl_turns=turns,
        curl_phase=curl_phase,
    )
    weights = torch.linspace(0.2, 1.3, points.numel(), dtype=points.dtype).reshape_as(points)
    (points * weights).sum().backward()

    for gradient in (backbone.grad, radius.grad, turns.grad, curl_phase.grad):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.abs().sum()) > 0.0

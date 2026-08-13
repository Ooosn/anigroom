from __future__ import annotations

import pytest
import torch

from anigroom.grooming.strand_deformations import (
    backbone_transverse_frame,
    backbone_transverse_frames,
    curl_backbone,
    deform_backbone,
    frizz_backbone,
)
from anigroom.grooming.strand_gaussians import GroomParameterField, build_brush_centerline


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


def test_frizz_seed_is_persistent_state_but_not_a_trainable_parameter() -> None:
    field = GroomParameterField(7)
    assert "frizz_seed_phase" in dict(field.named_buffers())
    assert "frizz_seed_phase" not in dict(field.named_parameters())

    clone = GroomParameterField(7)
    clone.load_state_dict(field.state_dict(), strict=True)
    torch.testing.assert_close(clone.frizz_seed_phase, field.frizz_seed_phase)


def test_retired_advanced_geometry_checkpoint_schema_is_rejected() -> None:
    field = GroomParameterField(7)
    retired_state = dict(field.state_dict())
    retired_state["curl_frequency_raw"] = retired_state.pop("curl_turns_raw")
    retired_state.pop("frizz_seed_phase")

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        GroomParameterField(7).load_state_dict(retired_state, strict=True)


def test_default_advanced_geometry_is_neutral() -> None:
    groom = GroomParameterField(7).decode()
    assert float(groom.curl_radius.max()) < 5.0e-8
    assert float(groom.frizz.max()) < 5.0e-8


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


def test_curl_and_frizz_preserve_root_and_root_tangent_but_may_move_tip() -> None:
    backbone, normals, directions, tangents = canonical_backbone(samples=513)
    shaped = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.tensor([[0.004]], dtype=torch.float64),
        curl_turns=torch.tensor([[1.7]], dtype=torch.float64),
        curl_phase=torch.tensor([[0.7]], dtype=torch.float64),
        frizz_amplitude=torch.tensor([[0.0015]], dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[1.3]], dtype=torch.float64),
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
        frizz_amplitude=torch.tensor([[0.001]], dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[1.1]], dtype=torch.float64),
    )
    longitudinal_offset = ((shaped - backbone) * curve_tangent).sum(dim=-1)
    torch.testing.assert_close(
        longitudinal_offset,
        torch.zeros_like(longitudinal_offset),
        atol=2.0e-12,
        rtol=0.0,
    )


def test_zero_curl_and_frizz_are_exact_identity() -> None:
    backbone, normals, directions, tangents = canonical_backbone()
    shaped = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.zeros((1, 1), dtype=torch.float64),
        curl_turns=torch.tensor([[5.0]], dtype=torch.float64),
        curl_phase=torch.tensor([[2.0]], dtype=torch.float64),
        frizz_amplitude=torch.zeros((1, 1), dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[0.4]], dtype=torch.float64),
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


def test_curl_and_frizz_are_additive_and_parameter_independent() -> None:
    backbone, normals, directions, tangents = canonical_backbone()
    curl = curl_backbone(
        backbone,
        normals,
        directions,
        tangents,
        radius=torch.tensor([[0.003]], dtype=torch.float64),
        turns=torch.tensor([[1.25]], dtype=torch.float64),
        phase=torch.tensor([[0.3]], dtype=torch.float64),
    )
    frizz = frizz_backbone(
        backbone,
        normals,
        directions,
        tangents,
        amplitude=torch.tensor([[0.001]], dtype=torch.float64),
        seed_phase=torch.tensor([[1.4]], dtype=torch.float64),
    )
    combined = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.tensor([[0.003]], dtype=torch.float64),
        curl_turns=torch.tensor([[1.25]], dtype=torch.float64),
        curl_phase=torch.tensor([[0.3]], dtype=torch.float64),
        frizz_amplitude=torch.tensor([[0.001]], dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[1.4]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        combined - backbone,
        (curl - backbone) + (frizz - backbone),
        atol=2.0e-12,
        rtol=0.0,
    )

    # Curl controls cannot alter frizz when curl radius is zero.
    frizz_a = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.zeros((1, 1), dtype=torch.float64),
        curl_turns=torch.tensor([[0.1]], dtype=torch.float64),
        curl_phase=torch.tensor([[0.2]], dtype=torch.float64),
        frizz_amplitude=torch.tensor([[0.001]], dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[1.4]], dtype=torch.float64),
    )
    frizz_b = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=torch.zeros((1, 1), dtype=torch.float64),
        curl_turns=torch.tensor([[7.0]], dtype=torch.float64),
        curl_phase=torch.tensor([[2.7]], dtype=torch.float64),
        frizz_amplitude=torch.tensor([[0.001]], dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[1.4]], dtype=torch.float64),
    )
    torch.testing.assert_close(frizz_a, frizz_b, atol=0.0, rtol=0.0)


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
        frizz_amplitude=torch.zeros((1, 1), dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[1.1]], dtype=torch.float64),
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
        frizz_amplitude=torch.tensor([[0.0008]], dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[1.0]], dtype=torch.float64),
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
        frizz_amplitude=torch.tensor([[0.002]], dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[0.9]], dtype=torch.float64),
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
        frizz_amplitude=torch.tensor([[0.002 * scale]], dtype=torch.float64),
        frizz_seed_phase=torch.tensor([[0.9]], dtype=torch.float64),
    )
    torch.testing.assert_close(scaled, shaped * scale, atol=2.0e-11, rtol=1.0e-10)


def test_frizz_is_stable_across_sampling_density() -> None:
    low, normals, directions, tangents = canonical_backbone(samples=65)
    high, _, _, _ = canonical_backbone(samples=129)
    kwargs = {
        "amplitude": torch.tensor([[0.002]], dtype=torch.float64),
        "seed_phase": torch.tensor([[0.9]], dtype=torch.float64),
    }
    low_frizz = frizz_backbone(low, normals, directions, tangents, **kwargs)
    high_frizz = frizz_backbone(high, normals, directions, tangents, **kwargs)
    torch.testing.assert_close(high_frizz[:, ::2], low_frizz, atol=5.0e-6, rtol=0.0)


def test_curl_frizz_geometry_has_finite_nonzero_gradients() -> None:
    backbone, normals, directions, tangents = canonical_backbone()
    backbone = backbone.detach().requires_grad_(True)
    radius = torch.tensor([[0.004]], dtype=torch.float64, requires_grad=True)
    turns = torch.tensor([[1.4]], dtype=torch.float64, requires_grad=True)
    curl_phase = torch.tensor([[0.6]], dtype=torch.float64, requires_grad=True)
    amplitude = torch.tensor([[0.002]], dtype=torch.float64, requires_grad=True)

    points = deform_backbone(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius=radius,
        curl_turns=turns,
        curl_phase=curl_phase,
        frizz_amplitude=amplitude,
        frizz_seed_phase=torch.tensor([[1.2]], dtype=torch.float64),
    )
    weights = torch.linspace(0.2, 1.3, points.numel(), dtype=points.dtype).reshape_as(points)
    (points * weights).sum().backward()

    for gradient in (backbone.grad, radius.grad, turns.grad, curl_phase.grad, amplitude.grad):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert float(gradient.abs().sum()) > 0.0

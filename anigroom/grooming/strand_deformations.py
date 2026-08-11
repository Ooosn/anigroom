"""Differentiable detail deformations around a groomed strand backbone.

The backbone owns root attachment, nominal length, 3D groom direction, and the
low-frequency normal-to-direction brush turn. Curl and frizz are independent
transverse detail layers around that backbone. They keep the root and its
tangent fixed, but may move the final point: forcing every detailed strand back
to the nominal tip creates the artificial S-curves that this module avoids.
"""

from __future__ import annotations

import math

import torch


EPS = 1.0e-8


def _normalize(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.norm(value, dim=-1, keepdim=True).clamp_min(EPS)


def _project_to_normal_plane(value: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return value - (value * axis).sum(dim=-1, keepdim=True) * axis


def backbone_transverse_frame(
    normals: torch.Tensor,
    directions: torch.Tensor,
    tangents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``axis, side, outward`` as an orthonormal backbone frame.

    ``axis`` is the exact root-to-tip direction.  ``outward`` is the mesh normal
    projected into the plane perpendicular to that axis.  A tangent-derived and
    finally a cardinal fallback make the frame defined even when direction and
    normal are parallel.
    """

    if normals.shape != directions.shape or normals.shape != tangents.shape:
        raise ValueError("normals, directions, and tangents must have matching [R, 3] shapes")

    axis = _normalize(directions)
    normal_reference = _project_to_normal_plane(_normalize(normals), axis)
    tangent_reference = _project_to_normal_plane(_normalize(tangents), axis)

    normal_norm = torch.linalg.norm(normal_reference, dim=-1, keepdim=True)
    tangent_norm = torch.linalg.norm(tangent_reference, dim=-1, keepdim=True)
    has_normal_reference = normal_norm > EPS
    reference = torch.where(has_normal_reference, normal_reference, tangent_reference)

    cardinal_index = torch.argmin(axis.abs(), dim=-1)
    cardinal = torch.nn.functional.one_hot(cardinal_index, num_classes=3).to(
        device=axis.device,
        dtype=axis.dtype,
    )
    cardinal_reference = _project_to_normal_plane(cardinal, axis)
    has_reference = has_normal_reference | (tangent_norm > EPS)
    reference = torch.where(has_reference, reference, cardinal_reference)

    outward = _normalize(reference)
    side = _normalize(torch.cross(axis, outward, dim=-1))
    # Reconstructing outward removes accumulated projection error and fixes the
    # handedness to axis x outward = side.
    outward = _normalize(torch.cross(side, axis, dim=-1))
    return axis, side, outward


def _backbone_tangents(backbone: torch.Tensor) -> torch.Tensor:
    """Return stable per-sample tangents for a sampled open curve."""

    segments = backbone[:, 1:] - backbone[:, :-1]
    segment_tangents = _normalize(segments)
    if backbone.shape[1] == 2:
        return torch.cat([segment_tangents, segment_tangents], dim=1)
    interior = _normalize(segment_tangents[:, :-1] + segment_tangents[:, 1:])
    return torch.cat(
        [segment_tangents[:, :1], interior, segment_tangents[:, -1:]],
        dim=1,
    )


def backbone_transverse_frames(
    backbone: torch.Tensor,
    normals: torch.Tensor,
    directions: torch.Tensor,
    tangents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return local ``curve_tangent, side, outward`` frames along a backbone.

    The brush backbone lies in the root-normal/groom-direction plane. Projecting
    its root transverse frame onto each sampled tangent follows that curve
    without adding an unrelated twist around the straight root-to-tip chord.
    """

    if backbone.ndim != 3 or backbone.shape[-1] != 3:
        raise ValueError("backbone must have shape [R, samples, 3]")
    _, root_side, root_outward = backbone_transverse_frame(
        normals,
        directions,
        tangents,
    )
    curve_tangent = _backbone_tangents(backbone)
    side_reference = root_side[:, None].expand_as(curve_tangent)
    side_projected = side_reference - (
        side_reference * curve_tangent
    ).sum(dim=-1, keepdim=True) * curve_tangent
    outward_reference = root_outward[:, None].expand_as(curve_tangent)
    outward_projected = outward_reference - (
        outward_reference * curve_tangent
    ).sum(dim=-1, keepdim=True) * curve_tangent
    side_norm = torch.linalg.norm(side_projected, dim=-1, keepdim=True)
    side = torch.where(side_norm > EPS, side_projected, outward_projected)
    side = _normalize(side)
    outward = _normalize(torch.cross(side, curve_tangent, dim=-1))
    return curve_tangent, side, outward


def root_fade_envelope(t: torch.Tensor) -> torch.Tensor:
    """Smoothly introduce detail while preserving root position and tangent."""

    return t.square() * (3.0 - 2.0 * t)


def _shape_offset(
    side_coefficient: torch.Tensor,
    outward_coefficient: torch.Tensor,
    side: torch.Tensor,
    outward: torch.Tensor,
) -> torch.Tensor:
    return side_coefficient * side + outward_coefficient * outward


def _curl_offset(
    backbone: torch.Tensor,
    normals: torch.Tensor,
    directions: torch.Tensor,
    tangents: torch.Tensor,
    radius: torch.Tensor,
    turns: torch.Tensor,
    phase: torch.Tensor,
    *,
    local_frames: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    t: torch.Tensor | None = None,
    envelope: torch.Tensor | None = None,
) -> torch.Tensor:
    root_count, samples, _ = backbone.shape
    for name, value in (("radius", radius), ("turns", turns), ("phase", phase)):
        if value.shape != (root_count, 1):
            raise ValueError(f"{name} must have shape [R, 1]")

    if local_frames is None:
        local_frames = backbone_transverse_frames(
            backbone,
            normals,
            directions,
            tangents,
        )
    _, side, outward = local_frames
    if t is None:
        t = torch.linspace(
            0.0,
            1.0,
            samples,
            device=backbone.device,
            dtype=backbone.dtype,
        ).view(1, samples, 1)
    angle = 2.0 * math.pi * turns[:, None] * t + phase[:, None]
    if envelope is None:
        envelope = root_fade_envelope(t)
    initial_side = torch.sin(phase)[:, None]
    initial_outward = torch.cos(phase)[:, None]
    unit_offset = _shape_offset(
        torch.sin(angle) - initial_side,
        torch.cos(angle) - initial_outward,
        side,
        outward,
    )
    return radius[:, None] * envelope * unit_offset


def curl_backbone(
    backbone: torch.Tensor,
    normals: torch.Tensor,
    directions: torch.Tensor,
    tangents: torch.Tensor,
    radius: torch.Tensor,
    turns: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Add a root-pinned curl around a strand backbone.

    ``radius`` is a physical transverse radius and ``turns`` is the number of
    rotations over the complete strand. The deformation follows the sampled
    backbone's local tangent frame. Detail fades in smoothly from the root so
    attachment and root tangent are unchanged. The nominal tip is deliberately
    not pinned; a real curl must be allowed to change the final point.
    """

    if backbone.ndim != 3 or backbone.shape[-1] != 3:
        raise ValueError("backbone must have shape [R, samples, 3]")
    return backbone + _curl_offset(
        backbone,
        normals,
        directions,
        tangents,
        radius,
        turns,
        phase,
    )


def _smooth_frizz_noise(
    phase: torch.Tensor,
    samples: int,
    *,
    channel_offset: float,
    knot_count: int = 8,
) -> torch.Tensor:
    """Produce deterministic, band-limited value noise for each strand.

    Phase acts as a seed, not as curl frequency.  The knot sequence is smoothed
    with cubic interpolation, centered, and normalized so frizz amplitude has a
    stable physical meaning independent of sample count.
    """

    root_count = int(phase.shape[0])
    knots = torch.arange(
        knot_count,
        device=phase.device,
        dtype=phase.dtype,
    ).view(1, knot_count)
    seed = phase.detach()
    values = (
        0.64
        * torch.sin(
            1.37 * seed
            + channel_offset
            + 2.173 * knots
            + 0.417 * knots.square()
        )
        + 0.36
        * torch.sin(
            2.11 * seed
            + 1.91 * channel_offset
            + 5.317 * knots
            + 0.173 * knots.square()
        )
    )
    values = values - values.mean(dim=1, keepdim=True)
    values = values / values.abs().amax(dim=1, keepdim=True).clamp_min(EPS)

    coordinate = torch.linspace(
        0.0,
        float(knot_count - 1),
        samples,
        device=phase.device,
        dtype=phase.dtype,
    )
    left = torch.floor(coordinate).to(dtype=torch.long)
    right = (left + 1).clamp_max(knot_count - 1)
    fraction = coordinate - left.to(dtype=coordinate.dtype)
    blend = fraction.square() * (3.0 - 2.0 * fraction)
    left_values = values[:, left]
    right_values = values[:, right]
    return ((1.0 - blend) * left_values + blend * right_values).view(
        root_count,
        samples,
        1,
    )


def _frizz_offset(
    backbone: torch.Tensor,
    normals: torch.Tensor,
    directions: torch.Tensor,
    tangents: torch.Tensor,
    amplitude: torch.Tensor,
    seed_phase: torch.Tensor,
    *,
    local_frames: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    t: torch.Tensor | None = None,
    envelope: torch.Tensor | None = None,
) -> torch.Tensor:
    root_count, samples, _ = backbone.shape
    if amplitude.shape != (root_count, 1):
        raise ValueError("amplitude must have shape [R, 1]")
    if seed_phase.shape != (root_count, 1):
        raise ValueError("seed_phase must have shape [R, 1]")

    if local_frames is None:
        local_frames = backbone_transverse_frames(
            backbone,
            normals,
            directions,
            tangents,
        )
    _, side, outward = local_frames
    if t is None:
        t = torch.linspace(
            0.0,
            1.0,
            samples,
            device=backbone.device,
            dtype=backbone.dtype,
        ).view(1, samples, 1)
    if envelope is None:
        envelope = root_fade_envelope(t)
    side_noise = _smooth_frizz_noise(
        seed_phase,
        samples,
        channel_offset=0.0,
    )
    outward_noise = _smooth_frizz_noise(
        seed_phase,
        samples,
        channel_offset=1.5707963267948966,
    )
    unit_offset = _shape_offset(
        side_noise,
        outward_noise,
        side,
        outward,
    )
    return amplitude[:, None] * envelope * unit_offset


def frizz_backbone(
    backbone: torch.Tensor,
    normals: torch.Tensor,
    directions: torch.Tensor,
    tangents: torch.Tensor,
    amplitude: torch.Tensor,
    seed_phase: torch.Tensor,
) -> torch.Tensor:
    """Add root-pinned, band-limited frizz around a backbone.

    Frizz is independent of curl turns.  ``seed_phase`` selects the fixed noise
    realization; it is intentionally detached because a random seed is not a
    geometric optimization coordinate.
    """

    if backbone.ndim != 3 or backbone.shape[-1] != 3:
        raise ValueError("backbone must have shape [R, samples, 3]")
    return backbone + _frizz_offset(
        backbone,
        normals,
        directions,
        tangents,
        amplitude,
        seed_phase,
    )


def deform_backbone(
    backbone: torch.Tensor,
    normals: torch.Tensor,
    directions: torch.Tensor,
    tangents: torch.Tensor,
    curl_radius: torch.Tensor,
    curl_turns: torch.Tensor,
    curl_phase: torch.Tensor,
    frizz_amplitude: torch.Tensor,
    frizz_seed_phase: torch.Tensor,
) -> torch.Tensor:
    """Compose independent curl and frizz offsets around one base backbone.

    Both layers use the same undeformed local frame. This makes their meaning
    independent and avoids the order-dependent behavior of applying frizz to
    an already curled curve.
    """

    samples = int(backbone.shape[1])
    local_frames = backbone_transverse_frames(
        backbone,
        normals,
        directions,
        tangents,
    )
    t = torch.linspace(
        0.0,
        1.0,
        samples,
        device=backbone.device,
        dtype=backbone.dtype,
    ).view(1, samples, 1)
    envelope = root_fade_envelope(t)
    curl = _curl_offset(
        backbone,
        normals,
        directions,
        tangents,
        curl_radius,
        curl_turns,
        curl_phase,
        local_frames=local_frames,
        t=t,
        envelope=envelope,
    )
    frizz = _frizz_offset(
        backbone,
        normals,
        directions,
        tangents,
        frizz_amplitude,
        frizz_seed_phase,
        local_frames=local_frames,
        t=t,
        envelope=envelope,
    )
    return backbone + curl + frizz

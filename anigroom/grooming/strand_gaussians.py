"""Differentiable groom parameters and strand-to-Gaussian conversion.

This module is deliberately narrow: it maps mesh-surface roots and explicit
grooming controls to strand samples, then to gsplat-style Gaussian parameters.
It does not own rendering, UV storage, densification, or training policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


EPS = 1e-8


def _normalize(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)


def _inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def decode_positive_asinh_ratio(
    raw: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Decode a positive field without a physical minimum or maximum.

    ``raw == 0`` returns the data-derived reference exactly.  ``asinh`` keeps
    the local log-ratio coordinate linear while avoiding exponential growth in
    the far raw-coordinate tail.
    """

    tiny = torch.as_tensor(
        torch.finfo(reference.dtype).tiny,
        device=reference.device,
        dtype=reference.dtype,
    )
    return reference.clamp_min(tiny) * torch.exp(torch.asinh(raw))


def encode_positive_asinh_ratio(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Encode a positive physical field relative to its local reference."""

    tiny = torch.as_tensor(
        torch.finfo(value.dtype).tiny,
        device=value.device,
        dtype=value.dtype,
    )
    log_ratio = torch.log(value.clamp_min(tiny) / reference.clamp_min(tiny))
    return torch.sinh(log_ratio)


def decode_positive_asinh(raw: torch.Tensor) -> torch.Tensor:
    """Decode a strictly positive scalar with neutral value one."""

    return torch.exp(torch.asinh(raw))


def encode_positive_asinh(value: torch.Tensor) -> torch.Tensor:
    """Encode a strictly positive scalar whose neutral value is one."""

    tiny = torch.as_tensor(
        torch.finfo(value.dtype).tiny,
        device=value.device,
        dtype=value.dtype,
    )
    return torch.sinh(torch.log(value.clamp_min(tiny)))


@dataclass(frozen=True)
class GroomRanges:
    """Physical ranges that remain intrinsic to bounded groom controls."""

    curl_radius: tuple[float, float] = (0.0, 0.030)
    curl_frequency: tuple[float, float] = (0.0, 8.0)
    frizz: tuple[float, float] = (0.0, 0.018)
    clump_strength: tuple[float, float] = (0.0, 1.0)


@dataclass
class DecodedGroom:
    """Decoded per-root groom controls.

    All tensors are shaped ``[R, C]``.  These are explicit editor-like controls:
    length, tapering width, a normalized 3D direction in the root frame, brush
    curve strength, bend, curl, frizz, child layout, color, and opacity. Extra
    growth gates and color-darkening shortcuts are intentionally outside this
    core parameter set.
    """

    length: torch.Tensor
    root_width: torch.Tensor
    tip_width: torch.Tensor
    width_taper: torch.Tensor
    direction_local: torch.Tensor
    brush_curve_strength: torch.Tensor
    bend: torch.Tensor
    curl_radius: torch.Tensor
    curl_frequency: torch.Tensor
    curl_phase: torch.Tensor
    frizz: torch.Tensor
    child_radius: torch.Tensor
    clump_strength: torch.Tensor
    root_color: torch.Tensor
    tip_color: torch.Tensor
    root_opacity: torch.Tensor
    tip_opacity: torch.Tensor
    opacity: torch.Tensor


@dataclass
class StrandGaussianOutput:
    """Flattened Gaussian parameters produced from strand segments."""

    means: torch.Tensor
    directions: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    root_indices: torch.Tensor
    segment_indices: torch.Tensor


@dataclass
class ResampledStrands:
    """Strand samples after adaptive contiguous arc-length resampling."""

    strands: torch.Tensor
    widths: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    segment_mask: torch.Tensor
    segment_counts: torch.Tensor
    stats: dict[str, float | int]


class GroomParameterField(nn.Module):
    """Trainable explicit grooming parameters for one strand per root."""

    def __init__(
        self,
        root_count: int,
        ranges: GroomRanges | None = None,
        init_length: float = 0.050,
        init_root_color: tuple[float, float, float] = (0.92, 0.90, 0.84),
        init_tip_color: tuple[float, float, float] = (0.86, 0.85, 0.78),
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if root_count <= 0:
            raise ValueError("root_count must be positive")
        self.root_count = int(root_count)
        self.ranges = ranges or GroomRanges()
        dev = torch.device(device) if device is not None else None

        def raw_from_range(value: float, bounds: tuple[float, float]) -> torch.Tensor:
            lo, hi = bounds
            rel = (float(value) - lo) / max(hi - lo, EPS)
            return _inverse_sigmoid(torch.tensor(rel, dtype=torch.float32, device=dev))

        def repeated(value: torch.Tensor | float, channels: int = 1) -> nn.Parameter:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=dev).reshape(1, channels)
            return nn.Parameter(tensor.repeat(self.root_count, 1))

        if not float(init_length) > 0.0:
            raise ValueError("init_length must be positive")
        self.register_buffer(
            "length_reference",
            torch.full(
                (self.root_count, 1),
                float(init_length),
                dtype=torch.float32,
                device=dev,
            ),
        )
        self.length_raw = repeated(0.0)
        self.register_buffer(
            "root_width_reference",
            torch.full(
                (self.root_count, 1),
                0.00065,
                dtype=torch.float32,
                device=dev,
            ),
        )
        self.root_width_raw = repeated(0.0)
        self.tip_width_ratio_raw = repeated(_inverse_sigmoid(torch.tensor(0.22, device=dev)))
        self.width_taper_raw = repeated(encode_positive_asinh(torch.tensor(1.25, device=dev)))
        self.direction_local_raw = nn.Parameter(
            torch.tensor([[0.55, 0.04, 0.22]], dtype=torch.float32, device=dev).repeat(self.root_count, 1)
        )
        self.brush_curve_strength_raw = repeated(0.0)
        self.bend_raw = repeated(0.0)
        self.curl_radius_raw = repeated(raw_from_range(0.001, self.ranges.curl_radius))
        self.curl_frequency_raw = repeated(raw_from_range(0.35, self.ranges.curl_frequency))
        self.curl_phase = nn.Parameter(torch.zeros((self.root_count, 1), dtype=torch.float32, device=dev))
        self.frizz_raw = repeated(raw_from_range(0.001, self.ranges.frizz))
        self.register_buffer(
            "child_radius_reference",
            torch.full(
                (self.root_count, 1),
                0.001,
                dtype=torch.float32,
                device=dev,
            ),
        )
        self.child_radius_raw = repeated(0.0)
        self.clump_strength_raw = repeated(raw_from_range(0.15, self.ranges.clump_strength))
        self.root_color_raw = nn.Parameter(_inverse_sigmoid(torch.tensor(init_root_color, device=dev)).view(1, 3).repeat(self.root_count, 1))
        self.tip_color_raw = nn.Parameter(_inverse_sigmoid(torch.tensor(init_tip_color, device=dev)).view(1, 3).repeat(self.root_count, 1))
        self.opacity_raw = repeated(_inverse_sigmoid(torch.tensor(0.72, device=dev)))
        self.tip_opacity_ratio_raw = repeated(_inverse_sigmoid(torch.tensor(0.68, device=dev)))

    @staticmethod
    def _decode_range(raw: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + (hi - lo) * torch.sigmoid(raw)

    def decode(self) -> DecodedGroom:
        ranges = self.ranges
        root_width = decode_positive_asinh_ratio(
            self.root_width_raw,
            self.root_width_reference,
        )
        tip_ratio = torch.sigmoid(self.tip_width_ratio_raw)
        return DecodedGroom(
            length=decode_positive_asinh_ratio(
                self.length_raw,
                self.length_reference,
            ),
            root_width=root_width,
            tip_width=root_width * tip_ratio,
            width_taper=decode_positive_asinh(self.width_taper_raw),
            direction_local=_normalize(self.direction_local_raw),
            brush_curve_strength=torch.sigmoid(self.brush_curve_strength_raw),
            bend=self.bend_raw,
            curl_radius=self._decode_range(self.curl_radius_raw, ranges.curl_radius),
            curl_frequency=self._decode_range(self.curl_frequency_raw, ranges.curl_frequency),
            curl_phase=self.curl_phase,
            frizz=self._decode_range(self.frizz_raw, ranges.frizz),
            child_radius=decode_positive_asinh_ratio(
                self.child_radius_raw,
                self.child_radius_reference,
            ),
            clump_strength=self._decode_range(self.clump_strength_raw, ranges.clump_strength),
            root_color=torch.sigmoid(self.root_color_raw),
            tip_color=torch.sigmoid(self.tip_color_raw),
            root_opacity=torch.sigmoid(self.opacity_raw),
            tip_opacity=torch.sigmoid(self.opacity_raw)
            * torch.sigmoid(self.tip_opacity_ratio_raw),
            opacity=torch.sigmoid(self.opacity_raw),
        )


def make_tangent_frames(normals: torch.Tensor, preferred_axis: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build stable tangent and bitangent vectors from surface normals."""

    normals = _normalize(normals)
    if preferred_axis is None:
        preferred_axis = normals.new_tensor([0.0, 1.0, 0.0])
    preferred_axis = preferred_axis.to(device=normals.device, dtype=normals.dtype).view(1, 3)
    fallback = normals.new_tensor([[1.0, 0.0, 0.0]])
    use_fallback = (torch.abs((normals * preferred_axis).sum(dim=-1, keepdim=True)) > 0.92).expand(-1, 3)
    axis = torch.where(use_fallback, fallback.expand_as(normals), preferred_axis.expand_as(normals))
    tangent = _normalize(torch.cross(axis, normals, dim=-1))
    bitangent = _normalize(torch.cross(normals, tangent, dim=-1))
    return tangent, bitangent


def build_brush_centerline(
    roots: torch.Tensor,
    normals: torch.Tensor,
    directions: torch.Tensor,
    lengths: torch.Tensor,
    brush_curve_strength: torch.Tensor,
    samples: int,
) -> torch.Tensor:
    """Build a smooth normal-to-groom curve with fixed root and tip.

    ``directions`` and ``lengths`` retain their direct editor meaning:
    ``tip = root + length * direction``.  A strength of zero reproduces that
    exact straight segment.  Increasing strength accumulates the endpoint's
    normal displacement earlier and its tangent displacement later without
    changing either endpoint.
    """

    if samples < 2:
        raise ValueError("samples must be at least 2")
    if roots.shape != normals.shape or roots.shape != directions.shape:
        raise ValueError("roots, normals, and directions must all have shape [R, 3]")
    if lengths.shape != (roots.shape[0], 1):
        raise ValueError("lengths must have shape [R, 1]")
    if brush_curve_strength.shape != (roots.shape[0], 1):
        raise ValueError("brush_curve_strength must have shape [R, 1]")

    normals = _normalize(normals)
    directions = _normalize(directions)
    delta = lengths * directions
    normal_delta = (delta * normals).sum(dim=-1, keepdim=True) * normals
    tangent_delta = delta - normal_delta

    t = torch.linspace(
        0.0,
        1.0,
        samples,
        device=roots.device,
        dtype=roots.dtype,
    ).view(1, samples, 1)
    strength = brush_curve_strength.clamp(0.0, 1.0)[:, None]
    transition = t * (1.0 - t)
    normal_progress = t + strength * transition
    tangent_progress = t - strength * transition
    return (
        roots[:, None]
        + normal_progress * normal_delta[:, None]
        + tangent_progress * tangent_delta[:, None]
    )


def build_strands(
    roots: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    groom: DecodedGroom,
    samples: int,
    shape_normal_mode: str = "full",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate differentiable strand samples from mesh roots and groom controls.

    Returns ``points, widths, colors, opacities`` with shapes
    ``[R, samples, 3]``, ``[R, samples, 1]``, ``[R, samples, 3]``, and
    ``[R, samples, 1]``.
    """

    if samples < 2:
        raise ValueError("samples must be at least 2")
    roots = roots.to(dtype=normals.dtype, device=normals.device)
    normals = _normalize(normals)
    tangents = _normalize(tangents)
    bitangents = _normalize(bitangents)
    if roots.shape != normals.shape or roots.shape != tangents.shape or roots.shape != bitangents.shape:
        raise ValueError("roots, normals, tangents, and bitangents must all have shape [R, 3]")

    direction_local = _normalize(groom.direction_local)
    groom_direction = _normalize(
        direction_local[:, [0]] * tangents
        + direction_local[:, [1]] * bitangents
        + direction_local[:, [2]] * normals
    )
    tangent_component = groom_direction - (groom_direction * normals).sum(dim=-1, keepdim=True) * normals
    tangent_norm = torch.linalg.norm(tangent_component, dim=-1, keepdim=True)
    flow = torch.where(tangent_norm > EPS, tangent_component / tangent_norm.clamp_min(EPS), tangents)

    if shape_normal_mode not in {"full", "outward", "tangent"}:
        raise ValueError(f"unknown shape_normal_mode: {shape_normal_mode}")
    side = _normalize(torch.cross(normals, flow, dim=-1))
    curl_up = _normalize(torch.cross(flow, side, dim=-1))
    t = torch.linspace(0.0, 1.0, samples, device=roots.device, dtype=roots.dtype).view(1, samples, 1)
    points = build_brush_centerline(
        roots,
        normals,
        groom_direction,
        groom.length,
        groom.brush_curve_strength,
        samples,
    )
    bend_envelope = 16.0 * t.square() * (1.0 - t).square()
    bend_offset = (
        groom.length[:, None]
        * groom.bend[:, None]
        * bend_envelope
        * side[:, None]
    )
    points = points + bend_offset
    phase = 2.0 * torch.pi * groom.curl_frequency[:, None] * t + groom.curl_phase[:, None]
    curl_envelope = torch.sin(0.5 * torch.pi * t).clamp(0.0, 1.0)
    curl_side = torch.sin(phase)
    curl_normal = torch.cos(phase)
    if shape_normal_mode == "outward":
        curl_normal = torch.relu(curl_normal)
    elif shape_normal_mode == "tangent":
        curl_normal = torch.zeros_like(curl_normal)
    curl_offset = groom.curl_radius[:, None] * curl_envelope * (
        curl_side * side[:, None] + curl_normal * curl_up[:, None]
    )
    frizz_phase = 2.0 * torch.pi * (3.0 * groom.curl_frequency[:, None] + 1.0) * t + 1.618 * groom.curl_phase[:, None]
    frizz_envelope = (t * (1.0 - 0.35 * t)).clamp(0.0, 1.0)
    frizz_normal = torch.sin(1.7 * frizz_phase + 0.3)
    if shape_normal_mode == "outward":
        frizz_normal = torch.relu(frizz_normal)
    elif shape_normal_mode == "tangent":
        frizz_normal = torch.zeros_like(frizz_normal)
    frizz_offset = groom.frizz[:, None] * frizz_envelope * (
        0.65 * torch.sin(frizz_phase) * side[:, None]
        + 0.35 * frizz_normal * curl_up[:, None]
    )
    points = points + curl_offset + frizz_offset

    taper_t = t.clamp(0.0, 1.0).pow(groom.width_taper[:, None])
    widths = groom.root_width[:, None] * (1.0 - taper_t) + groom.tip_width[:, None] * taper_t
    colors = groom.root_color[:, None] * (1.0 - t) + groom.tip_color[:, None] * t
    opacities = groom.root_opacity[:, None] * (1.0 - t) + groom.tip_opacity[:, None] * t
    return points, widths, colors, opacities


def expand_child_strands(
    strands: torch.Tensor,
    widths: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    root_normals: torch.Tensor,
    child_radius: torch.Tensor,
    clump_strength: torch.Tensor,
    child_count: int,
    child_width_scale: float = 0.82,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand guide strands into deterministic child strands.

    ``child_count`` is a discrete structural choice.  The continuous controls
    ``child_radius`` and ``clump_strength`` remain differentiable and are
    decoded from the groom field.  The first child is the guide strand itself;
    additional children start from tangent-plane offsets and converge toward
    the guide according to ``clump_strength``.
    """

    if child_count <= 1:
        root_ids = torch.arange(strands.shape[0], device=strands.device)
        return strands, widths, colors, opacities, root_ids
    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape [R, S, 3]")
    if root_normals.shape != (strands.shape[0], 3):
        raise ValueError("root_normals must have shape [R, 3]")
    if child_radius.shape[:1] != (strands.shape[0],) or clump_strength.shape[:1] != (strands.shape[0],):
        raise ValueError("child_radius and clump_strength must have one value per root")

    roots_n, samples, _ = strands.shape
    device = strands.device
    dtype = strands.dtype
    normals = _normalize(root_normals.to(device=device, dtype=dtype))
    root_tangent = _normalize(strands[:, 1] - strands[:, 0])
    side = _normalize(torch.cross(normals, root_tangent, dim=-1))
    tangent = _normalize(torch.cross(side, normals, dim=-1))

    child_ids = torch.arange(child_count, device=device, dtype=dtype)
    if child_count == 2:
        radius = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        angles = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
    else:
        outer = child_ids[1:]
        golden = torch.tensor(2.39996322972865332, device=device, dtype=dtype)
        angles_outer = outer * golden
        radius_outer = torch.sqrt(outer / max(float(child_count - 1), 1.0))
        angles = torch.cat([torch.zeros(1, device=device, dtype=dtype), angles_outer], dim=0)
        radius = torch.cat([torch.zeros(1, device=device, dtype=dtype), radius_outer], dim=0)

    offsets = (
        torch.cos(angles).view(1, child_count, 1) * tangent[:, None]
        + torch.sin(angles).view(1, child_count, 1) * side[:, None]
    )
    offsets = offsets * (child_radius.view(roots_n, 1, 1) * radius.view(1, child_count, 1))

    t = torch.linspace(0.0, 1.0, samples, device=device, dtype=dtype).view(1, 1, samples, 1)
    clump = clump_strength.view(roots_n, 1, 1, 1).clamp(0.0, 1.0)
    offset_envelope = 1.0 - clump * t.pow(1.35)
    child_strands = strands[:, None] + offsets[:, :, None] * offset_envelope

    width_scale = torch.ones((1, child_count, 1, 1), device=device, dtype=dtype)
    width_scale[:, 1:] = float(child_width_scale)
    child_widths = widths[:, None] * width_scale
    child_colors = colors[:, None].expand(roots_n, child_count, samples, 3)
    child_opacities = opacities[:, None].expand(roots_n, child_count, samples, 1)

    root_ids = torch.arange(roots_n, device=device)[:, None].expand(roots_n, child_count).reshape(-1)
    return (
        child_strands.reshape(roots_n * child_count, samples, 3),
        child_widths.reshape(roots_n * child_count, samples, 1),
        child_colors.reshape(roots_n * child_count, samples, 3),
        child_opacities.reshape(roots_n * child_count, samples, 1),
        root_ids,
    )


def strand_segment_budgets(
    strands: torch.Tensor,
    lengths: torch.Tensor,
    min_segments: int,
    length_origin: float,
    segments_per_unit_length: float,
    segments_per_unit_complexity: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Choose an uncapped segment budget from absolute length and complexity.

    The returned budget is the number of *consecutive* Gaussian segments that
    should represent each strand.  This is the accepted absolute-length linear
    allocator with its upper clamps removed: equal physical lengths receive the
    same length contribution, while curvature can only add representation.
    """

    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape [R, S, 3]")
    if strands.shape[1] < 2:
        raise ValueError("strands must contain at least two samples")
    if lengths.shape[0] != strands.shape[0]:
        raise ValueError("lengths must have one row per strand")
    min_segments = int(max(1, min_segments))
    length_origin = float(length_origin)
    segments_per_unit_length = float(segments_per_unit_length)
    segments_per_unit_complexity = float(segments_per_unit_complexity)
    if length_origin < 0.0:
        raise ValueError("length_origin must be non-negative")
    if segments_per_unit_length <= 0.0 or segments_per_unit_complexity < 0.0:
        raise ValueError("segment densities must be positive for length and non-negative for complexity")

    seg = strands[:, 1:] - strands[:, :-1]
    seg_length = torch.linalg.norm(seg, dim=-1)
    arc = seg_length.sum(dim=1, keepdim=True)
    chord = torch.linalg.norm(strands[:, -1] - strands[:, 0], dim=-1, keepdim=True).clamp_min(EPS)
    arc_excess = (arc / chord - 1.0).clamp_min(0.0)
    if seg.shape[1] > 1:
        dirs = _normalize(seg)
        turn = 1.0 - (dirs[:, 1:] * dirs[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
        valid_turn = (seg_length[:, 1:] > EPS) & (seg_length[:, :-1] > EPS)
        turn = torch.where(valid_turn, turn, torch.zeros_like(turn))
        turn = turn.sum(dim=1, keepdim=True) / valid_turn.sum(dim=1, keepdim=True).clamp_min(1)
        complexity = torch.maximum(arc_excess, 4.0 * turn)
    else:
        complexity = arc_excess

    absolute_length = lengths.detach().reshape(strands.shape[0], -1)[:, :1]
    length_extra = (absolute_length - length_origin).clamp_min(0.0) * segments_per_unit_length
    complexity_extra = complexity.detach().clamp_min(0.0) * segments_per_unit_complexity
    budgets = torch.round(float(min_segments) + length_extra + complexity_extra).long().view(-1)
    budgets = budgets.clamp_min(min_segments)

    stats = {
        "adaptive_mean_segments": float(budgets.float().mean().detach().cpu()),
        "adaptive_min_segments": int(budgets.min().detach().cpu()),
        "adaptive_max_segments": int(budgets.max().detach().cpu()),
        "adaptive_arc_length_mean": float(arc.mean().detach().cpu()),
        "adaptive_complexity_mean": float(complexity.mean().detach().cpu()),
        "adaptive_length_extra_mean": float(length_extra.mean().detach().cpu()),
        "adaptive_complexity_extra_mean": float(complexity_extra.mean().detach().cpu()),
    }
    return budgets, stats


def _gather_samples(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.gather(1, indices[..., None].expand(-1, -1, values.shape[-1]))


def resample_strands_to_segment_budgets(
    strands: torch.Tensor,
    widths: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    segment_counts: torch.Tensor,
) -> ResampledStrands:
    """Arc-length resample each strand into contiguous Gaussian segments.

    This is the formal bridge from a Blender/Groom-style polyline to 3DGS:
    every valid output segment is adjacent to the previous one. No segment is
    skipped, so rendered hair cannot become a dotted chain because of sampling.
    """

    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape [R, S, 3]")
    if widths.shape[:2] != strands.shape[:2] or colors.shape[:2] != strands.shape[:2] or opacities.shape[:2] != strands.shape[:2]:
        raise ValueError("widths/colors/opacities must match strand root and sample dimensions")
    if segment_counts.ndim != 1 or segment_counts.shape[0] != strands.shape[0]:
        raise ValueError("segment_counts must have shape [R]")
    if torch.any(segment_counts < 1):
        raise ValueError("all segment counts must be at least one")

    roots_n, samples, _ = strands.shape
    max_segments = int(segment_counts.max().detach().cpu())
    device = strands.device
    dtype = strands.dtype

    chord = strands[:, 1:] - strands[:, :-1]
    seg_len = torch.linalg.norm(chord, dim=-1).clamp_min(EPS)
    cumulative = torch.cat([torch.zeros(roots_n, 1, device=device, dtype=dtype), torch.cumsum(seg_len, dim=1)], dim=1)
    total = cumulative[:, -1:].clamp_min(EPS)

    out_ids = torch.arange(max_segments + 1, device=device)
    counts_f = segment_counts.to(device=device, dtype=dtype).view(-1, 1)
    rel = (out_ids.view(1, -1).to(dtype=dtype) / counts_f).clamp(0.0, 1.0)
    targets = rel * total

    upper = torch.searchsorted(cumulative.contiguous(), targets.contiguous(), right=True)
    upper = upper.clamp(1, samples - 1)
    lower = upper - 1

    lower_d = cumulative.gather(1, lower)
    upper_d = cumulative.gather(1, upper)
    w = ((targets - lower_d) / (upper_d - lower_d).clamp_min(EPS)).clamp(0.0, 1.0)

    def interp(values: torch.Tensor) -> torch.Tensor:
        lo = _gather_samples(values, lower)
        hi = _gather_samples(values, upper)
        return lo * (1.0 - w[..., None]) + hi * w[..., None]

    resampled_strands = interp(strands)
    resampled_widths = interp(widths)
    resampled_colors = interp(colors)
    resampled_opacities = interp(opacities)

    point_valid = out_ids.view(1, -1) <= segment_counts.view(-1, 1).to(device=device)
    segment_valid = out_ids[:max_segments].view(1, -1) < segment_counts.view(-1, 1).to(device=device)
    last_points = strands[:, -1:, :]
    resampled_strands = torch.where(point_valid[..., None], resampled_strands, last_points.expand_as(resampled_strands))
    resampled_widths = torch.where(point_valid[..., None], resampled_widths, widths[:, -1:, :].expand_as(resampled_widths))
    resampled_colors = torch.where(point_valid[..., None], resampled_colors, colors[:, -1:, :].expand_as(resampled_colors))
    resampled_opacities = torch.where(point_valid[..., None], resampled_opacities, opacities[:, -1:, :].expand_as(resampled_opacities))

    stats = {
        "adaptive_mean_segments": float(segment_counts.float().mean().detach().cpu()),
        "adaptive_min_segments": int(segment_counts.min().detach().cpu()),
        "adaptive_max_segments": int(segment_counts.max().detach().cpu()),
    }
    return ResampledStrands(
        strands=resampled_strands,
        widths=resampled_widths,
        colors=resampled_colors,
        opacities=resampled_opacities,
        segment_mask=segment_valid,
        segment_counts=segment_counts,
        stats=stats,
    )


def adaptive_resample_strands(
    strands: torch.Tensor,
    widths: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    lengths: torch.Tensor,
    min_segments: int,
    length_origin: float,
    segments_per_unit_length: float,
    segments_per_unit_complexity: float,
) -> ResampledStrands:
    """Adaptive but continuous strand sampling for 3DGS conversion."""

    counts, stats = strand_segment_budgets(
        strands,
        lengths,
        min_segments,
        length_origin,
        segments_per_unit_length,
        segments_per_unit_complexity,
    )
    resampled = resample_strands_to_segment_budgets(strands, widths, colors, opacities, counts)
    resampled.stats.update(stats)
    return resampled


def _quat_from_x_axis(direction: torch.Tensor) -> torch.Tensor:
    direction = _normalize(direction)
    x_axis = direction.new_tensor([1.0, 0.0, 0.0]).view(1, 3).expand_as(direction)
    cross = torch.cross(x_axis, direction, dim=-1)
    dot = (x_axis * direction).sum(dim=-1, keepdim=True)
    quat = torch.cat([1.0 + dot, cross], dim=-1)
    fallback = direction.new_tensor([0.0, 0.0, 1.0, 0.0]).view(1, 4).expand_as(quat)
    quat = torch.where((1.0 + dot).abs() < 1e-6, fallback, quat)
    return F.normalize(quat, dim=-1)


def strands_to_gaussians(
    strands: torch.Tensor,
    widths: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    segment_mask: torch.Tensor | None = None,
    strand_root_indices: torch.Tensor | None = None,
    width_floor: float = 1e-5,
    length_floor: float = 1e-5,
    length_overlap: float = 1.18,
) -> StrandGaussianOutput:
    """Convert strand segments to flattened anisotropic Gaussian parameters."""

    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape [R, S, 3]")
    if widths.shape[:2] != strands.shape[:2] or colors.shape[:2] != strands.shape[:2] or opacities.shape[:2] != strands.shape[:2]:
        raise ValueError("widths/colors/opacities must match strand root and sample dimensions")
    roots_n, samples, _ = strands.shape
    segment_total = samples - 1
    if segment_mask is None:
        segment_mask = torch.ones(roots_n, segment_total, dtype=torch.bool, device=strands.device)
    if segment_mask.shape != (roots_n, segment_total):
        raise ValueError("segment_mask must have shape [R, S - 1]")
    if strand_root_indices is None:
        guide_root_ids = torch.arange(roots_n, device=strands.device)
    else:
        if strand_root_indices.shape != (roots_n,):
            raise ValueError("strand_root_indices must have shape [R]")
        guide_root_ids = strand_root_indices.to(device=strands.device, dtype=torch.long)

    starts = strands[:, :-1]
    ends = strands[:, 1:]
    chords = ends - starts
    lengths = torch.linalg.norm(chords, dim=-1, keepdim=True).clamp_min(length_floor)
    directions = chords / lengths
    means = 0.5 * (starts + ends)
    widths_mid = (0.5 * (widths[:, :-1] + widths[:, 1:])).clamp_min(width_floor)
    colors_mid = (0.5 * (colors[:, :-1] + colors[:, 1:])).clamp(0.0, 1.0)
    opacities_mid = (0.5 * (opacities[:, :-1] + opacities[:, 1:])).clamp(0.0, 1.0)
    scales = torch.cat([0.5 * lengths * float(length_overlap), widths_mid, widths_mid], dim=-1)
    quats = _quat_from_x_axis(directions.reshape(-1, 3)).view(roots_n, segment_total, 4)

    root_ids = guide_root_ids[:, None].expand(roots_n, segment_total)
    seg_ids = torch.arange(segment_total, device=strands.device)[None, :].expand(roots_n, segment_total)
    keep = segment_mask & torch.isfinite(means).all(dim=-1) & torch.isfinite(scales).all(dim=-1)

    return StrandGaussianOutput(
        means=means[keep],
        directions=directions[keep],
        quats=quats[keep],
        scales=scales[keep],
        colors=colors_mid[keep],
        opacities=opacities_mid[keep],
        root_indices=root_ids[keep],
        segment_indices=seg_ids[keep],
    )

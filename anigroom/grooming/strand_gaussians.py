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


@dataclass(frozen=True)
class GroomRanges:
    """Physical ranges for decoded grooming controls."""

    length: tuple[float, float] = (0.012, 0.105)
    root_width: tuple[float, float] = (0.00008, 0.0020)
    tip_width_ratio: tuple[float, float] = (0.10, 0.85)
    width_taper: tuple[float, float] = (0.45, 3.00)
    flow_strength: tuple[float, float] = (0.05, 1.10)
    lift: tuple[float, float] = (0.04, 0.55)
    sag: tuple[float, float] = (0.00, 0.85)
    stiffness: tuple[float, float] = (0.05, 0.98)
    curl_radius: tuple[float, float] = (0.0, 0.030)
    curl_frequency: tuple[float, float] = (0.0, 8.0)
    frizz: tuple[float, float] = (0.0, 0.018)
    child_radius: tuple[float, float] = (0.0, 0.018)
    clump_strength: tuple[float, float] = (0.0, 1.0)
    opacity: tuple[float, float] = (0.05, 0.95)
    tip_opacity_ratio: tuple[float, float] = (0.10, 1.00)


@dataclass
class DecodedGroom:
    """Decoded per-root groom controls.

    All tensors are shaped ``[R, C]``.  These are explicit editor-like controls:
    length, tapering width, brushed flow, lift, bend, sag, stiffness, color, and
    opacity. Extra growth gates and color-darkening shortcuts are intentionally
    outside this core parameter set.
    """

    length: torch.Tensor
    root_width: torch.Tensor
    tip_width: torch.Tensor
    width_taper: torch.Tensor
    flow_xy: torch.Tensor
    flow_strength: torch.Tensor
    lift: torch.Tensor
    bend: torch.Tensor
    sag: torch.Tensor
    stiffness: torch.Tensor
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

        self.length_raw = repeated(raw_from_range(0.050, self.ranges.length))
        self.root_width_raw = repeated(raw_from_range(0.00065, self.ranges.root_width))
        self.tip_width_ratio_raw = repeated(raw_from_range(0.22, self.ranges.tip_width_ratio))
        self.width_taper_raw = repeated(raw_from_range(1.25, self.ranges.width_taper))
        self.flow_xy = nn.Parameter(torch.tensor([[0.55, 0.04]], dtype=torch.float32, device=dev).repeat(self.root_count, 1))
        self.flow_strength_raw = repeated(raw_from_range(0.72, self.ranges.flow_strength))
        self.lift_raw = repeated(raw_from_range(0.22, self.ranges.lift))
        self.bend_raw = repeated(-0.20)
        self.sag_raw = repeated(raw_from_range(0.20, self.ranges.sag))
        self.stiffness_raw = repeated(raw_from_range(0.72, self.ranges.stiffness))
        self.curl_radius_raw = repeated(raw_from_range(0.001, self.ranges.curl_radius))
        self.curl_frequency_raw = repeated(raw_from_range(0.35, self.ranges.curl_frequency))
        self.curl_phase = nn.Parameter(torch.zeros((self.root_count, 1), dtype=torch.float32, device=dev))
        self.frizz_raw = repeated(raw_from_range(0.001, self.ranges.frizz))
        self.child_radius_raw = repeated(raw_from_range(0.001, self.ranges.child_radius))
        self.clump_strength_raw = repeated(raw_from_range(0.15, self.ranges.clump_strength))
        self.root_color_raw = nn.Parameter(_inverse_sigmoid(torch.tensor(init_root_color, device=dev)).view(1, 3).repeat(self.root_count, 1))
        self.tip_color_raw = nn.Parameter(_inverse_sigmoid(torch.tensor(init_tip_color, device=dev)).view(1, 3).repeat(self.root_count, 1))
        self.opacity_raw = repeated(raw_from_range(0.72, self.ranges.opacity))
        self.tip_opacity_ratio_raw = repeated(raw_from_range(0.68, self.ranges.tip_opacity_ratio))

    @staticmethod
    def _decode_range(raw: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + (hi - lo) * torch.sigmoid(raw)

    def decode(self) -> DecodedGroom:
        ranges = self.ranges
        root_width = self._decode_range(self.root_width_raw, ranges.root_width)
        tip_ratio = self._decode_range(self.tip_width_ratio_raw, ranges.tip_width_ratio)
        return DecodedGroom(
            length=self._decode_range(self.length_raw, ranges.length),
            root_width=root_width,
            tip_width=root_width * tip_ratio,
            width_taper=self._decode_range(self.width_taper_raw, ranges.width_taper),
            flow_xy=self.flow_xy,
            flow_strength=self._decode_range(self.flow_strength_raw, ranges.flow_strength),
            lift=self._decode_range(self.lift_raw, ranges.lift),
            bend=torch.tanh(self.bend_raw),
            sag=self._decode_range(self.sag_raw, ranges.sag),
            stiffness=self._decode_range(self.stiffness_raw, ranges.stiffness),
            curl_radius=self._decode_range(self.curl_radius_raw, ranges.curl_radius),
            curl_frequency=self._decode_range(self.curl_frequency_raw, ranges.curl_frequency),
            curl_phase=self.curl_phase,
            frizz=self._decode_range(self.frizz_raw, ranges.frizz),
            child_radius=self._decode_range(self.child_radius_raw, ranges.child_radius),
            clump_strength=self._decode_range(self.clump_strength_raw, ranges.clump_strength),
            root_color=torch.sigmoid(self.root_color_raw),
            tip_color=torch.sigmoid(self.tip_color_raw),
            root_opacity=self._decode_range(self.opacity_raw, ranges.opacity),
            tip_opacity=self._decode_range(self.opacity_raw, ranges.opacity)
            * self._decode_range(self.tip_opacity_ratio_raw, ranges.tip_opacity_ratio),
            opacity=self._decode_range(self.opacity_raw, ranges.opacity),
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


def build_strands(
    roots: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    groom: DecodedGroom,
    samples: int,
    gravity_direction: torch.Tensor | tuple[float, float, float] = (0.0, -1.0, 0.0),
    use_gravity_sag: bool = True,
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

    flow_local = _normalize(groom.flow_xy)
    flow = _normalize(flow_local[:, [0]] * tangents + flow_local[:, [1]] * bitangents)

    side = _normalize(torch.cross(normals, flow, dim=-1))
    curl_up = _normalize(torch.cross(flow, side, dim=-1))
    flex = (1.0 - groom.stiffness).clamp(0.0, 1.0)
    length_lo, length_hi = GroomRanges().length
    length_norm = ((groom.length - length_lo) / max(length_hi - length_lo, EPS)).clamp(0.0, 1.0)
    if use_gravity_sag:
        gravity = torch.as_tensor(gravity_direction, device=roots.device, dtype=roots.dtype).view(1, 3)
        gravity = _normalize(gravity.expand_as(roots))
        gravity_tangent = gravity - (gravity * normals).sum(dim=-1, keepdim=True) * normals
        gravity_tangent = _normalize(gravity_tangent + 1e-4 * flow)
        sag = groom.sag * flex * (0.25 + 1.75 * length_norm.square())
    else:
        gravity_tangent = torch.zeros_like(flow)
        sag = torch.zeros_like(groom.sag)
    bend = groom.bend * (0.35 + 0.65 * flex)
    brush = groom.flow_strength

    p0 = roots
    p1 = roots + groom.length * (
        groom.lift * normals
        + brush * flow
        + sag * gravity_tangent
        + 0.24 * bend * side
    )
    m0 = groom.length * _normalize((0.75 + 0.60 * groom.lift) * normals + 0.18 * brush * flow)
    m1 = groom.length * (
        0.12 * normals
        + brush * flow
        + sag * gravity_tangent
        + 0.55 * bend * side
    )

    t = torch.linspace(0.0, 1.0, samples, device=roots.device, dtype=roots.dtype).view(1, samples, 1)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    points = h00 * p0[:, None] + h10 * m0[:, None] + h01 * p1[:, None] + h11 * m1[:, None]
    phase = 2.0 * torch.pi * groom.curl_frequency[:, None] * t + groom.curl_phase[:, None]
    curl_envelope = torch.sin(0.5 * torch.pi * t).clamp(0.0, 1.0)
    curl_offset = groom.curl_radius[:, None] * curl_envelope * (
        torch.sin(phase) * side[:, None] + torch.cos(phase) * curl_up[:, None]
    )
    frizz_phase = 2.0 * torch.pi * (3.0 * groom.curl_frequency[:, None] + 1.0) * t + 1.618 * groom.curl_phase[:, None]
    frizz_envelope = (t * (1.0 - 0.35 * t)).clamp(0.0, 1.0)
    frizz_offset = groom.frizz[:, None] * frizz_envelope * (
        0.65 * torch.sin(frizz_phase) * side[:, None]
        + 0.35 * torch.sin(1.7 * frizz_phase + 0.3) * curl_up[:, None]
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
    max_segments: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Choose a continuous segment budget from strand length and curvature.

    The returned budget is the number of *consecutive* Gaussian segments that
    should represent each strand. It does not skip existing segments.
    """

    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape [R, S, 3]")
    if strands.shape[1] < 2:
        raise ValueError("strands must contain at least two samples")
    if lengths.shape[0] != strands.shape[0]:
        raise ValueError("lengths must have one row per strand")

    min_segments = int(max(1, min_segments))
    max_segments = int(max(min_segments, max_segments))

    seg = strands[:, 1:] - strands[:, :-1]
    arc = torch.linalg.norm(seg, dim=-1).sum(dim=1, keepdim=True)
    chord = torch.linalg.norm(strands[:, -1] - strands[:, 0], dim=-1, keepdim=True).clamp_min(EPS)
    arc_excess = (arc / chord - 1.0).clamp_min(0.0)
    if seg.shape[1] > 1:
        dirs = _normalize(seg)
        turn = 1.0 - (dirs[:, 1:] * dirs[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)
        turn = turn.mean(dim=1, keepdim=True)
        complexity = torch.maximum(arc_excess, 4.0 * turn)
    else:
        complexity = arc_excess

    length_bounds = GroomRanges().length
    length_abs = ((lengths.detach() - length_bounds[0]) / max(length_bounds[1] - length_bounds[0], EPS)).clamp(0.0, 1.0)
    complexity_abs = (complexity.detach() / 0.35).clamp(0.0, 1.0)
    score = (0.68 * length_abs + 0.32 * complexity_abs).clamp(0.0, 1.0)
    budgets = torch.round(min_segments + score * (max_segments - min_segments)).long().view(-1)
    budgets = budgets.clamp(min_segments, max_segments)

    stats = {
        "adaptive_mean_segments": float(budgets.float().mean().detach().cpu()),
        "adaptive_min_segments": int(budgets.min().detach().cpu()),
        "adaptive_max_segments": int(budgets.max().detach().cpu()),
        "adaptive_complexity_mean": float(complexity.mean().detach().cpu()),
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
    max_segments: int,
) -> ResampledStrands:
    """Adaptive but continuous strand sampling for 3DGS conversion."""

    counts, stats = strand_segment_budgets(strands, lengths, min_segments, max_segments)
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

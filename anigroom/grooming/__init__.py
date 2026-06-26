"""Differentiable grooming primitives for AniGroom."""

from .strand_gaussians import (
    DecodedGroom,
    GroomParameterField,
    GroomRanges,
    ResampledStrands,
    StrandGaussianOutput,
    adaptive_resample_strands,
    build_strands,
    expand_child_strands,
    make_tangent_frames,
    resample_strands_to_segment_budgets,
    strand_segment_budgets,
    strands_to_gaussians,
)

__all__ = [
    "DecodedGroom",
    "GroomParameterField",
    "GroomRanges",
    "ResampledStrands",
    "StrandGaussianOutput",
    "adaptive_resample_strands",
    "build_strands",
    "expand_child_strands",
    "make_tangent_frames",
    "resample_strands_to_segment_budgets",
    "strand_segment_budgets",
    "strands_to_gaussians",
]

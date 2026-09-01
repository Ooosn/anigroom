"""Isolated guide-attribute Gaussian-field primitives."""

from .config import GuideGaussianFieldConfig
from .field import (
    GuideAttributeGaussianField,
    GuideGaussianWeights,
    c2_gaussian_taper,
)
from .initialization import (
    GuideGaussianBinding,
    initialize_guide_gaussian_binding,
)

__all__ = [
    "GuideGaussianFieldConfig",
    "GuideGaussianBinding",
    "GuideGaussianWeights",
    "GuideAttributeGaussianField",
    "initialize_guide_gaussian_binding",
    "c2_gaussian_taper",
]

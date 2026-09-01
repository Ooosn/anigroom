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
from .hierarchy import NestedTopologyFPS, density_preserving_topology_fps

__all__ = [
    "GuideGaussianFieldConfig",
    "GuideGaussianBinding",
    "GuideGaussianWeights",
    "GuideAttributeGaussianField",
    "NestedTopologyFPS",
    "density_preserving_topology_fps",
    "initialize_guide_gaussian_binding",
    "c2_gaussian_taper",
]

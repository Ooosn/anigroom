from __future__ import annotations

import numpy as np
import torch

from anigroom.grooming.guide_attribute_gaussian_field import (
    GuideAttributeGaussianField,
    GuideGaussianFieldConfig,
)
from tools.diagnose_r085_shared_gaussian_arms import (
    circular_gradient,
    fixed_radius_binding,
)
from tools.visualize_white_tiger_groom_attributes import _hsv_unit_value_rgb


def test_fixed_radius_binding_covers_and_reproduces_constant() -> None:
    x, y = np.meshgrid(np.arange(3), np.arange(3), indexing="xy")
    points = torch.as_tensor(
        np.stack((x.reshape(-1), y.reshape(-1), np.zeros(9)), axis=1),
        dtype=torch.float32,
    )
    config = GuideGaussianFieldConfig(neighbor_count=2)
    binding = fixed_radius_binding(
        points,
        points,
        np.full((9,), 2.0, dtype=np.float64),
        config,
    )
    field = GuideAttributeGaussianField(binding)
    output = field(torch.full((9,), 0.37))

    assert binding.report["all_queries_covered"] is True
    torch.testing.assert_close(output, torch.full((9,), 0.37), atol=1.0e-6, rtol=0.0)


def test_circular_gradient_uses_short_hue_wrap() -> None:
    hue = np.asarray([[359.0 / 360.0, 1.0 / 360.0]], dtype=np.float32)
    colors = _hsv_unit_value_rgb(hue)
    gradient, valid, report = circular_gradient(
        colors,
        np.ones((1, 2), dtype=bool),
    )

    assert bool(valid.all())
    assert float(gradient.max()) <= 2.1
    assert float(report["max"]) <= 2.1

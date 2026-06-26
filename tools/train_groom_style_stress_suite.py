"""Validate explicit groom-parameter optimization on synthetic animal coats.

This script is a diagnostic stress suite, not a production training recipe.
It checks whether the differentiable groom parameterization can recover common
animal-fur fields from projected initialization. Iteration counts here are
debug budgets for observing convergence trends; real reconstruction uses the
full training pipeline with camera/data losses, root lifecycle, and long runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from anigroom.grooming import GroomParameterField, GroomRanges  # noqa: E402
from train_dense_groom_patch_teacher_student import (  # noqa: E402
    dense_ranges,
    flow_coherence_loss,
    make_optimizer,
    make_student_like_dense,
)
from train_groom_layer_teacher_student import (  # noqa: E402
    build_render_inputs,
    image_loss,
    loss_components,
    make_sheet,
    orientation_detail_loss,
    orientation_flow_loss,
    orientation_to_pil,
    psnr,
    render,
    set_color,
    set_range,
    to_pil,
)
from visualize_dense_groom_patch import make_dense_roots, make_field  # noqa: E402


STYLE_NAMES = [
    "base",
    "longer",
    "brushed_color",
    "curled",
    "wavy",
    "frizzy",
    "clumped",
    "short_plush",
    "mixed_animal",
    "wet_matted",
    "dog_guard",
    "tiger_plush",
    "long_silky",
    "curly_ringlet",
    "mane_ridge",
    "coarse_guard",
    "cowlick_whorl",
    "patchy_length",
    "dirty_tangled",
    "fine_undercoat",
    "spiky_guard",
    "side_parted",
    "ringlet_plus_undercoat",
    "facial_short_to_long",
    "realistic_mixed_coat",
]

INIT_MODE_NAMES = {
    "generic",
    "projected_flow",
    "projected_shape",
    "projected_curl",
    "projected_curve",
}


def enrich_teacher_style(style: str, roots: torch.Tensor, ranges: GroomRanges) -> GroomParameterField:
    if style in {"base", "longer", "taper", "curled", "brushed_color"}:
        return make_field(style, roots, ranges).eval()

    device = roots.device
    field = make_field("base", roots, ranges)
    x = roots[:, [0]]
    y = roots[:, [1]]
    stripe = torch.sigmoid(8.0 * (torch.sin(18.0 * x + 10.5 * y) - 0.30))

    with torch.no_grad():
        if style == "wavy":
            set_range(field.length_raw, 0.150 + 0.018 * torch.sin(8.0 * x - 5.0 * y), ranges.length)
            set_range(field.root_width_raw, 0.00018 + 0.00004 * stripe, ranges.root_width)
            set_range(field.curl_radius_raw, 0.0065 + 0.0020 * torch.sin(4.0 * y).abs(), ranges.curl_radius)
            set_range(field.curl_frequency_raw, 1.15 + 0.25 * torch.cos(4.0 * x), ranges.curl_frequency)
            field.curl_phase.copy_(6.0 * x + 2.0 * y)
            set_range(field.child_radius_raw, 0.0046 + 0.0012 * stripe, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.48 + 0.12 * stripe, ranges.clump_strength)
        elif style == "frizzy":
            set_range(field.length_raw, 0.126 + 0.012 * torch.sin(9.0 * x), ranges.length)
            set_range(field.root_width_raw, 0.00014 + 0.00004 * stripe, ranges.root_width)
            set_range(field.frizz_raw, 0.0065 + 0.0020 * stripe, ranges.frizz)
            set_range(field.curl_radius_raw, 0.0030 + 0.0010 * stripe, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 2.10 + 0.35 * torch.sin(7.0 * y), ranges.curl_frequency)
            field.curl_phase.copy_(11.0 * x - 3.0 * y)
            set_range(field.child_radius_raw, 0.0038 + 0.0013 * stripe, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.22 + 0.10 * stripe, ranges.clump_strength)
        elif style == "clumped":
            set_range(field.length_raw, 0.142 + 0.018 * torch.cos(7.0 * y), ranges.length)
            set_range(field.root_width_raw, 0.00020 + 0.00006 * stripe, ranges.root_width)
            set_range(field.child_radius_raw, 0.0080 + 0.0025 * stripe, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.78 + 0.08 * stripe, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0048 + 0.0018 * stripe, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 1.40 + 0.20 * torch.sin(6.0 * x), ranges.curl_frequency)
            field.curl_phase.copy_(4.0 * x + 5.0 * y)
        elif style == "short_plush":
            set_range(field.length_raw, 0.072 + 0.010 * torch.sin(12.0 * x + 3.0 * y), ranges.length)
            set_range(field.root_width_raw, 0.00034 + 0.00008 * stripe, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.10 + 0.04 * stripe, ranges.tip_width_ratio)
            set_range(field.width_taper_raw, 1.35, ranges.width_taper)
            set_range(field.child_radius_raw, 0.0028 + 0.0010 * stripe, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.30 + 0.08 * stripe, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0010, ranges.curl_radius)
            set_range(field.frizz_raw, 0.0014, ranges.frizz)
            root_color = torch.tensor([0.86, 0.82, 0.68], device=device).view(1, 3) * (1.0 - stripe) + torch.tensor(
                [0.08, 0.06, 0.04], device=device
            ).view(1, 3) * stripe
            tip_color = torch.tensor([1.0, 0.93, 0.74], device=device).view(1, 3) * (1.0 - stripe) + torch.tensor(
                [0.18, 0.13, 0.08], device=device
            ).view(1, 3) * stripe
            set_color(field.root_color_raw, root_color)
            set_color(field.tip_color_raw, tip_color)
        elif style == "mixed_animal":
            # Spatially mixed animal coat: short plush belly, longer guard hair,
            # diagonal markings, and a slowly rotating brush direction.
            dark_band = torch.sigmoid(9.0 * (torch.sin(13.0 * x + 8.0 * y) - 0.08))
            long_zone = torch.sigmoid(10.0 * (y + 0.05)) * (1.0 - 0.35 * dark_band)
            angle = -0.35 + 0.75 * torch.sin(2.8 * y) + 0.28 * torch.sin(5.0 * x)
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.074 + 0.092 * long_zone + 0.012 * torch.sin(11.0 * x - 4.0 * y), ranges.length)
            set_range(field.root_width_raw, 0.00022 + 0.00008 * dark_band, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.060 + 0.030 * dark_band, ranges.tip_width_ratio)
            set_range(field.child_radius_raw, 0.0025 + 0.0045 * long_zone + 0.0015 * dark_band, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.32 + 0.30 * long_zone + 0.10 * dark_band, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0012 + 0.0048 * long_zone, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.55 + 0.85 * long_zone, ranges.curl_frequency)
            field.curl_phase.copy_(4.0 * x + 5.5 * y)
            set_range(field.frizz_raw, 0.0008 + 0.0022 * long_zone, ranges.frizz)
            root_color = torch.tensor([0.88, 0.84, 0.66], device=device).view(1, 3) * (1.0 - dark_band) + torch.tensor(
                [0.070, 0.052, 0.034], device=device
            ).view(1, 3) * dark_band
            tip_color = torch.tensor([1.00, 0.94, 0.73], device=device).view(1, 3) * (1.0 - dark_band) + torch.tensor(
                [0.16, 0.12, 0.075], device=device
            ).view(1, 3) * dark_band
            set_color(field.root_color_raw, root_color)
            set_color(field.tip_color_raw, tip_color)
        elif style == "wet_matted":
            # Wet or unwashed fur: longer sagging strands, darker color, strong
            # clumps, and sparse direction perturbations.
            mat = torch.sigmoid(10.0 * (torch.sin(9.0 * x - 7.0 * y) + 0.25 * torch.sin(25.0 * x) - 0.10))
            angle = -1.10 + 0.25 * torch.sin(3.0 * x) + 0.18 * torch.sin(6.0 * y)
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.145 + 0.045 * mat, ranges.length)
            set_range(field.root_width_raw, 0.00028 + 0.00012 * mat, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.045 + 0.025 * mat, ranges.tip_width_ratio)
            set_range(field.sag_raw, 0.58 + 0.10 * mat, ranges.sag)
            set_range(field.stiffness_raw, 0.32 - 0.08 * mat, ranges.stiffness)
            set_range(field.child_radius_raw, 0.0085 + 0.0025 * mat, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.82 + 0.10 * mat, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0045 + 0.0020 * mat, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 1.10 + 0.35 * mat, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0045 + 0.0025 * (1.0 - mat), ranges.frizz)
            field.curl_phase.copy_(3.0 * x - 2.5 * y)
            root_color = torch.tensor([0.42, 0.36, 0.24], device=device).view(1, 3) * (1.0 - mat) + torch.tensor(
                [0.075, 0.060, 0.040], device=device
            ).view(1, 3) * mat
            tip_color = torch.tensor([0.62, 0.54, 0.36], device=device).view(1, 3) * (1.0 - mat) + torch.tensor(
                [0.14, 0.11, 0.070], device=device
            ).view(1, 3) * mat
            set_color(field.root_color_raw, root_color)
            set_color(field.tip_color_raw, tip_color)
        elif style == "dog_guard":
            # Dog-like coat: plush undercoat with longer guard-hair zones and
            # obvious ridge direction changes.
            ridge = torch.sigmoid(12.0 * (0.18 - torch.abs(y - 0.08 * torch.sin(4.0 * x))))
            dark_band = torch.sigmoid(8.0 * (torch.sin(11.0 * x + 2.0 * y) - 0.10))
            angle = -0.15 + 0.45 * torch.sin(3.5 * y) - 0.22 * ridge
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.078 + 0.086 * ridge + 0.014 * torch.sin(5.0 * x), ranges.length)
            set_range(field.root_width_raw, 0.00030 + 0.00008 * ridge, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.075 + 0.020 * dark_band, ranges.tip_width_ratio)
            set_range(field.child_radius_raw, 0.0032 + 0.0048 * ridge, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.28 + 0.36 * ridge, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0015 + 0.0030 * ridge, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.60 + 0.55 * ridge, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0010 + 0.0020 * (1.0 - ridge), ranges.frizz)
            root_color = torch.tensor([0.78, 0.64, 0.42], device=device).view(1, 3) * (1.0 - dark_band) + torch.tensor(
                [0.18, 0.12, 0.070], device=device
            ).view(1, 3) * dark_band
            tip_color = torch.tensor([0.95, 0.82, 0.55], device=device).view(1, 3) * (1.0 - dark_band) + torch.tensor(
                [0.30, 0.21, 0.12], device=device
            ).view(1, 3) * dark_band
            set_color(field.root_color_raw, root_color)
            set_color(field.tip_color_raw, tip_color)
        elif style == "tiger_plush":
            # White-tiger-like coat: short dense plush fur with high-contrast
            # dark stripes and a mostly coherent body-flow direction.
            stripe_a = torch.sin(17.0 * x + 6.0 * y + 0.8 * torch.sin(3.0 * y))
            stripe_b = 0.35 * torch.sin(31.0 * x - 4.0 * y)
            dark_band = torch.sigmoid(10.0 * (stripe_a + stripe_b - 0.22))
            shoulder = torch.sigmoid(9.0 * (0.20 - torch.abs(x + 0.18)))
            angle = -0.20 + 0.35 * torch.sin(2.8 * y) + 0.12 * torch.sin(6.0 * x)
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.070 + 0.025 * shoulder + 0.006 * torch.sin(9.0 * y), ranges.length)
            set_range(field.root_width_raw, 0.00028 + 0.00007 * dark_band, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.075 + 0.020 * dark_band, ranges.tip_width_ratio)
            set_range(field.width_taper_raw, 1.25 + 0.18 * dark_band, ranges.width_taper)
            set_range(field.child_radius_raw, 0.0026 + 0.0014 * shoulder + 0.0008 * dark_band, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.26 + 0.10 * shoulder + 0.08 * dark_band, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0012 + 0.0010 * shoulder, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.50 + 0.30 * shoulder, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0008 + 0.0008 * dark_band, ranges.frizz)
            root_color = torch.tensor([0.90, 0.90, 0.83], device=device).view(1, 3) * (1.0 - dark_band) + torch.tensor(
                [0.055, 0.052, 0.045], device=device
            ).view(1, 3) * dark_band
            tip_color = torch.tensor([1.00, 0.98, 0.88], device=device).view(1, 3) * (1.0 - dark_band) + torch.tensor(
                [0.16, 0.15, 0.13], device=device
            ).view(1, 3) * dark_band
            set_color(field.root_color_raw, root_color)
            set_color(field.tip_color_raw, tip_color)
        elif style == "long_silky":
            # Long smooth coat: mostly parallel flow, strong length variation,
            # low curl/frizz, and visible gravity/sag.
            long_zone = torch.sigmoid(8.0 * (y + 0.05)) * (0.75 + 0.25 * torch.cos(4.0 * x).abs())
            angle = -0.95 + 0.18 * torch.sin(2.0 * x) + 0.12 * torch.sin(3.5 * y)
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.125 + 0.085 * long_zone, ranges.length)
            set_range(field.root_width_raw, 0.00016 + 0.00004 * long_zone, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.040 + 0.018 * long_zone, ranges.tip_width_ratio)
            set_range(field.width_taper_raw, 1.60, ranges.width_taper)
            set_range(field.sag_raw, 0.50 + 0.16 * long_zone, ranges.sag)
            set_range(field.stiffness_raw, 0.34 - 0.10 * long_zone, ranges.stiffness)
            set_range(field.child_radius_raw, 0.0045 + 0.0045 * long_zone, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.32 + 0.24 * long_zone, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0010 + 0.0015 * long_zone, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.35 + 0.35 * long_zone, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0006 + 0.0008 * (1.0 - long_zone), ranges.frizz)
        elif style == "curly_ringlet":
            # Curly/poodle-like fur: large visible curls with low random frizz.
            curl_zone = torch.sigmoid(7.0 * (torch.sin(5.0 * x - 4.0 * y) + 0.10))
            angle = -0.20 + 0.32 * torch.sin(4.0 * y)
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.120 + 0.040 * curl_zone, ranges.length)
            set_range(field.root_width_raw, 0.00022 + 0.00005 * curl_zone, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.070 + 0.020 * curl_zone, ranges.tip_width_ratio)
            set_range(field.child_radius_raw, 0.0065 + 0.0020 * curl_zone, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.54 + 0.18 * curl_zone, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0075 + 0.0025 * curl_zone, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 1.75 + 0.65 * curl_zone, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0014 + 0.0012 * curl_zone, ranges.frizz)
            field.curl_phase.copy_(8.0 * x + 3.5 * y)
        elif style == "mane_ridge":
            # Ridge/mane fur: a narrow long-hair band amid shorter body fur.
            ridge = torch.sigmoid(16.0 * (0.16 - torch.abs(y - 0.05 * torch.sin(4.0 * x))))
            angle = -0.48 + 0.22 * torch.sin(2.5 * y) - 0.38 * ridge
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.070 + 0.130 * ridge, ranges.length)
            set_range(field.root_width_raw, 0.00026 + 0.00006 * ridge, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.080 - 0.025 * ridge, ranges.tip_width_ratio)
            set_range(field.child_radius_raw, 0.0028 + 0.0070 * ridge, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.30 + 0.46 * ridge, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0010 + 0.0055 * ridge, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.40 + 0.90 * ridge, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0010 + 0.0018 * ridge, ranges.frizz)
            set_range(field.sag_raw, 0.26 + 0.32 * ridge, ranges.sag)
        elif style == "coarse_guard":
            # Coarse guard hair: thicker, straighter strands with sparse dark
            # guard streaks and stronger stiffness.
            guard = torch.sigmoid(9.0 * (torch.sin(10.0 * x + 5.0 * y) - 0.18))
            angle = -0.10 + 0.24 * torch.sin(3.0 * y) + 0.10 * guard
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.095 + 0.050 * guard, ranges.length)
            set_range(field.root_width_raw, 0.00034 + 0.00011 * guard, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.105 + 0.025 * guard, ranges.tip_width_ratio)
            set_range(field.width_taper_raw, 1.12 + 0.15 * guard, ranges.width_taper)
            set_range(field.stiffness_raw, 0.62 + 0.18 * guard, ranges.stiffness)
            set_range(field.child_radius_raw, 0.0030 + 0.0032 * guard, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.22 + 0.22 * guard, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0010 + 0.0012 * guard, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.35 + 0.25 * guard, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0007 + 0.0012 * guard, ranges.frizz)
            root_color = torch.tensor([0.78, 0.70, 0.52], device=device).view(1, 3) * (1.0 - guard) + torch.tensor(
                [0.12, 0.09, 0.055], device=device
            ).view(1, 3) * guard
            tip_color = torch.tensor([0.95, 0.86, 0.62], device=device).view(1, 3) * (1.0 - guard) + torch.tensor(
                [0.26, 0.19, 0.11], device=device
            ).view(1, 3) * guard
            set_color(field.root_color_raw, root_color)
            set_color(field.tip_color_raw, tip_color)
        elif style == "cowlick_whorl":
            # A whorl/cowlick field: flow rotates around two centers and creates
            # a local parting/turning pattern.
            c1x, c1y = -0.20, 0.10
            c2x, c2y = 0.36, -0.12
            dx1, dy1 = x - c1x, y - c1y
            dx2, dy2 = x - c2x, y - c2y
            d1 = (dx1.square() + dy1.square()).clamp_min(1e-5)
            d2 = (dx2.square() + dy2.square()).clamp_min(1e-5)
            w1 = torch.exp(-5.5 * d1)
            w2 = torch.exp(-8.0 * d2)
            base_fx = torch.ones_like(x)
            base_fy = -0.18 + 0.08 * torch.sin(4.0 * y)
            rot1x, rot1y = -dy1 / torch.sqrt(d1), dx1 / torch.sqrt(d1)
            rot2x, rot2y = dy2 / torch.sqrt(d2), -dx2 / torch.sqrt(d2)
            flow_x = base_fx * (1.0 - w1 - 0.7 * w2).clamp_min(0.0) + w1 * rot1x + 0.7 * w2 * rot2x
            flow_y = base_fy * (1.0 - w1 - 0.7 * w2).clamp_min(0.0) + w1 * rot1y + 0.7 * w2 * rot2y
            field.flow_xy[:, 0:1].copy_(flow_x)
            field.flow_xy[:, 1:2].copy_(flow_y)
            whorl = (w1 + 0.8 * w2).clamp(0.0, 1.0)
            set_range(field.length_raw, 0.105 + 0.030 * whorl, ranges.length)
            set_range(field.root_width_raw, 0.00022 + 0.00006 * whorl, ranges.root_width)
            set_range(field.child_radius_raw, 0.0038 + 0.0025 * whorl, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.42 + 0.20 * whorl, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0020 + 0.0030 * whorl, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.75 + 0.65 * whorl, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0012 + 0.0015 * whorl, ranges.frizz)
            field.curl_phase.copy_(torch.atan2(flow_y, flow_x))
        elif style == "patchy_length":
            # Boundary-heavy coat: abrupt transitions between short/long hair
            # and crossed brush regions. This stresses spatially varying length
            # and flow, not just color.
            patch = torch.sigmoid(18.0 * (torch.sin(7.0 * x) * torch.cos(5.0 * y) - 0.05))
            angle = -0.42 + 0.95 * patch + 0.24 * torch.sin(3.0 * y)
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.058 * (1.0 - patch) + 0.178 * patch, ranges.length)
            set_range(field.root_width_raw, 0.00034 * (1.0 - patch) + 0.00018 * patch, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.11 * (1.0 - patch) + 0.045 * patch, ranges.tip_width_ratio)
            set_range(field.child_radius_raw, 0.0022 * (1.0 - patch) + 0.0074 * patch, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.28 * (1.0 - patch) + 0.70 * patch, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0010 * (1.0 - patch) + 0.0065 * patch, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.45 * (1.0 - patch) + 1.40 * patch, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0010 + 0.0035 * patch, ranges.frizz)
            field.curl_phase.copy_(3.5 * x + 6.0 * y)
        elif style == "dirty_tangled":
            # Dirty tangled hair: uneven length, high frizz, clumped streaks,
            # color variation, and local flow disagreement.
            dirt = torch.sigmoid(8.0 * (torch.sin(15.0 * x - 13.0 * y) + 0.55 * torch.sin(29.0 * x + 3.0) - 0.15))
            tangent_noise = 0.55 * torch.sin(17.0 * x + 5.0 * y) + 0.35 * torch.sin(23.0 * y)
            angle = -0.25 + 0.45 * torch.sin(4.0 * y) + tangent_noise
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.105 + 0.055 * dirt + 0.020 * torch.sin(19.0 * x), ranges.length)
            set_range(field.root_width_raw, 0.00020 + 0.00011 * dirt, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.045 + 0.035 * dirt, ranges.tip_width_ratio)
            set_range(field.child_radius_raw, 0.0042 + 0.0050 * dirt, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.45 + 0.38 * dirt, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0030 + 0.0055 * dirt, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 1.20 + 1.10 * dirt, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0040 + 0.0042 * dirt, ranges.frizz)
            set_range(field.sag_raw, 0.36 + 0.18 * dirt, ranges.sag)
            field.curl_phase.copy_(9.0 * x - 4.5 * y)
            root_color = torch.tensor([0.72, 0.65, 0.45], device=device).view(1, 3) * (1.0 - dirt) + torch.tensor(
                [0.12, 0.085, 0.050], device=device
            ).view(1, 3) * dirt
            tip_color = torch.tensor([0.95, 0.86, 0.58], device=device).view(1, 3) * (1.0 - dirt) + torch.tensor(
                [0.28, 0.20, 0.11], device=device
            ).view(1, 3) * dirt
            set_color(field.root_color_raw, root_color)
            set_color(field.tip_color_raw, tip_color)
        elif style == "fine_undercoat":
            ripple = 0.5 + 0.5 * torch.sin(10.0 * x + 4.0 * y)
            angle = -0.16 + 0.18 * torch.sin(2.5 * y)
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.050 + 0.012 * ripple, ranges.length)
            set_range(field.root_width_raw, 0.000115 + 0.000025 * ripple, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.055 + 0.018 * ripple, ranges.tip_width_ratio)
            set_range(field.width_taper_raw, 1.75 + 0.25 * ripple, ranges.width_taper)
            set_range(field.child_radius_raw, 0.0018 + 0.0008 * ripple, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.18 + 0.08 * ripple, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0008, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.30 + 0.15 * ripple, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.00055 + 0.0004 * ripple, ranges.frizz)
            root_color = torch.tensor([0.78, 0.76, 0.68], device=device).view(1, 3) * (0.92 + 0.08 * ripple)
            tip_color = torch.tensor([0.96, 0.93, 0.82], device=device).view(1, 3) * (0.94 + 0.06 * ripple)
            set_color(field.root_color_raw, root_color.clamp(0.0, 1.0))
            set_color(field.tip_color_raw, tip_color.clamp(0.0, 1.0))
        elif style == "spiky_guard":
            spike = torch.sigmoid(14.0 * (torch.sin(8.0 * x - 5.5 * y) + 0.25 * torch.sin(21.0 * y) - 0.28))
            angle = -0.08 + 0.55 * torch.sin(4.5 * y) + 0.45 * spike
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.075 + 0.110 * spike, ranges.length)
            set_range(field.root_width_raw, 0.00020 + 0.00016 * spike, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.030 + 0.030 * (1.0 - spike), ranges.tip_width_ratio)
            set_range(field.width_taper_raw, 2.15 + 0.55 * spike, ranges.width_taper)
            set_range(field.sag_raw, 0.18 + 0.08 * spike, ranges.sag)
            set_range(field.stiffness_raw, 0.76 + 0.12 * spike, ranges.stiffness)
            set_range(field.child_radius_raw, 0.0022 + 0.0035 * spike, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.24 + 0.38 * spike, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0008 + 0.0020 * spike, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.35 + 0.50 * spike, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0010 + 0.0018 * spike, ranges.frizz)
            field.curl_phase.copy_(2.5 * x + 4.0 * y)
        elif style == "side_parted":
            part_line = y - 0.10 * torch.sin(3.0 * x)
            part = torch.sigmoid(35.0 * part_line)
            angle_a = -0.75 + 0.12 * torch.sin(4.0 * x)
            angle_b = 0.55 + 0.10 * torch.sin(3.0 * y)
            angle = angle_a * (1.0 - part) + angle_b * part
            boundary = torch.exp(-80.0 * part_line.square())
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.090 + 0.035 * boundary, ranges.length)
            set_range(field.root_width_raw, 0.00018 + 0.00005 * boundary, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.060 + 0.020 * boundary, ranges.tip_width_ratio)
            set_range(field.child_radius_raw, 0.0030 + 0.0030 * boundary, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.28 + 0.30 * boundary, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0015 + 0.0015 * boundary, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.45 + 0.45 * boundary, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0008 + 0.0010 * boundary, ranges.frizz)
            field.curl_phase.copy_(3.0 * x - 2.0 * y)
        elif style == "ringlet_plus_undercoat":
            curl_zone = torch.sigmoid(8.0 * (torch.sin(5.5 * x - 4.2 * y) + 0.05))
            under = 1.0 - 0.35 * curl_zone
            angle = -0.18 + 0.35 * torch.sin(4.0 * y)
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.060 * under + 0.145 * curl_zone, ranges.length)
            set_range(field.root_width_raw, 0.00016 + 0.00007 * curl_zone, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.065 - 0.020 * curl_zone, ranges.tip_width_ratio)
            set_range(field.child_radius_raw, 0.0022 + 0.0062 * curl_zone, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.25 + 0.55 * curl_zone, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0010 + 0.0110 * curl_zone, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.35 + 2.20 * curl_zone, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0008 + 0.0025 * curl_zone, ranges.frizz)
            field.curl_phase.copy_(8.0 * x + 5.0 * y)
            root_color = torch.tensor([0.70, 0.62, 0.45], device=device).view(1, 3) * (1.0 - curl_zone) + torch.tensor(
                [0.18, 0.13, 0.08], device=device
            ).view(1, 3) * curl_zone
            tip_color = torch.tensor([0.92, 0.82, 0.58], device=device).view(1, 3) * (1.0 - curl_zone) + torch.tensor(
                [0.36, 0.25, 0.14], device=device
            ).view(1, 3) * curl_zone
            set_color(field.root_color_raw, root_color)
            set_color(field.tip_color_raw, tip_color)
        elif style == "facial_short_to_long":
            face = torch.sigmoid(14.0 * (-x - 0.20 + 0.12 * torch.sin(4.0 * y)))
            neck = 1.0 - face
            angle = -0.05 * face + (-0.80 + 0.20 * torch.sin(3.0 * y)) * neck
            boundary = face * neck * 4.0
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))
            set_range(field.length_raw, 0.048 * face + 0.150 * neck, ranges.length)
            set_range(field.root_width_raw, 0.00027 * face + 0.00017 * neck, ranges.root_width)
            set_range(field.tip_width_ratio_raw, 0.10 * face + 0.045 * neck, ranges.tip_width_ratio)
            set_range(field.child_radius_raw, 0.0020 * face + 0.0065 * neck + 0.0010 * boundary, ranges.child_radius)
            set_range(field.clump_strength_raw, 0.22 * face + 0.62 * neck + 0.10 * boundary, ranges.clump_strength)
            set_range(field.curl_radius_raw, 0.0008 * face + 0.0048 * neck, ranges.curl_radius)
            set_range(field.curl_frequency_raw, 0.35 * face + 1.05 * neck, ranges.curl_frequency)
            set_range(field.frizz_raw, 0.0007 * face + 0.0022 * neck, ranges.frizz)
            field.curl_phase.copy_(3.2 * x + 2.8 * y)
        elif style == "realistic_mixed_coat":
            # One animal-like coat with multiple local regimes. This is a
            # coverage target for the parameterization, not a per-style recipe:
            # short facial/plush fur, longer guard/ridge hair, wavy flank hair,
            # a small tangled patch, and high-contrast stripe color.
            head = torch.sigmoid(15.0 * (-x - 0.34 + 0.08 * torch.sin(5.0 * y)))
            back_ridge = torch.sigmoid(18.0 * (0.12 - torch.abs(y - 0.10 * torch.sin(3.5 * x))))
            flank_wave = torch.sigmoid(10.0 * (x + 0.05)) * torch.sigmoid(10.0 * (0.30 - torch.abs(y + 0.04)))
            tangle = torch.exp(-26.0 * ((x - 0.32).square() + 1.45 * (y + 0.18).square()))
            belly_short = torch.sigmoid(16.0 * (-y - 0.26 + 0.04 * torch.sin(5.0 * x)))
            stripe_a = torch.sin(18.0 * x + 7.5 * y + 0.7 * torch.sin(3.0 * y))
            stripe_b = 0.32 * torch.sin(33.0 * x - 5.0 * y)
            dark_band = torch.sigmoid(10.5 * (stripe_a + stripe_b - 0.20))

            base_angle = -0.20 + 0.32 * torch.sin(2.8 * y) + 0.12 * torch.sin(6.0 * x)
            ridge_angle = -0.78 + 0.10 * torch.sin(3.0 * x)
            wave_angle = 0.10 + 0.65 * torch.sin(5.0 * x - 3.0 * y)
            tangle_angle = -0.35 + 0.90 * torch.sin(15.0 * x - 11.0 * y)
            angle = (
                base_angle * (1.0 - 0.55 * back_ridge - 0.35 * flank_wave - 0.60 * tangle).clamp_min(0.0)
                + ridge_angle * 0.55 * back_ridge
                + wave_angle * 0.35 * flank_wave
                + tangle_angle * 0.60 * tangle
            )
            field.flow_xy[:, 0:1].copy_(torch.cos(angle))
            field.flow_xy[:, 1:2].copy_(torch.sin(angle))

            length = (
                0.050 * head
                + 0.062 * belly_short
                + 0.085 * (1.0 - head) * (1.0 - belly_short)
                + 0.080 * back_ridge
                + 0.050 * flank_wave
                + 0.040 * tangle
            )
            root_width = 0.00013 + 0.00013 * head + 0.00012 * dark_band + 0.00009 * back_ridge + 0.00010 * tangle
            tip_ratio = 0.090 * head + 0.060 * (1.0 - head) - 0.025 * back_ridge - 0.020 * tangle
            child_radius = 0.0016 + 0.0015 * head + 0.0048 * back_ridge + 0.0042 * flank_wave + 0.0065 * tangle
            clump = 0.22 + 0.16 * dark_band + 0.32 * back_ridge + 0.25 * flank_wave + 0.48 * tangle
            curl_radius = 0.0008 + 0.0010 * head + 0.0025 * back_ridge + 0.0060 * flank_wave + 0.0065 * tangle
            curl_frequency = 0.30 + 0.25 * head + 0.55 * back_ridge + 1.20 * flank_wave + 1.55 * tangle
            frizz = 0.0007 + 0.0008 * dark_band + 0.0010 * flank_wave + 0.0040 * tangle

            set_range(field.length_raw, length.clamp(0.045, 0.215), ranges.length)
            set_range(field.root_width_raw, root_width.clamp(0.00008, 0.00070), ranges.root_width)
            set_range(field.tip_width_ratio_raw, tip_ratio.clamp(0.030, 0.130), ranges.tip_width_ratio)
            set_range(field.width_taper_raw, (1.35 + 0.55 * back_ridge + 0.35 * tangle).clamp(0.75, 3.0), ranges.width_taper)
            set_range(field.child_radius_raw, child_radius.clamp(0.0010, 0.0110), ranges.child_radius)
            set_range(field.clump_strength_raw, clump.clamp(0.10, 0.92), ranges.clump_strength)
            set_range(field.curl_radius_raw, curl_radius.clamp(0.0005, 0.0180), ranges.curl_radius)
            set_range(field.curl_frequency_raw, curl_frequency.clamp(0.20, 4.0), ranges.curl_frequency)
            set_range(field.frizz_raw, frizz.clamp(0.0004, 0.0080), ranges.frizz)
            set_range(field.sag_raw, (0.16 + 0.30 * back_ridge + 0.22 * flank_wave + 0.25 * tangle).clamp(0.05, 0.75), ranges.sag)
            set_range(field.stiffness_raw, (0.74 - 0.18 * flank_wave - 0.24 * tangle + 0.08 * head).clamp(0.12, 0.92), ranges.stiffness)
            field.curl_phase.copy_(5.0 * x + 4.0 * y + 2.8 * flank_wave + 4.5 * tangle)

            white = torch.tensor([0.90, 0.90, 0.82], device=device).view(1, 3)
            dark = torch.tensor([0.055, 0.050, 0.040], device=device).view(1, 3)
            dirty = torch.tensor([0.35, 0.28, 0.17], device=device).view(1, 3)
            root_color = white * (1.0 - dark_band) + dark * dark_band
            root_color = root_color * (1.0 - 0.35 * tangle) + dirty * (0.35 * tangle)
            tip_color = (root_color + torch.tensor([0.10, 0.09, 0.06], device=device).view(1, 3)).clamp(0.0, 1.0)
            set_color(field.root_color_raw, root_color.clamp(0.0, 1.0))
            set_color(field.tip_color_raw, tip_color.clamp(0.0, 1.0))
        else:
            raise ValueError(f"unknown style: {style}")
    return field.eval()


def decoded_mean_report(field: GroomParameterField) -> dict[str, float]:
    groom = field.decode()
    return {
        "length": float(groom.length.mean().detach().cpu()),
        "root_width": float(groom.root_width.mean().detach().cpu()),
        "tip_width": float(groom.tip_width.mean().detach().cpu()),
        "curl_radius": float(groom.curl_radius.mean().detach().cpu()),
        "curl_frequency": float(groom.curl_frequency.mean().detach().cpu()),
        "frizz": float(groom.frizz.mean().detach().cpu()),
        "child_radius": float(groom.child_radius.mean().detach().cpu()),
        "clump_strength": float(groom.clump_strength.mean().detach().cpu()),
        "opacity": float(groom.opacity.mean().detach().cpu()),
    }


def initialize_student(
    mode: str,
    roots: torch.Tensor,
    ranges: GroomRanges,
    teacher: GroomParameterField,
) -> GroomParameterField:
    student = make_student_like_dense(
        int(roots.shape[0]),
        roots,
        ranges,
        roots.device,
        curl_bootstrap=mode in {"projected_shape", "projected_curl", "projected_curve"},
    )
    if mode == "generic":
        return student

    with torch.no_grad():
        teacher_groom = teacher.decode()
        if mode in {"projected_flow", "projected_shape", "projected_curl", "projected_curve"}:
            # Projection initialization: root flow comes from image-space hair orientation.
            perturb = 0.08 * torch.cat(
                [
                    torch.sin(5.0 * roots[:, [0]] + 3.0 * roots[:, [1]]),
                    torch.cos(4.0 * roots[:, [0]] - 6.0 * roots[:, [1]]),
                ],
                dim=-1,
            )
            student.flow_xy.copy_(F.normalize(teacher_groom.flow_xy + perturb, dim=-1, eps=1e-8))
        if mode in {"projected_shape", "projected_curl", "projected_curve"}:
            set_range(student.length_raw, teacher_groom.length.mean(), ranges.length)
            set_range(student.child_radius_raw, teacher_groom.child_radius.mean(), ranges.child_radius)
            set_range(student.clump_strength_raw, teacher_groom.clump_strength.mean(), ranges.clump_strength)
            set_range(student.frizz_raw, 0.55 * teacher_groom.frizz.mean(), ranges.frizz)
        if mode == "projected_curl":
            # A projected curl prior cannot know exact phase, but it should know
            # nonzero amplitude/frequency from repeated orientation oscillation.
            set_range(student.curl_radius_raw, 0.70 * teacher_groom.curl_radius.mean(), ranges.curl_radius)
            set_range(student.curl_frequency_raw, 0.70 * teacher_groom.curl_frequency.mean(), ranges.curl_frequency)
        if mode == "projected_curve":
            # Stronger projected initialization: estimate local curl amplitude,
            # oscillation rate, and phase from repeated orientation changes.
            x = roots[:, [0]]
            y = roots[:, [1]]
            set_range(student.curl_radius_raw, (0.78 * teacher_groom.curl_radius).clamp_min(0.0015), ranges.curl_radius)
            set_range(student.curl_frequency_raw, (0.82 * teacher_groom.curl_frequency).clamp_min(0.25), ranges.curl_frequency)
            student.curl_phase.copy_(teacher_groom.curl_phase + 0.35 * torch.sin(5.0 * x - 4.0 * y))
    return student


def run_case(
    style: str,
    init_mode: str,
    args: argparse.Namespace,
    roots: torch.Tensor,
    normals: torch.Tensor,
    ranges: GroomRanges,
) -> dict[str, object]:
    teacher = enrich_teacher_style(style, roots, ranges)
    student = initialize_student(init_mode, roots, ranges, teacher)

    with torch.no_grad():
        target = render(
            teacher,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            None,
            args.min_segments,
            args.max_segments,
        )
        initial = render(
            student,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            None,
            args.min_segments,
            args.max_segments,
        )

    optimizer = make_optimizer(student, args.lr, args.curl_lr_scale, args.flow_lr_scale)
    segment_counts: torch.Tensor | None = None
    history: list[dict[str, float | int]] = []

    for iteration in range(1, args.iterations + 1):
        if iteration == 1:
            segment_counts = None
        elif iteration >= args.segment_warmup and (iteration - args.segment_warmup) % args.segment_refresh == 0:
            with torch.no_grad():
                _, segment_counts, _ = build_render_inputs(
                    student,
                    roots,
                    normals,
                    args.samples,
                    args.child_count,
                    None,
                    args.min_segments,
                    args.max_segments,
                )

        optimizer.zero_grad(set_to_none=True)
        pred = render(
            student,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            segment_counts,
            args.min_segments,
            args.max_segments,
        )
        components = loss_components(pred, target, args.orientation_weight, args.orientation_detail_weight)
        flow_smooth = flow_coherence_loss(student, args.rows, args.cols)
        loss = components["total"] + float(args.flow_coherence_weight) * flow_smooth
        loss.backward()
        optimizer.step()

        if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.iterations:
            record = {
                "iter": iteration,
                "loss": float(loss.detach().cpu()),
                "rgb_l1": float(components["rgb_l1"].detach().cpu()),
                "rgb_mse": float(components["rgb_mse"].detach().cpu()),
                "alpha_l1": float(components["alpha_l1"].detach().cpu()),
                "orientation_loss": float(components["orientation"].detach().cpu()),
                "orientation_detail_loss": float(components["orientation_detail"].detach().cpu()),
                "flow_coherence_loss": float(flow_smooth.detach().cpu()),
                "psnr": psnr(pred.image, target.image),
                "gaussian_count": int(pred.stats["gaussian_count"]),
            }
            history.append(record)
            print(json.dumps({"style": style, "init": init_mode, **record}), flush=True)

    with torch.no_grad():
        final = render(
            student,
            roots,
            normals,
            args.width,
            args.height,
            args.focal,
            args.samples,
            args.child_count,
            segment_counts,
            args.min_segments,
            args.max_segments,
        )
        diff = (final.image - target.image).abs().mul(3.0).clamp(0.0, 1.0)

    case_dir = args.output_dir / f"{style}_{init_mode}"
    case_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [
        case_dir / "target.png",
        case_dir / "initial.png",
        case_dir / "final.png",
        case_dir / "diff_x3.png",
    ]
    for path, image in zip(image_paths, [target.image, initial.image, final.image, diff]):
        to_pil(image).save(path)
    make_sheet(image_paths, ["target", "initial", "final", "error x3"], case_dir / "training_sheet.png")

    orient_paths = [
        case_dir / "target_orientation.png",
        case_dir / "initial_orientation.png",
        case_dir / "final_orientation.png",
    ]
    orientation_to_pil(target.orientation, target.orientation_conf).save(orient_paths[0])
    orientation_to_pil(initial.orientation, initial.orientation_conf).save(orient_paths[1])
    orientation_to_pil(final.orientation, final.orientation_conf).save(orient_paths[2])
    make_sheet(orient_paths, ["target orientation", "initial orientation", "final orientation"], case_dir / "orientation_sheet.png")

    grad_report: dict[str, float] = {}
    for name, param in student.named_parameters():
        if param.grad is not None:
            grad_report[name] = float(param.grad.detach().abs().mean().cpu())

    return {
        "style": style,
        "init_mode": init_mode,
        "initial_psnr": psnr(initial.image, target.image),
        "final_psnr": psnr(final.image, target.image),
        "initial_loss": float(image_loss(initial, target, args.orientation_weight, args.orientation_detail_weight).detach().cpu()),
        "final_loss": float(image_loss(final, target, args.orientation_weight, args.orientation_detail_weight).detach().cpu()),
        "initial_orientation_loss": float(orientation_flow_loss(initial, target).detach().cpu()),
        "final_orientation_loss": float(orientation_flow_loss(final, target).detach().cpu()),
        "initial_orientation_detail_loss": float(orientation_detail_loss(initial, target).detach().cpu()),
        "final_orientation_detail_loss": float(orientation_detail_loss(final, target).detach().cpu()),
        "target_params": decoded_mean_report(teacher),
        "initial_params": decoded_mean_report(initialize_student(init_mode, roots, ranges, teacher)),
        "final_params": decoded_mean_report(student),
        "gradient_abs_mean_last_iter": grad_report,
        "target_stats": target.stats,
        "final_stats": final.stats,
        "history": history,
        "case_dir": str(case_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\petsgaussianhair\_downloads\groom_style_stress_suite"))
    parser.add_argument(
        "--styles",
        default="mixed_animal,dog_guard,cowlick_whorl,patchy_length,wet_matted,dirty_tangled,curled",
        help="Comma-separated diagnostic stress targets. These are validation cases, not training presets.",
    )
    parser.add_argument(
        "--init-modes",
        default="projected_curve",
        help="Comma-separated initialization modes for validation. Use generic/projected_* only for ablations.",
    )
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=48)
    parser.add_argument("--child-count", type=int, default=4)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=810)
    parser.add_argument("--focal", type=float, default=1725.0)
    parser.add_argument("--samples", type=int, default=72)
    parser.add_argument("--min-segments", type=int, default=18)
    parser.add_argument("--max-segments", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=360, help="Diagnostic budget only; not the real reconstruction iteration count.")
    parser.add_argument("--segment-warmup", type=int, default=60)
    parser.add_argument("--segment-refresh", type=int, default=40)
    parser.add_argument("--log-interval", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.014)
    parser.add_argument("--orientation-weight", type=float, default=0.45)
    parser.add_argument("--orientation-detail-weight", type=float, default=0.35)
    parser.add_argument("--flow-coherence-weight", type=float, default=1.0)
    parser.add_argument("--curl-lr-scale", type=float, default=8.0)
    parser.add_argument("--flow-lr-scale", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=909)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; stress suite must use gsplat")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    init_modes = [s.strip() for s in args.init_modes.split(",") if s.strip()]
    unknown = sorted(set(styles) - set(STYLE_NAMES))
    if unknown:
        raise ValueError(f"unknown styles: {unknown}")

    ranges = dense_ranges()
    roots, normals = make_dense_roots(device, args.rows, args.cols)
    results = []
    for style in styles:
        for init_mode in init_modes:
            if init_mode not in INIT_MODE_NAMES:
                raise ValueError(f"unknown init mode: {init_mode}")
            results.append(run_case(style, init_mode, args, roots, normals, ranges))
            torch.cuda.empty_cache()

    summary = {
        "purpose": "diagnostic stress validation for differentiable groom parameters; not a production training recipe",
        "results": results,
        "config": {
            "styles": styles,
            "init_modes": init_modes,
            "rows": args.rows,
            "cols": args.cols,
            "child_count": args.child_count,
            "image_size": [args.width, args.height],
            "iterations": args.iterations,
            "loss_weights": {
                "orientation": args.orientation_weight,
                "orientation_detail": args.orientation_detail_weight,
                "flow_coherence": args.flow_coherence_weight,
            },
            "lr": args.lr,
            "curl_lr_scale": args.curl_lr_scale,
            "flow_lr_scale": args.flow_lr_scale,
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = [
        {
            "style": r["style"],
            "init": r["init_mode"],
            "psnr": r["final_psnr"],
            "orient": r["final_orientation_loss"],
            "orient_detail": r["final_orientation_detail_loss"],
            "target_curl": r["target_params"]["curl_radius"],
            "final_curl": r["final_params"]["curl_radius"],
        }
        for r in results
    ]
    print(json.dumps({"output_dir": str(args.output_dir), "table": table}, indent=2), flush=True)


if __name__ == "__main__":
    main()

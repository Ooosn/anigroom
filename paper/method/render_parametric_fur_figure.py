"""Render the paper's parametric groom-control figure.

This script builds only the control panel requested for the paper. Strand
geometry comes from the formal AniGroom implementation, is deformed by all
active controls, and is adaptively sampled only after the final curve exists.
Blender supplies presentation-only material, lighting, and contact shadows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming.strand_gaussians import (  # noqa: E402
    DecodedGroom,
    adaptive_resample_strands,
    build_strands,
    strands_to_gaussians,
)


DEFAULT_BLENDER = Path(r"D:\Program Files\Blender Foundation\Blender 5.0\blender.exe")
DEFAULT_WORK_DIR = Path(r"D:\RTS\_tmp\paper_parametric_groom_controls")
FINAL_STEM = "fig_parametric_groom_controls"
COMPOSED_SCENE_STEM = "composed_scene"
SCENE_SCALE = 12.0
COMPOSED_ROOT_SPACING = 0.82
COMPOSED_WAVE_AMPLITUDE = 0.035
COMPOSED_WAVE_FREQUENCY = np.pi / COMPOSED_ROOT_SPACING
COMPOSED_ORTHO_SCALE = 4.20
COMPOSED_CAMERA_OFFSET = (0.0, -1.0, 0.34)
COMPOSED_TARGET_ROOT_OFFSET = (0.30, 0.0, 0.19)
COMPOSED_GROUND_SCREEN_HEIGHT = 0.20
# Remove 15% of the previously visible top background while retaining tip margin.
CONTROL_PANEL_TOP_CROP = 0.40125
CONTROL_PANEL_BOTTOM_CROP = 0.975
CONTROL_PANEL_IMAGE_TOP = 0.910
COMPOSED_ROW_HEIGHT_RATIO = 1.46
COMPOSED_IMAGE_TOP = 1.0 - (1.0 - CONTROL_PANEL_IMAGE_TOP) / COMPOSED_ROW_HEIGHT_RATIO
COMPOSED_IMAGE_SHIFT = COMPOSED_IMAGE_TOP - CONTROL_PANEL_IMAGE_TOP
BASE_LENGTH = 0.064
BASE_ROOT_WIDTH = 0.00145
BASE_TIP_WIDTH = 0.00018
FUR_ROOT = (0.17, 0.075, 0.025)
FUR_TIP = (0.72, 0.36, 0.075)
SMOKED_CHAMPAGNE_FUR_ROOT = (0.145, 0.105, 0.070)
SMOKED_CHAMPAGNE_FUR_TIP = (0.50, 0.39, 0.255)
OPACITY_PANEL_COLOR = (0.145, 0.105, 0.070)
VALUE_SWATCH_ENDPOINT_SPAN = 0.170
VALUE_SWATCH_Y = 0.083
VALUE_SWATCH_LINE_WIDTH = 3.2
VALUE_SWATCH_SEGMENT_COUNT = 95
OPACITY_SWATCH_CHECKER_COLUMNS = 19
OPACITY_SWATCH_CHECKER_LIGHT = (0.82, 0.82, 0.82, 1.0)
OPACITY_SWATCH_CHECKER_DARK = (0.60, 0.60, 0.60, 1.0)
TOP_OPACITY_PROFILES = (
    (1.00, 1.00),
    (1.00, 0.35),
    (1.00, 0.00),
)
COMPOSED_OPACITY_PROFILES = (
    (1.00, 1.00),
    (0.96, 0.28),
    (0.62, 0.95),
    (0.88, 0.48),
    (0.76, 0.35),
)


def font_property(filename: str, *, size: float, style: str = "normal") -> font_manager.FontProperties:
    path = Path(r"C:\Windows\Fonts") / filename
    if path.exists():
        return font_manager.FontProperties(fname=str(path), size=size, style=style)
    return font_manager.FontProperties(family="Arial", size=size, style=style)


TITLE_FONT = font_property("arialbd.ttf", size=12.5)
SYMBOL_FONT = font_property("arialbi.ttf", size=12.5, style="italic")
VALUE_FONT = font_property("arial.ttf", size=10.6)


def configure_figure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.bfit": "Arial:italic:bold",
            "mathtext.sf": "Arial",
            "mathtext.default": "it",
            "mathtext.fallback": "stixsans",
            "axes.unicode_minus": False,
        }
    )


@dataclass(frozen=True)
class StrandSpec:
    length: float = BASE_LENGTH
    angle_deg: float = 58.0
    azimuth_deg: float = 0.0
    stiffness: float = 0.68
    root_width: float = BASE_ROOT_WIDTH
    tip_width: float = BASE_TIP_WIDTH
    width_taper: float = 1.45
    curl_radius: float = 0.0
    curl_turns: float = 1.5
    curl_phase: float = 0.35
    root_color: tuple[float, float, float] = FUR_ROOT
    tip_color: tuple[float, float, float] = FUR_TIP
    root_opacity: float = 1.0
    tip_opacity: float = 1.0


@dataclass(frozen=True)
class Panel:
    key: str
    title: str
    symbol: str
    note: str
    labels: tuple[str, ...]
    specs: tuple[StrandSpec, ...]
    resolution: tuple[int, int] = (1080, 620)
    frame_margin: float = 1.20
    ortho_scale: float = 0.0
    reference_extent: float = 0.0
    target_x_shifts: tuple[float, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "paper" / "method",
    )
    parser.add_argument("--render-samples", type=int, default=64)
    parser.add_argument(
        "--palette",
        choices=("smoked_champagne", "copper"),
        default="smoked_champagne",
        help="Hair palette only; geometry, lighting, camera, and layout stay fixed.",
    )
    parser.add_argument("--skip-blender", action="store_true")
    parser.add_argument(
        "--single-panel",
        default="",
        help="Render and export one control panel by key, for example 'direction'.",
    )
    return parser.parse_args()


def direction_local(spec: StrandSpec) -> tuple[float, float, float]:
    angle = np.deg2rad(spec.angle_deg)
    azimuth = np.deg2rad(spec.azimuth_deg)
    tangent = np.sin(angle) * np.cos(azimuth)
    bitangent = np.sin(angle) * np.sin(azimuth)
    normal = np.cos(angle)
    return float(tangent), float(bitangent), float(normal)


def tensor_column(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64).view(-1, 1)


def tensor_rgb(values: list[tuple[float, float, float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def build_value_strands(
    spec: StrandSpec,
    *,
    adaptive_samples: bool = False,
    gaussian_strand_index: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    strand_count = 3
    roots = torch.tensor(
        [
            [-0.0016, -0.0032, 0.0],
            [0.0, 0.0, 0.0],
            [0.0016, 0.0032, 0.0],
        ],
        dtype=torch.float64,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).repeat(strand_count, 1)
    tangents = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).repeat(strand_count, 1)
    bitangents = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64).repeat(strand_count, 1)
    length_scales = (0.982, 1.0, 1.018)
    lengths = tensor_column([spec.length * scale for scale in length_scales])
    groom = DecodedGroom(
        length=lengths,
        root_width=tensor_column([spec.root_width] * strand_count),
        tip_width=tensor_column([spec.tip_width] * strand_count),
        width_taper=tensor_column([spec.width_taper] * strand_count),
        direction_local=torch.tensor(
            [direction_local(spec)] * strand_count, dtype=torch.float64
        ),
        brush_stiffness=tensor_column([spec.stiffness] * strand_count),
        curl_radius_ratio=tensor_column([spec.curl_radius] * strand_count)
        / lengths,
        curl_turns=tensor_column([spec.curl_turns] * strand_count),
        curl_phase=tensor_column([spec.curl_phase] * strand_count),
        child_radius=torch.zeros((strand_count, 1), dtype=torch.float64),
        clump_strength=torch.zeros((strand_count, 1), dtype=torch.float64),
        root_color=tensor_rgb([spec.root_color] * strand_count),
        tip_color=tensor_rgb([spec.tip_color] * strand_count),
        root_opacity=tensor_column([spec.root_opacity] * strand_count),
        tip_opacity=tensor_column([spec.tip_opacity] * strand_count),
        opacity=torch.ones((strand_count, 1), dtype=torch.float64),
    )
    strands, widths, colors, opacities = build_strands(
        roots, normals, tangents, bitangents, groom, samples=257
    )
    arrays: dict[str, np.ndarray] = {
        "strands": (strands * SCENE_SCALE).detach().cpu().float().numpy(),
        "widths": (widths * SCENE_SCALE).detach().cpu().float().numpy(),
        "colors": colors.detach().cpu().float().numpy(),
        "opacities": opacities.detach().cpu().float().numpy(),
        "root_ids": np.arange(strand_count, dtype=np.int64),
    }
    report: dict[str, float | int] = {
        "strand_count": strand_count,
        "geometry_samples": 257,
    }
    if adaptive_samples:
        if gaussian_strand_index is None:
            gaussian_slice = slice(None)
        else:
            if not 0 <= gaussian_strand_index < strand_count:
                raise ValueError(
                    f"gaussian_strand_index must be in [0, {strand_count}), "
                    f"got {gaussian_strand_index}"
                )
            gaussian_slice = slice(gaussian_strand_index, gaussian_strand_index + 1)
        selected = adaptive_resample_strands(
            strands[gaussian_slice],
            widths[gaussian_slice],
            colors[gaussian_slice],
            opacities[gaussian_slice],
            groom.length[gaussian_slice],
            min_segments=10,
            length_origin=0.010,
            segments_per_unit_length=84.19047619047619,
            segments_per_unit_complexity=23.771428571428572,
        )
        segment_count = int(selected.segment_counts[0].item())
        gaussians = strands_to_gaussians(
            selected.strands,
            selected.widths,
            selected.colors,
            selected.opacities,
            selected.segment_mask,
        )
        arrays["gaussian_means"] = (
            gaussians.means * SCENE_SCALE
        ).detach().cpu().float().numpy()
        arrays["gaussian_directions"] = gaussians.directions.detach().cpu().float().numpy()
        arrays["gaussian_scales"] = (
            gaussians.scales * SCENE_SCALE
        ).detach().cpu().float().numpy()
        arrays["gaussian_opacities"] = (
            gaussians.opacities
        ).detach().cpu().float().numpy().reshape(-1)
        report["adaptive_segments"] = segment_count
        report["adaptive_points"] = segment_count + 1
        report["gaussian_strand_count"] = int(selected.strands.shape[0])
    return arrays, report


def palette_colors(
    palette: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if palette == "copper":
        return FUR_ROOT, FUR_TIP
    if palette == "smoked_champagne":
        return SMOKED_CHAMPAGNE_FUR_ROOT, SMOKED_CHAMPAGNE_FUR_TIP
    raise ValueError(f"Unsupported palette: {palette}")


def control_panels(*, palette: str = "smoked_champagne") -> tuple[Panel, ...]:
    root_color, tip_color = palette_colors(palette)
    base = StrandSpec(root_color=root_color, tip_color=tip_color)
    opacity_base = replace(
        base,
        length=0.074,
        root_color=OPACITY_PANEL_COLOR,
        tip_color=OPACITY_PANEL_COLOR,
    )
    # Keep appearance examples independent from the presentation hair palette:
    # these three neutral profiles are the established root-tip color controls.
    appearance_specs = (
        replace(base, root_color=(0.68, 0.67, 0.63), tip_color=(0.68, 0.67, 0.63)),
        replace(base, root_color=(0.08, 0.09, 0.10), tip_color=(0.66, 0.68, 0.70)),
        replace(base, root_color=(0.45, 0.42, 0.35), tip_color=(0.91, 0.89, 0.82)),
    )
    return (
        Panel(
            "direction",
            "3D direction",
            r"$\mathbfit{d}$",
            "angle from surface normal",
            ("8°", "48°", "82°"),
            tuple(
                replace(
                    base,
                    angle_deg=angle_deg,
                    stiffness=0.76,
                    length=BASE_LENGTH * length_scale,
                )
                for angle_deg, length_scale in ((8.0, 0.68), (48.0, 0.82), (82.0, 1.0))
            ),
            ortho_scale=1.65,
            reference_extent=0.78,
            target_x_shifts=(0.0, 0.0, 0.090),
        ),
        Panel(
            "length",
            "Length",
            r"$\mathbfit{L/L}_{\mathrm{ref}}$",
            "normalized by reference length",
            ("0.45", "1.00", "1.75"),
            tuple(replace(base, length=BASE_LENGTH * value) for value in (0.45, 1.0, 1.75)),
            ortho_scale=2.45,
            reference_extent=0.78,
            target_x_shifts=(0.0, 0.0, 0.090),
        ),
        Panel(
            "stiffness",
            "Brush stiffness",
            r"$\mathbfit{s}$",
            "one normal-to-flow turn",
            ("0", "0.5", "1"),
            tuple(replace(base, angle_deg=78.0, stiffness=value) for value in (0.0, 0.5, 1.0)),
            ortho_scale=1.70,
            reference_extent=0.78,
            target_x_shifts=(0.070, 0.070, 0.070),
        ),
        Panel(
            "width",
            "Width profile",
            r"$\mathbfit{w(u)}$",
            "root, tip, and taper",
            ("0.65", "1.55", "2.10"),
            (
                replace(base, root_width=0.00065, tip_width=0.00016, width_taper=1.0),
                replace(base, root_width=0.00155, tip_width=0.00055, width_taper=0.72),
                replace(base, root_width=0.00210, tip_width=0.000045, width_taper=2.25),
            ),
            ortho_scale=1.60,
            reference_extent=0.78,
        ),
        Panel(
            "curl_radius",
            "Curl radius",
            r"$\mathbfit{r/L}$",
            "normalized by strand length",
            ("0", "0.07", "0.16"),
            tuple(
                replace(base, length=0.074, curl_radius=0.074 * value, curl_turns=1.65)
                for value in (0.0, 0.07, 0.16)
            ),
            ortho_scale=1.95,
            reference_extent=0.95,
        ),
        Panel(
            "curl_turns",
            "Curl turns",
            r"$\mathbfit{f}$",
            "turns along the final strand",
            ("0.5", "1.5", "3.0"),
            tuple(
                replace(base, length=0.074, curl_radius=0.0090, curl_turns=value)
                for value in (0.5, 1.5, 3.0)
            ),
            ortho_scale=1.95,
            reference_extent=0.95,
        ),
        Panel(
            "opacity",
            "Root-tip opacity",
            r"$\mathbfit{\alpha(u)}$",
            "",
            ("", "", ""),
            tuple(
                replace(
                    opacity_base,
                    root_opacity=root_opacity,
                    tip_opacity=tip_opacity,
                )
                for root_opacity, tip_opacity in TOP_OPACITY_PROFILES
            ),
            ortho_scale=1.95,
            reference_extent=0.95,
        ),
        Panel(
            "appearance",
            "Root-tip color",
            r"$\mathbfit{c(u)}$",
            "continuous appearance profile",
            ("color_0", "color_1", "color_2"),
            appearance_specs,
            ortho_scale=1.60,
            reference_extent=0.78,
        ),
    )


def composed_panel(*, palette: str = "smoked_champagne") -> Panel:
    root_color, tip_color = palette_colors(palette)
    base = StrandSpec(root_color=root_color, tip_color=tip_color)
    if palette == "smoked_champagne":
        composed_colors = (
            ((0.080, 0.055, 0.035), (0.420, 0.310, 0.200)),
            ((0.145, 0.105, 0.070), (0.660, 0.550, 0.400)),
            ((0.540, 0.430, 0.300), (0.120, 0.085, 0.055)),
            ((0.100, 0.065, 0.040), (0.460, 0.340, 0.220)),
            ((0.150, 0.110, 0.075), (0.700, 0.620, 0.480)),
        )
    else:
        composed_colors = (
            (root_color, tip_color),
            ((0.120, 0.050, 0.018), (0.625, 0.285, 0.060)),
            ((0.170, 0.075, 0.025), (0.780, 0.465, 0.135)),
            ((0.215, 0.095, 0.025), (0.620, 0.275, 0.055)),
            ((0.135, 0.060, 0.025), (0.800, 0.530, 0.190)),
        )
    return Panel(
        "composed",
        "Composed grooms",
        "",
        "multiple controls act on one editable strand model",
        ("sleek taper", "swept plume", "ribbon wave", "compact coil", "airy fade"),
        (
            replace(
                base,
                length=0.059,
                angle_deg=76.0,
                azimuth_deg=0.0,
                stiffness=0.84,
                root_width=0.00185,
                tip_width=0.000055,
                width_taper=2.25,
                curl_radius=0.0038,
                curl_turns=0.72,
                curl_phase=0.15,
                root_color=composed_colors[0][0],
                tip_color=composed_colors[0][1],
                root_opacity=COMPOSED_OPACITY_PROFILES[0][0],
                tip_opacity=COMPOSED_OPACITY_PROFILES[0][1],
            ),
            replace(
                base,
                length=0.068,
                angle_deg=58.0,
                azimuth_deg=4.0,
                stiffness=0.42,
                root_width=0.00205,
                tip_width=0.00018,
                width_taper=0.90,
                curl_radius=0.0062,
                curl_turns=1.10,
                curl_phase=0.0,
                root_color=composed_colors[1][0],
                tip_color=composed_colors[1][1],
                root_opacity=COMPOSED_OPACITY_PROFILES[1][0],
                tip_opacity=COMPOSED_OPACITY_PROFILES[1][1],
            ),
            replace(
                base,
                length=0.066,
                angle_deg=72.0,
                azimuth_deg=-3.0,
                stiffness=0.67,
                root_width=0.00158,
                tip_width=0.000085,
                width_taper=1.72,
                curl_radius=0.0078,
                curl_turns=1.70,
                curl_phase=1.05,
                root_color=composed_colors[2][0],
                tip_color=composed_colors[2][1],
                root_opacity=COMPOSED_OPACITY_PROFILES[2][0],
                tip_opacity=COMPOSED_OPACITY_PROFILES[2][1],
            ),
            replace(
                base,
                length=0.062,
                angle_deg=55.0,
                azimuth_deg=2.0,
                stiffness=0.35,
                root_width=0.00212,
                tip_width=0.000050,
                width_taper=2.30,
                curl_radius=0.0105,
                curl_turns=2.65,
                curl_phase=0.20,
                root_color=composed_colors[3][0],
                tip_color=composed_colors[3][1],
                root_opacity=COMPOSED_OPACITY_PROFILES[3][0],
                tip_opacity=COMPOSED_OPACITY_PROFILES[3][1],
            ),
            replace(
                base,
                length=0.062,
                angle_deg=59.0,
                azimuth_deg=0.0,
                stiffness=0.46,
                root_width=0.00128,
                tip_width=0.000040,
                width_taper=1.95,
                curl_radius=0.0045,
                curl_turns=1.25,
                curl_phase=0.50,
                root_color=composed_colors[4][0],
                tip_color=composed_colors[4][1],
                root_opacity=COMPOSED_OPACITY_PROFILES[4][0],
                tip_opacity=COMPOSED_OPACITY_PROFILES[4][1],
            ),
        ),
        resolution=(4320, 700),
        frame_margin=1.08,
        ortho_scale=0.82,
        reference_extent=0.92,
    )


def build_composed_scene_arrays(
    panel: Panel,
    *,
    gaussian_outlines: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    root_x = np.linspace(
        -2.0 * COMPOSED_ROOT_SPACING,
        2.0 * COMPOSED_ROOT_SPACING,
        len(panel.specs),
        dtype=np.float32,
    )
    root_z = COMPOSED_WAVE_AMPLITUDE * np.cos(
        COMPOSED_WAVE_FREQUENCY * root_x
    )
    groups: list[dict[str, np.ndarray]] = []
    group_reports: list[dict[str, object]] = []
    for group_index, (spec, x, z) in enumerate(zip(panel.specs, root_x, root_z)):
        arrays, stats = build_value_strands(
            spec,
            adaptive_samples=gaussian_outlines,
            gaussian_strand_index=1 if gaussian_outlines else None,
        )
        translation = np.asarray([x, 0.0, z], dtype=np.float32)
        arrays["strands"] = arrays["strands"] + translation[None, None, :]
        if gaussian_outlines:
            arrays["gaussian_means"] = arrays["gaussian_means"] + translation[None, :]
        groups.append(arrays)
        group_reports.append(
            {
                "group_index": group_index,
                "label": panel.labels[group_index],
                "root_position": [float(x), 0.0, float(z)],
                **stats,
            }
        )

    combined: dict[str, np.ndarray] = {
        key: np.concatenate([group[key] for group in groups], axis=0)
        for key in ("strands", "widths", "colors", "opacities")
    }
    combined["root_ids"] = np.arange(combined["strands"].shape[0], dtype=np.int64)
    if gaussian_outlines:
        for key in (
            "gaussian_means",
            "gaussian_directions",
            "gaussian_scales",
            "gaussian_opacities",
        ):
            combined[key] = np.concatenate([group[key] for group in groups], axis=0)
    return combined, {
        "group_count": len(panel.specs),
        "root_spacing": COMPOSED_ROOT_SPACING,
        "wave_amplitude": COMPOSED_WAVE_AMPLITUDE,
        "wave_frequency": COMPOSED_WAVE_FREQUENCY,
        "groups": group_reports,
    }


def presentation_command(
    *,
    blender: Path,
    renderer: Path,
    npz_path: Path,
    image_path: Path,
    resolution: tuple[int, int],
    render_samples: int,
    camera_offset: tuple[float, float, float],
    target_root_offset: tuple[float, float, float],
    frame_margin: float,
    reference_extent: float,
    ortho_scale: float,
    ground_relief: float,
    ground_width_scale: float,
    ground_depth_scale: float,
    ground_screen_height: float,
    gaussian_outlines: bool,
    ground_wave_amplitude: float = 0.0,
    ground_wave_frequency: float = 0.0,
    ground_wave_phase: float = 0.0,
    ground_base_z: float | None = None,
    use_input_opacities: bool = True,
    ground_color: tuple[float, float, float] = (0.20, 0.20, 0.20),
    world_strength: float = 0.85,
    key_light_type: str = "area",
    key_light_energy: float = 900.0,
    key_light_offset: tuple[float, float, float] = (-0.12, -0.18, 1.80),
    sun_angle_deg: float = 8.0,
    fill_light_energy: float = 260.0,
    fill_light_size: float = 2.0,
    shadow_sun_energy: float = 1.5,
    shadow_sun_offset: tuple[float, float, float] = (0.45, -0.12, 1.80),
    shadow_sun_angle_deg: float = 5.0,
) -> list[str]:
    command = [
        str(blender), "--background", "--python", str(renderer), "--",
        "--input", str(npz_path), "--output", str(image_path),
        "--resolution", str(resolution[0]), str(resolution[1]),
        "--samples", str(render_samples), "--width-scale", "1.0",
        "--material-color", "0.78", "0.34", "0.07",
        "--material-roughness", "0.38",
        "--material-specular", "0.50",
        "--background-color", "0.6654", "0.6795", "0.7084",
        "--world-strength", str(world_strength),
        "--camera-background-strength", "1.0",
        "--key-light-type", key_light_type,
        "--key-light-energy", str(key_light_energy),
        "--key-light-size", "1.6",
        "--key-light-offset", *(str(value) for value in key_light_offset),
        "--sun-angle-deg", str(sun_angle_deg),
        "--fill-light-energy", str(fill_light_energy),
        "--fill-light-size", str(fill_light_size),
        "--shadow-sun-energy", str(shadow_sun_energy),
        "--shadow-sun-offset", *(str(value) for value in shadow_sun_offset),
        "--shadow-sun-angle-deg", str(shadow_sun_angle_deg),
        "--camera-offset", *(f"{value:.8f}" for value in camera_offset),
        "--target-root-offset", *(f"{value:.8f}" for value in target_root_offset),
        "--coord-system", "identity",
        "--frame-margin", str(frame_margin),
        "--reference-extent", str(reference_extent),
        "--ground-plane",
        "--ground-color", *(str(value) for value in ground_color),
        "--ground-relief", str(ground_relief),
        "--ground-width-scale", str(ground_width_scale),
        "--ground-depth-scale", str(ground_depth_scale),
        "--ground-screen-height", str(ground_screen_height),
    ]
    if ortho_scale > 0.0:
        command.extend(["--ortho-scale", str(ortho_scale)])
    if abs(ground_wave_amplitude) > 0.0:
        command.extend(
            [
                "--ground-wave-amplitude", str(ground_wave_amplitude),
                "--ground-wave-frequency", str(ground_wave_frequency),
                "--ground-wave-phase", str(ground_wave_phase),
            ]
        )
    if ground_base_z is not None:
        command.extend(["--ground-base-z", str(ground_base_z)])
    if gaussian_outlines:
        command.extend(
            [
                "--gaussian-outline-only",
                "--gaussian-outline-color", "0.10", "0.12", "0.15",
                "--gaussian-accent-color", "0.72", "0.36", "0.07",
                "--gaussian-outline-width", "0.00190",
                "--gaussian-outline-scale", "1.0",
            ]
        )
    else:
        command.append("--use-input-colors")
        if use_input_opacities:
            command.append("--use-input-opacities")
    return command


def render_panel(
    panel: Panel,
    *,
    blender: Path,
    renderer: Path,
    work_dir: Path,
    render_samples: int,
    final_sampling: bool,
    gaussian_outlines: bool = False,
    use_input_opacities: bool = False,
) -> dict[str, object]:
    value_reports: list[dict[str, object]] = []
    if panel.target_x_shifts and len(panel.target_x_shifts) != len(panel.specs):
        raise ValueError(
            f"{panel.key}: target_x_shifts must match specs; "
            f"got {len(panel.target_x_shifts)} and {len(panel.specs)}"
        )
    for value_index, spec in enumerate(panel.specs):
        suffix = "_gaussians" if gaussian_outlines else ""
        stem = f"{panel.key}_{value_index}{suffix}"
        npz_path = work_dir / f"{stem}.npz"
        image_path = work_dir / f"{stem}.png"
        arrays, stats = build_value_strands(
            spec,
            adaptive_samples=final_sampling or gaussian_outlines,
            gaussian_strand_index=1 if gaussian_outlines else None,
        )
        np.savez(npz_path, **arrays)
        item_width = panel.resolution[0] // len(panel.specs)
        item_aspect = float(item_width) / float(panel.resolution[1])
        target_x_shift = (
            panel.target_x_shifts[value_index] if panel.target_x_shifts else 0.0
        )
        root_target_offset = (
            (0.32 + target_x_shift) * panel.ortho_scale * item_aspect,
            0.0,
            0.30 * panel.ortho_scale,
        )
        command = presentation_command(
            blender=blender,
            renderer=renderer,
            npz_path=npz_path,
            image_path=image_path,
            resolution=(item_width, panel.resolution[1]),
            render_samples=render_samples,
            camera_offset=(0.0, -1.0, 0.09),
            target_root_offset=root_target_offset,
            frame_margin=panel.frame_margin,
            reference_extent=panel.reference_extent,
            ortho_scale=panel.ortho_scale,
            ground_relief=0.040,
            ground_width_scale=2.40,
            ground_depth_scale=0.55,
            ground_screen_height=0.10,
            gaussian_outlines=gaussian_outlines,
            use_input_opacities=use_input_opacities,
        )
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Blender failed for {stem}:\n{completed.stdout}\n{completed.stderr}"
            )
        if not image_path.exists():
            raise RuntimeError(f"Blender did not produce {image_path}")
        value_reports.append(
            {"npz": str(npz_path.resolve()), "image": str(image_path.resolve()), **stats}
        )
    return {
        "key": panel.key,
        "title": panel.title,
        "note": panel.note,
        "labels": list(panel.labels),
        "values": value_reports,
        "specs": [spec.__dict__ for spec in panel.specs],
        **value_reports[0],
    }


def render_composed_scene(
    panel: Panel,
    *,
    blender: Path,
    renderer: Path,
    work_dir: Path,
    render_samples: int,
) -> dict[str, object]:
    reports: dict[str, object] = {}
    for gaussian_outlines in (False, True):
        suffix = "_gaussians" if gaussian_outlines else ""
        stem = f"{COMPOSED_SCENE_STEM}{suffix}"
        npz_path = work_dir / f"{stem}.npz"
        image_path = work_dir / f"{stem}.png"
        arrays, scene_report = build_composed_scene_arrays(
            panel,
            gaussian_outlines=gaussian_outlines,
        )
        np.savez(npz_path, **arrays)
        command = presentation_command(
            blender=blender,
            renderer=renderer,
            npz_path=npz_path,
            image_path=image_path,
            resolution=panel.resolution,
            render_samples=render_samples,
            camera_offset=COMPOSED_CAMERA_OFFSET,
            target_root_offset=COMPOSED_TARGET_ROOT_OFFSET,
            frame_margin=1.0,
            reference_extent=1.0,
            ortho_scale=COMPOSED_ORTHO_SCALE,
            ground_relief=0.010,
            ground_width_scale=4.40,
            ground_depth_scale=0.55,
            ground_screen_height=COMPOSED_GROUND_SCREEN_HEIGHT,
            gaussian_outlines=gaussian_outlines,
            ground_wave_amplitude=COMPOSED_WAVE_AMPLITUDE,
            ground_wave_frequency=COMPOSED_WAVE_FREQUENCY,
            ground_wave_phase=0.0,
            ground_base_z=0.0,
            ground_color=(1.0, 1.0, 1.0),
            world_strength=0.32,
            key_light_type="sun",
            key_light_energy=3.4,
            key_light_offset=(-1.00, -0.05, 1.25),
            sun_angle_deg=4.0,
            fill_light_energy=110.0,
            fill_light_size=8.0,
            shadow_sun_energy=0.0,
            use_input_opacities=True,
        )
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Blender failed for {stem}:\n{completed.stdout}\n{completed.stderr}"
            )
        if not image_path.exists():
            raise RuntimeError(f"Blender did not produce {image_path}")
        reports["gaussians" if gaussian_outlines else "strands"] = {
            "npz": str(npz_path.resolve()),
            "image": str(image_path.resolve()),
            **scene_report,
        }
    return {
        "key": panel.key,
        "title": panel.title,
        "labels": list(panel.labels),
        "specs": [spec.__dict__ for spec in panel.specs],
        "scene": reports,
    }


def opacity_swatch_alpha_profile(
    root_opacity: float,
    tip_opacity: float,
) -> np.ndarray:
    return np.linspace(
        float(root_opacity),
        float(tip_opacity),
        VALUE_SWATCH_SEGMENT_COUNT,
        dtype=np.float32,
    )


def opacity_swatch_checkerboard(
    *,
    segment_count: int = VALUE_SWATCH_SEGMENT_COUNT,
) -> np.ndarray:
    if segment_count <= 0 or segment_count % OPACITY_SWATCH_CHECKER_COLUMNS:
        raise ValueError("segment_count must be a positive multiple of checker columns")
    column_width = segment_count // OPACITY_SWATCH_CHECKER_COLUMNS
    columns = np.arange(segment_count, dtype=np.int64) // column_width
    light = np.asarray(OPACITY_SWATCH_CHECKER_LIGHT, dtype=np.float32)
    dark = np.asarray(OPACITY_SWATCH_CHECKER_DARK, dtype=np.float32)
    return np.stack(
        [
            np.where(
                ((columns + row)[:, None] % 2) == 0,
                light,
                dark,
            )
            for row in range(2)
        ],
        axis=0,
    )


def _opacity_swatch_layer_offset(ax: plt.Axes) -> float:
    axis_height_pixels = float(ax.bbox.height)
    if not np.isfinite(axis_height_pixels) or axis_height_pixels <= 0.0:
        raise RuntimeError("opacity swatch requires a measurable axes height")
    return (
        VALUE_SWATCH_LINE_WIDTH
        * float(ax.figure.dpi)
        / (72.0 * 4.0 * axis_height_pixels)
    )


def add_opacity_swatch(
    ax: plt.Axes,
    *,
    position: float,
    spec: StrandSpec,
) -> None:
    endpoint_span = VALUE_SWATCH_ENDPOINT_SPAN
    profile_y = VALUE_SWATCH_Y
    profile_t = np.linspace(
        0.0,
        1.0,
        VALUE_SWATCH_SEGMENT_COUNT + 1,
        dtype=np.float32,
    )
    root_x = position - 0.5 * endpoint_span
    profile_x = root_x + endpoint_span * profile_t
    profile_points = np.column_stack(
        (profile_x, np.full_like(profile_x, profile_y))
    )
    profile_segments = np.stack((profile_points[:-1], profile_points[1:]), axis=1)

    checker = opacity_swatch_checkerboard()
    layer_offset = _opacity_swatch_layer_offset(ax)
    for row, sign in enumerate((-1.0, 1.0)):
        layer_points = profile_points.copy()
        layer_points[:, 1] += sign * layer_offset
        layer_segments = np.stack((layer_points[:-1], layer_points[1:]), axis=1)
        checker_layer = LineCollection(
            layer_segments,
            colors=checker[row],
            linewidths=VALUE_SWATCH_LINE_WIDTH / 2.0,
            capstyle="butt",
            transform=ax.transAxes,
            zorder=7,
        )
        ax.add_collection(checker_layer)

    alpha = opacity_swatch_alpha_profile(spec.root_opacity, spec.tip_opacity)
    rgb = np.asarray(OPACITY_PANEL_COLOR, dtype=np.float32)
    alpha_colors = np.column_stack(
        (np.repeat(rgb[None, :], VALUE_SWATCH_SEGMENT_COUNT, axis=0), alpha)
    )
    opacity_profile = LineCollection(
        profile_segments,
        colors=alpha_colors,
        linewidths=VALUE_SWATCH_LINE_WIDTH,
        capstyle="butt",
        transform=ax.transAxes,
        zorder=8,
    )
    ax.add_collection(opacity_profile)


def add_render_panel(
    ax: plt.Axes,
    *,
    panel: Panel,
    image_paths: tuple[Path, ...],
) -> None:
    positions = np.linspace(1.0 / 6.0, 5.0 / 6.0, len(panel.labels))
    half_width = 0.156
    for value_index, (position, image_path) in enumerate(zip(positions, image_paths)):
        image = Image.open(image_path).convert("RGB")
        image = image.crop(
            (
                0,
                int(round(image.height * CONTROL_PANEL_TOP_CROP)),
                image.width,
                int(round(image.height * CONTROL_PANEL_BOTTOM_CROP)),
            )
        )
        x0 = position - half_width
        x1 = position + half_width
        ax.imshow(image, extent=(x0, x1, 0.035, 0.91), aspect="auto")
        if panel.key == "appearance":
            spec = panel.specs[value_index]
            root_color = np.asarray(spec.root_color, dtype=np.float32)
            tip_color = np.asarray(spec.tip_color, dtype=np.float32)
            endpoint_span = VALUE_SWATCH_ENDPOINT_SPAN
            root_x = position - 0.5 * endpoint_span
            profile_y = VALUE_SWATCH_Y
            profile_t = np.linspace(
                0.0,
                1.0,
                VALUE_SWATCH_SEGMENT_COUNT + 1,
                dtype=np.float32,
            )
            profile_x = root_x + endpoint_span * profile_t
            profile_colors = (
                (1.0 - profile_t[:-1, None]) * root_color
                + profile_t[:-1, None] * tip_color
            )
            profile_points = np.column_stack(
                (profile_x, np.full_like(profile_x, profile_y))
            )
            profile_segments = np.stack((profile_points[:-1], profile_points[1:]), axis=1)
            color_profile = LineCollection(
                profile_segments,
                colors=profile_colors,
                linewidths=VALUE_SWATCH_LINE_WIDTH,
                capstyle="butt",
                transform=ax.transAxes,
                zorder=7,
            )
            ax.add_collection(color_profile)
        elif panel.key == "opacity":
            add_opacity_swatch(
                ax,
                position=position,
                spec=panel.specs[value_index],
            )
        else:
            text = ax.text(
                position,
                0.083,
                panel.labels[value_index],
                transform=ax.transAxes,
                fontproperties=VALUE_FONT,
                ha="center",
                va="center",
                color="#30363d",
                zorder=6,
            )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    title_parts = [
        TextArea(panel.title, textprops={"fontproperties": TITLE_FONT, "color": "#20242a"})
    ]
    if panel.symbol:
        title_parts.append(
            TextArea(panel.symbol, textprops={"fontproperties": SYMBOL_FONT, "color": "#20242a"})
        )
    title = AnchoredOffsetbox(
        loc="upper left",
        child=HPacker(children=title_parts, align="baseline", pad=0.0, sep=3.0),
        frameon=False,
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
        pad=0.0,
    )
    ax.add_artist(title)


def add_composed_panel(
    ax: plt.Axes,
    *,
    panel: Panel,
    work_dir: Path,
) -> None:
    upper = Image.open(work_dir / f"{COMPOSED_SCENE_STEM}.png").convert("RGB")
    lower = Image.open(work_dir / f"{COMPOSED_SCENE_STEM}_gaussians.png").convert("RGB")
    ax.imshow(
        upper,
        extent=(
            0.0,
            1.0,
            0.500 + COMPOSED_IMAGE_SHIFT,
            CONTROL_PANEL_IMAGE_TOP + COMPOSED_IMAGE_SHIFT,
        ),
        aspect="auto",
    )
    ax.imshow(
        lower,
        extent=(0.0, 1.0, 0.102 + COMPOSED_IMAGE_SHIFT, 0.512 + COMPOSED_IMAGE_SHIFT),
        aspect="auto",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    title = AnchoredOffsetbox(
        loc="upper left",
        child=TextArea(
            panel.title,
            textprops={"fontproperties": TITLE_FONT, "color": "#20242a"},
        ),
        frameon=False,
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
        pad=0.0,
    )
    ax.add_artist(title)


def compose_figure(
    *,
    panels: tuple[Panel, ...],
    composed: Panel,
    work_dir: Path,
    output_dir: Path,
    report: dict[str, object],
    output_stem: str = FINAL_STEM,
) -> None:
    configure_figure_fonts()
    figure = plt.figure(figsize=(7.50, 6.60), facecolor="white")
    grid = figure.add_gridspec(
        3,
        4,
        left=0.025,
        right=0.975,
        bottom=0.020,
        top=0.985,
        hspace=0.02,
        wspace=0.02,
        height_ratios=(1.0, 1.0, COMPOSED_ROW_HEIGHT_RATIO),
    )

    for index, panel in enumerate(panels):
        row, column = divmod(index, 4)
        add_render_panel(
            figure.add_subplot(grid[row, column]),
            panel=panel,
            image_paths=tuple(work_dir / f"{panel.key}_{i}.png" for i in range(len(panel.labels))),
        )
    add_composed_panel(
        figure.add_subplot(grid[2, :]),
        panel=composed,
        work_dir=work_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in (("png", 600), ("pdf", 300), ("svg", 300)):
        output_path = output_dir / f"{output_stem}.{suffix}"
        figure.savefig(
            output_path,
            dpi=dpi,
            facecolor="white",
            bbox_inches=None,
            pad_inches=0.0,
        )
        if suffix == "svg":
            svg = output_path.read_text(encoding="utf-8")
            output_path.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(figure)


def compose_single_panel(
    *,
    panel: Panel,
    work_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    configure_figure_fonts()
    figure = plt.figure(figsize=(4.6, 3.6), facecolor="white")
    axis = figure.add_axes((0.02, 0.02, 0.96, 0.96))
    add_render_panel(
        axis,
        panel=panel,
        image_paths=tuple(
            work_dir / f"{panel.key}_{index}.png"
            for index in range(len(panel.labels))
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig_parametric_groom_{panel.key}"
    outputs: dict[str, str] = {}
    for suffix, dpi in (("png", 600), ("pdf", 300), ("svg", 300)):
        output = output_dir / f"{stem}.{suffix}"
        figure.savefig(
            output,
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.03,
        )
        outputs[suffix] = str(output.resolve())
    plt.close(figure)
    return outputs


def main() -> None:
    args = parse_args()
    if not args.blender.exists():
        raise FileNotFoundError(args.blender)
    renderer = Path(__file__).with_name("render_parametric_groom_blender.py")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    output_stem = (
        FINAL_STEM
        if args.palette == "smoked_champagne"
        else f"{FINAL_STEM}_{args.palette}"
    )
    panels = control_panels(palette=args.palette)
    composed = composed_panel(palette=args.palette)

    if args.single_panel:
        matching_panels = [panel for panel in panels if panel.key == args.single_panel]
        if len(matching_panels) != 1:
            valid = ", ".join(panel.key for panel in panels)
            raise ValueError(
                f"Unknown single panel '{args.single_panel}'; choose one of: {valid}"
            )
        panel = matching_panels[0]
        if args.skip_blender:
            missing = [
                args.work_dir / f"{panel.key}_{index}.png"
                for index in range(len(panel.specs))
                if not (args.work_dir / f"{panel.key}_{index}.png").exists()
            ]
            if missing:
                raise FileNotFoundError(f"Missing cached Blender renders: {missing}")
            cached_report = args.work_dir / "render_report.json"
            if not cached_report.exists():
                raise FileNotFoundError(cached_report)
            panel_report = json.loads(cached_report.read_text(encoding="utf-8"))
        else:
            panel_report = render_panel(
                panel,
                blender=args.blender,
                renderer=renderer,
                work_dir=args.work_dir,
                render_samples=args.render_samples,
                final_sampling=False,
                use_input_opacities=(panel.key == "opacity"),
            )
            (args.work_dir / "render_report.json").write_text(
                json.dumps(panel_report, indent=2) + "\n",
                encoding="utf-8",
            )
        outputs = compose_single_panel(
            panel=panel,
            work_dir=args.work_dir,
            output_dir=args.output_dir,
        )
        report_path = args.work_dir / f"fig_parametric_groom_{panel.key}.json"
        report_path.write_text(
            json.dumps(panel_report, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs["report"] = str(report_path.resolve())
        print(json.dumps(outputs, indent=2))
        return

    report: dict[str, object] = {
        "scope": "control panel only; no surface-frame or standalone conversion panels",
        "geometry": "formal build_strands output",
        "sampling_order": "brush backbone -> curl -> final curve -> adaptive resampling",
        "renderer": str(renderer.resolve()),
        "palette": args.palette,
        "renders": {},
    }
    if args.skip_blender:
        missing = [
            f"{panel.key}_{index}"
            for panel in panels
            for index in range(len(panel.specs))
            if not (args.work_dir / f"{panel.key}_{index}.png").exists()
        ]
        missing.extend(
            stem
            for stem in (COMPOSED_SCENE_STEM, f"{COMPOSED_SCENE_STEM}_gaussians")
            if not (args.work_dir / f"{stem}.png").exists()
        )
        if missing:
            raise FileNotFoundError(f"Missing cached Blender renders: {missing}")
        cached_report = args.work_dir / "render_report.json"
        if not cached_report.exists():
            raise FileNotFoundError(cached_report)
        report = json.loads(cached_report.read_text(encoding="utf-8"))
    else:
        for panel in panels:
            report["renders"][panel.key] = render_panel(
                panel,
                blender=args.blender,
                renderer=renderer,
                work_dir=args.work_dir,
                render_samples=args.render_samples,
                final_sampling=False,
                use_input_opacities=(panel.key == "opacity"),
            )
        report["renders"]["composed"] = render_composed_scene(
            composed,
            blender=args.blender,
            renderer=renderer,
            work_dir=args.work_dir,
            render_samples=args.render_samples,
        )
        (args.work_dir / "render_report.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

    compose_figure(
        panels=panels,
        composed=composed,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        report=report,
        output_stem=output_stem,
    )
    print(
        json.dumps(
            {
                "png": str((args.output_dir / f"{output_stem}.png").resolve()),
                "pdf": str((args.output_dir / f"{output_stem}.pdf").resolve()),
                "svg": str((args.output_dir / f"{output_stem}.svg").resolve()),
                "render_report": str((args.work_dir / "render_report.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

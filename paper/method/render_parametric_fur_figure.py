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
from matplotlib.patches import Rectangle
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
SCENE_SCALE = 12.0
BASE_LENGTH = 0.064
BASE_ROOT_WIDTH = 0.00145
BASE_TIP_WIDTH = 0.00018
FUR_ROOT = (0.17, 0.075, 0.025)
FUR_TIP = (0.72, 0.36, 0.075)
SMOKED_CHAMPAGNE_FUR_ROOT = (0.145, 0.105, 0.070)
SMOKED_CHAMPAGNE_FUR_TIP = (0.50, 0.39, 0.255)


def font_property(filename: str, *, size: float, style: str = "normal") -> font_manager.FontProperties:
    path = Path(r"C:\Windows\Fonts") / filename
    if path.exists():
        return font_manager.FontProperties(fname=str(path), size=size, style=style)
    return font_manager.FontProperties(family="Arial", size=size, style=style)


TITLE_FONT = font_property("arialbd.ttf", size=12.5)
SYMBOL_FONT = font_property("arialbi.ttf", size=12.5, style="italic")
VALUE_FONT = font_property("arial.ttf", size=10.6)
COMPOSED_VALUE_FONT = font_property("arialbd.ttf", size=11.4)


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
    frizz: float = 0.0
    frizz_seed: float = 1.23
    root_color: tuple[float, float, float] = FUR_ROOT
    tip_color: tuple[float, float, float] = FUR_TIP


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
    groom = DecodedGroom(
        length=tensor_column([spec.length * scale for scale in length_scales]),
        root_width=tensor_column([spec.root_width] * strand_count),
        tip_width=tensor_column([spec.tip_width] * strand_count),
        width_taper=tensor_column([spec.width_taper] * strand_count),
        direction_local=torch.tensor(
            [direction_local(spec)] * strand_count, dtype=torch.float64
        ),
        brush_stiffness=tensor_column([spec.stiffness] * strand_count),
        curl_radius=tensor_column([spec.curl_radius] * strand_count),
        curl_turns=tensor_column([spec.curl_turns] * strand_count),
        curl_phase=tensor_column([spec.curl_phase] * strand_count),
        frizz=tensor_column([spec.frizz] * strand_count),
        frizz_seed_phase=tensor_column([spec.frizz_seed] * strand_count),
        child_radius=torch.zeros((strand_count, 1), dtype=torch.float64),
        clump_strength=torch.zeros((strand_count, 1), dtype=torch.float64),
        root_color=tensor_rgb([spec.root_color] * strand_count),
        tip_color=tensor_rgb([spec.tip_color] * strand_count),
        root_opacity=torch.ones((strand_count, 1), dtype=torch.float64),
        tip_opacity=torch.ones((strand_count, 1), dtype=torch.float64),
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
    if palette == "copper":
        appearance_specs = (
            replace(base, root_color=(0.68, 0.67, 0.63), tip_color=(0.68, 0.67, 0.63)),
            replace(base, root_color=(0.08, 0.09, 0.10), tip_color=(0.66, 0.68, 0.70)),
            replace(base, root_color=(0.45, 0.42, 0.35), tip_color=(0.91, 0.89, 0.82)),
        )
    else:
        root = np.asarray(root_color, dtype=np.float64)
        tip = np.asarray(tip_color, dtype=np.float64)
        dark = tuple(float(value) for value in np.clip(root * 0.58, 0.0, 1.0))
        light = tuple(float(value) for value in np.clip(tip * 1.18, 0.0, 0.92))
        appearance_specs = (
            replace(base, root_color=root_color, tip_color=root_color),
            replace(base, root_color=dark, tip_color=tip_color),
            replace(base, root_color=root_color, tip_color=light),
        )
    return (
        Panel(
            "direction",
            "3D direction",
            "d",
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
            r"$\ell/\ell_{\mathrm{ref}}$",
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
            "s",
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
            "w(u)",
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
            r"$r/\ell$",
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
            "f",
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
            "frizz",
            "Micro-frizz",
            r"$a/\ell$",
            "normalized by strand length",
            ("0", "0.035", "0.09"),
            tuple(
                replace(base, length=0.074, frizz=0.074 * value)
                for value in (0.0, 0.035, 0.09)
            ),
            ortho_scale=1.95,
            reference_extent=0.95,
        ),
        Panel(
            "appearance",
            "Root-tip color",
            "c(u)",
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
    return Panel(
        "composed",
        "Composed grooms",
        "",
        "multiple controls act on one editable strand model",
        ("sleek", "brushed", "soft wave", "loose curl", "spring curl", "fine frizz"),
        (
            replace(
                base,
                length=0.044,
                angle_deg=55.0,
                stiffness=0.24,
                root_width=0.00075,
                tip_width=0.00010,
            ),
            replace(
                base,
                length=0.060,
                angle_deg=74.0,
                stiffness=0.96,
                root_width=0.00145,
                tip_width=0.00011,
                frizz=0.00055,
            ),
            replace(
                base,
                length=0.064,
                angle_deg=57.0,
                stiffness=0.70,
                root_width=0.00165,
                tip_width=0.00016,
                curl_radius=0.0048,
                curl_turns=1.05,
                frizz=0.00070,
            ),
            replace(
                base,
                length=0.062,
                angle_deg=66.0,
                stiffness=0.80,
                root_width=0.00180,
                tip_width=0.00011,
                curl_radius=0.0068,
                curl_turns=1.55,
                frizz=0.0018,
            ),
            replace(
                base,
                length=0.064,
                angle_deg=62.0,
                stiffness=0.72,
                root_width=0.00172,
                tip_width=0.00009,
                curl_radius=0.0084,
                curl_turns=2.60,
                frizz=0.0025,
            ),
            replace(
                base,
                length=0.062,
                angle_deg=52.0,
                stiffness=0.58,
                root_width=0.00120,
                tip_width=0.00007,
                curl_radius=0.0030,
                curl_turns=1.35,
                frizz=0.0060,
                frizz_seed=2.17,
            ),
        ),
        resolution=(4320, 540),
        frame_margin=1.08,
        ortho_scale=0.82,
        reference_extent=0.92,
    )


def render_panel(
    panel: Panel,
    *,
    blender: Path,
    renderer: Path,
    work_dir: Path,
    render_samples: int,
    final_sampling: bool,
    gaussian_outlines: bool = False,
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
        camera_offset = ("0.07", "-1.0", "0.14") if panel.key == "composed" else ("0.0", "-1.0", "0.09")
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
        command = [
            str(blender), "--background", "--python", str(renderer), "--",
            "--input", str(npz_path), "--output", str(image_path),
            "--resolution", str(panel.resolution[0] // len(panel.specs)), str(panel.resolution[1]),
            "--samples", str(render_samples), "--width-scale", "1.0",
            "--material-color", "0.78", "0.34", "0.07",
            "--material-roughness", "0.38",
            "--material-specular", "0.50",
            "--background-color", "0.6654", "0.6795", "0.7084",
            "--world-strength", "0.85",
            "--camera-background-strength", "1.0",
            "--key-light-type", "area",
            "--key-light-energy", "900",
            "--key-light-size", "1.6",
            "--fill-light-energy", "260",
            "--shadow-sun-energy", "1.5",
            "--shadow-sun-offset", "0.45", "-0.12", "1.80",
            "--shadow-sun-angle-deg", "5.0",
            "--camera-offset", *camera_offset,
            "--target-root-offset", *(f"{value:.8f}" for value in root_target_offset),
            "--coord-system", "identity",
            "--frame-margin", str(panel.frame_margin),
            "--reference-extent", str(panel.reference_extent),
        ]
        if panel.ortho_scale > 0.0:
            command.extend(["--ortho-scale", str(panel.ortho_scale)])
        command.extend(
            [
                "--ground-plane",
                "--ground-color", "0.20", "0.20", "0.20",
                "--ground-relief", "0.040",
                "--ground-width-scale", "2.40",
                "--ground-depth-scale", "0.55",
                "--ground-screen-height", "0.10",
            ]
        )
        if not gaussian_outlines:
            command.append("--use-input-colors")
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
                int(round(image.height * 0.300)),
                image.width,
                int(round(image.height * 0.975)),
            )
        )
        x0 = position - half_width
        x1 = position + half_width
        ax.imshow(image, extent=(x0, x1, 0.035, 0.91), aspect="auto")
        if panel.key == "appearance":
            spec = panel.specs[value_index]
            root_color = np.asarray(spec.root_color, dtype=np.float32)
            tip_color = np.asarray(spec.tip_color, dtype=np.float32)
            endpoint_span = 0.170
            root_x = position - 0.5 * endpoint_span
            profile_y = 0.083
            profile_t = np.linspace(0.0, 1.0, 96, dtype=np.float32)
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
                linewidths=3.2,
                capstyle="butt",
                transform=ax.transAxes,
                zorder=7,
            )
            ax.add_collection(color_profile)
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
    positions = np.linspace(1.0 / 12.0, 11.0 / 12.0, len(panel.labels))
    half_width = 1.0 / 12.0
    for value_index, position in enumerate(positions):
        upper = Image.open(work_dir / f"{panel.key}_{value_index}.png").convert("RGB")
        lower = Image.open(work_dir / f"{panel.key}_{value_index}_gaussians.png").convert("RGB")
        upper = upper.crop((0, int(0.025 * upper.height), upper.width, int(0.975 * upper.height)))
        lower = lower.crop((0, int(0.025 * lower.height), lower.width, int(0.975 * lower.height)))
        x0 = position - half_width
        x1 = position + half_width
        ax.imshow(upper, extent=(x0, x1, 0.545, 0.910), aspect="auto")
        ax.imshow(lower, extent=(x0, x1, 0.075, 0.445), aspect="auto")
        ax.add_patch(
            Rectangle(
                (x0, 0.445),
                x1 - x0,
                0.100,
                transform=ax.transAxes,
                facecolor=(0.94, 0.945, 0.95, 0.88),
                edgecolor="none",
                zorder=4,
            )
        )
        ax.text(
            position,
            0.495,
            panel.labels[value_index],
            transform=ax.transAxes,
            fontproperties=COMPOSED_VALUE_FONT,
            ha="center",
            va="center",
            color="#30363d",
            zorder=6,
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
    plt.rcParams.update({"font.family": "Arial", "axes.unicode_minus": False})
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
        height_ratios=(1.0, 1.0, 1.46),
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
        figure.savefig(
            output_dir / f"{output_stem}.{suffix}",
            dpi=dpi,
            facecolor="white",
            bbox_inches=None,
            pad_inches=0.0,
        )
    plt.close(figure)


def compose_single_panel(
    *,
    panel: Panel,
    work_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
        }
    )
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
    all_panels = (*panels, composed)

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
        "sampling_order": "brush backbone -> curl/frizz -> final curve -> adaptive resampling",
        "renderer": str(renderer.resolve()),
        "palette": args.palette,
        "renders": {},
    }
    if args.skip_blender:
        missing = [
            f"{panel.key}_{index}"
            for panel in all_panels
            for index in range(len(panel.specs))
            if not (args.work_dir / f"{panel.key}_{index}.png").exists()
        ]
        missing.extend(
            f"{composed.key}_{index}_gaussians"
            for index in range(len(composed.specs))
            if not (args.work_dir / f"{composed.key}_{index}_gaussians.png").exists()
        )
        if missing:
            raise FileNotFoundError(f"Missing cached Blender renders: {missing}")
        cached_report = args.work_dir / "render_report.json"
        if not cached_report.exists():
            raise FileNotFoundError(cached_report)
        report = json.loads(cached_report.read_text(encoding="utf-8"))
    else:
        for panel in all_panels:
            report["renders"][panel.key] = render_panel(
                panel,
                blender=args.blender,
                renderer=renderer,
                work_dir=args.work_dir,
                render_samples=args.render_samples,
                final_sampling=False,
            )
        report["renders"]["composed_gaussians"] = render_panel(
            composed,
            blender=args.blender,
            renderer=renderer,
            work_dir=args.work_dir,
            render_samples=args.render_samples,
            final_sampling=True,
            gaussian_outlines=True,
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

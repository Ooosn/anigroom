"""Validate that groom strands map to continuous anisotropic 3D Gaussians.

This tool checks the narrow contract needed for Blender groom compatibility:
given strand polylines shaped [N, S, 3], the AniGroom strand-to-Gaussian path
must produce one elongated Gaussian for every consecutive output segment.
It does not train roots, run densification, or use a fallback renderer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import resample_strands_to_segment_budgets, strands_to_gaussians


def _load_npz(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(path)
    key = "strands" if "strands" in data else "points"
    if key not in data:
        raise ValueError(f"{path} must contain a 'strands' or 'points' array")
    strands = torch.tensor(data[key], dtype=torch.float32)
    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands/points must have shape [N, S, 3]")
    n, s, _ = strands.shape

    if "widths" in data:
        widths = torch.tensor(data["widths"], dtype=torch.float32)
    elif "width" in data:
        widths = torch.tensor(data["width"], dtype=torch.float32)
    else:
        widths = torch.full((n, s, 1), 0.003, dtype=torch.float32)
    if widths.ndim == 2:
        widths = widths[..., None]

    if "colors" in data:
        colors = torch.tensor(data["colors"], dtype=torch.float32)
    else:
        colors = torch.ones((n, s, 3), dtype=torch.float32) * torch.tensor([0.92, 0.90, 0.84])

    if "opacities" in data:
        opacities = torch.tensor(data["opacities"], dtype=torch.float32)
    else:
        opacities = torch.ones((n, s, 1), dtype=torch.float32) * 0.75
    if opacities.ndim == 2:
        opacities = opacities[..., None]

    if widths.shape[:2] != strands.shape[:2] or colors.shape[:2] != strands.shape[:2] or opacities.shape[:2] != strands.shape[:2]:
        raise ValueError("widths/colors/opacities must match strand sample dimensions")
    return strands, widths, colors, opacities


def _make_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.linspace(0.0, 1.0, 17)
    strands = []
    for i in range(5):
        x = -0.06 + 0.03 * i + 0.015 * t
        y = 0.02 * torch.sin((1.0 + 0.25 * i) * np.pi * t)
        z = 0.12 * t + 0.02 * i
        strands.append(torch.stack([x, y, z], dim=-1))
    strands_t = torch.stack(strands, dim=0)
    widths = (0.004 * (1.0 - 0.75 * t)).view(1, -1, 1).expand(strands_t.shape[0], -1, -1).contiguous()
    colors = torch.ones((*strands_t.shape[:2], 3), dtype=torch.float32)
    colors[..., 0] = 0.92
    colors[..., 1] = 0.90
    colors[..., 2] = 0.84
    opacities = torch.ones((*strands_t.shape[:2], 1), dtype=torch.float32) * 0.75
    return strands_t, widths, colors, opacities


def _rotate_x_axis(quats: torch.Tensor) -> torch.Tensor:
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dim=-1,
    )


def _save_preview(path: Path, strands: torch.Tensor, means: torch.Tensor) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 4.5), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    for strand in strands.detach().cpu().numpy():
        ax.plot(strand[:, 0], strand[:, 1], strand[:, 2], color="#1f77b4", linewidth=1.8)
    m = means.detach().cpu().numpy()
    ax.scatter(m[:, 0], m[:, 1], m[:, 2], s=5, color="#d62728", alpha=0.8)
    ax.set_title("strand polyline and Gaussian segment centers")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-65)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strand-to-Gaussian correspondence.")
    parser.add_argument("--input", type=Path, default=None, help="Blender-exported .npz containing strands/widths/colors/opacities.")
    parser.add_argument("--output-dir", type=Path, default=Path("D:/petsgaussianhair/_downloads/strand_gaussian_correspondence"))
    parser.add_argument("--segments", type=int, default=0, help="0 keeps the source segment count; otherwise arc-length resamples to this count.")
    parser.add_argument("--length-overlap", type=float, default=1.18)
    args = parser.parse_args()

    if args.input is None:
        strands, widths, colors, opacities = _make_fixture()
        source = "built_in_curve_fixture"
    else:
        strands, widths, colors, opacities = _load_npz(args.input)
        source = str(args.input)

    if args.segments > 0:
        counts = torch.full((strands.shape[0],), int(args.segments), dtype=torch.long)
    else:
        counts = torch.full((strands.shape[0],), strands.shape[1] - 1, dtype=torch.long)
    resampled = resample_strands_to_segment_budgets(strands, widths, colors, opacities, counts)
    gaussians = strands_to_gaussians(
        resampled.strands,
        resampled.widths,
        resampled.colors,
        resampled.opacities,
        resampled.segment_mask,
        length_overlap=args.length_overlap,
    )

    starts = resampled.strands[:, :-1]
    ends = resampled.strands[:, 1:]
    valid = resampled.segment_mask
    expected_means = (0.5 * (starts + ends))[valid]
    chords = (ends - starts)[valid]
    lengths = torch.linalg.norm(chords, dim=-1, keepdim=True).clamp_min(1e-8)
    expected_dirs = chords / lengths

    mean_error = torch.linalg.norm(gaussians.means - expected_means, dim=-1)
    axis = _rotate_x_axis(gaussians.quats)
    axis_alignment = (axis * expected_dirs).sum(dim=-1).abs()
    scale_ratio = gaussians.scales[:, 0:1] / (0.5 * lengths)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = args.output_dir / "strand_gaussian_correspondence.png"
    _save_preview(preview_path, resampled.strands, gaussians.means)

    report = {
        "source": source,
        "strand_count": int(strands.shape[0]),
        "source_samples": int(strands.shape[1]),
        "gaussian_count": int(gaussians.means.shape[0]),
        "segment_count_min": int(resampled.segment_counts.min().item()),
        "segment_count_max": int(resampled.segment_counts.max().item()),
        "mean_error_max": float(mean_error.max().item()),
        "mean_error_mean": float(mean_error.mean().item()),
        "axis_alignment_min_abs_cos": float(axis_alignment.min().item()),
        "axis_alignment_mean_abs_cos": float(axis_alignment.mean().item()),
        "scale_ratio_mean": float(scale_ratio.mean().item()),
        "scale_ratio_expected": float(args.length_overlap),
        "preview": str(preview_path),
    }
    report_path = args.output_dir / "strand_gaussian_correspondence.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

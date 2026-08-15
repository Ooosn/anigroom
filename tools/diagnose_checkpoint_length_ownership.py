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

from anigroom.grooming import decode_positive_asinh_ratio  # noqa: E402
from anigroom.surface_interpolation import local_surface_weights  # noqa: E402
from tools.train_white_tiger_stage1 import (  # noqa: E402
    load_stage1_checkpoint_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose effective strand length into the primary-guide field and "
            "the active zero-centered secondary/render residual."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mesh", default=None)
    parser.add_argument("--top-k", type=int, default=128)
    return parser.parse_args()


def residual_log_ratio(
    residual_raw: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Return the physical log length ratio represented by a residual."""

    return torch.asinh(residual_raw) * float(scale)


def compose_effective_length(
    primary_length: torch.Tensor,
    residual_raw: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose the exact unbounded positive length used by Stage1."""

    log_ratio = residual_log_ratio(residual_raw, scale)
    multiplier = torch.exp(log_ratio)
    return primary_length * multiplier, log_ratio, multiplier


def summarize(value: torch.Tensor) -> dict[str, float | int]:
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99, 0.999], device=flat.device),
    )
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "min": float(flat.min().cpu()),
        "p01": float(quantiles[0].cpu()),
        "p05": float(quantiles[1].cpu()),
        "p50": float(quantiles[2].cpu()),
        "p95": float(quantiles[3].cpu()),
        "p99": float(quantiles[4].cpu()),
        "p999": float(quantiles[5].cpu()),
        "max": float(flat.max().cpu()),
    }


def tensor_to_numpy(value: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def assert_exact_composition(
    expected: torch.Tensor,
    actual: torch.Tensor,
    label: str,
) -> dict[str, float]:
    absolute = (expected - actual).abs()
    relative = absolute / actual.abs().clamp_min(torch.finfo(actual.dtype).eps)
    maximum_absolute = float(absolute.max().cpu())
    maximum_relative = float(relative.max().cpu())
    if not torch.allclose(expected, actual, rtol=2.0e-5, atol=2.0e-7):
        raise RuntimeError(
            f"{label} decomposition does not reproduce effective length: "
            f"max_abs={maximum_absolute:.9g}, max_rel={maximum_relative:.9g}"
        )
    return {
        "max_absolute_error": maximum_absolute,
        "max_relative_error": maximum_relative,
    }


@torch.no_grad()
def diagnose(
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
    mesh_path: Path | None,
    top_k: int,
) -> dict[str, object]:
    model, config, checkpoint = load_stage1_checkpoint_model(
        checkpoint_path,
        device,
        mesh_path_override=mesh_path,
    )
    if not model.guide_enabled():
        raise RuntimeError("length ownership diagnosis requires primary guide roots")

    roots_world, root_normals, roots_local = model.roots_and_normals()
    root_tangents, root_bitangents = model.tangent_frames(root_normals)
    primary_support = model.guide_interpolation_support()
    primary_weights = model.guide_surface_interpolator().weights(
        roots_local,
        model.face_ids,
        primary_support,
    )
    primary_values, _ = model.interpolate_guide_controls(
        roots_local,
        root_normals,
        root_tangents,
        root_bitangents,
    )
    primary_length = primary_values["length"]

    residual_sample = model.geometry_residual_at_render_roots(
        roots_local,
        root_normals,
        root_tangents,
        root_bitangents,
    )
    if residual_sample is None:
        residual_raw = torch.zeros_like(primary_length)
        residual_support_ids = torch.empty(
            (roots_local.shape[0], 0), device=device, dtype=torch.long
        )
        residual_support_weights = torch.empty(
            (roots_local.shape[0], 0), device=device, dtype=roots_local.dtype
        )
    else:
        residual_raw = residual_sample.raw["length_raw"]
        if model.geometry_residual_domain == "secondary_guide":
            residual_support = model.secondary_render_support()
            residual_support_ids = residual_support.indices
            residual_support_weights = local_surface_weights(
                roots_local,
                model.secondary_guide_points_local,
                residual_support,
            )
        else:
            residual_support_ids = torch.arange(
                roots_local.shape[0], device=device, dtype=torch.long
            ).reshape(-1, 1)
            residual_support_weights = torch.ones(
                (roots_local.shape[0], 1), device=device, dtype=roots_local.dtype
            )

    residual_scale = (
        float(model.guide_length_residual_scale)
        * float(model.guide_residual_multiplier)
    )
    composed_length, log_ratio, multiplier = compose_effective_length(
        primary_length,
        residual_raw,
        residual_scale,
    )
    effective_groom = model.apply_guide_controls(
        model.groom.decode(),
        roots_local,
        root_normals,
        root_tangents,
        root_bitangents,
    )
    render_composition_error = assert_exact_composition(
        composed_length,
        effective_groom.length,
        "render-root length",
    )

    guide_length = decode_positive_asinh_ratio(
        model.guide_length_raw,
        model.guide_length_reference,
    )
    scale = torch.exp(model.log_scale).reshape(1, 1)
    guide_world = model.guide_points_local * scale + model.translation.reshape(1, 3)

    arrays: dict[str, np.ndarray] = {
        "root_points_local": tensor_to_numpy(roots_local, np.float32),
        "root_points_world": tensor_to_numpy(roots_world, np.float32),
        "root_face_ids": tensor_to_numpy(model.face_ids, np.int64),
        "primary_support_ids": tensor_to_numpy(primary_support.indices, np.int64),
        "primary_support_weights": tensor_to_numpy(primary_weights, np.float32),
        "primary_length": tensor_to_numpy(primary_length.reshape(-1), np.float32),
        "residual_length_raw": tensor_to_numpy(residual_raw.reshape(-1), np.float32),
        "residual_length_log_ratio": tensor_to_numpy(log_ratio.reshape(-1), np.float32),
        "residual_length_multiplier": tensor_to_numpy(multiplier.reshape(-1), np.float32),
        "effective_length": tensor_to_numpy(effective_groom.length.reshape(-1), np.float32),
        "residual_support_ids": tensor_to_numpy(residual_support_ids, np.int64),
        "residual_support_weights": tensor_to_numpy(
            residual_support_weights, np.float32
        ),
        "guide_points_local": tensor_to_numpy(model.guide_points_local, np.float32),
        "guide_points_world": tensor_to_numpy(guide_world, np.float32),
        "guide_face_ids": tensor_to_numpy(model.guide_face_ids, np.int64),
        "guide_length_reference": tensor_to_numpy(
            model.guide_length_reference.reshape(-1), np.float32
        ),
        "guide_length_raw": tensor_to_numpy(
            model.guide_length_raw.reshape(-1), np.float32
        ),
        "guide_length": tensor_to_numpy(guide_length.reshape(-1), np.float32),
    }

    secondary_report: dict[str, object] | None = None
    if model.secondary_guides_enabled():
        secondary_normals, secondary_tangents, secondary_bitangents = (
            model.tangent_frames_for_face_ids(model.secondary_guide_face_ids)
        )
        secondary_primary_values, _ = model.sample_guide_controls(
            model.secondary_guide_points_local,
            model.secondary_guide_face_ids,
            secondary_normals,
            secondary_tangents,
            secondary_bitangents,
            support=model.secondary_primary_support(),
        )
        secondary_raw = model.secondary_geometry_residual.length_raw
        secondary_composed, secondary_log_ratio, secondary_multiplier = (
            compose_effective_length(
                secondary_primary_values["length"],
                secondary_raw,
                residual_scale,
            )
        )
        secondary_effective = model.secondary_effective_groom().length
        secondary_composition_error = assert_exact_composition(
            secondary_composed,
            secondary_effective,
            "secondary-guide length",
        )
        secondary_world = (
            model.secondary_guide_points_local * scale
            + model.translation.reshape(1, 3)
        )
        arrays.update(
            {
                "secondary_points_local": tensor_to_numpy(
                    model.secondary_guide_points_local, np.float32
                ),
                "secondary_points_world": tensor_to_numpy(
                    secondary_world, np.float32
                ),
                "secondary_face_ids": tensor_to_numpy(
                    model.secondary_guide_face_ids, np.int64
                ),
                "secondary_parent_ids": tensor_to_numpy(
                    model.secondary_guide_parent_ids, np.int64
                ),
                "secondary_primary_length": tensor_to_numpy(
                    secondary_primary_values["length"].reshape(-1), np.float32
                ),
                "secondary_residual_length_raw": tensor_to_numpy(
                    secondary_raw.reshape(-1), np.float32
                ),
                "secondary_residual_length_log_ratio": tensor_to_numpy(
                    secondary_log_ratio.reshape(-1), np.float32
                ),
                "secondary_residual_length_multiplier": tensor_to_numpy(
                    secondary_multiplier.reshape(-1), np.float32
                ),
                "secondary_effective_length": tensor_to_numpy(
                    secondary_effective.reshape(-1), np.float32
                ),
            }
        )
        secondary_report = {
            "count": int(model.secondary_guide_points_local.shape[0]),
            "primary_length": summarize(secondary_primary_values["length"]),
            "residual_raw": summarize(secondary_raw),
            "residual_log_ratio": summarize(secondary_log_ratio),
            "residual_multiplier": summarize(secondary_multiplier),
            "effective_length": summarize(secondary_effective),
            "composition_error": secondary_composition_error,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    count = max(0, min(int(top_k), int(effective_groom.length.shape[0])))
    top_ids = torch.topk(effective_groom.length.reshape(-1), k=count).indices
    top_records = []
    for root_id in top_ids.detach().cpu().tolist():
        top_records.append(
            {
                "root_id": int(root_id),
                "point_local": [float(v) for v in roots_local[root_id].cpu()],
                "point_world": [float(v) for v in roots_world[root_id].cpu()],
                "primary_length": float(primary_length[root_id].cpu()),
                "residual_raw": float(residual_raw[root_id].cpu()),
                "residual_log_ratio": float(log_ratio[root_id].cpu()),
                "residual_multiplier": float(multiplier[root_id].cpu()),
                "effective_length": float(effective_groom.length[root_id].cpu()),
                "primary_support_ids": [
                    int(v) for v in primary_support.indices[root_id].cpu()
                ],
                "residual_support_ids": [
                    int(v) for v in residual_support_ids[root_id].cpu()
                ],
            }
        )

    report: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "iteration": int(checkpoint.get("iteration", -1)),
        "geometry_residual_domain": str(model.geometry_residual_domain),
        "render_geometry_parameterization": str(
            model.render_geometry_parameterization
        ),
        "guide_length_residual_scale": float(model.guide_length_residual_scale),
        "guide_residual_multiplier": float(model.guide_residual_multiplier),
        "effective_residual_scale": residual_scale,
        "render_root_count": int(roots_local.shape[0]),
        "guide_root_count": int(model.guide_points_local.shape[0]),
        "primary_guide_length": summarize(guide_length),
        "render_primary_length": summarize(primary_length),
        "render_residual_raw": summarize(residual_raw),
        "render_residual_log_ratio": summarize(log_ratio),
        "render_residual_multiplier": summarize(multiplier),
        "render_effective_length": summarize(effective_groom.length),
        "render_composition_error": render_composition_error,
        "secondary": secondary_report,
        "top_effective_length_roots": top_records,
        "config_child_count": int(config.child_count),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console_report = {
        key: value
        for key, value in report.items()
        if key != "top_effective_length_roots"
    }
    console_report["top_effective_length_root_count"] = len(top_records)
    print(json.dumps(console_report, indent=2))
    return report


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    mesh_path = (
        Path(args.mesh).expanduser().resolve() if args.mesh is not None else None
    )
    diagnose(
        checkpoint_path,
        output_path,
        device,
        mesh_path,
        args.top_k,
    )


if __name__ == "__main__":
    main()

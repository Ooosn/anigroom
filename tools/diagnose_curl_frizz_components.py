from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import build_strands  # noqa: E402
from tools.train_white_tiger_stage1 import (  # noqa: E402
    build_stage1_model_from_checkpoint,
    load_training_checkpoint,
    resolve_project_path,
    stage1_config_from_checkpoint_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose a fixed checkpoint into backbone, curl-only, frizz-only, "
            "primary-detail, and final-detail strand geometry."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-strands", type=int, default=100000)
    parser.add_argument("--top-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def select_groom(groom, ids: torch.Tensor):
    return replace(
        groom,
        **{field.name: getattr(groom, field.name)[ids] for field in fields(groom)},
    )


def select_detail(groom, *, curl: bool, frizz: bool):
    return replace(
        groom,
        curl_radius_ratio=(
            groom.curl_radius_ratio
            if curl
            else torch.zeros_like(groom.curl_radius_ratio)
        ),
        frizz_amplitude_ratio=(
            groom.frizz_amplitude_ratio
            if frizz
            else torch.zeros_like(groom.frizz_amplitude_ratio)
        ),
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def strand_metrics(strands: np.ndarray) -> dict[str, np.ndarray]:
    strands64 = np.asarray(strands, dtype=np.float64)
    segments = np.diff(strands64, axis=1)
    segment_length = np.linalg.norm(segments, axis=-1)
    unit = segments / np.maximum(segment_length[..., None], 1.0e-12)
    cosine = np.einsum("nsd,nsd->ns", unit[:, :-1], unit[:, 1:])
    turns = np.arccos(np.clip(cosine, -1.0, 1.0))

    chord = strands64[:, -1] - strands64[:, 0]
    chord_length = np.linalg.norm(chord, axis=-1)
    chord_unit = chord / np.maximum(chord_length[:, None], 1.0e-12)
    chord_projection = np.einsum("nsd,nd->ns", segments, chord_unit)
    arc_length = segment_length.sum(axis=1)
    return {
        "cumulative_turn_degrees": np.degrees(turns.sum(axis=1)),
        "maximum_local_turn_degrees": np.degrees(turns.max(axis=1)),
        "arc_chord_ratio": arc_length / np.maximum(chord_length, 1.0e-12),
        "backward_segment_count": (chord_projection < -1.0e-10).sum(axis=1),
    }


def summarize_metrics(metrics: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        name: quantiles(values)
        for name, values in metrics.items()
        if name != "backward_segment_count"
    } | {
        "strands_with_backward_segment": int(
            np.count_nonzero(metrics["backward_segment_count"])
        ),
        "backward_segment_fraction": float(
            np.count_nonzero(metrics["backward_segment_count"])
            / metrics["backward_segment_count"].size
        ),
    }


def displacement_metrics(
    strands: np.ndarray,
    reference: np.ndarray,
) -> dict[str, np.ndarray]:
    distance = np.linalg.norm(
        np.asarray(strands, dtype=np.float64)
        - np.asarray(reference, dtype=np.float64),
        axis=-1,
    )
    return {
        "mean_displacement": distance.mean(axis=1),
        "maximum_displacement": distance.max(axis=1),
        "tip_displacement": distance[:, -1],
    }


def top_mask(score: np.ndarray, count: int) -> np.ndarray:
    count = min(max(int(count), 0), int(score.size))
    mask = np.zeros(score.size, dtype=bool)
    if count:
        selected = np.argpartition(score, -count)[-count:]
        mask[selected] = True
    return mask


def save_variant(
    path: Path,
    *,
    strands: np.ndarray,
    widths: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
    root_ids: np.ndarray,
    masks: dict[str, np.ndarray],
    checkpoint_path: Path,
    iteration: int,
) -> None:
    np.savez_compressed(
        path,
        strands=strands,
        widths=widths,
        colors=colors,
        opacities=opacities,
        root_ids=root_ids,
        iteration=np.asarray([iteration], dtype=np.int64),
        source_checkpoint=np.asarray([str(checkpoint_path)], dtype="U512"),
        **masks,
    )


def main() -> None:
    args = parse_args()
    if args.samples < 3:
        raise ValueError("--samples must be at least 3")
    if args.max_strands <= 0:
        raise ValueError("--max-strands must be positive")
    if args.top_count <= 0:
        raise ValueError("--top-count must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested device {args.device!r}, but CUDA is unavailable")

    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_training_checkpoint(checkpoint_path)
    config = stage1_config_from_checkpoint_mapping(checkpoint["config"])
    model = build_stage1_model_from_checkpoint(checkpoint, config, device)
    model.eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    root_count = int(model.face_ids.shape[0])
    strand_count = min(root_count, int(args.max_strands))
    subset_cpu = torch.randperm(root_count, generator=generator)[:strand_count]
    subset = subset_cpu.to(device=device)

    with torch.no_grad():
        roots, normals, roots_local = model.roots_and_normals()
        tangents, bitangents = model.tangent_frames(normals)
        template = model.groom.decode()
        original_guide_multiplier = float(model.guide_residual_multiplier)
        original_secondary_multiplier = float(
            model.secondary_shape_residual_multiplier
        )

        model.guide_residual_multiplier = original_guide_multiplier
        model.secondary_shape_residual_multiplier = original_secondary_multiplier
        final_groom = select_groom(
            model.apply_guide_controls(template, roots_local), subset
        )

        model.guide_residual_multiplier = original_guide_multiplier
        model.secondary_shape_residual_multiplier = 0.0
        primary_detail_groom = select_groom(
            model.apply_guide_controls(template, roots_local), subset
        )

        model.guide_residual_multiplier = original_guide_multiplier
        model.secondary_shape_residual_multiplier = original_secondary_multiplier

        selected_roots = roots[subset]
        selected_normals = normals[subset]
        selected_tangents = tangents[subset]
        selected_bitangents = bitangents[subset]
        variants = {
            "backbone": select_detail(final_groom, curl=False, frizz=False),
            "curl_only": select_detail(final_groom, curl=True, frizz=False),
            "frizz_only": select_detail(final_groom, curl=False, frizz=True),
            "primary_curl_frizz": primary_detail_groom,
            "final_curl_frizz": final_groom,
        }

        variant_strands: dict[str, np.ndarray] = {}
        variant_metrics: dict[str, dict[str, np.ndarray]] = {}
        widths_np = colors_np = opacities_np = None
        for name, groom in variants.items():
            strands, widths, colors, opacities = build_strands(
                selected_roots,
                selected_normals,
                selected_tangents,
                selected_bitangents,
                groom,
                samples=int(args.samples),
            )
            strands_np = strands.detach().cpu().numpy().astype(np.float32)
            variant_strands[name] = strands_np
            variant_metrics[name] = strand_metrics(strands_np)
            if widths_np is None:
                widths_np = widths.detach().cpu().numpy().astype(np.float32)
                colors_np = colors.detach().cpu().numpy().astype(np.float32)
                opacities_np = opacities.detach().cpu().numpy().astype(np.float32)

        root_ids = subset_cpu.numpy().astype(np.int64)
        backbone = variant_strands["backbone"]
        curl_displacement = displacement_metrics(
            variant_strands["curl_only"], backbone
        )
        frizz_displacement = displacement_metrics(
            variant_strands["frizz_only"], backbone
        )
        final_displacement = displacement_metrics(
            variant_strands["final_curl_frizz"], backbone
        )
        curl_turn_excess = (
            variant_metrics["curl_only"]["cumulative_turn_degrees"]
            - variant_metrics["backbone"]["cumulative_turn_degrees"]
        )
        frizz_turn_excess = (
            variant_metrics["frizz_only"]["cumulative_turn_degrees"]
            - variant_metrics["backbone"]["cumulative_turn_degrees"]
        )
        final_turn_excess = (
            variant_metrics["final_curl_frizz"]["cumulative_turn_degrees"]
            - variant_metrics["backbone"]["cumulative_turn_degrees"]
        )
        masks = {
            "top_curl_turn_excess": top_mask(curl_turn_excess, args.top_count),
            "top_frizz_turn_excess": top_mask(frizz_turn_excess, args.top_count),
            "top_final_turn_excess": top_mask(final_turn_excess, args.top_count),
            "top_final_displacement": top_mask(
                final_displacement["maximum_displacement"], args.top_count
            ),
        }

        length = final_groom.length.detach().cpu().numpy().reshape(-1)
        curl_ratio = (
            final_groom.curl_radius_ratio.detach().cpu().numpy().reshape(-1)
        )
        frizz_ratio = (
            final_groom.frizz_amplitude_ratio.detach().cpu().numpy().reshape(-1)
        )
        curl_turns = final_groom.curl_turns.detach().cpu().numpy().reshape(-1)
        curl_phase = final_groom.curl_phase.detach().cpu().numpy().reshape(-1)
        attributes = {
            "length": length,
            "curl_radius_ratio": curl_ratio,
            "curl_radius": length * curl_ratio,
            "curl_turns": curl_turns,
            "curl_phase": curl_phase,
            "frizz_amplitude_ratio": frizz_ratio,
            "frizz_amplitude": length * frizz_ratio,
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, strands in variant_strands.items():
        save_variant(
            output_dir / f"{name}_{strand_count}_s{args.samples}.npz",
            strands=strands,
            widths=widths_np,
            colors=colors_np,
            opacities=opacities_np,
            root_ids=root_ids,
            masks=masks,
            checkpoint_path=checkpoint_path,
            iteration=int(checkpoint.get("iteration", -1)),
        )

    top_indices = np.argsort(final_turn_excess)[::-1][: int(args.top_count)]
    top_records = []
    for rank, index in enumerate(top_indices.tolist()):
        top_records.append(
            {
                "rank": rank,
                "subset_index": int(index),
                "root_id": int(root_ids[index]),
                "curl_turn_excess_degrees": float(curl_turn_excess[index]),
                "frizz_turn_excess_degrees": float(frizz_turn_excess[index]),
                "final_turn_excess_degrees": float(final_turn_excess[index]),
                "final_maximum_displacement": float(
                    final_displacement["maximum_displacement"][index]
                ),
                "attributes": {
                    name: float(values[index])
                    for name, values in attributes.items()
                },
            }
        )

    report = {
        "checkpoint": str(checkpoint_path),
        "iteration": int(checkpoint.get("iteration", -1)),
        "root_count": root_count,
        "strand_count": strand_count,
        "samples": int(args.samples),
        "seed": int(args.seed),
        "guide_residual_multiplier": original_guide_multiplier,
        "secondary_shape_residual_multiplier": original_secondary_multiplier,
        "variants": {
            name: summarize_metrics(metrics)
            for name, metrics in variant_metrics.items()
        },
        "component_displacement": {
            "curl_only": {
                name: quantiles(values)
                for name, values in curl_displacement.items()
            },
            "frizz_only": {
                name: quantiles(values)
                for name, values in frizz_displacement.items()
            },
            "final_curl_frizz": {
                name: quantiles(values)
                for name, values in final_displacement.items()
            },
        },
        "turn_excess_degrees": {
            "curl_only": quantiles(curl_turn_excess),
            "frizz_only": quantiles(frizz_turn_excess),
            "final_curl_frizz": quantiles(final_turn_excess),
        },
        "attributes": {
            name: quantiles(values) for name, values in attributes.items()
        },
        "curl_turns_unique": np.unique(curl_turns).astype(float).tolist(),
        "curl_phase_unique": np.unique(curl_phase).astype(float).tolist(),
        "top_final_turn_excess": top_records,
        "cuda_max_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
            if device.type == "cuda"
            else 0.0
        ),
    }
    report_path = output_dir / "curl_frizz_component_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

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

from anigroom.grooming import (  # noqa: E402
    GroomParameterField,
    build_strands,
    decode_positive_asinh_ratio,
)
from tools.train_white_tiger_stage1 import (  # noqa: E402
    build_stage1_model_from_checkpoint,
    load_training_checkpoint,
    resolve_project_path,
    stage1_config_from_checkpoint_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute strict strand foldback to the brush backbone, primary "
            "shape field, secondary residual, curl, or frizz."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-strands", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--reference-strands", default="")
    return parser.parse_args()


def select_decoded_groom(groom, ids: torch.Tensor):
    return replace(
        groom,
        **{
            field.name: getattr(groom, field.name)[ids]
            for field in fields(groom)
        },
    )


def replace_detail(groom, *, curl_radius=None, frizz=None):
    updates = {}
    if curl_radius is not None:
        updates["curl_radius"] = curl_radius
    if frizz is not None:
        updates["frizz"] = frizz
    return replace(groom, **updates)


def backward_report(strands: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    strands64 = strands.astype(np.float64, copy=False)
    segments = np.diff(strands64, axis=1)
    segment_length = np.linalg.norm(segments, axis=-1)
    arc_length = segment_length.sum(axis=1)
    chord = strands64[:, -1] - strands64[:, 0]
    chord_length = np.linalg.norm(chord, axis=-1)
    chord_unit = chord / np.maximum(chord_length[:, None], 1.0e-12)
    projection = np.einsum("nsd,nd->ns", segments, chord_unit)
    backward = projection < -1.0e-10
    mask = backward.any(axis=1)

    first = segments[:, :-1]
    second = segments[:, 1:]
    cosine = np.einsum("nsd,nsd->ns", first, second) / np.maximum(
        np.linalg.norm(first, axis=-1)
        * np.linalg.norm(second, axis=-1),
        1.0e-12,
    )
    maximum_turn = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    ).max(axis=1)
    arc_chord = arc_length / np.maximum(chord_length, 1.0e-12)

    bad_ids = np.flatnonzero(mask)
    bad_min_segment = (
        np.argmin(projection[bad_ids], axis=1).astype(np.int64)
        if bad_ids.size
        else np.empty((0,), dtype=np.int64)
    )
    report = {
        "strands_with_backward_segment": int(mask.sum()),
        "backward_segment_fraction": float(backward.mean()),
        "backward_subset_indices": bad_ids.tolist(),
        "backward_min_projection_segment": bad_min_segment.tolist(),
        "arc_chord_ratio_p95": float(np.quantile(arc_chord, 0.95)),
        "arc_chord_ratio_p99": float(np.quantile(arc_chord, 0.99)),
        "arc_chord_ratio_max": float(arc_chord.max()),
        "maximum_local_turn_p95_degrees": float(
            np.quantile(maximum_turn, 0.95)
        ),
        "maximum_local_turn_p99_degrees": float(
            np.quantile(maximum_turn, 0.99)
        ),
        "maximum_local_turn_max_degrees": float(maximum_turn.max()),
    }
    return report, mask


def summarize_group(values: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    good = values[~mask]
    bad = values[mask]

    def stats(group: np.ndarray) -> dict[str, float | int]:
        if group.size == 0:
            return {"count": 0}
        return {
            "count": int(group.size),
            "mean": float(group.mean()),
            "p10": float(np.quantile(group, 0.10)),
            "p50": float(np.quantile(group, 0.50)),
            "p90": float(np.quantile(group, 0.90)),
            "p95": float(np.quantile(group, 0.95)),
            "p99": float(np.quantile(group, 0.99)),
            "max": float(group.max()),
        }

    percentile_ranks = (
        np.searchsorted(np.sort(values), bad, side="right") / values.size
        if bad.size
        else np.empty((0,), dtype=np.float64)
    )
    return {
        "all": stats(values),
        "non_foldback": stats(good),
        "foldback": stats(bad),
        "foldback_percentile_rank_p50": (
            float(np.quantile(percentile_ranks, 0.50))
            if percentile_ranks.size
            else None
        ),
        "foldback_percentile_rank_p90": (
            float(np.quantile(percentile_ranks, 0.90))
            if percentile_ranks.size
            else None
        ),
    }


def percentile_ranks(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    selected = np.asarray(selected, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("percentile ranks require a non-empty population")
    return np.searchsorted(np.sort(values), selected, side="right") / values.size


def top_guide_records(
    *,
    score: np.ndarray,
    count: int,
    guide_points: np.ndarray,
    guide_length: np.ndarray,
    guide_curl: np.ndarray,
    guide_frizz: np.ndarray,
    guide_ratio: np.ndarray,
    guide_confidence: np.ndarray,
) -> list[dict[str, object]]:
    selected = np.argsort(np.asarray(score).reshape(-1))[::-1][:count]
    return [
        {
            "guide_id": int(guide_id),
            "point": [float(value) for value in guide_points[guide_id]],
            "length": float(guide_length[guide_id]),
            "curl_radius": float(guide_curl[guide_id]),
            "curl_radius_over_length": float(guide_ratio[guide_id]),
            "frizz": float(guide_frizz[guide_id]),
            "clean_flow_confidence": float(guide_confidence[guide_id]),
        }
        for guide_id in selected.tolist()
    ]


def main() -> None:
    args = parse_args()
    if args.samples < 3:
        raise ValueError("--samples must be at least 3")
    if args.max_strands <= 0:
        raise ValueError("--max-strands must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"requested diagnostic device {args.device!r}, but CUDA is unavailable"
        )
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_training_checkpoint(checkpoint_path)
    config = stage1_config_from_checkpoint_mapping(checkpoint["config"])
    model = build_stage1_model_from_checkpoint(checkpoint, config, device)
    model.eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    count = int(model.face_ids.shape[0])
    subset_count = min(count, int(args.max_strands))
    subset_cpu = torch.randperm(count, generator=generator)[:subset_count]
    subset = subset_cpu.to(device=device)

    with torch.no_grad():
        roots, normals, roots_local = model.roots_and_normals()
        tangents, bitangents = model.tangent_frames(normals)
        template = model.groom.decode()

        original_guide_residual_multiplier = float(
            model.guide_residual_multiplier
        )
        original_secondary_multiplier = float(
            model.secondary_shape_residual_multiplier
        )
        model.guide_residual_multiplier = original_guide_residual_multiplier
        model.secondary_shape_residual_multiplier = original_secondary_multiplier
        full_all = model.apply_guide_controls(template, roots_local)
        full = select_decoded_groom(full_all, subset)
        del full_all

        model.guide_residual_multiplier = original_guide_residual_multiplier
        model.secondary_shape_residual_multiplier = 0.0
        no_secondary_shape_all = model.apply_guide_controls(template, roots_local)
        no_secondary_shape = select_decoded_groom(
            no_secondary_shape_all, subset
        )
        del no_secondary_shape_all

        model.guide_residual_multiplier = 0.0
        model.secondary_shape_residual_multiplier = 0.0
        primary_all = model.apply_guide_controls(template, roots_local)
        primary = select_decoded_groom(primary_all, subset)
        del primary_all, template
        model.guide_residual_multiplier = original_guide_residual_multiplier
        model.secondary_shape_residual_multiplier = original_secondary_multiplier

        selected_roots = roots[subset]
        selected_normals = normals[subset]
        selected_tangents = tangents[subset]
        selected_bitangents = bitangents[subset]

        zeros_curl = torch.zeros_like(full.curl_radius)
        zeros_frizz = torch.zeros_like(full.frizz)
        variants = {
            "full_backbone": replace_detail(
                full, curl_radius=zeros_curl, frizz=zeros_frizz
            ),
            "full_all": full,
            "full_curl_only": replace_detail(full, frizz=zeros_frizz),
            "full_frizz_only": replace_detail(full, curl_radius=zeros_curl),
            "no_secondary_shape_backbone": replace_detail(
                no_secondary_shape,
                curl_radius=torch.zeros_like(no_secondary_shape.curl_radius),
                frizz=torch.zeros_like(no_secondary_shape.frizz),
            ),
            "no_secondary_shape_all": no_secondary_shape,
            "no_secondary_shape_curl_only": replace_detail(
                no_secondary_shape,
                frizz=torch.zeros_like(no_secondary_shape.frizz),
            ),
            "no_secondary_shape_frizz_only": replace_detail(
                no_secondary_shape,
                curl_radius=torch.zeros_like(no_secondary_shape.curl_radius),
            ),
            "primary_backbone": replace_detail(
                primary,
                curl_radius=torch.zeros_like(primary.curl_radius),
                frizz=torch.zeros_like(primary.frizz),
            ),
            "primary_all": primary,
            "primary_curl_only": replace_detail(
                primary, frizz=torch.zeros_like(primary.frizz)
            ),
            "primary_frizz_only": replace_detail(
                primary, curl_radius=torch.zeros_like(primary.curl_radius)
            ),
        }

        variant_strands: dict[str, np.ndarray] = {}
        reports: dict[str, dict[str, object]] = {}
        masks: dict[str, np.ndarray] = {}
        widths = None
        colors = None
        opacities = None
        for name, groom in variants.items():
            strands, variant_widths, variant_colors, variant_opacities = (
                build_strands(
                    selected_roots,
                    selected_normals,
                    selected_tangents,
                    selected_bitangents,
                    groom,
                    samples=int(args.samples),
                )
            )
            strands_np = strands.detach().cpu().numpy().astype(np.float32)
            report, mask = backward_report(strands_np)
            variant_strands[name] = strands_np
            reports[name] = report
            masks[name] = mask
            if widths is None:
                widths = variant_widths.detach().cpu().numpy().astype(np.float32)
                colors = variant_colors.detach().cpu().numpy().astype(np.float32)
                opacities = variant_opacities.detach().cpu().numpy().astype(np.float32)

        root_ids = subset_cpu.numpy().astype(np.int64)
        full_mask = masks["full_all"]
        full_bad = np.flatnonzero(full_mask)
        for name, mask in masks.items():
            reports[name]["original_full_foldbacks_remaining"] = int(
                mask[full_bad].sum()
            )
            reports[name]["new_foldbacks_outside_full_set"] = int(
                np.logical_and(mask, ~full_mask).sum()
            )
            reports[name]["backward_root_ids"] = root_ids[mask].tolist()

        direction_normal_cosine = full.direction_local[:, 2]
        direction_normal_cosine = (
            direction_normal_cosine.detach().cpu().numpy()
        )
        full_length = full.length.detach().cpu().numpy().reshape(-1)
        primary_length = primary.length.detach().cpu().numpy().reshape(-1)
        full_curl = full.curl_radius.detach().cpu().numpy().reshape(-1)
        primary_curl = primary.curl_radius.detach().cpu().numpy().reshape(-1)
        no_secondary_shape_curl = (
            no_secondary_shape.curl_radius.detach().cpu().numpy().reshape(-1)
        )
        full_frizz = full.frizz.detach().cpu().numpy().reshape(-1)
        primary_frizz = primary.frizz.detach().cpu().numpy().reshape(-1)
        no_secondary_shape_frizz = (
            no_secondary_shape.frizz.detach().cpu().numpy().reshape(-1)
        )
        curl_turns = full.curl_turns.detach().cpu().numpy().reshape(-1)
        curl_phase = full.curl_phase.detach().cpu().numpy().reshape(-1)
        frizz_seed_phase = (
            full.frizz_seed_phase.detach().cpu().numpy().reshape(-1)
        )
        stiffness = full.brush_stiffness.detach().cpu().numpy().reshape(-1)
        clean_confidence = model.clean_flow_anchor_confidence[subset]
        clean_confidence = clean_confidence.detach().cpu().numpy().reshape(-1)

        guide_support, guide_weights = model.guide_interpolation_attribution(
            roots_local
        )
        selected_guide_ids = guide_support.indices[subset]
        selected_guide_weights = guide_weights[subset]
        ranges = model.groom.ranges
        guide_length = decode_positive_asinh_ratio(
            model.guide_length_raw,
            model.guide_length_reference,
        ).reshape(-1)
        guide_curl = GroomParameterField._decode_range(
            model.guide_curl_radius_raw,
            ranges.curl_radius,
        ).reshape(-1)
        guide_frizz = GroomParameterField._decode_range(
            model.guide_frizz_raw,
            ranges.frizz,
        ).reshape(-1)
        guide_ratio = guide_curl / guide_length.clamp_min(1.0e-12)
        guide_confidence = model.guide_clean_flow_anchor_confidence.reshape(-1)
        guide_points = model.guide_points_local

        guide_length_np = guide_length.detach().cpu().numpy()
        guide_curl_np = guide_curl.detach().cpu().numpy()
        guide_frizz_np = guide_frizz.detach().cpu().numpy()
        guide_ratio_np = guide_ratio.detach().cpu().numpy()
        guide_confidence_np = guide_confidence.detach().cpu().numpy()
        guide_points_np = guide_points.detach().cpu().numpy()
        selected_guide_ids_np = selected_guide_ids.detach().cpu().numpy()
        selected_guide_weights_np = selected_guide_weights.detach().cpu().numpy()

        attributes = {
            "full_length": full_length,
            "primary_length": primary_length,
            "full_curl_radius": full_curl,
            "primary_curl_radius": primary_curl,
            "no_secondary_shape_curl_radius": no_secondary_shape_curl,
            "full_frizz": full_frizz,
            "primary_frizz": primary_frizz,
            "no_secondary_shape_frizz": no_secondary_shape_frizz,
            "curl_turns": curl_turns,
            "curl_phase_mod_2pi": np.mod(curl_phase, 2.0 * np.pi),
            "frizz_seed_phase_mod_2pi": np.mod(
                frizz_seed_phase, 2.0 * np.pi
            ),
            "brush_stiffness": stiffness,
            "direction_normal_cosine": direction_normal_cosine,
            "clean_flow_anchor_confidence": clean_confidence,
            "full_curl_radius_over_length": full_curl
            / np.maximum(full_length, 1.0e-12),
            "full_curl_wavenumber": (
                2.0 * np.pi * full_curl * curl_turns
                / np.maximum(full_length, 1.0e-12)
            ),
            "primary_curl_radius_over_length": primary_curl
            / np.maximum(primary_length, 1.0e-12),
            "primary_curl_wavenumber": (
                2.0 * np.pi * primary_curl * curl_turns
                / np.maximum(primary_length, 1.0e-12)
            ),
            "full_frizz_over_length": full_frizz
            / np.maximum(full_length, 1.0e-12),
            "primary_frizz_over_length": primary_frizz
            / np.maximum(primary_length, 1.0e-12),
            "secondary_curl_ratio": full_curl
            / np.maximum(primary_curl, 1.0e-12),
            "secondary_frizz_ratio": full_frizz
            / np.maximum(primary_frizz, 1.0e-12),
        }
        attribute_report = {
            name: summarize_group(values, full_mask)
            for name, values in attributes.items()
        }


        bad_support_ids = selected_guide_ids_np[full_mask]
        bad_support_weights = selected_guide_weights_np[full_mask]
        guide_weight_sum = np.zeros(guide_length_np.shape[0], dtype=np.float64)
        guide_occurrences = np.zeros(guide_length_np.shape[0], dtype=np.int64)
        if bad_support_ids.size:
            np.add.at(
                guide_weight_sum,
                bad_support_ids.reshape(-1),
                bad_support_weights.reshape(-1),
            )
            np.add.at(
                guide_occurrences,
                bad_support_ids.reshape(-1),
                np.ones(bad_support_ids.size, dtype=np.int64),
            )
        contributing_guides = np.flatnonzero(guide_weight_sum > 0.0)
        contributing_guides = contributing_guides[
            np.argsort(guide_weight_sum[contributing_guides])[::-1]
        ]
        guide_attribution = []
        for guide_id in contributing_guides.tolist():
            guide_attribution.append(
                {
                    "guide_id": int(guide_id),
                    "occurrences": int(guide_occurrences[guide_id]),
                    "total_bad_weight": float(guide_weight_sum[guide_id]),
                    "normalized_bad_weight": float(
                        guide_weight_sum[guide_id]
                        / max(float(guide_weight_sum.sum()), 1.0e-12)
                    ),
                    "point": [float(value) for value in guide_points_np[guide_id]],
                    "length": float(guide_length_np[guide_id]),
                    "curl_radius": float(guide_curl_np[guide_id]),
                    "curl_radius_over_length": float(guide_ratio_np[guide_id]),
                    "frizz": float(guide_frizz_np[guide_id]),
                    "clean_flow_confidence": float(guide_confidence_np[guide_id]),
                    "length_percentile_rank": float(
                        percentile_ranks(
                            guide_length_np,
                            guide_length_np[[guide_id]],
                        )[0]
                    ),
                    "curl_percentile_rank": float(
                        percentile_ranks(
                            guide_curl_np,
                            guide_curl_np[[guide_id]],
                        )[0]
                    ),
                    "curl_over_length_percentile_rank": float(
                        percentile_ranks(
                            guide_ratio_np,
                            guide_ratio_np[[guide_id]],
                        )[0]
                    ),
                }
            )

        focus_mask = np.logical_or.reduce(list(masks.values()))
        focus = np.flatnonzero(focus_mask)
        focus_tensor = torch.from_numpy(focus).to(device=device)
        variant_names = list(variants)
        focus_strands = np.stack(
            [variant_strands[name][focus] for name in variant_names], axis=0
        )
        focus_roots = (
            selected_roots[focus_tensor]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        focus_normals = (
            selected_normals[focus_tensor]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    reference_report: dict[str, object] | None = None
    if args.reference_strands:
        reference_path = Path(args.reference_strands)
        with np.load(reference_path, allow_pickle=False) as archive:
            reference_root_ids = archive["root_ids"].astype(np.int64)
            reference = archive["strands"].astype(np.float32)
        if not np.array_equal(reference_root_ids, root_ids):
            raise RuntimeError("reference root_ids do not match deterministic subset")
        absolute = np.abs(reference - variant_strands["full_all"])
        reference_report = {
            "path": str(reference_path.resolve()),
            "root_ids_exact": True,
            "max_absolute_difference": float(absolute.max()),
            "mean_absolute_difference": float(absolute.mean()),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "foldback_component_focus.npz",
        variant_names=np.asarray(variant_names, dtype="U32"),
        subset_indices=focus.astype(np.int64),
        root_ids=root_ids[focus],
        roots=focus_roots,
        normals=focus_normals,
        strands=focus_strands,
        widths=widths[focus],
        colors=colors[focus],
        opacities=opacities[focus],
        **{name: values[focus] for name, values in attributes.items()},
    )

    focus_position = {
        int(subset_index): position
        for position, subset_index in enumerate(focus)
    }
    per_foldback = []
    for index in full_bad:
        per_foldback.append(
            {
                "subset_index": int(index),
                "root_id": int(root_ids[index]),
                "root": [
                    float(value)
                    for value in focus_roots[focus_position[int(index)]]
                ],
                "remaining_by_variant": {
                    name: bool(mask[index]) for name, mask in masks.items()
                },
                "primary_guide_support": [
                    {
                        "guide_id": int(guide_id),
                        "weight": float(weight),
                        "length": float(guide_length_np[guide_id]),
                        "curl_radius": float(guide_curl_np[guide_id]),
                        "curl_radius_over_length": float(guide_ratio_np[guide_id]),
                        "frizz": float(guide_frizz_np[guide_id]),
                        "clean_flow_confidence": float(
                            guide_confidence_np[guide_id]
                        ),
                    }
                    for guide_id, weight in zip(
                        selected_guide_ids_np[index],
                        selected_guide_weights_np[index],
                    )
                ],
                "attributes": {
                    name: float(values[index])
                    for name, values in attributes.items()
                },
            }
        )

    result = {
        "checkpoint": str(checkpoint_path),
        "iteration": int(checkpoint.get("iteration", -1)),
        "root_count": count,
        "strand_count": subset_count,
        "samples": int(args.samples),
        "seed": int(args.seed),
        "original_secondary_shape_residual_multiplier": (
            original_secondary_multiplier
        ),
        "shape_detail_multiplier": float(model.shape_detail_multiplier),
        "gaussian_rgb_residual_multiplier": float(
            model.gaussian_rgb_residual_multiplier
        ),
        "original_guide_residual_multiplier": (
            original_guide_residual_multiplier
        ),
        "variants": reports,
        "full_foldback_root_ids": root_ids[full_mask].tolist(),
        "attribute_comparison": attribute_report,
        "primary_guide_attribution": {
            "guide_count": int(guide_length_np.shape[0]),
            "support_k": int(selected_guide_ids_np.shape[1]),
            "contributing_guide_count": int(contributing_guides.size),
            "contributing_guides": guide_attribution,
            "top_by_curl_radius": top_guide_records(
                score=guide_curl_np,
                count=16,
                guide_points=guide_points_np,
                guide_length=guide_length_np,
                guide_curl=guide_curl_np,
                guide_frizz=guide_frizz_np,
                guide_ratio=guide_ratio_np,
                guide_confidence=guide_confidence_np,
            ),
            "top_by_curl_radius_over_length": top_guide_records(
                score=guide_ratio_np,
                count=16,
                guide_points=guide_points_np,
                guide_length=guide_length_np,
                guide_curl=guide_curl_np,
                guide_frizz=guide_frizz_np,
                guide_ratio=guide_ratio_np,
                guide_confidence=guide_confidence_np,
            ),
        },
        "per_full_foldback": per_foldback,
        "reference_reproduction": reference_report,
        "focus_npz": str(
            (output_dir / "foldback_component_focus.npz").resolve()
        ),
        "cuda_max_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
            if device.type == "cuda"
            else 0.0
        ),
    }
    report_path = output_dir / "foldback_component_report.json"
    report_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

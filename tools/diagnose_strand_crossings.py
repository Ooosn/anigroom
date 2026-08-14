from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REPORT_ANGLES_DEGREES = (15.0, 30.0, 45.0, 60.0)
VISUALIZATION_ARRAY_KEYS = (
    "strands",
    "widths",
    "colors",
    "opacities",
    "root_ids",
    "iteration",
)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {}
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def closest_segment_parameters(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact closest-point parameters and distances for segment pairs."""

    first_start = np.asarray(first_start, dtype=np.float64)
    first_end = np.asarray(first_end, dtype=np.float64)
    second_start = np.asarray(second_start, dtype=np.float64)
    second_end = np.asarray(second_end, dtype=np.float64)
    if not (
        first_start.shape
        == first_end.shape
        == second_start.shape
        == second_end.shape
    ):
        raise ValueError("all segment endpoint arrays must have the same shape")
    if first_start.ndim != 2 or first_start.shape[1] != 3:
        raise ValueError("segment endpoint arrays must have shape [N, 3]")

    first_delta = first_end - first_start
    second_delta = second_end - second_start
    offset = first_start - second_start
    first_sq = np.einsum("nd,nd->n", first_delta, first_delta)
    cross = np.einsum("nd,nd->n", first_delta, second_delta)
    second_sq = np.einsum("nd,nd->n", second_delta, second_delta)
    first_offset = np.einsum("nd,nd->n", first_delta, offset)
    second_offset = np.einsum("nd,nd->n", second_delta, offset)
    denominator = first_sq * second_sq - cross * cross
    epsilon = np.finfo(np.float64).eps * 64.0

    first_numerator = cross * second_offset - second_sq * first_offset
    second_numerator = first_sq * second_offset - cross * first_offset
    first_denominator = denominator.copy()
    second_denominator = denominator.copy()

    parallel = denominator <= epsilon * np.maximum(first_sq * second_sq, 1.0)
    first_numerator[parallel] = 0.0
    first_denominator[parallel] = 1.0
    second_numerator[parallel] = second_offset[parallel]
    second_denominator[parallel] = second_sq[parallel]

    below_first = first_numerator < 0.0
    first_numerator[below_first] = 0.0
    second_numerator[below_first] = second_offset[below_first]
    second_denominator[below_first] = second_sq[below_first]

    above_first = first_numerator > first_denominator
    first_numerator[above_first] = first_denominator[above_first]
    second_numerator[above_first] = (
        second_offset[above_first] + cross[above_first]
    )
    second_denominator[above_first] = second_sq[above_first]

    below_second = second_numerator < 0.0
    second_numerator[below_second] = 0.0
    first_before_start = below_second & (-first_offset < 0.0)
    first_after_end = below_second & (-first_offset > first_sq)
    first_inside = below_second & ~(first_before_start | first_after_end)
    first_numerator[first_before_start] = 0.0
    first_numerator[first_after_end] = first_denominator[first_after_end]
    first_numerator[first_inside] = -first_offset[first_inside]
    first_denominator[first_inside] = first_sq[first_inside]

    above_second = second_numerator > second_denominator
    second_numerator[above_second] = second_denominator[above_second]
    projected_first = -first_offset + cross
    first_before_start = above_second & (projected_first < 0.0)
    first_after_end = above_second & (projected_first > first_sq)
    first_inside = above_second & ~(first_before_start | first_after_end)
    first_numerator[first_before_start] = 0.0
    first_numerator[first_after_end] = first_denominator[first_after_end]
    first_numerator[first_inside] = projected_first[first_inside]
    first_denominator[first_inside] = first_sq[first_inside]

    first_parameter = np.divide(
        first_numerator,
        first_denominator,
        out=np.zeros_like(first_numerator),
        where=np.abs(first_denominator) > epsilon,
    )
    second_parameter = np.divide(
        second_numerator,
        second_denominator,
        out=np.zeros_like(second_numerator),
        where=np.abs(second_denominator) > epsilon,
    )
    first_parameter = np.clip(first_parameter, 0.0, 1.0)
    second_parameter = np.clip(second_parameter, 0.0, 1.0)
    separation = (
        offset
        + first_parameter[:, None] * first_delta
        - second_parameter[:, None] * second_delta
    )
    distance = np.linalg.norm(separation, axis=1)
    return first_parameter, second_parameter, distance


def diagnose_crossings(
    strands: np.ndarray,
    widths: np.ndarray,
    *,
    query_batch: int = 50000,
    exact_pair_batch: int = 250000,
    workers: int = -1,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Diagnose native one-sigma Gaussian-envelope strand intersections.

    The broad phase is exact for the learned transverse Gaussian scale: each
    segment is enclosed by a sphere whose radius is half its segment length
    plus its maximum endpoint width. Exact segment distance and interpolated
    widths then decide contact. No body region, world-space distance, or strand
    length threshold is used.
    """

    strands = np.asarray(strands, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    if strands.ndim != 3 or strands.shape[1] < 2 or strands.shape[2] != 3:
        raise ValueError("strands must have shape [N, samples>=2, 3]")
    if widths.ndim == 3 and widths.shape[2] == 1:
        widths = widths[..., 0]
    if widths.shape != strands.shape[:2]:
        raise ValueError("widths must match strands as [N, samples] or [N, samples, 1]")
    if strands.shape[0] < 2:
        raise ValueError("crossing diagnosis requires at least two strands")
    if query_batch <= 0 or exact_pair_batch <= 0:
        raise ValueError("batch sizes must be positive")
    if not np.isfinite(strands).all() or not np.isfinite(widths).all():
        raise ValueError("strands and widths must be finite")
    if np.any(widths <= 0.0):
        raise ValueError("all learned transverse widths must be positive")

    strand_count, sample_count, _ = strands.shape
    segments_per_strand = sample_count - 1
    starts = strands[:, :-1].reshape(-1, 3)
    ends = strands[:, 1:].reshape(-1, 3)
    width_start = widths[:, :-1].reshape(-1)
    width_end = widths[:, 1:].reshape(-1)
    segment_delta = ends - starts
    segment_length = np.linalg.norm(segment_delta, axis=1)
    if np.any(segment_length <= np.finfo(np.float64).eps):
        raise ValueError("zero-length strand segments are not supported")
    midpoint = 0.5 * (starts + ends)
    maximum_width = np.maximum(width_start, width_end)
    reach = 0.5 * segment_length + maximum_width
    global_reach = float(reach.max())
    tree = cKDTree(midpoint)

    candidate_pair_count = 0
    tested_pair_count = 0
    contact_keys: list[np.ndarray] = []
    contact_scores: list[np.ndarray] = []
    contact_angles: list[np.ndarray] = []
    contact_overlap: list[np.ndarray] = []
    contact_distance_ratio: list[np.ndarray] = []
    contact_first_segment: list[np.ndarray] = []
    contact_second_segment: list[np.ndarray] = []
    contact_first_progress: list[np.ndarray] = []
    contact_second_progress: list[np.ndarray] = []

    segment_count = starts.shape[0]
    for query_start in range(0, segment_count, int(query_batch)):
        query_end = min(query_start + int(query_batch), segment_count)
        neighbor_lists = tree.query_ball_point(
            midpoint[query_start:query_end],
            reach[query_start:query_end] + global_reach,
            workers=int(workers),
        )
        counts = np.fromiter(
            (len(indices) for indices in neighbor_lists),
            dtype=np.int64,
            count=query_end - query_start,
        )
        if counts.sum() == 0:
            continue
        first_indices = np.repeat(
            np.arange(query_start, query_end, dtype=np.int64), counts
        )
        second_indices = np.concatenate(neighbor_lists).astype(np.int64, copy=False)
        unique_order = second_indices > first_indices
        first_indices = first_indices[unique_order]
        second_indices = second_indices[unique_order]
        first_strands = first_indices // segments_per_strand
        second_strands = second_indices // segments_per_strand
        different_strands = first_strands != second_strands
        first_indices = first_indices[different_strands]
        second_indices = second_indices[different_strands]
        candidate_pair_count += int(first_indices.size)

        for pair_start in range(0, first_indices.size, int(exact_pair_batch)):
            pair_end = min(pair_start + int(exact_pair_batch), first_indices.size)
            first = first_indices[pair_start:pair_end]
            second = second_indices[pair_start:pair_end]
            tested_pair_count += int(first.size)
            first_t, second_t, distance = closest_segment_parameters(
                starts[first], ends[first], starts[second], ends[second]
            )
            first_radius = width_start[first] + first_t * (
                width_end[first] - width_start[first]
            )
            second_radius = width_start[second] + second_t * (
                width_end[second] - width_start[second]
            )
            radius_sum = first_radius + second_radius
            intersects = distance < radius_sum
            if not np.any(intersects):
                continue

            first = first[intersects]
            second = second[intersects]
            first_t = first_t[intersects]
            second_t = second_t[intersects]
            distance = distance[intersects]
            radius_sum = radius_sum[intersects]
            first_direction = segment_delta[first] / segment_length[first, None]
            second_direction = segment_delta[second] / segment_length[second, None]
            axis_cosine = np.abs(
                np.einsum("nd,nd->n", first_direction, second_direction)
            )
            axis_cosine = np.clip(axis_cosine, 0.0, 1.0)
            angle = np.degrees(np.arccos(axis_cosine))
            overlap = np.clip((radius_sum - distance) / radius_sum, 0.0, 1.0)
            score = overlap * (1.0 - axis_cosine * axis_cosine)
            first_strand = first // segments_per_strand
            second_strand = second // segments_per_strand
            lower = np.minimum(first_strand, second_strand)
            upper = np.maximum(first_strand, second_strand)

            contact_keys.append((lower * strand_count + upper).astype(np.int64))
            contact_scores.append(score.astype(np.float32))
            contact_angles.append(angle.astype(np.float32))
            contact_overlap.append(overlap.astype(np.float32))
            contact_distance_ratio.append((distance / radius_sum).astype(np.float32))
            contact_first_segment.append(first.astype(np.int32))
            contact_second_segment.append(second.astype(np.int32))
            contact_first_progress.append(
                (
                    (first % segments_per_strand).astype(np.float64)
                    + first_t
                ).astype(np.float32)
                / float(segments_per_strand)
            )
            contact_second_progress.append(
                (
                    (second % segments_per_strand).astype(np.float64)
                    + second_t
                ).astype(np.float32)
                / float(segments_per_strand)
            )

    strand_score = np.zeros(strand_count, dtype=np.float32)
    strand_pair_count = np.zeros(strand_count, dtype=np.int32)
    high_angle_masks = {
        angle: np.zeros(strand_count, dtype=bool)
        for angle in REPORT_ANGLES_DEGREES
    }
    if contact_keys:
        keys = np.concatenate(contact_keys)
        scores = np.concatenate(contact_scores)
        angles = np.concatenate(contact_angles)
        overlap = np.concatenate(contact_overlap)
        distance_ratio = np.concatenate(contact_distance_ratio)
        first_segment = np.concatenate(contact_first_segment)
        second_segment = np.concatenate(contact_second_segment)
        first_progress = np.concatenate(contact_first_progress)
        second_progress = np.concatenate(contact_second_progress)
        order = np.lexsort((-scores, keys))
        sorted_keys = keys[order]
        starts_of_pair = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_keys))]
        selected = order[starts_of_pair]
        pair_keys = keys[selected]
        pair_scores = scores[selected]
        pair_angles = angles[selected]
        pair_overlap = overlap[selected]
        pair_distance_ratio = distance_ratio[selected]
        pair_first_segment = first_segment[selected]
        pair_second_segment = second_segment[selected]
        pair_first_progress = first_progress[selected]
        pair_second_progress = second_progress[selected]
        pair_first_strand = pair_keys // strand_count
        pair_second_strand = pair_keys % strand_count
        pair_endpoints = np.concatenate(
            [pair_first_strand, pair_second_strand]
        ).astype(np.int64, copy=False)
        np.maximum.at(
            strand_score,
            pair_endpoints,
            np.concatenate([pair_scores, pair_scores]),
        )
        np.add.at(strand_pair_count, pair_endpoints, 1)
        for threshold, mask in high_angle_masks.items():
            selected_pairs = pair_angles >= threshold
            mask[pair_first_strand[selected_pairs]] = True
            mask[pair_second_strand[selected_pairs]] = True
    else:
        keys = np.empty((0,), dtype=np.int64)
        pair_keys = np.empty((0,), dtype=np.int64)
        pair_scores = np.empty((0,), dtype=np.float32)
        pair_angles = np.empty((0,), dtype=np.float32)
        pair_overlap = np.empty((0,), dtype=np.float32)
        pair_distance_ratio = np.empty((0,), dtype=np.float32)
        pair_first_segment = np.empty((0,), dtype=np.int32)
        pair_second_segment = np.empty((0,), dtype=np.int32)
        pair_first_progress = np.empty((0,), dtype=np.float32)
        pair_second_progress = np.empty((0,), dtype=np.float32)
        pair_first_strand = np.empty((0,), dtype=np.int64)
        pair_second_strand = np.empty((0,), dtype=np.int64)

    segment_lengths = np.linalg.norm(np.diff(strands, axis=1), axis=-1)
    arc_length = segment_lengths.sum(axis=1)
    chord = strands[:, -1] - strands[:, 0]
    chord_length = np.linalg.norm(chord, axis=1)
    chord_direction = chord / np.maximum(chord_length[:, None], 1.0e-12)
    root_positions = strands[:, 0]
    root_distance = np.linalg.norm(
        root_positions[pair_first_strand]
        - root_positions[pair_second_strand],
        axis=1,
    )
    pair_mean_arc_length = 0.5 * (
        arc_length[pair_first_strand] + arc_length[pair_second_strand]
    )
    pair_chord_axis_cosine = np.abs(
        np.einsum(
            "nd,nd->n",
            chord_direction[pair_first_strand],
            chord_direction[pair_second_strand],
        )
    )
    pair_chord_axis_angle = np.degrees(
        np.arccos(np.clip(pair_chord_axis_cosine, 0.0, 1.0))
    ).astype(np.float32)
    nearest_root_distance, _ = cKDTree(root_positions).query(
        root_positions, k=2, workers=int(workers)
    )
    local_root_spacing = nearest_root_distance[:, 1]
    pair_local_spacing = np.sqrt(
        local_root_spacing[pair_first_strand]
        * local_root_spacing[pair_second_strand]
    )
    root_distance_over_arc = root_distance / np.maximum(
        pair_mean_arc_length, 1.0e-12
    )
    root_distance_over_local_spacing = root_distance / np.maximum(
        pair_local_spacing, 1.0e-12
    )
    minimum_contact_progress = np.minimum(
        pair_first_progress, pair_second_progress
    )
    maximum_contact_progress = np.maximum(
        pair_first_progress, pair_second_progress
    )

    contact_mask = strand_pair_count > 0
    top_count = min(100, pair_scores.size)
    top_order = np.argsort(pair_scores)[::-1][:top_count]
    top_pairs = [
        {
            "first_strand": int(pair_first_strand[index]),
            "second_strand": int(pair_second_strand[index]),
            "first_segment": int(pair_first_segment[index] % segments_per_strand),
            "second_segment": int(pair_second_segment[index] % segments_per_strand),
            "crossing_score": float(pair_scores[index]),
            "axis_angle_degrees": float(pair_angles[index]),
            "overlap_fraction": float(pair_overlap[index]),
            "distance_over_width_sum": float(pair_distance_ratio[index]),
            "first_progress": float(pair_first_progress[index]),
            "second_progress": float(pair_second_progress[index]),
            "root_distance": float(root_distance[index]),
            "root_distance_over_mean_arc_length": float(
                root_distance_over_arc[index]
            ),
            "contact_axis_angle_degrees": float(pair_angles[index]),
            "chord_axis_angle_degrees": float(
                pair_chord_axis_angle[index]
            ),
        }
        for index in top_order.tolist()
    ]
    angle_counts = {
        f">={int(threshold)}_degrees": int((pair_angles >= threshold).sum())
        for threshold in REPORT_ANGLES_DEGREES
    }
    angle_strand_counts = {
        f">={int(threshold)}_degrees": int(mask.sum())
        for threshold, mask in high_angle_masks.items()
    }
    high_angle = pair_angles >= 45.0
    field_conflict = high_angle & (pair_chord_axis_angle >= 45.0)
    local_shape_conflict = high_angle & (pair_chord_axis_angle < 15.0)
    mixed_conflict = high_angle & ~(field_conflict | local_shape_conflict)
    report = {
        "protocol": {
            "contact_envelope": "sum of learned transverse Gaussian scales (one sigma)",
            "crossing_score": "overlap_fraction * sin(axis_angle)^2",
            "same_strand_pairs_excluded": True,
            "absolute_world_distance_threshold": None,
            "body_region_rules": None,
        },
        "strand_count": int(strand_count),
        "samples_per_strand": int(sample_count),
        "segment_count": int(segment_count),
        "broadphase_candidate_segment_pairs": int(candidate_pair_count),
        "exact_tested_segment_pairs": int(tested_pair_count),
        "intersecting_segment_pairs": int(keys.size),
        "unique_intersecting_strand_pairs": int(pair_keys.size),
        "strands_with_any_contact": int(contact_mask.sum()),
        "strands_with_any_contact_fraction": float(contact_mask.mean()),
        "unique_pair_axis_angle_degrees": _quantiles(pair_angles),
        "unique_pair_overlap_fraction": _quantiles(pair_overlap),
        "unique_pair_crossing_score": _quantiles(pair_scores),
        "unique_pair_chord_axis_angle_degrees": _quantiles(
            pair_chord_axis_angle
        ),
        "unique_pair_root_distance": _quantiles(root_distance),
        "unique_pair_root_distance_over_mean_arc_length": _quantiles(
            root_distance_over_arc
        ),
        "unique_pair_root_distance_over_local_spacing": _quantiles(
            root_distance_over_local_spacing
        ),
        "unique_pair_minimum_contact_progress": _quantiles(
            minimum_contact_progress
        ),
        "unique_pair_maximum_contact_progress": _quantiles(
            maximum_contact_progress
        ),
        "unique_pair_angle_counts": angle_counts,
        "strand_angle_counts": angle_strand_counts,
        "high_angle_45_attribution": {
            "total": int(high_angle.sum()),
            "chord_axis_also_at_least_45_degrees": int(
                field_conflict.sum()
            ),
            "chord_axis_below_15_degrees": int(
                local_shape_conflict.sum()
            ),
            "chord_axis_between_15_and_45_degrees": int(
                mixed_conflict.sum()
            ),
        },
        "high_angle_45_contact_progress": {
            "first": _quantiles(pair_first_progress[high_angle]),
            "second": _quantiles(pair_second_progress[high_angle]),
            "both_in_root_quarter": int(
                (
                    high_angle
                    & (pair_first_progress < 0.25)
                    & (pair_second_progress < 0.25)
                ).sum()
            ),
            "either_in_tip_quarter": int(
                (
                    high_angle
                    & (
                        (pair_first_progress > 0.75)
                        | (pair_second_progress > 0.75)
                    )
                ).sum()
            ),
        },
        "high_angle_45_root_distance": _quantiles(
            root_distance[high_angle]
        ),
        "high_angle_45_root_distance_over_mean_arc_length": _quantiles(
            root_distance_over_arc[high_angle]
        ),
        "high_angle_45_root_distance_over_local_spacing": _quantiles(
            root_distance_over_local_spacing[high_angle]
        ),
        "high_angle_45_chord_axis_angle_degrees": _quantiles(
            pair_chord_axis_angle[high_angle]
        ),
        "per_strand_crossing_score": _quantiles(strand_score),
        "per_strand_contact_pair_count": _quantiles(strand_pair_count),
        "top_pairs": top_pairs,
    }
    arrays = {
        "crossing_contact_mask": contact_mask,
        "crossing_score": strand_score,
        "crossing_pair_count": strand_pair_count,
        "crossing_high_angle_15_mask": high_angle_masks[15.0],
        "crossing_high_angle_30_mask": high_angle_masks[30.0],
        "crossing_high_angle_45_mask": high_angle_masks[45.0],
        "crossing_high_angle_60_mask": high_angle_masks[60.0],
        "crossing_pair_first_strand": pair_first_strand.astype(np.int32),
        "crossing_pair_second_strand": pair_second_strand.astype(np.int32),
        "crossing_pair_first_segment": pair_first_segment,
        "crossing_pair_second_segment": pair_second_segment,
        "crossing_pair_first_progress": pair_first_progress,
        "crossing_pair_second_progress": pair_second_progress,
        "crossing_pair_score": pair_scores,
        "crossing_pair_contact_axis_angle_degrees": pair_angles,
        "crossing_pair_chord_axis_angle_degrees": pair_chord_axis_angle,
        "crossing_pair_overlap_fraction": pair_overlap,
        "crossing_pair_distance_over_width_sum": pair_distance_ratio,
        "crossing_pair_root_distance": root_distance.astype(np.float32),
    }
    return report, arrays


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose exact 3D inter-strand contacts using learned Gaussian "
            "transverse scales and angle-weighted crossing severity."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--query-batch", type=int, default=50000)
    parser.add_argument("--exact-pair-batch", type=int, default=250000)
    parser.add_argument("--workers", type=int, default=-1)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    with np.load(args.input, allow_pickle=False) as archive:
        if "strands" not in archive.files or "widths" not in archive.files:
            raise KeyError("input NPZ must contain strands and widths")
        strands = archive["strands"]
        widths = archive["widths"]
    report, diagnostic_arrays = diagnose_crossings(
        strands,
        widths,
        query_batch=int(args.query_batch),
        exact_pair_batch=int(args.exact_pair_batch),
        workers=int(args.workers),
    )
    report["input"] = str(args.input.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "strand_crossing_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with np.load(args.input, allow_pickle=False) as archive:
        source_arrays = {
            name: archive[name]
            for name in VISUALIZATION_ARRAY_KEYS
            if name in archive.files
        }
    output_npz = args.output_dir / "strand_crossing_diagnostic.npz"
    np.savez_compressed(output_npz, **source_arrays, **diagnostic_arrays)
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "diagnostic_npz": str(output_npz.resolve()),
                "strand_count": report["strand_count"],
                "unique_intersecting_strand_pairs": report[
                    "unique_intersecting_strand_pairs"
                ],
                "unique_pair_angle_counts": report[
                    "unique_pair_angle_counts"
                ],
                "strand_angle_counts": report["strand_angle_counts"],
                "high_angle_45_attribution": report[
                    "high_angle_45_attribution"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

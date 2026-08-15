from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare exact pair/contact visualization arrays from a strand-crossing diagnostic."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-angle-degrees", type=float, default=45.0)
    parser.add_argument("--min-overlap-fraction", type=float, default=0.0)
    parser.add_argument(
        "--pair-rank",
        type=int,
        default=-1,
        help="Select one pair by descending crossing score after filtering; negative keeps all pairs.",
    )
    parser.add_argument(
        "--isolate-selected-strands",
        action="store_true",
        help="Write only the selected pair strands; requires --pair-rank.",
    )
    return parser.parse_args()


def sample_polyline_at_progress(
    strands: np.ndarray, strand_ids: np.ndarray, progress: np.ndarray
) -> np.ndarray:
    segment_count = int(strands.shape[1] - 1)
    position = np.clip(progress, 0.0, 1.0) * float(segment_count)
    segment = np.minimum(np.floor(position).astype(np.int64), segment_count - 1)
    local_t = (position - segment).astype(np.float32)
    start = strands[strand_ids, segment]
    end = strands[strand_ids, segment + 1]
    return start + local_t[:, None] * (end - start)


def main() -> None:
    args = parse_args()
    source_path = Path(args.input)
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    source_arrays = arrays.copy()

    angles = np.asarray(
        arrays["crossing_pair_contact_axis_angle_degrees"], dtype=np.float32
    )
    overlap = np.asarray(
        arrays["crossing_pair_overlap_fraction"], dtype=np.float32
    )
    selected = np.logical_and(
        angles >= float(args.min_angle_degrees),
        overlap >= float(args.min_overlap_fraction),
    )
    selected_source_indices: np.ndarray
    if args.pair_rank >= 0:
        selected_indices = np.flatnonzero(selected)
        pair_scores = np.asarray(arrays["crossing_pair_score"], dtype=np.float32)
        selected_indices = selected_indices[
            np.argsort(pair_scores[selected_indices])[::-1]
        ]
        if args.pair_rank >= selected_indices.size:
            raise ValueError(
                f"pair rank {args.pair_rank} exceeds {selected_indices.size} filtered pairs"
            )
        selected[:] = False
        selected[selected_indices[int(args.pair_rank)]] = True
        selected_source_indices = selected_indices[
            int(args.pair_rank) : int(args.pair_rank) + 1
        ]
    elif args.isolate_selected_strands:
        raise ValueError("--isolate-selected-strands requires --pair-rank")
    else:
        selected_source_indices = np.flatnonzero(selected)
    first = np.asarray(arrays["crossing_pair_first_strand"], dtype=np.int64)[selected]
    second = np.asarray(arrays["crossing_pair_second_strand"], dtype=np.int64)[selected]
    first_progress = np.asarray(
        arrays["crossing_pair_first_progress"], dtype=np.float32
    )[selected]
    second_progress = np.asarray(
        arrays["crossing_pair_second_progress"], dtype=np.float32
    )[selected]
    strands = np.asarray(arrays["strands"], dtype=np.float32)
    first_points = sample_polyline_at_progress(strands, first, first_progress)
    second_points = sample_polyline_at_progress(strands, second, second_progress)
    contact_points = 0.5 * (first_points + second_points)
    selected_mask = np.zeros(strands.shape[0], dtype=bool)
    first_mask = np.zeros(strands.shape[0], dtype=bool)
    second_mask = np.zeros(strands.shape[0], dtype=bool)
    selected_mask[first] = True
    selected_mask[second] = True
    first_mask[first] = True
    second_mask[second] = True

    arrays["crossing_selected_mask"] = selected_mask
    arrays["crossing_selected_first_mask"] = first_mask
    arrays["crossing_selected_second_mask"] = second_mask
    arrays["crossing_selected_pair_mask"] = selected
    arrays["crossing_selected_first_strand"] = first.astype(np.int32)
    arrays["crossing_selected_second_strand"] = second.astype(np.int32)
    arrays["crossing_selected_contact_points"] = contact_points.astype(np.float32)
    arrays["crossing_selected_first_points"] = first_points.astype(np.float32)
    arrays["crossing_selected_second_points"] = second_points.astype(np.float32)

    first_tangent = None
    second_tangent = None
    recommended_camera_offset_project = None
    recommended_camera_offset_blender = None
    if first.size == 1:
        segment_count = int(strands.shape[1] - 1)
        first_segment = min(
            int(np.floor(float(first_progress[0]) * segment_count)),
            segment_count - 1,
        )
        second_segment = min(
            int(np.floor(float(second_progress[0]) * segment_count)),
            segment_count - 1,
        )
        first_tangent = strands[first[0], first_segment + 1] - strands[
            first[0], first_segment
        ]
        second_tangent = strands[second[0], second_segment + 1] - strands[
            second[0], second_segment
        ]
        first_tangent /= max(float(np.linalg.norm(first_tangent)), 1.0e-12)
        second_tangent /= max(float(np.linalg.norm(second_tangent)), 1.0e-12)
        view = np.cross(first_tangent, second_tangent)
        if float(np.linalg.norm(view)) < 1.0e-6:
            view = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        view /= max(float(np.linalg.norm(view)), 1.0e-12)
        recommended_camera_offset_project = view.astype(np.float32)
        recommended_camera_offset_blender = view[[2, 0, 1]].astype(np.float32)

    if args.isolate_selected_strands:
        isolated_ids = np.asarray([int(first[0]), int(second[0])], dtype=np.int64)
        arrays = {
            name: np.asarray(arrays[name])[isolated_ids]
            for name in ("strands", "widths", "colors", "opacities", "root_ids")
        }
        arrays["crossing_selected_mask"] = np.asarray([True, False])
        arrays["crossing_selected_first_mask"] = np.asarray([True, False])
        arrays["crossing_selected_second_mask"] = np.asarray([False, True])
        arrays["crossing_selected_contact_points"] = contact_points.astype(
            np.float32
        )
        arrays["crossing_selected_first_points"] = first_points.astype(np.float32)
        arrays["crossing_selected_second_points"] = second_points.astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    report = {
        "input": str(source_path.resolve()),
        "output": str(output_path.resolve()),
        "min_angle_degrees": float(args.min_angle_degrees),
        "min_overlap_fraction": float(args.min_overlap_fraction),
        "selected_pair_count": int(selected.sum()),
        "selected_strand_count": int(selected_mask.sum()),
        "pair_rank": int(args.pair_rank),
        "isolated": bool(args.isolate_selected_strands),
        "selected_pairs": [
            {
                "source_pair_index": int(source_index),
                "first_strand": int(
                    np.asarray(source_arrays["crossing_pair_first_strand"])[
                        source_index
                    ]
                ),
                "second_strand": int(
                    np.asarray(source_arrays["crossing_pair_second_strand"])[
                        source_index
                    ]
                ),
                "score": float(
                    np.asarray(source_arrays["crossing_pair_score"])[source_index]
                ),
                "contact_axis_angle_degrees": float(
                    np.asarray(
                        source_arrays["crossing_pair_contact_axis_angle_degrees"]
                    )[source_index]
                ),
                "chord_axis_angle_degrees": float(
                    np.asarray(
                        source_arrays["crossing_pair_chord_axis_angle_degrees"]
                    )[source_index]
                ),
                "overlap_fraction": float(
                    np.asarray(source_arrays["crossing_pair_overlap_fraction"])[
                        source_index
                    ]
                ),
                "first_progress": float(
                    np.asarray(source_arrays["crossing_pair_first_progress"])[
                        source_index
                    ]
                ),
                "second_progress": float(
                    np.asarray(source_arrays["crossing_pair_second_progress"])[
                        source_index
                    ]
                ),
            }
            for source_index in selected_source_indices
        ],
        "angle_degrees": {
            "minimum": float(angles[selected].min()) if np.any(selected) else None,
            "median": float(np.median(angles[selected])) if np.any(selected) else None,
            "maximum": float(angles[selected].max()) if np.any(selected) else None,
        },
        "overlap_fraction": {
            "minimum": float(overlap[selected].min()) if np.any(selected) else None,
            "median": float(np.median(overlap[selected])) if np.any(selected) else None,
            "maximum": float(overlap[selected].max()) if np.any(selected) else None,
        },
        "recommended_camera_offset_project": (
            recommended_camera_offset_project.tolist()
            if recommended_camera_offset_project is not None
            else None
        ),
        "recommended_camera_offset_blender": (
            recommended_camera_offset_blender.tolist()
            if recommended_camera_offset_blender is not None
            else None
        ),
        "first_tangent_project": (
            first_tangent.tolist() if first_tangent is not None else None
        ),
        "second_tangent_project": (
            second_tangent.tolist() if second_tangent is not None else None
        ),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

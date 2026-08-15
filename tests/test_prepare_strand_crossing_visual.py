import json
import sys

import numpy as np

from tools.prepare_strand_crossing_visual import (
    main,
    sample_polyline_at_progress,
)


def test_sample_polyline_at_progress_uses_continuous_segment_position():
    strands = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    sampled = sample_polyline_at_progress(
        strands,
        np.asarray([0, 0, 1]),
        np.asarray([0.0, 0.25, 1.0], dtype=np.float32),
    )

    np.testing.assert_allclose(
        sampled,
        np.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 2.0, 1.0]],
            dtype=np.float32,
        ),
    )


def test_pair_rank_isolates_two_strands_and_preserves_contact_metadata(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "crossing.npz"
    output_path = tmp_path / "pair.npz"
    strands = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        source_path,
        strands=strands,
        widths=np.full((3, 3), 0.01, dtype=np.float32),
        colors=np.full((3, 3, 3), 0.5, dtype=np.float32),
        opacities=np.ones((3, 3), dtype=np.float32),
        root_ids=np.asarray([10, 11, 12], dtype=np.int64),
        crossing_pair_first_strand=np.asarray([0, 1], dtype=np.int32),
        crossing_pair_second_strand=np.asarray([1, 2], dtype=np.int32),
        crossing_pair_first_progress=np.asarray([0.25, 0.5], dtype=np.float32),
        crossing_pair_second_progress=np.asarray([0.75, 0.5], dtype=np.float32),
        crossing_pair_contact_axis_angle_degrees=np.asarray(
            [60.0, 50.0], dtype=np.float32
        ),
        crossing_pair_chord_axis_angle_degrees=np.asarray(
            [70.0, 10.0], dtype=np.float32
        ),
        crossing_pair_overlap_fraction=np.asarray([0.8, 0.6], dtype=np.float32),
        crossing_pair_score=np.asarray([0.9, 0.5], dtype=np.float32),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_strand_crossing_visual.py",
            "--input",
            str(source_path),
            "--output",
            str(output_path),
            "--min-angle-degrees",
            "45",
            "--min-overlap-fraction",
            "0.5",
            "--pair-rank",
            "1",
            "--isolate-selected-strands",
        ],
    )

    main()

    with np.load(output_path, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["strands"], strands[[1, 2]])
        np.testing.assert_array_equal(
            result["crossing_selected_first_mask"], [True, False]
        )
        np.testing.assert_array_equal(
            result["crossing_selected_second_mask"], [False, True]
        )
        np.testing.assert_allclose(
            result["crossing_selected_contact_points"], [[1.0, 1.0, 0.5]]
        )

    report = json.loads(output_path.with_suffix(".json").read_text())
    assert report["selected_pair_count"] == 1
    assert report["selected_pairs"][0]["source_pair_index"] == 1
    assert report["selected_pairs"][0]["first_strand"] == 1
    assert report["selected_pairs"][0]["second_strand"] == 2
    assert report["selected_pairs"][0]["chord_axis_angle_degrees"] == 10.0

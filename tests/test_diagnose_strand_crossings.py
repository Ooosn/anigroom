from __future__ import annotations

import numpy as np

from tools.diagnose_strand_crossings import (
    closest_segment_parameters,
    diagnose_crossings,
)


def test_closest_segment_parameters_finds_perpendicular_crossing() -> None:
    first_t, second_t, distance = closest_segment_parameters(
        np.asarray([[-1.0, 0.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0]]),
        np.asarray([[0.0, -1.0, 0.0]]),
        np.asarray([[0.0, 1.0, 0.0]]),
    )

    np.testing.assert_allclose(first_t, [0.5])
    np.testing.assert_allclose(second_t, [0.5])
    np.testing.assert_allclose(distance, [0.0])


def test_crossing_diagnostic_separates_3d_crossing_from_projection_overlap() -> None:
    strands = np.asarray(
        [
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[-1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, -1.0, 1.5], [0.0, 0.0, 1.5], [0.0, 1.0, 1.5]],
        ],
        dtype=np.float32,
    )
    widths = np.full((4, 3, 1), 0.1, dtype=np.float32)

    report, arrays = diagnose_crossings(
        strands,
        widths,
        query_batch=4,
        exact_pair_batch=8,
        workers=1,
    )

    assert report["unique_intersecting_strand_pairs"] == 1
    assert report["unique_pair_angle_counts"][">=45_degrees"] == 1
    assert report["high_angle_45_attribution"] == {
        "total": 1,
        "chord_axis_also_at_least_45_degrees": 1,
        "chord_axis_below_15_degrees": 0,
        "chord_axis_between_15_and_45_degrees": 0,
    }
    np.testing.assert_array_equal(
        arrays["crossing_high_angle_45_mask"],
        np.asarray([True, True, False, False]),
    )
    assert arrays["crossing_pair_first_progress"].shape == (1,)
    assert arrays["crossing_pair_second_progress"].shape == (1,)


def test_parallel_envelope_contact_is_not_high_angle_crossing() -> None:
    strands = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.1, 0.0], [0.5, 0.1, 0.0], [1.0, 0.1, 0.0]],
        ],
        dtype=np.float32,
    )
    widths = np.full((2, 3, 1), 0.075, dtype=np.float32)

    report, arrays = diagnose_crossings(
        strands,
        widths,
        query_batch=2,
        exact_pair_batch=4,
        workers=1,
    )

    assert report["unique_intersecting_strand_pairs"] == 1
    assert report["unique_pair_angle_counts"][">=15_degrees"] == 0
    assert arrays["crossing_contact_mask"].all()
    assert not arrays["crossing_high_angle_15_mask"].any()
    np.testing.assert_allclose(arrays["crossing_score"], 0.0, atol=1.0e-8)

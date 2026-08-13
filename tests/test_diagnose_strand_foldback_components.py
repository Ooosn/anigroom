from __future__ import annotations

import numpy as np

from tools.diagnose_strand_foldback_components import (
    backward_report,
    percentile_ranks,
)


def test_backward_report_detects_local_foldback_not_world_direction() -> None:
    strands = np.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            [
                [2.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ],
        dtype=np.float32,
    )

    report, mask = backward_report(strands)

    assert mask.tolist() == [False, False, True]
    assert report["strands_with_backward_segment"] == 1
    assert report["backward_subset_indices"] == [2]
    assert report["backward_min_projection_segment"] == [1]


def test_percentile_ranks_use_the_complete_population() -> None:
    population = np.asarray([1.0, 2.0, 3.0, 4.0])
    selected = np.asarray([1.0, 2.5, 4.0])

    ranks = percentile_ranks(population, selected)

    np.testing.assert_allclose(ranks, [0.25, 0.5, 1.0])

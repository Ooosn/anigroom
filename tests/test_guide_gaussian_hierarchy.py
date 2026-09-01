from __future__ import annotations

import numpy as np
import pytest

from anigroom.grooming.guide_attribute_gaussian_field import (
    density_preserving_topology_fps,
)


def _chain(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.stack(
        [
            np.arange(count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
            np.zeros(count, dtype=np.float64),
        ],
        axis=1,
    )
    edges = np.stack(
        [np.arange(count - 1), np.arange(1, count)],
        axis=1,
    ).astype(np.int64)
    lengths = np.ones((count - 1,), dtype=np.float64)
    return points, edges, lengths


def test_uniform_chain_is_nested_deterministic_and_improves_cover() -> None:
    points, edges, lengths = _chain(7)
    first = density_preserving_topology_fps(
        points,
        edges,
        lengths,
        np.ones(7),
        5,
    )
    second = density_preserving_topology_fps(
        points,
        edges,
        lengths,
        np.ones(7),
        5,
    )

    np.testing.assert_array_equal(first.selected_ids, second.selected_ids)
    np.testing.assert_array_equal(
        first.normalized_cover_max,
        second.normalized_cover_max,
    )
    np.testing.assert_array_equal(first.selected_ids[:3], second.selected_ids[:3])
    assert np.unique(first.selected_ids).size == 5
    assert bool(np.all(first.normalized_cover_max[1:] <= first.normalized_cover_max[:-1]))
    assert first.report["nested_prefixes"] is True
    assert first.report["fallback_used"] is False
    assert first.report["initial_normalized_cover_max"] > first.report[
        "final_normalized_cover_max"
    ]


def test_small_density_spacing_changes_selection_priority() -> None:
    points, edges, lengths = _chain(7)
    uniform = density_preserving_topology_fps(
        points,
        edges,
        lengths,
        np.ones(7),
        3,
    )
    dense_right = density_preserving_topology_fps(
        points,
        edges,
        lengths,
        np.asarray([1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.2]),
        3,
    )

    assert int(uniform.selected_ids[0]) == 0
    assert int(dense_right.selected_ids[0]) == 6
    assert sum(int(value >= 4) for value in dense_right.selected_ids) > sum(
        int(value >= 4) for value in uniform.selected_ids
    )


def test_duplicate_directed_edges_keep_minimum_positive_length() -> None:
    points, edges, lengths = _chain(4)
    duplicate_edges = np.concatenate((edges, edges[:, ::-1], edges[:1]), axis=0)
    duplicate_lengths = np.concatenate(
        (lengths, lengths * 2.0, np.asarray([3.0])),
        axis=0,
    )
    result = density_preserving_topology_fps(
        points,
        duplicate_edges,
        duplicate_lengths,
        np.ones(4),
        3,
    )

    assert result.report["unique_undirected_edge_count"] == 3
    assert result.report["duplicate_undirected_edges_removed"] == 4


def test_disconnected_graph_fails() -> None:
    points, _edges, _lengths = _chain(4)
    with pytest.raises(RuntimeError, match="one connected guide graph"):
        density_preserving_topology_fps(
            points,
            np.asarray([[0, 1], [2, 3]], dtype=np.int64),
            np.ones(2),
            np.ones(4),
            2,
        )


@pytest.mark.parametrize(
    ("edges", "lengths", "spacing", "count", "error"),
    [
        (np.asarray([[0, 0]]), np.ones(1), np.ones(4), 2, ValueError),
        (np.asarray([[0, 4]]), np.ones(1), np.ones(4), 2, ValueError),
        (np.asarray([[0.0, 1.0]]), np.ones(1), np.ones(4), 2, TypeError),
        (np.asarray([[0, 1]]), np.zeros(1), np.ones(4), 2, ValueError),
        (np.asarray([[0, 1]]), np.ones(1), np.asarray([1.0, 1.0, 0.0, 1.0]), 2, ValueError),
        (np.asarray([[0, 1]]), np.ones(1), np.ones(4), 0, ValueError),
        (np.asarray([[0, 1]]), np.ones(1), np.ones(4), 5, ValueError),
    ],
)
def test_invalid_inputs_fail(
    edges: np.ndarray,
    lengths: np.ndarray,
    spacing: np.ndarray,
    count: int,
    error: type[Exception],
) -> None:
    points, _, _ = _chain(4)
    with pytest.raises(error):
        density_preserving_topology_fps(
            points,
            edges,
            lengths,
            spacing,
            count,
        )

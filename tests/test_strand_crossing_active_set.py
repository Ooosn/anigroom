from __future__ import annotations

import numpy as np
import torch

from anigroom.collision.strand_crossing import (
    GaussianSegmentSnapshot,
    StrandCrossingActiveSet,
    active_set_crossing_loss,
    discover_gaussian_segment_crossings,
)


def snapshot_from_segments(
    starts: np.ndarray,
    ends: np.ndarray,
    root_indices: np.ndarray,
    *,
    widths: float | np.ndarray = 0.1,
    length_overlap: float = 1.45,
) -> GaussianSegmentSnapshot:
    starts = np.asarray(starts, dtype=np.float32)
    ends = np.asarray(ends, dtype=np.float32)
    delta = ends - starts
    length = np.linalg.norm(delta, axis=1)
    direction = delta / length[:, None]
    transverse = np.broadcast_to(
        np.asarray(widths, dtype=np.float32), length.shape
    )
    scales = np.stack(
        [0.5 * length * float(length_overlap), transverse, transverse], axis=1
    )
    return GaussianSegmentSnapshot(
        means=0.5 * (starts + ends),
        directions=direction,
        scales=scales,
        root_indices=np.asarray(root_indices, dtype=np.int64),
        segment_indices=np.zeros(len(starts), dtype=np.int64),
        length_overlap=float(length_overlap),
    )


def test_discovery_finds_physical_crossing_but_not_projection_only_crossing() -> None:
    crossing = snapshot_from_segments(
        np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([0, 1]),
    )
    active, report = discover_gaussian_segment_crossings(crossing)
    assert active.pair_count == 1
    assert report["active_root_pair_count"] == 1
    assert np.isclose(active.first_progress[0], 0.5, atol=1.0e-6)
    assert np.isclose(active.second_progress[0], 0.5, atol=1.0e-6)
    assert np.isclose(active.angle_weights[0], 1.0, atol=1.0e-6)
    assert np.isclose(np.linalg.norm(active.separation_axes[0]), 1.0)

    projected_only = snapshot_from_segments(
        np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.5]]),
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]]),
        np.asarray([0, 1]),
    )
    active, report = discover_gaussian_segment_crossings(projected_only)
    assert active.pair_count == 0
    assert report["active_root_pair_count"] == 0


def test_parallel_dense_fur_has_zero_crossing_weight() -> None:
    snapshot = snapshot_from_segments(
        np.asarray([[-1.0, 0.0, 0.0], [-1.0, 0.1, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.1, 0.0]]),
        np.asarray([0, 1]),
    )
    active, _ = discover_gaussian_segment_crossings(snapshot)
    assert active.pair_count == 1
    assert active.angle_weights[0] == 0.0

    strands = torch.tensor(
        [[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
         [[-1.0, 0.1, 0.0], [1.0, 0.1, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    widths = torch.full((2, 2, 1), 0.1, dtype=torch.float64)
    loss, stats = active_set_crossing_loss(strands, widths, active.to_torch("cpu"))
    assert float(loss) == 0.0
    assert stats["active_pair_count"] == 1


def test_nonzero_contact_uses_the_physical_separation_axis() -> None:
    snapshot = snapshot_from_segments(
        np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.1]]),
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.1]]),
        np.asarray([0, 1]),
    )
    active, _ = discover_gaussian_segment_crossings(snapshot)
    assert active.pair_count == 1
    assert np.allclose(
        active.separation_axes[0],
        np.asarray([0.0, 0.0, -1.0]),
        atol=1.0e-6,
    )


def test_pair_reach_filter_removes_global_reach_false_candidates() -> None:
    snapshot = snapshot_from_segments(
        np.asarray(
            [
                [-0.05, 0.0, 0.0],
                [9.95, 0.0, 0.0],
                [100.0, -100.0, 0.0],
            ]
        ),
        np.asarray(
            [
                [0.05, 0.0, 0.0],
                [10.05, 0.0, 0.0],
                [100.0, 100.0, 0.0],
            ]
        ),
        np.asarray([0, 1, 2]),
        widths=0.05,
    )
    active, report = discover_gaussian_segment_crossings(snapshot)
    assert active.pair_count == 0
    assert (
        report["broadphase_candidate_segment_pairs"]
        > report["sphere_filtered_candidate_segment_pairs"]
    )
    assert (
        report["sphere_filtered_candidate_segment_pairs"]
        == report["exact_tested_segment_pairs"]
    )


def test_exact_perpendicular_crossing_has_separating_geometry_gradient_only() -> None:
    snapshot = snapshot_from_segments(
        np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([0, 1]),
    )
    active, _ = discover_gaussian_segment_crossings(snapshot)
    strands = torch.tensor(
        [[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
         [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    widths = torch.full(
        (2, 2, 1), 0.1, dtype=torch.float64, requires_grad=True
    )
    loss, stats = active_set_crossing_loss(strands, widths, active.to_torch("cpu"))
    assert torch.isclose(loss, torch.ones_like(loss))
    assert stats["positive_pair_count"] == 1
    loss.backward()
    assert strands.grad is not None
    assert float(strands.grad[..., 2].abs().sum()) > 0.0
    assert widths.grad is None


def test_active_set_checkpoint_round_trip() -> None:
    snapshot = snapshot_from_segments(
        np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([5, 9]),
    )
    active, _ = discover_gaussian_segment_crossings(snapshot)
    restored = StrandCrossingActiveSet.from_checkpoint_state(
        active.checkpoint_state()
    )
    assert restored.source_segment_count == active.source_segment_count
    assert np.array_equal(restored.first_root_indices, active.first_root_indices)
    assert np.array_equal(restored.second_root_indices, active.second_root_indices)
    assert np.allclose(restored.first_progress, active.first_progress)
    assert np.allclose(restored.second_progress, active.second_progress)
    assert np.allclose(restored.separation_axes, active.separation_axes)

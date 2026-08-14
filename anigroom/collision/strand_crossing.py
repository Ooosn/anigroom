from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F


_NUMPY_EPS = np.finfo(np.float64).eps * 64.0


@dataclass(frozen=True)
class GaussianSegmentSnapshot:
    """Detached rendered centerline segments used for active-set discovery."""

    means: np.ndarray
    directions: np.ndarray
    scales: np.ndarray
    root_indices: np.ndarray
    segment_indices: np.ndarray
    length_overlap: float

    @classmethod
    def from_tensors(
        cls,
        *,
        means: torch.Tensor,
        directions: torch.Tensor,
        scales: torch.Tensor,
        root_indices: torch.Tensor,
        segment_indices: torch.Tensor,
        length_overlap: float,
    ) -> "GaussianSegmentSnapshot":
        return cls(
            means=means.detach().to(device="cpu", dtype=torch.float32).numpy(),
            directions=directions.detach().to(device="cpu", dtype=torch.float32).numpy(),
            scales=scales.detach().to(device="cpu", dtype=torch.float32).numpy(),
            root_indices=root_indices.detach().to(device="cpu", dtype=torch.int64).numpy(),
            segment_indices=segment_indices.detach().to(device="cpu", dtype=torch.int64).numpy(),
            length_overlap=float(length_overlap),
        )

    def validate(self) -> None:
        count = int(self.means.shape[0])
        if self.means.shape != (count, 3):
            raise ValueError("Gaussian means must have shape [G, 3]")
        if self.directions.shape != (count, 3):
            raise ValueError("Gaussian directions must have shape [G, 3]")
        if self.scales.shape != (count, 3):
            raise ValueError("Gaussian scales must have shape [G, 3]")
        if self.root_indices.shape != (count,):
            raise ValueError("Gaussian root_indices must have shape [G]")
        if self.segment_indices.shape != (count,):
            raise ValueError("Gaussian segment_indices must have shape [G]")
        if count == 0:
            raise ValueError("crossing discovery requires at least one Gaussian segment")
        if not np.isfinite(self.means).all():
            raise ValueError("Gaussian means contain non-finite values")
        if not np.isfinite(self.directions).all():
            raise ValueError("Gaussian directions contain non-finite values")
        if not np.isfinite(self.scales).all() or np.any(self.scales <= 0.0):
            raise ValueError("Gaussian scales must be finite and positive")
        if np.any(self.root_indices < 0) or np.any(self.segment_indices < 0):
            raise ValueError("Gaussian ownership indices must be non-negative")
        if not np.isfinite(self.length_overlap) or self.length_overlap <= 0.0:
            raise ValueError("length_overlap must be finite and positive")


@dataclass(frozen=True)
class StrandCrossingActiveSet:
    """One strongest physical contact per unordered render-root pair."""

    first_root_indices: np.ndarray
    second_root_indices: np.ndarray
    first_progress: np.ndarray
    second_progress: np.ndarray
    separation_axes: np.ndarray
    angle_weights: np.ndarray
    discovery_overlap: np.ndarray
    discovery_scores: np.ndarray
    source_segment_count: int

    @property
    def pair_count(self) -> int:
        return int(self.first_root_indices.shape[0])

    def validate(self) -> None:
        count = self.pair_count
        vectors = {
            "second_root_indices": self.second_root_indices,
            "first_progress": self.first_progress,
            "second_progress": self.second_progress,
            "angle_weights": self.angle_weights,
            "discovery_overlap": self.discovery_overlap,
            "discovery_scores": self.discovery_scores,
        }
        if self.first_root_indices.shape != (count,):
            raise ValueError("first_root_indices must have shape [P]")
        for name, value in vectors.items():
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape [P]")
        if self.separation_axes.shape != (count, 3):
            raise ValueError("separation_axes must have shape [P, 3]")
        if np.any(self.first_root_indices >= self.second_root_indices):
            raise ValueError("active-set root pairs must be strictly ordered")
        if not (
            np.isfinite(self.first_progress).all()
            and np.isfinite(self.second_progress).all()
            and np.isfinite(self.separation_axes).all()
            and np.isfinite(self.angle_weights).all()
            and np.isfinite(self.discovery_overlap).all()
            and np.isfinite(self.discovery_scores).all()
        ):
            raise ValueError("active set contains non-finite values")
        if np.any(self.first_progress < 0.0) or np.any(self.first_progress > 1.0):
            raise ValueError("first_progress must lie in [0, 1]")
        if np.any(self.second_progress < 0.0) or np.any(self.second_progress > 1.0):
            raise ValueError("second_progress must lie in [0, 1]")
        if np.any(self.angle_weights < 0.0) or np.any(self.angle_weights > 1.0):
            raise ValueError("angle_weights must lie in [0, 1]")

    def to_torch(
        self,
        device: torch.device | str,
    ) -> "TorchStrandCrossingActiveSet":
        self.validate()
        roots = np.concatenate(
            [self.first_root_indices, self.second_root_indices]
        ).astype(np.int64, copy=False)
        unique_roots, inverse = np.unique(roots, return_inverse=True)
        pair_count = self.pair_count
        return TorchStrandCrossingActiveSet(
            unique_root_indices=torch.from_numpy(unique_roots).to(device=device),
            first_local_indices=torch.from_numpy(inverse[:pair_count]).to(
                device=device
            ),
            second_local_indices=torch.from_numpy(inverse[pair_count:]).to(
                device=device
            ),
            first_progress=torch.from_numpy(
                self.first_progress.astype(np.float32, copy=False)
            ).to(device=device),
            second_progress=torch.from_numpy(
                self.second_progress.astype(np.float32, copy=False)
            ).to(device=device),
            separation_axes=torch.from_numpy(
                self.separation_axes.astype(np.float32, copy=False)
            ).to(device=device),
            angle_weights=torch.from_numpy(
                self.angle_weights.astype(np.float32, copy=False)
            ).to(device=device),
        )

    def checkpoint_state(self) -> dict[str, Any]:
        self.validate()
        return {
            "first_root_indices": torch.from_numpy(
                self.first_root_indices.astype(np.int64, copy=False)
            ),
            "second_root_indices": torch.from_numpy(
                self.second_root_indices.astype(np.int64, copy=False)
            ),
            "first_progress": torch.from_numpy(
                self.first_progress.astype(np.float32, copy=False)
            ),
            "second_progress": torch.from_numpy(
                self.second_progress.astype(np.float32, copy=False)
            ),
            "separation_axes": torch.from_numpy(
                self.separation_axes.astype(np.float32, copy=False)
            ),
            "angle_weights": torch.from_numpy(
                self.angle_weights.astype(np.float32, copy=False)
            ),
            "discovery_overlap": torch.from_numpy(
                self.discovery_overlap.astype(np.float32, copy=False)
            ),
            "discovery_scores": torch.from_numpy(
                self.discovery_scores.astype(np.float32, copy=False)
            ),
            "source_segment_count": int(self.source_segment_count),
        }

    @classmethod
    def from_checkpoint_state(
        cls,
        state: dict[str, Any],
    ) -> "StrandCrossingActiveSet":
        def array(name: str, dtype: np.dtype) -> np.ndarray:
            value = state[name]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=dtype)

        active = cls(
            first_root_indices=array("first_root_indices", np.int64),
            second_root_indices=array("second_root_indices", np.int64),
            first_progress=array("first_progress", np.float32),
            second_progress=array("second_progress", np.float32),
            separation_axes=array("separation_axes", np.float32),
            angle_weights=array("angle_weights", np.float32),
            discovery_overlap=array("discovery_overlap", np.float32),
            discovery_scores=array("discovery_scores", np.float32),
            source_segment_count=int(state["source_segment_count"]),
        )
        active.validate()
        return active


@dataclass(frozen=True)
class TorchStrandCrossingActiveSet:
    unique_root_indices: torch.Tensor
    first_local_indices: torch.Tensor
    second_local_indices: torch.Tensor
    first_progress: torch.Tensor
    second_progress: torch.Tensor
    separation_axes: torch.Tensor
    angle_weights: torch.Tensor

    @property
    def pair_count(self) -> int:
        return int(self.first_local_indices.numel())


def closest_segment_parameters_numpy(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact closest-point parameters and distances for segment pairs."""

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

    first_numerator = cross * second_offset - second_sq * first_offset
    second_numerator = first_sq * second_offset - cross * first_offset
    first_denominator = denominator.copy()
    second_denominator = denominator.copy()

    parallel = denominator <= _NUMPY_EPS * np.maximum(first_sq * second_sq, 1.0)
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
    second_numerator[above_first] = second_offset[above_first] + cross[above_first]
    second_denominator[above_first] = second_sq[above_first]

    below_second = second_numerator < 0.0
    second_numerator[below_second] = 0.0
    first_before = below_second & (-first_offset < 0.0)
    first_after = below_second & (-first_offset > first_sq)
    first_inside = below_second & ~(first_before | first_after)
    first_numerator[first_before] = 0.0
    first_numerator[first_after] = first_denominator[first_after]
    first_numerator[first_inside] = -first_offset[first_inside]
    first_denominator[first_inside] = first_sq[first_inside]

    above_second = second_numerator > second_denominator
    second_numerator[above_second] = second_denominator[above_second]
    projected_first = -first_offset + cross
    first_before = above_second & (projected_first < 0.0)
    first_after = above_second & (projected_first > first_sq)
    first_inside = above_second & ~(first_before | first_after)
    first_numerator[first_before] = 0.0
    first_numerator[first_after] = first_denominator[first_after]
    first_numerator[first_inside] = projected_first[first_inside]
    first_denominator[first_inside] = first_sq[first_inside]

    first_parameter = np.divide(
        first_numerator,
        first_denominator,
        out=np.zeros_like(first_numerator),
        where=np.abs(first_denominator) > _NUMPY_EPS,
    )
    second_parameter = np.divide(
        second_numerator,
        second_denominator,
        out=np.zeros_like(second_numerator),
        where=np.abs(second_denominator) > _NUMPY_EPS,
    )
    first_parameter = np.clip(first_parameter, 0.0, 1.0)
    second_parameter = np.clip(second_parameter, 0.0, 1.0)
    separation = (
        offset
        + first_parameter[:, None] * first_delta
        - second_parameter[:, None] * second_delta
    )
    return first_parameter, second_parameter, np.linalg.norm(separation, axis=1)


def discover_gaussian_segment_crossings(
    snapshot: GaussianSegmentSnapshot,
    *,
    query_batch: int = 50000,
    exact_pair_batch: int = 250000,
    workers: int = -1,
) -> tuple[StrandCrossingActiveSet, dict[str, Any]]:
    """Discover all physical Gaussian-centerline contacts without gradients."""

    snapshot.validate()
    if query_batch <= 0 or exact_pair_batch <= 0:
        raise ValueError("discovery batch sizes must be positive")

    means = np.asarray(snapshot.means, dtype=np.float64)
    directions = np.asarray(snapshot.directions, dtype=np.float64)
    scales = np.asarray(snapshot.scales, dtype=np.float64)
    root_indices = np.asarray(snapshot.root_indices, dtype=np.int64)
    segment_indices = np.asarray(snapshot.segment_indices, dtype=np.int64)
    direction_norm = np.linalg.norm(directions, axis=1)
    if np.any(direction_norm <= _NUMPY_EPS):
        raise ValueError("Gaussian segment directions must be nonzero")
    directions = directions / direction_norm[:, None]
    half_length = scales[:, 0] / float(snapshot.length_overlap)
    radii = np.maximum(scales[:, 1], scales[:, 2])
    starts = means - half_length[:, None] * directions
    ends = means + half_length[:, None] * directions
    reach = half_length + radii
    tree = cKDTree(means)
    global_reach = float(reach.max())

    root_count = int(root_indices.max()) + 1
    segment_counts = np.zeros(root_count, dtype=np.int64)
    np.maximum.at(segment_counts, root_indices, segment_indices + 1)
    if np.any(segment_counts[root_indices] <= 0):
        raise RuntimeError("failed to recover per-root Gaussian segment counts")

    records: list[dict[str, np.ndarray]] = []
    broadphase_candidates = 0
    exact_tested = 0
    segment_count = int(means.shape[0])
    for query_start in range(0, segment_count, int(query_batch)):
        query_end = min(query_start + int(query_batch), segment_count)
        neighbors = tree.query_ball_point(
            means[query_start:query_end],
            reach[query_start:query_end] + global_reach,
            workers=int(workers),
        )
        counts = np.fromiter(
            (len(indices) for indices in neighbors),
            dtype=np.int64,
            count=query_end - query_start,
        )
        if int(counts.sum()) == 0:
            continue
        first = np.repeat(
            np.arange(query_start, query_end, dtype=np.int64), counts
        )
        second = np.concatenate(neighbors).astype(np.int64, copy=False)
        valid = (second > first) & (root_indices[first] != root_indices[second])
        first = first[valid]
        second = second[valid]
        broadphase_candidates += int(first.size)

        batch_records: list[dict[str, np.ndarray]] = []
        for pair_start in range(0, first.size, int(exact_pair_batch)):
            pair_end = min(pair_start + int(exact_pair_batch), first.size)
            first_segment = first[pair_start:pair_end]
            second_segment = second[pair_start:pair_end]
            exact_tested += int(first_segment.size)
            first_t, second_t, distance = closest_segment_parameters_numpy(
                starts[first_segment],
                ends[first_segment],
                starts[second_segment],
                ends[second_segment],
            )
            radius_sum = radii[first_segment] + radii[second_segment]
            contact = distance < radius_sum
            if not np.any(contact):
                continue
            first_segment = first_segment[contact]
            second_segment = second_segment[contact]
            first_t = first_t[contact]
            second_t = second_t[contact]
            distance = distance[contact]
            radius_sum = radius_sum[contact]
            first_root = root_indices[first_segment]
            second_root = root_indices[second_segment]
            first_progress = (
                segment_indices[first_segment].astype(np.float64) + first_t
            ) / segment_counts[first_root]
            second_progress = (
                segment_indices[second_segment].astype(np.float64) + second_t
            ) / segment_counts[second_root]
            axis_cosine = np.abs(
                np.einsum(
                    "nd,nd->n",
                    directions[first_segment],
                    directions[second_segment],
                )
            ).clip(0.0, 1.0)
            angle_weight = 1.0 - axis_cosine * axis_cosine
            overlap = ((radius_sum - distance) / radius_sum).clip(0.0, 1.0)
            score = overlap * angle_weight
            first_point = starts[first_segment] + first_t[:, None] * (
                ends[first_segment] - starts[first_segment]
            )
            second_point = starts[second_segment] + second_t[:, None] * (
                ends[second_segment] - starts[second_segment]
            )
            separation = first_point - second_point
            separation_norm = np.linalg.norm(separation, axis=1)
            axes = np.zeros_like(separation)
            separated = separation_norm > _NUMPY_EPS
            axes[separated] = (
                separation[separated] / separation_norm[separated, None]
            )
            intersecting = ~separated
            if np.any(intersecting):
                fallback = np.cross(
                    directions[first_segment[intersecting]],
                    directions[second_segment[intersecting]],
                )
                fallback_norm = np.linalg.norm(fallback, axis=1)
                usable = fallback_norm > _NUMPY_EPS
                fallback[usable] /= fallback_norm[usable, None]
                fallback[~usable] = 0.0
                axes[intersecting] = fallback

            swap = first_root > second_root
            lower = np.minimum(first_root, second_root)
            upper = np.maximum(first_root, second_root)
            normalized_first_progress = np.where(
                swap, second_progress, first_progress
            )
            normalized_second_progress = np.where(
                swap, first_progress, second_progress
            )
            normalized_axes = np.where(swap[:, None], -axes, axes)
            keys = lower * root_count + upper
            batch_records.append(
                {
                    "keys": keys,
                    "first_root": lower,
                    "second_root": upper,
                    "first_progress": normalized_first_progress,
                    "second_progress": normalized_second_progress,
                    "axes": normalized_axes,
                    "angle_weight": angle_weight,
                    "overlap": overlap,
                    "score": score,
                }
            )

        if batch_records:
            merged = {
                name: np.concatenate([record[name] for record in batch_records])
                for name in batch_records[0]
            }
            order = np.lexsort((-merged["score"], merged["keys"]))
            sorted_keys = merged["keys"][order]
            selected = order[np.r_[0, 1 + np.flatnonzero(np.diff(sorted_keys))]]
            records.append({name: value[selected] for name, value in merged.items()})

    if records:
        merged = {
            name: np.concatenate([record[name] for record in records])
            for name in records[0]
        }
        order = np.lexsort((-merged["score"], merged["keys"]))
        sorted_keys = merged["keys"][order]
        selected = order[np.r_[0, 1 + np.flatnonzero(np.diff(sorted_keys))]]
        first_root = merged["first_root"][selected]
        second_root = merged["second_root"][selected]
        first_progress = merged["first_progress"][selected]
        second_progress = merged["second_progress"][selected]
        axes = merged["axes"][selected]
        angle_weight = merged["angle_weight"][selected]
        overlap = merged["overlap"][selected]
        scores = merged["score"][selected]
    else:
        first_root = np.empty((0,), dtype=np.int64)
        second_root = np.empty((0,), dtype=np.int64)
        first_progress = np.empty((0,), dtype=np.float64)
        second_progress = np.empty((0,), dtype=np.float64)
        axes = np.empty((0, 3), dtype=np.float64)
        angle_weight = np.empty((0,), dtype=np.float64)
        overlap = np.empty((0,), dtype=np.float64)
        scores = np.empty((0,), dtype=np.float64)

    active = StrandCrossingActiveSet(
        first_root_indices=first_root.astype(np.int64, copy=False),
        second_root_indices=second_root.astype(np.int64, copy=False),
        first_progress=first_progress.astype(np.float32, copy=False),
        second_progress=second_progress.astype(np.float32, copy=False),
        separation_axes=axes.astype(np.float32, copy=False),
        angle_weights=angle_weight.astype(np.float32, copy=False),
        discovery_overlap=overlap.astype(np.float32, copy=False),
        discovery_scores=scores.astype(np.float32, copy=False),
        source_segment_count=segment_count,
    )
    active.validate()
    report = {
        "source_segment_count": segment_count,
        "source_root_count": int(np.unique(root_indices).size),
        "broadphase_candidate_segment_pairs": int(broadphase_candidates),
        "exact_tested_segment_pairs": int(exact_tested),
        "active_root_pair_count": active.pair_count,
        "active_root_count": int(
            np.unique(
                np.concatenate(
                    [active.first_root_indices, active.second_root_indices]
                )
            ).size
        )
        if active.pair_count
        else 0,
        "mean_discovery_overlap": float(active.discovery_overlap.mean())
        if active.pair_count
        else 0.0,
        "mean_discovery_score": float(active.discovery_scores.mean())
        if active.pair_count
        else 0.0,
        "maximum_discovery_score": float(active.discovery_scores.max())
        if active.pair_count
        else 0.0,
    }
    return active, report


def _sample_strand_field_at_arc_progress(
    strands: torch.Tensor,
    widths: torch.Tensor,
    local_root_indices: torch.Tensor,
    progress: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segment = strands[:, 1:] - strands[:, :-1]
    segment_length = torch.linalg.vector_norm(segment, dim=-1)
    cumulative = torch.cat(
        [
            segment_length.new_zeros((segment_length.shape[0], 1)),
            torch.cumsum(segment_length, dim=1),
        ],
        dim=1,
    )
    row_cumulative = cumulative[local_root_indices]
    target = progress.clamp(0.0, 1.0) * row_cumulative[:, -1]
    with torch.no_grad():
        segment_index = (
            torch.searchsorted(
                row_cumulative.contiguous(),
                target[:, None].contiguous(),
                right=True,
            ).squeeze(-1)
            - 1
        ).clamp(0, strands.shape[1] - 2)
        arc_start = row_cumulative.gather(1, segment_index[:, None]).squeeze(1)
        local_length = segment_length[
            local_root_indices, segment_index
        ].clamp_min(torch.finfo(strands.dtype).eps)
        local_t = ((target - arc_start) / local_length).clamp(0.0, 1.0)

    start = strands[local_root_indices, segment_index]
    end = strands[local_root_indices, segment_index + 1]
    point = start + local_t[:, None] * (end - start)
    tangent = F.normalize(end - start, dim=-1, eps=1.0e-8)
    width_start = widths[local_root_indices, segment_index, 0]
    width_end = widths[local_root_indices, segment_index + 1, 0]
    width = width_start + local_t * (width_end - width_start)
    return point, tangent, width


def active_set_crossing_loss(
    strands: torch.Tensor,
    widths: torch.Tensor,
    active_set: TorchStrandCrossingActiveSet,
    *,
    pair_batch: int = 65536,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Resolve active contacts without changing strand width or global pose."""

    if strands.ndim != 3 or strands.shape[-1] != 3 or strands.shape[1] < 2:
        raise ValueError("strands must have shape [R, S>=2, 3]")
    if widths.shape != (*strands.shape[:2], 1):
        raise ValueError("widths must have shape [R, S, 1]")
    if strands.shape[0] != active_set.unique_root_indices.numel():
        raise ValueError("strands must correspond to active_set.unique_root_indices")
    if pair_batch <= 0:
        raise ValueError("pair_batch must be positive")
    pair_count = active_set.pair_count
    if pair_count == 0:
        zero = strands.sum() * 0.0
        return zero, {
            "active_pair_count": 0,
            "positive_pair_count": 0,
            "positive_pair_fraction": zero.detach(),
            "mean_normalized_depth": zero.detach(),
            "maximum_normalized_depth": zero.detach(),
        }

    weighted_sum = strands.new_zeros(())
    weight_sum = strands.new_zeros(())
    depth_sum = strands.new_zeros(())
    maximum_depth = strands.new_zeros(())
    positive_count = 0
    for start in range(0, pair_count, int(pair_batch)):
        stop = min(start + int(pair_batch), pair_count)
        first_point, _, first_width = _sample_strand_field_at_arc_progress(
            strands,
            widths,
            active_set.first_local_indices[start:stop],
            active_set.first_progress[start:stop],
        )
        second_point, _, second_width = _sample_strand_field_at_arc_progress(
            strands,
            widths,
            active_set.second_local_indices[start:stop],
            active_set.second_progress[start:stop],
        )
        axes = active_set.separation_axes[start:stop].detach()
        pair_weight = active_set.angle_weights[start:stop].detach()
        radius_sum = (first_width + second_width).detach().clamp_min(
            torch.finfo(strands.dtype).eps
        )
        signed_clearance = ((first_point - second_point) * axes).sum(dim=-1)
        normalized_depth = torch.relu(
            (radius_sum - signed_clearance) / radius_sum
        )
        weighted_sum = weighted_sum + (pair_weight * normalized_depth).sum()
        weight_sum = weight_sum + pair_weight.sum()
        depth_sum = depth_sum + normalized_depth.detach().sum()
        maximum_depth = torch.maximum(
            maximum_depth, normalized_depth.detach().max()
        )
        positive_count += int((normalized_depth.detach() > 0.0).sum().cpu())

    loss = weighted_sum / weight_sum.clamp_min(
        torch.finfo(strands.dtype).eps
    )
    return loss, {
        "active_pair_count": pair_count,
        "positive_pair_count": positive_count,
        "positive_pair_fraction": strands.new_tensor(
            positive_count / float(pair_count)
        ),
        "mean_normalized_depth": depth_sum / float(pair_count),
        "maximum_normalized_depth": maximum_depth,
    }

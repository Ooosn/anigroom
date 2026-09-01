"""Deterministic nested guide subsets over a fixed surface-topology graph."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
import torch


def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _points(value: np.ndarray | torch.Tensor) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 2 or int(array.shape[1]) != 3:
        raise ValueError("points must have shape [G, 3]")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("points must be real floating-point values")
    result = np.ascontiguousarray(array, dtype=np.float64)
    if result.shape[0] == 0:
        raise ValueError("points must not be empty")
    if not np.isfinite(result).all():
        raise ValueError("points contain non-finite values")
    if np.unique(result, axis=0).shape[0] != result.shape[0]:
        raise ValueError("points must be unique")
    return result


def _integer_edges(value: np.ndarray | torch.Tensor, guide_count: int) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 2 or int(array.shape[1]) != 2:
        raise ValueError("graph_edges must have shape [E, 2]")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("graph_edges must use an integer dtype")
    edges = np.ascontiguousarray(array, dtype=np.int64)
    if edges.shape[0] == 0:
        raise ValueError("graph_edges must not be empty")
    if np.any(edges < 0) or np.any(edges >= int(guide_count)):
        raise ValueError("graph_edges contain out-of-range node IDs")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("graph_edges must not contain self edges")
    return edges


def _positive_vector(
    value: np.ndarray | torch.Tensor,
    *,
    name: str,
    count: int,
) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim != 1 or tuple(array.shape) != (int(count),):
        raise ValueError(f"{name} must have shape [{count}]")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must use a real floating dtype")
    result = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _stable_undirected_graph(
    edges: np.ndarray,
    lengths: np.ndarray,
    guide_count: int,
) -> tuple[Any, int]:
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    order = np.lexsort((lengths, hi, lo))
    lo = lo[order]
    hi = hi[order]
    lengths = lengths[order]
    pair_change = np.ones((lo.shape[0],), dtype=bool)
    if pair_change.size > 1:
        pair_change[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    starts = np.flatnonzero(pair_change)
    unique_lo = lo[starts]
    unique_hi = hi[starts]
    unique_lengths = np.minimum.reduceat(lengths, starts)
    rows = np.concatenate((unique_lo, unique_hi))
    cols = np.concatenate((unique_hi, unique_lo))
    data = np.concatenate((unique_lengths, unique_lengths))
    graph = coo_matrix(
        (data, (rows, cols)),
        shape=(int(guide_count), int(guide_count)),
        dtype=np.float64,
    ).tocsr()
    graph.sort_indices()
    return graph, int(unique_lo.shape[0])


@dataclass(frozen=True)
class NestedTopologyFPS:
    """One nested density-preserving topology-FPS order and cover history."""

    selected_ids: np.ndarray
    normalized_cover_max: np.ndarray
    report: dict[str, Any]


def density_preserving_topology_fps(
    points: np.ndarray | torch.Tensor,
    graph_edges: np.ndarray | torch.Tensor,
    edge_lengths: np.ndarray | torch.Tensor,
    density_spacing: np.ndarray | torch.Tensor,
    max_count: int,
) -> NestedTopologyFPS:
    """Select one nested guide order with topology distance and density scale.

    The score for an unselected node is its shortest topology distance to the
    selected set divided by its desired local spacing. Small-spacing regions
    therefore receive proportionally more controls without any semantic or
    image-space rule. Equal scores choose the smallest stable node ID.
    """

    pts = _points(points)
    guide_count = int(pts.shape[0])
    edges = _integer_edges(graph_edges, guide_count)
    edge_length = _positive_vector(
        edge_lengths,
        name="edge_lengths",
        count=int(edges.shape[0]),
    )
    spacing = _positive_vector(
        density_spacing,
        name="density_spacing",
        count=guide_count,
    )
    if isinstance(max_count, bool) or not isinstance(max_count, Integral):
        raise TypeError("max_count must be an integer")
    selected_count = int(max_count)
    if selected_count <= 0 or selected_count > guide_count:
        raise ValueError("max_count must lie in [1, guide_count]")

    graph, unique_edge_count = _stable_undirected_graph(
        edges,
        edge_length,
        guide_count,
    )
    component_count, _labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )
    if int(component_count) != 1:
        raise RuntimeError(
            "density-preserving topology FPS requires one connected guide graph"
        )

    centroid = pts.mean(axis=0, keepdims=True)
    start_score = np.linalg.norm(pts - centroid, axis=1) / spacing
    if not np.isfinite(start_score).all():
        raise RuntimeError("topology FPS produced an invalid start score")
    current = int(np.flatnonzero(start_score == start_score.max())[0])

    selected = np.empty((selected_count,), dtype=np.int64)
    cover_history = np.empty((selected_count,), dtype=np.float64)
    selected_mask = np.zeros((guide_count,), dtype=bool)
    minimum_distance = np.full((guide_count,), np.inf, dtype=np.float64)

    for output_index in range(selected_count):
        if selected_mask[current]:
            raise RuntimeError("topology FPS attempted to select one node twice")
        selected[output_index] = current
        selected_mask[current] = True
        distance = np.asarray(
            dijkstra(graph, directed=False, indices=current),
            dtype=np.float64,
        ).reshape(-1)
        if distance.shape != (guide_count,) or not np.isfinite(distance).all():
            raise RuntimeError("topology FPS encountered unreachable guide nodes")
        minimum_distance = np.minimum(minimum_distance, distance)
        normalized_cover = minimum_distance / spacing
        if not np.isfinite(normalized_cover).all():
            raise RuntimeError("topology FPS produced a non-finite cover score")
        cover_history[output_index] = float(normalized_cover.max())
        if output_index + 1 < selected_count:
            score = normalized_cover.copy()
            score[selected_mask] = -np.inf
            best = float(score.max())
            if not np.isfinite(best) or best < 0.0:
                raise RuntimeError("topology FPS exhausted valid unselected nodes")
            current = int(np.flatnonzero(score == best)[0])

    if np.unique(selected).shape[0] != selected.shape[0]:
        raise RuntimeError("topology FPS output contains duplicate node IDs")
    if np.any(cover_history[1:] > cover_history[:-1] + 1.0e-12):
        raise RuntimeError("topology FPS cover history must be non-increasing")
    selected.setflags(write=False)
    cover_history.setflags(write=False)
    return NestedTopologyFPS(
        selected_ids=selected,
        normalized_cover_max=cover_history,
        report={
            "schema": "anigroom.guide_gaussian.nested_topology_fps.v1",
            "guide_count": guide_count,
            "selected_count": selected_count,
            "input_directed_edge_count": int(edges.shape[0]),
            "unique_undirected_edge_count": unique_edge_count,
            "duplicate_undirected_edges_removed": int(
                edges.shape[0] - unique_edge_count
            ),
            "component_count": int(component_count),
            "density_spacing_min": float(spacing.min()),
            "density_spacing_max": float(spacing.max()),
            "start_node": int(selected[0]),
            "start_rule": "max_euclidean_centroid_distance_over_density_spacing",
            "selection_rule": "max_min_topology_distance_over_density_spacing",
            "tie_break": "smallest_stable_node_id",
            "initial_normalized_cover_max": float(cover_history[0]),
            "final_normalized_cover_max": float(cover_history[-1]),
            "nested_prefixes": True,
            "fallback_used": False,
        },
    )

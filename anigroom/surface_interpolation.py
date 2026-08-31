"""Topology-safe interpolation for mesh-rooted grooming attributes.

Neighbor selection is discrete and rebuilt only when root topology changes.
The inherited inverse-distance weights are differentiable while a cached
support is fixed, but truncated neighboring support sets do not guarantee one
globally continuous field across the surface. Attribute combination is typed:
physical scalars/colors use arithmetic interpolation, while 3D directions are
parallel transported before averaging.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

from anigroom.flow.direction_geometry import parallel_transport_vectors
from anigroom.flow.surface_graph import _augmented_surface_graph, _root_voronoi_graph


EPS = 1.0e-8


def adaptive_wendland_c2_weights(
    distances: torch.Tensor,
    active_neighbor_count: int,
    support_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return normalized compact-support Wendland C2 weights.

    The support radius for each query is the ``(K + 1)``-th smallest entry in
    ``distances``, where ``K`` is ``active_neighbor_count``.  The boundary
    entry therefore has exactly zero kernel mass.  Support selection and the
    radius remain part of the autograd graph for valid inputs.
    """

    if not isinstance(distances, torch.Tensor):
        raise TypeError("distances must be a torch.Tensor")
    if distances.ndim != 2:
        raise ValueError("distances must have shape [Q, S]")
    if not torch.is_floating_point(distances) or distances.is_complex():
        raise TypeError("distances must be a real floating-point tensor")
    if isinstance(active_neighbor_count, bool) or not isinstance(
        active_neighbor_count,
        Integral,
    ):
        raise ValueError("active_neighbor_count must be an integer")
    neighbor_count = int(active_neighbor_count)
    if neighbor_count <= 0:
        raise ValueError("active_neighbor_count must be positive")
    source_count = int(distances.shape[1])
    if source_count < neighbor_count + 1:
        raise ValueError("distances must contain at least K + 1 sources")
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("distances must be finite")
    if bool((distances < 0.0).any()):
        raise ValueError("distances must be nonnegative")

    if support_indices is not None:
        if not isinstance(support_indices, torch.Tensor):
            raise TypeError("support_indices must be a torch.Tensor")
        if support_indices.ndim != 2 or tuple(support_indices.shape) != tuple(
            distances.shape
        ):
            raise ValueError("support_indices must have shape [Q, S] matching distances")
        if support_indices.dtype not in {
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("support_indices must be an integer tensor")
        sorted_indices = torch.sort(support_indices, dim=1).values
        if sorted_indices.shape[1] > 1 and bool(
            (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()
        ):
            raise ValueError("support_indices must be unique within every query row")

    radius = distances.kthvalue(neighbor_count + 1, dim=1, keepdim=True).values
    if not bool(torch.isfinite(radius).all()) or bool((radius <= 0.0).any()):
        raise ValueError("every query support radius must be finite and positive")

    inside = distances < radius
    safe_distances = torch.where(inside, distances, torch.zeros_like(distances))
    normalized_distance = safe_distances / radius
    kernel = (1.0 - normalized_distance).pow(4) * (4.0 * normalized_distance + 1.0)
    raw_weights = torch.where(inside, kernel, torch.zeros_like(kernel))
    denominator = raw_weights.sum(dim=1, keepdim=True)
    valid_denominator = torch.isfinite(denominator) & (denominator > 0.0)
    if not bool(valid_denominator.all()):
        raise RuntimeError("adaptive Wendland weights have a zero or nonfinite denominator")

    weights = raw_weights / denominator
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0.0).any()):
        raise RuntimeError("adaptive Wendland weights must be finite and nonnegative")
    row_sum = weights.sum(dim=1, keepdim=True)
    if not bool(torch.isfinite(row_sum).all()) or not bool(
        torch.allclose(row_sum, torch.ones_like(row_sum), rtol=1.0e-5, atol=1.0e-6)
    ):
        raise RuntimeError("adaptive Wendland weights must be row-normalized")
    return weights


@dataclass(frozen=True)
class SurfaceSupport:
    """Topology-valid source IDs and source-to-query-face vertex paths."""

    indices: torch.Tensor
    vertex_path_distances: torch.Tensor
    report: dict[str, float | int]

    @property
    def query_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def neighbor_count(self) -> int:
        return int(self.indices.shape[1])


@dataclass(frozen=True)
class LocalSurfaceSupport:
    """Face-neighborhood support for a small set of lifecycle children."""

    indices: torch.Tensor
    report: dict[str, float | int]


@dataclass(frozen=True)
class SurfaceSourceGraph:
    """Intrinsic source neighborhoods with density-aware quadrature weights."""

    edges: torch.Tensor
    distances: torch.Tensor
    source_area_weights: torch.Tensor
    reference_spacing: torch.Tensor


def _source_neighbors(
    root_graph,
    *,
    neighbor_count: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite intrinsic source neighbors, including the source itself."""

    source_count = int(root_graph.shape[0])
    count = max(1, min(int(neighbor_count), source_count))
    indices = np.empty((source_count, count), dtype=np.int64)
    distances = np.empty((source_count, count), dtype=np.float64)
    for begin in range(0, source_count, int(chunk_size)):
        end = min(begin + int(chunk_size), source_count)
        source_ids = np.arange(begin, end, dtype=np.int64)
        batch = np.asarray(
            dijkstra(root_graph, directed=False, indices=source_ids),
            dtype=np.float64,
        )
        for row, source_id in enumerate(source_ids.tolist()):
            finite = np.flatnonzero(np.isfinite(batch[row]))
            order = finite[np.argsort(batch[row, finite], kind="stable")]
            chosen = order[:count]
            if chosen.size == 0:
                raise RuntimeError(f"surface source {source_id} has no finite topology support")
            if chosen.size < count:
                chosen = np.pad(chosen, (0, count - chosen.size), constant_values=int(chosen[0]))
            indices[source_id] = chosen
            distances[source_id] = batch[row, chosen]
    return indices, distances


class SurfaceFieldInterpolator:
    """Reusable intrinsic source topology with differentiable query weights."""

    def __init__(
        self,
        *,
        vertices: np.ndarray | torch.Tensor,
        faces: np.ndarray | torch.Tensor,
        source_points: np.ndarray | torch.Tensor,
        source_face_ids: np.ndarray | torch.Tensor,
        neighbor_count: int,
        device: torch.device | str,
    ) -> None:
        started = perf_counter()
        vertices_np = np.asarray(
            vertices.detach().cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices,
            dtype=np.float64,
        )
        faces_np = np.asarray(
            faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else faces,
            dtype=np.int64,
        )
        source_points_np = np.asarray(
            source_points.detach().cpu().numpy() if isinstance(source_points, torch.Tensor) else source_points,
            dtype=np.float64,
        )
        source_face_ids_np = np.asarray(
            source_face_ids.detach().cpu().numpy()
            if isinstance(source_face_ids, torch.Tensor)
            else source_face_ids,
            dtype=np.int64,
        ).reshape(-1)
        if source_points_np.ndim != 2 or source_points_np.shape[1] != 3:
            raise ValueError("source_points must have shape [S, 3]")
        if source_face_ids_np.shape[0] != source_points_np.shape[0]:
            raise ValueError("source_face_ids must have one entry per source")
        if source_points_np.shape[0] <= 0:
            raise ValueError("surface interpolation requires at least one source")

        graph, root_nodes, edge_u, edge_v = _augmented_surface_graph(
            vertices_np,
            faces_np,
            source_points_np,
            source_face_ids_np,
        )
        nearest_distance, _, nearest_source = dijkstra(
            graph,
            directed=False,
            indices=root_nodes,
            min_only=True,
            return_predecessors=True,
        )
        vertex_count = int(vertices_np.shape[0])
        vertex_source = np.asarray(nearest_source[:vertex_count], dtype=np.int64) - vertex_count
        vertex_distance = np.asarray(nearest_distance[:vertex_count], dtype=np.float64)
        if np.any(vertex_source < 0) or np.any(vertex_source >= source_points_np.shape[0]):
            raise RuntimeError("mesh topology contains vertices without a valid surface source")

        root_graph = _root_voronoi_graph(graph, root_nodes, edge_u, edge_v)
        candidate_count = min(
            int(source_points_np.shape[0]),
            max(int(neighbor_count) + 1, int(neighbor_count) * 2),
        )
        source_neighbors, source_neighbor_distances = _source_neighbors(
            root_graph,
            neighbor_count=candidate_count,
        )

        target_device = torch.device(device)
        self.vertices = torch.as_tensor(vertices_np, device=target_device, dtype=torch.float32)
        self.faces = torch.as_tensor(faces_np, device=target_device, dtype=torch.long)
        self.source_points_reference = torch.as_tensor(
            source_points_np,
            device=target_device,
            dtype=torch.float32,
        )
        self.source_face_ids = torch.as_tensor(
            source_face_ids_np,
            device=target_device,
            dtype=torch.long,
        )
        self.neighbor_count = max(1, min(int(neighbor_count), int(source_points_np.shape[0])))
        self._vertices_np = vertices_np
        self._faces_np = faces_np
        self._vertex_source = vertex_source
        self._vertex_distance = vertex_distance
        self._source_neighbors = source_neighbors
        self._source_neighbor_distances = source_neighbor_distances
        self._support_vertices = torch.as_tensor(
            vertices_np,
            device=target_device,
            dtype=torch.float64,
        )
        self._support_vertex_source = torch.as_tensor(
            vertex_source,
            device=target_device,
            dtype=torch.long,
        )
        self._support_vertex_distance = torch.as_tensor(
            vertex_distance,
            device=target_device,
            dtype=torch.float64,
        )
        self._support_source_neighbors = torch.as_tensor(
            source_neighbors,
            device=target_device,
            dtype=torch.long,
        )
        self._support_source_neighbor_distances = torch.as_tensor(
            source_neighbor_distances,
            device=target_device,
            dtype=torch.float64,
        )
        self.report: dict[str, float | int] = {
            "source_count": int(source_points_np.shape[0]),
            "mesh_vertex_count": int(vertices_np.shape[0]),
            "mesh_face_count": int(faces_np.shape[0]),
            "neighbor_count": int(self.neighbor_count),
            "source_graph_edge_count": int(root_graph.nnz // 2),
            "build_seconds": float(perf_counter() - started),
        }

    def source_neighbor_edges(self, neighbor_count: int) -> torch.Tensor:
        """Return intrinsic directed edges between the interpolation sources.

        The source Voronoi graph is already built to support surface-aware
        interpolation. Reusing it here keeps guide smoothing and guide-to-render
        interpolation on the same mesh-topology contract.
        """

        source_count = int(self.source_points_reference.shape[0])
        if source_count < 2 or int(neighbor_count) <= 0:
            return torch.empty((0, 2), device=self.vertices.device, dtype=torch.long)
        count = min(int(neighbor_count), source_count - 1)
        source_ids = np.arange(source_count, dtype=np.int64)
        selected = np.empty((source_count, count), dtype=np.int64)
        for source_id in source_ids.tolist():
            candidates = self._source_neighbors[source_id]
            distances = self._source_neighbor_distances[source_id]
            valid = (candidates != source_id) & np.isfinite(distances)
            candidates = candidates[valid]
            if candidates.size < count:
                raise RuntimeError(
                    f"surface source {source_id} has only {candidates.size} intrinsic "
                    f"neighbors for K={count}"
                )
            selected[source_id] = candidates[:count]
        src = np.repeat(source_ids, count)
        edges = np.stack([src, selected.reshape(-1)], axis=-1)
        return torch.as_tensor(edges, device=self.vertices.device, dtype=torch.long)

    def source_neighbor_graph(self, neighbor_count: int) -> SurfaceSourceGraph:
        """Return metric data for a sampling-density-invariant source field loss."""

        source_count = int(self.source_points_reference.shape[0])
        if source_count < 2 or int(neighbor_count) <= 0:
            empty_edges = torch.empty(
                (0, 2), device=self.vertices.device, dtype=torch.long
            )
            empty_values = torch.empty(
                (0,), device=self.vertices.device, dtype=self.vertices.dtype
            )
            return SurfaceSourceGraph(
                edges=empty_edges,
                distances=empty_values,
                source_area_weights=empty_values,
                reference_spacing=self.vertices.new_tensor(0.0),
            )

        count = min(int(neighbor_count), source_count - 1)
        source_ids = np.arange(source_count, dtype=np.int64)
        selected = np.empty((source_count, count), dtype=np.int64)
        selected_distances = np.empty((source_count, count), dtype=np.float64)
        for source_id in source_ids.tolist():
            candidates = self._source_neighbors[source_id]
            distances = self._source_neighbor_distances[source_id]
            valid = (candidates != source_id) & np.isfinite(distances)
            candidates = candidates[valid]
            distances = distances[valid]
            if candidates.size < count:
                raise RuntimeError(
                    f"surface source {source_id} has only {candidates.size} intrinsic "
                    f"neighbors for K={count}"
                )
            selected[source_id] = candidates[:count]
            selected_distances[source_id] = distances[:count]

        src = np.repeat(source_ids, count)
        edges = np.stack([src, selected.reshape(-1)], axis=-1)
        local_spacing = np.median(selected_distances, axis=1)
        if not np.isfinite(local_spacing).all() or np.any(local_spacing <= 0.0):
            raise RuntimeError("surface source graph contains a non-positive spacing")
        target_device = self.vertices.device
        return SurfaceSourceGraph(
            edges=torch.as_tensor(edges, device=target_device, dtype=torch.long),
            distances=torch.as_tensor(
                selected_distances.reshape(-1),
                device=target_device,
                dtype=self.vertices.dtype,
            ),
            source_area_weights=torch.as_tensor(
                np.square(local_spacing),
                device=target_device,
                dtype=self.vertices.dtype,
            ),
            reference_spacing=torch.as_tensor(
                np.median(local_spacing),
                device=target_device,
                dtype=self.vertices.dtype,
            ),
        )

    def build_support(
        self,
        query_points: np.ndarray | torch.Tensor,
        query_face_ids: np.ndarray | torch.Tensor,
    ) -> SurfaceSupport:
        """Build fixed source IDs for query roots without cross-surface KNN."""

        started = perf_counter()
        if isinstance(query_points, torch.Tensor) and self.vertices.device.type == "cuda":
            return self._build_support_cuda(query_points, query_face_ids, started=started)
        query_points_np = np.asarray(
            query_points.detach().cpu().numpy()
            if isinstance(query_points, torch.Tensor)
            else query_points,
            dtype=np.float64,
        )
        query_face_ids_np = np.asarray(
            query_face_ids.detach().cpu().numpy()
            if isinstance(query_face_ids, torch.Tensor)
            else query_face_ids,
            dtype=np.int64,
        ).reshape(-1)
        if query_points_np.ndim != 2 or query_points_np.shape[1] != 3:
            raise ValueError("query_points must have shape [Q, 3]")
        if query_face_ids_np.shape[0] != query_points_np.shape[0]:
            raise ValueError("query_face_ids must have one entry per query")

        query_count = int(query_points_np.shape[0])
        indices = np.empty((query_count, self.neighbor_count), dtype=np.int64)
        vertex_paths = np.full(
            (query_count, self.neighbor_count, 3),
            np.inf,
            dtype=np.float32,
        )
        candidate_count = int(self._source_neighbors.shape[1])
        fallback_queries: list[int] = []
        for begin in range(0, query_count, 2048):
            end = min(begin + 2048, query_count)
            query_faces = self._faces_np[query_face_ids_np[begin:end]]
            query_vertices = self._vertices_np[query_faces]
            query_vertex_distances = np.linalg.norm(
                query_points_np[begin:end, None, :] - query_vertices,
                axis=-1,
            )
            seeds = self._vertex_source[query_faces]
            candidate_ids = self._source_neighbors[seeds]
            candidate_paths = (
                self._vertex_distance[query_faces, None]
                + self._source_neighbor_distances[seeds]
            )
            candidate_scores = candidate_paths + query_vertex_distances[:, :, None]
            batch_size = end - begin
            flat_ids = candidate_ids.reshape(batch_size, 3 * candidate_count)
            flat_scores = candidate_scores.reshape(batch_size, 3 * candidate_count)

            # Preserve the reference dictionary's first-occurrence order while
            # aggregating candidates reached through multiple face vertices.
            equal_ids = flat_ids[:, :, None] == flat_ids[:, None, :]
            repeated_from_earlier = np.tril(equal_ids, k=-1).any(axis=2)
            aggregate_scores = np.where(
                equal_ids,
                flat_scores[:, None, :],
                np.inf,
            ).min(axis=2)
            aggregate_scores[repeated_from_earlier] = np.inf
            order = np.argsort(aggregate_scores, axis=1, kind="stable")[:, : self.neighbor_count]
            chosen_ids = np.take_along_axis(flat_ids, order, axis=1)
            chosen_scores = np.take_along_axis(aggregate_scores, order, axis=1)
            normal = np.isfinite(chosen_scores).all(axis=1)
            if np.any(normal):
                normal_ids = chosen_ids[normal]
                indices[begin:end][normal] = normal_ids
                matches = candidate_ids[normal, None, :, :] == normal_ids[:, :, None, None]
                selected_paths = np.where(
                    matches,
                    candidate_paths[normal, None, :, :],
                    np.inf,
                ).min(axis=3)
                vertex_paths[begin:end][normal] = selected_paths.astype(np.float32)
            fallback_queries.extend((begin + np.flatnonzero(~normal)).tolist())

        # Disconnected or very small source components can contain fewer than
        # K unique candidates. Preserve the original padded-support contract.
        for query_id in fallback_queries:
            query_face = self._faces_np[query_face_ids_np[query_id]]
            query_vertices = self._vertices_np[query_face]
            query_vertex_distances = np.linalg.norm(
                query_points_np[query_id : query_id + 1] - query_vertices,
                axis=-1,
            )
            paths: dict[int, np.ndarray] = {}
            for vertex_slot, vertex_id in enumerate(query_face.tolist()):
                seed = int(self._vertex_source[vertex_id])
                base = float(self._vertex_distance[vertex_id])
                for candidate_id, source_distance in zip(
                    self._source_neighbors[seed].tolist(),
                    self._source_neighbor_distances[seed].tolist(),
                ):
                    values = paths.setdefault(
                        int(candidate_id),
                        np.full((3,), np.inf, dtype=np.float64),
                    )
                    values[vertex_slot] = min(values[vertex_slot], base + float(source_distance))
            if not paths:
                raise RuntimeError(f"query {query_id} has no topology-valid interpolation source")
            candidate_ids = np.fromiter(paths.keys(), dtype=np.int64)
            candidate_paths = np.stack([paths[int(candidate_id)] for candidate_id in candidate_ids])
            score = np.min(candidate_paths + query_vertex_distances[None, :], axis=1)
            order = np.argsort(score, kind="stable")
            chosen_ids = candidate_ids[order[: self.neighbor_count]]
            chosen_paths = candidate_paths[order[: self.neighbor_count]]
            if chosen_ids.size < self.neighbor_count:
                pad = self.neighbor_count - int(chosen_ids.size)
                chosen_ids = np.pad(chosen_ids, (0, pad), constant_values=int(chosen_ids[0]))
                chosen_paths = np.concatenate(
                    [chosen_paths, np.repeat(chosen_paths[:1], pad, axis=0)],
                    axis=0,
                )
            indices[query_id] = chosen_ids
            vertex_paths[query_id] = chosen_paths.astype(np.float32)

        target_device = self.vertices.device
        return SurfaceSupport(
            indices=torch.as_tensor(indices, device=target_device, dtype=torch.long),
            vertex_path_distances=torch.as_tensor(
                vertex_paths,
                device=target_device,
                dtype=self.vertices.dtype,
            ),
            report={
                **self.report,
                "query_count": query_count,
                "support_seconds": float(perf_counter() - started),
                "support_bytes": int(indices.nbytes + vertex_paths.nbytes),
                "fallback_query_count": int(len(fallback_queries)),
            },
        )

    def _build_support_cuda(
        self,
        query_points: torch.Tensor,
        query_face_ids: np.ndarray | torch.Tensor,
        *,
        started: float,
    ) -> SurfaceSupport:
        """CUDA implementation of the exact fixed-support selection contract."""

        device = self.vertices.device
        points = query_points.detach().to(device=device, dtype=torch.float64)
        face_ids = torch.as_tensor(query_face_ids, device=device, dtype=torch.long).reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("query_points must have shape [Q, 3]")
        if int(face_ids.shape[0]) != int(points.shape[0]):
            raise ValueError("query_face_ids must have one entry per query")

        query_count = int(points.shape[0])
        indices = torch.empty(
            (query_count, self.neighbor_count),
            device=device,
            dtype=torch.long,
        )
        vertex_paths = torch.full(
            (query_count, self.neighbor_count, 3),
            torch.inf,
            device=device,
            dtype=self.vertices.dtype,
        )
        candidate_count = int(self._support_source_neighbors.shape[1])
        flat_candidate_count = 3 * candidate_count
        matrix_entries_per_query = flat_candidate_count * flat_candidate_count
        chunk_size = max(
            1,
            min(query_count, 20_000_000 // max(matrix_entries_per_query, 1)),
        )
        fallback_batches: list[torch.Tensor] = []
        for begin in range(0, query_count, chunk_size):
            end = min(begin + chunk_size, query_count)
            query_faces = self.faces[face_ids[begin:end]]
            query_vertices = self._support_vertices[query_faces]
            query_vertex_distances = torch.linalg.norm(
                points[begin:end, None, :] - query_vertices,
                dim=-1,
            )
            seeds = self._support_vertex_source[query_faces]
            candidate_ids = self._support_source_neighbors[seeds]
            candidate_paths = (
                self._support_vertex_distance[query_faces, None]
                + self._support_source_neighbor_distances[seeds]
            )
            candidate_scores = candidate_paths + query_vertex_distances[:, :, None]
            batch_size = end - begin
            flat_ids = candidate_ids.reshape(batch_size, flat_candidate_count)
            flat_scores = candidate_scores.reshape(batch_size, flat_candidate_count)

            equal_ids = flat_ids[:, :, None] == flat_ids[:, None, :]
            repeated_from_earlier = torch.tril(equal_ids, diagonal=-1).any(dim=2)
            aggregate_scores = torch.where(
                equal_ids,
                flat_scores[:, None, :],
                torch.inf,
            ).amin(dim=2)
            aggregate_scores.masked_fill_(repeated_from_earlier, torch.inf)
            order = torch.argsort(aggregate_scores, dim=1, stable=True)[
                :, : self.neighbor_count
            ]
            chosen_ids = torch.gather(flat_ids, 1, order)
            chosen_scores = torch.gather(aggregate_scores, 1, order)
            normal = torch.isfinite(chosen_scores).all(dim=1)

            batch_indices = indices[begin:end]
            batch_indices[normal] = chosen_ids[normal]
            normal_candidate_ids = candidate_ids[normal]
            normal_chosen_ids = chosen_ids[normal]
            matches = (
                normal_candidate_ids[:, None, :, :]
                == normal_chosen_ids[:, :, None, None]
            )
            selected_paths = torch.where(
                matches,
                candidate_paths[normal, None, :, :],
                torch.inf,
            ).amin(dim=3)
            batch_vertex_paths = vertex_paths[begin:end]
            batch_vertex_paths[normal] = selected_paths.to(dtype=self.vertices.dtype)
            fallback_batches.append(
                torch.arange(begin, end, device=device, dtype=torch.long)[~normal]
            )

        fallback_ids = torch.cat(fallback_batches).cpu().tolist() if fallback_batches else []
        if fallback_ids:
            query_points_np = np.asarray(points.cpu().numpy(), dtype=np.float64)
            query_face_ids_np = np.asarray(face_ids.cpu().numpy(), dtype=np.int64)
            for query_id in fallback_ids:
                query_face = self._faces_np[query_face_ids_np[query_id]]
                query_vertices = self._vertices_np[query_face]
                query_vertex_distances = np.linalg.norm(
                    query_points_np[query_id : query_id + 1] - query_vertices,
                    axis=-1,
                )
                paths: dict[int, np.ndarray] = {}
                for vertex_slot, vertex_id in enumerate(query_face.tolist()):
                    seed = int(self._vertex_source[vertex_id])
                    base = float(self._vertex_distance[vertex_id])
                    for candidate_id, source_distance in zip(
                        self._source_neighbors[seed].tolist(),
                        self._source_neighbor_distances[seed].tolist(),
                    ):
                        values = paths.setdefault(
                            int(candidate_id),
                            np.full((3,), np.inf, dtype=np.float64),
                        )
                        values[vertex_slot] = min(
                            values[vertex_slot],
                            base + float(source_distance),
                        )
                if not paths:
                    raise RuntimeError(
                        f"query {query_id} has no topology-valid interpolation source"
                    )
                candidate_ids_np = np.fromiter(paths.keys(), dtype=np.int64)
                candidate_paths_np = np.stack(
                    [paths[int(candidate_id)] for candidate_id in candidate_ids_np]
                )
                score = np.min(
                    candidate_paths_np + query_vertex_distances[None, :],
                    axis=1,
                )
                fallback_order = np.argsort(score, kind="stable")
                chosen_ids_np = candidate_ids_np[fallback_order[: self.neighbor_count]]
                chosen_paths_np = candidate_paths_np[
                    fallback_order[: self.neighbor_count]
                ]
                if chosen_ids_np.size < self.neighbor_count:
                    pad = self.neighbor_count - int(chosen_ids_np.size)
                    chosen_ids_np = np.pad(
                        chosen_ids_np,
                        (0, pad),
                        constant_values=int(chosen_ids_np[0]),
                    )
                    chosen_paths_np = np.concatenate(
                        [
                            chosen_paths_np,
                            np.repeat(chosen_paths_np[:1], pad, axis=0),
                        ],
                        axis=0,
                    )
                indices[query_id] = torch.as_tensor(
                    chosen_ids_np,
                    device=device,
                    dtype=torch.long,
                )
                vertex_paths[query_id] = torch.as_tensor(
                    chosen_paths_np,
                    device=device,
                    dtype=self.vertices.dtype,
                )

        torch.cuda.synchronize(device)
        return SurfaceSupport(
            indices=indices,
            vertex_path_distances=vertex_paths,
            report={
                **self.report,
                "query_count": query_count,
                "support_seconds": float(perf_counter() - started),
                "support_bytes": int(
                    indices.numel() * indices.element_size()
                    + vertex_paths.numel() * vertex_paths.element_size()
                ),
                "fallback_query_count": int(len(fallback_ids)),
                "backend": "cuda_exact",
            },
        )

    def weights(
        self,
        query_points: torch.Tensor,
        query_face_ids: torch.Tensor,
        support: SurfaceSupport,
        *,
        source_confidence: torch.Tensor | None = None,
        power: float = 2.0,
    ) -> torch.Tensor:
        if int(query_points.shape[0]) != support.query_count:
            raise ValueError("query count does not match cached surface support")
        distances = self.distances(query_points, query_face_ids, support)
        weights = distances.clamp_min(1.0e-6).pow(-float(power))
        if source_confidence is not None:
            confidence = source_confidence.reshape(-1).clamp(0.0, 1.0)
            weights = weights * confidence[support.indices]
        denominator = weights.sum(dim=-1, keepdim=True)
        if bool((denominator <= EPS).any()):
            raise RuntimeError("surface interpolation has a query with zero valid source weight")
        return weights / denominator

    def distances(
        self,
        query_points: torch.Tensor,
        query_face_ids: torch.Tensor,
        support: SurfaceSupport,
    ) -> torch.Tensor:
        """Return differentiable intrinsic-path distances for cached support."""

        if int(query_points.shape[0]) != support.query_count:
            raise ValueError("query count does not match cached surface support")
        query_faces = self.faces[query_face_ids.long()]
        query_vertices = self.vertices[query_faces]
        query_to_vertex = torch.linalg.norm(
            query_points[:, None, :] - query_vertices,
            dim=-1,
        )
        return (
            support.vertex_path_distances
            + query_to_vertex[:, None, :]
        ).amin(dim=-1)

    def source_support_radius(
        self,
        support: SurfaceSupport,
        weights: torch.Tensor,
        *,
        rank: int = 1,
    ) -> torch.Tensor:
        """Interpolate the intrinsic spacing between reliable source anchors."""

        if int(weights.shape[0]) != support.query_count:
            raise ValueError("weight query count does not match support")
        if int(weights.shape[1]) != support.neighbor_count:
            raise ValueError("weight neighbor count does not match support")
        source_distances = torch.as_tensor(
            self._source_neighbor_distances,
            device=weights.device,
            dtype=weights.dtype,
        )
        if int(source_distances.shape[1]) <= 1:
            source_radius = source_distances[:, 0].clamp_min(EPS)
        else:
            radius_rank = min(max(1, int(rank)), int(source_distances.shape[1]) - 1)
            source_radius = source_distances[:, radius_rank].clamp_min(EPS)
        return interpolate_physical(source_radius, support.indices, weights).clamp_min(EPS)


def interpolate_physical(
    source_values: torch.Tensor,
    support_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Arithmetic interpolation for physical scalar or color values."""

    if support_indices.numel() == 0:
        return source_values.new_empty((0,) + tuple(source_values.shape[1:]))
    if source_values.shape[0] <= int(support_indices.max().item()):
        raise ValueError("surface support contains an out-of-range source index")
    gathered = source_values[support_indices]
    if source_values.ndim == 1:
        return (gathered * weights).sum(dim=1)
    weight_shape = tuple(weights.shape) + (1,) * (gathered.ndim - weights.ndim)
    return (gathered * weights.reshape(weight_shape)).sum(dim=1)


def density_invariant_log_scalar_smoothness(
    positive_values: torch.Tensor,
    graph: SurfaceSourceGraph,
    reference_spacing: torch.Tensor | float,
) -> torch.Tensor:
    """Dirichlet-style smoothness independent of adaptive source density.

    The positive scalar is represented in log space, differentiated with
    intrinsic surface distances, and integrated with a local surface-area
    proxy. The fixed reference spacing preserves the original regularization
    scale when new sources are inserted.
    """

    if graph.edges.numel() == 0:
        return positive_values.sum() * 0.0
    values = positive_values.reshape(positive_values.shape[0], -1)
    if values.shape[1] != 1:
        raise ValueError("density-invariant log smoothness requires one scalar per source")
    if graph.source_area_weights.shape != (values.shape[0],):
        raise ValueError("surface source area weights do not match the scalar field")
    if graph.distances.shape != (graph.edges.shape[0],):
        raise ValueError("surface source distances do not match graph edges")

    src, dst = graph.edges[:, 0], graph.edges[:, 1]
    log_values = torch.log(values[:, 0].clamp_min(EPS))
    spacing = torch.as_tensor(
        reference_spacing,
        device=values.device,
        dtype=values.dtype,
    ).reshape(())
    if not bool(torch.isfinite(spacing).detach().cpu()) or float(spacing.detach().cpu()) <= 0.0:
        raise ValueError("reference spacing must be finite and positive")
    distance = graph.distances.to(device=values.device, dtype=values.dtype)
    distance = distance.clamp_min(spacing * 1.0e-6)
    edge_gradient_squared = ((log_values[src] - log_values[dst]) / distance).square()

    source_sum = torch.zeros_like(log_values)
    source_count = torch.zeros_like(log_values)
    source_sum.scatter_add_(0, src, edge_gradient_squared)
    source_count.scatter_add_(0, src, torch.ones_like(edge_gradient_squared))
    source_gradient_squared = source_sum / source_count.clamp_min(1.0)
    area = graph.source_area_weights.to(device=values.device, dtype=values.dtype)
    metric_mean = (source_gradient_squared * area).sum() / area.sum().clamp_min(EPS)
    return 0.25 * spacing.square() * metric_mean


@torch.no_grad()
def harmonic_inpaint_physical(
    values: torch.Tensor,
    points: torch.Tensor,
    reliable: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct an unreliable physical field on a topology-safe root graph.

    Reliable samples are fixed Dirichlet anchors. Every other value is the
    harmonic extension on the supplied surface graph, so folded or nearby
    disconnected mesh sheets cannot exchange values through ambient-space KNN.
    This is intended for one-time initialization, not the differentiable
    training path.
    """

    if values.shape[0] != points.shape[0] or reliable.reshape(-1).shape[0] != points.shape[0]:
        raise ValueError("surface inpaint inputs must have the same root count")
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("edges must have shape [E, 2]")

    root_count = int(points.shape[0])
    reliable_flat = reliable.reshape(-1).to(dtype=torch.bool)
    if root_count == 0:
        return values.clone()
    if not bool(reliable_flat.any()):
        raise RuntimeError("surface inpaint requires at least one reliable anchor")
    if bool(reliable_flat.all()):
        return values.clone()

    edges_np = edges.detach().cpu().numpy().astype(np.int64, copy=False)
    if edges_np.size == 0:
        raise RuntimeError("surface inpaint cannot reconstruct values without graph edges")
    src = edges_np[:, 0]
    dst = edges_np[:, 1]
    valid_edge = (src >= 0) & (src < root_count) & (dst >= 0) & (dst < root_count) & (src != dst)
    src = src[valid_edge]
    dst = dst[valid_edge]
    if src.size == 0:
        raise RuntimeError("surface inpaint graph contains no valid non-self edge")

    pair_lo = np.minimum(src, dst)
    pair_hi = np.maximum(src, dst)
    pairs = np.unique(np.stack([pair_lo, pair_hi], axis=-1), axis=0)
    points_np = points.detach().cpu().numpy().astype(np.float64, copy=False)
    distance = np.linalg.norm(points_np[pairs[:, 0]] - points_np[pairs[:, 1]], axis=-1)
    weight = 1.0 / np.maximum(distance, 1.0e-6) ** 2
    row = np.concatenate([pairs[:, 0], pairs[:, 1]])
    col = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.concatenate([weight, weight])
    adjacency = coo_matrix((data, (row, col)), shape=(root_count, root_count)).tocsr()
    laplacian = diags(np.asarray(adjacency.sum(axis=1)).reshape(-1)) - adjacency

    reliable_np = reliable_flat.detach().cpu().numpy().astype(bool, copy=False)
    unknown_np = ~reliable_np
    unknown_ids = np.flatnonzero(unknown_np)
    anchor_ids = np.flatnonzero(reliable_np)
    values_np = values.detach().cpu().numpy().astype(np.float64, copy=False)
    scalar_input = values_np.ndim == 1
    values_2d = values_np[:, None] if scalar_input else values_np.reshape(root_count, -1)
    system = laplacian[unknown_ids][:, unknown_ids].tocsc()
    rhs = -(laplacian[unknown_ids][:, anchor_ids] @ values_2d[anchor_ids])
    solved = np.asarray(spsolve(system, rhs), dtype=np.float64)
    if solved.ndim == 1:
        solved = solved[:, None]
    if solved.shape != (unknown_ids.size, values_2d.shape[1]) or not np.isfinite(solved).all():
        raise RuntimeError(
            "surface inpaint failed; every connected root component must contain a reliable anchor"
        )

    output = values_2d.copy()
    output[unknown_ids] = solved
    output = output[:, 0] if scalar_input else output.reshape(values_np.shape)
    return torch.as_tensor(output, device=values.device, dtype=values.dtype)


@torch.no_grad()
def reconstruct_surface_directions(
    directions: torch.Tensor,
    normals: torch.Tensor,
    points: torch.Tensor,
    confidence: torch.Tensor,
    edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repair isolated direction outliers with one topology-local consensus pass.

    Neighbor directions are parallel transported to each root before averaging.
    A confident observation is retained when it agrees with a coherent local
    field. An isolated disagreement in a coherent neighborhood loses reliability,
    while a genuinely multi-directional neighborhood stays data-driven. The
    operation is deliberately a single pass so initialization cannot iteratively
    flatten the groom field.
    """

    if directions.ndim != 2 or directions.shape[-1] != 3:
        raise ValueError("directions must have shape [N, 3]")
    if normals.shape != directions.shape or points.shape != directions.shape:
        raise ValueError("directions, normals, and points must have matching [N, 3] shapes")
    if confidence.reshape(-1).shape[0] != directions.shape[0]:
        raise ValueError("confidence must have one value per direction")
    if edges.ndim != 2 or edges.shape[-1] != 2:
        raise ValueError("edges must have shape [E, 2]")
    if edges.numel() == 0:
        raise RuntimeError("surface direction reconstruction requires graph edges")

    direction = F.normalize(directions, dim=-1, eps=EPS)
    normal = F.normalize(normals, dim=-1, eps=EPS)
    confidence_flat = confidence.reshape(-1).to(
        device=direction.device,
        dtype=direction.dtype,
    ).clamp(0.0, 1.0)
    if not bool((confidence_flat > 0.0).any()):
        raise RuntimeError("surface direction reconstruction requires confident anchors")

    edge_ids = edges.to(device=direction.device, dtype=torch.long)
    src, dst = edge_ids[:, 0], edge_ids[:, 1]
    root_count = int(direction.shape[0])
    if bool(((src < 0) | (src >= root_count) | (dst < 0) | (dst >= root_count)).any()):
        raise ValueError("surface direction graph contains an out-of-range root index")

    transported = parallel_transport_vectors(direction[dst], normal[dst], normal[src])
    distance = torch.linalg.norm(points[src] - points[dst], dim=-1).clamp_min(1.0e-6)
    weight = confidence_flat[dst] / distance.square()
    weighted_direction = transported * weight[:, None]
    vector_sum = torch.zeros_like(direction).scatter_add(
        0,
        src[:, None].expand_as(weighted_direction),
        weighted_direction,
    )
    weight_sum = torch.zeros_like(confidence_flat).scatter_add(0, src, weight)
    supported = weight_sum > EPS
    consensus = torch.where(
        supported[:, None],
        F.normalize(vector_sum, dim=-1, eps=EPS),
        direction,
    )
    concentration = torch.where(
        supported,
        vector_sum.norm(dim=-1) / weight_sum.clamp_min(EPS),
        torch.zeros_like(weight_sum),
    ).clamp(0.0, 1.0)

    disagreement = 1.0 - (direction * consensus).sum(dim=-1).clamp(-1.0, 1.0)
    dispersion = (1.0 - concentration).clamp_min(1.0e-4)
    agreement = torch.exp(-disagreement / dispersion)
    reliability = (confidence_flat * agreement).clamp(0.0, 1.0)
    reconstructed = F.normalize(
        reliability[:, None] * direction
        + (1.0 - reliability[:, None]) * consensus,
        dim=-1,
        eps=EPS,
    )
    reconstructed = torch.where(supported[:, None], reconstructed, direction)
    reliability = torch.where(supported, reliability, confidence_flat)
    return reconstructed, reliability, supported


def interpolate_directions(
    source_directions: torch.Tensor,
    source_normals: torch.Tensor,
    query_normals: torch.Tensor,
    support_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Parallel-transport root-to-tip directions before weighted averaging."""

    directions = source_directions[support_indices]
    normals = source_normals[support_indices]
    target_normals = query_normals[:, None, :].expand_as(normals)
    transported = parallel_transport_vectors(directions, normals, target_normals)
    return F.normalize(
        (transported * weights[..., None]).sum(dim=1),
        dim=-1,
        eps=EPS,
    )


def interpolate_periodic(
    source_angles: torch.Tensor,
    support_indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Interpolate periodic angles on the unit circle."""

    cosine = interpolate_physical(torch.cos(source_angles), support_indices, weights)
    sine = interpolate_physical(torch.sin(source_angles), support_indices, weights)
    return torch.atan2(sine, cosine)


def _face_adjacency(faces: np.ndarray) -> list[list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, face in enumerate(faces.tolist()):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces[(min(a, b), max(a, b))].append(face_id)
    adjacency: list[set[int]] = [set() for _ in range(int(faces.shape[0]))]
    for incident in edge_faces.values():
        for face_id in incident:
            adjacency[face_id].update(other for other in incident if other != face_id)
    return [sorted(values) for values in adjacency]


def build_local_surface_support(
    *,
    faces: np.ndarray | torch.Tensor,
    source_points: torch.Tensor,
    source_face_ids: torch.Tensor,
    query_points: torch.Tensor,
    query_face_ids: torch.Tensor,
    neighbor_count: int,
    max_rings: int | None = None,
) -> LocalSurfaceSupport:
    """Find dense-root child support through face rings, never global 3D KNN."""

    started = perf_counter()
    faces_np = np.asarray(
        faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor) else faces,
        dtype=np.int64,
    )
    source_face_np = source_face_ids.detach().cpu().numpy().astype(np.int64)
    query_face_np = query_face_ids.detach().cpu().numpy().astype(np.int64)
    adjacency = _face_adjacency(faces_np)
    roots_by_face: dict[int, list[int]] = defaultdict(list)
    for root_id, face_id in enumerate(source_face_np.tolist()):
        roots_by_face[int(face_id)].append(int(root_id))

    k = max(1, min(int(neighbor_count), int(source_points.shape[0])))
    all_indices: list[list[int]] = []
    rings_used: list[int] = []
    source_cpu = source_points.detach().cpu()
    query_cpu = query_points.detach().cpu()
    for query_id, start_face in enumerate(query_face_np.tolist()):
        queue = deque([(int(start_face), 0)])
        visited = {int(start_face)}
        candidates: list[int] = []
        used_ring = 0
        while queue:
            face_id, ring = queue.popleft()
            used_ring = max(used_ring, int(ring))
            candidates.extend(roots_by_face.get(face_id, ()))
            if len(candidates) >= k:
                continue
            if max_rings is not None and ring >= int(max_rings):
                continue
            for neighbor in adjacency[face_id]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, ring + 1))
        candidates = sorted(set(candidates))
        if not candidates:
            limit = "the connected mesh component" if max_rings is None else f"{max_rings} face rings"
            raise RuntimeError(
                f"no topology-local source root found within {limit} "
                f"for child query {query_id}"
            )
        candidate_tensor = torch.tensor(candidates, dtype=torch.long)
        distance = torch.linalg.norm(
            source_cpu[candidate_tensor] - query_cpu[query_id : query_id + 1],
            dim=-1,
        )
        chosen = candidate_tensor[torch.argsort(distance)[:k]].tolist()
        if len(chosen) < k:
            chosen.extend([chosen[0]] * (k - len(chosen)))
        all_indices.append(chosen)
        rings_used.append(used_ring)
    return LocalSurfaceSupport(
        indices=torch.tensor(
            all_indices,
            device=query_points.device,
            dtype=torch.long,
        ),
        report={
            "source_count": int(source_points.shape[0]),
            "query_count": int(query_points.shape[0]),
            "neighbor_count": int(k),
            "max_rings_used": int(max(rings_used, default=0)),
            "build_seconds": float(perf_counter() - started),
        },
    )


def local_surface_weights(
    query_points: torch.Tensor,
    source_points: torch.Tensor,
    support: LocalSurfaceSupport,
    *,
    power: float = 2.0,
) -> torch.Tensor:
    distances = torch.linalg.norm(
        query_points[:, None, :] - source_points[support.indices],
        dim=-1,
    )
    weights = distances.clamp_min(1.0e-6).pow(-float(power))
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(EPS)


def build_hierarchical_surface_edges(
    points: torch.Tensor,
    guide_support_indices: torch.Tensor,
    *,
    neighbor_count: int,
) -> torch.Tensor:
    """Build exact dense-root KNN edges inside guide-support neighborhoods.

    Each render root may only connect to roots whose primary guide belongs to
    its own guide support. Candidate generation and KNN selection stay on the
    input device; only the rare degenerate-support rows use the topology
    expansion below on CPU.
    """

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if guide_support_indices.ndim != 2 or guide_support_indices.shape[0] != points.shape[0]:
        raise ValueError("guide_support_indices must have shape [N, K]")
    root_count = int(points.shape[0])
    if root_count < 2 or int(neighbor_count) <= 0:
        return torch.empty((0, 2), device=points.device, dtype=torch.long)

    points_work = points.detach().to(dtype=torch.float32)
    support = guide_support_indices.detach().to(device=points.device, dtype=torch.long)
    primary = support[:, 0]
    k = min(int(neighbor_count), root_count - 1)
    support_width = int(support.shape[1])
    guide_count = int(torch.maximum(primary.max(), support.max()).item()) + 1

    bucket_order = torch.argsort(primary, stable=True)
    sorted_primary = primary[bucket_order]
    bucket_counts = torch.bincount(primary, minlength=guide_count)
    max_bucket_size = int(bucket_counts.max().item())
    bucket_offsets = torch.cumsum(bucket_counts, dim=0) - bucket_counts
    bucket_positions = torch.arange(root_count, device=points.device) - torch.repeat_interleave(
        bucket_offsets,
        bucket_counts,
    )
    buckets = torch.full(
        (guide_count, max_bucket_size),
        -1,
        device=points.device,
        dtype=torch.long,
    )
    buckets[sorted_primary, bucket_positions] = bucket_order

    sorted_support = torch.sort(support, dim=1).values
    duplicate_support = (sorted_support[:, 1:] == sorted_support[:, :-1]).any(dim=1)
    candidate_counts = bucket_counts[support].sum(dim=1) - 1
    fallback = duplicate_support | (candidate_counts < k)
    dst_out = torch.empty((root_count, k), device=points.device, dtype=torch.long)

    candidate_width = support_width * max_bucket_size
    # Bound temporary candidate coordinates independently of root count. This
    # changes only execution chunking, never graph semantics.
    chunk_size = max(1, min(root_count, 2_000_000 // max(candidate_width, 1)))
    tie_rows: list[torch.Tensor] = []
    if candidate_width >= k:
        for begin in range(0, root_count, chunk_size):
            end = min(begin + chunk_size, root_count)
            query_ids = torch.arange(begin, end, device=points.device)
            candidate_ids = buckets[support[begin:end]].reshape(end - begin, candidate_width)
            valid = candidate_ids >= 0
            safe_ids = candidate_ids.clamp_min(0)
            delta = points_work[begin:end, None, :] - points_work[safe_ids]
            distance_sq = (delta * delta).sum(dim=-1)
            distance_sq.masked_fill_(~valid | (candidate_ids == query_ids[:, None]), torch.inf)

            select_count = min(k + 1, candidate_width)
            selected_distance, selected_slots = torch.topk(
                distance_sq,
                k=select_count,
                dim=1,
                largest=False,
                sorted=True,
            )
            selected_ids = torch.gather(candidate_ids, 1, selected_slots)

            # Make equal-distance ordering identical to the reference contract:
            # distance first, then root ID.
            selected_ids_k = selected_ids[:, :k]
            selected_distance_k = selected_distance[:, :k]
            id_order = torch.argsort(selected_ids_k, dim=1, stable=True)
            selected_ids_k = torch.gather(selected_ids_k, 1, id_order)
            selected_distance_k = torch.gather(selected_distance_k, 1, id_order)
            distance_order = torch.argsort(selected_distance_k, dim=1, stable=True)
            dst_out[begin:end] = torch.gather(selected_ids_k, 1, distance_order)

            if select_count > k:
                boundary_tie = selected_distance[:, k - 1] == selected_distance[:, k]
                tie_rows.append(query_ids[boundary_tie])
    else:
        fallback.fill_(True)

    if tie_rows:
        fallback[torch.cat(tie_rows)] = True

    fallback_ids = torch.nonzero(fallback, as_tuple=False).reshape(-1)
    if fallback_ids.numel() > 0:
        points_np = np.ascontiguousarray(points_work.cpu().numpy(), dtype=np.float32)
        support_np = np.ascontiguousarray(support.cpu().numpy(), dtype=np.int64)
        primary_np = support_np[:, 0]
        bucket_order_np = np.argsort(primary_np, kind="stable")
        bucket_counts_np = np.bincount(primary_np, minlength=guide_count)
        bucket_offsets_np = np.concatenate([[0], np.cumsum(bucket_counts_np)])
        bucket_lookup = {
            int(guide_id): bucket_order_np[
                bucket_offsets_np[guide_id] : bucket_offsets_np[guide_id + 1]
            ]
            for guide_id in np.flatnonzero(bucket_counts_np).tolist()
        }
        fallback_dst = np.empty((int(fallback_ids.numel()), k), dtype=np.int64)
        for output_row, root_id in enumerate(fallback_ids.cpu().tolist()):
            candidates: set[int] = set()
            for guide_id in support_np[root_id].tolist():
                candidates.update(bucket_lookup.get(int(guide_id), ()))
            candidates.discard(root_id)
            if len(candidates) < k:
                # A newly inserted guide can temporarily own fewer than K render roots.
                # Expand through the same guide-support incidence graph instead of
                # duplicating neighbors, lowering K, or crossing disconnected topology.
                active_guides = np.unique(support_np[root_id])
                while len(candidates) < k:
                    overlap = np.isin(support_np, active_guides).any(axis=1)
                    expanded_ids = np.flatnonzero(overlap)
                    previous_count = len(candidates)
                    candidates.update(expanded_ids.tolist())
                    candidates.discard(root_id)
                    if len(candidates) >= k:
                        break
                    expanded_guides = np.unique(support_np[expanded_ids])
                    if expanded_guides.size == active_guides.size and np.array_equal(
                        expanded_guides,
                        active_guides,
                    ):
                        break
                    active_guides = expanded_guides
                    if len(candidates) == previous_count:
                        break
            if len(candidates) < k:
                raise RuntimeError(
                    f"render root {root_id} has only {len(candidates)} topology-valid "
                    f"neighbors for K={k}"
                )
            candidate_ids_np = np.asarray(sorted(candidates), dtype=np.int64)
            distance = np.linalg.norm(
                points_np[candidate_ids_np] - points_np[root_id : root_id + 1],
                axis=-1,
            )
            fallback_dst[output_row] = candidate_ids_np[
                np.argsort(distance, kind="stable")[:k]
            ]
        dst_out[fallback_ids] = torch.as_tensor(
            fallback_dst,
            device=points.device,
            dtype=torch.long,
        )

    if bool((dst_out < 0).any()) or bool((dst_out == torch.arange(root_count, device=points.device)[:, None]).any()):
        raise RuntimeError("hierarchical surface graph contains an invalid or self edge")

    src_out = torch.arange(root_count, device=points.device).repeat_interleave(k)
    return torch.stack([src_out, dst_out.reshape(-1)], dim=-1)

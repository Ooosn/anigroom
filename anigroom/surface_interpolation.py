"""Topology-safe interpolation for mesh-rooted grooming attributes.

Neighbor selection is discrete and rebuilt only when root topology changes.
Weights are continuous in the current query position, and attribute
combination is typed: physical scalars/colors use arithmetic interpolation,
while 3D directions are parallel transported before averaging.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import dijkstra

from anigroom.flow.direction_geometry import parallel_transport_vectors
from anigroom.flow.surface_graph import _augmented_surface_graph, _root_voronoi_graph


EPS = 1.0e-8


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
        self.report: dict[str, float | int] = {
            "source_count": int(source_points_np.shape[0]),
            "mesh_vertex_count": int(vertices_np.shape[0]),
            "mesh_face_count": int(faces_np.shape[0]),
            "neighbor_count": int(self.neighbor_count),
            "source_graph_edge_count": int(root_graph.nnz // 2),
            "build_seconds": float(perf_counter() - started),
        }

    def build_support(
        self,
        query_points: np.ndarray | torch.Tensor,
        query_face_ids: np.ndarray | torch.Tensor,
    ) -> SurfaceSupport:
        """Build fixed source IDs for query roots without cross-surface KNN."""

        started = perf_counter()
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
        query_faces = self._faces_np[query_face_ids_np]
        query_vertices = self._vertices_np[query_faces]
        query_vertex_distances = np.linalg.norm(
            query_points_np[:, None, :] - query_vertices,
            axis=-1,
        )

        for query_id in range(query_count):
            paths: dict[int, np.ndarray] = {}
            for vertex_slot, vertex_id in enumerate(query_faces[query_id].tolist()):
                seed = int(self._vertex_source[vertex_id])
                base = float(self._vertex_distance[vertex_id])
                candidate_ids = self._source_neighbors[seed]
                candidate_distances = self._source_neighbor_distances[seed]
                for candidate_id, source_distance in zip(
                    candidate_ids.tolist(),
                    candidate_distances.tolist(),
                ):
                    values = paths.get(int(candidate_id))
                    if values is None:
                        values = np.full((3,), np.inf, dtype=np.float64)
                        paths[int(candidate_id)] = values
                    values[vertex_slot] = min(
                        float(values[vertex_slot]),
                        base + float(source_distance),
                    )

            if not paths:
                raise RuntimeError(f"query {query_id} has no topology-valid interpolation source")
            candidate_ids = np.fromiter(paths.keys(), dtype=np.int64)
            candidate_paths = np.stack([paths[int(candidate_id)] for candidate_id in candidate_ids])
            score = np.min(
                candidate_paths + query_vertex_distances[query_id][None, :],
                axis=1,
            )
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
    """Build dense-root edges only inside topology-valid guide neighborhoods."""

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if guide_support_indices.ndim != 2 or guide_support_indices.shape[0] != points.shape[0]:
        raise ValueError("guide_support_indices must have shape [N, K]")
    root_count = int(points.shape[0])
    if root_count < 2 or int(neighbor_count) <= 0:
        return torch.empty((0, 2), device=points.device, dtype=torch.long)

    primary = guide_support_indices[:, 0].detach().cpu().numpy().astype(np.int64)
    support_np = guide_support_indices.detach().cpu().numpy().astype(np.int64)
    points_np = points.detach().cpu().numpy().astype(np.float32)
    buckets: dict[int, list[int]] = defaultdict(list)
    for root_id, guide_id in enumerate(primary.tolist()):
        buckets[int(guide_id)].append(int(root_id))

    k = min(int(neighbor_count), root_count - 1)
    src_out = np.repeat(np.arange(root_count, dtype=np.int64), k)
    dst_out = np.empty((root_count, k), dtype=np.int64)
    for root_id in range(root_count):
        candidates: set[int] = set()
        for guide_id in support_np[root_id].tolist():
            candidates.update(buckets.get(int(guide_id), ()))
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
        candidate_ids = np.asarray(sorted(candidates), dtype=np.int64)
        distance = np.linalg.norm(
            points_np[candidate_ids] - points_np[root_id : root_id + 1],
            axis=-1,
        )
        dst_out[root_id] = candidate_ids[np.argsort(distance, kind="stable")[:k]]
    return torch.as_tensor(
        np.stack([src_out, dst_out.reshape(-1)], axis=-1),
        device=points.device,
        dtype=torch.long,
    )

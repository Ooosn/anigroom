from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


EPS = 1.0e-8


@dataclass(frozen=True)
class SurfaceRootGraph:
    """Fixed-size intrinsic neighbors for roots attached to a triangle mesh."""

    indices: torch.Tensor
    distances: torch.Tensor
    report: dict[str, float | int]

    @property
    def root_count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def neighbor_count(self) -> int:
        return int(self.indices.shape[1])


def _validate_surface_inputs(
    vertices: np.ndarray,
    faces: np.ndarray,
    root_points: np.ndarray,
    root_face_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    root_points = np.asarray(root_points, dtype=np.float64)
    root_face_ids = np.asarray(root_face_ids, dtype=np.int64).reshape(-1)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be [V, 3], got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be triangular [F, 3], got {faces.shape}")
    if root_points.ndim != 2 or root_points.shape[1] != 3:
        raise ValueError(f"root_points must be [N, 3], got {root_points.shape}")
    if root_face_ids.shape[0] != root_points.shape[0]:
        raise ValueError("root_face_ids must have one entry per root")
    if root_points.shape[0] < 2:
        raise ValueError("surface graph requires at least two roots")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("faces contain out-of-range vertex ids")
    if np.any(root_face_ids < 0) or np.any(root_face_ids >= faces.shape[0]):
        raise ValueError("root_face_ids contain out-of-range face ids")
    k_eff = min(max(1, int(k)), int(root_points.shape[0]) - 1)
    return vertices, faces, root_points, root_face_ids, k_eff


def _mesh_edges(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    valid = np.isfinite(lengths) & (lengths > EPS)
    return edges[valid, 0], edges[valid, 1], lengths[valid]


def _augmented_surface_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
    root_points: np.ndarray,
    root_face_ids: np.ndarray,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    vertex_count = int(vertices.shape[0])
    root_count = int(root_points.shape[0])
    mesh_u, mesh_v, mesh_w = _mesh_edges(vertices, faces)

    root_nodes = vertex_count + np.arange(root_count, dtype=np.int64)
    root_vertices = faces[root_face_ids]
    root_u = np.repeat(root_nodes, 3)
    root_v = root_vertices.reshape(-1)
    root_w = np.linalg.norm(
        np.repeat(root_points, 3, axis=0) - vertices[root_v],
        axis=1,
    )
    valid_root_edges = np.isfinite(root_w)
    root_u = root_u[valid_root_edges]
    root_v = root_v[valid_root_edges]
    root_w = root_w[valid_root_edges]

    edge_u = np.concatenate([mesh_u, root_u]).astype(np.int64, copy=False)
    edge_v = np.concatenate([mesh_v, root_v]).astype(np.int64, copy=False)
    edge_w = np.concatenate([mesh_w, root_w]).astype(np.float64, copy=False)
    rows = np.concatenate([edge_u, edge_v])
    cols = np.concatenate([edge_v, edge_u])
    values = np.concatenate([edge_w, edge_w])
    node_count = vertex_count + root_count
    graph = coo_matrix((values, (rows, cols)), shape=(node_count, node_count)).tocsr()
    return graph, root_nodes, edge_u, edge_v


def _root_voronoi_graph(
    graph: csr_matrix,
    root_nodes: np.ndarray,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
) -> csr_matrix:
    nearest_distance, _, nearest_source = dijkstra(
        graph,
        directed=False,
        indices=root_nodes,
        min_only=True,
        return_predecessors=True,
    )
    vertex_count = int(graph.shape[0] - root_nodes.shape[0])
    source_root = np.asarray(nearest_source, dtype=np.int64) - vertex_count
    if np.any(source_root < 0) or np.any(source_root >= root_nodes.shape[0]):
        raise RuntimeError("mesh Voronoi assignment returned an invalid root source")

    edge_weight_lookup: dict[tuple[int, int], float] = {}
    for u, v in zip(edge_u.tolist(), edge_v.tolist()):
        source_u = int(source_root[u])
        source_v = int(source_root[v])
        if source_u == source_v:
            continue
        direct = float(graph[u, v])
        candidate = float(nearest_distance[u] + direct + nearest_distance[v])
        if not np.isfinite(candidate):
            continue
        pair = (source_u, source_v) if source_u < source_v else (source_v, source_u)
        previous = edge_weight_lookup.get(pair)
        if previous is None or candidate < previous:
            edge_weight_lookup[pair] = candidate

    if not edge_weight_lookup:
        raise RuntimeError("surface-root Voronoi graph has no adjacency edges")
    pairs = np.asarray(list(edge_weight_lookup.keys()), dtype=np.int64)
    weights = np.asarray(list(edge_weight_lookup.values()), dtype=np.float64)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    values = np.concatenate([weights, weights])
    count = int(root_nodes.shape[0])
    return coo_matrix((values, (rows, cols)), shape=(count, count)).tocsr()


def _nearest_root_neighbors(root_graph: csr_matrix, k: int, *, chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    root_count = int(root_graph.shape[0])
    all_indices = np.empty((root_count, int(k)), dtype=np.int64)
    all_distances = np.empty((root_count, int(k)), dtype=np.float32)
    for begin in range(0, root_count, int(chunk_size)):
        end = min(begin + int(chunk_size), root_count)
        source_ids = np.arange(begin, end, dtype=np.int64)
        distances = np.asarray(
            dijkstra(root_graph, directed=False, indices=source_ids),
            dtype=np.float64,
        )
        distances[np.arange(end - begin), source_ids] = np.inf
        candidate_ids = np.argpartition(distances, kth=int(k) - 1, axis=1)[:, : int(k)]
        candidate_distances = np.take_along_axis(distances, candidate_ids, axis=1)
        order = np.argsort(candidate_distances, axis=1)
        candidate_ids = np.take_along_axis(candidate_ids, order, axis=1)
        candidate_distances = np.take_along_axis(candidate_distances, order, axis=1)
        if not np.isfinite(candidate_distances).all():
            raise RuntimeError(
                "at least one connected mesh component contains too few roots for the requested surface K"
            )
        all_indices[begin:end] = candidate_ids
        all_distances[begin:end] = candidate_distances.astype(np.float32)
    return all_indices, all_distances


def build_surface_root_graph(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    root_points: np.ndarray,
    root_face_ids: np.ndarray,
    k: int,
    device: torch.device | str,
) -> SurfaceRootGraph:
    """Approximate intrinsic root KNN through the mesh, without Euclidean shortcuts."""

    started = perf_counter()
    vertices, faces, root_points, root_face_ids, k_eff = _validate_surface_inputs(
        vertices,
        faces,
        root_points,
        root_face_ids,
        k,
    )
    graph, root_nodes, edge_u, edge_v = _augmented_surface_graph(
        vertices,
        faces,
        root_points,
        root_face_ids,
    )
    root_graph = _root_voronoi_graph(graph, root_nodes, edge_u, edge_v)
    component_count, labels = connected_components(root_graph, directed=False)
    component_sizes = np.bincount(labels, minlength=component_count)
    if int(component_sizes.min()) <= int(k_eff):
        raise RuntimeError(
            f"surface-root graph component is too small for K={k_eff}: "
            f"minimum component size is {int(component_sizes.min())}"
        )
    indices, distances = _nearest_root_neighbors(root_graph, k_eff)
    degree = np.diff(root_graph.indptr)
    report: dict[str, float | int] = {
        "root_count": int(root_points.shape[0]),
        "neighbor_count": int(k_eff),
        "mesh_vertex_count": int(vertices.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
        "root_graph_edge_count": int(root_graph.nnz // 2),
        "root_graph_degree_mean": float(degree.mean()),
        "root_graph_degree_min": int(degree.min()),
        "root_graph_degree_max": int(degree.max()),
        "connected_components": int(component_count),
        "runtime_seconds": float(perf_counter() - started),
    }
    target_device = torch.device(device)
    return SurfaceRootGraph(
        indices=torch.from_numpy(indices).to(device=target_device, dtype=torch.long),
        distances=torch.from_numpy(distances).to(device=target_device, dtype=torch.float32),
        report=report,
    )

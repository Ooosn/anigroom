"""R084 Phase B1 offline topology-cover data algebra.

This module validates caller-supplied topology data, evaluates the continuous
piecewise-linear topology-distance proxy, chooses fixed patch radii, and builds
exact sparse vertex/face/query cover artifacts.  The proxy is not an exact
geodesic distance; no ambient-distance inference or top-K truncation is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class TopologyCoverError(RuntimeError):
    """Base error for invalid R084 topology-cover data or construction."""


class ZeroMassBoundaryError(TopologyCoverError):
    """A patch has no finite distinct zero-mass radius boundary."""


class PatchSelfMembershipError(TopologyCoverError):
    """A selected patch radius does not include its own guide."""


@dataclass(frozen=True)
class TopologyCoverInputs:
    guide_distances: np.ndarray
    vertex_seed_guide_ids: np.ndarray
    vertex_nearest_distances: np.ndarray
    faces: np.ndarray
    guide_face_ids: np.ndarray
    guide_barycentric: np.ndarray
    component_labels: np.ndarray
    report: dict[str, Any]

    @property
    def guide_count(self) -> int:
        return int(self.guide_distances.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertex_seed_guide_ids.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True)
class PatchGuideDistanceMatrix:
    values: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class PatchNodeCover:
    radii: np.ndarray
    node_distances: csr_matrix
    report: dict[str, Any]


@dataclass(frozen=True)
class VertexPatchCover:
    active_distances: csr_matrix
    patch_radii: np.ndarray
    patch_node_counts: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class FacePatchCover:
    candidate_counts: csr_matrix
    patch_radii: np.ndarray
    patch_node_counts: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class RaggedQueryTopologyDistances:
    indptr: np.ndarray
    patch_ids: np.ndarray
    distances: np.ndarray
    radii: np.ndarray
    patch_node_counts: np.ndarray
    report: dict[str, Any]


def _float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        raise TypeError(f"{name} must be a floating-point array")
    return np.ascontiguousarray(array, dtype=np.float64)


def _integer_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array")
    return np.ascontiguousarray(array, dtype=np.int64)


def _positive_chunk_size(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


def _array_bytes(*arrays: np.ndarray) -> int:
    return int(sum(array.nbytes for array in arrays))


def _csr_bytes(matrix: csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _validate_barycentric(
    barycentric: Any,
    expected_count: int,
    name: str,
    tolerance: float,
) -> np.ndarray:
    bary = _float_array(barycentric, name)
    if bary.shape != (expected_count, 3):
        raise ValueError(f"{name} must have shape [{expected_count}, 3]")
    if not np.isfinite(bary).all():
        raise ValueError(f"{name} must be finite")
    if np.any(bary < 0.0) or np.any(bary > 1.0):
        raise ValueError(f"{name} entries must lie in [0, 1]")
    sum_error = np.abs(bary.sum(axis=1) - 1.0)
    if np.any(sum_error > float(tolerance)):
        raise ValueError(f"{name} rows must sum to one within tolerance")
    return bary


def validate_topology_cover_inputs(
    guide_distances: Any,
    vertex_seed_guide_ids: Any,
    vertex_nearest_distances: Any,
    faces: Any,
    guide_face_ids: Any,
    guide_barycentric: Any,
    *,
    symmetry_tolerance: float = 1.0e-10,
    diagonal_tolerance: float = 1.0e-10,
    barycentric_tolerance: float = 1.0e-6,
) -> TopologyCoverInputs:
    """Strictly validate the Phase-B1 shortest-path and mesh attachment data."""

    for value, name in (
        (symmetry_tolerance, "symmetry_tolerance"),
        (diagonal_tolerance, "diagonal_tolerance"),
        (barycentric_tolerance, "barycentric_tolerance"),
    ):
        if not np.isfinite(value) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    distances = _float_array(guide_distances, "guide_distances")
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("guide_distances must have shape [G, G]")
    guide_count = int(distances.shape[0])
    if guide_count <= 0:
        raise ValueError("guide_distances must contain at least one guide")
    if np.isnan(distances).any() or np.isneginf(distances).any():
        raise ValueError("guide_distances may contain only finite values or +inf")
    finite = np.isfinite(distances)
    if np.any(distances[finite] < 0.0):
        raise ValueError("finite guide_distances must be nonnegative")
    diagonal = np.diag(distances)
    if not np.isfinite(diagonal).all() or np.any(
        np.abs(diagonal) > float(diagonal_tolerance)
    ):
        raise ValueError("guide_distances diagonal must be finite zero within tolerance")
    off_diagonal = ~np.eye(guide_count, dtype=bool)
    finite_off_diagonal = finite & off_diagonal
    if np.any(
        distances[finite_off_diagonal] <= float(diagonal_tolerance)
    ):
        raise ValueError(
            "every finite off-diagonal guide distance must be strictly greater "
            "than diagonal_tolerance"
        )
    if not np.array_equal(finite, finite.T):
        raise ValueError("guide_distances finite/+inf pattern must be symmetric")
    if np.any(
        np.abs(distances[finite] - distances.T[finite])
        > float(symmetry_tolerance)
    ):
        raise ValueError("guide_distances must be symmetric within tolerance")

    component_count, component_labels = connected_components(
        csr_matrix(finite.astype(np.uint8)),
        directed=False,
        return_labels=True,
    )
    component_labels = np.asarray(component_labels, dtype=np.int64)
    same_component = component_labels[:, None] == component_labels[None, :]
    if np.any(same_component & ~finite):
        raise ValueError("guide_distances must be finite within every component")
    if np.any(~same_component & finite):
        raise ValueError("guide_distances must be +inf across components")
    nonfinite = ~finite
    if np.any(nonfinite & ~np.isposinf(distances)):
        raise ValueError("cross-component guide distances must be +inf")

    seed = _integer_array(vertex_seed_guide_ids, "vertex_seed_guide_ids").reshape(-1)
    delta = _float_array(vertex_nearest_distances, "vertex_nearest_distances").reshape(-1)
    if seed.shape != delta.shape or seed.size <= 0:
        raise ValueError("vertex seed IDs and nearest distances must have matching [V]")
    if np.any(seed < 0) or np.any(seed >= guide_count):
        raise ValueError("vertex_seed_guide_ids contains an out-of-range guide ID")
    if not np.isfinite(delta).all() or np.any(delta < 0.0):
        raise ValueError("vertex_nearest_distances must be finite and nonnegative")

    triangle_faces = _integer_array(faces, "faces")
    if triangle_faces.ndim != 2 or triangle_faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if triangle_faces.shape[0] <= 0:
        raise ValueError("faces must contain at least one triangle")
    if np.any(triangle_faces < 0) or np.any(triangle_faces >= seed.shape[0]):
        raise ValueError("faces contains an out-of-range vertex ID")
    if np.any(np.sort(triangle_faces, axis=1)[:, 1:] == np.sort(triangle_faces, axis=1)[:, :-1]):
        raise ValueError("every face must contain three distinct vertex IDs")

    guide_faces = _integer_array(guide_face_ids, "guide_face_ids").reshape(-1)
    if guide_faces.shape != (guide_count,):
        raise ValueError("guide_face_ids must have shape [G]")
    if np.any(guide_faces < 0) or np.any(guide_faces >= triangle_faces.shape[0]):
        raise ValueError("guide_face_ids contains an out-of-range face ID")
    guide_bary = _validate_barycentric(
        guide_barycentric,
        guide_count,
        "guide_barycentric",
        float(barycentric_tolerance),
    )

    face_seed_components = component_labels[seed[triangle_faces]]
    if np.any(face_seed_components != face_seed_components[:, :1]):
        raise ValueError("all three vertex seeds of a face must belong to one component")
    face_components = face_seed_components[:, 0]
    if np.any(component_labels != face_components[guide_faces]):
        raise ValueError("each guide must lie on a face in its guide component")

    component_sizes = np.bincount(
        component_labels,
        minlength=int(component_count),
    ).astype(np.int64)
    report: dict[str, Any] = {
        "guide_count": guide_count,
        "vertex_count": int(seed.shape[0]),
        "face_count": int(triangle_faces.shape[0]),
        "component_count": int(component_count),
        "component_sizes": component_sizes.tolist(),
        "finite_distance_count": int(finite.sum()),
        "infinite_distance_count": int((~finite).sum()),
        "input_memory_bytes": _array_bytes(
            distances,
            seed,
            delta,
            triangle_faces,
            guide_faces,
            guide_bary,
            component_labels,
        ),
    }
    return TopologyCoverInputs(
        guide_distances=distances,
        vertex_seed_guide_ids=seed,
        vertex_nearest_distances=delta,
        faces=triangle_faces,
        guide_face_ids=guide_faces,
        guide_barycentric=guide_bary,
        component_labels=component_labels,
        report=report,
    )


def safe_barycentric_pl_sum(
    vertex_values: Any,
    barycentric: Any,
) -> np.ndarray:
    """Safely evaluate PL values without ever multiplying zero by infinity.

    ``vertex_values`` has shape ``[P, S, 3]`` and ``barycentric`` has shape
    ``[S, 3]``.  Positive barycentric weights propagate ``+inf`` normally;
    exactly-zero weights contribute exactly zero.
    """

    values = _float_array(vertex_values, "vertex_values")
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("vertex_values must have shape [P, S, 3]")
    if np.isnan(values).any() or np.isneginf(values).any():
        raise ValueError("vertex_values may contain only finite values or +inf")
    if np.any(values[np.isfinite(values)] < 0.0):
        raise ValueError("finite vertex_values must be nonnegative")
    bary = _validate_barycentric(
        barycentric,
        int(values.shape[1]),
        "barycentric",
        1.0e-6,
    )
    result = np.zeros((values.shape[0], values.shape[1]), dtype=np.float64)
    for corner in range(3):
        positive_sites = bary[:, corner] > 0.0
        if np.any(positive_sites):
            result[:, positive_sites] += (
                values[:, positive_sites, corner]
                * bary[positive_sites, corner][None, :]
            )
    if np.isnan(result).any() or np.isneginf(result).any():
        raise TopologyCoverError("PL barycentric evaluation produced an invalid value")
    return result


def compute_patch_guide_site_distances(
    inputs: TopologyCoverInputs,
    *,
    patch_chunk_size: int = 128,
    guide_chunk_size: int = 2048,
) -> PatchGuideDistanceMatrix:
    """Compute the dense ``M[p,j]`` continuous PL proxy at all guide sites.

    This is barycentric interpolation of vertex values
    ``D[p, seed[v]] + delta[v]`` and is explicitly not an exact geodesic.
    """

    if not isinstance(inputs, TopologyCoverInputs):
        raise TypeError("inputs must be validated TopologyCoverInputs")
    patch_chunk = _positive_chunk_size(patch_chunk_size, "patch_chunk_size")
    guide_chunk = _positive_chunk_size(guide_chunk_size, "guide_chunk_size")
    guide_count = inputs.guide_count
    matrix = np.empty((guide_count, guide_count), dtype=np.float64)
    guide_vertices = inputs.faces[inputs.guide_face_ids]
    guide_vertex_seeds = inputs.vertex_seed_guide_ids[guide_vertices]
    guide_vertex_delta = inputs.vertex_nearest_distances[guide_vertices]

    for patch_begin in range(0, guide_count, patch_chunk):
        patch_end = min(patch_begin + patch_chunk, guide_count)
        patch_distances = inputs.guide_distances[patch_begin:patch_end]
        for guide_begin in range(0, guide_count, guide_chunk):
            guide_end = min(guide_begin + guide_chunk, guide_count)
            seed_block = guide_vertex_seeds[guide_begin:guide_end]
            vertex_values = (
                patch_distances[:, seed_block]
                + guide_vertex_delta[guide_begin:guide_end][None, :, :]
            )
            matrix[patch_begin:patch_end, guide_begin:guide_end] = (
                safe_barycentric_pl_sum(
                    vertex_values,
                    inputs.guide_barycentric[guide_begin:guide_end],
                )
            )

    if np.isnan(matrix).any() or np.isneginf(matrix).any():
        raise TopologyCoverError("patch-guide PL distance matrix contains invalid values")
    finite = np.isfinite(matrix)
    if np.any(matrix[finite] < 0.0):
        raise TopologyCoverError("patch-guide PL distances must be nonnegative")
    report: dict[str, Any] = {
        "patch_count": guide_count,
        "guide_site_count": guide_count,
        "finite_distance_count": int(finite.sum()),
        "infinite_distance_count": int((~finite).sum()),
        "finite_distance_min": float(matrix[finite].min()) if np.any(finite) else None,
        "finite_distance_max": float(matrix[finite].max()) if np.any(finite) else None,
        "matrix_memory_bytes": int(matrix.nbytes),
        "patch_chunk_size": patch_chunk,
        "guide_chunk_size": guide_chunk,
        "distance_semantics": "continuous_PL_topology_proxy_not_exact_geodesic",
    }
    return PatchGuideDistanceMatrix(values=matrix, report=report)


def _validate_patch_guide_matrix(value: Any) -> np.ndarray:
    matrix = value.values if isinstance(value, PatchGuideDistanceMatrix) else value
    matrix = _float_array(matrix, "patch_guide_distances")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] <= 0:
        raise ValueError("patch_guide_distances must have shape [G, G]")
    if np.isnan(matrix).any() or np.isneginf(matrix).any():
        raise ValueError("patch_guide_distances may contain only finite values or +inf")
    if np.any(matrix[np.isfinite(matrix)] < 0.0):
        raise ValueError("finite patch_guide_distances must be nonnegative")
    return matrix


def select_patch_radii_and_nodes(
    patch_guide_distances: PatchGuideDistanceMatrix | np.ndarray,
    minimum_active_node_count: int,
) -> PatchNodeCover:
    """Choose exact finite zero-mass boundaries and all nodes strictly inside."""

    matrix = _validate_patch_guide_matrix(patch_guide_distances)
    if isinstance(minimum_active_node_count, bool) or not isinstance(
        minimum_active_node_count,
        Integral,
    ):
        raise TypeError("minimum_active_node_count must be an integer")
    minimum_count = int(minimum_active_node_count)
    guide_count = int(matrix.shape[0])
    if minimum_count <= 0:
        raise ValueError("minimum_active_node_count must be positive")
    if minimum_count > guide_count:
        raise ValueError("minimum_active_node_count exceeds guide count")

    radii = np.empty((guide_count,), dtype=np.float64)
    row_indices: list[np.ndarray] = []
    row_data: list[np.ndarray] = []
    node_counts = np.empty((guide_count,), dtype=np.int64)
    kth_values = np.empty((guide_count,), dtype=np.float64)

    for patch_id in range(guide_count):
        row = matrix[patch_id]
        finite_values = row[np.isfinite(row)]
        if finite_values.size < minimum_count:
            raise ZeroMassBoundaryError(
                f"patch {patch_id} has only {finite_values.size} finite guide distances "
                f"for minimum K={minimum_count}"
            )
        kth_value = float(np.partition(finite_values, minimum_count - 1)[minimum_count - 1])
        boundary_candidates = finite_values[finite_values > kth_value]
        if boundary_candidates.size == 0:
            raise ZeroMassBoundaryError(
                f"patch {patch_id} has no finite distinct distance above Kth value "
                f"{kth_value:.17g}"
            )
        radius = float(boundary_candidates.min())
        active_ids = np.flatnonzero(row < radius).astype(np.int64, copy=False)
        if active_ids.size < minimum_count:
            raise TopologyCoverError(
                f"patch {patch_id} selected only {active_ids.size} nodes for K={minimum_count}"
            )
        if patch_id not in active_ids:
            raise PatchSelfMembershipError(
                f"patch {patch_id} does not include its own guide below radius {radius:.17g}"
            )
        if np.any(row[active_ids] >= radius):
            raise TopologyCoverError("internal active-node boundary violation")
        boundary_ids = np.flatnonzero(row == radius)
        if boundary_ids.size == 0:
            raise TopologyCoverError("selected radius is not an exact row distance boundary")
        radii[patch_id] = radius
        kth_values[patch_id] = kth_value
        node_counts[patch_id] = int(active_ids.size)
        row_indices.append(active_ids)
        row_data.append(row[active_ids].astype(np.float64, copy=False))

    indptr = np.zeros((guide_count + 1,), dtype=np.int64)
    indptr[1:] = np.cumsum(node_counts, dtype=np.int64)
    indices = np.concatenate(row_indices).astype(np.int64, copy=False)
    data = np.concatenate(row_data).astype(np.float64, copy=False)
    node_csr = csr_matrix(
        (data, indices, indptr),
        shape=(guide_count, guide_count),
    )
    node_csr.sort_indices()
    if not node_csr.has_sorted_indices or not node_csr.has_canonical_format:
        raise TopologyCoverError("patch-node CSR rows are not sorted unique canonical rows")

    report: dict[str, Any] = {
        "patch_count": guide_count,
        "minimum_active_node_count": minimum_count,
        "node_counts": node_counts.tolist(),
        "node_count_min": int(node_counts.min()),
        "node_count_max": int(node_counts.max()),
        "node_count_mean": float(node_counts.mean()),
        "radii": radii.tolist(),
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "kth_active_values": kth_values.tolist(),
        "csr_nnz": int(node_csr.nnz),
        "csr_memory_bytes": _csr_bytes(node_csr),
        "boundary_rule": "smallest_finite_distinct_value_strictly_above_Kth",
        "ties_below_boundary_included": True,
        "boundary_nodes_excluded": True,
    }
    return PatchNodeCover(
        radii=radii,
        node_distances=node_csr,
        report=report,
    )


def _validate_patch_node_cover(
    inputs: TopologyCoverInputs,
    patch_node_cover: PatchNodeCover,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(patch_node_cover, PatchNodeCover):
        raise TypeError("patch_node_cover must be a PatchNodeCover")
    radii = _float_array(patch_node_cover.radii, "patch_node_cover.radii").reshape(-1)
    if radii.shape != (inputs.guide_count,):
        raise ValueError("patch_node_cover radii must have shape [G]")
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("patch_node_cover radii must be finite and positive")
    node_csr = patch_node_cover.node_distances
    if not isinstance(node_csr, csr_matrix):
        raise TypeError("patch_node_cover node_distances must be CSR")
    if node_csr.shape != (inputs.guide_count, inputs.guide_count):
        raise ValueError("patch_node_cover node_distances must have shape [G, G]")
    if not node_csr.has_sorted_indices or not node_csr.has_canonical_format:
        raise ValueError("patch_node_cover CSR rows must be sorted unique canonical rows")
    if np.any(node_csr.indices < 0) or np.any(node_csr.indices >= inputs.guide_count):
        raise ValueError("patch_node_cover contains an out-of-range patch-node ID")
    if not np.isfinite(node_csr.data).all() or np.any(node_csr.data < 0.0):
        raise ValueError("patch_node_cover distances must be finite and nonnegative")
    node_counts = np.diff(node_csr.indptr).astype(np.int64, copy=False)
    for patch_id in range(inputs.guide_count):
        begin, end = node_csr.indptr[patch_id : patch_id + 2]
        row_ids = node_csr.indices[begin:end]
        row_distances = node_csr.data[begin:end]
        if patch_id not in row_ids:
            raise PatchSelfMembershipError(
                f"patch_node_cover patch {patch_id} does not contain itself"
            )
        if np.any(row_distances >= radii[patch_id]):
            raise ValueError("patch_node_cover contains a node at or above its radius")
    return radii, node_counts


def build_vertex_patch_active_distances(
    inputs: TopologyCoverInputs,
    patch_node_cover: PatchNodeCover,
    *,
    vertex_chunk_size: int = 4096,
) -> VertexPatchCover:
    """Build exact vertex→patch active-distance CSR in bounded chunks."""

    if not isinstance(inputs, TopologyCoverInputs):
        raise TypeError("inputs must be validated TopologyCoverInputs")
    radii, node_counts = _validate_patch_node_cover(inputs, patch_node_cover)
    chunk_size = _positive_chunk_size(vertex_chunk_size, "vertex_chunk_size")
    vertex_count = inputs.vertex_count
    patch_count = inputs.guide_count
    active_counts = np.zeros((vertex_count,), dtype=np.int64)
    indices_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []

    for begin in range(0, vertex_count, chunk_size):
        end = min(begin + chunk_size, vertex_count)
        seeds = inputs.vertex_seed_guide_ids[begin:end]
        distances = (
            inputs.guide_distances[:, seeds].T
            + inputs.vertex_nearest_distances[begin:end, None]
        )
        active = distances < radii[None, :]
        local_rows, patch_ids = np.nonzero(active)
        active_counts[begin:end] = np.bincount(
            local_rows,
            minlength=end - begin,
        )
        indices_parts.append(patch_ids.astype(np.int64, copy=False))
        distance_parts.append(distances[local_rows, patch_ids].astype(np.float64, copy=False))

    indptr = np.zeros((vertex_count + 1,), dtype=np.int64)
    indptr[1:] = np.cumsum(active_counts, dtype=np.int64)
    indices = (
        np.concatenate(indices_parts).astype(np.int64, copy=False)
        if indices_parts
        else np.empty((0,), dtype=np.int64)
    )
    data = (
        np.concatenate(distance_parts).astype(np.float64, copy=False)
        if distance_parts
        else np.empty((0,), dtype=np.float64)
    )
    active_csr = csr_matrix(
        (data, indices, indptr),
        shape=(vertex_count, patch_count),
    )
    active_csr.sort_indices()
    if not active_csr.has_sorted_indices or not active_csr.has_canonical_format:
        raise TopologyCoverError("vertex-patch CSR rows are not sorted unique canonical rows")
    uncovered = np.flatnonzero(active_counts == 0).astype(np.int64, copy=False)
    report: dict[str, Any] = {
        "vertex_count": vertex_count,
        "patch_count": patch_count,
        "active_incidence_count": int(active_csr.nnz),
        "active_patch_count_min": int(active_counts.min()),
        "active_patch_count_max": int(active_counts.max()),
        "active_patch_count_mean": float(active_counts.mean()),
        "uncovered_vertex_count": int(uncovered.size),
        "uncovered_vertex_ids": uncovered.tolist(),
        "csr_memory_bytes": _csr_bytes(active_csr),
        "vertex_chunk_size": chunk_size,
        "selection_rule": "D[p,seed[v]]+delta[v] < radius[p]",
        "top_k_truncation": False,
    }
    return VertexPatchCover(
        active_distances=active_csr,
        patch_radii=radii.copy(),
        patch_node_counts=node_counts.copy(),
        report=report,
    )


def _validate_vertex_patch_cover(
    inputs: TopologyCoverInputs,
    vertex_cover: VertexPatchCover,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    if not isinstance(vertex_cover, VertexPatchCover):
        raise TypeError("vertex_cover must be a VertexPatchCover")
    matrix = vertex_cover.active_distances
    if not isinstance(matrix, csr_matrix):
        raise TypeError("vertex_cover active_distances must be CSR")
    if matrix.shape != (inputs.vertex_count, inputs.guide_count):
        raise ValueError("vertex_cover CSR has the wrong shape")
    if not matrix.has_sorted_indices or not matrix.has_canonical_format:
        raise ValueError("vertex_cover CSR rows must be sorted unique canonical rows")
    radii = _float_array(vertex_cover.patch_radii, "vertex_cover.patch_radii").reshape(-1)
    counts = _integer_array(
        vertex_cover.patch_node_counts,
        "vertex_cover.patch_node_counts",
    ).reshape(-1)
    if radii.shape != (inputs.guide_count,) or counts.shape != (inputs.guide_count,):
        raise ValueError("vertex_cover patch bindings must have shape [G]")
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("vertex_cover patch radii must be finite and positive")
    if np.any(counts <= 0):
        raise ValueError("vertex_cover patch node counts must be positive")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0.0):
        raise ValueError("vertex_cover active distances must be finite and nonnegative")
    row_ids = np.repeat(np.arange(inputs.vertex_count), np.diff(matrix.indptr))
    if np.any(matrix.data >= radii[matrix.indices]):
        raise ValueError("vertex_cover contains a distance at or above its patch radius")
    del row_ids
    return matrix, radii, counts


def build_face_patch_candidate_counts(
    inputs: TopologyCoverInputs,
    vertex_cover: VertexPatchCover,
) -> FacePatchCover:
    """Build exact face→patch active-vertex counts by sparse multiplication."""

    if not isinstance(inputs, TopologyCoverInputs):
        raise TypeError("inputs must be validated TopologyCoverInputs")
    vertex_csr, radii, node_counts = _validate_vertex_patch_cover(
        inputs,
        vertex_cover,
    )
    face_rows = np.repeat(np.arange(inputs.face_count, dtype=np.int64), 3)
    face_columns = inputs.faces.reshape(-1)
    face_vertex = csr_matrix(
        (
            np.ones((face_columns.size,), dtype=np.uint8),
            (face_rows, face_columns),
        ),
        shape=(inputs.face_count, inputs.vertex_count),
    )
    active_structure = vertex_csr.copy()
    active_structure.data = np.ones((active_structure.nnz,), dtype=np.uint8)
    candidate_counts = (face_vertex @ active_structure).tocsr()
    candidate_counts.sum_duplicates()
    candidate_counts.sort_indices()
    candidate_counts.data = candidate_counts.data.astype(np.uint8, copy=False)
    if np.any(candidate_counts.data < 1) or np.any(candidate_counts.data > 3):
        raise TopologyCoverError("face-patch candidate counts must lie in [1, 3]")
    if not candidate_counts.has_sorted_indices or not candidate_counts.has_canonical_format:
        raise TopologyCoverError("face-patch CSR rows are not sorted unique canonical rows")

    per_face_candidates = np.diff(candidate_counts.indptr).astype(np.int64, copy=False)
    no_candidate = np.flatnonzero(per_face_candidates == 0).astype(np.int64, copy=False)
    strong = np.zeros((inputs.face_count,), dtype=bool)
    for face_id in range(inputs.face_count):
        begin, end = candidate_counts.indptr[face_id : face_id + 2]
        strong[face_id] = bool(np.any(candidate_counts.data[begin:end] == 3))
    lacking_strong = np.flatnonzero(~strong).astype(np.int64, copy=False)
    report: dict[str, Any] = {
        "face_count": inputs.face_count,
        "patch_count": inputs.guide_count,
        "candidate_incidence_count": int(candidate_counts.nnz),
        "candidate_patch_count_min": int(per_face_candidates.min()),
        "candidate_patch_count_max": int(per_face_candidates.max()),
        "candidate_patch_count_mean": float(per_face_candidates.mean()),
        "faces_without_candidate_count": int(no_candidate.size),
        "faces_without_candidate_ids": no_candidate.tolist(),
        "strong_full_face_cover_count": int(strong.sum()),
        "faces_lacking_strong_full_face_cover_count": int(lacking_strong.size),
        "faces_lacking_strong_full_face_cover_ids": lacking_strong.tolist(),
        "strong_cover_definition": "at_least_one_patch_active_at_all_three_vertices",
        "strong_cover_is_sufficient_for_all_barycentric_points": True,
        "csr_memory_bytes": _csr_bytes(candidate_counts),
    }
    return FacePatchCover(
        candidate_counts=candidate_counts,
        patch_radii=radii.copy(),
        patch_node_counts=node_counts.copy(),
        report=report,
    )


def _validate_face_patch_cover(
    inputs: TopologyCoverInputs,
    face_cover: FacePatchCover,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    if not isinstance(face_cover, FacePatchCover):
        raise TypeError("face_cover must be a FacePatchCover")
    matrix = face_cover.candidate_counts
    if not isinstance(matrix, csr_matrix):
        raise TypeError("face_cover candidate_counts must be CSR")
    if matrix.shape != (inputs.face_count, inputs.guide_count):
        raise ValueError("face_cover CSR has the wrong shape")
    if not matrix.has_sorted_indices or not matrix.has_canonical_format:
        raise ValueError("face_cover CSR rows must be sorted unique canonical rows")
    if matrix.data.dtype.kind not in "iu":
        raise TypeError("face_cover candidate counts must use an integer dtype")
    if np.any(matrix.data < 1) or np.any(matrix.data > 3):
        raise ValueError("face_cover candidate counts must lie in [1, 3]")
    radii = _float_array(face_cover.patch_radii, "face_cover.patch_radii").reshape(-1)
    node_counts = _integer_array(
        face_cover.patch_node_counts,
        "face_cover.patch_node_counts",
    ).reshape(-1)
    if radii.shape != (inputs.guide_count,) or node_counts.shape != (inputs.guide_count,):
        raise ValueError("face_cover patch bindings must have shape [G]")
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("face_cover patch radii must be finite and positive")
    if np.any(node_counts <= 0):
        raise ValueError("face_cover patch node counts must be positive")
    return matrix, radii, node_counts


def _direct_face_patch_counts(
    inputs: TopologyCoverInputs,
    radii: np.ndarray,
    face_id: int,
    patch_chunk_size: int,
) -> np.ndarray:
    vertices = inputs.faces[face_id]
    seeds = inputs.vertex_seed_guide_ids[vertices]
    delta = inputs.vertex_nearest_distances[vertices]
    counts = np.empty((inputs.guide_count,), dtype=np.uint8)
    for begin in range(0, inputs.guide_count, patch_chunk_size):
        end = min(begin + patch_chunk_size, inputs.guide_count)
        values = inputs.guide_distances[begin:end][:, seeds] + delta[None, :]
        counts[begin:end] = np.sum(
            values < radii[begin:end, None],
            axis=1,
            dtype=np.uint8,
        )
    return counts


def evaluate_query_topology_distances(
    inputs: TopologyCoverInputs,
    face_cover: FacePatchCover,
    query_face_ids: Any,
    query_barycentric: Any,
    *,
    query_chunk_size: int = 2048,
    completeness_patch_chunk_size: int = 256,
) -> RaggedQueryTopologyDistances:
    """Evaluate all face candidates for arbitrary legal barycentric queries."""

    if not isinstance(inputs, TopologyCoverInputs):
        raise TypeError("inputs must be validated TopologyCoverInputs")
    candidate_csr, radii, node_counts = _validate_face_patch_cover(inputs, face_cover)
    query_chunk = _positive_chunk_size(query_chunk_size, "query_chunk_size")
    patch_chunk = _positive_chunk_size(
        completeness_patch_chunk_size,
        "completeness_patch_chunk_size",
    )
    face_ids = _integer_array(query_face_ids, "query_face_ids").reshape(-1)
    if np.any(face_ids < 0) or np.any(face_ids >= inputs.face_count):
        raise ValueError("query_face_ids contains an out-of-range face ID")
    bary = _validate_barycentric(
        query_barycentric,
        int(face_ids.shape[0]),
        "query_barycentric",
        1.0e-6,
    )

    for face_id in np.unique(face_ids):
        direct_counts = _direct_face_patch_counts(
            inputs,
            radii,
            int(face_id),
            patch_chunk,
        )
        expected_ids = np.flatnonzero(direct_counts > 0)
        begin, end = candidate_csr.indptr[face_id : face_id + 2]
        actual_ids = candidate_csr.indices[begin:end]
        actual_counts = candidate_csr.data[begin:end]
        if not np.array_equal(actual_ids, expected_ids) or not np.array_equal(
            actual_counts,
            direct_counts[expected_ids],
        ):
            raise TopologyCoverError(
                f"face_cover candidates are incomplete or inexact for face {int(face_id)}"
            )

    query_count = int(face_ids.shape[0])
    candidate_counts = np.zeros((query_count,), dtype=np.int64)
    active_counts = np.zeros((query_count,), dtype=np.int64)
    patch_parts: list[np.ndarray] = []
    distance_parts: list[np.ndarray] = []
    radius_parts: list[np.ndarray] = []
    zero_weight_candidate_count = 0

    for query_begin in range(0, query_count, query_chunk):
        query_end = min(query_begin + query_chunk, query_count)
        for query_id in range(query_begin, query_end):
            face_id = int(face_ids[query_id])
            begin, end = candidate_csr.indptr[face_id : face_id + 2]
            patch_ids = candidate_csr.indices[begin:end].astype(np.int64, copy=False)
            candidate_counts[query_id] = int(patch_ids.size)
            if patch_ids.size == 0:
                patch_parts.append(np.empty((0,), dtype=np.int64))
                distance_parts.append(np.empty((0,), dtype=np.float64))
                radius_parts.append(np.empty((0,), dtype=np.float64))
                continue
            vertices = inputs.faces[face_id]
            seeds = inputs.vertex_seed_guide_ids[vertices]
            vertex_delta = inputs.vertex_nearest_distances[vertices]
            vertex_values = (
                inputs.guide_distances[patch_ids][:, seeds][:, None, :]
                + vertex_delta[None, None, :]
            )
            query_distances = safe_barycentric_pl_sum(
                vertex_values,
                bary[query_id : query_id + 1],
            )[:, 0]
            query_radii = radii[patch_ids]
            if not np.isfinite(query_distances).all():
                raise TopologyCoverError("candidate query distances must be finite")
            active = query_distances < query_radii
            active_counts[query_id] = int(active.sum())
            zero_weight_candidate_count += int((~active).sum())
            patch_parts.append(patch_ids.copy())
            distance_parts.append(query_distances)
            radius_parts.append(query_radii.copy())

    indptr = np.zeros((query_count + 1,), dtype=np.int64)
    indptr[1:] = np.cumsum(candidate_counts, dtype=np.int64)
    patch_ids = (
        np.concatenate(patch_parts).astype(np.int64, copy=False)
        if patch_parts
        else np.empty((0,), dtype=np.int64)
    )
    distances = (
        np.concatenate(distance_parts).astype(np.float64, copy=False)
        if distance_parts
        else np.empty((0,), dtype=np.float64)
    )
    query_radii = (
        np.concatenate(radius_parts).astype(np.float64, copy=False)
        if radius_parts
        else np.empty((0,), dtype=np.float64)
    )
    uncovered = np.flatnonzero(active_counts == 0).astype(np.int64, copy=False)
    report: dict[str, Any] = {
        "query_count": query_count,
        "candidate_entry_count": int(patch_ids.size),
        "candidate_count_min": int(candidate_counts.min()) if query_count else 0,
        "candidate_count_max": int(candidate_counts.max()) if query_count else 0,
        "candidate_count_mean": float(candidate_counts.mean()) if query_count else 0.0,
        "active_patch_count_min": int(active_counts.min()) if query_count else 0,
        "active_patch_count_max": int(active_counts.max()) if query_count else 0,
        "active_patch_count_mean": float(active_counts.mean()) if query_count else 0.0,
        "uncovered_query_count": int(uncovered.size),
        "uncovered_query_ids": uncovered.tolist(),
        "retained_zero_weight_candidate_count": int(zero_weight_candidate_count),
        "completeness_verified": True,
        "omitted_patch_can_have_positive_PU_weight": False,
        "completeness_reason": (
            "omitted_patch_has_all_three_vertex_distances_at_or_above_radius; "
            "convex_barycentric_interpolation_cannot_drop_below_radius"
        ),
        "ragged_memory_bytes": _array_bytes(
            indptr,
            patch_ids,
            distances,
            query_radii,
            node_counts,
        ),
        "query_chunk_size": query_chunk,
        "completeness_patch_chunk_size": patch_chunk,
    }
    return RaggedQueryTopologyDistances(
        indptr=indptr,
        patch_ids=patch_ids,
        distances=distances,
        radii=query_radii,
        patch_node_counts=node_counts.copy(),
        report=report,
    )


__all__ = [
    "FacePatchCover",
    "PatchGuideDistanceMatrix",
    "PatchNodeCover",
    "PatchSelfMembershipError",
    "RaggedQueryTopologyDistances",
    "TopologyCoverError",
    "TopologyCoverInputs",
    "VertexPatchCover",
    "ZeroMassBoundaryError",
    "build_face_patch_candidate_counts",
    "build_vertex_patch_active_distances",
    "compute_patch_guide_site_distances",
    "evaluate_query_topology_distances",
    "safe_barycentric_pl_sum",
    "select_patch_radii_and_nodes",
    "validate_topology_cover_inputs",
]

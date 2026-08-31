"""R084 Phase B1 offline topology-cover data algebra.

This module validates caller-supplied topology data, evaluates the continuous
piecewise-linear topology-distance proxy at guide sites, and chooses fixed
patch radii with exact guide-node CSR incidence.  The proxy is not an exact
geodesic distance.  Vertex/face incidence and arbitrary-query support belong
to Phase B2 and are intentionally absent here.
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


__all__ = [
    "PatchGuideDistanceMatrix",
    "PatchNodeCover",
    "PatchSelfMembershipError",
    "TopologyCoverError",
    "TopologyCoverInputs",
    "ZeroMassBoundaryError",
    "compute_patch_guide_site_distances",
    "safe_barycentric_pl_sum",
    "select_patch_radii_and_nodes",
    "validate_topology_cover_inputs",
]

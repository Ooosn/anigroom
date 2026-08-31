"""R084 Phase C1 actual-checkpoint topology-cover diagnostic.

This module is diagnostic infrastructure only.  It reconstructs the exact
primary-guide topology of one Stage-1 checkpoint, evaluates the predeclared
RBF-PU cover candidates, and writes numeric ``.npy``/JSON evidence.  It does
not train, render, mutate a checkpoint, or alter formal configuration.

The guide-to-guide matrix ``D`` is shortest-path distance on the existing root
Voronoi graph.  The patch-to-guide matrix ``M`` is a continuous piecewise-
linear proxy evaluated on the original mesh.  Neither quantity is described
as an exact geodesic distance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import psutil
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.flow.surface_graph import (  # noqa: E402
    _augmented_surface_graph,
    _root_voronoi_graph,
)
from anigroom.rbf_partition_of_unity import (  # noqa: E402
    RBFAlgebraError,
    build_augmented_system,
    evaluate_local_interpolant,
    local_cardinal_weights,
    solve_augmented_system,
    validate_augmented_system,
)
from anigroom.rbf_topology_cover import (  # noqa: E402
    FacePatchCover,
    PatchGuideDistanceMatrix,
    PatchNodeCover,
    TopologyCoverError,
    TopologyCoverInputs,
    VertexPatchCover,
    build_face_patch_candidate_counts,
    build_vertex_patch_active_distances,
    compute_patch_guide_site_distances,
    select_patch_radii_and_nodes,
    validate_topology_cover_inputs,
)


SCHEMA = "r084.rbf_partition_cover.actual_checkpoint.phase_c1.v1"
CANDIDATE_K_SEQUENCE = (8, 12, 16, 24, 32, 48, 64)
MAX_LOCAL_CONDITION_NUMBER = 1.0e12
MAX_LOCAL_ERROR = 1.0e-10
MAX_GUIDE_POINT_RECONSTRUCTION_ERROR = 1.0e-6
MAX_PATCH_NODE_COUNT = 128
MAX_SERIALIZED_STATE_BYTES = 4 * 1024**3
DEFAULT_DEVICE = "cuda"
DEFAULT_DIJKSTRA_SOURCE_CHUNK_SIZE = 128
DEFAULT_PATCH_CHUNK_SIZE = 128
DEFAULT_GUIDE_CHUNK_SIZE = 2048
DEFAULT_VERTEX_CHUNK_SIZE = 4096
GIT_TIMEOUT_SECONDS = 30.0
MAX_GIT_OUTPUT_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_COMMIT_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")

BASE_STATE_ARRAY_NAMES = (
    "guide_distances.npy",
    "patch_guide_distances.npy",
    "vertex_seed_guide_ids.npy",
    "vertex_nearest_distances.npy",
    "component_labels.npy",
)
SELECTED_STATE_ARRAY_NAMES = (
    "patch_radii.npy",
    "patch_node_indptr.npy",
    "patch_node_indices.npy",
    "patch_node_distances.npy",
    "vertex_active_indptr.npy",
    "vertex_active_indices.npy",
    "vertex_active_distances.npy",
    "face_candidate_indptr.npy",
    "face_candidate_indices.npy",
    "face_candidate_counts.npy",
)
STATE_ARRAY_NAMES = (*BASE_STATE_ARRAY_NAMES, *SELECTED_STATE_ARRAY_NAMES)
REPORT_NAME = "report.json"
MANIFEST_NAME = "sha256_manifest.json"
ARTIFACT_NAMES = (*STATE_ARRAY_NAMES, REPORT_NAME, MANIFEST_NAME)

HARD_GATE_KEYS = (
    "topology_validation",
    "component_coverage",
    "finite_zero_mass_boundaries",
    "patch_self_membership",
    "local_systems_full_rank",
    "local_condition_at_most_1e12",
    "local_node_self_error_at_most_1e-10",
    "local_constant_error_at_most_1e-10",
    "local_cardinal_sum_error_at_most_1e-10",
    "all_vertices_covered",
    "all_faces_have_candidates",
    "all_faces_have_strong_full_face_cover",
    "patch_node_count_at_most_128",
    "serialized_state_at_most_4_gib",
)


class DiagnosticError(RuntimeError):
    """The Phase C1 diagnostic contract could not be completed safely."""


@dataclass(frozen=True)
class CheckpointTopologyArrays:
    vertices: np.ndarray
    faces: np.ndarray
    stored_guide_points_local: np.ndarray
    guide_points_local: np.ndarray
    guide_face_ids: np.ndarray
    guide_barycentric: np.ndarray
    guide_point_reconstruction: dict[str, Any]


@dataclass(frozen=True)
class ActualTopologyData:
    inputs: TopologyCoverInputs
    patch_guide_distances: PatchGuideDistanceMatrix
    report: dict[str, Any]


@dataclass(frozen=True)
class CandidateArtifacts:
    patch_nodes: PatchNodeCover
    vertex_cover: VertexPatchCover
    face_cover: FacePatchCover
    serialized_state_bytes: int


def normalize_sha256(value: str, name: str = "SHA256") -> str:
    normalized = str(value).strip().lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be exactly 64 hexadecimal characters")
    return normalized


def normalize_git_commit(value: str) -> str:
    normalized = str(value).strip().lower()
    if GIT_COMMIT_RE.fullmatch(normalized) is None:
        raise ValueError("source commit must be exactly 40 or 64 hexadecimal characters")
    return normalized


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def array_identity(values: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(values))
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "bytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _bounded_git_command(
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float = GIT_TIMEOUT_SECONDS,
) -> str:
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("git timeout must be finite and positive")
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(cwd),
            shell=False,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DiagnosticError(
            f"source git command unavailable or timed out: git {' '.join(arguments)}"
        ) from error
    stdout = bytes(completed.stdout or b"")
    stderr = bytes(completed.stderr or b"")
    if len(stdout) + len(stderr) > MAX_GIT_OUTPUT_BYTES:
        raise DiagnosticError("source git command output exceeded the bounded limit")
    if completed.returncode != 0:
        detail = (stderr or stdout).decode("utf-8", errors="replace").strip()
        raise DiagnosticError(
            f"source git command failed: git {' '.join(arguments)}"
            + (f": {detail}" if detail else "")
        )
    return stdout.decode("utf-8", errors="strict")


def get_clean_source_git_identity(
    repository: str | os.PathLike[str] = PROJECT_ROOT,
    *,
    timeout_seconds: float = GIT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    root = Path(repository).expanduser().resolve()
    head_text = _bounded_git_command(
        ["rev-parse", "--verify", "HEAD"],
        cwd=root,
        timeout_seconds=timeout_seconds,
    ).strip()
    try:
        head = normalize_git_commit(head_text)
    except ValueError as error:
        raise DiagnosticError(f"source git HEAD is not a full commit: {head_text!r}") from error
    porcelain = _bounded_git_command(
        ["status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        timeout_seconds=timeout_seconds,
    )
    if porcelain:
        raise DiagnosticError(
            "source git worktree is dirty; clean exact HEAD is mandatory: "
            + porcelain.strip()
        )
    return {
        "repository": str(root),
        "head": head,
        "porcelain_status": "",
        "clean": True,
    }


def validate_source_identity(
    observed_commit: str,
    expected_commit: str | None,
) -> dict[str, Any]:
    if expected_commit is None:
        raise DiagnosticError("expected-source-commit is mandatory")
    expected = normalize_git_commit(expected_commit)
    observed = normalize_git_commit(observed_commit)
    report = {
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }
    if not report["passed"]:
        raise DiagnosticError(
            f"source HEAD mismatch: expected {expected}, observed {observed}"
        )
    return report


def validate_checkpoint_identity(
    observed_sha256: str,
    expected_sha256: str | None,
    checkpoint_data: Mapping[str, Any],
    expected_iteration: int | None,
) -> dict[str, Any]:
    if expected_sha256 is None:
        raise DiagnosticError("expected-checkpoint-sha256 is mandatory")
    if expected_iteration is None:
        raise DiagnosticError("expected-iteration is mandatory")
    expected_hash = normalize_sha256(expected_sha256, "expected checkpoint SHA256")
    observed_hash = normalize_sha256(observed_sha256, "observed checkpoint SHA256")
    if observed_hash != expected_hash:
        raise DiagnosticError(
            f"checkpoint SHA256 mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    if isinstance(expected_iteration, bool) or not isinstance(
        expected_iteration,
        (int, np.integer),
    ):
        raise DiagnosticError("expected-iteration must be a non-negative integer")
    expected_iter = int(expected_iteration)
    if expected_iter < 0:
        raise DiagnosticError("expected-iteration must be a non-negative integer")
    observed_value = checkpoint_data.get("iteration")
    if isinstance(observed_value, bool) or not isinstance(
        observed_value,
        (int, np.integer),
    ):
        raise DiagnosticError("checkpoint iteration is missing or not an integer")
    observed_iter = int(observed_value)
    if observed_iter < 0:
        raise DiagnosticError("checkpoint iteration must be non-negative")
    if observed_iter != expected_iter:
        raise DiagnosticError(
            f"checkpoint iteration mismatch: expected {expected_iter}, observed {observed_iter}"
        )
    return {
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "expected_iteration": expected_iter,
        "observed_iteration": observed_iter,
        "passed": True,
    }


def _path_file(value: str | os.PathLike[str], name: str) -> Path:
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{name} does not exist: {value}") from error
    if not path.is_file():
        raise ValueError(f"{name} is not a regular file: {path}")
    return path


def prepare_output_dir(
    value: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(value).expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output-dir is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"refusing nonempty output-dir without --overwrite: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _publish_temporary_file(
    temporary: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        os.replace(temporary, destination)
        return
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(f"refusing existing result artifact: {destination}") from error
    temporary.unlink()


def _atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {path.parent}")
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        _publish_temporary_file(temporary, path, overwrite=overwrite)
        temporary = Path()
    finally:
        if not handle.closed:
            handle.close()
        if temporary != Path():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def write_deterministic_json(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(Path(path), encoded, overwrite=overwrite)


def write_atomic_npy(
    path: str | os.PathLike[str],
    values: Any,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {destination.parent}")
    array = np.ascontiguousarray(np.asarray(values))
    if array.dtype.hasobject:
        raise TypeError("state arrays may not use object dtype")
    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        _publish_temporary_file(temporary, destination, overwrite=overwrite)
        temporary = Path()
    finally:
        if not handle.closed:
            handle.close()
        if temporary != Path():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return {
        "filename": destination.name,
        "bytes": int(destination.stat().st_size),
        "sha256": sha256_file(destination),
        "array": array_identity(array),
        "format": "npy_uncompressed_allow_pickle_false",
    }


def _csr_bytes(matrix: csr_matrix) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _sparse_identity(matrix: csr_matrix) -> dict[str, Any]:
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "memory_bytes": _csr_bytes(matrix),
        "indptr": array_identity(matrix.indptr),
        "indices": array_identity(matrix.indices),
        "data": array_identity(matrix.data),
        "sorted_indices": bool(matrix.has_sorted_indices),
        "canonical_format": bool(matrix.has_canonical_format),
    }


def _require_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be positive")
    return converted


def _as_float64_points(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] <= 0:
        raise DiagnosticError(f"{name} must have shape [N, 3] with N > 0")
    if not np.isfinite(array).all():
        raise DiagnosticError(f"{name} must be finite")
    return np.ascontiguousarray(array)


def _as_int64_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise DiagnosticError(f"{name} must be an integer array")
    array = np.ascontiguousarray(array, dtype=np.int64)
    if array.shape != shape:
        raise DiagnosticError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def extract_checkpoint_topology_arrays(model: Any) -> CheckpointTopologyArrays:
    if not hasattr(model, "guide_enabled") or not bool(model.guide_enabled()):
        raise DiagnosticError("checkpoint has no primary guides")
    with torch.no_grad():
        vertices = model.vertices.detach().to(dtype=torch.float64).cpu().numpy()
        faces = model.faces.detach().cpu().numpy()
        stored_guide_points = (
            model.guide_points_local.detach().to(dtype=torch.float64).cpu().numpy()
        )
        guide_face_ids = model.guide_face_ids.detach().cpu().numpy()
        guide_barycentric = (
            model.guide_barycentric.detach().to(dtype=torch.float64).cpu().numpy()
        )
    vertices = _as_float64_points(vertices, "vertices")
    stored_guide_points = _as_float64_points(
        stored_guide_points,
        "stored guide_points_local",
    )
    raw_faces = np.asarray(faces)
    if raw_faces.ndim != 2:
        raise DiagnosticError(f"faces must have shape [F, 3], got {raw_faces.shape}")
    faces = _as_int64_array(faces, "faces", (int(raw_faces.shape[0]), 3))
    if faces.shape[0] <= 0:
        raise DiagnosticError("faces must contain at least one triangle")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise DiagnosticError("faces contain an out-of-range vertex ID")
    if np.any(
        np.sort(faces, axis=1)[:, 1:] == np.sort(faces, axis=1)[:, :-1]
    ):
        raise DiagnosticError("every face must contain three distinct vertex IDs")
    guide_count = int(stored_guide_points.shape[0])
    guide_face_ids = _as_int64_array(
        guide_face_ids,
        "guide_face_ids",
        (guide_count,),
    )
    if np.any(guide_face_ids < 0) or np.any(guide_face_ids >= faces.shape[0]):
        raise DiagnosticError("guide_face_ids contains an out-of-range face ID")
    guide_barycentric = np.asarray(guide_barycentric, dtype=np.float64)
    if guide_barycentric.shape != (guide_count, 3):
        raise DiagnosticError(
            f"guide_barycentric must have shape {(guide_count, 3)}, "
            f"got {guide_barycentric.shape}"
        )
    if not np.isfinite(guide_barycentric).all():
        raise DiagnosticError("guide_barycentric must be finite")
    if np.any(guide_barycentric < 0.0) or np.any(guide_barycentric > 1.0):
        raise DiagnosticError("guide_barycentric entries must lie in [0, 1]")
    barycentric_sum_error = np.abs(guide_barycentric.sum(axis=1) - 1.0)
    if np.any(barycentric_sum_error > 1.0e-6):
        raise DiagnosticError("guide_barycentric rows must sum to one within 1e-6")
    guide_barycentric = np.ascontiguousarray(guide_barycentric)
    guide_triangles = vertices[faces[guide_face_ids]]
    canonical_guide_points = np.sum(
        guide_triangles * guide_barycentric[:, :, None],
        axis=1,
        dtype=np.float64,
    )
    canonical_guide_points = _as_float64_points(
        canonical_guide_points,
        "canonical guide points",
    )
    reconstruction_errors = np.linalg.norm(
        stored_guide_points - canonical_guide_points,
        axis=1,
    )
    if not np.isfinite(reconstruction_errors).all():
        raise DiagnosticError("guide point reconstruction errors must be finite")
    max_error = float(reconstruction_errors.max())
    p95_error = float(
        np.quantile(reconstruction_errors, 0.95, method="linear")
    )
    reconstruction_report = {
        "passed": bool(max_error <= MAX_GUIDE_POINT_RECONSTRUCTION_ERROR),
        "count": guide_count,
        "max": max_error,
        "p95": p95_error,
        "predeclared_max_error": MAX_GUIDE_POINT_RECONSTRUCTION_ERROR,
        "canonical_definition": (
            "float64_sum(vertices[faces[guide_face_ids]]*guide_barycentric)"
        ),
    }
    if not reconstruction_report["passed"]:
        raise DiagnosticError(
            "stored guide_points_local differs from float64 barycentric reconstruction: "
            f"max {max_error:.17g} exceeds "
            f"{MAX_GUIDE_POINT_RECONSTRUCTION_ERROR:.17g}"
        )
    return CheckpointTopologyArrays(
        vertices=vertices,
        faces=faces,
        stored_guide_points_local=stored_guide_points,
        guide_points_local=canonical_guide_points,
        guide_face_ids=guide_face_ids,
        guide_barycentric=guide_barycentric,
        guide_point_reconstruction=reconstruction_report,
    )


def build_actual_topology_data(
    arrays: CheckpointTopologyArrays,
    *,
    dijkstra_source_chunk_size: int = DEFAULT_DIJKSTRA_SOURCE_CHUNK_SIZE,
    patch_chunk_size: int = DEFAULT_PATCH_CHUNK_SIZE,
    guide_chunk_size: int = DEFAULT_GUIDE_CHUNK_SIZE,
) -> ActualTopologyData:
    """Build actual root-Voronoi ``D`` and original-mesh PL proxy ``M`` once."""

    dijkstra_chunk = _require_positive_integer(
        dijkstra_source_chunk_size,
        "dijkstra_source_chunk_size",
    )
    patch_chunk = _require_positive_integer(patch_chunk_size, "patch_chunk_size")
    guide_chunk = _require_positive_integer(guide_chunk_size, "guide_chunk_size")
    started = time.perf_counter()
    graph_started = time.perf_counter()
    graph, root_nodes, edge_u, edge_v = _augmented_surface_graph(
        arrays.vertices,
        arrays.faces,
        arrays.guide_points_local,
        arrays.guide_face_ids,
    )
    augmented_graph_seconds = time.perf_counter() - graph_started

    nearest_started = time.perf_counter()
    nearest_distance, _, nearest_source = dijkstra(
        graph,
        directed=False,
        indices=root_nodes,
        min_only=True,
        return_predecessors=True,
    )
    vertex_count = int(arrays.vertices.shape[0])
    guide_count = int(arrays.guide_points_local.shape[0])
    vertex_seed = (
        np.asarray(nearest_source[:vertex_count], dtype=np.int64) - vertex_count
    )
    vertex_delta = np.asarray(nearest_distance[:vertex_count], dtype=np.float64)
    valid_seed = (vertex_seed >= 0) & (vertex_seed < guide_count)
    finite_delta = np.isfinite(vertex_delta)
    component_coverage_passed = bool(valid_seed.all() and finite_delta.all())
    nearest_seconds = time.perf_counter() - nearest_started
    if not component_coverage_passed:
        bad_vertices = np.flatnonzero(~valid_seed | ~finite_delta)
        raise TopologyCoverError(
            "mesh component coverage failed; vertices without a finite primary-guide "
            f"source: {bad_vertices[:32].tolist()}"
        )

    root_graph_started = time.perf_counter()
    try:
        root_graph = _root_voronoi_graph(
            graph,
            root_nodes,
            edge_u,
            edge_v,
            nearest_distance=np.asarray(nearest_distance, dtype=np.float64),
            nearest_source=np.asarray(nearest_source, dtype=np.int64),
        )
    except (ValueError, TypeError, RuntimeError) as error:
        raise TopologyCoverError(
            f"root Voronoi graph construction failed: {error}"
        ) from error
    root_graph_seconds = time.perf_counter() - root_graph_started

    component_count, augmented_labels = connected_components(graph, directed=False)
    augmented_labels = np.asarray(augmented_labels, dtype=np.int64)
    root_components = np.unique(augmented_labels[root_nodes])
    vertex_components = np.unique(augmented_labels[:vertex_count])
    uncovered_components = np.setdiff1d(vertex_components, root_components)
    if uncovered_components.size:
        raise TopologyCoverError(
            "mesh connected components without a primary guide: "
            f"{uncovered_components.tolist()}"
        )

    distance_started = time.perf_counter()
    guide_distances = np.empty((guide_count, guide_count), dtype=np.float64)
    for begin in range(0, guide_count, dijkstra_chunk):
        end = min(begin + dijkstra_chunk, guide_count)
        source_ids = np.arange(begin, end, dtype=np.int64)
        rows = np.asarray(
            dijkstra(root_graph, directed=False, indices=source_ids),
            dtype=np.float64,
        )
        guide_distances[begin:end] = rows.reshape(end - begin, guide_count)
    distance_seconds = time.perf_counter() - distance_started

    validation_started = time.perf_counter()
    try:
        inputs = validate_topology_cover_inputs(
            guide_distances,
            vertex_seed,
            vertex_delta,
            arrays.faces,
            arrays.guide_face_ids,
            arrays.guide_barycentric,
        )
    except (ValueError, TypeError) as error:
        raise TopologyCoverError(f"topology input validation failed: {error}") from error
    validation_seconds = time.perf_counter() - validation_started
    matrix_started = time.perf_counter()
    patch_guide = compute_patch_guide_site_distances(
        inputs,
        patch_chunk_size=patch_chunk,
        guide_chunk_size=guide_chunk,
    )
    matrix_seconds = time.perf_counter() - matrix_started
    root_component_count, root_component_labels = connected_components(
        root_graph,
        directed=False,
    )
    root_component_sizes = np.bincount(
        np.asarray(root_component_labels, dtype=np.int64),
        minlength=int(root_component_count),
    )
    report = {
        "passed": True,
        "distance_semantics": {
            "D": "shortest_path_on_existing_root_voronoi_graph_not_exact_geodesic",
            "M": "continuous_original_mesh_PL_proxy_not_exact_geodesic",
        },
        "component_coverage": {
            "passed": component_coverage_passed,
            "augmented_graph_component_count": int(component_count),
            "root_graph_component_count": int(root_component_count),
            "root_graph_component_sizes": root_component_sizes.tolist(),
            "uncovered_mesh_component_ids": uncovered_components.tolist(),
            "all_vertex_seed_ids_valid": bool(valid_seed.all()),
            "all_vertex_nearest_distances_finite": bool(finite_delta.all()),
        },
        "graph": {
            "augmented_shape": [int(graph.shape[0]), int(graph.shape[1])],
            "augmented_nnz": int(graph.nnz),
            "augmented_memory_bytes": _csr_bytes(graph),
            "root_voronoi_shape": [int(root_graph.shape[0]), int(root_graph.shape[1])],
            "root_voronoi_nnz": int(root_graph.nnz),
            "root_voronoi_memory_bytes": _csr_bytes(root_graph),
        },
        "inputs": inputs.report,
        "patch_guide_distances": patch_guide.report,
        "base_state_array_identities": {
            "guide_distances.npy": array_identity(inputs.guide_distances),
            "patch_guide_distances.npy": array_identity(patch_guide.values),
            "vertex_seed_guide_ids.npy": array_identity(
                inputs.vertex_seed_guide_ids
            ),
            "vertex_nearest_distances.npy": array_identity(
                inputs.vertex_nearest_distances
            ),
            "component_labels.npy": array_identity(inputs.component_labels),
        },
        "base_serialized_state_bytes": int(
            sum(
                _npy_serialized_size(array)
                for array in (
                    inputs.guide_distances,
                    patch_guide.values,
                    inputs.vertex_seed_guide_ids,
                    inputs.vertex_nearest_distances,
                    inputs.component_labels,
                )
            )
        ),
        "chunks": {
            "dijkstra_source_chunk_size": dijkstra_chunk,
            "patch_chunk_size": patch_chunk,
            "guide_chunk_size": guide_chunk,
        },
        "timings_seconds": {
            "augmented_surface_graph": float(augmented_graph_seconds),
            "root_voronoi_graph_from_precomputed_assignment": float(
                root_graph_seconds
            ),
            "multi_source_vertex_assignment": float(nearest_seconds),
            "all_pairs_root_voronoi_shortest_paths": float(distance_seconds),
            "topology_validation": float(validation_seconds),
            "patch_guide_PL_matrix": float(matrix_seconds),
            "total": float(time.perf_counter() - started),
        },
    }
    return ActualTopologyData(
        inputs=inputs,
        patch_guide_distances=patch_guide,
        report=report,
    )


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "mean": None, "max": None}
    if not np.isfinite(array).all():
        raise DiagnosticError("numeric summary received a nonfinite value")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "mean": float(array.mean(dtype=np.float64)),
        "max": float(array.max()),
    }


def audit_local_rbf_systems(
    guide_points_local: np.ndarray,
    patch_nodes: PatchNodeCover,
    *,
    device: torch.device | str,
) -> dict[str, Any]:
    """Audit every selected local patch through the strict R084 Torch API."""

    points = _as_float64_points(guide_points_local, "guide_points_local")
    patch_count = int(points.shape[0])
    if patch_nodes.node_distances.shape != (patch_count, patch_count):
        raise ValueError("patch-node CSR shape does not match guide points")
    target_device = torch.device(device)
    conditions: list[float] = []
    self_errors: list[float] = []
    constant_errors: list[float] = []
    cardinal_errors: list[float] = []
    full_rank_count = 0
    started = time.perf_counter()
    failed_patch_id: int | None = None
    failure_type: str | None = None
    failure_message: str | None = None

    with torch.no_grad():
        for patch_id in range(patch_count):
            begin, end = patch_nodes.node_distances.indptr[patch_id : patch_id + 2]
            source_ids = patch_nodes.node_distances.indices[begin:end]
            sources = torch.as_tensor(
                points[source_ids],
                dtype=torch.float64,
                device=target_device,
            )
            radius = torch.tensor(
                float(patch_nodes.radii[patch_id]),
                dtype=torch.float64,
                device=target_device,
            )
            try:
                system = build_augmented_system(
                    sources,
                    radius,
                    max_condition_number=float(np.finfo(np.float64).max),
                )
                system_report = validate_augmented_system(
                    system,
                    max_condition_number=float(np.finfo(np.float64).max),
                )
                full_rank_count += int(system_report.full_rank)
                conditions.append(float(system_report.condition_number))
                if system_report.condition_number > MAX_LOCAL_CONDITION_NUMBER:
                    failed_patch_id = patch_id
                    failure_type = "IllConditionedRBFSystemError"
                    failure_message = (
                        f"condition number {system_report.condition_number:.17g} exceeds "
                        f"{MAX_LOCAL_CONDITION_NUMBER:.17g}"
                    )
                    break

                cardinal = local_cardinal_weights(
                    sources,
                    sources,
                    radius,
                    augmented_system=system,
                    max_condition_number=MAX_LOCAL_CONDITION_NUMBER,
                )
                identity = torch.eye(
                    int(source_ids.size),
                    dtype=torch.float64,
                    device=target_device,
                )
                self_error = float(torch.max(torch.abs(cardinal - identity)).item())
                cardinal_error = float(
                    torch.max(
                        torch.abs(
                            cardinal.sum(dim=1)
                            - torch.ones(
                                (int(source_ids.size),),
                                dtype=torch.float64,
                                device=target_device,
                            )
                        )
                    ).item()
                )
                constant_solution = solve_augmented_system(
                    system,
                    torch.ones(
                        (int(source_ids.size),),
                        dtype=torch.float64,
                        device=target_device,
                    ),
                    max_condition_number=MAX_LOCAL_CONDITION_NUMBER,
                )
                constant_values = evaluate_local_interpolant(
                    sources,
                    sources,
                    radius,
                    constant_solution,
                )
                constant_error = float(
                    torch.max(torch.abs(constant_values - 1.0)).item()
                )
                if not all(
                    math.isfinite(value)
                    for value in (self_error, constant_error, cardinal_error)
                ):
                    raise RBFAlgebraError("local error audit produced a nonfinite value")
                self_errors.append(self_error)
                constant_errors.append(constant_error)
                cardinal_errors.append(cardinal_error)
            except RBFAlgebraError as error:
                failed_patch_id = patch_id
                failure_type = type(error).__name__
                failure_message = str(error)
                break

    audited_count = len(conditions)
    max_condition = max(conditions) if conditions else None
    max_self = max(self_errors) if self_errors else None
    max_constant = max(constant_errors) if constant_errors else None
    max_cardinal = max(cardinal_errors) if cardinal_errors else None
    completed_all = (
        failed_patch_id is None
        and audited_count == patch_count
        and len(self_errors) == patch_count
    )
    passed = bool(
        completed_all
        and full_rank_count == patch_count
        and max_condition is not None
        and max_condition <= MAX_LOCAL_CONDITION_NUMBER
        and max_self is not None
        and max_self <= MAX_LOCAL_ERROR
        and max_constant is not None
        and max_constant <= MAX_LOCAL_ERROR
        and max_cardinal is not None
        and max_cardinal <= MAX_LOCAL_ERROR
    )
    return {
        "passed": passed,
        "patch_count": patch_count,
        "audited_patch_count": audited_count,
        "full_rank_patch_count": int(full_rank_count),
        "condition_number": _numeric_summary(conditions),
        "node_self_error": _numeric_summary(self_errors),
        "constant_error": _numeric_summary(constant_errors),
        "cardinal_sum_error": _numeric_summary(cardinal_errors),
        "condition_limit": MAX_LOCAL_CONDITION_NUMBER,
        "error_limit": MAX_LOCAL_ERROR,
        "failed_patch_id": failed_patch_id,
        "failure_type": failure_type,
        "failure_message": failure_message,
        "device": str(target_device),
        "dtype": "torch.float64",
        "seconds": float(time.perf_counter() - started),
        "core_functions": [
            "build_augmented_system",
            "validate_augmented_system",
            "local_cardinal_weights",
            "solve_augmented_system",
            "evaluate_local_interpolant",
        ],
    }


def derive_patch_cover_evidence(
    patch_guide_distances: PatchGuideDistanceMatrix | np.ndarray,
    patch_nodes: PatchNodeCover,
) -> dict[str, Any]:
    """Derive boundary and self-membership evidence independently from state."""

    matrix = np.asarray(
        patch_guide_distances.values
        if isinstance(patch_guide_distances, PatchGuideDistanceMatrix)
        else patch_guide_distances,
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("patch_guide_distances must have shape [G, G]")
    patch_count = int(matrix.shape[0])
    radii = np.asarray(patch_nodes.radii, dtype=np.float64)
    node_csr = patch_nodes.node_distances
    if radii.shape != (patch_count,):
        raise ValueError("patch radii must have shape [G]")
    if not isinstance(node_csr, csr_matrix):
        raise TypeError("patch node distances must be CSR")
    if node_csr.shape != (patch_count, patch_count):
        raise ValueError("patch node CSR must have shape [G, G]")
    if not node_csr.has_sorted_indices or not node_csr.has_canonical_format:
        raise ValueError("patch node CSR must have sorted unique canonical rows")

    nonfinite_radius_ids: list[int] = []
    missing_self_ids: list[int] = []
    retained_at_or_above_radius_ids: list[int] = []
    missing_boundary_ids: list[int] = []
    included_boundary_ids: list[int] = []
    inexact_node_row_ids: list[int] = []
    data_mismatch_ids: list[int] = []
    for patch_id in range(patch_count):
        radius = float(radii[patch_id])
        begin, end = node_csr.indptr[patch_id : patch_id + 2]
        node_ids = node_csr.indices[begin:end]
        node_values = node_csr.data[begin:end]
        row = matrix[patch_id]
        if not math.isfinite(radius) or radius <= 0.0:
            nonfinite_radius_ids.append(patch_id)
            missing_self_ids.append(patch_id)
            continue
        expected_ids = np.flatnonzero(row < radius)
        if not np.array_equal(node_ids, expected_ids):
            inexact_node_row_ids.append(patch_id)
        if node_ids.size and not np.array_equal(node_values, row[node_ids]):
            data_mismatch_ids.append(patch_id)
        if patch_id not in node_ids or not bool(row[patch_id] < radius):
            missing_self_ids.append(patch_id)
        if node_ids.size and bool(np.any(row[node_ids] >= radius)):
            retained_at_or_above_radius_ids.append(patch_id)
        boundary_ids = np.flatnonzero(row == radius)
        if boundary_ids.size == 0:
            missing_boundary_ids.append(patch_id)
        elif np.intersect1d(node_ids, boundary_ids, assume_unique=True).size:
            included_boundary_ids.append(patch_id)

    finite_zero_boundary_passed = not any(
        (
            nonfinite_radius_ids,
            retained_at_or_above_radius_ids,
            missing_boundary_ids,
            included_boundary_ids,
            inexact_node_row_ids,
            data_mismatch_ids,
        )
    )
    self_membership_passed = len(missing_self_ids) == 0

    def violations(values: list[int]) -> dict[str, Any]:
        return {
            "count": len(values),
            "first_patch_ids": values[:32],
        }

    return {
        "passed": bool(finite_zero_boundary_passed and self_membership_passed),
        "patch_count": patch_count,
        "finite_zero_mass_boundaries_passed": finite_zero_boundary_passed,
        "patch_self_membership_passed": self_membership_passed,
        "exact_node_csr_passed": not inexact_node_row_ids and not data_mismatch_ids,
        "nonfinite_or_nonpositive_radii": violations(nonfinite_radius_ids),
        "missing_self_membership": violations(missing_self_ids),
        "retained_nodes_at_or_above_radius": violations(
            retained_at_or_above_radius_ids
        ),
        "radii_without_exact_M_boundary": violations(missing_boundary_ids),
        "boundary_nodes_included_in_csr": violations(included_boundary_ids),
        "inexact_node_csr_rows": violations(inexact_node_row_ids),
        "csr_data_not_equal_to_M": violations(data_mismatch_ids),
        "derivation": (
            "M[p,p]<R[p], exact CSR IDs are M[p,j]<R[p], retained values are "
            "below R[p], and each R[p] equals an excluded M row value"
        ),
    }


def evaluate_predeclared_hard_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    required = (
        "topology_validation_passed",
        "component_coverage_passed",
        "finite_zero_mass_boundaries_passed",
        "patch_self_membership_passed",
        "local_systems_full_rank_passed",
        "max_local_condition_number",
        "max_local_node_self_error",
        "max_local_constant_error",
        "max_local_cardinal_sum_error",
        "uncovered_vertex_count",
        "faces_without_candidate_count",
        "faces_lacking_strong_full_face_cover_count",
        "patch_node_count_max",
        "serialized_state_bytes",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise DiagnosticError("hard-gate metrics are missing: " + ", ".join(missing))

    def flag(name: str) -> bool:
        value = metrics[name]
        return bool(value) if isinstance(value, (bool, np.bool_)) else False

    def finite_at_most(name: str, limit: float) -> bool:
        value = metrics[name]
        return isinstance(value, (int, float, np.integer, np.floating)) and bool(
            math.isfinite(float(value)) and float(value) <= limit
        )

    def integer_at_most(name: str, limit: int) -> bool:
        value = metrics[name]
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, np.integer))
            and 0 <= int(value) <= limit
        )

    gates = {
        "topology_validation": flag("topology_validation_passed"),
        "component_coverage": flag("component_coverage_passed"),
        "finite_zero_mass_boundaries": flag(
            "finite_zero_mass_boundaries_passed"
        ),
        "patch_self_membership": flag("patch_self_membership_passed"),
        "local_systems_full_rank": flag("local_systems_full_rank_passed"),
        "local_condition_at_most_1e12": finite_at_most(
            "max_local_condition_number",
            MAX_LOCAL_CONDITION_NUMBER,
        ),
        "local_node_self_error_at_most_1e-10": finite_at_most(
            "max_local_node_self_error",
            MAX_LOCAL_ERROR,
        ),
        "local_constant_error_at_most_1e-10": finite_at_most(
            "max_local_constant_error",
            MAX_LOCAL_ERROR,
        ),
        "local_cardinal_sum_error_at_most_1e-10": finite_at_most(
            "max_local_cardinal_sum_error",
            MAX_LOCAL_ERROR,
        ),
        "all_vertices_covered": integer_at_most("uncovered_vertex_count", 0),
        "all_faces_have_candidates": integer_at_most(
            "faces_without_candidate_count",
            0,
        ),
        "all_faces_have_strong_full_face_cover": integer_at_most(
            "faces_lacking_strong_full_face_cover_count",
            0,
        ),
        "patch_node_count_at_most_128": integer_at_most(
            "patch_node_count_max",
            MAX_PATCH_NODE_COUNT,
        ),
        "serialized_state_at_most_4_gib": integer_at_most(
            "serialized_state_bytes",
            MAX_SERIALIZED_STATE_BYTES,
        ),
    }
    return gates


def aggregate_hard_gates(gates: Mapping[str, Any]) -> bool:
    missing = [name for name in HARD_GATE_KEYS if name not in gates]
    if missing:
        raise DiagnosticError("hard gates are missing: " + ", ".join(missing))
    invalid = [
        name
        for name in HARD_GATE_KEYS
        if not isinstance(gates[name], (bool, np.bool_))
    ]
    if invalid:
        raise DiagnosticError("hard gates must be boolean: " + ", ".join(invalid))
    return bool(all(bool(gates[name]) for name in HARD_GATE_KEYS))


def _empty_candidate_metrics(
    *,
    topology_validation_passed: bool,
    component_coverage_passed: bool,
) -> dict[str, Any]:
    return {
        "topology_validation_passed": bool(topology_validation_passed),
        "component_coverage_passed": bool(component_coverage_passed),
        "finite_zero_mass_boundaries_passed": False,
        "patch_self_membership_passed": False,
        "local_systems_full_rank_passed": False,
        "max_local_condition_number": None,
        "max_local_node_self_error": None,
        "max_local_constant_error": None,
        "max_local_cardinal_sum_error": None,
        "uncovered_vertex_count": None,
        "faces_without_candidate_count": None,
        "faces_lacking_strong_full_face_cover_count": None,
        "patch_node_count_max": None,
        "serialized_state_bytes": None,
    }


def base_topology_state_arrays(
    topology: ActualTopologyData,
) -> dict[str, np.ndarray]:
    return {
        "guide_distances.npy": topology.inputs.guide_distances,
        "patch_guide_distances.npy": topology.patch_guide_distances.values,
        "vertex_seed_guide_ids.npy": topology.inputs.vertex_seed_guide_ids,
        "vertex_nearest_distances.npy": topology.inputs.vertex_nearest_distances,
        "component_labels.npy": topology.inputs.component_labels,
    }


def selected_cover_state_arrays(
    artifacts: CandidateArtifacts,
) -> dict[str, np.ndarray]:
    patch_csr = artifacts.patch_nodes.node_distances
    vertex_csr = artifacts.vertex_cover.active_distances
    face_csr = artifacts.face_cover.candidate_counts
    return {
        "patch_radii.npy": artifacts.patch_nodes.radii,
        "patch_node_indptr.npy": patch_csr.indptr,
        "patch_node_indices.npy": patch_csr.indices,
        "patch_node_distances.npy": patch_csr.data,
        "vertex_active_indptr.npy": vertex_csr.indptr,
        "vertex_active_indices.npy": vertex_csr.indices,
        "vertex_active_distances.npy": vertex_csr.data,
        "face_candidate_indptr.npy": face_csr.indptr,
        "face_candidate_indices.npy": face_csr.indices,
        "face_candidate_counts.npy": face_csr.data,
    }


def serialized_state_arrays(
    topology: ActualTopologyData,
    artifacts: CandidateArtifacts | None = None,
) -> dict[str, np.ndarray]:
    arrays = base_topology_state_arrays(topology)
    if artifacts is not None:
        arrays.update(selected_cover_state_arrays(artifacts))
    return arrays


def _npy_serialized_size(values: Any) -> int:
    array = np.ascontiguousarray(np.asarray(values))
    if array.dtype.hasobject:
        raise TypeError("serialized state arrays may not use object dtype")
    header_buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header_buffer,
        np.lib.format.header_data_from_array_1_0(array),
    )
    return int(header_buffer.tell() + array.nbytes)


def compute_serialized_state_bytes(
    topology: ActualTopologyData,
    patch_nodes: PatchNodeCover,
    vertex_cover: VertexPatchCover,
    face_cover: FacePatchCover,
) -> int:
    temporary = CandidateArtifacts(
        patch_nodes=patch_nodes,
        vertex_cover=vertex_cover,
        face_cover=face_cover,
        serialized_state_bytes=0,
    )
    return int(
        sum(
            _npy_serialized_size(array)
            for array in serialized_state_arrays(topology, temporary).values()
        )
    )


def _rejected_candidate(
    result: dict[str, Any],
    metrics: dict[str, Any],
    *,
    stage: str,
    error: BaseException | None = None,
    message: str | None = None,
) -> tuple[dict[str, Any], None]:
    gates = evaluate_predeclared_hard_gates(metrics)
    result.update(
        {
            "status": "rejected",
            "rejection_stage": stage,
            "rejection_type": type(error).__name__ if error is not None else "HardGateFailure",
            "rejection_message": str(error) if error is not None else str(message),
            "metrics": metrics,
            "hard_gates": gates,
            "failed_hard_gates": [name for name in HARD_GATE_KEYS if not gates[name]],
            "all_hard_gates_passed": False,
        }
    )
    return result, None


def evaluate_candidate_k(
    topology: ActualTopologyData,
    guide_points_local: np.ndarray,
    k: int,
    *,
    device: torch.device | str,
    vertex_chunk_size: int = DEFAULT_VERTEX_CHUNK_SIZE,
) -> tuple[dict[str, Any], CandidateArtifacts | None]:
    candidate_k = _require_positive_integer(k, "k")
    result: dict[str, Any] = {
        "k": candidate_k,
        "status": "evaluating",
        "timings_seconds": {},
    }
    metrics = _empty_candidate_metrics(
        topology_validation_passed=bool(topology.report.get("passed", False)),
        component_coverage_passed=bool(
            topology.report.get("component_coverage", {}).get("passed", False)
        ),
    )

    radius_started = time.perf_counter()
    try:
        patch_nodes = select_patch_radii_and_nodes(
            topology.patch_guide_distances,
            candidate_k,
        )
    except (TopologyCoverError, ValueError, TypeError) as error:
        result["timings_seconds"]["patch_radius_and_nodes"] = float(
            time.perf_counter() - radius_started
        )
        return _rejected_candidate(result, metrics, stage="patch_radius_and_nodes", error=error)
    result["timings_seconds"]["patch_radius_and_nodes"] = float(
        time.perf_counter() - radius_started
    )
    node_counts = np.diff(patch_nodes.node_distances.indptr).astype(np.int64, copy=False)
    cover_evidence = derive_patch_cover_evidence(
        topology.patch_guide_distances,
        patch_nodes,
    )
    result["patch_cover_evidence"] = cover_evidence
    metrics["finite_zero_mass_boundaries_passed"] = bool(
        cover_evidence["finite_zero_mass_boundaries_passed"]
    )
    metrics["patch_self_membership_passed"] = bool(
        cover_evidence["patch_self_membership_passed"]
    )
    metrics["patch_node_count_max"] = int(node_counts.max())
    result["patch_nodes"] = patch_nodes.report
    if not bool(cover_evidence["passed"]):
        return _rejected_candidate(
            result,
            metrics,
            stage="independent_patch_cover_evidence",
            message="derived zero-boundary or self-membership evidence failed",
        )
    if int(node_counts.max()) > MAX_PATCH_NODE_COUNT:
        return _rejected_candidate(
            result,
            metrics,
            stage="patch_node_count",
            message=(
                f"patch node count max {int(node_counts.max())} exceeds "
                f"{MAX_PATCH_NODE_COUNT}"
            ),
        )

    local_started = time.perf_counter()
    local_report = audit_local_rbf_systems(
        guide_points_local,
        patch_nodes,
        device=device,
    )
    result["timings_seconds"]["local_rbf_algebra"] = float(
        time.perf_counter() - local_started
    )
    result["local_rbf_algebra"] = local_report
    metrics["local_systems_full_rank_passed"] = bool(
        local_report["full_rank_patch_count"] == local_report["patch_count"]
    )
    metrics["max_local_condition_number"] = local_report["condition_number"]["max"]
    metrics["max_local_node_self_error"] = local_report["node_self_error"]["max"]
    metrics["max_local_constant_error"] = local_report["constant_error"]["max"]
    metrics["max_local_cardinal_sum_error"] = local_report["cardinal_sum_error"]["max"]
    local_gate_preview = evaluate_predeclared_hard_gates(metrics)
    local_gate_names = (
        "local_systems_full_rank",
        "local_condition_at_most_1e12",
        "local_node_self_error_at_most_1e-10",
        "local_constant_error_at_most_1e-10",
        "local_cardinal_sum_error_at_most_1e-10",
    )
    if not all(local_gate_preview[name] for name in local_gate_names):
        return _rejected_candidate(
            result,
            metrics,
            stage="local_rbf_algebra",
            message=local_report.get("failure_message") or "local RBF hard gate failed",
        )

    cover_started = time.perf_counter()
    try:
        vertex_cover = build_vertex_patch_active_distances(
            topology.inputs,
            patch_nodes,
            vertex_chunk_size=vertex_chunk_size,
        )
        face_cover = build_face_patch_candidate_counts(topology.inputs, vertex_cover)
    except (TopologyCoverError, ValueError, TypeError) as error:
        result["timings_seconds"]["vertex_and_face_cover"] = float(
            time.perf_counter() - cover_started
        )
        return _rejected_candidate(result, metrics, stage="vertex_and_face_cover", error=error)
    result["timings_seconds"]["vertex_and_face_cover"] = float(
        time.perf_counter() - cover_started
    )
    result["vertex_cover"] = vertex_cover.report
    result["face_cover"] = face_cover.report
    metrics["uncovered_vertex_count"] = int(
        vertex_cover.report["uncovered_vertex_count"]
    )
    metrics["faces_without_candidate_count"] = int(
        face_cover.report["faces_without_candidate_count"]
    )
    metrics["faces_lacking_strong_full_face_cover_count"] = int(
        face_cover.report["faces_lacking_strong_full_face_cover_count"]
    )
    state_bytes = compute_serialized_state_bytes(
        topology,
        patch_nodes,
        vertex_cover,
        face_cover,
    )
    metrics["serialized_state_bytes"] = state_bytes
    result["serialized_state"] = {
        "bytes": state_bytes,
        "limit_bytes": MAX_SERIALIZED_STATE_BYTES,
        "definition": (
            "exact_uncompressed_npy_file_bytes_for_base_D_M_seed_delta_component_"
            "labels_plus_selected_radii_and_CSR_arrays"
        ),
        "resident_memory_metric": False,
    }
    gates = evaluate_predeclared_hard_gates(metrics)
    passed = aggregate_hard_gates(gates)
    if not passed:
        return _rejected_candidate(
            result,
            metrics,
            stage="predeclared_hard_gates",
            message="one or more predeclared hard gates failed",
        )
    artifacts = CandidateArtifacts(
        patch_nodes=patch_nodes,
        vertex_cover=vertex_cover,
        face_cover=face_cover,
        serialized_state_bytes=state_bytes,
    )
    result.update(
        {
            "status": "passed",
            "rejection_stage": None,
            "rejection_type": None,
            "rejection_message": None,
            "metrics": metrics,
            "hard_gates": gates,
            "failed_hard_gates": [],
            "all_hard_gates_passed": True,
        }
    )
    return result, artifacts


def choose_first_passing_candidate(
    candidate_results: Sequence[Mapping[str, Any]],
    *,
    sequence: Sequence[int] = CANDIDATE_K_SEQUENCE,
) -> int | None:
    allowed = [int(value) for value in sequence]
    positions = {value: index for index, value in enumerate(allowed)}
    passing: list[int] = []
    previous = -1
    for result in candidate_results:
        k = int(result["k"])
        if k not in positions:
            raise DiagnosticError(f"candidate result contains unexpected K={k}")
        position = positions[k]
        if position <= previous:
            raise DiagnosticError("candidate results are not in predeclared order")
        previous = position
        if result.get("all_hard_gates_passed") is True:
            passing.append(k)
    return passing[0] if passing else None


def scan_candidate_ks(
    topology: ActualTopologyData,
    guide_points_local: np.ndarray,
    *,
    device: torch.device | str,
    sequence: Sequence[int] = CANDIDATE_K_SEQUENCE,
    evaluator: Callable[
        [ActualTopologyData, np.ndarray, int],
        tuple[dict[str, Any], CandidateArtifacts | None],
    ]
    | None = None,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], int | None, CandidateArtifacts | None]:
    ordered = tuple(_require_positive_integer(value, "candidate K") for value in sequence)
    if len(set(ordered)) != len(ordered):
        raise ValueError("candidate K sequence must be unique")
    evaluate = evaluator
    results: list[dict[str, Any]] = []
    selected_k: int | None = None
    selected_artifacts: CandidateArtifacts | None = None
    for index, candidate_k in enumerate(ordered):
        if selected_k is not None:
            results.append(
                {
                    "k": candidate_k,
                    "status": "not_evaluated_after_first_pass",
                    "all_hard_gates_passed": None,
                }
            )
            continue
        if progress:
            print(f"R084_C1_PROGRESS stage=candidate_start k={candidate_k}", flush=True)
        if evaluate is None:
            result, artifacts = evaluate_candidate_k(
                topology,
                guide_points_local,
                candidate_k,
                device=device,
            )
        else:
            try:
                result, artifacts = evaluate(topology, guide_points_local, candidate_k)
            except (TopologyCoverError, RBFAlgebraError, ValueError) as error:
                result = {
                    "k": candidate_k,
                    "status": "rejected",
                    "rejection_stage": "candidate_evaluator",
                    "rejection_type": type(error).__name__,
                    "rejection_message": str(error),
                    "all_hard_gates_passed": False,
                }
                artifacts = None
        if int(result.get("k", -1)) != candidate_k:
            raise DiagnosticError("candidate evaluator returned the wrong K identity")
        passed = result.get("all_hard_gates_passed")
        if not isinstance(passed, (bool, np.bool_)):
            raise DiagnosticError("evaluated candidate must report a boolean gate result")
        if bool(passed) != (artifacts is not None):
            raise DiagnosticError("candidate artifact presence disagrees with hard gates")
        results.append(result)
        if bool(passed):
            selected_k = candidate_k
            selected_artifacts = artifacts
        if progress:
            print(
                f"R084_C1_PROGRESS stage=candidate_done k={candidate_k} "
                f"passed={str(bool(passed)).lower()}",
                flush=True,
            )
    chosen = choose_first_passing_candidate(results, sequence=ordered)
    if chosen != selected_k:
        raise DiagnosticError("first-pass selection disagrees with candidate results")
    return results, selected_k, selected_artifacts


def _artifact_identity(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def save_diagnostic_outputs(
    output_dir: str | os.PathLike[str],
    report: dict[str, Any],
    *,
    topology: ActualTopologyData | None,
    artifacts: CandidateArtifacts | None,
    overwrite: bool,
) -> dict[str, Any]:
    output_root = Path(output_dir).resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output_root}")
    if artifacts is not None and topology is None:
        raise ValueError("selected artifacts require completed base topology")
    arrays = (
        serialized_state_arrays(topology, artifacts)
        if topology is not None
        else {}
    )
    expected_names = (
        STATE_ARRAY_NAMES
        if artifacts is not None
        else BASE_STATE_ARRAY_NAMES if topology is not None else ()
    )
    if tuple(arrays) != expected_names:
        raise DiagnosticError("serialized state array ordering/names do not match")

    stage_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.r084-c1-stage.",
            dir=output_root.parent,
        )
    )
    state_artifacts: dict[str, Any] = {}
    try:
        state_write_started = time.perf_counter()
        for name, array in arrays.items():
            state_artifacts[name] = write_atomic_npy(
                stage_root / name,
                array,
                overwrite=False,
            )
        timings = report.get("timings_seconds")
        if isinstance(timings, dict):
            timings["staged_state_array_write"] = float(
                time.perf_counter() - state_write_started
            )
        report["artifacts"] = state_artifacts
        write_deterministic_json(
            stage_root / REPORT_NAME,
            report,
            overwrite=False,
        )
        manifest_artifacts = {
            name: {
                "bytes": int(data["bytes"]),
                "sha256": str(data["sha256"]),
            }
            for name, data in state_artifacts.items()
        }
        manifest_artifacts[REPORT_NAME] = _artifact_identity(
            stage_root / REPORT_NAME
        )
        manifest = {
            "schema": SCHEMA,
            "algorithm": "sha256",
            "artifacts": manifest_artifacts,
        }
        write_deterministic_json(
            stage_root / MANIFEST_NAME,
            manifest,
            overwrite=False,
        )

        published_names = (*arrays.keys(), REPORT_NAME)
        for name in published_names:
            _publish_temporary_file(
                stage_root / name,
                output_root / name,
                overwrite=overwrite,
            )
        if overwrite:
            new_names = {*published_names, MANIFEST_NAME}
            for stale_name in ARTIFACT_NAMES:
                stale_path = output_root / stale_name
                if stale_name in new_names or not (
                    stale_path.exists() or stale_path.is_symlink()
                ):
                    continue
                if stale_path.is_dir() and not stale_path.is_symlink():
                    raise IsADirectoryError(
                        f"result artifact is a directory: {stale_path}"
                    )
                stale_path.unlink()
        _publish_temporary_file(
            stage_root / MANIFEST_NAME,
            output_root / MANIFEST_NAME,
            overwrite=overwrite,
        )
        return manifest
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _resolve_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise DiagnosticError(f"invalid device: {value}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise DiagnosticError(
                "CUDA was requested but is unavailable; Phase C1 has no CPU fallback"
            )
        try:
            torch.cuda.get_device_properties(device)
        except (AssertionError, RuntimeError) as error:
            raise DiagnosticError(f"requested CUDA device is unavailable: {device}") from error
    return device


def _rss_bytes() -> int:
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _cuda_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"requested": False, "device": str(device)}
    return {
        "requested": True,
        "device": str(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _load_stage1_checkpoint_model() -> Any:
    from tools.train_white_tiger_stage1 import (  # type: ignore[import-not-found]
        load_stage1_checkpoint_model,
    )

    return load_stage1_checkpoint_model


def _versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _source_file_identities() -> dict[str, Any]:
    paths = {
        "diagnostic": Path(__file__).resolve(),
        "rbf_partition_of_unity": PROJECT_ROOT / "anigroom" / "rbf_partition_of_unity.py",
        "rbf_topology_cover": PROJECT_ROOT / "anigroom" / "rbf_topology_cover.py",
        "surface_graph": PROJECT_ROOT / "anigroom" / "flow" / "surface_graph.py",
    }
    return {
        name: {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def _not_evaluated_candidates(reason: str) -> list[dict[str, Any]]:
    return [
        {
            "k": int(candidate_k),
            "status": "not_evaluated",
            "reason": reason,
            "all_hard_gates_passed": None,
        }
        for candidate_k in CANDIDATE_K_SEQUENCE
    ]


def run_checkpoint_partition_cover_diagnostic(
    *,
    checkpoint: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    expected_checkpoint_sha256: str,
    expected_iteration: int,
    expected_source_commit: str,
    device: str | torch.device = DEFAULT_DEVICE,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the exact, bounded Phase C1 actual-checkpoint diagnostic."""

    total_started = time.perf_counter()
    checkpoint_path = _path_file(checkpoint, "checkpoint")
    output_root = prepare_output_dir(output_dir, overwrite=overwrite)
    expected_hash = normalize_sha256(
        expected_checkpoint_sha256,
        "expected checkpoint SHA256",
    )
    expected_commit = normalize_git_commit(expected_source_commit)
    if isinstance(expected_iteration, bool) or not isinstance(
        expected_iteration,
        (int, np.integer),
    ):
        raise DiagnosticError("expected-iteration must be a non-negative integer")
    expected_iter = int(expected_iteration)
    if expected_iter < 0:
        raise DiagnosticError("expected-iteration must be a non-negative integer")

    timings: dict[str, float] = {}
    rss_stages: dict[str, int] = {"start": _rss_bytes()}
    git_started = time.perf_counter()
    git_identity = get_clean_source_git_identity(PROJECT_ROOT)
    source_identity = validate_source_identity(git_identity["head"], expected_commit)
    timings["source_git_identity"] = float(time.perf_counter() - git_started)

    checkpoint_hash_started = time.perf_counter()
    checkpoint_hash = sha256_file(checkpoint_path)
    timings["checkpoint_sha256"] = float(time.perf_counter() - checkpoint_hash_started)
    if checkpoint_hash != expected_hash:
        raise DiagnosticError(
            f"checkpoint SHA256 mismatch: expected {expected_hash}, observed {checkpoint_hash}"
        )

    target_device = _resolve_device(device)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)

    load_started = time.perf_counter()
    loader = _load_stage1_checkpoint_model()
    model, config, checkpoint_data = loader(checkpoint_path, target_device)
    timings["checkpoint_model_load"] = float(time.perf_counter() - load_started)
    checkpoint_identity = validate_checkpoint_identity(
        checkpoint_hash,
        expected_hash,
        checkpoint_data,
        expected_iter,
    )
    rss_stages["after_checkpoint_load"] = _rss_bytes()

    extract_started = time.perf_counter()
    arrays = extract_checkpoint_topology_arrays(model)
    timings["model_array_extraction"] = float(time.perf_counter() - extract_started)
    model_array_identities = {
        "vertices": array_identity(arrays.vertices),
        "faces": array_identity(arrays.faces),
        "stored_guide_points_local": array_identity(
            arrays.stored_guide_points_local
        ),
        "canonical_guide_points_local": array_identity(arrays.guide_points_local),
        "guide_point_reconstruction": arrays.guide_point_reconstruction,
        "guide_face_ids": array_identity(arrays.guide_face_ids),
        "guide_barycentric": array_identity(arrays.guide_barycentric),
    }

    topology: ActualTopologyData | None = None
    selected_artifacts: CandidateArtifacts | None = None
    selected_k: int | None = None
    topology_failure: dict[str, Any] | None = None
    topology_started = time.perf_counter()
    try:
        topology = build_actual_topology_data(arrays)
        timings["actual_topology_build"] = float(time.perf_counter() - topology_started)
        rss_stages["after_topology"] = _rss_bytes()
    except TopologyCoverError as error:
        timings["actual_topology_build"] = float(time.perf_counter() - topology_started)
        topology_failure = {
            "type": type(error).__name__,
            "message": str(error),
        }
        candidate_results = _not_evaluated_candidates("topology_validation_failed")
    if topology is not None:
        scan_started = time.perf_counter()
        candidate_results, selected_k, selected_artifacts = scan_candidate_ks(
            topology,
            arrays.guide_points_local,
            device=target_device,
            progress=True,
        )
        timings["candidate_scan"] = float(time.perf_counter() - scan_started)
    rss_stages["after_candidate_scan"] = _rss_bytes()

    accepted = selected_k is not None and selected_artifacts is not None
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "accepted": bool(accepted),
        "selected_k": selected_k,
        "candidate_k_sequence": list(CANDIDATE_K_SEQUENCE),
        "selection_rule": "first_predeclared_K_passing_every_hard_gate",
        "thresholds": {
            "max_local_condition_number": MAX_LOCAL_CONDITION_NUMBER,
            "max_local_node_self_error": MAX_LOCAL_ERROR,
            "max_local_constant_error": MAX_LOCAL_ERROR,
            "max_local_cardinal_sum_error": MAX_LOCAL_ERROR,
            "max_guide_point_reconstruction_error": (
                MAX_GUIDE_POINT_RECONSTRUCTION_ERROR
            ),
            "max_patch_node_count": MAX_PATCH_NODE_COUNT,
            "max_serialized_state_bytes": MAX_SERIALIZED_STATE_BYTES,
        },
        "identities": {
            "source": {
                **git_identity,
                "expectation": source_identity,
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "bytes": int(checkpoint_path.stat().st_size),
                **checkpoint_identity,
            },
            "model_arrays": model_array_identities,
            "source_files": _source_file_identities(),
            "config_type": type(config).__name__,
            "model_type": type(model).__name__,
        },
        "topology": topology.report if topology is not None else {
            "passed": False,
            "failure": topology_failure,
        },
        "candidate_results": candidate_results,
        "selected_state": (
            {
                "serialized_state_bytes": selected_artifacts.serialized_state_bytes,
                "resident_rss_metric": False,
                "patch_node_csr": _sparse_identity(
                    selected_artifacts.patch_nodes.node_distances
                ),
                "vertex_active_csr": _sparse_identity(
                    selected_artifacts.vertex_cover.active_distances
                ),
                "face_candidate_csr": _sparse_identity(
                    selected_artifacts.face_cover.candidate_counts
                ),
            }
            if selected_artifacts is not None
            else None
        ),
        "timings_seconds": timings,
        "memory": {
            "rss_stage_bytes": rss_stages,
            "max_observed_rss_bytes": int(max(rss_stages.values())),
            "cuda": _cuda_memory(target_device),
        },
        "versions": _versions(),
        "no_training": True,
        "no_rendering_or_images": True,
        "no_checkpoint_mutation": True,
    }
    report["timings_seconds"]["total_before_output"] = float(
        time.perf_counter() - total_started
    )
    manifest = save_diagnostic_outputs(
        output_root,
        report,
        topology=topology,
        artifacts=selected_artifacts if accepted else None,
        overwrite=overwrite,
    )
    print(
        f"R084_C1_FINAL accepted={str(bool(accepted)).lower()} "
        f"selected_k={selected_k if selected_k is not None else 'none'} "
        f"guides={arrays.guide_points_local.shape[0]} "
        f"artifacts={len(manifest['artifacts'])}",
        flush=True,
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R084 Phase C1 actual-checkpoint RBF partition-cover diagnostic."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-iteration", required=True, type=int)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        run_checkpoint_partition_cover_diagnostic(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_iteration=args.expected_iteration,
            expected_source_commit=args.expected_source_commit,
            overwrite=bool(args.overwrite),
        )
    except Exception as error:
        print(f"R084_C1_ERROR={error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

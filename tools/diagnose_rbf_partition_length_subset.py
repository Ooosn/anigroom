"""R084 Phase C2 fixed-checkpoint RBF-PU scalar-field diagnostic.

The diagnostic consumes a fully accepted Phase-C1 state, evaluates the exact
fixed-checkpoint scalar length field on bounded query groups, and writes only
numeric evidence.  It does not train, render images, mutate checkpoints, or
change formal configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.rbf_partition_of_unity import wendland_c2  # noqa: E402
from anigroom.rbf_topology_cover import (  # noqa: E402
    FacePatchCover,
    PatchGuideDistanceMatrix,
    PatchNodeCover,
    RaggedQueryTopologyDistances,
    TopologyCoverInputs,
    VertexPatchCover,
    build_face_patch_candidate_counts,
    build_vertex_patch_active_distances,
    evaluate_query_topology_distances,
    safe_barycentric_pl_sum,
    validate_topology_cover_inputs,
)
from anigroom.surface_interpolation import SurfaceSupport  # noqa: E402
from tools import diagnose_rbf_partition_cover as c1  # noqa: E402


SCHEMA = "r084.rbf_partition_length_subset.actual_checkpoint.phase_c2.v1"
RENDER_ROOT_COUNT = 4096
SELECTION_SEED = 20260901
INTERIOR_EDGE_PROBE_COUNT = 1024
REQUIRED_ROOT_ID = 431701
EXPECTED_RENDER_POPULATION = 496632
EXPECTED_SELECTED_IDS_SHA256 = (
    "fc9862c75c240e8e3c5f3ffc6940264936fb53c789620b8b44c6697b89a43d56"
)
EXPECTED_REQUIRED_ROOT_ROW = 3552
EVALUATION_TIMEOUT_BUDGET_SECONDS = 600.0
EXPECTED_GUIDE_COUNT = 4500
EXPECTED_SELECTED_K = 32
EXPECTED_NODES_PER_PATCH = 32
MAX_INVERSE_RESIDUAL = 1.0e-10
MAX_FIELD_ERROR = 1.0e-10
DEFAULT_SYSTEM_CHUNK_SIZE = 256
DEFAULT_PAIR_CHUNK_SIZE = 8192
UINT64_MASK = (1 << 64) - 1

HARD_GATE_KEYS = (
    "identities_and_c1_state_exact",
    "all_query_groups_covered_and_finite",
    "guide_site_self_error_at_most_1e-10",
    "constant_reproduction_at_most_1e-10",
    "local_and_global_cardinal_error_at_most_1e-10",
    "edge_midpoint_cross_face_difference_at_most_1e-10",
    "all_evaluated_lengths_finite_positive",
    "required_root_431701_present_covered_finite_positive",
    "inverse_residual_at_most_1e-10",
    "field_evaluation_within_600_seconds",
)

ARRAY_NAMES = (
    "selected_root_ids.npy",
    "candidate_lengths.npy",
    "legacy_lengths.npy",
    "selected_global_weight_indptr.npy",
    "selected_global_weight_indices.npy",
    "selected_global_weight_data.npy",
    "guide_site_errors.npy",
    "edge_pair_values.npy",
)
REPORT_NAME = "report.json"
MANIFEST_NAME = "sha256_manifest.json"
ARTIFACT_NAMES = (*ARRAY_NAMES, REPORT_NAME, MANIFEST_NAME)


class DiagnosticError(RuntimeError):
    """The Phase-C2 evidence contract failed or could not execute safely."""


@dataclass(frozen=True)
class VerifiedC1Artifacts:
    state_dir: Path
    report: dict[str, Any]
    manifest: dict[str, Any]
    paths: dict[str, Path]
    identities: dict[str, Any]


@dataclass(frozen=True)
class LoadedC1State:
    inputs: TopologyCoverInputs
    patch_guide_distances: PatchGuideDistanceMatrix
    patch_nodes: PatchNodeCover
    vertex_cover: VertexPatchCover
    face_cover: FacePatchCover
    report: dict[str, Any]


@dataclass(frozen=True)
class BatchedPatchSystems:
    guide_points: torch.Tensor
    node_ids: torch.Tensor
    radii: torch.Tensor
    inverses: torch.Tensor
    report: dict[str, Any]

    @property
    def patch_count(self) -> int:
        return int(self.node_ids.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.node_ids.shape[1])


@dataclass(frozen=True)
class RaggedFieldResult:
    values: np.ndarray
    constant_values: np.ndarray
    cardinal_row_sums: np.ndarray
    covered: np.ndarray
    global_weights: csr_matrix | None
    report: dict[str, Any]


@dataclass(frozen=True)
class RenderGeometry:
    points_local: np.ndarray
    normals_local: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray
    report: dict[str, Any]


@dataclass(frozen=True)
class InteriorEdgeQueries:
    edge_vertices: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray
    points_local: np.ndarray
    report: dict[str, Any]


def sha256_file(path: str | os.PathLike[str]) -> str:
    return c1.sha256_file(path)


def array_identity(values: Any) -> dict[str, Any]:
    return c1.array_identity(values)


def normalize_sha256(value: str, name: str) -> str:
    return c1.normalize_sha256(value, name)


def normalize_git_commit(value: str) -> str:
    return c1.normalize_git_commit(value)


def get_clean_source_git_identity(
    repository: str | os.PathLike[str] = PROJECT_ROOT,
) -> dict[str, Any]:
    return c1.get_clean_source_git_identity(repository)


def _strict_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"invalid {name}: {path}") from error
    if not isinstance(payload, dict):
        raise DiagnosticError(f"{name} must contain a JSON object")
    return payload


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
    output = Path(value).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output-dir is not a directory: {output}")
        if any(output.iterdir()) and not overwrite:
            raise FileExistsError(
                f"refusing nonempty output-dir without --overwrite: {output}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _require_c1_report_contract(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("schema") != c1.SCHEMA:
        raise DiagnosticError(f"C1 report schema mismatch: {report.get('schema')!r}")
    if report.get("accepted") is not True:
        raise DiagnosticError("C1 report is not accepted")
    if report.get("selected_k") != EXPECTED_SELECTED_K:
        raise DiagnosticError(
            f"C1 selected_k must be {EXPECTED_SELECTED_K}, got {report.get('selected_k')!r}"
        )
    if len(c1.HARD_GATE_KEYS) != 14:
        raise DiagnosticError("C1 hard-gate schema no longer contains exactly 14 gates")
    exact_sequence = list(c1.CANDIDATE_K_SEQUENCE)
    if report.get("candidate_k_sequence") != exact_sequence:
        raise DiagnosticError("C1 candidate K sequence is not the exact predeclared sequence")
    results = report.get("candidate_results")
    if not isinstance(results, list) or len(results) != len(exact_sequence):
        raise DiagnosticError("C1 candidate_results must contain exactly seven rows")
    if [item.get("k") if isinstance(item, dict) else None for item in results] != exact_sequence:
        raise DiagnosticError("C1 candidate results are not in exact K order")
    for index, (candidate_k, result) in enumerate(zip(exact_sequence, results)):
        if not isinstance(result, dict):
            raise DiagnosticError(f"C1 K={candidate_k} result is not an object")
        if candidate_k in (48, 64):
            if result.get("status") != "not_evaluated_after_first_pass":
                raise DiagnosticError(
                    f"C1 K={candidate_k} must be not_evaluated_after_first_pass"
                )
            if result.get("all_hard_gates_passed") is not None:
                raise DiagnosticError(f"C1 K={candidate_k} aggregate must be null")
            if "hard_gates" in result:
                raise DiagnosticError(f"C1 K={candidate_k} must not report evaluated gates")
            continue
        expected_status = "passed" if candidate_k == 32 else "rejected"
        if result.get("status") != expected_status:
            raise DiagnosticError(
                f"C1 K={candidate_k} status must be {expected_status}"
            )
        gates = result.get("hard_gates")
        if not isinstance(gates, dict) or set(gates) != set(c1.HARD_GATE_KEYS):
            raise DiagnosticError(
                f"C1 K={candidate_k} hard-gate identities do not match all 14 gates"
            )
        if not all(isinstance(value, (bool, np.bool_)) for value in gates.values()):
            raise DiagnosticError(f"C1 K={candidate_k} hard gates must be boolean")
        aggregate = result.get("all_hard_gates_passed")
        expected_aggregate = bool(all(bool(value) for value in gates.values()))
        if aggregate is not expected_aggregate:
            raise DiagnosticError(
                f"C1 K={candidate_k} aggregate does not equal its exact gates"
            )
        if candidate_k == 32 and not expected_aggregate:
            raise DiagnosticError("C1 K=32 must pass every hard gate")
        if candidate_k != 32 and expected_aggregate:
            raise DiagnosticError(f"C1 K={candidate_k} must be rejected")
        if candidate_k == 32 and index != 4:
            raise DiagnosticError("C1 K=32 is not the first pass position")
    selected_gates = results[4]["hard_gates"]
    return {
        "passed": True,
        "schema": c1.SCHEMA,
        "selected_k": 32,
        "hard_gate_count": 14,
        "hard_gates": selected_gates,
        "candidate_k_sequence": exact_sequence,
        "first_and_only_passing_k": 32,
    }


def verify_c1_artifacts(
    state_dir: str | os.PathLike[str],
    *,
    expected_report_sha256: str,
    expected_manifest_sha256: str,
) -> VerifiedC1Artifacts:
    root = Path(state_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"C1 state dir is not a directory: {root}")
    report_path = _path_file(root / c1.REPORT_NAME, "C1 report")
    manifest_path = _path_file(root / c1.MANIFEST_NAME, "C1 manifest")
    expected_report = normalize_sha256(
        expected_report_sha256,
        "expected C1 report SHA256",
    )
    expected_manifest = normalize_sha256(
        expected_manifest_sha256,
        "expected C1 manifest SHA256",
    )
    observed_report = sha256_file(report_path)
    observed_manifest = sha256_file(manifest_path)
    if observed_report != expected_report:
        raise DiagnosticError(
            f"C1 report SHA256 mismatch: expected {expected_report}, got {observed_report}"
        )
    if observed_manifest != expected_manifest:
        raise DiagnosticError(
            f"C1 manifest SHA256 mismatch: expected {expected_manifest}, got {observed_manifest}"
        )
    manifest = _strict_json(manifest_path, "C1 manifest")
    if manifest.get("schema") != c1.SCHEMA or manifest.get("algorithm") != "sha256":
        raise DiagnosticError("C1 manifest schema or algorithm mismatch")
    entries = manifest.get("artifacts")
    if not isinstance(entries, dict):
        raise DiagnosticError("C1 manifest artifacts must be an object")
    expected_names = {*c1.STATE_ARRAY_NAMES, c1.REPORT_NAME}
    if set(entries) != expected_names:
        raise DiagnosticError("C1 manifest artifact names are incomplete or unexpected")
    paths: dict[str, Path] = {}
    identities: dict[str, Any] = {}
    for name in sorted(expected_names):
        entry = entries[name]
        if not isinstance(entry, dict):
            raise DiagnosticError(f"C1 manifest entry is invalid: {name}")
        expected_hash = normalize_sha256(
            entry.get("sha256", ""),
            f"C1 manifest {name} SHA256",
        )
        expected_bytes = entry.get("bytes")
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise DiagnosticError(f"C1 manifest {name} byte count is invalid")
        path = _path_file(root / name, f"C1 artifact {name}")
        observed_bytes = int(path.stat().st_size)
        observed_hash = sha256_file(path)
        if observed_bytes != expected_bytes or observed_hash != expected_hash:
            raise DiagnosticError(
                f"C1 artifact hash/size mismatch for {name}: "
                f"expected {expected_bytes}/{expected_hash}, "
                f"got {observed_bytes}/{observed_hash}"
            )
        paths[name] = path
        identities[name] = {
            "bytes": observed_bytes,
            "sha256": observed_hash,
        }
    if identities[c1.REPORT_NAME]["sha256"] != observed_report:
        raise DiagnosticError("C1 manifest report hash differs from expected report hash")
    # Every manifest artifact has now been hashed.  Only now parse report/state.
    report = _strict_json(report_path, "C1 report")
    contract = _require_c1_report_contract(report)
    identities["external_expectations"] = {
        "report_sha256": expected_report,
        "manifest_sha256": expected_manifest,
        "passed": True,
    }
    identities["report_contract"] = contract
    return VerifiedC1Artifacts(
        state_dir=root,
        report=report,
        manifest=manifest,
        paths=paths,
        identities=identities,
    )


def _checked_u64(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    converted = int(value)
    if converted < 0 or converted > UINT64_MASK:
        raise ValueError(f"{name} must lie in [0, 2**64-1]")
    return converted


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    mixed = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (mixed ^ (mixed >> 31)) & UINT64_MASK


def stable_root_rank(root_id: int, seed: int) -> int:
    root = _checked_u64(root_id, "root_id")
    selection_seed = _checked_u64(seed, "selection_seed")
    mixed_input = (
        selection_seed ^ ((root * 0xD6E8FEB86659FD93) & UINT64_MASK)
    ) & UINT64_MASK
    return _splitmix64(mixed_input)


def select_render_root_ids(
    population_count: int,
    requested_count: int = RENDER_ROOT_COUNT,
    selection_seed: int = SELECTION_SEED,
) -> np.ndarray:
    population = int(population_count)
    requested = int(requested_count)
    if population <= 0:
        raise ValueError("render population must be positive")
    if requested <= 0 or requested > population:
        raise ValueError("requested render root count is invalid")
    _checked_u64(selection_seed, "selection_seed")
    ranked = sorted(
        (stable_root_rank(root_id, selection_seed), root_id)
        for root_id in range(population)
    )
    return np.asarray(
        sorted(root_id for _, root_id in ranked[:requested]),
        dtype=np.int64,
    )


def _load_npy(path: Path, name: str) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise DiagnosticError(f"invalid C1 array {name}") from error
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise DiagnosticError(f"C1 array {name} is not a numeric ndarray")
    return np.ascontiguousarray(value)


def _strict_csr(
    *,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
    name: str,
    data_kind: str,
) -> csr_matrix:
    if indptr.dtype.kind not in "iu" or indptr.shape != (shape[0] + 1,):
        raise DiagnosticError(f"{name} indptr is invalid")
    if indices.dtype.kind not in "iu" or indices.ndim != 1:
        raise DiagnosticError(f"{name} indices are invalid")
    if data.ndim != 1 or data.shape != indices.shape:
        raise DiagnosticError(f"{name} data length is invalid")
    if int(indptr[0]) != 0 or np.any(indptr[1:] < indptr[:-1]):
        raise DiagnosticError(f"{name} indptr must be monotone from zero")
    if int(indptr[-1]) != int(indices.size):
        raise DiagnosticError(f"{name} final indptr does not equal nnz")
    if np.any(indices < 0) or np.any(indices >= shape[1]):
        raise DiagnosticError(f"{name} contains an out-of-range column")
    if data_kind == "float":
        if data.dtype.kind != "f" or not np.isfinite(data).all():
            raise DiagnosticError(f"{name} data must be finite floating point")
    elif data_kind == "integer":
        if data.dtype.kind not in "iu":
            raise DiagnosticError(f"{name} data must be integer")
    else:
        raise ValueError(f"unsupported CSR data kind: {data_kind}")
    matrix = csr_matrix(
        (
            np.ascontiguousarray(data),
            np.ascontiguousarray(indices),
            np.ascontiguousarray(indptr),
        ),
        shape=shape,
    )
    if not matrix.has_sorted_indices or not matrix.has_canonical_format:
        raise DiagnosticError(f"{name} rows must be sorted unique canonical rows")
    return matrix


def _assert_array_identity(
    values: np.ndarray,
    expected: Mapping[str, Any],
    name: str,
) -> None:
    observed = array_identity(values)
    for key in ("shape", "dtype", "bytes", "sha256"):
        if observed.get(key) != expected.get(key):
            raise DiagnosticError(
                f"C1 {name} array identity mismatch for {key}: "
                f"expected {expected.get(key)!r}, got {observed.get(key)!r}"
            )


def _verify_patch_guide_matrix_binding(
    inputs: TopologyCoverInputs,
    matrix: np.ndarray,
    *,
    patch_chunk_size: int = 128,
    guide_chunk_size: int = 1024,
) -> None:
    guide_count = inputs.guide_count
    guide_vertices = inputs.faces[inputs.guide_face_ids]
    seeds = inputs.vertex_seed_guide_ids[guide_vertices]
    delta = inputs.vertex_nearest_distances[guide_vertices]
    for patch_begin in range(0, guide_count, int(patch_chunk_size)):
        patch_end = min(patch_begin + int(patch_chunk_size), guide_count)
        d_rows = inputs.guide_distances[patch_begin:patch_end]
        for guide_begin in range(0, guide_count, int(guide_chunk_size)):
            guide_end = min(guide_begin + int(guide_chunk_size), guide_count)
            vertex_values = (
                d_rows[:, seeds[guide_begin:guide_end]]
                + delta[guide_begin:guide_end][None, :, :]
            )
            expected = safe_barycentric_pl_sum(
                vertex_values,
                inputs.guide_barycentric[guide_begin:guide_end],
            )
            actual = matrix[
                patch_begin:patch_end,
                guide_begin:guide_end,
            ]
            if not np.array_equal(actual, expected):
                raise DiagnosticError(
                    "C1 M is not exactly bound to D/seed/delta/guide barycentrics"
                )


def _assert_csr_equal(actual: csr_matrix, expected: csr_matrix, name: str) -> None:
    if actual.shape != expected.shape:
        raise DiagnosticError(f"{name} CSR shape mismatch")
    for field in ("indptr", "indices", "data"):
        if not np.array_equal(getattr(actual, field), getattr(expected, field)):
            raise DiagnosticError(f"{name} CSR {field} differs from reconstructed state")


def validate_c1_source_contract(
    verified: VerifiedC1Artifacts,
    *,
    expected_c1_source_commit: str,
) -> dict[str, Any]:
    expected_commit = normalize_git_commit(expected_c1_source_commit)
    identities = verified.report.get("identities")
    if not isinstance(identities, dict):
        raise DiagnosticError("C1 report identities are missing")
    source = identities.get("source")
    source_files = identities.get("source_files")
    if not isinstance(source, dict) or not isinstance(source_files, dict):
        raise DiagnosticError("C1 source identities are malformed")
    if source.get("clean") is not True:
        raise DiagnosticError("C1 source identity was not clean")
    observed_head = normalize_git_commit(source.get("head", ""))
    expectation = source.get("expectation")
    if not isinstance(expectation, dict):
        raise DiagnosticError("C1 source expectation is missing")
    if (
        observed_head != expected_commit
        or normalize_git_commit(expectation.get("expected", "")) != expected_commit
        or normalize_git_commit(expectation.get("observed", "")) != expected_commit
        or expectation.get("passed") is not True
    ):
        raise DiagnosticError("C1 report does not bind to expected-c1-source-commit")
    current_paths = {
        "diagnostic": Path(c1.__file__).resolve(),
        "rbf_partition_of_unity": (
            PROJECT_ROOT / "anigroom" / "rbf_partition_of_unity.py"
        ),
        "rbf_topology_cover": PROJECT_ROOT / "anigroom" / "rbf_topology_cover.py",
        "surface_graph": PROJECT_ROOT / "anigroom" / "flow" / "surface_graph.py",
    }
    if set(source_files) != set(current_paths):
        raise DiagnosticError("C1 source-file identity names are incomplete or unexpected")
    current: dict[str, Any] = {}
    for name, path in current_paths.items():
        expected = source_files[name]
        if not isinstance(expected, dict):
            raise DiagnosticError(f"C1 source-file identity is invalid: {name}")
        observed = {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if (
            expected.get("bytes") != observed["bytes"]
            or normalize_sha256(
                expected.get("sha256", ""),
                f"C1 source-file {name} SHA256",
            )
            != observed["sha256"]
        ):
            raise DiagnosticError(
                f"C1 source file changed since Phase C1: {name}"
            )
        current[name] = observed
    return {
        "passed": True,
        "expected_c1_source_commit": expected_commit,
        "observed_c1_source_commit": observed_head,
        "unchanged_source_files": current,
    }


def validate_c1_checkpoint_binding(
    verified: VerifiedC1Artifacts,
    checkpoint_arrays: c1.CheckpointTopologyArrays,
    *,
    checkpoint_sha256: str,
    checkpoint_iteration: int,
    expected_c1_source_commit: str,
) -> dict[str, Any]:
    report = verified.report
    identities = report.get("identities")
    if not isinstance(identities, dict):
        raise DiagnosticError("C1 report identities are missing")
    checkpoint_identity = identities.get("checkpoint")
    source_identity = identities.get("source")
    model_identities = identities.get("model_arrays")
    if not all(
        isinstance(value, dict)
        for value in (checkpoint_identity, source_identity, model_identities)
    ):
        raise DiagnosticError("C1 identity sections are malformed")
    observed_checkpoint = normalize_sha256(
        checkpoint_identity.get("observed_sha256", ""),
        "C1 checkpoint SHA256",
    )
    if observed_checkpoint != checkpoint_sha256:
        raise DiagnosticError("C1 state belongs to a different checkpoint SHA256")
    if checkpoint_identity.get("observed_iteration") != int(checkpoint_iteration):
        raise DiagnosticError("C1 state belongs to a different checkpoint iteration")
    c1_head = source_identity.get("head")
    if normalize_git_commit(c1_head) != normalize_git_commit(
        expected_c1_source_commit
    ):
        raise DiagnosticError("C1 state belongs to a different source commit")
    current_arrays = {
        "vertices": checkpoint_arrays.vertices,
        "faces": checkpoint_arrays.faces,
        "stored_guide_points_local": checkpoint_arrays.stored_guide_points_local,
        "canonical_guide_points_local": checkpoint_arrays.guide_points_local,
        "guide_face_ids": checkpoint_arrays.guide_face_ids,
        "guide_barycentric": checkpoint_arrays.guide_barycentric,
    }
    for name, values in current_arrays.items():
        expected_identity = model_identities.get(name)
        if not isinstance(expected_identity, dict):
            raise DiagnosticError(f"C1 model identity is missing {name}")
        _assert_array_identity(values, expected_identity, name)
    return {
        "passed": True,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_iteration": int(checkpoint_iteration),
        "c1_source_commit": normalize_git_commit(expected_c1_source_commit),
        "model_arrays": {
            name: array_identity(values) for name, values in current_arrays.items()
        },
    }


def load_c1_state(
    verified: VerifiedC1Artifacts,
    checkpoint_arrays: c1.CheckpointTopologyArrays,
    *,
    expected_guide_count: int = EXPECTED_GUIDE_COUNT,
    expected_nodes_per_patch: int = EXPECTED_NODES_PER_PATCH,
) -> LoadedC1State:
    arrays = {
        name: _load_npy(verified.paths[name], name)
        for name in c1.STATE_ARRAY_NAMES
    }
    guide_count = int(arrays["guide_distances.npy"].shape[0])
    vertex_count = int(checkpoint_arrays.vertices.shape[0])
    face_count = int(checkpoint_arrays.faces.shape[0])
    if guide_count != int(expected_guide_count):
        raise DiagnosticError(
            f"C1 guide count must be {expected_guide_count}, got {guide_count}"
        )
    if arrays["guide_distances.npy"].shape != (guide_count, guide_count):
        raise DiagnosticError("C1 D must have shape [G, G]")
    if arrays["patch_guide_distances.npy"].shape != (guide_count, guide_count):
        raise DiagnosticError("C1 M must have shape [G, G]")
    if arrays["patch_radii.npy"].shape != (guide_count,):
        raise DiagnosticError("C1 radii must have shape [G]")
    inputs = validate_topology_cover_inputs(
        arrays["guide_distances.npy"],
        arrays["vertex_seed_guide_ids.npy"],
        arrays["vertex_nearest_distances.npy"],
        checkpoint_arrays.faces,
        checkpoint_arrays.guide_face_ids,
        checkpoint_arrays.guide_barycentric,
    )
    if inputs.vertex_count != vertex_count or inputs.face_count != face_count:
        raise DiagnosticError("C1 topology dimensions differ from checkpoint mesh")
    if not np.array_equal(inputs.component_labels, arrays["component_labels.npy"]):
        raise DiagnosticError("C1 component labels differ from validated D components")
    patch_matrix = PatchGuideDistanceMatrix(
        values=np.ascontiguousarray(arrays["patch_guide_distances.npy"], dtype=np.float64),
        report={},
    )
    _verify_patch_guide_matrix_binding(inputs, patch_matrix.values)
    patch_csr = _strict_csr(
        indptr=arrays["patch_node_indptr.npy"],
        indices=arrays["patch_node_indices.npy"],
        data=arrays["patch_node_distances.npy"],
        shape=(guide_count, guide_count),
        name="patch-node",
        data_kind="float",
    )
    node_counts = np.diff(patch_csr.indptr)
    if not np.all(node_counts == int(expected_nodes_per_patch)):
        raise DiagnosticError(
            f"C1 must contain exactly {expected_nodes_per_patch} nodes per patch"
        )
    patch_nodes = PatchNodeCover(
        radii=np.ascontiguousarray(arrays["patch_radii.npy"], dtype=np.float64),
        node_distances=patch_csr,
        report={},
    )
    cover_evidence = c1.derive_patch_cover_evidence(patch_matrix, patch_nodes)
    if cover_evidence.get("passed") is not True:
        raise DiagnosticError("C1 patch boundary/self-membership evidence failed")
    vertex_csr = _strict_csr(
        indptr=arrays["vertex_active_indptr.npy"],
        indices=arrays["vertex_active_indices.npy"],
        data=arrays["vertex_active_distances.npy"],
        shape=(vertex_count, guide_count),
        name="vertex-active",
        data_kind="float",
    )
    vertex_cover = VertexPatchCover(
        active_distances=vertex_csr,
        patch_radii=patch_nodes.radii.copy(),
        patch_node_counts=node_counts.astype(np.int64, copy=True),
        report={},
    )
    face_csr = _strict_csr(
        indptr=arrays["face_candidate_indptr.npy"],
        indices=arrays["face_candidate_indices.npy"],
        data=arrays["face_candidate_counts.npy"],
        shape=(face_count, guide_count),
        name="face-candidate",
        data_kind="integer",
    )
    face_cover = FacePatchCover(
        candidate_counts=face_csr,
        patch_radii=patch_nodes.radii.copy(),
        patch_node_counts=node_counts.astype(np.int64, copy=True),
        report={},
    )
    reconstructed_vertex = build_vertex_patch_active_distances(inputs, patch_nodes)
    _assert_csr_equal(vertex_csr, reconstructed_vertex.active_distances, "vertex-active")
    reconstructed_face = build_face_patch_candidate_counts(inputs, reconstructed_vertex)
    _assert_csr_equal(face_csr, reconstructed_face.candidate_counts, "face-candidate")
    report_artifacts = verified.report.get("artifacts")
    if not isinstance(report_artifacts, dict):
        raise DiagnosticError("C1 report artifact identities are missing")
    for name, values in arrays.items():
        expected_identity = report_artifacts.get(name, {}).get("array")
        if not isinstance(expected_identity, dict):
            raise DiagnosticError(f"C1 report array identity is missing {name}")
        _assert_array_identity(values, expected_identity, name)
    report = {
        "passed": True,
        "guide_count": guide_count,
        "vertex_count": vertex_count,
        "face_count": face_count,
        "nodes_per_patch": int(expected_nodes_per_patch),
        "cover_evidence": cover_evidence,
        "array_identities": {
            name: array_identity(values) for name, values in arrays.items()
        },
        "D_semantics": "root_voronoi_graph_shortest_path_not_exact_geodesic",
        "M_semantics": "continuous_original_mesh_PL_proxy_not_exact_geodesic",
    }
    return LoadedC1State(
        inputs=inputs,
        patch_guide_distances=patch_matrix,
        patch_nodes=patch_nodes,
        vertex_cover=vertex_cover,
        face_cover=face_cover,
        report=report,
    )


def extract_fixed_patch_layout(
    guide_count: int,
    patch_nodes: PatchNodeCover,
    *,
    expected_node_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    patch_count = int(guide_count)
    if patch_count <= 0:
        raise ValueError("guide_count must be positive")
    matrix = patch_nodes.node_distances
    if matrix.shape != (patch_count, patch_count):
        raise ValueError("patch-node CSR shape does not match guide points")
    counts = np.diff(matrix.indptr)
    if counts.size == 0 or not np.all(counts == counts[0]):
        raise DiagnosticError("every patch must have the same fixed node count")
    node_count = int(counts[0])
    if node_count <= 0:
        raise DiagnosticError("patch node count must be positive")
    if expected_node_count is not None and node_count != int(expected_node_count):
        raise DiagnosticError(
            f"patch node count must be exactly {int(expected_node_count)}"
        )
    node_ids_np = np.ascontiguousarray(
        matrix.indices.reshape(patch_count, node_count),
        dtype=np.int64,
    )
    if np.any(node_ids_np < 0) or np.any(node_ids_np >= patch_count):
        raise DiagnosticError("patch layout contains an out-of-range guide ID")
    radii_np = np.asarray(patch_nodes.radii, dtype=np.float64)
    if radii_np.shape != (patch_count,) or not np.isfinite(radii_np).all():
        raise ValueError("patch radii must be finite with shape [P]")
    if np.any(radii_np <= 0.0):
        raise ValueError("patch radii must be positive")
    return node_ids_np, np.ascontiguousarray(radii_np), node_count


def build_batched_patch_systems(
    guide_points_local: np.ndarray | torch.Tensor,
    patch_nodes: PatchNodeCover,
    *,
    device: torch.device | str,
    system_chunk_size: int = DEFAULT_SYSTEM_CHUNK_SIZE,
) -> BatchedPatchSystems:
    chunk_size = int(system_chunk_size)
    if chunk_size <= 0:
        raise ValueError("system_chunk_size must be positive")
    target_device = torch.device(device)
    points = torch.as_tensor(
        guide_points_local,
        dtype=torch.float64,
        device=target_device,
    )
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] <= 0:
        raise ValueError("guide_points_local must have shape [G, 3]")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("guide_points_local must be finite")
    patch_count = int(points.shape[0])
    node_ids_np, radii_np, node_count = extract_fixed_patch_layout(
        patch_count,
        patch_nodes,
    )
    node_ids = torch.as_tensor(node_ids_np, dtype=torch.long, device=target_device)
    radii = torch.as_tensor(radii_np, dtype=torch.float64, device=target_device)
    matrix_size = node_count + 1
    inverses = torch.empty(
        (patch_count, matrix_size, matrix_size),
        dtype=torch.float64,
        device=target_device,
    )
    residuals = np.empty((patch_count,), dtype=np.float64)
    started = time.perf_counter()
    identity = torch.eye(matrix_size, dtype=torch.float64, device=target_device)
    with torch.no_grad():
        for begin in range(0, patch_count, chunk_size):
            end = min(begin + chunk_size, patch_count)
            ids = node_ids[begin:end]
            sources = points[ids]
            chord = torch.linalg.vector_norm(
                sources[:, :, None, :] - sources[:, None, :, :],
                dim=-1,
            )
            kernel = wendland_c2(
                chord / (2.0 * radii[begin:end, None, None])
            )
            systems = torch.empty(
                (end - begin, matrix_size, matrix_size),
                dtype=torch.float64,
                device=target_device,
            )
            systems[:, :-1, :-1] = kernel
            systems[:, :-1, -1] = 1.0
            systems[:, -1, :-1] = 1.0
            systems[:, -1, -1] = 0.0
            solved = torch.linalg.solve(
                systems,
                identity.expand(end - begin, -1, -1),
            )
            if not bool(torch.isfinite(solved).all()):
                raise DiagnosticError("batched patch inverse contains nonfinite values")
            residual = torch.amax(
                torch.abs(torch.bmm(systems, solved) - identity),
                dim=(1, 2),
            )
            if not bool(torch.isfinite(residual).all()):
                raise DiagnosticError("batched patch inverse residual is nonfinite")
            inverses[begin:end] = solved
            residuals[begin:end] = residual.detach().cpu().numpy()
    max_residual = float(residuals.max())
    return BatchedPatchSystems(
        guide_points=points,
        node_ids=node_ids,
        radii=radii,
        inverses=inverses,
        report={
            "patch_count": patch_count,
            "node_count": node_count,
            "matrix_size": matrix_size,
            "dtype": "torch.float64",
            "device": str(target_device),
            "system_chunk_size": chunk_size,
            "inverse_residual_max": max_residual,
            "inverse_residual_p95": float(
                np.quantile(residuals, 0.95, method="linear")
            ),
            "inverse_residual_limit": MAX_INVERSE_RESIDUAL,
            "inverse_residual_passed": bool(max_residual <= MAX_INVERSE_RESIDUAL),
            "seconds": float(time.perf_counter() - started),
            "no_oom_fallback": True,
        },
    )


def _validate_ragged(
    ragged: RaggedQueryTopologyDistances,
    query_count: int,
    patch_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.asarray(ragged.indptr)
    patch_ids = np.asarray(ragged.patch_ids)
    distances = np.asarray(ragged.distances, dtype=np.float64)
    radii = np.asarray(ragged.radii, dtype=np.float64)
    if indptr.dtype.kind not in "iu" or indptr.shape != (query_count + 1,):
        raise ValueError("ragged indptr is invalid")
    if int(indptr[0]) != 0 or np.any(indptr[1:] < indptr[:-1]):
        raise ValueError("ragged indptr must be monotone from zero")
    if int(indptr[-1]) != int(patch_ids.size):
        raise ValueError("ragged final indptr does not equal entry count")
    if patch_ids.dtype.kind not in "iu" or patch_ids.ndim != 1:
        raise ValueError("ragged patch IDs are invalid")
    if np.any(patch_ids < 0) or np.any(patch_ids >= patch_count):
        raise ValueError("ragged patch ID is out of range")
    if distances.shape != patch_ids.shape or radii.shape != patch_ids.shape:
        raise ValueError("ragged distances/radii shape mismatch")
    if not np.isfinite(distances).all() or not np.isfinite(radii).all():
        raise ValueError("ragged distances/radii must be finite")
    if np.any(distances < 0.0) or np.any(radii <= 0.0):
        raise ValueError("ragged distances/radii are outside their legal range")
    query_ids = np.repeat(
        np.arange(query_count, dtype=np.int64),
        np.diff(indptr).astype(np.int64, copy=False),
    )
    return query_ids, distances < radii


def validate_ragged_binding(
    ragged: RaggedQueryTopologyDistances,
    state: LoadedC1State,
    query_face_ids: np.ndarray,
    query_barycentric: np.ndarray,
) -> dict[str, Any]:
    face_ids_raw = np.asarray(query_face_ids)
    if face_ids_raw.dtype.kind not in "iu" or face_ids_raw.ndim != 1:
        raise DiagnosticError("query face IDs must be a one-dimensional integer array")
    face_ids = np.ascontiguousarray(face_ids_raw, dtype=np.int64)
    bary = np.ascontiguousarray(np.asarray(query_barycentric, dtype=np.float64))
    query_count = int(face_ids.size)
    if bary.shape != (query_count, 3):
        raise DiagnosticError("query barycentrics must have shape [Q, 3]")
    if not np.isfinite(bary).all() or np.any(bary < 0.0) or np.any(bary > 1.0):
        raise DiagnosticError("query barycentrics are invalid")
    if np.any(np.abs(bary.sum(axis=1) - 1.0) > 1.0e-6):
        raise DiagnosticError("query barycentric rows do not sum to one")
    if np.any(face_ids < 0) or np.any(face_ids >= state.inputs.face_count):
        raise DiagnosticError("query face ID is out of range")
    _validate_ragged(ragged, query_count, state.inputs.guide_count)
    expected_node_counts = np.diff(state.patch_nodes.node_distances.indptr).astype(
        np.int64,
        copy=False,
    )
    actual_node_counts = np.asarray(ragged.patch_node_counts)
    if actual_node_counts.dtype.kind not in "iu" or not np.array_equal(
        actual_node_counts,
        expected_node_counts,
    ):
        raise DiagnosticError("ragged patch_node_counts differ from C1 state")
    indptr = np.asarray(ragged.indptr, dtype=np.int64)
    patch_ids = np.asarray(ragged.patch_ids, dtype=np.int64)
    distances = np.asarray(ragged.distances, dtype=np.float64)
    radii = np.asarray(ragged.radii, dtype=np.float64)
    candidate_csr = state.face_cover.candidate_counts
    for query_id in range(query_count):
        begin, end = indptr[query_id : query_id + 2]
        face_id = int(face_ids[query_id])
        expected_begin, expected_end = candidate_csr.indptr[face_id : face_id + 2]
        expected_ids = candidate_csr.indices[expected_begin:expected_end]
        actual_ids = patch_ids[begin:end]
        if not np.array_equal(actual_ids, expected_ids):
            raise DiagnosticError(
                f"ragged row {query_id} patch IDs differ from face-candidate CSR"
            )
        if actual_ids.size > 1 and np.any(actual_ids[1:] <= actual_ids[:-1]):
            raise DiagnosticError(f"ragged row {query_id} patch IDs are not sorted unique")
        expected_radii = state.patch_nodes.radii[actual_ids]
        if not np.array_equal(radii[begin:end], expected_radii):
            raise DiagnosticError(f"ragged row {query_id} radii differ from C1 state")
        if actual_ids.size == 0:
            if begin != end:
                raise DiagnosticError("empty ragged candidate row has nonempty storage")
            continue
        vertices = state.inputs.faces[face_id]
        seeds = state.inputs.vertex_seed_guide_ids[vertices]
        delta = state.inputs.vertex_nearest_distances[vertices]
        vertex_values = (
            state.inputs.guide_distances[actual_ids][:, seeds][:, None, :]
            + delta[None, None, :]
        )
        expected_distances = safe_barycentric_pl_sum(
            vertex_values,
            bary[query_id : query_id + 1],
        )[:, 0]
        if not np.array_equal(distances[begin:end], expected_distances):
            raise DiagnosticError(
                f"ragged row {query_id} distances differ from D/seed/delta/face barycentrics"
            )
    return {
        "passed": True,
        "query_count": query_count,
        "candidate_entry_count": int(patch_ids.size),
        "patch_id_identity": array_identity(patch_ids),
        "distance_identity": array_identity(distances),
        "radius_identity": array_identity(radii),
        "patch_node_count_identity": array_identity(actual_node_counts),
        "zero_weight_candidates_preserved": True,
    }


def _global_weight_report(matrix: csr_matrix) -> dict[str, Any]:
    negative = matrix.data < 0.0
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    absolute = matrix.copy()
    absolute.data = np.abs(absolute.data)
    l1 = np.asarray(absolute.sum(axis=1)).reshape(-1)
    squared = matrix.copy()
    squared.data = squared.data * squared.data
    square_sum = np.asarray(squared.sum(axis=1)).reshape(-1)
    normalized_square = np.divide(
        square_sum,
        l1 * l1,
        out=np.zeros_like(square_sum),
        where=l1 > 0.0,
    )
    effective = np.divide(
        1.0,
        normalized_square,
        out=np.zeros_like(normalized_square),
        where=normalized_square > 0.0,
    )
    negative_matrix = matrix.copy()
    negative_matrix.data = np.where(negative, -negative_matrix.data, 0.0)
    negative_mass = np.asarray(negative_matrix.sum(axis=1)).reshape(-1)
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "negative_entry_count": int(negative.sum()),
        "negative_mass_total": float(negative_mass.sum(dtype=np.float64)),
        "negative_mass_max_per_row": float(negative_mass.max()) if negative_mass.size else 0.0,
        "signed_row_sum_max_error": float(np.max(np.abs(row_sum - 1.0)))
        if row_sum.size
        else 0.0,
        "l1_mass_max": float(l1.max()) if l1.size else 0.0,
        "effective_influence_min": float(effective.min()) if effective.size else 0.0,
        "effective_influence_mean": float(effective.mean(dtype=np.float64))
        if effective.size
        else 0.0,
        "effective_influence_max": float(effective.max()) if effective.size else 0.0,
        "effective_influence_definition": "1/sum((abs(global_weight)/row_L1)^2)",
    }


def evaluate_ragged_field(
    query_points_local: np.ndarray | torch.Tensor,
    ragged: RaggedQueryTopologyDistances,
    systems: BatchedPatchSystems,
    guide_values: np.ndarray | torch.Tensor,
    *,
    pair_chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
    collapse_global_weights: bool = False,
) -> RaggedFieldResult:
    chunk_size = int(pair_chunk_size)
    if chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")
    device = systems.guide_points.device
    points = torch.as_tensor(
        query_points_local,
        dtype=torch.float64,
        device=device,
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("query_points_local must have shape [Q, 3]")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("query_points_local must be finite")
    values = torch.as_tensor(guide_values, dtype=torch.float64, device=device).reshape(-1)
    if values.shape != (systems.guide_points.shape[0],):
        raise ValueError("guide_values must have shape [G]")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("guide_values must be finite")
    query_count = int(points.shape[0])
    query_ids_np, active_mask = _validate_ragged(
        ragged,
        query_count,
        systems.patch_count,
    )
    patch_ids_np = np.asarray(ragged.patch_ids, dtype=np.int64)
    distances_np = np.asarray(ragged.distances, dtype=np.float64)
    radii_np = np.asarray(ragged.radii, dtype=np.float64)
    active_entry_ids = np.flatnonzero(active_mask).astype(np.int64, copy=False)
    denominator = torch.zeros((query_count,), dtype=torch.float64, device=device)
    value_numerator = torch.zeros_like(denominator)
    constant_numerator = torch.zeros_like(denominator)
    max_pair_cardinal_error = 0.0
    coo_rows: list[np.ndarray] = []
    coo_columns: list[np.ndarray] = []
    coo_values: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.no_grad():
        for begin in range(0, int(active_entry_ids.size), chunk_size):
            end = min(begin + chunk_size, int(active_entry_ids.size))
            entry_ids = active_entry_ids[begin:end]
            query_ids = torch.as_tensor(
                query_ids_np[entry_ids],
                dtype=torch.long,
                device=device,
            )
            patch_ids = torch.as_tensor(
                patch_ids_np[entry_ids],
                dtype=torch.long,
                device=device,
            )
            node_ids = systems.node_ids[patch_ids]
            sources = systems.guide_points[node_ids]
            chord = torch.linalg.vector_norm(
                points[query_ids, None, :] - sources,
                dim=-1,
            )
            kernel = wendland_c2(
                chord / (2.0 * systems.radii[patch_ids, None])
            )
            evaluation_rows = torch.cat(
                (
                    kernel,
                    torch.ones(
                        (end - begin, 1),
                        dtype=torch.float64,
                        device=device,
                    ),
                ),
                dim=1,
            )
            cardinal = torch.bmm(
                evaluation_rows[:, None, :],
                systems.inverses[patch_ids, :, : systems.node_count],
            )[:, 0, :]
            if not bool(torch.isfinite(cardinal).all()):
                raise DiagnosticError("local cardinal weights are nonfinite")
            cardinal_sum = cardinal.sum(dim=1)
            pair_error = float(torch.max(torch.abs(cardinal_sum - 1.0)).item())
            max_pair_cardinal_error = max(max_pair_cardinal_error, pair_error)
            local_value = torch.sum(cardinal * values[node_ids], dim=1)
            raw_weight = wendland_c2(
                torch.as_tensor(
                    distances_np[entry_ids] / radii_np[entry_ids],
                    dtype=torch.float64,
                    device=device,
                )
            )
            if not bool((raw_weight > 0.0).all()):
                raise DiagnosticError("active topology pair has nonpositive raw PU weight")
            denominator.index_add_(0, query_ids, raw_weight)
            value_numerator.index_add_(0, query_ids, raw_weight * local_value)
            constant_numerator.index_add_(0, query_ids, raw_weight * cardinal_sum)
            if collapse_global_weights:
                coo_rows.append(
                    np.repeat(query_ids_np[entry_ids], systems.node_count)
                )
                coo_columns.append(node_ids.detach().cpu().numpy().reshape(-1))
                coo_values.append(
                    (raw_weight[:, None] * cardinal).detach().cpu().numpy().reshape(-1)
                )
    covered_tensor = denominator > 0.0
    covered = covered_tensor.detach().cpu().numpy()
    output = torch.full_like(denominator, torch.nan)
    constant = torch.full_like(denominator, torch.nan)
    output[covered_tensor] = (
        value_numerator[covered_tensor] / denominator[covered_tensor]
    )
    constant[covered_tensor] = (
        constant_numerator[covered_tensor] / denominator[covered_tensor]
    )
    output_np = output.detach().cpu().numpy()
    constant_np = constant.detach().cpu().numpy()
    global_weights: csr_matrix | None = None
    global_report: dict[str, Any] | None = None
    collapsed_field_error: float | None = None
    if collapse_global_weights:
        rows = np.concatenate(coo_rows) if coo_rows else np.empty((0,), dtype=np.int64)
        columns = (
            np.concatenate(coo_columns)
            if coo_columns
            else np.empty((0,), dtype=np.int64)
        )
        unnormalized = (
            np.concatenate(coo_values)
            if coo_values
            else np.empty((0,), dtype=np.float64)
        )
        denominator_np = denominator.detach().cpu().numpy()
        normalized = np.divide(
            unnormalized,
            denominator_np[rows],
            out=np.zeros_like(unnormalized),
            where=denominator_np[rows] > 0.0,
        )
        global_weights = coo_matrix(
            (normalized, (rows, columns)),
            shape=(query_count, systems.guide_points.shape[0]),
        ).tocsr()
        global_weights.sum_duplicates()
        global_weights.sort_indices()
        global_weights.eliminate_zeros()
        global_report = _global_weight_report(global_weights)
        global_row_sum = np.asarray(global_weights.sum(axis=1)).reshape(-1)
        if not np.allclose(
            global_row_sum[covered],
            constant_np[covered],
            rtol=0.0,
            atol=MAX_FIELD_ERROR,
        ):
            raise DiagnosticError("collapsed global weight sums disagree with field algebra")
        collapsed_values = np.asarray(
            global_weights @ values.detach().cpu().numpy()
        ).reshape(-1)
        collapsed_difference = np.abs(collapsed_values[covered] - output_np[covered])
        collapsed_field_error = (
            float(collapsed_difference.max())
            if collapsed_difference.size
            else 0.0
        )
        if (
            not math.isfinite(collapsed_field_error)
            or collapsed_field_error > MAX_FIELD_ERROR
        ):
            raise DiagnosticError(
                "collapsed global-weight CSR does not reproduce direct field values "
                f"within {MAX_FIELD_ERROR:g}"
            )
    cardinal_np = constant_np.copy()
    finite_covered = bool(np.isfinite(output_np[covered]).all()) and bool(
        np.isfinite(constant_np[covered]).all()
    )
    return RaggedFieldResult(
        values=np.ascontiguousarray(output_np, dtype=np.float64),
        constant_values=np.ascontiguousarray(constant_np, dtype=np.float64),
        cardinal_row_sums=np.ascontiguousarray(cardinal_np, dtype=np.float64),
        covered=np.ascontiguousarray(covered, dtype=np.bool_),
        global_weights=global_weights,
        report={
            "query_count": query_count,
            "candidate_pair_count": int(patch_ids_np.size),
            "active_pair_count": int(active_entry_ids.size),
            "retained_zero_weight_candidate_count": int(
                patch_ids_np.size - active_entry_ids.size
            ),
            "covered_query_count": int(covered.sum()),
            "uncovered_query_count": int((~covered).sum()),
            "uncovered_query_ids": np.flatnonzero(~covered)[:64].tolist(),
            "finite_covered_values": finite_covered,
            "max_local_pair_cardinal_sum_error": max_pair_cardinal_error,
            "max_global_cardinal_row_sum_error": float(
                np.max(np.abs(cardinal_np[covered] - 1.0))
            )
            if bool(covered.any())
            else None,
            "pair_chunk_size": chunk_size,
            "seconds": float(time.perf_counter() - started),
            "global_weights": global_report,
            "collapsed_CSR_field_reconstruction_max_abs_error": (
                collapsed_field_error
            ),
            "collapsed_CSR_field_reconstruction_limit": MAX_FIELD_ERROR,
            "signed_cardinal_weights_clamped_or_renormalized": False,
            "PU_normalization_only": True,
        },
    )


def extract_render_geometry(model: Any) -> RenderGeometry:
    with torch.no_grad():
        face_ids_tensor = model.face_ids.detach().long()
        bary = torch.softmax(model.bary_logits.detach(), dim=-1)
        triangles = model.vertices[model.faces[face_ids_tensor]]
        points = (triangles * bary[:, :, None]).sum(dim=1)
        _, normals, model_points = model.roots_and_normals()
        if not bool(torch.equal(points, model_points.detach())):
            raise DiagnosticError("current softmax root points differ from model roots")
        points_np = points.to(dtype=torch.float64).cpu().numpy()
        normals_np = normals.detach().to(dtype=torch.float64).cpu().numpy()
        face_ids = face_ids_tensor.cpu().numpy().astype(np.int64, copy=False)
        bary_np = bary.to(dtype=torch.float64).cpu().numpy()
    if points_np.ndim != 2 or points_np.shape[1] != 3:
        raise DiagnosticError("render root points must have shape [N, 3]")
    if normals_np.shape != points_np.shape or bary_np.shape != points_np.shape:
        raise DiagnosticError("render root normal/barycentric shapes are invalid")
    if not np.isfinite(points_np).all() or not np.isfinite(normals_np).all():
        raise DiagnosticError("render root geometry contains nonfinite values")
    if not np.isfinite(bary_np).all() or np.any(bary_np < 0.0):
        raise DiagnosticError("render root softmax barycentrics are invalid")
    bary_sum_error = np.abs(bary_np.sum(axis=1) - 1.0)
    if np.any(bary_sum_error > 1.0e-6):
        raise DiagnosticError("render root barycentrics do not sum to one")
    return RenderGeometry(
        points_local=np.ascontiguousarray(points_np, dtype=np.float64),
        normals_local=np.ascontiguousarray(normals_np, dtype=np.float64),
        face_ids=np.ascontiguousarray(face_ids, dtype=np.int64),
        barycentric=np.ascontiguousarray(bary_np, dtype=np.float64),
        report={
            "population_count": int(points_np.shape[0]),
            "point_identity": array_identity(points_np),
            "face_id_identity": array_identity(face_ids),
            "barycentric_identity": array_identity(bary_np),
            "barycentric_sum_error_max": float(bary_sum_error.max()),
            "construction": "torch.softmax(bary_logits)_on_fixed_face_ids",
        },
    )


def decode_guide_lengths(model: Any, decoder: Any) -> np.ndarray:
    if model.guide_length_raw is None:
        raise DiagnosticError("checkpoint has no primary guide length field")
    with torch.no_grad():
        lengths = decoder(
            model.guide_length_raw,
            model.guide_length_reference,
        ).detach().reshape(-1).to(dtype=torch.float64).cpu().numpy()
    lengths = np.ascontiguousarray(lengths, dtype=np.float64)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise DiagnosticError("decoded guide lengths must be finite and positive")
    return lengths


def evaluate_legacy_selected_lengths(
    model: Any,
    points_local: np.ndarray,
    normals_local: np.ndarray,
    face_ids: np.ndarray,
    selected_root_ids: np.ndarray,
) -> tuple[np.ndarray, SurfaceSupport, dict[str, Any]]:
    device = model.guide_points_local.device
    dtype = model.guide_points_local.dtype
    selected_ids = np.asarray(selected_root_ids, dtype=np.int64).reshape(-1)
    canonical_support = model.guide_interpolation_support()
    population_count = int(model.face_ids.shape[0])
    if canonical_support.query_count != population_count:
        raise DiagnosticError("canonical cached support does not span render population")
    if selected_ids.shape != (points_local.shape[0],):
        raise DiagnosticError("selected IDs do not match legacy query count")
    if np.any(selected_ids < 0) or np.any(selected_ids >= population_count):
        raise DiagnosticError("selected ID is outside canonical cached support")
    selected_tensor = torch.as_tensor(selected_ids, dtype=torch.long, device=device)
    selected_support = SurfaceSupport(
        indices=canonical_support.indices.index_select(0, selected_tensor).detach(),
        vertex_path_distances=canonical_support.vertex_path_distances.index_select(
            0,
            selected_tensor,
        ).detach(),
        report={
            **canonical_support.report,
            "provenance": "canonical_full_render_support_sliced_by_selected_root_ids",
            "canonical_query_count": population_count,
            "selected_query_count": int(selected_ids.size),
            "selected_root_ids_sha256": array_identity(selected_ids)["sha256"],
        },
    )
    with torch.no_grad():
        points = torch.as_tensor(points_local, dtype=dtype, device=device)
        normals = torch.as_tensor(normals_local, dtype=dtype, device=device)
        query_faces = torch.as_tensor(face_ids, dtype=torch.long, device=device)
        _, tangents, bitangents = model.tangent_frames_for_face_ids(query_faces)
        controls, _ = model.sample_guide_controls(
            points,
            query_faces,
            normals,
            tangents,
            bitangents,
            support=selected_support,
        )
        if "length" not in controls:
            raise DiagnosticError("legacy canonical sampling returned no length")
        lengths = controls["length"].detach().reshape(-1).to(
            dtype=torch.float64
        ).cpu().numpy()
    lengths = np.ascontiguousarray(lengths, dtype=np.float64)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise DiagnosticError("legacy selected lengths must be finite and positive")
    if lengths.shape != (selected_ids.size,):
        raise DiagnosticError("legacy selected lengths have the wrong shape")
    return lengths, selected_support, dict(selected_support.report)


def select_interior_edge_queries(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    count: int = INTERIOR_EDGE_PROBE_COUNT,
    seed: int = SELECTION_SEED,
) -> InteriorEdgeQueries:
    vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    requested = int(count)
    if requested <= 0:
        raise ValueError("interior edge probe count must be positive")
    edge_pairs = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    edge_pairs = np.sort(edge_pairs, axis=1)
    face_ids = np.tile(np.arange(faces.shape[0], dtype=np.int64), 3)
    order = np.lexsort((face_ids, edge_pairs[:, 1], edge_pairs[:, 0]))
    sorted_edges = edge_pairs[order]
    sorted_faces = face_ids[order]
    change = np.ones((sorted_edges.shape[0],), dtype=bool)
    change[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], sorted_edges.shape[0])
    multiplicity = ends - starts
    interior_group_ids = np.flatnonzero(multiplicity == 2)
    if interior_group_ids.size < requested:
        raise DiagnosticError(
            f"only {interior_group_ids.size} interior edges for {requested} probes"
        )
    ranked = sorted(
        (stable_root_rank(int(group_id), seed), int(group_id))
        for group_id in interior_group_ids.tolist()
    )
    selected_groups = np.asarray(
        sorted(group_id for _, group_id in ranked[:requested]),
        dtype=np.int64,
    )
    selected_edges = sorted_edges[starts[selected_groups]]
    adjacent_faces = np.stack(
        (
            sorted_faces[starts[selected_groups]],
            sorted_faces[starts[selected_groups] + 1],
        ),
        axis=1,
    )
    query_faces = adjacent_faces.reshape(-1)
    query_bary = np.zeros((requested * 2, 3), dtype=np.float64)
    query_points = np.repeat(
        ((vertices[selected_edges[:, 0]] + vertices[selected_edges[:, 1]]) * 0.5),
        2,
        axis=0,
    )
    for edge_id, (u, v) in enumerate(selected_edges.tolist()):
        for representation in range(2):
            query_id = edge_id * 2 + representation
            face = faces[query_faces[query_id]]
            positions_u = np.flatnonzero(face == u)
            positions_v = np.flatnonzero(face == v)
            if positions_u.size != 1 or positions_v.size != 1:
                raise DiagnosticError("interior edge is not present in adjacent face")
            query_bary[query_id, int(positions_u[0])] = 0.5
            query_bary[query_id, int(positions_v[0])] = 0.5
    return InteriorEdgeQueries(
        edge_vertices=np.ascontiguousarray(selected_edges, dtype=np.int64),
        face_ids=np.ascontiguousarray(query_faces, dtype=np.int64),
        barycentric=query_bary,
        points_local=np.ascontiguousarray(query_points, dtype=np.float64),
        report={
            "requested_count": requested,
            "available_interior_edge_count": int(interior_group_ids.size),
            "selection_seed": int(seed),
            "selection": "SplitMix64_rank_of_lexicographic_interior_edge_group_id",
            "edge_vertex_identity": array_identity(selected_edges),
            "face_id_identity": array_identity(query_faces),
        },
    )


def _max_abs_error(
    values: np.ndarray,
    target: float | np.ndarray,
) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    difference = np.abs(array - target)
    if not np.isfinite(difference).all():
        return None
    return float(difference.max()) if difference.size else 0.0


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "finite_count": 0,
            "nonfinite_count": 0,
            "min": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    finite = np.isfinite(array)
    finite_values = array[finite]
    if finite_values.size == 0:
        return {
            "count": int(array.size),
            "finite_count": 0,
            "nonfinite_count": int(array.size),
            "min": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(array.size),
        "finite_count": int(finite_values.size),
        "nonfinite_count": int((~finite).sum()),
        "min": float(finite_values.min()),
        "mean": float(finite_values.mean(dtype=np.float64)),
        "p95": float(np.quantile(finite_values, 0.95, method="linear")),
        "max": float(finite_values.max()),
    }


def _finite_max_or_none(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        return None
    return float(array.max())


def evaluate_hard_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    required = (
        "identities_and_c1_state_exact",
        "all_query_groups_covered",
        "all_query_groups_finite",
        "guide_site_max_abs_self_error",
        "constant_reproduction_max_abs_error",
        "local_pair_cardinal_sum_max_abs_error",
        "global_cardinal_row_sum_max_abs_error",
        "edge_midpoint_cross_face_max_abs_difference",
        "all_evaluated_lengths_finite_positive",
        "required_root_present",
        "required_root_covered",
        "required_root_finite_positive",
        "inverse_residual_max",
        "field_evaluation_seconds",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise DiagnosticError("C2 hard-gate metrics are missing: " + ", ".join(missing))

    def flag(name: str) -> bool:
        return metrics[name] is True or isinstance(metrics[name], np.bool_) and bool(metrics[name])

    def at_most(name: str, limit: float) -> bool:
        value = metrics[name]
        return isinstance(value, (int, float, np.integer, np.floating)) and bool(
            math.isfinite(float(value)) and float(value) <= limit
        )

    gates = {
        "identities_and_c1_state_exact": flag("identities_and_c1_state_exact"),
        "all_query_groups_covered_and_finite": flag("all_query_groups_covered")
        and flag("all_query_groups_finite"),
        "guide_site_self_error_at_most_1e-10": at_most(
            "guide_site_max_abs_self_error",
            MAX_FIELD_ERROR,
        ),
        "constant_reproduction_at_most_1e-10": at_most(
            "constant_reproduction_max_abs_error",
            MAX_FIELD_ERROR,
        ),
        "local_and_global_cardinal_error_at_most_1e-10": at_most(
            "local_pair_cardinal_sum_max_abs_error",
            MAX_FIELD_ERROR,
        )
        and at_most(
            "global_cardinal_row_sum_max_abs_error",
            MAX_FIELD_ERROR,
        ),
        "edge_midpoint_cross_face_difference_at_most_1e-10": at_most(
            "edge_midpoint_cross_face_max_abs_difference",
            MAX_FIELD_ERROR,
        ),
        "all_evaluated_lengths_finite_positive": flag(
            "all_evaluated_lengths_finite_positive"
        ),
        "required_root_431701_present_covered_finite_positive": flag(
            "required_root_present"
        )
        and flag("required_root_covered")
        and flag("required_root_finite_positive"),
        "inverse_residual_at_most_1e-10": at_most(
            "inverse_residual_max",
            MAX_INVERSE_RESIDUAL,
        ),
        "field_evaluation_within_600_seconds": at_most(
            "field_evaluation_seconds",
            EVALUATION_TIMEOUT_BUDGET_SECONDS,
        ),
    }
    return gates


def aggregate_hard_gates(gates: Mapping[str, Any]) -> bool:
    if set(gates) != set(HARD_GATE_KEYS):
        raise DiagnosticError("C2 hard-gate identities do not match predeclared gates")
    if not all(isinstance(value, (bool, np.bool_)) for value in gates.values()):
        raise DiagnosticError("C2 hard-gate values must be boolean")
    return bool(all(bool(value) for value in gates.values()))


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


def write_atomic_npy(
    path: str | os.PathLike[str],
    values: Any,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    return c1.write_atomic_npy(path, values, overwrite=overwrite)


def write_deterministic_json(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    c1.write_deterministic_json(path, payload, overwrite=overwrite)


def save_outputs_staged(
    output_dir: str | os.PathLike[str],
    report: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {output}")
    if tuple(arrays) != ARRAY_NAMES:
        raise DiagnosticError("C2 output array names/order do not match contract")
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.r084-c2-stage.",
            dir=output.parent,
        )
    )
    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.r084-c2-backup.",
            dir=output.parent,
        )
    )
    identities: dict[str, Any] = {}
    try:
        for name, values in arrays.items():
            identities[name] = write_atomic_npy(
                stage / name,
                values,
                overwrite=False,
            )
        report["artifacts"] = identities
        write_deterministic_json(stage / REPORT_NAME, report, overwrite=False)
        manifest_entries = {
            name: {
                "bytes": int(identity["bytes"]),
                "sha256": str(identity["sha256"]),
            }
            for name, identity in identities.items()
        }
        manifest_entries[REPORT_NAME] = {
            "bytes": int((stage / REPORT_NAME).stat().st_size),
            "sha256": sha256_file(stage / REPORT_NAME),
        }
        manifest = {
            "schema": SCHEMA,
            "algorithm": "sha256",
            "artifacts": manifest_entries,
        }
        write_deterministic_json(stage / MANIFEST_NAME, manifest, overwrite=False)
        publication_order = (*ARRAY_NAMES, REPORT_NAME, MANIFEST_NAME)
        backed_up: list[str] = []
        published: list[str] = []
        try:
            if overwrite:
                for name in publication_order:
                    destination = output / name
                    if destination.exists() or destination.is_symlink():
                        os.replace(destination, backup / name)
                        backed_up.append(name)
            for name in publication_order:
                _publish_temporary_file(
                    stage / name,
                    output / name,
                    overwrite=False,
                )
                published.append(name)
        except BaseException as publication_error:
            rollback_errors: list[str] = []
            for name in reversed(published):
                destination = output / name
                try:
                    if destination.is_dir() and not destination.is_symlink():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink(missing_ok=True)
                except OSError as error:
                    rollback_errors.append(f"remove {name}: {error}")
            for name in backed_up:
                try:
                    os.replace(backup / name, output / name)
                except OSError as error:
                    rollback_errors.append(f"restore {name}: {error}")
            if rollback_errors:
                raise DiagnosticError(
                    "C2 publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from publication_error
            raise
        return manifest
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(backup, ignore_errors=True)


def _import_stage1() -> tuple[Any, Any]:
    from tools.train_white_tiger_stage1 import (  # type: ignore[import-not-found]
        decode_positive_asinh_ratio,
        load_stage1_checkpoint_model,
    )

    return load_stage1_checkpoint_model, decode_positive_asinh_ratio


def _versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _source_identities() -> dict[str, Any]:
    paths = {
        "diagnostic": Path(__file__).resolve(),
        "phase_c1_diagnostic": Path(c1.__file__).resolve(),
        "rbf_partition_of_unity": PROJECT_ROOT / "anigroom" / "rbf_partition_of_unity.py",
        "rbf_topology_cover": PROJECT_ROOT / "anigroom" / "rbf_topology_cover.py",
    }
    return {
        name: {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


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


def run_fixed_checkpoint_diagnostic(
    *,
    checkpoint: str | os.PathLike[str],
    c1_state_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    device: str | torch.device,
    expected_checkpoint_sha256: str,
    expected_iteration: int,
    expected_source_commit: str,
    expected_c1_source_commit: str,
    expected_c1_report_sha256: str,
    expected_c1_manifest_sha256: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    checkpoint_path = _path_file(checkpoint, "checkpoint")
    output = prepare_output_dir(output_dir, overwrite=overwrite)
    expected_checkpoint = normalize_sha256(
        expected_checkpoint_sha256,
        "expected checkpoint SHA256",
    )
    expected_source = normalize_git_commit(expected_source_commit)
    expected_c1_source = normalize_git_commit(expected_c1_source_commit)
    if isinstance(expected_iteration, bool) or not isinstance(
        expected_iteration,
        (int, np.integer),
    ):
        raise DiagnosticError("expected-iteration must be a non-negative integer")
    expected_iter = int(expected_iteration)
    if expected_iter < 0:
        raise DiagnosticError("expected-iteration must be a non-negative integer")
    timings: dict[str, float] = {}
    rss: dict[str, int] = {"start": c1._rss_bytes()}

    source_started = time.perf_counter()
    source_git = get_clean_source_git_identity(PROJECT_ROOT)
    source_expectation = c1.validate_source_identity(source_git["head"], expected_source)
    timings["source_identity"] = float(time.perf_counter() - source_started)
    checkpoint_hash_started = time.perf_counter()
    checkpoint_hash = sha256_file(checkpoint_path)
    timings["checkpoint_sha256"] = float(
        time.perf_counter() - checkpoint_hash_started
    )
    if checkpoint_hash != expected_checkpoint:
        raise DiagnosticError(
            f"checkpoint SHA256 mismatch: expected {expected_checkpoint}, got {checkpoint_hash}"
        )
    c1_verify_started = time.perf_counter()
    verified = verify_c1_artifacts(
        c1_state_dir,
        expected_report_sha256=expected_c1_report_sha256,
        expected_manifest_sha256=expected_c1_manifest_sha256,
    )
    timings["c1_manifest_and_artifact_hash_verification"] = float(
        time.perf_counter() - c1_verify_started
    )
    c1_source_binding = validate_c1_source_contract(
        verified,
        expected_c1_source_commit=expected_c1_source,
    )
    target_device = c1._resolve_device(device)
    if target_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target_device)
    loader, decoder = _import_stage1()
    load_started = time.perf_counter()
    model, config, checkpoint_data = loader(checkpoint_path, target_device)
    timings["checkpoint_model_load"] = float(time.perf_counter() - load_started)
    checkpoint_identity = c1.validate_checkpoint_identity(
        checkpoint_hash,
        expected_checkpoint,
        checkpoint_data,
        expected_iter,
    )
    checkpoint_arrays = c1.extract_checkpoint_topology_arrays(model)
    c1_binding = validate_c1_checkpoint_binding(
        verified,
        checkpoint_arrays,
        checkpoint_sha256=checkpoint_hash,
        checkpoint_iteration=expected_iter,
        expected_c1_source_commit=expected_c1_source,
    )
    state_load_started = time.perf_counter()
    state = load_c1_state(verified, checkpoint_arrays)
    timings["c1_state_load_reconstruct_bind"] = float(
        time.perf_counter() - state_load_started
    )
    guide_lengths = decode_guide_lengths(model, decoder)
    if guide_lengths.shape != (EXPECTED_GUIDE_COUNT,):
        raise DiagnosticError("decoded guide length count is not exactly 4500")
    render = extract_render_geometry(model)
    if render.points_local.shape[0] != EXPECTED_RENDER_POPULATION:
        raise DiagnosticError(
            "render population mismatch: expected "
            f"{EXPECTED_RENDER_POPULATION}, got {render.points_local.shape[0]}"
        )
    selected_ids = select_render_root_ids(render.points_local.shape[0])
    if selected_ids.shape != (RENDER_ROOT_COUNT,):
        raise DiagnosticError("render root selection did not produce exactly 4096 IDs")
    required_positions = np.flatnonzero(selected_ids == REQUIRED_ROOT_ID)
    required_present = bool(required_positions.size == 1)
    selected_identity = array_identity(selected_ids)
    if selected_identity["sha256"] != EXPECTED_SELECTED_IDS_SHA256:
        raise DiagnosticError("selected render-root ID SHA256 differs from frozen contract")
    if not required_present or int(required_positions[0]) != EXPECTED_REQUIRED_ROOT_ROW:
        raise DiagnosticError(
            "required root 431701 is not at frozen selected row 3552"
        )
    edge_queries = select_interior_edge_queries(
        checkpoint_arrays.vertices,
        checkpoint_arrays.faces,
    )
    rss["after_checkpoint_and_state_load"] = c1._rss_bytes()

    evaluation_started = time.perf_counter()
    systems = build_batched_patch_systems(
        checkpoint_arrays.guide_points_local,
        state.patch_nodes,
        device=target_device,
    )
    guide_ragged = evaluate_query_topology_distances(
        state.inputs,
        state.face_cover,
        checkpoint_arrays.guide_face_ids,
        checkpoint_arrays.guide_barycentric,
    )
    guide_ragged_binding = validate_ragged_binding(
        guide_ragged,
        state,
        checkpoint_arrays.guide_face_ids,
        checkpoint_arrays.guide_barycentric,
    )
    guide_result = evaluate_ragged_field(
        checkpoint_arrays.guide_points_local,
        guide_ragged,
        systems,
        guide_lengths,
    )
    selected_points = render.points_local[selected_ids]
    selected_normals = render.normals_local[selected_ids]
    selected_faces = render.face_ids[selected_ids]
    selected_bary = render.barycentric[selected_ids]
    selected_ragged = evaluate_query_topology_distances(
        state.inputs,
        state.face_cover,
        selected_faces,
        selected_bary,
    )
    selected_ragged_binding = validate_ragged_binding(
        selected_ragged,
        state,
        selected_faces,
        selected_bary,
    )
    selected_result = evaluate_ragged_field(
        selected_points,
        selected_ragged,
        systems,
        guide_lengths,
        collapse_global_weights=True,
    )
    edge_ragged = evaluate_query_topology_distances(
        state.inputs,
        state.face_cover,
        edge_queries.face_ids,
        edge_queries.barycentric,
    )
    edge_ragged_binding = validate_ragged_binding(
        edge_ragged,
        state,
        edge_queries.face_ids,
        edge_queries.barycentric,
    )
    edge_result = evaluate_ragged_field(
        edge_queries.points_local,
        edge_ragged,
        systems,
        guide_lengths,
    )
    legacy_started = time.perf_counter()
    legacy_lengths, legacy_support, legacy_report = evaluate_legacy_selected_lengths(
        model,
        selected_points,
        selected_normals,
        selected_faces,
        selected_ids,
    )
    timings["legacy_canonical_cached_support_slice"] = float(
        time.perf_counter() - legacy_started
    )
    field_evaluation_seconds = float(time.perf_counter() - evaluation_started)
    timings["total_field_evaluation_excluding_checkpoint_and_state_load"] = (
        field_evaluation_seconds
    )
    rss["after_field_evaluation"] = c1._rss_bytes()

    if selected_result.global_weights is None:
        raise DiagnosticError("selected global weight CSR was not produced")
    global_weights = selected_result.global_weights
    guide_errors = guide_result.values - guide_lengths
    edge_pair_values = edge_result.values.reshape(INTERIOR_EDGE_PROBE_COUNT, 2)
    edge_difference = np.abs(edge_pair_values[:, 0] - edge_pair_values[:, 1])
    constant_errors = np.concatenate(
        (
            np.abs(guide_result.constant_values - 1.0),
            np.abs(selected_result.constant_values - 1.0),
            np.abs(edge_result.constant_values - 1.0),
        )
    )
    cardinal_errors = np.concatenate(
        (
            np.abs(guide_result.cardinal_row_sums - 1.0),
            np.abs(selected_result.cardinal_row_sums - 1.0),
            np.abs(edge_result.cardinal_row_sums - 1.0),
        )
    )
    all_covered = bool(
        guide_result.covered.all()
        and selected_result.covered.all()
        and edge_result.covered.all()
    )
    all_finite = bool(
        np.isfinite(guide_result.values).all()
        and np.isfinite(selected_result.values).all()
        and np.isfinite(edge_result.values).all()
        and np.isfinite(constant_errors).all()
        and np.isfinite(cardinal_errors).all()
    )
    all_evaluated_positive = bool(
        np.isfinite(guide_result.values).all()
        and np.all(guide_result.values > 0.0)
        and np.isfinite(selected_result.values).all()
        and np.all(selected_result.values > 0.0)
        and np.isfinite(edge_result.values).all()
        and np.all(edge_result.values > 0.0)
    )
    required_row = int(required_positions[0]) if required_present else None
    required_covered = bool(
        required_present and selected_result.covered[int(required_row)]
    )
    required_value_raw = (
        float(selected_result.values[int(required_row)])
        if required_present
        else math.nan
    )
    required_positive = bool(
        math.isfinite(required_value_raw) and required_value_raw > 0.0
    )
    required_value = required_value_raw if math.isfinite(required_value_raw) else None
    local_pair_cardinal_error = max(
        float(guide_result.report["max_local_pair_cardinal_sum_error"]),
        float(selected_result.report["max_local_pair_cardinal_sum_error"]),
        float(edge_result.report["max_local_pair_cardinal_sum_error"]),
    )
    metrics = {
        "identities_and_c1_state_exact": True,
        "all_query_groups_covered": all_covered,
        "all_query_groups_finite": all_finite,
        "guide_site_max_abs_self_error": _max_abs_error(guide_result.values, guide_lengths),
        "constant_reproduction_max_abs_error": _finite_max_or_none(constant_errors),
        "local_pair_cardinal_sum_max_abs_error": local_pair_cardinal_error,
        "global_cardinal_row_sum_max_abs_error": _finite_max_or_none(
            cardinal_errors
        ),
        "edge_midpoint_cross_face_max_abs_difference": _finite_max_or_none(
            edge_difference
        ),
        "all_evaluated_lengths_finite_positive": all_evaluated_positive,
        "required_root_present": required_present,
        "required_root_covered": required_covered,
        "required_root_finite_positive": required_positive,
        "inverse_residual_max": float(systems.report["inverse_residual_max"]),
        "field_evaluation_seconds": field_evaluation_seconds,
    }
    gates = evaluate_hard_gates(metrics)
    accepted = aggregate_hard_gates(gates)
    absolute_difference = np.abs(selected_result.values - legacy_lengths)
    relative_difference = absolute_difference / np.maximum(np.abs(legacy_lengths), 1.0e-12)
    arrays = {
        "selected_root_ids.npy": selected_ids,
        "candidate_lengths.npy": selected_result.values,
        "legacy_lengths.npy": legacy_lengths,
        "selected_global_weight_indptr.npy": global_weights.indptr,
        "selected_global_weight_indices.npy": global_weights.indices,
        "selected_global_weight_data.npy": global_weights.data,
        "guide_site_errors.npy": guide_errors,
        "edge_pair_values.npy": edge_pair_values,
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "accepted": accepted,
        "constants": {
            "render_root_count": RENDER_ROOT_COUNT,
            "selection_seed": SELECTION_SEED,
            "interior_edge_probe_count": INTERIOR_EDGE_PROBE_COUNT,
            "required_root_id": REQUIRED_ROOT_ID,
            "expected_render_population": EXPECTED_RENDER_POPULATION,
            "expected_selected_ids_sha256": EXPECTED_SELECTED_IDS_SHA256,
            "expected_required_root_row": EXPECTED_REQUIRED_ROOT_ROW,
            "evaluation_timeout_budget_seconds": EVALUATION_TIMEOUT_BUDGET_SECONDS,
        },
        "hard_gates": gates,
        "hard_gate_metrics": metrics,
        "identities": {
            "source": {**source_git, "expectation": source_expectation},
            "checkpoint": checkpoint_identity,
            "c1": verified.identities,
            "c1_source_binding": c1_source_binding,
            "c1_checkpoint_binding": c1_binding,
            "source_files": _source_identities(),
            "guide_lengths": array_identity(guide_lengths),
            "config_type": type(config).__name__,
            "model_type": type(model).__name__,
        },
        "c1_state": state.report,
        "patch_systems": systems.report,
        "render_geometry": render.report,
        "selection": {
            "selected_root_ids": selected_identity,
            "expected_population": EXPECTED_RENDER_POPULATION,
            "expected_selected_ids_sha256": EXPECTED_SELECTED_IDS_SHA256,
            "expected_required_root_row": EXPECTED_REQUIRED_ROOT_ROW,
            "required_root_selected_row": required_row,
            "required_root_candidate_length": required_value,
        },
        "interior_edges": edge_queries.report,
        "query_groups": {
            "guide_sites": {
                **guide_result.report,
                "ragged_binding": guide_ragged_binding,
            },
            "selected_render_roots": {
                **selected_result.report,
                "ragged_binding": selected_ragged_binding,
            },
            "edge_midpoint_representations": {
                **edge_result.report,
                "ragged_binding": edge_ragged_binding,
            },
        },
        "errors": {
            "guide_site": _summary(np.abs(guide_errors)),
            "constant": _summary(constant_errors),
            "global_cardinal_row_sum": _summary(cardinal_errors),
            "edge_cross_face": _summary(edge_difference),
        },
        "candidate_vs_canonical_legacy_descriptive_only": {
            "is_hard_gate": False,
            "absolute_difference": _summary(absolute_difference),
            "relative_difference": _summary(relative_difference),
            "legacy_support": legacy_report,
            "legacy_support_query_count": legacy_support.query_count,
        },
        "signed_global_weights_descriptive_only": {
            "is_hard_gate": False,
            **(selected_result.report["global_weights"] or {}),
        },
        "timings_seconds": timings,
        "memory": {
            "rss_stage_bytes": rss,
            "max_observed_rss_bytes": int(max(rss.values())),
            "cuda": _cuda_memory(target_device),
        },
        "versions": _versions(),
        "evaluation_budget": {
            "measured_seconds": field_evaluation_seconds,
            "limit_seconds": EVALUATION_TIMEOUT_BUDGET_SECONDS,
            "within_limit": bool(
                field_evaluation_seconds <= EVALUATION_TIMEOUT_BUDGET_SECONDS
            ),
            "enforcement": "outer_launcher_kill_not_internal_fallback",
        },
        "no_training": True,
        "no_rendering_or_images": True,
        "no_checkpoint_mutation": True,
    }
    report["timings_seconds"]["total_before_output"] = float(
        time.perf_counter() - total_started
    )
    manifest = save_outputs_staged(
        output,
        report,
        arrays,
        overwrite=overwrite,
    )
    print(
        f"R084_C2_FINAL accepted={str(accepted).lower()} "
        f"selected_roots={selected_ids.size} required_root={REQUIRED_ROOT_ID} "
        f"field_seconds={field_evaluation_seconds:.6f} "
        f"artifacts={len(manifest['artifacts'])}",
        flush=True,
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="R084 Phase C2 fixed-checkpoint RBF-PU length subset diagnostic."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--c1-state-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-iteration", required=True, type=int)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-c1-source-commit", required=True)
    parser.add_argument("--expected-c1-report-sha256", required=True)
    parser.add_argument("--expected-c1-manifest-sha256", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        run_fixed_checkpoint_diagnostic(
            checkpoint=args.checkpoint,
            c1_state_dir=args.c1_state_dir,
            output_dir=args.output_dir,
            device=args.device,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_iteration=args.expected_iteration,
            expected_source_commit=args.expected_source_commit,
            expected_c1_source_commit=args.expected_c1_source_commit,
            expected_c1_report_sha256=args.expected_c1_report_sha256,
            expected_c1_manifest_sha256=args.expected_c1_manifest_sha256,
            overwrite=bool(args.overwrite),
        )
    except Exception as error:
        print(f"R084_C2_ERROR={error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

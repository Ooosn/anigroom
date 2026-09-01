from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch

from anigroom.rbf_partition_of_unity import (
    build_augmented_system,
    local_cardinal_weights,
    wendland_c2,
)
from anigroom.rbf_topology_cover import (
    FacePatchCover,
    PatchGuideDistanceMatrix,
    PatchNodeCover,
    RaggedQueryTopologyDistances,
    VertexPatchCover,
    build_face_patch_candidate_counts,
    build_vertex_patch_active_distances,
    compute_patch_guide_site_distances,
    evaluate_query_topology_distances,
    select_patch_radii_and_nodes,
    validate_topology_cover_inputs,
)
from anigroom.surface_interpolation import SurfaceSupport
from tools import diagnose_rbf_partition_cover as c1
from tools import diagnose_rbf_partition_length_subset as diagnostic


def _line_state() -> tuple[np.ndarray, diagnostic.LoadedC1State]:
    count = 5
    vertices = np.stack(
        (
            np.arange(count, dtype=np.float64),
            np.zeros((count,), dtype=np.float64),
            np.zeros((count,), dtype=np.float64),
        ),
        axis=1,
    )
    faces = np.asarray([[0, 1, 2], [2, 3, 4]], dtype=np.int64)
    guide_face_ids = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    guide_bary = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ids = np.arange(count, dtype=np.float64)
    distances = np.abs(ids[:, None] - ids[None, :])
    inputs = validate_topology_cover_inputs(
        distances,
        np.arange(count, dtype=np.int64),
        np.zeros((count,), dtype=np.float64),
        faces,
        guide_face_ids,
        guide_bary,
    )
    matrix = compute_patch_guide_site_distances(inputs)
    nodes = select_patch_radii_and_nodes(matrix, 3)
    vertex = build_vertex_patch_active_distances(inputs, nodes, vertex_chunk_size=1)
    face = build_face_patch_candidate_counts(inputs, vertex)
    return vertices, diagnostic.LoadedC1State(
        inputs=inputs,
        patch_guide_distances=matrix,
        patch_nodes=nodes,
        vertex_cover=vertex,
        face_cover=face,
        report={"passed": True},
    )


def _systems_and_guide_query() -> tuple[
    np.ndarray,
    diagnostic.LoadedC1State,
    diagnostic.BatchedPatchSystems,
    RaggedQueryTopologyDistances,
]:
    points, state = _line_state()
    systems = diagnostic.build_batched_patch_systems(
        points,
        state.patch_nodes,
        device="cpu",
        system_chunk_size=2,
    )
    ragged = evaluate_query_topology_distances(
        state.inputs,
        state.face_cover,
        state.inputs.guide_face_ids,
        state.inputs.guide_barycentric,
        query_chunk_size=2,
        completeness_patch_chunk_size=2,
    )
    return points, state, systems, ragged


def _c1_report() -> dict[str, object]:
    rejected_gates = {name: True for name in c1.HARD_GATE_KEYS}
    rejected_gates[c1.HARD_GATE_KEYS[0]] = False
    return {
        "schema": c1.SCHEMA,
        "accepted": True,
        "selected_k": 32,
        "candidate_k_sequence": list(c1.CANDIDATE_K_SEQUENCE),
        "candidate_results": [
            *[
                {
                    "k": candidate_k,
                    "status": "rejected",
                    "hard_gates": dict(rejected_gates),
                    "all_hard_gates_passed": False,
                }
                for candidate_k in (8, 12, 16, 24)
            ],
            {
                "k": 32,
                "status": "passed",
                "hard_gates": {name: True for name in c1.HARD_GATE_KEYS},
                "all_hard_gates_passed": True,
            },
            {
                "k": 48,
                "status": "not_evaluated_after_first_pass",
                "all_hard_gates_passed": None,
            },
            {
                "k": 64,
                "status": "not_evaluated_after_first_pass",
                "all_hard_gates_passed": None,
            },
        ],
        "artifacts": {},
    }


def _write_verified_c1_fixture(
    root: Path,
    arrays: dict[str, np.ndarray],
) -> tuple[str, str]:
    root.mkdir()
    report = _c1_report()
    array_artifacts: dict[str, object] = {}
    for name in c1.STATE_ARRAY_NAMES:
        array_artifacts[name] = c1.write_atomic_npy(root / name, arrays[name])
    report["artifacts"] = array_artifacts
    c1.write_deterministic_json(root / c1.REPORT_NAME, report)
    manifest_entries = {
        name: {
            "bytes": int((root / name).stat().st_size),
            "sha256": c1.sha256_file(root / name),
        }
        for name in (*c1.STATE_ARRAY_NAMES, c1.REPORT_NAME)
    }
    manifest = {
        "schema": c1.SCHEMA,
        "algorithm": "sha256",
        "artifacts": manifest_entries,
    }
    c1.write_deterministic_json(root / c1.MANIFEST_NAME, manifest)
    return c1.sha256_file(root / c1.REPORT_NAME), c1.sha256_file(root / c1.MANIFEST_NAME)


def _synthetic_state_arrays(state: diagnostic.LoadedC1State) -> dict[str, np.ndarray]:
    return {
        "guide_distances.npy": state.inputs.guide_distances,
        "patch_guide_distances.npy": state.patch_guide_distances.values,
        "vertex_seed_guide_ids.npy": state.inputs.vertex_seed_guide_ids,
        "vertex_nearest_distances.npy": state.inputs.vertex_nearest_distances,
        "component_labels.npy": state.inputs.component_labels,
        "patch_radii.npy": state.patch_nodes.radii,
        "patch_node_indptr.npy": state.patch_nodes.node_distances.indptr,
        "patch_node_indices.npy": state.patch_nodes.node_distances.indices,
        "patch_node_distances.npy": state.patch_nodes.node_distances.data,
        "vertex_active_indptr.npy": state.vertex_cover.active_distances.indptr,
        "vertex_active_indices.npy": state.vertex_cover.active_distances.indices,
        "vertex_active_distances.npy": state.vertex_cover.active_distances.data,
        "face_candidate_indptr.npy": state.face_cover.candidate_counts.indptr,
        "face_candidate_indices.npy": state.face_cover.candidate_counts.indices,
        "face_candidate_counts.npy": state.face_cover.candidate_counts.data,
    }


def test_parser_requires_every_fixed_identity_input() -> None:
    parser = diagnostic.build_argument_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--c1-state-dir",
            "c1",
            "--output-dir",
            "out",
            "--device",
            "cuda",
            "--expected-checkpoint-sha256",
            "a" * 64,
            "--expected-iteration",
            "4000",
            "--expected-source-commit",
            "b" * 40,
            "--expected-c1-source-commit",
            "e" * 40,
            "--expected-c1-report-sha256",
            "c" * 64,
            "--expected-c1-manifest-sha256",
            "d" * 64,
        ]
    )
    assert args.device == "cuda"
    assert args.overwrite is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "checkpoint.pt"])


def test_exact_splitmix_selection_contains_required_root_fixture() -> None:
    selected = diagnostic.select_render_root_ids(
        diagnostic.EXPECTED_RENDER_POPULATION
    )
    assert selected.shape == (4096,)
    assert selected.tolist() == sorted(selected.tolist())
    assert len(set(selected.tolist())) == 4096
    assert diagnostic.REQUIRED_ROOT_ID in selected
    assert np.flatnonzero(selected == diagnostic.REQUIRED_ROOT_ID).tolist() == [3552]
    assert diagnostic.array_identity(selected)["sha256"] == (
        diagnostic.EXPECTED_SELECTED_IDS_SHA256
    )
    assert diagnostic.stable_root_rank(431701, 20260901) == 6709033549506372


def test_batched_evaluator_matches_per_patch_core_direct() -> None:
    points, state, systems, ragged = _systems_and_guide_query()
    values = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    query_id = 1
    begin, end = ragged.indptr[query_id : query_id + 2]
    one = RaggedQueryTopologyDistances(
        indptr=np.asarray([0, end - begin], dtype=np.int64),
        patch_ids=ragged.patch_ids[begin:end],
        distances=ragged.distances[begin:end],
        radii=ragged.radii[begin:end],
        patch_node_counts=ragged.patch_node_counts,
        report={},
    )
    result = diagnostic.evaluate_ragged_field(
        points[query_id : query_id + 1],
        one,
        systems,
        values,
        pair_chunk_size=1,
        collapse_global_weights=True,
    )
    numerator = 0.0
    denominator = 0.0
    for patch_id, distance, radius in zip(
        one.patch_ids.tolist(),
        one.distances.tolist(),
        one.radii.tolist(),
    ):
        if distance >= radius:
            continue
        row_begin, row_end = state.patch_nodes.node_distances.indptr[
            patch_id : patch_id + 2
        ]
        node_ids = state.patch_nodes.node_distances.indices[row_begin:row_end]
        sources = torch.as_tensor(points[node_ids], dtype=torch.float64)
        query = torch.as_tensor(points[query_id : query_id + 1], dtype=torch.float64)
        radius_tensor = torch.tensor(radius, dtype=torch.float64)
        system = build_augmented_system(sources, radius_tensor)
        cardinal = local_cardinal_weights(
            query,
            sources,
            radius_tensor,
            augmented_system=system,
        )[0].numpy()
        raw = float(wendland_c2(torch.tensor(distance / radius)).item())
        numerator += raw * float(cardinal @ values[node_ids])
        denominator += raw
    assert result.values[0] == pytest.approx(numerator / denominator, abs=1.0e-12)
    assert result.global_weights is not None
    collapsed_value = np.asarray(result.global_weights @ values).reshape(-1)[0]
    assert collapsed_value == pytest.approx(result.values[0], abs=1.0e-12)


def test_exact_identity_constant_and_outside_convex_hull_evaluation() -> None:
    points, _, systems, ragged = _systems_and_guide_query()
    values = np.asarray([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float64)
    guide_result = diagnostic.evaluate_ragged_field(
        points,
        ragged,
        systems,
        values,
    )
    np.testing.assert_allclose(guide_result.values, values, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        guide_result.constant_values,
        np.ones((5,)),
        rtol=0.0,
        atol=1.0e-12,
    )
    outside = RaggedQueryTopologyDistances(
        indptr=np.asarray([0, 1]),
        patch_ids=np.asarray([0]),
        distances=np.asarray([0.25]),
        radii=np.asarray([float(systems.radii[0])]),
        patch_node_counts=np.asarray([systems.node_count] * systems.patch_count),
        report={},
    )
    outside_result = diagnostic.evaluate_ragged_field(
        np.asarray([[20.0, -3.0, 2.0]], dtype=np.float64),
        outside,
        systems,
        np.ones((5,), dtype=np.float64),
    )
    assert outside_result.values[0] == pytest.approx(1.0, abs=1.0e-12)
    assert outside_result.constant_values[0] == pytest.approx(1.0, abs=1.0e-12)


def test_ragged_global_collapse_preserves_signed_direct_field() -> None:
    points, _, systems, ragged = _systems_and_guide_query()
    values = np.asarray([0.2, 0.8, 1.4, 2.1, 3.0], dtype=np.float64)
    result = diagnostic.evaluate_ragged_field(
        points,
        ragged,
        systems,
        values,
        collapse_global_weights=True,
        pair_chunk_size=2,
    )
    assert result.global_weights is not None
    np.testing.assert_allclose(
        np.asarray(result.global_weights @ values).reshape(-1),
        result.values,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(result.global_weights.sum(axis=1)).reshape(-1),
        result.cardinal_row_sums,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert result.report["signed_cardinal_weights_clamped_or_renormalized"] is False
    assert result.report["collapsed_CSR_field_reconstruction_max_abs_error"] <= 1.0e-10


def test_strict_ragged_binding_rejects_duplicate_omitted_radius_count_and_distance() -> None:
    _, state, _, ragged = _systems_and_guide_query()
    valid = diagnostic.validate_ragged_binding(
        ragged,
        state,
        state.inputs.guide_face_ids,
        state.inputs.guide_barycentric,
    )
    assert valid["passed"] is True
    assert valid["zero_weight_candidates_preserved"] is True

    def replace(**changes: np.ndarray) -> RaggedQueryTopologyDistances:
        values = {
            "indptr": ragged.indptr.copy(),
            "patch_ids": ragged.patch_ids.copy(),
            "distances": ragged.distances.copy(),
            "radii": ragged.radii.copy(),
            "patch_node_counts": ragged.patch_node_counts.copy(),
        }
        values.update(changes)
        return RaggedQueryTopologyDistances(report={}, **values)

    row_begin, row_end = ragged.indptr[:2]
    assert row_end - row_begin >= 2
    duplicate_ids = ragged.patch_ids.copy()
    duplicate_ids[row_begin + 1] = duplicate_ids[row_begin]
    duplicate = replace(patch_ids=duplicate_ids)

    omitted_indptr = ragged.indptr.copy()
    omitted_indptr[1:] -= 1
    omitted = replace(
        indptr=omitted_indptr,
        patch_ids=np.delete(ragged.patch_ids, row_begin),
        distances=np.delete(ragged.distances, row_begin),
        radii=np.delete(ragged.radii, row_begin),
    )
    wrong_radii = ragged.radii.copy()
    wrong_radii[0] = np.nextafter(wrong_radii[0], np.inf)
    wrong_counts = ragged.patch_node_counts.copy()
    wrong_counts[0] += 1
    wrong_distances = ragged.distances.copy()
    wrong_distances[0] = np.nextafter(wrong_distances[0], np.inf)
    for corrupted, message in (
        (duplicate, "patch IDs"),
        (omitted, "patch IDs"),
        (replace(radii=wrong_radii), "radii"),
        (replace(patch_node_counts=wrong_counts), "patch_node_counts"),
        (replace(distances=wrong_distances), "distances"),
    ):
        with pytest.raises(diagnostic.DiagnosticError, match=message):
            diagnostic.validate_ragged_binding(
                corrupted,
                state,
                state.inputs.guide_face_ids,
                state.inputs.guide_barycentric,
            )


def test_interior_edge_two_representations_are_continuous() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    d = np.asarray(
        [
            [0.0, 1.0, np.sqrt(2.0), 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [np.sqrt(2.0), 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    guide_faces = np.asarray([0, 0, 0, 1], dtype=np.int64)
    guide_bary = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
        dtype=np.float64,
    )
    inputs = validate_topology_cover_inputs(
        d,
        np.arange(4, dtype=np.int64),
        np.zeros((4,), dtype=np.float64),
        faces,
        guide_faces,
        guide_bary,
    )
    matrix = compute_patch_guide_site_distances(inputs)
    nodes = select_patch_radii_and_nodes(matrix, 2)
    vertex_cover = build_vertex_patch_active_distances(inputs, nodes)
    face_cover = build_face_patch_candidate_counts(inputs, vertex_cover)
    systems = diagnostic.build_batched_patch_systems(vertices, nodes, device="cpu")
    edge = diagnostic.select_interior_edge_queries(vertices, faces, count=1)
    ragged = evaluate_query_topology_distances(
        inputs,
        face_cover,
        edge.face_ids,
        edge.barycentric,
    )
    result = diagnostic.evaluate_ragged_field(
        edge.points_local,
        ragged,
        systems,
        np.asarray([0.5, 1.0, 1.5, 2.0]),
    )
    assert result.covered.tolist() == [True, True]
    assert result.values[0] == pytest.approx(result.values[1], abs=1.0e-12)


def test_c1_manifest_hashes_are_verified_before_state_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, state = _line_state()
    arrays = _synthetic_state_arrays(state)
    report_hash, manifest_hash = _write_verified_c1_fixture(tmp_path / "c1", arrays)
    verified = diagnostic.verify_c1_artifacts(
        tmp_path / "c1",
        expected_report_sha256=report_hash,
        expected_manifest_sha256=manifest_hash,
    )
    assert verified.identities["report_contract"]["hard_gate_count"] == 14
    monkeypatch.setattr(
        diagnostic,
        "_load_npy",
        lambda *_args, **_kwargs: pytest.fail("arrays must not load during hash verification"),
    )
    (tmp_path / "c1" / "guide_distances.npy").write_bytes(b"corrupt")
    with pytest.raises(diagnostic.DiagnosticError, match="hash/size mismatch"):
        diagnostic.verify_c1_artifacts(
            tmp_path / "c1",
            expected_report_sha256=report_hash,
            expected_manifest_sha256=manifest_hash,
        )


def test_c1_report_contract_requires_exact_candidate_history() -> None:
    valid = _c1_report()
    contract = diagnostic._require_c1_report_contract(valid)
    assert contract["first_and_only_passing_k"] == 32
    mutations: list[dict[str, object]] = []
    wrong_sequence = copy.deepcopy(valid)
    wrong_sequence["candidate_k_sequence"] = [8, 12, 16, 24, 32, 64, 48]
    mutations.append(wrong_sequence)
    early_pass = copy.deepcopy(valid)
    early_pass["candidate_results"][1]["status"] = "passed"  # type: ignore[index]
    early_pass["candidate_results"][1]["hard_gates"] = {  # type: ignore[index]
        name: True for name in c1.HARD_GATE_KEYS
    }
    early_pass["candidate_results"][1]["all_hard_gates_passed"] = True  # type: ignore[index]
    mutations.append(early_pass)
    evaluated_late = copy.deepcopy(valid)
    evaluated_late["candidate_results"][5]["status"] = "rejected"  # type: ignore[index]
    mutations.append(evaluated_late)
    aggregate_mismatch = copy.deepcopy(valid)
    aggregate_mismatch["candidate_results"][0]["all_hard_gates_passed"] = True  # type: ignore[index]
    mutations.append(aggregate_mismatch)
    for mutated in mutations:
        with pytest.raises(diagnostic.DiagnosticError):
            diagnostic._require_c1_report_contract(mutated)


def test_c1_source_commit_is_separate_and_source_files_must_be_unchanged(
    tmp_path: Path,
) -> None:
    c1_commit = "a" * 40
    c2_commit = "b" * 40
    report = _c1_report()
    current_paths = {
        "diagnostic": Path(c1.__file__).resolve(),
        "rbf_partition_of_unity": (
            diagnostic.PROJECT_ROOT / "anigroom" / "rbf_partition_of_unity.py"
        ),
        "rbf_topology_cover": (
            diagnostic.PROJECT_ROOT / "anigroom" / "rbf_topology_cover.py"
        ),
        "surface_graph": (
            diagnostic.PROJECT_ROOT / "anigroom" / "flow" / "surface_graph.py"
        ),
    }
    report["identities"] = {
        "source": {
            "head": c1_commit,
            "clean": True,
            "expectation": {
                "expected": c1_commit,
                "observed": c1_commit,
                "passed": True,
            },
        },
        "source_files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": diagnostic.sha256_file(path),
            }
            for name, path in current_paths.items()
        },
    }
    verified = diagnostic.VerifiedC1Artifacts(
        state_dir=tmp_path,
        report=report,
        manifest={},
        paths={},
        identities={},
    )
    binding = diagnostic.validate_c1_source_contract(
        verified,
        expected_c1_source_commit=c1_commit,
    )
    assert binding["passed"] is True
    assert binding["expected_c1_source_commit"] == c1_commit
    assert c2_commit != c1_commit
    corrupted = copy.deepcopy(report)
    corrupted["identities"]["source_files"]["surface_graph"]["sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(diagnostic.DiagnosticError, match="changed"):
        diagnostic.validate_c1_source_contract(
            diagnostic.VerifiedC1Artifacts(
                state_dir=tmp_path,
                report=corrupted,
                manifest={},
                paths={},
                identities={},
            ),
            expected_c1_source_commit=c1_commit,
        )


def test_mocked_orchestration_accepts_different_valid_c1_and_c2_commits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    c1_commit = "a" * 40
    c2_commit = "b" * 40
    points, state = _line_state()
    checkpoint_arrays = c1.CheckpointTopologyArrays(
        vertices=points,
        faces=state.inputs.faces,
        stored_guide_points_local=points.copy(),
        guide_points_local=points.copy(),
        guide_face_ids=state.inputs.guide_face_ids,
        guide_barycentric=state.inputs.guide_barycentric,
        guide_point_reconstruction={"passed": True, "max": 0.0, "p95": 0.0},
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mocked-C2-checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_paths = {
        "diagnostic": Path(c1.__file__).resolve(),
        "rbf_partition_of_unity": (
            diagnostic.PROJECT_ROOT / "anigroom" / "rbf_partition_of_unity.py"
        ),
        "rbf_topology_cover": (
            diagnostic.PROJECT_ROOT / "anigroom" / "rbf_topology_cover.py"
        ),
        "surface_graph": (
            diagnostic.PROJECT_ROOT / "anigroom" / "flow" / "surface_graph.py"
        ),
    }
    verified_report = _c1_report()
    verified_report["identities"] = {
        "source": {
            "head": c1_commit,
            "clean": True,
            "expectation": {
                "expected": c1_commit,
                "observed": c1_commit,
                "passed": True,
            },
        },
        "source_files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": diagnostic.sha256_file(path),
            }
            for name, path in source_paths.items()
        },
        "checkpoint": {
            "observed_sha256": checkpoint_hash,
            "observed_iteration": 7,
        },
        "model_arrays": {
            "vertices": diagnostic.array_identity(checkpoint_arrays.vertices),
            "faces": diagnostic.array_identity(checkpoint_arrays.faces),
            "stored_guide_points_local": diagnostic.array_identity(
                checkpoint_arrays.stored_guide_points_local
            ),
            "canonical_guide_points_local": diagnostic.array_identity(
                checkpoint_arrays.guide_points_local
            ),
            "guide_face_ids": diagnostic.array_identity(
                checkpoint_arrays.guide_face_ids
            ),
            "guide_barycentric": diagnostic.array_identity(
                checkpoint_arrays.guide_barycentric
            ),
        },
    }
    verified = diagnostic.VerifiedC1Artifacts(
        state_dir=tmp_path,
        report=verified_report,
        manifest={},
        paths={},
        identities={"mocked": True},
    )
    model = SimpleNamespace(
        guide_length_raw=torch.tensor(
            [1.0, 1.1, 1.2, 1.3, 1.4], dtype=torch.float64
        ),
        guide_length_reference=torch.ones((5,), dtype=torch.float64),
        guide_points_local=torch.as_tensor(points, dtype=torch.float64),
    )
    selected_ids = np.asarray([1, 4], dtype=np.int64)
    selected_hash = diagnostic.array_identity(selected_ids)["sha256"]
    render_points = np.repeat(points[1:2], 6, axis=0)
    render = diagnostic.RenderGeometry(
        points_local=render_points,
        normals_local=np.repeat([[0.0, 0.0, 1.0]], 6, axis=0),
        face_ids=np.zeros((6,), dtype=np.int64),
        barycentric=np.repeat([[0.0, 1.0, 0.0]], 6, axis=0),
        report={"population_count": 6},
    )
    edge = diagnostic.InteriorEdgeQueries(
        edge_vertices=np.asarray([[0, 1]], dtype=np.int64),
        face_ids=np.asarray([0, 0], dtype=np.int64),
        barycentric=np.asarray([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]),
        points_local=np.repeat([[0.5, 0.0, 0.0]], 2, axis=0),
        report={"requested_count": 1},
    )
    legacy_support = SurfaceSupport(
        indices=torch.zeros((2, 1), dtype=torch.long),
        vertex_path_distances=torch.zeros((2, 1, 3)),
        report={"provenance": "mocked canonical slice"},
    )
    monkeypatch.setattr(diagnostic, "EXPECTED_RENDER_POPULATION", 6)
    monkeypatch.setattr(diagnostic, "RENDER_ROOT_COUNT", 2)
    monkeypatch.setattr(diagnostic, "INTERIOR_EDGE_PROBE_COUNT", 1)
    monkeypatch.setattr(diagnostic, "REQUIRED_ROOT_ID", 4)
    monkeypatch.setattr(diagnostic, "EXPECTED_REQUIRED_ROOT_ROW", 1)
    monkeypatch.setattr(diagnostic, "EXPECTED_SELECTED_IDS_SHA256", selected_hash)
    monkeypatch.setattr(diagnostic, "EXPECTED_GUIDE_COUNT", 5)
    monkeypatch.setattr(
        diagnostic,
        "get_clean_source_git_identity",
        lambda *_args, **_kwargs: {
            "repository": str(diagnostic.PROJECT_ROOT),
            "head": c2_commit,
            "porcelain_status": "",
            "clean": True,
        },
    )
    monkeypatch.setattr(diagnostic, "verify_c1_artifacts", lambda *_a, **_k: verified)
    monkeypatch.setattr(
        diagnostic,
        "_import_stage1",
        lambda: (
            lambda _path, _device: (model, SimpleNamespace(), {"iteration": 7}),
            lambda raw, _reference: raw,
        ),
    )
    monkeypatch.setattr(
        diagnostic.c1,
        "extract_checkpoint_topology_arrays",
        lambda _model: checkpoint_arrays,
    )
    monkeypatch.setattr(diagnostic, "load_c1_state", lambda *_a, **_k: state)
    monkeypatch.setattr(diagnostic, "extract_render_geometry", lambda _model: render)
    monkeypatch.setattr(
        diagnostic,
        "select_render_root_ids",
        lambda _population: selected_ids,
    )
    monkeypatch.setattr(
        diagnostic,
        "select_interior_edge_queries",
        lambda *_a, **_k: edge,
    )
    monkeypatch.setattr(
        diagnostic,
        "evaluate_legacy_selected_lengths",
        lambda *_a, **_k: (
            np.asarray([1.1, 1.1], dtype=np.float64),
            legacy_support,
            dict(legacy_support.report),
        ),
    )
    report = diagnostic.run_fixed_checkpoint_diagnostic(
        checkpoint=checkpoint,
        c1_state_dir=tmp_path,
        output_dir=tmp_path / "output",
        device="cpu",
        expected_checkpoint_sha256=checkpoint_hash,
        expected_iteration=7,
        expected_source_commit=c2_commit,
        expected_c1_source_commit=c1_commit,
        expected_c1_report_sha256="c" * 64,
        expected_c1_manifest_sha256="d" * 64,
    )
    assert report["accepted"] is True
    assert report["identities"]["source"]["head"] == c2_commit
    assert report["identities"]["c1_source_binding"][
        "observed_c1_source_commit"
    ] == c1_commit


def test_synthetic_c1_state_reconstructs_and_rejects_corrupt_csr(tmp_path: Path) -> None:
    points, state = _line_state()
    arrays = _synthetic_state_arrays(state)
    report_hash, manifest_hash = _write_verified_c1_fixture(tmp_path / "valid", arrays)
    verified = diagnostic.verify_c1_artifacts(
        tmp_path / "valid",
        expected_report_sha256=report_hash,
        expected_manifest_sha256=manifest_hash,
    )
    checkpoint_arrays = c1.CheckpointTopologyArrays(
        vertices=points,
        faces=state.inputs.faces,
        stored_guide_points_local=points.copy(),
        guide_points_local=points.copy(),
        guide_face_ids=state.inputs.guide_face_ids,
        guide_barycentric=state.inputs.guide_barycentric,
        guide_point_reconstruction={"passed": True, "max": 0.0, "p95": 0.0},
    )
    loaded = diagnostic.load_c1_state(
        verified,
        checkpoint_arrays,
        expected_guide_count=5,
        expected_nodes_per_patch=3,
    )
    assert loaded.report["passed"] is True
    corrupt_arrays = {name: value.copy() for name, value in arrays.items()}
    corrupt_arrays["patch_node_indices.npy"][0] = 99
    bad_report, bad_manifest = _write_verified_c1_fixture(
        tmp_path / "bad",
        corrupt_arrays,
    )
    bad_verified = diagnostic.verify_c1_artifacts(
        tmp_path / "bad",
        expected_report_sha256=bad_report,
        expected_manifest_sha256=bad_manifest,
    )
    with pytest.raises(diagnostic.DiagnosticError, match="out-of-range"):
        diagnostic.load_c1_state(
            bad_verified,
            checkpoint_arrays,
            expected_guide_count=5,
            expected_nodes_per_patch=3,
        )


def test_legacy_comparison_slices_canonical_cached_full_support_without_grad() -> None:
    population = 6
    support = SurfaceSupport(
        indices=torch.arange(population * 2, dtype=torch.long).reshape(population, 2) % 4,
        vertex_path_distances=torch.arange(
            population * 2 * 3,
            dtype=torch.float32,
        ).reshape(population, 2, 3),
        report={"origin": "canonical-cache"},
    )
    calls: list[str] = []

    class FakeModel:
        guide_points_local = torch.zeros((4, 3), dtype=torch.float32)
        face_ids = torch.zeros((population,), dtype=torch.long)

        @staticmethod
        def guide_interpolation_support() -> SurfaceSupport:
            calls.append("canonical_support")
            return support

        @staticmethod
        def tangent_frames_for_face_ids(
            face_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            count = int(face_ids.shape[0])
            return (
                torch.zeros((count, 3)),
                torch.ones((count, 3)),
                torch.ones((count, 3)),
            )

        @staticmethod
        def sample_guide_controls(
            points: torch.Tensor,
            _faces: torch.Tensor,
            _normals: torch.Tensor,
            _tangents: torch.Tensor,
            _bitangents: torch.Tensor,
            *,
            support: SurfaceSupport,
        ) -> tuple[dict[str, torch.Tensor], None]:
            calls.append("sample")
            assert torch.is_grad_enabled() is False
            expected = torch.stack((
                diagnostic_support.indices[1],
                diagnostic_support.indices[4],
            ))
            torch.testing.assert_close(support.indices, expected)
            return {"length": points[:, 0].abs() + 1.0}, None

    diagnostic_support = support
    selected = np.asarray([1, 4], dtype=np.int64)
    points = np.asarray([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    lengths, sliced, report = diagnostic.evaluate_legacy_selected_lengths(
        FakeModel(),
        points,
        np.ones((2, 3), dtype=np.float64),
        np.zeros((2,), dtype=np.int64),
        selected,
    )
    np.testing.assert_array_equal(lengths, [3.0, 4.0])
    assert sliced.query_count == 2
    assert report["provenance"] == "canonical_full_render_support_sliced_by_selected_root_ids"
    assert calls == ["canonical_support", "sample"]


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("injected C2 OOM"),
        RuntimeError("injected C2 backend failure"),
    ],
)
def test_batched_system_build_propagates_oom_and_backend_failures(
    monkeypatch: pytest.MonkeyPatch,
    error: RuntimeError,
) -> None:
    points, state = _line_state()

    def raise_failure(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise error

    monkeypatch.setattr(diagnostic.torch.linalg, "solve", raise_failure)
    with pytest.raises(type(error), match="injected C2"):
        diagnostic.build_batched_patch_systems(
            points,
            state.patch_nodes,
            device="cpu",
        )


def test_realistic_4500_by_32_patch_layout_shape_construction() -> None:
    patch_count = 4500
    node_count = 32
    indptr = np.arange(patch_count + 1, dtype=np.int64) * node_count
    indices = np.tile(np.arange(node_count, dtype=np.int64), patch_count)
    matrix = csr_matrix(
        (
            np.zeros((indices.size,), dtype=np.float64),
            indices,
            indptr,
        ),
        shape=(patch_count, patch_count),
    )
    layout, radii, observed_count = diagnostic.extract_fixed_patch_layout(
        patch_count,
        PatchNodeCover(
            radii=np.ones((patch_count,), dtype=np.float64),
            node_distances=matrix,
            report={},
        ),
        expected_node_count=32,
    )
    assert layout.shape == (4500, 32)
    assert radii.shape == (4500,)
    assert observed_count == 32


def test_conditional_cpu_cuda_batched_evaluator_parity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    points, _, cpu_systems, ragged = _systems_and_guide_query()
    cuda_systems = diagnostic.build_batched_patch_systems(
        points,
        _line_state()[1].patch_nodes,
        device="cuda",
        system_chunk_size=2,
    )
    values = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    cpu_result = diagnostic.evaluate_ragged_field(
        points,
        ragged,
        cpu_systems,
        values,
        pair_chunk_size=2,
    )
    cuda_result = diagnostic.evaluate_ragged_field(
        points,
        ragged,
        cuda_systems,
        values,
        pair_chunk_size=2,
    )
    np.testing.assert_allclose(cuda_result.values, cpu_result.values, rtol=1.0e-10, atol=1.0e-10)
    torch.testing.assert_close(
        cuda_systems.inverses.cpu(),
        cpu_systems.inverses,
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_hard_gate_thresholds_are_inclusive_and_descriptive_metrics_are_excluded() -> None:
    metrics: dict[str, object] = {
        "identities_and_c1_state_exact": True,
        "all_query_groups_covered": True,
        "all_query_groups_finite": True,
        "guide_site_max_abs_self_error": 1.0e-10,
        "constant_reproduction_max_abs_error": 1.0e-10,
        "local_pair_cardinal_sum_max_abs_error": 1.0e-10,
        "global_cardinal_row_sum_max_abs_error": 1.0e-10,
        "edge_midpoint_cross_face_max_abs_difference": 1.0e-10,
        "all_evaluated_lengths_finite_positive": True,
        "required_root_present": True,
        "required_root_covered": True,
        "required_root_finite_positive": True,
        "inverse_residual_max": 1.0e-10,
        "field_evaluation_seconds": 600.0,
        "candidate_vs_legacy_difference": float("nan"),
        "negative_global_weight_mass": -100.0,
    }
    gates = diagnostic.evaluate_hard_gates(metrics)
    assert gates == {name: True for name in diagnostic.HARD_GATE_KEYS}
    assert diagnostic.aggregate_hard_gates(gates) is True
    for key, value in (
        ("guide_site_max_abs_self_error", np.nextafter(1.0e-10, np.inf)),
        ("constant_reproduction_max_abs_error", np.nextafter(1.0e-10, np.inf)),
        ("local_pair_cardinal_sum_max_abs_error", np.nextafter(1.0e-10, np.inf)),
        ("global_cardinal_row_sum_max_abs_error", np.nextafter(1.0e-10, np.inf)),
        ("edge_midpoint_cross_face_max_abs_difference", np.nextafter(1.0e-10, np.inf)),
        ("inverse_residual_max", np.nextafter(1.0e-10, np.inf)),
        ("field_evaluation_seconds", np.nextafter(600.0, np.inf)),
    ):
        changed = dict(metrics)
        changed[key] = value
        assert diagnostic.aggregate_hard_gates(
            diagnostic.evaluate_hard_gates(changed)
        ) is False
    nonpositive_group = dict(metrics)
    nonpositive_group["all_evaluated_lengths_finite_positive"] = False
    assert diagnostic.aggregate_hard_gates(
        diagnostic.evaluate_hard_gates(nonpositive_group)
    ) is False
    with pytest.raises(diagnostic.DiagnosticError, match="missing"):
        diagnostic.evaluate_hard_gates({})


def _output_arrays() -> dict[str, np.ndarray]:
    return {
        "selected_root_ids.npy": np.asarray([1, 4], dtype=np.int64),
        "candidate_lengths.npy": np.asarray([1.0, 2.0], dtype=np.float64),
        "legacy_lengths.npy": np.asarray([1.1, 1.9], dtype=np.float64),
        "selected_global_weight_indptr.npy": np.asarray([0, 1, 2], dtype=np.int64),
        "selected_global_weight_indices.npy": np.asarray([0, 1], dtype=np.int64),
        "selected_global_weight_data.npy": np.asarray([1.0, 1.0], dtype=np.float64),
        "guide_site_errors.npy": np.asarray([0.0, 1.0e-12], dtype=np.float64),
        "edge_pair_values.npy": np.asarray([[1.0, 1.0]], dtype=np.float64),
    }


def test_staged_output_is_deterministic_and_write_failure_preserves_prior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _output_arrays()
    hashes = []
    for name in ("first", "second"):
        output = tmp_path / name
        output.mkdir()
        report = {"schema": diagnostic.SCHEMA, "accepted": True}
        manifest = diagnostic.save_outputs_staged(
            output,
            report,
            arrays,
            overwrite=False,
        )
        hashes.append(
            {
                artifact: identity["sha256"]
                for artifact, identity in manifest["artifacts"].items()
                if artifact.endswith(".npy")
            }
        )
    assert hashes[0] == hashes[1]
    output = tmp_path / "first"
    before = {
        name: (output / name).read_bytes()
        for name in diagnostic.ARTIFACT_NAMES
        if (output / name).is_file()
    }
    original_writer = diagnostic.write_atomic_npy
    calls = 0

    def failing_writer(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected C2 staged write failure")
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "write_atomic_npy", failing_writer)
    with pytest.raises(OSError, match="staged write failure"):
        diagnostic.save_outputs_staged(
            output,
            {"schema": diagnostic.SCHEMA, "accepted": False},
            arrays,
            overwrite=True,
        )
    after = {
        name: (output / name).read_bytes()
        for name in diagnostic.ARTIFACT_NAMES
        if (output / name).is_file()
    }
    assert after == before
    assert not list(tmp_path.glob(".first.r084-c2-stage.*"))
    monkeypatch.setattr(diagnostic, "write_atomic_npy", original_writer)
    original_publish = diagnostic._publish_temporary_file
    publish_calls = 0

    def failing_publish(*args: object, **kwargs: object) -> None:
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 5:
            raise OSError("injected C2 publish-position failure")
        original_publish(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "_publish_temporary_file", failing_publish)
    changed_arrays = {name: value.copy() for name, value in arrays.items()}
    changed_arrays["candidate_lengths.npy"] += 10.0
    with pytest.raises(OSError, match="publish-position failure"):
        diagnostic.save_outputs_staged(
            output,
            {"schema": diagnostic.SCHEMA, "accepted": False},
            changed_arrays,
            overwrite=True,
        )
    after_publish_failure = {
        name: (output / name).read_bytes()
        for name in diagnostic.ARTIFACT_NAMES
        if (output / name).is_file()
    }
    assert after_publish_failure == before
    assert not list(tmp_path.glob(".first.r084-c2-stage.*"))
    assert not list(tmp_path.glob(".first.r084-c2-backup.*"))


def test_output_contract_has_no_images_training_or_checkpoint_mutation() -> None:
    assert all(
        name.endswith(".npy") or name.endswith(".json")
        for name in diagnostic.ARTIFACT_NAMES
    )
    assert diagnostic.RENDER_ROOT_COUNT == 4096
    assert diagnostic.SELECTION_SEED == 20260901
    assert diagnostic.INTERIOR_EDGE_PROBE_COUNT == 1024
    assert diagnostic.REQUIRED_ROOT_ID == 431701
    assert diagnostic.EXPECTED_RENDER_POPULATION == 496632
    assert diagnostic.EXPECTED_REQUIRED_ROOT_ROW == 3552
    assert diagnostic.EXPECTED_SELECTED_IDS_SHA256 == (
        "fc9862c75c240e8e3c5f3ffc6940264936fb53c789620b8b44c6697b89a43d56"
    )
    assert diagnostic.EVALUATION_TIMEOUT_BUDGET_SECONDS == 600.0

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import torch

from anigroom.flow import surface_graph
from anigroom.rbf_topology_cover import (
    PatchNodeCover,
    TopologyCoverError,
    compute_patch_guide_site_distances,
    validate_topology_cover_inputs,
)
from tools import diagnose_rbf_partition_cover as diagnostic


def _line_arrays(count: int) -> diagnostic.CheckpointTopologyArrays:
    vertices = np.stack(
        (
            np.arange(count, dtype=np.float64),
            np.zeros((count,), dtype=np.float64),
            np.zeros((count,), dtype=np.float64),
        ),
        axis=1,
    )
    faces = np.asarray(
        [[index, index + 1, index + 2] for index in range(count - 2)],
        dtype=np.int64,
    )
    guide_face_ids = np.empty((count,), dtype=np.int64)
    guide_barycentric = np.zeros((count, 3), dtype=np.float64)
    for guide_id in range(count):
        face_id = min(guide_id, count - 3)
        corner = int(np.flatnonzero(faces[face_id] == guide_id)[0])
        guide_face_ids[guide_id] = face_id
        guide_barycentric[guide_id, corner] = 1.0
    return diagnostic.CheckpointTopologyArrays(
        vertices=vertices,
        faces=faces,
        stored_guide_points_local=vertices.copy(),
        guide_points_local=vertices.copy(),
        guide_face_ids=guide_face_ids,
        guide_barycentric=guide_barycentric,
        guide_point_reconstruction={
            "passed": True,
            "count": count,
            "max": 0.0,
            "p95": 0.0,
            "predeclared_max_error": (
                diagnostic.MAX_GUIDE_POINT_RECONSTRUCTION_ERROR
            ),
        },
    )


def _synthetic_topology(
    count: int = 5,
) -> tuple[diagnostic.CheckpointTopologyArrays, diagnostic.ActualTopologyData]:
    arrays = _line_arrays(count)
    guide_ids = np.arange(count, dtype=np.float64)
    distances = np.abs(guide_ids[:, None] - guide_ids[None, :])
    inputs = validate_topology_cover_inputs(
        distances,
        np.arange(count, dtype=np.int64),
        np.zeros((count,), dtype=np.float64),
        arrays.faces,
        arrays.guide_face_ids,
        arrays.guide_barycentric,
    )
    matrix = compute_patch_guide_site_distances(
        inputs,
        patch_chunk_size=1,
        guide_chunk_size=2,
    )
    topology = diagnostic.ActualTopologyData(
        inputs=inputs,
        patch_guide_distances=matrix,
        report={
            "passed": True,
            "component_coverage": {"passed": True},
        },
    )
    return arrays, topology


def _fake_model(
    arrays: diagnostic.CheckpointTopologyArrays,
    *,
    stored_guide_points: np.ndarray | None = None,
) -> SimpleNamespace:
    stored = (
        arrays.stored_guide_points_local
        if stored_guide_points is None
        else stored_guide_points
    )
    return SimpleNamespace(
        vertices=torch.as_tensor(arrays.vertices, dtype=torch.float32),
        faces=torch.as_tensor(arrays.faces, dtype=torch.long),
        guide_points_local=torch.as_tensor(stored, dtype=torch.float64),
        guide_face_ids=torch.as_tensor(arrays.guide_face_ids, dtype=torch.long),
        guide_barycentric=torch.as_tensor(
            arrays.guide_barycentric,
            dtype=torch.float64,
        ),
        guide_enabled=lambda: True,
    )


def _passing_metrics() -> dict[str, object]:
    return {
        "topology_validation_passed": True,
        "component_coverage_passed": True,
        "finite_zero_mass_boundaries_passed": True,
        "patch_self_membership_passed": True,
        "local_systems_full_rank_passed": True,
        "max_local_condition_number": diagnostic.MAX_LOCAL_CONDITION_NUMBER,
        "max_local_node_self_error": diagnostic.MAX_LOCAL_ERROR,
        "max_local_constant_error": diagnostic.MAX_LOCAL_ERROR,
        "max_local_cardinal_sum_error": diagnostic.MAX_LOCAL_ERROR,
        "uncovered_vertex_count": 0,
        "faces_without_candidate_count": 0,
        "faces_lacking_strong_full_face_cover_count": 0,
        "patch_node_count_max": diagnostic.MAX_PATCH_NODE_COUNT,
        "serialized_state_bytes": diagnostic.MAX_SERIALIZED_STATE_BYTES,
    }


def test_parser_requires_all_identities_and_uses_predeclared_defaults() -> None:
    parser = diagnostic.build_argument_parser()
    args = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--output-dir",
            "out",
            "--expected-checkpoint-sha256",
            "a" * 64,
            "--expected-iteration",
            "4000",
            "--expected-source-commit",
            "b" * 40,
        ]
    )
    assert args.device == "cuda"
    assert args.overwrite is False
    assert args.expected_iteration == 4000
    assert diagnostic.CANDIDATE_K_SEQUENCE == (8, 12, 16, 24, 32, 48, 64)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--output-dir",
                "out",
            ]
        )


def test_mandatory_checkpoint_and_source_identities_are_exact() -> None:
    checkpoint_hash = "ab" * 32
    source_commit = "c" * 40
    assert diagnostic.validate_source_identity(source_commit, source_commit.upper())[
        "passed"
    ]
    accepted = diagnostic.validate_checkpoint_identity(
        checkpoint_hash,
        checkpoint_hash.upper(),
        {"iteration": 19},
        19,
    )
    assert accepted["passed"] is True
    with pytest.raises(diagnostic.DiagnosticError, match="mandatory"):
        diagnostic.validate_source_identity(source_commit, None)
    with pytest.raises(diagnostic.DiagnosticError, match="mandatory"):
        diagnostic.validate_checkpoint_identity(checkpoint_hash, None, {"iteration": 19}, 19)
    with pytest.raises(diagnostic.DiagnosticError, match="mandatory"):
        diagnostic.validate_checkpoint_identity(checkpoint_hash, checkpoint_hash, {}, None)
    with pytest.raises(diagnostic.DiagnosticError, match="source HEAD mismatch"):
        diagnostic.validate_source_identity("d" * 40, source_commit)
    with pytest.raises(diagnostic.DiagnosticError, match="SHA256 mismatch"):
        diagnostic.validate_checkpoint_identity("d" * 64, checkpoint_hash, {"iteration": 19}, 19)
    with pytest.raises(diagnostic.DiagnosticError, match="iteration mismatch"):
        diagnostic.validate_checkpoint_identity(checkpoint_hash, checkpoint_hash, {"iteration": 18}, 19)
    with pytest.raises(ValueError, match="40 or 64"):
        diagnostic.normalize_git_commit("e" * 41)


def test_guide_points_are_reconstructed_in_float64_and_mismatch_is_gated() -> None:
    arrays = _line_arrays(5)
    stored = arrays.guide_points_local.copy()
    stored[2, 1] += 0.5 * diagnostic.MAX_GUIDE_POINT_RECONSTRUCTION_ERROR
    extracted = diagnostic.extract_checkpoint_topology_arrays(
        _fake_model(arrays, stored_guide_points=stored)
    )
    np.testing.assert_array_equal(extracted.guide_points_local, arrays.vertices)
    np.testing.assert_array_equal(extracted.stored_guide_points_local, stored)
    assert extracted.guide_point_reconstruction["passed"] is True
    assert extracted.guide_point_reconstruction["max"] == pytest.approx(0.5e-6)
    assert 0.0 <= extracted.guide_point_reconstruction["p95"] <= 0.5e-6
    assert (
        diagnostic.array_identity(extracted.stored_guide_points_local)["sha256"]
        != diagnostic.array_identity(extracted.guide_points_local)["sha256"]
    )

    invalid = arrays.guide_points_local.copy()
    invalid[0, 2] += np.nextafter(
        diagnostic.MAX_GUIDE_POINT_RECONSTRUCTION_ERROR,
        np.inf,
    )
    with pytest.raises(diagnostic.DiagnosticError, match="barycentric reconstruction"):
        diagnostic.extract_checkpoint_topology_arrays(
            _fake_model(arrays, stored_guide_points=invalid)
        )


def test_clean_git_identity_uses_bounded_shell_false_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    outputs = [b"a" * 40 + b"\n", b""]

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, stdout=outputs.pop(0), stderr=b"")

    monkeypatch.setattr(diagnostic.subprocess, "run", fake_run)
    identity = diagnostic.get_clean_source_git_identity(tmp_path)
    assert identity["head"] == "a" * 40
    assert identity["clean"] is True
    assert len(calls) == 2
    assert all(call["shell"] is False for call in calls)
    assert all(call["timeout"] == diagnostic.GIT_TIMEOUT_SECONDS for call in calls)


def test_git_identity_rejects_dirty_or_oversized_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs = [b"a" * 40 + b"\n", b" M dirty.py\n"]

    def dirty_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, stdout=outputs.pop(0), stderr=b"")

    monkeypatch.setattr(diagnostic.subprocess, "run", dirty_run)
    with pytest.raises(diagnostic.DiagnosticError, match="dirty"):
        diagnostic.get_clean_source_git_identity(tmp_path)

    def huge_run(argv: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=b"x" * (diagnostic.MAX_GIT_OUTPUT_BYTES + 1),
            stderr=b"",
        )

    monkeypatch.setattr(diagnostic.subprocess, "run", huge_run)
    with pytest.raises(diagnostic.DiagnosticError, match="bounded limit"):
        diagnostic.get_clean_source_git_identity(tmp_path)


def test_hard_gate_aggregation_includes_exact_threshold_boundaries() -> None:
    metrics = _passing_metrics()
    gates = diagnostic.evaluate_predeclared_hard_gates(metrics)
    assert set(gates) == set(diagnostic.HARD_GATE_KEYS)
    assert diagnostic.aggregate_hard_gates(gates) is True

    boundary_fields = (
        ("max_local_condition_number", np.nextafter(1.0e12, np.inf)),
        ("max_local_node_self_error", np.nextafter(1.0e-10, np.inf)),
        ("max_local_constant_error", np.nextafter(1.0e-10, np.inf)),
        ("max_local_cardinal_sum_error", np.nextafter(1.0e-10, np.inf)),
        ("patch_node_count_max", 129),
        ("serialized_state_bytes", 4 * 1024**3 + 1),
        ("uncovered_vertex_count", 1),
        ("faces_without_candidate_count", 1),
        ("faces_lacking_strong_full_face_cover_count", 1),
    )
    for name, failing_value in boundary_fields:
        changed = dict(metrics)
        changed[name] = failing_value
        assert diagnostic.aggregate_hard_gates(
            diagnostic.evaluate_predeclared_hard_gates(changed)
        ) is False
    with pytest.raises(diagnostic.DiagnosticError, match="missing"):
        diagnostic.evaluate_predeclared_hard_gates({})
    with pytest.raises(diagnostic.DiagnosticError, match="missing"):
        diagnostic.aggregate_hard_gates({})


def test_candidate_ordering_and_first_pass_selection_are_exact() -> None:
    results = [
        {"k": 8, "all_hard_gates_passed": False},
        {"k": 12, "all_hard_gates_passed": True},
        {"k": 16, "all_hard_gates_passed": True},
    ]
    assert diagnostic.choose_first_passing_candidate(results) == 12
    assert diagnostic.choose_first_passing_candidate(results[:1]) is None
    with pytest.raises(diagnostic.DiagnosticError, match="order"):
        diagnostic.choose_first_passing_candidate(list(reversed(results)))


def test_scan_preserves_rejection_and_stops_after_first_pass() -> None:
    arrays, topology = _synthetic_topology()
    calls: list[int] = []
    accepted_sentinel = object()

    def evaluator(
        _topology: diagnostic.ActualTopologyData,
        _points: np.ndarray,
        k: int,
    ) -> tuple[dict[str, object], object | None]:
        calls.append(k)
        if k == 8:
            raise TopologyCoverError("predeclared rejection evidence")
        return {"k": k, "all_hard_gates_passed": True}, accepted_sentinel

    results, selected, artifacts = diagnostic.scan_candidate_ks(
        topology,
        arrays.guide_points_local,
        device="cpu",
        sequence=(8, 12, 16),
        evaluator=evaluator,  # type: ignore[arg-type]
    )
    assert calls == [8, 12]
    assert selected == 12
    assert artifacts is accepted_sentinel
    assert results[0]["status"] == "rejected"
    assert results[0]["rejection_message"] == "predeclared rejection evidence"
    assert results[2]["status"] == "not_evaluated_after_first_pass"


def test_exact_synthetic_candidate_runs_cover_and_rbf_core_end_to_end() -> None:
    arrays, topology = _synthetic_topology()
    result, artifacts = diagnostic.evaluate_candidate_k(
        topology,
        arrays.guide_points_local,
        3,
        device="cpu",
        vertex_chunk_size=1,
    )
    assert artifacts is not None
    assert result["all_hard_gates_passed"] is True
    assert result["hard_gates"] == {
        name: True for name in diagnostic.HARD_GATE_KEYS
    }
    assert result["metrics"]["uncovered_vertex_count"] == 0
    assert result["metrics"]["faces_without_candidate_count"] == 0
    assert result["metrics"]["faces_lacking_strong_full_face_cover_count"] == 0
    assert result["metrics"]["patch_node_count_max"] == 3
    assert result["local_rbf_algebra"]["core_functions"] == [
        "build_augmented_system",
        "validate_augmented_system",
        "local_cardinal_weights",
        "solve_augmented_system",
        "evaluate_local_interpolant",
    ]


def test_patch_boundary_and_self_metrics_are_independently_derived_from_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays, topology = _synthetic_topology()
    nodes = diagnostic.select_patch_radii_and_nodes(
        topology.patch_guide_distances,
        3,
    )
    valid = diagnostic.derive_patch_cover_evidence(
        topology.patch_guide_distances,
        nodes,
    )
    assert valid["passed"] is True

    original_csr = nodes.node_distances
    row_zero_begin, row_zero_end = original_csr.indptr[:2]
    self_position = row_zero_begin + int(
        np.flatnonzero(original_csr.indices[row_zero_begin:row_zero_end] == 0)[0]
    )
    missing_indptr = original_csr.indptr.copy()
    missing_indptr[1:] -= 1
    missing_self_matrix = csr_matrix(
        (
            np.delete(original_csr.data, self_position),
            np.delete(original_csr.indices, self_position),
            missing_indptr,
        ),
        shape=original_csr.shape,
    )
    missing_self = PatchNodeCover(
        radii=nodes.radii.copy(),
        node_distances=missing_self_matrix,
        report={},
    )
    self_evidence = diagnostic.derive_patch_cover_evidence(
        topology.patch_guide_distances,
        missing_self,
    )
    assert self_evidence["patch_self_membership_passed"] is False
    assert self_evidence["missing_self_membership"]["first_patch_ids"] == [0]

    shifted_radii = nodes.radii.copy()
    shifted_radii[0] = np.nextafter(shifted_radii[0], np.inf)
    no_exact_boundary = PatchNodeCover(
        radii=shifted_radii,
        node_distances=nodes.node_distances.copy(),
        report={},
    )
    boundary_evidence = diagnostic.derive_patch_cover_evidence(
        topology.patch_guide_distances,
        no_exact_boundary,
    )
    assert boundary_evidence["finite_zero_mass_boundaries_passed"] is False
    assert boundary_evidence["radii_without_exact_M_boundary"]["first_patch_ids"] == [0]

    boundary_id = int(
        np.flatnonzero(
            topology.patch_guide_distances.values[0] == nodes.radii[0]
        )[0]
    )
    included_matrix = nodes.node_distances.tolil(copy=True)
    included_matrix[0, boundary_id] = nodes.radii[0]
    included_boundary = PatchNodeCover(
        radii=nodes.radii.copy(),
        node_distances=included_matrix.tocsr(),
        report={},
    )
    included_evidence = diagnostic.derive_patch_cover_evidence(
        topology.patch_guide_distances,
        included_boundary,
    )
    assert included_evidence["finite_zero_mass_boundaries_passed"] is False
    assert included_evidence["boundary_nodes_included_in_csr"]["first_patch_ids"] == [0]

    monkeypatch.setattr(
        diagnostic,
        "select_patch_radii_and_nodes",
        lambda *_args, **_kwargs: missing_self,
    )
    monkeypatch.setattr(
        diagnostic,
        "audit_local_rbf_systems",
        lambda *_args, **_kwargs: pytest.fail("local audit must be skipped"),
    )
    result, artifacts = diagnostic.evaluate_candidate_k(
        topology,
        arrays.guide_points_local,
        3,
        device="cpu",
    )
    assert artifacts is None
    assert result["rejection_stage"] == "independent_patch_cover_evidence"
    assert result["metrics"]["patch_self_membership_passed"] is False


def test_per_k_zero_boundary_rejection_is_preserved_without_cover_work() -> None:
    arrays, topology = _synthetic_topology()
    result, artifacts = diagnostic.evaluate_candidate_k(
        topology,
        arrays.guide_points_local,
        5,
        device="cpu",
    )
    assert artifacts is None
    assert result["status"] == "rejected"
    assert result["rejection_stage"] == "patch_radius_and_nodes"
    assert "no finite distinct" in result["rejection_message"]
    assert "local_rbf_algebra" not in result
    assert "vertex_cover" not in result


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("injected C1 OOM"),
        RuntimeError("injected C1 backend synchronization failure"),
    ],
)
def test_local_audit_propagates_oom_and_backend_execution_failures(
    monkeypatch: pytest.MonkeyPatch,
    error: RuntimeError,
) -> None:
    arrays, topology = _synthetic_topology()
    nodes = diagnostic.select_patch_radii_and_nodes(
        topology.patch_guide_distances,
        3,
    )

    def raise_execution_failure(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise error

    monkeypatch.setattr(
        diagnostic,
        "build_augmented_system",
        raise_execution_failure,
    )
    with pytest.raises(type(error), match="injected C1"):
        diagnostic.audit_local_rbf_systems(
            arrays.guide_points_local,
            nodes,
            device="cpu",
        )


def test_actual_topology_builder_matches_line_shortest_paths_and_pl_proxy() -> None:
    arrays = _line_arrays(5)
    topology = diagnostic.build_actual_topology_data(
        arrays,
        dijkstra_source_chunk_size=1,
        patch_chunk_size=2,
        guide_chunk_size=1,
    )
    expected = np.abs(
        np.arange(5, dtype=np.float64)[:, None]
        - np.arange(5, dtype=np.float64)[None, :]
    )
    np.testing.assert_allclose(topology.inputs.guide_distances, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(topology.patch_guide_distances.values, expected, atol=0.0, rtol=0.0)
    assert topology.report["component_coverage"]["passed"] is True
    assert "not_exact_geodesic" in topology.report["distance_semantics"]["D"]
    assert "not_exact_geodesic" in topology.report["distance_semantics"]["M"]


def test_equal_distance_tie_uses_one_shared_voronoi_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices = np.asarray(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    arrays = diagnostic.CheckpointTopologyArrays(
        vertices=vertices,
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
        stored_guide_points_local=vertices[[0, 2]].copy(),
        guide_points_local=vertices[[0, 2]].copy(),
        guide_face_ids=np.asarray([0, 0], dtype=np.int64),
        guide_barycentric=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        guide_point_reconstruction={"passed": True, "max": 0.0, "p95": 0.0},
    )
    original_dijkstra = diagnostic.dijkstra
    assignment_calls = 0

    def counting_dijkstra(*args: object, **kwargs: object) -> object:
        nonlocal assignment_calls
        if kwargs.get("min_only") is True:
            assignment_calls += 1
        return original_dijkstra(*args, **kwargs)

    def forbidden_second_assignment(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("_root_voronoi_graph reran multi-source Dijkstra")

    original_root_graph = diagnostic._root_voronoi_graph
    captured: dict[str, np.ndarray] = {}

    def recording_root_graph(*args: object, **kwargs: object) -> csr_matrix:
        captured["distance"] = np.asarray(kwargs["nearest_distance"]).copy()
        captured["source"] = np.asarray(kwargs["nearest_source"]).copy()
        return original_root_graph(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "dijkstra", counting_dijkstra)
    monkeypatch.setattr(surface_graph, "dijkstra", forbidden_second_assignment)
    monkeypatch.setattr(diagnostic, "_root_voronoi_graph", recording_root_graph)
    topology = diagnostic.build_actual_topology_data(
        arrays,
        dijkstra_source_chunk_size=1,
    )
    assert assignment_calls == 1
    assert topology.inputs.vertex_nearest_distances[1] == 1.0
    assert topology.inputs.vertex_seed_guide_ids[1] in (0, 1)
    vertex_count = vertices.shape[0]
    assert (
        int(captured["source"][1]) - vertex_count
        == int(topology.inputs.vertex_seed_guide_ids[1])
    )
    assert captured["distance"][1] == topology.inputs.vertex_nearest_distances[1]


def test_root_voronoi_precomputed_assignment_pair_is_strict_and_backward_compatible() -> None:
    arrays = _line_arrays(5)
    graph, root_nodes, edge_u, edge_v = surface_graph._augmented_surface_graph(
        arrays.vertices,
        arrays.faces,
        arrays.guide_points_local,
        arrays.guide_face_ids,
    )
    nearest_distance, _, nearest_source = diagnostic.dijkstra(
        graph,
        directed=False,
        indices=root_nodes,
        min_only=True,
        return_predecessors=True,
    )
    legacy = surface_graph._root_voronoi_graph(
        graph,
        root_nodes,
        edge_u,
        edge_v,
    )
    precomputed = surface_graph._root_voronoi_graph(
        graph,
        root_nodes,
        edge_u,
        edge_v,
        nearest_distance=nearest_distance,
        nearest_source=nearest_source,
    )
    np.testing.assert_array_equal(precomputed.indptr, legacy.indptr)
    np.testing.assert_array_equal(precomputed.indices, legacy.indices)
    np.testing.assert_array_equal(precomputed.data, legacy.data)
    with pytest.raises(ValueError, match="supplied together"):
        surface_graph._root_voronoi_graph(
            graph,
            root_nodes,
            edge_u,
            edge_v,
            nearest_distance=nearest_distance,
        )
    with pytest.raises(ValueError, match="wrong shape"):
        surface_graph._root_voronoi_graph(
            graph,
            root_nodes,
            edge_u,
            edge_v,
            nearest_distance=np.asarray(nearest_distance)[:-1],
            nearest_source=nearest_source,
        )


def test_output_directory_refusal_and_explicit_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "occupied.txt").write_text("owned elsewhere", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty"):
        diagnostic.prepare_output_dir(output)
    assert diagnostic.prepare_output_dir(output, overwrite=True) == output.resolve()
    assert (output / "occupied.txt").read_text(encoding="utf-8") == "owned elsewhere"


def test_deterministic_json_is_atomic_and_refuses_implicit_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    diagnostic.write_deterministic_json(path, {"z": 1, "a": [2, 3]})
    first = path.read_bytes()
    assert first == b'{\n  "a": [\n    2,\n    3\n  ],\n  "z": 1\n}\n'
    with pytest.raises(FileExistsError, match="existing result artifact"):
        diagnostic.write_deterministic_json(path, {"a": [2, 3], "z": 1})
    diagnostic.write_deterministic_json(
        path,
        {"a": [2, 3], "z": 1},
        overwrite=True,
    )
    assert path.read_bytes() == first
    assert not list(tmp_path.glob(".*.tmp"))


def test_uncompressed_state_arrays_and_hash_manifest_are_deterministic(
    tmp_path: Path,
) -> None:
    arrays, topology = _synthetic_topology()
    result, artifacts = diagnostic.evaluate_candidate_k(
        topology,
        arrays.guide_points_local,
        3,
        device="cpu",
    )
    assert artifacts is not None
    manifests = []
    for directory_name in ("first", "second"):
        output = tmp_path / directory_name
        output.mkdir()
        report = {
            "schema": diagnostic.SCHEMA,
            "accepted": True,
            "selected_k": 3,
            "candidate": copy.deepcopy(result),
        }
        manifests.append(
            diagnostic.save_diagnostic_outputs(
                output,
                report,
                topology=topology,
                artifacts=artifacts,
                overwrite=False,
            )
        )
    first_hashes = {
        name: value["sha256"]
        for name, value in manifests[0]["artifacts"].items()
        if name.endswith(".npy")
    }
    second_hashes = {
        name: value["sha256"]
        for name, value in manifests[1]["artifacts"].items()
        if name.endswith(".npy")
    }
    assert first_hashes == second_hashes
    assert tuple(first_hashes) == diagnostic.STATE_ARRAY_NAMES
    for name, expected in diagnostic.serialized_state_arrays(topology, artifacts).items():
        loaded = np.load(tmp_path / "first" / name, allow_pickle=False)
        np.testing.assert_array_equal(loaded, expected)
    assert artifacts.serialized_state_bytes == sum(
        (tmp_path / "first" / name).stat().st_size
        for name in diagnostic.STATE_ARRAY_NAMES
    )
    manifest_on_disk = json.loads(
        (tmp_path / "first" / diagnostic.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest_on_disk == manifests[0]
    assert not list((tmp_path / "first").glob("*.npz"))


def test_rejected_candidate_persists_and_hashes_reusable_base_topology_state(
    tmp_path: Path,
) -> None:
    _, topology = _synthetic_topology()
    output = tmp_path / "rejected"
    output.mkdir()
    report = {
        "schema": diagnostic.SCHEMA,
        "accepted": False,
        "selected_k": None,
        "timings_seconds": {},
    }
    manifest = diagnostic.save_diagnostic_outputs(
        output,
        report,
        topology=topology,
        artifacts=None,
        overwrite=False,
    )
    assert tuple(report["artifacts"]) == diagnostic.BASE_STATE_ARRAY_NAMES
    assert set(manifest["artifacts"]) == {
        *diagnostic.BASE_STATE_ARRAY_NAMES,
        diagnostic.REPORT_NAME,
    }
    expected_arrays = diagnostic.base_topology_state_arrays(topology)
    for name, expected in expected_arrays.items():
        path = output / name
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)
        assert manifest["artifacts"][name]["sha256"] == diagnostic.sha256_file(path)
        assert (
            report["artifacts"][name]["array"]["sha256"]
            == diagnostic.array_identity(expected)["sha256"]
        )
    assert not any((output / name).exists() for name in diagnostic.SELECTED_STATE_ARRAY_NAMES)


def test_staging_write_failure_preserves_prior_named_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arrays, topology = _synthetic_topology()
    _, artifacts = diagnostic.evaluate_candidate_k(
        topology,
        arrays.guide_points_local,
        3,
        device="cpu",
    )
    assert artifacts is not None
    output = tmp_path / "stable-result"
    output.mkdir()
    diagnostic.save_diagnostic_outputs(
        output,
        {"schema": diagnostic.SCHEMA, "accepted": True, "timings_seconds": {}},
        topology=topology,
        artifacts=artifacts,
        overwrite=False,
    )
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
        if calls == 2:
            raise OSError("injected staged write failure")
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(diagnostic, "write_atomic_npy", failing_writer)
    with pytest.raises(OSError, match="staged write failure"):
        diagnostic.save_diagnostic_outputs(
            output,
            {"schema": diagnostic.SCHEMA, "accepted": False, "timings_seconds": {}},
            topology=topology,
            artifacts=None,
            overwrite=True,
        )
    after = {
        name: (output / name).read_bytes()
        for name in diagnostic.ARTIFACT_NAMES
        if (output / name).is_file()
    }
    assert after == before
    assert not list(tmp_path.glob(".stable-result.r084-c1-stage.*"))


def test_mocked_actual_checkpoint_run_selects_first_k_and_writes_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"phase-c1-checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    arrays = _line_arrays(10)
    stored = arrays.guide_points_local.copy()
    stored[0, 1] += 0.5 * diagnostic.MAX_GUIDE_POINT_RECONSTRUCTION_ERROR
    model = _fake_model(arrays, stored_guide_points=stored)
    local_point_inputs: list[np.ndarray] = []
    original_local_audit = diagnostic.audit_local_rbf_systems

    def recording_local_audit(
        points: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        local_point_inputs.append(np.asarray(points).copy())
        return original_local_audit(points, *args, **kwargs)

    monkeypatch.setattr(
        diagnostic,
        "get_clean_source_git_identity",
        lambda *_args, **_kwargs: {
            "repository": str(diagnostic.PROJECT_ROOT),
            "head": "a" * 40,
            "porcelain_status": "",
            "clean": True,
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "_load_stage1_checkpoint_model",
        lambda: (
            lambda _path, _device: (
                model,
                SimpleNamespace(name="fake-config"),
                {"iteration": 7},
            )
        ),
    )
    monkeypatch.setattr(diagnostic, "audit_local_rbf_systems", recording_local_audit)
    report = diagnostic.run_checkpoint_partition_cover_diagnostic(
        checkpoint=checkpoint,
        output_dir=tmp_path / "output",
        device="cpu",
        expected_checkpoint_sha256=checkpoint_hash,
        expected_iteration=7,
        expected_source_commit="a" * 40,
    )
    assert report["accepted"] is True
    assert report["selected_k"] == 8
    assert report["candidate_results"][0]["all_hard_gates_passed"] is True
    assert local_point_inputs
    assert all(
        np.array_equal(points, arrays.guide_points_local)
        for points in local_point_inputs
    )
    assert (
        report["identities"]["model_arrays"]["stored_guide_points_local"]["sha256"]
        != report["identities"]["model_arrays"]["canonical_guide_points_local"]["sha256"]
    )
    expected_d = np.abs(
        np.arange(10, dtype=np.float64)[:, None]
        - np.arange(10, dtype=np.float64)[None, :]
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "output" / "guide_distances.npy", allow_pickle=False),
        expected_d,
    )
    assert (tmp_path / "output" / diagnostic.REPORT_NAME).is_file()
    assert (tmp_path / "output" / diagnostic.MANIFEST_NAME).is_file()
    assert all((tmp_path / "output" / name).is_file() for name in diagnostic.STATE_ARRAY_NAMES)
    persisted = json.loads(
        (tmp_path / "output" / diagnostic.REPORT_NAME).read_text(encoding="utf-8")
    )
    assert persisted == report
    assert report["timings_seconds"]["staged_state_array_write"] >= 0.0


def test_mocked_no_k_pass_run_persists_base_state_and_exact_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "rejected-checkpoint.pt"
    checkpoint.write_bytes(b"phase-c1-rejected-checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    arrays = _line_arrays(5)
    model = _fake_model(arrays)
    monkeypatch.setattr(
        diagnostic,
        "get_clean_source_git_identity",
        lambda *_args, **_kwargs: {
            "repository": str(diagnostic.PROJECT_ROOT),
            "head": "b" * 40,
            "porcelain_status": "",
            "clean": True,
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "_load_stage1_checkpoint_model",
        lambda: (
            lambda _path, _device: (
                model,
                SimpleNamespace(name="fake-config"),
                {"iteration": 9},
            )
        ),
    )
    output = tmp_path / "rejected-output"
    report = diagnostic.run_checkpoint_partition_cover_diagnostic(
        checkpoint=checkpoint,
        output_dir=output,
        device="cpu",
        expected_checkpoint_sha256=checkpoint_hash,
        expected_iteration=9,
        expected_source_commit="b" * 40,
    )
    assert report["accepted"] is False
    assert report["selected_k"] is None
    assert all(
        result["status"] == "rejected"
        for result in report["candidate_results"]
    )
    assert tuple(report["artifacts"]) == diagnostic.BASE_STATE_ARRAY_NAMES
    for name in diagnostic.BASE_STATE_ARRAY_NAMES:
        path = output / name
        assert path.is_file()
        assert report["artifacts"][name]["sha256"] == diagnostic.sha256_file(path)
    assert not any((output / name).exists() for name in diagnostic.SELECTED_STATE_ARRAY_NAMES)
    np.testing.assert_array_equal(
        np.load(output / "vertex_seed_guide_ids.npy", allow_pickle=False),
        np.arange(5, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.load(output / "vertex_nearest_distances.npy", allow_pickle=False),
        np.zeros((5,), dtype=np.float64),
    )


def test_outer_run_propagates_backend_failure_without_topology_relabel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "backend-checkpoint.pt"
    checkpoint.write_bytes(b"phase-c1-backend-checkpoint")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    arrays, topology = _synthetic_topology()
    model = _fake_model(arrays)
    monkeypatch.setattr(
        diagnostic,
        "get_clean_source_git_identity",
        lambda *_args, **_kwargs: {
            "repository": str(diagnostic.PROJECT_ROOT),
            "head": "c" * 40,
            "porcelain_status": "",
            "clean": True,
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "_load_stage1_checkpoint_model",
        lambda: (
            lambda _path, _device: (
                model,
                SimpleNamespace(name="fake-config"),
                {"iteration": 11},
            )
        ),
    )
    monkeypatch.setattr(
        diagnostic,
        "build_actual_topology_data",
        lambda *_args, **_kwargs: topology,
    )

    def raise_backend(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected outer backend failure")

    monkeypatch.setattr(diagnostic, "scan_candidate_ks", raise_backend)
    output = tmp_path / "backend-output"
    with pytest.raises(RuntimeError, match="outer backend failure"):
        diagnostic.run_checkpoint_partition_cover_diagnostic(
            checkpoint=checkpoint,
            output_dir=output,
            device="cpu",
            expected_checkpoint_sha256=checkpoint_hash,
            expected_iteration=11,
            expected_source_commit="c" * 40,
        )
    assert not (output / diagnostic.REPORT_NAME).exists()
    assert not (output / diagnostic.MANIFEST_NAME).exists()


def test_requested_cuda_has_no_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostic.torch.cuda, "is_available", lambda: False)
    with pytest.raises(diagnostic.DiagnosticError, match="no CPU fallback"):
        diagnostic._resolve_device("cuda")
    assert diagnostic._resolve_device("cpu") == torch.device("cpu")


def test_import_surface_has_no_training_rendering_or_image_side_effects() -> None:
    source_path = Path(diagnostic.__file__).resolve()
    module_ast = ast.parse(source_path.read_text(encoding="utf-8"))
    top_level_imports = {
        node.module
        for node in module_ast.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "tools.train_white_tiger_stage1" not in top_level_imports
    assert not any("render" in name or "image" in name for name in top_level_imports)
    assert all(
        name.endswith(".npy") or name.endswith(".json")
        for name in diagnostic.ARTIFACT_NAMES
    )
    assert "tools.train_white_tiger_stage1" not in diagnostic.__dict__
    assert diagnostic.run_checkpoint_partition_cover_diagnostic.__module__ in sys.modules

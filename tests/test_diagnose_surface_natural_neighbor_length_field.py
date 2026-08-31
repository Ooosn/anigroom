from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools import diagnose_surface_natural_neighbor_length_field as diagnostic
from tools import surface_natural_neighbor_io as r083_io


def _output_bytes(
    *,
    guide_count: int = 4,
    row_offsets: tuple[int, ...] = (0, 2, 3),
    guide_ids: tuple[int, ...] = (1, 3, 0),
    weights: tuple[float, ...] = (0.25, 0.75, 1.0),
    barycentric_errors: tuple[float, ...] = (0.0, 0.1),
    success: tuple[int, ...] | None = None,
) -> bytes:
    query_count = len(row_offsets) - 1
    if success is None:
        success = (1,) * query_count
    nnz = len(guide_ids)
    method = r083_io.METHOD_IDENTITY_BYTES + b"\0" * (
        r083_io.OUTPUT_METHOD_BYTES - len(r083_io.METHOD_IDENTITY_BYTES)
    )
    header = r083_io.OUTPUT_HEADER.pack(
        r083_io.OUTPUT_MAGIC,
        r083_io.FORMAT_VERSION,
        r083_io.OUTPUT_HEADER_SIZE,
        guide_count,
        query_count,
        nnz,
        method,
    )
    return b"".join(
        [
            header,
            np.asarray(row_offsets, dtype="<u8").tobytes(),
            np.asarray(guide_ids, dtype="<u4").tobytes(),
            np.asarray(weights, dtype="<f8").tobytes(),
            np.asarray(success, dtype="<u1").tobytes(),
            np.asarray(barycentric_errors, dtype="<f8").tobytes(),
        ]
    )


def _read_output(tmp_path: Path, **kwargs: object) -> r083_io.SurfaceNaturalNeighborOutput:
    path = tmp_path / "output.bin"
    path.write_bytes(_output_bytes(**kwargs))
    return r083_io.read_output(path)


def _identity_output(tmp_path: Path, **kwargs: object) -> r083_io.SurfaceNaturalNeighborOutput:
    defaults: dict[str, object] = {
        "row_offsets": (0, 1, 2, 3, 4),
        "guide_ids": (0, 1, 2, 3),
        "weights": (1.0, 1.0, 1.0, 1.0),
        "barycentric_errors": (0.0, 0.0, 0.0, 0.0),
    }
    defaults.update(kwargs)
    path = tmp_path / "identity.bin"
    path.write_bytes(_output_bytes(**defaults))
    return r083_io.read_output(path)


def _manual_identity_output(
    barycentric_errors: np.ndarray,
) -> r083_io.SurfaceNaturalNeighborOutput:
    return r083_io.SurfaceNaturalNeighborOutput(
        guide_count=4,
        query_count=4,
        nnz=4,
        method=r083_io.METHOD_IDENTITY,
        row_offsets=np.asarray([0, 1, 2, 3, 4], dtype="<u8"),
        guide_ids=np.asarray([0, 1, 2, 3], dtype="<u4"),
        weights=np.ones((4,), dtype="<f8"),
        success=np.ones((4,), dtype=np.bool_),
        barycentric_errors=np.asarray(barycentric_errors, dtype="<f8"),
    )


def test_stable_root_selection_has_known_ids_and_is_seed_sensitive() -> None:
    selected = diagnostic.select_render_root_ids(20, 7, 20260901)
    assert selected.tolist() == [0, 2, 4, 5, 16, 17, 19]
    assert selected.tolist() == sorted(selected.tolist())
    assert len(set(selected.tolist())) == 7
    assert diagnostic.select_render_root_ids(20, 7, 20260902).tolist() == [
        3,
        4,
        8,
        12,
        13,
        15,
        16,
    ]
    assert diagnostic.stable_root_rank(0, 20260901) == 940296622445009932


@pytest.mark.parametrize(
    ("population", "requested", "message"),
    [
        (0, 0, "population is empty"),
        (10, -1, "positive"),
        (10, 11, "exceeds render population"),
    ],
)
def test_selection_count_errors_are_explicit(
    population: int,
    requested: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        diagnostic.select_render_root_ids(population, requested, 20260901)


def test_explicit_zero_selection_expands_to_every_population_root() -> None:
    selected = diagnostic.select_render_root_ids(6, 0, 20260901)
    assert selected.tolist() == [0, 1, 2, 3, 4, 5]


def test_explicit_zero_means_complete_population_and_default_is_bounded() -> None:
    assert diagnostic._selection_arguments(None, 20260901, 128)[0] == 4096
    assert diagnostic._selection_arguments(0, 20260901, 128)[0] == 0
    assert diagnostic._selection_arguments(17, 20260901, 128) == (17, 20260901, 128)
    parser = diagnostic.build_argument_parser()
    default_args = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint",
            "--builder",
            "builder",
            "--output-dir",
            "out",
            "--expected-checkpoint-sha256",
            "a" * 64,
            "--expected-iteration",
            "4000",
            "--expected-source-commit",
            "b" * 40,
            "--expected-builder-sha256",
            "c" * 64,
        ]
    )
    zero_args = parser.parse_args(
        [
            "--checkpoint",
            "checkpoint",
            "--builder",
            "builder",
            "--output-dir",
            "out",
            "--render-query-count",
            "0",
            "--expected-checkpoint-sha256",
            "a" * 64,
            "--expected-iteration",
            "4000",
            "--expected-source-commit",
            "b" * 40,
            "--expected-builder-sha256",
            "c" * 64,
        ]
    )
    assert default_args.render_query_count == 4096
    assert zero_args.render_query_count == 0
    assert default_args.builder_timeout_seconds == 300.0
    assert default_args.topology_candidate_audit_k == 128


def test_cli_requires_checkpoint_identity_and_iteration() -> None:
    parser = diagnostic.build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--checkpoint", "checkpoint", "--builder", "builder", "--output-dir", "out"]
        )


def test_checkpoint_identity_and_iteration_gate_is_strict() -> None:
    checkpoint_hash = "ab" * 32
    accepted = diagnostic.validate_checkpoint_expectations(
        checkpoint_hash,
        checkpoint_hash.upper(),
        {"iteration": 4000},
        4000,
    )
    assert accepted == {
        "expected_sha256": checkpoint_hash,
        "observed_sha256": checkpoint_hash,
        "expected_iteration": 4000,
        "observed_iteration": 4000,
        "passed": True,
    }
    with pytest.raises(diagnostic.DiagnosticError, match="mandatory"):
        diagnostic.validate_checkpoint_expectations(
            checkpoint_hash,
            None,
            {"iteration": 4000},
            4000,
        )
    with pytest.raises(diagnostic.DiagnosticError, match="mandatory"):
        diagnostic.validate_checkpoint_expectations(
            checkpoint_hash,
            checkpoint_hash,
            {"iteration": 4000},
            None,
        )
    with pytest.raises(diagnostic.DiagnosticError, match="mismatch"):
        diagnostic.validate_checkpoint_expectations(
            checkpoint_hash,
            checkpoint_hash,
            {"iteration": 3999},
            4000,
        )
    with pytest.raises(diagnostic.DiagnosticError, match="missing"):
        diagnostic.validate_checkpoint_expectations(
            checkpoint_hash,
            checkpoint_hash,
            {},
            4000,
        )


def test_source_and_builder_identities_are_mandatory_and_exact() -> None:
    source_commit = "a" * 40
    builder_hash = "b" * 64
    accepted = diagnostic.validate_source_builder_expectations(
        source_commit,
        source_commit.upper(),
        builder_hash,
        builder_hash.upper(),
    )
    assert accepted == {
        "source": {
            "expected": source_commit,
            "observed": source_commit,
            "passed": True,
        },
        "builder": {
            "expected_sha256": builder_hash,
            "observed_sha256": builder_hash,
            "passed": True,
        },
        "passed": True,
    }
    assert diagnostic.normalize_git_commit("c" * 64) == "c" * 64
    with pytest.raises(ValueError, match="40 or 64"):
        diagnostic.normalize_git_commit("a" * 41)
    with pytest.raises(diagnostic.IdentityExpectationError, match="source HEAD") as source_error:
        diagnostic.validate_source_builder_expectations(
            "d" * 40,
            source_commit,
            builder_hash,
            builder_hash,
        )
    assert source_error.value.report["source"]["passed"] is False
    with pytest.raises(diagnostic.IdentityExpectationError, match="builder bytes") as builder_error:
        diagnostic.validate_source_builder_expectations(
            source_commit,
            source_commit,
            "e" * 64,
            builder_hash,
        )
    assert builder_error.value.report["builder"]["passed"] is False
    with pytest.raises(diagnostic.DiagnosticError, match="expected-source-commit"):
        diagnostic.validate_source_builder_expectations(
            source_commit,
            None,
            builder_hash,
            builder_hash,
        )
    with pytest.raises(diagnostic.DiagnosticError, match="expected-builder-sha256"):
        diagnostic.validate_source_builder_expectations(
            source_commit,
            source_commit,
            builder_hash,
            None,
        )


def test_full_population_estimate_boundary_and_gate_aggregation() -> None:
    at_limit = diagnostic.estimate_full_population_wall_seconds(1, 2, 1800.0)
    over_limit = diagnostic.estimate_full_population_wall_seconds(1, 2, 1800.1)
    assert at_limit["predeclared_limit_seconds"] == 3600.0
    assert at_limit["estimated_sequential_full_query_wall_seconds"] == 3600.0
    assert at_limit["within_limit"] is True
    assert over_limit["within_limit"] is False
    validation = {
        key: True for key in diagnostic.PREDECLARED_HARD_GATE_KEYS
    }
    validation["candidate_vs_legacy_magnitude"] = False
    assert diagnostic.aggregate_predeclared_subset_gates(validation) is True
    validation["builder_identity"] = False
    assert diagnostic.aggregate_predeclared_subset_gates(validation) is False
    with pytest.raises(diagnostic.DiagnosticError, match="missing keys"):
        diagnostic.aggregate_predeclared_subset_gates({})


def test_csr_matvec_matches_dense_direct_weighted_sum_and_stays_positive(
    tmp_path: Path,
) -> None:
    output = _read_output(tmp_path)
    source = np.asarray([10.0, 20.0, 30.0, 40.0])
    actual = diagnostic.csr_weighted_sum(output, source)
    expected = np.asarray([0.25 * 20.0 + 0.75 * 40.0, 10.0])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert bool(np.isfinite(actual).all())
    assert bool((actual > 0.0).all())
    np.testing.assert_allclose(diagnostic._csr_row_sums(output), [1.0, 1.0], rtol=0.0, atol=0.0)


def test_guide_site_one_hot_identity_accepts_exact_rows(tmp_path: Path) -> None:
    output = _identity_output(tmp_path)
    report = diagnostic.validate_guide_site_identity(
        output,
        np.asarray([0.1, 0.2, 0.3, 0.4]),
    )
    assert report["passed"] is True
    assert report["absolute_error"]["max"] == 0.0
    assert report["row_count"] == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"guide_ids": (1, 1, 2, 3)},
        {"weights": (1.0 + 2.0e-12, 1.0, 1.0, 1.0)},
        {
            "row_offsets": (0, 2, 3, 4, 5),
            "guide_ids": (0, 1, 2, 3, 0),
            "weights": (0.5, 0.5, 1.0, 1.0, 1.0),
        },
    ],
)
def test_guide_site_one_hot_identity_rejects_nonidentity_rows(
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    output = _identity_output(tmp_path, **kwargs)
    with pytest.raises(diagnostic.GuideIdentityError, match="Kronecker") as error:
        diagnostic.validate_guide_site_identity(
            output,
            np.asarray([0.1, 0.2, 0.3, 0.4]),
        )
    assert error.value.report["passed"] is False


@pytest.mark.parametrize(
    ("errors", "expected_error"),
    [
        (np.asarray([np.nan, 0.0, 0.0, 0.0]), None),
        (np.asarray([2.0 * diagnostic.MACHINE_TOLERANCE, 0.0, 0.0, 0.0]), 2.0e-12),
    ],
)
def test_guide_site_identity_rejects_nonfinite_or_large_barycentric_error(
    errors: np.ndarray,
    expected_error: float | None,
) -> None:
    output = _manual_identity_output(errors)
    with pytest.raises(diagnostic.GuideIdentityError, match="barycentric") as error:
        diagnostic.validate_guide_site_identity(
            output,
            np.asarray([0.1, 0.2, 0.3, 0.4]),
        )
    assert error.value.report["passed"] is False
    assert error.value.report["bad_rows"] == [
        {"query_id": 0, "error": expected_error}
    ]


def test_topology_candidate_containment_reports_exact_missing_ids_on_failure(
    tmp_path: Path,
) -> None:
    output = _read_output(tmp_path)
    support_report = diagnostic.validate_topology_safe_candidate_support(
        np.asarray([[1, 3], [0, 2]], dtype=np.int64),
        query_count=2,
        guide_count=4,
        fallback_query_count=0,
    )
    support_report.update(
        {
            "label": "existing_topology_safe_K2_candidate_support",
            "certificate": (
                "containment_only_within_existing_topology_safe_candidate_support"
            ),
        }
    )
    accepted = diagnostic.audit_topology_safe_candidate_containment(
        output,
        np.asarray([[1, 3], [0, 2]], dtype=np.int64),
        support_report=support_report,
    )
    assert accepted["passed"] is True
    assert accepted["contained_neighbor_count"] == 3
    assert accepted["returned_neighbor_count"] == 3
    assert accepted["contained_fraction"] == 1.0
    assert accepted["support_validation"]["label"] == (
        "existing_topology_safe_K2_candidate_support"
    )

    with pytest.raises(diagnostic.TopologyCandidateContainmentError, match="3") as error:
        diagnostic.audit_topology_safe_candidate_containment(
            output,
            np.asarray([[1], [0]], dtype=np.int64),
        )
    assert error.value.report["missing_ids"] == [3]
    assert error.value.report["missing_by_query"] == [
        {"query_id": 0, "guide_ids": [3]}
    ]


def test_topology_candidate_support_rejects_duplicate_padding_and_fallback_coverage() -> None:
    with pytest.raises(
        diagnostic.TopologyCandidateContainmentError,
        match="duplicate/padded",
    ):
        diagnostic.validate_topology_safe_candidate_support(
            np.asarray([[1, 1], [0, 2]]),
            query_count=2,
            guide_count=4,
            fallback_query_count=0,
        )
    with pytest.raises(diagnostic.TopologyCandidateContainmentError, match="fallback"):
        diagnostic.validate_topology_safe_candidate_support(
            np.asarray([[1], [0]]),
            query_count=2,
            guide_count=4,
            fallback_query_count=1,
        )


def test_legacy_lengths_slice_canonical_support_and_disable_grad() -> None:
    import torch

    from anigroom.surface_interpolation import SurfaceSupport

    class FakeModel:
        def __init__(self) -> None:
            self.guide_points_local = torch.zeros(
                (4, 3),
                dtype=torch.float32,
                requires_grad=True,
            )
            self.face_ids = torch.arange(4, dtype=torch.long)
            self._canonical_support = SurfaceSupport(
                indices=torch.tensor(
                    [[10, 11], [20, 21], [30, 31], [40, 41]],
                    dtype=torch.long,
                ),
                vertex_path_distances=torch.zeros((4, 2, 3), dtype=torch.float32),
                report={"neighbor_count": 2, "fallback_query_count": 0},
            )
            self.received_support: SurfaceSupport | None = None
            self.received_length: torch.Tensor | None = None

        def guide_interpolation_support(self) -> SurfaceSupport:
            return self._canonical_support

        def tangent_frames_for_face_ids(
            self,
            face_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            shape = (int(face_ids.shape[0]), 3)
            normals = torch.zeros(shape, dtype=torch.float32)
            normals[:, 2] = 1.0
            tangents = torch.zeros_like(normals)
            tangents[:, 0] = 1.0
            bitangents = torch.zeros_like(normals)
            bitangents[:, 1] = 1.0
            return normals, tangents, bitangents

        def sample_guide_controls(self, points, query_faces, normals, tangents, bitangents, *, support):
            self.received_support = support
            self.received_length = points[:, :1] + self.guide_points_local[:1, :1] + 1.0
            return {"length": self.received_length}, None

    model = FakeModel()
    lengths, support, report = diagnostic._legacy_selected_lengths(
        model,
        np.zeros((2, 3), dtype=np.float64),
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([3, 1], dtype=np.int64),
    )
    assert model.received_support is not None
    assert model.received_support.indices.tolist() == [[40, 41], [20, 21]]
    assert support.report["provenance"] == (
        "canonical_full_render_support_sliced_by_selected_root_ids"
    )
    assert report["canonical_query_count"] == 4
    assert report["selected_query_count"] == 2
    assert model.received_length is not None
    assert model.received_length.requires_grad is False
    assert lengths.tolist() == [1.0, 1.0]


def test_normal_contract_accepts_float32_model_normals_and_rejects_outside_band() -> None:
    model_normal = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    model_normal /= np.linalg.norm(model_normal, axis=1, keepdims=True)
    accepted = diagnostic.validate_unit_normals(model_normal, "model normals")
    assert accepted["passed"] is True
    assert accepted["tolerance"] == 1.0e-5
    with pytest.raises(diagnostic.DiagnosticError, match="not unit length"):
        diagnostic.validate_unit_normals(
            np.asarray([[1.0 + 1.01e-5, 0.0, 0.0]]),
            "bad normals",
        )


def test_output_directory_refusal_and_explicit_overwrite(tmp_path: Path) -> None:
    output_dir = tmp_path / "results"
    assert diagnostic.prepare_output_dir(output_dir) == output_dir.resolve()
    (output_dir / "report.json").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="nonempty"):
        diagnostic.prepare_output_dir(output_dir)
    assert diagnostic.prepare_output_dir(output_dir, overwrite=True) == output_dir.resolve()
    assert (output_dir / "report.json").read_text(encoding="utf-8") == "old"


def test_builder_argv_and_bounded_subprocess_transport(tmp_path: Path) -> None:
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    argv = diagnostic.build_builder_argv(sys.executable, input_path, output_path)
    assert argv == [
        str(Path(sys.executable).resolve()),
        "--input",
        str(input_path.resolve()),
        "--output",
        str(output_path.resolve()),
    ]
    stdout_log = tmp_path / "stdout.log"
    stderr_log = tmp_path / "stderr.log"
    fixture_code = (
        "import sys; print('transport-out-' + 'x'*100); "
        "print('transport-err-' + 'y'*100, file=sys.stderr)"
    )
    result = diagnostic.run_builder(
        [sys.executable, "-c", fixture_code],
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        max_log_bytes=16,
    )
    assert result.returncode == 0
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert stdout_log.stat().st_size <= 16
    assert stderr_log.stat().st_size <= 16
    assert "transport-out" in stdout_log.read_text(encoding="utf-8")
    assert "transport-err" in stderr_log.read_text(encoding="utf-8")


def test_hanging_builder_timeout_cleans_direct_child_and_logs(tmp_path: Path) -> None:
    stdout_log = tmp_path / "timeout.stdout.log"
    stderr_log = tmp_path / "timeout.stderr.log"
    fixture_code = "import time; print('started', flush=True); time.sleep(60)"
    with pytest.raises(diagnostic.BuilderTimeoutError, match="timed out") as error:
        diagnostic.run_builder(
            [sys.executable, "-c", fixture_code],
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            timeout_seconds=0.1,
            max_log_bytes=128,
        )
    assert error.value.process.poll() is not None
    assert error.value.result.timeout_seconds == 0.1
    assert stdout_log.exists()
    assert stderr_log.exists()
    assert stdout_log.stat().st_size <= 128
    assert stderr_log.stat().st_size <= 128
    assert "started" in stdout_log.read_text(encoding="utf-8")


def test_clean_source_git_identity_records_head_and_empty_porcelain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        if argv[1] == "rev-parse":
            return subprocess.CompletedProcess(argv, 0, stdout="a" * 40 + "\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(diagnostic.subprocess, "run", fake_run)
    identity = diagnostic.get_clean_source_git_identity(tmp_path)
    assert identity == {
        "repository": str(tmp_path.resolve()),
        "head": "a" * 40,
        "porcelain_status": "",
        "clean": True,
    }
    assert len(calls) == 2
    assert all(call[1]["shell"] is False for call in calls)
    assert all(call[1]["timeout"] == 30.0 for call in calls)
    assert calls[0][0] == ["git", "rev-parse", "--verify", "HEAD"]
    assert calls[1][0] == [
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ]


def test_source_git_identity_rejects_dirty_or_unavailable_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dirty_run(argv, **kwargs):
        if argv[1] == "rev-parse":
            return subprocess.CompletedProcess(argv, 0, stdout="b" * 40 + "\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout=" M owned.py\n", stderr="")

    monkeypatch.setattr(diagnostic.subprocess, "run", dirty_run)
    with pytest.raises(diagnostic.DiagnosticError, match="dirty"):
        diagnostic.get_clean_source_git_identity(tmp_path)

    def unavailable_run(argv, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(diagnostic.subprocess, "run", unavailable_run)
    with pytest.raises(diagnostic.DiagnosticError, match="unavailable"):
        diagnostic.get_clean_source_git_identity(tmp_path)


def test_deterministic_json_writer_is_canonical_and_atomic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_first = {"z": [2, 1], "a": {"y": 3, "x": 4}}
    write_second = {"a": {"x": 4, "y": 3}, "z": [2, 1]}
    diagnostic.write_deterministic_json(first, write_first)
    diagnostic.write_deterministic_json(second, write_second)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == write_first
    with pytest.raises(FileExistsError, match="existing result artifact"):
        diagnostic.write_deterministic_json(first, write_first)
    diagnostic.write_deterministic_json(first, {"replaced": True}, overwrite=True)
    assert json.loads(first.read_text(encoding="utf-8")) == {"replaced": True}


def test_hash_and_array_identity_helpers_are_exact(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"r083")
    assert diagnostic.sha256_file(path) == (
        "fa145e12ed928e6f81ca4c536249d29aa23232975788b4e72adfb3c63ba1256c"
    )
    identity = diagnostic.array_identity(np.asarray([[1.0, 2.0]], dtype=np.float64))
    assert identity["shape"] == [1, 2]
    assert identity["bytes"] == 16
    assert len(identity["sha256"]) == 64

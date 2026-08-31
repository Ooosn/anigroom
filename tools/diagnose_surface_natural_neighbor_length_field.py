"""Bounded R083 actual-checkpoint natural-neighbor length-field diagnostic.

This is diagnostic infrastructure only.  It loads one exact Stage-1
checkpoint, evaluates the fixed 4,500 primary guides and a deterministic
render-root subset through the standalone CGAL builder, and writes numeric
JSON/log/bin artifacts.  It does not train, render, mutate checkpoints, edit
formal configuration, or authorize a full-population run by default.

The render selection rank is a documented stable 64-bit SplitMix64 hash:

    x = seed XOR (root_id * 0xd6e8feb86659fd93)  (mod 2**64)
    rank = SplitMix64(x)

Roots are ranked by ``(rank, root_id)`` and the selected root IDs are sorted
before they become query rows.  This avoids Python's process-randomized hash
and makes selection reproducible across processes and platforms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.surface_natural_neighbor_io import (  # noqa: E402
    METHOD_IDENTITY,
    NORMAL_NORM_TOLERANCE,
    ROW_SUM_TOLERANCE,
    SurfaceNaturalNeighborFormatError,
    SurfaceNaturalNeighborOutput,
    read_surface_natural_neighbor_output,
    sha256_file,
    write_surface_natural_neighbor_input,
)


SCHEMA = "r083.surface_natural_neighbor_length_field.phase_a_subset.v1"
EXPECTED_GUIDE_COUNT = 4500
DEFAULT_RENDER_QUERY_COUNT = 4096
DEFAULT_SELECTION_SEED = 20260901
DEFAULT_TOPOLOGY_CANDIDATE_AUDIT_K = 128
MACHINE_TOLERANCE = 1.0e-12
DEFAULT_BUILDER_TIMEOUT_SECONDS = 300.0
PREDECLARED_FULL_ESTIMATE_LIMIT_SECONDS = 3600.0
BUILDER_TERMINATION_GRACE_SECONDS = 2.0
MAX_BUILDER_LOG_BYTES = 4 * 1024 * 1024
UINT64_MASK = (1 << 64) - 1
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
GIT_HEAD_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")

ARTIFACT_NAMES = (
    "guide_sites.input.bin",
    "guide_sites.output.bin",
    "guide_sites.stdout.log",
    "guide_sites.stderr.log",
    "render_subset.input.bin",
    "render_subset.output.bin",
    "render_subset.stdout.log",
    "render_subset.stderr.log",
    "report.json",
)

PREDECLARED_HARD_GATE_KEYS = (
    "source_identity",
    "checkpoint_identity",
    "builder_identity",
    "builder_invocations_within_300_seconds",
    "guide_site_one_hot_identity",
    "guide_site_length_identity",
    "guide_normals_finite_unit",
    "render_normals_finite_unit",
    "render_csr_rows_positive_normalized",
    "topology_safe_candidate_support_containment",
    "topology_safe_candidate_support_no_fallback",
    "topology_safe_candidate_support_no_duplicate_or_padding",
    "full_estimate_within_3600_seconds",
)


class DiagnosticError(RuntimeError):
    """Raised when the bounded checkpoint diagnostic cannot pass its contract."""


class TopologyCandidateContainmentError(DiagnosticError):
    """A CGAL row contains a guide outside the topology-safe candidate support."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


class GuideIdentityError(DiagnosticError):
    """Guide-site CSR rows failed the exact one-hot identity contract."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


class IdentityExpectationError(DiagnosticError):
    """A required source, checkpoint, or builder identity did not match."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


class BuilderTimeoutError(DiagnosticError):
    """A builder exceeded its finite timeout after process cleanup."""

    def __init__(self, message: str, process: Any, result: "BuilderRun") -> None:
        super().__init__(message)
        self.process = process
        self.result = result


@dataclass(frozen=True)
class CheckpointArrays:
    guide_points_local: np.ndarray
    guide_normals_local: np.ndarray
    guide_face_ids: np.ndarray
    render_points_local: np.ndarray
    render_normals_local: np.ndarray
    render_face_ids: np.ndarray
    guide_lengths: np.ndarray


@dataclass(frozen=True)
class BuilderRun:
    argv: tuple[str, ...]
    returncode: int
    seconds: float
    timeout_seconds: float
    stdout_log: str
    stderr_log: str
    stdout_bytes_written: int
    stderr_bytes_written: int
    stdout_truncated: bool
    stderr_truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "seconds": self.seconds,
            "timeout_seconds": self.timeout_seconds,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "stdout_bytes_written": self.stdout_bytes_written,
            "stderr_bytes_written": self.stderr_bytes_written,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
        }


def _checked_u64(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    converted = int(value)
    if converted < 0 or converted > UINT64_MASK:
        raise ValueError(f"{name} must lie in [0, 2**64-1]")
    return converted


def normalize_sha256(value: str) -> str:
    normalized = str(value).strip()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError("expected checkpoint SHA256 must be exactly 64 hexadecimal characters")
    return normalized.lower()


def normalize_git_commit(value: str) -> str:
    normalized = str(value).strip().lower()
    if GIT_HEAD_RE.fullmatch(normalized) is None:
        raise ValueError("source commit must be exactly 40 or 64 hexadecimal characters")
    return normalized


def validate_checkpoint_expectations(
    checkpoint_sha256: str,
    expected_checkpoint_sha256: str | None,
    checkpoint_data: dict[str, Any],
    expected_iteration: int | None,
) -> dict[str, Any]:
    """Require the exact checkpoint hash and iteration for this gate."""

    if expected_checkpoint_sha256 is None:
        raise DiagnosticError(
            "expected-checkpoint-sha256 is mandatory for the actual-checkpoint gate"
        )
    expected_hash = normalize_sha256(expected_checkpoint_sha256)
    observed_hash = normalize_sha256(checkpoint_sha256)
    if observed_hash != expected_hash:
        raise DiagnosticError(
            f"checkpoint SHA256 mismatch: expected {expected_hash}, got {observed_hash}"
        )
    if expected_iteration is None:
        raise DiagnosticError(
            "expected-iteration is mandatory for the actual-checkpoint gate"
        )
    if isinstance(expected_iteration, bool) or not isinstance(
        expected_iteration, (int, np.integer)
    ):
        raise DiagnosticError("expected-iteration must be a non-negative integer")
    expected_iter = int(expected_iteration)
    if expected_iter < 0:
        raise DiagnosticError("expected-iteration must be a non-negative integer")
    observed_value = checkpoint_data.get("iteration")
    if isinstance(observed_value, bool) or not isinstance(
        observed_value, (int, np.integer)
    ):
        raise DiagnosticError("checkpoint iteration is missing or not an integer")
    observed_iter = int(observed_value)
    if observed_iter < 0:
        raise DiagnosticError("checkpoint iteration must be non-negative")
    if observed_iter != expected_iter:
        raise DiagnosticError(
            f"checkpoint iteration mismatch: expected {expected_iter}, got {observed_iter}"
        )
    return {
        "expected_sha256": expected_hash,
        "observed_sha256": observed_hash,
        "expected_iteration": expected_iter,
        "observed_iteration": observed_iter,
        "passed": True,
    }


def validate_source_builder_expectations(
    observed_source_commit: str,
    expected_source_commit: str | None,
    observed_builder_sha256: str,
    expected_builder_sha256: str | None,
) -> dict[str, Any]:
    """Require exact source HEAD and builder-byte identities for the gate."""

    if expected_source_commit is None:
        raise DiagnosticError(
            "expected-source-commit is mandatory for the actual-checkpoint gate"
        )
    if expected_builder_sha256 is None:
        raise DiagnosticError(
            "expected-builder-sha256 is mandatory for the actual-checkpoint gate"
        )
    expected_source = normalize_git_commit(expected_source_commit)
    observed_source = normalize_git_commit(observed_source_commit)
    expected_builder = normalize_sha256(expected_builder_sha256)
    observed_builder = normalize_sha256(observed_builder_sha256)
    report = {
        "source": {
            "expected": expected_source,
            "observed": observed_source,
            "passed": observed_source == expected_source,
        },
        "builder": {
            "expected_sha256": expected_builder,
            "observed_sha256": observed_builder,
            "passed": observed_builder == expected_builder,
        },
        "passed": observed_source == expected_source
        and observed_builder == expected_builder,
    }
    if not report["source"]["passed"]:
        raise IdentityExpectationError(
            "source HEAD differs from expected-source-commit", report
        )
    if not report["builder"]["passed"]:
        raise IdentityExpectationError(
            "builder bytes differ from expected-builder-sha256", report
        )
    return report


def estimate_full_population_wall_seconds(
    selected_count: int,
    population_count: int,
    measured_subset_seconds: float,
    *,
    limit_seconds: float = PREDECLARED_FULL_ESTIMATE_LIMIT_SECONDS,
) -> dict[str, Any]:
    """Estimate sequential full-query wall time from measured subset throughput."""

    selected = int(selected_count)
    population = int(population_count)
    elapsed = float(measured_subset_seconds)
    limit = float(limit_seconds)
    if selected <= 0:
        raise ValueError("selected_count must be positive for a throughput estimate")
    if population < 0:
        raise ValueError("population_count must be non-negative")
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("measured_subset_seconds must be finite and positive")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("limit_seconds must be finite and positive")
    throughput = selected / elapsed
    estimated = population / throughput
    return {
        "measured_subset_query_count": selected,
        "measured_subset_seconds": elapsed,
        "measured_subset_throughput_queries_per_second": float(throughput),
        "full_population_query_count": population,
        "estimated_sequential_full_query_wall_seconds": float(estimated),
        "predeclared_limit_seconds": limit,
        "within_limit": bool(estimated <= limit),
    }


def aggregate_predeclared_subset_gates(
    validation: dict[str, Any],
) -> bool:
    """Aggregate only declared hard gates; field-difference magnitude is excluded."""

    missing = [key for key in PREDECLARED_HARD_GATE_KEYS if key not in validation]
    if missing:
        raise DiagnosticError(
            "predeclared gate validation is missing keys: " + ", ".join(missing)
        )
    invalid = [
        key
        for key in PREDECLARED_HARD_GATE_KEYS
        if not isinstance(validation[key], (bool, np.bool_))
    ]
    if invalid:
        raise DiagnosticError(
            "predeclared gate validation values must be boolean: "
            + ", ".join(invalid)
        )
    return bool(all(bool(validation[key]) for key in PREDECLARED_HARD_GATE_KEYS))


def _run_bounded_git_command(
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> str:
    if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
        raise ValueError("git command timeout must be finite and positive")
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(cwd),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DiagnosticError(
            f"source git identity command unavailable or timed out: git {' '.join(arguments)}"
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise DiagnosticError(
            f"source git identity command failed: git {' '.join(arguments)}"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout


def get_clean_source_git_identity(
    repository: str | os.PathLike[str] = PROJECT_ROOT,
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Return HEAD and porcelain status, refusing unavailable or dirty source."""

    root = Path(repository).expanduser().resolve()
    head = _run_bounded_git_command(
        ["rev-parse", "--verify", "HEAD"],
        cwd=root,
        timeout_seconds=timeout_seconds,
    ).strip()
    try:
        head = normalize_git_commit(head)
    except ValueError as error:
        raise DiagnosticError(
            f"source git HEAD is not a full commit identity: {head!r}"
        ) from error
    porcelain = _run_bounded_git_command(
        ["status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        timeout_seconds=timeout_seconds,
    )
    if porcelain:
        raise DiagnosticError(
            "source git worktree is dirty; clean HEAD/status are mandatory: "
            + porcelain.strip()
        )
    return {
        "repository": str(root),
        "head": head,
        "porcelain_status": porcelain,
        "clean": True,
    }


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    mixed = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (mixed ^ (mixed >> 31)) & UINT64_MASK


def stable_root_rank(root_id: int, seed: int) -> int:
    """Return the documented stable 64-bit rank for one root ID and seed."""

    root = _checked_u64(root_id, "root_id")
    selection_seed = _checked_u64(seed, "selection_seed")
    mixed_input = (
        selection_seed ^ ((root * 0xD6E8FEB86659FD93) & UINT64_MASK)
    ) & UINT64_MASK
    return _splitmix64(mixed_input)


def select_render_root_ids(
    population_count: int,
    requested_count: int,
    selection_seed: int,
) -> np.ndarray:
    """Select deterministic root IDs without replacement and return them sorted."""

    population = int(population_count)
    requested = int(requested_count)
    if population < 0:
        raise ValueError("render population count must be non-negative")
    if population == 0:
        raise ValueError("render population is empty")
    _checked_u64(selection_seed, "selection_seed")
    if requested < 0:
        raise ValueError("render-query-count must be positive, or explicitly zero")
    if requested == 0:
        requested = population
    if requested > population:
        raise ValueError(
            f"render-query-count {requested} exceeds render population {population}"
        )
    ranked = sorted(
        (stable_root_rank(root_id, selection_seed), root_id)
        for root_id in range(population)
    )
    selected = sorted(root_id for _, root_id in ranked[:requested])
    return np.asarray(selected, dtype=np.int64)


def _summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    if not bool(np.isfinite(array).all()):
        raise DiagnosticError("numeric summary received a non-finite value")
    quantiles = np.quantile(array, [0.50, 0.95], method="linear")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array, dtype=np.float64)),
        "std": float(np.std(array, dtype=np.float64)),
        "min": float(np.min(array)),
        "p50": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "max": float(np.max(array)),
    }


def array_identity(values: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(values))
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "bytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def validate_finite_points(points: Any, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise DiagnosticError(f"{name} must have shape [N, 3], got {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise DiagnosticError(f"{name} contains a non-finite coordinate")
    return np.ascontiguousarray(array)


def validate_unit_normals(normals: Any, name: str) -> dict[str, Any]:
    array = validate_finite_points(normals, name)
    norms = np.hypot(np.hypot(array[:, 0], array[:, 1]), array[:, 2])
    error = np.abs(norms - 1.0)
    invalid = ~np.isfinite(norms) | (error > NORMAL_NORM_TOLERANCE)
    if bool(invalid.any()):
        index = int(np.flatnonzero(invalid)[0])
        raise DiagnosticError(
            f"{name} row {index} is not unit length within "
            f"{NORMAL_NORM_TOLERANCE:g}; normals are not normalized silently"
        )
    return {
        "passed": True,
        "count": int(array.shape[0]),
        "norm_min": float(np.min(norms)) if norms.size else None,
        "norm_max": float(np.max(norms)) if norms.size else None,
        "max_abs_error": float(np.max(error)) if error.size else 0.0,
        "tolerance": NORMAL_NORM_TOLERANCE,
    }


def _path_file(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} does not exist: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def prepare_output_dir(path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
    """Create an output directory or reject an existing nonempty identity."""

    output_dir = Path(path).expanduser().resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output-dir is not a directory: {output_dir}")
        entries = list(output_dir.iterdir())
        if entries and not overwrite:
            raise FileExistsError(
                f"refusing nonempty output-dir without --overwrite: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _remove_existing_artifact(path: Path, *, overwrite: bool) -> None:
    exists = path.exists() or path.is_symlink()
    if not exists:
        return
    if not overwrite:
        raise FileExistsError(f"refusing existing result artifact: {path}")
    if path.is_dir() and not path.is_symlink():
        raise IsADirectoryError(f"result artifact is a directory: {path}")
    # Explicit --overwrite authorizes replacement of a named result artifact;
    # unlink never follows a symlink.
    path.unlink()


def _atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing existing result artifact: {path}")
    if not path.parent.exists():
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
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(f"refusing existing result artifact: {path}") from error
            temporary.unlink()
            temporary = None  # type: ignore[assignment]
    finally:
        if not handle.closed:
            handle.close()
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def write_deterministic_json(
    path: str | os.PathLike[str],
    payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Serialize JSON canonically and publish it atomically."""

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


def build_builder_argv(
    builder: str | os.PathLike[str],
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> list[str]:
    return [
        str(Path(builder).expanduser().resolve()),
        "--input",
        str(Path(input_path).expanduser().resolve()),
        "--output",
        str(Path(output_path).expanduser().resolve()),
    ]


def _drain_pipe(
    pipe: Any,
    destination: Path,
    max_bytes: int,
    result_slot: list[dict[str, Any]],
) -> None:
    written = 0
    observed = 0
    truncated = False
    drain_error: str | None = None
    try:
        with destination.open("wb") as handle:
            while True:
                block = pipe.read(64 * 1024)
                if not block:
                    break
                observed += len(block)
                if written < max_bytes:
                    allowed = block[: max_bytes - written]
                    handle.write(allowed)
                    written += len(allowed)
                truncated = observed > written
    except Exception as error:
        drain_error = repr(error)
    finally:
        try:
            pipe.close()
        except Exception as error:
            drain_error = drain_error or repr(error)
        result_slot.append(
            {
                "bytes_written": written,
                "observed_bytes": observed,
                "truncated": truncated,
                "error": drain_error,
            }
        )


def _popen_process_group_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "creationflags": int(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        }
    return {"start_new_session": True}


def _wait_for_process(process: Any, timeout_seconds: float) -> bool:
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


def _terminate_builder_process_group(process: Any) -> None:
    """Terminate a timed-out isolated group, then force-kill it if needed."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        # terminate() handles the direct child.  taskkill /T is a bounded,
        # shell-free tree cleanup for descendants of the isolated group.
        process.terminate()
        if _wait_for_process(process, BUILDER_TERMINATION_GRACE_SECONDS):
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                shell=False,
                check=False,
                capture_output=True,
                timeout=BUILDER_TERMINATION_GRACE_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
            _wait_for_process(process, BUILDER_TERMINATION_GRACE_SECONDS)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if _wait_for_process(process, BUILDER_TERMINATION_GRACE_SECONDS):
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _wait_for_process(process, BUILDER_TERMINATION_GRACE_SECONDS)
    if process.poll() is None:
        raise DiagnosticError(
            f"timed-out builder process {process.pid} could not be terminated"
        )


def _join_log_drainers(
    process: Any,
    threads: Sequence[threading.Thread],
) -> None:
    for thread in threads:
        thread.join(timeout=BUILDER_TERMINATION_GRACE_SECONDS)
    if any(thread.is_alive() for thread in threads):
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except Exception:
                pass
        for thread in threads:
            thread.join(timeout=BUILDER_TERMINATION_GRACE_SECONDS)
    if any(thread.is_alive() for thread in threads):
        raise DiagnosticError("builder log-drain thread did not join after process cleanup")


def run_builder(
    argv: Sequence[str],
    *,
    stdout_log: str | os.PathLike[str],
    stderr_log: str | os.PathLike[str],
    cwd: str | os.PathLike[str] | None = None,
    timeout_seconds: float = DEFAULT_BUILDER_TIMEOUT_SECONDS,
    max_log_bytes: int = MAX_BUILDER_LOG_BYTES,
) -> BuilderRun:
    """Run one builder argv with a finite timeout and bounded preserved logs."""

    command = tuple(str(argument) for argument in argv)
    if not command:
        raise ValueError("builder argv must not be empty")
    if not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    if max_log_bytes <= 0:
        raise ValueError("max_log_bytes must be positive")
    stdout_path = Path(stdout_log)
    stderr_path = Path(stderr_log)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        list(command),
        cwd=None if cwd is None else str(cwd),
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_popen_process_group_kwargs(),
    )
    stdout_result: list[dict[str, Any]] = []
    stderr_result: list[dict[str, Any]] = []
    stdout_thread = threading.Thread(
        target=_drain_pipe,
        args=(process.stdout, stdout_path, max_log_bytes, stdout_result),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_drain_pipe,
        args=(process.stderr, stderr_path, max_log_bytes, stderr_result),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    termination_error: Exception | None = None
    try:
        returncode = process.wait(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            _terminate_builder_process_group(process)
        except Exception as error:
            termination_error = error
            if process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass
                _wait_for_process(process, BUILDER_TERMINATION_GRACE_SECONDS)
        returncode = process.poll()
        if returncode is None:
            returncode = -1
    _join_log_drainers(process, (stdout_thread, stderr_thread))
    elapsed = time.perf_counter() - started
    if not stdout_result or not stderr_result:
        raise DiagnosticError("builder log drain did not complete")
    if stdout_result[0]["error"] or stderr_result[0]["error"]:
        raise DiagnosticError(
            "builder log drain failed: "
            f"stdout={stdout_result[0]['error']!r} "
            f"stderr={stderr_result[0]['error']!r}"
        )
    result = BuilderRun(
        argv=command,
        returncode=int(returncode),
        seconds=float(elapsed),
        timeout_seconds=float(timeout_seconds),
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        stdout_bytes_written=int(stdout_result[0]["bytes_written"]),
        stderr_bytes_written=int(stderr_result[0]["bytes_written"]),
        stdout_truncated=bool(stdout_result[0]["truncated"]),
        stderr_truncated=bool(stderr_result[0]["truncated"]),
    )
    if timed_out:
        detail = (
            f"; cleanup={termination_error}"
            if termination_error is not None
            else ""
        )
        raise BuilderTimeoutError(
            f"builder timed out after {float(timeout_seconds):g}s; "
            f"process={process.pid} returncode={result.returncode}{detail}",
            process,
            result,
        )
    return result


def _csr_row_sums(output: SurfaceNaturalNeighborOutput) -> np.ndarray:
    if output.query_count == 0:
        return np.empty((0,), dtype=np.float64)
    values = np.empty((output.query_count,), dtype=np.float64)
    for row_id in range(output.query_count):
        begin = int(output.row_offsets[row_id])
        end = int(output.row_offsets[row_id + 1])
        values[row_id] = float(np.sum(output.weights[begin:end], dtype=np.float64))
    return values


def csr_weighted_sum(
    output: SurfaceNaturalNeighborOutput,
    source_values: Any,
) -> np.ndarray:
    """Evaluate one validated CSR field by exact row-wise weighted gathers."""

    values = np.asarray(source_values, dtype=np.float64)
    if values.ndim not in (1, 2) or values.shape[0] != output.guide_count:
        raise ValueError(
            "source_values must have shape [guide_count] or [guide_count, C]"
        )
    if not bool(np.isfinite(values).all()):
        raise DiagnosticError("CSR source values contain a non-finite value")
    if values.ndim == 1:
        evaluated = np.empty((output.query_count,), dtype=np.float64)
    else:
        evaluated = np.empty((output.query_count, values.shape[1]), dtype=np.float64)
    for row_id in range(output.query_count):
        begin = int(output.row_offsets[row_id])
        end = int(output.row_offsets[row_id + 1])
        ids = output.guide_ids[begin:end]
        weights = output.weights[begin:end]
        if values.ndim == 1:
            evaluated[row_id] = float(np.dot(weights, values[ids]))
        else:
            evaluated[row_id] = weights @ values[ids]
    if not bool(np.isfinite(evaluated).all()):
        raise DiagnosticError("CSR weighted evaluation produced a non-finite value")
    return evaluated


def validate_guide_site_identity(
    output: SurfaceNaturalNeighborOutput,
    stored_guide_lengths: Any,
    *,
    tolerance: float = MACHINE_TOLERANCE,
) -> dict[str, Any]:
    """Require every guide-site output row to be one exact guide one-hot."""

    lengths = np.asarray(stored_guide_lengths, dtype=np.float64).reshape(-1)
    if lengths.shape != (output.guide_count,):
        raise ValueError("stored guide lengths must match output guide count")
    if output.query_count != output.guide_count:
        raise GuideIdentityError(
            "guide-site output query count does not equal guide count",
            {"passed": False, "reason": "query_count_mismatch"},
        )
    if not bool(np.isfinite(lengths).all()) or bool(np.any(lengths <= 0.0)):
        raise DiagnosticError("stored guide lengths must be finite and positive")
    barycentric_errors = np.asarray(
        output.barycentric_errors,
        dtype=np.float64,
    ).reshape(-1)
    if barycentric_errors.shape != (output.guide_count,):
        raise GuideIdentityError(
            "guide-site barycentric error count does not equal guide count",
            {
                "passed": False,
                "reason": "barycentric_error_count_mismatch",
                "count": int(barycentric_errors.size),
            },
        )
    invalid_barycentric = ~np.isfinite(barycentric_errors) | (
        barycentric_errors < 0.0
    ) | (barycentric_errors > tolerance)
    if bool(invalid_barycentric.any()):
        bad_rows = []
        for row_id in np.flatnonzero(invalid_barycentric).tolist():
            value = barycentric_errors[row_id]
            bad_rows.append(
                {
                    "query_id": int(row_id),
                    "error": float(value) if np.isfinite(value) else None,
                }
            )
        raise GuideIdentityError(
            "guide-site barycentric reconstruction error is nonfinite or "
            f"exceeds tolerance {tolerance:g}",
            {
                "passed": False,
                "reason": "barycentric_error_contract",
                "barycentric_error_tolerance": tolerance,
                "bad_rows": bad_rows,
            },
        )

    candidate_lengths = csr_weighted_sum(output, lengths)
    absolute_error = np.abs(candidate_lengths - lengths)
    relative_error = absolute_error / np.maximum(np.abs(lengths), np.finfo(np.float64).tiny)
    bad_rows: list[dict[str, Any]] = []
    for row_id in range(output.query_count):
        begin = int(output.row_offsets[row_id])
        end = int(output.row_offsets[row_id + 1])
        row_ids = output.guide_ids[begin:end]
        row_weights = output.weights[begin:end]
        if row_ids.size != 1:
            bad_rows.append(
                {"query_id": row_id, "reason": "not_one_nnz", "ids": row_ids.tolist()}
            )
            continue
        if int(row_ids[0]) != row_id:
            bad_rows.append(
                {
                    "query_id": row_id,
                    "reason": "wrong_guide_id",
                    "guide_id": int(row_ids[0]),
                }
            )
        if abs(float(row_weights[0]) - 1.0) > tolerance:
            bad_rows.append(
                {
                    "query_id": row_id,
                    "reason": "weight_not_one",
                    "weight": float(row_weights[0]),
                }
            )
    report = {
        "passed": not bad_rows and bool(np.all(absolute_error <= tolerance)),
        "row_count": output.query_count,
        "one_hot_tolerance": tolerance,
        "bad_rows": bad_rows,
        "stored_length": _summary(lengths),
        "candidate_length": _summary(candidate_lengths),
        "absolute_error": _summary(absolute_error),
        "relative_error": _summary(relative_error),
        "barycentric_error": _summary(barycentric_errors),
        "barycentric_error_tolerance": tolerance,
    }
    if not report["passed"]:
        raise GuideIdentityError("guide-site Kronecker identity failed", report)
    return report


def validate_topology_safe_candidate_support(
    candidate_support: Any,
    *,
    query_count: int,
    guide_count: int,
    fallback_query_count: int,
) -> dict[str, Any]:
    support = np.asarray(candidate_support, dtype=np.int64)
    if support.ndim != 2 or support.shape[0] != query_count:
        raise TopologyCandidateContainmentError(
            "topology-safe candidate support has the wrong query shape",
            {"passed": False, "reason": "shape", "shape": list(support.shape)},
        )
    if support.shape[1] <= 0:
        raise TopologyCandidateContainmentError(
            "topology-safe candidate support has zero width",
            {"passed": False, "reason": "zero_width"},
        )
    if bool((support < 0).any()) or bool((support >= guide_count).any()):
        raise TopologyCandidateContainmentError(
            "topology-safe candidate support contains an out-of-range guide ID",
            {"passed": False, "reason": "out_of_range"},
        )
    duplicate_rows = [
        row_id
        for row_id in range(query_count)
        if np.unique(support[row_id]).size != support.shape[1]
    ]
    report = {
        "passed": not duplicate_rows and int(fallback_query_count) == 0,
        "query_count": query_count,
        "support_width": int(support.shape[1]),
        "duplicate_or_padded_query_ids": duplicate_rows,
        "fallback_query_count": int(fallback_query_count),
    }
    if duplicate_rows:
        raise TopologyCandidateContainmentError(
            "topology-safe candidate support has duplicate/padded rows: "
            f"{duplicate_rows}",
            report,
        )
    if int(fallback_query_count) != 0:
        raise TopologyCandidateContainmentError(
            "topology-safe candidate support reports fallback coverage: "
            f"{fallback_query_count}",
            report,
        )
    return report


def audit_topology_safe_candidate_containment(
    output: SurfaceNaturalNeighborOutput,
    candidate_support: Any,
    *,
    support_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check every CGAL ID against the selected-query candidate support."""

    support = np.asarray(candidate_support, dtype=np.int64)
    if support.ndim != 2 or support.shape[0] != output.query_count:
        raise ValueError(
            "topology-safe candidate support query count does not match CSR output"
        )
    fallback_count = -1 if support_report is None else int(
        support_report.get("fallback_query_count", -1)
    )
    if support_report is not None:
        validate_topology_safe_candidate_support(
            support,
            query_count=output.query_count,
            guide_count=output.guide_count,
            fallback_query_count=fallback_count,
        )
    missing_by_query: list[dict[str, Any]] = []
    contained_count = 0
    returned_count = 0
    missing_union: set[int] = set()
    for query_id in range(output.query_count):
        begin = int(output.row_offsets[query_id])
        end = int(output.row_offsets[query_id + 1])
        returned = {int(value) for value in output.guide_ids[begin:end].tolist()}
        allowed = {int(value) for value in support[query_id].tolist()}
        missing = sorted(returned - allowed)
        returned_count += len(returned)
        contained_count += len(returned) - len(missing)
        if missing:
            missing_union.update(missing)
            missing_by_query.append(
                {"query_id": query_id, "guide_ids": missing}
            )
    report = {
        "passed": not missing_by_query,
        "returned_neighbor_count": returned_count,
        "contained_neighbor_count": contained_count,
        "contained_fraction": (
            float(contained_count / returned_count) if returned_count else 1.0
        ),
        "missing_ids": sorted(missing_union),
        "missing_by_query": missing_by_query,
    }
    if support_report is not None:
        report["support_validation"] = support_report
    if missing_by_query:
        raise TopologyCandidateContainmentError(
            "CGAL returned guide IDs outside topology-safe candidate support: "
            f"{report['missing_by_query']}",
            report,
        )
    return report


def _import_stage1_loader() -> tuple[Any, Any]:
    from tools.train_white_tiger_stage1 import (  # type: ignore[import-not-found]
        decode_positive_asinh_ratio,
        load_stage1_checkpoint_model,
    )

    return load_stage1_checkpoint_model, decode_positive_asinh_ratio


def extract_checkpoint_arrays(model: Any, *, expected_guide_count: int = EXPECTED_GUIDE_COUNT) -> CheckpointArrays:
    """Extract current local guide/render geometry without normalizing it here."""

    import torch

    if not bool(model.guide_enabled()):
        raise DiagnosticError("checkpoint has no primary guides")
    with torch.no_grad():
        guide_points = model.guide_points_local.detach().to(dtype=torch.float64).cpu().numpy()
        guide_normals_tensor, _, _ = model.guide_normals_and_tangent_frames()
        guide_normals = guide_normals_tensor.detach().to(dtype=torch.float64).cpu().numpy()
        guide_face_ids = model.guide_face_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        _, render_normals_tensor, render_points_local_tensor = model.roots_and_normals()
        render_points_local = (
            render_points_local_tensor.detach().to(dtype=torch.float64).cpu().numpy()
        )
        render_normals = render_normals_tensor.detach().to(dtype=torch.float64).cpu().numpy()
        render_face_ids = model.face_ids.detach().cpu().numpy().astype(np.int64, copy=False)

    guide_points = validate_finite_points(guide_points, "guide_points_local")
    render_points_local = validate_finite_points(render_points_local, "render_points_local")
    if guide_points.shape[0] != expected_guide_count:
        raise DiagnosticError(
            f"expected exactly {expected_guide_count} primary guides, got {guide_points.shape[0]}"
        )
    if guide_normals.shape != guide_points.shape:
        raise DiagnosticError("guide point and normal counts do not match")
    if render_normals.shape != render_points_local.shape:
        raise DiagnosticError("render point and normal counts do not match")
    if render_face_ids.shape != (render_points_local.shape[0],):
        raise DiagnosticError("render face IDs do not match render point count")
    guide_normal_report = validate_unit_normals(guide_normals, "guide_normals_local")
    render_normal_report = validate_unit_normals(render_normals, "render_normals_local")
    del guide_normal_report, render_normal_report
    if bool((guide_face_ids < 0).any()) or bool(
        (guide_face_ids >= int(model.faces.shape[0])).any()
    ):
        raise DiagnosticError("guide face IDs are out of range")
    if bool((render_face_ids < 0).any()) or bool(
        (render_face_ids >= int(model.faces.shape[0])).any()
    ):
        raise DiagnosticError("render face IDs are out of range")
    return CheckpointArrays(
        guide_points_local=guide_points,
        guide_normals_local=np.ascontiguousarray(guide_normals),
        guide_face_ids=np.ascontiguousarray(guide_face_ids),
        render_points_local=render_points_local,
        render_normals_local=np.ascontiguousarray(render_normals),
        render_face_ids=np.ascontiguousarray(render_face_ids),
        guide_lengths=np.empty((0,), dtype=np.float64),
    )


def _decode_guide_lengths(model: Any, decoder: Any) -> np.ndarray:
    import torch

    if model.guide_length_raw is None:
        raise DiagnosticError("checkpoint has no primary guide length field")
    with torch.no_grad():
        lengths = decoder(
            model.guide_length_raw,
            model.guide_length_reference,
        ).detach().reshape(-1).to(dtype=torch.float64).cpu().numpy()
    lengths = np.ascontiguousarray(lengths, dtype=np.float64)
    if not bool(np.isfinite(lengths).all()) or bool(np.any(lengths <= 0.0)):
        raise DiagnosticError("decoded guide lengths must be finite and positive")
    return lengths


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _verify_builder_output(
    path: Path,
    *,
    guide_count: int,
    query_count: int,
) -> SurfaceNaturalNeighborOutput:
    try:
        output = read_surface_natural_neighbor_output(path)
    except (OSError, ValueError, SurfaceNaturalNeighborFormatError) as error:
        raise DiagnosticError(f"invalid builder output {path}: {error}") from error
    if output.method != METHOD_IDENTITY:
        raise DiagnosticError(f"builder output method mismatch: {output.method!r}")
    if output.guide_count != guide_count or output.query_count != query_count:
        raise DiagnosticError(
            "builder output count mismatch: "
            f"got G={output.guide_count}, Q={output.query_count}; "
            f"expected G={guide_count}, Q={query_count}"
        )
    return output


def _output_memory_summary(device: Any) -> dict[str, Any]:
    import torch

    if device.type != "cuda":
        return {
            "device": str(device),
            "cuda": False,
            "allocated_bytes": 0,
            "max_allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_reserved_bytes": 0,
        }
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "cuda": True,
        "device_name": str(properties.name),
        "total_memory_bytes": int(properties.total_memory),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _legacy_selected_lengths(
    model: Any,
    points_local: np.ndarray,
    normals_local: np.ndarray,
    face_ids: np.ndarray,
    selected_root_ids: np.ndarray,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """Evaluate selected roots from the canonical cached full-render support."""

    import torch
    from anigroom.surface_interpolation import SurfaceSupport

    device = model.guide_points_local.device
    dtype = model.guide_points_local.dtype
    selected_ids = np.asarray(selected_root_ids, dtype=np.int64).reshape(-1)
    if selected_ids.shape[0] != points_local.shape[0]:
        raise DiagnosticError("selected root IDs do not match selected legacy queries")
    canonical_support = model.guide_interpolation_support()
    canonical_query_count = int(canonical_support.query_count)
    render_population_count = int(model.face_ids.shape[0])
    if canonical_query_count != render_population_count:
        raise DiagnosticError(
            "canonical cached render support count does not match render population: "
            f"{canonical_query_count} != {render_population_count}"
        )
    if bool((selected_ids < 0).any()) or bool(
        (selected_ids >= canonical_query_count).any()
    ):
        raise DiagnosticError("selected root IDs are outside canonical render support")
    selected_id_tensor = torch.as_tensor(
        selected_ids,
        device=device,
        dtype=torch.long,
    )
    selected_support = SurfaceSupport(
        indices=canonical_support.indices.index_select(0, selected_id_tensor).detach(),
        vertex_path_distances=canonical_support.vertex_path_distances.index_select(
            0,
            selected_id_tensor,
        ).detach(),
        report={
            **canonical_support.report,
            "provenance": "canonical_full_render_support_sliced_by_selected_root_ids",
            "canonical_query_count": canonical_query_count,
            "selected_query_count": int(selected_ids.shape[0]),
            "selected_root_ids_sha256": array_identity(selected_ids)["sha256"],
        },
    )
    with torch.no_grad():
        points = torch.as_tensor(points_local, device=device, dtype=dtype)
        normals = torch.as_tensor(normals_local, device=device, dtype=dtype)
        query_faces = torch.as_tensor(face_ids, device=device, dtype=torch.long)
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
            raise DiagnosticError("legacy model sampling returned no primary length")
        lengths = (
            controls["length"]
            .detach()
            .reshape(-1)
            .to(dtype=torch.float64)
            .cpu()
            .numpy()
        )
    lengths = np.ascontiguousarray(lengths, dtype=np.float64)
    if not bool(np.isfinite(lengths).all()) or bool(np.any(lengths <= 0.0)):
        raise DiagnosticError("legacy selected lengths must be finite and positive")
    return lengths, selected_support, dict(selected_support.report)


def _existing_topology_safe_candidate_support(
    model: Any,
    points_local: np.ndarray,
    face_ids: np.ndarray,
    topology_candidate_audit_k: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build fresh existing topology-safe candidate support for selected queries."""

    import torch
    from anigroom.surface_interpolation import SurfaceFieldInterpolator

    guide_count = int(model.guide_points_local.shape[0])
    width = min(int(topology_candidate_audit_k), guide_count)
    if width <= 0:
        raise ValueError("topology-safe candidate support width must be positive")
    device = model.guide_points_local.device
    dtype = model.guide_points_local.dtype
    interpolator = SurfaceFieldInterpolator(
        vertices=model.vertices,
        faces=model.faces,
        source_points=model.guide_points_local,
        source_face_ids=model.guide_face_ids,
        neighbor_count=width,
        device=device,
    )
    points = torch.as_tensor(points_local, device=device, dtype=dtype)
    query_faces = torch.as_tensor(face_ids, device=device, dtype=torch.long)
    support = interpolator.build_support(points, query_faces)
    support_ids = support.indices.detach().cpu().numpy().astype(np.int64, copy=False)
    support_report = dict(support.report)
    if support_ids.ndim != 2 or support_ids.shape[1] != width:
        raise TopologyCandidateContainmentError(
            "topology-safe candidate support width mismatch",
            {
                "passed": False,
                "requested_width": width,
                "actual_width": int(support_ids.shape[1])
                if support_ids.ndim == 2
                else None,
            },
        )
    support_report["requested_width"] = width
    support_report["actual_width"] = int(support_ids.shape[1])
    support_report["label"] = (
        f"existing_topology_safe_K{width}_candidate_support"
    )
    support_report["certificate"] = (
        "containment_only_within_existing_topology_safe_candidate_support"
    )
    validate_topology_safe_candidate_support(
        support_ids,
        query_count=int(points_local.shape[0]),
        guide_count=guide_count,
        fallback_query_count=int(support_report.get("fallback_query_count", -1)),
    )
    return support_ids, support_report


def _difference_report(candidate: np.ndarray, legacy: np.ndarray) -> dict[str, Any]:
    if candidate.shape != legacy.shape:
        raise DiagnosticError("candidate and legacy length arrays have different shapes")
    absolute = np.abs(candidate - legacy)
    relative = absolute / np.maximum(np.abs(legacy), np.finfo(np.float64).tiny)
    return {
        "absolute": _summary(absolute),
        "relative": _summary(relative),
    }


def _selection_arguments(
    render_query_count: int | None,
    selection_seed: int,
    topology_candidate_audit_k: int,
) -> tuple[int, int, int]:
    requested = DEFAULT_RENDER_QUERY_COUNT if render_query_count is None else int(render_query_count)
    if requested < 0:
        raise ValueError("render-query-count must be positive, or explicitly zero")
    seed = _checked_u64(selection_seed, "selection-seed")
    audit_k = int(topology_candidate_audit_k)
    if audit_k <= 0:
        raise ValueError("topology-candidate-audit-k must be positive")
    return requested, seed, audit_k


def run_checkpoint_subset_diagnostic(
    *,
    checkpoint: str | os.PathLike[str],
    builder: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    device: str = "cuda",
    mesh: str | os.PathLike[str] | None = None,
    render_query_count: int | None = None,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    topology_candidate_audit_k: int = DEFAULT_TOPOLOGY_CANDIDATE_AUDIT_K,
    builder_timeout_seconds: float = DEFAULT_BUILDER_TIMEOUT_SECONDS,
    expected_checkpoint_sha256: str | None = None,
    expected_iteration: int | None = None,
    expected_source_commit: str | None = None,
    expected_builder_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the bounded actual-checkpoint Phase-A diagnostic."""

    requested_count, seed, audit_k = _selection_arguments(
        render_query_count,
        selection_seed,
        topology_candidate_audit_k,
    )
    if not math.isfinite(float(builder_timeout_seconds)) or float(builder_timeout_seconds) <= 0.0:
        raise ValueError("builder-timeout-seconds must be finite and positive")
    if expected_checkpoint_sha256 is None:
        raise DiagnosticError(
            "expected-checkpoint-sha256 is mandatory for the actual-checkpoint gate"
        )
    if expected_iteration is None:
        raise DiagnosticError(
            "expected-iteration is mandatory for the actual-checkpoint gate"
        )
    if expected_source_commit is None:
        raise DiagnosticError(
            "expected-source-commit is mandatory for the actual-checkpoint gate"
        )
    if expected_builder_sha256 is None:
        raise DiagnosticError(
            "expected-builder-sha256 is mandatory for the actual-checkpoint gate"
        )
    checkpoint_path = _path_file(Path(checkpoint), "checkpoint")
    builder_path = _path_file(Path(builder), "builder")
    output_root = prepare_output_dir(output_dir, overwrite=overwrite)
    expected_hash = (
        normalize_sha256(expected_checkpoint_sha256)
        if expected_checkpoint_sha256 is not None
        else None
    )
    source_git_started = time.perf_counter()
    source_git_identity = get_clean_source_git_identity(PROJECT_ROOT)
    source_git_seconds = time.perf_counter() - source_git_started
    checkpoint_started = time.perf_counter()
    checkpoint_digest = sha256_file(checkpoint_path)
    checkpoint_hash_seconds = time.perf_counter() - checkpoint_started
    if checkpoint_digest != expected_hash:
        raise DiagnosticError(
            f"checkpoint SHA256 mismatch: expected {expected_hash}, got {checkpoint_digest}"
        )
    # The loader below is the repository's exact checkpoint reconstruction;
    # the embedded iteration is checked immediately after that exact load.
    builder_digest = sha256_file(builder_path)
    source_builder_expectations = validate_source_builder_expectations(
        source_git_identity["head"],
        expected_source_commit,
        builder_digest,
        expected_builder_sha256,
    )
    source_path = Path(__file__).resolve()
    io_path = (PROJECT_ROOT / "tools" / "surface_natural_neighbor_io.py").resolve()
    mesh_path = None if mesh is None else _path_file(Path(mesh), "mesh")

    import torch

    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise DiagnosticError("requested CUDA device is unavailable; no CPU fallback is used")
    if device_object.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_object)

    timings: dict[str, float] = {
        "source_git_identity_seconds": float(source_git_seconds),
        "checkpoint_sha256_seconds": float(checkpoint_hash_seconds),
    }
    load_started = time.perf_counter()
    loader, decoder = _import_stage1_loader()
    model, config, checkpoint_data = loader(
        checkpoint_path,
        device_object,
        mesh_path_override=mesh_path,
    )
    timings["checkpoint_model_load_seconds"] = float(time.perf_counter() - load_started)
    checkpoint_expectations = validate_checkpoint_expectations(
        checkpoint_digest,
        expected_checkpoint_sha256,
        checkpoint_data,
        expected_iteration,
    )

    extract_started = time.perf_counter()
    arrays = extract_checkpoint_arrays(model)
    guide_lengths = _decode_guide_lengths(model, decoder)
    arrays = CheckpointArrays(
        guide_points_local=arrays.guide_points_local,
        guide_normals_local=arrays.guide_normals_local,
        guide_face_ids=arrays.guide_face_ids,
        render_points_local=arrays.render_points_local,
        render_normals_local=arrays.render_normals_local,
        render_face_ids=arrays.render_face_ids,
        guide_lengths=guide_lengths,
    )
    timings["model_array_extract_seconds"] = float(time.perf_counter() - extract_started)

    guide_input_path = output_root / "guide_sites.input.bin"
    guide_output_path = output_root / "guide_sites.output.bin"
    guide_stdout_path = output_root / "guide_sites.stdout.log"
    guide_stderr_path = output_root / "guide_sites.stderr.log"
    render_input_path = output_root / "render_subset.input.bin"
    render_output_path = output_root / "render_subset.output.bin"
    render_stdout_path = output_root / "render_subset.stdout.log"
    render_stderr_path = output_root / "render_subset.stderr.log"
    report_path = output_root / "report.json"
    for artifact_name in ARTIFACT_NAMES:
        _remove_existing_artifact(output_root / artifact_name, overwrite=overwrite)

    guide_input_started = time.perf_counter()
    write_surface_natural_neighbor_input(
        guide_input_path,
        arrays.guide_points_local,
        arrays.guide_points_local,
        arrays.guide_normals_local,
        overwrite=False,
    )
    timings["guide_input_write_seconds"] = float(time.perf_counter() - guide_input_started)

    guide_run = run_builder(
        build_builder_argv(builder_path, guide_input_path, guide_output_path),
        stdout_log=guide_stdout_path,
        stderr_log=guide_stderr_path,
        cwd=PROJECT_ROOT,
        timeout_seconds=float(builder_timeout_seconds),
    )
    timings["guide_builder_seconds"] = guide_run.seconds
    if guide_run.returncode != 0:
        raise DiagnosticError(
            f"guide-site builder exited {guide_run.returncode}; "
            f"logs={guide_stdout_path},{guide_stderr_path}"
        )
    guide_output_started = time.perf_counter()
    guide_output = _verify_builder_output(
        guide_output_path,
        guide_count=arrays.guide_points_local.shape[0],
        query_count=arrays.guide_points_local.shape[0],
    )
    guide_identity = validate_guide_site_identity(guide_output, arrays.guide_lengths)
    timings["guide_output_read_validate_seconds"] = float(
        time.perf_counter() - guide_output_started
    )

    selection_started = time.perf_counter()
    selected_ids = select_render_root_ids(
        arrays.render_points_local.shape[0],
        requested_count,
        seed,
    )
    timings["render_root_selection_seconds"] = float(time.perf_counter() - selection_started)
    selected_points = arrays.render_points_local[selected_ids]
    selected_normals = arrays.render_normals_local[selected_ids]
    selected_face_ids = arrays.render_face_ids[selected_ids]

    render_input_started = time.perf_counter()
    write_surface_natural_neighbor_input(
        render_input_path,
        arrays.guide_points_local,
        selected_points,
        selected_normals,
        overwrite=False,
    )
    timings["render_input_write_seconds"] = float(time.perf_counter() - render_input_started)
    render_run = run_builder(
        build_builder_argv(builder_path, render_input_path, render_output_path),
        stdout_log=render_stdout_path,
        stderr_log=render_stderr_path,
        cwd=PROJECT_ROOT,
        timeout_seconds=float(builder_timeout_seconds),
    )
    timings["render_builder_seconds"] = render_run.seconds
    if render_run.returncode != 0:
        raise DiagnosticError(
            f"render-subset builder exited {render_run.returncode}; "
            f"logs={render_stdout_path},{render_stderr_path}"
        )
    render_output_started = time.perf_counter()
    render_output = _verify_builder_output(
        render_output_path,
        guide_count=arrays.guide_points_local.shape[0],
        query_count=selected_points.shape[0],
    )
    timings["render_output_read_validate_seconds"] = float(
        time.perf_counter() - render_output_started
    )

    candidate_started = time.perf_counter()
    candidate_lengths = csr_weighted_sum(render_output, arrays.guide_lengths)
    if not bool(np.isfinite(candidate_lengths).all()) or bool(np.any(candidate_lengths <= 0.0)):
        raise DiagnosticError("candidate CSR lengths must be finite and positive")
    timings["candidate_csr_length_seconds"] = float(time.perf_counter() - candidate_started)

    candidate_support_started = time.perf_counter()
    candidate_support_ids, candidate_support_report = (
        _existing_topology_safe_candidate_support(
            model,
            selected_points,
            selected_face_ids,
            audit_k,
        )
    )
    candidate_support_validation = validate_topology_safe_candidate_support(
        candidate_support_ids,
        query_count=selected_points.shape[0],
        guide_count=arrays.guide_points_local.shape[0],
        fallback_query_count=int(
            candidate_support_report.get("fallback_query_count", -1)
        ),
    )
    candidate_containment = audit_topology_safe_candidate_containment(
        render_output,
        candidate_support_ids,
        support_report=candidate_support_report,
    )
    timings["topology_safe_candidate_support_and_containment_seconds"] = float(
        time.perf_counter() - candidate_support_started
    )

    legacy_started = time.perf_counter()
    legacy_lengths, legacy_support, legacy_support_report = _legacy_selected_lengths(
        model,
        selected_points,
        selected_normals,
        selected_face_ids,
        selected_ids,
    )
    timings["legacy_selected_length_seconds"] = float(time.perf_counter() - legacy_started)
    difference = _difference_report(candidate_lengths, legacy_lengths)

    row_sums = _csr_row_sums(render_output)
    effective_neighbor_count = np.empty((render_output.query_count,), dtype=np.float64)
    max_weight = np.empty((render_output.query_count,), dtype=np.float64)
    neighbor_count = (
        render_output.row_offsets[1:] - render_output.row_offsets[:-1]
    )
    for row_id in range(render_output.query_count):
        begin = int(render_output.row_offsets[row_id])
        end = int(render_output.row_offsets[row_id + 1])
        weights = render_output.weights[begin:end]
        effective_neighbor_count[row_id] = 1.0 / float(np.sum(weights * weights))
        max_weight[row_id] = float(np.max(weights))

    guide_normal_validation = validate_unit_normals(
        arrays.guide_normals_local,
        "guide_normals_local",
    )
    render_normal_validation = validate_unit_normals(
        arrays.render_normals_local,
        "render_normals_local",
    )
    row_sum_error = np.abs(row_sums - 1.0)
    full_population_estimate = estimate_full_population_wall_seconds(
        int(selected_ids.size),
        int(arrays.render_points_local.shape[0]),
        render_run.seconds,
    )
    validation: dict[str, Any] = {
        "source_identity": bool(source_builder_expectations["source"]["passed"]),
        "checkpoint_identity": bool(checkpoint_expectations["passed"]),
        "builder_identity": bool(source_builder_expectations["builder"]["passed"]),
        "builder_invocations_within_300_seconds": bool(
            guide_run.seconds <= DEFAULT_BUILDER_TIMEOUT_SECONDS
            and render_run.seconds <= DEFAULT_BUILDER_TIMEOUT_SECONDS
        ),
        "guide_count_exact_4500": int(arrays.guide_points_local.shape[0]) == EXPECTED_GUIDE_COUNT,
        "guide_site_one_hot_identity": bool(guide_identity["passed"]),
        "guide_site_length_identity": bool(guide_identity["passed"]),
        "guide_normals_finite_unit": bool(guide_normal_validation["passed"]),
        "render_normals_finite_unit": bool(render_normal_validation["passed"]),
        "render_csr_rows_positive_normalized": bool(
            np.all(candidate_lengths > 0.0)
            and np.all(row_sum_error <= ROW_SUM_TOLERANCE)
        ),
        "topology_safe_candidate_support_containment": bool(
            candidate_containment["passed"]
        ),
        "topology_safe_candidate_support_no_fallback": bool(
            candidate_support_validation["fallback_query_count"] == 0
        ),
        "topology_safe_candidate_support_no_duplicate_or_padding": not bool(
            candidate_support_validation["duplicate_or_padded_query_ids"]
        ),
        "full_estimate_within_3600_seconds": bool(
            full_population_estimate["within_limit"]
        ),
    }
    all_predeclared_subset_gates_passed = aggregate_predeclared_subset_gates(
        validation
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "diagnostic_only": True,
        "method": METHOD_IDENTITY,
        "arguments": {
            "checkpoint": str(checkpoint_path),
            "builder": str(builder_path),
            "output_dir": str(output_root),
            "device": str(device_object),
            "mesh": None if mesh_path is None else str(mesh_path),
            "render_query_count_requested": requested_count,
            "selection_seed": seed,
            "topology_candidate_audit_k_requested": audit_k,
            "builder_timeout_seconds": float(builder_timeout_seconds),
            "overwrite": bool(overwrite),
            "expected_checkpoint_sha256": expected_hash,
            "expected_iteration": int(expected_iteration),
            "expected_source_commit": normalize_git_commit(expected_source_commit),
            "expected_builder_sha256": normalize_sha256(expected_builder_sha256),
        },
        "source_identity": {
            "diagnostic_source": str(source_path),
            "diagnostic_source_sha256": sha256_file(source_path),
            "io_module": str(io_path),
            "io_module_sha256": sha256_file(io_path),
            "git": source_git_identity,
            "expectation": source_builder_expectations["source"],
            "passed": bool(source_builder_expectations["source"]["passed"]),
        },
        "checkpoint_identity": {
            "path": str(checkpoint_path),
            "bytes": int(checkpoint_path.stat().st_size),
            "sha256": checkpoint_digest,
            "iteration": int(checkpoint_data.get("iteration", 0)),
            "expectations": checkpoint_expectations,
            "passed": bool(checkpoint_expectations["passed"]),
            "config_mesh_path": str(getattr(config, "mesh_path", "")),
        },
        "builder_identity": {
            "path": str(builder_path),
            "bytes": int(builder_path.stat().st_size),
            "sha256": builder_digest,
            "expectation": source_builder_expectations["builder"],
            "passed": bool(source_builder_expectations["builder"]["passed"]),
        },
        "model_arrays": {
            "guide_points_local": array_identity(arrays.guide_points_local),
            "guide_normals_local": array_identity(arrays.guide_normals_local),
            "guide_face_ids": array_identity(arrays.guide_face_ids),
            "render_points_local": array_identity(arrays.render_points_local),
            "render_normals_local": array_identity(arrays.render_normals_local),
            "render_face_ids": array_identity(arrays.render_face_ids),
            "guide_lengths": array_identity(arrays.guide_lengths),
        },
        "guide_sites": {
            "guide_count": int(guide_output.guide_count),
            "query_count": int(guide_output.query_count),
            "nnz": int(guide_output.nnz),
            "input": _artifact(guide_input_path),
            "output": _artifact(guide_output_path),
            "builder_run": guide_run.as_dict(),
            "identity": guide_identity,
            "self_error": {
                "absolute": guide_identity["absolute_error"],
                "relative": guide_identity["relative_error"],
            },
        },
        "render_subset": {
            "render_population_count": int(arrays.render_points_local.shape[0]),
            "selected_count": int(selected_ids.size),
            "selected_root_ids": selected_ids.astype(np.int64).tolist(),
            "input": _artifact(render_input_path),
            "output": _artifact(render_output_path),
            "builder_run": render_run.as_dict(),
            "candidate_length": {
                "positive": True,
                "summary": _summary(candidate_lengths),
            },
            "legacy_primary_length": {"summary": _summary(legacy_lengths)},
            "candidate_vs_legacy": difference,
            "natural_neighbor_count": _summary(neighbor_count),
            "maximum_weight": _summary(max_weight),
            "effective_neighbor_count": _summary(effective_neighbor_count),
            "barycentric_error": _summary(render_output.barycentric_errors),
            "row_sum": {
                "summary": _summary(row_sums),
                "max_abs_error": float(np.max(row_sum_error)) if row_sum_error.size else 0.0,
                "tolerance": ROW_SUM_TOLERANCE,
                "all_within_tolerance": bool(np.all(row_sum_error <= ROW_SUM_TOLERANCE)),
            },
            "legacy_support": legacy_support_report,
        },
        "topology_safe_candidate_support_audit": {
            "label": (
                f"existing_topology_safe_K{int(candidate_support_ids.shape[1])}"
                "_candidate_support"
            ),
            "certificate": (
                "containment_only_within_existing_topology_safe_candidate_support"
            ),
            "support_width": int(candidate_support_ids.shape[1]),
            "support": candidate_support_report,
            "validation": candidate_support_validation,
            "containment": candidate_containment,
        },
        "normal_validation": {
            "guide": guide_normal_validation,
            "render_population": render_normal_validation,
        },
        "validation": validation,
        "all_predeclared_subset_gates_passed": all_predeclared_subset_gates_passed,
        "timings_seconds": timings,
        "cuda_memory": _output_memory_summary(device_object),
        "full_population_estimate": {
            **full_population_estimate,
            "subset_performance_gate_evidence": True,
            "full_population_measurement": False,
        },
        "artifacts": {
            "guide_stdout_log": _artifact(guide_stdout_path),
            "guide_stderr_log": _artifact(guide_stderr_path),
            "render_stdout_log": _artifact(render_stdout_path),
            "render_stderr_log": _artifact(render_stderr_path),
            "report": str(report_path),
        },
    }
    write_deterministic_json(report_path, report, overwrite=False)
    print(
        "R083_STATUS=complete "
        f"guides={guide_output.guide_count} "
        f"render_population={arrays.render_points_local.shape[0]} "
        f"render_selected={render_output.query_count} "
        f"guide_nnz={guide_output.nnz} render_nnz={render_output.nnz}",
        flush=True,
    )
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bounded diagnostic for the R083 actual-checkpoint length field."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--builder", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument(
        "--render-query-count",
        type=int,
        default=DEFAULT_RENDER_QUERY_COUNT,
        help="positive subset count; explicitly pass 0 for the complete render population",
    )
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument(
        "--topology-candidate-audit-k",
        type=int,
        default=DEFAULT_TOPOLOGY_CANDIDATE_AUDIT_K,
    )
    parser.add_argument(
        "--builder-timeout-seconds",
        type=float,
        default=DEFAULT_BUILDER_TIMEOUT_SECONDS,
    )
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-iteration", required=True, type=int)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-builder-sha256", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        run_checkpoint_subset_diagnostic(
            checkpoint=args.checkpoint,
            builder=args.builder,
            output_dir=args.output_dir,
            device=args.device,
            mesh=args.mesh,
            render_query_count=args.render_query_count,
            selection_seed=args.selection_seed,
            topology_candidate_audit_k=args.topology_candidate_audit_k,
            builder_timeout_seconds=args.builder_timeout_seconds,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_iteration=args.expected_iteration,
            expected_source_commit=args.expected_source_commit,
            expected_builder_sha256=args.expected_builder_sha256,
            overwrite=bool(args.overwrite),
        )
    except Exception as error:
        print(f"R083_ERROR={error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

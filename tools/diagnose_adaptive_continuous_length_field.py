"""Bounded R081 fixed-checkpoint continuous length-field diagnostic.

This module intentionally stays outside the formal training path.  It loads one
strict Stage1 checkpoint, evaluates the inherited primary-guide field and an
independent K+1-support Wendland candidate, and writes only numeric evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import decode_positive_asinh_ratio  # noqa: E402
from anigroom.surface_interpolation import (  # noqa: E402
    SurfaceFieldInterpolator,
    SurfaceSupport,
    adaptive_wendland_c2_weights,
    interpolate_physical,
)


SUMMARY_QUANTILES: tuple[tuple[str, float], ...] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q25", 0.25),
    ("q50", 0.50),
    ("q75", 0.75),
    ("q90", 0.90),
    ("q95", 0.95),
    ("q99", 0.99),
    ("q999", 0.999),
)
SUMMARY_VALUE_KEYS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    *(name for name, _ in SUMMARY_QUANTILES),
    "max",
)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _as_numpy_flat(values: torch.Tensor | np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    return np.asarray(array, dtype=np.float64).reshape(-1)


def summarize(values: torch.Tensor | np.ndarray | list[float] | tuple[float, ...]) -> dict[str, Any]:
    """Return deterministic finite-value distribution statistics."""

    flat = _as_numpy_flat(values)
    if flat.size == 0:
        return {
            "count": 0,
            **{key: None for key in SUMMARY_VALUE_KEYS},
        }
    if not bool(np.isfinite(flat).all()):
        raise ValueError("summary input contains non-finite values")
    result: dict[str, Any] = {
        "count": int(flat.size),
        "mean": float(np.mean(flat, dtype=np.float64)),
        "std": float(np.std(flat, dtype=np.float64, ddof=0)),
        "min": float(np.min(flat)),
    }
    quantile_values = np.quantile(
        flat,
        [probability for _, probability in SUMMARY_QUANTILES],
        method="linear",
    )
    result.update(
        {
            name: float(value)
            for (name, _), value in zip(SUMMARY_QUANTILES, quantile_values, strict=True)
        }
    )
    result["max"] = float(np.max(flat))
    return result


def _mean_p95(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "p95": None}
    return {
        "mean": float(np.mean(values, dtype=np.float64)),
        "p95": float(np.quantile(values, 0.95, method="linear")),
    }


def _ratio_summary(
    numerator: Mapping[str, Any],
    denominator: Mapping[str, Any],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in SUMMARY_VALUE_KEYS:
        top = numerator.get(key)
        bottom = denominator.get(key)
        if top is None or bottom is None or float(bottom) == 0.0:
            result[key] = None
        else:
            result[key] = float(top) / float(bottom)
    return result


def _as_long_tensor(value: torch.Tensor | np.ndarray, *, name: str) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if tensor.dtype not in {
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError(f"{name} must have an integer dtype")
    return tensor.to(dtype=torch.long)


def validate_support_ids(
    support_indices: torch.Tensor | np.ndarray,
    *,
    source_count: int,
    expected_width: int | None = None,
    name: str = "support",
) -> dict[str, Any]:
    """Validate exact, non-padded support IDs and return a small report."""

    support = _as_long_tensor(support_indices, name=name)
    if support.ndim != 2:
        raise ValueError(f"{name} must have shape [Q, S]")
    if expected_width is not None and int(support.shape[1]) != int(expected_width):
        raise RuntimeError(
            f"{name} width mismatch: {int(support.shape[1])} != {int(expected_width)}"
        )
    source_count = int(source_count)
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if support.numel() > 0:
        if bool((support < 0).any()) or bool((support >= source_count).any()):
            raise RuntimeError(f"{name} contains an out-of-range ID or coverage hole")
        sorted_support = torch.sort(support, dim=1).values
        duplicate_rows = (
            sorted_support[:, 1:] == sorted_support[:, :-1]
        ).any(dim=1) if int(support.shape[1]) > 1 else torch.zeros(
            (int(support.shape[0]),), device=support.device, dtype=torch.bool
        )
        duplicate_count = int(duplicate_rows.sum().detach().cpu())
        if duplicate_count:
            raise RuntimeError(
                f"{name} contains duplicate padded support IDs in {duplicate_count} rows"
            )
        unique_count = int(torch.unique(support).numel())
    else:
        duplicate_count = 0
        unique_count = 0
    return {
        "query_count": int(support.shape[0]),
        "support_width": int(support.shape[1]),
        "source_count": source_count,
        "unique_support_ids": True,
        "unique_support_id_count": unique_count,
        "duplicate_row_count": duplicate_count,
        "ids_in_range": True,
    }


def validate_surface_support(
    support: SurfaceSupport,
    *,
    source_count: int,
    expected_width: int,
    name: str,
) -> dict[str, Any]:
    """Reject padded, uncovered, or fallback support returned by the interpolator."""

    report = validate_support_ids(
        support.indices,
        source_count=source_count,
        expected_width=expected_width,
        name=name,
    )
    expected_paths = (
        int(support.indices.shape[0]),
        int(support.indices.shape[1]),
        3,
    )
    if tuple(support.vertex_path_distances.shape) != expected_paths:
        raise RuntimeError(
            f"{name} vertex-path shape mismatch: "
            f"{tuple(support.vertex_path_distances.shape)} != {expected_paths}"
        )
    paths = support.vertex_path_distances
    finite_paths = torch.isfinite(paths)
    if bool(torch.isnan(paths).any()) or bool(torch.isneginf(paths).any()):
        raise RuntimeError(f"{name} contains NaN or -inf intrinsic path entries")
    if bool((finite_paths & (paths < 0.0)).any()):
        raise RuntimeError(f"{name} contains negative finite intrinsic path entries")
    covered_slots = finite_paths.any(dim=-1)
    if not bool(covered_slots.all()):
        raise RuntimeError(f"{name} contains a coverage hole with all-three +inf path entries")
    path_entry_count = int(paths.numel())
    finite_path_entry_count = int(finite_paths.sum().detach().cpu())
    support_slot_count = int(covered_slots.numel())
    fully_covered_support_slot_count = int(covered_slots.sum().detach().cpu())
    fallback_count = int(support.report.get("fallback_query_count", 0))
    if fallback_count != 0:
        raise RuntimeError(
            f"{name} used fallback support for {fallback_count} queries; "
            "the R081 diagnostic requires exact support"
        )
    report.update(
        {
            "path_entry_count": path_entry_count,
            "finite_path_entry_count": finite_path_entry_count,
            "finite_path_entry_fraction": float(finite_path_entry_count / path_entry_count)
            if path_entry_count
            else 0.0,
            "finite_path_entries_nonnegative": True,
            "support_slot_count": support_slot_count,
            "fully_covered_support_slot_count": fully_covered_support_slot_count,
            "fully_covered_support_slot_fraction": float(
                fully_covered_support_slot_count / support_slot_count
            )
            if support_slot_count
            else 0.0,
            "fallback_query_count": fallback_count,
            "support_bytes": int(
                support.indices.numel() * support.indices.element_size()
                + paths.numel() * paths.element_size()
            ),
        }
    )
    return report


def validate_interpolation_invariants(
    source_values: torch.Tensor | np.ndarray,
    support_indices: torch.Tensor | np.ndarray,
    weights: torch.Tensor,
    *,
    interpolated: torch.Tensor | None = None,
    name: str = "candidate",
    expected_width: int | None = None,
    require_positive: bool = True,
    constant_value: float = 1.234567,
) -> dict[str, Any]:
    """Validate finite normalized interpolation, constants, positivity, and hull."""

    if not isinstance(weights, torch.Tensor) or weights.ndim != 2:
        raise ValueError("weights must be a rank-2 tensor")
    support = _as_long_tensor(support_indices, name=f"{name} support")
    if tuple(support.shape) != tuple(weights.shape):
        raise ValueError(f"{name} support and weights must have equal shape")
    source = (
        source_values
        if isinstance(source_values, torch.Tensor)
        else torch.as_tensor(source_values)
    )
    source = source.to(device=weights.device, dtype=weights.dtype).reshape(-1)
    support_report = validate_support_ids(
        support,
        source_count=int(source.shape[0]),
        expected_width=expected_width,
        name=f"{name} support",
    )
    if not bool(torch.isfinite(source).all()):
        raise RuntimeError(f"{name} source values are non-finite")
    if require_positive and bool((source <= 0.0).any()):
        raise RuntimeError(f"{name} source values must be finite and positive")
    finite_weights = bool(torch.isfinite(weights).all())
    nonnegative_weights = bool((weights >= 0.0).all())
    if not finite_weights or not nonnegative_weights:
        raise RuntimeError(f"{name} weights must be finite and nonnegative")
    row_sums = weights.sum(dim=1)
    row_sum_error = (row_sums - 1.0).abs()
    row_sums_allclose = bool(
        torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    )
    if not row_sums_allclose:
        raise RuntimeError(f"{name} weights are not row-normalized")

    if interpolated is None:
        field = interpolate_physical(source, support, weights)
    else:
        field = interpolated.to(device=weights.device, dtype=weights.dtype).reshape(-1)
    if int(field.shape[0]) != int(weights.shape[0]):
        raise ValueError(f"{name} interpolated field has the wrong query count")
    finite_values = bool(torch.isfinite(field).all())
    if not finite_values:
        raise RuntimeError(f"{name} interpolated field contains non-finite values")

    constant_source = torch.full_like(source, float(constant_value))
    constant_field = interpolate_physical(constant_source, support, weights)
    constant_error = (constant_field - float(constant_value)).abs()
    constant_allclose = bool(
        torch.allclose(
            constant_field,
            torch.full_like(constant_field, float(constant_value)),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    )
    if not constant_allclose:
        raise RuntimeError(f"{name} does not reproduce a constant field")

    source_min = source.min() if source.numel() else source.new_tensor(0.0)
    source_max = source.max() if source.numel() else source.new_tensor(0.0)
    scale = max(1.0, abs(float(source_min.detach().cpu())), abs(float(source_max.detach().cpu())))
    hull_tolerance = 1.0e-5 * scale
    in_convex_hull = bool(
        ((field >= source_min - hull_tolerance) & (field <= source_max + hull_tolerance)).all()
    )
    if not in_convex_hull:
        raise RuntimeError(f"{name} field leaves the source-value convex hull")
    positive_values = bool((field > 0.0).all()) if require_positive else True
    if require_positive and not positive_values:
        raise RuntimeError(f"{name} field is not positive")

    return {
        "finite_values": finite_values,
        "weights_finite": finite_weights,
        "weights_nonnegative": nonnegative_weights,
        "row_sums_allclose": row_sums_allclose,
        "row_sum_max_abs_error": float(row_sum_error.max().detach().cpu())
        if row_sum_error.numel()
        else 0.0,
        "unique_support_ids": support_report["unique_support_ids"],
        "ids_in_range": support_report["ids_in_range"],
        "constant_field_reproduction": {
            "ok": constant_allclose,
            "constant_value": float(constant_value),
            "max_abs_error": float(constant_error.max().detach().cpu())
            if constant_error.numel()
            else 0.0,
        },
        "convex_hull": {
            "ok": in_convex_hull,
            "source_min": float(source_min.detach().cpu()),
            "source_max": float(source_max.detach().cpu()),
            "field_min": float(field.min().detach().cpu()) if field.numel() else None,
            "field_max": float(field.max().detach().cpu()) if field.numel() else None,
            "tolerance": float(hull_tolerance),
        },
        "positivity": {
            "required": bool(require_positive),
            "ok": positive_values,
            "field_min": float(field.min().detach().cpu()) if field.numel() else None,
        },
    }


def _edge_support_overlap(
    source_support: torch.Tensor,
    destination_support: torch.Tensor,
) -> torch.Tensor:
    source_sorted = torch.sort(source_support, dim=1).values
    destination_sorted = torch.sort(destination_support, dim=1).values
    return (
        source_sorted[:, :, None] == destination_sorted[:, None, :]
    ).any(dim=2).sum(dim=1)


def _edge_group(
    baseline_jumps: np.ndarray,
    candidate_jumps: np.ndarray,
    mask: np.ndarray,
    total_count: int,
) -> dict[str, Any]:
    return {
        "edge_count": int(np.count_nonzero(mask)),
        "edge_fraction": float(np.count_nonzero(mask) / total_count)
        if total_count
        else 0.0,
        "jumps": {
            "baseline": summarize(baseline_jumps[mask]),
            "candidate": summarize(candidate_jumps[mask]),
        },
    }


def aggregate_edge_statistics(
    edges: torch.Tensor | np.ndarray,
    baseline_length: torch.Tensor | np.ndarray,
    candidate_length: torch.Tensor | np.ndarray,
    candidate_support_indices: torch.Tensor | np.ndarray,
    *,
    edge_chunk_size: int = 262144,
    support_source_count: int | None = None,
) -> dict[str, Any]:
    """Aggregate exact edge jumps and support partitions without sampling."""

    if isinstance(edge_chunk_size, bool) or int(edge_chunk_size) <= 0:
        raise ValueError("edge_chunk_size must be positive")
    base = (
        baseline_length
        if isinstance(baseline_length, torch.Tensor)
        else torch.as_tensor(baseline_length)
    ).reshape(-1)
    candidate = (
        candidate_length
        if isinstance(candidate_length, torch.Tensor)
        else torch.as_tensor(candidate_length)
    ).reshape(-1)
    if base.shape != candidate.shape:
        raise ValueError("baseline_length and candidate_length must have equal shape")
    if not bool(torch.isfinite(base).all()) or not bool(torch.isfinite(candidate).all()):
        raise ValueError("edge fields must be finite")
    if bool((base <= 0.0).any()) or bool((candidate <= 0.0).any()):
        raise ValueError("edge fields must be positive")
    edge_tensor = _as_long_tensor(edges, name="edges")
    if edge_tensor.ndim != 2 or edge_tensor.shape[1] != 2:
        raise ValueError("edges must have shape [E, 2]")
    support = _as_long_tensor(candidate_support_indices, name="candidate support")
    if support.ndim != 2 or int(support.shape[0]) != int(base.shape[0]):
        raise ValueError("candidate support must have shape [root_count, S]")
    validate_support_ids(
        support,
        source_count=(
            int(support_source_count)
            if support_source_count is not None
            else (int(support.max().detach().cpu()) + 1 if support.numel() else 1)
        ),
        name="candidate support",
    )

    device = base.device
    edge_tensor = edge_tensor.to(device=device)
    support = support.to(device=device)
    edge_count = int(edge_tensor.shape[0])
    root_count = int(base.shape[0])
    if edge_count:
        if bool((edge_tensor < 0).any()) or bool((edge_tensor >= root_count).any()):
            raise ValueError("edges contain an out-of-range root index")

    baseline_chunks: list[np.ndarray] = []
    candidate_chunks: list[np.ndarray] = []
    equal_chunks: list[np.ndarray] = []
    overlap_chunks: list[np.ndarray] = []
    chunk_size = int(edge_chunk_size)
    for begin in range(0, edge_count, chunk_size):
        end = min(begin + chunk_size, edge_count)
        edge_chunk = edge_tensor[begin:end]
        src = edge_chunk[:, 0]
        dst = edge_chunk[:, 1]
        baseline_jump = (torch.log(base[src]) - torch.log(base[dst])).abs()
        candidate_jump = (torch.log(candidate[src]) - torch.log(candidate[dst])).abs()
        source_support = support[src]
        destination_support = support[dst]
        overlap = _edge_support_overlap(source_support, destination_support)
        equal = overlap == int(support.shape[1])
        baseline_chunks.append(baseline_jump.detach().cpu().numpy().astype(np.float64, copy=False))
        candidate_chunks.append(candidate_jump.detach().cpu().numpy().astype(np.float64, copy=False))
        equal_chunks.append(equal.detach().cpu().numpy().astype(bool, copy=False))
        overlap_chunks.append(overlap.detach().cpu().numpy().astype(np.int64, copy=False))

    baseline_jumps = (
        np.concatenate(baseline_chunks) if baseline_chunks else np.empty((0,), dtype=np.float64)
    )
    candidate_jumps = (
        np.concatenate(candidate_chunks) if candidate_chunks else np.empty((0,), dtype=np.float64)
    )
    equal_support = (
        np.concatenate(equal_chunks) if equal_chunks else np.empty((0,), dtype=bool)
    )
    overlap = (
        np.concatenate(overlap_chunks) if overlap_chunks else np.empty((0,), dtype=np.int64)
    )
    if baseline_jumps.shape != (edge_count,) or candidate_jumps.shape != (edge_count,):
        raise RuntimeError("edge aggregation lost rows")
    if bool(np.any(overlap < 0)) or bool(np.any(overlap > int(support.shape[1]))):
        raise RuntimeError("support overlap is outside the exact 0..K+1 partition")

    unchanged = equal_support
    changed = ~unchanged
    unchanged_group = _edge_group(
        baseline_jumps,
        candidate_jumps,
        unchanged,
        edge_count,
    )
    changed_group = _edge_group(
        baseline_jumps,
        candidate_jumps,
        changed,
        edge_count,
    )
    overlap_counts = {
        str(overlap_count): int(np.count_nonzero(overlap == overlap_count))
        for overlap_count in range(int(support.shape[1]) + 1)
    }
    overlap_fractions = {
        key: float(value / edge_count) if edge_count else 0.0
        for key, value in overlap_counts.items()
    }
    by_overlap: dict[str, Any] = {}
    for overlap_count in range(int(support.shape[1]) + 1):
        mask = overlap == overlap_count
        by_overlap[str(overlap_count)] = {
            "edge_count": int(np.count_nonzero(mask)),
            "edge_fraction": float(np.count_nonzero(mask) / edge_count)
            if edge_count
            else 0.0,
            "jumps": {
                "baseline": _mean_p95(baseline_jumps[mask]),
                "candidate": _mean_p95(candidate_jumps[mask]),
            },
        }

    partition_exact = bool(
        unchanged_group["edge_count"] + changed_group["edge_count"] == edge_count
    )
    overlap_partition_exact = bool(sum(overlap_counts.values()) == edge_count)
    if not partition_exact or not overlap_partition_exact:
        raise RuntimeError("edge support partition is not exact")
    return {
        "edge_count": edge_count,
        "edge_chunk_size": chunk_size,
        "edge_chunk_count": int((edge_count + chunk_size - 1) // chunk_size),
        "baseline": summarize(baseline_jumps),
        "candidate": summarize(candidate_jumps),
        "full_edge_partition": {
            "definition": "candidate unordered K+1 support-ID sets are equal versus changed",
            "unchanged_support": unchanged_group,
            "changed_support": changed_group,
            "changed_to_unchanged_jump_ratio": {
                "baseline": _ratio_summary(
                    changed_group["jumps"]["baseline"],
                    unchanged_group["jumps"]["baseline"],
                ),
                "candidate": _ratio_summary(
                    changed_group["jumps"]["candidate"],
                    unchanged_group["jumps"]["candidate"],
                ),
            },
            "exact": partition_exact,
        },
        "support_overlap": {
            "definition": "unordered candidate support-ID intersection cardinality",
            "support_width": int(support.shape[1]),
            "counts": overlap_counts,
            "fractions": overlap_fractions,
            "by_overlap": by_overlap,
            "exact_full_edge_partition": overlap_partition_exact,
        },
        "exact_full_edge_partition": bool(partition_exact and overlap_partition_exact),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("cannot serialize a non-scalar tensor")
        return value.detach().cpu().item()
    return value


def normalize_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError("expected SHA-256 must be exactly 64 hexadecimal characters")
    return normalized


def select_mesh_path(
    config_mesh_path: str | Path,
    mesh_override: str | Path | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve the exact mesh path used by checkpoint reconstruction."""

    selected = Path(mesh_override) if mesh_override is not None else Path(config_mesh_path)
    if not selected.is_absolute():
        selected = Path(project_root) / selected
    return selected.expanduser().resolve()


def git_source_commit(project_root: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    commit = result.stdout.strip()
    if not commit:
        raise RuntimeError("git source commit is empty")
    return commit


def write_deterministic_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Write sorted, newline-terminated JSON and reject accidental overwrite."""

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if not overwrite:
        try:
            with output_path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite JSON output: {output_path}") from error
        return

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested": str(device),
        "type": device.type,
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        metadata.update(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return metadata


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_graph(
    edges: torch.Tensor,
    *,
    root_count: int,
    neighbor_count: int,
) -> dict[str, Any]:
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise RuntimeError("inherited render graph must have shape [E, 2]")
    effective_k = min(max(int(neighbor_count), 0), max(int(root_count) - 1, 0))
    expected_edge_count = int(root_count) * effective_k
    if int(edges.shape[0]) != expected_edge_count:
        raise RuntimeError(
            "inherited render graph is not complete: "
            f"{int(edges.shape[0])} != {expected_edge_count}"
        )
    if edges.numel():
        if bool((edges < 0).any()) or bool((edges >= int(root_count)).any()):
            raise RuntimeError("inherited render graph contains an out-of-range root ID")
        if bool((edges[:, 0] == edges[:, 1]).any()):
            raise RuntimeError("inherited render graph contains a self edge")
    return {
        "complete_edge_set": True,
        "root_count": int(root_count),
        "neighbor_count": int(effective_k),
        "edge_count": int(edges.shape[0]),
        "edge_count_expected": expected_edge_count,
        "ids_in_range": True,
        "no_self_edges": True,
    }


def _support_report_for_json(support: SurfaceSupport) -> dict[str, Any]:
    return {
        str(key): _jsonable(value)
        for key, value in support.report.items()
        if not isinstance(value, (torch.Tensor, np.ndarray))
    }


def _peak_memory_report(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "cuda": False,
        }
    _synchronize(device)
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cuda": True,
    }


def query_gradient_probe(
    interpolator: SurfaceFieldInterpolator,
    query_points: torch.Tensor,
    query_face_ids: torch.Tensor,
    support: SurfaceSupport,
    source_values: torch.Tensor,
    *,
    active_neighbor_count: int,
    max_query_count: int = 1024,
) -> dict[str, Any]:
    """Probe finite query-position gradients on a deterministic exact slice."""

    if int(max_query_count) <= 0:
        raise ValueError("max_query_count must be positive")
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError("query_points must have shape [Q, 3]")
    if int(query_face_ids.reshape(-1).shape[0]) != int(query_points.shape[0]):
        raise ValueError("query_face_ids must match query_points")
    if int(support.indices.shape[0]) != int(query_points.shape[0]):
        raise ValueError("gradient probe support must match query_points")
    probe_count = min(int(max_query_count), int(query_points.shape[0]))
    if probe_count <= 0:
        return {
            "query_count": 0,
            "max_query_count": int(max_query_count),
            "support_width": int(support.indices.shape[1]),
            "gradient_finite": True,
            "gradient_mean": None,
            "gradient_max": None,
        }

    probe_points = query_points[:probe_count].detach().clone().requires_grad_(True)
    probe_face_ids = query_face_ids[:probe_count].detach()
    probe_support = SurfaceSupport(
        indices=support.indices[:probe_count].detach(),
        vertex_path_distances=support.vertex_path_distances[:probe_count].detach(),
        report={
            **support.report,
            "query_count": probe_count,
            "probe": True,
        },
    )
    probe_source_values = source_values.detach()
    with torch.enable_grad():
        probe_distances = interpolator.distances(
            probe_points,
            probe_face_ids,
            probe_support,
        )
        probe_weights = adaptive_wendland_c2_weights(
            probe_distances,
            active_neighbor_count=int(active_neighbor_count),
            support_indices=probe_support.indices,
        )
        probe_field = interpolate_physical(
            probe_source_values,
            probe_support.indices,
            probe_weights,
        ).reshape(-1)
        (probe_gradient,) = torch.autograd.grad(
            probe_field.sum(),
            probe_points,
            create_graph=False,
            retain_graph=False,
        )
    gradient_magnitude = torch.linalg.vector_norm(probe_gradient.detach(), dim=-1)
    gradient_finite = bool(torch.isfinite(probe_gradient).all())
    if not gradient_finite:
        raise RuntimeError("candidate query-gradient probe contains non-finite values")
    result = {
        "query_count": probe_count,
        "max_query_count": int(max_query_count),
        "support_width": int(probe_support.indices.shape[1]),
        "gradient_finite": gradient_finite,
        "gradient_mean": float(gradient_magnitude.mean().cpu()),
        "gradient_max": float(gradient_magnitude.max().cpu()),
        "gradient_mean_abs_component": float(probe_gradient.detach().abs().mean().cpu()),
        "gradient_max_abs_component": float(probe_gradient.detach().abs().max().cpu()),
        "field_finite": bool(torch.isfinite(probe_field).all()),
        "weights_finite": bool(torch.isfinite(probe_weights).all()),
    }
    del (
        probe_points,
        probe_face_ids,
        probe_support,
        probe_source_values,
        probe_distances,
        probe_weights,
        probe_field,
        probe_gradient,
        gradient_magnitude,
    )
    return result


def diagnose(
    checkpoint_path: str | Path,
    output_path: str | Path,
    device: torch.device,
    mesh_path: Path | None = None,
    edge_chunk_size: int = 262144,
    expected_checkpoint_sha256: str | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the exact fixed-checkpoint R081 numeric comparison."""

    if isinstance(edge_chunk_size, bool) or int(edge_chunk_size) <= 0:
        raise ValueError("edge_chunk_size must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output == checkpoint:
        raise RuntimeError("diagnostic output must not overwrite the checkpoint")
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite JSON output: {output}")
    expected_hash = (
        normalize_sha256(expected_checkpoint_sha256)
        if expected_checkpoint_sha256 is not None
        else None
    )

    total_started = time.perf_counter()
    hash_started = time.perf_counter()
    checkpoint_hash = file_sha256(checkpoint)
    hash_seconds = time.perf_counter() - hash_started
    if expected_hash is not None and checkpoint_hash != expected_hash:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_hash}, got {checkpoint_hash}"
        )

    # Keep the pure summary/partition helpers importable on hosts that do not
    # have the renderer dependency installed.  A real diagnostic still uses
    # the formal checkpoint loader and graph builder below.
    from tools.train_white_tiger_stage1 import (  # noqa: PLC0415
        load_stage1_checkpoint_model,
        rebuild_graph_edges,
    )

    resolved_mesh_override = (
        select_mesh_path("", mesh_path) if mesh_path is not None else None
    )
    _synchronize(device)
    load_started = time.perf_counter()
    model, config, checkpoint_payload = load_stage1_checkpoint_model(
        checkpoint,
        device,
        mesh_path_override=resolved_mesh_override,
    )
    _synchronize(device)
    load_seconds = time.perf_counter() - load_started
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if not model.guide_enabled():
        raise RuntimeError("R081 length-field diagnostic requires primary guide roots")
    active_k = int(config.guide_interpolation_k)
    if active_k < 1:
        raise RuntimeError(f"R081 active guide interpolation K must be >= 1, got {active_k}")
    guide_count = int(model.guide_points_local.shape[0])
    if guide_count < active_k + 1:
        raise RuntimeError(
            "R081 K+1 candidate support is unavailable: "
            f"guide_count={guide_count}, K={active_k}"
        )
    if int(model.guide_interpolation_k) != active_k:
        raise RuntimeError("checkpoint model/config guide interpolation K disagree")

    with torch.no_grad():
        roots_world, root_normals, roots_local = model.roots_and_normals()
        root_tangents, root_bitangents = model.tangent_frames(root_normals)
    root_face_ids = model.face_ids.reshape(-1)
    root_count = int(roots_local.shape[0])
    if root_count <= 0:
        raise RuntimeError("R081 diagnostic requires at least one render root")
    if tuple(root_face_ids.shape) != (root_count,):
        raise RuntimeError("checkpoint render-root face-ID shape is invalid")
    if bool((root_face_ids < 0).any()) or bool(
        (root_face_ids >= int(model.faces.shape[0])).any()
    ):
        raise RuntimeError("checkpoint render-root face IDs are invalid")
    if not bool(torch.isfinite(roots_local).all()) or not bool(torch.isfinite(roots_world).all()):
        raise RuntimeError("current render roots contain non-finite values")
    legacy_support = model.guide_interpolation_support()
    legacy_support_validation = validate_surface_support(
        legacy_support,
        source_count=guide_count,
        expected_width=active_k,
        name="legacy primary-guide support",
    )
    _synchronize(device)
    legacy_started = time.perf_counter()
    with torch.no_grad():
        baseline_controls, _ = model.interpolate_guide_controls(
            roots_local,
            root_normals,
            root_tangents,
            root_bitangents,
        )
        if "length" not in baseline_controls:
            raise RuntimeError("legacy model guide interpolation did not return length")
        baseline_primary = baseline_controls["length"].reshape(-1)
        legacy_weights = model.guide_surface_interpolator().weights(
            roots_local,
            root_face_ids,
            legacy_support,
        )
        guide_length = decode_positive_asinh_ratio(
            model.guide_length_raw,
            model.guide_length_reference,
        ).reshape(-1)
        reconstructed_baseline = interpolate_physical(
            guide_length,
            legacy_support.indices,
            legacy_weights,
        )
    _synchronize(device)
    legacy_seconds = time.perf_counter() - legacy_started
    if not torch.allclose(
        baseline_primary,
        reconstructed_baseline,
        rtol=2.0e-5,
        atol=2.0e-7,
    ):
        error = (baseline_primary - reconstructed_baseline).abs()
        raise RuntimeError(
            "exact legacy model guide interpolation could not be reconstructed: "
            f"max_abs={float(error.max().detach().cpu()):.9g}"
        )
    legacy_validation = validate_interpolation_invariants(
        guide_length,
        legacy_support.indices,
        legacy_weights,
        interpolated=baseline_primary,
        name="legacy primary-guide interpolation",
        expected_width=active_k,
    )

    with torch.no_grad():
        effective_groom = model.apply_guide_controls(
            model.groom.decode(),
            roots_local,
            root_normals,
            root_tangents,
            root_bitangents,
        )
        effective_length = effective_groom.length.reshape(-1)
    if not bool(torch.isfinite(effective_length).all()) or bool((effective_length <= 0.0).any()):
        raise RuntimeError("exact model effective length is not finite and positive")
    effective_residual_scale = float(model.guide_length_residual_scale) * float(
        model.guide_residual_multiplier
    )
    effective_length_multiplier = effective_length / baseline_primary.clamp_min(
        torch.finfo(baseline_primary.dtype).tiny
    )

    _synchronize(device)
    candidate_build_started = time.perf_counter()
    candidate_interpolator = SurfaceFieldInterpolator(
        vertices=model.vertices,
        faces=model.faces,
        source_points=model.guide_points_local,
        source_face_ids=model.guide_face_ids,
        neighbor_count=active_k + 1,
        device=device,
    )
    _synchronize(device)
    candidate_build_seconds = time.perf_counter() - candidate_build_started
    if int(candidate_interpolator.neighbor_count) != active_k + 1:
        raise RuntimeError("candidate interpolator did not retain K+1 support width")

    _synchronize(device)
    candidate_support_started = time.perf_counter()
    with torch.no_grad():
        candidate_support = candidate_interpolator.build_support(
            roots_local,
            root_face_ids,
        )
    _synchronize(device)
    candidate_support_seconds = time.perf_counter() - candidate_support_started
    candidate_support_validation = validate_surface_support(
        candidate_support,
        source_count=guide_count,
        expected_width=active_k + 1,
        name="candidate render-root support",
    )

    _synchronize(device)
    candidate_forward_started = time.perf_counter()
    with torch.no_grad():
        candidate_distances = candidate_interpolator.distances(
            roots_local,
            root_face_ids,
            candidate_support,
        )
        candidate_weights = adaptive_wendland_c2_weights(
            candidate_distances,
            active_neighbor_count=active_k,
            support_indices=candidate_support.indices,
        )
        candidate_primary = interpolate_physical(
            guide_length,
            candidate_support.indices,
            candidate_weights,
        ).reshape(-1)
        candidate_radius = candidate_distances.kthvalue(
            active_k + 1,
            dim=1,
            keepdim=True,
        ).values.reshape(-1)
    _synchronize(device)
    candidate_forward_seconds = time.perf_counter() - candidate_forward_started
    candidate_validation = validate_interpolation_invariants(
        guide_length,
        candidate_support.indices,
        candidate_weights,
        interpolated=candidate_primary,
        name="candidate Wendland interpolation",
        expected_width=active_k + 1,
    )
    if not bool(torch.isfinite(candidate_distances).all()) or bool(
        (candidate_distances < 0.0).any()
    ):
        raise RuntimeError("candidate intrinsic distances are invalid")
    if not bool(torch.isfinite(candidate_radius).all()) or bool((candidate_radius <= 0.0).any()):
        raise RuntimeError("candidate support radius is not finite and positive")

    _synchronize(device)
    self_support_started = time.perf_counter()
    with torch.no_grad():
        guide_site_support = candidate_interpolator.build_support(
            model.guide_points_local,
            model.guide_face_ids,
        )
    _synchronize(device)
    self_support_seconds = time.perf_counter() - self_support_started
    self_support_validation = validate_surface_support(
        guide_site_support,
        source_count=guide_count,
        expected_width=active_k + 1,
        name="candidate guide-site support",
    )
    _synchronize(device)
    self_forward_started = time.perf_counter()
    with torch.no_grad():
        guide_site_distances = candidate_interpolator.distances(
            model.guide_points_local,
            model.guide_face_ids,
            guide_site_support,
        )
        guide_site_weights = adaptive_wendland_c2_weights(
            guide_site_distances,
            active_neighbor_count=active_k,
            support_indices=guide_site_support.indices,
        )
        guide_site_evaluated = interpolate_physical(
            guide_length,
            guide_site_support.indices,
            guide_site_weights,
        ).reshape(-1)
    _synchronize(device)
    self_forward_seconds = time.perf_counter() - self_forward_started
    self_validation = validate_interpolation_invariants(
        guide_length,
        guide_site_support.indices,
        guide_site_weights,
        interpolated=guide_site_evaluated,
        name="candidate guide-site self-evaluation",
        expected_width=active_k + 1,
    )
    self_absolute_error = (guide_site_evaluated - guide_length).abs()
    self_relative_error = self_absolute_error / guide_length.abs().clamp_min(
        torch.finfo(guide_length.dtype).tiny
    )

    _synchronize(device)
    gradient_probe_started = time.perf_counter()
    gradient_probe = query_gradient_probe(
        candidate_interpolator,
        roots_local,
        root_face_ids,
        candidate_support,
        guide_length,
        active_neighbor_count=active_k,
        max_query_count=1024,
    )
    _synchronize(device)
    gradient_probe_seconds = time.perf_counter() - gradient_probe_started

    _synchronize(device)
    graph_started = time.perf_counter()
    render_edges, graph_report = rebuild_graph_edges(
        model,
        mode=str(config.smooth_graph_mode),
        k=int(config.smooth_graph_k),
    )
    _synchronize(device)
    graph_seconds = time.perf_counter() - graph_started
    graph_validation = _validate_graph(
        render_edges,
        root_count=root_count,
        neighbor_count=int(config.smooth_graph_k),
    )

    field_absolute_error = (candidate_primary - baseline_primary).abs()
    field_relative_error = field_absolute_error / baseline_primary.abs().clamp_min(
        torch.finfo(baseline_primary.dtype).tiny
    )
    _synchronize(device)
    edge_started = time.perf_counter()
    with torch.no_grad():
        edge_statistics = aggregate_edge_statistics(
            render_edges,
            baseline_primary,
            candidate_primary,
            candidate_support.indices,
            edge_chunk_size=int(edge_chunk_size),
            support_source_count=guide_count,
        )
    _synchronize(device)
    edge_seconds = time.perf_counter() - edge_started

    mesh_path_resolved = (
        resolved_mesh_override
        if resolved_mesh_override is not None
        else select_mesh_path(config.mesh_path, project_root=PROJECT_ROOT)
    )
    mesh_hash = file_sha256(mesh_path_resolved)
    config_values = _jsonable(asdict(config))
    source_commit = git_source_commit(PROJECT_ROOT)
    script_path = Path(__file__).resolve()
    script_hash = file_sha256(script_path)
    checkpoint_iteration = int(checkpoint_payload.get("iteration", -1))
    _synchronize(device)
    memory_report = _peak_memory_report(device)

    legacy_support_report = _support_report_for_json(legacy_support)
    candidate_support_report = _support_report_for_json(candidate_support)
    guide_site_support_report = _support_report_for_json(guide_site_support)
    timings = {
        "checkpoint_sha256_seconds": float(hash_seconds),
        "checkpoint_model_load_seconds": float(load_seconds),
        "legacy_primary_interpolation_seconds": float(legacy_seconds),
        "candidate_interpolator_build_seconds": float(candidate_build_seconds),
        "candidate_render_support_seconds": float(candidate_support_seconds),
        "candidate_render_distance_weight_seconds": float(candidate_forward_seconds),
        "candidate_guide_site_support_seconds": float(self_support_seconds),
        "candidate_guide_site_distance_weight_seconds": float(self_forward_seconds),
        "candidate_query_gradient_probe_seconds": float(gradient_probe_seconds),
        "inherited_render_graph_seconds": float(graph_seconds),
        "edge_aggregation_seconds": float(edge_seconds),
        "total_seconds": float(time.perf_counter() - total_started),
    }

    report: dict[str, Any] = {
        "schema": "anigroom.r081_adaptive_continuous_length_field.v1",
        "diagnostic": "R081_fixed_checkpoint_adaptive_continuous_length_field",
        "status": "complete",
        "output": str(output),
        "deterministic_json": True,
        "provenance": {
            "project_root": str(PROJECT_ROOT),
            "source_commit": source_commit,
            "diagnostic_script": {
                "path": str(script_path),
                "sha256": script_hash,
            },
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
                "expected_sha256": expected_hash,
                "iteration": checkpoint_iteration,
            },
            "mesh": {
                "path": str(mesh_path_resolved),
                "sha256": mesh_hash,
            },
            "config_sha256": json_sha256(config_values),
        },
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_iteration": checkpoint_iteration,
        "config": config_values,
        "config_sha256": json_sha256(config_values),
        "execution": {
            "device": _device_metadata(device),
            "edge_chunk_size": int(edge_chunk_size),
            "sampling": "none; complete inherited render edge set",
            "visualization": False,
            "training": False,
        },
        "counts": {
            "render_root_count": root_count,
            "guide_site_count": guide_count,
            "legacy_support_width": active_k,
            "candidate_support_width": active_k + 1,
            "candidate_guide_site_support_width": active_k + 1,
            "edge_count": int(render_edges.shape[0]),
            "candidate_positive_weight_min": int(
                (candidate_weights > 0.0).sum(dim=1).min().detach().cpu()
            ),
            "candidate_positive_weight_max": int(
                (candidate_weights > 0.0).sum(dim=1).max().detach().cpu()
            ),
        },
        "timings": timings,
        "memory": memory_report,
        "supports": {
            "legacy_primary_render": legacy_support_report,
            "candidate_render": candidate_support_report,
            "candidate_guide_site": guide_site_support_report,
        },
        "graph": {
            "mode": str(config.smooth_graph_mode),
            "neighbor_count_requested": int(config.smooth_graph_k),
            "formal_report": _jsonable(graph_report),
            "validation": graph_validation,
        },
        "residual": {
            "guide_length_residual_scale": float(model.guide_length_residual_scale),
            "guide_residual_multiplier": float(model.guide_residual_multiplier),
            "effective_residual_scale": float(effective_residual_scale),
            "effective_residual_multiplier": summarize(effective_length_multiplier),
            "effective_length": summarize(effective_length),
            "effective_length_over_primary_multiplier": summarize(effective_length_multiplier),
        },
        "baseline": {
            "primary_guide_length": summarize(guide_length),
            "render_primary_length": summarize(baseline_primary),
            "render_effective_length": summarize(effective_length),
        },
        "candidate": {
            "render_primary_length": summarize(candidate_primary),
            "support_radius": summarize(candidate_radius),
            "maximum_weight": summarize(candidate_weights.max(dim=1).values),
            "effective_neighbor_count": summarize(
                1.0 / candidate_weights.square().sum(dim=1).clamp_min(torch.finfo(candidate_weights.dtype).tiny)
            ),
            "guide_site_self_evaluation": {
                "stored_guide_length": summarize(guide_length),
                "evaluated_length": summarize(guide_site_evaluated),
                "absolute_error": summarize(self_absolute_error),
                "relative_error": summarize(self_relative_error),
                "support_bytes": int(self_support_validation["support_bytes"]),
            },
        },
        "query_gradient_probe": gradient_probe,
        "field_difference": {
            "definition": "candidate primary length minus exact legacy primary-guide interpolation",
            "absolute": summarize(field_absolute_error),
            "relative_to_legacy": summarize(field_relative_error),
        },
        "edges": edge_statistics,
        "validations": {
            "checkpoint_schema_and_model_load": True,
            "primary_guides_present": True,
            "active_k_at_least_one": True,
            "legacy_support": legacy_support_validation,
            "candidate_support": candidate_support_validation,
            "candidate_guide_site_support": self_support_validation,
            "legacy_interpolation": legacy_validation,
            "candidate_interpolation": candidate_validation,
            "candidate_guide_site_self_evaluation": self_validation,
            "candidate_intrinsic_distances_finite_nonnegative": True,
            "candidate_support_radius_finite_positive": True,
            "effective_length_finite_positive": True,
            "candidate_query_gradient_probe": gradient_probe,
            "inherited_graph_complete": graph_validation,
            "edge_partition_exact": bool(edge_statistics["exact_full_edge_partition"]),
            "field_difference_finite": bool(
                torch.isfinite(field_absolute_error).all()
                and torch.isfinite(field_relative_error).all()
            ),
        },
        "root_face_id_count": root_count,
        "render_root_local_face_ids": {
            "count": root_count,
            "unique_face_count": int(torch.unique(root_face_ids).numel()),
        },
    }

    write_deterministic_json(output, report, overwrite=overwrite)
    return report


def concise_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the terminal report without the nested per-overlap details."""

    edges = report.get("edges", {})
    return {
        "status": report.get("status"),
        "schema": report.get("schema"),
        "output": report.get("output"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "checkpoint_iteration": report.get("checkpoint_iteration"),
        "source_commit": report.get("source_commit"),
        "render_root_count": report.get("counts", {}).get("render_root_count"),
        "guide_site_count": report.get("counts", {}).get("guide_site_count"),
        "edge_count": edges.get("edge_count"),
        "candidate_field_difference_p95": report.get("field_difference", {})
        .get("absolute", {})
        .get("q95"),
        "candidate_edge_jump_p95": edges.get("candidate", {}).get("q95"),
        "candidate_support_changed_edge_fraction": report.get("edges", {})
        .get("full_edge_partition", {})
        .get("changed_support", {})
        .get("edge_fraction"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded R081 fixed-checkpoint length-field diagnostic."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mesh", default=None, type=Path)
    parser.add_argument("--edge-chunk-size", type=int, default=262144)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = diagnose(
        args.checkpoint,
        args.output,
        torch.device(args.device),
        mesh_path=args.mesh.expanduser().resolve() if args.mesh is not None else None,
        edge_chunk_size=args.edge_chunk_size,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        overwrite=bool(args.overwrite),
    )
    concise = concise_report({**report, "output": str(args.output.expanduser().resolve())})
    print(json.dumps(concise, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

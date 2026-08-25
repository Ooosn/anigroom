"""Audit deterministic Gaussian counts reconstructed from an AniGroom checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FINGERPRINT_KEYS = (
    "total_gaussian_count",
    "root_count",
    "segment_histogram",
    "per_root_segment_counts_sha256",
    "root_indices_order_sha256",
    "segment_indices_order_sha256",
    "root_segment_order_sha256",
)


def _formal_checkpoint_helpers() -> tuple[Any, Any, Any, Any]:
    """Import the formal trainer only when a real checkpoint audit runs."""

    from tools.train_white_tiger_stage1 import (  # noqa: PLC0415
        build_stage1_model_from_checkpoint,
        load_training_checkpoint,
        resolve_project_path,
        stage1_config_from_checkpoint_mapping,
    )

    return (
        build_stage1_model_from_checkpoint,
        load_training_checkpoint,
        resolve_project_path,
        stage1_config_from_checkpoint_mapping,
    )


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _canonical_index_array(values: torch.Tensor | np.ndarray, name: str) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if values.dtype.is_floating_point or values.dtype.is_complex:
            raise TypeError(f"{name} must have an integer dtype")
        values = values.detach().to(device="cpu", dtype=torch.int64).numpy()
    else:
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError(f"{name} must have an integer dtype")
        values = values.astype(np.int64, copy=False)
    return np.ascontiguousarray(values, dtype=np.int64)


def derive_per_root_segment_counts(
    root_indices: torch.Tensor | np.ndarray,
    segment_indices: torch.Tensor | np.ndarray,
    *,
    root_count: int | None = None,
) -> dict[str, Any]:
    """Derive exact per-root segment counts from the flattened Gaussian order."""

    roots = _canonical_index_array(root_indices, "root_indices")
    segments = _canonical_index_array(segment_indices, "segment_indices")
    if roots.shape != segments.shape:
        raise ValueError(
            "root_indices and segment_indices must have equal shape: "
            f"{roots.shape} != {segments.shape}"
        )
    if np.any(roots < 0) or np.any(segments < 0):
        raise ValueError("root_indices and segment_indices must be non-negative")

    inferred_root_count = int(roots.max()) + 1 if roots.size else 0
    if root_count is None:
        root_count = inferred_root_count
    root_count = int(root_count)
    if root_count < inferred_root_count:
        raise ValueError(
            f"root_count {root_count} is smaller than the largest root index"
        )

    counts = np.zeros(root_count, dtype=np.int64)
    if roots.size:
        np.maximum.at(counts, roots, segments + 1)
    if int(counts.sum()) != int(roots.size):
        raise RuntimeError(
            "segment indices are not a contiguous per-root order: "
            f"sum(per_root_counts)={int(counts.sum())} != "
            f"gaussian_count={int(roots.size)}"
        )

    histogram_values, histogram_counts = np.unique(counts, return_counts=True)
    histogram = {
        str(int(value)): int(count)
        for value, count in zip(histogram_values, histogram_counts, strict=True)
    }
    root_segment_order = np.column_stack((roots, segments))
    return {
        "total_gaussian_count": int(roots.size),
        "root_count": root_count,
        "segment_histogram": histogram,
        "per_root_segment_counts": counts,
        "per_root_segment_counts_sha256": _sha256_array(counts),
        "root_indices_order_sha256": _sha256_array(roots),
        "segment_indices_order_sha256": _sha256_array(segments),
        "root_segment_order_sha256": _sha256_array(root_segment_order),
    }


def repeat_fingerprint(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result[key] for key in FINGERPRINT_KEYS}


def require_exact_repeat_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one reconstruction repeat is required")
    baseline = repeat_fingerprint(results[0])
    mismatches = []
    for index, result in enumerate(results[1:], start=1):
        current = repeat_fingerprint(result)
        if current != baseline:
            mismatches.append(
                {
                    "repeat": index,
                    "baseline": baseline,
                    "observed": current,
                }
            )
    if mismatches:
        raise RuntimeError(
            "checkpoint Gaussian reconstruction repeats differ: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return baseline


def pre_step_training_metric_minus_checkpoint_state(
    training_metric_count: int | None,
    checkpoint_state_count: int,
) -> int | None:
    if training_metric_count is None:
        return None
    return int(training_metric_count) - int(checkpoint_state_count)


def _config_for_render_parameters(config: Any) -> dict[str, int | float]:
    return {
        "samples": int(config.samples),
        "child_count": int(config.child_count),
        "min_segments": int(config.min_segments),
        "segment_length_origin": float(config.segment_length_origin),
        "segments_per_unit_length": float(config.segments_per_unit_length),
        "segments_per_unit_complexity": float(config.segments_per_unit_complexity),
        "length_overlap": float(config.gaussian_length_overlap),
    }


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {"requested": str(device), "type": device.type}
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


def _repeat_report(
    result: dict[str, Any],
    *,
    gaussian_dtype: torch.dtype,
    root_indices_dtype: torch.dtype,
    segment_indices_dtype: torch.dtype,
    repeat: int,
) -> dict[str, Any]:
    report = {
        key: value
        for key, value in result.items()
        if key != "per_root_segment_counts"
    }
    report.update(
        {
            "repeat": repeat,
            "gaussian_dtype": str(gaussian_dtype),
            "root_indices_dtype": str(root_indices_dtype),
            "segment_indices_dtype": str(segment_indices_dtype),
        }
    )
    return report


def audit_checkpoint(
    checkpoint: str | Path,
    *,
    output: str | Path,
    device: str = "cuda",
    repeats: int = 3,
    training_metric_count: int | None = None,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    (
        build_stage1_model_from_checkpoint,
        load_training_checkpoint,
        resolve_project_path,
        stage1_config_from_checkpoint_mapping,
    ) = _formal_checkpoint_helpers()
    checkpoint_path = resolve_project_path(checkpoint)
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    raw_checkpoint = load_training_checkpoint(checkpoint_path)
    config = stage1_config_from_checkpoint_mapping(raw_checkpoint["config"])
    config_dict = asdict(config)
    render_config = _config_for_render_parameters(config)
    device_obj = torch.device(device)
    model = build_stage1_model_from_checkpoint(raw_checkpoint, config, device_obj)
    model.eval()
    root_count = int(model.face_ids.shape[0])

    repeat_reports: list[dict[str, Any]] = []
    repeat_results: list[dict[str, Any]] = []
    for repeat in range(repeats):
        with torch.no_grad():
            gaussians, _, _, _, _, _, _ = model.render_parameters(
                samples=render_config["samples"],
                child_count=render_config["child_count"],
                min_segments=render_config["min_segments"],
                segment_length_origin=render_config["segment_length_origin"],
                segments_per_unit_length=render_config["segments_per_unit_length"],
                segments_per_unit_complexity=render_config["segments_per_unit_complexity"],
                length_overlap=render_config["length_overlap"],
            )
        if int(gaussians.means.shape[0]) != int(gaussians.root_indices.numel()):
            raise RuntimeError(
                "render_parameters returned mismatched Gaussian/root-index counts: "
                f"{int(gaussians.means.shape[0])} != "
                f"{int(gaussians.root_indices.numel())}"
            )
        result = derive_per_root_segment_counts(
            gaussians.root_indices,
            gaussians.segment_indices,
            root_count=root_count,
        )
        repeat_results.append(result)
        repeat_reports.append(
            _repeat_report(
                result,
                gaussian_dtype=gaussians.means.dtype,
                root_indices_dtype=gaussians.root_indices.dtype,
                segment_indices_dtype=gaussians.segment_indices.dtype,
                repeat=repeat,
            )
        )
        del gaussians
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
            torch.cuda.empty_cache()

    status = "pass"
    mismatch: str | None = None
    try:
        baseline = require_exact_repeat_results(repeat_results)
    except RuntimeError as error:
        status = "mismatch"
        mismatch = str(error)
        baseline = repeat_fingerprint(repeat_results[0])

    report: dict[str, Any] = {
        "status": status,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "iteration": int(raw_checkpoint.get("iteration", -1)),
        "device": _device_metadata(device_obj),
        "dtype": repeat_reports[0]["gaussian_dtype"],
        "config_sha256": _json_sha256(config_dict),
        "config_for_render_parameters": render_config,
        "repeats_requested": repeats,
        "repeats_completed": len(repeat_reports),
        "repeats": repeat_reports,
        **baseline,
        "training_metric_count": training_metric_count,
        "pre_step_training_metric_minus_checkpoint_state": (
            pre_step_training_metric_minus_checkpoint_state(
                training_metric_count,
                int(baseline["total_gaussian_count"]),
            )
        ),
    }
    if mismatch is not None:
        report["repeat_mismatch"] = mismatch
    atomic_write_json(output, report)
    if mismatch is not None:
        raise RuntimeError(mismatch)
    return report


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact checkpoint-derived Gaussian segment counts."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--training-metric-count", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_checkpoint(
            args.checkpoint,
            output=args.output,
            device=args.device,
            repeats=args.repeats,
            training_metric_count=args.training_metric_count,
        )
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

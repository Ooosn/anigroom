"""CUDA runner for the isolated post-V8 global direction-field refiner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from anigroom.flow.global_direction_field_refinement import (  # noqa: E402
    GlobalDirectionFieldRefinementConfig,
    refine_global_direction_field,
)


REQUIRED_FIELDS = (
    "root_points",
    "root_normals",
    "shell_h",
    "cleaned_directed_flow3d",
    "flow3d",
    "axis_view_cluster_selected_direct_vectors",
    "axis_view_cluster_selected_direct_weight",
    "axis_view_cluster_postratio_edge_u",
    "axis_view_cluster_postratio_edge_v",
)
VIEW_COUNT = 36
EXPECTED_ROOT_COUNT = 4500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(value: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "p01": float(np.quantile(finite, 0.01)),
        "p05": float(np.quantile(finite, 0.05)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def parse_exclude(value: str) -> tuple[int, ...]:
    entries = [item.strip() for item in str(value).split(",") if item.strip()]
    try:
        views = tuple(int(item) for item in entries)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("exclude must be comma-separated integers") from exc
    if len(set(views)) != len(views):
        raise argparse.ArgumentTypeError("exclude must not contain duplicate view ids")
    if any(view < 0 or view >= VIEW_COUNT for view in views):
        raise argparse.ArgumentTypeError("exclude view ids must be in [0, 35]")
    return views


def _finite_array(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise RuntimeError(f"target array {name} contains non-finite values")


def _validate_target_arrays(
    *,
    arrays: dict[str, np.ndarray],
    view_count: int,
) -> int:
    root_points = np.asarray(arrays["root_points"])
    root_normals = np.asarray(arrays["root_normals"])
    shell_h = np.asarray(arrays["shell_h"])
    cleaned = np.asarray(arrays["cleaned_directed_flow3d"])
    flow3d = np.asarray(arrays["flow3d"])
    if root_points.shape != (EXPECTED_ROOT_COUNT, 3):
        raise RuntimeError(f"root_points must have shape [{EXPECTED_ROOT_COUNT}, 3]")
    root_count = int(root_points.shape[0])
    if root_normals.shape != (root_count, 3):
        raise RuntimeError("root_normals must have shape [N, 3]")
    if shell_h.shape != (root_count,):
        raise RuntimeError("shell_h must have shape [N]")
    if cleaned.shape != (root_count, 3):
        raise RuntimeError("cleaned_directed_flow3d must have shape [N, 3]")
    if flow3d.shape != (root_count, 3):
        raise RuntimeError("flow3d must have shape [N, 3]")

    vectors = np.asarray(arrays["axis_view_cluster_selected_direct_vectors"])
    weights = np.asarray(arrays["axis_view_cluster_selected_direct_weight"])
    if vectors.shape != (view_count, root_count, 3):
        raise RuntimeError(
            "axis_view_cluster_selected_direct_vectors must have shape "
            "[selected_views, N, 3]"
        )
    if weights.shape != (view_count, root_count):
        raise RuntimeError(
            "axis_view_cluster_selected_direct_weight must have shape "
            "[selected_views, N]"
        )
    if bool((weights < 0.0).any()):
        raise RuntimeError("axis_view_cluster_selected_direct_weight must be non-negative")

    edge_u = np.asarray(arrays["axis_view_cluster_postratio_edge_u"])
    edge_v = np.asarray(arrays["axis_view_cluster_postratio_edge_v"])
    if edge_u.ndim != 1 or edge_v.shape != edge_u.shape or edge_u.size == 0:
        raise RuntimeError("postratio graph edges must be non-empty matching [E] arrays")
    if not np.issubdtype(edge_u.dtype, np.integer) or not np.issubdtype(
        edge_v.dtype, np.integer
    ):
        raise RuntimeError("postratio graph edges must be integer arrays")
    if bool(
        ((edge_u < 0) | (edge_u >= root_count) | (edge_v < 0) | (edge_v >= root_count)).any()
    ):
        raise RuntimeError("postratio graph contains an out-of-range root index")

    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number):
            _finite_array(name, value)
    for name in ("root_normals", "cleaned_directed_flow3d", "flow3d"):
        if float(np.linalg.norm(np.asarray(arrays[name]), axis=-1).min()) <= 1.0e-8:
            raise RuntimeError(f"target array {name} contains a zero vector")
    return root_count


def main() -> None:
    defaults = GlobalDirectionFieldRefinementConfig()
    parser = argparse.ArgumentParser(
        description="Run the isolated post-V8 global direction-field refinement on one target."
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-target-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude",
        type=parse_exclude,
        default=parse_exclude("4,24,25"),
        help="comma-separated stored view ids to exclude (default: 4,24,25)",
    )
    parser.add_argument("--smooth-weight", type=float, required=True)
    parser.add_argument(
        "--orientation-barrier-weight",
        type=float,
        default=defaults.orientation_barrier_weight,
    )
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument(
        "--backtracking-steps",
        type=int,
        default=defaults.backtracking_steps,
    )
    args = parser.parse_args()

    target = args.target.resolve()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise RuntimeError("refusing to overwrite global direction-field refinement output")
    if not target.is_file() or not data_root.is_dir():
        raise RuntimeError("target or data root is missing")

    expected_sha = str(args.expected_target_sha256).strip().lower()
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise RuntimeError("expected-target-sha256 must be a 64-character hexadecimal SHA-256")
    target_sha = sha256_file(target)
    if target_sha != expected_sha:
        raise RuntimeError(f"target SHA-256 mismatch: {target_sha}")

    excluded = set(args.exclude)
    view_ids = [view for view in range(VIEW_COUNT) if view not in excluded]
    if not view_ids:
        raise RuntimeError("exclude removes every stored camera view")

    with np.load(target, allow_pickle=False) as payload:
        missing = [key for key in REQUIRED_FIELDS if key not in payload]
        if missing:
            raise RuntimeError(f"target is missing required fields: {missing}")
        parent_arrays = {key: np.array(payload[key], copy=True) for key in payload.files}
        arrays = {key: np.asarray(payload[key]) for key in REQUIRED_FIELDS}
    root_count = _validate_target_arrays(arrays=arrays, view_count=len(view_ids))

    cameras_extr_path = data_root / "cameras_extr.npy"
    cameras_intr_path = data_root / "cameras_intr.npy"
    if not cameras_extr_path.is_file() or not cameras_intr_path.is_file():
        raise RuntimeError("data root is missing cameras_extr.npy or cameras_intr.npy")
    cameras_extr = np.load(cameras_extr_path, allow_pickle=False).astype(np.float32)
    cameras_intr = np.load(cameras_intr_path, allow_pickle=False).astype(np.float32)
    cameras_intr = cameras_intr[:, :3, :3]
    if cameras_extr.shape != (VIEW_COUNT, 4, 4) or cameras_intr.shape != (
        VIEW_COUNT,
        3,
        3,
    ):
        raise RuntimeError("camera arrays do not have the expected 36-view shape")
    _finite_array("cameras_extr", cameras_extr)
    _finite_array("cameras_intr", cameras_intr)

    config = GlobalDirectionFieldRefinementConfig(
        smooth_weight=float(args.smooth_weight),
        orientation_barrier_weight=float(args.orientation_barrier_weight),
        iterations=int(args.iterations),
        learning_rate=float(args.learning_rate),
        patience=int(args.patience),
        backtracking_steps=int(args.backtracking_steps),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("global direction-field refinement requires CUDA")

    started = time.perf_counter()
    device = torch.device("cuda")
    direction = F.normalize(
        torch.as_tensor(
            np.asarray(arrays["cleaned_directed_flow3d"], dtype=np.float32),
            device=device,
        ),
        dim=-1,
    )
    normals = F.normalize(
        torch.as_tensor(
            np.asarray(arrays["root_normals"], dtype=np.float32),
            device=device,
        ),
        dim=-1,
    )
    points = torch.as_tensor(
        np.asarray(arrays["root_points"], dtype=np.float32),
        device=device,
    )
    shell_h = torch.as_tensor(
        np.asarray(arrays["shell_h"], dtype=np.float32),
        device=device,
    )
    projection_points = points + shell_h[:, None] * normals
    per_view_axes = torch.as_tensor(
        np.asarray(
            arrays["axis_view_cluster_selected_direct_vectors"],
            dtype=np.float32,
        ),
        device=device,
    )
    per_view_weights = torch.as_tensor(
        np.asarray(
            arrays["axis_view_cluster_selected_direct_weight"],
            dtype=np.float32,
        ),
        device=device,
    )
    selected_view_ids = torch.as_tensor(view_ids, dtype=torch.long, device=device)
    viewmats = torch.as_tensor(cameras_extr, device=device)[selected_view_ids]
    intrinsics = torch.as_tensor(cameras_intr, device=device)[selected_view_ids]
    edge_u = torch.as_tensor(
        np.asarray(arrays["axis_view_cluster_postratio_edge_u"], dtype=np.int64),
        device=device,
    )
    edge_v = torch.as_tensor(
        np.asarray(arrays["axis_view_cluster_postratio_edge_v"], dtype=np.int64),
        device=device,
    )

    result = refine_global_direction_field(
        direction=direction,
        normals=normals,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
        edge_u=edge_u,
        edge_v=edge_v,
        config=config,
    )

    staging.mkdir(parents=True, exist_ok=False)
    final_path = staging / "final_directions.npy"
    final_direction_np = result.direction.detach().cpu().numpy().astype(np.float32)
    np.save(final_path, final_direction_np, allow_pickle=False)

    normal_np = np.asarray(arrays["root_normals"], dtype=np.float32)
    normal_np = normal_np / np.maximum(
        np.linalg.norm(normal_np, axis=-1, keepdims=True),
        1.0e-8,
    )
    tangent_axis_np = np.asarray(arrays["flow3d"], dtype=np.float32)
    tangent_axis_np = tangent_axis_np / np.maximum(
        np.linalg.norm(tangent_axis_np, axis=-1, keepdims=True),
        1.0e-8,
    )
    normal_component_np = np.clip(
        np.sum(final_direction_np * normal_np, axis=-1),
        0.0,
        1.0,
    )
    tangent_np = final_direction_np - normal_component_np[:, None] * normal_np
    tangent_length_np = np.maximum(np.linalg.norm(tangent_np, axis=-1), 1.0e-8)
    parent_direction_np = np.asarray(
        parent_arrays["cleaned_directed_flow3d"],
        dtype=np.float32,
    ).copy()
    parent_arrays["cleaned_directed_flow3d"] = final_direction_np
    parent_arrays["cleaned_direction_lambda"] = np.asarray(
        normal_component_np / tangent_length_np,
        dtype=np.float32,
    )
    parent_arrays["cleaned_direction_sign"] = np.where(
        np.sum(tangent_np * tangent_axis_np, axis=-1) >= 0.0,
        1.0,
        -1.0,
    ).astype(np.float32)

    metadata_prefix = "post_v8_global_direction_field_refinement"
    parent_arrays[f"{metadata_prefix}_parent_sha256"] = np.asarray(target_sha)
    parent_arrays[f"{metadata_prefix}_parent_direction"] = parent_direction_np
    parent_arrays[f"{metadata_prefix}_final_direction"] = final_direction_np
    parent_arrays[f"{metadata_prefix}_algorithm"] = np.asarray(
        "global_direction_field_refinement"
    )
    parent_arrays[f"{metadata_prefix}_smooth_weight"] = np.asarray(
        float(config.smooth_weight),
        dtype=np.float64,
    )
    parent_arrays[f"{metadata_prefix}_orientation_barrier_weight"] = np.asarray(
        float(config.orientation_barrier_weight),
        dtype=np.float64,
    )
    parent_arrays[f"{metadata_prefix}_training"] = np.asarray(False)

    candidate_target_path = staging / "candidate_target.npz"
    np.savez_compressed(candidate_target_path, **parent_arrays)

    report = {
        "schema": "anigroom.global_direction_field_refinement.actual_input.v1",
        "status": "complete",
        "training": False,
        "algorithm": "global_direction_field_refinement",
        "smooth_weight": float(config.smooth_weight),
        "orientation_barrier_weight": float(config.orientation_barrier_weight),
        "target": {"path": str(target), "sha256": target_sha},
        "data_root": str(data_root),
        "exclude": sorted(excluded),
        "view_ids": view_ids,
        "matrix_contract": {
            "root_count": root_count,
            "per_view_vectors_shape": list(per_view_axes.shape),
            "per_view_weights_shape": list(per_view_weights.shape),
            "positive_per_view_pair_count": int((per_view_weights > 0.0).sum().cpu()),
            "surface_edge_count": int(edge_u.numel()),
        },
        "per_view_weight": distribution(
            per_view_weights.detach().cpu().numpy()[
                per_view_weights.detach().cpu().numpy() > 0.0
            ]
        ),
        "solver": result.report,
        "artifacts": {
            "final_directions": {
                "name": final_path.name,
                "sha256": sha256_file(final_path),
            },
            "candidate_target": {
                "name": candidate_target_path.name,
                "sha256": sha256_file(candidate_target_path),
            },
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report_path = staging / "global_direction_field_refinement.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts = [report_path, final_path, candidate_target_path]
    manifest_path = staging / "manifest.sha256"
    manifest_path.write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in artifacts) + "\n",
        encoding="utf-8",
    )
    staging.rename(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "final_directions": str(output / final_path.name),
                "candidate_target": str(output / candidate_target_path.name),
                "report": str(output / report_path.name),
                "manifest": str(output / manifest_path.name),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

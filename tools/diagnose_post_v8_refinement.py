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

from anigroom.flow.post_v8_refinement import (  # noqa: E402
    PostV8RefinementConfig,
    run_post_v8_refinement,
)


REQUIRED_FIELDS = (
    "root_points",
    "root_normals",
    "shell_h",
    "cleaned_directed_flow3d",
    "flow3d",
    "observed",
    "axis_view_cluster_selected_direct_vectors",
    "axis_view_cluster_selected_direct_weight",
    "axis_view_cluster_final_confidence",
    "axis_view_cluster_global_unary_normalized_margin",
    "axis_view_cluster_global_unary_vote_coherence",
    "axis_view_cluster_global_canonical_rank",
    "axis_view_cluster_postratio_edge_u",
    "axis_view_cluster_postratio_edge_v",
    "axis_view_cluster_confidence_flow_joint_confidence",
)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run automatic BA/propagation cycles from one formal V8 target."
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expected-target-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude", default="4,24,25")
    args = parser.parse_args()

    target = args.target.resolve()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise RuntimeError("refusing to overwrite post-V8 refinement output")
    if not target.is_file() or not data_root.is_dir():
        raise RuntimeError("target or data root is missing")
    target_sha = sha256_file(target)
    if target_sha != str(args.expected_target_sha256).lower():
        raise RuntimeError(f"target SHA-256 mismatch: {target_sha}")
    excluded = {int(value) for value in str(args.exclude).split(",") if value.strip()}
    view_ids = [view for view in range(36) if view not in excluded]

    with np.load(target, allow_pickle=False) as payload:
        missing = [key for key in REQUIRED_FIELDS if key not in payload]
        if missing:
            raise RuntimeError(f"V8 target is missing required fields: {missing}")
        parent_arrays = {key: np.array(payload[key], copy=True) for key in payload.files}
        arrays = {key: np.asarray(payload[key]) for key in REQUIRED_FIELDS}
    root_count = int(arrays["root_points"].shape[0])
    view_count = int(arrays["axis_view_cluster_selected_direct_weight"].shape[0])
    if root_count != 4500 or view_count != len(view_ids):
        raise RuntimeError(
            f"unexpected V8 matrix shape: roots={root_count}, views={view_count}"
        )
    if arrays["axis_view_cluster_selected_direct_vectors"].shape != (
        view_count,
        root_count,
        3,
    ):
        raise RuntimeError("selected direct vector tensor has the wrong shape")
    if arrays["axis_view_cluster_selected_direct_weight"].shape != (
        view_count,
        root_count,
    ):
        raise RuntimeError("selected direct weight matrix has the wrong shape")

    field_confidence_np = np.asarray(
        arrays["axis_view_cluster_final_confidence"],
        dtype=np.float32,
    )
    unary_margin_np = np.asarray(
        arrays["axis_view_cluster_global_unary_normalized_margin"],
        dtype=np.float32,
    )
    vote_coherence_np = np.asarray(
        arrays["axis_view_cluster_global_unary_vote_coherence"],
        dtype=np.float32,
    )
    joint_np = field_confidence_np * unary_margin_np * vote_coherence_np
    stored_joint_np = np.asarray(
        arrays["axis_view_cluster_confidence_flow_joint_confidence"],
        dtype=np.float32,
    )
    joint_error = float(np.max(np.abs(joint_np - stored_joint_np)))
    if joint_error > 2.0e-7:
        raise RuntimeError(f"stored V8 joint confidence mismatch: {joint_error}")

    cameras_extr = np.load(data_root / "cameras_extr.npy", allow_pickle=False).astype(
        np.float32
    )
    cameras_intr = np.load(data_root / "cameras_intr.npy", allow_pickle=False).astype(
        np.float32
    )[:, :3, :3]
    if cameras_extr.shape != (36, 4, 4) or cameras_intr.shape != (36, 3, 3):
        raise RuntimeError("camera arrays do not have the expected 36-view shape")

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
    observed = torch.as_tensor(
        np.asarray(arrays["observed"], dtype=bool),
        device=device,
    )
    edge_u = torch.as_tensor(
        np.asarray(arrays["axis_view_cluster_postratio_edge_u"], dtype=np.int64),
        device=device,
    )
    edge_v = torch.as_tensor(
        np.asarray(arrays["axis_view_cluster_postratio_edge_v"], dtype=np.int64),
        device=device,
    )
    canonical_rank = torch.as_tensor(
        np.asarray(
            arrays["axis_view_cluster_global_canonical_rank"],
            dtype=np.int64,
        ),
        device=device,
    )
    config = PostV8RefinementConfig()
    result = run_post_v8_refinement(
        direction=direction,
        normals=normals,
        projection_points=projection_points,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
        observed=observed,
        edge_u=edge_u,
        edge_v=edge_v,
        field_confidence=torch.as_tensor(field_confidence_np, device=device),
        unary_normalized_margin=torch.as_tensor(unary_margin_np, device=device),
        unary_vote_coherence=torch.as_tensor(vote_coherence_np, device=device),
        canonical_rank=canonical_rank,
        config=config,
    )

    staging.mkdir(parents=True, exist_ok=False)
    artifacts: list[Path] = []
    for cycle, cycle_direction in enumerate(result.cycle_directions, start=1):
        path = staging / f"cycle_{cycle}_directions.npy"
        np.save(
            path,
            cycle_direction.detach().cpu().numpy().astype(np.float32),
            allow_pickle=False,
        )
        artifacts.append(path)
    final_path = staging / "final_directions.npy"
    np.save(
        final_path,
        result.direction.detach().cpu().numpy().astype(np.float32),
        allow_pickle=False,
    )
    artifacts.append(final_path)
    final_direction_np = result.direction.detach().cpu().numpy().astype(np.float32)
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
    parent_arrays["post_v8_refinement_parent_sha256"] = np.asarray(target_sha)
    parent_arrays["post_v8_refinement_parent_direction"] = parent_direction_np
    parent_arrays["post_v8_refinement_final_direction"] = final_direction_np
    parent_arrays["post_v8_refinement_accepted_cycle_count"] = np.asarray(
        len(result.cycle_directions),
        dtype=np.int32,
    )
    parent_arrays["post_v8_refinement_confidence_recomputed"] = np.asarray(False)
    candidate_target_path = staging / "candidate_target.npz"
    np.savez_compressed(candidate_target_path, **parent_arrays)
    artifacts.append(candidate_target_path)
    report = {
        "schema": "anigroom.post_v8_refinement.actual_input.v1",
        "status": "complete",
        "training": False,
        "target": {"path": str(target), "sha256": target_sha},
        "data_root": str(data_root),
        "view_ids": view_ids,
        "matrix_contract": {
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
        "joint_confidence": distribution(joint_np),
        "joint_confidence_reconstruction_max_abs_error": joint_error,
        "solver": result.report,
        "artifacts": {
            "cycles": [
                f"cycle_{cycle}_directions.npy"
                for cycle in range(1, len(result.cycle_directions) + 1)
            ],
            "final": final_path.name,
            "candidate_target": {
                "name": candidate_target_path.name,
                "sha256": sha256_file(candidate_target_path),
            },
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report_path = staging / "post_v8_refinement.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts.insert(0, report_path)
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
                "report": str(output / report_path.name),
                "manifest": str(output / manifest_path.name),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

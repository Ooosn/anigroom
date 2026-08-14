#!/usr/bin/env bash
set -euo pipefail

mode="${1:?expected preflight or final}"
project_root="${2:?missing project root}"
python_bin="${3:?missing Python interpreter}"
mesh_path="${4:?missing mesh path}"
output_dir="${5:?missing output directory}"
runtime_root="${6:?missing runtime root}"
sdf_path="${MESH_NO_PENETRATION_SDF:?missing mesh SDF path}"
expected_sdf_sha="${EXPECTED_MESH_NO_PENETRATION_SDF_SHA256:?missing expected SDF SHA256}"

case "$mode" in
  preflight)
    checkpoint="$output_dir/checkpoint_000002.pt"
    diagnostic_dir="$runtime_root/postprocess/r062_preflight_no_penetration"
    visual_strands=1
    ;;
  final)
    checkpoint="$output_dir/checkpoint_030000.pt"
    diagnostic_dir="$runtime_root/postprocess/r062_mesh_no_penetration/no_penetration"
    visual_strands=100000
    ;;
  *)
    echo "[r062] unsupported verification mode: $mode" >&2
    exit 2
    ;;
esac

[[ -s "$checkpoint" ]] || {
  echo "[r062] missing checkpoint for $mode verification: $checkpoint" >&2
  exit 2
}
if [[ -e "$diagnostic_dir" ]]; then
  echo "[r062] diagnostic output already exists: $diagnostic_dir" >&2
  exit 2
fi

"$python_bin" "$project_root/tools/diagnose_checkpoint_no_penetration.py" \
  --checkpoint "$checkpoint" \
  --sdf "$sdf_path" \
  --mesh "$mesh_path" \
  --output-dir "$diagnostic_dir" \
  --device cuda \
  --samples 64 \
  --query-root-chunk 16384 \
  --gradient-root-batch 16384 \
  --visual-strands "$visual_strands"

"$python_bin" - \
  "$mode" \
  "$checkpoint" \
  "$output_dir/config.json" \
  "$output_dir/metrics.jsonl" \
  "$diagnostic_dir/report.json" \
  "$expected_sdf_sha" \
  "${R061_BASELINE_PREFLIGHT_METRICS:?missing R061 preflight metrics}" \
  "${R061_NO_PENETRATION_REPORT:?missing R061 collision report}" \
  "$diagnostic_dir/acceptance.json" <<'PY'
import json
import math
from pathlib import Path
import sys

import torch

(
    mode,
    checkpoint_path,
    config_path,
    metrics_path,
    diagnostic_path,
    expected_sdf_sha,
    baseline_preflight_metrics_path,
    baseline_collision_path,
    output_path,
) = sys.argv[1:]

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
diagnostic = json.loads(Path(diagnostic_path).read_text(encoding="utf-8"))

def metric_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

rows = [row for row in metric_rows(metrics_path) if "train" in row]
if not rows:
    raise RuntimeError(f"R062 {mode} produced no metric rows")
last = rows[-1]

if not bool(config.get("mesh_no_penetration_support", False)):
    raise RuntimeError("R062 collision support is disabled")
if float(config.get("mesh_no_penetration_weight", 0.0)) != 256.0:
    raise RuntimeError("R062 collision weight is not the reviewed value 256")
if int(config.get("mesh_no_penetration_root_batch", 0)) != 16384:
    raise RuntimeError("R062 collision root batch is not 16384")
if bool(config.get("local_child_color_support", True)):
    raise RuntimeError("R062 re-enabled retired per-render-root color")
if checkpoint.get("mesh_no_penetration_sdf_sha256") != expected_sdf_sha:
    raise RuntimeError("R062 checkpoint did not freeze the reviewed SDF SHA256")
if diagnostic["sdf"]["sdf_sha256"] != expected_sdf_sha:
    raise RuntimeError("R062 diagnostic used an unexpected SDF")

collision_metric = last.get("mesh_no_penetration")
if not isinstance(collision_metric, dict):
    raise RuntimeError("R062 metrics are missing mesh_no_penetration")
sampled_roots = int(collision_metric.get("sampled_root_count", 0))
sampled_points = int(collision_metric.get("sampled_point_count", 0))
samples = int(config["samples"])
if sampled_roots <= 0 or sampled_points != sampled_roots * (samples - 1):
    raise RuntimeError(
        f"R062 collision sample shape is invalid: roots={sampled_roots} "
        f"points={sampled_points} samples={samples}"
    )
if not math.isfinite(float(collision_metric.get("loss", float("nan")))):
    raise RuntimeError("R062 collision loss is non-finite")

gradient_norms = diagnostic["parameter_gradient_norms"]
required_gradients = (
    "bary_logits",
    "guide_length_raw",
    "guide_direction_local_raw",
    "secondary_geometry_residual.length_raw",
    "secondary_geometry_residual.direction_local_raw",
)
for name in required_gradients:
    value = float(gradient_norms.get(name, {}).get("l2", 0.0))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"R062 collision gradient did not reach {name}: {value}")
for forbidden in ("translation", "log_scale"):
    if forbidden in gradient_norms:
        raise RuntimeError(f"R062 collision incorrectly reached {forbidden}")

baseline_preflight_rows = [
    row
    for row in metric_rows(baseline_preflight_metrics_path)
    if "train" in row
]
if not baseline_preflight_rows:
    raise RuntimeError("R061 preflight metrics contain no train rows")
baseline_preflight = baseline_preflight_rows[-1]
baseline_collision = json.loads(
    Path(baseline_collision_path).read_text(encoding="utf-8")
)

acceptance = {
    "mode": mode,
    "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
    "sdf_sha256": expected_sdf_sha,
    "sampled_root_count": sampled_roots,
    "sampled_point_count": sampled_points,
    "sampled_penetrating_fraction": float(
        collision_metric.get("penetrating_fraction", 0.0)
    ),
    "sampled_mean_depth": float(collision_metric.get("mean_depth", 0.0)),
    "sampled_maximum_depth": float(collision_metric.get("maximum_depth", 0.0)),
    "all_root": {
        "root_count": int(diagnostic["root_count"]),
        "penetrating_root_count": int(diagnostic["penetrating_root_count"]),
        "penetrating_root_fraction": float(diagnostic["penetrating_root_fraction"]),
        "penetrating_point_fraction": float(diagnostic["penetrating_point_fraction"]),
        "mean_dimensionless_depth": float(diagnostic["mean_dimensionless_depth"]),
        "maximum_dimensionless_depth": float(diagnostic["maximum_dimensionless_depth"]),
    },
    "gradient_l2": {
        name: float(gradient_norms[name]["l2"])
        for name in required_gradients
    },
    "global_calibration_gradients_absent": True,
    "fullres_elapsed_sec": float(last["elapsed_sec"]),
    "fullres_peak_allocated_mb": float(last["max_memory_mb"]),
    "r061_preflight_elapsed_sec": float(baseline_preflight["elapsed_sec"]),
    "r061_preflight_peak_allocated_mb": float(baseline_preflight["max_memory_mb"]),
    "preflight_added_elapsed_sec": float(last["elapsed_sec"])
    - float(baseline_preflight["elapsed_sec"]),
    "preflight_added_peak_allocated_mb": float(last["max_memory_mb"])
    - float(baseline_preflight["max_memory_mb"]),
    "r061_all_root": {
        "root_count": int(baseline_collision["root_count"]),
        "penetrating_root_count": int(baseline_collision["penetrating_root_count"]),
        "penetrating_root_fraction": float(baseline_collision["penetrating_root_fraction"]),
        "penetrating_point_fraction": float(baseline_collision["penetrating_point_fraction"]),
        "mean_dimensionless_depth": float(baseline_collision["mean_dimensionless_depth"]),
        "maximum_dimensionless_depth": float(baseline_collision["maximum_dimensionless_depth"]),
    },
}
Path(output_path).write_text(
    json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(acceptance, indent=2, sort_keys=True))
print(f"R062_{mode.upper()}_CHECK_PASS")
PY

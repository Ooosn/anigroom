#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
ulimit -v unlimited || true

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the reviewed source checkout}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT}"
: "${RUNTIME_ROOT:?Set RUNTIME_ROOT to a new R074 runtime directory}"
: "${CLEAN_FLOW_TARGET:?Set CLEAN_FLOW_TARGET to the formal Panda V8 target}"
: "${EXPECTED_TARGET_SHA256:?Set EXPECTED_TARGET_SHA256}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name the granted GPU}"

PYTHON_PATH="${PYTHON_PATH:-/home/wangyy/miniconda3/envs/mygs/bin/python}"
PANDA_DATA_ROOT="${PANDA_DATA_ROOT:-/home/wangyy/panda-r068-v5-assets-20260827/data}"
PANDA_MESH_PATH="${PANDA_MESH_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/mesh/furless.obj}"
PANDA_SDF_PATH="${PANDA_SDF_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/collision/panda_sdf_long512.npz}"
CONFIG_PATH="$PROJECT_ROOT/configs/r074_v8_confidence_flow_0_3k_gate.env"
CLEAN_FLOW_SUMMARY="$(dirname "$CLEAN_FLOW_TARGET")/summary.json"
OUTPUT_DIR="$RUNTIME_ROOT/outputs/panda_r074_v8_flow_0_3k_h100_20260830"
PREFLIGHT_OUTPUT="$RUNTIME_ROOT/preflight/fullres_view09"
RENDER_ROOT="$RUNTIME_ROOT/renders/iter_003000"

fail() {
  echo "[r074] $*" >&2
  exit 2
}

[[ ! -e "$RUNTIME_ROOT" ]] || fail "refusing existing runtime: $RUNTIME_ROOT"
for path in \
  "$PYTHON_PATH" \
  "$CONFIG_PATH" \
  "$CLEAN_FLOW_TARGET" \
  "$CLEAN_FLOW_SUMMARY" \
  "$PANDA_MESH_PATH" \
  "$PANDA_SDF_PATH" \
  "$PROJECT_ROOT/scripts/server/run_white_tiger_stage1.sh"; do
  [[ -f "$path" ]] || fail "missing required file: $path"
done
[[ -d "$PANDA_DATA_ROOT" ]] || fail "missing Panda data root: $PANDA_DATA_ROOT"

actual_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
[[ "$actual_commit" == "$EXPECTED_SOURCE_COMMIT" ]] || \
  fail "source commit mismatch: expected=$EXPECTED_SOURCE_COMMIT actual=$actual_commit"
[[ -z "$(git -C "$PROJECT_ROOT" status --short)" ]] || \
  fail "source checkout is dirty: $PROJECT_ROOT"
actual_target_sha256="$(sha256sum "$CLEAN_FLOW_TARGET" | awk '{print $1}')"
[[ "$actual_target_sha256" == "$EXPECTED_TARGET_SHA256" ]] || \
  fail "target hash mismatch: expected=$EXPECTED_TARGET_SHA256 actual=$actual_target_sha256"
bash -n "$PROJECT_ROOT/scripts/server/run_white_tiger_stage1.sh"

mkdir -p \
  "$RUNTIME_ROOT/contracts" \
  "$RUNTIME_ROOT/logs" \
  "$RUNTIME_ROOT/pids" \
  "$RUNTIME_ROOT/preflight" \
  "$RUNTIME_ROOT/renders"
printf '%s\n' "$$" > "$RUNTIME_ROOT/pids/r074.pid"

echo "R074_START $(date -Is) HOST=$(hostname) GPU=$CUDA_VISIBLE_DEVICES"
echo "ULIMIT_V=$(ulimit -v) SOURCE_COMMIT=$actual_commit"
echo "TARGET=$CLEAN_FLOW_TARGET TARGET_SHA256=$actual_target_sha256"
echo "DATA_ROOT=$PANDA_DATA_ROOT MESH=$PANDA_MESH_PATH SDF=$PANDA_SDF_PATH"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

"$PYTHON_PATH" -B - "$CLEAN_FLOW_TARGET" "$CLEAN_FLOW_SUMMARY" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

target_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
target = np.load(target_path, allow_pickle=False)
required = {
    "cleaned_directed_flow3d",
    "axis_view_cluster_confidence_flow_changed",
    "axis_view_cluster_confidence_flow_final_edge_dot",
    "axis_view_cluster_confidence_flow_new_severe_edge",
}
missing = sorted(required - set(target.files))
if missing:
    raise RuntimeError(f"formal V8 target is missing {missing}")
if np.any(target["axis_view_cluster_confidence_flow_new_severe_edge"]):
    raise RuntimeError("formal V8 target contains a newly severe edge")
for key in target.files:
    value = target[key]
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise RuntimeError(f"non-finite target array: {key}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
report = summary["confidence_guided_directed_flow"]
if report.get("enabled") != 1:
    raise RuntimeError("confidence-guided target report is not enabled")
if not report["zero_new_severe_verification"]["passed"]:
    raise RuntimeError("confidence-guided target verification failed")
print(json.dumps({
    "observed_roots": int(target["observed"].sum()),
    "changed_roots": int(target["axis_view_cluster_confidence_flow_changed"].sum()),
    "observed_edge_counts": report["counts"]["observed_edges"],
    "all_edge_counts": report["counts"]["all_edges"],
}, sort_keys=True))
PY

sha256sum \
  "$CONFIG_PATH" \
  "$CLEAN_FLOW_TARGET" \
  "$CLEAN_FLOW_SUMMARY" \
  "$PANDA_MESH_PATH" \
  "$PANDA_SDF_PATH" \
  "$PROJECT_ROOT/tools/train_white_tiger_stage1.py" \
  "$PROJECT_ROOT/anigroom/grooming/view_gated_ownership.py" \
  > "$RUNTIME_ROOT/contracts/inputs.sha256"

echo "R074_TESTS_START $(date -Is)"
(
  cd "$PROJECT_ROOT"
  "$PYTHON_PATH" -B -m pytest -q
)
echo "R074_TESTS_DONE $(date -Is)"

run_stage1() {
  local output_dir="$1"
  shift
  PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHON="$PYTHON_PATH" \
  CONFIG_PATH="$CONFIG_PATH" \
  DATA_ROOT="$PANDA_DATA_ROOT" \
  MESH_PATH="$PANDA_MESH_PATH" \
  CLEAN_FLOW_TARGET="$CLEAN_FLOW_TARGET" \
  MESH_NO_PENETRATION_SDF="$PANDA_SDF_PATH" \
  INIT_MESH_SCALE=1.0 \
  INIT_MESH_TRANSLATION=0,0,0 \
  OUTPUT_DIR="$output_dir" \
  "$@" \
  bash "$PROJECT_ROOT/scripts/server/run_white_tiger_stage1.sh"
}

echo "R074_PREFLIGHT_START $(date -Is)"
run_stage1 \
  "$RUNTIME_ROOT/preflight/main" \
  env \
  RUN_ID=panda_r074_v8_flow_fullres_preflight_20260830 \
  PREFLIGHT_OUTPUT_DIR="$PREFLIGHT_OUTPUT" \
  RUN_PREFLIGHT=1 \
  RUN_BATCH_PREFLIGHT=1 \
  PREFLIGHT_VIEW=9 \
  STAGE1_PREFLIGHT_ONLY=1

"$PYTHON_PATH" -B - "$PREFLIGHT_OUTPUT" "$CLEAN_FLOW_TARGET" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = sys.argv[2]
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
gate = json.loads((root / "view_gate_report.json").read_text(encoding="utf-8"))
clean = json.loads((root / "clean_flow_init_report.json").read_text(encoding="utf-8"))
records = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
]
metric = [record for record in records if record.get("iteration") == 1][-1]
assert config["view_gated_ownership_support"] is True
assert config["view_gate_floor"] == 0.0
assert config["view_gate_normalization"] == "equal_owner_budget"
assert gate["normalization_mode"] == "equal_owner_budget"
assert abs(gate["supported_guide_expected_multiplier_mean"] - 1.0) < 1.0e-6
assert gate["source_path"] == target
assert clean["clean_flow_source"] == target
assert metric["train"]["composite_psnr"] > 0.0
assert metric["max_memory_mb"] < 25000.0
print(json.dumps({
    "gate": gate,
    "clean_flow": clean,
    "train": metric["train"],
    "test": metric["test"],
    "max_memory_mb": metric["max_memory_mb"],
}, indent=2))
PY
printf '%s %s\n' "$actual_commit" "$actual_target_sha256" > "$RUNTIME_ROOT/PREFLIGHT_OK"
echo "R074_PREFLIGHT_DONE $(date -Is)"

echo "R074_TRAIN_START $(date -Is)"
run_stage1 \
  "$OUTPUT_DIR" \
  env \
  RUN_ID=panda_r074_v8_flow_0_3k_h100_20260830 \
  RUN_PREFLIGHT=0 \
  RUN_BATCH_PREFLIGHT=0
echo "R074_TRAIN_DONE $(date -Is)"

checkpoint="$OUTPUT_DIR/checkpoint_003000.pt"
[[ -f "$checkpoint" ]] || fail "missing final checkpoint: $checkpoint"
"$PYTHON_PATH" -B \
  "$PROJECT_ROOT/tools/render_white_tiger_stage1_checkpoint_views.py" \
  --checkpoint "$checkpoint" \
  --output-dir "$RENDER_ROOT" \
  --view-ids 9 \
  --device cuda

find "$RUNTIME_ROOT/renders" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$RUNTIME_ROOT/render_hashes.sha256"
sha256sum "$checkpoint" > "$RUNTIME_ROOT/checkpoint_hashes.sha256"
find "$OUTPUT_DIR" -maxdepth 1 -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$RUNTIME_ROOT/output_hashes.sha256"
echo "R074_DONE $(date -Is) CHECKPOINT=$checkpoint"

#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
ulimit -v unlimited || true

fail() {
  echo "[r080] $*" >&2
  exit 2
}

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the reviewed source checkout}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT}"
: "${RUNTIME_ROOT:?Set RUNTIME_ROOT to a new R080 runtime directory}"
: "${R074_CHECKPOINT:?Set R074_CHECKPOINT to the original R074 checkpoint at /home/wangyy/panda-r074-v8-runtime-20260830/outputs/panda_r074_v8_flow_0_3k_h100_20260830/checkpoint_003000.pt}"
: "${EXPECTED_R074_CHECKPOINT_SHA256:?Set EXPECTED_R074_CHECKPOINT_SHA256=fcd62694663a7ab9383ff0250fa6a44544b7bafff1ebc96ffd7a2e05ad8d013e}"
: "${CLEAN_FLOW_TARGET:?Set CLEAN_FLOW_TARGET to the formal Panda V8 target}"
: "${EXPECTED_TARGET_SHA256:?Set EXPECTED_TARGET_SHA256}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name the granted GPU}"

PYTHON_PATH="${PYTHON_PATH:-/home/wangyy/miniconda3/envs/mygs/bin/python}"
PANDA_DATA_ROOT="${PANDA_DATA_ROOT:-/home/wangyy/panda-r068-v5-assets-20260827/data}"
PANDA_MESH_PATH="${PANDA_MESH_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/mesh/furless.obj}"
PANDA_SDF_PATH="${PANDA_SDF_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/collision/panda_sdf_long512.npz}"
MIGRATION_UTILITY="${MIGRATION_UTILITY:-$PROJECT_ROOT/tools/migrate_stage1_schema12_checkpoint.py}"
CONFIG_PATH="$PROJECT_ROOT/configs/r080_all_guide_unlock_at3k_resume.env"
PREFLIGHT_CONFIG_PATH="$PROJECT_ROOT/configs/r080_all_guide_unlock_at3k_preflight.env"
CLEAN_FLOW_SUMMARY="$(dirname "$CLEAN_FLOW_TARGET")/summary.json"
MIGRATED_CHECKPOINT="$RUNTIME_ROOT/contracts/r080_migrated_checkpoint.pt"
MIGRATION_REPORT="$RUNTIME_ROOT/contracts/r080_migration_report.json"
PREFLIGHT_OUTPUT="$RUNTIME_ROOT/preflight/resume_3000_to_3001"
PREFLIGHT_LOG="$RUNTIME_ROOT/logs/resume_3000_to_3001.log"
OUTPUT_DIR="$RUNTIME_ROOT/outputs/panda_r080_all_guide_unlock_at3k_resume_4000_20260831"
FULL_LOG="$RUNTIME_ROOT/logs/resume_3000_to_4000.log"
RENDER_3000_ROOT="$RUNTIME_ROOT/renders/iter_003000_migrated"
RENDER_4000_ROOT="$RUNTIME_ROOT/renders/iter_004000"
EXPECTED_ORIGINAL_R074_CHECKPOINT_SHA256="fcd62694663a7ab9383ff0250fa6a44544b7bafff1ebc96ffd7a2e05ad8d013e"

EXPECTED_R074_CHECKPOINT_SHA256="$(printf '%s' "$EXPECTED_R074_CHECKPOINT_SHA256" | tr '[:upper:]' '[:lower:]')"
[[ "$EXPECTED_R074_CHECKPOINT_SHA256" == "$EXPECTED_ORIGINAL_R074_CHECKPOINT_SHA256" ]] || \
  fail "R074 checkpoint expected SHA must be $EXPECTED_ORIGINAL_R074_CHECKPOINT_SHA256"

[[ ! -e "$RUNTIME_ROOT" ]] || fail "refusing existing runtime: $RUNTIME_ROOT"
for path in \
  "$PYTHON_PATH" \
  "$CONFIG_PATH" \
  "$PREFLIGHT_CONFIG_PATH" \
  "$R074_CHECKPOINT" \
  "$MIGRATION_UTILITY" \
  "$CLEAN_FLOW_TARGET" \
  "$CLEAN_FLOW_SUMMARY" \
  "$PANDA_MESH_PATH" \
  "$PANDA_SDF_PATH" \
  "$PROJECT_ROOT/scripts/server/run_white_tiger_stage1.sh" \
  "$PROJECT_ROOT/tools/render_white_tiger_stage1_checkpoint_views.py"; do
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
actual_r074_checkpoint_sha256="$(sha256sum "$R074_CHECKPOINT" | awk '{print $1}')"
[[ "$actual_r074_checkpoint_sha256" == "$EXPECTED_R074_CHECKPOINT_SHA256" ]] || \
  fail "R074 checkpoint hash mismatch: expected=$EXPECTED_R074_CHECKPOINT_SHA256 actual=$actual_r074_checkpoint_sha256"
bash -n "$PROJECT_ROOT/scripts/server/run_white_tiger_stage1.sh"

mkdir -p \
  "$RUNTIME_ROOT/contracts" \
  "$RUNTIME_ROOT/logs" \
  "$RUNTIME_ROOT/pids" \
  "$RUNTIME_ROOT/preflight" \
  "$RUNTIME_ROOT/renders" \
  "$RUNTIME_ROOT/outputs"
printf '%s\n' "$$" > "$RUNTIME_ROOT/pids/r080.pid"

echo "R080_START $(date -Is) HOST=$(hostname) GPU=$CUDA_VISIBLE_DEVICES"
echo "ULIMIT_V=$(ulimit -v) SOURCE_COMMIT=$actual_commit"
echo "R074_CHECKPOINT=$R074_CHECKPOINT R074_CHECKPOINT_SHA256=$actual_r074_checkpoint_sha256"
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

"$PYTHON_PATH" -B - "$R074_CHECKPOINT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
if int(checkpoint.get("checkpoint_version", -1)) != 12:
    raise RuntimeError("original R074 checkpoint must be schema12")
if int(checkpoint.get("iteration", -1)) != 3000:
    raise RuntimeError("original R074 checkpoint must be iteration3000")
root_count = int(checkpoint["model"]["face_ids"].shape[0])
if root_count != 480292:
    raise RuntimeError(f"original R074 checkpoint root count must be 480292, got {root_count}")
print({"checkpoint_version": 12, "iteration": 3000, "root_count": root_count})
PY

sha256sum \
  "$CONFIG_PATH" \
  "$PREFLIGHT_CONFIG_PATH" \
  "$R074_CHECKPOINT" \
  "$CLEAN_FLOW_TARGET" \
  "$CLEAN_FLOW_SUMMARY" \
  "$PANDA_MESH_PATH" \
  "$PANDA_SDF_PATH" \
  "$PROJECT_ROOT/tools/train_white_tiger_stage1.py" \
  "$PROJECT_ROOT/anigroom/grooming/view_gated_ownership.py" \
  "$MIGRATION_UTILITY" \
  > "$RUNTIME_ROOT/contracts/inputs.sha256"

echo "R080_TESTS_START $(date -Is)"
(
  cd "$PROJECT_ROOT"
  "$PYTHON_PATH" -B -m pytest -q
) 2>&1 | tee "$RUNTIME_ROOT/logs/full_pytest.log"
echo "R080_TESTS_DONE $(date -Is)"

echo "R080_MIGRATION_START $(date -Is)"
"$PYTHON_PATH" -B "$MIGRATION_UTILITY" \
  --checkpoint "$R074_CHECKPOINT" \
  --output "$MIGRATED_CHECKPOINT" \
  --report "$MIGRATION_REPORT" \
  --expected-input-sha256 "$EXPECTED_R074_CHECKPOINT_SHA256"
echo "R080_MIGRATION_DONE $(date -Is)"
[[ -f "$MIGRATED_CHECKPOINT" ]] || fail "migration did not create $MIGRATED_CHECKPOINT"
[[ -f "$MIGRATION_REPORT" ]] || fail "migration did not create $MIGRATION_REPORT"
migrated_checkpoint_sha256="$(sha256sum "$MIGRATED_CHECKPOINT" | awk '{print $1}')"

"$PYTHON_PATH" -B - \
  "$R074_CHECKPOINT" \
  "$MIGRATED_CHECKPOINT" \
  "$MIGRATION_REPORT" \
  "$actual_r074_checkpoint_sha256" \
  "$migrated_checkpoint_sha256" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

source_path = Path(sys.argv[1]).resolve()
migrated_path = Path(sys.argv[2]).resolve()
report_path = Path(sys.argv[3])
source_sha256 = sys.argv[4]
migrated_sha256 = sys.argv[5]


def load(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    return checkpoint


def equal(left: Any, right: Any, path: str) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            raise AssertionError(f"tensor identity type mismatch at {path}")
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise AssertionError(f"tensor identity shape/dtype mismatch at {path}")
        if not torch.equal(left, right):
            raise AssertionError(f"tensor identity mismatch at {path}")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            raise AssertionError(f"array identity type mismatch at {path}")
        if not np.array_equal(left, right):
            raise AssertionError(f"array identity mismatch at {path}")
        return
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict)):
            raise AssertionError(f"mapping identity type mismatch at {path}")
        if set(left) != set(right):
            raise AssertionError(f"mapping identity keys mismatch at {path}")
        for key in left:
            equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not (isinstance(left, type(right)) and len(left) == len(right)):
            raise AssertionError(f"sequence identity mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            equal(left_item, right_item, f"{path}[{index}]")
        return
    if isinstance(left, float) or isinstance(right, float):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.0):
            raise AssertionError(f"scalar identity mismatch at {path}")
        return
    if left != right:
        raise AssertionError(f"scalar identity mismatch at {path}")


def report_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("passed", "verified", "equal", "identical"):
            if key in value:
                return bool(value[key])
        flags = [report_flag(item) for item in value.values()]
        return bool(flags) and all(flags)
    return bool(value)


source = load(source_path)
migrated = load(migrated_path)
for key in ("model", "optimizer", "optimizer_param_names", "rng_state", "lifecycle_history"):
    if key not in source or key not in migrated:
        raise AssertionError(f"migration omitted identity section {key}")
    equal(source[key], migrated[key], key)
if int(source["iteration"]) != 3000 or int(migrated["iteration"]) != 3000:
    raise AssertionError("migration must preserve iteration3000")
if int(source["checkpoint_version"]) != 12 or int(migrated["checkpoint_version"]) != 14:
    raise AssertionError("migration must convert schema12 to schema14")

report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("status") != "pass":
    raise AssertionError("migration report status is not pass")
report_source_sha256 = report.get("source_checkpoint_sha256", report.get("source_sha256"))
report_migrated_sha256 = report.get(
    "migrated_checkpoint_sha256",
    report.get("output_checkpoint_sha256", report.get("output_sha256")),
)
if report_source_sha256 != source_sha256:
    raise AssertionError("migration report source hash mismatch")
if report_migrated_sha256 != migrated_sha256:
    raise AssertionError("migration report migrated hash mismatch")
if int(report.get("source_iteration", report.get("input_iteration", -1))) != 3000:
    raise AssertionError("migration report source iteration mismatch")
report_output_iteration = report.get(
    "migrated_iteration",
    report.get("output_iteration", report.get("target_iteration", -1)),
)
if int(report_output_iteration) != 3000:
    raise AssertionError("migration report output iteration mismatch")
identity_report = report.get("tensor_identity", report.get("tensor_identity_report"))
if identity_report is None or not report_flag(identity_report):
    raise AssertionError("migration report tensor identity did not pass")
for identity_name in (
    "model",
    "optimizer",
    "individual_hashes",
    "aggregate_hashes",
    "object_key_shape_manifests",
):
    assert identity_report[identity_name] is True
assert report["tensor_integrity_checks"]["source_to_migrated_identical"] is True
assert report["tensor_integrity_checks"]["migrated_to_output_identical"] is True
defaults = report["schema14_defaults"]
assert defaults["view_gate_geometry_support"] is False
assert defaults["view_gate_length_confidence_support"] is False
assert float(defaults["clean_flow_guide_length_anchor_weight"]) == 0.0
assert defaults["clean_flow_guide_length_anchor_reduction"] == "mean_l1"


def optimizer_state(checkpoint: dict[str, Any], parameter_name: str) -> dict[str, Any]:
    names = checkpoint["optimizer_param_names"]
    groups = checkpoint["optimizer"]["param_groups"]
    states = checkpoint["optimizer"]["state"]
    for group_names, group in zip(names, groups):
        for name, parameter_id in zip(group_names, group["params"]):
            if name == parameter_name:
                return states.get(parameter_id, states.get(str(parameter_id)))
    raise AssertionError(f"missing optimizer parameter {parameter_name}")


guide_state = optimizer_state(migrated, "guide_length_raw")
if guide_state is None:
    raise AssertionError("guide_length_raw has no Adam state")
assert float(guide_state["step"]) == 3000.0
for moment_name in ("exp_avg", "exp_avg_sq"):
    moment = guide_state[moment_name]
    assert torch.equal(moment, torch.zeros_like(moment))
guide_report = report["guide_length_raw_optimizer"]["source"]
assert guide_report["step"]["value"] == 3000.0
assert guide_report["moment_nonzero_counts"] == {
    "exp_avg": 0,
    "exp_avg_sq": 0,
}
print(json.dumps({
    "source_iteration": int(source["iteration"]),
    "migrated_iteration": int(migrated["iteration"]),
    "source_sha256": source_sha256,
    "migrated_sha256": migrated_sha256,
    "tensor_identity": True,
    "guide_length_adam_step": float(guide_state["step"]),
    "guide_length_adam_moments_zero": True,
}, sort_keys=True))
PY

sha256sum "$R074_CHECKPOINT" "$MIGRATED_CHECKPOINT" > "$RUNTIME_ROOT/contracts/migration_hashes.sha256"

echo "R080_RENDER_MIGRATED_3000_START $(date -Is)"
"$PYTHON_PATH" -B \
  "$PROJECT_ROOT/tools/render_white_tiger_stage1_checkpoint_views.py" \
  --checkpoint "$MIGRATED_CHECKPOINT" \
  --output-dir "$RENDER_3000_ROOT" \
  --view-ids 9 \
  --device cuda 2>&1 | tee "$RUNTIME_ROOT/logs/render_migrated_003000.log"
echo "R080_RENDER_MIGRATED_3000_DONE $(date -Is)"

run_stage1() {
  local config_path="$1"
  local output_dir="$2"
  local log_path="$3"
  shift 3
  PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHON="$PYTHON_PATH" \
  CONFIG_PATH="$config_path" \
  DATA_ROOT="$PANDA_DATA_ROOT" \
  MESH_PATH="$PANDA_MESH_PATH" \
  CLEAN_FLOW_TARGET="$CLEAN_FLOW_TARGET" \
  MESH_NO_PENETRATION_SDF="$PANDA_SDF_PATH" \
  INIT_MESH_SCALE=1.0 \
  INIT_MESH_TRANSLATION=0,0,0 \
  VIEW_GATE_GEOMETRY_SUPPORT=0 \
  VIEW_GATE_LENGTH_CONFIDENCE_SUPPORT=0 \
  CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT=0.0 \
  CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_REDUCTION=mean_l1 \
  OUTPUT_DIR="$output_dir" \
  "$@" \
  bash "$PROJECT_ROOT/scripts/server/run_white_tiger_stage1.sh" 2>&1 | tee "$log_path"
}

echo "R080_PREFLIGHT_START $(date -Is)"
run_stage1 \
  "$PREFLIGHT_CONFIG_PATH" \
  "$PREFLIGHT_OUTPUT" \
  "$PREFLIGHT_LOG" \
  env \
  RUN_ID=panda_r080_all_guide_unlock_at3k_resume_preflight_20260831 \
  RESUME_CHECKPOINT="$MIGRATED_CHECKPOINT" \
  RESUME_OPTIMIZER=1 \
  RUN_PREFLIGHT=0 \
  RUN_BATCH_PREFLIGHT=0

preflight_checkpoint="$PREFLIGHT_OUTPUT/checkpoint_003001.pt"
[[ -f "$preflight_checkpoint" ]] || fail "missing one-step preflight checkpoint: $preflight_checkpoint"
"$PYTHON_PATH" -B - \
  "$PREFLIGHT_OUTPUT" \
  "$PREFLIGHT_LOG" \
  "$MIGRATED_CHECKPOINT" \
  "$preflight_checkpoint" \
  "$CLEAN_FLOW_TARGET" \
  "$actual_target_sha256" \
  "$PROJECT_ROOT" \
  <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

root = Path(sys.argv[1])
log_path = Path(sys.argv[2])
migrated_path = Path(sys.argv[3])
preflight_checkpoint_path = Path(sys.argv[4])
target = sys.argv[5]
target_path = Path(target).resolve()
expected_target_sha256 = sys.argv[6]
project_root = Path(sys.argv[7]).resolve()
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
gate = json.loads((root / "view_gate_report.json").read_text(encoding="utf-8"))
root_report = json.loads((root / "root_init_report.json").read_text(encoding="utf-8"))
records = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
]
metric = [record for record in records if record.get("iteration") == 3001][-1]
events: list[dict[str, Any]] = []
for line in log_path.read_text(encoding="utf-8").splitlines():
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict):
        events.append(value)


def event(name: str) -> dict[str, Any]:
    matches = [
        value
        for value in events
        if value.get("progress") == name or value.get("setup_progress") == name
    ]
    if not matches:
        raise AssertionError(f"missing progress event {name}")
    return matches[-1]


setup = event("setup_complete")
optimizer_resume = event("optimizer_resume_done")
rng_resume = event("rng_resume_done")
assert setup["start_iteration"] == 3000
assert setup["root_count"] == 480292
assert optimizer_resume["checkpoint_iteration"] == 3000
assert rng_resume["checkpoint_iteration"] == 3000
assert root_report["root_count"] == 480292
assert hashlib.sha256(target_path.read_bytes()).hexdigest() == expected_target_sha256
config_target = Path(str(config["clean_flow_target"]))
if not config_target.is_absolute():
    config_target = project_root / config_target
assert config_target.resolve() == target_path
assert gate["source_path"] == target
assert config["iterations"] == 3001
assert config["stage_save_iters"] == "3001"
assert config["save_every"] == 0
assert config["resume_optimizer"] is True
assert Path(str(config["resume_checkpoint"])).resolve() == migrated_path.resolve()
assert config["guide_length_freeze_until"] == 3000
assert config["guide_freeze_until"] == 3000
assert config["shape_detail_freeze_until"] == 20000
assert config["shape_detail_unlock_end"] == 25000
assert config["view_gated_ownership_support"] is True
assert config["view_gate_normalization"] == "equal_owner_budget"
assert config["view_gate_geometry_support"] is False
assert config["view_gate_length_confidence_support"] is False
assert float(config["clean_flow_guide_length_anchor_weight"]) == 0.0
assert config["clean_flow_guide_length_anchor_reduction"] == "mean_l1"
assert config["guide_support_gauge_weight"] == 0.0
assert config["guide_view_sh_support"] is False
assert gate["view_gate_geometry_support"] == 0
assert gate["view_gate_length_confidence_support"] == 0
assert metric["guide_length_frozen"] is False
assert metric["guide_frozen"] is False
assert metric["shape_detail_frozen"] is True
assert float(metric["shape_detail_multiplier"]) == 0.0
assert float(metric["gaussian_rgb_residual_multiplier"]) == 0.0
assert float(metric["clean_flow_guide_length_anchor_loss"]) == 0.0
length = metric["effective_groom"]["length"]
for key in ("min", "p05", "p50", "p95", "max", "mean"):
    value = float(length[key])
    assert math.isfinite(value) and value > 0.0, (key, value)
try:
    preflight_checkpoint = torch.load(
        preflight_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    preflight_checkpoint = torch.load(preflight_checkpoint_path, map_location="cpu")
assert int(preflight_checkpoint["iteration"]) == 3001
assert int(preflight_checkpoint["model"]["face_ids"].shape[0]) == 480292
print(json.dumps({
    "start_checkpoint_iteration": setup["start_iteration"],
    "preflight_iteration": int(preflight_checkpoint["iteration"]),
    "root_count": setup["root_count"],
    "guide_length_frozen": metric["guide_length_frozen"],
    "guide_frozen": metric["guide_frozen"],
    "new_features_off": True,
    "effective_length": length,
}, sort_keys=True))
PY
echo "R080_PREFLIGHT_DONE $(date -Is)"

migrated_checkpoint_sha256_after_preflight="$(sha256sum "$MIGRATED_CHECKPOINT" | awk '{print $1}')"
[[ "$migrated_checkpoint_sha256_after_preflight" == "$migrated_checkpoint_sha256" ]] || \
  fail "preflight modified the migrated checkpoint"

echo "R080_TRAIN_START $(date -Is)"
run_stage1 \
  "$CONFIG_PATH" \
  "$OUTPUT_DIR" \
  "$FULL_LOG" \
  env \
  RUN_ID=panda_r080_all_guide_unlock_at3k_resume_4000_20260831 \
  RESUME_CHECKPOINT="$MIGRATED_CHECKPOINT" \
  RESUME_OPTIMIZER=1 \
  RUN_PREFLIGHT=0 \
  RUN_BATCH_PREFLIGHT=0
echo "R080_TRAIN_DONE $(date -Is)"

checkpoint="$OUTPUT_DIR/checkpoint_004000.pt"
[[ -f "$checkpoint" ]] || fail "missing final checkpoint: $checkpoint"
"$PYTHON_PATH" -B - \
  "$OUTPUT_DIR" \
  "$FULL_LOG" \
  "$MIGRATED_CHECKPOINT" \
  "$checkpoint" \
  <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

root = Path(sys.argv[1])
log_path = Path(sys.argv[2])
migrated_path = Path(sys.argv[3])
checkpoint_path = Path(sys.argv[4])
config = json.loads((root / "config.json").read_text(encoding="utf-8"))
records = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
]
metric = [record for record in records if record.get("iteration") == 4000][-1]
events: list[dict[str, Any]] = []
for line in log_path.read_text(encoding="utf-8").splitlines():
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(value, dict):
        events.append(value)


def event(name: str) -> dict[str, Any]:
    matches = [
        value
        for value in events
        if value.get("progress") == name or value.get("setup_progress") == name
    ]
    if not matches:
        raise AssertionError(f"missing progress event {name}")
    return matches[-1]


setup = event("setup_complete")
optimizer_resume = event("optimizer_resume_done")
rng_resume = event("rng_resume_done")
assert setup["start_iteration"] == 3000
assert setup["root_count"] == 480292
assert setup["lifecycle_statistics_active"] is True
assert optimizer_resume["checkpoint_iteration"] == 3000
assert rng_resume["checkpoint_iteration"] == 3000
assert config["iterations"] == 4000
assert config["stage_save_iters"] == "4000"
assert config["resume_optimizer"] is True
assert Path(str(config["resume_checkpoint"])).resolve() == migrated_path.resolve()
assert config["guide_length_freeze_until"] == 3000
assert config["guide_freeze_until"] == 3000
assert config["shape_detail_freeze_until"] == 20000
assert config["shape_detail_unlock_end"] == 25000
assert config["view_gate_geometry_support"] is False
assert config["view_gate_length_confidence_support"] is False
assert float(config["clean_flow_guide_length_anchor_weight"]) == 0.0
assert config["clean_flow_guide_length_anchor_reduction"] == "mean_l1"
assert metric["iteration"] == 4000
assert metric["guide_length_frozen"] is False
assert metric["guide_frozen"] is False
assert metric["shape_detail_frozen"] is True
assert float(metric["shape_detail_multiplier"]) == 0.0
assert float(metric["gaussian_rgb_residual_multiplier"]) == 0.0
assert "train" in metric
assert "test" in metric
try:
    source_checkpoint = torch.load(migrated_path, map_location="cpu", weights_only=False)
    final_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
except TypeError:
    source_checkpoint = torch.load(migrated_path, map_location="cpu")
    final_checkpoint = torch.load(checkpoint_path, map_location="cpu")
assert int(final_checkpoint["iteration"]) == 4000
assert "optimizer" in final_checkpoint
assert "optimizer_param_names" in final_checkpoint
assert "rng_state" in final_checkpoint
assert "lifecycle_history" in final_checkpoint
source_history = source_checkpoint.get("lifecycle_history", [])
final_history = final_checkpoint["lifecycle_history"]
assert final_history[: len(source_history)] == source_history
assert final_checkpoint["optimizer_param_names"] == source_checkpoint["optimizer_param_names"]
names = final_checkpoint["optimizer_param_names"]
groups = final_checkpoint["optimizer"]["param_groups"]
states = final_checkpoint["optimizer"]["state"]
guide_state = None
for group_names, group in zip(names, groups):
    for name, parameter_id in zip(group_names, group["params"]):
        if name == "guide_length_raw":
            guide_state = states.get(parameter_id, states.get(str(parameter_id)))
            break
assert guide_state is not None
assert float(guide_state["step"]) == 4000.0
print(json.dumps({
    "start_checkpoint_iteration": setup["start_iteration"],
    "final_iteration": int(final_checkpoint["iteration"]),
    "root_count_at_setup": setup["root_count"],
    "guide_length_frozen": metric["guide_length_frozen"],
    "guide_frozen": metric["guide_frozen"],
    "optimizer_step_continued_to": float(guide_state["step"]),
    "rng_restored": True,
    "lifecycle_prefix_preserved": True,
}, sort_keys=True))
PY

echo "R080_RENDER_4000_START $(date -Is)"
"$PYTHON_PATH" -B \
  "$PROJECT_ROOT/tools/render_white_tiger_stage1_checkpoint_views.py" \
  --checkpoint "$checkpoint" \
  --output-dir "$RENDER_4000_ROOT" \
  --view-ids 9 \
  --device cuda 2>&1 | tee "$RUNTIME_ROOT/logs/render_004000.log"
echo "R080_RENDER_4000_DONE $(date -Is)"

"$PYTHON_PATH" -B - "$RENDER_3000_ROOT" 3000 "$RENDER_4000_ROOT" 4000 <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

for root_text, expected_iteration in (
    (sys.argv[1], int(sys.argv[2])),
    (sys.argv[3], int(sys.argv[4])),
):
    root = Path(root_text)
    report = json.loads((root / "render_report.json").read_text(encoding="utf-8"))
    assert report["iteration"] == expected_iteration
    assert report["view_ids"] == [9]
    assert (root / "view_09_pred.png").is_file()
    assert (root / "view_09_gt.png").is_file()
print(json.dumps({"migrated_iteration": 3000, "output_iteration": 4000, "view_ids": [9]}))
PY

migrated_checkpoint_sha256_after_full="$(sha256sum "$MIGRATED_CHECKPOINT" | awk '{print $1}')"
[[ "$migrated_checkpoint_sha256_after_full" == "$migrated_checkpoint_sha256" ]] || \
  fail "full resume modified the migrated checkpoint"

find "$RUNTIME_ROOT/renders" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$RUNTIME_ROOT/render_hashes.sha256"
sha256sum \
  "$R074_CHECKPOINT" \
  "$MIGRATED_CHECKPOINT" \
  "$checkpoint" \
  > "$RUNTIME_ROOT/checkpoint_hashes.sha256"
find "$OUTPUT_DIR" -maxdepth 1 -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$RUNTIME_ROOT/output_hashes.sha256"
find "$PREFLIGHT_OUTPUT" -maxdepth 1 -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$RUNTIME_ROOT/preflight_hashes.sha256"
printf '%s %s %s %s\n' \
  "$actual_commit" \
  "$actual_target_sha256" \
  "$actual_r074_checkpoint_sha256" \
  "$migrated_checkpoint_sha256" \
  > "$RUNTIME_ROOT/PREFLIGHT_OK"
echo "R080_DONE $(date -Is) CHECKPOINT=$checkpoint"

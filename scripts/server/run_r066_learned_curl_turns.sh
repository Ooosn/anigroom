#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the reviewed R066 checkout}"
RUNTIME_ROOT="${RUNTIME_ROOT:?set RUNTIME_ROOT to a new R066 runtime directory}"
PYTHON="${PYTHON:?set PYTHON to the verified mygs interpreter}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the frozen white-tiger images}"
MESH_PATH="${MESH_PATH:?set MESH_PATH to the frozen aligned mesh}"
MESH_NO_PENETRATION_SDF="${MESH_NO_PENETRATION_SDF:?set the reviewed mesh SDF path}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the reviewed R066 commit}"
RUN_ID="${RUN_ID:?set RUN_ID to a fresh R066 run id}"

R065_CONFIG_PATH="$PROJECT_ROOT/configs/r065_local_crossing_residual_0_30k.env"
R066_CONFIG_PATH="$PROJECT_ROOT/configs/r066_learned_curl_turns_0_30k.env"
TRAINER_PATH="$PROJECT_ROOT/tools/train_white_tiger_stage1.py"
OUTPUT_DIR="$RUNTIME_ROOT/outputs/$RUN_ID"
LOG_DIR="$RUNTIME_ROOT/logs"
CONTRACT_DIR="$RUNTIME_ROOT/contracts"

fail() {
  echo "[r066] $*" >&2
  exit 2
}

cd "$PROJECT_ROOT"

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "EXPECTED_COMMIT must be a full 40-character commit"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  fail "commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit"
fi
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "[r066] checkout is dirty; refusing formal execution" >&2
  git status --short >&2
  exit 2
fi

[[ ! -e "$RUNTIME_ROOT" ]] || fail "runtime root already exists; refusing reuse: $RUNTIME_ROOT"
[[ ! -e "$OUTPUT_DIR" ]] || fail "output already exists; refusing overwrite: $OUTPUT_DIR"
[[ -x "$PYTHON" ]] || fail "mygs interpreter is not executable: $PYTHON"
[[ -d "$DATA_ROOT" ]] || fail "data root does not exist: $DATA_ROOT"
[[ -f "$MESH_PATH" ]] || fail "mesh does not exist: $MESH_PATH"
[[ -f "$MESH_NO_PENETRATION_SDF" ]] || fail "mesh SDF does not exist: $MESH_NO_PENETRATION_SDF"

python_prefix="$("$PYTHON" -c 'import sys; print(sys.prefix)')"
case "$python_prefix" in
  */mygs|*/mygs/*) ;;
  *) fail "PYTHON is not from the mygs environment: $python_prefix" ;;
esac

grep -Eq '^CURRENT_CHECKPOINT_VERSION[[:space:]]*=[[:space:]]*8[[:space:]]*$' "$TRAINER_PATH" \
  || fail "reviewed trainer does not declare checkpoint schema 8"
if grep -Eq '^CURRENT_CHECKPOINT_VERSION[[:space:]]*=[[:space:]]*7[[:space:]]*$' "$TRAINER_PATH"; then
  fail "schema-7 trainer is not allowed for R066"
fi

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  fail "RESUME_CHECKPOINT is forbidden; R066 must start from zero"
fi
if [[ "${RESUME_OPTIMIZER:-1}" == "0" ]]; then
  fail "RESUME_OPTIMIZER=0 is forbidden; R066 has no resume path"
fi
if [[ "${STAGE1_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  fail "STAGE1_PREFLIGHT_ONLY=1 is forbidden; R066 requires formal training"
fi
if [[ "${RUN_PREFLIGHT:-1}" == "0" ]]; then
  fail "RUN_PREFLIGHT=0 is forbidden; R066 requires the full data preflight"
fi
if [[ "${RUN_BATCH_PREFLIGHT:-0}" != "0" ]]; then
  fail "reduced batch preflight is forbidden for R066"
fi

mkdir -p "$LOG_DIR" "$CONTRACT_DIR"

snapshot_config() {
  local config_path="$1"
  local output_path="$2"
  env -i \
    HOME="$HOME" \
    PATH="$PATH" \
    MESH_NO_PENETRATION_SDF="$MESH_NO_PENETRATION_SDF" \
    bash -c 'set -a; source "$1"; env' _ "$config_path" \
    | LC_ALL=C sort > "$output_path"
}

snapshot_config "$R065_CONFIG_PATH" "$CONTRACT_DIR/r065_0_30k.env"
snapshot_config "$R066_CONFIG_PATH" "$CONTRACT_DIR/r066_0_30k.env"

"$PYTHON" - \
  "$CONTRACT_DIR/r065_0_30k.env" \
  "$CONTRACT_DIR/r066_0_30k.env" \
  "$CONTRACT_DIR/config_delta.json" <<'PY'
import json
from pathlib import Path
import sys


ignored = {"PWD", "SHLVL", "_"}


def load(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key not in ignored:
            result[key] = value
    return result


base = load(sys.argv[1])
candidate = load(sys.argv[2])
keys = set(base) | set(candidate)
delta = {
    key: {"r065": base.get(key), "r066": candidate.get(key)}
    for key in sorted(keys)
    if base.get(key) != candidate.get(key)
}
if delta != {}:
    raise RuntimeError(f"R066 config delta is not exactly empty: {delta}")
for key, expected in {
    "EXPECTED_WIDTH": "1920",
    "EXPECTED_HEIGHT": "1080",
    "ITERATIONS": "30000",
}.items():
    if candidate.get(key) != expected:
        raise RuntimeError(
            f"R066 requires {key}={expected}, got {candidate.get(key)!r}"
        )
Path(sys.argv[3]).write_text("{}\n", encoding="utf-8")
print("R066_CONFIG_DELTA_PASS")
print("R066_FULL_RESOLUTION_0_30K_CONFIG_PASS")
PY

ulimit -v unlimited
[[ "$(ulimit -v)" == "unlimited" ]] || fail "virtual-memory limit is not unlimited"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "[r066] host=$(hostname) commit=$actual_commit ulimit_v=$(ulimit -v)"
"$PYTHON" -m pytest -q

echo "[r066] start formal from-zero 0-30k at=$(date --iso-8601=seconds)"
env \
  -u RESUME_CHECKPOINT \
  -u RESUME_OPTIMIZER \
  -u STAGE1_PREFLIGHT_ONLY \
  -u TRAIN_VIEWS \
  -u TEST_VIEWS \
  PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHON="$PYTHON" \
  DATA_ROOT="$DATA_ROOT" \
  MESH_PATH="$MESH_PATH" \
  MESH_NO_PENETRATION_SDF="$MESH_NO_PENETRATION_SDF" \
  RUN_ID="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  CONFIG_PATH="$R066_CONFIG_PATH" \
  RUN_PREFLIGHT=1 \
  RUN_BATCH_PREFLIGHT=0 \
    bash "$PROJECT_ROOT/scripts/server/run_white_tiger_stage1.sh" \
    2>&1 | tee "$LOG_DIR/$RUN_ID.log"

FINAL_CHECKPOINT="$OUTPUT_DIR/checkpoint_030000.pt"
[[ -s "$FINAL_CHECKPOINT" ]] || fail "missing final 30k checkpoint: $FINAL_CHECKPOINT"
"$PYTHON" - "$FINAL_CHECKPOINT" <<'PY'
import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
version = int(checkpoint.get("checkpoint_version", -1))
if version != 8:
    raise RuntimeError(f"R066 requires checkpoint schema 8, got {version}")
if int(checkpoint.get("iteration", -1)) != 30000:
    raise RuntimeError(
        f"R066 requires a from-zero 30k checkpoint, got iteration "
        f"{checkpoint.get('iteration')!r}"
    )
if checkpoint.get("checkpoint_kind") != "stage1_full":
    raise RuntimeError(
        f"R066 requires checkpoint_kind=stage1_full, got "
        f"{checkpoint.get('checkpoint_kind')!r}"
    )
print("R066_SCHEMA8_FROM_ZERO_30K_PASS")
PY
echo "[r066] formal run complete at=$(date --iso-8601=seconds)"

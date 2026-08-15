#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the reviewed R065 checkout}"
RUNTIME_ROOT="${RUNTIME_ROOT:?set RUNTIME_ROOT to a new R065 runtime directory}"
PYTHON="${PYTHON:?set PYTHON to the verified mygs interpreter}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the frozen white-tiger images}"
MESH_PATH="${MESH_PATH:?set MESH_PATH to the frozen aligned mesh}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the reviewed R065 commit}"
MESH_NO_PENETRATION_SDF="${MESH_NO_PENETRATION_SDF:?set the reviewed mesh SDF path}"

RUN_ID="${RUN_ID:-r065_local_crossing_residual_0_30k_h100_20260815}"
OUTPUT_DIR="$RUNTIME_ROOT/outputs/$RUN_ID"
LOG_DIR="$RUNTIME_ROOT/logs"
CONTRACT_DIR="$RUNTIME_ROOT/contracts"
mkdir -p "$LOG_DIR" "$CONTRACT_DIR"

cd "$PROJECT_ROOT"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "[r065] commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[r065] checkout is dirty; refusing formal execution" >&2
  git status --short >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[r065] output already exists; refusing overwrite: $OUTPUT_DIR" >&2
  exit 2
fi

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
snapshot_config \
  "$PROJECT_ROOT/configs/r062_mesh_no_penetration_0_30k.env" \
  "$CONTRACT_DIR/r062_0_30k.env"
snapshot_config \
  "$PROJECT_ROOT/configs/r065_local_crossing_residual_0_30k.env" \
  "$CONTRACT_DIR/r065_0_30k.env"

"$PYTHON" - \
  "$CONTRACT_DIR/r062_0_30k.env" \
  "$CONTRACT_DIR/r065_0_30k.env" \
  "$CONTRACT_DIR/config_delta.json" <<'PY'
import json
from pathlib import Path
import sys

allowed = {
    "STRAND_CROSSING_SUPPORT",
    "STRAND_CROSSING_WEIGHT",
    "STRAND_CROSSING_REFRESH_INTERVAL",
    "STRAND_CROSSING_QUERY_BATCH",
    "STRAND_CROSSING_EXACT_PAIR_BATCH",
}
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
    key: {"r062": base.get(key), "r065": candidate.get(key)}
    for key in sorted(keys)
    if base.get(key) != candidate.get(key)
}
unexpected = sorted(set(delta) - allowed)
missing = sorted(allowed - set(delta))
if unexpected or missing:
    raise RuntimeError(
        "R065 is not a strict crossing-only config: "
        f"unexpected={unexpected} missing={missing} delta={delta}"
    )
Path(sys.argv[3]).write_text(
    json.dumps(delta, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(delta, indent=2, sort_keys=True))
print("R065_CONFIG_DELTA_PASS")
PY

"$PYTHON" -m pytest -q

ulimit -v unlimited
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "[r065] start formal run at=$(date --iso-8601=seconds) commit=$actual_commit"
PROJECT_ROOT="$PROJECT_ROOT" \
PYTHON="$PYTHON" \
DATA_ROOT="$DATA_ROOT" \
MESH_PATH="$MESH_PATH" \
MESH_NO_PENETRATION_SDF="$MESH_NO_PENETRATION_SDF" \
RUN_ID="$RUN_ID" \
OUTPUT_DIR="$OUTPUT_DIR" \
CONFIG_PATH="$PROJECT_ROOT/configs/r065_local_crossing_residual_0_30k.env" \
RUN_PREFLIGHT=1 \
RUN_BATCH_PREFLIGHT=0 \
  bash "$PROJECT_ROOT/scripts/server/run_white_tiger_stage1.sh" \
  2>&1 | tee "$LOG_DIR/$RUN_ID.log"
echo "[r065] formal run complete at=$(date --iso-8601=seconds)"

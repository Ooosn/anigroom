#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the reviewed R068 checkout}"
RUNTIME_ROOT="${RUNTIME_ROOT:?set RUNTIME_ROOT to a new R068 runtime directory}"
PYTHON="${PYTHON:?set PYTHON to the verified mygs interpreter}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the frozen white-tiger images}"
MESH_PATH="${MESH_PATH:?set MESH_PATH to the frozen aligned mesh}"
MESH_NO_PENETRATION_SDF="${MESH_NO_PENETRATION_SDF:?set the reviewed mesh SDF path}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the reviewed R068 commit}"
RUN_ID="${RUN_ID:?set RUN_ID to a fresh R068 run id}"

R067_CONFIG_PATH="$PROJECT_ROOT/configs/r067_no_frizz_0_30k.env"
R068_CONFIG_PATH="$PROJECT_ROOT/configs/r068_no_crossing_zero_curl_0_30k.env"
TRAINER_PATH="$PROJECT_ROOT/tools/train_white_tiger_stage1.py"
STRAND_SOURCE_PATH="$PROJECT_ROOT/anigroom/grooming/strand_gaussians.py"
OUTPUT_DIR="$RUNTIME_ROOT/outputs/$RUN_ID"
LOG_DIR="$RUNTIME_ROOT/logs"
CONTRACT_DIR="$RUNTIME_ROOT/contracts"

fail() {
  echo "[r068] $*" >&2
  exit 2
}

cd "$PROJECT_ROOT"

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "EXPECTED_COMMIT must be a full 40-character commit"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  fail "commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit"
fi
if [[ -n "$(git status --porcelain=v1)" ]]; then
  echo "[r068] checkout is dirty; refusing formal execution" >&2
  git status --short >&2
  exit 2
fi

[[ ! -e "$RUNTIME_ROOT" ]] || fail "runtime root already exists; refusing reuse: $RUNTIME_ROOT"
[[ ! -e "$OUTPUT_DIR" ]] || fail "output already exists; refusing overwrite: $OUTPUT_DIR"
[[ -x "$PYTHON" ]] || fail "mygs interpreter is not executable: $PYTHON"
[[ -d "$DATA_ROOT" ]] || fail "data root does not exist: $DATA_ROOT"
[[ -f "$MESH_PATH" ]] || fail "mesh does not exist: $MESH_PATH"
[[ -f "$MESH_NO_PENETRATION_SDF" ]] || fail "mesh SDF does not exist: $MESH_NO_PENETRATION_SDF"
[[ -f "$R067_CONFIG_PATH" ]] || fail "R067 source config does not exist: $R067_CONFIG_PATH"
[[ -f "$R068_CONFIG_PATH" ]] || fail "R068 candidate config does not exist: $R068_CONFIG_PATH"

python_prefix="$("$PYTHON" -c 'import sys; print(sys.prefix)')"
case "$python_prefix" in
  */mygs|*/mygs/*) ;;
  *) fail "PYTHON is not from the mygs environment: $python_prefix" ;;
esac

grep -Eq '^CURRENT_CHECKPOINT_VERSION[[:space:]]*=[[:space:]]*9[[:space:]]*$' "$TRAINER_PATH" \
  || fail "reviewed trainer does not declare checkpoint schema 9"
if grep -Eq '^CURRENT_CHECKPOINT_VERSION[[:space:]]*=[[:space:]]*[78][[:space:]]*$' "$TRAINER_PATH"; then
  fail "schema 7/8 trainer is not allowed for R068"
fi

grep -Fq 'source "${CONFIG_DIR}/r067_no_frizz_0_30k.env"' "$R068_CONFIG_PATH" \
  || fail "R068 config does not source the exact R067 config"
grep -Fxq 'STRAND_CROSSING_SUPPORT=0' "$R068_CONFIG_PATH" \
  || fail "R068 config does not explicitly disable crossing support"
grep -Fxq 'STRAND_CROSSING_WEIGHT=0' "$R068_CONFIG_PATH" \
  || fail "R068 config does not explicitly set crossing weight to zero"
grep -Fxq 'STRAND_CROSSING_REFRESH_INTERVAL=0' "$R068_CONFIG_PATH" \
  || fail "R068 config does not explicitly set crossing refresh to zero"

grep -Eq 'enable_curl[[:space:]]*:[[:space:]]*bool[[:space:]]*=[[:space:]]*True' "$STRAND_SOURCE_PATH" \
  || fail "strand source lacks the explicit enable_curl flag"
if [[ "$(grep -Fc 'enable_curl=curl_enabled' "$TRAINER_PATH")" -lt 2 ]]; then
  fail "trainer does not forward the explicit zero-curl flag to both strand paths"
fi
grep -Fq 'self.shape_detail_multiplier > 0.0' "$TRAINER_PATH" \
  || fail "trainer lacks the exact zero-detail curl gate"
grep -Fq 'self.shape_curl_scale > 0.0' "$TRAINER_PATH" \
  || fail "trainer lacks the exact zero-curl-scale gate"

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  fail "RESUME_CHECKPOINT is forbidden; R068 must start from zero"
fi
if [[ -n "${RESUME_OPTIMIZER:-}" ]]; then
  fail "RESUME_OPTIMIZER is forbidden; R068 has no resume path"
fi
if [[ "${STAGE1_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  fail "STAGE1_PREFLIGHT_ONLY=1 is forbidden; R068 requires formal training"
fi
if [[ "${RUN_PREFLIGHT:-1}" == "0" ]]; then
  fail "RUN_PREFLIGHT=0 is forbidden; R068 requires the full data preflight"
fi
if [[ "${RUN_BATCH_PREFLIGHT:-0}" != "0" ]]; then
  fail "reduced batch preflight is forbidden for R068"
fi
if [[ -n "${TRAIN_VIEWS:-}" || -n "${TEST_VIEWS:-}" ]]; then
  fail "TRAIN_VIEWS/TEST_VIEWS overrides are forbidden; R068 requires the full view split"
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

snapshot_config "$R067_CONFIG_PATH" "$CONTRACT_DIR/r067_0_30k.env"
snapshot_config "$R068_CONFIG_PATH" "$CONTRACT_DIR/r068_0_30k.env"

"$PYTHON" - \
  "$CONTRACT_DIR/r067_0_30k.env" \
  "$CONTRACT_DIR/r068_0_30k.env" \
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


r067 = load(sys.argv[1])
r068 = load(sys.argv[2])
keys = set(r067) | set(r068)
delta = {
    key: {"r067": r067.get(key), "r068": r068.get(key)}
    for key in sorted(keys)
    if r067.get(key) != r068.get(key)
}
expected_delta = {
    "STRAND_CROSSING_SUPPORT": {"r067": "1", "r068": "0"},
    "STRAND_CROSSING_WEIGHT": {"r067": "0.001", "r068": "0"},
    "STRAND_CROSSING_REFRESH_INTERVAL": {"r067": "2000", "r068": "0"},
}
if delta != expected_delta:
    raise RuntimeError(f"R068 config delta is not exactly crossing removal: {delta}")

for key, expected in {
    "EXPECTED_WIDTH": "1920",
    "EXPECTED_HEIGHT": "1080",
    "ITERATIONS": "30000",
    "STRAND_CROSSING_QUERY_BATCH": "50000",
    "STRAND_CROSSING_EXACT_PAIR_BATCH": "250000",
}.items():
    if r068.get(key) != expected:
        raise RuntimeError(
            f"R068 requires {key}={expected}, got {r068.get(key)!r}"
        )
    if r067.get(key) != r068.get(key):
        raise RuntimeError(
            f"R068 changed protected value {key}: "
            f"{r067.get(key)!r} -> {r068.get(key)!r}"
        )

Path(sys.argv[3]).write_text(json.dumps(delta, indent=2) + "\n", encoding="utf-8")
print("R068_CONFIG_DELTA_PASS")
print("R068_FULL_RESOLUTION_0_30K_CONFIG_PASS")
print("R068_CROSSING_BATCHES_INERT_PASS")
PY

ulimit -v unlimited
[[ "$(ulimit -v)" == "unlimited" ]] || fail "virtual-memory limit is not unlimited"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "[r068] host=$(hostname) commit=$actual_commit ulimit_v=$(ulimit -v)"
"$PYTHON" -m pytest -q

echo "[r068] start formal from-zero 0-30k at=$(date --iso-8601=seconds)"
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
  CONFIG_PATH="$R068_CONFIG_PATH" \
  RUN_PREFLIGHT=1 \
  RUN_BATCH_PREFLIGHT=0 \
    bash "$PROJECT_ROOT/scripts/server/run_white_tiger_stage1.sh" \
    2>&1 | tee "$LOG_DIR/$RUN_ID.log"

FINAL_CHECKPOINT="$OUTPUT_DIR/checkpoint_030000.pt"
FINAL_LOG="$LOG_DIR/$RUN_ID.log"
METRICS_LOG="$OUTPUT_DIR/metrics.jsonl"
[[ -s "$FINAL_CHECKPOINT" ]] || fail "missing final 30k checkpoint: $FINAL_CHECKPOINT"
[[ -s "$FINAL_LOG" ]] || fail "missing final formal log: $FINAL_LOG"
[[ -s "$METRICS_LOG" ]] || fail "missing final metrics log: $METRICS_LOG"

"$PYTHON" - "$FINAL_CHECKPOINT" "$FINAL_LOG" "$METRICS_LOG" <<'PY'
import json
from pathlib import Path
import sys

import torch


checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
version = int(checkpoint.get("checkpoint_version", -1))
if version != 9:
    raise RuntimeError(f"R068 requires checkpoint schema 9, got {version}")
if int(checkpoint.get("iteration", -1)) != 30000:
    raise RuntimeError(
        f"R068 requires a from-zero 30k checkpoint, got iteration "
        f"{checkpoint.get('iteration')!r}"
    )
if checkpoint.get("checkpoint_kind") != "stage1_full":
    raise RuntimeError(
        f"R068 requires checkpoint_kind=stage1_full, got "
        f"{checkpoint.get('checkpoint_kind')!r}"
    )

config = checkpoint.get("config")
if not isinstance(config, dict):
    raise RuntimeError("R068 checkpoint is missing its resolved config mapping")
for key, expected in {
    "strand_crossing_support": False,
    "strand_crossing_weight": 0.0,
    "strand_crossing_refresh_interval": 0,
    "strand_crossing_query_batch": 50000,
    "strand_crossing_exact_pair_batch": 250000,
}.items():
    if config.get(key) != expected:
        raise RuntimeError(
            f"R068 checkpoint crossing config mismatch for {key}: "
            f"{config.get(key)!r} != {expected!r}"
        )

if checkpoint.get("strand_crossing_active_set") is not None:
    raise RuntimeError("R068 checkpoint contains a crossing active set")
if checkpoint.get("strand_crossing_last_refresh_iteration") != 0:
    raise RuntimeError("R068 checkpoint contains a crossing refresh iteration")
if checkpoint.get("strand_crossing_history") != []:
    raise RuntimeError("R068 checkpoint contains crossing active-set history")


def assert_no_frizz_keys(value, path="checkpoint"):
    if isinstance(value, dict):
        for key, item in value.items():
            if "frizz" in str(key).lower():
                raise RuntimeError(f"R068 no-frizz schema contains {path}.{key}")
            assert_no_frizz_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_frizz_keys(item, f"{path}[{index}]")


assert_no_frizz_keys(checkpoint)


def json_lines(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


formal_log = Path(sys.argv[2]).read_text(encoding="utf-8")
if '"strand_crossing_refresh"' in formal_log:
    raise RuntimeError("R068 formal log contains a crossing refresh event")
setup_records = [
    record
    for record in json_lines(sys.argv[2])
    if record.get("setup_progress") == "strand_crossing_state_ready"
]
if not setup_records:
    raise RuntimeError("R068 formal log lacks strand_crossing_state_ready")
state = setup_records[-1]
if state.get("enabled") is not False:
    raise RuntimeError(f"R068 crossing state is not disabled: {state}")
if state.get("active_pair_count") != 0 or state.get("last_refresh_iteration") != 0:
    raise RuntimeError(f"R068 crossing state is not empty: {state}")
if state.get("history_count") != 0:
    raise RuntimeError(f"R068 crossing history is not empty: {state}")

metric_records = json_lines(sys.argv[3])
if not metric_records:
    raise RuntimeError("R068 metrics log contains no JSON records")
for record in metric_records:
    crossing = record.get("strand_crossing")
    if not isinstance(crossing, dict):
        raise RuntimeError(f"R068 metrics omit the crossing state: {record}")
    if crossing.get("active_pair_count") != 0:
        raise RuntimeError(f"R068 metrics contain active crossing pairs: {record}")
    if crossing.get("last_refresh_iteration") != 0:
        raise RuntimeError(f"R068 metrics contain a crossing refresh: {record}")
final_records = [
    record for record in metric_records if int(record.get("iteration", -1)) == 30000
]
if not final_records:
    raise RuntimeError("R068 metrics log lacks the final 30k record")
if final_records[-1].get("strand_crossing", {}).get("active_pair_count") != 0:
    raise RuntimeError("R068 final metrics record contains active crossing pairs")
print("R068_SCHEMA9_NO_FRIZZ_PASS")
print("R068_CROSSING_DISABLED_CHECKPOINT_PASS")
print("R068_CROSSING_DISABLED_LOG_PASS")
PY
echo "[r068] formal run complete at=$(date --iso-8601=seconds)"

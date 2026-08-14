#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the clean R062 checkout}"
RUNTIME_ROOT="${RUNTIME_ROOT:?set RUNTIME_ROOT to a new R062 runtime directory}"
PYTHON="${PYTHON:?set PYTHON to the verified mygs interpreter}"

export PREFLIGHT_ID=r062_mesh_no_penetration_fullres_preflight_h100_20260814
export RUN_ID=r062_mesh_no_penetration_0_30k_h100_20260814
export LABEL=r062_mesh_no_penetration
export PREFLIGHT_CONFIG=r062_mesh_no_penetration_fullres_preflight.env
export RUN_CONFIG=r062_mesh_no_penetration_0_30k.env
export REQUIRE_NO_LOCAL_CHILD_COLOR=1

export MESH_NO_PENETRATION_SDF="${MESH_NO_PENETRATION_SDF:-/home/wangyy/anigroom-assets/collision/white_tiger_sdf_long512.npz}"
export EXPECTED_MESH_NO_PENETRATION_SDF_SHA256=766e177fbeeb89fc779292f56662c7c6b256f7d4365415baa366cef04af10530
export R061_BASELINE_CHECKPOINT="${R061_BASELINE_CHECKPOINT:-/home/wangyy/anigroom-r061-gaussian-only-appearance-runtime-20260814/outputs/r061_gaussian_only_appearance_0_30k_h100_20260814/checkpoint_030000.pt}"
export EXPECTED_R061_BASELINE_CHECKPOINT_SHA256=c90052175aa1d1b1a8cfe79fe52ae0e4fb9c9dd2fe8bf76472e2e572f993d538
export R061_BASELINE_PREFLIGHT_METRICS="${R061_BASELINE_PREFLIGHT_METRICS:-/home/wangyy/anigroom-r061-gaussian-only-appearance-runtime-20260814/outputs/r061_gaussian_only_appearance_fullres_preflight_h100_20260814/metrics.jsonl}"
export PREFLIGHT_POSTCHECK_SCRIPT="$PROJECT_ROOT/scripts/server/verify_r062_mesh_no_penetration.sh"
export POSTPROCESS_POSTCHECK_SCRIPT="$PROJECT_ROOT/scripts/server/verify_r062_mesh_no_penetration.sh"

actual_sdf_sha="$(sha256sum "$MESH_NO_PENETRATION_SDF" | awk '{print $1}')"
if [[ "$actual_sdf_sha" != "$EXPECTED_MESH_NO_PENETRATION_SDF_SHA256" ]]; then
  echo "[r062] SDF SHA256 mismatch: expected=$EXPECTED_MESH_NO_PENETRATION_SDF_SHA256 actual=$actual_sdf_sha" >&2
  exit 2
fi
actual_baseline_sha="$(sha256sum "$R061_BASELINE_CHECKPOINT" | awk '{print $1}')"
if [[ "$actual_baseline_sha" != "$EXPECTED_R061_BASELINE_CHECKPOINT_SHA256" ]]; then
  echo "[r062] R061 checkpoint SHA256 mismatch: expected=$EXPECTED_R061_BASELINE_CHECKPOINT_SHA256 actual=$actual_baseline_sha" >&2
  exit 2
fi
[[ -s "$R061_BASELINE_PREFLIGHT_METRICS" ]] || {
  echo "[r062] missing R061 preflight metrics: $R061_BASELINE_PREFLIGHT_METRICS" >&2
  exit 2
}

contract_dir="$RUNTIME_ROOT/contracts"
mkdir -p "$contract_dir"
snapshot_config() {
  local config_path="$1"
  local output_path="$2"
  local collision_sdf="${3:-}"
  env -i \
    HOME="$HOME" \
    PATH="$PATH" \
    MESH_NO_PENETRATION_SDF="$collision_sdf" \
    bash -c 'set -a; source "$1"; env' _ "$config_path" \
    | LC_ALL=C sort > "$output_path"
}
snapshot_config \
  "$PROJECT_ROOT/configs/r061_gaussian_only_appearance_0_30k.env" \
  "$contract_dir/r061_0_30k.env"
snapshot_config \
  "$PROJECT_ROOT/configs/r062_mesh_no_penetration_0_30k.env" \
  "$contract_dir/r062_0_30k.env" \
  "$MESH_NO_PENETRATION_SDF"
snapshot_config \
  "$PROJECT_ROOT/configs/r061_gaussian_only_appearance_fullres_preflight.env" \
  "$contract_dir/r061_preflight.env"
snapshot_config \
  "$PROJECT_ROOT/configs/r062_mesh_no_penetration_fullres_preflight.env" \
  "$contract_dir/r062_preflight.env" \
  "$MESH_NO_PENETRATION_SDF"

"$PYTHON" - \
  "$contract_dir/r061_0_30k.env" \
  "$contract_dir/r062_0_30k.env" \
  "$contract_dir/r061_preflight.env" \
  "$contract_dir/r062_preflight.env" \
  "$contract_dir/config_delta.json" <<'PY'
import json
from pathlib import Path
import sys

allowed = {
    "MESH_NO_PENETRATION_SUPPORT",
    "MESH_NO_PENETRATION_SDF",
    "MESH_NO_PENETRATION_WEIGHT",
    "MESH_NO_PENETRATION_ROOT_BATCH",
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

def compare(base_path, candidate_path):
    base = load(base_path)
    candidate = load(candidate_path)
    keys = set(base) | set(candidate)
    delta = {
        key: {"r061": base.get(key), "r062": candidate.get(key)}
        for key in sorted(keys)
        if base.get(key) != candidate.get(key)
    }
    unexpected = sorted(set(delta) - allowed)
    missing = sorted(allowed - set(delta))
    if unexpected or missing:
        raise RuntimeError(
            f"R062 is not a strict collision-only config: "
            f"unexpected={unexpected} missing={missing} delta={delta}"
        )
    return delta

report = {
    "formal_0_30k": compare(sys.argv[1], sys.argv[2]),
    "fullres_preflight": compare(sys.argv[3], sys.argv[4]),
}
Path(sys.argv[5]).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2, sort_keys=True))
print("R062_CONFIG_DELTA_PASS")
PY

reference_dir="$RUNTIME_ROOT/reference/r061_no_penetration"
if [[ -e "$reference_dir" ]]; then
  echo "[r062] reference output already exists: $reference_dir" >&2
  exit 2
fi
mkdir -p "$(dirname "$reference_dir")"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"
"$PYTHON" tools/diagnose_checkpoint_no_penetration.py \
  --checkpoint "$R061_BASELINE_CHECKPOINT" \
  --sdf "$MESH_NO_PENETRATION_SDF" \
  --output-dir "$reference_dir" \
  --device cuda \
  --samples 64 \
  --query-root-chunk 16384 \
  --gradient-root-batch 16384 \
  --visual-strands 1
export R061_NO_PENETRATION_REPORT="$reference_dir/report.json"

"$PYTHON" -m pytest tests/test_sdf_collision.py tests/test_stage1_no_penetration.py -q

exec bash "$PROJECT_ROOT/scripts/server/run_r061_gaussian_only_appearance.sh"

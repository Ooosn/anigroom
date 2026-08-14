#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the clean R060 checkout}"
RUNTIME_ROOT="${RUNTIME_ROOT:?set RUNTIME_ROOT to a new R060 runtime directory}"
PYTHON="${PYTHON:?set PYTHON to the verified mygs interpreter}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the frozen white-tiger images}"
MESH_PATH="${MESH_PATH:?set MESH_PATH to the frozen aligned mesh}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the reviewed R060 commit}"

PREFLIGHT_ID="${PREFLIGHT_ID:-r060_relative_shape_amplitudes_fullres_preflight_h100_20260813}"
RUN_ID="${RUN_ID:-r060_relative_shape_amplitudes_0_30k_h100_20260813}"
LABEL="${LABEL:-r060_relative_shape_amplitudes}"
PREFLIGHT_CONFIG="${PREFLIGHT_CONFIG:-r060_relative_shape_amplitudes_fullres_preflight.env}"
RUN_CONFIG="${RUN_CONFIG:-r060_relative_shape_amplitudes_0_30k.env}"
REQUIRE_NO_LOCAL_CHILD_COLOR="${REQUIRE_NO_LOCAL_CHILD_COLOR:-0}"
PREFLIGHT_POSTCHECK_SCRIPT="${PREFLIGHT_POSTCHECK_SCRIPT:-}"
POSTPROCESS_POSTCHECK_SCRIPT="${POSTPROCESS_POSTCHECK_SCRIPT:-}"

LOG_ROOT="$RUNTIME_ROOT/logs"
OUTPUT_ROOT="$RUNTIME_ROOT/outputs"
CONTROL_ROOT="$RUNTIME_ROOT/control"
mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$CONTROL_ROOT"

ulimit -v unlimited
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"
actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "[r060] commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[r060] checkout is dirty; refusing formal execution" >&2
  git status --short >&2
  exit 2
fi

echo "[r060] host=$(hostname) commit=$actual_commit ulimit_v=$(ulimit -v)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader

"$PYTHON" -m pytest \
  tests/test_strand_deformations.py \
  tests/test_zero_centered_render_geometry.py::test_structure_update_transports_residual_state_and_strict_checkpoint \
  tests/test_gaussian_rgb_residual.py::test_rgb_flow_backward_excludes_color_but_preserves_rgb_geometry_gradient \
  tests/test_gaussian_rgb_residual.py::test_shape_gate_is_zero_before_handoff_and_joint_controls_receive_gradients \
  -q

run_stage1() {
  local run_id="$1"
  local config_name="$2"
  local output_dir="$OUTPUT_ROOT/$run_id"
  local log_file="$LOG_ROOT/$run_id.log"

  if [[ -e "$output_dir" ]]; then
    echo "[r060] output already exists; refusing overwrite: $output_dir" >&2
    exit 2
  fi

  echo "[r060] start run=$run_id config=$config_name at=$(date --iso-8601=seconds)"
  PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHON="$PYTHON" \
  DATA_ROOT="$DATA_ROOT" \
  MESH_PATH="$MESH_PATH" \
  RUN_ID="$run_id" \
  OUTPUT_DIR="$output_dir" \
  CONFIG_PATH="$PROJECT_ROOT/configs/$config_name" \
  RUN_PREFLIGHT=1 \
  RUN_BATCH_PREFLIGHT=0 \
    bash scripts/server/run_white_tiger_stage1.sh 2>&1 | tee "$log_file"
  echo "[r060] finish run=$run_id at=$(date --iso-8601=seconds)"
}

verify_active_path_preflight() {
  local output_dir="$OUTPUT_ROOT/$PREFLIGHT_ID"
  local checkpoint="$output_dir/checkpoint_000002.pt"
  [[ -s "$checkpoint" ]] || {
    echo "[r060] missing preflight checkpoint: $checkpoint" >&2
    exit 2
  }

  "$PYTHON" - "$checkpoint" "$output_dir/metrics.jsonl" "$REQUIRE_NO_LOCAL_CHILD_COLOR" <<'PY'
import json
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
require_no_local_child_color = bool(int(sys.argv[3]))
if int(checkpoint.get("checkpoint_version", -1)) != 7:
    raise RuntimeError("R060 preflight did not write checkpoint schema 7")

config = checkpoint["config"]
if not bool(config.get("rgb_flow_exclude_color_gradients", False)):
    raise RuntimeError("R060 checkpoint lost R057 flow/color gradient ownership")
for key in (
    "shape_detail_multiplier",
    "secondary_shape_residual_multiplier",
    "guide_residual_multiplier",
    "gaussian_rgb_residual_multiplier",
):
    if float(checkpoint[key]) < 0.999:
        raise RuntimeError(f"R060 active-path preflight left {key} gated: {checkpoint[key]}")

model_state = checkpoint["model"]
required_model_state = {
    "groom.curl_radius_ratio_raw",
    "groom.frizz_amplitude_ratio_raw",
    "groom.curl_turns_raw",
    "groom.frizz_seed_phase",
    "groom.length_raw",
    "guide_curl_radius_ratio_raw",
    "guide_frizz_amplitude_ratio_raw",
    "secondary_geometry_residual.curl_radius_ratio_raw",
    "secondary_geometry_residual.frizz_amplitude_ratio_raw",
}
missing_state = sorted(required_model_state - set(model_state))
if missing_state:
    raise RuntimeError(f"R060 checkpoint is missing ratio geometry state: {missing_state}")
retired = sorted(
    key
    for key in model_state
    if key.endswith(".curl_radius_raw")
    or key.endswith(".frizz_raw")
    or "curl_frequency" in key
)
if retired:
    raise RuntimeError(f"R060 checkpoint contains retired absolute state: {retired}")

ratio_rows = model_state["groom.curl_radius_ratio_raw"]
frizz_rows = model_state["groom.frizz_amplitude_ratio_raw"]
turns = model_state["groom.curl_turns_raw"]
seed = model_state["groom.frizz_seed_phase"]
length = model_state["groom.length_raw"]
if not (
    ratio_rows.shape == frizz_rows.shape == turns.shape == seed.shape == length.shape
):
    raise RuntimeError(
        "R060 geometry rows do not match render roots: "
        f"curl={tuple(ratio_rows.shape)} frizz={tuple(frizz_rows.shape)} "
        f"turns={tuple(turns.shape)} seed={tuple(seed.shape)} "
        f"length={tuple(length.shape)}"
    )
for name, value in model_state.items():
    if "curl_radius_ratio_raw" in name or "frizz_amplitude_ratio_raw" in name:
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"R060 non-finite ratio geometry state: {name}")
if float(seed.std()) <= 0.0:
    raise RuntimeError("R060 frizz seed state collapsed to a constant")

required_optimizer = {
    "guide_direction_local_raw",
    "guide_brush_stiffness_raw",
    "guide_curl_radius_ratio_raw",
    "guide_frizz_amplitude_ratio_raw",
    "secondary_geometry_residual.direction_local_raw",
    "secondary_geometry_residual.curl_radius_ratio_raw",
    "secondary_geometry_residual.frizz_amplitude_ratio_raw",
    "groom.root_color_raw",
    "groom.tip_color_raw",
    "gaussian_rgb_residual.raw",
}
optimizer_names = checkpoint["optimizer_param_names"]
flat_optimizer_names = {name for group in optimizer_names for name in group}
if require_no_local_child_color:
    if bool(config.get("local_child_color_support", True)):
        raise RuntimeError("R061 preflight config enabled local child/render-root color")
    retired_local_color = sorted(
        name for name in model_state if "child_color_delta_raw" in name
    )
    if retired_local_color:
        raise RuntimeError(
            f"R061 checkpoint contains retired local color state: {retired_local_color}"
        )
    if "child_color_delta_raw" in flat_optimizer_names:
        raise RuntimeError("R061 optimizer contains retired local color state")
for fixed_coordinate in ("groom.frizz_seed_phase", "groom.curl_turns_raw", "groom.curl_phase"):
    if fixed_coordinate in flat_optimizer_names:
        raise RuntimeError(f"R060 changed R057 shape ownership: {fixed_coordinate}")

seen = set()
optimizer = checkpoint["optimizer"]
for group, names in zip(optimizer["param_groups"], optimizer_names, strict=True):
    for parameter_id, name in zip(group["params"], names, strict=True):
        if name not in required_optimizer:
            continue
        state = optimizer["state"].get(parameter_id, {})
        exp_avg = state.get("exp_avg")
        if exp_avg is None or not bool(torch.isfinite(exp_avg).all()):
            raise RuntimeError(f"R060 invalid Adam state for {name}")
        if int(torch.count_nonzero(exp_avg)) == 0:
            raise RuntimeError(f"R060 zero Adam first moment for {name}")
        print(
            f"[r060] active_path={name} "
            f"nonzero_adam_m={int(torch.count_nonzero(exp_avg))} "
            f"max_abs_adam_m={float(exp_avg.abs().max()):.9g}"
        )
        seen.add(name)
missing_optimizer = sorted(required_optimizer - seen)
if missing_optimizer:
    raise RuntimeError(f"R060 preflight is missing optimizer parameters: {missing_optimizer}")

rows = []
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if "train" in row:
            rows.append(row)
if not rows or not bool(rows[-1].get("rgb_flow_exclude_color_gradients", False)):
    raise RuntimeError("R060 metric record did not confirm gradient ownership")
print(
    f"[r060] relative_geometry_rows={int(seed.shape[0])} "
    f"frizz_seed_std={float(seed.std()):.9g}"
)
PY
}

postprocess() {
  local output_dir="$OUTPUT_ROOT/$RUN_ID"
  local checkpoint="$output_dir/checkpoint_030000.pt"
  local post_root="$RUNTIME_ROOT/postprocess/$LABEL"
  local log_file="$LOG_ROOT/${LABEL}_postprocess.log"

  [[ -s "$checkpoint" ]] || {
    echo "[r060] missing final checkpoint: $checkpoint" >&2
    exit 2
  }
  if [[ -e "$post_root" ]]; then
    echo "[r060] postprocess output already exists: $post_root" >&2
    exit 2
  fi
  mkdir -p "$post_root/rgb_views" "$post_root/attributes_view09" "$post_root/strands"

  {
    "$PYTHON" tools/render_white_tiger_stage1_checkpoint_views.py \
      --checkpoint "$checkpoint" \
      --output-dir "$post_root/rgb_views" \
      --view-ids "0 5 9 14 18 21 27 32" \
      --device cuda

    "$PYTHON" tools/visualize_white_tiger_groom_attributes.py \
      --checkpoint "$checkpoint" \
      --view 9 \
      --output-dir "$post_root/attributes_view09" \
      --base-image "$post_root/rgb_views/view_09_pred.png"

    "$PYTHON" tools/export_white_tiger_checkpoint_strands.py \
      --checkpoint "$checkpoint" \
      --output "$post_root/strands/${LABEL}_030000_child1_100k_samples32.npz" \
      --device cpu \
      --samples 32 \
      --child-count 1 \
      --max-strands 100000 \
      --seed 29 \
      --uniform-color 0.82 0.80 0.72

    find "$post_root" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
      > "$post_root/SHA256SUMS"
  } 2>&1 | tee "$log_file"
}

run_stage1 \
  "$PREFLIGHT_ID" \
  "$PREFLIGHT_CONFIG"
verify_active_path_preflight
if [[ -n "$PREFLIGHT_POSTCHECK_SCRIPT" ]]; then
  [[ -f "$PREFLIGHT_POSTCHECK_SCRIPT" ]] || {
    echo "[r060] missing preflight postcheck: $PREFLIGHT_POSTCHECK_SCRIPT" >&2
    exit 2
  }
  bash "$PREFLIGHT_POSTCHECK_SCRIPT" \
    preflight \
    "$PROJECT_ROOT" \
    "$PYTHON" \
    "$MESH_PATH" \
    "$OUTPUT_ROOT/$PREFLIGHT_ID" \
    "$RUNTIME_ROOT"
fi
touch "$CONTROL_ROOT/preflight_passed"

run_stage1 \
  "$RUN_ID" \
  "$RUN_CONFIG"
postprocess
if [[ -n "$POSTPROCESS_POSTCHECK_SCRIPT" ]]; then
  [[ -f "$POSTPROCESS_POSTCHECK_SCRIPT" ]] || {
    echo "[r060] missing postprocess postcheck: $POSTPROCESS_POSTCHECK_SCRIPT" >&2
    exit 2
  }
  bash "$POSTPROCESS_POSTCHECK_SCRIPT" \
    final \
    "$PROJECT_ROOT" \
    "$PYTHON" \
    "$MESH_PATH" \
    "$OUTPUT_ROOT/$RUN_ID" \
    "$RUNTIME_ROOT"
fi

touch "$CONTROL_ROOT/run_done"
echo "[r060] run and postprocess complete; waiting for $CONTROL_ROOT/release"
while [[ ! -e "$CONTROL_ROOT/release" ]]; do
  sleep 30
done

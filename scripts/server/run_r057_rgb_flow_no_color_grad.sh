#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the clean R057 checkout}"
RUNTIME_ROOT="${RUNTIME_ROOT:?set RUNTIME_ROOT to a new R057 runtime directory}"
PYTHON="${PYTHON:?set PYTHON to the verified mygs interpreter}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the frozen white-tiger images}"
MESH_PATH="${MESH_PATH:?set MESH_PATH to the frozen aligned mesh}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the reviewed R057 commit}"

PREFLIGHT_ID=r057_rgb_flow_no_color_grad_fullres_preflight_h100_20260811
RUN_ID=r057_rgb_flow_no_color_grad_0_30k_h100_20260811
LABEL=r057_rgb_flow_no_color_grad

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
  echo "[r057] commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[r057] checkout is dirty; refusing formal execution" >&2
  git status --short >&2
  exit 2
fi

echo "[r057] host=$(hostname) commit=$actual_commit ulimit_v=$(ulimit -v)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader

"$PYTHON" -m pytest \
  tests/test_gaussian_rgb_residual.py::test_rgb_flow_backward_excludes_color_but_preserves_rgb_geometry_gradient \
  -q

run_stage1() {
  local run_id="$1"
  local config_name="$2"
  local output_dir="$OUTPUT_ROOT/$run_id"
  local log_file="$LOG_ROOT/$run_id.log"

  if [[ -e "$output_dir" ]]; then
    echo "[r057] output already exists; refusing overwrite: $output_dir" >&2
    exit 2
  fi

  echo "[r057] start run=$run_id config=$config_name at=$(date --iso-8601=seconds)"
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
  echo "[r057] finish run=$run_id at=$(date --iso-8601=seconds)"
}

verify_active_path_preflight() {
  local output_dir="$OUTPUT_ROOT/$PREFLIGHT_ID"
  local checkpoint="$output_dir/checkpoint_000002.pt"
  [[ -s "$checkpoint" ]] || {
    echo "[r057] missing preflight checkpoint: $checkpoint" >&2
    exit 2
  }

  "$PYTHON" - "$checkpoint" "$output_dir/metrics.jsonl" <<'PY'
import json
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
config = checkpoint["config"]
if not bool(config.get("rgb_flow_exclude_color_gradients", False)):
    raise RuntimeError("R057 checkpoint did not enable flow/color gradient separation")
for key in (
    "shape_detail_multiplier",
    "secondary_shape_residual_multiplier",
    "guide_residual_multiplier",
    "gaussian_rgb_residual_multiplier",
):
    if float(checkpoint[key]) < 0.999:
        raise RuntimeError(f"R057 active-path preflight left {key} gated: {checkpoint[key]}")

required = {
    "guide_direction_local_raw",
    "guide_brush_stiffness_raw",
    "guide_curl_radius_raw",
    "guide_frizz_raw",
    "secondary_geometry_residual.direction_local_raw",
    "secondary_geometry_residual.curl_radius_raw",
    "secondary_geometry_residual.frizz_raw",
    "groom.root_color_raw",
    "groom.tip_color_raw",
    "gaussian_rgb_residual.raw",
}
seen = set()
optimizer = checkpoint["optimizer"]
for group, names in zip(
    optimizer["param_groups"], checkpoint["optimizer_param_names"], strict=True
):
    for parameter_id, name in zip(group["params"], names, strict=True):
        if name not in required:
            continue
        state = optimizer["state"].get(parameter_id, {})
        exp_avg = state.get("exp_avg")
        if exp_avg is None or not bool(torch.isfinite(exp_avg).all()):
            raise RuntimeError(f"R057 invalid Adam state for {name}")
        if int(torch.count_nonzero(exp_avg)) == 0:
            raise RuntimeError(f"R057 zero Adam first moment for {name}")
        print(
            f"[r057] active_path={name} nonzero_adam_m={int(torch.count_nonzero(exp_avg))} "
            f"max_abs_adam_m={float(exp_avg.abs().max()):.9g}"
        )
        seen.add(name)
missing = sorted(required - seen)
if missing:
    raise RuntimeError(f"R057 preflight is missing optimizer parameters: {missing}")

rows = []
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        if "train" in row:
            rows.append(row)
if not rows or not bool(rows[-1].get("rgb_flow_exclude_color_gradients", False)):
    raise RuntimeError("R057 metric record did not confirm gradient ownership")
PY
}

postprocess() {
  local output_dir="$OUTPUT_ROOT/$RUN_ID"
  local checkpoint="$output_dir/checkpoint_030000.pt"
  local post_root="$RUNTIME_ROOT/postprocess/$LABEL"
  local log_file="$LOG_ROOT/${LABEL}_postprocess.log"

  [[ -s "$checkpoint" ]] || {
    echo "[r057] missing final checkpoint: $checkpoint" >&2
    exit 2
  }
  if [[ -e "$post_root" ]]; then
    echo "[r057] postprocess output already exists: $post_root" >&2
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
  r057_rgb_flow_no_color_grad_fullres_preflight.env
verify_active_path_preflight
touch "$CONTROL_ROOT/preflight_passed"

run_stage1 \
  "$RUN_ID" \
  r057_rgb_flow_no_color_grad_0_30k.env
postprocess

touch "$CONTROL_ROOT/run_done"
echo "[r057] run and postprocess complete; waiting for $CONTROL_ROOT/release"
while [[ ! -e "$CONTROL_ROOT/release" ]]; do
  sleep 30
done

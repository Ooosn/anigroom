#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the clean R053 checkout}"
RUNTIME_ROOT="${RUNTIME_ROOT:?set RUNTIME_ROOT to a new R053 runtime directory}"
PYTHON="${PYTHON:?set PYTHON to the verified mygs interpreter}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the frozen white-tiger images}"
MESH_PATH="${MESH_PATH:?set MESH_PATH to the frozen aligned mesh}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the reviewed R053 commit}"

PREFLIGHT_ID=r053_shape_detail_gaussian_residual_fullres_preflight_h100_20260808
CONTROL_ID=r053_shape_detail_no_gaussian_residual_0_30k_h100_20260808
TREATMENT_ID=r053_shape_detail_gaussian_residual_0_30k_h100_20260808

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
  echo "[r053] commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "[r053] checkout is dirty; refusing formal execution" >&2
  git status --short >&2
  exit 2
fi

echo "[r053] host=$(hostname) commit=$actual_commit ulimit_v=$(ulimit -v)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader

run_stage1() {
  local run_id="$1"
  local config_name="$2"
  local output_dir="$OUTPUT_ROOT/$run_id"
  local log_file="$LOG_ROOT/$run_id.log"

  if [[ -e "$output_dir" ]]; then
    echo "[r053] output already exists; refusing overwrite: $output_dir" >&2
    exit 2
  fi

  echo "[r053] start run=$run_id config=$config_name at=$(date --iso-8601=seconds)"
  PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHON="$PYTHON" \
  DATA_ROOT="$DATA_ROOT" \
  MESH_PATH="$MESH_PATH" \
  RUN_ID="$run_id" \
  OUTPUT_DIR="$output_dir" \
  CONFIG_PATH="$PROJECT_ROOT/configs/$config_name" \
  RUN_PREFLIGHT=1 \
    bash scripts/server/run_white_tiger_stage1.sh 2>&1 | tee "$log_file"
  echo "[r053] finish run=$run_id at=$(date --iso-8601=seconds)"
}

postprocess() {
  local run_id="$1"
  local label="$2"
  local output_dir="$OUTPUT_ROOT/$run_id"
  local checkpoint="$output_dir/checkpoint_030000.pt"
  local post_root="$RUNTIME_ROOT/postprocess/$label"
  local log_file="$LOG_ROOT/${label}_postprocess.log"

  [[ -s "$checkpoint" ]] || {
    echo "[r053] missing final checkpoint: $checkpoint" >&2
    exit 2
  }
  if [[ -e "$post_root" ]]; then
    echo "[r053] postprocess output already exists: $post_root" >&2
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
      --output "$post_root/strands/${label}_030000_child1_100k_samples32.npz" \
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
  r053_shape_detail_gaussian_residual_fullres_preflight.env
touch "$CONTROL_ROOT/preflight_passed"
echo "[r053] preflight passed; waiting for $CONTROL_ROOT/authorize_pair"
while [[ ! -e "$CONTROL_ROOT/authorize_pair" ]]; do
  sleep 10
done

run_stage1 \
  "$CONTROL_ID" \
  r053_shape_detail_no_gaussian_residual_0_30k.env
run_stage1 \
  "$TREATMENT_ID" \
  r053_shape_detail_gaussian_residual_0_30k.env

postprocess "$CONTROL_ID" r053_control
postprocess "$TREATMENT_ID" r053_treatment

touch "$CONTROL_ROOT/pair_done"
echo "[r053] pair and postprocess complete; waiting for $CONTROL_ROOT/release"
while [[ ! -e "$CONTROL_ROOT/release" ]]; do
  sleep 30
done

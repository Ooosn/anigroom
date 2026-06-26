#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/neuralfur_work/whiteTiger_processed/roaringwalk}"
MESH_PATH="${MESH_PATH:-${PROJECT_ROOT}/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/white_tiger_stage1_formal.env}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[stage1] CONFIG_PATH does not exist: $CONFIG_PATH" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$CONFIG_PATH"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[stage1] missing required config: ${name}" >&2
    echo "[stage1] pass it via CONFIG_PATH or environment; no hidden algorithm defaults are allowed" >&2
    exit 2
  fi
}

required_config=(
  ROOT_COUNT
  CANDIDATE_MULTIPLIER
  ITERATIONS
  EVAL_EVERY
  SAVE_EVERY
  TEST_STRIDE
  EXPECTED_WIDTH
  EXPECTED_HEIGHT
  SAMPLES
  MIN_SEGMENTS
  MAX_SEGMENTS
  CHILD_COUNT
  PROJECTED_INIT_VIEWS
  PROJECTED_INIT_MIN_CONFIDENCE
  LR_GROOM
  LR_HIGH_FREQUENCY_SHAPE_SCALE
  LR_COLOR
  LR_ROOT
  LR_CALIBRATION
  RGB_WEIGHT
  RANDOM_BACKING_LOSS_WEIGHT
  MASK_WEIGHT
  ORIENTATION_WEIGHT
  ORIENTATION_DETAIL_WEIGHT
  SMOOTH_WEIGHT
  STRAND_SHAPE_SMOOTH_WEIGHT
  SHAPE_PRIOR_WEIGHT
  ROOT_MOVE_REG_WEIGHT
  ORIENTATION_MIN_CONFIDENCE
  RANDOM_BACKING_COLOR
  BACKING_COLOR_MIN
  BACKING_COLOR_MAX
  MESH_DEPTH_CLIPPING
  MESH_DEPTH_ABS_TOLERANCE
  MESH_DEPTH_REL_TOLERANCE
  MESH_DEPTH_LOCAL_KERNEL
  MESH_BACKING_COMPOSITING
  DENSIFY_WARMUP
  DENSIFY_INTERVAL
  DENSIFY_UNTIL
  DENSIFY_SCORE_THRESHOLD
  DENSIFY_MIN_CONTRIBUTION
  MAX_SPLITS_PER_EVENT
  SPLIT_CHILDREN_PER_PARENT
  SPLIT_NEIGHBOR_COUNT
  SPLIT_CANDIDATE_RINGS
  SPLIT_CANDIDATE_FACE_COUNT
  SPLIT_MIN_CHILD_DISTANCE
  PRUNE_START
  PRUNE_INTERVAL
  PRUNE_MIN_CONTRIBUTION
  PRUNE_MIN_OPACITY
  PRUNE_MAX_FRACTION
)

for name in "${required_config[@]}"; do
  require_var "$name"
done

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PROJECT_ROOT PYTHON DATA_ROOT MESH_PATH RUN_ID OUTPUT_DIR EXPECTED_WIDTH EXPECTED_HEIGHT

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "[stage1] preflight"
bash scripts/server/preflight_white_tiger_stage1.sh

echo "[stage1] output_dir=${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

cmd=(
  "$PYTHON" tools/train_white_tiger_stage1.py
  --data-root "$DATA_ROOT"
  --mesh-path "$MESH_PATH"
  --output-dir "$OUTPUT_DIR"
  --root-count "$ROOT_COUNT"
  --candidate-multiplier "$CANDIDATE_MULTIPLIER"
  --iterations "$ITERATIONS"
  --eval-every "$EVAL_EVERY"
  --save-every "$SAVE_EVERY"
  --test-stride "$TEST_STRIDE"
  --expected-width "$EXPECTED_WIDTH"
  --expected-height "$EXPECTED_HEIGHT"
  --samples "$SAMPLES"
  --min-segments "$MIN_SEGMENTS"
  --max-segments "$MAX_SEGMENTS"
  --child-count "$CHILD_COUNT"
  --projected-init-views "$PROJECTED_INIT_VIEWS"
  --projected-init-min-confidence "$PROJECTED_INIT_MIN_CONFIDENCE"
  --lr-groom "$LR_GROOM"
  --lr-high-frequency-shape-scale "$LR_HIGH_FREQUENCY_SHAPE_SCALE"
  --lr-color "$LR_COLOR"
  --lr-root "$LR_ROOT"
  --lr-calibration "$LR_CALIBRATION"
  --rgb-weight "$RGB_WEIGHT"
  --random-backing-loss-weight "$RANDOM_BACKING_LOSS_WEIGHT"
  --mask-weight "$MASK_WEIGHT"
  --orientation-weight "$ORIENTATION_WEIGHT"
  --orientation-detail-weight "$ORIENTATION_DETAIL_WEIGHT"
  --smooth-weight "$SMOOTH_WEIGHT"
  --strand-shape-smooth-weight "$STRAND_SHAPE_SMOOTH_WEIGHT"
  --shape-prior-weight "$SHAPE_PRIOR_WEIGHT"
  --root-move-reg-weight "$ROOT_MOVE_REG_WEIGHT"
  --orientation-min-confidence "$ORIENTATION_MIN_CONFIDENCE"
  --backing-color-min "$BACKING_COLOR_MIN"
  --backing-color-max "$BACKING_COLOR_MAX"
  --mesh-depth-abs-tolerance "$MESH_DEPTH_ABS_TOLERANCE"
  --mesh-depth-rel-tolerance "$MESH_DEPTH_REL_TOLERANCE"
  --mesh-depth-local-kernel "$MESH_DEPTH_LOCAL_KERNEL"
  --densify-warmup "$DENSIFY_WARMUP"
  --densify-interval "$DENSIFY_INTERVAL"
  --densify-until "$DENSIFY_UNTIL"
  --densify-score-threshold "$DENSIFY_SCORE_THRESHOLD"
  --densify-min-contribution "$DENSIFY_MIN_CONTRIBUTION"
  --max-splits-per-event "$MAX_SPLITS_PER_EVENT"
  --split-children-per-parent "$SPLIT_CHILDREN_PER_PARENT"
  --split-neighbor-count "$SPLIT_NEIGHBOR_COUNT"
  --split-candidate-rings "$SPLIT_CANDIDATE_RINGS"
  --split-candidate-face-count "$SPLIT_CANDIDATE_FACE_COUNT"
  --split-min-child-distance "$SPLIT_MIN_CHILD_DISTANCE"
  --prune-start "$PRUNE_START"
  --prune-interval "$PRUNE_INTERVAL"
  --prune-min-contribution "$PRUNE_MIN_CONTRIBUTION"
  --prune-min-opacity "$PRUNE_MIN_OPACITY"
  --prune-max-fraction "$PRUNE_MAX_FRACTION"
)

if [[ "$RANDOM_BACKING_COLOR" == "0" ]]; then
  cmd+=(--disable-random-backing-color)
fi
if [[ "$MESH_DEPTH_CLIPPING" == "0" ]]; then
  cmd+=(--disable-mesh-depth-clipping)
fi
if [[ "$MESH_BACKING_COMPOSITING" == "0" ]]; then
  cmd+=(--disable-mesh-backing-compositing)
fi
if [[ -n "${TRAIN_VIEWS:-}" ]]; then
  cmd+=(--train-views "$TRAIN_VIEWS")
fi
if [[ -n "${TEST_VIEWS:-}" ]]; then
  cmd+=(--test-views "$TEST_VIEWS")
fi

if [[ "${RUN_BATCH_PREFLIGHT:-1}" != "0" ]]; then
  PREFLIGHT_VIEW="${PREFLIGHT_VIEW:-9}"
  PREFLIGHT_OUTPUT_DIR="${PREFLIGHT_OUTPUT_DIR:-${OUTPUT_DIR}_batch_preflight}"
  preflight_cmd=("${cmd[@]}")
  preflight_cmd+=(
    --output-dir "$PREFLIGHT_OUTPUT_DIR"
    --iterations 1
    --eval-every 1
    --save-every 0
    --train-views "$PREFLIGHT_VIEW"
    --test-views "$PREFLIGHT_VIEW"
  )
  printf '[stage1] batch preflight command:'
  printf ' %q' "${preflight_cmd[@]}"
  printf '\n'
  "${preflight_cmd[@]}"
fi

printf '[stage1] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

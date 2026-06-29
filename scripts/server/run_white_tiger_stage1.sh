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
  INIT_GROOM_LENGTH
  INIT_GUIDE_LENGTH
  SAMPLES
  MIN_SEGMENTS
  MAX_SEGMENTS
  CHILD_COUNT
  GAUSSIAN_LENGTH_OVERLAP
  PROJECTED_INIT_VIEWS
  PROJECTED_INIT_MIN_CONFIDENCE
  GUIDE_ROOT_COUNT
  GUIDE_CANDIDATE_MULTIPLIER
  GUIDE_INTERPOLATION_K
  GUIDE_CONTROLS_FLOW
  GUIDE_LENGTH_RESIDUAL_SCALE
  GUIDE_BEND_RESIDUAL_SCALE
  GUIDE_FLOW_RESIDUAL_SCALE
  GUIDE_WIDTH_RESIDUAL_SCALE
  GUIDE_FLOW_STRENGTH_RESIDUAL_SCALE
  GUIDE_LIFT_RESIDUAL_SCALE
  GUIDE_STIFFNESS_RESIDUAL_SCALE
  GUIDE_CHILD_RADIUS_RESIDUAL_SCALE
  GUIDE_CLUMP_RESIDUAL_SCALE
  GUIDE_CURL_RESIDUAL_SCALE
  GUIDE_FRIZZ_RESIDUAL_SCALE
  GUIDE_PRIOR_WEIGHT
  GUIDE_PRIOR_FLOW_WEIGHT
  GUIDE_PRIOR_BEND_WEIGHT
  GUIDE_PRIOR_LIFT_WEIGHT
  GUIDE_PRIOR_STIFFNESS_WEIGHT
  GUIDE_PRIOR_CURL_WEIGHT
  GUIDE_PRIOR_LENGTH_WEIGHT
  GUIDE_PRIOR_WIDTH_WEIGHT
  GUIDE_PRIOR_CHILD_RADIUS_WEIGHT
  GUIDE_PRIOR_CLUMP_WEIGHT
  GUIDE_SMOOTH_WEIGHT
  GUIDE_RESIDUAL_UNLOCK_START
  GUIDE_RESIDUAL_UNLOCK_END
  GUIDE_RESIDUAL_INITIAL_MULTIPLIER
  GUIDE_FREEZE_UNTIL
  GUIDE_DENSIFY_START
  GUIDE_DENSIFY_INTERVAL
  GUIDE_DENSIFY_UNTIL
  GUIDE_DENSIFY_SCORE_THRESHOLD
  GUIDE_DENSIFY_MAX_SPLITS_PER_EVENT
  GUIDE_DENSIFY_CHILDREN_PER_PARENT
  GUIDE_DENSIFY_NEIGHBOR_COUNT
  GUIDE_DENSIFY_CANDIDATE_RINGS
  GUIDE_DENSIFY_CANDIDATE_FACE_COUNT
  GUIDE_DENSIFY_MIN_CHILD_DISTANCE
  GUIDE_DENSIFY_RENDER_ROOT_K
  LR_GROOM
  LR_HIGH_FREQUENCY_SHAPE_SCALE
  LR_COLOR
  COLOR_FREEZE_UNTIL
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
  EFFECTIVE_GEOMETRY_BUDGET_WEIGHT
  EFFECTIVE_LENGTH_TARGET
  EFFECTIVE_WIDTH_TARGET
  EFFECTIVE_CHILD_RADIUS_TARGET
  OVERPAINT_CAPACITY_WEIGHT
  OVERPAINT_RESIDUAL_THRESHOLD
  OVERPAINT_LENGTH_TARGET
  OVERPAINT_WIDTH_TARGET
  OVERPAINT_OPACITY_TARGET
  DARK_STROKE_CAPACITY_WEIGHT
  DARK_STROKE_LUMA_THRESHOLD
  DARK_STROKE_LENGTH_TARGET
  DARK_STROKE_WIDTH_TARGET
  DARK_STROKE_CHILD_RADIUS_TARGET
  DARK_STROKE_CLUMP_TARGET
  DARK_STROKE_OPACITY_TARGET
  SCREEN_STROKE_CAPACITY_WEIGHT
  SCREEN_STROKE_LUMA_THRESHOLD
  SCREEN_STROKE_DIAG_THRESHOLD
  SCREEN_STROKE_LENGTH_TARGET
  SCREEN_STROKE_WIDTH_TARGET
  SCREEN_STROKE_OPACITY_TARGET
  NEUTRAL_SCREEN_CAPACITY_WEIGHT
  NEUTRAL_SCREEN_LUMA_THRESHOLD
  NEUTRAL_SCREEN_DIAG_THRESHOLD
  NEUTRAL_SCREEN_LENGTH_TARGET
  NEUTRAL_SCREEN_WIDTH_TARGET
  NEUTRAL_SCREEN_OPACITY_TARGET
  COLOR_CONTRAST_CAPACITY_WEIGHT
  COLOR_CONTRAST_THRESHOLD
  COLOR_CONTRAST_LENGTH_TARGET
  COLOR_CONTRAST_WIDTH_TARGET
  COLOR_CONTRAST_OPACITY_TARGET
  EARLY_CAPACITY_WEIGHT
  EARLY_CAPACITY_UNTIL
  EARLY_CAPACITY_LENGTH_TARGET
  EARLY_CAPACITY_WIDTH_TARGET
  EARLY_CAPACITY_OPACITY_TARGET
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
  DENSIFY_RESIDUAL_WEIGHT
  DENSIFY_RESIDUAL_MODE
  DENSIFY_RESIDUAL_POOL_RADIUS
  DENSIFY_RESIDUAL_ALPHA_WEIGHT
  DENSIFY_RESIDUAL_RGB_WEIGHT
  DENSIFY_PIXEL_EVIDENCE_TOPK
  DENSIFY_PIXEL_EVIDENCE_ROOT_K
  DENSIFY_PIXEL_EVIDENCE_MIN
  DENSIFY_PIXEL_EVIDENCE_CHUNK
  DENSIFY_PARENT_SELECTION
  DENSIFY_TARGET_PLACEMENT_WEIGHT
  LIFECYCLE_SCORE_MODE
  STROKE_DRAG_DIAGNOSTICS
  LOCAL_CHILD_COLOR_SUPPORT
  LOCAL_CHILD_OPACITY_SUPPORT
  LOCAL_CHILD_COLOR_SCALE
  LOCAL_CHILD_OPACITY_SCALE
  MAX_SPLITS_PER_EVENT
  SPLIT_CHILDREN_PER_PARENT
  SPLIT_NEIGHBOR_COUNT
  SPLIT_CANDIDATE_RINGS
  SPLIT_CANDIDATE_FACE_COUNT
  SPLIT_MIN_CHILD_DISTANCE
  OVERLONG_SPLIT_LENGTH_THRESHOLD
  OVERLONG_SPLIT_WIDTH_THRESHOLD
  OVERLONG_SPLIT_OPACITY_THRESHOLD
  OVERLONG_SPLIT_MIN_CONTRIBUTION
  OVERLONG_SPLIT_MAX_PARENTS_PER_EVENT
  OVERLONG_SPLIT_CHILDREN_PER_PARENT
  OVERLONG_SPLIT_TARGET_DISTANCE
  OVERLONG_SPLIT_TARGET_WEIGHT
  OVERLONG_SPLIT_RESIDUAL_TARGET_WEIGHT
  OVERLONG_SPLIT_UNTIL
  OVERLONG_SPLIT_REPLACE_PARENT
  OVERLONG_SPLIT_CHILD_LENGTH_SCALE
  OVERLONG_SPLIT_CHILD_WIDTH_SCALE
  OVERLONG_SPLIT_CHILD_OPACITY_SCALE
  OVERLONG_SPLIT_CHILD_SPREAD_SCALE
  OVERLONG_SPLIT_CHILD_CLUMP_MIN
  SCREEN_FOOTPRINT_SPLIT_DIAG_THRESHOLD
  SCREEN_FOOTPRINT_SPLIT_LUMA_THRESHOLD
  SCREEN_FOOTPRINT_SPLIT_SCORE_WEIGHT
  SCREEN_FOOTPRINT_SPLIT_EXTRA_PARENTS_PER_EVENT
  SCREEN_FOOTPRINT_SPLIT_NEUTRAL_EXTRA_PARENTS_PER_EVENT
  SCREEN_FOOTPRINT_SPLIT_REPLACE_PARENT
  SCREEN_FOOTPRINT_SPLIT_CHILD_LENGTH_SCALE
  SCREEN_FOOTPRINT_SPLIT_CHILD_WIDTH_SCALE
  SCREEN_FOOTPRINT_SPLIT_CHILD_OPACITY_SCALE
  SCREEN_FOOTPRINT_SPLIT_CHILD_SPREAD_SCALE
  SCREEN_FOOTPRINT_SPLIT_CHILD_CLUMP_MIN
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
  --init-groom-length "$INIT_GROOM_LENGTH"
  --init-guide-length "$INIT_GUIDE_LENGTH"
  --samples "$SAMPLES"
  --min-segments "$MIN_SEGMENTS"
  --max-segments "$MAX_SEGMENTS"
  --child-count "$CHILD_COUNT"
  --gaussian-length-overlap "$GAUSSIAN_LENGTH_OVERLAP"
  --projected-init-views "$PROJECTED_INIT_VIEWS"
  --projected-init-min-confidence "$PROJECTED_INIT_MIN_CONFIDENCE"
  --guide-root-count "$GUIDE_ROOT_COUNT"
  --guide-candidate-multiplier "$GUIDE_CANDIDATE_MULTIPLIER"
  --guide-interpolation-k "$GUIDE_INTERPOLATION_K"
  --guide-length-residual-scale "$GUIDE_LENGTH_RESIDUAL_SCALE"
  --guide-bend-residual-scale "$GUIDE_BEND_RESIDUAL_SCALE"
  --guide-flow-residual-scale "$GUIDE_FLOW_RESIDUAL_SCALE"
  --guide-width-residual-scale "$GUIDE_WIDTH_RESIDUAL_SCALE"
  --guide-flow-strength-residual-scale "$GUIDE_FLOW_STRENGTH_RESIDUAL_SCALE"
  --guide-lift-residual-scale "$GUIDE_LIFT_RESIDUAL_SCALE"
  --guide-stiffness-residual-scale "$GUIDE_STIFFNESS_RESIDUAL_SCALE"
  --guide-child-radius-residual-scale "$GUIDE_CHILD_RADIUS_RESIDUAL_SCALE"
  --guide-clump-residual-scale "$GUIDE_CLUMP_RESIDUAL_SCALE"
  --guide-curl-residual-scale "$GUIDE_CURL_RESIDUAL_SCALE"
  --guide-frizz-residual-scale "$GUIDE_FRIZZ_RESIDUAL_SCALE"
  --guide-prior-weight "$GUIDE_PRIOR_WEIGHT"
  --guide-prior-flow-weight "$GUIDE_PRIOR_FLOW_WEIGHT"
  --guide-prior-bend-weight "$GUIDE_PRIOR_BEND_WEIGHT"
  --guide-prior-lift-weight "$GUIDE_PRIOR_LIFT_WEIGHT"
  --guide-prior-stiffness-weight "$GUIDE_PRIOR_STIFFNESS_WEIGHT"
  --guide-prior-curl-weight "$GUIDE_PRIOR_CURL_WEIGHT"
  --guide-prior-length-weight "$GUIDE_PRIOR_LENGTH_WEIGHT"
  --guide-prior-width-weight "$GUIDE_PRIOR_WIDTH_WEIGHT"
  --guide-prior-child-radius-weight "$GUIDE_PRIOR_CHILD_RADIUS_WEIGHT"
  --guide-prior-clump-weight "$GUIDE_PRIOR_CLUMP_WEIGHT"
  --guide-smooth-weight "$GUIDE_SMOOTH_WEIGHT"
  --guide-residual-unlock-start "$GUIDE_RESIDUAL_UNLOCK_START"
  --guide-residual-unlock-end "$GUIDE_RESIDUAL_UNLOCK_END"
  --guide-residual-initial-multiplier "$GUIDE_RESIDUAL_INITIAL_MULTIPLIER"
  --guide-freeze-until "$GUIDE_FREEZE_UNTIL"
  --guide-densify-start "$GUIDE_DENSIFY_START"
  --guide-densify-interval "$GUIDE_DENSIFY_INTERVAL"
  --guide-densify-until "$GUIDE_DENSIFY_UNTIL"
  --guide-densify-score-threshold "$GUIDE_DENSIFY_SCORE_THRESHOLD"
  --guide-densify-max-splits-per-event "$GUIDE_DENSIFY_MAX_SPLITS_PER_EVENT"
  --guide-densify-children-per-parent "$GUIDE_DENSIFY_CHILDREN_PER_PARENT"
  --guide-densify-neighbor-count "$GUIDE_DENSIFY_NEIGHBOR_COUNT"
  --guide-densify-candidate-rings "$GUIDE_DENSIFY_CANDIDATE_RINGS"
  --guide-densify-candidate-face-count "$GUIDE_DENSIFY_CANDIDATE_FACE_COUNT"
  --guide-densify-min-child-distance "$GUIDE_DENSIFY_MIN_CHILD_DISTANCE"
  --guide-densify-render-root-k "$GUIDE_DENSIFY_RENDER_ROOT_K"
  --lr-groom "$LR_GROOM"
  --lr-high-frequency-shape-scale "$LR_HIGH_FREQUENCY_SHAPE_SCALE"
  --lr-color "$LR_COLOR"
  --color-freeze-until "$COLOR_FREEZE_UNTIL"
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
  --effective-geometry-budget-weight "$EFFECTIVE_GEOMETRY_BUDGET_WEIGHT"
  --effective-length-target "$EFFECTIVE_LENGTH_TARGET"
  --effective-width-target "$EFFECTIVE_WIDTH_TARGET"
  --effective-child-radius-target "$EFFECTIVE_CHILD_RADIUS_TARGET"
  --overpaint-capacity-weight "$OVERPAINT_CAPACITY_WEIGHT"
  --overpaint-residual-threshold "$OVERPAINT_RESIDUAL_THRESHOLD"
  --overpaint-length-target "$OVERPAINT_LENGTH_TARGET"
  --overpaint-width-target "$OVERPAINT_WIDTH_TARGET"
  --overpaint-opacity-target "$OVERPAINT_OPACITY_TARGET"
  --dark-stroke-capacity-weight "$DARK_STROKE_CAPACITY_WEIGHT"
  --dark-stroke-luma-threshold "$DARK_STROKE_LUMA_THRESHOLD"
  --dark-stroke-length-target "$DARK_STROKE_LENGTH_TARGET"
  --dark-stroke-width-target "$DARK_STROKE_WIDTH_TARGET"
  --dark-stroke-child-radius-target "$DARK_STROKE_CHILD_RADIUS_TARGET"
  --dark-stroke-clump-target "$DARK_STROKE_CLUMP_TARGET"
  --dark-stroke-opacity-target "$DARK_STROKE_OPACITY_TARGET"
  --screen-stroke-capacity-weight "$SCREEN_STROKE_CAPACITY_WEIGHT"
  --screen-stroke-luma-threshold "$SCREEN_STROKE_LUMA_THRESHOLD"
  --screen-stroke-diag-threshold "$SCREEN_STROKE_DIAG_THRESHOLD"
  --screen-stroke-length-target "$SCREEN_STROKE_LENGTH_TARGET"
  --screen-stroke-width-target "$SCREEN_STROKE_WIDTH_TARGET"
  --screen-stroke-opacity-target "$SCREEN_STROKE_OPACITY_TARGET"
  --neutral-screen-capacity-weight "$NEUTRAL_SCREEN_CAPACITY_WEIGHT"
  --neutral-screen-luma-threshold "$NEUTRAL_SCREEN_LUMA_THRESHOLD"
  --neutral-screen-diag-threshold "$NEUTRAL_SCREEN_DIAG_THRESHOLD"
  --neutral-screen-length-target "$NEUTRAL_SCREEN_LENGTH_TARGET"
  --neutral-screen-width-target "$NEUTRAL_SCREEN_WIDTH_TARGET"
  --neutral-screen-opacity-target "$NEUTRAL_SCREEN_OPACITY_TARGET"
  --color-contrast-capacity-weight "$COLOR_CONTRAST_CAPACITY_WEIGHT"
  --color-contrast-threshold "$COLOR_CONTRAST_THRESHOLD"
  --color-contrast-length-target "$COLOR_CONTRAST_LENGTH_TARGET"
  --color-contrast-width-target "$COLOR_CONTRAST_WIDTH_TARGET"
  --color-contrast-opacity-target "$COLOR_CONTRAST_OPACITY_TARGET"
  --early-capacity-weight "$EARLY_CAPACITY_WEIGHT"
  --early-capacity-until "$EARLY_CAPACITY_UNTIL"
  --early-capacity-length-target "$EARLY_CAPACITY_LENGTH_TARGET"
  --early-capacity-width-target "$EARLY_CAPACITY_WIDTH_TARGET"
  --early-capacity-opacity-target "$EARLY_CAPACITY_OPACITY_TARGET"
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
  --densify-residual-weight "$DENSIFY_RESIDUAL_WEIGHT"
  --densify-residual-mode "$DENSIFY_RESIDUAL_MODE"
  --densify-residual-pool-radius "$DENSIFY_RESIDUAL_POOL_RADIUS"
  --densify-residual-alpha-weight "$DENSIFY_RESIDUAL_ALPHA_WEIGHT"
  --densify-residual-rgb-weight "$DENSIFY_RESIDUAL_RGB_WEIGHT"
  --densify-pixel-evidence-topk "$DENSIFY_PIXEL_EVIDENCE_TOPK"
  --densify-pixel-evidence-root-k "$DENSIFY_PIXEL_EVIDENCE_ROOT_K"
  --densify-pixel-evidence-min "$DENSIFY_PIXEL_EVIDENCE_MIN"
  --densify-pixel-evidence-chunk "$DENSIFY_PIXEL_EVIDENCE_CHUNK"
  --densify-parent-selection "$DENSIFY_PARENT_SELECTION"
  --densify-target-placement-weight "$DENSIFY_TARGET_PLACEMENT_WEIGHT"
  --lifecycle-score-mode "$LIFECYCLE_SCORE_MODE"
  --local-child-color-scale "$LOCAL_CHILD_COLOR_SCALE"
  --local-child-opacity-scale "$LOCAL_CHILD_OPACITY_SCALE"
  --max-splits-per-event "$MAX_SPLITS_PER_EVENT"
  --split-children-per-parent "$SPLIT_CHILDREN_PER_PARENT"
  --split-neighbor-count "$SPLIT_NEIGHBOR_COUNT"
  --split-candidate-rings "$SPLIT_CANDIDATE_RINGS"
  --split-candidate-face-count "$SPLIT_CANDIDATE_FACE_COUNT"
  --split-min-child-distance "$SPLIT_MIN_CHILD_DISTANCE"
  --overlong-split-length-threshold "$OVERLONG_SPLIT_LENGTH_THRESHOLD"
  --overlong-split-width-threshold "$OVERLONG_SPLIT_WIDTH_THRESHOLD"
  --overlong-split-opacity-threshold "$OVERLONG_SPLIT_OPACITY_THRESHOLD"
  --overlong-split-min-contribution "$OVERLONG_SPLIT_MIN_CONTRIBUTION"
  --overlong-split-max-parents-per-event "$OVERLONG_SPLIT_MAX_PARENTS_PER_EVENT"
  --overlong-split-children-per-parent "$OVERLONG_SPLIT_CHILDREN_PER_PARENT"
  --overlong-split-target-distance "$OVERLONG_SPLIT_TARGET_DISTANCE"
  --overlong-split-target-weight "$OVERLONG_SPLIT_TARGET_WEIGHT"
  --overlong-split-residual-target-weight "$OVERLONG_SPLIT_RESIDUAL_TARGET_WEIGHT"
  --overlong-split-until "$OVERLONG_SPLIT_UNTIL"
  --overlong-split-replace-parent "$OVERLONG_SPLIT_REPLACE_PARENT"
  --overlong-split-child-length-scale "$OVERLONG_SPLIT_CHILD_LENGTH_SCALE"
  --overlong-split-child-width-scale "$OVERLONG_SPLIT_CHILD_WIDTH_SCALE"
  --overlong-split-child-opacity-scale "$OVERLONG_SPLIT_CHILD_OPACITY_SCALE"
  --overlong-split-child-spread-scale "$OVERLONG_SPLIT_CHILD_SPREAD_SCALE"
  --overlong-split-child-clump-min "$OVERLONG_SPLIT_CHILD_CLUMP_MIN"
  --screen-footprint-split-diag-threshold "$SCREEN_FOOTPRINT_SPLIT_DIAG_THRESHOLD"
  --screen-footprint-split-luma-threshold "$SCREEN_FOOTPRINT_SPLIT_LUMA_THRESHOLD"
  --screen-footprint-split-score-weight "$SCREEN_FOOTPRINT_SPLIT_SCORE_WEIGHT"
  --screen-footprint-split-extra-parents-per-event "$SCREEN_FOOTPRINT_SPLIT_EXTRA_PARENTS_PER_EVENT"
  --screen-footprint-split-neutral-extra-parents-per-event "$SCREEN_FOOTPRINT_SPLIT_NEUTRAL_EXTRA_PARENTS_PER_EVENT"
  --screen-footprint-split-replace-parent "$SCREEN_FOOTPRINT_SPLIT_REPLACE_PARENT"
  --screen-footprint-split-child-length-scale "$SCREEN_FOOTPRINT_SPLIT_CHILD_LENGTH_SCALE"
  --screen-footprint-split-child-width-scale "$SCREEN_FOOTPRINT_SPLIT_CHILD_WIDTH_SCALE"
  --screen-footprint-split-child-opacity-scale "$SCREEN_FOOTPRINT_SPLIT_CHILD_OPACITY_SCALE"
  --screen-footprint-split-child-spread-scale "$SCREEN_FOOTPRINT_SPLIT_CHILD_SPREAD_SCALE"
  --screen-footprint-split-child-clump-min "$SCREEN_FOOTPRINT_SPLIT_CHILD_CLUMP_MIN"
  --prune-start "$PRUNE_START"
  --prune-interval "$PRUNE_INTERVAL"
  --prune-min-contribution "$PRUNE_MIN_CONTRIBUTION"
  --prune-min-opacity "$PRUNE_MIN_OPACITY"
  --prune-max-fraction "$PRUNE_MAX_FRACTION"
)

if [[ "$RANDOM_BACKING_COLOR" == "0" ]]; then
  cmd+=(--disable-random-backing-color)
fi
if [[ "$GUIDE_CONTROLS_FLOW" == "1" ]]; then
  cmd+=(--guide-controls-flow)
fi
if [[ "$MESH_DEPTH_CLIPPING" == "0" ]]; then
  cmd+=(--disable-mesh-depth-clipping)
fi
if [[ "$MESH_BACKING_COMPOSITING" == "0" ]]; then
  cmd+=(--disable-mesh-backing-compositing)
fi
if [[ "$STROKE_DRAG_DIAGNOSTICS" == "1" ]]; then
  cmd+=(--stroke-drag-diagnostics)
fi
if [[ "$LOCAL_CHILD_COLOR_SUPPORT" == "1" ]]; then
  cmd+=(--local-child-color-support)
fi
if [[ "$LOCAL_CHILD_OPACITY_SUPPORT" == "1" ]]; then
  cmd+=(--local-child-opacity-support)
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
  if [[ "${STAGE1_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[stage1] STAGE1_PREFLIGHT_ONLY=1; stopping after batch preflight"
    exit 0
  fi
fi

printf '[stage1] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

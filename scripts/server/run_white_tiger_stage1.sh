#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
EXPLICIT_DATA_ROOT="${DATA_ROOT+x}"
EXPLICIT_MESH_PATH="${MESH_PATH+x}"
EXPLICIT_CLEAN_FLOW_TARGET="${CLEAN_FLOW_TARGET+x}"
DATA_ROOT_OVERRIDE="${DATA_ROOT:-}"
MESH_PATH_OVERRIDE="${MESH_PATH:-}"
CLEAN_FLOW_TARGET_OVERRIDE="${CLEAN_FLOW_TARGET:-}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/neuralfur_work/whiteTiger_processed/roaringwalk}"
MESH_PATH="${MESH_PATH:-${PROJECT_ROOT}/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
CONFIG_PATH="${CONFIG_PATH:-}"

if [[ -z "$CONFIG_PATH" ]]; then
  echo "[stage1] CONFIG_PATH must be set explicitly; no historical default is allowed" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[stage1] explicit CONFIG_PATH does not exist: $CONFIG_PATH" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$CONFIG_PATH"
if [[ -n "$EXPLICIT_DATA_ROOT" ]]; then
  DATA_ROOT="$DATA_ROOT_OVERRIDE"
fi
if [[ -n "$EXPLICIT_MESH_PATH" ]]; then
  MESH_PATH="$MESH_PATH_OVERRIDE"
fi
if [[ -n "$EXPLICIT_CLEAN_FLOW_TARGET" ]]; then
  CLEAN_FLOW_TARGET="$CLEAN_FLOW_TARGET_OVERRIDE"
fi

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
  ROOT_INIT_METHOD
  CANDIDATE_MULTIPLIER
  ITERATIONS
  EVAL_EVERY
  SAVE_EVERY
  TEST_STRIDE
  EXPECTED_WIDTH
  EXPECTED_HEIGHT
  INIT_GROOM_LENGTH
  SAMPLES
  MIN_SEGMENTS
  SEGMENT_LENGTH_ORIGIN
  SEGMENTS_PER_UNIT_LENGTH
  SEGMENTS_PER_UNIT_COMPLEXITY
  CHILD_COUNT
  GAUSSIAN_LENGTH_OVERLAP
  PROJECTED_INIT_VIEWS
  PROJECTED_INIT_MIN_CONFIDENCE
  CLEAN_FLOW_INIT
  CLEAN_FLOW_INIT_K
  CLEAN_FLOW_INIT_MIN_CONFIDENCE
  CLEAN_FLOW_ANCHOR_MIN_CONFIDENCE
  CLEAN_FLOW_LENGTH_INIT
  CLEAN_FLOW_LENGTH_INIT_SCALE
  CLEAN_FLOW_LENGTH_INIT_MIN_CONFIDENCE
  CLEAN_FLOW_GUIDE_ANCHOR_WEIGHT
  CLEAN_FLOW_3D_SMOOTH_WEIGHT
  GUIDE_ROOT_COUNT
  GUIDE_CANDIDATE_MULTIPLIER
  GUIDE_ROOTS_FROM_CLEAN_FLOW
  GUIDE_INTERPOLATION_K
  RENDER_GEOMETRY_PARAMETERIZATION
  GUIDE_LENGTH_RESIDUAL_SCALE
  GUIDE_DIRECTION_RESIDUAL_SCALE
  GUIDE_WIDTH_RESIDUAL_SCALE
  GUIDE_CHILD_RADIUS_RESIDUAL_SCALE
  GUIDE_CLUMP_RESIDUAL_SCALE
  GUIDE_CURL_RESIDUAL_SCALE
  GUIDE_FRIZZ_RESIDUAL_SCALE
  GUIDE_PRIOR_WEIGHT
  GUIDE_PRIOR_DIRECTION_WEIGHT
  GUIDE_PRIOR_CURL_WEIGHT
  GUIDE_PRIOR_LENGTH_WEIGHT
  GUIDE_PRIOR_WIDTH_WEIGHT
  GUIDE_PRIOR_CHILD_RADIUS_WEIGHT
  GUIDE_PRIOR_CLUMP_WEIGHT
  RENDER_LENGTH_PRIOR_COORDINATE
  RENDER_LENGTH_PRIOR_REDUCTION
  GUIDE_SMOOTH_WEIGHT
  GUIDE_LENGTH_SMOOTH_MODE
  GUIDE_RESIDUAL_UNLOCK_START
  GUIDE_RESIDUAL_UNLOCK_END
  GUIDE_RESIDUAL_INITIAL_MULTIPLIER
  GUIDE_COVERAGE_RESIDUAL_UNLOCK_START
  GUIDE_COVERAGE_RESIDUAL_UNLOCK_END
  GUIDE_COVERAGE_RESIDUAL_INITIAL_MULTIPLIER
  GUIDE_FREEZE_UNTIL
  SHAPE_DETAIL_FREEZE_UNTIL
  SHAPE_CURL_SCALE
  SHAPE_FRIZZ_SCALE
  GUIDE_DENSIFY_START
  GUIDE_DENSIFY_INTERVAL
  GUIDE_DENSIFY_UNTIL
  GUIDE_DENSIFY_SCORE_THRESHOLD
  GUIDE_DENSIFY_MAX_SPLITS_PER_EVENT
  GUIDE_DENSIFY_POLICY
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
  RGB_FLOW_WEIGHT
  RGB_FLOW_DETAIL_WEIGHT
  RGB_FLOW_MIN_CONFIDENCE
  LOSS_MASK_EDGE_KERNEL
  SMOOTH_GRAPH_MODE
  SMOOTH_GRAPH_K
  SMOOTH_FIELD_METRIC
  SMOOTH_WEIGHT
  EFFECTIVE_SMOOTH_WEIGHT
  ROOT_MOVE_REG_WEIGHT
  RANDOM_BACKING_COLOR
  BACKING_COLOR_MIN
  BACKING_COLOR_MAX
  RANDOM_MESH_BACKING_TEXTURE
  MESH_BACKING_TEXTURE_STRENGTH
  MESH_BACKING_TEXTURE_OCTAVES
  MESH_DEPTH_CLIPPING
  MESH_DEPTH_ABS_TOLERANCE
  MESH_DEPTH_REL_TOLERANCE
  MESH_DEPTH_LOCAL_KERNEL
  MESH_BACKING_COMPOSITING
  STRAND_SHAPE_NORMAL_MODE
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
  LIFECYCLE_SCORE_MODE
  LOCAL_CHILD_COLOR_SUPPORT
  LOCAL_CHILD_COLOR_SCALE
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

GAUSSIAN_RGB_RESIDUAL_SUPPORT="${GAUSSIAN_RGB_RESIDUAL_SUPPORT:-0}"
if [[ "$GAUSSIAN_RGB_RESIDUAL_SUPPORT" == "1" ]]; then
  for name in \
    GAUSSIAN_RGB_RESIDUAL_CONTROL_POINTS \
    GAUSSIAN_RGB_RESIDUAL_SCALE \
    GAUSSIAN_RGB_RESIDUAL_UNLOCK_START \
    GAUSSIAN_RGB_RESIDUAL_UNLOCK_END \
    GAUSSIAN_RGB_RESIDUAL_INITIAL_MULTIPLIER; do
    require_var "$name"
  done
else
  GAUSSIAN_RGB_RESIDUAL_CONTROL_POINTS=36
  GAUSSIAN_RGB_RESIDUAL_SCALE=0.20
  GAUSSIAN_RGB_RESIDUAL_UNLOCK_START=10000
  GAUSSIAN_RGB_RESIDUAL_UNLOCK_END=20000
  GAUSSIAN_RGB_RESIDUAL_INITIAL_MULTIPLIER=0.0
fi

GEOMETRY_RESIDUAL_DOMAIN="${GEOMETRY_RESIDUAL_DOMAIN:-render}"
GEOMETRY_RESIDUAL_SMOOTH_SCALE="${GEOMETRY_RESIDUAL_SMOOTH_SCALE:-1.0}"
SECONDARY_GUIDE_COLOR_SUPPORT="${SECONDARY_GUIDE_COLOR_SUPPORT:-0}"
if [[ "$GEOMETRY_RESIDUAL_DOMAIN" == "secondary_guide" ]]; then
  for name in \
    SECONDARY_GUIDE_ROOT_COUNT \
    SECONDARY_GUIDE_CANDIDATE_MULTIPLIER \
    SECONDARY_GUIDE_INTERPOLATION_K \
    SECONDARY_GUIDE_SMOOTH_K; do
    require_var "$name"
  done
else
  SECONDARY_GUIDE_ROOT_COUNT=0
  SECONDARY_GUIDE_CANDIDATE_MULTIPLIER=16
  SECONDARY_GUIDE_INTERPOLATION_K=8
  SECONDARY_GUIDE_SMOOTH_K=32
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export PROJECT_ROOT PYTHON DATA_ROOT MESH_PATH RUN_ID OUTPUT_DIR EXPECTED_WIDTH EXPECTED_HEIGHT

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${RUN_PREFLIGHT:-1}" != "0" ]]; then
  echo "[stage1] preflight"
  bash scripts/server/preflight_white_tiger_stage1.sh
else
  echo "[stage1] RUN_PREFLIGHT=0; skipping environment/data preflight"
fi

echo "[stage1] output_dir=${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

cmd=(
  "$PYTHON" tools/train_white_tiger_stage1.py
  --data-root "$DATA_ROOT"
  --mesh-path "$MESH_PATH"
  --output-dir "$OUTPUT_DIR"
  --root-count "$ROOT_COUNT"
  --root-init-method "$ROOT_INIT_METHOD"
  --candidate-multiplier "$CANDIDATE_MULTIPLIER"
  --iterations "$ITERATIONS"
  --eval-every "$EVAL_EVERY"
  --save-every "$SAVE_EVERY"
  --stage-save-iters "${STAGE_SAVE_ITERS:-}"
  --test-stride "$TEST_STRIDE"
  --expected-width "$EXPECTED_WIDTH"
  --expected-height "$EXPECTED_HEIGHT"
  --init-groom-length "$INIT_GROOM_LENGTH"
  --samples "$SAMPLES"
  --min-segments "$MIN_SEGMENTS"
  --segment-length-origin "$SEGMENT_LENGTH_ORIGIN"
  --segments-per-unit-length "$SEGMENTS_PER_UNIT_LENGTH"
  --segments-per-unit-complexity "$SEGMENTS_PER_UNIT_COMPLEXITY"
  --child-count "$CHILD_COUNT"
  --gaussian-length-overlap "$GAUSSIAN_LENGTH_OVERLAP"
  --projected-init-views "$PROJECTED_INIT_VIEWS"
  --projected-init-min-confidence "$PROJECTED_INIT_MIN_CONFIDENCE"
  --clean-flow-init-k "$CLEAN_FLOW_INIT_K"
  --clean-flow-init-min-confidence "$CLEAN_FLOW_INIT_MIN_CONFIDENCE"
  --clean-flow-anchor-min-confidence "$CLEAN_FLOW_ANCHOR_MIN_CONFIDENCE"
  --clean-flow-length-init-scale "$CLEAN_FLOW_LENGTH_INIT_SCALE"
  --clean-flow-length-init-min-confidence "$CLEAN_FLOW_LENGTH_INIT_MIN_CONFIDENCE"
  --clean-flow-guide-anchor-weight "$CLEAN_FLOW_GUIDE_ANCHOR_WEIGHT"
  --clean-flow-3d-smooth-weight "$CLEAN_FLOW_3D_SMOOTH_WEIGHT"
  --guide-root-count "$GUIDE_ROOT_COUNT"
  --guide-candidate-multiplier "$GUIDE_CANDIDATE_MULTIPLIER"
  --guide-interpolation-k "$GUIDE_INTERPOLATION_K"
  --geometry-residual-domain "$GEOMETRY_RESIDUAL_DOMAIN"
  --secondary-guide-root-count "$SECONDARY_GUIDE_ROOT_COUNT"
  --secondary-guide-candidate-multiplier "$SECONDARY_GUIDE_CANDIDATE_MULTIPLIER"
  --secondary-guide-interpolation-k "$SECONDARY_GUIDE_INTERPOLATION_K"
  --secondary-guide-smooth-k "$SECONDARY_GUIDE_SMOOTH_K"
  --render-geometry-parameterization "$RENDER_GEOMETRY_PARAMETERIZATION"
  --guide-length-residual-scale "$GUIDE_LENGTH_RESIDUAL_SCALE"
  --guide-direction-residual-scale "$GUIDE_DIRECTION_RESIDUAL_SCALE"
  --guide-width-residual-scale "$GUIDE_WIDTH_RESIDUAL_SCALE"
  --guide-child-radius-residual-scale "$GUIDE_CHILD_RADIUS_RESIDUAL_SCALE"
  --guide-clump-residual-scale "$GUIDE_CLUMP_RESIDUAL_SCALE"
  --guide-curl-residual-scale "$GUIDE_CURL_RESIDUAL_SCALE"
  --guide-frizz-residual-scale "$GUIDE_FRIZZ_RESIDUAL_SCALE"
  --guide-prior-weight "$GUIDE_PRIOR_WEIGHT"
  --guide-prior-direction-weight "$GUIDE_PRIOR_DIRECTION_WEIGHT"
  --guide-prior-curl-weight "$GUIDE_PRIOR_CURL_WEIGHT"
  --guide-prior-length-weight "$GUIDE_PRIOR_LENGTH_WEIGHT"
  --guide-prior-width-weight "$GUIDE_PRIOR_WIDTH_WEIGHT"
  --guide-prior-child-radius-weight "$GUIDE_PRIOR_CHILD_RADIUS_WEIGHT"
  --guide-prior-clump-weight "$GUIDE_PRIOR_CLUMP_WEIGHT"
  --render-length-prior-coordinate "$RENDER_LENGTH_PRIOR_COORDINATE"
  --render-length-prior-reduction "$RENDER_LENGTH_PRIOR_REDUCTION"
  --guide-smooth-weight "$GUIDE_SMOOTH_WEIGHT"
  --guide-length-smooth-mode "$GUIDE_LENGTH_SMOOTH_MODE"
  --guide-residual-unlock-start "$GUIDE_RESIDUAL_UNLOCK_START"
  --guide-residual-unlock-end "$GUIDE_RESIDUAL_UNLOCK_END"
  --guide-residual-initial-multiplier "$GUIDE_RESIDUAL_INITIAL_MULTIPLIER"
  --guide-coverage-residual-unlock-start "$GUIDE_COVERAGE_RESIDUAL_UNLOCK_START"
  --guide-coverage-residual-unlock-end "$GUIDE_COVERAGE_RESIDUAL_UNLOCK_END"
  --guide-coverage-residual-initial-multiplier "$GUIDE_COVERAGE_RESIDUAL_INITIAL_MULTIPLIER"
  --guide-freeze-until "$GUIDE_FREEZE_UNTIL"
  --shape-detail-freeze-until "$SHAPE_DETAIL_FREEZE_UNTIL"
  --shape-curl-scale "$SHAPE_CURL_SCALE"
  --shape-frizz-scale "$SHAPE_FRIZZ_SCALE"
  --guide-densify-start "$GUIDE_DENSIFY_START"
  --guide-densify-interval "$GUIDE_DENSIFY_INTERVAL"
  --guide-densify-until "$GUIDE_DENSIFY_UNTIL"
  --guide-densify-score-threshold "$GUIDE_DENSIFY_SCORE_THRESHOLD"
  --guide-densify-max-splits-per-event "$GUIDE_DENSIFY_MAX_SPLITS_PER_EVENT"
  --guide-densify-policy "$GUIDE_DENSIFY_POLICY"
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
  --gaussian-rgb-residual-control-points "$GAUSSIAN_RGB_RESIDUAL_CONTROL_POINTS"
  --gaussian-rgb-residual-scale "$GAUSSIAN_RGB_RESIDUAL_SCALE"
  --gaussian-rgb-residual-unlock-start "$GAUSSIAN_RGB_RESIDUAL_UNLOCK_START"
  --gaussian-rgb-residual-unlock-end "$GAUSSIAN_RGB_RESIDUAL_UNLOCK_END"
  --gaussian-rgb-residual-initial-multiplier "$GAUSSIAN_RGB_RESIDUAL_INITIAL_MULTIPLIER"
  --lr-root "$LR_ROOT"
  --lr-calibration "$LR_CALIBRATION"
  --rgb-weight "$RGB_WEIGHT"
  --random-backing-loss-weight "$RANDOM_BACKING_LOSS_WEIGHT"
  --mask-weight "$MASK_WEIGHT"
  --rgb-flow-weight "$RGB_FLOW_WEIGHT"
  --rgb-flow-detail-weight "$RGB_FLOW_DETAIL_WEIGHT"
  --rgb-flow-min-confidence "$RGB_FLOW_MIN_CONFIDENCE"
  --loss-mask-edge-kernel "$LOSS_MASK_EDGE_KERNEL"
  --smooth-graph-mode "$SMOOTH_GRAPH_MODE"
  --smooth-graph-k "$SMOOTH_GRAPH_K"
  --smooth-field-metric "$SMOOTH_FIELD_METRIC"
  --smooth-weight "$SMOOTH_WEIGHT"
  --geometry-residual-smooth-scale "$GEOMETRY_RESIDUAL_SMOOTH_SCALE"
  --effective-smooth-weight "$EFFECTIVE_SMOOTH_WEIGHT"
  --root-move-reg-weight "$ROOT_MOVE_REG_WEIGHT"
  --backing-color-min "$BACKING_COLOR_MIN"
  --backing-color-max "$BACKING_COLOR_MAX"
  --mesh-backing-texture-strength "$MESH_BACKING_TEXTURE_STRENGTH"
  --mesh-backing-texture-octaves "$MESH_BACKING_TEXTURE_OCTAVES"
  --mesh-depth-abs-tolerance "$MESH_DEPTH_ABS_TOLERANCE"
  --mesh-depth-rel-tolerance "$MESH_DEPTH_REL_TOLERANCE"
  --mesh-depth-local-kernel "$MESH_DEPTH_LOCAL_KERNEL"
  --strand-shape-normal-mode "$STRAND_SHAPE_NORMAL_MODE"
  --gpu-memory-limit-gb "${GPU_MEMORY_LIMIT_GB:-0}"
  --gpu-memory-check-interval "${GPU_MEMORY_CHECK_INTERVAL:-20}"
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
  --lifecycle-score-mode "$LIFECYCLE_SCORE_MODE"
  --local-child-color-scale "$LOCAL_CHILD_COLOR_SCALE"
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
if [[ "$RANDOM_MESH_BACKING_TEXTURE" == "0" ]]; then
  cmd+=(--disable-random-mesh-backing-texture)
fi
if [[ "$GUIDE_ROOTS_FROM_CLEAN_FLOW" == "1" ]]; then
  cmd+=(--guide-roots-from-clean-flow)
fi
if [[ -n "${CLEAN_FLOW_TARGET:-}" ]]; then
  cmd+=(--clean-flow-target "$CLEAN_FLOW_TARGET")
elif [[ "$CLEAN_FLOW_INIT" == "1" || "$CLEAN_FLOW_LENGTH_INIT" == "1" || "$CLEAN_FLOW_GUIDE_ANCHOR_WEIGHT" != "0" ]]; then
  echo "[stage1] clean-flow requested but CLEAN_FLOW_TARGET is empty" >&2
  exit 2
fi
if [[ "$CLEAN_FLOW_INIT" == "1" ]]; then
  cmd+=(--clean-flow-init)
fi
if [[ "$CLEAN_FLOW_LENGTH_INIT" == "1" ]]; then
  cmd+=(--clean-flow-length-init)
fi
if [[ "$MESH_DEPTH_CLIPPING" == "0" ]]; then
  cmd+=(--disable-mesh-depth-clipping)
fi
if [[ "$MESH_BACKING_COMPOSITING" == "0" ]]; then
  cmd+=(--disable-mesh-backing-compositing)
fi
if [[ "$LOCAL_CHILD_COLOR_SUPPORT" == "1" ]]; then
  cmd+=(--local-child-color-support)
fi
if [[ "$GAUSSIAN_RGB_RESIDUAL_SUPPORT" == "1" ]]; then
  cmd+=(--gaussian-rgb-residual-support)
fi
if [[ "$SECONDARY_GUIDE_COLOR_SUPPORT" == "1" ]]; then
  cmd+=(--secondary-guide-color-support)
fi
if [[ -n "${TRAIN_VIEWS:-}" ]]; then
  cmd+=(--train-views "$TRAIN_VIEWS")
fi
if [[ -n "${TEST_VIEWS:-}" ]]; then
  cmd+=(--test-views "$TEST_VIEWS")
fi
if [[ "${RESUME_OPTIMIZER:-1}" == "0" ]]; then
  cmd+=(--no-resume-optimizer)
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

if [[ "${STAGE1_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[stage1] STAGE1_PREFLIGHT_ONLY=1; stopping before training"
  exit 0
fi

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "[stage1] RESUME_CHECKPOINT does not exist: $RESUME_CHECKPOINT" >&2
    exit 2
  fi
  cmd+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

printf '[stage1] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
rc=$?
echo "[stage1] exit_code=$rc"
exit "$rc"

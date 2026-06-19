#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-outputs/white_tiger_uv_groom/${RUN_NAME}}"
TRIPLANE_ARGS=()
if [[ -n "${USE_TRIPLANE:-}" && "${USE_TRIPLANE:-0}" != "0" ]]; then
  TRIPLANE_ARGS+=(--use-triplane)
fi
COARSE_TEXTURE_ARGS=()
if [[ -n "${USE_COARSE_TEXTURE:-}" && "${USE_COARSE_TEXTURE:-0}" != "0" ]]; then
  COARSE_TEXTURE_ARGS+=(--use-coarse-texture)
fi
ROOT_RESIDUAL_BASE_ARGS=()
if [[ "${NO_ROOT_RESIDUAL_BASE:-0}" != "0" ]]; then
  ROOT_RESIDUAL_BASE_ARGS+=(--no-root-residual-base)
fi
if [[ -n "${COARSE_INIT_SPLIT:-}" && "${COARSE_INIT_SPLIT:-1}" == "0" ]]; then
  COARSE_TEXTURE_ARGS+=(--no-coarse-init-split)
fi
HEAD_ATLAS_ARGS=()
if [[ -n "${USE_HEAD_ATLAS:-}" && "${USE_HEAD_ATLAS:-0}" != "0" ]]; then
  HEAD_ATLAS_ARGS+=(--use-head-atlas)
fi
STRAND_WIDTH_RADIUS_ARGS=()
if [[ "${STRAND_WIDTH_RADIUS:-0}" != "0" ]]; then
  STRAND_WIDTH_RADIUS_ARGS+=(--strand-width-radius)
fi
ADAPTIVE_RENDER_ARGS=()
if [[ "${ADAPTIVE_RENDER_SAMPLES:-1}" != "0" ]]; then
  ADAPTIVE_RENDER_ARGS+=(--adaptive-render-samples)
fi
INIT_CHECKPOINT_ARGS=()
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  INIT_CHECKPOINT_ARGS+=(--init-checkpoint "$INIT_CHECKPOINT")
fi
if [[ "${NO_INIT_CHECKPOINT_ROOTS:-0}" != "0" ]]; then
  INIT_CHECKPOINT_ARGS+=(--no-init-checkpoint-roots)
fi
if [[ "${NO_INIT_CHECKPOINT_MODEL:-0}" != "0" ]]; then
  INIT_CHECKPOINT_ARGS+=(--no-init-checkpoint-model)
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

"$PYTHON" tools/train_white_tiger_uv_groom.py \
  --project-root "$PROJECT_ROOT" \
  --mesh data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj \
  --data data/neuralfur_work/whiteTiger_processed/roaringwalk \
  --out "$OUT" \
  "${INIT_CHECKPOINT_ARGS[@]}" \
  --roots "${ROOTS:-200000}" \
  --extra-roots "${EXTRA_ROOTS:-0}" \
  --extra-root-source "${EXTRA_ROOT_SOURCE:-head_boundary}" \
  --extra-root-boost "${EXTRA_ROOT_BOOST:-2.0}" \
  --extra-root-gamma "${EXTRA_ROOT_GAMMA:-0.75}" \
  --extra-root-weight-cap "${EXTRA_ROOT_WEIGHT_CAP:-0.0}" \
  --extra-root-boundary-dilation "${EXTRA_ROOT_BOUNDARY_DILATION:-17}" \
  --extra-root-min-orient-confidence "${EXTRA_ROOT_MIN_ORIENT_CONFIDENCE:-0.04}" \
  --surface-roots "${SURFACE_ROOTS:-0}" \
  --root-sampling "${ROOT_SAMPLING:-view_importance}" \
  --root-distribution "${ROOT_DISTRIBUTION:-random}" \
  --root-importance-boost "${ROOT_IMPORTANCE_BOOST:-2.5}" \
  --root-importance-gamma "${ROOT_IMPORTANCE_GAMMA:-0.75}" \
  --root-residual-map "${ROOT_RESIDUAL_MAP:-}" \
  --root-residual-boost "${ROOT_RESIDUAL_BOOST:-2.0}" \
  --root-residual-gamma "${ROOT_RESIDUAL_GAMMA:-0.75}" \
  "${ROOT_RESIDUAL_BASE_ARGS[@]}" \
  --root-orient-boost "${ROOT_ORIENT_BOOST:-0.0}" \
  --root-orient-gamma "${ROOT_ORIENT_GAMMA:-0.75}" \
  --root-orient-min-confidence "${ROOT_ORIENT_MIN_CONFIDENCE:-0.04}" \
  --root-head-detail-boost "${ROOT_HEAD_DETAIL_BOOST:-0.0}" \
  --root-head-detail-gamma "${ROOT_HEAD_DETAIL_GAMMA:-0.75}" \
  --root-head-detail-start "${ROOT_HEAD_DETAIL_START:-0.70}" \
  --root-head-detail-sharpness "${ROOT_HEAD_DETAIL_SHARPNESS:-28.0}" \
  --root-head-orient-mix "${ROOT_HEAD_ORIENT_MIX:-0.35}" \
  --root-head-detail-min-orient-confidence "${ROOT_HEAD_DETAIL_MIN_ORIENT_CONFIDENCE:-0.04}" \
  --root-boundary-boost "${ROOT_BOUNDARY_BOOST:-0.0}" \
  --root-boundary-gamma "${ROOT_BOUNDARY_GAMMA:-0.75}" \
  --root-boundary-dilation "${ROOT_BOUNDARY_DILATION:-17}" \
  --head-root-boost "${HEAD_ROOT_BOOST:-0.0}" \
  --head-root-start "${HEAD_ROOT_START:-0.70}" \
  --head-root-sharpness "${HEAD_ROOT_SHARPNESS:-28.0}" \
  --curve-samples "${CURVE_SAMPLES:-24}" \
  --render-samples "${RENDER_SAMPLES:-16}" \
  "${ADAPTIVE_RENDER_ARGS[@]}" \
  --adaptive-min-render-samples "${ADAPTIVE_MIN_RENDER_SAMPLES:-6}" \
  --width "${WIDTH:-512}" \
  --uv-mode "${UV_MODE:-xatlas}" \
  --uv-cache "${UV_CACHE:-data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.xatlas_uv.npz}" \
  --tex-h "${TEX_H:-4096}" \
  --tex-w "${TEX_W:-4096}" \
  --texture-debug-max "${TEXTURE_DEBUG_MAX:-1024}" \
  "${TRIPLANE_ARGS[@]}" \
  --triplane-h "${TRIPLANE_H:-128}" \
  --triplane-w "${TRIPLANE_W:-128}" \
  --triplane-scale "${TRIPLANE_SCALE:-0.75}" \
  "${COARSE_TEXTURE_ARGS[@]}" \
  --coarse-h "${COARSE_H:-512}" \
  --coarse-w "${COARSE_W:-512}" \
  --coarse-scale "${COARSE_SCALE:-0.45}" \
  "${HEAD_ATLAS_ARGS[@]}" \
  --head-h "${HEAD_H:-128}" \
  --head-w "${HEAD_W:-128}" \
  --head-start "${HEAD_START:-0.70}" \
  --head-scale "${HEAD_SCALE:-1.0}" \
  --iters "${ITERS:-4000}" \
  --lr "${LR:-0.02}" \
  --coarse-lr-scale "${COARSE_LR_SCALE:-1.0}" \
  --head-lr-scale "${HEAD_LR_SCALE:-1.0}" \
  --triplane-lr-scale "${TRIPLANE_LR_SCALE:-1.0}" \
  --surface-lr-scale "${SURFACE_LR_SCALE:-1.0}" \
  --pose-lr "${POSE_LR:-0.0005}" \
  --alpha-scale "${ALPHA_SCALE:-0.45}" \
  --alpha-cap "${ALPHA_CAP:-0.50}" \
  --surface-alpha-scale "${SURFACE_ALPHA_SCALE:-0.0}" \
  --random-mesh-backing-weight "${RANDOM_MESH_BACKING_WEIGHT:-0.0}" \
  --random-mesh-backing-start-iter "${RANDOM_MESH_BACKING_START_ITER:-0}" \
  --random-mesh-backing-warmup-iters "${RANDOM_MESH_BACKING_WARMUP_ITERS:-0}" \
  --random-mesh-backing-strength "${RANDOM_MESH_BACKING_STRENGTH:-1.0}" \
  --random-mesh-backing-alpha-scale "${RANDOM_MESH_BACKING_ALPHA_SCALE:-0.75}" \
  --random-mesh-backing-normal-offset "${RANDOM_MESH_BACKING_NORMAL_OFFSET:-0.004}" \
  --root-surface-move-lr "${ROOT_SURFACE_MOVE_LR:-0.0}" \
  --root-surface-move-start-iter "${ROOT_SURFACE_MOVE_START_ITER:-0}" \
  --root-surface-move-reg-weight "${ROOT_SURFACE_MOVE_REG_WEIGHT:-0.0}" \
  --root-surface-move-logit-limit "${ROOT_SURFACE_MOVE_LOGIT_LIMIT:-10.0}" \
  --splat-radius "${SPLAT_RADIUS:-0.35}" \
  --splat-mode "${SPLAT_MODE:-point}" \
  --tangent-radius-scale "${TANGENT_RADIUS_SCALE:-1.8}" \
  --normal-radius-scale "${NORMAL_RADIUS_SCALE:-0.65}" \
  "${STRAND_WIDTH_RADIUS_ARGS[@]}" \
  --radius-width-ref "${RADIUS_WIDTH_REF:-0.0025}" \
  --radius-width-min-scale "${RADIUS_WIDTH_MIN_SCALE:-0.45}" \
  --radius-width-max-scale "${RADIUS_WIDTH_MAX_SCALE:-1.8}" \
  --depth-band "${DEPTH_BAND:-0.08}" \
  --depth-sharpness "${DEPTH_SHARPNESS:-0.0}" \
  --edge-loss-weight "${EDGE_LOSS_WEIGHT:-2.0}" \
  --dark-loss-weight "${DARK_LOSS_WEIGHT:-1.5}" \
  --grad-loss-weight "${GRAD_LOSS_WEIGHT:-0.2}" \
  --head-color-loss-weight "${HEAD_COLOR_LOSS_WEIGHT:-0.0}" \
  --head-mask-loss-weight "${HEAD_MASK_LOSS_WEIGHT:-0.0}" \
  --head-loss-start-iter "${HEAD_LOSS_START_ITER:-0}" \
  --head-loss-warmup-iters "${HEAD_LOSS_WARMUP_ITERS:-0}" \
  --head-loss-dilation "${HEAD_LOSS_DILATION:-13}" \
  --boundary-mask-loss-weight "${BOUNDARY_MASK_LOSS_WEIGHT:-0.0}" \
  --boundary-loss-start-iter "${BOUNDARY_LOSS_START_ITER:-300}" \
  --boundary-loss-warmup-iters "${BOUNDARY_LOSS_WARMUP_ITERS:-1000}" \
  --boundary-loss-dilation "${BOUNDARY_LOSS_DILATION:-17}" \
  --flow-orient-weight "${FLOW_ORIENT_WEIGHT:-0.0}" \
  --flow-orient-start-iter "${FLOW_ORIENT_START_ITER:-0}" \
  --flow-orient-warmup-iters "${FLOW_ORIENT_WARMUP_ITERS:-0}" \
  --flow-orient-end-iter "${FLOW_ORIENT_END_ITER:-0}" \
  --flow-orient-decay-iters "${FLOW_ORIENT_DECAY_ITERS:-0}" \
  --flow-orient-source "${FLOW_ORIENT_SOURCE:-rgb}" \
  --flow-orient-min-confidence "${FLOW_ORIENT_MIN_CONFIDENCE:-0.015}" \
  --flow-orient-max-segments "${FLOW_ORIENT_MAX_SEGMENTS:-60000}" \
  --flow-init-source "${FLOW_INIT_SOURCE:-none}" \
  --flow-init-min-confidence "${FLOW_INIT_MIN_CONFIDENCE:-0.04}" \
  --flow-init-scale "${FLOW_INIT_SCALE:-0.55}" \
  --flow-init-probe-length "${FLOW_INIT_PROBE_LENGTH:-0.03}" \
  --flow-hint-prior-weight "${FLOW_HINT_PRIOR_WEIGHT:-0.0}" \
  --flow-hint-prior-start-iter "${FLOW_HINT_PRIOR_START_ITER:-300}" \
  --flow-hint-prior-warmup-iters "${FLOW_HINT_PRIOR_WARMUP_ITERS:-1000}" \
  --flow-hint-prior-end-iter "${FLOW_HINT_PRIOR_END_ITER:-0}" \
  --flow-hint-prior-decay-iters "${FLOW_HINT_PRIOR_DECAY_ITERS:-0}" \
  --flow-hint-prior-min-confidence "${FLOW_HINT_PRIOR_MIN_CONFIDENCE:-0.04}" \
  --freeze-flow-after-iter "${FREEZE_FLOW_AFTER_ITER:-0}" \
  --flow-coherence-weight "${FLOW_COHERENCE_WEIGHT:-0.0}" \
  --flow-coherence-start-iter "${FLOW_COHERENCE_START_ITER:-300}" \
  --flow-coherence-warmup-iters "${FLOW_COHERENCE_WARMUP_ITERS:-1000}" \
  --flow-coherence-detail-relax "${FLOW_COHERENCE_DETAIL_RELAX:-0.75}" \
  --flow-coherence-min-weight "${FLOW_COHERENCE_MIN_WEIGHT:-0.25}" \
  --orientation-dir "${ORIENTATION_DIR:-orientations_2}" \
  --orientation-angle-bins "${ORIENTATION_ANGLE_BINS:-180}" \
  --detail-init-strength "${DETAIL_INIT_STRENGTH:-0.0}" \
  --detail-density-boost "${DETAIL_DENSITY_BOOST:-0.6}" \
  --detail-length-boost "${DETAIL_LENGTH_BOOST:-0.15}" \
  --detail-root-width-boost "${DETAIL_ROOT_WIDTH_BOOST:-0.0}" \
  --detail-tip-width-boost "${DETAIL_TIP_WIDTH_BOOST:-0.0}" \
  --detail-tv-relax "${DETAIL_TV_RELAX:-0.0}" \
  --detail-tv-min-weight "${DETAIL_TV_MIN_WEIGHT:-0.35}" \
  --texture-tv-weight "${TEXTURE_TV_WEIGHT:-0.025}" \
  --asset-param-smooth-weight "${ASSET_PARAM_SMOOTH_WEIGHT:-0.0}" \
  --groom-geometry-weight "${GROOM_GEOMETRY_WEIGHT:-0.0}" \
  --groom-length-floor "${GROOM_LENGTH_FLOOR:-0.48}" \
  --groom-detail-length-boost "${GROOM_DETAIL_LENGTH_BOOST:-0.12}" \
  --groom-head-length-boost "${GROOM_HEAD_LENGTH_BOOST:-0.10}" \
  --groom-boundary-length-boost "${GROOM_BOUNDARY_LENGTH_BOOST:-0.0}" \
  --groom-boundary-density-floor "${GROOM_BOUNDARY_DENSITY_FLOOR:-0.0}" \
  --groom-boundary-lift-floor "${GROOM_BOUNDARY_LIFT_FLOOR:-0.0}" \
  --groom-root-width-target "${GROOM_ROOT_WIDTH_TARGET:-0.0044}" \
  --groom-tip-width-target "${GROOM_TIP_WIDTH_TARGET:-0.00115}" \
  --groom-max-tip-root-ratio "${GROOM_MAX_TIP_ROOT_RATIO:-0.45}" \
  --gabor-orient-bins "${GABOR_ORIENT_BINS:-36}" \
  --gabor-orient-dog-low "${GABOR_ORIENT_DOG_LOW:-0.4}" \
  --gabor-orient-dog-high "${GABOR_ORIENT_DOG_HIGH:-10.0}" \
  --gabor-orient-sigma-x "${GABOR_ORIENT_SIGMA_X:-1.8}" \
  --gabor-orient-sigma-y "${GABOR_ORIENT_SIGMA_Y:-2.4}" \
  --gabor-orient-frequency "${GABOR_ORIENT_FREQUENCY:-0.23}" \
  --gabor-orient-chunk "${GABOR_ORIENT_CHUNK:-4}" \
  --save-every "${SAVE_EVERY:-200}"

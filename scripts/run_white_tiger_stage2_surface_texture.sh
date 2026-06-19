#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export PROJECT_ROOT
cd "$PROJECT_ROOT"

INIT_CHECKPOINT="${1:-${INIT_CHECKPOINT:-${STAGE1_CHECKPOINT:-}}}"
if [[ -z "$INIT_CHECKPOINT" ]]; then
  echo "Stage 2 requires a Stage 1 checkpoint." >&2
  echo "Usage: bash scripts/run_white_tiger_stage2_surface_texture.sh /path/to/stage1/latest.pt" >&2
  echo "Or set STAGE1_CHECKPOINT=/path/to/stage1/latest.pt before launching." >&2
  exit 2
fi

export RUN_NAME="${RUN_NAME:-$(date +%Y%m%d%H%M%S)}"
export INIT_CHECKPOINT
export SKIP_ORIENTATION_GENERATION="${SKIP_ORIENTATION_GENERATION:-1}"

# Stage 2 continues the trained fur asset. It should not resample roots or add
# residual capacity unless explicitly requested by a separate experiment.
export ROOTS="${ROOTS:-200000}"
export EXTRA_ROOTS="${EXTRA_ROOTS:-0}"
export SURFACE_ROOTS="${SURFACE_ROOTS:-120000}"
export SURFACE_ALPHA_SCALE="${SURFACE_ALPHA_SCALE:-0.75}"
export SURFACE_LR_SCALE="${SURFACE_LR_SCALE:-0.5}"

export RANDOM_MESH_BACKING_WEIGHT="${RANDOM_MESH_BACKING_WEIGHT:-0.0}"
export ROOT_SURFACE_MOVE_LR="${ROOT_SURFACE_MOVE_LR:-0.0}"

export ITERS="${ITERS:-2000}"
export SAVE_EVERY="${SAVE_EVERY:-250}"
export FLOW_ORIENT_WEIGHT="${FLOW_ORIENT_WEIGHT:-0.06}"
export FLOW_COHERENCE_WEIGHT="${FLOW_COHERENCE_WEIGHT:-0.02}"
export TEXTURE_TV_WEIGHT="${TEXTURE_TV_WEIGHT:-0.015}"
export ASSET_PARAM_SMOOTH_WEIGHT="${ASSET_PARAM_SMOOTH_WEIGHT:-0.06}"
export GROOM_GEOMETRY_WEIGHT="${GROOM_GEOMETRY_WEIGHT:-0.5}"

bash "$PROJECT_ROOT/scripts/train_white_tiger_uv_groom_server.sh"

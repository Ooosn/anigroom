#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/neuralfur_work/whiteTiger_processed/roaringwalk}"
MESH_PATH="${MESH_PATH:-${PROJECT_ROOT}/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/checkpoint_a_preflight}"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "[checkpoint_a] environment and input preflight"
PROJECT_ROOT="$PROJECT_ROOT" \
PYTHON="$PYTHON" \
DATA_ROOT="$DATA_ROOT" \
MESH_PATH="$MESH_PATH" \
REQUIRE_ORIENTATION="${REQUIRE_ORIENTATION:-1}" \
REQUIRE_CUDA="${REQUIRE_CUDA:-1}" \
REQUIRE_GSPLAT="${REQUIRE_GSPLAT:-1}" \
bash scripts/verify_white_tiger_server_env.sh | tee "$OUT_DIR/server_env.log"

"$PYTHON" tools/report_white_tiger_stage1_inputs.py \
  --data-root "$DATA_ROOT" \
  --mesh-path "$MESH_PATH" \
  --orientation-dir "${ORIENTATION_DIR:-orientations_2}" \
  --test-stride "${TEST_STRIDE:-6}" \
  --out "$OUT_DIR/stage1_inputs.json" | tee "$OUT_DIR/stage1_inputs.stdout.log"

echo "[checkpoint_a] preflight_ok=1"


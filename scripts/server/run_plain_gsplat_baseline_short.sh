#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/plain_gsplat_baseline_short}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

LPIPS_FLAG=()
if [[ "${COMPUTE_LPIPS:-0}" != "0" ]]; then
  LPIPS_FLAG=(--compute-lpips)
fi

"$PYTHON" tools/train_plain_gsplat_baseline.py \
  --data-root "${DATA_ROOT:-data/neuralfur_work/whiteTiger_processed/roaringwalk}" \
  --mesh-path "${MESH_PATH:-data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj}" \
  --output-dir "$OUT_DIR" \
  --num-gaussians "${NUM_GAUSSIANS:-2000}" \
  --iterations "${ITERATIONS:-20}" \
  --eval-every "${EVAL_EVERY:-10}" \
  --save-every "${SAVE_EVERY:-20}" \
  --test-stride "${TEST_STRIDE:-6}" \
  "${LPIPS_FLAG[@]}"

#!/usr/bin/env bash
set -euo pipefail

# This wrapper launches the audited historical V11 reference only. It contains
# no algorithmic override; the reproduction script owns the two phase configs.
source /home/wangyy/miniconda3/etc/profile.d/conda.sh
conda activate mygs

PROJECT_ROOT="${PROJECT_ROOT:-/work/anigroom-v11-reference}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/neuralfur_work/whiteTiger_processed/roaringwalk}"
MESH_PATH="${MESH_PATH:-${PROJECT_ROOT}/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj}"
RUN_ID="${RUN_ID:-v11_reference_h100_r001}"

export PROJECT_ROOT DATA_ROOT MESH_PATH RUN_ID
export PYTHON="${PYTHON:-python}"

cd "${PROJECT_ROOT}"
exec bash scripts/server/reproduce_v11_from_zero.sh

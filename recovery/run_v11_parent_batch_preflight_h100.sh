#!/usr/bin/env bash
set -euo pipefail

# Run the historical V11 parent command through the H100 batch scheduler.
# This stops after the official one-iteration batch preflight; it is not a
# training shortcut and keeps the exact launcher/configuration path intact.
source /home/wangyy/miniconda3/etc/profile.d/conda.sh
conda activate mygs

cd /work/anigroom-v11-reference
echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,utilization.gpu --format=csv,noheader
export PROJECT_ROOT="$PWD"
export PYTHON=python
export DATA_ROOT="$PROJECT_ROOT/data/neuralfur_work/whiteTiger_processed/roaringwalk"
export MESH_PATH="$PROJECT_ROOT/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj"
export OUTPUT_DIR="$PROJECT_ROOT/outputs/v11_reference_h100_batch_preflight_radii_fix_r002"
export CONFIG_PATH="$PROJECT_ROOT/configs/reproduce_v11_parent_0_9k.env"
export RUN_PREFLIGHT=1
export RUN_BATCH_PREFLIGHT=1
export STAGE1_PREFLIGHT_ONLY=1
export GPU_MEMORY_LIMIT_GB=0

exec bash scripts/server/run_white_tiger_stage1.sh

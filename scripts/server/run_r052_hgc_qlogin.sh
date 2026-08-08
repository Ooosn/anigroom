#!/usr/bin/env bash
set -uo pipefail

repo="${PROJECT_ROOT:-/home/wangyy/anigroom-r052-secondary-guide-color-decomposition-20260808}"
runtime="${RUNTIME_ROOT:-/home/wangyy/anigroom-r052-secondary-guide-color-decomposition-runtime-20260808}"
python_bin="${PYTHON:-/home/wangyy/miniconda3/envs/mygs/bin/python}"
data_root="${DATA_ROOT:-/home/wangyy/anigroom-r002-locked/data/neuralfur_work/whiteTiger_processed/roaringwalk}"
mesh_path="${MESH_PATH:-/home/wangyy/anigroom-r002-locked/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj}"
expected_commit="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
preflight_output="$runtime/preflight/fullres_active_v1"
formal_output="$runtime/outputs/r052_secondary_guide_color_gaussian_residual_0_30k_h100_20260808"
preflight_log=/home/wangyy/logs/anigroom_r052_fullres_active_preflight.log
formal_log=/home/wangyy/logs/anigroom_r052_secondary_guide_color_gaussian_residual_0_30k_h100.log
control=/home/wangyy/run_control

mkdir -p "$runtime/preflight" "$runtime/outputs" "$control"
if ! ulimit -v unlimited || [[ "$(ulimit -v)" != "unlimited" ]]; then
  echo "R052_ENV_REJECT ulimit_v=$(ulimit -v)"
  touch "$control/r052_preflight_done"
  while [[ ! -e "$control/r052_training_release" ]]; do sleep 10; done
  exit 4
fi
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export PYTHONPATH="$repo:${PYTHONPATH:-}"
cd "$repo"

rm -f \
  "$control/r052_preflight_done" \
  "$control/r052_preflight_approved" \
  "$control/r052_preflight_rejected" \
  "$control/r052_training_done" \
  "$control/r052_training_release"

actual_commit="$(git rev-parse HEAD)"
if [[ "$actual_commit" != "$expected_commit" || -n "$(git status --porcelain)" ]]; then
  echo "R052_SOURCE_REJECT expected=$expected_commit actual=$actual_commit dirty=$(git status --porcelain | wc -l)"
  touch "$control/r052_preflight_done"
  while [[ ! -e "$control/r052_training_release" ]]; do sleep 10; done
  exit 2
fi

echo "R052_PREFLIGHT_START host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} at=$(date --iso-8601=seconds)"
echo "R052_SOURCE repo=$repo commit=$actual_commit clean=1"
echo "R052_INPUT data=$data_root mesh=$mesh_path python=$python_bin"
echo "R052_OUTPUT preflight=$preflight_output formal=$formal_output"
rm -rf "$preflight_output"
PROJECT_ROOT="$repo" \
PYTHON="$python_bin" \
DATA_ROOT="$data_root" \
MESH_PATH="$mesh_path" \
CONFIG_PATH="$repo/configs/r052_secondary_guide_color_fullres_preflight.env" \
RUN_ID=r052_fullres_active_preflight \
OUTPUT_DIR="$preflight_output" \
RUN_BATCH_PREFLIGHT=0 \
bash scripts/server/run_white_tiger_stage1.sh > "$preflight_log" 2>&1
preflight_rc=$?
echo "R052_PREFLIGHT_EXIT=$preflight_rc at=$(date --iso-8601=seconds)"
touch "$control/r052_preflight_done"

if [[ "$preflight_rc" -ne 0 ]]; then
  while [[ ! -e "$control/r052_training_release" ]]; do sleep 10; done
  exit "$preflight_rc"
fi

while [[ ! -e "$control/r052_preflight_approved" && ! -e "$control/r052_preflight_rejected" ]]; do
  sleep 5
done
if [[ -e "$control/r052_preflight_rejected" ]]; then
  echo "R052_PREFLIGHT_REJECTED at=$(date --iso-8601=seconds)"
  while [[ ! -e "$control/r052_training_release" ]]; do sleep 10; done
  exit 3
fi

echo "R052_TRAIN_START host=$(hostname) at=$(date --iso-8601=seconds)"
rm -rf "$formal_output"
PROJECT_ROOT="$repo" \
PYTHON="$python_bin" \
DATA_ROOT="$data_root" \
MESH_PATH="$mesh_path" \
CONFIG_PATH="$repo/configs/r052_secondary_guide_color_gaussian_residual_0_30k.env" \
RUN_ID=r052_secondary_guide_color_gaussian_residual_0_30k_h100_20260808 \
OUTPUT_DIR="$formal_output" \
RUN_BATCH_PREFLIGHT=0 \
bash scripts/server/run_white_tiger_stage1.sh > "$formal_log" 2>&1
formal_rc=$?
echo "R052_TRAIN_EXIT=$formal_rc at=$(date --iso-8601=seconds)"
touch "$control/r052_training_done"
while [[ ! -e "$control/r052_training_release" ]]; do sleep 10; done
exit "$formal_rc"

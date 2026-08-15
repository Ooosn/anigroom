#!/usr/bin/env bash
set -euo pipefail

: "${CODE_ROOT:?set CODE_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${MESH_PATH:?set MESH_PATH}"
: "${R062_CHECKPOINT:?set R062_CHECKPOINT}"
: "${R063_CHECKPOINT:?set R063_CHECKPOINT}"
: "${R064_CHECKPOINT:?set R064_CHECKPOINT}"

source /home/wangyy/miniconda3/etc/profile.d/conda.sh
conda activate mygs
ulimit -v unlimited
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

cd "$CODE_ROOT"
mkdir -p "$OUTPUT_ROOT"

echo "LENGTH_DIAGNOSTIC_START=$(date -Iseconds)"
echo "HOST=$(hostname)"
echo "CODE_COMMIT=$(git rev-parse HEAD)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ULIMIT_V=$(ulimit -v)"

python tools/diagnose_checkpoint_length_ownership.py \
  --checkpoint "$R062_CHECKPOINT" \
  --mesh "$MESH_PATH" \
  --output "$OUTPUT_ROOT/r062_length_ownership.npz" \
  --device cuda

python tools/diagnose_checkpoint_length_ownership.py \
  --checkpoint "$R063_CHECKPOINT" \
  --mesh "$MESH_PATH" \
  --output "$OUTPUT_ROOT/r063_length_ownership.npz" \
  --device cuda

python tools/diagnose_checkpoint_length_ownership.py \
  --checkpoint "$R064_CHECKPOINT" \
  --mesh "$MESH_PATH" \
  --output "$OUTPUT_ROOT/r064_length_ownership.npz" \
  --device cuda

find "$OUTPUT_ROOT" -maxdepth 1 -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUTPUT_ROOT/SHA256SUMS"

echo "LENGTH_DIAGNOSTIC_COMPLETE=$(date -Iseconds)"

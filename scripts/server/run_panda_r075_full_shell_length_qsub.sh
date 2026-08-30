#!/usr/bin/env bash
set -euo pipefail

: "${JOB_ID:?R075 qsub wrapper requires JOB_ID}"
: "${PROJECT_ROOT:?R075 qsub wrapper requires PROJECT_ROOT}"

job_detail="$(qstat -j "$JOB_ID")"
mapfile -t granted_devices < <(
  printf '%s\n' "$job_detail" \
    | sed -nE 's#.*granted_devices.*\/dev\/nvidia([0-9]+).*#\1#p' \
    | sort -u
)
if [[ "${#granted_devices[@]}" -ne 1 ]]; then
  echo "R075 expected exactly one granted NVIDIA device; got ${#granted_devices[@]}" >&2
  printf '%s\n' "$job_detail" >&2
  exit 3
fi

physical_granted_device="${granted_devices[0]}"
unset CUDA_VISIBLE_DEVICES
mapfile -t visible_devices < <(
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits \
    | tr -d ' ' \
    | sed '/^$/d' \
    | sort -u
)
if [[ "${#visible_devices[@]}" -ne 1 ]]; then
  echo "R075 expected exactly one container-visible NVIDIA device; got ${#visible_devices[@]}" >&2
  nvidia-smi -L >&2 || true
  exit 4
fi

export CUDA_VISIBLE_DEVICES="${visible_devices[0]}"
echo "R075_QSUB_DEVICE JOB_ID=$JOB_ID PHYSICAL_GRANTED_DEVICE=$physical_granted_device CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

exec bash "$PROJECT_ROOT/scripts/server/run_panda_r075_full_shell_length.sh"

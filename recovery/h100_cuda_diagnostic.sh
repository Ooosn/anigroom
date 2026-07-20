#!/usr/bin/env bash
set -euo pipefail

# Scheduler-level diagnostic only. It does not import AniGroom or start training.
source /home/wangyy/miniconda3/etc/profile.d/conda.sh
conda activate mygs

printf 'host=%s\n' "$(hostname)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
print(f"device_name={torch.cuda.get_device_name(0)}")
print(f"mem_info={torch.cuda.mem_get_info(0)}")
torch.empty(1, device="cuda")
print("cuda_one_element_allocation=ok")
PY

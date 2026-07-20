#!/usr/bin/env bash
set -euo pipefail

# Source-independent scheduler diagnostic. It intentionally does not import or
# modify the trainer: it records the GPU visible to a fresh batch job and tests
# the exact CPU-to-CUDA tensor operation that failed before model construction.
source /home/wangyy/miniconda3/etc/profile.d/conda.sh
conda activate mygs

echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-<unset>}"
nvidia-smi

python - <<'PY'
import os

import numpy as np
import torch
import trimesh

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in this batch job")

device = torch.device("cuda:0")
print(f"device_name={torch.cuda.get_device_name(device)}")
print(f"mem_before={torch.cuda.mem_get_info(device)}")
for shape, dtype in (((1,), np.float32), ((4096, 3), np.float32), ((100_000, 3), np.float32), ((80_007, 3), np.float64)):
    source = np.zeros(shape, dtype=dtype)
    target = torch.from_numpy(source).to(device=device)
    torch.cuda.synchronize(device)
    print(f"cpu_to_cuda_shape={shape} dtype={source.dtype} mem_after={torch.cuda.mem_get_info(device)}")
    del target
mesh = trimesh.load(
    "/work/anigroom-v11-reference/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj",
    process=False,
)
vertices = np.asarray(mesh.vertices)
print(f"mesh_vertices_shape={vertices.shape} dtype={vertices.dtype} bytes={vertices.nbytes}")
target = torch.from_numpy(vertices).to(device=device)
torch.cuda.synchronize(device)
print(f"mesh_vertices_cpu_to_cuda=ok mem_after={torch.cuda.mem_get_info(device)}")
del target
torch.cuda.empty_cache()
print(f"mem_final={torch.cuda.mem_get_info(device)}")
PY

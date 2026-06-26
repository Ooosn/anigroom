#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
GSPLAT_WHEEL="${GSPLAT_WHEEL:-/ssdwork/liuhaohan/RTS/gs2dgs_gs3copy_notex_fresh/submodules/gsplat-1.1.1-py3-none-any.whl}"
PATCHED_BACKEND="${PATCHED_BACKEND:-/ssdwork/liuhaohan/RTS/gs2dgs_gs3copy_notex_fresh/submodules/gsplat-1.1.1/gsplat/cuda/_backend.py}"
SITE_BACKEND="$("$PYTHON" - <<'PY'
from pathlib import Path
import site

for base in site.getsitepackages():
    candidate = Path(base) / "gsplat" / "cuda" / "_backend.py"
    if candidate.exists():
        print(candidate)
        break
PY
)"

if [[ ! -f "$GSPLAT_WHEEL" ]]; then
  echo "Missing gsplat wheel: $GSPLAT_WHEEL" >&2
  exit 2
fi
if [[ ! -f "$PATCHED_BACKEND" ]]; then
  echo "Missing patched gsplat backend: $PATCHED_BACKEND" >&2
  exit 2
fi

"$PYTHON" -m pip install "$GSPLAT_WHEEL"

if [[ -z "$SITE_BACKEND" ]]; then
  SITE_BACKEND="$("$PYTHON" - <<'PY'
from pathlib import Path
import site

for base in site.getsitepackages():
    candidate = Path(base) / "gsplat" / "cuda" / "_backend.py"
    if candidate.exists():
        print(candidate)
        break
PY
)"
fi

if [[ -z "$SITE_BACKEND" || ! -f "$SITE_BACKEND" ]]; then
  echo "Could not locate installed gsplat backend" >&2
  exit 3
fi

cp "$PATCHED_BACKEND" "$SITE_BACKEND"

"$PYTHON" - <<'PY'
import time

import torch
from gsplat.cuda._backend import _C
from gsplat.rendering import rasterization

if _C is None:
    raise SystemExit("gsplat CUDA backend is None")

device = "cuda"
means = torch.tensor([[0.0, 0.0, 3.0]], device=device, requires_grad=True)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True)
scales = torch.tensor([[0.15, 0.15, 0.15]], device=device, requires_grad=True)
opacities = torch.tensor([0.8], device=device, requires_grad=True)
colors = torch.tensor([[1.0, 0.8, 0.6]], device=device, requires_grad=True)
viewmats = torch.eye(4, device=device)[None]
ks = torch.tensor(
    [[[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]]],
    device=device,
)
t0 = time.time()
render, alpha, _ = rasterization(
    means, quats, scales, opacities, colors, viewmats, ks, 64, 64, packed=False
)
loss = render.mean() + alpha.mean()
loss.backward()
print(
    {
        "gsplat_backend": str(_C),
        "render_shape": tuple(render.shape),
        "alpha_shape": tuple(alpha.shape),
        "loss": float(loss.detach().cpu()),
        "mean_grad_norm": float(means.grad.norm().detach().cpu()),
        "time_sec": round(time.time() - t0, 4),
    },
    flush=True,
)
if float(means.grad.norm().detach().cpu()) <= 0:
    raise SystemExit("gsplat backward produced zero mean gradient")
PY

echo "gsplat_cached_backend_ok=1"


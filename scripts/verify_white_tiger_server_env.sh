#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/neuralfur_work/whiteTiger_processed/roaringwalk}"
MESH_PATH="${MESH_PATH:-${PROJECT_ROOT}/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj}"
ORIENTATION_DIR="${ORIENTATION_DIR:-orientations_2}"
REQUIRE_ORIENTATION="${REQUIRE_ORIENTATION:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-0}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

"$PYTHON" - <<'PY'
import importlib.util
import json
from pathlib import Path

import torch

required = ["PIL", "numpy", "xatlas"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
out = {
    "python": True,
    "torch_version": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "missing": missing,
}
if torch.cuda.is_available():
    out["cuda_device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(out, ensure_ascii=False), flush=True)
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")
PY

if [[ ! -f "$MESH_PATH" ]]; then
  echo "Missing mesh: $MESH_PATH" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/images" ]]; then
  echo "Missing image directory: $DATA_ROOT/images" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT/silhouette" && ! -d "$DATA_ROOT/masks" ]]; then
  echo "Missing mask directory: expected $DATA_ROOT/silhouette or $DATA_ROOT/masks" >&2
  exit 1
fi
mask_root="$DATA_ROOT/silhouette"
if [[ ! -d "$mask_root" ]]; then
  mask_root="$DATA_ROOT/masks"
fi
image_count=$(find "$DATA_ROOT/images" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
mask_count=$(find "$mask_root" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
echo "image_count=$image_count"
echo "mask_count=$mask_count"
if [[ "$image_count" == "0" ]]; then
  echo "No PNG images found under $DATA_ROOT/images" >&2
  exit 1
fi
if [[ "$mask_count" == "0" ]]; then
  echo "No PNG masks found under $mask_root" >&2
  exit 1
fi
if [[ "$image_count" != "$mask_count" ]]; then
  echo "Image/mask count mismatch: images=$image_count masks=$mask_count" >&2
  exit 1
fi

if [[ "$REQUIRE_CUDA" != "0" ]]; then
  cuda_count=$("$PYTHON" -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
  echo "required_cuda_device_count=$cuda_count"
  if [[ "$cuda_count" == "0" ]]; then
    echo "CUDA is required for this preflight but no CUDA device is visible" >&2
    exit 1
  fi
fi

if [[ "$REQUIRE_ORIENTATION" != "0" ]]; then
  angle_dir="$DATA_ROOT/$ORIENTATION_DIR/angles"
  var_dir="$DATA_ROOT/$ORIENTATION_DIR/vars"
  angle_count=0
  var_count=0
  if [[ -d "$angle_dir" ]]; then
    angle_count=$(find "$angle_dir" -maxdepth 1 -type f | wc -l | tr -d ' ')
  fi
  if [[ -d "$var_dir" ]]; then
    var_count=$(find "$var_dir" -maxdepth 1 -type f | wc -l | tr -d ' ')
  fi
  echo "orientation_angle_count=$angle_count"
  echo "orientation_var_count=$var_count"
  if [[ "$angle_count" != "$image_count" || "$var_count" != "$image_count" ]]; then
    echo "Missing or incomplete generated orientation maps under $DATA_ROOT/$ORIENTATION_DIR: images=$image_count angles=$angle_count vars=$var_count" >&2
    exit 1
  fi
fi

echo "white_tiger_server_env_ok=1"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/neuralfur_work/whiteTiger_processed/roaringwalk}"
MESH_PATH="${MESH_PATH:-${PROJECT_ROOT}/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj}"
ORIENTATION_DIR="${ORIENTATION_DIR:-orientations_2}"
EXPECTED_WIDTH="${EXPECTED_WIDTH:-1920}"
EXPECTED_HEIGHT="${EXPECTED_HEIGHT:-1080}"
EXPECTED_IMAGE_COUNT="${EXPECTED_IMAGE_COUNT:-36}"

export EXPECTED_WIDTH EXPECTED_HEIGHT EXPECTED_IMAGE_COUNT

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p outputs

"$PYTHON" - <<'PY'
import importlib.util
import json

import torch

missing = [name for name in ["PIL", "numpy", "gsplat"] if importlib.util.find_spec(name) is None]
report = {
    "torch": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "missing": missing,
}
if torch.cuda.is_available():
    report["cuda_device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(report, ensure_ascii=False), flush=True)
if missing:
    raise SystemExit(f"missing required packages: {missing}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for white tiger Stage 1")
PY

"$PYTHON" -B -m py_compile tools/train_white_tiger_stage1.py
"$PYTHON" tools/train_white_tiger_stage1.py --help >/dev/null

"$PYTHON" tools/report_white_tiger_stage1_inputs.py \
  --data-root "$DATA_ROOT" \
  --mesh-path "$MESH_PATH" \
  --orientation-dir "$ORIENTATION_DIR" \
  --out outputs/preflight_white_tiger_stage1_input_report.json \
  >/dev/null

"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

report_path = Path("outputs/preflight_white_tiger_stage1_input_report.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
expected_width = int(os.environ["EXPECTED_WIDTH"])
expected_height = int(os.environ["EXPECTED_HEIGHT"])
expected_count = int(os.environ["EXPECTED_IMAGE_COUNT"])

errors = list(report.get("errors", []))
if report.get("image_size") != [expected_width, expected_height]:
    errors.append(f"expected image_size {[expected_width, expected_height]}, got {report.get('image_size')}")
if report.get("mask_size") != [expected_width, expected_height]:
    errors.append(f"expected mask_size {[expected_width, expected_height]}, got {report.get('mask_size')}")
if int(report.get("image_count", -1)) != expected_count:
    errors.append(f"expected image_count {expected_count}, got {report.get('image_count')}")
if int(report.get("mask_count", -1)) != expected_count:
    errors.append(f"expected mask_count {expected_count}, got {report.get('mask_count')}")
if int(report.get("orientation_angle_count", -1)) != expected_count:
    errors.append(f"expected orientation_angle_count {expected_count}, got {report.get('orientation_angle_count')}")
if int(report.get("orientation_conf_count", -1)) != expected_count:
    errors.append(f"expected orientation_conf_count {expected_count}, got {report.get('orientation_conf_count')}")
if len(report.get("train_indices", [])) != 30:
    errors.append(f"expected 30 train views, got {len(report.get('train_indices', []))}")
if len(report.get("test_indices", [])) != 6:
    errors.append(f"expected 6 test views, got {len(report.get('test_indices', []))}")

if errors:
    print(json.dumps({"white_tiger_stage1_preflight": "failed", "errors": errors}, ensure_ascii=False, indent=2))
    raise SystemExit(2)

print(json.dumps({
    "white_tiger_stage1_preflight": "ok",
    "resolution": report["image_size"],
    "image_count": report["image_count"],
    "train_count": len(report["train_indices"]),
    "test_count": len(report["test_indices"]),
}, ensure_ascii=False, indent=2))
PY

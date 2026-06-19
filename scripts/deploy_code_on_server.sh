#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
PACKAGE="${1:-}"

if [[ -z "$PACKAGE" ]]; then
  echo "Usage: bash deploy_code_on_server.sh /tmp/anigroom_code.tar.gz" >&2
  exit 2
fi
if [[ ! -f "$PACKAGE" ]]; then
  echo "Missing package: $PACKAGE" >&2
  exit 2
fi

mkdir -p "$PROJECT_ROOT"
for path in README.md .gitignore .gitattributes anigroom docs scripts tools; do
  if [[ -e "$PROJECT_ROOT/$path" ]]; then
    rm -rf "$PROJECT_ROOT/$path"
  fi
done
tar -xzf "$PACKAGE" -C "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
"$PYTHON" -m py_compile \
  tools/train_white_tiger_uv_groom.py \
  tools/render_white_tiger_uv_groom_orbit.py \
  tools/generate_orientation_maps.py \
  tools/make_video_from_frames.py
bash -n scripts/train_white_tiger_uv_groom_server.sh
bash -n scripts/run_white_tiger_stage1_reconstruction.sh
bash -n scripts/run_white_tiger_formal_asset_pipeline.sh
bash -n scripts/run_white_tiger_stage2_surface_texture.sh

echo "MAINLINE_DEPLOY_OK=$PROJECT_ROOT"

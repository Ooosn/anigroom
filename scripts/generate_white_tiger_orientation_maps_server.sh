#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
PYTHON="${PYTHON:-/opt/conda/envs/gs/bin/python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/neuralfur_work/whiteTiger_processed/roaringwalk}"
PREPROCESS_SCRIPT="${PREPROCESS_SCRIPT:-${PROJECT_ROOT}/external/NeuralFur_official/submodules/GaussianHaircut/src/preprocessing/calc_orientation_maps.py}"
LOCAL_GENERATOR="${LOCAL_GENERATOR:-${PROJECT_ROOT}/tools/generate_orientation_maps.py}"
ORIENTATION_GENERATOR="${ORIENTATION_GENERATOR:-anigroom}"

IMG_DIR="${IMG_DIR:-${DATA_ROOT}/images}"
MASK_DIR="${MASK_DIR:-${DATA_ROOT}/silhouette}"
ORIENTATION_ROOT="${ORIENTATION_ROOT:-${DATA_ROOT}/orientations_2}"
ANGLE_DIR="${ANGLE_DIR:-${ORIENTATION_ROOT}/angles}"
VAR_DIR="${VAR_DIR:-${ORIENTATION_ROOT}/vars}"
FILTERED_DIR="${FILTERED_DIR:-${ORIENTATION_ROOT}/filtered_imgs}"
VIS_DIR="${VIS_DIR:-${ORIENTATION_ROOT}/vis_imgs}"

count_files() {
  local dir="$1"
  local pattern="$2"
  if [[ ! -d "$dir" ]]; then
    echo 0
    return
  fi
  find "$dir" -maxdepth 1 -type f -name "$pattern" | wc -l | tr -d ' '
}

IMAGE_COUNT=$(count_files "$IMG_DIR" '*.png')
ANGLE_COUNT=$(count_files "$ANGLE_DIR" '*.png')
VAR_COUNT=$(count_files "$VAR_DIR" '*.npy')

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "ORIENTATION_GENERATOR=${ORIENTATION_GENERATOR}"
echo "PREPROCESS_SCRIPT=${PREPROCESS_SCRIPT}"
echo "LOCAL_GENERATOR=${LOCAL_GENERATOR}"
echo "IMAGE_COUNT=${IMAGE_COUNT}"
echo "ANGLE_DIR=${ANGLE_DIR}"
echo "VAR_DIR=${VAR_DIR}"
echo "EXISTING_ANGLE_COUNT=${ANGLE_COUNT}"
echo "EXISTING_VAR_COUNT=${VAR_COUNT}"
echo "ORIENTATION_GENERATOR_BINS=${ORIENTATION_GENERATOR_BINS:-180}"

if [[ "${DRY_RUN:-0}" != "0" ]]; then
  echo "DRY_RUN=1; not generating orientation maps"
  exit 0
fi

mkdir -p "$ANGLE_DIR" "$VAR_DIR" "$FILTERED_DIR" "$VIS_DIR"

if [[ "${FORCE:-0}" == "0" && "$IMAGE_COUNT" != "0" && "$ANGLE_COUNT" == "$IMAGE_COUNT" && "$VAR_COUNT" == "$IMAGE_COUNT" ]]; then
  echo "orientation maps already complete; set FORCE=1 to regenerate"
  exit 0
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [[ "$ORIENTATION_GENERATOR" == "official" ]]; then
  if [[ ! -f "$PREPROCESS_SCRIPT" ]]; then
    echo "Missing NeuralFur/GaussianHaircut orientation script: ${PREPROCESS_SCRIPT}" >&2
    exit 2
  fi
  "$PYTHON" "$PREPROCESS_SCRIPT" \
    --img_path "$IMG_DIR" \
    --mask_path "$MASK_DIR" \
    --orient_dir "$ANGLE_DIR" \
    --conf_dir "$VAR_DIR" \
    --filtered_img_dir "$FILTERED_DIR" \
    --vis_img_dir "$VIS_DIR"
elif [[ "$ORIENTATION_GENERATOR" == "anigroom" ]]; then
  if [[ ! -f "$LOCAL_GENERATOR" ]]; then
    echo "Missing AniGroom orientation generator: ${LOCAL_GENERATOR}" >&2
    exit 2
  fi
  "$PYTHON" "$LOCAL_GENERATOR" \
    --img-path "$IMG_DIR" \
    --mask-path "$MASK_DIR" \
    --orient-dir "$ANGLE_DIR" \
    --conf-dir "$VAR_DIR" \
    --filtered-img-dir "$FILTERED_DIR" \
    --vis-img-dir "$VIS_DIR" \
    --width "${ORIENTATION_WIDTH:-0}" \
    --bins "${ORIENTATION_GENERATOR_BINS:-180}" \
    --dog-low "${ORIENTATION_DOG_LOW:-0.4}" \
    --dog-high "${ORIENTATION_DOG_HIGH:-10.0}" \
    --sigma-x "${ORIENTATION_SIGMA_X:-1.8}" \
    --sigma-y "${ORIENTATION_SIGMA_Y:-2.4}" \
    --frequency "${ORIENTATION_FREQUENCY:-0.23}" \
    --device "${ORIENTATION_DEVICE:-auto}"
else
  echo "Unsupported ORIENTATION_GENERATOR=${ORIENTATION_GENERATOR}; expected anigroom or official" >&2
  exit 2
fi

ANGLE_COUNT=$(count_files "$ANGLE_DIR" '*.png')
VAR_COUNT=$(count_files "$VAR_DIR" '*.npy')
echo "FINAL_ANGLE_COUNT=${ANGLE_COUNT}"
echo "FINAL_VAR_COUNT=${VAR_COUNT}"

if [[ "$ANGLE_COUNT" != "$IMAGE_COUNT" || "$VAR_COUNT" != "$IMAGE_COUNT" ]]; then
  echo "Orientation generation incomplete: images=${IMAGE_COUNT}, angles=${ANGLE_COUNT}, vars=${VAR_COUNT}" >&2
  exit 3
fi

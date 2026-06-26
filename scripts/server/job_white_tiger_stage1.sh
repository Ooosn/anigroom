#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssdwork/liuhaohan/petsgaussianhair}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
JOB_LOG_DIR="${JOB_LOG_DIR:-${PROJECT_ROOT}/logs/${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${RUN_ID}}"

export PROJECT_ROOT RUN_ID OUTPUT_DIR

mkdir -p "$JOB_LOG_DIR" "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

{
  echo "[job-stage1] start $(date -Is)"
  echo "[job-stage1] project=${PROJECT_ROOT}"
  echo "[job-stage1] run_id=${RUN_ID}"
  echo "[job-stage1] output_dir=${OUTPUT_DIR}"
  echo "[job-stage1] log_dir=${JOB_LOG_DIR}"
  echo "[job-stage1] command=bash ${PROJECT_ROOT}/scripts/server/run_white_tiger_stage1.sh"
} | tee "${JOB_LOG_DIR}/stage1_job_header.log"

bash "${PROJECT_ROOT}/scripts/server/run_white_tiger_stage1.sh" 2>&1 | tee "${JOB_LOG_DIR}/stage1_train.log"

echo "[job-stage1] done $(date -Is)" | tee -a "${JOB_LOG_DIR}/stage1_job_header.log"

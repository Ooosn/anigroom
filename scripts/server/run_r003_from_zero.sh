#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/neuralfur_work/whiteTiger_processed/roaringwalk}"
MESH_PATH="${MESH_PATH:-${PROJECT_ROOT}/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj}"
RUN_ID="${RUN_ID:-r003_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"

PHASE_A_OUTPUT="${RUN_ROOT}/phase_a_0_9k"
PHASE_B_OUTPUT="${RUN_ROOT}/phase_b_9k_30k"
PHASE_A_CONFIG="${PROJECT_ROOT}/configs/v11_v4_parent_0_9k.env"
PHASE_B_CONFIG="${PROJECT_ROOT}/configs/white_tiger_stage1_local_rgb_groom_v11_appearance_lenfree_from9k.env"
FLOW_TARGET="${PROJECT_ROOT}/_downloads/tiger_hair_flow_36/shell_fused_smal_head500_body4000_candidate65536_headk24_bodyk12_v4_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

cat > "${RUN_ROOT}/expected_source.sha256" <<'EOF'
e783c59871ef98ef6261751b1fcce9dbd12ac640df141d7252f97aab14e9280e  tools/train_white_tiger_stage1.py
44572b47573bdc462f517946d968d073724987d37cee7ef44efce4cfcaa6e804  anigroom/grooming/strand_gaussians.py
9793028a9dd742ec86e4d832bf5bf8b4e67de280eba7a207c4a144a6951b1e57  anigroom/roots/statistics.py
f9d3f229922eea48e968d55068cabf7ca1c997e65d8d732c700f8d0139cf4153  anigroom/roots/lifecycle.py
c4b0ed9c51ff706288ddbe52da3005d154238b17055f856a7484e282631f9694  anigroom/surface_interpolation.py
ebfd78a846fd21d768d44068323dfa702d2c324fe447b2355647e199bad95430  anigroom/flow/clean_flow.py
d531fdd7b2d806f442c2a144d2cb2b135e83fd8a6633fcb1b5d84a44ec2c63b1  anigroom/flow/direction_geometry.py
7849b61f9d67b61a44ed1cc38f6cf935db0ab2388486336095cbeea195b661f3  anigroom/flow/surface_graph.py
51512dbc7f931c4a15605ad9098bd898b72c9f01e7e3768781e771c775fa8b25  anigroom/flow/__init__.py
0ac0c4f3f23230ae50d58bd73c5a4e7e212360ef603d5a3d23c18f8b3feceb2a  anigroom/mesh_roots.py
7a8f8e021859c602b996c978880b3f8463070b4856a597a5277bc3a829e8dffe  anigroom/projection/__init__.py
ee1913ff51821ea5f1b192984e088127bc00a51a8de577c961a442261e4eae14  anigroom/projection/mesh_visibility.py
f414da3a8b0bb06d7d4c41b4d35d95ea7a700ac31170c8812ec41c077e178bf1  configs/white_tiger_stage1_cleanflow_multiview_30k_rgbflow.env
ff68b33a6a38961b3b74697ee85cc571a58831abacc4536e1780edf6a680af8b  configs/white_tiger_stage1_cleanflow_view09.env
1899e345d289056a91cbb8cdd9a70d35ba355331be9124e4bd4bb25d1a6fbee8  configs/white_tiger_stage1_formal.env
16991a94a6308654de954a97a7b52b21d22709ba2d47d63c27ad1639ac3a861e  configs/v11_v4_parent_0_9k.env
74195ece58773d67d81cb3a0b2a80b136d8bea515a7a18864457321e4904fbfa  configs/white_tiger_stage1_local_rgb_groom_v11_appearance_lenfree_from9k.env
dcb1e8c90ff4d1fa4b98c399975448d499d113a8be070846989cff8739e53849  scripts/server/run_white_tiger_stage1.sh
EOF

sha256sum -c "${RUN_ROOT}/expected_source.sha256" | tee "${LOG_ROOT}/source_check.log"

check_file_hash() {
  local expected="$1"
  local path="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    echo "[r003] data hash mismatch: ${label}: ${actual} != ${expected}" >&2
    exit 4
  fi
  printf '%s  %s\n' "$actual" "$label"
}

check_directory_manifest() {
  local expected="$1"
  local path="$2"
  local label="$3"
  local actual
  actual="$(cd "$path" && find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum | sed 's/ \*/  /' | sha256sum | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    echo "[r003] directory manifest mismatch: ${label}: ${actual} != ${expected}" >&2
    exit 5
  fi
  printf '%s  %s\n' "$actual" "$label"
}

{
  check_file_hash b7688480a36489d67b1dec691745eb17ef7da006ddfeab46746b3f399b0ec750 "$MESH_PATH" mesh
  check_file_hash 60a33b360bb415cb47cd38173d6e0cf4504448203ef277a5861641b40fdb3141 "$FLOW_TARGET" clean_flow_target_v4
  check_file_hash f5f2c4798edc86c0e08bb846d15d8f48e67d80f9105a428fd0ede54ab6814c05 "$DATA_ROOT/cameras.npz" cameras.npz
  check_file_hash f5f2c4798edc86c0e08bb846d15d8f48e67d80f9105a428fd0ede54ab6814c05 "$DATA_ROOT/cameras_wo_scale.npz" cameras_wo_scale.npz
  check_file_hash 390ff086e6e01d56d971c86f4a13a5c855587691aefc95727dce11ad1171beeb "$DATA_ROOT/cameras_intr.npy" cameras_intr.npy
  check_file_hash 0607295cee5d1f78ed2c941ec9bd229fbfece2c6dbc7069dadb1ed312ae9359a "$DATA_ROOT/cameras_extr.npy" cameras_extr.npy
  check_file_hash d48dbb0f7e9ba125ec008396974cf63fc4fa98634996b830a7fe10d1d857004b "$DATA_ROOT/cameras_extr_wo_scale.npy" cameras_extr_wo_scale.npy
  check_directory_manifest 170a50710c3df1b6b704bc4bc2928cc17c503a1dac72a9afc9e34358798d2593 "$DATA_ROOT/images" images
  check_directory_manifest e1277f518367f9354527cac31f070d47000f2a91039703a0aba6aac04cdd3126 "$DATA_ROOT/silhouette" silhouette
  check_directory_manifest 172f74333ba0dd8fb1e65282de1660e45e48ce1b65434794abb49e8f834ac988 "$DATA_ROOT/orientations_2" orientations_2
} | tee "${LOG_ROOT}/data_check.log"

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'project_root=%s\n' "$PROJECT_ROOT"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'python=%s\n' "$PYTHON"
  printf 'flow_target=%s\n' "$FLOW_TARGET"
  git status --short
  sha256sum \
    tools/train_white_tiger_stage1.py \
    anigroom/grooming/strand_gaussians.py \
    anigroom/roots/statistics.py \
    anigroom/roots/lifecycle.py \
    anigroom/surface_interpolation.py \
    anigroom/flow/clean_flow.py \
    anigroom/flow/direction_geometry.py \
    anigroom/flow/surface_graph.py \
    anigroom/mesh_roots.py \
    anigroom/projection/__init__.py \
    anigroom/projection/mesh_visibility.py \
    configs/white_tiger_stage1_formal.env \
    configs/white_tiger_stage1_cleanflow_view09.env \
    configs/white_tiger_stage1_cleanflow_multiview_30k_rgbflow.env \
    configs/v11_v4_parent_0_9k.env \
    configs/white_tiger_stage1_local_rgb_groom_v11_appearance_lenfree_from9k.env \
    scripts/server/run_white_tiger_stage1.sh \
    scripts/server/run_r003_from_zero.sh
} > "${RUN_ROOT}/source_manifest.txt"

"$PYTHON" -m compileall -q anigroom tools/train_white_tiger_stage1.py

if [[ "${VERIFY_ONLY:-0}" == "1" ]]; then
  echo "[r003] VERIFY_ONLY=1; source and data contracts passed"
  exit 0
fi

echo "[r003] phase A: iteration 0 -> 9000"
PROJECT_ROOT="$PROJECT_ROOT" \
PYTHON="$PYTHON" \
DATA_ROOT="$DATA_ROOT" \
MESH_PATH="$MESH_PATH" \
RUN_ID="${RUN_ID}_a" \
OUTPUT_DIR="$PHASE_A_OUTPUT" \
CONFIG_PATH="$PHASE_A_CONFIG" \
RUN_PREFLIGHT=1 \
RUN_BATCH_PREFLIGHT=1 \
RESUME_CHECKPOINT= \
RESUME_OPTIMIZER=1 \
GPU_MEMORY_LIMIT_GB="${GPU_MEMORY_LIMIT_GB:-0}" \
bash scripts/server/run_white_tiger_stage1.sh 2>&1 | tee "${LOG_ROOT}/phase_a.log"

PHASE_A_CHECKPOINT="${PHASE_A_OUTPUT}/checkpoint_009000.pt"
if [[ ! -s "$PHASE_A_CHECKPOINT" ]]; then
  echo "[r003] missing phase-A checkpoint: $PHASE_A_CHECKPOINT" >&2
  exit 3
fi

echo "[r003] phase B: iteration 9000 -> 30000"
PROJECT_ROOT="$PROJECT_ROOT" \
PYTHON="$PYTHON" \
DATA_ROOT="$DATA_ROOT" \
MESH_PATH="$MESH_PATH" \
RUN_ID="${RUN_ID}_b" \
OUTPUT_DIR="$PHASE_B_OUTPUT" \
CONFIG_PATH="$PHASE_B_CONFIG" \
RESUME_CHECKPOINT="$PHASE_A_CHECKPOINT" \
RESUME_OPTIMIZER=0 \
RUN_PREFLIGHT=0 \
RUN_BATCH_PREFLIGHT=0 \
GPU_MEMORY_LIMIT_GB="${GPU_MEMORY_LIMIT_GB:-0}" \
bash scripts/server/run_white_tiger_stage1.sh 2>&1 | tee "${LOG_ROOT}/phase_b.log"

echo "[r003] complete: ${RUN_ROOT}"

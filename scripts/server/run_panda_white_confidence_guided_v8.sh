#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
ulimit -v unlimited || true

: "${RUNTIME_ROOT:?Set RUNTIME_ROOT to a new formal runtime directory}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the trusted Git checkout}"
: "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT to the reviewed commit}"

PYTHON_PATH="${PYTHON_PATH:-/home/wangyy/miniconda3/envs/mygs/bin/python}"
PANDA_RUNTIME="${PANDA_RUNTIME:-/home/wangyy/panda-v4-flow-runtime-20260826}"
WHITE_INPUT_ROOT="${WHITE_INPUT_ROOT:-/home/wangyy/anigroom-flow-trusted-20260827/white_input}"
PANDA_OUTPUT="$RUNTIME_ROOT/outputs/panda_v8_confidence_guided"
WHITE_OUTPUT="$RUNTIME_ROOT/outputs/white_v8_confidence_guided"

source_commit="$(git -C "$SOURCE_ROOT" rev-parse HEAD)"
if [[ "$source_commit" != "$EXPECTED_SOURCE_COMMIT" ]]; then
  echo "source commit mismatch: expected=$EXPECTED_SOURCE_COMMIT actual=$source_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --short)" ]]; then
  echo "refusing dirty source checkout: $SOURCE_ROOT" >&2
  git -C "$SOURCE_ROOT" status --short >&2
  exit 2
fi
for path in "$PYTHON_PATH" "$SOURCE_ROOT/tools/fuse_gpt_flow_shell_multiview.py"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 2
  fi
done
for output in "$PANDA_OUTPUT" "$WHITE_OUTPUT"; do
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite $output" >&2
    exit 3
  fi
done

mkdir -p "$RUNTIME_ROOT/outputs"
echo "START $(date -Is) HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ULIMIT_V=$(ulimit -v) PYTHON=$PYTHON_PATH"
echo "SOURCE_ROOT=$SOURCE_ROOT SOURCE_COMMIT=$source_commit"
echo "PANDA_RUNTIME=$PANDA_RUNTIME WHITE_INPUT_ROOT=$WHITE_INPUT_ROOT"
echo "PANDA_OUTPUT=$PANDA_OUTPUT WHITE_OUTPUT=$WHITE_OUTPUT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

echo "TESTS_START $(date -Is)"
(
  cd "$SOURCE_ROOT"
  "$PYTHON_PATH" -B -m pytest -q
)
echo "TESTS_DONE $(date -Is)"

run_fusion() {
  local data_root="$1"
  local mesh_path="$2"
  local flow_dir="$3"
  local roots_file="$4"
  local roots_space="$5"
  local scale="$6"
  local translation="$7"
  local output_dir="$8"
  mkdir -p "$output_dir"
  "$PYTHON_PATH" -B "$SOURCE_ROOT/tools/fuse_gpt_flow_shell_multiview.py" \
    --data-root "$data_root" \
    --mesh-path "$mesh_path" \
    --flow-dir "$flow_dir" \
    --output-dir "$output_dir" \
    --exclude 4,24,25 \
    --root-count 4500 \
    --candidate-multiplier 8.0 \
    --root-sampling-mode surface-roots-file \
    --surface-roots-file "$roots_file" \
    --surface-roots-file-space "$roots_space" \
    --scale "$scale" \
    --translation "$translation" \
    --min-confidence 0.04 \
    --depth-abs-tolerance 0.03 \
    --depth-rel-tolerance 0.01 \
    --local-depth-kernel 7 \
    --front-normal-z 0.15 \
    --view-angle-power 1.0 \
    --diag-view 27 \
    --direction-lambda-values 0.10,0.16,0.24,0.34,0.48,0.68,0.90 \
    --direction-field-mode continuous-ratio \
    --directed-flow-propagation-mode confidence-guided \
    --continuous-direction-iters 90 \
    --continuous-direction-lr 0.06 \
    --continuous-direction-smooth-weight 0.35 \
    --continuous-direction-anchor-weight 0.08 \
    --continuous-ratio-robust-max-quantile 0.995 \
    --continuous-ratio-robust-max-scale 1.35 \
    --direction-consensus-iters 16 \
    --direction-consensus-blend 0.45 \
    --direction-consensus-anchor-threshold 0.75 \
    --axis-field-mode trusted-view-cluster \
    --axis-field-iters 10 \
    --axis-field-smooth-strength 0.65 \
    --shell-count 9 \
    --shell-extent 2.5 \
    --shell-spacing-k 8 \
    --shell-smooth-iters 6 \
    --shell-smooth-strength 1.4 \
    --shell-score-temperature 0.08 \
    --shell-anchor-weight 0.7 \
    --shell-height-smooth-iters 16 \
    --shell-height-smooth-strength 0.35 \
    --shell-height-anchor-weight 0.7 \
    --silhouette-band-offsets 8,16,28,44,64 \
    --silhouette-band-weight 0.75 \
    --silhouette-mesh-dilate 9 \
    --silhouette-normal-screen-min 0.35 \
    --silhouette-sign-bias 0.35 \
    --clean-knn-k 12 \
    --clean-head-knn-k 24 \
    --clean-body-knn-k 12 \
    --clean-region-id-key root_file_region_ids \
    --root-neighborhood mesh-geodesic \
    --clean-sign-iters 12 \
    --clean-lambda-iters 4 \
    --clean-vector-iters 6 \
    --clean-anchor-margin 0.02 \
    --clean-anchor-weight 0.5 \
    --clean-smooth-strength 2.0 \
    --clean-vector-blend 0.35
}

echo "PANDA_START $(date -Is)"
run_fusion \
  "$PANDA_RUNTIME/input" \
  "$PANDA_RUNTIME/input/mesh/furless.obj" \
  "$PANDA_RUNTIME/input/flow" \
  "$PANDA_RUNTIME/output/guides_head500_body4000_candidates65536/white_tiger_smal_head_body_guide_roots.npz" \
  raw 1.0 0,0,0 "$PANDA_OUTPUT"
echo "PANDA_DONE $(date -Is)"

echo "WHITE_START $(date -Is)"
run_fusion \
  "$WHITE_INPUT_ROOT/data" \
  "$WHITE_INPUT_ROOT/mesh/furless_reshaped.obj" \
  "$WHITE_INPUT_ROOT/flow" \
  "$WHITE_INPUT_ROOT/white_tiger_surface_roots.npz" \
  camera 1.28 0,0.32,0.02 "$WHITE_OUTPUT"
echo "WHITE_DONE $(date -Is)"

"$PYTHON_PATH" -B - "$PANDA_OUTPUT" "$WHITE_OUTPUT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

required_arrays = (
    "cleaned_directed_flow3d",
    "axis_view_cluster_global_final_sign",
    "axis_view_cluster_global_canonical_rank",
    "axis_view_cluster_global_edge_new_severe_mask",
    "axis_view_cluster_postratio_final_ratio",
    "axis_view_cluster_postratio_accept_mask",
    "axis_view_cluster_postratio_edge_u",
    "axis_view_cluster_postratio_edge_v",
    "axis_view_cluster_confidence_flow_input_direction",
    "axis_view_cluster_confidence_flow_watershed_direction",
    "axis_view_cluster_confidence_flow_joint_confidence",
    "axis_view_cluster_confidence_flow_watershed_owner",
    "axis_view_cluster_confidence_flow_watershed_parent",
    "axis_view_cluster_confidence_flow_propagated_confidence",
    "axis_view_cluster_confidence_flow_watershed_changed",
    "axis_view_cluster_confidence_flow_local_changed",
    "axis_view_cluster_confidence_flow_changed",
    "axis_view_cluster_confidence_flow_protected_owner",
    "axis_view_cluster_confidence_flow_local_update_count",
    "axis_view_cluster_confidence_flow_edge_u",
    "axis_view_cluster_confidence_flow_edge_v",
    "axis_view_cluster_confidence_flow_initial_edge_dot",
    "axis_view_cluster_confidence_flow_watershed_edge_dot",
    "axis_view_cluster_confidence_flow_final_edge_dot",
    "axis_view_cluster_confidence_flow_new_severe_edge",
)

for output_arg in sys.argv[1:]:
    output = Path(output_arg)
    target_path = output / "guide_flow3d_shell_targets_exclude_004_024_025.npz"
    summary_path = output / "summary.json"
    target = np.load(target_path, allow_pickle=False)
    missing = sorted(set(required_arrays) - set(target.files))
    if missing:
        raise RuntimeError(f"{output}: missing arrays {missing}")
    for key in target.files:
        value = target[key]
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise RuntimeError(f"{output}: non-finite array {key}")
    if np.any(target["axis_view_cluster_global_edge_new_severe_mask"]):
        raise RuntimeError(f"{output}: global sign introduced a severe edge")
    if np.any(target["axis_view_cluster_confidence_flow_new_severe_edge"]):
        raise RuntimeError(f"{output}: confidence-guided flow introduced a severe edge")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    global_report = summary["global_sign_orientation"]
    ratio_report = summary["fixed_sign_directed_multiview_ratio"]
    confidence_report = summary["confidence_guided_directed_flow"]
    if not global_report["zero_new_severe_verification"]["passed"]:
        raise RuntimeError(f"{output}: global sign verification failed")
    if not ratio_report["zero_new_severe_verification"]["passed"]:
        raise RuntimeError(f"{output}: directed ratio verification failed")
    if not confidence_report["zero_new_severe_verification"]["passed"]:
        raise RuntimeError(f"{output}: confidence-guided flow verification failed")
    confidence_counts = confidence_report["counts"]["observed_edges"]
    print(
        json.dumps(
            {
                "output": str(output),
                "observed": int(target["observed"].sum()),
                "global_changed": int(target["axis_view_cluster_global_flip"].sum()),
                "postratio_accepted": int(target["axis_view_cluster_postratio_accept_mask"].sum()),
                "global_new_severe": int(target["axis_view_cluster_global_edge_new_severe_mask"].sum()),
                "confidence_guided_observed": {
                    "initial_negative": int(confidence_counts["initial_negative"]),
                    "final_negative": int(confidence_counts["final_negative"]),
                    "initial_severe": int(confidence_counts["initial_severe"]),
                    "final_severe": int(confidence_counts["final_severe"]),
                },
                "confidence_guided_changed_roots": int(
                    confidence_report["counts"]["changed_roots"]
                ),
            },
            sort_keys=True,
        )
    )
PY

for output in "$PANDA_OUTPUT" "$WHITE_OUTPUT"; do
  sha256sum \
    "$output/guide_flow3d_shell_targets_exclude_004_024_025.npz" \
    "$output/summary.json" \
    "$output/view27_shell_cleaned_3d_arrows_overlay.png"
done
echo "ALL_DONE $(date -Is)"

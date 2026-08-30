#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?R075 asset export requires PROJECT_ROOT}"
: "${EXPECTED_SOURCE_COMMIT:?R075 asset export requires EXPECTED_SOURCE_COMMIT}"
: "${CHECKPOINT:?R075 asset export requires CHECKPOINT}"
: "${EXPECTED_CHECKPOINT_SHA256:?R075 asset export requires EXPECTED_CHECKPOINT_SHA256}"
: "${ASSET_OUTPUT_ROOT:?R075 asset export requires ASSET_OUTPUT_ROOT}"
: "${CUDA_VISIBLE_DEVICES:?R075 asset export requires CUDA_VISIBLE_DEVICES}"

PYTHON="${PYTHON:-python}"

fail() {
  echo "[r075-assets] $*" >&2
  exit 2
}

[[ -d "$PROJECT_ROOT" ]] || fail "missing project root: $PROJECT_ROOT"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"
command -v "$PYTHON" >/dev/null 2>&1 || fail "Python executable is unavailable: $PYTHON"
[[ -f "$CHECKPOINT" ]] || fail "missing checkpoint: $CHECKPOINT"
[[ -s "$CHECKPOINT" ]] || fail "empty checkpoint: $CHECKPOINT"

actual_commit="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
[[ "$actual_commit" == "$EXPECTED_SOURCE_COMMIT" ]] || \
  fail "source commit mismatch: expected=$EXPECTED_SOURCE_COMMIT actual=$actual_commit"
source_status="$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)"
if [[ -n "$source_status" ]]; then
  echo "[r075-assets] source checkout is dirty: $PROJECT_ROOT" >&2
  printf '%s\n' "$source_status" >&2
  exit 2
fi

actual_checkpoint_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$actual_checkpoint_sha256" == "$EXPECTED_CHECKPOINT_SHA256" ]] || \
  fail "checkpoint hash mismatch: expected=$EXPECTED_CHECKPOINT_SHA256 actual=$actual_checkpoint_sha256"

checkpoint_iteration="$("$PYTHON" -B - "$CHECKPOINT" <<'PY'
from __future__ import annotations

import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
iteration = int(checkpoint.get("iteration", -1))
if iteration != 3000:
    raise RuntimeError(f"checkpoint iteration must be 3000, got {iteration}")
print(iteration)
PY
)"
[[ "$checkpoint_iteration" == "3000" ]] || \
  fail "checkpoint iteration must be 3000, got $checkpoint_iteration"

STRAND_EXPORTER="$PROJECT_ROOT/tools/export_white_tiger_checkpoint_strands.py"
GAUSSIAN_EXPORTER="$PROJECT_ROOT/tools/export_white_tiger_checkpoint_gaussians_ply.py"
[[ -f "$STRAND_EXPORTER" ]] || fail "missing strand exporter: $STRAND_EXPORTER"
[[ -f "$GAUSSIAN_EXPORTER" ]] || fail "missing Gaussian exporter: $GAUSSIAN_EXPORTER"

[[ ! -e "$ASSET_OUTPUT_ROOT" ]] || \
  fail "refusing existing output: $ASSET_OUTPUT_ROOT"
[[ ! -L "$ASSET_OUTPUT_ROOT" ]] || \
  fail "refusing existing output symlink: $ASSET_OUTPUT_ROOT"

OUTPUT_PARENT="$(dirname -- "$ASSET_OUTPUT_ROOT")"
mkdir -p "$OUTPUT_PARENT"
mkdir -- "$ASSET_OUTPUT_ROOT"

STRAND_OUTPUT_ROOT="$ASSET_OUTPUT_ROOT/strands"
GAUSSIAN_OUTPUT_ROOT="$ASSET_OUTPUT_ROOT/gaussians"
LOG_ROOT="$ASSET_OUTPUT_ROOT/logs"
mkdir -p "$STRAND_OUTPUT_ROOT" "$GAUSSIAN_OUTPUT_ROOT" "$LOG_ROOT"

MATCHED_OUTPUT="$STRAND_OUTPUT_ROOT/r075_003000_render_child1_100k_samples32.npz"
MATCHED_REPORT="$STRAND_OUTPUT_ROOT/r075_003000_render_child1_100k_samples32.json"
FULL_OUTPUT="$STRAND_OUTPUT_ROOT/r075_003000_render_child1_all_samples32.npz"
FULL_REPORT="$STRAND_OUTPUT_ROOT/r075_003000_render_child1_all_samples32.json"
GAUSSIAN_OUTPUT="$GAUSSIAN_OUTPUT_ROOT/r075_003000_full_3dgs.ply"
GAUSSIAN_REPORT="$GAUSSIAN_OUTPUT_ROOT/r075_003000_full_3dgs.json"
ASSET_REPORT="$ASSET_OUTPUT_ROOT/asset_export_report.json"

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec > >(tee "$LOG_ROOT/export.log") 2>&1

echo "R075_ASSET_EXPORT_START $(date -Is) HOST=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SOURCE_COMMIT=$actual_commit CHECKPOINT=$CHECKPOINT CHECKPOINT_SHA256=$actual_checkpoint_sha256 ITERATION=$checkpoint_iteration"
"${PYTHON}" -B -c 'import torch; print(torch.cuda.get_device_name(torch.cuda.current_device()))'
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

echo "R075_MATCHED100K_START $(date -Is)"
"$PYTHON" -B \
  "$STRAND_EXPORTER" \
  --checkpoint "$CHECKPOINT" \
  --output "$MATCHED_OUTPUT" \
  --device cuda \
  --samples 32 \
  --root-domain render \
  --child-count 1 \
  --max-strands 100000 \
  --seed 29 \
  --uniform-color 0.82 0.80 0.72 \
  2>&1 | tee "$LOG_ROOT/matched100k.log"
echo "R075_MATCHED100K_DONE $(date -Is)"

echo "R075_FULL_STRANDS_START $(date -Is)"
"$PYTHON" -B \
  "$STRAND_EXPORTER" \
  --checkpoint "$CHECKPOINT" \
  --output "$FULL_OUTPUT" \
  --device cuda \
  --samples 32 \
  --root-domain render \
  --child-count 1 \
  --max-strands 0 \
  --seed 29 \
  --uniform-color 0.82 0.80 0.72 \
  2>&1 | tee "$LOG_ROOT/all_strands.log"
echo "R075_FULL_STRANDS_DONE $(date -Is)"

echo "R075_FULL_GAUSSIANS_START $(date -Is)"
"$PYTHON" -B \
  "$GAUSSIAN_EXPORTER" \
  --checkpoint "$CHECKPOINT" \
  --output "$GAUSSIAN_OUTPUT" \
  --device cuda \
  --max-gaussians 0 \
  --seed 29 \
  --sh-degree 3 \
  2>&1 | tee "$LOG_ROOT/gaussian_ply.log"
echo "R075_FULL_GAUSSIANS_DONE $(date -Is)"

for generated in \
  "$MATCHED_OUTPUT" "$MATCHED_REPORT" \
  "$FULL_OUTPUT" "$FULL_REPORT" \
  "$GAUSSIAN_OUTPUT" "$GAUSSIAN_REPORT"; do
  [[ -s "$generated" ]] || fail "missing generated asset or report: $generated"
done

echo "R075_ASSET_VALIDATION_START $(date -Is)"
"$PYTHON" -B - \
  "$CHECKPOINT" \
  "$MATCHED_OUTPUT" "$MATCHED_REPORT" \
  "$FULL_OUTPUT" "$FULL_REPORT" \
  "$GAUSSIAN_OUTPUT" "$GAUSSIAN_REPORT" \
  "$ASSET_REPORT" \
  "$actual_commit" \
  "$actual_checkpoint_sha256" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_finite(value: object, location: str) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), f"non-finite report value: {location}")
    elif isinstance(value, dict):
        for key, child in value.items():
            require_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_finite(child, f"{location}[{index}]")


def load_report(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"report is not an object: {path}")
    require_finite(value, str(path))
    return value


def exact_int(value: object, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{label} is not an integer")
    return int(value)


def check_common_report(
    report: dict[str, object],
    label: str,
    checkpoint: Path,
) -> None:
    require(exact_int(report.get("iteration"), f"{label}.iteration") == 3000, f"{label} iteration is not 3000")
    require(report.get("root_domain") == "render", f"{label} root domain is not render")
    require(exact_int(report.get("child_count"), f"{label}.child_count") == 1, f"{label} child count is not 1")
    require(exact_int(report.get("samples"), f"{label}.samples") == 32, f"{label} samples are not 32")
    report_checkpoint = report.get("checkpoint")
    require(isinstance(report_checkpoint, str), f"{label} checkpoint is not a string")
    require(Path(report_checkpoint).resolve() == checkpoint.resolve(), f"{label} checkpoint identity mismatch")


def check_npz(
    path: Path,
    report: dict[str, object],
    label: str,
    checkpoint: Path,
    expected_count: int,
    full_root_count: int | None = None,
) -> int:
    required = {"strands", "widths", "colors", "opacities", "root_ids", "iteration", "source_checkpoint"}
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        require(not missing, f"{label} NPZ is missing {missing}")
        for key in data.files:
            value = data[key]
            if np.issubdtype(value.dtype, np.number):
                require(bool(np.isfinite(value).all()), f"non-finite NPZ array: {label}.{key}")

        strands = np.asarray(data["strands"])
        widths = np.asarray(data["widths"])
        colors = np.asarray(data["colors"])
        opacities = np.asarray(data["opacities"])
        root_ids = np.asarray(data["root_ids"])
        require(strands.ndim == 3 and strands.shape[1:] == (32, 3), f"{label} strand shape is {strands.shape}")
        count = int(strands.shape[0])
        require(count == expected_count, f"{label} strand count is {count}, expected {expected_count}")
        require(widths.shape == (count, 32, 1), f"{label} width shape is {widths.shape}")
        require(colors.shape == (count, 32, 3), f"{label} color shape is {colors.shape}")
        require(opacities.shape == (count, 32, 1), f"{label} opacity shape is {opacities.shape}")
        require(root_ids.shape == (count,), f"{label} root-id shape is {root_ids.shape}")
        require(np.issubdtype(root_ids.dtype, np.integer), f"{label} root IDs are not integers")
        require(bool((root_ids >= 0).all()), f"{label} contains a negative root ID")

        iteration = np.asarray(data["iteration"]).reshape(-1)
        require(iteration.size == 1 and int(iteration[0]) == 3000, f"{label} NPZ iteration is not 3000")
        source_checkpoint = np.asarray(data["source_checkpoint"]).reshape(-1)
        require(source_checkpoint.size == 1, f"{label} source checkpoint field is not scalar")
        require(Path(str(source_checkpoint[0])).resolve() == checkpoint.resolve(), f"{label} NPZ checkpoint identity mismatch")

        expected_color = np.asarray([0.82, 0.80, 0.72], dtype=colors.dtype).reshape(1, 1, 3)
        require(bool(np.array_equal(colors, np.broadcast_to(expected_color, colors.shape))), f"{label} is not uniformly colored")

        unique_root_ids = np.unique(root_ids)
        require(unique_root_ids.size == count, f"{label} has repeated root IDs")
        if full_root_count is not None:
            require(count == full_root_count, f"{label} count does not equal full root count")
            expected_root_ids = np.arange(full_root_count, dtype=root_ids.dtype)
            require(bool(np.array_equal(unique_root_ids, expected_root_ids)), f"{label} root IDs are not the exact full root population")
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ply_vertex_header(path: Path) -> tuple[int, list[str]]:
    lines: list[str] = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            require(raw, f"PLY header is truncated: {path}")
            line = raw.decode("ascii").strip()
            lines.append(line)
            if line == "end_header":
                break
    require(lines and lines[0] == "ply", f"not a PLY file: {path}")
    require(any(line.startswith("format binary_little_endian ") for line in lines), f"PLY is not binary little-endian: {path}")

    vertex_count: int | None = None
    vertex_properties: list[str] = []
    in_vertex = False
    for line in lines:
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "element" and fields[1] == "vertex":
            vertex_count = int(fields[2])
            in_vertex = True
        elif fields and fields[0] == "element":
            in_vertex = False
        elif in_vertex and len(fields) >= 3 and fields[0] == "property":
            vertex_properties.append(fields[-1])
    require(vertex_count is not None, f"PLY has no vertex element: {path}")
    return vertex_count, vertex_properties


checkpoint_path = Path(sys.argv[1])
matched_path, matched_report_path = Path(sys.argv[2]), Path(sys.argv[3])
full_path, full_report_path = Path(sys.argv[4]), Path(sys.argv[5])
gaussian_path, gaussian_report_path = Path(sys.argv[6]), Path(sys.argv[7])
asset_report_path = Path(sys.argv[8])
source_commit = sys.argv[9]
checkpoint_sha256 = sys.argv[10]

matched_report = load_report(matched_report_path)
full_report = load_report(full_report_path)
gaussian_report = load_report(gaussian_report_path)
check_common_report(matched_report, "matched100k", checkpoint_path)
check_common_report(full_report, "full_strands", checkpoint_path)

root_count = exact_int(full_report.get("root_count"), "full_strands.root_count")
matched_root_count = exact_int(matched_report.get("root_count"), "matched100k.root_count")
require(root_count > 100000, f"full root count is not larger than 100000: {root_count}")
require(matched_root_count == root_count, "matched and full root counts differ")
require(exact_int(matched_report.get("strand_count"), "matched100k.strand_count") == 100000, "matched export is not exactly 100000 strands")
full_strand_count = exact_int(full_report.get("strand_count"), "full_strands.strand_count")
require(full_strand_count == root_count, "full strand count does not equal the full render-root count")
check_npz(matched_path, matched_report, "matched100k", checkpoint_path, 100000)
check_npz(full_path, full_report, "full_strands", checkpoint_path, full_strand_count, root_count)

require(exact_int(gaussian_report.get("iteration"), "full_3dgs.iteration") == 3000, "full Gaussian report iteration is not 3000")
gaussian_checkpoint = gaussian_report.get("checkpoint")
require(isinstance(gaussian_checkpoint, str), "full Gaussian checkpoint is not a string")
require(Path(gaussian_checkpoint).resolve() == checkpoint_path.resolve(), "full Gaussian checkpoint identity mismatch")
require("SH degree 3" in str(gaussian_report.get("format", "")), "full Gaussian report is not SH3")
full_gaussians = exact_int(gaussian_report.get("full_gaussians"), "full_3dgs.full_gaussians")
exported_gaussians = exact_int(gaussian_report.get("exported_gaussians"), "full_3dgs.exported_gaussians")
require(full_gaussians > 0, "full Gaussian count is not positive")
require(exported_gaussians == full_gaussians, "Gaussian export is not the full population")
stats = gaussian_report.get("stats")
require(isinstance(stats, dict), "full Gaussian stats are not an object")
require(exact_int(stats.get("root_count"), "full_3dgs.stats.root_count") == root_count, "Gaussian stats root count mismatch")
require(exact_int(stats.get("gaussian_count"), "full_3dgs.stats.gaussian_count") == full_gaussians, "Gaussian stats count mismatch")
require(exact_int(stats.get("gaussian_unique_root_count"), "full_3dgs.stats.gaussian_unique_root_count") == root_count, "Gaussian stats unique-root count mismatch")

ply_vertex_count, ply_properties = read_ply_vertex_header(gaussian_path)
require(ply_vertex_count == full_gaussians, f"PLY vertex count {ply_vertex_count} != report count {full_gaussians}")
required_properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
required_properties.extend(f"f_rest_{index}" for index in range(45))
required_properties.extend(["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
require(ply_properties == required_properties, "PLY properties do not match the full SH3 3DGS schema")

output_hashes = {
    str(path.resolve()): sha256(path)
    for path in (
        matched_path,
        matched_report_path,
        full_path,
        full_report_path,
        gaussian_path,
        gaussian_report_path,
    )
}
payload = {
    "status": "PASS",
    "source_commit": source_commit,
    "checkpoint": str(checkpoint_path.resolve()),
    "checkpoint_sha256": checkpoint_sha256,
    "iteration": 3000,
    "root_domain": "render",
    "child_count": 1,
    "samples": 32,
    "seed": 29,
    "uniform_color": [0.82, 0.80, 0.72],
    "matched_strands": 100000,
    "full_root_count": root_count,
    "full_strands": full_strand_count,
    "full_gaussians": full_gaussians,
    "outputs": output_hashes,
}
require_finite(payload, "asset_export_report")
asset_report_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
echo "R075_ASSET_VALIDATION_DONE $(date -Is)"

printf '%s  %s\n' "$actual_checkpoint_sha256" "$CHECKPOINT" > "$ASSET_OUTPUT_ROOT/checkpoint_hashes.sha256"
sha256sum \
  "$MATCHED_OUTPUT" "$MATCHED_REPORT" \
  "$FULL_OUTPUT" "$FULL_REPORT" \
  > "$ASSET_OUTPUT_ROOT/strand_hashes.sha256"
sha256sum "$GAUSSIAN_OUTPUT" "$GAUSSIAN_REPORT" > "$ASSET_OUTPUT_ROOT/gaussian_ply_hashes.sha256"
find "$ASSET_OUTPUT_ROOT" -type f \
  ! -path "$LOG_ROOT/*" \
  ! -name 'SHA256SUMS' \
  ! -name '*.sha256' \
  -print0 | sort -z | xargs -0 sha256sum > "$ASSET_OUTPUT_ROOT/SHA256SUMS"

echo "R075_ASSET_EXPORT_DONE $(date -Is)"
cat "$ASSET_OUTPUT_ROOT/asset_export_report.json"
cat "$ASSET_OUTPUT_ROOT/strand_hashes.sha256"
cat "$ASSET_OUTPUT_ROOT/gaussian_ply_hashes.sha256"

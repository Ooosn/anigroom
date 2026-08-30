from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r074_v8_confidence_flow_0_3k_gate.env"
LAUNCHER = ROOT / "scripts" / "server" / "run_panda_r074_v8_confidence_flow.sh"
R073_SOURCE = 'source "${CONFIG_DIR}/r073_budget_normalized_ownership_0_3k_gate.env"'


def test_r074_sources_r073_without_executable_training_overrides() -> None:
    source = CONFIG.read_text(encoding="utf-8")

    assert source.count(R073_SOURCE) == 1
    after_source = source.split(R073_SOURCE, 1)[1]
    assignments = [
        line.strip()
        for line in after_source.splitlines()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", line.strip())
    ]
    assert assignments == []


def test_r074_launcher_enforces_v8_panda_3k_gate_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    required_inputs = (
        ': "${PROJECT_ROOT:?Set PROJECT_ROOT to the reviewed source checkout}"',
        ': "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT}"',
        ': "${RUNTIME_ROOT:?Set RUNTIME_ROOT to a new R074 runtime directory}"',
        ': "${CLEAN_FLOW_TARGET:?Set CLEAN_FLOW_TARGET to the formal Panda V8 target}"',
        ': "${EXPECTED_TARGET_SHA256:?Set EXPECTED_TARGET_SHA256}"',
        ': "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name the granted GPU}"',
    )
    for fragment in required_inputs:
        assert fragment in source

    required_fragments = (
        'CONFIG_PATH="$PROJECT_ROOT/configs/r074_v8_confidence_flow_0_3k_gate.env"',
        'PANDA_DATA_ROOT="${PANDA_DATA_ROOT:-/home/wangyy/panda-r068-v5-assets-20260827/data}"',
        'PANDA_MESH_PATH="${PANDA_MESH_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/mesh/furless.obj}"',
        'PANDA_SDF_PATH="${PANDA_SDF_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/collision/panda_sdf_long512.npz}"',
        'actual_target_sha256="$(sha256sum "$CLEAN_FLOW_TARGET" | awk \'{print $1}\')"',
        '[[ "$actual_target_sha256" == "$EXPECTED_TARGET_SHA256" ]] ||',
        'INIT_MESH_SCALE=1.0 \\',
        'INIT_MESH_TRANSLATION=0,0,0 \\',
        'OUTPUT_DIR="$RUNTIME_ROOT/outputs/panda_r074_v8_flow_0_3k_h100_20260830"',
        'PREFLIGHT_OUTPUT="$RUNTIME_ROOT/preflight/fullres_view09"',
        'RENDER_ROOT="$RUNTIME_ROOT/renders/iter_003000"',
        'RUN_PREFLIGHT=1 \\',
        'RUN_BATCH_PREFLIGHT=1 \\',
        'PREFLIGHT_VIEW=9 \\',
        'STAGE1_PREFLIGHT_ONLY=1',
        'checkpoint="$OUTPUT_DIR/checkpoint_003000.pt"',
        '--checkpoint "$checkpoint"',
        '--output-dir "$RENDER_ROOT"',
        '--view-ids 9',
        '--device cuda',
        '[[ ! -e "$RUNTIME_ROOT" ]] || fail "refusing existing runtime: $RUNTIME_ROOT"',
    )
    for fragment in required_fragments:
        assert fragment in source

    required_v8_arrays = (
        "cleaned_directed_flow3d",
        "axis_view_cluster_confidence_flow_changed",
        "axis_view_cluster_confidence_flow_final_edge_dot",
        "axis_view_cluster_confidence_flow_new_severe_edge",
    )
    for array_name in required_v8_arrays:
        assert f'"{array_name}"' in source
    assert 'target = np.load(target_path, allow_pickle=False)' in source
    assert 'missing = sorted(required - set(target.files))' in source
    assert 'if np.any(target["axis_view_cluster_confidence_flow_new_severe_edge"]):' in source
    assert 'report = summary["confidence_guided_directed_flow"]' in source
    assert 'if report.get("enabled") != 1:' in source
    assert 'if not report["zero_new_severe_verification"]["passed"]:' in source

    forbidden = ("resume", "s_vmem", "qsub", "qstat", "qdel", "qalter", "scheduler", "release", "kill")
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered, term

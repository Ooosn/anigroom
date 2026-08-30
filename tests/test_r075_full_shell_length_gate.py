from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r075_full_shell_length_0_3k_gate.env"
R074_CONFIG = ROOT / "configs" / "r074_v8_confidence_flow_0_3k_gate.env"
LAUNCHER = ROOT / "scripts" / "server" / "run_panda_r075_full_shell_length.sh"
QSUB_WRAPPER = (
    ROOT / "scripts" / "server" / "run_panda_r075_full_shell_length_qsub.sh"
)
R074_SOURCE = 'source "${CONFIG_DIR}/r074_v8_confidence_flow_0_3k_gate.env"'


def _config_source_chain(path: Path) -> list[tuple[Path, str]]:
    chain: list[tuple[Path, str]] = []
    current = path
    while True:
        source = current.read_text(encoding="utf-8")
        chain.append((current, source))
        match = re.search(
            r'^\s*source\s+"\$\{CONFIG_DIR\}/([^"\n]+)"\s*$',
            source,
            re.MULTILINE,
        )
        if match is None:
            return chain
        current = current.parent / match.group(1)


def test_r075_inherits_r074_and_changes_only_length_scale() -> None:
    source = CONFIG.read_text(encoding="utf-8")

    assert source.count(R074_SOURCE) == 1
    after_source = source.split(R074_SOURCE, 1)[1]
    assignments = [
        line.strip()
        for line in after_source.splitlines()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", line.strip())
    ]
    assert assignments == ["CLEAN_FLOW_LENGTH_INIT_SCALE=1.0"]

    inherited_scale_assignments = [
        line.strip()
        for _path, config_source in _config_source_chain(R074_CONFIG)
        for line in config_source.splitlines()
        if re.match(r"^CLEAN_FLOW_LENGTH_INIT_SCALE\s*=", line.strip())
    ]
    assert inherited_scale_assignments == ["CLEAN_FLOW_LENGTH_INIT_SCALE=0.30"]


def test_r075_launcher_enforces_v8_panda_3k_length_gate_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    required_inputs = (
        ': "${PROJECT_ROOT:?Set PROJECT_ROOT to the reviewed source checkout}"',
        ': "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT}"',
        ': "${RUNTIME_ROOT:?Set RUNTIME_ROOT to a new R075 runtime directory}"',
        ': "${CLEAN_FLOW_TARGET:?Set CLEAN_FLOW_TARGET to the formal Panda V8 target}"',
        ': "${EXPECTED_TARGET_SHA256:?Set EXPECTED_TARGET_SHA256}"',
        ': "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name the granted GPU}"',
    )
    for fragment in required_inputs:
        assert fragment in source

    required_fragments = (
        'CONFIG_PATH="$PROJECT_ROOT/configs/r075_full_shell_length_0_3k_gate.env"',
        'PANDA_DATA_ROOT="${PANDA_DATA_ROOT:-/home/wangyy/panda-r068-v5-assets-20260827/data}"',
        'PANDA_MESH_PATH="${PANDA_MESH_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/mesh/furless.obj}"',
        'PANDA_SDF_PATH="${PANDA_SDF_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/collision/panda_sdf_long512.npz}"',
        'actual_target_sha256="$(sha256sum "$CLEAN_FLOW_TARGET" | awk \'{print $1}\')"',
        '[[ "$actual_target_sha256" == "$EXPECTED_TARGET_SHA256" ]] ||',
        'INIT_MESH_SCALE=1.0 \\',
        'INIT_MESH_TRANSLATION=0,0,0 \\',
        'OUTPUT_DIR="$RUNTIME_ROOT/outputs/panda_r075_full_shell_length_0_3k_h100_20260830"',
        'PREFLIGHT_OUTPUT="$RUNTIME_ROOT/preflight/fullres_view09"',
        'RENDER_ROOT="$RUNTIME_ROOT/renders/iter_003000"',
        'RUN_ID=panda_r075_full_shell_length_fullres_preflight_20260830',
        'RUN_ID=panda_r075_full_shell_length_0_3k_h100_20260830',
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
    assert 'assert gate["source_path"] == target' in source

    assert 'actual_target_sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()' in source
    assert 'assert actual_target_sha256 == expected_target_sha256' in source
    assert 'assert config["clean_flow_length_init_scale"] == 1.0' in source
    assert 'assert clean["clean_flow_length_init_scale"] == 1.0' in source
    assert 'assert clean["clean_flow_source"] == target' in source
    assert 'target_reliable_shell_height_q05' in source
    assert 'target_reliable_shell_height_q95' in source
    assert 'expected_physical_range_at_scale_1.0' in source

    assert "r074" not in source.lower()
    forbidden = ("resume", "s_vmem", "qsub", "qstat", "qdel", "qalter", "scheduler", "release", "kill")
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered, term


def test_r075_qsub_wrapper_uses_the_single_scheduler_granted_device() -> None:
    source = QSUB_WRAPPER.read_text(encoding="utf-8")

    required = (
        ': "${JOB_ID:?R075 qsub wrapper requires JOB_ID}"',
        ': "${PROJECT_ROOT:?R075 qsub wrapper requires PROJECT_ROOT}"',
        'job_detail="$(qstat -j "$JOB_ID")"',
        "granted_devices",
        "/dev\\/nvidia([0-9]+)",
        '| sort -u',
        '[[ "${#granted_devices[@]}" -ne 1 ]]',
        'export CUDA_VISIBLE_DEVICES="${granted_devices[0]}"',
        'nvidia-smi -i "$CUDA_VISIBLE_DEVICES"',
        'exec bash "$PROJECT_ROOT/scripts/server/run_panda_r075_full_shell_length.sh"',
    )
    for fragment in required:
        assert fragment in source

    forbidden = (
        "CUDA_VISIBLE_DEVICES=0",
        "CUDA_VISIBLE_DEVICES=1",
        "qdel",
        "kill",
        "release",
        "s_vmem",
    )
    for fragment in forbidden:
        assert fragment not in source

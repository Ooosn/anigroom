from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r079_length_unlock_at3k_resume.env"
PREFLIGHT_CONFIG = ROOT / "configs" / "r079_length_unlock_at3k_preflight.env"
R074_CONFIG = ROOT / "configs" / "r074_v8_confidence_flow_0_3k_gate.env"
LAUNCHER = ROOT / "scripts" / "server" / "run_panda_r079_length_unlock_at3k.sh"
DOC = ROOT / "docs" / "r079_length_unlock_at3k.md"
R074_SOURCE = 'source "${CONFIG_DIR}/r074_v8_confidence_flow_0_3k_gate.env"'
MAIN_SOURCE = 'source "${CONFIG_DIR}/r079_length_unlock_at3k_resume.env"'


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


def _assignments_after(source: str, source_line: str) -> list[str]:
    after_source = source.split(source_line, 1)[1]
    return [
        line.strip()
        for line in after_source.splitlines()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", line.strip())
    ]


def test_r079_main_config_sources_r074_and_unlocks_only_length_at_3k() -> None:
    source = CONFIG.read_text(encoding="utf-8")

    assert source.count(R074_SOURCE) == 1
    assert _assignments_after(source, R074_SOURCE) == [
        "ITERATIONS=4000",
        "GUIDE_LENGTH_FREEZE_UNTIL=3000",
        "STAGE_SAVE_ITERS=4000",
    ]

    chain = _config_source_chain(CONFIG)
    assert chain[0][0] == CONFIG
    assert chain[1][0] == R074_CONFIG
    inherited_assignments = [
        line.strip()
        for _path, config_source in chain[1:]
        for line in config_source.splitlines()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", line.strip())
    ]
    assert "ROOT_COUNT=400000" in inherited_assignments
    assert "GUIDE_FREEZE_UNTIL=9000" in inherited_assignments
    assert "VIEW_GATE_NORMALIZATION=equal_owner_budget" in inherited_assignments
    assert "VIEW_GATE_GEOMETRY_SUPPORT=1" not in inherited_assignments
    assert "VIEW_GATE_LENGTH_CONFIDENCE_SUPPORT=1" not in inherited_assignments
    assert "CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT=0.080" not in inherited_assignments
    assert "CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_REDUCTION=tail_concentration" not in inherited_assignments
    assert "r077" not in source.lower()
    assert "r078" not in source.lower()


def test_r079_preflight_sources_main_and_changes_only_boundary_values() -> None:
    source = PREFLIGHT_CONFIG.read_text(encoding="utf-8")

    assert source.count(MAIN_SOURCE) == 1
    assert _assignments_after(source, MAIN_SOURCE) == [
        "ITERATIONS=3001",
        "STAGE_SAVE_ITERS=3001",
        "SAVE_EVERY=0",
        "TRAIN_VIEWS=9",
        "TEST_VIEWS=9",
    ]

    chain = _config_source_chain(PREFLIGHT_CONFIG)
    assert chain[0][0] == PREFLIGHT_CONFIG
    assert chain[1][0] == CONFIG
    assert chain[2][0] == R074_CONFIG


def test_r079_launcher_enforces_migration_resume_and_4000_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    required_inputs = (
        ': "${PROJECT_ROOT:?Set PROJECT_ROOT to the reviewed source checkout}"',
        ': "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT}"',
        ': "${RUNTIME_ROOT:?Set RUNTIME_ROOT to a new R079 runtime directory}"',
        ': "${R074_CHECKPOINT:?Set R074_CHECKPOINT to the original R074 checkpoint',
        ': "${EXPECTED_R074_CHECKPOINT_SHA256:?Set EXPECTED_R074_CHECKPOINT_SHA256=fcd62694663a7ab9383ff0250fa6a44544b7bafff1ebc96ffd7a2e05ad8d013e}"',
        ': "${CLEAN_FLOW_TARGET:?Set CLEAN_FLOW_TARGET to the formal Panda V8 target}"',
        ': "${EXPECTED_TARGET_SHA256:?Set EXPECTED_TARGET_SHA256}"',
        ': "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name the granted GPU}"',
    )
    for fragment in required_inputs:
        assert fragment in source

    required_fragments = (
        'CONFIG_PATH="$PROJECT_ROOT/configs/r079_length_unlock_at3k_resume.env"',
        'PREFLIGHT_CONFIG_PATH="$PROJECT_ROOT/configs/r079_length_unlock_at3k_preflight.env"',
        'PANDA_DATA_ROOT="${PANDA_DATA_ROOT:-/home/wangyy/panda-r068-v5-assets-20260827/data}"',
        'PANDA_MESH_PATH="${PANDA_MESH_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/mesh/furless.obj}"',
        'PANDA_SDF_PATH="${PANDA_SDF_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/collision/panda_sdf_long512.npz}"',
        'actual_target_sha256="$(sha256sum "$CLEAN_FLOW_TARGET" | awk \'{print $1}\')"',
        'actual_r074_checkpoint_sha256="$(sha256sum "$R074_CHECKPOINT" | awk \'{print $1}\')"',
        '[[ -z "$(git -C "$PROJECT_ROOT" status --short)" ]] ||',
        'EXPECTED_ORIGINAL_R074_CHECKPOINT_SHA256="fcd62694663a7ab9383ff0250fa6a44544b7bafff1ebc96ffd7a2e05ad8d013e"',
        'MIGRATION_UTILITY="${MIGRATION_UTILITY:-$PROJECT_ROOT/tools/migrate_stage1_schema12_checkpoint.py}"',
        'MIGRATED_CHECKPOINT="$RUNTIME_ROOT/contracts/r079_migrated_checkpoint.pt"',
        'MIGRATION_REPORT="$RUNTIME_ROOT/contracts/r079_migration_report.json"',
        '"$PYTHON_PATH" -B "$MIGRATION_UTILITY" \\',
        '--checkpoint "$R074_CHECKPOINT" \\',
        '--output "$MIGRATED_CHECKPOINT" \\',
        '--report "$MIGRATION_REPORT" \\',
        '--expected-input-sha256 "$EXPECTED_R074_CHECKPOINT_SHA256"',
        'if report.get("status") != "pass":',
        'identity_report[identity_name] is True',
        'report["tensor_integrity_checks"]["source_to_migrated_identical"] is True',
        'report["tensor_integrity_checks"]["migrated_to_output_identical"] is True',
        'defaults = report["schema14_defaults"]',
        '"$PYTHON_PATH" -B -m pytest -q',
        'RUN_ID=panda_r079_length_unlock_at3k_resume_preflight_20260831',
        'RUN_ID=panda_r079_length_unlock_at3k_resume_4000_20260831',
        'RESUME_CHECKPOINT="$MIGRATED_CHECKPOINT"',
        'RESUME_OPTIMIZER=1',
        'RUN_PREFLIGHT=0',
        'RUN_BATCH_PREFLIGHT=0',
        'assert setup["start_iteration"] == 3000',
        'assert setup["root_count"] == 480292',
        'assert hashlib.sha256(target_path.read_bytes()).hexdigest() == expected_target_sha256',
        'assert config_target.resolve() == target_path',
        'assert gate["source_path"] == target',
        'assert metric["guide_length_frozen"] is False',
        'assert metric["guide_frozen"] is True',
        'assert config["view_gate_geometry_support"] is False',
        'assert config["view_gate_length_confidence_support"] is False',
        'assert float(config["clean_flow_guide_length_anchor_weight"]) == 0.0',
        'assert config["clean_flow_guide_length_anchor_reduction"] == "mean_l1"',
        'assert gate["view_gate_geometry_support"] == 0',
        'assert gate["view_gate_length_confidence_support"] == 0',
        'assert float(metric["clean_flow_guide_length_anchor_loss"]) == 0.0',
        'assert math.isfinite(value) and value > 0.0',
        'assert float(guide_state["step"]) == 3000.0',
        'torch.equal(moment, torch.zeros_like(moment))',
        'guide_report["step"]["value"] == 3000.0',
        'guide_report["moment_nonzero_counts"] == {',
        'checkpoint="$OUTPUT_DIR/checkpoint_004000.pt"',
        'assert metric["iteration"] == 4000',
        'assert float(guide_state["step"]) == 4000.0',
        'assert "rng_state" in final_checkpoint',
        'assert "lifecycle_history" in final_checkpoint',
        'assert final_history[: len(source_history)] == source_history',
        '--checkpoint "$MIGRATED_CHECKPOINT"',
        '--checkpoint "$checkpoint"',
        '--output-dir "$RENDER_3000_ROOT"',
        '--output-dir "$RENDER_4000_ROOT"',
        '--view-ids 9',
        'sha256sum \\',
        '  "$R074_CHECKPOINT" \\',
        '  "$MIGRATED_CHECKPOINT" \\',
        '  "$checkpoint"',
    )
    for fragment in required_fragments:
        assert fragment in source

    assert source.count('"$PYTHON_PATH" -B "$MIGRATION_UTILITY"') == 1
    assert source.count('RUN_ID=') == 2
    assert "composite_psnr" not in source
    lowered = source.lower()
    for forbidden in ("s_vmem", "qsub", "qstat", "qdel", "qalter", "scheduler", "hgc"):
        assert forbidden not in lowered


def test_r079_doc_records_matched_steps_and_separate_freeze_hypothesis() -> None:
    source = DOC.read_text(encoding="utf-8").lower()

    required_phrases = (
        "r076 iteration 1000",
        "r079 iteration 4000",
        "1000 length-update steps",
        "same v8",
        "frozen 3k state",
        "freeze-state bug",
        "separate hypothesis",
        "historical step 3000",
        "optimizer",
        "rng",
        "lifecycle",
        "view 9",
        "no final psnr threshold",
        "no final length threshold",
    )
    for phrase in required_phrases:
        assert phrase in source

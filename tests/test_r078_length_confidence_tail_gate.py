from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r078_length_confidence_tail_0_3k_gate.env"
R077_CONFIG = ROOT / "configs" / "r077_confidence_owned_length_0_3k_gate.env"
LAUNCHER = (
    ROOT
    / "scripts"
    / "server"
    / "run_panda_r078_length_confidence_tail.sh"
)
DOC = ROOT / "docs" / "r078_length_confidence_tail.md"
R077_SOURCE = 'source "${CONFIG_DIR}/r077_confidence_owned_length_0_3k_gate.env"'


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


def test_r078_sources_r077_and_changes_exactly_two_assignments() -> None:
    source = CONFIG.read_text(encoding="utf-8")

    assert source.count(R077_SOURCE) == 1
    after_source = source.split(R077_SOURCE, 1)[1]
    assignments = [
        line.strip()
        for line in after_source.splitlines()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", line.strip())
    ]
    assert assignments == [
        "VIEW_GATE_LENGTH_CONFIDENCE_SUPPORT=1",
        "CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_REDUCTION=tail_concentration",
    ]

    chain = _config_source_chain(CONFIG)
    assert chain[0][0] == CONFIG
    assert chain[1][0] == R077_CONFIG
    inherited_assignments = [
        line.strip()
        for _path, config_source in chain[1:]
        for line in config_source.splitlines()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", line.strip())
    ]
    assert "CLEAN_FLOW_LENGTH_INIT_SCALE=0.30" in inherited_assignments
    assert "GUIDE_LENGTH_FREEZE_UNTIL=0" in inherited_assignments
    assert "GUIDE_FREEZE_UNTIL=9000" in inherited_assignments
    assert "ROOT_COUNT=400000" in inherited_assignments
    assert "ITERATIONS=3000" in inherited_assignments
    assert "VIEW_GATE_NORMALIZATION=equal_owner_budget" in inherited_assignments
    assert "VIEW_GATE_GEOMETRY_SUPPORT=1" in inherited_assignments
    assert "CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT=0.080" in inherited_assignments
    assert "VIEW_GATE_FLOOR=0.0" in inherited_assignments


def test_r078_launcher_enforces_strict_from_zero_h100_3k_gate() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    required_inputs = (
        ': "${PROJECT_ROOT:?Set PROJECT_ROOT to the reviewed source checkout}"',
        ': "${EXPECTED_SOURCE_COMMIT:?Set EXPECTED_SOURCE_COMMIT}"',
        ': "${RUNTIME_ROOT:?Set RUNTIME_ROOT to a new R078 runtime directory}"',
        ': "${CLEAN_FLOW_TARGET:?Set CLEAN_FLOW_TARGET to the formal Panda V8 target}"',
        ': "${EXPECTED_TARGET_SHA256:?Set EXPECTED_TARGET_SHA256}"',
        ': "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must name the granted GPU}"',
    )
    for fragment in required_inputs:
        assert fragment in source

    required_fragments = (
        'CONFIG_PATH="$PROJECT_ROOT/configs/r078_length_confidence_tail_0_3k_gate.env"',
        'PANDA_DATA_ROOT="${PANDA_DATA_ROOT:-/home/wangyy/panda-r068-v5-assets-20260827/data}"',
        'PANDA_MESH_PATH="${PANDA_MESH_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/mesh/furless.obj}"',
        'PANDA_SDF_PATH="${PANDA_SDF_PATH:-/home/wangyy/panda-r068-v5-assets-20260827/collision/panda_sdf_long512.npz}"',
        'actual_target_sha256="$(sha256sum "$CLEAN_FLOW_TARGET" | awk \'{print $1}\')"',
        '[[ "$actual_target_sha256" == "$EXPECTED_TARGET_SHA256" ]] ||',
        '[[ -z "$(git -C "$PROJECT_ROOT" status --short)" ]] ||',
        'INIT_MESH_SCALE=1.0 \\',
        'INIT_MESH_TRANSLATION=0,0,0 \\',
        'OUTPUT_DIR="$RUNTIME_ROOT/outputs/panda_r078_length_confidence_tail_0_3k_h100_20260831"',
        'PREFLIGHT_OUTPUT="$RUNTIME_ROOT/preflight/fullres_view09"',
        'RENDER_ROOT="$RUNTIME_ROOT/renders/iter_003000"',
        '"$PYTHON_PATH" -B -m pytest -q',
        'RUN_ID=panda_r078_length_confidence_tail_fullres_preflight_20260831',
        'RUN_ID=panda_r078_length_confidence_tail_0_3k_h100_20260831',
        'RUN_PREFLIGHT=1 \\',
        'RUN_BATCH_PREFLIGHT=1 \\',
        'PREFLIGHT_VIEW=9 \\',
        'STAGE1_PREFLIGHT_ONLY=1',
        'RUN_PREFLIGHT=0 \\',
        'RUN_BATCH_PREFLIGHT=0',
        'checkpoint="$OUTPUT_DIR/checkpoint_003000.pt"',
        '--checkpoint "$checkpoint"',
        '--output-dir "$RENDER_ROOT"',
        '--view-ids 9',
        '--device cuda',
        'find "$RUNTIME_ROOT/renders" -type f -print0 | sort -z |',
        'sha256sum "$checkpoint" > "$RUNTIME_ROOT/checkpoint_hashes.sha256"',
        'find "$OUTPUT_DIR" -maxdepth 1 -type f -print0 | sort -z |',
        '[[ ! -e "$RUNTIME_ROOT" ]] || fail "refusing existing runtime: $RUNTIME_ROOT"',
    )
    for fragment in required_fragments:
        assert fragment in source
    assert source.count("RUN_ID=") == 2
    assert "panda_r078_confidence_owned_length" not in source

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

    preflight_contract = (
        'assert config["clean_flow_length_init_scale"] == 0.30',
        'assert clean["clean_flow_length_init_scale"] == 0.30',
        'assert config["guide_length_freeze_until"] == 0',
        'assert config["guide_freeze_until"] == 9000',
        'assert config["root_count"] == 400000',
        'assert config["iterations"] == 1',
        '"training_checkpoint_iteration": 3000',
        'assert clean["clean_flow_source"] == target',
        'assert config["view_gated_ownership_support"] is True',
        'assert config["view_gate_geometry_support"] is True',
        'assert config["guide_roots_from_clean_flow"] is True',
        'assert config["view_gate_length_confidence_support"] is True',
        'assert config["clean_flow_guide_length_anchor_reduction"] == "tail_concentration"',
        'assert gate["view_gate_geometry_support"] == 1',
        'assert gate["view_gate_length_confidence_support"] == 1',
        'assert config["view_gate_floor"] == 0.0',
        'assert config["view_gate_normalization"] == "equal_owner_budget"',
        'assert gate["normalization_mode"] == "equal_owner_budget"',
        'assert abs(gate["supported_guide_expected_multiplier_mean"] - 1.0) < 1.0e-6',
        'assert metric["guide_length_frozen"] is False',
        'assert metric["guide_frozen"] is True',
        'assert config["clean_flow_guide_length_anchor_weight"] == 0.08',
        'length_anchor_loss = float(metric["clean_flow_guide_length_anchor_loss"])',
        'metric["clean_flow_guide_length_anchor_reliable_fraction"]',
        'assert math.isfinite(length_anchor_loss)',
        'assert length_anchor_loss > 0.0',
        'assert math.isfinite(length_anchor_fraction)',
        'assert length_anchor_fraction > 0.0',
        'initial_effective_mean = float(metric["effective_groom"]["length"]["mean"])',
        'with np.load(target_path, allow_pickle=False) as target_data:',
        'target_confidence = np.clip(target_weight / weight_q95, 0.0, 1.0)',
        'identity_q05, identity_q95 = np.quantile(shell_h[length_source], [0.05, 0.95])',
        'short_scale_q05 = float(identity_q05 * config["clean_flow_length_init_scale"])',
        'assert clean["clean_flow_guide_length_init_reliable_count"] > 0',
        'assert clean["clean_flow_guide_length_init_filled_count"] == config["guide_root_count"]',
        'assert short_scale_q05 <= initial_effective_mean <= short_scale_q95',
    )
    for fragment in preflight_contract:
        assert fragment in source

    assert "composite_psnr" not in source
    assert "final_effective" not in source
    assert "effective_max" not in source
    assert "r077" not in source.lower()
    assert "r074" not in source.lower()
    assert "r075" not in source.lower()
    assert "r076" not in source.lower()
    forbidden = (
        "resume",
        "s_vmem",
        "qsub",
        "qstat",
        "qdel",
        "qalter",
        "scheduler",
        "release",
        "kill",
    )
    lowered = source.lower()
    for term in forbidden:
        assert term not in lowered, term


def test_r078_doc_records_causal_evidence_and_method_boundary() -> None:
    source = DOC.read_text(encoding="utf-8").lower()

    required_phrases = (
        "r077 causal evidence",
        "top 1%",
        "45/45",
        "24/45",
        "zero length confidence",
        "21/45",
        "positive length confidence",
        "0.1708",
        "untrusted image gradients",
        "sparse trusted tails",
        "source-guide render-path-only confidence gate",
        "view-independent propagation",
        "weighted mean-l1",
        "l4-l2",
        "no physical length cap",
        "no species, region, or view rule",
        "no percentile behavior",
        "no final psnr threshold",
        "no final length threshold",
        "bounded 3k",
    )
    for phrase in required_phrases:
        assert phrase in source

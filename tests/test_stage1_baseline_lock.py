from pathlib import Path
import re

from tools.verify_stage1_baseline import PROJECT_ROOT, verify_baseline_lock


CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1_baseline.env"
R042_LOCK_PATH = PROJECT_ROOT / "configs" / "r042_exact_lifecycle_selection.lock.json"
R043_LOCK_PATH = PROJECT_ROOT / "configs" / "r043_density_matched_render_support.lock.json"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "server" / "run_white_tiger_stage1.sh"


def load_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        assert key not in result, f"duplicate baseline key: {key}"
        result[key] = value
    return result


def test_frozen_r036_files_match_lock() -> None:
    report = verify_baseline_lock()
    assert report["ok"], report["failures"]


def test_active_r042_files_match_lock() -> None:
    report = verify_baseline_lock(lock_path=R042_LOCK_PATH)
    assert report["baseline_id"] == "stage1-r042"
    assert report["source_ref"] == "stage1-r042"
    assert report["ok"], report["failures"]


def test_active_r043_files_match_lock() -> None:
    report = verify_baseline_lock(lock_path=R043_LOCK_PATH)
    assert report["baseline_id"] == "stage1-r043"
    assert report["source_ref"] == "stage1-r043"
    assert report["ok"], report["failures"]


def test_r036_semantic_contract() -> None:
    env = load_env(CONFIG_PATH)
    expected = {
        "CLEAN_FLOW_TARGET": "baseline_inputs/v4_surface_direction/guide_flow3d_shell_targets_exclude_004_024_025.npz",
        "EXPECTED_WIDTH": "1920",
        "EXPECTED_HEIGHT": "1080",
        "ROOT_COUNT": "100000",
        "GUIDE_ROOT_COUNT": "4500",
        "CHILD_COUNT": "4",
        "CLEAN_FLOW_INIT": "1",
        "CLEAN_FLOW_LENGTH_INIT": "1",
        "GUIDE_ROOTS_FROM_CLEAN_FLOW": "1",
        "RENDER_GEOMETRY_PARAMETERIZATION": "zero_centered_asinh_log_length_residual",
        "GUIDE_FREEZE_UNTIL": "9000",
        "GUIDE_RESIDUAL_UNLOCK_START": "10000",
        "GUIDE_RESIDUAL_UNLOCK_END": "20000",
        "GUIDE_COVERAGE_RESIDUAL_UNLOCK_START": "1000",
        "GUIDE_COVERAGE_RESIDUAL_UNLOCK_END": "7000",
        "DENSIFY_WARMUP": "600",
        "DENSIFY_INTERVAL": "100",
        "DENSIFY_UNTIL": "20000",
        "DENSIFY_SCORE_THRESHOLD": "0.00075",
        "GUIDE_DENSIFY_POLICY": "surface_attribution_local_max",
        "GUIDE_DENSIFY_START": "11000",
        "GUIDE_DENSIFY_INTERVAL": "200",
        "GUIDE_DENSIFY_UNTIL": "16000",
        "GUIDE_DENSIFY_MAX_SPLITS_PER_EVENT": "0",
        "DENSIFY_RESIDUAL_MODE": "pixel_to_root",
        "LIFECYCLE_SCORE_MODE": "raw",
        "MAX_SPLITS_PER_EVENT": "0",
        "SMOOTH_GRAPH_MODE": "surface_hierarchical",
        "SMOOTH_FIELD_METRIC": "surface_covariant_full",
        "GUIDE_LENGTH_SMOOTH_MODE": "intrinsic_density_invariant",
        "LOSS_MASK_EDGE_KERNEL": "5",
        "EFFECTIVE_SMOOTH_WEIGHT": "0.006",
        "RENDER_LENGTH_PRIOR_REDUCTION": "tail_concentration_handoff",
        "MIN_SEGMENTS": "10",
        "SEGMENT_LENGTH_ORIGIN": "0.010",
        "MESH_DEPTH_CLIPPING": "1",
        "MESH_BACKING_COMPOSITING": "1",
        "RANDOM_MESH_BACKING_TEXTURE": "1",
        "RANDOM_BACKING_COLOR": "1",
        "SHAPE_CURL_SCALE": "0.0",
        "SHAPE_FRIZZ_SCALE": "0.0",
        "PRUNE_INTERVAL": "0",
    }
    assert {key: env.get(key) for key in expected} == expected

    retired_prefixes = (
        "OVERLONG_SPLIT_",
        "SCREEN_FOOTPRINT_SPLIT_",
        "DARK_STROKE_",
        "NEUTRAL_SCREEN_",
        "EARLY_CAPACITY_",
    )
    assert not any(key.startswith(retired_prefixes) for key in env)


def test_runner_requires_every_declared_baseline_key() -> None:
    env = load_env(CONFIG_PATH)
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    match = re.search(r"required_config=\(\n(?P<body>.*?)\n\)", runner, re.DOTALL)
    assert match is not None
    required = {
        line.strip()
        for line in match.group("body").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert not (required - env.keys())
    assert 'CONFIG_PATH="${CONFIG_PATH:-}"' in runner

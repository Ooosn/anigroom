from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R040 = PROJECT_ROOT / "configs" / "r040_child1_dense_render_0_30k.env"
R043 = PROJECT_ROOT / "configs" / "r043_density_matched_render_support_0_30k.env"
TRAINER = PROJECT_ROOT / "tools" / "train_white_tiger_stage1.py"
LIFECYCLE = PROJECT_ROOT / "anigroom" / "roots" / "lifecycle.py"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        assert key not in values
        values[key] = value
    return values


def test_r043_only_expands_render_domain_support() -> None:
    r040 = load_env(R040)
    r043 = load_env(R043)

    assert set(r043) == set(r040)
    changed = {
        key: (r040[key], r043[key])
        for key in r043
        if r040[key] != r043[key]
    }
    assert changed == {"SMOOTH_GRAPH_K": ("8", "32")}
    assert r043["ROOT_COUNT"] == "400000"
    assert r043["CHILD_COUNT"] == "1"
    assert r043["GUIDE_INTERPOLATION_K"] == "8"
    assert "STRAND_SHAPE_SMOOTH_WEIGHT" not in r043


def test_render_and_guide_support_are_routed_independently() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    lifecycle = LIFECYCLE.read_text(encoding="utf-8")

    assert trainer.count("neighbor_count=config.smooth_graph_k") == 2
    assert trainer.count("k=config.guide_interpolation_k") >= 2
    assert "smooth_graph_k=config.guide_interpolation_k" in trainer
    assert "neighbor_count=self.guide_interpolation_k" in trainer
    assert "neighbor_count: int = 8" not in lifecycle


def test_sample_level_strand_smoothing_remains_removed() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    r043 = R043.read_text(encoding="utf-8")

    assert "strand_shape_consistency_loss" not in trainer
    assert "strand_shape_smooth_weight" not in trainer
    assert "STRAND_SHAPE_SMOOTH_WEIGHT" not in r043

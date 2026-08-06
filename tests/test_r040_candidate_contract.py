from pathlib import Path

import torch

from anigroom.grooming.strand_gaussians import expand_child_strands


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R039 = PROJECT_ROOT / "configs" / "r039_brush_centerline_0_30k.env"
R040 = PROJECT_ROOT / "configs" / "r040_child1_dense_render_0_30k.env"
TRAINER = PROJECT_ROOT / "tools" / "train_white_tiger_stage1.py"
RUNNER = PROJECT_ROOT / "scripts" / "server" / "run_white_tiger_stage1.sh"


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


def test_r040_replaces_child_expansion_with_dense_render_roots() -> None:
    r039 = load_env(R039)
    r040 = load_env(R040)

    assert set(r040) == set(r039) - {"STRAND_SHAPE_SMOOTH_WEIGHT"}
    changed = {
        key: (r039[key], r040[key])
        for key in r040
        if r039[key] != r040[key]
    }
    assert changed == {
        "CHILD_COUNT": ("4", "1"),
        "ROOT_COUNT": ("100000", "400000"),
    }
    assert int(r039["CHILD_COUNT"]) * int(r039["ROOT_COUNT"]) == int(
        r040["CHILD_COUNT"]
    ) * int(r040["ROOT_COUNT"])


def test_sample_level_strand_smoothing_is_not_in_the_r040_runtime() -> None:
    trainer = TRAINER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "strand_shape_consistency_loss" not in trainer
    assert "strand_shape_smooth_weight" not in trainer
    assert "STRAND_SHAPE_SMOOTH_WEIGHT" not in runner
    assert "--strand-shape-smooth-weight" not in runner


def test_child_count_one_is_an_exact_identity() -> None:
    strands = torch.randn(7, 9, 3)
    widths = torch.rand(7, 9, 1)
    colors = torch.rand(7, 9, 3)
    opacities = torch.rand(7, 9, 1)
    normals = torch.nn.functional.normalize(torch.randn(7, 3), dim=-1)

    expanded = expand_child_strands(
        strands,
        widths,
        colors,
        opacities,
        normals,
        torch.rand(7, 1),
        torch.rand(7, 1),
        child_count=1,
    )

    for actual, expected in zip(expanded[:4], (strands, widths, colors, opacities)):
        assert actual.data_ptr() == expected.data_ptr()
    assert torch.equal(expanded[4], torch.arange(7))

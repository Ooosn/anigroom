from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from tools import train_white_tiger_stage1 as stage1


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "tools" / "train_white_tiger_stage1.py"
SERVER_LAUNCHER = ROOT / "scripts" / "server" / "run_white_tiger_stage1.sh"

GUIDE_PARAMETER_NAMES = (
    "guide_length_raw",
    "guide_root_width_raw",
    "guide_tip_width_ratio_raw",
    "guide_width_taper_raw",
    "guide_brush_stiffness_raw",
    "guide_curl_radius_ratio_raw",
    "guide_curl_turns_raw",
    "guide_child_radius_raw",
    "guide_clump_strength_raw",
    "guide_direction_local_raw",
)

REQUIRED_PARSER_ARGS = (
    "--densify-warmup",
    "1",
    "--densify-interval",
    "2",
    "--densify-until",
    "3",
    "--densify-score-threshold",
    "0.1",
    "--densify-min-contribution",
    "0.2",
    "--max-splits-per-event",
    "4",
    "--split-children-per-parent",
    "2",
    "--split-neighbor-count",
    "3",
    "--split-candidate-rings",
    "1",
    "--split-candidate-face-count",
    "5",
    "--split-min-child-distance",
    "0.01",
    "--prune-start",
    "10",
    "--prune-interval",
    "5",
    "--prune-min-contribution",
    "0.1",
    "--prune-min-opacity",
    "0.2",
    "--prune-max-fraction",
    "0.3",
)


def _guide_fixture() -> SimpleNamespace:
    model = SimpleNamespace(guide_enabled=lambda: True)
    for name in GUIDE_PARAMETER_NAMES:
        setattr(
            model,
            name,
            torch.nn.Parameter(torch.ones((2, 1), dtype=torch.float32)),
        )
    return model


def _seed_guide_gradients(model: SimpleNamespace) -> None:
    for name in GUIDE_PARAMETER_NAMES:
        parameter = getattr(model, name)
        parameter.grad = torch.full_like(parameter, 3.0)


def _assert_guide_gradients(
    model: SimpleNamespace,
    preserved: set[str],
) -> None:
    for name in GUIDE_PARAMETER_NAMES:
        parameter = getattr(model, name)
        expected = 3.0 if name in preserved else 0.0
        torch.testing.assert_close(
            parameter.grad,
            torch.full_like(parameter, expected),
        )


def _config(**overrides: object) -> stage1.Stage1Config:
    values: dict[str, object] = {
        "data_root": "data",
        "mesh_path": "mesh.obj",
        "output_dir": "output",
    }
    values.update(overrides)
    return stage1.Stage1Config(**values)


def test_zero_guide_gradients_defaults_freeze_all_guide_parameters() -> None:
    model = _guide_fixture()
    _seed_guide_gradients(model)

    stage1.zero_guide_gradients(model)

    _assert_guide_gradients(model, preserved=set())


def test_zero_guide_gradients_can_preserve_length_while_freezing_other_guides() -> None:
    model = _guide_fixture()
    _seed_guide_gradients(model)

    stage1.zero_guide_gradients(
        model,
        freeze_length=False,
        freeze_other=True,
    )

    _assert_guide_gradients(model, preserved={"guide_length_raw"})


def test_zero_guide_gradients_can_freeze_length_while_preserving_other_guides() -> None:
    model = _guide_fixture()
    _seed_guide_gradients(model)

    stage1.zero_guide_gradients(
        model,
        freeze_length=True,
        freeze_other=False,
    )

    _assert_guide_gradients(
        model,
        preserved=set(GUIDE_PARAMETER_NAMES) - {"guide_length_raw"},
    )


def test_resolved_guide_length_freeze_until_supports_fallback_and_explicit_zero() -> None:
    fallback = _config(guide_freeze_until=9, guide_length_freeze_until=-1)
    explicit_zero = _config(guide_freeze_until=9, guide_length_freeze_until=0)

    assert stage1.resolved_guide_length_freeze_until(fallback) == 9
    assert stage1.resolved_guide_length_freeze_until(explicit_zero) == 0


def test_training_loop_uses_independent_guide_freeze_flags_and_metric() -> None:
    source = TRAINER.read_text(encoding="utf-8")

    for fragment in (
        "guide_length_freeze_until = resolved_guide_length_freeze_until(config)",
        "guide_length_frozen = (",
        "if guide_frozen or guide_length_frozen:",
        "freeze_length=guide_length_frozen,",
        "freeze_other=guide_frozen,",
        '"guide_frozen": bool(guide_frozen),',
        '"guide_length_frozen": bool(guide_length_frozen),',
    ):
        assert fragment in source, fragment


def test_parser_and_config_construct_guide_length_freeze_field() -> None:
    parser = stage1.build_arg_parser()
    default_args = parser.parse_args(list(REQUIRED_PARSER_ARGS))
    default_config = stage1.config_from_args(default_args)

    assert parser.get_default("guide_length_freeze_until") == -1
    assert default_args.guide_length_freeze_until == -1
    assert default_config.guide_length_freeze_until == -1

    explicit_args = parser.parse_args(
        [
            *REQUIRED_PARSER_ARGS,
            "--guide-freeze-until",
            "9",
            "--guide-length-freeze-until",
            "0",
        ]
    )
    explicit_config = stage1.config_from_args(explicit_args)

    assert explicit_config.guide_freeze_until == 9
    assert explicit_config.guide_length_freeze_until == 0


def test_server_launcher_falls_back_and_forwards_guide_length_freeze() -> None:
    source = SERVER_LAUNCHER.read_text(encoding="utf-8")
    fallback = 'GUIDE_LENGTH_FREEZE_UNTIL="${GUIDE_LENGTH_FREEZE_UNTIL:-$GUIDE_FREEZE_UNTIL}"'

    assert fallback in source
    assert source.index(fallback) < source.index("cmd=(")
    assert '--guide-freeze-until "$GUIDE_FREEZE_UNTIL"' in source
    assert '--guide-length-freeze-until "$GUIDE_LENGTH_FREEZE_UNTIL"' in source

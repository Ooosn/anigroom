from __future__ import annotations

import importlib.util
import sys
import types

if importlib.util.find_spec("gsplat") is None:
    gsplat = types.ModuleType("gsplat")
    rendering = types.ModuleType("gsplat.rendering")
    rendering.rasterization = lambda *args, **kwargs: None
    gsplat.rendering = rendering
    sys.modules["gsplat"] = gsplat
    sys.modules["gsplat.rendering"] = rendering

from tools import train_white_tiger_stage1 as trainer


REQUIRED_ARGS = [
    "--densify-warmup",
    "0",
    "--densify-interval",
    "0",
    "--densify-until",
    "0",
    "--densify-score-threshold",
    "0",
    "--densify-min-contribution",
    "0",
    "--max-splits-per-event",
    "0",
    "--split-children-per-parent",
    "1",
    "--split-neighbor-count",
    "1",
    "--split-candidate-rings",
    "1",
    "--split-candidate-face-count",
    "1",
    "--split-min-child-distance",
    "0",
    "--prune-start",
    "0",
    "--prune-interval",
    "0",
    "--prune-min-contribution",
    "0",
    "--prune-min-opacity",
    "0",
    "--prune-max-fraction",
    "0",
]


def _capture_main(monkeypatch, *alignment_args: str):
    captured = []
    monkeypatch.setattr(
        trainer,
        "train_white_tiger_stage1",
        lambda config: captured.append(config),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_white_tiger_stage1.py", *alignment_args, *REQUIRED_ARGS],
    )

    trainer.main()

    assert len(captured) == 1
    return captured[0]


def test_explicit_panda_alignment_survives_white_tiger_config(monkeypatch) -> None:
    config = _capture_main(
        monkeypatch,
        "--init-mesh-scale",
        "1.0",
        "--init-mesh-translation",
        "0",
        "0",
        "0",
    )

    assert config.init_mesh_scale == 1.0
    assert config.init_mesh_translation == (0.0, 0.0, 0.0)


def test_default_alignment_arguments_receive_white_tiger_defaults(monkeypatch) -> None:
    config = _capture_main(monkeypatch)

    assert config.init_mesh_scale == 1.28
    assert config.init_mesh_translation == (0.0, 0.32, 0.02)

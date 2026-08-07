from dataclasses import asdict

import pytest

from tools.train_white_tiger_stage1 import (
    Stage1Config,
    load_training_checkpoint,
    restored_lifecycle_history,
    stage1_config_from_checkpoint_mapping,
)


def checkpoint_config(**extra):
    data = asdict(Stage1Config(data_root="data", mesh_path="mesh.obj", output_dir="output"))
    data["init_mesh_translation"] = list(data["init_mesh_translation"])
    data.update(extra)
    return data


def test_checkpoint_loader_retries_numpy2_pickle_on_numpy1(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            error = ModuleNotFoundError("No module named 'numpy._core'")
            error.name = "numpy._core"
            raise error
        assert "numpy._core.multiarray" in __import__("sys").modules
        return {"iteration": 1}

    monkeypatch.setattr("tools.train_white_tiger_stage1.torch.load", fake_load)
    checkpoint = load_training_checkpoint(tmp_path / "checkpoint.pt")

    assert checkpoint == {"iteration": 1}
    assert len(calls) == 2


def test_current_checkpoint_config_loads_without_migration() -> None:
    config = stage1_config_from_checkpoint_mapping(checkpoint_config())
    assert config.init_mesh_translation == (0.0, 0.32, 0.02)


def test_old_checkpoint_defaults_geometry_residual_smooth_scale_to_one() -> None:
    data = checkpoint_config()
    del data["geometry_residual_smooth_scale"]

    config = stage1_config_from_checkpoint_mapping(data)

    assert config.geometry_residual_smooth_scale == 1.0


def test_unknown_checkpoint_field_is_rejected() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        stage1_config_from_checkpoint_mapping(
            checkpoint_config(unknown_future_field=1)
        )


def test_resume_preserves_complete_lifecycle_history() -> None:
    source = {
        "lifecycle_history": [
            {"iteration": 100, "root_count_after": 120},
            {"iteration": 200, "root_count_after": 135},
        ]
    }
    restored = restored_lifecycle_history(source, start_iteration=200)

    assert restored == source["lifecycle_history"]
    assert restored is not source["lifecycle_history"]


@pytest.mark.parametrize(
    "history,match",
    [
        ([{"iteration": 200}, {"iteration": 200}], "strictly increasing"),
        ([{"iteration": 300}], "after checkpoint iteration"),
    ],
)
def test_resume_rejects_invalid_lifecycle_history(history, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        restored_lifecycle_history(
            {"lifecycle_history": history},
            start_iteration=200,
        )

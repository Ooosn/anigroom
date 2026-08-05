from tools.train_white_tiger_stage1 import Stage1Config, lifecycle_statistics_active


def make_config(**overrides: object) -> Stage1Config:
    values: dict[str, object] = {
        "data_root": "data",
        "mesh_path": "mesh.obj",
        "output_dir": "output",
        "densify_warmup": 600,
        "densify_interval": 100,
        "densify_until": 9000,
        "guide_densify_start": 0,
        "guide_densify_interval": 0,
        "guide_densify_until": 0,
        "prune_start": 999999,
        "prune_interval": 0,
    }
    values.update(overrides)
    return Stage1Config(**values)


def test_render_lifecycle_keeps_final_window_and_stops_after_9000() -> None:
    config = make_config()
    assert lifecycle_statistics_active(config, 1, guide_enabled=True)
    assert lifecycle_statistics_active(config, 8999, guide_enabled=True)
    assert lifecycle_statistics_active(config, 9000, guide_enabled=True)
    assert not lifecycle_statistics_active(config, 9001, guide_enabled=True)
    assert not lifecycle_statistics_active(config, 30000, guide_enabled=True)


def test_future_guide_or_prune_event_keeps_statistics_active() -> None:
    guide_config = make_config(
        guide_densify_start=11000,
        guide_densify_interval=200,
        guide_densify_until=16000,
    )
    assert lifecycle_statistics_active(guide_config, 10000, guide_enabled=True)
    assert not lifecycle_statistics_active(guide_config, 10000, guide_enabled=False)
    assert not lifecycle_statistics_active(guide_config, 16001, guide_enabled=True)

    prune_config = make_config(prune_start=1000, prune_interval=200)
    assert lifecycle_statistics_active(prune_config, 30000, guide_enabled=False)

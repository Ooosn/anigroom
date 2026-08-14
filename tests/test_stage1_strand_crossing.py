from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from anigroom.collision.strand_crossing import StrandCrossingActiveSet
from anigroom.grooming import GroomRanges
from anigroom.mesh_roots import TriangleMesh
from tools.train_white_tiger_stage1 import (
    Stage1Config,
    WhiteTigerStage1Model,
    restore_strand_crossing_state,
    validate_strand_crossing_config,
)
from tools.calibrate_strand_crossing_loss import render_parameter_args


def base_config() -> Stage1Config:
    return Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        child_count=1,
        iterations=30000,
        densify_until=9000,
        prune_start=999999,
        prune_interval=0,
    )


def enabled_config() -> Stage1Config:
    return replace(
        base_config(),
        strand_crossing_support=True,
        strand_crossing_weight=0.01,
        strand_crossing_refresh_interval=1000,
    )


def test_calibration_uses_the_canonical_render_parameter_signature() -> None:
    config = replace(
        base_config(),
        samples=48,
        child_count=1,
        min_segments=7,
        segment_length_origin=0.012,
        segments_per_unit_length=81.0,
        segments_per_unit_complexity=19.0,
        gaussian_length_overlap=1.35,
    )
    assert render_parameter_args(config) == (
        48,
        1,
        7,
        0.012,
        81.0,
        19.0,
        1.35,
    )


def one_pair_active_set(
    first_root: int = 0,
    second_root: int = 1,
) -> StrandCrossingActiveSet:
    axis = np.asarray([[-1.0, 0.0, -1.0]], dtype=np.float32)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    return StrandCrossingActiveSet(
        first_root_indices=np.asarray([first_root], dtype=np.int64),
        second_root_indices=np.asarray([second_root], dtype=np.int64),
        first_progress=np.asarray([0.5], dtype=np.float32),
        second_progress=np.asarray([0.5], dtype=np.float32),
        separation_axes=axis,
        angle_weights=np.asarray([1.0], dtype=np.float32),
        discovery_overlap=np.asarray([1.0], dtype=np.float32),
        discovery_scores=np.asarray([1.0], dtype=np.float32),
        source_segment_count=2,
    )


def test_crossing_configuration_is_explicit_and_topology_safe() -> None:
    with pytest.raises(ValueError, match="weight must be zero"):
        validate_strand_crossing_config(
            replace(base_config(), strand_crossing_weight=0.01)
        )
    with pytest.raises(ValueError, match="positive loss weight"):
        validate_strand_crossing_config(
            replace(base_config(), strand_crossing_support=True)
        )
    with pytest.raises(ValueError, match="topology-stable interval"):
        validate_strand_crossing_config(
            replace(enabled_config(), densify_until=30000)
        )
    with pytest.raises(ValueError, match="pruning to remain disabled"):
        validate_strand_crossing_config(
            replace(enabled_config(), prune_start=12000, prune_interval=100)
        )
    validate_strand_crossing_config(enabled_config())


def test_crossing_checkpoint_state_restores_and_rejects_stale_root_ids() -> None:
    active = one_pair_active_set()
    checkpoint = {
        "strand_crossing_active_set": active.checkpoint_state(),
        "strand_crossing_last_refresh_iteration": 12000,
        "strand_crossing_history": [{"iteration": 12000}],
    }
    cpu_state, torch_state, last_refresh, history = restore_strand_crossing_state(
        enabled_config(),
        checkpoint,
        root_count=2,
        device=torch.device("cpu"),
    )
    assert cpu_state is not None and cpu_state.pair_count == 1
    assert torch_state is not None and torch_state.pair_count == 1
    assert last_refresh == 12000
    assert history == [{"iteration": 12000}]

    stale = one_pair_active_set(0, 2)
    checkpoint["strand_crossing_active_set"] = stale.checkpoint_state()
    with pytest.raises(RuntimeError, match="stale render-root indices"):
        restore_strand_crossing_state(
            enabled_config(),
            checkpoint,
            root_count=2,
            device=torch.device("cpu"),
        )


def test_model_crossing_loss_updates_geometry_not_width_or_global_pose() -> None:
    mesh = TriangleMesh(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    model = WhiteTigerStage1Model(
        mesh,
        np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]], dtype=np.float32),
        GroomRanges(),
        torch.device("cpu"),
        init_scale=1.75,
        init_translation=(0.2, 0.3, -0.4),
        init_groom_length=0.018,
        max_child_count=1,
    )
    active = one_pair_active_set().to_torch("cpu")

    _, _, _, _, _, crossing_loss, stats = model.render_parameters(
        16,
        1,
        10,
        0.010,
        84.19047619047619,
        23.771428571428572,
        1.45,
        strand_crossing_active_set=active,
    )
    assert stats["active_pair_count"] == 1
    assert float(crossing_loss.detach()) > 0.0
    crossing_loss.backward()

    assert model.groom.length_raw.grad is not None
    assert float(model.groom.length_raw.grad.abs().sum()) > 0.0
    assert model.bary_logits.grad is not None
    assert float(model.bary_logits.grad.abs().sum()) > 0.0
    assert model.groom.root_width_raw.grad is None
    assert model.groom.tip_width_ratio_raw.grad is None
    assert model.translation.grad is None or torch.count_nonzero(model.translation.grad) == 0
    assert model.log_scale.grad is None or torch.count_nonzero(model.log_scale.grad) == 0

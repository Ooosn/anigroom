from dataclasses import replace

import numpy as np
import pytest
import torch

from anigroom.grooming import GaussianRGBResidualField, GroomRanges
from anigroom.mesh_roots import TriangleMesh
from anigroom.roots.lifecycle import RootStructureUpdate
from tools.train_white_tiger_stage1 import (
    Stage1Config,
    WhiteTigerStage1Model,
    gaussian_rgb_residual_multiplier_for_iteration,
    make_stage1_optimizer,
    optimizer_row_transition,
    rebuild_stage1_optimizer_with_state,
    shape_detail_multiplier_for_iteration,
    stage1_optimizer_param_names,
)


def make_model(
    *,
    render_count: int = 4,
    gaussian_rgb_residual_support: bool = True,
) -> WhiteTigerStage1Model:
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
    face_normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    face_tangents = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    face_ids = np.arange(render_count, dtype=np.int64) % 2
    barycentric = np.asarray(
        [
            [0.72 - 0.04 * (index % 3), 0.14 + 0.02 * (index % 3), 0.14 + 0.02 * (index % 3)]
            for index in range(render_count)
        ],
        dtype=np.float32,
    )
    guide_face_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    guide_barycentric = np.asarray(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
        ],
        dtype=np.float32,
    )
    return WhiteTigerStage1Model(
        mesh,
        face_normals,
        face_tangents,
        face_ids,
        barycentric,
        GroomRanges(),
        torch.device("cpu"),
        init_groom_length=0.018,
        max_child_count=1,
        gaussian_rgb_residual_support=gaussian_rgb_residual_support,
        gaussian_rgb_residual_control_points=6,
        gaussian_rgb_residual_scale=0.20,
        guide_face_ids=guide_face_ids,
        guide_barycentric=guide_barycentric,
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
    )


def make_config() -> Stage1Config:
    return Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        child_count=1,
        gaussian_rgb_residual_support=True,
        gaussian_rgb_residual_control_points=6,
        gaussian_rgb_residual_scale=0.20,
        gaussian_rgb_residual_initial_multiplier=0.0,
        gaussian_rgb_residual_unlock_start=10_000,
        gaussian_rgb_residual_unlock_end=20_000,
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
    )


def test_zero_profile_and_zero_multiplier_are_exact_noops() -> None:
    field = GaussianRGBResidualField(2, 6, 0.2)
    colors = torch.rand(3, 3, requires_grad=True)
    roots = torch.tensor([0, 0, 1])
    segments = torch.tensor([0, 1, 0])
    counts = torch.tensor([2, 1])

    zero_profile = field.apply_to_colors(
        colors,
        roots,
        segments,
        counts,
        multiplier=1.0,
    )
    torch.testing.assert_close(zero_profile, colors, rtol=0.0, atol=0.0)

    with torch.no_grad():
        field.raw.fill_(0.4)
    disabled = field.apply_to_colors(
        colors,
        roots,
        segments,
        counts,
        multiplier=0.0,
    )
    assert disabled is colors
    disabled.sum().backward()
    assert field.raw.grad is None
    torch.testing.assert_close(colors.grad, torch.ones_like(colors))


def test_residual_is_segment_specific_and_normalized_across_segment_counts() -> None:
    field = GaussianRGBResidualField(2, 6, 0.2)
    with torch.no_grad():
        ramp = torch.linspace(-0.7, 0.7, 6)
        field.raw[0, :, 0].copy_(ramp)
        field.raw[1].copy_(field.raw[0])

    colors = torch.full((4, 3), 0.5)
    result = field.apply_to_colors(
        colors,
        torch.tensor([0, 0, 0, 0]),
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([4, 1]),
        multiplier=1.0,
    )
    assert bool(torch.all(result[1:, 0] > result[:-1, 0]))
    torch.testing.assert_close(result[:, 1:], colors[:, 1:])

    # (2 + 0.5) / 10 == (4 + 0.5) / 18 == 0.25.
    same_position = field.segment_residual(
        torch.tensor([0, 1]),
        torch.tensor([2, 4]),
        torch.tensor([10, 18]),
        multiplier=1.0,
    )
    torch.testing.assert_close(same_position[0], same_position[1], atol=1.0e-7, rtol=1.0e-7)


def test_gradient_reaches_only_profile_controls_supporting_the_gaussian() -> None:
    field = GaussianRGBResidualField(1, 6, 0.2)
    residual = field.segment_residual(
        torch.tensor([0]),
        torch.tensor([4]),
        torch.tensor([10]),
        multiplier=1.0,
    )
    residual.sum().backward()

    nonzero_controls = torch.nonzero(field.raw.grad[0].abs().sum(dim=-1) > 0.0).reshape(-1)
    # Position 0.45 maps to control coordinate 2.25.
    torch.testing.assert_close(nonzero_controls, torch.tensor([2, 3]))


def test_chunked_statistics_are_exact() -> None:
    field = GaussianRGBResidualField(5, 6, 0.2)
    with torch.no_grad():
        field.raw.copy_(torch.linspace(-1.0, 1.0, field.raw.numel()).reshape_as(field.raw))
    multiplier = 0.7
    decoded = torch.tanh(field.raw) * field.scale * multiplier
    stats = field.stats(multiplier=multiplier, root_chunk_size=2)

    assert stats["abs_mean"] == pytest.approx(float(decoded.abs().mean()), rel=1.0e-6)
    assert stats["rms"] == pytest.approx(float(decoded.square().mean().sqrt()), rel=1.0e-6)
    assert stats["abs_max"] == pytest.approx(float(decoded.abs().max()), rel=1.0e-6)
    assert stats["active_fraction"] == pytest.approx(
        float((decoded.abs() > 1.0e-4).float().mean()),
        rel=1.0e-6,
    )
    assert stats["saturation_fraction"] == pytest.approx(
        float((torch.tanh(field.raw).abs() > 0.95).float().mean()),
        rel=1.0e-6,
    )


def test_child_expansion_fails_instead_of_falling_back() -> None:
    mesh = TriangleMesh(
        vertices=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="child_count=1"):
        WhiteTigerStage1Model(
            mesh,
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([0], dtype=np.int64),
            np.asarray([[0.8, 0.1, 0.1]], dtype=np.float32),
            GroomRanges(),
            torch.device("cpu"),
            max_child_count=4,
            gaussian_rgb_residual_support=True,
        )


def test_lifecycle_keeps_survivors_and_zero_initializes_children_and_adam() -> None:
    model = make_model()
    config = make_config()
    optimizer = make_stage1_optimizer(model, config)
    names = stage1_optimizer_param_names(model, config)
    parameter = model.gaussian_rgb_residual.raw
    with torch.no_grad():
        for row in range(parameter.shape[0]):
            parameter[row].fill_(float(row + 1) / 10.0)
    moment = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(parameter)
    optimizer.state[parameter] = {
        "step": torch.tensor(9.0),
        "exp_avg": moment.clone(),
        "exp_avg_sq": moment.square(),
    }

    update = RootStructureUpdate(
        parent_indices=torch.tensor([0], dtype=torch.long),
        child_parent_indices=torch.tensor([0, 0], dtype=torch.long),
        new_face_ids=torch.tensor([0, 0], dtype=torch.long),
        new_barycentric=torch.tensor(
            [[0.55, 0.25, 0.20], [0.25, 0.55, 0.20]],
            dtype=torch.float32,
        ),
        prune_mask=torch.tensor([True, False, False, False]),
        scores={},
    )
    transition = optimizer_row_transition(update, old_count=4)
    model.apply_structure_update(update, neighbor_count=4)

    torch.testing.assert_close(model.gaussian_rgb_residual.raw[:3], parameter.detach()[1:])
    assert torch.count_nonzero(model.gaussian_rgb_residual.raw[3:]) == 0

    rebuilt, report = rebuild_stage1_optimizer_with_state(
        model,
        config,
        optimizer,
        names,
        render_transition=transition,
    )
    migrated = rebuilt.state[model.gaussian_rgb_residual.raw]
    torch.testing.assert_close(migrated["exp_avg"][:3], moment[1:])
    torch.testing.assert_close(migrated["exp_avg"][3:], torch.zeros_like(migrated["exp_avg"][3:]))
    assert float(migrated["step"]) == 9.0
    assert report["render"]["zero_initialized_child_count"] == 2


def test_state_dict_roundtrip_and_support_off_compatibility() -> None:
    enabled = make_model()
    enabled_clone = make_model()
    enabled_clone.load_state_dict(enabled.state_dict(), strict=True)
    torch.testing.assert_close(
        enabled_clone.gaussian_rgb_residual.raw,
        enabled.gaussian_rgb_residual.raw,
    )

    disabled = make_model(gaussian_rgb_residual_support=False)
    disabled_clone = make_model(gaussian_rgb_residual_support=False)
    disabled_clone.load_state_dict(disabled.state_dict(), strict=True)
    assert disabled.gaussian_rgb_residual is None
    assert not any(name.startswith("gaussian_rgb_residual") for name in disabled.state_dict())


def test_gaussian_rgb_residual_uses_shared_linear_schedule() -> None:
    config = make_config()
    assert gaussian_rgb_residual_multiplier_for_iteration(config, 9_999) == 0.0
    assert gaussian_rgb_residual_multiplier_for_iteration(config, 10_000) == 0.0
    assert gaussian_rgb_residual_multiplier_for_iteration(config, 15_000) == 0.5
    assert gaussian_rgb_residual_multiplier_for_iteration(config, 20_000) == 1.0
    assert gaussian_rgb_residual_multiplier_for_iteration(config, 25_000) == 1.0

    disabled = replace(config, gaussian_rgb_residual_support=False)
    assert gaussian_rgb_residual_multiplier_for_iteration(disabled, 20_000) == 0.0


def test_r053_shape_and_appearance_handoffs_are_synchronized() -> None:
    config = replace(
        make_config(),
        guide_residual_unlock_end=20_000,
        shape_detail_freeze_until=10_000,
        shape_curl_scale=1.0,
        shape_frizz_scale=1.0,
    )
    for iteration, expected in (
        (9_999, 0.0),
        (10_000, 0.0),
        (15_000, 0.5),
        (20_000, 1.0),
        (25_000, 1.0),
    ):
        assert shape_detail_multiplier_for_iteration(config, iteration) == expected
        assert gaussian_rgb_residual_multiplier_for_iteration(config, iteration) == expected


def test_shape_gate_is_zero_before_handoff_and_joint_controls_receive_gradients() -> None:
    model = make_model()
    config = replace(
        make_config(),
        shape_curl_scale=1.0,
        shape_frizz_scale=1.0,
        guide_curl_residual_scale=1.0,
        guide_frizz_residual_scale=1.0,
    )
    names = {
        name
        for group in stage1_optimizer_param_names(model, config)
        for name in group
    }
    assert "guide_curl_radius_raw" in names
    assert "guide_frizz_raw" in names
    assert "render_geometry_residual.curl_radius_raw" in names
    assert "render_geometry_residual.frizz_raw" in names
    assert "gaussian_rgb_residual.raw" in names

    _, _, roots_local = model.roots_and_normals()
    model.shape_detail_multiplier = 0.0
    frozen = model.apply_guide_controls(model.groom.decode(), roots_local)
    torch.testing.assert_close(frozen.curl_radius, torch.zeros_like(frozen.curl_radius))
    torch.testing.assert_close(frozen.frizz, torch.zeros_like(frozen.frizz))

    model.shape_detail_multiplier = 0.5
    active = model.apply_guide_controls(model.groom.decode(), roots_local)
    (active.curl_radius.mean() + active.frizz.mean()).backward()
    assert model.guide_curl_radius_raw.grad is not None
    assert model.guide_frizz_raw.grad is not None
    assert model.render_geometry_residual.curl_radius_raw.grad is not None
    assert model.render_geometry_residual.frizz_raw.grad is not None
    for parameter in (
        model.guide_curl_radius_raw,
        model.guide_frizz_raw,
        model.render_geometry_residual.curl_radius_raw,
        model.render_geometry_residual.frizz_raw,
    ):
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool((parameter.grad.abs() > 0.0).any())

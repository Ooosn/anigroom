import numpy as np
import pytest
import torch

from anigroom.grooming import (
    GroomRanges,
    RenderGeometryResidualField,
    apply_asinh_log_ratio_residual,
    apply_log_ratio_residual,
    direction_to_local_components,
    encode_positive_asinh,
    encode_positive_softplus,
    fourth_moment_norm,
    population_stable_residual_norm,
    length_residual_prior_coordinate,
    local_components_to_world,
    tail_concentration_residual_loss,
)
from anigroom.mesh_roots import TriangleMesh
from anigroom.roots.lifecycle import RootStructureUpdate
from tools.train_white_tiger_stage1 import (
    Stage1Config,
    WhiteTigerStage1Model,
    aggregate_render_need_to_guides,
    make_stage1_optimizer,
    optimizer_row_transition,
    rebuild_stage1_optimizer_with_state,
    render_geometry_residual_graph_smoothness,
    select_surface_graph_local_maxima,
    stage1_optimizer_param_names,
    zero_render_geometry_residual_gradients,
)


def make_model(
    guide_face_ids: np.ndarray | None = None,
    guide_barycentric: np.ndarray | None = None,
    render_geometry_parameterization: str = "zero_centered_residual",
    guide_length_residual_scale: float = 0.18,
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
    face_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    barycentric = np.asarray(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10],
        ],
        dtype=np.float32,
    )
    if guide_face_ids is None:
        guide_face_ids = face_ids
    if guide_barycentric is None:
        guide_barycentric = barycentric
    return WhiteTigerStage1Model(
        mesh,
        face_normals,
        face_tangents,
        face_ids,
        barycentric,
        GroomRanges(),
        torch.device("cpu"),
        init_groom_length=0.018,
        guide_face_ids=guide_face_ids,
        guide_barycentric=guide_barycentric,
        render_geometry_parameterization=render_geometry_parameterization,
        guide_length_residual_scale=guide_length_residual_scale,
        guide_direction_residual_scale=0.10,
        guide_clump_residual_scale=0.04,
    )


def effective_groom(model: WhiteTigerStage1Model):
    _, _, roots_local = model.roots_and_normals()
    return model.apply_guide_controls(model.groom.decode(), roots_local)


def test_local_direction_round_trip() -> None:
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    tangents = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bitangents = torch.cross(normals, tangents, dim=-1)
    direction = torch.nn.functional.normalize(
        torch.tensor([[0.4, 0.5, 0.7], [0.4, 0.7, 0.5]]), dim=-1
    )
    local = direction_to_local_components(direction, normals, tangents, bitangents)
    decoded = local_components_to_world(
        local, normals, tangents, bitangents, normalize=True
    )
    torch.testing.assert_close(decoded, direction, atol=1.0e-6, rtol=1.0e-6)


def test_zero_residual_ignores_render_endpoint_geometry() -> None:
    model = make_model()
    model.guide_residual_multiplier = 0.0
    before = effective_groom(model)
    with torch.no_grad():
        model.groom.length_raw.fill_(20.0)
        model.groom.root_width_raw.fill_(20.0)
        model.groom.tip_width_ratio_raw.fill_(20.0)
        model.groom.width_taper_raw.fill_(20.0)
        model.groom.brush_stiffness_raw.fill_(20.0)
        model.groom.direction_local_raw.copy_(
            torch.tensor([[-1.0, 0.0, 0.0]]).expand_as(
                model.groom.direction_local_raw
            )
        )
        model.groom.curl_radius_ratio_raw.fill_(20.0)
        model.groom.frizz_amplitude_ratio_raw.fill_(20.0)
        model.groom.clump_strength_raw.fill_(20.0)
    after = effective_groom(model)
    for name in (
        "length",
        "root_width",
        "tip_width",
        "width_taper",
        "brush_stiffness",
        "direction_local",
        "curl_radius_ratio",
        "frizz_amplitude_ratio",
        "clump_strength",
    ):
        torch.testing.assert_close(getattr(after, name), getattr(before, name))


def test_unlocked_residual_changes_effective_geometry_from_zero() -> None:
    model = make_model()
    model.guide_residual_multiplier = 1.0
    baseline = effective_groom(model)
    with torch.no_grad():
        model.render_geometry_residual.length_raw.fill_(0.35)
        model.render_geometry_residual.root_width_raw.fill_(0.35)
        model.render_geometry_residual.tip_width_ratio_raw.fill_(0.35)
        model.render_geometry_residual.width_taper_raw.fill_(0.35)
        model.render_geometry_residual.child_radius_raw.fill_(0.35)
        model.render_geometry_residual.direction_local_raw[:, 2].fill_(0.25)
    changed = effective_groom(model)
    assert torch.all(changed.length > baseline.length)
    assert torch.all(changed.root_width > baseline.root_width)
    assert torch.all(changed.tip_width / changed.root_width > baseline.tip_width / baseline.root_width)
    assert torch.all(changed.width_taper > baseline.width_taper)
    assert torch.all(changed.child_radius > baseline.child_radius)
    assert not torch.allclose(changed.direction_local, baseline.direction_local)


def test_child_spread_uses_coverage_multiplier() -> None:
    model = make_model()
    model.guide_residual_multiplier = 0.0
    model.guide_coverage_residual_multiplier = 1.0
    baseline = effective_groom(model)
    with torch.no_grad():
        model.render_geometry_residual.length_raw.fill_(0.35)
        model.render_geometry_residual.root_width_raw.fill_(0.35)
        model.render_geometry_residual.tip_width_ratio_raw.fill_(0.35)
        model.render_geometry_residual.width_taper_raw.fill_(0.35)
        model.render_geometry_residual.child_radius_raw.fill_(0.35)
        model.render_geometry_residual.direction_local_raw[:, 2].fill_(0.25)
    changed = effective_groom(model)

    torch.testing.assert_close(changed.length, baseline.length)
    torch.testing.assert_close(changed.direction_local, baseline.direction_local)
    torch.testing.assert_close(changed.root_width, baseline.root_width)
    torch.testing.assert_close(changed.tip_width, baseline.tip_width)
    torch.testing.assert_close(changed.width_taper, baseline.width_taper)
    assert torch.all(changed.child_radius > baseline.child_radius)


def test_hierarchical_width_profile_backpropagates_to_guide_and_residual_only() -> None:
    model = make_model()
    model.guide_residual_multiplier = 1.0
    groom = effective_groom(model)
    loss = (
        groom.root_width.log().mean()
        + (groom.tip_width / groom.root_width).mean()
        + groom.width_taper.log().mean()
        + groom.child_radius.log().mean()
    )

    loss.backward()

    for parameter in (
        model.guide_root_width_raw,
        model.guide_tip_width_ratio_raw,
        model.guide_width_taper_raw,
        model.guide_child_radius_raw,
        model.render_geometry_residual.root_width_raw,
        model.render_geometry_residual.tip_width_ratio_raw,
        model.render_geometry_residual.width_taper_raw,
        model.render_geometry_residual.child_radius_raw,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
    assert model.groom.root_width_raw.grad is None
    assert model.groom.tip_width_ratio_raw.grad is None
    assert model.groom.width_taper_raw.grad is None
    assert model.groom.child_radius_raw.grad is None


def test_brush_stiffness_is_guide_owned_and_differentiable() -> None:
    model = make_model()
    groom = effective_groom(model)
    groom.brush_stiffness.mean().backward()

    assert model.guide_brush_stiffness_raw.grad is not None
    assert bool(torch.isfinite(model.guide_brush_stiffness_raw.grad).all())
    assert bool((model.guide_brush_stiffness_raw.grad.abs() > 0.0).any())
    assert model.groom.brush_stiffness_raw.grad is None


def test_optimizer_contains_residuals_not_legacy_geometry() -> None:
    model = make_model()
    config = Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
        guide_clump_residual_scale=0.04,
    )
    optimizer = make_stage1_optimizer(model, config)
    names = {
        name
        for group in stage1_optimizer_param_names(model, config)
        for name in group
    }
    optimized_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    assert "render_geometry_residual.length_raw" in names
    assert "render_geometry_residual.direction_local_raw" in names
    assert "render_geometry_residual.root_width_raw" in names
    assert "render_geometry_residual.tip_width_ratio_raw" in names
    assert "render_geometry_residual.width_taper_raw" in names
    assert "render_geometry_residual.child_radius_raw" in names
    assert "guide_brush_stiffness_raw" in names
    assert "groom.length_raw" not in names
    assert "groom.root_width_raw" not in names
    assert "groom.tip_width_ratio_raw" not in names
    assert "groom.width_taper_raw" not in names
    assert "groom.child_radius_raw" not in names
    assert "groom.brush_stiffness_raw" not in names
    assert "groom.direction_local_raw" not in names
    assert id(model.render_geometry_residual.length_raw) in optimized_ids
    assert id(model.render_geometry_residual.root_width_raw) in optimized_ids
    assert id(model.render_geometry_residual.tip_width_ratio_raw) in optimized_ids
    assert id(model.render_geometry_residual.width_taper_raw) in optimized_ids
    assert id(model.render_geometry_residual.child_radius_raw) in optimized_ids
    assert id(model.guide_brush_stiffness_raw) in optimized_ids
    assert id(model.groom.length_raw) not in optimized_ids
    assert id(model.groom.direction_local_raw) not in optimized_ids


def test_disabled_curl_and_frizz_are_not_optimized() -> None:
    model = make_model()
    config = Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        shape_curl_scale=0.0,
        shape_frizz_scale=0.0,
    )
    names = {
        name
        for group in stage1_optimizer_param_names(model, config)
        for name in group
    }

    assert "guide_curl_radius_ratio_raw" not in names
    assert "guide_frizz_amplitude_ratio_raw" not in names


def test_late_geometry_freeze_preserves_early_child_spread_gradient() -> None:
    model = make_model()
    for parameter in model.render_geometry_residual.parameters():
        parameter.grad = torch.ones_like(parameter)

    zero_render_geometry_residual_gradients(model)

    for name, parameter in model.render_geometry_residual.named_parameters():
        assert parameter.grad is not None
        if name == "child_radius_raw":
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        else:
            torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))


def test_structure_update_transports_residual_state_and_strict_checkpoint() -> None:
    model = make_model()
    with torch.no_grad():
        for parameter in model.render_geometry_residual.parameters():
            parameter.fill_(0.2)
        model.groom.length_reference.copy_(
            torch.tensor([[0.012], [0.020], [0.031], [0.046]])
        )
        model.groom.tip_width_ratio_raw.fill_(torch.logit(torch.tensor(0.95)))
        model.groom.width_taper_raw.fill_(encode_positive_asinh(torch.tensor(8.0)))
        model.groom.curl_radius_ratio_raw.fill_(
            encode_positive_softplus(torch.tensor(0.17))
        )
        model.groom.frizz_amplitude_ratio_raw.fill_(
            encode_positive_softplus(torch.tensor(0.06))
        )
        model.groom.frizz_seed_phase.fill_(1.234)
    update = RootStructureUpdate(
        parent_indices=torch.tensor([0], dtype=torch.long),
        child_parent_indices=torch.tensor([0], dtype=torch.long),
        new_face_ids=torch.tensor([0], dtype=torch.long),
        new_barycentric=torch.tensor([[0.55, 0.25, 0.20]], dtype=torch.float32),
        prune_mask=torch.tensor([True, False, False, False]),
        scores={},
    )
    result = model.apply_structure_update(update, neighbor_count=8)
    assert result["root_count_after"] == 4
    assert model.render_geometry_residual.root_count == 4
    assert model.groom.length_reference.shape == (4, 1)
    assert bool((model.groom.length_reference > 0.0).all())
    torch.testing.assert_close(
        model.groom.frizz_seed_phase,
        torch.full_like(model.groom.frizz_seed_phase, 1.234),
    )
    decoded = model.groom.decode()
    torch.testing.assert_close(
        decoded.tip_width / decoded.root_width,
        torch.full_like(decoded.root_width, 0.95),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        decoded.width_taper,
        torch.full_like(decoded.width_taper, 8.0),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        decoded.curl_radius_ratio,
        torch.full_like(decoded.curl_radius_ratio, 0.17),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        decoded.frizz_amplitude_ratio,
        torch.full_like(decoded.frizz_amplitude_ratio, 0.06),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    for parameter in model.render_geometry_residual.parameters():
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, 0.2),
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    clone = make_model()
    clone.load_state_dict(model.state_dict(), strict=True)
    clone_parameters = dict(clone.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, clone_parameters[name])
    torch.testing.assert_close(
        clone.groom.length_reference,
        model.groom.length_reference,
    )
    torch.testing.assert_close(
        clone.groom.frizz_seed_phase,
        model.groom.frizz_seed_phase,
    )


def test_lifecycle_rebuild_preserves_surviving_adam_rows() -> None:
    model = make_model()
    config = Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
        guide_clump_residual_scale=0.04,
    )
    optimizer = make_stage1_optimizer(model, config)
    optimizer_names = stage1_optimizer_param_names(model, config)

    old_bary = model.bary_logits
    old_bary_moment = torch.arange(old_bary.numel(), dtype=old_bary.dtype).reshape_as(old_bary)
    optimizer.state[old_bary] = {
        "step": torch.tensor(17.0),
        "exp_avg": old_bary_moment.clone(),
        "exp_avg_sq": old_bary_moment.square(),
    }
    old_guide = model.guide_length_raw
    old_guide_moment = torch.arange(old_guide.numel(), dtype=old_guide.dtype).reshape_as(old_guide) + 20.0
    optimizer.state[old_guide] = {
        "step": torch.tensor(11.0),
        "exp_avg": old_guide_moment.clone(),
        "exp_avg_sq": old_guide_moment.square(),
    }

    guide_update = RootStructureUpdate(
        parent_indices=torch.tensor([0], dtype=torch.long),
        child_parent_indices=torch.tensor([0], dtype=torch.long),
        new_face_ids=torch.tensor([0], dtype=torch.long),
        new_barycentric=torch.tensor([[0.50, 0.30, 0.20]], dtype=torch.float32),
        prune_mask=torch.zeros(4, dtype=torch.bool),
        scores={},
    )
    render_update = RootStructureUpdate(
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
    guide_transition = optimizer_row_transition(guide_update, old_count=4)
    render_transition = optimizer_row_transition(render_update, old_count=4)

    model.apply_guide_structure_update(guide_update)
    model.apply_structure_update(render_update, neighbor_count=8)
    rebuilt, report = rebuild_stage1_optimizer_with_state(
        model,
        config,
        optimizer,
        optimizer_names,
        render_transition=render_transition,
        guide_transition=guide_transition,
    )

    bary_state = rebuilt.state[model.bary_logits]
    torch.testing.assert_close(bary_state["exp_avg"][:3], old_bary_moment[1:])
    torch.testing.assert_close(bary_state["exp_avg"][3:], torch.zeros_like(bary_state["exp_avg"][3:]))
    torch.testing.assert_close(bary_state["exp_avg_sq"][:3], old_bary_moment[1:].square())
    assert float(bary_state["step"]) == 17.0

    guide_state = rebuilt.state[model.guide_length_raw]
    torch.testing.assert_close(guide_state["exp_avg"][:4], old_guide_moment)
    torch.testing.assert_close(guide_state["exp_avg"][4:], torch.zeros_like(guide_state["exp_avg"][4:]))
    torch.testing.assert_close(guide_state["exp_avg_sq"][:4], old_guide_moment.square())
    assert float(guide_state["step"]) == 11.0

    assert report["render"] == {
        "old_root_count": 4,
        "retained_root_count": 3,
        "zero_initialized_child_count": 2,
        "new_root_count": 5,
    }
    assert report["guide"] == {
        "old_root_count": 4,
        "retained_root_count": 4,
        "zero_initialized_child_count": 1,
        "new_root_count": 5,
    }
    assert report["row_migrated_parameter_count"] >= 2


def test_guide_densification_updates_direct_direction_and_loads_strict_state() -> None:
    model = make_model()
    with torch.no_grad():
        model.guide_length_reference.fill_(0.031)
        model.guide_length_raw.copy_(
            torch.tensor([[0.0], [0.25], [0.50], [0.75]])
        )
        model.guide_root_width_reference.fill_(0.0002)
        model.guide_root_width_raw.fill_(0.3)
        model.guide_tip_width_ratio_reference.fill_(0.07)
        model.guide_tip_width_ratio_raw.fill_(0.2)
        model.guide_width_taper_reference.fill_(1.8)
        model.guide_width_taper_raw.fill_(0.4)
        model.guide_brush_stiffness_raw.fill_(0.3)
        model.guide_child_radius_reference.fill_(0.0028)
        model.guide_child_radius_raw.fill_(0.3)
    guide_before, _ = model.interpolate_guide_controls(
        model.roots_and_normals()[2],
        model.roots_and_normals()[1],
        *model.tangent_frames(model.roots_and_normals()[1]),
    )
    update = RootStructureUpdate(
        parent_indices=torch.tensor([0], dtype=torch.long),
        child_parent_indices=torch.tensor([0], dtype=torch.long),
        new_face_ids=torch.tensor([0], dtype=torch.long),
        new_barycentric=torch.tensor([[0.55, 0.25, 0.20]], dtype=torch.float32),
        prune_mask=torch.zeros((4,), dtype=torch.bool),
        scores={},
    )
    result = model.apply_guide_structure_update(update)
    assert result["guide_root_count_after"] == 5
    assert model.guide_direction_local_raw.shape == (5, 3)
    assert model.guide_length_reference.shape == (5, 1)
    assert model.guide_root_width_reference.shape == (5, 1)
    assert model.guide_tip_width_ratio_reference.shape == (5, 1)
    assert model.guide_width_taper_reference.shape == (5, 1)
    assert model.guide_brush_stiffness_raw.shape == (5, 1)
    assert model.guide_child_radius_reference.shape == (5, 1)
    torch.testing.assert_close(
        model.guide_length_reference[-1],
        torch.tensor([0.031]),
    )
    assert bool(
        torch.isfinite(
            model.guide_length_raw[-1]
        ).all()
    )
    guide_normals, guide_tangents, guide_bitangents = (
        model.guide_normals_and_tangent_frames()
    )
    guide_after = model.sample_guide_controls(
        model.guide_points_local[-1:],
        model.guide_face_ids[-1:],
        guide_normals[-1:],
        guide_tangents[-1:],
        guide_bitangents[-1:],
    )[0]
    for name in (
        "root_width",
        "tip_width_ratio",
        "width_taper",
        "brush_stiffness",
        "curl_radius_ratio",
        "frizz_amplitude_ratio",
        "child_radius",
    ):
        torch.testing.assert_close(
            guide_after[name],
            guide_before[name][0:1],
            atol=1.0e-5,
            rtol=1.0e-5,
        )

    clone = make_model(
        guide_face_ids=model.guide_face_ids.detach().cpu().numpy(),
        guide_barycentric=model.guide_barycentric.detach().cpu().numpy(),
    )
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        clone.guide_direction_local_raw,
        model.guide_direction_local_raw,
    )
    torch.testing.assert_close(
        clone.guide_brush_stiffness_raw,
        model.guide_brush_stiffness_raw,
    )
    torch.testing.assert_close(
        clone.guide_length_reference,
        model.guide_length_reference,
    )
    torch.testing.assert_close(
        clone.guide_root_width_reference,
        model.guide_root_width_reference,
    )
    torch.testing.assert_close(
        clone.guide_child_radius_reference,
        model.guide_child_radius_reference,
    )


def test_residual_field_starts_exactly_zero() -> None:
    field = RenderGeometryResidualField(7, device="cpu")
    for value in field.decode().__dict__.values():
        torch.testing.assert_close(value, torch.zeros_like(value))


def test_fourth_moment_norm_keeps_sparse_outlier_visible_without_a_threshold() -> None:
    sparse = torch.zeros((1000, 1), requires_grad=True)
    with torch.no_grad():
        sparse[0] = 4.0

    loss = fourth_moment_norm(sparse)
    mean_l1 = sparse.abs().mean()

    assert float(loss) > 100.0 * float(mean_l1)
    loss.backward()
    assert sparse.grad is not None
    assert torch.isfinite(sparse.grad).all()
    assert float(sparse.grad[0]) > 0.0


def test_population_stable_residual_norm_reuses_unlock_multiplier() -> None:
    value = torch.tensor([0.0, 0.1, -0.2, 1.5], requires_grad=True)
    mean_l1 = value.abs().mean()
    stable = fourth_moment_norm(value)

    torch.testing.assert_close(
        population_stable_residual_norm(value, 0.0),
        mean_l1,
    )
    torch.testing.assert_close(
        population_stable_residual_norm(value, 1.0),
        stable,
    )
    midpoint = population_stable_residual_norm(value, 0.5)
    torch.testing.assert_close(midpoint, torch.lerp(mean_l1, stable, 0.5))
    midpoint.backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_tail_concentration_loss_preserves_coherent_residuals_and_penalizes_sparse_tail() -> None:
    coherent = torch.full((16,), 0.75, requires_grad=True)
    coherent_loss = tail_concentration_residual_loss(coherent, 1.0)
    torch.testing.assert_close(
        coherent_loss,
        coherent.abs().mean(),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    sparse = torch.zeros((16,), requires_grad=True)
    with torch.no_grad():
        sparse[0] = 4.0
    baseline = sparse.abs().mean()
    concentrated = tail_concentration_residual_loss(sparse, 1.0)
    assert concentrated > baseline
    assert concentrated < fourth_moment_norm(sparse)
    torch.testing.assert_close(
        tail_concentration_residual_loss(sparse, 0.0),
        baseline,
    )
    concentrated.backward()
    assert sparse.grad is not None
    assert torch.isfinite(sparse.grad).all()


def test_tail_concentration_gradient_is_not_mean_population_diluted() -> None:
    def outlier_gradient(root_count: int, *, concentrated: bool) -> float:
        value = torch.zeros((root_count,), requires_grad=True)
        with torch.no_grad():
            value[0] = 4.0
        loss = (
            tail_concentration_residual_loss(value, 1.0)
            if concentrated
            else value.abs().mean()
        )
        loss.backward()
        assert value.grad is not None
        return float(value.grad[0])

    small_mean = outlier_gradient(64, concentrated=False)
    large_mean = outlier_gradient(4096, concentrated=False)
    small_tail = outlier_gradient(64, concentrated=True)
    large_tail = outlier_gradient(4096, concentrated=True)

    torch.testing.assert_close(
        torch.tensor(large_mean / small_mean),
        torch.tensor(64.0 / 4096.0),
    )
    assert large_tail > 100.0 * large_mean
    assert large_tail > 0.4 * small_tail


def test_guide_densification_propagates_length_target_conditioned_on_confidence() -> None:
    model = make_model()
    with torch.no_grad():
        model.guide_clean_flow_length_target.copy_(
            torch.tensor([0.021, 0.0, 0.0, 0.0])
        )
        model.guide_clean_flow_length_confidence.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0])
        )
    update = RootStructureUpdate(
        parent_indices=torch.tensor([0], dtype=torch.long),
        child_parent_indices=torch.tensor([0], dtype=torch.long),
        new_face_ids=torch.tensor([0], dtype=torch.long),
        new_barycentric=torch.tensor([[0.55, 0.25, 0.20]], dtype=torch.float32),
        prune_mask=torch.zeros((4,), dtype=torch.bool),
        scores={},
    )

    model.apply_guide_structure_update(update)

    assert float(model.guide_clean_flow_length_confidence[-1]) > 0.0
    torch.testing.assert_close(
        model.guide_clean_flow_length_target[-1],
        torch.tensor(0.021),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_guide_densification_surface_attribution_reuses_forward_support() -> None:
    model = make_model()
    _, _, roots = model.roots_and_normals()
    support, weights = model.guide_interpolation_attribution(roots)
    need = torch.tensor([1.0, 0.5, 0.0, 0.25])
    visible = torch.tensor([True, True, False, True])

    score, weight_sum, report = aggregate_render_need_to_guides(
        model,
        need,
        visible,
        policy="surface_attribution_local_max",
        legacy_neighbor_count=1,
    )

    expected_sum = torch.zeros_like(score)
    expected_weight = torch.zeros_like(weight_sum)
    active_weight = weights * visible[:, None]
    expected_sum.scatter_add_(
        0,
        support.indices.reshape(-1),
        (need[:, None] * active_weight).reshape(-1),
    )
    expected_weight.scatter_add_(0, support.indices.reshape(-1), active_weight.reshape(-1))
    torch.testing.assert_close(weight_sum, expected_weight)
    torch.testing.assert_close(score, expected_sum / expected_weight.clamp_min(1.0e-8))
    assert report == {
        "evidence_support": "forward_surface_interpolation",
        "render_root_k": int(support.indices.shape[1]),
    }


def test_guide_densification_legacy_attribution_keeps_euclidean_weighting() -> None:
    model = make_model()
    _, _, roots = model.roots_and_normals()
    need = torch.tensor([1.0, 0.5, 0.0, 0.25])
    visible = torch.tensor([True, True, False, True])

    score, weight_sum, report = aggregate_render_need_to_guides(
        model,
        need,
        visible,
        policy="global_score_budget",
        legacy_neighbor_count=2,
    )

    distance = torch.cdist(roots, model.guide_points_local)
    values, ids = torch.topk(distance, k=2, largest=False, dim=-1)
    weights = values.clamp_min(1.0e-6).pow(-2.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    weights = weights * visible[:, None]
    expected_sum = torch.zeros_like(score)
    expected_weight = torch.zeros_like(weight_sum)
    expected_sum.scatter_add_(0, ids.reshape(-1), (need[:, None] * weights).reshape(-1))
    expected_weight.scatter_add_(0, ids.reshape(-1), weights.reshape(-1))

    torch.testing.assert_close(weight_sum, expected_weight)
    torch.testing.assert_close(score, expected_sum / expected_weight.clamp_min(1.0e-8))
    assert report == {"evidence_support": "euclidean_knn", "render_root_k": 2}


def test_guide_densification_local_maxima_do_not_create_global_region_competition() -> None:
    score = torch.tensor([9.0, 8.0, 7.0, 3.0, 2.0, 1.0])
    valid = torch.ones((6,), dtype=torch.bool)
    edges = torch.tensor(
        [
            [0, 1],
            [1, 0],
            [1, 2],
            [2, 1],
            [3, 4],
            [4, 3],
            [4, 5],
            [5, 4],
        ],
        dtype=torch.long,
    )

    selected = select_surface_graph_local_maxima(score, valid, edges)

    torch.testing.assert_close(selected, torch.tensor([0, 3]))


def test_log_ratio_residual_is_zero_centered_positive_and_scale_equivariant() -> None:
    guide = torch.tensor([[0.012], [0.037], [1.5]], requires_grad=True)
    delta = torch.tensor([[-0.8], [0.0], [0.6]], requires_grad=True)
    scale = 0.35

    effective = apply_log_ratio_residual(guide, delta, scale)
    doubled = apply_log_ratio_residual(guide * 2.0, delta, scale)

    assert torch.all(effective > 0.0)
    torch.testing.assert_close(doubled, effective * 2.0)
    torch.testing.assert_close(
        effective / guide,
        torch.exp(delta * scale),
    )
    effective.sum().backward()
    assert torch.all(torch.isfinite(guide.grad))
    assert torch.all(torch.isfinite(delta.grad))
    assert torch.all(delta.grad != 0.0)


def test_log_ratio_model_zero_residual_exactly_follows_guide_without_range_clamp() -> None:
    model = make_model(
        render_geometry_parameterization="zero_centered_log_length_residual"
    )
    model.guide_residual_multiplier = 1.0
    baseline = effective_groom(model)
    _, normals, roots_local = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    guide, _ = model.interpolate_guide_controls(
        roots_local,
        normals,
        tangents,
        bitangents,
    )
    torch.testing.assert_close(baseline.length, guide["length"])

    with torch.no_grad():
        model.guide_length_raw.fill_(20.0)
        model.render_geometry_residual.length_raw.fill_(20.0)
    changed = effective_groom(model)
    assert float(changed.length.max()) > 10.0 * float(model.guide_length_reference.max())


def test_log_ratio_model_uses_equal_ratios_for_different_guide_lengths() -> None:
    model = make_model(
        render_geometry_parameterization="zero_centered_log_length_residual"
    )
    model.guide_residual_multiplier = 1.0
    with torch.no_grad():
        model.guide_length_raw.copy_(
            torch.tensor([[-4.0], [-1.0], [1.0], [4.0]])
        )
        model.render_geometry_residual.length_raw.fill_(0.7)
    _, normals, roots_local = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    guide, _ = model.interpolate_guide_controls(
        roots_local,
        normals,
        tangents,
        bitangents,
    )
    changed = effective_groom(model)
    ratio = changed.length / guide["length"]
    torch.testing.assert_close(ratio, ratio[:1].expand_as(ratio))


def test_raw_log_ratio_coordinate_is_positive_zero_centered_and_unbounded() -> None:
    guide = torch.tensor([[0.012], [0.037], [1.5]], requires_grad=True)
    delta = torch.tensor([[-3.0], [0.0], [2.0]], requires_grad=True)
    effective = apply_log_ratio_residual(guide, delta, 1.0)

    assert torch.all(effective > 0.0)
    torch.testing.assert_close(effective[1], guide[1])
    assert float(effective[2] / guide[2]) > 7.0
    effective.sum().backward()
    assert torch.all(torch.isfinite(guide.grad))
    assert torch.all(torch.isfinite(delta.grad))
    assert torch.all(delta.grad != 0.0)


def test_asinh_log_ratio_is_local_identity_positive_and_unbounded() -> None:
    guide = torch.tensor([[0.012], [0.037], [1.5]], requires_grad=True)
    delta = torch.tensor([[-100.0], [0.0], [100.0]], requires_grad=True)

    effective = apply_asinh_log_ratio_residual(guide, delta, 1.0)
    doubled = apply_asinh_log_ratio_residual(guide * 2.0, delta, 1.0)

    assert torch.all(effective > 0.0)
    torch.testing.assert_close(effective[1], guide[1])
    torch.testing.assert_close(doubled, effective * 2.0)
    torch.testing.assert_close(
        effective[0] / guide[0],
        guide[2] / effective[2],
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    assert float(effective[2] / guide[2]) > 100.0
    assert float(effective[2] / guide[2]) < 1000.0
    effective.sum().backward()
    assert torch.all(torch.isfinite(guide.grad))
    assert torch.all(torch.isfinite(delta.grad))
    assert torch.all(delta.grad != 0.0)


def test_asinh_log_ratio_matches_raw_log_ratio_slope_at_zero() -> None:
    guide = torch.tensor([[0.031]])
    raw_delta = torch.zeros((1, 1), requires_grad=True)
    robust_delta = torch.zeros((1, 1), requires_grad=True)
    scale = 0.6

    apply_log_ratio_residual(guide, raw_delta, scale).sum().backward()
    apply_asinh_log_ratio_residual(guide, robust_delta, scale).sum().backward()

    torch.testing.assert_close(raw_delta.grad, robust_delta.grad)


def test_natural_log_ratio_prior_keeps_tail_gradient_without_a_bound() -> None:
    legacy_raw = torch.tensor([[0.0], [2.0], [8.0]], requires_grad=True)
    natural_raw = legacy_raw.detach().clone().requires_grad_(True)

    legacy = length_residual_prior_coordinate(
        legacy_raw,
        "zero_centered_asinh_log_length_residual",
        "decoded",
    )
    natural = length_residual_prior_coordinate(
        natural_raw,
        "zero_centered_asinh_log_length_residual",
        "natural_log_ratio",
    )

    torch.testing.assert_close(legacy, torch.tanh(legacy_raw))
    torch.testing.assert_close(natural, torch.asinh(natural_raw))
    legacy.sum().backward()
    natural.sum().backward()
    assert legacy_raw.grad is not None
    assert natural_raw.grad is not None
    assert float(natural_raw.grad[-1]) > 0.1
    assert float(natural_raw.grad[-1]) > 1000.0 * float(legacy_raw.grad[-1])


def test_natural_log_ratio_prior_matches_each_log_decoder_coordinate() -> None:
    raw = torch.tensor([[-2.0], [0.0], [3.0]])
    torch.testing.assert_close(
        length_residual_prior_coordinate(
            raw,
            "zero_centered_unbounded_log_length_residual",
            "natural_log_ratio",
        ),
        raw,
    )
    torch.testing.assert_close(
        length_residual_prior_coordinate(
            raw,
            "zero_centered_log_length_residual",
            "natural_log_ratio",
        ),
        torch.tanh(raw),
    )


def test_raw_length_prior_has_non_vanishing_asinh_tail_gradient() -> None:
    natural_raw = torch.tensor([[8.0]], requires_grad=True)
    direct_raw = natural_raw.detach().clone().requires_grad_(True)

    natural = length_residual_prior_coordinate(
        natural_raw,
        "zero_centered_asinh_log_length_residual",
        "natural_log_ratio",
    )
    direct = length_residual_prior_coordinate(
        direct_raw,
        "zero_centered_asinh_log_length_residual",
        "raw",
    )

    natural.abs().sum().backward()
    direct.abs().sum().backward()
    torch.testing.assert_close(direct, direct_raw)
    torch.testing.assert_close(direct_raw.grad, torch.ones_like(direct_raw))
    assert natural_raw.grad is not None
    assert float(direct_raw.grad[-1]) > 8.0 * float(natural_raw.grad[-1])


def test_raw_length_prior_rejects_non_residual_parameterization() -> None:
    with pytest.raises(ValueError, match="zero-centered length residual"):
        length_residual_prior_coordinate(
            torch.zeros((1, 1)),
            "absolute_endpoint",
            "raw",
        )


def test_unbounded_log_ratio_model_uses_raw_coordinate_and_equal_ratios() -> None:
    model = make_model(
        render_geometry_parameterization="zero_centered_unbounded_log_length_residual",
        guide_length_residual_scale=1.0,
    )
    model.guide_residual_multiplier = 1.0
    with torch.no_grad():
        model.guide_length_raw.copy_(
            torch.tensor([[-4.0], [-1.0], [1.0], [4.0]])
        )
        model.render_geometry_residual.length_raw.fill_(2.0)
    _, normals, roots_local = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    guide, _ = model.interpolate_guide_controls(
        roots_local,
        normals,
        tangents,
        bitangents,
    )
    changed = effective_groom(model)
    ratio = changed.length / guide["length"]

    torch.testing.assert_close(ratio, ratio[:1].expand_as(ratio))
    torch.testing.assert_close(
        ratio,
        torch.full_like(ratio, torch.exp(torch.tensor(2.0))),
    )
    assert float(changed.length.max()) > 5.0 * float(model.guide_length_reference.max())


def test_unbounded_log_ratio_graph_smoothness_penalizes_raw_tail() -> None:
    model = make_model(
        render_geometry_parameterization="zero_centered_unbounded_log_length_residual",
        guide_length_residual_scale=1.0,
    )
    edges = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.long)
    _, normals, _ = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)

    with torch.no_grad():
        model.render_geometry_residual.length_raw.zero_()
        model.render_geometry_residual.length_raw[0] = 8.0

    loss = render_geometry_residual_graph_smoothness(
        model,
        edges,
        normals,
        tangents,
        bitangents,
    )
    loss.backward()

    assert float(loss) > 1.0
    gradient = model.render_geometry_residual.length_raw.grad
    assert gradient is not None
    assert float(gradient[0]) > 0.0
    assert float(gradient[1]) < 0.0


def test_asinh_log_ratio_model_is_scale_relative_and_has_no_absolute_bound() -> None:
    model = make_model(
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        guide_length_residual_scale=1.0,
    )
    model.guide_residual_multiplier = 1.0
    with torch.no_grad():
        model.guide_length_raw.copy_(
            torch.tensor([[-4.0], [-1.0], [1.0], [4.0]])
        )
        model.render_geometry_residual.length_raw.fill_(100.0)
    _, normals, roots_local = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    guide, _ = model.interpolate_guide_controls(
        roots_local,
        normals,
        tangents,
        bitangents,
    )
    changed = effective_groom(model)
    ratio = changed.length / guide["length"]

    torch.testing.assert_close(ratio, ratio[:1].expand_as(ratio))
    assert float(ratio[0]) > 100.0
    assert float(changed.length.max()) > 10.0 * float(model.guide_length_reference.max())


def test_asinh_log_ratio_structure_update_interpolates_raw_coordinate() -> None:
    model = make_model(
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        guide_length_residual_scale=1.0,
    )
    with torch.no_grad():
        model.render_geometry_residual.length_raw.fill_(5.0)
    update = RootStructureUpdate(
        parent_indices=torch.tensor([0], dtype=torch.long),
        child_parent_indices=torch.tensor([0], dtype=torch.long),
        new_face_ids=torch.tensor([0], dtype=torch.long),
        new_barycentric=torch.tensor([[0.55, 0.25, 0.20]], dtype=torch.float32),
        prune_mask=torch.tensor([True, False, False, False]),
        scores={},
    )

    model.apply_structure_update(update, neighbor_count=8)

    torch.testing.assert_close(
        model.render_geometry_residual.length_raw,
        torch.full_like(model.render_geometry_residual.length_raw, 5.0),
        atol=1.0e-5,
        rtol=1.0e-5,
    )

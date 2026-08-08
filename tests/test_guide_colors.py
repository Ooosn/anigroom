import numpy as np
import torch

from anigroom.grooming import GroomRanges, GuideColorField
from anigroom.mesh_roots import TriangleMesh
from anigroom.roots.lifecycle import RootStructureUpdate
from tools.train_white_tiger_stage1 import (
    Stage1Config,
    WhiteTigerStage1Model,
    make_stage1_optimizer,
    optimizer_row_transition,
    rebuild_stage1_optimizer_with_state,
    stage1_optimizer_param_names,
    zero_low_frequency_color_gradients,
)


def make_model(*, gaussian_rgb_residual: bool = True) -> WhiteTigerStage1Model:
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
    face_normals = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    face_tangents = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
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
        guide_face_ids=face_ids,
        guide_barycentric=barycentric,
        guide_color_support=True,
        gaussian_rgb_residual_support=gaussian_rgb_residual,
        render_geometry_parameterization="zero_centered_residual",
    )


def test_guide_color_field_round_trip() -> None:
    field = GuideColorField(2)
    root = torch.tensor([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
    tip = torch.tensor([[0.2, 0.3, 0.4], [0.8, 0.9, 0.95]])
    field.set_decoded(root, tip)

    decoded = field.decode()
    torch.testing.assert_close(decoded.root, root)
    torch.testing.assert_close(decoded.tip, tip)


def test_guide_colors_replace_render_root_color_capacity() -> None:
    model = make_model()
    assert model.guide_colors is not None
    root = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.7, 0.8, 0.9],
            [0.8, 0.9, 0.95],
        ]
    )
    tip = (0.9 * root + 0.1).clamp_max(0.99)
    model.guide_colors.set_decoded(root, tip)
    with torch.no_grad():
        model.groom.root_color_raw.fill_(-20.0)
        model.groom.tip_color_raw.fill_(-20.0)

    roots, normals, roots_local = model.roots_and_normals()
    del roots
    tangents, bitangents = model.tangent_frames(normals)
    effective = model.apply_guide_controls(
        model.groom.decode(), roots_local, normals, tangents, bitangents
    )

    assert float(effective.root_color.min()) >= float(root.min()) - 1.0e-5
    assert float(effective.root_color.max()) <= float(root.max()) + 1.0e-5
    assert float(effective.root_color.mean()) > 0.1


def test_optimizer_uses_guide_colors_and_handoff_freezes_only_base() -> None:
    model = make_model()
    config = Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        child_count=1,
        guide_color_support=True,
        gaussian_rgb_residual_support=True,
    )
    optimizer = make_stage1_optimizer(model, config)
    names = stage1_optimizer_param_names(model, config)
    flat_names = [name for group in names for name in group]
    assert "guide_colors.root_raw" in flat_names
    assert "guide_colors.tip_raw" in flat_names
    assert "groom.root_color_raw" not in flat_names
    assert "groom.tip_color_raw" not in flat_names
    assert "child_color_delta_raw" not in flat_names
    assert "gaussian_rgb_residual.raw" in flat_names

    optimizer.zero_grad(set_to_none=True)
    assert model.guide_colors is not None
    assert model.gaussian_rgb_residual is not None
    loss = (
        model.guide_colors.root_raw.sum()
        + model.guide_colors.tip_raw.sum()
        + model.gaussian_rgb_residual.raw.sum()
    )
    loss.backward()
    zero_low_frequency_color_gradients(model)

    assert model.guide_colors.root_raw.grad is None
    assert model.guide_colors.tip_raw.grad is None
    torch.testing.assert_close(
        model.gaussian_rgb_residual.raw.grad,
        torch.ones_like(model.gaussian_rgb_residual.raw),
    )


def test_color_handoff_blocks_adam_momentum_drift() -> None:
    model = make_model()
    config = Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        child_count=1,
        guide_color_support=True,
        gaussian_rgb_residual_support=True,
    )
    optimizer = make_stage1_optimizer(model, config)
    assert model.guide_colors is not None
    assert model.gaussian_rgb_residual is not None

    # Populate Adam momentum for both color layers as it would exist at handoff.
    optimizer.zero_grad(set_to_none=True)
    warmup_loss = (
        model.guide_colors.root_raw.sum()
        + model.guide_colors.tip_raw.sum()
        + model.gaussian_rgb_residual.raw.sum()
    )
    warmup_loss.backward()
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    handoff_loss = (
        model.guide_colors.root_raw.sum()
        + model.guide_colors.tip_raw.sum()
        + model.gaussian_rgb_residual.raw.sum()
    )
    handoff_loss.backward()
    zero_low_frequency_color_gradients(model)
    root_before = model.guide_colors.root_raw.detach().clone()
    tip_before = model.guide_colors.tip_raw.detach().clone()
    residual_before = model.gaussian_rgb_residual.raw.detach().clone()
    optimizer.step()

    torch.testing.assert_close(model.guide_colors.root_raw, root_before)
    torch.testing.assert_close(model.guide_colors.tip_raw, tip_before)
    assert not torch.equal(model.gaussian_rgb_residual.raw, residual_before)


def test_guide_color_lifecycle_interpolates_children() -> None:
    model = make_model(gaussian_rgb_residual=False)
    assert model.guide_colors is not None
    root = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.6, 0.7, 0.8],
            [0.7, 0.8, 0.9],
        ]
    )
    model.guide_colors.set_decoded(root, root)
    model.guide_color_observation_confidence.copy_(
        torch.tensor([1.0, 0.8, 0.6, 0.4])
    )
    update = RootStructureUpdate(
        parent_indices=torch.tensor([0], dtype=torch.long),
        child_parent_indices=torch.tensor([0], dtype=torch.long),
        new_face_ids=torch.tensor([0], dtype=torch.long),
        new_barycentric=torch.tensor(
            [[0.55, 0.25, 0.20]], dtype=torch.float32
        ),
        prune_mask=torch.zeros((4,), dtype=torch.bool),
        scores={},
    )

    result = model.apply_guide_structure_update(update)

    assert result["guide_root_count_after"] == 5
    assert model.guide_colors.root_count == 5
    child = model.guide_colors.decode().root[-1]
    assert bool(torch.isfinite(child).all())
    assert float(child.min()) >= float(root.min()) - 1.0e-5
    assert float(child.max()) <= float(root.max()) + 1.0e-5
    assert model.guide_color_observation_confidence.shape == (5,)


def test_guide_color_lifecycle_migrates_adam_rows() -> None:
    model = make_model(gaussian_rgb_residual=False)
    config = Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        child_count=1,
        guide_color_support=True,
    )
    optimizer = make_stage1_optimizer(model, config)
    optimizer_names = stage1_optimizer_param_names(model, config)
    assert model.guide_colors is not None
    old_parameter = model.guide_colors.root_raw
    old_moment = torch.arange(
        old_parameter.numel(), dtype=old_parameter.dtype
    ).reshape_as(old_parameter)
    optimizer.state[old_parameter] = {
        "step": torch.tensor(9.0),
        "exp_avg": old_moment.clone(),
        "exp_avg_sq": old_moment.square(),
    }
    update = RootStructureUpdate(
        parent_indices=torch.tensor([0], dtype=torch.long),
        child_parent_indices=torch.tensor([0], dtype=torch.long),
        new_face_ids=torch.tensor([0], dtype=torch.long),
        new_barycentric=torch.tensor(
            [[0.55, 0.25, 0.20]], dtype=torch.float32
        ),
        prune_mask=torch.zeros((4,), dtype=torch.bool),
        scores={},
    )
    transition = optimizer_row_transition(update, old_count=4)

    model.apply_guide_structure_update(update)
    rebuilt, report = rebuild_stage1_optimizer_with_state(
        model,
        config,
        optimizer,
        optimizer_names,
        guide_transition=transition,
    )

    assert model.guide_colors is not None
    state = rebuilt.state[model.guide_colors.root_raw]
    torch.testing.assert_close(state["exp_avg"][:4], old_moment)
    torch.testing.assert_close(
        state["exp_avg"][4:], torch.zeros_like(state["exp_avg"][4:])
    )
    assert float(state["step"]) == 9.0
    assert report["guide"]["new_root_count"] == 5

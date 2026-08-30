"""Stage-1 integration contract for R072 per-view trusted ownership."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from anigroom.grooming import GroomRanges
from anigroom.mesh_roots import TriangleMesh
from tools.train_white_tiger_stage1 import (
    Stage1Config,
    WhiteTigerStage1Model,
    validate_view_gated_ownership_config,
)


RENDER_ARGS = (8, 1, 4, 0.01, 100.0, 10.0, 0.5)
TRUSTED_VIEW = 9
UNTRUSTED_VIEW = 25


def make_model() -> WhiteTigerStage1Model:
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
    return WhiteTigerStage1Model(
        mesh,
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0, 1, 0], dtype=np.int64),
        np.asarray(
            [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
            dtype=np.float32,
        ),
        GroomRanges(),
        torch.device("cpu"),
        init_groom_length=0.018,
        max_child_count=1,
        guide_face_ids=np.asarray([0, 1], dtype=np.int64),
        guide_barycentric=np.asarray(
            [[0.65, 0.20, 0.15], [0.15, 0.20, 0.65]],
            dtype=np.float32,
        ),
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
    )


def install_gate(model: WhiteTigerStage1Model, values: list[float], floor: float = 0.0) -> None:
    model.set_view_gate(
        torch.tensor([TRUSTED_VIEW]),
        torch.tensor([values], dtype=torch.float32),
        floor,
    )


def render_and_backward(
    model: WhiteTigerStage1Model,
    *,
    view_index: int | None = None,
) -> dict[str, torch.Tensor | None]:
    gaussians = model.render_parameters(*RENDER_ARGS, view_index=view_index)[0]
    loss = gaussians.means.square().sum() + gaussians.opacities.square().sum()
    loss += gaussians.colors.square().sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {
        "bary": model.bary_logits.grad,
        "opacity": model.groom.opacity_raw.grad,
        "tip_opacity_ratio": model.groom.tip_opacity_ratio_raw.grad,
        "root_color": model.groom.root_color_raw.grad,
        "length": model.groom.length_raw.grad,
    }


def clone_grads(grads: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
    return {k: (None if v is None else v.detach().clone()) for k, v in grads.items()}


def test_support_off_needs_no_view_index_and_leaves_the_parent_path() -> None:
    model = make_model()
    assert not model.view_gate_enabled()
    grads = render_and_backward(model)
    assert grads["bary"] is not None
    assert float(grads["bary"].abs().sum()) > 0.0


def test_unit_gate_reproduces_the_ungated_gradients_exactly() -> None:
    baseline = clone_grads(render_and_backward(make_model()))

    gated_model = make_model()
    install_gate(gated_model, [1.0, 1.0])
    assert gated_model.view_gate_enabled()
    gated = render_and_backward(gated_model, view_index=TRUSTED_VIEW)

    # Placement and opacity must carry gradient for this comparison to mean
    # anything; other fields may legitimately be guide-owned and unused here.
    for required in ("bary", "opacity"):
        assert baseline[required] is not None

    for name, reference in baseline.items():
        if reference is None:
            assert gated[name] is None, f"{name} gained a gradient under a unit gate"
            continue
        assert gated[name] is not None, f"{name} lost its gradient under a unit gate"
        torch.testing.assert_close(gated[name], reference, rtol=0.0, atol=0.0)


def test_zero_gate_removes_geometry_and_opacity_ownership_but_not_color() -> None:
    model = make_model()
    install_gate(model, [0.0, 0.0])
    grads = render_and_backward(model, view_index=TRUSTED_VIEW)

    assert float(grads["bary"].abs().max()) == 0.0
    assert float(grads["opacity"].abs().max()) == 0.0
    assert float(grads["tip_opacity_ratio"].abs().max()) == 0.0
    # Appearance is deliberately not gated: R072 owns placement, not color.
    assert float(grads["root_color"].abs().sum()) > 0.0


def test_partial_gate_scales_geometry_gradient() -> None:
    baseline = clone_grads(render_and_backward(make_model()))

    scaled_model = make_model()
    share = 0.25
    install_gate(scaled_model, [share, share])
    scaled = render_and_backward(scaled_model, view_index=TRUSTED_VIEW)

    torch.testing.assert_close(
        scaled["bary"],
        baseline["bary"] * share,
        rtol=1.0e-5,
        atol=1.0e-8,
    )
    torch.testing.assert_close(
        scaled["opacity"],
        baseline["opacity"] * share,
        rtol=1.0e-5,
        atol=1.0e-8,
    )


def test_untrusted_training_view_owns_only_the_floor() -> None:
    model = make_model()
    install_gate(model, [1.0, 1.0], floor=0.0)
    grads = render_and_backward(model, view_index=UNTRUSTED_VIEW)
    assert float(grads["bary"].abs().max()) == 0.0
    assert float(grads["opacity"].abs().max()) == 0.0

    floored = make_model()
    install_gate(floored, [1.0, 1.0], floor=1.0)
    floored_grads = render_and_backward(floored, view_index=UNTRUSTED_VIEW)
    assert float(floored_grads["bary"].abs().sum()) > 0.0


def test_render_root_gate_is_interpolated_into_the_unit_interval() -> None:
    model = make_model()
    install_gate(model, [1.0, 0.0])
    _, _, roots_local = model.roots_and_normals()
    gate = model.view_gate_at_render_roots(roots_local, TRUSTED_VIEW)
    assert gate.shape == (int(model.face_ids.shape[0]), 1)
    assert float(gate.min()) >= 0.0
    assert float(gate.max()) <= 1.0
    # A mixed guide gate must actually vary across render roots.
    assert float(gate.max() - gate.min()) > 0.0


def test_render_root_multiplier_preserves_amplification() -> None:
    model = make_model()
    install_gate(model, [7.5, 3.0])
    _, _, roots_local = model.roots_and_normals()
    multiplier = model.view_gate_at_render_roots(roots_local, TRUSTED_VIEW)
    assert float(multiplier.min()) >= 3.0
    assert float(multiplier.max()) <= 7.5
    assert float(multiplier.max()) > 1.0


def test_gradient_disabled_rendering_does_not_require_a_view_index() -> None:
    model = make_model()
    install_gate(model, [1.0, 0.0])
    with torch.no_grad():
        gaussians = model.render_parameters(*RENDER_ARGS)[0]
    assert bool(torch.isfinite(gaussians.means).all())


def test_gradient_enabled_rendering_requires_a_view_index() -> None:
    model = make_model()
    install_gate(model, [1.0, 0.0])
    with pytest.raises(RuntimeError, match="view_index"):
        model.render_parameters(*RENDER_ARGS)


def test_gate_forward_value_is_unchanged_by_ownership() -> None:
    reference = make_model()
    with torch.no_grad():
        reference_means = reference.render_parameters(*RENDER_ARGS)[0].means.clone()

    gated = make_model()
    gated.load_state_dict(reference.state_dict(), strict=True)
    install_gate(gated, [0.0, 0.3])
    means = gated.render_parameters(*RENDER_ARGS, view_index=TRUSTED_VIEW)[0].means
    torch.testing.assert_close(means.detach(), reference_means, rtol=0.0, atol=0.0)


def test_set_view_gate_rejects_invalid_state() -> None:
    model = make_model()
    with pytest.raises(ValueError):
        model.set_view_gate(torch.tensor([1]), torch.zeros((1, 5)), 0.0)
    with pytest.raises(ValueError):
        model.set_view_gate(torch.tensor([1, 1]), torch.zeros((2, 2)), 0.0)
    with pytest.raises(ValueError):
        model.set_view_gate(torch.tensor([1]), torch.full((1, 2), float("nan")), 0.0)
    with pytest.raises(ValueError):
        model.set_view_gate(torch.tensor([1]), torch.full((1, 2), -0.1), 0.0)
    with pytest.raises(ValueError):
        model.set_view_gate(torch.tensor([1]), torch.zeros((1, 2)), 1.5)


def make_config(
    *,
    support: bool,
    floor: float = 0.0,
    normalization: str = "raw_q95",
) -> Stage1Config:
    return Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        child_count=1,
        guide_root_count=2,
        guide_roots_from_clean_flow=True,
        clean_flow_target="target.npz",
        view_gated_ownership_support=support,
        view_gate_floor=floor,
        view_gate_normalization=normalization,
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
    )


def test_config_validation_requires_clean_flow_guides() -> None:
    validate_view_gated_ownership_config(make_config(support=True))
    validate_view_gated_ownership_config(make_config(support=False))

    from dataclasses import replace as dataclass_replace

    with pytest.raises(ValueError, match="clean-flow"):
        validate_view_gated_ownership_config(
            dataclass_replace(make_config(support=True), clean_flow_target="   ")
        )
    with pytest.raises(ValueError, match="clean-flow-owned"):
        validate_view_gated_ownership_config(
            dataclass_replace(make_config(support=True), guide_root_count=0)
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_view_gated_ownership_config(make_config(support=True, floor=2.0))


def test_disabled_support_forbids_a_nonzero_floor() -> None:
    with pytest.raises(ValueError, match="must be zero"):
        validate_view_gated_ownership_config(make_config(support=False, floor=0.5))


def test_budget_normalization_requires_zero_floor_and_valid_mode() -> None:
    validate_view_gated_ownership_config(
        make_config(support=True, normalization="equal_owner_budget")
    )
    with pytest.raises(ValueError, match="requires view gate floor 0"):
        validate_view_gated_ownership_config(
            make_config(
                support=True,
                floor=0.1,
                normalization="equal_owner_budget",
            )
        )
    with pytest.raises(ValueError, match="normalization"):
        validate_view_gated_ownership_config(
            make_config(support=True, normalization="unknown")
        )
    with pytest.raises(ValueError, match="raw_q95"):
        validate_view_gated_ownership_config(
            make_config(support=False, normalization="equal_owner_budget")
        )

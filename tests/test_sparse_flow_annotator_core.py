from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from anigroom.grooming import (
    DECODED_GROOM_GEOMETRY_FIELDS,
    DecodedGroom,
    GroomRanges,
    encode_positive_asinh_ratio,
    straight_through_gate_geometry,
)
from anigroom.mesh_roots import TriangleMesh
from tools.train_white_tiger_stage1 import (
    CURRENT_CHECKPOINT_VERSION,
    Stage1Config,
    WhiteTigerStage1Model,
    build_arg_parser,
    clean_flow_guide_length_anchor_loss,
    config_from_args,
    guide_interpolation_regularization_losses,
    stage1_config_from_checkpoint_mapping,
    validate_clean_flow_guide_length_anchor_config,
    validate_view_gate_length_confidence_config,
    validate_view_gated_ownership_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _decoded_groom() -> DecodedGroom:
    values: dict[str, torch.Tensor] = {}
    for index, name in enumerate(DECODED_GROOM_GEOMETRY_FIELDS):
        channels = 3 if name == "direction_local" else 1
        value = torch.arange(12 if channels == 3 else 4, dtype=torch.float32)
        value = (value.reshape(4, channels) + float(index + 1)).requires_grad_()
        values[name] = value
    return DecodedGroom(
        **values,
        root_color=torch.full((4, 3), 0.25, requires_grad=True),
        tip_color=torch.full((4, 3), 0.75, requires_grad=True),
        root_opacity=torch.full((4, 1), 0.4, requires_grad=True),
        tip_opacity=torch.full((4, 1), 0.6, requires_grad=True),
        opacity=torch.full((4, 1), 0.5, requires_grad=True),
    )


def test_geometry_gate_is_identity_with_exact_geometry_gradients_only() -> None:
    groom = _decoded_groom()
    gate = torch.tensor([[0.0], [0.25], [1.0], [2.0]])
    gated = straight_through_gate_geometry(groom, gate)

    for name in DECODED_GROOM_GEOMETRY_FIELDS:
        assert torch.equal(getattr(gated, name).detach(), getattr(groom, name).detach())
    for name in ("root_color", "tip_color", "root_opacity", "tip_opacity", "opacity"):
        assert torch.equal(getattr(gated, name), getattr(groom, name))

    objective = sum(
        (getattr(gated, name).sum() for name in DECODED_GROOM_GEOMETRY_FIELDS),
        torch.zeros((), dtype=torch.float32),
    )
    objective = objective + sum(
        (getattr(gated, name).sum() for name in ("root_color", "tip_color", "root_opacity", "tip_opacity", "opacity")),
        torch.zeros((), dtype=torch.float32),
    )
    objective.backward()

    for name in DECODED_GROOM_GEOMETRY_FIELDS:
        expected = gate.expand_as(getattr(groom, name))
        torch.testing.assert_close(getattr(groom, name).grad, expected, rtol=0.0, atol=0.0)
    for name in ("root_color", "tip_color", "root_opacity", "tip_opacity", "opacity"):
        torch.testing.assert_close(
            getattr(groom, name).grad,
            torch.ones_like(getattr(groom, name)),
            rtol=0.0,
            atol=0.0,
        )


def _render_model(
    *,
    geometry_support: bool,
    length_confidence_support: bool = False,
    render_geometry_parameterization: str = "zero_centered_residual",
) -> WhiteTigerStage1Model:
    mesh = TriangleMesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
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
        render_geometry_parameterization=render_geometry_parameterization,
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
        view_gate_geometry_support=geometry_support,
        view_gate_length_confidence_support=length_confidence_support,
    )


def _render_length_gradient(*, geometry_support: bool) -> torch.Tensor:
    model = _render_model(geometry_support=geometry_support)
    model.set_view_gate(torch.tensor([9]), torch.zeros((1, 2)), 0.0)
    gaussians = model.render_parameters(8, 1, 4, 0.01, 100.0, 10.0, 0.5, view_index=9)[0]
    model.zero_grad(set_to_none=True)
    gaussians.means.square().sum().backward()
    assert model.guide_length_raw.grad is not None
    return model.guide_length_raw.grad.detach().clone()


def test_render_geometry_gate_is_config_gated() -> None:
    ungated = _render_length_gradient(geometry_support=False)
    gated = _render_length_gradient(geometry_support=True)
    assert float(ungated.abs().sum()) > 0.0
    assert float(gated.abs().max()) == 0.0


def _sample_guide_controls(
    model: WhiteTigerStage1Model,
    *,
    length_gradient_confidence: torch.Tensor | None = None,
):
    roots_local = model.guide_points_local.detach()
    root_face_ids = model.guide_face_ids.detach()
    root_normals, root_tangents, root_bitangents = model.tangent_frames_for_face_ids(
        root_face_ids
    )
    support = model.guide_surface_interpolator().build_support(
        roots_local,
        root_face_ids,
    )
    return model.sample_guide_controls(
        roots_local,
        root_face_ids,
        root_normals,
        root_tangents,
        root_bitangents,
        support=support,
        length_gradient_confidence=length_gradient_confidence,
    )


def _sample_length_gradient(
    length_gradient_confidence: torch.Tensor | None,
) -> torch.Tensor:
    model = _render_model(geometry_support=False)
    controls, _ = _sample_guide_controls(
        model,
        length_gradient_confidence=length_gradient_confidence,
    )
    (controls["length"] * torch.tensor([[1.0], [2.0]])).sum().backward()
    assert model.guide_length_raw.grad is not None
    return model.guide_length_raw.grad.detach().reshape(-1)


def test_source_length_confidence_gate_is_identity_and_scales_only_length() -> None:
    model = _render_model(geometry_support=False)
    plain, plain_direction = _sample_guide_controls(model)
    gated, gated_direction = _sample_guide_controls(
        model,
        length_gradient_confidence=torch.tensor([0.0, 0.25]),
    )
    for name in plain:
        assert torch.equal(gated[name], plain[name])
    assert plain_direction is not None
    assert gated_direction is not None
    assert torch.equal(gated_direction, plain_direction)

    plain_gradient = _sample_length_gradient(None)
    gated_gradient = _sample_length_gradient(torch.tensor([0.0, 0.25]))
    unit_gradient = _sample_length_gradient(torch.tensor([[1.0], [0.25]]))
    assert float(plain_gradient[0].abs()) > 0.0
    assert float(plain_gradient[1].abs()) > 0.0
    torch.testing.assert_close(
        gated_gradient[0],
        torch.zeros(()),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        gated_gradient[1],
        0.25 * plain_gradient[1],
        rtol=0.0,
        atol=1.0e-9,
    )
    torch.testing.assert_close(
        unit_gradient[0],
        plain_gradient[0],
        rtol=0.0,
        atol=1.0e-9,
    )


@pytest.mark.parametrize(
    ("confidence", "message"),
    [
        (torch.zeros((2, 2)), "shape"),
        (torch.tensor([float("nan"), 1.0]), "finite"),
        (torch.tensor([-0.1, 1.0]), "\\[0, 1\\]"),
        (torch.tensor([0.0, 1.1]), "\\[0, 1\\]"),
    ],
)
def test_source_length_confidence_gate_validates_shape_and_range(
    confidence: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _sample_guide_controls(
            _render_model(geometry_support=False),
            length_gradient_confidence=confidence,
        )


def test_render_length_confidence_gate_leaves_residual_length_gradient_active() -> None:
    model = _render_model(
        geometry_support=True,
        length_confidence_support=True,
    )
    model.guide_clean_flow_length_confidence.copy_(torch.tensor([0.0, 1.0]))
    model.set_view_gate(torch.tensor([9]), torch.ones((1, 2)), 0.0)
    gaussians = model.render_parameters(
        8,
        1,
        4,
        0.01,
        100.0,
        10.0,
        0.5,
        view_index=9,
    )[0]
    model.zero_grad(set_to_none=True)
    gaussians.means.square().sum().backward()
    assert model.guide_length_raw.grad is not None
    assert model.render_geometry_residual is not None
    assert model.render_geometry_residual.length_raw.grad is not None
    torch.testing.assert_close(
        model.guide_length_raw.grad[0],
        torch.zeros_like(model.guide_length_raw.grad[0]),
        rtol=0.0,
        atol=0.0,
    )
    assert float(model.guide_length_raw.grad[1].abs()) > 0.0
    assert float(model.render_geometry_residual.length_raw.grad.abs().sum()) > 0.0


def test_non_render_guide_regularizer_does_not_use_length_confidence_gate() -> None:
    model = _render_model(
        geometry_support=True,
        length_confidence_support=True,
        render_geometry_parameterization="absolute_endpoint",
    )
    model.guide_clean_flow_length_confidence.zero_()
    model.set_view_gate(torch.tensor([9]), torch.zeros((1, 2)), 0.0)
    with torch.no_grad():
        model.groom.length_raw.fill_(0.5)
    config = Stage1Config(
        data_root="data",
        mesh_path="mesh",
        output_dir="out",
        guide_prior_weight=1.0,
        guide_prior_length_weight=1.0,
    )
    loss = guide_interpolation_regularization_losses(model, config)
    loss.backward()
    assert model.guide_length_raw.grad is not None
    assert float(model.guide_length_raw.grad.abs().sum()) > 0.0


def _length_anchor(
    raw: torch.Tensor,
    reference: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    area: torch.Tensor,
    reduction: str = "mean_l1",
) -> torch.Tensor:
    return clean_flow_guide_length_anchor_loss(
        raw,
        reference,
        target,
        confidence,
        source_area_weights=area,
        clean_flow_length_init_scale=0.30,
        reduction=reduction,
    )


def test_length_anchor_is_zero_at_identity_and_corrects_a_030_scale() -> None:
    identity = torch.tensor([1.0])
    stored_target = identity * 0.30
    identity_raw = torch.zeros((1,), requires_grad=True)
    identity_loss = _length_anchor(
        identity_raw,
        identity,
        stored_target,
        torch.ones(1),
        torch.ones(1),
    )
    torch.testing.assert_close(identity_loss, torch.zeros(()), rtol=0.0, atol=0.0)

    short_raw = torch.zeros((1,), requires_grad=True)
    short_loss = _length_anchor(
        short_raw,
        stored_target,
        stored_target,
        torch.ones(1),
        torch.ones(1),
    )
    assert float(short_loss) > 0.0
    short_loss.backward()
    assert short_raw.grad is not None
    assert float(short_raw.grad[0]) < 0.0


def test_length_anchor_ignores_zero_confidence_targets() -> None:
    raw = torch.tensor([0.0, 3.0])
    reference = torch.tensor([0.30, 0.30])
    target = torch.tensor([0.30, 100.0])
    confidence = torch.tensor([1.0, 0.0])
    area = torch.tensor([1.0, 1.0])
    expected = _length_anchor(raw[:1], reference[:1], target[:1], confidence[:1], area[:1])
    actual = _length_anchor(raw, reference, target, confidence, area)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_length_anchor_is_invariant_to_area_compensated_duplication() -> None:
    identity = torch.tensor([1.0, 2.0])
    reference = torch.tensor([0.30, 0.60])
    current = torch.tensor([0.60, 1.80])
    raw = encode_positive_asinh_ratio(current, reference)
    target = identity * 0.30
    confidence = torch.tensor([0.25, 0.75])
    area = torch.tensor([2.0, 1.0])
    base = _length_anchor(raw, reference, target, confidence, area)

    # Duplicate only physical source 0 into two half-area rows. Source 1 has
    # a different error and remains one row, so this tests area compensation
    # rather than the trivial duplication of every weighted term.
    duplicated = _length_anchor(
        torch.cat((raw[:1], raw[:1], raw[1:])),
        torch.cat((reference[:1], reference[:1], reference[1:])),
        torch.cat((target[:1], target[:1], target[1:])),
        torch.cat((confidence[:1], confidence[:1], confidence[1:])),
        torch.cat((area[:1] * 0.5, area[:1] * 0.5, area[1:])),
    )
    torch.testing.assert_close(duplicated, base, rtol=0.0, atol=1.0e-7)


def test_length_anchor_tail_concentration_is_selective_and_population_stable() -> None:
    reference = torch.ones(8, dtype=torch.float64)
    target = reference * 0.30
    confidence = torch.ones(8, dtype=torch.float64)
    area = torch.ones(8, dtype=torch.float64)

    uniform_raw = encode_positive_asinh_ratio(
        torch.full_like(reference, 2.0),
        reference,
    )
    uniform_mean = _length_anchor(
        uniform_raw,
        reference,
        target,
        confidence,
        area,
        reduction="mean_l1",
    )
    uniform_tail = _length_anchor(
        uniform_raw,
        reference,
        target,
        confidence,
        area,
        reduction="tail_concentration",
    )
    torch.testing.assert_close(uniform_tail, uniform_mean, rtol=0.0, atol=0.0)

    sparse_current = torch.ones_like(reference)
    sparse_current[0] = 8.0
    sparse_raw = encode_positive_asinh_ratio(sparse_current, reference).detach()
    sparse_mean = _length_anchor(
        sparse_raw,
        reference,
        target,
        confidence,
        area,
        reduction="mean_l1",
    )
    sparse_tail = _length_anchor(
        sparse_raw,
        reference,
        target,
        confidence,
        area,
        reduction="tail_concentration",
    )
    assert float(sparse_tail) > float(sparse_mean)

    sparse_mean_raw = sparse_raw.clone().requires_grad_()
    _length_anchor(
        sparse_mean_raw,
        reference,
        target,
        confidence,
        area,
        reduction="mean_l1",
    ).backward()
    sparse_tail_raw = sparse_raw.clone().requires_grad_()
    _length_anchor(
        sparse_tail_raw,
        reference,
        target,
        confidence,
        area,
        reduction="tail_concentration",
    ).backward()
    assert sparse_mean_raw.grad is not None
    assert sparse_tail_raw.grad is not None
    assert float(sparse_tail_raw.grad[0]) > float(sparse_mean_raw.grad[0])

    duplicated = _length_anchor(
        torch.cat((sparse_raw[:1], sparse_raw[:1], sparse_raw[1:])),
        torch.cat((reference[:1], reference[:1], reference[1:])),
        torch.cat((target[:1], target[:1], target[1:])),
        torch.cat((confidence[:1], confidence[:1], confidence[1:])),
        torch.cat((area[:1] * 0.5, area[:1] * 0.5, area[1:])),
        reduction="tail_concentration",
    )
    torch.testing.assert_close(duplicated, sparse_tail, rtol=0.0, atol=1.0e-12)


def _minimal_parser_args(*extra: str) -> list[str]:
    return [
        "--densify-warmup", "0",
        "--densify-interval", "0",
        "--densify-until", "0",
        "--densify-score-threshold", "0",
        "--densify-min-contribution", "0",
        "--max-splits-per-event", "0",
        "--split-children-per-parent", "0",
        "--split-neighbor-count", "0",
        "--split-candidate-rings", "0",
        "--split-candidate-face-count", "0",
        "--split-min-child-distance", "0",
        "--prune-start", "0",
        "--prune-interval", "0",
        "--prune-min-contribution", "0",
        "--prune-min-opacity", "0",
        "--prune-max-fraction", "0",
        *extra,
    ]


def test_config_parser_checkpoint_and_launcher_contract() -> None:
    parser = build_arg_parser()
    defaults = config_from_args(parser.parse_args(_minimal_parser_args()))
    assert defaults.view_gate_geometry_support is False
    assert defaults.view_gate_length_confidence_support is False
    assert defaults.clean_flow_guide_length_anchor_weight == 0.0
    assert defaults.clean_flow_guide_length_anchor_reduction == "mean_l1"

    config = config_from_args(
        parser.parse_args(
            _minimal_parser_args(
                "--view-gate-geometry-support",
                "--view-gate-length-confidence-support",
                "--clean-flow-guide-length-anchor-weight",
                "0.7",
                "--clean-flow-guide-length-anchor-reduction",
                "tail_concentration",
            )
        )
    )
    assert config.view_gate_geometry_support is True
    assert config.view_gate_length_confidence_support is True
    assert config.clean_flow_guide_length_anchor_weight == 0.7
    assert config.clean_flow_guide_length_anchor_reduction == "tail_concentration"
    restored = stage1_config_from_checkpoint_mapping(asdict(config))
    assert restored == config
    for field_name in (
        "view_gate_geometry_support",
        "view_gate_length_confidence_support",
        "clean_flow_guide_length_anchor_weight",
        "clean_flow_guide_length_anchor_reduction",
    ):
        incomplete = asdict(config)
        del incomplete[field_name]
        with pytest.raises(TypeError, match="incomplete current"):
            stage1_config_from_checkpoint_mapping(incomplete)
    assert CURRENT_CHECKPOINT_VERSION == 14

    baseline = (ROOT / "configs/stage1_baseline.env").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/server/run_white_tiger_stage1.sh").read_text(encoding="utf-8")
    trainer = (ROOT / "tools/train_white_tiger_stage1.py").read_text(encoding="utf-8")
    assert "CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT=0.0" in baseline
    assert "CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_REDUCTION=mean_l1" in baseline
    assert "VIEW_GATE_GEOMETRY_SUPPORT=0" in baseline
    assert "VIEW_GATE_LENGTH_CONFIDENCE_SUPPORT=0" in baseline
    assert 'CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT="${CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT:-0.0}"' in launcher
    assert 'CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_REDUCTION="${CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_REDUCTION:-mean_l1}"' in launcher
    assert 'VIEW_GATE_GEOMETRY_SUPPORT="${VIEW_GATE_GEOMETRY_SUPPORT:-0}"' in launcher
    assert 'VIEW_GATE_LENGTH_CONFIDENCE_SUPPORT="${VIEW_GATE_LENGTH_CONFIDENCE_SUPPORT:-0}"' in launcher
    assert '--clean-flow-guide-length-anchor-weight "$CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT"' in launcher
    assert '--clean-flow-guide-length-anchor-reduction "$CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_REDUCTION"' in launcher
    assert 'cmd+=(--view-gate-geometry-support)' in launcher
    assert 'cmd+=(--view-gate-length-confidence-support)' in launcher
    assert trainer.count('"clean_flow_guide_length_anchor_reliable_fraction"') >= 2
    lock = json.loads((ROOT / "configs/stage1_baseline.lock.json").read_text(encoding="utf-8"))
    assert lock["schema_contract"] == {
        "checkpoint_version": 14,
        "defaults": {
            "clean_flow_guide_length_anchor_weight": 0.0,
            "clean_flow_guide_length_anchor_reduction": "mean_l1",
            "view_gate_geometry_support": False,
            "view_gate_length_confidence_support": False,
        },
    }


def test_new_feature_defaults_are_inert_and_prerequisites_are_strict() -> None:
    default = Stage1Config(data_root="data", mesh_path="mesh", output_dir="out")
    assert default.view_gate_geometry_support is False
    assert default.view_gate_length_confidence_support is False
    assert default.clean_flow_guide_length_anchor_weight == 0.0
    assert default.clean_flow_guide_length_anchor_reduction == "mean_l1"

    valid = replace(
        default,
        view_gated_ownership_support=True,
        view_gate_geometry_support=True,
        guide_root_count=2,
        guide_roots_from_clean_flow=True,
        clean_flow_target="target.npz",
        clean_flow_length_init=True,
        clean_flow_length_init_scale=0.30,
        clean_flow_guide_length_anchor_weight=0.1,
    )
    validate_view_gated_ownership_config(valid)
    validate_clean_flow_guide_length_anchor_config(valid)

    valid_confidence_gate = replace(
        valid,
        view_gate_length_confidence_support=True,
        clean_flow_guide_length_anchor_reduction="tail_concentration",
    )
    validate_view_gate_length_confidence_config(valid_confidence_gate)
    validate_view_gated_ownership_config(valid_confidence_gate)
    validate_clean_flow_guide_length_anchor_config(valid_confidence_gate)

    with pytest.raises(ValueError, match="view-gate geometry support"):
        validate_view_gate_length_confidence_config(
            replace(valid_confidence_gate, view_gate_geometry_support=False)
        )
    with pytest.raises(ValueError, match="view-gated ownership support"):
        validate_view_gate_length_confidence_config(
            replace(valid_confidence_gate, view_gated_ownership_support=False)
        )
    with pytest.raises(ValueError, match="clean-flow-owned"):
        validate_view_gate_length_confidence_config(
            replace(valid_confidence_gate, guide_roots_from_clean_flow=False)
        )
    with pytest.raises(ValueError, match="length init"):
        validate_view_gate_length_confidence_config(
            replace(valid_confidence_gate, clean_flow_length_init=False)
        )
    with pytest.raises(ValueError, match="target"):
        validate_view_gate_length_confidence_config(
            replace(valid_confidence_gate, clean_flow_target=" ")
        )
    with pytest.raises(ValueError, match="positive guide length anchor weight"):
        validate_clean_flow_guide_length_anchor_config(
            replace(
                valid,
                clean_flow_guide_length_anchor_weight=0.0,
                clean_flow_guide_length_anchor_reduction="tail_concentration",
            )
        )

    with pytest.raises(ValueError, match="view-gated ownership"):
        validate_view_gated_ownership_config(
            replace(valid, view_gated_ownership_support=False)
        )
    with pytest.raises(ValueError, match="clean-flow-owned"):
        validate_clean_flow_guide_length_anchor_config(
            replace(valid, guide_roots_from_clean_flow=False)
        )
    with pytest.raises(ValueError, match="length init"):
        validate_clean_flow_guide_length_anchor_config(
            replace(valid, clean_flow_length_init=False)
        )
    with pytest.raises(ValueError, match="target"):
        validate_clean_flow_guide_length_anchor_config(
            replace(valid, clean_flow_target=" ")
        )
    with pytest.raises(ValueError, match="strictly positive"):
        validate_clean_flow_guide_length_anchor_config(
            replace(valid, clean_flow_length_init_scale=0.0)
        )

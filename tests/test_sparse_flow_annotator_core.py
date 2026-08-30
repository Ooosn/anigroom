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
    stage1_config_from_checkpoint_mapping,
    validate_clean_flow_guide_length_anchor_config,
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


def _render_model(*, geometry_support: bool) -> WhiteTigerStage1Model:
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
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
        view_gate_geometry_support=geometry_support,
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


def _length_anchor(
    raw: torch.Tensor,
    reference: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    area: torch.Tensor,
) -> torch.Tensor:
    return clean_flow_guide_length_anchor_loss(
        raw,
        reference,
        target,
        confidence,
        source_area_weights=area,
        clean_flow_length_init_scale=0.30,
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
    assert defaults.clean_flow_guide_length_anchor_weight == 0.0

    config = config_from_args(
        parser.parse_args(
            _minimal_parser_args(
                "--view-gate-geometry-support",
                "--clean-flow-guide-length-anchor-weight",
                "0.7",
            )
        )
    )
    assert config.view_gate_geometry_support is True
    assert config.clean_flow_guide_length_anchor_weight == 0.7
    restored = stage1_config_from_checkpoint_mapping(asdict(config))
    assert restored == config
    for field_name in (
        "view_gate_geometry_support",
        "clean_flow_guide_length_anchor_weight",
    ):
        incomplete = asdict(config)
        del incomplete[field_name]
        with pytest.raises(TypeError, match="incomplete current"):
            stage1_config_from_checkpoint_mapping(incomplete)
    assert CURRENT_CHECKPOINT_VERSION == 13

    baseline = (ROOT / "configs/stage1_baseline.env").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/server/run_white_tiger_stage1.sh").read_text(encoding="utf-8")
    trainer = (ROOT / "tools/train_white_tiger_stage1.py").read_text(encoding="utf-8")
    assert "CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT=0.0" in baseline
    assert "VIEW_GATE_GEOMETRY_SUPPORT=0" in baseline
    assert 'CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT="${CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT:-0.0}"' in launcher
    assert 'VIEW_GATE_GEOMETRY_SUPPORT="${VIEW_GATE_GEOMETRY_SUPPORT:-0}"' in launcher
    assert '--clean-flow-guide-length-anchor-weight "$CLEAN_FLOW_GUIDE_LENGTH_ANCHOR_WEIGHT"' in launcher
    assert 'cmd+=(--view-gate-geometry-support)' in launcher
    assert trainer.count('"clean_flow_guide_length_anchor_reliable_fraction"') >= 2
    lock = json.loads((ROOT / "configs/stage1_baseline.lock.json").read_text(encoding="utf-8"))
    assert lock["schema_contract"] == {
        "checkpoint_version": 13,
        "defaults": {
            "clean_flow_guide_length_anchor_weight": 0.0,
            "view_gate_geometry_support": False,
        },
    }


def test_new_feature_defaults_are_inert_and_prerequisites_are_strict() -> None:
    default = Stage1Config(data_root="data", mesh_path="mesh", output_dir="out")
    assert default.view_gate_geometry_support is False
    assert default.clean_flow_guide_length_anchor_weight == 0.0

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

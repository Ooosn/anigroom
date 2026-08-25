from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest
import torch

import anigroom.grooming.strand_gaussians as strand_gaussians
import tools.train_white_tiger_stage1 as stage1
from anigroom.collision.sdf import SignedDistanceGrid
from anigroom.grooming import (
    GroomParameterField,
    GroomRanges,
    build_brush_centerline,
    build_strands,
)
from anigroom.mesh_roots import TriangleMesh


def _strand_inputs(
    *,
    count: int = 1,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    roots = torch.tensor([[0.03, -0.01, 0.0]], dtype=dtype).expand(count, -1).clone()
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype).expand(count, -1).clone()
    tangents = torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype).expand(count, -1).clone()
    bitangents = torch.cross(normals, tangents, dim=-1)
    return roots, normals, tangents, bitangents


def _groom(
    *,
    count: int = 1,
    dtype: torch.dtype = torch.float64,
    curl_radius_ratio: float = 0.12,
):
    base = GroomParameterField(count, device="cpu").decode()
    values = {
        field.name: getattr(base, field.name).detach().to(dtype=dtype)
        for field in fields(base)
    }
    values.update(
        length=torch.full((count, 1), 0.04, dtype=dtype, requires_grad=True),
        direction_local=torch.tensor(
            [[0.78, 0.21, 0.58]], dtype=dtype
        ).expand(count, -1).clone().requires_grad_(True),
        brush_stiffness=torch.full(
            (count, 1), 0.65, dtype=dtype, requires_grad=True
        ),
        curl_radius_ratio=torch.full(
            (count, 1), curl_radius_ratio, dtype=dtype, requires_grad=True
        ),
        curl_turns=torch.full((count, 1), 1.4, dtype=dtype, requires_grad=True),
        curl_phase=torch.full((count, 1), 0.4, dtype=dtype, requires_grad=True),
    )
    return replace(base, **values)


def _point_loss(points: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(
        -0.7,
        1.1,
        points.numel(),
        dtype=points.dtype,
        device=points.device,
    ).reshape_as(points)
    return (points * weights).sum()


def _clear_geometry_grads(groom) -> None:
    for name in (
        "length",
        "direction_local",
        "brush_stiffness",
        "curl_radius_ratio",
        "curl_turns",
        "curl_phase",
    ):
        getattr(groom, name).grad = None


def _geometry_grads(groom) -> dict[str, torch.Tensor | None]:
    return {
        name: (
            getattr(groom, name).grad.detach().clone()
            if getattr(groom, name).grad is not None
            else None
        )
        for name in (
            "length",
            "direction_local",
            "brush_stiffness",
            "curl_radius_ratio",
            "curl_turns",
            "curl_phase",
        )
    }


def _brush_reference(
    roots: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    groom,
    samples: int,
) -> torch.Tensor:
    roots = roots.to(dtype=normals.dtype, device=normals.device)
    normals = strand_gaussians._normalize(normals)
    tangents = strand_gaussians._normalize(tangents)
    bitangents = strand_gaussians._normalize(bitangents)
    direction_local = strand_gaussians._normalize(groom.direction_local)
    groom_direction = strand_gaussians._normalize(
        direction_local[:, [0]] * tangents
        + direction_local[:, [1]] * bitangents
        + direction_local[:, [2]] * normals
    )
    return build_brush_centerline(
        roots,
        normals,
        groom_direction,
        groom.length,
        groom.brush_stiffness,
        samples,
    )


def test_enabled_path_matches_default_reference_with_curl_gradients() -> None:
    roots, normals, tangents, bitangents = _strand_inputs()
    groom = _groom()
    reference = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=33,
    )
    _point_loss(reference[0]).backward()
    reference_grads = _geometry_grads(groom)
    reference_points = reference[0].detach().clone()
    _clear_geometry_grads(groom)

    explicit = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=33,
        enable_curl=True,
    )
    _point_loss(explicit[0]).backward()
    explicit_grads = _geometry_grads(groom)

    for reference_value, explicit_value in zip(reference, explicit):
        torch.testing.assert_close(
            reference_value,
            explicit_value,
            rtol=0.0,
            atol=0.0,
        )
    for name, reference_gradient in reference_grads.items():
        assert reference_gradient is not None
        assert explicit_grads[name] is not None
        torch.testing.assert_close(
            reference_gradient,
            explicit_grads[name],
            rtol=0.0,
            atol=0.0,
        )
        assert bool(torch.isfinite(explicit_grads[name]).all())
        assert bool((explicit_grads[name].abs() > 0.0).any())

    brush = _brush_reference(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=33,
    )
    assert bool((reference_points - brush).abs().max() > 1.0e-8)


def test_disabled_path_is_exact_brush_backbone() -> None:
    roots, normals, tangents, bitangents = _strand_inputs()
    groom = _groom()
    points, _, _, _ = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=33,
        enable_curl=False,
    )
    brush = _brush_reference(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=33,
    )
    assert torch.equal(points, brush)


def _model() -> stage1.WhiteTigerStage1Model:
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
    return stage1.WhiteTigerStage1Model(
        mesh,
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]], dtype=np.float32),
        GroomRanges(),
        torch.device("cpu"),
        init_groom_length=0.04,
        max_child_count=1,
    )


_RENDER_ARGS = (16, 1, 10, 0.010, 84.19047619047619, 23.771428571428572, 1.45)


def _sdf_field() -> SignedDistanceGrid:
    lower = torch.tensor([-1.0, -1.0, -1.0])
    upper = torch.tensor([2.0, 2.0, 1.0])
    x = torch.linspace(lower[0], upper[0], 17)
    z = torch.linspace(lower[2], upper[2], 33)
    values = (z[:, None, None] - 0.05 * x[None, None, :]).expand(
        33, 17, 17
    ).contiguous()
    return SignedDistanceGrid(values, lower, upper)


def test_model_multiplier_zero_bypasses_deform_backbone(monkeypatch) -> None:
    model = _model()
    model.shape_detail_multiplier = 0.0
    model.shape_curl_scale = 1.0

    def fail(*args, **kwargs):
        raise AssertionError("zero-curl render entered deform_backbone")

    monkeypatch.setattr(strand_gaussians, "deform_backbone", fail)
    with torch.no_grad():
        gaussians, _, _, _, _, _, _ = model.render_parameters(*_RENDER_ARGS)
    assert gaussians.means.ndim == 2


def test_model_multiplier_positive_calls_deform_and_preserves_output(monkeypatch) -> None:
    model = _model()
    model.shape_detail_multiplier = 1.0
    model.shape_curl_scale = 1.0
    with torch.no_grad():
        reference = model.render_parameters(*_RENDER_ARGS)

    real_deform = strand_gaussians.deform_backbone
    calls: list[tuple[object, object]] = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return real_deform(*args, **kwargs)

    monkeypatch.setattr(strand_gaussians, "deform_backbone", record)
    with torch.no_grad():
        actual = model.render_parameters(*_RENDER_ARGS)

    assert len(calls) == 1
    for name in ("means", "directions", "quats", "scales", "colors", "opacities"):
        torch.testing.assert_close(
            getattr(reference[0], name),
            getattr(actual[0], name),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("multiplier", "curl_scale", "expected"),
    ((0.0, 1.0, False), (1.0, 0.0, False), (0.5, 1.0, True)),
)
def test_render_parameters_forwards_one_flag_to_world_and_sdf_strands(
    monkeypatch,
    multiplier: float,
    curl_scale: float,
    expected: bool,
) -> None:
    model = _model()
    model.shape_detail_multiplier = multiplier
    model.shape_curl_scale = curl_scale
    flags: list[bool] = []
    real_build_strands = stage1.build_strands

    def record(*args, **kwargs):
        flags.append(kwargs["enable_curl"])
        return real_build_strands(*args, **kwargs)

    monkeypatch.setattr(stage1, "build_strands", record)
    with torch.no_grad():
        model.render_parameters(
            *_RENDER_ARGS,
            mesh_no_penetration_field=_sdf_field(),
            mesh_no_penetration_root_indices=torch.tensor([0, 1]),
        )
    assert flags == [expected, expected]


def test_disabled_curl_keeps_ordinary_geometry_gradients() -> None:
    roots, normals, tangents, bitangents = _strand_inputs()
    groom = _groom()
    points, _, _, _ = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=33,
        enable_curl=False,
    )
    _point_loss(points).backward()

    for name in ("curl_radius_ratio", "curl_turns", "curl_phase"):
        assert getattr(groom, name).grad is None
    for name in ("length", "direction_local", "brush_stiffness"):
        gradient = getattr(groom, name).grad
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert bool((gradient.abs() > 0.0).any())


def test_disabled_render_parameters_preserves_sdf_shapes_and_gradients() -> None:
    model = _model()
    model.shape_detail_multiplier = 0.0
    model.shape_curl_scale = 1.0
    _, roots, roots_local, _, depth, _, _ = model.render_parameters(
        *_RENDER_ARGS,
        mesh_no_penetration_field=_sdf_field(),
        mesh_no_penetration_root_indices=torch.tensor([0, 1]),
    )

    assert roots.shape == (2, 3)
    assert roots_local.shape == (2, 3)
    assert depth.shape == (2, _RENDER_ARGS[0] - 1)
    depth.mean().backward()

    for parameter in (model.groom.length_raw, model.bary_logits):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool((parameter.grad.abs() > 0.0).any())
    for parameter in (
        model.groom.curl_radius_ratio_raw,
        model.groom.curl_turns_raw,
        model.groom.curl_phase,
    ):
        assert parameter.grad is None

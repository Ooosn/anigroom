from __future__ import annotations

import numpy as np
import pytest
import torch

from anigroom.grooming.geometry_residuals import RenderGeometryResidualField
from anigroom.grooming.secondary_guides import (
    build_parent_conditioned_query_support,
    initialize_parent_conditioned_secondary_roots,
    interpolate_secondary_geometry_residuals,
)
from anigroom.grooming.strand_gaussians import make_tangent_frames
from anigroom.mesh_roots import SurfaceRoots, TriangleMesh
from anigroom.roots.lifecycle import RootStructureUpdate
from anigroom.surface_interpolation import SurfaceFieldInterpolator, SurfaceSupport
from tools.train_white_tiger_stage1 import (
    CURRENT_CHECKPOINT_VERSION,
    Stage1Config,
    WhiteTigerStage1Model,
    build_stage1_model_from_checkpoint,
    dense_groom_ranges,
    make_stage1_optimizer,
    stage1_optimizer_param_names,
)


def grid_mesh(size: int = 5) -> TriangleMesh:
    vertices = np.asarray(
        [[float(x), float(y), 0.0] for y in range(size) for x in range(size)],
        dtype=np.float32,
    )
    faces: list[list[int]] = []
    for y in range(size - 1):
        for x in range(size - 1):
            lower = y * size + x
            faces.append([lower, lower + 1, lower + size])
            faces.append([lower + 1, lower + size + 1, lower + size])
    return TriangleMesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64))


def primary_roots(mesh: TriangleMesh) -> SurfaceRoots:
    face_ids = np.asarray([0, 6, 24, 30], dtype=np.int64)
    barycentric = np.asarray(
        [[0.2, 0.4, 0.4]] * 4,
        dtype=np.float32,
    )
    triangles = mesh.vertices[mesh.faces[face_ids]]
    points = (triangles * barycentric[:, :, None]).sum(axis=1)
    return SurfaceRoots(
        points=points.astype(np.float32),
        face_ids=face_ids,
        barycentric=barycentric,
        selected_candidate_ids=np.arange(4, dtype=np.int64),
        candidate_count=4,
    )


def test_parent_conditioned_secondary_fps_is_balanced_and_deterministic() -> None:
    mesh = grid_mesh()
    primary = primary_roots(mesh)
    interpolator = SurfaceFieldInterpolator(
        vertices=mesh.vertices,
        faces=mesh.faces,
        source_points=primary.points,
        source_face_ids=primary.face_ids,
        neighbor_count=4,
        device="cpu",
    )

    first = initialize_parent_conditioned_secondary_roots(
        mesh,
        primary,
        interpolator,
        18,
        candidate_multiplier=32.0,
        seed=19,
        device="cpu",
    )
    second = initialize_parent_conditioned_secondary_roots(
        mesh,
        primary,
        interpolator,
        18,
        candidate_multiplier=32.0,
        seed=19,
        device="cpu",
    )

    assert first.roots.points.shape == (18, 3)
    np.testing.assert_array_equal(first.roots.face_ids, second.roots.face_ids)
    np.testing.assert_allclose(first.roots.barycentric, second.roots.barycentric)
    counts = np.bincount(first.parent_ids, minlength=4)
    np.testing.assert_array_equal(np.sort(counts), np.asarray([4, 4, 5, 5]))
    for parent_id in range(4):
        ids = np.flatnonzero(first.parent_ids == parent_id)
        np.testing.assert_allclose(first.roots.points[ids[0]], primary.points[parent_id])
        assert first.roots.selected_candidate_ids[ids[0]] == -1


def test_query_support_uses_only_primary_surface_neighborhoods() -> None:
    query_points = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    primary_support = SurfaceSupport(
        indices=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        vertex_path_distances=torch.zeros((2, 2, 3)),
        report={},
    )
    secondary_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.2, 0.2, 0.0],
            [10.0, 0.0, 0.0],
            [10.2, 0.0, 0.0],
            [10.0, 0.2, 0.0],
            [10.2, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    parent_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)

    support = build_parent_conditioned_query_support(
        query_points,
        primary_support,
        secondary_points,
        parent_ids,
        neighbor_count=3,
    )

    assert set(support.indices[0].tolist()) <= {0, 1, 2, 3}
    assert set(support.indices[1].tolist()) <= {4, 5, 6, 7}


def test_secondary_residual_zero_state_and_gradient_chain() -> None:
    source_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    query_points = torch.tensor([[0.25, 0.25, 0.0]], requires_grad=True)
    source_normals = torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1).clone()
    query_normals = torch.tensor([[0.0, 0.0, 1.0]])
    source_tangents, source_bitangents = make_tangent_frames(source_normals)
    query_tangents, query_bitangents = make_tangent_frames(query_normals)
    support = type("Support", (), {})()
    support.indices = torch.tensor([[0, 1, 2]], dtype=torch.long)
    support.report = {}
    field = RenderGeometryResidualField(3)

    zero = interpolate_secondary_geometry_residuals(
        field,
        source_normals,
        source_tangents,
        source_bitangents,
        query_points,
        query_normals,
        query_tangents,
        query_bitangents,
        source_points,
        support,
    )
    for value in zero.raw.values():
        torch.testing.assert_close(value, torch.zeros_like(value))
    torch.testing.assert_close(
        zero.decoded.direction_local,
        torch.zeros_like(zero.decoded.direction_local),
    )

    zero_direction_target = torch.tensor([[0.2, -0.1, 0.05]])
    zero_direction_loss = (
        zero.decoded.direction_local - zero_direction_target
    ).square().sum()
    zero_direction_loss.backward()
    assert field.direction_local_raw.grad is not None
    assert float(field.direction_local_raw.grad.abs().sum()) > 0.0
    field.zero_grad(set_to_none=True)
    query_points.grad = None

    with torch.no_grad():
        field.length_raw.copy_(torch.tensor([[0.0], [0.5], [-0.25]]))
        field.direction_local_raw.copy_(
            torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, -0.1, 0.0]])
        )
    sampled = interpolate_secondary_geometry_residuals(
        field,
        source_normals,
        source_tangents,
        source_bitangents,
        query_points,
        query_normals,
        query_tangents,
        query_bitangents,
        source_points,
        support,
    )
    loss = sampled.raw["length_raw"].sum() + sampled.decoded.direction_local.sum()
    loss.backward()
    assert field.length_raw.grad is not None
    assert field.direction_local_raw.grad is not None
    assert query_points.grad is not None
    assert torch.isfinite(field.length_raw.grad).all()
    assert torch.isfinite(field.direction_local_raw.grad).all()
    assert torch.isfinite(query_points.grad).all()


def test_direction_residual_transport_preserves_vector_magnitude() -> None:
    source_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    query_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    source_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    query_normals = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    source_tangents, source_bitangents = make_tangent_frames(source_normals)
    query_tangents, query_bitangents = make_tangent_frames(query_normals)
    support = type("Support", (), {})()
    support.indices = torch.tensor([[0]], dtype=torch.long)
    support.report = {}
    field = RenderGeometryResidualField(1)
    expected_local = torch.tensor([[0.30, -0.20, 0.40]], dtype=torch.float32)
    with torch.no_grad():
        field.direction_local_raw.copy_(torch.atanh(expected_local))

    sampled = interpolate_secondary_geometry_residuals(
        field,
        source_normals,
        source_tangents,
        source_bitangents,
        query_points,
        query_normals,
        query_tangents,
        query_bitangents,
        source_points,
        support,
    )

    torch.testing.assert_close(
        torch.linalg.norm(sampled.decoded.direction_local, dim=-1),
        torch.linalg.norm(expected_local, dim=-1),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_model_secondary_zero_state_matches_direct_primary_interpolation() -> None:
    mesh = grid_mesh()
    primary = primary_roots(mesh)
    render = primary_roots(mesh)
    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1.0e-8)
    interpolator = SurfaceFieldInterpolator(
        vertices=mesh.vertices,
        faces=mesh.faces,
        source_points=primary.points,
        source_face_ids=primary.face_ids,
        neighbor_count=4,
        device="cpu",
    )
    secondary = initialize_parent_conditioned_secondary_roots(
        mesh,
        primary,
        interpolator,
        8,
        candidate_multiplier=32.0,
        seed=23,
        device="cpu",
    )

    common = dict(
        mesh=mesh,
        face_normals=normals.astype(np.float32),
        face_tangents=None,
        face_ids=render.face_ids,
        barycentric=render.barycentric,
        ranges=dense_groom_ranges(),
        device=torch.device("cpu"),
        guide_face_ids=primary.face_ids,
        guide_barycentric=primary.barycentric,
        guide_interpolation_k=4,
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        guide_length_residual_scale=1.0,
        guide_direction_residual_scale=1.0,
    )
    direct = WhiteTigerStage1Model(
        **common,
        geometry_residual_domain="render",
    )
    hierarchical = WhiteTigerStage1Model(
        **common,
        geometry_residual_domain="secondary_guide",
        secondary_guide_face_ids=secondary.roots.face_ids,
        secondary_guide_barycentric=secondary.roots.barycentric,
        secondary_guide_parent_ids=secondary.parent_ids,
        secondary_guide_interpolation_k=4,
    )

    signed_turns = torch.linspace(
        -1.5,
        1.5,
        int(direct.guide_curl_turns_raw.shape[0]),
    ).view(-1, 1)
    with torch.no_grad():
        direct.guide_curl_turns_raw.copy_(signed_turns)
        hierarchical.guide_curl_turns_raw.copy_(signed_turns)
        direct.groom.curl_turns_raw.fill_(20.0)
        hierarchical.groom.curl_turns_raw.fill_(20.0)
        direct.groom.curl_phase.fill_(1.7)
        hierarchical.groom.curl_phase.fill_(1.7)

    _, direct_normals, direct_roots = direct.roots_and_normals()
    _, hierarchical_normals, hierarchical_roots = hierarchical.roots_and_normals()
    direct_groom = direct.apply_guide_controls(
        direct.groom.decode(),
        direct_roots,
        direct_normals,
    )
    hierarchical_groom = hierarchical.apply_guide_controls(
        hierarchical.groom.decode(),
        hierarchical_roots,
        hierarchical_normals,
    )
    for field_name in (
        "length",
        "root_width",
        "tip_width",
        "width_taper",
        "direction_local",
        "brush_stiffness",
        "curl_radius_ratio",
        "curl_turns",
        "curl_phase",
        "frizz_amplitude_ratio",
        "child_radius",
        "clump_strength",
    ):
        torch.testing.assert_close(
            getattr(hierarchical_groom, field_name),
            getattr(direct_groom, field_name),
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    assert float(direct_groom.curl_turns.min()) < 0.0
    assert float(direct_groom.curl_turns.max()) > 0.0
    torch.testing.assert_close(
        direct_groom.curl_phase,
        torch.zeros_like(direct_groom.curl_phase),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        hierarchical_groom.curl_phase,
        torch.zeros_like(hierarchical_groom.curl_phase),
        atol=0.0,
        rtol=0.0,
    )

    assert hierarchical.render_geometry_residual is None
    assert hierarchical.secondary_geometry_residual is not None
    assert not hasattr(hierarchical.secondary_geometry_residual, "curl_turns_raw")
    with torch.no_grad():
        hierarchical.secondary_geometry_residual.length_raw[1].fill_(0.2)
        hierarchical.secondary_geometry_residual.direction_local_raw[2].copy_(
            torch.tensor([0.1, -0.05, 0.02])
        )
    changed = hierarchical.apply_guide_controls(
        hierarchical.groom.decode(),
        hierarchical_roots,
        hierarchical_normals,
    )
    (changed.length.sum() + changed.direction_local.sum()).backward()
    assert hierarchical.secondary_geometry_residual.length_raw.grad is not None
    assert hierarchical.secondary_geometry_residual.direction_local_raw.grad is not None
    assert hierarchical.secondary_geometry_residual.length_raw.grad.abs().sum() > 0
    assert hierarchical.secondary_geometry_residual.direction_local_raw.grad.abs().sum() > 0

    config = Stage1Config(
        data_root="unused",
        mesh_path="unused",
        output_dir="unused",
        geometry_residual_domain="secondary_guide",
        secondary_guide_root_count=8,
        guide_length_residual_scale=1.0,
        guide_direction_residual_scale=1.0,
    )
    optimizer = make_stage1_optimizer(hierarchical, config)
    names = stage1_optimizer_param_names(hierarchical, config)
    flat_names = {name for group in names for name in group}
    assert "secondary_geometry_residual.length_raw" in flat_names
    assert "secondary_geometry_residual.direction_local_raw" in flat_names
    assert not any(name.startswith("render_geometry_residual.") for name in flat_names)
    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(hierarchical.secondary_geometry_residual.length_raw) in optimizer_parameters

    clone = WhiteTigerStage1Model(
        **common,
        geometry_residual_domain="secondary_guide",
        secondary_guide_face_ids=secondary.roots.face_ids,
        secondary_guide_barycentric=secondary.roots.barycentric,
        secondary_guide_parent_ids=secondary.parent_ids,
        secondary_guide_interpolation_k=4,
    )
    clone.load_state_dict(hierarchical.state_dict(), strict=True)
    for name, parameter in hierarchical.named_parameters():
        torch.testing.assert_close(parameter, dict(clone.named_parameters())[name])

    secondary_before = {
        name: parameter.detach().clone()
        for name, parameter in hierarchical.secondary_geometry_residual.named_parameters()
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
    result = hierarchical.apply_structure_update(update, neighbor_count=4)
    assert result["root_count_after"] == 5
    assert hierarchical.secondary_render_support().indices.shape == (5, 4)
    for name, parameter in hierarchical.secondary_geometry_residual.named_parameters():
        torch.testing.assert_close(parameter, secondary_before[name])


def test_formal_checkpoint_loader_restores_render_and_secondary_domains(
    monkeypatch,
) -> None:
    mesh = grid_mesh()
    primary = primary_roots(mesh)
    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1.0e-8)
    interpolator = SurfaceFieldInterpolator(
        vertices=mesh.vertices,
        faces=mesh.faces,
        source_points=primary.points,
        source_face_ids=primary.face_ids,
        neighbor_count=4,
        device="cpu",
    )
    secondary = initialize_parent_conditioned_secondary_roots(
        mesh,
        primary,
        interpolator,
        8,
        candidate_multiplier=32.0,
        seed=31,
        device="cpu",
    )
    monkeypatch.setattr(
        "tools.train_white_tiger_stage1.read_obj_mesh",
        lambda _path: mesh,
    )

    common_model = dict(
        mesh=mesh,
        face_normals=normals.astype(np.float32),
        face_tangents=None,
        face_ids=primary.face_ids,
        barycentric=primary.barycentric,
        ranges=dense_groom_ranges(),
        device=torch.device("cpu"),
        guide_face_ids=primary.face_ids,
        guide_barycentric=primary.barycentric,
        guide_interpolation_k=4,
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        guide_length_residual_scale=1.0,
        guide_direction_residual_scale=1.0,
    )
    render_model = WhiteTigerStage1Model(
        **common_model,
        geometry_residual_domain="render",
    )
    secondary_model = WhiteTigerStage1Model(
        **common_model,
        geometry_residual_domain="secondary_guide",
        secondary_guide_face_ids=secondary.roots.face_ids,
        secondary_guide_barycentric=secondary.roots.barycentric,
        secondary_guide_parent_ids=secondary.parent_ids,
        secondary_guide_interpolation_k=4,
    )

    common_config = dict(
        data_root="unused",
        mesh_path="unused.obj",
        output_dir="unused",
        guide_root_count=4,
        guide_interpolation_k=4,
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        guide_length_residual_scale=1.0,
        guide_direction_residual_scale=1.0,
        guide_residual_unlock_start=10,
        guide_residual_unlock_end=20,
        guide_residual_initial_multiplier=0.0,
        guide_coverage_residual_unlock_start=10,
        guide_coverage_residual_unlock_end=20,
        guide_coverage_residual_initial_multiplier=0.2,
        shape_detail_freeze_until=12,
    )
    cases = (
        (
            render_model,
            Stage1Config(
                **common_config,
                geometry_residual_domain="render",
            ),
        ),
        (
            secondary_model,
            Stage1Config(
                **common_config,
                geometry_residual_domain="secondary_guide",
                secondary_guide_root_count=8,
                secondary_guide_interpolation_k=4,
            ),
        ),
    )

    for source, config in cases:
        restored = build_stage1_model_from_checkpoint(
            {
                "checkpoint_version": CURRENT_CHECKPOINT_VERSION,
                "model": source.state_dict(),
                "iteration": 15,
            },
            config,
            torch.device("cpu"),
        )
        assert restored.geometry_residual_domain == source.geometry_residual_domain
        assert restored.secondary_guides_enabled() == source.secondary_guides_enabled()
        assert restored.training is False
        assert restored.guide_residual_multiplier == pytest.approx(0.5)
        assert restored.guide_coverage_residual_multiplier == pytest.approx(0.6)
        assert restored.shape_detail_multiplier == pytest.approx(3.0 / 8.0)
        assert restored.secondary_shape_residual_multiplier == pytest.approx(3.0 / 8.0)
        restored_state = restored.state_dict()
        source_state = source.state_dict()
        assert restored_state.keys() == source_state.keys()
        for name, value in source_state.items():
            torch.testing.assert_close(value, restored_state[name])
        missing_turn_state = dict(source_state)
        del missing_turn_state["guide_curl_turns_raw"]
        with pytest.raises(RuntimeError, match="Missing key"):
            build_stage1_model_from_checkpoint(
                {
                    "checkpoint_version": CURRENT_CHECKPOINT_VERSION,
                    "model": missing_turn_state,
                    "iteration": 15,
                },
                config,
                torch.device("cpu"),
            )


def test_formal_checkpoint_loader_rejects_incomplete_secondary_topology(
    monkeypatch,
) -> None:
    mesh = grid_mesh()
    primary = primary_roots(mesh)
    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1.0e-8)
    interpolator = SurfaceFieldInterpolator(
        vertices=mesh.vertices,
        faces=mesh.faces,
        source_points=primary.points,
        source_face_ids=primary.face_ids,
        neighbor_count=4,
        device="cpu",
    )
    secondary = initialize_parent_conditioned_secondary_roots(
        mesh,
        primary,
        interpolator,
        8,
        candidate_multiplier=32.0,
        seed=37,
        device="cpu",
    )
    source = WhiteTigerStage1Model(
        mesh=mesh,
        face_normals=normals.astype(np.float32),
        face_tangents=None,
        face_ids=primary.face_ids,
        barycentric=primary.barycentric,
        ranges=dense_groom_ranges(),
        device=torch.device("cpu"),
        guide_face_ids=primary.face_ids,
        guide_barycentric=primary.barycentric,
        guide_interpolation_k=4,
        geometry_residual_domain="secondary_guide",
        secondary_guide_face_ids=secondary.roots.face_ids,
        secondary_guide_barycentric=secondary.roots.barycentric,
        secondary_guide_parent_ids=secondary.parent_ids,
        secondary_guide_interpolation_k=4,
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        guide_length_residual_scale=1.0,
    )
    state = source.state_dict()
    del state["secondary_guide_parent_ids"]
    monkeypatch.setattr(
        "tools.train_white_tiger_stage1.read_obj_mesh",
        lambda _path: mesh,
    )
    config = Stage1Config(
        data_root="unused",
        mesh_path="unused.obj",
        output_dir="unused",
        guide_root_count=4,
        guide_interpolation_k=4,
        geometry_residual_domain="secondary_guide",
        secondary_guide_root_count=8,
        secondary_guide_interpolation_k=4,
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        guide_length_residual_scale=1.0,
    )

    with pytest.raises(RuntimeError, match="missing persistent topology"):
        build_stage1_model_from_checkpoint(
            {
                "checkpoint_version": CURRENT_CHECKPOINT_VERSION,
                "model": state,
                "iteration": 0,
            },
            config,
            torch.device("cpu"),
        )


def test_formal_checkpoint_loader_rejects_r065_schema_before_model_load(
    monkeypatch,
) -> None:
    mesh = grid_mesh()
    primary = primary_roots(mesh)
    triangles = mesh.vertices[mesh.faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1.0e-8)
    source = WhiteTigerStage1Model(
        mesh=mesh,
        face_normals=normals.astype(np.float32),
        face_tangents=None,
        face_ids=primary.face_ids,
        barycentric=primary.barycentric,
        ranges=dense_groom_ranges(),
        device=torch.device("cpu"),
        guide_face_ids=primary.face_ids,
        guide_barycentric=primary.barycentric,
        guide_interpolation_k=4,
        geometry_residual_domain="render",
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        guide_length_residual_scale=1.0,
    )
    monkeypatch.setattr(
        "tools.train_white_tiger_stage1.read_obj_mesh",
        lambda _path: mesh,
    )
    config = Stage1Config(
        data_root="unused",
        mesh_path="unused.obj",
        output_dir="unused",
        guide_root_count=4,
        guide_interpolation_k=4,
        geometry_residual_domain="render",
        render_geometry_parameterization="zero_centered_asinh_log_length_residual",
        guide_length_residual_scale=1.0,
    )

    with pytest.raises(RuntimeError, match="checkpoint schema mismatch"):
        build_stage1_model_from_checkpoint(
            {
                "checkpoint_version": 7,
                "model": source.state_dict(),
                "iteration": 30_000,
            },
            config,
            torch.device("cpu"),
        )

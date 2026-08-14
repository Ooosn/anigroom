from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from anigroom.collision.sdf import SignedDistanceGrid, strand_penetration_depth
from anigroom.grooming import GroomRanges, build_strands
from anigroom.mesh_roots import TriangleMesh
from tools.train_white_tiger_stage1 import (
    Stage1Config,
    WhiteTigerStage1Model,
    file_sha256,
    load_mesh_no_penetration_field,
    validate_mesh_no_penetration_config,
)


def base_config() -> Stage1Config:
    return Stage1Config(data_root="data", mesh_path="mesh.obj", output_dir="output")


def test_disabled_no_penetration_rejects_silent_inputs() -> None:
    with pytest.raises(ValueError, match="support is disabled"):
        validate_mesh_no_penetration_config(
            replace(base_config(), mesh_no_penetration_sdf="body.npz")
        )
    with pytest.raises(ValueError, match="weight must be zero"):
        validate_mesh_no_penetration_config(
            replace(base_config(), mesh_no_penetration_weight=1.0)
        )


def test_enabled_no_penetration_requires_complete_configuration() -> None:
    with pytest.raises(ValueError, match="requires --mesh-no-penetration-sdf"):
        validate_mesh_no_penetration_config(
            replace(base_config(), mesh_no_penetration_support=True)
        )
    with pytest.raises(ValueError, match="positive loss weight"):
        validate_mesh_no_penetration_config(
            replace(
                base_config(),
                mesh_no_penetration_support=True,
                mesh_no_penetration_sdf="body.npz",
            )
        )


def test_training_loader_verifies_mesh_identity_and_sdf_contract(tmp_path) -> None:
    mesh_path = tmp_path / "body.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="ascii")
    sdf_path = tmp_path / "body_sdf.npz"
    metadata = {
        "mesh_sha256": file_sha256(mesh_path),
        "sign_convention": "outside_positive_inside_negative",
        "storage_order": "zyx",
        "voxel_size": 0.25,
        "signed_distance_backend": "open3d_raycasting_scene",
        "absolute_error_voxels_p95": 0.1,
        "normal_offset_expected_sign_agreement": 1.0,
    }
    np.savez_compressed(
        sdf_path,
        sdf_zyx=np.zeros((2, 2, 2), dtype=np.float32),
        bounds_min=np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
        bounds_max=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    config = replace(
        base_config(),
        mesh_no_penetration_support=True,
        mesh_no_penetration_sdf=str(sdf_path),
        mesh_no_penetration_weight=1.0,
        mesh_no_penetration_root_batch=32,
    )

    field, report = load_mesh_no_penetration_field(
        config,
        mesh_path,
        torch.device("cpu"),
    )

    assert field is not None
    assert report is not None
    assert report["mesh_sha256"] == metadata["mesh_sha256"]
    assert report["shape_zyx"] == [2, 2, 2]


def test_training_loader_rejects_sdf_from_another_mesh(tmp_path) -> None:
    mesh_path = tmp_path / "body.obj"
    mesh_path.write_text("v 0 0 0\n", encoding="ascii")
    sdf_path = tmp_path / "wrong_body_sdf.npz"
    metadata = {
        "mesh_sha256": "0" * 64,
        "sign_convention": "outside_positive_inside_negative",
        "storage_order": "zyx",
        "voxel_size": 0.25,
        "signed_distance_backend": "open3d_raycasting_scene",
        "absolute_error_voxels_p95": 0.1,
        "normal_offset_expected_sign_agreement": 1.0,
    }
    np.savez_compressed(
        sdf_path,
        sdf_zyx=np.zeros((2, 2, 2), dtype=np.float32),
        bounds_min=np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
        bounds_max=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    config = replace(
        base_config(),
        mesh_no_penetration_support=True,
        mesh_no_penetration_sdf=str(sdf_path),
        mesh_no_penetration_weight=1.0,
    )

    with pytest.raises(RuntimeError, match="different mesh"):
        load_mesh_no_penetration_field(config, mesh_path, torch.device("cpu"))


def test_render_parameter_path_backpropagates_collision_to_groom_geometry() -> None:
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
    face_ids = np.asarray([0, 1], dtype=np.int64)
    barycentric = np.asarray(
        [[0.6, 0.2, 0.2], [0.6, 0.2, 0.2]],
        dtype=np.float32,
    )
    model = WhiteTigerStage1Model(
        mesh,
        np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        face_ids,
        barycentric,
        GroomRanges(),
        torch.device("cpu"),
        init_scale=1.75,
        init_translation=(0.2, 0.3, -0.4),
        init_groom_length=0.018,
        max_child_count=1,
    )
    lower = torch.tensor([-1.0, -1.0, -1.0])
    upper = torch.tensor([2.0, 2.0, 1.0])
    x = torch.linspace(lower[0], upper[0], 17)
    z = torch.linspace(lower[2], upper[2], 33)
    values = (z[:, None, None] - 0.05 * x[None, None, :]).expand(
        33, 17, 17
    ).contiguous()
    field = SignedDistanceGrid(values, lower, upper)

    _, _, _, _, depth = model.render_parameters(
        16,
        1,
        10,
        0.010,
        84.19047619047619,
        23.771428571428572,
        1.45,
        mesh_no_penetration_field=field,
        mesh_no_penetration_root_indices=torch.tensor([0, 1]),
    )
    assert depth.shape == (2, 15)
    assert bool((depth > 0.0).all())
    with torch.no_grad():
        roots, normals, roots_local = model.roots_and_normals()
        tangents, bitangents = model.tangent_frames(normals)
        groom = model.apply_guide_controls(model.groom.decode(), roots_local)
        world_strands, _, _, _ = build_strands(
            roots,
            normals,
            tangents,
            bitangents,
            groom,
            samples=16,
        )
        expected_local = (
            world_strands - model.translation.reshape(1, 1, 3)
        ) / torch.exp(model.log_scale).reshape(1, 1, 1)
        expected_depth = strand_penetration_depth(expected_local, field)
    assert torch.allclose(depth, expected_depth, atol=2.0e-7, rtol=2.0e-6)
    depth.mean().backward()
    assert model.groom.length_raw.grad is not None
    assert bool(torch.isfinite(model.groom.length_raw.grad).all())
    assert float(model.groom.length_raw.grad.abs().sum()) > 0.0
    assert model.bary_logits.grad is not None
    assert bool(torch.isfinite(model.bary_logits.grad).all())
    assert float(model.bary_logits.grad.abs().sum()) > 0.0
    assert model.translation.grad is None
    assert model.log_scale.grad is None

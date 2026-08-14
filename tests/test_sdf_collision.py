from __future__ import annotations

import numpy as np
import pytest
import torch
import trimesh

from anigroom.collision.mesh_sdf import (
    build_sdf_grid,
    close_boundary_loops,
    validate_sdf_grid,
)
from anigroom.collision.sdf import (
    SignedDistanceGrid,
    cyclic_strand_indices,
    no_penetration_loss,
    strand_no_penetration_loss,
    strands_world_to_mesh_local,
)


def affine_field() -> SignedDistanceGrid:
    lower = torch.tensor([-2.0, -3.0, -4.0])
    upper = torch.tensor([2.0, 3.0, 4.0])
    x = torch.linspace(lower[0], upper[0], 17)
    y = torch.linspace(lower[1], upper[1], 19)
    z = torch.linspace(lower[2], upper[2], 23)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    values = 2.0 * xx + 3.0 * yy + 5.0 * zz - 0.25
    return SignedDistanceGrid(values, lower, upper)


def sphere_field(scale: float = 1.0) -> SignedDistanceGrid:
    lower = torch.full((3,), -1.5 * scale)
    upper = torch.full((3,), 1.5 * scale)
    axis = torch.linspace(-1.5 * scale, 1.5 * scale, 65)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    values = torch.sqrt(xx.square() + yy.square() + zz.square()) - scale
    return SignedDistanceGrid(values, lower, upper)


def test_query_uses_xyz_points_and_zyx_volume_without_axis_swap() -> None:
    field = affine_field()
    points = torch.tensor(
        [[0.3, -0.5, 1.2], [-1.1, 2.2, -0.7], [1.8, -2.7, 3.5]],
        requires_grad=True,
    )
    expected = 2.0 * points[:, 0] + 3.0 * points[:, 1] + 5.0 * points[:, 2] - 0.25
    actual = field.query(points)
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)
    actual.sum().backward()
    torch.testing.assert_close(
        points.grad,
        torch.tensor([[2.0, 3.0, 5.0]]).expand_as(points),
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_sphere_sign_and_penetration_gradient_push_outward() -> None:
    field = sphere_field()
    points = torch.tensor(
        [[0.5, 0.0, 0.0], [1.2, 0.0, 0.0]],
        requires_grad=True,
    )
    distance = field.query(points)
    assert distance[0] < 0.0
    assert distance[1] > 0.0
    loss, stats = no_penetration_loss(points, field)
    assert stats.penetrating_count == 1
    assert stats.point_count == 2
    loss.backward()
    assert points.grad is not None
    # Gradient descent subtracts this negative x gradient and moves outward.
    assert points.grad[0, 0] < 0.0
    torch.testing.assert_close(points.grad[1], torch.zeros(3))


def test_out_of_volume_points_are_outside_without_boundary_clamping() -> None:
    field = sphere_field()
    points = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 2.0, 0.0]])
    distance = field.query(points)
    assert bool(torch.all(distance > 0.0))
    loss, stats = no_penetration_loss(points, field)
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert stats.penetrating_count == 0


def test_normalized_penetration_is_scale_invariant() -> None:
    unit = sphere_field(scale=1.0)
    scaled = sphere_field(scale=7.0)
    unit_loss, _ = no_penetration_loss(torch.tensor([[0.5, 0.0, 0.0]]), unit)
    scaled_loss, _ = no_penetration_loss(
        torch.tensor([[3.5, 0.0, 0.0]]),
        scaled,
    )
    torch.testing.assert_close(unit_loss, scaled_loss, atol=2.0e-6, rtol=2.0e-6)


def test_strand_loss_excludes_root_and_accepts_a_root_subset() -> None:
    field = sphere_field()
    strands = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.7, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    loss, stats = strand_no_penetration_loss(
        strands,
        field,
        strand_indices=torch.tensor([0]),
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1.0e-7, rtol=0.0)
    assert stats.point_count == 2
    assert stats.penetrating_count == 0


def test_npz_loader_requires_explicit_schema(tmp_path) -> None:
    field = sphere_field()
    path = tmp_path / "sphere_sdf.npz"
    np.savez_compressed(
        path,
        sdf_zyx=field.values_zyx.cpu().numpy(),
        bounds_min=field.bounds_min.cpu().numpy(),
        bounds_max=field.bounds_max.cpu().numpy(),
        metadata_json=np.asarray(
            '{"sign_convention":"outside_positive_inside_negative"}'
        ),
    )
    loaded = SignedDistanceGrid.from_npz(path)
    points = torch.tensor([[0.25, 0.1, -0.3]])
    torch.testing.assert_close(loaded.query(points), field.query(points))
    assert loaded.metadata["sign_convention"] == "outside_positive_inside_negative"
    assert loaded.source_path == str(path.resolve())


def test_npz_loader_rejects_missing_metadata(tmp_path) -> None:
    field = sphere_field()
    path = tmp_path / "missing_metadata.npz"
    np.savez_compressed(
        path,
        sdf_zyx=field.values_zyx.cpu().numpy(),
        bounds_min=field.bounds_min.cpu().numpy(),
        bounds_max=field.bounds_max.cpu().numpy(),
    )
    with pytest.raises(ValueError, match="metadata_json"):
        SignedDistanceGrid.from_npz(path)


def test_boundary_closure_is_explicit_and_watertight() -> None:
    box = trimesh.creation.box(extents=(2.0, 3.0, 4.0))
    top = np.flatnonzero(np.all(box.face_normals > np.asarray([-0.1, -0.1, 0.9]), axis=1))
    opened = trimesh.Trimesh(
        vertices=box.vertices.copy(),
        faces=np.delete(box.faces, top, axis=0),
        process=False,
    )
    assert not opened.is_watertight
    closed, loop_sizes = close_boundary_loops(opened)
    assert loop_sizes == [4]
    assert closed.is_watertight
    assert closed.is_winding_consistent


def test_built_box_sdf_passes_near_surface_accuracy_contract() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.5, 2.0))
    sdf, lower, upper, voxel = build_sdf_grid(
        mesh,
        longest_axis_resolution=33,
        padding_voxels=3,
        query_chunk_size=8192,
        sign_ray_samples=3,
    )
    report = validate_sdf_grid(
        mesh,
        sdf,
        lower,
        upper,
        voxel,
        sample_count=256,
        seed=7,
        sign_ray_samples=3,
    )
    assert report["absolute_error_voxels_p95"] < 0.25
    assert report["normal_offset_expected_sign_agreement"] > 0.99


def test_cyclic_root_blocks_cover_all_roots_without_duplicates() -> None:
    batches = [
        cyclic_strand_indices(10, 4, iteration, device="cpu")
        for iteration in (1, 2, 3)
    ]
    for batch in batches:
        assert int(torch.unique(batch).numel()) == int(batch.numel())
    assert set(torch.cat(batches).tolist()) == set(range(10))


def test_world_to_mesh_local_transform_is_exact_and_differentiable() -> None:
    local = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.25, -0.5, 1.0]]],
        requires_grad=True,
    )
    translation = torch.tensor([0.2, 0.3, -0.4], requires_grad=True)
    scale = torch.tensor([1.75], requires_grad=True)
    world = local * scale.reshape(1, 1, 1) + translation.reshape(1, 1, 3)
    reconstructed = strands_world_to_mesh_local(world, translation, scale)
    torch.testing.assert_close(reconstructed, local)
    reconstructed.square().sum().backward()
    assert local.grad is not None
    assert translation.grad is not None
    assert scale.grad is not None

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh

from .sdf import SignedDistanceGrid


@dataclass(frozen=True)
class MeshSDFBuild:
    sdf_zyx: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    voxel_size: float
    metadata: dict[str, object]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directed_boundary_loops(mesh: trimesh.Trimesh) -> list[list[int]]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    directed = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    boundary = directed[counts[inverse] == 1]
    if boundary.size == 0:
        return []

    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for start, end in boundary.tolist():
        if start in outgoing or end in incoming:
            raise RuntimeError("mesh boundary is not a collection of manifold loops")
        outgoing[int(start)] = int(end)
        incoming[int(end)] = int(start)
    if set(outgoing) != set(incoming):
        raise RuntimeError("mesh boundary contains an open chain")

    loops: list[list[int]] = []
    remaining = set(outgoing)
    while remaining:
        first = min(remaining)
        loop = [first]
        current = first
        while True:
            following = outgoing[current]
            if following == first:
                break
            if following in loop:
                raise RuntimeError("mesh boundary self-intersects in vertex topology")
            loop.append(following)
            current = following
        remaining.difference_update(loop)
        loops.append(loop)
    return loops


def close_boundary_loops(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[int]]:
    """Close oriented manifold boundary loops with centroid triangle fans."""

    loops = _directed_boundary_loops(mesh)
    if not loops:
        return mesh.copy(), []
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = [np.asarray(mesh.faces, dtype=np.int64)]
    for loop in loops:
        center_id = int(vertices.shape[0])
        center = vertices[np.asarray(loop, dtype=np.int64)].mean(axis=0)
        vertices = np.concatenate([vertices, center[None]], axis=0)
        cap = []
        for index, start in enumerate(loop):
            end = loop[(index + 1) % len(loop)]
            cap.append([end, start, center_id])
        faces.append(np.asarray(cap, dtype=np.int64))
    closed = trimesh.Trimesh(
        vertices=vertices,
        faces=np.concatenate(faces, axis=0),
        process=False,
        validate=False,
    )
    if not closed.is_winding_consistent:
        raise RuntimeError("boundary closure produced inconsistent face winding")
    if not closed.is_watertight:
        raise RuntimeError("boundary closure did not produce a watertight mesh")
    return closed, [len(loop) for loop in loops]


def load_collision_mesh(
    path: str | Path,
    *,
    close_boundaries: bool,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    mesh_path = Path(path)
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected a triangle mesh: {mesh_path}")
    if not loaded.is_winding_consistent:
        raise RuntimeError("collision mesh has inconsistent face winding")
    originally_watertight = bool(loaded.is_watertight)
    boundary_loops = _directed_boundary_loops(loaded)
    if boundary_loops and not close_boundaries:
        raise RuntimeError(
            "collision mesh is not watertight; rerun with explicit boundary "
            "closure after inspecting the reported mesh"
        )
    mesh, closed_loop_sizes = (
        close_boundary_loops(loaded) if boundary_loops else (loaded, [])
    )
    return mesh, {
        "mesh_path": str(mesh_path.resolve()),
        "mesh_sha256": file_sha256(mesh_path),
        "vertex_count_before": int(loaded.vertices.shape[0]),
        "face_count_before": int(loaded.faces.shape[0]),
        "originally_watertight": originally_watertight,
        "boundary_loop_sizes": [len(loop) for loop in boundary_loops],
        "closed_boundary_loop_sizes": closed_loop_sizes,
        "vertex_count_sdf": int(mesh.vertices.shape[0]),
        "face_count_sdf": int(mesh.faces.shape[0]),
        "sdf_mesh_watertight": bool(mesh.is_watertight),
    }


def build_sdf_grid(
    mesh: trimesh.Trimesh,
    *,
    longest_axis_resolution: int,
    padding_voxels: int,
    query_chunk_size: int,
    sign_ray_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if longest_axis_resolution < 8:
        raise ValueError("longest_axis_resolution must be at least 8")
    if padding_voxels < 2:
        raise ValueError("padding_voxels must be at least 2")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if sign_ray_samples <= 0 or sign_ray_samples % 2 != 1:
        raise ValueError("sign_ray_samples must be a positive odd integer")
    if not mesh.is_watertight:
        raise RuntimeError("SDF construction requires a watertight mesh")

    mesh_min, mesh_max = np.asarray(mesh.bounds, dtype=np.float64)
    extent = mesh_max - mesh_min
    voxel_size = float(extent.max() / float(longest_axis_resolution - 1))
    bounds_min = mesh_min - float(padding_voxels) * voxel_size
    requested_max = mesh_max + float(padding_voxels) * voxel_size
    shape_xyz = np.ceil((requested_max - bounds_min) / voxel_size).astype(np.int64) + 1
    bounds_max = bounds_min + (shape_xyz - 1) * voxel_size
    nx, ny, nz = (int(value) for value in shape_xyz)
    total = nx * ny * nz
    flat_sdf = np.empty((total,), dtype=np.float32)
    scene = _open3d_raycasting_scene(mesh)
    plane = nx * ny
    for start in range(0, total, int(query_chunk_size)):
        stop = min(start + int(query_chunk_size), total)
        linear = np.arange(start, stop, dtype=np.int64)
        z_index = linear // plane
        remainder = linear - z_index * plane
        y_index = remainder // nx
        x_index = remainder - y_index * nx
        points = np.stack(
            [
                bounds_min[0] + x_index * voxel_size,
                bounds_min[1] + y_index * voxel_size,
                bounds_min[2] + z_index * voxel_size,
            ],
            axis=1,
        )
        flat_sdf[start:stop] = _open3d_signed_distance(
            scene,
            points,
            sign_ray_samples=sign_ray_samples,
        )
    return flat_sdf.reshape(nz, ny, nx), bounds_min, bounds_max, voxel_size


def _open3d_raycasting_scene(mesh: trimesh.Trimesh):
    tensor_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(
            np.asarray(mesh.vertices, dtype=np.float32),
            dtype=o3d.core.Dtype.Float32,
        ),
        o3d.core.Tensor(
            np.asarray(mesh.faces, dtype=np.int32),
            dtype=o3d.core.Dtype.Int32,
        ),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def _open3d_signed_distance(
    scene,
    points: np.ndarray,
    *,
    sign_ray_samples: int,
) -> np.ndarray:
    query = o3d.core.Tensor(
        np.asarray(points, dtype=np.float32),
        dtype=o3d.core.Dtype.Float32,
    )
    return scene.compute_signed_distance(
        query,
        nthreads=0,
        nsamples=int(sign_ray_samples),
    ).numpy()


def validate_sdf_grid(
    mesh: trimesh.Trimesh,
    sdf_zyx: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    voxel_size: float,
    *,
    sample_count: int,
    seed: int,
    sign_ray_samples: int,
) -> dict[str, float | int]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    surface, face_ids = trimesh.sample.sample_surface(
        mesh,
        int(sample_count),
        seed=int(seed),
    )
    normals = np.asarray(mesh.face_normals[face_ids], dtype=np.float64)
    offsets_voxels = np.asarray([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    points = (
        surface[:, None, :]
        + normals[:, None, :] * offsets_voxels[None, :, None] * float(voxel_size)
    ).reshape(-1, 3)
    exact = _open3d_signed_distance(
        _open3d_raycasting_scene(mesh),
        points,
        sign_ray_samples=sign_ray_samples,
    )
    field = SignedDistanceGrid(
        torch.from_numpy(sdf_zyx),
        torch.from_numpy(np.asarray(bounds_min, dtype=np.float32)),
        torch.from_numpy(np.asarray(bounds_max, dtype=np.float32)),
    )
    with torch.no_grad():
        interpolated = field.query(
            torch.from_numpy(points.astype(np.float32))
        ).cpu().numpy()
    error_voxels = np.abs(interpolated - exact) / float(voxel_size)
    sign_expected = np.sign(offsets_voxels)[None, :].repeat(sample_count, axis=0).reshape(-1)
    sign_grid = np.sign(interpolated)
    sign_exact = np.sign(exact)
    return {
        "validation_point_count": int(points.shape[0]),
        "absolute_error_voxels_mean": float(error_voxels.mean()),
        "absolute_error_voxels_p95": float(np.quantile(error_voxels, 0.95)),
        "absolute_error_voxels_max": float(error_voxels.max()),
        "grid_exact_sign_agreement": float(np.mean(sign_grid == sign_exact)),
        "normal_offset_expected_sign_agreement": float(
            np.mean(sign_grid == sign_expected)
        ),
    }


def build_mesh_sdf(
    mesh_path: str | Path,
    *,
    longest_axis_resolution: int,
    padding_voxels: int,
    query_chunk_size: int,
    validation_samples: int,
    validation_seed: int,
    close_boundaries: bool,
    sign_ray_samples: int,
) -> MeshSDFBuild:
    mesh, mesh_report = load_collision_mesh(
        mesh_path,
        close_boundaries=close_boundaries,
    )
    sdf_zyx, bounds_min, bounds_max, voxel_size = build_sdf_grid(
        mesh,
        longest_axis_resolution=longest_axis_resolution,
        padding_voxels=padding_voxels,
        query_chunk_size=query_chunk_size,
        sign_ray_samples=sign_ray_samples,
    )
    validation = validate_sdf_grid(
        mesh,
        sdf_zyx,
        bounds_min,
        bounds_max,
        voxel_size,
        sample_count=validation_samples,
        seed=validation_seed,
        sign_ray_samples=sign_ray_samples,
    )
    metadata = {
        **mesh_report,
        "sign_convention": "outside_positive_inside_negative",
        "storage_order": "zyx",
        "shape_zyx": [int(value) for value in sdf_zyx.shape],
        "voxel_size": float(voxel_size),
        "longest_axis_resolution": int(longest_axis_resolution),
        "padding_voxels": int(padding_voxels),
        "signed_distance_backend": "open3d_raycasting_scene",
        "sign_ray_samples": int(sign_ray_samples),
        "validation_seed": int(validation_seed),
        **validation,
    }
    return MeshSDFBuild(
        sdf_zyx=sdf_zyx,
        bounds_min=np.asarray(bounds_min, dtype=np.float32),
        bounds_max=np.asarray(bounds_max, dtype=np.float32),
        voxel_size=float(voxel_size),
        metadata=metadata,
    )


def save_mesh_sdf(build: MeshSDFBuild, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SDF archive: {output}")
    np.savez_compressed(
        output,
        sdf_zyx=build.sdf_zyx.astype(np.float32, copy=False),
        bounds_min=build.bounds_min.astype(np.float32, copy=False),
        bounds_max=build.bounds_max.astype(np.float32, copy=False),
        voxel_size=np.asarray(build.voxel_size, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(build.metadata, sort_keys=True)),
    )

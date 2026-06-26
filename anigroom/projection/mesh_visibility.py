"""Mesh-depth visibility and UV surface projection.

These utilities make the white-tiger image projection explicit.  A root or UV
texel is allowed to sample an image only when it projects to the visible mesh
surface in that view.  There is no 2D-mask-only fallback path here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anigroom.mesh_roots import TriangleMesh


EPS = 1.0e-8


@dataclass(frozen=True)
class UVAtlas:
    uv_vertices: np.ndarray
    face_uvs: np.ndarray
    vmapping: np.ndarray

    def validate(self, mesh: TriangleMesh) -> None:
        if self.uv_vertices.ndim != 2 or self.uv_vertices.shape[1] != 2:
            raise ValueError("uv_vertices must have shape [U, 2]")
        if self.face_uvs.shape != mesh.faces.shape:
            raise ValueError(f"face_uvs shape {self.face_uvs.shape} does not match mesh faces {mesh.faces.shape}")
        if self.vmapping.ndim != 1 or self.vmapping.shape[0] != self.uv_vertices.shape[0]:
            raise ValueError("vmapping must map every UV vertex to a mesh vertex")
        if int(self.vmapping.min(initial=0)) < 0 or int(self.vmapping.max(initial=0)) >= mesh.vertex_count:
            raise ValueError("vmapping contains mesh vertex ids out of range")


@dataclass
class MeshDepthResult:
    depth: torch.Tensor
    face_id: torch.Tensor
    valid: torch.Tensor


@dataclass
class PointVisibility:
    xy: torch.Tensor
    depth: torch.Tensor
    mesh_depth: torch.Tensor
    depth_delta: torch.Tensor
    in_frame: torch.Tensor
    depth_visible: torch.Tensor
    front_facing: torch.Tensor
    visible: torch.Tensor


def load_xatlas_uv(path: Path) -> UVAtlas:
    data = np.load(path)
    required = {"uv_vertices", "face_uvs", "vmapping"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"UV atlas is missing keys: {sorted(missing)}")
    return UVAtlas(
        uv_vertices=data["uv_vertices"].astype(np.float32),
        face_uvs=data["face_uvs"].astype(np.int64),
        vmapping=data["vmapping"].astype(np.int64),
    )


def _require_nvdiffrast():
    try:
        import nvdiffrast.torch as dr
    except Exception as exc:  # pragma: no cover - hard dependency path
        raise RuntimeError("nvdiffrast is required for mesh-visible projection") from exc
    return dr


def _as_tensor(value: np.ndarray | torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def project_points(points: torch.Tensor, viewmat: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rot = viewmat[:3, :3]
    trans = viewmat[:3, 3]
    cam = points @ rot.T + trans.view(1, 3)
    z = cam[:, 2]
    safe_z = z.clamp_min(1.0e-6)
    x = k[0, 0] * (cam[:, 0] / safe_z) + k[0, 2]
    y = k[1, 1] * (cam[:, 1] / safe_z) + k[1, 2]
    return torch.stack([x, y], dim=-1), z, cam


def render_mesh_depth(
    mesh: TriangleMesh,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    *,
    device: torch.device,
    ctx=None,
) -> MeshDepthResult:
    """Rasterize the mesh depth in the dataset camera coordinate convention."""

    vertices = _as_tensor(mesh.vertices, device=device, dtype=torch.float32)
    faces = _as_tensor(mesh.faces, device=device, dtype=torch.int32).contiguous()
    return render_mesh_depth_from_tensors(vertices, faces, viewmat, k, width, height, device=device, ctx=ctx)


def render_mesh_depth_from_tensors(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    *,
    device: torch.device,
    ctx=None,
) -> MeshDepthResult:
    """Rasterize mesh depth from tensors, preserving gradients to transforms.

    The returned depth map is in the dataset image coordinate convention:
    ``depth[y, x]`` has row 0 at the top of the image.
    """

    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("render_mesh_depth requires CUDA; do not use a CPU fallback for formal projection")
    dr = _require_nvdiffrast()
    if ctx is None:
        ctx = dr.RasterizeCudaContext(device=device)

    vertices = vertices.to(device=device, dtype=torch.float32)
    faces = faces.to(device=device, dtype=torch.int32).contiguous()
    xy, depth, _ = project_points(vertices, viewmat.to(device=device), k.to(device=device))
    x_ndc = xy[:, 0] / max(int(width) - 1, 1) * 2.0 - 1.0
    y_ndc = 1.0 - xy[:, 1] / max(int(height) - 1, 1) * 2.0
    positive = depth > 1.0e-6
    if not bool(positive.any()):
        raise RuntimeError("all mesh vertices are behind the camera")
    near = depth[positive].min()
    far = depth[positive].max()
    z_clip = ((depth - near) / (far - near).clamp_min(1.0e-6)).clamp(0.0, 1.0) * 2.0 - 1.0
    # nvdiffrast keeps the smaller z/w triangle.  Camera z is positive and
    # smaller values are closer.  Use normalized clip z for rasterization and
    # interpolate the true camera depth as an attribute below.
    clip = torch.stack([x_ndc, y_ndc, z_clip, torch.ones_like(depth)], dim=-1)[None]
    rast, _ = dr.rasterize(ctx, clip, faces, resolution=[int(height), int(width)])
    depth_attr, _ = dr.interpolate(depth[None, :, None].contiguous(), rast.contiguous(), faces)
    face_id = rast[0, :, :, 3].long() - 1
    valid = face_id >= 0
    depth_map = depth_attr[0, :, :, 0]
    depth_map = torch.where(valid, depth_map, torch.full_like(depth_map, torch.inf))
    # nvdiffrast uses OpenGL raster coordinates.  Flip the raster output back
    # to the dataset image convention, where row 0 is the top scanline.
    depth_map = torch.flip(depth_map, dims=[0])
    face_id = torch.flip(face_id, dims=[0])
    valid = torch.flip(valid, dims=[0])
    return MeshDepthResult(depth=depth_map, face_id=face_id, valid=valid)


def _local_min_depth(depth: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return depth
    pad = int(kernel_size) // 2
    finite_depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, 1.0e10))
    local_min = -F.max_pool2d(-finite_depth[None, None], kernel_size=int(kernel_size), stride=1, padding=pad)[0, 0]
    return torch.where(local_min < 1.0e9, local_min, torch.full_like(local_min, torch.inf))


def sample_depth_nearest(depth: torch.Tensor, xy: torch.Tensor, *, kernel_size: int = 3) -> torch.Tensor:
    height, width = int(depth.shape[0]), int(depth.shape[1])
    source = _local_min_depth(depth, int(kernel_size))
    ix = torch.round(xy[:, 0]).long().clamp(0, width - 1)
    iy = torch.round(xy[:, 1]).long().clamp(0, height - 1)
    return source[iy, ix]


def sample_mesh_visible_points(
    points: torch.Tensor,
    normals: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    mesh_depth: torch.Tensor,
    *,
    depth_abs_tolerance: float = 2.0e-3,
    depth_rel_tolerance: float = 2.0e-3,
    local_depth_kernel: int = 3,
    front_normal_z: float | None = 0.15,
) -> PointVisibility:
    """Project points and test whether they lie on the visible mesh layer."""

    height, width = int(mesh_depth.shape[0]), int(mesh_depth.shape[1])
    xy, depth, _ = project_points(points, viewmat, k)
    in_frame = (
        (depth > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= height - 1)
    )
    sampled_depth = sample_depth_nearest(mesh_depth, xy, kernel_size=int(local_depth_kernel))
    tolerance = float(depth_abs_tolerance) + depth.abs() * float(depth_rel_tolerance)
    depth_delta = depth - sampled_depth
    depth_visible = in_frame & torch.isfinite(sampled_depth) & (depth_delta.abs() <= tolerance)

    normal_cam = normals @ viewmat[:3, :3].T
    if front_normal_z is None:
        front_facing = torch.ones_like(depth_visible, dtype=torch.bool)
    else:
        # Camera looks along +z.  A visible outward normal usually points
        # toward the camera, i.e. has negative z in camera space.  Keep a small
        # tolerance because animal meshes and reconstructed normals are noisy.
        front_facing = normal_cam[:, 2] <= float(front_normal_z)
    visible = depth_visible & front_facing
    return PointVisibility(
        xy=xy,
        depth=depth,
        mesh_depth=sampled_depth,
        depth_delta=depth_delta,
        in_frame=in_frame,
        depth_visible=depth_visible,
        front_facing=front_facing,
        visible=visible,
    )


def root_uv_from_atlas(atlas: UVAtlas, face_ids: np.ndarray, barycentric: np.ndarray) -> np.ndarray:
    uv_tri = atlas.uv_vertices[atlas.face_uvs[face_ids]]
    return (uv_tri * barycentric[:, :, None]).sum(axis=1).astype(np.float32)


def uv_surface_samples(
    mesh: TriangleMesh,
    atlas: UVAtlas,
    resolution: int,
    *,
    device: torch.device,
    ctx=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize UV texels to mesh surface points.

    Returns ``points, normals, face_ids, valid`` where points/normals are dense
    image-shaped tensors with shape ``[H, W, 3]`` and face_ids/valid have
    shape ``[H, W]``.
    """

    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("uv_surface_samples requires CUDA; do not use a CPU fallback for formal projection")
    atlas.validate(mesh)
    dr = _require_nvdiffrast()
    if ctx is None:
        ctx = dr.RasterizeCudaContext(device=device)

    uv = _as_tensor(atlas.uv_vertices, device=device, dtype=torch.float32)
    face_uvs = _as_tensor(atlas.face_uvs, device=device, dtype=torch.int32).contiguous()
    vmapping = _as_tensor(atlas.vmapping, device=device, dtype=torch.long)
    vertices = _as_tensor(mesh.vertices, device=device, dtype=torch.float32)
    faces = _as_tensor(mesh.faces, device=device, dtype=torch.long)
    uv_vertex_points = vertices[vmapping]

    tri = vertices[faces]
    face_normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    face_normals = F.normalize(face_normals, dim=-1, eps=1.0e-8)
    uv_vertex_normals = face_normals.new_zeros((uv.shape[0], 3))
    # Duplicate UV vertices map to a single mesh vertex, but face normals are
    # face-dependent.  Interpolate positions via UV vertices and gather normals
    # from the rasterized face id below.
    del uv_vertex_normals

    x_ndc = uv[:, 0] * 2.0 - 1.0
    y_ndc = 1.0 - uv[:, 1] * 2.0
    clip = torch.stack([x_ndc, y_ndc, torch.zeros_like(x_ndc), torch.ones_like(x_ndc)], dim=-1)[None]
    rast, _ = dr.rasterize(ctx, clip, face_uvs, resolution=[int(resolution), int(resolution)])
    surface_points, _ = dr.interpolate(uv_vertex_points[None].contiguous(), rast.contiguous(), face_uvs)
    face_ids = rast[0, :, :, 3].long() - 1
    valid = face_ids >= 0
    normals = torch.zeros_like(surface_points[0])
    normals[valid] = face_normals[face_ids[valid]]
    surface_points = torch.flip(surface_points[0], dims=[0])
    normals = torch.flip(normals, dims=[0])
    face_ids = torch.flip(face_ids, dims=[0])
    valid = torch.flip(valid, dims=[0])
    return surface_points, normals, face_ids, valid

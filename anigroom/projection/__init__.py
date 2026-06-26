"""Mesh-visible projection utilities for AniGroom."""

from .mesh_visibility import (
    MeshDepthResult,
    PointVisibility,
    UVAtlas,
    load_xatlas_uv,
    render_mesh_depth,
    render_mesh_depth_from_tensors,
    root_uv_from_atlas,
    sample_depth_nearest,
    sample_mesh_visible_points,
    uv_surface_samples,
)

__all__ = [
    "MeshDepthResult",
    "PointVisibility",
    "UVAtlas",
    "load_xatlas_uv",
    "render_mesh_depth",
    "render_mesh_depth_from_tensors",
    "root_uv_from_atlas",
    "sample_depth_nearest",
    "sample_mesh_visible_points",
    "uv_surface_samples",
]

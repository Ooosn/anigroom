from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1.0e-8


def parallel_transport_vectors(
    vectors: torch.Tensor,
    source_normals: torch.Tensor,
    target_normals: torch.Tensor,
) -> torch.Tensor:
    """Rotate 3D directions with the minimal rotation between surface normals."""

    vectors = F.normalize(vectors, dim=-1, eps=EPS)
    source_normals = F.normalize(source_normals, dim=-1, eps=EPS)
    target_normals = F.normalize(target_normals, dim=-1, eps=EPS)
    axis = torch.linalg.cross(source_normals, target_normals, dim=-1)
    cosine = (source_normals * target_normals).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    sine_sq = (axis * axis).sum(dim=-1, keepdim=True)
    first = torch.linalg.cross(axis, vectors, dim=-1)
    second = torch.linalg.cross(axis, first, dim=-1)
    rotated = vectors + first + second * ((1.0 - cosine) / sine_sq.clamp_min(EPS))

    parallel = sine_sq <= EPS
    basis_x = torch.zeros_like(source_normals)
    basis_x[..., 0] = 1.0
    basis_y = torch.zeros_like(source_normals)
    basis_y[..., 1] = 1.0
    helper = torch.where(source_normals[..., :1].abs() < 0.9, basis_x, basis_y)
    half_turn_axis = F.normalize(
        torch.linalg.cross(source_normals, helper, dim=-1),
        dim=-1,
        eps=EPS,
    )
    half_turn = 2.0 * (vectors * half_turn_axis).sum(dim=-1, keepdim=True) * half_turn_axis - vectors
    rotated = torch.where(parallel & (cosine >= 0.0), vectors, rotated)
    rotated = torch.where(parallel & (cosine < 0.0), half_turn, rotated)
    return F.normalize(rotated, dim=-1, eps=EPS)

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PenetrationStats:
    """Detached diagnostics for a no-penetration loss evaluation."""

    point_count: int
    penetrating_count: int
    penetrating_fraction: float
    mean_depth: float
    maximum_depth: float


class SignedDistanceGrid(nn.Module):
    """Trilinearly sampled mesh SDF with an outside-positive convention.

    The volume is stored in ``[z, y, x]`` order, matching PyTorch's
    ``[D, H, W]`` convention. World-space query points remain ``[x, y, z]``;
    no implicit axis permutation is performed.
    """

    def __init__(
        self,
        values_zyx: torch.Tensor,
        bounds_min: torch.Tensor,
        bounds_max: torch.Tensor,
    ) -> None:
        super().__init__()
        values = torch.as_tensor(values_zyx, dtype=torch.float32)
        lower = torch.as_tensor(bounds_min, dtype=torch.float32).reshape(3)
        upper = torch.as_tensor(bounds_max, dtype=torch.float32).reshape(3)
        if values.ndim != 3 or min(values.shape) < 2:
            raise ValueError("values_zyx must have shape [D, H, W], each at least 2")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("SDF grid contains non-finite values")
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()):
            raise ValueError("SDF bounds must be finite")
        if not bool(torch.all(upper > lower)):
            raise ValueError("SDF bounds_max must be greater than bounds_min")
        self.register_buffer("values_zyx", values.contiguous())
        self.register_buffer("bounds_min", lower.contiguous())
        self.register_buffer("bounds_max", upper.contiguous())
        self.metadata: dict[str, object] = {}
        self.source_path: str | None = None

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> "SignedDistanceGrid":
        source = Path(path).resolve()
        with np.load(source, allow_pickle=False) as archive:
            required = {"sdf_zyx", "bounds_min", "bounds_max", "metadata_json"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"SDF archive is missing arrays: {missing}")
            field = cls(
                torch.from_numpy(archive["sdf_zyx"]),
                torch.from_numpy(archive["bounds_min"]),
                torch.from_numpy(archive["bounds_max"]),
            )
            metadata = json.loads(str(archive["metadata_json"].item()))
            if not isinstance(metadata, dict):
                raise TypeError("SDF metadata_json must decode to an object")
            field.metadata = metadata
            field.source_path = str(source)
        return field.to(device=device) if device is not None else field

    @property
    def reference_length(self) -> torch.Tensor:
        """Mesh-volume diagonal used to make penetration depth dimensionless."""

        return torch.linalg.vector_norm(self.bounds_max - self.bounds_min)

    def query(self, points: torch.Tensor) -> torch.Tensor:
        """Return differentiable signed distances for ``[..., 3]`` points.

        The SDF volume encloses the full mesh. A point outside that volume is
        therefore known to be outside the mesh and receives a positive
        distance to the volume rather than a silently clamped boundary value.
        """

        if points.shape[-1] != 3:
            raise ValueError("SDF query points must have shape [..., 3]")
        if not points.is_floating_point():
            raise TypeError("SDF query points must use a floating dtype")
        original_shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        if flat.numel() == 0:
            return points.new_empty(original_shape)

        lower = self.bounds_min.to(device=flat.device, dtype=flat.dtype)
        upper = self.bounds_max.to(device=flat.device, dtype=flat.dtype)
        normalized = 2.0 * (flat - lower) / (upper - lower) - 1.0
        grid = normalized.view(1, 1, 1, -1, 3)
        volume = self.values_zyx.to(device=flat.device, dtype=flat.dtype)
        sampled = F.grid_sample(
            volume.view(1, 1, *volume.shape),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(-1)

        below = torch.relu(lower - flat)
        above = torch.relu(flat - upper)
        outside_distance = torch.linalg.vector_norm(below + above, dim=-1)
        in_bounds = torch.logical_and(flat >= lower, flat <= upper).all(dim=-1)
        distances = torch.where(in_bounds, sampled, outside_distance)
        return distances.reshape(original_shape)


def no_penetration_loss(
    points: torch.Tensor,
    field: SignedDistanceGrid,
) -> tuple[torch.Tensor, PenetrationStats]:
    """Penalize dimensionless depth inside a body SDF."""

    signed_distance = field.query(points)
    reference = field.reference_length.to(
        device=signed_distance.device,
        dtype=signed_distance.dtype,
    )
    depth = torch.relu(-signed_distance) / reference
    loss = depth.mean() if depth.numel() else points.sum() * 0.0
    detached = depth.detach()
    penetrating = detached > 0.0
    stats = PenetrationStats(
        point_count=int(detached.numel()),
        penetrating_count=int(penetrating.sum().cpu()),
        penetrating_fraction=(
            float(penetrating.float().mean().cpu()) if detached.numel() else 0.0
        ),
        mean_depth=float(detached.mean().cpu()) if detached.numel() else 0.0,
        maximum_depth=float(detached.max().cpu()) if detached.numel() else 0.0,
    )
    return loss, stats


def penetration_depth(
    points: torch.Tensor,
    field: SignedDistanceGrid,
) -> torch.Tensor:
    """Return dimensionless inside depth without host-side diagnostics."""

    signed_distance = field.query(points)
    reference = field.reference_length.to(
        device=signed_distance.device,
        dtype=signed_distance.dtype,
    )
    return torch.relu(-signed_distance) / reference


def cyclic_strand_indices(
    root_count: int,
    sample_count: int,
    iteration: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Select a deterministic rotating root block with complete coverage."""

    if root_count <= 0:
        raise ValueError("root_count must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    count = min(int(root_count), int(sample_count))
    start = ((max(int(iteration), 1) - 1) * count) % int(root_count)
    return (
        torch.arange(count, device=device, dtype=torch.long) + int(start)
    ) % int(root_count)


def strands_world_to_mesh_local(
    strands: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Undo the shared isotropic mesh transform for collision queries."""

    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape [R, S, 3]")
    offset = translation.reshape(1, 1, 3).to(
        device=strands.device,
        dtype=strands.dtype,
    )
    if scale.numel() != 1:
        raise ValueError("mesh collision transform requires one isotropic scale")
    factor = scale.reshape(1, 1, 1).to(
        device=strands.device,
        dtype=strands.dtype,
    )
    return (strands - offset) / factor


def strand_no_penetration_loss(
    strands: torch.Tensor,
    field: SignedDistanceGrid,
    *,
    strand_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, PenetrationStats]:
    """Apply no-penetration to strand samples while excluding every root."""

    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape [R, S, 3]")
    if strands.shape[1] < 2:
        raise ValueError("strands must contain a root and at least one non-root sample")
    selected = strands if strand_indices is None else strands[strand_indices]
    return no_penetration_loss(selected[:, 1:, :], field)


def strand_penetration_depth(
    strands: torch.Tensor,
    field: SignedDistanceGrid,
    *,
    strand_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return non-root penetration depths without synchronizing diagnostics."""

    if strands.ndim != 3 or strands.shape[-1] != 3:
        raise ValueError("strands must have shape [R, S, 3]")
    if strands.shape[1] < 2:
        raise ValueError("strands must contain a root and at least one non-root sample")
    selected = strands if strand_indices is None else strands[strand_indices]
    return penetration_depth(selected[:, 1:, :], field)

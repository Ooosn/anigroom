"""Persistent per-Gaussian RGB residuals over adaptive strand sampling."""

from __future__ import annotations

import torch
from torch import nn


class GaussianRGBResidualField(nn.Module):
    """A per-render-root RGB profile sampled by generated Gaussian segments.

    Adaptive strand sampling changes the number of generated Gaussians as the
    groom changes. Storing a parameter for the transient flattened Gaussian
    list would therefore invalidate optimizer state. This field stores a fixed
    normalized arc-length profile per render root and evaluates it at each
    generated segment midpoint. Under a fixed discretization this is a true
    per-Gaussian RGB residual; across discretizations its meaning stays stable.
    """

    def __init__(
        self,
        root_count: int,
        control_points: int,
        scale: float,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if int(root_count) <= 0:
            raise ValueError("root_count must be positive")
        if int(control_points) < 2:
            raise ValueError("control_points must be at least two")
        if float(scale) <= 0.0:
            raise ValueError("scale must be positive")
        self.root_count = int(root_count)
        self.control_points = int(control_points)
        self.scale = float(scale)
        self.raw = nn.Parameter(
            torch.zeros(
                (self.root_count, self.control_points, 3),
                dtype=torch.float32,
                device=device,
            )
        )

    def sample_raw(
        self,
        root_indices: torch.Tensor,
        normalized_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Linearly sample raw RGB controls at positions in ``[0, 1]``."""

        root_indices = root_indices.to(device=self.raw.device, dtype=torch.long)
        normalized_positions = normalized_positions.to(
            device=self.raw.device,
            dtype=self.raw.dtype,
        ).reshape(-1)
        if root_indices.ndim != 1 or root_indices.shape != normalized_positions.shape:
            raise ValueError("root_indices and normalized_positions must have shape [G]")
        if root_indices.numel() == 0:
            return self.raw.new_empty((0, 3))
        if int(root_indices.min()) < 0 or int(root_indices.max()) >= self.root_count:
            raise IndexError("root index is outside the RGB residual field")
        if not bool(torch.isfinite(normalized_positions).all()):
            raise ValueError("normalized_positions must be finite")

        coordinate = normalized_positions.clamp(0.0, 1.0) * float(self.control_points - 1)
        lower = torch.floor(coordinate).to(dtype=torch.long)
        upper = (lower + 1).clamp_max(self.control_points - 1)
        weight = (coordinate - lower.to(dtype=coordinate.dtype)).unsqueeze(-1)
        lower_value = self.raw[root_indices, lower]
        upper_value = self.raw[root_indices, upper]
        return lower_value * (1.0 - weight) + upper_value * weight

    def segment_residual(
        self,
        root_indices: torch.Tensor,
        segment_indices: torch.Tensor,
        segment_counts: torch.Tensor,
        *,
        multiplier: float,
    ) -> torch.Tensor:
        """Evaluate RGB deltas at normalized Gaussian-segment midpoints."""

        root_indices = root_indices.to(device=self.raw.device, dtype=torch.long)
        segment_indices = segment_indices.to(device=self.raw.device, dtype=torch.long)
        segment_counts = segment_counts.to(device=self.raw.device, dtype=torch.long)
        if root_indices.ndim != 1 or segment_indices.shape != root_indices.shape:
            raise ValueError("root_indices and segment_indices must have shape [G]")
        if segment_counts.shape != (self.root_count,):
            raise ValueError(
                "segment_counts must have one value per render root: "
                f"{tuple(segment_counts.shape)} != {(self.root_count,)}"
            )
        if root_indices.numel() == 0:
            return self.raw.new_empty((0, 3))
        counts = segment_counts[root_indices]
        if bool((counts < 1).any()):
            raise ValueError("segment counts must be positive")
        if bool(((segment_indices < 0) | (segment_indices >= counts)).any()):
            raise ValueError("segment index is outside its strand segment count")
        positions = (
            segment_indices.to(dtype=self.raw.dtype) + 0.5
        ) / counts.to(dtype=self.raw.dtype)
        sampled = self.sample_raw(root_indices, positions)
        return torch.tanh(sampled) * self.scale * float(multiplier)

    def apply_to_colors(
        self,
        colors: torch.Tensor,
        root_indices: torch.Tensor,
        segment_indices: torch.Tensor,
        segment_counts: torch.Tensor,
        *,
        multiplier: float,
    ) -> torch.Tensor:
        """Add the sampled residual without changing geometry or opacity."""

        if float(multiplier) == 0.0:
            return colors
        if colors.ndim != 2 or colors.shape[-1] != 3:
            raise ValueError("colors must have shape [G, 3]")
        residual = self.segment_residual(
            root_indices,
            segment_indices,
            segment_counts,
            multiplier=multiplier,
        )
        if residual.shape != colors.shape:
            raise RuntimeError(
                f"RGB residual shape mismatch: {tuple(residual.shape)} != {tuple(colors.shape)}"
            )
        return (colors + residual.to(device=colors.device, dtype=colors.dtype)).clamp(0.0, 1.0)

    @torch.no_grad()
    def stats(
        self,
        *,
        multiplier: float,
        root_chunk_size: int = 4096,
    ) -> dict[str, float | int]:
        """Return exact decoded statistics without materializing the full field."""

        root_chunk_size = max(1, int(root_chunk_size))
        scale = self.scale * float(multiplier)
        element_count = 0
        absolute_sum = 0.0
        square_sum = 0.0
        absolute_max = 0.0
        active_count = 0
        saturated_count = 0
        for start in range(0, self.root_count, root_chunk_size):
            normalized = torch.tanh(self.raw[start : start + root_chunk_size])
            decoded = normalized * scale
            absolute = decoded.abs()
            element_count += int(decoded.numel())
            absolute_sum += float(absolute.sum().cpu())
            square_sum += float(decoded.square().sum().cpu())
            absolute_max = max(absolute_max, float(absolute.max().cpu()))
            active_count += int((absolute > 1.0e-4).sum().cpu())
            saturated_count += int((normalized.abs() > 0.95).sum().cpu())

        return {
            "root_count": self.root_count,
            "control_points": self.control_points,
            "scale": self.scale,
            "multiplier": float(multiplier),
            "abs_mean": absolute_sum / float(element_count),
            "rms": (square_sum / float(element_count)) ** 0.5,
            "abs_max": absolute_max,
            "active_fraction": active_count / float(element_count),
            "saturation_fraction": saturated_count / float(element_count),
        }

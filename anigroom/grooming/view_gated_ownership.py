"""Per-view trusted ownership of render-root position, opacity, and lifecycle evidence.

R071 showed that a bounded view-dependent appearance field cannot repair the
Panda multiview defect: the guide SH saturated while the render-root population
stayed granular. The matched single-view control had already established that
the granularity comes from thirty views competing over the same pre-9k degrees
of freedom, not from appearance alone.

This module keeps the accepted V7 trusted-view evidence as the only source of
ownership and exposes it as a gradient gate. The forward value is never
modified, so a gate of one reproduces the parent run exactly and any measured
difference is attributable to gradient ownership alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .guide_view_sh import TrustedGuideViewConfidence


def straight_through_gate(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Scale the gradient reaching ``value`` without changing its forward value.

    ``gate`` must broadcast against ``value``. The returned tensor is
    numerically identical to ``value``; only the backward path is scaled.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError("value must be a tensor")
    if not isinstance(gate, torch.Tensor):
        raise TypeError("gate must be a tensor")
    gate = gate.to(device=value.device, dtype=value.dtype)
    if not bool(torch.isfinite(gate).all()):
        raise ValueError("gate must be finite")
    if bool((gate < 0.0).any()) or bool((gate > 1.0).any()):
        raise ValueError("gate must lie in [0, 1]")
    try:
        gate = torch.broadcast_to(gate, value.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"gate shape {tuple(gate.shape)} does not broadcast to {tuple(value.shape)}"
        ) from exc
    detached = value.detach()
    return detached + gate * (value - detached)


@dataclass(frozen=True)
class ViewGatedOwnership:
    """Turn V7 trusted-view evidence into a per-view guide gradient gate.

    ``floor`` is the gradient share retained where the trusted evidence is
    zero. ``0.0`` detaches untrusted roots for that view and leaves them to the
    view-independent surface regularizers; ``1.0`` disables gating entirely and
    reproduces the parent behavior.
    """

    confidence: TrustedGuideViewConfidence
    floor: float = 0.0

    def __post_init__(self) -> None:
        floor = float(self.floor)
        if not (0.0 <= floor <= 1.0):
            raise ValueError(f"floor must lie in [0, 1], got {floor}")

    @property
    def guide_count(self) -> int:
        return self.confidence.guide_count

    def has_view(self, view_index: int) -> bool:
        """Return whether this view carries trusted V7 evidence."""

        matches = torch.nonzero(
            self.confidence.view_indices == int(view_index),
            as_tuple=False,
        )
        return bool(matches.numel() > 0)

    def guide_gate(self, view_index: int) -> torch.Tensor:
        """Return the ``[G]`` gradient share this view owns per primary guide.

        A view absent from the trusted V7 set owns exactly ``floor`` everywhere.
        Those views still render forward; they simply do not claim ownership of
        root placement, opacity, or lifecycle evidence.
        """

        raw = self.confidence.confidence_for_view(int(view_index)).clamp(0.0, 1.0)
        floor = float(self.floor)
        if floor <= 0.0:
            return raw
        return raw * (1.0 - floor) + floor

    def report(self, view_indices: list[int] | tuple[int, ...]) -> dict:
        """Summarize how the trusted set covers a concrete training view list."""

        requested = [int(value) for value in view_indices]
        trusted = [value for value in requested if self.has_view(value)]
        untrusted = [value for value in requested if not self.has_view(value)]
        gates = [self.guide_gate(value) for value in requested]
        stacked = (
            torch.stack(gates, dim=0)
            if gates
            else self.confidence.confidence.new_zeros((0, self.guide_count))
        )
        owned = stacked > 0.0
        guide_support = owned.sum(dim=0) if owned.numel() else owned.new_zeros((self.guide_count,))
        return {
            "floor": float(self.floor),
            "requested_view_count": len(requested),
            "trusted_view_count": len(trusted),
            "untrusted_view_indices": untrusted,
            "guide_count": self.guide_count,
            "owned_fraction": float(owned.float().mean()) if owned.numel() else 0.0,
            "guides_with_owner_fraction": (
                float((guide_support > 0).float().mean()) if guide_support.numel() else 0.0
            ),
            "owner_views_per_guide_mean": (
                float(guide_support.float().mean()) if guide_support.numel() else 0.0
            ),
            "gate_mean": float(stacked.mean()) if stacked.numel() else 0.0,
            "source_path": self.confidence.source_path,
            "summary_path": self.confidence.summary_path,
        }

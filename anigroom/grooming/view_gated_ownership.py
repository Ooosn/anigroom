"""Per-view trusted ownership for roots, opacity, lifecycle, and opt-in geometry.

R071 showed that a bounded view-dependent appearance field cannot repair the
Panda multiview defect: the guide SH saturated while the render-root population
stayed granular. The matched single-view control had already established that
the granularity comes from thirty views competing over the same pre-9k degrees
of freedom, not from appearance alone.

This module keeps the accepted V7 trusted-view evidence as the only source of
ownership and exposes it as a gradient multiplier. The historical render-root
position, root/tip-opacity, and lifecycle ownership behavior remains unchanged.
When explicitly enabled by Stage 1, ``straight_through_gate_geometry`` applies
the same render-root multiplier to the eleven decoded groom geometry fields;
appearance and opacity fields remain outside that opt-in gate. Forward values
are never modified, so a multiplier of one reproduces the parent run exactly
and any measured difference is attributable to gradient ownership alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .guide_view_sh import TrustedGuideViewConfidence


DECODED_GROOM_GEOMETRY_FIELDS = (
    "length",
    "root_width",
    "tip_width",
    "width_taper",
    "direction_local",
    "brush_stiffness",
    "curl_radius_ratio",
    "curl_turns",
    "curl_phase",
    "child_radius",
    "clump_strength",
)


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
    if bool((gate < 0.0).any()):
        raise ValueError("gate must be non-negative")
    try:
        gate = torch.broadcast_to(gate, value.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"gate shape {tuple(gate.shape)} does not broadcast to {tuple(value.shape)}"
        ) from exc
    detached = value.detach()
    return detached + gate * (value - detached)


def straight_through_gate_geometry(groom, gate: torch.Tensor):
    """Straight-through gate every geometry field in a decoded groom.

    The decoded groom's appearance and opacity fields are intentionally left
    untouched.  Each geometry field keeps its exact forward value while its
    backward gradient is multiplied by the broadcast render-root ``gate``.
    """

    geometry = {
        name: straight_through_gate(getattr(groom, name), gate)
        for name in DECODED_GROOM_GEOMETRY_FIELDS
    }
    return replace(groom, **geometry)


# Descriptive aliases keep the helper convenient for callers that name the
# operation after the decoded groom rather than its straight-through detail.
gate_decoded_groom_geometry = straight_through_gate_geometry


@dataclass(frozen=True)
class ViewGatedOwnership:
    """Turn V7 evidence into the historical per-view ownership gradient gate.

    ``floor`` is the gradient share retained where the trusted evidence is
    zero. ``0.0`` detaches untrusted roots for that view and leaves them to the
    view-independent surface regularizers. This class supplies the gate used
    for root position, root/tip opacity, and lifecycle evidence; callers may
    separately pass it to the opt-in decoded-geometry helper. ``1.0`` disables
    gating entirely and reproduces the parent behavior.
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

    def cache_matrix(
        self,
        training_view_indices: list[int] | tuple[int, ...],
        *,
        mode: str = "raw_q95",
    ) -> torch.Tensor:
        """Return a ``[V,G]`` matrix aligned to ``confidence.view_indices``.

        ``raw_q95`` reproduces R072 exactly. ``equal_owner_budget`` keeps the
        same nonzero trusted support but gives each owner of guide ``g`` the
        multiplier ``N_train / k_g``. Uniform sampling over the concrete
        training-view list then has expected multiplier one for every guide
        with at least one owner, without introducing a tuned scale.
        """

        requested = [int(value) for value in training_view_indices]
        if not requested:
            raise ValueError("training_view_indices must not be empty")
        if len(set(requested)) != len(requested):
            raise ValueError("training_view_indices must be unique")
        if mode == "raw_q95":
            return torch.stack(
                [
                    self.guide_gate(int(view))
                    for view in self.confidence.view_indices.tolist()
                ],
                dim=0,
            )
        if mode != "equal_owner_budget":
            raise ValueError(f"unsupported ownership normalization mode: {mode}")
        if float(self.floor) != 0.0:
            raise ValueError("equal_owner_budget requires floor=0")

        requested_set = set(requested)
        in_training = torch.tensor(
            [
                int(view) in requested_set
                for view in self.confidence.view_indices.detach().cpu().tolist()
            ],
            device=self.confidence.confidence.device,
            dtype=torch.bool,
        )
        support = (self.confidence.confidence > 0.0) & in_training[:, None]
        owner_count = support.sum(dim=0)
        multiplier = torch.where(
            owner_count > 0,
            owner_count.new_full(owner_count.shape, len(requested), dtype=torch.float32)
            / owner_count.clamp_min(1).to(dtype=torch.float32),
            owner_count.new_zeros(owner_count.shape, dtype=torch.float32),
        ).to(device=self.confidence.confidence.device, dtype=self.confidence.confidence.dtype)
        return support.to(dtype=self.confidence.confidence.dtype) * multiplier[None]

    def report(
        self,
        view_indices: list[int] | tuple[int, ...],
        *,
        mode: str = "raw_q95",
    ) -> dict:
        """Summarize how the trusted set covers a concrete training view list."""

        requested = [int(value) for value in view_indices]
        trusted = [value for value in requested if self.has_view(value)]
        untrusted = [value for value in requested if not self.has_view(value)]
        cache = self.cache_matrix(requested, mode=mode)
        cache_views = {
            int(view): cache[row]
            for row, view in enumerate(self.confidence.view_indices.tolist())
        }
        gates = [
            cache_views.get(
                value,
                self.confidence.confidence.new_zeros((self.guide_count,)),
            )
            for value in requested
        ]
        stacked = (
            torch.stack(gates, dim=0)
            if gates
            else self.confidence.confidence.new_zeros((0, self.guide_count))
        )
        owned = stacked > 0.0
        guide_support = owned.sum(dim=0) if owned.numel() else owned.new_zeros((self.guide_count,))
        positive = stacked[stacked > 0.0]
        supported = guide_support > 0
        per_guide_expected = stacked.mean(dim=0) if stacked.numel() else stacked.new_zeros((self.guide_count,))
        quantiles = (
            torch.quantile(
                positive,
                positive.new_tensor([0.50, 0.90, 0.95, 0.99, 1.0]),
            )
            if positive.numel()
            else positive.new_zeros((5,))
        )
        return {
            "normalization_mode": str(mode),
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
            "supported_guide_expected_multiplier_mean": (
                float(per_guide_expected[supported].mean())
                if bool(supported.any())
                else 0.0
            ),
            "positive_multiplier_p50_p90_p95_p99_max": [
                float(value) for value in quantiles.detach().cpu()
            ],
            "zero_owner_guide_count": int((~supported).sum()),
            "zero_owner_guide_fraction": float((~supported).float().mean()),
            "source_path": self.confidence.source_path,
            "summary_path": self.confidence.summary_path,
        }

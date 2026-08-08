from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    return torch.logit(value.clamp(eps, 1.0 - eps))


@dataclass(frozen=True)
class SecondaryGuideColors:
    root: torch.Tensor
    tip: torch.Tensor


class SecondaryGuideColorField(nn.Module):
    """Topology-controlled low-frequency root/tip color field."""

    def __init__(
        self,
        root_count: int,
        *,
        init_root_color: tuple[float, float, float] = (0.88, 0.88, 0.82),
        init_tip_color: tuple[float, float, float] = (0.98, 0.96, 0.88),
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        count = int(root_count)
        if count <= 0:
            raise ValueError("secondary guide color field requires at least one root")
        root = torch.tensor(init_root_color, device=device, dtype=torch.float32)
        tip = torch.tensor(init_tip_color, device=device, dtype=torch.float32)
        self.root_raw = nn.Parameter(_inverse_sigmoid(root).view(1, 3).repeat(count, 1))
        self.tip_raw = nn.Parameter(_inverse_sigmoid(tip).view(1, 3).repeat(count, 1))

    @property
    def root_count(self) -> int:
        return int(self.root_raw.shape[0])

    def decode(self) -> SecondaryGuideColors:
        return SecondaryGuideColors(
            root=torch.sigmoid(self.root_raw),
            tip=torch.sigmoid(self.tip_raw),
        )

    @torch.no_grad()
    def set_decoded(self, root: torch.Tensor, tip: torch.Tensor) -> None:
        expected = (self.root_count, 3)
        if tuple(root.shape) != expected or tuple(tip.shape) != expected:
            raise ValueError(
                "secondary guide color shape mismatch: "
                f"root={tuple(root.shape)}, tip={tuple(tip.shape)}, expected={expected}"
            )
        self.root_raw.copy_(_inverse_sigmoid(root.to(self.root_raw).clamp(0.0, 1.0)))
        self.tip_raw.copy_(_inverse_sigmoid(tip.to(self.tip_raw).clamp(0.0, 1.0)))

    @torch.no_grad()
    def stats(self) -> dict[str, float | int]:
        decoded = self.decode()
        return {
            "root_count": self.root_count,
            "root_mean": float(decoded.root.mean().cpu()),
            "root_std": float(decoded.root.std().cpu()),
            "tip_mean": float(decoded.tip.mean().cpu()),
            "tip_std": float(decoded.tip.std().cpu()),
        }

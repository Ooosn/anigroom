"""Low-frequency guide-root color controls for multilevel fur."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DecodedGuideColors:
    root: torch.Tensor
    tip: torch.Tensor


def encode_color(value: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(value.dtype).eps
    return torch.logit(value.clamp(eps, 1.0 - eps))


class GuideColorField(nn.Module):
    """Root/tip colors stored only on the sparse primary guide roots.

    Render roots receive these colors through the same surface interpolation
    support used by the geometric guide controls. This makes the base fur
    appearance low-capacity by construction; generated-Gaussian RGB residuals
    remain the only high-frequency appearance outlet.
    """

    def __init__(
        self,
        root_count: int,
        *,
        root_color: tuple[float, float, float] = (0.88, 0.88, 0.82),
        tip_color: tuple[float, float, float] = (0.98, 0.96, 0.88),
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        root_count = int(root_count)
        if root_count <= 0:
            raise ValueError("guide color field requires at least one root")
        root = torch.tensor(root_color, dtype=torch.float32, device=device).view(1, 3)
        tip = torch.tensor(tip_color, dtype=torch.float32, device=device).view(1, 3)
        self.root_raw = nn.Parameter(encode_color(root).expand(root_count, -1).clone())
        self.tip_raw = nn.Parameter(encode_color(tip).expand(root_count, -1).clone())

    @property
    def root_count(self) -> int:
        return int(self.root_raw.shape[0])

    def decode(self) -> DecodedGuideColors:
        return DecodedGuideColors(
            root=torch.sigmoid(self.root_raw),
            tip=torch.sigmoid(self.tip_raw),
        )

    @torch.no_grad()
    def set_decoded(self, root: torch.Tensor, tip: torch.Tensor) -> None:
        if root.shape != self.root_raw.shape or tip.shape != self.tip_raw.shape:
            raise ValueError("guide color shape mismatch")
        self.root_raw.copy_(encode_color(root.to(self.root_raw)))
        self.tip_raw.copy_(encode_color(tip.to(self.tip_raw)))

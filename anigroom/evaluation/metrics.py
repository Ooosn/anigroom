"""Shared image metrics for AniGroom experiments."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def image_to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"expected NHWC image tensor, got shape {tuple(image.shape)}")
    return image.permute(0, 3, 1, 2).contiguous()


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    pred_bchw = image_to_bchw(pred).clamp(0.0, 1.0)
    target_bchw = image_to_bchw(target).clamp(0.0, 1.0)
    channels = pred_bchw.shape[1]
    window = gaussian_window(window_size, sigma, channels, pred_bchw.device)
    padding = window_size // 2

    mu_pred = F.conv2d(pred_bchw, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target_bchw, window, padding=padding, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred_bchw * pred_bchw, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target_bchw * target_bchw, window, padding=padding, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(pred_bchw * target_bchw, window, padding=padding, groups=channels) - mu_cross

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    )
    return ssim_map.mean()


@dataclass
class MetricComputer:
    compute_lpips: bool = False
    lpips_net: str = "alex"

    def __post_init__(self) -> None:
        self.lpips_available = False
        self._lpips_model = None
        if self.compute_lpips:
            if importlib.util.find_spec("lpips") is None:
                return
            import lpips  # type: ignore[import-not-found]

            self._lpips_model = lpips.LPIPS(net=self.lpips_net).eval()
            self.lpips_available = True

    def to(self, device: torch.device) -> "MetricComputer":
        if self._lpips_model is not None:
            self._lpips_model = self._lpips_model.to(device)
        return self

    @torch.no_grad()
    def image_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor | None]:
        pred = pred.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        mse = torch.mean((pred - target) ** 2)
        out: dict[str, torch.Tensor | None] = {
            "mse": mse,
            "psnr": psnr_from_mse(mse),
            "ssim": ssim(pred, target),
            "lpips": None,
        }
        if self._lpips_model is not None:
            pred_bchw = image_to_bchw(pred) * 2.0 - 1.0
            target_bchw = image_to_bchw(target) * 2.0 - 1.0
            out["lpips"] = self._lpips_model(pred_bchw, target_bchw).mean()
        return out


"""Windowed root statistics for formal densification/pruning.

The accumulator consumes real gsplat render outputs after ``loss.backward()``.
It does not invent visibility or substitute missing gradients.  If gsplat does
not provide the required fields, callers get a hard error so the training path
can be fixed instead of silently changing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .lifecycle import RootStats


EPS = 1e-8


@dataclass(frozen=True)
class RootStatsSummary:
    iterations: int
    root_count: int
    gaussian_grad_mean: float
    root_grad_mean: float
    contribution_mean: float
    visible_mean: float
    opacity_mean: float


class RootStatsWindow:
    """Accumulate root evidence over a densification/pruning window."""

    def __init__(self, root_count: int, device: torch.device | str) -> None:
        if root_count <= 0:
            raise ValueError("root_count must be positive")
        self.root_count = int(root_count)
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        shape = (self.root_count, 1)
        self.root_grad_abs_sum = torch.zeros(shape, device=self.device)
        self.gaussian_grad_abs_sum = torch.zeros(shape, device=self.device)
        self.gaussian_contrib_sum = torch.zeros(shape, device=self.device)
        self.visible_count = torch.zeros(shape, device=self.device)
        self.opacity_sum = torch.zeros(shape, device=self.device)
        self.opacity_count = torch.zeros(shape, device=self.device)
        self.residual_sum = torch.zeros(shape, device=self.device)
        self.iterations = 0

    def _validate_root_points(self, root_points: torch.Tensor) -> None:
        if root_points.shape != (self.root_count, 3):
            raise ValueError(f"root_points must have shape [{self.root_count}, 3], got {tuple(root_points.shape)}")
        if root_points.grad is None:
            raise RuntimeError("root_points.grad is missing; call retain_grad() before backward")

    @staticmethod
    def _visible_from_info(info: dict, gaussian_count: int, device: torch.device) -> torch.Tensor:
        if not isinstance(info, dict) or "radii" not in info:
            raise RuntimeError("gsplat info must contain radii for formal visibility accumulation")
        radii = info["radii"].detach().to(device=device)
        if radii.ndim == 2:
            if radii.shape[0] != 1:
                raise RuntimeError(f"expected one camera in radii, got shape {tuple(radii.shape)}")
            radii = radii[0]
        else:
            radii = radii.reshape(-1)
        if radii.numel() != gaussian_count:
            raise RuntimeError(f"radii count {radii.numel()} does not match gaussian count {gaussian_count}")
        return (radii > 0).float().reshape(-1, 1)

    def add(
        self,
        *,
        root_points: torch.Tensor,
        gaussians,
        infos: list[dict],
        residual_loss: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one backward pass.

        ``gaussians`` must be produced from the current ``root_points`` and must
        expose ``root_indices``, ``means``, ``scales``, and ``opacities``.
        """

        self._validate_root_points(root_points)
        if not infos:
            raise RuntimeError("at least one gsplat info dict is required")
        if not hasattr(gaussians, "root_indices"):
            raise RuntimeError("gaussian output is missing root_indices")
        root_ids = gaussians.root_indices.long().reshape(-1)
        gaussian_count = int(root_ids.numel())
        if gaussian_count <= 0:
            raise RuntimeError("no Gaussians were produced for this render")
        if int(root_ids.min().item()) < 0 or int(root_ids.max().item()) >= self.root_count:
            raise RuntimeError("gaussian root_indices are out of range")
        if gaussians.means.grad is None:
            raise RuntimeError("gaussian means gradient is missing")
        if gaussians.scales.grad is None:
            raise RuntimeError("gaussian scales gradient is missing")

        mean_grad = gaussians.means.grad.detach().abs().sum(dim=-1, keepdim=True)
        scale_grad = gaussians.scales.grad.detach().abs().sum(dim=-1, keepdim=True)
        if mean_grad.shape[0] != gaussian_count or scale_grad.shape[0] != gaussian_count:
            raise RuntimeError("gaussian gradient shape does not match root_indices")
        gaussian_grad = mean_grad + 0.25 * scale_grad

        visible_g = torch.zeros((gaussian_count, 1), device=self.device)
        for info in infos:
            visible_g += self._visible_from_info(info, gaussian_count, self.device)

        opacities = gaussians.opacities.detach().reshape(-1, 1).to(device=self.device)
        if opacities.shape[0] != gaussian_count:
            raise RuntimeError("gaussian opacities shape does not match root_indices")

        self.root_grad_abs_sum += root_points.grad.detach().abs().sum(dim=-1, keepdim=True)
        self.gaussian_grad_abs_sum.scatter_add_(0, root_ids[:, None], gaussian_grad.to(device=self.device))
        self.gaussian_contrib_sum.scatter_add_(0, root_ids[:, None], visible_g * opacities.clamp_min(EPS))
        self.visible_count.scatter_add_(0, root_ids[:, None], visible_g)
        self.opacity_sum.scatter_add_(0, root_ids[:, None], opacities)
        self.opacity_count.scatter_add_(0, root_ids[:, None], torch.ones_like(opacities))
        if residual_loss is not None:
            self.residual_sum += float(residual_loss.detach().cpu())
        self.iterations += 1

    def to_stats(self) -> RootStats:
        if self.iterations <= 0:
            raise RuntimeError("cannot export RootStats from an empty window")
        opacity_mean = self.opacity_sum / self.opacity_count.clamp_min(1.0)
        return RootStats(
            root_grad_abs_sum=self.root_grad_abs_sum.clone(),
            gaussian_grad_abs_sum=self.gaussian_grad_abs_sum.clone(),
            gaussian_contrib_sum=self.gaussian_contrib_sum.clone(),
            visible_count=self.visible_count.clone(),
            residual_sum=self.residual_sum.clone(),
            opacity_mean=opacity_mean.clone(),
        )

    def summary(self) -> RootStatsSummary:
        stats = self.to_stats()
        opacity_mean = stats.opacity_mean
        if opacity_mean is None:
            opacity_value = 0.0
        else:
            opacity_value = float(opacity_mean.mean().detach().cpu())
        return RootStatsSummary(
            iterations=int(self.iterations),
            root_count=int(self.root_count),
            gaussian_grad_mean=float(stats.gaussian_grad_abs_sum.mean().detach().cpu()),
            root_grad_mean=float(stats.root_grad_abs_sum.mean().detach().cpu()),
            contribution_mean=float(stats.gaussian_contrib_sum.mean().detach().cpu()),
            visible_mean=float(stats.visible_count.mean().detach().cpu()),
            opacity_mean=opacity_value,
        )

"""Primary-guide-owned view-dependent RGB using degree-one spherical harmonics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


FIRST_ORDER_SH_CONSTANT = 0.4886025119029199


@dataclass(frozen=True)
class TrustedGuideViewConfidence:
    """Normalized trusted multiview evidence aligned to primary guide roots."""

    view_indices: torch.Tensor
    confidence: torch.Tensor
    positive_q95: float
    source_path: str
    summary_path: str

    @property
    def view_count(self) -> int:
        return int(self.view_indices.shape[0])

    @property
    def guide_count(self) -> int:
        return int(self.confidence.shape[1])

    def confidence_for_view(self, view_index: int) -> torch.Tensor:
        matches = torch.nonzero(
            self.view_indices == int(view_index),
            as_tuple=False,
        ).reshape(-1)
        if matches.numel() == 0:
            return self.confidence.new_zeros((self.guide_count,))
        if matches.numel() != 1:
            raise RuntimeError(f"duplicate trusted view index: {int(view_index)}")
        return self.confidence[int(matches[0])]

    def report(self) -> dict[str, float | int | str | list[int]]:
        nonzero = self.confidence > 0.0
        guide_support = nonzero.sum(dim=0)
        return {
            "source_path": self.source_path,
            "summary_path": self.summary_path,
            "view_indices": [int(value) for value in self.view_indices.detach().cpu()],
            "view_count": self.view_count,
            "guide_count": self.guide_count,
            "positive_q95": float(self.positive_q95),
            "nonzero_fraction": float(nonzero.float().mean().detach().cpu()),
            "supported_guide_fraction": float((guide_support > 0).float().mean().detach().cpu()),
            "views_per_guide_mean": float(guide_support.float().mean().detach().cpu()),
            "confidence_mean": float(self.confidence.mean().detach().cpu()),
            "confidence_max": float(self.confidence.max().detach().cpu()),
        }


def load_trusted_guide_view_confidence(
    clean_flow_target_path: str | Path,
    *,
    summary_path: str | Path | None = None,
    expected_face_ids: np.ndarray | torch.Tensor | None = None,
    expected_barycentric: np.ndarray | torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> TrustedGuideViewConfidence:
    """Load V7 trusted-view weights and their explicit view-index mapping."""

    source = Path(clean_flow_target_path)
    summary = Path(summary_path) if summary_path is not None else source.parent / "summary.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    if not summary.is_file():
        raise FileNotFoundError(summary)

    with np.load(source, allow_pickle=False) as data:
        key = "axis_view_cluster_selected_direct_weight"
        if key not in data:
            raise RuntimeError(f"clean-flow target is missing {key}: {source}")
        raw_weight = np.asarray(data[key], dtype=np.float32)
        if raw_weight.ndim != 2:
            raise RuntimeError(
                f"{key} must have shape [V, G], got {raw_weight.shape}"
            )
        if not np.isfinite(raw_weight).all() or np.any(raw_weight < 0.0):
            raise RuntimeError(f"{key} must be finite and non-negative")

        if expected_face_ids is not None:
            if "face_ids" not in data:
                raise RuntimeError("clean-flow target is missing face_ids")
            expected = np.asarray(
                expected_face_ids.detach().cpu().numpy()
                if torch.is_tensor(expected_face_ids)
                else expected_face_ids,
                dtype=np.int64,
            )
            actual = np.asarray(data["face_ids"], dtype=np.int64)
            if not np.array_equal(actual, expected):
                raise RuntimeError("guide face_ids do not match the trusted-view target")

        if expected_barycentric is not None:
            if "barycentric" not in data:
                raise RuntimeError("clean-flow target is missing barycentric")
            expected = np.asarray(
                expected_barycentric.detach().cpu().numpy()
                if torch.is_tensor(expected_barycentric)
                else expected_barycentric,
                dtype=np.float32,
            )
            actual = np.asarray(data["barycentric"], dtype=np.float32)
            if not np.array_equal(actual, expected):
                raise RuntimeError(
                    "guide barycentric coordinates do not match the trusted-view target"
                )

    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    raw_views = summary_payload.get("views_used")
    if not isinstance(raw_views, list):
        raise RuntimeError(f"summary views_used must be a list: {summary}")
    try:
        view_indices_np = np.asarray([int(value) for value in raw_views], dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise RuntimeError("summary views_used must contain integer view IDs") from error
    if view_indices_np.ndim != 1 or view_indices_np.shape[0] != raw_weight.shape[0]:
        raise RuntimeError(
            "summary views_used length does not match trusted-view weights: "
            f"{view_indices_np.shape[0]} != {raw_weight.shape[0]}"
        )
    if np.unique(view_indices_np).shape[0] != view_indices_np.shape[0]:
        raise RuntimeError("summary views_used contains duplicate view IDs")

    positive = raw_weight[raw_weight > 0.0]
    if positive.size == 0:
        raise RuntimeError("trusted-view weights contain no positive evidence")
    positive_q95 = float(np.quantile(positive.astype(np.float64), 0.95))
    if not np.isfinite(positive_q95) or positive_q95 <= 0.0:
        raise RuntimeError("trusted-view positive q95 must be finite and positive")
    confidence_np = np.clip(raw_weight / positive_q95, 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )
    dev = torch.device(device) if device is not None else None
    return TrustedGuideViewConfidence(
        view_indices=torch.from_numpy(view_indices_np).to(device=dev),
        confidence=torch.from_numpy(confidence_np).to(device=dev),
        positive_q95=positive_q95,
        source_path=str(source),
        summary_path=str(summary),
    )


def first_order_sh_basis(local_view_directions: torch.Tensor) -> torch.Tensor:
    """Return the three real l=1 basis values without a DC component."""

    if local_view_directions.ndim != 2 or local_view_directions.shape[-1] != 3:
        raise ValueError("local_view_directions must have shape [G, 3]")
    if not bool(torch.isfinite(local_view_directions).all()):
        raise ValueError("local_view_directions must be finite")
    normalized = F.normalize(local_view_directions, dim=-1, eps=1.0e-8)
    return normalized * FIRST_ORDER_SH_CONSTANT


class GuideViewSHField(nn.Module):
    """One bounded non-DC degree-one RGB SH residual per primary guide."""

    def __init__(
        self,
        guide_count: int,
        scale: float,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if int(guide_count) <= 0:
            raise ValueError("guide_count must be positive")
        if float(scale) <= 0.0:
            raise ValueError("scale must be positive")
        self.guide_count = int(guide_count)
        self.scale = float(scale)
        self.raw = nn.Parameter(
            torch.zeros((self.guide_count, 3, 3), dtype=torch.float32, device=device)
        )

    def decoded_coefficients(self) -> torch.Tensor:
        return torch.tanh(self.raw) * self.scale

    def residual(
        self,
        local_view_directions: torch.Tensor,
        *,
        gradient_confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local_view_directions.shape != (self.guide_count, 3):
            raise ValueError(
                "local_view_directions must have one direction per guide: "
                f"{tuple(local_view_directions.shape)} != {(self.guide_count, 3)}"
            )
        coefficients = self.decoded_coefficients()
        if gradient_confidence is not None:
            confidence = gradient_confidence.to(
                device=coefficients.device,
                dtype=coefficients.dtype,
            ).reshape(-1)
            if confidence.shape != (self.guide_count,):
                raise ValueError("gradient_confidence must have shape [G]")
            if not bool(torch.isfinite(confidence).all()):
                raise ValueError("gradient_confidence must be finite")
            confidence = confidence.clamp(0.0, 1.0).view(-1, 1, 1)
            coefficients = coefficients.detach() + confidence * (
                coefficients - coefficients.detach()
            )
        basis = first_order_sh_basis(local_view_directions).to(
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        return torch.einsum("gbc,gb->gc", coefficients, basis)

    @torch.no_grad()
    def stats(self, *, guide_chunk_size: int = 4096) -> dict[str, float | int]:
        guide_chunk_size = max(1, int(guide_chunk_size))
        element_count = 0
        absolute_sum = 0.0
        square_sum = 0.0
        absolute_max = 0.0
        active_count = 0
        saturated_count = 0
        for start in range(0, self.guide_count, guide_chunk_size):
            normalized = torch.tanh(self.raw[start : start + guide_chunk_size])
            decoded = normalized * self.scale
            absolute = decoded.abs()
            element_count += int(decoded.numel())
            absolute_sum += float(absolute.sum().cpu())
            square_sum += float(decoded.square().sum().cpu())
            absolute_max = max(absolute_max, float(absolute.max().cpu()))
            active_count += int((absolute > 1.0e-4).sum().cpu())
            saturated_count += int((normalized.abs() > 0.95).sum().cpu())
        return {
            "guide_count": self.guide_count,
            "degree": 1,
            "coefficient_count": self.guide_count * 9,
            "scale": self.scale,
            "abs_mean": absolute_sum / float(element_count),
            "rms": (square_sum / float(element_count)) ** 0.5,
            "abs_max": absolute_max,
            "active_fraction": active_count / float(element_count),
            "saturation_fraction": saturated_count / float(element_count),
        }

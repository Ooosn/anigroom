"""Differentiable guide-attribute anisotropic Gaussian field."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

from .config import GuideGaussianFieldConfig
from .initialization import GuideGaussianBinding


def c2_gaussian_taper(
    rho: torch.Tensor | float,
    start: float = 2.5,
    end: float = 3.0,
) -> torch.Tensor:
    """Return a quintic C2 taper that is one then smoothly reaches zero."""

    if not math.isfinite(float(start)) or not math.isfinite(float(end)):
        raise ValueError("taper boundaries must be finite")
    if not 0.0 <= float(start) < float(end):
        raise ValueError("taper boundaries must satisfy 0 <= start < end")
    if isinstance(rho, torch.Tensor):
        if rho.is_complex() or not torch.is_floating_point(rho):
            raise TypeError("rho must be a real floating-point tensor")
        values = rho
    else:
        values = torch.as_tensor(float(rho), dtype=torch.float32)
    start_value = torch.as_tensor(start, device=values.device, dtype=values.dtype)
    end_value = torch.as_tensor(end, device=values.device, dtype=values.dtype)
    u = ((values - start_value) / (end_value - start_value)).clamp(0.0, 1.0)
    # This factored form is algebraically identical to
    # ``1 - 10u^3 + 15u^4 - 6u^5`` but avoids cancellation-driven negative
    # float32 values near ``u=1``.
    smooth_near_start = 1.0 - u.pow(3) * (
        10.0 - 15.0 * u + 6.0 * u.square()
    )
    one_minus_u = 1.0 - u
    smooth_near_end = one_minus_u.pow(3) * (
        1.0 + 3.0 * u + 6.0 * u.square()
    )
    smooth = torch.where(u <= 0.5, smooth_near_start, smooth_near_end)
    return torch.where(
        values <= start_value,
        torch.ones_like(values),
        torch.where(values >= end_value, torch.zeros_like(values), smooth),
    )


@dataclass(frozen=True)
class GuideGaussianWeights:
    """Per-candidate Gaussian weights and per-query normalization."""

    raw: torch.Tensor
    normalized: torch.Tensor
    denominator: torch.Tensor
    active_pair_count: int
    query_ids: torch.Tensor
    guide_ids: torch.Tensor
    candidate_pair_count: int
    query_count: int


class GuideAttributeGaussianField(nn.Module):
    """A fixed-candidate, differentiable guide-to-query Gaussian field.

    Quaternions use the ``[w, x, y, z]`` convention and map Gaussian-local
    vectors into world coordinates.  The only trainable quantities are the
    three log-scale coordinates and one quaternion per guide.
    """

    def __init__(
        self,
        binding: GuideGaussianBinding,
        config: GuideGaussianFieldConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(binding, GuideGaussianBinding):
            raise TypeError("binding must be GuideGaussianBinding")
        if config is None:
            config = binding.config
        if config is None:
            config = GuideGaussianFieldConfig()
        if not isinstance(config, GuideGaussianFieldConfig):
            raise TypeError("config must be GuideGaussianFieldConfig")
        if binding.config is not None and config != binding.config:
            raise ValueError(
                "field config must exactly match the config used to build the binding"
            )
        self.config = config

        self._validate_binding(binding)
        self.register_buffer("guide_points", binding.guide_points.detach().clone())
        self.register_buffer("query_points", binding.query_points.detach().clone())
        self.register_buffer(
            "reference_sigma",
            binding.reference_sigma.detach().clone(),
        )
        self.register_buffer("row_ptr", binding.row_ptr.detach().clone().long())
        self.register_buffer("guide_ids", binding.guide_ids.detach().clone().long())
        self.register_buffer("query_ids", binding.query_ids.detach().clone().long())
        self._binding_report = dict(binding.report)

        guide_count = int(binding.guide_points.shape[0])
        parameter_dtype = binding.guide_points.dtype
        parameter_device = binding.guide_points.device
        self.raw_scale_coordinate = nn.Parameter(
            torch.zeros(
                (guide_count, 3),
                device=parameter_device,
                dtype=parameter_dtype,
            )
        )
        initial_quaternion = torch.zeros(
            (guide_count, 4),
            device=parameter_device,
            dtype=parameter_dtype,
        )
        initial_quaternion[:, 0] = 1.0
        self.raw_quaternion = nn.Parameter(initial_quaternion)

    @staticmethod
    def _validate_binding(binding: GuideGaussianBinding) -> None:
        floating = (torch.float16, torch.float32, torch.float64)
        if binding.guide_points.dtype not in floating:
            raise TypeError("binding guide_points must use a real floating dtype")
        if binding.query_points.dtype != binding.guide_points.dtype:
            raise ValueError("binding guide/query dtype mismatch")
        if binding.query_points.device != binding.guide_points.device:
            raise ValueError("binding guide/query device mismatch")
        guide_count = int(binding.guide_points.shape[0])
        query_count = int(binding.query_points.shape[0])
        if tuple(binding.guide_points.shape) != (guide_count, 3):
            raise ValueError("binding guide_points must have shape [G, 3]")
        if tuple(binding.query_points.shape) != (query_count, 3):
            raise ValueError("binding query_points must have shape [R, 3]")
        if tuple(binding.reference_sigma.shape) != (guide_count,):
            raise ValueError("binding reference_sigma must have shape [G]")
        if binding.reference_sigma.device != binding.guide_points.device:
            raise ValueError("binding reference_sigma device mismatch")
        if tuple(binding.row_ptr.shape) != (query_count + 1,):
            raise ValueError("binding row_ptr must have shape [R + 1]")
        if binding.row_ptr.dtype != torch.long:
            raise TypeError("binding row_ptr must use torch.long")
        if binding.guide_ids.dtype != torch.long or binding.query_ids.dtype != torch.long:
            raise TypeError("binding candidate IDs must use torch.long")
        if binding.guide_ids.device != binding.guide_points.device:
            raise ValueError("binding guide_ids device mismatch")
        if binding.query_ids.device != binding.guide_points.device:
            raise ValueError("binding query_ids device mismatch")
        pair_count = int(binding.guide_ids.numel())
        if tuple(binding.guide_ids.shape) != (pair_count,) or tuple(
            binding.query_ids.shape
        ) != (pair_count,):
            raise ValueError("binding candidate IDs must be one-dimensional")
        if pair_count <= 0:
            raise ValueError("binding candidate pattern must be non-empty")
        if not bool(torch.isfinite(binding.guide_points).all()):
            raise ValueError("binding guide_points contain non-finite values")
        if not bool(torch.isfinite(binding.query_points).all()):
            raise ValueError("binding query_points contain non-finite values")
        if not bool(torch.isfinite(binding.reference_sigma).all()) or bool(
            (binding.reference_sigma <= 0.0).any()
        ):
            raise ValueError("binding reference_sigma must be finite and positive")
        if int(binding.row_ptr[0].item()) != 0 or int(binding.row_ptr[-1].item()) != pair_count:
            raise ValueError("binding row_ptr endpoints do not cover candidate pairs")
        if bool((binding.row_ptr[1:] < binding.row_ptr[:-1]).any()):
            raise ValueError("binding row_ptr must be nondecreasing")
        if bool((binding.query_ids < 0).any()) or bool(
            (binding.query_ids >= query_count).any()
        ):
            raise ValueError("binding query_ids are out of range")
        if bool((binding.guide_ids < 0).any()) or bool(
            (binding.guide_ids >= guide_count).any()
        ):
            raise ValueError("binding guide_ids are out of range")

        expected_query_ids = torch.repeat_interleave(
            torch.arange(
                query_count,
                device=binding.guide_points.device,
                dtype=torch.long,
            ),
            binding.row_ptr[1:] - binding.row_ptr[:-1],
        )
        if not torch.equal(expected_query_ids, binding.query_ids):
            raise ValueError("binding query_ids do not match CSR row_ptr")

    @property
    def guide_count(self) -> int:
        return int(self.guide_points.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.query_points.shape[0])

    @property
    def candidate_pair_count(self) -> int:
        return int(self.guide_ids.numel())

    @property
    def binding_report(self) -> dict[str, Any]:
        return dict(self._binding_report)

    def _check_finite(self, values: torch.Tensor, *, name: str) -> None:
        if not bool(torch.isfinite(values).all().detach().cpu()):
            raise ValueError(f"{name} contains non-finite values")

    def _normalized_quaternion(self) -> torch.Tensor:
        self._check_finite(self.raw_quaternion, name="raw_quaternion")
        norm = torch.linalg.vector_norm(self.raw_quaternion, dim=-1, keepdim=True)
        if not bool(torch.isfinite(norm).all().detach().cpu()) or bool(
            (norm <= torch.finfo(norm.dtype).eps).any().detach().cpu()
        ):
            raise ValueError("raw_quaternion must have finite nonzero norms")
        return self.raw_quaternion / norm

    def _log_scale_ratio(self) -> torch.Tensor:
        self._check_finite(self.raw_scale_coordinate, name="raw_scale_coordinate")
        lower = math.log(float(self.config.min_scale_ratio))
        upper = math.log(float(self.config.max_scale_ratio))
        coordinate = torch.tanh(self.raw_scale_coordinate)
        lower_span = torch.as_tensor(
            -lower,
            device=coordinate.device,
            dtype=coordinate.dtype,
        )
        upper_span = torch.as_tensor(
            upper,
            device=coordinate.device,
            dtype=coordinate.dtype,
        )
        log_ratio = torch.where(
            coordinate >= 0.0,
            upper_span * coordinate,
            lower_span * coordinate,
        )
        self._check_finite(log_ratio, name="decoded log scale ratio")
        if bool((log_ratio < lower).any().detach().cpu()) or bool(
            (log_ratio > upper).any().detach().cpu()
        ):
            raise RuntimeError("decoded scale ratio escaped configured log bounds")
        return log_ratio

    def decoded_scales(self) -> torch.Tensor:
        """Decode positive per-axis scales from bounded log coordinates."""

        scales = self.reference_sigma[:, None] * torch.exp(self._log_scale_ratio())
        self._check_finite(scales, name="decoded scales")
        if bool((scales <= 0.0).any().detach().cpu()):
            raise RuntimeError("decoded scales must be strictly positive")
        return scales

    def rotation_matrices(self) -> torch.Tensor:
        """Return normalized-quaternion local-to-world rotation matrices."""

        quaternion = self._normalized_quaternion()
        w, x, y, z = quaternion.unbind(dim=-1)
        matrix = torch.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(-1, 3, 3)
        self._check_finite(matrix, name="rotation matrices")
        return matrix

    def covariance_matrices(self) -> torch.Tensor:
        """Return ``R diag(scale**2) R^T`` for every guide."""

        scales = self.decoded_scales()
        rotations = self.rotation_matrices()
        covariance = rotations @ torch.diag_embed(scales.square()) @ rotations.transpose(
            -1,
            -2,
        )
        self._check_finite(covariance, name="covariance matrices")
        return covariance

    @staticmethod
    def _rotate_world_to_local(
        vector_world: torch.Tensor,
        quaternion_local_to_world: torch.Tensor,
    ) -> torch.Tensor:
        quaternion_conjugate = torch.cat(
            (
                quaternion_local_to_world[..., :1],
                -quaternion_local_to_world[..., 1:],
            ),
            dim=-1,
        )
        vector_part = quaternion_conjugate[..., 1:]
        twice_cross = 2.0 * torch.cross(vector_part, vector_world, dim=-1)
        return vector_world + quaternion_conjugate[..., :1] * twice_cross + torch.cross(
            vector_part,
            twice_cross,
            dim=-1,
        )

    def _prepare_query_points(self, query_points: torch.Tensor | None) -> torch.Tensor:
        if query_points is None:
            return self.query_points
        if not isinstance(query_points, torch.Tensor):
            raise TypeError("query_points must be a torch.Tensor or None")
        if query_points.ndim != 2 or tuple(query_points.shape) != (
            self.query_count,
            3,
        ):
            raise ValueError(
                f"query_points must have shape [{self.query_count}, 3]"
            )
        if query_points.is_complex() or not torch.is_floating_point(query_points):
            raise TypeError("query_points must be a real floating-point tensor")
        prepared = query_points.to(
            device=self.guide_points.device,
            dtype=self.guide_points.dtype,
        )
        self._check_finite(prepared, name="query_points")
        return prepared

    def evaluate_weights(
        self,
        query_points: torch.Tensor | None = None,
    ) -> GuideGaussianWeights:
        """Evaluate dynamic weights over the fixed candidate pair pattern."""

        query = self._prepare_query_points(query_points)
        quaternion = self._normalized_quaternion()
        scales = self.decoded_scales()
        query_delta_world = query[self.query_ids] - self.guide_points[self.guide_ids]
        query_delta_local = self._rotate_world_to_local(
            query_delta_world,
            quaternion[self.guide_ids],
        )
        normalized_delta = query_delta_local / scales[self.guide_ids]
        rho_squared = normalized_delta.square().sum(dim=-1)
        rho = torch.sqrt(
            rho_squared.clamp_min(torch.finfo(rho_squared.dtype).eps)
        )
        self._check_finite(rho, name="Gaussian Mahalanobis radius")
        raw = torch.exp(-0.5 * rho_squared) * c2_gaussian_taper(
            rho,
            start=float(self.config.taper_start_sigma),
            end=float(self.config.support_sigma),
        )
        self._check_finite(raw, name="raw Gaussian weights")
        if bool((raw < 0.0).any().detach().cpu()):
            raise RuntimeError("raw Gaussian weights must be nonnegative")

        denominator = torch.zeros(
            (self.query_count,),
            device=raw.device,
            dtype=raw.dtype,
        )
        denominator.index_add_(0, self.query_ids, raw)
        self._check_finite(denominator, name="Gaussian weight denominators")
        if bool(
            (denominator <= float(self.config.min_denominator)).any().detach().cpu()
        ):
            bad = torch.nonzero(
                denominator <= float(self.config.min_denominator),
                as_tuple=False,
            ).reshape(-1)
            raise RuntimeError(
                "Gaussian weight denominator is at or below min_denominator for "
                f"queries {bad[:8].detach().cpu().tolist()}"
            )
        normalized = raw / denominator[self.query_ids]
        self._check_finite(normalized, name="normalized Gaussian weights")
        if bool((normalized < 0.0).any().detach().cpu()):
            raise RuntimeError("normalized Gaussian weights must be nonnegative")
        active_pair_count = int((raw > 0.0).sum().detach().cpu())
        return GuideGaussianWeights(
            raw=raw,
            normalized=normalized,
            denominator=denominator,
            active_pair_count=active_pair_count,
            query_ids=self.query_ids,
            guide_ids=self.guide_ids,
            candidate_pair_count=self.candidate_pair_count,
            query_count=self.query_count,
        )

    def forward(
        self,
        guide_values: torch.Tensor,
        query_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Interpolate ``[G]`` or ``[G, *]`` guide attributes once by index-add."""

        if not isinstance(guide_values, torch.Tensor):
            raise TypeError("guide_values must be a torch.Tensor")
        if guide_values.ndim < 1 or int(guide_values.shape[0]) != self.guide_count:
            raise ValueError(
                f"guide_values must have first dimension {self.guide_count}"
            )
        if guide_values.is_complex() or not torch.is_floating_point(guide_values):
            raise TypeError("guide_values must be a real floating-point tensor")
        if guide_values.device != self.guide_points.device:
            raise ValueError("guide_values must be on the field device")
        self._check_finite(guide_values, name="guide_values")

        weights = self.evaluate_weights(query_points=query_points)
        gathered = guide_values[self.guide_ids]
        reshape = (self.candidate_pair_count,) + (1,) * (guide_values.ndim - 1)
        contribution = gathered * weights.normalized.reshape(reshape)
        output = contribution.new_zeros((self.query_count,) + tuple(guide_values.shape[1:]))
        output.index_add_(0, self.query_ids, contribution)
        self._check_finite(output, name="interpolated guide values")
        return output

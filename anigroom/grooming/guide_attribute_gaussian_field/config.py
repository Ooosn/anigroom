"""Configuration for the isolated guide-attribute Gaussian field."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class GuideGaussianFieldConfig:
    """Immutable numerical contract for guide Gaussian interpolation."""

    neighbor_count: int = 16
    support_sigma: float = 3.0
    taper_start_sigma: float = 2.5
    min_scale_ratio: float = 2.0 / 3.0
    max_scale_ratio: float = 1.5
    min_denominator: float = 1.0e-8

    def __post_init__(self) -> None:
        if isinstance(self.neighbor_count, bool) or not isinstance(
            self.neighbor_count, Integral
        ):
            raise TypeError("neighbor_count must be an integer")
        if int(self.neighbor_count) < 1:
            raise ValueError("neighbor_count must be at least one")

        for name in (
            "support_sigma",
            "taper_start_sigma",
            "min_scale_ratio",
            "max_scale_ratio",
            "min_denominator",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if float(self.support_sigma) <= 0.0:
            raise ValueError("support_sigma must be positive")
        if not (
            0.0 < float(self.taper_start_sigma) < float(self.support_sigma)
        ):
            raise ValueError(
                "taper_start_sigma must satisfy 0 < taper_start_sigma < support_sigma"
            )
        if not (
            0.0 < float(self.min_scale_ratio)
            <= 1.0
            <= float(self.max_scale_ratio)
        ):
            raise ValueError(
                "scale ratios must satisfy 0 < min_scale_ratio <= 1 <= max_scale_ratio"
            )
        if float(self.min_denominator) <= 0.0:
            raise ValueError("min_denominator must be positive")

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation without dataclass objects."""

        values = asdict(self)
        values["neighbor_count"] = int(values["neighbor_count"])
        for key in (
            "support_sigma",
            "taper_start_sigma",
            "min_scale_ratio",
            "max_scale_ratio",
            "min_denominator",
        ):
            values[key] = float(values[key])
        return values

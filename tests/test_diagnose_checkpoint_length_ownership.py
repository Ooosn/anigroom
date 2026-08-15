from __future__ import annotations

import torch

from tools.diagnose_checkpoint_length_ownership import (
    compose_effective_length,
    residual_log_ratio,
)


def test_zero_residual_reproduces_primary_length_exactly() -> None:
    primary = torch.tensor([[0.02], [0.07]], dtype=torch.float64)
    residual = torch.zeros_like(primary)

    effective, log_ratio, multiplier = compose_effective_length(
        primary,
        residual,
        1.0,
    )

    torch.testing.assert_close(log_ratio, torch.zeros_like(primary))
    torch.testing.assert_close(multiplier, torch.ones_like(primary))
    torch.testing.assert_close(effective, primary)


def test_residual_is_relative_positive_and_unbounded() -> None:
    primary = torch.tensor([[0.01], [0.10], [0.01]], dtype=torch.float64)
    residual = torch.tensor([[2.0], [2.0], [-2.0]], dtype=torch.float64)

    effective, _, multiplier = compose_effective_length(primary, residual, 1.0)

    torch.testing.assert_close(multiplier[0], multiplier[1])
    torch.testing.assert_close(
        effective[1] / effective[0],
        torch.tensor([10.0], dtype=primary.dtype),
    )
    assert float(effective[0]) > float(primary[0])
    assert 0.0 < float(effective[2]) < float(primary[2])


def test_schedule_scale_acts_in_physical_log_ratio_space() -> None:
    residual = torch.tensor([[-3.0], [0.0], [3.0]], dtype=torch.float64)

    full = residual_log_ratio(residual, 1.0)
    quarter = residual_log_ratio(residual, 0.25)

    torch.testing.assert_close(quarter, full * 0.25)

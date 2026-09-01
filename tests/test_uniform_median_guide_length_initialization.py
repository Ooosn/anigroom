from __future__ import annotations

import pytest
import torch

from tools.train_white_tiger_stage1 import (
    initialize_uniform_median_guide_length,
)


def test_uniform_reference_uses_masked_even_quantile_and_zeros_all_raw() -> None:
    reference = torch.tensor(
        [[0.011], [0.012], [0.013], [0.014]],
        dtype=torch.float64,
    )
    raw = torch.nn.Parameter(
        torch.tensor([[0.2], [-0.3], [0.4], [-0.5]], dtype=torch.float64)
    )
    evidence = torch.tensor(
        [[100.0], [1.0], [9.0], [200.0]],
        dtype=torch.float64,
    )
    filled = torch.tensor([False, True, True, False], dtype=torch.bool)
    reference_shape = reference.shape
    raw_shape = raw.shape

    median = initialize_uniform_median_guide_length(
        reference,
        raw,
        evidence,
        filled,
        label="unit guide length init",
    )

    assert median == 5.0
    assert reference.shape == reference_shape
    assert raw.shape == raw_shape
    assert reference.dtype == torch.float64
    assert raw.dtype == torch.float64
    torch.testing.assert_close(reference, torch.full_like(reference, 5.0))
    torch.testing.assert_close(raw, torch.zeros_like(raw))


def test_no_finite_positive_evidence_fails_before_mutation() -> None:
    reference = torch.tensor([[0.01], [0.02], [0.03]], dtype=torch.float32)
    raw = torch.nn.Parameter(
        torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)
    )
    evidence = torch.tensor([[float("nan")], [-1.0], [0.0]], dtype=torch.float32)
    filled = torch.ones((3,), dtype=torch.bool)
    reference_before = reference.clone()
    raw_before = raw.detach().clone()
    evidence_before = evidence.clone()
    filled_before = filled.clone()

    with pytest.raises(RuntimeError, match="no finite positive"):
        initialize_uniform_median_guide_length(
            reference,
            raw,
            evidence,
            filled,
            label="invalid guide length init",
        )

    torch.testing.assert_close(reference, reference_before)
    torch.testing.assert_close(raw, raw_before)
    torch.testing.assert_close(evidence, evidence_before, equal_nan=True)
    assert torch.equal(filled, filled_before)


@pytest.mark.parametrize(
    ("raw_shape", "evidence_shape", "mask_shape", "message"),
    [
        ((2, 1), (3, 1), (3,), "raw shape mismatch"),
        ((3, 1), (3,), (3,), "evidence shape mismatch"),
        ((3, 1), (3, 1), (3, 1), "filled mask shape mismatch"),
    ],
)
def test_shape_mismatch_fails_before_mutation(
    raw_shape: tuple[int, ...],
    evidence_shape: tuple[int, ...],
    mask_shape: tuple[int, ...],
    message: str,
) -> None:
    reference = torch.full((3, 1), 0.02, dtype=torch.float32)
    raw = torch.nn.Parameter(torch.full(raw_shape, 0.7, dtype=torch.float32))
    evidence = torch.full(evidence_shape, 0.03, dtype=torch.float32)
    filled = torch.ones(mask_shape, dtype=torch.bool)
    reference_before = reference.clone()
    raw_before = raw.detach().clone()
    evidence_before = evidence.clone()
    filled_before = filled.clone()

    with pytest.raises(ValueError, match=message):
        initialize_uniform_median_guide_length(
            reference,
            raw,
            evidence,
            filled,
            label="shape guide length init",
        )

    torch.testing.assert_close(reference, reference_before)
    torch.testing.assert_close(raw, raw_before)
    torch.testing.assert_close(evidence, evidence_before)
    assert torch.equal(filled, filled_before)

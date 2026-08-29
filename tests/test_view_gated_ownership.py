"""Contract tests for R072 per-view trusted ownership gating."""

from __future__ import annotations

import pytest
import torch

from anigroom.grooming import ViewGatedOwnership, straight_through_gate
from anigroom.grooming.guide_view_sh import TrustedGuideViewConfidence


def make_confidence(
    view_indices: list[int],
    values: list[list[float]],
) -> TrustedGuideViewConfidence:
    return TrustedGuideViewConfidence(
        view_indices=torch.tensor(view_indices, dtype=torch.long),
        confidence=torch.tensor(values, dtype=torch.float32),
        positive_q95=1.0,
        source_path="test-target.npz",
        summary_path="test-summary.json",
    )


def grad_of(value: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    leaf = value.clone().detach().requires_grad_(True)
    gated = leaf if gate is None else straight_through_gate(leaf, gate)
    gated.sum().backward()
    assert leaf.grad is not None
    return leaf.grad.detach().clone()


def test_gate_forward_value_is_bit_identical() -> None:
    value = torch.randn(64, 3, dtype=torch.float32, requires_grad=True)
    for share in (0.0, 0.25, 1.0):
        gate = torch.full((64, 1), share)
        assert torch.equal(straight_through_gate(value, gate).detach(), value.detach())


def test_unit_gate_preserves_gradient_exactly() -> None:
    value = torch.randn(32, 3)
    reference = grad_of(value, None)
    gated = grad_of(value, torch.ones(32, 1))
    assert torch.equal(gated, reference)


def test_zero_gate_removes_gradient_exactly() -> None:
    value = torch.randn(32, 3)
    gated = grad_of(value, torch.zeros(32, 1))
    assert torch.equal(gated, torch.zeros_like(gated))


def test_partial_gate_scales_gradient() -> None:
    value = torch.randn(16, 3)
    reference = grad_of(value, None)
    share = torch.rand(16, 1)
    gated = grad_of(value, share)
    assert torch.allclose(gated, reference * share, atol=1.0e-7)


def test_gate_broadcasts_over_trailing_dimensions() -> None:
    value = torch.randn(8, 3)
    share = torch.zeros(8, 1)
    share[:4] = 1.0
    gradient = grad_of(value, share)
    assert torch.equal(gradient[4:], torch.zeros_like(gradient[4:]))
    assert bool((gradient[:4] != 0.0).all())


def test_gate_rejects_invalid_values() -> None:
    value = torch.randn(4, 3, requires_grad=True)
    with pytest.raises(ValueError):
        straight_through_gate(value, torch.full((4, 1), 1.5))
    with pytest.raises(ValueError):
        straight_through_gate(value, torch.full((4, 1), -0.1))
    with pytest.raises(ValueError):
        straight_through_gate(value, torch.full((4, 1), float("nan")))
    with pytest.raises(ValueError):
        straight_through_gate(value, torch.ones(5, 1))


def test_guide_gate_returns_trusted_confidence() -> None:
    ownership = ViewGatedOwnership(
        confidence=make_confidence([3, 9], [[0.0, 0.5, 1.0], [0.2, 0.2, 0.2]]),
    )
    assert torch.allclose(ownership.guide_gate(3), torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(ownership.guide_gate(9), torch.tensor([0.2, 0.2, 0.2]))


def test_untrusted_view_owns_nothing_by_default() -> None:
    ownership = ViewGatedOwnership(
        confidence=make_confidence([3], [[0.4, 0.6, 0.8]]),
    )
    assert not ownership.has_view(25)
    assert torch.equal(ownership.guide_gate(25), torch.zeros(3))


def test_floor_retains_a_minimum_share_everywhere() -> None:
    ownership = ViewGatedOwnership(
        confidence=make_confidence([3], [[0.0, 0.5, 1.0]]),
        floor=0.25,
    )
    assert torch.allclose(ownership.guide_gate(3), torch.tensor([0.25, 0.625, 1.0]))
    assert torch.allclose(ownership.guide_gate(25), torch.full((3,), 0.25))


def test_unit_floor_disables_gating() -> None:
    ownership = ViewGatedOwnership(
        confidence=make_confidence([3], [[0.0, 0.5, 1.0]]),
        floor=1.0,
    )
    assert torch.allclose(ownership.guide_gate(3), torch.ones(3))
    assert torch.allclose(ownership.guide_gate(25), torch.ones(3))


def test_floor_outside_unit_interval_is_rejected() -> None:
    confidence = make_confidence([3], [[0.5, 0.5, 0.5]])
    with pytest.raises(ValueError):
        ViewGatedOwnership(confidence=confidence, floor=-0.1)
    with pytest.raises(ValueError):
        ViewGatedOwnership(confidence=confidence, floor=1.1)


def test_report_separates_trusted_and_untrusted_training_views() -> None:
    ownership = ViewGatedOwnership(
        confidence=make_confidence([1, 3], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    report = ownership.report([1, 3, 25])
    assert report["requested_view_count"] == 3
    assert report["trusted_view_count"] == 2
    assert report["untrusted_view_indices"] == [25]
    assert report["guide_count"] == 3
    # guides 0 and 1 have an owner; guide 2 has none.
    assert report["guides_with_owner_fraction"] == pytest.approx(2.0 / 3.0)
    assert report["owner_views_per_guide_mean"] == pytest.approx(2.0 / 3.0)

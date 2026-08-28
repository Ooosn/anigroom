from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from anigroom.grooming import (
    GuideViewSHField,
    first_order_sh_basis,
    load_trusted_guide_view_confidence,
)


def write_confidence_fixture(tmp_path, *, views=(1, 5, 9)):
    target = tmp_path / "target.npz"
    summary = tmp_path / "summary.json"
    weights = np.asarray(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.1, 0.0, 0.5, 0.3],
            [0.2, 0.4, 0.0, 0.8],
        ],
        dtype=np.float32,
    )
    face_ids = np.asarray([0, 1, 1, 2], dtype=np.int64)
    barycentric = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        target,
        axis_view_cluster_selected_direct_weight=weights,
        face_ids=face_ids,
        barycentric=barycentric,
    )
    summary.write_text(json.dumps({"views_used": list(views)}) + "\n", encoding="utf-8")
    return target, summary, weights, face_ids, barycentric


def test_zero_field_is_exact_zero() -> None:
    field = GuideViewSHField(3, 0.2)
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    residual = field.residual(directions)
    torch.testing.assert_close(residual, torch.zeros_like(residual), rtol=0.0, atol=0.0)


def test_degree_one_field_is_view_dependent_and_rgb_specific() -> None:
    field = GuideViewSHField(2, 0.2)
    with torch.no_grad():
        field.raw[0, 0, 0] = 0.7
        field.raw[1, 1, 2] = -0.5
    x_view = field.residual(torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    y_view = field.residual(torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
    assert float(x_view[0, 0].detach()) > 0.0
    assert float(y_view[0, 0].detach()) == pytest.approx(0.0, abs=1.0e-8)
    assert float(y_view[1, 2].detach()) < 0.0
    torch.testing.assert_close(x_view[:, 1], torch.zeros_like(x_view[:, 1]))


def test_gradient_confidence_changes_only_gradient_not_forward() -> None:
    field = GuideViewSHField(3, 0.2)
    with torch.no_grad():
        field.raw.fill_(0.4)
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    ungated = field.residual(directions)
    gated = field.residual(
        directions,
        gradient_confidence=torch.tensor([0.0, 0.5, 1.0]),
    )
    torch.testing.assert_close(gated, ungated, rtol=0.0, atol=0.0)
    gated.sum().backward()
    assert field.raw.grad is not None
    row_magnitude = field.raw.grad.abs().sum(dim=(1, 2))
    assert float(row_magnitude[0]) == 0.0
    assert float(row_magnitude[1]) > 0.0
    assert float(row_magnitude[2]) == pytest.approx(
        2.0 * float(row_magnitude[1]), rel=1.0e-6
    )


def test_basis_normalizes_input_and_rejects_nonfinite() -> None:
    basis = first_order_sh_basis(torch.tensor([[2.0, 0.0, 0.0]]))
    torch.testing.assert_close(
        basis,
        torch.tensor([[0.48860252, 0.0, 0.0]]),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    with pytest.raises(ValueError, match="finite"):
        first_order_sh_basis(torch.tensor([[float("nan"), 0.0, 0.0]]))


def test_decoded_coefficients_are_bounded_and_stats_are_exact() -> None:
    field = GuideViewSHField(5, 0.2)
    with torch.no_grad():
        field.raw.copy_(torch.linspace(-2.0, 2.0, field.raw.numel()).reshape_as(field.raw))
    decoded = field.decoded_coefficients()
    assert float(decoded.detach().abs().max()) < 0.2
    stats = field.stats(guide_chunk_size=2)
    assert stats["guide_count"] == 5
    assert stats["degree"] == 1
    assert stats["coefficient_count"] == 45
    assert stats["abs_mean"] == pytest.approx(float(decoded.abs().mean()), rel=1.0e-6)
    assert stats["rms"] == pytest.approx(float(decoded.square().mean().sqrt()), rel=1.0e-6)
    assert stats["abs_max"] == pytest.approx(float(decoded.abs().max()), rel=1.0e-6)


def test_confidence_loader_validates_identity_and_absent_view(tmp_path) -> None:
    target, summary, weights, face_ids, barycentric = write_confidence_fixture(tmp_path)
    loaded = load_trusted_guide_view_confidence(
        target,
        summary_path=summary,
        expected_face_ids=face_ids,
        expected_barycentric=barycentric,
    )
    assert loaded.view_count == 3
    assert loaded.guide_count == 4
    assert loaded.positive_q95 == pytest.approx(float(np.quantile(weights[weights > 0], 0.95)))
    torch.testing.assert_close(
        loaded.confidence_for_view(5),
        loaded.confidence[1],
    )
    torch.testing.assert_close(
        loaded.confidence_for_view(4),
        torch.zeros(4),
        rtol=0.0,
        atol=0.0,
    )
    assert loaded.report()["supported_guide_fraction"] == 1.0


def test_confidence_loader_rejects_missing_weight_key(tmp_path) -> None:
    target = tmp_path / "target.npz"
    np.savez_compressed(target, face_ids=np.asarray([0], dtype=np.int64))
    (tmp_path / "summary.json").write_text('{"views_used": [1]}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="selected_direct_weight"):
        load_trusted_guide_view_confidence(target)


@pytest.mark.parametrize(
    "views,match",
    [
        ((1, 1, 9), "duplicate"),
        ((1, 5), "length"),
    ],
)
def test_confidence_loader_rejects_invalid_view_mapping(tmp_path, views, match) -> None:
    target, summary, *_ = write_confidence_fixture(tmp_path, views=views)
    with pytest.raises(RuntimeError, match=match):
        load_trusted_guide_view_confidence(target, summary_path=summary)


def test_confidence_loader_rejects_guide_identity_mismatch(tmp_path) -> None:
    target, summary, _, face_ids, barycentric = write_confidence_fixture(tmp_path)
    wrong_faces = face_ids.copy()
    wrong_faces[0] = 99
    with pytest.raises(RuntimeError, match="face_ids"):
        load_trusted_guide_view_confidence(
            target,
            summary_path=summary,
            expected_face_ids=wrong_faces,
            expected_barycentric=barycentric,
        )
    wrong_barycentric = barycentric.copy()
    wrong_barycentric[0, 0] += np.float32(1.0e-4)
    with pytest.raises(RuntimeError, match="barycentric"):
        load_trusted_guide_view_confidence(
            target,
            summary_path=summary,
            expected_face_ids=face_ids,
            expected_barycentric=wrong_barycentric,
        )

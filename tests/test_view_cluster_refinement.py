from __future__ import annotations

import inspect

import torch

from anigroom.flow.view_cluster_refinement import (
    AXIAL_AGREEMENT_POWER,
    CONFIDENCE_DECAY,
    DIRECT_SUPPORT_ANGLE_DEG,
    DIRECT_SUPERMAJORITY,
    HARD_MARGIN_QUANTILE,
    RESIDUAL_QUANTILE,
    VIEW_CLUSTER_ITERATIONS,
    refine_trusted_multiview_axis_field,
)


def _inputs(
    *,
    n: int = 5,
    views: int = 3,
    axis: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    direct_weights: torch.Tensor | None = None,
    observed: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if axis is None:
        axis = torch.tensor([[1.0, 0.0, 0.0]] * n)
    normals = torch.tensor([[0.0, 0.0, 1.0]] * n)
    if observed is None:
        observed = torch.ones(n, dtype=torch.bool)
    if vectors is None:
        vectors = axis[None].expand(views, -1, -1).clone()
    if weights is None:
        weights = torch.ones((views, n))
    if direct_weights is None:
        direct_weights = weights.clone()
    knn = torch.tensor(
        [
            [1, 2],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, 2],
        ],
        dtype=torch.long,
    )[:n]
    edge_weight = torch.ones((n, 2))
    return {
        "initial_axis": axis,
        "normals": normals,
        "observed": observed,
        "per_view_vectors": vectors,
        "per_view_weights": weights,
        "per_view_direct_weights": direct_weights,
        "knn": knn,
        "edge_weight": edge_weight,
    }


def _run(**kwargs: torch.Tensor) -> dict[str, torch.Tensor | float | int]:
    return refine_trusted_multiview_axis_field(**_inputs(**kwargs))


def _root_tensor_keys(result: dict[str, torch.Tensor | float | int]) -> list[str]:
    return [
        key
        for key, value in result.items()
        if isinstance(value, torch.Tensor) and value.shape[0] == 5
    ]


def test_zero_evidence_is_finite_and_shaped() -> None:
    data = _inputs(n=5)
    data["per_view_vectors"] = torch.zeros_like(data["per_view_vectors"])
    data["per_view_weights"] = torch.zeros_like(data["per_view_weights"])
    data["per_view_direct_weights"] = torch.zeros_like(data["per_view_direct_weights"])

    result = refine_trusted_multiview_axis_field(**data)

    assert result["axis"].shape == (5, 3)
    assert result["anchor"].shape == (5, 3)
    assert result["hard_q95_mask"].shape == (5,)
    assert result["residual_mask"].shape == (5,)
    assert result["residual_supermajority_mask"].shape == (5,)
    for value in result.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert torch.isfinite(value).all()
    torch.testing.assert_close(result["axis"], data["initial_axis"])
    torch.testing.assert_close(result["trust"], torch.zeros(5))
    assert not bool(result["hard_q95_mask"].any())
    assert not bool(result["residual_supermajority_mask"].any())


def test_axial_sign_and_view_order_permutations_preserve_root_field() -> None:
    axis = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.20, 0.0],
            [0.94, -0.34, 0.0],
            [0.90, 0.44, 0.0],
            [0.86, -0.51, 0.0],
        ]
    )
    vectors = torch.stack(
        [
            axis,
            torch.tensor([[1.0, 0.0, 0.0], [0.98, 0.21, 0.0], [0.95, -0.31, 0.0], [0.89, 0.46, 0.0], [0.84, -0.54, 0.0]]),
            torch.tensor([[0.99, 0.04, 0.0], [0.97, 0.18, 0.0], [0.92, -0.38, 0.0], [0.92, 0.39, 0.0], [0.88, -0.47, 0.0]]),
        ]
    )
    weights = torch.tensor(
        [
            [1.0, 1.2, 0.8, 1.1, 0.9],
            [0.7, 1.8, 1.1, 0.6, 1.4],
            [1.6, 0.9, 1.3, 1.5, 0.5],
        ]
    )
    data = _inputs(axis=axis, vectors=vectors, weights=weights, direct_weights=weights)
    base = refine_trusted_multiview_axis_field(**data)

    signs = torch.tensor([1.0, -1.0, -1.0])[:, None, None]
    flipped = dict(data)
    flipped["per_view_vectors"] = vectors * signs
    signed = refine_trusted_multiview_axis_field(**flipped)
    perm = torch.tensor([2, 0, 1])
    reordered = dict(data)
    reordered["per_view_vectors"] = vectors[perm]
    reordered["per_view_weights"] = weights[perm]
    reordered["per_view_direct_weights"] = weights[perm]
    reordered_result = refine_trusted_multiview_axis_field(**reordered)

    for key in ("axis", "anchor", "trust", "spectral_gap", "n_eff", "hard_axis", "hard_margin", "residual_deg", "residual_direct_support", "confidence"):
        torch.testing.assert_close(signed[key], base[key], atol=1.0e-5, rtol=1.0e-5)
        torch.testing.assert_close(reordered_result[key], base[key], atol=1.0e-5, rtol=1.0e-5)
    for key in ("hard_q95_mask", "residual_mask", "residual_supermajority_mask"):
        torch.testing.assert_close(signed[key], base[key])
        torch.testing.assert_close(reordered_result[key], base[key])


def test_isolated_conflicting_root_is_corrected_from_supported_view_cluster() -> None:
    x = torch.tensor([1.0, 0.0, 0.0])
    y = torch.tensor([0.0, 1.0, 0.0])
    axis = torch.stack([x, y, x, x, x])
    vectors = torch.stack(
        [
            torch.stack([x, x, x, x, x]),
            torch.stack([x, y, x, x, x]),
            torch.stack([x, y, x, x, x]),
        ]
    )
    weights = torch.ones((3, 5))
    weights[:, 1] = torch.tensor([1.0, 3.0, 3.0])
    result = _run(axis=axis, vectors=vectors, weights=weights, direct_weights=torch.ones_like(weights))

    assert bool(result["hard_q95_mask"][1])
    assert float(result["hard_axis"][1, 0]) > 0.9
    assert float(result["hard_axis"][1, 1].abs()) < 0.45
    assert float((result["axis"][1] * x).sum()) > 0.9


def test_residual_interpolation_without_two_thirds_direct_support_is_rejected() -> None:
    x = torch.tensor([1.0, 0.0, 0.0])
    y = torch.tensor([0.0, 1.0, 0.0])
    axis = torch.stack([x, y, x, x, x])
    vectors = torch.stack([torch.stack([x, y, x, x, x])] * 3)
    direct = torch.ones((3, 5))
    result = _run(axis=axis, vectors=vectors, weights=torch.ones_like(direct), direct_weights=direct)

    assert bool(result["residual_mask"][1])
    assert float(result["residual_direct_support"][1]) < DIRECT_SUPERMAJORITY
    assert not bool(result["residual_supermajority_mask"][1])
    assert float((result["axis"][1] * y).sum()) > 0.9


def test_high_confidence_coherent_anchors_remain_stable() -> None:
    x = torch.tensor([1.0, 0.0, 0.0])
    axis = x.repeat(5, 1)
    weights = torch.tensor(
        [
            [10.0, 9.0, 11.0, 10.0, 9.5],
            [9.0, 10.0, 10.0, 11.0, 9.0],
            [11.0, 10.0, 9.0, 10.0, 10.5],
        ]
    )
    result = _run(axis=axis, weights=weights, direct_weights=weights)

    assert torch.all((result["axis"][:, 0] > 0.999))
    assert torch.all(result["axis"][:, 1].abs() < 1.0e-4)
    assert torch.all(result["hard_margin"] == 0)
    assert not bool(result["residual_mask"].any())
    assert torch.all(result["confidence"] >= result["trust"])


def test_constants_and_signature_are_semantic_and_view_independent() -> None:
    signature = inspect.signature(refine_trusted_multiview_axis_field)
    expected = {
        "initial_axis",
        "normals",
        "observed",
        "per_view_vectors",
        "per_view_weights",
        "per_view_direct_weights",
        "knn",
        "edge_weight",
    }
    assert set(signature.parameters) == expected
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in signature.parameters.values())
    assert AXIAL_AGREEMENT_POWER == 8
    assert VIEW_CLUSTER_ITERATIONS == 4
    assert HARD_MARGIN_QUANTILE == 0.95
    assert RESIDUAL_QUANTILE == 0.95
    assert DIRECT_SUPPORT_ANGLE_DEG == 30.0
    assert DIRECT_SUPERMAJORITY == 2.0 / 3.0
    assert CONFIDENCE_DECAY == 0.85

    data = _inputs()
    base = refine_trusted_multiview_axis_field(**data)
    root_perm = torch.tensor([2, 0, 4, 1, 3])
    inverse = torch.empty_like(root_perm)
    inverse[root_perm] = torch.arange(root_perm.numel())
    permuted = dict(data)
    permuted["initial_axis"] = data["initial_axis"][root_perm]
    permuted["normals"] = data["normals"][root_perm]
    permuted["observed"] = data["observed"][root_perm]
    permuted["per_view_vectors"] = data["per_view_vectors"][:, root_perm]
    permuted["per_view_weights"] = data["per_view_weights"][:, root_perm]
    permuted["per_view_direct_weights"] = data["per_view_direct_weights"][:, root_perm]
    permuted["knn"] = inverse[data["knn"]][root_perm]
    permuted["edge_weight"] = data["edge_weight"][root_perm]
    reordered = refine_trusted_multiview_axis_field(**permuted)
    for key in ("axis", "anchor", "trust", "spectral_gap", "n_eff", "hard_axis", "hard_margin", "residual_deg", "confidence"):
        torch.testing.assert_close(reordered[key], base[key][root_perm], atol=1.0e-5, rtol=1.0e-5)

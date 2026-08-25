from __future__ import annotations

import torch

from anigroom.grooming import GroomParameterField
from tools.train_white_tiger_stage1 import (
    packed_appearance_root_graph_smoothness,
    precompute_root_graph_edge_weight_cache,
    root_graph_smoothness,
)


def _field(root_count: int = 7) -> GroomParameterField:
    field = GroomParameterField(root_count, device="cpu")
    with torch.no_grad():
        field.root_color_raw.copy_(torch.randn_like(field.root_color_raw))
        field.tip_color_raw.copy_(torch.randn_like(field.tip_color_raw))
        field.opacity_raw.copy_(torch.randn_like(field.opacity_raw))
        field.tip_opacity_ratio_raw.copy_(torch.randn_like(field.tip_opacity_ratio_raw))
    return field


def _edges() -> torch.Tensor:
    return torch.tensor(
        [[0, 1], [1, 2], [2, 0], [3, 5], [5, 4], [6, 3]],
        dtype=torch.long,
    )


def test_packed_appearance_matches_explicit_scalar() -> None:
    field = _field()
    edges = _edges()
    confidence = torch.tensor([0.0, 0.2, 0.7, 1.0, 0.4, 0.8, 0.1])

    explicit = root_graph_smoothness(
        field,
        edges,
        confidence,
        include_geometry=False,
        appearance_only=True,
    )
    packed = packed_appearance_root_graph_smoothness(
        field,
        edges,
        edge_weight_cache=precompute_root_graph_edge_weight_cache(
            edges,
            confidence,
            dtype=field.root_color_raw.dtype,
        ),
    )

    torch.testing.assert_close(packed, explicit, rtol=1.0e-6, atol=1.0e-7)


def test_packed_path_has_all_appearance_gradients_and_no_geometry_gradients() -> None:
    field = _field()
    loss = packed_appearance_root_graph_smoothness(field, _edges())
    loss.backward()

    for parameter in (
        field.root_color_raw,
        field.tip_color_raw,
        field.opacity_raw,
        field.tip_opacity_ratio_raw,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert bool(parameter.grad.abs().sum() > 0.0)

    for parameter in (
        field.length_raw,
        field.root_width_raw,
        field.tip_width_ratio_raw,
        field.width_taper_raw,
        field.direction_local_raw,
        field.brush_stiffness_raw,
        field.curl_radius_ratio_raw,
        field.curl_turns_raw,
        field.curl_phase,
        field.child_radius_raw,
        field.clump_strength_raw,
    ):
        assert parameter.grad is None


def test_cached_and_uncached_weights_match() -> None:
    field = _field()
    edges = _edges()
    confidence = torch.tensor([0.1, 0.6, 0.3, 0.9, 0.0, 0.5, 0.2])
    cache = precompute_root_graph_edge_weight_cache(
        edges,
        confidence,
        dtype=field.root_color_raw.dtype,
    )

    uncached = packed_appearance_root_graph_smoothness(
        field,
        edges,
        confidence,
    )
    cached = packed_appearance_root_graph_smoothness(
        field,
        edges,
        edge_weight_cache=cache,
    )

    expected_weights = 0.25 + (
        1.0 - torch.minimum(confidence[edges[:, 0]], confidence[edges[:, 1]])
    )
    torch.testing.assert_close(cache.edge_weights, expected_weights)
    assert not cache.edge_weights.requires_grad
    assert not cache.denominator.requires_grad
    torch.testing.assert_close(cached, uncached, rtol=0.0, atol=0.0)


def test_empty_graph_is_exact_zero() -> None:
    field = _field(3)
    empty_edges = torch.empty((0, 2), dtype=torch.long)
    confidence = torch.zeros(3)

    explicit = root_graph_smoothness(
        field,
        empty_edges,
        confidence,
        include_geometry=False,
        appearance_only=True,
    )
    packed = packed_appearance_root_graph_smoothness(
        field,
        empty_edges,
        confidence,
        edge_weight_cache=precompute_root_graph_edge_weight_cache(
            empty_edges,
            confidence,
            dtype=field.root_color_raw.dtype,
        ),
    )

    assert explicit.shape == torch.Size([])
    assert packed.shape == torch.Size([])
    assert float(explicit) == 0.0
    assert float(packed) == 0.0
    assert packed.dtype == explicit.dtype
    assert packed.requires_grad == explicit.requires_grad


def test_weight_cache_refreshes_for_changed_edges_and_confidence() -> None:
    field = _field()
    edges_a = _edges()
    confidence_a = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.1])
    edges_b = torch.tensor(
        [[0, 6], [6, 2], [2, 5], [5, 1], [1, 4], [4, 3]],
        dtype=torch.long,
    )
    confidence_b = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    cache_a = precompute_root_graph_edge_weight_cache(edges_a, confidence_a)
    cache_b = precompute_root_graph_edge_weight_cache(edges_b, confidence_b)

    assert not torch.equal(cache_a.edge_weights, cache_b.edge_weights)
    assert float(cache_a.denominator) != float(cache_b.denominator)
    cached_b = packed_appearance_root_graph_smoothness(
        field,
        edges_b,
        edge_weight_cache=cache_b,
    )
    uncached_b = packed_appearance_root_graph_smoothness(
        field,
        edges_b,
        confidence_b,
    )
    torch.testing.assert_close(cached_b, uncached_b, rtol=0.0, atol=0.0)


def test_packed_path_does_not_decode_full_field(monkeypatch) -> None:
    field = _field()

    def fail_decode() -> None:
        raise AssertionError("packed appearance path called full field.decode()")

    monkeypatch.setattr(field, "decode", fail_decode)
    result = packed_appearance_root_graph_smoothness(field, _edges())
    assert torch.isfinite(result)

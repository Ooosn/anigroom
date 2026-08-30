from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from anigroom.flow.confidence_guided_direction import (
    refine_confidence_guided_directed_flow,
)
from anigroom.flow.global_sign_orientation import SEVERE_DOT_THRESHOLD


def _canonical_ranks(points: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(points[:, 0], stable=True)
    ranks = torch.empty(points.shape[0], dtype=torch.long)
    ranks[order] = torch.arange(points.shape[0], dtype=torch.long)
    return ranks


def _chain_points() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def _chain_inputs(*, reversed_patch: bool) -> dict[str, torch.Tensor]:
    points = _chain_points()
    count = int(points.shape[0])
    coherent = F.normalize(torch.tensor([1.0, 0.0, 0.25]), dim=0)
    reversed_direction = F.normalize(torch.tensor([-1.0, 0.0, 0.25]), dim=0)
    direction = coherent.repeat(count, 1)
    field_confidence = torch.ones(count, dtype=torch.float32)
    if reversed_patch:
        direction[2] = reversed_direction
        field_confidence[2] = 0.01
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).repeat(count, 1)
    return {
        "direction": direction,
        "normals": normals,
        "observed": torch.ones(count, dtype=torch.bool),
        "edge_u": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "edge_v": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        "field_confidence": field_confidence,
        "unary_normalized_margin": torch.ones(count, dtype=torch.float32),
        "unary_vote_coherence": torch.ones(count, dtype=torch.float32),
        "canonical_rank": _canonical_ranks(points),
    }


def _mild_two_root_inputs() -> dict[str, torch.Tensor]:
    points = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32
    )
    direction = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.4],
                [-0.8, 0.8, 0.4],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    return {
        "direction": direction,
        "normals": torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        ),
        "observed": torch.ones(2, dtype=torch.bool),
        "edge_u": torch.tensor([0], dtype=torch.long),
        "edge_v": torch.tensor([1], dtype=torch.long),
        "field_confidence": torch.tensor([1.0, 0.2], dtype=torch.float32),
        "unary_normalized_margin": torch.ones(2, dtype=torch.float32),
        "unary_vote_coherence": torch.ones(2, dtype=torch.float32),
        "canonical_rank": _canonical_ranks(points),
    }


def _assert_normalized_outward_finite(
    result: dict[str, object], normals: torch.Tensor
) -> None:
    normal = F.normalize(normals, dim=-1)
    for key in ("direction", "watershed_direction"):
        value = result[key]
        assert isinstance(value, torch.Tensor)
        assert bool(torch.isfinite(value).all())
        torch.testing.assert_close(
            torch.linalg.vector_norm(value, dim=-1),
            torch.ones(value.shape[0]),
            atol=3.0e-6,
            rtol=3.0e-6,
        )
        assert bool(((value * normal).sum(dim=-1) >= -3.0e-6).all())
    for value in result.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            assert bool(torch.isfinite(value).all())


def _assert_monotone_traces(result: dict[str, object]) -> None:
    report = result["report"]
    assert isinstance(report, dict)
    watershed = report["watershed"]
    local_cleanup = report["local_cleanup"]
    counts = report["counts"]
    assert isinstance(watershed, dict)
    assert isinstance(local_cleanup, dict)
    assert isinstance(counts, dict)

    watershed_trace = watershed["accepted_trace"]
    local_trace = local_cleanup["accepted_trace"]
    assert isinstance(watershed_trace, list)
    assert isinstance(local_trace, list)
    assert [entry["step"] for entry in watershed_trace] == list(
        range(1, len(watershed_trace) + 1)
    )
    assert [entry["step"] for entry in local_trace] == list(
        range(1, len(local_trace) + 1)
    )
    for entry in watershed_trace:
        assert entry["incident_severe_reduction"] > 0
        assert entry["net_incident_negative_reduction"] > 0
        assert entry["incident_hinge_improvement"] > 0.0
        assert (
            entry["incident_severe_reduction_per_changed_root"]
            > entry["baseline_graph_severe_per_root"]
        )
    for entry in local_trace:
        assert entry["net_incident_negative_reduction"] > 0
        assert entry["incident_hinge_improvement"] > 0.0
        assert entry["propagated_confidence"] > entry["trust"]

    all_edges = counts["all_edges"]
    observed_edges = counts["observed_edges"]
    assert isinstance(all_edges, dict)
    assert isinstance(observed_edges, dict)
    assert all_edges["initial_negative"] >= all_edges["post_watershed_negative"]
    assert all_edges["post_watershed_negative"] >= all_edges["final_negative"]
    assert all_edges["initial_severe"] >= all_edges["post_watershed_severe"]
    assert all_edges["post_watershed_severe"] >= all_edges["final_severe"]
    assert all_edges["new_severe"] == 0
    assert observed_edges == all_edges
    verification = report["zero_new_severe_verification"]
    assert isinstance(verification, dict)
    assert verification["new_severe_edge_count"] == 0
    assert verification["passed"] is True


def test_coherent_field_is_a_no_op() -> None:
    inputs = _chain_inputs(reversed_patch=False)
    result = refine_confidence_guided_directed_flow(**inputs)

    torch.testing.assert_close(result["direction"], inputs["direction"])
    torch.testing.assert_close(
        result["watershed_direction"], inputs["direction"]
    )
    assert not bool(result["changed_mask"].any())
    assert not bool(result["watershed_changed_mask"].any())
    assert not bool(result["local_changed_mask"].any())
    report = result["report"]
    assert report["watershed"]["accepted_trace"] == []
    assert report["local_cleanup"]["accepted_trace"] == []
    assert report["counts"]["all_edges"] == {
        "initial_negative": 0,
        "post_watershed_negative": 0,
        "final_negative": 0,
        "initial_severe": 0,
        "post_watershed_severe": 0,
        "final_severe": 0,
        "new_severe": 0,
    }
    _assert_monotone_traces(result)
    _assert_normalized_outward_finite(result, inputs["normals"])


def test_severe_low_confidence_patch_is_repaired_by_accepted_watershed() -> None:
    inputs = _chain_inputs(reversed_patch=True)
    result = refine_confidence_guided_directed_flow(**inputs)
    report = result["report"]
    counts = report["counts"]["all_edges"]

    assert counts["initial_severe"] == 2
    assert counts["final_severe"] == 0
    assert counts["new_severe"] == 0
    assert counts["final_negative"] == 0
    assert report["watershed"]["accepted_basin_count"] == 1
    trace = report["watershed"]["accepted_trace"]
    assert len(trace) == 1
    owner_root = int(trace[0]["owner_root"])
    assert owner_root == 1
    assert int(result["watershed_owner"][2]) == owner_root
    assert bool(result["protected_owner_mask"][owner_root])
    torch.testing.assert_close(
        result["direction"][owner_root], inputs["direction"][owner_root]
    )
    assert not bool(result["changed_mask"][owner_root])
    assert bool(result["changed_mask"][2])
    assert bool((result["final_edge_dots"] > 0.9).all())
    assert not bool(result["new_severe_edge_mask"].any())
    _assert_monotone_traces(result)
    _assert_normalized_outward_finite(result, inputs["normals"])


def test_mild_negative_two_root_edge_skips_watershed_and_uses_local_cleanup() -> None:
    inputs = _mild_two_root_inputs()
    result = refine_confidence_guided_directed_flow(**inputs)
    report = result["report"]
    counts = report["counts"]["all_edges"]

    initial_dot = float(result["initial_edge_dots"][0])
    assert SEVERE_DOT_THRESHOLD < initial_dot < 0.0
    assert counts["initial_negative"] == 1
    assert counts["initial_severe"] == 0
    assert report["watershed"]["accepted_basin_count"] == 0
    assert report["watershed"]["accepted_trace"] == []
    torch.testing.assert_close(result["watershed_direction"], inputs["direction"])
    assert report["local_cleanup"]["accepted_update_count"] == 1
    assert result["local_update_count"].tolist() == [0, 1]
    assert counts["post_watershed_negative"] == 1
    assert counts["final_negative"] == 0
    assert counts["final_severe"] == 0
    assert float(result["final_edge_dots"][0]) > 0.9
    assert bool(result["local_changed_mask"][1])
    _assert_monotone_traces(result)
    _assert_normalized_outward_finite(result, inputs["normals"])


def test_root_and_input_edge_order_permutations_preserve_directions_and_counts() -> None:
    inputs = _chain_inputs(reversed_patch=True)
    base = refine_confidence_guided_directed_flow(**inputs)
    permutation = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel(), dtype=torch.long)
    edge_order = torch.tensor([2, 0, 3, 1], dtype=torch.long)

    rootwise_keys = (
        "direction",
        "normals",
        "observed",
        "field_confidence",
        "unary_normalized_margin",
        "unary_vote_coherence",
    )
    permuted_inputs = {
        key: inputs[key][permutation] for key in rootwise_keys
    }
    permuted_inputs["edge_u"] = inverse[inputs["edge_u"]][edge_order]
    permuted_inputs["edge_v"] = inverse[inputs["edge_v"]][edge_order]
    permuted_inputs["canonical_rank"] = _canonical_ranks(
        _chain_points()[permutation]
    )
    permuted = refine_confidence_guided_directed_flow(**permuted_inputs)

    mapped_direction = torch.empty_like(permuted["direction"])
    mapped_direction[permutation] = permuted["direction"]
    torch.testing.assert_close(mapped_direction, base["direction"])
    mapped_watershed_direction = torch.empty_like(permuted["watershed_direction"])
    mapped_watershed_direction[permutation] = permuted["watershed_direction"]
    torch.testing.assert_close(
        mapped_watershed_direction, base["watershed_direction"]
    )
    mapped_owner = torch.empty_like(permuted["watershed_owner"])
    mapped_owner[permutation] = permutation[permuted["watershed_owner"]]
    torch.testing.assert_close(mapped_owner, base["watershed_owner"])
    torch.testing.assert_close(
        permutation[permuted["edge_u"]], base["edge_u"]
    )
    torch.testing.assert_close(
        permutation[permuted["edge_v"]], base["edge_v"]
    )
    assert permuted["report"]["counts"] == base["report"]["counts"]
    assert (
        permuted["report"]["watershed"]["accepted_basin_count"]
        == base["report"]["watershed"]["accepted_basin_count"]
    )
    assert (
        permuted["report"]["local_cleanup"]["accepted_update_count"]
        == base["report"]["local_cleanup"]["accepted_update_count"]
    )
    _assert_monotone_traces(base)
    _assert_monotone_traces(permuted)
    _assert_normalized_outward_finite(base, inputs["normals"])
    _assert_normalized_outward_finite(permuted, permuted_inputs["normals"])


@pytest.mark.parametrize(
    ("invalid_case", "expected_exception", "message"),
    [
        ("direction_shape", ValueError, "direction must have shape"),
        ("normals_shape", ValueError, "normals must have shape"),
        ("direction_dtype", TypeError, "floating-point"),
        ("edge_endpoint", ValueError, "out-of-range"),
    ],
)
def test_invalid_shapes_dtypes_and_endpoints_are_rejected(
    invalid_case: str, expected_exception: type[Exception], message: str
) -> None:
    inputs = _chain_inputs(reversed_patch=False)
    if invalid_case == "direction_shape":
        inputs["direction"] = inputs["direction"][:, :2]
    elif invalid_case == "normals_shape":
        inputs["normals"] = inputs["normals"][:, :2]
    elif invalid_case == "direction_dtype":
        inputs["direction"] = torch.ones((5, 3), dtype=torch.int64)
    else:
        inputs["edge_v"] = inputs["edge_v"].clone()
        inputs["edge_v"][-1] = inputs["direction"].shape[0]

    with pytest.raises(expected_exception, match=message):
        refine_confidence_guided_directed_flow(**inputs)

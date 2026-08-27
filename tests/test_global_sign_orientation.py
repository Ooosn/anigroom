from __future__ import annotations

import inspect
import json
import math

import pytest
import torch
import torch.nn.functional as F

from anigroom.flow.global_sign_orientation import (
    COS45,
    GLOBAL_SIGN_ALPHA_MULTIPLIER,
    MAX_BLOCK_STEPS,
    SEVERE_DOT_THRESHOLD,
    refine_global_tangent_sign_field,
)


def _project(
    points: torch.Tensor,
    directions: torch.Tensor,
    viewmat: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    rotation = viewmat[:3, :3]
    translation = viewmat[:3, 3]
    camera_points = points @ rotation.T + translation[None]
    camera_directions = directions @ rotation.T
    depth = camera_points[:, 2].clamp_min(1.0e-6)
    denominator = depth.square()
    return torch.stack(
        (
            intrinsic[0, 0]
            * (camera_directions[:, 0] * depth - camera_points[:, 0] * camera_directions[:, 2])
            / denominator,
            intrinsic[1, 1]
            * (camera_directions[:, 1] * depth - camera_points[:, 1] * camera_directions[:, 2])
            / denominator,
        ),
        dim=-1,
    )


def _base_inputs(
    *,
    n: int = 3,
    views: int = 1,
    ratio: float = 0.2,
    graph: bool = True,
) -> dict[str, torch.Tensor]:
    normal = F.normalize(torch.tensor([0.0, 1.0, 1.0]), dim=0)
    points = torch.tensor(
        [
            [0.40, 0.20, 2.00],
            [0.60, 0.10, 2.10],
            [-0.30, 0.35, 1.90],
            [0.10, -0.20, 2.30],
        ]
    )[:n]
    normals = normal.repeat(n, 1)
    tangent = torch.tensor([1.0, 0.0, 0.0]).repeat(n, 1)
    ratios = torch.full((n,), ratio)
    signs = torch.ones(n)
    d_plus = F.normalize(ratios[:, None] * normals + tangent, dim=-1)
    d_minus = F.normalize(ratios[:, None] * normals - tangent, dim=-1)
    per_view_axes = d_plus[None].repeat(views, 1, 1)
    per_view_weights = torch.ones((views, n))
    viewmats = torch.eye(4)[None].repeat(views, 1, 1)
    intrinsics = torch.diag(torch.tensor([700.0, 710.0, 1.0]))[None].repeat(views, 1, 1)

    if n == 1:
        knn = torch.empty((1, 0), dtype=torch.long)
        edge_weight = torch.empty((1, 0))
    elif graph:
        knn = torch.tensor(
            [
                [1, 2],
                [0, 2],
                [0, 1],
                [0, 1],
            ],
            dtype=torch.long,
        )[:n]
        edge_weight = torch.ones((n, 2))
    else:
        knn = torch.zeros((n, 2), dtype=torch.long)
        edge_weight = torch.zeros((n, 2))

    return {
        "points": points,
        "projection_points": points.clone(),
        "face_ids": torch.arange(n, dtype=torch.long),
        "barycentric": torch.tensor([[0.2, 0.3, 0.5]]).repeat(n, 1),
        "normals": normals,
        "tangent_axis": tangent,
        "normal_tangent_ratio": ratios,
        "initial_sign": signs,
        "per_view_axes": per_view_axes,
        "per_view_weights": per_view_weights,
        "viewmats": viewmats,
        "intrinsics": intrinsics,
        "knn": knn,
        "edge_weight": edge_weight,
        "observed": torch.ones(n, dtype=torch.bool),
    }


def _run(**overrides: torch.Tensor) -> dict[str, object]:
    data = _base_inputs()
    data.update(overrides)
    return refine_global_tangent_sign_field(**data)


def _assert_tensors_on_device(value: object, device: torch.device) -> None:
    if isinstance(value, torch.Tensor):
        assert value.device == device
    elif isinstance(value, dict):
        for item in value.values():
            _assert_tensors_on_device(item, device)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_tensors_on_device(item, device)


def test_analytic_projected_cos2_unary_uses_exact_plus_minus_ratio_field() -> None:
    data = _base_inputs(n=1, views=1, ratio=0.7, graph=False)
    normal = data["normals"]
    tangent = data["tangent_axis"]
    ratio = data["normal_tangent_ratio"]
    d_plus = F.normalize(ratio[:, None] * normal + tangent, dim=-1)
    d_minus = F.normalize(ratio[:, None] * normal - tangent, dim=-1)
    evidence = d_plus.clone()
    data["per_view_axes"] = evidence[None]
    result = refine_global_tangent_sign_field(**data)

    plus_screen = _project(data["projection_points"], d_plus, data["viewmats"][0], data["intrinsics"][0])
    minus_screen = _project(data["projection_points"], d_minus, data["viewmats"][0], data["intrinsics"][0])
    evidence_screen = _project(data["projection_points"], evidence, data["viewmats"][0], data["intrinsics"][0])
    plus_unit = F.normalize(plus_screen, dim=-1)
    minus_unit = F.normalize(minus_screen, dim=-1)
    evidence_unit = F.normalize(evidence_screen, dim=-1)
    expected_plus = (plus_unit * evidence_unit).sum(dim=-1).square()
    expected_minus = (minus_unit * evidence_unit).sum(dim=-1).square()
    expected_h = expected_plus - expected_minus

    torch.testing.assert_close(result["unary"]["score_plus"][0], expected_plus.to(torch.float64), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(result["unary"]["score_minus"][0], expected_minus.to(torch.float64), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(result["h"], expected_h.to(torch.float64), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(result["ratio"], data["normal_tangent_ratio"])
    assert result["candidate_direction"].shape == (1, 3)


def test_coherent_supernode_flip_and_zero_new_severe_invariant() -> None:
    data = _base_inputs(n=3, ratio=0.2)
    d_minus = F.normalize(
        data["normal_tangent_ratio"][:, None] * data["normals"] - data["tangent_axis"], dim=-1
    )
    data["per_view_axes"] = d_minus[None]
    data["per_view_weights"] = torch.tensor([[1.0, 1.0, 0.0]])
    data["knn"] = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.long)
    data["edge_weight"] = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])

    result = refine_global_tangent_sign_field(**data)
    edge = result["edge_diagnostics"]

    assert bool(result["equality_mask"].any())
    assert bool(result["supernode_ids"][0] == result["supernode_ids"][1])
    assert bool(result["flip_mask"][0])
    assert bool(result["flip_mask"][1])
    assert not bool(result["flip_mask"][2])
    assert not bool(edge["new_severe_mask"].any())
    assert bool(result["report"]["final"]["mathematical_zero_new_severe_guard_verified"])
    equality = edge["equality_mask"]
    flip_variable = result["flip_variable"]
    assert torch.equal(flip_variable[edge["u"][equality]], flip_variable[edge["v"][equality]])


def test_no_op_when_evidence_and_pairwise_field_are_zero() -> None:
    data = _base_inputs(n=4, graph=False)
    data["per_view_axes"] = torch.zeros_like(data["per_view_axes"])
    data["per_view_weights"] = torch.zeros_like(data["per_view_weights"])
    result = refine_global_tangent_sign_field(**data)

    torch.testing.assert_close(result["candidate_sign"], data["initial_sign"])
    torch.testing.assert_close(result["candidate_direction"], result["baseline_direction"])
    assert not bool(result["flip_mask"].any())
    assert result["optimization"]["iterations"] == 0
    assert result["optimization"]["converged"]
    assert result["alpha"] == GLOBAL_SIGN_ALPHA_MULTIPLIER


def test_view_order_reversal_preserves_root_solution_and_unary_h() -> None:
    data = _base_inputs(n=4, views=3)
    data["per_view_axes"] = torch.stack(
        [
            data["per_view_axes"][0],
            -data["per_view_axes"][0],
            data["per_view_axes"][0] * torch.tensor([1.0, 0.95, 1.05, 1.0])[:, None],
        ]
    )
    data["per_view_weights"] = torch.tensor(
        [[1.0, 0.8, 1.1, 0.9], [0.6, 1.3, 0.7, 1.2], [1.4, 0.9, 0.8, 1.1]]
    )
    base = refine_global_tangent_sign_field(**data)
    reversed_data = dict(data)
    reversed_data["per_view_axes"] = data["per_view_axes"].flip(0)
    reversed_data["per_view_weights"] = data["per_view_weights"].flip(0)
    reversed_data["viewmats"] = data["viewmats"].flip(0)
    reversed_data["intrinsics"] = data["intrinsics"].flip(0)
    reversed_result = refine_global_tangent_sign_field(**reversed_data)

    torch.testing.assert_close(reversed_result["candidate_sign"], base["candidate_sign"])
    torch.testing.assert_close(reversed_result["candidate_direction"], base["candidate_direction"], atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(reversed_result["h"], base["h"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        reversed_result["unary"]["score_plus"], base["unary"]["score_plus"].flip(0), atol=0.0, rtol=0.0
    )
    assert reversed_result["alpha"] == base["alpha"]


def test_three_root_permutations_map_back_to_exact_sign_and_direction() -> None:
    data = _base_inputs(n=4, views=2, ratio=0.35)
    data["per_view_axes"] = torch.stack(
        [
            data["per_view_axes"][0],
            F.normalize(data["per_view_axes"][0] + torch.tensor([[0.0, 0.04, 0.0]] * 4), dim=-1),
        ]
    )
    base = refine_global_tangent_sign_field(**data)
    for permutation in (
        torch.tensor([2, 0, 3, 1]),
        torch.tensor([1, 3, 0, 2]),
        torch.tensor([3, 2, 1, 0]),
    ):
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        permuted = dict(data)
        for name in ("points", "projection_points", "face_ids", "barycentric", "normals", "tangent_axis", "normal_tangent_ratio", "initial_sign", "observed"):
            permuted[name] = data[name][permutation]
        permuted["per_view_axes"] = data["per_view_axes"][:, permutation]
        permuted["per_view_weights"] = data["per_view_weights"][:, permutation]
        permuted["knn"] = inverse[data["knn"][permutation]]
        permuted["edge_weight"] = data["edge_weight"][permutation]
        candidate = refine_global_tangent_sign_field(**permuted)

        mapped_sign = torch.empty_like(candidate["candidate_sign"])
        mapped_sign[permutation] = candidate["candidate_sign"]
        mapped_direction = torch.empty_like(candidate["candidate_direction"])
        mapped_direction[permutation] = candidate["candidate_direction"]
        torch.testing.assert_close(mapped_sign, base["candidate_sign"])
        dot = (mapped_direction * base["candidate_direction"]).sum(dim=-1).abs().clamp(0.0, 1.0)
        angle = torch.rad2deg(torch.acos(dot))
        assert float(angle.max()) <= 0.05
        mapped_h = torch.empty_like(candidate["h"])
        mapped_h[permutation] = candidate["h"]
        torch.testing.assert_close(mapped_h, base["h"], atol=0.0, rtol=0.0)


def test_duplicate_canonical_identity_is_rejected() -> None:
    data = _base_inputs(n=3)
    data["points"] = data["points"].clone()
    data["points"][1] = data["points"][0]
    data["face_ids"] = data["face_ids"].clone()
    data["face_ids"][1] = data["face_ids"][0]
    data["barycentric"] = data["barycentric"].clone()
    data["barycentric"][1] = data["barycentric"][0]
    with pytest.raises(ValueError, match="duplicate"):
        refine_global_tangent_sign_field(**data)


def test_api_is_keyword_only_semantically_independent_and_serializable() -> None:
    signature = inspect.signature(refine_global_tangent_sign_field)
    expected = {
        "points",
        "projection_points",
        "face_ids",
        "barycentric",
        "normals",
        "tangent_axis",
        "normal_tangent_ratio",
        "initial_sign",
        "per_view_axes",
        "per_view_weights",
        "viewmats",
        "intrinsics",
        "knn",
        "edge_weight",
        "observed",
    }
    assert set(signature.parameters) == expected
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in signature.parameters.values())
    assert not {"species", "region", "root_ids", "view_ids", "image_coordinates", "file_paths"}.intersection(expected)
    assert GLOBAL_SIGN_ALPHA_MULTIPLIER == 0.5
    assert math.isclose(COS45, math.cos(math.radians(45.0)))
    assert SEVERE_DOT_THRESHOLD == -COS45
    assert MAX_BLOCK_STEPS == 256

    result = _run()
    canonical_order = result["canonical_order"]
    canonical_rank = result["canonical_rank"]
    assert isinstance(canonical_order, torch.Tensor)
    assert isinstance(canonical_rank, torch.Tensor)
    torch.testing.assert_close(
        canonical_rank[canonical_order],
        torch.arange(canonical_rank.numel(), dtype=torch.long),
    )
    json.dumps(result["report"], allow_nan=False, sort_keys=True)
    _assert_tensors_on_device(result, torch.device("cpu"))

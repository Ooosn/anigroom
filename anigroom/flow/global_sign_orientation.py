"""Deterministic global sign refinement for directed tangent fields.

The public entry point in this module is intentionally semantic-free.  It
accepts only the geometry, the supplied multiview evidence, and a fixed KNN
graph.  The discrete optimization is performed in a canonical CPU view so
that root-array permutations and view-row permutations cannot change the
answer.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import torch

from .direction_geometry import parallel_transport_vectors


# Public specification constants.  Keep the aliases stable because callers
# may record the exact discrete contract alongside an output artifact.
EPS = 1.0e-8
COS45 = float(math.cos(math.radians(45.0)))
SEVERE_DOT_THRESHOLD = -COS45
GLOBAL_SIGN_ALPHA_MULTIPLIER = 0.5
ALPHA_MULTIPLIER = GLOBAL_SIGN_ALPHA_MULTIPLIER
MAX_BLOCK_STEPS = 256
GLOBAL_SIGN_MAX_BLOCK_STEPS = MAX_BLOCK_STEPS
GLOBAL_SIGN_MAX_STEPS = MAX_BLOCK_STEPS
OBJECTIVE_INVARIANCE_TOL = 1.0e-8
PROJECTED_VECTOR_EPS = 1.0e-6


__all__ = [
    "EPS",
    "COS45",
    "SEVERE_DOT_THRESHOLD",
    "GLOBAL_SIGN_ALPHA_MULTIPLIER",
    "ALPHA_MULTIPLIER",
    "MAX_BLOCK_STEPS",
    "GLOBAL_SIGN_MAX_BLOCK_STEPS",
    "GLOBAL_SIGN_MAX_STEPS",
    "refine_global_tangent_sign_field",
]


def _require_tensor_inputs(tensors: dict[str, object]) -> None:
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all global sign orientation inputs must be torch.Tensor values")


def _require_floating(name: str, value: torch.Tensor) -> None:
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.is_complex():
        raise TypeError(f"{name} must be a real tensor")


def _require_integer(name: str, value: torch.Tensor) -> None:
    if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
        raise TypeError(f"{name} must be an integer tensor")


def _validate_inputs(
    *,
    points: torch.Tensor,
    projection_points: torch.Tensor,
    face_ids: torch.Tensor,
    barycentric: torch.Tensor,
    normals: torch.Tensor,
    tangent_axis: torch.Tensor,
    normal_tangent_ratio: torch.Tensor,
    initial_sign: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    knn: torch.Tensor,
    edge_weight: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[int, int, int, torch.device, torch.dtype]:
    tensors: dict[str, object] = {
        "points": points,
        "projection_points": projection_points,
        "face_ids": face_ids,
        "barycentric": barycentric,
        "normals": normals,
        "tangent_axis": tangent_axis,
        "normal_tangent_ratio": normal_tangent_ratio,
        "initial_sign": initial_sign,
        "per_view_axes": per_view_axes,
        "per_view_weights": per_view_weights,
        "viewmats": viewmats,
        "intrinsics": intrinsics,
        "knn": knn,
        "edge_weight": edge_weight,
        "observed": observed,
    }
    _require_tensor_inputs(tensors)

    if points.ndim != 2 or tuple(points.shape[1:]) != (3,):
        raise ValueError("points must have shape [N, 3]")
    n = int(points.shape[0])
    expected_root_shapes = {
        "projection_points": (n, 3),
        "barycentric": (n, 3),
        "normals": (n, 3),
        "tangent_axis": (n, 3),
        "normal_tangent_ratio": (n,),
        "initial_sign": (n,),
        "face_ids": (n,),
        "observed": (n,),
    }
    for name, shape in expected_root_shapes.items():
        if tuple(tensors[name].shape) != shape:  # type: ignore[union-attr]
            raise ValueError(f"{name} must have shape {shape}")

    if per_view_axes.ndim != 3 or tuple(per_view_axes.shape[2:]) != (3,):
        raise ValueError("per_view_axes must have shape [V, N, 3]")
    v = int(per_view_axes.shape[0])
    if int(per_view_axes.shape[1]) != n:
        raise ValueError("per_view_axes must have the same N as points")
    if tuple(per_view_weights.shape) != (v, n):
        raise ValueError("per_view_weights must have shape [V, N]")
    if tuple(viewmats.shape) != (v, 4, 4):
        raise ValueError("viewmats must have shape [V, 4, 4]")
    if tuple(intrinsics.shape) != (v, 3, 3):
        raise ValueError("intrinsics must have shape [V, 3, 3]")
    if knn.ndim != 2 or int(knn.shape[0]) != n:
        raise ValueError("knn must have shape [N, K]")
    k = int(knn.shape[1])
    if tuple(edge_weight.shape) != (n, k):
        raise ValueError("edge_weight must have the same shape as knn")

    floating_names = (
        "points",
        "projection_points",
        "barycentric",
        "normals",
        "tangent_axis",
        "normal_tangent_ratio",
        "per_view_axes",
        "per_view_weights",
        "viewmats",
        "intrinsics",
        "edge_weight",
    )
    for name in floating_names:
        _require_floating(name, tensors[name])  # type: ignore[arg-type]
    _require_integer("face_ids", face_ids)
    _require_integer("knn", knn)
    if initial_sign.is_complex() or initial_sign.dtype == torch.bool:
        raise TypeError("initial_sign must be a real +/-1 tensor")
    if not initial_sign.is_floating_point() and not initial_sign.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("initial_sign must be a real +/-1 tensor")
    if observed.dtype != torch.bool:
        raise TypeError("observed must be a torch.bool tensor")

    device = points.device
    if any(value.device != device for value in tensors.values()):  # type: ignore[union-attr]
        raise ValueError("all global sign orientation inputs must be on the same device")
    for name in floating_names + ("initial_sign",):
        value = tensors[name]
        if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):  # type: ignore[arg-type]
            raise ValueError(f"{name} contains non-finite values")
    if knn.numel():
        if bool((knn < 0).any()) or bool((knn >= n).any()):
            raise ValueError("knn contains an out-of-range root index")
    if normal_tangent_ratio.numel() and bool((normal_tangent_ratio < 0).any()):
        raise ValueError("normal_tangent_ratio must be non-negative")
    if per_view_weights.numel() and bool((per_view_weights < 0).any()):
        raise ValueError("per_view_weights must be non-negative")
    if edge_weight.numel() and bool((edge_weight < 0).any()):
        raise ValueError("edge_weight must be non-negative")

    sign_values = initial_sign.detach().cpu().numpy()
    if not np.all((sign_values == 1) | (sign_values == -1)):
        raise ValueError("initial_sign must contain only +/-1 values")

    if n:
        normal_length = torch.linalg.vector_norm(normals, dim=-1)
        tangent_length = torch.linalg.vector_norm(tangent_axis, dim=-1)
        if bool((normal_length <= EPS).any()):
            raise ValueError("normals must contain non-zero vectors")
        if bool((tangent_length <= EPS).any()):
            raise ValueError("tangent_axis must contain non-zero vectors")

    floating_dtypes = [
        value.dtype
        for name, value in tensors.items()
        if name != "initial_sign" and value.is_floating_point()
    ]
    dtype = torch.float64 if any(value == torch.float64 for value in floating_dtypes) else torch.float32
    return n, v, k, device, dtype


def _as_cpu_numpy(value: torch.Tensor, *, dtype: np.dtype[Any]) -> np.ndarray:
    cpu_value = value.detach().cpu()
    if cpu_value.dtype == torch.bfloat16:
        cpu_value = cpu_value.to(dtype=torch.float32)
    return np.ascontiguousarray(cpu_value.numpy(), dtype=dtype)


def _normalize_float32(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        length = np.linalg.norm(value, axis=-1, keepdims=True)
        result = value / np.maximum(length, np.float32(EPS))
    if not np.isfinite(result).all():
        raise ValueError("vector normalization produced non-finite values")
    if np.any(length <= np.float32(EPS)):
        raise ValueError("vector normalization received a zero vector")
    return np.ascontiguousarray(result, dtype=np.float32)


def _stats(value: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(value, dtype=np.float64).reshape(-1)
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    quantiles = np.quantile(selected, [0.50, 0.90, 0.95, 0.99])
    return {
        "count": int(selected.size),
        "mean": float(selected.mean()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "max": float(selected.max()),
    }


def _masked_quantile(value: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(value, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return 0.0
    return float(np.quantile(selected, 0.90))


def _identity_key(
    point: np.ndarray,
    face_id: np.int64 | int,
    bary: np.ndarray,
) -> tuple[float | int, ...]:
    return (
        float(np.float32(point[0])),
        float(np.float32(point[1])),
        float(np.float32(point[2])),
        int(face_id),
        float(np.float32(bary[0])),
        float(np.float32(bary[1])),
        float(np.float32(bary[2])),
    )


def _canonical_identity(
    points: np.ndarray,
    face_ids: np.ndarray,
    barycentric: np.ndarray,
) -> dict[str, Any]:
    n = int(points.shape[0])
    keys = [
        _identity_key(point, face_id, bary)
        for point, face_id, bary in zip(points, face_ids, barycentric)
    ]
    groups: dict[tuple[float | int, ...], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    duplicate_groups = [
        {"identity": list(key), "array_indices": values}
        for key, values in sorted(groups.items())
        if len(values) > 1
    ]
    if duplicate_groups:
        raise ValueError(
            "canonical root identity is not unique; duplicate groups: "
            + json.dumps(duplicate_groups, sort_keys=True)
        )
    order = np.asarray(sorted(range(n), key=lambda index: keys[index]), dtype=np.int64)
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n, dtype=np.int64)
    canonical_keys = [keys[int(index)] for index in order.tolist()]
    return {
        "order": order,
        "inverse": inverse,
        "keys": keys,
        "canonical_keys": canonical_keys,
        "report": {
            "identity_fields": [
                "points_float32",
                "face_ids_int64",
                "barycentric_float32",
            ],
            "primary_order": "exact float32 points lexicographic order",
            "secondary_order": "face_ids then exact float32 barycentric",
            "unique": True,
            "duplicate_group_count": 0,
            "duplicate_groups": [],
            "root_index_used_in_identity_or_order": False,
        },
    }


def _geometry_row_key(row: np.ndarray) -> tuple[float, ...]:
    return tuple(float(np.float32(value)) for value in np.asarray(row, dtype=np.float32).tolist())


def _identity_multiset_hash(identity: dict[str, Any], members: np.ndarray) -> str:
    ordered = tuple(identity["canonical_keys"][int(member)] for member in members.tolist())
    return hashlib.sha256(repr(ordered).encode("utf-8")).hexdigest()


def _build_graph(
    *,
    knn: np.ndarray,
    edge_weight: np.ndarray,
    identity: dict[str, Any],
    observed: np.ndarray,
) -> dict[str, Any]:
    """Build the maximum-weight unique undirected graph in canonical order."""

    n = int(knn.shape[0])
    inverse = np.asarray(identity["inverse"], dtype=np.int64)
    pair_weight: dict[tuple[int, int], float] = {}
    directed_positive_count = 0
    for source in range(n):
        source_canonical = int(inverse[source])
        for slot in range(int(knn.shape[1])):
            weight = float(np.float32(edge_weight[source, slot]))
            if weight <= 0.0:
                continue
            directed_positive_count += 1
            destination = int(knn[source, slot])
            destination_canonical = int(inverse[destination])
            if source_canonical == destination_canonical:
                raise ValueError("knn contains a positive self edge")
            pair = (
                min(source_canonical, destination_canonical),
                max(source_canonical, destination_canonical),
            )
            previous = pair_weight.get(pair)
            if previous is None or weight > previous:
                pair_weight[pair] = weight

    pairs = sorted(pair_weight)
    if pairs:
        u = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        v = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
        weight = np.asarray([pair_weight[pair] for pair in pairs], dtype=np.float32)
    else:
        u = np.empty(0, dtype=np.int64)
        v = np.empty(0, dtype=np.int64)
        weight = np.empty(0, dtype=np.float32)
    observed_canonical = np.asarray(observed, dtype=bool)[np.asarray(identity["order"], dtype=np.int64)]
    observed_edge = observed_canonical[u] & observed_canonical[v]
    degree = np.zeros(n, dtype=np.int64)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for first, second, value in zip(u.tolist(), v.tolist(), weight.tolist()):
        degree[int(first)] += 1
        degree[int(second)] += 1
        adjacency[int(first)].append((int(second), float(value)))
        adjacency[int(second)].append((int(first), float(value)))
    for node, row in enumerate(adjacency):
        row.sort(key=lambda item: (identity["canonical_keys"][item[0]], float(item[1])))
        adjacency[node] = row
    return {
        "u": u,
        "v": v,
        "weight": weight,
        "observed": observed_edge,
        "degree": degree,
        "adjacency": adjacency,
        "report": {
            "input_directed_positive_edge_count": int(directed_positive_count),
            "unique_undirected_edge_count": int(u.size),
            "deduplication": "maximum available directed edge weight per unordered pair",
            "canonical_order": "endpoint pairs sorted by canonical root identity",
            "observed_unique_edge_count": int(observed_edge.sum()),
        },
    }


def _transport_pairs(
    direction: np.ndarray,
    normals: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    if u.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    source_direction = torch.from_numpy(np.ascontiguousarray(direction[v], dtype=np.float32))
    source_normals = torch.from_numpy(np.ascontiguousarray(normals[v], dtype=np.float32))
    target_normals = torch.from_numpy(np.ascontiguousarray(normals[u], dtype=np.float32))
    with torch.no_grad():
        transported = parallel_transport_vectors(source_direction, source_normals, target_normals)
    result = transported.detach().cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("parallel transport produced non-finite vectors")
    return np.ascontiguousarray(result, dtype=np.float32)


def _edge_dots(
    direction: np.ndarray,
    normals: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    transported = _transport_pairs(direction, normals, u, v)
    with np.errstate(invalid="ignore"):
        dots = np.sum(direction[u] * transported, axis=-1).astype(np.float32)
    return np.clip(dots, -1.0, 1.0).astype(np.float32, copy=False)


def _project_direction(
    points: np.ndarray,
    directions: np.ndarray,
    viewmat: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    rotation = np.asarray(viewmat[:3, :3], dtype=np.float32)
    translation = np.asarray(viewmat[:3, 3], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        camera_points = points @ rotation.T + translation[None]
        camera_directions = directions @ rotation.T
        depth = np.maximum(camera_points[:, 2], np.float32(PROJECTED_VECTOR_EPS))
        denominator = depth * depth
        fx = np.float32(intrinsic[0, 0])
        fy = np.float32(intrinsic[1, 1])
        screen_x = fx * (
            camera_directions[:, 0] * depth - camera_points[:, 0] * camera_directions[:, 2]
        ) / denominator
        screen_y = fy * (
            camera_directions[:, 1] * depth - camera_points[:, 1] * camera_directions[:, 2]
        ) / denominator
    return np.stack((screen_x, screen_y), axis=-1).astype(np.float32, copy=False)


def _compute_unary(
    *,
    points: np.ndarray,
    normals: np.ndarray,
    axis: np.ndarray,
    ratio: np.ndarray,
    per_view_axes: np.ndarray,
    per_view_weights: np.ndarray,
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
    observed: np.ndarray,
    identity: dict[str, Any],
) -> dict[str, Any]:
    n = int(points.shape[0])
    v = int(per_view_axes.shape[0])
    d_plus = _normalize_float32(ratio[:, None] * normals + axis)
    d_minus = _normalize_float32(ratio[:, None] * normals - axis)
    score_plus = np.zeros((v, n), dtype=np.float64)
    score_minus = np.zeros((v, n), dtype=np.float64)
    valid_rows = np.zeros((v, n), dtype=bool)

    # The API deliberately has no view IDs.  A content signature gives the
    # float64 reduction a deterministic order while remaining unchanged when
    # callers reverse the view rows.
    view_signatures: list[tuple[str, int]] = []
    for slot in range(v):
        digest = hashlib.sha256()
        for value in (
            per_view_axes[slot],
            per_view_weights[slot],
            viewmats[slot],
            intrinsics[slot],
        ):
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        view_signatures.append((digest.hexdigest(), slot))
    view_order = [slot for _, slot in sorted(view_signatures, key=lambda item: item[0])]

    for slot in range(v):
        plus_screen = _project_direction(points, d_plus, viewmats[slot], intrinsics[slot])
        minus_screen = _project_direction(points, d_minus, viewmats[slot], intrinsics[slot])
        evidence_screen = _project_direction(
            points,
            per_view_axes[slot],
            viewmats[slot],
            intrinsics[slot],
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            plus_norm = np.linalg.norm(plus_screen, axis=-1)
            minus_norm = np.linalg.norm(minus_screen, axis=-1)
            evidence_norm = np.linalg.norm(evidence_screen, axis=-1)
            weight = per_view_weights[slot].astype(np.float64, copy=False)
            valid = (
                observed
                & (weight > 0.0)
                & np.isfinite(weight)
                & np.isfinite(plus_norm)
                & np.isfinite(minus_norm)
                & np.isfinite(evidence_norm)
                & (plus_norm > np.float32(PROJECTED_VECTOR_EPS))
                & (minus_norm > np.float32(PROJECTED_VECTOR_EPS))
                & (evidence_norm > np.float32(PROJECTED_VECTOR_EPS))
            )
            plus_unit = plus_screen / np.maximum(plus_norm[:, None], np.float32(PROJECTED_VECTOR_EPS))
            minus_unit = minus_screen / np.maximum(minus_norm[:, None], np.float32(PROJECTED_VECTOR_EPS))
            evidence_unit = evidence_screen / np.maximum(
                evidence_norm[:, None], np.float32(PROJECTED_VECTOR_EPS)
            )
            # The source diagnostic forms the dot in float32 and casts that
            # result to float64 before applying the score weight.
            plus_dot = np.sum(plus_unit * evidence_unit, axis=-1).astype(np.float64)
            minus_dot = np.sum(minus_unit * evidence_unit, axis=-1).astype(np.float64)
            score_plus[slot] = np.where(valid, weight * plus_dot * plus_dot, 0.0)
            score_minus[slot] = np.where(valid, weight * minus_dot * minus_dot, 0.0)
        valid_rows[slot] = valid

    vote = score_plus - score_minus
    h = np.sum(vote[view_order], axis=0, dtype=np.float64) if v else np.zeros(n, dtype=np.float64)
    total_score = (
        np.sum((score_plus + score_minus)[view_order], axis=0, dtype=np.float64)
        if v
        else np.zeros(n, dtype=np.float64)
    )
    signed_margin = np.divide(
        h,
        np.maximum(total_score, float(EPS)),
        out=np.zeros_like(h),
        where=total_score > float(EPS),
    )
    normalized_margin = np.abs(signed_margin)
    vote_abs_sum = (
        np.sum(np.abs(vote)[view_order], axis=0, dtype=np.float64)
        if v
        else np.zeros(n, dtype=np.float64)
    )
    vote_concentration = np.divide(
        np.max(np.abs(vote), axis=0) if v else np.zeros(n, dtype=np.float64),
        np.maximum(vote_abs_sum, float(EPS)),
        out=np.zeros(n, dtype=np.float64),
        where=vote_abs_sum > float(EPS),
    )
    vote_coherence = np.divide(
        np.abs(h),
        np.maximum(vote_abs_sum, float(EPS)),
        out=np.zeros_like(h),
        where=vote_abs_sum > float(EPS),
    )
    vote_count = np.sum(valid_rows, axis=0, dtype=np.int64)
    if v:
        dominant_rank = np.argmax(np.abs(vote[view_order]), axis=0).astype(np.int64)
        dominant_slot = np.asarray(view_order, dtype=np.int64)[dominant_rank]
    else:
        dominant_slot = np.full(n, -1, dtype=np.int64)
    dominant_slot = np.where(vote_abs_sum > float(EPS), dominant_slot, -1)
    positive_vote_count = np.sum(vote > 0.0, axis=0, dtype=np.int64)
    preferred_sign = np.sign(h).astype(np.int8)
    tie_resolved_sign = np.where(preferred_sign >= 0, 1, -1).astype(np.int8)
    diagnostic = {
        "score_definition": (
            "per_view_weight * (normalized projected +/- candidate screen vector "
            "dot normalized projected evidence screen vector)^2"
        ),
        "h_definition": "sum over views of score_plus - score_minus in canonical content order",
        "preferred_sign_definition": "sign(h); zero h resolves to +1 only for diagnostic metadata",
        "normalized_margin_definition": "abs(h) / max(score_plus + score_minus, eps)",
        "signed_margin_definition": "h / max(score_plus + score_minus, eps)",
        "view_order_definition": "content-sorted view rows; no view ID is consulted",
    }
    return {
        "d_plus": d_plus,
        "d_minus": d_minus,
        "score_plus": score_plus,
        "score_minus": score_minus,
        "valid_rows": valid_rows,
        "h": h,
        "total_score": total_score,
        "signed_margin": signed_margin,
        "normalized_margin": normalized_margin,
        "vote": vote,
        "vote_abs_sum": vote_abs_sum,
        "vote_concentration": vote_concentration,
        "vote_coherence": vote_coherence,
        "vote_count": vote_count,
        "dominant_slot": dominant_slot,
        "positive_vote_count": positive_vote_count,
        "preferred_sign": preferred_sign,
        "tie_resolved_sign": tie_resolved_sign,
        "view_order": np.asarray(view_order, dtype=np.int64),
        "diagnostic": diagnostic,
    }


def _pairwise_field(
    root_count: int,
    u: np.ndarray,
    v: np.ndarray,
    coupling: np.ndarray,
    signs: np.ndarray,
) -> np.ndarray:
    field = np.zeros(root_count, dtype=np.float64)
    if u.size:
        np.add.at(field, u, np.asarray(coupling, dtype=np.float64) * signs[v])
        np.add.at(field, v, np.asarray(coupling, dtype=np.float64) * signs[u])
    return field


def _objective(
    signs: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    coupling: np.ndarray,
    h: np.ndarray,
    alpha: float,
) -> tuple[float, float, float]:
    pair_value = float(np.sum(np.asarray(coupling, dtype=np.float64) * signs[u] * signs[v], dtype=np.float64))
    unary_value = float(alpha) * float(np.sum(np.asarray(h, dtype=np.float64) * signs, dtype=np.float64))
    return pair_value + unary_value, pair_value, unary_value


def _build_supernodes(
    *,
    root_count: int,
    equality_u: np.ndarray,
    equality_v: np.ndarray,
    identity: dict[str, Any],
    geometry_rows: np.ndarray,
) -> dict[str, Any]:
    parent = np.arange(root_count, dtype=np.int64)

    def find(value: int) -> int:
        root = int(value)
        while int(parent[root]) != root:
            root = int(parent[root])
        node = int(value)
        while int(parent[node]) != node:
            next_node = int(parent[node])
            parent[node] = root
            node = next_node
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if first_root < second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    equality_pairs = sorted(
        (int(first), int(second))
        for first, second in zip(equality_u.tolist(), equality_v.tolist())
    )
    for first, second in equality_pairs:
        union(first, second)
    representative = np.asarray([find(index) for index in range(root_count)], dtype=np.int64)
    grouped: dict[int, list[int]] = {}
    for index, rep in enumerate(representative.tolist()):
        grouped.setdefault(int(rep), []).append(index)
    raw_blocks = [np.asarray(grouped[rep], dtype=np.int64) for rep in sorted(grouped)]

    metadata: list[dict[str, Any]] = []
    for members in raw_blocks:
        ordered = np.asarray(
            sorted(members.tolist(), key=lambda member: identity["canonical_keys"][int(member)]),
            dtype=np.int64,
        )
        metadata.append(
            {
                "members": ordered,
                "geometry_key": identity["canonical_keys"][int(ordered[0])],
                "sorted_rows_hash": _identity_multiset_hash(identity, ordered),
                "root_count": int(ordered.size),
                "representative": int(representative[int(ordered[0])]),
            }
        )
    metadata.sort(
        key=lambda item: (
            tuple(item["geometry_key"]),
            str(item["sorted_rows_hash"]),
            int(item["root_count"]),
        )
    )
    blocks = [np.asarray(item["members"], dtype=np.int64) for item in metadata]
    block_ordinal = np.empty(root_count, dtype=np.int64)
    for ordinal, members in enumerate(blocks):
        block_ordinal[members] = ordinal
    representative_for_block = np.asarray(
        [int(item["representative"]) for item in metadata], dtype=np.int64
    )
    return {
        "representative": representative,
        "blocks": blocks,
        "block_ordinal": block_ordinal,
        "representative_for_block": representative_for_block,
        "metadata": metadata,
        "sizes": np.asarray([block.size for block in blocks], dtype=np.int64),
        "geometry_rows": geometry_rows,
        "identity_keys": identity["canonical_keys"],
    }


def _block_coordinate_ascent(
    *,
    current_sign: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    coupling: np.ndarray,
    adjacency: list[list[tuple[int, float]]],
    h: np.ndarray,
    alpha: float,
    supernodes: dict[str, Any],
) -> dict[str, Any]:
    blocks = [np.asarray(block, dtype=np.int64) for block in supernodes["blocks"]]
    metadata = list(supernodes["metadata"])
    root_count = int(current_sign.size)
    block_count = len(blocks)
    signs = np.asarray(current_sign, dtype=np.int8).copy()
    flip_variables = np.ones(root_count, dtype=np.int8)
    local_pair_field = _pairwise_field(root_count, u, v, coupling, signs)
    initial_objective, initial_pair, initial_unary = _objective(
        signs, u, v, coupling, h, alpha
    )
    running_objective = float(initial_objective)

    root_to_block = np.asarray(supernodes["block_ordinal"], dtype=np.int64)
    internal_pair_value = np.zeros(block_count, dtype=np.float64)
    for first, second, edge_value in zip(u.tolist(), v.tolist(), coupling.tolist()):
        first = int(first)
        second = int(second)
        first_block = int(root_to_block[first])
        if first_block == int(root_to_block[second]):
            internal_pair_value[first_block] += float(edge_value) * signs[first] * signs[second]

    geometry_rows = np.asarray(supernodes["geometry_rows"], dtype=np.float32)

    def dynamic_tie_hash(block_index: int) -> str:
        records: list[tuple[object, ...]] = []
        members = blocks[block_index]
        for member_value in members.tolist():
            member = int(member_value)
            records.append(
                (
                    "member",
                    tuple(supernodes["identity_keys"][member]),
                    float(h[member]),
                    int(signs[member]),
                    float(local_pair_field[member]),
                )
            )
            for neighbor, edge_value in adjacency[member]:
                neighbor = int(neighbor)
                neighbor_block = int(root_to_block[neighbor])
                if neighbor_block == block_index:
                    continue
                neighbor_metadata = metadata[neighbor_block]
                records.append(
                    (
                        "edge",
                        tuple(supernodes["identity_keys"][member]),
                        tuple(neighbor_metadata["geometry_key"]),
                        str(neighbor_metadata["sorted_rows_hash"]),
                        float(edge_value),
                        int(signs[neighbor]),
                    )
                )
        records.sort()
        return hashlib.sha256(repr(records).encode("utf-8")).hexdigest()

    accepted_records: list[dict[str, Any]] = []
    full_delta_stats: list[dict[str, Any]] = []
    accepted_deltas: list[float] = []
    max_delta_formula_error = 0.0
    converged = False
    unresolved_duplicate_tie_count = 0

    for step in range(1, MAX_BLOCK_STEPS + 1):
        deltas = np.empty(block_count, dtype=np.float64)
        for block_index, members in enumerate(blocks):
            ordered_members = np.asarray(
                sorted(
                    (int(member) for member in members.tolist()),
                    key=lambda member: (
                        tuple(supernodes["identity_keys"][member]),
                        float(h[member]),
                        int(signs[member]),
                        float(local_pair_field[member]),
                    ),
                ),
                dtype=np.int64,
            )
            member_pair_field = float(
                np.sum(signs[ordered_members] * local_pair_field[ordered_members], dtype=np.float64)
            )
            member_unary_field = float(
                np.sum(h[ordered_members] * signs[ordered_members], dtype=np.float64)
            )
            deltas[block_index] = (
                -2.0 * member_pair_field
                -2.0 * float(alpha) * member_unary_field
                +4.0 * float(internal_pair_value[block_index])
            )

        positive = np.isfinite(deltas) & (deltas > 0.0)
        if not positive.any():
            full_delta_stats.append(
                {
                    "step": int(step),
                    "block_count": int(block_count),
                    "positive_delta_count": 0,
                    "max_delta": float(np.max(deltas)) if deltas.size else 0.0,
                    "min_delta": float(np.min(deltas)) if deltas.size else 0.0,
                    "termination_check": True,
                }
            )
            converged = True
            break

        max_delta = float(np.max(deltas[positive]))
        tied = np.flatnonzero(positive & (deltas == max_delta)).astype(np.int64)
        tie_hashes = {
            int(block_index): dynamic_tie_hash(int(block_index))
            for block_index in tied.tolist()
        }
        tie_sort = sorted(
            tied.tolist(),
            key=lambda block_index: (
                tuple(metadata[block_index]["geometry_key"]),
                str(metadata[block_index]["sorted_rows_hash"]),
                tie_hashes[int(block_index)],
            ),
        )
        if len(tie_sort) > 1:
            first_key = (
                tuple(metadata[tie_sort[0]]["geometry_key"]),
                str(metadata[tie_sort[0]]["sorted_rows_hash"]),
                tie_hashes[tie_sort[0]],
            )
            if any(
                (
                    tuple(metadata[index]["geometry_key"]),
                    str(metadata[index]["sorted_rows_hash"]),
                    tie_hashes[index],
                )
                == first_key
                for index in tie_sort[1:]
            ):
                unresolved_duplicate_tie_count += 1
        selected_block = int(tie_sort[0])
        trial_signs = signs.copy()
        trial_signs[blocks[selected_block]] = -trial_signs[blocks[selected_block]]
        trial_objective = float(
            _objective(trial_signs, u, v, coupling, h, alpha)[0]
        )
        exact_delta = float(trial_objective - running_objective)
        delta_error = abs(exact_delta - max_delta)
        max_delta_formula_error = max(max_delta_formula_error, delta_error)
        if delta_error > OBJECTIVE_INVARIANCE_TOL:
            raise RuntimeError(
                f"full block delta verification failed at step {step}: "
                f"formula={max_delta:.17g}, direct={exact_delta:.17g}, error={delta_error:.3g}"
            )
        full_delta_stats.append(
            {
                "step": int(step),
                "block_count": int(block_count),
                "positive_delta_count": int(positive.sum()),
                "max_delta": max_delta,
                "min_delta": float(np.min(deltas)),
                "selected_delta": max_delta,
                "exact_delta": exact_delta,
                "exact_delta_error": delta_error,
                "exact_tie_count": int(tied.size),
                "termination_check": False,
            }
        )
        old_signs = signs[blocks[selected_block]].copy()
        signs[blocks[selected_block]] = -old_signs
        flip_variables[blocks[selected_block]] = -flip_variables[blocks[selected_block]]
        running_objective = trial_objective
        accepted_deltas.append(max_delta)
        accepted_records.append(
            {
                "step": int(step),
                "delta": max_delta,
                "exact_delta": exact_delta,
                "tie_count": int(tied.size),
                "geometry_key": list(metadata[selected_block]["geometry_key"]),
                "sorted_rows_hash": str(metadata[selected_block]["sorted_rows_hash"]),
                "dynamic_tie_hash": tie_hashes[selected_block],
                "root_count": int(blocks[selected_block].size),
            }
        )
        for member_index, member_value in enumerate(blocks[selected_block].tolist()):
            member = int(member_value)
            old_sign = int(old_signs[member_index])
            for neighbor, edge_value in adjacency[member]:
                local_pair_field[int(neighbor)] += -2.0 * float(edge_value) * old_sign

    final_objective, final_pair, final_unary = _objective(signs, u, v, coupling, h, alpha)
    changed = signs != np.asarray(current_sign, dtype=np.int8)
    changed_blocks = np.asarray(
        [int(flip_variables[members[0]]) < 0 for members in blocks],
        dtype=bool,
    )
    return {
        "signs": signs,
        "changed": changed,
        "flip_variables": flip_variables,
        "initial_objective": float(initial_objective),
        "initial_pairwise_objective": float(initial_pair),
        "initial_unary_objective": float(initial_unary),
        "objective": float(final_objective),
        "pairwise_objective": float(final_pair),
        "unary_objective": float(final_unary),
        "objective_gain": float(final_objective - initial_objective),
        "iterations": int(len(accepted_records)),
        "converged": bool(converged),
        "max_block_steps": int(MAX_BLOCK_STEPS),
        "safe_max_reached": bool(not converged and len(accepted_records) >= MAX_BLOCK_STEPS),
        "termination_reason": "no_positive_delta" if converged else "safe_max_reached",
        "accepted_block_deltas": accepted_deltas,
        "accepted_step_records": accepted_records,
        "full_delta_stats": full_delta_stats,
        "computed_exact_full_delta_for_every_supernode": True,
        "max_delta_formula_error": float(max_delta_formula_error),
        "unresolved_duplicate_tie_count": int(unresolved_duplicate_tie_count),
        "strictly_increasing_flips_only": True,
        "best_improvement_only": True,
        "initial_flip_variable_value": 1,
        "supernode_count": int(block_count),
        "supernode_size_stats": _stats(supernodes["sizes"]),
        "changed_block_count": int(changed_blocks.sum()),
        "changed_root_count": int(changed.sum()),
        "final_flip_variable_changed_block_count": int(changed_blocks.sum()),
        "final_flip_variable_minus_one_root_count": int(np.sum(flip_variables < 0)),
        "final_local_pair_field": local_pair_field,
        "final_local_field": local_pair_field + float(alpha) * np.asarray(h, dtype=np.float64),
    }


def _to_device_tensor(
    value: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=dtype)


def _root_order_out(value: np.ndarray, order: np.ndarray) -> np.ndarray:
    result = np.empty_like(value)
    result[order] = value
    return result


def _root_matrix_out(value: np.ndarray, order: np.ndarray) -> np.ndarray:
    result = np.empty_like(value)
    result[:, order] = value
    return result


def refine_global_tangent_sign_field(
    *,
    points: torch.Tensor,
    projection_points: torch.Tensor,
    face_ids: torch.Tensor,
    barycentric: torch.Tensor,
    normals: torch.Tensor,
    tangent_axis: torch.Tensor,
    normal_tangent_ratio: torch.Tensor,
    initial_sign: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    knn: torch.Tensor,
    edge_weight: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, object]:
    """Refine tangent signs with a canonical, guarded block objective.

    The candidate field is

    ``normalize(normal_tangent_ratio * normals + candidate_sign * tangent_axis)``.

    The unary term compares the projected ``+`` and ``-`` versions of that
    exact fixed-ratio field with each supplied per-view axis using squared
    cosine agreement.  Pairwise terms use a unique undirected KNN graph and
    minimal-rotation parallel transport between surface-normal frames.
    """

    n, view_count, neighbor_count, device, _ = _validate_inputs(
        points=points,
        projection_points=projection_points,
        face_ids=face_ids,
        barycentric=barycentric,
        normals=normals,
        tangent_axis=tangent_axis,
        normal_tangent_ratio=normal_tangent_ratio,
        initial_sign=initial_sign,
        per_view_axes=per_view_axes,
        per_view_weights=per_view_weights,
        viewmats=viewmats,
        intrinsics=intrinsics,
        knn=knn,
        edge_weight=edge_weight,
        observed=observed,
    )

    points_np = _as_cpu_numpy(points, dtype=np.dtype(np.float32))
    projection_points_np = _as_cpu_numpy(
        projection_points, dtype=np.dtype(np.float32)
    )
    face_ids_np = _as_cpu_numpy(face_ids, dtype=np.dtype(np.int64))
    barycentric_np = _as_cpu_numpy(barycentric, dtype=np.dtype(np.float32))
    normals_np = _normalize_float32(_as_cpu_numpy(normals, dtype=np.dtype(np.float32)))
    axis_np = _normalize_float32(_as_cpu_numpy(tangent_axis, dtype=np.dtype(np.float32)))
    ratio_np = _as_cpu_numpy(normal_tangent_ratio, dtype=np.dtype(np.float32))
    initial_sign_np = _as_cpu_numpy(initial_sign, dtype=np.dtype(np.float32))
    initial_sign_np = np.where(initial_sign_np > 0.0, 1, -1).astype(np.int8)
    per_view_axes_np = _as_cpu_numpy(per_view_axes, dtype=np.dtype(np.float32))
    per_view_weights_np = _as_cpu_numpy(per_view_weights, dtype=np.dtype(np.float32))
    viewmats_np = _as_cpu_numpy(viewmats, dtype=np.dtype(np.float32))
    intrinsics_np = _as_cpu_numpy(intrinsics, dtype=np.dtype(np.float32))
    knn_np = _as_cpu_numpy(knn, dtype=np.dtype(np.int64))
    edge_weight_np = _as_cpu_numpy(edge_weight, dtype=np.dtype(np.float32))
    observed_np = _as_cpu_numpy(observed, dtype=np.dtype(np.bool_))
    for name, value in (
        ("points", points_np),
        ("barycentric", barycentric_np),
        ("normals", normals_np),
        ("tangent_axis", axis_np),
        ("normal_tangent_ratio", ratio_np),
        ("per_view_axes", per_view_axes_np),
        ("per_view_weights", per_view_weights_np),
        ("viewmats", viewmats_np),
        ("intrinsics", intrinsics_np),
        ("edge_weight", edge_weight_np),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains values that are non-finite after float32 conversion")

    identity = _canonical_identity(points_np, face_ids_np, barycentric_np)
    order = np.asarray(identity["order"], dtype=np.int64)
    points_c = points_np[order]
    projection_points_c = projection_points_np[order]
    normals_c = normals_np[order]
    axis_c = axis_np[order]
    ratio_c = ratio_np[order]
    initial_sign_c = initial_sign_np[order]
    observed_c = observed_np[order]
    per_view_axes_c = per_view_axes_np[:, order]
    per_view_weights_c = per_view_weights_np[:, order]
    geometry_rows = np.concatenate((points_c, normals_c, axis_c), axis=1).astype(np.float32, copy=False)

    graph = _build_graph(
        knn=knn_np,
        edge_weight=edge_weight_np,
        identity=identity,
        observed=observed_np,
    )
    u = np.asarray(graph["u"], dtype=np.int64)
    v = np.asarray(graph["v"], dtype=np.int64)
    edge_weight_c = np.asarray(graph["weight"], dtype=np.float32)
    transported_axis = _transport_pairs(axis_c, normals_c, u, v)
    coupling = edge_weight_c.astype(np.float64) * np.sum(
        axis_c[u] * transported_axis,
        axis=-1,
        dtype=np.float64,
    )
    if not np.isfinite(coupling).all():
        raise ValueError("pairwise coupling contains non-finite values")
    coupling_adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for first, second, value in zip(u.tolist(), v.tolist(), coupling.tolist()):
        coupling_adjacency[int(first)].append((int(second), float(value)))
        coupling_adjacency[int(second)].append((int(first), float(value)))
    for node, row in enumerate(coupling_adjacency):
        row.sort(key=lambda item: (identity["canonical_keys"][item[0]], float(item[1])))
        coupling_adjacency[node] = row
    graph["adjacency"] = coupling_adjacency
    current_pair_field = _pairwise_field(n, u, v, coupling, initial_sign_c)

    unary = _compute_unary(
        points=projection_points_c,
        normals=normals_c,
        axis=axis_c,
        ratio=ratio_c,
        per_view_axes=per_view_axes_c,
        per_view_weights=per_view_weights_c,
        viewmats=viewmats_np,
        intrinsics=intrinsics_np,
        observed=observed_c,
        identity=identity,
    )
    h = np.asarray(unary["h"], dtype=np.float64)
    total_score = np.asarray(unary["total_score"], dtype=np.float64)
    alpha_scale_mask = (
        observed_c
        & (np.asarray(graph["degree"], dtype=np.int64) > 0)
        & (total_score > float(EPS))
        & np.isfinite(h)
        & np.isfinite(current_pair_field)
    )
    positive_unary_magnitude_mask = alpha_scale_mask & (np.abs(h) > 0.0)
    unary_q90 = _masked_quantile(np.abs(h), positive_unary_magnitude_mask)
    pairwise_q90 = _masked_quantile(np.abs(current_pair_field), alpha_scale_mask)
    alpha0_fallback = unary_q90 <= float(EPS)
    alpha0 = pairwise_q90 / max(unary_q90, float(EPS)) if not alpha0_fallback else 1.0
    if not np.isfinite(alpha0) or alpha0 <= 0.0:
        alpha0 = 1.0
        alpha0_fallback = True
    alpha = float(alpha0 * GLOBAL_SIGN_ALPHA_MULTIPLIER)

    ratio_axis_baseline = _normalize_float32(ratio_c[:, None] * normals_c + initial_sign_c[:, None] * axis_c)
    ratio_axis_one_flip = _normalize_float32(ratio_c[:, None] * normals_c - initial_sign_c[:, None] * axis_c)
    baseline_dot = _edge_dots(ratio_axis_baseline, normals_c, u, v)
    transported_baseline = _transport_pairs(ratio_axis_baseline, normals_c, u, v)
    transported_one_flip = _transport_pairs(ratio_axis_one_flip, normals_c, u, v)
    with np.errstate(invalid="ignore"):
        u_endpoint_flipped_dot = np.sum(
            ratio_axis_one_flip[u] * transported_baseline,
            axis=-1,
        ).astype(np.float32)
        v_endpoint_flipped_dot = np.sum(
            ratio_axis_baseline[u] * transported_one_flip,
            axis=-1,
        ).astype(np.float32)
        both_endpoints_flipped_dot = np.sum(
            ratio_axis_one_flip[u] * transported_one_flip,
            axis=-1,
        ).astype(np.float32)
    u_endpoint_flipped_dot = np.clip(u_endpoint_flipped_dot, -1.0, 1.0)
    v_endpoint_flipped_dot = np.clip(v_endpoint_flipped_dot, -1.0, 1.0)
    both_endpoints_flipped_dot = np.clip(both_endpoints_flipped_dot, -1.0, 1.0)
    baseline_clean = baseline_dot > SEVERE_DOT_THRESHOLD
    equality_mask = baseline_clean & (
        (u_endpoint_flipped_dot <= SEVERE_DOT_THRESHOLD)
        | (v_endpoint_flipped_dot <= SEVERE_DOT_THRESHOLD)
    )
    supernodes = _build_supernodes(
        root_count=n,
        equality_u=u[equality_mask],
        equality_v=v[equality_mask],
        identity=identity,
        geometry_rows=geometry_rows,
    )

    optimization = _block_coordinate_ascent(
        current_sign=initial_sign_c,
        u=u,
        v=v,
        coupling=coupling,
        adjacency=graph["adjacency"],
        h=h,
        alpha=alpha,
        supernodes=supernodes,
    )
    candidate_sign_c = np.asarray(optimization["signs"], dtype=np.int8)
    candidate_direction_c = _normalize_float32(
        ratio_c[:, None] * normals_c + candidate_sign_c[:, None] * axis_c
    )
    candidate_dot = _edge_dots(candidate_direction_c, normals_c, u, v)
    baseline_severe = baseline_dot <= SEVERE_DOT_THRESHOLD
    candidate_severe = candidate_dot <= SEVERE_DOT_THRESHOLD
    new_severe = (~baseline_severe) & candidate_severe
    resolved_severe = baseline_severe & (~candidate_severe)
    equality_satisfied = bool(
        np.all(
            np.asarray(optimization["flip_variables"], dtype=np.int8)[u[equality_mask]]
            == np.asarray(optimization["flip_variables"], dtype=np.int8)[v[equality_mask]]
        )
    ) if equality_mask.any() else True
    uncaught_clean_both_flip_severe = (
        baseline_clean
        & (u_endpoint_flipped_dot > SEVERE_DOT_THRESHOLD)
        & (v_endpoint_flipped_dot > SEVERE_DOT_THRESHOLD)
        & (both_endpoints_flipped_dot <= SEVERE_DOT_THRESHOLD)
    )

    identity_order = np.asarray(identity["order"], dtype=np.int64)
    canonical_rank = np.empty(n, dtype=np.int64)
    canonical_rank[identity_order] = np.arange(n, dtype=np.int64)
    candidate_sign = _root_order_out(candidate_sign_c, identity_order)
    candidate_direction = _root_order_out(candidate_direction_c, identity_order)
    flip_variables_c = np.asarray(optimization["flip_variables"], dtype=np.int8)
    flip_variables = _root_order_out(flip_variables_c, identity_order)
    flip_mask = candidate_sign != initial_sign_np
    representative_c = np.asarray(supernodes["representative"], dtype=np.int64)
    representative_out = _root_order_out(representative_c, identity_order)
    block_ordinal_out = _root_order_out(np.asarray(supernodes["block_ordinal"], dtype=np.int64), identity_order)
    canonical_edge_u = u.copy()
    canonical_edge_v = v.copy()
    edge_u = identity_order[u]
    edge_v = identity_order[v]

    candidate_pair_field_c = _pairwise_field(n, u, v, coupling, candidate_sign_c)
    candidate_pair_field = _root_order_out(candidate_pair_field_c, identity_order)
    current_pair_field_out = _root_order_out(current_pair_field, identity_order)

    block_members_out = [
        _to_device_tensor(identity_order[members], device=device, dtype=torch.long)
        for members in supernodes["blocks"]
    ]
    sign_output_dtype = initial_sign.dtype
    candidate_sign_tensor = _to_device_tensor(candidate_sign, device=device, dtype=sign_output_dtype)
    baseline_direction_tensor = _to_device_tensor(
        _root_order_out(ratio_axis_baseline, identity_order),
        device=device,
        dtype=torch.float32,
    )
    candidate_direction_tensor = _to_device_tensor(
        candidate_direction,
        device=device,
        dtype=torch.float32,
    )
    ratio_tensor = normal_tangent_ratio.detach().clone()

    unary_tensors: dict[str, object] = {
        "d_plus": _to_device_tensor(
            _root_order_out(np.asarray(unary["d_plus"], dtype=np.float32), identity_order),
            device=device,
            dtype=torch.float32,
        ),
        "d_minus": _to_device_tensor(
            _root_order_out(np.asarray(unary["d_minus"], dtype=np.float32), identity_order),
            device=device,
            dtype=torch.float32,
        ),
        "score_plus": _to_device_tensor(
            _root_matrix_out(np.asarray(unary["score_plus"], dtype=np.float64), identity_order),
            device=device,
            dtype=torch.float64,
        ),
        "score_minus": _to_device_tensor(
            _root_matrix_out(np.asarray(unary["score_minus"], dtype=np.float64), identity_order),
            device=device,
            dtype=torch.float64,
        ),
        "valid_rows": _to_device_tensor(
            _root_matrix_out(np.asarray(unary["valid_rows"], dtype=bool), identity_order),
            device=device,
            dtype=torch.bool,
        ),
        "h": _to_device_tensor(
            _root_order_out(h, identity_order), device=device, dtype=torch.float64
        ),
        "total_score": _to_device_tensor(
            _root_order_out(np.asarray(unary["total_score"], dtype=np.float64), identity_order),
            device=device,
            dtype=torch.float64,
        ),
        "alpha_scale_mask": _to_device_tensor(
            _root_order_out(alpha_scale_mask, identity_order),
            device=device,
            dtype=torch.bool,
        ),
        "signed_margin": _to_device_tensor(
            _root_order_out(np.asarray(unary["signed_margin"], dtype=np.float64), identity_order),
            device=device,
            dtype=torch.float64,
        ),
        "normalized_margin": _to_device_tensor(
            _root_order_out(np.asarray(unary["normalized_margin"], dtype=np.float64), identity_order),
            device=device,
            dtype=torch.float64,
        ),
        "vote": _to_device_tensor(
            _root_matrix_out(np.asarray(unary["vote"], dtype=np.float64), identity_order),
            device=device,
            dtype=torch.float64,
        ),
        "vote_abs_sum": _to_device_tensor(
            _root_order_out(np.asarray(unary["vote_abs_sum"], dtype=np.float64), identity_order),
            device=device,
            dtype=torch.float64,
        ),
        "vote_concentration": _to_device_tensor(
            _root_order_out(np.asarray(unary["vote_concentration"], dtype=np.float64), identity_order),
            device=device,
            dtype=torch.float64,
        ),
        "vote_coherence": _to_device_tensor(
            _root_order_out(np.asarray(unary["vote_coherence"], dtype=np.float64), identity_order),
            device=device,
            dtype=torch.float64,
        ),
        "vote_count": _to_device_tensor(
            _root_order_out(np.asarray(unary["vote_count"], dtype=np.int64), identity_order),
            device=device,
            dtype=torch.long,
        ),
        "dominant_slot": _to_device_tensor(
            _root_order_out(np.asarray(unary["dominant_slot"], dtype=np.int64), identity_order),
            device=device,
            dtype=torch.long,
        ),
        "positive_vote_count": _to_device_tensor(
            _root_order_out(np.asarray(unary["positive_vote_count"], dtype=np.int64), identity_order),
            device=device,
            dtype=torch.long,
        ),
        "preferred_sign": _to_device_tensor(
            _root_order_out(np.asarray(unary["preferred_sign"], dtype=np.int8), identity_order),
            device=device,
            dtype=torch.int8,
        ),
        "tie_resolved_sign": _to_device_tensor(
            _root_order_out(np.asarray(unary["tie_resolved_sign"], dtype=np.int8), identity_order),
            device=device,
            dtype=torch.int8,
        ),
        "canonical_view_order": _to_device_tensor(
            np.asarray(unary["view_order"], dtype=np.int64), device=device, dtype=torch.long
        ),
    }
    unary_tensors["diagnostic"] = unary["diagnostic"]

    edge_tensors: dict[str, object] = {
        "u": _to_device_tensor(edge_u, device=device, dtype=torch.long),
        "v": _to_device_tensor(edge_v, device=device, dtype=torch.long),
        "canonical_u": _to_device_tensor(canonical_edge_u, device=device, dtype=torch.long),
        "canonical_v": _to_device_tensor(canonical_edge_v, device=device, dtype=torch.long),
        "weight": _to_device_tensor(edge_weight_c, device=device, dtype=torch.float32),
        "observed": _to_device_tensor(np.asarray(graph["observed"], dtype=bool), device=device, dtype=torch.bool),
        "pairwise_coupling": _to_device_tensor(coupling, device=device, dtype=torch.float64),
        "current_pairwise_field": _to_device_tensor(current_pair_field_out, device=device, dtype=torch.float64),
        "candidate_pairwise_field": _to_device_tensor(candidate_pair_field, device=device, dtype=torch.float64),
        "transported_tangent_axis": _to_device_tensor(transported_axis, device=device, dtype=torch.float32),
        "baseline_dot": _to_device_tensor(baseline_dot, device=device, dtype=torch.float32),
        "candidate_dot": _to_device_tensor(candidate_dot, device=device, dtype=torch.float32),
        "u_endpoint_flipped_dot": _to_device_tensor(u_endpoint_flipped_dot, device=device, dtype=torch.float32),
        "v_endpoint_flipped_dot": _to_device_tensor(v_endpoint_flipped_dot, device=device, dtype=torch.float32),
        "both_endpoints_flipped_dot": _to_device_tensor(both_endpoints_flipped_dot, device=device, dtype=torch.float32),
        "baseline_clean_mask": _to_device_tensor(baseline_clean, device=device, dtype=torch.bool),
        "baseline_severe_mask": _to_device_tensor(baseline_severe, device=device, dtype=torch.bool),
        "candidate_severe_mask": _to_device_tensor(candidate_severe, device=device, dtype=torch.bool),
        "new_severe_mask": _to_device_tensor(new_severe, device=device, dtype=torch.bool),
        "resolved_severe_mask": _to_device_tensor(resolved_severe, device=device, dtype=torch.bool),
        "equality_mask": _to_device_tensor(equality_mask, device=device, dtype=torch.bool),
    }

    report: dict[str, object] = {
        "schema": "anigroom.global_sign_orientation.v1",
        "root_count": int(n),
        "view_count": int(view_count),
        "neighbor_count": int(neighbor_count),
        "observed_count": int(observed_c.sum()),
        "constants": {
            "alpha_multiplier": float(GLOBAL_SIGN_ALPHA_MULTIPLIER),
            "cos45": float(COS45),
            "severe_dot_threshold": float(SEVERE_DOT_THRESHOLD),
            "max_block_steps": int(MAX_BLOCK_STEPS),
        },
        "canonical_root_identity": identity["report"],
        "graph": graph["report"],
        "unary": {
            "score_definition": unary["diagnostic"]["score_definition"],
            "h_definition": unary["diagnostic"]["h_definition"],
            "h_abs_q90": float(unary_q90),
            "total_score_positive_root_count": int(np.sum(alpha_scale_mask)),
            "valid_pair_count": int(np.sum(unary["valid_rows"])),
            "canonical_view_order": [int(value) for value in np.asarray(unary["view_order"]).tolist()],
        },
        "alpha": {
            "alpha0": float(alpha0),
            "alpha": float(alpha),
            "pairwise_field_abs_q90": float(pairwise_q90),
            "unary_abs_h_q90": float(unary_q90),
            "fallback": bool(alpha0_fallback),
            "rule": "q90(abs current pairwise field) / q90(abs h), then multiply by public 0.5",
        },
        "alpha0": float(alpha0),
        "alpha": float(alpha),
        "equality_guard": {
            "severe_rule": "full-direction dot <= -cos45",
            "baseline_clean_rule": "full-direction dot > -cos45",
            "baseline_clean_edge_count": int(baseline_clean.sum()),
            "baseline_severe_edge_count": int(baseline_severe.sum()),
            "equality_edge_count": int(equality_mask.sum()),
            "supernode_count": int(len(supernodes["blocks"])),
            "uncaught_clean_both_endpoint_flip_severe_count": int(
                uncaught_clean_both_flip_severe.sum()
            ),
            "rule": (
                "baseline-clean edges whose exactly-one-endpoint flip is severe "
                "are unioned and require equal block flip variables"
            ),
        },
        "optimization": {
            "objective": float(optimization["objective"]),
            "initial_objective": float(optimization["initial_objective"]),
            "objective_gain": float(optimization["objective_gain"]),
            "pairwise_objective": float(optimization["pairwise_objective"]),
            "unary_objective": float(optimization["unary_objective"]),
            "iterations": int(optimization["iterations"]),
            "converged": bool(optimization["converged"]),
            "termination_reason": str(optimization["termination_reason"]),
            "max_block_steps": int(MAX_BLOCK_STEPS),
            "best_positive_delta_only": True,
            "strictly_increasing_objective": True,
            "max_delta_formula_error": float(optimization["max_delta_formula_error"]),
            "unresolved_duplicate_tie_count": int(
                optimization["unresolved_duplicate_tie_count"]
            ),
            "accepted_block_trace": optimization["accepted_step_records"],
        },
        "final": {
            "changed_root_count": int(flip_mask.sum()),
            "equality_constraints_satisfied": bool(equality_satisfied),
            "new_severe_edge_count": int(new_severe.sum()),
            "resolved_severe_edge_count": int(resolved_severe.sum()),
            "all_baseline_clean_edges_remain_nonsevere": bool(not new_severe.any()),
            "mathematical_zero_new_severe_guard_verified": bool(
                equality_satisfied and not new_severe.any()
            ),
        },
    }

    # Convert every array-bearing public field explicitly.  This keeps the
    # serializable report separate from tensors and guarantees device locality.
    supernode_id_tensor = _to_device_tensor(representative_out, device=device, dtype=torch.long)
    supernode_ids_tensor = _to_device_tensor(block_ordinal_out, device=device, dtype=torch.long)
    equality_tensor = edge_tensors["equality_mask"]
    assert isinstance(equality_tensor, torch.Tensor)
    result: dict[str, object] = {
        "candidate_sign": candidate_sign_tensor,
        "candidate_direction": candidate_direction_tensor,
        "direction": candidate_direction_tensor,
        "baseline_direction": baseline_direction_tensor,
        "normal_tangent_ratio": ratio_tensor,
        "ratio": ratio_tensor,
        "ratio_final": ratio_tensor,
        "candidate_ratio": ratio_tensor,
        "flip_mask": _to_device_tensor(flip_mask, device=device, dtype=torch.bool),
        "flip_variable": _to_device_tensor(flip_variables, device=device, dtype=torch.int8),
        "supernode_id": supernode_id_tensor,
        "supernode_ids": supernode_ids_tensor,
        "canonical_order": _to_device_tensor(
            identity_order, device=device, dtype=torch.long
        ),
        "canonical_rank": _to_device_tensor(
            canonical_rank, device=device, dtype=torch.long
        ),
        "canonical_supernode_id": _to_device_tensor(representative_c, device=device, dtype=torch.long),
        "canonical_supernode_ids": _to_device_tensor(block_ordinal_out, device=device, dtype=torch.long),
        "equality_mask": equality_tensor,
        "equality_edge_mask": equality_tensor,
        "supernodes": block_members_out,
        "blocks": block_members_out,
        "unary": unary_tensors,
        "unary_diagnostics": unary_tensors,
        "edge": edge_tensors,
        "edge_diagnostics": edge_tensors,
        "h": unary_tensors["h"],
        "alpha_scale_mask": unary_tensors["alpha_scale_mask"],
        "pairwise_coupling": edge_tensors["pairwise_coupling"],
        "current_pairwise_field": edge_tensors["current_pairwise_field"],
        "candidate_pairwise_field": edge_tensors["candidate_pairwise_field"],
        "alpha0": float(alpha0),
        "alpha": float(alpha),
        "optimization": {
            "initial_objective": float(optimization["initial_objective"]),
            "initial_pairwise_objective": float(optimization["initial_pairwise_objective"]),
            "initial_unary_objective": float(optimization["initial_unary_objective"]),
            "objective": float(optimization["objective"]),
            "pairwise_objective": float(optimization["pairwise_objective"]),
            "unary_objective": float(optimization["unary_objective"]),
            "objective_gain": float(optimization["objective_gain"]),
            "iterations": int(optimization["iterations"]),
            "converged": bool(optimization["converged"]),
            "max_block_steps": int(MAX_BLOCK_STEPS),
            "accepted_block_deltas": [float(value) for value in optimization["accepted_block_deltas"]],
            "accepted_step_records": optimization["accepted_step_records"],
            "full_delta_stats": optimization["full_delta_stats"],
            "termination_reason": str(optimization["termination_reason"]),
            "safe_max_reached": bool(optimization["safe_max_reached"]),
            "max_delta_formula_error": float(optimization["max_delta_formula_error"]),
            "unresolved_duplicate_tie_count": int(optimization["unresolved_duplicate_tie_count"]),
            "final_local_pair_field": _to_device_tensor(
                _root_order_out(np.asarray(optimization["final_local_pair_field"], dtype=np.float64), identity_order),
                device=device,
                dtype=torch.float64,
            ),
            "final_local_field": _to_device_tensor(
                _root_order_out(np.asarray(optimization["final_local_field"], dtype=np.float64), identity_order),
                device=device,
                dtype=torch.float64,
            ),
        },
        "report": report,
    }
    return result

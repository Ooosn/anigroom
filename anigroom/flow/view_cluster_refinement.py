"""Trusted multiview axial flow refinement.

This module contains the formal, semantic-free refinement pass used to turn
per-view axial evidence into a trusted tangent axis field.  All operations are
implemented with PyTorch tensors so the same code can run on CPU or CUDA.
"""

from __future__ import annotations

import math

import torch

from .direction_geometry import parallel_transport_vectors


EPS = 1.0e-8

# Public specification constants.  Keep these names stable: reports and tests
# can record the exact discrete refinement contract without importing helpers.
AXIAL_AGREEMENT_POWER = 8
VIEW_CLUSTER_POWER = AXIAL_AGREEMENT_POWER
VIEW_CLUSTER_ITERATIONS = 4
HARD_MARGIN_QUANTILE = 0.95
MARGIN_QUANTILE = HARD_MARGIN_QUANTILE
RESIDUAL_QUANTILE = 0.95
RESIDUAL_ANGLE_QUANTILE = RESIDUAL_QUANTILE
DIRECT_SUPPORT_ANGLE_DEG = 30.0
DIRECT_SUPPORT_ANGLE = DIRECT_SUPPORT_ANGLE_DEG
DIRECT_SUPERMAJORITY = 2.0 / 3.0
CONFIDENCE_DECAY = 0.85


__all__ = [
    "AXIAL_AGREEMENT_POWER",
    "VIEW_CLUSTER_POWER",
    "VIEW_CLUSTER_ITERATIONS",
    "HARD_MARGIN_QUANTILE",
    "MARGIN_QUANTILE",
    "RESIDUAL_QUANTILE",
    "RESIDUAL_ANGLE_QUANTILE",
    "DIRECT_SUPPORT_ANGLE_DEG",
    "DIRECT_SUPPORT_ANGLE",
    "DIRECT_SUPERMAJORITY",
    "CONFIDENCE_DECAY",
    "refine_trusted_multiview_axis_field",
    "refine_fixed_axis_multiview_ratio",
]


def _finite(value: torch.Tensor) -> torch.Tensor:
    """Replace non-finite data before it can enter an eigendecomposition."""

    return torch.where(torch.isfinite(value), value, torch.zeros_like(value))


def _safe_normalize(value: torch.Tensor) -> torch.Tensor:
    """Normalize vectors without producing NaNs for zero or large inputs."""

    value = _finite(value)
    scale = value.abs().amax(dim=-1, keepdim=True)
    scaled = value / scale.clamp_min(1.0)
    length = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    valid = (scale > EPS) & (length > EPS) & torch.isfinite(length)
    normalized = scaled / length.clamp_min(EPS)
    return torch.where(valid, normalized, torch.zeros_like(value))


def _vector_valid(value: torch.Tensor) -> torch.Tensor:
    value = _finite(value)
    scale = value.abs().amax(dim=-1)
    scaled = value / scale[..., None].clamp_min(1.0)
    length = torch.linalg.vector_norm(scaled, dim=-1)
    return (scale > EPS) & (length > EPS) & torch.isfinite(length)


def _safe_length(value: torch.Tensor) -> torch.Tensor:
    value = _finite(value)
    scale = value.abs().amax(dim=-1)
    safe_scale = scale.clamp_min(EPS)
    scaled = value / safe_scale[..., None]
    length = torch.linalg.vector_norm(scaled, dim=-1) * safe_scale
    length = torch.where(scale > EPS, length, torch.zeros_like(length))
    return torch.nan_to_num(length, nan=0.0, posinf=torch.finfo(value.dtype).max, neginf=0.0)


def _tangent_basis(normals: torch.Tensor) -> torch.Tensor:
    """Return a deterministic tangent basis, including for a zero normal."""

    x_axis = torch.zeros_like(normals)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(normals)
    y_axis[..., 1] = 1.0
    helper = torch.where(normals[..., :1].abs() < 0.9, x_axis, y_axis)
    candidate = torch.linalg.cross(normals, helper, dim=-1)
    candidate = _safe_normalize(candidate)
    normal_valid = _vector_valid(normals)
    return torch.where(normal_valid[..., None], candidate, helper)


def _project_tangent(value: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Project and normalize, retaining zero for an unusable input."""

    value = _finite(value)
    normals = _finite(normals)
    tangent = value - (value * normals).sum(dim=-1, keepdim=True) * normals
    return _safe_normalize(tangent)


def _project_tangent_with_fallback(
    value: torch.Tensor,
    normals: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    value = _finite(value)
    normals = _finite(normals)
    tangent = value - (value * normals).sum(dim=-1, keepdim=True) * normals
    valid = _vector_valid(tangent)
    projected = _safe_normalize(tangent)
    fallback = _safe_normalize(fallback)
    fallback = torch.where(_vector_valid(fallback)[..., None], fallback, _tangent_basis(normals))
    return torch.where(valid[..., None], projected, fallback)


def _align_axial(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dot = (value * reference).sum(dim=-1, keepdim=True)
    sign = torch.where(dot >= 0.0, torch.ones_like(dot), -torch.ones_like(dot))
    return value * sign


def _transport_neighbors(
    axis: torch.Tensor,
    normals: torch.Tensor,
    knn: torch.Tensor,
) -> torch.Tensor:
    n = int(axis.shape[0])
    k = int(knn.shape[1])
    if k == 0:
        return axis.new_empty((n, 0, 3))
    source_normals = normals[knn]
    target_normals = normals[:, None, :].expand(n, k, 3)
    transported = parallel_transport_vectors(axis[knn], source_normals, target_normals)
    return _finite(transported)


def _neighbor_mean_with_offset(
    axis: torch.Tensor,
    confidence: torch.Tensor,
    normals: torch.Tensor,
    knn: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    offset: float,
) -> torch.Tensor:
    """Compute an axial transported neighbor mean with an optional weight floor."""

    n = int(axis.shape[0])
    k = int(knn.shape[1])
    if k == 0:
        return axis.new_zeros((n, 3))
    transported = _transport_neighbors(axis, normals, knn)
    sign = torch.where(
        (transported * axis[:, None, :]).sum(dim=-1, keepdim=True) >= 0.0,
        torch.ones_like(transported[..., :1]),
        -torch.ones_like(transported[..., :1]),
    )
    aligned = transported * sign
    source_weight = edge_weight * (float(offset) + confidence[knn])
    source_weight = _finite(source_weight).clamp_min(0.0)
    denominator = source_weight.sum(dim=1, keepdim=True)
    weighted = (aligned * source_weight[..., None]).sum(dim=1) / denominator.clamp_min(EPS)
    return _project_tangent(weighted, normals)


def _confidence_neighbor_mean(
    axis: torch.Tensor,
    confidence: torch.Tensor,
    normals: torch.Tensor,
    knn: torch.Tensor,
    edge_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one confidence-weighted transported mean and its confidence."""

    n = int(axis.shape[0])
    k = int(knn.shape[1])
    if k == 0:
        return axis.clone(), axis.new_zeros((n,))

    transported = _transport_neighbors(axis, normals, knn)
    sign = torch.where(
        (transported * axis[:, None, :]).sum(dim=-1, keepdim=True) >= 0.0,
        torch.ones_like(transported[..., :1]),
        -torch.ones_like(transported[..., :1]),
    )
    aligned = transported * sign
    source_weight = _finite(edge_weight * confidence[knn]).clamp_min(0.0)
    weight_sum = source_weight.sum(dim=1)
    weighted_sum = (aligned * source_weight[..., None]).sum(dim=1)
    resultant = _safe_length(weighted_sum)
    usable = (weight_sum > EPS) & (resultant > EPS)
    candidate = _project_tangent(weighted_sum / weight_sum[:, None].clamp_min(EPS), normals)
    mean = torch.where(usable[:, None], candidate, axis)

    edge_sum = edge_weight.sum(dim=1)
    edge_mean_conf = (edge_weight * confidence[knn]).sum(dim=1) / edge_sum.clamp_min(EPS)
    concentration = resultant / weight_sum.clamp_min(EPS)
    concentration = torch.where(weight_sum > EPS, concentration, torch.zeros_like(concentration))
    concentration = concentration.clamp(0.0, 1.0)
    neighbor_confidence = (edge_mean_conf * concentration).clamp(0.0, 1.0)
    return _finite(mean), _finite(neighbor_confidence)


def _positive_quantile(value: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    selected = value[mask & torch.isfinite(value) & (value > 0.0)]
    if selected.numel() == 0:
        return value.new_zeros(())
    return torch.quantile(selected, float(quantile))


def _masked_quantile(value: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    selected = value[mask & torch.isfinite(value)]
    if selected.numel() == 0:
        return value.new_zeros(())
    return torch.quantile(selected, float(quantile))


def _python_scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _axial_angle_degrees(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = _safe_normalize(first)
    second = _safe_normalize(second)
    cosine = (first * second).sum(dim=-1).abs().clamp(0.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def _direct_support(
    candidate: torch.Tensor,
    view_axis: torch.Tensor,
    direct_weight: torch.Tensor,
) -> torch.Tensor:
    """Fraction of direct evidence within the specified axial angle."""

    if int(view_axis.shape[0]) == 0:
        return candidate.new_zeros((candidate.shape[0],))
    cosine = (view_axis * candidate[None, :, :]).sum(dim=-1).abs().clamp(0.0, 1.0)
    threshold = math.cos(math.radians(DIRECT_SUPPORT_ANGLE_DEG))
    supporting = cosine >= threshold
    numerator = (direct_weight * supporting.to(dtype=direct_weight.dtype)).sum(dim=0)
    denominator = direct_weight.sum(dim=0)
    support = numerator / denominator.clamp_min(EPS)
    return torch.where(denominator > EPS, support, torch.zeros_like(support)).clamp(0.0, 1.0)


def _empty_result(
    *,
    n: int,
    v: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | float | int]:
    empty_axis = torch.empty((n, 3), device=device, dtype=dtype)
    empty_scalar = torch.empty((n,), device=device, dtype=dtype)
    empty_mask = torch.empty((n,), device=device, dtype=torch.bool)
    zeros_axis = torch.zeros_like(empty_axis)
    zeros_scalar = torch.zeros_like(empty_scalar)
    zeros_mask = torch.zeros_like(empty_mask)
    constants = {
        "pairwise_power": AXIAL_AGREEMENT_POWER,
        "cluster_iterations": VIEW_CLUSTER_ITERATIONS,
        "margin_quantile": MARGIN_QUANTILE,
        "residual_quantile": RESIDUAL_QUANTILE,
        "direct_support_angle_deg": DIRECT_SUPPORT_ANGLE_DEG,
        "direct_supermajority": DIRECT_SUPERMAJORITY,
        "confidence_decay": CONFIDENCE_DECAY,
    }
    cutoffs = {
        "weight_q90": 0.0,
        "n_eff_q75": 0.0,
        "hard_margin_q95": 0.0,
        "residual_q95_degrees": 0.0,
    }
    counts = {
        "root_count": n,
        "view_count": v,
        "neighbor_count": k,
        "observed_roots": 0,
        "evidence_roots": 0,
        "hard_switch_roots": 0,
        "q95_roots": 0,
        "residual_roots": 0,
        "direct_supermajority_roots": 0,
        "supermajority_roots": 0,
    }
    report = {"constants": constants, "cutoffs": cutoffs, "counts": counts}
    return {
        "axis": zeros_axis,
        "final_axis": zeros_axis,
        "anchor": zeros_axis,
        "trust": zeros_scalar,
        "base_trust": zeros_scalar,
        "spectral_gap": zeros_scalar,
        "n_eff": zeros_scalar,
        "weight_sum": zeros_scalar,
        "hard_axis": zeros_axis,
        "hard_iteration_axis": zeros_axis,
        "hard_margin": zeros_scalar,
        "hard_q95_mask": zeros_mask,
        "q95_mask": zeros_mask,
        "hard_q95_cut": 0.0,
        "residual_axis": zeros_axis,
        "residual_deg": zeros_scalar,
        "residual_degrees": zeros_scalar,
        "residual_mask": zeros_mask,
        "residual_cut": 0.0,
        "residual_direct_support": zeros_scalar,
        "direct_support": zeros_scalar,
        "residual_supermajority_mask": zeros_mask,
        "supermajority_mask": zeros_mask,
        "confidence": zeros_scalar,
        "final_confidence": zeros_scalar,
        "hard_confidence": zeros_scalar,
        "neighbor_confidence": zeros_scalar,
        "local_agreement": zeros_scalar,
        "evidence_conf": zeros_scalar,
        "view_conf": zeros_scalar,
        "weight_q90": 0.0,
        "n_eff_q75": 0.0,
        "root_count": n,
        "view_count": v,
        "neighbor_count": k,
        "observed_count": 0,
        "hard_switch_count": 0,
        "hard_q95_count": 0,
        "residual_count": 0,
        "residual_supermajority_count": 0,
        "pairwise_power": AXIAL_AGREEMENT_POWER,
        "cluster_iterations": VIEW_CLUSTER_ITERATIONS,
        "margin_quantile": MARGIN_QUANTILE,
        "residual_quantile": RESIDUAL_QUANTILE,
        "direct_support_angle_deg": DIRECT_SUPPORT_ANGLE_DEG,
        "direct_supermajority": DIRECT_SUPERMAJORITY,
        "confidence_decay": CONFIDENCE_DECAY,
        "report": report,
        "constants": constants,
        "cutoffs": cutoffs,
        "counts": counts,
    }


def refine_trusted_multiview_axis_field(
    *,
    initial_axis: torch.Tensor,
    normals: torch.Tensor,
    observed: torch.Tensor,
    per_view_vectors: torch.Tensor,
    per_view_weights: torch.Tensor,
    per_view_direct_weights: torch.Tensor,
    knn: torch.Tensor,
    edge_weight: torch.Tensor,
) -> dict[str, torch.Tensor | float | int]:
    """Refine a legacy fused tangent axis with trusted multiview evidence.

    The function intentionally accepts no root IDs, semantic labels, species
    information, or view IDs.  All decisions are made from the supplied
    vectors, weights, normals, and surface neighborhood.
    """

    tensors = {
        "initial_axis": initial_axis,
        "normals": normals,
        "observed": observed,
        "per_view_vectors": per_view_vectors,
        "per_view_weights": per_view_weights,
        "per_view_direct_weights": per_view_direct_weights,
        "knn": knn,
        "edge_weight": edge_weight,
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all refinement inputs must be torch.Tensor values")

    if initial_axis.ndim != 2 or initial_axis.shape[-1] != 3:
        raise ValueError("initial_axis must have shape [N, 3]")
    if normals.shape != initial_axis.shape:
        raise ValueError("normals must have shape [N, 3]")
    if observed.ndim != 1 or observed.shape[0] != initial_axis.shape[0]:
        raise ValueError("observed must have shape [N]")
    if per_view_vectors.ndim != 3 or per_view_vectors.shape[-1] != 3:
        raise ValueError("per_view_vectors must have shape [V, N, 3]")
    n = int(initial_axis.shape[0])
    v = int(per_view_vectors.shape[0])
    if int(per_view_vectors.shape[1]) != n:
        raise ValueError("per_view_vectors must have the same N as initial_axis")
    if per_view_weights.shape != (v, n):
        raise ValueError("per_view_weights must have shape [V, N]")
    if per_view_direct_weights.shape != (v, n):
        raise ValueError("per_view_direct_weights must have shape [V, N]")
    if knn.ndim != 2 or int(knn.shape[0]) != n:
        raise ValueError("knn must have shape [N, K]")
    if edge_weight.shape != knn.shape:
        raise ValueError("edge_weight must have shape [N, K]")

    device = initial_axis.device
    if any(value.device != device for value in tensors.values()):
        raise ValueError("all refinement inputs must be on the same device")
    if initial_axis.is_complex() or normals.is_complex() or per_view_vectors.is_complex():
        raise TypeError("axis and normal inputs must be real tensors")
    if knn.is_floating_point() or knn.is_complex():
        raise TypeError("knn must be an integer tensor")

    dtype = torch.float64 if any(value.dtype == torch.float64 for value in tensors.values() if value.is_floating_point()) else torch.float32
    k = int(knn.shape[1])
    if knn.numel() > 0 and bool(((knn < 0) | (knn >= n)).any()):
        raise ValueError("knn contains an out-of-range root index")
    if n == 0:
        return _empty_result(n=n, v=v, k=k, device=device, dtype=dtype)

    normal = _safe_normalize(normals.to(dtype=dtype))
    initial = _finite(initial_axis.to(dtype=dtype))
    initial_tangent = _project_tangent_with_fallback(initial, normal, _tangent_basis(normal))

    raw_vectors = _finite(per_view_vectors.to(dtype=dtype))
    raw_weights = _finite(per_view_weights.to(dtype=dtype)).clamp_min(0.0)
    raw_direct_weights = _finite(per_view_direct_weights.to(dtype=dtype)).clamp_min(0.0)
    vector_valid = _vector_valid(raw_vectors)
    view_axis = _safe_normalize(raw_vectors)
    view_weight = torch.where(vector_valid & (raw_weights > 0.0), raw_weights, torch.zeros_like(raw_weights))
    valid_view = view_weight > 0.0
    direct_weight = torch.where(vector_valid, raw_direct_weights, torch.zeros_like(raw_direct_weights))

    weight_sum = _finite(view_weight.sum(dim=0))
    if v > 0:
        stable_scale = view_weight.amax(dim=0, keepdim=True)
        stable_weight = view_weight / stable_scale.clamp_min(EPS)
    else:
        stable_weight = view_weight
    stable_sum = stable_weight.sum(dim=0)
    scatter = torch.einsum("vn,vni,vnj->nij", stable_weight, view_axis, view_axis)
    scatter = scatter / stable_sum[:, None, None].clamp_min(EPS)
    scatter = _finite(scatter)
    eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
    eigen_anchor = eigenvectors[:, :, -1]
    anchor = _project_tangent_with_fallback(eigen_anchor, normal, initial_tangent)
    anchor = torch.where((weight_sum > EPS)[:, None], anchor, initial_tangent)
    anchor = _align_axial(anchor, initial_tangent)
    anchor = _finite(anchor)

    lambda_one = _finite(eigenvalues[:, -1])
    lambda_two = _finite(eigenvalues[:, -2])
    spectral_gap = ((lambda_one - lambda_two).clamp_min(0.0) / (lambda_one + lambda_two).clamp_min(EPS)).clamp(0.0, 1.0)
    spectral_gap = torch.where(weight_sum > EPS, spectral_gap, torch.zeros_like(spectral_gap))

    stable_square_sum = (stable_weight * stable_weight).sum(dim=0)
    n_eff = (stable_sum * stable_sum) / stable_square_sum.clamp_min(EPS)
    n_eff = torch.where(weight_sum > EPS, n_eff, torch.zeros_like(n_eff)).clamp(0.0, float(max(v, 0)))
    n_eff = _finite(n_eff)

    observed_mask = observed.to(device=device, dtype=torch.bool)
    weight_q90_tensor = _positive_quantile(weight_sum, observed_mask, 0.90)
    n_eff_q75_tensor = _positive_quantile(n_eff, observed_mask, 0.75)
    weight_confidence = torch.where(
        weight_q90_tensor > EPS,
        weight_sum / weight_q90_tensor.clamp_min(EPS),
        torch.zeros_like(weight_sum),
    ).clamp(0.0, 1.0)
    view_confidence = torch.where(
        n_eff_q75_tensor > EPS,
        n_eff / n_eff_q75_tensor.clamp_min(EPS),
        torch.zeros_like(n_eff),
    ).clamp(0.0, 1.0)
    base_trust = (spectral_gap.square() * weight_confidence * view_confidence).clamp(0.0, 1.0)

    initial_neighbor_mean = _neighbor_mean_with_offset(
        anchor,
        base_trust,
        normal,
        knn,
        _finite(edge_weight.to(dtype=dtype)).clamp_min(0.0),
        offset=0.01,
    )
    local_agreement = (anchor * initial_neighbor_mean).sum(dim=-1).abs().clamp(0.0, 1.0)
    trust = (base_trust * local_agreement.square()).clamp(0.0, 1.0)
    trust = _finite(trust)

    edge = _finite(edge_weight.to(dtype=dtype)).clamp_min(0.0)
    edge_sum = edge.sum(dim=1)
    total_weight_denom = weight_sum.clamp_min(EPS)

    if v > 0:
        pairwise = (view_axis[:, None, :, :] * view_axis[None, :, :, :]).sum(dim=-1).abs().clamp(0.0, 1.0)
        pairwise = pairwise.pow(AXIAL_AGREEMENT_POWER)
        unary = (pairwise * view_weight[None, :, :]).sum(dim=1) / total_weight_denom[None, :]
        unary = torch.where(valid_view, unary, torch.zeros_like(unary)).clamp(0.0, 1.0)
        anchor_view_similarity = (view_axis * anchor[None, :, :]).sum(dim=-1).abs().clamp(0.0, 1.0).pow(AXIAL_AGREEMENT_POWER)
        anchor_unary = (view_weight * anchor_view_similarity).sum(dim=0) / total_weight_denom
    else:
        unary = view_weight.new_empty((0, n))
        anchor_unary = torch.zeros_like(weight_sum)

    current_axis = anchor.clone()
    current_confidence = trust.clone()
    hard_best_score = torch.zeros_like(trust)
    hard_anchor_score = torch.zeros_like(trust)
    hard_best_unary = torch.zeros_like(trust)
    hard_best_spatial = torch.zeros_like(trust)
    hard_switch = torch.zeros_like(observed_mask)

    for _ in range(VIEW_CLUSTER_ITERATIONS):
        if v == 0:
            break
        transported = _transport_neighbors(current_axis, normal, knn)
        source_weight = (edge * current_confidence[knn]).clamp_min(0.0)
        source_weight_sum = source_weight.sum(dim=1)
        source_denom = source_weight_sum.clamp_min(EPS)
        edge_mean_confidence = (edge * current_confidence[knn]).sum(dim=1) / edge_sum.clamp_min(EPS)
        edge_mean_confidence = torch.where(edge_sum > EPS, edge_mean_confidence, torch.zeros_like(edge_mean_confidence)).clamp(0.0, 1.0)

        spatial_similarity = (view_axis[:, :, None, :] * transported[None, :, :, :]).sum(dim=-1).abs().clamp(0.0, 1.0)
        spatial_similarity = spatial_similarity.pow(AXIAL_AGREEMENT_POWER)
        spatial = (spatial_similarity * source_weight[None, :, :]).sum(dim=-1) / source_denom[None, :]
        spatial = torch.where(valid_view, spatial, torch.zeros_like(spatial)).clamp(0.0, 1.0)

        combined = trust[None, :] * unary + edge_mean_confidence[None, :] * spatial
        combined = torch.where(valid_view, combined, torch.full_like(combined, -1.0))
        best_view = torch.argmax(combined, dim=0)
        root_ids = torch.arange(n, device=device)
        best_score = combined[best_view, root_ids].clamp_min(0.0)
        best_unary = unary[best_view, root_ids].clamp(0.0, 1.0)
        best_spatial = spatial[best_view, root_ids].clamp(0.0, 1.0)
        seed_axis = view_axis[best_view, root_ids]
        seed_valid = vector_valid[best_view, root_ids] & valid_view[best_view, root_ids]

        cluster_dot = (view_axis * seed_axis[None, :, :]).sum(dim=-1)
        cluster_weight = view_weight * cluster_dot.abs().clamp(0.0, 1.0).pow(AXIAL_AGREEMENT_POWER)
        cluster_sign = torch.where(cluster_dot >= 0.0, torch.ones_like(cluster_dot), -torch.ones_like(cluster_dot))
        cluster_sum = (cluster_weight[..., None] * cluster_sign[..., None] * view_axis).sum(dim=0)
        cluster_fallback = torch.where(seed_valid[:, None], seed_axis, anchor)
        cluster_axis = _project_tangent_with_fallback(cluster_sum, normal, cluster_fallback)
        cluster_axis = _align_axial(cluster_axis, anchor)

        anchor_spatial_similarity = (anchor[:, None, :] * transported).sum(dim=-1).abs().clamp(0.0, 1.0).pow(AXIAL_AGREEMENT_POWER)
        anchor_spatial = (source_weight * anchor_spatial_similarity).sum(dim=1) / source_denom
        anchor_spatial = torch.where(source_weight_sum > EPS, anchor_spatial, torch.zeros_like(anchor_spatial)).clamp(0.0, 1.0)
        anchor_score = trust * anchor_unary + edge_mean_confidence * anchor_spatial

        advantage = (best_score - anchor_score).clamp_min(0.0)
        has_direct = valid_view.any(dim=0)
        has_neighbor = source_weight_sum > EPS
        usable = observed_mask & has_direct & has_neighbor
        switch = usable & (best_score > anchor_score + EPS)
        selected_confidence = torch.maximum(trust * best_unary, CONFIDENCE_DECAY * edge_mean_confidence * best_spatial).clamp(0.0, 1.0)
        next_confidence = torch.where(usable, torch.maximum(trust, selected_confidence), trust)
        next_axis = torch.where(switch[:, None], cluster_axis, anchor)

        current_axis = _finite(next_axis)
        current_confidence = _finite(next_confidence).clamp(0.0, 1.0)
        hard_best_score = _finite(best_score)
        hard_anchor_score = _finite(anchor_score)
        hard_best_unary = _finite(best_unary)
        hard_best_spatial = _finite(best_spatial)
        hard_switch = switch

    hard_margin = (hard_best_score - hard_anchor_score).clamp_min(0.0) / hard_best_score.clamp_min(EPS)
    hard_margin = torch.where(hard_best_score > EPS, hard_margin, torch.zeros_like(hard_margin)).clamp(0.0, 1.0)
    hard_margin = _finite(hard_margin)
    hard_q95_cut_tensor = _masked_quantile(hard_margin, observed_mask, MARGIN_QUANTILE)
    hard_q95_mask = observed_mask & hard_switch & (hard_margin >= hard_q95_cut_tensor) & (hard_margin > 0.0)
    hard_axis = torch.where(hard_q95_mask[:, None], current_axis, anchor)
    hard_confidence = torch.where(hard_q95_mask, current_confidence, trust).clamp(0.0, 1.0)

    residual_axis, neighbor_confidence = _confidence_neighbor_mean(
        hard_axis,
        hard_confidence,
        normal,
        knn,
        edge,
    )
    residual_deg = _axial_angle_degrees(hard_axis, residual_axis)
    residual_valid = observed_mask & (neighbor_confidence > EPS) & torch.isfinite(residual_deg)
    residual_cut_tensor = _masked_quantile(residual_deg, residual_valid, RESIDUAL_QUANTILE)
    residual_mask = residual_valid & (residual_deg > residual_cut_tensor) & (residual_deg > 0.0)
    residual_direct_support = _direct_support(residual_axis, view_axis, direct_weight)
    residual_supermajority_mask = residual_mask & (residual_direct_support >= DIRECT_SUPERMAJORITY)

    final_axis = torch.where(residual_supermajority_mask[:, None], residual_axis, hard_axis)
    confidence = torch.where(
        residual_supermajority_mask,
        torch.maximum(hard_confidence, CONFIDENCE_DECAY * neighbor_confidence),
        hard_confidence,
    ).clamp(0.0, 1.0)
    confidence = torch.maximum(confidence, trust).clamp(0.0, 1.0)

    constants = {
        "pairwise_power": AXIAL_AGREEMENT_POWER,
        "cluster_iterations": VIEW_CLUSTER_ITERATIONS,
        "margin_quantile": MARGIN_QUANTILE,
        "residual_quantile": RESIDUAL_QUANTILE,
        "direct_support_angle_deg": DIRECT_SUPPORT_ANGLE_DEG,
        "direct_supermajority": DIRECT_SUPERMAJORITY,
        "confidence_decay": CONFIDENCE_DECAY,
    }
    cutoffs = {
        "weight_q90": _python_scalar(weight_q90_tensor),
        "n_eff_q75": _python_scalar(n_eff_q75_tensor),
        "hard_margin_q95": _python_scalar(hard_q95_cut_tensor),
        "residual_q95_degrees": _python_scalar(residual_cut_tensor),
    }
    counts = {
        "root_count": n,
        "view_count": v,
        "neighbor_count": k,
        "observed_roots": int(observed_mask.sum().detach().cpu().item()),
        "evidence_roots": int((weight_sum > EPS).sum().detach().cpu().item()),
        "hard_switch_roots": int(hard_switch.sum().detach().cpu().item()),
        "q95_roots": int(hard_q95_mask.sum().detach().cpu().item()),
        "residual_roots": int(residual_mask.sum().detach().cpu().item()),
        "direct_supermajority_roots": int(
            (residual_direct_support >= DIRECT_SUPERMAJORITY).sum().detach().cpu().item()
        ),
        "supermajority_roots": int(residual_supermajority_mask.sum().detach().cpu().item()),
    }
    report = {"constants": constants, "cutoffs": cutoffs, "counts": counts}

    result: dict[str, torch.Tensor | float | int | dict[str, object]] = {
        "axis": _finite(final_axis),
        "final_axis": _finite(final_axis),
        "anchor": _finite(anchor),
        "trust": _finite(trust),
        "base_trust": _finite(base_trust),
        "spectral_gap": _finite(spectral_gap),
        "n_eff": _finite(n_eff),
        "weight_sum": _finite(weight_sum),
        "hard_axis": _finite(hard_axis),
        "hard_iteration_axis": _finite(current_axis),
        "hard_margin": _finite(hard_margin),
        "hard_q95_mask": hard_q95_mask,
        "q95_mask": hard_q95_mask,
        "hard_q95_cut": _python_scalar(hard_q95_cut_tensor),
        "residual_axis": _finite(residual_axis),
        "residual_deg": _finite(residual_deg),
        "residual_degrees": _finite(residual_deg),
        "residual_mask": residual_mask,
        "residual_cut": _python_scalar(residual_cut_tensor),
        "residual_direct_support": _finite(residual_direct_support),
        "direct_support": _finite(residual_direct_support),
        "residual_supermajority_mask": residual_supermajority_mask,
        "supermajority_mask": residual_supermajority_mask,
        "confidence": _finite(confidence),
        "final_confidence": _finite(confidence),
        "hard_confidence": _finite(hard_confidence),
        "neighbor_confidence": _finite(neighbor_confidence),
        "local_agreement": _finite(local_agreement),
        "evidence_conf": _finite(weight_confidence),
        "view_conf": _finite(view_confidence),
        "weight_q90": _python_scalar(weight_q90_tensor),
        "n_eff_q75": _python_scalar(n_eff_q75_tensor),
        "root_count": n,
        "view_count": v,
        "neighbor_count": k,
        "observed_count": int(observed_mask.sum().detach().cpu().item()),
        "hard_switch_count": int(hard_switch.sum().detach().cpu().item()),
        "hard_q95_count": int(hard_q95_mask.sum().detach().cpu().item()),
        "residual_count": int(residual_mask.sum().detach().cpu().item()),
        "residual_supermajority_count": int(residual_supermajority_mask.sum().detach().cpu().item()),
        "pairwise_power": AXIAL_AGREEMENT_POWER,
        "cluster_iterations": VIEW_CLUSTER_ITERATIONS,
        "margin_quantile": MARGIN_QUANTILE,
        "residual_quantile": RESIDUAL_QUANTILE,
        "direct_support_angle_deg": DIRECT_SUPPORT_ANGLE_DEG,
        "direct_supermajority": DIRECT_SUPERMAJORITY,
        "confidence_decay": CONFIDENCE_DECAY,
        "report": report,
        "constants": constants,
        "cutoffs": cutoffs,
        "counts": counts,
    }
    return result


def _project_fixed_axis_screen(
    points: torch.Tensor,
    directions: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Project a per-view 3-D direction field with the standalone formula."""

    points = _finite(points)
    directions = _finite(directions)
    viewmats = _finite(viewmats)
    intrinsics = _finite(intrinsics)
    rotation = viewmats[:, :3, :3]
    translation = viewmats[:, :3, 3]
    camera_points = torch.einsum("ni,vji->vnj", points, rotation) + translation[:, None, :]
    camera_directions = torch.einsum("vni,vji->vnj", directions, rotation)
    camera_points = _finite(camera_points)
    camera_directions = _finite(camera_directions)
    depth = camera_points[..., 2].clamp_min(EPS)
    denominator = depth.square().clamp_min(EPS)
    focal_x = intrinsics[:, 0, 0, None]
    focal_y = intrinsics[:, 1, 1, None]
    screen_x = focal_x * (
        camera_directions[..., 0] * depth - camera_points[..., 0] * camera_directions[..., 2]
    ) / denominator
    screen_y = focal_y * (
        camera_directions[..., 1] * depth - camera_points[..., 1] * camera_directions[..., 2]
    ) / denominator
    return _finite(torch.stack((screen_x, screen_y), dim=-1))


def _fixed_axis_screen_valid(value: torch.Tensor) -> torch.Tensor:
    """Match the standalone 1e-6 projected-vector validity check safely."""

    value = _finite(value)
    return _finite(_safe_length(value)) > 1.0e-6


def _fixed_axis_local_max_jump(
    direction: torch.Tensor,
    normals: torch.Tensor,
    knn: torch.Tensor,
    edge_weight: torch.Tensor,
    observed: torch.Tensor,
) -> torch.Tensor:
    """Return the maximum valid transported axial edge jump per root."""

    n = int(direction.shape[0])
    k = int(knn.shape[1])
    if k == 0:
        return direction.new_zeros((n,))
    transported = _transport_neighbors(direction, normals, knn)
    angle = _axial_angle_degrees(direction[:, None, :], transported)
    pair_valid = (edge_weight > 0.0) & observed[:, None] & observed[knn]
    return _finite(torch.where(pair_valid, angle, torch.zeros_like(angle)).amax(dim=1))


def _fixed_axis_qstats(value: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    """Compute finite scalar statistics without moving solver tensors to CPU."""

    selected = value[mask & torch.isfinite(value)]
    if selected.numel() == 0:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    quantiles = torch.quantile(selected, selected.new_tensor((0.50, 0.90, 0.95, 0.99)))
    selected = selected.detach()
    quantiles = quantiles.detach()
    return {
        "count": int(selected.numel()),
        "mean": float(selected.mean().cpu().item()),
        "p50": float(quantiles[0].cpu().item()),
        "p90": float(quantiles[1].cpu().item()),
        "p95": float(quantiles[2].cpu().item()),
        "p99": float(quantiles[3].cpu().item()),
        "max": float(selected.max().cpu().item()),
    }


def refine_fixed_axis_multiview_ratio(
    *,
    initial_direction: torch.Tensor,
    tangent_axis: torch.Tensor,
    normals: torch.Tensor,
    points: torch.Tensor,
    per_view_axes: torch.Tensor,
    per_view_weights: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    knn: torch.Tensor,
    edge_weight: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, object]:
    """Solve a nonnegative normal/tangent ratio around a fixed tangent axis.

    The implementation is the reusable Torch port of the accepted standalone
    solver.  It has no semantic, root-ID, or view-ID inputs: view evidence is
    combined only through the supplied projected directions and weights.
    """

    tensors = {
        "initial_direction": initial_direction,
        "tangent_axis": tangent_axis,
        "normals": normals,
        "points": points,
        "per_view_axes": per_view_axes,
        "per_view_weights": per_view_weights,
        "viewmats": viewmats,
        "intrinsics": intrinsics,
        "knn": knn,
        "edge_weight": edge_weight,
        "observed": observed,
    }
    if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("all fixed-axis ratio inputs must be torch.Tensor values")

    if initial_direction.ndim != 2 or initial_direction.shape[-1] != 3:
        raise ValueError("initial_direction must have shape [N, 3]")
    n = int(initial_direction.shape[0])
    for name, value in (("tangent_axis", tangent_axis), ("normals", normals), ("points", points)):
        if value.shape != (n, 3):
            raise ValueError(f"{name} must have shape [N, 3]")
    if per_view_axes.ndim != 3 or per_view_axes.shape[-1] != 3:
        raise ValueError("per_view_axes must have shape [V, N, 3]")
    v = int(per_view_axes.shape[0])
    if int(per_view_axes.shape[1]) != n:
        raise ValueError("per_view_axes must have the same N as initial_direction")
    if per_view_weights.shape != (v, n):
        raise ValueError("per_view_weights must have shape [V, N]")
    if viewmats.shape != (v, 4, 4):
        raise ValueError("viewmats must have shape [V, 4, 4]")
    if intrinsics.shape != (v, 3, 3):
        raise ValueError("intrinsics must have shape [V, 3, 3]")
    if knn.ndim != 2 or int(knn.shape[0]) != n:
        raise ValueError("knn must have shape [N, K]")
    if edge_weight.shape != knn.shape:
        raise ValueError("edge_weight must have shape [N, K]")
    if observed.ndim != 1 or int(observed.shape[0]) != n:
        raise ValueError("observed must have shape [N]")

    device = initial_direction.device
    if any(value.device != device for value in tensors.values()):
        raise ValueError("all fixed-axis ratio inputs must be on the same device")
    if any(value.is_complex() for value in tensors.values()):
        raise TypeError("fixed-axis ratio inputs must be real tensors")
    if knn.is_floating_point() or knn.dtype == torch.bool:
        raise TypeError("knn must be an integer tensor")
    if knn.numel() > 0 and bool(((knn < 0) | (knn >= n)).any()):
        raise ValueError("knn contains an out-of-range root index")

    floating_values = [value for value in tensors.values() if value.is_floating_point()]
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in floating_values) else torch.float32
    initial = _finite(initial_direction.to(dtype=dtype))
    tangent = _safe_normalize(tangent_axis.to(dtype=dtype))
    normal = _safe_normalize(normals.to(dtype=dtype))
    point = _finite(points.to(dtype=dtype))
    view_axis = _safe_normalize(per_view_axes.to(dtype=dtype))
    raw_weight = _finite(per_view_weights.to(dtype=dtype)).clamp_min(0.0)
    view = _finite(viewmats.to(dtype=dtype))
    intrinsic = _finite(intrinsics.to(dtype=dtype))
    edge = _finite(edge_weight.to(dtype=dtype)).clamp_min(0.0)
    observed_mask = observed.to(device=device, dtype=torch.bool)

    initial = _safe_normalize(initial)
    pre_normal = (initial * normal).sum(dim=-1)
    pre_tangent = initial - pre_normal[:, None] * normal
    pre_tangent_norm = _safe_length(pre_tangent)
    tangent_dot = (pre_tangent * tangent).sum(dim=-1)
    tangent_sign = torch.where(tangent_dot >= 0.0, torch.ones_like(tangent_dot), -torch.ones_like(tangent_dot))
    signed_axis = tangent_sign[:, None] * tangent
    rho0 = pre_normal / pre_tangent_norm.clamp_min(EPS)
    rho0 = _finite(rho0).clamp_min(0.0)

    tangent_for_views = signed_axis[None, :, :].expand(v, -1, -1)
    normal_for_views = normal[None, :, :].expand(v, -1, -1)
    projected_evidence = _project_fixed_axis_screen(point, view_axis, view, intrinsic)
    projected_tangent = _project_fixed_axis_screen(point, tangent_for_views, view, intrinsic)
    projected_normal = _project_fixed_axis_screen(point, normal_for_views, view, intrinsic)
    screen_evidence = _safe_normalize(projected_evidence)
    screen_valid = (
        _fixed_axis_screen_valid(projected_evidence)
        & _fixed_axis_screen_valid(projected_tangent)
        & _fixed_axis_screen_valid(projected_normal)
    )
    valid_weight = torch.where((raw_weight > 0.0) & screen_valid, raw_weight, torch.zeros_like(raw_weight))
    residual_weight = torch.where(raw_weight > 0.0, raw_weight, torch.zeros_like(raw_weight))

    screen_a = _finite(
        screen_evidence[..., 0] * projected_normal[..., 1]
        - screen_evidence[..., 1] * projected_normal[..., 0]
    )
    screen_b = _finite(
        screen_evidence[..., 0] * projected_tangent[..., 1]
        - screen_evidence[..., 1] * projected_tangent[..., 0]
    )

    accumulate_dtype = torch.float64
    valid_weight_acc = valid_weight.to(dtype=accumulate_dtype)
    residual_weight_acc = residual_weight.to(dtype=accumulate_dtype)
    a_acc = screen_a.to(dtype=accumulate_dtype)
    b_acc = screen_b.to(dtype=accumulate_dtype)
    rho0_acc = rho0.to(dtype=accumulate_dtype)
    numerator = _finite(torch.sum(_finite(valid_weight_acc * a_acc * b_acc), dim=0))
    denominator = _finite(torch.sum(_finite(valid_weight_acc * a_acc.square()), dim=0))
    fallback = denominator <= EPS
    rho_ls_acc = torch.where(fallback, rho0_acc, -numerator / denominator.clamp_min(EPS))
    rho_ls_acc = _finite(rho_ls_acc).clamp_min(0.0)
    rho_ls = rho_ls_acc.to(dtype=dtype)
    rho_ls = _finite(rho_ls).clamp_min(0.0)

    baseline_direction = _safe_normalize(rho0[:, None] * normal + signed_axis)
    ls_direction = _safe_normalize(rho_ls[:, None] * normal + signed_axis)

    residual0_num = _finite(
        torch.sum(
            _finite(residual_weight_acc * (a_acc * rho0_acc[None, :] + b_acc).square()),
            dim=0,
        )
    )
    rho_ls_acc_for_residual = rho_ls.to(dtype=accumulate_dtype)[None, :]
    residual1_num = _finite(
        torch.sum(
            _finite(residual_weight_acc * (a_acc * rho_ls_acc_for_residual + b_acc).square()),
            dim=0,
        )
    )
    weight_sum = _finite(valid_weight_acc.sum(dim=0))
    residual_before = _finite(residual0_num / weight_sum.clamp_min(EPS)).to(dtype=dtype)
    residual_after = _finite(residual1_num / weight_sum.clamp_min(EPS)).to(dtype=dtype)

    baseline_local_jump = _fixed_axis_local_max_jump(baseline_direction, normal, knn, edge, observed_mask)
    ls_local_jump = _fixed_axis_local_max_jump(ls_direction, normal, knn, edge, observed_mask)
    accept_mask = (
        observed_mask
        & ~fallback
        & (residual_after <= residual_before)
        & (ls_local_jump <= baseline_local_jump)
    )
    rho_ls_guarded = torch.where(accept_mask, rho_ls, rho0)
    rho_ls_guarded = _finite(rho_ls_guarded).clamp_min(0.0)
    guarded_direction = _safe_normalize(rho_ls_guarded[:, None] * normal + signed_axis)

    denominator_out = _finite(denominator).to(dtype=dtype)
    baseline_local_jump = _finite(baseline_local_jump).to(dtype=dtype)
    ls_local_jump = _finite(ls_local_jump).to(dtype=dtype)
    residual_before = _finite(residual_before)
    residual_after = _finite(residual_after)

    stats = {
        "rho0": _fixed_axis_qstats(rho0, observed_mask),
        "rho_ls": _fixed_axis_qstats(rho_ls, observed_mask),
        "rho_ls_guarded": _fixed_axis_qstats(rho_ls_guarded, observed_mask),
        "denominator": _fixed_axis_qstats(denominator_out, observed_mask),
        "screen_cross_residual_before": _fixed_axis_qstats(residual_before, observed_mask),
        "screen_cross_residual_after": _fixed_axis_qstats(residual_after, observed_mask),
        "baseline_local_jump_deg": _fixed_axis_qstats(baseline_local_jump, observed_mask),
        "ls_local_jump_deg": _fixed_axis_qstats(ls_local_jump, observed_mask),
    }
    counts = {
        "root_count": n,
        "view_count": v,
        "neighbor_count": int(knn.shape[1]),
        "observed_count": int(observed_mask.sum().detach().cpu().item()),
        "valid_evidence_pairs": int((valid_weight > 0.0).sum().detach().cpu().item()),
        "evidence_root_count": int((weight_sum > EPS).sum().detach().cpu().item()),
        "fallback_count": int(fallback.sum().detach().cpu().item()),
        "nonfallback_count": int((~fallback).sum().detach().cpu().item()),
        "ls_guarded_accept_count": int(accept_mask.sum().detach().cpu().item()),
        "ls_guarded_reject_count": int((~accept_mask).sum().detach().cpu().item()),
    }
    report = {
        "counts": counts,
        "statistics": stats,
        "rho0": stats["rho0"],
        "rho_ls": stats["rho_ls"],
        "rho_ls_guarded": stats["rho_ls_guarded"],
        "denominator": stats["denominator"],
        "screen_cross_residual_before": stats["screen_cross_residual_before"],
        "screen_cross_residual_after": stats["screen_cross_residual_after"],
        "baseline_local_jump_deg": stats["baseline_local_jump_deg"],
        "ls_local_jump_deg": stats["ls_local_jump_deg"],
    }

    return {
        "direction": _finite(guarded_direction),
        "guarded_direction": _finite(guarded_direction),
        "fixed_axis_rho0": _finite(baseline_direction),
        "fixed_axis_ls_ratio": _finite(ls_direction),
        "fixed_axis_ls_ratio_guarded": _finite(guarded_direction),
        "baseline_direction": _finite(baseline_direction),
        "ls_direction": _finite(ls_direction),
        "rho0": _finite(rho0),
        "rho_ls": _finite(rho_ls),
        "rho_ls_guarded": _finite(rho_ls_guarded),
        "guarded_ratio": _finite(rho_ls_guarded),
        "baseline_ratio": _finite(rho0),
        "ls_ratio": _finite(rho_ls),
        "ratio": _finite(rho_ls_guarded),
        "denominator": denominator_out,
        "fallback": fallback,
        "accept_mask": accept_mask,
        "ls_accept_mask": accept_mask,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "residual0": residual_before,
        "residual1": residual_after,
        "baseline_local_jump": baseline_local_jump,
        "ls_local_jump": ls_local_jump,
        "baseline_local_jump_deg": baseline_local_jump,
        "ls_local_jump_deg": ls_local_jump,
        "tangent_sign": _finite(tangent_sign),
        "weight_sum": weight_sum.to(dtype=dtype),
        "screen_evidence": _finite(screen_evidence),
        "screen_signed_tangent": _finite(projected_tangent),
        "screen_normal": _finite(projected_normal),
        "screen_A": _finite(screen_a).to(dtype=dtype),
        "screen_b": _finite(screen_b).to(dtype=dtype),
        "root_count": n,
        "view_count": v,
        "neighbor_count": int(knn.shape[1]),
        "observed_count": counts["observed_count"],
        "fallback_count": counts["fallback_count"],
        "ls_guarded_accept_count": counts["ls_guarded_accept_count"],
        "report": report,
        "stats": stats,
        "counts": counts,
    }

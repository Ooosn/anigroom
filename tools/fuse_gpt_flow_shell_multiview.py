from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import make_tangent_frames  # noqa: E402
from anigroom.flow.confidence_guided_direction import (  # noqa: E402
    refine_confidence_guided_directed_flow,
)
from anigroom.flow.direction_geometry import parallel_transport_vectors  # noqa: E402
from anigroom.flow.global_sign_orientation import refine_global_tangent_sign_field  # noqa: E402
from anigroom.flow.surface_graph import (  # noqa: E402
    SurfaceRootGraph,
    build_surface_root_graph,
)
from anigroom.flow.view_cluster_refinement import (  # noqa: E402
    refine_fixed_axis_multiview_ratio,
    refine_fixed_sign_directed_multiview_ratio,
    refine_trusted_multiview_axis_field,
)
from anigroom.mesh_roots import SurfaceRoots, TriangleMesh, initialize_surface_roots_fps, read_obj_mesh  # noqa: E402
from anigroom.mesh_roots import (  # noqa: E402
    initialize_surface_roots_from_candidates,
    sample_surface_candidates,
    weighted_farthest_point_sample,
)
from anigroom.projection import render_mesh_depth, sample_depth_nearest  # noqa: E402
from anigroom.projection.mesh_visibility import project_points  # noqa: E402
from tools.fuse_gpt_flow_multiview import (  # noqa: E402
    EPS,
    aligned_flow_strength,
    clean_directed_flow_on_graph,
    draw_root_flow_arrow_overlay,
    draw_root_flow_overlay,
    face_normals,
    knn_indices_chunked,
    orientation_from_line_strength,
    save_overlay,
)
from tools.train_white_tiger_stage1 import (  # noqa: E402
    bilinear_sample,
    load_camera_tensors,
    load_mask,
    project_directions,
    view_angle_weight,
)


def shell_candidates(
    root_points: torch.Tensor,
    root_normals: torch.Tensor,
    *,
    shell_count: int,
    shell_extent: float,
    spacing_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    knn, dist = knn_indices_chunked(root_points, k=max(2, int(spacing_k)), chunk=2048)
    del knn
    local_spacing = torch.median(dist, dim=1).values.clamp_min(1.0e-5)
    rel = torch.linspace(0.0, float(shell_extent), int(shell_count), device=root_points.device, dtype=root_points.dtype)
    shell_h = local_spacing[:, None] * rel[None]
    shell_points = root_points[:, None, :] + shell_h[..., None] * root_normals[:, None, :]
    return shell_points, shell_h, local_spacing


def sample_shell_visibility(
    points: torch.Tensor,
    normals: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    mesh_depth: torch.Tensor,
    *,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    local_depth_kernel: int,
    front_normal_z: float | None,
) -> dict[str, torch.Tensor]:
    height, width = int(mesh_depth.shape[0]), int(mesh_depth.shape[1])
    xy, depth, _ = project_points(points, viewmat, k)
    in_frame = (
        (depth > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= height - 1)
    )
    sampled_depth = sample_depth_nearest(mesh_depth, xy, kernel_size=int(local_depth_kernel))
    tolerance = float(depth_abs_tolerance) + depth.abs() * float(depth_rel_tolerance)
    # Shell points are on the visible hair layer, so they are allowed to be in
    # front of the mesh depth. They must not sit behind the body mesh.
    depth_visible = in_frame & torch.isfinite(sampled_depth) & (depth <= sampled_depth + tolerance)
    normal_cam = normals @ viewmat[:3, :3].T
    if front_normal_z is None:
        front_facing = torch.ones_like(depth_visible, dtype=torch.bool)
    else:
        front_facing = normal_cam[:, 2] <= float(front_normal_z)
    visible = depth_visible & front_facing
    return {
        "xy": xy,
        "depth": depth,
        "mesh_depth": sampled_depth,
        "depth_delta": depth - sampled_depth,
        "visible": visible,
        "in_frame": in_frame,
        "depth_visible": depth_visible,
        "front_facing": front_facing,
    }


def parse_float_list(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def load_surface_roots_file(path: Path, *, scale: float, translation: np.ndarray, space: str) -> tuple[SurfaceRoots, dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    if "points" in data.files:
        points = data["points"].astype(np.float32)
    elif "root_points" in data.files:
        points = data["root_points"].astype(np.float32)
    elif "root_positions" in data.files:
        points = data["root_positions"].astype(np.float32)
    else:
        raise KeyError(f"{path} does not contain points/root_points/root_positions")
    if "face_ids" not in data.files or "barycentric" not in data.files:
        raise KeyError(f"{path} must contain face_ids and barycentric")
    face_ids = data["face_ids"].astype(np.int64)
    barycentric = data["barycentric"].astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"root points must be [N, 3], got {points.shape}")
    if face_ids.shape[0] != points.shape[0] or barycentric.shape[:2] != (points.shape[0], 3):
        raise ValueError(f"root file has inconsistent shapes: points={points.shape}, face_ids={face_ids.shape}, barycentric={barycentric.shape}")
    if space == "raw":
        points = (points * float(scale) + translation[None]).astype(np.float32)
    elif space == "camera":
        points = points.astype(np.float32)
    else:
        raise ValueError(f"unknown root file space: {space}")
    selected = (
        data["selected_candidate_ids"].astype(np.int64)
        if "selected_candidate_ids" in data.files and data["selected_candidate_ids"].shape[0] == points.shape[0]
        else np.arange(points.shape[0], dtype=np.int64)
    )
    if "candidate_count" in data.files:
        candidate_count = int(np.asarray(data["candidate_count"]).reshape(-1)[0])
    else:
        candidate_count = int(points.shape[0])
    extras = {key: data[key] for key in data.files if key not in {"points", "root_points", "root_positions", "face_ids", "barycentric", "selected_candidate_ids", "candidate_count"}}
    return (
        SurfaceRoots(
            points=points.astype(np.float32),
            face_ids=face_ids,
            barycentric=barycentric,
            selected_candidate_ids=selected,
            candidate_count=candidate_count,
        ),
        extras,
    )


def in_frame_mask(xy: torch.Tensor, width: int, height: int) -> torch.Tensor:
    return (
        (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= int(width) - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= int(height) - 1)
    )


def build_region_knn_k_per_root(
    root_payload: dict[str, np.ndarray],
    *,
    key: str,
    head_knn_k: int,
    body_knn_k: int,
    fallback_knn_k: int,
    device: torch.device,
    root_count: int,
) -> torch.Tensor | None:
    """Return per-root K for head/body roots when region ids are available.

    Region id 1 is head and 0 is body, as written by
    build_white_tiger_smal_head_guides.py.  If head/body K are not requested,
    the caller keeps the old global-K behavior.
    """

    if int(head_knn_k) <= 0 and int(body_knn_k) <= 0:
        return None
    if key not in root_payload:
        raise ValueError(
            f"region-aware clean K was requested, but root payload has no '{key}'. "
            "Use a surface-roots file generated by build_white_tiger_smal_head_guides.py."
        )
    region_ids_np = np.asarray(root_payload[key]).reshape(-1).astype(np.int64)
    if int(region_ids_np.shape[0]) != int(root_count):
        raise ValueError(f"region id count {region_ids_np.shape[0]} does not match root count {root_count}")
    head_k = int(head_knn_k) if int(head_knn_k) > 0 else int(fallback_knn_k)
    body_k = int(body_knn_k) if int(body_knn_k) > 0 else int(fallback_knn_k)
    k_np = np.where(region_ids_np == 1, head_k, body_k).astype(np.int64)
    return torch.from_numpy(k_np).to(device=device)


def silhouette_band_confidence(
    target_conf: torch.Tensor,
    mesh_depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    mesh_dilate: int,
) -> torch.Tensor:
    """Confidence in the animal silhouette outside the projected furless mesh.

    Furless meshes miss visible outer fur, especially side tufts. This band is
    still constrained by the animal mask and GPT line confidence, but excludes
    the mesh projection so it only contributes evidence that normal-shell
    projection cannot sample directly.
    """

    mesh_inside = torch.isfinite(mesh_depth).float()
    k = max(1, int(mesh_dilate))
    if k % 2 == 0:
        k += 1
    if k > 1:
        mesh_inside = F.max_pool2d(mesh_inside[None, None], kernel_size=k, stride=1, padding=k // 2)[0, 0]
    band = (1.0 - mesh_inside.clamp(0.0, 1.0))[..., None]
    return target_conf * mask * band


def projected_normal_unit(
    points: torch.Tensor,
    normals: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    screen_normal = project_directions(points, normals, viewmat, k)
    screen_len = torch.linalg.norm(screen_normal, dim=-1)
    return screen_normal / screen_len.clamp_min(EPS)[..., None], screen_len


def recover_flow3d_from_screen_axis(
    sampled_ori: torch.Tensor,
    screen_t: torch.Tensor,
    screen_b: torch.Tensor,
    flat_tangents: torch.Tensor,
    flat_bitangents: torch.Tensor,
) -> torch.Tensor:
    coeff_direction, _ = _recover_tangent_axis(sampled_ori, screen_t, screen_b)
    return F.normalize(
        coeff_direction[:, 0:1] * flat_tangents + coeff_direction[:, 1:2] * flat_bitangents,
        dim=-1,
        eps=1.0e-8,
    )


def _recover_tangent_axis(
    sampled_ori: torch.Tensor,
    screen_t: torch.Tensor,
    screen_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    basis = torch.stack([screen_t, screen_b], dim=-1)
    coeff = (torch.linalg.pinv(basis) @ sampled_ori[:, :, None]).squeeze(-1)
    reconstructed = (basis @ coeff[:, :, None]).squeeze(-1)
    reprojection_fraction = torch.linalg.norm(reconstructed, dim=-1) / torch.linalg.norm(
        sampled_ori,
        dim=-1,
    ).clamp_min(EPS)
    coeff_direction = F.normalize(coeff, dim=-1, eps=1.0e-8)
    recovered_screen = (basis @ coeff_direction[:, :, None]).squeeze(-1)
    smax = torch.linalg.svdvals(basis)[..., 0]
    observability = (
        reprojection_fraction
        * torch.linalg.norm(recovered_screen, dim=-1)
        / smax.clamp_min(EPS)
    ).clamp(0.0, 1.0)
    return coeff_direction, observability


def tangent_axis_observability(
    sampled_ori: torch.Tensor,
    screen_t: torch.Tensor,
    screen_b: torch.Tensor,
) -> torch.Tensor:
    """Return continuous observability for normalized projected tangent axes."""
    return _recover_tangent_axis(sampled_ori, screen_t, screen_b)[1]


def accumulate_axis_evidence(
    *,
    flow3d_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    view_count: torch.Tensor,
    sampled_ori: torch.Tensor,
    weight_flat: torch.Tensor,
    screen_t: torch.Tensor,
    screen_b: torch.Tensor,
    flat_tangents: torch.Tensor,
    flat_bitangents: torch.Tensor,
    n_roots: int,
    n_shells: int,
    min_confidence: float,
    capture_contribution: bool = False,
) -> tuple[int, float, torch.Tensor | None, torch.Tensor | None]:
    good = weight_flat >= float(min_confidence)
    aligned_contribution = torch.zeros_like(flow3d_sum) if bool(capture_contribution) else None
    effective_weight = torch.zeros_like(weight_sum) if bool(capture_contribution) else None
    if bool(good.any()):
        coeff_direction, observability = _recover_tangent_axis(sampled_ori, screen_t, screen_b)
        flow3d_flat = F.normalize(
            coeff_direction[:, 0:1] * flat_tangents + coeff_direction[:, 1:2] * flat_bitangents,
            dim=-1,
            eps=1.0e-8,
        )
        flow3d = flow3d_flat.reshape(n_roots, n_shells, 3)
        weight = (weight_flat * observability).reshape(n_roots, n_shells)
        has_prev = weight_sum > 0.0
        prev = F.normalize(flow3d_sum, dim=-1, eps=1.0e-8)
        flip = has_prev & ((flow3d * prev).sum(dim=-1) < 0.0)
        flow3d = torch.where(flip[..., None], -flow3d, flow3d)
        good2 = good.reshape(n_roots, n_shells)
        if aligned_contribution is not None and effective_weight is not None:
            aligned_contribution[good2] = flow3d[good2] * weight[good2].unsqueeze(-1)
            effective_weight[good2] = weight[good2]
        flow3d_sum[good2] += flow3d[good2] * weight[good2].unsqueeze(-1)
        weight_sum[good2] += weight[good2]
        view_count[good2] += 1.0
    return (
        int(good.sum().detach().cpu()),
        float(weight_flat.sum().detach().cpu()),
        aligned_contribution,
        effective_weight,
    )


def collapse_per_view_shell_evidence(
    *,
    per_view_contribution: torch.Tensor,
    per_view_weight: torch.Tensor,
    per_view_direct_weight: torch.Tensor,
    global_weight_sum: torch.Tensor,
    shell_probability: torch.Tensor,
    shell_sign: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collapse shell evidence into the exact pre-clean per-view root decomposition."""

    if per_view_contribution.ndim != 4 or per_view_contribution.shape[-1] != 3:
        raise ValueError("per_view_contribution must have shape [V, N, S, 3]")
    expected_weight_shape = per_view_contribution.shape[:-1]
    if per_view_weight.shape != expected_weight_shape:
        raise ValueError("per_view_weight must have shape [V, N, S]")
    if per_view_direct_weight.shape != expected_weight_shape:
        raise ValueError("per_view_direct_weight must have shape [V, N, S]")
    root_shell_shape = per_view_contribution.shape[1:3]
    if global_weight_sum.shape != root_shell_shape:
        raise ValueError("global_weight_sum must have shape [N, S]")
    if shell_probability.shape != root_shell_shape:
        raise ValueError("shell_probability must have shape [N, S]")
    if shell_sign.shape not in {root_shell_shape, (*root_shell_shape, 1)}:
        raise ValueError("shell_sign must have shape [N, S] or [N, S, 1]")

    sign = shell_sign[..., None] if shell_sign.ndim == 2 else shell_sign
    per_view_vectors = (
        per_view_contribution
        / global_weight_sum.clamp_min(EPS)[None, ..., None]
        * shell_probability[None, ..., None]
        * sign[None]
    ).sum(dim=2)
    per_view_weights = (per_view_weight * shell_probability[None]).sum(dim=2)
    per_view_direct_weights = (per_view_direct_weight * shell_probability[None]).sum(dim=2)
    return per_view_vectors, per_view_weights, per_view_direct_weights


def root_graph_edges(
    points: torch.Tensor,
    normals: torch.Tensor,
    *,
    knn_k: int,
    knn_k_per_root: torch.Tensor | None = None,
    surface_graph: SurfaceRootGraph | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a local root graph for smoothing shell choices and directions."""

    if knn_k_per_root is not None:
        knn_k_per_root = knn_k_per_root.to(device=points.device, dtype=torch.long).reshape(-1)
        if int(knn_k_per_root.shape[0]) != int(points.shape[0]):
            raise ValueError(f"knn_k_per_root shape {tuple(knn_k_per_root.shape)} does not match root count {int(points.shape[0])}")
        effective_knn_k = max(int(knn_k), int(knn_k_per_root.max().detach().cpu()))
    else:
        effective_knn_k = int(knn_k)
    if surface_graph is None:
        knn, knn_dist = knn_indices_chunked(points, effective_knn_k, chunk=2048)
    else:
        if surface_graph.root_count != int(points.shape[0]):
            raise ValueError("surface graph root count does not match smoothing roots")
        if surface_graph.neighbor_count < effective_knn_k:
            raise ValueError(
                f"surface graph has K={surface_graph.neighbor_count}, but smoothing needs K={effective_knn_k}"
            )
        knn = surface_graph.indices[:, :effective_knn_k].to(device=points.device)
        knn_dist = surface_graph.distances[:, :effective_knn_k].to(device=points.device, dtype=points.dtype)
    if knn_k_per_root is not None:
        ranks = torch.arange(int(knn.shape[1]), device=points.device, dtype=torch.long)[None, :]
        active_edges = ranks < knn_k_per_root.clamp(1, int(knn.shape[1]))[:, None]
        kth_index = (knn_k_per_root.clamp(1, int(knn.shape[1])) - 1)[:, None]
        local_kth = knn_dist.gather(1, kth_index).reshape(-1)
        kth = local_kth[torch.isfinite(local_kth)].median().clamp_min(EPS)
    else:
        active_edges = torch.ones_like(knn_dist, dtype=torch.bool)
        kth = knn_dist[:, -1].median().clamp_min(EPS)
    dist_weight = torch.exp(-((knn_dist / kth) ** 2))
    normal_weight = ((normals[:, None, :] * normals[knn]).sum(dim=-1).clamp_min(0.0)) ** 2
    edge_weight = dist_weight * normal_weight * active_edges.float()
    return knn, edge_weight


def neighbor_vectors_in_target_frame(
    vectors: torch.Tensor,
    normals: torch.Tensor,
    knn: torch.Tensor,
    *,
    surface_graph: SurfaceRootGraph | None,
) -> torch.Tensor:
    neighbor_vectors = vectors[knn]
    if surface_graph is None:
        return neighbor_vectors
    source_normals = normals[knn]
    target_normals = normals[:, None, :].expand_as(source_normals)
    return parallel_transport_vectors(neighbor_vectors, source_normals, target_normals)


def face_curvature_score(mesh: TriangleMesh) -> np.ndarray:
    """Approximate local geometric complexity for each mesh face.

    The score is based on the disagreement between a face normal and the
    averaged normals at its vertices.  It is semantic-free: curved regions such
    as head, ears, paws, and tail tend to receive higher values, while broad
    body sheets stay low.
    """

    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int64)
    normals = face_normals(vertices, faces)
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1).astype(np.float32)
    vertex_normals = np.zeros_like(vertices, dtype=np.float32)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], normals * areas[:, None])
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=-1, keepdims=True), EPS)
    face_vertex_normals = vertex_normals[faces]
    agreement = np.sum(face_vertex_normals * normals[:, None, :], axis=-1).clip(-1.0, 1.0)
    return (1.0 - agreement.mean(axis=1)).astype(np.float32)


def robust01(values: torch.Tensor, valid: torch.Tensor | None = None, *, quantile: float = 0.90) -> torch.Tensor:
    selected = values if valid is None else values[valid]
    selected = selected[torch.isfinite(selected) & (selected > 0.0)]
    if selected.numel() == 0:
        return torch.zeros_like(values)
    scale = torch.quantile(selected, float(quantile)).clamp_min(EPS)
    return (values / scale).clamp(0.0, 1.0)


def score_surface_candidates_for_guide_roots(
    *,
    candidates_points_np: np.ndarray,
    candidates_face_ids: np.ndarray,
    mesh: TriangleMesh,
    views: list[int],
    data_root: Path,
    flow_dir: Path,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    width: int,
    height: int,
    device: torch.device,
    shell_count: int,
    shell_extent: float,
    shell_spacing_k: int,
    depth_abs_tolerance: float,
    depth_rel_tolerance: float,
    local_depth_kernel: int,
    front_normal_z: float | None,
    min_confidence: float,
    view_angle_power: float,
    silhouette_offsets: list[float],
    silhouette_band_weight: float,
    silhouette_mesh_dilate: int,
    silhouette_normal_screen_min: float,
    evidence_weight: float,
    geometry_weight: float,
    view_support_weight: float,
    importance_strength: float,
) -> dict[str, np.ndarray | float]:
    """Compute candidate density weights from multiview fur evidence.

    This is not a segmentation model.  It scores where additional guide roots
    are useful by combining actual projected fur-line evidence, silhouette-band
    evidence, and local surface complexity.
    """

    points = torch.from_numpy(candidates_points_np.astype(np.float32)).to(device=device)
    normals_np = face_normals(mesh.vertices, mesh.faces)[candidates_face_ids].astype(np.float32)
    normals = F.normalize(torch.from_numpy(normals_np).to(device=device), dim=-1, eps=EPS)
    shell_points, _, _ = shell_candidates(
        points,
        normals,
        shell_count=max(3, int(shell_count)),
        shell_extent=float(shell_extent),
        spacing_k=max(3, int(shell_spacing_k)),
    )
    n_candidates, n_shells = int(shell_points.shape[0]), int(shell_points.shape[1])
    flat_points = shell_points.reshape(-1, 3)
    flat_normals = normals[:, None, :].expand(n_candidates, n_shells, 3).reshape(-1, 3)

    direct_sum = torch.zeros((n_candidates,), device=device)
    band_sum = torch.zeros((n_candidates,), device=device)
    view_support = torch.zeros((n_candidates,), device=device)
    signed_offsets = [v for offset in silhouette_offsets for v in (-float(offset), float(offset))]

    for view_idx in views:
        flow_path = flow_dir / f"img_{view_idx:04d}.png"
        mask_path = data_root / "silhouette" / f"img_{view_idx:04d}.png"
        strength, _, _ = aligned_flow_strength(flow_path, mask_path, width, height)
        _, conf_np = orientation_from_line_strength(strength)
        conf = torch.from_numpy(conf_np).to(device=device)
        mask = load_mask(mask_path, device)
        target_conf = conf * mask
        mesh_depth = render_mesh_depth(mesh, viewmats[view_idx], ks[view_idx], width, height, device=device)
        shell_vis = sample_shell_visibility(
            flat_points,
            flat_normals,
            viewmats[view_idx],
            ks[view_idx],
            mesh_depth.depth,
            depth_abs_tolerance=float(depth_abs_tolerance),
            depth_rel_tolerance=float(depth_rel_tolerance),
            local_depth_kernel=int(local_depth_kernel),
            front_normal_z=front_normal_z,
        )
        angle_weight = view_angle_weight(flat_normals, viewmats[view_idx], float(view_angle_power))
        sampled_conf = bilinear_sample(target_conf, shell_vis["xy"])[:, 0]
        direct_weight = (
            sampled_conf
            * angle_weight
            * shell_vis["visible"].float()
        ).reshape(n_candidates, n_shells).amax(dim=1)
        direct_sum += direct_weight

        band_conf = silhouette_band_confidence(
            target_conf,
            mesh_depth.depth,
            mask,
            mesh_dilate=int(silhouette_mesh_dilate),
        )
        normal_screen, normal_screen_len = projected_normal_unit(flat_points, flat_normals, viewmats[view_idx], ks[view_idx])
        normal_screen_good = normal_screen_len >= float(silhouette_normal_screen_min)
        view_band = torch.zeros((n_candidates, n_shells), device=device)
        for offset in signed_offsets:
            band_xy = shell_vis["xy"] + normal_screen * float(offset)
            sampled_band_conf = bilinear_sample(band_conf, band_xy)[:, 0]
            band_weight = (
                sampled_band_conf
                * angle_weight
                * shell_vis["visible"].float()
                * normal_screen_good.float()
                * in_frame_mask(band_xy, width, height).float()
                * float(silhouette_band_weight)
            ).reshape(n_candidates, n_shells)
            view_band = torch.maximum(view_band, band_weight)
        band_weight = view_band.amax(dim=1)
        band_sum += band_weight
        view_support += ((direct_weight >= float(min_confidence)) | (band_weight >= float(min_confidence))).float()

    evidence = direct_sum + band_sum
    evidence_conf = robust01(evidence, quantile=0.90)
    view_conf = robust01(view_support, quantile=0.75)
    curvature_np = face_curvature_score(mesh)[candidates_face_ids].astype(np.float32)
    curvature = torch.from_numpy(curvature_np).to(device=device)
    geometry_conf = robust01(curvature, quantile=0.90)
    combined = (
        float(evidence_weight) * evidence_conf
        + float(geometry_weight) * geometry_conf
        + float(view_support_weight) * view_conf
    )
    combined = combined / max(float(evidence_weight) + float(geometry_weight) + float(view_support_weight), EPS)
    candidate_weight = 1.0 + float(importance_strength) * combined
    return {
        "candidate_weight": candidate_weight.detach().cpu().numpy().astype(np.float32),
        "candidate_evidence": evidence.detach().cpu().numpy().astype(np.float32),
        "candidate_evidence_conf": evidence_conf.detach().cpu().numpy().astype(np.float32),
        "candidate_geometry_conf": geometry_conf.detach().cpu().numpy().astype(np.float32),
        "candidate_view_conf": view_conf.detach().cpu().numpy().astype(np.float32),
        "candidate_view_support": view_support.detach().cpu().numpy().astype(np.float32),
        "weight_mean": float(candidate_weight.mean().detach().cpu()),
        "weight_p90": float(torch.quantile(candidate_weight, 0.90).detach().cpu()),
        "weight_max": float(candidate_weight.max().detach().cpu()),
    }


def robust_unit_confidence(values: torch.Tensor, valid: torch.Tensor, *, quantile: float = 0.90) -> torch.Tensor:
    """Map positive evidence to [0, 1] using a robust dataset scale."""

    selected = values[valid & torch.isfinite(values) & (values > 0.0)]
    if selected.numel() == 0:
        return torch.zeros_like(values)
    scale = torch.quantile(selected, float(quantile)).clamp_min(EPS)
    return (values / scale).clamp(0.0, 1.0)


def project_to_tangent(vectors: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    tangent = vectors - (vectors * normals).sum(dim=-1, keepdim=True) * normals
    return F.normalize(tangent, dim=-1, eps=EPS)


def regularize_axial_flow_on_graph(
    points: torch.Tensor,
    normals: torch.Tensor,
    axial_flow: torch.Tensor,
    observed: torch.Tensor,
    selected_weight: torch.Tensor,
    selected_view_count: torch.Tensor,
    axis_consistency: torch.Tensor,
    *,
    knn_k: int,
    knn_k_per_root: torch.Tensor | None = None,
    surface_graph: SurfaceRootGraph | None = None,
    iters: int,
    smooth_strength: float,
) -> dict[str, torch.Tensor]:
    """Turn noisy per-root 2D projections into a continuous 3D guide axis field.

    This pass is deliberately data-driven.  It does not know tiger parts or
    image coordinates.  A root is trusted only when it has enough multiview
    evidence, repeated visible support, a consistent vector average, and agrees
    with the local 3D neighborhood.  Weak roots are interpolated over the mesh
    graph instead of keeping their own noisy projected direction.
    """

    raw_axis = project_to_tangent(axial_flow, normals)
    axis = raw_axis.clone()
    knn, edge_weight = root_graph_edges(
        points,
        normals,
        knn_k=int(knn_k),
        knn_k_per_root=knn_k_per_root,
        surface_graph=surface_graph,
    )
    denom = edge_weight.sum(dim=1).clamp_min(EPS)

    evidence_conf = robust_unit_confidence(selected_weight, observed, quantile=0.90)
    view_conf = robust_unit_confidence(selected_view_count, observed, quantile=0.75)
    consistency_conf = axis_consistency.clamp(0.0, 1.0) * observed.float()

    anchor_conf = torch.zeros_like(selected_weight)
    local_agreement = torch.zeros_like(selected_weight)
    for _ in range(max(1, int(iters))):
        neighbor_axis = neighbor_vectors_in_target_frame(
            axis,
            normals,
            knn,
            surface_graph=surface_graph,
        )
        # Axis direction is unsigned here; align neighbors before averaging.
        sign = torch.where((axis[:, None, :] * neighbor_axis).sum(dim=-1, keepdim=True) >= 0.0, 1.0, -1.0)
        neighbor_axis = neighbor_axis * sign
        neighbor_mean = F.normalize(
            (edge_weight[..., None] * neighbor_axis).sum(dim=1) / denom[:, None],
            dim=-1,
            eps=EPS,
        )
        neighbor_mean = project_to_tangent(neighbor_mean, normals)
        local_agreement = (raw_axis * neighbor_mean).sum(dim=-1).abs().clamp(0.0, 1.0)
        anchor_conf = (evidence_conf * view_conf * consistency_conf * local_agreement).clamp(0.0, 1.0)
        neighbor_conf = (edge_weight * anchor_conf[knn]).sum(dim=1) / denom
        propagated_conf = torch.maximum(anchor_conf, 0.5 * neighbor_conf).clamp(0.0, 1.0)
        proposal = F.normalize(
            propagated_conf[:, None] * raw_axis + (1.0 - propagated_conf)[:, None] * neighbor_mean,
            dim=-1,
            eps=EPS,
        )
        proposal = project_to_tangent(proposal, normals)
        blend = (float(smooth_strength) * (1.0 - anchor_conf)).clamp(0.0, 1.0)
        axis = F.normalize((1.0 - blend)[:, None] * axis + blend[:, None] * proposal, dim=-1, eps=EPS)
        axis = project_to_tangent(axis, normals)

    return {
        "axis": axis.detach(),
        "raw_axis": raw_axis.detach(),
        "anchor_conf": anchor_conf.detach(),
        "local_agreement": local_agreement.detach(),
        "axis_consistency": axis_consistency.detach(),
        "evidence_conf": evidence_conf.detach(),
        "view_conf": view_conf.detach(),
    }


def smooth_shell_probabilities(
    shell_score: torch.Tensor,
    direction_weight: torch.Tensor,
    observed_shell: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    *,
    knn_k: int,
    knn_k_per_root: torch.Tensor | None = None,
    iters: int,
    strength: float,
    temperature: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth normal-shell layer selection without hard-coding a shell height.

    shell_score is accumulated 2D agreement; direction_weight is accumulated
    visibility/evidence. We first form per-shell agreement, then pass a shell
    probability distribution on the local root graph. High-evidence roots keep
    their own observation; ambiguous roots follow nearby roots with similar
    normals.
    """

    score = torch.where(
        observed_shell,
        shell_score / direction_weight.clamp_min(EPS),
        torch.full_like(shell_score, -1.0e4),
    )
    confidence = (direction_weight / max(float(anchor_weight), EPS)).clamp(0.0, 1.0)
    logits = score / max(float(temperature), EPS)
    logits = logits.masked_fill(~observed_shell, -1.0e4)
    prob = torch.softmax(logits, dim=1)

    if int(iters) <= 0 or float(strength) <= 0.0:
        return prob, logits, confidence

    knn, edge_weight = root_graph_edges(points, normals, knn_k=int(knn_k), knn_k_per_root=knn_k_per_root)
    denom = edge_weight.sum(dim=1, keepdim=True).clamp_min(EPS)
    root_conf = confidence.max(dim=1, keepdim=True).values

    for _ in range(int(iters)):
        neighbor_prob = (edge_weight[..., None] * prob[knn]).sum(dim=1) / denom
        prior_logits = torch.log(neighbor_prob.clamp_min(EPS))
        # Strong local evidence should not be overwritten; weak/noisy roots use
        # the neighborhood distribution to avoid salt-and-pepper shell jumps.
        mixed_strength = float(strength) * (1.0 - 0.75 * root_conf)
        logits = score / max(float(temperature), EPS) + mixed_strength * prior_logits
        logits = logits.masked_fill(~observed_shell, -1.0e4)
        prob = torch.softmax(logits, dim=1)
    return prob, logits, confidence


def refine_shell_height_on_graph(
    *,
    points: torch.Tensor,
    normals: torch.Tensor,
    shell_height: torch.Tensor,
    local_spacing: torch.Tensor,
    evidence_weight: torch.Tensor,
    observed: torch.Tensor,
    knn_k: int,
    knn_k_per_root: torch.Tensor | None,
    iters: int,
    strength: float,
    anchor_weight: float,
    max_normalized_height: float,
) -> dict[str, torch.Tensor | float | int]:
    """Smooth the selected normal-shell height as a mesh field.

    The shell height is a thickness/visibility cue, not a learned animal-part
    rule.  We smooth normalized height ``h / local_spacing`` over the same root
    graph used for direction cleaning, preserving roots with strong multiview
    evidence and letting weak roots follow nearby roots with compatible normals.
    """

    if int(iters) <= 0 or float(strength) <= 0.0:
        return {
            "height": shell_height.detach(),
            "confidence": torch.zeros_like(shell_height).detach(),
            "enabled": 0,
            "iters": int(iters),
            "strength": float(strength),
            "jump_before": 0.0,
            "jump_after": 0.0,
        }

    spacing = local_spacing.reshape(-1).clamp_min(EPS)
    normalized = (shell_height.reshape(-1) / spacing).clamp(0.0, float(max_normalized_height))
    observed = observed.reshape(-1).bool()
    evidence_weight = evidence_weight.reshape(-1).clamp_min(0.0)
    knn, edge_weight = root_graph_edges(points, normals, knn_k=int(knn_k), knn_k_per_root=knn_k_per_root)
    denom = edge_weight.sum(dim=1).clamp_min(EPS)
    valid_weight = evidence_weight[observed & torch.isfinite(evidence_weight) & (evidence_weight > 0.0)]
    scale = torch.quantile(valid_weight, 0.90).clamp_min(EPS) if valid_weight.numel() > 0 else evidence_weight.new_tensor(1.0)
    confidence = (evidence_weight / max(float(anchor_weight), EPS) / scale).clamp(0.0, 1.0) * observed.float()

    def mean_neighbor_jump(value: torch.Tensor) -> torch.Tensor:
        diff = (value[:, None] - value[knn]).abs()
        return (diff * edge_weight).sum() / edge_weight.sum().clamp_min(EPS)

    jump_before = mean_neighbor_jump(normalized)
    value = normalized
    for _ in range(int(iters)):
        neighbor_value = (edge_weight * value[knn]).sum(dim=1) / denom
        blend = (float(strength) * (1.0 - confidence)).clamp(0.0, 1.0)
        proposal = (1.0 - blend) * value + blend * neighbor_value
        value = torch.where(observed, proposal, neighbor_value)
        value = value.clamp(0.0, float(max_normalized_height))
    jump_after = mean_neighbor_jump(value)
    return {
        "height": (value * spacing).detach().clamp_min(0.0),
        "confidence": confidence.detach(),
        "enabled": 1,
        "iters": int(iters),
        "strength": float(strength),
        "anchor_weight": float(anchor_weight),
        "jump_before": float(jump_before.detach().cpu()),
        "jump_after": float(jump_after.detach().cpu()),
    }


def draw_point_heat_overlay(
    path: Path,
    base: np.ndarray,
    xy: torch.Tensor,
    valid: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    label: str,
    vmin: float,
    vmax: float,
    max_points: int = 7000,
) -> None:
    canvas = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    ids = torch.nonzero(valid & torch.isfinite(xy).all(dim=-1) & torch.isfinite(values), as_tuple=False).reshape(-1)
    if ids.numel() > int(max_points):
        score = weights[ids]
        _, order = torch.topk(score, k=int(max_points), largest=True)
        ids = ids[order]
    xy_np = xy[ids].detach().cpu().numpy()
    val_np = values[ids].detach().cpu().numpy()
    w_np = weights[ids].detach().cpu().numpy()
    denom = max(float(vmax) - float(vmin), 1.0e-8)
    for (x, y), value, weight in zip(xy_np, val_np, w_np):
        t = float(np.clip((float(value) - float(vmin)) / denom, 0.0, 1.0))
        color = cv2.applyColorMap(np.asarray([[int(round(255 * t))]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
        alpha = int(45 + 175 * float(np.clip(weight, 0.0, 1.0)))
        r = 2.0 + 2.0 * float(np.clip(weight, 0.0, 1.0))
        draw.ellipse((float(x - r), float(y - r), float(x + r), float(y + r)), fill=(int(color[2]), int(color[1]), int(color[0]), alpha))
    draw.rectangle((8, 8, min(canvas.size[0] - 8, 900), 48), fill=(255, 255, 255, 220))
    draw.text((16, 20), label, fill=(0, 0, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def make_contact_sheet(path: Path, items: list[tuple[str, Path]], *, thumb_width: int = 900) -> None:
    thumbs = []
    for label, image_path in items:
        image = Image.open(image_path).convert("RGB")
        scale = thumb_width / max(1, image.width)
        thumb = image.resize((thumb_width, int(round(image.height * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb.width, thumb.height + 34), (255, 255, 255))
        canvas.paste(thumb, (0, 34))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 10), label, fill=(0, 0, 0))
        thumbs.append(canvas)
    width = sum(t.width for t in thumbs)
    height = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for thumb in thumbs:
        sheet.paste(thumb, (x, 0))
        x += thumb.width
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def crop_image(path: Path, source: Path, box: tuple[int, int, int, int]) -> None:
    image = Image.open(source).convert("RGB")
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    image.crop((x0, y0, x1, y1)).save(path)


def direction_normal_ratio(direction: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    direction = F.normalize(direction, dim=-1, eps=EPS)
    normals = F.normalize(normals, dim=-1, eps=EPS)
    normal_component = (direction * normals).sum(dim=-1).clamp(0.0, 1.0)
    tangent_component = torch.sqrt((1.0 - normal_component.square()).clamp_min(EPS))
    return normal_component / tangent_component.clamp_min(EPS)


def collect_selected_tangent_axis_evidence(
    *,
    args: argparse.Namespace,
    views: list[int],
    selected_shell_points: torch.Tensor,
    root_normals: torch.Tensor,
    root_tangents: torch.Tensor,
    root_bitangents: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    mesh: TriangleMesh,
    width: int,
    height: int,
    observed: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Lift direct 2D axes exactly at the final selected shell points."""

    axis_rows: list[torch.Tensor] = []
    weight_rows: list[torch.Tensor] = []
    per_view_report: list[dict[str, int | float]] = []
    for view_idx in views:
        flow_path = args.flow_dir / f"img_{view_idx:04d}.png"
        mask_path = args.data_root / "silhouette" / f"img_{view_idx:04d}.png"
        strength, _, _ = aligned_flow_strength(flow_path, mask_path, width, height)
        ori_np, conf_np = orientation_from_line_strength(strength)
        ori = torch.from_numpy(ori_np).to(device=device)
        conf = torch.from_numpy(conf_np).to(device=device)
        mask = load_mask(mask_path, device)
        target_conf = conf * mask
        mesh_depth = render_mesh_depth(mesh, viewmats[view_idx], ks[view_idx], width, height, device=device)
        visibility = sample_shell_visibility(
            selected_shell_points,
            root_normals,
            viewmats[view_idx],
            ks[view_idx],
            mesh_depth.depth,
            depth_abs_tolerance=float(args.depth_abs_tolerance),
            depth_rel_tolerance=float(args.depth_rel_tolerance),
            local_depth_kernel=int(args.local_depth_kernel),
            front_normal_z=float(args.front_normal_z),
        )
        sampled_ori = F.normalize(bilinear_sample(ori, visibility["xy"]), dim=-1, eps=EPS)
        sampled_conf = bilinear_sample(target_conf, visibility["xy"])[:, 0]
        raw_weight = (
            sampled_conf
            * view_angle_weight(root_normals, viewmats[view_idx], float(args.view_angle_power))
            * visibility["visible"].float()
            * observed.float()
        ).clamp(0.0, 1.0)
        screen_t = project_directions(selected_shell_points, root_tangents, viewmats[view_idx], ks[view_idx])
        screen_b = project_directions(selected_shell_points, root_bitangents, viewmats[view_idx], ks[view_idx])
        coeff_direction, observability = _recover_tangent_axis(sampled_ori, screen_t, screen_b)
        axis = F.normalize(
            coeff_direction[:, 0:1] * root_tangents + coeff_direction[:, 1:2] * root_bitangents,
            dim=-1,
            eps=EPS,
        )
        good = raw_weight >= float(args.min_confidence)
        effective_weight = torch.where(good, raw_weight * observability, torch.zeros_like(raw_weight))
        axis_rows.append(axis.detach())
        weight_rows.append(effective_weight.detach())
        per_view_report.append(
            {
                "view": int(view_idx),
                "direct_root_count": int((effective_weight > 0.0).sum().detach().cpu()),
                "effective_weight_sum": float(effective_weight.sum().detach().cpu()),
            }
        )
    axes = torch.stack(axis_rows, dim=0)
    weights = torch.stack(weight_rows, dim=0)
    return axes, weights, {
        "view_count": int(len(axis_rows)),
        "direct_root_view_count": int((weights > 0.0).sum().detach().cpu()),
        "covered_root_count": int((weights > 0.0).any(dim=0).sum().detach().cpu()),
        "per_view": per_view_report,
    }


def collect_selected_direction_evidence(
    *,
    args,
    views: list[int],
    selected_shell_points: torch.Tensor,
    root_normals: torch.Tensor,
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    mesh: TriangleMesh,
    width: int,
    height: int,
    signed_silhouette_offsets: list[float],
    observed: torch.Tensor,
    device: torch.device,
) -> list[dict[str, torch.Tensor | int | bool]]:
    evidence: list[dict[str, torch.Tensor | int | bool]] = []
    for view_idx in views:
        flow_path = args.flow_dir / f"img_{view_idx:04d}.png"
        mask_path = args.data_root / "silhouette" / f"img_{view_idx:04d}.png"
        strength, _, _ = aligned_flow_strength(flow_path, mask_path, width, height)
        ori_np, conf_np = orientation_from_line_strength(strength)
        ori = torch.from_numpy(ori_np).to(device=device)
        conf = torch.from_numpy(conf_np).to(device=device)
        mask = load_mask(mask_path, device)
        target_conf = conf * mask
        mesh_depth = render_mesh_depth(mesh, viewmats[view_idx], ks[view_idx], width, height, device=device)
        shell_vis = sample_shell_visibility(
            selected_shell_points,
            root_normals,
            viewmats[view_idx],
            ks[view_idx],
            mesh_depth.depth,
            depth_abs_tolerance=float(args.depth_abs_tolerance),
            depth_rel_tolerance=float(args.depth_rel_tolerance),
            local_depth_kernel=int(args.local_depth_kernel),
            front_normal_z=float(args.front_normal_z),
        )
        normal_screen, normal_screen_len = projected_normal_unit(selected_shell_points, root_normals, viewmats[view_idx], ks[view_idx])
        normal_screen_good = normal_screen_len >= float(args.silhouette_normal_screen_min)
        sampled_ori = F.normalize(bilinear_sample(ori, shell_vis["xy"]), dim=-1, eps=EPS)
        sampled_conf = bilinear_sample(target_conf, shell_vis["xy"])[:, 0]
        angle_weight = view_angle_weight(root_normals, viewmats[view_idx], float(args.view_angle_power))
        direct_weight = (
            sampled_conf
            * angle_weight
            * shell_vis["visible"].float()
            * observed.float()
        ).clamp(0.0, 1.0)
        if bool((direct_weight >= float(args.min_confidence)).any()):
            evidence.append(
                {
                    "view": int(view_idx),
                    "axis": sampled_ori.detach(),
                    "weight": direct_weight.detach(),
                    "bias": torch.zeros_like(sampled_ori).detach(),
                    "has_bias": False,
                }
            )

        band_conf = silhouette_band_confidence(
            target_conf,
            mesh_depth.depth,
            mask,
            mesh_dilate=int(args.silhouette_mesh_dilate),
        )
        for offset in signed_silhouette_offsets:
            band_xy = shell_vis["xy"] + normal_screen * float(offset)
            sampled_band_conf = bilinear_sample(band_conf, band_xy)[:, 0]
            sampled_band_ori = F.normalize(bilinear_sample(ori, band_xy), dim=-1, eps=EPS)
            band_weight = (
                sampled_band_conf
                * angle_weight
                * shell_vis["visible"].float()
                * observed.float()
                * normal_screen_good.float()
                * in_frame_mask(band_xy, width, height).float()
                * float(args.silhouette_band_weight)
            ).clamp(0.0, 1.0)
            if bool((band_weight >= float(args.min_confidence)).any()):
                evidence.append(
                    {
                        "view": int(view_idx),
                        "axis": sampled_band_ori.detach(),
                        "weight": band_weight.detach(),
                        "bias": (normal_screen * (1.0 if float(offset) > 0.0 else -1.0)).detach(),
                        "has_bias": True,
                    }
                )
    return evidence


def optimize_continuous_direction_field(
    *,
    initial_direction: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    selected_shell_points: torch.Tensor,
    evidence: list[dict[str, torch.Tensor | int | bool]],
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    observed: torch.Tensor,
    direction_weight: torch.Tensor,
    knn_k: int,
    knn_k_per_root: torch.Tensor | None = None,
    surface_graph: SurfaceRootGraph | None = None,
    iters: int,
    lr: float,
    smooth_weight: float,
    anchor_weight: float,
    silhouette_sign_bias: float,
) -> dict[str, torch.Tensor]:
    if len(evidence) == 0 or int(iters) <= 0:
        direction = F.normalize(initial_direction, dim=-1, eps=EPS)
        lam = direction_normal_ratio(direction, normals)
        return {
            "flow": direction,
            "lambda": lam,
            "anchor_conf": torch.zeros_like(lam),
            "anchor": observed & (direction_weight > 0.0),
        }

    knn, edge_weight = root_graph_edges(
        points,
        normals,
        knn_k=int(knn_k),
        knn_k_per_root=knn_k_per_root,
        surface_graph=surface_graph,
    )
    confidence_scale = torch.quantile(direction_weight[direction_weight > 0.0], 0.90) if bool((direction_weight > 0.0).any()) else direction_weight.new_tensor(1.0)
    anchor_conf = (direction_weight / confidence_scale.clamp_min(EPS)).clamp(0.0, 1.0) * observed.float()
    anchor = observed & (anchor_conf >= 0.55)
    param = torch.nn.Parameter(F.normalize(initial_direction.detach(), dim=-1, eps=EPS))
    optimizer = torch.optim.Adam([param], lr=float(lr))
    observed_weight = observed.float()
    for _ in range(int(iters)):
        direction = F.normalize(param, dim=-1, eps=EPS)
        normal_dot = (direction * normals).sum(dim=-1, keepdim=True)
        direction = torch.where(normal_dot < 0.0, direction - 2.0 * normal_dot * normals, direction)
        direction = F.normalize(direction, dim=-1, eps=EPS)
        total = direction.sum() * 0.0
        total_weight = direction.new_tensor(0.0)
        for sample in evidence:
            view_idx = int(sample["view"])
            screen = F.normalize(
                project_directions(selected_shell_points, direction, viewmats[view_idx], ks[view_idx]),
                dim=-1,
                eps=EPS,
            )
            axis = sample["axis"].to(device=direction.device, dtype=direction.dtype)  # type: ignore[union-attr]
            weight = sample["weight"].to(device=direction.device, dtype=direction.dtype)  # type: ignore[union-attr]
            weight = weight * anchor_conf
            axis_agreement = (screen * axis).sum(dim=-1).square().clamp(0.0, 1.0)
            loss = 1.0 - axis_agreement
            if bool(sample["has_bias"]):
                bias = sample["bias"].to(device=direction.device, dtype=direction.dtype)  # type: ignore[union-attr]
                signed_agreement = (screen * bias).sum(dim=-1).clamp(0.0, 1.0)
                loss = (1.0 - float(silhouette_sign_bias)) * loss + float(silhouette_sign_bias) * (1.0 - signed_agreement)
            total = total + (loss * weight).sum()
            total_weight = total_weight + weight.sum()
        neighbor_direction = neighbor_vectors_in_target_frame(
            direction,
            normals,
            knn,
            surface_graph=surface_graph,
        )
        edge_diff = direction[:, None, :] - neighbor_direction
        smooth = (edge_diff.square().mean(dim=-1) * edge_weight).sum() / edge_weight.sum().clamp_min(EPS)
        anchor_loss = (1.0 - (direction * F.normalize(initial_direction, dim=-1, eps=EPS)).sum(dim=-1).clamp(-1.0, 1.0))
        anchor_loss = (anchor_loss * anchor_conf).sum() / anchor_conf.sum().clamp_min(EPS)
        loss_value = total / total_weight.clamp_min(EPS)
        loss_value = loss_value + float(smooth_weight) * smooth + float(anchor_weight) * anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        optimizer.step()
        with torch.no_grad():
            direction = F.normalize(param, dim=-1, eps=EPS)
            normal_dot = (direction * normals).sum(dim=-1, keepdim=True)
            direction = torch.where(normal_dot < 0.0, direction - 2.0 * normal_dot * normals, direction)
            direction = F.normalize(direction, dim=-1, eps=EPS)
            param.copy_(torch.where(observed_weight[:, None] > 0.0, direction, F.normalize(initial_direction, dim=-1, eps=EPS)))
    with torch.no_grad():
        direction = F.normalize(param.detach(), dim=-1, eps=EPS)
        normal_dot = (direction * normals).sum(dim=-1, keepdim=True)
        direction = torch.where(normal_dot < 0.0, direction - 2.0 * normal_dot * normals, direction)
        direction = F.normalize(direction, dim=-1, eps=EPS)
        lam = direction_normal_ratio(direction, normals)
    return {
        "flow": direction,
        "lambda": lam,
        "anchor_conf": anchor_conf.detach(),
        "anchor": anchor.detach(),
    }


def optimize_continuous_ratio_field(
    *,
    initial_direction: torch.Tensor,
    axial_flow: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    selected_shell_points: torch.Tensor,
    evidence: list[dict[str, torch.Tensor | int | bool]],
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    observed: torch.Tensor,
    direction_weight: torch.Tensor,
    knn_k: int,
    knn_k_per_root: torch.Tensor | None = None,
    surface_graph: SurfaceRootGraph | None = None,
    iters: int,
    lr: float,
    smooth_weight: float,
    anchor_weight: float,
    silhouette_sign_bias: float,
    robust_max_quantile: float,
    robust_max_scale: float,
) -> dict[str, torch.Tensor]:
    """Optimize a continuous outward/tangent ratio while preserving the flow axis.

    This is the formal path for the normal-shell target.  The image evidence
    should decide how much outward/lift component is needed, but it should not
    be allowed to freely rotate the local tangent axis into a noisy vector
    field.  Unlike the old lambda search, the ratio is continuous and has no
    hard maximum.
    """

    if len(evidence) == 0 or int(iters) <= 0:
        direction = F.normalize(initial_direction, dim=-1, eps=EPS)
        lam = direction_normal_ratio(direction, normals)
        return {
            "flow": direction,
            "lambda": lam,
            "anchor_conf": torch.zeros_like(lam),
            "anchor": observed & (direction_weight > 0.0),
        }

    axial_flow = F.normalize(axial_flow, dim=-1, eps=EPS)
    initial_direction = F.normalize(initial_direction, dim=-1, eps=EPS)
    sign = torch.where((initial_direction * axial_flow).sum(dim=-1) >= 0.0, torch.ones((points.shape[0],), device=points.device), -torch.ones((points.shape[0],), device=points.device))
    initial_ratio = direction_normal_ratio(initial_direction, normals).clamp_min(1.0e-4)
    initial_log_ratio = torch.log(initial_ratio.clamp_min(1.0e-5))
    raw = torch.log(torch.expm1(initial_ratio).clamp_min(1.0e-6))
    valid_initial_ratio = initial_ratio[observed & (direction_weight > 0.0)]
    if valid_initial_ratio.numel() > 0 and float(robust_max_quantile) > 0.0:
        q = float(max(0.50, min(0.999, robust_max_quantile)))
        ratio_hi = torch.quantile(valid_initial_ratio.detach(), q)
        ratio_hi = (ratio_hi * float(max(1.0, robust_max_scale))).clamp_min(1.0e-4)
    else:
        ratio_hi = initial_ratio.detach().max().clamp_min(1.0e-4)
    param = torch.nn.Parameter(raw.detach().clone())
    optimizer = torch.optim.Adam([param], lr=float(lr))

    knn, edge_weight = root_graph_edges(
        points,
        normals,
        knn_k=int(knn_k),
        knn_k_per_root=knn_k_per_root,
        surface_graph=surface_graph,
    )
    edge_denom = edge_weight.sum(dim=1).clamp_min(EPS)
    confidence_scale = torch.quantile(direction_weight[direction_weight > 0.0], 0.90) if bool((direction_weight > 0.0).any()) else direction_weight.new_tensor(1.0)
    anchor_conf = (direction_weight / confidence_scale.clamp_min(EPS)).clamp(0.0, 1.0) * observed.float()
    anchor = observed & (anchor_conf >= 0.55)
    observed_weight = observed.float()

    def build_direction(ratio_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = F.softplus(ratio_raw).clamp_min(1.0e-5)
        direction = F.normalize(ratio[:, None] * normals + sign[:, None] * axial_flow, dim=-1, eps=EPS)
        normal_dot = (direction * normals).sum(dim=-1, keepdim=True)
        direction = torch.where(normal_dot < 0.0, direction - 2.0 * normal_dot * normals, direction)
        return F.normalize(direction, dim=-1, eps=EPS), ratio

    for _ in range(int(iters)):
        direction, ratio = build_direction(param)
        total = direction.sum() * 0.0
        total_weight = direction.new_tensor(0.0)
        for sample in evidence:
            view_idx = int(sample["view"])
            screen = F.normalize(
                project_directions(selected_shell_points, direction, viewmats[view_idx], ks[view_idx]),
                dim=-1,
                eps=EPS,
            )
            axis = sample["axis"].to(device=direction.device, dtype=direction.dtype)  # type: ignore[union-attr]
            weight = sample["weight"].to(device=direction.device, dtype=direction.dtype)  # type: ignore[union-attr]
            weight = weight * anchor_conf
            axis_agreement = (screen * axis).sum(dim=-1).square().clamp(0.0, 1.0)
            loss = 1.0 - axis_agreement
            if bool(sample["has_bias"]):
                bias = sample["bias"].to(device=direction.device, dtype=direction.dtype)  # type: ignore[union-attr]
                signed_agreement = (screen * bias).sum(dim=-1).clamp(0.0, 1.0)
                loss = (1.0 - float(silhouette_sign_bias)) * loss + float(silhouette_sign_bias) * (1.0 - signed_agreement)
            total = total + (loss * weight).sum()
            total_weight = total_weight + weight.sum()
        neighbor_direction = neighbor_vectors_in_target_frame(
            direction,
            normals,
            knn,
            surface_graph=surface_graph,
        )
        edge_diff = direction[:, None, :] - neighbor_direction
        direction_smooth = (edge_diff.square().mean(dim=-1) * edge_weight).sum() / edge_weight.sum().clamp_min(EPS)
        log_ratio = torch.log(ratio.clamp_min(1.0e-5))
        ratio_diff = log_ratio[:, None] - log_ratio[knn]
        ratio_smooth = (ratio_diff.square() * edge_weight).sum() / edge_weight.sum().clamp_min(EPS)
        anchor_loss = (1.0 - (direction * initial_direction).sum(dim=-1).clamp(-1.0, 1.0))
        anchor_loss = (anchor_loss * anchor_conf).sum() / anchor_conf.sum().clamp_min(EPS)
        ratio_anchor = ((log_ratio - initial_log_ratio).square() * anchor_conf).sum() / anchor_conf.sum().clamp_min(EPS)
        loss_value = total / total_weight.clamp_min(EPS)
        loss_value = loss_value + float(smooth_weight) * (direction_smooth + ratio_smooth) + float(anchor_weight) * (anchor_loss + 0.25 * ratio_anchor)
        optimizer.zero_grad(set_to_none=True)
        loss_value.backward()
        optimizer.step()
        with torch.no_grad():
            _, ratio_after_step = build_direction(param)
            log_ratio = torch.log(ratio_after_step.clamp_min(1.0e-5))
            neighbor_log_ratio = (log_ratio[knn] * edge_weight).sum(dim=1) / edge_denom
            propagated_log_ratio = anchor_conf * log_ratio + (1.0 - anchor_conf) * neighbor_log_ratio
            propagated_ratio = torch.exp(propagated_log_ratio).clamp_min(1.0e-5)
            propagated_ratio = propagated_ratio.clamp(max=ratio_hi)
            propagated_raw = torch.log(torch.expm1(propagated_ratio).clamp_min(1.0e-6))
            param.copy_(torch.where(observed_weight > 0.0, propagated_raw, raw))

    with torch.no_grad():
        direction, ratio = build_direction(param)
    return {
        "flow": direction.detach(),
        "lambda": ratio.detach(),
        "anchor_conf": anchor_conf.detach(),
        "anchor": anchor.detach(),
        "robust_ratio_hi": ratio_hi.detach(),
    }


def refine_direction_with_anchor_consensus(
    *,
    points: torch.Tensor,
    normals: torch.Tensor,
    direction: torch.Tensor,
    anchor_confidence: torch.Tensor,
    observed: torch.Tensor,
    knn_k: int,
    knn_k_per_root: torch.Tensor | None,
    surface_graph: SurfaceRootGraph | None,
    iters: int,
    blend: float,
    anchor_threshold: float,
) -> dict[str, torch.Tensor]:
    """Propagate a smooth 3D direction field from sparse local anchor roots.

    Dense per-root anchors can preserve local projection noise.  This pass keeps
    only local confidence maxima as direction anchors, then lets the rest of the
    field follow the mesh-neighborhood consensus.  It is geometry/confidence
    driven and does not depend on any animal-specific region rule.
    """

    if int(iters) <= 0:
        return {
            "flow": direction.detach(),
            "anchor": observed & (anchor_confidence >= float(anchor_threshold)),
            "anchor_conf": anchor_confidence.detach(),
        }

    direction = F.normalize(direction, dim=-1, eps=EPS)
    anchor_confidence = anchor_confidence.clamp(0.0, 1.0) * observed.float()
    knn, edge_weight = root_graph_edges(
        points,
        normals,
        knn_k=int(knn_k),
        knn_k_per_root=knn_k_per_root,
        surface_graph=surface_graph,
    )
    active_edges = edge_weight > 0.0
    neighbor_conf = anchor_confidence[knn].masked_fill(~active_edges, -1.0)
    local_max = anchor_confidence >= (neighbor_conf.max(dim=1).values - 1.0e-6)
    anchor = observed & (anchor_confidence >= float(anchor_threshold)) & local_max
    anchor_conf = torch.where(anchor, anchor_confidence, torch.zeros_like(anchor_confidence))

    vector = direction
    for _ in range(int(iters)):
        denom = edge_weight.sum(dim=1).clamp_min(EPS)
        transported_neighbor = neighbor_vectors_in_target_frame(
            vector,
            normals,
            knn,
            surface_graph=surface_graph,
        )
        neighbor_vector = F.normalize(
            (edge_weight[..., None] * transported_neighbor).sum(dim=1) / denom[:, None],
            dim=-1,
            eps=EPS,
        )
        mix = (float(blend) * (1.0 - anchor_conf)).clamp(0.0, float(blend))
        vector = F.normalize((1.0 - mix[:, None]) * vector + mix[:, None] * neighbor_vector, dim=-1, eps=EPS)
        normal_dot = (vector * normals).sum(dim=-1, keepdim=True)
        vector = torch.where(normal_dot < 0.0, vector - 2.0 * normal_dot * normals, vector)
        vector = F.normalize(vector, dim=-1, eps=EPS)

    return {
        "flow": vector.detach(),
        "anchor": anchor.detach(),
        "anchor_conf": anchor_conf.detach(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normal-shell multiview GPT-flow fusion diagnostic.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk"))
    parser.add_argument("--mesh-path", type=Path, default=Path("D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj"))
    parser.add_argument("--flow-dir", type=Path, default=Path("D:/Downloads/tiger_hair_flow_36"))
    parser.add_argument("--output-dir", type=Path, default=Path("D:/petsgaussianhair/_downloads/tiger_hair_flow_36/shell_fused_formal_cleanflow"))
    parser.add_argument("--exclude", default="4,24,25")
    parser.add_argument("--root-count", type=int, default=4096)
    parser.add_argument("--candidate-multiplier", type=float, default=8.0)
    parser.add_argument("--root-sampling-mode", choices=["fps", "evidence-adaptive", "surface-roots-file"], default="evidence-adaptive")
    parser.add_argument("--surface-roots-file", type=Path)
    parser.add_argument("--surface-roots-file-space", choices=["raw", "camera"], default="raw")
    parser.add_argument("--root-importance-strength", type=float, default=2.0)
    parser.add_argument("--root-evidence-weight", type=float, default=0.55)
    parser.add_argument("--root-geometry-weight", type=float, default=0.30)
    parser.add_argument("--root-view-support-weight", type=float, default=0.15)
    parser.add_argument("--scale", type=float, default=1.28)
    parser.add_argument("--translation", default="0,0.32,0.02")
    parser.add_argument("--min-confidence", type=float, default=0.04)
    parser.add_argument("--depth-abs-tolerance", type=float, default=0.03)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.01)
    parser.add_argument("--local-depth-kernel", type=int, default=7)
    parser.add_argument("--front-normal-z", type=float, default=0.15)
    parser.add_argument("--view-angle-power", type=float, default=1.0)
    parser.add_argument("--diag-view", type=int, default=9)
    parser.add_argument("--direction-lambda-values", default="0.10,0.16,0.24,0.34,0.48,0.68,0.90")
    parser.add_argument("--direction-field-mode", choices=["discrete", "continuous", "continuous-ratio"], default="continuous-ratio")
    parser.add_argument("--continuous-direction-iters", type=int, default=90)
    parser.add_argument("--continuous-direction-lr", type=float, default=0.06)
    parser.add_argument("--continuous-direction-smooth-weight", type=float, default=0.35)
    parser.add_argument("--continuous-direction-anchor-weight", type=float, default=0.08)
    parser.add_argument("--continuous-ratio-robust-max-quantile", type=float, default=0.995)
    parser.add_argument("--continuous-ratio-robust-max-scale", type=float, default=1.35)
    parser.add_argument("--direction-consensus-iters", type=int, default=0)
    parser.add_argument("--direction-consensus-blend", type=float, default=0.45)
    parser.add_argument("--direction-consensus-anchor-threshold", type=float, default=0.75)
    parser.add_argument(
        "--directed-flow-propagation-mode",
        choices=["none", "confidence-guided"],
        default="none",
    )
    parser.add_argument("--axis-field-mode", choices=["raw", "anchor-propagated", "trusted-view-cluster"], default="trusted-view-cluster")
    parser.add_argument("--axis-field-iters", type=int, default=10)
    parser.add_argument("--axis-field-smooth-strength", type=float, default=0.65)
    parser.add_argument("--shell-count", type=int, default=9)
    parser.add_argument("--shell-extent", type=float, default=2.5)
    parser.add_argument("--shell-spacing-k", type=int, default=8)
    parser.add_argument("--shell-smooth-iters", type=int, default=6)
    parser.add_argument("--shell-smooth-strength", type=float, default=1.4)
    parser.add_argument("--shell-score-temperature", type=float, default=0.08)
    parser.add_argument("--shell-anchor-weight", type=float, default=0.7)
    parser.add_argument("--shell-height-smooth-iters", type=int, default=0)
    parser.add_argument("--shell-height-smooth-strength", type=float, default=0.35)
    parser.add_argument("--shell-height-anchor-weight", type=float, default=0.7)
    parser.add_argument("--silhouette-band-offsets", default="8,16,28,44,64")
    parser.add_argument("--silhouette-band-weight", type=float, default=0.75)
    parser.add_argument("--silhouette-mesh-dilate", type=int, default=9)
    parser.add_argument("--silhouette-normal-screen-min", type=float, default=0.35)
    parser.add_argument("--silhouette-sign-bias", type=float, default=0.35)
    parser.add_argument("--clean-knn-k", type=int, default=12)
    parser.add_argument("--clean-head-knn-k", type=int, default=0)
    parser.add_argument("--clean-body-knn-k", type=int, default=0)
    parser.add_argument("--clean-region-id-key", default="root_file_region_ids")
    parser.add_argument(
        "--root-neighborhood",
        choices=["euclidean", "mesh-geodesic"],
        default="euclidean",
        help="Use the accepted Euclidean v3 graph or an intrinsic mesh graph with vector transport.",
    )
    parser.add_argument("--clean-sign-iters", type=int, default=12)
    parser.add_argument("--clean-lambda-iters", type=int, default=4)
    parser.add_argument("--clean-vector-iters", type=int, default=6)
    parser.add_argument("--clean-anchor-margin", type=float, default=0.02)
    parser.add_argument("--clean-anchor-weight", type=float, default=0.5)
    parser.add_argument("--clean-smooth-strength", type=float, default=2.0)
    parser.add_argument("--clean-vector-blend", type=float, default=0.35)
    args = parser.parse_args()
    if (
        args.directed_flow_propagation_mode != "none"
        and args.axis_field_mode != "trusted-view-cluster"
    ):
        parser.error(
            "--directed-flow-propagation-mode requires "
            "--axis-field-mode trusted-view-cluster"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mesh-depth visibility; do not use a CPU fallback for this diagnostic.")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exclude = {int(v) for v in args.exclude.split(",") if v.strip()}
    views = [idx for idx in range(36) if idx not in exclude]
    translation = np.asarray([float(v) for v in args.translation.split(",")], dtype=np.float32)
    silhouette_offsets = sorted({abs(v) for v in parse_float_list(args.silhouette_band_offsets) if abs(v) > 0.0})
    signed_silhouette_offsets = [v for offset in silhouette_offsets for v in (-offset, offset)]

    raw_mesh = read_obj_mesh(args.mesh_path)
    vertices = (raw_mesh.vertices.astype(np.float32) * float(args.scale) + translation[None]).astype(np.float32)
    mesh = TriangleMesh(vertices=vertices, faces=raw_mesh.faces)
    normals_np = face_normals(mesh.vertices, mesh.faces)
    viewmats, ks = load_camera_tensors(args.data_root, device)
    width, height = Image.open(args.data_root / "images" / "img_0000.png").size

    root_sampling_report: dict[str, float | int | str] = {"mode": str(args.root_sampling_mode)}
    root_sampling_payload: dict[str, np.ndarray] = {}
    if args.root_sampling_mode == "surface-roots-file":
        if args.surface_roots_file is None:
            raise ValueError("--surface-roots-file is required when --root-sampling-mode=surface-roots-file")
        roots, root_file_extras = load_surface_roots_file(
            args.surface_roots_file,
            scale=float(args.scale),
            translation=translation,
            space=str(args.surface_roots_file_space),
        )
        root_sampling_report.update(
            {
                "surface_roots_file": str(args.surface_roots_file),
                "surface_roots_file_space": str(args.surface_roots_file_space),
                "root_count_from_file": int(roots.points.shape[0]),
                "candidate_count": int(roots.candidate_count),
            }
        )
        for key, value in root_file_extras.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == roots.points.shape[0] and value.dtype != object:
                root_sampling_payload[f"root_file_{key}"] = value
    elif args.root_sampling_mode == "evidence-adaptive":
        candidate_count = max(int(args.root_count), int(math.ceil(float(args.root_count) * float(args.candidate_multiplier))))
        candidates = sample_surface_candidates(mesh, candidate_count, seed=13)
        candidate_scores = score_surface_candidates_for_guide_roots(
            candidates_points_np=candidates.points,
            candidates_face_ids=candidates.face_ids,
            mesh=mesh,
            views=views,
            data_root=args.data_root,
            flow_dir=args.flow_dir,
            viewmats=viewmats,
            ks=ks,
            width=width,
            height=height,
            device=device,
            shell_count=max(3, min(int(args.shell_count), 5)),
            shell_extent=float(args.shell_extent),
            shell_spacing_k=int(args.shell_spacing_k),
            depth_abs_tolerance=float(args.depth_abs_tolerance),
            depth_rel_tolerance=float(args.depth_rel_tolerance),
            local_depth_kernel=int(args.local_depth_kernel),
            front_normal_z=float(args.front_normal_z),
            min_confidence=float(args.min_confidence),
            view_angle_power=float(args.view_angle_power),
            silhouette_offsets=silhouette_offsets,
            silhouette_band_weight=float(args.silhouette_band_weight),
            silhouette_mesh_dilate=int(args.silhouette_mesh_dilate),
            silhouette_normal_screen_min=float(args.silhouette_normal_screen_min),
            evidence_weight=float(args.root_evidence_weight),
            geometry_weight=float(args.root_geometry_weight),
            view_support_weight=float(args.root_view_support_weight),
            importance_strength=float(args.root_importance_strength),
        )
        selected_candidate_ids = weighted_farthest_point_sample(
            candidates.points,
            candidate_scores["candidate_weight"],  # type: ignore[arg-type]
            int(args.root_count),
            seed=13,
            device=device,
        )
        roots = initialize_surface_roots_from_candidates(candidates, selected_candidate_ids)
        for key in (
            "candidate_weight",
            "candidate_evidence",
            "candidate_evidence_conf",
            "candidate_geometry_conf",
            "candidate_view_conf",
            "candidate_view_support",
        ):
            value = candidate_scores[key]
            assert isinstance(value, np.ndarray)
            root_sampling_payload[f"root_sampling_{key}"] = value[selected_candidate_ids].astype(np.float32)
        root_sampling_report.update(
            {
                "candidate_count": int(candidate_count),
                "importance_strength": float(args.root_importance_strength),
                "evidence_weight": float(args.root_evidence_weight),
                "geometry_weight": float(args.root_geometry_weight),
                "view_support_weight": float(args.root_view_support_weight),
                "weight_mean": float(candidate_scores["weight_mean"]),
                "weight_p90": float(candidate_scores["weight_p90"]),
                "weight_max": float(candidate_scores["weight_max"]),
            }
        )
    else:
        roots = initialize_surface_roots_fps(
            mesh,
            int(args.root_count),
            candidate_multiplier=float(args.candidate_multiplier),
            seed=13,
            fps_device=device,
        )
        root_sampling_report.update({"candidate_count": int(roots.candidate_count)})
    root_points_np = roots.points.astype(np.float32)
    root_normals_np = normals_np[roots.face_ids]

    root_points = torch.from_numpy(root_points_np).to(device=device)
    root_normals = F.normalize(torch.from_numpy(root_normals_np).to(device=device), dim=-1, eps=1.0e-8)
    root_face_ids = torch.from_numpy(roots.face_ids.astype(np.int64)).to(device=device)
    root_barycentric = torch.from_numpy(roots.barycentric.astype(np.float32)).to(device=device)
    clean_knn_k_per_root = build_region_knn_k_per_root(
        root_sampling_payload,
        key=str(args.clean_region_id_key),
        head_knn_k=int(args.clean_head_knn_k),
        body_knn_k=int(args.clean_body_knn_k),
        fallback_knn_k=int(args.clean_knn_k),
        device=device,
        root_count=int(root_points.shape[0]),
    )
    if clean_knn_k_per_root is not None:
        root_sampling_report.update(
            {
                "clean_region_id_key": str(args.clean_region_id_key),
                "clean_head_knn_k": int(args.clean_head_knn_k) if int(args.clean_head_knn_k) > 0 else int(args.clean_knn_k),
                "clean_body_knn_k": int(args.clean_body_knn_k) if int(args.clean_body_knn_k) > 0 else int(args.clean_knn_k),
            }
        )
    surface_graph: SurfaceRootGraph | None = None
    surface_graph_report: dict[str, float | int | str] = {"mode": str(args.root_neighborhood)}
    if args.root_neighborhood == "mesh-geodesic":
        max_clean_k = int(args.clean_knn_k)
        if clean_knn_k_per_root is not None:
            max_clean_k = max(max_clean_k, int(clean_knn_k_per_root.max().detach().cpu()))
        graph_k = max_clean_k
        surface_graph = build_surface_root_graph(
            vertices=mesh.vertices,
            faces=mesh.faces,
            root_points=root_points_np,
            root_face_ids=roots.face_ids,
            k=graph_k,
            device=device,
        )
        surface_graph_report.update(surface_graph.report)
    root_tangents, root_bitangents = make_tangent_frames(root_normals)
    shell_points, shell_h, local_spacing = shell_candidates(
        root_points,
        root_normals,
        shell_count=int(args.shell_count),
        shell_extent=float(args.shell_extent),
        spacing_k=int(args.shell_spacing_k),
    )
    n_roots, n_shells = int(shell_points.shape[0]), int(shell_points.shape[1])
    flat_shell_points = shell_points.reshape(-1, 3)
    flat_normals = root_normals[:, None, :].expand(n_roots, n_shells, 3).reshape(-1, 3)
    flat_tangents = root_tangents[:, None, :].expand(n_roots, n_shells, 3).reshape(-1, 3)
    flat_bitangents = root_bitangents[:, None, :].expand(n_roots, n_shells, 3).reshape(-1, 3)

    flow3d_sum = torch.zeros((n_roots, n_shells, 3), device=device)
    weight_sum = torch.zeros((n_roots, n_shells), device=device)
    view_count = torch.zeros((n_roots, n_shells), device=device)
    per_view = []
    retain_view_cluster_evidence = args.axis_field_mode == "trusted-view-cluster"
    per_view_contribution_cpu: list[torch.Tensor] = []
    per_view_weight_cpu: list[torch.Tensor] = []
    per_view_direct_weight_cpu: list[torch.Tensor] = []

    for view_idx in views:
        flow_path = args.flow_dir / f"img_{view_idx:04d}.png"
        mask_path = args.data_root / "silhouette" / f"img_{view_idx:04d}.png"
        strength, _, align_matrix = aligned_flow_strength(flow_path, mask_path, width, height)
        ori_np, conf_np = orientation_from_line_strength(strength)
        ori = torch.from_numpy(ori_np).to(device=device)
        conf = torch.from_numpy(conf_np).to(device=device)
        mask = load_mask(mask_path, device)
        target_conf = conf * mask
        mesh_depth = render_mesh_depth(mesh, viewmats[view_idx], ks[view_idx], width, height, device=device)
        band_conf = silhouette_band_confidence(
            target_conf,
            mesh_depth.depth,
            mask,
            mesh_dilate=int(args.silhouette_mesh_dilate),
        )
        shell_vis = sample_shell_visibility(
            flat_shell_points,
            flat_normals,
            viewmats[view_idx],
            ks[view_idx],
            mesh_depth.depth,
            depth_abs_tolerance=float(args.depth_abs_tolerance),
            depth_rel_tolerance=float(args.depth_rel_tolerance),
            local_depth_kernel=int(args.local_depth_kernel),
            front_normal_z=float(args.front_normal_z),
        )
        screen_t = project_directions(flat_shell_points, flat_tangents, viewmats[view_idx], ks[view_idx])
        screen_b = project_directions(flat_shell_points, flat_bitangents, viewmats[view_idx], ks[view_idx])
        normal_screen, normal_screen_len = projected_normal_unit(flat_shell_points, flat_normals, viewmats[view_idx], ks[view_idx])
        normal_screen_good = normal_screen_len >= float(args.silhouette_normal_screen_min)
        angle_weight = view_angle_weight(flat_normals, viewmats[view_idx], float(args.view_angle_power))
        sampled_ori = F.normalize(bilinear_sample(ori, shell_vis["xy"]), dim=-1, eps=1.0e-8)
        sampled_conf = bilinear_sample(target_conf, shell_vis["xy"])[:, 0]
        weight_flat = (sampled_conf * angle_weight * shell_vis["visible"].float()).clamp(0.0, 1.0)
        direct_good, direct_weight_sum, direct_contribution, direct_effective_weight = accumulate_axis_evidence(
            flow3d_sum=flow3d_sum,
            weight_sum=weight_sum,
            view_count=view_count,
            sampled_ori=sampled_ori,
            weight_flat=weight_flat,
            screen_t=screen_t,
            screen_b=screen_b,
            flat_tangents=flat_tangents,
            flat_bitangents=flat_bitangents,
            n_roots=n_roots,
            n_shells=n_shells,
            min_confidence=float(args.min_confidence),
            capture_contribution=retain_view_cluster_evidence,
        )
        if retain_view_cluster_evidence:
            assert direct_contribution is not None
            assert direct_effective_weight is not None
            view_contribution = direct_contribution.clone()
            view_weight = direct_effective_weight.clone()
            view_direct_weight = direct_effective_weight.clone()
        band_good_total = 0
        band_weight_total = 0.0
        for offset in signed_silhouette_offsets:
            band_xy = shell_vis["xy"] + normal_screen * float(offset)
            sampled_band_conf = bilinear_sample(band_conf, band_xy)[:, 0]
            sampled_band_ori = F.normalize(bilinear_sample(ori, band_xy), dim=-1, eps=1.0e-8)
            band_weight = (
                sampled_band_conf
                * angle_weight
                * shell_vis["visible"].float()
                * normal_screen_good.float()
                * in_frame_mask(band_xy, width, height).float()
                * float(args.silhouette_band_weight)
            ).clamp(0.0, 1.0)
            band_good, band_weight_total_raw, band_contribution, band_effective_weight = accumulate_axis_evidence(
                flow3d_sum=flow3d_sum,
                weight_sum=weight_sum,
                view_count=view_count,
                sampled_ori=sampled_band_ori,
                weight_flat=band_weight,
                screen_t=screen_t,
                screen_b=screen_b,
                flat_tangents=flat_tangents,
                flat_bitangents=flat_bitangents,
                n_roots=n_roots,
                n_shells=n_shells,
                min_confidence=float(args.min_confidence),
                capture_contribution=retain_view_cluster_evidence,
            )
            band_good_total += band_good
            band_weight_total += band_weight_total_raw
            if retain_view_cluster_evidence:
                assert band_contribution is not None
                assert band_effective_weight is not None
                view_contribution += band_contribution
                view_weight += band_effective_weight
        if retain_view_cluster_evidence:
            per_view_contribution_cpu.append(view_contribution.detach().cpu())
            per_view_weight_cpu.append(view_weight.detach().cpu())
            per_view_direct_weight_cpu.append(view_direct_weight.detach().cpu())
        per_view.append(
            {
                "view": int(view_idx),
                "good_shell_samples": int(direct_good),
                "weight_sum": float(direct_weight_sum),
                "band_good_shell_samples": int(band_good_total),
                "band_weight_sum": float(band_weight_total),
                "align_matrix": align_matrix.tolist(),
            }
        )

    observed_shell = weight_sum > 0.0
    flow3d_mean_shell = torch.where(
        observed_shell[..., None],
        flow3d_sum / weight_sum.clamp_min(EPS)[..., None],
        torch.zeros_like(flow3d_sum),
    )
    axis_consistency_shell = torch.linalg.norm(flow3d_mean_shell, dim=-1).clamp(0.0, 1.0)
    flow3d_shell = F.normalize(flow3d_mean_shell, dim=-1, eps=1.0e-8)

    lambda_values = torch.tensor([float(v) for v in args.direction_lambda_values.split(",") if v.strip()], device=device, dtype=torch.float32)
    if lambda_values.numel() == 0:
        raise ValueError("--direction-lambda-values must contain at least one value")
    direction_score = torch.zeros((n_roots, n_shells, int(lambda_values.numel()), 2), device=device)
    direction_weight = torch.zeros((n_roots, n_shells), device=device)

    for view_idx in views:
        flow_path = args.flow_dir / f"img_{view_idx:04d}.png"
        mask_path = args.data_root / "silhouette" / f"img_{view_idx:04d}.png"
        strength, _, _ = aligned_flow_strength(flow_path, mask_path, width, height)
        ori_np, conf_np = orientation_from_line_strength(strength)
        ori = torch.from_numpy(ori_np).to(device=device)
        conf = torch.from_numpy(conf_np).to(device=device)
        mask = load_mask(mask_path, device)
        target_conf = conf * mask
        mesh_depth = render_mesh_depth(mesh, viewmats[view_idx], ks[view_idx], width, height, device=device)
        band_conf = silhouette_band_confidence(
            target_conf,
            mesh_depth.depth,
            mask,
            mesh_dilate=int(args.silhouette_mesh_dilate),
        )
        shell_vis = sample_shell_visibility(
            flat_shell_points,
            flat_normals,
            viewmats[view_idx],
            ks[view_idx],
            mesh_depth.depth,
            depth_abs_tolerance=float(args.depth_abs_tolerance),
            depth_rel_tolerance=float(args.depth_rel_tolerance),
            local_depth_kernel=int(args.local_depth_kernel),
            front_normal_z=float(args.front_normal_z),
        )
        normal_screen, normal_screen_len = projected_normal_unit(flat_shell_points, flat_normals, viewmats[view_idx], ks[view_idx])
        normal_screen_good = normal_screen_len >= float(args.silhouette_normal_screen_min)
        sampled_ori = F.normalize(bilinear_sample(ori, shell_vis["xy"]), dim=-1, eps=1.0e-8)
        sampled_conf = bilinear_sample(target_conf, shell_vis["xy"])[:, 0]
        angle_weight = view_angle_weight(flat_normals, viewmats[view_idx], float(args.view_angle_power))
        weight_flat = (
            sampled_conf
            * angle_weight
            * shell_vis["visible"].float()
            * observed_shell.reshape(-1).float()
        ).clamp(0.0, 1.0)
        good = weight_flat >= float(args.min_confidence)
        evidence_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
        if bool(good.any()):
            weight = weight_flat.reshape(n_roots, n_shells)
            direction_weight += weight
            evidence_samples.append((sampled_ori, weight, None))
        for offset in signed_silhouette_offsets:
            band_xy = shell_vis["xy"] + normal_screen * float(offset)
            sampled_band_conf = bilinear_sample(band_conf, band_xy)[:, 0]
            sampled_band_ori = F.normalize(bilinear_sample(ori, band_xy), dim=-1, eps=1.0e-8)
            band_weight_flat = (
                sampled_band_conf
                * angle_weight
                * shell_vis["visible"].float()
                * observed_shell.reshape(-1).float()
                * normal_screen_good.float()
                * in_frame_mask(band_xy, width, height).float()
                * float(args.silhouette_band_weight)
            ).clamp(0.0, 1.0)
            band_good = band_weight_flat >= float(args.min_confidence)
            if bool(band_good.any()):
                band_weight = band_weight_flat.reshape(n_roots, n_shells)
                direction_weight += band_weight
                evidence_samples.append((sampled_band_ori, band_weight, normal_screen * (1.0 if float(offset) > 0.0 else -1.0)))
        if len(evidence_samples) == 0:
            continue
        for li, lam in enumerate(lambda_values):
            for si, sign in enumerate((1.0, -1.0)):
                candidate = F.normalize(
                    lam * root_normals[:, None, :] + float(sign) * flow3d_shell,
                    dim=-1,
                    eps=1.0e-8,
                )
                screen = F.normalize(
                    project_directions(
                        flat_shell_points,
                        candidate.reshape(-1, 3),
                        viewmats[view_idx],
                        ks[view_idx],
                    ),
                    dim=-1,
                    eps=1.0e-8,
                )
                for sample_ori, sample_weight, bias_screen in evidence_samples:
                    axis_agreement = torch.abs((screen * sample_ori).sum(dim=-1)).clamp(0.0, 1.0)
                    if bias_screen is not None:
                        biased_outward_agreement = (screen * bias_screen).sum(dim=-1).clamp(0.0, 1.0)
                        agreement = (1.0 - float(args.silhouette_sign_bias)) * axis_agreement + float(args.silhouette_sign_bias) * biased_outward_agreement
                    else:
                        agreement = axis_agreement
                    direction_score[:, :, li, si] += agreement.reshape(n_roots, n_shells) * sample_weight

    per_shell_stride = int(lambda_values.numel()) * 2
    flat_raw = direction_score.view(n_roots, -1)
    _, raw_best_flat = torch.max(flat_raw, dim=1)
    raw_best_shell_idx = torch.div(raw_best_flat, per_shell_stride, rounding_mode="floor")
    shell_score = direction_score.amax(dim=(2, 3))
    shell_prob, shell_logits, shell_confidence = smooth_shell_probabilities(
        shell_score,
        direction_weight,
        observed_shell,
        root_points,
        root_normals,
        knn_k=int(args.clean_knn_k),
        knn_k_per_root=clean_knn_k_per_root,
        iters=int(args.shell_smooth_iters),
        strength=float(args.shell_smooth_strength),
        temperature=float(args.shell_score_temperature),
        anchor_weight=float(args.shell_anchor_weight),
    )
    best_shell_idx = torch.max(shell_prob, dim=1).indices
    raw_selected_h = shell_h[torch.arange(n_roots, device=device), raw_best_shell_idx]
    ref_axis = flow3d_shell[torch.arange(n_roots, device=device), best_shell_idx]
    axis_sign = torch.where(
        (flow3d_shell * ref_axis[:, None, :]).sum(dim=-1, keepdim=True) >= 0.0,
        torch.ones_like(shell_h[..., None]),
        -torch.ones_like(shell_h[..., None]),
    )
    aligned_shell_axis = flow3d_shell * axis_sign
    selected_axis = F.normalize((shell_prob[..., None] * aligned_shell_axis).sum(dim=1), dim=-1, eps=1.0e-8)
    raw_selected_axis = selected_axis
    axis_view_cluster_npz: dict[str, np.ndarray] = {}
    per_view_vectors: torch.Tensor | None = None
    per_view_weights: torch.Tensor | None = None
    per_view_direct_weights: torch.Tensor | None = None
    if retain_view_cluster_evidence:
        if len(per_view_contribution_cpu) != len(views):
            raise RuntimeError("trusted-view-cluster evidence count does not match used views")
        shell_probability_cpu = shell_prob.detach().cpu()
        shell_sign_cpu = axis_sign.detach().cpu()
        global_weight_sum_cpu = weight_sum.detach().cpu()
        contribution_cpu = torch.stack(per_view_contribution_cpu, dim=0)
        combined_weight_cpu = torch.stack(per_view_weight_cpu, dim=0)
        direct_weight_cpu = torch.stack(per_view_direct_weight_cpu, dim=0)
        per_view_vectors_cpu, per_view_weights_cpu, per_view_direct_weights_cpu = collapse_per_view_shell_evidence(
            per_view_contribution=contribution_cpu,
            per_view_weight=combined_weight_cpu,
            per_view_direct_weight=direct_weight_cpu,
            global_weight_sum=global_weight_sum_cpu,
            shell_probability=shell_probability_cpu,
            shell_sign=shell_sign_cpu,
        )
        per_view_vectors = per_view_vectors_cpu.to(device=device, dtype=flow3d_sum.dtype)
        per_view_weights = per_view_weights_cpu.to(device=device, dtype=flow3d_sum.dtype)
        per_view_direct_weights = per_view_direct_weights_cpu.to(device=device, dtype=flow3d_sum.dtype)
    selected_axis_consistency = (shell_prob * axis_consistency_shell).sum(dim=1)
    selected_h = (shell_prob * shell_h).sum(dim=1)
    selected_weight = (shell_prob * direction_weight).sum(dim=1)
    selected_view_count = (shell_prob * view_count).sum(dim=1)
    observed = selected_weight > 0.0
    shell_height_refine = refine_shell_height_on_graph(
        points=root_points,
        normals=root_normals,
        shell_height=selected_h,
        local_spacing=local_spacing,
        evidence_weight=selected_weight,
        observed=observed,
        knn_k=int(args.clean_knn_k),
        knn_k_per_root=clean_knn_k_per_root,
        iters=int(args.shell_height_smooth_iters),
        strength=float(args.shell_height_smooth_strength),
        anchor_weight=float(args.shell_height_anchor_weight),
        max_normalized_height=float(args.shell_extent),
    )
    shell_height_refine_report = {
        key: value
        for key, value in shell_height_refine.items()
        if key not in {"height", "confidence"}
    }
    selected_h = shell_height_refine["height"]  # type: ignore[assignment]
    selected_shell_points = root_points + selected_h[:, None] * root_normals
    axis_field_report: dict[str, object] = {"mode": str(args.axis_field_mode)}
    trusted_knn: torch.Tensor | None = None
    trusted_edge_weight: torch.Tensor | None = None
    selected_direct_axes: torch.Tensor | None = None
    selected_direct_weights: torch.Tensor | None = None
    selected_direct_report: dict[str, object] | None = None
    if args.axis_field_mode == "trusted-view-cluster":
        assert per_view_vectors is not None
        assert per_view_weights is not None
        assert per_view_direct_weights is not None
        trusted_knn, trusted_edge_weight = root_graph_edges(
            root_points,
            root_normals,
            knn_k=int(args.clean_knn_k),
            knn_k_per_root=clean_knn_k_per_root,
            surface_graph=surface_graph,
        )
        selected_direct_axes, selected_direct_weights, selected_direct_report = collect_selected_tangent_axis_evidence(
            args=args,
            views=views,
            selected_shell_points=selected_shell_points,
            root_normals=root_normals,
            root_tangents=root_tangents,
            root_bitangents=root_bitangents,
            viewmats=viewmats,
            ks=ks,
            mesh=mesh,
            width=width,
            height=height,
            observed=observed,
            device=device,
        )
        trusted_result = refine_trusted_multiview_axis_field(
            initial_axis=raw_selected_axis,
            normals=root_normals,
            observed=observed,
            per_view_vectors=per_view_vectors,
            per_view_weights=per_view_weights,
            per_view_direct_weights=per_view_direct_weights,
            knn=trusted_knn,
            edge_weight=trusted_edge_weight,
        )
        selected_axis = trusted_result["axis"]
        axis_field = {
            "axis": selected_axis,
            "raw_axis": raw_selected_axis,
            "anchor_conf": trusted_result["confidence"],
            "local_agreement": trusted_result["local_agreement"],
            "axis_consistency": trusted_result["trust"],
            "evidence_conf": trusted_result["evidence_conf"],
            "view_conf": trusted_result["view_conf"],
        }

        def trusted_npz_array(key: str, *, dtype: np.dtype) -> np.ndarray:
            value = trusted_result[key]
            return value.detach().cpu().numpy().astype(dtype)

        axis_view_cluster_npz.update(
            {
                "axis_view_cluster_per_view_vectors": per_view_vectors_cpu.numpy().astype(np.float32),
                "axis_view_cluster_per_view_weight": per_view_weights_cpu.numpy().astype(np.float32),
                "axis_view_cluster_per_view_direct_weight": per_view_direct_weights_cpu.numpy().astype(np.float32),
                "axis_view_cluster_selected_direct_vectors": selected_direct_axes.detach().cpu().numpy().astype(np.float32),
                "axis_view_cluster_selected_direct_weight": selected_direct_weights.detach().cpu().numpy().astype(np.float32),
                "axis_view_cluster_trust": trusted_npz_array("trust", dtype=np.dtype(np.float32)),
                "axis_view_cluster_spectral_gap": trusted_npz_array("spectral_gap", dtype=np.dtype(np.float32)),
                "axis_view_cluster_n_eff": trusted_npz_array("n_eff", dtype=np.dtype(np.float32)),
                "axis_view_cluster_hard_margin": trusted_npz_array("hard_margin", dtype=np.dtype(np.float32)),
                "axis_view_cluster_q95_mask": trusted_npz_array("q95_mask", dtype=np.dtype(np.bool_)),
                "axis_view_cluster_residual_degrees": trusted_npz_array("residual_degrees", dtype=np.dtype(np.float32)),
                "axis_view_cluster_residual_mask": trusted_npz_array("residual_mask", dtype=np.dtype(np.bool_)),
                "axis_view_cluster_direct_support": trusted_npz_array("direct_support", dtype=np.dtype(np.float32)),
                "axis_view_cluster_supermajority_mask": trusted_npz_array("supermajority_mask", dtype=np.dtype(np.bool_)),
                "axis_view_cluster_final_confidence": trusted_npz_array("final_confidence", dtype=np.dtype(np.float32)),
            }
        )
        trusted_confidence = trusted_result["confidence"]
        raw_axis_unit = F.normalize(raw_selected_axis, dim=-1, eps=EPS)
        selected_axis_unit = F.normalize(selected_axis, dim=-1, eps=EPS)
        axis_change = torch.acos(
            (raw_axis_unit * selected_axis_unit).sum(dim=-1).abs().clamp(0.0, 1.0)
        ) * (180.0 / math.pi)
        axis_change_valid = observed & torch.isfinite(axis_change)
        trusted_report = trusted_result.get("report", {})
        if not isinstance(trusted_report, dict):
            trusted_report = {}
        trusted_constants = trusted_report.get("constants", trusted_result.get("constants", {}))
        trusted_cutoffs = trusted_report.get("cutoffs", trusted_result.get("cutoffs", {}))
        trusted_counts = dict(trusted_report.get("counts", trusted_result.get("counts", {})))
        trusted_counts.setdefault("observed_roots", int(observed.sum().detach().cpu()))
        trusted_counts.setdefault("q95_roots", int(trusted_result["q95_mask"].sum().detach().cpu()))
        trusted_counts.setdefault("residual_roots", int(trusted_result["residual_mask"].sum().detach().cpu()))
        trusted_counts.setdefault(
            "direct_supermajority_roots",
            int((trusted_result["direct_support"] >= (2.0 / 3.0)).sum().detach().cpu()),
        )
        trusted_counts.setdefault("supermajority_roots", int(trusted_result["supermajority_mask"].sum().detach().cpu()))
        axis_field_report.update(
            {
                "constants": trusted_constants,
                "cutoffs": trusted_cutoffs,
                "counts": trusted_counts,
                "trusted_view_cluster": trusted_report,
                "selected_direct_evidence": selected_direct_report,
                "change_from_raw_median_degrees": float(torch.median(axis_change[axis_change_valid]).detach().cpu()) if bool(axis_change_valid.any()) else 0.0,
                "change_from_raw_p90_degrees": float(torch.quantile(axis_change[axis_change_valid], 0.90).detach().cpu()) if bool(axis_change_valid.any()) else 0.0,
                "final_confidence_mean": float(trusted_confidence[observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
                "final_confidence_p90": float(torch.quantile(trusted_confidence[observed], 0.90).detach().cpu()) if bool(observed.any()) else 0.0,
            }
        )
    elif args.axis_field_mode == "anchor-propagated":
        axis_field = regularize_axial_flow_on_graph(
            root_points,
            root_normals,
            selected_axis,
            observed,
            selected_weight,
            selected_view_count,
            selected_axis_consistency,
            knn_k=int(args.clean_knn_k),
            knn_k_per_root=clean_knn_k_per_root,
            surface_graph=surface_graph,
            iters=int(args.axis_field_iters),
            smooth_strength=float(args.axis_field_smooth_strength),
        )
        selected_axis = axis_field["axis"]
        axis_field_report.update(
            {
                "iters": int(args.axis_field_iters),
                "smooth_strength": float(args.axis_field_smooth_strength),
                "anchor_conf_mean": float(axis_field["anchor_conf"][observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
                "anchor_conf_p90": float(torch.quantile(axis_field["anchor_conf"][observed], 0.90).detach().cpu()) if bool(observed.any()) else 0.0,
                "local_agreement_mean": float(axis_field["local_agreement"][observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
                "axis_consistency_mean": float(axis_field["axis_consistency"][observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
            }
        )
    else:
        axis_field = {
            "axis": selected_axis,
            "raw_axis": raw_selected_axis,
            "anchor_conf": selected_weight.new_zeros(selected_weight.shape),
            "local_agreement": selected_weight.new_zeros(selected_weight.shape),
            "axis_consistency": selected_axis_consistency,
            "evidence_conf": selected_weight.new_zeros(selected_weight.shape),
            "view_conf": selected_weight.new_zeros(selected_weight.shape),
        }
    selected_direction_score = (shell_prob[..., None, None] * direction_score).sum(dim=1)
    flat_pair = selected_direction_score.view(n_roots, -1)
    best_score, best_pair = torch.max(flat_pair, dim=1)
    second_score = torch.topk(flat_pair, k=min(2, flat_pair.shape[1]), dim=1).values[:, -1]
    rem = best_pair
    best_lambda_idx = torch.div(rem, 2, rounding_mode="floor")
    best_sign_idx = rem - best_lambda_idx * 2
    best_lambda = lambda_values[best_lambda_idx]
    best_sign = torch.where(best_sign_idx == 0, torch.ones_like(best_lambda), -torch.ones_like(best_lambda))
    directed_flow3d = F.normalize(best_lambda[:, None] * root_normals + best_sign[:, None] * selected_axis, dim=-1, eps=1.0e-8)
    direction_margin = torch.where(
        selected_weight > 0.0,
        (best_score - second_score) / selected_weight.clamp_min(EPS),
        torch.zeros_like(best_score),
    )

    cleaned = clean_directed_flow_on_graph(
        root_points,
        root_normals,
        selected_axis,
        selected_direction_score,
        selected_weight,
        observed,
        lambda_values,
        knn_k=int(args.clean_knn_k),
        knn_k_per_root=clean_knn_k_per_root,
        surface_graph=surface_graph,
        sign_iters=int(args.clean_sign_iters),
        lambda_iters=int(args.clean_lambda_iters),
        vector_iters=int(args.clean_vector_iters),
        anchor_margin=float(args.clean_anchor_margin),
        anchor_weight=float(args.clean_anchor_weight),
        smooth_strength=float(args.clean_smooth_strength),
        vector_blend=float(args.clean_vector_blend),
    )
    discrete_cleaned_flow3d = cleaned["flow"]
    continuous_report: dict[str, float | int | str] = {"enabled": int(args.direction_field_mode in {"continuous", "continuous-ratio"})}
    if args.direction_field_mode in {"continuous", "continuous-ratio"}:
        continuous_evidence = collect_selected_direction_evidence(
            args=args,
            views=views,
            selected_shell_points=selected_shell_points,
            root_normals=root_normals,
            viewmats=viewmats,
            ks=ks,
            mesh=mesh,
            width=width,
            height=height,
            signed_silhouette_offsets=signed_silhouette_offsets,
            observed=observed,
            device=device,
        )
        if args.direction_field_mode == "continuous-ratio":
            continuous = optimize_continuous_ratio_field(
                initial_direction=discrete_cleaned_flow3d,
                axial_flow=selected_axis,
                points=root_points,
                normals=root_normals,
                selected_shell_points=selected_shell_points,
                evidence=continuous_evidence,
                viewmats=viewmats,
                ks=ks,
                observed=observed,
                direction_weight=selected_weight,
                knn_k=int(args.clean_knn_k),
                knn_k_per_root=clean_knn_k_per_root,
                surface_graph=surface_graph,
                iters=int(args.continuous_direction_iters),
                lr=float(args.continuous_direction_lr),
                smooth_weight=float(args.continuous_direction_smooth_weight),
                anchor_weight=float(args.continuous_direction_anchor_weight),
                silhouette_sign_bias=float(args.silhouette_sign_bias),
                robust_max_quantile=float(args.continuous_ratio_robust_max_quantile),
                robust_max_scale=float(args.continuous_ratio_robust_max_scale),
            )
        else:
            continuous = optimize_continuous_direction_field(
                initial_direction=discrete_cleaned_flow3d,
                points=root_points,
                normals=root_normals,
                selected_shell_points=selected_shell_points,
                evidence=continuous_evidence,
                viewmats=viewmats,
                ks=ks,
                observed=observed,
                direction_weight=selected_weight,
                knn_k=int(args.clean_knn_k),
                knn_k_per_root=clean_knn_k_per_root,
                surface_graph=surface_graph,
                iters=int(args.continuous_direction_iters),
                lr=float(args.continuous_direction_lr),
                smooth_weight=float(args.continuous_direction_smooth_weight),
                anchor_weight=float(args.continuous_direction_anchor_weight),
                silhouette_sign_bias=float(args.silhouette_sign_bias),
            )
        cleaned_directed_flow3d = continuous["flow"]
        cleaned["lambda"] = continuous["lambda"]
        cleaned["anchor_conf"] = torch.maximum(cleaned["anchor_conf"], continuous["anchor_conf"])
        cleaned["anchor"] = cleaned["anchor"] | continuous["anchor"]
        continuous_report.update(
            {
                "evidence_terms": int(len(continuous_evidence)),
                "mode": str(args.direction_field_mode),
                "iters": int(args.continuous_direction_iters),
                "lr": float(args.continuous_direction_lr),
                "smooth_weight": float(args.continuous_direction_smooth_weight),
                "anchor_weight": float(args.continuous_direction_anchor_weight),
                "robust_max_quantile": float(args.continuous_ratio_robust_max_quantile),
                "robust_max_scale": float(args.continuous_ratio_robust_max_scale),
                "robust_ratio_hi": float(continuous.get("robust_ratio_hi", torch.tensor(0.0, device=device)).detach().cpu()),
            }
        )
    else:
        cleaned_directed_flow3d = discrete_cleaned_flow3d

    pre_consensus_directed_flow3d = cleaned_directed_flow3d
    pre_consensus_anchor_confidence = cleaned["anchor_conf"]
    ratio_refinement_report: dict[str, object] = {"enabled": 0}
    global_orientation_report: dict[str, object] = {"enabled": 0}
    fixed_sign_directed_ratio_report: dict[str, object] = {"enabled": 0}
    confidence_guided_direction_report: dict[str, object] = {
        "enabled": 0,
        "mode": str(args.directed_flow_propagation_mode),
    }
    consensus_report: dict[str, object]
    if args.axis_field_mode == "trusted-view-cluster":
        assert trusted_knn is not None
        assert trusted_edge_weight is not None
        assert selected_direct_axes is not None
        assert selected_direct_weights is not None
        view_ids = torch.tensor(views, device=device, dtype=torch.long)
        provisional_result = refine_fixed_axis_multiview_ratio(
            initial_direction=pre_consensus_directed_flow3d,
            tangent_axis=selected_axis,
            normals=root_normals,
            points=selected_shell_points,
            per_view_axes=selected_direct_axes,
            per_view_weights=selected_direct_weights,
            viewmats=viewmats[view_ids],
            intrinsics=ks[view_ids],
            knn=trusted_knn,
            edge_weight=trusted_edge_weight,
            observed=observed,
        )
        provisional_ratio = provisional_result["ratio"]
        provisional_sign = provisional_result["tangent_sign"]
        cleaned_directed_flow3d = provisional_result["direction"]
        cleaned["lambda"] = provisional_ratio
        ratio_refinement_report = dict(provisional_result["report"])
        ratio_refinement_report["enabled"] = 1

        def ratio_npz_array(key: str, *, dtype: np.dtype) -> np.ndarray:
            value = provisional_result[key]
            return value.detach().cpu().numpy().astype(dtype)

        axis_view_cluster_npz.update(
            {
                "axis_view_cluster_ratio_baseline": ratio_npz_array("baseline_ratio", dtype=np.dtype(np.float32)),
                "axis_view_cluster_ratio_ls": ratio_npz_array("ls_ratio", dtype=np.dtype(np.float32)),
                "axis_view_cluster_ratio_final": ratio_npz_array("ratio", dtype=np.dtype(np.float32)),
                "axis_view_cluster_ratio_denominator": ratio_npz_array("denominator", dtype=np.dtype(np.float32)),
                "axis_view_cluster_ratio_fallback": ratio_npz_array("fallback", dtype=np.dtype(np.bool_)),
                "axis_view_cluster_ratio_accept": ratio_npz_array("accept_mask", dtype=np.dtype(np.bool_)),
                "axis_view_cluster_ratio_residual_before": ratio_npz_array("residual_before", dtype=np.dtype(np.float32)),
                "axis_view_cluster_ratio_residual_after": ratio_npz_array("residual_after", dtype=np.dtype(np.float32)),
                "axis_view_cluster_ratio_local_jump_before": ratio_npz_array("baseline_local_jump_deg", dtype=np.dtype(np.float32)),
                "axis_view_cluster_ratio_local_jump_ls": ratio_npz_array("ls_local_jump_deg", dtype=np.dtype(np.float32)),
                "axis_view_cluster_ratio_tangent_sign": ratio_npz_array("tangent_sign", dtype=np.dtype(np.float32)),
                "axis_view_cluster_provisional_ratio": ratio_npz_array("ratio", dtype=np.dtype(np.float32)),
                "axis_view_cluster_provisional_sign": ratio_npz_array("tangent_sign", dtype=np.dtype(np.float32)),
            }
        )

        global_result = refine_global_tangent_sign_field(
            points=root_points,
            projection_points=selected_shell_points,
            face_ids=root_face_ids,
            barycentric=root_barycentric,
            normals=root_normals,
            tangent_axis=selected_axis,
            normal_tangent_ratio=provisional_ratio,
            initial_sign=provisional_sign,
            per_view_axes=selected_direct_axes,
            per_view_weights=selected_direct_weights,
            viewmats=viewmats[view_ids],
            intrinsics=ks[view_ids],
            knn=trusted_knn,
            edge_weight=trusted_edge_weight,
            observed=observed,
        )
        global_sign = global_result["candidate_sign"]
        global_baseline_ratio = global_result["normal_tangent_ratio"]
        global_unary = global_result["unary"]
        global_edge = global_result["edge"]
        if not isinstance(global_unary, dict) or not isinstance(global_edge, dict):
            raise TypeError("global sign diagnostics must contain unary and edge mappings")

        def global_npz_array(value: object, *, dtype: np.dtype) -> np.ndarray:
            if not isinstance(value, torch.Tensor):
                raise TypeError("global sign diagnostics must be tensors")
            return value.detach().cpu().numpy().astype(dtype)

        axis_view_cluster_npz.update(
            {
                "axis_view_cluster_global_final_sign": global_npz_array(global_sign, dtype=np.dtype(np.int8)),
                "axis_view_cluster_global_flip": global_npz_array(global_result["flip_mask"], dtype=np.dtype(np.bool_)),
                "axis_view_cluster_global_canonical_rank": global_npz_array(global_result["canonical_rank"], dtype=np.dtype(np.int64)),
                "axis_view_cluster_global_supernode_id": global_npz_array(global_result["supernode_id"], dtype=np.dtype(np.int64)),
                "axis_view_cluster_global_unary_normalized_margin": global_npz_array(global_unary["normalized_margin"], dtype=np.dtype(np.float32)),
                "axis_view_cluster_global_unary_vote_coherence": global_npz_array(global_unary["vote_coherence"], dtype=np.dtype(np.float32)),
                "axis_view_cluster_global_pre_postratio_direction": global_npz_array(global_result["candidate_direction"], dtype=np.dtype(np.float32)),
                "axis_view_cluster_global_edge_u": global_npz_array(global_edge["u"], dtype=np.dtype(np.int64)),
                "axis_view_cluster_global_edge_v": global_npz_array(global_edge["v"], dtype=np.dtype(np.int64)),
                "axis_view_cluster_global_edge_baseline_dot": global_npz_array(global_edge["baseline_dot"], dtype=np.dtype(np.float32)),
                "axis_view_cluster_global_edge_final_dot": global_npz_array(global_edge["candidate_dot"], dtype=np.dtype(np.float32)),
                "axis_view_cluster_global_edge_equality_mask": global_npz_array(global_edge["equality_mask"], dtype=np.dtype(np.bool_)),
                "axis_view_cluster_global_edge_new_severe_mask": global_npz_array(global_edge["new_severe_mask"], dtype=np.dtype(np.bool_)),
            }
        )
        global_orientation_report = dict(global_result["report"])
        global_orientation_report["enabled"] = 1
        global_final_report = global_orientation_report.get("final", {})
        if not isinstance(global_final_report, dict):
            raise TypeError("global sign report final section must be serializable")
        global_zero_new_severe = bool(
            global_final_report.get("mathematical_zero_new_severe_guard_verified", False)
        )
        if not global_zero_new_severe:
            raise RuntimeError(
                "global sign orientation failed the zero-new-severe-edge invariant"
            )
        global_orientation_report["zero_new_severe_verification"] = {
            "mathematical_guard_verified": global_zero_new_severe,
            "passed": global_zero_new_severe,
        }

        postratio_result = refine_fixed_sign_directed_multiview_ratio(
            tangent_axis=selected_axis,
            tangent_sign=global_sign,
            baseline_ratio=global_baseline_ratio,
            normals=root_normals,
            points=selected_shell_points,
            per_view_axes=selected_direct_axes,
            per_view_weights=selected_direct_weights,
            viewmats=viewmats[view_ids],
            intrinsics=ks[view_ids],
            knn=trusted_knn,
            edge_weight=trusted_edge_weight,
            observed=observed,
            canonical_rank=global_result["canonical_rank"],
        )
        cleaned_directed_flow3d = postratio_result["direction"]
        cleaned["flow"] = cleaned_directed_flow3d
        cleaned["lambda"] = postratio_result["ratio"]
        cleaned["sign"] = global_sign
        cleaned["flipped"] = global_result["flip_mask"]
        fixed_sign_directed_ratio_report = dict(postratio_result["report"])
        fixed_sign_directed_ratio_report["enabled"] = 1
        postratio_new_severe = postratio_result["new_severe_edge_mask"]
        if not isinstance(postratio_new_severe, torch.Tensor):
            raise TypeError("postratio new severe diagnostic must be a tensor")
        postratio_zero_new_severe = not bool(postratio_new_severe.any())
        if not postratio_zero_new_severe:
            raise RuntimeError(
                "fixed-sign directed ratio refit introduced a new severe edge"
            )
        fixed_sign_directed_ratio_report["zero_new_severe_verification"] = {
            "global_sign_guard_verified": global_zero_new_severe,
            "postratio_new_severe_edge_count": int(postratio_new_severe.sum().detach().cpu()),
            "passed": bool(global_zero_new_severe and postratio_zero_new_severe),
        }

        def postratio_npz_array(key: str, *, dtype: np.dtype) -> np.ndarray:
            value = postratio_result[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError("postratio diagnostics must be tensors")
            return value.detach().cpu().numpy().astype(dtype)

        eligibility_rejection_masks = postratio_result["eligibility_rejection_masks"]
        guard_rejection_masks = postratio_result["guard_rejection_masks"]
        if not isinstance(eligibility_rejection_masks, dict) or not isinstance(guard_rejection_masks, dict):
            raise TypeError("postratio rejection diagnostics must be mappings")
        axis_view_cluster_npz.update(
            {
                "axis_view_cluster_postratio_ls_ratio": postratio_npz_array("raw_ls_ratio", dtype=np.dtype(np.float32)),
                "axis_view_cluster_postratio_final_ratio": postratio_npz_array("final_ratio", dtype=np.dtype(np.float32)),
                "axis_view_cluster_postratio_eligible_mask": postratio_npz_array("eligible_mask", dtype=np.dtype(np.bool_)),
                "axis_view_cluster_postratio_accept_mask": postratio_npz_array("accepted_mask", dtype=np.dtype(np.bool_)),
                "axis_view_cluster_postratio_rejected_mask": postratio_npz_array("rejected_mask", dtype=np.dtype(np.bool_)),
                "axis_view_cluster_postratio_rejection_denominator": global_npz_array(eligibility_rejection_masks["denominator_not_finite_or_nonpositive"], dtype=np.dtype(np.bool_)),
                "axis_view_cluster_postratio_rejection_residual": global_npz_array(eligibility_rejection_masks["no_strict_direct_residual_improvement"], dtype=np.dtype(np.bool_)),
                "axis_view_cluster_postratio_rejection_nonsevere_edge": global_npz_array(guard_rejection_masks["nonsevere_edge_would_become_severe"], dtype=np.dtype(np.bool_)),
                "axis_view_cluster_postratio_rejection_directed_angle": global_npz_array(guard_rejection_masks["directed_incident_angle_increase"], dtype=np.dtype(np.bool_)),
                "axis_view_cluster_postratio_rejection_nonfinite": global_npz_array(guard_rejection_masks["nonfinite_or_negative_ratio_or_direction"], dtype=np.dtype(np.bool_)),
                "axis_view_cluster_postratio_denominator": postratio_npz_array("denominator", dtype=np.dtype(np.float32)),
                "axis_view_cluster_postratio_residual_before": postratio_npz_array("residual_before", dtype=np.dtype(np.float32)),
                "axis_view_cluster_postratio_residual_after": postratio_npz_array("residual_after", dtype=np.dtype(np.float32)),
                "axis_view_cluster_postratio_normalized_residual_improvement": postratio_npz_array("normalized_residual_improvement", dtype=np.dtype(np.float32)),
                "axis_view_cluster_postratio_edge_u": postratio_npz_array("edge_u", dtype=np.dtype(np.int64)),
                "axis_view_cluster_postratio_edge_v": postratio_npz_array("edge_v", dtype=np.dtype(np.int64)),
                "axis_view_cluster_postratio_baseline_edge_dot": postratio_npz_array("baseline_edge_dots", dtype=np.dtype(np.float32)),
                "axis_view_cluster_postratio_final_edge_dot": postratio_npz_array("final_edge_dots", dtype=np.dtype(np.float32)),
            }
        )
        if args.directed_flow_propagation_mode == "confidence-guided":
            confidence_result = refine_confidence_guided_directed_flow(
                direction=postratio_result["direction"],
                normals=root_normals,
                observed=observed,
                edge_u=postratio_result["edge_u"],
                edge_v=postratio_result["edge_v"],
                field_confidence=trusted_result["final_confidence"],
                unary_normalized_margin=global_unary["normalized_margin"],
                unary_vote_coherence=global_unary["vote_coherence"],
                canonical_rank=global_result["canonical_rank"],
            )
            confidence_new_severe = confidence_result["new_severe_edge_mask"]
            if not isinstance(confidence_new_severe, torch.Tensor):
                raise TypeError(
                    "confidence-guided new-severe diagnostic must be a tensor"
                )
            if bool(confidence_new_severe.any()):
                raise RuntimeError(
                    "confidence-guided direction propagation introduced a new "
                    "severe edge"
                )
            cleaned_directed_flow3d = confidence_result["direction"]
            if not isinstance(cleaned_directed_flow3d, torch.Tensor):
                raise TypeError("confidence-guided direction must be a tensor")
            cleaned["flow"] = cleaned_directed_flow3d
            final_normal_component = (
                cleaned_directed_flow3d * root_normals
            ).sum(dim=-1).clamp_min(0.0)
            final_tangent = (
                cleaned_directed_flow3d
                - final_normal_component[:, None] * root_normals
            )
            final_tangent_length = torch.linalg.vector_norm(
                final_tangent, dim=-1
            ).clamp_min(EPS)
            cleaned["lambda"] = final_normal_component / final_tangent_length
            cleaned["sign"] = torch.where(
                (final_tangent * selected_axis).sum(dim=-1) >= 0.0,
                torch.ones_like(final_normal_component),
                -torch.ones_like(final_normal_component),
            )
            confidence_guided_direction_report = dict(confidence_result["report"])
            confidence_guided_direction_report["enabled"] = 1
            confidence_guided_direction_report["mode"] = str(
                args.directed_flow_propagation_mode
            )
            confidence_guided_direction_report["zero_new_severe_verification"] = {
                "new_severe_edge_count": int(
                    confidence_new_severe.sum().detach().cpu()
                ),
                "passed": not bool(confidence_new_severe.any()),
            }

            def confidence_npz_array(key: str, *, dtype: np.dtype) -> np.ndarray:
                value = confidence_result[key]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        "confidence-guided diagnostics must be tensors"
                    )
                return value.detach().cpu().numpy().astype(dtype)

            axis_view_cluster_npz.update(
                {
                    "axis_view_cluster_confidence_flow_input_direction": postratio_npz_array("direction", dtype=np.dtype(np.float32)),
                    "axis_view_cluster_confidence_flow_watershed_direction": confidence_npz_array("watershed_direction", dtype=np.dtype(np.float32)),
                    "axis_view_cluster_confidence_flow_joint_confidence": confidence_npz_array("joint_confidence", dtype=np.dtype(np.float32)),
                    "axis_view_cluster_confidence_flow_watershed_owner": confidence_npz_array("watershed_owner", dtype=np.dtype(np.int64)),
                    "axis_view_cluster_confidence_flow_watershed_parent": confidence_npz_array("watershed_parent", dtype=np.dtype(np.int64)),
                    "axis_view_cluster_confidence_flow_propagated_confidence": confidence_npz_array("watershed_propagated_confidence", dtype=np.dtype(np.float32)),
                    "axis_view_cluster_confidence_flow_watershed_changed": confidence_npz_array("watershed_changed_mask", dtype=np.dtype(np.bool_)),
                    "axis_view_cluster_confidence_flow_local_changed": confidence_npz_array("local_changed_mask", dtype=np.dtype(np.bool_)),
                    "axis_view_cluster_confidence_flow_changed": confidence_npz_array("changed_mask", dtype=np.dtype(np.bool_)),
                    "axis_view_cluster_confidence_flow_protected_owner": confidence_npz_array("protected_owner_mask", dtype=np.dtype(np.bool_)),
                    "axis_view_cluster_confidence_flow_local_update_count": confidence_npz_array("local_update_count", dtype=np.dtype(np.int64)),
                    "axis_view_cluster_confidence_flow_edge_u": confidence_npz_array("edge_u", dtype=np.dtype(np.int64)),
                    "axis_view_cluster_confidence_flow_edge_v": confidence_npz_array("edge_v", dtype=np.dtype(np.int64)),
                    "axis_view_cluster_confidence_flow_initial_edge_dot": confidence_npz_array("initial_edge_dots", dtype=np.dtype(np.float32)),
                    "axis_view_cluster_confidence_flow_watershed_edge_dot": confidence_npz_array("watershed_edge_dots", dtype=np.dtype(np.float32)),
                    "axis_view_cluster_confidence_flow_final_edge_dot": confidence_npz_array("final_edge_dots", dtype=np.dtype(np.float32)),
                    "axis_view_cluster_confidence_flow_new_severe_edge": confidence_npz_array("new_severe_edge_mask", dtype=np.dtype(np.bool_)),
                }
            )
        consensus_report = {
            "enabled": 0,
            "mode": "superseded-by-global-sign-and-fixed-sign-directed-multiview-ratio",
            "requested_iters": int(args.direction_consensus_iters),
        }
    else:
        consensus_report = {"enabled": int(int(args.direction_consensus_iters) > 0)}
    if args.axis_field_mode != "trusted-view-cluster" and int(args.direction_consensus_iters) > 0:
        consensus = refine_direction_with_anchor_consensus(
            points=root_points,
            normals=root_normals,
            direction=cleaned_directed_flow3d,
            anchor_confidence=cleaned["anchor_conf"],
            observed=observed,
            knn_k=int(args.clean_knn_k),
            knn_k_per_root=clean_knn_k_per_root,
            surface_graph=surface_graph,
            iters=int(args.direction_consensus_iters),
            blend=float(args.direction_consensus_blend),
            anchor_threshold=float(args.direction_consensus_anchor_threshold),
        )
        cleaned_directed_flow3d = consensus["flow"]
        cleaned["anchor"] = consensus["anchor"]
        cleaned["anchor_conf"] = consensus["anchor_conf"]
        consensus_report.update(
            {
                "iters": int(args.direction_consensus_iters),
                "blend": float(args.direction_consensus_blend),
                "anchor_threshold": float(args.direction_consensus_anchor_threshold),
                "anchor_roots": int((cleaned["anchor"] & observed).sum().detach().cpu()),
            }
        )

    diag_idx = int(args.diag_view)
    diag_gt = np.asarray(Image.open(args.data_root / "images" / f"img_{diag_idx:04d}.png").convert("RGB"))
    diag_mask_path = args.data_root / "silhouette" / f"img_{diag_idx:04d}.png"
    mesh_depth = render_mesh_depth(mesh, viewmats[diag_idx], ks[diag_idx], width, height, device=device)
    diag_vis = sample_shell_visibility(
        selected_shell_points,
        root_normals,
        viewmats[diag_idx],
        ks[diag_idx],
        mesh_depth.depth,
        depth_abs_tolerance=float(args.depth_abs_tolerance),
        depth_rel_tolerance=float(args.depth_rel_tolerance),
        local_depth_kernel=int(args.local_depth_kernel),
        front_normal_z=float(args.front_normal_z),
    )
    screen_cleaned = project_directions(selected_shell_points, cleaned_directed_flow3d, viewmats[diag_idx], ks[diag_idx])
    diag_valid = diag_vis["visible"] & observed
    normalized_weight = (selected_weight / torch.quantile(selected_weight[observed], 0.95).clamp_min(EPS)).clamp(0.0, 1.0) if bool(observed.any()) else selected_weight
    cleaned_path = args.output_dir / f"view{diag_idx:02d}_shell_cleaned_3d_arrows_overlay.png"
    draw_root_flow_arrow_overlay(
        cleaned_path,
        diag_gt,
        diag_vis["xy"],
        screen_cleaned,
        diag_valid,
        normalized_weight,
        color=(245, 40, 120),
    )
    normal_dot = (cleaned_directed_flow3d * root_normals).sum(dim=-1).clamp(0.0, 1.0)

    np.savez_compressed(
        args.output_dir / "guide_flow3d_shell_targets_exclude_004_024_025.npz",
        root_points=root_points_np,
        root_normals=root_normals_np,
        face_ids=roots.face_ids.astype(np.int64),
        barycentric=roots.barycentric.astype(np.float32),
        shell_h=selected_h.detach().cpu().numpy().astype(np.float32),
        raw_shell_h=raw_selected_h.detach().cpu().numpy().astype(np.float32),
        pre_height_smooth_shell_h=(shell_prob * shell_h).sum(dim=1).detach().cpu().numpy().astype(np.float32),
        local_spacing=local_spacing.detach().cpu().numpy().astype(np.float32),
        flow3d=selected_axis.detach().cpu().numpy().astype(np.float32),
        raw_flow3d=raw_selected_axis.detach().cpu().numpy().astype(np.float32),
        axis_anchor_confidence=axis_field["anchor_conf"].detach().cpu().numpy().astype(np.float32),
        axis_local_agreement=axis_field["local_agreement"].detach().cpu().numpy().astype(np.float32),
        axis_consistency=axis_field["axis_consistency"].detach().cpu().numpy().astype(np.float32),
        axis_evidence_confidence=axis_field["evidence_conf"].detach().cpu().numpy().astype(np.float32),
        axis_view_confidence=axis_field["view_conf"].detach().cpu().numpy().astype(np.float32),
        directed_flow3d=directed_flow3d.detach().cpu().numpy().astype(np.float32),
        cleaned_directed_flow3d=cleaned_directed_flow3d.detach().cpu().numpy().astype(np.float32),
        pre_consensus_directed_flow3d=pre_consensus_directed_flow3d.detach().cpu().numpy().astype(np.float32),
        discrete_cleaned_directed_flow3d=discrete_cleaned_flow3d.detach().cpu().numpy().astype(np.float32),
        sign_cleaned_directed_flow3d=cleaned["pre_vector_smooth_flow"].detach().cpu().numpy().astype(np.float32),
        direction_lambda=best_lambda.detach().cpu().numpy().astype(np.float32),
        direction_sign=best_sign.detach().cpu().numpy().astype(np.float32),
        direction_margin=direction_margin.detach().cpu().numpy().astype(np.float32),
        cleaned_direction_lambda=cleaned["lambda"].detach().cpu().numpy().astype(np.float32),
        cleaned_direction_sign=cleaned["sign"].detach().cpu().numpy().astype(np.float32),
        direction_anchor=cleaned["anchor"].detach().cpu().numpy().astype(np.bool_),
        direction_anchor_confidence=cleaned["anchor_conf"].detach().cpu().numpy().astype(np.float32),
        pre_consensus_direction_anchor_confidence=pre_consensus_anchor_confidence.detach().cpu().numpy().astype(np.float32),
        direction_flipped=cleaned["flipped"].detach().cpu().numpy().astype(np.bool_),
        raw_shell_idx=raw_best_shell_idx.detach().cpu().numpy().astype(np.int64),
        shell_idx=best_shell_idx.detach().cpu().numpy().astype(np.int64),
        shell_probability=shell_prob.detach().cpu().numpy().astype(np.float32),
        shell_confidence=shell_confidence.detach().cpu().numpy().astype(np.float32),
        weight=selected_weight.detach().cpu().numpy().astype(np.float32),
        observed=observed.detach().cpu().numpy().astype(np.bool_),
        view_count=selected_view_count.detach().cpu().numpy().astype(np.float32),
        **axis_view_cluster_npz,
        **root_sampling_payload,
    )
    summary = {
        "views_used": views,
        "views_excluded": sorted(exclude),
        "root_count": int(n_roots),
        "root_sampling": root_sampling_report,
        "root_neighborhood": surface_graph_report,
        "shell_count": int(n_shells),
        "shell_extent": float(args.shell_extent),
        "axis_field": axis_field_report,
        "direction_field_mode": str(args.direction_field_mode),
        "continuous_direction": continuous_report,
        "fixed_axis_multiview_ratio": ratio_refinement_report,
        "global_sign_orientation": global_orientation_report,
        "fixed_sign_directed_multiview_ratio": fixed_sign_directed_ratio_report,
        "confidence_guided_directed_flow": confidence_guided_direction_report,
        "direction_consensus": consensus_report,
        "shell_height_refine": shell_height_refine_report,
        "observed_roots": int(observed.sum().detach().cpu()),
        "observed_fraction": float(observed.float().mean().detach().cpu()),
        "selected_h_mean": float(selected_h[observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
        "selected_h_median": float(torch.median(selected_h[observed]).detach().cpu()) if bool(observed.any()) else 0.0,
        "selected_h_p90": float(torch.quantile(selected_h[observed], 0.90).detach().cpu()) if bool(observed.any()) else 0.0,
        "raw_selected_h_mean": float(raw_selected_h[observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
        "raw_selected_h_median": float(torch.median(raw_selected_h[observed]).detach().cpu()) if bool(observed.any()) else 0.0,
        "raw_selected_h_p90": float(torch.quantile(raw_selected_h[observed], 0.90).detach().cpu()) if bool(observed.any()) else 0.0,
        "selected_shell_hist": torch.bincount(best_shell_idx.detach().cpu(), minlength=n_shells).tolist(),
        "raw_selected_shell_hist": torch.bincount(raw_best_shell_idx.detach().cpu(), minlength=n_shells).tolist(),
        "shell_choice_changed": int((raw_best_shell_idx != best_shell_idx).sum().detach().cpu()),
        "shell_choice_changed_fraction": float((raw_best_shell_idx != best_shell_idx).float().mean().detach().cpu()),
        "normal_dot_mean": float(normal_dot[observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
        "normal_dot_median": float(torch.median(normal_dot[observed]).detach().cpu()) if bool(observed.any()) else 0.0,
        "normal_dot_p90": float(torch.quantile(normal_dot[observed], 0.90).detach().cpu()) if bool(observed.any()) else 0.0,
        "cleaned_lambda_mean": float(cleaned["lambda"][observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
        "cleaned_lambda_median": float(torch.median(cleaned["lambda"][observed]).detach().cpu()) if bool(observed.any()) else 0.0,
        "cleaned_lambda_p90": float(torch.quantile(cleaned["lambda"][observed], 0.90).detach().cpu()) if bool(observed.any()) else 0.0,
        "cleaned_lambda_max": float(cleaned["lambda"][observed].max().detach().cpu()) if bool(observed.any()) else 0.0,
        "clean_anchor_roots": int((cleaned["anchor"] & observed).sum().detach().cpu()),
        "clean_flipped_roots": int((cleaned["flipped"] & observed).sum().detach().cpu()),
        "per_view": per_view,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

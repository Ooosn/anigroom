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
from anigroom.flow.direction_geometry import parallel_transport_vectors  # noqa: E402
from anigroom.flow.surface_graph import SurfaceRootGraph  # noqa: E402
from anigroom.mesh_roots import TriangleMesh, initialize_surface_roots_fps, read_obj_mesh  # noqa: E402
from anigroom.projection import render_mesh_depth, sample_mesh_visible_points  # noqa: E402
from tools.train_white_tiger_stage1 import (  # noqa: E402
    bilinear_sample,
    load_camera_tensors,
    load_mask,
    project_directions,
    view_angle_weight,
)


EPS = 1.0e-8


def bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def bbox_transform(src: list[int], dst: list[int]) -> np.ndarray:
    sx0, sy0, sx1, sy1 = src
    dx0, dy0, dx1, dy1 = dst
    sw, sh = max(sx1 - sx0, 1), max(sy1 - sy0, 1)
    dw, dh = max(dx1 - dx0, 1), max(dy1 - dy0, 1)
    sx, sy = dw / sw, dh / sh
    scx, scy = (sx0 + sx1) * 0.5, (sy0 + sy1) * 0.5
    dcx, dcy = (dx0 + dx1) * 0.5, (dy0 + dy1) * 0.5
    return np.asarray([[sx, 0.0, dcx - sx * scx], [0.0, sy, dcy - sy * scy]], dtype=np.float32)


def load_flow_strength(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    inv = 255 - image.astype(np.int16)
    strength = np.maximum.reduce([inv[..., 0], inv[..., 1], inv[..., 2]]).astype(np.uint8)
    return strength


def aligned_flow_strength(
    flow_path: Path,
    mask_path: Path,
    width: int,
    height: int,
    *,
    threshold: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strength = load_flow_strength(flow_path)
    ink = strength > int(threshold)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    with Image.open(mask_path) as mask_img:
        mask = np.asarray(mask_img.convert("L")) > 127
    if mask.mean() > 0.7:
        mask = ~mask
    matrix = bbox_transform(bbox(ink), bbox(mask))
    warped = cv2.warpAffine(
        strength,
        matrix,
        (int(width), int(height)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_soft = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (0, 0), 1.2).astype(np.float32) / 255.0
    clipped = (warped.astype(np.float32) * mask_soft).clip(0, 255).astype(np.uint8)
    return clipped, mask.astype(bool), matrix


def orientation_from_line_strength(strength: np.ndarray, *, sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Return local line tangent axis and confidence from a cleaned line drawing.

    A single large structure-tensor scale washes dense short fur strokes into a
    coarse dominant direction.  We therefore evaluate a small scale pyramid and
    let each pixel keep the scale with the strongest line/coherence evidence.
    This is still a local image method: it has no animal-part rules.
    """

    src = strength.astype(np.float32) / 255.0
    src = cv2.GaussianBlur(src, (0, 0), 0.8)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    line = src
    if np.isfinite(line).any() and float(line.max()) > 1.0e-6:
        line = line / max(float(np.quantile(line[line > 0], 0.95)) if np.any(line > 0) else float(line.max()), 1.0e-6)

    scales = sorted({max(0.8, float(sigma) / 3.0), max(1.2, float(sigma) / 2.0), max(1.6, float(sigma))})
    best_conf: np.ndarray | None = None
    best_ori: np.ndarray | None = None
    for scale in scales:
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), scale)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), scale)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), scale)
        # Dominant eigenvector of J is image-gradient normal; line tangent is +90 degrees.
        theta_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        theta_line = theta_grad + 0.5 * math.pi
        ori = np.stack([np.cos(theta_line), np.sin(theta_line)], axis=-1).astype(np.float32)
        coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (jxx + jyy + 1.0e-8)
        conf = np.clip(line * coherence, 0.0, 1.0).astype(np.float32)
        if best_conf is None:
            best_conf = conf
            best_ori = ori
        else:
            take = conf > best_conf
            best_conf = np.where(take, conf, best_conf)
            best_ori = np.where(take[..., None], ori, best_ori)

    assert best_conf is not None and best_ori is not None
    norm = np.linalg.norm(best_ori, axis=-1, keepdims=True)
    best_ori = best_ori / np.maximum(norm, 1.0e-8)
    return best_ori.astype(np.float32), best_conf[..., None].astype(np.float32)


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals / np.maximum(norms, EPS)).astype(np.float32)


def save_overlay(path: Path, base: np.ndarray, strength: np.ndarray) -> None:
    alpha = np.clip(strength.astype(np.float32) / 110.0, 0.0, 0.82)[..., None]
    blue = np.asarray([35, 105, 220], dtype=np.float32)
    overlay = np.clip(base.astype(np.float32) * (1.0 - alpha) + blue * alpha, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)


def draw_root_flow_overlay(
    path: Path,
    base: np.ndarray,
    xy: torch.Tensor,
    screen_dirs: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_points: int = 7000,
    color: tuple[int, int, int] = (0, 210, 70),
) -> None:
    canvas = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    valid_ids = torch.nonzero(valid & torch.isfinite(screen_dirs).all(dim=-1), as_tuple=False).reshape(-1)
    if valid_ids.numel() > max_points:
        score = weights[valid_ids]
        _, order = torch.topk(score, k=max_points, largest=True)
        valid_ids = valid_ids[order]
    xy_np = xy[valid_ids].detach().cpu().numpy()
    dirs_np = F.normalize(screen_dirs[valid_ids], dim=-1, eps=1.0e-8).detach().cpu().numpy()
    weight_np = weights[valid_ids].detach().cpu().numpy()
    for (x, y), (dx, dy), w in zip(xy_np, dirs_np, weight_np):
        length = 5.0 + 9.0 * float(np.clip(w, 0.0, 1.0))
        x0, y0 = float(x - dx * length), float(y - dy * length)
        x1, y1 = float(x + dx * length), float(y + dy * length)
        a = int(50 + 180 * float(np.clip(w, 0.0, 1.0)))
        draw.line((x0, y0, x1, y1), fill=(*color, a), width=1)
    canvas.save(path)


def draw_root_flow_arrow_overlay(
    path: Path,
    base: np.ndarray,
    xy: torch.Tensor,
    screen_dirs: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_points: int = 5000,
    color: tuple[int, int, int] = (255, 120, 0),
) -> None:
    canvas = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    valid_ids = torch.nonzero(valid & torch.isfinite(screen_dirs).all(dim=-1), as_tuple=False).reshape(-1)
    if valid_ids.numel() > max_points:
        score = weights[valid_ids]
        _, order = torch.topk(score, k=max_points, largest=True)
        valid_ids = valid_ids[order]
    xy_np = xy[valid_ids].detach().cpu().numpy()
    dirs_np = F.normalize(screen_dirs[valid_ids], dim=-1, eps=1.0e-8).detach().cpu().numpy()
    weight_np = weights[valid_ids].detach().cpu().numpy()
    for (x, y), (dx, dy), w in zip(xy_np, dirs_np, weight_np):
        length = 7.0 + 10.0 * float(np.clip(w, 0.0, 1.0))
        x0, y0 = float(x), float(y)
        x1, y1 = float(x + dx * length), float(y + dy * length)
        alpha = int(60 + 180 * float(np.clip(w, 0.0, 1.0)))
        draw.line((x0, y0, x1, y1), fill=(*color, alpha), width=1)
        # Small arrow head. Keep it screen-space only; it is for direction diagnostics.
        side = np.asarray([-dy, dx], dtype=np.float32)
        head = 3.0 + 2.0 * float(np.clip(w, 0.0, 1.0))
        p0 = (x1, y1)
        p1 = (float(x1 - dx * head - side[0] * head * 0.55), float(y1 - dy * head - side[1] * head * 0.55))
        p2 = (float(x1 - dx * head + side[0] * head * 0.55), float(y1 - dy * head + side[1] * head * 0.55))
        draw.polygon([p0, p1, p2], fill=(*color, alpha))
    canvas.save(path)


def knn_indices_chunked(points: torch.Tensor, k: int, *, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """KNN on root centers. Chunked to avoid retaining a full NxN distance tensor."""

    n = int(points.shape[0])
    all_idx = []
    all_dist = []
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dist = torch.cdist(points[start:end], points)
        row = torch.arange(end - start, device=points.device)
        col = torch.arange(start, end, device=points.device)
        dist[row, col] = float("inf")
        values, idx = torch.topk(dist, k=min(k, n - 1), dim=1, largest=False)
        all_idx.append(idx)
        all_dist.append(values)
    return torch.cat(all_idx, dim=0), torch.cat(all_dist, dim=0)


def clean_directed_flow_on_graph(
    points: torch.Tensor,
    normals: torch.Tensor,
    axial_flow: torch.Tensor,
    direction_score: torch.Tensor,
    direction_weight: torch.Tensor,
    observed: torch.Tensor,
    lambda_values: torch.Tensor,
    *,
    knn_k: int,
    sign_iters: int,
    lambda_iters: int,
    vector_iters: int,
    anchor_margin: float,
    anchor_weight: float,
    smooth_strength: float,
    vector_blend: float,
    knn_k_per_root: torch.Tensor | None = None,
    surface_graph: SurfaceRootGraph | None = None,
) -> dict[str, torch.Tensor]:
    """Clean ambiguous directed flow with mesh-neighborhood consistency.

    The multiview line drawings mostly constrain a local axis. This pass keeps
    high-margin signs as anchors and lets low-confidence signs follow nearby
    roots with compatible 3D tangent axes.
    """

    if knn_k_per_root is not None:
        knn_k_per_root = knn_k_per_root.to(device=points.device, dtype=torch.long).reshape(-1)
        if int(knn_k_per_root.shape[0]) != int(points.shape[0]):
            raise ValueError(f"knn_k_per_root shape {tuple(knn_k_per_root.shape)} does not match root count {int(points.shape[0])}")
        effective_knn_k = max(int(knn_k), int(knn_k_per_root.max().detach().cpu()))
    else:
        effective_knn_k = int(knn_k)
    if surface_graph is None:
        knn, knn_dist = knn_indices_chunked(points, effective_knn_k)
    else:
        if surface_graph.root_count != int(points.shape[0]):
            raise ValueError("surface graph root count does not match clean-flow roots")
        if surface_graph.neighbor_count < effective_knn_k:
            raise ValueError(
                f"surface graph has K={surface_graph.neighbor_count}, but clean flow needs K={effective_knn_k}"
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
    neighbor_axis = axial_flow[knn]
    if surface_graph is not None:
        source_normals = normals[knn]
        target_normals = normals[:, None, :].expand_as(source_normals)
        neighbor_axis = parallel_transport_vectors(neighbor_axis, source_normals, target_normals)
    axis_dot = (axial_flow[:, None, :] * neighbor_axis).sum(dim=-1)
    edge_weight = dist_weight * normal_weight * axis_dot.abs().clamp_min(0.05)
    edge_weight = edge_weight * active_edges.float()

    sign_score = direction_score.max(dim=1).values
    sign_delta = (sign_score[:, 0] - sign_score[:, 1]) / direction_weight.clamp_min(EPS)
    sign = torch.where(sign_delta >= 0.0, torch.ones_like(sign_delta), -torch.ones_like(sign_delta))
    unary_conf = (
        (sign_delta.abs() / max(float(anchor_margin), EPS)).clamp(0.0, 1.0)
        * (direction_weight / max(float(anchor_weight), EPS)).clamp(0.0, 1.0)
    )
    anchor = torch.zeros_like(observed)
    anchor_conf = unary_conf

    for _ in range(int(sign_iters)):
        neighbor_vote = (edge_weight * axis_dot * sign[knn]).sum(dim=1) / edge_weight.sum(dim=1).clamp_min(EPS)
        neighbor_conf = neighbor_vote.abs().clamp(0.0, 1.0)
        # Confidence is not static: a root is reliable only if its unary
        # evidence agrees with the surrounding root field.
        agreement = (sign * neighbor_vote).clamp(-1.0, 1.0)
        anchor_conf = (0.55 * unary_conf + 0.45 * neighbor_conf).clamp(0.0, 1.0)
        anchor = observed & (unary_conf >= 0.95) & (agreement > 0.12)
        total_vote = sign_delta + float(smooth_strength) * neighbor_vote * (0.25 + 0.75 * neighbor_conf)
        proposal = torch.where(total_vote >= 0.0, torch.ones_like(sign), -torch.ones_like(sign))
        sign = torch.where(anchor, sign, proposal)

    flat = direction_score.view(direction_score.shape[0], -1)
    best_flat = torch.max(flat, dim=1).indices
    best_lambda_idx = torch.div(best_flat, 2, rounding_mode="floor")
    lam = lambda_values[best_lambda_idx]
    lam_conf = (sign_delta.abs() / max(float(anchor_margin), EPS)).clamp(0.0, 1.0)
    for _ in range(int(lambda_iters)):
        neighbor_lam = (edge_weight * lam[knn]).sum(dim=1) / edge_weight.sum(dim=1).clamp_min(EPS)
        lam = torch.where(anchor, lam, lam_conf * lam + (1.0 - lam_conf) * neighbor_lam)
    lam = lam.clamp(float(lambda_values.min()), float(lambda_values.max()))

    cleaned = F.normalize(lam[:, None] * normals + sign[:, None] * axial_flow, dim=-1, eps=EPS)
    vector = cleaned
    for _ in range(int(vector_iters)):
        neighbor_vector = vector[knn]
        if surface_graph is not None:
            source_normals = normals[knn]
            target_normals = normals[:, None, :].expand_as(source_normals)
            neighbor_vector = parallel_transport_vectors(neighbor_vector, source_normals, target_normals)
        neighbor_vector = F.normalize(
            (edge_weight[..., None] * neighbor_vector).sum(dim=1) / edge_weight.sum(dim=1).clamp_min(EPS)[:, None],
            dim=-1,
            eps=EPS,
        )
        blend = float(vector_blend) * (1.0 - 0.75 * anchor_conf)
        vector = F.normalize((1.0 - blend[:, None]) * vector + blend[:, None] * neighbor_vector, dim=-1, eps=EPS)
        normal_dot = (vector * normals).sum(dim=-1, keepdim=True)
        # Keep the groom direction on the outward side of the surface.
        vector = torch.where(normal_dot < 0.0, vector - 2.0 * normal_dot * normals, vector)
        vector = F.normalize(vector, dim=-1, eps=EPS)
    raw_sign = torch.where(sign_delta >= 0.0, torch.ones_like(sign_delta), -torch.ones_like(sign_delta))
    return {
        "flow": vector,
        "pre_vector_smooth_flow": cleaned,
        "sign": sign,
        "lambda": lam,
        "anchor": anchor,
        "sign_delta": sign_delta,
        "anchor_conf": anchor_conf,
        "flipped": observed & (sign != raw_sign),
        "knn": knn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse GPT line-art hair flow into guide-root 3D tangent flow.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk"))
    parser.add_argument("--mesh-path", type=Path, default=Path("D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj"))
    parser.add_argument("--flow-dir", type=Path, default=Path("D:/petsgaussianhair/_downloads/tiger_hair_flow_36"))
    parser.add_argument("--output-dir", type=Path, default=Path("D:/petsgaussianhair/_downloads/tiger_hair_flow_36/multiview_fused_exclude_004_024_025"))
    parser.add_argument("--exclude", default="4,24,25")
    parser.add_argument("--root-count", type=int, default=4096)
    parser.add_argument("--candidate-multiplier", type=float, default=8.0)
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
    parser.add_argument("--clean-knn-k", type=int, default=12)
    parser.add_argument("--clean-sign-iters", type=int, default=12)
    parser.add_argument("--clean-lambda-iters", type=int, default=4)
    parser.add_argument("--clean-vector-iters", type=int, default=6)
    parser.add_argument("--clean-anchor-margin", type=float, default=0.02)
    parser.add_argument("--clean-anchor-weight", type=float, default=0.5)
    parser.add_argument("--clean-smooth-strength", type=float, default=2.0)
    parser.add_argument("--clean-vector-blend", type=float, default=0.35)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for mesh-depth visibility; do not use a CPU fallback for this diagnostic.")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exclude = {int(v) for v in args.exclude.split(",") if v.strip()}
    views = [idx for idx in range(36) if idx not in exclude]
    translation = np.asarray([float(v) for v in args.translation.split(",")], dtype=np.float32)

    raw_mesh = read_obj_mesh(args.mesh_path)
    vertices = (raw_mesh.vertices.astype(np.float32) * float(args.scale) + translation[None]).astype(np.float32)
    mesh = TriangleMesh(vertices=vertices, faces=raw_mesh.faces)
    normals_np = face_normals(mesh.vertices, mesh.faces)
    roots = initialize_surface_roots_fps(
        mesh,
        int(args.root_count),
        candidate_multiplier=float(args.candidate_multiplier),
        seed=13,
        fps_device=device,
    )
    root_points_np = roots.points.astype(np.float32)
    root_normals_np = normals_np[roots.face_ids]

    root_points = torch.from_numpy(root_points_np).to(device=device)
    root_normals = F.normalize(torch.from_numpy(root_normals_np).to(device=device), dim=-1, eps=1.0e-8)
    root_tangents, root_bitangents = make_tangent_frames(root_normals)
    viewmats, ks = load_camera_tensors(args.data_root, device)
    width, height = Image.open(args.data_root / "images" / "img_0000.png").size

    flow3d_sum = torch.zeros((args.root_count, 3), device=device)
    weight_sum = torch.zeros((args.root_count,), device=device)
    view_count = torch.zeros((args.root_count,), device=device)
    per_view = []

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
        root_vis = sample_mesh_visible_points(
            root_points,
            root_normals,
            viewmats[view_idx],
            ks[view_idx],
            mesh_depth.depth,
            depth_abs_tolerance=float(args.depth_abs_tolerance),
            depth_rel_tolerance=float(args.depth_rel_tolerance),
            local_depth_kernel=int(args.local_depth_kernel),
            front_normal_z=float(args.front_normal_z),
        )
        sampled_ori = F.normalize(bilinear_sample(ori, root_vis.xy), dim=-1, eps=1.0e-8)
        sampled_conf = bilinear_sample(target_conf, root_vis.xy)[:, 0]
        angle_weight = view_angle_weight(root_normals, viewmats[view_idx], float(args.view_angle_power))
        weight = (sampled_conf * angle_weight * root_vis.visible.float()).clamp(0.0, 1.0)
        good = weight >= float(args.min_confidence)
        if bool(good.any()):
            screen_t = project_directions(root_points, root_tangents, viewmats[view_idx], ks[view_idx])
            screen_b = project_directions(root_points, root_bitangents, viewmats[view_idx], ks[view_idx])
            basis = torch.stack([screen_t, screen_b], dim=-1)
            coeff = (torch.linalg.pinv(basis) @ sampled_ori[:, :, None]).squeeze(-1)
            coeff = F.normalize(coeff, dim=-1, eps=1.0e-8)
            flow3d = F.normalize(coeff[:, 0:1] * root_tangents + coeff[:, 1:2] * root_bitangents, dim=-1, eps=1.0e-8)
            # Axial sign: make contributions locally agree with current accumulated direction if possible.
            has_prev = weight_sum > 0.0
            prev = F.normalize(flow3d_sum, dim=-1, eps=1.0e-8)
            flip = has_prev & ((flow3d * prev).sum(dim=-1) < 0.0)
            flow3d = torch.where(flip[:, None], -flow3d, flow3d)
            flow3d_sum[good] += flow3d[good] * weight[good, None]
            weight_sum[good] += weight[good]
            view_count[good] += 1.0
        per_view.append(
            {
                "view": int(view_idx),
                "good_roots": int(good.sum().detach().cpu()),
                "weight_sum": float(weight.sum().detach().cpu()),
                "align_matrix": align_matrix.tolist(),
            }
        )

    observed = weight_sum > 0.0
    flow3d = F.normalize(flow3d_sum / weight_sum.clamp_min(EPS)[:, None], dim=-1, eps=1.0e-8)

    # Convert axial tangent flow into a directed 3D groom vector:
    #   d = normalize(lambda * normal +/- tangent_axis).
    # Front-facing views mainly constrain the tangent axis; grazing views see the
    # normal component and can distinguish normal+axis from normal-axis.
    lambda_values = torch.tensor(
        [float(v) for v in args.direction_lambda_values.split(",") if v.strip()],
        device=device,
        dtype=torch.float32,
    )
    if lambda_values.numel() == 0:
        raise ValueError("--direction-lambda-values must contain at least one value")
    direction_score = torch.zeros((args.root_count, int(lambda_values.numel()), 2), device=device)
    direction_weight = torch.zeros((args.root_count,), device=device)
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
        root_vis = sample_mesh_visible_points(
            root_points,
            root_normals,
            viewmats[view_idx],
            ks[view_idx],
            mesh_depth.depth,
            depth_abs_tolerance=float(args.depth_abs_tolerance),
            depth_rel_tolerance=float(args.depth_rel_tolerance),
            local_depth_kernel=int(args.local_depth_kernel),
            front_normal_z=float(args.front_normal_z),
        )
        sampled_ori = F.normalize(bilinear_sample(ori, root_vis.xy), dim=-1, eps=1.0e-8)
        sampled_conf = bilinear_sample(target_conf, root_vis.xy)[:, 0]
        angle_weight = view_angle_weight(root_normals, viewmats[view_idx], float(args.view_angle_power))
        weight = (sampled_conf * angle_weight * root_vis.visible.float() * observed.float()).clamp(0.0, 1.0)
        good = weight >= float(args.min_confidence)
        if not bool(good.any()):
            continue
        direction_weight[good] += weight[good]
        for li, lam in enumerate(lambda_values):
            for si, sign in enumerate((1.0, -1.0)):
                candidate = F.normalize(lam * root_normals + float(sign) * flow3d, dim=-1, eps=1.0e-8)
                screen = F.normalize(project_directions(root_points, candidate, viewmats[view_idx], ks[view_idx]), dim=-1, eps=1.0e-8)
                # The line-art observation is still axial, so compare by abs(dot).
                agreement = torch.abs((screen * sampled_ori).sum(dim=-1)).clamp(0.0, 1.0)
                direction_score[:, li, si] += agreement * weight

    flat = direction_score.view(args.root_count, -1)
    best_score, best_flat = torch.max(flat, dim=1)
    second_score = torch.topk(flat, k=min(2, flat.shape[1]), dim=1).values[:, -1]
    best_lambda_idx = torch.div(best_flat, 2, rounding_mode="floor")
    best_sign_idx = best_flat - best_lambda_idx * 2
    best_lambda = lambda_values[best_lambda_idx]
    best_sign = torch.where(best_sign_idx == 0, torch.ones_like(best_lambda), -torch.ones_like(best_lambda))
    directed_flow3d = F.normalize(best_lambda[:, None] * root_normals + best_sign[:, None] * flow3d, dim=-1, eps=1.0e-8)
    direction_margin = torch.where(
        direction_weight > 0.0,
        (best_score - second_score) / direction_weight.clamp_min(EPS),
        torch.zeros_like(best_score),
    )
    cleaned = clean_directed_flow_on_graph(
        root_points,
        root_normals,
        flow3d,
        direction_score,
        direction_weight,
        observed,
        lambda_values,
        knn_k=int(args.clean_knn_k),
        sign_iters=int(args.clean_sign_iters),
        lambda_iters=int(args.clean_lambda_iters),
        vector_iters=int(args.clean_vector_iters),
        anchor_margin=float(args.clean_anchor_margin),
        anchor_weight=float(args.clean_anchor_weight),
        smooth_strength=float(args.clean_smooth_strength),
        vector_blend=float(args.clean_vector_blend),
    )
    cleaned_directed_flow3d = cleaned["flow"]

    # Project fused 3D flow back to view09 for visual inspection.
    diag_idx = int(args.diag_view)
    diag_gt = np.asarray(Image.open(args.data_root / "images" / f"img_{diag_idx:04d}.png").convert("RGB"))
    diag_mask_path = args.data_root / "silhouette" / f"img_{diag_idx:04d}.png"
    diag_strength, _, _ = aligned_flow_strength(args.flow_dir / f"img_{diag_idx:04d}.png", diag_mask_path, width, height)
    save_overlay(args.output_dir / f"view{diag_idx:02d}_gpt_aligned_overlay.png", diag_gt, diag_strength)

    mesh_depth = render_mesh_depth(mesh, viewmats[diag_idx], ks[diag_idx], width, height, device=device)
    diag_vis = sample_mesh_visible_points(
        root_points,
        root_normals,
        viewmats[diag_idx],
        ks[diag_idx],
        mesh_depth.depth,
        depth_abs_tolerance=float(args.depth_abs_tolerance),
        depth_rel_tolerance=float(args.depth_rel_tolerance),
        local_depth_kernel=int(args.local_depth_kernel),
        front_normal_z=float(args.front_normal_z),
    )
    screen_flow = project_directions(root_points, flow3d, viewmats[diag_idx], ks[diag_idx])
    screen_directed = project_directions(root_points, directed_flow3d, viewmats[diag_idx], ks[diag_idx])
    screen_cleaned = project_directions(root_points, cleaned_directed_flow3d, viewmats[diag_idx], ks[diag_idx])
    diag_valid = diag_vis.visible & observed
    normalized_weight = (weight_sum / torch.quantile(weight_sum[observed], 0.95).clamp_min(EPS)).clamp(0.0, 1.0) if bool(observed.any()) else weight_sum
    draw_root_flow_overlay(
        args.output_dir / f"view{diag_idx:02d}_fused_3d_flow_overlay.png",
        diag_gt,
        diag_vis.xy,
        screen_flow,
        diag_valid,
        normalized_weight,
    )
    draw_root_flow_arrow_overlay(
        args.output_dir / f"view{diag_idx:02d}_directed_3d_groom_arrows_overlay.png",
        diag_gt,
        diag_vis.xy,
        screen_directed,
        diag_valid,
        normalized_weight,
    )
    draw_root_flow_arrow_overlay(
        args.output_dir / f"view{diag_idx:02d}_cleaned_directed_3d_groom_arrows_overlay.png",
        diag_gt,
        diag_vis.xy,
        screen_cleaned,
        diag_valid,
        normalized_weight,
        color=(245, 40, 120),
    )

    # Save a compact root target file for future training integration.
    np.savez_compressed(
        args.output_dir / "guide_flow3d_targets_exclude_004_024_025.npz",
        root_points=root_points_np,
        root_normals=root_normals_np,
        face_ids=roots.face_ids.astype(np.int64),
        barycentric=roots.barycentric.astype(np.float32),
        flow3d=flow3d.detach().cpu().numpy().astype(np.float32),
        directed_flow3d=directed_flow3d.detach().cpu().numpy().astype(np.float32),
        cleaned_directed_flow3d=cleaned_directed_flow3d.detach().cpu().numpy().astype(np.float32),
        sign_cleaned_directed_flow3d=cleaned["pre_vector_smooth_flow"].detach().cpu().numpy().astype(np.float32),
        direction_lambda=best_lambda.detach().cpu().numpy().astype(np.float32),
        direction_sign=best_sign.detach().cpu().numpy().astype(np.float32),
        direction_margin=direction_margin.detach().cpu().numpy().astype(np.float32),
        cleaned_direction_lambda=cleaned["lambda"].detach().cpu().numpy().astype(np.float32),
        cleaned_direction_sign=cleaned["sign"].detach().cpu().numpy().astype(np.float32),
        direction_anchor=cleaned["anchor"].detach().cpu().numpy().astype(np.bool_),
        direction_anchor_confidence=cleaned["anchor_conf"].detach().cpu().numpy().astype(np.float32),
        direction_flipped=cleaned["flipped"].detach().cpu().numpy().astype(np.bool_),
        weight=weight_sum.detach().cpu().numpy().astype(np.float32),
        observed=observed.detach().cpu().numpy().astype(np.bool_),
        view_count=view_count.detach().cpu().numpy().astype(np.float32),
    )
    summary = {
        "views_used": views,
        "views_excluded": sorted(exclude),
        "root_count": int(args.root_count),
        "observed_roots": int(observed.sum().detach().cpu()),
        "observed_fraction": float(observed.float().mean().detach().cpu()),
        "mean_view_count_observed": float(view_count[observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
        "mean_weight_observed": float(weight_sum[observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
        "direction_lambda_mean": float(best_lambda[observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
        "direction_lambda_median": float(torch.median(best_lambda[observed]).detach().cpu()) if bool(observed.any()) else 0.0,
        "direction_margin_mean": float(direction_margin[observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
        "direction_margin_median": float(torch.median(direction_margin[observed]).detach().cpu()) if bool(observed.any()) else 0.0,
        "clean_anchor_roots": int((cleaned["anchor"] & observed).sum().detach().cpu()),
        "clean_flipped_roots": int((cleaned["flipped"] & observed).sum().detach().cpu()),
        "clean_lambda_mean": float(cleaned["lambda"][observed].mean().detach().cpu()) if bool(observed.any()) else 0.0,
        "clean_lambda_median": float(torch.median(cleaned["lambda"][observed]).detach().cpu()) if bool(observed.any()) else 0.0,
        "per_view": per_view,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(args.output_dir)


if __name__ == "__main__":
    main()

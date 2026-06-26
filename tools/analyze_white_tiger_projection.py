"""Analyze mesh-visible projection for white tiger Stage 1 initialization.

This tool compares two initialization routes:

1. direct root projection: root -> visible camera samples -> root attributes
2. UV bake projection: visible camera samples -> surface UV maps -> root sampling

Both routes use mesh depth visibility.  A root/texel that is behind another
mesh surface in the current view is not allowed to sample image color or flow.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.data.white_tiger import build_stage1_input_report, list_images  # noqa: E402
from anigroom.data.alignment import apply_alignment_to_namespace, load_alignment_config  # noqa: E402
from anigroom.grooming import make_tangent_frames  # noqa: E402
from anigroom.mesh_roots import TriangleMesh  # noqa: E402
from anigroom.mesh_roots import initialize_surface_roots_fps, read_obj_mesh, validate_surface_roots  # noqa: E402
from anigroom.projection import (  # noqa: E402
    load_xatlas_uv,
    mesh_visibility,
    render_mesh_depth,
    root_uv_from_atlas,
    sample_mesh_visible_points,
    uv_surface_samples,
)
from tools.train_white_tiger_stage1 import (  # noqa: E402
    bilinear_sample,
    face_normals_np,
    load_image,
    load_mask,
    load_orientation,
    project_directions,
    resolve_project_path,
    save_image,
)


EPS = 1.0e-8


def parse_views(text: str, train_indices: list[int], count: int) -> list[int]:
    if text.strip():
        return [int(v.strip()) for v in text.split(",") if v.strip()]
    return train_indices[: int(count)]


def normalize_depth_image(depth: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(depth)
    out = torch.zeros((*depth.shape, 3), device=depth.device)
    if not bool(finite.any()):
        return out
    values = depth[finite]
    lo = torch.quantile(values, 0.02)
    hi = torch.quantile(values, 0.98)
    norm = ((depth - lo) / (hi - lo).clamp_min(EPS)).clamp(0.0, 1.0)
    norm = torch.where(finite, 1.0 - norm, torch.zeros_like(norm))
    out[..., 0] = norm
    out[..., 1] = norm
    out[..., 2] = norm
    return out


def save_overlay(
    path: Path,
    image: torch.Tensor,
    xy: torch.Tensor,
    visible: torch.Tensor,
    in_frame: torch.Tensor,
    *,
    max_points: int,
) -> None:
    arr = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    canvas = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    xy_np = xy.detach().cpu().numpy()
    visible_np = visible.detach().cpu().numpy().astype(bool)
    in_frame_np = in_frame.detach().cpu().numpy().astype(bool)
    ids = np.nonzero(in_frame_np)[0]
    if ids.size > int(max_points):
        rng = np.random.default_rng(13)
        ids = rng.choice(ids, size=int(max_points), replace=False)
    for idx in ids:
        x, y = xy_np[idx]
        color = (20, 220, 60) if visible_np[idx] else (235, 40, 35)
        r = 2 if visible_np[idx] else 1
        draw.ellipse((float(x) - r, float(y) - r, float(x) + r, float(y) + r), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_attribute_overlay(
    path: Path,
    image: torch.Tensor,
    xy: torch.Tensor,
    valid: torch.Tensor,
    rgb: torch.Tensor,
    *,
    max_points: int,
    radius: int = 2,
) -> None:
    arr = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    canvas = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    xy_np = xy.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy().astype(bool)
    rgb_np = (rgb.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    ids = np.nonzero(valid_np)[0]
    if ids.size > int(max_points):
        rng = np.random.default_rng(17)
        ids = rng.choice(ids, size=int(max_points), replace=False)
    for idx in ids:
        x, y = xy_np[idx]
        color = tuple(int(v) for v in rgb_np[idx].tolist())
        draw.ellipse((float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_mask_overlay(path: Path, image: torch.Tensor, mask: torch.Tensor, color: tuple[int, int, int]) -> None:
    base = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    mask_np = mask.detach().cpu().numpy().astype(bool)
    overlay = base.copy()
    tint = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    overlay[mask_np] = (0.55 * overlay[mask_np].astype(np.float32) + 0.45 * tint).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path)


def save_scalar_heat(path: Path, value: torch.Tensor) -> None:
    finite = torch.isfinite(value)
    img = torch.zeros((*value.shape, 3), device=value.device)
    if bool(finite.any()):
        vals = value[finite]
        hi = torch.quantile(vals, 0.99).clamp_min(EPS)
        norm = (value / hi).clamp(0.0, 1.0)
        img[..., 0] = norm
        img[..., 1] = torch.sqrt(norm.clamp_min(0.0))
        img[..., 2] = 1.0 - norm
        img = torch.where(finite[..., None], img, torch.zeros_like(img))
    save_image(path, img)


def mask_edge_confidence(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if int(kernel_size) <= 1:
        return mask
    pad = int(kernel_size) // 2
    m = mask[..., 0].permute(0, 1) if mask.ndim == 3 else mask
    eroded = -F.max_pool2d(-m[None, None], kernel_size=int(kernel_size), stride=1, padding=pad)[0, 0]
    return eroded.clamp(0.0, 1.0)[..., None]


def view_angle_weight(normals: torch.Tensor, viewmat: torch.Tensor, power: float) -> torch.Tensor:
    normal_cam = normals @ viewmat[:3, :3].T
    weight = (-normal_cam[:, 2]).clamp(0.0, 1.0)
    if float(power) != 1.0:
        weight = weight.pow(float(power))
    return weight


def save_uv_root_scatter(
    path: Path,
    uv: torch.Tensor,
    rgb: torch.Tensor,
    observed: torch.Tensor,
    *,
    size: int,
    radius: int = 2,
) -> None:
    canvas = Image.new("RGB", (int(size), int(size)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    uv_np = uv.detach().cpu().numpy()
    rgb_np = (rgb.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    obs_np = observed.detach().cpu().numpy().astype(bool)
    ids = np.nonzero(obs_np)[0]
    for idx in ids:
        x = float(uv_np[idx, 0]) * (int(size) - 1)
        y = float(uv_np[idx, 1]) * (int(size) - 1)
        color = tuple(int(v) for v in rgb_np[idx].tolist())
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_uv_root_weight_scatter(
    path: Path,
    uv: torch.Tensor,
    weight: torch.Tensor,
    observed: torch.Tensor,
    *,
    size: int,
    radius: int = 2,
) -> None:
    max_weight = weight[observed].max().clamp_min(EPS) if bool(observed.any()) else torch.tensor(1.0, device=weight.device)
    norm = (weight / max_weight).clamp(0.0, 1.0)
    rgb = torch.stack([norm, torch.sqrt(norm.clamp_min(0.0)), 1.0 - norm], dim=-1)
    save_uv_root_scatter(path, uv, rgb, observed, size=size, radius=radius)


def flow_to_rgb(flow: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    norm = F.normalize(flow, dim=-1, eps=1.0e-8)
    rgb = torch.zeros((*flow.shape[:2], 3), device=flow.device)
    rgb[..., 0] = norm[..., 0] * 0.5 + 0.5
    rgb[..., 1] = norm[..., 1] * 0.5 + 0.5
    rgb[..., 2] = 0.5
    rgb = torch.where((weight > 0.0).expand_as(rgb), rgb, torch.zeros_like(rgb))
    return rgb


def summarize_bool(mask: torch.Tensor) -> dict[str, float]:
    count = int(mask.numel())
    true_count = int(mask.detach().sum().cpu())
    return {"count": float(count), "true": float(true_count), "fraction": float(true_count / max(count, 1))}


def mask_bbox(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    ys, xs = np.nonzero(arr > 0.5)
    if xs.size == 0:
        raise RuntimeError(f"empty mask: {path}")
    bbox = np.asarray([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
    return torch.from_numpy(bbox).to(device=device)


def load_named_camera_tensors(data_root: Path, intrinsics_file: str, extrinsics_file: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    intr = np.load(data_root / intrinsics_file).astype(np.float32)
    extr = np.load(data_root / extrinsics_file).astype(np.float32)
    return torch.from_numpy(extr).to(device=device), torch.from_numpy(intr[:, :3, :3]).to(device=device)


def load_k_rt_from_projection(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match NeuralFur/NeuS camera decomposition for cameras.npz/projection.npy."""

    out = cv.decomposeProjectionMatrix(p[:3, :4].astype(np.float64))
    k = out[0]
    r = out[1]
    t = out[2]
    k = k / k[2, 2]
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = r.T.astype(np.float32)
    pose[:3, 3] = (t[:3] / t[3])[:, 0].astype(np.float32)
    intr = np.eye(4, dtype=np.float32)
    intr[:3, :3] = k.astype(np.float32)
    return intr, pose


def load_projection_camera_tensors(data_root: Path, projection_file: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    path = data_root / projection_file
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if "arr_0" not in data.files:
            raise ValueError(f"{path} does not contain arr_0")
        projections = data["arr_0"].astype(np.float32)
    else:
        projections = np.load(path).astype(np.float32)
    intrinsics, poses = [], []
    for p in projections:
        intr, pose = load_k_rt_from_projection(p)
        intrinsics.append(intr)
        poses.append(pose)
    ks = torch.from_numpy(np.stack(intrinsics)[:, :3, :3]).to(device=device)
    viewmats = torch.linalg.inv(torch.from_numpy(np.stack(poses)).to(device=device))
    return viewmats, ks


def bbox_to_center_size(bbox: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = bbox.unbind(dim=-1)
    return torch.stack([(x0 + x1) * 0.5, (y0 + y1) * 0.5, (x1 - x0).clamp_min(1.0), (y1 - y0).clamp_min(1.0)], dim=-1)


def transformed_mesh(mesh: TriangleMesh, scale: float, translation: np.ndarray) -> TriangleMesh:
    return TriangleMesh(
        vertices=(mesh.vertices.astype(np.float32) * float(scale) + translation.astype(np.float32)[None]).astype(np.float32),
        faces=mesh.faces.copy(),
    )


def read_triangle_mesh_any(path: Path) -> TriangleMesh:
    if path.suffix.lower() == ".obj":
        return read_obj_mesh(path)
    if path.suffix.lower() == ".ply":
        import trimesh

        loaded = trimesh.load(path, force="mesh", process=False)
        if loaded.vertices is None or loaded.faces is None or len(loaded.vertices) == 0 or len(loaded.faces) == 0:
            raise ValueError(f"invalid PLY mesh: {path}")
        return TriangleMesh(
            vertices=np.asarray(loaded.vertices, dtype=np.float32),
            faces=np.asarray(loaded.faces, dtype=np.int64),
        )
    raise ValueError(f"unsupported mesh format: {path}")


def estimate_mesh_alignment(
    mesh: TriangleMesh,
    mask_paths: list[Path],
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    views: list[int],
    width: int,
    height: int,
    *,
    steps: int,
    lr: float,
    initial_scale: float,
    initial_translation: list[float],
    device: torch.device,
) -> dict[str, object]:
    vertices = torch.from_numpy(mesh.vertices).to(device=device, dtype=torch.float32)
    log_scale = torch.nn.Parameter(torch.tensor(math.log(float(initial_scale)), device=device))
    translation = torch.nn.Parameter(torch.tensor(initial_translation, device=device, dtype=torch.float32))
    targets = torch.stack([bbox_to_center_size(mask_bbox(mask_paths[idx], device)) for idx in views], dim=0)
    normalizer = torch.tensor([width, height, width, height], device=device, dtype=torch.float32)

    def projected_bbox_params(points: torch.Tensor, view_idx: int) -> torch.Tensor:
        xy, depth, _ = mesh_visibility.project_points(points, viewmats[view_idx], ks[view_idx])
        valid = depth > 1.0e-6
        if int(valid.sum().detach().cpu()) < 16:
            raise RuntimeError(f"too few vertices in front of camera for view {view_idx}")
        xy_valid = xy[valid]
        x0 = torch.quantile(xy_valid[:, 0], 0.01)
        y0 = torch.quantile(xy_valid[:, 1], 0.01)
        x1 = torch.quantile(xy_valid[:, 0], 0.99)
        y1 = torch.quantile(xy_valid[:, 1], 0.99)
        return bbox_to_center_size(torch.stack([x0, y0, x1, y1]))

    with torch.no_grad():
        initial_points = vertices * torch.exp(log_scale) + translation.view(1, 3)
        initial = torch.stack([projected_bbox_params(initial_points, idx) for idx in views], dim=0)
        initial_loss = ((initial - targets) / normalizer).square().mean()

    if int(steps) > 0:
        optimizer = torch.optim.Adam([log_scale, translation], lr=float(lr))
        for _ in range(int(steps)):
            optimizer.zero_grad(set_to_none=True)
            points = vertices * torch.exp(log_scale) + translation.view(1, 3)
            pred = torch.stack([projected_bbox_params(points, idx) for idx in views], dim=0)
            loss = ((pred - targets) / normalizer).square().mean()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        final_points = vertices * torch.exp(log_scale) + translation.view(1, 3)
        final = torch.stack([projected_bbox_params(final_points, idx) for idx in views], dim=0)
        final_loss = ((final - targets) / normalizer).square().mean()
    return {
        "enabled": bool(int(steps) > 0),
        "views": [int(v) for v in views],
        "initial_loss": float(initial_loss.detach().cpu()),
        "final_loss": float(final_loss.detach().cpu()),
        "scale": float(torch.exp(log_scale).detach().cpu()),
        "translation": [float(v) for v in translation.detach().cpu().tolist()],
        "target_center_size": targets.detach().cpu().tolist(),
        "final_center_size": final.detach().cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze white-tiger root/UV projection with mesh depth visibility.")
    parser.add_argument("--alignment-config", type=Path, default=Path("configs/white_tiger_mesh_alignment.json"))
    parser.add_argument("--data-root", type=Path, default=Path("D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk"))
    parser.add_argument("--mesh-path", type=Path, default=Path("D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj"))
    parser.add_argument("--uv-atlas", type=Path, default=Path("D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.xatlas_uv.npz"))
    parser.add_argument("--skip-uv", action="store_true")
    parser.add_argument("--camera-source", choices=["projection", "named"], default="projection")
    parser.add_argument("--projection-file", default="cameras.npz")
    parser.add_argument("--intrinsics-file", default="cameras_intr.npy")
    parser.add_argument("--extrinsics-file", default="cameras_extr.npy")
    parser.add_argument("--invert-extrinsics", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("D:/petsgaussianhair/_downloads/white_tiger_projection_analysis"))
    parser.add_argument("--root-count", type=int, default=20000)
    parser.add_argument("--candidate-multiplier", type=float, default=8.0)
    parser.add_argument("--uv-size", type=int, default=2048)
    parser.add_argument("--views", default="")
    parser.add_argument("--view-count", type=int, default=6)
    parser.add_argument("--test-stride", type=int, default=6)
    parser.add_argument("--depth-abs-tolerance", type=float, default=0.03)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.01)
    parser.add_argument("--local-depth-kernel", type=int, default=7)
    parser.add_argument("--front-normal-z", type=float, default=0.15)
    parser.add_argument("--min-confidence", type=float, default=0.08)
    parser.add_argument("--mask-edge-kernel", type=int, default=9)
    parser.add_argument("--view-angle-power", type=float, default=1.0)
    parser.add_argument("--uv-chunk-size", type=int, default=262144)
    parser.add_argument("--overlay-max-points", type=int, default=12000)
    parser.add_argument("--route-diagnostic-views", default="5,9,17,25,33")
    parser.add_argument("--init-mesh-scale", type=float, default=1.25)
    parser.add_argument("--init-mesh-translation", type=float, nargs=3, default=[0.0, 0.32, 0.0])
    parser.add_argument("--calibrate-steps", type=int, default=0)
    parser.add_argument("--calibration-lr", type=float, default=0.03)
    args = parser.parse_args()
    apply_alignment_to_namespace(args, load_alignment_config(args.alignment_config), include_uv=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal projection analysis; do not run a CPU fallback")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report = build_stage1_input_report(args.data_root, args.mesh_path, test_stride=args.test_stride)
    if report.errors:
        raise RuntimeError("input report errors: " + "; ".join(report.errors))
    width, height = report.image_size or (0, 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid image size")
    views = parse_views(args.views, report.train_indices, args.view_count)

    image_paths = list_images(Path(report.image_dir))
    mask_paths = list_images(Path(report.mask_dir))
    angle_paths = sorted((Path(report.orientation_root) / "angles").glob("*.png"))
    conf_paths = sorted((Path(report.orientation_root) / "vars").glob("*.npy"))
    if args.camera_source == "projection":
        viewmats, ks = load_projection_camera_tensors(args.data_root, args.projection_file, device)
    else:
        viewmats, ks = load_named_camera_tensors(args.data_root, args.intrinsics_file, args.extrinsics_file, device)
        if args.invert_extrinsics:
            viewmats = torch.linalg.inv(viewmats)

    mesh_raw = read_triangle_mesh_any(args.mesh_path)
    atlas = None if args.skip_uv else load_xatlas_uv(args.uv_atlas)
    if atlas is not None:
        atlas.validate(mesh_raw)
    calibration = estimate_mesh_alignment(
        mesh_raw,
        mask_paths,
        viewmats,
        ks,
        views,
        width,
        height,
        steps=args.calibrate_steps,
        lr=args.calibration_lr,
        initial_scale=args.init_mesh_scale,
        initial_translation=[float(v) for v in args.init_mesh_translation],
        device=device,
    )
    mesh = transformed_mesh(
        mesh_raw,
        float(calibration["scale"]),
        np.asarray(calibration["translation"], dtype=np.float32),
    )
    roots = initialize_surface_roots_fps(
        mesh_raw,
        args.root_count,
        candidate_multiplier=args.candidate_multiplier,
        seed=13,
        fps_device=device,
    )
    root_report = validate_surface_roots(mesh_raw, roots)
    face_normals = torch.from_numpy(face_normals_np(mesh)).to(device=device)
    root_points = torch.from_numpy(roots.points * float(calibration["scale"]) + np.asarray(calibration["translation"], dtype=np.float32)[None]).to(device=device)
    root_normals = F.normalize(face_normals[torch.from_numpy(roots.face_ids).to(device=device)], dim=-1, eps=1.0e-8)
    root_tangents, root_bitangents = make_tangent_frames(root_normals)

    import nvdiffrast.torch as dr

    ctx = dr.RasterizeCudaContext(device=device)
    if atlas is not None:
        root_uv = torch.from_numpy(root_uv_from_atlas(atlas, roots.face_ids, roots.barycentric)).to(device=device)
        uv_points, uv_normals, uv_face_ids, uv_valid = uv_surface_samples(mesh, atlas, args.uv_size, device=device, ctx=ctx)
        del uv_face_ids
        uv_flat_ids = torch.nonzero(uv_valid.reshape(-1), as_tuple=False).reshape(-1)
        uv_points_flat = uv_points.reshape(-1, 3)
        uv_normals_flat = uv_normals.reshape(-1, 3)
        uv_tangents_flat, uv_bitangents_flat = make_tangent_frames(uv_normals_flat)
    else:
        root_uv = None
        uv_valid = None
        uv_flat_ids = None
        uv_points_flat = None
        uv_normals_flat = None
        uv_tangents_flat = None
        uv_bitangents_flat = None

    direct_color_sum = torch.zeros((args.root_count, 3), device=device)
    direct_flow_sum = torch.zeros((args.root_count, 2), device=device)
    direct_weight_sum = torch.zeros((args.root_count, 1), device=device)

    uv_color_sum = torch.zeros((args.uv_size, args.uv_size, 3), device=device) if atlas is not None else None
    uv_flow_sum = torch.zeros((args.uv_size, args.uv_size, 2), device=device) if atlas is not None else None
    uv_weight_sum = torch.zeros((args.uv_size, args.uv_size, 1), device=device) if atlas is not None else None
    uv_observed_views = torch.zeros((args.uv_size, args.uv_size, 1), device=device) if atlas is not None else None

    per_view: list[dict[str, object]] = []
    for view_idx in views:
        image = load_image(image_paths[view_idx], device)
        mask = load_mask(mask_paths[view_idx], device)
        mask_conf = mask_edge_confidence(mask, args.mask_edge_kernel)
        target_orientation, target_conf = load_orientation(angle_paths[view_idx], conf_paths[view_idx], (width, height), bins=180, device=device)
        target_conf = target_conf * mask * mask_conf
        mesh_depth = render_mesh_depth(mesh, viewmats[view_idx], ks[view_idx], width, height, device=device, ctx=ctx)

        root_vis = sample_mesh_visible_points(
            root_points,
            root_normals,
            viewmats[view_idx],
            ks[view_idx],
            mesh_depth.depth,
            depth_abs_tolerance=args.depth_abs_tolerance,
            depth_rel_tolerance=args.depth_rel_tolerance,
            local_depth_kernel=args.local_depth_kernel,
            front_normal_z=args.front_normal_z,
        )
        sampled_mask = bilinear_sample(mask, root_vis.xy)[:, 0]
        sampled_conf = bilinear_sample(target_conf, root_vis.xy)[:, 0]
        root_angle = view_angle_weight(root_normals, viewmats[view_idx], args.view_angle_power)
        direct_weight = (sampled_mask * sampled_conf * root_angle * root_vis.visible.float()).clamp(0.0, 1.0)
        good_direct = direct_weight >= float(args.min_confidence)
        sampled_color = bilinear_sample(image, root_vis.xy).clamp(0.0, 1.0)
        sampled_ori = F.normalize(bilinear_sample(target_orientation, root_vis.xy), dim=-1, eps=1.0e-8)
        screen_t = project_directions(root_points, root_tangents, viewmats[view_idx], ks[view_idx])
        screen_b = project_directions(root_points, root_bitangents, viewmats[view_idx], ks[view_idx])
        basis = torch.stack([screen_t, screen_b], dim=-1)
        coeff = (torch.linalg.pinv(basis) @ sampled_ori[:, :, None]).squeeze(-1)
        coeff = F.normalize(coeff, dim=-1, eps=1.0e-8)
        direct_color_sum[good_direct] += sampled_color[good_direct] * direct_weight[good_direct, None]
        direct_flow_sum[good_direct] += coeff[good_direct] * direct_weight[good_direct, None]
        direct_weight_sum[good_direct] += direct_weight[good_direct, None]

        save_image(args.output_dir / f"view_{view_idx:03d}_mesh_depth.png", normalize_depth_image(mesh_depth.depth))
        save_mask_overlay(
            args.output_dir / f"view_{view_idx:03d}_mesh_depth_overlay.png",
            image,
            mesh_depth.valid,
            (40, 170, 255),
        )
        save_overlay(
            args.output_dir / f"view_{view_idx:03d}_root_visibility_overlay.png",
            image,
            root_vis.xy,
            root_vis.visible,
            root_vis.in_frame,
            max_points=args.overlay_max_points,
        )
        depth_delta = torch.full((height, width), torch.nan, device=device)
        finite_root = root_vis.in_frame & torch.isfinite(root_vis.mesh_depth)

        uv_view_visible_total = 0
        uv_view_weight_total = 0.0
        if atlas is not None:
            assert uv_flat_ids is not None and uv_points_flat is not None and uv_normals_flat is not None
            assert uv_tangents_flat is not None and uv_bitangents_flat is not None
            assert uv_color_sum is not None and uv_flow_sum is not None and uv_weight_sum is not None and uv_observed_views is not None
            for begin in range(0, int(uv_flat_ids.numel()), int(args.uv_chunk_size)):
                ids = uv_flat_ids[begin : begin + int(args.uv_chunk_size)]
                pts = uv_points_flat[ids]
                nrm = uv_normals_flat[ids]
                vis = sample_mesh_visible_points(
                    pts,
                    nrm,
                    viewmats[view_idx],
                    ks[view_idx],
                    mesh_depth.depth,
                    depth_abs_tolerance=args.depth_abs_tolerance,
                    depth_rel_tolerance=args.depth_rel_tolerance,
                    local_depth_kernel=args.local_depth_kernel,
                    front_normal_z=args.front_normal_z,
                )
                sm = bilinear_sample(mask, vis.xy)[:, 0]
                sc = bilinear_sample(target_conf, vis.xy)[:, 0]
                angle_w = view_angle_weight(nrm, viewmats[view_idx], args.view_angle_power)
                weight = (sm * sc * angle_w * vis.visible.float()).clamp(0.0, 1.0)
                good = weight >= float(args.min_confidence)
                if not bool(good.any()):
                    continue
                ids_good = ids[good]
                ys = torch.div(ids_good, args.uv_size, rounding_mode="floor")
                xs = ids_good - ys * args.uv_size
                col = bilinear_sample(image, vis.xy[good]).clamp(0.0, 1.0)
                ori = F.normalize(bilinear_sample(target_orientation, vis.xy[good]), dim=-1, eps=1.0e-8)
                st = project_directions(pts[good], uv_tangents_flat[ids][good], viewmats[view_idx], ks[view_idx])
                sb = project_directions(pts[good], uv_bitangents_flat[ids][good], viewmats[view_idx], ks[view_idx])
                uv_basis = torch.stack([st, sb], dim=-1)
                uv_coeff = (torch.linalg.pinv(uv_basis) @ ori[:, :, None]).squeeze(-1)
                uv_coeff = F.normalize(uv_coeff, dim=-1, eps=1.0e-8)
                w = weight[good, None]
                uv_color_sum[ys, xs] += col * w
                uv_flow_sum[ys, xs] += uv_coeff * w
                uv_weight_sum[ys, xs] += w
                uv_observed_views[ys, xs] += 1.0
                uv_view_visible_total += int(good.sum().detach().cpu())
                uv_view_weight_total += float(w.sum().detach().cpu())

        per_view.append(
            {
                "view": int(view_idx),
                "root": {
                    "in_frame": summarize_bool(root_vis.in_frame),
                    "depth_visible": summarize_bool(root_vis.depth_visible),
                    "front_facing": summarize_bool(root_vis.front_facing),
                    "visible": summarize_bool(root_vis.visible),
                    "good_projection": summarize_bool(good_direct),
                    "mean_abs_depth_delta_visible": float(root_vis.depth_delta[root_vis.visible].abs().mean().detach().cpu())
                    if bool(root_vis.visible.any())
                    else None,
                    "mean_abs_depth_delta_in_frame": float(root_vis.depth_delta[finite_root].abs().mean().detach().cpu())
                    if bool(finite_root.any())
                    else None,
                },
                "uv": {
                    "good_texels": float(uv_view_visible_total),
                    "weight_sum": float(uv_view_weight_total),
                },
            }
        )

    direct_observed = direct_weight_sum[:, 0] > 0.0
    direct_color = torch.where(direct_weight_sum > 0.0, direct_color_sum / direct_weight_sum.clamp_min(EPS), torch.zeros_like(direct_color_sum))
    direct_flow = F.normalize(direct_flow_sum / direct_weight_sum.clamp_min(EPS), dim=-1, eps=1.0e-8)

    if atlas is not None:
        assert root_uv is not None and uv_color_sum is not None and uv_flow_sum is not None and uv_weight_sum is not None and uv_observed_views is not None
        uv_color = torch.where(uv_weight_sum > 0.0, uv_color_sum / uv_weight_sum.clamp_min(EPS), torch.zeros_like(uv_color_sum))
        uv_flow = F.normalize(uv_flow_sum / uv_weight_sum.clamp_min(EPS), dim=-1, eps=1.0e-8)
        root_uv_xy = torch.stack([root_uv[:, 0] * (args.uv_size - 1), root_uv[:, 1] * (args.uv_size - 1)], dim=-1)
        root_uv_weight = bilinear_sample(uv_weight_sum, root_uv_xy)[:, 0]
        root_uv_color = bilinear_sample(uv_color, root_uv_xy)
        root_uv_flow = F.normalize(bilinear_sample(uv_flow, root_uv_xy), dim=-1, eps=1.0e-8)
        uv_root_observed = root_uv_weight > 0.0

        save_image(args.output_dir / "uv_baked_color.png", uv_color)
        save_image(args.output_dir / "uv_baked_flow.png", flow_to_rgb(uv_flow, uv_weight_sum))
        save_scalar_heat(args.output_dir / "uv_baked_weight.png", uv_weight_sum[..., 0])
        save_scalar_heat(args.output_dir / "uv_observed_view_count.png", uv_observed_views[..., 0])
        save_uv_root_scatter(args.output_dir / "root_direct_color_uv_scatter.png", root_uv, direct_color, direct_observed, size=args.uv_size)
        save_uv_root_scatter(args.output_dir / "root_direct_flow_uv_scatter.png", root_uv, torch.cat([direct_flow * 0.5 + 0.5, torch.full_like(direct_flow[:, :1], 0.5)], dim=-1), direct_observed, size=args.uv_size)
        save_uv_root_weight_scatter(args.output_dir / "root_direct_weight_uv_scatter.png", root_uv, direct_weight_sum[:, 0], direct_observed, size=args.uv_size)
        save_uv_root_scatter(args.output_dir / "root_from_uv_color_uv_scatter.png", root_uv, root_uv_color, uv_root_observed, size=args.uv_size)
        save_uv_root_scatter(args.output_dir / "root_from_uv_flow_uv_scatter.png", root_uv, torch.cat([root_uv_flow * 0.5 + 0.5, torch.full_like(root_uv_flow[:, :1], 0.5)], dim=-1), uv_root_observed, size=args.uv_size)
        save_uv_root_weight_scatter(args.output_dir / "root_from_uv_weight_uv_scatter.png", root_uv, root_uv_weight, uv_root_observed, size=args.uv_size)

        np.savez_compressed(
            args.output_dir / "projected_root_attributes.npz",
            root_points=root_points.detach().cpu().numpy(),
            root_normals=root_normals.detach().cpu().numpy(),
            root_uv=root_uv.detach().cpu().numpy(),
            direct_color=direct_color.detach().cpu().numpy(),
            direct_flow=direct_flow.detach().cpu().numpy(),
            direct_weight=direct_weight_sum[:, 0].detach().cpu().numpy(),
            direct_observed=direct_observed.detach().cpu().numpy(),
            uv_color=root_uv_color.detach().cpu().numpy(),
            uv_flow=root_uv_flow.detach().cpu().numpy(),
            uv_weight=root_uv_weight.detach().cpu().numpy(),
            uv_observed=uv_root_observed.detach().cpu().numpy(),
        )

        diagnostic_views = parse_views(args.route_diagnostic_views, views, len(views))
        for view_idx in diagnostic_views:
            if int(view_idx) not in views:
                continue
            image = load_image(image_paths[view_idx], device)
            mesh_depth = render_mesh_depth(mesh, viewmats[view_idx], ks[view_idx], width, height, device=device, ctx=ctx)
            root_vis = sample_mesh_visible_points(
                root_points,
                root_normals,
                viewmats[view_idx],
                ks[view_idx],
                mesh_depth.depth,
                depth_abs_tolerance=args.depth_abs_tolerance,
                depth_rel_tolerance=args.depth_rel_tolerance,
                local_depth_kernel=args.local_depth_kernel,
                front_normal_z=args.front_normal_z,
            )
            view_visible_direct = root_vis.visible & direct_observed
            view_visible_uv = root_vis.visible & uv_root_observed
            save_attribute_overlay(
                args.output_dir / f"view_{view_idx:03d}_root_direct_color_projected.png",
                image,
                root_vis.xy,
                view_visible_direct,
                direct_color,
                max_points=args.overlay_max_points,
            )
            save_attribute_overlay(
                args.output_dir / f"view_{view_idx:03d}_root_from_uv_color_projected.png",
                image,
                root_vis.xy,
                view_visible_uv,
                root_uv_color,
                max_points=args.overlay_max_points,
            )
            save_attribute_overlay(
                args.output_dir / f"view_{view_idx:03d}_root_direct_flow_projected.png",
                image,
                root_vis.xy,
                view_visible_direct,
                torch.cat([direct_flow * 0.5 + 0.5, torch.full_like(direct_flow[:, :1], 0.5)], dim=-1),
                max_points=args.overlay_max_points,
            )
            save_attribute_overlay(
                args.output_dir / f"view_{view_idx:03d}_root_from_uv_flow_projected.png",
                image,
                root_vis.xy,
                view_visible_uv,
                torch.cat([root_uv_flow * 0.5 + 0.5, torch.full_like(root_uv_flow[:, :1], 0.5)], dim=-1),
                max_points=args.overlay_max_points,
            )

        direct_uv_overlap = direct_observed & uv_root_observed
        if bool(direct_uv_overlap.any()):
            color_l1 = (direct_color[direct_uv_overlap] - root_uv_color[direct_uv_overlap]).abs().mean()
            flow_dot = (direct_flow[direct_uv_overlap] * root_uv_flow[direct_uv_overlap]).sum(dim=-1).abs().mean()
        else:
            color_l1 = torch.tensor(float("nan"), device=device)
            flow_dot = torch.tensor(float("nan"), device=device)
    else:
        uv_root_observed = torch.zeros_like(direct_observed)
        root_uv_weight = torch.zeros_like(direct_weight_sum[:, 0])
        direct_uv_overlap = torch.zeros_like(direct_observed)
        color_l1 = torch.tensor(float("nan"), device=device)
        flow_dot = torch.tensor(float("nan"), device=device)

    summary = {
        "data_root": str(args.data_root),
        "mesh_path": str(args.mesh_path),
        "uv_atlas": str(args.uv_atlas),
        "camera_source": str(args.camera_source),
        "projection_file": str(args.projection_file),
        "intrinsics_file": str(args.intrinsics_file),
        "extrinsics_file": str(args.extrinsics_file),
        "invert_extrinsics": bool(args.invert_extrinsics),
        "views": [int(v) for v in views],
        "image_size": [int(width), int(height)],
        "fusion_rules": {
            "depth_visibility": True,
            "front_facing_normal_z": float(args.front_normal_z),
            "mask_edge_kernel": int(args.mask_edge_kernel),
            "view_angle_power": float(args.view_angle_power),
            "min_confidence": float(args.min_confidence),
        },
        "calibration": calibration,
        "root_report": root_report,
        "uv": {
            "enabled": bool(atlas is not None),
            "resolution": int(args.uv_size),
            "valid_texel_count": int(uv_valid.sum().detach().cpu()) if uv_valid is not None else 0,
            "valid_texel_fraction": float(uv_valid.float().mean().detach().cpu()) if uv_valid is not None else 0.0,
            "observed_texel_count": int((uv_weight_sum[..., 0] > 0.0).sum().detach().cpu()) if uv_weight_sum is not None else 0,
            "observed_texel_fraction_of_valid": float(
                ((uv_weight_sum[..., 0] > 0.0) & uv_valid).float().sum().detach().cpu()
                / max(float(uv_valid.float().sum().detach().cpu()), 1.0)
            )
            if uv_weight_sum is not None and uv_valid is not None
            else 0.0,
        },
        "root_direct": {
            "observed_count": int(direct_observed.sum().detach().cpu()),
            "observed_fraction": float(direct_observed.float().mean().detach().cpu()),
            "weight_mean": float(direct_weight_sum.mean().detach().cpu()),
            "weight_nonzero_mean": float(direct_weight_sum[direct_observed].mean().detach().cpu())
            if bool(direct_observed.any())
            else None,
        },
        "root_from_uv": {
            "observed_count": int(uv_root_observed.sum().detach().cpu()),
            "observed_fraction": float(uv_root_observed.float().mean().detach().cpu()),
            "weight_mean": float(root_uv_weight.mean().detach().cpu()),
            "weight_nonzero_mean": float(root_uv_weight[uv_root_observed].mean().detach().cpu())
            if bool(uv_root_observed.any())
            else None,
        },
        "direct_vs_uv_on_overlap": {
            "overlap_root_count": int(direct_uv_overlap.sum().detach().cpu()),
            "overlap_fraction": float(direct_uv_overlap.float().mean().detach().cpu()),
            "color_l1": float(color_l1.detach().cpu()) if torch.isfinite(color_l1) else None,
            "abs_flow_dot": float(flow_dot.detach().cpu()) if torch.isfinite(flow_dot) else None,
        },
        "per_view": per_view,
    }
    (args.output_dir / "projection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

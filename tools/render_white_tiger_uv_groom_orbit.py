import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Older checkpoints may have been written by NumPy 2.x, while the active
# environment may expose the old numpy.core module path.
try:
    import numpy as _np

    sys.modules.setdefault("numpy._core", _np.core)
    sys.modules.setdefault("numpy._core.multiarray", _np.core.multiarray)
except Exception:
    pass

from anigroom.mesh import face_geometry, keep_largest_face_component, stable_frame
from anigroom.stage_a import generate_stage_a_curves
from tools.train_white_tiger_uv_groom import (
    TextureGroom,
    adaptive_render_sample_mask,
    append_surface_layer,
    load_camera_centers,
    load_obj_mesh_with_uv,
    load_or_build_xatlas_uv,
    load_projection_mats,
    sample_curve_tangents,
    sample_full_body_roots,
    save_tensor_image,
    splat_radius_from_strand_width,
    splat_render,
)


def _resolve(project_root: Path, path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        if p.exists():
            return p
        # Server checkpoints store /ssdwork/liuhaohan/petsgaussianhair paths.
        parts = p.parts
        if "petsgaussianhair" in parts:
            rel = Path(*parts[parts.index("petsgaussianhair") + 1 :])
            candidate = project_root / rel
            if candidate.exists():
                return candidate
        return p
    return project_root / p


def _rebuild_root_data(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_mode: str,
    uv_vertices: np.ndarray | None,
    face_uvs: np.ndarray | None,
    saved_root_data: dict,
) -> dict[str, np.ndarray]:
    face_ids = np.asarray(saved_root_data["face_ids"], dtype=np.int64).reshape(-1)
    bary = np.asarray(saved_root_data["bary"], dtype=np.float32).reshape(-1, 3)
    axis = np.asarray(saved_root_data["axis"], dtype=np.float32).reshape(3)
    center = np.asarray(saved_root_data["center"], dtype=np.float32).reshape(3)

    centers, normals, areas = face_geometry(vertices, faces)
    tri = vertices[faces[face_ids]]
    roots = (tri * bary[:, :, None]).sum(axis=1)
    chosen_normals = normals[face_ids]
    tangents, bitangents = stable_frame(chosen_normals, axis)

    if uv_mode in {"xatlas", "obj"}:
        if uv_vertices is None or face_uvs is None:
            raise ValueError(f"uv_mode={uv_mode} requires UV vertices and face UVs")
        uv_tri = uv_vertices[face_uvs[face_ids]]
        uv = (uv_tri * bary[:, :, None]).sum(axis=1)
        uv = np.mod(uv, 1.0)
        uv = np.clip(uv, 0.0, 1.0).astype(np.float32)
    elif uv_mode == "cylindrical":
        b1, b2 = stable_frame(axis[None, :], axis)
        rel = roots - center[None, :]
        a = rel @ b1[0]
        b = rel @ b2[0]
        theta = np.arctan2(b, a)
        body_tmp = roots @ axis
        body_u_tmp = (body_tmp - body_tmp.min()) / max(float(body_tmp.max() - body_tmp.min()), 1e-8)
        uv = np.stack([body_u_tmp, (theta + math.pi) / (2.0 * math.pi)], axis=1).astype(np.float32)
    else:
        raise ValueError(f"unsupported uv_mode: {uv_mode}")

    candidates = keep_largest_face_component(faces, np.where(areas > 0)[0])
    axis_coord = roots @ axis
    axis_lo, axis_hi = np.quantile(centers[candidates] @ axis, [0.002, 0.998])
    body_u = np.clip((axis_coord - axis_lo) / max(float(axis_hi - axis_lo), 1e-8), 0.0, 1.0)
    bbox_min = vertices.min(axis=0, keepdims=True)
    bbox_max = vertices.max(axis=0, keepdims=True)
    coord = np.clip((roots - bbox_min) / np.maximum(bbox_max - bbox_min, 1e-8), 0.0, 1.0)
    return {
        "roots": roots.astype(np.float32),
        "normals": chosen_normals.astype(np.float32),
        "tangents": tangents.astype(np.float32),
        "bitangents": bitangents.astype(np.float32),
        "uv": uv.astype(np.float32),
        "coord": coord.astype(np.float32),
        "body_u": body_u.astype(np.float32)[:, None],
        "face_ids": face_ids,
        "bary": bary,
        "axis": axis,
        "center": center,
    }


def _edit_region_mask(uv: torch.Tensor, body_u: torch.Tensor, edit_args: argparse.Namespace) -> torch.Tensor:
    region = str(edit_args.edit_region)
    if region == "none":
        return torch.zeros_like(body_u[:, :1])
    body_axis = body_u[:, :1].clamp(0.0, 1.0)
    if region == "head":
        mask = torch.sigmoid(35.0 * (body_axis - float(edit_args.edit_head_start)))
    elif region == "torso":
        lo = float(edit_args.edit_body_u_min)
        hi = float(edit_args.edit_body_u_max)
        mask = torch.sigmoid(45.0 * (body_axis - lo)) * torch.sigmoid(45.0 * (hi - body_axis))
    elif region == "tail":
        mask = torch.sigmoid(35.0 * (float(edit_args.edit_tail_end) - body_axis))
    elif region == "body_u":
        lo = float(edit_args.edit_body_u_min)
        hi = float(edit_args.edit_body_u_max)
        mask = torch.sigmoid(45.0 * (body_axis - lo)) * torch.sigmoid(45.0 * (hi - body_axis))
    elif region == "uv_circle":
        center = torch.tensor(edit_args.edit_uv_center, device=uv.device, dtype=uv.dtype).view(1, 2)
        radius = max(float(edit_args.edit_uv_radius), 1e-6)
        dist = torch.linalg.norm(uv - center, dim=-1, keepdim=True)
        mask = torch.sigmoid(40.0 * (radius - dist))
    else:
        raise ValueError(f"unsupported edit region: {region}")
    return mask.clamp(0.0, 1.0)


def apply_edit_to_params(
    params: dict[str, torch.Tensor],
    uv: torch.Tensor,
    body_u: torch.Tensor,
    edit_args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    mask = _edit_region_mask(uv, body_u, edit_args)
    if float(mask.max().detach().cpu()) <= 1e-6:
        return params
    edited = dict(params)
    if float(edit_args.edit_coverage_logit) != 0.0:
        edited["coverage_logit"] = edited["coverage_logit"] + mask * float(edit_args.edit_coverage_logit)
    if float(edit_args.edit_density_logit) != 0.0:
        edited["density_logit"] = edited["density_logit"] + mask * float(edit_args.edit_density_logit)
    if float(edit_args.edit_length_logit) != 0.0:
        edited["length_logit"] = edited["length_logit"] + mask * float(edit_args.edit_length_logit)
    if float(edit_args.edit_root_width_logit) != 0.0:
        edited["root_width_raw"] = edited["root_width_raw"] + mask * float(edit_args.edit_root_width_logit)
    if float(edit_args.edit_tip_width_logit) != 0.0:
        edited["tip_width_raw"] = edited["tip_width_raw"] + mask * float(edit_args.edit_tip_width_logit)
    if float(edit_args.edit_lift_logit) != 0.0:
        edited["lift"] = edited["lift"] + mask * float(edit_args.edit_lift_logit)
    if float(edit_args.edit_sag_logit) != 0.0:
        edited["sag"] = edited["sag"] + mask * float(edit_args.edit_sag_logit)
    if float(edit_args.edit_bend_logit) != 0.0:
        edited["bend"] = edited["bend"] + mask * float(edit_args.edit_bend_logit)
    if float(edit_args.edit_stiffness_logit) != 0.0:
        edited["stiffness_logit"] = edited["stiffness_logit"] + mask * float(edit_args.edit_stiffness_logit)
    if float(edit_args.edit_flow_scale) != 1.0 or float(edit_args.edit_flow_rotate_deg) != 0.0:
        fx = edited["flow_x"]
        fy = edited["flow_y"]
        angle = math.radians(float(edit_args.edit_flow_rotate_deg))
        c = math.cos(angle)
        s = math.sin(angle)
        scale = float(edit_args.edit_flow_scale)
        rx = scale * (c * fx - s * fy)
        ry = scale * (s * fx + c * fy)
        edited["flow_x"] = fx * (1.0 - mask) + rx * mask
        edited["flow_y"] = fy * (1.0 - mask) + ry * mask
    blend = float(edit_args.edit_color_blend)
    if blend > 0.0:
        tint = torch.tensor(edit_args.edit_color, device=uv.device, dtype=uv.dtype).view(1, 3).clamp(0.0, 1.0)
        w = (mask * blend).clamp(0.0, 1.0)
        edited["root_rgb"] = edited["root_rgb"] * (1.0 - w) + tint * w
        edited["tip_rgb"] = edited["tip_rgb"] * (1.0 - w) + tint * w
    return edited


def render_frame(
    model: TextureGroom,
    args: argparse.Namespace,
    edit_args: argparse.Namespace,
    roots: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    uv: torch.Tensor,
    coord: torch.Tensor,
    body_u: torch.Tensor,
    sample_idx: torch.Tensor,
    pmat: torch.Tensor,
    cam_center: torch.Tensor,
    height: int,
    width: int,
    surface_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    params = model.sample(uv, coord, body_u)
    params = apply_edit_to_params(params, uv, body_u, edit_args)
    groom = generate_stage_a_curves(roots, normals, tangents, bitangents, params, int(args.curve_samples))
    scale = torch.exp(model.log_scale).clamp(0.25, 4.0)
    curves = groom.curves * scale + model.trans.view(1, 1, 3)
    pts = curves[:, sample_idx].reshape(-1, 3)
    nrm = normals[:, None, :].expand(-1, int(args.render_samples), -1).reshape(-1, 3)
    cols = groom.color[:, sample_idx].reshape(-1, 3)
    alpha = groom.alpha.expand(-1, int(args.render_samples), -1).reshape(-1, 1) * float(args.alpha_scale)
    tangent_vectors = sample_curve_tangents(curves, sample_idx) if args.splat_mode == "oriented" else None
    point_radius = None
    if bool(args.strand_width_radius):
        point_radius = splat_radius_from_strand_width(
            groom.width[:, sample_idx],
            float(args.splat_radius),
            float(args.radius_width_ref),
            float(args.radius_width_min_scale),
            float(args.radius_width_max_scale),
        )
    if bool(args.adaptive_render_samples):
        active_mask, _ = adaptive_render_sample_mask(
            curves,
            groom.length,
            int(sample_idx.numel()),
            int(args.adaptive_min_render_samples),
        )
        flat_active = active_mask.reshape(-1)
        pts = pts[flat_active]
        nrm = nrm[flat_active]
        cols = cols[flat_active]
        alpha = alpha[flat_active]
        if tangent_vectors is not None:
            tangent_vectors = tangent_vectors[flat_active]
        if point_radius is not None:
            point_radius = point_radius.reshape(-1, 1)[flat_active]

    if surface_tensors is None:
        base_pts = roots * scale + model.trans.view(1, 3)
        base_nrm = normals
        base_cols = groom.color[:, 0]
        base_alpha = groom.alpha[:, 0] * float(args.surface_alpha_scale)
        pts = torch.cat([base_pts, pts], dim=0)
        nrm = torch.cat([base_nrm, nrm], dim=0)
        cols = torch.cat([base_cols, cols], dim=0)
        alpha = torch.cat([base_alpha, alpha], dim=0)
        if tangent_vectors is not None:
            tangent_vectors = torch.cat([torch.zeros_like(base_pts), tangent_vectors], dim=0)
        if point_radius is not None:
            base_radius = splat_radius_from_strand_width(
                groom.root_width,
                float(args.splat_radius),
                float(args.radius_width_ref),
                float(args.radius_width_min_scale),
                float(args.radius_width_max_scale),
            )
            point_radius = torch.cat([base_radius, point_radius], dim=0)
    else:
        surface_roots, surface_normals, surface_uv, surface_coord, surface_body_u = surface_tensors
        pts, nrm, cols, alpha = append_surface_layer(
            model,
            pts,
            nrm,
            cols,
            alpha,
            surface_roots,
            surface_normals,
            surface_uv,
            scale,
            float(args.surface_alpha_scale),
        )
        if tangent_vectors is not None:
            tangent_vectors = torch.cat([torch.zeros_like(surface_roots), tangent_vectors], dim=0)
        if point_radius is not None:
            surface_radius = torch.full(
                (surface_roots.shape[0], 1),
                float(args.splat_radius),
                device=point_radius.device,
                dtype=point_radius.dtype,
            )
            point_radius = torch.cat([surface_radius, point_radius], dim=0)

    image, _ = splat_render(
        pts,
        nrm,
        cols,
        alpha,
        pmat,
        cam_center,
        height,
        width,
        float(args.splat_radius),
        float(args.alpha_cap),
        float(args.depth_band),
        float(args.depth_sharpness),
        tangent_vectors,
        float(args.tangent_radius_scale),
        float(args.normal_radius_scale),
        point_radius,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--views", default="all", help="'all' or comma-separated camera indices")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-targets", action="store_true")
    parser.add_argument("--edit-region", choices=["none", "head", "torso", "tail", "body_u", "uv_circle"], default="none")
    parser.add_argument("--edit-head-start", type=float, default=0.70)
    parser.add_argument("--edit-tail-end", type=float, default=0.18)
    parser.add_argument("--edit-body-u-min", type=float, default=0.30)
    parser.add_argument("--edit-body-u-max", type=float, default=0.70)
    parser.add_argument("--edit-uv-center", type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument("--edit-uv-radius", type=float, default=0.15)
    parser.add_argument("--edit-coverage-logit", type=float, default=0.0)
    parser.add_argument("--edit-density-logit", type=float, default=0.0)
    parser.add_argument("--edit-length-logit", type=float, default=0.0)
    parser.add_argument("--edit-root-width-logit", type=float, default=0.0)
    parser.add_argument("--edit-tip-width-logit", type=float, default=0.0)
    parser.add_argument("--edit-lift-logit", type=float, default=0.0)
    parser.add_argument("--edit-sag-logit", type=float, default=0.0)
    parser.add_argument("--edit-bend-logit", type=float, default=0.0)
    parser.add_argument("--edit-stiffness-logit", type=float, default=0.0)
    parser.add_argument("--edit-flow-rotate-deg", type=float, default=0.0)
    parser.add_argument("--edit-flow-scale", type=float, default=1.0)
    parser.add_argument("--edit-color", type=float, nargs=3, default=[0.92, 0.90, 0.84])
    parser.add_argument("--edit-color-blend", type=float, default=0.0)
    cli = parser.parse_args()

    project_root = Path(cli.project_root).resolve()
    out = Path(cli.out)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cli.device if cli.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    mesh_path = _resolve(project_root, ckpt_args.mesh)
    data_root = _resolve(project_root, ckpt_args.data)
    width = int(cli.width or ckpt_args.width)
    height = int(round(width * 1080.0 / 1920.0))

    vertices, faces, obj_uv_vertices, obj_face_uvs = load_obj_mesh_with_uv(mesh_path)
    if ckpt_args.uv_mode == "obj":
        uv_vertices, face_uvs = obj_uv_vertices, obj_face_uvs
    elif ckpt_args.uv_mode == "xatlas":
        uv_cache = Path(ckpt_args.uv_cache) if ckpt_args.uv_cache else mesh_path.with_suffix(".xatlas_uv.npz")
        uv_cache = _resolve(project_root, str(uv_cache))
        uv_vertices, face_uvs = load_or_build_xatlas_uv(mesh_path, vertices, faces, uv_cache)
    else:
        uv_vertices, face_uvs = None, None

    root_data = _rebuild_root_data(vertices, faces, ckpt_args.uv_mode, uv_vertices, face_uvs, ckpt["root_data"])
    roots = torch.tensor(root_data["roots"], device=device)
    normals = torch.tensor(root_data["normals"], device=device)
    tangents = torch.tensor(root_data["tangents"], device=device)
    bitangents = torch.tensor(root_data["bitangents"], device=device)
    uv = torch.tensor(root_data["uv"], device=device)
    coord = torch.tensor(root_data["coord"], device=device)
    body_u = torch.tensor(root_data["body_u"], device=device)

    model = TextureGroom(
        int(ckpt_args.tex_h),
        int(ckpt_args.tex_w),
        float(ckpt_args.init_scale),
        list(ckpt_args.init_trans),
        bool(ckpt_args.use_triplane),
        int(ckpt_args.triplane_h),
        int(ckpt_args.triplane_w),
        float(ckpt_args.triplane_scale),
        bool(ckpt_args.use_coarse_texture),
        int(ckpt_args.coarse_h),
        int(ckpt_args.coarse_w),
        float(ckpt_args.coarse_scale),
        bool(ckpt_args.use_head_atlas),
        int(ckpt_args.head_h),
        int(ckpt_args.head_w),
        float(ckpt_args.head_start),
        float(ckpt_args.head_scale),
        int(ckpt_args.surface_roots) > 0 and float(ckpt_args.surface_alpha_scale) > 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    surface_tensors = None
    if int(ckpt_args.surface_roots) > 0 and model.surface_texture is not None:
        surface_data = sample_full_body_roots(
            vertices,
            faces,
            int(ckpt_args.surface_roots),
            int(ckpt_args.seed) + 97,
            ckpt_args.uv_mode,
            uv_vertices,
            face_uvs,
            None,
            ckpt_args.root_distribution,
        )
        surface_tensors = (
            torch.tensor(surface_data["roots"], device=device),
            torch.tensor(surface_data["normals"], device=device),
            torch.tensor(surface_data["uv"], device=device),
            torch.tensor(surface_data["coord"], device=device),
            torch.tensor(surface_data["body_u"], device=device),
        )

    pmats = load_projection_mats(data_root, width, device)
    cam_centers = load_camera_centers(data_root, device)
    if cli.views == "all":
        view_ids = list(range(int(pmats.shape[0])))
    else:
        view_ids = [int(v) for v in cli.views.split(",") if v.strip()]
    sample_idx = torch.linspace(
        0,
        int(ckpt_args.curve_samples) - 1,
        int(ckpt_args.render_samples),
        device=device,
    ).long()

    with torch.no_grad():
        for frame_idx, view_id in enumerate(view_ids):
            image = render_frame(
                model,
                ckpt_args,
                cli,
                roots,
                normals,
                tangents,
                bitangents,
                uv,
                coord,
                body_u,
                sample_idx,
                pmats[view_id],
                cam_centers[view_id],
                height,
                width,
                surface_tensors,
            )
            save_tensor_image(frames_dir / f"frame_{frame_idx:04d}.png", image)
            if cli.include_targets:
                from PIL import Image

                image_paths = sorted((data_root / "images").glob("*.png"))
                if view_id < len(image_paths):
                    target = Image.open(image_paths[view_id]).convert("RGB").resize((width, height))
                    target.save(frames_dir / f"target_{frame_idx:04d}.png")
            print(f"rendered frame {frame_idx + 1}/{len(view_ids)} view {view_id}", flush=True)

    meta = {
        "checkpoint": str(Path(cli.checkpoint).resolve()),
        "iter": int(ckpt.get("iter", -1)),
        "data_root": str(data_root),
        "mesh": str(mesh_path),
        "width": width,
        "height": height,
        "views": view_ids,
        "frame_count": len(view_ids),
        "edit": {
            "region": cli.edit_region,
            "head_start": cli.edit_head_start,
            "tail_end": cli.edit_tail_end,
            "body_u_min": cli.edit_body_u_min,
            "body_u_max": cli.edit_body_u_max,
            "uv_center": cli.edit_uv_center,
            "uv_radius": cli.edit_uv_radius,
            "coverage_logit": cli.edit_coverage_logit,
            "density_logit": cli.edit_density_logit,
            "length_logit": cli.edit_length_logit,
            "root_width_logit": cli.edit_root_width_logit,
            "tip_width_logit": cli.edit_tip_width_logit,
            "lift_logit": cli.edit_lift_logit,
            "sag_logit": cli.edit_sag_logit,
            "bend_logit": cli.edit_bend_logit,
            "stiffness_logit": cli.edit_stiffness_logit,
            "flow_rotate_deg": cli.edit_flow_rotate_deg,
            "flow_scale": cli.edit_flow_scale,
            "color": cli.edit_color,
            "color_blend": cli.edit_color_blend,
        },
    }
    (out / "orbit_meta.json").write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")
    print(str(out.resolve()), flush=True)


if __name__ == "__main__":
    main()

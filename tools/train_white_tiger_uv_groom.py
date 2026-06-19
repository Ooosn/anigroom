import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.mesh import (
    face_geometry,
    keep_largest_face_component,
    normalize as normalize_np,
    stable_frame,
)
from anigroom.preview import draw_curve_projection
from anigroom.stage_a import fixed_gaussian_proxy, generate_stage_a_curves


EPS = 1e-8
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
NUMPY_EXTENSIONS = (".npy", ".npz")
ORIENTATION_EXTENSIONS = IMAGE_EXTENSIONS + NUMPY_EXTENSIONS


def torch_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=EPS)


def infer_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(points), 1)
    eigval, eigvec = np.linalg.eigh(cov)
    axis = eigvec[:, int(np.argmax(eigval))].astype(np.float32)
    if axis @ np.array([0.0, 0.0, 1.0], dtype=np.float32) < 0:
        axis = -axis
    return normalize_np(axis[None, :])[0]


def make_perp_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    b1 = up - axis * float(up @ axis)
    if np.linalg.norm(b1) < 1e-4:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b1 = up - axis * float(up @ axis)
    b1 = normalize_np(b1[None, :])[0]
    b2 = normalize_np(np.cross(axis, b1)[None, :])[0]
    return b1.astype(np.float32), b2.astype(np.float32)


def radical_inverse_sequence(count: int, base: int, start_index: int) -> np.ndarray:
    idx = np.arange(start_index, start_index + count, dtype=np.int64)
    out = np.zeros(count, dtype=np.float32)
    factor = 1.0 / float(base)
    while np.any(idx > 0):
        out += (idx % base).astype(np.float32) * factor
        idx //= base
        factor /= float(base)
    return out


def low_discrepancy_unit_square(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    start = int(abs(seed) % 8191) + 1
    return (
        radical_inverse_sequence(count, 2, start),
        radical_inverse_sequence(count, 3, start),
    )


def sample_full_body_roots(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
    uv_mode: str = "cylindrical",
    uv_vertices: np.ndarray | None = None,
    face_uvs: np.ndarray | None = None,
    face_sampling_weights: np.ndarray | None = None,
    root_distribution: str = "random",
) -> dict[str, np.ndarray]:
    centers, normals, areas = face_geometry(vertices, faces)
    candidates = keep_largest_face_component(faces, np.where(areas > 0)[0])
    axis = infer_axis(centers[candidates])
    b1, b2 = make_perp_basis(axis)
    center = centers[candidates].mean(axis=0).astype(np.float32)

    rng = np.random.default_rng(seed)
    weighted_areas = areas[candidates].copy()
    if face_sampling_weights is not None:
        if face_sampling_weights.shape[0] != faces.shape[0]:
            raise ValueError(
                f"face_sampling_weights has {face_sampling_weights.shape[0]} entries, expected {faces.shape[0]}"
            )
        weighted_areas = weighted_areas * np.clip(face_sampling_weights[candidates], 1e-4, None)
    probs = weighted_areas / np.maximum(weighted_areas.sum(), EPS)
    if root_distribution == "random":
        chosen = rng.choice(candidates, size=count, replace=True, p=probs)
        r1 = rng.random(count, dtype=np.float32)
        r2 = rng.random(count, dtype=np.float32)
    elif root_distribution == "stratified":
        # Systematic resampling keeps the target face distribution unbiased while
        # avoiding multinomial clumping over large smooth surface regions.
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        positions = (float(rng.random()) + np.arange(count, dtype=np.float64)) / float(max(count, 1))
        indices = np.searchsorted(cdf, positions, side="right")
        indices = np.clip(indices, 0, candidates.shape[0] - 1)
        chosen = candidates[indices]
        rng.shuffle(chosen)
        r1, r2 = low_discrepancy_unit_square(count, seed)
    else:
        raise ValueError(f"unsupported root_distribution: {root_distribution}")
    tri = vertices[faces[chosen]]
    sr1 = np.sqrt(r1)
    bary = np.stack([1.0 - sr1, sr1 * (1.0 - r2), sr1 * r2], axis=1).astype(np.float32)
    roots = (tri * bary[:, :, None]).sum(axis=1)
    chosen_normals = normals[chosen]
    tangents, bitangents = stable_frame(chosen_normals, axis)

    rel = roots - center[None, :]
    axis_coord = roots @ axis
    axis_lo, axis_hi = np.quantile(centers[candidates] @ axis, [0.002, 0.998])
    body_u = np.clip((axis_coord - axis_lo) / max(float(axis_hi - axis_lo), EPS), 0.0, 1.0)
    if uv_mode in {"xatlas", "obj"}:
        if uv_vertices is None or face_uvs is None:
            raise ValueError(f"uv_mode={uv_mode} requires uv_vertices and face_uvs")
        uv_tri = uv_vertices[face_uvs[chosen]]
        uv = (uv_tri * bary[:, :, None]).sum(axis=1)
        uv = np.mod(uv, 1.0)
        uv = np.clip(uv, 0.0, 1.0).astype(np.float32)
    elif uv_mode == "cylindrical":
        a = rel @ b1
        b = rel @ b2
        theta = np.arctan2(b, a)
        v = (theta + math.pi) / (2.0 * math.pi)
        uv = np.stack([body_u, v], axis=1).astype(np.float32)
    else:
        raise ValueError(f"unsupported uv_mode: {uv_mode}")
    bbox_min = vertices.min(axis=0, keepdims=True)
    bbox_max = vertices.max(axis=0, keepdims=True)
    coord = np.clip((roots - bbox_min) / np.maximum(bbox_max - bbox_min, EPS), 0.0, 1.0)

    return {
        "roots": roots.astype(np.float32),
        "normals": chosen_normals.astype(np.float32),
        "tangents": tangents.astype(np.float32),
        "bitangents": bitangents.astype(np.float32),
        "uv": uv,
        "coord": coord.astype(np.float32),
        "body_u": body_u.astype(np.float32)[:, None],
        "face_ids": chosen.astype(np.int64),
        "bary": bary.astype(np.float32),
        "axis": axis.astype(np.float32),
        "center": center.astype(np.float32),
    }


def concat_root_data(base: dict[str, np.ndarray], extra: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    base_count = int(np.asarray(base["roots"]).shape[0])
    extra_count = int(np.asarray(extra["roots"]).shape[0])
    out: dict[str, np.ndarray] = {}
    for key, value in base.items():
        extra_value = extra.get(key)
        if (
            isinstance(value, np.ndarray)
            and isinstance(extra_value, np.ndarray)
            and value.ndim >= 1
            and extra_value.ndim >= 1
            and value.shape[0] == base_count
            and extra_value.shape[0] == extra_count
            and value.shape[1:] == extra_value.shape[1:]
        ):
            out[key] = np.concatenate([value, extra_value], axis=0)
        else:
            out[key] = value
    return out


def build_face_adjacency(faces: np.ndarray) -> list[np.ndarray]:
    vertex_to_faces: dict[int, list[int]] = {}
    for face_id, tri in enumerate(faces):
        for vertex_id in tri:
            vertex_to_faces.setdefault(int(vertex_id), []).append(int(face_id))
    neighbors: list[set[int]] = [set() for _ in range(int(faces.shape[0]))]
    for face_list in vertex_to_faces.values():
        if len(face_list) <= 1:
            continue
        for face_id in face_list:
            neighbors[face_id].update(face_list)
    out: list[np.ndarray] = []
    for face_id, neigh in enumerate(neighbors):
        neigh.discard(face_id)
        out.append(np.asarray(sorted(neigh), dtype=np.int64))
    return out


def build_root_graph_edges(
    face_ids: np.ndarray,
    face_neighbors: list[np.ndarray],
    edge_count: int,
    seed: int,
) -> np.ndarray:
    root_count = int(face_ids.shape[0])
    if root_count < 2 or edge_count <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    face_to_roots: dict[int, list[int]] = {}
    for root_id, face_id in enumerate(face_ids.astype(np.int64, copy=False)):
        face_to_roots.setdefault(int(face_id), []).append(int(root_id))
    face_to_roots_np = {
        face_id: np.asarray(root_ids, dtype=np.int64)
        for face_id, root_ids in face_to_roots.items()
    }

    rng = np.random.default_rng(seed)
    source_ids = rng.integers(0, root_count, size=max(edge_count * 4, edge_count + 256), endpoint=False)
    edges: list[tuple[int, int]] = []
    for src in source_ids:
        if len(edges) >= edge_count:
            break
        face_id = int(face_ids[int(src)])
        pools = [face_to_roots_np.get(face_id)]
        if 0 <= face_id < len(face_neighbors):
            for neigh_face in face_neighbors[face_id]:
                pools.append(face_to_roots_np.get(int(neigh_face)))
        candidates = [pool for pool in pools if pool is not None and pool.size > 0]
        if not candidates:
            continue
        cand = np.concatenate(candidates, axis=0)
        if cand.size <= 1:
            continue
        dst = int(cand[int(rng.integers(0, cand.size))])
        if dst == int(src):
            if cand.size <= 2:
                continue
            dst = int(cand[int(rng.integers(0, cand.size))])
            if dst == int(src):
                continue
        edges.append((int(src), dst))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def mesh_graph_groom_smooth_loss(
    params: dict[str, torch.Tensor],
    edges: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if edges is None or edges.numel() == 0:
        return params["length_logit"].new_tensor(0.0), {
            "mesh_graph_smooth_edges": 0,
            "mesh_graph_smooth_geom": 0.0,
            "mesh_graph_smooth_flow": 0.0,
            "mesh_graph_smooth_color": 0.0,
        }
    src = edges[:, 0].long()
    dst = edges[:, 1].long()
    geom = torch.cat(
        [
            0.35 * torch.sigmoid(params["coverage_logit"]),
            0.75 * torch.sigmoid(params["density_logit"]),
            1.25 * torch.sigmoid(params["length_logit"]),
            1.00 * params["root_width_raw"],
            1.00 * params["tip_width_raw"],
            0.90 * params["lift"],
            0.90 * params["sag"],
            0.90 * params["bend"],
            0.65 * torch.sigmoid(params["stiffness_logit"]),
        ],
        dim=-1,
    )
    geom_loss = F.smooth_l1_loss(geom[src], geom[dst], reduction="mean")

    flow = torch.cat([params["flow_x"], params["flow_y"]], dim=-1)
    flow = torch_normalize(flow)
    flow_dot = (flow[src] * flow[dst]).sum(dim=-1).clamp(-1.0, 1.0)
    flow_loss = (1.0 - flow_dot).mean()

    color = torch.cat(
        [
            params["root_rgb"],
            params["tip_rgb"],
            0.25 * torch.tanh(params["darkness"]),
        ],
        dim=-1,
    )
    color_loss = F.smooth_l1_loss(color[src], color[dst], reduction="mean")
    loss = geom_loss + 0.75 * flow_loss + 0.12 * color_loss
    stats = {
        "mesh_graph_smooth_edges": int(edges.shape[0]),
        "mesh_graph_smooth_geom": float(geom_loss.detach().cpu()),
        "mesh_graph_smooth_flow": float(flow_loss.detach().cpu()),
        "mesh_graph_smooth_color": float(color_loss.detach().cpu()),
    }
    return loss, stats


@torch.no_grad()
def compute_dynamic_residual_face_weights(
    residual_values: torch.Tensor,
    residual_weight: torch.Tensor,
    root_face_ids: torch.Tensor,
    face_count: int,
    boost: float,
    gamma: float,
    min_weight: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    device = residual_values.device
    face_score = torch.zeros(face_count, 1, device=device, dtype=residual_values.dtype)
    face_weight = torch.zeros_like(face_score)
    valid = residual_weight[:, 0] > float(min_weight)
    if not bool(valid.any()):
        weights = np.ones(face_count, dtype=np.float32)
        return weights, {
            "root_densify_valid_roots": 0,
            "root_densify_visible_face_fraction": 0.0,
            "root_densify_residual_mean": 0.0,
            "root_densify_residual_p95": 0.0,
            "root_densify_weight_min": 1.0,
            "root_densify_weight_mean": 1.0,
            "root_densify_weight_max": 1.0,
        }
    face_ids = root_face_ids[valid].long()
    conf = residual_weight[valid].clamp(0.0, 1.0)
    score = residual_values[valid].clamp(0.0, 1.0) * conf
    face_score.index_add_(0, face_ids, score)
    face_weight.index_add_(0, face_ids, conf)
    raw = face_score / torch.clamp(face_weight, min=1e-6)
    raw = torch.where(face_weight > 1e-6, raw, torch.zeros_like(raw))
    known = raw[:, 0] > 0.0
    if int(known.sum().item()) > 16:
        known_values = raw[known, 0]
        lo = torch.quantile(known_values, 0.10)
        hi = torch.quantile(known_values, 0.98)
        norm = ((raw - lo) / torch.clamp(hi - lo, min=1e-6)).clamp(0.0, 1.0)
    else:
        norm = raw.clamp(0.0, 1.0)
    weights_t = 1.0 + float(boost) * torch.pow(norm, float(gamma))
    stats = {
        "root_densify_valid_roots": int(valid.detach().sum().cpu()),
        "root_densify_visible_face_fraction": float((face_weight[:, 0] > 1e-6).float().mean().cpu()),
        "root_densify_residual_mean": float(norm.mean().cpu()),
        "root_densify_residual_p95": float(torch.quantile(norm[:, 0], 0.95).cpu()),
        "root_densify_weight_min": float(weights_t.min().cpu()),
        "root_densify_weight_mean": float(weights_t.mean().cpu()),
        "root_densify_weight_max": float(weights_t.max().cpu()),
    }
    return weights_t[:, 0].detach().cpu().numpy().astype(np.float32), stats


def rebuild_root_data_from_checkpoint(
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
    candidates = keep_largest_face_component(faces, np.where(areas > 0)[0])
    tri = vertices[faces[face_ids]]
    roots = (tri * bary[:, :, None]).sum(axis=1)
    chosen_normals = normals[face_ids]
    tangents, bitangents = stable_frame(chosen_normals, axis)

    if uv_mode in {"xatlas", "obj"}:
        if uv_vertices is None or face_uvs is None:
            raise ValueError(f"uv_mode={uv_mode} requires uv_vertices and face_uvs")
        uv_tri = uv_vertices[face_uvs[face_ids]]
        uv = (uv_tri * bary[:, :, None]).sum(axis=1)
        uv = np.mod(uv, 1.0)
        uv = np.clip(uv, 0.0, 1.0).astype(np.float32)
    elif uv_mode == "cylindrical":
        b1, b2 = make_perp_basis(axis)
        rel = roots - center[None, :]
        a = rel @ b1
        b = rel @ b2
        theta = np.arctan2(b, a)
        axis_coord = roots @ axis
        axis_lo, axis_hi = np.quantile(centers[candidates] @ axis, [0.002, 0.998])
        body_u_tmp = np.clip((axis_coord - axis_lo) / max(float(axis_hi - axis_lo), EPS), 0.0, 1.0)
        uv = np.stack([body_u_tmp, (theta + math.pi) / (2.0 * math.pi)], axis=1).astype(np.float32)
    else:
        raise ValueError(f"unsupported uv_mode: {uv_mode}")

    axis_coord = roots @ axis
    axis_lo, axis_hi = np.quantile(centers[candidates] @ axis, [0.002, 0.998])
    body_u = np.clip((axis_coord - axis_lo) / max(float(axis_hi - axis_lo), EPS), 0.0, 1.0)
    bbox_min = vertices.min(axis=0, keepdims=True)
    bbox_max = vertices.max(axis=0, keepdims=True)
    coord = np.clip((roots - bbox_min) / np.maximum(bbox_max - bbox_min, EPS), 0.0, 1.0)
    return {
        "roots": roots.astype(np.float32),
        "normals": chosen_normals.astype(np.float32),
        "tangents": tangents.astype(np.float32),
        "bitangents": bitangents.astype(np.float32),
        "uv": uv.astype(np.float32),
        "coord": coord.astype(np.float32),
        "body_u": body_u.astype(np.float32)[:, None],
        "face_ids": face_ids.astype(np.int64),
        "bary": bary.astype(np.float32),
        "axis": axis.astype(np.float32),
        "center": center.astype(np.float32),
    }


def load_compatible_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, object]:
    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: dict[str, str] = {}
    for key, value in state.items():
        if key not in current:
            skipped[key] = "missing_in_current_model"
            continue
        if tuple(current[key].shape) != tuple(value.shape):
            skipped[key] = f"shape {tuple(value.shape)} != {tuple(current[key].shape)}"
            continue
        compatible[key] = value
    missing = [key for key in current.keys() if key not in compatible]
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded_param_count": int(len(compatible)),
        "missing_param_count": int(len(missing)),
        "skipped_param_count": int(len(skipped)),
        "missing_params": missing[:32],
        "skipped_params": skipped,
    }


def load_obj_mesh_with_uv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_uvs: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                vals = line.split()[1:3]
                texcoords.append([float(vals[0]), float(vals[1])])
            elif line.startswith("f "):
                toks = line.split()[1:]
                v_idx: list[int] = []
                vt_idx: list[int] = []
                has_uv = True
                for tok in toks:
                    parts = tok.split("/")
                    v_idx.append(int(parts[0]) - 1)
                    if len(parts) >= 2 and parts[1] != "":
                        vt_idx.append(int(parts[1]) - 1)
                    else:
                        has_uv = False
                for i in range(1, len(v_idx) - 1):
                    faces.append([v_idx[0], v_idx[i], v_idx[i + 1]])
                    if has_uv:
                        face_uvs.append([vt_idx[0], vt_idx[i], vt_idx[i + 1]])
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if texcoords and len(face_uvs) == len(faces):
        uv = np.asarray(texcoords, dtype=np.float32)
        fuv = np.asarray(face_uvs, dtype=np.int64)
    else:
        uv = None
        fuv = None
    return v, f, uv, fuv


def load_or_build_xatlas_uv(
    mesh_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path)
        uv_vertices = cached["uv_vertices"].astype(np.float32)
        face_uvs = cached["face_uvs"].astype(np.int64)
        if face_uvs.shape == faces.shape:
            return uv_vertices, face_uvs
        raise RuntimeError(f"UV cache face count mismatch: {cache_path}")

    try:
        import xatlas  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "uv-mode=xatlas needs the Python xatlas package. Install it in the active environment with `pip install xatlas`."
        ) from exc

    vmapping, indices, uvs = xatlas.parametrize(vertices.astype(np.float32), faces.astype(np.uint32))
    face_uvs = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    uv_vertices = np.asarray(uvs, dtype=np.float32)
    if face_uvs.shape != faces.shape:
        raise RuntimeError(
            f"xatlas returned {face_uvs.shape[0]} faces for {faces.shape[0]} input faces on {mesh_path}"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        uv_vertices=uv_vertices,
        face_uvs=face_uvs,
        vmapping=np.asarray(vmapping, dtype=np.int64),
        mesh=str(mesh_path),
    )
    return uv_vertices, face_uvs


class TextureGroom(torch.nn.Module):
    def __init__(
        self,
        tex_h: int,
        tex_w: int,
        init_scale: float,
        init_trans: list[float],
        use_triplane: bool,
        triplane_h: int,
        triplane_w: int,
        triplane_scale: float,
        use_coarse_texture: bool,
        coarse_h: int,
        coarse_w: int,
        coarse_scale: float,
        use_head_atlas: bool,
        head_h: int,
        head_w: int,
        head_start: float,
        head_scale: float,
        use_surface_layer: bool,
    ) -> None:
        super().__init__()
        self.tex_h = tex_h
        self.tex_w = tex_w
        self.use_triplane = use_triplane
        self.triplane_scale = triplane_scale
        self.coarse_scale = coarse_scale
        self.head_start = head_start
        self.head_scale = head_scale
        init = torch.zeros(1, 20, tex_h, tex_w)
        init[:, 0] = 3.0  # coverage logit
        init[:, 1] = 2.2  # density logit
        init[:, 2] = -0.25  # length logit
        init[:, 3] = -5.5  # root width raw
        init[:, 4] = -6.8  # tip width raw
        init[:, 5] = 0.55  # flow x
        init[:, 6] = 0.04  # flow y
        init[:, 7] = -0.70  # lift
        init[:, 8] = -1.30  # sag
        init[:, 9] = -0.40  # bend
        init[:, 10] = 1.15  # stiffness logit
        init[:, 11:14] = torch.tensor([0.86, 0.84, 0.78])[None, :, None, None]
        init[:, 14:17] = torch.tensor([0.92, 0.90, 0.84])[None, :, None, None]
        init[:, 17] = -1.7  # darkness
        init[:, 18] = 0.0  # flow confidence
        init[:, 19] = 0.0  # backprojected local detail evidence
        # Low-amplitude deterministic stripe/noise breaks symmetry but remains trainable.
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, tex_h),
            torch.linspace(0, 1, tex_w),
            indexing="ij",
        )
        stripe = torch.sin(36.0 * xx + 7.0 * torch.sin(8.0 * yy))
        init[:, 17] += 0.45 * stripe[None]
        self.texture = torch.nn.Parameter(init)
        if use_triplane:
            self.triplanes = torch.nn.Parameter(torch.zeros(3, 20, triplane_h, triplane_w))
        else:
            self.register_parameter("triplanes", None)
        if use_coarse_texture:
            self.coarse_texture = torch.nn.Parameter(torch.zeros(1, 20, coarse_h, coarse_w))
        else:
            self.register_parameter("coarse_texture", None)
        if use_head_atlas:
            self.head_texture = torch.nn.Parameter(torch.zeros(1, 20, head_h, head_w))
        else:
            self.register_parameter("head_texture", None)
        if use_surface_layer:
            surface_init = torch.zeros(1, 4, tex_h, tex_w)
            surface_init[:, 0:3] = torch.logit(torch.tensor([0.92, 0.90, 0.84]).view(1, 3, 1, 1))
            surface_init[:, 3:4] = torch.logit(torch.tensor(0.90))
            self.surface_texture = torch.nn.Parameter(surface_init)
        else:
            self.register_parameter("surface_texture", None)
        self.log_scale = torch.nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))
        self.trans = torch.nn.Parameter(torch.tensor(init_trans, dtype=torch.float32))

    def sample(
        self,
        uv: torch.Tensor,
        coord: torch.Tensor | None = None,
        body_u: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        grid = uv.clone()
        grid = grid * 2.0 - 1.0
        grid = grid.view(1, -1, 1, 2)
        vals = F.grid_sample(
            self.texture,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, :, :, 0].T
        if self.coarse_texture is not None:
            coarse_vals = F.grid_sample(
                self.coarse_texture,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0, :, :, 0].T
            vals = vals + self.coarse_scale * coarse_vals
        if self.triplanes is not None and coord is not None:
            plane_uv = torch.stack(
                [
                    coord[:, [0, 1]],
                    coord[:, [1, 2]],
                    coord[:, [0, 2]],
                ],
                dim=0,
            )
            plane_grid = plane_uv * 2.0 - 1.0
            plane_grid = plane_grid.view(3, -1, 1, 2)
            plane_vals = F.grid_sample(
                self.triplanes,
                plane_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[:, :, :, 0].permute(0, 2, 1).mean(dim=0)
            vals = vals + self.triplane_scale * plane_vals
        if self.head_texture is not None:
            head_axis = body_u if body_u is not None else uv[:, 0:1]
            head_axis = head_axis.clamp(0.0, 1.0)
            head_u = ((head_axis - self.head_start) / max(1.0 - self.head_start, EPS)).clamp(0.0, 1.0)
            head_gate = torch.sigmoid(40.0 * (head_axis - self.head_start))
            head_grid = torch.cat([head_u, uv[:, 1:2]], dim=-1)
            head_grid = head_grid * 2.0 - 1.0
            head_grid = head_grid.view(1, -1, 1, 2)
            head_vals = F.grid_sample(
                self.head_texture,
                head_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[0, :, :, 0].T
            vals = vals + self.head_scale * head_gate * head_vals
        return {
            "coverage_logit": vals[:, [0]],
            "density_logit": vals[:, [1]],
            "length_logit": vals[:, [2]],
            "root_width_raw": vals[:, [3]],
            "tip_width_raw": vals[:, [4]],
            "flow_x": vals[:, [5]],
            "flow_y": vals[:, [6]],
            "lift": vals[:, [7]],
            "sag": vals[:, [8]],
            "bend": vals[:, [9]],
            "stiffness_logit": vals[:, [10]],
            "root_rgb": torch.sigmoid(vals[:, 11:14]),
            "tip_rgb": torch.sigmoid(vals[:, 14:17]),
            "darkness": vals[:, [17]],
            "detail_evidence": vals[:, [19]].detach().clamp(0.0, 1.0),
        }

    def sample_surface(self, uv: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.surface_texture is None:
            raise RuntimeError("surface layer was not enabled")
        grid = uv.clone()
        grid = grid * 2.0 - 1.0
        grid = grid.view(1, -1, 1, 2)
        vals = F.grid_sample(
            self.surface_texture,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, :, :, 0].T
        return {
            "surface_rgb": torch.sigmoid(vals[:, 0:3]),
            "surface_alpha": torch.sigmoid(vals[:, 3:4]),
        }

    @staticmethod
    def _active_tv_loss(tex: torch.Tensor, detail_tv_relax: float, detail_tv_min_weight: float) -> torch.Tensor:
        # Channels 18 and 19 are confidence/evidence diagnostics. They should
        # condition optimization, not be smoothed away as optimized appearance.
        active = tex[:, :18]
        dx = torch.abs(active[:, :, :, 1:] - active[:, :, :, :-1])
        dy = torch.abs(active[:, :, 1:, :] - active[:, :, :-1, :])
        if detail_tv_relax > 0.0 and tex.shape[1] > 19:
            detail = tex[:, 19:20].detach().clamp(0.0, 1.0)
            wx = 1.0 - float(detail_tv_relax) * 0.5 * (detail[:, :, :, 1:] + detail[:, :, :, :-1])
            wy = 1.0 - float(detail_tv_relax) * 0.5 * (detail[:, :, 1:, :] + detail[:, :, :-1, :])
            wx = wx.clamp(float(detail_tv_min_weight), 1.0)
            wy = wy.clamp(float(detail_tv_min_weight), 1.0)
            dx = dx * wx
            dy = dy * wy
        return dx.mean() + dy.mean()

    def tv_loss(self, detail_tv_relax: float = 0.0, detail_tv_min_weight: float = 0.35) -> torch.Tensor:
        tex = self.texture
        loss = self._active_tv_loss(tex, detail_tv_relax, detail_tv_min_weight)
        if self.triplanes is not None:
            tri = self.triplanes
            loss = loss + 0.5 * self._active_tv_loss(tri, detail_tv_relax, detail_tv_min_weight)
        if self.coarse_texture is not None:
            coarse = self.coarse_texture
            loss = loss + 0.25 * self._active_tv_loss(coarse, detail_tv_relax, detail_tv_min_weight)
        if self.head_texture is not None:
            head = self.head_texture
            loss = loss + 0.5 * self._active_tv_loss(head, detail_tv_relax, detail_tv_min_weight)
        if self.surface_texture is not None:
            surf = self.surface_texture
            sx = torch.abs(surf[:, :, :, 1:] - surf[:, :, :, :-1]).mean()
            sy = torch.abs(surf[:, :, 1:, :] - surf[:, :, :-1, :]).mean()
            loss = loss + 0.5 * (sx + sy)
        return loss

    @staticmethod
    def _asset_parameter_tv_loss(
        tex: torch.Tensor,
        detail_tv_relax: float,
        detail_tv_min_weight: float,
    ) -> torch.Tensor:
        # Smooth only the groom-shape parameters that should describe a usable
        # asset. Keep color and stripe/detail channels free enough for tiger
        # appearance, and keep diagnostic channels out of the loss.
        channel_weights = tex.new_zeros((tex.shape[1],))
        channel_weights[0] = 0.20  # coverage
        channel_weights[1] = 0.65  # density
        channel_weights[2] = 1.50  # length
        channel_weights[3] = 1.25  # root width
        channel_weights[4] = 1.25  # tip width
        channel_weights[5] = 2.00  # flow x
        channel_weights[6] = 2.00  # flow y
        channel_weights[7] = 1.10  # lift
        channel_weights[8] = 1.10  # sag
        channel_weights[9] = 1.10  # bend
        channel_weights[10] = 0.85  # stiffness
        weights = channel_weights.view(1, -1, 1, 1)

        dx = torch.abs(tex[:, :, :, 1:] - tex[:, :, :, :-1]) * weights
        dy = torch.abs(tex[:, :, 1:, :] - tex[:, :, :-1, :]) * weights
        if detail_tv_relax > 0.0 and tex.shape[1] > 19:
            detail = tex[:, 19:20].detach().clamp(0.0, 1.0)
            wx = 1.0 - float(detail_tv_relax) * 0.5 * (detail[:, :, :, 1:] + detail[:, :, :, :-1])
            wy = 1.0 - float(detail_tv_relax) * 0.5 * (detail[:, :, 1:, :] + detail[:, :, :-1, :])
            dx = dx * wx.clamp(float(detail_tv_min_weight), 1.0)
            dy = dy * wy.clamp(float(detail_tv_min_weight), 1.0)
        active = channel_weights.sum().clamp_min(1.0)
        return (dx.mean() + dy.mean()) * (tex.shape[1] / active)

    def asset_parameter_smooth_loss(
        self,
        detail_tv_relax: float = 0.0,
        detail_tv_min_weight: float = 0.35,
    ) -> torch.Tensor:
        loss = self._asset_parameter_tv_loss(self.texture, detail_tv_relax, detail_tv_min_weight)
        if self.coarse_texture is not None:
            loss = loss + 0.5 * self._asset_parameter_tv_loss(
                self.coarse_texture,
                detail_tv_relax,
                detail_tv_min_weight,
            )
        if self.head_texture is not None:
            loss = loss + 0.75 * self._asset_parameter_tv_loss(
                self.head_texture,
                detail_tv_relax,
                detail_tv_min_weight,
            )
        return loss

    @staticmethod
    def _flow_coherence_for_texture(
        tex: torch.Tensor,
        detail_relax: float,
        detail_min_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if tex.shape[1] <= 6:
            zero = tex.new_tensor(0.0)
            return zero, {
                "flow_coherence_dx": 0.0,
                "flow_coherence_dy": 0.0,
                "flow_coherence_norm_mean": 0.0,
            }
        flow = tex[:, 5:7]
        norm = torch.linalg.norm(flow, dim=1, keepdim=True).clamp_min(1e-6)
        direction = flow / norm
        dot_x = (direction[:, :, :, 1:] * direction[:, :, :, :-1]).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        dot_y = (direction[:, :, 1:, :] * direction[:, :, :-1, :]).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        loss_x = 1.0 - dot_x
        loss_y = 1.0 - dot_y

        wx = torch.ones_like(loss_x)
        wy = torch.ones_like(loss_y)
        if tex.shape[1] > 19 and detail_relax > 0.0:
            detail = tex[:, 19:20].detach().clamp(0.0, 1.0)
            wx = wx * (1.0 - float(detail_relax) * 0.5 * (detail[:, :, :, 1:] + detail[:, :, :, :-1]))
            wy = wy * (1.0 - float(detail_relax) * 0.5 * (detail[:, :, 1:, :] + detail[:, :, :-1, :]))
            wx = wx.clamp(float(detail_min_weight), 1.0)
            wy = wy.clamp(float(detail_min_weight), 1.0)
        if tex.shape[1] > 18:
            conf = tex[:, 18:19].detach().clamp(0.0, 1.0)
            wx = wx * (0.25 + 0.75 * 0.5 * (conf[:, :, :, 1:] + conf[:, :, :, :-1]))
            wy = wy * (0.25 + 0.75 * 0.5 * (conf[:, :, 1:, :] + conf[:, :, :-1, :]))

        loss_x_term = (loss_x * wx).sum() / wx.sum().clamp_min(1.0)
        loss_y_term = (loss_y * wy).sum() / wy.sum().clamp_min(1.0)
        loss = loss_x_term + loss_y_term
        return loss, {
            "flow_coherence_dx": float(loss_x_term.detach().cpu()),
            "flow_coherence_dy": float(loss_y_term.detach().cpu()),
            "flow_coherence_norm_mean": float(norm.detach().mean().cpu()),
        }

    def flow_coherence_loss(
        self,
        detail_relax: float = 0.75,
        detail_min_weight: float = 0.25,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss, stats = self._flow_coherence_for_texture(self.texture, detail_relax, detail_min_weight)
        main_loss = loss
        if self.coarse_texture is not None:
            coarse_loss, coarse_stats = self._flow_coherence_for_texture(
                self.coarse_texture,
                detail_relax,
                detail_min_weight,
            )
            loss = loss + 0.5 * coarse_loss
            stats["flow_coherence_coarse"] = float(coarse_loss.detach().cpu())
            stats["flow_coherence_coarse_norm_mean"] = coarse_stats["flow_coherence_norm_mean"]
        if self.head_texture is not None:
            head_loss, head_stats = self._flow_coherence_for_texture(
                self.head_texture,
                detail_relax,
                detail_min_weight,
            )
            loss = loss + 0.5 * head_loss
            stats["flow_coherence_head"] = float(head_loss.detach().cpu())
            stats["flow_coherence_head_norm_mean"] = head_stats["flow_coherence_norm_mean"]
        stats["flow_coherence_main"] = float(main_loss.detach().cpu())
        stats["flow_coherence_loss"] = float(loss.detach().cpu())
        return loss, stats


def training_image_mask_paths(data_root: Path) -> tuple[list[Path], list[Path]]:
    image_paths = sorted((data_root / "images").glob("*.png"))
    mask_paths = sorted((data_root / "silhouette").glob("*.png"))
    if len(image_paths) != len(mask_paths) or not image_paths:
        raise RuntimeError(f"Bad image/mask count: {len(image_paths)} images, {len(mask_paths)} masks")
    return image_paths, mask_paths


def load_training_images(data_root: Path, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image_paths, mask_paths = training_image_mask_paths(data_root)
    images = []
    masks = []
    for ip, mp in zip(image_paths, mask_paths):
        im = Image.open(ip).convert("RGB")
        scale = width / im.size[0]
        size = (width, int(round(im.size[1] * scale)))
        im = im.resize(size, Image.Resampling.BILINEAR)
        ms = Image.open(mp).convert("L").resize(size, Image.Resampling.BILINEAR)
        images.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0))
        masks.append(torch.from_numpy(np.asarray(ms, dtype=np.float32) / 255.0)[..., None])
    return torch.stack(images).to(device), torch.stack(masks).to(device)


def load_projection_mats(data_root: Path, width: int, device: torch.device) -> torch.Tensor:
    p = np.load(data_root / "cameras.npz")["arr_0"].astype(np.float32)
    p = torch.from_numpy(p).to(device)
    sx = width / 1920.0
    sy = (width * 1080.0 / 1920.0) / 1080.0
    p = p.clone()
    p[:, 0, :] *= sx
    p[:, 1, :] *= sy
    return p


def make_detail_weights(
    images: torch.Tensor,
    masks: torch.Tensor,
    edge_weight: float,
    dark_weight: float,
) -> torch.Tensor:
    luma = (
        0.2126 * images[..., 0:1]
        + 0.7152 * images[..., 1:2]
        + 0.0722 * images[..., 2:3]
    )
    src = luma.permute(0, 3, 1, 2)
    blur = F.avg_pool2d(src, kernel_size=7, stride=1, padding=3).permute(0, 2, 3, 1)
    edge = torch.abs(luma - blur)
    edge = edge / torch.clamp(edge.flatten(1).mean(dim=1)[:, None, None, None], min=1e-4)
    dark = torch.clamp(0.72 - luma, min=0.0) / 0.72
    weights = 1.0 + edge_weight * edge.clamp(0.0, 4.0) + dark_weight * dark.clamp(0.0, 1.0)
    return weights * masks


def build_mask_boundary_weights(masks: torch.Tensor, dilation: int) -> torch.Tensor:
    kernel = max(3, int(dilation))
    if kernel % 2 == 0:
        kernel += 1
    x = masks.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    maxp = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    minp = -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=kernel // 2)
    band = (maxp - minp).clamp(0.0, 1.0)
    band = gaussian_blur2d(band, sigma=max(1.0, float(kernel) / 6.0)).clamp(0.0, 1.0)
    return band.permute(0, 2, 3, 1).detach()


@torch.no_grad()
def compute_root_boundary_evidence(
    roots: torch.Tensor,
    normals: torch.Tensor,
    boundary_weights: torch.Tensor,
    masks: torch.Tensor,
    pmats: torch.Tensor,
    cam_centers: torch.Tensor,
    scale: torch.Tensor,
    trans: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    points = roots * scale + trans.view(1, 3)
    acc = torch.zeros(roots.shape[0], 1, device=roots.device, dtype=roots.dtype)
    weight = torch.zeros_like(acc)
    h, w = masks.shape[1:3]
    for view in range(boundary_weights.shape[0]):
        xy, z = project_points(points, pmats[view])
        inb = (z > 1e-5) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 2) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 2)
        if not inb.any():
            continue
        view_dir = torch_normalize(cam_centers[view].view(1, 3) - points)
        front = torch.sigmoid(24.0 * ((normals * view_dir).sum(dim=-1, keepdim=True) - 0.02))
        sampled_mask = bilinear_sample_image(masks[view], xy)[:, :1]
        sampled_boundary = bilinear_sample_image(boundary_weights[view], xy)[:, :1].clamp(0.0, 1.0)
        conf = sampled_mask * front * inb[:, None].float()
        acc += sampled_boundary * conf
        weight += conf
    evidence = acc / torch.clamp(weight, min=1e-6)
    evidence = torch.where(weight > 1e-5, evidence, torch.zeros_like(evidence)).clamp(0.0, 1.0).detach()
    stats = {
        "root_boundary_evidence_count": int(evidence.numel()),
        "root_boundary_evidence_known_fraction": float((weight[:, 0] > 1e-5).float().mean().cpu()),
        "root_boundary_evidence_mean": float(evidence.mean().cpu()),
        "root_boundary_evidence_p50": float(torch.quantile(evidence[:, 0], 0.50).cpu()),
        "root_boundary_evidence_p90": float(torch.quantile(evidence[:, 0], 0.90).cpu()),
        "root_boundary_evidence_p95": float(torch.quantile(evidence[:, 0], 0.95).cpu()),
    }
    return evidence, stats


def sobel_gradient_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred_luma = 0.2126 * pred[..., 0:1] + 0.7152 * pred[..., 1:2] + 0.0722 * pred[..., 2:3]
    target_luma = 0.2126 * target[..., 0:1] + 0.7152 * target[..., 1:2] + 0.0722 * target[..., 2:3]
    src = torch.cat([pred_luma, target_luma], dim=-1).permute(2, 0, 1).unsqueeze(0)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(src, kx.expand(2, -1, -1, -1), padding=1, groups=2)[0].permute(1, 2, 0)
    gy = F.conv2d(src, ky.expand(2, -1, -1, -1), padding=1, groups=2)[0].permute(1, 2, 0)
    diff = torch.abs(gx[..., 0:1] - gx[..., 1:2]) + torch.abs(gy[..., 0:1] - gy[..., 1:2])
    edge_mask = F.max_pool2d(mask.permute(2, 0, 1).unsqueeze(0), kernel_size=5, stride=1, padding=2)[0].permute(1, 2, 0)
    return (diff * edge_mask).sum() / torch.clamp(edge_mask.sum() * 2.0, min=1.0)


@torch.no_grad()
def gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(float(sigma) * 3.0)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / float(sigma)).square())
    kernel = kernel / kernel.sum().clamp_min(EPS)
    return kernel


@torch.no_grad()
def gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel1d(sigma, x.device, x.dtype)
    groups = x.shape[1]
    kx = kernel.view(1, 1, 1, -1).expand(groups, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(groups, 1, -1, 1)
    pad = kernel.numel() // 2
    x = F.conv2d(x, kx, padding=(0, pad), groups=groups)
    x = F.conv2d(x, ky, padding=(pad, 0), groups=groups)
    return x


@torch.no_grad()
def build_gabor_kernels(
    bins: int,
    sigma_x: float,
    sigma_y: float,
    frequency: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = max(3, int(math.ceil(max(float(sigma_x), float(sigma_y)) * 4.0)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    angles = torch.linspace(0.0, math.pi * (bins - 1) / bins, bins, device=device, dtype=dtype)
    kernels = []
    for theta in angles:
        c = torch.cos(math.pi - theta)
        s = torch.sin(math.pi - theta)
        x_theta = xx * c + yy * s
        y_theta = -xx * s + yy * c
        envelope = torch.exp(
            -0.5 * ((x_theta / float(sigma_x)).square() + (y_theta / float(sigma_y)).square())
        )
        carrier = torch.cos(2.0 * math.pi * float(frequency) * x_theta)
        kernel = envelope * carrier
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.abs().sum().clamp_min(EPS)
        kernels.append(kernel)
    weight = torch.stack(kernels, dim=0)[:, None]
    return weight, angles


@torch.no_grad()
def build_gabor_orientation_fields(
    images: torch.Tensor,
    masks: torch.Tensor,
    bins: int,
    dog_low: float,
    dog_high: float,
    sigma_x: float,
    sigma_y: float,
    frequency: float,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    gray = 0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2]
    alpha = masks[..., 0].clamp(0.0, 1.0)
    inner_alpha = 1.0 - F.max_pool2d(
        (1.0 - alpha[:, None]).clamp(0.0, 1.0),
        kernel_size=5,
        stride=1,
        padding=2,
    )[:, 0]
    alpha = alpha * inner_alpha.clamp(0.0, 1.0)

    src = gray[:, None]
    dog = gaussian_blur2d(src, dog_low) - gaussian_blur2d(src, dog_high)
    kernels, angles = build_gabor_kernels(
        bins,
        sigma_x,
        sigma_y,
        frequency,
        images.device,
        images.dtype,
    )
    padding = kernels.shape[-1] // 2

    orientation_chunks = []
    confidence_chunks = []
    for start in range(0, dog.shape[0], max(1, chunk)):
        end = min(start + max(1, chunk), dog.shape[0])
        response = F.conv2d(dog[start:end], kernels, padding=padding).abs()
        max_response, idx = response.max(dim=1)
        response_sum = response.sum(dim=1).clamp_min(EPS)
        peak = max_response / response_sum
        uniform = 1.0 / float(bins)
        confidence = ((peak - uniform) / max(1.0 - uniform, EPS)).clamp(0.0, 1.0)
        confidence = confidence * alpha[start:end]
        angle = angles[idx]
        orientation = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        orientation_chunks.append(orientation)
        confidence_chunks.append(confidence[..., None])

    orientation = torch.cat(orientation_chunks, dim=0)
    confidence = torch.cat(confidence_chunks, dim=0)
    stats = {
        "source": "gabor",
        "bins": int(bins),
        "dog_low": float(dog_low),
        "dog_high": float(dog_high),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "frequency": float(frequency),
        "confidence_mean": float(confidence.detach().mean().cpu()),
        "confidence_max": float(confidence.detach().max().cpu()),
        "confidence_nonzero_pixels": int((confidence.detach() > 1e-5).sum().cpu()),
    }
    return orientation.detach(), confidence.detach(), stats


def _load_np_first_array(path: Path) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        keys = sorted(loaded.files)
        if not keys:
            raise RuntimeError(f"empty npz orientation file: {path}")
        return np.asarray(loaded[keys[0]])
    return np.asarray(loaded)


def _resize_scalar_array(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] in (1, 2, 3, 4) and arr.shape[-1] not in (1, 2, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected scalar orientation/confidence map, got shape {arr.shape}")
    tensor = torch.from_numpy(arr.astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=(size[1], size[0]), mode="bilinear", align_corners=False)[0, 0]
    return resized.numpy()


def _load_scalar_map(path: Path, size: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() in NUMPY_EXTENSIONS:
        return _resize_scalar_array(_load_np_first_array(path), size)
    image = Image.open(path)
    if image.mode not in ("L", "I", "F"):
        image = image.convert("L")
    image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def _find_file_by_stem(directory: Path, stem: str, extensions: tuple[str, ...]) -> Path | None:
    for ext in extensions:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
    matches = sorted(p for p in directory.iterdir() if p.is_file() and p.stem == stem)
    for path in matches:
        if path.suffix.lower() in extensions:
            return path
    return None


def _resolve_orientation_dirs(data_root: Path, orientation_dir: Path) -> tuple[Path, Path | None, str]:
    base = orientation_dir if orientation_dir.is_absolute() else data_root / orientation_dir
    if not base.exists():
        raise FileNotFoundError(f"orientation directory does not exist: {base}")

    if (base / "angles").is_dir():
        angle_dir = base / "angles"
        conf_dir = base / "vars" if (base / "vars").is_dir() else None
        conf_kind = "variance"
    elif base.name == "angles":
        angle_dir = base
        conf_dir = base.parent / "vars" if (base.parent / "vars").is_dir() else None
        conf_kind = "variance"
    elif (base / "orientation_maps").is_dir():
        angle_dir = base / "orientation_maps"
        conf_dir = base / "confidence_maps" if (base / "confidence_maps").is_dir() else None
        conf_kind = "confidence"
    else:
        angle_dir = base
        if (base.parent / "vars").is_dir():
            conf_dir = base.parent / "vars"
            conf_kind = "variance"
        elif (base.parent / "confidence_maps").is_dir():
            conf_dir = base.parent / "confidence_maps"
            conf_kind = "confidence"
        else:
            conf_dir = None
            conf_kind = "none"

    if not angle_dir.is_dir():
        raise FileNotFoundError(f"orientation angle directory does not exist: {angle_dir}")
    return angle_dir, conf_dir, conf_kind


@torch.no_grad()
def load_neuralfur_orientation_fields(
    data_root: Path,
    orientation_dir: Path,
    width: int,
    masks: torch.Tensor,
    angle_bins: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    image_paths, _ = training_image_mask_paths(data_root)
    angle_dir, conf_dir, conf_kind = _resolve_orientation_dirs(data_root, orientation_dir)
    height = int(masks.shape[1])
    image_size = (int(width), height)

    orientations = []
    confidences = []
    missing_angles = []
    missing_conf = []
    for image_path in image_paths:
        stem = image_path.stem
        angle_path = _find_file_by_stem(angle_dir, stem, ORIENTATION_EXTENSIONS)
        if angle_path is None:
            missing_angles.append(stem)
            continue
        raw_angle = _load_scalar_map(angle_path, image_size)
        if raw_angle.max() <= 1.0 and raw_angle.min() >= 0.0:
            angle = raw_angle * math.pi
            angle_scale = "unit_pi"
        elif raw_angle.max() <= float(angle_bins) + 1.0:
            angle = raw_angle / max(float(angle_bins), 1.0) * math.pi
            angle_scale = f"bins_{angle_bins}"
        else:
            angle = raw_angle / 255.0 * math.pi
            angle_scale = "uint8_pi"
        orientation = np.stack([np.cos(angle), np.sin(angle)], axis=-1).astype(np.float32)

        confidence = None
        if conf_dir is not None:
            conf_path = _find_file_by_stem(conf_dir, stem, ORIENTATION_EXTENSIONS)
            if conf_path is None:
                missing_conf.append(stem)
            else:
                conf_raw = _load_scalar_map(conf_path, image_size).astype(np.float32)
                if conf_kind == "variance":
                    var = np.maximum(conf_raw / (math.pi**2), 0.0)
                    confidence = 1.0 / (var * var + 1e-7)
                    finite = np.isfinite(confidence)
                    if finite.any():
                        norm = max(float(np.quantile(confidence[finite], 0.95)), 1e-6)
                        confidence = np.clip(confidence / norm, 0.0, 1.0)
                    else:
                        confidence = np.zeros_like(conf_raw, dtype=np.float32)
                else:
                    confidence = conf_raw
                    if confidence.max() > 1.5:
                        confidence = confidence / 255.0
                    confidence = np.clip(confidence, 0.0, 1.0)
        if confidence is None:
            confidence = np.ones((height, int(width)), dtype=np.float32)
        orientations.append(torch.from_numpy(orientation))
        confidences.append(torch.from_numpy(confidence[..., None].astype(np.float32)))

    if missing_angles:
        preview = ", ".join(missing_angles[:5])
        raise FileNotFoundError(
            f"missing orientation angle maps for {len(missing_angles)} images in {angle_dir}; first missing: {preview}"
        )
    if missing_conf and conf_dir is not None:
        preview = ", ".join(missing_conf[:5])
        raise FileNotFoundError(
            f"missing orientation confidence/variance maps for {len(missing_conf)} images in {conf_dir}; first missing: {preview}"
        )

    orientation = torch.stack(orientations).to(device=masks.device, dtype=masks.dtype)
    confidence = torch.stack(confidences).to(device=masks.device, dtype=masks.dtype)
    confidence = confidence * masks.clamp(0.0, 1.0)
    stats = {
        "source": "map",
        "format": "neuralfur_angle",
        "orientation_dir": str(orientation_dir),
        "angle_dir": str(angle_dir),
        "confidence_dir": str(conf_dir) if conf_dir is not None else "",
        "confidence_kind": conf_kind,
        "angle_bins": int(angle_bins),
        "angle_scale": angle_scale if orientations else "",
        "map_count": int(len(orientations)),
        "confidence_mean": float(confidence.detach().mean().cpu()),
        "confidence_max": float(confidence.detach().max().cpu()),
        "confidence_nonzero_pixels": int((confidence.detach() > 1e-5).sum().cpu()),
    }
    return orientation.detach(), confidence.detach(), stats


@torch.no_grad()
def build_target_orientation_fields(
    images: torch.Tensor,
    masks: torch.Tensor,
    source: str,
    data_root: Path,
    orientation_dir: Path,
    width: int,
    orientation_angle_bins: int,
    gabor_bins: int,
    gabor_dog_low: float,
    gabor_dog_high: float,
    gabor_sigma_x: float,
    gabor_sigma_y: float,
    gabor_frequency: float,
    gabor_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    if source == "map":
        return load_neuralfur_orientation_fields(
            data_root,
            orientation_dir,
            width,
            masks,
            orientation_angle_bins,
        )

    if source == "gabor":
        return build_gabor_orientation_fields(
            images,
            masks,
            gabor_bins,
            gabor_dog_low,
            gabor_dog_high,
            gabor_sigma_x,
            gabor_sigma_y,
            gabor_frequency,
            gabor_chunk,
        )

    if source == "rgb":
        gray = 0.2126 * images[..., 0] + 0.7152 * images[..., 1] + 0.0722 * images[..., 2]
        confidence_alpha = masks[..., 0].clamp(0.0, 1.0)
        inner_alpha = 1.0 - F.max_pool2d(
            (1.0 - confidence_alpha[:, None]).clamp(0.0, 1.0),
            kernel_size=5,
            stride=1,
            padding=2,
        )[:, 0]
        confidence_alpha = confidence_alpha * inner_alpha.clamp(0.0, 1.0)
    elif source == "alpha":
        gray = masks[..., 0].clamp(0.0, 1.0)
        confidence_alpha = torch.ones_like(gray)
    else:
        raise ValueError(f"unsupported flow orientation source: {source}")

    src = gray[:, None]
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    grad_x = F.conv2d(src, kx, padding=1)
    grad_y = F.conv2d(src, ky, padding=1)
    confidence = torch.sqrt(grad_x.square() + grad_y.square())[:, 0, :, :, None]
    confidence = confidence * confidence_alpha[:, :, :, None]
    tangent = torch.cat([-grad_y, grad_x], dim=1).permute(0, 2, 3, 1)
    tangent = torch_normalize(tangent)
    stats = {
        "source": source,
        "confidence_mean": float(confidence.detach().mean().cpu()),
        "confidence_max": float(confidence.detach().max().cpu()),
        "confidence_nonzero_pixels": int((confidence.detach() > 1e-5).sum().cpu()),
    }
    return tangent.detach(), confidence.detach(), stats


def flow_orientation_loss_for_view(
    strand_points: torch.Tensor,
    pmat: torch.Tensor,
    target_orientation: torch.Tensor,
    target_confidence: torch.Tensor,
    width: int,
    height: int,
    segment_ids: torch.Tensor | None,
    min_confidence: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if strand_points.shape[1] < 2:
        zero = strand_points.new_tensor(0.0)
        return zero, {
            "flow_orient_loss": 0.0,
            "flow_orient_weight_sum": 0.0,
            "flow_orient_valid_samples": 0,
            "flow_orient_mean_confidence": 0.0,
        }

    segment_count = strand_points.shape[1] - 1
    total_segments = strand_points.shape[0] * segment_count
    if segment_ids is not None:
        ids = segment_ids[(segment_ids >= 0) & (segment_ids < total_segments)]
        if ids.numel() == 0:
            zero = strand_points.new_tensor(0.0)
            return zero, {
                "flow_orient_loss": 0.0,
                "flow_orient_weight_sum": 0.0,
                "flow_orient_valid_samples": 0,
                "flow_orient_mean_confidence": 0.0,
            }
        root_ids = torch.div(ids, segment_count, rounding_mode="floor")
        local_ids = ids - root_ids * segment_count
        p0 = strand_points[root_ids, local_ids]
        p1 = strand_points[root_ids, local_ids + 1]
    else:
        p0 = strand_points[:, :-1, :].reshape(-1, 3)
        p1 = strand_points[:, 1:, :].reshape(-1, 3)

    xy0, z0 = project_points(p0, pmat)
    xy1, z1 = project_points(p1, pmat)
    mid = 0.5 * (xy0 + xy1)
    screen_vec = xy1 - xy0
    screen_len = torch.linalg.norm(screen_vec, dim=-1)
    screen_dir = screen_vec / torch.clamp(screen_len[:, None], min=1e-6)

    sampled_orientation = bilinear_sample_image(target_orientation, mid)
    sampled_confidence = bilinear_sample_image(target_confidence, mid)[:, 0]
    sampled_orientation = torch_normalize(sampled_orientation)

    valid = (
        (z0 > 1e-5)
        & (z1 > 1e-5)
        & (mid[:, 0] >= 0.0)
        & (mid[:, 0] <= width - 1)
        & (mid[:, 1] >= 0.0)
        & (mid[:, 1] <= height - 1)
        & (screen_len > 0.25)
        & (sampled_confidence >= min_confidence)
    )
    weight = sampled_confidence * valid.to(sampled_confidence.dtype)
    dot = (screen_dir * sampled_orientation).sum(dim=-1).clamp(-1.0, 1.0)
    orient_error = 1.0 - dot.square()
    denom = weight.sum().clamp_min(1.0)
    loss = (orient_error * weight).sum() / denom
    stats = {
        "flow_orient_loss": float(loss.detach().cpu()),
        "flow_orient_weight_sum": float(weight.detach().sum().cpu()),
        "flow_orient_valid_samples": int((weight.detach() > 0.0).sum().cpu()),
        "flow_orient_mean_confidence": float(sampled_confidence.detach()[valid.detach()].mean().cpu())
        if bool(valid.detach().any())
        else 0.0,
    }
    return loss, stats


def flow_hint_prior_loss(
    params: dict[str, torch.Tensor],
    flow_hint: torch.Tensor | None,
    flow_hint_confidence: torch.Tensor | None,
    min_confidence: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    pred = torch.cat([params["flow_x"], params["flow_y"]], dim=-1)
    zero = pred.new_tensor(0.0)
    if flow_hint is None or flow_hint_confidence is None:
        return zero, {
            "flow_hint_prior_loss": 0.0,
            "flow_hint_prior_direction_loss": 0.0,
            "flow_hint_prior_value_loss": 0.0,
            "flow_hint_prior_valid_roots": 0,
            "flow_hint_prior_mean_confidence": 0.0,
            "flow_hint_prior_pred_norm": 0.0,
            "flow_hint_prior_target_norm": 0.0,
        }

    target = flow_hint.to(device=pred.device, dtype=pred.dtype)
    confidence = flow_hint_confidence.to(device=pred.device, dtype=pred.dtype).reshape(-1)
    valid = (
        torch.isfinite(pred).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & (confidence >= float(min_confidence))
        & (torch.linalg.norm(target, dim=-1) > 1e-5)
    )
    if not bool(valid.detach().any()):
        return zero, {
            "flow_hint_prior_loss": 0.0,
            "flow_hint_prior_direction_loss": 0.0,
            "flow_hint_prior_value_loss": 0.0,
            "flow_hint_prior_valid_roots": 0,
            "flow_hint_prior_mean_confidence": 0.0,
            "flow_hint_prior_pred_norm": 0.0,
            "flow_hint_prior_target_norm": 0.0,
        }

    pred_valid = pred[valid]
    target_valid = target[valid]
    weight = confidence[valid].clamp(0.0, 1.0)
    pred_dir = torch_normalize(pred_valid)
    target_dir = torch_normalize(target_valid)
    dot = (pred_dir * target_dir).sum(dim=-1).clamp(-1.0, 1.0)
    direction_loss = 1.0 - dot.square()
    value_loss = F.smooth_l1_loss(pred_valid, target_valid, reduction="none").sum(dim=-1)
    denom = weight.sum().clamp_min(1.0)
    direction_term = (direction_loss * weight).sum() / denom
    value_term = (value_loss * weight).sum() / denom
    loss = direction_term
    stats = {
        "flow_hint_prior_loss": float(loss.detach().cpu()),
        "flow_hint_prior_direction_loss": float(direction_term.detach().cpu()),
        "flow_hint_prior_value_loss": float(value_term.detach().cpu()),
        "flow_hint_prior_valid_roots": int(valid.detach().sum().cpu()),
        "flow_hint_prior_mean_confidence": float(weight.detach().mean().cpu()),
        "flow_hint_prior_pred_norm": float(torch.linalg.norm(pred_valid.detach(), dim=-1).mean().cpu()),
        "flow_hint_prior_target_norm": float(torch.linalg.norm(target_valid.detach(), dim=-1).mean().cpu()),
    }
    return loss, stats


@torch.no_grad()
def compute_root_flow_hints_from_orientation(
    roots: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    target_orientation: torch.Tensor,
    target_confidence: torch.Tensor,
    masks: torch.Tensor,
    pmats: torch.Tensor,
    cam_centers: torch.Tensor,
    init_scale: torch.Tensor,
    init_trans: torch.Tensor,
    min_confidence: float,
    flow_scale: float,
    probe_length: float,
    default_flow: tuple[float, float] = (0.55, 0.04),
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    points = roots * init_scale + init_trans.view(1, 3)
    tangent_probe = tangents * (init_scale * float(probe_length))
    bitangent_probe = bitangents * (init_scale * float(probe_length))
    flow_acc = torch.zeros(roots.shape[0], 2, device=roots.device, dtype=roots.dtype)
    weight_acc = torch.zeros(roots.shape[0], 1, device=roots.device, dtype=roots.dtype)
    default = torch.tensor(default_flow, device=roots.device, dtype=roots.dtype).view(1, 2)
    height, width = masks.shape[1:3]

    for view in range(target_orientation.shape[0]):
        xy, z = project_points(points, pmats[view])
        xy_t, z_t = project_points(points + tangent_probe, pmats[view])
        xy_b, z_b = project_points(points + bitangent_probe, pmats[view])
        inb = (
            (z > 1e-5)
            & (z_t > 1e-5)
            & (z_b > 1e-5)
            & (xy[:, 0] >= 0.0)
            & (xy[:, 0] <= width - 2)
            & (xy[:, 1] >= 0.0)
            & (xy[:, 1] <= height - 2)
        )
        if not bool(inb.any()):
            continue

        st = xy_t - xy
        sb = xy_b - xy
        det = st[:, 0] * sb[:, 1] - st[:, 1] * sb[:, 0]
        basis_ok = (
            det.abs() > 1e-4
        ) & (torch.linalg.norm(st, dim=-1) > 0.05) & (torch.linalg.norm(sb, dim=-1) > 0.05)

        orient = bilinear_sample_image(target_orientation[view], xy)
        orient = torch_normalize(orient)
        conf = bilinear_sample_image(target_confidence[view], xy)[:, 0]
        sampled_mask = bilinear_sample_image(masks[view], xy)[:, 0]
        view_dir = torch_normalize(cam_centers[view][None, :] - points)
        front = torch.sigmoid(24.0 * ((normals * view_dir).sum(dim=-1) - 0.02))
        valid = inb & basis_ok & (conf >= min_confidence) & (sampled_mask > 0.85) & (front > 0.01)
        if not bool(valid.any()):
            continue

        det_safe = torch.where(det.abs() > 1e-6, det, torch.sign(det + 1e-12) * 1e-6)
        coeff_x = (orient[:, 0] * sb[:, 1] - orient[:, 1] * sb[:, 0]) / det_safe
        coeff_y = (-orient[:, 0] * st[:, 1] + orient[:, 1] * st[:, 0]) / det_safe
        coeff = torch.stack([coeff_x, coeff_y], dim=-1)
        coeff = torch_normalize(coeff)
        sign = torch.where((coeff * default).sum(dim=-1, keepdim=True) >= 0.0, 1.0, -1.0)
        coeff = coeff * sign * float(flow_scale)
        weight = (conf * sampled_mask * front * valid.to(conf.dtype)).view(-1, 1)
        flow_acc += coeff * weight
        weight_acc += weight

    known = weight_acc[:, 0] > 1e-5
    flow = default.expand_as(flow_acc).clone()
    flow[known] = flow_acc[known] / torch.clamp(weight_acc[known], min=1e-6)
    confidence = weight_acc / torch.clamp(weight_acc.max(), min=1e-6)
    stats = {
        "known_roots": int(known.detach().sum().cpu()),
        "known_fraction": float(known.detach().float().mean().cpu()),
        "weight_mean": float(weight_acc.detach().mean().cpu()),
        "weight_max": float(weight_acc.detach().max().cpu()),
        "flow_x_mean": float(flow[:, 0].detach().mean().cpu()),
        "flow_y_mean": float(flow[:, 1].detach().mean().cpu()),
        "flow_norm_mean": float(torch.linalg.norm(flow, dim=-1).detach().mean().cpu()),
    }
    return flow.detach(), confidence.detach().clamp(0.0, 1.0), stats


def load_camera_centers(data_root: Path, device: torch.device) -> torch.Tensor:
    extr = np.load(data_root / "cameras_extr.npy").astype(np.float32)
    centers = []
    for e in extr:
        r = e[:3, :3]
        t = e[:3, 3]
        centers.append(-(r.T @ t))
    return torch.tensor(np.stack(centers, axis=0), device=device)


def project_points(points: torch.Tensor, pmat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    homog = torch.cat([points, ones], dim=-1)
    clip = homog @ pmat.T
    z = clip[:, 2]
    xy = clip[:, :2] / torch.clamp(z[:, None], min=1e-6)
    return xy, z


def splat_render(
    points: torch.Tensor,
    normals: torch.Tensor,
    colors: torch.Tensor,
    alpha: torch.Tensor,
    pmat: torch.Tensor,
    cam_center: torch.Tensor,
    height: int,
    width: int,
    radius: float,
    alpha_cap: float,
    depth_band: float,
    depth_sharpness: float,
    tangent_vectors: torch.Tensor | None = None,
    tangent_radius_scale: float = 1.0,
    normal_radius_scale: float = 1.0,
    point_radius: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    xy, z = project_points(points, pmat)
    tangent_xy = None
    if tangent_vectors is not None:
        tangent_tip_xy, _ = project_points(points + tangent_vectors, pmat)
        tangent_xy = tangent_tip_xy - xy
    x = xy[:, 0]
    y = xy[:, 1]
    view_dir = torch_normalize(cam_center[None, :] - points)
    front = torch.sigmoid(24.0 * ((normals * view_dir).sum(dim=-1, keepdim=True) - 0.02))
    valid = (z > 1e-5) & (x >= 0) & (x <= width - 2) & (y >= 0) & (y <= height - 2) & (front[:, 0] > 0.01)
    if valid.sum() == 0:
        return torch.ones(height, width, 3, device=points.device), torch.zeros(height, width, 1, device=points.device)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    c = colors[valid]
    a = (alpha[valid] * front[valid]).clamp(0.0, alpha_cap)
    if point_radius is None:
        point_radius_valid = torch.full_like(x, float(radius))
    else:
        point_radius_valid = point_radius.reshape(-1)[valid].to(x.dtype).clamp(0.18, 4.0)
    if tangent_xy is not None:
        tangent_xy = tangent_xy[valid]
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    if tangent_xy is None:
        sigma2 = torch.clamp(point_radius_valid.square(), min=0.25)
        max_sigma = point_radius_valid.detach().max()
    else:
        tangent_sigma = torch.clamp(point_radius_valid * tangent_radius_scale, min=0.18)
        normal_sigma = torch.clamp(point_radius_valid * normal_radius_scale, min=0.18)
        max_sigma = torch.maximum(tangent_sigma, normal_sigma).detach().max()
    splat_extent = max(1, int(math.ceil(float(max_sigma.cpu()) * 2.0)))
    lin_chunks = []
    point_chunks = []
    weight_chunks = []
    for oy in range(-splat_extent, splat_extent + 1):
        for ox in range(-splat_extent, splat_extent + 1):
            xi = x0 + ox
            yi = y0 + oy
            keep = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            if keep.any():
                point_ids = torch.where(keep)[0]
                dx = x[keep] - xi[keep].float()
                dy = y[keep] - yi[keep].float()
                if tangent_xy is None:
                    wi = torch.exp(-0.5 * (dx * dx + dy * dy) / sigma2[keep])
                else:
                    t = tangent_xy[keep]
                    t_len = torch.linalg.norm(t, dim=-1, keepdim=True)
                    t_dir = torch.where(
                        t_len > 1e-4,
                        t / torch.clamp(t_len, min=1e-4),
                        torch.tensor([1.0, 0.0], device=points.device, dtype=points.dtype).view(1, 2),
                    )
                    n_dir = torch.stack([-t_dir[:, 1], t_dir[:, 0]], dim=-1)
                    offset = torch.stack([dx, dy], dim=-1)
                    along = (offset * t_dir).sum(dim=-1)
                    across = (offset * n_dir).sum(dim=-1)
                    wi = torch.exp(
                        -0.5
                        * (
                            torch.square(along / tangent_sigma[keep])
                            + torch.square(across / normal_sigma[keep])
                        )
                    )
                lin_chunks.append(yi[keep] * width + xi[keep])
                point_chunks.append(point_ids)
                weight_chunks.append(wi)
    if not lin_chunks:
        return torch.ones(height, width, 3, device=points.device), torch.zeros(height, width, 1, device=points.device)
    lin_all = torch.cat(lin_chunks, dim=0)
    point_all = torch.cat(point_chunks, dim=0)
    weight_all = torch.cat(weight_chunks, dim=0)
    weight_sum = torch.zeros_like(x).scatter_add_(0, point_all, weight_all).clamp_min(1e-6)

    z_all = z[point_all]
    z_min = torch.full((height * width,), float("inf"), device=points.device, dtype=points.dtype)
    z_min.scatter_reduce_(0, lin_all, z_all.detach(), reduce="amin", include_self=True)
    depth_delta = z_all - z_min[lin_all]
    depth_gate = torch.exp(-depth_sharpness * torch.relu(depth_delta - depth_band)).clamp(0.0, 1.0)

    flat_alpha = torch.zeros(height * width, 1, device=points.device)
    flat_color = torch.zeros(height * width, 3, device=points.device)
    point_w = (weight_all / weight_sum[point_all])[:, None]
    aw = a[point_all] * point_w * depth_gate[:, None]
    flat_alpha.scatter_add_(0, lin_all[:, None], aw)
    flat_color.scatter_add_(0, lin_all[:, None].expand(-1, 3), aw * c[point_all])
    density = flat_alpha.view(height, width, 1)
    mask = 1.0 - torch.exp(-density)
    color = flat_color.view(height, width, 3) / torch.clamp(density, min=1e-6)
    image = color * mask + (1.0 - mask)
    return image.clamp(0, 1), mask.clamp(0, 1)


def sample_curve_tangents(curves: torch.Tensor, sample_idx: torch.Tensor) -> torch.Tensor:
    prev_idx = torch.clamp(sample_idx - 1, min=0)
    next_idx = torch.clamp(sample_idx + 1, max=curves.shape[1] - 1)
    tangents = curves[:, next_idx] - curves[:, prev_idx]
    return tangents.reshape(-1, 3)


def adaptive_render_sample_mask(
    curves: torch.Tensor,
    length: torch.Tensor,
    render_samples: int,
    min_samples: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if render_samples <= 1:
        mask = torch.ones(curves.shape[0], render_samples, device=curves.device, dtype=torch.bool)
        return mask, {
            "adaptive_mean_samples": float(render_samples),
            "adaptive_min_samples": int(render_samples),
            "adaptive_max_samples": int(render_samples),
            "adaptive_complexity_mean": 0.0,
        }

    min_samples = int(max(2, min(min_samples, render_samples)))
    seg = curves[:, 1:] - curves[:, :-1]
    arc = torch.linalg.norm(seg, dim=-1).sum(dim=1, keepdim=True)
    chord = torch.linalg.norm(curves[:, -1] - curves[:, 0], dim=-1, keepdim=True)
    arc_over_chord = (arc / torch.clamp(chord, min=EPS) - 1.0).clamp(min=0.0)
    if seg.shape[1] > 1:
        seg_dir = torch_normalize(seg)
        turn = (1.0 - (seg_dir[:, 1:] * seg_dir[:, :-1]).sum(dim=-1).clamp(-1.0, 1.0)).sum(dim=1, keepdim=True)
        turn = turn / max(int(seg.shape[1]) - 1, 1)
        complexity = torch.maximum(arc_over_chord, 4.0 * turn)
    else:
        complexity = arc_over_chord

    def combined_absolute_relative_score(value: torch.Tensor, absolute: torch.Tensor) -> torch.Tensor:
        flat = value.detach().reshape(-1)
        if flat.numel() < 16:
            return absolute.clamp(0.0, 1.0)
        lo = torch.quantile(flat, 0.10)
        hi = torch.quantile(flat, 0.90)
        rel = ((value.detach() - lo) / torch.clamp(hi - lo, min=EPS)).clamp(0.0, 1.0)
        return (0.45 * absolute + 0.55 * rel).clamp(0.0, 1.0)

    length_abs = ((length.detach() - 0.012) / (0.105 - 0.012)).clamp(0.0, 1.0)
    complexity_abs = (complexity.detach() / 0.35).clamp(0.0, 1.0)
    length_score = combined_absolute_relative_score(length, length_abs)
    complexity_score = combined_absolute_relative_score(complexity, complexity_abs)
    score = (0.65 * length_score + 0.35 * complexity_score).clamp(0.0, 1.0)
    budget = torch.round(min_samples + score * (render_samples - min_samples)).long()
    budget = budget.clamp(min_samples, render_samples).view(-1)

    patterns = torch.zeros(render_samples + 1, render_samples, device=curves.device, dtype=torch.bool)
    for k in range(min_samples, render_samples + 1):
        ids = torch.linspace(0, render_samples - 1, k, device=curves.device).round().long().unique()
        patterns[k, ids] = True
        patterns[k, 0] = True
        patterns[k, -1] = True
    mask = patterns[budget]
    active = mask.sum(dim=1)
    stats = {
        "adaptive_mean_samples": float(active.float().mean().detach().cpu()),
        "adaptive_min_samples": int(active.min().detach().cpu()),
        "adaptive_max_samples": int(active.max().detach().cpu()),
        "adaptive_complexity_mean": float(complexity.mean().detach().cpu()),
        "adaptive_length_score_mean": float(length_score.mean().detach().cpu()),
        "adaptive_complexity_score_mean": float(complexity_score.mean().detach().cpu()),
    }
    return mask, stats


def splat_radius_from_strand_width(
    widths: torch.Tensor,
    base_radius: float,
    width_ref: float,
    min_scale: float,
    max_scale: float,
) -> torch.Tensor:
    scale = widths.reshape(-1, 1) / max(float(width_ref), EPS)
    scale = scale.clamp(float(min_scale), float(max_scale))
    return scale * float(base_radius)


def groom_geometry_prior_loss(
    groom,
    lift_raw: torch.Tensor,
    detail_evidence: torch.Tensor,
    boundary_evidence: torch.Tensor | None,
    body_u: torch.Tensor,
    head_start: float,
    head_sharpness: float,
    length_floor: float,
    detail_length_boost: float,
    head_length_boost: float,
    boundary_length_boost: float,
    boundary_density_floor: float,
    boundary_lift_floor: float,
    root_width_target: float,
    tip_width_target: float,
    max_tip_root_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    length_norm = ((groom.length - 0.012) / (0.105 - 0.012)).clamp(0.0, 1.0)
    head_gate = torch.sigmoid(float(head_sharpness) * (body_u - float(head_start)))
    detail = detail_evidence.detach().clamp(0.0, 1.0)
    if boundary_evidence is None:
        boundary = torch.zeros_like(detail)
    else:
        boundary = boundary_evidence.detach().to(device=detail.device, dtype=detail.dtype).clamp(0.0, 1.0)
    target_floor = (
        float(length_floor)
        + float(detail_length_boost) * detail
        + float(head_length_boost) * head_gate
        + float(boundary_length_boost) * boundary
    ).clamp(0.0, 0.95)
    length_floor_loss = torch.relu(target_floor - length_norm).square().mean()

    density_floor_loss = length_norm.new_tensor(0.0)
    if float(boundary_density_floor) > 0.0:
        density_floor = (float(boundary_density_floor) * boundary).clamp(0.0, 0.98)
        density_weight = (0.20 + 0.80 * boundary).detach()
        density_floor_loss = (torch.relu(density_floor - groom.density).square() * density_weight).sum()
        density_floor_loss = density_floor_loss / density_weight.sum().clamp_min(1.0)

    lift_floor_loss = length_norm.new_tensor(0.0)
    if float(boundary_lift_floor) > 0.0:
        lift_norm = torch.sigmoid(lift_raw).clamp(0.0, 1.0)
        lift_weight = boundary.detach()
        lift_floor_loss = (torch.relu(float(boundary_lift_floor) - lift_norm).square() * lift_weight).sum()
        lift_floor_loss = lift_floor_loss / lift_weight.sum().clamp_min(1.0)

    root_log = torch.log(torch.clamp(groom.root_width, min=1e-6))
    tip_log = torch.log(torch.clamp(groom.tip_width, min=1e-6))
    root_target = math.log(max(float(root_width_target), 1e-6))
    tip_target = math.log(max(float(tip_width_target), 1e-6))
    width_prior_loss = F.smooth_l1_loss(root_log, torch.full_like(root_log, root_target))
    width_prior_loss = width_prior_loss + F.smooth_l1_loss(tip_log, torch.full_like(tip_log, tip_target))

    taper_loss = torch.relu(groom.tip_width - groom.root_width * float(max_tip_root_ratio)).square().mean()
    loss = length_floor_loss + density_floor_loss + lift_floor_loss + width_prior_loss + taper_loss
    stats = {
        "groom_length_floor_loss": float(length_floor_loss.detach().cpu()),
        "groom_boundary_density_floor_loss": float(density_floor_loss.detach().cpu()),
        "groom_boundary_lift_floor_loss": float(lift_floor_loss.detach().cpu()),
        "groom_width_prior_loss": float(width_prior_loss.detach().cpu()),
        "groom_taper_loss": float(taper_loss.detach().cpu()),
        "groom_length_floor_mean": float(target_floor.detach().mean().cpu()),
        "groom_length_norm_mean": float(length_norm.detach().mean().cpu()),
        "groom_length_norm_p05": float(torch.quantile(length_norm.detach().reshape(-1), 0.05).cpu()),
        "groom_boundary_evidence_mean": float(boundary.detach().mean().cpu()),
        "groom_boundary_evidence_p95": float(torch.quantile(boundary.detach().reshape(-1), 0.95).cpu()),
        "groom_density_mean": float(groom.density.detach().mean().cpu()),
        "groom_lift_mean": float(torch.sigmoid(lift_raw.detach()).mean().cpu()),
    }
    return loss, stats


def bilinear_sample_image(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    h, w = image.shape[:2]
    grid = xy.clone()
    grid[:, 0] = grid[:, 0] / max(w - 1, 1) * 2.0 - 1.0
    grid[:, 1] = grid[:, 1] / max(h - 1, 1) * 2.0 - 1.0
    grid = grid.view(1, -1, 1, 2)
    src = image.permute(2, 0, 1).unsqueeze(0)
    return F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, :, :, 0].T


@torch.no_grad()
def compute_view_importance_face_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    init_scale: float,
    init_trans: list[float],
    images: torch.Tensor,
    masks: torch.Tensor,
    detail_weights: torch.Tensor,
    pmats: torch.Tensor,
    cam_centers: torch.Tensor,
    boost: float,
    gamma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    centers_np, normals_np, _ = face_geometry(vertices, faces)
    device = images.device
    centers = torch.tensor(centers_np, device=device, dtype=torch.float32)
    normals = torch_normalize(torch.tensor(normals_np, device=device, dtype=torch.float32))
    points = centers * float(init_scale) + torch.tensor(init_trans, device=device, dtype=torch.float32).view(1, 3)
    score = torch.zeros(points.shape[0], 1, device=device)
    visible = torch.zeros_like(score)
    h, w = masks.shape[1:3]

    for view in range(images.shape[0]):
        xy, z = project_points(points, pmats[view])
        inb = (z > 1e-5) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 2) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 2)
        if not inb.any():
            continue
        view_dir = torch_normalize(cam_centers[view].view(1, 3) - points)
        front = torch.sigmoid(24.0 * ((normals * view_dir).sum(dim=-1, keepdim=True) - 0.02))
        sampled_mask = bilinear_sample_image(masks[view], xy)[:, :1]
        sampled_detail = bilinear_sample_image(detail_weights[view], xy)[:, :1]
        conf = sampled_mask * front * inb[:, None].float()
        score += conf * sampled_detail.clamp(0.0, 8.0)
        visible += conf

    raw = score / torch.clamp(visible, min=1e-4)
    raw = torch.where(visible > 1e-4, raw, torch.zeros_like(raw))
    nonzero = raw[raw[:, 0] > 0.0, 0]
    if nonzero.numel() > 16:
        lo = torch.quantile(nonzero, 0.10)
        hi = torch.quantile(nonzero, 0.98)
        norm = ((raw - lo) / torch.clamp(hi - lo, min=1e-4)).clamp(0.0, 1.0)
    else:
        norm = raw.clamp(0.0, 1.0)
    weights = 1.0 + float(boost) * torch.pow(norm, float(gamma))
    stats = {
        "visible_face_fraction": float((visible[:, 0] > 1e-4).float().mean().cpu()),
        "importance_mean": float(norm.mean().cpu()),
        "importance_p95": float(torch.quantile(norm[:, 0], 0.95).cpu()),
        "weight_min": float(weights.min().cpu()),
        "weight_mean": float(weights.mean().cpu()),
        "weight_max": float(weights.max().cpu()),
    }
    return weights[:, 0].detach().cpu().numpy().astype(np.float32), stats


@torch.no_grad()
def compute_orientation_confidence_face_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    init_scale: float,
    init_trans: list[float],
    masks: torch.Tensor,
    target_confidence: torch.Tensor,
    pmats: torch.Tensor,
    cam_centers: torch.Tensor,
    boost: float,
    gamma: float,
    min_confidence: float,
) -> tuple[np.ndarray, dict[str, float]]:
    centers_np, normals_np, _ = face_geometry(vertices, faces)
    device = masks.device
    centers = torch.tensor(centers_np, device=device, dtype=torch.float32)
    normals = torch_normalize(torch.tensor(normals_np, device=device, dtype=torch.float32))
    points = centers * float(init_scale) + torch.tensor(init_trans, device=device, dtype=torch.float32).view(1, 3)
    score = torch.zeros(points.shape[0], 1, device=device)
    visible = torch.zeros_like(score)
    h, w = masks.shape[1:3]

    for view in range(target_confidence.shape[0]):
        xy, z = project_points(points, pmats[view])
        inb = (z > 1e-5) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 2) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 2)
        if not inb.any():
            continue
        view_dir = torch_normalize(cam_centers[view].view(1, 3) - points)
        front = torch.sigmoid(24.0 * ((normals * view_dir).sum(dim=-1, keepdim=True) - 0.02))
        sampled_mask = bilinear_sample_image(masks[view], xy)[:, :1]
        sampled_conf = bilinear_sample_image(target_confidence[view], xy)[:, :1].clamp(0.0, 1.0)
        valid_conf = torch.where(sampled_conf >= float(min_confidence), sampled_conf, torch.zeros_like(sampled_conf))
        conf = sampled_mask * front * inb[:, None].float()
        score += conf * valid_conf
        visible += conf

    raw = score / torch.clamp(visible, min=1e-4)
    raw = torch.where(visible > 1e-4, raw, torch.zeros_like(raw))
    nonzero = raw[raw[:, 0] > 0.0, 0]
    if nonzero.numel() > 16:
        lo = torch.quantile(nonzero, 0.10)
        hi = torch.quantile(nonzero, 0.98)
        norm = ((raw - lo) / torch.clamp(hi - lo, min=1e-4)).clamp(0.0, 1.0)
    else:
        norm = raw.clamp(0.0, 1.0)
    weights = 1.0 + float(boost) * torch.pow(norm, float(gamma))
    stats = {
        "root_orient_visible_face_fraction": float((visible[:, 0] > 1e-4).float().mean().cpu()),
        "root_orient_conf_mean": float(norm.mean().cpu()),
        "root_orient_conf_p95": float(torch.quantile(norm[:, 0], 0.95).cpu()),
        "root_orient_weight_min": float(weights.min().cpu()),
        "root_orient_weight_mean": float(weights.mean().cpu()),
        "root_orient_weight_max": float(weights.max().cpu()),
    }
    return weights[:, 0].detach().cpu().numpy().astype(np.float32), stats


@torch.no_grad()
def compute_boundary_face_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    init_scale: float,
    init_trans: list[float],
    masks: torch.Tensor,
    boundary_weights: torch.Tensor,
    pmats: torch.Tensor,
    cam_centers: torch.Tensor,
    boost: float,
    gamma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    centers_np, normals_np, _ = face_geometry(vertices, faces)
    device = masks.device
    centers = torch.tensor(centers_np, device=device, dtype=torch.float32)
    normals = torch_normalize(torch.tensor(normals_np, device=device, dtype=torch.float32))
    points = centers * float(init_scale) + torch.tensor(init_trans, device=device, dtype=torch.float32).view(1, 3)
    score = torch.zeros(points.shape[0], 1, device=device)
    visible = torch.zeros_like(score)
    h, w = masks.shape[1:3]

    for view in range(boundary_weights.shape[0]):
        xy, z = project_points(points, pmats[view])
        inb = (z > 1e-5) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 2) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 2)
        if not inb.any():
            continue
        view_dir = torch_normalize(cam_centers[view].view(1, 3) - points)
        front = torch.sigmoid(24.0 * ((normals * view_dir).sum(dim=-1, keepdim=True) - 0.02))
        sampled_mask = bilinear_sample_image(masks[view], xy)[:, :1]
        sampled_boundary = bilinear_sample_image(boundary_weights[view], xy)[:, :1].clamp(0.0, 1.0)
        conf = sampled_mask * front * inb[:, None].float()
        score += conf * sampled_boundary
        visible += conf

    raw = score / torch.clamp(visible, min=1e-4)
    raw = torch.where(visible > 1e-4, raw, torch.zeros_like(raw))
    nonzero = raw[raw[:, 0] > 0.0, 0]
    if nonzero.numel() > 16:
        lo = torch.quantile(nonzero, 0.10)
        hi = torch.quantile(nonzero, 0.98)
        norm = ((raw - lo) / torch.clamp(hi - lo, min=1e-4)).clamp(0.0, 1.0)
    else:
        norm = raw.clamp(0.0, 1.0)
    weights = 1.0 + float(boost) * torch.pow(norm, float(gamma))
    stats = {
        "root_boundary_boost": float(boost),
        "root_boundary_gamma": float(gamma),
        "root_boundary_visible_face_fraction": float((visible[:, 0] > 1e-4).float().mean().cpu()),
        "root_boundary_score_mean": float(norm.mean().cpu()),
        "root_boundary_score_p95": float(torch.quantile(norm[:, 0], 0.95).cpu()),
        "root_boundary_weight_min": float(weights.min().cpu()),
        "root_boundary_weight_mean": float(weights.mean().cpu()),
        "root_boundary_weight_max": float(weights.max().cpu()),
    }
    return weights[:, 0].detach().cpu().numpy().astype(np.float32), stats


@torch.no_grad()
def compute_head_detail_face_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    init_scale: float,
    init_trans: list[float],
    masks: torch.Tensor,
    detail_weights: torch.Tensor,
    target_confidence: torch.Tensor | None,
    pmats: torch.Tensor,
    cam_centers: torch.Tensor,
    boost: float,
    gamma: float,
    head_start: float,
    head_sharpness: float,
    orient_mix: float,
    min_orient_confidence: float,
) -> tuple[np.ndarray, dict[str, float]]:
    centers_np, normals_np, areas_np = face_geometry(vertices, faces)
    valid_faces_np = np.where(areas_np > 0)[0]
    if valid_faces_np.size == 0:
        weights = np.ones(faces.shape[0], dtype=np.float32)
        return weights, {
            "root_head_detail_valid_face_fraction": 0.0,
            "root_head_detail_weight_min": 1.0,
            "root_head_detail_weight_mean": 1.0,
            "root_head_detail_weight_max": 1.0,
        }

    axis_np = infer_axis(centers_np[valid_faces_np])
    axis_coord = centers_np @ axis_np
    axis_lo, axis_hi = np.quantile(axis_coord[valid_faces_np], [0.002, 0.998])
    face_body_u_np = np.clip((axis_coord - axis_lo) / max(float(axis_hi - axis_lo), EPS), 0.0, 1.0)

    device = masks.device
    centers = torch.tensor(centers_np, device=device, dtype=torch.float32)
    normals = torch_normalize(torch.tensor(normals_np, device=device, dtype=torch.float32))
    points = centers * float(init_scale) + torch.tensor(init_trans, device=device, dtype=torch.float32).view(1, 3)
    head_gate = torch.tensor(face_body_u_np, device=device, dtype=torch.float32).view(-1, 1)
    head_gate = torch.sigmoid(float(head_sharpness) * (head_gate - float(head_start)))

    detail_score = torch.zeros(points.shape[0], 1, device=device)
    orient_score = torch.zeros_like(detail_score)
    visible = torch.zeros_like(detail_score)
    h, w = masks.shape[1:3]
    use_orient = target_confidence is not None and float(orient_mix) > 0.0

    for view in range(masks.shape[0]):
        xy, z = project_points(points, pmats[view])
        inb = (z > 1e-5) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 2) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 2)
        if not inb.any():
            continue
        view_dir = torch_normalize(cam_centers[view].view(1, 3) - points)
        front = torch.sigmoid(24.0 * ((normals * view_dir).sum(dim=-1, keepdim=True) - 0.02))
        sampled_mask = bilinear_sample_image(masks[view], xy)[:, :1]
        sampled_detail = bilinear_sample_image(detail_weights[view], xy)[:, :1].clamp(0.0, 8.0)
        conf = sampled_mask * front * inb[:, None].float()
        detail_score += conf * sampled_detail
        if use_orient and target_confidence is not None:
            sampled_orient = bilinear_sample_image(target_confidence[view], xy)[:, :1].clamp(0.0, 1.0)
            valid_orient = torch.where(
                sampled_orient >= float(min_orient_confidence),
                sampled_orient,
                torch.zeros_like(sampled_orient),
            )
            orient_score += conf * valid_orient
        visible += conf

    def normalize_score(raw_score: torch.Tensor) -> torch.Tensor:
        raw = raw_score / torch.clamp(visible, min=1e-4)
        raw = torch.where(visible > 1e-4, raw, torch.zeros_like(raw))
        nonzero = raw[raw[:, 0] > 0.0, 0]
        if nonzero.numel() > 16:
            lo = torch.quantile(nonzero, 0.10)
            hi = torch.quantile(nonzero, 0.98)
            return ((raw - lo) / torch.clamp(hi - lo, min=1e-4)).clamp(0.0, 1.0)
        return raw.clamp(0.0, 1.0)

    detail_norm = normalize_score(detail_score)
    if use_orient:
        orient_norm = normalize_score(orient_score)
        orient_mix_clamped = max(0.0, min(float(orient_mix), 1.0))
        evidence = (1.0 - orient_mix_clamped) * detail_norm + orient_mix_clamped * orient_norm
    else:
        orient_norm = torch.zeros_like(detail_norm)
        orient_mix_clamped = 0.0
        evidence = detail_norm
    head_score = (head_gate * evidence).clamp(0.0, 1.0)
    weights = 1.0 + float(boost) * torch.pow(head_score, float(gamma))
    valid_faces = torch.tensor(valid_faces_np, device=device, dtype=torch.long)
    stats = {
        "root_head_detail_boost": float(boost),
        "root_head_detail_gamma": float(gamma),
        "root_head_detail_start": float(head_start),
        "root_head_detail_sharpness": float(head_sharpness),
        "root_head_detail_orient_mix": float(orient_mix_clamped),
        "root_head_detail_min_orient_confidence": float(min_orient_confidence),
        "root_head_detail_valid_face_fraction": float(valid_faces_np.size / max(faces.shape[0], 1)),
        "root_head_detail_visible_face_fraction": float((visible[:, 0] > 1e-4).float().mean().cpu()),
        "root_head_detail_gate_mean": float(head_gate[valid_faces].mean().cpu()),
        "root_head_detail_gate_p95": float(torch.quantile(head_gate[valid_faces, 0], 0.95).cpu()),
        "root_head_detail_detail_mean": float(detail_norm[valid_faces].mean().cpu()),
        "root_head_detail_detail_p95": float(torch.quantile(detail_norm[valid_faces, 0], 0.95).cpu()),
        "root_head_detail_orient_mean": float(orient_norm[valid_faces].mean().cpu()),
        "root_head_detail_orient_p95": float(torch.quantile(orient_norm[valid_faces, 0], 0.95).cpu()),
        "root_head_detail_evidence_mean": float(evidence[valid_faces].mean().cpu()),
        "root_head_detail_evidence_p95": float(torch.quantile(evidence[valid_faces, 0], 0.95).cpu()),
        "root_head_detail_score_mean": float(head_score[valid_faces].mean().cpu()),
        "root_head_detail_score_p95": float(torch.quantile(head_score[valid_faces, 0], 0.95).cpu()),
        "root_head_detail_weight_min": float(weights[valid_faces].min().cpu()),
        "root_head_detail_weight_mean": float(weights[valid_faces].mean().cpu()),
        "root_head_detail_weight_max": float(weights[valid_faces].max().cpu()),
    }
    return weights[:, 0].detach().cpu().numpy().astype(np.float32), stats


def representative_face_uvs(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_mode: str,
    uv_vertices: np.ndarray | None,
    face_uvs: np.ndarray | None,
) -> np.ndarray:
    centers, _, areas = face_geometry(vertices, faces)
    candidates = keep_largest_face_component(faces, np.where(areas > 0)[0])
    if uv_mode in {"xatlas", "obj"}:
        if uv_vertices is None or face_uvs is None:
            raise ValueError(f"uv_mode={uv_mode} requires uv_vertices and face_uvs")
        uv_tri = uv_vertices[face_uvs]
        return np.clip(np.mod(uv_tri.mean(axis=1), 1.0), 0.0, 1.0).astype(np.float32)
    if uv_mode == "cylindrical":
        axis = infer_axis(centers[candidates])
        b1, b2 = make_perp_basis(axis)
        center = centers[candidates].mean(axis=0).astype(np.float32)
        rel = centers - center[None, :]
        axis_coord = centers @ axis
        axis_lo, axis_hi = np.quantile(centers[candidates] @ axis, [0.002, 0.998])
        body_u = np.clip((axis_coord - axis_lo) / max(float(axis_hi - axis_lo), EPS), 0.0, 1.0)
        theta = np.arctan2(rel @ b2, rel @ b1)
        v = (theta + math.pi) / (2.0 * math.pi)
        return np.stack([body_u, v], axis=1).astype(np.float32)
    raise ValueError(f"unsupported uv_mode: {uv_mode}")


def sample_debug_uv_map(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.clip(uv[:, 0] * max(w - 1, 1), 0.0, max(w - 1, 0))
    # residual_evidence.png and uv coverage debug images are written with V
    # flipped for visual inspection, so sample them the same way.
    y = np.clip((1.0 - uv[:, 1]) * max(h - 1, 1), 0.0, max(h - 1, 0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]
    return (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    ).astype(np.float32)


def compute_residual_map_face_weights(
    residual_map_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_mode: str,
    uv_vertices: np.ndarray | None,
    face_uvs: np.ndarray | None,
    boost: float,
    gamma: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    image = Image.open(residual_map_path).convert("L")
    residual = np.asarray(image, dtype=np.float32) / 255.0
    face_uv = representative_face_uvs(vertices, faces, uv_mode, uv_vertices, face_uvs)
    raw = sample_debug_uv_map(residual, face_uv)
    valid = np.isfinite(raw) & (raw > 1e-6)
    norm = np.zeros_like(raw, dtype=np.float32)
    if int(valid.sum()) > 16:
        lo, hi = np.quantile(raw[valid], [0.10, 0.98])
        norm = np.clip((raw - float(lo)) / max(float(hi - lo), EPS), 0.0, 1.0).astype(np.float32)
    elif raw.size:
        norm = np.clip(raw, 0.0, 1.0).astype(np.float32)
    weights = 1.0 + float(boost) * np.power(norm, float(gamma))
    stats: dict[str, float | str] = {
        "residual_map": str(residual_map_path),
        "residual_valid_face_fraction": float(valid.mean()) if raw.size else 0.0,
        "residual_mean": float(norm.mean()) if norm.size else 0.0,
        "residual_p95": float(np.quantile(norm, 0.95)) if norm.size else 0.0,
        "residual_weight_min": float(weights.min()) if weights.size else 1.0,
        "residual_weight_mean": float(weights.mean()) if weights.size else 1.0,
        "residual_weight_max": float(weights.max()) if weights.size else 1.0,
    }
    return weights.astype(np.float32), stats


def compute_uv_atlas_stats(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_mode: str,
    uv_vertices: np.ndarray | None,
    face_uvs: np.ndarray | None,
    tex_h: int,
    tex_w: int,
    head_start: float,
) -> dict[str, float | int | str]:
    centers, _, mesh_areas = face_geometry(vertices, faces)
    valid_faces = np.where(mesh_areas > 0)[0]
    stats: dict[str, float | int | str] = {
        "uv_mode": uv_mode,
        "tex_h": int(tex_h),
        "tex_w": int(tex_w),
        "texel_count": int(tex_h) * int(tex_w),
        "face_count": int(faces.shape[0]),
        "valid_face_count": int(valid_faces.size),
    }
    if valid_faces.size == 0:
        return stats

    axis = infer_axis(centers[valid_faces])
    axis_coord = centers @ axis
    axis_lo, axis_hi = np.quantile(axis_coord[valid_faces], [0.002, 0.998])
    face_body_u = np.clip((axis_coord - axis_lo) / max(float(axis_hi - axis_lo), EPS), 0.0, 1.0)
    head_gate = 1.0 / (1.0 + np.exp(-28.0 * (face_body_u - float(head_start))))
    mesh_area_sum = float(mesh_areas[valid_faces].sum())
    stats.update(
        {
            "mesh_area_sum": mesh_area_sum,
            "mesh_head_area_fraction": float((mesh_areas * (face_body_u >= float(head_start))).sum() / max(mesh_area_sum, EPS)),
            "mesh_soft_head_area_fraction": float((mesh_areas * head_gate).sum() / max(mesh_area_sum, EPS)),
        }
    )
    if uv_mode not in {"xatlas", "obj"} or uv_vertices is None or face_uvs is None:
        return stats

    uv_tri = np.mod(uv_vertices[face_uvs], 1.0).astype(np.float32)
    uv_area = 0.5 * np.abs(
        (uv_tri[:, 1, 0] - uv_tri[:, 0, 0]) * (uv_tri[:, 2, 1] - uv_tri[:, 0, 1])
        - (uv_tri[:, 1, 1] - uv_tri[:, 0, 1]) * (uv_tri[:, 2, 0] - uv_tri[:, 0, 0])
    )
    uv_valid = np.isfinite(uv_area) & (uv_area > 0)
    uv_area_sum = float(uv_area[uv_valid].sum())
    texel_count = float(max(int(tex_h) * int(tex_w), 1))
    stats.update(
        {
            "uv_vertex_count": int(uv_vertices.shape[0]),
            "uv_area_sum": uv_area_sum,
            "uv_area_fraction": uv_area_sum,
            "uv_estimated_texels": float(uv_area_sum * texel_count),
            "uv_head_area_fraction": float((uv_area * (face_body_u >= float(head_start))).sum() / max(uv_area_sum, EPS)),
            "uv_soft_head_area_fraction": float((uv_area * head_gate).sum() / max(uv_area_sum, EPS)),
            "uv_soft_head_estimated_texels": float((uv_area * head_gate).sum() * texel_count),
            "uv_min_u": float(np.nanmin(uv_tri[:, :, 0])),
            "uv_max_u": float(np.nanmax(uv_tri[:, :, 0])),
            "uv_min_v": float(np.nanmin(uv_tri[:, :, 1])),
            "uv_max_v": float(np.nanmax(uv_tri[:, :, 1])),
        }
    )
    return stats


def summarize_root_uv_coverage(
    uv: np.ndarray,
    body_u: np.ndarray,
    tex_h: int,
    tex_w: int,
    head_start: float,
    tile_res: int = 64,
) -> dict[str, float | int]:
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    body_u = np.asarray(body_u, dtype=np.float32).reshape(-1)
    if uv.shape[0] == 0:
        return {"root_uv_count": 0}
    h, w = int(tex_h), int(tex_w)
    gx = uv[:, 0] * max(w - 1, 1)
    gy = uv[:, 1] * max(h - 1, 1)
    x = np.clip(np.floor(gx).astype(np.int64), 0, max(w - 1, 0))
    y = np.clip(np.floor(gy).astype(np.int64), 0, max(h - 1, 0))
    lin = y * max(w, 1) + x
    unique_texels = np.unique(lin).size
    head_mask = body_u >= float(head_start)
    head_unique_texels = np.unique(lin[head_mask]).size if head_mask.any() else 0
    x1 = np.clip(x + 1, 0, max(w - 1, 0))
    y1 = np.clip(y + 1, 0, max(h - 1, 0))
    bilinear_lin = np.concatenate(
        [
            y * max(w, 1) + x,
            y * max(w, 1) + x1,
            y1 * max(w, 1) + x,
            y1 * max(w, 1) + x1,
        ],
        axis=0,
    )
    bilinear_unique_texels = np.unique(bilinear_lin).size
    if head_mask.any():
        hx = x[head_mask]
        hy = y[head_mask]
        hx1 = x1[head_mask]
        hy1 = y1[head_mask]
        head_bilinear_lin = np.concatenate(
            [
                hy * max(w, 1) + hx,
                hy * max(w, 1) + hx1,
                hy1 * max(w, 1) + hx,
                hy1 * max(w, 1) + hx1,
            ],
            axis=0,
        )
        head_bilinear_unique_texels = np.unique(head_bilinear_lin).size
    else:
        head_bilinear_unique_texels = 0
    tile_h = min(max(int(tile_res), 1), max(h, 1))
    tile_w = min(max(int(tile_res), 1), max(w, 1))
    tx = np.clip(np.floor(uv[:, 0] * tile_w).astype(np.int64), 0, tile_w - 1)
    ty = np.clip(np.floor((1.0 - uv[:, 1]) * tile_h).astype(np.int64), 0, tile_h - 1)
    tile_lin = ty * tile_w + tx
    unique_tiles = np.unique(tile_lin).size
    head_unique_tiles = np.unique(tile_lin[head_mask]).size if head_mask.any() else 0
    texel_count = max(h * w, 1)
    tile_count = max(tile_h * tile_w, 1)
    return {
        "root_uv_count": int(uv.shape[0]),
        "root_uv_unique_texels": int(unique_texels),
        "root_uv_unique_texel_fraction": float(unique_texels / texel_count),
        "root_uv_bilinear_unique_texels": int(bilinear_unique_texels),
        "root_uv_bilinear_unique_texel_fraction": float(bilinear_unique_texels / texel_count),
        "root_uv_head_unique_texels": int(head_unique_texels),
        "root_uv_head_unique_texel_fraction": float(head_unique_texels / texel_count),
        "root_uv_head_bilinear_unique_texels": int(head_bilinear_unique_texels),
        "root_uv_head_bilinear_unique_texel_fraction": float(head_bilinear_unique_texels / texel_count),
        "root_uv_occupied_tiles": int(unique_tiles),
        "root_uv_occupied_tile_fraction": float(unique_tiles / tile_count),
        "root_uv_head_occupied_tiles": int(head_unique_tiles),
        "root_uv_head_occupied_tile_fraction": float(head_unique_tiles / tile_count),
        "root_uv_roots_per_occupied_texel": float(uv.shape[0] / max(unique_texels, 1)),
        "root_uv_head_roots_per_occupied_texel": float(head_mask.sum() / max(head_unique_texels, 1)),
    }


def summarize_sampled_roots(root_data: dict[str, np.ndarray], head_start: float) -> dict[str, float]:
    body_u = np.asarray(root_data["body_u"], dtype=np.float32).reshape(-1)
    uv = np.asarray(root_data["uv"], dtype=np.float32)
    face_ids = np.asarray(root_data["face_ids"], dtype=np.int64).reshape(-1)
    unique_faces = np.unique(face_ids).size
    return {
        "sampled_body_u_mean": float(np.mean(body_u)) if body_u.size else 0.0,
        "sampled_body_u_p05": float(np.quantile(body_u, 0.05)) if body_u.size else 0.0,
        "sampled_body_u_p50": float(np.quantile(body_u, 0.50)) if body_u.size else 0.0,
        "sampled_body_u_p95": float(np.quantile(body_u, 0.95)) if body_u.size else 0.0,
        "sampled_head_fraction": float(np.mean(body_u >= float(head_start))) if body_u.size else 0.0,
        "sampled_unique_face_fraction": float(unique_faces / max(face_ids.size, 1)),
        "sampled_uv_u_mean": float(np.mean(uv[:, 0])) if uv.size else 0.0,
        "sampled_uv_v_mean": float(np.mean(uv[:, 1])) if uv.size else 0.0,
    }


def uv_scatter_average(
    uv: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    tex_h: int,
    tex_w: int,
    default: torch.Tensor,
) -> torch.Tensor:
    x = torch.clamp(torch.round(uv[:, 0] * (tex_w - 1)).long(), 0, tex_w - 1)
    y = torch.clamp(torch.round(uv[:, 1] * (tex_h - 1)).long(), 0, tex_h - 1)
    lin = y * tex_w + x
    c = values.shape[1]
    acc = torch.zeros(tex_h * tex_w, c, device=uv.device)
    cnt = torch.zeros(tex_h * tex_w, 1, device=uv.device)
    w = weights.clamp(0, 1)
    acc.scatter_add_(0, lin[:, None].expand(-1, c), values * w)
    cnt.scatter_add_(0, lin[:, None], w)
    out = default.view(1, c).expand(tex_h * tex_w, c).clone()
    known = cnt[:, 0] > 1e-4
    out[known] = acc[known] / torch.clamp(cnt[known], min=1e-6)
    tex = out.view(tex_h, tex_w, c).permute(2, 0, 1).unsqueeze(0)
    mask = known.float().view(1, 1, tex_h, tex_w)
    for _ in range(32):
        smooth = F.avg_pool2d(tex, kernel_size=3, stride=1, padding=1)
        tex = tex * mask + smooth * (1.0 - mask)
    return tex


def init_texture_from_images(
    model: TextureGroom,
    roots: torch.Tensor,
    normals: torch.Tensor,
    uv: torch.Tensor,
    images: torch.Tensor,
    masks: torch.Tensor,
    detail_weights: torch.Tensor,
    pmats: torch.Tensor,
    cam_centers: torch.Tensor,
    flow_hint: torch.Tensor | None = None,
    flow_hint_confidence: torch.Tensor | None = None,
    detail_init_strength: float = 0.0,
    detail_density_boost: float = 0.0,
    detail_length_boost: float = 0.0,
    detail_root_width_boost: float = 0.0,
    detail_tip_width_boost: float = 0.0,
) -> None:
    with torch.no_grad():
        scale = torch.exp(model.log_scale).clamp(0.25, 4.0)
        points = roots * scale + model.trans.view(1, 3)
        color_acc = torch.zeros_like(roots)
        detail_acc = torch.zeros(roots.shape[0], 1, device=roots.device)
        conf_acc = torch.zeros(roots.shape[0], 1, device=roots.device)
        for view in range(images.shape[0]):
            xy, z = project_points(points, pmats[view])
            h, w = images.shape[1], images.shape[2]
            inb = (z > 1e-5) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 2) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 2)
            view_dir = torch_normalize(cam_centers[view][None, :] - points)
            front = torch.sigmoid(24.0 * ((normals * view_dir).sum(dim=-1, keepdim=True) - 0.02))
            sampled_mask = bilinear_sample_image(masks[view], xy)[:, :1]
            sampled_color = bilinear_sample_image(images[view], xy)
            sampled_detail = bilinear_sample_image(detail_weights[view], xy)[:, :1]
            # Reduce trust close to silhouette edges by suppressing uncertain mask values.
            conf = sampled_mask * front * inb[:, None].float()
            conf = torch.where(sampled_mask > 0.85, conf, conf * 0.25)
            color_acc += sampled_color * conf
            detail_acc += sampled_detail.clamp(0.0, 8.0) * conf
            conf_acc += conf
        root_color = color_acc / torch.clamp(conf_acc, min=1e-6)
        root_detail = detail_acc / torch.clamp(conf_acc, min=1e-6)
        conf = (conf_acc / torch.clamp(conf_acc.max(), min=1e-6)).clamp(0, 1)
        detail_norm = torch.zeros_like(root_detail)
        detail_known = (conf_acc[:, 0] > 1e-4) & torch.isfinite(root_detail[:, 0])
        if int(detail_known.sum().item()) > 16:
            known_detail = root_detail[detail_known, 0]
            lo = torch.quantile(known_detail, 0.10)
            hi = torch.quantile(known_detail, 0.98)
            detail_norm = ((root_detail - lo) / torch.clamp(hi - lo, min=1e-4)).clamp(0.0, 1.0)
        luminance = (0.2126 * root_color[:, 0:1] + 0.7152 * root_color[:, 1:2] + 0.0722 * root_color[:, 2:3]).clamp(0, 1)
        darkness = ((0.78 - luminance) / 0.62).clamp(0.0, 0.92)
        default_rgb = torch.tensor([0.92, 0.90, 0.84], device=roots.device).view(1, 3)
        root_rgb = torch.where(conf_acc > 1e-4, root_color.clamp(0.03, 0.97), default_rgb.expand_as(root_color))
        rgb_tex = uv_scatter_average(uv, root_rgb, conf, model.tex_h, model.tex_w, torch.tensor([0.92, 0.90, 0.84], device=roots.device))
        dark_tex = uv_scatter_average(uv, darkness, conf, model.tex_h, model.tex_w, torch.tensor([0.08], device=roots.device))
        detail_tex = uv_scatter_average(
            uv,
            detail_norm,
            conf,
            model.tex_h,
            model.tex_w,
            torch.tensor([0.0], device=roots.device),
        )
        density_root = conf.clamp(0.35, 1.0)
        if detail_init_strength > 0.0 and detail_density_boost != 0.0:
            density_root = density_root * (1.0 + float(detail_init_strength) * float(detail_density_boost) * detail_norm)
        density_hint = uv_scatter_average(
            uv,
            density_root.clamp(0.03, 0.97),
            conf,
            model.tex_h,
            model.tex_w,
            torch.tensor([0.75], device=roots.device),
        )
        tex = model.texture.data
        tex[:, 1:2] = torch.logit(density_hint.clamp(0.03, 0.97))
        if detail_init_strength > 0.0 and detail_length_boost != 0.0:
            length_base = torch.sigmoid(tex[:, 2:3])
            length_hint = (length_base + float(detail_init_strength) * float(detail_length_boost) * detail_tex).clamp(0.03, 0.97)
            tex[:, 2:3] = torch.logit(length_hint)
        if detail_init_strength > 0.0 and detail_root_width_boost != 0.0:
            tex[:, 3:4] = tex[:, 3:4] + float(detail_init_strength) * float(detail_root_width_boost) * detail_tex
        if detail_init_strength > 0.0 and detail_tip_width_boost != 0.0:
            tex[:, 4:5] = tex[:, 4:5] + float(detail_init_strength) * float(detail_tip_width_boost) * detail_tex
        tex[:, 11:14] = torch.logit(rgb_tex.clamp(0.03, 0.97))
        tex[:, 14:17] = torch.logit(rgb_tex.clamp(0.03, 0.97))
        tex[:, 17:18] = torch.logit(dark_tex.clamp(0.03, 0.97))
        tex[:, 19:20] = detail_tex.clamp(0.0, 1.0)
        if flow_hint is not None and flow_hint_confidence is not None:
            flow_default = torch.tensor([0.55, 0.04], device=roots.device)
            flow_tex = uv_scatter_average(
                uv,
                flow_hint,
                flow_hint_confidence.clamp(0.0, 1.0),
                model.tex_h,
                model.tex_w,
                flow_default,
            )
            flow_conf_tex = uv_scatter_average(
                uv,
                flow_hint_confidence.clamp(0.0, 1.0),
                flow_hint_confidence.clamp(0.0, 1.0),
                model.tex_h,
                model.tex_w,
                torch.tensor([0.0], device=roots.device),
            )
            tex[:, 5:7] = flow_tex
            tex[:, 18:19] = flow_conf_tex.clamp(0.0, 1.0)


def split_initialized_texture_into_coarse_base(model: TextureGroom) -> dict[str, float | int]:
    if model.coarse_texture is None:
        return {"coarse_init_enabled": 0}
    with torch.no_grad():
        active_channels = 18
        scale = float(model.coarse_scale)
        if abs(scale) < 1e-8:
            return {"coarse_init_enabled": 0, "coarse_init_reason": "coarse_scale_zero"}
        high = model.texture.data[:, :active_channels].clone()
        coarse = F.interpolate(
            high,
            size=model.coarse_texture.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        up = F.interpolate(
            coarse,
            size=model.texture.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        model.coarse_texture.data.zero_()
        model.coarse_texture.data[:, :active_channels] = coarse
        model.texture.data[:, :active_channels] = high - scale * up
        combined = model.texture.data[:, :active_channels] + scale * up
        err = (combined - high).abs()
        return {
            "coarse_init_enabled": 1,
            "coarse_init_active_channels": int(active_channels),
            "coarse_init_scale": float(scale),
            "coarse_init_mean_abs": float(coarse.abs().mean().detach().cpu()),
            "coarse_init_p95_abs": float(torch.quantile(coarse.abs().reshape(-1), 0.95).detach().cpu()),
            "highres_residual_mean_abs": float(model.texture.data[:, :active_channels].abs().mean().detach().cpu()),
            "combined_grid_mean_abs_error": float(err.mean().detach().cpu()),
            "combined_grid_max_abs_error": float(err.max().detach().cpu()),
        }


def init_surface_texture_from_images(
    model: TextureGroom,
    surface_points: torch.Tensor,
    surface_normals: torch.Tensor,
    surface_uv: torch.Tensor,
    images: torch.Tensor,
    masks: torch.Tensor,
    pmats: torch.Tensor,
    cam_centers: torch.Tensor,
) -> None:
    if model.surface_texture is None:
        return
    with torch.no_grad():
        scale = torch.exp(model.log_scale).clamp(0.25, 4.0)
        points = surface_points * scale + model.trans.view(1, 3)
        color_acc = torch.zeros_like(surface_points)
        conf_acc = torch.zeros(surface_points.shape[0], 1, device=surface_points.device)
        for view in range(images.shape[0]):
            xy, z = project_points(points, pmats[view])
            h, w = images.shape[1], images.shape[2]
            inb = (z > 1e-5) & (xy[:, 0] >= 0) & (xy[:, 0] <= w - 2) & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 2)
            view_dir = torch_normalize(cam_centers[view][None, :] - points)
            front = torch.sigmoid(24.0 * ((surface_normals * view_dir).sum(dim=-1, keepdim=True) - 0.02))
            sampled_mask = bilinear_sample_image(masks[view], xy)[:, :1]
            sampled_color = bilinear_sample_image(images[view], xy)
            conf = sampled_mask * front * inb[:, None].float()
            conf = torch.where(sampled_mask > 0.88, conf, conf * 0.20)
            color_acc += sampled_color * conf
            conf_acc += conf
        surface_color = color_acc / torch.clamp(conf_acc, min=1e-6)
        conf = (conf_acc / torch.clamp(conf_acc.max(), min=1e-6)).clamp(0, 1)
        default_rgb = torch.tensor([0.92, 0.90, 0.84], device=surface_points.device)
        rgb = torch.where(conf_acc > 1e-4, surface_color.clamp(0.03, 0.97), default_rgb.view(1, 3).expand_as(surface_color))
        rgb_tex = uv_scatter_average(surface_uv, rgb, conf, model.tex_h, model.tex_w, default_rgb)
        alpha_hint = uv_scatter_average(surface_uv, conf.clamp(0.45, 1.0), conf, model.tex_h, model.tex_w, torch.tensor([0.75], device=surface_points.device))
        tex = model.surface_texture.data
        tex[:, 0:3] = torch.logit(rgb_tex.clamp(0.03, 0.97))
        tex[:, 3:4] = torch.logit(alpha_hint.clamp(0.03, 0.97))


def append_surface_layer(
    model: TextureGroom,
    pts: torch.Tensor,
    nrm: torch.Tensor,
    cols: torch.Tensor,
    alpha: torch.Tensor,
    surface_roots: torch.Tensor | None,
    surface_normals: torch.Tensor | None,
    surface_uv: torch.Tensor | None,
    scale: torch.Tensor,
    surface_alpha_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        model.surface_texture is None
        or surface_roots is None
        or surface_normals is None
        or surface_uv is None
        or surface_alpha_scale <= 0.0
    ):
        return pts, nrm, cols, alpha
    surface = model.sample_surface(surface_uv)
    surface_pts = surface_roots * scale + model.trans.view(1, 3)
    surface_alpha = surface["surface_alpha"] * surface_alpha_scale
    return (
        torch.cat([surface_pts, pts], dim=0),
        torch.cat([surface_normals, nrm], dim=0),
        torch.cat([surface["surface_rgb"], cols], dim=0),
        torch.cat([surface_alpha, alpha], dim=0),
    )


def append_random_mesh_backing(
    model: TextureGroom,
    pts: torch.Tensor,
    nrm: torch.Tensor,
    cols: torch.Tensor,
    alpha: torch.Tensor,
    surface_roots: torch.Tensor | None,
    surface_normals: torch.Tensor | None,
    scale: torch.Tensor,
    alpha_scale: float,
    color_strength: float,
    normal_offset: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if surface_roots is None or surface_normals is None or alpha_scale <= 0.0 or color_strength <= 0.0:
        return pts, nrm, cols, alpha
    surface_pts = surface_roots * scale + model.trans.view(1, 3)
    surface_pts = surface_pts - surface_normals * float(normal_offset)
    rand = torch.rand(
        surface_roots.shape[0],
        3,
        device=surface_roots.device,
        dtype=cols.dtype,
    )
    backing_cols = (0.5 + (rand - 0.5) * float(color_strength)).clamp(0.02, 0.98)
    backing_alpha = torch.full(
        (surface_roots.shape[0], 1),
        float(alpha_scale),
        device=surface_roots.device,
        dtype=alpha.dtype,
    )
    return (
        torch.cat([surface_pts, pts], dim=0),
        torch.cat([surface_normals, nrm], dim=0),
        torch.cat([backing_cols, cols], dim=0),
        torch.cat([backing_alpha, alpha], dim=0),
    )


def save_tensor_image(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (image.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(path)


def image_metrics(pred: torch.Tensor, target: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> dict[str, float]:
    mse = torch.mean(torch.square(pred - target)).clamp_min(1e-12)
    l1 = torch.mean(torch.abs(pred - target))
    mask_l1 = torch.mean(torch.abs(pred_mask - target_mask))
    psnr = -10.0 * torch.log10(mse)
    return {
        "psnr": float(psnr.detach().cpu()),
        "l1": float(l1.detach().cpu()),
        "mask_l1": float(mask_l1.detach().cpu()),
    }


def weighted_image_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    weight: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    weight = weight.clamp(min=0.0)
    if weight.ndim == 2:
        weight = weight[..., None]
    denom = weight.sum().clamp_min(1.0)
    color_denom = (denom * pred.shape[-1]).clamp_min(1.0)
    mse = (torch.square(pred - target) * weight).sum() / color_denom
    l1 = (torch.abs(pred - target) * weight).sum() / color_denom
    mask_l1 = (torch.abs(pred_mask - target_mask) * weight).sum() / denom
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    coverage = (weight > 1e-5).float().mean()
    mean_weight = weight.mean()
    return {
        f"{prefix}_psnr": float(psnr.detach().cpu()),
        f"{prefix}_l1": float(l1.detach().cpu()),
        f"{prefix}_mask_l1": float(mask_l1.detach().cpu()),
        f"{prefix}_coverage": float(coverage.detach().cpu()),
        f"{prefix}_mean_weight": float(mean_weight.detach().cpu()),
    }


@torch.no_grad()
def projected_head_weight_mask(
    roots_world: torch.Tensor,
    normals: torch.Tensor,
    body_u: torch.Tensor,
    pmat: torch.Tensor,
    cam_center: torch.Tensor,
    target_mask: torch.Tensor,
    head_start: float,
    dilation: int = 13,
) -> torch.Tensor:
    height, width = int(target_mask.shape[0]), int(target_mask.shape[1])
    xy, z = project_points(roots_world, pmat)
    view_dir = torch_normalize(cam_center.view(1, 3) - roots_world)
    front = torch.sigmoid(24.0 * ((normals * view_dir).sum(dim=-1, keepdim=True) - 0.02))[:, 0]
    head = body_u.reshape(-1) >= float(head_start)
    valid = (
        head
        & (front > 0.15)
        & (z > 1e-5)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= height - 1)
    )
    mask = torch.zeros(height * width, device=roots_world.device, dtype=roots_world.dtype)
    if bool(valid.any()):
        x = xy[valid, 0].round().long().clamp(0, width - 1)
        y = xy[valid, 1].round().long().clamp(0, height - 1)
        lin = y * width + x
        values = front[valid].to(mask.dtype)
        mask.index_put_((lin,), values, accumulate=True)
    mask = mask.view(1, 1, height, width).clamp(0.0, 1.0)
    if dilation > 1:
        dilation = int(dilation)
        if dilation % 2 == 0:
            dilation += 1
        mask = F.max_pool2d(mask, kernel_size=dilation, stride=1, padding=dilation // 2)
    mask = gaussian_blur2d(mask, sigma=max(1.0, float(dilation) / 6.0)).clamp(0.0, 1.0)
    return (mask[0].permute(1, 2, 0) * target_mask.clamp(0.0, 1.0)).clamp(0.0, 1.0)


@torch.no_grad()
def build_head_view_weights(
    roots: torch.Tensor,
    normals: torch.Tensor,
    body_u: torch.Tensor,
    pmats: torch.Tensor,
    cam_centers: torch.Tensor,
    masks: torch.Tensor,
    init_scale: float,
    init_trans: list[float],
    head_start: float,
    dilation: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    roots_world = roots * float(init_scale) + torch.tensor(init_trans, device=roots.device, dtype=roots.dtype).view(1, 3)
    weights = []
    for view in range(masks.shape[0]):
        weights.append(
            projected_head_weight_mask(
                roots_world,
                normals,
                body_u,
                pmats[view],
                cam_centers[view],
                masks[view],
                head_start,
                dilation,
            )
        )
    stacked = torch.stack(weights, dim=0)
    stats: dict[str, float | int] = {
        "head_train_weight_views": int(stacked.shape[0]),
        "head_train_weight_coverage": float((stacked > 1e-5).float().mean().cpu()),
        "head_train_weight_mean": float(stacked.mean().cpu()),
        "head_train_weight_p95": float(torch.quantile(stacked.reshape(-1), 0.95).cpu()),
        "head_train_weight_max": float(stacked.max().cpu()),
        "head_train_weight_dilation": int(dilation),
    }
    return stacked.detach(), stats


def _tensor_image_to_pil(img: torch.Tensor, max_size: int) -> Image.Image:
    if max_size > 0 and max(int(img.shape[-2]), int(img.shape[-1])) > max_size:
        scale = max_size / float(max(int(img.shape[-2]), int(img.shape[-1])))
        new_h = max(1, int(round(int(img.shape[-2]) * scale)))
        new_w = max(1, int(round(int(img.shape[-1]) * scale)))
        img = F.interpolate(img[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
    if img.shape[0] == 1:
        out = img[0].numpy()
        return Image.fromarray((out * 255).astype(np.uint8))
    out = img.permute(1, 2, 0).numpy()
    return Image.fromarray((out * 255).astype(np.uint8))


def _normalize_debug_map(img: torch.Tensor) -> torch.Tensor:
    lo = img.amin(dim=(-2, -1), keepdim=True)
    hi = img.amax(dim=(-2, -1), keepdim=True)
    return (img - lo) / torch.clamp(hi - lo, min=1e-6)


def save_debug_contact_sheet(path: Path, images: dict[str, Image.Image], tile: int = 180, cols: int = 5) -> None:
    if not images:
        return
    names = list(images.keys())
    rows = int(math.ceil(len(names) / float(cols)))
    label_h = 22
    sheet = Image.new("RGB", (cols * tile, rows * (tile + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, name in enumerate(names):
        row = idx // cols
        col = idx % cols
        x = col * tile
        y = row * (tile + label_h)
        img = images[name].convert("RGB")
        scale = min(tile / max(img.width, 1), tile / max(img.height, 1))
        new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
        resample = Image.Resampling.NEAREST if scale > 1.0 else Image.Resampling.LANCZOS
        img = img.resize(new_size, resample)
        px = x + (tile - img.width) // 2
        py = y + label_h + (tile - img.height) // 2
        draw.text((x + 6, y + 4), name, fill=(0, 0, 0))
        sheet.paste(img, (px, py))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=95)


def save_texture_debug(path: Path, texture: torch.Tensor, max_size: int, contact_path: Path | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tex = texture.detach().cpu()[0]
    channels = {
        "coverage": torch.sigmoid(tex[0:1]),
        "density": torch.sigmoid(tex[1:2]),
        "length": torch.sigmoid(tex[2:3]),
        "root_width_rel": _normalize_debug_map(F.softplus(tex[3:4])),
        "tip_width_rel": _normalize_debug_map(F.softplus(tex[4:5])),
        "flow_x": (torch.tanh(tex[5:6]) + 1) * 0.5,
        "flow_y": (torch.tanh(tex[6:7]) + 1) * 0.5,
        "lift": torch.sigmoid(tex[7:8]),
        "sag": torch.sigmoid(tex[8:9]),
        "bend": torch.sigmoid(tex[9:10]),
        "stiffness": torch.sigmoid(tex[10:11]),
        "darkness": torch.sigmoid(tex[17:18]),
        "flow_conf": tex[18:19].clamp(0.0, 1.0),
        "detail_evidence": tex[19:20].clamp(0.0, 1.0),
        "root_rgb": torch.sigmoid(tex[11:14]),
    }
    contact_images: dict[str, Image.Image] = {}
    for name, img in channels.items():
        pil = _tensor_image_to_pil(img, max_size)
        pil.save(path / f"{name}.png")
        contact_images[name] = pil
    save_debug_contact_sheet(path / "texture_contact_sheet.png", contact_images)
    if contact_path is not None:
        save_debug_contact_sheet(contact_path, contact_images)


def save_groom_stats(path: Path, groom) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def stats(name: str, value: torch.Tensor) -> dict[str, float]:
        v = value.detach().reshape(-1).float().cpu()
        return {
            f"{name}_mean": float(v.mean()),
            f"{name}_p05": float(torch.quantile(v, 0.05)),
            f"{name}_p50": float(torch.quantile(v, 0.50)),
            f"{name}_p95": float(torch.quantile(v, 0.95)),
            f"{name}_min": float(v.min()),
            f"{name}_max": float(v.max()),
        }

    out: dict[str, float] = {}
    for name, value in {
        "coverage": groom.coverage,
        "density": groom.density,
        "alpha": groom.alpha,
        "length": groom.length,
        "root_width": groom.root_width,
        "tip_width": groom.tip_width,
    }.items():
        out.update(stats(name, value))
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def save_surface_texture_debug(path: Path, texture: torch.Tensor | None, max_size: int) -> None:
    if texture is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    tex = texture.detach().cpu()[0]
    channels = {
        "surface_rgb": torch.sigmoid(tex[0:3]),
        "surface_alpha": torch.sigmoid(tex[3:4]),
    }
    for name, img in channels.items():
        if max_size > 0 and max(int(img.shape[-2]), int(img.shape[-1])) > max_size:
            scale = max_size / float(max(int(img.shape[-2]), int(img.shape[-1])))
            new_h = max(1, int(round(int(img.shape[-2]) * scale)))
            new_w = max(1, int(round(int(img.shape[-1]) * scale)))
            img = F.interpolate(img[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]
        if img.shape[0] == 1:
            out = img[0].numpy()
            Image.fromarray((out * 255).astype(np.uint8)).save(path / f"{name}.png")
        else:
            out = img.permute(1, 2, 0).numpy()
            Image.fromarray((out * 255).astype(np.uint8)).save(path / f"{name}.png")


def save_uv_coverage_debug(path: Path, uv: np.ndarray, tex_h: int, tex_w: int, max_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(tex_h), int(tex_w)
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        h = max(1, int(round(h * scale)))
        w = max(1, int(round(w * scale)))
    x = np.clip(np.round(uv[:, 0] * (w - 1)).astype(np.int64), 0, w - 1)
    y = np.clip(np.round((1.0 - uv[:, 1]) * (h - 1)).astype(np.int64), 0, h - 1)
    canvas = np.zeros((h, w), dtype=np.float32)
    np.add.at(canvas, (y, x), 1.0)
    canvas = np.log1p(canvas)
    if canvas.max() > 0:
        canvas = canvas / canvas.max()
    Image.fromarray((canvas * 255.0).astype(np.uint8)).save(path)


def save_uv_root_overlay_debug(
    path: Path,
    uv: torch.Tensor,
    body_u: torch.Tensor,
    values: dict[str, torch.Tensor],
    tex_h: int,
    tex_w: int,
    max_size: int,
    head_start: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(tex_h), int(tex_w)
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        h = max(1, int(round(h * scale)))
        w = max(1, int(round(w * scale)))
    with torch.no_grad():
        uv_cpu = uv.detach().float().cpu().reshape(-1, 2)
        body_cpu = body_u.detach().float().cpu().reshape(-1)
        if uv_cpu.numel() == 0:
            return
        x = torch.clamp(torch.floor(uv_cpu[:, 0] * max(w - 1, 1)).long(), 0, w - 1)
        y = torch.clamp(torch.floor((1.0 - uv_cpu[:, 1]) * max(h - 1, 1)).long(), 0, h - 1)
        lin = y * w + x
        one = torch.ones(lin.shape[0], 1)
        head = (body_cpu >= float(head_start)).float().view(-1, 1)

        def count_image(mask: torch.Tensor) -> Image.Image:
            acc = torch.zeros(h * w, 1)
            acc.scatter_add_(0, lin[:, None], mask)
            img = torch.log1p(acc.view(h, w))
            if float(img.max()) > 0:
                img = img / img.max()
            arr = _heat_color(img.numpy())
            arr[img.numpy() <= 0.0] = 0
            return Image.fromarray(arr)

        def average_image(value: torch.Tensor) -> Image.Image:
            val = value.detach().float().cpu().reshape(-1, 1).clamp(0.0, 1.0)
            acc = torch.zeros(h * w, 1)
            cnt = torch.zeros(h * w, 1)
            acc.scatter_add_(0, lin[:, None], val)
            cnt.scatter_add_(0, lin[:, None], one)
            img = torch.zeros(h * w, 1)
            known = cnt[:, 0] > 0
            img[known] = acc[known] / cnt[known].clamp_min(1.0)
            img2 = img.view(h, w)
            arr = _heat_color(img2.numpy())
            arr[cnt.view(h, w).numpy() <= 0.0] = 0
            return Image.fromarray(arr)

        images: dict[str, Image.Image] = {
            "all_roots": count_image(one),
            "head_roots": count_image(head),
        }
        for name, value in values.items():
            images[name] = average_image(value)
    save_debug_contact_sheet(path, images, tile=180, cols=4)


def save_root_scalar_uv_debug(
    path: Path,
    uv: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
    tex_h: int,
    tex_w: int,
    max_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(tex_h), int(tex_w)
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        h = max(1, int(round(h * scale)))
        w = max(1, int(round(w * scale)))
    with torch.no_grad():
        uv_cpu = uv.detach().cpu()
        values_cpu = values.detach().cpu().reshape(-1, 1).clamp(0.0, 1.0)
        weights_cpu = weights.detach().cpu().reshape(-1, 1).clamp(0.0, 1.0)
        x = torch.clamp(torch.round(uv_cpu[:, 0] * (w - 1)).long(), 0, w - 1)
        y = torch.clamp(torch.round((1.0 - uv_cpu[:, 1]) * (h - 1)).long(), 0, h - 1)
        lin = y * w + x
        acc = torch.zeros(h * w, 1)
        cnt = torch.zeros(h * w, 1)
        acc.scatter_add_(0, lin[:, None], values_cpu * weights_cpu)
        cnt.scatter_add_(0, lin[:, None], weights_cpu)
        img = torch.zeros(h * w, 1)
        known = cnt[:, 0] > 1e-6
        img[known] = acc[known] / cnt[known].clamp_min(1e-6)
        img = img.view(h, w).numpy()
    Image.fromarray((img * 255.0).astype(np.uint8)).save(path)


def _heat_color(value: np.ndarray) -> np.ndarray:
    v = np.clip(value.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 * v, 0.0, 1.0)
    g = np.clip(1.5 * (1.0 - np.abs(v - 0.55) / 0.55), 0.0, 1.0)
    b = np.clip(1.5 * (1.0 - v), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def save_root_projection_debug(
    path: Path,
    images: torch.Tensor,
    roots_world: torch.Tensor,
    pmats: torch.Tensor,
    values: dict[str, torch.Tensor],
    view_ids: list[int],
    max_roots: int = 25000,
    tile_w: int = 220,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        return
    root_count = int(roots_world.shape[0])
    if root_count == 0:
        return
    if root_count > max_roots:
        idx = torch.linspace(0, root_count - 1, max_roots, device=roots_world.device).round().long()
    else:
        idx = torch.arange(root_count, device=roots_world.device)
    points = roots_world[idx]
    value_cpu = {
        name: val.detach().reshape(-1)[idx].float().clamp(0.0, 1.0).cpu().numpy()
        for name, val in values.items()
    }
    cols = len(value_cpu) + 1
    label_h = 22
    rows = len(view_ids)
    tile_h = int(round(tile_w * images.shape[1] / max(images.shape[2], 1)))
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for row, view in enumerate(view_ids):
        xy, z = project_points(points, pmats[view])
        h, w = int(images.shape[1]), int(images.shape[2])
        xy_cpu = xy.detach().cpu().numpy()
        z_cpu = z.detach().cpu().numpy()
        valid = (
            (z_cpu > 1e-5)
            & (xy_cpu[:, 0] >= 0.0)
            & (xy_cpu[:, 0] <= w - 1)
            & (xy_cpu[:, 1] >= 0.0)
            & (xy_cpu[:, 1] <= h - 1)
        )
        base = Image.fromarray((images[view].detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)).convert("RGB")
        y0 = row * (tile_h + label_h)
        draw.text((6, y0 + 4), f"view_{view:02d} target", fill=(0, 0, 0))
        target_tile = base.copy()
        target_tile.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(target_tile, ((tile_w - target_tile.width) // 2, y0 + label_h + (tile_h - target_tile.height) // 2))
        sx = tile_w / max(w, 1)
        sy = tile_h / max(h, 1)
        for col, (name, scalar) in enumerate(value_cpu.items(), start=1):
            tile = base.copy().resize((tile_w, tile_h), Image.Resampling.BILINEAR)
            overlay = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            colors = _heat_color(scalar)
            xs = np.round(xy_cpu[:, 0] * sx).astype(np.int32)
            ys = np.round(xy_cpu[:, 1] * sy).astype(np.int32)
            for i in np.where(valid)[0]:
                c = tuple(int(x) for x in colors[i]) + (150,)
                x = int(xs[i])
                y = int(ys[i])
                od.rectangle((x - 1, y - 1, x + 1, y + 1), fill=c)
            tile = Image.alpha_composite(tile.convert("RGBA"), overlay).convert("RGB")
            x0 = col * tile_w
            draw.text((x0 + 6, y0 + 4), f"view_{view:02d} {name}", fill=(0, 0, 0))
            sheet.paste(tile, (x0, y0 + label_h))
    sheet.save(path, quality=95)


def sample_texture_at_roots(texture: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    grid = uv.clone() * 2.0 - 1.0
    grid = grid.view(1, -1, 1, 2)
    return F.grid_sample(
        texture,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, :, :, 0].T


def save_orientation_debug(path: Path, orientation: torch.Tensor, confidence: torch.Tensor) -> None:
    path.mkdir(parents=True, exist_ok=True)
    orient = (orientation.detach().cpu().clamp(-1.0, 1.0).numpy() * 0.5 + 0.5)
    conf = confidence.detach().cpu().numpy()
    conf_vis = conf / max(float(np.quantile(conf, 0.98)), 1e-6)
    conf_vis = np.clip(conf_vis, 0.0, 1.0)
    Image.fromarray((orient * 255.0).astype(np.uint8)).save(path / "target_orientation.png")
    Image.fromarray((conf_vis[..., 0] * 255.0).astype(np.uint8)).save(path / "target_orientation_confidence.png")


def save_flow_orientation_overlay(
    path: Path,
    image: torch.Tensor,
    strand_points: torch.Tensor,
    pmat: torch.Tensor,
    target_orientation: torch.Tensor,
    target_confidence: torch.Tensor,
    width: int,
    height: int,
    min_confidence: float,
    max_segments: int = 12000,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if strand_points.shape[1] < 2:
        return
    p0 = strand_points[:, :-1, :].reshape(-1, 3)
    p1 = strand_points[:, 1:, :].reshape(-1, 3)
    total = int(p0.shape[0])
    if total == 0:
        return
    if total > max_segments:
        ids = torch.linspace(0, total - 1, max_segments, device=p0.device).round().long()
        p0 = p0[ids]
        p1 = p1[ids]
    xy0, z0 = project_points(p0, pmat)
    xy1, z1 = project_points(p1, pmat)
    mid = 0.5 * (xy0 + xy1)
    screen_vec = xy1 - xy0
    screen_len = torch.linalg.norm(screen_vec, dim=-1)
    screen_dir = screen_vec / torch.clamp(screen_len[:, None], min=1e-6)
    sampled_orientation = torch_normalize(bilinear_sample_image(target_orientation, mid))
    sampled_confidence = bilinear_sample_image(target_confidence, mid)[:, 0]
    valid = (
        (z0 > 1e-5)
        & (z1 > 1e-5)
        & (xy0[:, 0] >= 0.0)
        & (xy0[:, 0] <= width - 1)
        & (xy0[:, 1] >= 0.0)
        & (xy0[:, 1] <= height - 1)
        & (xy1[:, 0] >= 0.0)
        & (xy1[:, 0] <= width - 1)
        & (xy1[:, 1] >= 0.0)
        & (xy1[:, 1] <= height - 1)
        & (screen_len > 0.25)
        & (sampled_confidence >= min_confidence)
    )
    dot = (screen_dir * sampled_orientation).sum(dim=-1).clamp(-1.0, 1.0)
    err = (1.0 - dot.square()).clamp(0.0, 1.0)
    xy0_np = xy0.detach().cpu().numpy()
    xy1_np = xy1.detach().cpu().numpy()
    err_np = err.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()
    conf_np = sampled_confidence.detach().cpu().numpy()
    base = Image.fromarray((image.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for i in np.where(valid_np)[0]:
        e = float(err_np[i])
        conf = float(np.clip(conf_np[i], 0.0, 1.0))
        # green means aligned, red means perpendicular to the target 2D flow.
        color = (
            int(255 * e),
            int(220 * (1.0 - e)),
            35,
            int(80 + 140 * conf),
        )
        draw.line(
            [
                (float(xy0_np[i, 0]), float(xy0_np[i, 1])),
                (float(xy1_np[i, 0]), float(xy1_np[i, 1])),
            ],
            fill=color,
            width=1,
        )
    Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB").save(path / "predicted_flow_error.png", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/ssdwork/liuhaohan/petsgaussianhair")
    parser.add_argument("--mesh", default="data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj")
    parser.add_argument("--data", default="data/neuralfur_work/whiteTiger_processed/roaringwalk")
    parser.add_argument("--out", default="outputs/white_tiger_uv_groom/train")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--no-init-checkpoint-roots", action="store_true")
    parser.add_argument("--no-init-checkpoint-model", action="store_true")
    parser.add_argument("--roots", type=int, default=30000)
    parser.add_argument("--extra-roots", type=int, default=0)
    parser.add_argument(
        "--extra-root-source",
        choices=["base", "head_detail", "boundary", "head_boundary", "residual", "combined"],
        default="head_boundary",
    )
    parser.add_argument("--extra-root-boost", type=float, default=2.0)
    parser.add_argument("--extra-root-gamma", type=float, default=0.75)
    parser.add_argument("--extra-root-weight-cap", type=float, default=0.0)
    parser.add_argument("--extra-root-boundary-dilation", type=int, default=17)
    parser.add_argument("--extra-root-min-orient-confidence", type=float, default=0.04)
    parser.add_argument("--surface-roots", type=int, default=0)
    parser.add_argument("--root-sampling", choices=["area", "view_importance"], default="area")
    parser.add_argument("--root-distribution", choices=["random", "stratified"], default="random")
    parser.add_argument("--root-importance-boost", type=float, default=2.5)
    parser.add_argument("--root-importance-gamma", type=float, default=0.75)
    parser.add_argument("--root-residual-map", default="")
    parser.add_argument("--root-residual-boost", type=float, default=2.0)
    parser.add_argument("--root-residual-gamma", type=float, default=0.75)
    parser.add_argument("--no-root-residual-base", action="store_true")
    parser.add_argument("--root-orient-boost", type=float, default=0.0)
    parser.add_argument("--root-orient-gamma", type=float, default=0.75)
    parser.add_argument("--root-orient-min-confidence", type=float, default=0.04)
    parser.add_argument("--root-head-detail-boost", type=float, default=0.0)
    parser.add_argument("--root-head-detail-gamma", type=float, default=0.75)
    parser.add_argument("--root-head-detail-start", type=float, default=0.70)
    parser.add_argument("--root-head-detail-sharpness", type=float, default=28.0)
    parser.add_argument("--root-head-orient-mix", type=float, default=0.35)
    parser.add_argument("--root-head-detail-min-orient-confidence", type=float, default=0.04)
    parser.add_argument("--root-boundary-boost", type=float, default=0.0)
    parser.add_argument("--root-boundary-gamma", type=float, default=0.75)
    parser.add_argument("--root-boundary-dilation", type=int, default=17)
    parser.add_argument("--head-root-boost", type=float, default=0.0)
    parser.add_argument("--head-root-start", type=float, default=0.70)
    parser.add_argument("--head-root-sharpness", type=float, default=28.0)
    parser.add_argument("--curve-samples", type=int, default=24)
    parser.add_argument("--render-samples", type=int, default=12)
    parser.add_argument("--adaptive-render-samples", action="store_true")
    parser.add_argument("--adaptive-min-render-samples", type=int, default=6)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--uv-mode", choices=["xatlas", "obj", "cylindrical"], default="xatlas")
    parser.add_argument("--uv-cache", default="")
    parser.add_argument("--tex-h", type=int, default=4096)
    parser.add_argument("--tex-w", type=int, default=4096)
    parser.add_argument("--texture-debug-max", type=int, default=1024)
    parser.add_argument("--use-triplane", action="store_true")
    parser.add_argument("--triplane-h", type=int, default=128)
    parser.add_argument("--triplane-w", type=int, default=128)
    parser.add_argument("--triplane-scale", type=float, default=0.75)
    parser.add_argument("--use-coarse-texture", action="store_true")
    parser.add_argument("--coarse-h", type=int, default=512)
    parser.add_argument("--coarse-w", type=int, default=512)
    parser.add_argument("--coarse-scale", type=float, default=0.45)
    parser.add_argument("--no-coarse-init-split", action="store_true")
    parser.add_argument("--use-head-atlas", action="store_true")
    parser.add_argument("--head-h", type=int, default=128)
    parser.add_argument("--head-w", type=int, default=128)
    parser.add_argument("--head-start", type=float, default=0.70)
    parser.add_argument("--head-scale", type=float, default=1.0)
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--coarse-lr-scale", type=float, default=1.0)
    parser.add_argument("--head-lr-scale", type=float, default=1.0)
    parser.add_argument("--triplane-lr-scale", type=float, default=1.0)
    parser.add_argument("--surface-lr-scale", type=float, default=1.0)
    parser.add_argument("--alpha-scale", type=float, default=0.35)
    parser.add_argument("--alpha-cap", type=float, default=0.35)
    parser.add_argument("--surface-alpha-scale", type=float, default=0.0)
    parser.add_argument("--random-mesh-backing-weight", type=float, default=0.0)
    parser.add_argument("--random-mesh-backing-start-iter", type=int, default=0)
    parser.add_argument("--random-mesh-backing-warmup-iters", type=int, default=0)
    parser.add_argument("--random-mesh-backing-strength", type=float, default=1.0)
    parser.add_argument("--random-mesh-backing-alpha-scale", type=float, default=0.75)
    parser.add_argument("--random-mesh-backing-normal-offset", type=float, default=0.004)
    parser.add_argument("--root-surface-move-lr", type=float, default=0.0)
    parser.add_argument("--root-surface-move-start-iter", type=int, default=0)
    parser.add_argument("--root-surface-move-reg-weight", type=float, default=0.0)
    parser.add_argument("--root-surface-move-logit-limit", type=float, default=10.0)
    parser.add_argument("--root-densify-start-iter", type=int, default=0)
    parser.add_argument("--root-densify-stop-iter", type=int, default=0)
    parser.add_argument("--root-densify-interval", type=int, default=0)
    parser.add_argument("--root-densify-count", type=int, default=0)
    parser.add_argument("--root-densify-max-roots", type=int, default=0)
    parser.add_argument("--root-densify-boost", type=float, default=4.0)
    parser.add_argument("--root-densify-gamma", type=float, default=0.75)
    parser.add_argument("--root-densify-static-mix", type=float, default=0.25)
    parser.add_argument("--root-densify-min-weight", type=float, default=1e-4)
    parser.add_argument("--mesh-graph-smooth-weight", type=float, default=0.0)
    parser.add_argument("--mesh-graph-smooth-start-iter", type=int, default=0)
    parser.add_argument("--mesh-graph-smooth-warmup-iters", type=int, default=0)
    parser.add_argument("--mesh-graph-smooth-edges", type=int, default=60000)
    parser.add_argument("--splat-radius", type=float, default=0.90)
    parser.add_argument("--splat-mode", choices=["point", "oriented"], default="point")
    parser.add_argument("--tangent-radius-scale", type=float, default=1.8)
    parser.add_argument("--normal-radius-scale", type=float, default=0.65)
    parser.add_argument("--strand-width-radius", action="store_true")
    parser.add_argument("--radius-width-ref", type=float, default=0.0025)
    parser.add_argument("--radius-width-min-scale", type=float, default=0.45)
    parser.add_argument("--radius-width-max-scale", type=float, default=1.8)
    parser.add_argument("--depth-band", type=float, default=0.08)
    parser.add_argument("--depth-sharpness", type=float, default=0.0)
    parser.add_argument("--pose-lr", type=float, default=0.0005)
    parser.add_argument("--freeze-pose", action="store_true")
    parser.add_argument("--edge-loss-weight", type=float, default=2.0)
    parser.add_argument("--dark-loss-weight", type=float, default=1.5)
    parser.add_argument("--grad-loss-weight", type=float, default=0.0)
    parser.add_argument("--head-color-loss-weight", type=float, default=0.0)
    parser.add_argument("--head-mask-loss-weight", type=float, default=0.0)
    parser.add_argument("--head-loss-start-iter", type=int, default=0)
    parser.add_argument("--head-loss-warmup-iters", type=int, default=0)
    parser.add_argument("--head-loss-dilation", type=int, default=13)
    parser.add_argument("--boundary-mask-loss-weight", type=float, default=0.0)
    parser.add_argument("--boundary-loss-start-iter", type=int, default=300)
    parser.add_argument("--boundary-loss-warmup-iters", type=int, default=1000)
    parser.add_argument("--boundary-loss-dilation", type=int, default=17)
    parser.add_argument("--flow-orient-weight", type=float, default=0.0)
    parser.add_argument("--flow-orient-start-iter", type=int, default=0)
    parser.add_argument("--flow-orient-warmup-iters", type=int, default=0)
    parser.add_argument("--flow-orient-end-iter", type=int, default=0)
    parser.add_argument("--flow-orient-decay-iters", type=int, default=0)
    parser.add_argument("--flow-orient-source", choices=["rgb", "alpha", "gabor", "map"], default="rgb")
    parser.add_argument("--flow-orient-min-confidence", type=float, default=0.015)
    parser.add_argument("--flow-orient-max-segments", type=int, default=60000)
    parser.add_argument("--flow-init-source", choices=["none", "rgb", "alpha", "gabor", "map"], default="none")
    parser.add_argument("--flow-init-min-confidence", type=float, default=0.04)
    parser.add_argument("--flow-init-scale", type=float, default=0.55)
    parser.add_argument("--flow-init-probe-length", type=float, default=0.03)
    parser.add_argument("--flow-hint-prior-weight", type=float, default=0.0)
    parser.add_argument("--flow-hint-prior-start-iter", type=int, default=300)
    parser.add_argument("--flow-hint-prior-warmup-iters", type=int, default=1000)
    parser.add_argument("--flow-hint-prior-end-iter", type=int, default=0)
    parser.add_argument("--flow-hint-prior-decay-iters", type=int, default=0)
    parser.add_argument("--flow-hint-prior-min-confidence", type=float, default=0.04)
    parser.add_argument("--freeze-flow-after-iter", type=int, default=0)
    parser.add_argument("--flow-coherence-weight", type=float, default=0.0)
    parser.add_argument("--flow-coherence-start-iter", type=int, default=300)
    parser.add_argument("--flow-coherence-warmup-iters", type=int, default=1000)
    parser.add_argument("--flow-coherence-detail-relax", type=float, default=0.75)
    parser.add_argument("--flow-coherence-min-weight", type=float, default=0.25)
    parser.add_argument("--orientation-dir", default="orientations_2")
    parser.add_argument("--orientation-angle-bins", type=int, default=180)
    parser.add_argument("--detail-init-strength", type=float, default=0.0)
    parser.add_argument("--detail-density-boost", type=float, default=0.6)
    parser.add_argument("--detail-length-boost", type=float, default=0.15)
    parser.add_argument("--detail-root-width-boost", type=float, default=0.0)
    parser.add_argument("--detail-tip-width-boost", type=float, default=0.0)
    parser.add_argument("--detail-tv-relax", type=float, default=0.0)
    parser.add_argument("--detail-tv-min-weight", type=float, default=0.35)
    parser.add_argument("--texture-tv-weight", type=float, default=0.025)
    parser.add_argument("--asset-param-smooth-weight", type=float, default=0.0)
    parser.add_argument("--groom-geometry-weight", type=float, default=0.0)
    parser.add_argument("--groom-length-floor", type=float, default=0.48)
    parser.add_argument("--groom-detail-length-boost", type=float, default=0.12)
    parser.add_argument("--groom-head-length-boost", type=float, default=0.10)
    parser.add_argument("--groom-boundary-length-boost", type=float, default=0.0)
    parser.add_argument("--groom-boundary-density-floor", type=float, default=0.0)
    parser.add_argument("--groom-boundary-lift-floor", type=float, default=0.0)
    parser.add_argument("--groom-root-width-target", type=float, default=0.0044)
    parser.add_argument("--groom-tip-width-target", type=float, default=0.00115)
    parser.add_argument("--groom-max-tip-root-ratio", type=float, default=0.45)
    parser.add_argument("--gabor-orient-bins", type=int, default=36)
    parser.add_argument("--gabor-orient-dog-low", type=float, default=0.4)
    parser.add_argument("--gabor-orient-dog-high", type=float, default=10.0)
    parser.add_argument("--gabor-orient-sigma-x", type=float, default=1.8)
    parser.add_argument("--gabor-orient-sigma-y", type=float, default=2.4)
    parser.add_argument("--gabor-orient-frequency", type=float, default=0.23)
    parser.add_argument("--gabor-orient-chunk", type=int, default=4)
    parser.add_argument("--init-scale", type=float, default=1.25)
    parser.add_argument("--init-trans", type=float, nargs=3, default=[0.0, 0.32, 0.0])
    parser.add_argument("--no-backproject-init", action="store_true")
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--save-every", type=int, default=200)
    args = parser.parse_args()
    if args.random_mesh_backing_weight > 0.0 and args.surface_roots <= 0:
        raise ValueError("--random-mesh-backing-weight requires --surface-roots > 0")
    if args.root_surface_move_lr > 0.0 and args.uv_mode not in {"xatlas", "obj"}:
        raise ValueError("--root-surface-move-lr currently requires --uv-mode xatlas or obj")
    if args.root_densify_interval > 0 and args.root_densify_count <= 0:
        raise ValueError("--root-densify-interval requires --root-densify-count > 0")
    if args.root_densify_count > 0 and args.root_densify_interval <= 0:
        raise ValueError("--root-densify-count requires --root-densify-interval > 0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    project_root = Path(args.project_root)
    out = project_root / args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh_path = project_root / args.mesh
    vertices, faces, obj_uv_vertices, obj_face_uvs = load_obj_mesh_with_uv(mesh_path)
    face_neighbors = build_face_adjacency(faces)
    uv_vertices: np.ndarray | None = None
    face_uvs: np.ndarray | None = None
    if args.uv_mode == "obj":
        if obj_uv_vertices is None or obj_face_uvs is None:
            raise RuntimeError(f"uv-mode=obj requested but {mesh_path} has no OBJ vt/f-vt data")
        uv_vertices, face_uvs = obj_uv_vertices, obj_face_uvs
    elif args.uv_mode == "xatlas":
        uv_cache = Path(args.uv_cache) if args.uv_cache else mesh_path.with_suffix(".xatlas_uv.npz")
        if not uv_cache.is_absolute():
            uv_cache = project_root / uv_cache
        uv_vertices, face_uvs = load_or_build_xatlas_uv(mesh_path, vertices, faces, uv_cache)
    init_checkpoint = None
    init_checkpoint_path = None
    init_checkpoint_info: dict[str, object] = {"enabled": False}
    if args.init_checkpoint:
        init_checkpoint_path = Path(args.init_checkpoint)
        if not init_checkpoint_path.is_absolute():
            init_checkpoint_path = project_root / init_checkpoint_path
        if not init_checkpoint_path.is_file():
            raise FileNotFoundError(f"init checkpoint does not exist: {init_checkpoint_path}")
        init_checkpoint = torch.load(init_checkpoint_path, map_location="cpu", weights_only=False)
        init_checkpoint_info = {
            "enabled": True,
            "path": str(init_checkpoint_path),
            "load_roots": not bool(args.no_init_checkpoint_roots),
            "load_model": not bool(args.no_init_checkpoint_model),
            "checkpoint_iter": int(init_checkpoint.get("iter", -1)),
        }

    data_root = project_root / args.data
    orientation_dir = Path(args.orientation_dir)
    images, masks = load_training_images(data_root, args.width, device)
    pmats = load_projection_mats(data_root, args.width, device)
    cam_centers = load_camera_centers(data_root, device)
    detail_weights = make_detail_weights(images, masks, args.edge_loss_weight, args.dark_loss_weight).detach()
    target_orientation = None
    target_orientation_confidence = None
    target_orientation_stats: dict[str, float | int | str] | None = None
    flow_init_orientation = None
    flow_init_orientation_confidence = None
    flow_init_target_stats: dict[str, float | int | str] | None = None
    if args.flow_orient_weight > 0.0:
        target_orientation, target_orientation_confidence, target_orientation_stats = build_target_orientation_fields(
            images,
            masks,
            args.flow_orient_source,
            data_root,
            orientation_dir,
            args.width,
            args.orientation_angle_bins,
            args.gabor_orient_bins,
            args.gabor_orient_dog_low,
            args.gabor_orient_dog_high,
            args.gabor_orient_sigma_x,
            args.gabor_orient_sigma_y,
            args.gabor_orient_frequency,
            args.gabor_orient_chunk,
        )
    if args.flow_init_source != "none":
        if (
            target_orientation is not None
            and target_orientation_confidence is not None
            and args.flow_init_source == args.flow_orient_source
        ):
            flow_init_orientation = target_orientation
            flow_init_orientation_confidence = target_orientation_confidence
            flow_init_target_stats = dict(target_orientation_stats or {})
            flow_init_target_stats["reused_flow_orientation_target"] = 1
        else:
            flow_init_orientation, flow_init_orientation_confidence, flow_init_target_stats = build_target_orientation_fields(
                images,
                masks,
                args.flow_init_source,
                data_root,
                orientation_dir,
                args.width,
                args.orientation_angle_bins,
                args.gabor_orient_bins,
                args.gabor_orient_dog_low,
                args.gabor_orient_dog_high,
                args.gabor_orient_sigma_x,
                args.gabor_orient_sigma_y,
                args.gabor_orient_frequency,
                args.gabor_orient_chunk,
            )
    height, width = images.shape[1], images.shape[2]

    face_sampling_weights = None
    root_sampling_stats: dict[str, float | str] = {"mode": args.root_sampling, "root_distribution": args.root_distribution}
    if args.root_sampling == "view_importance":
        face_sampling_weights, stats = compute_view_importance_face_weights(
            vertices,
            faces,
            args.init_scale,
            args.init_trans,
            images,
            masks,
            detail_weights,
            pmats,
            cam_centers,
            args.root_importance_boost,
            args.root_importance_gamma,
        )
        root_sampling_stats.update(stats)
    if args.head_root_boost > 0.0:
        centers_np, _, areas_np = face_geometry(vertices, faces)
        valid_faces = np.where(areas_np > 0)[0]
        axis_np = infer_axis(centers_np[valid_faces])
        axis_coord = centers_np @ axis_np
        axis_lo, axis_hi = np.quantile(axis_coord[valid_faces], [0.002, 0.998])
        face_body_u = np.clip((axis_coord - axis_lo) / max(float(axis_hi - axis_lo), EPS), 0.0, 1.0)
        gate = 1.0 / (1.0 + np.exp(-float(args.head_root_sharpness) * (face_body_u - float(args.head_root_start))))
        boost = 1.0 + float(args.head_root_boost) * gate
        if face_sampling_weights is None:
            face_sampling_weights = np.ones(faces.shape[0], dtype=np.float32)
        face_sampling_weights = face_sampling_weights.astype(np.float32) * boost.astype(np.float32)
        root_sampling_stats.update(
            {
                "head_root_boost": float(args.head_root_boost),
                "head_root_start": float(args.head_root_start),
                "head_root_gate_mean": float(gate[valid_faces].mean()),
                "head_root_gate_p95": float(np.quantile(gate[valid_faces], 0.95)),
                "head_root_weight_mean": float(face_sampling_weights[valid_faces].mean()),
                "head_root_weight_max": float(face_sampling_weights[valid_faces].max()),
            }
        )
    if args.root_orient_boost > 0.0:
        root_orient_confidence = flow_init_orientation_confidence
        if root_orient_confidence is None:
            root_orient_confidence = target_orientation_confidence
        if root_orient_confidence is None:
            raise RuntimeError(
                "--root-orient-boost requires an orientation confidence source. "
                "Enable --flow-init-source map/gabor or --flow-orient-weight with an orientation source."
            )
        orient_face_weights, stats = compute_orientation_confidence_face_weights(
            vertices,
            faces,
            args.init_scale,
            args.init_trans,
            masks,
            root_orient_confidence,
            pmats,
            cam_centers,
            args.root_orient_boost,
            args.root_orient_gamma,
            args.root_orient_min_confidence,
        )
        if face_sampling_weights is None:
            face_sampling_weights = np.ones(faces.shape[0], dtype=np.float32)
        face_sampling_weights = face_sampling_weights.astype(np.float32) * orient_face_weights
        root_sampling_stats.update(
            {
                "root_orient_boost": float(args.root_orient_boost),
                "root_orient_gamma": float(args.root_orient_gamma),
                "root_orient_min_confidence": float(args.root_orient_min_confidence),
                **stats,
            }
        )
    if args.root_head_detail_boost > 0.0:
        head_detail_confidence = flow_init_orientation_confidence
        if head_detail_confidence is None:
            head_detail_confidence = target_orientation_confidence
        head_detail_face_weights, stats = compute_head_detail_face_weights(
            vertices,
            faces,
            args.init_scale,
            args.init_trans,
            masks,
            detail_weights,
            head_detail_confidence,
            pmats,
            cam_centers,
            args.root_head_detail_boost,
            args.root_head_detail_gamma,
            args.root_head_detail_start,
            args.root_head_detail_sharpness,
            args.root_head_orient_mix,
            args.root_head_detail_min_orient_confidence,
        )
        if face_sampling_weights is None:
            face_sampling_weights = np.ones(faces.shape[0], dtype=np.float32)
        face_sampling_weights = face_sampling_weights.astype(np.float32) * head_detail_face_weights
        root_sampling_stats.update(stats)
    root_sampling_boundary_weights = None
    if args.root_boundary_boost > 0.0:
        root_sampling_boundary_weights = build_mask_boundary_weights(masks, args.root_boundary_dilation)
        boundary_face_weights, stats = compute_boundary_face_weights(
            vertices,
            faces,
            args.init_scale,
            args.init_trans,
            masks,
            root_sampling_boundary_weights,
            pmats,
            cam_centers,
            args.root_boundary_boost,
            args.root_boundary_gamma,
        )
        if face_sampling_weights is None:
            face_sampling_weights = np.ones(faces.shape[0], dtype=np.float32)
        face_sampling_weights = face_sampling_weights.astype(np.float32) * boundary_face_weights
        root_sampling_stats.update(
            {
                "root_boundary_dilation": int(args.root_boundary_dilation),
                **stats,
            }
        )
    if args.root_residual_map and not args.no_root_residual_base:
        residual_map = Path(args.root_residual_map)
        if not residual_map.is_absolute():
            residual_map = project_root / residual_map
        if not residual_map.exists():
            raise FileNotFoundError(f"root residual map does not exist: {residual_map}")
        residual_face_weights, stats = compute_residual_map_face_weights(
            residual_map,
            vertices,
            faces,
            args.uv_mode,
            uv_vertices,
            face_uvs,
            args.root_residual_boost,
            args.root_residual_gamma,
        )
        if face_sampling_weights is None:
            face_sampling_weights = np.ones(faces.shape[0], dtype=np.float32)
        face_sampling_weights = face_sampling_weights.astype(np.float32) * residual_face_weights
        root_sampling_stats.update(stats)

    if init_checkpoint is not None and not args.no_init_checkpoint_roots:
        root_data = rebuild_root_data_from_checkpoint(
            vertices,
            faces,
            args.uv_mode,
            uv_vertices,
            face_uvs,
            init_checkpoint["root_data"],
        )
        root_sampling_stats.update(
            {
                "init_checkpoint_root_count": int(root_data["roots"].shape[0]),
                "init_checkpoint_root_source": str(init_checkpoint_path),
            }
        )
    else:
        root_data = sample_full_body_roots(
            vertices,
            faces,
            args.roots,
            args.seed,
            args.uv_mode,
            uv_vertices,
            face_uvs,
            face_sampling_weights,
            args.root_distribution,
        )
    base_root_count_for_config = int(root_data["roots"].shape[0])
    extra_root_stats: dict[str, object] | None = None
    extra_sampled_root_stats: dict[str, float] | None = None
    if args.extra_roots > 0:
        extra_source = str(args.extra_root_source)
        extra_weights: np.ndarray | None = None
        extra_components: list[str] = []

        def multiply_extra_weights(weights: np.ndarray, name: str) -> None:
            nonlocal extra_weights
            clipped = np.clip(weights.astype(np.float32), 1e-4, None)
            extra_weights = clipped if extra_weights is None else extra_weights.astype(np.float32) * clipped
            extra_components.append(name)

        if extra_source in {"base", "combined"}:
            if face_sampling_weights is None:
                multiply_extra_weights(np.ones(faces.shape[0], dtype=np.float32), "base_area")
            else:
                multiply_extra_weights(face_sampling_weights.astype(np.float32), "base_sampling")

        if extra_source in {"head_detail", "head_boundary", "combined"}:
            extra_head_confidence = flow_init_orientation_confidence
            if extra_head_confidence is None:
                extra_head_confidence = target_orientation_confidence
            extra_head_weights, extra_head_stats = compute_head_detail_face_weights(
                vertices,
                faces,
                args.init_scale,
                args.init_trans,
                masks,
                detail_weights,
                extra_head_confidence,
                pmats,
                cam_centers,
                args.extra_root_boost,
                args.extra_root_gamma,
                args.root_head_detail_start,
                args.root_head_detail_sharpness,
                args.root_head_orient_mix,
                args.extra_root_min_orient_confidence,
            )
            multiply_extra_weights(extra_head_weights, "head_detail")

        if extra_source in {"boundary", "head_boundary", "combined"}:
            extra_boundary_map = root_sampling_boundary_weights
            if extra_boundary_map is None:
                extra_boundary_map = build_mask_boundary_weights(masks, args.extra_root_boundary_dilation)
            extra_boundary_weights, extra_boundary_stats = compute_boundary_face_weights(
                vertices,
                faces,
                args.init_scale,
                args.init_trans,
                masks,
                extra_boundary_map,
                pmats,
                cam_centers,
                args.extra_root_boost,
                args.extra_root_gamma,
            )
            multiply_extra_weights(extra_boundary_weights, "boundary")

        if extra_source in {"residual", "combined"} and args.root_residual_map:
            residual_map = Path(args.root_residual_map)
            if not residual_map.is_absolute():
                residual_map = project_root / residual_map
            if not residual_map.exists():
                raise FileNotFoundError(f"root residual map does not exist: {residual_map}")
            extra_residual_weights, extra_residual_stats = compute_residual_map_face_weights(
                residual_map,
                vertices,
                faces,
                args.uv_mode,
                uv_vertices,
                face_uvs,
                args.extra_root_boost,
                args.extra_root_gamma,
            )
            multiply_extra_weights(extra_residual_weights, "residual")
        elif extra_source == "residual":
            raise RuntimeError("--extra-root-source residual requires --root-residual-map")

        if extra_weights is None:
            raise RuntimeError(f"--extra-root-source {extra_source} did not produce any sampling weights")
        extra_weights = np.clip(extra_weights.astype(np.float32), 1e-4, None)
        extra_weight_uncapped_min = float(np.min(extra_weights))
        extra_weight_uncapped_mean = float(np.mean(extra_weights))
        extra_weight_uncapped_max = float(np.max(extra_weights))
        if args.extra_root_weight_cap > 0.0:
            extra_weights = np.minimum(extra_weights, float(args.extra_root_weight_cap)).astype(np.float32)
        extra_data = sample_full_body_roots(
            vertices,
            faces,
            args.extra_roots,
            args.seed + 1009,
            args.uv_mode,
            uv_vertices,
            face_uvs,
            extra_weights,
            args.root_distribution,
        )
        extra_sampled_root_stats = summarize_sampled_roots(extra_data, args.head_root_start)
        root_data = concat_root_data(root_data, extra_data)
        extra_root_stats = {
            "extra_roots": int(args.extra_roots),
            "extra_root_source": extra_source,
            "extra_root_components": extra_components,
            "extra_root_boost": float(args.extra_root_boost),
            "extra_root_gamma": float(args.extra_root_gamma),
            "extra_root_weight_cap": float(args.extra_root_weight_cap),
            "extra_root_weight_uncapped_min": extra_weight_uncapped_min,
            "extra_root_weight_uncapped_mean": extra_weight_uncapped_mean,
            "extra_root_weight_uncapped_max": extra_weight_uncapped_max,
            "extra_root_weight_min": float(np.min(extra_weights)),
            "extra_root_weight_mean": float(np.mean(extra_weights)),
            "extra_root_weight_max": float(np.max(extra_weights)),
            "extra_sampled_roots": extra_sampled_root_stats,
        }
    sampled_root_stats = summarize_sampled_roots(root_data, args.head_root_start)
    uv_atlas_stats = compute_uv_atlas_stats(
        vertices,
        faces,
        args.uv_mode,
        uv_vertices,
        face_uvs,
        args.tex_h,
        args.tex_w,
        args.root_head_detail_start,
    )
    root_uv_stats = summarize_root_uv_coverage(
        root_data["uv"],
        root_data["body_u"],
        args.tex_h,
        args.tex_w,
        args.root_head_detail_start,
    )
    coarse_uv_stats = None
    if args.use_coarse_texture:
        coarse_uv_stats = summarize_root_uv_coverage(
            root_data["uv"],
            root_data["body_u"],
            args.coarse_h,
            args.coarse_w,
            args.root_head_detail_start,
        )
    roots = torch.tensor(root_data["roots"], device=device)
    normals = torch.tensor(root_data["normals"], device=device)
    tangents = torch.tensor(root_data["tangents"], device=device)
    bitangents = torch.tensor(root_data["bitangents"], device=device)
    uv = torch.tensor(root_data["uv"], device=device)
    coord = torch.tensor(root_data["coord"], device=device)
    body_u = torch.tensor(root_data["body_u"], device=device)
    root_base_roots = roots
    root_base_uv = uv
    root_base_coord = coord
    root_base_body_u = body_u
    root_base_bary = torch.tensor(root_data["bary"], device=device)
    root_face_ids = torch.tensor(root_data["face_ids"], device=device, dtype=torch.long)
    root_bary_logits = None
    root_surface_move_stats: dict[str, float | int | bool] = {
        "root_surface_move_enabled": bool(args.root_surface_move_lr > 0.0),
        "root_surface_move_active": False,
        "root_surface_move_rms": 0.0,
        "root_surface_move_p95": 0.0,
    }
    mesh_vertices_t = torch.tensor(vertices, device=device, dtype=torch.float32)
    mesh_faces_t = torch.tensor(faces, device=device, dtype=torch.long)
    root_tri = mesh_vertices_t[mesh_faces_t[root_face_ids]]
    root_uv_tri = None
    if args.uv_mode in {"xatlas", "obj"}:
        if uv_vertices is None or face_uvs is None:
            raise RuntimeError(f"uv_mode={args.uv_mode} requires UV triangles for root surface movement")
        uv_vertices_t = torch.tensor(uv_vertices, device=device, dtype=torch.float32)
        face_uvs_t = torch.tensor(face_uvs, device=device, dtype=torch.long)
        root_uv_tri = uv_vertices_t[face_uvs_t[root_face_ids]]
    root_bbox_min = torch.tensor(vertices.min(axis=0), device=device, dtype=torch.float32).view(1, 3)
    root_bbox_extent = torch.tensor(
        np.maximum(vertices.max(axis=0) - vertices.min(axis=0), EPS),
        device=device,
        dtype=torch.float32,
    ).view(1, 3)
    centers_for_axis, _, areas_for_axis = face_geometry(vertices, faces)
    candidates_for_axis = keep_largest_face_component(faces, np.where(areas_for_axis > 0)[0])
    root_axis_np = np.asarray(root_data["axis"], dtype=np.float32).reshape(3)
    axis_values = centers_for_axis[candidates_for_axis] @ root_axis_np
    root_axis_lo, root_axis_hi = np.quantile(axis_values, [0.002, 0.998])
    root_axis_t = torch.tensor(root_axis_np, device=device, dtype=torch.float32).view(3)
    root_axis_lo_t = torch.tensor(float(root_axis_lo), device=device, dtype=torch.float32)
    root_axis_extent_t = torch.tensor(max(float(root_axis_hi - root_axis_lo), EPS), device=device, dtype=torch.float32)
    tri_np = vertices[faces[np.asarray(root_data["face_ids"], dtype=np.int64)]]
    edge_scale_np = (
        np.linalg.norm(tri_np[:, 1] - tri_np[:, 0], axis=1)
        + np.linalg.norm(tri_np[:, 2] - tri_np[:, 1], axis=1)
        + np.linalg.norm(tri_np[:, 0] - tri_np[:, 2], axis=1)
    ) / 3.0
    root_edge_scale = torch.tensor(np.maximum(edge_scale_np, EPS), device=device, dtype=torch.float32).view(-1, 1)
    if args.root_surface_move_lr > 0.0:
        root_bary_logits = torch.nn.Parameter(torch.log(root_base_bary.clamp_min(1e-5)))

    def root_state_from_bary(bary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        roots_cur = (root_tri * bary[:, :, None]).sum(dim=1)
        if root_uv_tri is None:
            uv_cur = root_base_uv
        else:
            uv_cur = (root_uv_tri * bary[:, :, None]).sum(dim=1).remainder(1.0).clamp(0.0, 1.0)
        coord_cur = ((roots_cur - root_bbox_min) / root_bbox_extent).clamp(0.0, 1.0)
        axis_coord = roots_cur @ root_axis_t
        body_u_cur = ((axis_coord - root_axis_lo_t) / root_axis_extent_t).clamp(0.0, 1.0).view(-1, 1)
        return roots_cur, uv_cur, coord_cur, body_u_cur

    def current_root_bary(iteration: int | None = None) -> tuple[torch.Tensor, bool]:
        if root_bary_logits is None:
            return root_base_bary, False
        active = iteration is None or iteration >= int(args.root_surface_move_start_iter)
        if not active:
            return root_base_bary, False
        return torch.softmax(root_bary_logits, dim=-1), True

    def root_surface_movement_stats(roots_cur: torch.Tensor, active: bool) -> dict[str, float | int | bool]:
        delta = torch.linalg.norm(roots_cur - root_base_roots, dim=-1, keepdim=True) / root_edge_scale
        return {
            "root_surface_move_enabled": bool(root_bary_logits is not None),
            "root_surface_move_active": bool(active),
            "root_surface_move_start_iter": int(args.root_surface_move_start_iter),
            "root_surface_move_rms": float(torch.sqrt(torch.mean(torch.square(delta))).detach().cpu()),
            "root_surface_move_p95": float(torch.quantile(delta.reshape(-1), 0.95).detach().cpu()),
        }
    head_train_weights = None
    head_train_stats: dict[str, float | int] | None = None
    if args.head_color_loss_weight > 0.0 or args.head_mask_loss_weight > 0.0:
        head_train_weights, head_train_stats = build_head_view_weights(
            roots,
            normals,
            body_u,
            pmats,
            cam_centers,
            masks,
            args.init_scale,
            args.init_trans,
            args.root_head_detail_start,
            args.head_loss_dilation,
        )
    boundary_weights = None
    boundary_stats: dict[str, float | int] | None = None
    needs_boundary_weights = (
        args.boundary_mask_loss_weight > 0.0
        or args.groom_boundary_length_boost > 0.0
        or args.groom_boundary_density_floor > 0.0
        or args.groom_boundary_lift_floor > 0.0
    )
    if needs_boundary_weights:
        if (
            root_sampling_boundary_weights is not None
            and int(args.root_boundary_dilation) == int(args.boundary_loss_dilation)
        ):
            boundary_weights = root_sampling_boundary_weights
        else:
            boundary_weights = build_mask_boundary_weights(masks, args.boundary_loss_dilation)
        boundary_stats = {
            "boundary_weight_views": int(boundary_weights.shape[0]),
            "boundary_weight_coverage": float((boundary_weights > 1e-5).float().mean().cpu()),
            "boundary_weight_mean": float(boundary_weights.mean().cpu()),
            "boundary_weight_p95": float(torch.quantile(boundary_weights.reshape(-1), 0.95).cpu()),
            "boundary_loss_dilation": int(args.boundary_loss_dilation),
        }

    model = TextureGroom(
        args.tex_h,
        args.tex_w,
        args.init_scale,
        args.init_trans,
        args.use_triplane,
        args.triplane_h,
        args.triplane_w,
        args.triplane_scale,
        args.use_coarse_texture,
        args.coarse_h,
        args.coarse_w,
        args.coarse_scale,
        args.use_head_atlas,
        args.head_h,
        args.head_w,
        args.head_start,
        args.head_scale,
        args.surface_roots > 0 and args.surface_alpha_scale > 0.0,
    ).to(device)
    surface_data = None
    surface_roots = None
    surface_normals = None
    surface_uv = None
    surface_coord = None
    surface_body_u = None
    if args.surface_roots > 0:
        surface_data = sample_full_body_roots(
            vertices,
            faces,
            args.surface_roots,
            args.seed + 97,
            args.uv_mode,
            uv_vertices,
            face_uvs,
            face_sampling_weights,
            args.root_distribution,
        )
        surface_roots = torch.tensor(surface_data["roots"], device=device)
        surface_normals = torch.tensor(surface_data["normals"], device=device)
        surface_uv = torch.tensor(surface_data["uv"], device=device)
        surface_coord = torch.tensor(surface_data["coord"], device=device)
        surface_body_u = torch.tensor(surface_data["body_u"], device=device)
    flow_hint = None
    flow_hint_confidence = None
    flow_init_stats: dict[str, float | int | str] | None = None
    if (
        args.flow_init_source != "none"
        and flow_init_orientation is not None
        and flow_init_orientation_confidence is not None
    ):
        flow_hint, flow_hint_confidence, flow_init_stats = compute_root_flow_hints_from_orientation(
            roots,
            normals,
            tangents,
            bitangents,
            flow_init_orientation,
            flow_init_orientation_confidence,
            masks,
            pmats,
            cam_centers,
            torch.exp(model.log_scale).clamp(0.25, 4.0),
            model.trans,
            args.flow_init_min_confidence,
            args.flow_init_scale,
            args.flow_init_probe_length,
        )
    root_boundary_evidence = None
    root_boundary_stats: dict[str, float | int] | None = None
    if (
        boundary_weights is not None
        and (
            args.groom_boundary_length_boost > 0.0
            or args.groom_boundary_density_floor > 0.0
            or args.groom_boundary_lift_floor > 0.0
        )
    ):
        root_boundary_evidence, root_boundary_stats = compute_root_boundary_evidence(
            roots,
            normals,
            boundary_weights,
            masks,
            pmats,
            cam_centers,
            torch.exp(model.log_scale).clamp(0.25, 4.0),
            model.trans,
        )
    initial_mesh_graph_edges_np = None
    if args.mesh_graph_smooth_weight > 0.0 and args.mesh_graph_smooth_edges > 0:
        initial_mesh_graph_edges_np = build_root_graph_edges(
            np.asarray(root_data["face_ids"], dtype=np.int64),
            face_neighbors,
            int(args.mesh_graph_smooth_edges),
            args.seed + 8117,
        )
    initial_mesh_graph_edge_count = int(initial_mesh_graph_edges_np.shape[0]) if initial_mesh_graph_edges_np is not None else 0
    with (out / "run_config.json").open("w", encoding="utf-8") as cfg:
        json.dump(
            {
                "args": vars(args),
                "root_count": int(root_data["roots"].shape[0]),
                "base_root_count": int(base_root_count_for_config),
                "extra_root_count": int(args.extra_roots),
                "surface_root_count": int(args.surface_roots),
                "root_distribution": args.root_distribution,
                "init_checkpoint": init_checkpoint_info,
                "root_sampling": root_sampling_stats,
                "root_residual_base_enabled": bool(args.root_residual_map and not args.no_root_residual_base),
                "extra_root_sampling": extra_root_stats,
                "sampled_roots": sampled_root_stats,
                "uv_atlas": uv_atlas_stats,
                "root_uv_coverage": root_uv_stats,
                "coarse_uv_coverage": coarse_uv_stats,
                "optimizer": {
                    "lr": float(args.lr),
                    "coarse_lr_scale": float(args.coarse_lr_scale),
                    "head_lr_scale": float(args.head_lr_scale),
                    "triplane_lr_scale": float(args.triplane_lr_scale),
                    "surface_lr_scale": float(args.surface_lr_scale),
                    "pose_lr": float(args.pose_lr),
                },
                "root_surface_movement": {
                    "enabled": bool(args.root_surface_move_lr > 0.0),
                    "lr": float(args.root_surface_move_lr),
                    "start_iter": int(args.root_surface_move_start_iter),
                    "reg_weight": float(args.root_surface_move_reg_weight),
                    "logit_limit": float(args.root_surface_move_logit_limit),
                    "parameterization": "per-root barycentric softmax constrained to original mesh face",
                },
                "root_densification": {
                    "enabled": bool(args.root_densify_interval > 0 and args.root_densify_count > 0),
                    "start_iter": int(args.root_densify_start_iter),
                    "stop_iter": int(args.root_densify_stop_iter),
                    "interval": int(args.root_densify_interval),
                    "count": int(args.root_densify_count),
                    "max_roots": int(args.root_densify_max_roots),
                    "boost": float(args.root_densify_boost),
                    "gamma": float(args.root_densify_gamma),
                    "static_mix": float(args.root_densify_static_mix),
                    "min_weight": float(args.root_densify_min_weight),
                    "source": "held-out render residual over current root projections, optionally mixed with static image/detail sampling weights",
                },
                "mesh_graph_smooth": {
                    "weight": float(args.mesh_graph_smooth_weight),
                    "start_iter": int(args.mesh_graph_smooth_start_iter),
                    "warmup_iters": int(args.mesh_graph_smooth_warmup_iters),
                    "target_edges": int(args.mesh_graph_smooth_edges),
                    "actual_edges": int(initial_mesh_graph_edge_count),
                    "graph": "root pairs on same or adjacent mesh faces",
                },
                "flow_orientation": {
                    "weight": float(args.flow_orient_weight),
                    "start_iter": int(args.flow_orient_start_iter),
                    "warmup_iters": int(args.flow_orient_warmup_iters),
                    "end_iter": int(args.flow_orient_end_iter),
                    "decay_iters": int(args.flow_orient_decay_iters),
                    "source": args.flow_orient_source,
                    "min_confidence": float(args.flow_orient_min_confidence),
                    "max_segments": int(args.flow_orient_max_segments),
                    "target_stats": target_orientation_stats,
                },
                "flow_initialization": {
                    "source": args.flow_init_source,
                    "min_confidence": float(args.flow_init_min_confidence),
                    "scale": float(args.flow_init_scale),
                    "probe_length": float(args.flow_init_probe_length),
                    "hint_prior_weight": float(args.flow_hint_prior_weight),
                    "hint_prior_start_iter": int(args.flow_hint_prior_start_iter),
                    "hint_prior_warmup_iters": int(args.flow_hint_prior_warmup_iters),
                    "hint_prior_end_iter": int(args.flow_hint_prior_end_iter),
                    "hint_prior_decay_iters": int(args.flow_hint_prior_decay_iters),
                    "hint_prior_min_confidence": float(args.flow_hint_prior_min_confidence),
                    "freeze_after_iter": int(args.freeze_flow_after_iter),
                    "target_stats": flow_init_target_stats,
                    "hint_stats": flow_init_stats,
                },
                "flow_coherence": {
                    "weight": float(args.flow_coherence_weight),
                    "start_iter": int(args.flow_coherence_start_iter),
                    "warmup_iters": int(args.flow_coherence_warmup_iters),
                    "detail_relax": float(args.flow_coherence_detail_relax),
                    "min_weight": float(args.flow_coherence_min_weight),
                },
                "detail_initialization": {
                    "strength": float(args.detail_init_strength),
                    "density_boost": float(args.detail_density_boost),
                    "length_boost": float(args.detail_length_boost),
                    "root_width_boost": float(args.detail_root_width_boost),
                    "tip_width_boost": float(args.detail_tip_width_boost),
                    "tv_relax": float(args.detail_tv_relax),
                    "tv_min_weight": float(args.detail_tv_min_weight),
                },
                "groom_geometry_prior": {
                    "weight": float(args.groom_geometry_weight),
                    "length_floor": float(args.groom_length_floor),
                    "detail_length_boost": float(args.groom_detail_length_boost),
                    "head_length_boost": float(args.groom_head_length_boost),
                    "boundary_length_boost": float(args.groom_boundary_length_boost),
                    "boundary_density_floor": float(args.groom_boundary_density_floor),
                    "boundary_lift_floor": float(args.groom_boundary_lift_floor),
                    "root_boundary_stats": root_boundary_stats,
                    "root_width_target": float(args.groom_root_width_target),
                    "tip_width_target": float(args.groom_tip_width_target),
                    "max_tip_root_ratio": float(args.groom_max_tip_root_ratio),
                },
                "head_training": {
                    "color_loss_weight": float(args.head_color_loss_weight),
                    "mask_loss_weight": float(args.head_mask_loss_weight),
                    "start_iter": int(args.head_loss_start_iter),
                    "warmup_iters": int(args.head_loss_warmup_iters),
                    "dilation": int(args.head_loss_dilation),
                    "weight_stats": head_train_stats,
                },
                "boundary_training": {
                    "mask_loss_weight": float(args.boundary_mask_loss_weight),
                    "start_iter": int(args.boundary_loss_start_iter),
                    "warmup_iters": int(args.boundary_loss_warmup_iters),
                    "dilation": int(args.boundary_loss_dilation),
                    "weight_stats": boundary_stats,
                },
                "adaptive_rendering": {
                    "enabled": bool(args.adaptive_render_samples),
                    "curve_samples": int(args.curve_samples),
                    "max_render_samples": int(args.render_samples),
                    "min_render_samples": int(args.adaptive_min_render_samples),
                    "budget_rule": "0.65 * length_score + 0.35 * curve_complexity_score; each score blends absolute physical scale with within-run relative rank",
                    "length_range": [0.012, 0.105],
                    "complexity_scale": 0.35,
                },
            },
            cfg,
            indent=2,
        )
    print(json.dumps({"root_sampling": root_sampling_stats}), flush=True)
    if extra_root_stats is not None:
        print(json.dumps({"extra_root_sampling": extra_root_stats}), flush=True)
    print(json.dumps({"sampled_roots": sampled_root_stats}), flush=True)
    print(json.dumps({"uv_atlas": uv_atlas_stats}), flush=True)
    print(json.dumps({"root_uv_coverage": root_uv_stats}), flush=True)
    if coarse_uv_stats is not None:
        print(json.dumps({"coarse_uv_coverage": coarse_uv_stats}), flush=True)
    if target_orientation_stats is not None:
        print(json.dumps({"target_orientation": target_orientation_stats}), flush=True)
    if flow_init_target_stats is not None:
        print(json.dumps({"flow_init_target": flow_init_target_stats}), flush=True)
    if flow_init_stats is not None:
        print(json.dumps({"flow_init": flow_init_stats}), flush=True)
    if head_train_stats is not None:
        print(json.dumps({"head_training": head_train_stats}), flush=True)
    if boundary_stats is not None:
        print(json.dumps({"boundary_training": boundary_stats}), flush=True)
    if root_boundary_stats is not None:
        print(json.dumps({"root_boundary_evidence": root_boundary_stats}), flush=True)
    if not args.no_backproject_init:
        init_texture_from_images(
            model,
            roots,
            normals,
            uv,
            images,
            masks,
            detail_weights,
            pmats,
            cam_centers,
            flow_hint,
            flow_hint_confidence,
            args.detail_init_strength,
            args.detail_density_boost,
            args.detail_length_boost,
            args.detail_root_width_boost,
            args.detail_tip_width_boost,
        )
        if surface_roots is not None and surface_normals is not None and surface_uv is not None:
            init_surface_texture_from_images(model, surface_roots, surface_normals, surface_uv, images, masks, pmats, cam_centers)
    initialization_stats: dict[str, float | int | str] = {}
    if model.coarse_texture is not None and not args.no_coarse_init_split:
        initialization_stats.update(split_initialized_texture_into_coarse_base(model))
    elif model.coarse_texture is not None:
        initialization_stats.update({"coarse_init_enabled": 0, "coarse_init_reason": "disabled"})
    if init_checkpoint is not None and not args.no_init_checkpoint_model:
        load_stats = load_compatible_model_state(model, init_checkpoint["model"])
        initialization_stats.update(
            {
                "init_checkpoint_enabled": 1,
                "init_checkpoint_model_loaded": 1,
                "init_checkpoint_path": str(init_checkpoint_path),
                "init_checkpoint_iter": int(init_checkpoint.get("iter", -1)),
                "init_checkpoint_root_count": int(root_sampling_stats.get("init_checkpoint_root_count", -1)),
                **{f"init_checkpoint_{key}": value for key, value in load_stats.items()},
            }
        )
    elif init_checkpoint is not None:
        initialization_stats.update(
            {
                "init_checkpoint_enabled": 1,
                "init_checkpoint_model_loaded": 0,
                "init_checkpoint_path": str(init_checkpoint_path),
                "init_checkpoint_iter": int(init_checkpoint.get("iter", -1)),
                "init_checkpoint_root_count": int(root_sampling_stats.get("init_checkpoint_root_count", -1)),
                "init_checkpoint_model_load": "disabled",
            }
        )
    if initialization_stats:
        (out / "initialization_stats.json").write_text(json.dumps(initialization_stats, indent=2), encoding="utf-8")
        print(json.dumps({"initialization": initialization_stats}), flush=True)
    if args.freeze_pose or args.pose_lr <= 0.0:
        model.log_scale.requires_grad_(False)
        model.trans.requires_grad_(False)

    def build_optimizer() -> torch.optim.Optimizer:
        param_groups = [{"params": [model.texture], "lr": args.lr, "name": "texture"}]
        if model.triplanes is not None:
            param_groups.append(
                {"params": [model.triplanes], "lr": args.lr * float(args.triplane_lr_scale), "name": "triplanes"}
            )
        if model.coarse_texture is not None:
            param_groups.append(
                {
                    "params": [model.coarse_texture],
                    "lr": args.lr * float(args.coarse_lr_scale),
                    "name": "coarse_texture",
                }
            )
        if model.head_texture is not None:
            param_groups.append(
                {"params": [model.head_texture], "lr": args.lr * float(args.head_lr_scale), "name": "head_texture"}
            )
        if model.surface_texture is not None:
            param_groups.append(
                {
                    "params": [model.surface_texture],
                    "lr": args.lr * float(args.surface_lr_scale),
                    "name": "surface_texture",
                }
            )
        if root_bary_logits is not None:
            param_groups.append({"params": [root_bary_logits], "lr": args.root_surface_move_lr, "name": "root_bary_logits"})
        if not (args.freeze_pose or args.pose_lr <= 0.0):
            param_groups.append({"params": [model.log_scale, model.trans], "lr": args.pose_lr, "name": "pose"})
        return torch.optim.Adam(param_groups)

    opt = build_optimizer()
    sample_idx = torch.linspace(0, args.curve_samples - 1, args.render_samples, device=device).long()
    def build_flow_orient_segment_ids(root_count: int, iteration: int = 0) -> torch.Tensor | None:
        total_flow_segments = int(root_count) * max(int(sample_idx.numel()) - 1, 0)
        if (
            args.flow_orient_weight <= 0.0
            or args.flow_orient_max_segments <= 0
            or total_flow_segments <= args.flow_orient_max_segments
        ):
            return None
        rng = np.random.default_rng(args.seed + 1337 + int(iteration))
        ids = np.sort(
            rng.choice(total_flow_segments, size=args.flow_orient_max_segments, replace=False)
        ).astype(np.int64)
        return torch.tensor(ids, device=device)

    def build_mesh_graph_edges_tensor(iteration: int = 0) -> torch.Tensor | None:
        if args.mesh_graph_smooth_weight <= 0.0 or args.mesh_graph_smooth_edges <= 0:
            return None
        if int(iteration) == 0 and initial_mesh_graph_edges_np is not None:
            edges_np = initial_mesh_graph_edges_np
        else:
            edges_np = build_root_graph_edges(
                np.asarray(root_data["face_ids"], dtype=np.int64),
                face_neighbors,
                int(args.mesh_graph_smooth_edges),
                args.seed + 8117 + int(iteration),
            )
        if edges_np.shape[0] == 0:
            return None
        return torch.tensor(edges_np, device=device, dtype=torch.long)

    flow_orient_segment_ids = build_flow_orient_segment_ids(int(root_base_roots.shape[0]))
    mesh_graph_edges = build_mesh_graph_edges_tensor(0)

    def scheduled_flow_orient_weight(iteration: int) -> float:
        if args.flow_orient_weight <= 0.0:
            return 0.0
        if iteration < args.flow_orient_start_iter:
            return 0.0
        warmup = max(0, int(args.flow_orient_warmup_iters))
        if warmup <= 0:
            weight = float(args.flow_orient_weight)
        else:
            progress = (iteration - args.flow_orient_start_iter + 1) / float(warmup)
            progress = min(1.0, max(0.0, progress))
            weight = float(args.flow_orient_weight) * progress
        end_iter = int(args.flow_orient_end_iter)
        if end_iter <= 0:
            return weight
        if iteration > end_iter:
            return 0.0
        decay = max(0, int(args.flow_orient_decay_iters))
        if decay <= 0:
            return weight
        decay_start = max(int(args.flow_orient_start_iter), end_iter - decay + 1)
        if iteration < decay_start:
            return weight
        decay_progress = (end_iter - iteration + 1) / float(max(end_iter - decay_start + 1, 1))
        decay_progress = min(1.0, max(0.0, decay_progress))
        return weight * decay_progress

    def scheduled_flow_hint_prior_weight(iteration: int) -> float:
        if args.flow_hint_prior_weight <= 0.0 or flow_hint is None or flow_hint_confidence is None:
            return 0.0
        if iteration < args.flow_hint_prior_start_iter:
            return 0.0
        warmup = max(0, int(args.flow_hint_prior_warmup_iters))
        if warmup <= 0:
            weight = float(args.flow_hint_prior_weight)
        else:
            progress = (iteration - args.flow_hint_prior_start_iter + 1) / float(warmup)
            progress = min(1.0, max(0.0, progress))
            weight = float(args.flow_hint_prior_weight) * progress
        end_iter = int(args.flow_hint_prior_end_iter)
        if end_iter <= 0:
            return weight
        if iteration > end_iter:
            return 0.0
        decay = max(0, int(args.flow_hint_prior_decay_iters))
        if decay <= 0:
            return weight
        decay_start = max(int(args.flow_hint_prior_start_iter), end_iter - decay + 1)
        if iteration < decay_start:
            return weight
        decay_progress = (end_iter - iteration + 1) / float(max(end_iter - decay_start + 1, 1))
        decay_progress = min(1.0, max(0.0, decay_progress))
        return weight * decay_progress

    def scheduled_flow_coherence_weight(iteration: int) -> float:
        if args.flow_coherence_weight <= 0.0:
            return 0.0
        if iteration < args.flow_coherence_start_iter:
            return 0.0
        warmup = max(0, int(args.flow_coherence_warmup_iters))
        if warmup <= 0:
            return float(args.flow_coherence_weight)
        progress = (iteration - args.flow_coherence_start_iter + 1) / float(warmup)
        progress = min(1.0, max(0.0, progress))
        return float(args.flow_coherence_weight) * progress

    def scheduled_head_weight(base_weight: float, iteration: int) -> float:
        if base_weight <= 0.0 or head_train_weights is None:
            return 0.0
        if iteration < args.head_loss_start_iter:
            return 0.0
        warmup = max(0, int(args.head_loss_warmup_iters))
        if warmup <= 0:
            return float(base_weight)
        progress = (iteration - args.head_loss_start_iter + 1) / float(warmup)
        progress = min(1.0, max(0.0, progress))
        return float(base_weight) * progress

    def scheduled_boundary_weight(iteration: int) -> float:
        if args.boundary_mask_loss_weight <= 0.0 or boundary_weights is None:
            return 0.0
        if iteration < args.boundary_loss_start_iter:
            return 0.0
        warmup = max(0, int(args.boundary_loss_warmup_iters))
        if warmup <= 0:
            return float(args.boundary_mask_loss_weight)
        progress = (iteration - args.boundary_loss_start_iter + 1) / float(warmup)
        progress = min(1.0, max(0.0, progress))
        return float(args.boundary_mask_loss_weight) * progress

    def scheduled_random_mesh_backing_weight(iteration: int) -> float:
        if args.random_mesh_backing_weight <= 0.0:
            return 0.0
        if iteration < args.random_mesh_backing_start_iter:
            return 0.0
        warmup = max(0, int(args.random_mesh_backing_warmup_iters))
        if warmup <= 0:
            return float(args.random_mesh_backing_weight)
        progress = (iteration - args.random_mesh_backing_start_iter + 1) / float(warmup)
        progress = min(1.0, max(0.0, progress))
        return float(args.random_mesh_backing_weight) * progress

    def scheduled_mesh_graph_smooth_weight(iteration: int) -> float:
        if args.mesh_graph_smooth_weight <= 0.0 or mesh_graph_edges is None:
            return 0.0
        if iteration < args.mesh_graph_smooth_start_iter:
            return 0.0
        warmup = max(0, int(args.mesh_graph_smooth_warmup_iters))
        if warmup <= 0:
            return float(args.mesh_graph_smooth_weight)
        progress = (iteration - args.mesh_graph_smooth_start_iter + 1) / float(warmup)
        progress = min(1.0, max(0.0, progress))
        return float(args.mesh_graph_smooth_weight) * progress

    def zero_flow_grads_if_frozen(iteration: int) -> bool:
        freeze_after = int(args.freeze_flow_after_iter)
        if freeze_after <= 0 or iteration < freeze_after:
            return False
        for tex in [model.texture, model.coarse_texture, model.head_texture]:
            if tex is not None and tex.grad is not None and tex.grad.shape[1] > 6:
                tex.grad[:, 5:7].zero_()
        if model.triplanes is not None and model.triplanes.grad is not None and model.triplanes.grad.shape[1] > 6:
            model.triplanes.grad[:, 5:7].zero_()
        return True

    log_path = out / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for it in range(1, args.iters + 1):
            view = int(torch.randint(0, images.shape[0], (1,), device=device).item())
            root_bary, root_move_active = current_root_bary(it)
            roots, uv, coord, body_u = root_state_from_bary(root_bary)
            root_surface_move_reg = roots.new_tensor(0.0)
            if root_bary_logits is not None and root_move_active and args.root_surface_move_reg_weight > 0.0:
                root_delta = (roots - root_base_roots) / root_edge_scale
                root_surface_move_reg = torch.mean(torch.square(root_delta))
            root_surface_move_stats = root_surface_movement_stats(roots, root_move_active)
            params = model.sample(uv, coord, body_u)
            groom = generate_stage_a_curves(roots, normals, tangents, bitangents, params, args.curve_samples)
            scale = torch.exp(model.log_scale).clamp(0.25, 4.0)
            curves = groom.curves * scale + model.trans.view(1, 1, 3)
            pts = curves[:, sample_idx].reshape(-1, 3)
            nrm = normals[:, None, :].expand(-1, args.render_samples, -1).reshape(-1, 3)
            cols = groom.color[:, sample_idx].reshape(-1, 3)
            per_sample_alpha = (groom.alpha.expand(-1, args.render_samples, -1).reshape(-1, 1) * args.alpha_scale)
            tangent_vectors = sample_curve_tangents(curves, sample_idx) if args.splat_mode == "oriented" else None
            point_radius = None
            if args.strand_width_radius:
                point_radius = splat_radius_from_strand_width(
                    groom.width[:, sample_idx],
                    args.splat_radius,
                    args.radius_width_ref,
                    args.radius_width_min_scale,
                    args.radius_width_max_scale,
                )
            adaptive_stats: dict[str, float | int] = {}
            if args.adaptive_render_samples:
                active_mask, adaptive_stats = adaptive_render_sample_mask(
                    curves,
                    groom.length,
                    int(sample_idx.numel()),
                    args.adaptive_min_render_samples,
                )
                flat_active = active_mask.reshape(-1)
                pts = pts[flat_active]
                nrm = nrm[flat_active]
                cols = cols[flat_active]
                per_sample_alpha = per_sample_alpha[flat_active]
                if tangent_vectors is not None:
                    tangent_vectors = tangent_vectors[flat_active]
                if point_radius is not None:
                    point_radius = point_radius.reshape(-1, 1)[flat_active]
            fur_pts = pts
            fur_nrm = nrm
            fur_cols = cols
            fur_alpha = per_sample_alpha
            fur_tangent_vectors = tangent_vectors
            fur_point_radius = point_radius
            if surface_roots is None:
                base_pts = roots * scale + model.trans.view(1, 3)
                base_nrm = normals
                base_cols = groom.color[:, 0]
                base_alpha = groom.alpha[:, 0] * args.surface_alpha_scale
                pts = torch.cat([base_pts, pts], dim=0)
                nrm = torch.cat([base_nrm, nrm], dim=0)
                cols = torch.cat([base_cols, cols], dim=0)
                per_sample_alpha = torch.cat([base_alpha, per_sample_alpha], dim=0)
                if tangent_vectors is not None:
                    tangent_vectors = torch.cat([torch.zeros_like(base_pts), tangent_vectors], dim=0)
                if point_radius is not None:
                    base_radius = splat_radius_from_strand_width(
                        groom.root_width,
                        args.splat_radius,
                        args.radius_width_ref,
                        args.radius_width_min_scale,
                        args.radius_width_max_scale,
                    )
                    point_radius = torch.cat([base_radius, point_radius], dim=0)
            else:
                surface_layer_active = model.surface_texture is not None and args.surface_alpha_scale > 0.0
                pts, nrm, cols, per_sample_alpha = append_surface_layer(
                    model,
                    pts,
                    nrm,
                    cols,
                    per_sample_alpha,
                    surface_roots,
                    surface_normals,
                    surface_uv,
                    scale,
                    args.surface_alpha_scale,
                )
                if surface_layer_active and tangent_vectors is not None:
                    tangent_vectors = torch.cat([torch.zeros_like(surface_roots), tangent_vectors], dim=0)
                if surface_layer_active and point_radius is not None:
                    surface_radius = torch.full(
                        (surface_roots.shape[0], 1),
                        float(args.splat_radius),
                        device=device,
                        dtype=point_radius.dtype,
                    )
                    point_radius = torch.cat([surface_radius, point_radius], dim=0)
            pred_img, pred_mask = splat_render(
                pts,
                nrm,
                cols,
                per_sample_alpha,
                pmats[view],
                cam_centers[view],
                height,
                width,
                args.splat_radius,
                args.alpha_cap,
                args.depth_band,
                args.depth_sharpness,
                tangent_vectors,
                args.tangent_radius_scale,
                args.normal_radius_scale,
                point_radius,
            )
            target_img = images[view]
            target_mask = masks[view]
            random_mesh_backing_loss = pred_img.new_tensor(0.0)
            random_mesh_backing_weight = scheduled_random_mesh_backing_weight(it)
            random_mesh_backed_img = None
            if random_mesh_backing_weight > 0.0:
                if surface_roots is None or surface_normals is None:
                    raise RuntimeError("random mesh backing requires sampled surface roots")
                backing_pts, backing_nrm, backing_cols, backing_alpha = append_random_mesh_backing(
                    model,
                    fur_pts,
                    fur_nrm,
                    fur_cols,
                    fur_alpha,
                    surface_roots,
                    surface_normals,
                    scale,
                    args.random_mesh_backing_alpha_scale,
                    args.random_mesh_backing_strength,
                    args.random_mesh_backing_normal_offset,
                )
                backing_tangent_vectors = fur_tangent_vectors
                if backing_tangent_vectors is not None:
                    backing_tangent_vectors = torch.cat(
                        [torch.zeros_like(surface_roots), backing_tangent_vectors],
                        dim=0,
                    )
                backing_point_radius = fur_point_radius
                if backing_point_radius is not None:
                    backing_surface_radius = torch.full(
                        (surface_roots.shape[0], 1),
                        float(args.splat_radius),
                        device=device,
                        dtype=backing_point_radius.dtype,
                    )
                    backing_point_radius = torch.cat([backing_surface_radius, backing_point_radius], dim=0)
                backed_img, _ = splat_render(
                    backing_pts,
                    backing_nrm,
                    backing_cols,
                    backing_alpha,
                    pmats[view],
                    cam_centers[view],
                    height,
                    width,
                    args.splat_radius,
                    args.alpha_cap,
                    args.depth_band,
                    args.depth_sharpness,
                    backing_tangent_vectors,
                    args.tangent_radius_scale,
                    args.normal_radius_scale,
                    backing_point_radius,
                )
                random_mesh_backed_img = backed_img
                backing_loss_mask = (detail_weights[view].detach() * target_mask.detach()).clamp(min=0.0)
                random_mesh_backing_loss = (
                    torch.abs(backed_img - target_img) * backing_loss_mask
                ).sum() / torch.clamp(backing_loss_mask.sum() * 3.0, min=1.0)
            flow_orient = pred_img.new_tensor(0.0)
            flow_orient_stats: dict[str, float | int] = {}
            flow_orient_weight = scheduled_flow_orient_weight(it)
            if (
                flow_orient_weight > 0.0
                and target_orientation is not None
                and target_orientation_confidence is not None
            ):
                flow_curves = curves[:, sample_idx]
                flow_orient, flow_orient_stats = flow_orientation_loss_for_view(
                    flow_curves,
                    pmats[view],
                    target_orientation[view],
                    target_orientation_confidence[view],
                    width,
                    height,
                    flow_orient_segment_ids,
                    args.flow_orient_min_confidence,
                )
            mask_loss = F.mse_loss(pred_mask, target_mask)
            fg = detail_weights[view]
            color_loss = (torch.abs(pred_img - target_img) * fg).sum() / torch.clamp(fg.sum() * 3.0, min=1.0)
            head_color_loss = pred_img.new_tensor(0.0)
            head_mask_loss = pred_img.new_tensor(0.0)
            boundary_mask_loss = pred_img.new_tensor(0.0)
            head_color_weight = scheduled_head_weight(args.head_color_loss_weight, it)
            head_mask_weight = scheduled_head_weight(args.head_mask_loss_weight, it)
            if (head_color_weight > 0.0 or head_mask_weight > 0.0) and head_train_weights is not None:
                head_weight = head_train_weights[view]
                head_denom = head_weight.sum().clamp_min(1.0)
                if head_color_weight > 0.0:
                    head_color_loss = (
                        torch.abs(pred_img - target_img) * head_weight
                    ).sum() / torch.clamp(head_denom * 3.0, min=1.0)
                if head_mask_weight > 0.0:
                    head_mask_loss = (torch.abs(pred_mask - target_mask) * head_weight).sum() / head_denom
            boundary_weight = scheduled_boundary_weight(it)
            if boundary_weight > 0.0 and boundary_weights is not None:
                bw = boundary_weights[view]
                boundary_mask_loss = (torch.abs(pred_mask - target_mask) * bw).sum() / bw.sum().clamp_min(1.0)
            grad_loss = sobel_gradient_loss(pred_img, target_img, target_mask)
            tv = model.tv_loss(args.detail_tv_relax, args.detail_tv_min_weight)
            asset_param_smooth = model.asset_parameter_smooth_loss(
                args.detail_tv_relax,
                args.detail_tv_min_weight,
            )
            mesh_graph_smooth = pred_img.new_tensor(0.0)
            mesh_graph_smooth_stats: dict[str, float | int] = {}
            mesh_graph_smooth_weight = scheduled_mesh_graph_smooth_weight(it)
            if mesh_graph_smooth_weight > 0.0:
                mesh_graph_smooth, mesh_graph_smooth_stats = mesh_graph_groom_smooth_loss(
                    params,
                    mesh_graph_edges,
                )
            density_mean = torch.sigmoid(model.texture[:, 1:2]).mean()
            flow_hint_prior = pred_img.new_tensor(0.0)
            flow_hint_prior_stats: dict[str, float | int] = {}
            flow_hint_prior_weight = scheduled_flow_hint_prior_weight(it)
            if flow_hint_prior_weight > 0.0:
                flow_hint_prior, flow_hint_prior_stats = flow_hint_prior_loss(
                    params,
                    flow_hint,
                    flow_hint_confidence,
                    args.flow_hint_prior_min_confidence,
                )
            flow_coherence = pred_img.new_tensor(0.0)
            flow_coherence_stats: dict[str, float] = {}
            flow_coherence_weight = scheduled_flow_coherence_weight(it)
            if flow_coherence_weight > 0.0:
                flow_coherence, flow_coherence_stats = model.flow_coherence_loss(
                    args.flow_coherence_detail_relax,
                    args.flow_coherence_min_weight,
                )
            groom_geometry_loss = pred_img.new_tensor(0.0)
            groom_geometry_stats: dict[str, float] = {}
            if args.groom_geometry_weight > 0.0:
                groom_geometry_loss, groom_geometry_stats = groom_geometry_prior_loss(
                    groom,
                    params["lift"],
                    params["detail_evidence"],
                    root_boundary_evidence,
                    body_u,
                    args.head_start,
                    args.root_head_detail_sharpness,
                    args.groom_length_floor,
                    args.groom_detail_length_boost,
                    args.groom_head_length_boost,
                    args.groom_boundary_length_boost,
                    args.groom_boundary_density_floor,
                    args.groom_boundary_lift_floor,
                    args.groom_root_width_target,
                    args.groom_tip_width_target,
                    args.groom_max_tip_root_ratio,
                )
            loss = (
                1.8 * mask_loss
                + color_loss
                + head_color_weight * head_color_loss
                + head_mask_weight * head_mask_loss
                + boundary_weight * boundary_mask_loss
                + args.grad_loss_weight * grad_loss
                + flow_orient_weight * flow_orient
                + flow_hint_prior_weight * flow_hint_prior
                + flow_coherence_weight * flow_coherence
                + args.groom_geometry_weight * groom_geometry_loss
                + random_mesh_backing_weight * random_mesh_backing_loss
                + args.root_surface_move_reg_weight * root_surface_move_reg
                + args.texture_tv_weight * tv
                + args.asset_param_smooth_weight * asset_param_smooth
                + mesh_graph_smooth_weight * mesh_graph_smooth
                + 0.005 * density_mean
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            flow_grads_frozen = zero_flow_grads_if_frozen(it)
            grad_params = list(model.parameters())
            if root_bary_logits is not None:
                grad_params.append(root_bary_logits)
            torch.nn.utils.clip_grad_norm_(grad_params, 1.0)
            opt.step()
            with torch.no_grad():
                model.log_scale.clamp_(math.log(0.85), math.log(1.65))
                model.trans.clamp_(-0.45, 0.75)
                if root_bary_logits is not None and args.root_surface_move_logit_limit > 0.0:
                    limit = float(args.root_surface_move_logit_limit)
                    root_bary_logits.clamp_(-limit, limit)

            if it == 1 or it % 20 == 0 or it % args.save_every == 0:
                rec = {
                    "iter": it,
                    "view": view,
                    "loss": float(loss.detach().cpu()),
                    "mask_loss": float(mask_loss.detach().cpu()),
                    "color_loss": float(color_loss.detach().cpu()),
                    "head_color_loss": float(head_color_loss.detach().cpu()),
                    "head_mask_loss": float(head_mask_loss.detach().cpu()),
                    "head_color_weight": float(head_color_weight),
                    "head_mask_weight": float(head_mask_weight),
                    "boundary_mask_loss": float(boundary_mask_loss.detach().cpu()),
                    "boundary_mask_weight": float(boundary_weight),
                    "grad_loss": float(grad_loss.detach().cpu()),
                    "flow_orient": float(flow_orient.detach().cpu()),
                    "flow_orient_weight": float(flow_orient_weight),
                    "flow_hint_prior": float(flow_hint_prior.detach().cpu()),
                    "flow_hint_prior_weight": float(flow_hint_prior_weight),
                    "flow_grads_frozen": bool(flow_grads_frozen),
                    "flow_coherence": float(flow_coherence.detach().cpu()),
                    "flow_coherence_weight": float(flow_coherence_weight),
                    "groom_geometry_loss": float(groom_geometry_loss.detach().cpu()),
                    "groom_geometry_weight": float(args.groom_geometry_weight),
                    "random_mesh_backing_loss": float(random_mesh_backing_loss.detach().cpu()),
                    "random_mesh_backing_weight": float(random_mesh_backing_weight),
                    "root_surface_move_reg": float(root_surface_move_reg.detach().cpu()),
                    "root_surface_move_reg_weight": float(args.root_surface_move_reg_weight),
                    "tv": float(tv.detach().cpu()),
                    "texture_tv_weight": float(args.texture_tv_weight),
                    "asset_param_smooth": float(asset_param_smooth.detach().cpu()),
                    "asset_param_smooth_weight": float(args.asset_param_smooth_weight),
                    "mesh_graph_smooth": float(mesh_graph_smooth.detach().cpu()),
                    "mesh_graph_smooth_weight": float(mesh_graph_smooth_weight),
                    "density_mean": float(density_mean.detach().cpu()),
                    "scale": float(torch.exp(model.log_scale).detach().cpu()),
                    "trans": [float(x) for x in model.trans.detach().cpu()],
                }
                rec.update(root_surface_move_stats)
                rec.update(flow_orient_stats)
                rec.update(flow_hint_prior_stats)
                rec.update(flow_coherence_stats)
                rec.update(mesh_graph_smooth_stats)
                rec.update(adaptive_stats)
                rec.update(groom_geometry_stats)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                if it % args.save_every == 0 and random_mesh_backed_img is not None:
                    save_tensor_image(
                        out / f"iter_{it:06d}" / f"view_{view:02d}_random_mesh_backed_train.png",
                        random_mesh_backed_img,
                    )

            if it == 1 or it % args.save_every == 0:
                with torch.no_grad():
                    eval_bary, _ = current_root_bary(it)
                    roots, uv, coord, body_u = root_state_from_bary(eval_bary)
                    eval_views = [0, 9, 18, 27]
                    eval_records = []
                    residual_acc = torch.zeros(roots.shape[0], 1, device=device)
                    residual_weight = torch.zeros_like(residual_acc)
                    residual_scale = torch.exp(model.log_scale).clamp(0.25, 4.0)
                    residual_root_points = roots * residual_scale + model.trans.view(1, 3)
                    for ev in eval_views:
                        params = model.sample(uv, coord, body_u)
                        groom = generate_stage_a_curves(roots, normals, tangents, bitangents, params, args.curve_samples)
                        curves = groom.curves * torch.exp(model.log_scale).clamp(0.25, 4.0) + model.trans.view(1, 1, 3)
                        pts = curves[:, sample_idx].reshape(-1, 3)
                        nrm = normals[:, None, :].expand(-1, args.render_samples, -1).reshape(-1, 3)
                        cols = groom.color[:, sample_idx].reshape(-1, 3)
                        alpha = groom.alpha.expand(-1, args.render_samples, -1).reshape(-1, 1) * args.alpha_scale
                        tangent_vectors = sample_curve_tangents(curves, sample_idx) if args.splat_mode == "oriented" else None
                        point_radius = None
                        if args.strand_width_radius:
                            point_radius = splat_radius_from_strand_width(
                                groom.width[:, sample_idx],
                                args.splat_radius,
                                args.radius_width_ref,
                                args.radius_width_min_scale,
                                args.radius_width_max_scale,
                            )
                        if args.adaptive_render_samples:
                            active_mask, _ = adaptive_render_sample_mask(
                                curves,
                                groom.length,
                                int(sample_idx.numel()),
                                args.adaptive_min_render_samples,
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
                        eval_scale = torch.exp(model.log_scale).clamp(0.25, 4.0)
                        if surface_roots is None:
                            base_pts = roots * eval_scale + model.trans.view(1, 3)
                            base_nrm = normals
                            base_cols = groom.color[:, 0]
                            base_alpha = groom.alpha[:, 0] * args.surface_alpha_scale
                            pts = torch.cat([base_pts, pts], dim=0)
                            nrm = torch.cat([base_nrm, nrm], dim=0)
                            cols = torch.cat([base_cols, cols], dim=0)
                            alpha = torch.cat([base_alpha, alpha], dim=0)
                            if tangent_vectors is not None:
                                tangent_vectors = torch.cat([torch.zeros_like(base_pts), tangent_vectors], dim=0)
                            if point_radius is not None:
                                base_radius = splat_radius_from_strand_width(
                                    groom.root_width,
                                    args.splat_radius,
                                    args.radius_width_ref,
                                    args.radius_width_min_scale,
                                    args.radius_width_max_scale,
                                )
                                point_radius = torch.cat([base_radius, point_radius], dim=0)
                        else:
                            surface_layer_active = model.surface_texture is not None and args.surface_alpha_scale > 0.0
                            pts, nrm, cols, alpha = append_surface_layer(
                                model,
                                pts,
                                nrm,
                                cols,
                                alpha,
                                surface_roots,
                                surface_normals,
                                surface_uv,
                                eval_scale,
                                args.surface_alpha_scale,
                            )
                            if surface_layer_active and tangent_vectors is not None:
                                tangent_vectors = torch.cat([torch.zeros_like(surface_roots), tangent_vectors], dim=0)
                            if surface_layer_active and point_radius is not None:
                                surface_radius = torch.full(
                                    (surface_roots.shape[0], 1),
                                    float(args.splat_radius),
                                    device=device,
                                    dtype=point_radius.dtype,
                                )
                                point_radius = torch.cat([surface_radius, point_radius], dim=0)
                        pred_img, pred_mask = splat_render(
                            pts,
                            nrm,
                            cols,
                            alpha,
                            pmats[ev],
                            cam_centers[ev],
                            height,
                            width,
                            args.splat_radius,
                            args.alpha_cap,
                            args.depth_band,
                            args.depth_sharpness,
                            tangent_vectors,
                            args.tangent_radius_scale,
                            args.normal_radius_scale,
                            point_radius,
                        )
                        save_tensor_image(out / f"iter_{it:06d}" / f"view_{ev:02d}_render.png", pred_img)
                        save_tensor_image(out / f"iter_{it:06d}" / f"view_{ev:02d}_mask.png", pred_mask)
                        save_tensor_image(out / f"iter_{it:06d}" / f"view_{ev:02d}_target.png", images[ev])
                        save_tensor_image(out / f"iter_{it:06d}" / f"view_{ev:02d}_target_mask.png", masks[ev])
                        rec_metrics = image_metrics(pred_img, images[ev], pred_mask, masks[ev])
                        rec_metrics.update({"iter": int(it), "view": int(ev)})
                        eval_detail_weight = detail_weights[ev].clamp(min=0.0) * masks[ev]
                        nonzero_detail = eval_detail_weight[eval_detail_weight > 1e-5]
                        if nonzero_detail.numel() > 16:
                            detail_norm = (
                                eval_detail_weight
                                / torch.quantile(nonzero_detail, 0.95).clamp_min(1e-4)
                            ).clamp(0.0, 1.0)
                        else:
                            detail_norm = eval_detail_weight.clamp(0.0, 1.0)
                        head_weight = projected_head_weight_mask(
                            residual_root_points,
                            normals,
                            body_u,
                            pmats[ev],
                            cam_centers[ev],
                            masks[ev],
                            args.root_head_detail_start,
                        )
                        rec_metrics.update(
                            weighted_image_metrics(
                                pred_img,
                                images[ev],
                                pred_mask,
                                masks[ev],
                                detail_norm,
                                "detail",
                            )
                        )
                        rec_metrics.update(
                            weighted_image_metrics(
                                pred_img,
                                images[ev],
                                pred_mask,
                                masks[ev],
                                head_weight,
                                "head",
                            )
                        )
                        save_tensor_image(out / f"iter_{it:06d}" / f"view_{ev:02d}_detail_weight.png", detail_norm)
                        save_tensor_image(out / f"iter_{it:06d}" / f"view_{ev:02d}_head_weight.png", head_weight)
                        if (
                            args.flow_orient_weight > 0.0
                            and target_orientation is not None
                            and target_orientation_confidence is not None
                        ):
                            flow_curves = curves[:, sample_idx]
                            eval_flow_orient, eval_flow_stats = flow_orientation_loss_for_view(
                                flow_curves,
                                pmats[ev],
                                target_orientation[ev],
                                target_orientation_confidence[ev],
                                width,
                                height,
                                flow_orient_segment_ids,
                                args.flow_orient_min_confidence,
                            )
                            rec_metrics.update(
                                {
                                    "flow_orient_loss": float(eval_flow_orient.detach().cpu()),
                                    **eval_flow_stats,
                                }
                            )
                            orient_dir = out / f"iter_{it:06d}" / f"view_{ev:02d}_orientation"
                            save_orientation_debug(
                                orient_dir,
                                target_orientation[ev],
                                target_orientation_confidence[ev],
                            )
                            save_flow_orientation_overlay(
                                orient_dir,
                                images[ev],
                                flow_curves,
                                pmats[ev],
                                target_orientation[ev],
                                target_orientation_confidence[ev],
                                width,
                                height,
                                args.flow_orient_min_confidence,
                            )
                        eval_records.append(rec_metrics)
                        residual_img = torch.mean(torch.abs(pred_img - images[ev]), dim=-1, keepdim=True)
                        root_xy, root_z = project_points(residual_root_points, pmats[ev])
                        root_inb = (
                            (root_z > 1e-5)
                            & (root_xy[:, 0] >= 0)
                            & (root_xy[:, 0] <= width - 2)
                            & (root_xy[:, 1] >= 0)
                            & (root_xy[:, 1] <= height - 2)
                        )
                        root_view_dir = torch_normalize(cam_centers[ev].view(1, 3) - residual_root_points)
                        root_front = torch.sigmoid(24.0 * ((normals * root_view_dir).sum(dim=-1, keepdim=True) - 0.02))
                        root_mask = bilinear_sample_image(masks[ev], root_xy)[:, :1]
                        root_conf = root_mask * root_front * root_inb[:, None].float()
                        root_residual = bilinear_sample_image(residual_img, root_xy)[:, :1]
                        residual_acc += root_residual.clamp(0.0, 1.0) * root_conf
                        residual_weight += root_conf
                    if eval_records:
                        mean_rec = {
                            "iter": int(it),
                            "view": "mean",
                            "psnr": float(np.mean([r["psnr"] for r in eval_records])),
                            "l1": float(np.mean([r["l1"] for r in eval_records])),
                            "mask_l1": float(np.mean([r["mask_l1"] for r in eval_records])),
                            "detail_psnr": float(np.mean([r["detail_psnr"] for r in eval_records])),
                            "detail_l1": float(np.mean([r["detail_l1"] for r in eval_records])),
                            "detail_mask_l1": float(np.mean([r["detail_mask_l1"] for r in eval_records])),
                            "detail_coverage": float(np.mean([r["detail_coverage"] for r in eval_records])),
                            "head_psnr": float(np.mean([r["head_psnr"] for r in eval_records])),
                            "head_l1": float(np.mean([r["head_l1"] for r in eval_records])),
                            "head_mask_l1": float(np.mean([r["head_mask_l1"] for r in eval_records])),
                            "head_coverage": float(np.mean([r["head_coverage"] for r in eval_records])),
                        }
                        orient_records = [r for r in eval_records if "flow_orient_loss" in r]
                        if orient_records:
                            mean_rec.update(
                                {
                                    "flow_orient_loss": float(np.mean([r["flow_orient_loss"] for r in orient_records])),
                                    "flow_orient_weight_sum": float(
                                        np.sum([r.get("flow_orient_weight_sum", 0.0) for r in orient_records])
                                    ),
                                    "flow_orient_valid_samples": int(
                                        np.sum([r.get("flow_orient_valid_samples", 0) for r in orient_records])
                                    ),
                                    "flow_orient_mean_confidence": float(
                                        np.mean([r.get("flow_orient_mean_confidence", 0.0) for r in orient_records])
                                    ),
                                }
                            )
                        with (out / "eval_metrics.jsonl").open("a", encoding="utf-8") as mf:
                            for rec_metrics in eval_records:
                                mf.write(json.dumps(rec_metrics) + "\n")
                            mf.write(json.dumps(mean_rec) + "\n")
                        eval_log_rec = {"eval": mean_rec}
                        log.write(json.dumps(eval_log_rec) + "\n")
                        log.flush()
                        print(json.dumps(eval_log_rec), flush=True)
                    residual_values = residual_acc / torch.clamp(residual_weight, min=1e-6)
                    known_residual = residual_weight[:, 0] > 1e-5
                    residual_norm = torch.zeros_like(residual_values)
                    if int(known_residual.sum().item()) > 16:
                        known_values = residual_values[known_residual, 0]
                        lo = torch.quantile(known_values, 0.10)
                        hi = torch.quantile(known_values, 0.98)
                        residual_norm = ((residual_values - lo) / torch.clamp(hi - lo, min=1e-6)).clamp(0.0, 1.0)
                    save_root_scalar_uv_debug(
                        out / f"iter_{it:06d}" / "textures" / "residual_evidence.png",
                        uv,
                        residual_norm,
                        residual_weight.clamp(0.0, 1.0),
                        model.tex_h,
                        model.tex_w,
                        args.texture_debug_max,
                    )
                    save_texture_debug(
                        out / f"iter_{it:06d}" / "textures",
                        model.texture,
                        args.texture_debug_max,
                        out / f"iter_{it:06d}" / "groom_maps_contact_sheet.png",
                    )
                    if model.coarse_texture is not None:
                        save_texture_debug(
                            out / f"iter_{it:06d}" / "coarse_textures",
                            model.coarse_texture,
                            args.texture_debug_max,
                            out / f"iter_{it:06d}" / "coarse_groom_maps_contact_sheet.png",
                        )
                    save_groom_stats(out / f"iter_{it:06d}" / "groom_stats.json", groom)
                    adaptive_budget_rel = None
                    if args.adaptive_render_samples:
                        debug_curves = groom.curves * torch.exp(model.log_scale).clamp(0.25, 4.0) + model.trans.view(1, 1, 3)
                        debug_mask, _ = adaptive_render_sample_mask(
                            debug_curves,
                            groom.length,
                            int(sample_idx.numel()),
                            args.adaptive_min_render_samples,
                        )
                        debug_budget = debug_mask.sum(dim=1, keepdim=True).float()
                        denom = max(int(sample_idx.numel()) - int(args.adaptive_min_render_samples), 1)
                        adaptive_budget_rel = (
                            (debug_budget - float(args.adaptive_min_render_samples)) / float(denom)
                        ).clamp(0.0, 1.0)
                        save_root_scalar_uv_debug(
                            out / f"iter_{it:06d}" / "textures" / "adaptive_sample_budget.png",
                            uv,
                            adaptive_budget_rel,
                            torch.ones_like(adaptive_budget_rel),
                            model.tex_h,
                            model.tex_w,
                            args.texture_debug_max,
                        )
                    root_diag = sample_texture_at_roots(model.texture, uv)
                    root_width_rel = _normalize_debug_map(F.softplus(root_diag[:, 3:4]).T[:, :, None]).reshape(-1, 1)
                    tip_width_rel = _normalize_debug_map(F.softplus(root_diag[:, 4:5]).T[:, :, None]).reshape(-1, 1)
                    root_debug_values = {
                        "coverage": groom.coverage,
                        "density": groom.density,
                        "length": ((groom.length - 0.012) / (0.105 - 0.012)).clamp(0.0, 1.0),
                        "root_width": root_width_rel,
                        "flow_conf": root_diag[:, 18:19].clamp(0.0, 1.0),
                        "detail": root_diag[:, 19:20].clamp(0.0, 1.0),
                    }
                    if root_bary_logits is not None:
                        root_move_rel = (
                            torch.linalg.norm(roots - root_base_roots, dim=-1, keepdim=True) / root_edge_scale
                        ).clamp(0.0, 1.0)
                        root_debug_values["root_move"] = root_move_rel
                    if adaptive_budget_rel is not None:
                        root_debug_values["adaptive_budget"] = adaptive_budget_rel
                    save_root_projection_debug(
                        out / f"iter_{it:06d}" / "root_projection_contact_sheet.png",
                        images,
                        residual_root_points,
                        pmats,
                        root_debug_values,
                        eval_views,
                    )
                    save_surface_texture_debug(
                        out / f"iter_{it:06d}" / "textures",
                        model.surface_texture,
                        args.texture_debug_max,
                    )
                    save_uv_coverage_debug(
                        out / f"iter_{it:06d}" / "uv_root_coverage.png",
                        uv.detach().cpu().numpy(),
                        model.tex_h,
                        model.tex_w,
                        args.texture_debug_max,
                    )
                    save_uv_root_overlay_debug(
                        out / f"iter_{it:06d}" / "uv_root_overlay_contact_sheet.png",
                        uv,
                        body_u,
                        root_debug_values,
                        model.tex_h,
                        model.tex_w,
                        args.texture_debug_max,
                        args.root_head_detail_start,
                    )
                    if surface_data is not None:
                        save_uv_coverage_debug(
                            out / f"iter_{it:06d}" / "uv_surface_coverage.png",
                            surface_data["uv"],
                            model.tex_h,
                            model.tex_w,
                            args.texture_debug_max,
                        )
                    draw_curve_projection(
                        groom.curves.detach().cpu().numpy(),
                        groom.width.detach().cpu().numpy(),
                        groom.color.detach().cpu().numpy(),
                        groom.alpha.detach().cpu().numpy(),
                        f"white tiger uv groom iter {it}",
                        out / f"iter_{it:06d}" / "curve_preview.png",
                    )
                    save_bary, _ = current_root_bary(it)
                    checkpoint_root_data = {
                        "axis": root_data["axis"],
                        "center": root_data["center"],
                        "face_ids": root_data["face_ids"],
                        "bary": save_bary.detach().cpu().numpy().astype(np.float32),
                    }
                    torch.save(
                        {
                            "iter": it,
                            "model": model.state_dict(),
                            "root_data": checkpoint_root_data,
                            "args": vars(args),
                        },
                        out / "latest.pt",
                    )
                    densify_enabled = args.root_densify_interval > 0 and args.root_densify_count > 0
                    densify_stop_ok = args.root_densify_stop_iter <= 0 or it <= args.root_densify_stop_iter
                    densify_due = (
                        densify_enabled
                        and it >= args.root_densify_start_iter
                        and densify_stop_ok
                        and it % args.root_densify_interval == 0
                    )
                    if densify_due:
                        current_root_count = int(root_data["roots"].shape[0])
                        max_roots = int(args.root_densify_max_roots)
                        if max_roots <= 0:
                            max_roots = current_root_count + int(args.root_densify_count)
                        add_count = min(int(args.root_densify_count), max(0, max_roots - current_root_count))
                        if add_count > 0:
                            residual_face_weights, densify_stats = compute_dynamic_residual_face_weights(
                                residual_values,
                                residual_weight,
                                root_face_ids,
                                int(faces.shape[0]),
                                args.root_densify_boost,
                                args.root_densify_gamma,
                                args.root_densify_min_weight,
                            )
                            static_mix = max(0.0, min(float(args.root_densify_static_mix), 1.0))
                            if face_sampling_weights is not None and static_mix > 0.0:
                                static_weights = np.asarray(face_sampling_weights, dtype=np.float32)
                                static_weights = static_weights / max(float(static_weights.mean()), EPS)
                                static_weights = np.clip(static_weights, 0.25, 8.0)
                                residual_face_weights = residual_face_weights * (
                                    (1.0 - static_mix) + static_mix * static_weights
                                )
                            if root_bary_logits is not None:
                                root_data["bary"] = torch.softmax(root_bary_logits, dim=-1).detach().cpu().numpy().astype(np.float32)
                            else:
                                root_data["bary"] = root_base_bary.detach().cpu().numpy().astype(np.float32)
                            new_root_data = sample_full_body_roots(
                                vertices,
                                faces,
                                add_count,
                                args.seed + 50021 + int(it),
                                args.uv_mode,
                                uv_vertices,
                                face_uvs,
                                residual_face_weights,
                                args.root_distribution,
                            )
                            root_data = concat_root_data(root_data, new_root_data)
                            roots = torch.tensor(root_data["roots"], device=device)
                            normals = torch.tensor(root_data["normals"], device=device)
                            tangents = torch.tensor(root_data["tangents"], device=device)
                            bitangents = torch.tensor(root_data["bitangents"], device=device)
                            uv = torch.tensor(root_data["uv"], device=device)
                            coord = torch.tensor(root_data["coord"], device=device)
                            body_u = torch.tensor(root_data["body_u"], device=device)
                            root_base_roots = roots
                            root_base_uv = uv
                            root_base_coord = coord
                            root_base_body_u = body_u
                            root_base_bary = torch.tensor(root_data["bary"], device=device)
                            root_face_ids = torch.tensor(root_data["face_ids"], device=device, dtype=torch.long)
                            root_tri = mesh_vertices_t[mesh_faces_t[root_face_ids]]
                            if args.uv_mode in {"xatlas", "obj"}:
                                if uv_vertices is None or face_uvs is None:
                                    raise RuntimeError(f"uv_mode={args.uv_mode} requires UV triangles for root densification")
                                root_uv_tri = uv_vertices_t[face_uvs_t[root_face_ids]]
                            tri_np = vertices[faces[np.asarray(root_data["face_ids"], dtype=np.int64)]]
                            edge_scale_np = (
                                np.linalg.norm(tri_np[:, 1] - tri_np[:, 0], axis=1)
                                + np.linalg.norm(tri_np[:, 2] - tri_np[:, 1], axis=1)
                                + np.linalg.norm(tri_np[:, 0] - tri_np[:, 2], axis=1)
                            ) / 3.0
                            root_edge_scale = torch.tensor(np.maximum(edge_scale_np, EPS), device=device, dtype=torch.float32).view(-1, 1)
                            optimizer_rebuilt = False
                            if args.root_surface_move_lr > 0.0:
                                root_bary_logits = torch.nn.Parameter(torch.log(root_base_bary.clamp_min(1e-5)))
                                opt = build_optimizer()
                                optimizer_rebuilt = True
                            if (
                                args.flow_init_source != "none"
                                and flow_init_orientation is not None
                                and flow_init_orientation_confidence is not None
                            ):
                                flow_hint, flow_hint_confidence, flow_init_stats = compute_root_flow_hints_from_orientation(
                                    roots,
                                    normals,
                                    tangents,
                                    bitangents,
                                    flow_init_orientation,
                                    flow_init_orientation_confidence,
                                    masks,
                                    pmats,
                                    cam_centers,
                                    torch.exp(model.log_scale).clamp(0.25, 4.0),
                                    model.trans,
                                    args.flow_init_min_confidence,
                                    args.flow_init_scale,
                                    args.flow_init_probe_length,
                                )
                            if (
                                boundary_weights is not None
                                and (
                                    args.groom_boundary_length_boost > 0.0
                                    or args.groom_boundary_density_floor > 0.0
                                    or args.groom_boundary_lift_floor > 0.0
                                )
                            ):
                                root_boundary_evidence, root_boundary_stats = compute_root_boundary_evidence(
                                    roots,
                                    normals,
                                    boundary_weights,
                                    masks,
                                    pmats,
                                    cam_centers,
                                    torch.exp(model.log_scale).clamp(0.25, 4.0),
                                    model.trans,
                                )
                            flow_orient_segment_ids = build_flow_orient_segment_ids(int(root_base_roots.shape[0]), it)
                            mesh_graph_edges = build_mesh_graph_edges_tensor(it)
                            densify_event = {
                                "iter": int(it),
                                "added_roots": int(add_count),
                                "root_count_before": int(current_root_count),
                                "root_count_after": int(root_data["roots"].shape[0]),
                                "optimizer_rebuilt": bool(optimizer_rebuilt),
                                "mesh_graph_edges": int(mesh_graph_edges.shape[0]) if mesh_graph_edges is not None else 0,
                                "flow_orient_segment_ids": int(flow_orient_segment_ids.shape[0]) if flow_orient_segment_ids is not None else 0,
                                **densify_stats,
                            }
                            with (out / "root_densify_log.jsonl").open("a", encoding="utf-8") as df:
                                df.write(json.dumps(densify_event) + "\n")
                            log.write(json.dumps({"root_densify": densify_event}) + "\n")
                            log.flush()
                            print(json.dumps({"root_densify": densify_event}), flush=True)

    print(out.resolve())


if __name__ == "__main__":
    main()

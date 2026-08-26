from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.mesh_roots import (
    SurfaceCandidates,
    TriangleMesh,
    farthest_point_sample,
    read_obj_mesh,
    sample_surface_candidates,
)
from anigroom.projection.mesh_visibility import render_mesh_depth, sample_mesh_visible_points


DEFAULT_HEAD_LABELS = (
    "face",
    "ears",
    "inner_earcanal",
    "inner_ear_canal",
    "eyes",
    "nosetip",
    "nose_tip",
    "mane",
    "neck",
)


@dataclass(frozen=True)
class TransferReport:
    mode: str
    source_mesh: str
    target_mesh: str
    source_annotation: str | None
    output_dir: str
    transferred_labels: dict[str, int]
    body_vertex_count: int
    head_labels: list[str]
    head_vertex_count: int
    head_face_count: int
    body_candidate_count: int
    head_candidate_count: int
    selected_body_roots: int
    selected_head_roots: int
    validation_views: list[int]


SMAL_HEAD_LANDMARKS = {
    "nose": 1863,
    "chin": 26,
    "right_ear_tip": 2124,
    "left_ear_tip": 150,
    "left_eye": 3055,
    "right_eye": 1097,
}
SMAL_HEAD_BOUNDARY_LANDMARKS = {
    "throat": 6,
    "withers": 20,
}


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_annotation(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"annotation JSON must be a dict[label -> indices]: {path}")
    out: dict[str, np.ndarray] = {}
    for label, values in raw.items():
        arr = np.asarray(values, dtype=np.int64).reshape(-1)
        if arr.size:
            out[str(label)] = arr
    if not out:
        raise ValueError(f"annotation JSON has no non-empty labels: {path}")
    return out


def apply_neuralfur_white_tiger_smal_transform(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Match NeuralFur transfer_smal_to_neus.py for whiteTiger/default animals."""

    out = mesh.copy()
    rot_x = trimesh.transformations.rotation_matrix(np.radians(-90.0), [1.0, 0.0, 0.0])
    rot_y = trimesh.transformations.rotation_matrix(np.radians(-90.0), [0.0, 1.0, 0.0])
    out.apply_transform(rot_x)
    out.apply_transform(rot_y)
    return out


def load_smal_template_mesh(path: Path) -> trimesh.Trimesh:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    vertices = np.asarray(data["v_template"], dtype=np.float32)
    faces = np.asarray(data["f"], dtype=np.int64)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def align_source_bbox_to_target(source_vertices: np.ndarray, target_vertices: np.ndarray) -> np.ndarray:
    """Affine bbox normalization for the SMAL-template anatomy mode.

    The official annotation-transfer mode should use a fitted SMAL mesh.  When
    only the generic SMAL template is available, this deterministic
    normalization puts the template into the target mesh coordinate box before
    nearest-neighbor label transfer.  It is still SMAL-derived anatomy, not an
    image-space or PCA head detector.
    """

    src_min = source_vertices.min(axis=0)
    src_max = source_vertices.max(axis=0)
    tgt_min = target_vertices.min(axis=0)
    tgt_max = target_vertices.max(axis=0)
    src_extent = np.maximum(src_max - src_min, 1.0e-8)
    tgt_extent = np.maximum(tgt_max - tgt_min, 1.0e-8)
    scale = tgt_extent / src_extent
    return (source_vertices - src_min[None, :]) * scale[None, :] + tgt_min[None, :]


def mesh_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges[:, 0], edges[:, 1]


def smal_template_anatomy_labels(mesh: trimesh.Trimesh, *, neck_margin: float) -> dict[str, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_count = int(vertices.shape[0])
    for name, index in {**SMAL_HEAD_LANDMARKS, **SMAL_HEAD_BOUNDARY_LANDMARKS}.items():
        if index < 0 or index >= vertex_count:
            raise ValueError(f"SMAL landmark {name} index {index} is outside template vertex count {vertex_count}")

    src, dst = mesh_edges(faces)
    edge_len = np.linalg.norm(vertices[src] - vertices[dst], axis=1)
    graph = coo_matrix(
        (
            np.concatenate([edge_len, edge_len], axis=0),
            (np.concatenate([src, dst], axis=0), np.concatenate([dst, src], axis=0)),
        ),
        shape=(vertex_count, vertex_count),
    ).tocsr()
    head_seeds = np.asarray(list(SMAL_HEAD_LANDMARKS.values()), dtype=np.int64)
    boundary_seeds = np.asarray(list(SMAL_HEAD_BOUNDARY_LANDMARKS.values()), dtype=np.int64)
    head_dist = dijkstra(graph, directed=False, indices=head_seeds, min_only=True)
    boundary_dist = dijkstra(graph, directed=False, indices=boundary_seeds, min_only=True)

    # Use the head-vs-neck geodesic watershed as the primary anatomy signal.
    # The margin includes the upper neck/mane area that the user wants treated
    # as head-detail guide roots, without using an image-space box.
    head_like = head_dist <= boundary_dist * float(neck_margin)
    head_like[head_seeds] = True

    # Split labels are coarse but explicit; they let the same output path be
    # consumed by the annotation-transfer guide selector.
    eye_ear_nose = vertices[head_seeds]
    center = eye_ear_nose.mean(axis=0)
    local_dist = np.linalg.norm(vertices - center[None, :], axis=1)
    local_limit = np.quantile(local_dist[head_like], 0.60)
    face = head_like & (local_dist <= local_limit)
    neck_mane = head_like & ~face
    ear_seed_set = np.asarray([SMAL_HEAD_LANDMARKS["left_ear_tip"], SMAL_HEAD_LANDMARKS["right_ear_tip"]], dtype=np.int64)
    ear_dist = dijkstra(graph, directed=False, indices=ear_seed_set, min_only=True)
    ear_limit = np.quantile(ear_dist[head_like], 0.18)
    ears = head_like & (ear_dist <= ear_limit)
    face = face & ~ears
    neck_mane = neck_mane & ~ears
    return {
        "face": np.nonzero(face)[0].astype(np.int64),
        "ears": np.nonzero(ears)[0].astype(np.int64),
        "mane": np.nonzero(neck_mane)[0].astype(np.int64),
    }


def transfer_nearest_labels(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    source_annotations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    vertex_to_label: dict[int, str] = {}
    for label, indices in source_annotations.items():
        for index in indices.tolist():
            vertex_to_label[int(index)] = label

    tree = cKDTree(source_vertices.astype(np.float64, copy=False))
    _, nearest = tree.query(target_vertices.astype(np.float64, copy=False), k=1)

    transferred: dict[str, list[int]] = {}
    for target_index, source_index in enumerate(nearest.tolist()):
        label = vertex_to_label.get(int(source_index))
        if label is None:
            continue
        transferred.setdefault(label, []).append(int(target_index))

    return {
        label: np.asarray(indices, dtype=np.int64)
        for label, indices in transferred.items()
        if len(indices) > 0
    }


def add_body_label(labels: dict[str, np.ndarray], vertex_count: int) -> dict[str, np.ndarray]:
    used = np.zeros((vertex_count,), dtype=bool)
    for indices in labels.values():
        valid = indices[(indices >= 0) & (indices < vertex_count)]
        used[valid] = True
    complete = dict(labels)
    complete["body"] = np.nonzero(~used)[0].astype(np.int64)
    return complete


def write_annotation_json(path: Path, labels: dict[str, np.ndarray]) -> None:
    serializable = {label: indices.astype(int).tolist() for label, indices in labels.items()}
    path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")


def export_colored_labels(path: Path, mesh: trimesh.Trimesh, labels: dict[str, np.ndarray]) -> None:
    names = sorted(labels.keys())
    rng = np.random.default_rng(20260701)
    palette = rng.integers(40, 240, size=(len(names), 3), dtype=np.uint8)
    colors = np.full((len(mesh.vertices), 4), 255, dtype=np.uint8)
    for label_index, label in enumerate(names):
        indices = labels[label]
        valid = indices[(indices >= 0) & (indices < len(colors))]
        colors[valid, :3] = palette[label_index]
    mesh_out = mesh.copy()
    mesh_out.visual = trimesh.visual.ColorVisuals(mesh_out, vertex_colors=colors)
    mesh_out.export(path)


def make_vertex_mask(labels: dict[str, np.ndarray], selected_labels: set[str], vertex_count: int) -> np.ndarray:
    mask = np.zeros((vertex_count,), dtype=bool)
    for label in selected_labels:
        indices = labels.get(label)
        if indices is None:
            continue
        valid = indices[(indices >= 0) & (indices < vertex_count)]
        mask[valid] = True
    return mask


def candidate_region_scores(candidates: SurfaceCandidates, faces: np.ndarray, vertex_mask: np.ndarray) -> np.ndarray:
    face_vertex_mask = vertex_mask[faces[candidates.face_ids]]
    return (face_vertex_mask.astype(np.float32) * candidates.barycentric).sum(axis=1)


def select_region_roots(
    candidates: SurfaceCandidates,
    region_scores: np.ndarray,
    count: int,
    *,
    threshold: float,
    seed: int,
    device: str,
) -> np.ndarray:
    valid = np.nonzero(region_scores >= float(threshold))[0]
    if valid.size < count:
        raise RuntimeError(
            f"region has only {valid.size} candidates above threshold {threshold}, "
            f"cannot select {count} roots"
        )
    selected_local = farthest_point_sample(
        candidates.points[valid],
        count,
        seed=seed,
        device=device,
        start="centroid",
    )
    return valid[selected_local].astype(np.int64)


def save_roots(path: Path, candidates: SurfaceCandidates, selected: np.ndarray, metadata: dict[str, object]) -> None:
    np.savez_compressed(
        path,
        points=candidates.points[selected].astype(np.float32),
        face_ids=candidates.face_ids[selected].astype(np.int64),
        barycentric=candidates.barycentric[selected].astype(np.float32),
        selected_candidate_ids=selected.astype(np.int64),
        candidate_count=np.asarray([candidates.points.shape[0]], dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False), dtype=object),
    )


def list_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def parse_views(text: str) -> list[int]:
    views = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not views:
        raise ValueError("at least one validation view is required")
    return views


def load_projection_cameras(data_root: Path, projection_file: str) -> tuple[np.ndarray, np.ndarray]:
    import cv2 as cv

    path = data_root / projection_file
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if "arr_0" not in data.files:
            raise ValueError(f"{path} does not contain arr_0")
        projections = data["arr_0"].astype(np.float64)
    else:
        projections = np.load(path).astype(np.float64)
    ks: list[np.ndarray] = []
    viewmats: list[np.ndarray] = []
    for projection in projections:
        k, r, t, *_ = cv.decomposeProjectionMatrix(projection[:3, :4])
        k = k / k[2, 2]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = r.T
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        ks.append(k.astype(np.float32))
        viewmats.append(np.linalg.inv(pose).astype(np.float32))
    return np.stack(viewmats), np.stack(ks)


def transform_points(points: np.ndarray, scale: float, translation: np.ndarray) -> np.ndarray:
    return (points.astype(np.float32) * float(scale) + translation.astype(np.float32)[None]).astype(np.float32)


def project_points(points: np.ndarray, viewmat: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (viewmat.astype(np.float32) @ points_h.T).T[:, :3]
    depth = cam[:, 2]
    pix_h = (k.astype(np.float32) @ cam.T).T
    xy = pix_h[:, :2] / np.maximum(pix_h[:, 2:3], 1.0e-8)
    return xy, depth


def face_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(norm, 1.0e-8)).astype(np.float32)


def draw_overlay_points(
    image_path: Path,
    output_path: Path,
    xy: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int,
    max_points: int,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    valid = (
        mask
        & np.isfinite(xy).all(axis=1)
        & np.isfinite(depth)
        & (depth > 1.0e-6)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    ids = np.nonzero(valid)[0]
    if ids.size > max_points:
        rng = np.random.default_rng(20260701)
        ids = rng.choice(ids, size=int(max_points), replace=False)
    rgba = (int(color[0]), int(color[1]), int(color[2]), 175)
    for idx in ids:
        x, y = float(xy[idx, 0]), float(xy[idx, 1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rgba)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_root_regions(
    image_path: Path,
    output_path: Path,
    root_xy: np.ndarray,
    root_depth: np.ndarray,
    region_ids: np.ndarray,
    visible_mask: np.ndarray,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    valid = (
        visible_mask.astype(bool)
        &
        np.isfinite(root_xy).all(axis=1)
        & np.isfinite(root_depth)
        & (root_depth > 1.0e-6)
        & (root_xy[:, 0] >= 0)
        & (root_xy[:, 0] < width)
        & (root_xy[:, 1] >= 0)
        & (root_xy[:, 1] < height)
    )
    for idx in np.nonzero(valid)[0].tolist():
        x, y = float(root_xy[idx, 0]), float(root_xy[idx, 1])
        if int(region_ids[idx]) == 1:
            color = (255, 30, 155, 220)
            radius = 4
        else:
            color = (35, 180, 255, 190)
            radius = 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def write_validation_images(
    output_dir: Path,
    data_root: Path,
    projection_file: str,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    head_face_mask: np.ndarray,
    root_points: np.ndarray,
    root_face_ids: np.ndarray,
    region_ids: np.ndarray,
    *,
    views: list[int],
    scale: float,
    translation: np.ndarray,
    device: str,
) -> list[int]:
    image_paths = list_image_files(data_root / "images")
    if not image_paths:
        raise RuntimeError(f"no validation images found under {data_root / 'images'}")
    viewmats, ks = load_projection_cameras(data_root, projection_file)
    face_centers = mesh_vertices[mesh_faces].mean(axis=1)
    face_centers_world = transform_points(face_centers, scale, translation)
    root_world = transform_points(root_points, scale, translation)
    mesh_world = TriangleMesh(
        vertices=transform_points(mesh_vertices, scale, translation),
        faces=mesh_faces.astype(np.int64, copy=False),
    )
    face_normals = face_normals_np(mesh_vertices, mesh_faces)
    face_center_normals = face_normals
    root_normals = face_normals[root_face_ids.astype(np.int64)]
    torch_device = torch.device("cuda" if str(device) == "auto" else str(device))
    if torch_device.type != "cuda":
        raise RuntimeError("validation image depth visibility requires CUDA; do not use a CPU fallback")
    written: list[int] = []
    visibility_stats: list[dict[str, int | float]] = []
    for view in views:
        if view < 0 or view >= len(image_paths) or view >= viewmats.shape[0]:
            raise ValueError(f"validation view {view} is outside image/camera count")
        with Image.open(image_paths[view]) as image:
            width, height = image.size
        viewmat_t = torch.as_tensor(viewmats[view], dtype=torch.float32, device=torch_device)
        k_t = torch.as_tensor(ks[view], dtype=torch.float32, device=torch_device)
        mesh_depth = render_mesh_depth(
            mesh_world,
            viewmat_t,
            k_t,
            width,
            height,
            device=torch_device,
        )
        face_vis = sample_mesh_visible_points(
            torch.as_tensor(face_centers_world, dtype=torch.float32, device=torch_device),
            torch.as_tensor(face_center_normals, dtype=torch.float32, device=torch_device),
            viewmat_t,
            k_t,
            mesh_depth.depth,
            front_normal_z=0.15,
        )
        root_vis = sample_mesh_visible_points(
            torch.as_tensor(root_world, dtype=torch.float32, device=torch_device),
            torch.as_tensor(root_normals, dtype=torch.float32, device=torch_device),
            viewmat_t,
            k_t,
            mesh_depth.depth,
            front_normal_z=0.15,
        )
        face_xy = face_vis.xy.detach().cpu().numpy()
        face_depth = face_vis.depth.detach().cpu().numpy()
        face_visible = face_vis.visible.detach().cpu().numpy()
        root_xy = root_vis.xy.detach().cpu().numpy()
        root_depth = root_vis.depth.detach().cpu().numpy()
        root_visible = root_vis.visible.detach().cpu().numpy()
        visible_head_roots = root_visible & (region_ids.astype(np.int64) == 1)
        visible_body_roots = root_visible & (region_ids.astype(np.int64) == 0)
        visibility_stats.append(
            {
                "view": int(view),
                "visible_head_roots": int(np.count_nonzero(visible_head_roots)),
                "visible_body_roots": int(np.count_nonzero(visible_body_roots)),
                "visible_head_faces": int(np.count_nonzero(head_face_mask.astype(bool) & face_visible)),
                "visible_face_centers": int(np.count_nonzero(face_visible)),
                "visible_roots": int(np.count_nonzero(root_visible)),
            }
        )
        draw_overlay_points(
            image_paths[view],
            output_dir / f"view{view:02d}_head_face_mask_overlay.png",
            face_xy,
            face_depth,
            head_face_mask.astype(bool) & face_visible,
            color=(255, 35, 155),
            radius=2,
            max_points=18000,
        )
        draw_root_regions(
            image_paths[view],
            output_dir / f"view{view:02d}_head_body_guide_roots_overlay.png",
            root_xy,
            root_depth,
            region_ids,
            root_visible,
        )
        written.append(int(view))
    (output_dir / "validation_visibility_report.json").write_text(
        json.dumps(visibility_stats, indent=2) + "\n",
        encoding="utf-8",
    )
    return written


def parse_labels(value: str) -> list[str]:
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise ValueError("at least one head label is required")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build white-tiger head/body guide roots using the NeuralFur SMAL "
            "body-part annotation transfer path. No heuristic fallback is used."
        )
    )
    parser.add_argument("--mode", choices=["annotation-transfer", "template-anatomy"], default="annotation-transfer")
    parser.add_argument("--source-smal-mesh", type=Path)
    parser.add_argument("--source-annotation-json", type=Path)
    parser.add_argument(
        "--source-smal-template-pkl",
        type=Path,
        default=Path("D:/petsgaussianhair/data/external_data/smal/my_smpl_39dogsnorm_newv3_dog.pkl"),
    )
    parser.add_argument(
        "--target-mesh",
        type=Path,
        default=Path("D:/petsgaussianhair/data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--head-root-count", type=int, required=True)
    parser.add_argument("--body-root-count", type=int, required=True)
    parser.add_argument("--head-labels", default=",".join(DEFAULT_HEAD_LABELS))
    parser.add_argument("--region-threshold", type=float, default=0.5)
    parser.add_argument("--template-neck-margin", type=float, default=1.18)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", type=Path, default=Path("D:/petsgaussianhair/data/neuralfur_work/whiteTiger_processed/roaringwalk"))
    parser.add_argument("--alignment-config", type=Path, default=Path("D:/petsgaussianhair/configs/white_tiger_mesh_alignment.json"))
    parser.add_argument("--projection-file", default="cameras.npz")
    parser.add_argument("--validation-views", default="0,9,18,27")
    parser.add_argument("--skip-validation-images", action="store_true")
    args = parser.parse_args()
    target_mesh_path = require_file(args.target_mesh, "target mesh")

    if args.candidate_count <= 0:
        raise ValueError("--candidate-count must be positive")
    if args.head_root_count <= 0 or args.body_root_count <= 0:
        raise ValueError("--head-root-count and --body-root-count must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_mesh_tri = read_obj_mesh(target_mesh_path)
    target_mesh = trimesh.load_mesh(target_mesh_path, process=False)
    if not isinstance(target_mesh, trimesh.Trimesh):
        raise TypeError(f"target mesh is not a Trimesh: {target_mesh_path}")

    source_annotation_path: Path | None = None
    if args.mode == "annotation-transfer":
        if args.source_smal_mesh is None or args.source_annotation_json is None:
            raise ValueError("--source-smal-mesh and --source-annotation-json are required in annotation-transfer mode")
        source_mesh_path = require_file(args.source_smal_mesh, "SMAL source mesh")
        source_annotation_path = require_file(args.source_annotation_json, "SMAL source annotation JSON")
        source_mesh = trimesh.load_mesh(source_mesh_path, process=False)
        if not isinstance(source_mesh, trimesh.Trimesh):
            raise TypeError(f"source mesh is not a Trimesh: {source_mesh_path}")
        source_mesh = apply_neuralfur_white_tiger_smal_transform(source_mesh)
        source_vertices_for_transfer = np.asarray(source_mesh.vertices, dtype=np.float32)
        source_annotations = load_annotation(source_annotation_path)
    else:
        source_mesh_path = require_file(args.source_smal_template_pkl, "SMAL template pickle")
        source_mesh = load_smal_template_mesh(source_mesh_path)
        source_mesh = apply_neuralfur_white_tiger_smal_transform(source_mesh)
        source_annotations = smal_template_anatomy_labels(source_mesh, neck_margin=args.template_neck_margin)
        source_vertices_for_transfer = align_source_bbox_to_target(
            np.asarray(source_mesh.vertices, dtype=np.float32),
            target_mesh_tri.vertices,
        )

    labels = transfer_nearest_labels(
        source_vertices_for_transfer,
        target_mesh_tri.vertices,
        source_annotations,
    )
    labels = add_body_label(labels, target_mesh_tri.vertex_count)

    transferred_json = args.output_dir / "annotations_furless_reshaped_smal_transfer.json"
    write_annotation_json(transferred_json, labels)
    export_colored_labels(args.output_dir / "furless_reshaped_smal_parts.ply", target_mesh, labels)

    head_labels = parse_labels(args.head_labels)
    selected_head_labels = set(head_labels)
    existing_head_labels = sorted(label for label in selected_head_labels if label in labels)
    if not existing_head_labels:
        raise RuntimeError(
            f"none of the requested head labels exist after transfer: {head_labels}; "
            f"available labels: {sorted(labels.keys())}"
        )

    head_vertex_mask = make_vertex_mask(labels, set(existing_head_labels), target_mesh_tri.vertex_count)
    head_face_score = head_vertex_mask[target_mesh_tri.faces].mean(axis=1)
    head_face_mask = head_face_score >= float(args.region_threshold)

    candidates = sample_surface_candidates(target_mesh_tri, args.candidate_count, args.seed)
    head_scores = candidate_region_scores(candidates, target_mesh_tri.faces, head_vertex_mask)
    head_selected = select_region_roots(
        candidates,
        head_scores,
        args.head_root_count,
        threshold=args.region_threshold,
        seed=args.seed + 11,
        device=args.device,
    )
    body_selected = select_region_roots(
        candidates,
        1.0 - head_scores,
        args.body_root_count,
        threshold=args.region_threshold,
        seed=args.seed + 23,
        device=args.device,
    )
    combined_selected = np.concatenate([head_selected, body_selected], axis=0)
    region_ids = np.concatenate(
        [
            np.ones((head_selected.shape[0],), dtype=np.int64),
            np.zeros((body_selected.shape[0],), dtype=np.int64),
        ],
        axis=0,
    )

    np.savez_compressed(
        args.output_dir / "white_tiger_smal_head_mask.npz",
        head_vertex_mask=head_vertex_mask,
        head_face_mask=head_face_mask,
        head_face_score=head_face_score.astype(np.float32),
        head_labels=np.asarray(existing_head_labels, dtype=object),
    )
    save_roots(
        args.output_dir / "white_tiger_smal_head_body_guide_roots.npz",
        candidates,
        combined_selected,
        {
            "region_id_meaning": {"0": "body", "1": "head"},
            "head_count": int(args.head_root_count),
            "body_count": int(args.body_root_count),
            "head_labels": existing_head_labels,
        },
    )
    with np.load(args.output_dir / "white_tiger_smal_head_body_guide_roots.npz", allow_pickle=True) as data:
        root_payload = {key: data[key] for key in data.files}
    root_payload["region_ids"] = region_ids
    np.savez_compressed(args.output_dir / "white_tiger_smal_head_body_guide_roots.npz", **root_payload)

    validation_views: list[int] = []
    if not args.skip_validation_images:
        alignment = read_json(args.alignment_config) if args.alignment_config.is_file() else {}
        transform = alignment.get("mesh_to_camera_initial", {}) if isinstance(alignment.get("mesh_to_camera_initial", {}), dict) else {}
        mesh_scale = float(transform.get("scale", 1.0))
        mesh_translation = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
        if mesh_translation.shape != (3,):
            raise ValueError("alignment mesh_to_camera_initial.translation must have three values")
        validation_views = write_validation_images(
            args.output_dir / "validation_images",
            args.data_root,
            args.projection_file,
            target_mesh_tri.vertices,
            target_mesh_tri.faces,
            head_face_mask,
            candidates.points[combined_selected],
            candidates.face_ids[combined_selected],
            region_ids,
            views=parse_views(args.validation_views),
            scale=mesh_scale,
            translation=mesh_translation,
            device=args.device,
        )

    report = TransferReport(
        mode=str(args.mode),
        source_mesh=str(source_mesh_path),
        target_mesh=str(target_mesh_path),
        source_annotation=None if source_annotation_path is None else str(source_annotation_path),
        output_dir=str(args.output_dir),
        transferred_labels={label: int(indices.shape[0]) for label, indices in sorted(labels.items())},
        body_vertex_count=int(labels["body"].shape[0]),
        head_labels=existing_head_labels,
        head_vertex_count=int(head_vertex_mask.sum()),
        head_face_count=int(head_face_mask.sum()),
        body_candidate_count=int(np.count_nonzero((1.0 - head_scores) >= args.region_threshold)),
        head_candidate_count=int(np.count_nonzero(head_scores >= args.region_threshold)),
        selected_body_roots=int(body_selected.shape[0]),
        selected_head_roots=int(head_selected.shape[0]),
        validation_views=validation_views,
    )
    (args.output_dir / "smal_head_guide_report.json").write_text(
        json.dumps(asdict(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()

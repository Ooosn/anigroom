"""Mesh-surface root initialization for AniGroom.

This module is intentionally narrow: it reads a mesh, samples dense surface
candidates, and selects approximately uniform roots with farthest point
sampling. It does not contain rendering, training, or grooming losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True)
class SurfaceCandidates:
    points: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray


@dataclass(frozen=True)
class SurfaceRoots:
    points: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray
    selected_candidate_ids: np.ndarray
    candidate_count: int


def read_obj_mesh(path: Path) -> TriangleMesh:
    """Read vertices and triangulated faces from an OBJ file."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                items = line.strip().split()[1:]
                ids = [int(item.split("/")[0]) - 1 for item in items]
                if len(ids) < 3:
                    continue
                for idx in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[idx], ids[idx + 1]])
    if not vertices:
        raise ValueError(f"no vertices found in OBJ: {path}")
    if not faces:
        raise ValueError(f"no faces found in OBJ: {path}")
    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
    )


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return (0.5 * np.linalg.norm(cross, axis=-1)).astype(np.float64)


def barycentric_to_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    tri = vertices[faces[face_ids]]
    return (tri * barycentric[:, :, None]).sum(axis=1).astype(np.float32)


def random_barycentric(count: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random barycentric coordinates over a triangle."""

    u = rng.random(count, dtype=np.float32)
    v = rng.random(count, dtype=np.float32)
    sqrt_u = np.sqrt(u)
    b0 = 1.0 - sqrt_u
    b1 = sqrt_u * (1.0 - v)
    b2 = sqrt_u * v
    return np.stack([b0, b1, b2], axis=-1).astype(np.float32)


def sample_surface_candidates(mesh: TriangleMesh, count: int, seed: int) -> SurfaceCandidates:
    """Sample dense candidate roots from mesh faces proportional to area."""

    if count <= 0:
        raise ValueError("candidate count must be positive")
    areas = triangle_areas(mesh.vertices, mesh.faces)
    total_area = float(areas.sum())
    if total_area <= 0.0:
        raise ValueError("mesh has zero total face area")
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(mesh.face_count, size=int(count), replace=True, p=areas / total_area)
    barycentric = random_barycentric(int(count), rng)
    points = barycentric_to_points(mesh.vertices, mesh.faces, face_ids, barycentric)
    return SurfaceCandidates(
        points=points,
        face_ids=face_ids.astype(np.int64),
        barycentric=barycentric.astype(np.float32),
    )


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def farthest_point_sample(
    points: np.ndarray,
    count: int,
    *,
    seed: int,
    device: str | torch.device = "auto",
    chunk_size: int = 262_144,
    start: Literal["centroid", "random"] = "centroid",
) -> np.ndarray:
    """Run exact FPS over a candidate point set.

    The implementation keeps only one distance vector and updates it in chunks,
    so memory stays bounded. Runtime is still O(candidate_count * root_count);
    use this as the correctness path and run it on CUDA for larger root counts.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    candidate_count = int(points.shape[0])
    if count <= 0:
        raise ValueError("sample count must be positive")
    if count > candidate_count:
        raise ValueError(f"cannot sample {count} roots from {candidate_count} candidates")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    resolved = _resolve_device(device)
    pts = torch.as_tensor(points, dtype=torch.float32, device=resolved)
    selected = torch.empty((int(count),), dtype=torch.long, device=resolved)
    min_dist = torch.full((candidate_count,), torch.inf, dtype=torch.float32, device=resolved)

    if start == "random":
        generator = torch.Generator(device=resolved)
        generator.manual_seed(int(seed))
        current = torch.randint(0, candidate_count, (1,), generator=generator, device=resolved).long()[0]
    elif start == "centroid":
        centroid = pts.mean(dim=0, keepdim=True)
        dist_to_center = (pts - centroid).square().sum(dim=1)
        current = torch.argmax(dist_to_center)
    else:
        raise ValueError(f"unknown FPS start mode: {start}")

    for sample_idx in range(int(count)):
        selected[sample_idx] = current
        current_point = pts[current : current + 1]
        for begin in range(0, candidate_count, int(chunk_size)):
            end = min(begin + int(chunk_size), candidate_count)
            dist = (pts[begin:end] - current_point).square().sum(dim=1)
            min_dist[begin:end] = torch.minimum(min_dist[begin:end], dist)
        current = torch.argmax(min_dist)

    return selected.cpu().numpy().astype(np.int64)


def initialize_surface_roots_fps(
    mesh: TriangleMesh,
    root_count: int,
    *,
    candidate_multiplier: float = 20.0,
    min_candidate_count: int | None = None,
    seed: int = 13,
    fps_device: str | torch.device = "auto",
    fps_chunk_size: int = 262_144,
) -> SurfaceRoots:
    """Create surface roots with dense candidates followed by FPS."""

    if root_count <= 0:
        raise ValueError("root_count must be positive")
    candidate_count = int(np.ceil(float(root_count) * float(candidate_multiplier)))
    if min_candidate_count is not None:
        candidate_count = max(candidate_count, int(min_candidate_count))
    candidate_count = max(candidate_count, int(root_count))
    candidates = sample_surface_candidates(mesh, candidate_count, seed)
    selected = farthest_point_sample(
        candidates.points,
        int(root_count),
        seed=seed,
        device=fps_device,
        chunk_size=fps_chunk_size,
    )
    return SurfaceRoots(
        points=candidates.points[selected].astype(np.float32),
        face_ids=candidates.face_ids[selected].astype(np.int64),
        barycentric=candidates.barycentric[selected].astype(np.float32),
        selected_candidate_ids=selected.astype(np.int64),
        candidate_count=int(candidate_count),
    )


def validate_surface_roots(
    mesh: TriangleMesh,
    roots: SurfaceRoots,
    *,
    tolerance: float = 1.0e-4,
) -> dict[str, float]:
    """Return numeric checks for root legality and surface reconstruction error."""

    bary = roots.barycentric
    reconstructed = barycentric_to_points(mesh.vertices, mesh.faces, roots.face_ids, bary)
    error = np.linalg.norm(reconstructed - roots.points, axis=1)
    bary_sum_error = np.abs(bary.sum(axis=1) - 1.0)
    return {
        "root_count": float(roots.points.shape[0]),
        "candidate_count": float(roots.candidate_count),
        "unique_face_fraction": float(np.unique(roots.face_ids).size / max(mesh.face_count, 1)),
        "bary_min": float(bary.min()),
        "bary_max": float(bary.max()),
        "bary_sum_error_max": float(bary_sum_error.max(initial=0.0)),
        "surface_error_mean": float(error.mean() if error.size else 0.0),
        "surface_error_max": float(error.max(initial=0.0)),
        "valid_barycentric": float(bool((bary >= -tolerance).all() and (bary <= 1.0 + tolerance).all())),
        "valid_surface": float(bool((error <= tolerance).all())),
    }


def save_surface_roots(path: Path, roots: SurfaceRoots, report: dict[str, float] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "root_positions": roots.points.astype(np.float32),
        "face_ids": roots.face_ids.astype(np.int64),
        "barycentric": roots.barycentric.astype(np.float32),
        "selected_candidate_ids": roots.selected_candidate_ids.astype(np.int64),
        "candidate_count": np.asarray([roots.candidate_count], dtype=np.int64),
    }
    if report is not None:
        for key, value in report.items():
            payload[f"report_{key}"] = np.asarray([value], dtype=np.float64)
    np.savez_compressed(path, **payload)

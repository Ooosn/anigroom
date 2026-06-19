from pathlib import Path

import numpy as np


EPS = 1e-8


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def load_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = vertices[faces]
    centers = tri.mean(axis=1)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    normals = normalize(cross)
    return centers.astype(np.float32), normals.astype(np.float32), areas.astype(np.float32)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def keep_largest_face_component(faces: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if len(candidates) < 2:
        return candidates
    local = {int(face_id): i for i, face_id in enumerate(candidates)}
    vertex_to_faces: dict[int, list[int]] = {}
    for face_id in candidates:
        li = local[int(face_id)]
        for vertex_id in faces[int(face_id)]:
            vertex_to_faces.setdefault(int(vertex_id), []).append(li)
    uf = _UnionFind(len(candidates))
    for linked in vertex_to_faces.values():
        if len(linked) < 2:
            continue
        first = linked[0]
        for other in linked[1:]:
            uf.union(first, other)
    labels = np.array([uf.find(i) for i in range(len(candidates))], dtype=np.int64)
    roots, counts = np.unique(labels, return_counts=True)
    keep_root = roots[int(np.argmax(counts))]
    kept = candidates[labels == keep_root]
    return kept if len(kept) > 0 else candidates


def infer_body_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(points), 1)
    eigval, eigvec = np.linalg.eigh(cov)
    axis = eigvec[:, int(np.argmax(eigval))].astype(np.float32)
    if axis @ np.array([0.0, 0.0, 1.0], dtype=np.float32) < 0:
        axis = -axis
    return normalize(axis[None, :])[0]


def stable_frame(normals: np.ndarray, body_axis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    world_long = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    world_side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    gravity = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if body_axis is None:
        body_axis = world_long
    body_axis = normalize(body_axis[None, :])[0]
    body_tangent = body_axis[None, :] - (normals @ body_axis)[:, None] * normals
    gravity_tangent = gravity[None, :] - (normals @ gravity)[:, None] * normals
    tangent = body_tangent + 0.42 * gravity_tangent
    bad = np.linalg.norm(tangent, axis=1) < 1e-4
    if np.any(bad):
        tangent[bad] = world_side[None, :] - (normals[bad] @ world_side)[:, None] * normals[bad]
    tangent = normalize(tangent)
    bitangent = normalize(np.cross(normals, tangent))
    return tangent.astype(np.float32), bitangent.astype(np.float32)


def sample_body_patch_roots(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers, normals, areas = face_geometry(vertices, faces)
    y0, y1 = np.quantile(centers[:, 1], [0.42, 0.86])
    z0, z1 = np.quantile(centers[:, 2], [0.20, 0.74])
    x0, x1 = np.quantile(centers[:, 0], [0.06, 0.94])
    mask = (
        (centers[:, 1] >= y0)
        & (centers[:, 1] <= y1)
        & (centers[:, 2] >= z0)
        & (centers[:, 2] <= z1)
        & (centers[:, 0] >= x0)
        & (centers[:, 0] <= x1)
        & (areas > 0)
    )
    candidates = np.where(mask)[0]
    if len(candidates) < count:
        candidates = np.where(areas > 0)[0]
    candidates = keep_largest_face_component(faces, candidates)
    body_axis = infer_body_axis(centers[candidates])
    rng = np.random.default_rng(seed)
    probs = areas[candidates] / np.maximum(areas[candidates].sum(), EPS)
    chosen = rng.choice(candidates, size=count, replace=True, p=probs)
    tri = vertices[faces[chosen]]
    r1 = rng.random(count, dtype=np.float32)
    r2 = rng.random(count, dtype=np.float32)
    sr1 = np.sqrt(r1)
    bary = np.stack([1.0 - sr1, sr1 * (1.0 - r2), sr1 * r2], axis=1).astype(np.float32)
    roots = (tri * bary[:, :, None]).sum(axis=1)
    tangent, bitangent = stable_frame(normals[chosen], body_axis)
    return roots.astype(np.float32), normals[chosen], tangent, bitangent

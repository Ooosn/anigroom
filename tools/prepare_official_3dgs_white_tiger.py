"""Prepare the processed white-tiger capture for the official 3DGS code.

The official GraphDeco loader is safest through its COLMAP-text path here:
the processed data already stores world-to-camera matrices, while the Blender
JSON path applies an extra OpenGL-to-COLMAP axis conversion.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rot.flat
    k = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec / np.linalg.norm(qvec)


def read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z, *_ = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                items = line.strip().split()[1:]
                idx = [int(item.split("/")[0]) - 1 for item in items]
                if len(idx) >= 3:
                    for j in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[j], idx[j + 1]])
    if not vertices:
        raise ValueError(f"no vertices found in {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def sample_mesh_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if faces.size == 0 or count <= len(vertices):
        if count <= len(vertices):
            choice = rng.choice(len(vertices), size=count, replace=False)
            return vertices[choice]
        choice = rng.choice(len(vertices), size=count, replace=True)
        return vertices[choice]

    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    areas = np.maximum(areas, 1.0e-12)
    probs = areas / areas.sum()
    chosen = rng.choice(len(faces), size=count, replace=True, p=probs)
    picked = tri[chosen]
    u = rng.random((count, 1), dtype=np.float32)
    v = rng.random((count, 1), dtype=np.float32)
    swap = (u + v) > 1.0
    u[swap] = 1.0 - u[swap]
    v[swap] = 1.0 - v[swap]
    return picked[:, 0] + u * (picked[:, 1] - picked[:, 0]) + v * (picked[:, 2] - picked[:, 0])


def write_points3d(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for idx, p in enumerate(points, start=1):
            handle.write(
                f"{idx} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{color[0]} {color[1]} {color[2]} 0.0\n"
            )


def prepare_scene(
    data_root: Path,
    mesh_path: Path,
    output_dir: Path,
    point_count: int,
    test_stride: int,
    seed: int,
) -> None:
    image_paths = list_images(data_root / "images")
    if not image_paths:
        raise ValueError(f"no images found under {data_root / 'images'}")

    intr = np.load(data_root / "cameras_intr.npy").astype(np.float64)
    extr = np.load(data_root / "cameras_extr.npy").astype(np.float64)
    if len(image_paths) != len(intr) or len(image_paths) != len(extr):
        raise ValueError(
            f"image/camera count mismatch: images={len(image_paths)} intr={len(intr)} extr={len(extr)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_out = output_dir / "images"
    sparse_out = output_dir / "sparse" / "0"
    if image_out.exists():
        shutil.rmtree(image_out)
    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    image_out.mkdir(parents=True)
    sparse_out.mkdir(parents=True)

    with Image.open(image_paths[0]) as image:
        width, height = image.size
    fx = float(intr[0, 0, 0])
    fy = float(intr[0, 1, 1])
    cx = float(intr[0, 0, 2])
    cy = float(intr[0, 1, 2])

    with (sparse_out / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"1 PINHOLE {width} {height} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")

    test_names: list[str] = []
    with (sparse_out / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, image_path in enumerate(image_paths):
            name = image_path.name
            shutil.copy2(image_path, image_out / name)
            qvec = rotmat_to_qvec(extr[idx, :3, :3])
            tvec = extr[idx, :3, 3]
            image_id = idx + 1
            handle.write(
                f"{image_id} "
                f"{qvec[0]:.12f} {qvec[1]:.12f} {qvec[2]:.12f} {qvec[3]:.12f} "
                f"{tvec[0]:.12f} {tvec[1]:.12f} {tvec[2]:.12f} "
                f"1 {name}\n\n"
            )
            if test_stride > 0 and idx % test_stride == 0:
                test_names.append(name)

    vertices, faces = read_obj_mesh(mesh_path)
    points = sample_mesh_points(vertices, faces, point_count, seed)
    write_points3d(sparse_out / "points3D.txt", points, color=(210, 210, 210))

    with (sparse_out / "test.txt").open("w", encoding="utf-8") as handle:
        for name in test_names:
            handle.write(name + "\n")

    fovx = 2.0 * math.atan(width / (2.0 * fx))
    fovy = 2.0 * math.atan(height / (2.0 * fy))
    summary = output_dir / "white_tiger_3dgs_scene_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"data_root={data_root}",
                f"mesh_path={mesh_path}",
                f"images={len(image_paths)}",
                f"train={len(image_paths) - len(test_names)}",
                f"test={len(test_names)}",
                f"test_stride={test_stride}",
                f"resolution={width}x{height}",
                f"fx={fx:.6f}",
                f"fy={fy:.6f}",
                f"cx={cx:.6f}",
                f"cy={cy:.6f}",
                f"fovx={fovx:.9f}",
                f"fovy={fovy:.9f}",
                f"point_count={len(points)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--mesh-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--point-count", type=int, default=100_000)
    parser.add_argument("--test-stride", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    prepare_scene(
        data_root=args.data_root,
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        point_count=args.point_count,
        test_stride=args.test_stride,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

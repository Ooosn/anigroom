"""Initialize mesh-surface AniGroom roots with dense candidates + FPS.

This is a verification/asset-preparation tool. It does not train a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anigroom.mesh_roots import (
    initialize_surface_roots_fps,
    read_obj_mesh,
    sample_surface_candidates,
    save_surface_roots,
    validate_surface_roots,
)


def nearest_neighbor_distances(points: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    """Compute nearest-neighbor distances with bounded memory."""

    count = int(points.shape[0])
    if count <= 1:
        return np.zeros((count,), dtype=np.float32)
    out = np.empty((count,), dtype=np.float32)
    points64 = points.astype(np.float32, copy=False)
    for begin in range(0, count, chunk_size):
        end = min(begin + chunk_size, count)
        diff = points64[begin:end, None, :] - points64[None, :, :]
        dist2 = np.einsum("bij,bij->bi", diff, diff)
        row = np.arange(begin, end)
        dist2[np.arange(end - begin), row] = np.inf
        out[begin:end] = np.sqrt(np.min(dist2, axis=1)).astype(np.float32)
    return out


def make_root_visualization(
    mesh_vertices: np.ndarray,
    random_points: np.ndarray,
    fps_points: np.ndarray,
    output_path: Path,
    *,
    max_points: int = 8000,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(123)

    def take(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= max_points:
            return points
        ids = rng.choice(points.shape[0], size=max_points, replace=False)
        return points[ids]

    random_show = take(random_points)
    fps_show = take(fps_points)
    center = mesh_vertices.mean(axis=0)
    span = np.maximum(np.ptp(mesh_vertices, axis=0).max(), 1.0e-8)
    random_show = (random_show - center) / span
    fps_show = (fps_show - center) / span

    random_nn = nearest_neighbor_distances(take(random_points), chunk_size=1024)
    fps_nn = nearest_neighbor_distances(take(fps_points), chunk_size=1024)

    fig = plt.figure(figsize=(14, 9), dpi=180)
    fig.patch.set_facecolor("white")

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.scatter(random_show[:, 0], random_show[:, 2], s=0.45, alpha=0.75, c="#4c78a8", linewidths=0)
    ax0.set_title("Area-random candidates selected directly")
    ax0.set_xlabel("x")
    ax0.set_ylabel("z")
    ax0.set_aspect("equal", adjustable="box")

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.scatter(fps_show[:, 0], fps_show[:, 2], s=0.45, alpha=0.75, c="#f58518", linewidths=0)
    ax1.set_title("Dense candidates + FPS roots")
    ax1.set_xlabel("x")
    ax1.set_ylabel("z")
    ax1.set_aspect("equal", adjustable="box")

    ax2 = fig.add_subplot(2, 2, 3)
    bins = 48
    ax2.hist(random_nn, bins=bins, alpha=0.65, label="random", color="#4c78a8")
    ax2.hist(fps_nn, bins=bins, alpha=0.65, label="fps", color="#f58518")
    ax2.set_title("Nearest-neighbor distance distribution")
    ax2.set_xlabel("distance")
    ax2.set_ylabel("count")
    ax2.legend(frameon=False)

    ax3 = fig.add_subplot(2, 2, 4)
    labels = ["random", "fps"]
    means = [float(random_nn.mean()), float(fps_nn.mean())]
    p10s = [float(np.quantile(random_nn, 0.10)), float(np.quantile(fps_nn, 0.10))]
    ax3.bar(np.arange(2) - 0.16, means, width=0.32, label="mean nn", color="#72b7b2")
    ax3.bar(np.arange(2) + 0.16, p10s, width=0.32, label="p10 nn", color="#e45756")
    ax3.set_xticks(np.arange(2), labels)
    ax3.set_title("Uniformity summary")
    ax3.legend(frameon=False)

    fig.suptitle("Mesh root initialization check: FPS should reduce clumping", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output .npz path.")
    parser.add_argument("--root-count", type=int, default=10000)
    parser.add_argument("--candidate-multiplier", type=float, default=20.0)
    parser.add_argument("--min-candidate-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--fps-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--fps-chunk-size", type=int, default=262144)
    parser.add_argument("--visualization", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh = read_obj_mesh(args.mesh_path)
    roots = initialize_surface_roots_fps(
        mesh,
        args.root_count,
        candidate_multiplier=args.candidate_multiplier,
        min_candidate_count=args.min_candidate_count if args.min_candidate_count > 0 else None,
        seed=args.seed,
        fps_device=args.fps_device,
        fps_chunk_size=args.fps_chunk_size,
    )
    report = validate_surface_roots(mesh, roots)
    report.update(
        {
            "mesh_vertex_count": float(mesh.vertex_count),
            "mesh_face_count": float(mesh.face_count),
            "candidate_multiplier": float(args.candidate_multiplier),
            "seed": float(args.seed),
        }
    )
    save_surface_roots(args.output, roots, report)

    if args.visualization is not None:
        random_candidates = sample_surface_candidates(mesh, args.root_count, args.seed + 1009)
        make_root_visualization(mesh.vertices, random_candidates.points, roots.points, args.visualization)

    report_path = args.report if args.report is not None else args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.collision.mesh_sdf import build_mesh_sdf, save_mesh_sdf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate an outside-positive mesh SDF archive."
    )
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--longest-axis-resolution", type=int, required=True)
    parser.add_argument("--padding-voxels", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=65536)
    parser.add_argument("--validation-samples", type=int, default=4096)
    parser.add_argument("--validation-seed", type=int, default=29)
    parser.add_argument("--sign-ray-samples", type=int, default=5)
    parser.add_argument("--max-p95-error-voxels", type=float, default=1.0)
    parser.add_argument("--min-sign-agreement", type=float, default=0.99)
    parser.add_argument("--close-boundary-loops", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build = build_mesh_sdf(
        args.mesh,
        longest_axis_resolution=args.longest_axis_resolution,
        padding_voxels=args.padding_voxels,
        query_chunk_size=args.query_chunk_size,
        validation_samples=args.validation_samples,
        validation_seed=args.validation_seed,
        close_boundaries=args.close_boundary_loops,
        sign_ray_samples=args.sign_ray_samples,
    )
    p95 = float(build.metadata["absolute_error_voxels_p95"])
    sign = float(build.metadata["normal_offset_expected_sign_agreement"])
    if p95 > float(args.max_p95_error_voxels):
        raise RuntimeError(
            f"SDF p95 interpolation error {p95:.6f} voxels exceeds "
            f"{float(args.max_p95_error_voxels):.6f}"
        )
    if sign < float(args.min_sign_agreement):
        raise RuntimeError(
            f"SDF near-surface sign agreement {sign:.6f} is below "
            f"{float(args.min_sign_agreement):.6f}"
        )
    save_mesh_sdf(build, args.output)
    print(json.dumps(build.metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

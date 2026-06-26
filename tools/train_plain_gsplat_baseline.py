from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.baselines.plain_gsplat import BaselineConfig, train_plain_gsplat_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the plain gsplat white tiger baseline.")
    parser.add_argument("--data-root", default="data/neuralfur_work/whiteTiger_processed/roaringwalk")
    parser.add_argument("--mesh-path", default="data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj")
    parser.add_argument("--output-dir", default="outputs/plain_gsplat_baseline")
    parser.add_argument("--num-gaussians", type=int, default=5000)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--test-stride", type=int, default=6)
    parser.add_argument(
        "--train-indices",
        default="",
        help="Optional comma-separated training view indices for gate tests, e.g. '9' or '1,2,3'.",
    )
    parser.add_argument("--compute-lpips", action="store_true")
    args = parser.parse_args()
    train_indices = (
        tuple(int(part) for part in args.train_indices.split(",") if part.strip())
        if args.train_indices.strip()
        else None
    )

    config = BaselineConfig(
        data_root=args.data_root,
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        num_gaussians=args.num_gaussians,
        iterations=args.iterations,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        test_stride=args.test_stride,
        train_indices=train_indices,
        compute_lpips=args.compute_lpips,
    )
    train_plain_gsplat_baseline(config)


if __name__ == "__main__":
    main()

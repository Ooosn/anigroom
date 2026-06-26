from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.data.white_tiger import build_stage1_input_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Report AniGroom white tiger Stage 1 inputs.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/neuralfur_work/whiteTiger_processed/roaringwalk"),
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=Path("data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj"),
    )
    parser.add_argument("--orientation-dir", default="orientations_2")
    parser.add_argument("--test-stride", type=int, default=6)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = build_stage1_input_report(
        data_root=args.data_root,
        mesh_path=args.mesh_path,
        orientation_dir=args.orientation_dir,
        test_stride=args.test_stride,
    )
    text = json.dumps(report.to_json_dict(), indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report.errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

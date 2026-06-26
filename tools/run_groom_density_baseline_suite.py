from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def cli_name(name: str) -> str:
    return "--" + name.replace("_", "-")


def case_name(style: str, density: str, iterations: int) -> str:
    return f"{style}_{density}_{iterations}"


def build_command(config: dict, case: dict, output_root: Path) -> tuple[list[str], Path, Path]:
    schedule = config[f"{case['schedule']}_schedule"]
    common = config["common_args"]
    iterations = int(schedule["iterations"])
    name = case_name(case["style"], case["density"], iterations)
    case_dir = output_root / name
    log_path = output_root / f"{name}.log"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / config["script"]),
        "--output-dir",
        str(case_dir),
        "--teacher-style",
        case["style"],
        "--density-mode",
        case["density"],
    ]
    merged = {
        **common,
        "iterations": iterations,
        "densify_until": int(schedule["densify_until"]),
        "max_splits_per_event": int(schedule["max_splits_per_event"]),
    }
    for key, value in merged.items():
        cmd.extend([cli_name(key), str(value)])
    return cmd, case_dir, log_path


def load_summary(case_dir: Path) -> dict:
    summary_path = case_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary after run: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_summary_csv(rows: list[dict], output_root: Path) -> None:
    path = output_root / "suite_summary.csv"
    fields = [
        "style",
        "density",
        "iterations",
        "initial_roots",
        "final_roots",
        "initial_psnr",
        "final_psnr",
        "orientation",
        "orientation_detail",
        "flow_coherence",
        "case_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed groom-density baseline suite without fallback.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "groom_density_v22_baseline_suite.json",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--case", action="append", default=None, help="Optional style name filter; can be repeated.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a case only when its summary.json already exists.")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = args.output_root or (PROJECT_ROOT / config["output_root"])
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.case or [])
    rows: list[dict] = []
    for case in config["cases"]:
        if selected and case["style"] not in selected:
            continue
        cmd, case_dir, log_path = build_command(config, case, output_root)
        summary_path = case_dir / "summary.json"
        if args.skip_existing and summary_path.exists():
            print(f"[skip-existing] {case['style']} -> {summary_path}", flush=True)
        else:
            print(f"[run] {case['style']} {case['density']} -> {case_dir}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:])
                raise RuntimeError(f"case failed: {case['style']} exit={result.returncode}\n{tail}")

        summary = load_summary(case_dir)
        last = summary["history"][-1]
        rows.append(
            {
                "style": summary["config"]["teacher_style"],
                "density": summary["config"]["density_mode"],
                "iterations": summary["config"]["iterations"],
                "initial_roots": summary["initial_roots"],
                "final_roots": summary["final_roots"],
                "initial_psnr": f"{summary['initial_psnr']:.4f}",
                "final_psnr": f"{summary['final_psnr']:.4f}",
                "orientation": f"{last['orientation']:.6f}",
                "orientation_detail": f"{last['orientation_detail']:.6f}",
                "flow_coherence": f"{last['flow_coherence']:.6f}",
                "case_dir": str(case_dir),
            }
        )
    write_summary_csv(rows, output_root)
    print(f"[done] wrote {output_root / 'suite_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

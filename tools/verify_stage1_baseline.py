from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = PROJECT_ROOT / "configs" / "stage1_baseline.lock.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_baseline_lock(
    project_root: Path = PROJECT_ROOT,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, object]:
    manifest = json.loads(lock_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = []
    for relative_path, expected in manifest["local_files"].items():
        path = project_root / relative_path
        if not path.is_file():
            failures.append(
                {"path": relative_path, "expected": expected, "actual": "missing"}
            )
            continue
        actual = file_sha256(path)
        if actual != expected:
            failures.append(
                {"path": relative_path, "expected": expected, "actual": actual}
            )
    return {
        "baseline_id": manifest["baseline_id"],
        "checked_files": len(manifest["local_files"]),
        "ok": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen Stage 1 baseline.")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = verify_baseline_lock(lock_path=args.lock.resolve())
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

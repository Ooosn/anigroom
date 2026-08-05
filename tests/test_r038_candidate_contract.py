from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE = PROJECT_ROOT / "configs" / "stage1_baseline.env"
CANDIDATE = PROJECT_ROOT / "configs" / "r038_brush_curve_0_30k.env"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        assert key not in values
        values[key] = value
    return values


def test_r038_changes_only_the_declared_lifecycle_schedule() -> None:
    baseline = load_env(BASELINE)
    candidate = load_env(CANDIDATE)
    assert candidate.keys() == baseline.keys()
    changed = {
        key: (baseline[key], candidate[key])
        for key in baseline
        if baseline[key] != candidate[key]
    }
    assert changed == {
        "DENSIFY_UNTIL": ("20000", "9000"),
        "GUIDE_DENSIFY_INTERVAL": ("200", "0"),
        "GUIDE_DENSIFY_START": ("11000", "0"),
        "GUIDE_DENSIFY_UNTIL": ("16000", "0"),
        "STAGE_SAVE_ITERS": (
            "10000\\,12000\\,14000\\,16000\\,18000\\,20000\\,22000\\,25000\\,27000\\,30000",
            "9000\\,10000\\,12000\\,14000\\,16000\\,18000\\,20000\\,22000\\,25000\\,27000\\,30000",
        ),
    }
    assert candidate["SHAPE_CURL_SCALE"] == "0.0"
    assert candidate["SHAPE_FRIZZ_SCALE"] == "0.0"
    assert candidate["PRUNE_INTERVAL"] == "0"

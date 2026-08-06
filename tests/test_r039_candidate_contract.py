from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R038 = PROJECT_ROOT / "configs" / "r038_brush_curve_0_30k.env"
R039 = PROJECT_ROOT / "configs" / "r039_brush_centerline_0_30k.env"


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


def test_r039_removes_only_the_legacy_bend_configuration() -> None:
    r038 = load_env(R038)
    r039 = load_env(R039)
    removed = {
        "GUIDE_BEND_RESIDUAL_SCALE",
        "GUIDE_PRIOR_BEND_WEIGHT",
    }
    assert set(r039) == set(r038) - removed
    assert all(r039[key] == value for key, value in r038.items() if key not in removed)
    assert all("BEND" not in key for key in r039)

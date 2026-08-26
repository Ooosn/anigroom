from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
R067_CONFIG = ROOT / "configs/r067_no_frizz_0_30k.env"
R068_CONFIG = ROOT / "configs/r068_no_crossing_zero_curl_0_30k.env"
RUNNER = ROOT / "scripts/server/run_r068_no_crossing_zero_curl.sh"
GENERIC_LAUNCHER = ROOT / "scripts/server/run_white_tiger_stage1.sh"


def _bash() -> str | None:
    return shutil.which("bash") or (
        r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else None
    )


def _shell_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if os.name == "nt" and len(value) > 1 and value[1] == ":":
        return f"/{value[0].lower()}{value[2:]}"
    return value


def _snapshot_config(config: Path, output: Path) -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is required for shell config snapshots")
    script = "\n".join(
        [
            "set -euo pipefail",
            "env -i HOME=\"${HOME:-}\" PATH=\"$PATH\" "
            "MESH_NO_PENETRATION_SDF=/tmp/white_tiger_sdf.npz "
            f"bash -c 'set -a; source \"$1\"; env' _ {shlex.quote(_shell_path(config))} "
            f"> {shlex.quote(_shell_path(output))}",
        ]
    )
    result = subprocess.run(
        [bash, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def _load_snapshot(path: Path) -> dict[str, str]:
    ignored = {"PWD", "SHLVL", "_"}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key not in ignored:
            result[key] = value
    return result


def test_r068_config_shell_snapshot_has_only_crossing_delta() -> None:
    source = R068_CONFIG.read_text(encoding="utf-8")
    assert 'source "${CONFIG_DIR}/r067_no_frizz_0_30k.env"' in source
    assert source.count("STRAND_CROSSING_SUPPORT=0") == 1
    assert source.count("STRAND_CROSSING_WEIGHT=0") == 1
    assert source.count("STRAND_CROSSING_REFRESH_INTERVAL=0") == 1
    assert not re.search(r"^STRAND_CROSSING_QUERY_BATCH=", source, re.MULTILINE)
    assert not re.search(r"^STRAND_CROSSING_EXACT_PAIR_BATCH=", source, re.MULTILINE)

    with tempfile.TemporaryDirectory(prefix="r068-config-") as directory:
        root = Path(directory)
        r067_path = root / "r067.env"
        r068_path = root / "r068.env"
        _snapshot_config(R067_CONFIG, r067_path)
        _snapshot_config(R068_CONFIG, r068_path)
        r067 = _load_snapshot(r067_path)
        r068 = _load_snapshot(r068_path)

    delta = {
        key: {"r067": r067.get(key), "r068": r068.get(key)}
        for key in sorted(set(r067) | set(r068))
        if r067.get(key) != r068.get(key)
    }
    assert delta == {
        "STRAND_CROSSING_SUPPORT": {"r067": "1", "r068": "0"},
        "STRAND_CROSSING_WEIGHT": {"r067": "0.001", "r068": "0"},
        "STRAND_CROSSING_REFRESH_INTERVAL": {"r067": "2000", "r068": "0"},
    }
    assert r067["STRAND_CROSSING_QUERY_BATCH"] == r068["STRAND_CROSSING_QUERY_BATCH"] == "50000"
    assert r067["STRAND_CROSSING_EXACT_PAIR_BATCH"] == r068["STRAND_CROSSING_EXACT_PAIR_BATCH"] == "250000"
    assert r068["EXPECTED_WIDTH"] == "1920"
    assert r068["EXPECTED_HEIGHT"] == "1080"
    assert r068["ITERATIONS"] == "30000"


def test_r068_runner_static_contract_is_strict() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    required_fragments = (
        'EXPECTED_COMMIT="${EXPECTED_COMMIT:?',
        '[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]',
        'git status --porcelain=v1',
        '[[ ! -e "$RUNTIME_ROOT" ]]',
        '[[ ! -e "$OUTPUT_DIR" ]]',
        '[[ -x "$PYTHON" ]]',
        '*/mygs|*/mygs/*',
        '[[ -d "$DATA_ROOT" ]]',
        '[[ -f "$MESH_PATH" ]]',
        '[[ -f "$MESH_NO_PENETRATION_SDF" ]]',
        '[[ -f "$CLEAN_FLOW_TARGET" ]]',
        'CLEAN_FLOW_TARGET="$CLEAN_FLOW_TARGET"',
        'INIT_MESH_SCALE="$INIT_MESH_SCALE"',
        'INIT_MESH_TRANSLATION="$INIT_MESH_TRANSLATION"',
        'CURRENT_CHECKPOINT_VERSION[[:space:]]*=[[:space:]]*9',
        'ulimit -v unlimited',
        '"$PYTHON" -m pytest -q',
        'RUN_PREFLIGHT=1',
        'RUN_BATCH_PREFLIGHT=0',
        '-u RESUME_CHECKPOINT',
        '-u RESUME_OPTIMIZER',
        'R068_CONFIG_PATH="$PROJECT_ROOT/configs/r068_no_crossing_zero_curl_0_30k.env"',
        'enable_curl',
        'strand_crossing_active_set',
        'strand_crossing_history',
        'training_metric_records = [',
        'isinstance(record.get("train"), dict)',
        'isinstance(record.get("test"), dict)',
        'checkpoint_version',
        'checkpoint_kind',
        'assert_no_frizz_keys',
    )
    for fragment in required_fragments:
        assert fragment in source, fragment
    assert "--resume-checkpoint" not in source
    assert "|| true" not in source
    assert "for record in metric_records:\n    crossing" not in source


def test_generic_launcher_shell_snapshot_emits_disabled_crossing_args() -> None:
    launcher_source = GENERIC_LAUNCHER.read_text(encoding="utf-8")
    assert 'if [[ "$STRAND_CROSSING_SUPPORT" == "1" ]]' in launcher_source
    assert "--strand-crossing-support" in launcher_source
    assert '"$STRAND_CROSSING_WEIGHT"' in launcher_source
    assert '"$STRAND_CROSSING_REFRESH_INTERVAL"' in launcher_source
    assert '"$STRAND_CROSSING_QUERY_BATCH"' in launcher_source
    assert '"$STRAND_CROSSING_EXACT_PAIR_BATCH"' in launcher_source

    bash = _bash()
    if bash is None:
        pytest.skip("bash is required for generic launcher shell snapshots")

    with tempfile.TemporaryDirectory(prefix="r068-launcher-") as directory:
        root = Path(directory)
        stub = root / "python-stub.sh"
        capture = root / "args.txt"
        output = root / "output"
        sdf = root / "mesh_sdf.npz"
        sdf.write_bytes(b"test placeholder")
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf '%s\\n' \"$@\" > \"$CAPTURE_PATH\"\n",
            encoding="ascii",
        )
        stub.chmod(0o755)
        script = "\n".join(
            [
                "set -euo pipefail",
                f"export PROJECT_ROOT={shlex.quote(_shell_path(ROOT))}",
                f"export PYTHON={shlex.quote(_shell_path(stub))}",
                f"export CAPTURE_PATH={shlex.quote(_shell_path(capture))}",
                f"export DATA_ROOT={shlex.quote(_shell_path(ROOT / 'data'))}",
                f"export MESH_PATH={shlex.quote(_shell_path(ROOT / 'mesh.obj'))}",
                f"export MESH_NO_PENETRATION_SDF={shlex.quote(_shell_path(sdf))}",
                "export RUN_ID=r068_shell_snapshot",
                f"export OUTPUT_DIR={shlex.quote(_shell_path(output))}",
                f"export CONFIG_PATH={shlex.quote(_shell_path(R068_CONFIG))}",
                "export RUN_PREFLIGHT=0",
                "export RUN_BATCH_PREFLIGHT=0",
                "unset RESUME_CHECKPOINT RESUME_OPTIMIZER TRAIN_VIEWS TEST_VIEWS",
                f"bash {shlex.quote(_shell_path(GENERIC_LAUNCHER))}",
            ]
        )
        result = subprocess.run(
            [bash, "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        arguments = capture.read_text(encoding="utf-8").splitlines()

    assert "--strand-crossing-support" not in arguments

    def value(flag: str) -> str:
        index = arguments.index(flag)
        return arguments[index + 1]

    assert value("--strand-crossing-weight") == "0"
    assert value("--strand-crossing-refresh-interval") == "0"
    assert value("--strand-crossing-query-batch") == "50000"
    assert value("--strand-crossing-exact-pair-batch") == "250000"

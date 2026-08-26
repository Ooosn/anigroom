from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "r068_no_crossing_zero_curl_0_30k.env"
LAUNCHER = ROOT / "scripts" / "server" / "run_white_tiger_stage1.sh"


def _bash() -> str | None:
    return shutil.which("bash") or (
        r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else None
    )


def _shell_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if os.name == "nt" and len(value) > 1 and value[1] == ":":
        return f"/{value[0].lower()}{value[2:]}"
    return value


def _run_launcher(
    root: Path,
    *,
    scale: str | None = None,
    translation: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is required for launcher behavior tests")

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

    script = [
        "set -euo pipefail",
        f"export PROJECT_ROOT={shlex.quote(_shell_path(ROOT))}",
        f"export PYTHON={shlex.quote(_shell_path(stub))}",
        f"export CAPTURE_PATH={shlex.quote(_shell_path(capture))}",
        f"export DATA_ROOT={shlex.quote(_shell_path(ROOT / 'data'))}",
        f"export MESH_PATH={shlex.quote(_shell_path(ROOT / 'mesh.obj'))}",
        f"export MESH_NO_PENETRATION_SDF={shlex.quote(_shell_path(sdf))}",
        "export RUN_ID=stage1_mesh_alignment_test",
        f"export OUTPUT_DIR={shlex.quote(_shell_path(output))}",
        f"export CONFIG_PATH={shlex.quote(_shell_path(CONFIG))}",
        "export RUN_PREFLIGHT=0",
        "export RUN_BATCH_PREFLIGHT=0",
        "unset RESUME_CHECKPOINT RESUME_OPTIMIZER TRAIN_VIEWS TEST_VIEWS",
    ]
    if scale is None:
        script.append("unset INIT_MESH_SCALE")
    else:
        script.append(f"export INIT_MESH_SCALE={shlex.quote(scale)}")
    if translation is None:
        script.append("unset INIT_MESH_TRANSLATION")
    else:
        script.append(f"export INIT_MESH_TRANSLATION={shlex.quote(translation)}")
    script.append(f"bash {shlex.quote(_shell_path(LAUNCHER))}")

    result = subprocess.run(
        [bash, "-c", "\n".join(script)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result, capture


def _flag_values(arguments: list[str], flag: str, count: int) -> list[str]:
    index = arguments.index(flag)
    return arguments[index + 1 : index + 1 + count]


def test_launcher_declares_mesh_alignment_defaults_and_flags() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")

    assert 'INIT_MESH_SCALE="${INIT_MESH_SCALE:-1.28}"' in source
    assert 'INIT_MESH_TRANSLATION="${INIT_MESH_TRANSLATION:-0,0.32,0.02}"' in source
    assert "--init-mesh-scale \"$INIT_MESH_SCALE\"" in source
    assert '--init-mesh-translation "${INIT_MESH_TRANSLATION_VALUES[@]}"' in source


def test_launcher_passes_default_mesh_alignment_args(tmp_path: Path) -> None:
    result, capture = _run_launcher(tmp_path)

    assert result.returncode == 0, result.stderr + result.stdout
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert _flag_values(arguments, "--init-mesh-scale", 1) == ["1.28"]
    assert _flag_values(arguments, "--init-mesh-translation", 3) == [
        "0",
        "0.32",
        "0.02",
    ]


def test_launcher_passes_configured_mesh_alignment_args(tmp_path: Path) -> None:
    result, capture = _run_launcher(
        tmp_path,
        scale="1.5",
        translation="-0.1,0.4,0.03",
    )

    assert result.returncode == 0, result.stderr + result.stdout
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert _flag_values(arguments, "--init-mesh-scale", 1) == ["1.5"]
    assert _flag_values(arguments, "--init-mesh-translation", 3) == [
        "-0.1",
        "0.4",
        "0.03",
    ]


@pytest.mark.parametrize(
    "translation",
    [
        "0,0.32",
        "0,0.32,0.02,0",
        "0,,0.02",
        ",0.32,0.02",
        "0,0.32,",
    ],
)
def test_launcher_rejects_malformed_mesh_translation(
    tmp_path: Path, translation: str
) -> None:
    result, _ = _run_launcher(tmp_path, translation=translation)

    assert result.returncode == 2
    assert "INIT_MESH_TRANSLATION must contain exactly 3 non-empty" in result.stderr

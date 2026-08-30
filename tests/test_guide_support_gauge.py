from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import torch

from anigroom.grooming import GuideSupportGaugeTerms, guide_support_gauge
from tools.train_white_tiger_stage1 import (
    CURRENT_CHECKPOINT_VERSION,
    Stage1Config,
    build_arg_parser,
    config_from_args,
)


ROOT = Path(__file__).resolve().parents[1]


def _gauge(
    length_raw: torch.Tensor,
    width_raw: torch.Tensor,
    confidence: torch.Tensor | None = None,
    area: torch.Tensor | None = None,
) -> GuideSupportGaugeTerms:
    if confidence is None:
        confidence = torch.ones_like(length_raw)
    return guide_support_gauge(
        length_raw,
        width_raw,
        confidence,
        source_area_weights=area,
    )


def test_guide_support_gauge_is_zero_at_initialization() -> None:
    result = _gauge(torch.zeros(5), torch.zeros(5))

    torch.testing.assert_close(result.total, torch.zeros(()))
    torch.testing.assert_close(result.length_collapse, torch.zeros(()))
    torch.testing.assert_close(result.slenderness_expansion, torch.zeros(()))


def test_equal_positive_global_length_width_rescale_is_zero() -> None:
    common_log_ratio = torch.full((5,), 0.7)
    raw = torch.sinh(common_log_ratio)

    result = _gauge(raw, raw)

    torch.testing.assert_close(result.total, torch.zeros(()))


def test_trusted_local_length_collapse_is_positive() -> None:
    result = _gauge(
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.zeros(3),
        confidence=torch.ones(3),
    )

    assert float(result.length_collapse) > 0.0
    assert float(result.total) > 0.0


def test_width_inflation_relative_to_length_is_positive() -> None:
    result = _gauge(torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]))

    torch.testing.assert_close(result.length_collapse, torch.zeros(()))
    assert float(result.slenderness_expansion) > 0.0
    assert float(result.total) > 0.0


def test_zero_confidence_keeps_lower_nonzero_floor_influence() -> None:
    length = torch.tensor([-1.0, 0.0, 0.0, 0.0])
    width = torch.zeros(4)
    trusted = _gauge(length, width, confidence=torch.ones(4))
    untrusted = _gauge(
        length,
        width,
        confidence=torch.tensor([0.0, 1.0, 1.0, 1.0]),
    )

    assert 0.0 < float(untrusted.total) < float(trusted.total)


def test_area_weighting_is_permutation_invariant() -> None:
    length = torch.tensor([-1.0, 0.2, -0.4, 0.0])
    width = torch.tensor([0.5, -0.1, 0.8, 0.0])
    confidence = torch.tensor([1.0, 0.4, 0.0, 0.8])
    area = torch.tensor([0.5, 2.0, 1.5, 0.75])
    permutation = torch.tensor([2, 0, 3, 1])

    original = _gauge(length, width, confidence, area)
    unweighted = _gauge(length, width, confidence)
    permuted = _gauge(
        length[permutation],
        width[permutation],
        confidence[permutation],
        area[permutation],
    )

    assert not torch.allclose(original.total, unweighted.total)
    torch.testing.assert_close(permuted.total, original.total)
    torch.testing.assert_close(
        permuted.length_collapse,
        original.length_collapse,
    )
    torch.testing.assert_close(
        permuted.slenderness_expansion,
        original.slenderness_expansion,
    )


def test_gauge_has_finite_corrective_length_and_width_gradients() -> None:
    length = torch.tensor([-1.0, 0.0], requires_grad=True)
    width = torch.tensor([1.0, 0.0], requires_grad=True)

    _gauge(length, width).total.backward()

    assert length.grad is not None
    assert width.grad is not None
    assert torch.isfinite(length.grad).all()
    assert torch.isfinite(width.grad).all()
    assert float(length.grad[0]) < 0.0
    assert float(width.grad[0]) > 0.0


def test_sparse_failure_remains_population_visible() -> None:
    length = torch.cat((torch.tensor([-4.0]), torch.zeros(4095))).requires_grad_()
    width = torch.zeros_like(length)
    gauge = _gauge(length, width)
    gauge.total.backward()
    gauge_gradient = abs(float(length.grad[0]))

    baseline_length = length.detach().clone().requires_grad_()
    baseline = torch.relu(-torch.asinh(baseline_length)).mean()
    baseline.backward()
    mean_gradient = abs(float(baseline_length.grad[0]))

    assert gauge_gradient > 10.0 * mean_gradient
    assert gauge_gradient > 1.0e-3


def _minimal_parser_args(*extra: str) -> list[str]:
    return [
        "--densify-warmup", "0",
        "--densify-interval", "0",
        "--densify-until", "0",
        "--densify-score-threshold", "0",
        "--densify-min-contribution", "0",
        "--max-splits-per-event", "0",
        "--split-children-per-parent", "0",
        "--split-neighbor-count", "0",
        "--split-candidate-rings", "0",
        "--split-candidate-face-count", "0",
        "--split-min-child-distance", "0",
        "--prune-start", "0",
        "--prune-interval", "0",
        "--prune-min-contribution", "0",
        "--prune-min-opacity", "0",
        "--prune-max-fraction", "0",
        *extra,
    ]


def test_zero_weight_keeps_config_path_inert_and_cli_wires_candidate() -> None:
    parser = build_arg_parser()
    default_config = config_from_args(parser.parse_args(_minimal_parser_args()))
    candidate_config = config_from_args(
        parser.parse_args(_minimal_parser_args("--guide-support-gauge-weight", "0.001"))
    )

    assert Stage1Config(data_root="data", mesh_path="mesh", output_dir="out").guide_support_gauge_weight == 0.0
    assert default_config.guide_support_gauge_weight == 0.0
    assert candidate_config.guide_support_gauge_weight == 0.001
    assert CURRENT_CHECKPOINT_VERSION == 13


def test_r069_config_inherits_r068_and_assigns_only_gauge_weight() -> None:
    config_path = ROOT / "configs" / "r069_guide_support_gauge_0_30k.env"
    source = config_path.read_text(encoding="utf-8")

    assert 'source "${CONFIG_DIR}/r068_no_crossing_zero_curl_0_30k.env"' in source
    assignments = [
        line.strip()
        for line in source.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.startswith("CONFIG_DIR=")
        and not line.startswith("source ")
        and not line.startswith("unset ")
    ]
    assert assignments == ["GUIDE_SUPPORT_GAUGE_WEIGHT=0.001"]


def test_generic_launcher_defaults_and_passes_gauge_weight() -> None:
    launcher = (ROOT / "scripts/server/run_white_tiger_stage1.sh").read_text(
        encoding="utf-8"
    )

    assert 'GUIDE_SUPPORT_GAUGE_WEIGHT="${GUIDE_SUPPORT_GAUGE_WEIGHT:-0}"' in launcher
    assert '--guide-support-gauge-weight "$GUIDE_SUPPORT_GAUGE_WEIGHT"' in launcher


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
        raise RuntimeError("bash is required for config shell snapshots")
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


def test_r069_12k_gate_changes_only_iterations_and_keeps_weight() -> None:
    r069_path = ROOT / "configs" / "r069_guide_support_gauge_0_30k.env"
    gate_path = ROOT / "configs" / "r069_guide_support_gauge_0_12k_gate.env"
    gate_source = gate_path.read_text(encoding="utf-8")

    assert 'source "${CONFIG_DIR}/r069_guide_support_gauge_0_30k.env"' in gate_source
    assignments = [
        line.strip()
        for line in gate_source.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and not line.startswith("CONFIG_DIR=")
        and not line.startswith("source ")
        and not line.startswith("unset ")
    ]
    assert assignments == ["ITERATIONS=12000"]

    bash = _bash()
    if bash is None:
        raise RuntimeError("bash is required for config shell snapshots")
    with tempfile.TemporaryDirectory(prefix="r069-12k-config-") as directory:
        root = Path(directory)
        r069_snapshot_path = root / "r069.env"
        gate_snapshot_path = root / "gate.env"
        _snapshot_config(r069_path, r069_snapshot_path)
        _snapshot_config(gate_path, gate_snapshot_path)
        r069 = _load_snapshot(r069_snapshot_path)
        gate = _load_snapshot(gate_snapshot_path)

    delta = {
        key: {"r069_30k": r069.get(key), "r069_12k_gate": gate.get(key)}
        for key in sorted(set(r069) | set(gate))
        if r069.get(key) != gate.get(key)
    }
    assert delta == {
        "ITERATIONS": {"r069_30k": "30000", "r069_12k_gate": "12000"}
    }
    assert r069["GUIDE_SUPPORT_GAUGE_WEIGHT"] == gate["GUIDE_SUPPORT_GAUGE_WEIGHT"] == "0.001"

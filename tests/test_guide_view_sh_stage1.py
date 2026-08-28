from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from anigroom.grooming import GroomRanges
from anigroom.mesh_roots import TriangleMesh
from tools.train_white_tiger_stage1 import (
    CURRENT_CHECKPOINT_VERSION,
    Stage1Config,
    WhiteTigerStage1Model,
    backward_rgb_and_flow_without_color_flow_gradients,
    make_stage1_optimizer,
    stage1_color_parameters,
    stage1_optimizer_param_names,
    validate_guide_view_sh_config,
    zero_color_gradients,
    zero_guide_gradients,
)


ROOT = Path(__file__).resolve().parents[1]


def make_model(*, support: bool = True) -> WhiteTigerStage1Model:
    mesh = TriangleMesh(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    return WhiteTigerStage1Model(
        mesh,
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0, 1, 0], dtype=np.int64),
        np.asarray(
            [[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
            dtype=np.float32,
        ),
        GroomRanges(),
        torch.device("cpu"),
        init_groom_length=0.018,
        max_child_count=1,
        guide_face_ids=np.asarray([0, 1], dtype=np.int64),
        guide_barycentric=np.asarray(
            [[0.65, 0.20, 0.15], [0.15, 0.20, 0.65]],
            dtype=np.float32,
        ),
        guide_view_sh_support=support,
        guide_view_sh_scale=0.2,
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
    )


def make_config(*, support: bool = True) -> Stage1Config:
    return Stage1Config(
        data_root="data",
        mesh_path="mesh.obj",
        output_dir="output",
        child_count=1,
        guide_root_count=2,
        guide_roots_from_clean_flow=True,
        clean_flow_target="target.npz",
        guide_view_sh_support=support,
        guide_view_sh_scale=0.2,
        lr_guide_view_sh=0.02,
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=0.18,
        guide_direction_residual_scale=0.10,
    )


def test_support_off_has_no_parameter_and_roundtrip_is_strict() -> None:
    disabled = make_model(support=False)
    disabled_clone = make_model(support=False)
    disabled_clone.load_state_dict(disabled.state_dict(), strict=True)
    assert disabled.guide_view_sh is None
    assert not any(key.startswith("guide_view_sh.") for key in disabled.state_dict())

    enabled = make_model(support=True)
    enabled_clone = make_model(support=True)
    enabled_clone.load_state_dict(enabled.state_dict(), strict=True)
    torch.testing.assert_close(enabled.guide_view_sh.raw, enabled_clone.guide_view_sh.raw)
    assert CURRENT_CHECKPOINT_VERSION == 11


def test_view_confidence_gates_only_sh_gradient_and_detaches_geometry() -> None:
    model = make_model()
    model.set_guide_view_sh_confidence(
        torch.tensor([9]),
        torch.tensor([[1.0, 0.0]]),
    )
    _, _, roots_local = model.roots_and_normals()
    viewmat = torch.eye(4)
    viewmat[2, 3] = -2.0
    residual = model.guide_view_sh_residual_at_render_roots(
        roots_local,
        viewmat,
        view_index=9,
    )
    assert residual.shape == (3, 3)
    residual.sum().backward()
    assert model.guide_view_sh.raw.grad is not None
    assert float(model.guide_view_sh.raw.grad[0].abs().sum()) > 0.0
    assert float(model.guide_view_sh.raw.grad[1].abs().sum()) == 0.0
    for parameter in (
        model.bary_logits,
        model.guide_direction_local_raw,
        model.log_scale,
        model.translation,
    ):
        assert parameter.grad is None


def test_absent_view_has_forward_but_zero_sh_gradient() -> None:
    model = make_model()
    model.set_guide_view_sh_confidence(
        torch.tensor([9]),
        torch.ones((1, 2)),
    )
    with torch.no_grad():
        model.guide_view_sh.raw.fill_(0.4)
    _, _, roots_local = model.roots_and_normals()
    viewmat = torch.eye(4)
    viewmat[2, 3] = -2.0
    residual = model.guide_view_sh_residual_at_render_roots(
        roots_local,
        viewmat,
        view_index=4,
    )
    assert float(residual.detach().abs().sum()) > 0.0
    residual.sum().backward()
    torch.testing.assert_close(
        model.guide_view_sh.raw.grad,
        torch.zeros_like(model.guide_view_sh.raw.grad),
    )


def test_one_root_residual_is_shared_by_all_of_its_gaussians() -> None:
    model = make_model()
    model.set_guide_view_sh_confidence(torch.tensor([9]), torch.ones((1, 2)))
    with torch.no_grad():
        model.guide_view_sh.raw.fill_(0.3)
    _, _, roots_local = model.roots_and_normals()
    viewmat = torch.eye(4)
    viewmat[2, 3] = -2.0
    root_residual = model.guide_view_sh_residual_at_render_roots(
        roots_local,
        viewmat,
        view_index=9,
    )
    gaussian_root_ids = torch.tensor([0, 0, 0, 1, 2, 2])
    gaussian_residual = root_residual[gaussian_root_ids]
    torch.testing.assert_close(gaussian_residual[0], gaussian_residual[1])
    torch.testing.assert_close(gaussian_residual[1], gaussian_residual[2])
    torch.testing.assert_close(gaussian_residual[4], gaussian_residual[5])


def test_guide_freeze_preserves_sh_gradient_but_color_freeze_zeros_it() -> None:
    model = make_model()
    model.guide_view_sh.raw.sum().backward()
    before = model.guide_view_sh.raw.grad.clone()
    zero_guide_gradients(model)
    torch.testing.assert_close(model.guide_view_sh.raw.grad, before)
    zero_color_gradients(model)
    torch.testing.assert_close(
        model.guide_view_sh.raw.grad,
        torch.zeros_like(model.guide_view_sh.raw.grad),
    )


def test_optimizer_and_rgb_flow_treat_sh_as_appearance() -> None:
    model = make_model()
    config = make_config()
    optimizer = make_stage1_optimizer(model, config)
    names = stage1_optimizer_param_names(model, config)
    assert ["guide_view_sh.raw"] in names
    assert any(
        parameter is model.guide_view_sh.raw
        for parameter in stage1_color_parameters(model)
    )

    geometry = model.guide_direction_local_raw
    sh = model.guide_view_sh.raw
    rgb_loss = 2.0 * geometry.sum() + 3.0 * sh.sum()
    flow_loss = 5.0 * geometry.sum() + 7.0 * sh.sum()
    optimizer.zero_grad(set_to_none=True)
    backward_rgb_and_flow_without_color_flow_gradients(
        model,
        optimizer,
        rgb_and_regularization_loss=rgb_loss,
        flow_loss=flow_loss,
    )
    torch.testing.assert_close(geometry.grad, torch.full_like(geometry, 7.0))
    torch.testing.assert_close(sh.grad, torch.full_like(sh, 3.0))


def test_config_validation_is_explicit() -> None:
    validate_guide_view_sh_config(make_config())
    with pytest.raises(ValueError, match="primary guides"):
        validate_guide_view_sh_config(replace(make_config(), guide_root_count=0))
    with pytest.raises(ValueError, match="scale"):
        validate_guide_view_sh_config(replace(make_config(), guide_view_sh_scale=0.0))
    with pytest.raises(ValueError, match="learning rate"):
        validate_guide_view_sh_config(replace(make_config(), lr_guide_view_sh=0.0))
    with pytest.raises(ValueError, match="lifecycle"):
        validate_guide_view_sh_config(replace(make_config(), guide_densify_interval=100))


def test_r071_config_is_an_exact_r068_0_9k_sh_gate() -> None:
    bash = shutil.which("bash") or (
        r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else None
    )
    if bash is None:
        pytest.skip("bash is required for config snapshot validation")

    def shell_path(path: Path) -> str:
        value = path.resolve().as_posix()
        if os.name == "nt" and len(value) > 1 and value[1] == ":":
            return f"/{value[0].lower()}{value[2:]}"
        return value

    def load(path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                if key not in {"PWD", "SHLVL", "_"}:
                    values[key] = value
        return values

    with tempfile.TemporaryDirectory(prefix="r071-config-") as directory:
        temp = Path(directory)
        parent_output = temp / "parent.env"
        candidate_output = temp / "candidate.env"
        parent = ROOT / "configs/r068_no_crossing_zero_curl_0_30k.env"
        candidate = ROOT / "configs/r071_guide_view_sh_0_9k_gate.env"
        script = "\n".join(
            [
                "set -euo pipefail",
                "set -a",
                "export MESH_NO_PENETRATION_SDF=/tmp/white_tiger_sdf.npz",
                f"source {shlex.quote(shell_path(parent))}",
                f"env > {shlex.quote(shell_path(parent_output))}",
                f"source {shlex.quote(shell_path(candidate))}",
                f"env > {shlex.quote(shell_path(candidate_output))}",
            ]
        )
        result = subprocess.run(
            [bash, "-c", script],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        parent_values = load(parent_output)
        candidate_values = load(candidate_output)

    delta = {
        key: (parent_values.get(key), candidate_values.get(key))
        for key in sorted(set(parent_values) | set(candidate_values))
        if parent_values.get(key) != candidate_values.get(key)
    }
    assert delta == {
        "GUIDE_SUPPORT_GAUGE_WEIGHT": (None, "0"),
        "GUIDE_VIEW_SH_SCALE": (None, "0.20"),
        "GUIDE_VIEW_SH_SUPPORT": (None, "1"),
        "ITERATIONS": ("30000", "9000"),
        "LR_GUIDE_VIEW_SH": (None, "0.020"),
        "STAGE_SAVE_ITERS": (
            "9000,10000,12000,14000,16000,18000,20000,22000,25000,27000,30000",
            "3000,6000,9000",
        ),
    }

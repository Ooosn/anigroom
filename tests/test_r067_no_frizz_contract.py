from __future__ import annotations

from dataclasses import asdict
import inspect
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from anigroom.grooming import GroomParameterField, GroomRanges, RenderGeometryResidualField
from anigroom.mesh_roots import TriangleMesh
from tools.train_white_tiger_stage1 import (
    Stage1Config,
    WhiteTigerStage1Model,
    load_stage1_checkpoint_model,
    make_stage1_optimizer,
    require_current_checkpoint_version,
    stage1_config_from_checkpoint_mapping,
    stage1_optimizer_param_names,
)
from anigroom.grooming.strand_deformations import deform_backbone, frizz_backbone
from anigroom.grooming.strand_gaussians import build_strands


ROOT = Path(__file__).resolve().parents[1]


def tiny_model() -> WhiteTigerStage1Model:
    mesh = TriangleMesh(
        vertices=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    return WhiteTigerStage1Model(
        mesh,
        np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]], dtype=np.float32),
        GroomRanges(),
        torch.device("cpu"),
        max_child_count=1,
        gaussian_rgb_residual_support=False,
        guide_face_ids=np.asarray([0, 1], dtype=np.int64),
        guide_barycentric=np.asarray(
            [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]], dtype=np.float32
        ),
        render_geometry_parameterization="zero_centered_residual",
        guide_length_residual_scale=1.0,
        guide_direction_residual_scale=0.1,
        guide_curl_residual_scale=1.0,
    )


def assert_no_frizz_keys(value) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert "frizz" not in str(key).lower()
            assert_no_frizz_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_no_frizz_keys(item)


def test_r067_model_residual_and_optimizer_state_have_no_frizz_keys() -> None:
    field = GroomParameterField(3)
    residual = RenderGeometryResidualField(3)
    model = tiny_model()
    config = Stage1Config(data_root="data", mesh_path="mesh", output_dir="out")
    optimizer = make_stage1_optimizer(model, config)
    names = stage1_optimizer_param_names(model, config)
    assert_no_frizz_keys(field.state_dict())
    assert_no_frizz_keys(dict(field.named_parameters()))
    assert_no_frizz_keys(field.decode().__dict__)
    assert_no_frizz_keys(residual.state_dict())
    assert_no_frizz_keys(model.state_dict())
    assert_no_frizz_keys(asdict(config))
    assert_no_frizz_keys(names)
    assert all(
        parameter is not None
        for group in optimizer.param_groups
        for parameter in group["params"]
    )


def test_build_strands_does_not_call_standalone_procedural_utility(monkeypatch) -> None:
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
    tangents = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    roots = torch.zeros((1, 3), dtype=torch.float64)
    field = GroomParameterField(1).decode()
    monkeypatch.setattr(
        "anigroom.grooming.strand_deformations.frizz_backbone",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("standalone utility called")
        ),
    )
    points, *_ = build_strands(
        roots,
        normals,
        tangents,
        torch.cross(normals, tangents, dim=-1),
        field,
        samples=9,
    )
    assert points.shape == (1, 9, 3)


def test_standalone_deformation_signature_is_curl_only() -> None:
    assert "frizz" not in inspect.signature(deform_backbone).parameters
    assert {"amplitude", "seed_phase"}.issubset(
        inspect.signature(frizz_backbone).parameters
    )


def test_schema8_checkpoint_is_rejected_before_model_load_under_schema14(tmp_path) -> None:
    checkpoint = tmp_path / "r066_schema8.pt"
    torch.save({"checkpoint_version": 8, "model": {}, "config": {}, "optimizer": {}}, checkpoint)
    with pytest.raises(RuntimeError, match="expected 14"):
        load_stage1_checkpoint_model(checkpoint, torch.device("cpu"))


def test_historical_schema9_checkpoint_is_rejected_before_model_load(tmp_path) -> None:
    checkpoint = tmp_path / "r067_schema9.pt"
    torch.save({"checkpoint_version": 9, "model": {}, "config": {}, "optimizer": {}}, checkpoint)
    with pytest.raises(RuntimeError, match="expected 14"):
        load_stage1_checkpoint_model(checkpoint, torch.device("cpu"))


def test_schema10_checkpoint_is_rejected_before_model_load(tmp_path) -> None:
    checkpoint = tmp_path / "r069_schema10.pt"
    torch.save({"checkpoint_version": 10, "model": {}, "config": {}, "optimizer": {}}, checkpoint)
    with pytest.raises(RuntimeError, match="expected 14"):
        load_stage1_checkpoint_model(checkpoint, torch.device("cpu"))


def test_schema11_checkpoint_is_rejected_before_model_load(tmp_path) -> None:
    checkpoint = tmp_path / "r072_schema11.pt"
    torch.save({"checkpoint_version": 11, "model": {}, "config": {}, "optimizer": {}}, checkpoint)
    with pytest.raises(RuntimeError, match="expected 14"):
        load_stage1_checkpoint_model(checkpoint, torch.device("cpu"))


def test_schema13_checkpoint_version_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="expected 14"):
        require_current_checkpoint_version({"checkpoint_version": 13})


def test_schema14_checkpoint_version_is_current() -> None:
    require_current_checkpoint_version({"checkpoint_version": 14})


def test_removed_frizz_config_mapping_is_rejected() -> None:
    data = asdict(Stage1Config(data_root="data", mesh_path="mesh", output_dir="out"))
    data["guide_frizz_residual_scale"] = 1.0
    data["shape_frizz_scale"] = 1.0
    with pytest.raises(TypeError, match="unsupported"):
        stage1_config_from_checkpoint_mapping(data)


def test_r067_config_delta_is_exactly_two_removed_keys() -> None:
    bash = shutil.which("bash") or (r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else None)
    if bash is None:
        pytest.skip("bash is required for shell config snapshot validation")
    parent = ROOT / "configs/r066_learned_curl_turns_0_30k.env"
    candidate = ROOT / "configs/r067_no_frizz_0_30k.env"
    def shell_path(path: Path) -> str:
        value = path.as_posix()
        if os.name == "nt" and len(value) > 1 and value[1] == ":":
            return f"/{value[0].lower()}{value[2:]}"
        return value
    script = "\n".join([
        "set -a",
        f"MESH_NO_PENETRATION_SDF=/tmp/white_tiger_sdf.npz; source '{shell_path(parent)}'; env > /tmp/r067_parent.env",
        f"source '{shell_path(candidate)}'; env > /tmp/r067_candidate.env",
        "python - /tmp/r067_parent.env /tmp/r067_candidate.env <<'PY'",
        "import sys",
        "def load(path):",
        "  out = {}",
        "  for line in open(path):",
        "    if '=' in line:",
        "      key, value = line.rstrip().split('=', 1)",
        "      if key not in {'PWD', 'SHLVL', '_'}: out[key] = value",
        "  return out",
        "base, candidate = load(sys.argv[1]), load(sys.argv[2])",
        "delta = {key: {'r066': base.get(key), 'r067': candidate.get(key)} for key in sorted(set(base) | set(candidate)) if base.get(key) != candidate.get(key)}",
        "expected = {'GUIDE_FRIZZ_RESIDUAL_SCALE': {'r066': '1.0', 'r067': None}, 'SHAPE_FRIZZ_SCALE': {'r066': '1.0', 'r067': None}}",
        "assert delta == expected, delta",
        "print('R067_CONFIG_DELTA_PASS')",
        "PY",
    ])
    result = subprocess.run([bash, "-c", script], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "R067_CONFIG_DELTA_PASS" in result.stdout


def test_crossing_ownership_is_direction_and_curl_radius_only() -> None:
    from tools.train_white_tiger_stage1 import _STRAND_CROSSING_LOCAL_RESIDUAL_PARAMETER_NAMES
    assert _STRAND_CROSSING_LOCAL_RESIDUAL_PARAMETER_NAMES == (
        "direction_local_raw",
        "curl_radius_ratio_raw",
    )


def test_active_r067_postprocess_tools_have_no_frizz_assumptions() -> None:
    paths = [
        ROOT / "tools/diagnose_curl_components.py",
        ROOT / "tools/diagnose_strand_foldback_components.py",
        ROOT / "tools/export_white_tiger_checkpoint_strands.py",
        ROOT / "tools/render_white_tiger_stage1_checkpoint_views.py",
        ROOT / "tools/visualize_white_tiger_groom_attributes.py",
        ROOT / "tools/diagnose_white_tiger_streak_roots.py",
    ]
    for path in paths:
        assert "frizz" not in path.read_text(encoding="utf-8").lower(), path

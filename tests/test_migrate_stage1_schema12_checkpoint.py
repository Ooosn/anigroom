from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from tools.migrate_stage1_schema12_checkpoint import (
    CURRENT_CHECKPOINT_VERSION,
    MIGRATED_CONFIG_DEFAULTS,
    SOURCE_CHECKPOINT_VERSION,
    checkpoint_tensor_integrity,
    current_stage1_config_fields,
    migrate_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_TOOL = ROOT / "tools" / "migrate_stage1_schema12_checkpoint.py"

STRING_FIELDS = {
    "data_root",
    "mesh_path",
    "output_dir",
    "face_tangent_field",
    "stage_save_iters",
    "train_views",
    "test_views",
    "clean_flow_target",
    "geometry_residual_domain",
    "render_geometry_parameterization",
    "view_gate_normalization",
    "render_length_prior_coordinate",
    "render_length_prior_reduction",
    "guide_length_smooth_mode",
    "guide_densify_policy",
    "mesh_no_penetration_sdf",
    "densify_residual_mode",
    "lifecycle_score_mode",
    "resume_checkpoint",
}
BOOLEAN_FIELDS = {
    "clean_flow_init",
    "clean_flow_length_init",
    "guide_roots_from_clean_flow",
    "guide_view_sh_support",
    "view_gated_ownership_support",
    "gaussian_rgb_residual_support",
    "rgb_flow_exclude_color_gradients",
    "compute_lpips",
    "white_background",
    "random_backing_color",
    "random_mesh_backing_texture",
    "mesh_no_penetration_support",
    "strand_crossing_support",
    "mesh_depth_clipping",
    "mesh_backing_compositing",
    "local_child_color_support",
    "resume_optimizer",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema12_config() -> dict[str, object]:
    fields = current_stage1_config_fields() - set(MIGRATED_CONFIG_DEFAULTS)
    config: dict[str, object] = {}
    for name in sorted(fields):
        if name in {"data_root", "mesh_path", "output_dir"}:
            config[name] = f"synthetic/{name}"
        elif name in STRING_FIELDS:
            config[name] = ""
        elif name in BOOLEAN_FIELDS:
            config[name] = False
        elif name == "init_mesh_translation":
            config[name] = [0.0, 0.32, 0.02]
        elif name == "iterations":
            config[name] = 30_000
        elif name == "seed":
            config[name] = 13
        elif name == "init_mesh_scale":
            config[name] = 1.28
        elif name == "init_groom_length":
            config[name] = 0.060
        elif name.endswith(("_weight", "_scale", "_ratio", "_power", "_overlap")):
            config[name] = 0.25
        else:
            config[name] = 1
    config["resume_checkpoint"] = "legacy_resume.pt"
    config["resume_optimizer"] = True
    return config


def _schema12_checkpoint() -> dict[str, object]:
    return {
        "checkpoint_version": SOURCE_CHECKPOINT_VERSION,
        "checkpoint_kind": "stage1_full",
        "iteration": 3000,
        "config": _schema12_config(),
        "model": {
            "bary_logits": torch.tensor([[1.0, -2.0]], dtype=torch.float32),
            "nested": [
                torch.tensor([3, 4, 5], dtype=torch.int64),
                {"leaf": torch.tensor([[0.25]], dtype=torch.float64)},
            ],
        },
        "optimizer": {
            "state": {
                0: {
                    "step": torch.tensor(3000.0, dtype=torch.float32),
                    "exp_avg": torch.tensor([0.0, 1.0, -2.0], dtype=torch.float32),
                    "exp_avg_sq": torch.tensor([0.0, 0.5, 4.0], dtype=torch.float32),
                    "nested": {"moment_aux": torch.tensor([7], dtype=torch.int16)},
                }
            },
            "param_groups": [
                {
                    "params": [0],
                    "lr": 0.014,
                    "betas": (0.9, 0.999),
                    "eps": 1.0e-8,
                }
            ],
        },
        "optimizer_param_names": [["guide_length_raw"]],
        "rng_state": {
            "torch_cpu": torch.tensor([11, 12, 13], dtype=torch.uint8),
            "nested": [torch.tensor([14.0], dtype=torch.float32)],
        },
        "lifecycle_history": [
            {"iteration": 100, "root_count_after": 12},
            {"iteration": 200, "root_count_after": 15},
        ],
        "save_reason": "regular",
        "metadata": {"owner": "synthetic-r074", "unchanged": 17},
    }


def _write_source(path: Path, checkpoint: dict[str, object]) -> str:
    torch.save(checkpoint, path)
    return _sha256(path)


def test_schema12_migration_preserves_payload_and_reports_exact_digests(tmp_path) -> None:
    source_path = tmp_path / "source_schema12.pt"
    output_path = tmp_path / "migrated_schema14.pt"
    report_path = tmp_path / "migration_report.json"
    source = _schema12_checkpoint()
    source_sha256 = _write_source(source_path, source)

    report = migrate_checkpoint(
        source_path,
        output_path,
        report_path,
        source_sha256,
    )

    assert output_path.is_file()
    assert report_path.is_file()
    migrated = torch.load(output_path, map_location="cpu", weights_only=False)
    assert migrated["checkpoint_version"] == CURRENT_CHECKPOINT_VERSION
    assert migrated["checkpoint_kind"] == source["checkpoint_kind"]
    assert migrated["iteration"] == source["iteration"]
    assert set(migrated["config"]) == current_stage1_config_fields()
    for key, value in source["config"].items():
        assert migrated["config"][key] == value
    assert {
        key: migrated["config"][key] for key in MIGRATED_CONFIG_DEFAULTS
    } == MIGRATED_CONFIG_DEFAULTS

    assert checkpoint_tensor_integrity(source) == checkpoint_tensor_integrity(migrated)
    assert torch.equal(
        migrated["rng_state"]["torch_cpu"], source["rng_state"]["torch_cpu"]
    )
    assert migrated["lifecycle_history"] == source["lifecycle_history"]
    assert migrated["save_reason"] == source["save_reason"]
    assert migrated["metadata"] == source["metadata"]

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == saved_report
    assert report["source_sha256"] == source_sha256
    assert report["source"]["sha256"] == source_sha256
    assert report["output_sha256"] == _sha256(output_path)
    assert report["output"]["sha256"] == _sha256(output_path)
    assert report["config_delta"] == {
        "removed": [],
        "added": MIGRATED_CONFIG_DEFAULTS,
        "changed": {},
    }
    assert report["tensor_integrity_checks"] == {
        "source_to_migrated_identical": True,
        "migrated_to_output_identical": True,
        "individual_and_aggregate_hashes_identical": True,
        "object_key_shape_manifests_identical": True,
    }
    guide_report = report["guide_length_raw_optimizer"]
    assert guide_report["group_index"] == 0
    assert guide_report["state_id"] == 0
    assert guide_report["source"] == guide_report["output"]
    assert guide_report["source"]["step"]["value"] == 3000.0
    assert guide_report["source"]["step_nonzero_count"] == 1
    assert guide_report["source"]["moment_nonzero_counts"] == {
        "exp_avg": 2,
        "exp_avg_sq": 2,
    }
    assert set(
        record["path"]
        for record in report["tensor_manifests"]["source"]["optimizer"][
            "tensor_manifest"
        ]
    ) == {
        'optimizer["state"][0]["exp_avg"]',
        'optimizer["state"][0]["exp_avg_sq"]',
        'optimizer["state"][0]["nested"]["moment_aux"]',
        'optimizer["state"][0]["step"]',
    }
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_schema12_config_key_set_is_strict(tmp_path, mutation: str) -> None:
    source = _schema12_checkpoint()
    if mutation == "extra":
        source["config"]["unexpected_future_field"] = 1
    else:
        del source["config"]["guide_support_gauge_weight"]
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "output.pt"
    report_path = tmp_path / "report.json"
    source_sha256 = _write_source(source_path, source)

    with pytest.raises(RuntimeError, match="config key set mismatch"):
        migrate_checkpoint(source_path, output_path, report_path, source_sha256)

    assert not output_path.exists()
    assert not report_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_schema12_migration_rejects_wrong_input_hash(tmp_path) -> None:
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "output.pt"
    report_path = tmp_path / "report.json"
    _write_source(source_path, _schema12_checkpoint())

    with pytest.raises(RuntimeError, match="input SHA256 mismatch"):
        migrate_checkpoint(source_path, output_path, report_path, "0" * 64)

    assert not output_path.exists()
    assert not report_path.exists()


@pytest.mark.parametrize("existing", ["output", "report"])
def test_schema12_migration_refuses_existing_outputs(tmp_path, existing: str) -> None:
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "output.pt"
    report_path = tmp_path / "report.json"
    source_sha256 = _write_source(source_path, _schema12_checkpoint())
    (output_path if existing == "output" else report_path).write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="already exists"):
        migrate_checkpoint(source_path, output_path, report_path, source_sha256)

    assert (output_path if existing == "output" else report_path).read_bytes() == b"keep me"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_version", 11),
        ("checkpoint_kind", "not_stage1"),
        ("iteration", 2999),
    ],
)
def test_schema12_migration_requires_r074_checkpoint_identity(
    tmp_path,
    field: str,
    value: object,
) -> None:
    source = _schema12_checkpoint()
    source[field] = value
    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "output.pt"
    report_path = tmp_path / "report.json"
    source_sha256 = _write_source(source_path, source)

    expected_error = field if field != "checkpoint_kind" else "checkpoint_kind"
    with pytest.raises(RuntimeError, match=expected_error):
        migrate_checkpoint(source_path, output_path, report_path, source_sha256)


def test_cli_requires_all_paths_and_expected_hash() -> None:
    result = subprocess.run(
        [sys.executable, str(MIGRATION_TOOL)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--input" in result.stderr
    assert "--output" in result.stderr
    assert "--report" in result.stderr
    assert "--expected-input-sha256" in result.stderr


def test_schema14_loader_accepts_migrated_metadata_without_model_reconstruction(
    tmp_path,
    monkeypatch,
) -> None:
    try:
        gsplat_available = "gsplat" in sys.modules or (
            importlib.util.find_spec("gsplat") is not None
        )
    except ValueError:
        gsplat_available = "gsplat" in sys.modules
    if not gsplat_available:
        gsplat = types.ModuleType("gsplat")
        rendering = types.ModuleType("gsplat.rendering")
        rendering.rasterization = object()
        gsplat.rendering = rendering
        monkeypatch.setitem(sys.modules, "gsplat", gsplat)
        monkeypatch.setitem(sys.modules, "gsplat.rendering", rendering)
    from tools.train_white_tiger_stage1 import (  # noqa: PLC0415
        load_training_checkpoint,
        stage1_config_from_checkpoint_mapping,
    )

    source_path = tmp_path / "source.pt"
    output_path = tmp_path / "output.pt"
    report_path = tmp_path / "report.json"
    source_sha256 = _write_source(source_path, _schema12_checkpoint())
    migrate_checkpoint(source_path, output_path, report_path, source_sha256)

    loaded = load_training_checkpoint(output_path)
    config = stage1_config_from_checkpoint_mapping(loaded["config"])
    assert config.guide_length_freeze_until == -1
    assert config.clean_flow_guide_length_anchor_weight == 0.0
    assert config.clean_flow_guide_length_anchor_reduction == "mean_l1"
    assert config.view_gate_geometry_support is False
    assert config.view_gate_length_confidence_support is False

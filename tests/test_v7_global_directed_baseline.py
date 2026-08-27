from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET_NAME = "guide_flow3d_shell_targets_exclude_004_024_025.npz"
V6_TARGET = ROOT / "baseline_inputs" / "v6_surface_direction" / TARGET_NAME
V7_TARGET = ROOT / "baseline_inputs" / "v7_surface_direction" / TARGET_NAME
V7_SHA256 = "f009af820560adf19b6eedbb8bf2c5d29df00cca576be13161b4ee2ebaed6510"


def test_v7_global_directed_baseline_contract() -> None:
    assert V7_TARGET.is_file()
    assert hashlib.sha256(V7_TARGET.read_bytes()).hexdigest() == V7_SHA256

    v6 = np.load(V6_TARGET, allow_pickle=False)
    v7 = np.load(V7_TARGET, allow_pickle=False)
    required = {
        "cleaned_directed_flow3d",
        "axis_view_cluster_global_final_sign",
        "axis_view_cluster_global_flip",
        "axis_view_cluster_global_canonical_rank",
        "axis_view_cluster_global_edge_new_severe_mask",
        "axis_view_cluster_postratio_final_ratio",
        "axis_view_cluster_postratio_accept_mask",
        "axis_view_cluster_postratio_final_edge_dot",
    }
    assert not (required - set(v7.files))

    for key in (
        "root_points",
        "root_normals",
        "face_ids",
        "barycentric",
        "shell_h",
        "raw_shell_h",
        "local_spacing",
        "flow3d",
        "observed",
        "weight",
        "view_count",
    ):
        assert np.array_equal(v7[key], v6[key]), key

    assert int(v7["observed"].sum()) == 4407
    assert int(v7["axis_view_cluster_global_flip"].sum()) == 62
    assert int(v7["axis_view_cluster_postratio_accept_mask"].sum()) == 490
    assert not bool(v7["axis_view_cluster_global_edge_new_severe_mask"].any())

    sign = v7["axis_view_cluster_global_final_sign"]
    assert np.all((sign == -1) | (sign == 1))
    canonical_rank = v7["axis_view_cluster_global_canonical_rank"]
    assert np.array_equal(np.sort(canonical_rank), np.arange(canonical_rank.size))
    assert np.all(v7["axis_view_cluster_postratio_final_ratio"] >= 0.0)

    direction = v7["cleaned_directed_flow3d"]
    assert np.isfinite(direction).all()
    np.testing.assert_allclose(np.linalg.norm(direction, axis=-1), 1.0, atol=2.0e-6)
    assert np.all(np.sum(direction * v7["root_normals"], axis=-1) >= -2.0e-6)

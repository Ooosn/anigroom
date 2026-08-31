from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from anigroom.surface_interpolation import (
    SurfaceFieldInterpolator,
    SurfaceSupport,
    adaptive_wendland_c2_weights,
    interpolate_physical,
)
from tools.diagnose_adaptive_continuous_length_field import (
    _edge_support_overlap,
    aggregate_edge_statistics,
    concise_report,
    query_gradient_probe,
    parse_args,
    resolve_candidate_active_neighbor_count,
    select_mesh_path,
    summarize,
    validate_interpolation_invariants,
    validate_surface_support,
    validate_support_ids,
    write_deterministic_json,
)


def test_candidate_k_default_override_and_cli_validation(monkeypatch) -> None:
    assert resolve_candidate_active_neighbor_count(8, None) == 8
    assert resolve_candidate_active_neighbor_count(8, 8) == 8
    assert resolve_candidate_active_neighbor_count(8, 32) == 32
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            resolve_candidate_active_neighbor_count(8, invalid)

    monkeypatch.setattr(
        sys,
        "argv",
        ["diagnose", "--checkpoint", "checkpoint.pt", "--output", "report.json"],
    )
    default_args = parse_args()
    assert default_args.candidate_active_neighbor_count is None

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "report.json",
            "--candidate-active-neighbor-count",
            "32",
        ],
    )
    override_args = parse_args()
    assert override_args.candidate_active_neighbor_count == 32

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "report.json",
            "--candidate-active-neighbor-count",
            "0",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_concise_report_exposes_baseline_and_candidate_k() -> None:
    concise = concise_report(
        {
            "status": "complete",
            "schema": "anigroom.r082_fixed_neighbor_mass_length_field.v1",
            "output": "report.json",
            "checkpoint_sha256": "0" * 64,
            "checkpoint_iteration": 4000,
            "source_commit": "deadbeef",
            "counts": {
                "render_root_count": 10,
                "guide_site_count": 8,
                "baseline_active_neighbor_count": 8,
                "baseline_support_width": 8,
                "candidate_active_neighbor_count": 32,
                "candidate_support_width": 33,
            },
            "edges": {"edge_count": 20, "candidate": {"q95": 0.1}},
            "field_difference": {"absolute": {"q95": 0.02}},
        }
    )
    assert concise["baseline_active_neighbor_count"] == 8
    assert concise["candidate_active_neighbor_count"] == 32
    assert concise["baseline_support_width"] == 8
    assert concise["candidate_support_width"] == 33


def test_summary_and_partition_helpers_use_exact_small_tensors() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    result = summarize(values)

    assert result["count"] == 4
    assert result["mean"] == 2.5
    assert result["q50"] == 2.5
    assert result["q95"] == pytest.approx(3.85)
    assert result["q999"] == pytest.approx(3.997)

    support = torch.tensor(
        [[0, 1, 2], [0, 1, 2], [1, 2, 3], [0, 2, 3]],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [
            [0.60, 0.30, 0.10],
            [0.50, 0.25, 0.25],
            [0.20, 0.30, 0.50],
            [0.40, 0.20, 0.40],
        ],
        dtype=torch.float64,
    )
    source_values = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    field = interpolate_physical(source_values, support, weights)
    validation = validate_interpolation_invariants(
        source_values,
        support,
        weights,
        interpolated=field,
        expected_width=3,
    )

    assert validation["finite_values"]
    assert validation["row_sums_allclose"]
    assert validation["unique_support_ids"]
    assert validation["constant_field_reproduction"]["ok"]
    assert validation["convex_hull"]["ok"]
    assert validation["positivity"]["ok"]


def test_mixed_infinite_vertex_paths_are_accepted_and_distances_are_finite() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    interpolator = SurfaceFieldInterpolator(
        vertices=vertices,
        faces=faces,
        source_points=torch.tensor(
            [[0.10, 0.10, 0.0], [0.70, 0.10, 0.0]],
            dtype=torch.float32,
        ),
        source_face_ids=torch.zeros((2,), dtype=torch.long),
        neighbor_count=2,
        device="cpu",
    )
    support = SurfaceSupport(
        indices=torch.tensor([[0, 1]], dtype=torch.long),
        vertex_path_distances=torch.tensor(
            [[[0.0, float("inf"), 0.25], [float("inf"), 0.10, 0.30]]],
            dtype=torch.float32,
        ),
        report={"fallback_query_count": 0},
    )

    report = validate_surface_support(
        support,
        source_count=2,
        expected_width=2,
        name="mixed support",
    )
    distances = interpolator.distances(
        torch.tensor([[0.20, 0.20, 0.0]], dtype=torch.float32),
        torch.zeros((1,), dtype=torch.long),
        support,
    )

    assert bool(torch.isfinite(distances).all())
    assert report["path_entry_count"] == 6
    assert report["finite_path_entry_count"] == 4
    assert report["finite_path_entry_fraction"] == pytest.approx(4.0 / 6.0)
    assert report["support_slot_count"] == 2
    assert report["fully_covered_support_slot_count"] == 2
    assert report["fully_covered_support_slot_fraction"] == 1.0


def test_all_three_infinite_paths_are_rejected_as_coverage_hole() -> None:
    support = SurfaceSupport(
        indices=torch.tensor([[0, 1]], dtype=torch.long),
        vertex_path_distances=torch.tensor(
            [[[0.0, 0.0, 0.0], [float("inf"), float("inf"), float("inf")]]],
            dtype=torch.float32,
        ),
        report={"fallback_query_count": 0},
    )

    with pytest.raises(RuntimeError, match="coverage hole"):
        validate_surface_support(
            support,
            source_count=2,
            expected_width=2,
            name="hole support",
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("-inf"), -0.25])
def test_invalid_path_entries_are_rejected(bad_value: float) -> None:
    support = SurfaceSupport(
        indices=torch.tensor([[0, 1]], dtype=torch.long),
        vertex_path_distances=torch.tensor(
            [[[0.0, 0.0, 0.0], [bad_value, 0.10, 0.20]]],
            dtype=torch.float32,
        ),
        report={"fallback_query_count": 0},
    )

    with pytest.raises(RuntimeError, match="(NaN|-inf|negative finite)"):
        validate_surface_support(
            support,
            source_count=2,
            expected_width=2,
            name="invalid support",
        )


def test_deterministic_json_and_overwrite_refusal(tmp_path) -> None:
    payload = {"z": [3, 2, 1], "a": {"nested": True}, "value": 1.25}
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_deterministic_json(first, payload)
    write_deterministic_json(second, payload)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == payload

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_deterministic_json(first, {"changed": True})
    write_deterministic_json(first, {"changed": True}, overwrite=True)
    assert json.loads(first.read_text(encoding="utf-8")) == {"changed": True}


def _direct_edge_reference(
    edges: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src, dst = edges[:, 0], edges[:, 1]
    baseline_jump = (torch.log(baseline[src]) - torch.log(baseline[dst])).abs()
    candidate_jump = (torch.log(candidate[src]) - torch.log(candidate[dst])).abs()
    overlap = (
        (support[src, :, None] == support[dst, None, :]).any(dim=2).sum(dim=1)
    )
    equal = overlap == support.shape[1]
    return baseline_jump, candidate_jump, equal, overlap


def _pairwise_overlap_reference(
    source_support: torch.Tensor,
    destination_support: torch.Tensor,
) -> torch.Tensor:
    return (
        source_support[:, :, None] == destination_support[:, None, :]
    ).any(dim=2).sum(dim=1)


def _random_unique_support(
    batch_size: int,
    width: int,
    source_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.randperm(source_count, generator=generator)[:width]
            for _ in range(batch_size)
        ],
        dim=0,
    )


def test_optimized_overlap_matches_pairwise_reference_for_wide_supports() -> None:
    generator = torch.Generator().manual_seed(20260901)
    for width in (3, 9, 33, 65):
        source_count = width + 17
        for batch_size in (1, 7, 64):
            source_support = _random_unique_support(
                batch_size,
                width,
                source_count,
                generator,
            )
            destination_support = _random_unique_support(
                batch_size,
                width,
                source_count,
                generator,
            )
            expected = _pairwise_overlap_reference(
                source_support,
                destination_support,
            )
            actual = _edge_support_overlap(source_support, destination_support)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_edge_chunk_aggregation_matches_direct_reference_and_partitions() -> None:
    baseline = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    candidate = torch.tensor([1.0, 2.5, 3.0, 7.0], dtype=torch.float64)
    support = torch.tensor(
        [[0, 1, 2], [0, 1, 2], [1, 2, 3], [0, 2, 3]],
        dtype=torch.long,
    )
    edges = torch.tensor(
        [[0, 1], [0, 2], [1, 3], [2, 3], [3, 0]],
        dtype=torch.long,
    )

    reference_baseline, reference_candidate, reference_equal, reference_overlap = (
        _direct_edge_reference(edges, baseline, candidate, support)
    )
    result = aggregate_edge_statistics(
        edges,
        baseline,
        candidate,
        support,
        edge_chunk_size=2,
    )
    repeat = aggregate_edge_statistics(
        edges,
        baseline,
        candidate,
        support,
        edge_chunk_size=99,
    )

    for key in (
        "baseline",
        "candidate",
        "full_edge_partition",
        "support_overlap",
        "exact_full_edge_partition",
    ):
        assert result[key] == repeat[key]
    assert result["edge_chunk_size"] == 2
    assert result["edge_chunk_count"] == 3
    assert repeat["edge_chunk_size"] == 99
    assert repeat["edge_chunk_count"] == 1
    np.testing.assert_allclose(
        result["baseline"]["mean"],
        float(reference_baseline.mean()),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        result["candidate"]["mean"],
        float(reference_candidate.mean()),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert result["full_edge_partition"]["unchanged_support"]["edge_count"] == int(
        reference_equal.sum()
    )
    assert result["full_edge_partition"]["changed_support"]["edge_count"] == int(
        (~reference_equal).sum()
    )
    for overlap_count in range(4):
        key = str(overlap_count)
        assert result["support_overlap"]["counts"][key] == int(
            (reference_overlap == overlap_count).sum()
        )
    assert result["exact_full_edge_partition"]
    assert sum(result["support_overlap"]["counts"].values()) == int(edges.shape[0])


def test_candidate_self_evaluation_and_invariants_on_topology_safe_fixture() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    guide_points = torch.tensor(
        [[0.15, 0.15, 0.0], [0.60, 0.20, 0.0], [0.20, 0.60, 0.0]],
        dtype=torch.float32,
    )
    guide_faces = torch.zeros((3,), dtype=torch.long)
    interpolator = SurfaceFieldInterpolator(
        vertices=vertices,
        faces=faces,
        source_points=guide_points,
        source_face_ids=guide_faces,
        neighbor_count=3,
        device="cpu",
    )
    support = interpolator.build_support(guide_points, guide_faces)
    validate_support_ids(support.indices, source_count=3, expected_width=3)
    distances = interpolator.distances(guide_points, guide_faces, support)
    weights = adaptive_wendland_c2_weights(
        distances,
        active_neighbor_count=2,
        support_indices=support.indices,
    )
    stored_length = torch.tensor([0.8, 1.4, 2.2], dtype=torch.float32)
    evaluated_length = interpolate_physical(stored_length, support.indices, weights)
    validation = validate_interpolation_invariants(
        stored_length,
        support.indices,
        weights,
        interpolated=evaluated_length,
        expected_width=3,
    )
    relative_error = (evaluated_length - stored_length).abs() / stored_length
    gradient_probe = query_gradient_probe(
        interpolator,
        guide_points,
        guide_faces,
        support,
        stored_length,
        active_neighbor_count=2,
        max_query_count=1024,
    )

    assert bool(torch.isfinite(evaluated_length).all())
    assert bool(torch.isfinite(relative_error).all())
    self_error = (evaluated_length - stored_length).abs()
    assert bool(torch.isfinite(self_error).all())
    assert float(self_error.max()) > 0.0
    assert validation["constant_field_reproduction"]["ok"]
    assert validation["convex_hull"]["ok"]
    assert validation["positivity"]["ok"]
    assert gradient_probe["query_count"] == 3
    assert gradient_probe["gradient_finite"]
    assert gradient_probe["gradient_mean"] > 0.0
    assert gradient_probe["gradient_max"] >= gradient_probe["gradient_mean"]


def test_mesh_path_selection_prefers_the_resolved_override(tmp_path) -> None:
    config_path = tmp_path / "configured" / "mesh.obj"
    override_path = tmp_path / "override" / "mesh.obj"

    selected_override = select_mesh_path(config_path, override_path)
    selected_config = select_mesh_path("configured/mesh.obj", project_root=tmp_path)

    assert selected_override == override_path.resolve()
    assert selected_config == config_path.resolve()


def test_duplicate_support_is_rejected_as_padded_support() -> None:
    with pytest.raises(RuntimeError, match="duplicate padded support"):
        validate_support_ids(
            torch.tensor([[0, 1, 1]], dtype=torch.long),
            source_count=3,
            expected_width=3,
        )

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import anigroom.rbf_topology_cover as topology_cover
from anigroom.rbf_topology_cover import (
    PatchSelfMembershipError,
    ZeroMassBoundaryError,
    compute_patch_guide_site_distances,
    safe_barycentric_pl_sum,
    select_patch_radii_and_nodes,
    validate_topology_cover_inputs,
)


def _connected_fixture():
    guide_coordinate = np.asarray([0.0, 1.0, 2.0, 5.0], dtype=np.float64)
    distances = np.abs(guide_coordinate[:, None] - guide_coordinate[None, :])
    seed = np.asarray([0, 1, 2, 3], dtype=np.int64)
    delta = np.zeros((4,), dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    guide_face_ids = np.asarray([0, 0, 0, 1], dtype=np.int64)
    guide_barycentric = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return distances, seed, delta, faces, guide_face_ids, guide_barycentric


def _validated_connected():
    return validate_topology_cover_inputs(*_connected_fixture())


def _disconnected_fixture():
    distances = np.asarray(
        [
            [0.0, 1.0, np.inf, np.inf],
            [1.0, 0.0, np.inf, np.inf],
            [np.inf, np.inf, 0.0, 2.0],
            [np.inf, np.inf, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    seed = np.asarray([0, 1, 0, 2, 3, 2], dtype=np.int64)
    delta = np.zeros((6,), dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    guide_face_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    guide_barycentric = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    return distances, seed, delta, faces, guide_face_ids, guide_barycentric


def test_shared_edge_piecewise_linear_guide_site_evaluation_is_continuous() -> None:
    coordinate = np.arange(6, dtype=np.float64)
    distances = np.abs(coordinate[:, None] - coordinate[None, :])
    seed = np.asarray([0, 1, 2, 3], dtype=np.int64)
    delta = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    guide_face_ids = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int64)
    guide_barycentric = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    inputs = validate_topology_cover_inputs(
        distances,
        seed,
        delta,
        faces,
        guide_face_ids,
        guide_barycentric,
    )
    matrix = compute_patch_guide_site_distances(
        inputs,
        patch_chunk_size=2,
        guide_chunk_size=2,
    ).values
    np.testing.assert_allclose(matrix[:, 4], matrix[:, 5], rtol=0.0, atol=0.0)

    face_zero_vertices = faces[0]
    face_one_vertices = faces[1]
    values_zero = distances[:, seed[face_zero_vertices]][:, None, :] + delta[
        face_zero_vertices
    ][None, None, :]
    values_one = distances[:, seed[face_one_vertices]][:, None, :] + delta[
        face_one_vertices
    ][None, None, :]
    edge_from_zero = safe_barycentric_pl_sum(
        values_zero,
        np.asarray([[0.5, 0.0, 0.5]], dtype=np.float64),
    )[:, 0]
    edge_from_one = safe_barycentric_pl_sum(
        values_one,
        np.asarray([[0.5, 0.5, 0.0]], dtype=np.float64),
    )[:, 0]
    np.testing.assert_allclose(edge_from_zero, edge_from_one, rtol=0.0, atol=0.0)


def test_disconnected_folded_components_keep_inf_and_have_no_cross_membership() -> None:
    inputs = validate_topology_cover_inputs(*_disconnected_fixture())
    result = compute_patch_guide_site_distances(inputs, patch_chunk_size=1, guide_chunk_size=1)
    matrix = result.values
    assert not np.isnan(matrix).any()
    assert np.isposinf(matrix[:2, 2:]).all()
    assert np.isposinf(matrix[2:, :2]).all()
    cover = select_patch_radii_and_nodes(result, minimum_active_node_count=1)
    assert cover.report["radii"] == [1.0, 1.0, 2.0, 2.0]
    assert cover.node_distances[0].indices.tolist() == [0]
    assert cover.node_distances[1].indices.tolist() == [1]
    assert cover.node_distances[2].indices.tolist() == [2]
    assert cover.node_distances[3].indices.tolist() == [3]


def test_finite_off_diagonal_zero_or_near_zero_distance_is_rejected() -> None:
    values = [np.array(value, copy=True) for value in _connected_fixture()]
    values[0][0, 1] = 5.0e-11
    values[0][1, 0] = 5.0e-11
    with pytest.raises(ValueError, match="finite off-diagonal"):
        validate_topology_cover_inputs(*values, diagonal_tolerance=1.0e-10)

    disconnected = validate_topology_cover_inputs(*_disconnected_fixture())
    assert disconnected.report["component_count"] == 2
    assert np.isposinf(disconnected.guide_distances[:2, 2:]).all()


def test_safe_zero_barycentric_times_infinity_is_exactly_zero() -> None:
    vertex_values = np.asarray(
        [[[np.inf, 2.5, np.inf]], [[np.inf, 7.0, 9.0]]],
        dtype=np.float64,
    )
    barycentric = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)
    actual = safe_barycentric_pl_sum(vertex_values, barycentric)
    np.testing.assert_array_equal(actual[:, 0], np.asarray([2.5, 7.0]))
    assert not np.isnan(actual).any()


def test_unequal_radii_ties_and_exact_boundary_exclusion() -> None:
    inputs = _validated_connected()
    matrix_result = compute_patch_guide_site_distances(inputs)
    np.testing.assert_array_equal(matrix_result.values, inputs.guide_distances)
    cover = select_patch_radii_and_nodes(
        matrix_result,
        minimum_active_node_count=2,
    )
    np.testing.assert_array_equal(cover.radii, np.asarray([2.0, 4.0, 2.0, 4.0]))
    expected_rows = [[0, 1], [0, 1, 2], [1, 2], [2, 3]]
    for patch_id, expected in enumerate(expected_rows):
        row = cover.node_distances.getrow(patch_id)
        assert row.indices.tolist() == expected
        assert row.indices.tolist() == sorted(set(row.indices.tolist()))
        assert bool((matrix_result.values[patch_id, row.indices] < cover.radii[patch_id]).all())
        assert not bool((matrix_result.values[patch_id] == cover.radii[patch_id])[row.indices].any())
    assert cover.node_distances.getrow(1).indices.tolist() == [0, 1, 2]
    assert cover.report["node_counts"] == [2, 3, 2, 2]


def test_active_guide_incidence_theorem_matches_csr_exactly() -> None:
    matrix = compute_patch_guide_site_distances(_validated_connected())
    cover = select_patch_radii_and_nodes(matrix, minimum_active_node_count=2)
    for patch_id in range(matrix.values.shape[0]):
        expected = np.flatnonzero(matrix.values[patch_id] < cover.radii[patch_id])
        row = cover.node_distances.getrow(patch_id)
        np.testing.assert_array_equal(row.indices, expected)
        np.testing.assert_array_equal(row.data, matrix.values[patch_id, expected])
        assert patch_id in row.indices
    assert isinstance(cover.node_distances, csr_matrix)
    assert cover.node_distances.has_sorted_indices
    assert cover.node_distances.has_canonical_format


def test_chunk_size_determinism_and_memory_bytes_are_invariant() -> None:
    inputs = _validated_connected()
    first = compute_patch_guide_site_distances(
        inputs,
        patch_chunk_size=1,
        guide_chunk_size=1,
    )
    second = compute_patch_guide_site_distances(
        inputs,
        patch_chunk_size=3,
        guide_chunk_size=4,
    )
    np.testing.assert_array_equal(first.values, second.values)
    assert first.report["matrix_memory_bytes"] == second.report["matrix_memory_bytes"]
    repeat = compute_patch_guide_site_distances(
        inputs,
        patch_chunk_size=1,
        guide_chunk_size=1,
    )
    assert first.report == repeat.report
    first_cover = select_patch_radii_and_nodes(first, 2)
    second_cover = select_patch_radii_and_nodes(second, 2)
    np.testing.assert_array_equal(first_cover.radii, second_cover.radii)
    np.testing.assert_array_equal(first_cover.node_distances.indptr, second_cover.node_distances.indptr)
    np.testing.assert_array_equal(first_cover.node_distances.indices, second_cover.node_distances.indices)
    np.testing.assert_array_equal(first_cover.node_distances.data, second_cover.node_distances.data)
    assert first_cover.report == second_cover.report


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("asymmetry", "symmetric"),
        ("diagonal", "diagonal"),
        ("seed", "out-of-range"),
        ("delta", "nonnegative"),
        ("face", "out-of-range"),
        ("bary_negative", r"\[0, 1\]"),
        ("bary_sum", "sum to one"),
    ],
)
def test_strict_validation_failures(mutation: str, message: str) -> None:
    values = [np.array(value, copy=True) for value in _connected_fixture()]
    distances, seed, delta, faces, guide_face_ids, guide_barycentric = values
    if mutation == "asymmetry":
        distances[0, 1] += 0.25
    elif mutation == "diagonal":
        distances[0, 0] = 0.1
    elif mutation == "seed":
        seed[0] = 99
    elif mutation == "delta":
        delta[0] = -0.1
    elif mutation == "face":
        faces[0, 0] = 99
    elif mutation == "bary_negative":
        guide_barycentric[0] = [-0.1, 0.5, 0.6]
    else:
        guide_barycentric[0] = [0.2, 0.2, 0.2]
    with pytest.raises((TypeError, ValueError), match=message):
        validate_topology_cover_inputs(
            distances,
            seed,
            delta,
            faces,
            guide_face_ids,
            guide_barycentric,
        )


def test_invalid_component_inf_rules_and_cross_component_face_fail() -> None:
    distances = np.asarray(
        [[0.0, 1.0, np.inf], [1.0, 0.0, 1.0], [np.inf, 1.0, 0.0]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="finite within"):
        validate_topology_cover_inputs(
            distances,
            np.asarray([0, 1, 2]),
            np.zeros(3),
            np.asarray([[0, 1, 2]]),
            np.asarray([0, 0, 0]),
            np.asarray([[1.0, 0.0, 0.0]] * 3),
        )

    values = [np.array(value, copy=True) for value in _disconnected_fixture()]
    values[3][0] = [0, 1, 3]
    with pytest.raises(ValueError, match="one component"):
        validate_topology_cover_inputs(*values)


def test_no_boundary_and_self_membership_fail_without_fallback() -> None:
    with pytest.raises(ZeroMassBoundaryError, match="no finite distinct"):
        select_patch_radii_and_nodes(
            np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
            minimum_active_node_count=2,
        )
    self_missing = np.asarray(
        [[2.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    with pytest.raises(PatchSelfMembershipError, match="own guide"):
        select_patch_radii_and_nodes(self_missing, minimum_active_node_count=1)
    with pytest.raises(TypeError, match="integer"):
        select_patch_radii_and_nodes(self_missing, minimum_active_node_count=1.5)


def test_phase_b2_apis_are_intentionally_absent() -> None:
    assert not hasattr(topology_cover, "build_vertex_patch_incidence")
    assert not hasattr(topology_cover, "build_face_patch_candidates")
    assert not hasattr(topology_cover, "evaluate_query_topology_distances")

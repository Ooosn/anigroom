from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import anigroom.rbf_topology_cover as topology_cover
from anigroom.rbf_topology_cover import (
    FacePatchCover,
    PatchNodeCover,
    PatchSelfMembershipError,
    VertexPatchCover,
    ZeroMassBoundaryError,
    build_face_patch_candidate_counts,
    build_vertex_patch_active_distances,
    compute_patch_guide_site_distances,
    evaluate_query_topology_distances,
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


def _connected_b2_cover(vertex_chunk_size: int = 2):
    inputs = _validated_connected()
    matrix = compute_patch_guide_site_distances(inputs)
    nodes = select_patch_radii_and_nodes(matrix, 2)
    vertices = build_vertex_patch_active_distances(
        inputs,
        nodes,
        vertex_chunk_size=vertex_chunk_size,
    )
    faces = build_face_patch_candidate_counts(inputs, vertices)
    return inputs, nodes, vertices, faces


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


def test_vertex_active_distance_csr_matches_dense_and_is_chunk_deterministic() -> None:
    inputs, nodes, first, _ = _connected_b2_cover(vertex_chunk_size=1)
    second = build_vertex_patch_active_distances(
        inputs,
        nodes,
        vertex_chunk_size=99,
    )
    dense_distances = (
        inputs.guide_distances[:, inputs.vertex_seed_guide_ids].T
        + inputs.vertex_nearest_distances[:, None]
    )
    dense_active = dense_distances < nodes.radii[None, :]
    for vertex_id in range(inputs.vertex_count):
        row = first.active_distances.getrow(vertex_id)
        expected_ids = np.flatnonzero(dense_active[vertex_id])
        np.testing.assert_array_equal(row.indices, expected_ids)
        np.testing.assert_array_equal(row.data, dense_distances[vertex_id, expected_ids])
        assert row.indices.tolist() == sorted(set(row.indices.tolist()))
    np.testing.assert_array_equal(first.active_distances.indptr, second.active_distances.indptr)
    np.testing.assert_array_equal(first.active_distances.indices, second.active_distances.indices)
    np.testing.assert_array_equal(first.active_distances.data, second.active_distances.data)
    assert first.report["uncovered_vertex_count"] == 0
    assert first.report["csr_memory_bytes"] == second.report["csr_memory_bytes"]
    assert isinstance(first.active_distances, csr_matrix)
    assert not isinstance(first.active_distances, np.ndarray)


def test_face_candidate_counts_are_exact_sparse_incidence_and_strong_cover_reports() -> None:
    inputs, nodes, vertex_cover, face_cover = _connected_b2_cover()
    dense_vertex_active = np.zeros(
        (inputs.vertex_count, inputs.guide_count),
        dtype=np.uint8,
    )
    for vertex_id in range(inputs.vertex_count):
        dense_vertex_active[vertex_id, vertex_cover.active_distances.getrow(vertex_id).indices] = 1
    expected_counts = dense_vertex_active[inputs.faces].sum(axis=1)
    for face_id in range(inputs.face_count):
        row = face_cover.candidate_counts.getrow(face_id)
        expected_ids = np.flatnonzero(expected_counts[face_id] > 0)
        np.testing.assert_array_equal(row.indices, expected_ids)
        np.testing.assert_array_equal(row.data, expected_counts[face_id, expected_ids])
    assert face_cover.candidate_counts.has_sorted_indices
    assert face_cover.candidate_counts.has_canonical_format
    assert face_cover.report["strong_full_face_cover_count"] == 1
    assert face_cover.report["faces_lacking_strong_full_face_cover_ids"] == [1]
    assert face_cover.report["strong_cover_is_sufficient_for_all_barycentric_points"] is True
    np.testing.assert_array_equal(face_cover.patch_radii, nodes.radii)


def test_face_without_candidates_is_reported_by_exact_sparse_multiplication() -> None:
    inputs, nodes, vertex_cover, _ = _connected_b2_cover()
    matrix = vertex_cover.active_distances.copy()
    for vertex_id in inputs.faces[1]:
        matrix.data[matrix.indptr[vertex_id] : matrix.indptr[vertex_id + 1]] = 0.0
    matrix.eliminate_zeros()
    empty_face_vertex_cover = VertexPatchCover(
        active_distances=matrix,
        patch_radii=nodes.radii.copy(),
        patch_node_counts=np.diff(nodes.node_distances.indptr),
        report={},
    )
    face_cover = build_face_patch_candidate_counts(inputs, empty_face_vertex_cover)
    assert face_cover.report["faces_without_candidate_ids"] == [1]
    assert face_cover.candidate_counts.getrow(1).nnz == 0


def _direct_query_distances(inputs, face_id: int, barycentric: np.ndarray) -> np.ndarray:
    vertices = inputs.faces[face_id]
    seeds = inputs.vertex_seed_guide_ids[vertices]
    values = (
        inputs.guide_distances[:, seeds][:, None, :]
        + inputs.vertex_nearest_distances[vertices][None, None, :]
    )
    return safe_barycentric_pl_sum(values, barycentric.reshape(1, 3))[:, 0]


def test_query_ragged_values_match_all_patch_brute_force_and_retain_zero_candidates() -> None:
    inputs, nodes, _, face_cover = _connected_b2_cover()
    face_ids = np.asarray([0, 0, 1, 1], dtype=np.int64)
    barycentric = np.asarray(
        [[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.5, 0.5, 0.0], [0.1, 0.2, 0.7]],
        dtype=np.float64,
    )
    ragged = evaluate_query_topology_distances(
        inputs,
        face_cover,
        face_ids,
        barycentric,
        query_chunk_size=1,
        completeness_patch_chunk_size=1,
    )
    retained_zero = 0
    for query_id in range(face_ids.size):
        begin, end = ragged.indptr[query_id : query_id + 2]
        ids = ragged.patch_ids[begin:end]
        direct = _direct_query_distances(inputs, int(face_ids[query_id]), barycentric[query_id])
        face_row = face_cover.candidate_counts.getrow(int(face_ids[query_id]))
        np.testing.assert_array_equal(ids, face_row.indices)
        np.testing.assert_allclose(ragged.distances[begin:end], direct[ids], rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(ragged.radii[begin:end], nodes.radii[ids])
        omitted = np.setdiff1d(np.arange(inputs.guide_count), ids)
        assert bool((direct[omitted] >= nodes.radii[omitted]).all())
        retained_zero += int((direct[ids] >= nodes.radii[ids]).sum())
    assert retained_zero > 0
    assert ragged.report["retained_zero_weight_candidate_count"] == retained_zero
    assert ragged.report["completeness_verified"] is True
    assert ragged.report["omitted_patch_can_have_positive_PU_weight"] is False


def test_shared_edge_query_distance_continuity_from_adjacent_faces() -> None:
    inputs, _, _, face_cover = _connected_b2_cover()
    ragged = evaluate_query_topology_distances(
        inputs,
        face_cover,
        np.asarray([0, 1]),
        np.asarray([[0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], dtype=np.float64),
    )
    rows = []
    for query_id in range(2):
        begin, end = ragged.indptr[query_id : query_id + 2]
        rows.append(dict(zip(ragged.patch_ids[begin:end], ragged.distances[begin:end])))
    for patch_id in sorted(set(rows[0]).intersection(rows[1])):
        assert rows[0][patch_id] == rows[1][patch_id]


def test_disconnected_folded_b2_has_no_cross_component_candidates() -> None:
    inputs = validate_topology_cover_inputs(*_disconnected_fixture())
    nodes = select_patch_radii_and_nodes(
        compute_patch_guide_site_distances(inputs),
        1,
    )
    vertices = build_vertex_patch_active_distances(inputs, nodes, vertex_chunk_size=1)
    faces = build_face_patch_candidate_counts(inputs, vertices)
    assert set(faces.candidate_counts.getrow(0).indices).issubset({0, 1})
    assert set(faces.candidate_counts.getrow(1).indices).issubset({2, 3})
    ragged = evaluate_query_topology_distances(
        inputs,
        faces,
        np.asarray([0, 1]),
        np.asarray([[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]]),
    )
    assert set(ragged.patch_ids[ragged.indptr[0] : ragged.indptr[1]]).issubset({0, 1})
    assert set(ragged.patch_ids[ragged.indptr[1] : ragged.indptr[2]]).issubset({2, 3})
    assert np.isfinite(ragged.distances).all()


def test_moving_interior_queries_preserve_candidate_completeness() -> None:
    inputs, nodes, _, face_cover = _connected_b2_cover()
    barycentric_rows = []
    face_ids = []
    for face_id in range(inputs.face_count):
        for a in (0.1, 0.3, 0.6):
            for b in (0.1, 0.2):
                if a + b < 1.0:
                    barycentric_rows.append([a, b, 1.0 - a - b])
                    face_ids.append(face_id)
    barycentric = np.asarray(barycentric_rows, dtype=np.float64)
    ragged = evaluate_query_topology_distances(
        inputs,
        face_cover,
        np.asarray(face_ids, dtype=np.int64),
        barycentric,
        query_chunk_size=2,
        completeness_patch_chunk_size=2,
    )
    for query_id, face_id in enumerate(face_ids):
        direct = _direct_query_distances(inputs, face_id, barycentric[query_id])
        begin, end = ragged.indptr[query_id : query_id + 2]
        included = ragged.patch_ids[begin:end]
        omitted = np.setdiff1d(np.arange(inputs.guide_count), included)
        assert bool((direct[omitted] >= nodes.radii[omitted]).all())


def test_b2_invalid_cover_radius_csr_face_bary_and_chunk_failures() -> None:
    inputs, nodes, vertex_cover, face_cover = _connected_b2_cover()
    with pytest.raises(ValueError, match="positive"):
        build_vertex_patch_active_distances(inputs, nodes, vertex_chunk_size=0)
    bad_radii = PatchNodeCover(
        radii=np.asarray([1.0]),
        node_distances=nodes.node_distances,
        report={},
    )
    with pytest.raises(ValueError, match="shape"):
        build_vertex_patch_active_distances(inputs, bad_radii)
    wrong_vertex_shape = VertexPatchCover(
        active_distances=csr_matrix((1, 1)),
        patch_radii=nodes.radii,
        patch_node_counts=np.diff(nodes.node_distances.indptr),
        report={},
    )
    with pytest.raises(ValueError, match="wrong shape"):
        build_face_patch_candidate_counts(inputs, wrong_vertex_shape)

    corrupted_counts = face_cover.candidate_counts.copy().astype(np.int64)
    corrupted_counts.data[0] = 4
    bad_face_cover = FacePatchCover(
        candidate_counts=corrupted_counts,
        patch_radii=face_cover.patch_radii,
        patch_node_counts=face_cover.patch_node_counts,
        report={},
    )
    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        evaluate_query_topology_distances(
            inputs,
            bad_face_cover,
            np.asarray([0]),
            np.asarray([[1.0, 0.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="out-of-range"):
        evaluate_query_topology_distances(
            inputs,
            face_cover,
            np.asarray([99]),
            np.asarray([[1.0, 0.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_query_topology_distances(
            inputs,
            face_cover,
            np.asarray([0]),
            np.asarray([[0.2, 0.2, 0.2]]),
        )
    with pytest.raises(ValueError, match="positive"):
        evaluate_query_topology_distances(
            inputs,
            face_cover,
            np.asarray([0]),
            np.asarray([[1.0, 0.0, 0.0]]),
            query_chunk_size=0,
        )

    omitted = face_cover.candidate_counts.copy().tolil()
    omitted[0, 0] = 0
    omitted = omitted.tocsr()
    omitted.eliminate_zeros()
    incomplete = FacePatchCover(
        candidate_counts=omitted,
        patch_radii=face_cover.patch_radii,
        patch_node_counts=face_cover.patch_node_counts,
        report={},
    )
    with pytest.raises(topology_cover.TopologyCoverError, match="incomplete"):
        evaluate_query_topology_distances(
            inputs,
            incomplete,
            np.asarray([0]),
            np.asarray([[1.0, 0.0, 0.0]]),
        )


def test_b2_has_no_dense_g_by_v_return_or_forbidden_selection_path() -> None:
    _, _, vertex_cover, face_cover = _connected_b2_cover()
    assert isinstance(vertex_cover.active_distances, csr_matrix)
    assert isinstance(face_cover.candidate_counts, csr_matrix)
    assert not hasattr(topology_cover, "build_dense_vertex_patch_matrix")
    assert not hasattr(topology_cover, "truncate_vertex_patches_topk")

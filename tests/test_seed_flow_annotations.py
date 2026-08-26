from __future__ import annotations

import json

import numpy as np
import pytest

from anigroom.flow_annotations import SparseFlowAnnotations, make_arrow, save_annotations
from anigroom.seed_flow_annotations import (
    SCHEMA_NAME,
    SeedFlowAnnotations,
    SeedFlowValidationError,
    build_seed_neighbor_graph,
    load_seed_flow_annotations,
    make_flow_seed,
    nearest_seed_direction,
    propagate_follower_directions,
    save_seed_flow_annotations,
)


def _image(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"seed-image")
    return path


def test_seed_schema_roundtrip_contains_no_endpoint_or_length(tmp_path) -> None:
    image = _image(tmp_path)
    seeds = (
        make_flow_seed("a", (2, 3), (3, 4), 10, 8, manual=True),
        make_flow_seed("b", (6, 4), (-1, 0), 10, 8, manual=False),
    )
    document = SeedFlowAnnotations.from_image(image, 10, 8, seeds, updated_utc="2026-08-26T08:00:00Z")
    output = save_seed_flow_annotations(document, tmp_path)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema"] == SCHEMA_NAME
    assert "end_px" not in output.read_text(encoding="utf-8")
    assert "length" not in output.read_text(encoding="utf-8")
    assert load_seed_flow_annotations(output) == document


def test_legacy_arrows_load_as_manual_seeds(tmp_path) -> None:
    image = _image(tmp_path)
    legacy = SparseFlowAnnotations.from_image(
        image,
        10,
        8,
        [make_arrow("old", (2, 3), (5, 7), 10, 8, 0.7)],
        updated_utc="2026-08-26T08:00:00Z",
    )
    path = save_annotations(legacy, tmp_path)
    loaded = load_seed_flow_annotations(path, image_path=image, verify_image=True)

    assert len(loaded.seeds) == 1
    assert loaded.seeds[0].position_px == (2.0, 3.0)
    assert loaded.seeds[0].direction_px == pytest.approx((0.6, 0.8))
    assert loaded.seeds[0].manual is True


def test_seed_validation_rejects_non_unit_and_unknown_fields(tmp_path) -> None:
    image = _image(tmp_path)
    seed = make_flow_seed("a", (2, 3), (1, 0), 10, 8, manual=True)
    document = SeedFlowAnnotations.from_image(image, 10, 8, [seed], updated_utc="2026-08-26T08:00:00Z")
    payload = document.to_dict()
    payload["seeds"][0]["direction_px"] = [2.0, 0.0]
    path = tmp_path / "bad.flow.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SeedFlowValidationError):
        load_seed_flow_annotations(path)

    payload = document.to_dict()
    payload["seeds"][0]["length_px"] = 20
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SeedFlowValidationError):
        load_seed_flow_annotations(path)


def test_anchor_driven_propagation_keeps_manual_values_exact() -> None:
    positions = np.stack([np.arange(7), np.zeros(7)], axis=1)
    graph = build_seed_neighbor_graph(positions, neighbor_count=2)
    directions = np.tile(np.array([[1.0, 0.0]]), (7, 1))
    directions[0] = (0.0, 1.0)
    directions[-1] = (0.0, 1.0)
    manual = np.array([True, False, False, False, False, False, True])

    output, active = propagate_follower_directions(
        directions,
        manual,
        graph,
        iterations=30,
        relaxation=0.8,
    )

    np.testing.assert_allclose(output[[0, -1]], np.array([[0.0, 1.0], [0.0, 1.0]]))
    assert output[3, 1] > output[3, 0]
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1.0, atol=1.0e-7)
    assert set(active) == {1, 2, 3, 4, 5}


def test_local_ring_propagation_does_not_touch_distant_followers() -> None:
    positions = np.stack([np.arange(12), np.zeros(12)], axis=1)
    graph = build_seed_neighbor_graph(positions, neighbor_count=2)
    directions = np.tile(np.array([[1.0, 0.0]]), (12, 1))
    directions[0] = (0.0, 1.0)
    manual = np.zeros(12, dtype=bool)
    manual[0] = True

    output, active = propagate_follower_directions(
        directions,
        manual,
        graph,
        changed_indices=[0],
        rings=2,
        iterations=12,
    )

    assert active.max() <= 4
    np.testing.assert_allclose(output[8:], directions[8:])


def test_nearest_direction_initializes_new_seeds_and_handles_empty_input() -> None:
    direction = nearest_seed_direction(
        (0.2, 0.0),
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        neighbor_count=2,
    )
    assert direction[1] > 0.95
    assert nearest_seed_direction((1, 1), np.empty((0, 2)), np.empty((0, 2))) == (0.0, 1.0)


def test_graph_handles_zero_and_one_seed() -> None:
    empty = build_seed_neighbor_graph(np.empty((0, 2)))
    single = build_seed_neighbor_graph(np.array([[1.0, 2.0]]))
    assert empty.indices.shape == (0, 0)
    assert single.indices.shape == (1, 0)

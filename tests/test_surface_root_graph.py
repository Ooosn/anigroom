from __future__ import annotations

import numpy as np

from anigroom.flow.surface_graph import build_surface_root_graph


def test_surface_graph_does_not_connect_nearby_disconnected_sheets() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.01],
            [1.0, 0.0, 0.01],
            [1.0, 1.0, 0.01],
            [0.0, 1.0, 0.01],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
        ],
        dtype=np.int64,
    )
    root_points = np.asarray(
        [
            [0.2, 0.1, 0.0],
            [0.2, 0.9, 0.0],
            [0.2, 0.1, 0.01],
            [0.2, 0.9, 0.01],
        ],
        dtype=np.float32,
    )
    root_face_ids = np.asarray([0, 1, 2, 3], dtype=np.int64)

    graph = build_surface_root_graph(
        vertices=vertices,
        faces=faces,
        root_points=root_points,
        root_face_ids=root_face_ids,
        k=1,
        device="cpu",
    )

    np.testing.assert_array_equal(graph.indices.numpy().reshape(-1), np.asarray([1, 0, 3, 2]))
    assert graph.report["connected_components"] == 2

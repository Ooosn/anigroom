from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.surface_interpolation import (
    SurfaceFieldInterpolator,
    build_hierarchical_surface_edges,
    density_invariant_log_scalar_smoothness,
    harmonic_inpaint_physical,
)
from anigroom.flow.clean_flow import (
    clean_flow_smoothness_loss,
)
from anigroom.flow.direction_geometry import parallel_transport_vectors
from anigroom.grooming import direction_to_local_components, local_components_to_world
from tools.train_white_tiger_stage1 import (
    data_clamped_clean_flow_length,
    smooth_metric_uses_full_relative_length_field,
    symmetric_relative_edge_difference,
)


def square_face_id(x: float, y: float, offset: int) -> int:
    return offset + (0 if y <= x else 1)


def regular_square_mesh(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [x / resolution, y / resolution, 0.0]
            for y in range(resolution + 1)
            for x in range(resolution + 1)
        ],
        dtype=np.float32,
    )
    faces: list[list[int]] = []
    stride = resolution + 1
    for y in range(resolution):
        for x in range(resolution):
            v00 = y * stride + x
            v10 = v00 + 1
            v01 = v00 + stride
            v11 = v01 + 1
            faces.extend([[v00, v10, v11], [v00, v11, v01]])
    return vertices, np.asarray(faces, dtype=np.int64)


def regular_square_sources(
    count_per_axis: int,
    mesh_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    points: list[list[float]] = []
    face_ids: list[int] = []
    for y_index in range(count_per_axis):
        for x_index in range(count_per_axis):
            x = (x_index + 0.5) / count_per_axis
            y = (y_index + 0.5) / count_per_axis
            cell_x = min(int(x * mesh_resolution), mesh_resolution - 1)
            cell_y = min(int(y * mesh_resolution), mesh_resolution - 1)
            local_x = x * mesh_resolution - cell_x
            local_y = y * mesh_resolution - cell_y
            cell_id = cell_y * mesh_resolution + cell_x
            face_ids.append(2 * cell_id + (0 if local_y <= local_x else 1))
            points.append([x, y, 0.0])
    return torch.tensor(points, dtype=torch.float32), torch.tensor(face_ids, dtype=torch.long)


def main() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.001],
            [1.0, 0.0, 0.001],
            [1.0, 1.0, 0.001],
            [0.0, 1.0, 0.001],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
        dtype=np.int64,
    )
    guide_xy = np.asarray(
        [[0.20, 0.20], [0.75, 0.25], [0.80, 0.80], [0.25, 0.75]],
        dtype=np.float32,
    )
    render_xy = np.asarray(
        [
            [0.12, 0.12],
            [0.40, 0.18],
            [0.72, 0.20],
            [0.88, 0.40],
            [0.85, 0.85],
            [0.55, 0.82],
            [0.25, 0.72],
            [0.15, 0.45],
        ],
        dtype=np.float32,
    )

    guide_points = []
    guide_faces = []
    render_points = []
    render_faces = []
    for sheet, z in enumerate((0.0, 0.001)):
        for x, y in guide_xy.tolist():
            guide_points.append([x, y, z])
            guide_faces.append(square_face_id(x, y, 2 * sheet))
        for x, y in render_xy.tolist():
            render_points.append([x, y, z])
            render_faces.append(square_face_id(x, y, 2 * sheet))

    guide_points_t = torch.tensor(guide_points, dtype=torch.float32)
    guide_faces_t = torch.tensor(guide_faces, dtype=torch.long)
    render_points_t = torch.tensor(render_points, dtype=torch.float32)
    render_faces_t = torch.tensor(render_faces, dtype=torch.long)
    interpolator = SurfaceFieldInterpolator(
        vertices=vertices,
        faces=faces,
        source_points=guide_points_t,
        source_face_ids=guide_faces_t,
        neighbor_count=2,
        device="cpu",
    )

    support = interpolator.build_support(render_points_t, render_faces_t)
    render_edges = build_hierarchical_surface_edges(
        render_points_t,
        support.indices,
        neighbor_count=2,
    )
    guide_edges = interpolator.source_neighbor_edges(2)
    guide_sheet = torch.arange(guide_points_t.shape[0]) // guide_xy.shape[0]
    render_sheet = torch.arange(render_points_t.shape[0]) // render_xy.shape[0]

    assert render_edges.shape == (render_points_t.shape[0] * 2, 2)
    assert guide_edges.shape == (guide_points_t.shape[0] * 2, 2)
    assert not bool((render_edges[:, 0] == render_edges[:, 1]).any())
    assert not bool((guide_edges[:, 0] == guide_edges[:, 1]).any())
    assert bool((render_sheet[render_edges[:, 0]] == render_sheet[render_edges[:, 1]]).all())
    assert bool((guide_sheet[guide_edges[:, 0]] == guide_sheet[guide_edges[:, 1]]).all())

    distance = torch.cdist(render_points_t, render_points_t)
    distance.fill_diagonal_(float("inf"))
    euclidean_neighbor = distance.argmin(dim=1)
    assert bool((render_sheet != render_sheet[euclidean_neighbor]).any()), (
        "the regression geometry must expose an invalid Euclidean cross-sheet edge"
    )

    folded_values = torch.zeros((guide_points_t.shape[0], 1), dtype=torch.float32)
    folded_reliable = torch.zeros((guide_points_t.shape[0],), dtype=torch.bool)
    folded_reliable[0] = True
    folded_reliable[guide_xy.shape[0]] = True
    folded_values[0] = 1.0
    folded_values[guide_xy.shape[0]] = 10.0
    folded_filled = harmonic_inpaint_physical(
        folded_values,
        guide_points_t,
        folded_reliable,
        guide_edges,
    )
    torch.testing.assert_close(
        folded_filled[: guide_xy.shape[0]],
        torch.ones((guide_xy.shape[0], 1)),
    )
    torch.testing.assert_close(
        folded_filled[guide_xy.shape[0] :],
        torch.full((guide_xy.shape[0], 1), 10.0),
    )

    new_point = torch.tensor([[0.52, 0.48, 0.0]], dtype=torch.float32)
    new_face = torch.tensor([square_face_id(0.52, 0.48, 0)], dtype=torch.long)
    densified_points = torch.cat([render_points_t, new_point], dim=0)
    densified_faces = torch.cat([render_faces_t, new_face], dim=0)
    densified_support = interpolator.build_support(densified_points, densified_faces)
    densified_edges = build_hierarchical_surface_edges(
        densified_points,
        densified_support.indices,
        neighbor_count=2,
    )
    densified_sheet = torch.cat([render_sheet, torch.zeros((1,), dtype=torch.long)])
    assert densified_edges.shape == (densified_points.shape[0] * 2, 2)
    assert bool(
        (densified_sheet[densified_edges[:, 0]] == densified_sheet[densified_edges[:, 1]]).all()
    )

    new_guide_point = torch.tensor([[0.52, 0.48, 0.0]], dtype=torch.float32)
    new_guide_face = torch.tensor([square_face_id(0.52, 0.48, 0)], dtype=torch.long)
    densified_guide_points = torch.cat([guide_points_t, new_guide_point], dim=0)
    densified_guide_faces = torch.cat([guide_faces_t, new_guide_face], dim=0)
    densified_interpolator = SurfaceFieldInterpolator(
        vertices=vertices,
        faces=faces,
        source_points=densified_guide_points,
        source_face_ids=densified_guide_faces,
        neighbor_count=2,
        device="cpu",
    )
    guide_densified_support = densified_interpolator.build_support(render_points_t, render_faces_t)
    guide_densified_render_edges = build_hierarchical_surface_edges(
        render_points_t,
        guide_densified_support.indices,
        neighbor_count=2,
    )
    guide_densified_edges = densified_interpolator.source_neighbor_edges(2)
    densified_guide_sheet = torch.cat([guide_sheet, torch.zeros((1,), dtype=torch.long)])
    assert guide_densified_render_edges.shape == (render_points_t.shape[0] * 2, 2)
    assert guide_densified_edges.shape == (densified_guide_points.shape[0] * 2, 2)
    assert bool(
        (
            render_sheet[guide_densified_render_edges[:, 0]]
            == render_sheet[guide_densified_render_edges[:, 1]]
        ).all()
    )
    assert bool(
        (
            densified_guide_sheet[guide_densified_edges[:, 0]]
            == densified_guide_sheet[guide_densified_edges[:, 1]]
        ).all()
    )

    curved_normals = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    source_direction = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    target_direction = parallel_transport_vectors(
        source_direction,
        curved_normals[:1],
        curved_normals[1:],
    )
    curved_directions = torch.cat([source_direction, target_direction], dim=0)
    curved_edge = torch.tensor([[0, 1]], dtype=torch.long)
    ambient_direction_loss = clean_flow_smoothness_loss(curved_directions, curved_edge)
    transported_direction_loss = clean_flow_smoothness_loss(
        curved_directions,
        curved_edge,
        normals=curved_normals,
    )
    assert float(ambient_direction_loss) > 0.1
    assert float(transported_direction_loss) < 1.0e-7

    lengths = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    scaled_lengths = 3.0 * lengths
    relative = symmetric_relative_edge_difference(lengths, curved_edge[:, 0], curved_edge[:, 1])
    scaled_relative = symmetric_relative_edge_difference(
        scaled_lengths,
        curved_edge[:, 0],
        curved_edge[:, 1],
    )
    assert torch.allclose(relative, scaled_relative, atol=1.0e-7, rtol=0.0)
    assert smooth_metric_uses_full_relative_length_field("surface_covariant_full")
    assert not smooth_metric_uses_full_relative_length_field("surface_covariant")

    density_vertices, density_faces = regular_square_mesh(16)
    coarse_points, coarse_faces = regular_square_sources(4, 16)
    dense_points, dense_faces = regular_square_sources(8, 16)
    coarse_interpolator = SurfaceFieldInterpolator(
        vertices=density_vertices,
        faces=density_faces,
        source_points=coarse_points,
        source_face_ids=coarse_faces,
        neighbor_count=8,
        device="cpu",
    )
    dense_interpolator = SurfaceFieldInterpolator(
        vertices=density_vertices,
        faces=density_faces,
        source_points=dense_points,
        source_face_ids=dense_faces,
        neighbor_count=8,
        device="cpu",
    )
    coarse_graph = coarse_interpolator.source_neighbor_graph(8)
    dense_graph = dense_interpolator.source_neighbor_graph(8)
    coarse_values = torch.exp(
        0.7 * coarse_points[:, :1] - 0.4 * coarse_points[:, 1:2]
    ).requires_grad_(True)
    dense_values = torch.exp(0.7 * dense_points[:, :1] - 0.4 * dense_points[:, 1:2])
    coarse_old = symmetric_relative_edge_difference(
        coarse_values,
        coarse_graph.edges[:, 0],
        coarse_graph.edges[:, 1],
    ).square().mean()
    dense_old = symmetric_relative_edge_difference(
        dense_values,
        dense_graph.edges[:, 0],
        dense_graph.edges[:, 1],
    ).square().mean()
    coarse_metric = density_invariant_log_scalar_smoothness(
        coarse_values,
        coarse_graph,
        coarse_graph.reference_spacing,
    )
    dense_metric = density_invariant_log_scalar_smoothness(
        dense_values,
        dense_graph,
        coarse_graph.reference_spacing,
    )
    assert float(dense_old / coarse_old) < 0.45
    assert abs(float(dense_metric / coarse_metric) - 1.0) < 0.20
    coarse_metric.backward()
    assert coarse_values.grad is not None
    assert bool(torch.isfinite(coarse_values.grad).all())
    assert float(coarse_values.grad.abs().sum()) > 0.0

    length_points = torch.stack(
        [
            torch.arange(10, dtype=torch.float32),
            torch.zeros(10, dtype=torch.float32),
            torch.zeros(10, dtype=torch.float32),
        ],
        dim=-1,
    )
    length_edges = torch.tensor(
        [[i, i + 1] for i in range(9)] + [[i + 1, i] for i in range(9)],
        dtype=torch.long,
    )
    shell_height = torch.tensor(
        [0.001, 0.020, 0.024, 0.028, 0.032, 0.036, 0.040, 0.044, 0.048, 0.500],
        dtype=torch.float32,
    )
    length_sample = {
        "confidence": torch.ones(10, dtype=torch.float32),
        "valid": torch.ones(10, dtype=torch.bool),
        "shell_height": shell_height,
    }
    (
        reconstructed_length,
        reliable_length,
        filled_length,
        length_q05,
        length_q95,
        observed_length_count,
        reliable_length_count,
    ) = data_clamped_clean_flow_length(
        length_points,
        length_edges,
        length_sample,
        SimpleNamespace(
            clean_flow_length_init_min_confidence=0.5,
            clean_flow_length_init_scale=1.0,
        ),
        label="surface length regression",
    )
    assert observed_length_count == 10
    assert reliable_length_count == 8
    assert int(reliable_length.sum()) == 8
    assert bool(filled_length.all())
    assert np.isclose(length_q05, float(torch.quantile(shell_height, 0.05)))
    assert np.isclose(length_q95, float(torch.quantile(shell_height, 0.95)))
    torch.testing.assert_close(reconstructed_length[0], shell_height[1:2])
    torch.testing.assert_close(reconstructed_length[-1], shell_height[-2:-1])
    torch.testing.assert_close(reconstructed_length[1:-1, 0], shell_height[1:-1])
    assert bool((reconstructed_length > 0.0).all())

    roundtrip_normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    roundtrip_tangents = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    roundtrip_bitangents = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    roundtrip_directions = torch.nn.functional.normalize(
        torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.6, 0.0, 0.8]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    direction_local = direction_to_local_components(
        roundtrip_directions,
        roundtrip_normals,
        roundtrip_tangents,
        roundtrip_bitangents,
    )
    reconstructed = local_components_to_world(
        direction_local,
        roundtrip_normals,
        roundtrip_tangents,
        roundtrip_bitangents,
        normalize=True,
    )
    roundtrip_error = 1.0 - (reconstructed * roundtrip_directions).sum(dim=-1)
    assert float(roundtrip_error.abs().max()) < 1.0e-7
    print(
        {
            "guide_edges": int(guide_edges.shape[0]),
            "render_edges": int(render_edges.shape[0]),
            "densified_render_edges": int(densified_edges.shape[0]),
            "guide_densified_edges": int(guide_densified_edges.shape[0]),
            "guide_densified_render_edges": int(guide_densified_render_edges.shape[0]),
            "euclidean_cross_sheet_roots": int(
                (render_sheet != render_sheet[euclidean_neighbor]).sum().item()
            ),
            "ambient_direction_loss": float(ambient_direction_loss),
            "transported_direction_loss": float(transported_direction_loss),
            "relative_length_difference": float(relative.item()),
            "density_old_ratio": float(dense_old / coarse_old),
            "density_invariant_ratio": float(dense_metric / coarse_metric),
            "direction_roundtrip_max_error": float(roundtrip_error.abs().max()),
            "surface_length_reliable_count": int(reliable_length_count),
            "surface_length_min": float(reconstructed_length.min()),
            "surface_length_max": float(reconstructed_length.max()),
        }
    )


if __name__ == "__main__":
    main()

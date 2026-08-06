from __future__ import annotations

import numpy as np
import pytest
import torch

from anigroom.surface_interpolation import (
    SurfaceFieldInterpolator,
    build_hierarchical_surface_edges,
)


def _reference_edges(
    points: torch.Tensor,
    support: torch.Tensor,
    neighbor_count: int,
) -> torch.Tensor:
    points_np = np.asarray(points.detach().cpu(), dtype=np.float32)
    support_np = np.asarray(support.detach().cpu(), dtype=np.int64)
    root_count = int(points_np.shape[0])
    k = min(int(neighbor_count), root_count - 1)
    primary = support_np[:, 0]
    buckets = {
        int(guide_id): np.flatnonzero(primary == guide_id)
        for guide_id in np.unique(primary).tolist()
    }
    dst = np.empty((root_count, k), dtype=np.int64)
    for root_id in range(root_count):
        candidates: set[int] = set()
        for guide_id in support_np[root_id].tolist():
            candidates.update(buckets.get(int(guide_id), ()))
        candidates.discard(root_id)
        if len(candidates) < k:
            active_guides = np.unique(support_np[root_id])
            while len(candidates) < k:
                expanded_ids = np.flatnonzero(
                    np.isin(support_np, active_guides).any(axis=1)
                )
                previous_count = len(candidates)
                candidates.update(expanded_ids.tolist())
                candidates.discard(root_id)
                if len(candidates) >= k:
                    break
                expanded_guides = np.unique(support_np[expanded_ids])
                if expanded_guides.size == active_guides.size and np.array_equal(
                    expanded_guides,
                    active_guides,
                ):
                    break
                active_guides = expanded_guides
                if len(candidates) == previous_count:
                    break
        if len(candidates) < k:
            raise RuntimeError("reference graph has insufficient topology-valid neighbors")
        candidate_ids = np.asarray(sorted(candidates), dtype=np.int64)
        distances = np.linalg.norm(
            points_np[candidate_ids] - points_np[root_id : root_id + 1],
            axis=-1,
        )
        order = np.lexsort((candidate_ids, distances))[:k]
        dst[root_id] = candidate_ids[order]
    src = np.repeat(np.arange(root_count, dtype=np.int64), k)
    return torch.as_tensor(
        np.stack([src, dst.reshape(-1)], axis=-1),
        device=points.device,
        dtype=torch.long,
    )


def _graph_fixture(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(41)
    guide_count = 8
    roots_per_guide = 12
    root_count = guide_count * roots_per_guide
    points = torch.randn((root_count, 3), generator=generator, dtype=torch.float32)
    primary = torch.arange(guide_count).repeat_interleave(roots_per_guide)
    support = torch.stack(
        [
            primary,
            (primary + 1) % guide_count,
            (primary + 3) % guide_count,
            (primary + 5) % guide_count,
        ],
        dim=1,
    )
    return points.to(device), support.to(device)


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_accelerated_graph_matches_exact_reference(device: torch.device) -> None:
    points, support = _graph_fixture(device)
    actual = build_hierarchical_surface_edges(points, support, neighbor_count=8)
    expected = _reference_edges(points, support, neighbor_count=8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_accelerated_graph_preserves_duplicate_support_expansion() -> None:
    points = torch.tensor(
        [[float(index), float(index % 3), 0.0] for index in range(16)],
        dtype=torch.float32,
    )
    primary = torch.arange(4).repeat_interleave(4)
    support = torch.stack(
        [primary, primary, (primary + 1) % 4],
        dim=1,
    )
    actual = build_hierarchical_surface_edges(points, support, neighbor_count=6)
    expected = _reference_edges(points, support, neighbor_count=6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_support_matches_cpu_reference_exactly() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 7],
            [3, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ],
        dtype=np.int64,
    )
    source_points = torch.tensor(
        [
            [0.10, 0.10, 0.0],
            [0.42, 0.12, 0.0],
            [0.82, 0.15, 0.0],
            [0.88, 0.45, 0.0],
            [0.78, 0.82, 0.0],
            [0.48, 0.88, 0.0],
            [0.18, 0.78, 0.0],
            [0.12, 0.42, 0.0],
        ],
        dtype=torch.float32,
    )
    source_faces = torch.tensor([0, 0, 2, 2, 6, 7, 5, 1], dtype=torch.long)
    query_points = torch.tensor(
        [
            [0.15, 0.15, 0.0],
            [0.35, 0.30, 0.0],
            [0.70, 0.20, 0.0],
            [0.80, 0.42, 0.0],
            [0.72, 0.72, 0.0],
            [0.45, 0.78, 0.0],
            [0.22, 0.70, 0.0],
            [0.18, 0.38, 0.0],
        ],
        dtype=torch.float32,
    )
    query_faces = torch.tensor([0, 0, 2, 3, 6, 7, 5, 1], dtype=torch.long)

    cpu_interpolator = SurfaceFieldInterpolator(
        vertices=vertices,
        faces=faces,
        source_points=source_points,
        source_face_ids=source_faces,
        neighbor_count=4,
        device="cpu",
    )
    cuda_interpolator = SurfaceFieldInterpolator(
        vertices=vertices,
        faces=faces,
        source_points=source_points,
        source_face_ids=source_faces,
        neighbor_count=4,
        device="cuda",
    )
    expected = cpu_interpolator.build_support(query_points, query_faces)
    actual = cuda_interpolator.build_support(
        query_points.cuda(),
        query_faces.cuda(),
    )

    torch.testing.assert_close(actual.indices.cpu(), expected.indices, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.vertex_path_distances.cpu(),
        expected.vertex_path_distances,
        rtol=0,
        atol=0,
    )

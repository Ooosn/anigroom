from __future__ import annotations

import torch
import torch.nn.functional as F

from anigroom.flow.clean_flow import (
    CleanFlowTargets,
    clean_flow_smoothness_loss,
    sample_clean_flow_targets,
)
from anigroom.flow.direction_geometry import (
    parallel_transport_vector_field,
    parallel_transport_vectors,
)
from anigroom.surface_interpolation import reconstruct_surface_directions


def _targets() -> CleanFlowTargets:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.01, 0.0, 0.001],
            [1.01, 0.0, 0.001],
        ],
        dtype=torch.float32,
    )
    normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    directions = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.25],
                [1.0, 0.0, 0.25],
                [0.0, 1.0, -0.25],
                [0.0, 1.0, -0.25],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    count = int(points.shape[0])
    ones = torch.ones((count,), dtype=torch.float32)
    return CleanFlowTargets(
        points=points,
        normals=normals,
        directions=directions,
        confidence=ones,
        anchor_confidence=ones,
        lambda_values=torch.tensor([1.0, 2.0, 10.0, 20.0]),
        shell_height=torch.tensor([1.0, 2.0, 10.0, 20.0]),
        raw_shell_height=torch.tensor([3.0, 4.0, 30.0, 40.0]),
        local_spacing=ones,
        observed=torch.ones((count,), dtype=torch.bool),
        anchor=torch.ones((count,), dtype=torch.bool),
        source_path="synthetic",
    )


def test_parallel_transport_preserves_surface_relative_lift() -> None:
    source_normal = torch.tensor([[0.0, 0.0, 1.0]])
    target_normal = torch.tensor([[0.0, 1.0, 0.0]])
    source_direction = F.normalize(torch.tensor([[1.0, 0.0, 0.25]]), dim=-1)
    expected = F.normalize(torch.tensor([[1.0, 0.25, 0.0]]), dim=-1)

    transported = parallel_transport_vectors(source_direction, source_normal, target_normal)

    torch.testing.assert_close(transported, expected, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(
        (source_direction * source_normal).sum(dim=-1),
        (transported * target_normal).sum(dim=-1),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_vector_field_transport_has_nonzero_gradient_at_zero() -> None:
    source_normal = torch.tensor([[0.0, 0.0, 1.0]])
    target_normal = torch.tensor([[0.0, 1.0, 0.0]])
    source_vector = torch.zeros((1, 3), requires_grad=True)
    target_vector = torch.tensor([[0.25, -0.40, 0.15]])

    transported = parallel_transport_vector_field(
        source_vector,
        source_normal,
        target_normal,
    )
    loss = (transported - target_vector).square().sum()
    loss.backward()

    assert source_vector.grad is not None
    assert torch.isfinite(source_vector.grad).all()
    assert float(source_vector.grad.abs().sum()) > 0.0


def test_direction_is_surface_aware_while_scalar_sampling_stays_legacy() -> None:
    targets = _targets()
    query = torch.tensor([[0.02, 0.0, 0.0]], dtype=torch.float32)
    query_normal = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    sampled = sample_clean_flow_targets(targets, query, query_normal, k=2, chunk_size=1)

    expected_direction = F.normalize(torch.tensor([[1.0, 0.0, 0.25]]), dim=-1)
    assert float((sampled["direction"] * expected_direction).sum()) > 0.999

    distances = torch.cdist(query, targets.points)
    values, ids = torch.topk(distances, k=2, dim=-1, largest=False)
    legacy_weights = 1.0 / values.clamp_min(1.0e-6).square()
    legacy_weights = legacy_weights / legacy_weights.sum(dim=-1, keepdim=True)
    expected_shell = (targets.shell_height[ids] * legacy_weights).sum(dim=1)
    expected_raw_shell = (targets.raw_shell_height[ids] * legacy_weights).sum(dim=1)
    expected_lambda = (targets.lambda_values[ids] * legacy_weights).sum(dim=1)

    torch.testing.assert_close(sampled["shell_height"], expected_shell)
    torch.testing.assert_close(sampled["raw_shell_height"], expected_raw_shell)
    torch.testing.assert_close(sampled["lambda"], expected_lambda)


def test_direction_smoothness_prioritizes_uncertain_edges() -> None:
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    edges = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    uncertain_error = clean_flow_smoothness_loss(
        directions,
        edges,
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    observed_error = clean_flow_smoothness_loss(
        directions,
        edges,
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
    )

    assert float(uncertain_error) > float(observed_error)


def test_surface_direction_reconstruction_preserves_coherent_field() -> None:
    directions = F.normalize(
        torch.tensor(
            [[1.0, 0.0, 0.2], [1.0, 0.0, 0.2], [1.0, 0.0, 0.2]],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=torch.float32)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edges = torch.tensor([[0, 1], [1, 0], [1, 2], [2, 1]], dtype=torch.long)

    reconstructed, reliability, supported = reconstruct_surface_directions(
        directions,
        normals,
        points,
        torch.ones(3),
        edges,
    )

    torch.testing.assert_close(reconstructed, directions, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(reliability, torch.ones(3), atol=1.0e-6, rtol=1.0e-6)
    assert bool(supported.all())


def test_surface_direction_reconstruction_repairs_isolated_outlier() -> None:
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=torch.float32)
    points = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges = torch.tensor([[0, 2], [1, 0], [1, 2], [2, 0]], dtype=torch.long)

    reconstructed, reliability, _ = reconstruct_surface_directions(
        directions,
        normals,
        points,
        torch.ones(3),
        edges,
    )

    assert float(reconstructed[1, 0]) > 0.999
    assert float(reconstructed[1, 1].abs()) < 1.0e-4
    assert float(reliability[1]) < 1.0e-4


def test_surface_direction_reconstruction_preserves_multidirectional_boundary() -> None:
    directions = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]] * 3, dtype=torch.float32)
    points = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edges = torch.tensor([[0, 1], [0, 2], [1, 0], [2, 0]], dtype=torch.long)

    reconstructed, _, _ = reconstruct_surface_directions(
        directions,
        normals,
        points,
        torch.ones(3),
        edges,
    )

    torch.testing.assert_close(reconstructed[0], directions[0], atol=1.0e-6, rtol=1.0e-6)

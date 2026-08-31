"""Pure Torch algebra for topology-covered RBF partition-of-unity fields.

The caller owns topology.  Euclidean chord distances are used only inside a
caller-certified local patch.  Partition-of-unity support is determined only
from caller-supplied topology distances.  This module does not construct
patches, incidence, KNN neighborhoods, overlap, fallbacks, or regularization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


SUPPORTED_DTYPES = (torch.float32, torch.float64)
DEFAULT_MAX_CONDITION_NUMBER_FLOAT32 = 1.0e6
DEFAULT_MAX_CONDITION_NUMBER_FLOAT64 = 1.0e12


class RBFAlgebraError(RuntimeError):
    """Base error for an invalid or unsolvable local RBF algebra problem."""


class SingularRBFSystemError(RBFAlgebraError):
    """The constant-augmented local system is rank deficient."""


class IllConditionedRBFSystemError(RBFAlgebraError):
    """The local system exceeds the declared condition-number limit."""


class PartitionCoverageError(RBFAlgebraError):
    """At least one query has no positive partition-of-unity coverage."""


@dataclass(frozen=True)
class AugmentedSystemReport:
    source_count: int
    matrix_size: int
    rank: int
    condition_number: float
    max_condition_number: float
    full_rank: bool
    within_condition_limit: bool


@dataclass(frozen=True)
class LocalRBFSolution:
    """RBF coefficients and the one constant polynomial coefficient."""

    coefficients: torch.Tensor
    constant_term: torch.Tensor
    augmented_system: torch.Tensor
    report: AugmentedSystemReport


def _require_float_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"{name} must use torch.float32 or torch.float64")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _require_same_dtype_device(
    value: torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> None:
    if value.dtype != reference.dtype:
        raise TypeError(
            f"{name} dtype {value.dtype} does not match reference dtype {reference.dtype}"
        )
    if value.device != reference.device:
        raise ValueError(
            f"{name} device {value.device} does not match reference device {reference.device}"
        )


def _require_positions(value: object, name: str) -> torch.Tensor:
    positions = _require_float_tensor(value, name)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")
    if positions.shape[0] <= 0:
        raise ValueError(f"{name} must contain at least one position")
    return positions


def _require_radius(
    topology_radius: object,
    reference: torch.Tensor,
) -> torch.Tensor:
    radius = _require_float_tensor(topology_radius, "topology_radius")
    if radius.ndim != 0:
        raise ValueError("topology_radius must be a scalar tensor")
    _require_same_dtype_device(radius, reference, "topology_radius")
    if not bool((radius > 0.0).item()):
        raise ValueError("topology_radius must be positive")
    return radius


def _require_topology_radii(
    topology_radii: object,
    reference: torch.Tensor,
    patch_count: int,
) -> torch.Tensor:
    radii = _require_float_tensor(topology_radii, "topology_radii")
    if radii.ndim != 1 or radii.shape[0] != patch_count:
        raise ValueError(
            f"topology_radii must have shape [P] matching P={patch_count}; "
            "scalar broadcast is not supported"
        )
    _require_same_dtype_device(radii, reference, "topology_radii")
    if bool((radii <= 0.0).any()):
        raise ValueError("every topology_radii entry must be positive")
    return radii


def _default_condition_limit(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return DEFAULT_MAX_CONDITION_NUMBER_FLOAT32
    if dtype == torch.float64:
        return DEFAULT_MAX_CONDITION_NUMBER_FLOAT64
    raise TypeError(f"unsupported dtype for condition limit: {dtype}")


def _resolve_condition_limit(
    dtype: torch.dtype,
    max_condition_number: float | None,
) -> float:
    limit = (
        _default_condition_limit(dtype)
        if max_condition_number is None
        else float(max_condition_number)
    )
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_condition_number must be finite and positive")
    return limit


def _matrix_tolerance(dtype: torch.dtype) -> float:
    return 1.0e-5 if dtype == torch.float32 else 1.0e-12


def _sum_tolerance(dtype: torch.dtype) -> float:
    return 1.0e-5 if dtype == torch.float32 else 1.0e-10


def wendland_c2(normalized_distance: torch.Tensor) -> torch.Tensor:
    """Evaluate ``(1-t)^4_+ (4t+1)`` with exact zero for ``t >= 1``."""

    distance = _require_float_tensor(normalized_distance, "normalized_distance")
    if bool((distance < 0.0).any()):
        raise ValueError("normalized_distance must be nonnegative")
    inside = distance < 1.0
    safe = torch.where(inside, distance, torch.zeros_like(distance))
    profile = (1.0 - safe).pow(4) * (4.0 * safe + 1.0)
    return torch.where(inside, profile, torch.zeros_like(profile))


def local_kernel_values(
    query_positions: torch.Tensor,
    source_positions: torch.Tensor,
    topology_radius: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the local chord kernel inside one topology-certified patch.

    The caller guarantees that ``source_positions`` belong to one radius-R
    topology patch.  This function uses only Euclidean chord distance and the
    mandated scale ``2R``.
    """

    queries = _require_positions(query_positions, "query_positions")
    sources = _require_positions(source_positions, "source_positions")
    _require_same_dtype_device(queries, sources, "query_positions")
    radius = _require_radius(topology_radius, sources)
    chord_distance = torch.linalg.vector_norm(
        queries[:, None, :] - sources[None, :, :],
        dim=-1,
    )
    return wendland_c2(chord_distance / (2.0 * radius))


def validate_augmented_system(
    augmented_system: torch.Tensor,
    *,
    max_condition_number: float | None = None,
) -> AugmentedSystemReport:
    """Validate constant augmentation, rank, and numerical conditioning."""

    system = _require_float_tensor(augmented_system, "augmented_system")
    if system.ndim != 2 or system.shape[0] != system.shape[1]:
        raise ValueError("augmented_system must be a square rank-2 tensor")
    if system.shape[0] < 2:
        raise ValueError("augmented_system must contain at least one source and one constant")
    source_count = int(system.shape[0]) - 1
    if not bool(torch.equal(system[:-1, -1], torch.ones_like(system[:-1, -1]))):
        raise ValueError("augmented_system last column must be the constant-one term")
    if not bool(torch.equal(system[-1, :-1], torch.ones_like(system[-1, :-1]))):
        raise ValueError("augmented_system last row must be the constant-one term")
    if float(system[-1, -1].detach()) != 0.0:
        raise ValueError("augmented_system polynomial corner must be exactly zero")
    tolerance = _matrix_tolerance(system.dtype)
    if not bool(
        torch.allclose(
            system,
            system.transpose(0, 1),
            rtol=0.0,
            atol=tolerance,
        )
    ):
        raise ValueError("augmented_system must be symmetric")
    kernel = system[:-1, :-1]
    if bool((kernel < -tolerance).any()) or bool((kernel > 1.0 + tolerance).any()):
        raise ValueError("augmented_system kernel block must lie in [0, 1]")
    if not bool(
        torch.allclose(
            torch.diagonal(kernel),
            torch.ones((source_count,), dtype=system.dtype, device=system.device),
            rtol=0.0,
            atol=tolerance,
        )
    ):
        raise ValueError("augmented_system kernel diagonal must equal one")

    limit = _resolve_condition_limit(system.dtype, max_condition_number)
    detached = system.detach()
    try:
        rank = int(torch.linalg.matrix_rank(detached).item())
        condition = float(torch.linalg.cond(detached).item())
    except torch.OutOfMemoryError:
        raise
    except RuntimeError:
        # Backend, device, and synchronization failures are execution errors,
        # not evidence that this mathematical system is singular.
        raise
    matrix_size = int(system.shape[0])
    if rank != matrix_size:
        raise SingularRBFSystemError(
            f"augmented_system is singular: rank {rank} != {matrix_size}"
        )
    if not math.isfinite(condition):
        raise SingularRBFSystemError("augmented_system condition number is nonfinite")
    if condition > limit:
        raise IllConditionedRBFSystemError(
            f"augmented_system condition number {condition:.17g} exceeds {limit:.17g}"
        )
    return AugmentedSystemReport(
        source_count=source_count,
        matrix_size=matrix_size,
        rank=rank,
        condition_number=condition,
        max_condition_number=limit,
        full_rank=True,
        within_condition_limit=True,
    )


def _assemble_augmented_system(
    source_positions: torch.Tensor,
    topology_radius: torch.Tensor,
) -> torch.Tensor:
    kernel = local_kernel_values(
        source_positions,
        source_positions,
        topology_radius,
    )
    ones = torch.ones(
        (source_positions.shape[0], 1),
        dtype=source_positions.dtype,
        device=source_positions.device,
    )
    top = torch.cat((kernel, ones), dim=1)
    bottom = torch.cat(
        (
            ones.transpose(0, 1),
            torch.zeros(
                (1, 1),
                dtype=source_positions.dtype,
                device=source_positions.device,
            ),
        ),
        dim=1,
    )
    return torch.cat((top, bottom), dim=0)


def _validate_system_matches_patch(
    augmented_system: torch.Tensor,
    source_positions: torch.Tensor,
    topology_radius: torch.Tensor,
    *,
    max_condition_number: float | None = None,
) -> AugmentedSystemReport:
    system = _require_float_tensor(augmented_system, "augmented_system")
    _require_same_dtype_device(system, source_positions, "augmented_system")
    report = validate_augmented_system(
        system,
        max_condition_number=max_condition_number,
    )
    if report.source_count != source_positions.shape[0]:
        raise ValueError("augmented_system source count does not match source_positions")
    expected = _assemble_augmented_system(source_positions, topology_radius)
    if not bool(
        torch.allclose(
            system,
            expected,
            rtol=0.0,
            atol=_matrix_tolerance(system.dtype),
        )
    ):
        raise ValueError(
            "augmented_system does not match supplied source_positions and topology_radius"
        )
    return report


def build_augmented_system(
    source_positions: torch.Tensor,
    topology_radius: torch.Tensor,
    *,
    max_condition_number: float | None = None,
) -> torch.Tensor:
    """Build and strictly validate ``[[A, 1], [1^T, 0]]`` for one patch."""

    sources = _require_positions(source_positions, "source_positions")
    radius = _require_radius(topology_radius, sources)
    system = _assemble_augmented_system(sources, radius)
    validate_augmented_system(
        system,
        max_condition_number=max_condition_number,
    )
    return system


def solve_augmented_system(
    augmented_system: torch.Tensor,
    source_values: torch.Tensor,
    *,
    max_condition_number: float | None = None,
) -> LocalRBFSolution:
    """Solve scalar, multi-channel, or batched multi-channel source values.

    Supported value shapes are ``[N]``, ``[N, C]``, and ``[..., N, C]``.
    The source axis is the second-to-last axis whenever a channel axis exists.
    """

    system = _require_float_tensor(augmented_system, "augmented_system")
    report = validate_augmented_system(
        system,
        max_condition_number=max_condition_number,
    )
    values = _require_float_tensor(source_values, "source_values")
    _require_same_dtype_device(values, system, "source_values")
    source_count = report.source_count

    scalar = values.ndim == 1
    if scalar:
        if values.shape[0] != source_count:
            raise ValueError("scalar source_values must have shape [N]")
        flat_values = values.reshape(source_count, 1)
        batch_shape: tuple[int, ...] = ()
        channel_count = 1
    else:
        if values.ndim < 2 or values.shape[-2] != source_count:
            raise ValueError("multi-channel source_values must have shape [..., N, C]")
        if values.shape[-1] <= 0:
            raise ValueError("source_values channel count must be positive")
        batch_shape = tuple(int(value) for value in values.shape[:-2])
        channel_count = int(values.shape[-1])
        flat_values = values.movedim(-2, 0).reshape(source_count, -1)

    rhs = torch.cat(
        (
            flat_values,
            torch.zeros((1, flat_values.shape[1]), dtype=values.dtype, device=values.device),
        ),
        dim=0,
    )
    try:
        solved = torch.linalg.solve(system, rhs)
    except torch.OutOfMemoryError:
        raise
    except RuntimeError as error:
        if "singular" not in str(error).lower() and "not invertible" not in str(error).lower():
            raise
        raise RBFAlgebraError("augmented_system solve failed") from error
    if not bool(torch.isfinite(solved).all()):
        raise RBFAlgebraError("augmented_system solve produced nonfinite coefficients")

    if scalar:
        coefficients = solved[:-1, 0]
        constant_term = solved[-1, 0]
    else:
        coefficients = solved[:-1].reshape(
            (source_count, *batch_shape, channel_count)
        ).movedim(0, -2)
        constant_term = solved[-1].reshape((*batch_shape, channel_count))
    return LocalRBFSolution(
        coefficients=coefficients,
        constant_term=constant_term,
        augmented_system=system,
        report=report,
    )


def evaluate_local_interpolant(
    query_positions: torch.Tensor,
    source_positions: torch.Tensor,
    topology_radius: torch.Tensor,
    solution: LocalRBFSolution,
) -> torch.Tensor:
    """Evaluate a solved local interpolant at arbitrary 3D query positions."""

    queries = _require_positions(query_positions, "query_positions")
    sources = _require_positions(source_positions, "source_positions")
    _require_same_dtype_device(queries, sources, "query_positions")
    radius = _require_radius(topology_radius, sources)
    if not isinstance(solution, LocalRBFSolution):
        raise TypeError("solution must be a LocalRBFSolution")
    bound_system = _require_float_tensor(
        solution.augmented_system,
        "solution.augmented_system",
    )
    _validate_system_matches_patch(
        bound_system,
        sources,
        radius,
        max_condition_number=solution.report.max_condition_number,
    )
    coefficients = _require_float_tensor(solution.coefficients, "solution.coefficients")
    constant_term = _require_float_tensor(solution.constant_term, "solution.constant_term")
    _require_same_dtype_device(coefficients, sources, "solution.coefficients")
    _require_same_dtype_device(constant_term, sources, "solution.constant_term")
    source_count = int(sources.shape[0])
    if solution.report.source_count != source_count:
        raise ValueError("solution source count does not match source_positions")
    kernel = local_kernel_values(queries, sources, radius)
    if coefficients.ndim == 1:
        if coefficients.shape != (source_count,) or constant_term.ndim != 0:
            raise ValueError("scalar solution coefficient shapes are invalid")
        result = kernel @ coefficients + constant_term
    else:
        if coefficients.shape[-2] != source_count:
            raise ValueError("solution coefficient source axis does not match source_positions")
        expected_constant_shape = coefficients.shape[:-2] + coefficients.shape[-1:]
        if constant_term.shape != expected_constant_shape:
            raise ValueError("solution constant-term shape does not match coefficients")
        result = torch.einsum("qn,...nc->...qc", kernel, coefficients)
        result = result + constant_term.unsqueeze(-2)
    if not bool(torch.isfinite(result).all()):
        raise RBFAlgebraError("local interpolant evaluation produced nonfinite values")
    return result


def local_cardinal_weights(
    query_positions: torch.Tensor,
    source_positions: torch.Tensor,
    topology_radius: torch.Tensor,
    *,
    augmented_system: torch.Tensor | None = None,
    max_condition_number: float | None = None,
) -> torch.Tensor:
    """Return linear weights mapping local source values to query values."""

    queries = _require_positions(query_positions, "query_positions")
    sources = _require_positions(source_positions, "source_positions")
    _require_same_dtype_device(queries, sources, "query_positions")
    radius = _require_radius(topology_radius, sources)
    if augmented_system is None:
        system = build_augmented_system(
            sources,
            radius,
            max_condition_number=max_condition_number,
        )
    else:
        system = _require_float_tensor(augmented_system, "augmented_system")
        _validate_system_matches_patch(
            system,
            sources,
            radius,
            max_condition_number=max_condition_number,
        )
    kernel = local_kernel_values(queries, sources, radius)
    evaluation_rows = torch.cat(
        (
            kernel,
            torch.ones((queries.shape[0], 1), dtype=queries.dtype, device=queries.device),
        ),
        dim=1,
    )
    try:
        dual = torch.linalg.solve(
            system.transpose(0, 1),
            evaluation_rows.transpose(0, 1),
        ).transpose(0, 1)
    except torch.OutOfMemoryError:
        raise
    except RuntimeError as error:
        if "singular" not in str(error).lower() and "not invertible" not in str(error).lower():
            raise
        raise RBFAlgebraError("cardinal weight solve failed") from error
    weights = dual[:, :-1]
    if not bool(torch.isfinite(weights).all()):
        raise RBFAlgebraError("cardinal weights are nonfinite")
    row_sum = weights.sum(dim=1)
    if not bool(
        torch.allclose(
            row_sum,
            torch.ones_like(row_sum),
            rtol=0.0,
            atol=_sum_tolerance(weights.dtype),
        )
    ):
        raise RBFAlgebraError("constant augmentation failed to produce cardinal weight sum one")
    return weights


def raw_partition_of_unity_weights(
    topology_distances: torch.Tensor,
    topology_radii: torch.Tensor,
) -> torch.Tensor:
    """Compute raw PU weights with one caller-supplied radius per patch.

    ``topology_distances`` has shape ``[Q, P]`` and ``topology_radii`` must
    have shape ``[P]``.  Scalar radius broadcast is deliberately unsupported.
    """

    distances = _require_float_tensor(topology_distances, "topology_distances")
    if distances.ndim != 2 or distances.shape[1] <= 0:
        raise ValueError("topology_distances must have shape [Q, P] with P > 0")
    if bool((distances < 0.0).any()):
        raise ValueError("topology_distances must be nonnegative")
    radii = _require_topology_radii(
        topology_radii,
        distances,
        int(distances.shape[1]),
    )
    return wendland_c2(distances / radii.unsqueeze(0))


def normalize_partition_of_unity_weights(raw_weights: torch.Tensor) -> torch.Tensor:
    """Normalize nonnegative raw patch weights and reject uncovered queries."""

    weights = _require_float_tensor(raw_weights, "raw_weights")
    if weights.ndim != 2 or weights.shape[1] <= 0:
        raise ValueError("raw_weights must have shape [Q, P] with P > 0")
    if bool((weights < 0.0).any()):
        raise ValueError("raw_weights must be nonnegative")
    denominator = weights.sum(dim=1, keepdim=True)
    if not bool(torch.isfinite(denominator).all()):
        raise PartitionCoverageError("partition-of-unity denominator is nonfinite")
    uncovered = denominator <= 0.0
    if bool(uncovered.any()):
        rows = torch.nonzero(uncovered[:, 0], as_tuple=False).reshape(-1).tolist()
        raise PartitionCoverageError(
            f"partition-of-unity queries have zero denominator: {rows}"
        )
    normalized = weights / denominator
    if not bool(torch.isfinite(normalized).all()):
        raise PartitionCoverageError("normalized partition-of-unity weights are nonfinite")
    return normalized


def blend_partition_of_unity(
    local_patch_values: torch.Tensor,
    raw_weights: torch.Tensor,
) -> torch.Tensor:
    """Blend already-evaluated local patch values using normalized PU weights."""

    values = _require_float_tensor(local_patch_values, "local_patch_values")
    weights = _require_float_tensor(raw_weights, "raw_weights")
    _require_same_dtype_device(weights, values, "raw_weights")
    if values.ndim < 2:
        raise ValueError("local_patch_values must have shape [Q, P, ...]")
    if weights.ndim != 2 or tuple(values.shape[:2]) != tuple(weights.shape):
        raise ValueError(
            "raw_weights must have shape [Q, P] matching local_patch_values"
        )
    normalized = normalize_partition_of_unity_weights(weights)
    trailing = (1,) * (values.ndim - 2)
    blended = (
        values
        * normalized.reshape(
            normalized.shape[0],
            normalized.shape[1],
            *trailing,
        )
    ).sum(dim=1)
    if not bool(torch.isfinite(blended).all()):
        raise RBFAlgebraError("partition-of-unity blend produced nonfinite values")
    return blended


__all__ = [
    "AugmentedSystemReport",
    "DEFAULT_MAX_CONDITION_NUMBER_FLOAT32",
    "DEFAULT_MAX_CONDITION_NUMBER_FLOAT64",
    "IllConditionedRBFSystemError",
    "LocalRBFSolution",
    "PartitionCoverageError",
    "RBFAlgebraError",
    "SUPPORTED_DTYPES",
    "SingularRBFSystemError",
    "blend_partition_of_unity",
    "build_augmented_system",
    "evaluate_local_interpolant",
    "local_cardinal_weights",
    "local_kernel_values",
    "normalize_partition_of_unity_weights",
    "raw_partition_of_unity_weights",
    "solve_augmented_system",
    "validate_augmented_system",
    "wendland_c2",
]

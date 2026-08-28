"""Zero-centered render-root geometry residuals for multi-level grooming."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

EPS = 1.0e-8


@dataclass(frozen=True)
class DecodedGeometryResiduals:
    """Dimensionless residual coordinates around the interpolated guide field."""

    length: torch.Tensor
    root_width_log_ratio: torch.Tensor
    tip_width_logit_delta: torch.Tensor
    width_taper_log_ratio: torch.Tensor
    curl_radius_log_ratio: torch.Tensor
    child_radius_log_ratio: torch.Tensor
    clump_strength: torch.Tensor
    direction_local: torch.Tensor


@dataclass(frozen=True)
class GuideSupportGaugeTerms:
    """Population-stable primary-guide support-gauge terms."""

    total: torch.Tensor
    length_collapse: torch.Tensor
    slenderness_expansion: torch.Tensor


class RenderGeometryResidualField(nn.Module):
    """Late render-root geometry expressed as zero-centered residuals.

    The field contains no absolute grooming endpoint. A zero tensor means that
    the render root exactly follows its interpolated guide field. Semantic
    bounded controls use their intrinsic domains; positive physical fields
    are composed separately in scale-relative coordinates.
    """

    SCALAR_NAMES = (
        "length",
        "root_width",
        "tip_width_ratio",
        "width_taper",
        "curl_radius_ratio",
        "child_radius",
        "clump_strength",
    )

    def __init__(self, root_count: int, device: torch.device | str | None = None) -> None:
        super().__init__()
        if int(root_count) <= 0:
            raise ValueError("root_count must be positive")
        self.root_count = int(root_count)
        dev = torch.device(device) if device is not None else None
        for name in self.SCALAR_NAMES:
            self.register_parameter(
                f"{name}_raw",
                nn.Parameter(torch.zeros((self.root_count, 1), device=dev)),
            )
        self.direction_local_raw = nn.Parameter(
            torch.zeros((self.root_count, 3), device=dev)
        )

    def decode(self) -> DecodedGeometryResiduals:
        return DecodedGeometryResiduals(
            length=torch.tanh(self.length_raw),
            root_width_log_ratio=torch.asinh(self.root_width_raw),
            tip_width_logit_delta=torch.asinh(self.tip_width_ratio_raw),
            width_taper_log_ratio=torch.asinh(self.width_taper_raw),
            curl_radius_log_ratio=torch.asinh(self.curl_radius_ratio_raw),
            child_radius_log_ratio=torch.asinh(self.child_radius_raw),
            clump_strength=torch.tanh(self.clump_strength_raw),
            direction_local=torch.tanh(self.direction_local_raw),
        )

    @staticmethod
    def scalar_domain_delta(
        normalized_delta: torch.Tensor,
        bounds: tuple[float, float],
    ) -> torch.Tensor:
        return normalized_delta * float(bounds[1] - bounds[0])


def direction_to_local_components(
    direction: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
) -> torch.Tensor:
    """Encode a world-space direction in a root's orthonormal surface frame."""

    direction = F.normalize(direction, dim=-1, eps=EPS)
    return torch.stack(
        [
            (direction * tangents).sum(dim=-1),
            (direction * bitangents).sum(dim=-1),
            (direction * normals).sum(dim=-1),
        ],
        dim=-1,
    )


def vector_to_local_components(
    vector: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
) -> torch.Tensor:
    """Encode a world-space vector without discarding its magnitude."""

    return torch.stack(
        [
            (vector * tangents).sum(dim=-1),
            (vector * bitangents).sum(dim=-1),
            (vector * normals).sum(dim=-1),
        ],
        dim=-1,
    )


def local_components_to_world(
    components: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    *,
    normalize: bool,
) -> torch.Tensor:
    """Decode local surface-frame components into world-space vectors."""

    world = (
        components[..., 0:1] * tangents
        + components[..., 1:2] * bitangents
        + components[..., 2:3] * normals
    )
    if normalize:
        world = F.normalize(world, dim=-1, eps=EPS)
    return world


def apply_direction_residual(
    guide_direction: torch.Tensor,
    residual_local: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """Apply a bounded local 3D perturbation to a guide direction."""

    residual_world = local_components_to_world(
        residual_local,
        normals,
        tangents,
        bitangents,
        normalize=False,
    )
    return F.normalize(guide_direction + residual_world * scale, dim=-1, eps=EPS)


def apply_log_ratio_residual(
    guide_value: torch.Tensor,
    normalized_log_delta: torch.Tensor,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """Apply a zero-centered, scale-relative residual to a positive field.

    ``normalized_log_delta == 0`` is exactly the guide value. The result stays
    positive without using an absolute decoder minimum or maximum, and the
    same residual produces the same ratio for short and long guide values.
    """

    return guide_value * torch.exp(normalized_log_delta * scale)


def apply_asinh_log_ratio_residual(
    guide_value: torch.Tensor,
    log_coordinate: torch.Tensor,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """Apply an unbounded relative residual without exponential tail growth.

    Near zero, ``asinh(x)`` has unit slope, so this decoder has the same local
    optimization behavior as a raw log-ratio coordinate. Far from zero its
    log-ratio grows only logarithmically. The effective physical field
    therefore stays positive and unbounded while isolated residual coordinates
    cannot produce an exponential tail.
    """

    return guide_value * torch.exp(torch.asinh(log_coordinate) * scale)


def apply_asinh_logit_residual(
    guide_ratio: torch.Tensor,
    logit_coordinate: torch.Tensor,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """Apply a zero-centered residual to a semantic ratio in ``[0, 1]``.

    The residual acts in log-odds space, so zero returns the guide value
    exactly and no animal-specific lower or upper endpoint is introduced.
    ``asinh`` preserves a linear neighborhood around zero while moderating
    isolated raw-coordinate tails.
    """

    eps = torch.as_tensor(
        torch.finfo(guide_ratio.dtype).eps,
        device=guide_ratio.device,
        dtype=guide_ratio.dtype,
    )
    guide_logit = torch.logit(guide_ratio.clamp(eps, 1.0 - eps))
    return torch.sigmoid(guide_logit + torch.asinh(logit_coordinate) * scale)


def encode_asinh_logit_residual(
    value: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Encode a semantic ratio relative to a local reference."""

    eps = torch.as_tensor(
        torch.finfo(value.dtype).eps,
        device=value.device,
        dtype=value.dtype,
    )
    value_logit = torch.logit(value.clamp(eps, 1.0 - eps))
    reference_logit = torch.logit(reference.clamp(eps, 1.0 - eps))
    return torch.sinh(value_logit - reference_logit)


def length_residual_prior_coordinate(
    raw_coordinate: torch.Tensor,
    parameterization: str,
    mode: str,
) -> torch.Tensor:
    """Return the coordinate regularized around the interpolated guide field.

    ``decoded`` preserves the historical bounded residual prior.
    ``natural_log_ratio`` acts on the logarithmic ratio used by the positive
    length composition. ``raw`` acts on the decoder's zero-centered residual
    coordinate, which keeps a non-vanishing prior gradient when an asinh
    coordinate moves into its tail. None of these modes clamps the resulting
    physical length.
    """

    if mode == "decoded":
        return torch.tanh(raw_coordinate)
    if mode == "raw":
        if parameterization not in {
            "zero_centered_log_length_residual",
            "zero_centered_unbounded_log_length_residual",
            "zero_centered_asinh_log_length_residual",
        }:
            raise ValueError(
                "raw length prior requires a zero-centered length residual, "
                f"got {parameterization}"
            )
        return raw_coordinate
    if mode != "natural_log_ratio":
        raise ValueError(f"unsupported render length prior coordinate: {mode}")
    if parameterization == "zero_centered_asinh_log_length_residual":
        return torch.asinh(raw_coordinate)
    if parameterization == "zero_centered_unbounded_log_length_residual":
        return raw_coordinate
    if parameterization == "zero_centered_log_length_residual":
        return torch.tanh(raw_coordinate)
    raise ValueError(
        "natural_log_ratio prior requires a log-ratio length parameterization, "
        f"got {parameterization}"
    )


def fourth_moment_norm(
    value: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a differentiable tail-sensitive norm without a value threshold.

    A mean L1/L2 reduction makes a fixed number of bad roots progressively
    cheaper as densification increases the total root count.  The fourth-root
    fourth moment keeps ordinary residuals smooth while preserving useful
    gradient on sparse, extreme residuals.  It does not clamp or prescribe a
    physical grooming value.
    """

    if value.numel() == 0:
        return value.sum() * 0.0
    magnitude4 = value.abs().pow(4)
    if weight is None:
        moment = magnitude4.mean()
    else:
        w = weight.detach().to(device=value.device, dtype=value.dtype)
        while w.ndim < value.ndim:
            w = w.unsqueeze(-1)
        w = torch.broadcast_to(w, value.shape)
        moment = (magnitude4 * w).sum() / w.sum().clamp_min(EPS)
    tiny = torch.as_tensor(
        torch.finfo(value.dtype).tiny,
        device=value.device,
        dtype=value.dtype,
    )
    return (moment + tiny).pow(0.25) - tiny.pow(0.25)


def guide_support_gauge(
    guide_length_raw: torch.Tensor,
    guide_root_width_raw: torch.Tensor,
    guide_clean_flow_length_confidence: torch.Tensor,
    source_area_weights: torch.Tensor | None = None,
) -> GuideSupportGaugeTerms:
    """Penalize only primary-guide support failures relative to clean flow.

    The positive reference-relative coordinates are ``asinh(raw)``. Shortening
    below the stored length reference and width growth beyond length growth are
    measured separately with the population-stable fourth moment. Confidence
    contributes a continuous floor of ``0.25`` and is optionally multiplied by
    intrinsic source-area quadrature weights.
    """

    length_raw = guide_length_raw.reshape(-1)
    width_raw = guide_root_width_raw.reshape(-1)
    confidence = guide_clean_flow_length_confidence.reshape(-1)
    if length_raw.shape != width_raw.shape:
        raise ValueError("guide length and root-width coordinates must have equal size")
    if confidence.shape != length_raw.shape:
        raise ValueError("guide clean-flow confidence must match guide coordinates")
    confidence = confidence.to(device=length_raw.device, dtype=length_raw.dtype)

    trust = 0.25 + 0.75 * confidence.clamp(0.0, 1.0)
    if source_area_weights is not None:
        area = source_area_weights.reshape(-1)
        if area.shape != length_raw.shape:
            raise ValueError("guide source-area weights must match guide coordinates")
        trust = trust * area.to(device=trust.device, dtype=trust.dtype)

    log_length_ratio = torch.asinh(length_raw)
    log_width_ratio = torch.asinh(width_raw)
    length_collapse = F.relu(-log_length_ratio)
    slenderness_expansion = F.relu(log_width_ratio - log_length_ratio)
    length_term = fourth_moment_norm(length_collapse, trust)
    slenderness_term = fourth_moment_norm(slenderness_expansion, trust)
    return GuideSupportGaugeTerms(
        total=length_term + slenderness_term,
        length_collapse=length_term,
        slenderness_expansion=slenderness_term,
    )


def population_stable_residual_norm(
    value: torch.Tensor,
    unlock_multiplier: float,
) -> torch.Tensor:
    """Transition from the accepted mean prior to a population-stable norm.

    The same multiplier that unlocks a render residual controls this handoff.
    Before the residual contributes to geometry, the loss is exactly the
    accepted mean-L1 prior. Once fully unlocked, it becomes the fourth-moment
    norm that does not dilute sparse failures as quickly under densification.
    No iteration, value threshold, physical bound, or region rule is encoded
    here.
    """

    if value.numel() == 0:
        return value.sum() * 0.0
    transition = max(0.0, min(1.0, float(unlock_multiplier)))
    mean_l1 = value.abs().mean()
    stable = fourth_moment_norm(value)
    return torch.lerp(mean_l1, stable, transition)


def tail_concentration_residual_loss(
    value: torch.Tensor,
    unlock_multiplier: float,
) -> torch.Tensor:
    """Keep the accepted mean prior and penalize only residual concentration.

    ``L4 - L2`` is zero when every root carries the same residual magnitude and
    positive when the energy is concentrated in a sparse tail. Adding this term
    to the historical mean-L1 prior discourages isolated runaway roots without
    imposing a physical length bound or suppressing a coherent region whose
    roots jointly need a larger residual. The existing geometry unlock
    multiplier controls the handoff, so no second iteration schedule is added.
    """

    if value.numel() == 0:
        return value.sum() * 0.0
    transition = max(0.0, min(1.0, float(unlock_multiplier)))
    magnitude = value.abs()
    mean_l1 = magnitude.mean()
    tiny = torch.as_tensor(
        torch.finfo(value.dtype).tiny,
        device=value.device,
        dtype=value.dtype,
    )
    second_moment = (magnitude.square().mean() + tiny).sqrt() - tiny.sqrt()
    fourth_moment = fourth_moment_norm(value)
    tail_concentration = (fourth_moment - second_moment).clamp_min(0.0)
    return mean_l1 + transition * tail_concentration

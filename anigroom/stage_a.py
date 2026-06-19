from dataclasses import dataclass

import torch


EPS = 1e-8


def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=EPS)


@dataclass
class GroomOutput:
    curves: torch.Tensor
    width: torch.Tensor
    color: torch.Tensor
    alpha: torch.Tensor
    coverage: torch.Tensor
    density: torch.Tensor
    length: torch.Tensor
    root_width: torch.Tensor
    tip_width: torch.Tensor


def make_default_stage_a_params(n: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "coverage_logit": torch.full((n, 1), 3.0, device=device, requires_grad=True),
        "density_logit": torch.full((n, 1), 2.2, device=device, requires_grad=True),
        "length_logit": torch.full((n, 1), -0.25, device=device, requires_grad=True),
        "root_width_raw": torch.full((n, 1), -5.5, device=device, requires_grad=True),
        "tip_width_raw": torch.full((n, 1), -6.8, device=device, requires_grad=True),
        "flow_x": torch.full((n, 1), 0.55, device=device, requires_grad=True),
        "flow_y": torch.full((n, 1), 0.04, device=device, requires_grad=True),
        "lift": torch.full((n, 1), -0.70, device=device, requires_grad=True),
        "sag": torch.full((n, 1), -1.30, device=device, requires_grad=True),
        "bend": torch.full((n, 1), -0.40, device=device, requires_grad=True),
        "stiffness_logit": torch.full((n, 1), 1.15, device=device, requires_grad=True),
        "root_rgb": torch.tensor([[0.96, 0.94, 0.88]], device=device).repeat(n, 1).requires_grad_(True),
        "tip_rgb": torch.tensor([[0.88, 0.87, 0.80]], device=device).repeat(n, 1).requires_grad_(True),
        "darkness": torch.full((n, 1), -1.7, device=device, requires_grad=True),
    }


def clone_params(params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}


def apply_procedural_white_tiger_maps(
    roots: torch.Tensor,
    params: dict[str, torch.Tensor],
    stripe_strength: float = 2.8,
) -> dict[str, torch.Tensor]:
    out = clone_params(params)
    x = roots[:, [0]]
    z = roots[:, [2]]
    y = roots[:, [1]]
    stripes = (
        torch.sin(28.0 * z + 9.0 * x)
        + 0.55 * torch.sin(17.0 * z - 13.0 * y)
        + 0.35 * torch.sin(41.0 * x)
    )
    stripe_mask = torch.sigmoid(3.0 * (stripes - 0.65))
    out["darkness"] = (out["darkness"].detach() + stripe_strength * stripe_mask).requires_grad_(True)
    out["length_logit"] = (out["length_logit"].detach() + 0.45 * torch.sin(8.0 * z)).requires_grad_(True)
    out["flow_x"] = (out["flow_x"].detach() + 0.20 * torch.sin(5.5 * z)).requires_grad_(True)
    out["flow_y"] = (out["flow_y"].detach() + 0.15 * torch.cos(6.0 * x)).requires_grad_(True)
    return out


def generate_stage_a_curves(
    roots: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    params: dict[str, torch.Tensor],
    samples: int,
    length_range: tuple[float, float] = (0.012, 0.105),
) -> GroomOutput:
    s = torch.linspace(0.0, 1.0, samples, device=roots.device)[None, :, None]
    coverage = torch.sigmoid(params["coverage_logit"])
    density = torch.sigmoid(params["density_logit"])
    alpha = coverage * density
    lmin, lmax = length_range
    length = lmin + (lmax - lmin) * torch.sigmoid(params["length_logit"])
    root_width = torch.nn.functional.softplus(params["root_width_raw"]) + 8e-5
    tip_width = torch.nn.functional.softplus(params["tip_width_raw"]) + 3e-5
    flow = normalize(params["flow_x"] * tangents + params["flow_y"] * bitangents + 1e-4 * tangents)

    # Root direction should follow the surface normal, but the whole strand
    # should not be a normal spike. A cubic Hermite curve keeps the root tangent
    # normal-anchored and lets brushing/gravity dominate toward the tip.
    down = torch.tensor([0.0, -1.0, 0.0], device=roots.device)[None, :]
    gravity_tangent = down - (normals * down).sum(dim=-1, keepdim=True) * normals
    gravity_tangent = normalize(gravity_tangent + 1e-4 * flow)

    root_normal = 0.22 + 0.08 * torch.sigmoid(params["lift"])
    tip_lift = 0.07 + 0.28 * torch.sigmoid(params["lift"])
    brush = 0.08 + 0.80 * torch.sigmoid(params["bend"])
    sag = 0.02 + 0.55 * torch.sigmoid(params["sag"])
    stiffness = torch.sigmoid(params["stiffness_logit"])
    length_factor = torch.square(length / lmax)
    sag = sag * (1.0 - stiffness) * (0.25 + 1.75 * length_factor)

    p0 = roots
    p1 = roots + length * (tip_lift * normals + brush * flow + sag * gravity_tangent)
    m0 = length * root_normal * normals
    m1 = length * (0.10 * normals + brush * flow + sag * gravity_tangent)
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    curves = h00 * p0[:, None, :] + h10 * m0[:, None, :] + h01 * p1[:, None, :] + h11 * m1[:, None, :]
    width = root_width[:, None, :] * (1.0 - s) + tip_width[:, None, :] * s
    color = params["root_rgb"][:, None, :] * (1.0 - s) + params["tip_rgb"][:, None, :] * s
    dark = torch.sigmoid(params["darkness"])[:, None, :]
    color = torch.clamp(color * (1.0 - 0.9 * dark), 0.0, 1.0)
    return GroomOutput(curves, width, color, alpha[:, None, :], coverage, density, length, root_width, tip_width)


def fixed_gaussian_proxy(groom: GroomOutput, step: int) -> dict[str, torch.Tensor]:
    curves = groom.curves
    starts = torch.arange(0, curves.shape[1] - 1, step, device=curves.device)
    ends = torch.clamp(starts + step, max=curves.shape[1] - 1)
    mean = 0.5 * (curves[:, starts] + curves[:, ends])
    chord = curves[:, ends] - curves[:, starts]
    width = 0.5 * (groom.width[:, starts] + groom.width[:, ends])
    color = 0.5 * (groom.color[:, starts] + groom.color[:, ends])
    alpha = groom.alpha.expand(-1, len(starts), -1)
    return {"mean": mean, "chord": chord, "width": width, "color": color, "alpha": alpha}

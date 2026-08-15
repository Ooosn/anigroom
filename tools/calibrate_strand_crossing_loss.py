from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.collision.strand_crossing import (  # noqa: E402
    GaussianSegmentSnapshot,
    discover_gaussian_segment_crossings,
)
from anigroom.flow import groom_direction_3d  # noqa: E402
from tools.train_white_tiger_stage1 import (  # noqa: E402
    clean_flow_anchor_loss,
    clean_flow_smoothness_loss,
    build_guide_graph_edges,
    effective_groom_graph_smoothness,
    guide_interpolation_regularization_losses,
    guide_root_graph_smoothness,
    load_stage1_checkpoint_model,
    rebuild_graph_edges,
    render_geometry_residual_graph_smoothness,
    root_graph_smoothness,
    smooth_metric_uses_transport,
    strand_crossing_local_shape_named_parameters,
)


def render_parameter_args(config) -> tuple[int | float, ...]:
    return (
        int(config.samples),
        int(config.child_count),
        int(config.min_segments),
        float(config.segment_length_origin),
        float(config.segments_per_unit_length),
        float(config.segments_per_unit_complexity),
        float(config.gaussian_length_overlap),
    )


def geometry_parameters(
    model: torch.nn.Module,
) -> list[tuple[str, torch.nn.Parameter]]:
    selected = [
        (name, parameter)
        for name, parameter in strand_crossing_local_shape_named_parameters(model)
        if parameter.requires_grad
    ]
    if not selected:
        raise RuntimeError(
            "checkpoint model exposes no active local crossing-shape residual"
        )
    return selected


def parameter_domain(name: str) -> str:
    if name == "bary_logits":
        return "render_root_position"
    if name.startswith("guide_"):
        return "primary_guide"
    if name.startswith("secondary_geometry_residual."):
        return "secondary_geometry_residual"
    if name.startswith("render_geometry_residual."):
        return "render_geometry_residual"
    if name.startswith("groom."):
        return "render_groom"
    return "other_geometry"


def gradient_report(
    loss: torch.Tensor,
    named_parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, object]:
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named_parameters],
        allow_unused=True,
    )
    domains: dict[str, dict[str, float | int]] = {}
    total_squared = 0.0
    maximum = 0.0
    nonzero_tensors = 0
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        detached = gradient.detach()
        squared = float(detached.double().square().sum().cpu())
        tensor_maximum = float(detached.abs().max().cpu())
        nonzero = int(torch.count_nonzero(detached).cpu())
        domain = parameter_domain(name)
        record = domains.setdefault(
            domain,
            {
                "squared_l2": 0.0,
                "maximum_absolute": 0.0,
                "nonzero_elements": 0,
                "parameter_tensors": 0,
            },
        )
        record["squared_l2"] = float(record["squared_l2"]) + squared
        record["maximum_absolute"] = max(
            float(record["maximum_absolute"]), tensor_maximum
        )
        record["nonzero_elements"] = int(record["nonzero_elements"]) + nonzero
        record["parameter_tensors"] = int(record["parameter_tensors"]) + 1
        total_squared += squared
        maximum = max(maximum, tensor_maximum)
        nonzero_tensors += int(nonzero > 0)
    for record in domains.values():
        record["l2"] = float(record.pop("squared_l2")) ** 0.5
    return {
        "loss": float(loss.detach().cpu()),
        "l2": total_squared**0.5,
        "maximum_absolute": maximum,
        "nonzero_parameter_tensors": nonzero_tensors,
        "domains": domains,
    }


def weighted_structure_loss(model, config) -> tuple[torch.Tensor, dict[str, float]]:
    graph_edges, _ = rebuild_graph_edges(
        model,
        mode=config.smooth_graph_mode,
        k=config.smooth_graph_k,
    )
    guide_graph_edges, _ = build_guide_graph_edges(
        model,
        mode=config.smooth_graph_mode,
        k=config.guide_interpolation_k,
    )
    _, normals, roots_local = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    if model.secondary_guides_enabled():
        geometry_normals, geometry_tangents, geometry_bitangents = (
            model.tangent_frames_for_face_ids(model.secondary_guide_face_ids)
        )
        geometry_effective_groom = model.secondary_effective_groom()
        geometry_confidence = model.secondary_clean_flow_confidence()
        geometry_graph_edges = model.secondary_surface_smoothing_edges(
            config.secondary_guide_smooth_k
        )
    else:
        geometry_normals = normals
        geometry_tangents = tangents
        geometry_bitangents = bitangents
        geometry_effective_groom = model.apply_guide_controls(
            model.groom.decode(), roots_local
        )
        geometry_confidence = model.root_observation_confidence
        geometry_graph_edges = graph_edges

    smooth = root_graph_smoothness(
        model.groom,
        graph_edges,
        model.root_observation_confidence,
        normals=normals,
        tangents=tangents,
        bitangents=bitangents,
        smooth_field_metric=config.smooth_field_metric,
        include_geometry=not model.uses_zero_centered_geometry(),
        appearance_only=model.secondary_guides_enabled(),
    )
    geometry_residual_smooth = render_geometry_residual_graph_smoothness(
        model,
        geometry_graph_edges,
        geometry_normals,
        geometry_tangents,
        geometry_bitangents,
        geometry_confidence,
    )
    smooth = smooth + float(config.geometry_residual_smooth_scale) * (
        geometry_residual_smooth
    )
    guide_smooth = guide_root_graph_smoothness(
        model,
        guide_graph_edges,
        smooth_field_metric=config.smooth_field_metric,
        guide_length_smooth_mode=config.guide_length_smooth_mode,
        smooth_graph_k=config.guide_interpolation_k,
    )
    effective_smooth = effective_groom_graph_smoothness(
        geometry_effective_groom,
        geometry_graph_edges,
        geometry_normals,
        geometry_tangents,
        geometry_bitangents,
        model.groom.ranges,
        geometry_confidence,
        smooth_field_metric=config.smooth_field_metric,
    )
    guide_prior = guide_interpolation_regularization_losses(model, config)
    clean_direction = groom_direction_3d(
        geometry_effective_groom,
        geometry_normals,
        geometry_tangents,
        geometry_bitangents,
    )
    clean_smooth = clean_flow_smoothness_loss(
        clean_direction,
        geometry_graph_edges,
        geometry_confidence,
        normals=(
            geometry_normals
            if smooth_metric_uses_transport(config.smooth_field_metric)
            else None
        ),
    )
    guide_direction = model.guide_direction_world()
    if guide_direction is None:
        guide_anchor = model.groom.length_raw.sum() * 0.0
    else:
        guide_anchor = clean_flow_anchor_loss(
            guide_direction,
            model.guide_clean_flow_direction_target,
            model.guide_clean_flow_anchor_confidence,
            min_confidence=float(config.clean_flow_anchor_min_confidence),
        )
    root_move = torch.mean((roots_local - model.anchor_local).square())
    weighted = (
        float(config.smooth_weight) * smooth
        + float(config.guide_smooth_weight) * guide_smooth
        + float(config.effective_smooth_weight) * effective_smooth
        + float(config.guide_prior_weight) * guide_prior
        + float(config.clean_flow_guide_anchor_weight) * guide_anchor
        + float(config.clean_flow_3d_smooth_weight) * clean_smooth
        + float(config.root_move_reg_weight) * root_move
    )
    values = {
        "smooth": float(smooth.detach().cpu()),
        "guide_smooth": float(guide_smooth.detach().cpu()),
        "effective_smooth": float(effective_smooth.detach().cpu()),
        "guide_prior": float(guide_prior.detach().cpu()),
        "guide_anchor": float(guide_anchor.detach().cpu()),
        "clean_smooth": float(clean_smooth.detach().cpu()),
        "root_move": float(root_move.detach().cpu()),
    }
    return weighted, values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate differentiable strand-crossing gradients."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-batch", type=int, default=50000)
    parser.add_argument("--exact-pair-batch", type=int, default=250000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("crossing calibration requires CUDA")
    device = torch.device("cuda")
    model, config, checkpoint = load_stage1_checkpoint_model(args.checkpoint, device)
    model.train()

    with torch.no_grad():
        gaussians, _, _, _, _, _, _ = model.render_parameters(
            *render_parameter_args(config),
        )
        snapshot = GaussianSegmentSnapshot.from_tensors(
            means=gaussians.means,
            directions=gaussians.directions,
            scales=gaussians.scales,
            root_indices=gaussians.root_indices,
            segment_indices=gaussians.segment_indices,
            length_overlap=float(config.gaussian_length_overlap),
        )
    del gaussians
    torch.cuda.empty_cache()

    discovery_started = time.perf_counter()
    active, discovery = discover_gaussian_segment_crossings(
        snapshot,
        query_batch=args.query_batch,
        exact_pair_batch=args.exact_pair_batch,
        workers=args.workers,
    )
    discovery["elapsed_seconds"] = float(time.perf_counter() - discovery_started)
    active_torch = active.to_torch(device)
    del snapshot

    named_geometry = geometry_parameters(model)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    crossing_forward_started = time.perf_counter()
    _, _, _, _, _, crossing_loss, crossing_stats = model.render_parameters(
        *render_parameter_args(config),
        strand_crossing_active_set=active_torch,
    )
    torch.cuda.synchronize(device)
    crossing_forward_elapsed = time.perf_counter() - crossing_forward_started
    crossing_forward_peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    crossing_gradients = gradient_report(crossing_loss, named_geometry)
    structure_loss, structure_values = weighted_structure_loss(model, config)
    structure_gradients = gradient_report(structure_loss, named_geometry)

    crossing_l2 = float(crossing_gradients["l2"])
    structure_l2 = float(structure_gradients["l2"])
    equal_l2_weight = structure_l2 / crossing_l2 if crossing_l2 > 0.0 else 0.0
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_iteration": int(checkpoint.get("iteration", 0)),
        "root_count": int(model.face_ids.shape[0]),
        "gaussian_count": int(active.source_segment_count),
        "discovery": discovery,
        "crossing_runtime": {
            **{
                key: (
                    int(value)
                    if isinstance(value, int)
                    else float(value.detach().cpu())
                )
                for key, value in crossing_stats.items()
            },
            "forward_elapsed_seconds": float(crossing_forward_elapsed),
            "forward_peak_allocated_mb": float(crossing_forward_peak_mb),
        },
        "crossing_gradients_unweighted": crossing_gradients,
        "existing_structure_gradients_weighted": structure_gradients,
        "existing_structure_loss_terms_unweighted": structure_values,
        "crossing_weight_for_equal_geometry_gradient_l2": equal_l2_weight,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.collision.sdf import (  # noqa: E402
    cyclic_strand_indices,
    strand_penetration_depth,
)
from anigroom.grooming import build_strands  # noqa: E402
from tools.train_white_tiger_stage1 import (  # noqa: E402
    build_guide_graph_edges,
    clean_flow_anchor_loss,
    clean_flow_smoothness_loss,
    effective_groom_graph_smoothness,
    groom_direction_3d,
    guide_interpolation_regularization_losses,
    guide_root_graph_smoothness,
    load_mesh_no_penetration_field,
    load_stage1_checkpoint_model,
    rebuild_graph_edges,
    render_geometry_residual_graph_smoothness,
    resolve_project_path,
    root_graph_smoothness,
    smooth_metric_uses_transport,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure continuous-strand mesh penetration and its geometry "
            "gradients for a formal Stage1 checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sdf", required=True)
    parser.add_argument(
        "--mesh",
        default="",
        help="Explicit source mesh for SDF identity verification.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--query-root-chunk", type=int, default=16384)
    parser.add_argument("--gradient-root-batch", type=int, default=16384)
    parser.add_argument("--visual-strands", type=int, default=100000)
    parser.add_argument(
        "--visual-color",
        nargs=3,
        type=float,
        default=[0.82, 0.80, 0.72],
    )
    return parser.parse_args()


def summarize(values: torch.Tensor) -> dict[str, float | int]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(
        flat,
        flat.new_tensor([0.50, 0.90, 0.95, 0.99]),
    )
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().cpu()),
        "p50": float(quantiles[0].cpu()),
        "p90": float(quantiles[1].cpu()),
        "p95": float(quantiles[2].cpu()),
        "p99": float(quantiles[3].cpu()),
        "maximum": float(flat.max().cpu()),
    }


def build_checkpoint_strands(model, samples: int, *, mesh_local: bool):
    roots, normals, roots_local = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    groom = model.apply_guide_controls(model.groom.decode(), roots_local)
    if mesh_local:
        groom = replace(
            groom,
            length=groom.length / torch.exp(model.log_scale.detach()),
        )
        roots = roots_local
    strands = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=int(samples),
    )
    return strands


def collect_gradient_norms(model) -> tuple[dict[str, dict[str, float | int]], dict[str, torch.Tensor]]:
    summaries: dict[str, dict[str, float | int]] = {}
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        summaries[name] = {
            "l2": float(torch.linalg.vector_norm(gradient).cpu()),
            "maximum_abs": float(gradient.abs().max().cpu()),
            "element_count": int(parameter.numel()),
        }
        gradients[name] = gradient.clone()
    return summaries, gradients


def weighted_structure_regularization(model, config):
    graph_edges, graph_report = rebuild_graph_edges(
        model,
        mode=config.smooth_graph_mode,
        k=config.smooth_graph_k,
    )
    guide_graph_edges, guide_graph_report = build_guide_graph_edges(
        model,
        mode=config.smooth_graph_mode,
        k=config.guide_interpolation_k,
    )
    if model.secondary_guides_enabled():
        geometry_graph_edges = model.secondary_surface_smoothing_edges(
            config.secondary_guide_smooth_k
        )
        geometry_graph_report = {
            "mode": "secondary_parent_conditioned",
            "root_count": int(model.secondary_guide_points_local.shape[0]),
            "neighbor_count": int(config.secondary_guide_smooth_k),
            "edge_count": int(geometry_graph_edges.shape[0]),
        }
    else:
        geometry_graph_edges = graph_edges
        geometry_graph_report = dict(graph_report)

    _, normals, roots_local = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    if model.secondary_guides_enabled():
        geometry_normals, geometry_tangents, geometry_bitangents = (
            model.tangent_frames_for_face_ids(model.secondary_guide_face_ids)
        )
        geometry_groom = model.secondary_effective_groom()
        geometry_confidence = model.secondary_clean_flow_confidence()
    else:
        geometry_normals = normals
        geometry_tangents = tangents
        geometry_bitangents = bitangents
        geometry_groom = model.apply_guide_controls(
            model.groom.decode(),
            roots_local,
        )
        geometry_confidence = model.root_observation_confidence

    render_smooth = root_graph_smoothness(
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
    render_smooth = (
        render_smooth
        + float(config.geometry_residual_smooth_scale)
        * geometry_residual_smooth
    )
    guide_smooth = guide_root_graph_smoothness(
        model,
        guide_graph_edges,
        smooth_field_metric=config.smooth_field_metric,
        guide_length_smooth_mode=config.guide_length_smooth_mode,
        smooth_graph_k=config.guide_interpolation_k,
    )
    effective_smooth = effective_groom_graph_smoothness(
        geometry_groom,
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
        geometry_groom,
        geometry_normals,
        geometry_tangents,
        geometry_bitangents,
    )
    clean_flow_smooth = clean_flow_smoothness_loss(
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

    components = {
        "render_smooth": render_smooth,
        "guide_smooth": guide_smooth,
        "effective_smooth": effective_smooth,
        "guide_prior": guide_prior,
        "clean_flow_guide_anchor": guide_anchor,
        "clean_flow_3d_smooth": clean_flow_smooth,
        "root_move": root_move,
    }
    weights = {
        "render_smooth": float(config.smooth_weight),
        "guide_smooth": float(config.guide_smooth_weight),
        "effective_smooth": float(config.effective_smooth_weight),
        "guide_prior": float(config.guide_prior_weight),
        "clean_flow_guide_anchor": float(config.clean_flow_guide_anchor_weight),
        "clean_flow_3d_smooth": float(config.clean_flow_3d_smooth_weight),
        "root_move": float(config.root_move_reg_weight),
    }
    weighted = {
        name: value * float(weights[name])
        for name, value in components.items()
    }
    total = torch.stack(list(weighted.values())).sum()
    report = {
        "raw_components": {
            name: float(value.detach().cpu())
            for name, value in components.items()
        },
        "weights": weights,
        "weighted_components": {
            name: float(value.detach().cpu())
            for name, value in weighted.items()
        },
        "graph_reports": {
            "render": graph_report,
            "guide": guide_graph_report,
            "geometry": geometry_graph_report,
        },
    }
    return total, report


def main() -> None:
    args = parse_args()
    if args.samples < 2:
        raise ValueError("--samples must include a root and a non-root sample")
    if args.query_root_chunk <= 0 or args.gradient_root_batch <= 0:
        raise ValueError("query and gradient root batches must be positive")
    if args.visual_strands <= 0:
        raise ValueError("--visual-strands must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA diagnostic requested but CUDA is unavailable")

    device = torch.device(args.device)
    checkpoint_path = resolve_project_path(args.checkpoint)
    mesh_override = resolve_project_path(args.mesh) if args.mesh else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    model, config, checkpoint = load_stage1_checkpoint_model(
        checkpoint_path,
        device,
        mesh_path_override=mesh_override,
    )
    model.eval()
    mesh_path = resolve_project_path(config.mesh_path)
    collision_config = replace(
        config,
        mesh_no_penetration_support=True,
        mesh_no_penetration_sdf=str(Path(args.sdf).resolve()),
        mesh_no_penetration_weight=1.0,
        mesh_no_penetration_root_batch=int(args.gradient_root_batch),
    )
    field, sdf_report = load_mesh_no_penetration_field(
        collision_config,
        mesh_path,
        device,
    )
    if field is None or sdf_report is None:
        raise RuntimeError("failed to load the requested mesh SDF")

    with torch.no_grad():
        strands, widths, colors, opacities = build_checkpoint_strands(
            model,
            int(args.samples),
            mesh_local=True,
        )
        root_count = int(strands.shape[0])
        root_max = torch.empty((root_count,), device="cpu", dtype=torch.float32)
        root_mean = torch.empty_like(root_max)
        penetrating_points = 0
        point_count = 0
        depth_sum = 0.0
        maximum_depth = 0.0
        for start in range(0, root_count, int(args.query_root_chunk)):
            stop = min(start + int(args.query_root_chunk), root_count)
            depth = strand_penetration_depth(strands[start:stop], field)
            root_max[start:stop] = depth.max(dim=1).values.cpu()
            root_mean[start:stop] = depth.mean(dim=1).cpu()
            penetrating_points += int((depth > 0.0).sum().cpu())
            point_count += int(depth.numel())
            depth_sum += float(depth.sum().cpu())
            maximum_depth = max(maximum_depth, float(depth.max().cpu()))

        visual_count = min(int(args.visual_strands), root_count)
        visual_ids = torch.linspace(
            0,
            root_count - 1,
            steps=visual_count,
            device=device,
        ).long()
        visual_color = strands.new_tensor(args.visual_color).view(1, 1, 3)
        visual_colors = visual_color.expand(
            visual_count,
            int(args.samples),
            3,
        )
        visual_ids_cpu = visual_ids.cpu()
        visual_world = (
            strands[visual_ids] * torch.exp(model.log_scale.detach())
            + model.translation.detach().reshape(1, 1, 3)
        )
        np.savez_compressed(
            output_dir / "penetration_visual_strands.npz",
            strands=visual_world.cpu().numpy().astype(np.float32),
            widths=widths[visual_ids].cpu().numpy().astype(np.float32),
            colors=visual_colors.cpu().numpy().astype(np.float32),
            opacities=opacities[visual_ids].cpu().numpy().astype(np.float32),
            root_ids=visual_ids_cpu.numpy().astype(np.int64),
            penetration_mask=(root_max[visual_ids_cpu] > 0.0).numpy(),
            penetration_max_depth=root_max[visual_ids_cpu].numpy(),
        )
        np.savez_compressed(
            output_dir / "root_penetration.npz",
            root_max_depth=root_max.numpy(),
            root_mean_depth=root_mean.numpy(),
        )

    del strands, widths, colors, opacities, visual_colors, visual_world
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model.zero_grad(set_to_none=True)
    strands, _, _, _ = build_checkpoint_strands(
        model,
        int(args.samples),
        mesh_local=True,
    )
    gradient_ids = cyclic_strand_indices(
        int(strands.shape[0]),
        int(args.gradient_root_batch),
        1,
        device=device,
    )
    gradient_depth = strand_penetration_depth(strands[gradient_ids], field)
    gradient_loss = gradient_depth.mean()
    gradient_penetrating_fraction = float(
        (gradient_depth.detach() > 0.0).float().mean().cpu()
    )
    gradient_loss.backward()
    gradient_norms, collision_gradients = collect_gradient_norms(model)

    del strands, gradient_depth
    model.zero_grad(set_to_none=True)
    weighted_structure_loss, structure_report = weighted_structure_regularization(
        model,
        config,
    )
    weighted_structure_loss.backward()
    structure_gradient_norms, structure_gradients = collect_gradient_norms(model)
    shared_gradient_comparison = {}
    for name in sorted(set(collision_gradients) & set(structure_gradients)):
        collision_gradient = collision_gradients[name].reshape(-1)
        structure_gradient = structure_gradients[name].reshape(-1)
        collision_l2 = torch.linalg.vector_norm(collision_gradient)
        structure_l2 = torch.linalg.vector_norm(structure_gradient)
        denominator = (collision_l2 * structure_l2).clamp_min(1.0e-30)
        shared_gradient_comparison[name] = {
            "collision_l2": float(collision_l2.cpu()),
            "weighted_structure_l2": float(structure_l2.cpu()),
            "equal_l2_collision_weight": float(
                (structure_l2 / collision_l2.clamp_min(1.0e-30)).cpu()
            ),
            "cosine": float(
                (torch.dot(collision_gradient, structure_gradient) / denominator).cpu()
            ),
        }

    penetrating_roots = root_max > 0.0
    positive_root_depth = root_max[penetrating_roots]
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": int(checkpoint.get("iteration", -1)),
        "sdf": sdf_report,
        "samples_per_strand": int(args.samples),
        "root_count": int(root_max.numel()),
        "point_count": point_count,
        "penetrating_point_count": penetrating_points,
        "penetrating_point_fraction": (
            float(penetrating_points / point_count) if point_count else 0.0
        ),
        "mean_dimensionless_depth": (
            float(depth_sum / point_count) if point_count else 0.0
        ),
        "maximum_dimensionless_depth": maximum_depth,
        "penetrating_root_count": int(penetrating_roots.sum()),
        "penetrating_root_fraction": float(penetrating_roots.float().mean()),
        "root_max_depth_all": summarize(root_max),
        "root_max_depth_positive": summarize(positive_root_depth),
        "gradient_root_batch": int(gradient_ids.numel()),
        "gradient_batch_penetrating_point_fraction": (
            gradient_penetrating_fraction
        ),
        "unweighted_gradient_loss": float(gradient_loss.detach().cpu()),
        "parameter_gradient_norms": gradient_norms,
        "weighted_structure_loss": float(weighted_structure_loss.detach().cpu()),
        "weighted_structure": structure_report,
        "weighted_structure_parameter_gradient_norms": structure_gradient_norms,
        "collision_to_weighted_structure_gradients": shared_gradient_comparison,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

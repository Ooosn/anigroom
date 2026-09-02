from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from anigroom.grooming import decode_positive_asinh_ratio  # noqa: E402
from anigroom.grooming.guide_attribute_gaussian_field import (  # noqa: E402
    GuideAttributeGaussianField,
    GuideGaussianBinding,
    GuideGaussianFieldConfig,
    density_preserving_topology_fps,
    initialize_guide_gaussian_binding,
)
from tools import train_white_tiger_stage1 as stage1  # noqa: E402
from tools import visualize_white_tiger_groom_attributes as visualizer  # noqa: E402
from tools.diagnose_nested_guide_gaussian_counts import (  # noqa: E402
    _binding_to_device,
    _topology_binding,
    _undirected_graph,
    summarize,
)
from tools.visualize_r085_direction_color_maps import (  # noqa: E402
    direction_difference,
    transported_direction_field,
)
from tools.visualize_r085_direction_surface_maps import (  # noqa: E402
    rasterize_direction_hue,
    vertex_normals,
)
from tools.visualize_r085_guide_density import rasterize_scalar_field  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guide_d8(points: np.ndarray, neighbor_count: int = 8) -> np.ndarray:
    tree = cKDTree(np.asarray(points, dtype=np.float64))
    distances, ids = tree.query(points, k=int(neighbor_count) + 1, workers=1)
    distances = np.asarray(distances, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if distances.shape != (points.shape[0], int(neighbor_count) + 1):
        raise RuntimeError("guide KNN returned an invalid shape")
    if not np.all(ids[:, 0] == np.arange(points.shape[0])):
        raise RuntimeError("guide KNN did not return self first")
    result = distances[:, int(neighbor_count)]
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise RuntimeError("guide d8 must be finite and positive")
    return result


def fixed_radius_binding(
    guide_points: torch.Tensor,
    query_points: torch.Tensor,
    nominal_radius: np.ndarray,
    config: GuideGaussianFieldConfig,
) -> GuideGaussianBinding:
    guides = guide_points.detach().cpu().numpy().astype(np.float64)
    queries = query_points.detach().cpu().numpy().astype(np.float64)
    radius = np.asarray(nominal_radius, dtype=np.float64).reshape(-1)
    if radius.shape != (guides.shape[0],):
        raise ValueError("nominal_radius must have one value per guide")
    if not np.isfinite(radius).all() or np.any(radius <= 0.0):
        raise ValueError("nominal_radius must be finite and positive")
    query_tree = cKDTree(queries)
    query_chunks: list[np.ndarray] = []
    guide_chunks: list[np.ndarray] = []
    candidate_radius = float(config.max_scale_ratio) * radius
    for guide_id, current_radius in enumerate(candidate_radius.tolist()):
        current = np.asarray(
            query_tree.query_ball_point(guides[guide_id], current_radius),
            dtype=np.int64,
        )
        if current.size:
            current.sort(kind="stable")
            query_chunks.append(current)
            guide_chunks.append(np.full(current.shape, guide_id, dtype=np.int64))
    if not query_chunks:
        raise RuntimeError("fixed-radius candidate binding is empty")
    query_ids = np.concatenate(query_chunks)
    guide_ids = np.concatenate(guide_chunks)
    order = np.lexsort((guide_ids, query_ids))
    query_ids = query_ids[order]
    guide_ids = guide_ids[order]
    keep = np.ones((query_ids.size,), dtype=bool)
    keep[1:] = (query_ids[1:] != query_ids[:-1]) | (
        guide_ids[1:] != guide_ids[:-1]
    )
    query_ids = query_ids[keep]
    guide_ids = guide_ids[keep]
    row_count = np.bincount(query_ids, minlength=queries.shape[0]).astype(np.int64)
    if np.any(row_count <= 0):
        uncovered = np.flatnonzero(row_count <= 0)
        raise RuntimeError(
            f"fixed-radius binding leaves queries uncovered: {uncovered[:8].tolist()}"
        )
    row_ptr = np.empty((queries.shape[0] + 1,), dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(row_count, out=row_ptr[1:])
    return GuideGaussianBinding(
        guide_points=guide_points.detach().cpu().clone(),
        query_points=query_points.detach().cpu().clone(),
        reference_sigma=torch.as_tensor(
            radius / float(config.support_sigma),
            dtype=torch.float32,
        ),
        row_ptr=torch.as_tensor(row_ptr, dtype=torch.long),
        guide_ids=torch.as_tensor(guide_ids, dtype=torch.long),
        query_ids=torch.as_tensor(query_ids, dtype=torch.long),
        report={
            "guide_count": int(guides.shape[0]),
            "query_count": int(queries.shape[0]),
            "candidate_pair_count": int(guide_ids.size),
            "candidate_count": summarize(row_count),
            "nominal_radius": summarize(radius),
            "candidate_radius": summarize(candidate_radius),
            "all_queries_covered": True,
            "fallback_used": False,
            "config": config.to_json_dict(),
        },
        config=config,
    )


def circular_gradient(
    colors: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    rgb = (np.clip(colors, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    hue = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0].astype(np.float32) * 2.0
    valid_mask = np.asarray(valid, dtype=bool)
    gradient = np.zeros(valid_mask.shape, dtype=np.float32)
    values = []
    right_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
    right = np.abs(hue[:, 1:] - hue[:, :-1])
    right = np.minimum(right, 360.0 - right)
    gradient[:, 1:] = np.maximum(
        gradient[:, 1:],
        np.where(right_valid, right, 0.0),
    )
    gradient[:, :-1] = np.maximum(
        gradient[:, :-1],
        np.where(right_valid, right, 0.0),
    )
    values.append(right[right_valid])
    down_valid = valid_mask[1:, :] & valid_mask[:-1, :]
    down = np.abs(hue[1:, :] - hue[:-1, :])
    down = np.minimum(down, 360.0 - down)
    gradient[1:, :] = np.maximum(
        gradient[1:, :],
        np.where(down_valid, down, 0.0),
    )
    gradient[:-1, :] = np.maximum(
        gradient[:-1, :],
        np.where(down_valid, down, 0.0),
    )
    values.append(down[down_valid])
    edge_values = np.concatenate(values)
    report = summarize(edge_values)
    report.update(
        {
            "fraction_over_5_deg_per_pixel": float(np.mean(edge_values > 5.0)),
            "fraction_over_10_deg_per_pixel": float(np.mean(edge_values > 10.0)),
            "fraction_over_20_deg_per_pixel": float(np.mean(edge_values > 20.0)),
            "fraction_over_45_deg_per_pixel": float(np.mean(edge_values > 45.0)),
        }
    )
    return gradient, valid_mask, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare shared Gaussian guide-count and bandwidth arms.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--base-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=9)
    parser.add_argument("--subset-count", type=int, default=2048)
    parser.add_argument("--shared-length-lo", type=float, required=True)
    parser.add_argument("--shared-length-hi", type=float, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    base_image = args.base_image.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise RuntimeError("refusing to overwrite R085 shared-field output")
    if not checkpoint.is_file() or not base_image.is_file():
        raise RuntimeError("required shared-field input is missing")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    staging.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    started = time.perf_counter()

    print("LOAD_CHECKPOINT_START", flush=True)
    model, config, checkpoint_payload = stage1.load_stage1_checkpoint_model(checkpoint, device)
    print("LOAD_CHECKPOINT_DONE", flush=True)
    guide_points = model.guide_points_local.detach()
    guide_count = int(guide_points.shape[0])
    guide_np = guide_points.cpu().numpy().astype(np.float64)
    original_d8 = guide_d8(guide_np)
    original_median = float(np.median(original_d8))
    graph = model.guide_surface_interpolator().source_neighbor_graph(12)
    density_graph = model.guide_surface_interpolator().source_neighbor_graph(8)
    density_spacing = density_graph.source_area_weights.sqrt().detach().cpu().numpy()
    fps = density_preserving_topology_fps(
        guide_np,
        graph.edges.detach().cpu().numpy(),
        graph.distances.detach().cpu().numpy(),
        density_spacing,
        int(args.subset_count),
    )
    subset_ids = np.array(fps.selected_ids, dtype=np.int64, copy=True)
    subset_np = guide_np[subset_ids]
    subset_d8 = guide_d8(subset_np)
    subset_median = float(np.median(subset_d8))
    field_config = GuideGaussianFieldConfig(neighbor_count=8)
    topology = _undirected_graph(
        graph.edges.detach().cpu().numpy().astype(np.int64),
        graph.distances.detach().cpu().numpy().astype(np.float64),
        guide_count,
    )
    bandwidth_count = 256
    bandwidth_ids = np.array(subset_ids[:bandwidth_count], copy=True)
    bandwidth_distance = np.asarray(
        dijkstra(topology, directed=False, indices=bandwidth_ids),
        dtype=np.float64,
    )
    bandwidth_binding, _bandwidth_spacing, bandwidth_report = _topology_binding(
        guide_points.cpu(),
        guide_points.cpu(),
        bandwidth_ids,
        bandwidth_distance,
        field_config,
    )
    bandwidth_field = GuideAttributeGaussianField(
        _binding_to_device(bandwidth_binding, device)
    )
    bandwidth_ids_t = torch.as_tensor(
        bandwidth_ids,
        device=device,
        dtype=torch.long,
    )
    smooth_radius = torch.exp(
        bandwidth_field(
            torch.log(
                torch.as_tensor(
                    original_d8,
                    device=device,
                    dtype=torch.float32,
                )[bandwidth_ids_t]
            )
        )
    ).detach().cpu().numpy().astype(np.float64)

    arms = [
        {
            "slug": "adaptive4500",
            "label": "current adaptive d8, 4500 guides",
            "ids": np.arange(guide_count, dtype=np.int64),
            "radius": original_d8,
            "binding_mode": "existing_initializer",
        },
        {
            "slug": "uniform4500",
            "label": "uniform median d8, 4500 guides",
            "ids": np.arange(guide_count, dtype=np.int64),
            "radius": np.full((guide_count,), original_median),
            "binding_mode": "fixed_radius",
        },
        {
            "slug": "uniform125_4500",
            "label": "uniform 1.25x median d8, 4500 guides",
            "ids": np.arange(guide_count, dtype=np.int64),
            "radius": np.full((guide_count,), original_median * 1.25),
            "binding_mode": "fixed_radius",
        },
        {
            "slug": "smooth256_4500",
            "label": "4500 guides, d8 smoothed by nested Gaussian M=256",
            "ids": np.arange(guide_count, dtype=np.int64),
            "radius": smooth_radius,
            "binding_mode": "fixed_radius",
        },
        {
            "slug": "adaptive2048",
            "label": "unified topology-FPS 2048, adaptive d8",
            "ids": subset_ids,
            "radius": subset_d8,
            "binding_mode": "fixed_radius",
        },
        {
            "slug": "uniform2048",
            "label": "unified topology-FPS 2048, uniform median d8",
            "ids": subset_ids,
            "radius": np.full((subset_ids.size,), subset_median),
            "binding_mode": "fixed_radius",
        },
    ]
    vertices_local = model.vertices.detach()
    vertices_world = (
        vertices_local * torch.exp(model.log_scale.detach()).view(1, 1)
        + model.translation.detach().view(1, 3)
    )
    faces = model.faces.detach().long()
    vertex_normal, vertex_normal_report = vertex_normals(vertices_local, faces)
    guide_normals, _guide_tangent, _guide_bitangent = model.guide_normals_and_tangent_frames()
    clean_direction = F.normalize(model.guide_clean_flow_direction_target.detach(), dim=-1, eps=1.0e-8)
    learned_length = decode_positive_asinh_ratio(
        model.guide_length_raw,
        model.guide_length_reference,
    ).reshape(-1).detach()
    base = visualizer._read_base(
        base_image,
        int(config.expected_width),
        int(config.expected_height),
    )

    artifacts: list[Path] = []
    reports = []
    baseline_direction = None
    for arm in arms:
        slug = str(arm["slug"])
        print(f"ARM_START {slug}", flush=True)
        ids_np = np.asarray(arm["ids"], dtype=np.int64)
        ids_t = torch.as_tensor(ids_np, device=device, dtype=torch.long)
        source_points = guide_points[ids_t]
        source_direction = clean_direction[ids_t]
        source_normal = guide_normals[ids_t]
        source_length = learned_length[ids_t]
        radius = np.asarray(arm["radius"], dtype=np.float64)

        binding_started = time.perf_counter()
        if arm["binding_mode"] == "existing_initializer":
            vertex_binding = initialize_guide_gaussian_binding(
                source_points,
                vertices_local,
                field_config,
                device="cpu",
                dtype=torch.float32,
            )
        else:
            vertex_binding = fixed_radius_binding(
                source_points,
                vertices_local,
                radius,
                field_config,
            )
        binding_seconds = time.perf_counter() - binding_started
        field = GuideAttributeGaussianField(_binding_to_device(vertex_binding, device))
        direction, pre_magnitude = transported_direction_field(
            field,
            source_direction,
            source_normal,
            vertex_normal,
        )
        vertex_length = field(source_length)
        constant = field(torch.ones((ids_np.size,), device=device))
        constant_error = float((constant - 1.0).abs().max().cpu())

        if baseline_direction is None:
            baseline_direction = direction.detach().clone()
        relative_direction = direction_difference(direction, baseline_direction)
        colors, valid, raster_report = rasterize_direction_hue(
            model,
            config,
            vertices_world,
            direction,
            int(args.view_index),
            device,
        )
        direction_path = staging / f"view09_direction_surface_{slug}.png"
        direction_overlay = visualizer._overlay_direction_surface_colors(
            base,
            colors,
            valid,
            title=f"R085 shared field: {arm['label']}",
            out_path=direction_path,
            alpha=1.0,
        )
        artifacts.append(direction_path)

        gradient, gradient_valid, gradient_report = circular_gradient(colors, valid)
        gradient_path = staging / f"view09_direction_gradient_{slug}.png"
        gradient_overlay = visualizer._overlay_scalar_surface_values(
            base,
            gradient,
            gradient_valid,
            title=f"R085 angular gradient deg/pixel: {arm['label']}",
            out_path=gradient_path,
            lo=0.0,
            hi=20.0,
            alpha=1.0,
        )
        artifacts.append(gradient_path)

        scalar, scalar_valid = rasterize_scalar_field(
            model,
            config,
            vertices_world,
            vertex_length,
            int(args.view_index),
            device,
        )
        length_path = staging / f"view09_length_surface_{slug}.png"
        length_overlay = visualizer._overlay_scalar_surface_values(
            base,
            scalar,
            scalar_valid,
            title=f"R085 learned-length stress: {arm['label']}",
            out_path=length_path,
            lo=float(args.shared_length_lo),
            hi=float(args.shared_length_hi),
            alpha=1.0,
        )
        artifacts.append(length_path)

        reports.append(
            {
                "slug": slug,
                "label": arm["label"],
                "guide_count": int(ids_np.size),
                "nominal_radius": summarize(radius),
                "binding_seconds": float(binding_seconds),
                "vertex_binding": vertex_binding.report,
                "constant_reproduction_max_abs_error": constant_error,
                "pre_normalization_magnitude": summarize(pre_magnitude),
                "relative_to_adaptive4500": relative_direction,
                "direction_raster": raster_report,
                "direction_gradient": gradient_report,
                "direction_image": direction_path.name,
                "direction_gradient_image": gradient_path.name,
                "length": summarize(vertex_length),
                "length_image": length_path.name,
                "overlays": {
                    "direction": direction_overlay,
                    "gradient": gradient_overlay,
                    "length": length_overlay,
                },
            }
        )
        del field, vertex_binding, direction, vertex_length, constant
        torch.cuda.empty_cache()
        print(f"ARM_DONE {slug}", flush=True)

    subset_path = staging / "unified_topology_fps_2048.npy"
    np.save(subset_path, subset_ids, allow_pickle=False)
    artifacts.append(subset_path)
    report = {
        "schema": "anigroom.r085.shared_gaussian_arms.v1",
        "status": "complete",
        "training": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "iteration": int(checkpoint_payload["iteration"]),
        },
        "contract": {
            "shared_weights_across_attributes": True,
            "direction_specific_field": False,
            "post_blur": False,
            "arms_predeclared": [str(arm["slug"]) for arm in arms],
        },
        "counts": {
            "original_guide": guide_count,
            "subset_guide": int(subset_ids.size),
            "mesh_vertex": int(vertices_local.shape[0]),
            "mesh_face": int(faces.shape[0]),
        },
        "reference_radius": {
            "original_adaptive_d8": summarize(original_d8),
            "original_median_d8": original_median,
            "subset_adaptive_d8": summarize(subset_d8),
            "subset_median_d8": subset_median,
            "smooth256_d8": summarize(smooth_radius),
        },
        "bandwidth_field": {
            "guide_count": bandwidth_count,
            "binding": bandwidth_report,
            "semantics": "nested M256 Gaussian reconstruction of log d8 at all 4500 guides",
        },
        "subset_fps": {**fps.report, "ids_file": subset_path.name},
        "vertex_normals": vertex_normal_report,
        "arms": reports,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report_path = staging / "r085_shared_gaussian_arms.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts.insert(0, report_path)
    manifest_path = staging / "manifest.sha256"
    manifest_path.write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in artifacts) + "\n",
        encoding="utf-8",
    )
    staging.rename(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "report": str(output / report_path.name),
                "manifest": str(output / manifest_path.name),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

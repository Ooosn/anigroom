from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import torch


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray | torch.Tensor) -> dict[str, float | int]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("summary requires non-empty finite values")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p01": float(np.quantile(array, 0.01)),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _undirected_graph(
    edges: np.ndarray,
    lengths: np.ndarray,
    count: int,
):
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    order = np.lexsort((lengths, hi, lo))
    lo, hi, lengths = lo[order], hi[order], lengths[order]
    start = np.ones((lo.shape[0],), dtype=bool)
    start[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    offsets = np.flatnonzero(start)
    lo, hi = lo[offsets], hi[offsets]
    lengths = np.minimum.reduceat(lengths, offsets)
    graph = coo_matrix(
        (
            np.concatenate((lengths, lengths)),
            (np.concatenate((lo, hi)), np.concatenate((hi, lo))),
        ),
        shape=(count, count),
        dtype=np.float64,
    ).tocsr()
    graph.sort_indices()
    return graph


def _topology_binding(
    guide_points: torch.Tensor,
    query_points: torch.Tensor,
    selected_ids: np.ndarray,
    selected_to_query_distance: np.ndarray,
    config: GuideGaussianFieldConfig,
) -> tuple[GuideGaussianBinding, np.ndarray, dict[str, object]]:
    selected_count = int(selected_ids.shape[0])
    query_count = int(query_points.shape[0])
    selected_distance = selected_to_query_distance[:, selected_ids]
    if selected_distance.shape != (selected_count, selected_count):
        raise RuntimeError("selected topology-distance matrix has invalid shape")
    spacing = np.empty((selected_count,), dtype=np.float64)
    for row in range(selected_count):
        values = selected_distance[row]
        valid = np.isfinite(values) & (np.arange(selected_count) != row)
        ordered = np.sort(values[valid], kind="stable")
        if ordered.size < int(config.neighbor_count):
            raise RuntimeError("coarse guide has insufficient topology neighbors")
        spacing[row] = float(ordered[int(config.neighbor_count) - 1])
    if not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise RuntimeError("coarse topology spacing is invalid")
    reference_sigma = spacing / float(config.support_sigma)
    candidate_radius = (
        float(config.support_sigma)
        * float(config.max_scale_ratio)
        * reference_sigma
    )
    candidate_mask = selected_to_query_distance <= candidate_radius[:, None]
    guide_ids, query_ids = np.nonzero(candidate_mask)
    order = np.lexsort((guide_ids, query_ids))
    query_ids = np.asarray(query_ids[order], dtype=np.int64)
    guide_ids = np.asarray(guide_ids[order], dtype=np.int64)
    row_count = np.bincount(query_ids, minlength=query_count).astype(np.int64)
    if row_count.shape != (query_count,) or np.any(row_count <= 0):
        uncovered = np.flatnonzero(row_count <= 0)
        raise RuntimeError(
            f"topology-gated Gaussian candidates leave queries uncovered: {uncovered[:8].tolist()}"
        )
    row_ptr = np.empty((query_count + 1,), dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(row_count, out=row_ptr[1:])
    device = guide_points.device
    dtype = guide_points.dtype
    selected_tensor = torch.as_tensor(selected_ids, device=device, dtype=torch.long)
    report = {
        "guide_count": selected_count,
        "query_count": query_count,
        "candidate_pair_count": int(guide_ids.size),
        "candidate_count": summarize(row_count),
        "topology_spacing": summarize(spacing),
        "reference_sigma": summarize(reference_sigma),
        "max_candidate_radius": summarize(candidate_radius),
        "topology_gated": True,
        "fallback_used": False,
        "all_queries_candidate_covered": True,
        "config": config.to_json_dict(),
    }
    return (
        GuideGaussianBinding(
            guide_points=guide_points[selected_tensor].detach().clone(),
            query_points=query_points.detach().clone(),
            reference_sigma=torch.as_tensor(
                reference_sigma,
                device=device,
                dtype=dtype,
            ),
            row_ptr=torch.as_tensor(row_ptr, device=device, dtype=torch.long),
            guide_ids=torch.as_tensor(guide_ids, device=device, dtype=torch.long),
            query_ids=torch.as_tensor(query_ids, device=device, dtype=torch.long),
            report=report,
            config=config,
        ),
        spacing,
        report,
    )


def _binding_to_device(
    binding: GuideGaussianBinding,
    device: torch.device,
) -> GuideGaussianBinding:
    return replace(
        binding,
        guide_points=binding.guide_points.to(device=device),
        query_points=binding.query_points.to(device=device),
        reference_sigma=binding.reference_sigma.to(device=device),
        row_ptr=binding.row_ptr.to(device=device),
        guide_ids=binding.guide_ids.to(device=device),
        query_ids=binding.query_ids.to(device=device),
        report={**binding.report, "device": str(device)},
    )


def _row_weight_metrics(field: GuideAttributeGaussianField):
    weights = field.evaluate_weights()
    active = torch.zeros(
        (field.query_count,),
        device=weights.raw.device,
        dtype=weights.raw.dtype,
    )
    active.index_add_(0, weights.query_ids, (weights.raw > 0.0).float())
    square_sum = torch.zeros_like(active)
    square_sum.index_add_(0, weights.query_ids, weights.normalized.square())
    effective = square_sum.clamp_min(1.0e-12).reciprocal()
    maximum = torch.full_like(active, -torch.inf)
    maximum.scatter_reduce_(
        0,
        weights.query_ids,
        weights.normalized,
        reduce="amax",
        include_self=True,
    )
    row_sum = torch.zeros_like(active)
    row_sum.index_add_(0, weights.query_ids, weights.normalized)
    return weights, active, effective, maximum, row_sum


def _edge_log_jump(values: torch.Tensor, edges: torch.Tensor) -> dict[str, float | int]:
    jump = (
        torch.log(values[edges[:, 0]].clamp_min(1.0e-12))
        - torch.log(values[edges[:, 1]].clamp_min(1.0e-12))
    ).abs()
    return summarize(jump)


@torch.no_grad()
def _visible_projection(
    model,
    config,
    points_world: torch.Tensor,
    view_index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    data_root = stage1.resolve_project_path(config.data_root)
    viewmats, ks = stage1.load_camera_tensors(data_root, device)
    width, height = int(config.expected_width), int(config.expected_height)
    viewmat, k = viewmats[view_index], ks[view_index]
    xy, depth = stage1.project_points(points_world, viewmat, k)
    import nvdiffrast.torch as dr

    ctx = dr.RasterizeCudaContext(device=device)
    mesh_depth = stage1.render_model_mesh_depth(
        model,
        viewmat,
        k,
        width,
        height,
        device=device,
        ctx=ctx,
    )
    sampled_depth = stage1.sample_depth_nearest(
        mesh_depth.depth,
        xy,
        kernel_size=int(config.mesh_depth_local_kernel),
    )
    tolerance = float(config.mesh_depth_abs_tolerance) + depth.abs() * float(
        config.mesh_depth_rel_tolerance
    )
    visible = (
        torch.isfinite(sampled_depth)
        & (depth > 1.0e-6)
        & (depth <= sampled_depth + tolerance)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < height)
    )
    ids = torch.nonzero(visible, as_tuple=False).reshape(-1).cpu().numpy()
    return xy.detach().cpu().numpy()[ids], ids


def _parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not counts or any(count <= 8 for count in counts):
        raise ValueError("counts must contain integers greater than Gaussian K")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose nested coarse-guide Gaussian counts on one fixed checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--base-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", default="256,512,1024")
    parser.add_argument("--view-index", type=int, default=9)
    parser.add_argument("--shared-length-lo", type=float, required=True)
    parser.add_argument("--shared-length-hi", type=float, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    base_image = args.base_image.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise RuntimeError("refusing to overwrite nested Gaussian diagnostic output")
    if not checkpoint.is_file() or not base_image.is_file():
        raise RuntimeError("checkpoint or canonical base image is missing")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    counts = _parse_counts(args.counts)
    staging.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    started = time.perf_counter()

    print("LOAD_CHECKPOINT_START", flush=True)
    model, config, checkpoint_payload = stage1.load_stage1_checkpoint_model(
        checkpoint,
        device,
    )
    print("LOAD_CHECKPOINT_DONE", flush=True)
    if max(counts) >= int(model.guide_points_local.shape[0]):
        raise RuntimeError("coarse count must be smaller than primary guide count")
    guide_points = model.guide_points_local.detach()
    roots, normals, roots_local = model.roots_and_normals()
    guide_graph = model.guide_surface_interpolator().source_neighbor_graph(12)
    density_graph = model.guide_surface_interpolator().source_neighbor_graph(8)
    graph_edges_np = guide_graph.edges.detach().cpu().numpy().astype(np.int64)
    graph_lengths_np = guide_graph.distances.detach().cpu().numpy().astype(np.float64)
    density_spacing_np = (
        density_graph.source_area_weights.detach().sqrt().cpu().numpy().astype(np.float64)
    )
    fps = density_preserving_topology_fps(
        guide_points,
        graph_edges_np,
        graph_lengths_np,
        density_spacing_np,
        max(counts),
    )
    topology_graph = _undirected_graph(
        graph_edges_np,
        graph_lengths_np,
        int(guide_points.shape[0]),
    )
    print("NESTED_FPS_DONE", flush=True)
    selected_to_query = np.asarray(
        dijkstra(
            topology_graph,
            directed=False,
            indices=fps.selected_ids,
        ),
        dtype=np.float64,
    )
    if selected_to_query.shape != (max(counts), int(guide_points.shape[0])):
        raise RuntimeError("nested topology-distance matrix has invalid shape")

    field_config = GuideGaussianFieldConfig(neighbor_count=8)
    render_binding_cpu = initialize_guide_gaussian_binding(
        guide_points,
        roots_local,
        field_config,
        device="cpu",
        dtype=torch.float32,
    )
    render_field = GuideAttributeGaussianField(
        _binding_to_device(render_binding_cpu, device)
    )
    learned_guide_length = decode_positive_asinh_ratio(
        model.guide_length_raw,
        model.guide_length_reference,
    ).reshape(-1).detach()
    guide_edges = density_graph.edges.detach().to(device=device, dtype=torch.long)

    render_xy, render_visible_ids = _visible_projection(
        model,
        config,
        roots,
        int(args.view_index),
        device,
    )
    guide_world = (
        guide_points * torch.exp(model.log_scale.detach()).view(1, 1)
        + model.translation.detach().view(1, 3)
    )
    base = visualizer._read_base(
        base_image,
        int(config.expected_width),
        int(config.expected_height),
    )

    arms_runtime: list[dict[str, object]] = []
    support_values: list[np.ndarray] = []
    for count in counts:
        print(f"ARM_START count={count}", flush=True)
        selected_ids = np.asarray(fps.selected_ids[:count], dtype=np.int64)
        binding_cpu, topology_spacing, binding_report = _topology_binding(
            guide_points.cpu(),
            guide_points.cpu(),
            selected_ids,
            selected_to_query[:count],
            field_config,
        )
        field = GuideAttributeGaussianField(_binding_to_device(binding_cpu, device))
        weights, active, effective, maximum, row_sum = _row_weight_metrics(field)
        constant = field(torch.ones((count,), device=device))
        constant_error = float((constant - 1.0).abs().max().cpu())
        if constant_error > 1.0e-6:
            raise RuntimeError(f"count={count} failed constant reproduction")
        if float((row_sum - 1.0).abs().max().cpu()) > 2.0e-6:
            raise RuntimeError(f"count={count} failed row normalization")

        selected_tensor = torch.as_tensor(selected_ids, device=device, dtype=torch.long)
        coarse_values = learned_guide_length[selected_tensor]
        reconstructed_guide = field(coarse_values)
        stress_render = render_field(reconstructed_guide)
        effective_render = render_field(effective)
        maximum_render = render_field(maximum)
        relative_error = (
            (reconstructed_guide - learned_guide_length).abs()
            / learned_guide_length.clamp_min(1.0e-12)
        )
        spacing_array = np.asarray(topology_spacing, dtype=np.float64)
        support_values.append(spacing_array)
        arm = {
            "count": int(count),
            "selected_ids": selected_ids,
            "topology_spacing": spacing_array,
            "binding_report": binding_report,
            "candidate_count": summarize(
                np.diff(binding_cpu.row_ptr.detach().cpu().numpy())
            ),
            "active_count": summarize(active),
            "effective_neighbor_count": summarize(effective),
            "maximum_weight": summarize(maximum),
            "row_sum_max_abs_error": float((row_sum - 1.0).abs().max().cpu()),
            "constant_reproduction_max_abs_error": constant_error,
            "reconstructed_guide_length": reconstructed_guide.detach().cpu().numpy(),
            "stress_render_length": stress_render.detach().cpu().numpy(),
            "effective_render": effective_render.detach().cpu().numpy(),
            "maximum_render": maximum_render.detach().cpu().numpy(),
            "relative_to_learned_guide": summarize(relative_error),
            "reconstructed_guide_edge_log_jump": _edge_log_jump(
                reconstructed_guide,
                guide_edges,
            ),
        }
        arms_runtime.append(arm)
        del field, binding_cpu, weights
        torch.cuda.empty_cache()
        print(f"ARM_DONE count={count}", flush=True)

    support_all = np.concatenate(support_values)
    support_lo = float(np.quantile(support_all, 0.02))
    support_hi = float(np.quantile(support_all, 0.98))
    if support_hi <= support_lo:
        support_hi = support_lo + 1.0e-6
    effective_lo = 1.0
    effective_hi = float(
        max(
            np.quantile(np.asarray(arm["effective_render"]), 0.98)
            for arm in arms_runtime
        )
    )
    maximum_lo, maximum_hi = 0.0, 1.0

    report_arms: list[dict[str, object]] = []
    artifact_paths: list[Path] = []
    for arm in arms_runtime:
        count = int(arm["count"])
        selected_ids = np.asarray(arm.pop("selected_ids"), dtype=np.int64)
        topology_spacing = np.asarray(arm.pop("topology_spacing"), dtype=np.float64)
        reconstructed_guide = np.asarray(
            arm.pop("reconstructed_guide_length"),
            dtype=np.float32,
        )
        stress_render = np.asarray(arm.pop("stress_render_length"), dtype=np.float32)
        effective_render = np.asarray(arm.pop("effective_render"), dtype=np.float32)
        maximum_render = np.asarray(arm.pop("maximum_render"), dtype=np.float32)

        selected_tensor = torch.as_tensor(selected_ids, device=device, dtype=torch.long)
        coarse_xy, coarse_visible_local = _visible_projection(
            model,
            config,
            guide_world[selected_tensor],
            int(args.view_index),
            device,
        )
        coarse_image = staging / f"view09_coarse_root_support_m{count}.png"
        visualizer._overlay_points(
            base,
            coarse_xy,
            topology_spacing[coarse_visible_local],
            title=f"R085 view09 nested coarse roots M={count} topology d8",
            out_path=coarse_image,
            lo=support_lo,
            hi=support_hi,
        )
        stress_image = staging / f"view09_length_stress_m{count}.png"
        visualizer._overlay_points(
            base,
            render_xy,
            stress_render[render_visible_ids],
            title=f"R085 view09 learned-length bandwidth stress M={count}",
            out_path=stress_image,
            lo=float(args.shared_length_lo),
            hi=float(args.shared_length_hi),
        )
        effective_image = staging / f"view09_effective_neighbors_m{count}.png"
        visualizer._overlay_points(
            base,
            render_xy,
            effective_render[render_visible_ids],
            title=f"R085 view09 level1 effective-neighbor field M={count}",
            out_path=effective_image,
            lo=effective_lo,
            hi=effective_hi,
        )
        maximum_image = staging / f"view09_maximum_weight_m{count}.png"
        visualizer._overlay_points(
            base,
            render_xy,
            maximum_render[render_visible_ids],
            title=f"R085 view09 level1 maximum-weight field M={count}",
            out_path=maximum_image,
            lo=maximum_lo,
            hi=maximum_hi,
        )
        array_path = staging / f"nested_gaussian_m{count}.npz"
        np.savez_compressed(
            array_path,
            selected_ids=selected_ids,
            topology_spacing=topology_spacing.astype(np.float32),
            reconstructed_guide_length=reconstructed_guide,
            stress_render_length=stress_render,
            effective_render=effective_render,
            maximum_render=maximum_render,
        )
        arm.update(
            {
                "coarse_visible_view09": int(coarse_visible_local.size),
                "coarse_root_support_image": coarse_image.name,
                "length_stress_image": stress_image.name,
                "effective_neighbor_image": effective_image.name,
                "maximum_weight_image": maximum_image.name,
                "arrays": array_path.name,
            }
        )
        report_arms.append(arm)
        artifact_paths.extend(
            [coarse_image, stress_image, effective_image, maximum_image, array_path]
        )

    nested_ids_path = staging / "nested_topology_fps_order.npy"
    np.save(nested_ids_path, fps.selected_ids, allow_pickle=False)
    artifact_paths.append(nested_ids_path)
    report = {
        "schema": "anigroom.r085_nested_guide_gaussian_count_diagnostic.v1",
        "status": "complete",
        "training": False,
        "definition": (
            "nested density-preserving topology FPS from fixed 4500 primary guides; "
            "coarse Gaussian evaluated only at primary-guide sites; learned 4k length "
            "used only as a bandwidth stress signal"
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "iteration": int(checkpoint_payload["iteration"]),
        },
        "base_image": str(base_image),
        "view_index": int(args.view_index),
        "counts": {
            "primary_guide": int(guide_points.shape[0]),
            "render_root": int(roots_local.shape[0]),
            "render_visible": int(render_visible_ids.size),
            "coarse_candidates": list(counts),
        },
        "fps": {
            **fps.report,
            "prefix_cover_max": {
                str(count): float(fps.normalized_cover_max[count - 1])
                for count in counts
            },
            "order_file": nested_ids_path.name,
        },
        "guide_graph": {
            "selection_neighbor_count": 12,
            "density_spacing_neighbor_count": 8,
            "edge_count": int(guide_graph.edges.shape[0]),
            "density_spacing": summarize(density_spacing_np),
            "distance_semantics": "primary-guide intrinsic topology graph shortest path",
        },
        "level2_render_binding": render_binding_cpu.report,
        "learned_guide_length": summarize(learned_guide_length),
        "shared_visual_ranges": {
            "length": [float(args.shared_length_lo), float(args.shared_length_hi)],
            "coarse_topology_d8": [support_lo, support_hi],
            "effective_neighbor": [effective_lo, effective_hi],
            "maximum_weight": [maximum_lo, maximum_hi],
        },
        "arms": report_arms,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report_path = staging / "r085_nested_gaussian_count_diagnostic.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_paths.insert(0, report_path)
    manifest = staging / "manifest.sha256"
    manifest.write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in artifact_paths)
        + "\n",
        encoding="utf-8",
    )
    staging.rename(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "report": str(output / report_path.name),
                "manifest": str(output / manifest.name),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

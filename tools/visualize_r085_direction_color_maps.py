from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from anigroom.flow.direction_geometry import parallel_transport_vectors  # noqa: E402
from anigroom.grooming.guide_attribute_gaussian_field import (  # noqa: E402
    GuideAttributeGaussianField,
    GuideGaussianFieldConfig,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transported_direction_field(
    field: GuideAttributeGaussianField,
    source_direction: torch.Tensor,
    source_normal: torch.Tensor,
    query_normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport, blend with Gaussian weights, and normalize one direction field."""

    if tuple(source_direction.shape) != (field.guide_count, 3):
        raise ValueError("source_direction shape does not match Gaussian guides")
    if tuple(source_normal.shape) != tuple(source_direction.shape):
        raise ValueError("source_normal must match source_direction")
    if tuple(query_normal.shape) != (field.query_count, 3):
        raise ValueError("query_normal shape does not match Gaussian queries")
    weights = field.evaluate_weights()
    source = F.normalize(source_direction, dim=-1, eps=1.0e-8)
    source_n = F.normalize(source_normal, dim=-1, eps=1.0e-8)
    query_n = F.normalize(query_normal, dim=-1, eps=1.0e-8)
    transported = parallel_transport_vectors(
        source[weights.guide_ids],
        source_n[weights.guide_ids],
        query_n[weights.query_ids],
    )
    contribution = transported * weights.normalized[:, None]
    vector_sum = torch.zeros(
        (field.query_count, 3),
        device=contribution.device,
        dtype=contribution.dtype,
    )
    vector_sum.index_add_(0, weights.query_ids, contribution)
    magnitude = torch.linalg.vector_norm(vector_sum, dim=-1)
    if not bool(torch.isfinite(magnitude).all()) or bool(
        (magnitude <= 1.0e-6).any()
    ):
        bad = torch.nonzero(magnitude <= 1.0e-6, as_tuple=False).reshape(-1)
        raise RuntimeError(
            f"transported Gaussian direction has degenerate rows: {bad[:8].cpu().tolist()}"
        )
    return F.normalize(vector_sum, dim=-1, eps=1.0e-8), magnitude


@torch.no_grad()
def project_direction_for_view(
    model,
    config,
    roots: torch.Tensor,
    direction: torch.Tensor,
    view_index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_root = stage1.resolve_project_path(config.data_root)
    viewmats, ks = stage1.load_camera_tensors(data_root, device)
    width, height = int(config.expected_width), int(config.expected_height)
    viewmat, k = viewmats[view_index], ks[view_index]
    xy, depth = stage1.project_points(roots, viewmat, k)
    xy2, depth2 = stage1.project_points(roots + direction * 0.055, viewmat, k)
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
        & (depth2 > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] < height)
    )
    ids = torch.nonzero(visible, as_tuple=False).reshape(-1).cpu().numpy()
    return (
        xy.detach().cpu().numpy()[ids],
        (xy2 - xy).detach().cpu().numpy()[ids],
        ids,
    )


def direction_difference(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | int]:
    dot = (candidate * reference).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(dot))
    report = summarize(angle)
    report.update(
        {
            "over_90_deg": int((angle > 90.0).sum().cpu()),
            "over_120_deg": int((angle > 120.0).sum().cpu()),
            "negative_dot": int((dot < 0.0).sum().cpu()),
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render R085 direction fields as cyclic hue maps.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--base-image", type=Path, required=True)
    parser.add_argument("--nested-order", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", default="256,512,1024")
    parser.add_argument("--view-index", type=int, default=9)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    base_image = args.base_image.resolve()
    nested_order_path = args.nested_order.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise RuntimeError("refusing to overwrite R085 direction-color output")
    if not checkpoint.is_file() or not base_image.is_file() or not nested_order_path.is_file():
        raise RuntimeError("required direction-color input is missing")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    counts = tuple(sorted({int(value) for value in args.counts.split(",")}))
    if not counts or min(counts) <= 8:
        raise ValueError("counts must contain values greater than Gaussian K")
    order = np.load(nested_order_path, allow_pickle=False).astype(np.int64)
    if order.ndim != 1 or order.size < max(counts) or np.unique(order).size != order.size:
        raise RuntimeError("nested order is invalid")
    staging.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device("cuda")

    print("LOAD_CHECKPOINT_START", flush=True)
    model, config, checkpoint_payload = stage1.load_stage1_checkpoint_model(checkpoint, device)
    print("LOAD_CHECKPOINT_DONE", flush=True)
    roots, render_normals, roots_local = model.roots_and_normals()
    guide_points = model.guide_points_local.detach()
    guide_normals, _guide_tangents, _guide_bitangents = (
        model.guide_normals_and_tangent_frames()
    )
    clean_direction = F.normalize(
        model.guide_clean_flow_direction_target.detach(),
        dim=-1,
        eps=1.0e-8,
    )
    learned_direction = F.normalize(
        model.guide_direction_world().detach(),
        dim=-1,
        eps=1.0e-8,
    )
    config_field = GuideGaussianFieldConfig(neighbor_count=8)
    render_binding = initialize_guide_gaussian_binding(
        guide_points,
        roots_local,
        config_field,
        device="cpu",
        dtype=torch.float32,
    )
    render_field = GuideAttributeGaussianField(_binding_to_device(render_binding, device))
    clean_render, clean_magnitude = transported_direction_field(
        render_field,
        clean_direction,
        guide_normals,
        render_normals,
    )
    learned_render, learned_magnitude = transported_direction_field(
        render_field,
        learned_direction,
        guide_normals,
        render_normals,
    )

    graph = model.guide_surface_interpolator().source_neighbor_graph(12)
    graph_edges = graph.edges.detach().cpu().numpy().astype(np.int64)
    graph_lengths = graph.distances.detach().cpu().numpy().astype(np.float64)
    topology = _undirected_graph(graph_edges, graph_lengths, int(guide_points.shape[0]))
    selected_to_query = np.asarray(
        dijkstra(topology, directed=False, indices=order[: max(counts)]),
        dtype=np.float64,
    )
    base = visualizer._read_base(
        base_image,
        int(config.expected_width),
        int(config.expected_height),
    )

    variants: list[tuple[str, str, torch.Tensor, torch.Tensor, dict[str, object]]] = [
        (
            "clean_flow_4500",
            "fixed 4500 clean-flow base",
            clean_render,
            clean_magnitude,
            {"relative_to_clean_flow": direction_difference(clean_render, clean_render)},
        ),
        (
            "learned_4k_4500",
            "learned R080 4k guide direction through Gaussian K8",
            learned_render,
            learned_magnitude,
            {"relative_to_clean_flow": direction_difference(learned_render, clean_render)},
        ),
    ]
    for count in counts:
        selected = np.asarray(order[:count], dtype=np.int64)
        binding, _spacing, binding_report = _topology_binding(
            guide_points.cpu(),
            guide_points.cpu(),
            selected,
            selected_to_query[:count],
            config_field,
        )
        coarse_field = GuideAttributeGaussianField(_binding_to_device(binding, device))
        selected_tensor = torch.as_tensor(selected, device=device, dtype=torch.long)
        coarse_guide_direction, coarse_guide_magnitude = transported_direction_field(
            coarse_field,
            clean_direction[selected_tensor],
            guide_normals[selected_tensor],
            guide_normals,
        )
        coarse_render_direction, coarse_render_magnitude = transported_direction_field(
            render_field,
            coarse_guide_direction,
            guide_normals,
            render_normals,
        )
        variants.append(
            (
                f"stress_m{count}",
                f"clean-flow bandwidth stress M={count}",
                coarse_render_direction,
                coarse_render_magnitude,
                {
                    "coarse_binding": binding_report,
                    "coarse_guide_pre_normalization_magnitude": summarize(
                        coarse_guide_magnitude
                    ),
                    "relative_to_clean_flow": direction_difference(
                        coarse_render_direction,
                        clean_render,
                    ),
                },
            )
        )

    artifact_paths: list[Path] = []
    variant_reports = []
    for slug, label, direction, magnitude, extra in variants:
        xy, screen_vector, visible_ids = project_direction_for_view(
            model,
            config,
            roots,
            direction,
            int(args.view_index),
            device,
        )
        image_path = staging / f"view09_direction_color_{slug}.png"
        overlay_report = visualizer._overlay_direction_colors(
            base,
            xy,
            screen_vector,
            title=f"R085 view09 direction hue: {label}",
            out_path=image_path,
        )
        artifact_paths.append(image_path)
        variant_reports.append(
            {
                "slug": slug,
                "label": label,
                "image": image_path.name,
                "visible_root_count": int(visible_ids.size),
                "pre_normalization_magnitude": summarize(magnitude),
                "overlay": overlay_report,
                **extra,
            }
        )

    report = {
        "schema": "anigroom.r085.direction_color_maps.v1",
        "status": "complete",
        "training": False,
        "color_contract": {
            "space": "screen-space directed projection",
            "hue_zero": "right",
            "hue_order": ["right", "down", "left", "up", "right"],
            "saturation": 1.0,
            "value": 1.0,
            "zero_projected_direction": "neutral gray",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "iteration": int(checkpoint_payload["iteration"]),
        },
        "nested_order": {
            "path": str(nested_order_path),
            "sha256": sha256_file(nested_order_path),
        },
        "counts": {
            "guide": int(guide_points.shape[0]),
            "render_root": int(roots_local.shape[0]),
            "coarse_candidates": list(counts),
        },
        "variants": variant_reports,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report_path = staging / "r085_direction_color_maps.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact_paths.insert(0, report_path)
    manifest_path = staging / "manifest.sha256"
    manifest_path.write_text(
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
                "manifest": str(output / manifest_path.name),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

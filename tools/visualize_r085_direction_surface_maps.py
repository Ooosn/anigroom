from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.sparse.csgraph import dijkstra
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from anigroom.grooming.guide_attribute_gaussian_field import (  # noqa: E402
    GuideAttributeGaussianField,
    GuideGaussianFieldConfig,
    initialize_guide_gaussian_binding,
)
from anigroom.projection.mesh_visibility import (  # noqa: E402
    render_mesh_vertex_color_from_tensors,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vertex_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    triangle = vertices[faces.long()]
    face_vector = torch.cross(
        triangle[:, 1] - triangle[:, 0],
        triangle[:, 2] - triangle[:, 0],
        dim=-1,
    )
    result = torch.zeros_like(vertices)
    for corner in range(3):
        result.index_add_(0, faces[:, corner].long(), face_vector)
    magnitude = torch.linalg.vector_norm(result, dim=-1)
    if not bool(torch.isfinite(magnitude).all()):
        raise RuntimeError("mesh contains a vertex with non-finite accumulated normal")
    invalid = magnitude <= 1.0e-8
    fallback_count = int(invalid.sum().cpu())
    orphan_count = 0
    if fallback_count:
        face_magnitude = torch.linalg.vector_norm(face_vector, dim=-1)
        valid_face = face_magnitude > 1.0e-8
        face_ids = torch.arange(
            faces.shape[0],
            device=faces.device,
            dtype=torch.long,
        )
        incident_vertex = faces[valid_face].reshape(-1).long()
        incident_face = face_ids[valid_face].repeat_interleave(3)
        sentinel = int(faces.shape[0])
        first_face = torch.full(
            (vertices.shape[0],),
            sentinel,
            device=faces.device,
            dtype=torch.long,
        )
        first_face.scatter_reduce_(
            0,
            incident_vertex,
            incident_face,
            reduce="amin",
            include_self=True,
        )
        has_face = invalid & (first_face < sentinel)
        if bool(has_face.any()):
            result[has_face] = F.normalize(
                face_vector[first_face[has_face]],
                dim=-1,
                eps=1.0e-8,
            )
        orphan = invalid & ~has_face
        orphan_count = int(orphan.sum().cpu())
        if orphan_count:
            result[orphan] = result.new_tensor([0.0, 0.0, 1.0])
    final_magnitude = torch.linalg.vector_norm(result, dim=-1)
    if not bool(torch.isfinite(final_magnitude).all()) or bool(
        (final_magnitude <= 1.0e-8).any()
    ):
        raise RuntimeError("mesh vertex-normal fallback did not produce valid normals")
    return F.normalize(result, dim=-1, eps=1.0e-8), {
        "accumulated_normal_fallback_count": fallback_count,
        "orphan_vertex_count": orphan_count,
    }


@torch.no_grad()
def rasterize_direction_hue(
    model,
    config,
    vertices_world: torch.Tensor,
    direction: torch.Tensor,
    view_index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    data_root = stage1.resolve_project_path(config.data_root)
    viewmats, ks = stage1.load_camera_tensors(data_root, device)
    viewmat, k = viewmats[view_index], ks[view_index]
    width, height = int(config.expected_width), int(config.expected_height)
    xy, depth = stage1.project_points(vertices_world, viewmat, k)
    xy2, depth2 = stage1.project_points(vertices_world + direction * 0.055, viewmat, k)
    screen = xy2 - xy
    magnitude = torch.linalg.vector_norm(screen, dim=-1)
    unit = screen / magnitude[:, None].clamp_min(1.0e-8)
    packed = torch.cat(
        ((unit + 1.0) * 0.5, torch.full_like(unit[:, :1], 0.5)),
        dim=-1,
    )
    import nvdiffrast.torch as dr

    ctx = dr.RasterizeCudaContext(device=device)
    interpolated, valid = render_mesh_vertex_color_from_tensors(
        vertices_world,
        model.faces.to(device=device, dtype=torch.int32),
        packed,
        viewmat,
        k,
        width,
        height,
        device=device,
        ctx=ctx,
        background=torch.tensor([0.5, 0.5, 0.5], device=device),
    )
    pixel_vector = interpolated[..., :2] * 2.0 - 1.0
    pixel_magnitude = torch.linalg.vector_norm(pixel_vector, dim=-1)
    valid_direction = valid & (pixel_magnitude > 1.0e-6)
    pixel_unit = pixel_vector / pixel_magnitude[..., None].clamp_min(1.0e-8)
    pixel_unit_np = pixel_unit.detach().cpu().numpy()
    valid_np = valid_direction.detach().cpu().numpy()
    colors = np.full((height, width, 3), 0.5, dtype=np.float32)
    valid_colors, _ = visualizer._screen_direction_colors(pixel_unit_np[valid_np])
    colors[valid_np] = valid_colors
    report = {
        "surface_pixel_count": int(valid.sum().cpu()),
        "valid_direction_pixel_count": int(valid_direction.sum().cpu()),
        "degenerate_direction_pixel_count": int((valid & ~valid_direction).sum().cpu()),
        "vertex_projected_magnitude": summarize(magnitude),
        "vertex_endpoint_behind_count": int((depth2 <= 1.0e-6).sum().cpu()),
        "pixel_interpolated_vector_magnitude": summarize(pixel_magnitude[valid]),
    }
    return colors, valid_np, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render R085 direction fields as smooth mesh-surface hue maps.")
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
    order_path = args.nested_order.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise RuntimeError("refusing to overwrite R085 surface direction output")
    if not checkpoint.is_file() or not base_image.is_file() or not order_path.is_file():
        raise RuntimeError("required surface-direction input is missing")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError(f"checkpoint SHA mismatch: {checkpoint_sha}")
    counts = tuple(sorted({int(value) for value in args.counts.split(",")}))
    order = np.load(order_path, allow_pickle=False).astype(np.int64)
    if not counts or min(counts) <= 8 or order.size < max(counts):
        raise RuntimeError("invalid coarse counts or nested order")
    staging.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    started = time.perf_counter()

    print("LOAD_CHECKPOINT_START", flush=True)
    model, config, checkpoint_payload = stage1.load_stage1_checkpoint_model(checkpoint, device)
    print("LOAD_CHECKPOINT_DONE", flush=True)
    guide_points = model.guide_points_local.detach()
    guide_normals, _guide_tangent, _guide_bitangent = model.guide_normals_and_tangent_frames()
    clean_direction = F.normalize(model.guide_clean_flow_direction_target.detach(), dim=-1, eps=1.0e-8)
    learned_direction = F.normalize(model.guide_direction_world().detach(), dim=-1, eps=1.0e-8)
    vertices_local = model.vertices.detach()
    faces = model.faces.detach().long()
    normals_vertex, vertex_normal_report = vertex_normals(vertices_local, faces)
    vertices_world = (
        vertices_local * torch.exp(model.log_scale.detach()).view(1, 1)
        + model.translation.detach().view(1, 3)
    )

    field_config = GuideGaussianFieldConfig(neighbor_count=8)
    print("BUILD_VERTEX_BINDING_START", flush=True)
    vertex_binding = initialize_guide_gaussian_binding(
        guide_points,
        vertices_local,
        field_config,
        device="cpu",
        dtype=torch.float32,
    )
    vertex_field = GuideAttributeGaussianField(_binding_to_device(vertex_binding, device))
    print("BUILD_VERTEX_BINDING_DONE", flush=True)
    clean_vertex, clean_magnitude = transported_direction_field(
        vertex_field,
        clean_direction,
        guide_normals,
        normals_vertex,
    )
    learned_vertex, learned_magnitude = transported_direction_field(
        vertex_field,
        learned_direction,
        guide_normals,
        normals_vertex,
    )

    graph = model.guide_surface_interpolator().source_neighbor_graph(12)
    graph_edges = graph.edges.detach().cpu().numpy().astype(np.int64)
    graph_lengths = graph.distances.detach().cpu().numpy().astype(np.float64)
    topology = _undirected_graph(graph_edges, graph_lengths, int(guide_points.shape[0]))
    selected_to_query = np.asarray(
        dijkstra(topology, directed=False, indices=order[: max(counts)]),
        dtype=np.float64,
    )
    variants: list[tuple[str, str, torch.Tensor, torch.Tensor, dict[str, object]]] = [
        (
            "clean_flow_4500",
            "fixed 4500 clean-flow base",
            clean_vertex,
            clean_magnitude,
            {"relative_to_clean_flow": direction_difference(clean_vertex, clean_vertex)},
        ),
        (
            "learned_4k_4500",
            "learned R080 4k guide direction through Gaussian K8",
            learned_vertex,
            learned_magnitude,
            {"relative_to_clean_flow": direction_difference(learned_vertex, clean_vertex)},
        ),
    ]
    for count in counts:
        selected = np.asarray(order[:count], dtype=np.int64)
        coarse_binding, _spacing, binding_report = _topology_binding(
            guide_points.cpu(),
            guide_points.cpu(),
            selected,
            selected_to_query[:count],
            field_config,
        )
        coarse_field = GuideAttributeGaussianField(_binding_to_device(coarse_binding, device))
        selected_tensor = torch.as_tensor(selected, device=device, dtype=torch.long)
        reconstructed_guide, coarse_magnitude = transported_direction_field(
            coarse_field,
            clean_direction[selected_tensor],
            guide_normals[selected_tensor],
            guide_normals,
        )
        reconstructed_vertex, vertex_magnitude = transported_direction_field(
            vertex_field,
            reconstructed_guide,
            guide_normals,
            normals_vertex,
        )
        variants.append(
            (
                f"stress_m{count}",
                f"clean-flow bandwidth stress M={count}",
                reconstructed_vertex,
                vertex_magnitude,
                {
                    "coarse_binding": binding_report,
                    "coarse_guide_pre_normalization_magnitude": summarize(coarse_magnitude),
                    "relative_to_clean_flow": direction_difference(
                        reconstructed_vertex,
                        clean_vertex,
                    ),
                },
            )
        )

    base = visualizer._read_base(
        base_image,
        int(config.expected_width),
        int(config.expected_height),
    )
    artifacts: list[Path] = []
    variant_reports = []
    for slug, label, direction, magnitude, extra in variants:
        colors, valid, raster_report = rasterize_direction_hue(
            model,
            config,
            vertices_world,
            direction,
            int(args.view_index),
            device,
        )
        image_path = staging / f"view09_direction_surface_{slug}.png"
        overlay_report = visualizer._overlay_direction_surface_colors(
            base,
            colors,
            valid,
            title=f"R085 view09 surface direction hue: {label}",
            out_path=image_path,
            alpha=1.0,
        )
        artifacts.append(image_path)
        variant_reports.append(
            {
                "slug": slug,
                "label": label,
                "image": image_path.name,
                "pre_normalization_magnitude": summarize(magnitude),
                "raster": raster_report,
                "overlay": overlay_report,
                **extra,
            }
        )

    report = {
        "schema": "anigroom.r085.direction_surface_maps.v1",
        "status": "complete",
        "training": False,
        "visualization_contract": {
            "field_queries": "mesh vertices",
            "surface_interpolation": "shared triangle barycentric rasterization",
            "direction_encoding": "interpolated unit screen vector then cyclic hue",
            "post_blur": False,
            "point_stamping": False,
            "base_image_alpha_inside_surface": 0.0,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "iteration": int(checkpoint_payload["iteration"]),
        },
        "nested_order": {"path": str(order_path), "sha256": sha256_file(order_path)},
        "counts": {
            "guide": int(guide_points.shape[0]),
            "mesh_vertex": int(vertices_local.shape[0]),
            "mesh_face": int(faces.shape[0]),
            "coarse_candidates": list(counts),
        },
        "vertex_normals": vertex_normal_report,
        "vertex_binding": vertex_binding.report,
        "variants": variant_reports,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report_path = staging / "r085_direction_surface_maps.json"
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

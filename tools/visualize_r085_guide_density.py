from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
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
from anigroom.flow.direction_geometry import parallel_transport_vectors  # noqa: E402
from anigroom.projection.mesh_visibility import (  # noqa: E402
    render_mesh_vertex_color_from_tensors,
)
from tools import train_white_tiger_stage1 as stage1  # noqa: E402
from tools import visualize_white_tiger_groom_attributes as visualizer  # noqa: E402
from tools.diagnose_nested_guide_gaussian_counts import (  # noqa: E402
    _binding_to_device,
    _visible_projection,
    summarize,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def rasterize_scalar_field(
    model,
    config,
    vertices_world: torch.Tensor,
    vertex_values: torch.Tensor,
    view_index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    data_root = stage1.resolve_project_path(config.data_root)
    viewmats, ks = stage1.load_camera_tensors(data_root, device)
    viewmat, k = viewmats[view_index], ks[view_index]
    width, height = int(config.expected_width), int(config.expected_height)
    lo = float(vertex_values.min().cpu())
    hi = float(vertex_values.max().cpu())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise RuntimeError("vertex scalar field has no finite range")
    normalized = ((vertex_values - lo) / (hi - lo)).clamp(0.0, 1.0)
    packed = normalized[:, None].expand(-1, 3).contiguous()
    import nvdiffrast.torch as dr

    colors, valid = render_mesh_vertex_color_from_tensors(
        vertices_world,
        model.faces.to(device=device, dtype=torch.int32),
        packed,
        viewmat,
        k,
        width,
        height,
        device=device,
        ctx=dr.RasterizeCudaContext(device=device),
        background=torch.zeros(3, device=device),
    )
    scalar = colors[..., 0] * (hi - lo) + lo
    return scalar.detach().cpu().numpy(), valid.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize R085 primary-guide positions and density.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--base-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-index", type=int, default=9)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    base_image = args.base_image.resolve()
    output = args.output_dir.resolve()
    staging = output.with_name(output.name + ".staging")
    if output.exists() or staging.exists():
        raise RuntimeError("refusing to overwrite R085 guide-density output")
    if not checkpoint.is_file() or not base_image.is_file():
        raise RuntimeError("required guide-density input is missing")
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
    tree = cKDTree(guide_np)
    distances, ids = tree.query(guide_np, k=9, workers=1)
    distances = np.asarray(distances, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    if distances.shape != (guide_count, 9) or not np.all(ids[:, 0] == np.arange(guide_count)):
        raise RuntimeError("guide Euclidean KNN has unexpected self ordering")
    d1 = distances[:, 1]
    d4 = distances[:, 4]
    d8 = distances[:, 8]

    graph = model.guide_surface_interpolator().source_neighbor_graph(8)
    topology_distances = graph.distances.reshape(guide_count, 8).detach().cpu().numpy().astype(np.float64)
    topology_d1 = topology_distances[:, 0]
    topology_d4 = topology_distances[:, 3]
    topology_d8 = topology_distances[:, 7]
    topology_euclidean_ratio = topology_d8 / np.maximum(d8, 1.0e-12)
    log_density = -2.0 * np.log(np.maximum(d8, 1.0e-12))
    support_radius = d8.copy()
    anisotropy_proxy = d8 / np.maximum(d1, 1.0e-12)

    field_config = GuideGaussianFieldConfig(neighbor_count=8)
    vertex_binding = initialize_guide_gaussian_binding(
        guide_points,
        model.vertices.detach(),
        field_config,
        device="cpu",
        dtype=torch.float32,
    )
    vertex_field = GuideAttributeGaussianField(_binding_to_device(vertex_binding, device))
    values = {
        "euclidean_d8": d8,
        "topology_d8": topology_d8,
        "topology_euclidean_ratio": topology_euclidean_ratio,
        "log_density": log_density,
        "d8_over_d1": anisotropy_proxy,
    }
    vertex_values = {
        name: vertex_field(torch.as_tensor(value, device=device, dtype=torch.float32))
        for name, value in values.items()
    }
    vertices_world = (
        model.vertices.detach() * torch.exp(model.log_scale.detach()).view(1, 1)
        + model.translation.detach().view(1, 3)
    )
    base = visualizer._read_base(
        base_image,
        int(config.expected_width),
        int(config.expected_height),
    )
    guide_world = (
        guide_points * torch.exp(model.log_scale.detach()).view(1, 1)
        + model.translation.detach().view(1, 3)
    )
    guide_xy, visible_guide_ids = _visible_projection(
        model,
        config,
        guide_world,
        int(args.view_index),
        device,
    )

    artifacts: list[Path] = []
    point_path = staging / "view09_guide_root_positions_d8_4500.png"
    point_lo = float(np.quantile(d8, 0.02))
    point_hi = float(np.quantile(d8, 0.98))
    point_report = visualizer._overlay_points(
        base,
        guide_xy,
        d8[visible_guide_ids],
        title="R085 view09 all 4500 primary guide roots; color = Euclidean d8",
        out_path=point_path,
        radius=3,
        lo=point_lo,
        hi=point_hi,
    )
    artifacts.append(point_path)

    surface_reports = {}
    titles = {
        "euclidean_d8": "R085 view09 Gaussian nominal support radius d8",
        "topology_d8": "R085 view09 topology K8 distance",
        "topology_euclidean_ratio": "R085 view09 topology / Euclidean d8 ratio",
        "log_density": "R085 view09 guide log-density proxy -2 log(d8)",
        "d8_over_d1": "R085 view09 local spacing spread d8 / d1",
    }
    ranges = {
        name: (
            float(np.quantile(value, 0.02)),
            float(np.quantile(value, 0.98)),
        )
        for name, value in values.items()
    }
    for name, vertex_value in vertex_values.items():
        scalar, valid = rasterize_scalar_field(
            model,
            config,
            vertices_world,
            vertex_value,
            int(args.view_index),
            device,
        )
        image_path = staging / f"view09_guide_{name}_surface.png"
        lo, hi = ranges[name]
        surface_reports[name] = visualizer._overlay_scalar_surface_values(
            base,
            scalar,
            valid,
            title=titles[name],
            out_path=image_path,
            lo=lo,
            hi=hi,
            alpha=1.0,
        )
        surface_reports[name]["image"] = image_path.name
        artifacts.append(image_path)

    histogram_path = staging / "guide_spacing_histograms.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    axes[0].hist(d1, bins=80, alpha=0.65, label="Euclidean d1")
    axes[0].hist(d4, bins=80, alpha=0.65, label="Euclidean d4")
    axes[0].hist(d8, bins=80, alpha=0.65, label="Euclidean d8")
    axes[0].set_xlabel("distance")
    axes[0].set_ylabel("guide count")
    axes[0].set_title("Primary-guide Euclidean spacing")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].hist(topology_euclidean_ratio, bins=80, color="#5a4fa3", alpha=0.85)
    axes[1].set_xlabel("topology d8 / Euclidean d8")
    axes[1].set_ylabel("guide count")
    axes[1].set_title("Topology-to-chord spacing ratio")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(histogram_path)
    plt.close(fig)
    artifacts.append(histogram_path)

    region = (model.guide_region_weight.detach().reshape(-1).cpu().numpy() > 0.5)
    guide_normals, _guide_tangent, _guide_bitangent = model.guide_normals_and_tangent_frames()
    clean_direction = F.normalize(
        model.guide_clean_flow_direction_target.detach(),
        dim=-1,
        eps=1.0e-8,
    )
    nearest_ids = torch.as_tensor(ids[:, 1], device=device, dtype=torch.long)
    transported_nearest = parallel_transport_vectors(
        clean_direction[nearest_ids],
        guide_normals[nearest_ids],
        guide_normals,
    )
    nearest_angle = torch.rad2deg(
        torch.acos(
            (clean_direction * transported_nearest).sum(dim=-1).clamp(-1.0, 1.0)
        )
    ).detach().cpu().numpy()
    cross_region = region != region[ids[:, 1]]
    median_d1 = float(np.median(d1))
    close = d1 < 0.5 * median_d1
    extreme_spread = anisotropy_proxy > 3.0

    def nearest_subset(mask: np.ndarray) -> dict[str, object]:
        return {
            "count": int(mask.sum()),
            "cross_region_count": int((mask & cross_region).sum()),
            "distance": summarize(d1[mask]) if bool(mask.any()) else None,
            "direction_angle_deg": (
                summarize(nearest_angle[mask]) if bool(mask.any()) else None
            ),
        }

    region_reports = {
        "region_weight_zero": {
            "count": int((~region).sum()),
            "euclidean_d8": summarize(d8[~region]),
            "topology_d8": summarize(topology_d8[~region]),
        },
        "region_weight_one": {
            "count": int(region.sum()),
            "euclidean_d8": summarize(d8[region]),
            "topology_d8": summarize(topology_d8[region]),
        },
        "note": "historical input-label audit only; not an R085 sampling rule",
    }
    report = {
        "schema": "anigroom.r085.primary_guide_density.v1",
        "status": "complete",
        "training": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "iteration": int(checkpoint_payload["iteration"]),
        },
        "counts": {
            "guide": guide_count,
            "visible_view09": int(visible_guide_ids.size),
            "mesh_vertex": int(model.vertices.shape[0]),
            "mesh_face": int(model.faces.shape[0]),
        },
        "spacing": {
            "euclidean_d1": summarize(d1),
            "euclidean_d4": summarize(d4),
            "euclidean_d8": summarize(d8),
            "topology_d1": summarize(topology_d1),
            "topology_d4": summarize(topology_d4),
            "topology_d8": summarize(topology_d8),
            "topology_euclidean_d8_ratio": summarize(topology_euclidean_ratio),
            "log_density_proxy": summarize(log_density),
            "d8_over_d1": summarize(anisotropy_proxy),
        },
        "extremes": {
            "densest_ids": np.argsort(d8, kind="stable")[:16].tolist(),
            "sparsest_ids": np.argsort(d8, kind="stable")[-16:].tolist(),
        },
        "historical_region_labels": region_reports,
        "nearest_direction_attribution": {
            "all": nearest_subset(np.ones((guide_count,), dtype=bool)),
            "cross_region": nearest_subset(cross_region),
            "distance_below_half_global_median": nearest_subset(close),
            "d8_over_d1_above_3": nearest_subset(extreme_spread),
            "global_median_d1": median_d1,
            "note": (
                "region labels are historical evidence only; thresholds are "
                "diagnostic summaries and not a proposed sampling rule"
            ),
        },
        "vertex_binding": vertex_binding.report,
        "point_overlay": {**point_report, "image": point_path.name},
        "surface_maps": surface_reports,
        "histogram": histogram_path.name,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report_path = staging / "r085_primary_guide_density.json"
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

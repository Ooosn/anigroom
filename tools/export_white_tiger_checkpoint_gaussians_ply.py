from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.mesh_roots import read_obj_mesh  # noqa: E402
from tools.train_white_tiger_stage1 import (  # noqa: E402
    Stage1Config,
    WhiteTigerStage1Model,
    dense_groom_ranges,
    face_normals_np,
    guide_residual_multiplier_for_iteration,
    resolve_project_path,
    shape_detail_multiplier_for_iteration,
    stage1_config_from_checkpoint_mapping,
)

C0 = 0.28209479177387814
EPS = 1.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real training Gaussians from a White Tiger Stage1 checkpoint as 3DGS PLY.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=-1, help="Use checkpoint config when negative.")
    parser.add_argument("--min-segments", type=int, default=-1)
    parser.add_argument("--segment-length-origin", type=float, default=-1.0)
    parser.add_argument("--segments-per-unit-length", type=float, default=-1.0)
    parser.add_argument("--segments-per-unit-complexity", type=float, default=-1.0)
    parser.add_argument("--child-count", type=int, default=-1)
    parser.add_argument("--length-overlap", type=float, default=-1.0)
    parser.add_argument("--max-gaussians", type=int, default=0, help="0 exports all; positive exports a deterministic subset.")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--sh-degree", type=int, default=3)
    return parser.parse_args()


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def inverse_sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, EPS, 1.0 - EPS)
    return np.log(x / (1.0 - x))


def build_model(checkpoint: dict, config: Stage1Config, device: torch.device) -> WhiteTigerStage1Model:
    state = checkpoint["model"]
    mesh = read_obj_mesh(resolve_project_path(config.mesh_path))
    face_ids = tensor_to_numpy(state["face_ids"]).astype(np.int64)
    if "bary_initial" in state:
        barycentric = tensor_to_numpy(state["bary_initial"]).astype(np.float32)
    else:
        barycentric = tensor_to_numpy(torch.softmax(state["bary_logits"], dim=-1)).astype(np.float32)

    guide_face_ids = None
    guide_barycentric = None
    if "guide_face_ids" in state and state["guide_face_ids"].numel() > 0:
        guide_face_ids = tensor_to_numpy(state["guide_face_ids"]).astype(np.int64)
        guide_barycentric = tensor_to_numpy(state["guide_barycentric"]).astype(np.float32)
    face_tangents_tensor = state.get("face_tangents")
    face_tangents = None
    if face_tangents_tensor is not None and int(face_tangents_tensor.numel()) > 0:
        face_tangents = tensor_to_numpy(face_tangents_tensor).astype(np.float32)

    model = WhiteTigerStage1Model(
        mesh,
        face_normals_np(mesh),
        face_tangents,
        face_ids,
        barycentric,
        dense_groom_ranges(),
        device,
        init_scale=config.init_mesh_scale,
        init_translation=config.init_mesh_translation,
        init_groom_length=config.init_groom_length,
        max_child_count=config.child_count,
        local_child_color_support=config.local_child_color_support,
        local_child_color_scale=config.local_child_color_scale,
        guide_face_ids=guide_face_ids,
        guide_barycentric=guide_barycentric,
        guide_interpolation_k=config.guide_interpolation_k,
        render_geometry_parameterization=config.render_geometry_parameterization,
        guide_length_residual_scale=config.guide_length_residual_scale,
        guide_direction_residual_scale=config.guide_direction_residual_scale,
        guide_width_residual_scale=config.guide_width_residual_scale,
        guide_child_radius_residual_scale=config.guide_child_radius_residual_scale,
        guide_clump_residual_scale=config.guide_clump_residual_scale,
        guide_curl_residual_scale=config.guide_curl_residual_scale,
        guide_frizz_residual_scale=config.guide_frizz_residual_scale,
        shape_curl_scale=getattr(config, "shape_curl_scale", 1.0),
        shape_frizz_scale=getattr(config, "shape_frizz_scale", 1.0),
        strand_shape_normal_mode=getattr(config, "strand_shape_normal_mode", "full"),
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint load failed: missing={missing}, unexpected={unexpected}")
    iteration = int(checkpoint.get("iteration", 0))
    model.guide_residual_multiplier = guide_residual_multiplier_for_iteration(config, iteration)
    model.shape_detail_multiplier = shape_detail_multiplier_for_iteration(config, iteration)
    model.eval()
    return model


def write_3dgs_ply(path: Path, means: np.ndarray, quats: np.ndarray, scales: np.ndarray, colors: np.ndarray, opacities: np.ndarray, sh_degree: int) -> None:
    if sh_degree < 0:
        raise ValueError("sh_degree must be non-negative")
    n = int(means.shape[0])
    f_rest_count = 3 * ((int(sh_degree) + 1) ** 2 - 1)
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names.extend(f"f_rest_{i}" for i in range(f_rest_count))
    names.append("opacity")
    names.extend(["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])
    elements = np.empty(n, dtype=[(name, "f4") for name in names])

    f_dc = (np.clip(colors, 0.0, 1.0) - 0.5) / C0
    opacity_raw = inverse_sigmoid_np(opacities.reshape(-1, 1))
    scale_raw = np.log(np.clip(scales, EPS, None))
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True).clip(EPS, None)

    elements["x"] = means[:, 0]
    elements["y"] = means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["f_dc_0"] = f_dc[:, 0]
    elements["f_dc_1"] = f_dc[:, 1]
    elements["f_dc_2"] = f_dc[:, 2]
    for i in range(f_rest_count):
        elements[f"f_rest_{i}"] = 0.0
    elements["opacity"] = opacity_raw[:, 0]
    elements["scale_0"] = scale_raw[:, 0]
    elements["scale_1"] = scale_raw[:, 1]
    elements["scale_2"] = scale_raw[:, 2]
    elements["rot_0"] = quats[:, 0]
    elements["rot_1"] = quats[:, 1]
    elements["rot_2"] = quats[:, 2]
    elements["rot_3"] = quats[:, 3]

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = stage1_config_from_checkpoint_mapping(checkpoint["config"])
    model = build_model(checkpoint, config, device)

    samples = int(config.samples if args.samples < 0 else args.samples)
    min_segments = int(config.min_segments if args.min_segments < 0 else args.min_segments)
    segment_length_origin = float(
        config.segment_length_origin if args.segment_length_origin < 0 else args.segment_length_origin
    )
    segments_per_unit_length = float(
        config.segments_per_unit_length if args.segments_per_unit_length < 0 else args.segments_per_unit_length
    )
    segments_per_unit_complexity = float(
        config.segments_per_unit_complexity
        if args.segments_per_unit_complexity < 0
        else args.segments_per_unit_complexity
    )
    child_count = int(config.child_count if args.child_count < 0 else args.child_count)
    length_overlap = float(config.gaussian_length_overlap if args.length_overlap <= 0 else args.length_overlap)

    with torch.no_grad():
        gaussians, _, _, stats = model.render_parameters(
            samples=samples,
            child_count=child_count,
            min_segments=min_segments,
            segment_length_origin=segment_length_origin,
            segments_per_unit_length=segments_per_unit_length,
            segments_per_unit_complexity=segments_per_unit_complexity,
            length_overlap=length_overlap,
        )
        if args.max_gaussians > 0 and int(gaussians.means.shape[0]) > int(args.max_gaussians):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(args.seed))
            ids = torch.randperm(int(gaussians.means.shape[0]), generator=gen)[: int(args.max_gaussians)].to(device=gaussians.means.device)
            means = gaussians.means[ids]
            quats = gaussians.quats[ids]
            scales = gaussians.scales[ids]
            colors = gaussians.colors[ids]
            opacities = gaussians.opacities[ids]
        else:
            means = gaussians.means
            quats = gaussians.quats
            scales = gaussians.scales
            colors = gaussians.colors
            opacities = gaussians.opacities

    output_path = Path(args.output)
    write_3dgs_ply(
        output_path,
        tensor_to_numpy(means).astype(np.float32),
        tensor_to_numpy(quats).astype(np.float32),
        tensor_to_numpy(scales).astype(np.float32),
        tensor_to_numpy(colors).astype(np.float32),
        tensor_to_numpy(opacities).astype(np.float32),
        sh_degree=int(args.sh_degree),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "iteration": int(checkpoint.get("iteration", -1)),
        "exported_gaussians": int(means.shape[0]),
        "full_gaussians": int(gaussians.means.shape[0]),
        "samples": samples,
        "min_segments": min_segments,
        "segment_length_origin": segment_length_origin,
        "segments_per_unit_length": segments_per_unit_length,
        "segments_per_unit_complexity": segments_per_unit_complexity,
        "child_count": child_count,
        "length_overlap": length_overlap,
        "stats": stats,
        "format": "3DGS binary PLY, SH degree 3 with zero f_rest by default",
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

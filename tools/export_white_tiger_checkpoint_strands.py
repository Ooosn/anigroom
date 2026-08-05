from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import build_strands, expand_child_strands  # noqa: E402
from anigroom.mesh_roots import read_obj_mesh  # noqa: E402
from tools.train_white_tiger_stage1 import (  # noqa: E402
    Stage1Config,
    WhiteTigerStage1Model,
    dense_groom_ranges,
    face_normals_np,
    guide_coverage_residual_multiplier_for_iteration,
    guide_residual_multiplier_for_iteration,
    resolve_project_path,
    shape_detail_multiplier_for_iteration,
    stage1_config_from_checkpoint_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real strand curves from a White Tiger Stage1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--child-count", type=int, default=-1, help="Use checkpoint config when negative.")
    parser.add_argument("--max-strands", type=int, default=0, help="0 exports all strands; positive value exports a deterministic subset.")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--uniform-color", nargs=3, type=float, default=[0.82, 0.80, 0.72])
    return parser.parse_args()


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_project_path(args.checkpoint)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = stage1_config_from_checkpoint_mapping(checkpoint["config"])
    state = checkpoint["model"]

    mesh = read_obj_mesh(resolve_project_path(config.mesh_path))
    face_normals = face_normals_np(mesh)
    face_tangents = None
    if config.face_tangent_field:
        tangent_path = resolve_project_path(config.face_tangent_field)
        face_tangents = np.load(tangent_path).astype(np.float32)
        if face_tangents.shape != (mesh.face_count, 3):
            raise RuntimeError(f"face tangent field shape mismatch: {face_tangents.shape} != {(mesh.face_count, 3)}")
        tangent_norm = np.linalg.norm(face_tangents, axis=-1, keepdims=True)
        face_tangents = face_tangents / np.maximum(tangent_norm, 1.0e-8)
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

    model = WhiteTigerStage1Model(
        mesh,
        face_normals,
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
        guide_bend_residual_scale=config.guide_bend_residual_scale,
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
    model.guide_coverage_residual_multiplier = guide_coverage_residual_multiplier_for_iteration(config, iteration)
    model.shape_detail_multiplier = shape_detail_multiplier_for_iteration(config, iteration)
    model.eval()

    with torch.no_grad():
        roots, normals, roots_local = model.roots_and_normals()
        tangents, bitangents = model.tangent_frames(normals)
        groom = model.apply_guide_controls(model.groom.decode(), roots_local)
        strands, widths, colors, opacities = build_strands(
            roots,
            normals,
            tangents,
            bitangents,
            groom,
            samples=int(args.samples),
            shape_normal_mode=getattr(config, "strand_shape_normal_mode", "full"),
        )
        child_count = int(config.child_count if args.child_count < 0 else args.child_count)
        strands, widths, colors, opacities, root_ids = expand_child_strands(
            strands,
            widths,
            colors,
            opacities,
            normals,
            groom.child_radius,
            groom.clump_strength,
            child_count=child_count,
        )
        if args.max_strands > 0 and int(strands.shape[0]) > int(args.max_strands):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(args.seed))
            ids = torch.randperm(int(strands.shape[0]), generator=gen)[: int(args.max_strands)].to(device=strands.device)
            strands = strands[ids]
            widths = widths[ids]
            colors = colors[ids]
            opacities = opacities[ids]
            root_ids = root_ids[ids]
        uniform = torch.tensor(args.uniform_color, device=device, dtype=strands.dtype).view(1, 1, 3)
        colors = uniform.expand_as(colors).contiguous()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        strands=tensor_to_numpy(strands).astype(np.float32),
        widths=tensor_to_numpy(widths).astype(np.float32),
        colors=tensor_to_numpy(colors).astype(np.float32),
        opacities=tensor_to_numpy(opacities).astype(np.float32),
        root_ids=tensor_to_numpy(root_ids).astype(np.int64),
        iteration=np.asarray([int(checkpoint.get("iteration", -1))], dtype=np.int64),
        source_checkpoint=np.asarray([str(checkpoint_path)], dtype=object),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "iteration": int(checkpoint.get("iteration", -1)),
        "guide_residual_multiplier": float(model.guide_residual_multiplier),
        "shape_detail_multiplier": float(model.shape_detail_multiplier),
        "root_count": int(face_ids.shape[0]),
        "guide_root_count": int(0 if guide_face_ids is None else guide_face_ids.shape[0]),
        "child_count": child_count,
        "strand_count": int(strands.shape[0]),
        "samples": int(strands.shape[1]),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

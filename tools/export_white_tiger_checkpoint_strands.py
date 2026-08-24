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
from tools.train_white_tiger_stage1 import (  # noqa: E402
    build_stage1_model_from_checkpoint,
    load_training_checkpoint,
    resolve_project_path,
    stage1_config_from_checkpoint_mapping,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real strand curves from a White Tiger Stage1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument(
        "--root-domain",
        choices=("render", "secondary_guide"),
        default="render",
        help="Export rendered roots or the persistent secondary geometry-control field.",
    )
    parser.add_argument("--child-count", type=int, default=-1, help="Use checkpoint config when negative.")
    parser.add_argument("--max-strands", type=int, default=0, help="0 exports all strands; positive value exports a deterministic subset.")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--uniform-color", nargs=3, type=float, default=[0.82, 0.80, 0.72])
    parser.add_argument(
        "--shape-detail-multiplier",
        type=float,
        default=None,
        help="Override the checkpoint curl multiplier for diagnostic export.",
    )
    parser.add_argument(
        "--guide-residual-multiplier",
        type=float,
        default=None,
        help="Override the checkpoint render-root/secondary geometry residual multiplier for diagnostic export.",
    )
    parser.add_argument(
        "--secondary-shape-residual-multiplier",
        type=float,
        default=None,
        help="Override only the secondary curl residual multiplier.",
    )
    return parser.parse_args()


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = load_training_checkpoint(checkpoint_path)
    config = stage1_config_from_checkpoint_mapping(checkpoint["config"])
    model = build_stage1_model_from_checkpoint(checkpoint, config, device)
    if args.shape_detail_multiplier is not None:
        model.shape_detail_multiplier = float(args.shape_detail_multiplier)
    if args.guide_residual_multiplier is not None:
        model.guide_residual_multiplier = float(args.guide_residual_multiplier)
    if args.secondary_shape_residual_multiplier is not None:
        model.secondary_shape_residual_multiplier = float(
            args.secondary_shape_residual_multiplier
        )

    with torch.no_grad():
        if args.root_domain == "render":
            roots, normals, roots_local = model.roots_and_normals()
            tangents, bitangents = model.tangent_frames(normals)
            groom = model.apply_guide_controls(model.groom.decode(), roots_local)
            child_count = int(
                config.child_count if args.child_count < 0 else args.child_count
            )
        else:
            if not model.secondary_guides_enabled():
                raise RuntimeError(
                    "secondary-guide export requires a checkpoint with persistent secondary guides"
                )
            if args.child_count not in (-1, 1):
                raise ValueError(
                    "secondary-guide control strands do not support child expansion"
                )
            roots_local = model.secondary_guide_points_local
            roots = (
                roots_local * torch.exp(model.log_scale).view(1, 1)
                + model.translation.view(1, 3)
            )
            normals, tangents, bitangents = model.tangent_frames_for_face_ids(
                model.secondary_guide_face_ids
            )
            groom = model.secondary_effective_groom()
            child_count = 1
        strands, widths, colors, opacities = build_strands(
            roots,
            normals,
            tangents,
            bitangents,
            groom,
            samples=int(args.samples),
        )
        if args.root_domain == "render":
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
        else:
            root_ids = torch.arange(
                strands.shape[0], device=strands.device, dtype=torch.long
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
        source_checkpoint=np.asarray([str(checkpoint_path)]),
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "iteration": int(checkpoint.get("iteration", -1)),
        "guide_residual_multiplier": float(model.guide_residual_multiplier),
        "shape_detail_multiplier": float(model.shape_detail_multiplier),
        "secondary_shape_residual_multiplier": float(
            model.secondary_shape_residual_multiplier
        ),
        "root_domain": str(args.root_domain),
        "root_count": int(model.face_ids.shape[0]),
        "guide_root_count": int(model.guide_face_ids.shape[0]),
        "child_count": child_count,
        "strand_count": int(strands.shape[0]),
        "samples": int(strands.shape[1]),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

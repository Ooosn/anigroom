"""Render plain gsplat baseline checkpoints on fixed train/test views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from gsplat.rendering import rasterization

from anigroom.baselines.plain_gsplat import PlainGsplatModel, load_camera_tensors, read_obj_vertices
from anigroom.data.white_tiger import build_stage1_input_report


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0], mode="L")
    return Image.fromarray(arr, mode="RGB")


def parse_indices(text: str, train_indices: list[int], test_indices: list[int], image_count: int) -> list[int]:
    if text == "test":
        return list(test_indices)
    if text == "train":
        return list(train_indices)
    if text == "all":
        return list(range(image_count))
    return [int(part) for part in text.split(",") if part.strip()]


def make_contact_sheet(paths: list[Path], out_path: Path, tile_width: int = 320) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_width, int(tile_width * 9 / 16)), Image.Resampling.LANCZOS)
        thumbs.append((path.stem, image.copy()))
    cols = min(3, len(thumbs))
    label_h = 24
    tile_h = max(image.height for _, image in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_width, rows * (tile_h + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, image) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = col * tile_width + (tile_width - image.width) // 2
        y = row * (tile_h + label_h)
        sheet.paste(image, (x, y))
        draw.text((col * tile_width + 8, y + tile_h + 4), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-root",
        default="data/neuralfur_work/whiteTiger_processed/roaringwalk",
    )
    parser.add_argument(
        "--mesh-path",
        default="data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", default="test")
    parser.add_argument("--test-stride", type=int, default=6)
    parser.add_argument("--save-alpha", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("rendering requires CUDA")
    device = torch.device("cuda")
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    render_dir = output_dir / "renders"
    alpha_dir = output_dir / "alpha"
    render_dir.mkdir(parents=True, exist_ok=True)
    if args.save_alpha:
        alpha_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    data_root = Path(args.data_root)
    mesh_path = Path(args.mesh_path)
    report = build_stage1_input_report(data_root, mesh_path, test_stride=args.test_stride)
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    width, height = report.image_size or (0, 0)
    viewmats, ks = load_camera_tensors(data_root, device)
    vertices = read_obj_vertices(mesh_path)
    model = PlainGsplatModel(vertices, int(config["num_gaussians"]), int(config["seed"]), device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    indices = parse_indices(args.indices, report.train_indices, report.test_indices, report.image_count)
    background = torch.ones((1, 3), device=device) if bool(config.get("white_background", True)) else None

    means, quats, scales, opacities, colors = model.parameters_for_render()
    written = []
    with torch.no_grad():
        for idx in indices:
            render, alpha, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats[idx : idx + 1],
                ks[idx : idx + 1],
                width,
                height,
                packed=False,
                backgrounds=background,
            )
            out_path = render_dir / f"view_{idx:04d}.png"
            tensor_to_image(render[0]).save(out_path)
            written.append(out_path)
            if args.save_alpha:
                tensor_to_image(alpha[0]).save(alpha_dir / f"view_{idx:04d}.png")

    make_contact_sheet(written, output_dir / "contact_sheet.jpg")
    (output_dir / "render_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "indices": indices,
                "render_count": len(written),
                "output_dir": str(output_dir),
                "num_gaussians": int(config["num_gaussians"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print({"render_count": len(written), "output_dir": str(output_dir), "contact_sheet": str(output_dir / "contact_sheet.jpg")})


if __name__ == "__main__":
    main()

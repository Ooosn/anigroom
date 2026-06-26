from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def image_to_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def psnr(pred: Image.Image, gt: Image.Image) -> float:
    pred_arr = image_to_float(pred)
    gt_arr = image_to_float(gt)
    mse = float(np.mean((pred_arr - gt_arr) ** 2))
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def diff_x4(pred: Image.Image, gt: Image.Image) -> Image.Image:
    pred_arr = image_to_float(pred)
    gt_arr = image_to_float(gt)
    diff = np.clip(np.abs(pred_arr - gt_arr) * 4.0, 0.0, 1.0)
    return Image.fromarray((diff * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def fit_tile(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=(0, 0, 0), font=font)


def make_panel(gs3: Image.Image, ours: Image.Image, gt: Image.Image, output: Path, gs3_label: str) -> None:
    if not (gs3.size == ours.size == gt.size):
        raise ValueError(f"image sizes differ: 3dgs={gs3.size}, ours={ours.size}, gt={gt.size}")

    tile_w = 480
    label_h = 42
    pad = 18
    tiles = [
        (f"{gs3_label}  PSNR {psnr(gs3, gt):.2f}", gs3),
        (f"Ours  PSNR {psnr(ours, gt):.2f}", ours),
        ("GT", gt),
        ("Ours diff x4", diff_x4(ours, gt)),
    ]
    fitted = [(label, fit_tile(image, tile_w)) for label, image in tiles]
    tile_h = max(image.height for _, image in fitted)
    panel = Image.new("RGB", (len(fitted) * tile_w + (len(fitted) + 1) * pad, tile_h + label_h + 2 * pad), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    for i, (label, image) in enumerate(fitted):
        x = pad + i * (tile_w + pad)
        y = pad + label_h
        draw_label(draw, (x, pad), label)
        panel.paste(image, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the formal full-res RGB view09 panel. Requires a real 3DGS render; does not accept placeholders."
    )
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--gs3", required=True, type=Path)
    parser.add_argument("--gs3-label", default="3DGS")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    gt = load_rgb(args.gt)
    ours = load_rgb(args.ours)
    gs3 = load_rgb(args.gs3)
    make_panel(gs3=gs3, ours=ours, gt=gt, output=args.output, gs3_label=args.gs3_label)
    print({"output": str(args.output), "ours_psnr": psnr(ours, gt), "gs3_psnr": psnr(gs3, gt)})


if __name__ == "__main__":
    main()

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


EPS = 1e-8


def gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(float(sigma) * 3.0)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / float(sigma)).square())
    return kernel / kernel.sum().clamp_min(EPS)


def gaussian_blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel1d(sigma, x.device, x.dtype)
    pad = kernel.numel() // 2
    kx = kernel.view(1, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1)
    x = F.conv2d(x, kx, padding=(0, pad))
    x = F.conv2d(x, ky, padding=(pad, 0))
    return x


def build_gabor_kernels(
    bins: int,
    sigma_x: float,
    sigma_y: float,
    frequency: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = max(3, int(math.ceil(max(float(sigma_x), float(sigma_y)) * 4.0)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    angles = torch.linspace(0.0, math.pi * (bins - 1) / bins, bins, device=device, dtype=dtype)
    kernels = []
    for theta in angles:
        c = torch.cos(math.pi - theta)
        s = torch.sin(math.pi - theta)
        x_theta = xx * c + yy * s
        y_theta = -xx * s + yy * c
        envelope = torch.exp(
            -0.5 * ((x_theta / float(sigma_x)).square() + (y_theta / float(sigma_y)).square())
        )
        carrier = torch.cos(2.0 * math.pi * float(frequency) * x_theta)
        kernel = envelope * carrier
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.abs().sum().clamp_min(EPS)
        kernels.append(kernel)
    return torch.stack(kernels, dim=0)[:, None], angles


def resize_pair(image: Image.Image, mask: Image.Image, width: int) -> tuple[Image.Image, Image.Image]:
    if width <= 0:
        return image, mask
    scale = width / float(image.size[0])
    size = (width, int(round(image.size[1] * scale)))
    return image.resize(size, Image.Resampling.BILINEAR), mask.resize(size, Image.Resampling.BILINEAR)


def load_gray_image(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def colorize_orientation(angle_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rad = angle_u8.astype(np.float32)
    red = np.clip(1.0 - np.abs(rad - 0.0) / 45.0, 0.0, 1.0) + np.clip(
        1.0 - np.abs(rad - 180.0) / 45.0, 0.0, 1.0
    )
    green = np.clip(1.0 - np.abs(rad - 90.0) / 45.0, 0.0, 1.0)
    magenta = np.clip(1.0 - np.abs(rad - 45.0) / 45.0, 0.0, 1.0)
    teal = np.clip(1.0 - np.abs(rad - 135.0) / 45.0, 0.0, 1.0)
    rgb = (
        np.array([0.0, 0.0, 1.0], dtype=np.float32)[None, None] * red[..., None]
        + np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, None] * green[..., None]
        + np.array([1.0, 0.0, 1.0], dtype=np.float32)[None, None] * magenta[..., None]
        + np.array([1.0, 1.0, 0.0], dtype=np.float32)[None, None] * teal[..., None]
    )
    return (np.clip(rgb, 0.0, 1.0) * mask[..., None] * 255.0).astype(np.uint8)


def confidence_from_variance(variance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    var = np.maximum(variance.astype(np.float32) / (math.pi**2), 0.0)
    confidence = 1.0 / (var * var + 1e-7)
    valid = (mask > 0.1) & np.isfinite(confidence)
    if np.any(valid):
        norm = max(float(np.quantile(confidence[valid], 0.95)), 1e-6)
        confidence = confidence / norm
    else:
        confidence = np.zeros_like(confidence, dtype=np.float32)
    return np.clip(confidence, 0.0, 1.0) * (mask > 0.1).astype(np.float32)


def paste_fit(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    tile = image.copy()
    tile.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    px = x0 + (x1 - x0 - tile.width) // 2
    py = y0 + (y1 - y0 - tile.height) // 2
    canvas.paste(tile, (px, py))


def make_orientation_contact_sheet(
    records: list[dict[str, object]],
    out_path: Path,
    tile_w: int,
    max_images: int,
) -> None:
    if not records:
        return
    count = min(max_images, len(records))
    if count < len(records):
        idx = np.linspace(0, len(records) - 1, count).round().astype(int).tolist()
        chosen = [records[i] for i in idx]
    else:
        chosen = records
    cols = 3
    label_h = 26
    tile_h = int(round(tile_w * 9 / 16))
    row_h = tile_h + label_h
    sheet = Image.new("RGB", (cols * tile_w, row_h * len(chosen) + label_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    headers = ["image", "orientation", "confidence"]
    for c, header in enumerate(headers):
        draw.text((c * tile_w + 8, 5), header, fill=(0, 0, 0))
    for row, rec in enumerate(chosen):
        y = label_h + row * row_h
        image = Image.open(str(rec["image_path"])).convert("RGB")
        orient = Image.open(str(rec["vis_path"])).convert("RGB")
        confidence = Image.open(str(rec["confidence_vis_path"])).convert("L").convert("RGB")
        for col, img in enumerate([image, orient, confidence]):
            paste_fit(sheet, img, (col * tile_w, y, (col + 1) * tile_w, y + tile_h))
        label = (
            f"{rec['image']}  mask={float(rec['mask_fraction']):.3f} "
            f"var={float(rec['variance_mean']):.3f} conf={float(rec['confidence_mean']):.3f}"
        )
        draw.text((8, y + tile_h + 4), label, fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


@torch.no_grad()
def compute_orientation(
    gray_np: np.ndarray,
    mask_np: np.ndarray,
    bins: int,
    dog_low: float,
    dog_high: float,
    sigma_x: float,
    sigma_y: float,
    frequency: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtype = torch.float32
    gray = torch.from_numpy(gray_np.astype(np.float32))[None, None].to(device=device, dtype=dtype)
    dog = gaussian_blur2d(gray, dog_low) - gaussian_blur2d(gray, dog_high)
    kernels, angles = build_gabor_kernels(bins, sigma_x, sigma_y, frequency, device, dtype)
    response = F.conv2d(dog, kernels, padding=kernels.shape[-1] // 2).abs()[0]
    response_norm = response / response.sum(dim=0, keepdim=True).clamp_min(EPS)
    best = response.argmax(dim=0)
    best_angle = angles[best]
    dists = torch.minimum(
        (best_angle[None] - angles[:, None, None]).abs(),
        torch.minimum(
            (best_angle[None] - angles[:, None, None] - math.pi).abs(),
            (best_angle[None] - angles[:, None, None] + math.pi).abs(),
        ),
    )
    variance = (dists.square() * response_norm).sum(dim=0).clamp(0.0, math.pi * math.pi)
    mask = torch.from_numpy(mask_np.astype(np.float32)).to(device=device, dtype=dtype)
    variance = variance * mask + (1.0 - mask) * (math.pi * math.pi)
    angle_u8 = torch.round(best.float() * 180.0 / max(float(bins), 1.0)).clamp(0.0, 179.0)
    filtered = dog[0, 0]
    filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min()).clamp_min(EPS)
    return (
        angle_u8.detach().cpu().numpy().astype(np.uint8),
        variance.detach().cpu().numpy().astype(np.float16),
        filtered.detach().cpu().numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--orient-dir", required=True)
    parser.add_argument("--conf-dir", required=True)
    parser.add_argument("--filtered-img-dir", default="")
    parser.add_argument("--vis-img-dir", default="")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--bins", type=int, default=180)
    parser.add_argument("--dog-low", type=float, default=0.4)
    parser.add_argument("--dog-high", type=float, default=10.0)
    parser.add_argument("--sigma-x", type=float, default=1.8)
    parser.add_argument("--sigma-y", type=float, default=2.4)
    parser.add_argument("--frequency", type=float, default=0.23)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--manifest-name", default="orientation_manifest.json")
    parser.add_argument("--sheet-name", default="orientation_contact_sheet.jpg")
    parser.add_argument("--sheet-max-images", type=int, default=12)
    parser.add_argument("--sheet-tile", type=int, default=192)
    args = parser.parse_args()

    img_dir = Path(args.img_path)
    mask_dir = Path(args.mask_path)
    orient_dir = Path(args.orient_dir)
    conf_dir = Path(args.conf_dir)
    filtered_dir = Path(args.filtered_img_dir) if args.filtered_img_dir else None
    vis_dir = Path(args.vis_img_dir) if args.vis_img_dir else None
    orient_dir.mkdir(parents=True, exist_ok=True)
    conf_dir.mkdir(parents=True, exist_ok=True)
    if filtered_dir is not None:
        filtered_dir.mkdir(parents=True, exist_ok=True)
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    image_paths = sorted(img_dir.glob("*.png"))
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise RuntimeError(f"no png images found in {img_dir}")

    print(
        {
            "img_path": str(img_dir),
            "mask_path": str(mask_dir),
            "orient_dir": str(orient_dir),
            "conf_dir": str(conf_dir),
            "count": len(image_paths),
            "width": int(args.width),
            "bins": int(args.bins),
            "device": str(device),
        },
        flush=True,
    )

    records: list[dict[str, object]] = []
    confidence_means = []
    variance_means = []
    mask_fractions = []
    for idx, image_path in enumerate(image_paths, start=1):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"missing mask for {image_path.name}: {mask_path}")
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = resize_pair(image, mask, args.width)
        gray = np.asarray(image, dtype=np.float32) / 255.0
        gray = 0.2126 * gray[..., 0] + 0.7152 * gray[..., 1] + 0.0722 * gray[..., 2]
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        angle_u8, variance, filtered = compute_orientation(
            gray,
            mask_np,
            args.bins,
            args.dog_low,
            args.dog_high,
            args.sigma_x,
            args.sigma_y,
            args.frequency,
            device,
        )
        stem = image_path.stem
        angle_path = orient_dir / f"{stem}.png"
        var_path = conf_dir / f"{stem}.npy"
        Image.fromarray(angle_u8).save(angle_path)
        np.save(var_path, variance)
        if filtered_dir is not None:
            Image.fromarray((filtered * 255.0).astype(np.uint8)).save(filtered_dir / f"{stem}.png")
        confidence = confidence_from_variance(variance.astype(np.float32), mask_np)
        confidence_vis_path = orient_dir.parent / "confidence_vis" / f"{stem}.png"
        confidence_vis_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((confidence * 255.0).astype(np.uint8)).save(confidence_vis_path)
        vis_path = None
        if vis_dir is not None:
            vis_path = vis_dir / f"{stem}.png"
            Image.fromarray(colorize_orientation(angle_u8, mask_np)).save(vis_path)
        valid = mask_np > 0.1
        variance_mean = float(np.mean(variance[valid])) if np.any(valid) else 0.0
        confidence_mean = float(np.mean(confidence[valid])) if np.any(valid) else 0.0
        mask_fraction = float(np.mean(valid))
        variance_means.append(variance_mean)
        confidence_means.append(confidence_mean)
        mask_fractions.append(mask_fraction)
        rec = {
            "index": idx,
            "image": image_path.name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "angle_path": str(angle_path),
            "var_path": str(var_path),
            "vis_path": str(vis_path) if vis_path is not None else str(angle_path),
            "confidence_vis_path": str(confidence_vis_path),
            "mask_fraction": mask_fraction,
            "variance_mean": variance_mean,
            "confidence_mean": confidence_mean,
            "angle_unique": int(np.unique(angle_u8[valid]).size) if np.any(valid) else 0,
        }
        records.append(rec)
        print(
            {
                "index": idx,
                "image": image_path.name,
                "angle": str(angle_path),
                "var": str(var_path),
                "variance_mean": variance_mean,
                "confidence_mean": confidence_mean,
                "angle_unique": rec["angle_unique"],
            },
            flush=True,
        )

    manifest_path = orient_dir.parent / args.manifest_name
    sheet_path = orient_dir.parent / args.sheet_name
    make_orientation_contact_sheet(records, sheet_path, int(args.sheet_tile), int(args.sheet_max_images))
    summary = {
        "image_count": len(records),
        "bins": int(args.bins),
        "width": int(args.width),
        "device": str(device),
        "mask_fraction_mean": float(np.mean(mask_fractions)) if mask_fractions else 0.0,
        "variance_mean": float(np.mean(variance_means)) if variance_means else 0.0,
        "confidence_mean": float(np.mean(confidence_means)) if confidence_means else 0.0,
        "angle_unique_min": int(min(int(r["angle_unique"]) for r in records)) if records else 0,
        "angle_unique_mean": float(np.mean([int(r["angle_unique"]) for r in records])) if records else 0.0,
    }
    manifest = {
        "summary": summary,
        "args": vars(args),
        "outputs": {
            "orient_dir": str(orient_dir),
            "conf_dir": str(conf_dir),
            "sheet": str(sheet_path),
        },
        "records": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print({"manifest": str(manifest_path), "sheet": str(sheet_path), "summary": summary}, flush=True)


if __name__ == "__main__":
    main()

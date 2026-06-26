"""Plain gsplat baseline for Checkpoint A.

This baseline deliberately does not use AniGroom roots, strands, or grooming
parameters. It exists to validate cameras, data loading, metrics, and the
gsplat backend before the mesh-rooted fur representation is implemented.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.rendering import rasterization

from anigroom.data.white_tiger import build_stage1_input_report, list_images
from anigroom.evaluation.metrics import MetricComputer


@dataclass(frozen=True)
class BaselineConfig:
    data_root: str
    mesh_path: str
    output_dir: str
    num_gaussians: int = 5000
    iterations: int = 1000
    lr_means: float = 1.6e-4
    lr_features: float = 2.5e-3
    lr_opacity: float = 5.0e-2
    lr_scale: float = 5.0e-3
    seed: int = 7
    test_stride: int = 6
    eval_every: int = 100
    save_every: int = 1000
    train_indices: tuple[int, ...] | None = None
    white_background: bool = True
    compute_lpips: bool = False


def read_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"no vertices found in OBJ: {path}")
    return np.asarray(vertices, dtype=np.float32)


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device)


def load_mask(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        mask = image.convert("L")
        arr = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(arr[..., None]).to(device=device)


def load_camera_tensors(data_root: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    intr = np.load(data_root / "cameras_intr.npy").astype(np.float32)
    extr = np.load(data_root / "cameras_extr.npy").astype(np.float32)
    viewmats = torch.from_numpy(extr).to(device=device)
    ks = torch.from_numpy(intr[:, :3, :3]).to(device=device)
    return viewmats, ks


class PlainGsplatModel(torch.nn.Module):
    def __init__(self, vertices: np.ndarray, num_gaussians: int, seed: int, device: torch.device):
        super().__init__()
        rng = np.random.default_rng(seed)
        replace = vertices.shape[0] < num_gaussians
        ids = rng.choice(vertices.shape[0], size=num_gaussians, replace=replace)
        means = torch.from_numpy(vertices[ids]).to(device=device)
        means = means + 0.002 * torch.randn_like(means)
        self.means = torch.nn.Parameter(means)
        self.quats_raw = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_gaussians, 1)
        )
        self.scales_raw = torch.nn.Parameter(torch.full((num_gaussians, 3), -4.8, device=device))
        self.opacities_raw = torch.nn.Parameter(torch.full((num_gaussians,), -1.4, device=device))
        self.colors_raw = torch.nn.Parameter(torch.full((num_gaussians, 3), 1.8, device=device))

    def parameters_for_render(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        quats = F.normalize(self.quats_raw, dim=-1)
        scales = F.softplus(self.scales_raw) + 1e-4
        opacities = torch.sigmoid(self.opacities_raw)
        colors = torch.sigmoid(self.colors_raw)
        return self.means, quats, scales, opacities, colors


@torch.no_grad()
def evaluate_views(
    model: PlainGsplatModel,
    image_paths: list[Path],
    mask_paths: list[Path],
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    indices: Iterable[int],
    width: int,
    height: int,
    device: torch.device,
    white_background: bool,
    metric_computer: MetricComputer,
) -> dict[str, float]:
    psnrs = []
    mses = []
    ssims = []
    lpips_values = []
    mask_l1s = []
    means, quats, scales, opacities, colors = model.parameters_for_render()
    background = torch.ones((1, 3), device=device) if white_background else None
    for idx in indices:
        target = load_image(image_paths[idx], device)
        mask = load_mask(mask_paths[idx], device)
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
        pred = render[0].clamp(0.0, 1.0)
        pred_alpha = alpha[0].clamp(0.0, 1.0)
        image_metrics = metric_computer.image_metrics(pred, target)
        psnrs.append(image_metrics["psnr"].detach())
        mses.append(image_metrics["mse"].detach())
        ssims.append(image_metrics["ssim"].detach())
        if image_metrics["lpips"] is not None:
            lpips_values.append(image_metrics["lpips"].detach())
        mask_l1s.append(torch.mean(torch.abs(pred_alpha - mask)).detach())
    if not mses:
        return {"psnr": 0.0, "mse": 0.0, "ssim": 0.0, "lpips": None, "mask_l1": 0.0, "view_count": 0.0}
    return {
        "psnr": float(torch.stack(psnrs).mean().cpu()),
        "mse": float(torch.stack(mses).mean().cpu()),
        "ssim": float(torch.stack(ssims).mean().cpu()),
        "lpips": float(torch.stack(lpips_values).mean().cpu()) if lpips_values else None,
        "mask_l1": float(torch.stack(mask_l1s).mean().cpu()),
        "view_count": float(len(mses)),
    }


def train_plain_gsplat_baseline(config: BaselineConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("plain gsplat baseline requires CUDA")
    device = torch.device("cuda")
    data_root = Path(config.data_root)
    mesh_path = Path(config.mesh_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_stage1_input_report(data_root, mesh_path, test_stride=config.test_stride)
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    (output_dir / "stage1_inputs.json").write_text(
        json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    image_paths = list_images(Path(report.image_dir))
    mask_paths = list_images(Path(report.mask_dir))
    width, height = report.image_size or (0, 0)
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid image size in input report")
    viewmats, ks = load_camera_tensors(data_root, device)
    vertices = read_obj_vertices(mesh_path)
    model = PlainGsplatModel(vertices, config.num_gaussians, config.seed, device)
    metric_computer = MetricComputer(compute_lpips=config.compute_lpips).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": [model.means], "lr": config.lr_means},
            {"params": [model.colors_raw], "lr": config.lr_features},
            {"params": [model.opacities_raw], "lr": config.lr_opacity},
            {"params": [model.scales_raw, model.quats_raw], "lr": config.lr_scale},
        ]
    )
    train_indices = list(config.train_indices) if config.train_indices is not None else report.train_indices
    if not train_indices:
        raise RuntimeError("empty training index list")
    invalid_train_indices = [idx for idx in train_indices if idx < 0 or idx >= report.image_count]
    if invalid_train_indices:
        raise RuntimeError(f"train indices out of range: {invalid_train_indices}")
    test_indices = report.test_indices
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    log_path = output_dir / "metrics.jsonl"
    start = time.time()
    background = torch.ones((1, 3), device=device) if config.white_background else None

    with log_path.open("a", encoding="utf-8") as log:
        for iteration in range(1, config.iterations + 1):
            view_idx = int(train_indices[int(torch.randint(len(train_indices), (1,), generator=generator))])
            target = load_image(image_paths[view_idx], device)
            mask = load_mask(mask_paths[view_idx], device)
            means, quats, scales, opacities, colors = model.parameters_for_render()
            render, alpha, _ = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats[view_idx : view_idx + 1],
                ks[view_idx : view_idx + 1],
                width,
                height,
                packed=False,
                backgrounds=background,
            )
            pred = render[0].clamp(0.0, 1.0)
            pred_alpha = alpha[0].clamp(0.0, 1.0)
            rgb_loss = torch.mean(torch.abs(pred - target))
            mask_loss = torch.mean(torch.abs(pred_alpha - mask))
            loss = rgb_loss + 0.1 * mask_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if iteration == 1 or iteration % config.eval_every == 0 or iteration == config.iterations:
                train_eval = evaluate_views(
                    model,
                    image_paths,
                    mask_paths,
                    viewmats,
                    ks,
                    train_indices,
                    width,
                    height,
                    device,
                    config.white_background,
                    metric_computer,
                )
                test_eval = evaluate_views(
                    model,
                    image_paths,
                    mask_paths,
                    viewmats,
                    ks,
                    test_indices,
                    width,
                    height,
                    device,
                    config.white_background,
                    metric_computer,
                )
                record = {
                    "iteration": iteration,
                    "elapsed_sec": round(time.time() - start, 3),
                    "loss": float(loss.detach().cpu()),
                    "rgb_l1": float(rgb_loss.detach().cpu()),
                    "mask_l1": float(mask_loss.detach().cpu()),
                    "train": train_eval,
                    "test": test_eval,
                    "num_gaussians": config.num_gaussians,
                    "max_memory_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
                    "lpips_available": bool(metric_computer.lpips_available),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)

            if config.save_every > 0 and (iteration % config.save_every == 0 or iteration == config.iterations):
                torch.save(
                    {
                        "iteration": iteration,
                        "config": asdict(config),
                        "state_dict": model.state_dict(),
                    },
                    output_dir / f"checkpoint_{iteration:06d}.pt",
                )

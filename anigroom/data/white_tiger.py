"""White tiger Stage 1 dataset inspection.

This module is intentionally small. It verifies the processed NeuralFur-style
white tiger layout used by Checkpoint A and produces a reproducible train/test
split report. Training loaders should build on this module instead of adding
separate path and split logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class FileGroupReport:
    exists: bool
    count: int
    first: str | None
    first_size: tuple[int, int] | None
    first_mode: str | None


@dataclass(frozen=True)
class CameraReport:
    exists: bool
    kind: str
    keys: list[str]
    shapes: dict[str, list[int]]
    dtypes: dict[str, str]


@dataclass(frozen=True)
class Stage1InputReport:
    data_root: str
    mesh_path: str
    image_dir: str
    mask_dir: str
    orientation_root: str
    image_count: int
    mask_count: int
    orientation_angle_count: int
    orientation_conf_count: int
    image_size: tuple[int, int] | None
    mask_size: tuple[int, int] | None
    train_indices: list[int]
    test_indices: list[int]
    groups: dict[str, FileGroupReport]
    cameras: dict[str, CameraReport]
    errors: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def list_npy(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == ".npy")


def inspect_image_group(directory: Path) -> FileGroupReport:
    files = list_images(directory)
    first_size: tuple[int, int] | None = None
    first_mode: str | None = None
    if files:
        with Image.open(files[0]) as image:
            first_size = tuple(int(v) for v in image.size)
            first_mode = str(image.mode)
    return FileGroupReport(
        exists=directory.is_dir(),
        count=len(files),
        first=files[0].name if files else None,
        first_size=first_size,
        first_mode=first_mode,
    )


def inspect_camera(path: Path) -> CameraReport:
    if not path.exists():
        return CameraReport(False, path.suffix.lower(), [], {}, {})
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        keys = list(data.keys())
        return CameraReport(
            True,
            "npz",
            keys,
            {key: [int(v) for v in data[key].shape] for key in keys},
            {key: str(data[key].dtype) for key in keys},
        )
    data = np.load(path)
    return CameraReport(
        True,
        "npy",
        ["array"],
        {"array": [int(v) for v in data.shape]},
        {"array": str(data.dtype)},
    )


def fixed_stride_split(count: int, test_stride: int) -> tuple[list[int], list[int]]:
    if count <= 0:
        return [], []
    if test_stride <= 0:
        return list(range(count)), []
    test = [idx for idx in range(count) if idx % test_stride == 0]
    if len(test) == count and count > 1:
        test = [0]
    train = [idx for idx in range(count) if idx not in set(test)]
    return train, test


def build_stage1_input_report(
    data_root: Path,
    mesh_path: Path,
    orientation_dir: str = "orientations_2",
    test_stride: int = 6,
) -> Stage1InputReport:
    data_root = data_root.resolve()
    mesh_path = mesh_path.resolve()
    image_dir = data_root / "images"
    mask_dir = data_root / "silhouette"
    if not mask_dir.is_dir() and (data_root / "masks").is_dir():
        mask_dir = data_root / "masks"
    orientation_root = data_root / orientation_dir
    angle_dir = orientation_root / "angles"
    conf_dir = orientation_root / "vars"

    groups = {
        "images": inspect_image_group(image_dir),
        "masks": inspect_image_group(mask_dir),
        "orientation_angles": inspect_image_group(angle_dir),
    }
    angle_count = groups["orientation_angles"].count
    conf_count = len(list_npy(conf_dir))

    cameras = {
        "cameras.npz": inspect_camera(data_root / "cameras.npz"),
        "cameras_wo_scale.npz": inspect_camera(data_root / "cameras_wo_scale.npz"),
        "cameras_intr.npy": inspect_camera(data_root / "cameras_intr.npy"),
        "cameras_extr.npy": inspect_camera(data_root / "cameras_extr.npy"),
        "cameras_extr_wo_scale.npy": inspect_camera(data_root / "cameras_extr_wo_scale.npy"),
    }

    errors: list[str] = []
    if not data_root.is_dir():
        errors.append(f"missing data_root: {data_root}")
    if not mesh_path.is_file():
        errors.append(f"missing mesh_path: {mesh_path}")
    image_count = groups["images"].count
    mask_count = groups["masks"].count
    if image_count == 0:
        errors.append(f"no images found under {image_dir}")
    if mask_count == 0:
        errors.append(f"no masks found under {mask_dir}")
    if image_count and mask_count and image_count != mask_count:
        errors.append(f"image/mask count mismatch: {image_count} vs {mask_count}")
    if image_count and angle_count and image_count != angle_count:
        errors.append(f"image/orientation angle count mismatch: {image_count} vs {angle_count}")
    if image_count and conf_count and image_count != conf_count:
        errors.append(f"image/orientation confidence count mismatch: {image_count} vs {conf_count}")
    if image_count and not cameras["cameras.npz"].exists:
        errors.append("missing cameras.npz")
    train_indices, test_indices = fixed_stride_split(image_count, test_stride)
    if not train_indices:
        errors.append("empty train split")

    return Stage1InputReport(
        data_root=str(data_root),
        mesh_path=str(mesh_path),
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        orientation_root=str(orientation_root),
        image_count=image_count,
        mask_count=mask_count,
        orientation_angle_count=angle_count,
        orientation_conf_count=conf_count,
        image_size=groups["images"].first_size,
        mask_size=groups["masks"].first_size,
        train_indices=train_indices,
        test_indices=test_indices,
        groups=groups,
        cameras=cameras,
        errors=errors,
    )


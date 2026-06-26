"""Dataset alignment configuration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path, *, project_root: Path = PROJECT_ROOT) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return project_root / value


def load_alignment_config(path: str | Path | None, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = resolve_project_path(path, project_root=project_root)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["_config_path"] = str(config_path)
    return data


def apply_alignment_to_namespace(args: Any, alignment: dict[str, Any], *, include_uv: bool = True) -> None:
    """Apply common alignment fields to an argparse namespace in-place."""

    if not alignment:
        return

    def assign_path_attr(name: str, value: str) -> None:
        current = getattr(args, name, None)
        setattr(args, name, Path(value) if isinstance(current, Path) else value)

    if "data_root" in alignment:
        assign_path_attr("data_root", alignment["data_root"])
    if "mesh_path" in alignment:
        assign_path_attr("mesh_path", alignment["mesh_path"])
    if include_uv and "uv_atlas" in alignment and hasattr(args, "uv_atlas"):
        assign_path_attr("uv_atlas", alignment["uv_atlas"])
    if "camera_source" in alignment and hasattr(args, "camera_source"):
        args.camera_source = alignment["camera_source"]
    if "projection_file" in alignment and hasattr(args, "projection_file"):
        args.projection_file = alignment["projection_file"]

    transform = alignment.get("mesh_to_camera_initial", {})
    if "scale" in transform and hasattr(args, "init_mesh_scale"):
        args.init_mesh_scale = float(transform["scale"])
    if "translation" in transform and hasattr(args, "init_mesh_translation"):
        translation = transform["translation"]
        if len(translation) != 3:
            raise ValueError("mesh_to_camera_initial.translation must have 3 values")
        args.init_mesh_translation = [float(v) for v in translation]

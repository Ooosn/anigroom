from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Iterable

# Different views retain different numbers of depth-clipped Gaussians. Expandable
# segments prevent the CUDA allocator from accumulating incompatible large blocks.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from gsplat.rendering import rasterization
from torch.utils.checkpoint import checkpoint as activation_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ACTIVATION_CHECKPOINT_MAX_DEVICE_MEMORY_BYTES = 48 * 1024**3
CURRENT_CHECKPOINT_VERSION = 14


def memory_constrained_activation_checkpointing(device: torch.device) -> bool:
    """Use recomputation on GPUs where the full strand graph risks paging.

    This changes only autograd storage. Forward values, rendered Gaussians, and
    gradients are unchanged. Larger accelerator GPUs retain the faster path.
    """

    if not torch.is_grad_enabled() or device.type != "cuda" or not torch.cuda.is_available():
        return False
    properties = torch.cuda.get_device_properties(device)
    return int(properties.total_memory) <= ACTIVATION_CHECKPOINT_MAX_DEVICE_MEMORY_BYTES

from anigroom.data.white_tiger import build_stage1_input_report, list_images  # noqa: E402
from anigroom.data.alignment import apply_alignment_to_namespace, load_alignment_config  # noqa: E402
from anigroom.collision.sdf import (  # noqa: E402
    SignedDistanceGrid,
    cyclic_strand_indices,
    strand_penetration_depth,
)
from anigroom.collision.strand_crossing import (  # noqa: E402
    GaussianSegmentSnapshot,
    StrandCrossingActiveSet,
    TorchStrandCrossingActiveSet,
    active_set_crossing_loss,
    discover_gaussian_segment_crossings,
)
from anigroom.evaluation.metrics import MetricComputer  # noqa: E402
from anigroom.flow import (  # noqa: E402
    CleanFlowTargets,
    clean_flow_anchor_loss,
    clean_flow_smoothness_loss,
    groom_direction_3d,
    load_clean_flow_targets,
    sample_clean_flow_targets,
)
from anigroom.flow.direction_geometry import (  # noqa: E402
    parallel_transport_vector_field,
    parallel_transport_vectors,
)
from anigroom.grooming import (  # noqa: E402
    GaussianRGBResidualField,
    GuideViewSHField,
    GroomParameterField,
    GroomRanges,
    RenderGeometryResidualField,
    ViewGatedOwnership,
    apply_asinh_logit_residual,
    apply_asinh_log_ratio_residual,
    apply_direction_residual,
    apply_log_ratio_residual,
    build_strands,
    decode_brush_stiffness,
    decode_positive_asinh,
    decode_positive_asinh_ratio,
    decode_positive_softplus,
    direction_to_local_components,
    encode_brush_stiffness,
    encode_positive_asinh,
    encode_positive_asinh_ratio,
    encode_positive_softplus,
    encode_asinh_logit_residual,
    expand_child_strands,
    fourth_moment_norm,
    guide_support_gauge,
    load_trusted_guide_view_confidence,
    length_residual_prior_coordinate,
    local_components_to_world,
    make_tangent_frames,
    population_stable_residual_norm,
    resample_strands_to_segment_budgets,
    straight_through_gate,
    straight_through_gate_geometry,
    strand_segment_budgets,
    strands_to_gaussians,
    tail_concentration_residual_loss,
    vector_to_local_components,
)
from anigroom.grooming.secondary_guides import (  # noqa: E402
    InterpolatedGeometryResiduals,
    build_parent_conditioned_query_support,
    initialize_parent_conditioned_secondary_roots,
    interpolate_secondary_geometry_residuals,
)
from anigroom.mesh_roots import (  # noqa: E402
    SurfaceRoots,
    TriangleMesh,
    barycentric_to_points,
    initialize_surface_roots_fps,
    initialize_surface_roots_stratified,
    read_obj_mesh,
    validate_surface_roots,
)
from anigroom.projection import (  # noqa: E402
    MeshDepthResult,
    render_mesh_depth,
    render_mesh_depth_from_tensors,
    render_mesh_vertex_color_from_tensors,
    sample_depth_nearest,
    sample_mesh_visible_points,
)
from anigroom.roots.lifecycle import (  # noqa: E402
    DensifyConfig,
    FaceAdjacencyIndex,
    PruneConfig,
    RootLifecycleState,
    RootStructureUpdate,
    apply_attribute_update,
    apply_structure_update,
    interpolate_child_attributes,
    propose_split_children,
    propose_structure_update,
    normalized_root_need,
)
from anigroom.roots.statistics import RootStatsWindow  # noqa: E402
from anigroom.surface_interpolation import (  # noqa: E402
    SurfaceFieldInterpolator,
    LocalSurfaceSupport,
    SurfaceSourceGraph,
    SurfaceSupport,
    build_hierarchical_surface_edges,
    build_local_surface_support,
    density_invariant_log_scalar_smoothness,
    interpolate_directions,
    harmonic_inpaint_physical,
    interpolate_periodic,
    interpolate_physical,
    local_surface_weights,
    reconstruct_surface_directions,
)


EPS = 1.0e-8
CLEAN_FLOW_INIT_QUANTILE_LOW = 0.05
CLEAN_FLOW_INIT_QUANTILE_HIGH = 0.95


def release_cuda_cache() -> None:
    """Return unused cached CUDA blocks to the driver after large transient work."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def current_process_gpu_memory_mb() -> float | None:
    """Return nvidia-smi memory for this Python process, when available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    current_pid = os.getpid()
    total_mb = 0.0
    found = False
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            used_mb = float(parts[1])
        except ValueError:
            continue
        if pid == current_pid:
            total_mb += used_mb
            found = True
    return total_mb if found else None


def cuda_memory_guard_payload(device: torch.device) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "memory_allocated_mb": 0.0,
            "memory_reserved_mb": 0.0,
            "max_memory_allocated_mb": 0.0,
            "max_memory_reserved_mb": 0.0,
            "nvidia_smi_process_mb": 0.0,
            "device_used_mb": 0.0,
            "device_free_mb": 0.0,
            "device_total_mb": 0.0,
        }
    nvidia_smi_mb = current_process_gpu_memory_mb()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    mib = 1024.0 * 1024.0
    return {
        "memory_allocated_mb": float(torch.cuda.memory_allocated(device) / mib),
        "memory_reserved_mb": float(torch.cuda.memory_reserved(device) / mib),
        "max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / mib),
        "max_memory_reserved_mb": float(torch.cuda.max_memory_reserved(device) / mib),
        "nvidia_smi_process_mb": float(nvidia_smi_mb) if nvidia_smi_mb is not None else 0.0,
        "device_used_mb": float((total_bytes - free_bytes) / mib),
        "device_free_mb": float(free_bytes / mib),
        "device_total_mb": float(total_bytes / mib),
    }


def enforce_cuda_memory_guard(
    config: "Stage1Config",
    device: torch.device,
    *,
    iteration: int,
    stage: str,
    progress_event: object | None = None,
) -> None:
    limit_gb = float(config.gpu_memory_limit_gb)
    if limit_gb <= 0.0:
        return
    def tracked_process_memory(payload: dict[str, float]) -> tuple[float, float, float]:
        allocated_peak_mb = max(
            float(payload["memory_allocated_mb"]),
            float(payload["max_memory_allocated_mb"]),
        )
        nvidia_smi_process_mb = float(payload["nvidia_smi_process_mb"])
        external_process_overhead_mb = max(
            0.0,
            nvidia_smi_process_mb - float(payload["memory_reserved_mb"]),
        )
        process_tracked_mb = allocated_peak_mb + external_process_overhead_mb
        if nvidia_smi_process_mb <= 0.0:
            # WDDM often hides per-process usage. PyTorch's current allocator
            # reservation is still process-local; whole-device usage is not.
            process_tracked_mb = max(process_tracked_mb, float(payload["memory_reserved_mb"]))
        return process_tracked_mb, allocated_peak_mb, external_process_overhead_mb

    payload = cuda_memory_guard_payload(device)
    process_tracked_mb, allocated_peak_mb, external_process_overhead_mb = tracked_process_memory(payload)
    tracked_mb = process_tracked_mb
    limit_mb = limit_gb * 1024.0
    if tracked_mb <= limit_mb:
        return

    # A transient render can leave a large unused allocator cache. Release it
    # once and remeasure before treating cached capacity as a live-memory fault.
    if allocated_peak_mb <= limit_mb:
        tracked_before_release_mb = tracked_mb
        release_cuda_cache()
        payload = cuda_memory_guard_payload(device)
        process_tracked_mb, allocated_peak_mb, external_process_overhead_mb = tracked_process_memory(payload)
        tracked_mb = process_tracked_mb
        if callable(progress_event):
            progress_event(
                "gpu_memory_cache_released",
                iteration=int(iteration),
                guard_stage=stage,
                limit_gb=limit_gb,
                tracked_before_release_mb=tracked_before_release_mb,
                tracked_after_release_mb=tracked_mb,
                **payload,
            )
        if tracked_mb <= limit_mb:
            return

    report = {
        "iteration": int(iteration),
        "guard_stage": stage,
        "limit_gb": limit_gb,
        "tracked_mb": tracked_mb,
        "process_tracked_mb": process_tracked_mb,
        "allocated_peak_mb": allocated_peak_mb,
        "external_process_overhead_mb": external_process_overhead_mb,
        **payload,
    }
    if callable(progress_event):
        progress_event("gpu_memory_limit_exceeded", **report)
    raise RuntimeError("GPU memory limit exceeded: " + json.dumps(report, sort_keys=True))


def parse_iteration_set(text: str) -> set[int]:
    if not str(text).strip():
        return set()
    values: set[int] = set()
    for chunk in str(text).split(","):
        item = chunk.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"checkpoint iteration must be positive, got {value}")
        values.add(value)
    return values


def restored_lifecycle_history(
    checkpoint: dict[str, object] | None,
    *,
    start_iteration: int,
) -> list[dict[str, object]]:
    if checkpoint is None:
        return []
    raw_history = checkpoint.get("lifecycle_history", [])
    if not isinstance(raw_history, list):
        raise TypeError("checkpoint lifecycle_history must be a list")
    history: list[dict[str, object]] = []
    previous_iteration = -1
    for index, raw_record in enumerate(raw_history):
        if not isinstance(raw_record, dict):
            raise TypeError(f"checkpoint lifecycle_history[{index}] must be a mapping")
        if "iteration" not in raw_record:
            raise ValueError(f"checkpoint lifecycle_history[{index}] has no iteration")
        iteration = int(raw_record["iteration"])
        if iteration <= previous_iteration:
            raise ValueError("checkpoint lifecycle_history iterations must be strictly increasing")
        if iteration > int(start_iteration):
            raise ValueError(
                f"checkpoint lifecycle event {iteration} is after checkpoint iteration {start_iteration}"
            )
        history.append(dict(raw_record))
        previous_iteration = iteration
    return history


def optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def capture_training_rng_state(generator: torch.Generator) -> dict[str, object]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "train_view_generator": generator.get_state(),
    }


def restore_training_rng_state(state: dict[str, object], generator: torch.Generator) -> None:
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "train_view_generator" in state:
        generator.set_state(state["train_view_generator"])


@contextmanager
def numpy_pickle_compat_aliases():
    """Temporarily map NumPy 2 pickle module paths onto NumPy 1.x."""

    module_aliases = {
        "numpy._core": "numpy.core",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core._multiarray_umath": "numpy.core._multiarray_umath",
    }
    installed: dict[str, object] = {}
    for alias, source in module_aliases.items():
        if alias not in sys.modules:
            module = importlib.import_module(source)
            sys.modules[alias] = module
            installed[alias] = module
    try:
        yield
    finally:
        for alias, module in installed.items():
            if sys.modules.get(alias) is module:
                del sys.modules[alias]


def load_training_checkpoint(path: Path) -> dict[str, object]:
    def load() -> object:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    try:
        checkpoint = load()
    except ModuleNotFoundError as error:
        if error.name is None or not error.name.startswith("numpy._core"):
            raise
        with numpy_pickle_compat_aliases():
            checkpoint = load()
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint is not a dict: {path}")
    require_current_checkpoint_version(checkpoint)
    return checkpoint


def require_current_checkpoint_version(checkpoint: dict[str, object]) -> None:
    checkpoint_version = int(checkpoint.get("checkpoint_version", -1))
    if checkpoint_version != CURRENT_CHECKPOINT_VERSION:
        raise RuntimeError(
            "checkpoint schema mismatch: "
            f"expected {CURRENT_CHECKPOINT_VERSION}, got {checkpoint_version}; "
            "current strict-schema reconstruction must start from zero"
        )


def setup_progress(stage: str, **extra: object) -> None:
    payload = {"setup_progress": stage, **extra}
    print(json.dumps(payload), flush=True)


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def inv_sigmoid(x: torch.Tensor, eps: float = 1.0e-5) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def set_range(raw: torch.Tensor, value: torch.Tensor | float, bounds: tuple[float, float]) -> None:
    lo, hi = bounds
    v = torch.as_tensor(value, device=raw.device, dtype=raw.dtype)
    rel = (v - lo) / max(hi - lo, EPS)
    raw.copy_(inv_sigmoid(rel).expand_as(raw))


def set_color(raw: torch.Tensor, value: torch.Tensor) -> None:
    raw.copy_(inv_sigmoid(value).expand_as(raw))


def set_unit_interval(raw: torch.Tensor, value: torch.Tensor | float) -> None:
    v = torch.as_tensor(value, device=raw.device, dtype=raw.dtype)
    raw.copy_(inv_sigmoid(v).expand_as(raw))


def set_positive_asinh(raw: torch.Tensor, value: torch.Tensor | float) -> None:
    v = torch.as_tensor(value, device=raw.device, dtype=raw.dtype)
    raw.copy_(encode_positive_asinh(v).expand_as(raw))


def dense_groom_ranges() -> GroomRanges:
    return GroomRanges(
        curl_turns=(0.0, 5.5),
        clump_strength=(0.0, 1.0),
    )


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device)


def load_mask(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr[..., None]).to(device=device)


def load_scalar_map(path: Path, size: tuple[int, int]) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        if not data.files:
            raise ValueError(f"empty npz map: {path}")
        arr = data[data.files[0]].astype(np.float32)
    else:
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"map resolution mismatch for {path}: {image.size} != {size}")
            arr = np.asarray(image.convert("L"), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.shape != (size[1], size[0]):
        raise ValueError(f"map shape mismatch for {path}: {arr.shape} != {(size[1], size[0])}")
    return arr


def load_orientation(path: Path, conf_path: Path, size: tuple[int, int], bins: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    raw_angle = load_scalar_map(path, size)
    if raw_angle.max() <= 1.0 and raw_angle.min() >= 0.0:
        angle = raw_angle * math.pi
    elif raw_angle.max() <= float(bins) + 1.0:
        angle = raw_angle / max(float(bins), 1.0) * math.pi
    else:
        angle = raw_angle / 255.0 * math.pi
    orientation = np.stack([np.cos(angle), np.sin(angle)], axis=-1).astype(np.float32)

    conf_raw = load_scalar_map(conf_path, size).astype(np.float32)
    if conf_path.suffix.lower() in {".npy", ".npz"}:
        var = np.maximum(conf_raw / (math.pi**2), 0.0)
        confidence = 1.0 / (var * var + 1.0e-7)
        finite = np.isfinite(confidence)
        if finite.any():
            norm = max(float(np.quantile(confidence[finite], 0.95)), 1.0e-6)
            confidence = np.clip(confidence / norm, 0.0, 1.0)
        else:
            confidence = np.zeros_like(conf_raw, dtype=np.float32)
    else:
        confidence = conf_raw
        if confidence.max() > 1.5:
            confidence = confidence / 255.0
        confidence = np.clip(confidence, 0.0, 1.0)
    return (
        torch.from_numpy(orientation).to(device=device),
        torch.from_numpy(confidence[..., None].astype(np.float32)).to(device=device),
    )


def load_camera_tensors(data_root: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    intr = np.load(data_root / "cameras_intr.npy").astype(np.float32)
    extr = np.load(data_root / "cameras_extr.npy").astype(np.float32)
    return torch.from_numpy(extr).to(device=device), torch.from_numpy(intr[:, :3, :3]).to(device=device)


def face_normals_np(mesh: TriangleMesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norms, EPS)
    return normals.astype(np.float32)


def save_image(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    Image.fromarray(arr).save(path)


def depth_to_image(depth: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(depth)
    if not bool(finite.any()):
        return torch.zeros((*depth.shape, 1), device=depth.device)
    values = depth[finite]
    lo = torch.quantile(values, 0.02)
    hi = torch.quantile(values, 0.98)
    norm = (depth - lo) / (hi - lo).clamp_min(EPS)
    norm = torch.where(finite, norm.clamp(0.0, 1.0), torch.zeros_like(norm))
    return norm[..., None]


@torch.no_grad()
def save_clip_overlay(
    path: Path,
    base_image: torch.Tensor,
    means: torch.Tensor,
    keep_mask: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    *,
    behind_mesh_mask: torch.Tensor | None = None,
    max_points: int = 12000,
    mode: str = "both",
) -> None:
    if mode not in {"both", "kept", "clipped"}:
        raise ValueError(f"Unknown clip overlay mode: {mode}")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (base_image.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    canvas = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    xy, depth = project_points(means, viewmat, k)
    height, width = int(base_image.shape[0]), int(base_image.shape[1])
    valid = (
        (depth > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= height - 1)
    )

    def draw_subset(mask: torch.Tensor, color: tuple[int, int, int, int], radius: int) -> None:
        ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if ids.numel() == 0:
            return
        if ids.numel() > int(max_points):
            step = max(1, int(math.ceil(ids.numel() / int(max_points))))
            ids = ids[::step]
        pts = xy[ids].detach().cpu().numpy()
        for x, y in pts:
            draw.ellipse((float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius), fill=color)

    if behind_mesh_mask is None:
        behind_mesh_mask = torch.zeros_like(keep_mask)

    if mode in {"both", "kept"}:
        draw_subset(valid & keep_mask, (40, 220, 80, 110), 1)
    if mode in {"both", "clipped"}:
        draw_subset(valid & behind_mesh_mask, (255, 40, 20, 185), 2)
    label = {
        "both": "green=kept Gaussians, red=depth-clipped Gaussians",
        "kept": "kept Gaussians only",
        "clipped": "depth-clipped Gaussians only",
    }[mode]
    draw.rectangle((10, 10, 610, 42), fill=(255, 255, 255, 220))
    draw.text((18, 17), label, fill=(0, 0, 0, 255))
    canvas.save(path)


def project_points(points: torch.Tensor, viewmat: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot = viewmat[:3, :3]
    trans = viewmat[:3, 3]
    cam = points @ rot.T + trans.view(1, 3)
    z = cam[:, 2].clamp_min(1.0e-6)
    x = k[0, 0] * (cam[:, 0] / z) + k[0, 2]
    y = k[1, 1] * (cam[:, 1] / z) + k[1, 2]
    return torch.stack([x, y], dim=-1), cam[:, 2]


def project_directions(points: torch.Tensor, directions: torch.Tensor, viewmat: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    rot = viewmat[:3, :3]
    trans = viewmat[:3, 3]
    cam = points @ rot.T + trans.view(1, 3)
    dirs_cam = directions @ rot.T
    z = cam[:, 2].clamp_min(1.0e-6)
    du = k[0, 0] * (dirs_cam[:, 0] * z - cam[:, 0] * dirs_cam[:, 2]) / z.square()
    dv = k[1, 1] * (dirs_cam[:, 1] * z - cam[:, 1] * dirs_cam[:, 2]) / z.square()
    return torch.stack([du, dv], dim=-1)


def bilinear_sample(image: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    height, width = int(image.shape[0]), int(image.shape[1])
    grid_x = (xy[:, 0] / max(width - 1, 1)) * 2.0 - 1.0
    grid_y = (xy[:, 1] / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(-1).T


def mask_edge_confidence(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if int(kernel_size) <= 1:
        return mask
    pad = int(kernel_size) // 2
    if mask.ndim == 4 and mask.shape[-1] == 1:
        pooled_input = mask.permute(0, 3, 1, 2)
        restore = "nhwc"
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        pooled_input = mask[..., 0][None, None]
        restore = "hwc"
    elif mask.ndim == 3:
        pooled_input = mask[:, None]
        restore = "bhw"
    elif mask.ndim == 2:
        pooled_input = mask[None, None]
        restore = "hw"
    else:
        raise ValueError(f"unsupported mask shape for edge confidence: {tuple(mask.shape)}")
    eroded = -F.max_pool2d(-pooled_input, kernel_size=int(kernel_size), stride=1, padding=pad)
    eroded = eroded.clamp(0.0, 1.0)
    if restore == "nhwc":
        return eroded.permute(0, 2, 3, 1)
    if restore == "hwc":
        return eroded[0, 0][..., None]
    if restore == "bhw":
        return eroded[:, 0]
    return eroded[0, 0]


def loss_mask_edge_weight(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Drop a silhouette band from image losses while keeping safe inside/outside pixels."""
    if int(kernel_size) <= 1:
        return torch.ones_like(mask)
    inside = mask_edge_confidence(mask, kernel_size)
    outside = mask_edge_confidence(1.0 - mask, kernel_size)
    return torch.maximum(inside, outside).detach().clamp(0.0, 1.0)


def view_angle_weight(normals: torch.Tensor, viewmat: torch.Tensor, power: float) -> torch.Tensor:
    normal_cam = normals @ viewmat[:3, :3].T
    weight = (-normal_cam[:, 2]).clamp(0.0, 1.0)
    if float(power) != 1.0:
        weight = weight.pow(float(power))
    return weight


def image_structure_flow(
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable structure-tensor orientation from an RGB image.

    This is intentionally the same operator for prediction and target. It is
    not a replacement for the clean 3D guide anchor; it gives an image-domain
    flow signal for the RGB-rendered result.
    """

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"image_structure_flow expects [H, W, 3], got {tuple(image.shape)}")
    rgb = image.clamp(0.0, 1.0).permute(2, 0, 1).unsqueeze(0)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    sobel_x = gray.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    sobel_y = gray.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)

    # Smooth the tensor products, not the final angle. This keeps the operator
    # differentiable while reducing stripe/noise-induced one-pixel direction flips.
    kernel = gray.new_tensor(
        [[1.0, 4.0, 6.0, 4.0, 1.0],
         [4.0, 16.0, 24.0, 16.0, 4.0],
         [6.0, 24.0, 36.0, 24.0, 6.0],
         [4.0, 16.0, 24.0, 16.0, 4.0],
         [1.0, 4.0, 6.0, 4.0, 1.0]]
    ).view(1, 1, 5, 5) / 256.0
    jxx = F.conv2d(gx * gx, kernel, padding=2)
    jyy = F.conv2d(gy * gy, kernel, padding=2)
    jxy = F.conv2d(gx * gy, kernel, padding=2)

    # Edge tangent orientation is perpendicular to the image gradient. In
    # double-angle form, perpendicular rotation is a sign flip.
    cos2 = jyy - jxx
    sin2 = -2.0 * jxy
    orient = F.normalize(torch.cat([cos2, sin2], dim=1), dim=1, eps=eps)
    energy = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square()).squeeze(0).permute(1, 2, 0)
    conf = energy / energy.detach().amax().clamp_min(eps)
    conf = conf.clamp(0.0, 1.0)
    if mask is not None:
        conf = conf * mask.detach().clamp(0.0, 1.0)
    return orient.squeeze(0).permute(1, 2, 0), conf


def image_structure_flow_losses(
    pred_image: torch.Tensor,
    target_image: torch.Tensor,
    mask: torch.Tensor,
    *,
    min_confidence: float,
    target_flow: torch.Tensor | None = None,
    target_confidence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
    pred_flow, pred_conf = image_structure_flow(pred_image, mask)
    if target_flow is None or target_confidence is None:
        with torch.no_grad():
            target_flow, target_conf = image_structure_flow(target_image, mask)
    else:
        target_flow = target_flow.to(device=pred_image.device, dtype=pred_image.dtype)
        target_conf = target_confidence.to(device=pred_image.device, dtype=pred_image.dtype)
    valid_target = (target_conf >= float(min_confidence)).to(dtype=pred_image.dtype)
    pred_visible = (pred_conf.detach() > 0.02).to(dtype=pred_image.dtype)
    weight = target_conf.detach() * valid_target * pred_visible
    dot = (pred_flow * target_flow.detach()).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    flow_loss = ((1.0 - dot) * weight).sum() / weight.sum().clamp_min(1.0)

    dx_weight = torch.minimum(weight[:, 1:], weight[:, :-1])
    dy_weight = torch.minimum(weight[1:, :], weight[:-1, :])
    dx = (pred_flow[:, 1:] - pred_flow[:, :-1]) - (target_flow[:, 1:] - target_flow[:, :-1]).detach()
    dy = (pred_flow[1:, :] - pred_flow[:-1, :]) - (target_flow[1:, :] - target_flow[:-1, :]).detach()
    dx_loss = (dx.abs() * dx_weight).sum() / dx_weight.sum().clamp_min(1.0)
    dy_loss = (dy.abs() * dy_weight).sum() / dy_weight.sum().clamp_min(1.0)
    detail_loss = 0.5 * (dx_loss + dy_loss)
    return flow_loss, detail_loss, {
        "rgb_flow_loss": float(flow_loss.detach().cpu()),
        "rgb_flow_detail_loss": float(detail_loss.detach().cpu()),
        "rgb_flow_weight_sum": float(weight.detach().sum().cpu()),
        "rgb_flow_valid_pixels": int((weight.detach() > 0.0).sum().cpu()),
    }


def build_knn_edges(points: torch.Tensor, k: int, chunk_size: int = 2048) -> torch.Tensor:
    if k <= 0 or points.shape[0] < 2:
        return torch.empty((0, 2), dtype=torch.long, device=points.device)
    root_count = int(points.shape[0])
    neighbor_count = min(int(k), root_count - 1)
    query_count = neighbor_count + 1
    pts_np = np.ascontiguousarray(points.detach().to(device="cpu", dtype=torch.float32).numpy())
    tree = cKDTree(pts_np)
    _, nn = tree.query(pts_np, k=query_count, workers=-1)
    nn = np.asarray(nn, dtype=np.int64)
    if nn.ndim == 1:
        nn = nn[:, None]
    src_np = np.arange(root_count, dtype=np.int64)[:, None]
    valid = nn != src_np
    order = np.argsort(~valid, axis=1, kind="stable")
    nn_sorted = np.take_along_axis(nn, order, axis=1)
    valid_sorted = np.take_along_axis(valid, order, axis=1)
    nn = nn_sorted[:, :neighbor_count].copy()
    selected_valid = valid_sorted[:, :neighbor_count]
    if not bool(selected_valid.all()):
        missing_rows = np.flatnonzero(~selected_valid.all(axis=1))
        for row in missing_rows.tolist():
            row_query_count = query_count
            row_nn = np.asarray(nn_sorted[row], dtype=np.int64)
            row_valid = row_nn[row_nn != row]
            while row_valid.size < neighbor_count and row_query_count < root_count:
                row_query_count = min(root_count, max(row_query_count * 2, neighbor_count + 1))
                _, row_nn = tree.query(pts_np[row], k=row_query_count, workers=-1)
                row_nn = np.asarray(row_nn, dtype=np.int64).reshape(-1)
                row_valid = row_nn[row_nn != row]
            if row_valid.size == 0:
                raise RuntimeError("KNN edge construction found no non-self neighbor")
            if row_valid.size < neighbor_count:
                repeats = int(np.ceil(neighbor_count / row_valid.size))
                row_valid = np.tile(row_valid, repeats)
            nn[row] = row_valid[:neighbor_count]
    src = np.repeat(np.arange(root_count, dtype=np.int64), neighbor_count)
    dst = nn.reshape(-1)
    edges_np = np.stack([src, dst], axis=-1)
    return torch.as_tensor(edges_np, dtype=torch.long, device=points.device)


def interpolate_unobserved_root_values(
    roots: torch.Tensor,
    values: torch.Tensor,
    observed: torch.Tensor,
    confidence: torch.Tensor,
    *,
    neighbor_count: int = 8,
    chunk_size: int = 2048,
    normalize_vectors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[0] != roots.shape[0] or observed.shape[0] != roots.shape[0]:
        raise RuntimeError("root interpolation shape mismatch")
    if bool(observed.all()):
        return values, observed.clone()
    if not bool(observed.any()):
        return values, observed.clone()

    filled = values.clone()
    obs_idx = torch.nonzero(observed, as_tuple=False).reshape(-1)
    miss_idx = torch.nonzero(~observed, as_tuple=False).reshape(-1)
    obs_roots = roots[obs_idx].detach()
    obs_values = values[obs_idx]
    obs_conf = confidence[obs_idx].reshape(-1).clamp_min(1.0e-4)
    k = min(int(neighbor_count), int(obs_idx.numel()))
    for begin in range(0, int(miss_idx.numel()), int(chunk_size)):
        ids = miss_idx[begin : begin + int(chunk_size)]
        dist = torch.cdist(roots[ids].detach(), obs_roots)
        nn_dist, nn_local = torch.topk(dist, k=k, dim=1, largest=False)
        weights = (1.0 / nn_dist.clamp_min(1.0e-6)) * obs_conf[nn_local]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        interp = (obs_values[nn_local] * weights[..., None]).sum(dim=1)
        if normalize_vectors:
            interp = F.normalize(interp, dim=-1, eps=1.0e-8)
        filled[ids] = interp
    interpolated = observed.clone()
    interpolated[miss_idx] = True
    return filled, interpolated


def observed_quantile_bounds(
    values: torch.Tensor,
    observed: torch.Tensor,
    *,
    label: str,
) -> tuple[float, float, int]:
    flat_values = values.reshape(-1)
    observed_flat = observed.reshape(-1).to(device=flat_values.device)
    valid = observed_flat & torch.isfinite(flat_values)
    count = int(valid.sum().detach().cpu())
    if count < 4:
        raise RuntimeError(f"{label} needs at least 4 observed values for 5%-95% clamp; got {count}")
    selected = flat_values[valid]
    lo = torch.quantile(selected, CLEAN_FLOW_INIT_QUANTILE_LOW)
    hi = torch.quantile(selected, CLEAN_FLOW_INIT_QUANTILE_HIGH)
    if bool((hi < lo).detach().cpu()):
        lo, hi = hi, lo
    return float(lo.detach().cpu()), float(hi.detach().cpu()), count


def data_clamped_clean_flow_length(
    roots: torch.Tensor,
    surface_edges: torch.Tensor,
    sample: dict[str, torch.Tensor],
    config: Stage1Config,
    *,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, int, int]:
    confidence = sample["confidence"].reshape(-1).clamp(0.0, 1.0)
    observed = (
        sample["valid"].reshape(-1)
        & (sample["shell_height"].reshape(-1) > 0.0)
        & (confidence >= float(config.clean_flow_length_init_min_confidence))
    )
    length = (sample["shell_height"].reshape(-1, 1) * float(config.clean_flow_length_init_scale))
    clamp_lo, clamp_hi, observed_count = observed_quantile_bounds(
        length,
        observed,
        label=label,
    )
    reliable = (
        observed
        & (length.reshape(-1) >= float(clamp_lo))
        & (length.reshape(-1) <= float(clamp_hi))
    )
    reliable_count = int(reliable.sum().detach().cpu())
    if reliable_count < 4:
        raise RuntimeError(
            f"{label} has only {reliable_count} topology anchors after the 5%-95% "
            f"data filter (observed={observed_count})"
        )
    filled_length = harmonic_inpaint_physical(
        length,
        roots,
        reliable,
        surface_edges,
    )
    if not bool(torch.isfinite(filled_length).all()):
        raise RuntimeError(f"{label} produced non-finite inpainted lengths")
    if not bool((filled_length > 0.0).all()):
        raise RuntimeError(f"{label} produced non-positive inpainted lengths")
    filled = torch.ones_like(reliable)
    return (
        filled_length,
        reliable,
        filled,
        clamp_lo,
        clamp_hi,
        observed_count,
        reliable_count,
    )


def reconstruct_clean_flow_directions(
    points: torch.Tensor,
    directions: torch.Tensor,
    normals: torch.Tensor,
    observed: torch.Tensor,
    confidence: torch.Tensor,
    surface_edges: torch.Tensor,
    *,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one surface-consistent 3D direction field without scalar proxies."""

    directions = F.normalize(directions, dim=-1, eps=1.0e-8)
    normals = F.normalize(normals, dim=-1, eps=1.0e-8)
    confidence = torch.where(
        observed.reshape(-1),
        confidence.reshape(-1).clamp(0.0, 1.0),
        torch.zeros_like(confidence.reshape(-1)),
    )
    anchor_count = int((confidence > 0.0).sum().detach().cpu())
    if anchor_count < 4:
        raise RuntimeError(f"{label} has only {anchor_count} reliable 3D direction anchors")
    reconstructed, reliability, supported = reconstruct_surface_directions(
        directions,
        normals,
        points,
        confidence,
        surface_edges,
    )
    return reconstructed, reliability, supported


SMOOTH_FIELD_METRICS = (
    "ambient",
    "transport_direction",
    "relative_length",
    "surface_covariant",
    "surface_covariant_full",
)
GUIDE_LENGTH_SMOOTH_MODES = (
    "edge_relative",
    "intrinsic_density_invariant",
)


def smooth_metric_uses_transport(metric: str) -> bool:
    if metric not in SMOOTH_FIELD_METRICS:
        raise ValueError(f"unknown smooth field metric: {metric}")
    return metric in ("transport_direction", "surface_covariant", "surface_covariant_full")


def smooth_metric_uses_relative_length(metric: str) -> bool:
    if metric not in SMOOTH_FIELD_METRICS:
        raise ValueError(f"unknown smooth field metric: {metric}")
    return metric in ("relative_length", "surface_covariant", "surface_covariant_full")


def smooth_metric_uses_full_relative_length_field(metric: str) -> bool:
    if metric not in SMOOTH_FIELD_METRICS:
        raise ValueError(f"unknown smooth field metric: {metric}")
    return metric == "surface_covariant_full"


def surface_direction_edge_difference(
    direction: torch.Tensor,
    normals: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    transported_dst = parallel_transport_vectors(
        direction[dst],
        normals[dst],
        normals[src],
    )
    return F.normalize(direction[src], dim=-1, eps=EPS) - transported_dst


def symmetric_relative_edge_difference(
    value: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """Bounded, scale-invariant local difference for positive attributes."""

    lhs = value[src]
    rhs = value[dst]
    return (lhs - rhs) / (lhs + rhs).clamp_min(EPS)


def root_graph_smoothness(
    field: GroomParameterField,
    edges: torch.Tensor,
    observation_confidence: torch.Tensor | None = None,
    *,
    normals: torch.Tensor | None = None,
    tangents: torch.Tensor | None = None,
    bitangents: torch.Tensor | None = None,
    smooth_field_metric: str = "ambient",
    include_geometry: bool = True,
    appearance_only: bool = False,
) -> torch.Tensor:
    if edges.numel() == 0:
        return next(field.parameters()).new_tensor(0.0)
    groom = field.decode()
    src, dst = edges[:, 0], edges[:, 1]
    if observation_confidence is None:
        edge_weight = groom.length.new_ones((edges.shape[0],))
    else:
        conf = observation_confidence.detach().reshape(-1).clamp(0.0, 1.0)
        edge_weight = 0.25 + (1.0 - torch.minimum(conf[src], conf[dst]))

    def weighted_mean(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 1:
            value = value.mean(dim=tuple(range(1, value.ndim)))
        return (value * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)

    def uniform_mean(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 1:
            value = value.mean(dim=tuple(range(1, value.ndim)))
        return value.mean()

    terms = [
        0.25 * weighted_mean((groom.root_color[src] - groom.root_color[dst]).square()),
        0.15 * weighted_mean((groom.tip_color[src] - groom.tip_color[dst]).square()),
        0.5 * weighted_mean((groom.opacity[src] - groom.opacity[dst]).square()),
        0.25 * weighted_mean((groom.tip_opacity[src] - groom.tip_opacity[dst]).square()),
    ]
    if not appearance_only:
        terms[:0] = [
            2.0 * weighted_mean((torch.log(groom.root_width[src].clamp_min(1.0e-6)) - torch.log(groom.root_width[dst].clamp_min(1.0e-6))).square()),
            0.8 * weighted_mean((torch.log(groom.tip_width[src].clamp_min(1.0e-6)) - torch.log(groom.tip_width[dst].clamp_min(1.0e-6))).square()),
            0.4
            * weighted_mean(
                (
                    torch.log(groom.width_taper[src].clamp_min(EPS))
                    - torch.log(groom.width_taper[dst].clamp_min(EPS))
                ).square()
            ),
            0.8
            * weighted_mean(
                (
                    torch.log(groom.child_radius[src].clamp_min(EPS))
                    - torch.log(groom.child_radius[dst].clamp_min(EPS))
                ).square()
            ),
        ]
    if include_geometry:
        if smooth_metric_uses_transport(smooth_field_metric):
            if normals is None or tangents is None or bitangents is None:
                raise ValueError("surface-aware root smoothing requires normals and tangent frames")
            direction = groom_direction_3d(groom, normals, tangents, bitangents)
            direction_difference = surface_direction_edge_difference(
                direction,
                normals,
                src,
                dst,
            )
        else:
            direction_difference = (
                groom.direction_local[src] - groom.direction_local[dst]
            )
        if smooth_metric_uses_full_relative_length_field(smooth_field_metric):
            length_difference = symmetric_relative_edge_difference(groom.length, src, dst)
            length_term = uniform_mean(length_difference.square())
        else:
            length_term = 4.0 * weighted_mean((groom.length[src] - groom.length[dst]).square())
        terms.extend(
            [
                length_term,
                1.0 * weighted_mean(direction_difference.square()),
                1.0
                * weighted_mean(
                    (
                        groom.brush_stiffness[src]
                        - groom.brush_stiffness[dst]
                    ).square()
                ),
                0.6 * weighted_mean((groom.curl_radius_ratio[src] - groom.curl_radius_ratio[dst]).square()),
                0.35 * weighted_mean((groom.curl_turns[src] - groom.curl_turns[dst]).square()),
                0.25
                * weighted_mean(
                    (torch.cos(groom.curl_phase[src]) - torch.cos(groom.curl_phase[dst])).square()
                    + (torch.sin(groom.curl_phase[src]) - torch.sin(groom.curl_phase[dst])).square()
                ),
                0.8 * weighted_mean((groom.clump_strength[src] - groom.clump_strength[dst]).square()),
            ]
        )
    return torch.stack(terms).sum()


def render_geometry_residual_graph_smoothness(
    model: WhiteTigerStage1Model,
    edges: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    observation_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth the active zero-centered geometry residual field."""

    field = model.active_geometry_residual()
    if field is None or edges.numel() == 0:
        return model.groom.length_raw.sum() * 0.0
    decoded = field.decode()
    src, dst = edges[:, 0], edges[:, 1]
    if observation_confidence is None:
        edge_weight = decoded.length.new_ones((edges.shape[0],))
    else:
        confidence = observation_confidence.detach().reshape(-1).clamp(0.0, 1.0)
        edge_weight = 0.25 + (1.0 - torch.minimum(confidence[src], confidence[dst]))

    def weighted_mean(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 1:
            value = value.mean(dim=tuple(range(1, value.ndim)))
        return (value * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)

    residual_world = local_components_to_world(
        decoded.direction_local,
        normals,
        tangents,
        bitangents,
        normalize=False,
    )
    transported_dst = parallel_transport_vector_field(
        residual_world[dst],
        normals[dst],
        normals[src],
    )
    direction_difference = residual_world[src] - transported_dst
    terms = [
        weighted_mean(
            (
                decoded.root_width_log_ratio[src]
                - decoded.root_width_log_ratio[dst]
            ).square()
        ),
        weighted_mean(
            (
                decoded.tip_width_logit_delta[src]
                - decoded.tip_width_logit_delta[dst]
            ).square()
        ),
        weighted_mean(
            (
                decoded.width_taper_log_ratio[src]
                - decoded.width_taper_log_ratio[dst]
            ).square()
        ),
        weighted_mean((decoded.curl_radius_log_ratio[src] - decoded.curl_radius_log_ratio[dst]).square()),
        weighted_mean(
            (
                decoded.child_radius_log_ratio[src]
                - decoded.child_radius_log_ratio[dst]
            ).square()
        ),
        weighted_mean((decoded.clump_strength[src] - decoded.clump_strength[dst]).square()),
        2.0 * weighted_mean(direction_difference.square().sum(dim=-1)),
    ]
    length_coordinate = (
        field.length_raw
        if model.render_geometry_parameterization
        in {
            "zero_centered_unbounded_log_length_residual",
            "zero_centered_asinh_log_length_residual",
        }
        else decoded.length
    )
    terms.insert(
        0,
        weighted_mean(
            (length_coordinate[src] - length_coordinate[dst]).square()
        ),
    )
    return torch.stack(terms).sum()


def guide_root_graph_smoothness(
    model: WhiteTigerStage1Model,
    edges: torch.Tensor,
    *,
    smooth_field_metric: str = "ambient",
    guide_length_smooth_mode: str = "edge_relative",
    smooth_graph_k: int = 8,
) -> torch.Tensor:
    if not model.guide_enabled() or edges.numel() == 0:
        return model.groom.length_raw.sum() * 0.0
    ranges = model.groom.ranges
    src, dst = edges[:, 0], edges[:, 1]
    guide_length = decode_positive_asinh_ratio(
        model.guide_length_raw,
        model.guide_length_reference,
    )
    guide_width = decode_positive_asinh_ratio(
        model.guide_root_width_raw,
        model.guide_root_width_reference,
    )
    guide_tip_ratio_coordinate = torch.asinh(model.guide_tip_width_ratio_raw)
    guide_taper_coordinate = torch.asinh(model.guide_width_taper_raw)
    guide_brush_stiffness = decode_brush_stiffness(
        model.guide_brush_stiffness_raw
    )
    guide_child_radius = decode_positive_asinh_ratio(
        model.guide_child_radius_raw,
        model.guide_child_radius_reference,
    )
    guide_clump = GroomParameterField._decode_range(model.guide_clump_strength_raw, ranges.clump_strength)

    def mean_edge(value: torch.Tensor) -> torch.Tensor:
        diff = value[src] - value[dst]
        return diff.square().mean()

    if guide_length_smooth_mode not in GUIDE_LENGTH_SMOOTH_MODES:
        raise ValueError(f"unknown guide length smooth mode: {guide_length_smooth_mode}")
    if guide_length_smooth_mode == "intrinsic_density_invariant":
        metric_graph = model.guide_surface_smoothing_graph(smooth_graph_k)
        if (
            metric_graph.edges.shape != edges.shape
            or metric_graph.edges.data_ptr() != edges.data_ptr()
        ):
            raise RuntimeError(
                "intrinsic density-invariant guide smoothing requires the surface guide graph"
            )
        guide_length_term = density_invariant_log_scalar_smoothness(
            guide_length,
            metric_graph,
            metric_graph.reference_spacing,
        )
    elif smooth_metric_uses_full_relative_length_field(smooth_field_metric):
        guide_length_term = symmetric_relative_edge_difference(
            guide_length,
            src,
            dst,
        ).square().mean()
    else:
        guide_length_term = 4.0 * mean_edge(guide_length)

    terms = [
        guide_length_term,
        1.6 * mean_edge(torch.log(guide_width.clamp_min(1.0e-6))),
        0.8 * mean_edge(guide_tip_ratio_coordinate),
        0.4 * mean_edge(guide_taper_coordinate),
        1.0 * mean_edge(guide_brush_stiffness),
        0.8 * mean_edge(torch.log(guide_child_radius.clamp_min(EPS))),
        0.8 * mean_edge(guide_clump),
    ]
    if float(model.shape_curl_scale) > 0.0:
        guide_curl = decode_positive_softplus(
            model.guide_curl_radius_ratio_raw
        )
        terms.append(0.7 * mean_edge(guide_curl))
        guide_turn_coordinate = torch.asinh(model.guide_curl_turns_raw)
        terms.append(0.35 * mean_edge(guide_turn_coordinate))
    guide_direction = model.guide_direction_world()
    if guide_direction is not None:
        guide_normals, guide_tangents, guide_bitangents = model.guide_normals_and_tangent_frames()
        if smooth_metric_uses_transport(smooth_field_metric):
            direction_difference = surface_direction_edge_difference(
                guide_direction,
                guide_normals,
                src,
                dst,
            )
            terms.append(1.2 * direction_difference.square().mean())
        else:
            terms.append(1.2 * mean_edge(guide_direction))
    return torch.stack(terms).sum()


def effective_groom_graph_smoothness(
    groom,
    edges: torch.Tensor,
    normals: torch.Tensor,
    tangents: torch.Tensor,
    bitangents: torch.Tensor,
    ranges: GroomRanges,
    observation_confidence: torch.Tensor | None = None,
    *,
    smooth_field_metric: str = "ambient",
) -> torch.Tensor:
    """Smooth the final groom field after guide interpolation and residuals.

    The raw render-root field and the guide field can both be smooth while their
    combined effective controls still contain isolated long/curly strokes.  This
    regularizer acts on the actual controls passed to strand generation.
    """
    if edges.numel() == 0:
        return groom.length.sum() * 0.0
    src, dst = edges[:, 0], edges[:, 1]
    if observation_confidence is None:
        edge_weight = groom.length.new_ones((edges.shape[0],))
    else:
        conf = observation_confidence.detach().reshape(-1).clamp(0.0, 1.0)
        edge_weight = 0.25 + (1.0 - torch.minimum(conf[src], conf[dst]))

    def weighted_mean(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 1:
            value = value.mean(dim=tuple(range(1, value.ndim)))
        return (value * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)

    direction = groom_direction_3d(groom, normals, tangents, bitangents)
    tip_ratio = groom.tip_width / groom.root_width.clamp_min(EPS)
    tip_ratio_eps = torch.as_tensor(
        torch.finfo(tip_ratio.dtype).eps,
        device=tip_ratio.device,
        dtype=tip_ratio.dtype,
    )
    tip_ratio_logit = torch.logit(
        tip_ratio.clamp(tip_ratio_eps, 1.0 - tip_ratio_eps)
    )
    curl_wavenumber_magnitude = (
        2.0 * torch.pi * groom.curl_radius_ratio * groom.curl_turns.abs()
    )

    length_difference = symmetric_relative_edge_difference(groom.length, src, dst)
    if smooth_metric_uses_full_relative_length_field(smooth_field_metric):
        # Flow confidence describes orientation evidence, not length evidence.
        # Length therefore uses the complete surface graph uniformly.
        length_term = length_difference.square().mean()
    else:
        length_term = weighted_mean(length_difference.square())

    if smooth_metric_uses_transport(smooth_field_metric):
        direction_difference = surface_direction_edge_difference(direction, normals, src, dst)
    else:
        direction_difference = direction[src] - direction[dst]

    terms = [
        length_term,
        1.4
        * weighted_mean(
            (
                torch.log(groom.root_width[src].clamp_min(1.0e-6))
                - torch.log(groom.root_width[dst].clamp_min(1.0e-6))
            ).square()
        ),
        0.8 * weighted_mean((tip_ratio_logit[src] - tip_ratio_logit[dst]).square()),
        0.4
        * weighted_mean(
            (
                torch.log(groom.width_taper[src].clamp_min(EPS))
                - torch.log(groom.width_taper[dst].clamp_min(EPS))
            ).square()
        ),
        1.0
        * weighted_mean(
            (
                groom.brush_stiffness[src]
                - groom.brush_stiffness[dst]
            ).square()
        ),
        1.8 * weighted_mean((groom.curl_radius_ratio[src] - groom.curl_radius_ratio[dst]).square()),
        1.6
        * weighted_mean(
            (
                torch.log1p(curl_wavenumber_magnitude[src])
                - torch.log1p(curl_wavenumber_magnitude[dst])
            ).square()
        ),
        1.4
        * weighted_mean(
            (
                torch.log(groom.child_radius[src].clamp_min(EPS))
                - torch.log(groom.child_radius[dst].clamp_min(EPS))
            ).square()
        ),
        1.0 * weighted_mean((groom.clump_strength[src] - groom.clump_strength[dst]).square()),
        2.0 * weighted_mean(direction_difference.square().sum(dim=-1)),
    ]
    return torch.stack(terms).sum()


@torch.no_grad()
def groom_parameter_stats(field: GroomParameterField) -> dict[str, dict[str, float]]:
    groom = field.decode()

    def summarize(value: torch.Tensor) -> dict[str, float]:
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        q = torch.quantile(flat, torch.tensor([0.05, 0.50, 0.95], device=flat.device))
        return {
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()),
            "min": float(flat.min().cpu()),
            "p05": float(q[0].cpu()),
            "p50": float(q[1].cpu()),
            "p95": float(q[2].cpu()),
            "max": float(flat.max().cpu()),
        }

    return {
        "length": summarize(groom.length),
        "root_width": summarize(groom.root_width),
        "tip_width": summarize(groom.tip_width),
        "width_taper": summarize(groom.width_taper),
        "direction_local": summarize(groom.direction_local),
        "brush_stiffness": summarize(groom.brush_stiffness),
        "curl_radius_ratio": summarize(groom.curl_radius_ratio),
        "curl_radius": summarize(groom.length * groom.curl_radius_ratio),
        "curl_turns": summarize(groom.curl_turns),
        "curl_radius_x_abs_turns": summarize(
            groom.length * groom.curl_radius_ratio * groom.curl_turns.abs()
        ),
        "curl_wavenumber_magnitude": summarize(
            2.0 * torch.pi * groom.curl_radius_ratio * groom.curl_turns.abs()
        ),
        "child_radius": summarize(groom.child_radius),
        "clump_strength": summarize(groom.clump_strength),
        "opacity": summarize(groom.opacity),
        "tip_opacity": summarize(groom.tip_opacity),
    }


@torch.no_grad()
def effective_groom_stats(model: WhiteTigerStage1Model) -> dict[str, dict[str, float]] | None:
    if not model.guide_enabled():
        return None
    _, _, roots_local = model.roots_and_normals()
    groom = model.apply_guide_controls(model.groom.decode(), roots_local)

    def summarize(value: torch.Tensor) -> dict[str, float]:
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        q = torch.quantile(flat, torch.tensor([0.05, 0.50, 0.95], device=flat.device))
        return {
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()),
            "min": float(flat.min().cpu()),
            "p05": float(q[0].cpu()),
            "p50": float(q[1].cpu()),
            "p95": float(q[2].cpu()),
            "max": float(flat.max().cpu()),
        }

    return {
        "length": summarize(groom.length),
        "root_width": summarize(groom.root_width),
        "direction_local": summarize(groom.direction_local),
        "brush_stiffness": summarize(groom.brush_stiffness),
        "curl_radius_ratio": summarize(groom.curl_radius_ratio),
        "curl_radius": summarize(groom.length * groom.curl_radius_ratio),
        "curl_turns": summarize(groom.curl_turns),
        "curl_radius_x_abs_turns": summarize(
            groom.length * groom.curl_radius_ratio * groom.curl_turns.abs()
        ),
        "curl_wavenumber_magnitude": summarize(
            2.0 * torch.pi * groom.curl_radius_ratio * groom.curl_turns.abs()
        ),
        "child_radius": summarize(groom.child_radius),
        "clump_strength": summarize(groom.clump_strength),
    }


@torch.no_grad()
def render_geometry_residual_stats(
    model: WhiteTigerStage1Model,
) -> dict[str, dict[str, float]] | None:
    """Report the active normalized residual field without legacy endpoints."""

    field = model.active_geometry_residual()
    if field is None:
        return None
    residual = field.decode()
    length_coordinate = (
        field.length_raw
        if model.render_geometry_parameterization
        in {
            "zero_centered_unbounded_log_length_residual",
            "zero_centered_asinh_log_length_residual",
        }
        else residual.length
    )

    def summarize(value: torch.Tensor) -> dict[str, float]:
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "p05": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "max": 0.0,
            }
        q = torch.quantile(
            flat,
            torch.tensor([0.05, 0.50, 0.95], device=flat.device),
        )
        return {
            "mean": float(flat.mean().cpu()),
            "std": float(flat.std(unbiased=False).cpu()),
            "min": float(flat.min().cpu()),
            "p05": float(q[0].cpu()),
            "p50": float(q[1].cpu()),
            "p95": float(q[2].cpu()),
            "max": float(flat.max().cpu()),
        }

    return {
        "length": summarize(length_coordinate),
        "curl_radius_log_ratio": summarize(residual.curl_radius_log_ratio),
        "child_radius_log_ratio": summarize(residual.child_radius_log_ratio),
        "clump_strength": summarize(residual.clump_strength),
        "direction_component": summarize(residual.direction_local),
        "direction_magnitude": summarize(
            torch.linalg.norm(residual.direction_local, dim=-1)
        ),
    }


@torch.no_grad()
def summarize_values(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(flat.numel()),
        "mean": float(flat.mean().cpu()),
        "p50": float(torch.quantile(flat, 0.50).cpu()),
        "p90": float(torch.quantile(flat, 0.90).cpu()),
        "p95": float(torch.quantile(flat, 0.95).cpu()),
        "max": float(flat.max().cpu()),
    }


@torch.no_grad()
def lifecycle_effective_groom(model: WhiteTigerStage1Model):
    """Decode the controls that actually generate strands at this iteration."""

    base_groom = model.groom.decode()
    if not model.guide_enabled():
        return base_groom
    _, normals, roots_local = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    return model.apply_guide_controls(
        base_groom,
        roots_local,
        normals,
        tangents,
        bitangents,
    )


@torch.no_grad()
def lifecycle_subset_report(
    model: WhiteTigerStage1Model,
    stats,
    scores: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, dict[str, float]]:
    if indices.numel() == 0:
        return {}
    ids = indices.detach().long().to(device=model.bary_logits.device)
    base_groom = model.groom.decode()
    groom = lifecycle_effective_groom(model)
    contribution = stats.gaussian_contrib_sum.reshape(-1).to(device=ids.device)
    visible = stats.visible_count.reshape(-1).to(device=ids.device)
    sample_count = (
        stats.gaussian_sample_count.reshape(-1).to(device=ids.device)
        if getattr(stats, "gaussian_sample_count", None) is not None
        else torch.ones_like(visible)
    )
    return {
        "need": summarize_values(scores["need"][ids]),
        "residual": summarize_values(scores["residual"][ids]),
        "gaussian_grad": summarize_values(scores["gaussian_grad"][ids]),
        "root_grad": summarize_values(scores["root_grad"][ids]),
        "contribution": summarize_values(contribution[ids]),
        "visible": summarize_values(visible[ids]),
        "sample_count": summarize_values(sample_count[ids]),
        "visible_per_sample": summarize_values((visible / sample_count.clamp_min(1.0))[ids]),
        "contribution_per_sample": summarize_values((contribution / sample_count.clamp_min(1.0))[ids]),
        "length": summarize_values(groom.length[ids]),
        "base_length": summarize_values(base_groom.length[ids]),
        "root_width": summarize_values(groom.root_width[ids]),
        "tip_width": summarize_values(groom.tip_width[ids]),
        "opacity": summarize_values(groom.opacity[ids]),
        "direction_local": summarize_values(groom.direction_local[ids]),
    }


@torch.no_grad()
def lifecycle_global_report(model: WhiteTigerStage1Model, stats, scores: dict[str, torch.Tensor]) -> dict[str, dict[str, float] | float]:
    base_groom = model.groom.decode()
    groom = lifecycle_effective_groom(model)
    visible = stats.visible_count.reshape(-1)
    contribution = stats.gaussian_contrib_sum.reshape(-1)
    sample_count = (
        stats.gaussian_sample_count.reshape(-1)
        if getattr(stats, "gaussian_sample_count", None) is not None
        else torch.ones_like(visible)
    )
    return {
        "need": summarize_values(scores["need"]),
        "residual": summarize_values(scores["residual"]),
        "contribution": summarize_values(contribution),
        "visible": summarize_values(visible),
        "sample_count": summarize_values(sample_count),
        "visible_per_sample": summarize_values(visible / sample_count.clamp_min(1.0)),
        "contribution_per_sample": summarize_values(contribution / sample_count.clamp_min(1.0)),
        "length": summarize_values(groom.length),
        "base_length": summarize_values(base_groom.length),
        "root_width": summarize_values(groom.root_width),
        "opacity": summarize_values(groom.opacity),
    }


@torch.no_grad()
def lifecycle_selection_report(scores: dict[str, torch.Tensor]) -> dict[str, float]:
    report: dict[str, float] = {}
    for key in [
        "threshold_candidate_count",
        "local_max_candidate_count",
        "parent_budget",
        "budget_saturated",
    ]:
        value = scores.get(key)
        if value is None:
            continue
        report[key] = float(value.detach().reshape(-1)[0].cpu())
    return report


@torch.no_grad()
def lifecycle_spatial_report(
    state,
    indices: torch.Tensor,
    *,
    bin_count: int = 10,
) -> dict[str, object]:
    if indices.numel() == 0:
        return {}
    points = state.points.detach().float()
    ids = indices.detach().long().to(device=points.device)
    selected = points[ids]
    mins = points.amin(dim=0)
    spans = (points.amax(dim=0) - mins).clamp_min(1.0e-8)
    all_norm = ((points - mins) / spans).clamp(0.0, 1.0)
    selected_norm = ((selected - mins) / spans).clamp(0.0, 1.0)

    def hist_fraction(values: torch.Tensor) -> torch.Tensor:
        bucket = torch.clamp((values * float(bin_count)).long(), min=0, max=bin_count - 1)
        hist = torch.bincount(bucket, minlength=bin_count).float()
        return hist / hist.sum().clamp_min(1.0)

    axes: dict[str, dict[str, object]] = {}
    for axis, name in enumerate(("x", "y", "z")):
        all_hist = hist_fraction(all_norm[:, axis])
        selected_hist = hist_fraction(selected_norm[:, axis])
        selected_safe = selected_hist.clamp_min(1.0e-12)
        entropy = -(selected_safe * selected_safe.log()).sum() / torch.log(
            torch.tensor(float(bin_count), device=selected_safe.device)
        )
        axes[name] = {
            "selected_fraction": [float(v) for v in selected_hist.cpu().tolist()],
            "all_fraction": [float(v) for v in all_hist.cpu().tolist()],
            "l1_from_all": float(torch.abs(selected_hist - all_hist).sum().cpu()),
            "selected_max_bin_fraction": float(selected_hist.max().cpu()),
            "selected_entropy": float(entropy.cpu()),
            "selected_mean": float(selected_norm[:, axis].mean().cpu()),
            "selected_p50": float(torch.quantile(selected_norm[:, axis], 0.50).cpu()),
        }

    face_ids = state.face_ids.detach().long().to(device=points.device)
    selected_faces = face_ids[ids]
    return {
        "selected_root_count": int(ids.numel()),
        "selected_unique_face_count": int(torch.unique(selected_faces).numel()),
        "all_unique_face_count": int(torch.unique(face_ids).numel()),
        "axes": axes,
    }


@dataclass(frozen=True)
class Stage1Config:
    data_root: str
    mesh_path: str
    output_dir: str
    face_tangent_field: str = ""
    root_count: int = 10000
    root_init_method: str = "fps"
    candidate_multiplier: float = 10.0
    iterations: int = 30000
    eval_every: int = 1000
    save_every: int = 5000
    stage_save_iters: str = ""
    test_stride: int = 6
    train_views: str = ""
    test_views: str = ""
    seed: int = 13
    expected_width: int = 1920
    expected_height: int = 1080
    init_mesh_scale: float = 1.28
    init_mesh_translation: tuple[float, float, float] = (0.0, 0.32, 0.02)
    init_groom_length: float = 0.060
    samples: int = 64
    min_segments: int = 10
    segment_length_origin: float = 0.010
    segments_per_unit_length: float = 84.19047619047619
    segments_per_unit_complexity: float = 23.771428571428572
    child_count: int = 8
    gaussian_length_overlap: float = 1.45
    projected_init_views: int = 24
    projected_init_min_confidence: float = 0.08
    projected_init_depth_abs_tolerance: float = 0.03
    projected_init_depth_rel_tolerance: float = 0.01
    projected_init_local_depth_kernel: int = 7
    projected_init_front_normal_z: float = 0.15
    projected_init_mask_edge_kernel: int = 9
    projected_init_view_angle_power: float = 1.0
    clean_flow_target: str = ""
    clean_flow_init: bool = False
    clean_flow_init_k: int = 8
    clean_flow_init_min_confidence: float = 0.03
    clean_flow_anchor_min_confidence: float = 0.35
    clean_flow_length_init: bool = False
    clean_flow_length_init_scale: float = 0.30
    clean_flow_length_init_min_confidence: float = 0.50
    clean_flow_guide_anchor_weight: float = 0.0
    clean_flow_guide_length_anchor_weight: float = 0.0
    clean_flow_guide_length_anchor_reduction: str = "mean_l1"
    clean_flow_3d_smooth_weight: float = 0.0
    guide_root_count: int = 0
    guide_candidate_multiplier: float = 8.0
    guide_roots_from_clean_flow: bool = False
    guide_interpolation_k: int = 8
    geometry_residual_domain: str = "render"
    secondary_guide_root_count: int = 0
    secondary_guide_candidate_multiplier: float = 16.0
    secondary_guide_interpolation_k: int = 8
    secondary_guide_smooth_k: int = 32
    render_geometry_parameterization: str = "absolute_endpoint"
    guide_length_residual_scale: float = 0.0
    guide_direction_residual_scale: float = 1.0
    guide_width_residual_scale: float = 1.0
    guide_child_radius_residual_scale: float = 1.0
    guide_clump_residual_scale: float = 1.0
    guide_curl_residual_scale: float = 1.0
    guide_prior_weight: float = 0.0
    guide_prior_direction_weight: float = 1.0
    guide_prior_curl_weight: float = 0.08
    guide_prior_length_weight: float = 0.0
    guide_prior_width_weight: float = 0.0
    guide_prior_child_radius_weight: float = 0.0
    guide_prior_clump_weight: float = 0.0
    guide_support_gauge_weight: float = 0.0
    guide_view_sh_support: bool = False
    guide_view_sh_scale: float = 0.20
    lr_guide_view_sh: float = 2.0e-2
    view_gated_ownership_support: bool = False
    view_gate_geometry_support: bool = False
    view_gate_length_confidence_support: bool = False
    view_gate_floor: float = 0.0
    view_gate_normalization: str = "raw_q95"
    render_length_prior_coordinate: str = "decoded"
    render_length_prior_reduction: str = "mean_l1"
    guide_smooth_weight: float = 0.0
    guide_length_smooth_mode: str = "edge_relative"
    guide_residual_unlock_start: int = 0
    guide_residual_unlock_end: int = 0
    guide_residual_initial_multiplier: float = 1.0
    guide_coverage_residual_unlock_start: int = 0
    guide_coverage_residual_unlock_end: int = 0
    guide_coverage_residual_initial_multiplier: float = 1.0
    guide_freeze_until: int = 0
    guide_length_freeze_until: int = -1
    shape_detail_freeze_until: int = 0
    shape_detail_unlock_end: int = 0
    secondary_shape_residual_unlock_start: int = 0
    secondary_shape_residual_unlock_end: int = 0
    shape_curl_scale: float = 1.0
    guide_densify_start: int = 0
    guide_densify_interval: int = 0
    guide_densify_until: int = 0
    guide_densify_score_threshold: float = 0.0
    guide_densify_max_splits_per_event: int = 0
    guide_densify_policy: str = "global_score_budget"
    guide_densify_children_per_parent: int = 1
    guide_densify_neighbor_count: int = 12
    guide_densify_candidate_rings: int = 3
    guide_densify_candidate_face_count: int = 32
    guide_densify_min_child_distance: float = 0.0
    guide_densify_render_root_k: int = 8
    lr_groom: float = 1.4e-2
    lr_high_frequency_shape_scale: float = 1.0
    lr_color: float = 2.0e-2
    color_freeze_until: int = 0
    gaussian_rgb_residual_support: bool = False
    gaussian_rgb_residual_control_points: int = 36
    gaussian_rgb_residual_scale: float = 0.20
    gaussian_rgb_residual_unlock_start: int = 10000
    gaussian_rgb_residual_unlock_end: int = 20000
    gaussian_rgb_residual_initial_multiplier: float = 0.0
    lr_root: float = 7.5e-4
    lr_calibration: float = 5.0e-4
    rgb_weight: float = 1.0
    random_backing_loss_weight: float = 0.25
    mask_weight: float = 0.15
    rgb_flow_weight: float = 0.0
    rgb_flow_detail_weight: float = 0.0
    rgb_flow_min_confidence: float = 0.08
    rgb_flow_exclude_color_gradients: bool = False
    loss_mask_edge_kernel: int = 1
    smooth_graph_mode: str = "euclidean_knn"
    smooth_graph_k: int = 8
    smooth_field_metric: str = "ambient"
    smooth_weight: float = 0.04
    geometry_residual_smooth_scale: float = 1.0
    effective_smooth_weight: float = 0.0
    root_move_reg_weight: float = 0.003
    compute_lpips: bool = False
    white_background: bool = True
    random_backing_color: bool = True
    backing_color_min: float = 0.05
    backing_color_max: float = 0.85
    random_mesh_backing_texture: bool = True
    mesh_backing_texture_strength: float = 0.30
    mesh_backing_texture_octaves: int = 5
    mesh_no_penetration_support: bool = False
    mesh_no_penetration_sdf: str = ""
    mesh_no_penetration_weight: float = 0.0
    mesh_no_penetration_root_batch: int = 16384
    strand_crossing_support: bool = False
    strand_crossing_weight: float = 0.0
    strand_crossing_refresh_interval: int = 0
    strand_crossing_query_batch: int = 50000
    strand_crossing_exact_pair_batch: int = 250000
    mesh_depth_clipping: bool = True
    mesh_depth_abs_tolerance: float = 0.018
    mesh_depth_rel_tolerance: float = 0.004
    mesh_depth_local_kernel: int = 1
    mesh_backing_compositing: bool = True
    gpu_memory_limit_gb: float = 0.0
    gpu_memory_check_interval: int = 20
    densify_warmup: int = 500
    densify_interval: int = 100
    densify_until: int = 12000
    densify_score_threshold: float = 2.5e-5
    densify_min_contribution: float = 0.45
    densify_residual_weight: float = 0.0
    densify_residual_mode: str = "root_pixel"
    densify_residual_pool_radius: int = 15
    densify_residual_alpha_weight: float = 1.0
    densify_residual_rgb_weight: float = 0.25
    densify_pixel_evidence_topk: int = 4096
    densify_pixel_evidence_root_k: int = 4
    densify_pixel_evidence_min: float = 0.02
    densify_pixel_evidence_chunk: int = 512
    lifecycle_score_mode: str = "raw"
    local_child_color_support: bool = False
    local_child_color_scale: float = 0.20
    max_splits_per_event: int = 256
    split_children_per_parent: int = 2
    split_neighbor_count: int = 12
    split_candidate_rings: int = 3
    split_candidate_face_count: int = 32
    split_min_child_distance: float = 0.0
    prune_start: int = 999999
    prune_interval: int = 100
    prune_min_contribution: float = 0.08
    prune_min_opacity: float = 0.0
    prune_max_fraction: float = 0.05
    resume_checkpoint: str = ""
    resume_optimizer: bool = True


def stage1_config_from_checkpoint_mapping(raw: dict) -> Stage1Config:
    """Load only checkpoints whose config exactly matches the current schema."""

    data = dict(raw)
    if isinstance(data.get("init_mesh_translation"), list):
        data["init_mesh_translation"] = tuple(float(v) for v in data["init_mesh_translation"])
    known = {field.name for field in fields(Stage1Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise TypeError(f"unsupported Stage1 checkpoint config fields: {unknown}")
    missing = sorted(known - set(data))
    if missing:
        raise TypeError(f"incomplete current checkpoint config fields: {missing}")
    return Stage1Config(**data)


def clean_flow_guide_length_anchor_reliable_fraction(
    guide_clean_flow_length_target: torch.Tensor,
    guide_clean_flow_length_confidence: torch.Tensor,
) -> torch.Tensor:
    """Return the fraction of primary guides with reliable length anchors."""

    target = guide_clean_flow_length_target.reshape(-1)
    confidence = guide_clean_flow_length_confidence.to(
        device=target.device,
        dtype=target.dtype,
    ).reshape(-1)
    if target.shape != confidence.shape:
        raise ValueError(
            "guide clean-flow length target and confidence must have equal size"
        )
    if target.numel() == 0:
        return target.new_zeros(())
    reliable = (
        torch.isfinite(target)
        & (target > 0.0)
        & torch.isfinite(confidence)
        & (confidence > 0.0)
    )
    return reliable.to(dtype=target.dtype).mean()


def clean_flow_guide_length_anchor_loss(
    guide_length_raw: torch.Tensor,
    guide_length_reference: torch.Tensor,
    guide_clean_flow_length_target: torch.Tensor,
    guide_clean_flow_length_confidence: torch.Tensor,
    source_area_weights: torch.Tensor | None = None,
    clean_flow_length_init_scale: float = 1.0,
    reduction: str = "mean_l1",
) -> torch.Tensor:
    """Anchor primary-guide physical length to clean-flow data identity.

    The stored target is the clean-flow target after the initialization scale;
    dividing by that scale restores the data-identity length. Only finite,
    positive targets with positive stored confidence contribute. Confidence
    and intrinsic source-area quadrature weights are detached from the loss.
    """

    reduction = str(reduction)
    if reduction not in {"mean_l1", "tail_concentration"}:
        raise ValueError(
            "clean-flow guide length anchor reduction must be mean_l1 or "
            "tail_concentration"
        )
    raw = guide_length_raw.reshape(-1)
    reference = guide_length_reference.to(
        device=raw.device,
        dtype=raw.dtype,
    ).reshape(-1)
    target = guide_clean_flow_length_target.to(
        device=raw.device,
        dtype=raw.dtype,
    ).reshape(-1)
    confidence = guide_clean_flow_length_confidence.to(
        device=raw.device,
        dtype=raw.dtype,
    ).reshape(-1)
    if not (raw.shape == reference.shape == target.shape == confidence.shape):
        raise ValueError(
            "guide length raw, reference, target, and confidence must have equal size"
        )
    scale = float(clean_flow_length_init_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("clean-flow length initialization scale must be positive")
    if source_area_weights is None:
        area = torch.ones_like(target)
    else:
        area = source_area_weights.to(device=raw.device, dtype=raw.dtype).reshape(-1)
        if area.shape != target.shape:
            raise ValueError("guide source-area weights must match guide coordinates")
        if not bool(torch.isfinite(area).all()):
            raise ValueError("guide source-area weights must be finite")
        if bool((area < 0.0).any()):
            raise ValueError("guide source-area weights must be non-negative")
    reliable = (
        torch.isfinite(target)
        & (target > 0.0)
        & torch.isfinite(confidence)
        & (confidence > 0.0)
    )
    if not bool(reliable.any()):
        return raw.sum() * 0.0
    current = decode_positive_asinh_ratio(raw, reference)
    identity_target = target[reliable].detach() / scale
    weight = confidence[reliable].detach() * area[reliable].detach()
    error = torch.abs(torch.log(current[reliable] / identity_target))
    weight_sum = weight.sum().clamp_min(EPS)
    weighted_mean_abs = (error * weight).sum() / weight_sum
    if reduction == "mean_l1":
        return weighted_mean_abs

    weighted_l2 = (
        (error.square() * weight).sum() / weight_sum
        + torch.as_tensor(
            torch.finfo(error.dtype).tiny,
            device=error.device,
            dtype=error.dtype,
        )
    ).sqrt()
    weighted_l2 = weighted_l2 - torch.as_tensor(
        torch.finfo(error.dtype).tiny,
        device=error.device,
        dtype=error.dtype,
    ).sqrt()
    weighted_l4 = fourth_moment_norm(error, weight)
    concentration = (weighted_l4 - weighted_l2).clamp_min(0.0)
    active = weight > 0.0
    if not bool(active.any()) or bool(torch.all(error[active] == error[active][0])):
        concentration = concentration * 0.0
    return weighted_mean_abs + concentration


def primary_guide_length_anchor_metrics(
    model: "WhiteTigerStage1Model",
    config: Stage1Config,
    *,
    source_area_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the primary-guide length-anchor loss and reliable fraction."""

    if not model.guide_enabled():
        zero = model.groom.length_raw.sum() * 0.0
        return zero, zero.detach()
    if source_area_weights is None:
        source_area_weights = model.guide_surface_smoothing_graph(
            config.guide_interpolation_k
        ).source_area_weights
    loss = clean_flow_guide_length_anchor_loss(
        model.guide_length_raw,
        model.guide_length_reference,
        model.guide_clean_flow_length_target,
        model.guide_clean_flow_length_confidence,
        source_area_weights,
        config.clean_flow_length_init_scale,
        reduction=config.clean_flow_guide_length_anchor_reduction,
    )
    fraction = clean_flow_guide_length_anchor_reliable_fraction(
        model.guide_clean_flow_length_target,
        model.guide_clean_flow_length_confidence,
    )
    return loss, fraction


class WhiteTigerStage1Model(torch.nn.Module):
    def __init__(
        self,
        mesh: TriangleMesh,
        face_normals: np.ndarray,
        face_tangents: np.ndarray | None,
        face_ids: np.ndarray,
        barycentric: np.ndarray,
        ranges: GroomRanges,
        device: torch.device,
        init_scale: float = 1.25,
        init_translation: tuple[float, float, float] = (0.0, 0.32, 0.0),
        init_groom_length: float = 0.060,
        max_child_count: int = 8,
        local_child_color_support: bool = False,
        local_child_color_scale: float = 0.20,
        gaussian_rgb_residual_support: bool = False,
        gaussian_rgb_residual_control_points: int = 36,
        gaussian_rgb_residual_scale: float = 0.20,
        guide_view_sh_support: bool = False,
        guide_view_sh_scale: float = 0.20,
        guide_face_ids: np.ndarray | None = None,
        guide_barycentric: np.ndarray | None = None,
        guide_region_ids: np.ndarray | None = None,
        guide_interpolation_k: int = 8,
        geometry_residual_domain: str = "render",
        secondary_guide_face_ids: np.ndarray | None = None,
        secondary_guide_barycentric: np.ndarray | None = None,
        secondary_guide_parent_ids: np.ndarray | None = None,
        secondary_guide_interpolation_k: int = 8,
        render_geometry_parameterization: str = "absolute_endpoint",
        guide_length_residual_scale: float = 0.0,
        guide_direction_residual_scale: float = 1.0,
        guide_width_residual_scale: float = 1.0,
        guide_child_radius_residual_scale: float = 1.0,
        guide_clump_residual_scale: float = 1.0,
        guide_curl_residual_scale: float = 1.0,
        shape_curl_scale: float = 1.0,
        view_gate_geometry_support: bool = False,
        view_gate_length_confidence_support: bool = False,
    ) -> None:
        super().__init__()
        self.max_child_count = max(1, int(max_child_count))
        self.local_child_color_scale = float(local_child_color_scale)
        self.view_gate_geometry_support = bool(view_gate_geometry_support)
        self.view_gate_length_confidence_support = bool(
            view_gate_length_confidence_support
        )
        self.gaussian_rgb_residual_multiplier = 1.0
        if bool(gaussian_rgb_residual_support) and self.max_child_count != 1:
            raise ValueError(
                "Gaussian RGB residual currently requires child_count=1 so every "
                "profile row has one persistent render-root owner"
            )
        self.guide_interpolation_k = max(1, int(guide_interpolation_k))
        self.geometry_residual_domain = str(geometry_residual_domain)
        if self.geometry_residual_domain not in {"render", "secondary_guide"}:
            raise ValueError(
                "geometry_residual_domain must be render or secondary_guide"
            )
        self.secondary_guide_interpolation_k = max(
            1,
            int(secondary_guide_interpolation_k),
        )
        self.render_geometry_parameterization = str(render_geometry_parameterization)
        if self.render_geometry_parameterization not in {
            "absolute_endpoint",
            "zero_centered_residual",
            "zero_centered_log_length_residual",
            "zero_centered_unbounded_log_length_residual",
            "zero_centered_asinh_log_length_residual",
        }:
            raise ValueError(
                "render_geometry_parameterization must be absolute_endpoint, "
                "zero_centered_residual, zero_centered_log_length_residual, "
                "zero_centered_unbounded_log_length_residual, or "
                "zero_centered_asinh_log_length_residual"
            )
        self.guide_length_residual_scale = max(0.0, min(1.0, float(guide_length_residual_scale)))
        self.guide_direction_residual_scale = max(
            0.0,
            min(1.0, float(guide_direction_residual_scale)),
        )
        self.guide_width_residual_scale = max(0.0, min(1.0, float(guide_width_residual_scale)))
        self.guide_child_radius_residual_scale = max(0.0, min(1.0, float(guide_child_radius_residual_scale)))
        self.guide_clump_residual_scale = max(0.0, min(1.0, float(guide_clump_residual_scale)))
        self.guide_curl_residual_scale = max(0.0, min(1.0, float(guide_curl_residual_scale)))
        self.guide_residual_multiplier = 1.0
        self.guide_coverage_residual_multiplier = 1.0
        self.shape_detail_multiplier = 1.0
        self.secondary_shape_residual_multiplier = 1.0
        self.shape_curl_scale = max(0.0, float(shape_curl_scale))
        self.init_groom_length = float(init_groom_length)
        self.register_buffer("vertices", torch.from_numpy(mesh.vertices).to(device=device))
        self.register_buffer("faces", torch.from_numpy(mesh.faces).to(device=device, dtype=torch.long))
        self.register_buffer("face_ids", torch.from_numpy(face_ids).to(device=device, dtype=torch.long))
        self.register_buffer("face_normals", torch.from_numpy(face_normals).to(device=device))
        if face_tangents is None:
            self.register_buffer("face_tangents", torch.empty((0, 3), device=device))
        else:
            self.register_buffer("face_tangents", torch.from_numpy(face_tangents).to(device=device))
        self.register_buffer("bary_initial", torch.from_numpy(barycentric).to(device=device))
        tri = self.vertices[self.faces[self.face_ids]]
        self.register_buffer("anchor_local", (tri * self.bary_initial[:, :, None]).sum(dim=1))
        self.register_buffer("root_observation_confidence", torch.zeros((int(face_ids.shape[0]),), device=device))
        self.register_buffer("clean_flow_direction_target", torch.zeros((int(face_ids.shape[0]), 3), device=device))
        self.register_buffer("clean_flow_anchor_confidence", torch.zeros((int(face_ids.shape[0]),), device=device))
        self.register_buffer("clean_flow_length_target", torch.zeros((int(face_ids.shape[0]),), device=device))
        self.register_buffer("clean_flow_length_confidence", torch.zeros((int(face_ids.shape[0]),), device=device))
        self.bary_logits = torch.nn.Parameter(torch.log(self.bary_initial.clamp_min(1.0e-5)))
        self.groom = GroomParameterField(
            int(face_ids.shape[0]),
            ranges=ranges,
            init_length=self.init_groom_length,
            device=device,
        )
        if (
            self.render_geometry_parameterization != "absolute_endpoint"
            and self.geometry_residual_domain == "render"
        ):
            self.render_geometry_residual = RenderGeometryResidualField(
                int(face_ids.shape[0]),
                device=device,
            )
        else:
            self.render_geometry_residual = None
        self.secondary_geometry_residual: RenderGeometryResidualField | None = None
        self.log_scale = torch.nn.Parameter(torch.tensor([math.log(float(init_scale))], device=device))
        self.translation = torch.nn.Parameter(torch.tensor(init_translation, device=device, dtype=torch.float32))
        if bool(local_child_color_support):
            self.child_color_delta_raw = torch.nn.Parameter(torch.zeros((int(face_ids.shape[0]), self.max_child_count, 3), device=device))
        else:
            self.register_parameter("child_color_delta_raw", None)
        self.gaussian_rgb_residual = (
            GaussianRGBResidualField(
                int(face_ids.shape[0]),
                int(gaussian_rgb_residual_control_points),
                float(gaussian_rgb_residual_scale),
                device=device,
            )
            if bool(gaussian_rgb_residual_support)
            else None
        )
        self.guide_view_sh: GuideViewSHField | None = None
        if guide_face_ids is not None and guide_barycentric is not None:
            self.register_buffer("guide_face_ids", torch.from_numpy(guide_face_ids).to(device=device, dtype=torch.long))
            self.register_buffer("guide_barycentric", torch.from_numpy(guide_barycentric).to(device=device))
            guide_tri = self.vertices[self.faces[self.guide_face_ids]]
            self.register_buffer("guide_points_local", (guide_tri * self.guide_barycentric[:, :, None]).sum(dim=1))
            guide_count = int(guide_face_ids.shape[0])
            if guide_region_ids is None:
                guide_region_weight_np = np.zeros((guide_count, 1), dtype=np.float32)
            else:
                guide_region_ids_np = np.asarray(guide_region_ids, dtype=np.int64).reshape(-1)
                if guide_region_ids_np.shape[0] != guide_count:
                    raise RuntimeError(
                        f"guide_region_ids length mismatch: {guide_region_ids_np.shape[0]} != {guide_count}"
                    )
                guide_region_weight_np = (guide_region_ids_np > 0).astype(np.float32).reshape(-1, 1)
            self.register_buffer("guide_region_weight", torch.from_numpy(guide_region_weight_np).to(device=device))
            self.register_buffer("guide_clean_flow_direction_target", torch.zeros((guide_count, 3), device=device))
            self.register_buffer("guide_clean_flow_anchor_confidence", torch.zeros((guide_count,), device=device))
            self.register_buffer("guide_clean_flow_length_target", torch.zeros((guide_count,), device=device))
            self.register_buffer("guide_clean_flow_length_confidence", torch.zeros((guide_count,), device=device))
            self.register_buffer(
                "guide_length_reference",
                torch.full(
                    (guide_count, 1),
                    self.init_groom_length,
                    device=device,
                ),
            )
            self.register_buffer(
                "guide_root_width_reference",
                torch.full(
                    (guide_count, 1),
                    0.00016,
                    device=device,
                ),
            )
            self.register_buffer(
                "guide_tip_width_ratio_reference",
                torch.full((guide_count, 1), 0.070, device=device),
            )
            self.register_buffer(
                "guide_width_taper_reference",
                torch.full((guide_count, 1), 1.80, device=device),
            )
            self.register_buffer(
                "guide_child_radius_reference",
                torch.full((guide_count, 1), 0.0028, device=device),
            )
            self.guide_length_raw = torch.nn.Parameter(torch.zeros((guide_count, 1), device=device))
            self.guide_root_width_raw = torch.nn.Parameter(torch.zeros((guide_count, 1), device=device))
            self.guide_tip_width_ratio_raw = torch.nn.Parameter(torch.zeros((guide_count, 1), device=device))
            self.guide_width_taper_raw = torch.nn.Parameter(torch.zeros((guide_count, 1), device=device))
            self.guide_brush_stiffness_raw = torch.nn.Parameter(
                encode_brush_stiffness(
                    torch.full((guide_count, 1), 0.5, device=device)
                )
            )
            self.guide_curl_radius_ratio_raw = torch.nn.Parameter(torch.zeros((guide_count, 1), device=device))
            self.guide_curl_turns_raw = torch.nn.Parameter(torch.zeros((guide_count, 1), device=device))
            self.guide_child_radius_raw = torch.nn.Parameter(torch.zeros((guide_count, 1), device=device))
            self.guide_clump_strength_raw = torch.nn.Parameter(torch.zeros((guide_count, 1), device=device))
            self.guide_direction_local_raw = torch.nn.Parameter(
                torch.zeros((guide_count, 3), device=device)
            )
            if bool(guide_view_sh_support):
                self.guide_view_sh = GuideViewSHField(
                    guide_count,
                    float(guide_view_sh_scale),
                    device=device,
                )
        else:
            self.register_buffer("guide_face_ids", torch.empty((0,), device=device, dtype=torch.long))
            self.register_buffer("guide_barycentric", torch.empty((0, 3), device=device))
            self.register_buffer("guide_points_local", torch.empty((0, 3), device=device))
            self.register_buffer("guide_region_weight", torch.empty((0, 1), device=device))
            self.register_buffer("guide_clean_flow_direction_target", torch.empty((0, 3), device=device))
            self.register_buffer("guide_clean_flow_anchor_confidence", torch.empty((0,), device=device))
            self.register_buffer("guide_clean_flow_length_target", torch.empty((0,), device=device))
            self.register_buffer("guide_clean_flow_length_confidence", torch.empty((0,), device=device))
            self.register_buffer("guide_length_reference", torch.empty((0, 1), device=device))
            self.register_buffer("guide_root_width_reference", torch.empty((0, 1), device=device))
            self.register_buffer("guide_tip_width_ratio_reference", torch.empty((0, 1), device=device))
            self.register_buffer("guide_width_taper_reference", torch.empty((0, 1), device=device))
            self.register_buffer("guide_child_radius_reference", torch.empty((0, 1), device=device))
            self.register_parameter("guide_length_raw", None)
            self.register_parameter("guide_root_width_raw", None)
            self.register_parameter("guide_tip_width_ratio_raw", None)
            self.register_parameter("guide_width_taper_raw", None)
            self.register_parameter("guide_brush_stiffness_raw", None)
            self.register_parameter("guide_curl_radius_ratio_raw", None)
            self.register_parameter("guide_curl_turns_raw", None)
            self.register_parameter("guide_child_radius_raw", None)
            self.register_parameter("guide_clump_strength_raw", None)
            self.register_parameter("guide_direction_local_raw", None)
        if bool(guide_view_sh_support) and self.guide_view_sh is None:
            raise ValueError("guide-view SH requires primary guide roots")
        self.register_buffer(
            "guide_view_sh_view_indices_cache",
            torch.empty((0,), device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "guide_view_sh_confidence_cache",
            torch.empty((0, 0), device=device),
            persistent=False,
        )
        # R072 per-view ownership. The floor is baked into the cached matrix so
        # the training-loop lookup stays a single index operation.
        self.view_gate_floor = 1.0
        self.register_buffer(
            "view_gate_view_indices_cache",
            torch.empty((0,), device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "view_gate_cache",
            torch.empty((0, 0), device=device),
            persistent=False,
        )
        if self.render_geometry_parameterization != "absolute_endpoint" and not self.guide_enabled():
            raise ValueError("zero-centered geometry requires primary guide roots")
        self.register_buffer(
            "secondary_guide_face_ids",
            torch.empty((0,), device=device, dtype=torch.long),
        )
        self.register_buffer(
            "secondary_guide_barycentric",
            torch.empty((0, 3), device=device),
        )
        self.register_buffer(
            "secondary_guide_points_local",
            torch.empty((0, 3), device=device),
        )
        self.register_buffer(
            "secondary_guide_parent_ids",
            torch.empty((0,), device=device, dtype=torch.long),
        )
        self.register_buffer(
            "secondary_primary_support_ids_cache",
            torch.empty((0, 0), device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "secondary_primary_support_paths_cache",
            torch.empty((0, 0, 3), device=device),
            persistent=False,
        )
        self.register_buffer(
            "secondary_render_support_ids_cache",
            torch.empty((0, 0), device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "secondary_smooth_edges_cache",
            torch.empty((0, 2), device=device, dtype=torch.long),
            persistent=False,
        )
        self._secondary_smooth_graph_k = -1
        self._secondary_support_report: dict[str, object] = {}
        self.register_buffer("guide_interp_ids_cache", torch.empty((0, 0), device=device, dtype=torch.long), persistent=False)
        self.register_buffer("guide_interp_vertex_paths_cache", torch.empty((0, 0, 3), device=device), persistent=False)
        self.register_buffer(
            "guide_smooth_reference_spacing_local",
            torch.zeros((), device=device),
        )
        self.register_buffer(
            "guide_smooth_edges_cache",
            torch.empty((0, 2), device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "guide_smooth_distances_cache",
            torch.empty((0,), device=device),
            persistent=False,
        )
        self.register_buffer(
            "guide_smooth_area_cache",
            torch.empty((0,), device=device),
            persistent=False,
        )
        self._guide_smooth_graph_k = -1
        self._guide_surface_interpolator: SurfaceFieldInterpolator | None = None
        self._guide_support_report: dict[str, float | int] = {}
        self.initialize_default_groom()
        self.rebuild_guide_surface_interpolation()
        if secondary_guide_face_ids is not None:
            if secondary_guide_barycentric is None or secondary_guide_parent_ids is None:
                raise ValueError(
                    "secondary guide face IDs require barycentric coordinates and parent IDs"
                )
            self.attach_secondary_guides(
                secondary_guide_face_ids,
                secondary_guide_barycentric,
                secondary_guide_parent_ids,
            )
        elif self.geometry_residual_domain == "secondary_guide":
            # From-zero setup attaches the parent-conditioned FPS layer after
            # the primary guide interpolator has been constructed.
            self.secondary_geometry_residual = None

    def initialize_default_groom(self) -> None:
        ranges = self.groom.ranges
        with torch.no_grad():
            self.groom.length_reference.fill_(self.init_groom_length)
            self.groom.length_raw.zero_()
            self.groom.root_width_reference.fill_(0.00016)
            self.groom.root_width_raw.zero_()
            set_unit_interval(self.groom.tip_width_ratio_raw, 0.070)
            set_positive_asinh(self.groom.width_taper_raw, 1.80)
            self.groom.brush_stiffness_raw.copy_(
                encode_brush_stiffness(
                    torch.full_like(self.groom.brush_stiffness_raw, 0.5)
                )
            )
            self.groom.direction_local_raw.copy_(
                self.groom.direction_local_raw.new_tensor([0.92 * 0.86, -0.12 * 0.86, 0.018])
            )
            self.groom.curl_radius_ratio_raw.copy_(
                encode_positive_softplus(
                    torch.full_like(self.groom.curl_radius_ratio_raw, EPS)
                )
            )
            set_range(self.groom.curl_turns_raw, 1.20, ranges.curl_turns)
            self.groom.child_radius_reference.fill_(0.0028)
            self.groom.child_radius_raw.zero_()
            set_range(self.groom.clump_strength_raw, 0.25, ranges.clump_strength)
            set_unit_interval(self.groom.opacity_raw, 0.74)
            set_unit_interval(self.groom.tip_opacity_ratio_raw, 0.45)
            root_color = torch.tensor([0.88, 0.88, 0.82], device=self.bary_logits.device).view(1, 3)
            tip_color = torch.tensor([0.98, 0.96, 0.88], device=self.bary_logits.device).view(1, 3)
            set_color(self.groom.root_color_raw, root_color)
            set_color(self.groom.tip_color_raw, tip_color)
            if self.guide_length_raw is not None:
                self.guide_length_reference.fill_(self.init_groom_length)
                self.guide_length_raw.zero_()
                self.guide_root_width_reference.fill_(0.00016)
                self.guide_root_width_raw.zero_()
                self.guide_tip_width_ratio_reference.fill_(0.070)
                self.guide_tip_width_ratio_raw.zero_()
                self.guide_width_taper_reference.fill_(1.80)
                self.guide_width_taper_raw.zero_()
                self.guide_brush_stiffness_raw.copy_(
                    encode_brush_stiffness(
                        torch.full_like(self.guide_brush_stiffness_raw, 0.5)
                    )
                )
                self.guide_curl_turns_raw.zero_()
                self.guide_child_radius_reference.fill_(0.0028)
                self.initialize_guide_shape_ratios_from_current_scale()
                self.guide_child_radius_raw.zero_()
                set_range(self.guide_clump_strength_raw, 0.25, ranges.clump_strength)
                self.guide_direction_local_raw.copy_(
                    self.guide_direction_local_raw.new_tensor(
                        [0.92 * 0.86, -0.12 * 0.86, 0.018]
                    )
                )

    @torch.no_grad()
    def initialize_guide_shape_ratios_from_current_scale(self) -> None:
        """Preserve the accepted near-neutral physical detail at initialization.

        Curl is represented as a fraction of each strand's current length.
        Clean-flow initialization may replace those lengths after the model is
        constructed, so this conversion must happen again after that
        replacement. This is initialization only: training, interpolation,
        smoothing, and lifecycle updates operate directly on the ratio.
        """

        if not self.guide_enabled():
            return
        guide_length = decode_positive_asinh_ratio(
            self.guide_length_raw,
            self.guide_length_reference,
        )
        initial_physical_amplitude = decode_positive_asinh_ratio(
            self.guide_root_width_raw,
            self.guide_root_width_reference,
        )
        initial_shape_ratio = initial_physical_amplitude / guide_length.clamp_min(EPS)
        self.guide_curl_radius_ratio_raw.copy_(
            encode_positive_softplus(initial_shape_ratio)
        )

    def uses_zero_centered_geometry(self) -> bool:
        return self.active_geometry_residual() is not None

    def active_geometry_residual(self) -> RenderGeometryResidualField | None:
        if self.geometry_residual_domain == "secondary_guide":
            return self.secondary_geometry_residual
        return self.render_geometry_residual

    def geometry_residual_parameter_prefix(self) -> str:
        if self.geometry_residual_domain == "secondary_guide":
            return "secondary_geometry_residual"
        return "render_geometry_residual"

    def guide_direction_world(self) -> torch.Tensor | None:
        if not self.guide_enabled():
            return None
        guide_normals, guide_tangents, guide_bitangents = self.guide_normals_and_tangent_frames()
        return local_components_to_world(
            F.normalize(self.guide_direction_local_raw, dim=-1, eps=EPS),
            guide_normals,
            guide_tangents,
            guide_bitangents,
            normalize=True,
        )

    def guide_enabled(self) -> bool:
        return self.guide_length_raw is not None and self.guide_points_local.numel() > 0

    def guide_surface_interpolator(self) -> SurfaceFieldInterpolator:
        if not self.guide_enabled() or self._guide_surface_interpolator is None:
            raise RuntimeError("primary guide surface interpolator is unavailable")
        return self._guide_surface_interpolator

    def secondary_guides_enabled(self) -> bool:
        return (
            self.secondary_geometry_residual is not None
            and self.secondary_guide_points_local.numel() > 0
        )

    @torch.no_grad()
    def attach_secondary_guides(
        self,
        face_ids: np.ndarray | torch.Tensor,
        barycentric: np.ndarray | torch.Tensor,
        parent_ids: np.ndarray | torch.Tensor,
    ) -> dict[str, object]:
        if self.geometry_residual_domain != "secondary_guide":
            raise RuntimeError(
                "secondary guides require geometry_residual_domain=secondary_guide"
            )
        if not self.guide_enabled() or self._guide_surface_interpolator is None:
            raise RuntimeError("secondary guides require initialized primary guides")
        device = self.vertices.device
        face_tensor = torch.as_tensor(face_ids, device=device, dtype=torch.long).reshape(-1)
        bary_tensor = torch.as_tensor(
            barycentric,
            device=device,
            dtype=self.vertices.dtype,
        )
        parent_tensor = torch.as_tensor(
            parent_ids,
            device=device,
            dtype=torch.long,
        ).reshape(-1)
        count = int(face_tensor.shape[0])
        if bary_tensor.shape != (count, 3) or parent_tensor.shape != (count,):
            raise ValueError("secondary guide arrays have inconsistent shapes")
        if count <= 0:
            raise ValueError("secondary guide count must be positive")
        if bool((face_tensor < 0).any()) or bool((face_tensor >= self.faces.shape[0]).any()):
            raise ValueError("secondary guide face IDs are out of range")
        if bool((parent_tensor < 0).any()) or bool((parent_tensor >= self.guide_points_local.shape[0]).any()):
            raise ValueError("secondary guide parent IDs are out of range")
        if not bool((torch.bincount(parent_tensor, minlength=self.guide_points_local.shape[0]) > 0).all()):
            raise ValueError("every primary guide must own at least one secondary guide")

        triangles = self.vertices[self.faces[face_tensor]]
        self.secondary_guide_face_ids = face_tensor.detach()
        self.secondary_guide_barycentric = bary_tensor.detach()
        self.secondary_guide_points_local = (
            triangles * bary_tensor[:, :, None]
        ).sum(dim=1).detach()
        self.secondary_guide_parent_ids = parent_tensor.detach()
        self.secondary_geometry_residual = RenderGeometryResidualField(
            count,
            device=device,
        )
        self.secondary_primary_support_ids_cache = torch.empty(
            (0, 0), device=device, dtype=torch.long
        )
        self.secondary_primary_support_paths_cache = torch.empty(
            (0, 0, 3), device=device
        )
        self.secondary_render_support_ids_cache = torch.empty(
            (0, 0), device=device, dtype=torch.long
        )
        self.secondary_smooth_edges_cache = torch.empty(
            (0, 2), device=device, dtype=torch.long
        )
        self._secondary_smooth_graph_k = -1
        primary_support = self._guide_surface_interpolator.build_support(
            self.secondary_guide_points_local,
            self.secondary_guide_face_ids,
        )
        self.secondary_primary_support_ids_cache = primary_support.indices.detach()
        self.secondary_primary_support_paths_cache = (
            primary_support.vertex_path_distances.detach()
        )
        self._secondary_support_report = {
            "primary": dict(primary_support.report),
        }
        render_report = self.rebuild_secondary_render_support()
        return {
            "secondary_root_count": count,
            "primary_support": dict(primary_support.report),
            "render_support": render_report,
        }

    @torch.no_grad()
    def secondary_primary_support(self) -> SurfaceSupport:
        if not self.secondary_guides_enabled():
            raise RuntimeError("secondary guide support requested while disabled")
        expected = (
            int(self.secondary_guide_points_local.shape[0]),
            min(int(self.guide_interpolation_k), int(self.guide_points_local.shape[0])),
        )
        if tuple(self.secondary_primary_support_ids_cache.shape) != expected:
            if self._guide_surface_interpolator is None:
                raise RuntimeError("primary guide interpolator is unavailable")
            support = self._guide_surface_interpolator.build_support(
                self.secondary_guide_points_local,
                self.secondary_guide_face_ids,
            )
            self.secondary_primary_support_ids_cache = support.indices.detach()
            self.secondary_primary_support_paths_cache = (
                support.vertex_path_distances.detach()
            )
            self._secondary_support_report["primary"] = dict(support.report)
        return SurfaceSupport(
            indices=self.secondary_primary_support_ids_cache,
            vertex_path_distances=self.secondary_primary_support_paths_cache,
            report=dict(self._secondary_support_report.get("primary", {})),
        )

    @torch.no_grad()
    def rebuild_secondary_render_support(self) -> dict[str, object]:
        if not self.secondary_guides_enabled():
            self.secondary_render_support_ids_cache = torch.empty(
                (0, 0), device=self.vertices.device, dtype=torch.long
            )
            return {}
        support = build_parent_conditioned_query_support(
            self.anchor_local,
            self.guide_interpolation_support(),
            self.secondary_guide_points_local,
            self.secondary_guide_parent_ids,
            neighbor_count=self.secondary_guide_interpolation_k,
        )
        self.secondary_render_support_ids_cache = support.indices.detach()
        self._secondary_support_report["render"] = dict(support.report)
        return dict(support.report)

    @torch.no_grad()
    def secondary_render_support(self) -> LocalSurfaceSupport:
        if not self.secondary_guides_enabled():
            raise RuntimeError("secondary render support requested while disabled")
        expected = (
            int(self.anchor_local.shape[0]),
            min(
                int(self.secondary_guide_interpolation_k),
                int(self.secondary_guide_points_local.shape[0]),
            ),
        )
        if tuple(self.secondary_render_support_ids_cache.shape) != expected:
            self.rebuild_secondary_render_support()
        return LocalSurfaceSupport(
            indices=self.secondary_render_support_ids_cache,
            report=dict(self._secondary_support_report.get("render", {})),
        )

    @torch.no_grad()
    def secondary_surface_smoothing_edges(self, neighbor_count: int) -> torch.Tensor:
        if not self.secondary_guides_enabled():
            return torch.empty(
                (0, 2), device=self.vertices.device, dtype=torch.long
            )
        expected_edges = int(self.secondary_guide_points_local.shape[0]) * min(
            max(int(neighbor_count), 0),
            max(int(self.secondary_guide_points_local.shape[0]) - 1, 0),
        )
        if (
            self._secondary_smooth_graph_k != int(neighbor_count)
            or tuple(self.secondary_smooth_edges_cache.shape) != (expected_edges, 2)
        ):
            self.secondary_smooth_edges_cache = build_hierarchical_surface_edges(
                self.secondary_guide_points_local,
                self.secondary_primary_support().indices,
                neighbor_count=int(neighbor_count),
            ).detach()
            self._secondary_smooth_graph_k = int(neighbor_count)
        return self.secondary_smooth_edges_cache

    def invalidate_guide_interpolation_cache(self) -> None:
        self.invalidate_guide_surface_support_cache()
        device = self.vertices.device
        self.guide_smooth_edges_cache = torch.empty((0, 2), device=device, dtype=torch.long)
        self.guide_smooth_distances_cache = torch.empty((0,), device=device)
        self.guide_smooth_area_cache = torch.empty((0,), device=device)
        self._guide_smooth_graph_k = -1

    @torch.no_grad()
    def invalidate_guide_surface_support_cache(self) -> None:
        device = self.vertices.device
        self.guide_interp_ids_cache = torch.empty((0, 0), device=device, dtype=torch.long)
        self.guide_interp_vertex_paths_cache = torch.empty((0, 0, 3), device=device)
        self.secondary_render_support_ids_cache = torch.empty(
            (0, 0), device=device, dtype=torch.long
        )
        self._guide_support_report = {}

    @torch.no_grad()
    def rebuild_guide_surface_interpolation(self) -> dict[str, float | int]:
        self.invalidate_guide_interpolation_cache()
        if not self.guide_enabled():
            self._guide_surface_interpolator = None
            return {}
        self._guide_surface_interpolator = SurfaceFieldInterpolator(
            vertices=self.vertices,
            faces=self.faces,
            source_points=self.guide_points_local,
            source_face_ids=self.guide_face_ids,
            neighbor_count=self.guide_interpolation_k,
            device=self.vertices.device,
        )
        return self.rebuild_guide_surface_support()

    @torch.no_grad()
    def rebuild_guide_surface_support(self) -> dict[str, float | int]:
        """Recompute exact render-to-guide support without rebuilding guide topology."""

        self.invalidate_guide_surface_support_cache()
        if not self.guide_enabled():
            return {}
        if self._guide_surface_interpolator is None:
            raise RuntimeError("guide surface interpolator is unavailable")
        support = self._guide_surface_interpolator.build_support(self.anchor_local, self.face_ids)
        self.guide_interp_ids_cache = support.indices.detach()
        self.guide_interp_vertex_paths_cache = support.vertex_path_distances.detach()
        self._guide_support_report = dict(support.report)
        report: dict[str, object] = dict(self._guide_support_report)
        if self.secondary_guides_enabled():
            report["secondary_render_support"] = self.rebuild_secondary_render_support()
        return report

    @torch.no_grad()
    def guide_interpolation_support(self) -> SurfaceSupport:
        if not self.guide_enabled():
            raise RuntimeError("guide interpolation support requested but guide roots are disabled")
        expected = (
            int(self.anchor_local.shape[0]),
            min(int(self.guide_interpolation_k), int(self.guide_points_local.shape[0])),
        )
        if tuple(self.guide_interp_ids_cache.shape) != expected:
            if self._guide_surface_interpolator is None:
                self.rebuild_guide_surface_interpolation()
            else:
                self.rebuild_guide_surface_support()
        return SurfaceSupport(
            indices=self.guide_interp_ids_cache,
            vertex_path_distances=self.guide_interp_vertex_paths_cache,
            report={
                **self._guide_support_report,
                "query_count": int(self.guide_interp_ids_cache.shape[0]),
                "neighbor_count": int(self.guide_interp_ids_cache.shape[1]),
            },
        )

    @torch.no_grad()
    def guide_interpolation_attribution(
        self,
        roots_local: torch.Tensor,
    ) -> tuple[SurfaceSupport, torch.Tensor]:
        """Return the exact guide support and weights used by render roots."""

        support = self.guide_interpolation_support()
        if self._guide_surface_interpolator is None:
            raise RuntimeError("guide surface interpolator is unavailable")
        weights = self._guide_surface_interpolator.weights(
            roots_local.detach(),
            self.face_ids.detach(),
            support,
        )
        return support, weights.detach()

    @torch.no_grad()
    def guide_surface_smoothing_edges(self, neighbor_count: int) -> torch.Tensor:
        """Use the interpolation source graph for guide-root smoothing."""

        return self.guide_surface_smoothing_graph(neighbor_count).edges

    @torch.no_grad()
    def guide_surface_smoothing_graph(self, neighbor_count: int) -> SurfaceSourceGraph:
        """Return cached intrinsic metric data for the current guide topology."""

        if not self.guide_enabled():
            empty_edges = torch.empty(
                (0, 2), device=self.vertices.device, dtype=torch.long
            )
            empty_values = torch.empty((0,), device=self.vertices.device)
            return SurfaceSourceGraph(
                edges=empty_edges,
                distances=empty_values,
                source_area_weights=empty_values,
                reference_spacing=self.guide_smooth_reference_spacing_local,
            )
        if self._guide_surface_interpolator is None:
            self.rebuild_guide_surface_interpolation()
        if self._guide_surface_interpolator is None:
            raise RuntimeError("guide surface interpolator is unavailable")
        expected_edges = int(self.guide_points_local.shape[0]) * min(
            max(int(neighbor_count), 0),
            max(int(self.guide_points_local.shape[0]) - 1, 0),
        )
        cache_valid = (
            self._guide_smooth_graph_k == int(neighbor_count)
            and tuple(self.guide_smooth_edges_cache.shape) == (expected_edges, 2)
            and tuple(self.guide_smooth_distances_cache.shape) == (expected_edges,)
            and tuple(self.guide_smooth_area_cache.shape)
            == (int(self.guide_points_local.shape[0]),)
        )
        if not cache_valid:
            graph = self._guide_surface_interpolator.source_neighbor_graph(neighbor_count)
            self.guide_smooth_edges_cache = graph.edges.detach()
            self.guide_smooth_distances_cache = graph.distances.detach()
            self.guide_smooth_area_cache = graph.source_area_weights.detach()
            self._guide_smooth_graph_k = int(neighbor_count)
            if float(self.guide_smooth_reference_spacing_local.detach().cpu()) <= 0.0:
                self.guide_smooth_reference_spacing_local.copy_(graph.reference_spacing)
        return SurfaceSourceGraph(
            edges=self.guide_smooth_edges_cache,
            distances=self.guide_smooth_distances_cache,
            source_area_weights=self.guide_smooth_area_cache,
            reference_spacing=self.guide_smooth_reference_spacing_local,
        )

    def guide_lifecycle_state(self) -> RootLifecycleState:
        if not self.guide_enabled():
            raise RuntimeError("guide lifecycle requested but guide roots are disabled")
        return RootLifecycleState(
            points=self.guide_points_local.detach(),
            face_ids=self.guide_face_ids.detach().clone(),
            barycentric=self.guide_barycentric.detach().clone(),
        )

    def apply_guide_structure_update(self, update: RootStructureUpdate) -> dict[str, int]:
        if not self.guide_enabled():
            raise RuntimeError("guide structure update requested but guide roots are disabled")
        old_state = self.guide_lifecycle_state()
        old_count = int(old_state.points.shape[0])
        if update.new_barycentric.numel() == 0 and not bool(update.prune_mask.any()):
            return {"old_guide_root_count": old_count, "guide_root_count_after": old_count}

        device = self.vertices.device
        child_count = int(update.new_barycentric.shape[0])
        child_points = (
            (
                self.vertices[self.faces[update.new_face_ids]]
                * update.new_barycentric[:, :, None]
            ).sum(dim=1)
            if child_count
            else old_state.points.new_empty((0, 3))
        )
        if child_count:
            child_support = build_local_surface_support(
                faces=self.faces,
                source_points=old_state.points,
                source_face_ids=old_state.face_ids,
                query_points=child_points,
                query_face_ids=update.new_face_ids,
                neighbor_count=self.guide_interpolation_k,
            )
            child_weights = local_surface_weights(child_points, old_state.points, child_support)
            child_ids = child_support.indices
        else:
            child_ids = old_state.face_ids.new_empty((0, 0))
            child_weights = old_state.points.new_empty((0, 0))

        ranges = self.groom.ranges
        old_length_reference = self.guide_length_reference.detach()
        old_root_width_reference = self.guide_root_width_reference.detach()
        old_tip_width_ratio_reference = self.guide_tip_width_ratio_reference.detach()
        old_width_taper_reference = self.guide_width_taper_reference.detach()
        old_child_radius_reference = self.guide_child_radius_reference.detach()
        child_length_reference = (
            interpolate_physical(
                old_length_reference,
                child_ids,
                child_weights,
            )
            if child_count
            else old_length_reference.new_empty((0, 1))
        )
        child_root_width_reference = (
            interpolate_physical(
                old_root_width_reference,
                child_ids,
                child_weights,
            )
            if child_count
            else old_root_width_reference.new_empty((0, 1))
        )
        child_tip_width_ratio_reference = (
            interpolate_physical(
                old_tip_width_ratio_reference,
                child_ids,
                child_weights,
            )
            if child_count
            else old_tip_width_ratio_reference.new_empty((0, 1))
        )
        child_width_taper_reference = (
            interpolate_physical(
                old_width_taper_reference,
                child_ids,
                child_weights,
            )
            if child_count
            else old_width_taper_reference.new_empty((0, 1))
        )
        child_child_radius_reference = (
            interpolate_physical(
                old_child_radius_reference,
                child_ids,
                child_weights,
            )
            if child_count
            else old_child_radius_reference.new_empty((0, 1))
        )
        physical_sources = {
            "guide_length_raw": decode_positive_asinh_ratio(
                self.guide_length_raw.detach(),
                old_length_reference,
            ),
            "guide_root_width_raw": decode_positive_asinh_ratio(
                self.guide_root_width_raw.detach(),
                old_root_width_reference,
            ),
            "guide_tip_width_ratio_raw": apply_asinh_logit_residual(
                old_tip_width_ratio_reference,
                self.guide_tip_width_ratio_raw.detach(),
                1.0,
            ),
            "guide_width_taper_raw": decode_positive_asinh_ratio(
                self.guide_width_taper_raw.detach(),
                old_width_taper_reference,
            ),
            "guide_brush_stiffness_raw": decode_brush_stiffness(
                self.guide_brush_stiffness_raw.detach()
            ),
            "guide_curl_radius_ratio_raw": decode_positive_softplus(
                self.guide_curl_radius_ratio_raw.detach()
            ),
            "guide_curl_turns_raw": self.guide_curl_turns_raw.detach(),
            "guide_child_radius_raw": decode_positive_asinh_ratio(
                self.guide_child_radius_raw.detach(),
                old_child_radius_reference,
            ),
            "guide_clump_strength_raw": GroomParameterField._decode_range(self.guide_clump_strength_raw.detach(), ranges.clump_strength),
        }
        raw_bounds = {
            "guide_clump_strength_raw": ranges.clump_strength,
        }
        old_raw = {
            "guide_length_raw": self.guide_length_raw.detach(),
            "guide_root_width_raw": self.guide_root_width_raw.detach(),
            "guide_tip_width_ratio_raw": self.guide_tip_width_ratio_raw.detach(),
            "guide_width_taper_raw": self.guide_width_taper_raw.detach(),
            "guide_brush_stiffness_raw": self.guide_brush_stiffness_raw.detach(),
            "guide_curl_radius_ratio_raw": self.guide_curl_radius_ratio_raw.detach(),
            "guide_curl_turns_raw": self.guide_curl_turns_raw.detach(),
            "guide_child_radius_raw": self.guide_child_radius_raw.detach(),
            "guide_clump_strength_raw": self.guide_clump_strength_raw.detach(),
        }
        new_params: dict[str, torch.Tensor] = {}
        for name, source in physical_sources.items():
            child_physical = (
                interpolate_physical(source, child_ids, child_weights)
                if child_count
                else source.new_empty((0, *source.shape[1:]))
            )
            if name == "guide_brush_stiffness_raw":
                child_raw = encode_brush_stiffness(child_physical)
            elif name == "guide_length_raw":
                child_raw = encode_positive_asinh_ratio(
                    child_physical,
                    child_length_reference,
                )
            elif name == "guide_root_width_raw":
                child_raw = encode_positive_asinh_ratio(
                    child_physical,
                    child_root_width_reference,
                )
            elif name == "guide_tip_width_ratio_raw":
                child_raw = encode_asinh_logit_residual(
                    child_physical,
                    child_tip_width_ratio_reference,
                )
            elif name == "guide_width_taper_raw":
                child_raw = encode_positive_asinh_ratio(
                    child_physical,
                    child_width_taper_reference,
                )
            elif name == "guide_child_radius_raw":
                child_raw = encode_positive_asinh_ratio(
                    child_physical,
                    child_child_radius_reference,
                )
            elif name in {
                "guide_curl_radius_ratio_raw",
            }:
                child_raw = encode_positive_softplus(child_physical)
            elif name == "guide_curl_turns_raw":
                child_raw = child_physical
            else:
                child_raw = raw_from_range(child_physical, raw_bounds[name])
            new_params[name] = apply_attribute_update(old_raw[name], update, child_raw)
        new_params["guide_length_reference"] = apply_attribute_update(
            old_length_reference,
            update,
            child_length_reference,
        )
        new_params["guide_root_width_reference"] = apply_attribute_update(
            old_root_width_reference,
            update,
            child_root_width_reference,
        )
        new_params["guide_tip_width_ratio_reference"] = apply_attribute_update(
            old_tip_width_ratio_reference,
            update,
            child_tip_width_ratio_reference,
        )
        new_params["guide_width_taper_reference"] = apply_attribute_update(
            old_width_taper_reference,
            update,
            child_width_taper_reference,
        )
        new_params["guide_child_radius_reference"] = apply_attribute_update(
            old_child_radius_reference,
            update,
            child_child_radius_reference,
        )

        guide_normals, guide_tangents, guide_bitangents = self.guide_normals_and_tangent_frames()
        child_normals, child_tangents, child_bitangents = self.tangent_frames_for_face_ids(update.new_face_ids)
        source_direction = self.guide_direction_world()
        if source_direction is None:
            raise RuntimeError("guide direction state is unavailable during densification")
        source_direction = source_direction.detach()
        child_direction = (
            interpolate_directions(
                source_direction,
                guide_normals,
                child_normals,
                child_ids,
                child_weights,
            )
            if child_count
            else source_direction.new_empty((0, 3))
        )
        child_local = direction_to_local_components(
            child_direction,
            child_normals,
            child_tangents,
            child_bitangents,
        )
        new_params["guide_direction_local_raw"] = apply_attribute_update(
            self.guide_direction_local_raw.detach(),
            update,
            child_local,
        )

        evidence_sources = {
            "guide_region_weight": self.guide_region_weight.detach(),
            "guide_clean_flow_anchor_confidence": self.guide_clean_flow_anchor_confidence.detach().reshape(-1, 1),
        }
        for name, source in evidence_sources.items():
            child = (
                interpolate_physical(source, child_ids, child_weights)
                if child_count
                else source.new_empty((0, *source.shape[1:]))
            )
            new_params[name] = apply_attribute_update(source, update, child)
        length_target = self.guide_clean_flow_length_target.detach().reshape(-1, 1)
        length_confidence = self.guide_clean_flow_length_confidence.detach().reshape(-1, 1)
        if child_count:
            child_length_confidence = interpolate_physical(
                length_confidence,
                child_ids,
                child_weights,
            )
            child_weighted_length = interpolate_physical(
                length_target * length_confidence,
                child_ids,
                child_weights,
            )
            child_length_target = torch.where(
                child_length_confidence > EPS,
                child_weighted_length / child_length_confidence.clamp_min(EPS),
                torch.zeros_like(child_weighted_length),
            )
        else:
            child_length_target = length_target.new_empty((0, 1))
            child_length_confidence = length_confidence.new_empty((0, 1))
        new_params["guide_clean_flow_length_target"] = apply_attribute_update(
            length_target,
            update,
            child_length_target,
        )
        new_params["guide_clean_flow_length_confidence"] = apply_attribute_update(
            length_confidence,
            update,
            child_length_confidence,
        )
        child_clean_direction = (
            interpolate_directions(
                self.guide_clean_flow_direction_target.detach(),
                guide_normals,
                child_normals,
                child_ids,
                child_weights,
            )
            if child_count
            else self.guide_clean_flow_direction_target.new_empty((0, 3))
        )
        new_params["guide_clean_flow_direction_target"] = apply_attribute_update(
            self.guide_clean_flow_direction_target.detach(),
            update,
            child_clean_direction,
        )

        new_state = apply_structure_update(old_state, update, self.vertices, self.faces)
        new_count = int(new_state.points.shape[0])
        self.guide_face_ids = new_state.face_ids.detach().long()
        self.guide_barycentric = new_state.barycentric.detach()
        self.guide_points_local = new_state.points.detach()
        self.guide_length_reference = new_params["guide_length_reference"].to(device=device).detach()
        self.guide_root_width_reference = new_params["guide_root_width_reference"].to(device=device).detach()
        self.guide_tip_width_ratio_reference = new_params["guide_tip_width_ratio_reference"].to(device=device).detach()
        self.guide_width_taper_reference = new_params["guide_width_taper_reference"].to(device=device).detach()
        self.guide_child_radius_reference = new_params["guide_child_radius_reference"].to(device=device).detach()
        self.guide_length_raw = torch.nn.Parameter(new_params["guide_length_raw"].to(device=device))
        self.guide_root_width_raw = torch.nn.Parameter(new_params["guide_root_width_raw"].to(device=device))
        self.guide_tip_width_ratio_raw = torch.nn.Parameter(new_params["guide_tip_width_ratio_raw"].to(device=device))
        self.guide_width_taper_raw = torch.nn.Parameter(new_params["guide_width_taper_raw"].to(device=device))
        self.guide_brush_stiffness_raw = torch.nn.Parameter(
            new_params["guide_brush_stiffness_raw"].to(device=device)
        )
        self.guide_curl_radius_ratio_raw = torch.nn.Parameter(new_params["guide_curl_radius_ratio_raw"].to(device=device))
        self.guide_curl_turns_raw = torch.nn.Parameter(new_params["guide_curl_turns_raw"].to(device=device))
        self.guide_child_radius_raw = torch.nn.Parameter(new_params["guide_child_radius_raw"].to(device=device))
        self.guide_clump_strength_raw = torch.nn.Parameter(new_params["guide_clump_strength_raw"].to(device=device))
        self.guide_region_weight = new_params["guide_region_weight"].to(device=device).detach().clamp(0.0, 1.0)
        self.guide_clean_flow_direction_target = F.normalize(
            new_params["guide_clean_flow_direction_target"].to(device=device), dim=-1, eps=1.0e-8
        ).detach()
        self.guide_clean_flow_anchor_confidence = new_params["guide_clean_flow_anchor_confidence"].to(device=device).reshape(-1).detach().clamp(0.0, 1.0)
        self.guide_clean_flow_length_target = new_params["guide_clean_flow_length_target"].to(device=device).reshape(-1).detach().clamp_min(0.0)
        self.guide_clean_flow_length_confidence = new_params["guide_clean_flow_length_confidence"].to(device=device).reshape(-1).detach().clamp(0.0, 1.0)
        self.guide_direction_local_raw = torch.nn.Parameter(
            new_params["guide_direction_local_raw"].to(device=device)
        )
        self.rebuild_guide_surface_interpolation()
        return {"old_guide_root_count": old_count, "guide_root_count_after": new_count}

    def sample_guide_controls(
        self,
        roots_local: torch.Tensor,
        root_face_ids: torch.Tensor,
        root_normals: torch.Tensor,
        root_tangents: torch.Tensor,
        root_bitangents: torch.Tensor,
        support: SurfaceSupport | None = None,
        length_gradient_confidence: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        if length_gradient_confidence is not None:
            if not isinstance(length_gradient_confidence, torch.Tensor):
                raise TypeError("length gradient confidence must be a tensor")
            guide_count = int(self.guide_points_local.shape[0])
            if tuple(length_gradient_confidence.shape) not in {
                (guide_count,),
                (guide_count, 1),
            }:
                raise ValueError(
                    "length gradient confidence must have shape [G] or [G, 1]"
                )
            length_gradient_confidence = length_gradient_confidence.to(
                device=self.vertices.device,
                dtype=self.vertices.dtype,
            ).reshape(-1, 1)
            if not bool(torch.isfinite(length_gradient_confidence).all()):
                raise ValueError("length gradient confidence must be finite")
            if bool(
                (
                    (length_gradient_confidence < 0.0)
                    | (length_gradient_confidence > 1.0)
                ).any()
            ):
                raise ValueError("length gradient confidence must lie in [0, 1]")
        if not self.guide_enabled():
            return {}, None
        if self._guide_surface_interpolator is None:
            self.rebuild_guide_surface_interpolation()
        if self._guide_surface_interpolator is None:
            raise RuntimeError("guide surface interpolator is unavailable")
        if support is None:
            support = self._guide_surface_interpolator.build_support(
                roots_local.detach(),
                root_face_ids.detach(),
            )
        weights = self._guide_surface_interpolator.weights(
            roots_local,
            root_face_ids,
            support,
        )
        ids = support.indices
        ranges = self.groom.ranges
        guide_values = {
            "length": decode_positive_asinh_ratio(
                self.guide_length_raw,
                self.guide_length_reference,
            ),
            "root_width": decode_positive_asinh_ratio(
                self.guide_root_width_raw,
                self.guide_root_width_reference,
            ),
            "tip_width_ratio": apply_asinh_logit_residual(
                self.guide_tip_width_ratio_reference,
                self.guide_tip_width_ratio_raw,
                1.0,
            ),
            "width_taper": decode_positive_asinh_ratio(
                self.guide_width_taper_raw,
                self.guide_width_taper_reference,
            ),
            "brush_stiffness": decode_brush_stiffness(
                self.guide_brush_stiffness_raw
            ),
            "curl_radius_ratio": decode_positive_softplus(
                self.guide_curl_radius_ratio_raw
            ),
            "curl_turns": self.guide_curl_turns_raw,
            "child_radius": decode_positive_asinh_ratio(
                self.guide_child_radius_raw,
                self.guide_child_radius_reference,
            ),
            "clump_strength": GroomParameterField._decode_range(self.guide_clump_strength_raw, ranges.clump_strength),
        }
        if length_gradient_confidence is not None:
            guide_values["length"] = straight_through_gate(
                guide_values["length"],
                length_gradient_confidence,
            )
        guide_direction = self.guide_direction_world()
        if guide_direction is not None:
            guide_normals, _, _ = self.guide_normals_and_tangent_frames()
        guide_interp = {
            name: interpolate_physical(guide_value, ids, weights)
            for name, guide_value in guide_values.items()
        }
        if guide_direction is not None:
            guide_direction_out = interpolate_directions(
                guide_direction,
                guide_normals,
                root_normals,
                ids,
                weights,
            )
        else:
            guide_direction_out = None
        return guide_interp, guide_direction_out

    def interpolate_guide_controls(
        self,
        roots_local: torch.Tensor,
        root_normals: torch.Tensor,
        root_tangents: torch.Tensor,
        root_bitangents: torch.Tensor,
        length_gradient_confidence: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        if int(roots_local.shape[0]) != int(self.anchor_local.shape[0]):
            raise RuntimeError("guide interpolation root count does not match render root support count")
        return self.sample_guide_controls(
            roots_local,
            self.face_ids,
            root_normals,
            root_tangents,
            root_bitangents,
            support=self.guide_interpolation_support(),
            length_gradient_confidence=length_gradient_confidence,
        )

    @torch.no_grad()
    def set_view_gate(
        self,
        view_indices: torch.Tensor,
        gate: torch.Tensor,
        floor: float,
    ) -> None:
        """Install the R072 per-view ownership gate with its floor baked in."""

        view_indices = view_indices.to(
            device=self.vertices.device,
            dtype=torch.long,
        ).reshape(-1)
        gate = gate.to(device=self.vertices.device, dtype=self.vertices.dtype)
        expected = (int(view_indices.shape[0]), int(self.guide_points_local.shape[0]))
        if tuple(gate.shape) != expected:
            raise ValueError(
                f"view gate must have shape [V, G]: {tuple(gate.shape)} != {expected}"
            )
        if torch.unique(view_indices).numel() != view_indices.numel():
            raise ValueError("view gate view indices must be unique")
        if not bool(torch.isfinite(gate).all()):
            raise ValueError("view gate must be finite")
        if bool((gate < 0.0).any()):
            raise ValueError("view gate must be non-negative")
        floor = float(floor)
        if not (0.0 <= floor <= 1.0):
            raise ValueError(f"view gate floor must lie in [0, 1], got {floor}")
        self.view_gate_view_indices_cache = view_indices.detach()
        self.view_gate_cache = gate.detach().clamp_min(0.0)
        self.view_gate_floor = floor

    def view_gate_enabled(self) -> bool:
        return int(self.view_gate_cache.numel()) > 0

    def view_gate_for_guides(self, view_index: int) -> torch.Tensor:
        """Return the ``[G]`` gradient share this view owns per primary guide."""

        if not self.view_gate_enabled():
            raise RuntimeError("view gate has not been initialized")
        guide_count = int(self.guide_points_local.shape[0])
        matches = torch.nonzero(
            self.view_gate_view_indices_cache == int(view_index),
            as_tuple=False,
        ).reshape(-1)
        if matches.numel() == 0:
            # A training view outside the trusted V7 set still renders forward
            # but claims only the configured floor of geometry ownership.
            return self.view_gate_cache.new_full((guide_count,), self.view_gate_floor)
        if matches.numel() != 1:
            raise RuntimeError(f"duplicate view gate index: {int(view_index)}")
        return self.view_gate_cache[int(matches[0])]

    def view_gate_at_render_roots(
        self,
        roots_local: torch.Tensor,
        view_index: int,
    ) -> torch.Tensor:
        """Interpolate the per-view guide gate onto render roots as ``[R, 1]``.

        This reuses the accepted K8 primary-guide surface support, so the gate
        follows the same ownership contract as every other guide-owned field.
        """

        guide_gate = self.view_gate_for_guides(view_index).reshape(-1, 1)
        support = self.guide_interpolation_support()
        weights = self.guide_surface_interpolator().weights(
            roots_local.detach(),
            self.face_ids,
            support,
        )
        gate = interpolate_physical(guide_gate, support.indices, weights)
        return gate.reshape(-1, 1).clamp_min(0.0)

    def set_guide_view_sh_confidence(
        self,
        view_indices: torch.Tensor,
        confidence: torch.Tensor,
    ) -> None:
        if self.guide_view_sh is None:
            raise RuntimeError("guide-view SH confidence supplied while support is disabled")
        view_indices = view_indices.to(
            device=self.vertices.device,
            dtype=torch.long,
        ).reshape(-1)
        confidence = confidence.to(
            device=self.vertices.device,
            dtype=self.vertices.dtype,
        )
        expected = (int(view_indices.shape[0]), int(self.guide_points_local.shape[0]))
        if tuple(confidence.shape) != expected:
            raise ValueError(
                "guide-view SH confidence must have shape [V, G]: "
                f"{tuple(confidence.shape)} != {expected}"
            )
        if torch.unique(view_indices).numel() != view_indices.numel():
            raise ValueError("guide-view SH view indices must be unique")
        if not bool(torch.isfinite(confidence).all()):
            raise ValueError("guide-view SH confidence must be finite")
        self.guide_view_sh_view_indices_cache = view_indices.detach()
        self.guide_view_sh_confidence_cache = confidence.detach().clamp(0.0, 1.0)

    def guide_view_sh_confidence_for_view(self, view_index: int) -> torch.Tensor:
        if self.guide_view_sh is None:
            raise RuntimeError("guide-view SH support is disabled")
        if self.guide_view_sh_confidence_cache.numel() == 0:
            raise RuntimeError("guide-view SH confidence has not been initialized")
        matches = torch.nonzero(
            self.guide_view_sh_view_indices_cache == int(view_index),
            as_tuple=False,
        ).reshape(-1)
        if matches.numel() == 0:
            return self.guide_view_sh_confidence_cache.new_zeros(
                (int(self.guide_points_local.shape[0]),)
            )
        if matches.numel() != 1:
            raise RuntimeError(f"duplicate guide-view SH view index: {int(view_index)}")
        return self.guide_view_sh_confidence_cache[int(matches[0])]

    def guide_view_sh_residual_at_render_roots(
        self,
        roots_local: torch.Tensor,
        viewmat: torch.Tensor,
        *,
        view_index: int | None,
    ) -> torch.Tensor:
        """Evaluate guide SH in a detached hair-local frame and interpolate it."""

        if self.guide_view_sh is None:
            return roots_local.new_zeros((int(roots_local.shape[0]), 3))
        if int(roots_local.shape[0]) != int(self.face_ids.shape[0]):
            raise RuntimeError("guide-view SH render-root count mismatch")

        with torch.no_grad():
            detached_viewmat = viewmat.detach().to(
                device=self.vertices.device,
                dtype=self.vertices.dtype,
            )
            rotation = detached_viewmat[:3, :3]
            translation = detached_viewmat[:3, 3]
            camera_center = -(rotation.transpose(0, 1) @ translation)
            guide_points_world = (
                self.guide_points_local.detach()
                * torch.exp(self.log_scale.detach()).reshape(1, 1)
                + self.translation.detach().reshape(1, 3)
            )
            view_direction = F.normalize(
                camera_center.reshape(1, 3) - guide_points_world,
                dim=-1,
                eps=EPS,
            )
            guide_normals, guide_tangents, guide_bitangents = (
                self.guide_normals_and_tangent_frames()
            )
            guide_normals = guide_normals.detach()
            guide_tangents = guide_tangents.detach()
            guide_bitangents = guide_bitangents.detach()
            hair_axis = self.guide_direction_world()
            if hair_axis is None:
                raise RuntimeError("guide-view SH requires guide directions")
            hair_axis = hair_axis.detach()
            side_raw = torch.cross(guide_normals, hair_axis, dim=-1)
            fallback_side = torch.cross(guide_tangents, hair_axis, dim=-1)
            use_fallback = torch.linalg.vector_norm(side_raw, dim=-1, keepdim=True) <= EPS
            side_raw = torch.where(use_fallback, fallback_side, side_raw)
            second_fallback = torch.cross(guide_bitangents, hair_axis, dim=-1)
            use_second = torch.linalg.vector_norm(side_raw, dim=-1, keepdim=True) <= EPS
            side_raw = torch.where(use_second, second_fallback, side_raw)
            if bool((torch.linalg.vector_norm(side_raw, dim=-1) <= EPS).any()):
                raise RuntimeError("guide-view SH could not construct a hair-local frame")
            side_axis = F.normalize(side_raw, dim=-1, eps=EPS)
            up_axis = F.normalize(
                torch.cross(hair_axis, side_axis, dim=-1),
                dim=-1,
                eps=EPS,
            )
            local_view_direction = torch.stack(
                (
                    (view_direction * hair_axis).sum(dim=-1),
                    (view_direction * side_axis).sum(dim=-1),
                    (view_direction * up_axis).sum(dim=-1),
                ),
                dim=-1,
            )

        gradient_confidence = None
        if torch.is_grad_enabled():
            if view_index is None:
                raise RuntimeError(
                    "gradient-enabled guide-view SH rendering requires view_index"
                )
            gradient_confidence = self.guide_view_sh_confidence_for_view(view_index)
        guide_residual = self.guide_view_sh.residual(
            local_view_direction,
            gradient_confidence=gradient_confidence,
        )
        support = self.guide_interpolation_support()
        weights = self.guide_surface_interpolator().weights(
            roots_local.detach(),
            self.face_ids,
            support,
        )
        return interpolate_physical(guide_residual, support.indices, weights)

    def geometry_residual_at_render_roots(
        self,
        roots_local: torch.Tensor,
        normals: torch.Tensor,
        tangents: torch.Tensor,
        bitangents: torch.Tensor,
    ) -> InterpolatedGeometryResiduals | None:
        """Return the active residual coordinates in render-root frames.

        Primary guide controls are never routed through the secondary layer.
        In secondary-guide mode only the zero-centered residual field is
        interpolated, so an all-zero secondary field reproduces the direct
        primary-guide result exactly.
        """

        field = self.active_geometry_residual()
        if field is None:
            return None
        if self.geometry_residual_domain == "render":
            if int(field.root_count) != int(roots_local.shape[0]):
                raise RuntimeError(
                    "render geometry residual count does not match render roots"
                )
            return InterpolatedGeometryResiduals(
                raw=dict(field.named_parameters()),
                decoded=field.decode(),
            )
        if not self.secondary_guides_enabled():
            raise RuntimeError(
                "secondary geometry residual domain has no attached secondary guides"
            )
        source_normals, source_tangents, source_bitangents = (
            self.tangent_frames_for_face_ids(self.secondary_guide_face_ids)
        )
        return interpolate_secondary_geometry_residuals(
            field,
            source_normals,
            source_tangents,
            source_bitangents,
            roots_local,
            normals,
            tangents,
            bitangents,
            self.secondary_guide_points_local,
            self.secondary_render_support(),
        )

    def apply_guide_controls(
        self,
        groom,
        roots_local: torch.Tensor,
        normals: torch.Tensor | None = None,
        tangents: torch.Tensor | None = None,
        bitangents: torch.Tensor | None = None,
        *,
        root_face_ids: torch.Tensor | None = None,
        guide_support: SurfaceSupport | None = None,
        residual_sample_override: InterpolatedGeometryResiduals | None = None,
        length_gradient_confidence: torch.Tensor | None = None,
    ):
        if not self.guide_enabled():
            return self.apply_shape_detail_gate(groom)
        if normals is None:
            normals = F.normalize(self.face_normals[self.face_ids], dim=-1, eps=1.0e-8)
        if tangents is None or bitangents is None:
            tangents, bitangents = self.tangent_frames(normals)
        ranges = self.groom.ranges
        if guide_support is None:
            guide_interp, guide_direction = self.interpolate_guide_controls(
                roots_local,
                normals,
                tangents,
                bitangents,
                length_gradient_confidence=length_gradient_confidence,
            )
        else:
            if root_face_ids is None:
                raise ValueError("explicit guide support requires root_face_ids")
            guide_interp, guide_direction = self.sample_guide_controls(
                roots_local,
                root_face_ids,
                normals,
                tangents,
                bitangents,
                support=guide_support,
                length_gradient_confidence=length_gradient_confidence,
            )
        residual_sample = residual_sample_override
        if residual_sample is None:
            residual_sample = self.geometry_residual_at_render_roots(
                roots_local,
                normals,
                tangents,
                bitangents,
            )
        residual_decoded = residual_sample.decoded if residual_sample is not None else None
        residual_raw = residual_sample.raw if residual_sample is not None else {}
        residual_physical = (
            {
                "clump_strength": RenderGeometryResidualField.scalar_domain_delta(
                    residual_decoded.clump_strength,
                    ranges.clump_strength,
                ),
            }
            if residual_decoded is not None
            else {}
        )

        def mix_scalar(
            name: str,
            render_value: torch.Tensor,
            residual_scale: float,
            bounds: tuple[float, float] | None = None,
            residual_multiplier: float | None = None,
        ) -> torch.Tensor:
            guide_value = guide_interp[name]
            multiplier = self.guide_residual_multiplier if residual_multiplier is None else residual_multiplier
            effective_residual_scale = float(residual_scale) * float(multiplier)
            should_apply = effective_residual_scale > 0.0
            if (
                self.render_geometry_parameterization
                == "zero_centered_unbounded_log_length_residual"
                and name == "length"
            ):
                guide_value = apply_log_ratio_residual(
                    guide_value,
                    residual_raw["length_raw"],
                    effective_residual_scale,
                )
            elif (
                self.render_geometry_parameterization
                == "zero_centered_asinh_log_length_residual"
                and name == "length"
            ):
                guide_value = apply_asinh_log_ratio_residual(
                    guide_value,
                    residual_raw["length_raw"],
                    effective_residual_scale,
                )
            elif (
                self.render_geometry_parameterization == "zero_centered_log_length_residual"
                and name == "length"
            ):
                guide_value = apply_log_ratio_residual(
                    guide_value,
                    residual_decoded.length,
                    effective_residual_scale,
                )
            elif (
                self.render_geometry_parameterization == "zero_centered_residual"
                and name == "length"
            ):
                tiny = torch.as_tensor(
                    torch.finfo(guide_value.dtype).tiny,
                    device=guide_value.device,
                    dtype=guide_value.dtype,
                )
                guide_value = guide_value * (
                    1.0 + effective_residual_scale * residual_decoded.length
                ).clamp_min(tiny)
            elif residual_sample is not None and name in residual_physical:
                guide_value = guide_value + effective_residual_scale * residual_physical[name]
            elif should_apply:
                guide_value = guide_value + effective_residual_scale * (render_value - guide_value)
            range_free_log_length = (
                self.render_geometry_parameterization
                in {
                    "zero_centered_log_length_residual",
                    "zero_centered_unbounded_log_length_residual",
                    "zero_centered_asinh_log_length_residual",
                }
                and name == "length"
            )
            if bounds is not None and not range_free_log_length:
                lo, hi = bounds
                guide_value = guide_value.clamp(float(lo), float(hi))
            return guide_value

        length = mix_scalar(
            "length",
            groom.length,
            self.guide_length_residual_scale,
            None,
        )
        coverage_multiplier = float(getattr(self, "guide_coverage_residual_multiplier", self.guide_residual_multiplier))
        if residual_sample is not None:
            width_residual_scale = (
                float(self.guide_width_residual_scale)
                * float(self.guide_residual_multiplier)
            )
            root_width = apply_asinh_log_ratio_residual(
                guide_interp["root_width"],
                residual_raw["root_width_raw"],
                width_residual_scale,
            )
            tip_ratio = apply_asinh_logit_residual(
                guide_interp["tip_width_ratio"],
                residual_raw["tip_width_ratio_raw"],
                width_residual_scale,
            )
            width_taper = apply_asinh_log_ratio_residual(
                guide_interp["width_taper"],
                residual_raw["width_taper_raw"],
                width_residual_scale,
            )
        else:
            root_width = mix_scalar(
                "root_width",
                groom.root_width,
                self.guide_width_residual_scale,
                None,
                residual_multiplier=coverage_multiplier,
            )
            tip_ratio = groom.tip_width / groom.root_width.clamp_min(EPS)
            width_taper = groom.width_taper
        if residual_sample is not None:
            secondary_shape_multiplier = float(
                getattr(self, "secondary_shape_residual_multiplier", 1.0)
            )
            curl_radius_ratio = apply_asinh_log_ratio_residual(
                guide_interp["curl_radius_ratio"],
                residual_raw["curl_radius_ratio_raw"],
                float(self.guide_curl_residual_scale)
                * secondary_shape_multiplier,
            )
        else:
            curl_radius_ratio = mix_scalar(
                "curl_radius_ratio",
                groom.curl_radius_ratio,
                self.guide_curl_residual_scale,
                None,
            )
        if residual_sample is not None:
            child_radius = apply_asinh_log_ratio_residual(
                guide_interp["child_radius"],
                residual_raw["child_radius_raw"],
                float(self.guide_child_radius_residual_scale)
                * coverage_multiplier,
            )
        else:
            child_radius = mix_scalar(
                "child_radius",
                groom.child_radius,
                self.guide_child_radius_residual_scale,
                None,
                residual_multiplier=coverage_multiplier,
            )
        clump_strength = mix_scalar(
            "clump_strength",
            groom.clump_strength,
            self.guide_clump_residual_scale,
            ranges.clump_strength,
        )
        kwargs = {
            "length": length,
            "root_width": root_width,
            "tip_width": root_width * tip_ratio,
            "width_taper": width_taper,
            "brush_stiffness": guide_interp["brush_stiffness"],
            "curl_radius_ratio": curl_radius_ratio,
            "curl_turns": guide_interp["curl_turns"],
            "curl_phase": torch.zeros_like(guide_interp["curl_turns"]),
            "child_radius": child_radius,
            "clump_strength": clump_strength,
        }
        if guide_direction is not None:
            effective_direction_residual_scale = (
                float(self.guide_direction_residual_scale)
                * float(self.guide_residual_multiplier)
            )
            should_apply_direction_residual = effective_direction_residual_scale > 0.0
            if residual_sample is not None:
                guide_direction = apply_direction_residual(
                    guide_direction,
                    residual_decoded.direction_local,
                    normals,
                    tangents,
                    bitangents,
                    effective_direction_residual_scale,
                )
            elif should_apply_direction_residual:
                render_direction = local_components_to_world(
                    groom.direction_local,
                    normals,
                    tangents,
                    bitangents,
                    normalize=True,
                )
                guide_direction = F.normalize(
                    guide_direction
                    + effective_direction_residual_scale
                    * (render_direction - guide_direction),
                    dim=-1,
                    eps=1.0e-8,
                )
            kwargs["direction_local"] = direction_to_local_components(
                    guide_direction,
                    normals,
                    tangents,
                    bitangents,
                )
        return self.apply_shape_detail_gate(replace(groom, **kwargs))

    def secondary_effective_groom(self):
        """Evaluate the effective geometry on the fixed secondary control set."""

        if not self.secondary_guides_enabled():
            raise RuntimeError("secondary effective groom requested while disabled")
        residual = self.secondary_geometry_residual
        if residual is None:
            raise RuntimeError("secondary geometry residual is unavailable")
        count = int(self.secondary_guide_points_local.shape[0])
        template = self.groom.decode()
        template_values = {}
        for field in fields(template):
            value = getattr(template, field.name)
            template_values[field.name] = value[:1].detach().expand(
                count,
                *value.shape[1:],
            )
        secondary_template = replace(template, **template_values)
        normals, tangents, bitangents = self.tangent_frames_for_face_ids(
            self.secondary_guide_face_ids
        )
        residual_sample = InterpolatedGeometryResiduals(
            raw=dict(residual.named_parameters()),
            decoded=residual.decode(),
        )
        return self.apply_guide_controls(
            secondary_template,
            self.secondary_guide_points_local,
            normals,
            tangents,
            bitangents,
            root_face_ids=self.secondary_guide_face_ids,
            guide_support=self.secondary_primary_support(),
            residual_sample_override=residual_sample,
        )

    def secondary_clean_flow_confidence(self) -> torch.Tensor:
        """Interpolate primary clean-flow evidence onto secondary controls."""

        if not self.secondary_guides_enabled() or self._guide_surface_interpolator is None:
            raise RuntimeError("secondary clean-flow confidence requested while disabled")
        support = self.secondary_primary_support()
        weights = self._guide_surface_interpolator.weights(
            self.secondary_guide_points_local,
            self.secondary_guide_face_ids,
            support,
        )
        return interpolate_physical(
            self.guide_clean_flow_anchor_confidence[:, None],
            support.indices,
            weights,
        ).reshape(-1)

    def apply_shape_detail_gate(self, groom):
        """Stage1-A should render neutral curl detail, not frozen values."""
        multiplier = float(getattr(self, "shape_detail_multiplier", 1.0))
        curl_scale = float(getattr(self, "shape_curl_scale", 1.0))
        if multiplier >= 0.999 and curl_scale >= 0.999:
            return groom
        m = max(0.0, multiplier)
        return replace(
            groom,
            curl_radius_ratio=groom.curl_radius_ratio * m * max(0.0, curl_scale),
        )

    def roots_and_normals(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tri = self.vertices[self.faces[self.face_ids]]
        bary = torch.softmax(self.bary_logits, dim=-1)
        roots_local = (tri * bary[:, :, None]).sum(dim=1)
        roots = roots_local * torch.exp(self.log_scale).view(1, 1) + self.translation.view(1, 3)
        normals = F.normalize(self.face_normals[self.face_ids], dim=-1, eps=1.0e-8)
        return roots, normals, roots_local

    def tangent_frames(self, normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.face_tangents.numel() == 0:
            return make_tangent_frames(normals)
        tangents = self.face_tangents[self.face_ids]
        tangents = tangents - (tangents * normals).sum(dim=-1, keepdim=True) * normals
        tangents = F.normalize(tangents, dim=-1, eps=1.0e-8)
        bitangents = F.normalize(torch.cross(normals, tangents, dim=-1), dim=-1, eps=1.0e-8)
        return tangents, bitangents

    def guide_normals_and_tangent_frames(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.guide_enabled():
            empty = self.vertices.new_empty((0, 3))
            return empty, empty, empty
        normals = F.normalize(self.face_normals[self.guide_face_ids], dim=-1, eps=1.0e-8)
        if self.face_tangents.numel() == 0:
            tangents, bitangents = make_tangent_frames(normals)
        else:
            tangents = self.face_tangents[self.guide_face_ids]
            tangents = tangents - (tangents * normals).sum(dim=-1, keepdim=True) * normals
            tangents = F.normalize(tangents, dim=-1, eps=1.0e-8)
            bitangents = F.normalize(torch.cross(normals, tangents, dim=-1), dim=-1, eps=1.0e-8)
        return normals, tangents, bitangents

    def tangent_frames_for_face_ids(self, face_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normals = F.normalize(self.face_normals[face_ids], dim=-1, eps=1.0e-8)
        if self.face_tangents.numel() == 0:
            tangents, bitangents = make_tangent_frames(normals)
        else:
            tangents = self.face_tangents[face_ids]
            tangents = tangents - (tangents * normals).sum(dim=-1, keepdim=True) * normals
            tangents = F.normalize(tangents, dim=-1, eps=1.0e-8)
            bitangents = F.normalize(torch.cross(normals, tangents, dim=-1), dim=-1, eps=1.0e-8)
        return normals, tangents, bitangents

    def render_parameters(
        self,
        samples: int,
        child_count: int,
        min_segments: int,
        segment_length_origin: float,
        segments_per_unit_length: float,
        segments_per_unit_complexity: float,
        length_overlap: float,
        mesh_no_penetration_field: SignedDistanceGrid | None = None,
        mesh_no_penetration_root_indices: torch.Tensor | None = None,
        strand_crossing_active_set: TorchStrandCrossingActiveSet | None = None,
        view_index: int | None = None,
    ):
        roots, normals, roots_local = self.roots_and_normals()
        tangents, bitangents = self.tangent_frames(normals)
        # R072: a view only owns the image gradient of the roots it is trusted
        # on. Every gate below is straight-through, so the forward value is
        # unchanged and a unit gate reproduces the parent run exactly.
        #
        # Both image-side consumers of the root position are gated: the strand
        # roots themselves, and the copy that drives guide surface-support
        # weights. The mesh-no-penetration constraint and the returned
        # `roots_local` stay ungated, because those are view-independent
        # geometry, and the surface regularizers are what propagate corrections
        # into roots this view is not allowed to move.
        view_gate = None
        if self.view_gate_enabled() and torch.is_grad_enabled():
            if view_index is None:
                raise RuntimeError(
                    "gradient-enabled rendering requires view_index while view gating is active"
                )
            view_gate = self.view_gate_at_render_roots(
                roots_local,
                int(view_index),
            ).reshape(-1, 1)
        guide_sample_local = (
            roots_local
            if view_gate is None
            else straight_through_gate(roots_local, view_gate)
        )
        length_gradient_confidence = (
            self.guide_clean_flow_length_confidence
            if view_gate is not None and self.view_gate_length_confidence_support
            else None
        )
        groom = self.apply_guide_controls(
            self.groom.decode(),
            guide_sample_local,
            length_gradient_confidence=length_gradient_confidence,
        )
        if view_gate is not None and self.view_gate_geometry_support:
            groom = straight_through_gate_geometry(groom, view_gate)
        curl_enabled = (
            self.shape_detail_multiplier > 0.0
            and self.shape_curl_scale > 0.0
        )
        if view_gate is not None:
            roots = straight_through_gate(roots, view_gate)
            groom = replace(
                groom,
                root_opacity=straight_through_gate(groom.root_opacity, view_gate),
                tip_opacity=straight_through_gate(groom.tip_opacity, view_gate),
            )
        strands, widths, colors, opacities = build_strands(
            roots,
            normals,
            tangents,
            bitangents,
            groom,
            samples=samples,
            enable_curl=curl_enabled,
        )
        if strand_crossing_active_set is None:
            strand_crossing_loss = strands.sum() * 0.0
            strand_crossing_stats: dict[str, torch.Tensor | int] = {
                "active_pair_count": 0,
                "positive_pair_count": 0,
                "positive_pair_fraction": strand_crossing_loss.detach(),
                "mean_normalized_depth": strand_crossing_loss.detach(),
                "maximum_normalized_depth": strand_crossing_loss.detach(),
            }
        else:
            crossing_root_indices = strand_crossing_active_set.unique_root_indices
            if crossing_root_indices.numel() and (
                int(crossing_root_indices.min().detach().cpu()) < 0
                or int(crossing_root_indices.max().detach().cpu())
                >= int(roots.shape[0])
            ):
                raise RuntimeError(
                    "strand-crossing active set contains a stale render-root index"
                )
            # Reuse the canonical strands already built for this render. The
            # training backward router restricts this loss to centerline
            # geometry parameters, so global calibration and appearance do not
            # receive crossing gradients without duplicating strand generation.
            selected_strands = strands[crossing_root_indices]
            selected_widths = widths[crossing_root_indices]
            strand_crossing_loss, strand_crossing_stats = (
                active_set_crossing_loss(
                    selected_strands,
                    selected_widths,
                    strand_crossing_active_set,
                )
            )
        if mesh_no_penetration_field is None:
            mesh_no_penetration_depth = roots.new_empty((0, max(int(samples) - 1, 0)))
        else:
            if mesh_no_penetration_root_indices is None:
                raise RuntimeError(
                    "mesh no-penetration is enabled without a root sample"
                )
            selected_groom_values = {
                field.name: getattr(groom, field.name)[
                    mesh_no_penetration_root_indices
                ]
                for field in fields(groom)
            }
            selected_groom_values["length"] = (
                selected_groom_values["length"]
                / torch.exp(self.log_scale.detach())
            )
            selected_groom = replace(groom, **selected_groom_values)
            selected_local, _, _, _ = build_strands(
                roots_local[mesh_no_penetration_root_indices],
                normals[mesh_no_penetration_root_indices],
                tangents[mesh_no_penetration_root_indices],
                bitangents[mesh_no_penetration_root_indices],
                selected_groom,
                samples=samples,
                enable_curl=curl_enabled,
            )
            mesh_no_penetration_depth = strand_penetration_depth(
                selected_local,
                mesh_no_penetration_field,
            )
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
        colors, opacities = self.apply_local_child_support(colors, opacities, root_ids, child_count)
        counts, count_stats = strand_segment_budgets(
            strands.detach(),
            groom.length[root_ids].detach(),
            min_segments,
            segment_length_origin,
            segments_per_unit_length,
            segments_per_unit_complexity,
        )
        resampled = resample_strands_to_segment_budgets(strands, widths, colors, opacities, counts)
        gaussians = strands_to_gaussians(
            resampled.strands,
            resampled.widths,
            resampled.colors,
            resampled.opacities,
            resampled.segment_mask,
            strand_root_indices=root_ids,
            length_overlap=float(length_overlap),
        )
        if self.gaussian_rgb_residual is not None:
            if int(child_count) != 1:
                raise RuntimeError(
                    "Gaussian RGB residual requires child_count=1; child expansion "
                    "has no persistent per-Gaussian ownership contract"
                )
            gaussians = replace(
                gaussians,
                colors=self.gaussian_rgb_residual.apply_to_colors(
                    gaussians.colors,
                    gaussians.root_indices,
                    gaussians.segment_indices,
                    resampled.segment_counts,
                    multiplier=self.gaussian_rgb_residual_multiplier,
                ),
            )
        unique_gaussian_roots = int(torch.unique(gaussians.root_indices.detach()).numel()) if gaussians.root_indices.numel() else 0
        min_expected_roots = max(1, int(0.99 * int(roots.shape[0])))
        if unique_gaussian_roots < min_expected_roots:
            detail = render_parameter_finite_detail(
                roots=roots,
                roots_local=roots_local,
                groom=groom,
                strands=strands,
                widths=widths,
                colors=colors,
                opacities=opacities,
                expanded_root_ids=root_ids,
                gaussians=gaussians,
            )
            detail["min_expected_unique_root_count"] = min_expected_roots
            raise RuntimeError("strand-to-gaussian finite coverage failed: " + json.dumps(detail, sort_keys=True))
        stats = {
            **count_stats,
            **resampled.stats,
            "root_count": int(roots.shape[0]),
            "gaussian_count": int(gaussians.means.shape[0]),
            "gaussian_unique_root_count": unique_gaussian_roots,
            "scale": float(torch.exp(self.log_scale.detach()).cpu()),
            "translation_norm": float(torch.linalg.norm(self.translation.detach()).cpu()),
            "gaussian_rgb_residual_multiplier": float(
                self.gaussian_rgb_residual_multiplier
            ),
        }
        return (
            gaussians,
            roots,
            roots_local,
            stats,
            mesh_no_penetration_depth,
            strand_crossing_loss,
            strand_crossing_stats,
        )

    def apply_local_child_support(
        self,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        root_ids: torch.Tensor,
        child_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.child_color_delta_raw is None:
            return colors, opacities
        child_count = int(child_count)
        if child_count < 1 or child_count > self.max_child_count:
            raise ValueError(f"child_count {child_count} exceeds local child support capacity {self.max_child_count}")
        root_count = int(self.groom.root_color_raw.shape[0])
        expected = root_count * child_count
        if int(root_ids.shape[0]) != expected:
            raise RuntimeError(f"unexpected child expansion shape: {int(root_ids.shape[0])} != {expected}")
        child_ids = torch.arange(child_count, device=colors.device).view(1, child_count).expand(root_count, child_count).reshape(-1)
        if self.child_color_delta_raw is not None:
            delta = torch.tanh(self.child_color_delta_raw[root_ids, child_ids]) * float(self.local_child_color_scale)
            colors = (colors + delta[:, None, :]).clamp(0.0, 1.0)
        return colors, opacities

    def lifecycle_state(self) -> RootLifecycleState:
        _, _, roots_local = self.roots_and_normals()
        return RootLifecycleState(
            points=roots_local.detach(),
            face_ids=self.face_ids.detach().clone(),
            barycentric=torch.softmax(self.bary_logits.detach(), dim=-1),
        )

    def apply_structure_update(
        self,
        update,
        *,
        neighbor_count: int,
    ) -> dict[str, int]:
        old_state = self.lifecycle_state()
        old_count = int(old_state.points.shape[0])
        if update.new_barycentric.numel() == 0 and not bool(update.prune_mask.any()):
            return {"old_root_count": old_count, "root_count_after": old_count}

        ranges = self.groom.ranges
        device = self.vertices.device
        old_params = {name: param.detach() for name, param in self.groom.named_parameters()}
        old_geometry_residual = (
            {name: param.detach() for name, param in self.render_geometry_residual.named_parameters()}
            if self.render_geometry_residual is not None
            else None
        )
        old_child_color_delta = self.child_color_delta_raw.detach() if self.child_color_delta_raw is not None else None
        old_gaussian_rgb_residual = (
            self.gaussian_rgb_residual.raw.detach()
            if self.gaussian_rgb_residual is not None
            else None
        )
        child_count = int(update.new_barycentric.shape[0])
        child_points = (
            (
                self.vertices[self.faces[update.new_face_ids]]
                * update.new_barycentric[:, :, None]
            ).sum(dim=1)
            if child_count
            else old_state.points.new_empty((0, 3))
        )
        if child_count:
            child_support = build_local_surface_support(
                faces=self.faces,
                source_points=old_state.points,
                source_face_ids=old_state.face_ids,
                query_points=child_points,
                query_face_ids=update.new_face_ids,
                neighbor_count=max(1, int(neighbor_count)),
            )
            child_ids = child_support.indices
            child_weights = local_surface_weights(child_points, old_state.points, child_support)
            old_normals, old_tangents, old_bitangents = self.tangent_frames_for_face_ids(old_state.face_ids)
            child_normals, child_tangents, child_bitangents = self.tangent_frames_for_face_ids(update.new_face_ids)
        else:
            child_ids = old_state.face_ids.new_empty((0, 0))
            child_weights = old_state.points.new_empty((0, 0))
            old_normals = old_state.points.new_empty((old_count, 3))
            old_tangents = old_state.points.new_empty((old_count, 3))
            old_bitangents = old_state.points.new_empty((old_count, 3))
            child_normals = old_state.points.new_empty((0, 3))
            child_tangents = old_state.points.new_empty((0, 3))
            child_bitangents = old_state.points.new_empty((0, 3))

        def empty_like_source(source: torch.Tensor) -> torch.Tensor:
            return source.new_empty((0, *source.shape[1:]))

        def interpolate_source(source: torch.Tensor) -> torch.Tensor:
            return interpolate_physical(source, child_ids, child_weights) if child_count else empty_like_source(source)

        decoded = self.groom.decode()
        old_length_reference = self.groom.length_reference.detach()
        old_root_width_reference = self.groom.root_width_reference.detach()
        old_child_radius_reference = self.groom.child_radius_reference.detach()
        child_length_reference = interpolate_source(old_length_reference)
        child_root_width_reference = interpolate_source(old_root_width_reference)
        child_child_radius_reference = interpolate_source(old_child_radius_reference)
        tip_width_ratio = decoded.tip_width / decoded.root_width.clamp_min(EPS)
        tip_opacity_ratio = decoded.tip_opacity / decoded.root_opacity.clamp_min(EPS)
        physical_sources = {
            "length_raw": decoded.length.detach(),
            "root_width_raw": decoded.root_width.detach(),
            "tip_width_ratio_raw": tip_width_ratio.detach(),
            "width_taper_raw": decoded.width_taper.detach(),
            "brush_stiffness_raw": decoded.brush_stiffness.detach(),
            "curl_radius_ratio_raw": decoded.curl_radius_ratio.detach(),
            "curl_turns_raw": decoded.curl_turns.detach(),
            "child_radius_raw": decoded.child_radius.detach(),
            "clump_strength_raw": decoded.clump_strength.detach(),
            "root_color_raw": decoded.root_color.detach(),
            "tip_color_raw": decoded.tip_color.detach(),
            "opacity_raw": decoded.root_opacity.detach(),
            "tip_opacity_ratio_raw": tip_opacity_ratio.detach(),
        }
        raw_bounds = {
            "curl_turns_raw": ranges.curl_turns,
            "clump_strength_raw": ranges.clump_strength,
        }
        child_raw: dict[str, torch.Tensor] = {}
        for name, source in physical_sources.items():
            child_value = interpolate_source(source)
            if name == "brush_stiffness_raw":
                child_raw[name] = encode_brush_stiffness(child_value)
            elif name == "length_raw":
                child_raw[name] = encode_positive_asinh_ratio(
                    child_value,
                    child_length_reference,
                )
            elif name == "root_width_raw":
                child_raw[name] = encode_positive_asinh_ratio(
                    child_value,
                    child_root_width_reference,
                )
            elif name == "width_taper_raw":
                child_raw[name] = encode_positive_asinh(child_value)
            elif name == "child_radius_raw":
                child_raw[name] = encode_positive_asinh_ratio(
                    child_value,
                    child_child_radius_reference,
                )
            elif name in {
                "curl_radius_ratio_raw",
            }:
                child_raw[name] = encode_positive_softplus(child_value)
            elif name in {
                "tip_width_ratio_raw",
                "root_color_raw",
                "tip_color_raw",
                "opacity_raw",
                "tip_opacity_ratio_raw",
            }:
                child_raw[name] = inv_sigmoid(child_value)
            else:
                child_raw[name] = raw_from_range(child_value, raw_bounds[name])
        child_raw["curl_phase"] = (
            interpolate_periodic(self.groom.curl_phase.detach(), child_ids, child_weights)
            if child_count
            else self.groom.curl_phase.new_empty((0, 1))
        )
        if child_count:
            source_direction = groom_direction_3d(
                decoded,
                old_normals,
                old_tangents,
                old_bitangents,
            )
            child_direction = interpolate_directions(
                source_direction,
                old_normals,
                child_normals,
                child_ids,
                child_weights,
            )
            child_raw["direction_local_raw"] = direction_to_local_components(
                child_direction,
                child_normals,
                child_tangents,
                child_bitangents,
            )
        else:
            child_raw["direction_local_raw"] = self.groom.direction_local_raw.new_empty((0, 3))

        new_values = {
            name: apply_attribute_update(values, update, child_raw[name])
            for name, values in old_params.items()
        }

        new_geometry_residual_values: dict[str, torch.Tensor] | None = None
        if self.render_geometry_residual is not None:
            residual_decoded = self.render_geometry_residual.decode()
            new_geometry_residual_values = {}
            asinh_coordinate_names = {
                "root_width": "root_width_log_ratio",
                "tip_width_ratio": "tip_width_logit_delta",
                "width_taper": "width_taper_log_ratio",
                "curl_radius_ratio": "curl_radius_log_ratio",
                "child_radius": "child_radius_log_ratio",
            }
            for name in RenderGeometryResidualField.SCALAR_NAMES:
                raw_name = f"{name}_raw"
                if (
                    name == "length"
                    and self.render_geometry_parameterization
                    in {
                        "zero_centered_unbounded_log_length_residual",
                        "zero_centered_asinh_log_length_residual",
                    }
                ):
                    child_raw_value = interpolate_source(
                        old_geometry_residual[raw_name]
                    )
                elif name in asinh_coordinate_names:
                    source = getattr(
                        residual_decoded,
                        asinh_coordinate_names[name],
                    ).detach()
                    child_coordinate = interpolate_source(source)
                    child_raw_value = torch.sinh(child_coordinate)
                else:
                    source = getattr(residual_decoded, name).detach()
                    child_value = interpolate_source(source)
                    child_raw_value = torch.atanh(child_value.clamp(-0.999, 0.999))
                new_geometry_residual_values[raw_name] = apply_attribute_update(
                    old_geometry_residual[raw_name],
                    update,
                    child_raw_value,
                )

            source_local = residual_decoded.direction_local.detach()
            if child_count:
                source_world = local_components_to_world(
                    source_local,
                    old_normals,
                    old_tangents,
                    old_bitangents,
                    normalize=False,
                )
                neighbor_world = source_world[child_ids]
                transported = parallel_transport_vector_field(
                    neighbor_world,
                    old_normals[child_ids],
                    child_normals[:, None, :].expand_as(neighbor_world),
                )
                child_world = (transported * child_weights[..., None]).sum(dim=1)
                child_local = vector_to_local_components(
                    child_world,
                    child_normals,
                    child_tangents,
                    child_bitangents,
                )
                child_direction_raw = torch.atanh(child_local.clamp(-0.999, 0.999))
            else:
                child_direction_raw = source_local.new_empty((0, 3))
            new_geometry_residual_values["direction_local_raw"] = apply_attribute_update(
                old_geometry_residual["direction_local_raw"],
                update,
                child_direction_raw,
            )

        new_state = apply_structure_update(old_state, update, self.vertices, self.faces)
        new_count = int(new_state.points.shape[0])
        new_length_reference = apply_attribute_update(
            old_length_reference,
            update,
            child_length_reference,
        )
        new_root_width_reference = apply_attribute_update(
            old_root_width_reference,
            update,
            child_root_width_reference,
        )
        new_child_radius_reference = apply_attribute_update(
            old_child_radius_reference,
            update,
            child_child_radius_reference,
        )
        new_groom = GroomParameterField(
            new_count,
            ranges=ranges,
            init_length=self.init_groom_length,
            device=device,
        )
        with torch.no_grad():
            new_groom.length_reference.copy_(
                new_length_reference.to(
                    device=device,
                    dtype=new_groom.length_reference.dtype,
                )
            )
            new_groom.root_width_reference.copy_(
                new_root_width_reference.to(
                    device=device,
                    dtype=new_groom.root_width_reference.dtype,
                )
            )
            new_groom.child_radius_reference.copy_(
                new_child_radius_reference.to(
                    device=device,
                    dtype=new_groom.child_radius_reference.dtype,
                )
            )
            new_params = dict(new_groom.named_parameters())
            for name, value in new_values.items():
                if name not in new_params:
                    raise KeyError(f"unknown groom parameter during structure update: {name}")
                if new_params[name].shape != value.shape:
                    raise RuntimeError(f"groom parameter shape mismatch for {name}: {tuple(new_params[name].shape)} != {tuple(value.shape)}")
                new_params[name].copy_(value.to(device=device, dtype=new_params[name].dtype))

        self.face_ids = new_state.face_ids.detach().long()
        self.bary_initial = new_state.barycentric.detach()
        self.anchor_local = new_state.points.detach()
        self.bary_logits = torch.nn.Parameter(torch.log(self.bary_initial.clamp_min(1.0e-5)))
        self.groom = new_groom
        if new_geometry_residual_values is not None:
            new_geometry_residual = RenderGeometryResidualField(new_count, device=device)
            with torch.no_grad():
                for name, param in new_geometry_residual.named_parameters():
                    value = new_geometry_residual_values[name]
                    if tuple(param.shape) != tuple(value.shape):
                        raise RuntimeError(
                            f"geometry residual shape mismatch for {name}: "
                            f"{tuple(param.shape)} != {tuple(value.shape)}"
                        )
                    param.copy_(value.to(device=device, dtype=param.dtype))
            self.render_geometry_residual = new_geometry_residual
        if old_child_color_delta is not None:
            child_delta = interpolate_source(old_child_color_delta)
            self.child_color_delta_raw = torch.nn.Parameter(apply_attribute_update(old_child_color_delta, update, child_delta))
        if old_gaussian_rgb_residual is not None:
            new_rgb_residual = GaussianRGBResidualField(
                new_count,
                self.gaussian_rgb_residual.control_points,
                self.gaussian_rgb_residual.scale,
                device=device,
            )
            zero_child_residual = old_gaussian_rgb_residual.new_zeros(
                (child_count, *old_gaussian_rgb_residual.shape[1:])
            )
            updated_residual = apply_attribute_update(
                old_gaussian_rgb_residual,
                update,
                zero_child_residual,
            )
            with torch.no_grad():
                new_rgb_residual.raw.copy_(updated_residual)
            self.gaussian_rgb_residual = new_rgb_residual
        old_conf = self.root_observation_confidence.detach()
        child_conf = (
            interpolate_source(old_conf[:, None]).reshape(-1)
            if update.new_barycentric.numel() > 0
            else old_conf.new_empty((0,))
        )
        self.root_observation_confidence = apply_attribute_update(old_conf, update, child_conf).detach().clamp(0.0, 1.0)
        old_clean_dir = self.clean_flow_direction_target.detach()
        old_clean_conf = self.clean_flow_anchor_confidence.detach()
        if update.new_barycentric.numel() > 0:
            child_clean_dir = interpolate_directions(
                old_clean_dir,
                old_normals,
                child_normals,
                child_ids,
                child_weights,
            )
            child_clean_dir = F.normalize(child_clean_dir, dim=-1, eps=1.0e-8)
            child_clean_conf = interpolate_source(old_clean_conf[:, None]).reshape(-1)
        else:
            child_clean_dir = old_clean_dir.new_empty((0, 3))
            child_clean_conf = old_clean_conf.new_empty((0,))
        self.clean_flow_direction_target = F.normalize(
            apply_attribute_update(old_clean_dir, update, child_clean_dir), dim=-1, eps=1.0e-8
        ).detach()
        self.clean_flow_anchor_confidence = apply_attribute_update(old_clean_conf, update, child_clean_conf).detach().clamp(0.0, 1.0)
        old_length_target = self.clean_flow_length_target.detach()
        old_length_conf = self.clean_flow_length_confidence.detach()
        if update.new_barycentric.numel() > 0:
            child_length_target = interpolate_source(old_length_target[:, None]).reshape(-1)
            child_length_conf = interpolate_source(old_length_conf[:, None]).reshape(-1)
        else:
            child_length_target = old_length_target.new_empty((0,))
            child_length_conf = old_length_conf.new_empty((0,))
        self.clean_flow_length_target = apply_attribute_update(old_length_target, update, child_length_target).detach().clamp_min(0.0)
        self.clean_flow_length_confidence = apply_attribute_update(old_length_conf, update, child_length_conf).detach().clamp(0.0, 1.0)
        support_report = self.rebuild_guide_surface_support()
        return {
            "old_root_count": old_count,
            "root_count_after": new_count,
            "guide_support_rebuild": support_report,
        }


@torch.no_grad()
def initialize_groom_from_projections(
    model: WhiteTigerStage1Model,
    image_paths: list[Path],
    mask_paths: list[Path],
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    train_indices: list[int],
    width: int,
    height: int,
    config: Stage1Config,
    device: torch.device,
) -> dict[str, float | int]:
    if config.projected_init_views <= 0:
        return {"projected_init_view_count": 0}
    roots, normals, _ = model.roots_and_normals()
    root_count = int(roots.shape[0])
    color_sum = torch.zeros((root_count, 3), device=device)
    weight_sum = torch.zeros((root_count, 1), device=device)
    chosen = train_indices[: max(1, min(int(config.projected_init_views), len(train_indices)))]
    mesh_for_visibility = TriangleMesh(
        vertices=(
            model.vertices.detach().cpu().numpy() * float(torch.exp(model.log_scale.detach()).cpu())
            + model.translation.detach().cpu().numpy().reshape(1, 3)
        ).astype(np.float32),
        faces=model.faces.detach().cpu().numpy().astype(np.int32),
    )

    setup_progress(
        "projected_init_start",
        root_count=int(root_count),
        view_count=int(len(chosen)),
    )
    for idx in chosen:
        setup_progress("projected_init_view_start", view_index=int(idx))
        image = load_image(image_paths[idx], device)
        mask = load_mask(mask_paths[idx], device)
        mask_conf = mask_edge_confidence(mask, config.projected_init_mask_edge_kernel)
        mesh_depth = render_mesh_depth(mesh_for_visibility, viewmats[idx], ks[idx], width, height, device=device)
        root_vis = sample_mesh_visible_points(
            roots,
            normals,
            viewmats[idx],
            ks[idx],
            mesh_depth.depth,
            depth_abs_tolerance=config.projected_init_depth_abs_tolerance,
            depth_rel_tolerance=config.projected_init_depth_rel_tolerance,
            local_depth_kernel=config.projected_init_local_depth_kernel,
            front_normal_z=config.projected_init_front_normal_z,
        )
        sampled_mask = bilinear_sample(mask, root_vis.xy)[:, 0]
        sampled_conf = bilinear_sample(mask_conf, root_vis.xy)[:, 0]
        angle_weight = view_angle_weight(normals, viewmats[idx], config.projected_init_view_angle_power)
        weight = (sampled_mask * sampled_conf * angle_weight * root_vis.visible.to(sampled_mask.dtype)).clamp(0.0, 1.0)
        good = weight >= float(config.projected_init_min_confidence)
        if not bool(good.any()):
            continue
        sampled_color = bilinear_sample(image, root_vis.xy).clamp(0.0, 1.0)
        w = weight[:, None]
        color_sum[good] += sampled_color[good] * w[good]
        weight_sum[good] += w[good]
        setup_progress(
            "projected_init_view_done",
            view_index=int(idx),
            good_roots=int(good.sum().detach().cpu()),
            observed_roots=int((weight_sum[:, 0] > 0.0).sum().detach().cpu()),
        )

    observed = weight_sum[:, 0] > 0.0
    if bool(observed.any()):
        setup_progress(
            "projected_init_interpolate_start",
            observed_roots=int(observed.sum().detach().cpu()),
            root_count=int(root_count),
        )
        groom = model.groom.decode()
        root_conf = weight_sum[:, 0]
        conf_norm = torch.quantile(root_conf[observed], 0.95).clamp_min(1.0e-6)
        root_conf = (root_conf / conf_norm).clamp(0.0, 1.0)
        colors = groom.root_color.detach().clone()
        colors[observed] = (color_sum[observed] / weight_sum[observed].clamp_min(EPS)).clamp(0.02, 0.98)
        colors, filled_color = interpolate_unobserved_root_values(
            roots,
            colors,
            observed,
            root_conf,
            neighbor_count=config.smooth_graph_k,
            normalize_vectors=False,
        )
        filled = filled_color
        model.groom.root_color_raw[filled] = inv_sigmoid(colors[filled].clamp(0.02, 0.98))
        model.groom.tip_color_raw[filled] = inv_sigmoid((0.88 * colors[filled] + 0.12).clamp(0.02, 0.98))
        model.root_observation_confidence = root_conf.detach()
    else:
        filled = observed
    setup_progress(
        "projected_init_done",
        observed_roots=int(observed.sum().detach().cpu()),
        filled_roots=int(filled.sum().detach().cpu()),
    )
    return {
        "projected_init_view_count": int(len(chosen)),
        "projected_init_observed_roots": int(observed.sum().detach().cpu()),
        "projected_init_observed_fraction": float(observed.float().mean().detach().cpu()),
        "projected_init_interpolated_roots": int((filled & ~observed).sum().detach().cpu()),
        "projected_init_filled_fraction": float(filled.float().mean().detach().cpu()),
    }


@torch.no_grad()
def initialize_groom_from_clean_flow(
    model: WhiteTigerStage1Model,
    targets: CleanFlowTargets,
    config: Stage1Config,
) -> dict[str, float | int | str]:
    roots, normals, _ = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    guide_driven_clean_flow = bool(model.guide_enabled()) and bool(config.guide_roots_from_clean_flow)
    init_sample = sample_clean_flow_targets(
        targets,
        roots,
        normals,
        k=int(config.clean_flow_init_k),
        confidence_floor=float(config.clean_flow_init_min_confidence),
        anchor_only=False,
    )
    anchor_sample = sample_clean_flow_targets(
        targets,
        roots,
        normals,
        k=int(config.clean_flow_init_k),
        confidence_floor=float(config.clean_flow_anchor_min_confidence),
        anchor_only=True,
    )
    length_sample = sample_clean_flow_targets(
        targets,
        roots,
        normals,
        k=int(config.clean_flow_init_k),
        confidence_floor=float(config.clean_flow_length_init_min_confidence),
        anchor_only=False,
    )
    valid_init = init_sample["valid"] & (init_sample["confidence"] >= float(config.clean_flow_init_min_confidence))
    valid_anchor = anchor_sample["valid"] & (anchor_sample["confidence"] >= float(config.clean_flow_anchor_min_confidence))
    render_valid_init = valid_init if not guide_driven_clean_flow else torch.zeros_like(valid_init)
    render_valid_anchor = valid_anchor if not guide_driven_clean_flow else torch.zeros_like(valid_anchor)
    if bool(config.clean_flow_length_init) and not guide_driven_clean_flow:
        render_length_edges, _ = rebuild_graph_edges(
            model,
            mode="surface_hierarchical",
            k=max(1, int(config.clean_flow_init_k)),
        )
        (
            root_length,
            valid_length,
            filled_length,
            length_min,
            length_max,
            length_source_count,
            length_reliable_count,
        ) = data_clamped_clean_flow_length(
            roots,
            render_length_edges,
            length_sample,
            config,
            label="render-root clean-flow length init",
        )
    else:
        root_length = torch.zeros((roots.shape[0], 1), device=roots.device, dtype=roots.dtype)
        valid_length = torch.zeros((roots.shape[0],), device=roots.device, dtype=torch.bool)
        filled_length = valid_length.clone()
        length_min = 0.0
        length_max = 0.0
        length_source_count = 0
        length_reliable_count = 0
    model.clean_flow_length_target = torch.where(
        valid_length,
        root_length.reshape(-1),
        torch.zeros_like(length_sample["shell_height"]),
    ).detach()
    model.clean_flow_length_confidence = torch.where(
        valid_length,
        length_sample["confidence"],
        torch.zeros_like(length_sample["confidence"]),
    ).detach().clamp(0.0, 1.0)
    if bool(config.clean_flow_init) and not guide_driven_clean_flow:
        render_direction_edges, _ = rebuild_graph_edges(
            model,
            mode="surface_hierarchical",
            k=max(1, int(config.clean_flow_init_k)),
        )
        render_direction, render_direction_reliability, render_direction_supported = (
            reconstruct_clean_flow_directions(
                roots,
                init_sample["direction"],
                normals,
                valid_init,
                init_sample["confidence"],
                render_direction_edges,
                label="render-root clean-flow direction init",
            )
        )
        model.groom.direction_local_raw.copy_(
            direction_to_local_components(
                render_direction,
                normals,
                tangents,
                bitangents,
            )
        )
        filled_init = render_direction_supported
        if bool(config.clean_flow_length_init) and bool(filled_length.any()):
            model.groom.length_reference[filled_length] = root_length[filled_length]
            model.groom.length_raw[filled_length].zero_()
        model.root_observation_confidence = torch.maximum(
            model.root_observation_confidence.detach(),
            init_sample["confidence"].detach().clamp(0.0, 1.0),
        )
    else:
        filled_init = torch.zeros((roots.shape[0],), device=roots.device, dtype=torch.bool)
        render_direction_reliability = torch.zeros_like(init_sample["confidence"])
        render_direction_supported = filled_init
    model.clean_flow_direction_target = anchor_sample["direction"].detach()
    model.clean_flow_anchor_confidence = torch.where(
        render_valid_anchor,
        anchor_sample["confidence"],
        torch.zeros_like(anchor_sample["confidence"]),
    ).detach().clamp(0.0, 1.0)

    guide_observed = 0
    guide_anchor = 0
    guide_direction_reconstruction_count = 0
    guide_direction_reconstruction_change_mean = 0.0
    guide_direction_reconstruction_change_p95 = 0.0
    guide_direction_reconstruction_reliability_mean = 0.0
    if model.guide_enabled():
        guide_points = model.guide_points_local * torch.exp(model.log_scale.detach()).view(1, 1) + model.translation.detach().view(1, 3)
        guide_normals, guide_tangents, guide_bitangents = model.guide_normals_and_tangent_frames()
        guide_init = sample_clean_flow_targets(
            targets,
            guide_points,
            guide_normals,
            k=int(config.clean_flow_init_k),
            confidence_floor=float(config.clean_flow_init_min_confidence),
            anchor_only=False,
        )
        guide_anchor_sample = sample_clean_flow_targets(
            targets,
            guide_points,
            guide_normals,
            k=int(config.clean_flow_init_k),
            confidence_floor=float(config.clean_flow_anchor_min_confidence),
            anchor_only=True,
        )
        guide_length_sample = sample_clean_flow_targets(
            targets,
            guide_points,
            guide_normals,
            k=int(config.clean_flow_init_k),
            confidence_floor=float(config.clean_flow_length_init_min_confidence),
            anchor_only=False,
        )
        guide_valid_init = guide_init["valid"] & (guide_init["confidence"] >= float(config.clean_flow_init_min_confidence))
        guide_valid_anchor = guide_anchor_sample["valid"] & (guide_anchor_sample["confidence"] >= float(config.clean_flow_anchor_min_confidence))
        if bool(config.clean_flow_length_init):
            guide_length_edges = model.guide_surface_smoothing_edges(
                max(1, int(config.clean_flow_init_k))
            )
            (
                guide_length,
                guide_valid_length,
                guide_filled_length,
                guide_length_min,
                guide_length_max,
                guide_length_source_count,
                guide_length_reliable_count,
            ) = data_clamped_clean_flow_length(
                guide_points,
                guide_length_edges,
                guide_length_sample,
                config,
                label="guide-root clean-flow length init",
            )
        else:
            guide_length = torch.zeros((guide_points.shape[0], 1), device=guide_points.device, dtype=guide_points.dtype)
            guide_valid_length = torch.zeros((guide_points.shape[0],), device=guide_points.device, dtype=torch.bool)
            guide_filled_length = guide_valid_length.clone()
            guide_length_min = 0.0
            guide_length_max = 0.0
            guide_length_source_count = 0
            guide_length_reliable_count = 0
        model.guide_clean_flow_length_target = torch.where(
            guide_valid_length,
            guide_length.reshape(-1),
            torch.zeros_like(guide_length_sample["shell_height"]),
        ).detach()
        model.guide_clean_flow_length_confidence = torch.where(
            guide_valid_length,
            guide_length_sample["confidence"],
            torch.zeros_like(guide_length_sample["confidence"]),
        ).detach().clamp(0.0, 1.0)
        guide_anchor_confidence = torch.where(
            guide_valid_anchor,
            guide_anchor_sample["confidence"],
            torch.zeros_like(guide_anchor_sample["confidence"]),
        ).detach().clamp(0.0, 1.0)
        if bool(config.clean_flow_init):
            guide_direction_edges = model.guide_surface_smoothing_edges(
                max(1, int(config.clean_flow_init_k))
            )
            guide_direction_before = torch.where(
                guide_valid_anchor[:, None],
                guide_anchor_sample["direction"],
                guide_init["direction"],
            )
            (
                guide_direction_reconstructed,
                guide_direction_reliability,
                guide_direction_supported,
            ) = reconstruct_clean_flow_directions(
                guide_points,
                guide_direction_before,
                guide_normals,
                guide_valid_anchor,
                guide_anchor_confidence,
                guide_direction_edges,
                label="guide-root clean-flow direction init",
            )
            model.guide_direction_local_raw.copy_(
                direction_to_local_components(
                    guide_direction_reconstructed,
                    guide_normals,
                    guide_tangents,
                    guide_bitangents,
                )
            )
            guide_filled_init = guide_direction_supported
            if bool(config.clean_flow_length_init) and bool(guide_filled_length.any()):
                model.guide_length_reference[guide_filled_length] = guide_length[
                    guide_filled_length
                ]
                model.guide_length_raw[guide_filled_length].zero_()
                model.initialize_guide_shape_ratios_from_current_scale()
            direction_cosine = (
                guide_direction_before * guide_direction_reconstructed
            ).sum(dim=-1).clamp(-1.0, 1.0)
            direction_change = torch.rad2deg(torch.acos(direction_cosine))
            guide_direction_reconstruction_count = int(
                guide_direction_supported.sum().detach().cpu()
            )
            guide_direction_reconstruction_change_mean = float(
                direction_change.mean().detach().cpu()
            )
            guide_direction_reconstruction_change_p95 = float(
                torch.quantile(direction_change, 0.95).detach().cpu()
            )
            guide_direction_reconstruction_reliability_mean = float(
                guide_direction_reliability.mean().detach().cpu()
            )
            model.guide_clean_flow_direction_target = guide_direction_reconstructed.detach()
            model.guide_clean_flow_anchor_confidence = guide_direction_reliability.detach()
        else:
            guide_filled_init = torch.zeros_like(guide_valid_init)
            model.guide_clean_flow_direction_target = guide_anchor_sample["direction"].detach()
            model.guide_clean_flow_anchor_confidence = guide_anchor_confidence
        guide_observed = int(guide_valid_init.sum().detach().cpu())
        guide_anchor = int(guide_valid_anchor.sum().detach().cpu())
    else:
        guide_length_source_count = 0
        guide_length_reliable_count = 0
        guide_filled_length = torch.zeros((0,), dtype=torch.bool, device=roots.device)
        guide_filled_init = torch.zeros((0,), dtype=torch.bool, device=roots.device)
        guide_length_min = 0.0
        guide_length_max = 0.0

    observed = int(render_valid_init.sum().detach().cpu())
    anchor = int(render_valid_anchor.sum().detach().cpu())
    length_count = int(valid_length.sum().detach().cpu())
    length_filled_count = int(filled_length.sum().detach().cpu())
    length_values = root_length[valid_length].reshape(-1) if length_count else torch.empty((0,), device=roots.device)
    return {
        "clean_flow_enabled": 1,
        "clean_flow_source": targets.source_path,
        "clean_flow_init": int(bool(config.clean_flow_init)),
        "clean_flow_guide_driven": int(guide_driven_clean_flow),
        "clean_flow_render_root_init_skipped": int(guide_driven_clean_flow),
        "clean_flow_length_init": int(bool(config.clean_flow_length_init)),
        "clean_flow_length_init_scale": float(config.clean_flow_length_init_scale),
        "clean_flow_length_init_min_confidence": float(config.clean_flow_length_init_min_confidence),
        "clean_flow_length_init_count": length_count,
        "clean_flow_length_init_source_count": int(length_source_count),
        "clean_flow_length_init_reliable_count": int(length_reliable_count),
        "clean_flow_length_init_filled_count": length_filled_count,
        "clean_flow_length_init_q05": float(length_min),
        "clean_flow_length_init_q95": float(length_max),
        "clean_flow_length_init_mean": float(length_values.mean().detach().cpu()) if length_count else 0.0,
        "clean_flow_length_init_p50": float(torch.quantile(length_values, 0.50).detach().cpu()) if length_count else 0.0,
        "clean_flow_length_init_p95": float(torch.quantile(length_values, 0.95).detach().cpu()) if length_count else 0.0,
        "clean_flow_root_init_count": observed,
        "clean_flow_root_init_filled_count": int(filled_init.sum().detach().cpu()),
        "clean_flow_root_anchor_count": anchor,
        "clean_flow_root_init_fraction": float(render_valid_init.float().mean().detach().cpu()),
        "clean_flow_root_anchor_fraction": float(render_valid_anchor.float().mean().detach().cpu()),
        "clean_flow_root_nearest_mean": float(init_sample["nearest_distance"].mean().detach().cpu()),
        "clean_flow_root_nearest_max": float(init_sample["nearest_distance"].max().detach().cpu()),
        "clean_flow_init_confidence_mean": float(init_sample["confidence"][valid_init].mean().detach().cpu()) if observed else 0.0,
        "clean_flow_anchor_confidence_mean": float(anchor_sample["confidence"][valid_anchor].mean().detach().cpu()) if anchor else 0.0,
        "clean_flow_guide_init_count": guide_observed,
        "clean_flow_guide_init_filled_count": int(guide_filled_init.sum().detach().cpu()),
        "clean_flow_guide_anchor_count": guide_anchor,
        "clean_flow_guide_direction_reconstruction_supported_count": int(guide_direction_reconstruction_count),
        "clean_flow_guide_direction_reconstruction_change_mean_deg": float(guide_direction_reconstruction_change_mean),
        "clean_flow_guide_direction_reconstruction_change_p95_deg": float(guide_direction_reconstruction_change_p95),
        "clean_flow_guide_direction_reconstruction_reliability_mean": float(guide_direction_reconstruction_reliability_mean),
        "clean_flow_guide_length_init_source_count": int(guide_length_source_count),
        "clean_flow_guide_length_init_reliable_count": int(guide_length_reliable_count),
        "clean_flow_guide_length_init_filled_count": int(guide_filled_length.sum().detach().cpu()),
        "clean_flow_direction_representation": "direct_local_3d",
    }


@torch.no_grad()
def initialize_guide_view_sh_confidence(
    model: WhiteTigerStage1Model,
    clean_flow_target_path: Path,
) -> dict[str, float | int | str | list[int]]:
    if model.guide_view_sh is None:
        return {"guide_view_sh_support": 0}
    trusted = load_trusted_guide_view_confidence(
        clean_flow_target_path,
        expected_face_ids=model.guide_face_ids,
        expected_barycentric=model.guide_barycentric,
        device=model.vertices.device,
    )
    model.set_guide_view_sh_confidence(
        trusted.view_indices,
        trusted.confidence,
    )
    return {
        "guide_view_sh_support": 1,
        **trusted.report(),
    }


def initialize_view_gate(
    model: WhiteTigerStage1Model,
    clean_flow_target_path: Path,
    config: Stage1Config,
    train_indices: list[int],
) -> dict:
    """Install the R072 per-view ownership gate from the accepted V7 target."""

    trusted = load_trusted_guide_view_confidence(
        clean_flow_target_path,
        expected_face_ids=model.guide_face_ids,
        expected_barycentric=model.guide_barycentric,
        device=model.vertices.device,
    )
    ownership = ViewGatedOwnership(
        confidence=trusted,
        floor=float(config.view_gate_floor),
    )
    gate = ownership.cache_matrix(
        train_indices,
        mode=str(config.view_gate_normalization),
    )
    model.set_view_gate(trusted.view_indices, gate, float(config.view_gate_floor))
    return {
        "view_gated_ownership_support": 1,
        "view_gate_geometry_support": int(bool(config.view_gate_geometry_support)),
        "view_gate_length_confidence_support": int(
            bool(config.view_gate_length_confidence_support)
        ),
        **ownership.report(
            train_indices,
            mode=str(config.view_gate_normalization),
        ),
    }


def sample_backing_color(config: Stage1Config, device: torch.device, *, train: bool) -> torch.Tensor:
    if train and config.random_backing_color:
        lo = float(config.backing_color_min)
        hi = float(config.backing_color_max)
        return torch.empty((3,), device=device).uniform_(lo, hi)
    if config.white_background:
        return torch.ones((3,), device=device)
    return torch.zeros((3,), device=device)


def scene_background_color(config: Stage1Config, device: torch.device) -> torch.Tensor:
    if config.white_background:
        return torch.ones((3,), device=device)
    return torch.zeros((3,), device=device)


def current_model_mesh_tensors(model: "WhiteTigerStage1Model", device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.exp(model.log_scale.detach()).view(1, 1)
    vertices = (model.vertices.detach() * scale + model.translation.detach().view(1, 3)).contiguous().to(device=device)
    faces = model.faces.detach().to(device=device, dtype=torch.int32).contiguous()
    return vertices, faces


def sample_mesh_backing_vertex_colors(
    vertices: torch.Tensor,
    config: Stage1Config,
    device: torch.device,
    *,
    train: bool,
) -> torch.Tensor:
    base = sample_backing_color(config, device, train=train)
    if (not train) or (not config.random_mesh_backing_texture):
        return base.view(1, 3).expand(vertices.shape[0], 3).contiguous()

    verts = vertices.detach().to(device=device, dtype=torch.float32)
    center = verts.mean(dim=0, keepdim=True)
    span = (verts.max(dim=0).values - verts.min(dim=0).values).amax().clamp_min(1.0e-6)
    coords = (verts - center) / span
    colors = base.view(1, 3).expand(verts.shape[0], 3).clone()
    octaves = max(int(config.mesh_backing_texture_octaves), 1)
    strength = float(config.mesh_backing_texture_strength)
    for octave in range(octaves):
        direction = F.normalize(torch.randn((3,), device=device), dim=0)
        channel = F.normalize(torch.randn((3,), device=device), dim=0)
        phase = torch.rand((), device=device) * (2.0 * math.pi)
        frequency = 0.75 + 0.55 * float(octave + 1)
        wave = torch.sin((coords @ direction) * (2.0 * math.pi * frequency) + phase)
        colors = colors + (strength / float(octaves)) * wave[:, None] * channel.view(1, 3)
    return colors.clamp(float(config.backing_color_min), float(config.backing_color_max)).contiguous()


def make_mesh_backing_image(
    mesh_depth: MeshDepthResult,
    mesh_color: torch.Tensor,
    scene_background: torch.Tensor,
    *,
    model: "WhiteTigerStage1Model | None" = None,
    viewmat: torch.Tensor | None = None,
    k: torch.Tensor | None = None,
    width: int | None = None,
    height: int | None = None,
    config: Stage1Config | None = None,
    device: torch.device | None = None,
    ctx=None,
    train: bool = False,
) -> torch.Tensor:
    if (
        config is not None
        and model is not None
        and viewmat is not None
        and k is not None
        and width is not None
        and height is not None
        and device is not None
        and config.random_mesh_backing_texture
    ):
        vertices, faces = current_model_mesh_tensors(model, device)
        vertex_colors = sample_mesh_backing_vertex_colors(vertices, config, device, train=train)
        mesh_image, _ = render_mesh_vertex_color_from_tensors(
            vertices,
            faces,
            vertex_colors,
            viewmat,
            k,
            int(width),
            int(height),
            device=device,
            ctx=ctx,
            background=scene_background,
        )
        bg_rgb = scene_background.view(1, 1, 3).expand_as(mesh_image)
        return torch.where(mesh_depth.valid[..., None], mesh_image, bg_rgb)
    mesh_rgb = mesh_color.view(1, 1, 3).expand((*mesh_depth.depth.shape, 3))
    bg_rgb = scene_background.view(1, 1, 3).expand_as(mesh_rgb)
    return torch.where(mesh_depth.valid[..., None], mesh_rgb, bg_rgb)


def composite_target(target: torch.Tensor, mask: torch.Tensor, backing: torch.Tensor) -> torch.Tensor:
    if backing.ndim == 1:
        backing = backing.view(1, 1, 3)
    return target * mask + backing * (1.0 - mask)


@torch.no_grad()
def root_projected_residual(
    model: WhiteTigerStage1Model,
    roots_local: torch.Tensor,
    residual_image: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    mesh_depth: MeshDepthResult,
    config: Stage1Config,
) -> torch.Tensor:
    scale = torch.exp(model.log_scale.detach()).view(1, 1)
    roots = roots_local.detach() * scale + model.translation.detach().view(1, 3)
    xy, depth = project_points(roots, viewmat, k)
    height, width = int(residual_image.shape[0]), int(residual_image.shape[1])
    in_frame = (
        (depth > 1.0e-6)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= height - 1)
    )
    sampled_mesh_depth = sample_depth_nearest(
        mesh_depth.depth,
        xy,
        kernel_size=int(config.mesh_depth_local_kernel),
    )
    tolerance = float(config.mesh_depth_abs_tolerance) + depth.abs() * float(config.mesh_depth_rel_tolerance)
    visible = in_frame & torch.isfinite(sampled_mesh_depth) & (depth <= sampled_mesh_depth + tolerance)
    sampled = bilinear_sample(residual_image, xy).reshape(-1, 1)
    return sampled * visible.float().reshape(-1, 1)


@torch.no_grad()
def pixel_to_root_evidence(
    model: WhiteTigerStage1Model,
    roots_local: torch.Tensor,
    residual_image: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    mesh_depth: MeshDepthResult,
    config: Stage1Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = torch.exp(model.log_scale.detach()).view(1, 1)
    roots = roots_local.detach() * scale + model.translation.detach().view(1, 3)
    root_xy, root_depth = project_points(roots, viewmat, k)
    height, width = int(residual_image.shape[0]), int(residual_image.shape[1])
    in_frame = (
        (root_depth > 1.0e-6)
        & (root_xy[:, 0] >= 0.0)
        & (root_xy[:, 0] <= width - 1)
        & (root_xy[:, 1] >= 0.0)
        & (root_xy[:, 1] <= height - 1)
    )
    sampled_mesh_depth = sample_depth_nearest(
        mesh_depth.depth,
        root_xy,
        kernel_size=int(config.mesh_depth_local_kernel),
    )
    tolerance = float(config.mesh_depth_abs_tolerance) + root_depth.abs() * float(config.mesh_depth_rel_tolerance)
    root_visible = in_frame & torch.isfinite(sampled_mesh_depth) & (root_depth <= sampled_mesh_depth + tolerance)
    visible_ids = torch.nonzero(root_visible, as_tuple=False).reshape(-1)
    residual = torch.zeros((int(root_xy.shape[0]), 1), device=residual_image.device, dtype=residual_image.dtype)
    norm = torch.zeros_like(residual)
    target_sum = torch.zeros((int(root_xy.shape[0]), 3), device=residual_image.device, dtype=residual_image.dtype)
    target_weight = torch.zeros_like(residual)
    invalid_targets = torch.full_like(target_sum, torch.nan)
    if int(visible_ids.numel()) == 0:
        return residual, invalid_targets, target_weight

    flat = residual_image[..., 0].detach().reshape(-1)
    topk = min(max(1, int(config.densify_pixel_evidence_topk)), int(flat.numel()))
    values, flat_ids = torch.topk(flat, k=topk, largest=True, sorted=False)
    keep = values >= float(config.densify_pixel_evidence_min)
    if not bool(keep.any()):
        return residual, invalid_targets, target_weight
    values = values[keep]
    flat_ids = flat_ids[keep]
    pixel_depth = mesh_depth.depth.reshape(-1)[flat_ids].to(dtype=residual_image.dtype)
    depth_valid = torch.isfinite(pixel_depth) & (pixel_depth > 1.0e-6)
    if not bool(depth_valid.any()):
        return residual, invalid_targets, target_weight
    values = values[depth_valid]
    flat_ids = flat_ids[depth_valid]
    pixel_depth = pixel_depth[depth_valid]
    pixel_xy = torch.stack(
        [(flat_ids % width).to(dtype=residual_image.dtype), (flat_ids // width).to(dtype=residual_image.dtype)],
        dim=-1,
    )
    cam_x = (pixel_xy[:, 0] - k[0, 2].to(dtype=residual_image.dtype)) * pixel_depth / k[0, 0].to(dtype=residual_image.dtype)
    cam_y = (pixel_xy[:, 1] - k[1, 2].to(dtype=residual_image.dtype)) * pixel_depth / k[1, 1].to(dtype=residual_image.dtype)
    cam = torch.stack([cam_x, cam_y, pixel_depth], dim=-1)
    rot = viewmat[:3, :3].to(dtype=residual_image.dtype)
    trans = viewmat[:3, 3].to(dtype=residual_image.dtype)
    world = (cam - trans.view(1, 3)) @ rot
    target_local = (world - model.translation.detach().to(dtype=residual_image.dtype).view(1, 3)) / scale.to(dtype=residual_image.dtype)

    visible_xy = root_xy[visible_ids].to(dtype=residual_image.dtype)
    root_k = min(max(1, int(config.densify_pixel_evidence_root_k)), int(visible_ids.numel()))
    chunk = max(1, int(config.densify_pixel_evidence_chunk))
    residual_flat = residual.reshape(-1)
    norm_flat = norm.reshape(-1)
    target_weight_flat = target_weight.reshape(-1)
    for start in range(0, int(pixel_xy.shape[0]), chunk):
        end = min(start + chunk, int(pixel_xy.shape[0]))
        px = pixel_xy[start:end]
        val = values[start:end].to(dtype=residual_image.dtype).reshape(-1, 1)
        target = target_local[start:end]
        dist2 = (px[:, None, :] - visible_xy[None, :, :]).square().sum(dim=-1)
        nn_dist2, nn_local = torch.topk(dist2, k=root_k, largest=False, sorted=False)
        nn_root_ids = visible_ids[nn_local]
        weights = 1.0 / (nn_dist2 + 4.0)
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1.0e-8)
        weighted = val * weights
        residual_flat.scatter_add_(0, nn_root_ids.reshape(-1), weighted.reshape(-1))
        norm_flat.scatter_add_(0, nn_root_ids.reshape(-1), weights.reshape(-1))
        target_weight_flat.scatter_add_(0, nn_root_ids.reshape(-1), weighted.reshape(-1))
        target_sum.scatter_add_(
            0,
            nn_root_ids.reshape(-1, 1).expand(-1, 3),
            (target[:, None, :] * weighted[:, :, None]).reshape(-1, 3),
        )
    averaged_residual = residual / torch.clamp(norm, min=1.0)
    averaged_target = target_sum / torch.clamp(target_weight, min=1.0e-8)
    averaged_target = torch.where(target_weight > 0.0, averaged_target, invalid_targets)
    return averaged_residual, averaged_target, target_weight


@torch.no_grad()
def pixel_to_root_residual(
    model: WhiteTigerStage1Model,
    roots_local: torch.Tensor,
    residual_image: torch.Tensor,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    mesh_depth: MeshDepthResult,
    config: Stage1Config,
) -> torch.Tensor:
    residual, _, _ = pixel_to_root_evidence(model, roots_local, residual_image, viewmat, k, mesh_depth, config)
    return residual


@torch.no_grad()
def densification_residual_image(
    pred_fixed: torch.Tensor,
    target_fixed: torch.Tensor,
    alpha: torch.Tensor,
    mask: torch.Tensor,
    config: Stage1Config,
) -> torch.Tensor:
    rgb_residual = torch.abs(pred_fixed.detach() - target_fixed.detach()).mean(dim=-1, keepdim=True)
    mask_residual = torch.abs(alpha.detach() - mask.detach())
    mode = str(config.densify_residual_mode)
    if mode == "root_pixel":
        return rgb_residual + 0.35 * mask_residual
    if mode == "pixel_to_root":
        alpha_deficit = torch.relu(mask.detach() - alpha.detach())
        detail_residual = rgb_residual * mask.detach()
        return (
            float(config.densify_residual_alpha_weight) * alpha_deficit
            + float(config.densify_residual_rgb_weight) * detail_residual
        )
    if mode != "coverage_pooled":
        raise RuntimeError(f"unknown densify_residual_mode: {mode}")

    alpha_deficit = torch.relu(mask.detach() - alpha.detach())
    detail_residual = rgb_residual * mask.detach()
    radius = max(0, int(config.densify_residual_pool_radius))
    if radius > 0:
        kernel = 2 * radius + 1
        alpha_deficit = F.max_pool2d(alpha_deficit.permute(2, 0, 1).unsqueeze(0), kernel, stride=1, padding=radius)[0].permute(1, 2, 0)
        detail_residual = F.avg_pool2d(detail_residual.permute(2, 0, 1).unsqueeze(0), kernel, stride=1, padding=radius)[0].permute(1, 2, 0)
    return (
        float(config.densify_residual_alpha_weight) * alpha_deficit
        + float(config.densify_residual_rgb_weight) * detail_residual
    )


def render_model_mesh_depth(
    model: WhiteTigerStage1Model,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    *,
    device: torch.device,
    ctx=None,
) -> MeshDepthResult:
    with torch.no_grad():
        vertices, faces = current_model_mesh_tensors(model, device)
        return render_mesh_depth_from_tensors(vertices, faces, viewmat, k, width, height, device=device, ctx=ctx)


def mesh_depth_clip_gaussians(
    gaussians,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    mesh_depth: MeshDepthResult,
    config: Stage1Config,
) -> tuple[object, torch.Tensor, dict[str, float | int], dict[str, torch.Tensor]]:
    if int(config.mesh_depth_local_kernel) != 1:
        raise RuntimeError("formal mesh-depth clipping must use exact per-pixel depth; set mesh_depth_local_kernel=1")
    gaussian_xy, gaussian_depth = project_points(gaussians.means, viewmat, k)
    height, width = int(mesh_depth.depth.shape[0]), int(mesh_depth.depth.shape[1])
    in_frame = (
        (gaussian_depth > 1.0e-6)
        & (gaussian_xy[:, 0] >= 0.0)
        & (gaussian_xy[:, 0] <= width - 1)
        & (gaussian_xy[:, 1] >= 0.0)
        & (gaussian_xy[:, 1] <= height - 1)
    )
    sampled_mesh_depth = sample_depth_nearest(
        mesh_depth.depth,
        gaussian_xy,
        kernel_size=int(config.mesh_depth_local_kernel),
    )
    tolerance = float(config.mesh_depth_abs_tolerance) + gaussian_depth.abs() * float(config.mesh_depth_rel_tolerance)
    behind_mesh = in_frame & torch.isfinite(sampled_mesh_depth) & (gaussian_depth > sampled_mesh_depth + tolerance)
    keep = ~behind_mesh
    if not bool(keep.any()):
        finite_sampled = torch.isfinite(sampled_mesh_depth)
        valid_pair = in_frame & finite_sampled
        margin = gaussian_depth - sampled_mesh_depth - tolerance

        def masked_stats(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
            selected = values[mask]
            if selected.numel() == 0:
                return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
            return {
                "count": float(selected.numel()),
                "mean": float(selected.mean().detach().cpu()),
                "min": float(selected.min().detach().cpu()),
                "max": float(selected.max().detach().cpu()),
            }

        detail = {
            "preclip_gaussian_count": int(gaussians.means.shape[0]),
            "in_frame_count": int(in_frame.sum().detach().cpu()),
            "finite_mesh_depth_count": int(finite_sampled.sum().detach().cpu()),
            "valid_pair_count": int(valid_pair.sum().detach().cpu()),
            "behind_mesh_count": int(behind_mesh.sum().detach().cpu()),
            "positive_depth_count": int((gaussian_depth > 1.0e-6).sum().detach().cpu()),
            "root_index_min": int(gaussians.root_indices.min().detach().cpu()) if gaussians.root_indices.numel() else -1,
            "root_index_max": int(gaussians.root_indices.max().detach().cpu()) if gaussians.root_indices.numel() else -1,
            "unique_root_index_count": int(torch.unique(gaussians.root_indices.detach()).numel()) if gaussians.root_indices.numel() else 0,
            "gaussian_depth": masked_stats(gaussian_depth, torch.isfinite(gaussian_depth)),
            "sampled_mesh_depth": masked_stats(sampled_mesh_depth, finite_sampled),
            "depth_margin_valid_pair": masked_stats(margin, valid_pair),
            "tolerance_valid_pair": masked_stats(tolerance, valid_pair),
            "xy_x_in_frame": masked_stats(gaussian_xy[:, 0], in_frame),
            "xy_y_in_frame": masked_stats(gaussian_xy[:, 1], in_frame),
        }
        raise RuntimeError("mesh-depth clipping removed every Gaussian: " + json.dumps(detail, sort_keys=True))
    clipped = replace(
        gaussians,
        means=gaussians.means[keep],
        directions=gaussians.directions[keep],
        quats=gaussians.quats[keep],
        scales=gaussians.scales[keep],
        colors=gaussians.colors[keep],
        opacities=gaussians.opacities[keep],
        root_indices=gaussians.root_indices[keep],
        segment_indices=gaussians.segment_indices[keep],
    )
    stats = {
        "preclip_gaussian_count": int(gaussians.means.shape[0]),
        "clipped_gaussian_count": int((~keep).sum().detach().cpu()),
        "behind_mesh_gaussian_count": int(behind_mesh.sum().detach().cpu()),
        "kept_gaussian_count": int(keep.sum().detach().cpu()),
        "clip_keep_fraction": float(keep.float().mean().detach().cpu()),
    }
    masks = {
        "behind_mesh_mask": behind_mesh.detach(),
    }
    return clipped, keep, stats, masks


def render_view(
    model: WhiteTigerStage1Model,
    viewmat: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
    config: Stage1Config,
    *,
    background: torch.Tensor,
    mesh_depth: MeshDepthResult | None = None,
    backing_image: torch.Tensor | None = None,
    retain_lifecycle_grad: bool = False,
    mesh_no_penetration_field: SignedDistanceGrid | None = None,
    mesh_no_penetration_root_indices: torch.Tensor | None = None,
    strand_crossing_active_set: TorchStrandCrossingActiveSet | None = None,
    capture_strand_crossing_snapshot: bool = False,
    view_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, object, torch.Tensor, dict[str, float | int], dict]:
    render_args = (
        config.samples,
        config.child_count,
        config.min_segments,
        config.segment_length_origin,
        config.segments_per_unit_length,
        config.segments_per_unit_complexity,
        config.gaussian_length_overlap,
    )
    if memory_constrained_activation_checkpointing(model.groom.length_raw.device):
        (
            gaussians,
            roots,
            roots_local,
            stats,
            mesh_no_penetration_depth,
            strand_crossing_loss,
            strand_crossing_stats,
        ) = activation_checkpoint(
            lambda _anchor: model.render_parameters(
                *render_args,
                mesh_no_penetration_field=mesh_no_penetration_field,
                mesh_no_penetration_root_indices=mesh_no_penetration_root_indices,
                strand_crossing_active_set=strand_crossing_active_set,
                view_index=view_index,
            ),
            model.groom.length_raw,
            use_reentrant=False,
            preserve_rng_state=False,
        )
    else:
        (
            gaussians,
            roots,
            roots_local,
            stats,
            mesh_no_penetration_depth,
            strand_crossing_loss,
            strand_crossing_stats,
        ) = model.render_parameters(
            *render_args,
            mesh_no_penetration_field=mesh_no_penetration_field,
            mesh_no_penetration_root_indices=mesh_no_penetration_root_indices,
            strand_crossing_active_set=strand_crossing_active_set,
            view_index=view_index,
        )
    if model.guide_view_sh is not None:
        root_sh_residual = model.guide_view_sh_residual_at_render_roots(
            roots_local,
            viewmat,
            view_index=view_index,
        )
        gaussian_sh_residual = root_sh_residual[gaussians.root_indices]
        gaussians = replace(
            gaussians,
            colors=(gaussians.colors + gaussian_sh_residual).clamp(0.0, 1.0),
        )
    preclip_gaussians = gaussians
    if config.mesh_depth_clipping:
        if mesh_depth is None:
            raise RuntimeError("mesh_depth_clipping is enabled but render_view received no mesh_depth")
        gaussians, keep_mask, clip_stats, clip_masks = mesh_depth_clip_gaussians(gaussians, viewmat, k, mesh_depth, config)
        stats = {**stats, **clip_stats}
    else:
        keep_mask = torch.ones((gaussians.means.shape[0],), device=gaussians.means.device, dtype=torch.bool)
        clip_masks = {
            "behind_mesh_mask": torch.zeros_like(keep_mask),
        }
        stats = {
            **stats,
            "preclip_gaussian_count": int(gaussians.means.shape[0]),
            "clipped_gaussian_count": 0,
            "kept_gaussian_count": int(gaussians.means.shape[0]),
            "clip_keep_fraction": 1.0,
            "behind_mesh_gaussian_count": 0,
            "no_mesh_depth_gaussian_count": 0,
        }
    if retain_lifecycle_grad:
        roots_local.retain_grad()
        gaussians.means.retain_grad()
        gaussians.scales.retain_grad()
    if config.mesh_backing_compositing:
        if backing_image is None:
            raise RuntimeError("mesh_backing_compositing is enabled but render_view received no backing_image")
        raster_background = torch.zeros((1, 3), device=background.device, dtype=background.dtype)
    else:
        raster_background = background.view(1, 3)
    image, alpha, info = rasterization(
        gaussians.means,
        gaussians.quats,
        gaussians.scales,
        gaussians.opacities.reshape(-1),
        gaussians.colors,
        viewmat.view(1, 4, 4),
        k.view(1, 3, 3),
        width,
        height,
        packed=False,
        backgrounds=raster_background,
        rasterize_mode="antialiased",
    )
    raw_image = image
    if config.mesh_backing_compositing:
        image = image + (1.0 - alpha) * backing_image.view(1, height, width, 3)
    radii = info["radii"].detach()
    tiles_per_gauss = info.get("tiles_per_gauss")
    if torch.is_tensor(tiles_per_gauss):
        tiles_detached = tiles_per_gauss.detach()
        tile_stats = {
            "tile_intersection_count": int(tiles_detached.sum().cpu()),
            "tile_intersections_per_gaussian_mean": float(tiles_detached.float().mean().cpu()),
            "tile_intersections_per_gaussian_max": int(tiles_detached.max().cpu()),
        }
    else:
        tile_stats = {
            "tile_intersection_count": -1,
            "tile_intersections_per_gaussian_mean": -1.0,
            "tile_intersections_per_gaussian_max": -1,
        }
    stats = {
        **stats,
        "visible_gaussian_count": int((radii > 0).sum().cpu()),
        "screen_radius_max": int(radii.max().cpu()),
        **tile_stats,
    }
    info["mesh_depth_keep_mask"] = keep_mask.detach()
    info["mesh_depth_behind_mesh_mask"] = clip_masks["behind_mesh_mask"].detach()
    info["preclip_means"] = preclip_gaussians.means.detach()
    info["raw_fur_image"] = raw_image[0]
    info["mesh_no_penetration_depth"] = mesh_no_penetration_depth
    info["strand_crossing_loss"] = strand_crossing_loss
    info["strand_crossing_stats"] = strand_crossing_stats
    info["strand_crossing_snapshot"] = (
        GaussianSegmentSnapshot.from_tensors(
            means=preclip_gaussians.means,
            directions=preclip_gaussians.directions,
            scales=preclip_gaussians.scales,
            root_indices=preclip_gaussians.root_indices,
            segment_indices=preclip_gaussians.segment_indices,
            length_overlap=float(config.gaussian_length_overlap),
        )
        if capture_strand_crossing_snapshot
        else None
    )
    return image[0].clamp(0.0, 1.0), alpha[0].clamp(0.0, 1.0), gaussians, roots_local, stats, info


@torch.no_grad()
def evaluate(
    model: WhiteTigerStage1Model,
    image_paths: list[Path],
    mask_paths: list[Path],
    viewmats: torch.Tensor,
    ks: torch.Tensor,
    indices: list[int],
    width: int,
    height: int,
    config: Stage1Config,
    metric_computer: MetricComputer,
    device: torch.device,
    mesh_depth_ctx=None,
) -> dict[str, float]:
    raw_psnrs, raw_ssims = [], []
    composite_psnrs, composite_ssims = [], []
    mask_l1s = []
    mesh_color = sample_backing_color(config, device, train=False)
    scene_bg = scene_background_color(config, device)
    for idx in indices:
        target = load_image(image_paths[idx], device)
        mask = load_mask(mask_paths[idx], device)
        mesh_depth = render_model_mesh_depth(model, viewmats[idx], ks[idx], width, height, device=device, ctx=mesh_depth_ctx)
        backing_image = make_mesh_backing_image(
            mesh_depth,
            mesh_color,
            scene_bg,
            model=model,
            viewmat=viewmats[idx],
            k=ks[idx],
            width=width,
            height=height,
            config=config,
            device=device,
            ctx=mesh_depth_ctx,
            train=False,
        )
        try:
            pred, alpha, _, _, _, _ = render_view(
                model,
                viewmats[idx],
                ks[idx],
                width,
                height,
                config,
                background=mesh_color,
                mesh_depth=mesh_depth,
                backing_image=backing_image,
                view_index=idx,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"evaluate render failed: view_index={idx}, root_count={int(model.face_ids.shape[0])}"
            ) from exc
        target_eval = composite_target(target, mask, backing_image)
        raw_metrics = metric_computer.image_metrics(pred, target)
        composite_metrics = metric_computer.image_metrics(pred, target_eval)
        raw_psnrs.append(raw_metrics["psnr"].detach())
        raw_ssims.append(raw_metrics["ssim"].detach())
        composite_psnrs.append(composite_metrics["psnr"].detach())
        composite_ssims.append(composite_metrics["ssim"].detach())
        mask_l1s.append(torch.mean(torch.abs(alpha - mask)).detach())
    if not raw_psnrs:
        return {
            "psnr": 0.0,
            "ssim": 0.0,
            "composite_psnr": 0.0,
            "composite_ssim": 0.0,
            "mask_l1": 0.0,
            "view_count": 0.0,
        }
    return {
        "psnr": float(torch.stack(raw_psnrs).mean().cpu()),
        "ssim": float(torch.stack(raw_ssims).mean().cpu()),
        "composite_psnr": float(torch.stack(composite_psnrs).mean().cpu()),
        "composite_ssim": float(torch.stack(composite_ssims).mean().cpu()),
        "mask_l1": float(torch.stack(mask_l1s).mean().cpu()),
        "view_count": float(len(raw_psnrs)),
    }


def make_stage1_optimizer(model: WhiteTigerStage1Model, config: Stage1Config) -> torch.optim.Optimizer:
    high_frequency_lr = config.lr_groom * float(config.lr_high_frequency_shape_scale)
    groom_params = [
        model.groom.opacity_raw,
        model.groom.tip_opacity_ratio_raw,
    ]
    if model.guide_enabled():
        groom_params.extend(
            [
                model.guide_length_raw,
                model.guide_root_width_raw,
                model.guide_tip_width_ratio_raw,
                model.guide_width_taper_raw,
                model.guide_brush_stiffness_raw,
                model.guide_child_radius_raw,
                model.guide_clump_strength_raw,
                model.guide_direction_local_raw,
            ]
        )
        if model.uses_zero_centered_geometry():
            residual = model.active_geometry_residual()
            if residual is None:
                raise RuntimeError("active geometry residual is unavailable")
            if float(config.guide_length_residual_scale) > 0.0:
                groom_params.append(residual.length_raw)
            if float(config.guide_direction_residual_scale) > 0.0:
                groom_params.append(residual.direction_local_raw)
        else:
            if float(config.guide_length_residual_scale) > 0.0:
                groom_params.append(model.groom.length_raw)
            if float(config.guide_direction_residual_scale) > 0.0:
                groom_params.append(model.groom.direction_local_raw)
        if float(config.guide_width_residual_scale) > 0.0:
            if model.uses_zero_centered_geometry():
                residual = model.active_geometry_residual()
                if residual is None:
                    raise RuntimeError("active geometry residual is unavailable")
                groom_params.extend(
                    [
                        residual.root_width_raw,
                        residual.tip_width_ratio_raw,
                        residual.width_taper_raw,
                    ]
                )
            else:
                groom_params.extend(
                    [
                        model.groom.root_width_raw,
                        model.groom.tip_width_ratio_raw,
                        model.groom.width_taper_raw,
                    ]
                )
    else:
        groom_params.extend(
            [
                model.groom.length_raw,
                model.groom.root_width_raw,
                model.groom.tip_width_ratio_raw,
                model.groom.width_taper_raw,
                model.groom.brush_stiffness_raw,
                model.groom.direction_local_raw,
            ]
        )
    color_params = [model.groom.root_color_raw, model.groom.tip_color_raw]
    if model.child_color_delta_raw is not None:
        color_params.append(model.child_color_delta_raw)
    if model.gaussian_rgb_residual is not None:
        color_params.append(model.gaussian_rgb_residual.raw)
    high_frequency_params = []
    if model.guide_enabled():
        if float(config.shape_curl_scale) > 0.0:
            high_frequency_params.append(model.guide_curl_radius_ratio_raw)
            high_frequency_params.append(model.guide_curl_turns_raw)
        if model.uses_zero_centered_geometry():
            residual = model.active_geometry_residual()
            if residual is None:
                raise RuntimeError("active geometry residual is unavailable")
            if float(config.guide_curl_residual_scale) > 0.0:
                high_frequency_params.append(residual.curl_radius_ratio_raw)
            if float(config.guide_child_radius_residual_scale) > 0.0:
                high_frequency_params.append(residual.child_radius_raw)
            if float(config.guide_clump_residual_scale) > 0.0:
                high_frequency_params.append(residual.clump_strength_raw)
        else:
            if float(config.guide_curl_residual_scale) > 0.0:
                high_frequency_params.extend([model.groom.curl_radius_ratio_raw, model.groom.curl_turns_raw, model.groom.curl_phase])
            if float(config.guide_clump_residual_scale) > 0.0:
                high_frequency_params.append(model.groom.clump_strength_raw)
    else:
        high_frequency_params.extend(
            [
                model.groom.curl_radius_ratio_raw,
                model.groom.curl_turns_raw,
                model.groom.curl_phase,
                model.groom.child_radius_raw,
                model.groom.clump_strength_raw,
            ]
        )
    optimizer_groups = [
        {"params": [model.bary_logits], "lr": config.lr_root},
        {"params": groom_params, "lr": config.lr_groom},
        {"params": high_frequency_params, "lr": high_frequency_lr},
        {"params": color_params, "lr": config.lr_color},
    ]
    if model.guide_view_sh is not None:
        optimizer_groups.append(
            {"params": [model.guide_view_sh.raw], "lr": config.lr_guide_view_sh}
        )
    if float(config.lr_calibration) > 0.0:
        optimizer_groups.insert(
            1,
            {"params": [model.log_scale, model.translation], "lr": config.lr_calibration},
        )
    return torch.optim.Adam(optimizer_groups)


def stage1_optimizer_param_names(model: WhiteTigerStage1Model, config: Stage1Config) -> list[list[str]]:
    groom_names = [
        "groom.opacity_raw",
        "groom.tip_opacity_ratio_raw",
    ]
    if model.guide_enabled():
        residual_prefix = model.geometry_residual_parameter_prefix()
        groom_names.extend(
            [
                "guide_length_raw",
                "guide_root_width_raw",
                "guide_tip_width_ratio_raw",
                "guide_width_taper_raw",
                "guide_brush_stiffness_raw",
                "guide_child_radius_raw",
                "guide_clump_strength_raw",
                "guide_direction_local_raw",
            ]
        )
        if model.uses_zero_centered_geometry():
            if float(config.guide_length_residual_scale) > 0.0:
                groom_names.append(f"{residual_prefix}.length_raw")
            if float(config.guide_direction_residual_scale) > 0.0:
                groom_names.append(f"{residual_prefix}.direction_local_raw")
        else:
            if float(config.guide_length_residual_scale) > 0.0:
                groom_names.append("groom.length_raw")
            if float(config.guide_direction_residual_scale) > 0.0:
                groom_names.append("groom.direction_local_raw")
        if float(config.guide_width_residual_scale) > 0.0:
            if model.uses_zero_centered_geometry():
                groom_names.extend(
                    [
                        f"{residual_prefix}.root_width_raw",
                        f"{residual_prefix}.tip_width_ratio_raw",
                        f"{residual_prefix}.width_taper_raw",
                    ]
                )
            else:
                groom_names.extend(
                    [
                        "groom.root_width_raw",
                        "groom.tip_width_ratio_raw",
                        "groom.width_taper_raw",
                    ]
                )
    else:
        groom_names.extend(
            [
                "groom.length_raw",
                "groom.root_width_raw",
                "groom.tip_width_ratio_raw",
                "groom.width_taper_raw",
                "groom.brush_stiffness_raw",
                "groom.direction_local_raw",
            ]
        )
    color_names = ["groom.root_color_raw", "groom.tip_color_raw"]
    if model.child_color_delta_raw is not None:
        color_names.append("child_color_delta_raw")
    if model.gaussian_rgb_residual is not None:
        color_names.append("gaussian_rgb_residual.raw")
    high_frequency_names = []
    if model.guide_enabled():
        if float(config.shape_curl_scale) > 0.0:
            high_frequency_names.append("guide_curl_radius_ratio_raw")
            high_frequency_names.append("guide_curl_turns_raw")
        if model.uses_zero_centered_geometry():
            if float(config.guide_curl_residual_scale) > 0.0:
                high_frequency_names.append(f"{residual_prefix}.curl_radius_ratio_raw")
            if float(config.guide_child_radius_residual_scale) > 0.0:
                high_frequency_names.append(f"{residual_prefix}.child_radius_raw")
            if float(config.guide_clump_residual_scale) > 0.0:
                high_frequency_names.append(f"{residual_prefix}.clump_strength_raw")
        else:
            if float(config.guide_curl_residual_scale) > 0.0:
                high_frequency_names.extend(["groom.curl_radius_ratio_raw", "groom.curl_turns_raw", "groom.curl_phase"])
            if float(config.guide_clump_residual_scale) > 0.0:
                high_frequency_names.append("groom.clump_strength_raw")
    else:
        high_frequency_names.extend(
            [
                "groom.curl_radius_ratio_raw",
                "groom.curl_turns_raw",
                "groom.curl_phase",
                "groom.child_radius_raw",
                "groom.clump_strength_raw",
            ]
        )
    names = [
        ["bary_logits"],
        groom_names,
        high_frequency_names,
        color_names,
    ]
    if model.guide_view_sh is not None:
        names.append(["guide_view_sh.raw"])
    if float(config.lr_calibration) > 0.0:
        names.insert(1, ["log_scale", "translation"])
    return names


def require_checkpoint_optimizer_param_names(
    checkpoint: dict[str, object],
    model: WhiteTigerStage1Model,
    config: Stage1Config,
) -> None:
    saved_names = checkpoint.get("optimizer_param_names")
    expected_names = stage1_optimizer_param_names(model, config)
    if saved_names != expected_names:
        raise RuntimeError(
            "checkpoint optimizer parameter names mismatch: "
            f"expected={expected_names}, got={saved_names}"
        )


def require_checkpoint_optimizer_state(checkpoint: dict[str, object]) -> None:
    if "optimizer" not in checkpoint:
        raise RuntimeError(
            "resume_optimizer=True requires checkpoint optimizer state"
        )
    if "optimizer_param_names" not in checkpoint:
        raise RuntimeError(
            "resume_optimizer=True requires checkpoint optimizer_param_names"
        )


@dataclass(frozen=True)
class OptimizerRowTransition:
    old_count: int
    keep_mask: torch.Tensor
    child_count: int

    @property
    def retained_count(self) -> int:
        return int(self.keep_mask.sum().detach().cpu())

    @property
    def new_count(self) -> int:
        return self.retained_count + int(self.child_count)


def optimizer_row_transition(
    update: RootStructureUpdate,
    old_count: int,
) -> OptimizerRowTransition:
    if update.prune_mask.shape != (int(old_count),):
        raise ValueError(
            "optimizer row transition prune mask mismatch: "
            f"{tuple(update.prune_mask.shape)} != {(int(old_count),)}"
        )
    return OptimizerRowTransition(
        old_count=int(old_count),
        keep_mask=~update.prune_mask.detach().bool(),
        child_count=int(update.new_barycentric.shape[0]),
    )


def _optimizer_parameter_domain(name: str) -> str | None:
    if name.startswith("guide_"):
        return "guide"
    if name.startswith("secondary_geometry_residual."):
        # Secondary rows are fixed while render/primary lifecycle events run.
        return None
    if (
        name == "bary_logits"
        or name == "child_color_delta_raw"
        or name.startswith("gaussian_rgb_residual.")
        or name.startswith("groom.")
        or name.startswith("render_geometry_residual.")
    ):
        return "render"
    return None


def _optimizer_named_parameters(
    optimizer: torch.optim.Optimizer,
    names: list[list[str]],
) -> dict[str, torch.nn.Parameter]:
    if len(optimizer.param_groups) != len(names):
        raise RuntimeError(
            "optimizer group/name count mismatch: "
            f"{len(optimizer.param_groups)} != {len(names)}"
        )
    result: dict[str, torch.nn.Parameter] = {}
    for group_index, (group, group_names) in enumerate(zip(optimizer.param_groups, names)):
        params = list(group["params"])
        if len(params) != len(group_names):
            raise RuntimeError(
                f"optimizer group {group_index} parameter/name count mismatch: "
                f"{len(params)} != {len(group_names)}"
            )
        for name, parameter in zip(group_names, params):
            if name in result:
                raise RuntimeError(f"duplicate optimizer parameter name: {name}")
            result[name] = parameter
    return result


def rebuild_stage1_optimizer_with_state(
    model: WhiteTigerStage1Model,
    config: Stage1Config,
    old_optimizer: torch.optim.Optimizer,
    old_param_names: list[list[str]],
    *,
    render_transition: OptimizerRowTransition | None = None,
    guide_transition: OptimizerRowTransition | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, object]]:
    """Rebuild Adam after root topology changes without resetting surviving rows."""

    old_named = _optimizer_named_parameters(old_optimizer, old_param_names)
    new_optimizer = make_stage1_optimizer(model, config)
    new_names = stage1_optimizer_param_names(model, config)
    new_named = _optimizer_named_parameters(new_optimizer, new_names)
    if set(old_named) != set(new_named):
        missing = sorted(set(old_named) - set(new_named))
        added = sorted(set(new_named) - set(old_named))
        raise RuntimeError(
            "optimizer parameter contract changed during lifecycle update: "
            f"missing={missing}, added={added}"
        )

    transitions = {
        "render": render_transition,
        "guide": guide_transition,
    }
    exact_parameter_count = 0
    row_migrated_parameter_count = 0
    restored_state_parameter_count = 0
    uninitialized_parameter_count = 0

    for name, old_parameter in old_named.items():
        new_parameter = new_named[name]
        old_state = old_optimizer.state.get(old_parameter)
        if not old_state:
            uninitialized_parameter_count += 1
            continue
        domain = _optimizer_parameter_domain(name)
        transition = transitions.get(domain) if domain is not None else None
        use_row_transition = (
            transition is not None
            and old_parameter.ndim > 0
            and new_parameter.ndim > 0
            and int(old_parameter.shape[0]) == int(transition.old_count)
            and int(new_parameter.shape[0]) == int(transition.new_count)
        )
        if use_row_transition:
            row_migrated_parameter_count += 1
        elif tuple(old_parameter.shape) == tuple(new_parameter.shape):
            exact_parameter_count += 1
        else:
            raise RuntimeError(
                f"optimizer state shape changed without a matching row transition for {name}: "
                f"{tuple(old_parameter.shape)} -> {tuple(new_parameter.shape)}"
            )

        migrated_state: dict[str, object] = {}
        for state_name, value in old_state.items():
            if not torch.is_tensor(value):
                migrated_state[state_name] = value
                continue
            if value.ndim == 0:
                migrated_state[state_name] = value.detach().clone().to(device=new_parameter.device)
                continue
            if tuple(value.shape) != tuple(old_parameter.shape):
                raise RuntimeError(
                    f"unexpected optimizer tensor shape for {name}.{state_name}: "
                    f"{tuple(value.shape)} != {tuple(old_parameter.shape)}"
                )
            if use_row_transition:
                keep = transition.keep_mask.to(device=value.device)
                retained = value[keep]
                migrated = value.new_zeros(tuple(new_parameter.shape), device=new_parameter.device)
                migrated[: retained.shape[0]].copy_(retained.to(device=new_parameter.device))
                migrated_state[state_name] = migrated
            else:
                migrated_state[state_name] = value.detach().clone().to(device=new_parameter.device)
        new_optimizer.state[new_parameter] = migrated_state
        restored_state_parameter_count += 1

    def transition_report(transition: OptimizerRowTransition | None) -> dict[str, int] | None:
        if transition is None:
            return None
        return {
            "old_root_count": int(transition.old_count),
            "retained_root_count": int(transition.retained_count),
            "zero_initialized_child_count": int(transition.child_count),
            "new_root_count": int(transition.new_count),
        }

    return new_optimizer, {
        "old_state_parameter_count": int(len(old_optimizer.state)),
        "restored_state_parameter_count": int(restored_state_parameter_count),
        "uninitialized_parameter_count": int(uninitialized_parameter_count),
        "exact_parameter_count": int(exact_parameter_count),
        "row_migrated_parameter_count": int(row_migrated_parameter_count),
        "render": transition_report(render_transition),
        "guide": transition_report(guide_transition),
    }


def finite_tensor_report(name: str, tensor: torch.Tensor) -> dict[str, object]:
    values = tensor.detach().reshape(-1)
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().detach().cpu())
    count = int(values.numel())
    report: dict[str, object] = {
        "name": name,
        "shape": list(tensor.shape),
        "count": count,
        "finite_count": finite_count,
        "finite_fraction": float(finite_count / max(count, 1)),
    }
    if finite_count > 0:
        selected = values[finite]
        report.update(
            {
                "min": float(selected.min().detach().cpu()),
                "max": float(selected.max().detach().cpu()),
                "mean": float(selected.mean().detach().cpu()),
            }
        )
    return report


def assert_named_tensors_finite(named_tensors: Iterable[tuple[str, torch.Tensor]], context: str) -> None:
    bad = []
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        if not bool(torch.isfinite(tensor).all().detach().cpu()):
            bad.append(finite_tensor_report(name, tensor))
    if bad:
        raise RuntimeError(f"{context}: " + json.dumps({"bad_tensors": bad}, sort_keys=True))


def assert_model_parameters_finite(model: torch.nn.Module, context: str) -> None:
    assert_named_tensors_finite(((name, param) for name, param in model.named_parameters()), context)


def assert_model_gradients_finite(model: torch.nn.Module, context: str) -> None:
    bad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        if not bool(torch.isfinite(grad).all().detach().cpu()):
            bad.append(finite_tensor_report(name, grad))
    if bad:
        raise RuntimeError(f"{context}: " + json.dumps({"bad_gradients": bad}, sort_keys=True))


def render_parameter_finite_detail(
    *,
    roots: torch.Tensor,
    roots_local: torch.Tensor,
    groom,
    strands: torch.Tensor,
    widths: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    expanded_root_ids: torch.Tensor,
    gaussians,
) -> dict[str, object]:
    strand_root_finite = (
        torch.isfinite(strands).all(dim=-1).all(dim=-1)
        & torch.isfinite(widths).all(dim=-1).all(dim=-1)
        & torch.isfinite(colors).all(dim=-1).all(dim=-1)
        & torch.isfinite(opacities).all(dim=-1).all(dim=-1)
    )
    unique_roots = torch.unique(gaussians.root_indices.detach()).numel() if gaussians.root_indices.numel() else 0
    return {
        "root_count": int(roots.shape[0]),
        "expanded_strand_count": int(strands.shape[0]),
        "expanded_finite_strand_count": int(strand_root_finite.sum().detach().cpu()),
        "gaussian_count": int(gaussians.means.shape[0]),
        "gaussian_unique_root_count": int(unique_roots),
        "gaussian_root_index_min": int(gaussians.root_indices.min().detach().cpu()) if gaussians.root_indices.numel() else -1,
        "gaussian_root_index_max": int(gaussians.root_indices.max().detach().cpu()) if gaussians.root_indices.numel() else -1,
        "roots": finite_tensor_report("roots", roots),
        "roots_local": finite_tensor_report("roots_local", roots_local),
        "expanded_root_ids": finite_tensor_report("expanded_root_ids", expanded_root_ids.to(dtype=roots.dtype)),
        "groom_length": finite_tensor_report("groom.length", groom.length),
        "groom_root_width": finite_tensor_report("groom.root_width", groom.root_width),
        "groom_direction_local": finite_tensor_report("groom.direction_local", groom.direction_local),
        "groom_brush_stiffness": finite_tensor_report(
            "groom.brush_stiffness",
            groom.brush_stiffness,
        ),
        "groom_curl_radius_ratio": finite_tensor_report(
            "groom.curl_radius_ratio",
            groom.curl_radius_ratio,
        ),
        "groom_curl_turns": finite_tensor_report("groom.curl_turns", groom.curl_turns),
        "groom_child_radius": finite_tensor_report("groom.child_radius", groom.child_radius),
        "groom_opacity": finite_tensor_report("groom.opacity", groom.opacity),
        "strands": finite_tensor_report("strands", strands),
        "widths": finite_tensor_report("widths", widths),
        "opacities": finite_tensor_report("opacities", opacities),
        "gaussian_means": finite_tensor_report("gaussians.means", gaussians.means),
        "gaussian_scales": finite_tensor_report("gaussians.scales", gaussians.scales),
    }


def zero_color_gradients(model: WhiteTigerStage1Model) -> None:
    params = [model.groom.root_color_raw, model.groom.tip_color_raw]
    if model.child_color_delta_raw is not None:
        params.append(model.child_color_delta_raw)
    if model.gaussian_rgb_residual is not None:
        params.append(model.gaussian_rgb_residual.raw)
    if model.guide_view_sh is not None:
        params.append(model.guide_view_sh.raw)
    for param in params:
        if param.grad is not None:
            param.grad.zero_()


def zero_guide_gradients(
    model: WhiteTigerStage1Model,
    *,
    freeze_length: bool = True,
    freeze_other: bool = True,
) -> None:
    if not model.guide_enabled():
        return
    params: list[torch.nn.Parameter] = []
    if bool(freeze_length):
        params.append(model.guide_length_raw)
    if bool(freeze_other):
        params.extend(
            [
                model.guide_root_width_raw,
                model.guide_tip_width_ratio_raw,
                model.guide_width_taper_raw,
                model.guide_brush_stiffness_raw,
                model.guide_curl_radius_ratio_raw,
                model.guide_curl_turns_raw,
                model.guide_child_radius_raw,
                model.guide_clump_strength_raw,
            ]
        )
        if model.guide_direction_local_raw is not None:
            params.append(model.guide_direction_local_raw)
    for param in params:
        if param is not None and param.grad is not None:
            param.grad.zero_()


def zero_primary_shape_detail_gradients(model: WhiteTigerStage1Model) -> None:
    params = [
        model.groom.curl_radius_ratio_raw,
        model.groom.curl_turns_raw,
        model.groom.curl_phase,
    ]
    if model.guide_enabled():
        params.extend(
            [
                model.guide_curl_radius_ratio_raw,
                model.guide_curl_turns_raw,
            ]
        )
    for param in params:
        if param is not None and param.grad is not None:
            param.grad.zero_()


def zero_secondary_shape_detail_gradients(model: WhiteTigerStage1Model) -> None:
    residual = model.active_geometry_residual()
    if residual is None:
        return
    for param in (residual.curl_radius_ratio_raw,):
        if param.grad is not None:
            param.grad.zero_()


def zero_render_geometry_residual_gradients(model: WhiteTigerStage1Model) -> None:
    """Freeze late geometry residuals while preserving coverage-residual timing.

    Child spread is a coverage control and already has its own early ramp. It
    must not be frozen until the later direction/length residual phase merely
    because both controls share the same lifecycle-aware residual container.
    """

    residual = model.active_geometry_residual()
    if residual is None:
        return
    for name, param in residual.named_parameters():
        if name == "child_radius_raw":
            continue
        if param.grad is not None:
            param.grad.zero_()


def unique_trainable_parameters(
    parameters: list[torch.nn.Parameter],
) -> list[torch.nn.Parameter]:
    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if parameter is None or not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(parameter)
    return unique


def stage1_color_parameters(
    model: WhiteTigerStage1Model,
) -> list[torch.nn.Parameter]:
    parameters = [model.groom.root_color_raw, model.groom.tip_color_raw]
    if model.child_color_delta_raw is not None:
        parameters.append(model.child_color_delta_raw)
    if model.gaussian_rgb_residual is not None:
        parameters.append(model.gaussian_rgb_residual.raw)
    if model.guide_view_sh is not None:
        parameters.append(model.guide_view_sh.raw)
    return unique_trainable_parameters(parameters)


def optimizer_non_color_parameters(
    model: WhiteTigerStage1Model,
    optimizer: torch.optim.Optimizer,
) -> list[torch.nn.Parameter]:
    color_ids = {id(parameter) for parameter in stage1_color_parameters(model)}
    return unique_trainable_parameters(
        [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if id(parameter) not in color_ids
        ]
    )


_STRAND_CROSSING_LOCAL_RESIDUAL_PARAMETER_NAMES = (
    "direction_local_raw",
    "curl_radius_ratio_raw",
)


def strand_crossing_local_shape_named_parameters(
    model: WhiteTigerStage1Model,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the dense local shape field that may resolve a crossing.

    The primary guide field owns the low-frequency groom. A local intersection
    must therefore be corrected by the active zero-centered residual layer,
    rather than rotating a shared primary guide and forcing RGB to compensate
    through guide length.
    """

    residual = model.active_geometry_residual()
    if residual is None:
        return []
    prefix = model.geometry_residual_parameter_prefix()
    return [
        (f"{prefix}.{name}", parameter)
        for name, parameter in residual.named_parameters(recurse=False)
        if name in _STRAND_CROSSING_LOCAL_RESIDUAL_PARAMETER_NAMES
    ]


def optimizer_strand_crossing_shape_parameters(
    model: WhiteTigerStage1Model,
    optimizer: torch.optim.Optimizer,
) -> list[torch.nn.Parameter]:
    """Return optimizer-owned dense local controls for crossing correction."""

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    return unique_trainable_parameters(
        [
            parameter
            for _, parameter in strand_crossing_local_shape_named_parameters(model)
            if id(parameter) in optimizer_parameter_ids
        ]
    )


def backward_stage1_losses(
    model: WhiteTigerStage1Model,
    optimizer: torch.optim.Optimizer,
    *,
    rgb_and_regularization_loss: torch.Tensor,
    flow_loss: torch.Tensor,
    exclude_color_flow_gradients: bool,
    strand_crossing_loss: torch.Tensor | None = None,
) -> None:
    """Backpropagate independently routed RGB, flow, and crossing losses."""

    crossing_parameters = optimizer_strand_crossing_shape_parameters(
        model, optimizer
    )
    route_crossing = (
        strand_crossing_loss is not None
        and strand_crossing_loss.requires_grad
        and bool(crossing_parameters)
    )
    if exclude_color_flow_gradients:
        flow_parameters = optimizer_non_color_parameters(model, optimizer)
        route_flow = bool(flow_parameters) and flow_loss.requires_grad
        rgb_and_regularization_loss.backward(
            retain_graph=route_flow or route_crossing
        )
        if route_flow:
            flow_loss.backward(
                inputs=flow_parameters,
                retain_graph=route_crossing,
            )
    else:
        (rgb_and_regularization_loss + flow_loss).backward(
            retain_graph=route_crossing
        )
    if route_crossing:
        strand_crossing_loss.backward(inputs=crossing_parameters)


def backward_rgb_and_flow_without_color_flow_gradients(
    model: WhiteTigerStage1Model,
    optimizer: torch.optim.Optimizer,
    *,
    rgb_and_regularization_loss: torch.Tensor,
    flow_loss: torch.Tensor,
) -> None:
    """Backpropagate flow only to non-color optimizer parameters."""

    backward_stage1_losses(
        model,
        optimizer,
        rgb_and_regularization_loss=rgb_and_regularization_loss,
        flow_loss=flow_loss,
        exclude_color_flow_gradients=True,
    )


def raw_from_range(value: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    lo, hi = bounds
    rel = (value - lo) / max(hi - lo, EPS)
    return inv_sigmoid(rel)


def rebuild_graph_edges(
    model: WhiteTigerStage1Model,
    *,
    mode: str,
    k: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    started = time.perf_counter()
    with torch.no_grad():
        roots, _, _ = model.roots_and_normals()
        if mode == "euclidean_knn":
            edges = build_knn_edges(roots, k=k)
        elif mode == "surface_hierarchical":
            if not model.guide_enabled():
                raise RuntimeError("surface_hierarchical smoothing requires guide roots")
            support = model.guide_interpolation_support()
            edges = build_hierarchical_surface_edges(
                roots,
                support.indices,
                neighbor_count=k,
            )
        else:
            raise ValueError(f"unknown smoothing graph mode: {mode}")
        return edges, {
            "mode": mode,
            "root_count": int(roots.shape[0]),
            "neighbor_count": min(max(int(k), 0), max(int(roots.shape[0]) - 1, 0)),
            "edge_count": int(edges.shape[0]),
            "build_seconds": float(time.perf_counter() - started),
        }


def build_guide_graph_edges(
    model: WhiteTigerStage1Model,
    *,
    mode: str,
    k: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    started = time.perf_counter()
    if not model.guide_enabled():
        edges = torch.empty((0, 2), dtype=torch.long, device=model.vertices.device)
    elif mode == "euclidean_knn":
        edges = build_knn_edges(model.guide_points_local, k=k)
    elif mode == "surface_hierarchical":
        edges = model.guide_surface_smoothing_edges(k)
    else:
        raise ValueError(f"unknown smoothing graph mode: {mode}")
    return edges, {
        "mode": mode,
        "root_count": int(model.guide_points_local.shape[0]),
        "neighbor_count": min(
            max(int(k), 0),
            max(int(model.guide_points_local.shape[0]) - 1, 0),
        ),
        "edge_count": int(edges.shape[0]),
        "build_seconds": float(time.perf_counter() - started),
    }


@torch.no_grad()
def aggregate_render_need_to_guides(
    model: WhiteTigerStage1Model,
    render_need: torch.Tensor,
    render_visible: torch.Tensor,
    *,
    policy: str,
    legacy_neighbor_count: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int | str]]:
    """Attribute render-root evidence to the guide controls that produced it."""

    _, _, roots_local = model.roots_and_normals()
    guide_count = int(model.guide_points_local.shape[0])
    root_count = int(roots_local.shape[0])
    if render_need.reshape(-1).shape[0] != root_count:
        raise ValueError("render_need must have one value per render root")
    if render_visible.reshape(-1).shape[0] != root_count:
        raise ValueError("render_visible must have one value per render root")

    valid = render_visible.reshape(-1).to(device=roots_local.device, dtype=torch.bool)
    need = render_need.reshape(-1).to(device=roots_local.device, dtype=roots_local.dtype).clamp_min(0.0)
    guide_score_sum = torch.zeros((guide_count,), device=roots_local.device, dtype=roots_local.dtype)
    guide_weight_sum = torch.zeros_like(guide_score_sum)

    if policy == "global_score_budget":
        k = min(max(1, int(legacy_neighbor_count)), guide_count)
        for begin in range(0, root_count, 4096):
            end = min(begin + 4096, root_count)
            distance = torch.cdist(roots_local[begin:end].detach(), model.guide_points_local)
            values, support_ids = torch.topk(distance, k=k, largest=False, dim=-1)
            support_weights = values.clamp_min(1.0e-6).pow(-2.0)
            support_weights = support_weights / support_weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
            support_weights = support_weights * valid[begin:end, None].to(dtype=support_weights.dtype)
            guide_score_sum.scatter_add_(
                0,
                support_ids.reshape(-1),
                (need[begin:end, None] * support_weights).reshape(-1),
            )
            guide_weight_sum.scatter_add_(0, support_ids.reshape(-1), support_weights.reshape(-1))
        support_name = "euclidean_knn"
    elif policy == "surface_attribution_local_max":
        support, support_weights = model.guide_interpolation_attribution(roots_local)
        support_ids = support.indices
        k = int(support_ids.shape[1])
        support_weights = support_weights * valid[:, None].to(dtype=support_weights.dtype)
        guide_score_sum.scatter_add_(
            0,
            support_ids.reshape(-1),
            (need[:, None] * support_weights).reshape(-1),
        )
        guide_weight_sum.scatter_add_(0, support_ids.reshape(-1), support_weights.reshape(-1))
        support_name = "forward_surface_interpolation"
    else:
        raise ValueError(f"unknown guide densification policy: {policy}")

    guide_score = guide_score_sum / guide_weight_sum.clamp_min(EPS)
    return guide_score, guide_weight_sum, {
        "evidence_support": support_name,
        "render_root_k": int(k),
    }


@torch.no_grad()
def select_surface_graph_local_maxima(
    score: torch.Tensor,
    valid: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Keep threshold-valid maxima in each intrinsic guide neighborhood."""

    score = score.reshape(-1)
    valid = valid.reshape(-1).to(device=score.device, dtype=torch.bool)
    if score.shape != valid.shape:
        raise ValueError("score and valid must have identical shapes")
    candidates = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if candidates.numel() == 0:
        return candidates
    if edges.numel() == 0:
        return candidates[torch.argsort(score[candidates], descending=True)]
    edges = edges.to(device=score.device, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape [E, 2]")
    candidate_score = torch.where(valid, score, torch.full_like(score, -torch.inf))
    local_max = candidate_score.clone()
    local_max.scatter_reduce_(
        0,
        edges[:, 0],
        candidate_score[edges[:, 1]],
        reduce="amax",
        include_self=True,
    )
    selected = candidates[score[candidates] >= (local_max[candidates] - 1.0e-12)]
    return selected[torch.argsort(score[selected], descending=True)]


@torch.no_grad()
def propose_guide_densify_update(
    model: WhiteTigerStage1Model,
    stats,
    config: Stage1Config,
    face_adjacency_index: FaceAdjacencyIndex,
) -> tuple[RootStructureUpdate | None, dict[str, float | int | str | dict[str, object]]]:
    if not model.guide_enabled():
        return None, {"enabled": 0, "reason": "guide_disabled"}
    policy = str(config.guide_densify_policy)
    max_splits = int(config.guide_densify_max_splits_per_event)
    if policy == "global_score_budget" and max_splits <= 0:
        return None, {"enabled": 0, "reason": "max_splits_zero"}
    if policy == "surface_attribution_local_max" and max_splits != 0:
        raise ValueError(
            "surface_attribution_local_max guide densification is threshold/local-max driven; "
            "guide_densify_max_splits_per_event must be 0"
        )
    if float(config.guide_densify_score_threshold) <= 0.0:
        return None, {"enabled": 0, "reason": "threshold_zero"}

    _, _, roots_local = model.roots_and_normals()
    guide_count = int(model.guide_points_local.shape[0])
    root_count = int(roots_local.shape[0])
    if guide_count <= 0 or root_count <= 0:
        return None, {"enabled": 0, "reason": "empty_roots"}

    root_scores = normalized_root_need(stats, score_mode=str(config.lifecycle_score_mode))
    render_need = root_scores["need"].detach().reshape(-1)
    render_visible = root_scores["raw_visibility"].detach().reshape(-1) > 0
    if render_need.shape[0] != root_count:
        raise RuntimeError("guide densification root score count does not match render root count")
    valid_render = render_visible & torch.isfinite(render_need)
    if not bool(valid_render.any()):
        return None, {"enabled": 1, "candidate_count": 0, "selected_parent_count": 0, "reason": "no_visible_render_roots"}

    guide_score, guide_weight_sum, attribution_report = aggregate_render_need_to_guides(
        model,
        render_need,
        valid_render,
        policy=policy,
        legacy_neighbor_count=int(config.guide_densify_render_root_k),
    )
    valid_guide = guide_weight_sum > 0.0
    valid_guide = valid_guide & (guide_score >= float(config.guide_densify_score_threshold))
    candidates = torch.nonzero(valid_guide, as_tuple=False).reshape(-1)
    if candidates.numel() == 0:
        return None, {
            "enabled": 1,
            "candidate_count": 0,
            "selected_parent_count": 0,
            "score_mean": float(guide_score.mean().detach().cpu()),
            "score_max": float(guide_score.max().detach().cpu()),
            "threshold": float(config.guide_densify_score_threshold),
        }

    if policy == "global_score_budget":
        order = torch.argsort(guide_score[candidates], descending=True)
        parents = candidates[order[:max_splits]]
        local_max_candidate_count = int(candidates.numel())
        parent_budget = int(max_splits)
    else:
        guide_edges = model.guide_surface_smoothing_edges(int(config.guide_densify_neighbor_count))
        parents = select_surface_graph_local_maxima(guide_score, valid_guide, guide_edges)
        local_max_candidate_count = int(parents.numel())
        parent_budget = -1
    state = model.guide_lifecycle_state()
    child_parent_indices, new_face_ids, new_barycentric = propose_split_children(
        state,
        parents,
        int(config.guide_densify_children_per_parent),
        0.05,
        vertices=model.vertices,
        faces=model.faces,
        neighbor_count=int(config.guide_densify_neighbor_count),
        candidate_rings=int(config.guide_densify_candidate_rings),
        candidate_face_count=int(config.guide_densify_candidate_face_count),
        min_child_distance=float(config.guide_densify_min_child_distance),
        face_adjacency_index=face_adjacency_index,
    )
    prune_mask = torch.zeros((guide_count,), device=roots_local.device, dtype=torch.bool)
    update = RootStructureUpdate(
        parent_indices=parents,
        child_parent_indices=child_parent_indices,
        new_face_ids=new_face_ids,
        new_barycentric=new_barycentric,
        prune_mask=prune_mask,
        scores={
            "guide_score": guide_score,
            "guide_weight": guide_weight_sum,
        },
    )
    selected_scores = guide_score[parents] if parents.numel() > 0 else guide_score.new_empty((0,))
    record = {
        "enabled": 1,
        "candidate_count": int(candidates.numel()),
        "selected_parent_count": int(parents.numel()),
        "inserted_child_count": int(new_barycentric.shape[0]),
        "score_mean": float(guide_score.mean().detach().cpu()),
        "score_max": float(guide_score.max().detach().cpu()),
        "selected_score_mean": float(selected_scores.mean().detach().cpu()) if selected_scores.numel() else 0.0,
        "threshold": float(config.guide_densify_score_threshold),
        "policy": policy,
        **attribution_report,
        "local_max_candidate_count": int(local_max_candidate_count),
        "parent_budget": int(parent_budget),
        "budget_saturated": int(parent_budget > 0 and local_max_candidate_count > parent_budget),
        "replace_parent": 0,
    }
    return update, record


@torch.no_grad()
def surface_flow_directions_local(model: WhiteTigerStage1Model) -> torch.Tensor:
    _, normals, _ = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    groom = model.groom.decode()
    direction = groom_direction_3d(groom, normals, tangents, bitangents)
    tangent = direction - (direction * normals).sum(dim=-1, keepdim=True) * normals
    return F.normalize(tangent, dim=-1, eps=1.0e-8)


@torch.no_grad()
def effective_groom_for_current_roots(model: WhiteTigerStage1Model):
    _, _, roots_local = model.roots_and_normals()
    return model.apply_guide_controls(model.groom.decode(), roots_local)


def scheduled_multiplier(
    iteration: int,
    *,
    initial: float,
    start: int,
    end: int,
) -> float:
    initial = max(0.0, min(1.0, float(initial)))
    start = int(start)
    end = int(end)
    if end <= start:
        return 1.0
    if iteration <= start:
        return initial
    if iteration >= end:
        return 1.0
    t = float(iteration - start) / float(max(1, end - start))
    return initial + (1.0 - initial) * t


def guide_residual_multiplier_for_iteration(config: Stage1Config, iteration: int) -> float:
    return scheduled_multiplier(
        iteration,
        initial=config.guide_residual_initial_multiplier,
        start=config.guide_residual_unlock_start,
        end=config.guide_residual_unlock_end,
    )


def guide_coverage_residual_multiplier_for_iteration(config: Stage1Config, iteration: int) -> float:
    start = int(config.guide_coverage_residual_unlock_start)
    end = int(config.guide_coverage_residual_unlock_end)
    if end <= start:
        return guide_residual_multiplier_for_iteration(config, iteration)
    return scheduled_multiplier(
        iteration,
        initial=config.guide_coverage_residual_initial_multiplier,
        start=start,
        end=end,
    )


def resolved_guide_length_freeze_until(config: Stage1Config) -> int:
    configured = int(config.guide_length_freeze_until)
    return int(config.guide_freeze_until) if configured < 0 else configured


def shape_detail_multiplier_for_iteration(config: Stage1Config, iteration: int) -> float:
    freeze_until = int(config.shape_detail_freeze_until)
    if freeze_until <= 0:
        return 1.0
    configured_end = int(config.shape_detail_unlock_end)
    ramp_end = (
        configured_end
        if configured_end > freeze_until
        else max(freeze_until + 1, int(config.guide_residual_unlock_end))
    )
    return scheduled_multiplier(
        iteration,
        initial=0.0,
        start=freeze_until,
        end=ramp_end,
    )


def secondary_shape_residual_multiplier_for_iteration(
    config: Stage1Config,
    iteration: int,
) -> float:
    start = int(config.secondary_shape_residual_unlock_start)
    end = int(config.secondary_shape_residual_unlock_end)
    if end <= start:
        return shape_detail_multiplier_for_iteration(config, iteration)
    return scheduled_multiplier(
        iteration,
        initial=0.0,
        start=start,
        end=end,
    )


def gaussian_rgb_residual_multiplier_for_iteration(
    config: Stage1Config,
    iteration: int,
) -> float:
    if not config.gaussian_rgb_residual_support:
        return 0.0
    return scheduled_multiplier(
        iteration,
        initial=config.gaussian_rgb_residual_initial_multiplier,
        start=config.gaussian_rgb_residual_unlock_start,
        end=config.gaussian_rgb_residual_unlock_end,
    )


def build_stage1_model_from_checkpoint(
    checkpoint: dict[str, object],
    config: Stage1Config,
    device: torch.device,
) -> WhiteTigerStage1Model:
    """Reconstruct the exact Stage1 model topology stored in a checkpoint."""

    require_current_checkpoint_version(checkpoint)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        raise RuntimeError("Stage1 checkpoint has no model state dictionary")

    def state_array(name: str, dtype: np.dtype) -> np.ndarray:
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Stage1 checkpoint is missing tensor: {name}")
        return value.detach().cpu().numpy().astype(dtype, copy=False)

    mesh = read_obj_mesh(resolve_project_path(config.mesh_path))
    normals = face_normals_np(mesh)
    face_ids = state_array("face_ids", np.int64)
    barycentric = state_array("bary_initial", np.float32)

    face_tangents = None
    face_tangent_state = state.get("face_tangents")
    if isinstance(face_tangent_state, torch.Tensor) and face_tangent_state.numel() > 0:
        face_tangents = face_tangent_state.detach().cpu().numpy().astype(np.float32)
        if face_tangents.shape != (mesh.face_count, 3):
            raise RuntimeError(
                "checkpoint face tangent field shape mismatch: "
                f"{face_tangents.shape} != {(mesh.face_count, 3)}"
            )

    guide_face_ids = None
    guide_barycentric = None
    guide_region_ids = None
    guide_face_state = state.get("guide_face_ids")
    guide_bary_state = state.get("guide_barycentric")
    if isinstance(guide_face_state, torch.Tensor) and guide_face_state.numel() > 0:
        if not isinstance(guide_bary_state, torch.Tensor):
            raise RuntimeError(
                "Stage1 checkpoint has guide face IDs but no guide barycentric coordinates"
            )
        guide_face_ids = guide_face_state.detach().cpu().numpy().astype(np.int64)
        guide_barycentric = guide_bary_state.detach().cpu().numpy().astype(np.float32)
        guide_region_state = state.get("guide_region_weight")
        if isinstance(guide_region_state, torch.Tensor) and guide_region_state.numel() > 0:
            guide_region_ids = (
                guide_region_state.detach().reshape(-1).cpu().numpy() > 0.5
            ).astype(np.int64)
    elif isinstance(guide_bary_state, torch.Tensor) and guide_bary_state.numel() > 0:
        raise RuntimeError(
            "Stage1 checkpoint has guide barycentric coordinates but no guide face IDs"
        )

    secondary_names = (
        "secondary_guide_face_ids",
        "secondary_guide_barycentric",
        "secondary_guide_parent_ids",
    )
    secondary_present = [name in state for name in secondary_names]
    if any(secondary_present) and not all(secondary_present):
        missing = [
            name for name, present in zip(secondary_names, secondary_present) if not present
        ]
        raise RuntimeError(
            "secondary-guide checkpoint is missing persistent topology: "
            + ", ".join(missing)
        )

    secondary_guide_face_ids = None
    secondary_guide_barycentric = None
    secondary_guide_parent_ids = None
    secondary_count = 0
    if all(secondary_present):
        secondary_guide_face_ids = state_array(
            "secondary_guide_face_ids",
            np.int64,
        )
        secondary_guide_barycentric = state_array(
            "secondary_guide_barycentric",
            np.float32,
        )
        secondary_guide_parent_ids = state_array(
            "secondary_guide_parent_ids",
            np.int64,
        )
        secondary_count = int(secondary_guide_face_ids.shape[0])
        if secondary_guide_barycentric.shape != (secondary_count, 3):
            raise RuntimeError(
                "secondary-guide barycentric shape mismatch: "
                f"{secondary_guide_barycentric.shape} != {(secondary_count, 3)}"
            )
        if secondary_guide_parent_ids.shape != (secondary_count,):
            raise RuntimeError(
                "secondary-guide parent shape mismatch: "
                f"{secondary_guide_parent_ids.shape} != {(secondary_count,)}"
            )

    if config.geometry_residual_domain == "secondary_guide":
        if secondary_count <= 0:
            raise RuntimeError(
                "secondary-guide checkpoint contains no persistent secondary roots"
            )
        configured_count = int(config.secondary_guide_root_count)
        if configured_count > 0 and secondary_count != configured_count:
            raise RuntimeError(
                "secondary-guide checkpoint count does not match config: "
                f"{secondary_count} != {configured_count}"
            )
    elif secondary_count > 0:
        raise RuntimeError(
            "checkpoint contains secondary guides but config geometry_residual_domain "
            f"is {config.geometry_residual_domain!r}"
        )
    if secondary_count == 0:
        secondary_guide_face_ids = None
        secondary_guide_barycentric = None
        secondary_guide_parent_ids = None

    model = WhiteTigerStage1Model(
        mesh,
        normals,
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
        gaussian_rgb_residual_support=config.gaussian_rgb_residual_support,
        gaussian_rgb_residual_control_points=config.gaussian_rgb_residual_control_points,
        gaussian_rgb_residual_scale=config.gaussian_rgb_residual_scale,
        guide_view_sh_support=config.guide_view_sh_support,
        guide_view_sh_scale=config.guide_view_sh_scale,
        view_gate_geometry_support=config.view_gate_geometry_support,
        view_gate_length_confidence_support=config.view_gate_length_confidence_support,
        guide_face_ids=guide_face_ids,
        guide_barycentric=guide_barycentric,
        guide_region_ids=guide_region_ids,
        guide_interpolation_k=config.guide_interpolation_k,
        geometry_residual_domain=config.geometry_residual_domain,
        secondary_guide_face_ids=secondary_guide_face_ids,
        secondary_guide_barycentric=secondary_guide_barycentric,
        secondary_guide_parent_ids=secondary_guide_parent_ids,
        secondary_guide_interpolation_k=config.secondary_guide_interpolation_k,
        render_geometry_parameterization=config.render_geometry_parameterization,
        guide_length_residual_scale=config.guide_length_residual_scale,
        guide_direction_residual_scale=config.guide_direction_residual_scale,
        guide_width_residual_scale=config.guide_width_residual_scale,
        guide_child_radius_residual_scale=config.guide_child_radius_residual_scale,
        guide_clump_residual_scale=config.guide_clump_residual_scale,
        guide_curl_residual_scale=config.guide_curl_residual_scale,
        shape_curl_scale=config.shape_curl_scale,
    )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict checkpoint load failed: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    iteration = int(checkpoint.get("iteration", 0))
    model.guide_residual_multiplier = guide_residual_multiplier_for_iteration(
        config,
        iteration,
    )
    model.guide_coverage_residual_multiplier = (
        guide_coverage_residual_multiplier_for_iteration(config, iteration)
    )
    model.shape_detail_multiplier = shape_detail_multiplier_for_iteration(
        config,
        iteration,
    )
    model.secondary_shape_residual_multiplier = (
        secondary_shape_residual_multiplier_for_iteration(config, iteration)
    )
    model.gaussian_rgb_residual_multiplier = (
        gaussian_rgb_residual_multiplier_for_iteration(config, iteration)
    )
    model.eval()
    return model


def load_stage1_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    *,
    mesh_path_override: Path | None = None,
) -> tuple[WhiteTigerStage1Model, Stage1Config, dict[str, object]]:
    checkpoint = load_training_checkpoint(checkpoint_path)
    require_current_checkpoint_version(checkpoint)
    config_mapping = checkpoint.get("config")
    if config_mapping is None:
        config_path = checkpoint_path.parent / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                "checkpoint has no embedded config and no config exists next to it: "
                f"{config_path}"
            )
        config_mapping = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_mapping, dict):
        raise RuntimeError("Stage1 checkpoint config is not a dictionary")
    config = stage1_config_from_checkpoint_mapping(config_mapping)
    if mesh_path_override is not None:
        mesh_path = mesh_path_override.resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"checkpoint mesh override does not exist: {mesh_path}"
            )
        config = replace(config, mesh_path=str(mesh_path))
    model = build_stage1_model_from_checkpoint(checkpoint, config, device)
    return model, config, checkpoint


def lifecycle_statistics_active(
    config: Stage1Config,
    iteration: int,
    *,
    guide_enabled: bool,
) -> bool:
    """Whether this iteration can still contribute to a future lifecycle event."""

    render_future = (
        int(config.densify_interval) > 0
        and int(config.densify_until) >= max(int(iteration), int(config.densify_warmup))
    )
    guide_future = (
        guide_enabled
        and int(config.guide_densify_interval) > 0
        and int(config.guide_densify_start) > 0
        and int(config.guide_densify_until)
        >= max(int(iteration), int(config.guide_densify_start))
    )
    prune_future = (
        int(config.prune_interval) > 0
        and int(config.prune_start) > 0
    )
    return bool(render_future or guide_future or prune_future)


def guide_interpolation_regularization_losses(
    model: WhiteTigerStage1Model,
    config: Stage1Config,
) -> torch.Tensor:
    """Return the configured render-root prior around interpolated guide controls."""

    zero = model.groom.length_raw.sum() * 0.0
    needs_prior = float(config.guide_prior_weight) > 0.0
    if not model.guide_enabled() or not needs_prior:
        return zero
    ranges = model.groom.ranges

    def normalized_l1(value: torch.Tensor, target: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        scale = max(float(hi) - float(lo), EPS)
        return torch.abs(value - target).mean() / scale

    terms: list[torch.Tensor] = []
    weights: list[float] = []
    residual_field = model.active_geometry_residual()
    if residual_field is not None:
        residual = residual_field.decode()

        def add_residual(
            value: torch.Tensor,
            weight: float,
            *,
            reduction: str = "mean_l1",
        ) -> None:
            if weight > 0.0:
                if reduction == "mean_l1":
                    term = torch.abs(value).mean()
                elif reduction == "fourth_moment":
                    term = fourth_moment_norm(value)
                elif reduction == "population_stable_handoff":
                    term = population_stable_residual_norm(
                        value,
                        model.guide_residual_multiplier,
                    )
                elif reduction == "tail_concentration_handoff":
                    term = tail_concentration_residual_loss(
                        value,
                        model.guide_residual_multiplier,
                    )
                else:
                    raise ValueError(
                        f"unsupported render length prior reduction: {reduction}"
                    )
                terms.append(term)
                weights.append(weight)

        add_residual(
            residual.direction_local,
            float(config.guide_prior_direction_weight),
        )
        add_residual(
            residual.curl_radius_log_ratio,
            float(config.guide_prior_curl_weight) if float(config.shape_curl_scale) > 0.0 else 0.0,
        )
        add_residual(
            length_residual_prior_coordinate(
                residual_field.length_raw,
                model.render_geometry_parameterization,
                config.render_length_prior_coordinate,
            ),
            float(config.guide_prior_length_weight),
            reduction=str(config.render_length_prior_reduction),
        )
        add_residual(residual.clump_strength, float(config.guide_prior_clump_weight))
        if float(config.guide_prior_width_weight) > 0.0:
            width_weight = float(config.guide_prior_width_weight)
            add_residual(residual.root_width_log_ratio, width_weight)
            add_residual(residual.tip_width_logit_delta, width_weight)
            add_residual(residual.width_taper_log_ratio, width_weight)
            guide_profile_anchor = 0.5 * (
                torch.asinh(model.guide_tip_width_ratio_raw).abs().mean()
                + torch.asinh(model.guide_width_taper_raw).abs().mean()
            )
            terms.append(guide_profile_anchor)
            weights.append(width_weight)
        add_residual(
            residual.child_radius_log_ratio,
            float(config.guide_prior_child_radius_weight),
        )
        if not terms:
            return zero
        prior_loss = terms[0] * weights[0]
        for term, weight in zip(terms[1:], weights[1:]):
            prior_loss = prior_loss + term * weight
        return prior_loss / max(sum(weights), EPS)

    _, normals, roots_local = model.roots_and_normals()
    tangents, bitangents = model.tangent_frames(normals)
    guide_interp, guide_direction = model.interpolate_guide_controls(
        roots_local,
        normals,
        tangents,
        bitangents,
    )
    if not guide_interp:
        return zero
    groom = model.groom.decode()

    if guide_direction is not None and float(config.guide_prior_direction_weight) > 0.0:
        render_direction = groom_direction_3d(groom, normals, tangents, bitangents)
        guide_direction = F.normalize(guide_direction, dim=-1, eps=1.0e-8)
        direction_loss = 1.0 - (render_direction * guide_direction).sum(dim=-1).clamp(-1.0, 1.0)
        terms.append(direction_loss.mean())
        weights.append(float(config.guide_prior_direction_weight))
    if float(config.guide_prior_length_weight) > 0.0:
        length_log_ratio = torch.log(
            groom.length.clamp_min(EPS)
            / guide_interp["length"].clamp_min(EPS)
        )
        terms.append(length_log_ratio.abs().mean())
        weights.append(float(config.guide_prior_length_weight))
    scalar_terms = [
        (
            "curl_radius_ratio",
            groom.curl_radius_ratio,
            None,
            float(config.guide_prior_curl_weight) if float(config.shape_curl_scale) > 0.0 else 0.0,
        ),
        ("clump_strength", groom.clump_strength, ranges.clump_strength, float(config.guide_prior_clump_weight)),
    ]
    for name, value, bounds, weight in scalar_terms:
        if weight <= 0.0:
            continue
        if bounds is None:
            terms.append(torch.abs(value - guide_interp[name]).mean())
        else:
            terms.append(normalized_l1(value, guide_interp[name], bounds))
        weights.append(weight)
    if float(config.guide_prior_child_radius_weight) > 0.0:
        child_radius_log_ratio = torch.log(
            groom.child_radius.clamp_min(EPS)
            / guide_interp["child_radius"].clamp_min(EPS)
        )
        terms.append(child_radius_log_ratio.abs().mean())
        weights.append(float(config.guide_prior_child_radius_weight))
    if float(config.guide_prior_width_weight) > 0.0:
        width_weight = float(config.guide_prior_width_weight)
        root_width_log_ratio = torch.log(
            groom.root_width.clamp_min(EPS)
            / guide_interp["root_width"].clamp_min(EPS)
        )
        tip_ratio = groom.tip_width / groom.root_width.clamp_min(EPS)
        tip_target = guide_interp["tip_width_ratio"]
        ratio_eps = torch.as_tensor(
            torch.finfo(tip_ratio.dtype).eps,
            device=tip_ratio.device,
            dtype=tip_ratio.dtype,
        )
        tip_logit_delta = torch.logit(
            tip_ratio.clamp(ratio_eps, 1.0 - ratio_eps)
        ) - torch.logit(tip_target.clamp(ratio_eps, 1.0 - ratio_eps))
        taper_log_ratio = torch.log(
            groom.width_taper.clamp_min(EPS)
            / guide_interp["width_taper"].clamp_min(EPS)
        )
        terms.extend(
            [
                root_width_log_ratio.abs().mean(),
                tip_logit_delta.abs().mean(),
                taper_log_ratio.abs().mean(),
            ]
        )
        weights.extend([width_weight, width_weight, width_weight])
    if not needs_prior or not terms:
        prior_loss = zero
    else:
        prior_loss = terms[0] * weights[0]
        for term, weight in zip(terms[1:], weights[1:]):
            prior_loss = prior_loss + term * weight
        prior_loss = prior_loss / max(sum(weights), EPS)
    return prior_loss


def parse_index_override(text: str, default: list[int]) -> list[int]:
    if not text.strip():
        return list(default)
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("view override is empty after parsing")
    return values


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_mesh_no_penetration_config(config: Stage1Config) -> None:
    if config.mesh_no_penetration_support:
        if not config.mesh_no_penetration_sdf.strip():
            raise ValueError(
                "mesh no-penetration requires --mesh-no-penetration-sdf"
            )
        if float(config.mesh_no_penetration_weight) <= 0.0:
            raise ValueError(
                "mesh no-penetration requires a positive loss weight"
            )
        if int(config.mesh_no_penetration_root_batch) <= 0:
            raise ValueError(
                "mesh no-penetration root batch must be positive"
            )
        return
    if config.mesh_no_penetration_sdf.strip():
        raise ValueError(
            "mesh no-penetration SDF was provided while support is disabled"
        )
    if float(config.mesh_no_penetration_weight) != 0.0:
        raise ValueError(
            "mesh no-penetration weight must be zero while support is disabled"
        )


def validate_strand_crossing_config(config: Stage1Config) -> None:
    if config.strand_crossing_support:
        if float(config.strand_crossing_weight) <= 0.0:
            raise ValueError("strand crossing requires a positive loss weight")
        if int(config.strand_crossing_refresh_interval) <= 0:
            raise ValueError("strand crossing refresh interval must be positive")
        if int(config.strand_crossing_query_batch) <= 0:
            raise ValueError("strand crossing query batch must be positive")
        if int(config.strand_crossing_exact_pair_batch) <= 0:
            raise ValueError("strand crossing exact-pair batch must be positive")
        if int(config.child_count) != 1:
            raise ValueError("strand crossing currently requires child_count=1")
        if int(config.guide_root_count) <= 0:
            raise ValueError(
                "strand crossing local correction requires primary guides"
            )
        if config.render_geometry_parameterization == "absolute_endpoint":
            raise ValueError(
                "strand crossing local correction requires zero-centered geometry"
            )
        if (
            config.geometry_residual_domain == "secondary_guide"
            and int(config.secondary_guide_root_count) <= 0
        ):
            raise ValueError(
                "strand crossing secondary correction requires secondary guides"
            )
        if float(config.guide_direction_residual_scale) <= 0.0:
            raise ValueError(
                "strand crossing local correction requires a direction residual"
            )
        if int(config.densify_until) >= int(config.iterations):
            raise ValueError(
                "strand crossing requires a topology-stable interval after densification"
            )
        if int(config.prune_interval) > 0 and int(config.prune_start) <= int(
            config.iterations
        ):
            raise ValueError(
                "strand crossing active-set root IDs require pruning to remain disabled"
            )
        return
    if float(config.strand_crossing_weight) != 0.0:
        raise ValueError(
            "strand crossing weight must be zero while support is disabled"
        )
    if int(config.strand_crossing_refresh_interval) != 0:
        raise ValueError(
            "strand crossing refresh interval must be zero while support is disabled"
        )


def validate_guide_view_sh_config(config: Stage1Config) -> None:
    if not config.guide_view_sh_support:
        return
    if int(config.guide_root_count) <= 0 or not bool(config.guide_roots_from_clean_flow):
        raise ValueError("guide-view SH requires clean-flow-owned primary guides")
    if not config.clean_flow_target.strip():
        raise ValueError("guide-view SH requires --clean-flow-target")
    if float(config.guide_view_sh_scale) <= 0.0:
        raise ValueError("guide-view SH scale must be positive")
    if float(config.lr_guide_view_sh) <= 0.0:
        raise ValueError("guide-view SH learning rate must be positive")
    if int(config.guide_densify_interval) > 0:
        raise ValueError("guide-view SH does not yet support guide-root lifecycle changes")


def validate_view_gate_length_confidence_config(config: Stage1Config) -> None:
    if not config.view_gate_length_confidence_support:
        return
    if not config.view_gate_geometry_support:
        raise ValueError(
            "view-gate length confidence support requires view-gate geometry support"
        )
    if not config.view_gated_ownership_support:
        raise ValueError(
            "view-gate length confidence support requires view-gated ownership support"
        )
    if int(config.guide_root_count) <= 0 or not bool(config.guide_roots_from_clean_flow):
        raise ValueError(
            "view-gate length confidence support requires clean-flow-owned primary guides"
        )
    if not bool(config.clean_flow_length_init):
        raise ValueError(
            "view-gate length confidence support requires clean-flow length init"
        )
    if not config.clean_flow_target.strip():
        raise ValueError(
            "view-gate length confidence support requires --clean-flow-target"
        )


def validate_view_gated_ownership_config(config: Stage1Config) -> None:
    validate_view_gate_length_confidence_config(config)
    floor = float(config.view_gate_floor)
    if config.view_gate_geometry_support and not config.view_gated_ownership_support:
        raise ValueError(
            "view-gate geometry support requires view-gated ownership support"
        )
    if not config.view_gated_ownership_support:
        if floor != 0.0:
            raise ValueError(
                "view gate floor must be zero while view-gated ownership is disabled"
            )
        if str(config.view_gate_normalization) != "raw_q95":
            raise ValueError(
                "view gate normalization must be raw_q95 while support is disabled"
            )
        return
    if str(config.view_gate_normalization) not in {
        "raw_q95",
        "equal_owner_budget",
    }:
        raise ValueError(
            "view gate normalization must be raw_q95 or equal_owner_budget"
        )
    if int(config.guide_root_count) <= 0 or not bool(config.guide_roots_from_clean_flow):
        raise ValueError("view-gated ownership requires clean-flow-owned primary guides")
    if not config.clean_flow_target.strip():
        raise ValueError("view-gated ownership requires --clean-flow-target")
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"view gate floor must lie in [0, 1], got {floor}")
    if str(config.view_gate_normalization) == "equal_owner_budget" and floor != 0.0:
        raise ValueError("equal_owner_budget requires view gate floor 0")
    if int(config.guide_densify_interval) > 0:
        raise ValueError(
            "view-gated ownership does not yet support guide-root lifecycle changes"
        )


def validate_clean_flow_guide_length_anchor_config(config: Stage1Config) -> None:
    """Validate prerequisites for the optional data-identity length anchor."""

    weight = float(config.clean_flow_guide_length_anchor_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(
            "clean-flow guide length anchor weight must be finite and non-negative"
        )
    reduction = str(config.clean_flow_guide_length_anchor_reduction)
    if reduction not in {"mean_l1", "tail_concentration"}:
        raise ValueError(
            "clean-flow guide length anchor reduction must be mean_l1 or "
            "tail_concentration"
        )
    if reduction == "tail_concentration" and weight <= 0.0:
        raise ValueError(
            "tail_concentration clean-flow guide length anchor reduction "
            "requires a positive guide length anchor weight"
        )
    if weight == 0.0:
        return
    if int(config.guide_root_count) <= 0 or not bool(config.guide_roots_from_clean_flow):
        raise ValueError(
            "clean-flow guide length anchor requires clean-flow-owned primary guides"
        )
    if not bool(config.clean_flow_length_init):
        raise ValueError(
            "clean-flow guide length anchor requires clean-flow length init"
        )
    if not config.clean_flow_target.strip():
        raise ValueError(
            "clean-flow guide length anchor requires --clean-flow-target"
        )
    scale = float(config.clean_flow_length_init_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "clean-flow guide length anchor requires a strictly positive "
            "CLEAN_FLOW_LENGTH_INIT_SCALE"
        )


def restore_strand_crossing_state(
    config: Stage1Config,
    checkpoint: dict[str, object] | None,
    *,
    root_count: int,
    device: torch.device,
) -> tuple[
    StrandCrossingActiveSet | None,
    TorchStrandCrossingActiveSet | None,
    int,
    list[dict[str, object]],
]:
    raw_state = (
        checkpoint.get("strand_crossing_active_set")
        if checkpoint is not None
        else None
    )
    raw_last_refresh = (
        checkpoint.get("strand_crossing_last_refresh_iteration")
        if checkpoint is not None
        else None
    )
    raw_history = (
        checkpoint.get("strand_crossing_history", [])
        if checkpoint is not None
        else []
    )
    if not isinstance(raw_history, list):
        raise RuntimeError("strand-crossing checkpoint history is not a list")
    history = [dict(record) for record in raw_history]

    if not config.strand_crossing_support:
        if raw_state is not None or raw_last_refresh not in (None, 0) or history:
            raise RuntimeError(
                "checkpoint contains strand-crossing state while support is disabled"
            )
        return None, None, 0, []
    if raw_state is None:
        if raw_last_refresh not in (None, 0):
            raise RuntimeError(
                "strand-crossing checkpoint has a refresh iteration but no active set"
            )
        return None, None, 0, history
    if not isinstance(raw_state, dict):
        raise RuntimeError("strand-crossing checkpoint active set is not a dictionary")
    if raw_last_refresh is None:
        raise RuntimeError(
            "strand-crossing checkpoint active set has no refresh iteration"
        )

    active_set = StrandCrossingActiveSet.from_checkpoint_state(raw_state)
    if active_set.pair_count:
        minimum_root = int(
            min(
                active_set.first_root_indices.min(),
                active_set.second_root_indices.min(),
            )
        )
        maximum_root = int(
            max(
                active_set.first_root_indices.max(),
                active_set.second_root_indices.max(),
            )
        )
        if minimum_root < 0 or maximum_root >= int(root_count):
            raise RuntimeError(
                "strand-crossing checkpoint contains stale render-root indices: "
                f"range=[{minimum_root}, {maximum_root}], root_count={root_count}"
            )
    return (
        active_set,
        active_set.to_torch(device),
        int(raw_last_refresh),
        history,
    )


def load_mesh_no_penetration_field(
    config: Stage1Config,
    mesh_path: Path,
    device: torch.device,
) -> tuple[SignedDistanceGrid | None, dict[str, object] | None]:
    validate_mesh_no_penetration_config(config)
    if not config.mesh_no_penetration_support:
        return None, None

    sdf_path = resolve_project_path(config.mesh_no_penetration_sdf)
    if not sdf_path.is_file():
        raise FileNotFoundError(f"mesh no-penetration SDF does not exist: {sdf_path}")
    field = SignedDistanceGrid.from_npz(sdf_path, device=device)
    metadata = dict(field.metadata)
    expected = {
        "sign_convention": "outside_positive_inside_negative",
        "storage_order": "zyx",
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            raise RuntimeError(
                f"mesh no-penetration SDF {name} mismatch: "
                f"expected {value!r}, got {metadata.get(name)!r}"
            )
    mesh_hash = file_sha256(mesh_path)
    if metadata.get("mesh_sha256") != mesh_hash:
        raise RuntimeError(
            "mesh no-penetration SDF was built from a different mesh: "
            f"expected {mesh_hash}, got {metadata.get('mesh_sha256')!r}"
        )
    field.requires_grad_(False)
    field.eval()
    report = {
        "sdf_path": str(sdf_path),
        "sdf_sha256": file_sha256(sdf_path),
        "mesh_sha256": mesh_hash,
        "shape_zyx": [int(value) for value in field.values_zyx.shape],
        "reference_length": float(field.reference_length.detach().cpu()),
        "voxel_size": float(metadata["voxel_size"]),
        "signed_distance_backend": metadata["signed_distance_backend"],
        "validation_p95_error_voxels": float(
            metadata["absolute_error_voxels_p95"]
        ),
        "validation_sign_agreement": float(
            metadata["normal_offset_expected_sign_agreement"]
        ),
    }
    return field, report


def train_white_tiger_stage1(config: Stage1Config) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("White Tiger Stage 1 requires CUDA")
    validate_strand_crossing_config(config)
    validate_guide_view_sh_config(config)
    validate_view_gated_ownership_config(config)
    validate_clean_flow_guide_length_anchor_config(config)
    if float(config.geometry_residual_smooth_scale) < 0.0:
        raise ValueError("geometry residual smooth scale must be non-negative")
    shape_detail_enabled = (
        float(config.shape_curl_scale) > 0.0
    )
    if (
        shape_detail_enabled
        and int(config.shape_detail_freeze_until) > 0
        and int(config.shape_detail_unlock_end)
        <= int(config.shape_detail_freeze_until)
    ):
        raise ValueError(
            "enabled primary shape detail requires SHAPE_DETAIL_UNLOCK_END "
            "after SHAPE_DETAIL_FREEZE_UNTIL"
        )
    secondary_shape_enabled = (
        int(config.guide_root_count) > 0
        and config.render_geometry_parameterization != "absolute_endpoint"
        and (
            float(config.guide_curl_residual_scale) > 0.0
        )
    )
    if (
        secondary_shape_enabled
        and int(config.secondary_shape_residual_unlock_end)
        <= int(config.secondary_shape_residual_unlock_start)
    ):
        raise ValueError(
            "enabled secondary shape residual requires "
            "SECONDARY_SHAPE_RESIDUAL_UNLOCK_END after "
            "SECONDARY_SHAPE_RESIDUAL_UNLOCK_START"
        )
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.geometry_residual_domain == "secondary_guide":
        if config.render_geometry_parameterization == "absolute_endpoint":
            raise ValueError(
                "secondary-guide geometry requires a zero-centered residual parameterization"
            )
        if int(config.secondary_guide_root_count) <= 0:
            raise ValueError(
                "secondary-guide geometry requires SECONDARY_GUIDE_ROOT_COUNT > 0"
            )
        if (
            int(config.guide_densify_interval) > 0
            or int(config.guide_densify_max_splits_per_event) > 0
        ):
            raise ValueError(
                "primary guide densification is not implemented for fixed secondary guides"
            )
    elif int(config.secondary_guide_root_count) != 0:
        raise ValueError(
            "SECONDARY_GUIDE_ROOT_COUNT must be zero unless geometry residual domain is secondary_guide"
        )

    data_root = resolve_project_path(config.data_root)
    mesh_path = resolve_project_path(config.mesh_path)
    output_dir = resolve_project_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_stage1_input_report(data_root, mesh_path, test_stride=config.test_stride)
    if report.errors:
        raise RuntimeError(f"input report errors: {report.errors}")
    if tuple(report.image_size or ()) != (config.expected_width, config.expected_height):
        raise RuntimeError(f"expected native {config.expected_width}x{config.expected_height}, got {report.image_size}")
    mesh_no_penetration_field, mesh_no_penetration_report = (
        load_mesh_no_penetration_field(config, mesh_path, device)
    )
    (output_dir / "stage1_inputs.json").write_text(json.dumps(report.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    if mesh_no_penetration_report is not None:
        (output_dir / "mesh_no_penetration_sdf.json").write_text(
            json.dumps(mesh_no_penetration_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    image_paths = list_images(Path(report.image_dir))
    mask_paths = list_images(Path(report.mask_dir))
    angle_paths = sorted((Path(report.orientation_root) / "angles").glob("*.png"))
    conf_paths = sorted((Path(report.orientation_root) / "vars").glob("*.npy"))
    if len(angle_paths) != report.image_count or len(conf_paths) != report.image_count:
        raise RuntimeError("orientation map count mismatch")

    resume_checkpoint = None
    resume_model_state = None
    if config.resume_checkpoint:
        checkpoint_path = resolve_project_path(config.resume_checkpoint)
        resume_checkpoint = load_training_checkpoint(checkpoint_path)
        resume_model_state = resume_checkpoint["model"]

    mesh = read_obj_mesh(mesh_path)
    if resume_model_state is not None:
        resume_face_ids = resume_model_state["face_ids"].detach().cpu().numpy().astype(np.int64)
        resume_barycentric = resume_model_state["bary_initial"].detach().cpu().numpy().astype(np.float32)
        surface_roots = SurfaceRoots(
            points=barycentric_to_points(mesh.vertices, mesh.faces, resume_face_ids, resume_barycentric),
            face_ids=resume_face_ids,
            barycentric=resume_barycentric,
            selected_candidate_ids=np.arange(int(resume_face_ids.shape[0]), dtype=np.int64),
            candidate_count=int(resume_face_ids.shape[0]),
        )
    else:
        if str(config.root_init_method) == "fps":
            surface_roots = initialize_surface_roots_fps(
                mesh,
                config.root_count,
                candidate_multiplier=config.candidate_multiplier,
                seed=config.seed,
                fps_device=device,
            )
        elif str(config.root_init_method) == "stratified":
            surface_roots = initialize_surface_roots_stratified(
                mesh,
                config.root_count,
                seed=config.seed,
            )
        else:
            raise RuntimeError(f"unknown root_init_method: {config.root_init_method}")
    root_report = validate_surface_roots(mesh, surface_roots)
    root_report["root_init_method"] = str(config.root_init_method)
    if resume_model_state is not None:
        root_report["source"] = "resume_checkpoint"
    (output_dir / "root_init_report.json").write_text(json.dumps(root_report, indent=2) + "\n", encoding="utf-8")
    clean_flow_target_path = resolve_project_path(config.clean_flow_target) if config.clean_flow_target else None

    guide_surface_roots = None
    guide_region_ids = None
    if resume_model_state is not None and "guide_face_ids" in resume_model_state and "guide_barycentric" in resume_model_state:
        guide_face_ids = resume_model_state["guide_face_ids"].detach().cpu().numpy().astype(np.int64)
        guide_barycentric = resume_model_state["guide_barycentric"].detach().cpu().numpy().astype(np.float32)
        guide_surface_roots = SurfaceRoots(
            points=barycentric_to_points(mesh.vertices, mesh.faces, guide_face_ids, guide_barycentric),
            face_ids=guide_face_ids,
            barycentric=guide_barycentric,
            selected_candidate_ids=np.arange(int(guide_face_ids.shape[0]), dtype=np.int64),
            candidate_count=int(guide_face_ids.shape[0]),
        )
        guide_report = validate_surface_roots(mesh, guide_surface_roots)
        guide_report["source"] = "resume_checkpoint"
        (output_dir / "guide_root_init_report.json").write_text(json.dumps(guide_report, indent=2) + "\n", encoding="utf-8")
    elif bool(config.guide_roots_from_clean_flow):
        if clean_flow_target_path is None:
            raise RuntimeError("--guide-roots-from-clean-flow requires --clean-flow-target")
        clean_flow_np = np.load(clean_flow_target_path)
        if "face_ids" not in clean_flow_np or "barycentric" not in clean_flow_np:
            raise RuntimeError(
                "clean-flow target cannot provide guide roots: missing face_ids/barycentric "
                f"in {clean_flow_target_path}"
            )
        guide_face_ids = clean_flow_np["face_ids"].astype(np.int64)
        guide_barycentric = clean_flow_np["barycentric"].astype(np.float32)
        if "root_file_region_ids" in clean_flow_np:
            guide_region_ids = clean_flow_np["root_file_region_ids"].astype(np.int64)
            if guide_region_ids.shape[0] != guide_face_ids.shape[0]:
                raise RuntimeError(
                    f"clean-flow root_file_region_ids length mismatch: {guide_region_ids.shape[0]} != {guide_face_ids.shape[0]}"
                )
        else:
            guide_region_ids = np.zeros((int(guide_face_ids.shape[0]),), dtype=np.int64)
        guide_surface_roots = SurfaceRoots(
            points=barycentric_to_points(mesh.vertices, mesh.faces, guide_face_ids, guide_barycentric),
            face_ids=guide_face_ids,
            barycentric=guide_barycentric,
            selected_candidate_ids=np.arange(int(guide_face_ids.shape[0]), dtype=np.int64),
            candidate_count=int(guide_face_ids.shape[0]),
        )
        guide_report = validate_surface_roots(mesh, guide_surface_roots)
        guide_report["source"] = "clean_flow_target"
        guide_report["clean_flow_target"] = str(clean_flow_target_path)
        guide_report["configured_guide_root_count"] = int(config.guide_root_count)
        unique_regions, region_counts = np.unique(guide_region_ids, return_counts=True)
        guide_report["region_counts"] = {
            str(int(region)): int(count) for region, count in zip(unique_regions, region_counts)
        }
        (output_dir / "guide_root_init_report.json").write_text(json.dumps(guide_report, indent=2) + "\n", encoding="utf-8")
    elif int(config.guide_root_count) > 0:
        guide_surface_roots = initialize_surface_roots_fps(
            mesh,
            int(config.guide_root_count),
            candidate_multiplier=float(config.guide_candidate_multiplier),
            seed=config.seed + 17,
            fps_device=device,
        )
        guide_report = validate_surface_roots(mesh, guide_surface_roots)
        (output_dir / "guide_root_init_report.json").write_text(json.dumps(guide_report, indent=2) + "\n", encoding="utf-8")

    normals = face_normals_np(mesh)
    face_tangents = None
    if config.face_tangent_field:
        tangent_path = resolve_project_path(config.face_tangent_field)
        loaded_tangents = np.load(tangent_path).astype(np.float32)
        if loaded_tangents.shape != (mesh.face_count, 3):
            raise RuntimeError(
                f"face tangent field shape mismatch: {loaded_tangents.shape} != {(mesh.face_count, 3)}"
            )
        tangent_norm = np.linalg.norm(loaded_tangents, axis=-1)
        normal_dot = np.abs((loaded_tangents * normals).sum(axis=-1))
        face_tangents = loaded_tangents / np.maximum(tangent_norm[:, None], EPS)
        (output_dir / "face_tangent_field_report.json").write_text(
            json.dumps(
                {
                    "path": str(tangent_path),
                    "shape": list(loaded_tangents.shape),
                    "norm_min": float(tangent_norm.min(initial=0.0)),
                    "norm_mean": float(tangent_norm.mean()),
                    "norm_max": float(tangent_norm.max(initial=0.0)),
                    "abs_dot_normal_mean": float(normal_dot.mean()),
                    "abs_dot_normal_max": float(normal_dot.max(initial=0.0)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    secondary_guide_face_ids = None
    secondary_guide_barycentric = None
    secondary_guide_parent_ids = None
    if resume_model_state is not None and config.geometry_residual_domain == "secondary_guide":
        required_secondary_state = (
            "secondary_guide_face_ids",
            "secondary_guide_barycentric",
            "secondary_guide_parent_ids",
        )
        missing_secondary_state = [
            name for name in required_secondary_state if name not in resume_model_state
        ]
        if missing_secondary_state:
            raise RuntimeError(
                "secondary-guide checkpoint is missing persistent topology: "
                + ", ".join(missing_secondary_state)
            )
        secondary_guide_face_ids = (
            resume_model_state["secondary_guide_face_ids"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        secondary_guide_barycentric = (
            resume_model_state["secondary_guide_barycentric"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        secondary_guide_parent_ids = (
            resume_model_state["secondary_guide_parent_ids"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

    model = WhiteTigerStage1Model(
        mesh,
        normals,
        face_tangents,
        surface_roots.face_ids,
        surface_roots.barycentric,
        dense_groom_ranges(),
        device,
        init_scale=config.init_mesh_scale,
        init_translation=config.init_mesh_translation,
        init_groom_length=config.init_groom_length,
        max_child_count=config.child_count,
        local_child_color_support=config.local_child_color_support,
        local_child_color_scale=config.local_child_color_scale,
        gaussian_rgb_residual_support=config.gaussian_rgb_residual_support,
        gaussian_rgb_residual_control_points=config.gaussian_rgb_residual_control_points,
        gaussian_rgb_residual_scale=config.gaussian_rgb_residual_scale,
        guide_view_sh_support=config.guide_view_sh_support,
        guide_view_sh_scale=config.guide_view_sh_scale,
        view_gate_geometry_support=config.view_gate_geometry_support,
        view_gate_length_confidence_support=config.view_gate_length_confidence_support,
        guide_face_ids=guide_surface_roots.face_ids if guide_surface_roots is not None else None,
        guide_barycentric=guide_surface_roots.barycentric if guide_surface_roots is not None else None,
        guide_region_ids=guide_region_ids,
        guide_interpolation_k=config.guide_interpolation_k,
        geometry_residual_domain=config.geometry_residual_domain,
        secondary_guide_face_ids=secondary_guide_face_ids,
        secondary_guide_barycentric=secondary_guide_barycentric,
        secondary_guide_parent_ids=secondary_guide_parent_ids,
        secondary_guide_interpolation_k=config.secondary_guide_interpolation_k,
        render_geometry_parameterization=config.render_geometry_parameterization,
        guide_length_residual_scale=config.guide_length_residual_scale,
        guide_direction_residual_scale=config.guide_direction_residual_scale,
        guide_width_residual_scale=config.guide_width_residual_scale,
        guide_child_radius_residual_scale=config.guide_child_radius_residual_scale,
        guide_clump_residual_scale=config.guide_clump_residual_scale,
        guide_curl_residual_scale=config.guide_curl_residual_scale,
        shape_curl_scale=config.shape_curl_scale,
    )
    if config.geometry_residual_domain == "secondary_guide":
        if resume_model_state is None:
            if guide_surface_roots is None:
                raise RuntimeError(
                    "secondary-guide initialization requires primary surface guides"
                )
            secondary_roots = initialize_parent_conditioned_secondary_roots(
                mesh,
                guide_surface_roots,
                model.guide_surface_interpolator(),
                int(config.secondary_guide_root_count),
                candidate_multiplier=float(config.secondary_guide_candidate_multiplier),
                seed=int(config.seed) + 29,
                device=device,
            )
            support_report = model.attach_secondary_guides(
                secondary_roots.roots.face_ids,
                secondary_roots.roots.barycentric,
                secondary_roots.parent_ids,
            )
            secondary_report = {
                **secondary_roots.report,
                "source": "parent_conditioned_local_fps",
                "support": support_report,
            }
        else:
            secondary_report = {
                "source": "resume_checkpoint",
                "secondary_root_count": int(model.secondary_guide_points_local.shape[0]),
                "support": dict(model._secondary_support_report),
            }
        if int(model.secondary_guide_points_local.shape[0]) != int(
            config.secondary_guide_root_count
        ):
            raise RuntimeError(
                "secondary guide count does not match config: "
                f"{int(model.secondary_guide_points_local.shape[0])} != "
                f"{int(config.secondary_guide_root_count)}"
            )
        (output_dir / "secondary_guide_init_report.json").write_text(
            json.dumps(secondary_report, indent=2) + "\n",
            encoding="utf-8",
        )
        setup_progress(
            "secondary_guide_init_done",
            secondary_guide_root_count=int(model.secondary_guide_points_local.shape[0]),
        )
    viewmats, ks = load_camera_tensors(data_root, device)
    width, height = config.expected_width, config.expected_height

    if resume_model_state is None:
        setup_progress("projected_init_main_start")
        init_report = initialize_groom_from_projections(
            model,
            image_paths,
            mask_paths,
            viewmats,
            ks,
            report.train_indices,
            width,
            height,
            config,
            device,
        )
        (output_dir / "projected_init_report.json").write_text(json.dumps(init_report, indent=2) + "\n", encoding="utf-8")
        setup_progress("projected_init_main_done", observed_roots=int(init_report.get("projected_init_observed_roots", 0)))
        if config.clean_flow_target:
            setup_progress("clean_flow_load_start", path=str(clean_flow_target_path))
            clean_flow_targets = load_clean_flow_targets(clean_flow_target_path, device=device)
            setup_progress("clean_flow_load_done", target_count=int(clean_flow_targets.points.shape[0]))
            setup_progress("clean_flow_init_start")
            clean_flow_report = initialize_groom_from_clean_flow(model, clean_flow_targets, config)
        else:
            clean_flow_report = {"clean_flow_enabled": 0}
        (output_dir / "clean_flow_init_report.json").write_text(
            json.dumps(clean_flow_report, indent=2) + "\n",
            encoding="utf-8",
        )
        setup_progress(
            "clean_flow_init_done",
            root_init_count=int(clean_flow_report.get("clean_flow_root_init_count", 0)),
            guide_init_count=int(clean_flow_report.get("clean_flow_guide_init_count", 0)),
        )
        if model.guide_enabled():
            if not (
                bool(config.guide_roots_from_clean_flow)
                and bool(config.clean_flow_target)
                and bool(config.clean_flow_init)
            ):
                raise RuntimeError(
                    "guide roots require direct clean-flow initialization; "
                    "render-to-guide reverse initialization was deleted"
                )
            guide_init_report = {
                "guide_init_enabled": 1,
                "guide_init_count": int(model.guide_points_local.shape[0]),
                "source": "direct_clean_flow_3d",
            }
        else:
            guide_init_report = {"guide_init_enabled": 0}
        (output_dir / "guide_control_init_report.json").write_text(
            json.dumps(guide_init_report, indent=2) + "\n",
            encoding="utf-8",
        )
        setup_progress("guide_control_init_done", guide_count=int(guide_init_report.get("guide_init_count", 0)))
    else:
        (output_dir / "projected_init_report.json").write_text(
            json.dumps({"skipped_for_resume": 1}, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "clean_flow_init_report.json").write_text(
            json.dumps({"skipped_for_resume": 1}, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "guide_control_init_report.json").write_text(
            json.dumps({"skipped_for_resume": 1}, indent=2) + "\n",
            encoding="utf-8",
        )

    start_iteration = 0
    if config.resume_checkpoint:
        checkpoint = resume_checkpoint
        if checkpoint is None:
            checkpoint_path = resolve_project_path(config.resume_checkpoint)
            checkpoint = load_training_checkpoint(checkpoint_path)
        require_current_checkpoint_version(checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)
        start_iteration = int(checkpoint.get("iteration", 0))

    if bool(config.guide_view_sh_support):
        if clean_flow_target_path is None:
            raise RuntimeError("guide-view SH requires --clean-flow-target")
        guide_view_sh_confidence_report = initialize_guide_view_sh_confidence(
            model,
            clean_flow_target_path,
        )
    else:
        guide_view_sh_confidence_report = {"guide_view_sh_support": 0}
    (output_dir / "guide_view_sh_confidence_report.json").write_text(
        json.dumps(guide_view_sh_confidence_report, indent=2) + "\n",
        encoding="utf-8",
    )

    setup_progress("root_graph_start", root_count=int(model.face_ids.shape[0]))
    graph_edges, graph_report = rebuild_graph_edges(
        model,
        mode=config.smooth_graph_mode,
        k=config.smooth_graph_k,
    )
    setup_progress("root_graph_done", **graph_report)
    setup_progress("guide_graph_start", guide_root_count=int(model.guide_face_ids.shape[0]) if model.guide_enabled() else 0)
    guide_graph_edges, guide_graph_report = build_guide_graph_edges(
        model,
        mode=config.smooth_graph_mode,
        k=config.guide_interpolation_k,
    )
    guide_source_area_weights = None
    if model.guide_enabled() and (
        float(config.guide_support_gauge_weight) > 0.0
        or float(config.clean_flow_guide_length_anchor_weight) > 0.0
    ):
        guide_source_area_weights = model.guide_surface_smoothing_graph(
            config.guide_interpolation_k
        ).source_area_weights
    setup_progress("guide_graph_done", **guide_graph_report)
    if model.secondary_guides_enabled():
        setup_progress(
            "secondary_graph_start",
            secondary_guide_root_count=int(model.secondary_guide_points_local.shape[0]),
        )
        secondary_graph_started = time.perf_counter()
        geometry_graph_edges = model.secondary_surface_smoothing_edges(
            config.secondary_guide_smooth_k
        )
        geometry_graph_report = {
            "mode": "secondary_parent_conditioned",
            "root_count": int(model.secondary_guide_points_local.shape[0]),
            "neighbor_count": int(config.secondary_guide_smooth_k),
            "edge_count": int(geometry_graph_edges.shape[0]),
            "build_seconds": float(time.perf_counter() - secondary_graph_started),
        }
        setup_progress("secondary_graph_done", **geometry_graph_report)
    else:
        geometry_graph_edges = graph_edges
        geometry_graph_report = dict(graph_report)
    face_adjacency_started = time.perf_counter()
    face_adjacency_index = FaceAdjacencyIndex.from_faces(model.faces)
    setup_progress(
        "lifecycle_face_adjacency_done",
        face_count=int(face_adjacency_index.face_count),
        build_seconds=float(time.perf_counter() - face_adjacency_started),
    )
    (output_dir / "root_graph.json").write_text(
        json.dumps(
            {
                "render": graph_report,
                "guide": guide_graph_report,
                "geometry": geometry_graph_report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metric_computer = MetricComputer(compute_lpips=config.compute_lpips).to(device)
    mesh_depth_ctx = None
    if config.mesh_depth_clipping or config.mesh_backing_compositing:
        import nvdiffrast.torch as dr

        mesh_depth_ctx = dr.RasterizeCudaContext(device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    train_indices = parse_index_override(config.train_views, report.train_indices)
    test_indices = parse_index_override(config.test_views, report.test_indices)
    if bool(config.view_gated_ownership_support):
        if clean_flow_target_path is None:
            raise RuntimeError("view-gated ownership requires --clean-flow-target")
        view_gate_report = initialize_view_gate(
            model,
            clean_flow_target_path,
            config,
            train_indices,
        )
    else:
        view_gate_report = {
            "view_gated_ownership_support": 0,
            "view_gate_geometry_support": 0,
            "view_gate_length_confidence_support": 0,
        }
    (output_dir / "view_gate_report.json").write_text(
        json.dumps(view_gate_report, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint_rng_state = resume_checkpoint.get("rng_state") if resume_checkpoint is not None else None
    if start_iteration > 0 and len(train_indices) > 0 and checkpoint_rng_state is None:
        torch.randint(len(train_indices), (int(start_iteration),), generator=generator)
    optimizer = make_stage1_optimizer(model, config)
    if config.resume_checkpoint and config.resume_optimizer:
        if resume_checkpoint is None:
            raise RuntimeError(
                "resume_optimizer=True requires a loaded resume checkpoint"
            )
        require_checkpoint_optimizer_state(resume_checkpoint)
        require_checkpoint_optimizer_param_names(
            resume_checkpoint,
            model,
            config,
        )
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        optimizer_state_to_device(optimizer, device)
        setup_progress(
            "optimizer_resume_done",
            checkpoint_iteration=int(start_iteration),
            optimizer_state_entries=int(len(optimizer.state)),
        )
    if checkpoint_rng_state is not None:
        restore_training_rng_state(checkpoint_rng_state, generator)
        setup_progress("rng_resume_done", checkpoint_iteration=int(start_iteration))
    root_accum = (
        RootStatsWindow(int(model.face_ids.shape[0]), device)
        if lifecycle_statistics_active(
            config,
            start_iteration + 1,
            guide_enabled=model.guide_enabled(),
        )
        else None
    )
    previous_lifecycle_stats_active = root_accum is not None
    lifecycle_history = restored_lifecycle_history(
        resume_checkpoint,
        start_iteration=int(start_iteration),
    )
    (
        strand_crossing_active_set_cpu,
        strand_crossing_active_set_torch,
        strand_crossing_last_refresh_iteration,
        strand_crossing_history,
    ) = restore_strand_crossing_state(
        config,
        resume_checkpoint,
        root_count=int(model.face_ids.shape[0]),
        device=device,
    )
    if int(strand_crossing_last_refresh_iteration) > int(start_iteration):
        raise RuntimeError(
            "strand-crossing refresh iteration is newer than the resumed training "
            f"iteration: {strand_crossing_last_refresh_iteration} > {start_iteration}"
        )
    setup_progress(
        "strand_crossing_state_ready",
        enabled=bool(config.strand_crossing_support),
        active_pair_count=(
            int(strand_crossing_active_set_cpu.pair_count)
            if strand_crossing_active_set_cpu is not None
            else 0
        ),
        last_refresh_iteration=int(strand_crossing_last_refresh_iteration),
        history_count=int(len(strand_crossing_history)),
    )
    stage_save_iters = parse_iteration_set(config.stage_save_iters)
    if float(config.gpu_memory_limit_gb) > 0.0 and int(config.gpu_memory_check_interval) <= 0:
        raise RuntimeError("--gpu-memory-limit-gb requires --gpu-memory-check-interval > 0")

    log_path = output_dir / "metrics.jsonl"
    start = time.time()
    needs_rgb_flow_loss = float(config.rgb_flow_weight) > 0.0 or float(config.rgb_flow_detail_weight) > 0.0
    rgb_flow_target_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def progress_event(stage: str, **extra: object) -> None:
        payload = {"progress": stage, **extra}
        print(json.dumps(payload), flush=True)

    progress_event(
        "setup_complete",
        start_iteration=int(start_iteration),
        target_iteration=int(config.iterations),
        root_count=int(model.face_ids.shape[0]),
        guide_root_count=int(model.guide_face_ids.shape[0]) if model.guide_enabled() else 0,
        graph_edges=int(graph_edges.shape[0]),
        guide_graph_edges=int(guide_graph_edges.shape[0]),
        activation_checkpointing=memory_constrained_activation_checkpointing(device),
        lifecycle_statistics_active=bool(previous_lifecycle_stats_active),
        mesh_no_penetration_support=bool(config.mesh_no_penetration_support),
        mesh_no_penetration_sdf_sha256=(
            mesh_no_penetration_report["sdf_sha256"]
            if mesh_no_penetration_report is not None
            else None
        ),
        mesh_no_penetration_root_batch=int(
            config.mesh_no_penetration_root_batch
        ),
        strand_crossing_support=bool(config.strand_crossing_support),
        strand_crossing_active_pair_count=(
            int(strand_crossing_active_set_cpu.pair_count)
            if strand_crossing_active_set_cpu is not None
            else 0
        ),
        strand_crossing_last_refresh_iteration=int(
            strand_crossing_last_refresh_iteration
        ),
        pytorch_alloc_conf=os.environ.get("PYTORCH_ALLOC_CONF", ""),
        device_memory_gb=float(torch.cuda.get_device_properties(device).total_memory / 1024**3) if device.type == "cuda" else 0.0,
    )
    enforce_cuda_memory_guard(config, device, iteration=int(start_iteration), stage="setup_complete", progress_event=progress_event)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with log_path.open("a", encoding="utf-8") as log:
        for iteration in range(start_iteration + 1, config.iterations + 1):
            memory_guard_due = (
                float(config.gpu_memory_limit_gb) > 0.0
                and int(config.gpu_memory_check_interval) > 0
                and (iteration == start_iteration + 1 or iteration % int(config.gpu_memory_check_interval) == 0)
            )
            if memory_guard_due:
                enforce_cuda_memory_guard(config, device, iteration=int(iteration), stage="iteration_start", progress_event=progress_event)
            model.guide_residual_multiplier = guide_residual_multiplier_for_iteration(config, iteration)
            model.guide_coverage_residual_multiplier = guide_coverage_residual_multiplier_for_iteration(config, iteration)
            model.shape_detail_multiplier = shape_detail_multiplier_for_iteration(config, iteration)
            model.secondary_shape_residual_multiplier = (
                secondary_shape_residual_multiplier_for_iteration(config, iteration)
            )
            model.gaussian_rgb_residual_multiplier = (
                gaussian_rgb_residual_multiplier_for_iteration(config, iteration)
            )
            should_densify = (
                iteration >= config.densify_warmup
                and iteration <= config.densify_until
                and config.densify_interval > 0
                and (iteration - config.densify_warmup) % config.densify_interval == 0
            )
            should_prune = (
                iteration >= config.prune_start
                and config.prune_interval > 0
                and (iteration - config.prune_start) % config.prune_interval == 0
            )
            should_guide_densify = (
                model.guide_enabled()
                and int(config.guide_densify_interval) > 0
                and int(config.guide_densify_start) > 0
                and iteration >= int(config.guide_densify_start)
                and iteration <= int(config.guide_densify_until)
                and (iteration - int(config.guide_densify_start)) % int(config.guide_densify_interval) == 0
            )
            should_refresh_strand_crossing = (
                bool(config.strand_crossing_support)
                and iteration > int(config.densify_until)
                and (
                    strand_crossing_active_set_cpu is None
                    or iteration - int(strand_crossing_last_refresh_iteration)
                    >= int(config.strand_crossing_refresh_interval)
                )
            )
            lifecycle_stats_active = lifecycle_statistics_active(
                config,
                iteration,
                guide_enabled=model.guide_enabled(),
            )
            if lifecycle_stats_active != previous_lifecycle_stats_active:
                progress_event(
                    "lifecycle_statistics_state",
                    iteration=int(iteration),
                    active=bool(lifecycle_stats_active),
                    render_densify_until=int(config.densify_until),
                    guide_densify_until=int(config.guide_densify_until),
                )
                previous_lifecycle_stats_active = lifecycle_stats_active
            idx = int(train_indices[int(torch.randint(len(train_indices), (1,), generator=generator))])
            trace_iteration = iteration == start_iteration + 1 or iteration % 20 == 0
            if trace_iteration or iteration % 20 == 0:
                progress_event(
                    "iteration_start",
                    iteration=int(iteration),
                    view_index=int(idx),
                    elapsed_sec=float(time.time() - start),
                )
            target = load_image(image_paths[idx], device)
            mask = load_mask(mask_paths[idx], device)
            flow_loss_mask_vis = torch.zeros((*target.shape[:2], 1), device=device, dtype=target.dtype)
            target_flow_vis = None
            mesh_color = sample_backing_color(config, device, train=True)
            scene_bg = scene_background_color(config, device)
            if trace_iteration:
                progress_event("before_mesh_depth", iteration=int(iteration))
            mesh_depth = render_model_mesh_depth(model, viewmats[idx], ks[idx], width, height, device=device, ctx=mesh_depth_ctx)
            if trace_iteration:
                progress_event("after_mesh_depth", iteration=int(iteration))
            backing_image = make_mesh_backing_image(
                mesh_depth,
                mesh_color,
                scene_bg,
                model=model,
                viewmat=viewmats[idx],
                k=ks[idx],
                width=width,
                height=height,
                config=config,
                device=device,
                ctx=mesh_depth_ctx,
                train=True,
            )
            target_with_backing = composite_target(target, mask, backing_image)
            mesh_no_penetration_root_indices = (
                cyclic_strand_indices(
                    int(model.face_ids.shape[0]),
                    int(config.mesh_no_penetration_root_batch),
                    int(iteration),
                    device=device,
                )
                if mesh_no_penetration_field is not None
                else None
            )

            try:
                if trace_iteration:
                    progress_event("before_render_view", iteration=int(iteration))
                pred, alpha, gaussians, roots_local_for_grad, render_stats, render_info = render_view(
                    model,
                    viewmats[idx],
                    ks[idx],
                    width,
                    height,
                    config,
                    background=mesh_color,
                    mesh_depth=mesh_depth,
                    backing_image=backing_image,
                    retain_lifecycle_grad=lifecycle_stats_active,
                    mesh_no_penetration_field=mesh_no_penetration_field,
                    mesh_no_penetration_root_indices=(
                        mesh_no_penetration_root_indices
                    ),
                    strand_crossing_active_set=strand_crossing_active_set_torch,
                    capture_strand_crossing_snapshot=should_refresh_strand_crossing,
                    view_index=idx,
                )
                if trace_iteration:
                    progress_event(
                        "after_render_view",
                        iteration=int(iteration),
                        gaussian_count=int(gaussians.means.shape[0]),
                        kept_count=int(render_stats.get("kept_gaussian_count", gaussians.means.shape[0])),
                    )
                if memory_guard_due:
                    enforce_cuda_memory_guard(config, device, iteration=int(iteration), stage="after_render_view", progress_event=progress_event)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"train render failed: iteration={iteration}, view_index={idx}, root_count={int(model.face_ids.shape[0])}"
                ) from exc
            mesh_no_penetration_depth = render_info[
                "mesh_no_penetration_depth"
            ]
            if mesh_no_penetration_field is not None:
                if mesh_no_penetration_root_indices is None:
                    raise RuntimeError(
                        "mesh no-penetration field has no sampled root indices"
                    )
                expected_shape = (
                    int(mesh_no_penetration_root_indices.numel()),
                    int(config.samples) - 1,
                )
                if tuple(mesh_no_penetration_depth.shape) != expected_shape:
                    raise RuntimeError(
                        "mesh no-penetration depth shape mismatch: "
                        f"expected {expected_shape}, got "
                        f"{tuple(mesh_no_penetration_depth.shape)}"
                    )
                mesh_no_penetration_loss = mesh_no_penetration_depth.mean()
            else:
                if mesh_no_penetration_depth.numel() != 0:
                    raise RuntimeError(
                        "disabled mesh no-penetration produced depth samples"
                    )
                mesh_no_penetration_loss = model.groom.length_raw.sum() * 0.0
            strand_crossing_loss = render_info["strand_crossing_loss"]
            strand_crossing_stats = render_info["strand_crossing_stats"]
            strand_crossing_snapshot = render_info["strand_crossing_snapshot"]
            if should_refresh_strand_crossing:
                if not isinstance(strand_crossing_snapshot, GaussianSegmentSnapshot):
                    raise RuntimeError(
                        "strand-crossing refresh was requested without a Gaussian snapshot"
                    )
            elif strand_crossing_snapshot is not None:
                raise RuntimeError(
                    "strand-crossing snapshot was captured outside a refresh iteration"
                )
            fixed_bg = scene_background_color(config, device).view(1, 1, 3)
            pred_fixed = render_info["raw_fur_image"] + (1.0 - alpha) * fixed_bg
            target_fixed = composite_target(target, mask, fixed_bg)
            edge_loss_weight = loss_mask_edge_weight(mask, int(config.loss_mask_edge_kernel))
            flow_supervision_mask = (mask * mask_edge_confidence(mask, int(config.loss_mask_edge_kernel))).detach().clamp(0.0, 1.0)
            densify_supervision_mask = (mask * edge_loss_weight).detach().clamp(0.0, 1.0)
            rgb_weight = ((0.25 + 1.75 * mask) * edge_loss_weight).detach()
            fixed_rgb_loss = (torch.abs(pred_fixed - target_fixed) * rgb_weight).sum() / torch.clamp(rgb_weight.sum() * 3.0, min=1.0)
            random_backing_loss = (
                torch.abs((pred - pred_fixed) - (target_with_backing - target_fixed)) * rgb_weight
            ).sum() / torch.clamp(rgb_weight.sum() * 3.0, min=1.0)
            rgb_loss = fixed_rgb_loss + float(config.random_backing_loss_weight) * random_backing_loss
            mask_loss = (torch.abs(alpha - mask) * edge_loss_weight).sum() / torch.clamp(edge_loss_weight.sum(), min=1.0)
            residual_per_root = None
            residual_image = None
            if float(config.densify_residual_weight) > 0.0:
                needs_residual_per_root = should_densify
                if needs_residual_per_root:
                    residual_image = densification_residual_image(pred_fixed, target_fixed, alpha, densify_supervision_mask, config)
                if str(config.densify_residual_mode) == "pixel_to_root":
                    if needs_residual_per_root and trace_iteration:
                        progress_event("before_pixel_to_root_evidence", iteration=int(iteration))
                    if needs_residual_per_root:
                        if residual_image is None:
                            raise RuntimeError("pixel_to_root evidence requested without residual_image")
                        residual_per_root, _, _ = pixel_to_root_evidence(
                            model,
                            roots_local_for_grad,
                            residual_image,
                            viewmats[idx],
                            ks[idx],
                            mesh_depth,
                            config,
                        )
                    if needs_residual_per_root and trace_iteration:
                        progress_event("after_pixel_to_root_evidence", iteration=int(iteration))
                else:
                    if needs_residual_per_root:
                        if residual_image is None:
                            raise RuntimeError("projected residual requested without residual_image")
                        residual_per_root = root_projected_residual(
                            model,
                            roots_local_for_grad,
                            residual_image,
                            viewmats[idx],
                            ks[idx],
                            mesh_depth,
                            config,
                        )
                if residual_per_root is not None:
                    residual_per_root = residual_per_root * float(config.densify_residual_weight)
                    if model.view_gate_enabled():
                        # R072: a view may only contribute densification
                        # evidence for the roots it is trusted on. Visibility
                        # and opacity history stay ungated, because a root is
                        # genuinely visible regardless of direction trust.
                        with torch.no_grad():
                            evidence_gate = model.view_gate_at_render_roots(
                                roots_local_for_grad,
                                int(idx),
                            )
                        residual_per_root = (
                            residual_per_root.reshape(-1, 1) * evidence_gate
                        )
            if needs_rgb_flow_loss:
                if trace_iteration:
                    progress_event("before_rgb_flow_loss", iteration=int(iteration))
                if idx not in rgb_flow_target_cache:
                    with torch.no_grad():
                        target_flow_cached, target_conf_cached = image_structure_flow(target_fixed, flow_supervision_mask)
                    rgb_flow_target_cache[idx] = (
                        target_flow_cached.detach(),
                        target_conf_cached.detach(),
                    )
                target_flow_cached, target_conf_cached = rgb_flow_target_cache[idx]
                rgb_flow_target_valid = (target_conf_cached >= float(config.rgb_flow_min_confidence)).to(dtype=target_conf_cached.dtype)
                flow_loss_mask_vis = (target_conf_cached.detach() * rgb_flow_target_valid.detach()).clamp(0.0, 1.0)
                target_flow_vis = target_flow_cached.detach()
                rgb_flow_loss, rgb_flow_detail_loss, rgb_flow_stats = image_structure_flow_losses(
                    pred_fixed,
                    target_fixed,
                    flow_supervision_mask,
                    min_confidence=float(config.rgb_flow_min_confidence),
                    target_flow=target_flow_cached,
                    target_confidence=target_conf_cached,
                )
                if trace_iteration:
                    progress_event("after_rgb_flow_loss", iteration=int(iteration))
            else:
                rgb_flow_loss = pred_fixed.sum() * 0.0
                rgb_flow_detail_loss = pred_fixed.sum() * 0.0
                rgb_flow_stats = {
                    "rgb_flow_loss": 0.0,
                    "rgb_flow_detail_loss": 0.0,
                    "rgb_flow_weight_sum": 0.0,
                    "rgb_flow_valid_pixels": 0,
                }
            if trace_iteration:
                progress_event("before_regularizers", iteration=int(iteration))
            _, normals_now, roots_local = model.roots_and_normals()
            tangents_now, bitangents_now = model.tangent_frames(normals_now)
            if model.secondary_guides_enabled():
                geometry_normals, geometry_tangents, geometry_bitangents = (
                    model.tangent_frames_for_face_ids(model.secondary_guide_face_ids)
                )
                geometry_effective_groom = model.secondary_effective_groom()
                geometry_confidence = model.secondary_clean_flow_confidence()
            else:
                geometry_normals = normals_now
                geometry_tangents = tangents_now
                geometry_bitangents = bitangents_now
                geometry_effective_groom = model.apply_guide_controls(
                    model.groom.decode(),
                    roots_local,
                )
                geometry_confidence = model.root_observation_confidence
            zero_loss = model.groom.length_raw.sum() * 0.0
            smooth_loss = root_graph_smoothness(
                model.groom,
                graph_edges,
                model.root_observation_confidence,
                normals=normals_now,
                tangents=tangents_now,
                bitangents=bitangents_now,
                smooth_field_metric=config.smooth_field_metric,
                include_geometry=not model.uses_zero_centered_geometry(),
                appearance_only=model.secondary_guides_enabled(),
            )
            geometry_residual_smooth_loss = render_geometry_residual_graph_smoothness(
                model,
                geometry_graph_edges,
                geometry_normals,
                geometry_tangents,
                geometry_bitangents,
                geometry_confidence,
            )
            smooth_loss = (
                smooth_loss
                + float(config.geometry_residual_smooth_scale)
                * geometry_residual_smooth_loss
            )
            guide_smooth_loss = guide_root_graph_smoothness(
                model,
                guide_graph_edges,
                smooth_field_metric=config.smooth_field_metric,
                guide_length_smooth_mode=config.guide_length_smooth_mode,
                smooth_graph_k=config.guide_interpolation_k,
            )
            effective_smooth_loss = effective_groom_graph_smoothness(
                geometry_effective_groom,
                geometry_graph_edges,
                geometry_normals,
                geometry_tangents,
                geometry_bitangents,
                model.groom.ranges,
                geometry_confidence,
                smooth_field_metric=config.smooth_field_metric,
            )
            guide_prior_loss = guide_interpolation_regularization_losses(model, config)
            guide_support_gauge_loss = zero_loss
            guide_support_gauge_length_collapse = zero_loss
            guide_support_gauge_slenderness_expansion = zero_loss
            if (
                model.guide_enabled()
                and float(config.guide_support_gauge_weight) > 0.0
            ):
                guide_support_gauge_terms = guide_support_gauge(
                    model.guide_length_raw,
                    model.guide_root_width_raw,
                    model.guide_clean_flow_length_confidence,
                    source_area_weights=guide_source_area_weights,
                )
                guide_support_gauge_loss = guide_support_gauge_terms.total
                guide_support_gauge_length_collapse = (
                    guide_support_gauge_terms.length_collapse
                )
                guide_support_gauge_slenderness_expansion = (
                    guide_support_gauge_terms.slenderness_expansion
                )
            clean_pred_direction = groom_direction_3d(
                geometry_effective_groom,
                geometry_normals,
                geometry_tangents,
                geometry_bitangents,
            )
            clean_flow_smooth_loss = clean_flow_smoothness_loss(
                clean_pred_direction,
                geometry_graph_edges,
                geometry_confidence,
                normals=(
                    geometry_normals
                    if smooth_metric_uses_transport(config.smooth_field_metric)
                    else None
                ),
            )
            if trace_iteration:
                progress_event("after_regularizers", iteration=int(iteration))
            guide_clean_pred_direction = model.guide_direction_world()
            if guide_clean_pred_direction is not None:
                guide_clean_flow_loss = clean_flow_anchor_loss(
                    guide_clean_pred_direction,
                    model.guide_clean_flow_direction_target,
                    model.guide_clean_flow_anchor_confidence,
                    min_confidence=float(config.clean_flow_anchor_min_confidence),
                )
            else:
                guide_clean_flow_loss = model.groom.length_raw.sum() * 0.0
            if model.guide_enabled():
                guide_clean_flow_length_reliable_fraction = (
                    clean_flow_guide_length_anchor_reliable_fraction(
                        model.guide_clean_flow_length_target,
                        model.guide_clean_flow_length_confidence,
                    )
                )
                if float(config.clean_flow_guide_length_anchor_weight) > 0.0:
                    guide_clean_flow_length_loss, _ = primary_guide_length_anchor_metrics(
                        model,
                        config,
                        source_area_weights=guide_source_area_weights,
                    )
                else:
                    guide_clean_flow_length_loss = model.groom.length_raw.sum() * 0.0
            else:
                guide_clean_flow_length_loss = model.groom.length_raw.sum() * 0.0
                guide_clean_flow_length_reliable_fraction = (
                    guide_clean_flow_length_loss.detach()
                )
            weighted_guide_clean_flow_length_loss = (
                float(config.clean_flow_guide_length_anchor_weight)
                * guide_clean_flow_length_loss
                if float(config.clean_flow_guide_length_anchor_weight) > 0.0
                else zero_loss
            )
            root_move_loss = torch.mean((roots_local - model.anchor_local).square())
            loss = (
                config.rgb_weight * rgb_loss
                + config.mask_weight * mask_loss
                + config.rgb_flow_weight * rgb_flow_loss
                + config.rgb_flow_detail_weight * rgb_flow_detail_loss
                + config.smooth_weight * smooth_loss
                + config.guide_smooth_weight * guide_smooth_loss
                + config.effective_smooth_weight * effective_smooth_loss
                + config.guide_prior_weight * guide_prior_loss
                + config.guide_support_gauge_weight * guide_support_gauge_loss
                + config.clean_flow_guide_anchor_weight * guide_clean_flow_loss
                + weighted_guide_clean_flow_length_loss
                + config.clean_flow_3d_smooth_weight * clean_flow_smooth_loss
                + config.root_move_reg_weight * root_move_loss
                + config.mesh_no_penetration_weight * mesh_no_penetration_loss
                + config.strand_crossing_weight * strand_crossing_loss
            )
            weighted_rgb_flow_loss = (
                config.rgb_flow_weight * rgb_flow_loss
                + config.rgb_flow_detail_weight * rgb_flow_detail_loss
            )
            rgb_and_regularization_loss = (
                config.rgb_weight * rgb_loss
                + config.mask_weight * mask_loss
                + config.smooth_weight * smooth_loss
                + config.guide_smooth_weight * guide_smooth_loss
                + config.effective_smooth_weight * effective_smooth_loss
                + config.guide_prior_weight * guide_prior_loss
                + config.guide_support_gauge_weight * guide_support_gauge_loss
                + config.clean_flow_guide_anchor_weight * guide_clean_flow_loss
                + weighted_guide_clean_flow_length_loss
                + config.clean_flow_3d_smooth_weight * clean_flow_smooth_loss
                + config.root_move_reg_weight * root_move_loss
                + config.mesh_no_penetration_weight * mesh_no_penetration_loss
            )
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise RuntimeError(
                    "non-finite loss before backward: "
                    + json.dumps(
                        {
                            "iteration": iteration,
                            "view_index": idx,
                            "loss": float(loss.detach().cpu()) if loss.detach().numel() == 1 else None,
                            "rgb_loss": float(rgb_loss.detach().cpu()),
                            "mask_loss": float(mask_loss.detach().cpu()),
                            "smooth_loss": float(smooth_loss.detach().cpu()),
                            "guide_prior_loss": float(guide_prior_loss.detach().cpu()),
                            "guide_support_gauge_loss": float(
                                guide_support_gauge_loss.detach().cpu()
                            ),
                            "guide_clean_flow_loss": float(guide_clean_flow_loss.detach().cpu()),
                            "clean_flow_guide_length_anchor_loss": float(
                                guide_clean_flow_length_loss.detach().cpu()
                            ),
                            "mesh_no_penetration_loss": float(
                                mesh_no_penetration_loss.detach().cpu()
                            ),
                            "strand_crossing_loss": float(
                                strand_crossing_loss.detach().cpu()
                            ),
                        },
                        sort_keys=True,
                    )
                )

            optimizer.zero_grad(set_to_none=True)
            if trace_iteration:
                progress_event("before_backward", iteration=int(iteration))
            weighted_strand_crossing_loss = None
            if (
                strand_crossing_active_set_torch is not None
                and strand_crossing_active_set_torch.pair_count > 0
            ):
                weighted_strand_crossing_loss = (
                    float(config.strand_crossing_weight) * strand_crossing_loss
                )
            backward_stage1_losses(
                model,
                optimizer,
                rgb_and_regularization_loss=rgb_and_regularization_loss,
                flow_loss=weighted_rgb_flow_loss,
                exclude_color_flow_gradients=bool(
                    config.rgb_flow_exclude_color_gradients
                ),
                strand_crossing_loss=weighted_strand_crossing_loss,
            )
            if trace_iteration:
                progress_event("after_backward", iteration=int(iteration))
            if memory_guard_due:
                enforce_cuda_memory_guard(config, device, iteration=int(iteration), stage="after_backward", progress_event=progress_event)
            assert_model_gradients_finite(model, f"non-finite gradient after backward at iteration={iteration}, view_index={idx}")
            if lifecycle_stats_active:
                if root_accum is None:
                    raise RuntimeError(
                        "lifecycle statistics are active without an accumulator"
                    )
                root_accum.add(
                    root_points=roots_local_for_grad,
                    gaussians=gaussians,
                    infos=[render_info],
                    residual_per_root=residual_per_root,
                )
            guide_frozen = (
                int(config.guide_freeze_until) > 0
                and iteration <= int(config.guide_freeze_until)
            )
            guide_length_freeze_until = resolved_guide_length_freeze_until(config)
            guide_length_frozen = (
                guide_length_freeze_until > 0
                and iteration <= guide_length_freeze_until
            )
            if guide_frozen or guide_length_frozen:
                zero_guide_gradients(
                    model,
                    freeze_length=guide_length_frozen,
                    freeze_other=guide_frozen,
                )
            if (
                model.uses_zero_centered_geometry()
                and iteration <= int(config.guide_residual_unlock_start)
            ):
                zero_render_geometry_residual_gradients(model)
            shape_detail_frozen = (
                int(config.shape_detail_freeze_until) > 0
                and iteration <= int(config.shape_detail_freeze_until)
            )
            if shape_detail_frozen:
                zero_primary_shape_detail_gradients(model)
            secondary_shape_residual_frozen = (
                int(config.secondary_shape_residual_unlock_start) > 0
                and iteration <= int(config.secondary_shape_residual_unlock_start)
            )
            if secondary_shape_residual_frozen:
                zero_secondary_shape_detail_gradients(model)
            if int(config.color_freeze_until) > 0 and iteration <= int(config.color_freeze_until):
                zero_color_gradients(model)
            optimizer.step()
            if trace_iteration or iteration % 20 == 0:
                progress_event(
                    "iteration_done",
                    iteration=int(iteration),
                    elapsed_sec=float(time.time() - start),
                    root_count=int(model.face_ids.shape[0]),
                    max_memory_allocated_mb=float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0,
                )
            assert_model_parameters_finite(model, f"non-finite parameter after optimizer.step at iteration={iteration}, view_index={idx}")

            if should_densify or should_prune or should_guide_densify:
                lifecycle_started = time.perf_counter()
                lifecycle_timing: dict[str, float] = {}
                if root_accum is None:
                    raise RuntimeError(
                        "lifecycle event requested without accumulated statistics"
                    )
                stats = root_accum.to_stats()
                root_count_before = int(model.face_ids.shape[0])
                optimizer_param_names_before_structure = stage1_optimizer_param_names(model, config)
                render_optimizer_transition: OptimizerRowTransition | None = None
                guide_optimizer_transition: OptimizerRowTransition | None = None
                densify_cfg = DensifyConfig(
                    grad_threshold=float(config.densify_score_threshold) if should_densify else float("inf"),
                    visibility_threshold=1.0,
                    score_mode=str(config.lifecycle_score_mode),
                    parent_selection_mode="evidence_local_max" if should_densify else "score",
                    max_new_roots=int(config.max_splits_per_event) * int(config.split_children_per_parent),
                    children_per_parent=int(config.split_children_per_parent),
                    replace_parent=True,
                    neighbor_count=int(config.split_neighbor_count),
                    candidate_rings=int(config.split_candidate_rings),
                    candidate_face_count=int(config.split_candidate_face_count),
                    min_child_distance=float(config.split_min_child_distance),
                )
                prune_cfg = PruneConfig(
                    min_visible_count=1.0 if should_prune else -1.0,
                    min_contribution=float(config.prune_min_contribution) if should_prune else -1.0,
                    min_opacity=float(config.prune_min_opacity) if should_prune else 0.0,
                    max_prune_fraction=float(config.prune_max_fraction) if should_prune else 0.0,
                )
                selection_started = time.perf_counter()
                update = propose_structure_update(
                    model.lifecycle_state(),
                    stats,
                    densify_cfg,
                    prune_cfg,
                    vertices=model.vertices,
                    faces=model.faces,
                    face_adjacency_index=face_adjacency_index,
                )
                lifecycle_timing["render_selection_seconds"] = float(
                    time.perf_counter() - selection_started
                )
                lifecycle_timing.update(
                    {
                        f"render_{name}": float(value)
                        for name, value in update.timing.items()
                    }
                )
                if (
                    should_densify
                    and float(config.densify_min_contribution) > 0.0
                    and update.parent_indices.numel() > 0
                ):
                    contribution = stats.gaussian_contrib_sum.reshape(-1)
                    keep_parent = contribution[update.parent_indices] >= float(config.densify_min_contribution)
                    if not bool(keep_parent.all()):
                        original_parents = update.parent_indices
                        kept_parents = update.parent_indices[keep_parent]
                        child_keep = torch.isin(update.child_parent_indices, kept_parents)
                        update.parent_indices = kept_parents
                        update.child_parent_indices = update.child_parent_indices[child_keep]
                        update.new_face_ids = update.new_face_ids[child_keep]
                        update.new_barycentric = update.new_barycentric[child_keep]
                        new_prune = torch.zeros_like(update.prune_mask)
                        if should_prune:
                            new_prune |= update.prune_mask
                        new_prune[original_parents] = False
                        new_prune[kept_parents] = True
                        update.prune_mask = new_prune
                current_state = model.lifecycle_state()
                changed = update.new_barycentric.numel() > 0 or bool(update.prune_mask.any())
                lifecycle_record = {
                    "iteration": iteration,
                    "root_count_before": root_count_before,
                    "selected_parent_count": int(update.parent_indices.numel()),
                    "inserted_child_count": int(update.new_barycentric.shape[0]),
                    "prune_count": int(update.prune_mask.sum().detach().cpu()),
                    "diagnostics": {
                        "selection": lifecycle_selection_report(update.scores),
                        "spatial_selected_parents": lifecycle_spatial_report(current_state, update.parent_indices),
                        "global": lifecycle_global_report(model, stats, update.scores),
                        "selected_parents": lifecycle_subset_report(model, stats, update.scores, update.parent_indices),
                        "pruned_roots": lifecycle_subset_report(
                            model,
                            stats,
                            update.scores,
                            torch.nonzero(update.prune_mask, as_tuple=False).reshape(-1),
                        ),
                    },
                }
                guide_changed = False
                if should_guide_densify:
                    guide_update_started = time.perf_counter()
                    guide_update, guide_record = propose_guide_densify_update(
                        model,
                        stats,
                        config,
                        face_adjacency_index,
                    )
                    if guide_update is not None and guide_update.new_barycentric.numel() > 0:
                        guide_record["spatial_selected_parents"] = lifecycle_spatial_report(
                            model.guide_lifecycle_state(),
                            guide_update.parent_indices,
                        )
                        guide_optimizer_transition = optimizer_row_transition(
                            guide_update,
                            old_count=int(model.guide_face_ids.shape[0]),
                        )
                        guide_result = model.apply_guide_structure_update(guide_update)
                        guide_record.update(guide_result)
                        guide_changed = True
                    lifecycle_record["guide_densify"] = guide_record
                    lifecycle_timing["guide_update_seconds"] = float(
                        time.perf_counter() - guide_update_started
                    )
                if changed:
                    render_optimizer_transition = optimizer_row_transition(
                        update,
                        old_count=root_count_before,
                    )
                    render_update_started = time.perf_counter()
                    result = model.apply_structure_update(
                        update,
                        neighbor_count=config.smooth_graph_k,
                    )
                    lifecycle_timing["render_update_seconds"] = float(
                        time.perf_counter() - render_update_started
                    )
                    lifecycle_record.update(result)
                else:
                    lifecycle_record["root_count_after"] = root_count_before
                if changed or guide_changed:
                    graph_started = time.perf_counter()
                    graph_edges, graph_report = rebuild_graph_edges(
                        model,
                        mode=config.smooth_graph_mode,
                        k=config.smooth_graph_k,
                    )
                    guide_graph_edges, guide_graph_report = build_guide_graph_edges(
                        model,
                        mode=config.smooth_graph_mode,
                        k=config.guide_interpolation_k,
                    )
                    if model.guide_enabled() and (
                        float(config.guide_support_gauge_weight) > 0.0
                        or float(config.clean_flow_guide_length_anchor_weight) > 0.0
                    ):
                        guide_source_area_weights = model.guide_surface_smoothing_graph(
                            config.guide_interpolation_k
                        ).source_area_weights
                    if not model.secondary_guides_enabled():
                        geometry_graph_edges = graph_edges
                        geometry_graph_report = dict(graph_report)
                    lifecycle_record["smoothing_graph"] = {
                        "render": graph_report,
                        "guide": guide_graph_report,
                        "geometry": geometry_graph_report,
                    }
                    lifecycle_timing["graph_update_seconds"] = float(
                        time.perf_counter() - graph_started
                    )
                    optimizer_started = time.perf_counter()
                    optimizer, optimizer_migration = rebuild_stage1_optimizer_with_state(
                        model,
                        config,
                        optimizer,
                        optimizer_param_names_before_structure,
                        render_transition=render_optimizer_transition,
                        guide_transition=guide_optimizer_transition,
                    )
                    lifecycle_record["optimizer_state_migration"] = optimizer_migration
                    lifecycle_timing["optimizer_update_seconds"] = float(
                        time.perf_counter() - optimizer_started
                    )
                lifecycle_timing["total_seconds"] = float(
                    time.perf_counter() - lifecycle_started
                )
                lifecycle_record["timing"] = lifecycle_timing
                lifecycle_history.append(lifecycle_record)
                log.write(json.dumps({"lifecycle": lifecycle_record}) + "\n")
                log.flush()
                print(json.dumps({"lifecycle": lifecycle_record}), flush=True)
                root_accum = (
                    RootStatsWindow(int(model.face_ids.shape[0]), device)
                    if lifecycle_statistics_active(
                        config,
                        iteration + 1,
                        guide_enabled=model.guide_enabled(),
                    )
                    else None
                )
                release_cuda_cache()
                progress_event(
                    "after_lifecycle_cache_release",
                    iteration=int(iteration),
                    memory_allocated_mb=float(torch.cuda.memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0,
                    memory_reserved_mb=float(torch.cuda.memory_reserved() / (1024 * 1024)) if torch.cuda.is_available() else 0.0,
                )
                if memory_guard_due:
                    enforce_cuda_memory_guard(config, device, iteration=int(iteration), stage="after_lifecycle", progress_event=progress_event)

            if should_refresh_strand_crossing:
                crossing_refresh_started = time.perf_counter()
                discovery_workers = int(os.environ.get("NSLOTS", "1"))
                if discovery_workers <= 0:
                    raise RuntimeError("NSLOTS must be positive for crossing discovery")
                refreshed_active_set, crossing_discovery_report = (
                    discover_gaussian_segment_crossings(
                        strand_crossing_snapshot,
                        query_batch=int(config.strand_crossing_query_batch),
                        exact_pair_batch=int(
                            config.strand_crossing_exact_pair_batch
                        ),
                        workers=discovery_workers,
                    )
                )
                current_root_count = int(model.face_ids.shape[0])
                if int(crossing_discovery_report["source_root_count"]) != current_root_count:
                    raise RuntimeError(
                        "strand-crossing discovery did not cover every render root: "
                        f"{crossing_discovery_report['source_root_count']} != "
                        f"{current_root_count}"
                    )
                strand_crossing_active_set_cpu = refreshed_active_set
                strand_crossing_active_set_torch = refreshed_active_set.to_torch(
                    device
                )
                strand_crossing_last_refresh_iteration = int(iteration)
                crossing_refresh_record = {
                    "iteration": int(iteration),
                    "elapsed_seconds": float(
                        time.perf_counter() - crossing_refresh_started
                    ),
                    "workers": int(discovery_workers),
                    **crossing_discovery_report,
                }
                strand_crossing_history.append(crossing_refresh_record)
                log.write(
                    json.dumps(
                        {"strand_crossing_refresh": crossing_refresh_record}
                    )
                    + "\n"
                )
                log.flush()
                print(
                    json.dumps(
                        {"strand_crossing_refresh": crossing_refresh_record}
                    ),
                    flush=True,
                )
                render_info["strand_crossing_snapshot"] = None
                del strand_crossing_snapshot

            if iteration == 1 or iteration % config.eval_every == 0 or iteration == config.iterations:
                train_eval = evaluate(model, image_paths, mask_paths, viewmats, ks, train_indices, width, height, config, metric_computer, device, mesh_depth_ctx=mesh_depth_ctx)
                test_eval = evaluate(model, image_paths, mask_paths, viewmats, ks, test_indices, width, height, config, metric_computer, device, mesh_depth_ctx=mesh_depth_ctx)
                with torch.no_grad():
                    if model.guide_enabled():
                        eval_guide_length_anchor_reliable_fraction = (
                            clean_flow_guide_length_anchor_reliable_fraction(
                                model.guide_clean_flow_length_target,
                                model.guide_clean_flow_length_confidence,
                            )
                        )
                        if float(config.clean_flow_guide_length_anchor_weight) > 0.0:
                            eval_guide_length_anchor_loss, _ = (
                                primary_guide_length_anchor_metrics(
                                    model,
                                    config,
                                    source_area_weights=guide_source_area_weights,
                                )
                            )
                        else:
                            eval_guide_length_anchor_loss = (
                                model.groom.length_raw.sum() * 0.0
                            )
                    else:
                        eval_guide_length_anchor_loss = model.groom.length_raw.sum() * 0.0
                        eval_guide_length_anchor_reliable_fraction = (
                            eval_guide_length_anchor_loss
                        )
                for eval_metrics in (train_eval, test_eval):
                    eval_metrics[
                        "clean_flow_guide_length_anchor_loss"
                    ] = float(eval_guide_length_anchor_loss.cpu())
                    eval_metrics[
                        "clean_flow_guide_length_anchor_reliable_fraction"
                    ] = float(eval_guide_length_anchor_reliable_fraction.cpu())
                memory_payload = cuda_memory_guard_payload(device)
                record = {
                    "iteration": iteration,
                    "elapsed_sec": round(time.time() - start, 3),
                    "loss": float(loss.detach().cpu()),
                    "rgb_l1": float(rgb_loss.detach().cpu()),
                    "fixed_rgb_l1": float(fixed_rgb_loss.detach().cpu()),
                    "random_backing_l1": float(random_backing_loss.detach().cpu()),
                    "mask_l1": float(mask_loss.detach().cpu()),
                    "rgb_flow_loss": float(rgb_flow_loss.detach().cpu()),
                    "rgb_flow_detail_loss": float(rgb_flow_detail_loss.detach().cpu()),
                    "rgb_flow_exclude_color_gradients": bool(
                        config.rgb_flow_exclude_color_gradients
                    ),
                    "smooth_loss": float(smooth_loss.detach().cpu()),
                    "geometry_residual_smooth_loss": float(geometry_residual_smooth_loss.detach().cpu()),
                    "guide_smooth_loss": float(guide_smooth_loss.detach().cpu()),
                    "effective_smooth_loss": float(effective_smooth_loss.detach().cpu()),
                    "guide_prior_loss": float(guide_prior_loss.detach().cpu()),
                    "guide_support_gauge_total": float(
                        guide_support_gauge_loss.detach().cpu()
                    ),
                    "guide_support_gauge_length_collapse": float(
                        guide_support_gauge_length_collapse.detach().cpu()
                    ),
                    "guide_support_gauge_slenderness_expansion": float(
                        guide_support_gauge_slenderness_expansion.detach().cpu()
                    ),
                    "clean_flow_guide_anchor_loss": float(guide_clean_flow_loss.detach().cpu()),
                    "clean_flow_guide_length_anchor_loss": float(
                        eval_guide_length_anchor_loss.cpu()
                    ),
                    "clean_flow_guide_length_anchor_reliable_fraction": float(
                        eval_guide_length_anchor_reliable_fraction.cpu()
                    ),
                    "clean_flow_3d_smooth_loss": float(clean_flow_smooth_loss.detach().cpu()),
                    "clean_flow_guide_anchor_fraction": float((model.guide_clean_flow_anchor_confidence >= float(config.clean_flow_anchor_min_confidence)).float().mean().detach().cpu()) if model.guide_clean_flow_anchor_confidence.numel() else 0.0,
                    "guide_residual_multiplier": float(model.guide_residual_multiplier),
                    "guide_coverage_residual_multiplier": float(model.guide_coverage_residual_multiplier),
                    "shape_detail_multiplier": float(model.shape_detail_multiplier),
                    "secondary_shape_residual_multiplier": float(
                        model.secondary_shape_residual_multiplier
                    ),
                    "gaussian_rgb_residual_multiplier": float(
                        model.gaussian_rgb_residual_multiplier
                    ),
                    "guide_frozen": bool(guide_frozen),
                    "guide_length_frozen": bool(guide_length_frozen),
                    "shape_detail_frozen": bool(shape_detail_frozen),
                    "secondary_shape_residual_frozen": bool(
                        secondary_shape_residual_frozen
                    ),
                    "lifecycle_statistics_active": bool(lifecycle_stats_active),
                    "root_move_loss": float(root_move_loss.detach().cpu()),
                    "mesh_no_penetration": {
                        "loss": float(mesh_no_penetration_loss.detach().cpu()),
                        "weight": float(config.mesh_no_penetration_weight),
                        "sampled_root_count": int(
                            mesh_no_penetration_root_indices.numel()
                        ) if mesh_no_penetration_root_indices is not None else 0,
                        "sampled_point_count": int(
                            mesh_no_penetration_depth.numel()
                        ),
                        "penetrating_fraction": float(
                            (mesh_no_penetration_depth.detach() > 0.0)
                            .float()
                            .mean()
                            .cpu()
                        ) if mesh_no_penetration_depth.numel() else 0.0,
                        "mean_depth": float(
                            mesh_no_penetration_depth.detach().mean().cpu()
                        ) if mesh_no_penetration_depth.numel() else 0.0,
                        "maximum_depth": float(
                            mesh_no_penetration_depth.detach().max().cpu()
                        ) if mesh_no_penetration_depth.numel() else 0.0,
                    },
                    "strand_crossing": {
                        "loss": float(strand_crossing_loss.detach().cpu()),
                        "weight": float(config.strand_crossing_weight),
                        "active_pair_count": int(
                            strand_crossing_stats["active_pair_count"]
                        ),
                        "positive_pair_count": int(
                            strand_crossing_stats["positive_pair_count"]
                        ),
                        "positive_pair_fraction": float(
                            strand_crossing_stats["positive_pair_fraction"]
                            .detach()
                            .cpu()
                        ),
                        "mean_normalized_depth": float(
                            strand_crossing_stats["mean_normalized_depth"]
                            .detach()
                            .cpu()
                        ),
                        "maximum_normalized_depth": float(
                            strand_crossing_stats["maximum_normalized_depth"]
                            .detach()
                            .cpu()
                        ),
                        "last_refresh_iteration": int(
                            strand_crossing_last_refresh_iteration
                        ),
                    },
                    "train": train_eval,
                    "test": test_eval,
                    "render": render_stats,
                    "groom": groom_parameter_stats(model.groom),
                    "effective_groom": effective_groom_stats(model),
                    "geometry_residual": render_geometry_residual_stats(model),
                    "gaussian_rgb_residual": (
                        model.gaussian_rgb_residual.stats(
                            multiplier=model.gaussian_rgb_residual_multiplier
                        )
                        if model.gaussian_rgb_residual is not None
                        else None
                    ),
                    "guide_view_sh": (
                        model.guide_view_sh.stats()
                        if model.guide_view_sh is not None
                        else None
                    ),
                    "rgb_flow": rgb_flow_stats,
                    "loss_mask": {
                        "edge_kernel": int(config.loss_mask_edge_kernel),
                        "edge_weight_mean": float(edge_loss_weight.detach().mean().cpu()),
                        "edge_weight_active_fraction": float((edge_loss_weight.detach() > 0.0).float().mean().cpu()),
                        "flow_mask_active_fraction": float((flow_supervision_mask.detach() > 0.0).float().mean().cpu()),
                    },
                    "max_memory_mb": round(memory_payload["max_memory_allocated_mb"], 2),
                    "memory_allocated_mb": round(memory_payload["memory_allocated_mb"], 2),
                    "memory_reserved_mb": round(memory_payload["memory_reserved_mb"], 2),
                    "max_memory_reserved_mb": round(memory_payload["max_memory_reserved_mb"], 2),
                    "nvidia_smi_process_mb": round(memory_payload["nvidia_smi_process_mb"], 2),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                eval_dir = output_dir / f"iter_{iteration:06d}"
                save_image(eval_dir / f"view_{idx:02d}_train_pred.png", pred)
                save_image(eval_dir / f"view_{idx:02d}_train_pred_fixed_bg.png", pred_fixed)
                save_image(eval_dir / f"view_{idx:02d}_train_alpha.png", alpha)
                save_image(eval_dir / f"view_{idx:02d}_mesh_depth.png", depth_to_image(mesh_depth.depth))
                save_image(eval_dir / f"view_{idx:02d}_mesh_valid.png", mesh_depth.valid[..., None].float())
                save_image(eval_dir / f"view_{idx:02d}_backing.png", backing_image)
                save_image(eval_dir / f"view_{idx:02d}_target_with_backing.png", target_with_backing)
                save_image(eval_dir / f"view_{idx:02d}_loss_edge_weight.png", edge_loss_weight)
                save_image(eval_dir / f"view_{idx:02d}_flow_supervision_mask.png", flow_supervision_mask)
                save_image(eval_dir / f"view_{idx:02d}_flow_loss_mask.png", flow_loss_mask_vis)
                if target_flow_vis is not None:
                    save_image(
                        eval_dir / f"view_{idx:02d}_target_rgb_flow.png",
                        torch.cat(
                            [
                                0.5 + 0.5 * target_flow_vis,
                                flow_loss_mask_vis.clamp(0.0, 1.0),
                            ],
                            dim=-1,
                        ),
                    )
                save_image(eval_dir / f"view_{idx:02d}_raw_diff.png", torch.abs(pred - target) * 4.0)
                save_image(eval_dir / f"view_{idx:02d}_composite_diff.png", torch.abs(pred - target_with_backing) * 4.0)
                save_clip_overlay(
                    eval_dir / f"view_{idx:02d}_clipped_visibility_overlay.png",
                    target_with_backing,
                    render_info["preclip_means"],
                    render_info["mesh_depth_keep_mask"],
                    viewmats[idx],
                    ks[idx],
                    behind_mesh_mask=render_info["mesh_depth_behind_mesh_mask"],
                )
                save_clip_overlay(
                    eval_dir / f"view_{idx:02d}_kept_gaussians_overlay.png",
                    target_with_backing,
                    render_info["preclip_means"],
                    render_info["mesh_depth_keep_mask"],
                    viewmats[idx],
                    ks[idx],
                    behind_mesh_mask=render_info["mesh_depth_behind_mesh_mask"],
                    mode="kept",
                )
                save_clip_overlay(
                    eval_dir / f"view_{idx:02d}_depth_clipped_gaussians_overlay.png",
                    target_with_backing,
                    render_info["preclip_means"],
                    render_info["mesh_depth_keep_mask"],
                    viewmats[idx],
                    ks[idx],
                    behind_mesh_mask=render_info["mesh_depth_behind_mesh_mask"],
                    mode="clipped",
                )
                diag_idx = int(test_indices[0] if test_indices else idx)
                diag_target = load_image(image_paths[diag_idx], device)
                diag_mask = load_mask(mask_paths[diag_idx], device)
                diag_mesh_color = sample_backing_color(config, device, train=False)
                diag_scene_bg = scene_background_color(config, device)
                diag_mesh_depth = render_model_mesh_depth(
                    model,
                    viewmats[diag_idx],
                    ks[diag_idx],
                    width,
                    height,
                    device=device,
                    ctx=mesh_depth_ctx,
                )
                diag_backing = make_mesh_backing_image(
                    diag_mesh_depth,
                    diag_mesh_color,
                    diag_scene_bg,
                    model=model,
                    viewmat=viewmats[diag_idx],
                    k=ks[diag_idx],
                    width=width,
                    height=height,
                    config=config,
                    device=device,
                    ctx=mesh_depth_ctx,
                    train=False,
                )
                diag_pred, diag_alpha, diag_gaussians, _, _, diag_info = render_view(
                    model,
                    viewmats[diag_idx],
                    ks[diag_idx],
                    width,
                    height,
                    config,
                    background=diag_mesh_color,
                    mesh_depth=diag_mesh_depth,
                    backing_image=diag_backing,
                    view_index=diag_idx,
                )
                diag_target_eval = composite_target(diag_target, diag_mask, diag_backing)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_gt.png", diag_target)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_target.png", diag_target_eval)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_pred.png", diag_pred)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_alpha.png", diag_alpha)
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_raw_diff_x4.png", torch.abs(diag_pred - diag_target) * 4.0)
                save_image(
                    eval_dir / f"view_{diag_idx:02d}_eval_composite_diff_x4.png",
                    torch.abs(diag_pred - diag_target_eval) * 4.0,
                )
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_mesh_depth.png", depth_to_image(diag_mesh_depth.depth))
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_mesh_valid.png", diag_mesh_depth.valid[..., None].float())
                save_image(eval_dir / f"view_{diag_idx:02d}_eval_backing.png", diag_backing)
                save_clip_overlay(
                    eval_dir / f"view_{diag_idx:02d}_eval_clipped_visibility_overlay.png",
                    diag_target_eval,
                    diag_info["preclip_means"],
                    diag_info["mesh_depth_keep_mask"],
                    viewmats[diag_idx],
                    ks[diag_idx],
                    behind_mesh_mask=diag_info["mesh_depth_behind_mesh_mask"],
                )
                save_clip_overlay(
                    eval_dir / f"view_{diag_idx:02d}_eval_kept_gaussians_overlay.png",
                    diag_target_eval,
                    diag_info["preclip_means"],
                    diag_info["mesh_depth_keep_mask"],
                    viewmats[diag_idx],
                    ks[diag_idx],
                    behind_mesh_mask=diag_info["mesh_depth_behind_mesh_mask"],
                    mode="kept",
                )
                save_clip_overlay(
                    eval_dir / f"view_{diag_idx:02d}_eval_depth_clipped_gaussians_overlay.png",
                    diag_target_eval,
                    diag_info["preclip_means"],
                    diag_info["mesh_depth_keep_mask"],
                    viewmats[diag_idx],
                    ks[diag_idx],
                    behind_mesh_mask=diag_info["mesh_depth_behind_mesh_mask"],
                    mode="clipped",
                )
                del (
                    train_eval,
                    test_eval,
                    diag_target,
                    diag_mask,
                    diag_mesh_color,
                    diag_scene_bg,
                    diag_mesh_depth,
                    diag_backing,
                    diag_pred,
                    diag_alpha,
                    diag_gaussians,
                    diag_info,
                    diag_target_eval,
                )
                release_cuda_cache()
                progress_event(
                    "after_eval_cache_release",
                    iteration=int(iteration),
                    memory_allocated_mb=float(torch.cuda.memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0,
                    memory_reserved_mb=float(torch.cuda.memory_reserved() / (1024 * 1024)) if torch.cuda.is_available() else 0.0,
                )
                enforce_cuda_memory_guard(config, device, iteration=int(iteration), stage="after_eval", progress_event=progress_event)

            should_regular_save = config.save_every > 0 and (
                iteration % config.save_every == 0 or iteration == config.iterations
            )
            should_stage_save = iteration in stage_save_iters
            if should_regular_save or should_stage_save:
                if should_regular_save and should_stage_save:
                    save_reason = "regular+stage"
                elif should_stage_save:
                    save_reason = "stage"
                else:
                    save_reason = "regular"
                torch.save(
                    {
                        "checkpoint_version": CURRENT_CHECKPOINT_VERSION,
                        "checkpoint_kind": "stage1_full",
                        "iteration": iteration,
                        "config": asdict(config),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "optimizer_param_names": stage1_optimizer_param_names(model, config),
                        "rng_state": capture_training_rng_state(generator),
                        "guide_residual_multiplier": float(model.guide_residual_multiplier),
                        "guide_coverage_residual_multiplier": float(model.guide_coverage_residual_multiplier),
                        "shape_detail_multiplier": float(model.shape_detail_multiplier),
                        "secondary_shape_residual_multiplier": float(
                            model.secondary_shape_residual_multiplier
                        ),
                        "gaussian_rgb_residual_multiplier": float(
                            model.gaussian_rgb_residual_multiplier
                        ),
                        "mesh_no_penetration_sdf_sha256": (
                            mesh_no_penetration_report["sdf_sha256"]
                            if mesh_no_penetration_report is not None
                            else None
                        ),
                        "lifecycle_history": lifecycle_history,
                        "strand_crossing_active_set": (
                            strand_crossing_active_set_cpu.checkpoint_state()
                            if strand_crossing_active_set_cpu is not None
                            else None
                        ),
                        "strand_crossing_last_refresh_iteration": int(
                            strand_crossing_last_refresh_iteration
                        ),
                        "strand_crossing_history": strand_crossing_history,
                        "save_reason": save_reason,
                    },
                    output_dir / f"checkpoint_{iteration:06d}.pt",
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train clean AniGroom White Tiger Stage 1.")
    parser.add_argument("--alignment-config", default="configs/white_tiger_mesh_alignment.json")
    parser.add_argument("--data-root", default="data/neuralfur_work/whiteTiger_processed/roaringwalk")
    parser.add_argument("--mesh-path", default="data_sources/neuralfur_official_results/whiteTiger/furless_reshaped.obj")
    parser.add_argument("--output-dir", default="outputs/white_tiger_stage1")
    parser.add_argument("--root-count", type=int, default=10000)
    parser.add_argument("--root-init-method", choices=("fps", "stratified"), default="fps")
    parser.add_argument("--candidate-multiplier", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--stage-save-iters", default="")
    parser.add_argument("--test-stride", type=int, default=6)
    parser.add_argument("--train-views", default="")
    parser.add_argument("--test-views", default="")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--expected-width", type=int, default=1920)
    parser.add_argument("--expected-height", type=int, default=1080)
    parser.add_argument("--init-mesh-scale", type=float, default=1.28)
    parser.add_argument("--init-mesh-translation", type=float, nargs=3, default=[0.0, 0.32, 0.02])
    parser.add_argument("--init-groom-length", type=float, default=0.060)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--min-segments", type=int, default=10)
    parser.add_argument("--segment-length-origin", type=float, default=0.010)
    parser.add_argument("--segments-per-unit-length", type=float, default=84.19047619047619)
    parser.add_argument("--segments-per-unit-complexity", type=float, default=23.771428571428572)
    parser.add_argument("--child-count", type=int, default=8)
    parser.add_argument("--gaussian-length-overlap", type=float, default=1.45)
    parser.add_argument("--projected-init-views", type=int, default=24)
    parser.add_argument("--projected-init-min-confidence", type=float, default=0.08)
    parser.add_argument("--projected-init-depth-abs-tolerance", type=float, default=0.03)
    parser.add_argument("--projected-init-depth-rel-tolerance", type=float, default=0.01)
    parser.add_argument("--projected-init-local-depth-kernel", type=int, default=7)
    parser.add_argument("--projected-init-front-normal-z", type=float, default=0.15)
    parser.add_argument("--projected-init-mask-edge-kernel", type=int, default=9)
    parser.add_argument("--projected-init-view-angle-power", type=float, default=1.0)
    parser.add_argument("--clean-flow-target", default="")
    parser.add_argument("--clean-flow-init", action="store_true")
    parser.add_argument("--clean-flow-init-k", type=int, default=8)
    parser.add_argument("--clean-flow-init-min-confidence", type=float, default=0.03)
    parser.add_argument("--clean-flow-anchor-min-confidence", type=float, default=0.35)
    parser.add_argument("--clean-flow-length-init", action="store_true")
    parser.add_argument("--clean-flow-length-init-scale", type=float, default=0.30)
    parser.add_argument("--clean-flow-length-init-min-confidence", type=float, default=0.50)
    parser.add_argument("--clean-flow-guide-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--clean-flow-guide-length-anchor-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--clean-flow-guide-length-anchor-reduction",
        choices=("mean_l1", "tail_concentration"),
        default="mean_l1",
    )
    parser.add_argument("--clean-flow-3d-smooth-weight", type=float, default=0.0)
    parser.add_argument("--guide-root-count", type=int, default=0)
    parser.add_argument("--guide-candidate-multiplier", type=float, default=8.0)
    parser.add_argument("--guide-roots-from-clean-flow", action="store_true")
    parser.add_argument("--guide-interpolation-k", type=int, default=8)
    parser.add_argument(
        "--geometry-residual-domain",
        choices=("render", "secondary_guide"),
        default="render",
    )
    parser.add_argument("--secondary-guide-root-count", type=int, default=0)
    parser.add_argument("--secondary-guide-candidate-multiplier", type=float, default=16.0)
    parser.add_argument("--secondary-guide-interpolation-k", type=int, default=8)
    parser.add_argument("--secondary-guide-smooth-k", type=int, default=32)
    parser.add_argument(
        "--render-geometry-parameterization",
        choices=(
            "absolute_endpoint",
            "zero_centered_residual",
            "zero_centered_log_length_residual",
            "zero_centered_unbounded_log_length_residual",
            "zero_centered_asinh_log_length_residual",
        ),
        default="absolute_endpoint",
    )
    parser.add_argument("--guide-length-residual-scale", type=float, default=0.0)
    parser.add_argument("--guide-direction-residual-scale", type=float, default=1.0)
    parser.add_argument("--guide-width-residual-scale", type=float, default=1.0)
    parser.add_argument("--guide-child-radius-residual-scale", type=float, default=1.0)
    parser.add_argument("--guide-clump-residual-scale", type=float, default=1.0)
    parser.add_argument("--guide-curl-residual-scale", type=float, default=1.0)
    parser.add_argument("--guide-prior-weight", type=float, default=0.0)
    parser.add_argument("--guide-prior-direction-weight", type=float, default=1.0)
    parser.add_argument("--guide-prior-curl-weight", type=float, default=0.08)
    parser.add_argument("--guide-prior-length-weight", type=float, default=0.0)
    parser.add_argument("--guide-prior-width-weight", type=float, default=0.0)
    parser.add_argument("--guide-prior-child-radius-weight", type=float, default=0.0)
    parser.add_argument("--guide-prior-clump-weight", type=float, default=0.0)
    parser.add_argument("--guide-support-gauge-weight", type=float, default=0.0)
    parser.add_argument("--guide-view-sh-support", action="store_true")
    parser.add_argument("--guide-view-sh-scale", type=float, default=0.20)
    parser.add_argument("--lr-guide-view-sh", type=float, default=2.0e-2)
    parser.add_argument("--view-gated-ownership-support", action="store_true")
    parser.add_argument("--view-gate-geometry-support", action="store_true")
    parser.add_argument("--view-gate-length-confidence-support", action="store_true")
    parser.add_argument("--view-gate-floor", type=float, default=0.0)
    parser.add_argument(
        "--view-gate-normalization",
        choices=("raw_q95", "equal_owner_budget"),
        default="raw_q95",
    )
    parser.add_argument(
        "--render-length-prior-coordinate",
        choices=("decoded", "natural_log_ratio", "raw"),
        default="decoded",
    )
    parser.add_argument(
        "--render-length-prior-reduction",
        choices=(
            "mean_l1",
            "fourth_moment",
            "population_stable_handoff",
            "tail_concentration_handoff",
        ),
        default="mean_l1",
    )
    parser.add_argument("--guide-smooth-weight", type=float, default=0.0)
    parser.add_argument(
        "--guide-length-smooth-mode",
        choices=GUIDE_LENGTH_SMOOTH_MODES,
        default="edge_relative",
    )
    parser.add_argument("--guide-residual-unlock-start", type=int, default=0)
    parser.add_argument("--guide-residual-unlock-end", type=int, default=0)
    parser.add_argument("--guide-residual-initial-multiplier", type=float, default=1.0)
    parser.add_argument("--guide-coverage-residual-unlock-start", type=int, default=0)
    parser.add_argument("--guide-coverage-residual-unlock-end", type=int, default=0)
    parser.add_argument("--guide-coverage-residual-initial-multiplier", type=float, default=1.0)
    parser.add_argument("--guide-freeze-until", type=int, default=0)
    parser.add_argument("--guide-length-freeze-until", type=int, default=-1)
    parser.add_argument("--shape-detail-freeze-until", type=int, default=0)
    parser.add_argument("--shape-detail-unlock-end", type=int, default=0)
    parser.add_argument("--secondary-shape-residual-unlock-start", type=int, default=0)
    parser.add_argument("--secondary-shape-residual-unlock-end", type=int, default=0)
    parser.add_argument("--shape-curl-scale", type=float, default=1.0)
    parser.add_argument("--guide-densify-start", type=int, default=0)
    parser.add_argument("--guide-densify-interval", type=int, default=0)
    parser.add_argument("--guide-densify-until", type=int, default=0)
    parser.add_argument("--guide-densify-score-threshold", type=float, default=0.0)
    parser.add_argument("--guide-densify-max-splits-per-event", type=int, default=0)
    parser.add_argument(
        "--guide-densify-policy",
        choices=("global_score_budget", "surface_attribution_local_max"),
        default="global_score_budget",
    )
    parser.add_argument("--guide-densify-children-per-parent", type=int, default=1)
    parser.add_argument("--guide-densify-neighbor-count", type=int, default=12)
    parser.add_argument("--guide-densify-candidate-rings", type=int, default=3)
    parser.add_argument("--guide-densify-candidate-face-count", type=int, default=32)
    parser.add_argument("--guide-densify-min-child-distance", type=float, default=0.0)
    parser.add_argument("--guide-densify-render-root-k", type=int, default=8)
    parser.add_argument("--lr-groom", type=float, default=1.4e-2)
    parser.add_argument("--lr-high-frequency-shape-scale", type=float, default=0.35)
    parser.add_argument("--lr-color", type=float, default=2.0e-2)
    parser.add_argument("--color-freeze-until", type=int, default=0)
    parser.add_argument("--gaussian-rgb-residual-support", action="store_true")
    parser.add_argument("--gaussian-rgb-residual-control-points", type=int, default=36)
    parser.add_argument("--gaussian-rgb-residual-scale", type=float, default=0.20)
    parser.add_argument("--gaussian-rgb-residual-unlock-start", type=int, default=10000)
    parser.add_argument("--gaussian-rgb-residual-unlock-end", type=int, default=20000)
    parser.add_argument(
        "--gaussian-rgb-residual-initial-multiplier",
        type=float,
        default=0.0,
    )
    parser.add_argument("--lr-root", type=float, default=7.5e-4)
    parser.add_argument("--lr-calibration", type=float, default=5.0e-4)
    parser.add_argument("--rgb-weight", type=float, default=1.0)
    parser.add_argument("--random-backing-loss-weight", type=float, default=0.25)
    parser.add_argument("--mask-weight", type=float, default=0.15)
    parser.add_argument("--rgb-flow-weight", type=float, default=0.0)
    parser.add_argument("--rgb-flow-detail-weight", type=float, default=0.0)
    parser.add_argument("--rgb-flow-min-confidence", type=float, default=0.08)
    parser.add_argument("--rgb-flow-exclude-color-gradients", action="store_true")
    parser.add_argument("--loss-mask-edge-kernel", type=int, default=1)
    parser.add_argument(
        "--smooth-graph-mode",
        choices=("euclidean_knn", "surface_hierarchical"),
        default="euclidean_knn",
    )
    parser.add_argument("--smooth-graph-k", type=int, default=8)
    parser.add_argument(
        "--smooth-field-metric",
        choices=SMOOTH_FIELD_METRICS,
        default="ambient",
    )
    parser.add_argument("--smooth-weight", type=float, default=0.04)
    parser.add_argument("--geometry-residual-smooth-scale", type=float, default=1.0)
    parser.add_argument("--effective-smooth-weight", type=float, default=0.0)
    parser.add_argument("--root-move-reg-weight", type=float, default=0.003)
    parser.add_argument("--compute-lpips", action="store_true")
    parser.add_argument("--black-background", action="store_true")
    parser.add_argument("--disable-random-backing-color", action="store_true")
    parser.add_argument("--backing-color-min", type=float, default=0.05)
    parser.add_argument("--backing-color-max", type=float, default=0.85)
    parser.add_argument("--disable-random-mesh-backing-texture", action="store_true")
    parser.add_argument("--mesh-backing-texture-strength", type=float, default=0.30)
    parser.add_argument("--mesh-backing-texture-octaves", type=int, default=5)
    parser.add_argument("--mesh-no-penetration-support", action="store_true")
    parser.add_argument("--mesh-no-penetration-sdf", default="")
    parser.add_argument("--mesh-no-penetration-weight", type=float, default=0.0)
    parser.add_argument("--mesh-no-penetration-root-batch", type=int, default=16384)
    parser.add_argument("--strand-crossing-support", action="store_true")
    parser.add_argument("--strand-crossing-weight", type=float, default=0.0)
    parser.add_argument("--strand-crossing-refresh-interval", type=int, default=0)
    parser.add_argument("--strand-crossing-query-batch", type=int, default=50000)
    parser.add_argument("--strand-crossing-exact-pair-batch", type=int, default=250000)
    parser.add_argument("--disable-mesh-depth-clipping", action="store_true")
    parser.add_argument("--mesh-depth-abs-tolerance", type=float, default=0.018)
    parser.add_argument("--mesh-depth-rel-tolerance", type=float, default=0.004)
    parser.add_argument("--mesh-depth-local-kernel", type=int, default=1)
    parser.add_argument("--disable-mesh-backing-compositing", action="store_true")
    parser.add_argument("--gpu-memory-limit-gb", type=float, default=0.0)
    parser.add_argument("--gpu-memory-check-interval", type=int, default=20)
    parser.add_argument("--densify-warmup", type=int, required=True)
    parser.add_argument("--densify-interval", type=int, required=True)
    parser.add_argument("--densify-until", type=int, required=True)
    parser.add_argument("--densify-score-threshold", type=float, required=True)
    parser.add_argument("--densify-min-contribution", type=float, required=True)
    parser.add_argument("--densify-residual-weight", type=float, default=0.0)
    parser.add_argument("--densify-residual-mode", choices=("root_pixel", "coverage_pooled", "pixel_to_root"), default="root_pixel")
    parser.add_argument("--densify-residual-pool-radius", type=int, default=15)
    parser.add_argument("--densify-residual-alpha-weight", type=float, default=1.0)
    parser.add_argument("--densify-residual-rgb-weight", type=float, default=0.25)
    parser.add_argument("--densify-pixel-evidence-topk", type=int, default=4096)
    parser.add_argument("--densify-pixel-evidence-root-k", type=int, default=4)
    parser.add_argument("--densify-pixel-evidence-min", type=float, default=0.02)
    parser.add_argument("--densify-pixel-evidence-chunk", type=int, default=512)
    parser.add_argument("--lifecycle-score-mode", choices=("raw", "sample_normalized", "mean_visible"), default="raw")
    parser.add_argument("--local-child-color-support", action="store_true")
    parser.add_argument("--local-child-color-scale", type=float, default=0.20)
    parser.add_argument("--max-splits-per-event", type=int, required=True)
    parser.add_argument("--split-children-per-parent", type=int, required=True)
    parser.add_argument("--split-neighbor-count", type=int, required=True)
    parser.add_argument("--split-candidate-rings", type=int, required=True)
    parser.add_argument("--split-candidate-face-count", type=int, required=True)
    parser.add_argument("--split-min-child-distance", type=float, required=True)
    parser.add_argument("--prune-start", type=int, required=True)
    parser.add_argument("--prune-interval", type=int, required=True)
    parser.add_argument("--prune-min-contribution", type=float, required=True)
    parser.add_argument("--prune-min-opacity", type=float, required=True)
    parser.add_argument("--prune-max-fraction", type=float, required=True)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--no-resume-optimizer", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Stage1Config:
    config = Stage1Config(
        data_root=args.data_root,
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        root_count=args.root_count,
        root_init_method=args.root_init_method,
        candidate_multiplier=args.candidate_multiplier,
        iterations=args.iterations,
        eval_every=args.eval_every,
        save_every=args.save_every,
        stage_save_iters=args.stage_save_iters,
        test_stride=args.test_stride,
        train_views=args.train_views,
        test_views=args.test_views,
        seed=args.seed,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        init_mesh_scale=args.init_mesh_scale,
        init_mesh_translation=tuple(float(v) for v in args.init_mesh_translation),
        init_groom_length=args.init_groom_length,
        samples=args.samples,
        min_segments=args.min_segments,
        segment_length_origin=args.segment_length_origin,
        segments_per_unit_length=args.segments_per_unit_length,
        segments_per_unit_complexity=args.segments_per_unit_complexity,
        child_count=args.child_count,
        gaussian_length_overlap=args.gaussian_length_overlap,
        projected_init_views=args.projected_init_views,
        projected_init_min_confidence=args.projected_init_min_confidence,
        projected_init_depth_abs_tolerance=args.projected_init_depth_abs_tolerance,
        projected_init_depth_rel_tolerance=args.projected_init_depth_rel_tolerance,
        projected_init_local_depth_kernel=args.projected_init_local_depth_kernel,
        projected_init_front_normal_z=args.projected_init_front_normal_z,
        projected_init_mask_edge_kernel=args.projected_init_mask_edge_kernel,
        projected_init_view_angle_power=args.projected_init_view_angle_power,
        clean_flow_target=args.clean_flow_target,
        clean_flow_init=args.clean_flow_init,
        clean_flow_init_k=args.clean_flow_init_k,
        clean_flow_init_min_confidence=args.clean_flow_init_min_confidence,
        clean_flow_anchor_min_confidence=args.clean_flow_anchor_min_confidence,
        clean_flow_length_init=args.clean_flow_length_init,
        clean_flow_length_init_scale=args.clean_flow_length_init_scale,
        clean_flow_length_init_min_confidence=args.clean_flow_length_init_min_confidence,
        clean_flow_guide_anchor_weight=args.clean_flow_guide_anchor_weight,
        clean_flow_guide_length_anchor_weight=(
            args.clean_flow_guide_length_anchor_weight
        ),
        clean_flow_guide_length_anchor_reduction=(
            args.clean_flow_guide_length_anchor_reduction
        ),
        clean_flow_3d_smooth_weight=args.clean_flow_3d_smooth_weight,
        guide_root_count=args.guide_root_count,
        guide_candidate_multiplier=args.guide_candidate_multiplier,
        guide_roots_from_clean_flow=args.guide_roots_from_clean_flow,
        guide_interpolation_k=args.guide_interpolation_k,
        geometry_residual_domain=args.geometry_residual_domain,
        secondary_guide_root_count=args.secondary_guide_root_count,
        secondary_guide_candidate_multiplier=args.secondary_guide_candidate_multiplier,
        secondary_guide_interpolation_k=args.secondary_guide_interpolation_k,
        secondary_guide_smooth_k=args.secondary_guide_smooth_k,
        render_geometry_parameterization=args.render_geometry_parameterization,
        guide_length_residual_scale=args.guide_length_residual_scale,
        guide_direction_residual_scale=args.guide_direction_residual_scale,
        guide_width_residual_scale=args.guide_width_residual_scale,
        guide_child_radius_residual_scale=args.guide_child_radius_residual_scale,
        guide_clump_residual_scale=args.guide_clump_residual_scale,
        guide_curl_residual_scale=args.guide_curl_residual_scale,
        guide_prior_weight=args.guide_prior_weight,
        guide_prior_direction_weight=args.guide_prior_direction_weight,
        guide_prior_curl_weight=args.guide_prior_curl_weight,
        guide_prior_length_weight=args.guide_prior_length_weight,
        guide_prior_width_weight=args.guide_prior_width_weight,
        guide_prior_child_radius_weight=args.guide_prior_child_radius_weight,
        guide_prior_clump_weight=args.guide_prior_clump_weight,
        guide_support_gauge_weight=args.guide_support_gauge_weight,
        guide_view_sh_support=args.guide_view_sh_support,
        guide_view_sh_scale=args.guide_view_sh_scale,
        lr_guide_view_sh=args.lr_guide_view_sh,
        view_gated_ownership_support=args.view_gated_ownership_support,
        view_gate_geometry_support=args.view_gate_geometry_support,
        view_gate_length_confidence_support=args.view_gate_length_confidence_support,
        view_gate_floor=args.view_gate_floor,
        view_gate_normalization=args.view_gate_normalization,
        render_length_prior_coordinate=args.render_length_prior_coordinate,
        render_length_prior_reduction=args.render_length_prior_reduction,
        guide_smooth_weight=args.guide_smooth_weight,
        guide_length_smooth_mode=args.guide_length_smooth_mode,
        guide_residual_unlock_start=args.guide_residual_unlock_start,
        guide_residual_unlock_end=args.guide_residual_unlock_end,
        guide_residual_initial_multiplier=args.guide_residual_initial_multiplier,
        guide_coverage_residual_unlock_start=args.guide_coverage_residual_unlock_start,
        guide_coverage_residual_unlock_end=args.guide_coverage_residual_unlock_end,
        guide_coverage_residual_initial_multiplier=args.guide_coverage_residual_initial_multiplier,
        guide_freeze_until=args.guide_freeze_until,
        guide_length_freeze_until=args.guide_length_freeze_until,
        shape_detail_freeze_until=args.shape_detail_freeze_until,
        shape_detail_unlock_end=args.shape_detail_unlock_end,
        secondary_shape_residual_unlock_start=(
            args.secondary_shape_residual_unlock_start
        ),
        secondary_shape_residual_unlock_end=args.secondary_shape_residual_unlock_end,
        shape_curl_scale=args.shape_curl_scale,
        guide_densify_start=args.guide_densify_start,
        guide_densify_interval=args.guide_densify_interval,
        guide_densify_until=args.guide_densify_until,
        guide_densify_score_threshold=args.guide_densify_score_threshold,
        guide_densify_max_splits_per_event=args.guide_densify_max_splits_per_event,
        guide_densify_policy=args.guide_densify_policy,
        guide_densify_children_per_parent=args.guide_densify_children_per_parent,
        guide_densify_neighbor_count=args.guide_densify_neighbor_count,
        guide_densify_candidate_rings=args.guide_densify_candidate_rings,
        guide_densify_candidate_face_count=args.guide_densify_candidate_face_count,
        guide_densify_min_child_distance=args.guide_densify_min_child_distance,
        guide_densify_render_root_k=args.guide_densify_render_root_k,
        lr_groom=args.lr_groom,
        lr_high_frequency_shape_scale=args.lr_high_frequency_shape_scale,
        lr_color=args.lr_color,
        color_freeze_until=args.color_freeze_until,
        gaussian_rgb_residual_support=args.gaussian_rgb_residual_support,
        gaussian_rgb_residual_control_points=args.gaussian_rgb_residual_control_points,
        gaussian_rgb_residual_scale=args.gaussian_rgb_residual_scale,
        gaussian_rgb_residual_unlock_start=args.gaussian_rgb_residual_unlock_start,
        gaussian_rgb_residual_unlock_end=args.gaussian_rgb_residual_unlock_end,
        gaussian_rgb_residual_initial_multiplier=(
            args.gaussian_rgb_residual_initial_multiplier
        ),
        lr_root=args.lr_root,
        lr_calibration=args.lr_calibration,
        rgb_weight=args.rgb_weight,
        random_backing_loss_weight=args.random_backing_loss_weight,
        mask_weight=args.mask_weight,
        rgb_flow_weight=args.rgb_flow_weight,
        rgb_flow_detail_weight=args.rgb_flow_detail_weight,
        rgb_flow_min_confidence=args.rgb_flow_min_confidence,
        rgb_flow_exclude_color_gradients=args.rgb_flow_exclude_color_gradients,
        loss_mask_edge_kernel=args.loss_mask_edge_kernel,
        smooth_graph_mode=args.smooth_graph_mode,
        smooth_graph_k=args.smooth_graph_k,
        smooth_field_metric=args.smooth_field_metric,
        smooth_weight=args.smooth_weight,
        geometry_residual_smooth_scale=args.geometry_residual_smooth_scale,
        effective_smooth_weight=args.effective_smooth_weight,
        root_move_reg_weight=args.root_move_reg_weight,
        compute_lpips=args.compute_lpips,
        white_background=not args.black_background,
        random_backing_color=not args.disable_random_backing_color,
        backing_color_min=args.backing_color_min,
        backing_color_max=args.backing_color_max,
        random_mesh_backing_texture=not args.disable_random_mesh_backing_texture,
        mesh_backing_texture_strength=args.mesh_backing_texture_strength,
        mesh_backing_texture_octaves=args.mesh_backing_texture_octaves,
        mesh_no_penetration_support=args.mesh_no_penetration_support,
        mesh_no_penetration_sdf=args.mesh_no_penetration_sdf,
        mesh_no_penetration_weight=args.mesh_no_penetration_weight,
        mesh_no_penetration_root_batch=args.mesh_no_penetration_root_batch,
        strand_crossing_support=args.strand_crossing_support,
        strand_crossing_weight=args.strand_crossing_weight,
        strand_crossing_refresh_interval=args.strand_crossing_refresh_interval,
        strand_crossing_query_batch=args.strand_crossing_query_batch,
        strand_crossing_exact_pair_batch=args.strand_crossing_exact_pair_batch,
        mesh_depth_clipping=not args.disable_mesh_depth_clipping,
        mesh_depth_abs_tolerance=args.mesh_depth_abs_tolerance,
        mesh_depth_rel_tolerance=args.mesh_depth_rel_tolerance,
        mesh_depth_local_kernel=args.mesh_depth_local_kernel,
        mesh_backing_compositing=not args.disable_mesh_backing_compositing,
        gpu_memory_limit_gb=args.gpu_memory_limit_gb,
        gpu_memory_check_interval=args.gpu_memory_check_interval,
        densify_warmup=args.densify_warmup,
        densify_interval=args.densify_interval,
        densify_until=args.densify_until,
        densify_score_threshold=args.densify_score_threshold,
        densify_min_contribution=args.densify_min_contribution,
        densify_residual_weight=args.densify_residual_weight,
        densify_residual_mode=args.densify_residual_mode,
        densify_residual_pool_radius=args.densify_residual_pool_radius,
        densify_residual_alpha_weight=args.densify_residual_alpha_weight,
        densify_residual_rgb_weight=args.densify_residual_rgb_weight,
        densify_pixel_evidence_topk=args.densify_pixel_evidence_topk,
        densify_pixel_evidence_root_k=args.densify_pixel_evidence_root_k,
        densify_pixel_evidence_min=args.densify_pixel_evidence_min,
        densify_pixel_evidence_chunk=args.densify_pixel_evidence_chunk,
        lifecycle_score_mode=args.lifecycle_score_mode,
        local_child_color_support=args.local_child_color_support,
        local_child_color_scale=args.local_child_color_scale,
        max_splits_per_event=args.max_splits_per_event,
        split_children_per_parent=args.split_children_per_parent,
        split_neighbor_count=args.split_neighbor_count,
        split_candidate_rings=args.split_candidate_rings,
        split_candidate_face_count=args.split_candidate_face_count,
        split_min_child_distance=args.split_min_child_distance,
        prune_start=args.prune_start,
        prune_interval=args.prune_interval,
        prune_min_contribution=args.prune_min_contribution,
        prune_min_opacity=args.prune_min_opacity,
        prune_max_fraction=args.prune_max_fraction,
        resume_checkpoint=args.resume_checkpoint,
        resume_optimizer=not args.no_resume_optimizer,
    )
    return config


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    explicit_paths = {
        "data_root": args.data_root,
        "mesh_path": args.mesh_path,
        "output_dir": args.output_dir,
    }
    default_paths = {
        "data_root": parser.get_default("data_root"),
        "mesh_path": parser.get_default("mesh_path"),
        "output_dir": parser.get_default("output_dir"),
    }
    explicit_alignment = {
        "init_mesh_scale": args.init_mesh_scale,
        "init_mesh_translation": tuple(args.init_mesh_translation),
    }
    default_alignment = {
        "init_mesh_scale": parser.get_default("init_mesh_scale"),
        "init_mesh_translation": tuple(parser.get_default("init_mesh_translation")),
    }
    apply_alignment_to_namespace(args, load_alignment_config(args.alignment_config), include_uv=False)
    for name, value in explicit_paths.items():
        if value != default_paths[name]:
            setattr(args, name, value)
    for name, value in explicit_alignment.items():
        if value != default_alignment[name]:
            setattr(args, name, value)
    train_white_tiger_stage1(config_from_args(args))


if __name__ == "__main__":
    main()

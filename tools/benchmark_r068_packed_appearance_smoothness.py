from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import GroomParameterField
from tools.train_white_tiger_stage1 import (
    packed_appearance_root_graph_smoothness,
    precompute_root_graph_edge_weight_cache,
    root_graph_smoothness,
)


ROOT_COUNT = 471_673
GRAPH_K = 32
DEFAULT_WARMUP = 5
DEFAULT_REPETITIONS = 20
DEFAULT_OUTPUT = Path("outputs/r068_packed_appearance_smoothness/benchmark.json")


def synthetic_directed_edges(
    root_count: int = ROOT_COUNT,
    k: int = GRAPH_K,
    *,
    device: torch.device,
) -> torch.Tensor:
    source = torch.arange(root_count, device=device, dtype=torch.long).repeat_interleave(k)
    offsets = torch.arange(1, k + 1, device=device, dtype=torch.long).repeat(root_count)
    destination = (source + offsets).remainder(root_count)
    return torch.stack((source, destination), dim=1)


def clear_gradients(field: GroomParameterField) -> None:
    field.zero_grad(set_to_none=True)


def capture_appearance_gradients(field: GroomParameterField) -> dict[str, torch.Tensor]:
    names = (
        "root_color_raw",
        "tip_color_raw",
        "opacity_raw",
        "tip_opacity_ratio_raw",
    )
    return {
        name: getattr(field, name).grad.detach().clone()
        for name in names
    }


def evaluate_once(
    field: GroomParameterField,
    loss_fn,
) -> tuple[float, dict[str, torch.Tensor], float]:
    clear_gradients(field)
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = loss_fn()
    loss.backward()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return float(loss.detach().cpu()), capture_appearance_gradients(field), elapsed_ms


def summarize_times(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": float(np.median(np.asarray(values, dtype=np.float64))),
        "p95_ms": float(np.percentile(np.asarray(values, dtype=np.float64), 95.0)),
    }


def run_benchmark(
    *,
    warmup: int,
    repetitions: int,
    output: Path,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("R068 benchmark requires CUDA")
    if repetitions < 20:
        raise ValueError("repetitions must be at least 20")
    device = torch.device("cuda")
    torch.manual_seed(68)
    field = GroomParameterField(ROOT_COUNT, device=device)
    with torch.no_grad():
        field.root_color_raw.copy_(torch.randn_like(field.root_color_raw))
        field.tip_color_raw.copy_(torch.randn_like(field.tip_color_raw))
        field.opacity_raw.copy_(torch.randn_like(field.opacity_raw))
        field.tip_opacity_ratio_raw.copy_(torch.randn_like(field.tip_opacity_ratio_raw))
    edges = synthetic_directed_edges(device=device)
    confidence = torch.rand((ROOT_COUNT,), device=device)
    cache = precompute_root_graph_edge_weight_cache(
        edges,
        confidence,
        dtype=field.root_color_raw.dtype,
    )

    packed_fn = lambda: packed_appearance_root_graph_smoothness(
        field,
        edges,
        edge_weight_cache=cache,
    )
    explicit_fn = lambda: root_graph_smoothness(
        field,
        edges,
        confidence,
        include_geometry=False,
        appearance_only=True,
    )

    for _ in range(warmup):
        evaluate_once(field, packed_fn)
    for _ in range(warmup):
        evaluate_once(field, explicit_fn)

    packed_loss, packed_gradients, _ = evaluate_once(field, packed_fn)
    explicit_loss, explicit_gradients, _ = evaluate_once(field, explicit_fn)
    gradient_errors = {
        name: float((packed_gradients[name] - explicit_gradients[name]).abs().max().cpu())
        for name in packed_gradients
    }

    torch.cuda.reset_peak_memory_stats(device)
    packed_times = [evaluate_once(field, packed_fn)[2] for _ in range(repetitions)]
    packed_peak = int(torch.cuda.max_memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    explicit_times = [evaluate_once(field, explicit_fn)[2] for _ in range(repetitions)]
    explicit_peak = int(torch.cuda.max_memory_allocated(device))

    report: dict[str, object] = {
        "root_count": ROOT_COUNT,
        "graph_k": GRAPH_K,
        "directed_edge_count": int(edges.shape[0]),
        "warmup_repetitions": int(warmup),
        "measured_repetitions": int(repetitions),
        "device": torch.cuda.get_device_name(device),
        "packed": {
            **summarize_times(packed_times),
            "peak_allocation_bytes": packed_peak,
        },
        "explicit": {
            **summarize_times(explicit_times),
            "peak_allocation_bytes": explicit_peak,
        },
        "errors": {
            "scalar_max_abs_error": abs(packed_loss - explicit_loss),
            "gradient_max_abs_error": max(gradient_errors.values()),
            "gradient_max_abs_error_by_field": gradient_errors,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_benchmark(
        warmup=max(0, int(args.warmup)),
        repetitions=int(args.repetitions),
        output=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

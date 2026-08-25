from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anigroom.grooming import GroomParameterField, build_strands


ROOT_COUNT = 471_673
SAMPLES = 64
DEFAULT_WARMUP = 5
DEFAULT_REPETITIONS = 20
DEFAULT_OUTPUT = Path("outputs/r068_zero_curl_fastpath/benchmark.json")
ACTIVE_GRADIENT_NAMES = ("length", "direction_local", "brush_stiffness")
CURL_GRADIENT_NAMES = ("curl_radius_ratio", "curl_turns", "curl_phase")


def synthetic_inputs(
    *,
    root_count: int = ROOT_COUNT,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    roots = torch.zeros((root_count, 3), device=device, dtype=torch.float32)
    roots[:, 0] = torch.linspace(-0.2, 0.2, root_count, device=device)
    normals = torch.zeros_like(roots)
    normals[:, 2] = 1.0
    tangents = torch.zeros_like(roots)
    tangents[:, 0] = 1.0
    bitangents = torch.cross(normals, tangents, dim=-1)
    return roots, normals, tangents, bitangents


def synthetic_groom(
    *,
    root_count: int = ROOT_COUNT,
    device: torch.device,
):
    field = GroomParameterField(root_count, device=device)
    decoded = field.decode()
    values = {
        field_info.name: getattr(decoded, field_info.name).detach()
        for field_info in fields(decoded)
    }
    values.update(
        length=torch.full(
            (root_count, 1), 0.04, device=device, dtype=torch.float32, requires_grad=True
        ),
        direction_local=torch.tensor(
            [[0.78, 0.21, 0.58]], device=device, dtype=torch.float32
        ).expand(root_count, -1).clone().requires_grad_(True),
        brush_stiffness=torch.full(
            (root_count, 1), 0.65, device=device, dtype=torch.float32, requires_grad=True
        ),
        curl_radius_ratio=torch.zeros(
            (root_count, 1), device=device, dtype=torch.float32, requires_grad=True
        ),
        curl_turns=torch.full(
            (root_count, 1), 1.4, device=device, dtype=torch.float32, requires_grad=True
        ),
        curl_phase=torch.full(
            (root_count, 1), 0.4, device=device, dtype=torch.float32, requires_grad=True
        ),
    )
    return replace(decoded, **values)


def clone_groom(groom):
    values = {}
    for field_info in fields(groom):
        value = getattr(groom, field_info.name)
        values[field_info.name] = value.detach().clone().requires_grad_(value.requires_grad)
    return replace(groom, **values)


def clear_gradients(groom) -> None:
    for field_info in fields(groom):
        getattr(groom, field_info.name).grad = None


def capture_gradients(groom, names: tuple[str, ...]) -> dict[str, torch.Tensor | None]:
    return {
        name: (
            getattr(groom, name).grad.detach().clone()
            if getattr(groom, name).grad is not None
            else None
        )
        for name in names
    }


def scalar_loss(output: tuple[torch.Tensor, ...]) -> torch.Tensor:
    points, widths, colors, opacities = output
    return (
        points.square().mean()
        + widths.square().mean()
        + colors.square().mean()
        + opacities.square().mean()
    )


def evaluate_once(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    groom,
    *,
    enable_curl: bool | None,
    capture: bool,
) -> tuple[float, tuple[torch.Tensor, ...] | None, dict[str, torch.Tensor | None], float]:
    clear_gradients(groom)
    torch.cuda.synchronize()
    started = time.perf_counter()
    if enable_curl is None:
        output = build_strands(*inputs, groom, samples=SAMPLES)
    else:
        output = build_strands(
            *inputs,
            groom,
            samples=SAMPLES,
            enable_curl=enable_curl,
        )
    loss = scalar_loss(output)
    loss.backward()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    gradients = capture_gradients(
        groom,
        ACTIVE_GRADIENT_NAMES + CURL_GRADIENT_NAMES,
    )
    saved_output = (
        tuple(value.detach().clone() for value in output)
        if capture
        else None
    )
    scalar = float(loss.detach().cpu())
    return scalar, saved_output, gradients, elapsed_ms


def summarize_times(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
    }


def measure_variant(
    inputs,
    groom,
    *,
    enable_curl: bool | None,
    repetitions: int,
) -> tuple[dict[str, float | int], list[float]]:
    torch.cuda.reset_peak_memory_stats()
    times = [
        evaluate_once(
            inputs,
            groom,
            enable_curl=enable_curl,
            capture=False,
        )[-1]
        for _ in range(repetitions)
    ]
    report = {
        **summarize_times(times),
        "peak_allocation_bytes": int(torch.cuda.max_memory_allocated()),
    }
    return report, times


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
    inputs = synthetic_inputs(device=device)
    full_groom = synthetic_groom(device=device)
    disabled_groom = clone_groom(full_groom)

    for _ in range(warmup):
        evaluate_once(inputs, full_groom, enable_curl=None, capture=False)
    for _ in range(warmup):
        evaluate_once(inputs, disabled_groom, enable_curl=False, capture=False)

    full_scalar, full_output, full_gradients, _ = evaluate_once(
        inputs,
        full_groom,
        enable_curl=None,
        capture=True,
    )
    disabled_scalar, disabled_output, disabled_gradients, _ = evaluate_once(
        inputs,
        disabled_groom,
        enable_curl=False,
        capture=True,
    )
    if full_output is None or disabled_output is None:
        raise RuntimeError("benchmark reference capture unexpectedly returned no output")
    output_errors = [
        float((full_value - disabled_value).abs().max().cpu())
        for full_value, disabled_value in zip(full_output, disabled_output)
    ]
    gradient_errors = {
        name: float(
            (full_gradients[name] - disabled_gradients[name]).abs().max().cpu()
        )
        for name in ACTIVE_GRADIENT_NAMES
        if full_gradients[name] is not None and disabled_gradients[name] is not None
    }
    full_curl_gradients = {
        name: (
            float(full_gradients[name].abs().max().cpu())
            if full_gradients[name] is not None
            else None
        )
        for name in CURL_GRADIENT_NAMES
    }

    full_report, _ = measure_variant(
        inputs,
        full_groom,
        enable_curl=None,
        repetitions=repetitions,
    )
    disabled_report, _ = measure_variant(
        inputs,
        disabled_groom,
        enable_curl=False,
        repetitions=repetitions,
    )
    report: dict[str, object] = {
        "root_count": ROOT_COUNT,
        "samples": SAMPLES,
        "warmup_repetitions": int(warmup),
        "measured_repetitions": int(repetitions),
        "device": torch.cuda.get_device_name(device),
        "current_full_path": full_report,
        "explicit_disabled_path": disabled_report,
        "errors": {
            "scalar_max_abs_error": abs(full_scalar - disabled_scalar),
            "output_max_abs_error": max(output_errors),
            "output_max_abs_error_by_field": {
                name: error
                for name, error in zip(
                    ("points", "widths", "colors", "opacities"),
                    output_errors,
                )
            },
            "gradient_max_abs_error": max(gradient_errors.values()),
            "gradient_max_abs_error_by_field": gradient_errors,
            "full_path_curl_gradient_max_abs_by_field": full_curl_gradients,
            "disabled_path_curl_gradients_are_none": all(
                disabled_gradients[name] is None for name in CURL_GRADIENT_NAMES
            ),
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

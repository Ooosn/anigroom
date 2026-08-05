from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tools import train_white_tiger_stage1 as stage1


def test_memory_guard_ignores_released_reserved_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stage1,
        "cuda_memory_guard_payload",
        lambda _device: {
            "memory_allocated_mb": 2048.0,
            "memory_reserved_mb": 4096.0,
            "max_memory_allocated_mb": 25600.0,
            "max_memory_reserved_mb": 32768.0,
            "nvidia_smi_process_mb": 4800.0,
        },
    )

    stage1.enforce_cuda_memory_guard(
        SimpleNamespace(gpu_memory_limit_gb=30.0),
        torch.device("cpu"),
        iteration=19460,
        stage="iteration_start",
    )


def test_memory_guard_does_not_count_pytorch_cache_as_external_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage1,
        "cuda_memory_guard_payload",
        lambda _device: {
            "memory_allocated_mb": 2589.0,
            "memory_reserved_mb": 30340.0,
            "max_memory_allocated_mb": 25474.0,
            "max_memory_reserved_mb": 30340.0,
            "nvidia_smi_process_mb": 31074.0,
        },
    )

    stage1.enforce_cuda_memory_guard(
        SimpleNamespace(gpu_memory_limit_gb=30.0),
        torch.device("cpu"),
        iteration=19000,
        stage="iteration_start",
    )


def test_memory_guard_reports_real_allocation_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stage1,
        "cuda_memory_guard_payload",
        lambda _device: {
            "memory_allocated_mb": 2048.0,
            "memory_reserved_mb": 4096.0,
            "max_memory_allocated_mb": 31744.0,
            "max_memory_reserved_mb": 32768.0,
            "nvidia_smi_process_mb": 4800.0,
        },
    )
    events: list[tuple[str, dict[str, object]]] = []

    with pytest.raises(RuntimeError, match="GPU memory limit exceeded"):
        stage1.enforce_cuda_memory_guard(
            SimpleNamespace(gpu_memory_limit_gb=30.0),
            torch.device("cpu"),
            iteration=19460,
            stage="iteration_start",
            progress_event=lambda event, **payload: events.append((event, payload)),
        )

    assert events == [
        (
            "gpu_memory_limit_exceeded",
            {
                "iteration": 19460,
                "guard_stage": "iteration_start",
                "limit_gb": 30.0,
                "tracked_mb": 32448.0,
                "process_tracked_mb": 32448.0,
                "allocated_peak_mb": 31744.0,
                "external_process_overhead_mb": 704.0,
                "memory_allocated_mb": 2048.0,
                "memory_reserved_mb": 4096.0,
                "max_memory_allocated_mb": 31744.0,
                "max_memory_reserved_mb": 32768.0,
                "nvidia_smi_process_mb": 4800.0,
            },
        )
    ]


def test_memory_guard_counts_cuda_memory_outside_pytorch_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage1,
        "cuda_memory_guard_payload",
        lambda _device: {
            "memory_allocated_mb": 6144.0,
            "memory_reserved_mb": 8192.0,
            "max_memory_allocated_mb": 10240.0,
            "max_memory_reserved_mb": 12288.0,
            "nvidia_smi_process_mb": 32768.0,
        },
    )

    with pytest.raises(RuntimeError, match="GPU memory limit exceeded"):
        stage1.enforce_cuda_memory_guard(
            SimpleNamespace(gpu_memory_limit_gb=30.0),
            torch.device("cpu"),
            iteration=19000,
            stage="iteration_start",
        )


def test_memory_guard_does_not_charge_other_wddm_processes_to_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage1,
        "cuda_memory_guard_payload",
        lambda _device: {
            "memory_allocated_mb": 18432.0,
            "memory_reserved_mb": 23808.0,
            "max_memory_allocated_mb": 18432.0,
            "max_memory_reserved_mb": 23808.0,
            "nvidia_smi_process_mb": 0.0,
            "device_used_mb": 31744.0,
            "device_free_mb": 1024.0,
            "device_total_mb": 32768.0,
        },
    )

    stage1.enforce_cuda_memory_guard(
        SimpleNamespace(gpu_memory_limit_gb=30.0),
        torch.device("cpu"),
        iteration=19000,
        stage="iteration_start",
    )


def test_memory_guard_releases_unused_cache_before_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = iter(
        [
            {
                "memory_allocated_mb": 3072.0,
                "memory_reserved_mb": 30720.0,
                "max_memory_allocated_mb": 18432.0,
                "max_memory_reserved_mb": 30720.0,
                "nvidia_smi_process_mb": 0.0,
                "device_used_mb": 31744.0,
                "device_free_mb": 1024.0,
                "device_total_mb": 32768.0,
            },
            {
                "memory_allocated_mb": 3072.0,
                "memory_reserved_mb": 4096.0,
                "max_memory_allocated_mb": 18432.0,
                "max_memory_reserved_mb": 30720.0,
                "nvidia_smi_process_mb": 0.0,
                "device_used_mb": 5120.0,
                "device_free_mb": 27648.0,
                "device_total_mb": 32768.0,
            },
        ]
    )
    released: list[bool] = []
    events: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(stage1, "cuda_memory_guard_payload", lambda _device: next(payloads))
    monkeypatch.setattr(stage1, "release_cuda_cache", lambda: released.append(True))

    stage1.enforce_cuda_memory_guard(
        SimpleNamespace(gpu_memory_limit_gb=25.0),
        torch.device("cpu"),
        iteration=9620,
        stage="iteration_start",
        progress_event=lambda event, **payload: events.append((event, payload)),
    )

    assert released == [True]
    assert events[0][0] == "gpu_memory_cache_released"
    assert events[0][1]["tracked_before_release_mb"] == 30720.0
    assert events[0][1]["tracked_after_release_mb"] == 18432.0

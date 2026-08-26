from __future__ import annotations

import importlib

import pytest
import torch


MODULE_NAME = "tools.fuse_gpt_flow_shell_multiview"


def _module() -> object:
    return importlib.import_module(MODULE_NAME)


def _observability(basis: torch.Tensor, axis: torch.Tensor) -> float:
    module = _module()
    screen_t = basis[:, 0].unsqueeze(0)
    screen_b = basis[:, 1].unsqueeze(0)
    sampled_ori = axis.unsqueeze(0)
    return float(module.tangent_axis_observability(sampled_ori, screen_t, screen_b)[0])


def test_flow_shell_module_imports_with_observability_helper() -> None:
    module = _module()
    assert callable(module.main)
    assert callable(module.tangent_axis_observability)


@pytest.mark.parametrize(
    ("basis", "axis", "expected"),
    [
        (torch.eye(2), torch.tensor([1.0, 0.0]), 1.0),
        (
            torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            torch.tensor([1.0, 0.0]),
            1.0,
        ),
        (
            torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            torch.tensor([0.0, 1.0]),
            0.0,
        ),
        (
            torch.tensor([[1.0, 0.0], [0.0, 0.1]]),
            torch.tensor([0.0, 1.0]),
            0.1,
        ),
    ],
)
def test_tangent_axis_observability_cases(
    basis: torch.Tensor,
    axis: torch.Tensor,
    expected: float,
) -> None:
    assert _observability(basis, axis) == pytest.approx(expected, abs=1.0e-5)

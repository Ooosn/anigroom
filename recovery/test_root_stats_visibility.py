"""Portable validation for the gsplat radii visibility contract.

Run this file from an AniGroom checkout after applying
0001-root-stats-portable-radii.patch. It deliberately tests only the
RootStatsWindow visibility layout semantics; it does not run training.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anigroom.roots.statistics import RootStatsWindow  # noqa: E402


def assert_visibility(radii: torch.Tensor, expected: list[float]) -> None:
    actual = RootStatsWindow._visible_from_info(
        {"radii": radii},
        gaussian_count=len(expected),
        device=torch.device("cpu"),
    )
    assert actual.shape == (len(expected), 1)
    assert actual.tolist() == [[value] for value in expected], actual


def main() -> None:
    assert_visibility(torch.tensor([-1.0, 0.0, 2.0]), [0.0, 0.0, 1.0])
    assert_visibility(torch.tensor([[0.0, 3.0, 0.0]]), [0.0, 1.0, 0.0])
    assert_visibility(
        torch.tensor([[1.0, 0.0], [0.0, 0.0], [-1.0, 4.0]]),
        [1.0, 0.0, 1.0],
    )
    try:
        RootStatsWindow._visible_from_info(
            {"radii": torch.ones((2, 3))},
            gaussian_count=3,
            device=torch.device("cpu"),
        )
    except RuntimeError as error:
        assert "radii shape" in str(error), error
    else:
        raise AssertionError("invalid radii layout was accepted")
    print("root_stats_visibility: passed")


if __name__ == "__main__":
    main()

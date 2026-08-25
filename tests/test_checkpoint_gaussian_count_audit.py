from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from tools.audit_white_tiger_checkpoint_gaussian_count import (
    atomic_write_json,
    derive_per_root_segment_counts,
    pre_step_training_metric_minus_checkpoint_state,
    require_exact_repeat_results,
    repeat_fingerprint,
)


ROOT_INDICES = np.asarray([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
SEGMENT_INDICES = np.asarray([0, 1, 2, 0, 1, 0, 1, 2, 3], dtype=np.int64)


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def test_derive_counts_histogram_and_canonical_order_hashes() -> None:
    result = derive_per_root_segment_counts(
        torch.from_numpy(ROOT_INDICES),
        torch.from_numpy(SEGMENT_INDICES),
        root_count=3,
    )

    assert result["total_gaussian_count"] == 9
    assert result["root_count"] == 3
    np.testing.assert_array_equal(result["per_root_segment_counts"], [3, 2, 4])
    assert result["segment_histogram"] == {"2": 1, "3": 1, "4": 1}
    assert result["per_root_segment_counts_sha256"] == _sha256_array(
        np.asarray([3, 2, 4], dtype=np.int64)
    )
    assert result["root_indices_order_sha256"] == _sha256_array(ROOT_INDICES)
    assert result["segment_indices_order_sha256"] == _sha256_array(SEGMENT_INDICES)
    assert result["root_segment_order_sha256"] == _sha256_array(
        np.column_stack((ROOT_INDICES, SEGMENT_INDICES))
    )


def test_order_hash_changes_when_pairs_are_reordered() -> None:
    reordered = np.asarray([2, 2, 2, 2, 0, 0, 0, 1, 1], dtype=np.int64)
    reordered_segments = np.asarray([3, 2, 1, 0, 2, 1, 0, 1, 0], dtype=np.int64)
    first = derive_per_root_segment_counts(ROOT_INDICES, SEGMENT_INDICES, root_count=3)
    second = derive_per_root_segment_counts(reordered, reordered_segments, root_count=3)

    np.testing.assert_array_equal(
        first["per_root_segment_counts"], second["per_root_segment_counts"]
    )
    assert first["per_root_segment_counts_sha256"] == second[
        "per_root_segment_counts_sha256"
    ]
    assert first["root_segment_order_sha256"] != second["root_segment_order_sha256"]


def test_noncontiguous_segment_order_is_rejected_without_adjustment() -> None:
    with pytest.raises(RuntimeError, match="not a contiguous per-root order"):
        derive_per_root_segment_counts(
            np.asarray([0, 0, 1], dtype=np.int64),
            np.asarray([0, 2, 0], dtype=np.int64),
            root_count=2,
        )


def test_repeat_fingerprint_requires_exact_match() -> None:
    result = derive_per_root_segment_counts(ROOT_INDICES, SEGMENT_INDICES, root_count=3)
    assert require_exact_repeat_results([result, result]) == repeat_fingerprint(result)

    changed = dict(result)
    changed["total_gaussian_count"] = 10
    with pytest.raises(RuntimeError, match="repeats differ"):
        require_exact_repeat_results([result, changed])


def test_training_metric_delta_is_labeled_and_nonfatal() -> None:
    assert pre_step_training_metric_minus_checkpoint_state(5_382_959, 5_382_896) == 63
    assert pre_step_training_metric_minus_checkpoint_state(None, 5_382_896) is None


def test_atomic_json_replaces_output_without_leaving_temp(tmp_path) -> None:
    output = tmp_path / "audit.json"
    atomic_write_json(output, {"status": "pass", "count": 9})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "pass",
        "count": 9,
    }
    assert list(tmp_path.glob("*.tmp")) == []

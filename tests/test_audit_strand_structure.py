from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.audit_strand_structure import audit_strands


def test_audit_strands_detects_only_the_folded_curve(tmp_path: Path) -> None:
    strands = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.5], [1.0, 0.0, 1.0]],
            [[2.0, 0.0, 0.0], [2.0, 0.0, 1.0], [2.0, 0.0, 0.5]],
        ],
        dtype=np.float32,
    )
    path = tmp_path / "strands.npz"
    np.savez_compressed(path, strands=strands)

    report = audit_strands(path, neighbor_count=1)

    assert report["strand_count"] == 3
    assert report["samples"] == 3
    assert report["strands_with_backward_segment"] == 1
    assert report["backward_segment_fraction"] == 1.0 / 6.0
    assert report["arc_chord_ratio"]["max"] == 3.0

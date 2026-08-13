from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def normalized(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(
        np.linalg.norm(values, axis=-1, keepdims=True), 1.0e-12
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p999": float(np.quantile(values, 0.999)),
        "max": float(values.max()),
    }


def audit_strands(
    path: Path, *, neighbor_count: int = 4
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        if "strands" not in archive:
            raise KeyError(f"{path} does not contain a 'strands' array")
        strands = archive["strands"].astype(np.float64)
    if strands.ndim != 3 or strands.shape[1] < 3 or strands.shape[2] != 3:
        raise ValueError(
            "strands must have shape [strand_count, samples>=3, 3], got "
            f"{strands.shape}"
        )
    if strands.shape[0] <= neighbor_count:
        raise ValueError(
            f"neighbor_count={neighbor_count} requires more than "
            f"{neighbor_count} strands, got {strands.shape[0]}"
        )

    segments = np.diff(strands, axis=1)
    segment_length = np.linalg.norm(segments, axis=-1)
    arc_length = segment_length.sum(axis=1)
    chord = strands[:, -1] - strands[:, 0]
    chord_length = np.linalg.norm(chord, axis=-1)
    chord_unit = normalized(chord)
    backward = np.einsum("nsd,nd->ns", segments, chord_unit) < -1.0e-10

    first = segments[:, :-1]
    second = segments[:, 1:]
    cosine = np.einsum("nsd,nsd->ns", first, second) / np.maximum(
        np.linalg.norm(first, axis=-1)
        * np.linalg.norm(second, axis=-1),
        1.0e-12,
    )
    maximum_local_turn = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    ).max(axis=1)
    arc_chord = arc_length / np.maximum(chord_length, 1.0e-12)

    roots = strands[:, 0]
    _, neighbors = cKDTree(roots).query(roots, k=neighbor_count + 1)
    neighbors = np.asarray(neighbors)[:, 1:]
    neighbor_length = arc_length[neighbors]
    relative_length_difference = np.abs(
        neighbor_length - arc_length[:, None]
    ) / np.maximum(
        0.5 * (neighbor_length + arc_length[:, None]), 1.0e-12
    )
    neighbor_direction = chord_unit[neighbors]
    direction_cosine = np.einsum(
        "nkd,nd->nk", neighbor_direction, chord_unit
    )
    direction_difference = np.degrees(
        np.arccos(np.clip(direction_cosine, -1.0, 1.0))
    )

    prefix = f"local_{neighbor_count}nn"
    return {
        "path": str(path.resolve()),
        "strand_count": int(strands.shape[0]),
        "samples": int(strands.shape[1]),
        f"{prefix}_relative_length_difference_mean": float(
            relative_length_difference.mean()
        ),
        f"{prefix}_relative_length_difference_p95": float(
            np.quantile(relative_length_difference, 0.95)
        ),
        f"{prefix}_chord_direction_difference_mean_degrees": float(
            direction_difference.mean()
        ),
        f"{prefix}_chord_direction_difference_p95_degrees": float(
            np.quantile(direction_difference, 0.95)
        ),
        "backward_segment_fraction": float(backward.mean()),
        "strands_with_backward_segment": int(backward.any(axis=1).sum()),
        "chord_length": quantiles(chord_length),
        "arc_length": quantiles(arc_length),
        "arc_length_tail_counts": {
            ">0.12": int((arc_length > 0.12).sum()),
            ">0.15": int((arc_length > 0.15).sum()),
            ">0.20": int((arc_length > 0.20).sum()),
        },
        "arc_chord_ratio": quantiles(arc_chord),
        "maximum_local_turn_degrees": quantiles(maximum_local_turn),
    }


def parse_named_input(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "--input must use LABEL=PATH syntax"
        )
    return label.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit fixed-protocol strand NPZ exports."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_named_input,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbor-count", type=int, default=4)
    args = parser.parse_args()

    if args.neighbor_count < 1:
        parser.error("--neighbor-count must be positive")
    labels = [label for label, _ in args.input]
    if len(labels) != len(set(labels)):
        parser.error("--input labels must be unique")
    missing = [str(path) for _, path in args.input if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing strand exports: {missing}")

    report = {
        label: audit_strands(path, neighbor_count=args.neighbor_count)
        for label, path in args.input
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

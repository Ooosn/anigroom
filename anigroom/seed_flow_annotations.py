"""Sparse 2D seed directions and fast anchor-driven interpolation.

The seed format deliberately stores no endpoint or hair length.  Arrow size is
an editor display setting; reconstruction input is only image position,
directed unit flow, and whether the user explicitly edited the seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .flow_annotations import (
    SCHEMA_NAME as LEGACY_SCHEMA_NAME,
    load_annotations as load_legacy_annotations,
    pixel_to_uv,
    sha256_file,
)


SCHEMA_NAME = "anigroom.seed_flow.v1"
_DOCUMENT_KEYS = {
    "schema",
    "image_filename",
    "width",
    "height",
    "sha256",
    "updated_utc",
    "seeds",
}
_SEED_KEYS = {"id", "position_px", "position_uv", "direction_px", "manual"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EPS = 1.0e-12


class SeedFlowValidationError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SeedFlowValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_pair(value: object, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SeedFlowValidationError(f"{label} must contain two numbers")
    try:
        pair = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise SeedFlowValidationError(f"{label} must contain two numbers") from exc
    if not all(math.isfinite(component) for component in pair):
        raise SeedFlowValidationError(f"{label} must be finite")
    return pair


def _unit_direction(value: object, *, strict: bool) -> tuple[float, float]:
    x, y = _finite_pair(value, label="direction_px")
    norm = math.hypot(x, y)
    if norm <= _EPS:
        raise SeedFlowValidationError("direction_px must be non-zero")
    if strict and abs(norm - 1.0) > 1.0e-5:
        raise SeedFlowValidationError("direction_px must be unit length")
    return x / norm, y / norm


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class FlowSeed:
    id: str
    position_px: tuple[float, float]
    position_uv: tuple[float, float]
    direction_px: tuple[float, float]
    manual: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "position_px": list(self.position_px),
            "position_uv": list(self.position_uv),
            "direction_px": list(self.direction_px),
            "manual": self.manual,
        }

    @classmethod
    def from_dict(cls, value: object) -> "FlowSeed":
        if not isinstance(value, Mapping) or set(value) != _SEED_KEYS:
            raise SeedFlowValidationError("seed has missing or unknown fields")
        seed_id = value["id"]
        if not isinstance(seed_id, str) or not seed_id or "\x00" in seed_id:
            raise SeedFlowValidationError("seed id must be a non-empty string")
        manual = value["manual"]
        if not isinstance(manual, bool):
            raise SeedFlowValidationError("manual must be boolean")
        return cls(
            id=seed_id,
            position_px=_finite_pair(value["position_px"], label="position_px"),
            position_uv=_finite_pair(value["position_uv"], label="position_uv"),
            direction_px=_unit_direction(value["direction_px"], strict=True),
            manual=manual,
        )


def make_flow_seed(
    seed_id: str,
    position_px: Iterable[float],
    direction_px: Iterable[float],
    width: int,
    height: int,
    *,
    manual: bool,
) -> FlowSeed:
    position = _finite_pair(tuple(position_px), label="position_px")
    direction = _unit_direction(tuple(direction_px), strict=False)
    if not (0.0 <= position[0] <= width - 1 and 0.0 <= position[1] <= height - 1):
        raise SeedFlowValidationError("position_px lies outside the image")
    return FlowSeed(
        id=seed_id,
        position_px=position,
        position_uv=pixel_to_uv(position, width, height),
        direction_px=direction,
        manual=bool(manual),
    )


@dataclass(frozen=True)
class SeedFlowAnnotations:
    image_filename: str
    width: int
    height: int
    sha256: str
    seeds: tuple[FlowSeed, ...]
    updated_utc: str

    def __post_init__(self) -> None:
        if not isinstance(self.image_filename, str) or not self.image_filename:
            raise SeedFlowValidationError("image_filename must be non-empty")
        if Path(self.image_filename).name != self.image_filename:
            raise SeedFlowValidationError("image_filename must not contain a directory")
        if not isinstance(self.width, int) or not isinstance(self.height, int) or self.width < 2 or self.height < 2:
            raise SeedFlowValidationError("width and height must be integers of at least two")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise SeedFlowValidationError("sha256 must be a lowercase hexadecimal digest")
        if not isinstance(self.updated_utc, str) or not self.updated_utc.endswith("Z"):
            raise SeedFlowValidationError("updated_utc must be UTC and end with Z")
        ids: set[str] = set()
        for seed in self.seeds:
            if not isinstance(seed, FlowSeed):
                raise SeedFlowValidationError("seeds must contain FlowSeed records")
            if seed.id in ids:
                raise SeedFlowValidationError(f"duplicate seed id: {seed.id}")
            ids.add(seed.id)
            x, y = seed.position_px
            if not (0.0 <= x <= self.width - 1 and 0.0 <= y <= self.height - 1):
                raise SeedFlowValidationError(f"seed {seed.id} lies outside the image")
            expected_uv = pixel_to_uv(seed.position_px, self.width, self.height)
            if max(abs(expected_uv[i] - seed.position_uv[i]) for i in range(2)) > 1.0e-6:
                raise SeedFlowValidationError(f"seed {seed.id} position_uv does not match position_px")
            _unit_direction(seed.direction_px, strict=True)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA_NAME,
            "image_filename": self.image_filename,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
            "updated_utc": self.updated_utc,
            "seeds": [seed.to_dict() for seed in self.seeds],
        }

    @classmethod
    def from_dict(cls, value: object) -> "SeedFlowAnnotations":
        if not isinstance(value, Mapping) or set(value) != _DOCUMENT_KEYS:
            raise SeedFlowValidationError("document has missing or unknown fields")
        if value["schema"] != SCHEMA_NAME:
            raise SeedFlowValidationError(f"unknown schema: {value['schema']!r}")
        if not isinstance(value["seeds"], list):
            raise SeedFlowValidationError("seeds must be a JSON array")
        return cls(
            image_filename=value["image_filename"],
            width=value["width"],
            height=value["height"],
            sha256=value["sha256"],
            updated_utc=value["updated_utc"],
            seeds=tuple(FlowSeed.from_dict(seed) for seed in value["seeds"]),
        )

    @classmethod
    def from_image(
        cls,
        image_path: str | Path,
        width: int,
        height: int,
        seeds: Iterable[FlowSeed],
        *,
        updated_utc: str | None = None,
    ) -> "SeedFlowAnnotations":
        path = Path(image_path)
        return cls(
            image_filename=path.name,
            width=int(width),
            height=int(height),
            sha256=sha256_file(path),
            seeds=tuple(seeds),
            updated_utc=updated_utc or _utc_now(),
        )


def _annotation_path(destination: str | Path, image_filename: str) -> Path:
    path = Path(destination)
    if path.suffix.lower() == ".json":
        return path
    return path / f"{Path(image_filename).stem}.flow.json"


def save_seed_flow_annotations(
    annotations: SeedFlowAnnotations,
    destination: str | Path,
) -> Path:
    target = _annotation_path(destination, annotations.image_filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    payload = json.dumps(annotations.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(target)
    return target


def _legacy_to_seed_document(path: Path, image_path: Path | None, verify_image: bool) -> SeedFlowAnnotations:
    legacy = load_legacy_annotations(path, image_path=image_path, verify_hash=verify_image)
    return SeedFlowAnnotations(
        image_filename=legacy.image_filename,
        width=legacy.width,
        height=legacy.height,
        sha256=legacy.sha256,
        updated_utc=legacy.updated_utc,
        seeds=tuple(
            make_flow_seed(
                arrow.id,
                arrow.start_px,
                arrow.direction_px,
                legacy.width,
                legacy.height,
                manual=True,
            )
            for arrow in legacy.arrows
        ),
    )


def load_seed_flow_annotations(
    source: str | Path,
    *,
    image_path: str | Path | None = None,
    verify_image: bool = False,
) -> SeedFlowAnnotations:
    path = Path(source)
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise SeedFlowValidationError(f"cannot read seed-flow JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SeedFlowValidationError("seed-flow document must be a JSON object")
    if value.get("schema") == LEGACY_SCHEMA_NAME:
        return _legacy_to_seed_document(path, Path(image_path) if image_path else None, verify_image)
    annotations = SeedFlowAnnotations.from_dict(value)
    if image_path is not None:
        image = Path(image_path)
        if image.name != annotations.image_filename:
            raise SeedFlowValidationError("annotation image_filename does not match image path")
        if verify_image and sha256_file(image) != annotations.sha256:
            raise SeedFlowValidationError("annotation image hash does not match image file")
    return annotations


def scan_seed_flow_directory(directory: str | Path) -> dict[str, SeedFlowAnnotations]:
    result: dict[str, SeedFlowAnnotations] = {}
    for path in sorted(Path(directory).glob("*.flow.json")):
        annotations = load_seed_flow_annotations(path)
        if annotations.image_filename in result:
            raise SeedFlowValidationError(f"duplicate annotation for {annotations.image_filename}")
        result[annotations.image_filename] = annotations
    return result


@dataclass(frozen=True)
class SeedNeighborGraph:
    indices: np.ndarray
    weights: np.ndarray

    @property
    def count(self) -> int:
        return int(self.indices.shape[0])


def _positions_array(positions: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    value = np.asarray(positions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("positions must be finite with shape [N,2]")
    return value


def _directions_array(directions: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    value = np.asarray(directions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("directions must be finite with shape [N,2]")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= _EPS):
        raise ValueError("directions must be non-zero")
    return value / norms


def build_seed_neighbor_graph(
    positions: np.ndarray | Iterable[Iterable[float]],
    *,
    neighbor_count: int = 8,
) -> SeedNeighborGraph:
    points = _positions_array(positions)
    count = int(points.shape[0])
    if count <= 1:
        return SeedNeighborGraph(
            indices=np.empty((count, 0), dtype=np.int64),
            weights=np.empty((count, 0), dtype=np.float64),
        )
    k = min(max(1, int(neighbor_count)), count - 1)
    distances, indices = cKDTree(points).query(points, k=k + 1, workers=-1)
    distances = np.asarray(distances[:, 1:], dtype=np.float64)
    indices = np.asarray(indices[:, 1:], dtype=np.int64)
    scale = np.maximum(distances[:, [-1]], 1.0e-6)
    weights = np.exp(-np.square(distances / scale))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), _EPS)
    return SeedNeighborGraph(indices=indices, weights=weights)


def graph_ring_indices(
    graph: SeedNeighborGraph,
    changed_indices: Iterable[int],
    *,
    rings: int,
) -> np.ndarray:
    frontier = {int(index) for index in changed_indices if 0 <= int(index) < graph.count}
    visited = set(frontier)
    for _ in range(max(0, int(rings))):
        if not frontier:
            break
        next_frontier: set[int] = set()
        for index in frontier:
            next_frontier.update(int(value) for value in graph.indices[index])
        next_frontier.difference_update(visited)
        visited.update(next_frontier)
        frontier = next_frontier
    return np.asarray(sorted(visited), dtype=np.int64)


def propagate_follower_directions(
    directions: np.ndarray | Iterable[Iterable[float]],
    manual_mask: np.ndarray | Iterable[bool],
    graph: SeedNeighborGraph,
    *,
    changed_indices: Iterable[int] | None = None,
    rings: int = 6,
    iterations: int = 8,
    relaxation: float = 0.72,
) -> tuple[np.ndarray, np.ndarray]:
    output = _directions_array(directions)
    manual = np.asarray(manual_mask, dtype=bool).reshape(-1)
    if len(output) != graph.count or len(manual) != graph.count:
        raise ValueError("directions, manual_mask, and graph must have matching counts")
    if graph.count <= 1 or graph.indices.shape[1] == 0:
        return output, np.empty((0,), dtype=np.int64)
    if changed_indices is None:
        active = np.flatnonzero(~manual)
    else:
        local = graph_ring_indices(graph, changed_indices, rings=rings)
        active = local[~manual[local]]
    if active.size == 0:
        return output, active
    blend = min(1.0, max(0.0, float(relaxation)))
    anchors = output[manual].copy()
    for _ in range(max(0, int(iterations))):
        neighbor_average = (
            output[graph.indices[active]] * graph.weights[active, :, None]
        ).sum(axis=1)
        candidate = (1.0 - blend) * output[active] + blend * neighbor_average
        norms = np.linalg.norm(candidate, axis=1, keepdims=True)
        valid = norms[:, 0] > _EPS
        output[active[valid]] = candidate[valid] / norms[valid]
        output[manual] = anchors
    return output, active


def nearest_seed_direction(
    position: Iterable[float],
    positions: np.ndarray | Iterable[Iterable[float]],
    directions: np.ndarray | Iterable[Iterable[float]],
    *,
    neighbor_count: int = 8,
    fallback: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    points = _positions_array(positions)
    vectors = _directions_array(directions)
    if len(points) != len(vectors):
        raise ValueError("positions and directions must have matching counts")
    if len(points) == 0:
        return _unit_direction(fallback, strict=False)
    query = np.asarray(tuple(position), dtype=np.float64)
    if query.shape != (2,) or not np.isfinite(query).all():
        raise ValueError("position must be a finite pair")
    k = min(max(1, int(neighbor_count)), len(points))
    distances, indices = cKDTree(points).query(query, k=k)
    distances = np.atleast_1d(distances).astype(np.float64)
    indices = np.atleast_1d(indices).astype(np.int64)
    weights = 1.0 / np.maximum(distances, 1.0e-3)
    combined = (vectors[indices] * weights[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(combined))
    if norm <= _EPS:
        return _unit_direction(fallback, strict=False)
    return float(combined[0] / norm), float(combined[1] / norm)

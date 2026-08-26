"""Strict, standard-library sparse 2D flow annotations.

The on-disk format is one JSON sidecar per image.  A sidecar named
``<stem>.flow.json`` contains both pixel and normalized coordinates so that an
annotator can be inspected without losing the original image-space evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import ClassVar, TypeAlias


SCHEMA_NAME = "anigroom.sparse_flow.v1"
FLOW_FILE_SUFFIX = ".flow.json"
Point2D: TypeAlias = tuple[float, float]

_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
_REQUIRED_DOCUMENT_KEYS = {
    "schema",
    "image_filename",
    "width",
    "height",
    "sha256",
    "arrows",
    "updated_utc",
}
_REQUIRED_ARROW_KEYS = {
    "id",
    "start_px",
    "end_px",
    "start_uv",
    "end_uv",
    "confidence",
    "root_to_tip",
}


class SparseFlowError(ValueError):
    """Base error for invalid sparse-flow data."""


class SparseFlowValidationError(SparseFlowError):
    """Raised when a sparse-flow document or arrow violates the schema."""


def _error(message: str) -> SparseFlowValidationError:
    return SparseFlowValidationError(message)


def _validate_dimensions(width: object, height: object) -> tuple[int, int]:
    if type(width) is not int or type(height) is not int:
        raise _error("width and height must be integers")
    if width < 2 or height < 2:
        raise _error("width and height must both be at least 2")
    return width, height


def _coerce_point(value: object, *, label: str) -> Point2D:
    if isinstance(value, (str, bytes, bytearray)):
        raise _error(f"{label} must contain exactly two numbers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise _error(f"{label} must contain exactly two numbers") from exc
    if len(values) != 2:
        raise _error(f"{label} must contain exactly two numbers")
    result: list[float] = []
    for component in values:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise _error(f"{label} must contain only numbers")
        number = float(component)
        if not math.isfinite(number):
            raise _error(f"{label} must contain only finite numbers")
        result.append(number)
    return result[0], result[1]


def _validate_pixel_point(point: Point2D, width: int, height: int, *, label: str) -> None:
    x, y = point
    if not (0.0 <= x <= float(width - 1) and 0.0 <= y <= float(height - 1)):
        raise _error(f"{label} is outside the image bounds")


def _validate_uv_point(point: Point2D, *, label: str) -> None:
    if not (0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0):
        raise _error(f"{label} must be within [0, 1]")


def _validate_sha256(value: object, *, label: str = "sha256") -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise _error(f"{label} must be a 64-character hexadecimal SHA256 digest")
    return value.lower()


def _validate_updated_utc(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _error("updated_utc must be a UTC ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _error("updated_utc must be a UTC ISO-8601 string") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise _error("updated_utc must include a UTC timezone")
    return value


def _validate_image_filename(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error("image_filename must be a non-empty filename")
    if "\x00" in value or "/" in value or "\\" in value:
        raise _error("image_filename must be a filename, not a path")
    if value in {".", ".."}:
        raise _error("image_filename must be a regular filename")
    return value


def _validate_arrow_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error("arrow id must be a non-empty string")
    if "\x00" in value:
        raise _error("arrow id must not contain a NUL character")
    return value


def _validate_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error("confidence must be a number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise _error("confidence must be a number in [0, 1]")
    return result


def _validate_keys(value: object, required: set[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise _error(f"{label} must be a JSON object")
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise _error(f"{label} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise _error(f"{label} contains unknown keys: {', '.join(unknown)}")
    return value


def _points_match(expected: Point2D, actual: Point2D) -> bool:
    return all(
        math.isclose(expected[index], actual[index], rel_tol=1.0e-9, abs_tol=1.0e-9)
        for index in range(2)
    )


def utc_now() -> str:
    """Return a canonical UTC timestamp suitable for ``updated_utc``."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pixel_to_uv(point: Iterable[float], width: int, height: int) -> Point2D:
    """Convert an in-bounds pixel coordinate to normalized image coordinates."""

    width, height = _validate_dimensions(width, height)
    pixel = _coerce_point(point, label="pixel point")
    _validate_pixel_point(pixel, width, height, label="pixel point")
    return pixel[0] / float(width - 1), pixel[1] / float(height - 1)


def uv_to_pixel(point: Iterable[float], width: int, height: int) -> Point2D:
    """Convert a normalized image coordinate to a floating-point pixel coordinate."""

    width, height = _validate_dimensions(width, height)
    uv = _coerce_point(point, label="UV point")
    _validate_uv_point(uv, label="UV point")
    return uv[0] * float(width - 1), uv[1] * float(height - 1)


def normalized_direction_length(start: Iterable[float], end: Iterable[float]) -> tuple[Point2D, float]:
    """Return the normalized 2D direction and Euclidean length from start to end."""

    start_point = _coerce_point(start, label="start point")
    end_point = _coerce_point(end, label="end point")
    delta = (end_point[0] - start_point[0], end_point[1] - start_point[1])
    length = math.hypot(delta[0], delta[1])
    if length == 0.0:
        raise _error("an arrow must have non-zero length")
    return (delta[0] / length, delta[1] / length), length


def direction_and_length(start: Iterable[float], end: Iterable[float]) -> tuple[Point2D, float]:
    """Alias for :func:`normalized_direction_length`."""

    return normalized_direction_length(start, end)


@dataclass(frozen=True, slots=True)
class SparseArrow:
    """A directed, sparse image-space flow arrow."""

    id: str
    start_px: Point2D
    end_px: Point2D
    start_uv: Point2D
    end_uv: Point2D
    confidence: float
    root_to_tip: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_px", _coerce_point(self.start_px, label="start_px"))
        object.__setattr__(self, "end_px", _coerce_point(self.end_px, label="end_px"))
        object.__setattr__(self, "start_uv", _coerce_point(self.start_uv, label="start_uv"))
        object.__setattr__(self, "end_uv", _coerce_point(self.end_uv, label="end_uv"))
        _validate_arrow_id(self.id)
        _validate_uv_point(self.start_uv, label="start_uv")
        _validate_uv_point(self.end_uv, label="end_uv")
        _validate_confidence(self.confidence)
        if type(self.root_to_tip) is not bool or not self.root_to_tip:
            raise _error("root_to_tip must be true")
        if self.start_px == self.end_px:
            raise _error("an arrow must have non-zero length")

    @classmethod
    def from_pixels(
        cls,
        arrow_id: str,
        start_px: Iterable[float],
        end_px: Iterable[float],
        width: int,
        height: int,
        confidence: float,
        *,
        root_to_tip: bool = True,
    ) -> "SparseArrow":
        """Build an arrow and derive its normalized coordinates from pixels."""

        start = _coerce_point(start_px, label="start_px")
        end = _coerce_point(end_px, label="end_px")
        return cls(
            id=arrow_id,
            start_px=start,
            end_px=end,
            start_uv=pixel_to_uv(start, width, height),
            end_uv=pixel_to_uv(end, width, height),
            confidence=confidence,
            root_to_tip=root_to_tip,
        )

    @classmethod
    def from_uv(
        cls,
        arrow_id: str,
        start_uv: Iterable[float],
        end_uv: Iterable[float],
        width: int,
        height: int,
        confidence: float,
        *,
        root_to_tip: bool = True,
    ) -> "SparseArrow":
        """Build an arrow and derive its pixel coordinates from UV coordinates."""

        start = _coerce_point(start_uv, label="start_uv")
        end = _coerce_point(end_uv, label="end_uv")
        return cls(
            id=arrow_id,
            start_px=uv_to_pixel(start, width, height),
            end_px=uv_to_pixel(end, width, height),
            start_uv=start,
            end_uv=end,
            confidence=confidence,
            root_to_tip=root_to_tip,
        )

    @property
    def direction_px(self) -> Point2D:
        return normalized_direction_length(self.start_px, self.end_px)[0]

    @property
    def length_px(self) -> float:
        return normalized_direction_length(self.start_px, self.end_px)[1]

    @property
    def direction_uv(self) -> Point2D:
        return normalized_direction_length(self.start_uv, self.end_uv)[0]

    @property
    def length_uv(self) -> float:
        return normalized_direction_length(self.start_uv, self.end_uv)[1]

    @property
    def direction(self) -> Point2D:
        """The normalized direction in pixel coordinates."""

        return self.direction_px

    @property
    def length(self) -> float:
        """The Euclidean length in pixel coordinates."""

        return self.length_px

    def direction_and_length(self, *, space: str = "pixel") -> tuple[Point2D, float]:
        if space == "pixel":
            return normalized_direction_length(self.start_px, self.end_px)
        if space == "uv":
            return normalized_direction_length(self.start_uv, self.end_uv)
        raise _error("space must be 'pixel' or 'uv'")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "start_px": list(self.start_px),
            "end_px": list(self.end_px),
            "start_uv": list(self.start_uv),
            "end_uv": list(self.end_uv),
            "confidence": self.confidence,
            "root_to_tip": self.root_to_tip,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SparseArrow":
        data = _validate_keys(value, _REQUIRED_ARROW_KEYS, label="arrow")
        return cls(
            id=data["id"],  # type: ignore[arg-type]
            start_px=data["start_px"],  # type: ignore[arg-type]
            end_px=data["end_px"],  # type: ignore[arg-type]
            start_uv=data["start_uv"],  # type: ignore[arg-type]
            end_uv=data["end_uv"],  # type: ignore[arg-type]
            confidence=data["confidence"],  # type: ignore[arg-type]
            root_to_tip=data["root_to_tip"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class SparseFlowAnnotations:
    """A validated ``anigroom.sparse_flow.v1`` image sidecar."""

    image_filename: str
    width: int
    height: int
    sha256: str
    arrows: tuple[SparseArrow, ...]
    updated_utc: str = field(default_factory=utc_now)

    schema: ClassVar[str] = SCHEMA_NAME

    def __post_init__(self) -> None:
        image_filename = _validate_image_filename(self.image_filename)
        width, height = _validate_dimensions(self.width, self.height)
        digest = _validate_sha256(self.sha256)
        updated_utc = _validate_updated_utc(self.updated_utc)
        try:
            arrows = tuple(self.arrows)
        except TypeError as exc:
            raise _error("arrows must be an iterable of SparseArrow records") from exc
        if any(not isinstance(arrow, SparseArrow) for arrow in arrows):
            raise _error("arrows must contain only SparseArrow records")
        ids: set[str] = set()
        for arrow in arrows:
            if arrow.id in ids:
                raise _error(f"duplicate arrow id: {arrow.id}")
            ids.add(arrow.id)
            _validate_pixel_point(arrow.start_px, width, height, label=f"arrow {arrow.id} start_px")
            _validate_pixel_point(arrow.end_px, width, height, label=f"arrow {arrow.id} end_px")
            expected_start_uv = pixel_to_uv(arrow.start_px, width, height)
            expected_end_uv = pixel_to_uv(arrow.end_px, width, height)
            if not _points_match(expected_start_uv, arrow.start_uv):
                raise _error(f"arrow {arrow.id} start_uv does not match start_px")
            if not _points_match(expected_end_uv, arrow.end_uv):
                raise _error(f"arrow {arrow.id} end_uv does not match end_px")
        object.__setattr__(self, "image_filename", image_filename)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "arrows", arrows)
        object.__setattr__(self, "updated_utc", updated_utc)

    @classmethod
    def from_image(
        cls,
        image_path: str | Path,
        width: int,
        height: int,
        arrows: Iterable[SparseArrow],
        *,
        updated_utc: str | None = None,
    ) -> "SparseFlowAnnotations":
        """Create a document using an image's basename and SHA256 digest."""

        image = Path(image_path)
        return cls(
            image_filename=image.name,
            width=width,
            height=height,
            sha256=sha256_file(image),
            arrows=tuple(arrows),
            updated_utc=utc_now() if updated_utc is None else updated_utc,
        )

    @property
    def SHA256(self) -> str:
        """Compatibility spelling for callers that use the schema label."""

        return self.sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA_NAME,
            "image_filename": self.image_filename,
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
            "arrows": [arrow.to_dict() for arrow in self.arrows],
            "updated_utc": self.updated_utc,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SparseFlowAnnotations":
        data = _validate_keys(value, _REQUIRED_DOCUMENT_KEYS, label="sparse-flow document")
        if data["schema"] != SCHEMA_NAME:
            raise _error(f"unknown schema: {data['schema']!r}")
        if not isinstance(data["arrows"], list):
            raise _error("arrows must be a JSON array")
        arrows = tuple(SparseArrow.from_dict(item) for item in data["arrows"])
        return cls(
            image_filename=data["image_filename"],  # type: ignore[arg-type]
            width=data["width"],  # type: ignore[arg-type]
            height=data["height"],  # type: ignore[arg-type]
            sha256=data["sha256"],  # type: ignore[arg-type]
            arrows=arrows,
            updated_utc=data["updated_utc"],  # type: ignore[arg-type]
        )


ArrowAnnotation = SparseArrow
SparseFlowDocument = SparseFlowAnnotations
SparseFlowFile = SparseFlowAnnotations
FlowArrow = SparseArrow
ImageFlowAnnotations = SparseFlowAnnotations


def make_arrow(
    arrow_id: str,
    start_px: Iterable[float],
    end_px: Iterable[float],
    width: int,
    height: int,
    confidence: float,
    *,
    root_to_tip: bool = True,
) -> SparseArrow:
    """Convenience wrapper around :meth:`SparseArrow.from_pixels`."""

    return SparseArrow.from_pixels(
        arrow_id,
        start_px,
        end_px,
        width,
        height,
        confidence,
        root_to_tip=root_to_tip,
    )


def make_flow_arrow(
    arrow_id: str,
    start_px: Iterable[float],
    end_px: Iterable[float],
    width: int,
    height: int,
    confidence: float,
    *,
    root_to_tip: bool = True,
) -> FlowArrow:
    """Create a UI-facing :class:`FlowArrow` from pixel endpoints."""

    return make_arrow(
        arrow_id,
        start_px,
        end_px,
        width,
        height,
        confidence,
        root_to_tip=root_to_tip,
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA256 digest of a file."""

    if type(chunk_size) is not int or chunk_size <= 0:
        raise _error("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_path(directory: str | Path, image_filename: str) -> Path:
    """Return the sidecar path for an image filename."""

    filename = _validate_image_filename(image_filename)
    return Path(directory) / f"{Path(filename).stem}{FLOW_FILE_SUFFIX}"


def _resolve_save_path(destination: str | Path, image_filename: str) -> Path:
    target = Path(destination)
    if target.exists() and target.is_dir():
        return annotation_path(target, image_filename)
    if target.name.endswith(FLOW_FILE_SUFFIX):
        return target
    if target.exists() and not target.is_dir():
        raise _error("save destination must be a directory or a .flow.json file")
    return annotation_path(target, image_filename)


def save_annotations(annotations: SparseFlowAnnotations, destination: str | Path) -> Path:
    """Atomically write a document to its image sidecar path."""

    if not isinstance(annotations, SparseFlowAnnotations):
        raise _error("annotations must be a SparseFlowAnnotations record")
    target = _resolve_save_path(destination, annotations.image_filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(annotations.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary_path: str | None = None
    try:
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    return target


def _expected_dimension(value: object, *, label: str) -> int:
    if type(value) is not int or value < 2:
        raise _error(f"{label} must be an integer at least 2")
    return value


def _validate_expected_dimensions(
    expected_dimensions: Sequence[object] | None,
    expected_width: object | None,
    expected_height: object | None,
) -> tuple[int | None, int | None]:
    if expected_dimensions is not None:
        if isinstance(expected_dimensions, (str, bytes, bytearray)) or len(expected_dimensions) != 2:
            raise _error("expected_dimensions must contain width and height")
        dimension_width = _expected_dimension(expected_dimensions[0], label="expected width")
        dimension_height = _expected_dimension(expected_dimensions[1], label="expected height")
        if expected_width is not None and expected_width != dimension_width:
            raise _error("conflicting expected widths")
        if expected_height is not None and expected_height != dimension_height:
            raise _error("conflicting expected heights")
        expected_width, expected_height = dimension_width, dimension_height
    if expected_width is not None:
        expected_width = _expected_dimension(expected_width, label="expected width")
    if expected_height is not None:
        expected_height = _expected_dimension(expected_height, label="expected height")
    return expected_width, expected_height


def load_annotations(
    path: str | Path,
    image_path: str | Path | None = None,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_dimensions: Sequence[object] | None = None,
    expected_sha256: str | None = None,
    verify_hash: bool = False,
    verify_sha256: bool | None = None,
) -> SparseFlowAnnotations:
    """Load and strictly validate a sparse-flow sidecar.

    Expected dimensions and digest are optional checks.  Supplying
    ``image_path`` with ``verify_hash=True`` checks the stored digest against
    the image bytes; dimensions remain explicit because image decoding is not
    part of this standard-library module.
    """

    if verify_sha256 is not None:
        if verify_hash and verify_hash != verify_sha256:
            raise _error("conflicting hash-verification flags")
        verify_hash = verify_sha256
    expected_width, expected_height = _validate_expected_dimensions(
        expected_dimensions,
        expected_width,
        expected_height,
    )
    if verify_hash and image_path is None:
        raise _error("verify_hash requires image_path")
    if expected_sha256 is not None:
        expected_sha256 = _validate_sha256(expected_sha256, label="expected_sha256")

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, _StrictJSONError) as exc:
        raise _error(f"malformed JSON: {source}") from exc
    annotations = SparseFlowAnnotations.from_dict(value)

    if expected_width is not None and annotations.width != expected_width:
        raise _error(f"width mismatch: expected {expected_width}, got {annotations.width}")
    if expected_height is not None and annotations.height != expected_height:
        raise _error(f"height mismatch: expected {expected_height}, got {annotations.height}")
    if expected_sha256 is not None and annotations.sha256 != expected_sha256:
        raise _error("sha256 mismatch")
    if image_path is not None:
        image = Path(image_path)
        if image.name != annotations.image_filename:
            raise _error(
                f"image filename mismatch: expected {annotations.image_filename!r}, got {image.name!r}"
            )
        if verify_hash and sha256_file(image) != annotations.sha256:
            raise _error("sha256 does not match image bytes")
    return annotations


class _StrictJSONError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise _StrictJSONError(f"invalid JSON constant: {value}")


def scan_annotations(
    directory: str | Path,
    *,
    expected_dimensions: Mapping[str, Sequence[object]] | None = None,
    expected_sha256: Mapping[str, str] | None = None,
    image_directory: str | Path | None = None,
    verify_hash: bool = False,
    verify_sha256: bool | None = None,
) -> dict[str, SparseFlowAnnotations]:
    """Read all sidecars in a directory keyed by ``image_filename``."""

    if verify_sha256 is not None:
        if verify_hash and verify_hash != verify_sha256:
            raise _error("conflicting hash-verification flags")
        verify_hash = verify_sha256
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(root)
    image_root = root if image_directory is None else Path(image_directory)
    result: dict[str, SparseFlowAnnotations] = {}
    sidecars = sorted(
        (candidate for candidate in root.iterdir() if candidate.is_file() and candidate.name.endswith(FLOW_FILE_SUFFIX)),
        key=lambda candidate: candidate.name,
    )
    for sidecar in sidecars:
        stem = sidecar.name[: -len(FLOW_FILE_SUFFIX)]
        dimensions = expected_dimensions.get(stem) if expected_dimensions is not None else None
        digest = expected_sha256.get(stem) if expected_sha256 is not None else None
        annotations = load_annotations(
            sidecar,
            expected_dimensions=dimensions,
            expected_sha256=digest,
        )
        if Path(annotations.image_filename).stem != stem:
            raise _error(
                f"sidecar filename mismatch: {sidecar.name} does not belong to {annotations.image_filename!r}"
            )
        if annotations.image_filename in result:
            raise _error(f"duplicate image filename: {annotations.image_filename}")
        if verify_hash:
            image_path = image_root / annotations.image_filename
            if sha256_file(image_path) != annotations.sha256:
                raise _error(f"sha256 does not match image bytes for {annotations.image_filename}")
        result[annotations.image_filename] = annotations
    return result


scan_annotation_directory = scan_annotations
load_annotation_directory = scan_annotations


def load_flow_annotations(
    path: str | Path,
    image_path: str | Path | None = None,
    verify_image: bool = False,
) -> ImageFlowAnnotations:
    """Load UI annotations, optionally verifying the referenced image digest."""

    return load_annotations(path, image_path=image_path, verify_hash=verify_image)


def save_flow_annotations(path: str | Path, annotations: ImageFlowAnnotations) -> Path:
    """Atomically save UI annotations to a directory or explicit sidecar path."""

    return save_annotations(annotations, path)


def scan_flow_annotation_directory(path: str | Path) -> dict[str, ImageFlowAnnotations]:
    """Load all UI sidecars in ``path``, keyed by image filename."""

    return scan_annotations(path)

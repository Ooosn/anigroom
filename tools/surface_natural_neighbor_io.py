"""Strict R083 surface natural-neighbor binary I/O.

This module is deliberately limited to the Phase-A file contract.  It does
not launch the CGAL builder, load checkpoints, or expose a training-time
integration path.

Phase-A files are trusted private pipeline artifacts.  External SHA-256
verification owns snapshot and large-file integrity; this module does not
enforce arbitrary data-tuned size limits.  It checks only representational and
serialization arithmetic required by the binary contract.

The input file is little-endian and has this exact layout::

    header: <8sIIQQ
        magic, format version, header byte count, guide count, query count
    guide points: guide_count * 3 float64 values, C order
    query points: query_count * 3 float64 values, C order
    query normals: query_count * 3 float64 values, C order

The output file is little-endian and has this exact layout::

    header: <8sIIQQQ64s
        magic, format version, header byte count, guide count, query count,
        nnz, NUL-padded method identity
    row offsets: (query_count + 1) uint64 values
    guide IDs: nnz uint32 values
    normalized weights: nnz float64 values
    per-query success flags: query_count uint8 values (all must be 1)
    barycentric reconstruction errors: query_count float64 values

All arrays are stored in C/row-major order.  The reader intentionally checks
the complete file length before decoding payload arrays, so both truncation
and trailing bytes are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Iterable, TypeAlias

import numpy as np


INPUT_MAGIC = b"R083SNNI"
OUTPUT_MAGIC = b"R083SNNO"
FORMAT_VERSION = 1

INPUT_HEADER_FORMAT = "<8sIIQQ"
INPUT_HEADER = struct.Struct(INPUT_HEADER_FORMAT)
INPUT_HEADER_SIZE = INPUT_HEADER.size

OUTPUT_METHOD_BYTES = 64
OUTPUT_HEADER_FORMAT = f"<8sIIQQQ{OUTPUT_METHOD_BYTES}s"
OUTPUT_HEADER = struct.Struct(OUTPUT_HEADER_FORMAT)
OUTPUT_HEADER_SIZE = OUTPUT_HEADER.size

METHOD_IDENTITY = "CGAL.surface_neighbor_coordinates_3.Delaunay.v1"
METHOD_IDENTITY_BYTES = METHOD_IDENTITY.encode("ascii")
if len(METHOD_IDENTITY_BYTES) > OUTPUT_METHOD_BYTES:  # pragma: no cover
    raise RuntimeError("R083 method identity does not fit the output header")

UINT32_MAX = int(np.iinfo(np.uint32).max)
UINT64_MAX = int(np.iinfo(np.uint64).max)

# These constants are part of the validation contract and are mirrored by
# the standalone C++ builder.  Model surface normals are commonly normalized
# in float32 and then converted to float64 for this file, so the fixed 1e-5
# acceptance band accounts for that quantization without normalizing input.
NORMAL_NORM_TOLERANCE = 1.0e-5
ROW_SUM_TOLERANCE = 1.0e-10

PathLike: TypeAlias = str | os.PathLike[str]


class SurfaceNaturalNeighborFormatError(ValueError):
    """Raised when an R083 binary file violates its strict format contract."""


@dataclass(frozen=True)
class SurfaceNaturalNeighborInput:
    """Decoded and validated R083 builder input arrays."""

    guide_points: np.ndarray
    query_points: np.ndarray
    query_normals: np.ndarray

    @property
    def guide_count(self) -> int:
        return int(self.guide_points.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.query_points.shape[0])

    @property
    def guides(self) -> np.ndarray:
        return self.guide_points

    @property
    def queries(self) -> np.ndarray:
        return self.query_points

    @property
    def normals(self) -> np.ndarray:
        return self.query_normals

    @property
    def version(self) -> int:
        return FORMAT_VERSION

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "guide_count": self.guide_count,
            "query_count": self.query_count,
        }


@dataclass(frozen=True)
class SurfaceNaturalNeighborOutput:
    """Decoded and validated R083 CSR output and per-query evidence."""

    guide_count: int
    query_count: int
    nnz: int
    method: str
    row_offsets: np.ndarray
    guide_ids: np.ndarray
    weights: np.ndarray
    success: np.ndarray
    barycentric_errors: np.ndarray

    @property
    def ids(self) -> np.ndarray:
        return self.guide_ids

    @property
    def errors(self) -> np.ndarray:
        return self.barycentric_errors

    @property
    def barycentric_error(self) -> np.ndarray:
        return self.barycentric_errors

    @property
    def row_count(self) -> int:
        return self.query_count

    @property
    def version(self) -> int:
        return FORMAT_VERSION

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "guide_count": self.guide_count,
            "query_count": self.query_count,
            "nnz": self.nnz,
            "method": self.method,
        }

    def __getitem__(self, key: str) -> Any:
        """Provide small dict-like convenience for diagnostic callers."""

        if key in self.metadata:
            return self.metadata[key]
        try:
            return getattr(self, key)
        except AttributeError as error:
            raise KeyError(key) from error


def _require_little_endian() -> None:
    if sys.byteorder != "little":
        raise RuntimeError(
            "R083 surface natural-neighbor binary I/O requires a little-endian "
            f"platform; detected {sys.byteorder!r}"
        )


def _as_u64_count(value: int | np.integer[Any], name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    count = int(value)
    if count < 0 or count > UINT64_MAX:
        raise SurfaceNaturalNeighborFormatError(
            f"{name} is outside the uint64 range: {count}"
        )
    return count


def _checked_add(first: int, second: int, label: str) -> int:
    if first < 0 or second < 0 or first > UINT64_MAX or second > UINT64_MAX:
        raise SurfaceNaturalNeighborFormatError(f"{label} count arithmetic overflow")
    if first > UINT64_MAX - second:
        raise SurfaceNaturalNeighborFormatError(f"{label} count arithmetic overflow")
    return first + second


def _checked_mul(first: int, second: int, label: str) -> int:
    if first < 0 or second < 0 or first > UINT64_MAX or second > UINT64_MAX:
        raise SurfaceNaturalNeighborFormatError(f"{label} count arithmetic overflow")
    if first and second > UINT64_MAX // first:
        raise SurfaceNaturalNeighborFormatError(f"{label} count arithmetic overflow")
    return first * second


def _checked_numpy_count(value: int, name: str) -> int:
    max_count = min(sys.maxsize, int(np.iinfo(np.intp).max))
    if value > max_count:
        raise SurfaceNaturalNeighborFormatError(
            f"{name} is too large for this Python platform"
        )
    return value


def _input_payload_size(guide_count: int, query_count: int) -> int:
    guide_bytes = _checked_mul(guide_count, 3 * 8, "input guide")
    query_bytes = _checked_mul(query_count, 3 * 8, "input query")
    normal_bytes = _checked_mul(query_count, 3 * 8, "input normal")
    total = _checked_add(INPUT_HEADER_SIZE, guide_bytes, "input file")
    total = _checked_add(total, query_bytes, "input file")
    return _checked_add(total, normal_bytes, "input file")


def _output_payload_size(guide_count: int, query_count: int, nnz: int) -> int:
    del guide_count  # The guide count is validated separately; it has no payload.
    offset_count = _checked_add(query_count, 1, "output row offset")
    total = OUTPUT_HEADER_SIZE
    total = _checked_add(
        total,
        _checked_mul(offset_count, 8, "output row offset"),
        "output file",
    )
    total = _checked_add(
        total,
        _checked_mul(nnz, 4, "output guide ID"),
        "output file",
    )
    total = _checked_add(
        total,
        _checked_mul(nnz, 8, "output weight"),
        "output file",
    )
    total = _checked_add(total, query_count, "output success flag")
    total = _checked_add(
        total,
        _checked_mul(query_count, 8, "output barycentric error"),
        "output file",
    )
    return total


def _as_float64_matrix(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.dtype("<f8"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a numeric float64 matrix") from error
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {array.shape}")
    # Copy to detach the bytes written by the atomic writer from caller-owned
    # mutable storage.  This also guarantees C-contiguous little-endian bytes.
    array = np.array(array, dtype=np.dtype("<f8"), order="C", copy=True)
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains non-finite coordinates")
    return array


def _validate_input_arrays(
    guide_points: Any,
    query_points: Any,
    query_normals: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    guides = _as_float64_matrix(guide_points, "guide_points")
    queries = _as_float64_matrix(query_points, "query_points")
    normals = _as_float64_matrix(query_normals, "query_normals")

    if normals.shape[0] != queries.shape[0]:
        raise ValueError(
            "query_normals row count must exactly equal query_points row count; "
            f"got {normals.shape[0]} and {queries.shape[0]}"
        )

    guide_count = _as_u64_count(guides.shape[0], "guide count")
    query_count = _as_u64_count(queries.shape[0], "query count")
    if guide_count == 0:
        raise ValueError("guide_points must contain at least one guide")
    if guide_count > UINT32_MAX:
        raise ValueError("guide count exceeds the uint32 guide-ID contract")
    _input_payload_size(guide_count, query_count)

    if np.unique(guides, axis=0).shape[0] != guides.shape[0]:
        raise ValueError("guide_points contains duplicate guide coordinates")

    norms = np.hypot(np.hypot(normals[:, 0], normals[:, 1]), normals[:, 2])
    invalid_norms = ~np.isfinite(norms) | (
        np.abs(norms - 1.0) > NORMAL_NORM_TOLERANCE
    )
    if bool(invalid_norms.any()):
        first_bad = int(np.flatnonzero(invalid_norms)[0])
        raise ValueError(
            "query_normals must already be unit length; "
            f"row {first_bad} norm={norms[first_bad]!r} is outside "
            f"the {NORMAL_NORM_TOLERANCE:g} tolerance (no silent normalization)"
        )
    return guides, queries, normals


def _temporary_sibling(path: Path) -> tuple[tempfile._TemporaryFileWrapper, Path]:
    """Open a named sibling temporary file; kept separate for cleanup clarity."""

    handle = tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    return handle, Path(handle.name)


def _array_bytes(array: np.ndarray) -> bytes | memoryview:
    if array.nbytes == 0:
        return b""
    return memoryview(array).cast("B")


def _atomic_write_chunks(
    path: PathLike,
    chunks: Iterable[bytes | bytearray | memoryview],
    *,
    overwrite: bool,
) -> None:
    _require_little_endian()
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing R083 file: {destination}"
        )
    if not destination.parent.exists():
        raise FileNotFoundError(f"parent directory does not exist: {destination.parent}")

    handle: tempfile._TemporaryFileWrapper | None = None
    temporary: Path | None = None
    try:
        handle, temporary = _temporary_sibling(destination)
        for chunk in chunks:
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None

        if overwrite:
            os.replace(temporary, destination)
        else:
            # Hard-linking a complete sibling temp file gives a no-replace,
            # atomic publication on both POSIX and Windows filesystems.  A
            # concurrent creator therefore cannot turn refusal into overwrite.
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite existing R083 file: {destination}"
                ) from error
            temporary.unlink()
            temporary = None
    finally:
        if handle is not None:
            handle.close()
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def write_surface_natural_neighbor_input(
    path: PathLike,
    guide_points: Any,
    query_points: Any,
    query_normals: Any,
    *,
    overwrite: bool = False,
) -> None:
    """Validate and atomically write a deterministic R083 builder input."""

    guides, queries, normals = _validate_input_arrays(
        guide_points,
        query_points,
        query_normals,
    )
    guide_count = int(guides.shape[0])
    query_count = int(queries.shape[0])
    header = INPUT_HEADER.pack(
        INPUT_MAGIC,
        FORMAT_VERSION,
        INPUT_HEADER_SIZE,
        guide_count,
        query_count,
    )
    _atomic_write_chunks(
        path,
        (
            header,
            _array_bytes(guides),
            _array_bytes(queries),
            _array_bytes(normals),
        ),
        overwrite=overwrite,
    )


def _read_prefix(path: Path, size: int) -> tuple[bytes, int]:
    try:
        actual_size = int(path.stat().st_size)
    except OSError as error:
        raise OSError(f"cannot stat R083 file {path}: {error}") from error
    if actual_size < size:
        raise SurfaceNaturalNeighborFormatError(
            f"truncated R083 file: {path} has {actual_size} bytes, needs at least {size}"
        )
    try:
        with path.open("rb") as handle:
            prefix = handle.read(size)
    except OSError as error:
        raise OSError(f"cannot read R083 file {path}: {error}") from error
    if len(prefix) != size:
        raise SurfaceNaturalNeighborFormatError(
            f"truncated R083 file while reading its {size}-byte header"
        )
    return prefix, actual_size


def _read_exact_payload(path: Path, header: bytes, total_size: int, actual_size: int) -> bytes:
    if actual_size != total_size:
        if actual_size < total_size:
            raise SurfaceNaturalNeighborFormatError(
                f"truncated R083 file: expected {total_size} bytes, got {actual_size}"
            )
        raise SurfaceNaturalNeighborFormatError(
            f"trailing bytes in R083 file: expected {total_size} bytes, got {actual_size}"
        )
    payload_size = total_size - len(header)
    try:
        with path.open("rb") as handle:
            complete = handle.read(total_size)
            trailing = handle.read(1)
    except OSError as error:
        raise OSError(f"cannot read R083 file {path}: {error}") from error
    if len(complete) != total_size:
        raise SurfaceNaturalNeighborFormatError(
            f"truncated R083 file while reading {payload_size} payload bytes"
        )
    if trailing:
        raise SurfaceNaturalNeighborFormatError("trailing bytes in R083 file")
    if complete[: len(header)] != header:
        raise SurfaceNaturalNeighborFormatError("R083 file changed while being read")
    return complete


def _validate_input_header(prefix: bytes) -> tuple[int, int]:
    magic, version, header_size, guide_count_raw, query_count_raw = INPUT_HEADER.unpack(
        prefix
    )
    if magic != INPUT_MAGIC:
        raise SurfaceNaturalNeighborFormatError(f"bad input magic: {magic!r}")
    if version != FORMAT_VERSION:
        raise SurfaceNaturalNeighborFormatError(
            f"unsupported input version: {version}"
        )
    if header_size != INPUT_HEADER_SIZE:
        raise SurfaceNaturalNeighborFormatError(
            f"bad input header size: {header_size}"
        )
    guide_count = _as_u64_count(guide_count_raw, "guide count")
    query_count = _as_u64_count(query_count_raw, "query count")
    if guide_count == 0:
        raise SurfaceNaturalNeighborFormatError("guide count must be positive")
    # Exercise the uint64 file-size arithmetic before platform allocation
    # checks, so a wrapped count is reported as a format overflow rather than
    # as a later NumPy capacity failure.
    _input_payload_size(guide_count, query_count)
    if guide_count > UINT32_MAX:
        raise SurfaceNaturalNeighborFormatError(
            "guide count exceeds the uint32 guide-ID contract"
        )
    _checked_numpy_count(guide_count, "guide count")
    _checked_numpy_count(query_count, "query count")
    return guide_count, query_count


def read_surface_natural_neighbor_input(path: PathLike) -> SurfaceNaturalNeighborInput:
    """Read and strictly validate an R083 builder input file."""

    _require_little_endian()
    source = Path(path)
    prefix, actual_size = _read_prefix(source, INPUT_HEADER_SIZE)
    guide_count, query_count = _validate_input_header(prefix)
    total_size = _input_payload_size(guide_count, query_count)
    complete = _read_exact_payload(source, prefix, total_size, actual_size)

    guide_value_count = _checked_numpy_count(
        _checked_mul(guide_count, 3, "input guide value"),
        "input guide value count",
    )
    query_value_count = _checked_numpy_count(
        _checked_mul(query_count, 3, "input query value"),
        "input query value count",
    )
    cursor = INPUT_HEADER_SIZE
    guides = np.frombuffer(
        complete,
        dtype=np.dtype("<f8"),
        count=guide_value_count,
        offset=cursor,
    ).reshape((guide_count, 3)).copy()
    cursor += guide_value_count * 8
    queries = np.frombuffer(
        complete,
        dtype=np.dtype("<f8"),
        count=query_value_count,
        offset=cursor,
    ).reshape((query_count, 3)).copy()
    cursor += query_value_count * 8
    normals = np.frombuffer(
        complete,
        dtype=np.dtype("<f8"),
        count=query_value_count,
        offset=cursor,
    ).reshape((query_count, 3)).copy()
    cursor += query_value_count * 8
    if cursor != total_size:
        raise SurfaceNaturalNeighborFormatError(
            f"input parser consumed {cursor} bytes but file has {total_size}"
        )

    guides, queries, normals = _validate_input_arrays(guides, queries, normals)
    for array in (guides, queries, normals):
        array.setflags(write=False)
    return SurfaceNaturalNeighborInput(guides, queries, normals)


def _validate_output_header(prefix: bytes) -> tuple[int, int, int, str]:
    (
        magic,
        version,
        header_size,
        guide_count_raw,
        query_count_raw,
        nnz_raw,
        method_raw,
    ) = OUTPUT_HEADER.unpack(prefix)
    if magic != OUTPUT_MAGIC:
        raise SurfaceNaturalNeighborFormatError(f"bad output magic: {magic!r}")
    if version != FORMAT_VERSION:
        raise SurfaceNaturalNeighborFormatError(
            f"unsupported output version: {version}"
        )
    if header_size != OUTPUT_HEADER_SIZE:
        raise SurfaceNaturalNeighborFormatError(
            f"bad output header size: {header_size}"
        )
    expected_method = METHOD_IDENTITY_BYTES + b"\0" * (
        OUTPUT_METHOD_BYTES - len(METHOD_IDENTITY_BYTES)
    )
    if method_raw != expected_method:
        raise SurfaceNaturalNeighborFormatError(
            f"bad output method identity: {method_raw!r}"
        )
    guide_count = _as_u64_count(guide_count_raw, "guide count")
    query_count = _as_u64_count(query_count_raw, "query count")
    nnz = _as_u64_count(nnz_raw, "nnz")
    if guide_count == 0:
        raise SurfaceNaturalNeighborFormatError("output guide count must be positive")
    if guide_count > UINT32_MAX:
        raise SurfaceNaturalNeighborFormatError(
            "output guide count exceeds the uint32 guide-ID contract"
        )
    return guide_count, query_count, nnz, METHOD_IDENTITY


def _as_output_array(
    complete: bytes,
    *,
    dtype: np.dtype[Any],
    count: int,
    cursor: int,
    name: str,
) -> tuple[np.ndarray, int]:
    byte_count = _checked_mul(count, int(dtype.itemsize), f"output {name}")
    next_cursor = _checked_add(cursor, byte_count, f"output {name}")
    if next_cursor > len(complete):
        raise SurfaceNaturalNeighborFormatError(
            f"truncated output while reading {name}"
        )
    values = np.frombuffer(
        complete,
        dtype=dtype,
        count=count,
        offset=cursor,
    ).copy()
    return values, next_cursor


def _validate_output_rows(
    row_offsets: np.ndarray,
    guide_ids: np.ndarray,
    weights: np.ndarray,
    guide_count: int,
    query_count: int,
    nnz: int,
) -> None:
    if row_offsets.size != query_count + 1:
        raise SurfaceNaturalNeighborFormatError("wrong row-offset count")
    if row_offsets[0] != 0:
        raise SurfaceNaturalNeighborFormatError("CSR row offsets must start at zero")
    if row_offsets.size > 1 and bool(
        np.any(row_offsets[1:] < row_offsets[:-1])
    ):
        raise SurfaceNaturalNeighborFormatError("non-monotone CSR row offsets")
    if int(row_offsets[-1]) != nnz:
        raise SurfaceNaturalNeighborFormatError(
            f"wrong final nnz: row offsets end at {int(row_offsets[-1])}, header says {nnz}"
        )

    if guide_ids.size and bool(np.any(guide_ids.astype(np.uint64) >= guide_count)):
        raise SurfaceNaturalNeighborFormatError("out-of-range guide ID")
    if not bool(np.isfinite(weights).all()):
        raise SurfaceNaturalNeighborFormatError("non-finite weight")
    if bool(np.any(weights < 0.0)):
        raise SurfaceNaturalNeighborFormatError("negative weight")

    for row_index in range(query_count):
        start = int(row_offsets[row_index])
        end = int(row_offsets[row_index + 1])
        if end <= start:
            raise SurfaceNaturalNeighborFormatError(
                f"bad row sum for empty query row {row_index}"
            )
        row_ids = guide_ids[start:end]
        if row_ids.size > 1:
            equal = row_ids[1:] == row_ids[:-1]
            if bool(equal.any()):
                raise SurfaceNaturalNeighborFormatError(
                    f"duplicate guide ID in query row {row_index}"
                )
            if bool(np.any(row_ids[1:] < row_ids[:-1])):
                raise SurfaceNaturalNeighborFormatError(
                    f"guide IDs are not sorted in query row {row_index}"
                )
        row_sum = float(np.sum(weights[start:end], dtype=np.float64))
        if not np.isfinite(row_sum) or abs(row_sum - 1.0) > ROW_SUM_TOLERANCE:
            raise SurfaceNaturalNeighborFormatError(
                f"bad row sum for query row {row_index}: {row_sum!r}"
            )


def read_surface_natural_neighbor_output(path: PathLike) -> SurfaceNaturalNeighborOutput:
    """Read and strictly validate a CGAL-produced R083 CSR output file."""

    _require_little_endian()
    source = Path(path)
    prefix, actual_size = _read_prefix(source, OUTPUT_HEADER_SIZE)
    guide_count, query_count, nnz, method = _validate_output_header(prefix)
    total_size = _output_payload_size(guide_count, query_count, nnz)
    complete = _read_exact_payload(source, prefix, total_size, actual_size)

    offset_count = _checked_numpy_count(
        _checked_add(query_count, 1, "output row offset"),
        "output row offset count",
    )
    cursor = OUTPUT_HEADER_SIZE
    row_offsets, cursor = _as_output_array(
        complete,
        dtype=np.dtype("<u8"),
        count=offset_count,
        cursor=cursor,
        name="row offsets",
    )
    guide_ids, cursor = _as_output_array(
        complete,
        dtype=np.dtype("<u4"),
        count=_checked_numpy_count(nnz, "nnz"),
        cursor=cursor,
        name="guide IDs",
    )
    weights, cursor = _as_output_array(
        complete,
        dtype=np.dtype("<f8"),
        count=_checked_numpy_count(nnz, "nnz"),
        cursor=cursor,
        name="weights",
    )
    success_raw, cursor = _as_output_array(
        complete,
        dtype=np.dtype("<u1"),
        count=_checked_numpy_count(query_count, "query count"),
        cursor=cursor,
        name="success flags",
    )
    barycentric_errors, cursor = _as_output_array(
        complete,
        dtype=np.dtype("<f8"),
        count=_checked_numpy_count(query_count, "query count"),
        cursor=cursor,
        name="barycentric errors",
    )
    if cursor != total_size:
        raise SurfaceNaturalNeighborFormatError(
            f"output parser consumed {cursor} bytes but file has {total_size}"
        )

    if bool(np.any(success_raw != 1)):
        raise SurfaceNaturalNeighborFormatError(
            "output contains a query whose success flag is not true"
        )
    if not bool(np.isfinite(barycentric_errors).all()):
        raise SurfaceNaturalNeighborFormatError("non-finite barycentric error")
    if bool(np.any(barycentric_errors < 0.0)):
        raise SurfaceNaturalNeighborFormatError("negative barycentric error")
    _validate_output_rows(
        row_offsets,
        guide_ids,
        weights,
        guide_count,
        query_count,
        nnz,
    )

    success = (success_raw == 1).astype(np.bool_)
    for array in (
        row_offsets,
        guide_ids,
        weights,
        success,
        barycentric_errors,
    ):
        array.setflags(write=False)
    return SurfaceNaturalNeighborOutput(
        guide_count=guide_count,
        query_count=query_count,
        nnz=nnz,
        method=method,
        row_offsets=row_offsets,
        guide_ids=guide_ids,
        weights=weights,
        success=success,
        barycentric_errors=barycentric_errors,
    )


def sha256_file(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of an input or output file."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def input_sha256(path: PathLike) -> str:
    return sha256_file(path)


def output_sha256(path: PathLike) -> str:
    return sha256_file(path)


def sha256_input(path: PathLike) -> str:
    return input_sha256(path)


def sha256_output(path: PathLike) -> str:
    return output_sha256(path)


def row_sums(output: SurfaceNaturalNeighborOutput) -> np.ndarray:
    """Return validated float64 CSR row sums in query order."""

    if output.query_count == 0:
        return np.empty((0,), dtype=np.float64)
    starts = output.row_offsets[:-1].astype(np.intp, copy=False)
    sums = np.add.reduceat(output.weights, starts).astype(np.float64, copy=False)
    sums.setflags(write=False)
    return sums


def row_sum_summary(output: SurfaceNaturalNeighborOutput) -> dict[str, Any]:
    """Return compact partition-of-unity evidence for a validated output."""

    sums = row_sums(output)
    if sums.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "max": None,
            "max_abs_error": None,
            "all_within_tolerance": True,
        }
    errors = np.abs(sums - 1.0)
    return {
        "count": int(sums.size),
        "min": float(np.min(sums)),
        "mean": float(np.mean(sums, dtype=np.float64)),
        "max": float(np.max(sums)),
        "max_abs_error": float(np.max(errors)),
        "all_within_tolerance": bool(np.all(errors <= ROW_SUM_TOLERANCE)),
    }


def barycentric_error_summary(
    output: SurfaceNaturalNeighborOutput,
) -> dict[str, Any]:
    """Return compact finite per-query reconstruction-error evidence."""

    errors = output.barycentric_errors
    if errors.size == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    quantiles = np.quantile(errors, [0.50, 0.95], method="linear")
    return {
        "count": int(errors.size),
        "mean": float(np.mean(errors, dtype=np.float64)),
        "min": float(np.min(errors)),
        "p50": float(quantiles[0]),
        "p95": float(quantiles[1]),
        "max": float(np.max(errors)),
    }


def summarize_input(value: SurfaceNaturalNeighborInput | PathLike) -> dict[str, Any]:
    """Return a concise validated input summary, optionally including SHA-256."""

    if isinstance(value, SurfaceNaturalNeighborInput):
        decoded = value
        digest = None
        byte_count = None
    else:
        decoded = read_surface_natural_neighbor_input(value)
        digest = input_sha256(value)
        byte_count = int(Path(value).stat().st_size)
    norms = np.hypot(
        np.hypot(decoded.query_normals[:, 0], decoded.query_normals[:, 1]),
        decoded.query_normals[:, 2],
    )
    summary: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "guide_count": decoded.guide_count,
        "query_count": decoded.query_count,
        "normal_norm_max_abs_error": float(
            np.max(np.abs(norms - 1.0))
        )
        if norms.size
        else 0.0,
    }
    if digest is not None:
        summary["bytes"] = byte_count
        summary["sha256"] = digest
    return summary


def summarize_output(value: SurfaceNaturalNeighborOutput | PathLike) -> dict[str, Any]:
    """Return a concise validated output summary with row/evidence statistics."""

    if isinstance(value, SurfaceNaturalNeighborOutput):
        decoded = value
        digest = None
        byte_count = None
    else:
        decoded = read_surface_natural_neighbor_output(value)
        digest = output_sha256(value)
        byte_count = int(Path(value).stat().st_size)
    summary: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "guide_count": decoded.guide_count,
        "query_count": decoded.query_count,
        "nnz": decoded.nnz,
        "method": decoded.method,
        "success_count": int(np.count_nonzero(decoded.success)),
        "row_sum": row_sum_summary(decoded),
        "barycentric_error": barycentric_error_summary(decoded),
    }
    if digest is not None:
        summary["bytes"] = byte_count
        summary["sha256"] = digest
    return summary


# Short names keep the diagnostic module convenient without creating a second
# format-specific API.  The long names above remain the canonical contract.
write_input = write_surface_natural_neighbor_input
read_input = read_surface_natural_neighbor_input
read_output = read_surface_natural_neighbor_output
summarize = summarize_output


__all__ = [
    "FORMAT_VERSION",
    "INPUT_HEADER",
    "INPUT_HEADER_FORMAT",
    "INPUT_HEADER_SIZE",
    "INPUT_MAGIC",
    "METHOD_IDENTITY",
    "METHOD_IDENTITY_BYTES",
    "NORMAL_NORM_TOLERANCE",
    "OUTPUT_HEADER",
    "OUTPUT_HEADER_FORMAT",
    "OUTPUT_HEADER_SIZE",
    "OUTPUT_MAGIC",
    "OUTPUT_METHOD_BYTES",
    "ROW_SUM_TOLERANCE",
    "SurfaceNaturalNeighborFormatError",
    "SurfaceNaturalNeighborInput",
    "SurfaceNaturalNeighborOutput",
    "barycentric_error_summary",
    "input_sha256",
    "output_sha256",
    "read_input",
    "read_output",
    "read_surface_natural_neighbor_input",
    "read_surface_natural_neighbor_output",
    "row_sum_summary",
    "row_sums",
    "sha256_file",
    "sha256_input",
    "sha256_output",
    "summarize",
    "summarize_input",
    "summarize_output",
    "write_input",
    "write_surface_natural_neighbor_input",
]

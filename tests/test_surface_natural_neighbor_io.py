from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

from tools import surface_natural_neighbor_io as r083_io


GUIDES = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
QUERIES = np.asarray(
    [[0.2, 0.2, 0.2], [0.1, 0.3, 0.2]],
    dtype=np.float64,
)
NORMALS = np.asarray(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
COPLANAR_GUIDES = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _write_valid_input(path: Path, *, query_count: int = 2) -> bytes:
    queries = QUERIES[:query_count]
    normals = NORMALS[:query_count]
    r083_io.write_surface_natural_neighbor_input(
        path,
        GUIDES,
        queries,
        normals,
    )
    return path.read_bytes()


def _output_bytes(
    *,
    guide_count: int = 4,
    row_offsets: tuple[int, ...] = (0, 2, 3),
    guide_ids: tuple[int, ...] = (1, 3, 0),
    weights: tuple[float, ...] = (0.25, 0.75, 1.0),
    success: tuple[int, ...] | None = None,
    barycentric_errors: tuple[float, ...] = (0.1, 0.4),
    nnz: int | None = None,
    method: bytes | None = None,
) -> bytes:
    query_count = len(row_offsets) - 1
    if nnz is None:
        nnz = len(guide_ids)
    if success is None:
        success = (1,) * query_count
    method_field = (
        r083_io.METHOD_IDENTITY_BYTES
        if method is None
        else method
    )
    method_field = method_field + b"\0" * (
        r083_io.OUTPUT_METHOD_BYTES - len(method_field)
    )
    header = r083_io.OUTPUT_HEADER.pack(
        r083_io.OUTPUT_MAGIC,
        r083_io.FORMAT_VERSION,
        r083_io.OUTPUT_HEADER_SIZE,
        guide_count,
        query_count,
        nnz,
        method_field,
    )
    return b"".join(
        [
            header,
            np.asarray(row_offsets, dtype="<u8").tobytes(),
            np.asarray(guide_ids, dtype="<u4").tobytes(),
            np.asarray(weights, dtype="<f8").tobytes(),
            np.asarray(success, dtype="<u1").tobytes(),
            np.asarray(barycentric_errors, dtype="<f8").tobytes(),
        ]
    )


def _write_output(path: Path, **kwargs: object) -> None:
    path.write_bytes(_output_bytes(**kwargs))


def test_input_bytes_are_deterministic_and_roundtrip_exactly(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first_bytes = _write_valid_input(first)
    second_bytes = _write_valid_input(second)

    assert first_bytes == second_bytes
    assert len(first_bytes) == r083_io.INPUT_HEADER_SIZE + 4 * 3 * 8 + 2 * 3 * 8 * 2
    decoded = r083_io.read_surface_natural_neighbor_input(first)
    np.testing.assert_array_equal(decoded.guide_points, GUIDES)
    np.testing.assert_array_equal(decoded.query_points, QUERIES)
    np.testing.assert_array_equal(decoded.query_normals, NORMALS)
    assert decoded.metadata == {
        "format_version": 1,
        "guide_count": 4,
        "query_count": 2,
    }
    expected_sha = hashlib.sha256(first_bytes).hexdigest()
    assert r083_io.sha256_file(first) == expected_sha
    assert r083_io.input_sha256(first) == expected_sha
    assert r083_io.sha256_input(first) == expected_sha


def test_golden_headers_match_literal_little_endian_bytes(tmp_path: Path) -> None:
    input_path = tmp_path / "golden-input.bin"
    input_bytes = _write_valid_input(input_path)
    expected_input_header = bytes.fromhex(
        "52303833534e4e49"
        "01000000"
        "20000000"
        "0400000000000000"
        "0200000000000000"
    )
    assert input_bytes[:32] == expected_input_header

    output_path = tmp_path / "golden-output.bin"
    output_bytes = _output_bytes()
    output_path.write_bytes(output_bytes)
    expected_output_header = bytes.fromhex(
        "52303833534e4e4f"
        "01000000"
        "68000000"
        "0400000000000000"
        "0200000000000000"
        "0300000000000000"
    ) + b"CGAL.surface_neighbor_coordinates_3.Delaunay.v1" + b"\0" * 17
    assert output_bytes[:104] == expected_output_header
    assert len(expected_input_header) == 32
    assert len(expected_output_header) == 104
    assert r083_io.read_output(output_path).method == (
        "CGAL.surface_neighbor_coordinates_3.Delaunay.v1"
    )


def test_zero_query_input_and_output_are_valid(tmp_path: Path) -> None:
    input_path = tmp_path / "zero-input.bin"
    _write_valid_input(input_path, query_count=0)
    decoded_input = r083_io.read_input(input_path)
    assert decoded_input.guide_count == 4
    assert decoded_input.query_count == 0
    assert decoded_input.query_points.shape == (0, 3)
    assert decoded_input.query_normals.shape == (0, 3)

    output_path = tmp_path / "zero-output.bin"
    _write_output(
        output_path,
        row_offsets=(0,),
        guide_ids=(),
        weights=(),
        barycentric_errors=(),
    )
    decoded_output = r083_io.read_output(output_path)
    assert decoded_output.query_count == 0
    assert decoded_output.nnz == 0
    assert decoded_output.row_offsets.tolist() == [0]
    assert decoded_output.success.shape == (0,)
    assert r083_io.row_sum_summary(decoded_output)["count"] == 0
    assert r083_io.barycentric_error_summary(decoded_output)["count"] == 0


def test_zero_query_contract_does_not_require_full_dimensional_guides(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "zero-query-coplanar-input.bin"
    r083_io.write_input(
        input_path,
        COPLANAR_GUIDES,
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
    )
    decoded_input = r083_io.read_input(input_path)
    assert decoded_input.guide_count == 3
    assert decoded_input.query_count == 0

    output_path = tmp_path / "zero-query-coplanar-output.bin"
    _write_output(
        output_path,
        guide_count=3,
        row_offsets=(0,),
        guide_ids=(),
        weights=(),
        barycentric_errors=(),
    )
    decoded_output = r083_io.read_output(output_path)
    assert decoded_output.guide_count == 3
    assert decoded_output.query_count == 0
    assert decoded_output.nnz == 0


def test_query_normal_cardinality_must_match_query_points(tmp_path: Path) -> None:
    query_points = QUERIES[:1]
    too_few_normals = np.empty((0, 3), dtype=np.float64)
    too_many_normals = np.vstack([NORMALS, [0.0, 1.0, 0.0]])
    for normals, expected in (
        (too_few_normals, "got 0 and 1"),
        (too_many_normals, "got 3 and 1"),
    ):
        with pytest.raises(ValueError, match="row count must exactly equal") as error:
            r083_io.write_input(
                tmp_path / f"cardinality-{normals.shape[0]}.bin",
                GUIDES,
                query_points,
                normals,
            )
        assert expected in str(error.value)


@pytest.mark.parametrize("scale", [0.99, 1.0 - 1.0e-6])
def test_float32_normal_tolerance_accepts_values_inside_boundary(
    tmp_path: Path,
    scale: float,
) -> None:
    assert r083_io.NORMAL_NORM_TOLERANCE == 1.0e-5
    path = tmp_path / f"normal-inside-{scale}.bin"
    normal = np.asarray(
        [[1.0 + r083_io.NORMAL_NORM_TOLERANCE * scale, 0.0, 0.0]],
        dtype=np.float64,
    )
    r083_io.write_input(path, GUIDES, QUERIES[:1], normal)
    assert r083_io.read_input(path).query_count == 1


def test_normalized_float32_model_normal_survives_float64_file_conversion(
    tmp_path: Path,
) -> None:
    model_normal = np.asarray([[1.0, 1.0, 1.0]], dtype=np.float32)
    model_normal /= np.linalg.norm(model_normal, axis=1, keepdims=True)
    path = tmp_path / "float32-model-normal.bin"
    r083_io.write_input(path, GUIDES, QUERIES[:1], model_normal)
    decoded = r083_io.read_input(path)
    assert decoded.query_normals.dtype == np.float64
    assert abs(float(np.linalg.norm(decoded.query_normals[0])) - 1.0) <= (
        r083_io.NORMAL_NORM_TOLERANCE
    )


@pytest.mark.parametrize("scale", [1.01, 1.0 + 1.0e-6])
def test_float32_normal_tolerance_rejects_values_outside_boundary(
    tmp_path: Path,
    scale: float,
) -> None:
    assert r083_io.NORMAL_NORM_TOLERANCE == 1.0e-5
    normal = np.asarray(
        [[1.0 + r083_io.NORMAL_NORM_TOLERANCE * scale, 0.0, 0.0]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="unit length"):
        r083_io.write_input(
            tmp_path / f"normal-outside-{scale}.bin",
            GUIDES,
            QUERIES[:1],
            normal,
        )


def test_fabricated_valid_csr_output_roundtrips_and_exposes_metadata(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "valid-output.bin"
    _write_output(output_path)

    decoded = r083_io.read_surface_natural_neighbor_output(output_path)
    assert decoded.metadata == {
        "format_version": 1,
        "guide_count": 4,
        "query_count": 2,
        "nnz": 3,
        "method": r083_io.METHOD_IDENTITY,
    }
    np.testing.assert_array_equal(decoded.row_offsets, [0, 2, 3])
    np.testing.assert_array_equal(decoded.guide_ids, [1, 3, 0])
    np.testing.assert_allclose(decoded.weights, [0.25, 0.75, 1.0], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(decoded.success, [True, True])
    np.testing.assert_allclose(
        decoded.barycentric_errors,
        [0.1, 0.4],
        rtol=0.0,
        atol=0.0,
    )
    assert r083_io.output_sha256(output_path) == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()


def test_row_sum_and_barycentric_summaries_are_finite_and_deterministic(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "summary-output.bin"
    _write_output(output_path)
    decoded = r083_io.read_output(output_path)

    assert r083_io.row_sums(decoded).tolist() == [1.0, 1.0]
    assert r083_io.row_sum_summary(decoded) == {
        "count": 2,
        "min": 1.0,
        "mean": 1.0,
        "max": 1.0,
        "max_abs_error": 0.0,
        "all_within_tolerance": True,
    }
    error_summary = r083_io.barycentric_error_summary(decoded)
    assert error_summary["count"] == 2
    assert error_summary["min"] == 0.1
    assert error_summary["p50"] == 0.25
    assert error_summary["p95"] == pytest.approx(0.385)
    assert error_summary["max"] == 0.4

    input_path = tmp_path / "summary-input.bin"
    _write_valid_input(input_path)
    input_summary = r083_io.summarize_input(input_path)
    assert input_summary["guide_count"] == 4
    assert input_summary["query_count"] == 2
    assert input_summary["normal_norm_max_abs_error"] == 0.0
    assert input_summary["sha256"] == r083_io.input_sha256(input_path)
    output_summary = r083_io.summarize_output(output_path)
    assert output_summary["nnz"] == 3
    assert output_summary["success_count"] == 2
    assert output_summary["row_sum"]["all_within_tolerance"] is True
    assert output_summary["barycentric_error"]["max"] == 0.4


def test_atomic_input_writer_refuses_without_overwrite_and_replaces_explicitly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "atomic-input.bin"
    first_bytes = _write_valid_input(path, query_count=1)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        r083_io.write_input(path, GUIDES, QUERIES, NORMALS)
    assert path.read_bytes() == first_bytes

    r083_io.write_input(path, GUIDES, QUERIES, NORMALS, overwrite=True)
    assert path.read_bytes() != first_bytes
    assert r083_io.read_input(path).query_count == 2

    invalid_path = tmp_path / "must-not-appear.bin"
    with pytest.raises(ValueError, match="unit length"):
        r083_io.write_input(
            invalid_path,
            GUIDES,
            QUERIES[:1],
            np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64),
        )
    assert not invalid_path.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.__setitem__(0, 0), "bad input magic"),
        (
            lambda data: struct.pack_into("<I", data, 8, 2),
            "unsupported input version",
        ),
        (
            lambda data: struct.pack_into("<I", data, 12, 31),
            "bad input header size",
        ),
        (lambda data: data.__delitem__(-1), "truncated"),
        (lambda data: data.extend(b"trailing"), "trailing bytes"),
        (
            lambda data: struct.pack_into("<Q", data, 16, r083_io.UINT64_MAX),
            "count arithmetic overflow",
        ),
    ],
)
def test_malformed_input_headers_and_lengths_are_rejected(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    path = tmp_path / "corrupt-input.bin"
    data = bytearray(_write_valid_input(path))
    path.write_bytes(data)
    mutation(data)  # type: ignore[operator]
    path.write_bytes(data)
    with pytest.raises((ValueError, RuntimeError), match=message):
        r083_io.read_input(path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("duplicate", "duplicate guide coordinates"),
        ("guide_nonfinite", "non-finite coordinates"),
        ("query_nonfinite", "non-finite coordinates"),
        ("normal_nonfinite", "non-finite coordinates"),
        ("normal_not_unit", "unit length"),
    ],
)
def test_malformed_input_payload_values_are_rejected(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    path = tmp_path / f"bad-{field}.bin"
    data = bytearray(_write_valid_input(path))
    guide_start = r083_io.INPUT_HEADER_SIZE
    query_start = guide_start + GUIDES.size * 8
    normal_start = query_start + QUERIES.size * 8
    if field == "duplicate":
        data[guide_start + 3 * 8 : guide_start + 6 * 8] = data[
            guide_start : guide_start + 3 * 8
        ]
    elif field == "guide_nonfinite":
        struct.pack_into("<d", data, guide_start, math.nan)
    elif field == "query_nonfinite":
        struct.pack_into("<d", data, query_start, math.inf)
    elif field == "normal_nonfinite":
        struct.pack_into("<d", data, normal_start, math.nan)
    else:
        struct.pack_into("<d", data, normal_start + 16, 1.0 + 2.0e-5)
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        r083_io.read_input(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.__setitem__(0, 0), "bad output magic"),
        (
            lambda data: struct.pack_into("<I", data, 8, 2),
            "unsupported output version",
        ),
        (
            lambda data: struct.pack_into("<I", data, 12, r083_io.OUTPUT_HEADER_SIZE - 1),
            "bad output header size",
        ),
        (
            lambda data: data.__setitem__(40, ord("X")),
            "bad output method identity",
        ),
        (lambda data: data.__delitem__(-1), "truncated"),
        (lambda data: data.extend(b"trailing"), "trailing bytes"),
    ],
)
def test_malformed_output_headers_and_lengths_are_rejected(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    path = tmp_path / "corrupt-output.bin"
    data = bytearray(_output_bytes())
    mutation(data)  # type: ignore[operator]
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        r083_io.read_output(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"row_offsets": (0, 2, 1)}, "non-monotone"),
        ({"row_offsets": (0, 2, 2)}, "wrong final nnz"),
        ({"guide_ids": (1, 4, 0)}, "out-of-range guide ID"),
        ({"weights": (math.nan, 0.75, 1.0)}, "non-finite weight"),
        ({"weights": (-0.25, 1.25, 1.0)}, "negative weight"),
        ({"weights": (0.25, 0.25, 1.0)}, "bad row sum"),
        ({"guide_ids": (1, 1, 0)}, "duplicate guide ID"),
        ({"guide_ids": (3, 1, 0)}, "not sorted"),
        ({"barycentric_errors": (math.nan, 0.4)}, "non-finite barycentric error"),
        ({"barycentric_errors": (-0.1, 0.4)}, "negative barycentric error"),
        ({"success": (1, 0)}, "success flag"),
    ],
)
def test_malformed_output_csr_and_evidence_are_rejected(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "bad-csr-output.bin"
    _write_output(path, **kwargs)
    with pytest.raises(ValueError, match=message):
        r083_io.read_output(path)


@pytest.mark.parametrize(
    "header_offset,header_value,message",
    [
        (24, r083_io.UINT64_MAX, "count arithmetic overflow"),
        (32, r083_io.UINT64_MAX, "count arithmetic overflow"),
    ],
)
def test_output_count_overflow_is_rejected_before_payload_decode(
    tmp_path: Path,
    header_offset: int,
    header_value: int,
    message: str,
) -> None:
    path = tmp_path / "overflow-output.bin"
    data = bytearray(_output_bytes())
    struct.pack_into("<Q", data, header_offset, header_value)
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        r083_io.read_output(path)


def test_little_endian_platform_requirement_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.byteorder != "little":
        pytest.fail("the test host must be little-endian to exercise the contract")
    monkeypatch.setattr(r083_io.sys, "byteorder", "big")
    with pytest.raises(RuntimeError, match="little-endian"):
        r083_io.write_input(tmp_path / "not-written.bin", GUIDES, QUERIES, NORMALS)

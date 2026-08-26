import hashlib
import json

import pytest

from anigroom.flow_annotations import (
    SCHEMA_NAME,
    FlowArrow,
    ImageFlowAnnotations,
    SparseFlowAnnotations,
    SparseArrow,
    SparseFlowValidationError,
    annotation_path,
    load_annotations,
    load_flow_annotations,
    make_arrow,
    make_flow_arrow,
    normalized_direction_length,
    pixel_to_uv,
    save_annotations,
    save_flow_annotations,
    scan_annotations,
    scan_flow_annotation_directory,
    sha256_file,
    uv_to_pixel,
)


def _image(tmp_path, name: str, content: bytes) -> tuple[object, str]:
    path = tmp_path / name
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def _document(image_path, digest: str, *, arrow_id: str = "a") -> SparseFlowAnnotations:
    return SparseFlowAnnotations.from_image(
        image_path,
        width=5,
        height=4,
        arrows=[make_arrow(arrow_id, (0, 0), (4, 3), 5, 4, 0.75)],
        updated_utc="2026-08-26T05:00:00Z",
    )


def test_roundtrip_preserves_schema_and_arrow_records(tmp_path) -> None:
    image, digest = _image(tmp_path, "frame.png", b"image-bytes")
    document = _document(image, digest)

    output = save_annotations(document, tmp_path)
    loaded = load_annotations(
        output,
        expected_dimensions=(5, 4),
        expected_sha256=digest,
        image_path=image,
        verify_hash=True,
    )

    assert output == annotation_path(tmp_path, "frame.png")
    assert loaded == document
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == SCHEMA_NAME
    assert loaded.arrows[0].root_to_tip is True


def test_ui_facing_api_wraps_the_validated_records(tmp_path) -> None:
    image, digest = _image(tmp_path, "frame.png", b"image-bytes")
    arrow: FlowArrow = make_flow_arrow("ui", (0, 0), (4, 3), 5, 4, 0.9)
    annotations: ImageFlowAnnotations = ImageFlowAnnotations.from_image(
        image,
        5,
        4,
        [arrow],
        updated_utc="2026-08-26T05:00:00Z",
    )

    output = save_flow_annotations(tmp_path, annotations)
    loaded = load_flow_annotations(output, image, verify_image=True)

    assert loaded == annotations
    assert scan_flow_annotation_directory(tmp_path)["frame.png"] == annotations
    assert loaded.sha256 == digest


def test_pixel_uv_conversion_includes_both_boundaries() -> None:
    assert pixel_to_uv((0, 0), 5, 4) == (0.0, 0.0)
    assert pixel_to_uv((4, 3), 5, 4) == (1.0, 1.0)
    assert uv_to_pixel((0, 0), 5, 4) == (0.0, 0.0)
    assert uv_to_pixel((1, 1), 5, 4) == (4.0, 3.0)
    assert uv_to_pixel(pixel_to_uv((2, 1), 5, 4), 5, 4) == (2.0, 1.0)


def test_direction_and_length_are_normalized() -> None:
    direction, length = normalized_direction_length((1, 2), (4, 6))
    assert direction == pytest.approx((0.6, 0.8))
    assert length == pytest.approx(5.0)

    arrow = SparseArrow.from_pixels("a", (1, 2), (4, 6), 5, 8, 1.0)
    assert arrow.direction_px == pytest.approx((0.6, 0.8))
    assert arrow.length_px == pytest.approx(5.0)


def test_atomic_overwrite_replaces_sidecar_without_temp_files(tmp_path) -> None:
    image, digest = _image(tmp_path, "frame.png", b"image-bytes")
    first = _document(image, digest, arrow_id="first")
    second = _document(image, digest, arrow_id="second")

    output = save_annotations(first, tmp_path)
    output_again = save_annotations(second, tmp_path)

    assert output_again == output
    assert load_annotations(output).arrows[0].id == "second"
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.flow.json.*")) == []


def test_directory_scan_is_keyed_by_image_filename(tmp_path) -> None:
    image_a, digest_a = _image(tmp_path, "a.png", b"a")
    image_b, digest_b = _image(tmp_path, "b.jpg", b"b")
    save_annotations(_document(image_a, digest_a), tmp_path)
    save_annotations(_document(image_b, digest_b), tmp_path)

    scanned = scan_annotations(tmp_path)

    assert set(scanned) == {"a.png", "b.jpg"}
    assert scanned["a.png"].image_filename == "a.png"
    assert scanned["b.jpg"].sha256 == sha256_file(image_b)


def _write_payload(tmp_path, payload: object) -> object:
    path = tmp_path / "frame.flow.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_payload() -> dict[str, object]:
    arrow = make_arrow("a", (0, 0), (4, 3), 5, 4, 0.5)
    return SparseFlowAnnotations(
        image_filename="frame.png",
        width=5,
        height=4,
        sha256="0" * 64,
        arrows=(arrow,),
        updated_utc="2026-08-26T05:00:00Z",
    ).to_dict()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "anigroom.sparse_flow.v0"),
        ("width", 1),
        ("sha256", "not-a-digest"),
        ("updated_utc", "2026-08-26T05:00:00"),
    ],
)
def test_rejects_invalid_document_metadata(tmp_path, field: str, value: object) -> None:
    payload = _valid_payload()
    payload[field] = value

    with pytest.raises(SparseFlowValidationError):
        load_annotations(_write_payload(tmp_path, payload))


def test_rejects_zero_length_and_out_of_bounds_arrows(tmp_path) -> None:
    zero_length = _valid_payload()
    zero_length_arrow = zero_length["arrows"][0]
    zero_length_arrow["end_px"] = zero_length_arrow["start_px"]
    zero_length_arrow["end_uv"] = zero_length_arrow["start_uv"]
    with pytest.raises(SparseFlowValidationError):
        load_annotations(_write_payload(tmp_path, zero_length))

    out_of_bounds = _valid_payload()
    out_of_bounds_arrow = out_of_bounds["arrows"][0]
    out_of_bounds_arrow["end_px"] = [5, 3]
    out_of_bounds_arrow["end_uv"] = [1.25, 1.0]
    with pytest.raises(SparseFlowValidationError):
        load_annotations(_write_payload(tmp_path, out_of_bounds))


def test_rejects_duplicate_ids_and_unknown_arrow_fields(tmp_path) -> None:
    payload = _valid_payload()
    payload["arrows"] = [payload["arrows"][0], dict(payload["arrows"][0])]
    with pytest.raises(SparseFlowValidationError):
        load_annotations(_write_payload(tmp_path, payload))

    unknown_field = _valid_payload()
    unknown_field["arrows"][0]["extra"] = 1
    with pytest.raises(SparseFlowValidationError):
        load_annotations(_write_payload(tmp_path, unknown_field))


def test_rejects_malformed_json_and_dimension_or_hash_mismatch(tmp_path) -> None:
    malformed = tmp_path / "malformed.flow.json"
    malformed.write_text("{not json", encoding="utf-8")
    with pytest.raises(SparseFlowValidationError):
        load_annotations(malformed)

    image, digest = _image(tmp_path, "frame.png", b"image-bytes")
    output = save_annotations(_document(image, digest), tmp_path)
    with pytest.raises(SparseFlowValidationError):
        load_annotations(output, expected_width=6)
    with pytest.raises(SparseFlowValidationError):
        load_annotations(output, expected_sha256="1" * 64)
    image.write_bytes(b"changed")
    with pytest.raises(SparseFlowValidationError):
        load_annotations(output, image_path=image, verify_hash=True)


def test_rejects_non_root_to_tip_and_zero_or_one_confidence(tmp_path) -> None:
    for value in (False, -0.01, 1.01):
        payload = _valid_payload()
        payload["arrows"][0]["root_to_tip"] = value if isinstance(value, bool) else True
        payload["arrows"][0]["confidence"] = value
        with pytest.raises(SparseFlowValidationError):
            load_annotations(_write_payload(tmp_path, payload))

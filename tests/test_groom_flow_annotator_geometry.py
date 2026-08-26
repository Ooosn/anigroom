from __future__ import annotations

from PIL import Image
import pytest

from tools.groom_flow_annotator import FlowAnnotatorApp


def _transform_app() -> FlowAnnotatorApp:
    app = object.__new__(FlowAnnotatorApp)
    app.current_image = Image.new("RGB", (320, 180))
    app.scale = 2.0
    app.offset_x = 15.0
    app.offset_y = 25.0
    return app


def test_canvas_image_coordinate_roundtrip() -> None:
    app = _transform_app()
    canvas = app.image_to_canvas(123.25, 77.5)
    image = app.canvas_to_image(*canvas)
    assert image == pytest.approx((123.25, 77.5))


def test_canvas_to_image_rejects_or_clamps_outside_points() -> None:
    app = _transform_app()
    assert app.canvas_to_image(0.0, 0.0) is None
    assert app.canvas_to_image(0.0, 0.0, clamp=True) == (0.0, 0.0)
    assert app.canvas_to_image(10000.0, 10000.0, clamp=True) == (319.0, 179.0)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((5.0, 0.0), 0.0),
        ((5.0, 3.0), 3.0),
        ((-2.0, 0.0), 2.0),
        ((12.0, 0.0), 2.0),
    ],
)
def test_point_segment_distance(point: tuple[float, float], expected: float) -> None:
    distance = FlowAnnotatorApp._point_segment_distance(point, (0.0, 0.0), (10.0, 0.0))
    assert distance == pytest.approx(expected)

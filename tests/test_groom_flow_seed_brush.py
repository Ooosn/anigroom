from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from tools.groom_flow_annotator import CanvasSeed, FlowAnnotatorApp


def _app() -> FlowAnnotatorApp:
    app = object.__new__(FlowAnnotatorApp)
    app.current_image = Image.new("RGB", (320, 180))
    app.scale = 2.0
    app.offset_x = 15.0
    app.offset_y = 25.0
    return app


def test_canvas_image_coordinate_roundtrip() -> None:
    app = _app()
    canvas = app.image_to_canvas(123.25, 77.5)
    assert app.canvas_to_image(*canvas) == pytest.approx((123.25, 77.5))


def test_canvas_to_image_rejects_or_clamps_outside_points() -> None:
    app = _app()
    assert app.canvas_to_image(0.0, 0.0) is None
    assert app.canvas_to_image(0.0, 0.0, clamp=True) == (0.0, 0.0)
    assert app.canvas_to_image(10000.0, 10000.0, clamp=True) == (319.0, 179.0)


def test_stroke_sampling_is_even_and_includes_endpoint() -> None:
    samples = FlowAnnotatorApp.stroke_samples((0.0, 0.0), (10.0, 0.0), spacing=3.0)
    assert samples[-1] == (10.0, 0.0)
    gaps = np.diff([0.0] + [point[0] for point in samples])
    assert gaps.max() - gaps.min() < 1.0e-8


def test_canvas_seed_has_no_length_or_endpoint_field() -> None:
    seed = CanvasSeed("a", (1.0, 2.0), (0.0, 1.0), False)
    assert not hasattr(seed, "length_px")
    assert not hasattr(seed, "end_px")


def test_traditional_drag_arrow_produces_only_a_unit_direction() -> None:
    direction = FlowAnnotatorApp.drag_direction((10.0, 20.0), (13.0, 24.0))
    assert direction == pytest.approx((0.6, 0.8))
    assert FlowAnnotatorApp.drag_direction((1.0, 1.0), (1.5, 1.5)) is None


def test_modifier_wheel_control_adjustment_is_bounded() -> None:
    adjust = FlowAnnotatorApp.adjusted_control_value
    assert adjust(50.0, 1, step=4.0, minimum=8.0, maximum=180.0) == 54.0
    assert adjust(179.0, 1, step=4.0, minimum=8.0, maximum=180.0) == 180.0
    assert adjust(0.06, -1, step=0.04, minimum=0.05, maximum=1.0) == 0.05


@pytest.mark.parametrize(("point", "expected"), [((5.0, 0.0), 0.0), ((5.0, 3.0), 3.0), ((-2.0, 0.0), 2.0), ((12.0, 0.0), 2.0)])
def test_point_segment_distance(point, expected) -> None:
    assert FlowAnnotatorApp._point_segment_distance(point, (0.0, 0.0), (10.0, 0.0)) == pytest.approx(expected)

from __future__ import annotations

import numpy as np
from PIL import Image

from tools.visualize_white_tiger_groom_attributes import (
    _overlay_direction_colors,
    _screen_direction_colors,
)


def test_screen_direction_colors_follow_clockwise_image_axes() -> None:
    vectors = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    colors, magnitude = _screen_direction_colors(vectors)

    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(colors[2], [0.0, 1.0, 1.0], atol=1.0e-6)
    assert int(np.argmax(colors[1])) == 1
    assert int(np.argmax(colors[3])) in {0, 2}
    np.testing.assert_allclose(magnitude, np.ones(4), atol=1.0e-6)


def test_screen_direction_color_is_scale_invariant_and_zero_is_gray() -> None:
    colors, magnitude = _screen_direction_colors(
        np.asarray([[1.0, 1.0], [5.0, 5.0], [0.0, 0.0]], dtype=np.float32)
    )

    np.testing.assert_allclose(colors[0], colors[1], atol=1.0e-6)
    np.testing.assert_allclose(colors[2], [0.5, 0.5, 0.5], atol=1.0e-6)
    assert float(magnitude[1]) > float(magnitude[0])


def test_direction_overlay_writes_canonical_image(tmp_path) -> None:
    base = Image.new("RGB", (640, 360), (240, 240, 240))
    xy = np.asarray([[120.0, 150.0], [320.0, 180.0], [520.0, 220.0]])
    vectors = np.asarray([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
    output = tmp_path / "direction.png"

    report = _overlay_direction_colors(
        base,
        xy,
        vectors,
        title="direction test",
        out_path=output,
    )

    assert output.is_file()
    assert Image.open(output).size == base.size
    assert report["visible_roots"] == 3
    assert report["zero_projected_count"] == 0

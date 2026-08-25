from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest

from paper.method.render_parametric_fur_figure import (
    COMPOSED_CAMERA_OFFSET,
    COMPOSED_OPACITY_PROFILES,
    COMPOSED_GROUND_SCREEN_HEIGHT,
    COMPOSED_IMAGE_SHIFT,
    COMPOSED_ORTHO_SCALE,
    COMPOSED_ROOT_SPACING,
    COMPOSED_ROW_HEIGHT_RATIO,
    COMPOSED_TARGET_ROOT_OFFSET,
    COMPOSED_WAVE_AMPLITUDE,
    COMPOSED_WAVE_FREQUENCY,
    CONTROL_PANEL_BOTTOM_CROP,
    CONTROL_PANEL_IMAGE_TOP,
    CONTROL_PANEL_TOP_CROP,
    OPACITY_PANEL_COLOR,
    OPACITY_SWATCH_CHECKER_COLUMNS,
    OPACITY_SWATCH_CHECKER_DARK,
    OPACITY_SWATCH_CHECKER_LIGHT,
    Panel,
    StrandSpec,
    TOP_OPACITY_PROFILES,
    VALUE_SWATCH_ENDPOINT_SPAN,
    VALUE_SWATCH_LINE_WIDTH,
    VALUE_SWATCH_SEGMENT_COUNT,
    VALUE_SWATCH_Y,
    build_composed_scene_arrays,
    build_value_strands,
    composed_panel,
    control_panels,
    opacity_swatch_alpha_profile,
    opacity_swatch_checkerboard,
    presentation_command,
    render_composed_scene,
)
from paper.method.render_parametric_groom_blender import (
    constant_alpha_node_metadata,
    root_tip_alpha_node_metadata,
    sample_strands,
)


def test_figure_specs_replace_frizz_with_three_opacity_profiles() -> None:
    assert "frizz" not in StrandSpec.__dataclass_fields__
    panels = control_panels()
    assert [panel.key for panel in panels] == [
        "direction",
        "length",
        "stiffness",
        "width",
        "curl_radius",
        "curl_turns",
        "opacity",
        "appearance",
    ]
    opacity = panels[6]
    assert opacity.title == "Root-tip opacity"
    assert opacity.labels == ("", "", "")
    assert tuple((spec.root_opacity, spec.tip_opacity) for spec in opacity.specs) == TOP_OPACITY_PROFILES
    assert all(spec.root_color == OPACITY_PANEL_COLOR for spec in opacity.specs)
    assert all(spec.tip_color == OPACITY_PANEL_COLOR for spec in opacity.specs)
    assert all(spec.root_color == spec.tip_color for spec in opacity.specs)
    assert len({(spec.root_color, spec.tip_color) for spec in opacity.specs}) == 1
    geometry_and_appearance = tuple(
        replace(spec, root_opacity=1.0, tip_opacity=1.0) for spec in opacity.specs
    )
    assert geometry_and_appearance[0] == geometry_and_appearance[1] == geometry_and_appearance[2]
    assert "frizz=" not in inspect.getsource(control_panels)
    assert "frizz=" not in inspect.getsource(composed_panel)


def test_opacity_swatch_checker_and_alpha_profiles_are_deterministic() -> None:
    checker = opacity_swatch_checkerboard()
    assert checker.shape == (2, VALUE_SWATCH_SEGMENT_COUNT, 4)
    np.testing.assert_allclose(checker[0, 0], OPACITY_SWATCH_CHECKER_LIGHT, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(checker[0, 5], OPACITY_SWATCH_CHECKER_DARK, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(checker[1, 0], OPACITY_SWATCH_CHECKER_DARK, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(checker[1, 5], OPACITY_SWATCH_CHECKER_LIGHT, rtol=0.0, atol=1e-7)
    assert VALUE_SWATCH_SEGMENT_COUNT % OPACITY_SWATCH_CHECKER_COLUMNS == 0

    alpha = opacity_swatch_alpha_profile(1.0, 0.35)
    assert alpha.shape == (VALUE_SWATCH_SEGMENT_COUNT,)
    assert alpha[0] == pytest.approx(1.0)
    assert alpha[-1] == pytest.approx(0.35)
    assert np.all(np.diff(alpha) < 0.0)
    np.testing.assert_allclose(
        opacity_swatch_alpha_profile(1.0, 0.0)[[0, -1]],
        [1.0, 0.0],
    )


def test_opacity_swatch_geometry_matches_root_tip_color_swatch_contract() -> None:
    assert VALUE_SWATCH_ENDPOINT_SPAN == 0.170
    assert VALUE_SWATCH_Y == 0.083
    assert VALUE_SWATCH_LINE_WIDTH == 3.2
    positions = np.linspace(1.0 / 6.0, 5.0 / 6.0, 3)
    spans = tuple(
        (
            float(position - VALUE_SWATCH_ENDPOINT_SPAN / 2.0),
            float(position + VALUE_SWATCH_ENDPOINT_SPAN / 2.0),
            VALUE_SWATCH_Y,
        )
        for position in positions
    )
    assert spans[0] == pytest.approx((0.0816666667, 0.2516666667, 0.083))
    assert spans[1] == pytest.approx((0.415, 0.585, 0.083))
    assert spans[2] == pytest.approx((0.7483333333, 0.9183333333, 0.083))

    appearance = control_panels()[-1]
    assert appearance.labels == ("color_0", "color_1", "color_2")
    assert appearance.specs[0].root_color == (0.68, 0.67, 0.63)
    assert appearance.specs[0].tip_color == (0.68, 0.67, 0.63)


def test_value_and_gaussian_npz_transport_root_tip_opacity_without_frizz() -> None:
    spec = StrandSpec(root_opacity=0.91, tip_opacity=0.23)
    arrays, report = build_value_strands(
        spec,
        adaptive_samples=True,
        gaussian_strand_index=1,
    )

    assert report["gaussian_strand_count"] == 1
    assert arrays["opacities"].shape == (3, 257, 1)
    np.testing.assert_allclose(arrays["opacities"][:, 0, 0], 0.91)
    np.testing.assert_allclose(arrays["opacities"][:, -1, 0], 0.23)
    assert arrays["gaussian_opacities"].shape == (
        arrays["gaussian_means"].shape[0],
    )
    assert np.all((arrays["gaussian_opacities"] >= 0.0) & (arrays["gaussian_opacities"] <= 1.0))


def test_composed_profiles_are_exact_and_nonmonotonic() -> None:
    def is_monotonic(values: np.ndarray) -> bool:
        differences = np.diff(values)
        return bool(np.all(differences >= 0.0) or np.all(differences <= 0.0))

    expected_opacities = (
        (1.00, 1.00),
        (0.96, 0.28),
        (0.62, 0.95),
        (0.88, 0.48),
        (0.76, 0.35),
    )
    assert COMPOSED_OPACITY_PROFILES == expected_opacities
    assert all(root != tip for root, tip in expected_opacities[1:])

    expected_colors = {
        "smoked_champagne": (
            ((0.080, 0.055, 0.035), (0.420, 0.310, 0.200)),
            ((0.145, 0.105, 0.070), (0.660, 0.550, 0.400)),
            ((0.540, 0.430, 0.300), (0.120, 0.085, 0.055)),
            ((0.100, 0.065, 0.040), (0.460, 0.340, 0.220)),
            ((0.150, 0.110, 0.075), (0.700, 0.620, 0.480)),
        ),
        "copper": (
            ((0.170, 0.075, 0.025), (0.720, 0.360, 0.075)),
            ((0.120, 0.050, 0.018), (0.625, 0.285, 0.060)),
            ((0.170, 0.075, 0.025), (0.780, 0.465, 0.135)),
            ((0.215, 0.095, 0.025), (0.620, 0.275, 0.055)),
            ((0.135, 0.060, 0.025), (0.800, 0.530, 0.190)),
        ),
    }
    for palette, colors in expected_colors.items():
        panel = composed_panel(palette=palette)
        assert tuple((spec.root_color, spec.tip_color) for spec in panel.specs) == colors
        assert tuple((spec.root_opacity, spec.tip_opacity) for spec in panel.specs) == expected_opacities
        for endpoint in (0, 1):
            luminance = np.asarray(
                [np.dot(profile[endpoint], (0.2126, 0.7152, 0.0722)) for profile in colors]
            )
            assert not is_monotonic(luminance)

    for endpoint in (0, 1):
        values = np.asarray([profile[endpoint] for profile in COMPOSED_OPACITY_PROFILES])
        assert np.all((values >= 0.0) & (values <= 1.0))
        assert not is_monotonic(values)

    panel = composed_panel()
    assert panel.resolution == (4320, 700)
    assert panel.frame_margin == 1.08
    assert panel.ortho_scale == 0.82
    assert panel.reference_extent == 0.92
    _, scene_report = build_composed_scene_arrays(panel, gaussian_outlines=False)
    np.testing.assert_allclose(
        [group["root_position"] for group in scene_report["groups"]],
        [
            (-1.64, 0.0, 0.035),
            (-0.82, 0.0, -0.035),
            (0.0, 0.0, 0.035),
            (0.82, 0.0, -0.035),
            (1.64, 0.0, 0.035),
        ],
        rtol=0.0,
        atol=1e-6,
    )


def test_composed_appearance_is_shared_by_each_grooms_three_strands() -> None:
    panel = composed_panel()
    assert panel.labels == (
        "sleek taper",
        "swept plume",
        "ribbon wave",
        "compact coil",
        "airy fade",
    )
    arrays, _ = build_composed_scene_arrays(panel, gaussian_outlines=False)
    gaussian, _ = build_composed_scene_arrays(panel, gaussian_outlines=True)
    np.testing.assert_allclose(arrays["strands"], gaussian["strands"])
    np.testing.assert_allclose(arrays["opacities"], gaussian["opacities"])

    for index, spec in enumerate(panel.specs):
        assert (spec.root_opacity, spec.tip_opacity) == COMPOSED_OPACITY_PROFILES[index]
        group = slice(3 * index, 3 * (index + 1))
        np.testing.assert_allclose(
            arrays["colors"][group, 0],
            np.repeat(np.asarray(spec.root_color)[None, :], 3, axis=0),
        )
        np.testing.assert_allclose(
            arrays["colors"][group, -1],
            np.repeat(np.asarray(spec.tip_color)[None, :], 3, axis=0),
        )
        np.testing.assert_allclose(
            arrays["opacities"][group, 0, 0],
            np.full(3, spec.root_opacity),
        )
        np.testing.assert_allclose(
            arrays["opacities"][group, -1, 0],
            np.full(3, spec.tip_opacity),
        )

    assert gaussian["gaussian_opacities"].shape == (
        gaussian["gaussian_means"].shape[0],
    )
    assert np.all(
        (gaussian["gaussian_opacities"] >= 0.0)
        & (gaussian["gaussian_opacities"] <= 1.0)
    )


def test_composed_lighting_matches_accepted_original_settings() -> None:
    composed_source = inspect.getsource(render_composed_scene)
    assert "ground_color=(1.0, 1.0, 1.0)" in composed_source
    assert "world_strength=0.32" in composed_source
    assert 'key_light_type="sun"' in composed_source
    assert "key_light_energy=3.4" in composed_source
    assert "key_light_offset=(-1.00, -0.05, 1.25)" in composed_source
    assert "sun_angle_deg=4.0" in composed_source
    assert "fill_light_energy=110.0" in composed_source
    assert "fill_light_size=8.0" in composed_source
    assert "shadow_sun_energy=0.0" in composed_source
    assert "shadow_sun_offset=" not in composed_source
    assert "shadow_sun_angle_deg=" not in composed_source
    assert "use_input_opacities=True" in composed_source
    assert "key_light_size=" not in composed_source

    presentation_source = inspect.getsource(presentation_command)
    assert "key_light_size:" not in presentation_source

    command = presentation_command(
        blender="blender",
        renderer="renderer.py",
        npz_path="input.npz",
        image_path="output.png",
        resolution=(360, 620),
        render_samples=64,
        camera_offset=(0.0, -1.0, 0.09),
        target_root_offset=(0.1, 0.0, 0.2),
        frame_margin=1.2,
        reference_extent=0.95,
        ortho_scale=1.95,
        ground_relief=0.040,
        ground_width_scale=2.40,
        ground_depth_scale=0.55,
        ground_screen_height=0.10,
        gaussian_outlines=False,
    )
    size_index = command.index("--key-light-size")
    assert command[size_index + 1] == "1.6"


def test_sample_strands_transports_opacity_rows_with_sampling() -> None:
    strands = np.arange(5 * 3 * 3, dtype=np.float32).reshape(5, 3, 3)
    widths = np.arange(5 * 3, dtype=np.float32).reshape(5, 3, 1)
    colors = np.arange(5 * 3 * 3, dtype=np.float32).reshape(5, 3, 3)
    opacities = np.arange(5 * 3, dtype=np.float32).reshape(5, 3, 1) / 20.0

    sampled = sample_strands(strands, widths, colors, opacities, 2, 29)
    sampled_strands, sampled_widths, sampled_colors, sampled_opacities = sampled
    selected = np.sort(np.random.default_rng(29).choice(5, size=2, replace=False))
    np.testing.assert_array_equal(sampled_strands, strands[selected])
    np.testing.assert_array_equal(sampled_widths, widths[selected])
    np.testing.assert_array_equal(sampled_colors, colors[selected])
    np.testing.assert_array_equal(sampled_opacities, opacities[selected])


def test_renderer_opacity_metadata_contract() -> None:
    root_tip = root_tip_alpha_node_metadata(1.0, 0.23)
    assert root_tip["mapping_node"] == "ShaderNodeMapRange"
    assert root_tip["mapping_to"] == [1.0, 0.23]
    assert root_tip["transparent_node"] == "ShaderNodeBsdfTransparent"
    assert root_tip["surface_node"] == "ShaderNodeMixShader"
    assert root_tip["mix_factor_semantics"] == "0=transparent, 1=principled"

    constant = constant_alpha_node_metadata(0.64)
    assert constant["constant_opacity"] == 0.64
    assert constant["transparent_node"] == "ShaderNodeBsdfTransparent"
    assert constant["surface_node"] == "ShaderNodeMixShader"

    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        root_tip_alpha_node_metadata(1.01, 0.5)


def test_frozen_layout_and_presentation_command_are_unchanged_except_opacity_flag() -> None:
    assert CONTROL_PANEL_TOP_CROP == 0.40125
    assert CONTROL_PANEL_BOTTOM_CROP == 0.975
    assert CONTROL_PANEL_IMAGE_TOP == 0.910
    assert COMPOSED_ROW_HEIGHT_RATIO == 1.46
    assert COMPOSED_ROOT_SPACING == 0.82
    assert COMPOSED_WAVE_AMPLITUDE == 0.035
    assert COMPOSED_WAVE_FREQUENCY == np.pi / 0.82
    assert COMPOSED_ORTHO_SCALE == 4.20
    assert COMPOSED_CAMERA_OFFSET == (0.0, -1.0, 0.34)
    assert COMPOSED_TARGET_ROOT_OFFSET == (0.30, 0.0, 0.19)
    assert COMPOSED_GROUND_SCREEN_HEIGHT == 0.20
    assert COMPOSED_IMAGE_SHIFT == pytest.approx(
        (1.0 - (1.0 - 0.910) / 1.46) - 0.910
    )

    panel = Panel(
        "test",
        "Test",
        "",
        "",
        ("a", "b", "c"),
        (StrandSpec(), StrandSpec(), StrandSpec()),
        resolution=(1080, 620),
        frame_margin=1.2,
        ortho_scale=1.95,
        reference_extent=0.95,
    )
    command = presentation_command(
        blender="blender",
        renderer="renderer.py",
        npz_path="input.npz",
        image_path="output.png",
        resolution=(360, 620),
        render_samples=64,
        camera_offset=(0.0, -1.0, 0.09),
        target_root_offset=(0.1, 0.0, 0.2),
        frame_margin=panel.frame_margin,
        reference_extent=panel.reference_extent,
        ortho_scale=panel.ortho_scale,
        ground_relief=0.040,
        ground_width_scale=2.40,
        ground_depth_scale=0.55,
        ground_screen_height=0.10,
        gaussian_outlines=False,
    )
    assert "--use-input-colors" in command
    assert "--use-input-opacities" in command

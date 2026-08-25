from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest
import paper.method.render_parametric_groom_blender as blender_renderer

from paper.method.render_parametric_fur_figure import (
    COMPOSED_CAMERA_OFFSET,
    COMPOSED_GROUND_SCREEN_HEIGHT,
    COMPOSED_IMAGE_SHIFT,
    COMPOSED_ORTHO_SCALE,
    COMPOSED_OPACITY_PROFILES,
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
    palette_colors,
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


def test_composed_profiles_are_distinct_and_spatial_alignment_is_unchanged() -> None:
    for palette in ("smoked_champagne", "copper"):
        panel = composed_panel(palette=palette)
        assert panel.labels == (
            "sleek taper",
            "swept plume",
            "ribbon wave",
            "compact coil",
            "airy fade",
        )
        profiles = tuple((spec.root_opacity, spec.tip_opacity) for spec in panel.specs)
        assert profiles == COMPOSED_OPACITY_PROFILES
        assert len(set(profiles)) == 5
        root_color, tip_color = palette_colors(palette)
        actual_roots = np.asarray([spec.root_color for spec in panel.specs])
        actual_tips = np.asarray([spec.tip_color for spec in panel.specs])
        progress = np.linspace(0.0, 1.0, len(panel.specs))[:, None]
        expected_tips = (
            np.asarray(root_color)[None, :]
            + progress * (np.asarray(tip_color)[None, :] - np.asarray(root_color)[None, :])
        )
        np.testing.assert_allclose(
            actual_roots,
            np.repeat(np.asarray(root_color)[None, :], len(panel.specs), axis=0),
        )
        np.testing.assert_allclose(actual_tips, expected_tips)
        assert np.all(np.diff(actual_tips, axis=0) >= -1e-8)

        strands, _ = build_composed_scene_arrays(panel, gaussian_outlines=False)
        gaussian, _ = build_composed_scene_arrays(panel, gaussian_outlines=True)
        np.testing.assert_allclose(strands["strands"], gaussian["strands"])
        np.testing.assert_allclose(strands["opacities"], gaussian["opacities"])
        assert gaussian["gaussian_opacities"].shape == (
            gaussian["gaussian_means"].shape[0],
        )
        assert np.all(
            (gaussian["gaussian_opacities"] >= 0.0)
            & (gaussian["gaussian_opacities"] <= 1.0)
        )


def test_composed_only_lighting_is_broad_area_and_top_defaults_are_unchanged() -> None:
    composed_source = inspect.getsource(render_composed_scene)
    assert 'key_light_type="area"' in composed_source
    assert "key_light_energy=900.0" in composed_source
    assert "key_light_size=4.5" in composed_source
    assert 'key_light_type="sun"' not in composed_source

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


def test_alpha_node_metadata_requires_true_transparent_mix() -> None:
    root_tip = root_tip_alpha_node_metadata(1.0, 0.23)
    assert root_tip["driver_node"] == "ShaderNodeHairInfo"
    assert root_tip["driver_output"] == "Intercept"
    assert root_tip["driver_semantics"] == "Hair Info intercept from root (0) to tip (1)"
    assert root_tip["hair_info_node"] == "ShaderNodeHairInfo"
    assert root_tip["hair_info_output"] == "Intercept"
    assert root_tip["color_alpha_shared_driver"] is True
    assert "Parametric" not in root_tip
    assert "ShaderNodeNewGeometry" not in root_tip
    assert root_tip["mapping_node"] == "ShaderNodeMapRange"
    assert root_tip["mapping_to"] == [1.0, 0.23]
    assert root_tip["transparent_node"] == "ShaderNodeBsdfTransparent"
    assert root_tip["surface_node"] == "ShaderNodeMixShader"

    constant = constant_alpha_node_metadata(0.64)
    assert constant["constant_opacity"] == 0.64
    assert constant["transparent_node"] == "ShaderNodeBsdfTransparent"
    assert constant["surface_node"] == "ShaderNodeMixShader"

    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        root_tip_alpha_node_metadata(1.01, 0.5)

    alpha_source = inspect.getsource(blender_renderer._connect_alpha_surface)
    curve_source = inspect.getsource(blender_renderer.add_strand_curve_objects)
    material_source = inspect.getsource(blender_renderer.make_root_tip_material)
    assert "intercept_socket" in alpha_source
    assert 'links.new(intercept_socket, alpha_map.inputs["Value"])' in alpha_source
    assert "ShaderNodeNewGeometry" not in alpha_source
    assert "Parametric" not in alpha_source
    assert 'bpy.data.hair_curves.new' in curve_source
    assert 'hair_info.outputs["Intercept"]' in material_source


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

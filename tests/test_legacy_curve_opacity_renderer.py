from __future__ import annotations

import inspect

import pytest

import paper.method.render_parametric_groom_blender as renderer


def test_strand_curve_construction_is_legacy_curve_poly_with_bevel_radius() -> None:
    source = inspect.getsource(renderer.add_strand_curve_objects)
    signature = inspect.signature(renderer.add_strand_curve_objects)

    assert "bpy.data.curves.new" in source
    assert '"CURVE"' in source
    assert 'curve.splines.new("POLY")' in source
    assert "curve.bevel_depth = base_width" in source
    assert "point.radius = float(rad)" in source
    assert "use_native_hair_intercept" not in signature.parameters
    assert "hair_curves" not in source
    assert "native_hair" not in source


def test_renderer_has_no_native_hair_compatibility_symbols() -> None:
    source = inspect.getsource(renderer)

    assert "hair_curves" not in source
    assert "native_hair" not in source
    assert "use_native_hair_intercept" not in source
    assert "curve.add_curves" not in source
    assert "hair_curve.points" not in source


def test_root_tip_alpha_metadata_uses_legacy_geometry_parametric_driver() -> None:
    metadata = renderer.root_tip_alpha_node_metadata(1.0, 0.23)

    assert metadata == {
        "driver_node": "ShaderNodeNewGeometry",
        "driver_output": "Parametric",
        "driver_semantics": "curve intercept from root (0) to tip (1)",
        "hair_info_node": "ShaderNodeHairInfo",
        "hair_info_output": "Intercept",
        "mapping_node": "ShaderNodeMapRange",
        "mapping_from": [0.0, 1.0],
        "mapping_to": [1.0, 0.23],
        "transparent_node": "ShaderNodeBsdfTransparent",
        "surface_node": "ShaderNodeMixShader",
        "mix_factor_semantics": "0=transparent, 1=principled",
    }

    source = inspect.getsource(renderer._connect_alpha_surface)
    root_material_source = inspect.getsource(renderer.make_root_tip_material)
    assert 'nodes.new("ShaderNodeNewGeometry")' in source
    assert 'curve_intercept.outputs["Parametric"]' in source
    assert 'links.new(alpha_map.outputs["Result"], mix_shader.inputs[0])' in source
    assert 'transparent.outputs["BSDF"], mix_shader.inputs[1]' in source
    assert 'bsdf.outputs["BSDF"], mix_shader.inputs[2]' in source
    assert 'hair_info.outputs["Intercept"]' in root_material_source


def test_constant_alpha_metadata_and_true_transparent_mix_remain_unchanged() -> None:
    metadata = renderer.constant_alpha_node_metadata(0.64)

    assert metadata == {
        "driver_node": None,
        "mapping_node": None,
        "constant_opacity": 0.64,
        "transparent_node": "ShaderNodeBsdfTransparent",
        "surface_node": "ShaderNodeMixShader",
        "mix_factor_semantics": "0=transparent, 1=principled",
    }
    source = inspect.getsource(renderer.make_constant_alpha_material)
    assert "_connect_alpha_surface(material=material, bsdf=bsdf, opacity=opacity)" in source


@pytest.mark.parametrize(
    ("root", "tip", "message"),
    [
        (-0.01, 0.5, "root_opacity"),
        (1.01, 0.5, "root_opacity"),
        (0.5, -0.01, "tip_opacity"),
        (0.5, 1.01, "tip_opacity"),
    ],
)
def test_root_tip_opacity_validation_rejects_values_outside_unit_interval(
    root: float,
    tip: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        renderer.root_tip_alpha_node_metadata(root, tip)


def test_constant_opacity_validation_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        renderer.constant_alpha_node_metadata(float("nan"))

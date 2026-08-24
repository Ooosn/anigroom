from __future__ import annotations

import numpy as np
import pytest
import torch

from anigroom.grooming import (
    GroomParameterField,
    adaptive_resample_strands,
    build_strands,
    make_tangent_frames,
    resample_strands_to_segment_budgets,
    strand_segment_budgets,
)
from tools.diagnose_curl_components import fixed_turn_curl, select_curl
from tools.visualize_white_tiger_groom_attributes import (
    _signed_colormap,
    project_primary_guide_curl_turns,
    summarize_attribute_values,
)


def test_fixed_turn_counterfactual_changes_only_requested_curl_controls() -> None:
    field = GroomParameterField(3, init_length=0.02, device="cpu")
    with torch.no_grad():
        field.curl_turns_raw.copy_(torch.tensor([[-2.0], [0.5], [3.0]]))
        field.curl_phase.copy_(torch.tensor([[0.4], [-0.3], [0.1]]))
    groom = field.decode()

    learned = select_curl(groom, enabled=True)
    fixed = fixed_turn_curl(groom)

    for name in (
        "length",
        "root_width",
        "tip_width",
        "width_taper",
        "direction_local",
        "brush_stiffness",
        "curl_radius_ratio",
        "child_radius",
        "clump_strength",
        "root_color",
        "tip_color",
        "root_opacity",
        "tip_opacity",
        "opacity",
    ):
        torch.testing.assert_close(getattr(fixed, name), getattr(learned, name))
    torch.testing.assert_close(
        fixed.curl_phase,
        torch.zeros_like(fixed.curl_phase),
    )
    torch.testing.assert_close(
        fixed.curl_turns,
        torch.full_like(fixed.curl_turns, 1.2),
        rtol=0.0,
        atol=0.0,
    )

    roots = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=torch.float32,
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]]).expand_as(roots)
    tangents, bitangents = make_tangent_frames(normals)
    learned_output = build_strands(
        roots, normals, tangents, bitangents, learned, samples=17
    )
    fixed_output = build_strands(
        roots, normals, tangents, bitangents, fixed, samples=17
    )
    for learned_value, fixed_value in zip(
        learned_output[1:], fixed_output[1:], strict=True
    ):
        torch.testing.assert_close(learned_value, fixed_value, rtol=0.0, atol=0.0)


def test_primary_guide_projection_and_signed_statistics_are_shape_safe() -> None:
    guide_points_local = torch.tensor(
        [[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [-0.5, 0.0, 1.0]],
        dtype=torch.float32,
    )
    guide_turns = torch.tensor([[-1.5], [0.0], [2.5]], dtype=torch.float32)
    viewmat = torch.eye(4, dtype=torch.float32)
    intrinsics = torch.tensor(
        [[2.0, 0.0, 5.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    mesh_depth = torch.full((10, 10), 2.1, dtype=torch.float32)

    points_world, xy, values, visible = project_primary_guide_curl_turns(
        guide_points_local,
        guide_turns,
        log_scale=torch.log(torch.tensor(2.0)),
        translation=torch.tensor([0.2, 0.1, 0.0]),
        viewmat=viewmat,
        k=intrinsics,
        width=10,
        height=10,
        mesh_depth=mesh_depth,
        mesh_depth_kernel=1,
    )

    assert points_world.shape == (3, 3)
    assert xy.shape == (3, 2)
    assert values.shape == (3,)
    assert visible.shape == (3,)
    assert visible.tolist() == [True, True, True]
    torch.testing.assert_close(points_world[:, 2], torch.full((3,), 2.0))
    torch.testing.assert_close(values, guide_turns.reshape(-1))

    stats = summarize_attribute_values(values.numpy())
    assert stats["min"] == -1.5
    assert stats["p50"] == 0.0
    assert stats["max"] == 2.5

    colors = _signed_colormap(np.asarray([-1.0, 0.0, 1.0], dtype=np.float32))
    assert colors.shape == (3, 3)
    assert not np.array_equal(colors[0], colors[2])

    with pytest.raises(ValueError, match="one value per guide"):
        project_primary_guide_curl_turns(
            guide_points_local,
            torch.zeros((2, 1)),
            log_scale=torch.tensor(0.0),
            translation=torch.zeros(3),
            viewmat=viewmat,
            k=intrinsics,
            width=10,
            height=10,
        )


def test_adaptive_resample_preserves_final_curl_and_attribute_transport() -> None:
    field = GroomParameterField(2, init_length=0.04, device="cpu")
    with torch.no_grad():
        field.curl_radius_ratio_raw.fill_(0.45)
        field.curl_turns_raw.copy_(torch.tensor([[1.5], [-2.0]]))
        field.curl_phase.copy_(torch.tensor([[0.2], [-0.4]]))
        field.root_color_raw.copy_(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        )
        field.tip_color_raw.copy_(
            torch.tensor([[0.7, 0.6, 0.5], [0.3, 0.2, 0.1]])
        )
        field.opacity_raw.copy_(torch.tensor([[-0.3], [0.6]]))
        field.tip_opacity_ratio_raw.copy_(torch.tensor([[-0.2], [0.4]]))
    groom = field.decode()
    roots = torch.tensor(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32).expand_as(roots)
    tangents, bitangents = make_tangent_frames(normals)
    strands, widths, colors, opacities = build_strands(
        roots,
        normals,
        tangents,
        bitangents,
        groom,
        samples=65,
    )

    counts, budget_stats = strand_segment_budgets(
        strands,
        groom.length,
        min_segments=4,
        length_origin=0.01,
        segments_per_unit_length=24.0,
        segments_per_unit_complexity=80.0,
    )
    adaptive = adaptive_resample_strands(
        strands,
        widths,
        colors,
        opacities,
        groom.length,
        min_segments=4,
        length_origin=0.01,
        segments_per_unit_length=24.0,
        segments_per_unit_complexity=80.0,
    )
    direct = resample_strands_to_segment_budgets(
        strands,
        widths,
        colors,
        opacities,
        counts,
    )

    torch.testing.assert_close(adaptive.strands, direct.strands, rtol=0.0, atol=0.0)
    torch.testing.assert_close(adaptive.widths, direct.widths, rtol=0.0, atol=0.0)
    torch.testing.assert_close(adaptive.colors, direct.colors, rtol=0.0, atol=0.0)
    torch.testing.assert_close(adaptive.opacities, direct.opacities, rtol=0.0, atol=0.0)
    torch.testing.assert_close(adaptive.segment_mask, direct.segment_mask)
    torch.testing.assert_close(adaptive.segment_counts, counts)
    assert adaptive.stats["adaptive_min_segments"] == budget_stats["adaptive_min_segments"]
    assert adaptive.stats["adaptive_max_segments"] == budget_stats["adaptive_max_segments"]
    assert adaptive.stats["adaptive_mean_segments"] == budget_stats["adaptive_mean_segments"]
    assert bool(torch.isfinite(adaptive.strands).all())
    assert bool(torch.isfinite(adaptive.widths).all())
    assert bool(torch.isfinite(adaptive.colors).all())
    assert bool(torch.isfinite(adaptive.opacities).all())

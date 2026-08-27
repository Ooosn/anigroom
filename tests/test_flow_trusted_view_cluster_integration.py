from __future__ import annotations

import argparse
import importlib
import inspect

import pytest
import torch


MODULE_NAME = "tools.fuse_gpt_flow_shell_multiview"


def _module() -> object:
    return importlib.import_module(MODULE_NAME)


def test_accumulate_axis_evidence_returns_exact_mutation_deltas_on_cpu() -> None:
    module = _module()
    flow3d_sum = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]],
        ]
    )
    weight_sum = torch.tensor([[0.0, 0.0], [0.0, 0.4]])
    view_count = torch.zeros((2, 2))
    flow3d_before = flow3d_sum.clone()
    weight_before = weight_sum.clone()
    count_before = view_count.clone()

    sampled_ori = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]
    )
    weight_flat = torch.tensor([0.5, 0.2, 0.05, 0.7])
    screen_t = torch.tensor([[1.0, 0.0]] * 4)
    screen_b = torch.tensor([[0.0, 1.0]] * 4)
    flat_tangents = torch.tensor([[1.0, 0.0, 0.0]] * 4)
    flat_bitangents = torch.tensor([[0.0, 1.0, 0.0]] * 4)

    result = module.accumulate_axis_evidence(
        flow3d_sum=flow3d_sum,
        weight_sum=weight_sum,
        view_count=view_count,
        sampled_ori=sampled_ori,
        weight_flat=weight_flat,
        screen_t=screen_t,
        screen_b=screen_b,
        flat_tangents=flat_tangents,
        flat_bitangents=flat_bitangents,
        n_roots=2,
        n_shells=2,
        min_confidence=0.1,
        capture_contribution=True,
    )

    assert len(result) == 4
    good_count, raw_weight, aligned_contribution, effective_weight = result
    assert good_count == 3
    assert raw_weight == pytest.approx(float(weight_flat.sum()))
    torch.testing.assert_close(
        aligned_contribution,
        torch.tensor(
            [
                [[0.5, 0.0, 0.0], [0.0, 0.2, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.7, 0.0]],
            ]
        ),
    )
    torch.testing.assert_close(
        effective_weight,
        torch.tensor([[0.5, 0.2], [0.0, 0.7]]),
    )
    torch.testing.assert_close(flow3d_sum, flow3d_before + aligned_contribution)
    torch.testing.assert_close(weight_sum, weight_before + effective_weight)
    torch.testing.assert_close(
        view_count,
        count_before + torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    )


def test_accumulate_axis_evidence_effective_weight_includes_observability() -> None:
    module = _module()
    flow3d_sum = torch.zeros((1, 1, 3))
    weight_sum = torch.zeros((1, 1))
    view_count = torch.zeros((1, 1))
    sampled_ori = torch.tensor([[0.0, 1.0]])
    weight_flat = torch.tensor([0.8])
    screen_t = torch.tensor([[1.0, 0.0]])
    screen_b = torch.tensor([[0.0, 0.25]])
    flat_tangents = torch.tensor([[1.0, 0.0, 0.0]])
    flat_bitangents = torch.tensor([[0.0, 1.0, 0.0]])

    observability = module.tangent_axis_observability(sampled_ori, screen_t, screen_b)
    _, _, _, effective_weight = module.accumulate_axis_evidence(
        flow3d_sum=flow3d_sum,
        weight_sum=weight_sum,
        view_count=view_count,
        sampled_ori=sampled_ori,
        weight_flat=weight_flat,
        screen_t=screen_t,
        screen_b=screen_b,
        flat_tangents=flat_tangents,
        flat_bitangents=flat_bitangents,
        n_roots=1,
        n_shells=1,
        min_confidence=0.1,
        capture_contribution=True,
    )

    torch.testing.assert_close(observability, torch.tensor([0.25]))
    torch.testing.assert_close(effective_weight, torch.tensor([[0.8 * 0.25]]))
    torch.testing.assert_close(weight_sum, effective_weight)


def test_accumulate_axis_evidence_default_path_does_not_retain_diagnostics() -> None:
    module = _module()
    flow_sum = torch.zeros((1, 1, 3))
    weight_sum = torch.zeros((1, 1))
    view_count = torch.zeros((1, 1))
    result = module.accumulate_axis_evidence(
        flow3d_sum=flow_sum,
        weight_sum=weight_sum,
        view_count=view_count,
        sampled_ori=torch.tensor([[1.0, 0.0]]),
        weight_flat=torch.tensor([0.5]),
        screen_t=torch.tensor([[1.0, 0.0]]),
        screen_b=torch.tensor([[0.0, 1.0]]),
        flat_tangents=torch.tensor([[1.0, 0.0, 0.0]]),
        flat_bitangents=torch.tensor([[0.0, 1.0, 0.0]]),
        n_roots=1,
        n_shells=1,
        min_confidence=0.1,
    )
    assert result[2] is None
    assert result[3] is None
    torch.testing.assert_close(flow_sum, torch.tensor([[[0.5, 0.0, 0.0]]]))
    torch.testing.assert_close(weight_sum, torch.tensor([[0.5]]))
    torch.testing.assert_close(view_count, torch.ones((1, 1)))


def test_collapse_per_view_shell_evidence_matches_additive_decomposition() -> None:
    module = _module()
    per_view_contribution = torch.tensor(
        [
            [
                [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
                [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            ],
            [
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
            ],
        ]
    )
    per_view_weight = torch.tensor(
        [
            [[4.0, 6.0], [10.0, 8.0]],
            [[2.0, 8.0], [4.0, 12.0]],
        ]
    )
    per_view_direct_weight = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )
    global_weight_sum = torch.tensor([[2.0, 4.0], [0.0, 3.0]])
    shell_probability = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    shell_sign = torch.tensor([[1.0, -1.0], [1.0, -1.0]])

    expected_vectors = torch.tensor(
        [
            [[0.25, -0.5625, 0.0], [-2.0 / 3.0, 0.0, 0.0]],
            [[0.125, 0.1875, 0.0], [1.0 / 3.0, 0.0, 0.0]],
        ]
    )
    expected_weight = torch.tensor([[5.5, 9.0], [6.5, 8.0]])
    expected_direct_weight = torch.tensor([[1.75, 3.5], [3.75, 5.5]])

    result_2d = module.collapse_per_view_shell_evidence(
        per_view_contribution=per_view_contribution,
        per_view_weight=per_view_weight,
        per_view_direct_weight=per_view_direct_weight,
        global_weight_sum=global_weight_sum,
        shell_probability=shell_probability,
        shell_sign=shell_sign,
    )
    result_3d = module.collapse_per_view_shell_evidence(
        per_view_contribution=per_view_contribution,
        per_view_weight=per_view_weight,
        per_view_direct_weight=per_view_direct_weight,
        global_weight_sum=global_weight_sum,
        shell_probability=shell_probability,
        shell_sign=shell_sign[..., None],
    )

    for result in (result_2d, result_3d):
        vectors, combined_weight, direct_weight = result
        assert all(torch.isfinite(value).all() for value in result)
        torch.testing.assert_close(vectors, expected_vectors)
        torch.testing.assert_close(combined_weight, expected_weight)
        torch.testing.assert_close(direct_weight, expected_direct_weight)
    for first, second in zip(result_2d, result_3d):
        torch.testing.assert_close(first, second)


def test_fusion_module_imports_trusted_refinement_and_parser_keeps_modes() -> None:
    module = _module()
    from anigroom.flow.view_cluster_refinement import (
        refine_fixed_axis_multiview_ratio,
        refine_trusted_multiview_axis_field,
    )

    assert module.refine_trusted_multiview_axis_field is refine_trusted_multiview_axis_field
    assert module.refine_fixed_axis_multiview_ratio is refine_fixed_axis_multiview_ratio

    class _ParserCaptured(Exception):
        pass

    captured: dict[str, object] = {}

    def capture_parse_args(
        parser: argparse.ArgumentParser,
        args: object = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        del args, namespace
        action = next(item for item in parser._actions if item.dest == "axis_field_mode")
        captured["choices"] = tuple(action.choices or ())
        captured["default"] = action.default
        raise _ParserCaptured

    original_parse_args = argparse.ArgumentParser.parse_args
    try:
        argparse.ArgumentParser.parse_args = capture_parse_args  # type: ignore[method-assign]
        with pytest.raises(_ParserCaptured):
            module.main()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args  # type: ignore[method-assign]

    assert captured["choices"] == ("raw", "anchor-propagated", "trusted-view-cluster")
    assert captured["default"] == "trusted-view-cluster"


def test_fusion_helpers_have_no_species_region_or_view_index_parameters() -> None:
    module = _module()
    helper_names = (
        "accumulate_axis_evidence",
        "collapse_per_view_shell_evidence",
        "refine_trusted_multiview_axis_field",
    )
    forbidden = {"species", "region", "view_index", "view_idx", "view_id"}

    for helper_name in helper_names:
        parameters = inspect.signature(getattr(module, helper_name)).parameters
        assert not forbidden.intersection(parameters)


def test_selected_point_collector_and_ratio_solver_are_wired_only_to_trusted_mode() -> None:
    module = _module()
    collector = inspect.signature(module.collect_selected_tangent_axis_evidence)
    assert tuple(collector.parameters) == (
        "args",
        "views",
        "selected_shell_points",
        "root_normals",
        "root_tangents",
        "root_bitangents",
        "viewmats",
        "ks",
        "mesh",
        "width",
        "height",
        "observed",
        "device",
    )
    source = inspect.getsource(module.main)
    assert 'if args.axis_field_mode == "trusted-view-cluster"' in source
    assert "refine_fixed_axis_multiview_ratio(" in source
    assert '"superseded-by-fixed-axis-multiview-ratio"' in source
    assert 'args.axis_field_mode != "trusted-view-cluster"' in source
    assert 'default="trusted-view-cluster"' in source

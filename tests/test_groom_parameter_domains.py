import torch

from anigroom.grooming import (
    GroomParameterField,
    decode_positive_asinh,
    decode_positive_asinh_ratio,
    encode_positive_asinh,
    encode_positive_asinh_ratio,
    strand_segment_budgets,
)


def test_positive_asinh_ratio_is_zero_centered_unbounded_and_round_trips() -> None:
    reference = torch.tensor([[0.012], [0.035], [0.080]], dtype=torch.float32)
    raw = torch.tensor([[-100.0], [0.0], [100.0]], dtype=torch.float32)

    decoded = decode_positive_asinh_ratio(raw, reference)

    assert bool((decoded > 0.0).all())
    torch.testing.assert_close(decoded[1], reference[1])
    assert float(decoded[0]) < 0.01 * float(reference[0])
    assert float(decoded[2]) > 100.0 * float(reference[2])
    torch.testing.assert_close(
        encode_positive_asinh_ratio(decoded, reference),
        raw,
        atol=2.0e-4,
        rtol=2.0e-5,
    )


def test_opacity_uses_full_semantic_unit_interval_without_padding() -> None:
    field = GroomParameterField(2, init_length=0.02)
    with torch.no_grad():
        field.opacity_raw.copy_(torch.tensor([[-14.0], [14.0]]))
        field.tip_opacity_ratio_raw.copy_(torch.tensor([[-14.0], [14.0]]))

    decoded = field.decode()

    assert 0.0 < float(decoded.root_opacity[0]) < 1.0e-6
    assert 1.0 - 1.0e-6 < float(decoded.root_opacity[1]) < 1.0
    tip_ratio = decoded.tip_opacity / decoded.root_opacity
    assert 0.0 < float(tip_ratio[0]) < 1.0e-6
    assert 1.0 - 1.0e-6 < float(tip_ratio[1]) < 1.0


def test_brush_stiffness_uses_the_semantic_unit_interval() -> None:
    field = GroomParameterField(3, init_length=0.02)
    with torch.no_grad():
        field.brush_stiffness_raw.copy_(
            torch.tensor([[-14.0], [0.0], [14.0]])
        )

    strength = field.decode().brush_stiffness
    assert 0.0 < float(strength[0]) < 1.0e-6
    torch.testing.assert_close(strength[1], torch.tensor([0.5]))
    assert 1.0 - 1.0e-6 < float(strength[2]) < 1.0


def test_width_profile_uses_semantic_tip_ratio_and_unbounded_positive_taper() -> None:
    field = GroomParameterField(3, init_length=0.02)
    with torch.no_grad():
        field.tip_width_ratio_raw.copy_(torch.tensor([[-14.0], [0.0], [14.0]]))
        field.width_taper_raw.copy_(torch.tensor([[-100.0], [0.0], [100.0]]))

    decoded = field.decode()
    tip_ratio = decoded.tip_width / decoded.root_width

    assert 0.0 < float(tip_ratio[0]) < 1.0e-6
    torch.testing.assert_close(tip_ratio[1], torch.tensor([0.5]))
    assert 1.0 - 1.0e-6 < float(tip_ratio[2]) < 1.0
    assert 0.0 < float(decoded.width_taper[0]) < 0.01
    torch.testing.assert_close(decoded.width_taper[1], torch.tensor([1.0]))
    assert float(decoded.width_taper[2]) > 100.0
    torch.testing.assert_close(
        encode_positive_asinh(decoded.width_taper),
        field.width_taper_raw,
        atol=2.0e-4,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        decode_positive_asinh(encode_positive_asinh(decoded.width_taper)),
        decoded.width_taper,
    )


def test_root_width_is_positive_unbounded_and_reference_centered() -> None:
    field = GroomParameterField(3, init_length=0.02)
    with torch.no_grad():
        field.root_width_reference.copy_(
            torch.tensor([[0.00012], [0.00065], [0.00110]])
        )
        field.root_width_raw.copy_(torch.tensor([[-100.0], [0.0], [100.0]]))

    decoded = field.decode().root_width

    assert bool((decoded > 0.0).all())
    torch.testing.assert_close(decoded[1], field.root_width_reference[1])
    assert float(decoded[0]) < 0.01 * float(field.root_width_reference[0])
    assert float(decoded[2]) > 100.0 * float(field.root_width_reference[2])
    torch.testing.assert_close(
        encode_positive_asinh_ratio(decoded, field.root_width_reference),
        field.root_width_raw,
        atol=2.0e-4,
        rtol=2.0e-5,
    )


def test_child_radius_is_positive_unbounded_and_reference_centered() -> None:
    field = GroomParameterField(3, init_length=0.02)
    with torch.no_grad():
        field.child_radius_reference.copy_(
            torch.tensor([[0.0008], [0.0028], [0.0060]])
        )
        field.child_radius_raw.copy_(torch.tensor([[-100.0], [0.0], [100.0]]))

    decoded = field.decode().child_radius

    assert bool((decoded > 0.0).all())
    torch.testing.assert_close(decoded[1], field.child_radius_reference[1])
    assert float(decoded[0]) < 0.01 * float(field.child_radius_reference[0])
    assert float(decoded[2]) > 100.0 * float(field.child_radius_reference[2])
    torch.testing.assert_close(
        encode_positive_asinh_ratio(decoded, field.child_radius_reference),
        field.child_radius_raw,
        atol=2.0e-4,
        rtol=2.0e-5,
    )


def test_segment_budget_tracks_fixed_length_and_curvature_resolution() -> None:
    straight_short = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    straight_long = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    straight_very_long = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    half_turn = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [0.7071, 0.7071, 0.0],
                [0.0, 1.0, 0.0],
                [-0.7071, 0.7071, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    short_budget, _ = strand_segment_budgets(
        straight_short,
        lengths=torch.tensor([[1.0]]),
        min_segments=4,
        length_origin=1.0,
        segments_per_unit_length=4.0,
        segments_per_unit_complexity=10.0,
    )
    long_budget, _ = strand_segment_budgets(
        straight_long,
        lengths=torch.tensor([[2.0]]),
        min_segments=4,
        length_origin=1.0,
        segments_per_unit_length=4.0,
        segments_per_unit_complexity=10.0,
    )
    very_long_budget, _ = strand_segment_budgets(
        straight_very_long,
        lengths=torch.tensor([[10.0]]),
        min_segments=4,
        length_origin=1.0,
        segments_per_unit_length=4.0,
        segments_per_unit_complexity=10.0,
    )
    curved_budget, _ = strand_segment_budgets(
        half_turn,
        lengths=torch.tensor([[2.0]]),
        min_segments=4,
        length_origin=1.0,
        segments_per_unit_length=4.0,
        segments_per_unit_complexity=10.0,
    )
    collapsed_budget, _ = strand_segment_budgets(
        torch.zeros((1, 5, 3), dtype=torch.float32),
        lengths=torch.tensor([[0.0]]),
        min_segments=4,
        length_origin=1.0,
        segments_per_unit_length=4.0,
        segments_per_unit_complexity=10.0,
    )

    assert int(short_budget) == 4
    assert int(long_budget) == 8
    assert int(very_long_budget) == 40
    assert int(curved_budget) > int(long_budget)
    assert int(collapsed_budget) == 4

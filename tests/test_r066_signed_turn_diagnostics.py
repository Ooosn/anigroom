import numpy as np
import pytest

from tools.diagnose_strand_foldback_components import validate_attribute_shapes


def test_signed_turn_diagnostic_keeps_render_and_guide_shapes_separate() -> None:
    render_attributes = {
        "curl_turns": np.asarray([0.0, -1.0, 1.0]),
        "curl_wavenumber": np.asarray([0.0, 0.2, 0.3]),
    }
    guide_attributes = {
        "curl_turns": np.asarray([-2.0, 0.0]),
        "curl_wavenumber_magnitude": np.asarray([0.4, 0.0]),
    }

    validate_attribute_shapes(
        render_attributes,
        guide_attributes,
        render_count=3,
        guide_count=2,
        focus_mask=np.asarray([False, True, False]),
    )

    with pytest.raises(RuntimeError, match="render attribute shape mismatch"):
        validate_attribute_shapes(
            {**render_attributes, "guide_curl_turns": np.zeros(2)},
            guide_attributes,
            render_count=3,
            guide_count=2,
            focus_mask=np.asarray([False, True, False]),
        )


"""The UNDER confidence score must treat a projection above the line as a weaker UNDER."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictions"))
analyze_v3 = pytest.importorskip("analyze_v3")

PLAYER = {"l10_hit_rate": 50, "season_hit_rate": 50, "l10_miss_count": 5}


def test_projection_above_line_scores_lower_than_below():
    below = analyze_v3.under_confidence_score(PLAYER, "pts", -4.0)
    above = analyze_v3.under_confidence_score(PLAYER, "pts", 4.0)
    assert above < below


def test_call_site_passes_signed_gap():
    # analyze_player flips `gap` negative for display on UNDER calls; the score
    # must use the signed projection-minus-line value instead.
    src = open(analyze_v3.__file__).read()
    call = src[src.index("under_conf_score = under_confidence_score("):][:200]
    assert "adjusted_proj - line" in call

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictions"))

from box_score_match import find_actual, norm_name  # noqa: E402
from nba_fetcher import _parse_minutes  # noqa: E402


def _row(team, pts, mins=30.0, played=True):
    return {"team": team, "pts": pts, "min": mins, "played": played}


ACTUALS = {
    "Jaylin Williams": _row("OKC", 4),
    "Nikola Jokić": _row("DEN", 31),
    "Jaren Jackson Jr.": _row("MEM", 22),
    "Benched Guy": _row("BOS", 0, mins=0.0, played=False),
}


def test_exact_and_accent_match():
    assert find_actual("Nikola Jokic", "pts", ACTUALS) == 31
    assert find_actual("Jaren Jackson", "PTS", ACTUALS, team="MEM") == 22


def test_missing_player_is_not_graded_off_a_similar_name():
    # Jalen Williams isn't in the box score; Jaylin Williams must not stand in.
    assert find_actual("Jalen Williams", "pts", ACTUALS, team="OKC") is None


def test_single_fuzzy_match_on_other_team_is_rejected():
    assert find_actual("Jalen Williams", "pts", ACTUALS, team="PHI") is None


def test_dnp_player_is_void_not_zero():
    # An UNDER on a scratched player must not grade as a hit.
    assert find_actual("Benched Guy", "pts", ACTUALS) is None


def test_norm_name_strips_suffix_and_punctuation():
    assert norm_name("Jaren Jackson Jr.") == "jaren jackson"
    assert norm_name("De'Aaron Fox") == "deaaron fox"


def test_parse_minutes_formats():
    assert round(_parse_minutes("PT34M12.00S"), 1) == 34.2
    assert round(_parse_minutes("34:12"), 1) == 34.2
    assert _parse_minutes("") == 0.0
    assert _parse_minutes("garbage") is None

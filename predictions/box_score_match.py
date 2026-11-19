"""Match a prediction's player to a box-score row, safely.

Shared by grade_results.py, grade_primary_parlays.py, grade_shadow_parlays.py
and analyze_v3.grade_predictions so every grader voids the same legs.

Two rules keep grading honest:

1. A player who did not play is never graded. Box scores list inactive and
   DNP players with 0 minutes and all-zero stats; grading those would turn
   every UNDER on a late scratch into a "hit" (books void these legs), and the
   graded files feed model training.
2. A fuzzy name match must be unambiguous. "Jalen Williams" must not be graded
   off "Jaylin Williams" just because the first one is missing from the box
   score. A fuzzy match needs the same last name and a first name that is an
   initial or a prefix of the other ("Cam" / "Cameron"). When that matches
   more than one player and the team can't separate them, or the single match
   is on another team, the leg is left ungraded.
"""

import re
import unicodedata

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name):
    """Lower-case, strip accents, punctuation and Jr./Sr./II-style suffixes."""
    nfkd = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    s = re.sub(r"[.'`’]", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    parts = [p for p in s.split() if p]
    while len(parts) > 2 and parts[-1] in _SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _first_names_compatible(a, b):
    """"J" ~ "Jalen", "Cam" ~ "Cameron", "Nic" ~ "Nicolas"; "Jalen" !~ "Jaylin"."""
    if len(a) == 1 or len(b) == 1:
        return a[0] == b[0]
    return a.startswith(b) or b.startswith(a)


def did_not_play(stats):
    """True when the box score shows the player never took the floor."""
    if stats.get("played") is False:
        return True
    mins = stats.get("min")
    return mins is not None and mins <= 0


def find_player_stats(player_name, actuals, team=None):
    """Return the box-score stats dict for a player, or None if not matched."""
    p_norm = norm_name(player_name)
    if not p_norm:
        return None
    p_parts = p_norm.split()
    team_u = (team or "").upper()

    exact, fuzzy = [], []
    for name, stats in actuals.items():
        n_norm = norm_name(name)
        if n_norm == p_norm:
            exact.append(stats)
            continue
        n_parts = n_norm.split()
        if (len(p_parts) >= 2 and len(n_parts) >= 2
                and p_parts[-1] == n_parts[-1]
                and _first_names_compatible(p_parts[0], n_parts[0])):
            fuzzy.append(stats)

    pool = exact or fuzzy
    if not pool:
        return None
    if team_u:
        on_team = [s for s in pool if str(s.get("team", "")).upper() == team_u]
        if on_team:
            pool = on_team
        elif not exact:
            # Single fuzzy hit on another team is a different player.
            return None
    if len(pool) > 1:
        return None  # ambiguous: don't guess which player it was
    return pool[0]


def find_actual(player_name, stat, actuals, team=None):
    """Actual value of ``stat`` for a player, or None (not found / DNP)."""
    stats = find_player_stats(player_name, actuals, team)
    if stats is None or did_not_play(stats):
        return None
    val = stats.get(stat)
    if val is None and isinstance(stat, str):
        val = stats.get(stat.lower())
    return val

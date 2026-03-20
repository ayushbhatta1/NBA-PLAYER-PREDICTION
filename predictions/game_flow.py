#!/usr/bin/env python3
"""
Game Flow Predictor v1 — model the GAME first, derive player impact from context.

Instead of predicting "will LeBron score 25?" in isolation, this module asks:
  1. What kind of game will this be? (blowout, close, OT?)
  2. How does game script affect minutes?
  3. How do minutes + pace + script affect each stat projection?

Built on 2,973 historical NBA games (SGO data, 2024-2026) with real spreads,
game totals, and final scores. Distributions are empirical, not assumed.

Pipeline integration via enrich_with_game_flow(results, GAMES).
CLI: python3 game_flow.py --build | --matchup HOME AWAY SPREAD TOTAL

Zero API calls. All data from cached historical_events.json + venue_data.py.
"""

import json
import math
import os
import sys
from pathlib import Path

# ── PATHS ──
SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR / "cache"
HISTORICAL_PATH = CACHE_DIR / "sgo" / "historical_events.json"
DISTRIBUTIONS_PATH = CACHE_DIR / "game_flow_distributions.json"

# ── CONSTANTS ──

# Spread buckets (abs_spread ranges)
SPREAD_BUCKETS = [
    ("pick",    0,  3),   # pick'em / coin-flip
    ("lean",    3,  6),   # slight favorite
    ("solid",   6, 10),   # clear favorite
    ("heavy",  10, 15),   # heavy favorite
    ("crush",  15, 50),   # blowout-expected
]

# NBA reference pace (league average ~100 possessions per 48 min)
LEAGUE_AVG_PACE = 100.0

# Minutes adjustments by game script
# Derived from NBA rotation research:
#   - Stars average ~34-36 min in normal games
#   - Blowout wins: starters sit by mid-Q4, lose ~5-8 min
#   - Blowout losses: starters pulled late Q3 or early Q4, lose ~4-7 min
#   - Close games: starters play 38-40 min, gain ~2-5 min
#   - OT: +5 min per OT period for starters
STAR_MINS_NORMAL = 35.0
ROLE_MINS_NORMAL = 26.0

# Stat sensitivity to pace (per 1% pace change above/below league avg)
# Volume stats scale with possessions; efficiency stats less so
PACE_SENSITIVITY = {
    "pts":     0.008,   # 0.8% per 1% pace delta
    "reb":     0.005,   # less pace-sensitive (time on court matters more)
    "ast":     0.007,
    "3pm":     0.006,
    "stl":     0.003,   # opportunity-based, less volume-dependent
    "blk":     0.002,
    "pra":     0.007,
    "pr":      0.007,
    "pa":      0.008,
    "ra":      0.006,
    "stl_blk": 0.003,
}

# Stats that inflate in garbage time for losing team's secondary scorers
GARBAGE_TIME_INFLATE_STATS = {"pts", "reb", "ast", "pra", "pr", "pa", "ra", "3pm"}


# ═════════════════════════════════════════════════════════════════
# 1. BUILD HISTORICAL DISTRIBUTIONS
# ═════════════════════════════════════════════════════════════════

def _load_historical_events():
    """Load the 3,188 historical games from SGO cache."""
    if not HISTORICAL_PATH.exists():
        print(f"[game_flow] WARNING: {HISTORICAL_PATH} not found")
        return []
    data = json.loads(HISTORICAL_PATH.read_text())
    events = data.get("events", [])
    # Filter to complete records (spread + scores + game_total)
    return [
        e for e in events
        if e.get("spread") is not None
        and e.get("home_score") is not None
        and e.get("away_score") is not None
        and e.get("game_total") is not None
    ]


def _classify_spread(abs_spread):
    """Return bucket name for a given abs(spread)."""
    for name, lo, hi in SPREAD_BUCKETS:
        if lo <= abs_spread < hi:
            return name
    return "crush"


def build_distributions(save=True):
    """
    Build empirical game flow distributions from historical events.
    For each spread bucket, compute:
      - blowout probability (15+ and 20+ margin)
      - close game probability (<=5 and <=3 margin)
      - overtime proxy rate
      - margin percentiles (p10, p25, p50, p75, p90)
      - average actual total
      - favorite cover rate
      - favorite blowout rate
      - underdog outright win rate
    """
    events = _load_historical_events()
    if not events:
        print("[game_flow] No historical events found. Cannot build distributions.")
        return {}

    distributions = {
        "meta": {
            "total_games": len(events),
            "source": "SGO historical_events.json",
            "spread_convention": "home_spread, negative=home_favored",
        },
        "buckets": {},
    }

    for bname, blo, bhi in SPREAD_BUCKETS:
        games = [e for e in events if blo <= abs(e["spread"]) < bhi]
        if not games:
            continue

        n = len(games)
        margins = [abs(e["home_score"] - e["away_score"]) for e in games]
        totals = [e["home_score"] + e["away_score"] for e in games]
        sorted_margins = sorted(margins)

        # Blowout rates
        blowout_15 = sum(1 for m in margins if m >= 15) / n
        blowout_20 = sum(1 for m in margins if m >= 20) / n

        # Close game rates
        close_5 = sum(1 for m in margins if m <= 5) / n
        close_3 = sum(1 for m in margins if m <= 3) / n

        # OT proxy: margin <= 5 AND actual total > line + 10 (extra scoring)
        ot_proxy = sum(
            1 for e in games
            if abs(e["home_score"] - e["away_score"]) <= 5
            and (e["home_score"] + e["away_score"]) > e["game_total"] + 10
        ) / n

        # Margin percentiles
        def pctile(arr, p):
            idx = max(0, min(int(len(arr) * p), len(arr) - 1))
            return arr[idx]

        # Favorite analysis
        # spread is home_spread: negative = home favored
        # expected_home_margin = -spread
        fav_cover = 0
        fav_blowout = 0
        dog_wins = 0
        for e in games:
            home_margin = e["home_score"] - e["away_score"]
            expected_margin = -e["spread"]
            # Favorite's actual margin (positive = fav winning)
            if expected_margin > 0:
                fav_margin = home_margin
            else:
                fav_margin = -home_margin
            fav_spread = abs(e["spread"])
            if fav_margin > fav_spread:
                fav_cover += 1
            if fav_margin >= 15:
                fav_blowout += 1
            if fav_margin < 0:
                dog_wins += 1

        # Total vs line
        overs = sum(1 for e in games if (e["home_score"] + e["away_score"]) > e["game_total"]) / n
        avg_total_diff = sum(
            (e["home_score"] + e["away_score"]) - e["game_total"] for e in games
        ) / n

        distributions["buckets"][bname] = {
            "spread_range": [blo, bhi],
            "games": n,
            "avg_margin": round(sum(margins) / n, 1),
            "median_margin": round(sorted_margins[n // 2], 1),
            "margin_p10": pctile(sorted_margins, 0.10),
            "margin_p25": pctile(sorted_margins, 0.25),
            "margin_p50": pctile(sorted_margins, 0.50),
            "margin_p75": pctile(sorted_margins, 0.75),
            "margin_p90": pctile(sorted_margins, 0.90),
            "blowout_15_pct": round(blowout_15, 4),
            "blowout_20_pct": round(blowout_20, 4),
            "close_5_pct": round(close_5, 4),
            "close_3_pct": round(close_3, 4),
            "ot_proxy_pct": round(ot_proxy, 4),
            "avg_total": round(sum(totals) / n, 1),
            "over_pct": round(overs, 4),
            "avg_total_diff": round(avg_total_diff, 1),
            "fav_cover_pct": round(fav_cover / n, 4),
            "fav_blowout_pct": round(fav_blowout / n, 4),
            "dog_wins_pct": round(dog_wins / n, 4),
        }

    if save:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        DISTRIBUTIONS_PATH.write_text(json.dumps(distributions, indent=2))
        print(f"[game_flow] Saved distributions to {DISTRIBUTIONS_PATH}")
        print(f"[game_flow] {len(events)} games across {len(distributions['buckets'])} buckets")

    return distributions


def _load_distributions():
    """Load cached distributions, or build if missing."""
    if DISTRIBUTIONS_PATH.exists():
        return json.loads(DISTRIBUTIONS_PATH.read_text())
    return build_distributions(save=True)


# ═════════════════════════════════════════════════════════════════
# 2. PREDICT GAME FLOW
# ═════════════════════════════════════════════════════════════════

def predict_game_flow(home_team, away_team, spread, total, GAMES=None):
    """
    Predict game flow for a matchup.

    Args:
        home_team: Home team abbreviation (e.g., "LAL")
        away_team: Away team abbreviation (e.g., "BOS")
        spread: Home spread (negative = home favored). E.g., -5.5 means home favored by 5.5
        total: Game total (over/under line). E.g., 220.5
        GAMES: Optional GAMES dict for pace/defense data

    Returns dict:
        predicted_margin, blowout_prob, close_game_prob, overtime_prob,
        pace_estimate, pace_delta_pct, star_minutes_impact, flow_label,
        fav_team, dog_team, fav_spread
    """
    dists = _load_distributions()
    buckets = dists.get("buckets", {})

    abs_spread = abs(spread) if spread else 0
    bucket_name = _classify_spread(abs_spread)
    bucket = buckets.get(bucket_name, {})

    # Who is favored?
    if spread and spread < 0:
        fav_team, dog_team = home_team, away_team
    elif spread and spread > 0:
        fav_team, dog_team = away_team, home_team
    else:
        fav_team, dog_team = home_team, away_team  # pick'em: slight home edge

    # ── Predicted margin (from spread, which IS the market's best estimate) ──
    predicted_margin = -spread if spread else 0  # positive = home wins by X

    # ── Blowout probability (empirical from bucket) ──
    blowout_prob = bucket.get("blowout_15_pct", 0.05)

    # Interpolate between buckets for smoother estimates
    # E.g., spread of 9.5 is near the solid/heavy boundary
    blowout_prob = _interpolate_metric(abs_spread, buckets, "blowout_15_pct")
    blowout_20_prob = _interpolate_metric(abs_spread, buckets, "blowout_20_pct")

    # ── Close game probability ──
    close_game_prob = _interpolate_metric(abs_spread, buckets, "close_5_pct")
    close_3_prob = _interpolate_metric(abs_spread, buckets, "close_3_pct")

    # ── Overtime probability ──
    overtime_prob = _interpolate_metric(abs_spread, buckets, "ot_proxy_pct")

    # ── Pace estimate ──
    pace_estimate = LEAGUE_AVG_PACE
    pace_label = "average"
    if GAMES:
        game_key = f"{away_team}@{home_team}"
        gctx = GAMES.get(game_key, {})
        pace_data = gctx.get("pace", {})
        if pace_data.get("projected"):
            pace_estimate = pace_data["projected"]
            pace_label = pace_data.get("label", "average")

    pace_delta_pct = (pace_estimate - LEAGUE_AVG_PACE) / LEAGUE_AVG_PACE * 100

    # ── Star minutes impact ──
    # In blowouts: starters lose minutes. In close/OT: starters gain.
    # Weighted by probability of each scenario.
    blowout_mins_loss = -6.0   # avg minutes lost in a blowout
    close_mins_gain = +3.0     # avg minutes gained in close game
    ot_mins_gain = +5.0        # avg minutes gained in OT

    star_minutes_impact = (
        blowout_prob * blowout_mins_loss
        + close_game_prob * close_mins_gain
        + overtime_prob * ot_mins_gain
        + (1 - blowout_prob - close_game_prob - overtime_prob) * 0  # normal game
    )

    # ── Flow label ──
    if blowout_prob >= 0.35:
        flow_label = "BLOWOUT_EXPECTED"
    elif blowout_prob >= 0.15:
        flow_label = "BLOWOUT_POSSIBLE"
    elif close_game_prob >= 0.50:
        flow_label = "CLOSE_GAME"
    elif overtime_prob >= 0.04:
        flow_label = "OT_RISK"
    else:
        flow_label = "NORMAL"

    # ── Underdog win probability ──
    dog_wins_pct = _interpolate_metric(abs_spread, buckets, "dog_wins_pct")

    return {
        "home_team": home_team,
        "away_team": away_team,
        "spread": spread,
        "total": total,
        "predicted_margin": round(predicted_margin, 1),
        "fav_team": fav_team,
        "dog_team": dog_team,
        "fav_spread": round(abs_spread, 1),
        "bucket": bucket_name,
        "blowout_prob": round(blowout_prob, 4),
        "blowout_20_prob": round(blowout_20_prob, 4),
        "close_game_prob": round(close_game_prob, 4),
        "close_3_prob": round(close_3_prob, 4),
        "overtime_prob": round(overtime_prob, 4),
        "dog_wins_prob": round(dog_wins_pct, 4),
        "pace_estimate": round(pace_estimate, 1),
        "pace_delta_pct": round(pace_delta_pct, 1),
        "pace_label": pace_label,
        "star_minutes_impact": round(star_minutes_impact, 1),
        "flow_label": flow_label,
    }


def _interpolate_metric(abs_spread, buckets, metric):
    """
    Interpolate a metric between spread buckets for smooth transitions.
    Uses the bucket midpoints as anchor values, linearly interpolates between.
    """
    # Build sorted anchor points: (midpoint_spread, metric_value)
    anchors = []
    for bname, blo, bhi in SPREAD_BUCKETS:
        b = buckets.get(bname, {})
        if metric in b:
            # Use midpoint of bucket as anchor (cap high bucket at 20)
            mid = (blo + min(bhi, 25)) / 2
            anchors.append((mid, b[metric]))

    if not anchors:
        return 0.0

    # Clamp to range
    if abs_spread <= anchors[0][0]:
        return anchors[0][1]
    if abs_spread >= anchors[-1][0]:
        return anchors[-1][1]

    # Linear interpolation between nearest anchors
    for i in range(len(anchors) - 1):
        x0, y0 = anchors[i]
        x1, y1 = anchors[i + 1]
        if x0 <= abs_spread <= x1:
            t = (abs_spread - x0) / (x1 - x0) if x1 != x0 else 0
            return y0 + t * (y1 - y0)

    return anchors[-1][1]


# ═════════════════════════════════════════════════════════════════
# 3. ESTIMATE PLAYER MINUTES
# ═════════════════════════════════════════════════════════════════

def estimate_player_minutes(player_data, game_flow, GAMES=None):
    """
    Estimate a player's minutes given the game flow prediction.

    Args:
        player_data: Dict with season_mins_avg, l5_mins_avg, mins_30plus_pct, team (abbr)
        game_flow: Output of predict_game_flow()
        GAMES: Optional GAMES dict

    Returns dict:
        expected_minutes, minutes_upside, minutes_downside,
        blowout_risk_pct, minutes_delta
    """
    season_avg = player_data.get("season_mins_avg", 28.0)
    l5_avg = player_data.get("l5_mins_avg", season_avg)
    mins_30plus_pct = player_data.get("mins_30plus_pct", 50)
    team = player_data.get("team", "")

    # Base minutes: blend of season and recent
    base_minutes = 0.6 * season_avg + 0.4 * l5_avg

    # Classify player role by minutes
    if base_minutes >= 32:
        role = "star"
    elif base_minutes >= 25:
        role = "starter"
    elif base_minutes >= 16:
        role = "rotation"
    else:
        role = "bench"

    # ── Is this player on the favored or underdog team? ──
    is_on_fav_team = (team == game_flow.get("fav_team"))
    is_on_dog_team = (team == game_flow.get("dog_team"))

    blowout_prob = game_flow.get("blowout_prob", 0)
    close_prob = game_flow.get("close_game_prob", 0)
    ot_prob = game_flow.get("overtime_prob", 0)

    # ── Minutes adjustments by scenario ──
    minutes_delta = 0.0

    if role in ("star", "starter"):
        # Blowout scenario: stars/starters lose minutes
        if is_on_fav_team:
            # Favored team blowout win: starters benched Q4
            blowout_mins_loss = -7.0 if role == "star" else -5.0
        else:
            # Underdog blowout loss: coach may pull starters late Q3/Q4
            blowout_mins_loss = -5.0 if role == "star" else -4.0

        # Close game: starters play deep, stars may see 38-40 min
        close_mins_gain = +4.0 if role == "star" else +3.0

        # OT: extra 5 min per period for starters
        ot_mins_gain = +5.0

        # Weighted expected delta
        minutes_delta = (
            blowout_prob * blowout_mins_loss
            + close_prob * close_mins_gain
            + ot_prob * ot_mins_gain
        )
    elif role == "rotation":
        # Rotation players: inverse effect in blowouts (they GET more minutes)
        if is_on_fav_team:
            # Bench mob cleanup: rotation guys get Q4 run
            blowout_mins_gain = +4.0
        else:
            blowout_mins_gain = +2.0  # less certain in losses
        close_mins_loss = -2.0  # tighter rotation in close games
        minutes_delta = (
            blowout_prob * blowout_mins_gain
            + close_prob * close_mins_loss
        )
    else:
        # Bench: significant upside in blowouts only
        minutes_delta = blowout_prob * 5.0

    expected_minutes = base_minutes + minutes_delta

    # ── Upside / downside scenarios ──
    # Upside: close game or OT
    minutes_upside = base_minutes + (close_prob + ot_prob) * (5.0 if role in ("star", "starter") else 2.0)
    # Downside: blowout
    minutes_downside = base_minutes + blowout_prob * (-8.0 if role == "star" else -5.0)

    # Blowout risk pct: how likely minutes get cut
    blowout_risk_pct = blowout_prob * 100 if role in ("star", "starter") else 0

    return {
        "role": role,
        "base_minutes": round(base_minutes, 1),
        "expected_minutes": round(expected_minutes, 1),
        "minutes_delta": round(minutes_delta, 1),
        "minutes_upside": round(minutes_upside, 1),
        "minutes_downside": round(minutes_downside, 1),
        "blowout_risk_pct": round(blowout_risk_pct, 1),
        "is_on_fav_team": is_on_fav_team,
    }


# ═════════════════════════════════════════════════════════════════
# 4. ADJUST PROJECTION FOR GAME FLOW
# ═════════════════════════════════════════════════════════════════

def adjust_projection_for_flow(projection, stat, game_flow, player_minutes, player_data=None):
    """
    Adjust a raw stat projection based on game flow context.

    Args:
        projection: Raw stat projection (e.g., 25.3 points)
        stat: Stat type (pts, reb, ast, etc.)
        game_flow: Output of predict_game_flow()
        player_minutes: Output of estimate_player_minutes()
        player_data: Optional dict with additional player context

    Returns:
        (adjusted_projection, flow_adjustment, confidence, breakdown)
    """
    if not projection or projection <= 0:
        return projection, 0, 0.5, {}

    adj = 0.0
    breakdown = {}

    role = player_minutes.get("role", "rotation")
    is_fav = player_minutes.get("is_on_fav_team", False)
    blowout_prob = game_flow.get("blowout_prob", 0)
    close_prob = game_flow.get("close_game_prob", 0)
    ot_prob = game_flow.get("overtime_prob", 0)
    pace_delta = game_flow.get("pace_delta_pct", 0)

    # ── (A) Minutes-based adjustment ──
    # If expected minutes shift, scale projection proportionally
    base_mins = player_minutes.get("base_minutes", 30)
    expected_mins = player_minutes.get("expected_minutes", base_mins)
    if base_mins > 0:
        mins_ratio = expected_mins / base_mins
        mins_adj = projection * (mins_ratio - 1.0)
        # Cap at reasonable bounds
        mins_adj = max(-projection * 0.20, min(projection * 0.15, mins_adj))
        adj += mins_adj
        breakdown["minutes_adj"] = round(mins_adj, 2)

    # ── (B) Pace adjustment ──
    # Higher pace = more possessions = more volume stat opportunities
    stat_lower = stat.lower() if stat else ""
    pace_sens = PACE_SENSITIVITY.get(stat_lower, 0.005)
    pace_adj = projection * pace_sens * pace_delta
    # Cap pace adjustment at +/-5% of projection
    pace_adj = max(-projection * 0.05, min(projection * 0.05, pace_adj))
    adj += pace_adj
    breakdown["pace_adj"] = round(pace_adj, 2)

    # ── (C) Blowout script adjustment ──
    # Beyond just minutes: game script changes HOW stats accumulate
    blowout_script_adj = 0.0
    if blowout_prob >= 0.10 and role in ("star", "starter"):
        if is_fav:
            # Favored star in likely blowout WIN:
            # - Reduced minutes (already in A)
            # - Lower urgency to score (less ISO, more ball movement)
            # - May coast in Q3 if lead is big
            # Net: slight additional reduction beyond minutes
            blowout_script_adj = -projection * 0.02 * (blowout_prob / 0.35)
            if stat_lower in ("pts", "pra", "pr", "pa"):
                blowout_script_adj *= 1.5  # scoring most affected
        else:
            # Underdog star in likely blowout LOSS:
            # - Garbage time can inflate PTS (keep shooting)
            # - REB less affected (game flow doesn't change rebounding much)
            # - AST may decrease (less structured offense in garbage time)
            if stat_lower in GARBAGE_TIME_INFLATE_STATS:
                # Garbage time partially offsets minutes loss for scoring stats
                blowout_script_adj = projection * 0.01 * (blowout_prob / 0.35)
            if stat_lower in ("ast", "pa", "ra"):
                # Assists decrease: less ball movement, ISO scoring
                blowout_script_adj = -projection * 0.015 * (blowout_prob / 0.35)

    adj += blowout_script_adj
    breakdown["blowout_script_adj"] = round(blowout_script_adj, 2)

    # ── (D) Close game adjustment ──
    # Close games favor stars: more ISO possessions, higher usage
    close_adj = 0.0
    if close_prob >= 0.30 and role == "star":
        if stat_lower in ("pts", "pra", "pr", "pa"):
            # Stars get more touches in crunch time
            close_adj = projection * 0.02 * (close_prob / 0.70)
        elif stat_lower in ("ast",):
            # More ISO = fewer assists
            close_adj = -projection * 0.01 * (close_prob / 0.70)
    adj += close_adj
    breakdown["close_game_adj"] = round(close_adj, 2)

    # ── (E) OT bonus ──
    ot_adj = 0.0
    if ot_prob >= 0.02 and role in ("star", "starter"):
        # OT adds ~5 min = ~15% more minutes for starters
        # But it's probability-weighted
        ot_adj = projection * 0.12 * ot_prob
        if stat_lower in ("pts", "pra"):
            ot_adj *= 1.2  # scoring inflates most in OT
    adj += ot_adj
    breakdown["ot_adj"] = round(ot_adj, 2)

    # ── Total flow adjustment ──
    # Cap total adjustment at +/-15% of projection
    adj = max(-projection * 0.15, min(projection * 0.15, adj))
    adjusted_projection = projection + adj

    # ── Confidence ──
    # Higher confidence when game flow is more predictable
    # Blowout-expected games are more predictable than close games
    if game_flow.get("flow_label") == "BLOWOUT_EXPECTED":
        confidence = 0.75  # high certainty game will be lopsided
    elif game_flow.get("flow_label") == "CLOSE_GAME":
        confidence = 0.55  # close games are inherently uncertain
    elif game_flow.get("flow_label") == "OT_RISK":
        confidence = 0.50
    else:
        confidence = 0.60

    return round(adjusted_projection, 1), round(adj, 2), round(confidence, 2), breakdown


# ═════════════════════════════════════════════════════════════════
# 5. PIPELINE INTEGRATION
# ═════════════════════════════════════════════════════════════════

def enrich_with_game_flow(results, GAMES):
    """
    Enrich a list of pick dicts with game flow data.
    Called in the pipeline after analyze_v3 and before parlay scoring.

    For each prop:
      - Computes game flow prediction for the matchup
      - Estimates player minutes impact
      - Adjusts the projection
      - Adds fields: flow_adj, blowout_risk, minutes_impact, pace_impact,
                     flow_label, flow_confidence

    Args:
        results: List of pick dicts from analyze_v3 (must have game, team, spread, etc.)
        GAMES: GAMES dict from game_researcher

    Returns:
        results (mutated in place, also returned)
    """
    if not GAMES:
        for r in results:
            r.setdefault("flow_adj", 0)
            r.setdefault("flow_label", "NO_DATA")
            r.setdefault("blowout_risk", 0)
            r.setdefault("close_game_pct", 0)
            r.setdefault("minutes_impact", 0)
            r.setdefault("pace_impact", 0)
        return 0

    # Cache game flow predictions per game key (no need to recompute per prop)
    flow_cache = {}

    enriched = 0
    for r in results:
        game_key = r.get("game", "")
        team = r.get("team", "")
        stat = r.get("stat", "")
        projection = r.get("projection", 0)

        if not game_key or game_key not in GAMES:
            r["flow_adj"] = 0
            r["flow_label"] = "UNKNOWN"
            continue

        gctx = GAMES[game_key]

        # ── Get or compute game flow ──
        if game_key not in flow_cache:
            home_team = gctx.get("home_abr", "")
            away_team = gctx.get("away_abr", "")
            spread = gctx.get("spread", 0)
            total = gctx.get("over_under", 220)
            flow_cache[game_key] = predict_game_flow(
                home_team, away_team, spread, total, GAMES
            )
        game_flow = flow_cache[game_key]

        # ── Estimate player minutes ──
        player_data_for_mins = {
            "season_mins_avg": r.get("season_mins_avg", 28),
            "l5_mins_avg": r.get("l5_mins_avg", r.get("season_mins_avg", 28)),
            "mins_30plus_pct": r.get("mins_30plus_pct", 50),
            "team": team,
        }
        player_mins = estimate_player_minutes(player_data_for_mins, game_flow, GAMES)

        # ── Adjust projection ──
        adj_proj, flow_adj, confidence, breakdown = adjust_projection_for_flow(
            projection, stat, game_flow, player_mins
        )

        # ── Write enrichment fields ──
        r["flow_adj"] = flow_adj
        r["flow_adjusted_projection"] = adj_proj
        r["flow_label"] = game_flow.get("flow_label", "NORMAL")
        r["flow_confidence"] = confidence
        r["blowout_risk"] = round(game_flow.get("blowout_prob", 0) * 100, 1)
        r["close_game_pct"] = round(game_flow.get("close_game_prob", 0) * 100, 1)
        r["minutes_impact"] = player_mins.get("minutes_delta", 0)
        r["pace_impact"] = breakdown.get("pace_adj", 0)
        r["player_role"] = player_mins.get("role", "rotation")
        r["expected_minutes"] = player_mins.get("expected_minutes", 0)

        # ── Recalculate direction if flow shifts projection across the line ──
        line = r.get("line", 0)
        if line and adj_proj:
            new_gap = adj_proj - line
            old_gap = projection - line if projection else 0
            # If flow adjustment flipped the gap sign, note it
            if old_gap * new_gap < 0:
                r["flow_direction_flip"] = True
                r["flow_flip_note"] = (
                    f"Flow shifted projection from {projection:.1f} to {adj_proj:.1f} "
                    f"(line {line}) - direction flipped"
                )

        enriched += 1

    if enriched > 0:
        print(f"[game_flow] Enriched {enriched} props with game flow data")
        # Summary of flow labels
        labels = {}
        for r in results:
            lbl = r.get("flow_label", "UNKNOWN")
            labels[lbl] = labels.get(lbl, 0) + 1
        for lbl, cnt in sorted(labels.items(), key=lambda x: -x[1]):
            print(f"  {lbl}: {cnt} props")

    return enriched


# ═════════════════════════════════════════════════════════════════
# 6. CLI
# ═════════════════════════════════════════════════════════════════

def _cli_build():
    """Build and save distributions from historical data."""
    dists = build_distributions(save=True)
    if not dists:
        return

    print("\n=== GAME FLOW DISTRIBUTIONS ===")
    for bname, bdata in dists.get("buckets", {}).items():
        print(f"\n  {bname.upper()} (spread {bdata['spread_range'][0]}-{bdata['spread_range'][1]}, {bdata['games']} games):")
        print(f"    Avg margin: {bdata['avg_margin']}")
        print(f"    Blowout (15+): {bdata['blowout_15_pct']*100:.1f}%")
        print(f"    Blowout (20+): {bdata['blowout_20_pct']*100:.1f}%")
        print(f"    Close (<=5):   {bdata['close_5_pct']*100:.1f}%")
        print(f"    Close (<=3):   {bdata['close_3_pct']*100:.1f}%")
        print(f"    OT proxy:      {bdata['ot_proxy_pct']*100:.1f}%")
        print(f"    Avg total:     {bdata['avg_total']}")
        print(f"    Margin p10/p25/p50/p75/p90: {bdata['margin_p10']}/{bdata['margin_p25']}/{bdata['margin_p50']}/{bdata['margin_p75']}/{bdata['margin_p90']}")
        print(f"    Fav covers: {bdata['fav_cover_pct']*100:.1f}% | Fav blowout: {bdata['fav_blowout_pct']*100:.1f}% | Dog wins: {bdata['dog_wins_pct']*100:.1f}%")


def _cli_matchup(args):
    """Predict game flow for a specific matchup."""
    if len(args) < 4:
        print("Usage: python3 game_flow.py --matchup HOME AWAY SPREAD TOTAL")
        print("  Example: python3 game_flow.py --matchup LAL BOS -5.5 220")
        print("  Spread convention: negative = home favored")
        sys.exit(1)

    home = args[0].upper()
    away = args[1].upper()
    spread = float(args[2])
    total = float(args[3])

    flow = predict_game_flow(home, away, spread, total)

    print(f"\n=== GAME FLOW PREDICTION: {away} @ {home} ===")
    print(f"  Spread: {spread:+.1f} (home {'favored' if spread < 0 else 'underdog'} by {abs(spread)})")
    print(f"  Total: {total}")
    print(f"  Bucket: {flow['bucket']}")
    print(f"  Flow: {flow['flow_label']}")
    print()
    print(f"  Predicted margin: {flow['predicted_margin']:+.1f} (home)")
    print(f"  Favorite: {flow['fav_team']} by {flow['fav_spread']}")
    print()
    print(f"  Blowout (15+):  {flow['blowout_prob']*100:.1f}%")
    print(f"  Blowout (20+):  {flow['blowout_20_prob']*100:.1f}%")
    print(f"  Close (<=5):    {flow['close_game_prob']*100:.1f}%")
    print(f"  Close (<=3):    {flow['close_3_prob']*100:.1f}%")
    print(f"  OT chance:      {flow['overtime_prob']*100:.1f}%")
    print(f"  Dog wins:       {flow['dog_wins_prob']*100:.1f}%")
    print()
    print(f"  Pace estimate:  {flow['pace_estimate']} ({flow['pace_label']}, {flow['pace_delta_pct']:+.1f}% vs league avg)")
    print(f"  Star mins impact: {flow['star_minutes_impact']:+.1f} min (probability-weighted)")

    # Show impact on a hypothetical star player
    print("\n  --- Hypothetical star (35 min avg, on favored team) ---")
    star_data = {"season_mins_avg": 35, "l5_mins_avg": 35, "mins_30plus_pct": 95, "team": flow["fav_team"]}
    star_mins = estimate_player_minutes(star_data, flow)
    print(f"    Expected minutes: {star_mins['expected_minutes']} (delta: {star_mins['minutes_delta']:+.1f})")
    print(f"    Minutes upside:   {star_mins['minutes_upside']}")
    print(f"    Minutes downside: {star_mins['minutes_downside']}")
    print(f"    Blowout risk:     {star_mins['blowout_risk_pct']:.1f}%")

    for stat_name, proj_val in [("pts", 25.0), ("reb", 8.0), ("ast", 6.0)]:
        adj, fadj, conf, bkdn = adjust_projection_for_flow(proj_val, stat_name, flow, star_mins)
        print(f"    {stat_name.upper()} {proj_val} -> {adj} (flow adj: {fadj:+.2f}, confidence: {conf})")
        if bkdn:
            parts = [f"{k}={v:+.2f}" for k, v in bkdn.items() if v != 0]
            if parts:
                print(f"      Breakdown: {', '.join(parts)}")

    # Also show underdog star
    print(f"\n  --- Hypothetical star (33 min avg, on underdog {flow['dog_team']}) ---")
    dog_data = {"season_mins_avg": 33, "l5_mins_avg": 33, "mins_30plus_pct": 90, "team": flow["dog_team"]}
    dog_mins = estimate_player_minutes(dog_data, flow)
    print(f"    Expected minutes: {dog_mins['expected_minutes']} (delta: {dog_mins['minutes_delta']:+.1f})")
    for stat_name, proj_val in [("pts", 22.0), ("reb", 7.0), ("ast", 5.0)]:
        adj, fadj, conf, bkdn = adjust_projection_for_flow(proj_val, stat_name, flow, dog_mins)
        print(f"    {stat_name.upper()} {proj_val} -> {adj} (flow adj: {fadj:+.2f})")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 game_flow.py --build              # Build distributions from historical data")
        print("  python3 game_flow.py --matchup HOME AWAY SPREAD TOTAL")
        print("    Example: python3 game_flow.py --matchup LAL BOS -5.5 220")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "--build":
        _cli_build()
    elif cmd == "--matchup":
        _cli_matchup(sys.argv[2:])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()

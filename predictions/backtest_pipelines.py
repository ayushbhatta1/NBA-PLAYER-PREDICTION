#!/usr/bin/env python3
"""
Backtest: Old Pipeline vs NEXUS v4 vs Hybrid vs XGBoost-only (3-Leg Parlays)

Simulates each pipeline's 3-leg parlay selection on graded data from
Mar 11-14 and checks against actuals. No API calls needed.

Strategies:
  OLD PIPELINE  — tier+HR filter, gap-weighted scoring, BLK/STL boost
  NEXUS v4      — soft screen (CORE/FLEX/REACH/KILL), profile scoring
  HYBRID        — old pipeline filters + XGBoost ranking (falls back to old score)
  FLOOR-FIRST   — L10 HR >= 70%, no combos, base stats only, consistency-weighted
  XGBoost-only  — sort by xgb_prob, minimal filters (Mar 14 only)
"""

import json
import os
from pathlib import Path

DATES = [
    ("2026-03-11", "predictions/2026-03-11/v3_graded_full.json"),
    ("2026-03-12", "predictions/2026-03-12/graded_full_board.json"),
    ("2026-03-13", "predictions/2026-03-13/v4_graded_755_lines.json"),
    ("2026-03-14", "predictions/2026-03-14/v4_graded_392_lines.json"),
]

COMBO_STATS = {"pra", "pr", "pa", "ra"}
BOOST_STATS = {"blk", "stl"}


def load_props(date, path):
    """Load and normalize props from graded files."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        raw = data  # Mar 11 v3 format
    else:
        raw = data.get("results", [])

    props = []
    for r in raw:
        # Skip props without actuals (game not played / DNP)
        if r.get("actual") is None:
            continue

        # Normalize v3 fields
        p = dict(r)
        p.setdefault("streak_status", p.get("streak", "NEUTRAL"))
        p.setdefault("abs_gap", abs(p.get("gap") or 0))
        p.setdefault("l5_hit_rate", p.get("l10_hit_rate", 50))
        p.setdefault("season_avg", p.get("projection", p.get("line", 0)))
        p.setdefault("mins_30plus_pct", 70)
        p.setdefault("l10_miss_count", round((1 - p.get("l10_hit_rate", 50) / 100) * 10))
        p.setdefault("l10_floor", p.get("line", 0) - 5)
        p.setdefault("l10_values", [])
        p.setdefault("game", None)
        p.setdefault("is_home", None)
        p.setdefault("combo_penalized", False)
        p.setdefault("player_injury_status", None)

        # Normalize result
        if "result" not in p:
            p["result"] = "HIT" if p.get("hit", False) else "MISS"

        props.append(p)

    return props


def check_hit(prop):
    """Check if a prop hit based on actual vs line."""
    actual = prop["actual"]
    line = prop["line"]
    direction = prop.get("direction", "OVER").upper()
    if direction == "OVER":
        return actual > line
    else:
        return actual < line


# ── Old Pipeline ──────────────────────────────────────────────────────

def old_pipeline_select(props):
    """Simulate old pipeline 3-leg parlay selection."""
    # Pool: S/A OVER + S UNDER + B OVER (no error props)
    pool = []
    for p in props:
        tier = p.get("tier", "F")
        direction = p.get("direction", "OVER").upper()
        if tier in ("S", "A") and direction == "OVER":
            pool.append(p)
        elif tier == "S" and direction == "UNDER":
            pool.append(p)
        elif tier == "B" and direction == "OVER":
            pool.append(p)

    # Dedup: first by (player, stat)
    seen = set()
    deduped = []
    for p in pool:
        key = (p["player"], p["stat"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    pool = deduped

    # Score each prop
    for p in pool:
        abs_gap = p.get("abs_gap", abs(p.get("gap", 0)))
        l10_hr = p.get("l10_hit_rate", 50)
        l5_hr = p.get("l5_hit_rate", l10_hr)
        organic = 1.0 if not p.get("combo_penalized", False) else 0.5

        score = abs_gap * 0.4 + (l10_hr / 100) * 0.25 + (l5_hr / 100) * 0.15 + organic * 0.2

        # Modifiers
        stat = p.get("stat", "").lower()
        if stat in BOOST_STATS:
            score *= 1.2
        if stat in COMBO_STATS and abs_gap < 4:
            score *= 0.7
        streak = p.get("streak_status", "NEUTRAL")
        if streak == "HOT":
            score *= 1.1
        elif streak == "COLD":
            score *= 0.85
        status = (p.get("player_injury_status") or "").upper()
        if "GTD" in status or "QUESTIONABLE" in status:
            score *= 0.5
        if p.get("mins_30plus_pct", 70) < 50:
            score *= 0.7

        p["_old_score"] = score

    # Filter: L10 HR >= 60%, L5 HR >= 40%
    pool = [p for p in pool if p.get("l10_hit_rate", 50) >= 60 and p.get("l5_hit_rate", 50) >= 40]

    # Sort by score desc
    pool.sort(key=lambda p: p["_old_score"], reverse=True)

    # Pick top 3 with no same-game, no same-team
    selected = []
    used_games = set()
    used_teams = set()
    for p in pool:
        if len(selected) >= 3:
            break
        game = p.get("game")
        player = p["player"]

        # Extract team from game if available
        team = _get_team(p)

        # Game diversity (skip if game is None — Mar 11)
        if game and game in used_games:
            continue
        if team and team in used_teams:
            continue

        selected.append(p)
        if game:
            used_games.add(game)
        if team:
            used_teams.add(team)

    return selected


# ── NEXUS v4 ──────────────────────────────────────────────────────────

def nexus_v4_select(props):
    """Simulate NEXUS v4 3-leg parlay selection."""
    pool = []
    for p in props:
        tier = p.get("tier", "F")
        l10_hr = p.get("l10_hit_rate", 50)
        mins = p.get("mins_30plus_pct", 70)
        status = (p.get("player_injury_status") or "").upper()

        # Hard kills
        if tier in ("D", "F"):
            continue
        if mins < 40:
            continue
        if l10_hr < 40:
            continue
        if "OUT" in status or "DOUBTFUL" in status:
            continue

        # Soft filter count
        soft_fails = 0
        l5_hr = p.get("l5_hit_rate", l10_hr)
        abs_gap = p.get("abs_gap", abs(p.get("gap", 0)))
        miss_count = p.get("l10_miss_count", round((1 - l10_hr / 100) * 10))

        if 50 <= mins <= 59:
            soft_fails += 1
        if 55 <= l10_hr <= 59:
            soft_fails += 1
        if 30 <= l5_hr <= 39:
            soft_fails += 1
        if 1.0 <= abs_gap <= 1.49:
            soft_fails += 1
        if miss_count == 3:
            soft_fails += 1
        if "GTD" in status or "QUESTIONABLE" in status:
            soft_fails += 1

        # Season avg wrong side check
        season_avg = p.get("season_avg", p.get("projection", p.get("line", 0)))
        direction = p.get("direction", "OVER").upper()
        line = p.get("line", 0)
        if direction == "OVER" and season_avg < line:
            soft_fails += 1
        elif direction == "UNDER" and season_avg > line:
            soft_fails += 1

        # Floor risk
        l10_floor = p.get("l10_floor", line - 5)
        if direction == "OVER" and l10_floor < line - 5:
            soft_fails += 1

        if soft_fails >= 3:
            continue  # KILL

        # Screen multiplier
        if soft_fails == 0:
            screen_mult = 1.0
        elif soft_fails == 1:
            screen_mult = 0.85
        else:
            screen_mult = 0.70

        # Profile score: gap(30%) + season_margin(20%) + consistency(25%) + mins(10%) + floor(5%) + context(10%)
        gap_score = min(abs_gap / 5.0, 1.0) * 30
        season_margin = abs(season_avg - line)
        margin_score = min(season_margin / 5.0, 1.0) * 20
        consistency_score = (l10_hr / 100) * 25
        mins_score = min(mins / 100, 1.0) * 10
        floor_score = max(0, 5 - miss_count) / 5.0 * 5

        # Context score
        ctx = 0
        streak = p.get("streak_status", "NEUTRAL")
        if streak == "HOT":
            ctx += 3
        elif streak == "COLD":
            ctx -= 2
        if p.get("is_home"):
            ctx += 1
        stat = p.get("stat", "").lower()
        if stat in COMBO_STATS:
            ctx -= 4
        context_score = max(0, min((ctx + 5) / 10.0, 1.0)) * 10

        nexus_score = (gap_score + margin_score + consistency_score + mins_score + floor_score + context_score) * screen_mult
        p["_nexus_score"] = nexus_score
        pool.append(p)

    # Sort by nexus_score desc
    pool.sort(key=lambda p: p["_nexus_score"], reverse=True)

    # Pick top 3 with diversity
    selected = []
    used_games = set()
    used_teams = set()
    for p in pool:
        if len(selected) >= 3:
            break
        game = p.get("game")
        team = _get_team(p)

        if game and game in used_games:
            continue
        if team and team in used_teams:
            continue

        selected.append(p)
        if game:
            used_games.add(game)
        if team:
            used_teams.add(team)

    return selected


# ── XGBoost-only ──────────────────────────────────────────────────────

def xgb_select(props):
    """Simulate XGBoost-only 3-leg parlay selection."""
    pool = [p for p in props
            if p.get("xgb_prob") is not None
            and p.get("tier", "F") != "F"
            and p.get("l10_hit_rate", 50) >= 40]

    if not pool:
        return []

    pool.sort(key=lambda p: p["xgb_prob"], reverse=True)

    selected = []
    used_games = set()
    for p in pool:
        if len(selected) >= 3:
            break
        game = p.get("game")
        if game and game in used_games:
            continue
        selected.append(p)
        if game:
            used_games.add(game)

    return selected


# ── Hybrid (Old Filters + XGBoost Ranking) ────────────────────────────

def hybrid_select(props):
    """Old pipeline's proven filters + XGBoost ranking when available."""
    # Same pool as old pipeline: S/A OVER + S UNDER + B OVER
    pool = []
    for p in props:
        tier = p.get("tier", "F")
        direction = p.get("direction", "OVER").upper()
        if tier in ("S", "A") and direction == "OVER":
            pool.append(p)
        elif tier == "S" and direction == "UNDER":
            pool.append(p)
        elif tier == "B" and direction == "OVER":
            pool.append(p)

    # Dedup
    seen = set()
    deduped = []
    for p in pool:
        key = (p["player"], p["stat"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    pool = deduped

    # Same HR filters as old pipeline
    pool = [p for p in pool if p.get("l10_hit_rate", 50) >= 60 and p.get("l5_hit_rate", 50) >= 40]

    # Apply old pipeline modifiers for penalty/boost (GTD, low mins, combo)
    for p in pool:
        penalty = 1.0
        stat = p.get("stat", "").lower()
        if stat in COMBO_STATS and p.get("abs_gap", 0) < 4:
            penalty *= 0.7
        status = (p.get("player_injury_status") or "").upper()
        if "GTD" in status or "QUESTIONABLE" in status:
            penalty *= 0.5
        if p.get("mins_30plus_pct", 70) < 50:
            penalty *= 0.7
        streak = p.get("streak_status", "NEUTRAL")
        if streak == "COLD":
            penalty *= 0.85
        p["_hybrid_penalty"] = penalty

    # Rank by XGBoost prob if available, else fall back to old score
    has_xgb = any(p.get("xgb_prob") is not None for p in pool)
    if has_xgb:
        for p in pool:
            p["_hybrid_score"] = (p.get("xgb_prob") or 0) * p["_hybrid_penalty"]
    else:
        # Fall back to old pipeline scoring
        for p in pool:
            abs_gap = p.get("abs_gap", 0)
            l10_hr = p.get("l10_hit_rate", 50)
            l5_hr = p.get("l5_hit_rate", l10_hr)
            organic = 1.0 if not p.get("combo_penalized", False) else 0.5
            score = abs_gap * 0.4 + (l10_hr / 100) * 0.25 + (l5_hr / 100) * 0.15 + organic * 0.2
            stat = p.get("stat", "").lower()
            if stat in BOOST_STATS:
                score *= 1.2
            if p.get("streak_status", "NEUTRAL") == "HOT":
                score *= 1.1
            p["_hybrid_score"] = score * p["_hybrid_penalty"]

    pool.sort(key=lambda p: p["_hybrid_score"], reverse=True)

    # Pick top 3, game diversity
    selected = []
    used_games = set()
    for p in pool:
        if len(selected) >= 3:
            break
        game = p.get("game")
        if game and game in used_games:
            continue
        selected.append(p)
        if game:
            used_games.add(game)

    return selected


# ── Floor-First (Consistency Over Gap) ────────────────────────────────

def floor_first_select(props):
    """Prioritize consistency: high HR, base stats only, no combos."""
    pool = []
    for p in props:
        tier = p.get("tier", "F")
        stat = p.get("stat", "").lower()
        l10_hr = p.get("l10_hit_rate", 50)
        l5_hr = p.get("l5_hit_rate", l10_hr)
        mins = p.get("mins_30plus_pct", 70)
        status = (p.get("player_injury_status") or "").upper()

        # Strict filters: no combos, no injured, high HR, solid minutes
        if stat in COMBO_STATS:
            continue
        if tier in ("D", "F"):
            continue
        if l10_hr < 70:
            continue
        if l5_hr < 50:
            continue
        if mins < 60:
            continue
        if "OUT" in status or "DOUBTFUL" in status or "GTD" in status or "QUESTIONABLE" in status:
            continue

        # Score: consistency-heavy (HR 50% + gap 20% + mins 15% + BLK/STL 15%)
        hr_score = (l10_hr / 100) * 0.35 + (l5_hr / 100) * 0.15
        gap_score = min(p.get("abs_gap", 0) / 5.0, 1.0) * 0.20
        mins_s = min(mins / 100, 1.0) * 0.15
        boost = 1.15 if stat in BOOST_STATS else 1.0

        # XGBoost tiebreaker when available
        xgb = p.get("xgb_prob")
        xgb_bonus = (xgb * 0.15) if xgb is not None else 0

        p["_floor_score"] = (hr_score + gap_score + mins_s + xgb_bonus) * boost
        pool.append(p)

    pool.sort(key=lambda p: p["_floor_score"], reverse=True)

    selected = []
    used_games = set()
    for p in pool:
        if len(selected) >= 3:
            break
        game = p.get("game")
        if game and game in used_games:
            continue
        selected.append(p)
        if game:
            used_games.add(game)

    return selected


# ── Helpers ───────────────────────────────────────────────────────────

def _get_team(prop):
    """Extract team identifier from game field."""
    return None  # Game diversity handles correlation risk


def format_leg(p):
    """Format a single leg for display."""
    hit = check_hit(p)
    mark = "\u2713" if hit else "\u2717"
    return f"{p['player']} {p['stat']} {p['direction']} {p['line']} ({mark})"


def _run_strategy(name, selector, props, stats, all_picks):
    """Run a single strategy and collect stats."""
    picks = selector(props)
    if not picks:
        print(f"  {name:14s}  No qualifying picks")
        return
    hits = [check_hit(p) for p in picks]
    leg_hits = sum(hits)
    parlay_hit = all(hits)
    stats["parlays"] += 1
    stats["parlay_hits"] += int(parlay_hit)
    stats["legs"] += len(picks)
    stats["leg_hits"] += leg_hits
    all_picks.extend(picks)
    legs_str = " | ".join(format_leg(p) for p in picks)
    result = "HIT" if parlay_hit else "MISS"
    print(f"  {name:14s}  {legs_str}  -> {result} ({leg_hits}/{len(picks)})")


def run_backtest():
    print("=" * 70)
    print("BACKTEST: 5 Strategies x 4 Days (3-Leg Parlays)")
    print("=" * 70)

    strategies = {
        "OLD PIPELINE": {"fn": old_pipeline_select, "parlays": 0, "parlay_hits": 0, "legs": 0, "leg_hits": 0, "picks": []},
        "NEXUS v4":     {"fn": nexus_v4_select,     "parlays": 0, "parlay_hits": 0, "legs": 0, "leg_hits": 0, "picks": []},
        "HYBRID":       {"fn": hybrid_select,       "parlays": 0, "parlay_hits": 0, "legs": 0, "leg_hits": 0, "picks": []},
        "FLOOR-FIRST":  {"fn": floor_first_select,  "parlays": 0, "parlay_hits": 0, "legs": 0, "leg_hits": 0, "picks": []},
        "XGBoost":      {"fn": xgb_select,          "parlays": 0, "parlay_hits": 0, "legs": 0, "leg_hits": 0, "picks": []},
    }

    for date, path in DATES:
        if not os.path.exists(path):
            print(f"\n--- {date} --- SKIPPED (file not found)")
            continue

        props = load_props(date, path)
        has_xgb = any(p.get("xgb_prob") is not None for p in props)
        print(f"\n--- {date} --- ({len(props)} graded props)")

        for name, s in strategies.items():
            if name == "XGBoost" and not has_xgb:
                continue
            _run_strategy(name, s["fn"], props, s, s["picks"])

    # ── Summary Table ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Strategy':14s}  {'Parlays':>12s}  {'Legs':>12s}")
    print("-" * 44)
    for name, s in strategies.items():
        if s["parlays"] == 0:
            continue
        p_pct = s["parlay_hits"] / s["parlays"] * 100
        l_pct = s["leg_hits"] / s["legs"] * 100
        p_str = f"{s['parlay_hits']}/{s['parlays']} ({p_pct:.0f}%)"
        l_str = f"{s['leg_hits']}/{s['legs']} ({l_pct:.0f}%)"
        note = " *" if name == "XGBoost" else ""
        print(f"{name:14s}  {p_str:>12s}  {l_str:>12s}{note}")
    print("* XGBoost: Mar 14 only (no xgb_prob on earlier dates)")

    # ── Pattern Analysis ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PATTERN ANALYSIS: What do winning legs share?")
    print("=" * 70)

    # Collect all unique picked legs across strategies with hit/miss
    all_legs = []
    for name, s in strategies.items():
        for p in s["picks"]:
            all_legs.append({
                "strategy": name,
                "player": p["player"],
                "stat": p.get("stat", ""),
                "direction": p.get("direction", "OVER"),
                "tier": p.get("tier", "?"),
                "l10_hr": p.get("l10_hit_rate", 0),
                "l5_hr": p.get("l5_hit_rate", 0),
                "abs_gap": p.get("abs_gap", 0),
                "mins": p.get("mins_30plus_pct", 70),
                "is_combo": p.get("stat", "").lower() in COMBO_STATS,
                "hit": check_hit(p),
            })

    if not all_legs:
        return

    hits = [l for l in all_legs if l["hit"]]
    misses = [l for l in all_legs if not l["hit"]]

    def avg(lst, key):
        vals = [l[key] for l in lst if l[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    print(f"\n  {'Metric':20s}  {'Hits':>8s}  {'Misses':>8s}")
    print("  " + "-" * 40)
    print(f"  {'Avg L10 HR':20s}  {avg(hits, 'l10_hr'):>7.1f}%  {avg(misses, 'l10_hr'):>7.1f}%")
    print(f"  {'Avg L5 HR':20s}  {avg(hits, 'l5_hr'):>7.1f}%  {avg(misses, 'l5_hr'):>7.1f}%")
    print(f"  {'Avg Gap':20s}  {avg(hits, 'abs_gap'):>8.2f}  {avg(misses, 'abs_gap'):>8.2f}")
    print(f"  {'Avg Mins 30+%':20s}  {avg(hits, 'mins'):>7.1f}%  {avg(misses, 'mins'):>7.1f}%")

    combo_hits = sum(1 for l in hits if l["is_combo"])
    combo_misses = sum(1 for l in misses if l["is_combo"])
    print(f"  {'Combo stat legs':20s}  {combo_hits:>5d}/{len(hits)}  {combo_misses:>5d}/{len(misses)}")

    # Stat breakdown
    print(f"\n  Stat type breakdown (hit rate across all strategies):")
    stat_groups = {}
    for l in all_legs:
        st = l["stat"].lower()
        if st not in stat_groups:
            stat_groups[st] = {"hit": 0, "total": 0}
        stat_groups[st]["total"] += 1
        if l["hit"]:
            stat_groups[st]["hit"] += 1
    for st, v in sorted(stat_groups.items(), key=lambda x: -x[1]["total"]):
        pct = v["hit"] / v["total"] * 100 if v["total"] else 0
        print(f"    {st:6s}  {v['hit']}/{v['total']} ({pct:.0f}%)")

    # Tier breakdown
    print(f"\n  Tier breakdown:")
    tier_groups = {}
    for l in all_legs:
        t = l["tier"]
        if t not in tier_groups:
            tier_groups[t] = {"hit": 0, "total": 0}
        tier_groups[t]["total"] += 1
        if l["hit"]:
            tier_groups[t]["hit"] += 1
    for t in ["S", "A", "B", "C"]:
        if t in tier_groups:
            v = tier_groups[t]
            pct = v["hit"] / v["total"] * 100
            print(f"    Tier {t}:  {v['hit']}/{v['total']} ({pct:.0f}%)")


if __name__ == "__main__":
    run_backtest()

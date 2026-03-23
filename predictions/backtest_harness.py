#!/usr/bin/env python3
"""
Backtesting Harness v1 — Replay any pipeline version against any historical board.

Usage:
    # Compare current pipeline vs a variant on all graded dates
    python3 predictions/backtest_harness.py --all

    # Test a specific date
    python3 predictions/backtest_harness.py --date 2026-03-17

    # Run a specific experiment (defined in EXPERIMENTS below)
    python3 predictions/backtest_harness.py --experiment under_bias_sweep

    # List available experiments
    python3 predictions/backtest_harness.py --list

    # Quick summary of all graded days
    python3 predictions/backtest_harness.py --summary

Architecture:
    1. Load graded data (predictions + actuals) from historical date folders
    2. Apply a "pipeline variant" function that re-scores/re-selects from the same data
    3. Compare variant results vs baseline (what actually shipped that day)
    4. Report: per-leg accuracy, parlay cash rate, stat breakdown, tier breakdown
    5. Track experiments in experiments_log.json for longitudinal analysis

A pipeline variant is a function: f(props) -> selected_legs
The harness handles loading, grading, comparison, and reporting.
"""

import json
import os
import sys
import copy
from pathlib import Path
from datetime import datetime
from collections import defaultdict

BASE_DIR = Path(__file__).parent
DATES_DIR = BASE_DIR
LOG_FILE = BASE_DIR / "experiments_log.json"
COMBO_STATS = {"pra", "pr", "pa", "ra", "stl_blk"}
BOOST_STATS = {"blk", "stl"}


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def discover_graded_dates():
    """Find all date folders that have graded results."""
    dates = []
    for d in sorted(DATES_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if len(name) != 10 or not name.startswith("2026-03-"):
            continue
        # Look for graded files
        graded = list(d.glob("v4_graded_*.json")) + list(d.glob("graded_*.json")) + list(d.glob("v3_graded_*.json"))
        if graded:
            dates.append((name, str(graded[0])))
    return dates


def load_graded_props(date, path):
    """Load graded props with actuals. Returns list of normalized prop dicts."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        raw = data
    else:
        raw = data.get("results", [])

    props = []
    for r in raw:
        if r.get("actual") is None:
            continue

        p = dict(r)
        # Normalize fields across pipeline versions
        p.setdefault("streak_status", p.get("streak", "NEUTRAL"))
        p.setdefault("abs_gap", abs(p.get("gap") or 0))
        p.setdefault("l5_hit_rate", p.get("l10_hit_rate", 50))
        p.setdefault("season_avg", p.get("projection", p.get("line", 0)))
        p.setdefault("mins_30plus_pct", 70)
        p.setdefault("l10_miss_count", round((1 - p.get("l10_hit_rate", 50) / 100) * 10))
        p.setdefault("l10_floor", p.get("line", 0) - 5)
        p.setdefault("l10_values", [])
        p.setdefault("game", "UNK")
        p.setdefault("is_home", None)
        p.setdefault("player_injury_status", None)
        p.setdefault("ensemble_prob", p.get("xgb_prob", 0.5))
        p.setdefault("xgb_prob", 0.5)
        p.setdefault("sim_prob", None)
        p.setdefault("sniper_score", 0)
        p.setdefault("composite_score", 0)

        # Compute actual hit
        actual = p["actual"]
        line = p["line"]
        direction = p.get("direction", "OVER").upper()
        if direction == "OVER":
            p["_hit"] = actual > line
        else:
            p["_hit"] = actual < line

        props.append(p)

    return props


def load_full_board(date):
    """Load the full board predictions (before grading) for richer features."""
    full_path = DATES_DIR / date / f"{date}_full_board.json"
    if not full_path.exists():
        return {}
    with open(full_path) as f:
        data = json.load(f)
    # Index by (player, stat) for lookup
    index = {}
    results = data if isinstance(data, list) else data.get("results", data.get("props", []))
    for p in results:
        key = (p.get("player", ""), p.get("stat", ""))
        index[key] = p
    return index


# ═══════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════

def get_team(p):
    """Extract player's team from game string."""
    game = p.get("game", "") or ""
    is_home = p.get("is_home")
    if "@" in game:
        away, home = game.split("@")
        if is_home is True:
            return home
        elif is_home is False:
            return away
    return None


def select_diverse(pool, n=3, key="_score"):
    """Pick top N from pool with game + team diversity."""
    pool_sorted = sorted(pool, key=lambda p: p.get(key, 0), reverse=True)
    selected = []
    used_games = set()
    used_teams = set()
    for p in pool_sorted:
        if len(selected) >= n:
            break
        game = p.get("game")
        team = get_team(p)
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


def parlay_hit(legs):
    """Check if all legs in a parlay hit."""
    return all(leg.get("_hit", False) for leg in legs)


def leg_hit_rate(legs):
    """Compute hit rate across legs."""
    if not legs:
        return 0.0
    return sum(1 for l in legs if l.get("_hit", False)) / len(legs) * 100


# ═══════════════════════════════════════════════════════════════════════
#  PIPELINE VARIANTS (strategies to backtest)
# ═══════════════════════════════════════════════════════════════════════

def baseline_engine_safe(props):
    """Current Engine SAFE 3-leg: S/A/B UNDER-first, ensemble_prob >= 0.50, composite score."""
    pool = []
    for p in props:
        tier = p.get("tier", "F")
        if tier not in ("S", "A", "B"):
            continue
        if p.get("mins_30plus_pct", 70) < 60:
            continue
        if p.get("l10_hit_rate", 50) < 60:
            continue
        if p.get("stat", "") in COMBO_STATS:
            continue
        ens = p.get("ensemble_prob", 0.5)
        if ens < 0.50:
            continue

        # Composite score (simplified v11)
        direction = p.get("direction", "OVER").upper()
        score = ens
        if direction == "UNDER":
            score += 0.30
        streak = p.get("streak_status", "NEUTRAL")
        if streak == "COLD" and direction == "UNDER":
            score += 0.12
        elif streak == "HOT":
            score -= 0.08
        if p.get("stat", "") in BOOST_STATS and direction == "UNDER":
            score += 0.05
        hr = p.get("l10_hit_rate", 50)
        if hr >= 70:
            score += 0.10

        p["_score"] = score
        pool.append(p)

    return select_diverse(pool, n=3)


def baseline_nexus_safe(props):
    """Current NEXUS v4 SAFE 3-leg: soft screen + profile scoring."""
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
        miss_count = p.get("l10_miss_count", 5)

        if 50 <= mins <= 59: soft_fails += 1
        if 55 <= l10_hr <= 59: soft_fails += 1
        if 30 <= l5_hr <= 39: soft_fails += 1
        if 1.0 <= abs_gap <= 1.49: soft_fails += 1
        if miss_count == 3: soft_fails += 1
        if "GTD" in status or "QUESTIONABLE" in status: soft_fails += 1

        if soft_fails >= 3:
            continue

        screen_mult = {0: 1.0, 1: 0.85, 2: 0.70}.get(soft_fails, 0.70)

        # Profile score
        gap_score = min(abs_gap / 5.0, 1.0) * 30
        margin_score = min(abs(p.get("season_avg", 0) - p.get("line", 0)) / 5.0, 1.0) * 20
        consistency_score = (l10_hr / 100) * 25
        mins_score = min(mins / 100, 1.0) * 10
        floor_score = max(0, 5 - miss_count) / 5.0 * 5
        context_score = 5  # simplified

        p["_score"] = (gap_score + margin_score + consistency_score + mins_score + floor_score + context_score) * screen_mult
        pool.append(p)

    return select_diverse(pool, n=3)


def variant_under_only(props):
    """ALL UNDER, no exceptions. S/A/B tier, L10 HR >= 50%."""
    pool = []
    for p in props:
        if p.get("direction", "OVER").upper() != "UNDER":
            continue
        tier = p.get("tier", "F")
        if tier not in ("S", "A", "B"):
            continue
        if p.get("l10_hit_rate", 50) < 50:
            continue
        if p.get("stat", "") in COMBO_STATS:
            continue
        abs_gap = p.get("abs_gap", 0)
        hr = p.get("l10_hit_rate", 50)
        p["_score"] = abs_gap * 0.4 + hr * 0.006
        pool.append(p)
    return select_diverse(pool, n=3)


def variant_blk_stl_under(props):
    """BLK/STL UNDER only. Our historically best stat category."""
    pool = []
    for p in props:
        if p.get("direction", "OVER").upper() != "UNDER":
            continue
        if p.get("stat", "") not in BOOST_STATS:
            continue
        abs_gap = p.get("abs_gap", 0)
        hr = p.get("l10_hit_rate", 50)
        miss_count = p.get("l10_miss_count", 5)
        p["_score"] = abs_gap * 0.3 + (1 - hr / 100) * 0.3 + miss_count * 0.05
        pool.append(p)
    return select_diverse(pool, n=3)


def variant_high_hr_under(props):
    """UNDER with L10 HR >= 70%. The "hit rate floor" approach."""
    pool = []
    for p in props:
        if p.get("direction", "OVER").upper() != "UNDER":
            continue
        if p.get("l10_hit_rate", 50) < 70:
            continue
        tier = p.get("tier", "F")
        if tier in ("D", "F"):
            continue
        if p.get("stat", "") in COMBO_STATS:
            continue
        abs_gap = p.get("abs_gap", 0)
        p["_score"] = p.get("l10_hit_rate", 50) / 100 + abs_gap * 0.1
        pool.append(p)
    return select_diverse(pool, n=3)


def variant_sim_driven(props):
    """Use sim_prob as primary signal. Monte Carlo > heuristics."""
    pool = []
    for p in props:
        sim = p.get("sim_prob")
        if sim is None or sim == 0:
            # Fall back to ensemble_prob
            sim = p.get("ensemble_prob", 0.5)
        if sim < 0.55:
            continue
        tier = p.get("tier", "F")
        if tier in ("D", "F"):
            continue
        if p.get("stat", "") in COMBO_STATS:
            continue
        p["_score"] = sim
        pool.append(p)
    return select_diverse(pool, n=3)


def variant_ensemble_top(props):
    """Pure ensemble_prob ranking. Let the ML decide."""
    pool = []
    for p in props:
        ens = p.get("ensemble_prob", 0.5)
        if ens < 0.55:
            continue
        tier = p.get("tier", "F")
        if tier in ("D", "F"):
            continue
        if p.get("stat", "") in COMBO_STATS:
            continue
        p["_score"] = ens
        pool.append(p)
    return select_diverse(pool, n=3)


def variant_cold_under_strict(props):
    """COLD streak + UNDER + L10 HR < 50%. Betting on continuation of misses."""
    pool = []
    for p in props:
        if p.get("direction", "OVER").upper() != "UNDER":
            continue
        if p.get("l10_hit_rate", 50) >= 50:
            continue  # We want LOW hit rate = player consistently going under
        streak = p.get("streak_status", p.get("streak", "NEUTRAL"))
        if streak != "COLD":
            continue
        tier = p.get("tier", "F")
        if tier in ("D", "F"):
            continue
        if p.get("stat", "") in COMBO_STATS:
            continue
        abs_gap = p.get("abs_gap", 0)
        miss = p.get("l10_miss_count", 5)
        p["_score"] = abs_gap * 0.3 + miss * 0.07
        pool.append(p)
    return select_diverse(pool, n=3)


def variant_line_floor_blk(props):
    """BLK UNDER on low lines (0.5-1.5). Line floor exploitation."""
    pool = []
    for p in props:
        if p.get("direction", "OVER").upper() != "UNDER":
            continue
        if p.get("stat", "") != "blk":
            continue
        line = p.get("line", 99)
        if line > 1.5:
            continue
        l10_avg = p.get("l10_avg", p.get("season_avg", line))
        if l10_avg and l10_avg > 0:
            inflation = (line - l10_avg) / l10_avg * 100
        else:
            inflation = 0
        p["_score"] = inflation * 0.01 + (1 - p.get("l10_hit_rate", 50) / 100) * 0.5
        pool.append(p)
    return select_diverse(pool, n=3)


def variant_gap_monster(props):
    """Biggest absolute gap, any direction. The line is very wrong."""
    pool = []
    for p in props:
        tier = p.get("tier", "F")
        if tier in ("D", "F"):
            continue
        if p.get("stat", "") in COMBO_STATS:
            continue
        abs_gap = p.get("abs_gap", 0)
        if abs_gap < 3.0:
            continue
        p["_score"] = abs_gap
        pool.append(p)
    return select_diverse(pool, n=3)


def variant_anti_over(props):
    """Avoid all OVERs. UNDER-only, any tier S/A/B/C, ensemble >= 0.55."""
    pool = []
    for p in props:
        if p.get("direction", "OVER").upper() != "UNDER":
            continue
        tier = p.get("tier", "F")
        if tier in ("D", "F"):
            continue
        ens = p.get("ensemble_prob", 0.5)
        if ens < 0.55:
            continue
        p["_score"] = ens + p.get("abs_gap", 0) * 0.05
        pool.append(p)
    return select_diverse(pool, n=3)


def variant_composite_v2(props):
    """Enhanced composite: ensemble + sim + gap + HR + UNDER bias + miss streak."""
    pool = []
    for p in props:
        tier = p.get("tier", "F")
        if tier in ("D", "F"):
            continue
        if p.get("stat", "") in COMBO_STATS:
            continue
        if p.get("mins_30plus_pct", 70) < 50:
            continue

        direction = p.get("direction", "OVER").upper()
        ens = p.get("ensemble_prob", 0.5)
        sim = p.get("sim_prob") or ens
        abs_gap = p.get("abs_gap", 0)
        hr = p.get("l10_hit_rate", 50)
        miss_count = p.get("l10_miss_count", 5)

        score = 0.0
        # ML signal (40%)
        score += (ens * 0.25 + sim * 0.15)
        # Gap signal (20%)
        score += min(abs_gap / 8.0, 0.20)
        # Hit rate (15%)
        score += (hr / 100) * 0.15
        # UNDER bias (15%)
        if direction == "UNDER":
            score += 0.15
        # Miss streak bonus for UNDER (10%)
        if direction == "UNDER" and miss_count >= 6:
            score += 0.10
        elif direction == "UNDER" and miss_count >= 4:
            score += 0.05

        p["_score"] = score
        pool.append(p)
    return select_diverse(pool, n=3)


# ═══════════════════════════════════════════════════════════════════════
#  EXPERIMENTS (named collections of variants to compare)
# ═══════════════════════════════════════════════════════════════════════

VARIANTS = {
    "engine_safe": baseline_engine_safe,
    "nexus_safe": baseline_nexus_safe,
    "under_only": variant_under_only,
    "blk_stl_under": variant_blk_stl_under,
    "high_hr_under": variant_high_hr_under,
    "sim_driven": variant_sim_driven,
    "ensemble_top": variant_ensemble_top,
    "cold_under_strict": variant_cold_under_strict,
    "line_floor_blk": variant_line_floor_blk,
    "gap_monster": variant_gap_monster,
    "anti_over": variant_anti_over,
    "composite_v2": variant_composite_v2,
}

EXPERIMENTS = {
    "full_comparison": {
        "description": "Compare all variants head-to-head across all graded dates",
        "variants": list(VARIANTS.keys()),
    },
    "under_bias_sweep": {
        "description": "Compare different UNDER-focused strategies",
        "variants": ["engine_safe", "under_only", "blk_stl_under", "high_hr_under", "cold_under_strict", "anti_over"],
    },
    "ml_vs_heuristic": {
        "description": "ML-driven (ensemble, sim) vs heuristic (gap, HR) selection",
        "variants": ["engine_safe", "nexus_safe", "sim_driven", "ensemble_top", "composite_v2"],
    },
    "stat_specialist": {
        "description": "BLK/STL specialists vs general strategies",
        "variants": ["engine_safe", "blk_stl_under", "line_floor_blk"],
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate_variant(variant_fn, dates_data):
    """Run a variant across all dates, return comprehensive stats."""
    results = {
        "dates": {},
        "total_parlays": 0,
        "parlays_cashed": 0,
        "total_legs": 0,
        "legs_hit": 0,
        "by_stat": defaultdict(lambda: {"hit": 0, "total": 0}),
        "by_tier": defaultdict(lambda: {"hit": 0, "total": 0}),
        "by_direction": defaultdict(lambda: {"hit": 0, "total": 0}),
        "empty_days": 0,
    }

    for date, props in dates_data:
        # Deep copy so variants can modify props
        props_copy = [copy.deepcopy(p) for p in props]
        legs = variant_fn(props_copy)

        if not legs or len(legs) < 3:
            results["empty_days"] += 1
            results["dates"][date] = {"legs": 0, "hits": 0, "cashed": False, "note": "NO BUILD"}
            continue

        hits = sum(1 for l in legs if l.get("_hit", False))
        cashed = hits == len(legs)

        results["total_parlays"] += 1
        if cashed:
            results["parlays_cashed"] += 1
        results["total_legs"] += len(legs)
        results["legs_hit"] += hits

        results["dates"][date] = {
            "legs": len(legs),
            "hits": hits,
            "cashed": cashed,
            "picks": [(l["player"], l.get("stat","?"), l.get("direction","?"), l.get("line",0), l.get("_hit",False)) for l in legs],
        }

        for l in legs:
            stat = l.get("stat", "?")
            tier = l.get("tier", "?")
            direction = l.get("direction", "?").upper()
            hit = l.get("_hit", False)
            results["by_stat"][stat]["total"] += 1
            results["by_tier"][tier]["total"] += 1
            results["by_direction"][direction]["total"] += 1
            if hit:
                results["by_stat"][stat]["hit"] += 1
                results["by_tier"][tier]["hit"] += 1
                results["by_direction"][direction]["hit"] += 1

    # Compute rates
    results["parlay_cash_rate"] = (results["parlays_cashed"] / results["total_parlays"] * 100) if results["total_parlays"] > 0 else 0
    results["leg_hit_rate"] = (results["legs_hit"] / results["total_legs"] * 100) if results["total_legs"] > 0 else 0

    return results


# ═══════════════════════════════════════════════════════════════════════
#  REPORTING
# ═══════════════════════════════════════════════════════════════════════

def print_comparison(experiment_name, variant_results):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT: {experiment_name}")
    print(f"{'='*80}\n")

    # Sort by parlay cash rate, then leg hit rate
    ranked = sorted(variant_results.items(),
                    key=lambda x: (x[1]["parlay_cash_rate"], x[1]["leg_hit_rate"]),
                    reverse=True)

    # Header
    print(f"  {'Variant':<25} {'Parlays':>8} {'Cashed':>8} {'Cash%':>7} {'Legs':>6} {'Hit':>5} {'Leg%':>7} {'Empty':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*5} {'-'*7} {'-'*6}")

    for name, r in ranked:
        print(f"  {name:<25} {r['total_parlays']:>8} {r['parlays_cashed']:>8} {r['parlay_cash_rate']:>6.1f}% {r['total_legs']:>6} {r['legs_hit']:>5} {r['leg_hit_rate']:>6.1f}% {r['empty_days']:>6}")

    # Winner
    if ranked:
        winner = ranked[0]
        print(f"\n  WINNER: {winner[0]} ({winner[1]['parlay_cash_rate']:.1f}% cash rate, {winner[1]['leg_hit_rate']:.1f}% per-leg)")

    # Per-date breakdown for top 3
    print(f"\n  {'─'*70}")
    print(f"  PER-DATE BREAKDOWN (top 3 variants)")
    print(f"  {'─'*70}\n")

    for name, r in ranked[:3]:
        print(f"  {name}:")
        for date in sorted(r["dates"].keys()):
            d = r["dates"][date]
            if d.get("note"):
                print(f"    {date}: {d['note']}")
            else:
                status = "CASHED" if d["cashed"] else f"{d['hits']}/{d['legs']} legs"
                picks_str = ""
                if d.get("picks"):
                    picks_str = " | " + ", ".join(
                        f"{'✓' if hit else '✗'} {player} {dir} {stat} {line}"
                        for player, stat, dir, line, hit in d["picks"]
                    )
                print(f"    {date}: {status}{picks_str}")
        print()

    # Stat/Tier/Direction breakdown for winner
    winner_name, winner_r = ranked[0]
    print(f"  {'─'*70}")
    print(f"  WINNER BREAKDOWN: {winner_name}")
    print(f"  {'─'*70}\n")

    print(f"  By Stat:")
    for stat, counts in sorted(winner_r["by_stat"].items(), key=lambda x: x[1]["total"], reverse=True):
        rate = counts["hit"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"    {stat:>6}: {counts['hit']}/{counts['total']} ({rate:.0f}%)")

    print(f"\n  By Tier:")
    for tier, counts in sorted(winner_r["by_tier"].items()):
        rate = counts["hit"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"    {tier:>6}: {counts['hit']}/{counts['total']} ({rate:.0f}%)")

    print(f"\n  By Direction:")
    for direction, counts in sorted(winner_r["by_direction"].items()):
        rate = counts["hit"] / counts["total"] * 100 if counts["total"] > 0 else 0
        print(f"    {direction:>6}: {counts['hit']}/{counts['total']} ({rate:.0f}%)")


def print_summary(dates_data):
    """Print a quick summary of all graded days."""
    print(f"\n{'='*60}")
    print(f"  GRADED DATA SUMMARY")
    print(f"{'='*60}\n")

    total_props = 0
    total_hits = 0
    under_hits = 0
    under_total = 0
    over_hits = 0
    over_total = 0

    for date, props in dates_data:
        hits = sum(1 for p in props if p.get("_hit", False))
        total = len(props)
        acc = hits / total * 100 if total > 0 else 0

        u_hit = sum(1 for p in props if p.get("direction", "").upper() == "UNDER" and p.get("_hit", False))
        u_tot = sum(1 for p in props if p.get("direction", "").upper() == "UNDER")
        o_hit = sum(1 for p in props if p.get("direction", "").upper() == "OVER" and p.get("_hit", False))
        o_tot = sum(1 for p in props if p.get("direction", "").upper() == "OVER")

        print(f"  {date}: {hits}/{total} ({acc:.1f}%) | UNDER {u_hit}/{u_tot} ({u_hit/u_tot*100:.0f}% ) | OVER {o_hit}/{o_tot} ({o_hit/o_tot*100:.0f}%)" if u_tot > 0 and o_tot > 0 else f"  {date}: {hits}/{total} ({acc:.1f}%)")

        total_props += total
        total_hits += hits
        under_hits += u_hit
        under_total += u_tot
        over_hits += o_hit
        over_total += o_tot

    print(f"\n  TOTAL: {total_hits}/{total_props} ({total_hits/total_props*100:.1f}%)")
    if under_total > 0:
        print(f"  UNDER: {under_hits}/{under_total} ({under_hits/under_total*100:.1f}%)")
    if over_total > 0:
        print(f"  OVER:  {over_hits}/{over_total} ({over_hits/over_total*100:.1f}%)")
    print(f"  Days:  {len(dates_data)}")


def save_experiment_log(experiment_name, variant_results):
    """Append experiment results to log file for longitudinal tracking."""
    log = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            log = json.load(f)

    entry = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "results": {},
    }

    for name, r in variant_results.items():
        entry["results"][name] = {
            "parlay_cash_rate": round(r["parlay_cash_rate"], 1),
            "leg_hit_rate": round(r["leg_hit_rate"], 1),
            "total_parlays": r["total_parlays"],
            "parlays_cashed": r["parlays_cashed"],
            "total_legs": r["total_legs"],
            "legs_hit": r["legs_hit"],
            "empty_days": r["empty_days"],
        }

    log.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Experiment logged to {LOG_FILE}")


# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM VARIANT REGISTRATION
# ═══════════════════════════════════════════════════════════════════════

def register_variant(name, fn):
    """Register a custom variant function for backtesting."""
    VARIANTS[name] = fn


def make_parameterized_under(min_hr=50, min_tier="C", under_bonus=0.30, gap_weight=0.4,
                              hr_weight=0.3, min_ens=0.0, block_combos=True, n_legs=3):
    """Factory for parameterized UNDER strategies."""
    tier_order = ["S", "A", "B", "C", "D", "F"]
    min_tier_idx = tier_order.index(min_tier) if min_tier in tier_order else 5

    def variant(props):
        pool = []
        for p in props:
            if p.get("direction", "OVER").upper() != "UNDER":
                continue
            tier = p.get("tier", "F")
            tier_idx = tier_order.index(tier) if tier in tier_order else 5
            if tier_idx > min_tier_idx:
                continue
            if p.get("l10_hit_rate", 50) < min_hr:
                continue
            if block_combos and p.get("stat", "") in COMBO_STATS:
                continue
            ens = p.get("ensemble_prob", 0.5)
            if ens < min_ens:
                continue
            abs_gap = p.get("abs_gap", 0)
            hr = p.get("l10_hit_rate", 50)
            p["_score"] = abs_gap * gap_weight + (hr / 100) * hr_weight + under_bonus
            pool.append(p)
        return select_diverse(pool, n=n_legs)

    return variant


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="NBA Prop Backtesting Harness v1")
    parser.add_argument("--all", action="store_true", help="Run full comparison on all graded dates")
    parser.add_argument("--date", type=str, help="Test a specific date (YYYY-MM-DD)")
    parser.add_argument("--experiment", type=str, help="Run a named experiment")
    parser.add_argument("--list", action="store_true", help="List available experiments and variants")
    parser.add_argument("--summary", action="store_true", help="Summary of all graded data")
    parser.add_argument("--variant", type=str, help="Run a single variant (use with --date or --all)")
    parser.add_argument("--legs", type=int, default=3, help="Number of parlay legs (default 3)")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable Variants:")
        for name in sorted(VARIANTS.keys()):
            print(f"  - {name}")
        print("\nAvailable Experiments:")
        for name, exp in EXPERIMENTS.items():
            print(f"  - {name}: {exp['description']}")
            print(f"    Variants: {', '.join(exp['variants'])}")
        return

    # Discover and load data
    all_dates = discover_graded_dates()
    if not all_dates:
        print("No graded data found!")
        return

    if args.date:
        all_dates = [(d, p) for d, p in all_dates if d == args.date]
        if not all_dates:
            print(f"No graded data for {args.date}")
            return

    print(f"Loading graded data from {len(all_dates)} dates...")
    dates_data = []
    for date, path in all_dates:
        props = load_graded_props(date, path)
        if props:
            dates_data.append((date, props))
            print(f"  {date}: {len(props)} graded props")

    if not dates_data:
        print("No usable graded data!")
        return

    if args.summary:
        print_summary(dates_data)
        return

    # Determine which variants to run
    if args.experiment:
        if args.experiment not in EXPERIMENTS:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return
        exp = EXPERIMENTS[args.experiment]
        variant_names = exp["variants"]
        experiment_name = args.experiment
    elif args.variant:
        if args.variant not in VARIANTS:
            print(f"Unknown variant: {args.variant}")
            return
        variant_names = [args.variant]
        experiment_name = f"single_{args.variant}"
    else:
        # Default: full comparison
        variant_names = list(VARIANTS.keys())
        experiment_name = "full_comparison"

    # Run all variants
    variant_results = {}
    for name in variant_names:
        fn = VARIANTS[name]
        print(f"  Running {name}...")
        variant_results[name] = evaluate_variant(fn, dates_data)

    # Report
    print_comparison(experiment_name, variant_results)
    save_experiment_log(experiment_name, variant_results)


if __name__ == "__main__":
    main()

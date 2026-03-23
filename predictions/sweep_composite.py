#!/usr/bin/env python3
"""
Parameterized Sweep on composite_v2 — find optimal weights for 3-leg parlays.

Sweeps across:
  - ML weight (ensemble_prob contribution)
  - Gap weight
  - HR weight
  - UNDER bonus strength
  - Miss streak bonus
  - Min ensemble threshold
  - Min tier
  - Direction (UNDER-only vs mixed)
  - Game diversity enforcement
  - Correlation penalty (same-game avoidance)

Uses the backtest_harness data loader. Outputs ranked results.
"""

import json
import copy
import sys
from pathlib import Path
from collections import defaultdict
from itertools import product as cartesian

sys.path.insert(0, str(Path(__file__).parent))
from backtest_harness import (
    discover_graded_dates, load_graded_props, get_team,
    COMBO_STATS, BOOST_STATS
)

# ═══════════════════════════════════════════════════════════════════════
#  PARAMETERIZED COMPOSITE BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_composite(props, params):
    """
    Parameterized composite strategy.

    params dict:
        ml_weight:       float  — weight on ensemble_prob (0.0 - 0.5)
        gap_weight:      float  — weight on abs_gap (0.0 - 0.3)
        hr_weight:       float  — weight on L10 HR (0.0 - 0.3)
        under_bonus:     float  — flat bonus for UNDER direction (0.0 - 0.3)
        miss_bonus:      float  — bonus per miss_count for UNDER (0.0 - 0.02)
        min_ens:         float  — minimum ensemble_prob threshold (0.0 - 0.65)
        min_tier:        str    — minimum tier to allow (S, A, B, C)
        under_only:      bool   — restrict to UNDER only
        block_combos:    bool   — block combo stats (pra, pr, pa, ra)
        min_mins:        int    — minimum mins_30plus_pct
        min_hr:          int    — minimum L10 HR
        blk_stl_bonus:   float  — extra bonus for BLK/STL UNDER (0.0 - 0.15)
        cold_bonus:      float  — bonus for COLD streak + UNDER (0.0 - 0.15)
        hot_penalty:     float  — penalty for HOT streak (0.0 - 0.15)
        n_legs:          int    — number of parlay legs
        spread_penalty:  float  — penalty for high spread (blowout risk) (0.0 - 0.1)
    """
    tier_order = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4, "F": 5}
    min_tier_idx = tier_order.get(params.get("min_tier", "C"), 3)

    pool = []
    for p in props:
        p = copy.deepcopy(p)
        tier = p.get("tier", "F")
        tier_idx = tier_order.get(tier, 5)
        if tier_idx > min_tier_idx:
            continue

        direction = p.get("direction", "OVER").upper()
        if params.get("under_only", False) and direction != "UNDER":
            continue

        if params.get("block_combos", True) and p.get("stat", "") in COMBO_STATS:
            continue

        if p.get("mins_30plus_pct", 70) < params.get("min_mins", 40):
            continue

        hr = p.get("l10_hit_rate", 50)
        if hr < params.get("min_hr", 0):
            continue

        ens = p.get("ensemble_prob", 0.5)
        if ens < params.get("min_ens", 0.0):
            continue

        abs_gap = p.get("abs_gap", 0)
        miss_count = p.get("l10_miss_count", 5)
        streak = p.get("streak_status", p.get("streak", "NEUTRAL"))
        stat = p.get("stat", "")
        spread = abs(p.get("spread", 0) or 0)

        # Score
        score = 0.0
        score += ens * params.get("ml_weight", 0.25)
        score += min(abs_gap / 8.0, 0.25) * (params.get("gap_weight", 0.20) / 0.20) if params.get("gap_weight", 0.20) > 0 else 0
        score += (hr / 100) * params.get("hr_weight", 0.15)

        if direction == "UNDER":
            score += params.get("under_bonus", 0.15)
            score += miss_count * params.get("miss_bonus", 0.008)
            if stat in BOOST_STATS:
                score += params.get("blk_stl_bonus", 0.05)
            if streak == "COLD":
                score += params.get("cold_bonus", 0.05)

        if streak == "HOT":
            score -= params.get("hot_penalty", 0.05)

        if spread >= 12 and direction == "OVER":
            score -= params.get("spread_penalty", 0.05)

        p["_score"] = score
        pool.append(p)

    # Select with diversity
    n_legs = params.get("n_legs", 3)
    pool_sorted = sorted(pool, key=lambda p: p.get("_score", 0), reverse=True)
    selected = []
    used_games = set()
    used_teams = set()
    for p in pool_sorted:
        if len(selected) >= n_legs:
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


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════

def evaluate(dates_data, params):
    """Run parameterized composite across all dates, return stats."""
    total_parlays = 0
    cashed = 0
    total_legs = 0
    legs_hit = 0
    empty = 0
    per_date = {}

    for date, props in dates_data:
        legs = build_composite(props, params)
        if not legs or len(legs) < params.get("n_legs", 3):
            empty += 1
            per_date[date] = {"cashed": False, "hits": 0, "legs": 0}
            continue

        hits = sum(1 for l in legs if l.get("_hit", False))
        is_cashed = hits == len(legs)
        total_parlays += 1
        if is_cashed:
            cashed += 1
        total_legs += len(legs)
        legs_hit += hits
        per_date[date] = {
            "cashed": is_cashed,
            "hits": hits,
            "legs": len(legs),
            "picks": [(l["player"], l.get("stat","?"), l.get("direction","?"), l.get("line",0), l.get("_hit",False)) for l in legs],
        }

    cash_rate = cashed / total_parlays * 100 if total_parlays > 0 else 0
    leg_rate = legs_hit / total_legs * 100 if total_legs > 0 else 0

    return {
        "cash_rate": cash_rate,
        "leg_rate": leg_rate,
        "total_parlays": total_parlays,
        "cashed": cashed,
        "total_legs": total_legs,
        "legs_hit": legs_hit,
        "empty": empty,
        "per_date": per_date,
    }


# ═══════════════════════════════════════════════════════════════════════
#  SWEEP DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

def sweep_core_weights(dates_data):
    """Sweep ML weight, gap weight, HR weight, UNDER bonus."""
    print("\n" + "="*80)
    print("  SWEEP 1: Core Weight Optimization")
    print("="*80)

    configs = []
    for ml_w in [0.10, 0.20, 0.30, 0.40]:
        for gap_w in [0.05, 0.15, 0.25]:
            for hr_w in [0.05, 0.10, 0.20]:
                for under_b in [0.10, 0.20, 0.30]:
                    configs.append({
                        "ml_weight": ml_w,
                        "gap_weight": gap_w,
                        "hr_weight": hr_w,
                        "under_bonus": under_b,
                        "miss_bonus": 0.008,
                        "min_ens": 0.0,
                        "min_tier": "C",
                        "under_only": False,
                        "block_combos": True,
                        "min_mins": 40,
                        "min_hr": 0,
                        "blk_stl_bonus": 0.05,
                        "cold_bonus": 0.05,
                        "hot_penalty": 0.05,
                        "spread_penalty": 0.05,
                        "n_legs": 3,
                    })

    return run_sweep(dates_data, configs, "Core Weights")


def sweep_under_strictness(dates_data):
    """Sweep UNDER-only vs mixed, min HR, min ensemble."""
    print("\n" + "="*80)
    print("  SWEEP 2: UNDER Strictness")
    print("="*80)

    configs = []
    for under_only in [True, False]:
        for min_hr in [0, 30, 50, 60]:
            for min_ens in [0.0, 0.50, 0.55, 0.60]:
                for min_tier in ["B", "C"]:
                    configs.append({
                        "ml_weight": 0.25,
                        "gap_weight": 0.15,
                        "hr_weight": 0.15,
                        "under_bonus": 0.20,
                        "miss_bonus": 0.008,
                        "min_ens": min_ens,
                        "min_tier": min_tier,
                        "under_only": under_only,
                        "block_combos": True,
                        "min_mins": 40,
                        "min_hr": min_hr,
                        "blk_stl_bonus": 0.05,
                        "cold_bonus": 0.05,
                        "hot_penalty": 0.05,
                        "spread_penalty": 0.05,
                        "n_legs": 3,
                    })

    return run_sweep(dates_data, configs, "UNDER Strictness")


def sweep_streaks_and_stats(dates_data):
    """Sweep streak bonuses, BLK/STL bonus, miss bonus."""
    print("\n" + "="*80)
    print("  SWEEP 3: Streaks & Stat Bonuses")
    print("="*80)

    configs = []
    for cold_b in [0.0, 0.05, 0.10, 0.15]:
        for hot_p in [0.0, 0.05, 0.10, 0.15]:
            for blk_b in [0.0, 0.05, 0.10, 0.15]:
                for miss_b in [0.0, 0.005, 0.01, 0.015]:
                    configs.append({
                        "ml_weight": 0.25,
                        "gap_weight": 0.15,
                        "hr_weight": 0.15,
                        "under_bonus": 0.20,
                        "miss_bonus": miss_b,
                        "min_ens": 0.0,
                        "min_tier": "C",
                        "under_only": False,
                        "block_combos": True,
                        "min_mins": 40,
                        "min_hr": 0,
                        "blk_stl_bonus": blk_b,
                        "cold_bonus": cold_b,
                        "hot_penalty": hot_p,
                        "spread_penalty": 0.05,
                        "n_legs": 3,
                    })

    return run_sweep(dates_data, configs, "Streaks & Stat Bonuses")


def sweep_leg_count(dates_data):
    """Sweep 2-leg through 6-leg parlays with best params."""
    print("\n" + "="*80)
    print("  SWEEP 4: Leg Count (2-6)")
    print("="*80)

    configs = []
    for n_legs in [2, 3, 4, 5, 6]:
        configs.append({
            "ml_weight": 0.25,
            "gap_weight": 0.15,
            "hr_weight": 0.15,
            "under_bonus": 0.20,
            "miss_bonus": 0.008,
            "min_ens": 0.0,
            "min_tier": "C",
            "under_only": False,
            "block_combos": True,
            "min_mins": 40,
            "min_hr": 0,
            "blk_stl_bonus": 0.05,
            "cold_bonus": 0.05,
            "hot_penalty": 0.05,
            "spread_penalty": 0.05,
            "n_legs": n_legs,
        })

    return run_sweep(dates_data, configs, "Leg Count")


def run_sweep(dates_data, configs, sweep_name):
    """Run a sweep and return top results."""
    results = []
    for i, params in enumerate(configs):
        r = evaluate(dates_data, params)
        r["params"] = params
        results.append(r)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(configs)}] configs evaluated...")

    print(f"  [{len(configs)}/{len(configs)}] configs evaluated")

    # Filter: must build at least 5 parlays (not too selective)
    viable = [r for r in results if r["total_parlays"] >= 5]
    if not viable:
        viable = [r for r in results if r["total_parlays"] >= 3]
    if not viable:
        viable = [r for r in results if r["total_parlays"] >= 1]

    # Sort by cash rate, then leg rate
    viable.sort(key=lambda r: (r["cash_rate"], r["leg_rate"]), reverse=True)

    # Print top 15
    print(f"\n  TOP 15 — {sweep_name} (from {len(configs)} configs, {len(viable)} viable)")
    print(f"  {'#':>3} {'Cash%':>7} {'Leg%':>7} {'Built':>6} {'Cashed':>7} {'Empty':>6} | Key Params")
    print(f"  {'─'*3} {'─'*7} {'─'*7} {'─'*6} {'─'*7} {'─'*6} {'─'*50}")

    for i, r in enumerate(viable[:15]):
        p = r["params"]
        # Format key params that differ from defaults
        key_parts = []
        if "ml_weight" in p: key_parts.append(f"ml={p['ml_weight']:.2f}")
        if "gap_weight" in p: key_parts.append(f"gap={p['gap_weight']:.2f}")
        if "hr_weight" in p: key_parts.append(f"hr={p['hr_weight']:.2f}")
        if "under_bonus" in p: key_parts.append(f"ub={p['under_bonus']:.2f}")
        if p.get("under_only"): key_parts.append("UNDER_ONLY")
        if p.get("min_hr", 0) > 0: key_parts.append(f"minHR={p['min_hr']}")
        if p.get("min_ens", 0) > 0: key_parts.append(f"minEns={p['min_ens']:.2f}")
        if "cold_bonus" in p and p["cold_bonus"] != 0.05: key_parts.append(f"cold={p['cold_bonus']:.2f}")
        if "hot_penalty" in p and p["hot_penalty"] != 0.05: key_parts.append(f"hot={p['hot_penalty']:.2f}")
        if "blk_stl_bonus" in p and p["blk_stl_bonus"] != 0.05: key_parts.append(f"blk={p['blk_stl_bonus']:.2f}")
        if "miss_bonus" in p and p["miss_bonus"] != 0.008: key_parts.append(f"miss={p['miss_bonus']:.3f}")
        if "n_legs" in p and p["n_legs"] != 3: key_parts.append(f"legs={p['n_legs']}")
        if p.get("min_tier", "C") != "C": key_parts.append(f"tier>={p['min_tier']}")

        param_str = " | ".join(key_parts) if key_parts else "(default)"
        print(f"  {i+1:>3} {r['cash_rate']:>6.1f}% {r['leg_rate']:>6.1f}% {r['total_parlays']:>6} {r['cashed']:>7} {r['empty']:>6} | {param_str}")

    return viable[:15]


# ═══════════════════════════════════════════════════════════════════════
#  FINAL OPTIMIZED VARIANT
# ═══════════════════════════════════════════════════════════════════════

def print_best_config(all_tops):
    """Identify and print the single best configuration across all sweeps."""
    flat = []
    for tops in all_tops:
        flat.extend(tops)

    # Filter: >= 5 parlays built
    viable = [r for r in flat if r["total_parlays"] >= 5]
    if not viable:
        viable = flat

    viable.sort(key=lambda r: (r["cash_rate"], r["leg_rate"]), reverse=True)

    if not viable:
        print("\nNo viable configurations found!")
        return

    best = viable[0]
    p = best["params"]

    print(f"\n{'='*80}")
    print(f"  OPTIMAL CONFIGURATION")
    print(f"{'='*80}")
    print(f"\n  Cash Rate:   {best['cash_rate']:.1f}%")
    print(f"  Leg HR:      {best['leg_rate']:.1f}%")
    print(f"  Parlays:     {best['total_parlays']} built, {best['cashed']} cashed")
    print(f"  Empty Days:  {best['empty']}")
    print(f"\n  Parameters:")
    for k, v in sorted(p.items()):
        print(f"    {k:<20} = {v}")

    # Per-date breakdown
    print(f"\n  Per-Date:")
    for date in sorted(best["per_date"].keys()):
        d = best["per_date"][date]
        if d["legs"] == 0:
            print(f"    {date}: NO BUILD")
        else:
            status = "CASHED" if d["cashed"] else f"{d['hits']}/{d['legs']} legs"
            picks = ""
            if d.get("picks"):
                picks = " | " + ", ".join(
                    f"{'✓' if hit else '✗'} {player} {dir} {stat} {line}"
                    for player, stat, dir, line, hit in d["picks"]
                )
            print(f"    {date}: {status}{picks}")

    # Output as JSON for integration
    config_path = Path(__file__).parent / "optimal_composite_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "params": p,
            "backtest_results": {
                "cash_rate": best["cash_rate"],
                "leg_rate": best["leg_rate"],
                "total_parlays": best["total_parlays"],
                "cashed": best["cashed"],
                "empty_days": best["empty"],
                "dates_tested": len(best["per_date"]),
            },
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\n  Config saved to {config_path}")

    return p


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Loading graded data...")
    all_dates = discover_graded_dates()
    dates_data = []
    for date, path in all_dates:
        props = load_graded_props(date, path)
        if props:
            dates_data.append((date, props))
            print(f"  {date}: {len(props)} props")

    if not dates_data:
        print("No graded data!")
        return

    print(f"\n  Total: {sum(len(p) for _, p in dates_data)} props across {len(dates_data)} days")

    all_tops = []

    # Sweep 1: Core weights
    tops = sweep_core_weights(dates_data)
    all_tops.append(tops)

    # Sweep 2: UNDER strictness
    tops = sweep_under_strictness(dates_data)
    all_tops.append(tops)

    # Sweep 3: Streaks & bonuses
    tops = sweep_streaks_and_stats(dates_data)
    all_tops.append(tops)

    # Sweep 4: Leg count
    tops = sweep_leg_count(dates_data)
    all_tops.append(tops)

    # Find optimal
    best_params = print_best_config(all_tops)


if __name__ == "__main__":
    main()

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
import random
import math
from pathlib import Path
from collections import defaultdict
from itertools import product as cartesian
from statistics import stdev, median

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

        # ── Regression margin signal ──
        reg_margin = p.get("reg_margin")
        reg_margin_weight = params.get("reg_margin_weight", 0.0)
        reg_margin_min = params.get("reg_margin_min", 0.0)
        if reg_margin is not None and reg_margin_weight > 0:
            abs_rm = abs(reg_margin)
            # Direction confirmation: positive margin = OVER expected, negative = UNDER
            reg_confirms = (reg_margin < 0 and direction == "UNDER") or (reg_margin > 0 and direction == "OVER")
            if reg_confirms:
                score += min(abs_rm / 5.0, 0.25) * reg_margin_weight
            else:
                score -= min(abs_rm / 5.0, 0.15) * reg_margin_weight * 0.5
        if reg_margin_min > 0 and reg_margin is not None:
            abs_rm = abs(reg_margin)
            if abs_rm < reg_margin_min:
                continue  # hard filter

        # ── Consistency (low std = predictable) ──
        max_std = params.get("max_std", 999)
        consistency_weight = params.get("consistency_weight", 0.0)
        l10_values = p.get("l10_values") or []
        if len(l10_values) >= 3:
            try:
                l10_std = stdev(l10_values)
            except Exception:
                l10_std = 99
        else:
            l10_std = 99
        if l10_std > max_std:
            continue  # hard filter
        if consistency_weight > 0 and l10_std < 99:
            # Lower std = higher bonus (normalized: std of 0 -> full bonus, std of 10 -> 0)
            score += max(0, (1.0 - l10_std / 10.0)) * consistency_weight

        # ── Fatigue signals ──
        fatigue_weight = params.get("fatigue_weight", 0.0)
        if fatigue_weight > 0:
            games_in_7 = p.get("games_in_7", 0) or 0
            travel_miles = p.get("travel_miles_7day", 0) or 0
            rest_days = p.get("rest_days", 2) or 2
            # Fatigue penalty: more games + more travel + less rest = bad for OVER
            fatigue_score = 0.0
            if games_in_7 >= 4:
                fatigue_score -= 0.04
            elif games_in_7 >= 3:
                fatigue_score -= 0.02
            if travel_miles > 3000:
                fatigue_score -= 0.04
            elif travel_miles > 1500:
                fatigue_score -= 0.02
            if rest_days == 0:
                fatigue_score -= 0.03
            elif rest_days >= 2:
                fatigue_score += 0.02
            # Fatigue hurts OVER, helps UNDER
            if direction == "OVER":
                score += fatigue_score * fatigue_weight
            else:
                score -= fatigue_score * fatigue_weight  # inverted: fatigue helps UNDER

        # ── Multi-model consensus bonus ──
        consensus_bonus = params.get("consensus_bonus", 0.0)
        consensus_min = params.get("consensus_min_models", 0)
        if consensus_bonus > 0 or consensus_min > 0:
            models_agree = 0
            ens_thresh = params.get("consensus_ens_thresh", 0.55)
            sim_thresh = params.get("consensus_sim_thresh", 0.55)
            reg_thresh = params.get("consensus_reg_thresh", 1.5)

            if ens >= ens_thresh:
                models_agree += 1
            sim_p = p.get("sim_prob")
            if sim_p is not None and sim_p >= sim_thresh:
                models_agree += 1
            if reg_margin is not None:
                reg_confirms_dir = (reg_margin < 0 and direction == "UNDER") or (reg_margin > 0 and direction == "OVER")
                if reg_confirms_dir and abs(reg_margin) >= reg_thresh:
                    models_agree += 1
            # Calibrated prob as 4th signal
            cal = p.get("xgb_prob_calibrated")
            if cal is not None and cal >= ens_thresh:
                models_agree += 1

            if consensus_min > 0 and models_agree < consensus_min:
                continue  # hard filter
            score += models_agree * consensus_bonus

        # ── Opponent matchup delta ──
        opp_weight = params.get("opp_matchup_weight", 0.0)
        if opp_weight > 0:
            opp_delta = p.get("opp_matchup_delta", 0) or 0
            opp_allowed = p.get("opp_stat_allowed_vs_league_avg", 0) or 0
            # Positive opp_delta = opponent gives up more of this stat = good for OVER
            if direction == "OVER":
                score += min(opp_delta * 0.01, 0.10) * opp_weight
                score += min(opp_allowed * 0.5, 0.10) * opp_weight
            else:
                score -= min(opp_delta * 0.01, 0.10) * opp_weight * 0.5

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
        # New sweep params display
        if p.get("reg_margin_weight", 0) > 0: key_parts.append(f"regW={p['reg_margin_weight']:.2f}")
        if p.get("reg_margin_min", 0) > 0: key_parts.append(f"regMin={p['reg_margin_min']:.1f}")
        if p.get("max_std", 999) < 999: key_parts.append(f"maxStd={p['max_std']:.1f}")
        if p.get("consistency_weight", 0) > 0: key_parts.append(f"consW={p['consistency_weight']:.2f}")
        if p.get("fatigue_weight", 0) > 0: key_parts.append(f"fatW={p['fatigue_weight']:.2f}")
        if p.get("consensus_bonus", 0) > 0: key_parts.append(f"consBon={p['consensus_bonus']:.2f}")
        if p.get("consensus_min_models", 0) > 0: key_parts.append(f"consMin={p['consensus_min_models']}")
        if p.get("opp_matchup_weight", 0) > 0: key_parts.append(f"oppW={p['opp_matchup_weight']:.2f}")

        param_str = " | ".join(key_parts) if key_parts else "(default)"
        print(f"  {i+1:>3} {r['cash_rate']:>6.1f}% {r['leg_rate']:>6.1f}% {r['total_parlays']:>6} {r['cashed']:>7} {r['empty']:>6} | {param_str}")

    return viable[:15]


# ═══════════════════════════════════════════════════════════════════════
#  SWEEP 5-10: ADVANCED SIGNAL SWEEPS
# ═══════════════════════════════════════════════════════════════════════

# Default base params shared by new sweeps
_BASE_PARAMS = {
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
    "n_legs": 3,
}


def _make_config(**overrides):
    """Create config from base with overrides."""
    cfg = dict(_BASE_PARAMS)
    cfg.update(overrides)
    return cfg


def sweep_regression_margin(dates_data):
    """Sweep reg_margin as hard filter vs soft bonus at various weights."""
    print("\n" + "=" * 80)
    print("  SWEEP 5: Regression Margin Signal")
    print("=" * 80)

    configs = []

    # A) reg_margin as soft bonus weight (no hard filter)
    for reg_w in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        for ml_w in [0.15, 0.25, 0.35]:
            for under_b in [0.15, 0.20, 0.30]:
                configs.append(_make_config(
                    reg_margin_weight=reg_w,
                    reg_margin_min=0.0,
                    ml_weight=ml_w,
                    under_bonus=under_b,
                ))

    # B) reg_margin as hard filter (|margin| >= threshold) + soft bonus
    for reg_min in [1.5, 2.0, 2.5, 3.0]:
        for reg_w in [0.10, 0.20, 0.30]:
            for ml_w in [0.15, 0.25]:
                configs.append(_make_config(
                    reg_margin_weight=reg_w,
                    reg_margin_min=reg_min,
                    ml_weight=ml_w,
                ))

    # C) reg_margin only (high weight, low ML dependence)
    for reg_w in [0.40, 0.50, 0.60]:
        for reg_min in [0.0, 1.0, 2.0]:
            configs.append(_make_config(
                reg_margin_weight=reg_w,
                reg_margin_min=reg_min,
                ml_weight=0.10,
                gap_weight=0.05,
            ))

    return run_sweep(dates_data, configs, "Regression Margin")


def sweep_multi_model_consensus(dates_data):
    """Sweep requiring N models to agree with different thresholds."""
    print("\n" + "=" * 80)
    print("  SWEEP 6: Multi-Model Consensus")
    print("=" * 80)

    configs = []

    # Sweep consensus count + thresholds
    for min_models in [0, 2, 3, 4]:
        for ens_thresh in [0.50, 0.55, 0.60]:
            for sim_thresh in [0.50, 0.55, 0.60]:
                for reg_thresh in [1.0, 1.5, 2.0, 3.0]:
                    for cons_bonus in [0.0, 0.03, 0.06, 0.10]:
                        configs.append(_make_config(
                            consensus_min_models=min_models,
                            consensus_bonus=cons_bonus,
                            consensus_ens_thresh=ens_thresh,
                            consensus_sim_thresh=sim_thresh,
                            consensus_reg_thresh=reg_thresh,
                        ))

    # Consensus + UNDER-only
    for min_models in [2, 3]:
        for ens_thresh in [0.52, 0.55, 0.58]:
            for cons_bonus in [0.05, 0.10]:
                configs.append(_make_config(
                    consensus_min_models=min_models,
                    consensus_bonus=cons_bonus,
                    consensus_ens_thresh=ens_thresh,
                    consensus_sim_thresh=0.55,
                    consensus_reg_thresh=1.5,
                    under_only=True,
                ))

    return run_sweep(dates_data, configs, "Multi-Model Consensus")


def sweep_consistency(dates_data):
    """Sweep l10_std (from l10_values) as predictability signal."""
    print("\n" + "=" * 80)
    print("  SWEEP 7: Consistency (Low Variance)")
    print("=" * 80)

    configs = []

    # A) max_std as hard filter
    for max_std in [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 999]:
        for cons_w in [0.0, 0.05, 0.10, 0.15, 0.20]:
            for under_b in [0.15, 0.20, 0.30]:
                configs.append(_make_config(
                    max_std=max_std,
                    consistency_weight=cons_w,
                    under_bonus=under_b,
                ))

    # B) consistency weight with various ML weights
    for cons_w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        for ml_w in [0.10, 0.20, 0.30]:
            for max_std in [4.0, 6.0, 999]:
                configs.append(_make_config(
                    consistency_weight=cons_w,
                    ml_weight=ml_w,
                    max_std=max_std,
                ))

    # C) consistency + UNDER only (low variance + UNDER = safest bets)
    for max_std in [3.0, 4.0, 5.0, 6.0]:
        for cons_w in [0.10, 0.20]:
            configs.append(_make_config(
                max_std=max_std,
                consistency_weight=cons_w,
                under_only=True,
                under_bonus=0.25,
            ))

    return run_sweep(dates_data, configs, "Consistency (Low Variance)")


def sweep_fatigue(dates_data):
    """Sweep travel/rest/schedule fatigue signals."""
    print("\n" + "=" * 80)
    print("  SWEEP 8: Fatigue & Travel Signals")
    print("=" * 80)

    configs = []

    # A) fatigue_weight sweep with different base configs
    for fat_w in [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0]:
        for ml_w in [0.15, 0.25, 0.35]:
            for under_b in [0.15, 0.20, 0.30]:
                for under_only in [True, False]:
                    configs.append(_make_config(
                        fatigue_weight=fat_w,
                        ml_weight=ml_w,
                        under_bonus=under_b,
                        under_only=under_only,
                    ))

    # B) fatigue + opponent matchup (contextual signals together)
    for fat_w in [0.50, 1.0, 1.5]:
        for opp_w in [0.25, 0.50, 1.0]:
            for ml_w in [0.15, 0.25]:
                configs.append(_make_config(
                    fatigue_weight=fat_w,
                    opp_matchup_weight=opp_w,
                    ml_weight=ml_w,
                ))

    return run_sweep(dates_data, configs, "Fatigue & Travel")


def sweep_mega_composite(dates_data):
    """
    Mega sweep across ALL dimensions simultaneously.
    Uses Latin Hypercube Sampling to cover the space with ~2000 configs
    instead of full cartesian (which would be millions).
    """
    print("\n" + "=" * 80)
    print("  SWEEP 9: MEGA COMPOSITE (Latin Hypercube ~2000 configs)")
    print("=" * 80)

    random.seed(42)  # reproducible

    # Define parameter ranges: (name, [values]) or (name, low, high, type)
    param_ranges = {
        "ml_weight":            (0.05, 0.50),
        "gap_weight":           (0.0,  0.30),
        "hr_weight":            (0.0,  0.25),
        "under_bonus":          (0.0,  0.40),
        "miss_bonus":           (0.0,  0.020),
        "min_ens":              (0.0,  0.65),
        "min_hr":               (0,    70),
        "blk_stl_bonus":        (0.0,  0.20),
        "cold_bonus":           (0.0,  0.20),
        "hot_penalty":          (0.0,  0.20),
        "spread_penalty":       (0.0,  0.15),
        "reg_margin_weight":    (0.0,  0.50),
        "reg_margin_min":       (0.0,  3.0),
        "max_std":              (2.0,  999.0),
        "consistency_weight":   (0.0,  0.30),
        "fatigue_weight":       (0.0,  2.0),
        "consensus_bonus":      (0.0,  0.12),
        "consensus_min_models": (0,    4),
        "opp_matchup_weight":   (0.0,  1.0),
    }

    # Categorical params
    cat_params = {
        "min_tier":    ["S", "A", "B", "C"],
        "under_only":  [True, False],
        "block_combos": [True, False],
    }

    n_samples = 2000
    param_names = list(param_ranges.keys())
    n_dims = len(param_names)

    # Latin Hypercube: divide each dimension into n_samples equal bins
    configs = []
    for i in range(n_samples):
        cfg = dict(_BASE_PARAMS)

        # Continuous/int params via LHS
        for j, name in enumerate(param_names):
            lo, hi = param_ranges[name]
            # LHS: sample from bin i with random offset
            bin_start = i / n_samples
            bin_end = (i + 1) / n_samples
            u = random.uniform(bin_start, bin_end)
            val = lo + u * (hi - lo)
            # Round integers
            if name in ("min_hr", "consensus_min_models"):
                val = int(round(val))
            elif name == "max_std" and val > 100:
                val = 999  # effectively no filter
            else:
                val = round(val, 4)
            cfg[name] = val

        # Categorical params: random choice
        for name, options in cat_params.items():
            cfg[name] = random.choice(options)

        # Fixed
        cfg["n_legs"] = 3
        cfg["min_mins"] = random.choice([30, 40, 50])
        cfg["consensus_ens_thresh"] = random.choice([0.50, 0.55, 0.60])
        cfg["consensus_sim_thresh"] = random.choice([0.50, 0.55, 0.60])
        cfg["consensus_reg_thresh"] = random.choice([1.0, 1.5, 2.0])

        configs.append(cfg)

    # Shuffle to break LHS correlation across dimensions
    for name in param_names:
        vals = [c[name] for c in configs]
        random.shuffle(vals)
        for idx, c in enumerate(configs):
            c[name] = vals[idx]

    return run_sweep(dates_data, configs, "MEGA COMPOSITE (LHS)")


def sweep_all_fast(dates_data):
    """
    Focused sweep of ~500 configs across the most impactful dimensions.
    Targets: reg_margin weight, consistency threshold, model consensus,
    under bonus, ML weight. Uses best ranges from prior sweeps.
    """
    print("\n" + "=" * 80)
    print("  SWEEP 10: ALL-FAST (~500 focused configs)")
    print("=" * 80)

    configs = []

    # Focused grid on most impactful params
    ml_weights = [0.15, 0.25, 0.35]
    under_bonuses = [0.15, 0.25, 0.35]
    reg_weights = [0.0, 0.10, 0.20, 0.35]
    consistency_weights = [0.0, 0.10, 0.20]
    max_stds = [4.0, 6.0, 999]
    consensus_counts = [0, 2, 3]
    consensus_bonuses = [0.0, 0.05]

    for ml_w in ml_weights:
        for ub in under_bonuses:
            for reg_w in reg_weights:
                for cons_w in consistency_weights:
                    for max_std in max_stds:
                        for cons_min in consensus_counts:
                            for cons_bon in consensus_bonuses:
                                # Skip redundant combos where consensus params
                                # don't matter
                                if cons_min == 0 and cons_bon > 0:
                                    continue
                                configs.append(_make_config(
                                    ml_weight=ml_w,
                                    under_bonus=ub,
                                    reg_margin_weight=reg_w,
                                    consistency_weight=cons_w,
                                    max_std=max_std,
                                    consensus_min_models=cons_min,
                                    consensus_bonus=cons_bon,
                                    consensus_ens_thresh=0.55,
                                    consensus_sim_thresh=0.55,
                                    consensus_reg_thresh=1.5,
                                ))

    # Add UNDER-only variants of top combos
    under_only_configs = []
    for ml_w in [0.15, 0.25]:
        for ub in [0.20, 0.30]:
            for reg_w in [0.0, 0.15, 0.30]:
                for cons_w in [0.0, 0.15]:
                    for max_std in [5.0, 999]:
                        under_only_configs.append(_make_config(
                            ml_weight=ml_w,
                            under_bonus=ub,
                            under_only=True,
                            reg_margin_weight=reg_w,
                            consistency_weight=cons_w,
                            max_std=max_std,
                        ))

    configs.extend(under_only_configs)

    # Add fatigue combos with best other params
    for fat_w in [0.5, 1.0, 1.5]:
        for ml_w in [0.20, 0.30]:
            for ub in [0.20, 0.30]:
                for reg_w in [0.0, 0.20]:
                    configs.append(_make_config(
                        fatigue_weight=fat_w,
                        ml_weight=ml_w,
                        under_bonus=ub,
                        reg_margin_weight=reg_w,
                    ))

    print(f"  Generated {len(configs)} focused configs")
    return run_sweep(dates_data, configs, "ALL-FAST (Focused)")


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

    # Sweep 5: Regression margin
    tops = sweep_regression_margin(dates_data)
    all_tops.append(tops)

    # Sweep 6: Multi-model consensus
    tops = sweep_multi_model_consensus(dates_data)
    all_tops.append(tops)

    # Sweep 7: Consistency (low variance)
    tops = sweep_consistency(dates_data)
    all_tops.append(tops)

    # Sweep 8: Fatigue & travel
    tops = sweep_fatigue(dates_data)
    all_tops.append(tops)

    # Sweep 9: Mega composite (LHS ~2000 configs)
    tops = sweep_mega_composite(dates_data)
    all_tops.append(tops)

    # Sweep 10: All-fast focused (~500 configs)
    tops = sweep_all_fast(dates_data)
    all_tops.append(tops)

    # Find optimal across ALL sweeps
    best_params = print_best_config(all_tops)


if __name__ == "__main__":
    main()

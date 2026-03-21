#!/usr/bin/env python3
"""
NBA Parlay Mass Backtest & Self-Optimization Engine
Generates 3,000-10,000 parlays per game day, grades them,
iterates filters to maximize parlay win rate.
"""

import json
import os
import sys
import time
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
from datetime import datetime

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(BASE, 'cache')
OUTPUT = os.path.join(BASE, 'backtesting')
os.makedirs(OUTPUT, exist_ok=True)

TIER_ORDER = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
BANNED_STATS = {'fg3m', '3pm', 'FG3M'}  # Permanently banned from parlays

# ─────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────

def load_sgo_data():
    """Load SGO backfill (200K records, 462 dates, real player names)."""
    path = os.path.join(CACHE, 'sgo_backfill_training_data.json')
    if not os.path.exists(path):
        print("ERROR: SGO backfill not found. Run backfill_sgo_box_scores.py first.")
        return []
    with open(path) as f:
        data = json.load(f)
    print(f"  SGO backfill: {len(data):,} records loaded")
    return data


def load_backfill_data():
    """Load nba_api backfill (242K records, 120 dates, hashed player IDs)."""
    path = os.path.join(CACHE, 'backfill_training_data.json')
    if not os.path.exists(path):
        print("ERROR: Backfill not found. Run backfill_training_data.py first.")
        return []
    with open(path) as f:
        data = json.load(f)
    print(f"  nba_api backfill: {len(data):,} records loaded")
    return data


def normalize_record(r, source='unknown'):
    """Normalize a record to standard schema for backtesting."""
    return {
        'player': r.get('player', ''),
        'stat': (r.get('stat', '') or '').lower(),
        'line': float(r.get('line', 0) or 0),
        'projection': float(r.get('projection', 0) or 0),
        'gap': float(r.get('gap', 0) or 0),
        'abs_gap': abs(float(r.get('gap', 0) or 0)),
        'direction': (r.get('direction', '') or '').upper(),
        'tier': (r.get('tier', 'F') or 'F').upper(),
        'hit': bool(r.get('_hit_label', False)),
        'date': r.get('_date', ''),
        'source': source,
        # Features
        'l10_avg': float(r.get('l10_avg', 0) or 0),
        'l5_avg': float(r.get('l5_avg', 0) or 0),
        'l3_avg': float(r.get('l3_avg', 0) or 0),
        'season_avg': float(r.get('season_avg', 0) or 0),
        'l10_hit_rate': float(r.get('l10_hit_rate', 0) or 0),
        'l5_hit_rate': float(r.get('l5_hit_rate', 0) or 0),
        'season_hit_rate': float(r.get('season_hit_rate', 0) or 0),
        'mins_30plus_pct': float(r.get('mins_30plus_pct', 0) or 0),
        'l10_miss_count': int(r.get('l10_miss_count', 0) or 0),
        'l10_std': float(r.get('l10_std', 0) or 0),
        'l10_floor': float(r.get('l10_floor', 0) or 0),
        'is_home': int(r.get('is_home', 0) or 0),
        'is_b2b': int(r.get('is_b2b', 0) or 0),
        'spread': float(r.get('spread', 0) or 0) if r.get('spread') and str(r.get('spread')) != 'nan' else 0,
        'streak_status': (r.get('streak_status', '') or '').upper(),
        'usage_rate': float(r.get('usage_rate', 0) or 0),
        'travel_distance': float(r.get('travel_distance', 0) or 0),
        'games_used': int(r.get('games_used', 0) or 0),
        'matchup_adjustment': float(r.get('matchup_adjustment', 0) or 0),
        'opp_stat_allowed_vs_league_avg': float(r.get('opp_stat_allowed_vs_league_avg', 0) or 0),
    }


def load_all_data():
    """Load and merge all data sources."""
    print("Loading data sources...")

    sgo = load_sgo_data()
    bf = load_backfill_data()

    all_records = []

    for r in sgo:
        nr = normalize_record(r, 'sgo')
        if nr['date'] and nr['stat'] and nr['stat'] not in BANNED_STATS:
            all_records.append(nr)

    for r in bf:
        nr = normalize_record(r, 'backfill')
        if nr['date'] and nr['stat'] and nr['stat'] not in BANNED_STATS:
            all_records.append(nr)

    print(f"  Total normalized records: {len(all_records):,}")

    # Group by date
    by_date = defaultdict(list)
    for r in all_records:
        by_date[r['date']].append(r)

    dates_sorted = sorted(by_date.keys())
    print(f"  Total dates: {len(dates_sorted)}")
    print(f"  Date range: {dates_sorted[0]} to {dates_sorted[-1]}")

    # Stats summary
    hit_count = sum(1 for r in all_records if r['hit'])
    print(f"  Overall hit rate: {hit_count/len(all_records):.1%}")

    return all_records, by_date, dates_sorted


# ─────────────────────────────────────────────────────────
# FILTERING
# ─────────────────────────────────────────────────────────

def apply_filters(picks, filters):
    """Apply filter config to a list of picks. Returns eligible candidates."""
    candidates = []

    min_tier_val = TIER_ORDER.get(filters.get('min_tier', 'F'), 0)
    min_gap = filters.get('min_gap', 0)
    min_l10_hr = filters.get('min_l10_hr', 0)
    min_season_hr = filters.get('min_season_hr', 0)
    min_mins_pct = filters.get('min_mins_pct', 0)
    min_ensemble = filters.get('min_ensemble_prob', 0)
    allowed_stats = set(filters.get('allowed_stats', []))
    exclude_combos = filters.get('exclude_combos', False)
    direction_strategy = filters.get('direction_strategy', 'both')
    max_l10_miss = filters.get('max_l10_miss', 10)
    max_l10_std = filters.get('max_l10_std', None)
    min_games = filters.get('min_games', 0)
    streak_filter = filters.get('streak_filter', 'none')
    require_under = filters.get('require_under', False)

    combo_stats = {'pra', 'pr', 'pa', 'ra'}

    for p in picks:
        stat = p['stat']

        # Stat filter
        if allowed_stats and stat not in allowed_stats:
            continue

        # Combo exclusion
        if exclude_combos and stat in combo_stats:
            continue

        # Banned stats
        if stat in BANNED_STATS:
            continue

        # Tier filter
        tier_val = TIER_ORDER.get(p['tier'], 0)
        if tier_val < min_tier_val:
            continue

        # Gap filter
        if p['abs_gap'] < min_gap:
            continue

        # Direction filter
        direction = p['direction']
        if direction_strategy == 'under_only' and direction != 'UNDER':
            continue
        elif direction_strategy == 'over_only' and direction != 'OVER':
            continue
        elif direction_strategy == 'under_heavy':
            # Allow UNDERs freely, OVERs only if gap >= 3
            if direction == 'OVER' and p['abs_gap'] < 3.0:
                continue
        elif direction_strategy == 'cold_under':
            # Only UNDER + COLD/NEUTRAL streak
            if direction != 'UNDER':
                continue
            if p['streak_status'] == 'HOT':
                continue

        # Hit rate filter
        if p['l10_hit_rate'] < min_l10_hr:
            continue
        if p['season_hit_rate'] < min_season_hr:
            continue

        # Minutes filter
        if p['mins_30plus_pct'] < min_mins_pct:
            continue

        # Miss count filter
        if p['l10_miss_count'] > max_l10_miss:
            continue

        # Consistency filter
        if max_l10_std is not None and p['l10_std'] > max_l10_std:
            continue

        # Games played
        if p['games_used'] < min_games:
            continue

        # Streak filter
        if streak_filter == 'no_cold' and p['streak_status'] == 'COLD':
            continue
        elif streak_filter == 'hot_only' and p['streak_status'] != 'HOT':
            continue
        elif streak_filter == 'cold_under_only':
            if not (p['streak_status'] == 'COLD' and p['direction'] == 'UNDER'):
                if not (p['streak_status'] != 'HOT'):  # also allow neutral
                    continue

        candidates.append(p)

    return candidates


# ─────────────────────────────────────────────────────────
# PARLAY GENERATION
# ─────────────────────────────────────────────────────────

def generate_parlays(candidates, n_legs, max_parlays=5000):
    """Generate N-leg parlays from candidates via sampling."""
    if len(candidates) < n_legs:
        return []

    n_cands = len(candidates)
    # Estimate total combos
    from math import comb
    total_combos = comb(n_cands, n_legs)

    if total_combos <= max_parlays:
        # Exhaustive
        parlays = []
        for combo in combinations(range(n_cands), n_legs):
            legs = [candidates[i] for i in combo]
            # No duplicate players
            players = [l['player'] for l in legs]
            if len(set(players)) != len(players):
                continue
            parlays.append(legs)
            if len(parlays) >= max_parlays:
                break
        return parlays
    else:
        # Random sampling
        parlays = []
        seen = set()
        attempts = 0
        max_attempts = max_parlays * 15
        while len(parlays) < max_parlays and attempts < max_attempts:
            attempts += 1
            indices = tuple(sorted(random.sample(range(n_cands), n_legs)))
            if indices in seen:
                continue
            seen.add(indices)
            legs = [candidates[i] for i in indices]
            players = [l['player'] for l in legs]
            if len(set(players)) != len(players):
                continue
            parlays.append(legs)
        return parlays


def evaluate_parlay(legs):
    """Check if all legs hit."""
    for leg in legs:
        if not leg['hit']:
            return 'LOSS'
    return 'WIN'


# ─────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────

def run_backtest(by_date, dates_sorted, filters, max_parlays_per_day=5000,
                 min_dates=20, verbose=True):
    """Run full backtest across all dates with given filters."""

    results = {
        'by_date': {},
        'all_eligible': [],
        'total_2leg': 0, 'wins_2leg': 0,
        'total_3leg': 0, 'wins_3leg': 0,
    }

    for i, date in enumerate(dates_sorted):
        day_picks = by_date[date]
        candidates = apply_filters(day_picks, filters)

        if len(candidates) < 2:
            continue

        # Individual hit rate for this day's eligible picks
        day_hits = sum(1 for c in candidates if c['hit'])
        day_hr = day_hits / len(candidates) if candidates else 0

        results['all_eligible'].extend(candidates)

        # Generate parlays
        parlays_2 = generate_parlays(candidates, 2, max_parlays_per_day) if len(candidates) >= 2 else []
        parlays_3 = generate_parlays(candidates, 3, max_parlays_per_day) if len(candidates) >= 3 else []

        wins_2 = sum(1 for p in parlays_2 if evaluate_parlay(p) == 'WIN')
        wins_3 = sum(1 for p in parlays_3 if evaluate_parlay(p) == 'WIN')

        wr_2 = wins_2 / len(parlays_2) if parlays_2 else 0
        wr_3 = wins_3 / len(parlays_3) if parlays_3 else 0

        results['by_date'][date] = {
            'candidates': len(candidates),
            'ind_hr': day_hr,
            'parlays_2': len(parlays_2), 'wins_2': wins_2, 'wr_2': wr_2,
            'parlays_3': len(parlays_3), 'wins_3': wins_3, 'wr_3': wr_3,
        }
        results['total_2leg'] += len(parlays_2)
        results['wins_2leg'] += wins_2
        results['total_3leg'] += len(parlays_3)
        results['wins_3leg'] += wins_3

        if verbose and (i + 1) % 50 == 0:
            avg_2 = results['wins_2leg'] / results['total_2leg'] if results['total_2leg'] else 0
            avg_3 = results['wins_3leg'] / results['total_3leg'] if results['total_3leg'] else 0
            print(f"  [{i+1}/{len(dates_sorted)}] 2-leg: {avg_2:.1%} | 3-leg: {avg_3:.1%}")

    # Compute summaries
    active_dates = [d for d in results['by_date'].values() if d['parlays_2'] > 0]

    results['summary'] = {
        'dates_with_parlays': len(active_dates),
        'wr_2leg': results['wins_2leg'] / results['total_2leg'] if results['total_2leg'] else 0,
        'wr_3leg': results['wins_3leg'] / results['total_3leg'] if results['total_3leg'] else 0,
        'total_2leg': results['total_2leg'],
        'total_3leg': results['total_3leg'],
        'avg_candidates_per_day': np.mean([d['candidates'] for d in active_dates]) if active_dates else 0,
        'avg_ind_hr': np.mean([d['ind_hr'] for d in active_dates]) if active_dates else 0,
        'avg_parlays_2_per_day': np.mean([d['parlays_2'] for d in active_dates]) if active_dates else 0,
        'avg_parlays_3_per_day': np.mean([d['parlays_3'] for d in active_dates]) if active_dates else 0,
    }

    if active_dates:
        wr_2_list = [d['wr_2'] for d in active_dates if d['parlays_2'] > 0]
        wr_3_list = [d['wr_3'] for d in active_dates if d['parlays_3'] > 0]
        results['summary']['std_2leg'] = np.std(wr_2_list) if wr_2_list else 0
        results['summary']['std_3leg'] = np.std(wr_3_list) if wr_3_list else 0
        results['summary']['min_2leg'] = min(wr_2_list) if wr_2_list else 0
        results['summary']['max_2leg'] = max(wr_2_list) if wr_2_list else 0
        results['summary']['min_3leg'] = min(wr_3_list) if wr_3_list else 0
        results['summary']['max_3leg'] = max(wr_3_list) if wr_3_list else 0

    return results


# ─────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────

def print_diagnostics(results, filters, iteration):
    """Print detailed diagnostic report."""
    s = results['summary']
    eligible = results['all_eligible']

    ind_hr = sum(1 for p in eligible if p['hit']) / len(eligible) if eligible else 0

    print(f"\n{'='*60}")
    print(f"  ITERATION #{iteration}")
    print(f"{'='*60}")

    print(f"\n  INDIVIDUAL PICKS:")
    print(f"    Total eligible: {len(eligible):,}")
    print(f"    Hit rate: {ind_hr:.1%}")

    # By direction
    for d in ['OVER', 'UNDER']:
        picks = [p for p in eligible if p['direction'] == d]
        if picks:
            hr = sum(1 for p in picks if p['hit']) / len(picks)
            print(f"    {d}: {hr:.1%} ({len(picks):,})")

    print(f"\n  2-LEG PARLAYS:")
    print(f"    Total: {s['total_2leg']:,} | Wins: {results['wins_2leg']:,} | WR: {s['wr_2leg']:.1%} {'✅' if s['wr_2leg'] >= 0.80 else '❌'}")
    if s.get('std_2leg') is not None:
        print(f"    Range: {s.get('min_2leg',0):.1%} – {s.get('max_2leg',0):.1%} | StdDev: {s.get('std_2leg',0):.1%}")
    print(f"    Avg/day: {s.get('avg_parlays_2_per_day',0):.0f}")

    print(f"\n  3-LEG PARLAYS:")
    print(f"    Total: {s['total_3leg']:,} | Wins: {results['wins_3leg']:,} | WR: {s['wr_3leg']:.1%} {'✅' if s['wr_3leg'] >= 0.80 else '❌'}")
    if s.get('std_3leg') is not None:
        print(f"    Range: {s.get('min_3leg',0):.1%} – {s.get('max_3leg',0):.1%} | StdDev: {s.get('std_3leg',0):.1%}")
    print(f"    Avg/day: {s.get('avg_parlays_3_per_day',0):.0f}")

    print(f"\n  FILTERS:")
    for k, v in sorted(filters.items()):
        print(f"    {k}: {v}")

    # Breakdown by tier
    print(f"\n  BY TIER:")
    for tier in ['S', 'A', 'B', 'C', 'D', 'F']:
        picks = [p for p in eligible if p['tier'] == tier]
        if picks:
            hr = sum(1 for p in picks if p['hit']) / len(picks)
            print(f"    {tier}: {hr:.1%} ({len(picks):,})")

    # Breakdown by stat
    print(f"\n  BY STAT:")
    for stat in sorted(set(p['stat'] for p in eligible)):
        picks = [p for p in eligible if p['stat'] == stat]
        if picks:
            hr = sum(1 for p in picks if p['hit']) / len(picks)
            print(f"    {stat}: {hr:.1%} ({len(picks):,})")

    # Breakdown by gap bucket
    print(f"\n  BY GAP:")
    for lo, hi in [(0,1), (1,2), (2,3), (3,4), (4,5), (5,7), (7,10), (10,99)]:
        picks = [p for p in eligible if lo <= p['abs_gap'] < hi]
        if picks:
            hr = sum(1 for p in picks if p['hit']) / len(picks)
            print(f"    [{lo}-{hi}): {hr:.1%} ({len(picks):,})")

    # Breakdown by L10 HR bucket
    print(f"\n  BY L10 HIT RATE:")
    for lo, hi in [(0,30), (30,50), (50,60), (60,70), (70,80), (80,90), (90,101)]:
        picks = [p for p in eligible if lo <= p['l10_hit_rate'] < hi]
        if picks:
            hr = sum(1 for p in picks if p['hit']) / len(picks)
            print(f"    [{lo}-{hi}%): {hr:.1%} ({len(picks):,})")

    # Streak analysis
    print(f"\n  BY STREAK + DIRECTION:")
    for streak in ['HOT', 'COLD', 'NEUTRAL', '']:
        for direction in ['OVER', 'UNDER']:
            picks = [p for p in eligible if p['streak_status'] == streak and p['direction'] == direction]
            if len(picks) >= 20:
                hr = sum(1 for p in picks if p['hit']) / len(picks)
                print(f"    {streak or 'NONE'}+{direction}: {hr:.1%} ({len(picks):,})")

    print(f"\n  MATH CHECK:")
    print(f"    Ind HR: {ind_hr:.1%}")
    print(f"    Theoretical 2-leg: {ind_hr**2:.1%}")
    print(f"    Theoretical 3-leg: {ind_hr**3:.1%}")
    print(f"    Actual 2-leg: {s['wr_2leg']:.1%}")
    print(f"    Actual 3-leg: {s['wr_3leg']:.1%}")
    print(f"    For 80% 2-leg need: {0.80**0.5:.1%} per leg")
    print(f"    For 80% 3-leg need: {0.80**(1/3):.1%} per leg")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────
# ITERATION STRATEGIES
# ─────────────────────────────────────────────────────────

def get_iteration_configs():
    """Return ordered list of filter configs to test."""
    configs = []

    # Iteration 1: Baseline — loose filters
    configs.append({
        'name': 'baseline_loose',
        'min_gap': 1.0, 'min_tier': 'C', 'min_l10_hr': 0,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 10,
        'streak_filter': 'none', 'min_games': 5, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 2: UNDER-only (the big insight from Engine data)
    configs.append({
        'name': 'under_only_loose',
        'min_gap': 1.0, 'min_tier': 'C', 'min_l10_hr': 0,
        'direction_strategy': 'under_only', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 10,
        'streak_filter': 'none', 'min_games': 5, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 3: High gap only
    configs.append({
        'name': 'high_gap_3plus',
        'min_gap': 3.0, 'min_tier': 'B', 'min_l10_hr': 0,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 10,
        'streak_filter': 'none', 'min_games': 5, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 4: High gap + UNDER only
    configs.append({
        'name': 'under_gap3',
        'min_gap': 3.0, 'min_tier': 'B', 'min_l10_hr': 0,
        'direction_strategy': 'under_only', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 10,
        'streak_filter': 'none', 'min_games': 5, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 5: S/A tier only + high gap
    configs.append({
        'name': 'sa_tier_gap3',
        'min_gap': 3.0, 'min_tier': 'A', 'min_l10_hr': 50,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 50, 'max_l10_miss': 5,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 6: S/A + UNDER only + gap 3
    configs.append({
        'name': 'sa_under_gap3',
        'min_gap': 3.0, 'min_tier': 'A', 'min_l10_hr': 50,
        'direction_strategy': 'under_only', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 50, 'max_l10_miss': 5,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 7: Extreme gap (5+)
    configs.append({
        'name': 'extreme_gap5',
        'min_gap': 5.0, 'min_tier': 'B', 'min_l10_hr': 0,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 10,
        'streak_filter': 'none', 'min_games': 5, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 8: L10 HR >= 70%
    configs.append({
        'name': 'high_l10hr_70',
        'min_gap': 1.0, 'min_tier': 'C', 'min_l10_hr': 70,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 3,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 9: L10 HR >= 80%
    configs.append({
        'name': 'high_l10hr_80',
        'min_gap': 1.0, 'min_tier': 'C', 'min_l10_hr': 80,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 2,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 10: L10 HR 80 + UNDER
    configs.append({
        'name': 'hr80_under',
        'min_gap': 1.0, 'min_tier': 'C', 'min_l10_hr': 80,
        'direction_strategy': 'under_only', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 2,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 11: HR 90+ (ultra selective)
    configs.append({
        'name': 'hr90_ultra',
        'min_gap': 1.0, 'min_tier': 'C', 'min_l10_hr': 90,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 1,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 12: Gap 5 + HR 70 + S/A tier
    configs.append({
        'name': 'gap5_hr70_sa',
        'min_gap': 5.0, 'min_tier': 'A', 'min_l10_hr': 70,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 50, 'max_l10_miss': 3,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 50,
        'max_l10_std': None,
    })

    # Iteration 13: No combos + gap 3 + HR 70
    configs.append({
        'name': 'no_combos_gap3_hr70',
        'min_gap': 3.0, 'min_tier': 'B', 'min_l10_hr': 70,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': True, 'min_mins_pct': 50, 'max_l10_miss': 3,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 14: BLK/STL UNDER only (Engine's best combo: 73.3%)
    configs.append({
        'name': 'blk_stl_under',
        'min_gap': 0.5, 'min_tier': 'C', 'min_l10_hr': 0,
        'direction_strategy': 'under_only', 'allowed_stats': ['blk', 'stl'],
        'exclude_combos': True, 'min_mins_pct': 0, 'max_l10_miss': 10,
        'streak_filter': 'none', 'min_games': 5, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 15: Low variance + high HR
    configs.append({
        'name': 'low_var_hr70',
        'min_gap': 1.5, 'min_tier': 'B', 'min_l10_hr': 70,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 50, 'max_l10_miss': 3,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 50,
        'max_l10_std': 6.0,
    })

    # Iteration 16: COLD + UNDER (Engine: 75.3%)
    configs.append({
        'name': 'cold_under',
        'min_gap': 1.0, 'min_tier': 'C', 'min_l10_hr': 0,
        'direction_strategy': 'cold_under', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 10,
        'streak_filter': 'none', 'min_games': 5, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 17: Gap 4 + HR 60 + UNDER
    configs.append({
        'name': 'gap4_hr60_under',
        'min_gap': 4.0, 'min_tier': 'B', 'min_l10_hr': 60,
        'direction_strategy': 'under_only', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 4,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 18: Gap 7+ mega gap
    configs.append({
        'name': 'mega_gap7',
        'min_gap': 7.0, 'min_tier': 'C', 'min_l10_hr': 0,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': False, 'min_mins_pct': 0, 'max_l10_miss': 10,
        'streak_filter': 'none', 'min_games': 5, 'min_season_hr': 0,
        'max_l10_std': None,
    })

    # Iteration 19: Compound — gap 3 + HR 80 + S/A + UNDER + low var
    configs.append({
        'name': 'compound_optimal',
        'min_gap': 3.0, 'min_tier': 'A', 'min_l10_hr': 80,
        'direction_strategy': 'under_only', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': True, 'min_mins_pct': 50, 'max_l10_miss': 2,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 60,
        'max_l10_std': 8.0,
    })

    # Iteration 20: Season HR 60+ HR 70+ gap 2
    configs.append({
        'name': 'season_hr60_l10hr70_gap2',
        'min_gap': 2.0, 'min_tier': 'B', 'min_l10_hr': 70,
        'direction_strategy': 'both', 'allowed_stats': ['pts', 'pra', 'reb', 'ast', 'blk', 'stl'],
        'exclude_combos': True, 'min_mins_pct': 60, 'max_l10_miss': 3,
        'streak_filter': 'none', 'min_games': 10, 'min_season_hr': 60,
        'max_l10_std': None,
    })

    return configs


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    start_time = time.time()

    # Load data
    all_records, by_date, dates_sorted = load_all_data()

    configs = get_iteration_configs()

    best_2leg_wr = 0
    best_3leg_wr = 0
    best_config = None
    all_results = []

    print(f"\n{'='*60}")
    print(f"  STARTING MASS BACKTEST — {len(configs)} ITERATIONS")
    print(f"  {len(all_records):,} records across {len(dates_sorted)} dates")
    print(f"{'='*60}\n")

    for i, filters in enumerate(configs):
        name = filters.pop('name', f'iter_{i+1}')
        print(f"\n--- Iteration {i+1}/{len(configs)}: {name} ---")

        t0 = time.time()
        results = run_backtest(by_date, dates_sorted, filters,
                              max_parlays_per_day=5000, verbose=(i < 3))
        elapsed = time.time() - t0

        s = results['summary']
        eligible = results['all_eligible']
        ind_hr = sum(1 for p in eligible if p['hit']) / len(eligible) if eligible else 0

        print(f"  Time: {elapsed:.1f}s | Eligible: {len(eligible):,} | Ind HR: {ind_hr:.1%}")
        print(f"  2-leg: {s['wr_2leg']:.1%} ({s['total_2leg']:,}) | 3-leg: {s['wr_3leg']:.1%} ({s['total_3leg']:,})")
        print(f"  Avg candidates/day: {s['avg_candidates_per_day']:.1f}")

        # Track best
        combined = s['wr_2leg'] * 0.5 + s['wr_3leg'] * 0.5
        if combined > best_2leg_wr * 0.5 + best_3leg_wr * 0.5:
            best_2leg_wr = s['wr_2leg']
            best_3leg_wr = s['wr_3leg']
            best_config = {'name': name, 'filters': filters.copy(), 'iteration': i+1}
            best_results = results

        all_results.append({
            'iteration': i+1,
            'name': name,
            'filters': {k: v for k, v in filters.items() if not isinstance(v, (list, set)) or len(v) < 20},
            'ind_hr': ind_hr,
            'wr_2leg': s['wr_2leg'],
            'wr_3leg': s['wr_3leg'],
            'total_2leg': s['total_2leg'],
            'total_3leg': s['total_3leg'],
            'eligible': len(eligible),
            'avg_cands_per_day': s['avg_candidates_per_day'],
            'elapsed': elapsed,
        })

        # Check convergence
        if s['wr_2leg'] >= 0.80 and s['wr_3leg'] >= 0.80 and s['avg_candidates_per_day'] >= 3:
            print(f"\n  ✅ CONVERGED at iteration {i+1}!")
            break

    # Print best result diagnostics
    if best_config:
        print(f"\n\n{'='*60}")
        print(f"  BEST CONFIG: {best_config['name']} (iteration {best_config['iteration']})")
        print_diagnostics(best_results, best_config['filters'], best_config['iteration'])

    # Print all iterations summary
    print(f"\n{'='*60}")
    print(f"  ALL ITERATIONS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'#':>3} {'Name':<30} {'IndHR':>6} {'2-leg':>7} {'3-leg':>7} {'Elig':>7} {'Cands/d':>8}")
    print(f"  {'-'*75}")
    for r in all_results:
        marker = ' ◀' if r['name'] == best_config['name'] else ''
        print(f"  {r['iteration']:>3} {r['name']:<30} {r['ind_hr']:>5.1%} {r['wr_2leg']:>6.1%} {r['wr_3leg']:>6.1%} {r['eligible']:>7,} {r['avg_cands_per_day']:>7.1f}{marker}")

    # Mathematical reality check
    print(f"\n{'='*60}")
    print(f"  MATHEMATICAL REALITY CHECK")
    print(f"{'='*60}")
    best_ind = max(r['ind_hr'] for r in all_results)
    print(f"  Best individual HR achieved: {best_ind:.1%}")
    print(f"  Theoretical 2-leg ceiling: {best_ind**2:.1%}")
    print(f"  Theoretical 3-leg ceiling: {best_ind**3:.1%}")
    print(f"  For 80% 2-leg: need {0.80**0.5:.1%} per leg")
    print(f"  For 80% 3-leg: need {0.80**(1/3):.1%} per leg")
    print(f"  Best actual 2-leg: {best_2leg_wr:.1%}")
    print(f"  Best actual 3-leg: {best_3leg_wr:.1%}")

    if best_ind >= 0.93:
        print(f"\n  3-leg 80% is ACHIEVABLE (ind HR {best_ind:.1%} >= 93%)")
    elif best_ind >= 0.894:
        print(f"\n  2-leg 80% is ACHIEVABLE (ind HR {best_ind:.1%} >= 89.4%)")
        print(f"  3-leg 80% needs tighter filters (need 93%)")
    else:
        print(f"\n  80% parlay WR is NOT achievable at {best_ind:.1%} per-leg HR.")
        print(f"  Best realistic targets:")
        print(f"    2-leg: {best_ind**2:.1%}")
        print(f"    3-leg: {best_ind**3:.1%}")

    total_time = time.time() - start_time
    print(f"\n  Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Save results
    output_path = os.path.join(OUTPUT, 'backtest_results.json')
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(all_records),
        'total_dates': len(dates_sorted),
        'iterations': all_results,
        'best_config': best_config,
        'best_2leg_wr': best_2leg_wr,
        'best_3leg_wr': best_3leg_wr,
    }
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")


if __name__ == '__main__':
    main()

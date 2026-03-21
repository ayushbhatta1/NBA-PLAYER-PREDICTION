#!/usr/bin/env python3
"""
Backtest SNIPER V3 — validates scoring upgrades against graded data.

Simulates SNIPER V3 (with ML signals, floor gates, composite scoring)
on historical graded days. Compares against original SNIPER baseline.

Usage:
    python3 predictions/backtest_sniper_v3.py
"""

import json
import os
import sys
import glob
from collections import defaultdict

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))

# Import scoring functions
sys.path.insert(0, PREDICTIONS_DIR)
from parlay_engine import (
    _sniper_score, _floor_score, _composite_safe_score,
    _is_eligible, _make_leg, _kelly_fraction,
    BLACKLISTED_PLAYERS, COMBO_STATS
)


def load_graded_days():
    """Load all graded results from predictions/YYYY-MM-DD/ folders."""
    days = {}
    for d in sorted(glob.glob(os.path.join(PREDICTIONS_DIR, '20*-*-*'))):
        date = os.path.basename(d)
        # Look for graded results
        for pattern in ['*graded*.json', '*results*.json', '*full_board*.json']:
            files = glob.glob(os.path.join(d, pattern))
            for f in files:
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    # Handle both list and dict formats
                    records = None
                    if isinstance(data, list) and len(data) > 0:
                        records = data
                    elif isinstance(data, dict):
                        # Try common keys
                        for key in ['results', 'picks', 'props']:
                            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                                records = data[key]
                                break
                    if records is None:
                        continue
                    # Check if it has grading info (hit/result field)
                    has_grade = any(
                        r.get('hit') is not None or r.get('result') in ('HIT', 'MISS')
                        for r in records
                    )
                    if has_grade:
                        days[date] = records
                        break
                except Exception:
                    continue
        if date in days:
            continue
    return days


def extract_hit(record):
    """Extract hit/miss from graded record."""
    if 'hit' in record and record['hit'] is not None:
        return bool(record['hit'])
    result = record.get('result', '')
    if isinstance(result, str):
        if result.upper() == 'HIT':
            return True
        if result.upper() == 'MISS':
            return False
    return None


def simulate_sniper_v3(props, use_composite=False, use_floor_gate=True):
    """Simulate SNIPER V3 pick selection on a set of graded props.

    Returns list of selected picks (with hit/miss info preserved).
    """
    def _l5_trending_down(p):
        l5 = p.get('l5_avg', 0) or 0
        l10 = p.get('l10_avg', 0) or 0
        return l5 > 0 and l10 > 0 and l5 < l10

    score_fn = _composite_safe_score if use_composite else _sniper_score

    # Multi-pass selection (mirrors build_primary_safe)
    for pass_num, filters in enumerate([
        # Pass 1: UNDER + L5<L10 + small line + line above avg + no stars
        lambda p: (
            _is_eligible(p) and
            p.get('direction', '').upper() == 'UNDER' and
            _l5_trending_down(p) and
            (p.get('line', 0) or 0) <= 15 and
            ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 0.5 and
            (p.get('season_avg', 0) or 0) < 22
        ),
        # Pass 2: UNDER + L5<L10 + any line + line above avg
        lambda p: (
            _is_eligible(p) and
            p.get('direction', '').upper() == 'UNDER' and
            _l5_trending_down(p) and
            ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 1 and
            (p.get('season_avg', 0) or 0) < 25
        ),
        # Pass 3: Any UNDER with line above avg
        lambda p: (
            _is_eligible(p) and
            p.get('direction', '').upper() == 'UNDER' and
            ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 0.5
        ),
    ]):
        candidates = [p for p in props if filters(p)]
        candidates.sort(key=score_fn, reverse=True)

        used_games = set()
        picks = []
        for p in candidates:
            # Floor gate
            if use_floor_gate and _floor_score(p) < 0.3:
                continue
            # Game diversification
            g = p.get('game', '')
            if g and g in used_games:
                continue
            picks.append(p)
            if g:
                used_games.add(g)
            if len(picks) >= 3:
                return picks

    return picks


def simulate_original_sniper(props):
    """Simulate original SNIPER (pre-V3) pick selection."""
    return simulate_sniper_v3(props, use_composite=False, use_floor_gate=False)


def grade_parlay(picks):
    """Grade a 3-leg parlay. Returns (cashed, legs_hit, legs_total)."""
    hits = 0
    total = 0
    for p in picks:
        hit = extract_hit(p)
        if hit is not None:
            total += 1
            if hit:
                hits += 1
    cashed = (hits == total and total >= 3)
    return cashed, hits, total


def main():
    print("=" * 70)
    print("  SNIPER V3 BACKTEST — Graded Data Validation")
    print("=" * 70)

    days = load_graded_days()
    if not days:
        print("\n  No graded data found!")
        return

    print(f"\n  Found {len(days)} graded days: {', '.join(sorted(days.keys()))}")

    # Strategies to compare
    strategies = {
        'original_sniper': lambda props: simulate_original_sniper(props),
        'sniper_v3_pure': lambda props: simulate_sniper_v3(props, use_composite=False, use_floor_gate=True),
        'sniper_v3_composite': lambda props: simulate_sniper_v3(props, use_composite=True, use_floor_gate=True),
        'sniper_v3_no_gate': lambda props: simulate_sniper_v3(props, use_composite=True, use_floor_gate=False),
    }

    # Track results per strategy
    results = {name: {'wins': 0, 'losses': 0, 'dnp': 0, 'legs_hit': 0, 'legs_total': 0, 'daily': []}
               for name in strategies}

    for date in sorted(days.keys()):
        props = days[date]
        graded_props = [p for p in props if extract_hit(p) is not None]

        if len(graded_props) < 10:
            continue

        print(f"\n  {date}: {len(graded_props)} graded props")

        for name, strategy_fn in strategies.items():
            picks = strategy_fn(graded_props)

            if len(picks) < 3:
                results[name]['dnp'] += 1
                results[name]['daily'].append((date, 'DNP', 0, 0))
                continue

            cashed, hits, total = grade_parlay(picks)
            results[name]['legs_hit'] += hits
            results[name]['legs_total'] += total

            if cashed:
                results[name]['wins'] += 1
                results[name]['daily'].append((date, 'WIN', hits, total))
            else:
                results[name]['losses'] += 1
                results[name]['daily'].append((date, 'LOSS', hits, total))

            # Print picks for this strategy
            status = "WIN" if cashed else f"LOSS ({hits}/{total})"
            player_names = [p.get('player', '?')[:15] for p in picks[:3]]
            print(f"    {name:25s}: {status:12s}  [{', '.join(player_names)}]")

    # Summary
    print(f"\n{'='*70}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Strategy':<25s} {'W':>3s} {'L':>3s} {'DNP':>4s} {'Cash%':>7s} {'Leg HR':>8s}")
    print(f"  {'-'*55}")

    for name, r in results.items():
        total_played = r['wins'] + r['losses']
        cash_rate = (r['wins'] / total_played * 100) if total_played > 0 else 0
        leg_hr = (r['legs_hit'] / r['legs_total'] * 100) if r['legs_total'] > 0 else 0
        print(f"  {name:<25s} {r['wins']:3d} {r['losses']:3d} {r['dnp']:4d} {cash_rate:6.1f}% {leg_hr:7.1f}%")

    # Triple-SAFE simulation
    print(f"\n{'='*70}")
    print(f"  TRIPLE-SAFE SIMULATION")
    print(f"{'='*70}")

    triple_at_least_1 = 0
    triple_total = 0

    for date in sorted(days.keys()):
        props = days[date]
        graded_props = [p for p in props if extract_hit(p) is not None]
        if len(graded_props) < 15:
            continue

        used_players = set()
        parlays_cashed = 0

        # SAFE #1: Pure SNIPER
        picks1 = simulate_sniper_v3(graded_props, use_composite=False, use_floor_gate=True)
        if len(picks1) >= 3:
            cashed1, _, _ = grade_parlay(picks1)
            if cashed1:
                parlays_cashed += 1
            used_players.update(p.get('player', '') for p in picks1)

        # SAFE #2: Composite (exclude #1 players)
        remaining = [p for p in graded_props if p.get('player', '') not in used_players]
        picks2 = simulate_sniper_v3(remaining, use_composite=True, use_floor_gate=True)
        if len(picks2) >= 3:
            cashed2, _, _ = grade_parlay(picks2)
            if cashed2:
                parlays_cashed += 1
            used_players.update(p.get('player', '') for p in picks2)

        # SAFE #3: From remaining
        remaining = [p for p in graded_props if p.get('player', '') not in used_players]
        picks3 = simulate_sniper_v3(remaining, use_composite=True, use_floor_gate=False)
        if len(picks3) >= 3:
            cashed3, _, _ = grade_parlay(picks3)
            if cashed3:
                parlays_cashed += 1

        at_least_1 = parlays_cashed >= 1
        if at_least_1:
            triple_at_least_1 += 1
        triple_total += 1

        print(f"  {date}: {parlays_cashed}/3 cashed {'Y' if at_least_1 else 'N'}")

    if triple_total > 0:
        rate = triple_at_least_1 / triple_total * 100
        print(f"\n  Triple-SAFE at-least-1 rate: {triple_at_least_1}/{triple_total} = {rate:.1f}%")
        print(f"  Target: >= 95%")

    print(f"\n  Done.")


if __name__ == '__main__':
    main()

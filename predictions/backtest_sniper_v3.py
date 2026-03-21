#!/usr/bin/env python3
"""
Backtest SNIPER V3 — validates scoring upgrades against graded data.

Now includes:
- Play/Skip day classifier from 1M simulation
- 2-leg vs 3-leg comparison
- EV tracking (cash rate × multiplier)

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
    _is_eligible, _make_leg, _kelly_fraction, _sim_sort,
    should_play_today, build_primary_safe, build_2leg_safe,
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
    """Simulate SNIPER V3 pick selection on a set of graded props."""
    def _l5_trending_down(p):
        l5 = p.get('l5_avg', 0) or 0
        l10 = p.get('l10_avg', 0) or 0
        return l5 > 0 and l10 > 0 and l5 < l10

    score_fn = _composite_safe_score if use_composite else _sniper_score

    for pass_num, filters in enumerate([
        lambda p: (
            _is_eligible(p) and
            p.get('direction', '').upper() == 'UNDER' and
            _l5_trending_down(p) and
            (p.get('line', 0) or 0) <= 15 and
            ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 0.5 and
            (p.get('season_avg', 0) or 0) < 22
        ),
        lambda p: (
            _is_eligible(p) and
            p.get('direction', '').upper() == 'UNDER' and
            _l5_trending_down(p) and
            ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 1 and
            (p.get('season_avg', 0) or 0) < 25
        ),
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
            if use_floor_gate and _floor_score(p) < 0.3:
                continue
            g = p.get('game', '')
            if g and g in used_games:
                continue
            picks.append(p)
            if g:
                used_games.add(g)
            if len(picks) >= 3:
                return picks

    return picks


def grade_parlay(picks):
    """Grade a parlay. Returns (cashed, legs_hit, legs_total)."""
    hits = 0
    total = 0
    for p in picks:
        hit = extract_hit(p)
        if hit is not None:
            total += 1
            if hit:
                hits += 1
    cashed = (hits == total and total >= len(picks))
    return cashed, hits, total


def main():
    print("=" * 70)
    print("  SNIPER V3 BACKTEST — with Play/Skip + 2-Leg Options")
    print("=" * 70)

    days = load_graded_days()
    if not days:
        print("\n  No graded data found!")
        return

    print(f"\n  Found {len(days)} graded days: {', '.join(sorted(days.keys()))}")

    # ═══ STRATEGY COMPARISON ═══
    strategies = {
        'sim_3leg': lambda props: build_primary_safe([p for p in props if _is_eligible(p)]),
        'sim_2leg': lambda props: build_2leg_safe([p for p in props if _is_eligible(p)]),
        'composite_3leg': lambda props: simulate_sniper_v3(props, use_composite=True, use_floor_gate=True),
    }

    results = {name: {'wins': 0, 'losses': 0, 'dnp': 0, 'skip': 0,
                       'legs_hit': 0, 'legs_total': 0, 'daily': []}
               for name in strategies}

    # Also track play/skip separately
    play_skip_results = {'play': 0, 'skip': 0, 'play_win': 0, 'play_loss': 0}

    for date in sorted(days.keys()):
        props = days[date]
        graded_props = [p for p in props if extract_hit(p) is not None]

        if len(graded_props) < 10:
            continue

        # Day classifier
        pool = [p for p in graded_props if _is_eligible(p)]
        should_play, reason, n_qual, n_games = should_play_today(pool)

        print(f"\n  {date}: {len(graded_props)} graded | {reason}")

        for name, strategy_fn in strategies.items():
            picks = strategy_fn(graded_props)
            n_legs = 2 if '2leg' in name else 3

            if len(picks) < n_legs:
                results[name]['dnp'] += 1
                results[name]['daily'].append((date, 'DNP', 0, 0, should_play))
                continue

            cashed, hits, total = grade_parlay(picks)
            results[name]['legs_hit'] += hits
            results[name]['legs_total'] += total

            # Track with play/skip
            if not should_play:
                results[name]['skip'] += 1
                results[name]['daily'].append((date, 'SKIP', hits, total, should_play))
                if name == 'sim_3leg':
                    play_skip_results['skip'] += 1
            elif cashed:
                results[name]['wins'] += 1
                results[name]['daily'].append((date, 'WIN', hits, total, should_play))
                if name == 'sim_3leg':
                    play_skip_results['play'] += 1
                    play_skip_results['play_win'] += 1
            else:
                results[name]['losses'] += 1
                results[name]['daily'].append((date, 'LOSS', hits, total, should_play))
                if name == 'sim_3leg':
                    play_skip_results['play'] += 1
                    play_skip_results['play_loss'] += 1

            # Print picks
            status = "WIN" if cashed else f"LOSS ({hits}/{total})"
            if not should_play:
                status = f"SKIP ({status})"
            player_names = [p.get('player', '?')[:15] for p in picks[:n_legs]]
            print(f"    {name:20s}: {status:18s}  [{', '.join(player_names)}]")

    # ═══ SUMMARY ═══
    print(f"\n{'='*70}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Strategy':<20s} {'W':>3s} {'L':>3s} {'Skip':>5s} {'DNP':>4s} {'Cash%':>7s} {'EV':>7s} {'Leg HR':>8s}")
    print(f"  {'-'*60}")

    for name, r in results.items():
        total_played = r['wins'] + r['losses']
        cash_rate = (r['wins'] / total_played * 100) if total_played > 0 else 0
        leg_hr = (r['legs_hit'] / r['legs_total'] * 100) if r['legs_total'] > 0 else 0
        mult = 3.0 if '2leg' in name else 6.0
        ev = cash_rate / 100 * mult
        print(f"  {name:<20s} {r['wins']:3d} {r['losses']:3d} {r['skip']:5d} {r['dnp']:4d} {cash_rate:6.1f}% {ev:6.2f}x {leg_hr:7.1f}%")

    # ═══ PLAY/SKIP ANALYSIS ═══
    print(f"\n{'='*70}")
    print(f"  PLAY/SKIP DAY CLASSIFIER (sim_3leg)")
    print(f"{'='*70}")
    ps = play_skip_results
    if ps['play'] > 0:
        play_cash = ps['play_win'] / ps['play'] * 100
        print(f"  PLAY days:  {ps['play']} ({ps['play_win']}W / {ps['play_loss']}L = {play_cash:.1f}% cash)")
    print(f"  SKIP days:  {ps['skip']}")

    # What would have happened on skip days?
    skip_would_have_won = 0
    skip_would_have_lost = 0
    for entry in results['sim_3leg']['daily']:
        if len(entry) >= 5 and not entry[4]:  # not should_play
            if entry[1] == 'SKIP':
                # Check actual result
                actual_hits = entry[2]
                actual_total = entry[3]
                if actual_hits == actual_total and actual_total >= 3:
                    skip_would_have_won += 1
                else:
                    skip_would_have_lost += 1
    if skip_would_have_won + skip_would_have_lost > 0:
        print(f"  Skip days actual: {skip_would_have_won}W / {skip_would_have_lost}L "
              f"(correctly skipped {skip_would_have_lost} losses)")

    # ═══ TRIPLE-SAFE SIMULATION ═══
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

    print(f"\n  Done.")


if __name__ == '__main__':
    main()

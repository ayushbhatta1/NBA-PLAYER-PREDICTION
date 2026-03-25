#!/usr/bin/env python3
"""
Simple Edge Finder — No ML, just median vs line.

Logic: If the sportsbook line is obviously wrong (3+ points from L10 median),
take the other side. Skip everything else.

Singles only. 1-2 picks per day max.

Usage:
    python3 predictions/simple_edge.py score <results_file>    # Score a board
    python3 predictions/simple_edge.py backtest                # Backtest on all graded data
    python3 predictions/simple_edge.py today <board_file>      # Quick picks from board
"""

import json
import os
import sys
import glob
from collections import defaultdict

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))

# Minimum gap between L10 median and line to trigger a pick
# Scaled by stat type (BLK/STL have smaller ranges than PTS)
STAT_MIN_GAP = {
    'pts': 3.0,
    'reb': 2.0,
    'ast': 1.5,
    '3pm': 1.0,
    'stl': 0.5,
    'blk': 0.5,
    'pra': 4.0,
    'pr': 3.0,
    'pa': 3.0,
    'ra': 2.5,
    'stl_blk': 0.5,
}

DEFAULT_MIN_GAP = 3.0


def find_edges(results):
    """Find props where the line is obviously mispriced.

    Uses L10 median (not mean) vs the sportsbook line.
    Only returns picks where the gap exceeds the stat-specific threshold.
    """
    picks = []

    for r in results:
        stat = r.get('stat', '')
        line = r.get('line')
        l10_avg = r.get('l10_avg')  # We'll use this as proxy; median would be better
        l10_median = r.get('l10_median', l10_avg)  # Use median if available, else avg
        season_avg = r.get('season_avg')
        l5_avg = r.get('l5_avg')
        l10_hr = r.get('l10_hit_rate', 50)
        l5_hr = r.get('l5_hit_rate', 50)
        mins_pct = r.get('mins_30plus_pct', 50)

        if line is None or l10_median is None:
            continue

        # Skip if player has unstable minutes (<40% games with 30+ mins)
        if mins_pct is not None and mins_pct < 40:
            continue

        min_gap = STAT_MIN_GAP.get(stat, DEFAULT_MIN_GAP)

        # The edge: how far is the median from the line?
        gap = l10_median - line

        # Direction: if median is well below line → UNDER edge
        #            if median is well above line → OVER edge
        if gap < -min_gap:
            direction = 'UNDER'
            edge = abs(gap)
        elif gap > min_gap:
            direction = 'OVER'
            edge = abs(gap)
        else:
            continue  # No edge — skip

        # Confirmation checks (optional boosters, not required)
        confidence = 0

        # 1. Hit rate confirms direction
        if direction == 'UNDER' and l10_hr is not None and l10_hr <= 40:
            confidence += 1  # L10 HR says UNDER hitting
        elif direction == 'OVER' and l10_hr is not None and l10_hr >= 60:
            confidence += 1

        # 2. L5 trend confirms (recent form)
        if l5_avg is not None:
            l5_gap = l5_avg - line
            if direction == 'UNDER' and l5_gap < 0:
                confidence += 1
            elif direction == 'OVER' and l5_gap > 0:
                confidence += 1

        # 3. Season avg also confirms
        if season_avg is not None:
            season_gap = season_avg - line
            if direction == 'UNDER' and season_gap < 0:
                confidence += 1
            elif direction == 'OVER' and season_gap > 0:
                confidence += 1

        pick = {
            'player': r.get('player', '?'),
            'stat': stat,
            'line': line,
            'direction': direction,
            'l10_median': round(l10_median, 1),
            'l10_avg': round(l10_avg, 1) if l10_avg else None,
            'l5_avg': round(l5_avg, 1) if l5_avg else None,
            'season_avg': round(season_avg, 1) if season_avg else None,
            'edge': round(edge, 1),
            'gap_pct': round(edge / line * 100, 1) if line > 0 else 0,
            'confidence': confidence,  # 0-3: how many confirmations
            'l10_hr': l10_hr,
            'l5_hr': l5_hr,
            'mins_pct': mins_pct,
            'game': r.get('game', ''),
            'tier': r.get('tier', '?'),
        }

        # Copy over actual result if grading
        if 'actual' in r:
            pick['actual'] = r['actual']
            pick['result'] = r.get('result', '')
            if direction == 'UNDER':
                pick['hit'] = r['actual'] < line
            else:
                pick['hit'] = r['actual'] > line

        picks.append(pick)

    # Sort by edge size (biggest mispricing first)
    picks.sort(key=lambda p: p['edge'], reverse=True)

    return picks


def print_picks(picks, max_show=20):
    """Print picks in a clean format."""
    if not picks:
        print("  No edges found today.")
        return

    print(f"\n  {'Player':22s} {'Stat':4s} {'Dir':5s} {'Line':>5s} {'Med':>5s} "
          f"{'Edge':>5s} {'Edge%':>5s} {'Conf':>4s} {'L10':>4s} {'Game':>10s}")
    print(f"  {'─'*75}")

    for p in picks[:max_show]:
        conf_stars = '*' * p['confidence']
        print(f"  {p['player']:22s} {p['stat']:4s} {p['direction']:5s} {p['line']:5.1f} "
              f"{p['l10_median']:5.1f} {p['edge']:5.1f} {p['gap_pct']:4.1f}% "
              f"{conf_stars:>4s} {p.get('l10_hr', 0):3.0f}% {p['game']:>10s}")


def backtest():
    """Backtest on all graded data."""
    print("=" * 60)
    print("  Simple Edge Finder — Backtest")
    print("=" * 60)

    graded_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, '*', '*graded*.json')))

    all_picks = []
    daily_results = {}

    for fpath in graded_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except:
            continue

        if isinstance(data, dict):
            records = data.get('results', [])
        else:
            records = data

        # Only graded records with actuals
        records = [r for r in records if r.get('actual') is not None]
        if not records:
            continue

        date = os.path.basename(os.path.dirname(fpath))

        picks = find_edges(records)
        if not picks:
            continue

        # Take top 2 by edge (highest conviction only)
        top2 = picks[:2]

        hits = sum(1 for p in top2 if p.get('hit'))
        total = len(top2)

        daily_results[date] = {
            'hits': hits,
            'total': total,
            'picks': top2,
            'all_edges': len(picks),
        }
        all_picks.extend(top2)

    if not all_picks:
        print("  No graded data found.")
        return

    # Overall
    total_hits = sum(1 for p in all_picks if p.get('hit'))
    total = len(all_picks)
    print(f"\n  TOP 2 PICKS PER DAY (biggest edge only):")
    print(f"  Overall: {total_hits}/{total} = {total_hits/total:.1%}")

    # By day
    print(f"\n  {'Date':12s} {'Hits':>5s} {'Total':>5s} {'Acc':>6s} {'Edges':>6s}")
    print(f"  {'─'*40}")
    for date in sorted(daily_results.keys()):
        d = daily_results[date]
        acc = d['hits'] / d['total'] if d['total'] > 0 else 0
        print(f"  {date:12s} {d['hits']:>5d} {d['total']:>5d} {acc:>6.1%} {d['all_edges']:>6d}")

    # By confidence level
    print(f"\n  By Confirmation Level:")
    for conf in range(4):
        subset = [p for p in all_picks if p['confidence'] == conf]
        if subset:
            hits = sum(1 for p in subset if p.get('hit'))
            print(f"    {conf} confirmations: {hits}/{len(subset)} = {hits/len(subset):.1%}")

    # By stat
    print(f"\n  By Stat Type:")
    stat_results = defaultdict(lambda: [0, 0])
    for p in all_picks:
        stat_results[p['stat']][1] += 1
        if p.get('hit'):
            stat_results[p['stat']][0] += 1
    for stat, (h, t) in sorted(stat_results.items(), key=lambda x: x[1][0]/max(x[1][1],1), reverse=True):
        print(f"    {stat:6s}: {h}/{t} = {h/t:.1%}")

    # By direction
    print(f"\n  By Direction:")
    for d in ['UNDER', 'OVER']:
        subset = [p for p in all_picks if p['direction'] == d]
        if subset:
            hits = sum(1 for p in subset if p.get('hit'))
            print(f"    {d:6s}: {hits}/{len(subset)} = {hits/len(subset):.1%}")

    # All edges (not just top 2)
    print(f"\n  ALL EDGES (every qualifying pick):")
    all_edges = []
    for fpath in graded_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except:
            continue
        if isinstance(data, dict):
            records = data.get('results', [])
        elif isinstance(data, list):
            records = data
        else:
            continue
        records = [r for r in records if isinstance(r, dict) and r.get('actual') is not None]
        edges = find_edges(records)
        all_edges.extend(edges)

    if all_edges:
        hits = sum(1 for p in all_edges if p.get('hit'))
        print(f"  Total: {hits}/{len(all_edges)} = {hits/len(all_edges):.1%}")

        # By edge size buckets
        print(f"\n  By Edge Size:")
        for label, lo, hi in [
            ('Huge (6+)', 6, 100),
            ('Large (4-6)', 4, 6),
            ('Medium (3-4)', 3, 4),
            ('Small (2-3)', 2, 3),
            ('Tiny (1-2)', 1, 2),
        ]:
            subset = [p for p in all_edges if lo <= p['edge'] < hi]
            if subset:
                h = sum(1 for p in subset if p.get('hit'))
                print(f"    {label:15s}: {h}/{len(subset)} = {h/len(subset):.1%}")

    # Simulated parlay (top 2 per day, both must hit)
    print(f"\n  SIMULATED 2-LEG PARLAY (top 2 per day, both must hit):")
    parlay_wins = 0
    parlay_days = 0
    for date in sorted(daily_results.keys()):
        d = daily_results[date]
        if d['total'] == 2:
            parlay_days += 1
            if d['hits'] == 2:
                parlay_wins += 1
    if parlay_days > 0:
        print(f"  {parlay_wins}/{parlay_days} days cashed = {parlay_wins/parlay_days:.1%}")
        print(f"  At 3x payout: EV = {parlay_wins/parlay_days * 3:.2f}x")


def score_board(filepath):
    """Score a board file and show edges."""
    print("=" * 60)
    print("  Simple Edge Finder — Today's Picks")
    print("=" * 60)

    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, dict):
        results = data.get('results', data.get('picks', []))
        if not results and 'parlays' not in data:
            results = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    else:
        results = data

    picks = find_edges(results)

    print(f"\n  Total props: {len(results)}")
    print(f"  Edges found: {len(picks)}")

    if picks:
        print(f"\n  TOP PICKS (play these as singles):")
        print_picks(picks[:5])

        if len(picks) >= 2:
            print(f"\n  BEST 2-LEG PARLAY:")
            # Pick top 2 from different games
            used_games = set()
            parlay = []
            for p in picks:
                if p['game'] not in used_games:
                    parlay.append(p)
                    used_games.add(p['game'])
                    if len(parlay) >= 2:
                        break
            print_picks(parlay)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'backtest':
        backtest()
    elif cmd in ('score', 'today'):
        if len(sys.argv) < 3:
            print("Usage: simple_edge.py score <file>")
            sys.exit(1)
        score_board(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()

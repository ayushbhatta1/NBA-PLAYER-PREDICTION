#!/usr/bin/env python3
"""
Primary Parlay Grader — Tracks cumulative W/L for SAFE + AGGRESSIVE parlays.

Grades the actual parlays we bet on (from primary_parlays.json) and maintains
a cumulative tracker at predictions/primary_parlay_tracker.json.

Usage: python3 predictions/grade_primary_parlays.py 2026-03-16
"""
import sys
import json
import os
import unicodedata
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_FILE = os.path.join(PREDICTIONS_DIR, 'primary_parlay_tracker.json')


def _norm(s):
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _find_actual(player_name, stat, actuals):
    """Find a player's actual stat value with fuzzy matching."""
    p_norm = _norm(player_name)
    p_parts = p_norm.split()
    stat_lower = stat.lower()

    for name, stats in actuals.items():
        n_norm = _norm(name)
        if p_norm == n_norm:
            return stats.get(stat) or stats.get(stat_lower)
        n_parts = n_norm.split()
        if len(p_parts) >= 2 and len(n_parts) >= 2:
            if p_parts[-1] == n_parts[-1] and p_parts[0][0] == n_parts[0][0]:
                return stats.get(stat) or stats.get(stat_lower)
    return None


def grade_primary_parlays(date_str):
    """Grade primary parlays for a given date."""
    primary_file = os.path.join(PREDICTIONS_DIR, date_str, 'primary_parlays.json')
    if not os.path.exists(primary_file):
        print(f"No primary parlays found for {date_str}")
        return None

    with open(primary_file) as f:
        data = json.load(f)

    parlays = data.get('parlays', {})
    if not parlays:
        print(f"Empty primary parlays for {date_str}")
        return None

    # Fetch actuals
    print(f"Fetching box scores for {date_str}...")
    try:
        from analyze_v3 import get_fetcher
        fetcher = get_fetcher()
        actuals = fetcher.get_box_scores(date_str)
        if not actuals:
            print(f"[WARN] No box scores for {date_str}.")
            return None
        print(f"  Got stats for {len(actuals)} players")
    except Exception as e:
        print(f"[ERROR] Failed to fetch box scores: {e}")
        return None

    # Grade each parlay
    graded = {}
    for pname, parlay in parlays.items():
        legs = parlay.get('legs', [])
        if not legs:
            continue

        leg_results = []
        hits = 0
        has_dnp = False

        for leg in legs:
            player = leg.get('player', '')
            stat = leg.get('stat', '')
            line = leg.get('line', 0)
            direction = leg.get('direction', '')

            actual_val = _find_actual(player, stat, actuals)

            if actual_val is None:
                leg_results.append({
                    'player': player, 'stat': stat, 'line': line,
                    'direction': direction, 'actual': None, 'result': 'DNP',
                })
                has_dnp = True
                continue

            if direction == 'OVER':
                hit = actual_val > line
            else:
                hit = actual_val < line

            if hit:
                hits += 1

            leg_results.append({
                'player': player, 'stat': stat, 'line': line,
                'direction': direction, 'actual': actual_val,
                'result': 'HIT' if hit else 'MISS',
                'margin': round(actual_val - line, 1),
            })

        if has_dnp:
            parlay_result = 'DNP'
        elif hits == len(legs):
            parlay_result = 'HIT'
        else:
            parlay_result = 'MISS'

        graded[pname] = {
            'name': parlay.get('name', pname),
            'method': parlay.get('method', ''),
            'result': parlay_result,
            'legs_hit': hits,
            'legs_total': len(legs),
            'leg_results': leg_results,
        }

    # Print report
    print(f"\n{'='*60}")
    print(f"  PRIMARY PARLAY GRADING — {date_str}")
    print(f"{'='*60}")

    for pname, pg in graded.items():
        tag = {'HIT': 'HIT ', 'MISS': 'MISS', 'DNP': 'DNP '}.get(pg['result'], '????')
        print(f"\n  [{tag}] {pg['name']} ({pg['legs_hit']}/{pg['legs_total']} legs)")
        for lr in pg.get('leg_results', []):
            res_tag = {'HIT': 'v', 'MISS': 'x', 'DNP': '?'}.get(lr['result'], ' ')
            actual = lr.get('actual')
            actual_str = f"{actual}" if actual is not None else 'DNP'
            print(f"    [{res_tag}] {lr['player']:22s} {lr['stat'].upper():4s} "
                  f"{lr['direction']:5s} {lr['line']:5.1f}  actual={actual_str}")

    # Save graded
    graded_file = os.path.join(PREDICTIONS_DIR, date_str, 'primary_parlays_graded.json')
    with open(graded_file, 'w') as f:
        json.dump({
            'date': date_str,
            'graded_at': datetime.now().isoformat(),
            'parlays': graded,
        }, f, indent=2)
    print(f"\n  Saved: {graded_file}")

    # Update tracker
    update_tracker(date_str, graded)

    return graded


def update_tracker(date_str, graded):
    """Update cumulative primary parlay tracker."""
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE) as f:
            tracker = json.load(f)
    else:
        tracker = {
            'last_updated': None,
            'daily_results': [],
            'cumulative': {
                'safe': {'wins': 0, 'losses': 0, 'dnp': 0, 'total_legs_hit': 0, 'total_legs': 0},
                'aggressive': {'wins': 0, 'losses': 0, 'dnp': 0, 'total_legs_hit': 0, 'total_legs': 0},
            },
        }

    # Remove existing entry for this date (re-grade support)
    tracker['daily_results'] = [
        d for d in tracker['daily_results'] if d.get('date') != date_str
    ]

    # Add today's results
    day_entry = {'date': date_str}
    for pname, pg in graded.items():
        day_entry[pname] = {
            'result': pg['result'],
            'legs_hit': pg['legs_hit'],
            'legs_total': pg['legs_total'],
        }
    tracker['daily_results'].append(day_entry)
    tracker['daily_results'].sort(key=lambda d: d['date'])

    # Recalculate cumulative stats
    for parlay_type in ('safe', 'aggressive'):
        wins = losses = dnp = total_legs_hit = total_legs = 0
        for day in tracker['daily_results']:
            entry = day.get(parlay_type, {})
            result = entry.get('result', '')
            if result == 'HIT':
                wins += 1
            elif result == 'MISS':
                losses += 1
            elif result == 'DNP':
                dnp += 1
            total_legs_hit += entry.get('legs_hit', 0)
            total_legs += entry.get('legs_total', 0)

        total = wins + losses
        tracker['cumulative'][parlay_type] = {
            'wins': wins,
            'losses': losses,
            'dnp': dnp,
            'win_rate': round(wins / total * 100, 1) if total > 0 else 0.0,
            'total_legs_hit': total_legs_hit,
            'total_legs': total_legs,
            'leg_hit_rate': round(total_legs_hit / total_legs * 100, 1) if total_legs > 0 else 0.0,
        }

    tracker['last_updated'] = datetime.now().isoformat()
    tracker['total_days'] = len(tracker['daily_results'])

    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)

    # Print cumulative
    print(f"\n  CUMULATIVE RECORD ({tracker['total_days']} days)")
    for ptype in ('safe', 'aggressive'):
        c = tracker['cumulative'][ptype]
        print(f"    {ptype.upper():12s}  {c['wins']}W-{c['losses']}L  "
              f"({c['win_rate']}%)  legs: {c['total_legs_hit']}/{c['total_legs']} "
              f"({c['leg_hit_rate']}%)")

    print(f"\n  Tracker: {TRACKER_FILE}")


def grade_engine_backup(date_str):
    """Grade Engine backup parlays from engine_parlays.json (separate from primary tracker)."""
    engine_file = os.path.join(PREDICTIONS_DIR, date_str, 'engine_parlays.json')
    if not os.path.exists(engine_file):
        return None

    with open(engine_file) as f:
        data = json.load(f)

    parlays = data.get('parlays', {})
    if not parlays:
        return None

    # Fetch actuals
    try:
        from analyze_v3 import get_fetcher
        fetcher = get_fetcher()
        actuals = fetcher.get_box_scores(date_str)
        if not actuals:
            return None
    except Exception:
        return None

    graded = {}
    for pname, parlay in parlays.items():
        legs = parlay.get('legs', [])
        if not legs:
            continue

        leg_results = []
        hits = 0
        has_dnp = False

        for leg in legs:
            player = leg.get('player', '')
            stat = leg.get('stat', '')
            line = leg.get('line', 0)
            direction = leg.get('direction', '')

            actual_val = _find_actual(player, stat, actuals)

            if actual_val is None:
                leg_results.append({
                    'player': player, 'stat': stat, 'line': line,
                    'direction': direction, 'actual': None, 'result': 'DNP',
                })
                has_dnp = True
                continue

            if direction == 'OVER':
                hit = actual_val > line
            else:
                hit = actual_val < line

            if hit:
                hits += 1

            leg_results.append({
                'player': player, 'stat': stat, 'line': line,
                'direction': direction, 'actual': actual_val,
                'result': 'HIT' if hit else 'MISS',
                'margin': round(actual_val - line, 1),
            })

        if has_dnp:
            parlay_result = 'DNP'
        elif hits == len(legs):
            parlay_result = 'HIT'
        else:
            parlay_result = 'MISS'

        graded[pname] = {
            'name': parlay.get('name', pname),
            'method': parlay.get('method', 'parlay_engine_v1'),
            'result': parlay_result,
            'legs_hit': hits,
            'legs_total': len(legs),
            'leg_results': leg_results,
        }

    # Print Engine backup report
    if graded:
        print(f"\n{'='*60}")
        print(f"  ENGINE BACKUP GRADING — {date_str}")
        print(f"{'='*60}")
        for pname, pg in graded.items():
            tag = {'HIT': 'HIT ', 'MISS': 'MISS', 'DNP': 'DNP '}.get(pg['result'], '????')
            print(f"\n  [{tag}] {pg['name']} ({pg['legs_hit']}/{pg['legs_total']} legs)")
            for lr in pg.get('leg_results', []):
                res_tag = {'HIT': 'v', 'MISS': 'x', 'DNP': '?'}.get(lr['result'], ' ')
                actual = lr.get('actual')
                actual_str = f"{actual}" if actual is not None else 'DNP'
                print(f"    [{res_tag}] {lr['player']:22s} {lr['stat'].upper():4s} "
                      f"{lr['direction']:5s} {lr['line']:5.1f}  actual={actual_str}")

        # Save engine graded
        engine_graded_file = os.path.join(PREDICTIONS_DIR, date_str, 'engine_parlays_graded.json')
        with open(engine_graded_file, 'w') as f:
            json.dump({
                'date': date_str,
                'graded_at': datetime.now().isoformat(),
                'parlays': graded,
            }, f, indent=2)
        print(f"\n  Saved: {engine_graded_file}")

    return graded


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 grade_primary_parlays.py YYYY-MM-DD")
        sys.exit(1)
    grade_primary_parlays(sys.argv[1])
    grade_engine_backup(sys.argv[1])

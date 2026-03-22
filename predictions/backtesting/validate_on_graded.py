#!/usr/bin/env python3
"""
Validate parlay filters on REAL graded data only.
No backfill, no synthetic lines — only actual sportsbook lines + actual box scores.

Usage:
    python3 predictions/backtesting/validate_on_graded.py
"""

import json
import os
import sys
import glob
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_graded_data():
    """Load all graded data with full pipeline features."""
    all_records = []

    # Source 1: Graded day folders (real sportsbook lines + real outcomes)
    for day_dir in sorted(glob.glob(os.path.join(BASE, '2026-*'))):
        date = os.path.basename(day_dir)
        # Look for results files with full features
        for fname in os.listdir(day_dir):
            if fname.endswith('.json') and ('result' in fname.lower() or 'board' in fname.lower()):
                fp = os.path.join(day_dir, fname)
                try:
                    with open(fp) as f:
                        data = json.load(f)
                except:
                    continue

                records = []
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    for key in ['picks', 'results', 'predictions']:
                        if key in data and isinstance(data[key], list):
                            records = data[key]
                            break

                for r in records:
                    if not isinstance(r, dict):
                        continue
                    if r.get('actual') is None and r.get('actual_val') is None and r.get('_actual') is None:
                        continue
                    # Compute hit from actual
                    actual = r.get('actual_val') or r.get('_actual') or r.get('actual')
                    if isinstance(actual, str):
                        # Parse "15 AST" format
                        try:
                            actual = float(actual.split()[0])
                        except:
                            continue
                    if actual is None:
                        continue

                    line = r.get('line', 0) or 0
                    direction = (r.get('direction', '') or '').upper()

                    if direction == 'OVER':
                        hit = actual > line
                    elif direction == 'UNDER':
                        hit = actual < line
                    else:
                        hit = r.get('hit', False)

                    r['_computed_hit'] = hit
                    r['_date'] = date
                    r['_actual_val'] = actual
                    all_records.append(r)

    # Source 2: Focused training data (real sportsbook lines applied retroactively)
    focused_path = os.path.join(BASE, 'cache', 'focused_training_data.json')
    if os.path.exists(focused_path):
        with open(focused_path) as f:
            focused = json.load(f)
        for r in focused:
            actual = r.get('_actual')
            if actual is None:
                continue
            line = r.get('line', 0) or 0
            direction = (r.get('direction', '') or '').upper()
            if direction == 'OVER':
                hit = actual > line
            elif direction == 'UNDER':
                hit = actual < line
            else:
                hit = bool(r.get('_hit_label', False))
            r['_computed_hit'] = hit
            r['_date'] = r.get('_date', '')
            r['_actual_val'] = actual
            all_records.append(r)

    print(f"Loaded {len(all_records):,} graded records")
    dates = sorted(set(r.get('_date', '') for r in all_records if r.get('_date')))
    print(f"  Dates: {len(dates)} ({dates[0] if dates else '?'} to {dates[-1] if dates else '?'})")
    hr = sum(1 for r in all_records if r['_computed_hit']) / len(all_records) if all_records else 0
    print(f"  Overall HR: {hr:.1%}")
    return all_records


def test_filter(records, name, filter_fn, max_parlays=5000):
    """Test a filter on graded data. Returns stats dict."""
    eligible = [r for r in records if filter_fn(r)]
    if not eligible:
        return None

    ind_hr = sum(1 for r in eligible if r['_computed_hit']) / len(eligible)

    # Group by date for parlays
    by_date = defaultdict(list)
    for r in eligible:
        by_date[r.get('_date', '')].append(r)

    total_2w, total_2t, total_3w, total_3t = 0, 0, 0, 0
    for date, cands in by_date.items():
        n = len(cands)
        if n >= 2:
            pairs = list(combinations(range(n), 2))
            if len(pairs) > max_parlays:
                pairs = random.sample(pairs, max_parlays)
            for c in pairs:
                if cands[c[0]].get('player') != cands[c[1]].get('player'):
                    total_2t += 1
                    if cands[c[0]]['_computed_hit'] and cands[c[1]]['_computed_hit']:
                        total_2w += 1
        if n >= 3:
            triples = list(combinations(range(n), 3))
            if len(triples) > max_parlays:
                triples = random.sample(triples, max_parlays)
            for c in triples:
                ps = [cands[c[k]].get('player') for k in range(3)]
                if len(set(ps)) == 3:
                    total_3t += 1
                    if all(cands[c[k]]['_computed_hit'] for k in range(3)):
                        total_3w += 1

    return {
        'name': name,
        'eligible': len(eligible),
        'ind_hr': ind_hr,
        'days': len(by_date),
        'wr_2leg': total_2w / total_2t if total_2t else 0,
        'wr_3leg': total_3w / total_3t if total_3t else 0,
        'total_2leg': total_2t,
        'total_3leg': total_3t,
    }


def main():
    records = load_graded_data()
    if not records:
        print("No graded data found!")
        return

    # Define filters to test
    filters = [
        ("Baseline (all picks)", lambda r: True),
        ("UNDER only", lambda r: (r.get('direction', '') or '').upper() == 'UNDER'),
        ("UNDER + COLD", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and r.get('streak_status') == 'COLD'),
        ("UNDER + line>L10+2", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         ((r.get('line', 0) or 0) - (r.get('l10_avg', 0) or 0)) >= 2.0),
        ("UNDER + line>L10+3", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         ((r.get('line', 0) or 0) - (r.get('l10_avg', 0) or 0)) >= 3.0),
        ("UNDER + line>L10+2 + COLD", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         r.get('streak_status') == 'COLD' and
         ((r.get('line', 0) or 0) - (r.get('l10_avg', 0) or 0)) >= 2.0),
        ("UNDER + line>L10+3 + COLD", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         r.get('streak_status') == 'COLD' and
         ((r.get('line', 0) or 0) - (r.get('l10_avg', 0) or 0)) >= 3.0),
        ("UNDER + line>L10+5", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         ((r.get('line', 0) or 0) - (r.get('l10_avg', 0) or 0)) >= 5.0),
        ("UNDER + gap>=5", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         abs(float(r.get('gap', 0) or r.get('abs_gap', 0) or 0)) >= 5),
        ("UNDER + gap>=7", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         abs(float(r.get('gap', 0) or r.get('abs_gap', 0) or 0)) >= 7),
        ("UNDER + reg_margin<-3", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         (r.get('reg_margin', 0) or 0) < -3),
        ("UNDER + ensemble>0.55", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         (r.get('ensemble_prob', r.get('xgb_prob', 0)) or 0) > 0.55),
        ("NOT HOT", lambda r: r.get('streak_status') != 'HOT'),
        ("UNDER + NOT HOT + line>L10+2", lambda r: (r.get('direction', '') or '').upper() == 'UNDER' and
         r.get('streak_status') != 'HOT' and
         ((r.get('line', 0) or 0) - (r.get('l10_avg', 0) or 0)) >= 2.0),
    ]

    print(f"\n{'='*80}")
    print(f"  FILTER VALIDATION ON REAL GRADED DATA")
    print(f"{'='*80}")
    print(f"{'Filter':<42} {'HR':>6} {'2-leg':>7} {'3-leg':>7} {'Picks':>6} {'Days':>5}")
    print(f"{'-'*80}")

    results = []
    for name, filt in filters:
        r = test_filter(records, name, filt)
        if r and r['eligible'] >= 5:
            m2 = '✅' if r['wr_2leg'] >= 0.80 else '  '
            m3 = '✅' if r['wr_3leg'] >= 0.80 else '  '
            print(f"{r['name']:<42} {r['ind_hr']:>5.1%} {r['wr_2leg']:>5.1%}{m2} {r['wr_3leg']:>5.1%}{m3} {r['eligible']:>6} {r['days']:>5}")
            results.append(r)

    # Save results
    output_path = os.path.join(BASE, 'backtesting', 'graded_validation_results.json')
    with open(output_path, 'w') as f:
        json.dump({'results': results}, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()

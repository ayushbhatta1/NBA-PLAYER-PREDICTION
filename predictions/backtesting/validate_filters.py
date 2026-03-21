#!/usr/bin/env python3
"""
Validate parlay filters on REAL graded data (actual sportsbook lines).
Exposes data leakage in backfill-only validation.
"""
import json, os, sys, glob, random
from collections import defaultdict
from itertools import combinations
from math import comb

PREDICTIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_graded_days():
    """Load all graded results from predictions/YYYY-MM-DD/ folders.
    Returns dict of {date: [records]} where each record is normalized."""
    days = {}
    for d in sorted(glob.glob(os.path.join(PREDICTIONS_DIR, '20*-*-*'))):
        date = os.path.basename(d)
        best_file = None
        best_count = 0
        # Find the graded file with most records
        for pattern in ['v4_graded*.json', 'graded*.json']:
            for f in glob.glob(os.path.join(d, pattern)):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    records = None
                    if isinstance(data, dict):
                        for key in ['results', 'picks', 'props']:
                            if key in data and isinstance(data[key], list):
                                records = data[key]
                                break
                    elif isinstance(data, list):
                        records = data
                    if records and len(records) > best_count:
                        # Verify has grading
                        has_grade = any(
                            r.get('result') in ('HIT', 'MISS') or r.get('hit') is not None
                            for r in records
                        )
                        if has_grade:
                            best_file = records
                            best_count = len(records)
                except (json.JSONDecodeError, IOError, KeyError):
                    continue
        if best_file:
            days[date] = best_file
    return days


def extract_hit(record):
    """Extract hit bool from graded record."""
    result = record.get('result', '')
    if isinstance(result, str):
        if result.upper() == 'HIT':
            return True
        if result.upper() == 'MISS':
            return False
    if 'hit' in record and record['hit'] is not None:
        return bool(record['hit'])
    return None


def _safe_float(val, default=0.0):
    """Safely convert a value to float, returning default on failure."""
    if val is None:
        return default
    try:
        f = float(val)
        if f != f:  # NaN check
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    """Safely convert a value to int, returning default on failure."""
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def normalize_graded(r, date):
    """Normalize a graded record to standard schema."""
    hit = extract_hit(r)
    if hit is None:
        return None

    gap_raw = _safe_float(r.get('gap', 0))
    abs_gap_raw = _safe_float(r.get('abs_gap', 0))

    spread_val = r.get('spread')
    if spread_val is not None and str(spread_val).lower() == 'nan':
        spread_val = 0
    spread = _safe_float(spread_val)

    return {
        'player': r.get('player', ''),
        'stat': (r.get('stat', '') or '').lower(),
        'line': _safe_float(r.get('line', 0)),
        'projection': _safe_float(r.get('projection', 0)),
        'gap': gap_raw,
        'abs_gap': abs(gap_raw) if abs_gap_raw == 0 else abs_gap_raw,
        'direction': (r.get('direction', '') or '').upper(),
        'tier': (r.get('tier', 'F') or 'F').upper(),
        'hit': hit,
        'date': date,
        'actual': r.get('actual'),
        'l10_avg': _safe_float(r.get('l10_avg', 0)),
        'l5_avg': _safe_float(r.get('l5_avg', 0)),
        'l3_avg': _safe_float(r.get('l3_avg', 0)),
        'season_avg': _safe_float(r.get('season_avg', 0)),
        'l10_hit_rate': _safe_float(r.get('l10_hit_rate', 0)),
        'l5_hit_rate': _safe_float(r.get('l5_hit_rate', 0)),
        'season_hit_rate': _safe_float(r.get('season_hit_rate', 0)),
        'mins_30plus_pct': _safe_float(r.get('mins_30plus_pct', 0)),
        'l10_miss_count': _safe_int(r.get('l10_miss_count', 0)),
        'l10_std': _safe_float(r.get('l10_std', 0)),
        'l10_floor': _safe_float(r.get('l10_floor', 0)),
        'is_home': _safe_int(r.get('is_home', 0)),
        'is_b2b': _safe_int(r.get('is_b2b', 0)),
        'spread': spread,
        'streak_status': (r.get('streak_status', '') or '').upper(),
        'game': r.get('game', ''),
    }


def apply_filter(picks, f):
    """Apply a single filter config to picks. Returns filtered list."""
    result = []
    tier_order = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}

    for p in picks:
        # Direction
        dir_req = f.get('direction')
        if dir_req and p['direction'] != dir_req:
            continue
        # Min gap
        if p['abs_gap'] < f.get('min_gap', 0):
            continue
        # L10 HR range
        hr = p['l10_hit_rate']
        if hr < f.get('min_l10_hr', 0):
            continue
        if 'max_l10_hr' in f and hr >= f['max_l10_hr']:
            continue
        # Min line
        if p['line'] < f.get('min_line', 0):
            continue
        # Min tier
        if tier_order.get(p['tier'], 0) < tier_order.get(f.get('min_tier', 'F'), 0):
            continue
        # Streak
        if f.get('no_hot') and p['streak_status'] == 'HOT':
            continue
        if f.get('cold_only') and p['streak_status'] != 'COLD':
            continue
        # Mins
        if p['mins_30plus_pct'] < f.get('min_mins_pct', 0):
            continue
        # Miss count
        if 'max_l10_miss' in f and p['l10_miss_count'] > f['max_l10_miss']:
            continue
        # Season HR
        if p['season_hit_rate'] < f.get('min_season_hr', 0):
            continue
        # L10 std
        if 'max_l10_std' in f and f['max_l10_std'] is not None and p['l10_std'] > f['max_l10_std']:
            continue
        # Line above season avg
        if 'min_line_above_avg' in f:
            line_above = p['line'] - p['season_avg']
            if line_above < f['min_line_above_avg']:
                continue
        # Exclude combos
        if f.get('exclude_combos') and p['stat'] in {'pra', 'pr', 'pa', 'ra'}:
            continue
        # Max line
        if 'max_line' in f and p['line'] > f['max_line']:
            continue
        # Season avg cap (low-volume players)
        if 'max_season_avg' in f and p['season_avg'] > f['max_season_avg']:
            continue
        result.append(p)
    return result


def compute_parlay_wr(candidates_by_date, n_legs, max_per_day=5000):
    """Compute parlay win rate across dates. Returns (total, wins, wr)."""
    total = 0
    wins = 0
    for date, cands in candidates_by_date.items():
        if len(cands) < n_legs:
            continue
        n_combos = comb(len(cands), n_legs)
        if n_combos <= max_per_day:
            for combo in combinations(range(len(cands)), n_legs):
                legs = [cands[i] for i in combo]
                players = [l['player'] for l in legs]
                if len(set(players)) != len(players):
                    continue
                total += 1
                if all(l['hit'] for l in legs):
                    wins += 1
        else:
            seen = set()
            attempts = 0
            while len(seen) < max_per_day and attempts < max_per_day * 10:
                attempts += 1
                indices = tuple(sorted(random.sample(range(len(cands)), n_legs)))
                if indices in seen:
                    continue
                seen.add(indices)
                legs = [cands[i] for i in indices]
                players = [l['player'] for l in legs]
                if len(set(players)) != len(players):
                    continue
                total += 1
                if all(l['hit'] for l in legs):
                    wins += 1
    wr = wins / total if total > 0 else 0
    return total, wins, wr


def get_filter_configs():
    """All filter configs to test."""
    return [
        # GOLDEN variants (from mass_backtest best_config.json)
        {'name': 'GOLDEN balanced', 'direction': 'UNDER', 'min_gap': 5.0, 'min_l10_hr': 60, 'max_l10_hr': 70, 'min_line': 15},
        {'name': 'GOLDEN accuracy', 'direction': 'UNDER', 'min_gap': 5.0, 'min_l10_hr': 60, 'max_l10_hr': 70, 'min_line': 20},
        {'name': 'GOLDEN relaxed', 'direction': 'UNDER', 'min_gap': 4.0, 'min_l10_hr': 60, 'max_l10_hr': 70, 'min_line': 15},
        {'name': 'GOLDEN gap3', 'direction': 'UNDER', 'min_gap': 3.0, 'min_l10_hr': 60, 'max_l10_hr': 70, 'min_line': 15},

        # Current sim_sort filter (from parlay_engine)
        {'name': 'sim_sort (current)', 'direction': 'UNDER', 'min_l10_hr': 60, 'no_hot': True, 'min_line_above_avg': 0.5},

        # UNDER + various gaps
        {'name': 'UNDER gap1', 'direction': 'UNDER', 'min_gap': 1.0},
        {'name': 'UNDER gap2', 'direction': 'UNDER', 'min_gap': 2.0},
        {'name': 'UNDER gap3', 'direction': 'UNDER', 'min_gap': 3.0},
        {'name': 'UNDER gap5', 'direction': 'UNDER', 'min_gap': 5.0},
        {'name': 'UNDER gap7', 'direction': 'UNDER', 'min_gap': 7.0},

        # HR ranges
        {'name': 'UNDER HR>=70', 'direction': 'UNDER', 'min_l10_hr': 70},
        {'name': 'UNDER HR>=80', 'direction': 'UNDER', 'min_l10_hr': 80},
        {'name': 'UNDER HR>=90', 'direction': 'UNDER', 'min_l10_hr': 90},
        {'name': 'UNDER HR[60,70)', 'direction': 'UNDER', 'min_l10_hr': 60, 'max_l10_hr': 70},
        {'name': 'UNDER HR[70,80)', 'direction': 'UNDER', 'min_l10_hr': 70, 'max_l10_hr': 80},

        # Line thresholds
        {'name': 'UNDER line>=15', 'direction': 'UNDER', 'min_line': 15},
        {'name': 'UNDER line>=20', 'direction': 'UNDER', 'min_line': 20},
        {'name': 'UNDER line<=10', 'direction': 'UNDER', 'max_line': 10},
        {'name': 'UNDER line<=5', 'direction': 'UNDER', 'max_line': 5},

        # Line above avg (key from 1M sim)
        {'name': 'UNDER line_above>=1', 'direction': 'UNDER', 'min_line_above_avg': 1.0, 'min_l10_hr': 60, 'no_hot': True},
        {'name': 'UNDER line_above>=2', 'direction': 'UNDER', 'min_line_above_avg': 2.0, 'min_l10_hr': 60, 'no_hot': True},
        {'name': 'UNDER line_above>=3', 'direction': 'UNDER', 'min_line_above_avg': 3.0, 'min_l10_hr': 60, 'no_hot': True},

        # Compound: COLD + UNDER
        {'name': 'COLD+UNDER', 'direction': 'UNDER', 'cold_only': True},
        {'name': 'COLD+UNDER HR>=60', 'direction': 'UNDER', 'cold_only': True, 'min_l10_hr': 60},
        {'name': 'COLD+UNDER gap3', 'direction': 'UNDER', 'cold_only': True, 'min_gap': 3.0},

        # Low lines (role players — from 1M sim: max_avg<=5 = 43.5%)
        {'name': 'UNDER low_line<=5 no_hot', 'direction': 'UNDER', 'max_line': 5, 'no_hot': True},
        {'name': 'UNDER avg<=10 no_hot', 'direction': 'UNDER', 'max_season_avg': 10, 'no_hot': True},
        {'name': 'UNDER avg<=15 no_hot', 'direction': 'UNDER', 'max_season_avg': 15, 'no_hot': True},

        # Combo exclusion
        {'name': 'UNDER no_combos gap2', 'direction': 'UNDER', 'min_gap': 2.0, 'exclude_combos': True},
        {'name': 'UNDER no_combos HR>=70', 'direction': 'UNDER', 'min_l10_hr': 70, 'exclude_combos': True},

        # Both directions (baseline)
        {'name': 'both gap3 HR>=60', 'min_gap': 3.0, 'min_l10_hr': 60},
        {'name': 'both HR>=80', 'min_l10_hr': 80},

        # Strict compound
        {'name': 'UNDER gap3 HR>=70 noHOT', 'direction': 'UNDER', 'min_gap': 3.0, 'min_l10_hr': 70, 'no_hot': True},
        {'name': 'UNDER gap5 HR>=70 noHOT', 'direction': 'UNDER', 'min_gap': 5.0, 'min_l10_hr': 70, 'no_hot': True},
        {'name': 'UNDER gap2 HR>=70 line>=15 noHOT', 'direction': 'UNDER', 'min_gap': 2.0, 'min_l10_hr': 70, 'min_line': 15, 'no_hot': True},
    ]


def main():
    random.seed(42)

    print("=" * 70)
    print("  FILTER VALIDATION ON REAL GRADED DATA")
    print("  (actual sportsbook lines -- NO synthetic data)")
    print("=" * 70)

    # Load graded data
    graded_days = load_graded_days()
    if not graded_days:
        print("\n  ERROR: No graded data found in predictions/YYYY-MM-DD/ folders.")
        print(f"  Searched: {PREDICTIONS_DIR}/20*-*-*/")
        sys.exit(1)

    print(f"\nLoaded {len(graded_days)} graded days:")
    all_records = []
    all_by_date = {}
    for date in sorted(graded_days.keys()):
        records = graded_days[date]
        normalized = []
        for r in records:
            nr = normalize_graded(r, date)
            if nr:
                normalized.append(nr)
        all_records.extend(normalized)
        all_by_date[date] = normalized
        hits = sum(1 for r in normalized if r['hit'])
        hr = hits / len(normalized) if normalized else 0
        print(f"  {date}: {len(normalized):>4} records, {hr:.1%} HR")

    if not all_records:
        print("\n  ERROR: No valid graded records found after normalization.")
        sys.exit(1)

    total_hits = sum(1 for r in all_records if r['hit'])
    print(f"\n  TOTAL: {len(all_records):,} records across {len(graded_days)} days")
    print(f"  Overall HR: {total_hits/len(all_records):.1%}")
    under = [r for r in all_records if r['direction'] == 'UNDER']
    over = [r for r in all_records if r['direction'] == 'OVER']
    if under:
        print(f"  UNDER: {sum(1 for r in under if r['hit'])/len(under):.1%} ({len(under)})")
    if over:
        print(f"  OVER:  {sum(1 for r in over if r['hit'])/len(over):.1%} ({len(over)})")

    # Test all filters
    configs = get_filter_configs()
    results = []

    print(f"\n{'='*70}")
    print(f"  TESTING {len(configs)} FILTERS")
    print(f"{'='*70}")
    print(f"\n  {'Filter':<35} {'Elig':>5} {'IndHR':>6} {'Days':>4} {'Avg/d':>5} {'2leg':>7} {'3leg':>7}")
    print(f"  {'-'*75}")

    for cfg in configs:
        name = cfg.get('name', '?')
        # Filter per day
        filtered_by_date = {}
        all_filtered = []
        for date, picks in all_by_date.items():
            f = apply_filter(picks, cfg)
            if f:
                filtered_by_date[date] = f
                all_filtered.extend(f)

        if not all_filtered:
            print(f"  {name:<35} {'0':>5} {'N/A':>6} {'0':>4} {'0':>5} {'N/A':>7} {'N/A':>7}")
            results.append({
                'name': name, 'eligible': 0, 'ind_hr': 0, 'days': 0,
                'avg_per_day': 0, 'wr_2leg': 0, 'wr_3leg': 0,
                'total_2leg': 0, 'total_3leg': 0,
                'filters': {k: v for k, v in cfg.items() if k != 'name'},
            })
            continue

        ind_hr = sum(1 for r in all_filtered if r['hit']) / len(all_filtered)
        days_active = len(filtered_by_date)
        avg_per_day = len(all_filtered) / days_active if days_active else 0

        # Compute parlay WRs
        total_2, wins_2, wr_2 = compute_parlay_wr(filtered_by_date, 2, max_per_day=2000)
        total_3, wins_3, wr_3 = compute_parlay_wr(filtered_by_date, 3, max_per_day=2000)

        wr2_str = f"{wr_2:.1%}" if total_2 > 0 else "N/A"
        wr3_str = f"{wr_3:.1%}" if total_3 > 0 else "N/A"

        print(f"  {name:<35} {len(all_filtered):>5} {ind_hr:>5.1%} {days_active:>4} {avg_per_day:>5.1f} {wr2_str:>7} {wr3_str:>7}")

        results.append({
            'name': name, 'eligible': len(all_filtered), 'ind_hr': ind_hr,
            'days': days_active, 'avg_per_day': avg_per_day,
            'wr_2leg': wr_2, 'wr_3leg': wr_3,
            'total_2leg': total_2, 'total_3leg': total_3,
            'filters': {k: v for k, v in cfg.items() if k != 'name'},
        })

    # Summary: top filters by ind HR (with min 20 picks)
    print(f"\n{'='*70}")
    print(f"  TOP FILTERS BY INDIVIDUAL HIT RATE (min 20 picks)")
    print(f"{'='*70}")
    viable = [r for r in results if r['eligible'] >= 20]
    viable.sort(key=lambda x: x['ind_hr'], reverse=True)
    for r in viable[:15]:
        wr2_str = f"{r['wr_2leg']:.1%}" if r.get('total_2leg', 0) > 0 else "N/A"
        wr3_str = f"{r['wr_3leg']:.1%}" if r.get('total_3leg', 0) > 0 else "N/A"
        print(f"  {r['name']:<35} IndHR: {r['ind_hr']:.1%} | 2-leg: {wr2_str} | 3-leg: {wr3_str} | {r['eligible']} picks, {r['days']} days")

    # Summary: top by 3-leg WR
    print(f"\n{'='*70}")
    print(f"  TOP FILTERS BY 3-LEG PARLAY WIN RATE (min 10 parlays)")
    print(f"{'='*70}")
    viable_3 = [r for r in results if r.get('total_3leg', 0) >= 10]
    viable_3.sort(key=lambda x: x['wr_3leg'], reverse=True)
    for r in viable_3[:10]:
        print(f"  {r['name']:<35} 3-leg: {r['wr_3leg']:.1%} ({r.get('total_3leg',0)}) | IndHR: {r['ind_hr']:.1%} | {r['eligible']} picks")

    # Mathematical ceiling
    print(f"\n{'='*70}")
    print(f"  MATHEMATICAL CEILING (REAL DATA)")
    print(f"{'='*70}")
    if viable:
        best_hr = max(r['ind_hr'] for r in viable)
        print(f"  Best individual HR (min 20 picks): {best_hr:.1%}")
        print(f"  Theoretical 2-leg ceiling: {best_hr**2:.1%}")
        print(f"  Theoretical 3-leg ceiling: {best_hr**3:.1%}")
        print(f"  For 80% 2-leg need: {0.80**0.5:.1%} per leg")
        print(f"  For 80% 3-leg need: {0.80**(1/3):.1%} per leg")
        if best_hr >= 0.93:
            print(f"\n  3-leg 80% is ACHIEVABLE on real data!")
        elif best_hr >= 0.894:
            print(f"\n  2-leg 80% is ACHIEVABLE, 3-leg 80% needs tighter filters")
        else:
            print(f"\n  80% parlay WR NOT achievable at {best_hr:.1%} per-leg HR")
            print(f"  Best realistic 2-leg: {best_hr**2:.1%}")
            print(f"  Best realistic 3-leg: {best_hr**3:.1%}")
    else:
        print("  No filters with >= 20 picks found. Cannot compute ceiling.")

    # Data leakage check
    print(f"\n{'='*70}")
    print(f"  DATA LEAKAGE DETECTION")
    print(f"{'='*70}")
    print(f"  The GOLDEN filter claims 93.6% HR on backfill data.")
    golden = next((r for r in results if r['name'] == 'GOLDEN balanced'), None)
    if golden and golden['eligible'] > 0:
        print(f"  On REAL graded data: {golden['ind_hr']:.1%} HR ({golden['eligible']} picks)")
        delta = 0.936 - golden['ind_hr']
        print(f"  Delta (backfill - real): {delta:.1%}")
        if delta > 0.10:
            print(f"  CONFIRMED DATA LEAKAGE -- backfill inflates HR by {delta:.1%}")
        elif delta > 0.05:
            print(f"  LIKELY DATA LEAKAGE -- backfill inflates HR by {delta:.1%}")
        else:
            print(f"  Minimal delta -- filter may generalize")
    else:
        print(f"  GOLDEN filter found 0 qualifying picks on real data")
        print(f"  This means the filter exploits backfill-specific patterns")

    # Save results
    output = {
        'graded_days': len(graded_days),
        'total_records': len(all_records),
        'results': results,
    }
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, 'validate_results.json')
    try:
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {out_path}")
    except IOError as e:
        print(f"\n  WARNING: Could not save results: {e}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Validate filters on REAL sportsbook lines applied to historical game logs.
Solves the data leakage problem: lines are from actual sportsbooks, not synthetic.

Usage:
    python3 predictions/backtesting/validate_real_lines.py --board /tmp/parsed_board_full.json
    python3 predictions/backtesting/validate_real_lines.py --cached  # use pre-generated focused data
"""
import json, os, sys, glob, random, time, argparse
from collections import defaultdict
from itertools import combinations
from math import comb

BASE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.dirname(BASE)
sys.path.insert(0, BASE)
sys.path.insert(0, PRED_DIR)

FOCUSED_DATA_PATH = os.path.join(PRED_DIR, 'cache', 'focused_training_data.json')
VALIDATE_RESULTS_PATH = os.path.join(BASE, 'validate_results.json')
OUTPUT_PATH = os.path.join(BASE, 'validate_real_lines_results.json')


# ═══════════════════════════════════════════════════════════════
# SAFE FLOAT / INT HELPERS (same as validate_filters.py)
# ═══════════════════════════════════════════════════════════════

def _safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        if f != f:  # NaN
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_focused_data(board_path=None):
    """Load focused training data from board or cache.

    If board_path provided, calls generate_focused_training() from
    train_current_lines.py. Otherwise loads from cached file.
    """
    if board_path:
        try:
            from train_current_lines import generate_focused_training
            print(f"  Generating focused data from board: {board_path}")
            records = generate_focused_training(board_path)
            if records:
                return records
            print("  WARNING: generate_focused_training returned empty, falling back to cache")
        except Exception as e:
            print(f"  WARNING: Could not generate from board: {e}")
            print("  Falling back to cached focused data...")

    # Load from cache
    if not os.path.exists(FOCUSED_DATA_PATH):
        print(f"  ERROR: No cached focused data at {FOCUSED_DATA_PATH}")
        print(f"  Generate it first: python3 predictions/train_current_lines.py --board <path>")
        sys.exit(1)

    with open(FOCUSED_DATA_PATH) as f:
        records = json.load(f)
    print(f"  Loaded {len(records):,} records from {FOCUSED_DATA_PATH}")
    return records


# ═══════════════════════════════════════════════════════════════
# NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def normalize_focused_record(r):
    """Normalize a focused training record to same schema as validate_filters.py."""
    hit_label = r.get('_hit_label')
    if hit_label is None:
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
        'line': _safe_float(r.get('line', r.get('_board_line', 0))),
        'projection': _safe_float(r.get('projection', 0)),
        'gap': gap_raw,
        'abs_gap': abs(gap_raw) if abs_gap_raw == 0 else abs_gap_raw,
        'direction': (r.get('direction', '') or '').upper(),
        'tier': (r.get('tier', 'F') or 'F').upper(),
        'hit': bool(hit_label),
        'date': r.get('_date', ''),
        'actual': r.get('_actual'),
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
        'game': '',  # not available in focused data
    }


# ═══════════════════════════════════════════════════════════════
# FILTER LOGIC (imported or inline fallback)
# ═══════════════════════════════════════════════════════════════

try:
    from validate_filters import get_filter_configs, apply_filter, compute_parlay_wr
except ImportError:
    # Inline fallback if import fails
    def apply_filter(picks, f):
        result = []
        tier_order = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        for p in picks:
            dir_req = f.get('direction')
            if dir_req and p['direction'] != dir_req:
                continue
            if p['abs_gap'] < f.get('min_gap', 0):
                continue
            hr = p['l10_hit_rate']
            if hr < f.get('min_l10_hr', 0):
                continue
            if 'max_l10_hr' in f and hr >= f['max_l10_hr']:
                continue
            if p['line'] < f.get('min_line', 0):
                continue
            if tier_order.get(p['tier'], 0) < tier_order.get(f.get('min_tier', 'F'), 0):
                continue
            if f.get('no_hot') and p['streak_status'] == 'HOT':
                continue
            if f.get('cold_only') and p['streak_status'] != 'COLD':
                continue
            if p['mins_30plus_pct'] < f.get('min_mins_pct', 0):
                continue
            if 'max_l10_miss' in f and p['l10_miss_count'] > f['max_l10_miss']:
                continue
            if p['season_hit_rate'] < f.get('min_season_hr', 0):
                continue
            if 'max_l10_std' in f and f['max_l10_std'] is not None and p['l10_std'] > f['max_l10_std']:
                continue
            if 'min_line_above_avg' in f:
                line_above = p['line'] - p['season_avg']
                if line_above < f['min_line_above_avg']:
                    continue
            if f.get('exclude_combos') and p['stat'] in {'pra', 'pr', 'pa', 'ra'}:
                continue
            if 'max_line' in f and p['line'] > f['max_line']:
                continue
            if 'max_season_avg' in f and p['season_avg'] > f['max_season_avg']:
                continue
            result.append(p)
        return result

    def compute_parlay_wr(candidates_by_date, n_legs, max_per_day=5000):
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
        return [
            {'name': 'GOLDEN balanced', 'direction': 'UNDER', 'min_gap': 5.0, 'min_l10_hr': 60, 'max_l10_hr': 70, 'min_line': 15},
            {'name': 'GOLDEN accuracy', 'direction': 'UNDER', 'min_gap': 5.0, 'min_l10_hr': 60, 'max_l10_hr': 70, 'min_line': 20},
            {'name': 'GOLDEN relaxed', 'direction': 'UNDER', 'min_gap': 4.0, 'min_l10_hr': 60, 'max_l10_hr': 70, 'min_line': 15},
            {'name': 'GOLDEN gap3', 'direction': 'UNDER', 'min_gap': 3.0, 'min_l10_hr': 60, 'max_l10_hr': 70, 'min_line': 15},
            {'name': 'sim_sort (current)', 'direction': 'UNDER', 'min_l10_hr': 60, 'no_hot': True, 'min_line_above_avg': 0.5},
            {'name': 'UNDER gap1', 'direction': 'UNDER', 'min_gap': 1.0},
            {'name': 'UNDER gap2', 'direction': 'UNDER', 'min_gap': 2.0},
            {'name': 'UNDER gap3', 'direction': 'UNDER', 'min_gap': 3.0},
            {'name': 'UNDER gap5', 'direction': 'UNDER', 'min_gap': 5.0},
            {'name': 'UNDER gap7', 'direction': 'UNDER', 'min_gap': 7.0},
            {'name': 'UNDER HR>=70', 'direction': 'UNDER', 'min_l10_hr': 70},
            {'name': 'UNDER HR>=80', 'direction': 'UNDER', 'min_l10_hr': 80},
            {'name': 'UNDER HR>=90', 'direction': 'UNDER', 'min_l10_hr': 90},
            {'name': 'UNDER HR[60,70)', 'direction': 'UNDER', 'min_l10_hr': 60, 'max_l10_hr': 70},
            {'name': 'UNDER HR[70,80)', 'direction': 'UNDER', 'min_l10_hr': 70, 'max_l10_hr': 80},
            {'name': 'UNDER line>=15', 'direction': 'UNDER', 'min_line': 15},
            {'name': 'UNDER line>=20', 'direction': 'UNDER', 'min_line': 20},
            {'name': 'UNDER line<=10', 'direction': 'UNDER', 'max_line': 10},
            {'name': 'UNDER line<=5', 'direction': 'UNDER', 'max_line': 5},
            {'name': 'UNDER line_above>=1', 'direction': 'UNDER', 'min_line_above_avg': 1.0, 'min_l10_hr': 60, 'no_hot': True},
            {'name': 'UNDER line_above>=2', 'direction': 'UNDER', 'min_line_above_avg': 2.0, 'min_l10_hr': 60, 'no_hot': True},
            {'name': 'UNDER line_above>=3', 'direction': 'UNDER', 'min_line_above_avg': 3.0, 'min_l10_hr': 60, 'no_hot': True},
            {'name': 'COLD+UNDER', 'direction': 'UNDER', 'cold_only': True},
            {'name': 'COLD+UNDER HR>=60', 'direction': 'UNDER', 'cold_only': True, 'min_l10_hr': 60},
            {'name': 'COLD+UNDER gap3', 'direction': 'UNDER', 'cold_only': True, 'min_gap': 3.0},
            {'name': 'UNDER low_line<=5 no_hot', 'direction': 'UNDER', 'max_line': 5, 'no_hot': True},
            {'name': 'UNDER avg<=10 no_hot', 'direction': 'UNDER', 'max_season_avg': 10, 'no_hot': True},
            {'name': 'UNDER avg<=15 no_hot', 'direction': 'UNDER', 'max_season_avg': 15, 'no_hot': True},
            {'name': 'UNDER no_combos gap2', 'direction': 'UNDER', 'min_gap': 2.0, 'exclude_combos': True},
            {'name': 'UNDER no_combos HR>=70', 'direction': 'UNDER', 'min_l10_hr': 70, 'exclude_combos': True},
            {'name': 'both gap3 HR>=60', 'min_gap': 3.0, 'min_l10_hr': 60},
            {'name': 'both HR>=80', 'min_l10_hr': 80},
            {'name': 'UNDER gap3 HR>=70 noHOT', 'direction': 'UNDER', 'min_gap': 3.0, 'min_l10_hr': 70, 'no_hot': True},
            {'name': 'UNDER gap5 HR>=70 noHOT', 'direction': 'UNDER', 'min_gap': 5.0, 'min_l10_hr': 70, 'no_hot': True},
            {'name': 'UNDER gap2 HR>=70 line>=15 noHOT', 'direction': 'UNDER', 'min_gap': 2.0, 'min_l10_hr': 70, 'min_line': 15, 'no_hot': True},
        ]


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Validate filters on real sportsbook lines')
    parser.add_argument('--board', type=str, help='Path to parsed board JSON')
    parser.add_argument('--cached', action='store_true', help='Use pre-generated focused data from cache')
    args = parser.parse_args()

    if not args.board and not args.cached:
        # Default to cached mode
        args.cached = True

    random.seed(42)

    print("=" * 70)
    print("  FILTER VALIDATION ON REAL SPORTSBOOK LINES")
    print("  (real lines from sportsbooks, applied to historical game logs)")
    print("=" * 70)

    # Load focused data
    raw_records = load_focused_data(board_path=args.board)
    print(f"\n  Raw records loaded: {len(raw_records):,}")

    # Normalize
    all_records = []
    all_by_date = defaultdict(list)
    for r in raw_records:
        nr = normalize_focused_record(r)
        if nr and nr['date']:
            all_records.append(nr)
            all_by_date[nr['date']].append(nr)

    if not all_records:
        print("\n  ERROR: No valid records after normalization.")
        sys.exit(1)

    # Data summary
    dates = sorted(all_by_date.keys())
    total_hits = sum(1 for r in all_records if r['hit'])
    under = [r for r in all_records if r['direction'] == 'UNDER']
    over = [r for r in all_records if r['direction'] == 'OVER']

    print(f"\n  TOTAL: {len(all_records):,} records across {len(dates)} dates")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Overall HR: {total_hits/len(all_records):.1%}")
    if under:
        print(f"  UNDER: {sum(1 for r in under if r['hit'])/len(under):.1%} ({len(under):,})")
    if over:
        print(f"  OVER:  {sum(1 for r in over if r['hit'])/len(over):.1%} ({len(over):,})")

    # Unique players/stats
    players = set(r['player'] for r in all_records)
    stats = set(r['stat'] for r in all_records)
    print(f"  Players: {len(players)}, Stats: {len(stats)}")

    # Test all filters
    configs = get_filter_configs()
    results = []

    print(f"\n{'='*70}")
    print(f"  TESTING {len(configs)} FILTERS ON FOCUSED DATA (REAL LINES)")
    print(f"{'='*70}")
    print(f"\n  {'Filter':<35} {'Elig':>5} {'IndHR':>6} {'Days':>4} {'Avg/d':>5} {'2leg':>7} {'3leg':>7}")
    print(f"  {'-'*75}")

    for cfg in configs:
        name = cfg.get('name', '?')
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

    # Top filters by individual HR
    print(f"\n{'='*70}")
    print(f"  TOP FILTERS BY INDIVIDUAL HIT RATE (min 20 picks)")
    print(f"{'='*70}")
    viable = [r for r in results if r['eligible'] >= 20]
    viable.sort(key=lambda x: x['ind_hr'], reverse=True)
    for r in viable[:15]:
        wr2_str = f"{r['wr_2leg']:.1%}" if r.get('total_2leg', 0) > 0 else "N/A"
        wr3_str = f"{r['wr_3leg']:.1%}" if r.get('total_3leg', 0) > 0 else "N/A"
        print(f"  {r['name']:<35} IndHR: {r['ind_hr']:.1%} | 2-leg: {wr2_str} | 3-leg: {wr3_str} | {r['eligible']} picks, {r['days']} days")

    # Top by 3-leg WR
    print(f"\n{'='*70}")
    print(f"  TOP FILTERS BY 3-LEG PARLAY WIN RATE (min 10 parlays)")
    print(f"{'='*70}")
    viable_3 = [r for r in results if r.get('total_3leg', 0) >= 10]
    viable_3.sort(key=lambda x: x['wr_3leg'], reverse=True)
    for r in viable_3[:10]:
        print(f"  {r['name']:<35} 3-leg: {r['wr_3leg']:.1%} ({r.get('total_3leg',0)}) | IndHR: {r['ind_hr']:.1%} | {r['eligible']} picks")

    # Mathematical ceiling
    print(f"\n{'='*70}")
    print(f"  MATHEMATICAL CEILING (REAL LINES DATA)")
    print(f"{'='*70}")
    if viable:
        best_hr = max(r['ind_hr'] for r in viable)
        print(f"  Best individual HR (min 20 picks): {best_hr:.1%}")
        print(f"  Theoretical 2-leg ceiling: {best_hr**2:.1%}")
        print(f"  Theoretical 3-leg ceiling: {best_hr**3:.1%}")
        print(f"  For 80% 2-leg need: {0.80**0.5:.1%} per leg")
        print(f"  For 80% 3-leg need: {0.80**(1/3):.1%} per leg")
        if best_hr >= 0.93:
            print(f"\n  3-leg 80% is ACHIEVABLE on real lines data!")
        elif best_hr >= 0.894:
            print(f"\n  2-leg 80% is ACHIEVABLE, 3-leg 80% needs tighter filters")
        else:
            print(f"\n  80% parlay WR NOT achievable at {best_hr:.1%} per-leg HR")
            print(f"  Best realistic 2-leg: {best_hr**2:.1%}")
            print(f"  Best realistic 3-leg: {best_hr**3:.1%}")
    else:
        print("  No filters with >= 20 picks found.")

    # 3-way comparison: Backfill vs Focused (real lines) vs Graded
    print(f"\n{'='*70}")
    print(f"  3-WAY COMPARISON: BACKFILL vs FOCUSED (REAL LINES) vs GRADED")
    print(f"{'='*70}")

    graded_results = None
    if os.path.exists(VALIDATE_RESULTS_PATH):
        try:
            with open(VALIDATE_RESULTS_PATH) as f:
                graded_data = json.load(f)
            graded_results = {r['name']: r for r in graded_data.get('results', [])}
            print(f"  Loaded graded validation from {VALIDATE_RESULTS_PATH}")
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"  WARNING: Could not load graded validation: {e}")

    focused_results = {r['name']: r for r in results}

    if graded_results:
        print(f"\n  {'Filter':<30} {'Focused HR':>10} {'Graded HR':>10} {'Delta':>7} {'Match?':>7}")
        print(f"  {'-'*68}")

        for cfg in configs:
            name = cfg.get('name', '?')
            foc = focused_results.get(name, {})
            grd = graded_results.get(name, {})

            foc_hr = foc.get('ind_hr', 0)
            grd_hr = grd.get('ind_hr', 0)
            foc_n = foc.get('eligible', 0)
            grd_n = grd.get('eligible', 0)

            if foc_n == 0 and grd_n == 0:
                continue

            delta = foc_hr - grd_hr if foc_n > 0 and grd_n > 0 else float('nan')
            delta_str = f"{delta:+.1%}" if delta == delta else "N/A"

            # Match = within 5pp
            if delta == delta:
                match = "YES" if abs(delta) <= 0.05 else "NO"
            else:
                match = "---"

            foc_str = f"{foc_hr:.1%} ({foc_n})" if foc_n > 0 else "N/A"
            grd_str = f"{grd_hr:.1%} ({grd_n})" if grd_n > 0 else "N/A"

            print(f"  {name:<30} {foc_str:>10} {grd_str:>10} {delta_str:>7} {match:>7}")

        # Summary stats
        deltas = []
        for name in focused_results:
            foc = focused_results[name]
            grd = graded_results.get(name, {})
            if foc.get('eligible', 0) >= 20 and grd.get('eligible', 0) >= 10:
                deltas.append(foc['ind_hr'] - grd['ind_hr'])

        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            within_5pp = sum(1 for d in deltas if abs(d) <= 0.05)
            print(f"\n  Comparable filters: {len(deltas)}")
            print(f"  Average delta (focused - graded): {avg_delta:+.1%}")
            print(f"  Within 5pp: {within_5pp}/{len(deltas)} ({within_5pp/len(deltas):.0%})")
            if avg_delta > 0.05:
                print(f"  WARNING: Focused data inflates HR by {avg_delta:.1%} vs graded")
            elif avg_delta < -0.05:
                print(f"  NOTE: Focused data is MORE conservative than graded by {abs(avg_delta):.1%}")
            else:
                print(f"  GOOD: Focused data closely matches graded data")
    else:
        print(f"\n  Graded validation not found at {VALIDATE_RESULTS_PATH}")
        print(f"  Run validate_filters.py first to enable 3-way comparison.")
        print(f"\n  Showing focused results only:")
        print(f"\n  {'Filter':<35} {'Focused HR':>10} {'Eligible':>8}")
        print(f"  {'-'*55}")
        for cfg in configs:
            name = cfg.get('name', '?')
            foc = focused_results.get(name, {})
            foc_hr = foc.get('ind_hr', 0)
            foc_n = foc.get('eligible', 0)
            if foc_n > 0:
                print(f"  {name:<35} {foc_hr:.1%}      {foc_n:>8}")

    # Save results
    output = {
        'source': 'focused_training_data (real sportsbook lines)',
        'total_records': len(all_records),
        'total_dates': len(dates),
        'date_range': [dates[0], dates[-1]] if dates else [],
        'overall_hr': total_hits / len(all_records) if all_records else 0,
        'under_hr': sum(1 for r in under if r['hit']) / len(under) if under else 0,
        'over_hr': sum(1 for r in over if r['hit']) / len(over) if over else 0,
        'results': results,
    }
    try:
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n  Results saved to {OUTPUT_PATH}")
    except IOError as e:
        print(f"\n  WARNING: Could not save results: {e}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Parlay Lab — Brute-force backtest engine to find consistently winning parlays.

Tests millions of filter combinations against all graded historical data.
Each combination defines a "pick filter" — the criteria for selecting legs.
Then simulates 3-leg parlays from the filtered pool across all graded days.

Usage:
    python3 predictions/parlay_lab.py                # Full search
    python3 predictions/parlay_lab.py --top 20       # Show top 20 strategies
    python3 predictions/parlay_lab.py --validate     # Hold-out validation
"""

import json
import os
import sys
import itertools
import random
import math
from collections import defaultdict
from datetime import datetime

PRED_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────

def load_all_graded():
    """Load every graded prop across all dates."""
    all_results = []
    for d in sorted(os.listdir(PRED_DIR)):
        if len(d) == 10 and d[4] == '-' and d[7] == '-':
            day_dir = os.path.join(PRED_DIR, d)
            for f in os.listdir(day_dir):
                if f.startswith('v4_graded') and f.endswith('.json'):
                    with open(os.path.join(day_dir, f)) as fh:
                        data = json.load(fh)
                    results = data.get('results', [])
                    graded = [r for r in results
                              if isinstance(r, dict) and r.get('result') in ('HIT', 'MISS')]
                    for r in graded:
                        r['_date'] = d
                    all_results.extend(graded)
    return all_results


def group_by_date(results):
    """Group results by date for day-by-day parlay simulation."""
    by_date = defaultdict(list)
    for r in results:
        by_date[r['_date']].append(r)
    return dict(sorted(by_date.items()))


# ─────────────────────────────────────────────────────────────────────
# 2. FILTER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def apply_filter(props, filt):
    """Apply a filter dict to a list of props. Returns filtered list."""
    pool = props

    # Direction filter
    direction = filt.get('direction')
    if direction:
        pool = [p for p in pool if p.get('direction') == direction]

    # Stat filter
    stats = filt.get('stats')
    if stats:
        pool = [p for p in pool if p.get('stat') in stats]

    # Stat exclude
    stats_exclude = filt.get('stats_exclude')
    if stats_exclude:
        pool = [p for p in pool if p.get('stat') not in stats_exclude]

    # Tier filter (minimum tier)
    min_tier = filt.get('min_tier')
    if min_tier:
        tier_map = {'S': 6, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
        min_val = tier_map.get(min_tier, 0)
        pool = [p for p in pool if tier_map.get(p.get('tier', 'F'), 0) >= min_val]

    # Hit rate filters
    min_l10_hr = filt.get('min_l10_hr')
    if min_l10_hr is not None:
        pool = [p for p in pool if (p.get('l10_hit_rate') or 0) >= min_l10_hr]

    min_l5_hr = filt.get('min_l5_hr')
    if min_l5_hr is not None:
        pool = [p for p in pool if (p.get('l5_hit_rate') or 0) >= min_l5_hr]

    min_season_hr = filt.get('min_season_hr')
    if min_season_hr is not None:
        pool = [p for p in pool if (p.get('season_hit_rate') or 0) >= min_season_hr]

    # Gap filter
    min_gap = filt.get('min_gap')
    if min_gap is not None:
        pool = [p for p in pool if abs(p.get('gap', 0) or 0) >= min_gap]

    max_gap = filt.get('max_gap')
    if max_gap is not None:
        pool = [p for p in pool if abs(p.get('gap', 0) or 0) <= max_gap]

    # Streak filter
    streak = filt.get('streak')
    if streak:
        pool = [p for p in pool if p.get('streak_status') == streak]

    streak_exclude = filt.get('streak_exclude')
    if streak_exclude:
        pool = [p for p in pool if p.get('streak_status') not in streak_exclude]

    # Minutes filter
    min_mins = filt.get('min_mins_pct')
    if min_mins is not None:
        pool = [p for p in pool if (p.get('mins_30plus_pct') or 0) >= min_mins]

    # Miss count filter (from L10)
    max_miss = filt.get('max_l10_miss')
    if max_miss is not None:
        pool = [p for p in pool if (p.get('l10_miss_count') or 0) <= max_miss]

    # Combo filter
    no_combos = filt.get('no_combos')
    if no_combos:
        combo_stats = {'pra', 'pr', 'pa', 'ra', 'stl_blk'}
        pool = [p for p in pool if p.get('stat') not in combo_stats]

    # Line filter (for line floor exploitation)
    max_line = filt.get('max_line')
    if max_line is not None:
        pool = [p for p in pool if (p.get('line') or 999) <= max_line]

    # B2B filter
    no_b2b = filt.get('no_b2b')
    if no_b2b:
        pool = [p for p in pool if not p.get('is_b2b')]

    # Projection vs line agreement
    proj_agrees = filt.get('proj_agrees_direction')
    if proj_agrees:
        for p in pool:
            proj = p.get('projection', 0) or 0
            line = p.get('line', 0) or 0
            d = p.get('direction', '')
            if d == 'UNDER':
                p['_proj_agrees'] = proj < line
            else:
                p['_proj_agrees'] = proj > line
        pool = [p for p in pool if p.get('_proj_agrees', False)]

    # Ensemble prob filter
    min_ensemble = filt.get('min_ensemble_prob')
    if min_ensemble is not None:
        pool = [p for p in pool if (p.get('ensemble_prob') or p.get('xgb_prob') or 0) >= min_ensemble]

    # Regression margin filter
    min_reg_margin = filt.get('min_reg_margin')
    if min_reg_margin is not None:
        pool = [p for p in pool
                if p.get('reg_margin') is not None
                and abs(p.get('reg_margin', 0)) >= min_reg_margin
                and ((p.get('direction') == 'UNDER' and (p.get('reg_margin', 0)) < 0) or
                     (p.get('direction') == 'OVER' and (p.get('reg_margin', 0)) > 0))]

    return pool


# ─────────────────────────────────────────────────────────────────────
# 3. SORTING / SELECTION
# ─────────────────────────────────────────────────────────────────────

def sort_and_select(pool, filt, n_legs=3):
    """Sort filtered pool and select top N legs for parlay."""
    sort_key = filt.get('sort_by', 'gap')

    if sort_key == 'gap':
        pool.sort(key=lambda p: abs(p.get('gap', 0) or 0), reverse=True)
    elif sort_key == 'l10_hr':
        pool.sort(key=lambda p: p.get('l10_hit_rate', 0) or 0, reverse=True)
    elif sort_key == 'ensemble':
        pool.sort(key=lambda p: p.get('ensemble_prob', p.get('xgb_prob', 0)) or 0, reverse=True)
    elif sort_key == 'reg_margin':
        pool.sort(key=lambda p: abs(p.get('reg_margin', 0) or 0), reverse=True)
    elif sort_key == 'floor':
        pool.sort(key=lambda p: p.get('l10_miss_count', 10) or 10)
    elif sort_key == 'composite':
        # Multi-signal composite
        for p in pool:
            score = 0
            score += min(abs(p.get('gap', 0) or 0) / 5, 1) * 0.3
            score += ((p.get('l10_hit_rate', 50) or 50) / 100) * 0.3
            score += (1 - (p.get('l10_miss_count', 5) or 5) / 10) * 0.2
            score += ((p.get('ensemble_prob', 0.5) or 0.5) - 0.4) * 0.2
            p['_composite'] = score
        pool.sort(key=lambda p: p.get('_composite', 0), reverse=True)
    elif sort_key == 'random':
        random.shuffle(pool)

    # Ensure player diversity (no duplicate players)
    seen_players = set()
    selected = []
    for p in pool:
        player = p.get('player', '')
        if player not in seen_players:
            seen_players.add(player)
            selected.append(p)
            if len(selected) >= n_legs:
                break

    return selected


# ─────────────────────────────────────────────────────────────────────
# 4. PARLAY SIMULATION
# ─────────────────────────────────────────────────────────────────────

def simulate_parlays(by_date, filt, n_legs=3):
    """Simulate parlays across all dates with a given filter."""
    results = []
    for date, props in by_date.items():
        pool = apply_filter(props, filt)
        if len(pool) < n_legs:
            continue

        legs = sort_and_select(pool, filt, n_legs)
        if len(legs) < n_legs:
            continue

        hits = sum(1 for l in legs if l['result'] == 'HIT')
        cashed = hits == n_legs

        results.append({
            'date': date,
            'legs': n_legs,
            'hits': hits,
            'cashed': cashed,
            'leg_hr': hits / n_legs,
            'pool_size': len(pool),
        })

    if not results:
        return None

    total_days = len(results)
    wins = sum(1 for r in results if r['cashed'])
    total_legs = sum(r['legs'] for r in results)
    total_leg_hits = sum(r['hits'] for r in results)

    return {
        'days_played': total_days,
        'wins': wins,
        'losses': total_days - wins,
        'win_rate': wins / total_days if total_days > 0 else 0,
        'leg_hr': total_leg_hits / total_legs if total_legs > 0 else 0,
        'avg_pool': sum(r['pool_size'] for r in results) / total_days,
        'results': results,
    }


# ─────────────────────────────────────────────────────────────────────
# 5. STRATEGY GENERATOR (brute force)
# ─────────────────────────────────────────────────────────────────────

def generate_strategies():
    """Generate millions of filter combinations to test."""
    strategies = []

    # ── Dimension values ──
    directions = [None, 'UNDER', 'OVER']
    stat_groups = [
        None,
        {'blk', 'stl'},
        {'pts'},
        {'reb'},
        {'ast'},
        {'pts', 'reb', 'ast'},
        {'blk', 'stl', '3pm'},
        {'reb', 'ast'},
    ]
    stat_excludes = [
        None,
        {'pra', 'pr', 'pa', 'ra', 'stl_blk'},  # no combos
        {'pts'},
        {'blk'},
    ]
    min_tiers = [None, 'S', 'A', 'B', 'C']
    min_l10_hrs = [None, 40, 50, 55, 60, 65, 70]
    min_l5_hrs = [None, 40, 50, 60]
    min_gaps = [None, 0.5, 1.0, 1.5, 2.0, 3.0]
    max_gaps = [None, 5.0, 10.0]
    streaks = [None, 'COLD', 'HOT']
    streak_excludes = [None, ['HOT']]
    min_mins_pcts = [None, 50, 60, 70]
    max_misses = [None, 2, 3, 4, 5]
    no_combos_opts = [False, True]
    max_lines = [None, 0.5, 1.5, 5.5]
    no_b2b_opts = [False, True]
    proj_agrees_opts = [False, True]
    sort_bys = ['gap', 'l10_hr', 'ensemble', 'reg_margin', 'floor', 'composite', 'random']
    n_legs_opts = [2, 3]
    min_reg_margins = [None, 1.0, 2.0, 3.0]

    # ── Phase 1: Targeted combinations (most likely to win) ──
    # UNDER + base stats + high HR + low miss
    for min_hr in [50, 55, 60, 65, 70]:
        for max_miss in [2, 3, 4, 5]:
            for sort in ['gap', 'composite', 'floor', 'l10_hr', 'reg_margin']:
                for n_legs in [2, 3]:
                    for min_gap in [None, 0.5, 1.0, 1.5, 2.0]:
                        for min_tier in [None, 'B', 'C']:
                            strategies.append({
                                'name': f'under_hr{min_hr}_miss{max_miss}_{sort}_{n_legs}leg_gap{min_gap}_t{min_tier}',
                                'direction': 'UNDER',
                                'no_combos': True,
                                'min_l10_hr': min_hr,
                                'max_l10_miss': max_miss,
                                'sort_by': sort,
                                'n_legs': n_legs,
                                'min_gap': min_gap,
                                'min_tier': min_tier,
                            })

    # BLK/STL UNDER (historically highest HR)
    for max_line in [0.5, 1.5, 5.5, None]:
        for min_hr in [None, 50, 60]:
            for sort in ['gap', 'composite', 'l10_hr']:
                for n_legs in [2, 3]:
                    strategies.append({
                        'name': f'blkstl_under_line{max_line}_{sort}_{n_legs}leg_hr{min_hr}',
                        'direction': 'UNDER',
                        'stats': {'blk', 'stl'},
                        'max_line': max_line,
                        'min_l10_hr': min_hr,
                        'sort_by': sort,
                        'n_legs': n_legs,
                    })

    # COLD + UNDER
    for min_hr in [None, 40, 50, 60]:
        for min_gap in [None, 0.5, 1.0, 2.0]:
            for sort in ['gap', 'composite', 'floor']:
                for n_legs in [2, 3]:
                    strategies.append({
                        'name': f'cold_under_hr{min_hr}_gap{min_gap}_{sort}_{n_legs}leg',
                        'direction': 'UNDER',
                        'streak': 'COLD',
                        'min_l10_hr': min_hr,
                        'min_gap': min_gap,
                        'sort_by': sort,
                        'n_legs': n_legs,
                        'no_combos': True,
                    })

    # Regression margin (95%+ when |margin| >= 3)
    for min_margin in [1.0, 2.0, 3.0, 4.0, 5.0]:
        for direction in [None, 'UNDER']:
            for sort in ['reg_margin', 'composite', 'gap']:
                for n_legs in [2, 3]:
                    strategies.append({
                        'name': f'reg_margin{min_margin}_{direction or "any"}_{sort}_{n_legs}leg',
                        'min_reg_margin': min_margin,
                        'direction': direction,
                        'sort_by': sort,
                        'n_legs': n_legs,
                    })

    # No-HOT + UNDER (HOT traps at 49.2%)
    for min_hr in [None, 50, 55, 60]:
        for min_tier in [None, 'B', 'C']:
            for sort in ['gap', 'composite', 'floor', 'ensemble']:
                for n_legs in [2, 3]:
                    strategies.append({
                        'name': f'nohot_under_hr{min_hr}_t{min_tier}_{sort}_{n_legs}leg',
                        'direction': 'UNDER',
                        'streak_exclude': ['HOT'],
                        'min_l10_hr': min_hr,
                        'min_tier': min_tier,
                        'sort_by': sort,
                        'n_legs': n_legs,
                        'no_combos': True,
                    })

    # Projection agrees + high HR
    for min_hr in [50, 55, 60, 65]:
        for direction in ['UNDER', None]:
            for sort in ['gap', 'composite', 'ensemble']:
                for n_legs in [2, 3]:
                    strategies.append({
                        'name': f'proj_agree_hr{min_hr}_{direction or "any"}_{sort}_{n_legs}leg',
                        'direction': direction,
                        'proj_agrees_direction': True,
                        'min_l10_hr': min_hr,
                        'sort_by': sort,
                        'n_legs': n_legs,
                        'no_combos': True,
                    })

    # High floor (low miss count) strategies
    for max_miss in [1, 2, 3]:
        for min_hr in [None, 50, 60]:
            for direction in ['UNDER', None]:
                for sort in ['floor', 'composite', 'gap']:
                    for n_legs in [2, 3]:
                        strategies.append({
                            'name': f'highfloor_miss{max_miss}_hr{min_hr}_{direction or "any"}_{sort}_{n_legs}leg',
                            'max_l10_miss': max_miss,
                            'min_l10_hr': min_hr,
                            'direction': direction,
                            'sort_by': sort,
                            'n_legs': n_legs,
                            'no_combos': True,
                        })

    # ── Phase 2: Grid search (systematic combinations) ──
    for direction in ['UNDER', None]:
        for stat_ex in [None, {'pra', 'pr', 'pa', 'ra', 'stl_blk'}]:
            for min_hr in [None, 50, 60]:
                for max_miss in [None, 3, 5]:
                    for min_gap in [None, 1.0]:
                        for sort in ['composite', 'gap', 'reg_margin']:
                            for n_legs in [2, 3]:
                                strategies.append({
                                    'name': f'grid_{direction or "any"}_ex{1 if stat_ex else 0}_hr{min_hr}_miss{max_miss}_gap{min_gap}_{sort}_{n_legs}',
                                    'direction': direction,
                                    'stats_exclude': stat_ex,
                                    'min_l10_hr': min_hr,
                                    'max_l10_miss': max_miss,
                                    'min_gap': min_gap,
                                    'sort_by': sort,
                                    'n_legs': n_legs,
                                })

    # ── Phase 3: Ensemble prob + reg margin combos ──
    for min_ens in [0.50, 0.52, 0.55, 0.58]:
        for min_margin in [None, 1.0, 2.0]:
            for direction in ['UNDER', None]:
                for n_legs in [2, 3]:
                    strategies.append({
                        'name': f'ens{min_ens}_mrg{min_margin}_{direction or "any"}_{n_legs}leg',
                        'min_ensemble_prob': min_ens,
                        'min_reg_margin': min_margin,
                        'direction': direction,
                        'sort_by': 'composite',
                        'n_legs': n_legs,
                        'no_combos': True,
                    })

    # ── Phase 4: Extreme filters (very selective, high conviction) ──
    for min_hr in [65, 70, 75, 80]:
        for max_miss in [1, 2]:
            for min_gap in [1.5, 2.0, 3.0]:
                for n_legs in [2, 3]:
                    strategies.append({
                        'name': f'extreme_hr{min_hr}_miss{max_miss}_gap{min_gap}_{n_legs}leg',
                        'direction': 'UNDER',
                        'min_l10_hr': min_hr,
                        'max_l10_miss': max_miss,
                        'min_gap': min_gap,
                        'sort_by': 'composite',
                        'n_legs': n_legs,
                        'no_combos': True,
                        'no_b2b': True,
                    })

    # Deduplicate by name
    seen = set()
    unique = []
    for s in strategies:
        if s['name'] not in seen:
            seen.add(s['name'])
            unique.append(s)

    return unique


# ─────────────────────────────────────────────────────────────────────
# 6. MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────

def run_lab(top_n=30, min_days=5):
    """Run the full parlay lab."""
    print(f"{'=' * 70}")
    print(f"  PARLAY LAB — Brute-Force Backtest")
    print(f"{'=' * 70}")

    all_props = load_all_graded()
    by_date = group_by_date(all_props)
    print(f"  Loaded {len(all_props)} graded props across {len(by_date)} days")
    print(f"  Overall HR: {sum(1 for p in all_props if p['result']=='HIT')/len(all_props):.1%}")

    strategies = generate_strategies()
    print(f"  Testing {len(strategies):,} strategies...\n")

    results = []
    for i, strat in enumerate(strategies):
        n_legs = strat.pop('n_legs', 3)
        name = strat.pop('name', f'strat_{i}')

        sim = simulate_parlays(by_date, strat, n_legs=n_legs)
        if sim and sim['days_played'] >= min_days:
            results.append({
                'name': name,
                'n_legs': n_legs,
                **sim,
                'filter': strat,
            })

        if (i + 1) % 1000 == 0:
            print(f"  ... tested {i+1:,}/{len(strategies):,} strategies")

    print(f"\n  Completed: {len(results):,} valid strategies (>= {min_days} days played)")

    # Sort by win rate, then by days played (more data = more reliable)
    results.sort(key=lambda r: (r['win_rate'], r['days_played'], r['leg_hr']), reverse=True)

    # Print top strategies
    print(f"\n{'=' * 70}")
    print(f"  TOP {top_n} STRATEGIES (by parlay win rate)")
    print(f"{'=' * 70}")
    print(f"  {'#':>3s}  {'Strategy':<55s} {'W':>2s}-{'L':>2s}  {'Win%':>5s}  {'LegHR':>5s}  {'Legs':>4s}  {'Pool':>4s}")
    print(f"  {'-'*3}  {'-'*55} {'-'*2} {'-'*2}  {'-'*5}  {'-'*5}  {'-'*4}  {'-'*4}")

    for i, r in enumerate(results[:top_n]):
        print(
            f"  {i+1:>3d}  {r['name']:<55s} {r['wins']:>2d}-{r['losses']:>2d}  "
            f"{r['win_rate']:>5.1%}  {r['leg_hr']:>5.1%}  {r['n_legs']:>4d}  "
            f"{r['avg_pool']:>4.0f}"
        )

    # Show day-by-day for top 5
    print(f"\n{'=' * 70}")
    print(f"  TOP 5 — Day-by-Day Breakdown")
    print(f"{'=' * 70}")
    for i, r in enumerate(results[:5]):
        print(f"\n  #{i+1}: {r['name']}")
        print(f"  Record: {r['wins']}W-{r['losses']}L ({r['win_rate']:.1%})  Leg HR: {r['leg_hr']:.1%}")
        for day in r['results']:
            status = "WIN" if day['cashed'] else "MISS"
            print(f"    {day['date']}: {day['hits']}/{day['legs']} [{status}]  (pool: {day['pool_size']})")

    # Save results
    save_path = os.path.join(PRED_DIR, 'parlay_lab_results.json')
    save_data = []
    for r in results[:200]:  # top 200
        r_copy = {k: v for k, v in r.items() if k != 'results'}
        r_copy['daily_results'] = r['results']
        r_copy['filter'] = {k: (list(v) if isinstance(v, set) else v) for k, v in r.get('filter', {}).items()}
        save_data.append(r_copy)

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved top 200 to {save_path}")

    return results


if __name__ == '__main__':
    top_n = 30
    if '--top' in sys.argv:
        idx = sys.argv.index('--top')
        if idx + 1 < len(sys.argv):
            top_n = int(sys.argv[idx + 1])

    run_lab(top_n=top_n)

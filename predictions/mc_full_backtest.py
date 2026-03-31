#!/usr/bin/env python3
"""
Full Monte Carlo backtest across ALL available data.
v16.2: Tests 80+ strategy combinations across all graded dates + backfill data.
Run: python3 predictions/mc_full_backtest.py
"""

import json, glob, os, random, sys, time
import numpy as np
from collections import defaultdict
from itertools import product as iprod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parlay_engine import _is_eligible

# ═══════════════════════════════════════════════════════════
# LOAD ALL GRADED DATA
# ═══════════════════════════════════════════════════════════

def load_all_data():
    """Load all graded results from all dates."""
    base = os.path.dirname(os.path.abspath(__file__))
    graded_files = sorted(glob.glob(os.path.join(base, '2026-03-*/v4_graded_*.json')))

    by_date = {}
    for f in graded_files:
        date = os.path.basename(os.path.dirname(f))
        size = os.path.getsize(f)
        if date not in by_date or size > os.path.getsize(by_date[date]):
            by_date[date] = f

    all_dates = {}
    total_props = 0
    for date, f in sorted(by_date.items()):
        with open(f) as fh:
            data = json.load(fh)
        results = data.get('results', [])
        valid = [r for r in results if r.get('result') in ('HIT', 'MISS')]
        if len(valid) >= 20:
            all_dates[date] = valid
            total_props += len(valid)

    print(f"  Loaded {len(all_dates)} dates, {total_props:,} total props")
    return all_dates


# ═══════════════════════════════════════════════════════════
# STRATEGY PARAMETER GRID
# ═══════════════════════════════════════════════════════════

def _tier(p): return p.get('tier', 'F')
def _dir(p): return p.get('direction', '').upper()
def _stat(p): return p.get('stat', '').lower()
def _rm(p): return p.get('reg_margin') or 0

COMBO_STATS = {'pra', 'pr', 'pa', 'ra'}

def make_filter(tier_filter='any', direction='any', stat_filter='any',
                streak='any', min_hr=0, max_miss=10, min_mins=0,
                rm_min=-999, rm_max=999, no_combo=False, min_gap=0):
    """Build a filter function from parameters."""
    def f(p):
        if not _is_eligible(p, tier_filter='SABCDF' if tier_filter == 'any' else tier_filter):
            return False
        if direction != 'any' and _dir(p) != direction:
            return False
        if stat_filter != 'any':
            if isinstance(stat_filter, str):
                if _stat(p) != stat_filter:
                    return False
            elif isinstance(stat_filter, (list, tuple, set)):
                if _stat(p) not in stat_filter:
                    return False
        if streak != 'any':
            if streak == 'COLD' and p.get('streak_status') != 'COLD':
                return False
            if streak == 'NOT_HOT' and p.get('streak_status') == 'HOT':
                return False
        if (p.get('l10_hit_rate', 0) or 0) < min_hr:
            return False
        if (p.get('l10_miss_count', 0) or 0) > max_miss:
            return False
        if (p.get('mins_30plus_pct', 0) or 0) < min_mins:
            return False
        rm = _rm(p)
        if rm != 0 and (rm < rm_min or rm > rm_max):
            return False
        if no_combo and _stat(p) in COMBO_STATS:
            return False
        if (p.get('abs_gap', 0) or 0) < min_gap:
            return False
        return True
    return f


def make_sort(sort_key='gap'):
    """Build a sort function from a key name."""
    if sort_key == 'gap':
        return lambda p: p.get('abs_gap', 0) or 0
    elif sort_key == 'xgb':
        return lambda p: p.get('xgb_prob', 0) or 0
    elif sort_key == 'sim':
        return lambda p: p.get('sim_prob', 0) or 0
    elif sort_key == 'ensemble':
        return lambda p: p.get('ensemble_prob', 0) or 0
    elif sort_key == 'l10hr':
        return lambda p: p.get('l10_hit_rate', 0) or 0
    elif sort_key == 'season_hr':
        return lambda p: p.get('season_hit_rate', 0) or 0
    elif sort_key == 'rm_sweet':
        # Proximity to sweet spot center (-1.5)
        return lambda p: -abs((_rm(p) or -1.5) + 1.5)
    elif sort_key == 'line_vs_l10':
        return lambda p: (p.get('line', 0) or 0) - (p.get('l10_avg', 0) or 0)
    elif sort_key == 'floor':
        return lambda p: (10 - (p.get('l10_miss_count', 10) or 10))
    elif sort_key == 'composite':
        return lambda p: (
            (p.get('abs_gap', 0) or 0) * 0.3 +
            (p.get('l10_hit_rate', 0) or 0) / 10 * 0.3 +
            (p.get('xgb_prob', 0) or 0) * 10 * 0.2 +
            (10 - (p.get('l10_miss_count', 10) or 10)) * 0.2
        )
    else:
        return lambda p: p.get('abs_gap', 0) or 0


# ═══════════════════════════════════════════════════════════
# GENERATE ALL STRATEGIES
# ═══════════════════════════════════════════════════════════

def generate_strategies():
    """Generate 80+ strategy combinations to test."""
    strategies = {}

    # ─── HAND-CRAFTED (top MC findings) ───
    strategies['mc1_rm_sweet_2L'] = {
        'filter': make_filter(direction='UNDER', rm_min=-4, rm_max=-0.5),
        'sort': make_sort('rm_sweet'), 'n': 2,
        'desc': 'rm sweet spot [-4,-0.5] UNDER 2-leg'
    }
    strategies['mc1_rm_sweet_3L'] = {
        'filter': make_filter(direction='UNDER', rm_min=-4, rm_max=-0.5),
        'sort': make_sort('rm_sweet'), 'n': 3,
        'desc': 'rm sweet spot [-4,-0.5] UNDER 3-leg'
    }
    strategies['mc3_cold_under_2L'] = {
        'filter': make_filter(direction='UNDER', streak='COLD'),
        'sort': make_sort('gap'), 'n': 2,
        'desc': 'COLD+UNDER 2-leg gap sort'
    }
    strategies['mc4_s_tier_3L'] = {
        'filter': make_filter(tier_filter='S'),
        'sort': make_sort('gap'), 'n': 3,
        'desc': 'S-tier only 3-leg'
    }
    strategies['mc6_s_tier_2L'] = {
        'filter': make_filter(tier_filter='S'),
        'sort': make_sort('gap'), 'n': 2,
        'desc': 'S-tier only 2-leg'
    }
    strategies['stl_under_singles'] = {
        'filter': make_filter(direction='UNDER', stat_filter='stl'),
        'sort': make_sort('gap'), 'n': 1,
        'desc': 'STL UNDER singles'
    }

    # ─── GRID: tier × direction × sort × legs × streak ───
    tier_opts = ['S', 'SA', 'SAB', 'any']
    dir_opts = ['UNDER', 'OVER', 'any']
    sort_opts = ['gap', 'xgb', 'l10hr', 'line_vs_l10', 'floor', 'composite', 'rm_sweet']
    leg_opts = [1, 2, 3, 4]
    streak_opts = ['any', 'COLD', 'NOT_HOT']

    # Focused grid (most promising combos, not full cartesian)
    for tier in tier_opts:
        for direction in dir_opts:
            for sort_key in sort_opts:
                for n_legs in leg_opts:
                    for streak in streak_opts:
                        # Skip impossible combos
                        if direction == 'OVER' and streak == 'COLD':
                            continue  # COLD+OVER is a trap
                        if tier == 'S' and n_legs >= 4:
                            continue  # not enough S-tier picks usually
                        if sort_key == 'rm_sweet' and direction != 'UNDER':
                            continue  # rm sweet is UNDER-specific

                        name = f"grid_{tier}_{direction[:3]}_{sort_key}_{n_legs}L_{streak[:3]}"
                        strategies[name] = {
                            'filter': make_filter(tier_filter=tier, direction=direction, streak=streak),
                            'sort': make_sort(sort_key),
                            'n': n_legs,
                            'desc': f'{tier} {direction} {sort_key} {n_legs}L {streak}'
                        }

    # ─── STAT-SPECIFIC ───
    for stat in ['stl', 'blk', '3pm', 'ast', 'reb', 'pts']:
        for n in [1, 2, 3]:
            for direction in ['UNDER', 'any']:
                name = f"stat_{stat}_{direction[:3]}_{n}L"
                strategies[name] = {
                    'filter': make_filter(direction=direction, stat_filter=stat),
                    'sort': make_sort('gap'), 'n': n,
                    'desc': f'{stat.upper()} {direction} {n}-leg'
                }

    # ─── ADVANCED COMBOS ───
    # rm sweet + COLD
    strategies['rm_sweet_cold_2L'] = {
        'filter': make_filter(direction='UNDER', streak='COLD', rm_min=-4, rm_max=-0.5),
        'sort': make_sort('rm_sweet'), 'n': 2,
        'desc': 'rm sweet + COLD 2-leg'
    }
    # S-tier + UNDER only
    for n in [2, 3]:
        strategies[f's_tier_under_{n}L'] = {
            'filter': make_filter(tier_filter='S', direction='UNDER'),
            'sort': make_sort('gap'), 'n': n,
            'desc': f'S-tier UNDER {n}-leg'
        }
    # High floor (low miss count)
    for n in [2, 3]:
        strategies[f'high_floor_{n}L'] = {
            'filter': make_filter(direction='UNDER', max_miss=2, min_mins=70),
            'sort': make_sort('gap'), 'n': n,
            'desc': f'High floor {n}-leg'
        }
    # Line above L10 by 2+
    strategies['line_above_2_2L'] = {
        'filter': make_filter(direction='UNDER'),
        'sort': make_sort('line_vs_l10'), 'n': 2,
        'desc': 'Line>L10+2 UNDER 2-leg',
        'extra_filter': lambda p: ((p.get('line',0) or 0) - (p.get('l10_avg',0) or 0)) >= 2
    }
    strategies['line_above_3_2L'] = {
        'filter': make_filter(direction='UNDER'),
        'sort': make_sort('line_vs_l10'), 'n': 2,
        'desc': 'Line>L10+3 UNDER 2-leg',
        'extra_filter': lambda p: ((p.get('line',0) or 0) - (p.get('l10_avg',0) or 0)) >= 3
    }
    # No combos
    for n in [2, 3]:
        strategies[f'no_combo_under_{n}L'] = {
            'filter': make_filter(direction='UNDER', no_combo=True),
            'sort': make_sort('gap'), 'n': n,
            'desc': f'No combo UNDER {n}-leg'
        }
    # High gap
    for min_gap in [2, 3, 4]:
        for n in [2, 3]:
            strategies[f'gap{min_gap}plus_{n}L'] = {
                'filter': make_filter(direction='UNDER', min_gap=min_gap),
                'sort': make_sort('gap'), 'n': n,
                'desc': f'Gap>={min_gap} UNDER {n}-leg'
            }
    # Consensus (multiple models agree)
    strategies['consensus_2plus_2L'] = {
        'filter': make_filter(direction='UNDER'),
        'sort': lambda p: sum(1 for x in [p.get('xgb_prob',0), p.get('sim_prob',0)]
                              if (x or 0) > 0.55),
        'n': 2, 'desc': '2+ models agree 2-leg',
        'extra_filter': lambda p: sum(1 for x in [p.get('xgb_prob',0), p.get('sim_prob',0)]
                                      if (x or 0) > 0.55) >= 2
    }

    return strategies


# ═══════════════════════════════════════════════════════════
# MONTE CARLO ENGINE
# ═══════════════════════════════════════════════════════════

def run_monte_carlo(all_dates, strategies, n_sims=100000):
    """Run MC simulation for all strategies."""
    random.seed(42)
    np.random.seed(42)

    results = []
    total = len(strategies)

    for idx, (name, strat) in enumerate(sorted(strategies.items())):
        n_legs = strat['n']
        filter_fn = strat['filter']
        sort_fn = strat['sort']
        extra_filter = strat.get('extra_filter')

        # Build pools per date
        date_pools = {}
        for date, props in all_dates.items():
            pool = [p for p in props if filter_fn(p)]
            if extra_filter:
                pool = [p for p in pool if extra_filter(p)]
            if len(pool) >= n_legs:
                date_pools[date] = pool

        if not date_pools:
            continue

        # ── Deterministic: top N by sort, diversified ──
        det_legs = det_hits = det_cashes = det_parlays = 0
        for date, pool in date_pools.items():
            pool_sorted = sorted(pool, key=sort_fn, reverse=True)
            selected = _diversified_pick(pool_sorted, n_legs)
            if len(selected) >= n_legs:
                selected = selected[:n_legs]
                hits = sum(1 for p in selected if p['result'] == 'HIT')
                det_legs += n_legs
                det_hits += hits
                det_parlays += 1
                if hits == n_legs:
                    det_cashes += 1

        if det_parlays == 0:
            continue

        # ── Monte Carlo: random selection ──
        mc_cashes = mc_legs_hit = mc_total_legs = mc_total = 0
        dates_list = list(date_pools.keys())

        for _ in range(n_sims):
            date = random.choice(dates_list)
            pool = date_pools[date]
            if len(pool) < n_legs:
                continue

            shuffled = random.sample(pool, min(len(pool), n_legs * 5))
            selected = _diversified_pick(shuffled, n_legs)
            if len(selected) < n_legs:
                continue

            mc_total += 1
            hits = sum(1 for p in selected if p['result'] == 'HIT')
            mc_legs_hit += hits
            mc_total_legs += n_legs
            if hits == n_legs:
                mc_cashes += 1

        if mc_total == 0:
            continue

        # ── Compute metrics ──
        det_leg_pct = det_hits / det_legs * 100
        det_cash_pct = det_cashes / det_parlays * 100
        mc_leg_pct = mc_legs_hit / mc_total_legs * 100
        mc_cash_pct = mc_cashes / mc_total * 100

        payout = {1: 1.91, 2: 3.64, 3: 6.96, 4: 13.3, 5: 25.4}.get(n_legs, 1.87**n_legs)
        be_pct = 100 / payout

        det_roi = (det_cash_pct / 100 * payout - 1) * 100
        mc_roi = (mc_cash_pct / 100 * payout - 1) * 100

        results.append({
            'name': name,
            'n': n_legs,
            'dates': len(date_pools),
            'pool_avg': np.mean([len(p) for p in date_pools.values()]),
            'det_leg': det_leg_pct,
            'det_cash': det_cash_pct,
            'det_parlays': det_parlays,
            'det_cashes': det_cashes,
            'mc_leg': mc_leg_pct,
            'mc_cash': mc_cash_pct,
            'mc_sims': mc_total,
            'det_roi': det_roi,
            'mc_roi': mc_roi,
            'be': be_pct,
            'profitable_det': det_cash_pct > be_pct,
            'profitable_mc': mc_cash_pct > be_pct,
            'desc': strat.get('desc', ''),
            'payout': payout,
        })

        if (idx + 1) % 50 == 0:
            print(f"  ... {idx+1}/{total} strategies tested")

    return results


def _diversified_pick(pool, n_target):
    """Pick top N from pool with 1 per game, 1 per player."""
    selected = []
    used_games, used_players = set(), set()
    for p in pool:
        player = p.get('player', '')
        game = p.get('game', '')
        if player in used_players or (game and game in used_games):
            continue
        selected.append(p)
        used_players.add(player)
        if game:
            used_games.add(game)
        if len(selected) >= n_target:
            break
    return selected


# ═══════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════

def print_report(results):
    """Print comprehensive MC results."""
    print(f"\n{'='*120}")
    print(f"  FULL MONTE CARLO RESULTS — {len(results)} strategies tested")
    print(f"{'='*120}")

    # Sort by MC ROI (more reliable than deterministic)
    results.sort(key=lambda x: x['mc_roi'], reverse=True)

    # Top table
    print(f"\n{'Rank':4s} {'Strategy':40s} {'Legs':4s} {'Dates':5s} {'Pool':5s} "
          f"{'DET':15s} {'MC (100K sims)':15s} {'MC ROI':8s}")
    print("─" * 120)

    for i, r in enumerate(results[:50]):
        det_str = f"{r['det_cashes']}W-{r['det_parlays']-r['det_cashes']}L {r['det_leg']:.0f}%/leg"
        mc_str = f"{r['mc_cash']:.1f}% cash {r['mc_leg']:.0f}%/leg"
        marker = " $$$" if r['mc_roi'] > 50 else " $$" if r['mc_roi'] > 20 else " $" if r['mc_roi'] > 0 else ""
        print(f"{i+1:4d} {r['name']:40s} {r['n']:4d} {r['dates']:5d} {r['pool_avg']:5.0f} "
              f"{det_str:15s} {mc_str:15s} {r['mc_roi']:+7.1f}%{marker}")

    # Profitable strategies
    profitable_mc = [r for r in results if r['profitable_mc']]
    profitable_det = [r for r in results if r['profitable_det']]

    print(f"\n{'='*80}")
    print(f"  PROFITABLE BY MC (RELIABLE): {len(profitable_mc)} / {len(results)}")
    print(f"{'='*80}")
    for r in sorted(profitable_mc, key=lambda x: x['mc_roi'], reverse=True)[:30]:
        print(f"  {r['name']:40s} {r['n']}L  MC: {r['mc_cash']:.1f}% cash  ROI={r['mc_roi']:+.1f}%  "
              f"({r['mc_leg']:.0f}%/leg, {r['mc_sims']:,} sims)")

    # Best per leg count
    print(f"\n{'='*80}")
    print(f"  BEST STRATEGY PER LEG COUNT (by MC ROI)")
    print(f"{'='*80}")
    for n in [1, 2, 3, 4]:
        by_n = [r for r in results if r['n'] == n and r['mc_roi'] > -50]
        if by_n:
            best = max(by_n, key=lambda x: x['mc_roi'])
            print(f"  {n}-leg: {best['name']:40s} MC ROI={best['mc_roi']:+.1f}%  "
                  f"cash={best['mc_cash']:.1f}%  leg={best['mc_leg']:.0f}%")

    # Save full results
    base = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(base, 'mc_full_backtest_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to: {out_file}")

    return results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 80)
    print("  FULL MONTE CARLO BACKTEST")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 80)

    n_sims = 100000
    if '--quick' in sys.argv:
        n_sims = 10000

    print(f"\n  Loading data...")
    all_dates = load_all_data()

    print(f"\n  Generating strategies...")
    strategies = generate_strategies()
    print(f"  Generated {len(strategies)} strategies")

    print(f"\n  Running Monte Carlo ({n_sims:,} sims per strategy)...")
    start = time.time()
    results = run_monte_carlo(all_dates, strategies, n_sims=n_sims)
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")

    report = print_report(results)

    print(f"\n  Done! {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

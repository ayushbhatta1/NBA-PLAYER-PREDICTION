#!/usr/bin/env python3
"""
Auto-optimization grid search for NBA parlay filters.
Tests hundreds of filter combinations, refines top performers,
then cross-validates on real graded data.

Usage:
    python3 predictions/backtesting/auto_optimize.py
    python3 predictions/backtesting/auto_optimize.py --graded-only  # skip backfill, validate only
"""
import json, os, sys, time, random
import numpy as np
from collections import defaultdict
from itertools import product
from datetime import datetime
from copy import deepcopy

BASE = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.dirname(BASE)
sys.path.insert(0, BASE)
sys.path.insert(0, PRED_DIR)

from mass_backtest import load_all_data, apply_filters, run_backtest, TIER_ORDER

# Try importing validate_filters; provide inline fallbacks if not yet created
try:
    from validate_filters import (load_graded_days, normalize_graded,
                                  apply_filter, compute_parlay_wr, extract_hit)
    HAS_VALIDATE = True
except ImportError:
    HAS_VALIDATE = False

OUTPUT_DIR = BASE
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'auto_optimize_results.json')
RECOMMENDED_PATH = os.path.join(OUTPUT_DIR, 'recommended_configs.json')

COMBO_STATS = {'pra', 'pr', 'pa', 'ra'}
ALL_STATS = ['pts', 'reb', 'ast', 'blk', 'stl', 'pra', 'pr', 'pa', 'ra']
BASE_STATS = ['pts', 'reb', 'ast', 'blk', 'stl']


# ─────────────────────────────────────────────────────────
# GRADED DATA LOADING (inline fallback)
# ─────────────────────────────────────────────────────────

def _load_graded_days_fallback():
    """Load all graded result files from predictions/YYYY-MM-DD/ folders."""
    all_picks = []
    dates_dir = PRED_DIR
    for folder in sorted(os.listdir(dates_dir)):
        if not folder.startswith('20') or len(folder) != 10:
            continue
        folder_path = os.path.join(dates_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        # Try multiple graded file naming patterns
        for fname in ['graded_results.json', 'graded_full_board.json',
                       'graded_results_mar12.json']:
            path = os.path.join(folder_path, fname)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    records = data.get('results', data) if isinstance(data, dict) else data
                    if isinstance(records, list):
                        for r in records:
                            r['_date'] = folder
                        all_picks.extend(records)
                except Exception:
                    pass
                break  # use first match per date folder
    return all_picks


def _normalize_graded_record(r):
    """Normalize a graded record to match backtest schema."""
    result = (r.get('result', '') or '').upper()
    hit = result == 'HIT'

    return {
        'player': r.get('player', ''),
        'stat': (r.get('stat', '') or '').lower(),
        'line': float(r.get('line', 0) or 0),
        'direction': (r.get('direction', '') or '').upper(),
        'tier': (r.get('tier', 'F') or 'F').upper(),
        'gap': float(r.get('gap', 0) or 0),
        'abs_gap': abs(float(r.get('gap', 0) or 0)),
        'hit': hit,
        'date': r.get('_date', ''),
        'l10_hit_rate': float(r.get('l10_hit_rate', 0) or 0),
        'l5_hit_rate': float(r.get('l5_hit_rate', 0) or 0),
        'season_hit_rate': float(r.get('season_hit_rate', 0) or 0),
        'l10_avg': float(r.get('l10_avg', 0) or 0),
        'l5_avg': float(r.get('l5_avg', 0) or 0),
        'l3_avg': float(r.get('l3_avg', 0) or 0),
        'season_avg': float(r.get('season_avg', 0) or 0),
        'mins_30plus_pct': float(r.get('mins_30plus_pct', 0) or 0),
        'l10_miss_count': int(r.get('l10_miss_count', 0) or 0),
        'l10_std': float(r.get('l10_std', 0) or 0),
        'l10_floor': float(r.get('l10_floor', 0) or 0),
        'is_home': int(r.get('is_home', 0) or 0),
        'is_b2b': int(r.get('is_b2b', 0) or 0),
        'spread': float(r.get('spread', 0) or 0) if r.get('spread') and str(r.get('spread')) != 'nan' else 0,
        'streak_status': (r.get('streak_status', '') or '').upper(),
        'games_used': int(r.get('games_used', 0) or 0),
        'source': 'graded',
    }


def _apply_filter_fallback(picks, direction=None, min_gap=0, min_l10_hr=0,
                            max_l10_hr=100, min_line=0, no_hot=False,
                            exclude_combos=False, min_tier='F',
                            min_season_hr=0, min_mins_pct=0,
                            max_l10_miss=10, max_l10_std=None):
    """Apply filters to graded picks (standalone fallback)."""
    filtered = []
    min_tier_val = TIER_ORDER.get(min_tier, 0)
    for p in picks:
        if direction and p['direction'] != direction.upper():
            continue
        if p['abs_gap'] < min_gap:
            continue
        if p['l10_hit_rate'] < min_l10_hr:
            continue
        if p['l10_hit_rate'] >= max_l10_hr:
            continue
        if p['line'] < min_line:
            continue
        if no_hot and p.get('streak_status') == 'HOT':
            continue
        if exclude_combos and p['stat'] in COMBO_STATS:
            continue
        tier_val = TIER_ORDER.get(p['tier'], 0)
        if tier_val < min_tier_val:
            continue
        if p['season_hit_rate'] < min_season_hr:
            continue
        if p['mins_30plus_pct'] < min_mins_pct:
            continue
        if p.get('l10_miss_count', 0) > max_l10_miss:
            continue
        if max_l10_std is not None and p.get('l10_std', 0) > max_l10_std:
            continue
        filtered.append(p)
    return filtered


def _compute_parlay_wr_fallback(picks, n_legs=3, n_samples=3000):
    """Compute parlay win rate from picks via sampling."""
    from itertools import combinations
    from math import comb

    if len(picks) < n_legs:
        return 0, 0, 0

    total_combos = comb(len(picks), n_legs)
    wins = 0
    total = 0

    if total_combos <= n_samples:
        for combo in combinations(range(len(picks)), n_legs):
            legs = [picks[i] for i in combo]
            players = [l['player'] for l in legs]
            if len(set(players)) != len(players):
                continue
            total += 1
            if all(l['hit'] for l in legs):
                wins += 1
    else:
        seen = set()
        attempts = 0
        while total < n_samples and attempts < n_samples * 15:
            attempts += 1
            indices = tuple(sorted(random.sample(range(len(picks)), n_legs)))
            if indices in seen:
                continue
            seen.add(indices)
            legs = [picks[i] for i in indices]
            players = [l['player'] for l in legs]
            if len(set(players)) != len(players):
                continue
            total += 1
            if all(l['hit'] for l in legs):
                wins += 1

    wr = wins / total if total else 0
    return wr, wins, total


# ─────────────────────────────────────────────────────────
# FORMAT CONVERTERS
# ─────────────────────────────────────────────────────────

def config_to_backtest(cfg):
    """Convert a grid config dict to mass_backtest apply_filters format."""
    filters = {}
    filters['min_gap'] = cfg.get('min_gap', 0)
    filters['min_tier'] = cfg.get('min_tier', 'F')
    filters['min_l10_hr'] = cfg.get('min_l10_hr', 0)
    filters['min_season_hr'] = cfg.get('min_season_hr', 0)
    filters['min_mins_pct'] = cfg.get('min_mins_pct', 0)
    filters['direction_strategy'] = cfg.get('direction_strategy', 'both')
    filters['max_l10_miss'] = cfg.get('max_l10_miss', 10)
    filters['max_l10_std'] = cfg.get('max_l10_std', None)
    filters['min_games'] = cfg.get('min_games', 0)
    filters['streak_filter'] = cfg.get('streak_filter', 'none')
    filters['exclude_combos'] = cfg.get('exclude_combos', False)
    filters['require_under'] = cfg.get('require_under', False)
    if 'allowed_stats' in cfg:
        filters['allowed_stats'] = cfg['allowed_stats']
    # New filters (will be added to mass_backtest apply_filters)
    if 'max_l10_hr' in cfg:
        filters['max_l10_hr'] = cfg['max_l10_hr']
    if 'min_line' in cfg:
        filters['min_line'] = cfg['min_line']
    return filters


def config_to_graded(cfg):
    """Convert a grid config dict to validate_filters apply_filter kwargs."""
    kwargs = {}

    # Direction mapping
    ds = cfg.get('direction_strategy', 'both')
    if ds == 'under_only':
        kwargs['direction'] = 'UNDER'
    elif ds == 'over_only':
        kwargs['direction'] = 'OVER'
    else:
        kwargs['direction'] = None

    kwargs['min_gap'] = cfg.get('min_gap', 0)
    kwargs['min_l10_hr'] = cfg.get('min_l10_hr', 0)
    kwargs['max_l10_hr'] = cfg.get('max_l10_hr', 100)
    kwargs['min_line'] = cfg.get('min_line', 0)
    kwargs['min_tier'] = cfg.get('min_tier', 'F')
    kwargs['min_season_hr'] = cfg.get('min_season_hr', 0)
    kwargs['min_mins_pct'] = cfg.get('min_mins_pct', 0)
    kwargs['max_l10_miss'] = cfg.get('max_l10_miss', 10)
    kwargs['max_l10_std'] = cfg.get('max_l10_std', None)
    kwargs['exclude_combos'] = cfg.get('exclude_combos', False)

    # no_hot if streak_filter blocks hot
    sf = cfg.get('streak_filter', 'none')
    kwargs['no_hot'] = sf in ('no_hot', 'cold_under_only')

    return kwargs


def config_name(cfg):
    """Return the name from a config, or generate one."""
    return cfg.get('name', 'unnamed')


# ─────────────────────────────────────────────────────────
# GRID GENERATORS
# ─────────────────────────────────────────────────────────

def generate_coarse_grid():
    """Generate ~100+ configs covering the primary parameter space."""
    configs = []

    # Reduced grid for speed (~200 configs instead of ~1400)
    directions = ['under_only', 'both']
    gaps = [1, 2, 3, 5, 7]
    hr_ranges = [
        (0, 100),    # no HR filter
        (50, 100),   # moderate floor
        (60, 70),    # sweet spot (from safe_filter)
        (60, 80),    # mid-range
        (70, 100),   # high floor
        (80, 100),   # very high floor
    ]
    lines = [0, 10, 15, 20]
    # Tier fixed at F (no tier filter) — refinement can add it
    tier = 'F'

    idx = 0
    for direction in directions:
        for gap in gaps:
            for (hr_lo, hr_hi) in hr_ranges:
                for min_line in lines:
                    # Skip obviously redundant combos
                    if direction == 'both' and gap <= 1 and hr_lo == 0 and min_line == 0:
                        continue
                    if gap >= 7 and min_line >= 20:
                        continue
                    if gap >= 5 and hr_lo >= 80 and min_line >= 15:
                        continue

                    idx += 1
                    dir_tag = 'U' if direction == 'under_only' else 'B'
                    name = f"g{gap}_hr{hr_lo}-{hr_hi}_ln{min_line}_{dir_tag}"

                    cfg = {
                        'name': name,
                        'direction_strategy': direction,
                        'min_gap': float(gap),
                        'min_l10_hr': hr_lo,
                        'max_l10_hr': hr_hi,
                        'min_line': float(min_line),
                        'min_tier': tier,
                        'min_season_hr': 0, 'min_mins_pct': 0,
                        'max_l10_miss': 10, 'max_l10_std': None,
                        'min_games': 0, 'streak_filter': 'none',
                        'exclude_combos': False, 'allowed_stats': ALL_STATS,
                    }
                    configs.append(cfg)

    print(f"  Coarse grid: {len(configs)} configs generated")
    return configs


def generate_refinement_grid(top_configs, n=10):
    """For each of the top N configs, generate ~8 perturbations."""
    configs = []
    for rank, cfg in enumerate(top_configs[:n]):
        base = deepcopy(cfg)
        base_name = config_name(base)

        perturbations = []

        # Gap +/- 1
        for delta in [-1, 1]:
            new_gap = max(0.5, base['min_gap'] + delta)
            p = deepcopy(base)
            p['min_gap'] = new_gap
            p['name'] = f"ref_{base_name}_gap{new_gap:.0f}"
            perturbations.append(p)

        # Line +/- 5
        for delta in [-5, 5]:
            new_line = max(0, base.get('min_line', 0) + delta)
            p = deepcopy(base)
            p['min_line'] = float(new_line)
            p['name'] = f"ref_{base_name}_ln{new_line:.0f}"
            perturbations.append(p)

        # HR range shift by 10
        hr_lo = base.get('min_l10_hr', 0)
        hr_hi = base.get('max_l10_hr', 100)
        for shift in [-10, 10]:
            new_lo = max(0, hr_lo + shift)
            new_hi = min(100, hr_hi + shift)
            if new_lo < new_hi:
                p = deepcopy(base)
                p['min_l10_hr'] = new_lo
                p['max_l10_hr'] = new_hi
                p['name'] = f"ref_{base_name}_hr{new_lo}-{new_hi}"
                perturbations.append(p)

        # Tier shift
        tier_list = ['F', 'D', 'C', 'B', 'A']
        cur_tier = base.get('min_tier', 'F')
        cur_idx = tier_list.index(cur_tier) if cur_tier in tier_list else 0
        for delta in [-1, 1]:
            new_idx = cur_idx + delta
            if 0 <= new_idx < len(tier_list):
                p = deepcopy(base)
                p['min_tier'] = tier_list[new_idx]
                p['name'] = f"ref_{base_name}_t{tier_list[new_idx]}"
                perturbations.append(p)

        # Deduplicate by name
        seen_names = set()
        for p in perturbations:
            if p['name'] not in seen_names:
                seen_names.add(p['name'])
                configs.append(p)

    print(f"  Refinement grid: {len(configs)} configs generated from top {n}")
    return configs


def generate_compound_grid(top_configs, n=5):
    """For each of the top N, add secondary filters one at a time."""
    configs = []
    for rank, cfg in enumerate(top_configs[:n]):
        base = deepcopy(cfg)
        base_name = config_name(base)

        # Max L10 miss count
        for max_miss in [3, 5]:
            p = deepcopy(base)
            p['max_l10_miss'] = max_miss
            p['name'] = f"cmp_{base_name}_miss{max_miss}"
            configs.append(p)

        # Max L10 std dev
        for max_std in [5, 8]:
            p = deepcopy(base)
            p['max_l10_std'] = float(max_std)
            p['name'] = f"cmp_{base_name}_std{max_std}"
            configs.append(p)

        # Streak filter
        for sf in ['no_cold', 'cold_under_only']:
            p = deepcopy(base)
            p['streak_filter'] = sf
            p['name'] = f"cmp_{base_name}_{sf}"
            configs.append(p)

        # Exclude combos
        p = deepcopy(base)
        p['exclude_combos'] = True
        p['name'] = f"cmp_{base_name}_noCombos"
        configs.append(p)

        # Min minutes pct
        for mins in [50, 60]:
            p = deepcopy(base)
            p['min_mins_pct'] = mins
            p['name'] = f"cmp_{base_name}_mins{mins}"
            configs.append(p)

        # Min season HR
        for shr in [50, 60]:
            p = deepcopy(base)
            p['min_season_hr'] = shr
            p['name'] = f"cmp_{base_name}_shr{shr}"
            configs.append(p)

    print(f"  Compound grid: {len(configs)} configs generated from top {n}")
    return configs


# ─────────────────────────────────────────────────────────
# BACKFILL RUNNER
# ─────────────────────────────────────────────────────────

def run_coarse_grid(by_date, dates_sorted, configs, max_parlays=500, full_dates=False):
    """Run each config through backtest. Returns sorted results list.

    Args:
        full_dates: If True, use all dates. If False, sample every 3rd date for speed.
    """
    results = []
    total = len(configs)
    t_start = time.time()

    if full_dates:
        run_dates = dates_sorted
        run_by_date = by_date
        print(f"  Using ALL {len(run_dates)} dates (full mode)")
    else:
        run_dates = dates_sorted[::3]
        run_by_date = {d: by_date[d] for d in run_dates}
        print(f"  Using {len(run_dates)}/{len(dates_sorted)} sampled dates for speed")

    for i, cfg in enumerate(configs):
        name = config_name(cfg)
        filters = config_to_backtest(cfg)

        # Quick eligibility pre-check on 10 dates
        check_dates = run_dates[::max(1, len(run_dates) // 10)]
        sample_eligible = sum(len(apply_filters(run_by_date[d], filters)) for d in check_dates)
        est_total = sample_eligible * (len(run_dates) / len(check_dates))

        if est_total < 30:
            results.append({
                'name': name, 'config': cfg, 'skipped': True,
                'reason': f'too_few_picks (est {est_total:.0f})',
                'eligible': 0, 'ind_hr': 0, 'wr_2leg': 0, 'wr_3leg': 0,
                'combined_wr': 0, 'avg_cands_per_day': 0,
            })
            continue

        bt = run_backtest(run_by_date, run_dates, filters,
                          max_parlays_per_day=max_parlays, verbose=False)
        s = bt['summary']
        eligible = bt['all_eligible']
        ind_hr = sum(1 for p in eligible if p['hit']) / len(eligible) if eligible else 0
        combined = s['wr_2leg'] * 0.4 + s['wr_3leg'] * 0.6

        results.append({
            'name': name, 'config': cfg, 'skipped': False,
            'eligible': len(eligible), 'ind_hr': ind_hr,
            'wr_2leg': s['wr_2leg'], 'wr_3leg': s['wr_3leg'],
            'combined_wr': combined,
            'total_2leg': s['total_2leg'], 'total_3leg': s['total_3leg'],
            'avg_cands_per_day': s.get('avg_candidates_per_day', 0),
            'dates_with_parlays': s.get('dates_with_parlays', 0),
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            best_so_far = max((r['combined_wr'] for r in results if not r.get('skipped')), default=0)
            print(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining | "
                  f"best combined WR: {best_so_far:.1%}")

    # Sort by combined WR descending
    results.sort(key=lambda r: r.get('combined_wr', 0), reverse=True)

    # Print top 15
    print(f"\n  {'Rank':>4} {'Name':<45} {'IndHR':>6} {'2-leg':>7} {'3-leg':>7} {'Cands/d':>8} {'Elig':>7}")
    print(f"  {'-'*90}")
    for rank, r in enumerate(results[:15]):
        if r.get('skipped'):
            continue
        print(f"  {rank+1:>4} {r['name']:<45} {r['ind_hr']:>5.1%} "
              f"{r['wr_2leg']:>6.1%} {r['wr_3leg']:>6.1%} "
              f"{r['avg_cands_per_day']:>7.1f} {r['eligible']:>7,}")

    return results


# ─────────────────────────────────────────────────────────
# CROSS-VALIDATION ON GRADED DATA
# ─────────────────────────────────────────────────────────

def cross_validate_on_graded(top_configs):
    """Load real graded data and validate each config against it."""
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION ON REAL GRADED DATA")
    print(f"{'='*60}")

    # Load graded data
    if HAS_VALIDATE:
        raw_days = load_graded_days()
        picks = []
        for date, records in raw_days.items():
            for r in records:
                nr = normalize_graded(r, date)
                if nr:
                    picks.append(nr)
    else:
        raw_picks = _load_graded_days_fallback()
        picks = [_normalize_graded_record(r) for r in raw_picks]

    # Filter out records without result
    picks = [p for p in picks if p.get('hit') is not None and p.get('stat')]

    if not picks:
        print("  WARNING: No graded data found. Skipping cross-validation.")
        return []

    # Group by date
    by_date_graded = defaultdict(list)
    for p in picks:
        by_date_graded[p['date']].append(p)

    total_picks = len(picks)
    total_hits = sum(1 for p in picks if p['hit'])
    n_dates = len(by_date_graded)
    print(f"  Loaded {total_picks} graded picks across {n_dates} dates")
    print(f"  Baseline HR: {total_hits/total_picks:.1%}")

    results = []
    for cfg in top_configs:
        name = config_name(cfg)

        # Apply filters using graded-compatible format
        if HAS_VALIDATE:
            filter_cfg = config_to_graded(cfg)
            filtered = apply_filter(picks, filter_cfg)
        else:
            kwargs = config_to_graded(cfg)
            filtered = _apply_filter_fallback(picks, **kwargs)

        if not filtered:
            results.append({
                'name': name,
                'config': cfg,
                'graded_eligible': 0,
                'graded_hr': 0,
                'graded_2leg_wr': 0,
                'graded_3leg_wr': 0,
                'graded_avg_per_day': 0,
            })
            continue

        n_eligible = len(filtered)
        n_hits = sum(1 for p in filtered if p['hit'])
        hr = n_hits / n_eligible if n_eligible else 0

        # Compute parlay WR
        if HAS_VALIDATE:
            # compute_parlay_wr expects {date: [picks]} dict
            filtered_by_date = defaultdict(list)
            for p in filtered:
                filtered_by_date[p['date']].append(p)
            _, _, wr_2 = compute_parlay_wr(filtered_by_date, 2, max_per_day=2000)
            _, _, wr_3 = compute_parlay_wr(filtered_by_date, 3, max_per_day=2000)
        else:
            wr_2, _, _ = _compute_parlay_wr_fallback(filtered, n_legs=2, n_samples=3000)
            wr_3, _, _ = _compute_parlay_wr_fallback(filtered, n_legs=3, n_samples=3000)

        # Per-day stats
        daily_counts = defaultdict(int)
        for p in filtered:
            daily_counts[p['date']] += 1
        avg_per_day = np.mean(list(daily_counts.values())) if daily_counts else 0

        results.append({
            'name': name,
            'config': cfg,
            'graded_eligible': n_eligible,
            'graded_hr': hr,
            'graded_2leg_wr': wr_2,
            'graded_3leg_wr': wr_3,
            'graded_avg_per_day': avg_per_day,
            'graded_dates': len(daily_counts),
        })

    # Sort by graded 3-leg WR
    results.sort(key=lambda r: r.get('graded_3leg_wr', 0), reverse=True)

    # Print results
    print(f"\n  {'Rank':>4} {'Name':<45} {'GradedHR':>9} {'G-2leg':>7} {'G-3leg':>7} {'Picks':>6} {'Days':>5}")
    print(f"  {'-'*90}")
    for rank, r in enumerate(results[:20]):
        if r['graded_eligible'] == 0:
            continue
        print(f"  {rank+1:>4} {r['name']:<45} {r['graded_hr']:>8.1%} "
              f"{r['graded_2leg_wr']:>6.1%} {r['graded_3leg_wr']:>6.1%} "
              f"{r['graded_eligible']:>6} {r.get('graded_dates', 0):>5}")

    return results


# ─────────────────────────────────────────────────────────
# COMPARISON & OUTPUT
# ─────────────────────────────────────────────────────────

def build_comparison_table(backfill_results, graded_results):
    """Print side-by-side backfill vs graded performance."""
    graded_map = {config_name(r['config']): r for r in graded_results}

    print(f"\n{'='*60}")
    print(f"  BACKFILL vs GRADED COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Name':<40} {'BF-3leg':>8} {'GR-3leg':>8} {'Delta':>7} {'GR-HR':>7} {'Vol/d':>6}")
    print(f"  {'-'*80}")

    rows = []
    for r in backfill_results:
        if r.get('skipped'):
            continue
        name = r['name']
        gr = graded_map.get(name, {})
        bf_wr = r.get('wr_3leg', 0)
        gr_wr = gr.get('graded_3leg_wr', 0)
        delta = gr_wr - bf_wr
        gr_hr = gr.get('graded_hr', 0)
        vol = gr.get('graded_avg_per_day', 0)
        rows.append((name, bf_wr, gr_wr, delta, gr_hr, vol))

    # Sort by graded WR
    rows.sort(key=lambda x: x[2], reverse=True)
    for name, bf_wr, gr_wr, delta, gr_hr, vol in rows[:25]:
        delta_str = f"{delta:+.1%}" if gr_wr > 0 else "  N/A"
        print(f"  {name:<40} {bf_wr:>7.1%} {gr_wr:>7.1%} {delta_str:>7} {gr_hr:>6.1%} {vol:>5.1f}")


def save_recommended(backfill_results, graded_results):
    """Save recommended configs to JSON."""
    # Clean config for JSON serialization
    def clean_cfg(cfg):
        c = {}
        for k, v in cfg.items():
            if isinstance(v, (list, set)):
                c[k] = list(v)
            elif isinstance(v, (np.floating, np.integer)):
                c[k] = float(v)
            else:
                c[k] = v
        return c

    graded_map = {config_name(r['config']): r for r in graded_results}

    # Best overall: highest combined (backfill + graded) 3-leg WR
    best_overall = None
    best_overall_score = -1
    for r in backfill_results:
        if r.get('skipped'):
            continue
        name = r['name']
        gr = graded_map.get(name, {})
        bf_wr = r.get('wr_3leg', 0)
        gr_wr = gr.get('graded_3leg_wr', 0)
        score = bf_wr * 0.4 + gr_wr * 0.6 if gr_wr > 0 else bf_wr * 0.3
        if score > best_overall_score:
            best_overall_score = score
            best_overall = {
                'config': clean_cfg(r['config']),
                'backfill_3leg_wr': bf_wr,
                'graded_3leg_wr': gr_wr,
                'score': score,
            }

    # Best graded: highest WR on real data
    best_graded = None
    best_graded_wr = -1
    for r in graded_results:
        if r.get('graded_eligible', 0) < 5:
            continue
        if r['graded_3leg_wr'] > best_graded_wr:
            best_graded_wr = r['graded_3leg_wr']
            best_graded = {
                'config': clean_cfg(r['config']),
                'graded_3leg_wr': r['graded_3leg_wr'],
                'graded_hr': r['graded_hr'],
                'graded_eligible': r['graded_eligible'],
            }

    # Best volume: best WR with avg 5+ picks/day
    best_volume = None
    best_vol_wr = -1
    for r in graded_results:
        if r.get('graded_avg_per_day', 0) >= 5 and r.get('graded_eligible', 0) >= 10:
            if r['graded_3leg_wr'] > best_vol_wr:
                best_vol_wr = r['graded_3leg_wr']
                best_volume = {
                    'config': clean_cfg(r['config']),
                    'graded_3leg_wr': r['graded_3leg_wr'],
                    'graded_hr': r['graded_hr'],
                    'graded_avg_per_day': r['graded_avg_per_day'],
                }

    # All ranked: combine backfill and graded scores
    all_ranked = []
    for r in backfill_results:
        if r.get('skipped'):
            continue
        name = r['name']
        gr = graded_map.get(name, {})
        entry = {
            'config': clean_cfg(r['config']),
            'backfill_ind_hr': r.get('ind_hr', 0),
            'backfill_2leg_wr': r.get('wr_2leg', 0),
            'backfill_3leg_wr': r.get('wr_3leg', 0),
            'backfill_eligible': r.get('eligible', 0),
            'backfill_avg_cands': r.get('avg_cands_per_day', 0),
            'graded_hr': gr.get('graded_hr', 0),
            'graded_2leg_wr': gr.get('graded_2leg_wr', 0),
            'graded_3leg_wr': gr.get('graded_3leg_wr', 0),
            'graded_eligible': gr.get('graded_eligible', 0),
            'graded_avg_per_day': gr.get('graded_avg_per_day', 0),
        }
        all_ranked.append(entry)

    # Sort by combined score
    all_ranked.sort(
        key=lambda x: x['backfill_3leg_wr'] * 0.4 + x['graded_3leg_wr'] * 0.6
            if x['graded_3leg_wr'] > 0 else x['backfill_3leg_wr'] * 0.3,
        reverse=True
    )

    recommended = {
        'generated': datetime.now().isoformat(),
        'best_overall': best_overall,
        'best_graded': best_graded,
        'best_volume': best_volume,
        'all_ranked': all_ranked[:50],  # top 50
    }

    with open(RECOMMENDED_PATH, 'w') as f:
        json.dump(recommended, f, indent=2, default=str)
    print(f"\n  Recommended configs saved to: {RECOMMENDED_PATH}")

    return recommended


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    graded_only = '--graded-only' in sys.argv
    full_mode = '--full' in sys.argv
    start_time = time.time()

    all_backfill_results = []

    if not graded_only:
        # ── Load backfill data ──
        print(f"\n{'='*60}")
        print(f"  AUTO-OPTIMIZER: SYSTEMATIC GRID SEARCH")
        print(f"{'='*60}")

        all_records, by_date, dates_sorted = load_all_data()

        # ── Round 1: Coarse grid ──
        print(f"\n{'='*60}")
        print(f"  ROUND 1: COARSE GRID SEARCH")
        print(f"{'='*60}")

        coarse_configs = generate_coarse_grid()
        coarse_results = run_coarse_grid(by_date, dates_sorted, coarse_configs, full_dates=full_mode)
        all_backfill_results.extend(coarse_results)

        # Extract top configs (non-skipped)
        top_coarse = [r for r in coarse_results if not r.get('skipped')][:10]
        top_configs_r1 = [r['config'] for r in top_coarse]

        # ── Round 2: Refinement ──
        print(f"\n{'='*60}")
        print(f"  ROUND 2: REFINEMENT OF TOP 10")
        print(f"{'='*60}")

        refine_configs = generate_refinement_grid(top_configs_r1, n=10)
        refine_results = run_coarse_grid(by_date, dates_sorted, refine_configs, full_dates=full_mode)
        all_backfill_results.extend(refine_results)

        # Merge top from round 1 + round 2
        all_non_skipped = [r for r in all_backfill_results if not r.get('skipped')]
        all_non_skipped.sort(key=lambda r: r.get('combined_wr', 0), reverse=True)
        top_configs_r2 = [r['config'] for r in all_non_skipped[:5]]

        # ── Round 3: Compound filters ──
        print(f"\n{'='*60}")
        print(f"  ROUND 3: COMPOUND FILTERS ON TOP 5")
        print(f"{'='*60}")

        compound_configs = generate_compound_grid(top_configs_r2, n=5)
        compound_results = run_coarse_grid(by_date, dates_sorted, compound_configs, full_dates=full_mode)
        all_backfill_results.extend(compound_results)

        # Final sort
        all_backfill_results = [r for r in all_backfill_results if not r.get('skipped')]
        all_backfill_results.sort(key=lambda r: r.get('combined_wr', 0), reverse=True)

        elapsed_backfill = time.time() - start_time
        print(f"\n  Backfill rounds complete: {elapsed_backfill:.0f}s "
              f"({elapsed_backfill/60:.1f}min)")
        print(f"  Total configs tested: {len(all_backfill_results)}")
        print(f"  Best 3-leg WR: {all_backfill_results[0]['wr_3leg']:.1%} "
              f"({all_backfill_results[0]['name']})")

    else:
        # Load pre-existing results if available
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH) as f:
                saved = json.load(f)
            all_backfill_results = saved.get('backfill_results', [])
            print(f"  Loaded {len(all_backfill_results)} pre-existing backfill results")
        else:
            print("  No pre-existing results found. Run without --graded-only first.")

    # ── Round 4: Cross-validate on graded data ──
    print(f"\n{'='*60}")
    print(f"  ROUND 4: CROSS-VALIDATION ON GRADED DATA")
    print(f"{'='*60}")

    # Take top 30 from backfill for cross-validation
    top_for_grading = [r['config'] for r in all_backfill_results[:30]] if all_backfill_results else []

    # Also add some known-good configs that might not be in top 30
    known_good = [
        {
            'name': 'safe_filter_balanced',
            'direction_strategy': 'under_only',
            'min_gap': 5.0,
            'min_l10_hr': 60,
            'max_l10_hr': 70,
            'min_line': 15.0,
            'min_tier': 'F',
            'min_season_hr': 0, 'min_mins_pct': 0, 'max_l10_miss': 10,
            'max_l10_std': None, 'min_games': 0, 'streak_filter': 'none',
            'exclude_combos': False, 'allowed_stats': ALL_STATS,
        },
        {
            'name': 'safe_filter_relaxed',
            'direction_strategy': 'under_only',
            'min_gap': 4.0,
            'min_l10_hr': 60,
            'max_l10_hr': 70,
            'min_line': 15.0,
            'min_tier': 'F',
            'min_season_hr': 0, 'min_mins_pct': 0, 'max_l10_miss': 10,
            'max_l10_std': None, 'min_games': 0, 'streak_filter': 'none',
            'exclude_combos': False, 'allowed_stats': ALL_STATS,
        },
    ]
    # Avoid duplicates by name
    existing_names = {config_name(c) for c in top_for_grading}
    for kg in known_good:
        if config_name(kg) not in existing_names:
            top_for_grading.append(kg)

    graded_results = cross_validate_on_graded(top_for_grading)

    # ── Comparison & output ──
    if all_backfill_results and graded_results:
        build_comparison_table(all_backfill_results, graded_results)

    recommended = save_recommended(all_backfill_results, graded_results)

    # Save all results
    save_data = {
        'generated': datetime.now().isoformat(),
        'backfill_results': [
            {k: v for k, v in r.items() if k != 'config' or True}
            for r in all_backfill_results[:100]
        ],
        'graded_results': [
            {k: (list(v) if isinstance(v, set) else v) for k, v in r.items()}
            for r in graded_results[:50]
        ],
    }

    # Clean numpy types for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, set):
            return list(obj)
        return obj

    save_data = clean_for_json(save_data)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  All results saved to: {RESULTS_PATH}")

    # ── Final summary ──
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  AUTO-OPTIMIZER COMPLETE")
    print(f"{'='*60}")
    print(f"  Total runtime: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Configs tested: {len(all_backfill_results)}")

    if recommended.get('best_overall'):
        bo = recommended['best_overall']
        print(f"\n  BEST OVERALL: {config_name(bo['config'])}")
        print(f"    Backfill 3-leg WR: {bo.get('backfill_3leg_wr', 0):.1%}")
        print(f"    Graded 3-leg WR:   {bo.get('graded_3leg_wr', 0):.1%}")

    if recommended.get('best_graded'):
        bg = recommended['best_graded']
        print(f"\n  BEST ON GRADED DATA: {config_name(bg['config'])}")
        print(f"    Graded 3-leg WR: {bg.get('graded_3leg_wr', 0):.1%}")
        print(f"    Graded HR:       {bg.get('graded_hr', 0):.1%}")
        print(f"    Graded picks:    {bg.get('graded_eligible', 0)}")

    if recommended.get('best_volume'):
        bv = recommended['best_volume']
        print(f"\n  BEST VOLUME (5+/day): {config_name(bv['config'])}")
        print(f"    Graded 3-leg WR: {bv.get('graded_3leg_wr', 0):.1%}")
        print(f"    Avg picks/day:   {bv.get('graded_avg_per_day', 0):.1f}")

    print()


if __name__ == '__main__':
    main()

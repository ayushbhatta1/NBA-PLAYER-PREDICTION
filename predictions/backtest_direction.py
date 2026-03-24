#!/usr/bin/env python3
"""
Massive direction model backtest using 10 years of NBA data.
For each player-game (starting from game 11), reconstructs what a sportsbook
line would look like, computes rolling features, and tests whether OVER or
UNDER was the correct call. Tests multiple direction strategies.

Output: JSON with strategy comparison results.
"""

import csv
import json
import sys
import os
from collections import defaultdict
from datetime import datetime

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "NBA Database (1947 - Present)", "PlayerStatistics.csv")

# Stats we care about (matching sportsbook prop categories)
STAT_MAP = {
    'pts': lambda r: float(r.get('points', 0) or 0),
    'reb': lambda r: float(r.get('reboundsTotal', 0) or 0),
    'ast': lambda r: float(r.get('assists', 0) or 0),
    '3pm': lambda r: float(r.get('threePointersMade', 0) or 0),
    'blk': lambda r: float(r.get('blocks', 0) or 0),
    'stl': lambda r: float(r.get('steals', 0) or 0),
}

COMBO_MAP = {
    'pra': lambda r: float(r.get('points', 0) or 0) + float(r.get('reboundsTotal', 0) or 0) + float(r.get('assists', 0) or 0),
    'pr': lambda r: float(r.get('points', 0) or 0) + float(r.get('reboundsTotal', 0) or 0),
    'pa': lambda r: float(r.get('points', 0) or 0) + float(r.get('assists', 0) or 0),
    'ra': lambda r: float(r.get('reboundsTotal', 0) or 0) + float(r.get('assists', 0) or 0),
}

ALL_STATS = {**STAT_MAP, **COMBO_MAP}

# Stat-type UNDER bonus for confidence scoring
STAT_UNDER_BONUS = {
    'blk': 2.0, 'stl': 1.5, 'stl_blk': 2.5,
    '3pm': 1.0, 'pa': 0.8, 'ast': 0.5, 'ra': 0.3,
}

MIN_GAMES = 10  # Need at least 10 prior games for rolling stats
MIN_MINUTES = 10  # Skip DNP games


def parse_minutes(mins_str):
    """Parse minutes string like '35:12' or '35.12' to float."""
    if not mins_str:
        return 0
    try:
        if ':' in str(mins_str):
            parts = str(mins_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60
        return float(mins_str)
    except (ValueError, IndexError):
        return 0


def compute_rolling_features(games, stat_fn, line, window_l10=10, window_l5=5, window_l3=3):
    """Compute rolling features from prior games. Returns dict of features."""
    vals = [stat_fn(g) for g in games]
    n = len(vals)
    
    season_avg = sum(vals) / n
    l10_avg = sum(vals[-window_l10:]) / min(n, window_l10)
    l5_avg = sum(vals[-window_l5:]) / min(n, window_l5)
    l3_avg = sum(vals[-window_l3:]) / min(n, window_l3)
    
    # Hit rates (how often player goes OVER the line)
    l10_vals = vals[-window_l10:]
    l10_hr = sum(1 for v in l10_vals if v > line) / len(l10_vals) * 100
    season_hr = sum(1 for v in vals if v > line) / n * 100
    l5_vals = vals[-window_l5:]
    l5_hr = sum(1 for v in l5_vals if v > line) / len(l5_vals) * 100
    
    # Miss count (games under line in L10)
    l10_miss_count = sum(1 for v in l10_vals if v < line)
    
    # Blend hit rate
    blend_hr = 0.6 * l10_hr + 0.4 * season_hr
    
    # Streak detection
    if l3_avg > l10_avg * 1.15:
        streak = 'HOT'
    elif l3_avg < l10_avg * 0.85:
        streak = 'COLD'
    else:
        streak = 'NEUTRAL'
    
    # Projection (weighted average, same as analyze_v3)
    raw_proj = 0.4 * season_avg + 0.35 * l10_avg + 0.25 * l5_avg
    
    # Market-calibrated projection (blend with line)
    if blend_hr >= 70 or blend_hr <= 30:
        mkt_wt = 0.15
    elif blend_hr >= 60 or blend_hr <= 40:
        mkt_wt = 0.25
    else:
        mkt_wt = 0.40
    base_proj = (1 - mkt_wt) * raw_proj + mkt_wt * line
    
    # Gap
    gap = base_proj - line
    
    # Floor (minimum in L10)
    l10_floor = min(l10_vals) if l10_vals else 0
    
    return {
        'season_avg': season_avg,
        'l10_avg': l10_avg,
        'l5_avg': l5_avg,
        'l3_avg': l3_avg,
        'l10_hr': l10_hr,
        'l5_hr': l5_hr,
        'season_hr': season_hr,
        'blend_hr': blend_hr,
        'l10_miss_count': l10_miss_count,
        'streak': streak,
        'raw_proj': raw_proj,
        'base_proj': base_proj,
        'gap': gap,
        'l10_floor': l10_floor,
        'games_played': n,
    }


def under_confidence_score(features, stat):
    """Same composite UNDER confidence score as analyze_v3.py v14."""
    score = 0.0
    
    hr = features['l10_hr']
    if hr < 20: score += 3.0
    elif hr < 35: score += 2.0
    elif hr < 45: score += 1.0
    elif hr < 55: score += 0
    elif hr < 65: score -= 0.5
    elif hr < 80: score -= 1.0
    else: score -= 2.0
    
    shr = features['season_hr']
    if shr < 30: score += 2.0
    elif shr < 45: score += 0.5
    elif shr < 55: score += 0
    elif shr < 70: score -= 0.5
    else: score -= 1.0
    
    score += STAT_UNDER_BONUS.get(stat, 0)
    
    streak = features['streak']
    if streak == 'COLD': score += 1.0
    elif streak == 'HOT': score -= 0.5
    
    gap = features['gap']
    if gap < -5: score += 2.0
    elif gap < -3: score += 1.5
    elif gap < -1.5: score += 1.0
    elif gap < 0: score += 0.5
    elif gap < 1.5: score += 0
    elif gap < 3: score -= 0.5
    else: score -= 0.5
    
    mc = features['l10_miss_count']
    if mc >= 9: score += 2.0
    elif mc >= 7: score += 1.0
    elif mc >= 5: score += 0.3
    elif mc < 3: score -= 0.5
    
    return round(score, 1)


def main():
    print("Loading CSV...", file=sys.stderr)
    
    # Load all regular season games from 2016 onwards, sorted by date per player
    player_games = defaultdict(list)  # player_id -> list of game dicts sorted by date
    
    row_count = 0
    skip_count = 0
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get('gameDateTimeEst', '')
            if not date or date < '2016-01-01':
                continue
            if row.get('gameType', '') != 'Regular Season':
                continue
            
            mins = parse_minutes(row.get('numMinutes', 0))
            if mins < MIN_MINUTES:
                skip_count += 1
                continue
            
            pid = row.get('personId', '')
            if not pid:
                continue
            
            player_games[pid].append({
                'date': date[:10],
                'name': f"{row.get('firstName', '')} {row.get('lastName', '')}",
                'opponent': f"{row.get('opponentteamCity', '')} {row.get('opponentteamName', '')}",
                'home': row.get('home', '0'),
                'minutes': mins,
                'points': row.get('points', 0),
                'assists': row.get('assists', 0),
                'blocks': row.get('blocks', 0),
                'steals': row.get('steals', 0),
                'reboundsTotal': row.get('reboundsTotal', 0),
                'threePointersMade': row.get('threePointersMade', 0),
                'foulsPersonal': row.get('foulsPersonal', 0),
                'plusMinusPoints': row.get('plusMinusPoints', 0),
                'win': row.get('win', 0),
            })
            row_count += 1
    
    print(f"Loaded {row_count} games for {len(player_games)} players (skipped {skip_count} DNP)", file=sys.stderr)
    
    # Sort each player's games by date
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g['date'])
    
    # For each player-game (starting from game 11), simulate a prop bet
    # Use L10 average as the "sportsbook line" (what the book would set)
    # Then test if actual went OVER or UNDER
    
    print("Running backtest...", file=sys.stderr)
    
    # Track results by strategy
    results = {
        'total_props': 0,
        'over_actual': 0,
        'under_actual': 0,
        'by_stat': {},
        'by_year': {},
        'by_confidence_bucket': {},
        'strategies': {},
    }
    
    # Define strategies to test
    strategy_names = [
        'always_under',
        'always_over', 
        'gap_based_v13',          # old: OVER if projection > line
        'v14_under_dominant',      # new: UNDER unless gap>3 AND HR>=65
        'hr_under_50',            # UNDER if L10 HR < 50
        'hr_under_55',            # UNDER if L10 HR < 55
        'hr_under_60',            # UNDER if L10 HR < 60
        'conf_score_top',         # UNDER if conf_score > 0
        'conf_score_s_tier',      # UNDER if conf_score >= 5
        'conf_score_sa_tier',     # UNDER if conf_score >= 3.5
        'miss_count_5plus',       # UNDER if miss_count >= 5
        'miss_count_7plus',       # UNDER if miss_count >= 7
        'cold_under',             # UNDER if COLD streak
        'season_under_line',      # UNDER if season_avg < line
    ]
    
    for s in strategy_names:
        results['strategies'][s] = {'correct': 0, 'total': 0, 'over_calls': 0, 'under_calls': 0, 
                                     'over_correct': 0, 'under_correct': 0}
    
    processed = 0
    # Only test base stats (not combos) for speed, plus pra as the main combo
    test_stats = ['pts', 'reb', 'ast', '3pm', 'blk', 'stl', 'pra']
    
    for pid, games in player_games.items():
        if len(games) < MIN_GAMES + 1:
            continue
        
        for i in range(MIN_GAMES, len(games)):
            prior_games = games[:i]  # All games before this one
            current_game = games[i]
            
            for stat in test_stats:
                stat_fn = ALL_STATS[stat]
                actual = stat_fn(current_game)
                
                # Simulate sportsbook line: use L10 average (industry standard)
                l10_games = prior_games[-10:]
                l10_values = [stat_fn(g) for g in l10_games]
                line = sum(l10_values) / len(l10_values)
                
                # Some lines are too low to be realistic props
                if stat == 'pts' and line < 5: continue
                if stat == 'reb' and line < 2: continue
                if stat == 'ast' and line < 1: continue
                if stat == 'pra' and line < 10: continue
                if stat in ('blk', 'stl') and line < 0.5: continue
                if stat == '3pm' and line < 0.5: continue
                
                # Round line to 0.5 (like real sportsbooks)
                line = round(line * 2) / 2
                
                # Skip pushes
                if actual == line:
                    continue
                
                # Compute features
                features = compute_rolling_features(prior_games, stat_fn, line)
                conf_score = under_confidence_score(features, stat)
                
                actual_dir = 'OVER' if actual > line else 'UNDER'
                year = current_game['date'][:4]
                
                results['total_props'] += 1
                if actual_dir == 'OVER':
                    results['over_actual'] += 1
                else:
                    results['under_actual'] += 1
                
                # Track by stat
                results['by_stat'].setdefault(stat, {'over': 0, 'under': 0, 'total': 0})
                results['by_stat'][stat]['total'] += 1
                results['by_stat'][stat][actual_dir.lower()] += 1
                
                # Track by year
                results['by_year'].setdefault(year, {'over': 0, 'under': 0, 'total': 0})
                results['by_year'][year]['total'] += 1
                results['by_year'][year][actual_dir.lower()] += 1
                
                # Track by confidence bucket
                if conf_score >= 5:
                    bucket = 'S (>=5)'
                elif conf_score >= 3.5:
                    bucket = 'A (3.5-5)'
                elif conf_score >= 2:
                    bucket = 'B (2-3.5)'
                elif conf_score >= 0.5:
                    bucket = 'C (0.5-2)'
                elif conf_score >= -1:
                    bucket = 'D (-1-0.5)'
                else:
                    bucket = 'F (<-1)'
                
                results['by_confidence_bucket'].setdefault(bucket, {'under_hit': 0, 'total': 0})
                results['by_confidence_bucket'][bucket]['total'] += 1
                if actual_dir == 'UNDER':
                    results['by_confidence_bucket'][bucket]['under_hit'] += 1
                
                # Test each strategy
                gap = features['gap']
                blend_hr = features['blend_hr']
                l10_hr = features['l10_hr']
                mc = features['l10_miss_count']
                streak = features['streak']
                
                strategy_calls = {
                    'always_under': 'UNDER',
                    'always_over': 'OVER',
                    'gap_based_v13': 'OVER' if gap > 0 else 'UNDER',
                    'v14_under_dominant': 'OVER' if (gap > 3.0 and blend_hr >= 65) else 'UNDER',
                    'hr_under_50': 'UNDER' if l10_hr < 50 else 'OVER',
                    'hr_under_55': 'UNDER' if l10_hr < 55 else 'OVER',
                    'hr_under_60': 'UNDER' if l10_hr < 60 else 'OVER',
                    'conf_score_top': 'UNDER' if conf_score > 0 else 'OVER',
                    'conf_score_s_tier': 'UNDER' if conf_score >= 5 else 'OVER',
                    'conf_score_sa_tier': 'UNDER' if conf_score >= 3.5 else 'OVER',
                    'miss_count_5plus': 'UNDER' if mc >= 5 else 'OVER',
                    'miss_count_7plus': 'UNDER' if mc >= 7 else 'OVER',
                    'cold_under': 'UNDER' if streak == 'COLD' else 'OVER',
                    'season_under_line': 'UNDER' if features['season_avg'] < line else 'OVER',
                }
                
                for sname, call in strategy_calls.items():
                    s = results['strategies'][sname]
                    s['total'] += 1
                    if call == actual_dir:
                        s['correct'] += 1
                    if call == 'OVER':
                        s['over_calls'] += 1
                        if actual_dir == 'OVER':
                            s['over_correct'] += 1
                    else:
                        s['under_calls'] += 1
                        if actual_dir == 'UNDER':
                            s['under_correct'] += 1
                
                processed += 1
                if processed % 500000 == 0:
                    print(f"  Processed {processed:,} props...", file=sys.stderr)
    
    print(f"Done. Total props evaluated: {processed:,}", file=sys.stderr)
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), 'backtest_direction_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"BACKTEST RESULTS: {results['total_props']:,} props ({results['over_actual']:,} OVER / {results['under_actual']:,} UNDER)")
    print(f"Base OVER rate: {results['over_actual']/results['total_props']:.1%}")
    print(f"Base UNDER rate: {results['under_actual']/results['total_props']:.1%}")
    
    print(f"\n{'STRATEGY':<25} {'ACCURACY':>10} {'OVER_CALLS':>12} {'OVER_HR':>10} {'UNDER_CALLS':>13} {'UNDER_HR':>10}")
    print("=" * 82)
    for sname in strategy_names:
        s = results['strategies'][sname]
        acc = s['correct'] / max(s['total'], 1)
        o_hr = s['over_correct'] / max(s['over_calls'], 1)
        u_hr = s['under_correct'] / max(s['under_calls'], 1)
        print(f"  {sname:<23} {acc:>9.1%} {s['over_calls']:>12,} {o_hr:>9.1%} {s['under_calls']:>13,} {u_hr:>9.1%}")
    
    print(f"\n=== BY STAT (UNDER rate) ===")
    for stat in test_stats:
        d = results['by_stat'].get(stat, {})
        total = d.get('total', 0)
        under = d.get('under', 0)
        if total > 0:
            print(f"  {stat:5s}: {under}/{total} = {under/total:.1%}")
    
    print(f"\n=== BY YEAR (UNDER rate) ===")
    for year in sorted(results['by_year'].keys()):
        d = results['by_year'][year]
        total = d['total']
        under = d['under']
        print(f"  {year}: {under}/{total} = {under/total:.1%}")
    
    print(f"\n=== CONFIDENCE TIER UNDER HR ===")
    for bucket in ['S (>=5)', 'A (3.5-5)', 'B (2-3.5)', 'C (0.5-2)', 'D (-1-0.5)', 'F (<-1)']:
        d = results['by_confidence_bucket'].get(bucket, {})
        total = d.get('total', 0)
        under_hit = d.get('under_hit', 0)
        if total > 0:
            print(f"  {bucket:15s}: {under_hit}/{total} = {under_hit/total:.1%}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Deep backtest: finer confidence thresholds, sportsbook line inflation simulation,
stat-specific optimization, and parlay simulation across 10 years.
"""

import csv
import json
import sys
import os
from collections import defaultdict

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "NBA Database (1947 - Present)", "PlayerStatistics.csv")

STAT_MAP = {
    'pts': lambda r: float(r.get('points', 0) or 0),
    'reb': lambda r: float(r.get('reboundsTotal', 0) or 0),
    'ast': lambda r: float(r.get('assists', 0) or 0),
    '3pm': lambda r: float(r.get('threePointersMade', 0) or 0),
    'blk': lambda r: float(r.get('blocks', 0) or 0),
    'stl': lambda r: float(r.get('steals', 0) or 0),
    'pra': lambda r: float(r.get('points', 0) or 0) + float(r.get('reboundsTotal', 0) or 0) + float(r.get('assists', 0) or 0),
}

STAT_UNDER_BONUS = {
    'blk': 2.0, 'stl': 1.5, '3pm': 1.0, 'pa': 0.8, 'ast': 0.5, 'ra': 0.3,
}

MIN_GAMES = 10
MIN_MINUTES = 10


def parse_minutes(mins_str):
    if not mins_str: return 0
    try:
        if ':' in str(mins_str):
            parts = str(mins_str).split(':')
            return float(parts[0]) + float(parts[1]) / 60
        return float(mins_str)
    except: return 0


def compute_features(games, stat_fn, line):
    vals = [stat_fn(g) for g in games]
    n = len(vals)
    season_avg = sum(vals) / n
    l10 = vals[-10:]
    l5 = vals[-5:]
    l3 = vals[-3:]
    l10_avg = sum(l10) / len(l10)
    l5_avg = sum(l5) / len(l5)
    l3_avg = sum(l3) / len(l3)
    
    l10_hr = sum(1 for v in l10 if v > line) / len(l10) * 100
    season_hr = sum(1 for v in vals if v > line) / n * 100
    blend_hr = 0.6 * l10_hr + 0.4 * season_hr
    l10_miss = sum(1 for v in l10 if v < line)
    
    if l3_avg > l10_avg * 1.15: streak = 'COLD'  # Flipped from before — COLD means trending DOWN
    elif l3_avg < l10_avg * 0.85: streak = 'COLD'
    else: streak = 'NEUTRAL'
    # Fix: HOT = L3 > L10*1.15, COLD = L3 < L10*0.85
    if l3_avg > l10_avg * 1.15: streak = 'HOT'
    elif l3_avg < l10_avg * 0.85: streak = 'COLD'
    else: streak = 'NEUTRAL'
    
    raw_proj = 0.4 * season_avg + 0.35 * l10_avg + 0.25 * l5_avg
    if blend_hr >= 70 or blend_hr <= 30: mkt_wt = 0.15
    elif blend_hr >= 60 or blend_hr <= 40: mkt_wt = 0.25
    else: mkt_wt = 0.40
    base_proj = (1 - mkt_wt) * raw_proj + mkt_wt * line
    gap = base_proj - line
    
    return {
        'season_avg': season_avg, 'l10_avg': l10_avg, 'l5_avg': l5_avg,
        'l10_hr': l10_hr, 'season_hr': season_hr, 'blend_hr': blend_hr,
        'l10_miss': l10_miss, 'streak': streak, 'gap': gap, 'raw_proj': raw_proj,
    }


def conf_score(f, stat, is_home=None):
    s = 0.0
    hr = f['l10_hr']
    if hr < 20: s += 3.0
    elif hr < 35: s += 2.0
    elif hr < 45: s += 1.0
    elif hr >= 80: s -= 2.0
    elif hr >= 65: s -= 1.0
    elif hr >= 55: s -= 0.5

    shr = f['season_hr']
    if shr < 30: s += 2.0
    elif shr < 45: s += 0.5
    elif shr >= 70: s -= 1.0
    elif shr >= 55: s -= 0.5

    s += STAT_UNDER_BONUS.get(stat, 0)
    if f['streak'] == 'COLD': s += 1.0
    elif f['streak'] == 'HOT': s -= 0.5

    gap = f['gap']
    if gap < -5: s += 2.0
    elif gap < -3: s += 1.5
    elif gap < -1.5: s += 1.0
    elif gap < 0: s += 0.5
    elif gap < 3: s -= 0.5
    else: s -= 0.5

    mc = f['l10_miss']
    if mc >= 9: s += 2.0
    elif mc >= 7: s += 1.0
    elif mc >= 5: s += 0.3
    elif mc < 3: s -= 0.5

    # Away game bonus: players tend to underperform on the road
    if is_home is not None and not is_home:
        s += 0.5

    return round(s, 1)


def main():
    print("Loading CSV...", file=sys.stderr)
    player_games = defaultdict(list)
    row_count = 0
    
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get('gameDateTimeEst', '')
            if not date or date < '2016-01-01': continue
            if row.get('gameType', '') != 'Regular Season': continue
            mins = parse_minutes(row.get('numMinutes', 0))
            if mins < MIN_MINUTES: continue
            pid = row.get('personId', '')
            if not pid: continue
            player_games[pid].append({
                'date': date[:10],
                'name': f"{row.get('firstName', '')} {row.get('lastName', '')}",
                'points': row.get('points', 0),
                'assists': row.get('assists', 0),
                'blocks': row.get('blocks', 0),
                'steals': row.get('steals', 0),
                'reboundsTotal': row.get('reboundsTotal', 0),
                'threePointersMade': row.get('threePointersMade', 0),
                'home': row.get('home', ''),
            })
            row_count += 1
    
    print(f"Loaded {row_count} games for {len(player_games)} players", file=sys.stderr)
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g['date'])
    
    # Track finer confidence buckets and line inflation effects
    conf_buckets = defaultdict(lambda: [0, 0])  # conf_score -> [under_hits, total]
    stat_conf = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # stat -> conf -> [under, total]
    inflation_results = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # inflation% -> conf_bucket -> [under, total]

    # Parlay simulation: track daily S-tier picks (WITH away bonus)
    daily_picks = defaultdict(list)  # date -> [(conf, actual_under, player, stat)]
    # Also track WITHOUT away bonus for comparison
    daily_picks_no_away = defaultdict(list)
    
    processed = 0
    test_stats = ['pts', 'reb', 'ast', '3pm', 'blk', 'stl', 'pra']
    
    for pid, games in player_games.items():
        if len(games) < MIN_GAMES + 1: continue
        
        for i in range(MIN_GAMES, len(games)):
            prior = games[:i]
            current = games[i]
            
            for stat in test_stats:
                stat_fn = STAT_MAP[stat]
                actual = stat_fn(current)
                
                l10 = prior[-10:]
                l10_vals = [stat_fn(g) for g in l10]
                base_line = sum(l10_vals) / len(l10_vals)
                
                # Skip unrealistic props
                if stat == 'pts' and base_line < 5: continue
                if stat == 'reb' and base_line < 2: continue
                if stat == 'ast' and base_line < 1: continue
                if stat == 'pra' and base_line < 10: continue
                if stat in ('blk', 'stl') and base_line < 0.5: continue
                if stat == '3pm' and base_line < 0.5: continue
                
                # Determine home/away for this game
                home_val = current.get('home', '')
                if home_val in (True, 'True', 'true', '1', 1):
                    is_home = True
                elif home_val in (False, 'False', 'false', '0', 0):
                    is_home = False
                else:
                    is_home = None

                # Test multiple line inflation levels (simulating sportsbook markup)
                for inflation_pct in [0, 2, 4, 6, 8]:
                    line = round((base_line * (1 + inflation_pct / 100)) * 2) / 2
                    if actual == line: continue

                    f = compute_features(prior, stat_fn, line)
                    cs_with_away = conf_score(f, stat, is_home=is_home)
                    cs_no_away = conf_score(f, stat, is_home=None)  # baseline without away bonus
                    actual_under = actual < line

                    if inflation_pct == 0:
                        # Fine-grained confidence tracking (WITH away bonus)
                        cs_bucket = round(cs_with_away)
                        conf_buckets[cs_bucket][1] += 1
                        if actual_under: conf_buckets[cs_bucket][0] += 1

                        stat_conf[stat][cs_bucket][1] += 1
                        if actual_under: stat_conf[stat][cs_bucket][0] += 1

                        # Daily picks for parlay sim (WITH away bonus)
                        if cs_with_away >= 5:
                            daily_picks[current['date']].append((cs_with_away, actual_under, current['name'], stat))
                        # Daily picks WITHOUT away bonus for comparison
                        if cs_no_away >= 5:
                            daily_picks_no_away[current['date']].append((cs_no_away, actual_under, current['name'], stat))
                    
                    # Track by inflation level (using away-bonus version)
                    cs = cs_with_away
                    if cs >= 5: ibucket = 'S'
                    elif cs >= 3.5: ibucket = 'A'
                    elif cs >= 2: ibucket = 'B'
                    elif cs >= 0: ibucket = 'C+'
                    else: ibucket = 'D/F'
                    
                    inflation_results[inflation_pct][ibucket][1] += 1
                    if actual_under:
                        inflation_results[inflation_pct][ibucket][0] += 1
                
                processed += 1
                if processed % 500000 == 0:
                    print(f"  {processed:,}...", file=sys.stderr)
    
    print(f"Done: {processed:,} props", file=sys.stderr)
    
    # === RESULTS ===
    print(f"\n{'='*70}")
    print(f"DEEP BACKTEST: {processed:,} base props")
    
    # 1. Fine-grained confidence score UNDER rates
    print(f"\n=== CONFIDENCE SCORE UNDER HR (integer buckets) ===")
    for cs in sorted(conf_buckets.keys()):
        u, t = conf_buckets[cs]
        if t >= 100:
            print(f"  conf={cs:+3d}: {u}/{t:>7,} = {u/t:.1%}")
    
    # 2. By stat type
    print(f"\n=== CONF SCORE UNDER HR BY STAT (S-tier only, conf>=5) ===")
    for stat in test_stats:
        total_u = total_t = 0
        for cs, (u, t) in stat_conf[stat].items():
            if cs >= 5:
                total_u += u; total_t += t
        if total_t >= 50:
            print(f"  {stat:5s}: {total_u}/{total_t:>6,} = {total_u/total_t:.1%}")
    
    # 3. Line inflation effect on tier accuracy
    print(f"\n=== LINE INFLATION EFFECT (simulates sportsbook markup) ===")
    print(f"  {'Inflation':>10} {'S-tier':>12} {'A-tier':>12} {'B-tier':>12} {'C+':>12}")
    for inf in [0, 2, 4, 6, 8]:
        parts = []
        for tier in ['S', 'A', 'B', 'C+']:
            u, t = inflation_results[inf][tier]
            if t > 0:
                parts.append(f"{u/t:.1%} ({t//1000}K)")
            else:
                parts.append("-")
        print(f"  {inf:>8}%  {'  '.join(f'{p:>12}' for p in parts)}")
    
    # 4. Parlay simulation — compare WITH vs WITHOUT away bonus
    def run_parlay_sim(picks_dict, label):
        parlay_wins = 0
        parlay_total = 0
        total_legs_hit = 0
        total_legs = 0
        yearly_results = defaultdict(lambda: [0, 0])

        for date in sorted(picks_dict.keys()):
            picks = sorted(picks_dict[date], key=lambda x: x[0], reverse=True)
            sel = []
            used_names = set()
            for cs, u, name, stat in picks:
                if name in used_names: continue
                sel.append((cs, u, name, stat))
                used_names.add(name)
                if len(sel) >= 3: break

            if len(sel) < 3: continue
            all_hit = all(u for _, u, _, _ in sel)
            legs = sum(u for _, u, _, _ in sel)
            parlay_total += 1
            total_legs += 3
            total_legs_hit += legs
            if all_hit: parlay_wins += 1

            year = date[:4]
            yearly_results[year][1] += 1
            if all_hit: yearly_results[year][0] += 1

        print(f"\n=== {label} ===")
        print(f"  TOTAL: {parlay_wins}/{parlay_total} = {parlay_wins/max(parlay_total,1):.1%} ({total_legs_hit}/{total_legs} = {total_legs_hit/max(total_legs,1):.1%} legs)")
        print(f"\n  By year:")
        for year in sorted(yearly_results.keys()):
            w, t = yearly_results[year]
            print(f"    {year}: {w}/{t} = {w/max(t,1):.1%}")
        return parlay_wins, parlay_total

    w1, t1 = run_parlay_sim(daily_picks, "PARLAY SIM: WITH AWAY BONUS (conf_score + 0.5 for away games)")
    w2, t2 = run_parlay_sim(daily_picks_no_away, "PARLAY SIM: WITHOUT AWAY BONUS (baseline)")
    print(f"\n=== COMPARISON ===")
    r1 = w1/max(t1,1)*100; r2 = w2/max(t2,1)*100
    print(f"  WITH away bonus:    {w1}/{t1} = {r1:.1f}%")
    print(f"  WITHOUT away bonus: {w2}/{t2} = {r2:.1f}%")
    print(f"  Difference: {r1-r2:+.1f}pp")
    
    # Save detailed results
    output = {
        'conf_buckets': {str(k): {'under': v[0], 'total': v[1]} for k, v in conf_buckets.items()},
        'stat_stier': {},
        'inflation': {},
        'parlay_with_away': {'wins': w1, 'total': t1},
        'parlay_no_away': {'wins': w2, 'total': t2},
    }
    
    with open(os.path.join(os.path.dirname(__file__), 'backtest_deep_results.json'), 'w') as f:
        json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()

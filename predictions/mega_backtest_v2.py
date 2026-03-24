#!/usr/bin/env python3
"""
MEGA BACKTEST V2: Deep dive into the winning patterns from V1.
Focus on: 2-leg parlays, BLK/STL specialization, floor analysis,
hybrid multi-stat combos, and ultra-selective filtering.
Also: dynamic skip-day logic (only bet when conditions are perfect).
"""

import csv, json, sys, os, math
from collections import defaultdict
from datetime import datetime

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "NBA Database (1947 - Present)", "PlayerStatistics.csv")

MIN_LINES = {'pts': 5, 'reb': 2, 'ast': 1, '3pm': 0.5, 'blk': 0.5, 'stl': 0.5, 'pra': 10}

def parse_minutes(s):
    if not s: return 0
    try:
        if ':' in str(s): p = str(s).split(':'); return float(p[0]) + float(p[1])/60
        return float(s)
    except: return 0

def load_data():
    print("Loading CSV...", file=sys.stderr)
    player_games = defaultdict(list)
    count = 0
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            date = row.get('gameDateTimeEst', '')
            if not date or date < '2016-01-01': continue
            if row.get('gameType', '') != 'Regular Season': continue
            mins = parse_minutes(row.get('numMinutes', 0))
            if mins < 10: continue
            pid = row.get('personId', '')
            if not pid: continue
            home_val = row.get('home', '')
            is_home = True if home_val in ('True', 'true', '1') else False if home_val in ('False', 'false', '0') else None
            player_games[pid].append({
                'date': date[:10],
                'name': f"{row.get('firstName','')} {row.get('lastName','')}",
                'home': is_home,
                'mins': mins,
                'pts': float(row.get('points', 0) or 0),
                'reb': float(row.get('reboundsTotal', 0) or 0),
                'ast': float(row.get('assists', 0) or 0),
                '3pm': float(row.get('threePointersMade', 0) or 0),
                'blk': float(row.get('blocks', 0) or 0),
                'stl': float(row.get('steals', 0) or 0),
                'pf': float(row.get('foulsPersonal', 0) or 0),
                'pm': float(row.get('plusMinusPoints', 0) or 0),
                'pra': float(row.get('points',0) or 0)+float(row.get('reboundsTotal',0) or 0)+float(row.get('assists',0) or 0),
            })
            count += 1
    print(f"Loaded {count:,} games for {len(player_games)} players", file=sys.stderr)
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g['date'])
    return player_games


def compute_all_props(player_games, stats=['pts','reb','ast','3pm','blk','stl','pra'],
                      inflation=4):
    props = []
    for pid, games in player_games.items():
        if len(games) < 15: continue
        for i in range(15, len(games)):
            current = games[i]
            prior = games[:i]
            for stat in stats:
                l10 = prior[-10:]
                l10_vals = [g[stat] for g in l10]
                l10_avg = sum(l10_vals) / 10
                if l10_avg < MIN_LINES.get(stat, 0.5): continue
                line = round((l10_avg * (1 + inflation/100)) * 2) / 2
                actual = current[stat]
                if actual == line: continue

                l5 = prior[-5:]
                l3 = prior[-3:]
                l5_vals = [g[stat] for g in l5]
                l3_vals = [g[stat] for g in l3]
                l5_avg = sum(l5_vals)/5
                l3_avg = sum(l3_vals)/3
                season_vals = [g[stat] for g in prior if g['date'][:4] == current['date'][:4]] or [g[stat] for g in prior[-30:]]
                season_avg = sum(season_vals)/len(season_vals)

                l10_hr = sum(1 for v in l10_vals if v > line) / 10 * 100
                l5_hr = sum(1 for v in l5_vals if v > line) / 5 * 100
                season_hr = sum(1 for v in season_vals if v > line) / len(season_vals) * 100
                l10_miss = sum(1 for v in l10_vals if v < line)

                l10_std = (sum((v - l10_avg)**2 for v in l10_vals) / 10) ** 0.5
                l10_cv = l10_std / max(l10_avg, 0.1)
                l10_floor = sorted(l10_vals)[1] if len(l10_vals) > 1 else min(l10_vals)
                l10_median = sorted(l10_vals)[5]

                gap = l10_avg - line
                if l3_avg > l10_avg * 1.15: streak = 'HOT'
                elif l3_avg < l10_avg * 0.85: streak = 'COLD'
                else: streak = 'NEUTRAL'

                trend = (l5_avg - l10_avg) / max(l10_avg, 0.1) * 100

                consec_under = 0
                for v in reversed(l10_vals):
                    if v < line: consec_under += 1
                    else: break

                # Rest days
                if i > 0:
                    try:
                        d1 = datetime.strptime(prior[-1]['date'], '%Y-%m-%d')
                        d2 = datetime.strptime(current['date'], '%Y-%m-%d')
                        rest_days = (d2 - d1).days - 1
                    except: rest_days = 1
                else: rest_days = 1

                # Minutes
                l10_mins = [g['mins'] for g in l10]
                mins_avg = sum(l10_mins)/10
                mins_std = (sum((m - mins_avg)**2 for m in l10_mins)/10)**0.5

                props.append({
                    'date': current['date'], 'name': current['name'], 'stat': stat,
                    'line': line, 'actual': actual, 'under': actual < line,
                    'home': current['home'], 'is_away': not current['home'] if current['home'] is not None else None,
                    'l10_avg': l10_avg, 'l5_avg': l5_avg, 'l3_avg': l3_avg, 'season_avg': season_avg,
                    'gap': gap, 'l10_hr': l10_hr, 'l5_hr': l5_hr, 'season_hr': season_hr,
                    'l10_miss': l10_miss, 'l10_std': l10_std, 'l10_cv': l10_cv,
                    'l10_floor': l10_floor, 'l10_median': l10_median,
                    'streak': streak, 'trend': trend, 'consec_under': consec_under,
                    'rest_days': rest_days, 'is_b2b': rest_days == 0,
                    'mins_avg': mins_avg, 'mins_std': mins_std,
                    'l10_pf': sum(g['pf'] for g in l10)/10,
                    'l10_pm': sum(g['pm'] for g in l10)/10,
                })
    return props


def conf_score_v2(p, away_w=0.5, b2b_w=0.3, stat_w=None):
    """Enhanced confidence score."""
    s = 0.0
    hr = p['l10_hr']
    if hr < 20: s += 3.0
    elif hr < 35: s += 2.0
    elif hr < 45: s += 1.0
    elif hr >= 80: s -= 2.0
    elif hr >= 65: s -= 1.0
    elif hr >= 55: s -= 0.5

    shr = p['season_hr']
    if shr < 30: s += 2.0
    elif shr < 45: s += 0.5
    elif shr >= 70: s -= 1.0
    elif shr >= 55: s -= 0.5

    defaults = {'blk': 2.0, 'stl': 1.5, '3pm': 1.0, 'ast': 0.5}
    bonuses = stat_w if stat_w else defaults
    s += bonuses.get(p['stat'], 0)

    if p['streak'] == 'COLD': s += 1.0
    elif p['streak'] == 'HOT': s -= 0.5

    gap = p['gap']
    if gap < -5: s += 2.0
    elif gap < -3: s += 1.5
    elif gap < -1.5: s += 1.0
    elif gap < 0: s += 0.5
    elif gap < 3: s -= 0.5
    else: s -= 0.5

    mc = p['l10_miss']
    if mc >= 9: s += 2.0
    elif mc >= 7: s += 1.0
    elif mc >= 5: s += 0.3
    elif mc < 3: s -= 0.5

    if p.get('is_away'): s += away_w
    if p.get('is_b2b'): s += b2b_w

    return s


def run_strategy(props, filter_fn, sort_fn, legs, name, min_candidates=0, skip_fn=None):
    by_date = defaultdict(list)
    for p in props:
        if filter_fn(p):
            by_date[p['date']].append(p)

    wins = total = leg_hits = leg_total = streak = max_streak = 0
    streaks = []
    daily = []
    skipped = 0

    for date in sorted(by_date.keys()):
        candidates = by_date[date]

        # Selectivity: skip days without enough candidates
        if len(candidates) < max(min_candidates, legs):
            skipped += 1
            continue

        # Dynamic skip: skip bad days
        if skip_fn and skip_fn(candidates):
            skipped += 1
            continue

        candidates.sort(key=sort_fn, reverse=True)
        selected = []
        used_names = set()
        for p in candidates:
            if p['name'] in used_names: continue
            selected.append(p)
            used_names.add(p['name'])
            if len(selected) >= legs: break
        if len(selected) < legs: continue

        all_hit = all(p['under'] for p in selected)
        n_hit = sum(p['under'] for p in selected)
        total += 1
        leg_total += legs
        leg_hits += n_hit
        if all_hit:
            wins += 1
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            if streak > 0: streaks.append(streak)
            streak = 0
        daily.append({'date': date, 'hit': all_hit, 'legs_hit': n_hit})

    if streak > 0: streaks.append(streak)

    return {
        'name': name, 'wins': wins, 'total': total,
        'rate': wins/max(total,1), 'leg_hits': leg_hits, 'leg_total': leg_total,
        'leg_rate': leg_hits/max(leg_total,1), 'max_streak': max_streak,
        'streaks_5plus': sum(1 for s in streaks if s >= 5),
        'streaks_10plus': sum(1 for s in streaks if s >= 10),
        'skipped': skipped, 'daily': daily,
    }


def gen_strategies_v2():
    strategies = []

    # === BLOCK 1: BLK-FOCUSED (our #1 finding) ===
    # BLK only with various score thresholds and sorting
    for thresh in range(2, 9):
        strategies.append((
            lambda p, t=thresh: p['stat'] == 'blk' and conf_score_v2(p) >= t,
            lambda p: conf_score_v2(p),
            2, f"blk_conf{thresh}_2leg"
        ))

    # BLK + floor analysis
    for thresh in [3, 4, 5, 6]:
        strategies.append((
            lambda p, t=thresh: p['stat'] == 'blk' and conf_score_v2(p) >= t and p['l10_floor'] < p['line'],
            lambda p: conf_score_v2(p) + (p['line'] - p['l10_floor']),
            2, f"blk_floor_conf{thresh}_2leg"
        ))

    # BLK + away
    for thresh in [3, 4, 5]:
        strategies.append((
            lambda p, t=thresh: p['stat'] == 'blk' and conf_score_v2(p) >= t and p.get('is_away'),
            lambda p: conf_score_v2(p),
            2, f"blk_away_conf{thresh}_2leg"
        ))

    # BLK + low variance
    for cv in [0.3, 0.5, 0.7, 1.0]:
        strategies.append((
            lambda p, c=cv: p['stat'] == 'blk' and conf_score_v2(p) >= 4 and p['l10_cv'] <= c,
            lambda p: conf_score_v2(p),
            2, f"blk_cv{cv}_2leg"
        ))

    # BLK + miss count
    for mm in [5, 6, 7, 8]:
        strategies.append((
            lambda p, m=mm: p['stat'] == 'blk' and p['l10_miss'] >= m,
            lambda p: p['l10_miss'] * 3 + conf_score_v2(p),
            2, f"blk_miss{mm}_2leg"
        ))

    # BLK + median below line
    strategies.append((
        lambda p: p['stat'] == 'blk' and p['l10_median'] < p['line'],
        lambda p: (p['line'] - p['l10_median']) + conf_score_v2(p),
        2, "blk_median_below_2leg"
    ))

    # === BLOCK 2: BLK + STL COMBOS ===
    for thresh in [3, 4, 5, 6, 7]:
        strategies.append((
            lambda p, t=thresh: p['stat'] in ['blk', 'stl'] and conf_score_v2(p) >= t,
            lambda p: conf_score_v2(p),
            2, f"blkstl_conf{thresh}_2leg"
        ))

    # BLK/STL + floor
    for thresh in [3, 4, 5]:
        strategies.append((
            lambda p, t=thresh: p['stat'] in ['blk', 'stl'] and conf_score_v2(p) >= t and p['l10_floor'] < p['line'],
            lambda p: conf_score_v2(p) + (p['line'] - p['l10_floor']),
            2, f"blkstl_floor_conf{thresh}_2leg"
        ))

    # === BLOCK 3: FLOOR-BASED (our #4 finding) ===
    # Floor across all stats
    for thresh in [3, 4, 5, 6]:
        for legs in [2, 3]:
            strategies.append((
                lambda p, t=thresh: conf_score_v2(p) >= t and p['l10_floor'] < p['line'],
                lambda p: conf_score_v2(p) + (p['line'] - p['l10_floor']) / max(p['line'],1) * 3,
                legs, f"floor_conf{thresh}_{legs}leg"
            ))

    # Floor + median both below line (super safe)
    for legs in [2, 3]:
        strategies.append((
            lambda p: p['l10_floor'] < p['line'] and p['l10_median'] < p['line'] and conf_score_v2(p) >= 4,
            lambda p: conf_score_v2(p) + (p['line'] - p['l10_median']),
            legs, f"floor_median_conf4_{legs}leg"
        ))

    # === BLOCK 4: SKIP-DAY STRATEGIES ===
    # Only bet when average candidate quality is high
    def high_quality_day(cands, min_avg_conf=6.0):
        scores = [conf_score_v2(p) for p in cands]
        return sum(scores)/len(scores) < min_avg_conf

    for min_avg in [5.0, 5.5, 6.0, 6.5, 7.0]:
        for legs in [2, 3]:
            strategies.append((
                lambda p: conf_score_v2(p) >= 5,
                lambda p: conf_score_v2(p),
                legs, f"skip_avgconf{min_avg}_{legs}leg",
                0,  # min_candidates
                lambda cands, mac=min_avg: sum(conf_score_v2(p) for p in cands)/len(cands) < mac
            ))

    # Only bet when N+ candidates available (selective days)
    for min_cand in [5, 8, 10, 15, 20]:
        for legs in [2]:
            strategies.append((
                lambda p: conf_score_v2(p) >= 5,
                lambda p: conf_score_v2(p),
                legs, f"minpool_{min_cand}_conf5_2leg",
                min_cand, None
            ))

    # === BLOCK 5: COMPOSITE SCORING EXPERIMENTS ===
    # Score A: miss_weight dominant
    for legs in [2, 3]:
        strategies.append((
            lambda p: p['l10_miss'] >= 6,
            lambda p: p['l10_miss'] * 5 + (100 - p['l10_hr'])/10 + (1 - p['l10_cv']) * 3,
            legs, f"score_miss_dominant_{legs}leg"
        ))

    # Score B: floor distance dominant
    for legs in [2, 3]:
        strategies.append((
            lambda p: p['l10_floor'] < p['line'] and p['l10_miss'] >= 5,
            lambda p: (p['line'] - p['l10_floor']) * 5 + p['l10_miss'] * 2 + abs(p['gap']),
            legs, f"score_floor_dominant_{legs}leg"
        ))

    # Score C: pure statistical (no conf_score, just raw numbers)
    for legs in [2, 3]:
        strategies.append((
            lambda p: p['l10_hr'] <= 30 and p['l10_floor'] < p['line'],
            lambda p: (p['line'] - p['l10_avg']) * 3 + (p['line'] - p['l10_floor']) * 2 + p['l10_miss'],
            legs, f"score_pure_stats_{legs}leg"
        ))

    # Score D: weighted multi-signal
    for legs in [2, 3]:
        def multi_signal_score(p):
            s = 0
            s += max(0, p['l10_miss'] - 4) * 3  # miss bonus
            s += max(0, p['line'] - p['l10_floor']) * 2  # floor gap
            s += max(0, -p['gap']) * 1.5  # line above avg
            s += max(0, 50 - p['l10_hr']) / 10  # low HR
            if p['streak'] == 'COLD': s += 2
            if p.get('is_away'): s += 1
            if p['stat'] in ['blk', 'stl']: s += 3
            if p['consec_under'] >= 3: s += p['consec_under']
            return s
        strategies.append((
            lambda p: True,
            multi_signal_score,
            legs, f"multi_signal_{legs}leg"
        ))

    # === BLOCK 6: MIXED-STAT PARLAYS ===
    # One BLK/STL leg + one other stat (diversified)
    # This requires special handling — different sort for each leg
    # We'll approximate by requiring diversity in stat types
    for legs in [2]:
        strategies.append((
            lambda p: conf_score_v2(p) >= 5,
            lambda p: conf_score_v2(p) + ({'blk': 3, 'stl': 2, '3pm': 1}.get(p['stat'], 0)),
            legs, f"stat_diverse_conf5_2leg"
        ))

    # === BLOCK 7: INFLATION SENSITIVITY ===
    # Test different inflation levels
    # (We'll run the main inflation=4 but also tag which gap ranges work best)
    for min_gap in [0.5, 1, 2, 3, 4, 5]:
        strategies.append((
            lambda p, mg=min_gap: p['gap'] <= -mg and p['stat'] in ['blk','stl'] and conf_score_v2(p) >= 3,
            lambda p: abs(p['gap']) + conf_score_v2(p),
            2, f"blkstl_gap{min_gap}_2leg"
        ))

    # === BLOCK 8: CONSECUTIVE UNDER MOMENTUM ===
    for min_c in range(2, 8):
        strategies.append((
            lambda p, mc=min_c: p['consec_under'] >= mc and conf_score_v2(p) >= 3,
            lambda p: p['consec_under'] * 5 + conf_score_v2(p),
            2, f"consec{min_c}_conf3_2leg"
        ))

    # === BLOCK 9: EXTREME FILTERS ===
    # Miss 10/10 (perfect miss streak)
    strategies.append((
        lambda p: p['l10_miss'] >= 10,
        lambda p: conf_score_v2(p),
        2, "perfect_miss_2leg"
    ))

    # L10 HR = 0 (never hit in last 10)
    strategies.append((
        lambda p: p['l10_hr'] == 0,
        lambda p: abs(p['gap']) + conf_score_v2(p),
        2, "zero_hr_2leg"
    ))

    # Max below floor (actual max in L10 still below line)
    strategies.append((
        lambda p: max([p['l10_avg']]) < p['line'] * 0.85,
        lambda p: p['line'] - p['l10_avg'],
        2, "avg_way_below_2leg"
    ))

    # === BLOCK 10: 3-LEG WITH BEST INSIGHTS ===
    # Apply BLK/STL insight to 3-leggers
    for thresh in [5, 6, 7]:
        strategies.append((
            lambda p, t=thresh: p['stat'] in ['blk','stl'] and conf_score_v2(p) >= t,
            lambda p: conf_score_v2(p),
            3, f"blkstl_conf{thresh}_3leg"
        ))

    # BLK only 3-leg
    for thresh in [5, 6, 7]:
        strategies.append((
            lambda p, t=thresh: p['stat'] == 'blk' and conf_score_v2(p) >= t,
            lambda p: conf_score_v2(p),
            3, f"blk_conf{thresh}_3leg"
        ))

    # Floor + BLK/STL 3-leg
    strategies.append((
        lambda p: p['stat'] in ['blk','stl'] and p['l10_floor'] < p['line'] and conf_score_v2(p) >= 5,
        lambda p: conf_score_v2(p) + (p['line'] - p['l10_floor']),
        3, "blkstl_floor_conf5_3leg"
    ))

    print(f"Generated {len(strategies)} V2 strategies", file=sys.stderr)
    return strategies


def main():
    player_games = load_data()
    print("Computing props...", file=sys.stderr)
    props = compute_all_props(player_games, inflation=4)
    print(f"Generated {len(props):,} props", file=sys.stderr)

    strategies = gen_strategies_v2()
    results = []

    for i, strat in enumerate(strategies):
        if len(strat) == 4:
            filt, sort, legs, name = strat
            min_cand, skip_fn = 0, None
        elif len(strat) == 5:
            filt, sort, legs, name, min_cand = strat
            skip_fn = None
        else:
            filt, sort, legs, name, min_cand, skip_fn = strat

        try:
            r = run_strategy(props, filt, sort, legs, name, min_cand, skip_fn)
            if r['total'] >= 30:
                results.append(r)
        except Exception as e:
            print(f"  Error in {name}: {e}", file=sys.stderr)

        if (i+1) % 30 == 0:
            print(f"  {i+1}/{len(strategies)} tested...", file=sys.stderr)

    results.sort(key=lambda r: r['rate'], reverse=True)

    print(f"\n{'='*100}")
    print(f"MEGA BACKTEST V2: {len(results)} strategies with 30+ parlays")
    print(f"{'='*100}")

    print(f"\n{'='*100}")
    print(f"TOP 60 BY PARLAY CASH RATE")
    print(f"{'='*100}")
    print(f"{'#':>3} {'Strategy':<45} {'W/L':>12} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4} {'Skip':>5}")
    for i, r in enumerate(results[:60]):
        print(f"{i+1:>3} {r['name']:<45} {r['wins']:>4}/{r['total']:<6} {r['rate']:>6.1%} {r['leg_rate']:>6.1%} {r['max_streak']:>6} {r['streaks_5plus']:>4} {r['streaks_10plus']:>4} {r['skipped']:>5}")

    # Top by max streak
    by_streak = sorted(results, key=lambda r: (r['max_streak'], r['rate']), reverse=True)
    print(f"\n{'='*100}")
    print(f"TOP 30 BY MAX WIN STREAK")
    print(f"{'='*100}")
    print(f"{'#':>3} {'Strategy':<45} {'W/L':>12} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4} {'Skip':>5}")
    for i, r in enumerate(by_streak[:30]):
        print(f"{i+1:>3} {r['name']:<45} {r['wins']:>4}/{r['total']:<6} {r['rate']:>6.1%} {r['leg_rate']:>6.1%} {r['max_streak']:>6} {r['streaks_5plus']:>4} {r['streaks_10plus']:>4} {r['skipped']:>5}")

    # === DEEP ANALYSIS: Why does BLK work? ===
    print(f"\n{'='*100}")
    print(f"STAT ANALYSIS: UNDER rate by stat type (all props)")
    print(f"{'='*100}")
    stat_totals = defaultdict(lambda: [0, 0])
    for p in props:
        stat_totals[p['stat']][1] += 1
        if p['under']: stat_totals[p['stat']][0] += 1
    for stat in sorted(stat_totals.keys()):
        u, t = stat_totals[stat]
        print(f"  {stat:5s}: {u:>7,}/{t:>7,} = {u/t:.1%}")

    # Yearly breakdown of best strategy
    if results:
        best = results[0]
        print(f"\n{'='*100}")
        print(f"YEARLY: {best['name']} ({best['rate']:.1%})")
        print(f"{'='*100}")
        yearly = defaultdict(lambda: [0, 0])
        for d in best['daily']:
            yearly[d['date'][:4]][1] += 1
            if d['hit']: yearly[d['date'][:4]][0] += 1
        for y in sorted(yearly.keys()):
            w, t = yearly[y]
            print(f"  {y}: {w:>3}/{t:<3} = {w/max(t,1):.1%}")

    # === STREAK ANALYSIS: Show longest streaks ===
    if by_streak:
        top = by_streak[0]
        print(f"\n{'='*100}")
        print(f"STREAK MAP: {top['name']} (max streak: {top['max_streak']})")
        print(f"{'='*100}")
        cur_streak = 0
        for d in top['daily']:
            if d['hit']:
                cur_streak += 1
            else:
                if cur_streak >= 5:
                    print(f"  Streak of {cur_streak} ending around {d['date']}")
                cur_streak = 0

    # Save
    output = [{
        'name': r['name'], 'wins': r['wins'], 'total': r['total'],
        'rate': round(r['rate'],4), 'leg_rate': round(r['leg_rate'],4),
        'max_streak': r['max_streak'], 'streaks_5plus': r['streaks_5plus'],
        'streaks_10plus': r['streaks_10plus'], 'skipped': r.get('skipped', 0),
    } for r in results[:100]]

    out_path = os.path.join(os.path.dirname(__file__), 'mega_backtest_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()

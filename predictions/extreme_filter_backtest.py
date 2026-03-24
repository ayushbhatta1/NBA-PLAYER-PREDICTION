#!/usr/bin/env python3
"""
EXTREME FILTER BACKTEST: Push per-leg hit rate to 78%+ through brutal selectivity.
"""

import csv, json, sys, os, gc
from collections import defaultdict
from datetime import datetime

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "NBA Database (1947 - Present)", "PlayerStatistics.csv")
MIN_LINES = {'pts': 5, 'reb': 2, 'ast': 1, '3pm': 0.5, 'blk': 0.5, 'stl': 0.5}

def pm(s):
    if not s: return 0
    try:
        if ':' in str(s): p = str(s).split(':'); return float(p[0]) + float(p[1])/60
        return float(s)
    except: return 0

def load_and_compute(inflation=4):
    print(f"Loading (inflation={inflation}%)...", file=sys.stderr)
    player_games = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            d = row.get('gameDateTimeEst', '')
            if not d or d < '2016-01-01': continue
            if row.get('gameType', '') != 'Regular Season': continue
            mins = pm(row.get('numMinutes', 0))
            if mins < 10: continue
            pid = row.get('personId', '')
            if not pid: continue
            hv = row.get('home', '')
            is_home = True if hv in ('True','true','1') else (False if hv in ('False','false','0') else None)
            player_games[pid].append({
                'date': d[:10], 'name': f"{row.get('firstName','')} {row.get('lastName','')}",
                'home': is_home, 'mins': mins,
                'pts': float(row.get('points',0) or 0), 'reb': float(row.get('reboundsTotal',0) or 0),
                'ast': float(row.get('assists',0) or 0), '3pm': float(row.get('threePointersMade',0) or 0),
                'blk': float(row.get('blocks',0) or 0), 'stl': float(row.get('steals',0) or 0),
                'pf': float(row.get('foulsPersonal',0) or 0), 'pm': float(row.get('plusMinusPoints',0) or 0),
            })
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g['date'])

    stats = ['pts', 'reb', 'ast', '3pm', 'blk', 'stl']
    date_props = defaultdict(list)
    for pid, games in player_games.items():
        if len(games) < 15: continue
        for i in range(15, len(games)):
            cur = games[i]
            prior = games[:i]
            for stat in stats:
                l10 = prior[-10:]
                v10 = [g[stat] for g in l10]
                avg10 = sum(v10)/10
                if avg10 < MIN_LINES.get(stat, 0.5): continue
                line = round((avg10 * (1 + inflation/100)) * 2) / 2
                actual = cur[stat]
                if actual == line: continue
                v3 = [g[stat] for g in prior[-3:]]
                sv = [g[stat] for g in prior if g['date'][:4] == cur['date'][:4]] or [g[stat] for g in prior[-30:]]
                hr10 = sum(1 for v in v10 if v > line)/10*100
                shr = sum(1 for v in sv if v > line)/len(sv)*100
                miss = sum(1 for v in v10 if v < line)
                std = (sum((v-avg10)**2 for v in v10)/10)**0.5
                flr = sorted(v10)[1] if len(v10) > 1 else min(v10)
                gap = avg10 - line
                if sum(v3)/3 > avg10*1.15: stk = 'HOT'
                elif sum(v3)/3 < avg10*0.85: stk = 'COLD'
                else: stk = 'NEUTRAL'
                cu = 0
                for v in reversed(v10):
                    if v < line: cu += 1
                    else: break
                try:
                    d1 = datetime.strptime(prior[-1]['date'], '%Y-%m-%d')
                    d2 = datetime.strptime(cur['date'], '%Y-%m-%d')
                    rest = (d2-d1).days - 1
                except: rest = 1
                mavg = sum(g['mins'] for g in l10)/10
                pf10 = sum(g['pf'] for g in l10)/10
                pm10 = sum(g['pm'] for g in l10)/10
                all_under = all(v < line for v in v10)

                date_props[cur['date']].append({
                    'date': cur['date'], 'name': cur['name'], 'stat': stat,
                    'line': line, 'actual': actual, 'under': actual < line,
                    'away': not cur['home'] if cur['home'] is not None else None,
                    'avg10': avg10, 'gap': gap, 'hr10': hr10, 'shr': shr,
                    'miss': miss, 'std': std, 'flr': flr,
                    'stk': stk, 'cu': cu, 'b2b': rest == 0,
                    'mavg': mavg, 'pf10': pf10, 'pm10': pm10,
                    'all_under': all_under,
                })
    del player_games
    gc.collect()
    total = sum(len(v) for v in date_props.values())
    print(f"  {total:,} props across {len(date_props)} dates", file=sys.stderr)
    return date_props


def uber(p):
    s = 0.0
    hr = p['hr10']
    if hr <= 10: s += 4.0
    elif hr <= 20: s += 3.0
    elif hr <= 30: s += 2.0
    elif hr <= 40: s += 1.0
    elif hr >= 70: s -= 2.0
    elif hr >= 60: s -= 1.0
    shr = p['shr']
    if shr < 30: s += 2.0
    elif shr < 45: s += 0.5
    elif shr >= 60: s -= 1.0
    sw = {'blk': 3.0, 'stl': 2.0, '3pm': 1.0, 'ast': 0.5}
    s += sw.get(p['stat'], 0)
    if p['flr'] < p['line']:
        s += min((p['line']-p['flr'])/max(p['line'],0.5)*5, 3.0)
    g = p['gap']
    if g < -5: s += 2.5
    elif g < -3: s += 2.0
    elif g < -1.5: s += 1.0
    elif g < 0: s += 0.5
    mc = p['miss']
    if mc >= 9: s += 2.5
    elif mc >= 7: s += 1.5
    elif mc >= 5: s += 0.5
    elif mc <= 2: s -= 1.0
    if p['stk'] == 'COLD': s += 1.5
    elif p['stk'] == 'HOT': s -= 1.0
    if p.get('away'): s += 0.7
    if p.get('b2b'): s += 0.5
    if p['cu'] >= 5: s += 2.0
    elif p['cu'] >= 3: s += 1.0
    if p['std'] < p['avg10'] * 0.25: s += 1.0
    elif p['std'] < p['avg10'] * 0.35: s += 0.5
    if p['pm10'] < -5: s += 0.5
    if p['all_under']: s += 2.0
    return s


def run_strategy(dates, date_props, n_legs, stat_filter, min_uber_val,
                 min_avg_uber=None, skip_fn=None):
    w, t, lh, lt, skipped = 0, 0, 0, 0, 0
    streaks = []
    cs = 0
    for d in dates:
        pool = [p for p in date_props[d] if uber(p) >= min_uber_val]
        if stat_filter:
            pool = [p for p in pool if p['stat'] in stat_filter]
        if len(pool) < n_legs:
            skipped += 1; continue
        ranked = sorted(pool, key=lambda p: -uber(p))
        seen = set()
        top = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                top.append(p)
            if len(top) >= n_legs + 3: break
        if len(top) < n_legs:
            skipped += 1; continue
        if min_avg_uber:
            avg = sum(uber(p) for p in top[:n_legs]) / n_legs
            if avg < min_avg_uber:
                skipped += 1; continue
        if skip_fn:
            if skip_fn.startswith('min_nth_'):
                mn = int(skip_fn.split('_')[-1])
                if uber(top[n_legs-1]) < mn:
                    skipped += 1; continue
            elif skip_fn.startswith('min_cold_'):
                mc = int(skip_fn.split('_')[-1])
                if sum(1 for p in top[:n_legs] if p['stk'] == 'COLD') < mc:
                    skipped += 1; continue
            elif skip_fn.startswith('min_perfect_'):
                mp = int(skip_fn.split('_')[-1])
                if sum(1 for p in top[:n_legs] if p.get('all_under')) < mp:
                    skipped += 1; continue
        picks = top[:n_legs]
        t += 1
        hits = sum(1 for p in picks if p['under'])
        lh += hits; lt += n_legs
        if hits == n_legs:
            w += 1; cs += 1
        else:
            if cs > 0: streaks.append(cs)
            cs = 0
    if cs > 0: streaks.append(cs)
    ms = max(streaks) if streaks else 0
    s5 = sum(1 for s in streaks if s >= 5)
    s10 = sum(1 for s in streaks if s >= 10)
    lr = lh/lt if lt else 0
    return w, t, lr, ms, s5, s10, skipped


def main():
    date_props = load_and_compute(inflation=4)
    dates = sorted(date_props.keys())

    print(f"\n{'='*130}")
    print("EXTREME FILTER BACKTEST")
    print(f"{'='*130}")

    # PHASE 1: Per-leg hit rate by uber threshold
    print(f"\nPHASE 1: Per-leg hit rate by uber score threshold")
    print(f"{'─'*80}")
    for sf, sf_name in [(['blk','stl'], 'BLK/STL'), (None, 'ALL')]:
        print(f"\n  {sf_name}:")
        for thresh in [3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20]:
            h, t = 0, 0
            for d in dates:
                for p in date_props[d]:
                    if sf and p['stat'] not in sf: continue
                    if uber(p) >= thresh:
                        t += 1
                        if p['under']: h += 1
            if t > 0:
                print(f"    Uber>={thresh:>2d}: {h:>7d}/{t:>7d} = {h/t*100:>5.1f}%  ({t/len(dates):.1f}/day)")

    # PHASE 2: Per-leg HR by pick rank position
    print(f"\nPHASE 2: Hit rate by pick position (BLK/STL, uber>=8)")
    print(f"{'─'*80}")
    rank_hits = defaultdict(lambda: [0, 0])
    for d in dates:
        pool = [p for p in date_props[d] if p['stat'] in ('blk','stl') and uber(p) >= 8]
        ranked = sorted(pool, key=lambda p: -uber(p))
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) >= 10: break
        for r, p in enumerate(picks):
            rank_hits[r][0] += 1
            if p['under']: rank_hits[r][1] += 1
    for r in sorted(rank_hits.keys()):
        t, h = rank_hits[r]
        if t > 0: print(f"  Pick #{r+1}: {h}/{t} = {h/t*100:.1f}%")

    # PHASE 3: 5-leg and 6-leg strategies
    print(f"\nPHASE 3: 5-leg parlay strategies")
    print(f"{'─'*130}")
    hdr = f"  {'Strategy':<50s} {'W':>5s} {'T':>6s} {'Rate':>7s} {'LegHR':>7s} {'MaxS':>5s} {'5+S':>5s} {'10+S':>5s} {'Skip%':>6s}"
    print(hdr)

    configs = []
    # BLK/STL with various min uber
    for mu in [6, 8, 10, 12, 14, 16]:
        configs.append((f'blkstl_u{mu}_5leg', 5, ['blk','stl'], mu, None, None))
        configs.append((f'all_u{mu}_5leg', 5, None, mu, None, None))
    # Min average uber
    for ma in [8, 10, 12, 14, 16]:
        configs.append((f'blkstl_avg{ma}_5leg', 5, ['blk','stl'], 6, ma, None))
        configs.append((f'all_avg{ma}_5leg', 5, None, 6, ma, None))
    # Min Nth pick
    for mn in [6, 8, 10, 12]:
        configs.append((f'blkstl_min5th_{mn}_5leg', 5, ['blk','stl'], 5, None, f'min_nth_{mn}'))
        configs.append((f'all_min5th_{mn}_5leg', 5, None, 5, None, f'min_nth_{mn}'))
    # Cold + BLK/STL
    for nc in [2, 3, 4, 5]:
        configs.append((f'blkstl_cold{nc}_5leg', 5, ['blk','stl'], 6, None, f'min_cold_{nc}'))
    # Perfect under streak
    for mp in [2, 3, 4, 5]:
        configs.append((f'perfect{mp}_5leg', 5, None, 5, None, f'min_perfect_{mp}'))
        configs.append((f'perfect{mp}_blkstl_5leg', 5, ['blk','stl'], 5, None, f'min_perfect_{mp}'))
    # 6-leg versions
    for mu in [8, 10, 12, 14]:
        configs.append((f'blkstl_u{mu}_6leg', 6, ['blk','stl'], mu, None, None))
    for ma in [8, 10, 12, 14]:
        configs.append((f'blkstl_avg{ma}_6leg', 6, ['blk','stl'], 6, ma, None))

    results = []
    for name, nl, sf, mu, mau, sfn in configs:
        w, t, lr, ms, s5, s10, sk = run_strategy(dates, date_props, nl, sf, mu, mau, sfn)
        if t >= 20:
            r = w/t
            sp = sk/(sk+t)*100
            results.append((name, w, t, r, lr, ms, s5, s10, sp))
            print(f"  {name:<50s} {w:>5d} {t:>6d} {r:>6.1%} {lr:>6.1%} {ms:>5d} {s5:>5d} {s10:>5d} {sp:>5.0f}%")

    # PHASE 4: Test different line inflations (memory-efficient)
    print(f"\nPHASE 4: Line inflation sensitivity (BLK/STL, uber>=8, 5-leg)")
    print(f"{'─'*80}")
    del date_props
    gc.collect()
    for infl in [0, 2, 3, 4, 5, 6, 8]:
        dp = load_and_compute(inflation=infl)
        ds = sorted(dp.keys())
        w, t, lr, ms, s5, s10, sk = run_strategy(ds, dp, 5, ['blk','stl'], 8, None, None)
        if t > 0:
            r = w/t
            print(f"  Inflation {infl}%: W={w:>4d} T={t:>5d} Rate={r:>6.1%} LegHR={lr:>6.1%} MaxStrk={ms} 5+Strk={s5}")
        del dp
        gc.collect()

    print(f"\nDone.", file=sys.stderr)


if __name__ == '__main__':
    main()

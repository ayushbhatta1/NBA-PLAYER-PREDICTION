#!/usr/bin/env python3
"""
INFLATION DEEP DIVE — precompute rolling stats ONCE, then just vary inflation in line calc.
"""
import csv, json, sys, os
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

def load_raw_props():
    """Load once, compute all rolling stats. Store raw data (no line yet)."""
    print("Loading and precomputing rolling stats...", file=sys.stderr)
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
    raw_props = []  # List of dicts with precomputed stats (no line/under yet)

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
                actual = cur[stat]
                v3 = [g[stat] for g in prior[-3:]]
                sv = [g[stat] for g in prior if g['date'][:4] == cur['date'][:4]] or [g[stat] for g in prior[-30:]]
                std = (sum((v-avg10)**2 for v in v10)/10)**0.5
                flr = sorted(v10)[1] if len(v10) > 1 else min(v10)
                if sum(v3)/3 > avg10*1.15: stk = 'HOT'
                elif sum(v3)/3 < avg10*0.85: stk = 'COLD'
                else: stk = 'NEUTRAL'
                cu = 0
                for v in reversed(v10):
                    if v < avg10: cu += 1  # Note: using avg not line for cu
                    else: break
                try:
                    d1 = datetime.strptime(prior[-1]['date'], '%Y-%m-%d')
                    d2 = datetime.strptime(cur['date'], '%Y-%m-%d')
                    rest = (d2-d1).days - 1
                except: rest = 1
                mavg = sum(g['mins'] for g in l10)/10
                pf10 = sum(g['pf'] for g in l10)/10
                pm10 = sum(g['pm'] for g in l10)/10

                raw_props.append({
                    'date': cur['date'], 'name': cur['name'], 'stat': stat,
                    'actual': actual, 'avg10': avg10, 'v10': v10,
                    'away': not cur['home'] if cur['home'] is not None else None,
                    'std': std, 'flr': flr, 'stk': stk,
                    'b2b': rest == 0, 'mavg': mavg, 'pf10': pf10, 'pm10': pm10,
                    'sv': sv,
                })

    del player_games
    print(f"  Precomputed {len(raw_props):,} raw props", file=sys.stderr)
    return raw_props


def apply_inflation(raw_props, stat_infl):
    """Apply inflation to get lines, compute line-dependent features, group by date."""
    date_props = defaultdict(list)
    for rp in raw_props:
        stat = rp['stat']
        avg10 = rp['avg10']
        infl = stat_infl.get(stat, 4)
        line = round((avg10 * (1 + infl/100)) * 2) / 2
        actual = rp['actual']
        if actual == line: continue
        v10 = rp['v10']
        sv = rp['sv']
        hr10 = sum(1 for v in v10 if v > line)/10*100
        shr = sum(1 for v in sv if v > line)/len(sv)*100
        miss = sum(1 for v in v10 if v < line)
        gap = avg10 - line
        flr = rp['flr']
        # Recompute cu with line
        cu = 0
        for v in reversed(v10):
            if v < line: cu += 1
            else: break
        all_under = all(v < line for v in v10)

        date_props[rp['date']].append({
            'name': rp['name'], 'stat': stat,
            'line': line, 'actual': actual, 'under': actual < line,
            'away': rp['away'], 'avg10': avg10, 'gap': gap,
            'hr10': hr10, 'shr': shr, 'miss': miss,
            'std': rp['std'], 'flr': flr, 'stk': rp['stk'],
            'cu': cu, 'b2b': rp['b2b'], 'mavg': rp['mavg'],
            'pf10': rp['pf10'], 'pm10': rp['pm10'],
            'all_under': all_under,
        })
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
    if p.get('all_under'): s += 2.0
    return s


def run_parlay(dates, dp, n, sf, mu, ma=None):
    w, t, lh, lt, sk = 0, 0, 0, 0, 0
    streaks = []; cs = 0
    for d in dates:
        pool = [p for p in dp[d] if uber(p) >= mu and (not sf or p['stat'] in sf)]
        if len(pool) < n: sk += 1; continue
        ranked = sorted(pool, key=lambda p: -uber(p))
        seen = set(); picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name']); picks.append(p)
            if len(picks) >= n: break
        if len(picks) < n: sk += 1; continue
        if ma and sum(uber(p) for p in picks[:n])/n < ma: sk += 1; continue
        t += 1
        hits = sum(1 for p in picks if p['under'])
        lh += hits; lt += n
        if hits == n: w += 1; cs += 1
        else:
            if cs: streaks.append(cs)
            cs = 0
    if cs: streaks.append(cs)
    ms = max(streaks) if streaks else 0
    s5 = sum(1 for s in streaks if s >= 5)
    s10 = sum(1 for s in streaks if s >= 10)
    lr = lh/lt if lt else 0
    return w, t, lr, ms, s5, s10, sk


def main():
    raw = load_raw_props()

    # ================================================================
    # PHASE 1: Uniform inflation sweep
    # ================================================================
    print(f"\n{'='*100}")
    print("PHASE 1: Uniform inflation 0-20% for BLK/STL 5-leg (uber>=8)")
    print(f"{'─'*100}")
    for infl in [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
        im = {s: infl for s in ['pts','reb','ast','3pm','blk','stl']}
        dp = apply_inflation(raw, im)
        ds = sorted(dp.keys())
        w, t, lr, ms, s5, s10, _ = run_parlay(ds, dp, 5, ['blk','stl'], 8)
        if t > 0:
            print(f"  {infl:>2d}%: {w:>4d}/{t:>5d} = {w/t*100:>5.1f}% | LegHR={lr*100:>5.1f}% | MaxStrk={ms:>2d} | 5+S={s5} 10+S={s10}")
        del dp

    # ================================================================
    # PHASE 2: BLK/STL-specific inflation
    # ================================================================
    print(f"\nPHASE 2: BLK/STL inflation (others=4%)")
    print(f"{'─'*100}")
    for bi in [4, 6, 8, 10, 12, 15, 20, 25, 30]:
        im = {'pts':4,'reb':4,'ast':4,'3pm':4,'blk':bi,'stl':bi}
        dp = apply_inflation(raw, im)
        ds = sorted(dp.keys())
        w, t, lr, ms, s5, s10, _ = run_parlay(ds, dp, 5, ['blk','stl'], 8)
        if t > 0:
            print(f"  BLK/STL={bi:>2d}%: {w:>4d}/{t:>5d} = {w/t*100:>5.1f}% | LegHR={lr*100:>5.1f}% | MaxStrk={ms:>2d} | 5+S={s5} 10+S={s10}")
        del dp

    # ================================================================
    # PHASE 3: Per-stat under rate
    # ================================================================
    print(f"\nPHASE 3: Natural under rate by stat at different inflation")
    print(f"{'─'*100}")
    for infl in [0, 4, 8, 12, 20]:
        im = {s: infl for s in ['pts','reb','ast','3pm','blk','stl']}
        dp = apply_inflation(raw, im)
        sh = defaultdict(lambda: [0,0])
        for d in dp:
            for p in dp[d]:
                sh[p['stat']][0] += 1
                if p['under']: sh[p['stat']][1] += 1
        line = f"  Infl={infl:>2d}%: "
        for s in ['blk','stl','3pm','ast','reb','pts']:
            t, h = sh[s]
            line += f"{s}={h/t*100:.0f}% " if t else ""
        print(line)
        del dp

    # ================================================================
    # PHASE 4: Key models for 5/6 leg
    # ================================================================
    print(f"\nPHASE 4: Model comparison (5-leg and 6-leg)")
    print(f"{'─'*100}")
    models = [
        ('uniform_4', {s:4 for s in ['pts','reb','ast','3pm','blk','stl']}),
        ('uniform_8', {s:8 for s in ['pts','reb','ast','3pm','blk','stl']}),
        ('uniform_10', {s:10 for s in ['pts','reb','ast','3pm','blk','stl']}),
        ('uniform_12', {s:12 for s in ['pts','reb','ast','3pm','blk','stl']}),
        ('blk15_stl12_rest4', {'pts':4,'reb':4,'ast':4,'3pm':4,'blk':15,'stl':12}),
        ('blk20_stl15_rest4', {'pts':4,'reb':4,'ast':4,'3pm':4,'blk':20,'stl':15}),
        ('aggressive', {'pts':6,'reb':6,'ast':6,'3pm':8,'blk':12,'stl':10}),
        ('realistic', {'pts':3,'reb':4,'ast':4,'3pm':6,'blk':10,'stl':8}),
    ]
    for name, im in models:
        dp = apply_inflation(raw, im)
        ds = sorted(dp.keys())
        print(f"\n  {name}:")
        for n in [5, 6]:
            for sf, sfn in [(['blk','stl'], 'BLK/STL'), (None, 'ALL')]:
                for mu in [6, 8, 10]:
                    w, t, lr, ms, s5, s10, sk = run_parlay(ds, dp, n, sf, mu)
                    if t >= 50:
                        print(f"    {sfn:>7s} u>={mu} {n}L: {w:>4d}/{t:>5d} = {w/t*100:>5.1f}% LR={lr*100:>5.1f}% MS={ms:>2d} 5+={s5} 10+={s10}")
        del dp

    # ================================================================
    # PHASE 5: Best + skip-day combos
    # ================================================================
    print(f"\nPHASE 5: Skip-day filtering at best inflation levels")
    print(f"{'─'*100}")
    for infl in [8, 10, 12]:
        im = {s: infl for s in ['pts','reb','ast','3pm','blk','stl']}
        dp = apply_inflation(raw, im)
        ds = sorted(dp.keys())
        for n in [5, 6]:
            for ma in [None, 10, 12, 14]:
                w, t, lr, ms, s5, s10, sk = run_parlay(ds, dp, n, ['blk','stl'], 8, ma=ma)
                if t >= 30:
                    sp = sk/(sk+t)*100
                    man = f'_avg{ma}' if ma else ''
                    print(f"  i{infl}_blkstl{man}_{n}L: {w:>4d}/{t:>5d} = {w/t*100:>5.1f}% LR={lr*100:>5.1f}% MS={ms:>2d} 5+={s5} 10+={s10} Skip={sp:.0f}%")
        del dp

    # Save summary
    out = os.path.join(os.path.dirname(__file__), 'inflation_results.json')
    # Quick save of best configs
    best_dp = apply_inflation(raw, {s:10 for s in ['pts','reb','ast','3pm','blk','stl']})
    ds = sorted(best_dp.keys())
    configs = [
        ('i10_blkstl_5L', 5, ['blk','stl'], 8, None),
        ('i10_blkstl_6L', 6, ['blk','stl'], 8, None),
        ('i10_all_5L', 5, None, 8, None),
        ('i10_blkstl_avg12_5L', 5, ['blk','stl'], 8, 12),
    ]
    results = []
    for name, n, sf, mu, ma in configs:
        w, t, lr, ms, s5, s10, sk = run_parlay(ds, best_dp, n, sf, mu, ma=ma)
        results.append({'name': name, 'w': w, 't': t, 'r': w/t if t else 0,
                       'lr': lr, 'ms': ms, 's5': s5, 's10': s10, 'sk': sk})
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}", file=sys.stderr)
    print(f"\nDone.", file=sys.stderr)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
MEGA BACKTEST FINAL: Test the absolute best combos we've found.
Specifically testing:
1. 2-leg: BLK-away + conf + uber (the 59.6% winner)
2. 3-leg: Can we make 3-leggers work with BLK/STL focus?
3. Skip-day: Can we skip bad days to push rate even higher?
4. "Nuclear" combos: Multiple strong signals required
5. Adaptive bet sizing: Bet more when confidence is highest
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

def load():
    pg = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            d = r.get('gameDateTimeEst','')
            if not d or d < '2016-01-01' or r.get('gameType','') != 'Regular Season': continue
            m = pm(r.get('numMinutes',0))
            if m < 10: continue
            pid = r.get('personId','')
            if not pid: continue
            hv = r.get('home','')
            ih = True if hv in ('True','true','1') else (False if hv in ('False','false','0') else None)
            pg[pid].append({
                'date': d[:10], 'name': f"{r.get('firstName','')} {r.get('lastName','')}",
                'home': ih, 'mins': m,
                'blk': float(r.get('blocks',0) or 0), 'stl': float(r.get('steals',0) or 0),
                'pts': float(r.get('points',0) or 0), 'reb': float(r.get('reboundsTotal',0) or 0),
                'ast': float(r.get('assists',0) or 0), '3pm': float(r.get('threePointersMade',0) or 0),
                'pf': float(r.get('foulsPersonal',0) or 0), 'pm': float(r.get('plusMinusPoints',0) or 0),
            })
    for pid in pg: pg[pid].sort(key=lambda g: g['date'])
    return pg

def compute(pg, inflation=4):
    props = []
    stats = ['pts','reb','ast','3pm','blk','stl']
    for pid, games in pg.items():
        if len(games) < 15: continue
        for i in range(15, len(games)):
            cur = games[i]
            prior = games[:i]
            for stat in stats:
                l10 = prior[-10:]
                v10 = [g[stat] for g in l10]
                avg10 = sum(v10)/10
                if avg10 < MIN_LINES.get(stat, 0.5): continue
                line = round((avg10*(1+inflation/100))*2)/2
                actual = cur[stat]
                if actual == line: continue
                v5 = [g[stat] for g in prior[-5:]]
                v3 = [g[stat] for g in prior[-3:]]
                sv = [g[stat] for g in prior if g['date'][:4]==cur['date'][:4]] or [g[stat] for g in prior[-30:]]
                hr10 = sum(1 for v in v10 if v > line)/10*100
                shr = sum(1 for v in sv if v > line)/len(sv)*100
                miss = sum(1 for v in v10 if v < line)
                std = (sum((v-avg10)**2 for v in v10)/10)**0.5
                cv = std/max(avg10,0.1)
                flr = sorted(v10)[1] if len(v10) > 1 else min(v10)
                med = sorted(v10)[5]
                gap = avg10 - line
                a5 = sum(v5)/5; a3 = sum(v3)/3
                if a3 > avg10*1.15: stk = 'HOT'
                elif a3 < avg10*0.85: stk = 'COLD'
                else: stk = 'NEUTRAL'
                cu = 0
                for v in reversed(v10):
                    if v < line: cu += 1
                    else: break
                try:
                    d1 = datetime.strptime(prior[-1]['date'],'%Y-%m-%d')
                    d2 = datetime.strptime(cur['date'],'%Y-%m-%d')
                    rest = (d2-d1).days - 1
                except: rest = 1
                mns = [g['mins'] for g in l10]
                props.append({
                    'date': cur['date'], 'name': cur['name'], 'stat': stat,
                    'line': line, 'actual': actual, 'under': actual < line,
                    'home': cur['home'], 'away': not cur['home'] if cur['home'] is not None else None,
                    'avg10': avg10, 'avg5': a5, 'avg3': a3,
                    'gap': gap, 'hr10': hr10, 'shr': shr,
                    'miss': miss, 'std': std, 'cv': cv,
                    'flr': flr, 'med': med,
                    'stk': stk, 'cu': cu,
                    'rest': rest, 'b2b': rest==0,
                    'mavg': sum(mns)/10,
                })
    return props

def conf(p):
    s = 0.0
    hr = p['hr10']
    if hr < 20: s += 3.0
    elif hr < 35: s += 2.0
    elif hr < 45: s += 1.0
    elif hr >= 80: s -= 2.0
    elif hr >= 65: s -= 1.0
    elif hr >= 55: s -= 0.5
    shr = p['shr']
    if shr < 30: s += 2.0
    elif shr < 45: s += 0.5
    elif shr >= 70: s -= 1.0
    elif shr >= 55: s -= 0.5
    sw = {'blk': 2.0, 'stl': 1.5, '3pm': 1.0, 'ast': 0.5}
    s += sw.get(p['stat'], 0)
    if p['stk'] == 'COLD': s += 1.0
    elif p['stk'] == 'HOT': s -= 0.5
    g = p['gap']
    if g < -5: s += 2.0
    elif g < -3: s += 1.5
    elif g < -1.5: s += 1.0
    elif g < 0: s += 0.5
    elif g < 3: s -= 0.5
    else: s -= 0.5
    mc = p['miss']
    if mc >= 9: s += 2.0
    elif mc >= 7: s += 1.0
    elif mc >= 5: s += 0.3
    elif mc < 3: s -= 0.5
    if p.get('away'): s += 0.5
    if p.get('b2b'): s += 0.3
    return s

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
    if p['flr'] < p['line']: s += min((p['line']-p['flr'])/max(p['line'],0.5)*5, 3.0)
    if p['med'] < p['line']: s += 1.0
    g = p['gap']
    if g < -5: s += 2.5
    elif g < -3: s += 2.0
    elif g < -1.5: s += 1.0
    elif g < 0: s += 0.5
    elif g > 3: s -= 1.0
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
    if p['cv'] < 0.3: s += 0.5
    elif p['cv'] > 1.5: s -= 0.5
    if p['mavg'] >= 30: s += 0.3
    return s

def run(props, filt, sort_fn, legs, name, min_c=0, skip_fn=None, mps=None):
    bd = defaultdict(list)
    for p in props:
        if filt(p): bd[p['date']].append(p)
    w = t = lh = lt = cs = ms = 0
    ss = []; dy = []; sk = 0
    for date in sorted(bd.keys()):
        c = bd[date]
        if len(c) < max(min_c, legs): sk += 1; continue
        if skip_fn and skip_fn(c): sk += 1; continue
        c.sort(key=sort_fn, reverse=True)
        sel = []; un = set(); sc = defaultdict(int)
        for p in c:
            if p['name'] in un: continue
            if mps and sc[p['stat']] >= mps: continue
            sel.append(p); un.add(p['name']); sc[p['stat']] += 1
            if len(sel) >= legs: break
        if len(sel) < legs: continue
        ah = all(p['under'] for p in sel)
        nh = sum(p['under'] for p in sel)
        t += 1; lt += legs; lh += nh
        if ah: w += 1; cs += 1; ms = max(ms, cs)
        else:
            if cs > 0: ss.append(cs)
            cs = 0
        dy.append({'date': date, 'hit': ah, 'nh': nh})
    if cs > 0: ss.append(cs)
    return {'name': name, 'w': w, 't': t, 'r': w/max(t,1),
            'lr': lh/max(lt,1), 'ms': ms,
            's5': sum(1 for s in ss if s >= 5),
            's10': sum(1 for s in ss if s >= 10),
            'sk': sk, 'dy': dy}

def main():
    print("Loading...", file=sys.stderr)
    pg = load()
    print("Computing...", file=sys.stderr)
    props = compute(pg)
    print(f"  {len(props):,} props", file=sys.stderr)

    strats = []

    # ======== THE WINNERS: Fine-tune the best combos ========

    # A. BLK + away + various conf thresholds (V3 winner at 59.6%)
    for ct in range(3, 10):
        strats.append((lambda p,t=ct: p['stat']=='blk' and conf(p)>=t and p.get('away'), uber, 2, f"blk_away_c{ct}_2"))

    # B. BLK + away + uber threshold
    for ut in [10, 12, 14, 15, 16, 17, 18]:
        strats.append((lambda p,t=ut: p['stat']=='blk' and p.get('away') and uber(p)>=t, uber, 2, f"blk_away_u{ut}_2"))

    # C. BLK + away + floor
    strats.append((lambda p: p['stat']=='blk' and p.get('away') and p['flr']<p['line'], uber, 2, "blk_away_flr_2"))
    strats.append((lambda p: p['stat']=='blk' and p.get('away') and p['med']<p['line'], uber, 2, "blk_away_med_2"))
    strats.append((lambda p: p['stat']=='blk' and p.get('away') and p['flr']<p['line'] and conf(p)>=5, uber, 2, "blk_away_flr_c5_2"))
    strats.append((lambda p: p['stat']=='blk' and p.get('away') and p['flr']<p['line'] and conf(p)>=6, uber, 2, "blk_away_flr_c6_2"))
    strats.append((lambda p: p['stat']=='blk' and p.get('away') and p['flr']<p['line'] and conf(p)>=7, uber, 2, "blk_away_flr_c7_2"))

    # D. BLK + away + miss count
    for mm in [6, 7, 8]:
        strats.append((lambda p,m=mm: p['stat']=='blk' and p.get('away') and p['miss']>=m, uber, 2, f"blk_away_m{mm}_2"))

    # E. BLK + away + cold
    strats.append((lambda p: p['stat']=='blk' and p.get('away') and p['stk']=='COLD', uber, 2, "blk_away_cold_2"))
    strats.append((lambda p: p['stat']=='blk' and p.get('away') and p['stk']!='HOT', uber, 2, "blk_away_nohot_2"))

    # F. BLK + away + consec under
    for mc in [2, 3, 4]:
        strats.append((lambda p,c=mc: p['stat']=='blk' and p.get('away') and p['cu']>=c, uber, 2, f"blk_away_cu{mc}_2"))

    # G. BLK + away + low HR
    for mhr in [10, 20, 30, 40]:
        strats.append((lambda p,h=mhr: p['stat']=='blk' and p.get('away') and p['hr10']<=h, uber, 2, f"blk_away_hr{mhr}_2"))

    # H. SKIP-DAY: BLK away, but only on days with many quality picks
    for mc in [3, 4, 5, 6]:
        def mk_skip(n):
            def sf(c):
                blk_away = [p for p in c if p['stat']=='blk' and p.get('away')]
                return len(blk_away) < n
            return sf
        strats.append((lambda p: p['stat']=='blk' and p.get('away'), uber, 2, f"blk_away_minpool{mc}_2", 0, mk_skip(mc)))

    # I. SKIP: only when top 2 blk-away have uber >= X
    for mt in [12, 14, 16, 18]:
        def mk_skip2(threshold):
            def sf(c):
                blk_away = [p for p in c if p['stat']=='blk' and p.get('away')]
                if len(blk_away) < 2: return True
                scores = sorted([uber(p) for p in blk_away], reverse=True)
                return scores[1] < threshold  # 2nd best must meet threshold
            return sf
        strats.append((lambda p: p['stat']=='blk' and p.get('away'), uber, 2, f"blk_away_skip2u{mt}_2", 0, mk_skip2(mt)))

    # ======== 3-LEG STRATEGIES (for platforms requiring 3+) ========

    # J. BLK-heavy 3-leg (2 BLK + 1 STL)
    for ct in [4, 5, 6, 7]:
        strats.append((lambda p,t=ct: p['stat'] in ['blk','stl'] and conf(p)>=t, uber, 3, f"blkstl_c{ct}_3"))

    # K. BLK 3-leg (all BLK)
    for ct in [4, 5, 6, 7]:
        strats.append((lambda p,t=ct: p['stat']=='blk' and conf(p)>=t, uber, 3, f"blk_c{ct}_3"))

    # L. BLK-away 3-leg
    for ct in [4, 5, 6]:
        strats.append((lambda p,t=ct: p['stat']=='blk' and conf(p)>=t and p.get('away'), uber, 3, f"blk_away_c{ct}_3"))

    # M. BLK/STL + away 3-leg
    for ct in [4, 5, 6]:
        strats.append((lambda p,t=ct: p['stat'] in ['blk','stl'] and conf(p)>=t and p.get('away'), uber, 3, f"blkstl_away_c{ct}_3"))

    # N. Mixed 3-leg: 1 BLK + max 1 per stat
    for ct in [5, 6, 7]:
        def blk_prio(p):
            return uber(p) + (100 if p['stat']=='blk' else 0)
        strats.append((lambda p,t=ct: conf(p)>=t, blk_prio, 3, f"blk_prio_c{ct}_3", 0, None, 1))

    # O. BLK/STL + floor 3-leg
    strats.append((lambda p: p['stat'] in ['blk','stl'] and p['flr']<p['line'] and conf(p)>=5, uber, 3, "blkstl_flr_c5_3"))

    # ======== NUCLEAR COMBOS ========

    # P. Every strong signal aligned
    strats.append((
        lambda p: (p['stat']=='blk' and p.get('away') and p['miss']>=7 and
                   p['flr']<p['line'] and p['stk']!='HOT' and conf(p)>=5),
        uber, 2, "nuclear_blk_away_m7_flr_nohot_c5_2"
    ))
    strats.append((
        lambda p: (p['stat'] in ['blk','stl'] and p.get('away') and p['miss']>=7 and
                   p['flr']<p['line'] and p['stk']!='HOT'),
        uber, 2, "nuclear_blkstl_away_m7_flr_nohot_2"
    ))

    # Q. B2B + away + BLK (triple fatigue)
    strats.append((
        lambda p: p['stat']=='blk' and p.get('away') and p.get('b2b') and conf(p)>=3,
        uber, 2, "blk_away_b2b_c3_2"
    ))

    print(f"Testing {len(strats)} strategies...", file=sys.stderr)

    results = []
    for i, s in enumerate(strats):
        f, sf, l, n = s[0], s[1], s[2], s[3]
        mc = s[4] if len(s) > 4 else 0
        skf = s[5] if len(s) > 5 else None
        mps = s[6] if len(s) > 6 else None
        try:
            r = run(props, f, sf, l, n, mc, skf, mps)
            if r['t'] >= 15:
                results.append(r)
        except Exception as e:
            print(f"  ERR {n}: {e}", file=sys.stderr)

    results.sort(key=lambda r: r['r'], reverse=True)

    print(f"\n{'='*110}")
    print(f"FINAL RESULTS: {len(results)} strategies")
    print(f"{'='*110}")
    print(f"{'#':>3} {'Strategy':<50} {'W/L':>12} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4} {'Skip':>6}")
    for i, r in enumerate(results[:80]):
        print(f"{i+1:>3} {r['name']:<50} {r['w']:>4}/{r['t']:<6} {r['r']:>6.1%} {r['lr']:>6.1%} {r['ms']:>6} {r['s5']:>4} {r['s10']:>4} {r['sk']:>6}")

    # 3-leg section
    r3 = [r for r in results if r['name'].endswith('_3')]
    if r3:
        r3.sort(key=lambda r: r['r'], reverse=True)
        print(f"\n{'='*110}")
        print(f"BEST 3-LEG STRATEGIES")
        print(f"{'='*110}")
        print(f"{'#':>3} {'Strategy':<50} {'W/L':>12} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4} {'Skip':>6}")
        for i, r in enumerate(r3[:30]):
            print(f"{i+1:>3} {r['name']:<50} {r['w']:>4}/{r['t']:<6} {r['r']:>6.1%} {r['lr']:>6.1%} {r['ms']:>6} {r['s5']:>4} {r['s10']:>4} {r['sk']:>6}")

    # Year breakdown of #1
    if results:
        b = results[0]
        print(f"\n{'='*110}")
        print(f"YEARLY: {b['name']} ({b['r']:.1%})")
        yearly = defaultdict(lambda: [0,0])
        for d in b['dy']:
            yearly[d['date'][:4]][1] += 1
            if d['hit']: yearly[d['date'][:4]][0] += 1
        for y in sorted(yearly.keys()):
            w, t = yearly[y]
            print(f"  {y}: {w:>3}/{t:<3} = {w/max(t,1):.1%}")

    # Streak map of best streak strategy
    by_s = sorted(results, key=lambda r: (r['ms'], r['r']), reverse=True)
    if by_s:
        top = by_s[0]
        print(f"\n{'='*110}")
        print(f"STREAK MAP: {top['name']} (max: {top['ms']})")
        cs = 0
        for d in top['dy']:
            if d['hit']: cs += 1
            else:
                if cs >= 5: print(f"  {cs}-day streak ending {d['date']}")
                cs = 0

    # Save
    out = [{'name':r['name'],'w':r['w'],'t':r['t'],'r':round(r['r'],4),
            'lr':round(r['lr'],4),'ms':r['ms'],'s5':r['s5'],'s10':r['s10']}
           for r in results[:100]]
    path = os.path.join(os.path.dirname(__file__), 'mega_backtest_final_results.json')
    with open(path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved to {path}", file=sys.stderr)


if __name__ == '__main__':
    main()

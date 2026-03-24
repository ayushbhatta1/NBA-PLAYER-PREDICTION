#!/usr/bin/env python3
"""
LINE FLOOR EXPLOITATION BACKTEST: The secret to profitable 5-6 leg parlays.

DISCOVERY: Sportsbooks can't set BLK/STL lines below 0.5. This creates:
- BLK line = 0.5 for players averaging 0.1-0.4 blocks → 67-400% inflation → UNDER hits 80%+
- STL line = 0.5 for players averaging 0.2-0.4 steals → 25-150% inflation → UNDER hits 70%+

Strategy: Stack UNDER 0.5 BLK/STL picks where the line is MASSIVELY inflated.
These are structurally the most favorable bets in sports betting.

Uses REALISTIC line generation:
- Lines rounded to 0.5 increments
- BLK/STL have 0.5 minimum (can't go lower on most books)
- This naturally creates the inflation we observed in real SGO data
"""

import csv, json, sys, os
from collections import defaultdict
from datetime import datetime

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "NBA Database (1947 - Present)", "PlayerStatistics.csv")

def pm(s):
    if not s: return 0
    try:
        if ':' in str(s): p = str(s).split(':'); return float(p[0]) + float(p[1])/60
        return float(s)
    except: return 0


def load_players():
    print("Loading...", file=sys.stderr)
    pg = defaultdict(list)
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
            pg[pid].append({
                'date': d[:10], 'name': f"{row.get('firstName','')} {row.get('lastName','')}",
                'home': is_home, 'mins': mins,
                'pts': float(row.get('points',0) or 0), 'reb': float(row.get('reboundsTotal',0) or 0),
                'ast': float(row.get('assists',0) or 0), '3pm': float(row.get('threePointersMade',0) or 0),
                'blk': float(row.get('blocks',0) or 0), 'stl': float(row.get('steals',0) or 0),
                'pf': float(row.get('foulsPersonal',0) or 0), 'pm': float(row.get('plusMinusPoints',0) or 0),
            })
    for pid in pg:
        pg[pid].sort(key=lambda g: g['date'])
    print(f"Loaded {len(pg)} players", file=sys.stderr)
    return pg


def realistic_line(avg10, stat, base_inflation=4):
    """Generate a realistic sportsbook line.

    Key: Sportsbooks round to 0.5 increments and have minimums.
    BLK/STL minimum is typically 0.5.
    """
    # Apply base inflation (sportsbook wants vig)
    projected = avg10 * (1 + base_inflation / 100)

    # Round to nearest 0.5
    line = round(projected * 2) / 2

    # Apply sportsbook minimums for low-frequency stats
    if stat in ('blk', 'stl') and line < 0.5:
        line = 0.5
    if stat == '3pm' and line < 0.5:
        line = 0.5

    return line


def compute_props(player_games, line_fn):
    """Compute props using a given line function."""
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

                # Skip if average is basically zero
                if stat in ('pts',) and avg10 < 5: continue
                if stat in ('reb',) and avg10 < 2: continue
                if stat in ('ast',) and avg10 < 1: continue
                if stat in ('3pm', 'blk', 'stl') and avg10 < 0.1: continue

                line = line_fn(avg10, stat)
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
                inflation = (line - avg10) / max(avg10, 0.01) * 100

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
                    'name': cur['name'], 'stat': stat,
                    'line': line, 'actual': actual, 'under': actual < line,
                    'away': not cur['home'] if cur['home'] is not None else None,
                    'avg10': avg10, 'gap': gap, 'inflation': inflation,
                    'hr10': hr10, 'shr': shr, 'miss': miss,
                    'std': std, 'flr': flr, 'stk': stk,
                    'cu': cu, 'b2b': rest == 0,
                    'mavg': mavg, 'pf10': pf10, 'pm10': pm10,
                    'all_under': all_under,
                })

    total = sum(len(v) for v in date_props.values())
    print(f"  {total:,} props", file=sys.stderr)
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
    # HIGH INFLATION BONUS — the key new signal
    infl = p.get('inflation', 0)
    if infl >= 100: s += 3.0
    elif infl >= 50: s += 2.0
    elif infl >= 25: s += 1.0
    return s


def run_parlay(dates, dp, n, sf, mu, extra_filter=None, min_avg=None):
    w, t, lh, lt, sk = 0, 0, 0, 0, 0
    streaks = []; cs = 0
    for d in dates:
        pool = [p for p in dp[d] if uber(p) >= mu]
        if sf:
            pool = [p for p in pool if p['stat'] in sf]
        if extra_filter:
            pool = [p for p in pool if extra_filter(p)]
        if len(pool) < n: sk += 1; continue
        ranked = sorted(pool, key=lambda p: -uber(p))
        seen = set(); picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name']); picks.append(p)
            if len(picks) >= n: break
        if len(picks) < n: sk += 1; continue
        if min_avg and sum(uber(p) for p in picks[:n])/n < min_avg: sk += 1; continue
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
    pg = load_players()

    # ================================================================
    # PHASE 1: Per-leg hit rate by inflation level (realistic lines)
    # ================================================================
    print(f"\n{'='*120}")
    print("PHASE 1: Under hit rate by inflation bracket (realistic sportsbook lines)")
    print(f"{'─'*120}")

    dp = compute_props(pg, lambda a, s: realistic_line(a, s, 4))
    dates = sorted(dp.keys())

    # Group props by inflation level and stat
    for stat in ['blk', 'stl', '3pm', 'ast', 'reb', 'pts']:
        inflations = []
        for d in dates:
            for p in dp[d]:
                if p['stat'] == stat:
                    inflations.append((p['inflation'], p['under']))
        if not inflations: continue

        print(f"\n  {stat.upper()} ({len(inflations):,} props):")
        for lo, hi in [(-100, 0), (0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 500)]:
            bucket = [(i, u) for i, u in inflations if lo <= i < hi]
            if len(bucket) >= 50:
                rate = sum(u for _, u in bucket) / len(bucket) * 100
                print(f"    Inflation {lo:>+4d} to {hi:>+4d}%: {sum(u for _, u in bucket):>6d}/{len(bucket):>6d} = {rate:>5.1f}% under")

    # ================================================================
    # PHASE 2: Line floor exploitation — BLK under 0.5 for low-average players
    # ================================================================
    print(f"\n{'='*120}")
    print("PHASE 2: LINE FLOOR EXPLOITATION (BLK/STL line=0.5, player avg well below)")
    print(f"{'─'*120}")

    # Group by line level
    for stat in ['blk', 'stl']:
        print(f"\n  {stat.upper()}:")
        for line_val in [0.5, 1.0, 1.5, 2.0, 2.5]:
            hits, total = 0, 0
            for d in dates:
                for p in dp[d]:
                    if p['stat'] == stat and p['line'] == line_val:
                        total += 1
                        if p['under']: hits += 1
            if total >= 50:
                print(f"    Line={line_val}: {hits}/{total} = {hits/total*100:.1f}% under (avg inflation: n/a)")

        # BLK line=0.5 broken down by player average
        for avg_lo, avg_hi in [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]:
            hits, total = 0, 0
            for d in dates:
                for p in dp[d]:
                    if p['stat'] == stat and p['line'] == 0.5 and avg_lo <= p['avg10'] < avg_hi:
                        total += 1
                        if p['under']: hits += 1
            if total >= 30:
                print(f"      Line=0.5, avg {avg_lo:.1f}-{avg_hi:.1f}: {hits}/{total} = {hits/total*100:.1f}% under")

    # ================================================================
    # PHASE 3: Build 5-leg and 6-leg parlays with line floor awareness
    # ================================================================
    print(f"\n{'='*120}")
    print("PHASE 3: 5-leg and 6-leg parlays with line floor exploitation")
    print(f"{'─'*120}")
    print(f"  {'Strategy':<55s} {'W':>5s} {'T':>6s} {'Rate':>7s} {'LegHR':>7s} {'MaxS':>5s} {'5+S':>4s} {'10+S':>5s} {'Skip%':>6s}")

    configs = [
        # Baseline: BLK/STL any
        ('blkstl_uber8_5L', 5, ['blk','stl'], 8, None, None),
        ('blkstl_uber8_6L', 6, ['blk','stl'], 8, None, None),

        # High inflation only (line >> avg)
        ('blkstl_infl25+_uber6_5L', 5, ['blk','stl'], 6, lambda p: p['inflation'] >= 25, None),
        ('blkstl_infl25+_uber6_6L', 6, ['blk','stl'], 6, lambda p: p['inflation'] >= 25, None),
        ('blkstl_infl50+_uber6_5L', 5, ['blk','stl'], 6, lambda p: p['inflation'] >= 50, None),
        ('blkstl_infl50+_uber6_6L', 6, ['blk','stl'], 6, lambda p: p['inflation'] >= 50, None),

        # Line floor: only line=0.5 BLK/STL
        ('blkstl_line05_uber6_5L', 5, ['blk','stl'], 6, lambda p: p['line'] == 0.5, None),
        ('blkstl_line05_uber6_6L', 6, ['blk','stl'], 6, lambda p: p['line'] == 0.5, None),

        # Line floor + low average
        ('blkstl_line05_avg04_5L', 5, ['blk','stl'], 4, lambda p: p['line'] == 0.5 and p['avg10'] < 0.4, None),
        ('blkstl_line05_avg04_6L', 6, ['blk','stl'], 4, lambda p: p['line'] == 0.5 and p['avg10'] < 0.4, None),
        ('blkstl_line05_avg03_5L', 5, ['blk','stl'], 4, lambda p: p['line'] == 0.5 and p['avg10'] < 0.3, None),

        # Mix: some line-floor + some high-conf
        ('all_infl25+_uber8_5L', 5, None, 8, lambda p: p['inflation'] >= 25, None),
        ('all_infl25+_uber8_6L', 6, None, 8, lambda p: p['inflation'] >= 25, None),
        ('all_infl50+_uber6_5L', 5, None, 6, lambda p: p['inflation'] >= 50, None),

        # High inflation + away
        ('blkstl_infl25+_away_5L', 5, ['blk','stl'], 6, lambda p: p['inflation'] >= 25 and p.get('away'), None),
        ('blkstl_infl25+_away_6L', 6, ['blk','stl'], 6, lambda p: p['inflation'] >= 25 and p.get('away'), None),

        # High inflation + COLD
        ('blkstl_infl25+_cold_5L', 5, ['blk','stl'], 4, lambda p: p['inflation'] >= 25 and p['stk'] == 'COLD', None),

        # Ultra-selective: high inflation + high uber
        ('blkstl_infl50+_uber10_5L', 5, ['blk','stl'], 10, lambda p: p['inflation'] >= 50, None),
        ('blkstl_infl25+_uber12_5L', 5, ['blk','stl'], 12, lambda p: p['inflation'] >= 25, None),

        # All stats with high inflation
        ('all_infl50+_uber10_5L', 5, None, 10, lambda p: p['inflation'] >= 50, None),
        ('all_infl50+_uber10_6L', 6, None, 10, lambda p: p['inflation'] >= 50, None),

        # Include 3PM (also has line floor at 0.5)
        ('blkstl3pm_infl25+_5L', 5, ['blk','stl','3pm'], 6, lambda p: p['inflation'] >= 25, None),
        ('blkstl3pm_infl25+_6L', 6, ['blk','stl','3pm'], 6, lambda p: p['inflation'] >= 25, None),
    ]

    for name, n, sf, mu, ef, ma in configs:
        w, t, lr, ms, s5, s10, sk = run_parlay(dates, dp, n, sf, mu, extra_filter=ef, min_avg=ma)
        if t >= 20:
            sp = sk/(sk+t)*100
            print(f"  {name:<55s} {w:>5d} {t:>6d} {w/t*100:>6.1f}% {lr*100:>6.1f}% {ms:>5d} {s5:>4d} {s10:>5d} {sp:>5.0f}%")

    # ================================================================
    # PHASE 4: Year-by-year consistency of best strategies
    # ================================================================
    print(f"\n{'='*120}")
    print("PHASE 4: Year-by-year consistency of line-floor strategy")
    print(f"{'─'*120}")

    # Best strategy: BLK/STL with inflation >= 25%
    for year in range(2016, 2027):
        year_dates = [d for d in dates if d.startswith(str(year))]
        if not year_dates: continue
        for name, n, sf, mu, ef in [
            ('blkstl_infl25+_5L', 5, ['blk','stl'], 6, lambda p: p['inflation'] >= 25),
            ('blkstl_infl50+_5L', 5, ['blk','stl'], 6, lambda p: p['inflation'] >= 50),
        ]:
            w, t, lr, ms, s5, s10, _ = run_parlay(year_dates, dp, n, sf, mu, extra_filter=ef)
            if t > 0:
                print(f"  {year} {name}: {w:>3d}/{t:>4d} = {w/t*100:>5.1f}% LR={lr*100:.1f}% MS={ms}")

    # Save
    out = os.path.join(os.path.dirname(__file__), 'line_floor_results.json')
    # Quick summary
    best_results = {}
    for name, n, sf, mu, ef, ma in configs[:6]:
        w, t, lr, ms, s5, s10, sk = run_parlay(dates, dp, n, sf, mu, extra_filter=ef, min_avg=ma)
        best_results[name] = {'w': w, 't': t, 'r': w/t if t else 0, 'lr': lr, 'ms': ms}
    with open(out, 'w') as f:
        json.dump(best_results, f, indent=2)
    print(f"\nSaved to {out}", file=sys.stderr)


if __name__ == '__main__':
    main()

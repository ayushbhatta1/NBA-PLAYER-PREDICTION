#!/usr/bin/env python3
"""
MEGA BACKTEST V3: Hybrids + Skip-Day + Ultra-Selective.
Goal: Push 2-leg parlay rate above 60% and find 20+ day streaks.
Key insights from V1/V2:
  - BLK UNDER is king (64% base UNDER rate)
  - Away game boosts BLK UNDER to 58.6%
  - Floor analysis (2nd-lowest < line) adds reliability
  - stat_diverse with BLK priority gets 18-day streaks
  - Skip-day (only bet good days) can sacrifice volume for rate

New ideas to test:
  - Combine BLK-away + floor + skip-day
  - "Quality gate": only bet when top picks exceed threshold
  - Opponent-specific: teams that allow fewer BLK
  - Minutes threshold: only pick players with 25+ mins
  - Blend multiple signals into uber-score
  - 2-leg with one BLK + one STL (mixed defense stats)
  - Dynamic: skip if recent streak is cold (our own streak)
"""

import csv, json, sys, os
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

def load_and_compute():
    print("Loading...", file=sys.stderr)
    player_games = defaultdict(list)
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
            is_home = True if home_val in ('True','true','1') else (False if home_val in ('False','false','0') else None)
            player_games[pid].append({
                'date': date[:10], 'name': f"{row.get('firstName','')} {row.get('lastName','')}",
                'home': is_home, 'mins': mins,
                'blk': float(row.get('blocks',0) or 0), 'stl': float(row.get('steals',0) or 0),
                'pts': float(row.get('points',0) or 0), 'reb': float(row.get('reboundsTotal',0) or 0),
                'ast': float(row.get('assists',0) or 0), '3pm': float(row.get('threePointersMade',0) or 0),
                'pf': float(row.get('foulsPersonal',0) or 0), 'pm': float(row.get('plusMinusPoints',0) or 0),
            })
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g['date'])

    print("Computing props...", file=sys.stderr)
    props = []
    stats = ['pts','reb','ast','3pm','blk','stl']
    inflation = 4

    for pid, games in player_games.items():
        if len(games) < 15: continue
        for i in range(15, len(games)):
            cur = games[i]
            prior = games[:i]
            for stat in stats:
                l10 = prior[-10:]
                l10_vals = [g[stat] for g in l10]
                l10_avg = sum(l10_vals)/10
                if l10_avg < MIN_LINES.get(stat, 0.5): continue
                line = round((l10_avg * (1 + inflation/100)) * 2) / 2
                actual = cur[stat]
                if actual == line: continue

                l5_vals = [g[stat] for g in prior[-5:]]
                l3_vals = [g[stat] for g in prior[-3:]]
                season_vals = [g[stat] for g in prior if g['date'][:4] == cur['date'][:4]] or [g[stat] for g in prior[-30:]]

                l5_avg = sum(l5_vals)/5
                l3_avg = sum(l3_vals)/3
                l10_hr = sum(1 for v in l10_vals if v > line)/10*100
                season_hr = sum(1 for v in season_vals if v > line)/len(season_vals)*100
                l10_miss = sum(1 for v in l10_vals if v < line)
                l10_std = (sum((v-l10_avg)**2 for v in l10_vals)/10)**0.5
                l10_cv = l10_std / max(l10_avg, 0.1)
                l10_floor = sorted(l10_vals)[1] if len(l10_vals) > 1 else min(l10_vals)
                l10_median = sorted(l10_vals)[5]
                gap = l10_avg - line

                if l3_avg > l10_avg * 1.15: streak = 'HOT'
                elif l3_avg < l10_avg * 0.85: streak = 'COLD'
                else: streak = 'NEUTRAL'

                consec_under = 0
                for v in reversed(l10_vals):
                    if v < line: consec_under += 1
                    else: break

                l10_mins = [g['mins'] for g in l10]
                mins_avg = sum(l10_mins)/10

                # Rest
                try:
                    d1 = datetime.strptime(prior[-1]['date'], '%Y-%m-%d')
                    d2 = datetime.strptime(cur['date'], '%Y-%m-%d')
                    rest = (d2-d1).days - 1
                except: rest = 1

                props.append({
                    'date': cur['date'], 'name': cur['name'], 'stat': stat,
                    'line': line, 'actual': actual, 'under': actual < line,
                    'home': cur['home'], 'is_away': not cur['home'] if cur['home'] is not None else None,
                    'l10_avg': l10_avg, 'l5_avg': l5_avg, 'l3_avg': l3_avg,
                    'gap': gap, 'l10_hr': l10_hr, 'season_hr': season_hr,
                    'l10_miss': l10_miss, 'l10_std': l10_std, 'l10_cv': l10_cv,
                    'l10_floor': l10_floor, 'l10_median': l10_median,
                    'streak': streak, 'consec_under': consec_under,
                    'rest': rest, 'is_b2b': rest == 0,
                    'mins_avg': mins_avg,
                })

    print(f"  {len(props):,} props generated", file=sys.stderr)
    return props


def uber_score(p):
    """Ultimate scoring function combining all winning signals."""
    s = 0.0

    # 1. Hit rate signal (strongest predictor)
    hr = p['l10_hr']
    if hr <= 10: s += 4.0
    elif hr <= 20: s += 3.0
    elif hr <= 30: s += 2.0
    elif hr <= 40: s += 1.0
    elif hr >= 70: s -= 2.0
    elif hr >= 60: s -= 1.0

    # 2. Season HR
    shr = p['season_hr']
    if shr < 30: s += 2.0
    elif shr < 45: s += 0.5
    elif shr >= 60: s -= 1.0

    # 3. Stat type (BLK >>> STL >>> others)
    stat_w = {'blk': 3.0, 'stl': 2.0, '3pm': 1.0, 'ast': 0.5}
    s += stat_w.get(p['stat'], 0)

    # 4. Floor analysis
    if p['l10_floor'] < p['line']:
        s += min((p['line'] - p['l10_floor']) / max(p['line'], 0.5) * 5, 3.0)

    # 5. Median below line
    if p['l10_median'] < p['line']:
        s += 1.0

    # 6. Gap (line above average)
    gap = p['gap']
    if gap < -5: s += 2.5
    elif gap < -3: s += 2.0
    elif gap < -1.5: s += 1.0
    elif gap < 0: s += 0.5
    elif gap > 3: s -= 1.0

    # 7. Miss count
    mc = p['l10_miss']
    if mc >= 9: s += 2.5
    elif mc >= 7: s += 1.5
    elif mc >= 5: s += 0.5
    elif mc <= 2: s -= 1.0

    # 8. Streak
    if p['streak'] == 'COLD': s += 1.5
    elif p['streak'] == 'HOT': s -= 1.0

    # 9. Away game
    if p.get('is_away'): s += 0.7

    # 10. B2B
    if p.get('is_b2b'): s += 0.5

    # 11. Consecutive unders
    if p['consec_under'] >= 5: s += 2.0
    elif p['consec_under'] >= 3: s += 1.0

    # 12. Low variance (predictability)
    if p['l10_cv'] < 0.3: s += 0.5
    elif p['l10_cv'] > 1.5: s -= 0.5

    # 13. High minutes (reliable role)
    if p['mins_avg'] >= 30: s += 0.3

    return s


def run_strat(props, filt, sort_fn, legs, name, min_cand=0, skip_fn=None, max_per_stat=None):
    by_date = defaultdict(list)
    for p in props:
        if filt(p):
            by_date[p['date']].append(p)

    wins = total = leg_hits = leg_total = cur_str = max_str = 0
    streaks = []; daily = []; skipped = 0

    for date in sorted(by_date.keys()):
        cands = by_date[date]
        if len(cands) < max(min_cand, legs):
            skipped += 1; continue
        if skip_fn and skip_fn(cands):
            skipped += 1; continue

        cands.sort(key=sort_fn, reverse=True)
        sel = []; used_names = set(); stat_counts = defaultdict(int)
        for p in cands:
            if p['name'] in used_names: continue
            if max_per_stat and stat_counts[p['stat']] >= max_per_stat: continue
            sel.append(p)
            used_names.add(p['name'])
            stat_counts[p['stat']] += 1
            if len(sel) >= legs: break
        if len(sel) < legs: continue

        hit = all(p['under'] for p in sel)
        nh = sum(p['under'] for p in sel)
        total += 1; leg_total += legs; leg_hits += nh
        if hit: wins += 1; cur_str += 1; max_str = max(max_str, cur_str)
        else:
            if cur_str > 0: streaks.append(cur_str)
            cur_str = 0
        daily.append({'date': date, 'hit': hit, 'legs_hit': nh})

    if cur_str > 0: streaks.append(cur_str)
    return {
        'name': name, 'wins': wins, 'total': total,
        'rate': wins/max(total,1), 'leg_rate': leg_hits/max(leg_total,1),
        'max_streak': max_str,
        'streaks_5plus': sum(1 for s in streaks if s >= 5),
        'streaks_10plus': sum(1 for s in streaks if s >= 10),
        'skipped': skipped, 'daily': daily,
    }


def main():
    props = load_and_compute()

    strategies = []

    # ============================================================
    # SECTION A: UBER SCORE STRATEGIES
    # ============================================================
    for thresh in [6, 7, 8, 9, 10, 11, 12, 13, 14]:
        for legs in [2, 3]:
            strategies.append((
                lambda p, t=thresh: uber_score(p) >= t,
                uber_score, legs, f"uber>={thresh}_{legs}leg"
            ))

    # Uber + stat filter
    for thresh in [7, 8, 9, 10]:
        strategies.append((
            lambda p, t=thresh: uber_score(p) >= t and p['stat'] in ['blk','stl'],
            uber_score, 2, f"uber{thresh}_blkstl_2leg"
        ))
        strategies.append((
            lambda p, t=thresh: uber_score(p) >= t and p['stat'] == 'blk',
            uber_score, 2, f"uber{thresh}_blk_2leg"
        ))

    # ============================================================
    # SECTION B: HYBRID BLK-AWAY-FLOOR COMBOS
    # ============================================================
    # BLK + away + floor
    strategies.append((
        lambda p: p['stat'] == 'blk' and p.get('is_away') and p['l10_floor'] < p['line'],
        uber_score, 2, "blk_away_floor_2leg"
    ))
    # BLK + away + miss >= 7
    strategies.append((
        lambda p: p['stat'] == 'blk' and p.get('is_away') and p['l10_miss'] >= 7,
        uber_score, 2, "blk_away_miss7_2leg"
    ))
    # BLK + away + cold
    strategies.append((
        lambda p: p['stat'] == 'blk' and p.get('is_away') and p['streak'] == 'COLD',
        uber_score, 2, "blk_away_cold_2leg"
    ))
    # BLK + away + low HR
    strategies.append((
        lambda p: p['stat'] == 'blk' and p.get('is_away') and p['l10_hr'] <= 30,
        uber_score, 2, "blk_away_lowhr_2leg"
    ))
    # BLK/STL + away + floor
    strategies.append((
        lambda p: p['stat'] in ['blk','stl'] and p.get('is_away') and p['l10_floor'] < p['line'],
        uber_score, 2, "blkstl_away_floor_2leg"
    ))

    # ============================================================
    # SECTION C: SKIP-DAY STRATEGIES (only bet on good days)
    # ============================================================
    # Skip unless top-2 picks both have uber >= X
    for min_top2 in [8, 9, 10, 11, 12]:
        def make_skip(mt2):
            def skip_fn(cands):
                scores = sorted([uber_score(p) for p in cands], reverse=True)
                if len(scores) < 2: return True
                return scores[1] < mt2  # 2nd best must meet threshold
            return skip_fn
        strategies.append((
            lambda p: True,
            uber_score, 2, f"skip_top2uber{min_top2}_2leg",
            0, make_skip(min_top2)
        ))

    # Skip unless N+ BLK props available
    for min_blk in [2, 3, 4, 5]:
        def make_skip_blk(mb):
            def skip_fn(cands):
                return sum(1 for p in cands if p['stat'] == 'blk') < mb
            return skip_fn
        strategies.append((
            lambda p: p['stat'] == 'blk',
            uber_score, 2, f"skip_minblk{min_blk}_2leg",
            0, make_skip_blk(min_blk)
        ))

    # Skip unless average uber score of BLK candidates > threshold
    for min_avg_uber in [8, 9, 10, 11]:
        def make_skip_avg(mau):
            def skip_fn(cands):
                blk_cands = [p for p in cands if p['stat'] == 'blk']
                if len(blk_cands) < 2: return True
                return sum(uber_score(p) for p in blk_cands)/len(blk_cands) < mau
            return skip_fn
        strategies.append((
            lambda p: p['stat'] == 'blk',
            uber_score, 2, f"skip_blkavg{min_avg_uber}_2leg",
            0, make_skip_avg(min_avg_uber)
        ))

    # ============================================================
    # SECTION D: STAT DIVERSITY (forced mix)
    # ============================================================
    # Max 1 per stat type — forces diversification
    for thresh in [7, 8, 9, 10]:
        strategies.append((
            lambda p, t=thresh: uber_score(p) >= t,
            uber_score, 2, f"diverse1_uber{thresh}_2leg",
            0, None, 1  # max_per_stat=1
        ))
    for thresh in [6, 7, 8]:
        strategies.append((
            lambda p, t=thresh: uber_score(p) >= t,
            uber_score, 3, f"diverse1_uber{thresh}_3leg",
            0, None, 1
        ))

    # ============================================================
    # SECTION E: ONE BLK + ONE OTHER
    # ============================================================
    # Force 1 BLK leg + 1 best-other-stat leg
    # Approximate: sort BLK first, then diversity kicks in
    for legs in [2]:
        def blk_first_sort(p):
            base = uber_score(p)
            if p['stat'] == 'blk': base += 100  # BLK always first
            return base
        strategies.append((
            lambda p: uber_score(p) >= 6,
            blk_first_sort, 2, "blk_first_uber6_diverse_2leg",
            0, None, 1
        ))
        strategies.append((
            lambda p: uber_score(p) >= 7,
            blk_first_sort, 2, "blk_first_uber7_diverse_2leg",
            0, None, 1
        ))

    # ============================================================
    # SECTION F: ADAPTIVE — use our own streak to decide
    # ============================================================
    # After 2 losses in a row, skip next day
    # After 3 wins, increase bet (lower threshold)
    # These need special runner logic — skip for now, handle in a custom loop

    # ============================================================
    # SECTION G: ULTRA THRESHOLDS
    # ============================================================
    for thresh in [15, 16, 17, 18, 20]:
        strategies.append((
            lambda p, t=thresh: uber_score(p) >= t,
            uber_score, 2, f"ultra_uber{thresh}_2leg"
        ))

    # ============================================================
    # SECTION H: SPECIFIC WINNING COMBOS FROM V2
    # ============================================================
    # BLK + conf >= 6 + away
    from mega_backtest_v2 import conf_score_v2
    for ct in [5, 6, 7, 8]:
        strategies.append((
            lambda p, t=ct: p['stat'] == 'blk' and conf_score_v2(p) >= t and p.get('is_away'),
            lambda p: conf_score_v2(p) + uber_score(p) / 5,
            2, f"blk_away_conf{ct}_uber_2leg"
        ))

    print(f"Generated {len(strategies)} V3 strategies", file=sys.stderr)

    results = []
    for i, strat in enumerate(strategies):
        filt, sort_fn, legs, name = strat[0], strat[1], strat[2], strat[3]
        min_cand = strat[4] if len(strat) > 4 else 0
        skip_fn = strat[5] if len(strat) > 5 else None
        max_per_stat = strat[6] if len(strat) > 6 else None

        try:
            r = run_strat(props, filt, sort_fn, legs, name, min_cand, skip_fn, max_per_stat)
            if r['total'] >= 20:
                results.append(r)
        except Exception as e:
            print(f"  ERR {name}: {e}", file=sys.stderr)

        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(strategies)}...", file=sys.stderr)

    results.sort(key=lambda r: r['rate'], reverse=True)

    print(f"\n{'='*105}")
    print(f"V3 RESULTS: {len(results)} strategies")
    print(f"{'='*105}")
    print(f"{'#':>3} {'Strategy':<48} {'W/L':>12} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4} {'Skip':>5}")
    for i, r in enumerate(results[:80]):
        print(f"{i+1:>3} {r['name']:<48} {r['wins']:>4}/{r['total']:<6} {r['rate']:>6.1%} {r['leg_rate']:>6.1%} {r['max_streak']:>6} {r['streaks_5plus']:>4} {r['streaks_10plus']:>4} {r['skipped']:>5}")

    # Best by streak
    by_s = sorted(results, key=lambda r: (r['max_streak'], r['rate']), reverse=True)
    print(f"\n{'='*105}")
    print(f"TOP 30 BY MAX STREAK")
    print(f"{'='*105}")
    print(f"{'#':>3} {'Strategy':<48} {'W/L':>12} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4} {'Skip':>5}")
    for i, r in enumerate(by_s[:30]):
        print(f"{i+1:>3} {r['name']:<48} {r['wins']:>4}/{r['total']:<6} {r['rate']:>6.1%} {r['leg_rate']:>6.1%} {r['max_streak']:>6} {r['streaks_5plus']:>4} {r['streaks_10plus']:>4} {r['skipped']:>5}")

    # Best high-rate with 10+ streak
    hr_10 = [r for r in results if r['max_streak'] >= 10]
    if hr_10:
        hr_10.sort(key=lambda r: r['rate'], reverse=True)
        print(f"\n{'='*105}")
        print(f"STRATEGIES WITH 10+ DAY STREAKS (sorted by rate)")
        print(f"{'='*105}")
        print(f"{'#':>3} {'Strategy':<48} {'W/L':>12} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4} {'Skip':>5}")
        for i, r in enumerate(hr_10[:30]):
            print(f"{i+1:>3} {r['name']:<48} {r['wins']:>4}/{r['total']:<6} {r['rate']:>6.1%} {r['leg_rate']:>6.1%} {r['max_streak']:>6} {r['streaks_5plus']:>4} {r['streaks_10plus']:>4} {r['skipped']:>5}")

    # Deep dive best
    if results:
        best = results[0]
        print(f"\n{'='*105}")
        print(f"YEARLY: {best['name']} ({best['rate']:.1%})")
        print(f"{'='*105}")
        yearly = defaultdict(lambda: [0,0])
        for d in best['daily']:
            yearly[d['date'][:4]][1] += 1
            if d['hit']: yearly[d['date'][:4]][0] += 1
        for y in sorted(yearly.keys()):
            w, t = yearly[y]
            print(f"  {y}: {w:>3}/{t:<3} = {w/max(t,1):.1%}")

    # Save
    out = [{'name':r['name'],'wins':r['wins'],'total':r['total'],'rate':round(r['rate'],4),
            'max_streak':r['max_streak'],'s5':r['streaks_5plus'],'s10':r['streaks_10plus']}
           for r in results[:100]]
    path = os.path.join(os.path.dirname(__file__), 'mega_backtest_v3_results.json')
    with open(path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved to {path}", file=sys.stderr)


if __name__ == '__main__':
    main()

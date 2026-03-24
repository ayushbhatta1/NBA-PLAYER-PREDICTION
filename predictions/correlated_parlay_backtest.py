#!/usr/bin/env python3
"""
CORRELATED PARLAY BACKTEST: The secret to 5-8 leg parlays.

People who hit long parlays aren't picking independent legs — they're
STACKING correlated legs from the same game. When a game goes low-scoring
or a blowout happens, 5+ players ALL go under together.

This script:
1. Groups all props by GAME (date + team matchup)
2. Measures how correlated unders are within a game
3. Finds game-level predictors (spread, total, pace, etc.)
4. Tests same-game stacking strategies for 5-8 leg parlays
5. Tests cross-game but correlated strategies
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


def load_games():
    """Load data grouped by game (date + teams)."""
    print("Loading...", file=sys.stderr)
    # Group by player first to compute rolling stats
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
                'date': d[:10],
                'name': f"{row.get('firstName','')} {row.get('lastName','')}",
                'team': row.get('playerteamName', ''),
                'opp': row.get('opponentteamName', ''),
                'home': is_home, 'mins': mins,
                'pts': float(row.get('points',0) or 0),
                'reb': float(row.get('reboundsTotal',0) or 0),
                'ast': float(row.get('assists',0) or 0),
                '3pm': float(row.get('threePointersMade',0) or 0),
                'blk': float(row.get('blocks',0) or 0),
                'stl': float(row.get('steals',0) or 0),
                'pf': float(row.get('foulsPersonal',0) or 0),
                'pm': float(row.get('plusMinusPoints',0) or 0),
                'win': row.get('win','') in ('True','true','1'),
                'fga': float(row.get('fieldGoalsAttempted',0) or 0),
                'fta': float(row.get('freeThrowsAttempted',0) or 0),
                'gameId': row.get('gameId', ''),
            })
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g['date'])
    print(f"Loaded {len(player_games)} players", file=sys.stderr)
    return player_games


def compute_game_props(player_games, inflation=4):
    """Generate props and group by game. Each prop has game_key for correlation."""
    stats = ['pts', 'reb', 'ast', '3pm', 'blk', 'stl']
    # game_key -> list of props
    game_props = defaultdict(list)
    all_props = []

    for pid, games in player_games.items():
        if len(games) < 15: continue
        for i in range(15, len(games)):
            cur = games[i]
            prior = games[:i]

            # Game-level context
            game_key = f"{cur['date']}_{cur['gameId']}" if cur.get('gameId') else f"{cur['date']}_{sorted([cur['team'], cur['opp']])}"

            # Compute game-level features from this player's perspective
            # Team total pts (proxy for game pace)
            team_pts_l10 = sum(g['pts'] for g in prior[-10:]) / 10

            for stat in stats:
                l10 = prior[-10:]
                v10 = [g[stat] for g in l10]
                avg10 = sum(v10)/10
                if avg10 < MIN_LINES.get(stat, 0.5): continue
                line = round((avg10 * (1 + inflation/100)) * 2) / 2
                actual = cur[stat]
                if actual == line: continue

                v5 = [g[stat] for g in prior[-5:]]
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

                # Rest days
                try:
                    d1 = datetime.strptime(prior[-1]['date'], '%Y-%m-%d')
                    d2 = datetime.strptime(cur['date'], '%Y-%m-%d')
                    rest = (d2-d1).days - 1
                except: rest = 1

                # Win/loss L10
                l10_wins = sum(1 for g in prior[-10:] if g.get('win'))

                prop = {
                    'date': cur['date'], 'name': cur['name'], 'stat': stat,
                    'team': cur['team'], 'opp': cur['opp'], 'game_key': game_key,
                    'line': line, 'actual': actual, 'under': actual < line,
                    'home': cur['home'], 'away': not cur['home'] if cur['home'] is not None else None,
                    'avg10': avg10, 'gap': gap, 'hr10': hr10, 'shr': shr,
                    'miss': miss, 'std': std, 'flr': flr,
                    'stk': stk, 'cu': cu, 'rest': rest, 'b2b': rest == 0,
                    'mins': cur['mins'], 'mavg': sum(g['mins'] for g in l10)/10,
                    'team_pts_l10': team_pts_l10,
                    'l10_wins': l10_wins,
                    'win': cur.get('win', False),
                    'pm': cur.get('pm', 0),
                }
                game_props[game_key].append(prop)
                all_props.append(prop)

    print(f"  {len(all_props):,} props across {len(game_props):,} games", file=sys.stderr)
    return game_props, all_props


def conf(p):
    """Confidence score for UNDER."""
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
    return s


def main():
    player_games = load_games()
    game_props, all_props = compute_game_props(player_games)

    # ================================================================
    # ANALYSIS 1: How correlated are unders within the same game?
    # ================================================================
    print(f"\n{'='*90}")
    print("ANALYSIS 1: UNDER CORRELATION WITHIN GAMES")
    print(f"{'='*90}")

    # For each game with 8+ props, what % of props went under?
    game_under_rates = []
    for gk, props in game_props.items():
        if len(props) < 8: continue
        ur = sum(p['under'] for p in props) / len(props)
        game_under_rates.append((gk, ur, len(props), props))

    if game_under_rates:
        # Distribution of game-level under rates
        buckets = defaultdict(int)
        for _, ur, _, _ in game_under_rates:
            b = round(ur * 10) * 10  # 0%, 10%, 20%...
            buckets[b] += 1

        print(f"\nGame-level UNDER rate distribution ({len(game_under_rates)} games with 8+ props):")
        for b in sorted(buckets.keys()):
            print(f"  {b:>3}% under: {buckets[b]:>5} games ({buckets[b]/len(game_under_rates)*100:.1f}%)")

        # Key insight: what % of games have 70%+ under rate?
        high_under = sum(1 for _, ur, _, _ in game_under_rates if ur >= 0.70)
        very_high = sum(1 for _, ur, _, _ in game_under_rates if ur >= 0.80)
        print(f"\n  Games with 70%+ under rate: {high_under}/{len(game_under_rates)} = {high_under/len(game_under_rates)*100:.1f}%")
        print(f"  Games with 80%+ under rate: {very_high}/{len(game_under_rates)} = {very_high/len(game_under_rates)*100:.1f}%")

    # ================================================================
    # ANALYSIS 2: Can we PREDICT which games will be high-under?
    # ================================================================
    print(f"\n{'='*90}")
    print("ANALYSIS 2: PREDICTING HIGH-UNDER GAMES")
    print(f"{'='*90}")

    # For each game, compute pre-game signals
    game_signals = []
    for gk, props in game_props.items():
        if len(props) < 8: continue
        ur = sum(p['under'] for p in props) / len(props)

        # Game-level signals (computed from individual player histories)
        avg_conf = sum(conf(p) for p in props) / len(props)
        avg_miss = sum(p['miss'] for p in props) / len(props)
        avg_hr = sum(p['hr10'] for p in props) / len(props)
        n_cold = sum(1 for p in props if p['stk'] == 'COLD')
        n_hot = sum(1 for p in props if p['stk'] == 'HOT')
        n_away = sum(1 for p in props if p.get('away'))
        n_b2b = sum(1 for p in props if p.get('b2b'))
        n_blk = sum(1 for p in props if p['stat'] == 'blk')
        n_stl = sum(1 for p in props if p['stat'] == 'stl')
        avg_gap = sum(p['gap'] for p in props) / len(props)

        # Team-level: average wins in L10 (bad teams go under more)
        avg_wins = sum(p['l10_wins'] for p in props) / len(props)

        # Number of high-confidence UNDER picks
        n_high_conf = sum(1 for p in props if conf(p) >= 5)
        n_ultra_conf = sum(1 for p in props if conf(p) >= 8)

        game_signals.append({
            'gk': gk, 'date': props[0]['date'], 'ur': ur, 'n': len(props),
            'avg_conf': avg_conf, 'avg_miss': avg_miss, 'avg_hr': avg_hr,
            'n_cold': n_cold, 'n_hot': n_hot, 'n_away': n_away, 'n_b2b': n_b2b,
            'n_blk': n_blk, 'n_stl': n_stl, 'avg_gap': avg_gap,
            'avg_wins': avg_wins, 'n_high_conf': n_high_conf, 'n_ultra_conf': n_ultra_conf,
            'props': props,
        })

    # Test different game-level filters
    print(f"\nGame-level predictors of 70%+ under rate:")
    print(f"  {'Signal':<45} {'70%+ Rate':>12} {'Sample':>8}")

    tests = [
        ("All games", lambda g: True),
        ("avg_conf >= 4", lambda g: g['avg_conf'] >= 4),
        ("avg_conf >= 5", lambda g: g['avg_conf'] >= 5),
        ("avg_conf >= 6", lambda g: g['avg_conf'] >= 6),
        ("avg_miss >= 6", lambda g: g['avg_miss'] >= 6),
        ("avg_miss >= 7", lambda g: g['avg_miss'] >= 7),
        ("avg_hr <= 35", lambda g: g['avg_hr'] <= 35),
        ("avg_hr <= 30", lambda g: g['avg_hr'] <= 30),
        ("n_high_conf >= 5", lambda g: g['n_high_conf'] >= 5),
        ("n_high_conf >= 8", lambda g: g['n_high_conf'] >= 8),
        ("n_high_conf >= 10", lambda g: g['n_high_conf'] >= 10),
        ("n_ultra_conf >= 3", lambda g: g['n_ultra_conf'] >= 3),
        ("n_ultra_conf >= 5", lambda g: g['n_ultra_conf'] >= 5),
        ("n_cold >= 3", lambda g: g['n_cold'] >= 3),
        ("n_cold >= 5", lambda g: g['n_cold'] >= 5),
        ("avg_gap <= -2", lambda g: g['avg_gap'] <= -2),
        ("avg_gap <= -3", lambda g: g['avg_gap'] <= -3),
        ("n_b2b >= 2", lambda g: g['n_b2b'] >= 2),
        ("avg_wins <= 4 (bad teams)", lambda g: g['avg_wins'] <= 4),
    ]

    for label, filt in tests:
        matching = [g for g in game_signals if filt(g)]
        if len(matching) < 30: continue
        rate_70 = sum(1 for g in matching if g['ur'] >= 0.70) / len(matching)
        print(f"  {label:<45} {rate_70:>11.1%} {len(matching):>8}")

    # ================================================================
    # ANALYSIS 3: SAME-GAME PARLAY SIMULATION (5-8 legs)
    # ================================================================
    print(f"\n{'='*90}")
    print("ANALYSIS 3: SAME-GAME PARLAY SIMULATION")
    print(f"{'='*90}")

    def run_sgp(game_signals, game_filter, pick_filter, sort_fn, legs, name):
        """Run same-game parlay strategy.
        game_filter: which games to consider
        pick_filter: which props within a game to consider
        sort_fn: how to rank picks within a game
        legs: how many legs per parlay
        """
        by_date = defaultdict(list)
        for g in game_signals:
            if game_filter(g):
                by_date[g['date']].append(g)

        wins = total = leg_hits = leg_total = 0
        cs = ms = 0
        streaks = []
        daily = []

        for date in sorted(by_date.keys()):
            # Pick the best game for this date
            day_games = by_date[date]
            best_game = None
            best_score = -999

            for game in day_games:
                cands = [p for p in game['props'] if pick_filter(p)]
                if len(cands) < legs: continue
                game_score = sum(sort_fn(p) for p in sorted(cands, key=sort_fn, reverse=True)[:legs]) / legs
                if game_score > best_score:
                    best_score = game_score
                    best_game = game

            if best_game is None: continue

            # Select top picks from the best game
            cands = [p for p in best_game['props'] if pick_filter(p)]
            cands.sort(key=sort_fn, reverse=True)
            sel = []
            used = set()
            for p in cands:
                if p['name'] in used: continue
                sel.append(p)
                used.add(p['name'])
                if len(sel) >= legs: break
            if len(sel) < legs: continue

            hit = all(p['under'] for p in sel)
            nh = sum(p['under'] for p in sel)
            total += 1; leg_total += legs; leg_hits += nh
            if hit: wins += 1; cs += 1; ms = max(ms, cs)
            else:
                if cs > 0: streaks.append(cs)
                cs = 0
            daily.append({'date': date, 'hit': hit, 'nh': nh})

        if cs > 0: streaks.append(cs)
        return {
            'name': name, 'w': wins, 't': total,
            'r': wins/max(total,1), 'lr': leg_hits/max(leg_total,1),
            'ms': ms, 's5': sum(1 for s in streaks if s >= 5),
            's3': sum(1 for s in streaks if s >= 3),
            'daily': daily,
        }

    # SAME-GAME STRATEGIES
    sgp_strats = []

    # --- 5-leg strategies ---
    for min_hc in [3, 5, 8]:
        sgp_strats.append((
            lambda g, mhc=min_hc: g['n_high_conf'] >= mhc,
            lambda p: conf(p) >= 3,
            conf, 5, f"sgp5_hc{min_hc}_conf3"
        ))

    # All UNDER picks, sorted by confidence
    sgp_strats.append((
        lambda g: g['n_high_conf'] >= 5,
        lambda p: True,
        conf, 5, "sgp5_all_byconf"
    ))

    # BLK/STL focused same-game (pick defensive stats)
    sgp_strats.append((
        lambda g: g['n_blk'] + g['n_stl'] >= 5,
        lambda p: p['stat'] in ['blk', 'stl'],
        conf, 5, "sgp5_defense_only"
    ))

    # High miss games
    sgp_strats.append((
        lambda g: g['avg_miss'] >= 6,
        lambda p: p['miss'] >= 5,
        lambda p: p['miss'] * 3 + conf(p),
        5, "sgp5_highmiss_game"
    ))

    # --- 6-leg strategies ---
    for min_hc in [5, 8, 10]:
        sgp_strats.append((
            lambda g, mhc=min_hc: g['n_high_conf'] >= mhc,
            lambda p: conf(p) >= 3,
            conf, 6, f"sgp6_hc{min_hc}_conf3"
        ))

    sgp_strats.append((
        lambda g: g['avg_miss'] >= 6,
        lambda p: p['miss'] >= 5,
        lambda p: p['miss'] * 3 + conf(p),
        6, "sgp6_highmiss"
    ))

    # --- 8-leg strategies ---
    for min_hc in [8, 10, 12]:
        sgp_strats.append((
            lambda g, mhc=min_hc: g['n_high_conf'] >= mhc,
            lambda p: conf(p) >= 3,
            conf, 8, f"sgp8_hc{min_hc}"
        ))

    sgp_strats.append((
        lambda g: g['avg_miss'] >= 6 and g['n_high_conf'] >= 8,
        lambda p: True,
        conf, 8, "sgp8_highmiss_hc8"
    ))

    # ================================================================
    # ANALYSIS 4: CROSS-GAME CORRELATED (pick best from multiple games)
    # ================================================================

    # --- Cross-game: pick top N from ALL games that day ---
    def run_xgame(all_props, pick_filter, sort_fn, legs, name):
        by_date = defaultdict(list)
        for p in all_props:
            if pick_filter(p):
                by_date[p['date']].append(p)

        wins = total = lh = lt = cs = ms = 0
        streaks = []; daily = []

        for date in sorted(by_date.keys()):
            cands = by_date[date]
            cands.sort(key=sort_fn, reverse=True)
            sel = []; used = set()
            for p in cands:
                if p['name'] in used: continue
                sel.append(p)
                used.add(p['name'])
                if len(sel) >= legs: break
            if len(sel) < legs: continue

            hit = all(p['under'] for p in sel)
            nh = sum(p['under'] for p in sel)
            total += 1; lt += legs; lh += nh
            if hit: wins += 1; cs += 1; ms = max(ms, cs)
            else:
                if cs > 0: streaks.append(cs)
                cs = 0
            daily.append({'date': date, 'hit': hit, 'nh': nh})

        if cs > 0: streaks.append(cs)
        return {
            'name': name, 'w': wins, 't': total,
            'r': wins/max(total,1), 'lr': lh/max(lt,1),
            'ms': ms, 's5': sum(1 for s in streaks if s >= 5),
            's3': sum(1 for s in streaks if s >= 3),
        }

    xgame_strats = []

    # Cross-game: top N by conf, BLK/STL only
    for legs in [5, 6, 8]:
        xgame_strats.append((
            lambda p: p['stat'] in ['blk','stl'] and conf(p) >= 5,
            conf, legs, f"xg{legs}_blkstl_conf5"
        ))
        xgame_strats.append((
            lambda p: p['stat'] in ['blk','stl'] and conf(p) >= 3,
            conf, legs, f"xg{legs}_blkstl_conf3"
        ))
        xgame_strats.append((
            lambda p: p['stat'] == 'blk' and conf(p) >= 3,
            conf, legs, f"xg{legs}_blk_conf3"
        ))
        # BLK/STL + away
        xgame_strats.append((
            lambda p: p['stat'] in ['blk','stl'] and p.get('away') and conf(p) >= 3,
            conf, legs, f"xg{legs}_blkstl_away_conf3"
        ))
        # Any stat, ultra high conf
        xgame_strats.append((
            lambda p: conf(p) >= 8,
            conf, legs, f"xg{legs}_conf8"
        ))
        xgame_strats.append((
            lambda p: conf(p) >= 10,
            conf, legs, f"xg{legs}_conf10"
        ))
        # High miss count
        xgame_strats.append((
            lambda p: p['miss'] >= 8,
            lambda p: p['miss'] * 3 + conf(p),
            legs, f"xg{legs}_miss8"
        ))
        xgame_strats.append((
            lambda p: p['miss'] >= 7 and p['stat'] in ['blk','stl'],
            lambda p: p['miss'] * 3 + conf(p),
            legs, f"xg{legs}_blkstl_miss7"
        ))

    # ================================================================
    # RUN ALL STRATEGIES
    # ================================================================
    print(f"\n{'='*90}")
    print("SAME-GAME PARLAY RESULTS")
    print(f"{'='*90}")
    print(f"{'#':>3} {'Strategy':<40} {'W/L':>10} {'Rate':>7} {'LegHR':>7} {'MaxStr':>6} {'3+':>4} {'5+':>4}")

    sgp_results = []
    for gf, pf, sf, legs, name in sgp_strats:
        try:
            r = run_sgp(game_signals, gf, pf, sf, legs, name)
            if r['t'] >= 20:
                sgp_results.append(r)
        except Exception as e:
            print(f"  ERR {name}: {e}", file=sys.stderr)

    sgp_results.sort(key=lambda r: r['r'], reverse=True)
    for i, r in enumerate(sgp_results[:30]):
        print(f"{i+1:>3} {r['name']:<40} {r['w']:>3}/{r['t']:<5} {r['r']:>6.1%} {r['lr']:>6.1%} {r['ms']:>5} {r['s3']:>4} {r['s5']:>4}")

    print(f"\n{'='*90}")
    print("CROSS-GAME PARLAY RESULTS (top N across all games)")
    print(f"{'='*90}")
    print(f"{'#':>3} {'Strategy':<40} {'W/L':>10} {'Rate':>7} {'LegHR':>7} {'MaxStr':>6} {'3+':>4} {'5+':>4}")

    xg_results = []
    for pf, sf, legs, name in xgame_strats:
        try:
            r = run_xgame(all_props, pf, sf, legs, name)
            if r['t'] >= 20:
                xg_results.append(r)
        except Exception as e:
            print(f"  ERR {name}: {e}", file=sys.stderr)

    xg_results.sort(key=lambda r: r['r'], reverse=True)
    for i, r in enumerate(xg_results[:40]):
        print(f"{i+1:>3} {r['name']:<40} {r['w']:>3}/{r['t']:<5} {r['r']:>6.1%} {r['lr']:>6.1%} {r['ms']:>5} {r['s3']:>4} {r['s5']:>4}")

    # ================================================================
    # DEEP DIVE: Best strategy yearly breakdown
    # ================================================================
    all_results = sgp_results + xg_results
    if all_results:
        all_results.sort(key=lambda r: r['r'], reverse=True)
        for best in all_results[:5]:
            if 'daily' not in best: continue
            print(f"\n{'='*90}")
            print(f"YEARLY: {best['name']} ({best['r']:.1%})")
            yearly = defaultdict(lambda: [0,0])
            for d in best['daily']:
                yearly[d['date'][:4]][1] += 1
                if d['hit']: yearly[d['date'][:4]][0] += 1
            for y in sorted(yearly.keys()):
                w, t = yearly[y]
                print(f"  {y}: {w:>3}/{t:<3} = {w/max(t,1):.1%}")

    # Save
    out = []
    for r in (sgp_results + xg_results):
        out.append({'name': r['name'], 'w': r['w'], 't': r['t'],
                    'r': round(r['r'],4), 'lr': round(r['lr'],4),
                    'ms': r['ms'], 's3': r['s3'], 's5': r['s5']})
    out.sort(key=lambda x: x['r'], reverse=True)
    path = os.path.join(os.path.dirname(__file__), 'correlated_parlay_results.json')
    with open(path, 'w') as f: json.dump(out, f, indent=2)
    print(f"\nSaved to {path}", file=sys.stderr)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
LONG PARLAY BACKTEST v2: Cracking 5-6 leg parlays.

Key insights from v1 that failed:
- Independent 5-leg at 73% per leg = 20.5% → math kills you
- Same-game correlation exists but we couldn't predict WHICH games

NEW APPROACHES:
1. HYBRID PARLAYS: 2-3 correlated legs from Game A + 2-3 from Game B = 5-6 legs
   with INTRA-game correlation boosting actual hit rate above independent math
2. REAL VEGAS LINES: Use SGO historical spreads/totals to predict game conditions
3. SELECTIVE BETTING: Skip days with bad slates, only bet when conditions are right
4. CONDITIONAL CORRELATION: P(all 5 under | right conditions) >> P(all 5 under)
5. ANTI-CORRELATION EXPLOITATION: Within a team, when one player underperforms,
   the others often do too (team effect), OR the team compensates (anti-correlation)
6. BLOWOUT STACKING: Large spreads → starters sit → ALL starters go under
7. DYNAMIC LEG COUNT: Some days 5 legs, some days 3, adapt to conditions
"""

import csv, json, sys, os
from collections import defaultdict
from datetime import datetime, timedelta
import random

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "NBA Database (1947 - Present)", "PlayerStatistics.csv")
SGO_PATH = os.path.join(os.path.dirname(__file__), "cache", "sgo", "historical_events.json")

MIN_LINES = {'pts': 5, 'reb': 2, 'ast': 1, '3pm': 0.5, 'blk': 0.5, 'stl': 0.5}

def pm(s):
    if not s: return 0
    try:
        if ':' in str(s): p = str(s).split(':'); return float(p[0]) + float(p[1])/60
        return float(s)
    except: return 0


def load_sgo_games():
    """Load real Vegas lines from SGO historical data."""
    if not os.path.exists(SGO_PATH):
        print("No SGO data found", file=sys.stderr)
        return {}
    with open(SGO_PATH) as f:
        data = json.load(f)
    events = data.get('events', [])
    # Index by date + teams
    sgo = {}
    for e in events:
        d = e.get('date', '')
        home = e.get('home', '')
        away = e.get('away', '')
        if d and home and away:
            key = f"{d}_{home}_{away}"
            sgo[key] = {
                'spread': e.get('spread'),         # Home spread (negative = home favored)
                'total': e.get('game_total'),       # Over/under total
                'home_score': e.get('home_score'),
                'away_score': e.get('away_score'),
                'home_ml': e.get('home_ml'),
                'away_ml': e.get('away_ml'),
            }
            # Also index reversed
            sgo[f"{d}_{away}_{home}"] = sgo[key]
    print(f"Loaded {len(events)} SGO events ({len(sgo)} indexed)", file=sys.stderr)
    return sgo


# Standard NBA team abbreviation mappings
TEAM_MAP = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
}


def load_data():
    """Load player games from CSV."""
    print("Loading CSV...", file=sys.stderr)
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
            team_name = row.get('playerteamName', '')
            opp_name = row.get('opponentteamName', '')
            player_games[pid].append({
                'date': d[:10],
                'name': f"{row.get('firstName','')} {row.get('lastName','')}",
                'team': team_name, 'opp': opp_name,
                'team_abr': TEAM_MAP.get(team_name, team_name[:3].upper()),
                'opp_abr': TEAM_MAP.get(opp_name, opp_name[:3].upper()),
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
                'gameId': row.get('gameId', ''),
            })
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g['date'])
    print(f"Loaded {len(player_games)} players", file=sys.stderr)
    return player_games


def compute_all_props(player_games, inflation=4):
    """Generate props with rich features. Group by date AND by game."""
    stats = ['pts', 'reb', 'ast', '3pm', 'blk', 'stl']
    date_props = defaultdict(list)     # date -> [props]
    game_props = defaultdict(list)     # game_key -> [props]

    for pid, games in player_games.items():
        if len(games) < 15: continue
        for i in range(15, len(games)):
            cur = games[i]
            prior = games[:i]

            game_key = f"{cur['date']}_{cur.get('gameId', '')}"

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

                # Mins trend
                m10 = [g['mins'] for g in l10]
                mavg = sum(m10)/10
                m3 = sum(g['mins'] for g in prior[-3:])/3

                # PF trend (foul trouble)
                pf10 = sum(g['pf'] for g in l10)/10

                # Plus/minus trend
                pm10 = sum(g['pm'] for g in l10)/10

                # Win rate L10
                w10 = sum(1 for g in l10 if g.get('win'))/10

                prop = {
                    'date': cur['date'], 'name': cur['name'], 'stat': stat,
                    'team': cur['team'], 'opp': cur['opp'],
                    'team_abr': cur.get('team_abr', ''), 'opp_abr': cur.get('opp_abr', ''),
                    'game_key': game_key,
                    'line': line, 'actual': actual, 'under': actual < line,
                    'home': cur['home'], 'away': not cur['home'] if cur['home'] is not None else None,
                    'avg10': avg10, 'gap': gap, 'hr10': hr10, 'shr': shr,
                    'miss': miss, 'std': std, 'flr': flr,
                    'stk': stk, 'cu': cu, 'rest': rest, 'b2b': rest == 0,
                    'mins': cur['mins'], 'mavg': mavg, 'm3': m3,
                    'pf10': pf10, 'pm10': pm10, 'w10': w10,
                    'pid': pid,
                }
                date_props[cur['date']].append(prop)
                game_props[game_key].append(prop)

    total = sum(len(v) for v in date_props.values())
    print(f"  {total:,} props across {len(date_props)} dates, {len(game_props)} games", file=sys.stderr)
    return date_props, game_props


def conf_score(p):
    """Confidence score for UNDER pick."""
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
    return s


def uber_score(p):
    """Multi-factor uber score for ranking UNDER picks."""
    s = conf_score(p)
    # Minutes stability (less risky if consistent)
    if p['mavg'] >= 28 and abs(p['m3'] - p['mavg']) < 3: s += 0.5
    # Low variance = more predictable
    if p['std'] < p['avg10'] * 0.25: s += 1.0
    elif p['std'] < p['avg10'] * 0.35: s += 0.5
    # Foul trouble for overs
    if p['pf10'] >= 4.0: s += 0.5
    # Plus/minus (losing team = less minutes)
    if p['pm10'] < -3: s += 0.5
    return s


def build_parlay_legs(pool, n_legs, strategy, game_props_map=None, sgo=None):
    """Build an n-leg parlay from pool using specified strategy.

    Returns list of n props, or None if not enough qualifying legs.
    """
    if strategy == 'top_conf':
        # Just pick top N by confidence score (baseline)
        ranked = sorted(pool, key=lambda p: -conf_score(p))
        # Dedupe by player name
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) == n_legs: return picks
        return None

    elif strategy == 'top_uber':
        # Top N by uber score
        ranked = sorted(pool, key=lambda p: -uber_score(p))
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) == n_legs: return picks
        return None

    elif strategy == 'blk_stl_only':
        # Only BLK and STL props
        filtered = [p for p in pool if p['stat'] in ('blk', 'stl')]
        ranked = sorted(filtered, key=lambda p: -uber_score(p))
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) == n_legs: return picks
        return None

    elif strategy == 'hybrid_2game':
        # 2-3 legs from best game + rest from 2nd best game
        if not game_props_map: return None
        # Score each game
        game_scores = {}
        for gk, gprops in game_props_map.items():
            under_props = [p for p in gprops if p in pool]
            if len(under_props) < 2: continue
            # Game quality = average uber score of top picks
            top = sorted(under_props, key=lambda p: -uber_score(p))[:4]
            game_scores[gk] = (sum(uber_score(p) for p in top) / len(top), top)

        if len(game_scores) < 2: return None
        ranked_games = sorted(game_scores.items(), key=lambda x: -x[1][0])

        # Take ceil(n/2) from best game, rest from 2nd
        n1 = (n_legs + 1) // 2  # 3 from best game for 5-leg
        n2 = n_legs - n1

        g1_picks = ranked_games[0][1][1][:n1]
        if len(g1_picks) < n1: return None

        # Dedupe by name from game 1
        names = set(p['name'] for p in g1_picks)
        g2_pool = [p for p in ranked_games[1][1][1] if p['name'] not in names]
        g2_picks = g2_pool[:n2]
        if len(g2_picks) < n2: return None

        return g1_picks + g2_picks

    elif strategy == 'hybrid_3game':
        # 2 legs from each of 2-3 games
        if not game_props_map: return None
        game_scores = {}
        for gk, gprops in game_props_map.items():
            under_props = [p for p in gprops if p in pool]
            if len(under_props) < 2: continue
            top = sorted(under_props, key=lambda p: -uber_score(p))[:3]
            game_scores[gk] = (sum(uber_score(p) for p in top) / len(top), top)

        if len(game_scores) < 2: return None
        ranked_games = sorted(game_scores.items(), key=lambda x: -x[1][0])

        picks = []
        names = set()
        legs_per_game = max(2, n_legs // min(3, len(ranked_games)))

        for gk, (score, top) in ranked_games:
            for p in top:
                if p['name'] not in names:
                    names.add(p['name'])
                    picks.append(p)
                if len([pk for pk in picks if pk['game_key'] == gk]) >= legs_per_game:
                    break
            if len(picks) >= n_legs:
                return picks[:n_legs]
        return None if len(picks) < n_legs else picks[:n_legs]

    elif strategy == 'same_game_blowout':
        # Stack unders from games with large predicted spread (SGO data)
        if not game_props_map or not sgo: return None

        # Find games with large spreads
        blowout_games = []
        for gk, gprops in game_props_map.items():
            if len(gprops) < 3: continue
            # Try to match to SGO
            date = gprops[0]['date']
            team_abr = gprops[0].get('team_abr', '')
            opp_abr = gprops[0].get('opp_abr', '')

            sgo_key1 = f"{date}_{team_abr}_{opp_abr}"
            sgo_key2 = f"{date}_{opp_abr}_{team_abr}"
            sgo_data = sgo.get(sgo_key1) or sgo.get(sgo_key2)

            if sgo_data and sgo_data.get('spread') is not None:
                spread = abs(sgo_data['spread'])
                if spread >= 8:  # Blowout territory
                    under_props = sorted([p for p in gprops if p in pool], key=lambda p: -uber_score(p))
                    if len(under_props) >= 3:
                        blowout_games.append((spread, gk, under_props))

        if not blowout_games: return None
        blowout_games.sort(key=lambda x: -x[0])  # Biggest spread first

        picks = []
        names = set()
        for spread, gk, props in blowout_games:
            for p in props:
                if p['name'] not in names:
                    names.add(p['name'])
                    picks.append(p)
                if len(picks) >= n_legs:
                    return picks[:n_legs]
        return None if len(picks) < n_legs else picks[:n_legs]

    elif strategy == 'same_game_low_total':
        # Stack unders from games with low predicted totals
        if not game_props_map or not sgo: return None

        low_total_games = []
        for gk, gprops in game_props_map.items():
            if len(gprops) < 3: continue
            date = gprops[0]['date']
            team_abr = gprops[0].get('team_abr', '')
            opp_abr = gprops[0].get('opp_abr', '')

            sgo_key1 = f"{date}_{team_abr}_{opp_abr}"
            sgo_key2 = f"{date}_{opp_abr}_{team_abr}"
            sgo_data = sgo.get(sgo_key1) or sgo.get(sgo_key2)

            if sgo_data and sgo_data.get('total') is not None:
                total = sgo_data['total']
                if total <= 215:  # Low-scoring game prediction
                    under_props = sorted([p for p in gprops if p in pool], key=lambda p: -uber_score(p))
                    if len(under_props) >= 3:
                        low_total_games.append((total, gk, under_props))

        if not low_total_games: return None
        low_total_games.sort(key=lambda x: x[0])  # Lowest total first

        picks = []
        names = set()
        for total, gk, props in low_total_games:
            for p in props:
                if p['name'] not in names:
                    names.add(p['name'])
                    picks.append(p)
                if len(picks) >= n_legs:
                    return picks[:n_legs]
        return None if len(picks) < n_legs else picks[:n_legs]

    elif strategy.startswith('sgp_blowout_'):
        # Same-game parlay from single biggest blowout
        if not game_props_map or not sgo: return None
        min_spread = int(strategy.split('_')[-1])

        best = None
        for gk, gprops in game_props_map.items():
            if len(gprops) < n_legs: continue
            date = gprops[0]['date']
            team_abr = gprops[0].get('team_abr', '')
            opp_abr = gprops[0].get('opp_abr', '')
            sgo_key1 = f"{date}_{team_abr}_{opp_abr}"
            sgo_key2 = f"{date}_{opp_abr}_{team_abr}"
            sgo_data = sgo.get(sgo_key1) or sgo.get(sgo_key2)

            if sgo_data and sgo_data.get('spread') is not None:
                spread = abs(sgo_data['spread'])
                if spread >= min_spread:
                    score = spread + sum(uber_score(p) for p in sorted([p for p in gprops if p in pool], key=lambda p: -uber_score(p))[:n_legs]) / n_legs
                    if best is None or score > best[0]:
                        best = (score, gk, [p for p in gprops if p in pool])

        if not best: return None
        ranked = sorted(best[2], key=lambda p: -uber_score(p))
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) == n_legs: return picks
        return None

    elif strategy == 'diverse_stat_same_game':
        # One BLK + one STL + one AST + one REB + one 3PM from same game
        if not game_props_map: return None

        best_game = None
        best_score = -999
        for gk, gprops in game_props_map.items():
            avail_stats = set(p['stat'] for p in gprops if p in pool)
            if len(avail_stats) < n_legs: continue

            # Pick best prop per stat type
            by_stat = defaultdict(list)
            for p in gprops:
                if p in pool:
                    by_stat[p['stat']].append(p)

            stat_order = ['blk', 'stl', '3pm', 'ast', 'reb', 'pts']
            picked = []
            names = set()
            for st in stat_order:
                if st not in by_stat: continue
                best_p = max([p for p in by_stat[st] if p['name'] not in names],
                           key=lambda p: uber_score(p), default=None)
                if best_p:
                    picked.append(best_p)
                    names.add(best_p['name'])
                if len(picked) >= n_legs: break

            if len(picked) >= n_legs:
                total_score = sum(uber_score(p) for p in picked[:n_legs])
                if total_score > best_score:
                    best_score = total_score
                    best_game = picked[:n_legs]

        return best_game

    elif strategy == 'underdog_team_stack':
        # Stack unders from the underdog team (they often underperform)
        if not game_props_map or not sgo: return None

        # Find games where one team is big underdog
        underdog_picks = []
        for gk, gprops in game_props_map.items():
            if len(gprops) < 2: continue
            date = gprops[0]['date']
            team_abr = gprops[0].get('team_abr', '')
            opp_abr = gprops[0].get('opp_abr', '')
            sgo_key1 = f"{date}_{team_abr}_{opp_abr}"
            sgo_key2 = f"{date}_{opp_abr}_{team_abr}"
            sgo_data = sgo.get(sgo_key1) or sgo.get(sgo_key2)

            if not sgo_data or sgo_data.get('spread') is None: continue
            spread = sgo_data['spread']

            # Determine which team is underdog based on spread
            # For our props, figure out which team each player is on
            for p in gprops:
                if p not in pool: continue
                p_team = p.get('team_abr', '')
                is_underdog = False
                # If home team has negative spread (favored), away team is underdog
                if p.get('home') and spread > 5:  # Home team is underdog
                    is_underdog = True
                elif p.get('away') and spread < -5:  # Away team is underdog (home favored)
                    is_underdog = True

                if is_underdog:
                    underdog_picks.append(p)

        if len(underdog_picks) < n_legs: return None
        ranked = sorted(underdog_picks, key=lambda p: -uber_score(p))
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) == n_legs: return picks
        return None

    elif strategy == 'cold_stack':
        # Stack COLD players going under
        cold = [p for p in pool if p['stk'] == 'COLD']
        ranked = sorted(cold, key=lambda p: -uber_score(p))
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) == n_legs: return picks
        return None

    elif strategy == 'floor_below_line':
        # Players whose 2nd-lowest L10 game is below the line
        floor_picks = [p for p in pool if p['flr'] < p['line'] * 0.85]
        ranked = sorted(floor_picks, key=lambda p: -uber_score(p))
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) == n_legs: return picks
        return None

    elif strategy == 'miss_streak':
        # Players on long miss streaks (consecutive unders)
        miss_picks = [p for p in pool if p['cu'] >= 3]
        ranked = sorted(miss_picks, key=lambda p: (-p['cu'], -uber_score(p)))
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) == n_legs: return picks
        return None

    elif strategy == 'low_var_under':
        # Low variance players (predictable) going under
        low_var = [p for p in pool if p['std'] < p['avg10'] * 0.30]
        ranked = sorted(low_var, key=lambda p: -uber_score(p))
        seen = set()
        picks = []
        for p in ranked:
            if p['name'] not in seen:
                seen.add(p['name'])
                picks.append(p)
            if len(picks) == n_legs: return picks
        return None

    return None


def run_backtest():
    """Run all strategies across all dates."""
    player_games = load_data()
    sgo = load_sgo_games()
    date_props, game_props = compute_all_props(player_games)

    # Build per-date game_props index
    date_game_props = defaultdict(lambda: defaultdict(list))
    for gk, props in game_props.items():
        if props:
            d = props[0]['date']
            date_game_props[d][gk] = props

    # STRATEGIES TO TEST
    # Format: (name, n_legs, strategy, min_conf, min_pool, stat_filter, skip_logic)
    strategies = []

    # ============================================================
    # CATEGORY 1: Pure cross-game (enhanced from v1)
    # ============================================================
    for n in [5, 6]:
        for min_c in [3, 5, 7, 9]:
            strategies.append((f'xg{n}_conf{min_c}_uber', n, 'top_uber', min_c, n, None, None))
            strategies.append((f'xg{n}_conf{min_c}_blkstl', n, 'blk_stl_only', min_c, n, ['blk','stl'], None))
        # Cold stacking
        strategies.append((f'xg{n}_cold', n, 'cold_stack', 3, n, None, None))
        # Floor below line
        strategies.append((f'xg{n}_floor', n, 'floor_below_line', 3, n, None, None))
        # Miss streak
        strategies.append((f'xg{n}_miss_streak', n, 'miss_streak', 3, n, None, None))
        # Low variance
        strategies.append((f'xg{n}_low_var', n, 'low_var_under', 3, n, None, None))

    # ============================================================
    # CATEGORY 2: Hybrid (2-3 games)
    # ============================================================
    for n in [5, 6]:
        for min_c in [3, 5, 7]:
            strategies.append((f'hyb2_{n}leg_c{min_c}', n, 'hybrid_2game', min_c, n, None, None))
            strategies.append((f'hyb3_{n}leg_c{min_c}', n, 'hybrid_3game', min_c, n, None, None))

    # ============================================================
    # CATEGORY 3: Real Vegas lines (SGO-powered)
    # ============================================================
    for n in [5, 6]:
        # Blowout stacking
        strategies.append((f'blowout_{n}leg', n, 'same_game_blowout', 3, n, None, None))
        # Low total game stacking
        strategies.append((f'low_total_{n}leg', n, 'same_game_low_total', 3, n, None, None))
        # Underdog team stacking
        strategies.append((f'underdog_{n}leg', n, 'underdog_team_stack', 3, n, None, None))
        # SGP from blowout games
        for ms in [6, 8, 10, 12]:
            strategies.append((f'sgp_blow{ms}_{n}leg', n, f'sgp_blowout_{ms}', 3, n, None, None))
        # Diverse stat same game
        strategies.append((f'diverse_sgp_{n}leg', n, 'diverse_stat_same_game', 3, n, None, None))

    # ============================================================
    # CATEGORY 4: Skip-day conditional strategies
    # ============================================================
    for n in [5, 6]:
        for min_c in [5, 7, 9]:
            # Only bet when avg conf score of top N picks is above threshold
            strategies.append((f'skip_avgconf{min_c}_{n}leg', n, 'top_uber', min_c, n, None, 'avg_conf'))
            strategies.append((f'skip_cold3_{n}leg', n, 'cold_stack', 3, n, None, 'min_cold_3'))
            strategies.append((f'skip_blkstl_c{min_c}_{n}leg', n, 'blk_stl_only', min_c, n, ['blk','stl'], 'min_pool_8'))

    # ============================================================
    # CATEGORY 5: Mixed strategies (BLK/STL core + other stat flex)
    # ============================================================
    for n in [5, 6]:
        strategies.append((f'core_blk_flex_{n}leg', n, 'core_blk_flex', 5, n, None, None))

    print(f"\nRunning {len(strategies)} strategies...", file=sys.stderr)

    results = {}
    dates = sorted(date_props.keys())

    for si, (name, n_legs, strat, min_conf, min_pool, stat_filter, skip_logic) in enumerate(strategies):
        wins = 0
        total = 0
        streaks = []
        cur_streak = 0
        leg_hits = 0
        leg_total = 0
        skipped = 0

        for date in dates:
            props = date_props[date]
            dgp = date_game_props[date]

            # Filter pool: UNDER candidates with minimum confidence
            pool = [p for p in props if conf_score(p) >= min_conf]

            # Stat filter
            if stat_filter:
                pool = [p for p in pool if p['stat'] in stat_filter]

            # Skip logic
            if skip_logic:
                if skip_logic == 'avg_conf':
                    if len(pool) < n_legs:
                        skipped += 1; continue
                    top_scores = sorted([uber_score(p) for p in pool], reverse=True)[:n_legs]
                    if sum(top_scores)/n_legs < min_conf + 2:
                        skipped += 1; continue
                elif skip_logic == 'min_cold_3':
                    cold = [p for p in pool if p['stk'] == 'COLD']
                    if len(cold) < 3:
                        skipped += 1; continue
                elif skip_logic == 'min_pool_8':
                    if len(pool) < max(8, n_legs * 2):
                        skipped += 1; continue

            if len(pool) < n_legs:
                skipped += 1
                continue

            # Build parlay
            if strat == 'core_blk_flex':
                # Special: 2-3 BLK/STL legs + rest from any stat
                blk_pool = [p for p in pool if p['stat'] in ('blk', 'stl')]
                other_pool = [p for p in pool if p['stat'] not in ('blk', 'stl')]
                blk_ranked = sorted(blk_pool, key=lambda p: -uber_score(p))
                other_ranked = sorted(other_pool, key=lambda p: -uber_score(p))

                n_core = min(3, len(blk_ranked), n_legs - 1)
                if n_core < 2: skipped += 1; continue

                picks = []
                names = set()
                for p in blk_ranked:
                    if p['name'] not in names:
                        names.add(p['name'])
                        picks.append(p)
                    if len(picks) == n_core: break

                for p in other_ranked:
                    if p['name'] not in names:
                        names.add(p['name'])
                        picks.append(p)
                    if len(picks) == n_legs: break

                if len(picks) < n_legs: skipped += 1; continue
            else:
                picks = build_parlay_legs(pool, n_legs, strat, dgp, sgo)

            if not picks or len(picks) < n_legs:
                skipped += 1
                continue

            # Grade
            total += 1
            all_hit = all(p['under'] for p in picks)
            hits = sum(1 for p in picks if p['under'])
            leg_hits += hits
            leg_total += len(picks)

            if all_hit:
                wins += 1
                cur_streak += 1
            else:
                if cur_streak > 0:
                    streaks.append(cur_streak)
                cur_streak = 0

        if cur_streak > 0:
            streaks.append(cur_streak)

        if total > 0:
            rate = wins/total
            lr = leg_hits/leg_total if leg_total else 0
            ms = max(streaks) if streaks else 0
            s3 = sum(1 for s in streaks if s >= 3)
            s5 = sum(1 for s in streaks if s >= 5)
            s10 = sum(1 for s in streaks if s >= 10)
            results[name] = {
                'w': wins, 't': total, 'r': rate, 'lr': lr,
                'ms': ms, 's3': s3, 's5': s5, 's10': s10,
                'skipped': skipped, 'legs': n_legs,
            }

        if (si+1) % 20 == 0:
            print(f"  {si+1}/{len(strategies)} done...", file=sys.stderr)

    return results


def main():
    results = run_backtest()

    # Sort by cash rate
    ranked = sorted(results.items(), key=lambda x: -x[1]['r'])

    print(f"\n{'='*120}")
    print(f"LONG PARLAY BACKTEST v2 — {len(results)} strategies")
    print(f"{'='*120}")

    # Print by category
    categories = {
        '5-LEG PARLAYS': [(k,v) for k,v in ranked if v['legs'] == 5],
        '6-LEG PARLAYS': [(k,v) for k,v in ranked if v['legs'] == 6],
    }

    for cat_name, cat_results in categories.items():
        if not cat_results: continue
        print(f"\n{'─'*120}")
        print(f"  {cat_name}")
        print(f"{'─'*120}")
        print(f"  {'Strategy':<40s} {'W':>5s} {'T':>6s} {'Rate':>7s} {'LegHR':>7s} {'MaxStrk':>8s} {'3+Strk':>7s} {'5+Strk':>7s} {'10+Strk':>8s} {'Skip':>6s}")
        print(f"  {'─'*40} {'─'*5} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*6}")

        for name, r in cat_results[:30]:
            print(f"  {name:<40s} {r['w']:>5d} {r['t']:>6d} {r['r']:>6.1%} {r['lr']:>6.1%} {r['ms']:>8d} {r['s3']:>7d} {r['s5']:>7d} {r['s10']:>8d} {r['skipped']:>6d}")

    # Special analysis: What is the ACTUAL conditional parlay rate given correlation?
    print(f"\n{'='*120}")
    print("KEY INSIGHTS")
    print(f"{'='*120}")

    # Best by max streak
    best_streak = max(ranked, key=lambda x: x[1]['ms'])
    print(f"\n  Best max streak: {best_streak[0]} = {best_streak[1]['ms']} days")

    # Best by cash rate
    best_rate_5 = max([(k,v) for k,v in ranked if v['legs'] == 5 and v['t'] >= 100],
                       key=lambda x: x[1]['r'], default=None)
    best_rate_6 = max([(k,v) for k,v in ranked if v['legs'] == 6 and v['t'] >= 100],
                       key=lambda x: x[1]['r'], default=None)

    if best_rate_5:
        print(f"  Best 5-leg (100+ samples): {best_rate_5[0]} = {best_rate_5[1]['r']:.1%} ({best_rate_5[1]['w']}/{best_rate_5[1]['t']})")
    if best_rate_6:
        print(f"  Best 6-leg (100+ samples): {best_rate_6[0]} = {best_rate_6[1]['r']:.1%} ({best_rate_6[1]['w']}/{best_rate_6[1]['t']})")

    # Comparison: theoretical vs actual
    for n in [5, 6]:
        best = max([(k,v) for k,v in ranked if v['legs'] == n and v['t'] >= 100],
                    key=lambda x: x[1]['r'], default=None)
        if best:
            lr = best[1]['lr']
            theoretical = lr ** n
            actual = best[1]['r']
            boost = actual / theoretical if theoretical > 0 else 0
            print(f"\n  {n}-leg: Leg HR={lr:.1%}, Theoretical independent={theoretical:.1%}, Actual={actual:.1%}, Correlation boost={boost:.2f}x")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'long_parlay_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump([{'name': k, **v} for k, v in ranked], f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()

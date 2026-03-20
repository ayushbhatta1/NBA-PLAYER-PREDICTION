#!/usr/bin/env python3
"""
Player Correlation & Matchup Engine

4 analysis functions powered by SGO season box scores (342K records):
1. teammate_impact(player_a, player_b) — WITH/WITHOUT stat deltas
2. opponent_matchup(player, team) — per-team performance vs overall
3. same_game_correlation(player_a, player_b) — Pearson on shared games
4. usage_redistribution(team, player_out) — who benefits when star sits

Used by run_board_v5.py Phase 4c to enrich picks before parlay building.
"""

import json
import os
import sys
import math
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache', 'sgo')
_SGO_PATH = os.path.join(_CACHE_DIR, 'season_box_scores.json')
_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'NBA Database (1947 - Present)', 'PlayerStatistics.csv')

_box_scores = None  # lazy-loaded
_csv_data = None    # lazy-loaded fallback

STAT_MAP = {
    'pts': 'pts', 'reb': 'reb', 'ast': 'ast', '3pm': '3pm',
    'stl': 'stl', 'blk': 'blk', 'min': 'min',
    # combos computed on the fly
    'pra': ['pts', 'reb', 'ast'],
    'pr': ['pts', 'reb'],
    'pa': ['pts', 'ast'],
    'ra': ['reb', 'ast'],
    'stl_blk': ['stl', 'blk'],
}

# CSV column mapping
_CSV_STAT_MAP = {
    'pts': 'points', 'reb': 'reboundsTotal', 'ast': 'assists',
    '3pm': 'threePointersMade', 'stl': 'steals', 'blk': 'blocks',
    'min': 'numMinutes',
}


def _load_box_scores():
    """Load SGO season box scores (342K records). Cached after first call."""
    global _box_scores
    if _box_scores is not None:
        return _box_scores
    if not os.path.exists(_SGO_PATH):
        _box_scores = []
        return _box_scores
    with open(_SGO_PATH) as f:
        _box_scores = json.load(f)
    return _box_scores


def _load_csv_data():
    """Load historical CSV as fallback. Only recent seasons (2023+)."""
    global _csv_data
    if _csv_data is not None:
        return _csv_data
    _csv_data = []
    if not os.path.exists(_CSV_PATH):
        return _csv_data
    import csv
    with open(_CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get('gameDateTimeEst', '')
            if date >= '2023':
                _csv_data.append(row)
    return _csv_data


def _get_stat_value(record, stat, source='sgo'):
    """Extract stat value from a record. Handles combos."""
    stat = stat.lower()
    mapping = STAT_MAP.get(stat, stat)
    if isinstance(mapping, list):
        total = 0
        for s in mapping:
            if source == 'csv':
                val = record.get(_CSV_STAT_MAP.get(s, s), 0)
            else:
                val = record.get(s, 0)
            total += float(val or 0)
        return total
    if source == 'csv':
        col = _CSV_STAT_MAP.get(mapping, mapping)
        return float(record.get(col, 0) or 0)
    return float(record.get(mapping, 0) or 0)


def _normalize_name(name):
    """Normalize player name for matching."""
    return name.strip().lower()


def _game_key(record, source='sgo'):
    """Unique game identifier from a record."""
    if source == 'sgo':
        return f"{record.get('date', '')}_{record.get('game', '')}"
    return f"{record.get('gameDateTimeEst', '')[:10]}_{record.get('gameId', '')}"


def _player_team_from_game(record, source='sgo'):
    """Extract player's team from record."""
    if source == 'sgo':
        game = record.get('game', '')
        # No direct team field in SGO, infer from game string
        # We'll need to match by player name across games
        return game
    return f"{record.get('playerteamCity', '')} {record.get('playerteamName', '')}".strip()


# ═══════════════════════════════════════════════════════════════
# INDEX BUILDERS
# ═══════════════════════════════════════════════════════════════

_player_index = None  # {normalized_name: [records]}
_game_index = None    # {game_key: [records]}
_team_index = None    # {team_abbrev: set(normalized_names)}


_player_team_map = None  # {normalized_name: team_abbrev}


def _build_indices():
    """Build lookup indices from box scores. Called once."""
    global _player_index, _game_index, _team_index, _player_team_map
    if _player_index is not None:
        return

    records = _load_box_scores()
    _player_index = defaultdict(list)
    _game_index = defaultdict(list)
    _team_index = defaultdict(set)

    # Track team frequency per player to infer team membership
    _player_team_freq = defaultdict(lambda: defaultdict(int))

    for rec in records:
        name = _normalize_name(rec.get('player', ''))
        gk = _game_key(rec)
        _player_index[name].append(rec)
        _game_index[gk].append(rec)

        # Track which team abbreviations appear in this player's games
        game = rec.get('game', '')
        if '@' in game:
            away, home = game.split('@', 1)
            _player_team_freq[name][away.strip().upper()] += 1
            _player_team_freq[name][home.strip().upper()] += 1

    # Infer player→team: the team that appears in ALL their games is their team
    # (opponents rotate, but own team is constant)
    _player_team_map = {}
    for name, freq in _player_team_freq.items():
        total_games = len(_player_index[name])
        for team, count in freq.items():
            # If this team appears in >= 80% of player's games, it's their team
            if count >= total_games * 0.8:
                _player_team_map[name] = team
                _team_index[team].add(name)
                break

    # If SGO empty, try CSV
    if not records:
        csv_data = _load_csv_data()
        for rec in csv_data:
            first = rec.get('firstName', '')
            last = rec.get('lastName', '')
            name = _normalize_name(f"{first} {last}")
            gk = _game_key(rec, source='csv')
            _player_index[name].append(rec)
            _game_index[gk].append(rec)
            team_name = rec.get('playerteamName', '').upper()
            if team_name:
                _player_team_map[name] = team_name
                _team_index[team_name].add(name)


def _find_player(name):
    """Find player records. Fuzzy match on last name if exact fails.
    No minimum game threshold — role players with even 1 game get included."""
    _build_indices()
    norm = _normalize_name(name)
    if norm in _player_index:
        return _player_index[norm]
    # Try last-name match — accept any player, not just stars with 5+ games
    last = norm.split()[-1] if norm.split() else norm
    candidates = []
    for key, recs in _player_index.items():
        if key.endswith(last):
            candidates.append((key, recs))
    # Pick the candidate with the most games (most likely correct match)
    if candidates:
        candidates.sort(key=lambda x: len(x[1]), reverse=True)
        return candidates[0][1]
    return []


def _detect_source(records):
    """Detect if records are SGO or CSV format."""
    if not records:
        return 'sgo'
    return 'csv' if 'gameId' in records[0] else 'sgo'


# ═══════════════════════════════════════════════════════════════
# 1. TEAMMATE IMPACT
# ═══════════════════════════════════════════════════════════════

def teammate_impact(player_a, player_b, stats=None):
    """
    When Player B is OUT, how do Player A's stats change?

    Uses game-level data: finds games where both played vs games where B was missing.
    Returns per-stat deltas: {stat: {avg_with, avg_without, delta, games_with, games_without}}
    """
    _build_indices()
    recs_a = _find_player(player_a)
    recs_b = _find_player(player_b)

    if not recs_a or not recs_b:
        return None

    source = _detect_source(recs_a)

    # Get dates where B played (min >= 5 minutes — lowered from 10 to catch role players)
    b_active_games = set()
    for rec in recs_b:
        mins = float(rec.get('min' if source == 'sgo' else 'numMinutes', 0) or 0)
        if mins >= 5:
            b_active_games.add(_game_key(rec, source))

    # Split A's games by B's presence
    stats_to_check = stats or ['pts', 'reb', 'ast', '3pm', 'stl', 'blk']
    with_games = []
    without_games = []

    for rec in recs_a:
        mins = float(rec.get('min' if source == 'sgo' else 'numMinutes', 0) or 0)
        if mins < 5:
            continue
        gk = _game_key(rec, source)
        if gk in b_active_games:
            with_games.append(rec)
        else:
            without_games.append(rec)

    if len(without_games) < 2:
        return None

    result = {}
    for stat in stats_to_check:
        vals_with = [_get_stat_value(r, stat, source) for r in with_games]
        vals_without = [_get_stat_value(r, stat, source) for r in without_games]

        avg_with = sum(vals_with) / len(vals_with) if vals_with else 0
        avg_without = sum(vals_without) / len(vals_without) if vals_without else 0

        result[stat] = {
            'avg_with': round(avg_with, 1),
            'avg_without': round(avg_without, 1),
            'delta': round(avg_without - avg_with, 1),
            'games_with': len(with_games),
            'games_without': len(without_games),
        }

    return result


# ═══════════════════════════════════════════════════════════════
# 2. OPPONENT MATCHUP
# ═══════════════════════════════════════════════════════════════

def opponent_matchup(player, opponent_team, stats=None):
    """
    How does Player perform against a specific team?

    Returns per-stat: {stat: {vs_team_avg, overall_avg, delta, games_vs}}
    """
    _build_indices()
    recs = _find_player(player)
    if not recs:
        return None

    source = _detect_source(recs)
    stats_to_check = stats or ['pts', 'reb', 'ast', '3pm', 'stl', 'blk']
    opp = opponent_team.strip().upper()

    # Find games vs this opponent
    vs_games = []
    all_games = []

    for rec in recs:
        mins = float(rec.get('min' if source == 'sgo' else 'numMinutes', 0) or 0)
        if mins < 5:  # lowered from 10 — role players get props too
            continue
        all_games.append(rec)

        if source == 'sgo':
            game = rec.get('game', '').upper()
            if opp in game:
                vs_games.append(rec)
        else:
            opp_city = rec.get('opponentteamCity', '').upper()
            opp_name = rec.get('opponentteamName', '').upper()
            if opp in opp_city or opp in opp_name:
                vs_games.append(rec)

    if not vs_games:
        return None

    result = {}
    for stat in stats_to_check:
        vals_vs = [_get_stat_value(r, stat, source) for r in vs_games]
        vals_all = [_get_stat_value(r, stat, source) for r in all_games]

        avg_vs = sum(vals_vs) / len(vals_vs) if vals_vs else 0
        avg_all = sum(vals_all) / len(vals_all) if vals_all else 0

        result[stat] = {
            'vs_team_avg': round(avg_vs, 1),
            'overall_avg': round(avg_all, 1),
            'delta': round(avg_vs - avg_all, 1),
            'games_vs': len(vs_games),
        }

    return result


# ═══════════════════════════════════════════════════════════════
# 3. SAME-GAME CORRELATION
# ═══════════════════════════════════════════════════════════════

def same_game_correlation(player_a, player_b, stats=None):
    """
    When A scores high, does B score high or low?

    Pearson correlation on pts/reb/ast in shared games.
    Returns {stat: {correlation, shared_games, interpretation}}
    """
    _build_indices()
    recs_a = _find_player(player_a)
    recs_b = _find_player(player_b)

    if not recs_a or not recs_b:
        return None

    source = _detect_source(recs_a)
    stats_to_check = stats or ['pts', 'reb', 'ast']

    # Index B's records by game_key
    b_by_game = {}
    for rec in recs_b:
        mins = float(rec.get('min' if source == 'sgo' else 'numMinutes', 0) or 0)
        if mins >= 5:  # lowered from 10 for role players
            b_by_game[_game_key(rec, source)] = rec

    # Find shared games
    pairs = []
    for rec in recs_a:
        mins = float(rec.get('min' if source == 'sgo' else 'numMinutes', 0) or 0)
        if mins < 5:
            continue
        gk = _game_key(rec, source)
        if gk in b_by_game:
            pairs.append((rec, b_by_game[gk]))

    if len(pairs) < 3:  # lowered from 5 — role players have fewer shared games
        return None

    result = {}
    for stat in stats_to_check:
        a_vals = [_get_stat_value(p[0], stat, source) for p in pairs]
        b_vals = [_get_stat_value(p[1], stat, source) for p in pairs]

        corr = _pearson(a_vals, b_vals)
        if corr is None:
            continue

        if corr > 0.3:
            interp = 'POSITIVE'
        elif corr < -0.3:
            interp = 'NEGATIVE'
        else:
            interp = 'WEAK'

        result[stat] = {
            'correlation': round(corr, 3),
            'shared_games': len(pairs),
            'interpretation': interp,
        }

    return result


def _pearson(x, y):
    """Pearson correlation coefficient. Returns None if insufficient variance."""
    n = len(x)
    if n < 3:  # lowered from 5 for role player coverage
        return None
    mx = sum(x) / n
    my = sum(y) / n
    dx = [xi - mx for xi in x]
    dy = [yi - my for yi in y]
    num = sum(a * b for a, b in zip(dx, dy))
    den_x = math.sqrt(sum(a * a for a in dx))
    den_y = math.sqrt(sum(b * b for b in dy))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


# ═══════════════════════════════════════════════════════════════
# 4. USAGE REDISTRIBUTION
# ═══════════════════════════════════════════════════════════════

def usage_redistribution(team, player_out, stats=None):
    """
    When a star sits, which teammates benefit most?

    Finds games where player_out was absent, compares teammates' stats.
    Returns list of {player, stat, avg_with, avg_without, delta} sorted by biggest delta.
    """
    _build_indices()
    recs_out = _find_player(player_out)
    if not recs_out:
        return None

    source = _detect_source(recs_out)
    stats_to_check = stats or ['pts', 'reb', 'ast']
    team_upper = team.strip().upper()

    # Games where player_out was active vs absent
    out_active_games = set()
    for rec in recs_out:
        mins = float(rec.get('min' if source == 'sgo' else 'numMinutes', 0) or 0)
        gk = _game_key(rec, source)
        if mins >= 10:
            out_active_games.add(gk)

    # Find teammates — only players on the SAME team
    teammates = set()
    for name, t in _player_team_map.items():
        if t == team_upper and name != _normalize_name(player_out):
            teammates.add(name)

    if not teammates:
        return None

    results = []
    for tm_name in teammates:
        tm_recs = _player_index.get(tm_name, [])
        if len(tm_recs) < 2:  # lowered from 5 — include role players
            continue

        for stat in stats_to_check:
            vals_with = []
            vals_without = []
            for rec in tm_recs:
                mins = float(rec.get('min' if source == 'sgo' else 'numMinutes', 0) or 0)
                if mins < 5:  # lowered from 10
                    continue
                gk = _game_key(rec, source)
                val = _get_stat_value(rec, stat, source)
                if gk in out_active_games:
                    vals_with.append(val)
                else:
                    vals_without.append(val)

            if len(vals_without) < 2 or not vals_with:
                continue

            avg_with = sum(vals_with) / len(vals_with)
            avg_without = sum(vals_without) / len(vals_without)
            delta = avg_without - avg_with

            if abs(delta) >= 1.0:  # only report meaningful bumps
                results.append({
                    'player': tm_name.title(),
                    'stat': stat,
                    'avg_with': round(avg_with, 1),
                    'avg_without': round(avg_without, 1),
                    'delta': round(delta, 1),
                    'games_without': len(vals_without),
                })

    results.sort(key=lambda x: abs(x['delta']), reverse=True)
    return results[:20]


# ═══════════════════════════════════════════════════════════════
# 5. TEAM-VS-TEAM MATCHUP
# ═══════════════════════════════════════════════════════════════

def team_vs_team_matchup(team_a, team_b):
    """
    How does Team A's entire roster perform against Team B?
    Uses SGO box scores + _player_team_map and _team_index.

    Returns: {team_delta, games_vs, player_count, player_deltas: [{player, stat, delta, ...}]}
    """
    _build_indices()
    team_a_upper = team_a.strip().upper()
    team_b_upper = team_b.strip().upper()

    team_a_players = _team_index.get(team_a_upper, set())
    if not team_a_players:
        return None

    player_deltas = []
    games_vs = 0

    for player_name in team_a_players:
        matchup = opponent_matchup(player_name.title(), team_b_upper,
                                   stats=['pts', 'reb', 'ast', '3pm', 'stl', 'blk'])
        if not matchup:
            continue
        for stat_key, data in matchup.items():
            gv = data.get('games_vs', 0)
            if gv >= 2:
                games_vs = max(games_vs, gv)
                player_deltas.append({
                    'player': player_name.title(),
                    'stat': stat_key,
                    'delta': data['delta'],
                    'vs_avg': data['vs_team_avg'],
                    'overall_avg': data['overall_avg'],
                    'games_vs': gv,
                })

    if not player_deltas:
        return None

    player_deltas.sort(key=lambda x: abs(x['delta']), reverse=True)
    # Team-level delta: average pts delta across players
    pts_deltas = [d['delta'] for d in player_deltas if d['stat'] == 'pts']
    team_delta = round(sum(pts_deltas) / len(pts_deltas), 1) if pts_deltas else 0

    return {
        'team_delta': team_delta,
        'games_vs': games_vs,
        'player_count': len(set(d['player'] for d in player_deltas)),
        'player_deltas': player_deltas[:15],
    }


# ═══════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════

def enrich_picks(results, GAMES=None):
    """
    Enrich a list of pick dicts with correlation data.
    Called by run_board_v5.py Phase 4c.

    Adds to each pick:
      - opp_matchup_delta: delta vs opponent for this stat
      - opp_matchup_games: number of games vs this opponent
      - teammate_correlations: list of {player, correlation} for same-game players
      - max_same_game_corr: highest |correlation| with any same-game pick
      - usage_boost: delta when an injured teammate is out (from injury context)
    """
    _build_indices()

    if not _player_index:
        return 0

    # Group picks by game for same-game correlation
    by_game = defaultdict(list)
    for pick in results:
        game = pick.get('game', '')
        if game:
            by_game[game].append(pick)

    # Build set of injured/OUT players from picks for usage boost
    out_players = set()
    player_teams = {}
    for pick in results:
        status = (pick.get('player_injury_status') or '').upper()
        if 'OUT' in status:
            out_players.add(pick.get('player', ''))
        # Track player→team mapping from pick data
        player = pick.get('player', '')
        game = pick.get('game', '')
        if game and '@' in game:
            away, home = game.split('@', 1)
            is_home = pick.get('is_home')
            if is_home is True:
                player_teams[player] = home.strip().upper()
            elif is_home is False:
                player_teams[player] = away.strip().upper()

    # Pre-compute usage boosts for OUT players' teammates
    _usage_cache = {}
    for out_player in out_players:
        team = player_teams.get(out_player)
        if team:
            key = f"{team}_{out_player}"
            redistribution = usage_redistribution(team, out_player)
            if redistribution:
                _usage_cache[key] = {
                    (r['player'].lower(), r['stat']): r['delta']
                    for r in redistribution
                }

    enriched = 0

    for pick in results:
        player = pick.get('player', '')
        stat = pick.get('stat', '').lower()
        game = pick.get('game', '')
        got_data = False

        # 1. Opponent matchup delta — for this player's actual prop stat
        if game and '@' in game:
            away, home = game.split('@', 1)
            is_home = pick.get('is_home')
            opp_team = away if is_home else home

            # Check the actual stat on the line, plus base stats for combos
            stat_check = stat
            matchup = opponent_matchup(player, opp_team, stats=[stat_check])
            if matchup and stat_check in matchup:
                pick['opp_matchup_delta'] = matchup[stat_check]['delta']
                pick['opp_matchup_games'] = matchup[stat_check]['games_vs']
                got_data = True

        # 2. Same-game correlations with other picks on the board
        game_picks = by_game.get(game, [])
        corr_list = []
        for other in game_picks:
            other_player = other.get('player', '')
            if other_player == player:
                continue
            # Correlate on the actual stat, not just pts
            corr_stat = stat if stat in ('pts', 'reb', 'ast', '3pm', 'stl', 'blk') else 'pts'
            corr = same_game_correlation(player, other_player, stats=[corr_stat])
            if corr and corr_stat in corr:
                corr_list.append({
                    'player': other_player,
                    'stat': corr_stat,
                    'correlation': corr[corr_stat]['correlation'],
                    'shared_games': corr[corr_stat]['shared_games'],
                    'interpretation': corr[corr_stat]['interpretation'],
                })

        if corr_list:
            pick['teammate_correlations'] = corr_list
            pick['max_same_game_corr'] = max(abs(c['correlation']) for c in corr_list)
            got_data = True

        # 3. Usage boost from injured teammates
        my_team = player_teams.get(player)
        if my_team:
            total_boost = 0
            boost_sources = []
            for out_player in out_players:
                out_team = player_teams.get(out_player)
                if out_team == my_team and out_player != player:
                    key = f"{out_team}_{out_player}"
                    cache = _usage_cache.get(key, {})
                    delta = cache.get((player.lower(), stat), 0)
                    if delta != 0:
                        total_boost += delta
                        boost_sources.append({'out_player': out_player, 'delta': delta})
            if total_boost != 0:
                pick['usage_boost'] = round(total_boost, 1)
                pick['usage_boost_sources'] = boost_sources
                got_data = True

        if got_data:
            enriched += 1

    # 4. Game-level context from GAMES dict
    if GAMES:
        for pick in results:
            game = pick.get('game', '')
            if not game or '@' not in game:
                continue

            game_data = GAMES.get(game)
            if not game_data:
                continue

            is_home = pick.get('is_home')
            stat = pick.get('stat', '').lower()

            # 4a: Opponent offensive pressure
            opp_form = game_data.get('away_form' if is_home else 'home_form', {})
            opp_scored = (opp_form.get('avg_scored', 0) or 0) if isinstance(opp_form, dict) else 0
            if opp_scored > 115:
                pick['opp_off_pressure'] = 1
            elif opp_scored < 105:
                pick['opp_off_pressure'] = -1
            else:
                pick['opp_off_pressure'] = 0

            # 4b: Game total signal (vs 225 league avg)
            gt = game_data.get('over_under') or 0
            if not gt:
                pace = game_data.get('pace', {})
                gt = (pace.get('projected_total', 0) or 0) if isinstance(pace, dict) else 0
            if gt:
                pick['game_total_signal'] = round(gt - 225, 1)

            # 4c: Team vs team matchup delta
            parts = game.split('@')
            away_team = parts[0].strip().upper()
            home_team = parts[1].strip().upper() if len(parts) > 1 else ''
            player_team = home_team if is_home else away_team
            opp_team = away_team if is_home else home_team

            if player_team and opp_team:
                tvt = team_vs_team_matchup(player_team, opp_team)
                if tvt:
                    pick['team_vs_opp_delta'] = tvt['team_delta']
                    # Find this specific player's delta vs opponent
                    player_norm = _normalize_name(pick.get('player', ''))
                    for pd_item in tvt.get('player_deltas', []):
                        if _normalize_name(pd_item['player']) == player_norm and pd_item['stat'] == stat:
                            pick['player_vs_team_delta'] = pd_item['delta']
                            break

    return enriched


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _print_result(title, data):
    """Pretty-print analysis results."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if data is None:
        print("  No data found.")
        return
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(val, dict):
                parts = [f"{k}={v}" for k, v in val.items()]
                print(f"  {key:6s}: {', '.join(parts)}")
            else:
                print(f"  {key}: {val}")
    elif isinstance(data, list):
        for item in data[:15]:
            if isinstance(item, dict):
                parts = [f"{k}={v}" for k, v in item.items()]
                print(f"  {', '.join(parts)}")
            else:
                print(f"  {item}")


def main():
    """
    CLI usage:
      python3 correlations.py "Jayson Tatum" "Jaylen Brown"          # teammate impact
      python3 correlations.py --matchup "Nikola Jokic" "GSW"         # opponent matchup
      python3 correlations.py --correlation "LeBron James" "AD"       # same-game correlation
      python3 correlations.py --usage "BOS" "Jaylen Brown"            # usage redistribution
    """
    args = sys.argv[1:]

    if not args or args[0] in ('-h', '--help'):
        print(main.__doc__)
        return

    # Count records available
    records = _load_box_scores()
    print(f"  Loaded {len(records):,} SGO box score records")

    if args[0] == '--team-vs-team' and len(args) >= 3:
        team_a, team_b = args[1], args[2]
        result = team_vs_team_matchup(team_a, team_b)
        _print_result(f"TEAM-VS-TEAM: {team_a} vs {team_b}", result)
        if result:
            print(f"\n  Team delta: {result['team_delta']} pts | {result['player_count']} players | {result['games_vs']} games")
            for pd_item in result.get('player_deltas', [])[:10]:
                print(f"    {pd_item['player']:20s} {pd_item['stat']:5s} delta={pd_item['delta']:+.1f} ({pd_item['games_vs']} games)")
        return

    elif args[0] == '--matchup' and len(args) >= 3:
        player, team = args[1], args[2]
        result = opponent_matchup(player, team)
        _print_result(f"OPPONENT MATCHUP: {player} vs {team}", result)

    elif args[0] == '--correlation' and len(args) >= 3:
        p_a, p_b = args[1], args[2]
        result = same_game_correlation(p_a, p_b)
        _print_result(f"SAME-GAME CORRELATION: {p_a} ↔ {p_b}", result)

    elif args[0] == '--usage' and len(args) >= 3:
        team, player_out = args[1], args[2]
        result = usage_redistribution(team, player_out)
        _print_result(f"USAGE REDISTRIBUTION: {team} without {player_out}", result)

    elif len(args) >= 2 and not args[0].startswith('--'):
        p_a, p_b = args[0], args[1]
        # Default: teammate impact
        result = teammate_impact(p_a, p_b)
        _print_result(f"TEAMMATE IMPACT: {p_a} when {p_b} is OUT", result)

        # Also show correlation
        corr = same_game_correlation(p_a, p_b)
        _print_result(f"SAME-GAME CORRELATION: {p_a} ↔ {p_b}", corr)
    else:
        print("Usage: python3 correlations.py <player_a> <player_b>")
        print("       python3 correlations.py --matchup <player> <team>")
        print("       python3 correlations.py --correlation <player_a> <player_b>")
        print("       python3 correlations.py --usage <team> <player_out>")


if __name__ == '__main__':
    main()

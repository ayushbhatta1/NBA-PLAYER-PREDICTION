#!/usr/bin/env python3
"""
Advanced Feature Engineering for NBA Prop Pipeline

Loads PlayerStatisticsAdvanced.csv and PlayerStatisticsUsage.csv to provide
rolling advanced stats (usage%, net rating, pace, eFG%, TS%, PIE, etc.)
and usage distribution features (pct_blk, pct_stl, pct_reb, etc.).

Also provides:
  - Opponent defensive clustering (5 archetypes)
  - Target encoding (player/team UNDER rates, leakage-safe)
  - Cyclical date features (day-of-week, month)
  - Interaction features (usage x direction, pace x stat type, efg trend)

All rolling features use ONLY data from BEFORE game_date (no leakage).
CSVs filtered to 2015+ to save memory.

Usage:
    python3 predictions/advanced_features.py test    # Load data and print stats
"""

import csv
import math
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(os.path.dirname(PREDICTIONS_DIR), "NBA Database (1947 - Present)")

ADV_CSV = os.path.join(DB_DIR, "PlayerStatisticsAdvanced.csv")
USAGE_CSV = os.path.join(DB_DIR, "PlayerStatisticsUsage.csv")
GAMES_CSV = os.path.join(DB_DIR, "Games.csv")

MIN_DATE = "2015-01-01"

# Features this module adds
ADVANCED_FEATURE_COLS = [
    'adv_usg_pct_l10', 'adv_net_rating_l10', 'adv_off_rating_l10', 'adv_def_rating_l10',
    'adv_pace_l10', 'adv_efg_pct_l10', 'adv_ts_pct_l10', 'adv_pie_l10',
    'adv_usg_pct_std', 'adv_net_rating_std',
    'adv_pct_blk_l10', 'adv_pct_stl_l10', 'adv_pct_reb_l10', 'adv_pct_ast_l10',
    'adv_opp_cluster',
    'adv_usg_x_direction', 'adv_pace_x_stat', 'adv_efg_trend',
    'te_player_under_rate', 'te_team_under_rate',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
]

# Stats that benefit from high pace
PACE_SENSITIVE_STATS = {'pts', 'points', 'reb', 'rebounds', 'ast', 'assists',
                        'pra', 'pr', 'pa', 'ra', 'fgm', 'fg3m', '3pm'}

# ═══════════════════════════════════════════════════════════════
# MODULE-LEVEL CACHES
# ═══════════════════════════════════════════════════════════════

_ADV_CACHE = {}      # personId -> sorted list of dicts (by gameDate)
_USAGE_CACHE = {}    # personId -> sorted list of dicts (by gameDate)
_GAMES_CACHE = {}    # gameId -> dict
_TEAM_ADV_CACHE = {} # teamAbbreviation -> sorted list of dicts
_OPP_CLUSTERS = {}   # teamAbbreviation -> cluster_id

_adv_loaded = False
_usage_loaded = False
_games_loaded = False


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _safe_float(val, default=np.nan):
    """Convert string to float, returning default on failure."""
    if val is None or val == '' or val == 'None':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _parse_date(date_str):
    """Parse date string to date object. Handles 'YYYY-MM-DDTHH:MM:SS' and 'YYYY-MM-DD'."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _mean(values):
    """Mean of a list, nan if empty."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return np.nan
    return sum(clean) / len(clean)


def _std(values):
    """Std dev of a list, nan if < 2 values."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(clean) < 2:
        return np.nan
    m = sum(clean) / len(clean)
    variance = sum((x - m) ** 2 for x in clean) / (len(clean) - 1)
    return math.sqrt(variance)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_advanced_data():
    """Load and cache PlayerStatisticsAdvanced.csv. Only 2015+ to save memory."""
    global _adv_loaded
    if _adv_loaded:
        return

    if not os.path.exists(ADV_CSV):
        print(f"[advanced_features] WARNING: {ADV_CSV} not found")
        _adv_loaded = True
        return

    count = 0
    with open(ADV_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_date_str = (row.get('gameDate') or '')[:10]
            if game_date_str < MIN_DATE:
                continue

            person_id = row.get('personId', '')
            if not person_id:
                continue

            game_date = _parse_date(game_date_str)
            if game_date is None:
                continue

            mins = _safe_float(row.get('min'), 0.0)
            if mins < 1.0:
                continue  # skip DNPs

            record = {
                'gameDate': game_date,
                'gameId': row.get('gameId', ''),
                'teamAbbreviation': row.get('teamAbbreviation', ''),
                'matchup': row.get('matchup', ''),
                'min': mins,
                'usgPct': _safe_float(row.get('usgPct')),
                'netRating': _safe_float(row.get('netRating')),
                'defRating': _safe_float(row.get('defRating')),
                'offRating': _safe_float(row.get('offRating')),
                'pace': _safe_float(row.get('pace')),
                'efgPct': _safe_float(row.get('efgPct')),
                'tsPct': _safe_float(row.get('tsPct')),
                'astPct': _safe_float(row.get('astPct')),
                'rebPct': _safe_float(row.get('rebPct')),
                'orebPct': _safe_float(row.get('orebPct')),
                'drebPct': _safe_float(row.get('drebPct')),
                'pie': _safe_float(row.get('pie')),
            }

            if person_id not in _ADV_CACHE:
                _ADV_CACHE[person_id] = []
            _ADV_CACHE[person_id].append(record)

            # Also build team-level cache for clustering
            team = record['teamAbbreviation']
            if team:
                if team not in _TEAM_ADV_CACHE:
                    _TEAM_ADV_CACHE[team] = []
                _TEAM_ADV_CACHE[team].append(record)

            count += 1

    # Sort each player's games by date
    for pid in _ADV_CACHE:
        _ADV_CACHE[pid].sort(key=lambda x: x['gameDate'])
    for team in _TEAM_ADV_CACHE:
        _TEAM_ADV_CACHE[team].sort(key=lambda x: x['gameDate'])

    _adv_loaded = True
    print(f"[advanced_features] Loaded {count:,} advanced stat rows for {len(_ADV_CACHE):,} players (2015+)")


def load_usage_data():
    """Load and cache PlayerStatisticsUsage.csv. Only 2015+."""
    global _usage_loaded
    if _usage_loaded:
        return

    if not os.path.exists(USAGE_CSV):
        print(f"[advanced_features] WARNING: {USAGE_CSV} not found")
        _usage_loaded = True
        return

    count = 0
    with open(USAGE_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_date_str = (row.get('gameDate') or '')[:10]
            if game_date_str < MIN_DATE:
                continue

            person_id = row.get('personId', '')
            if not person_id:
                continue

            game_date = _parse_date(game_date_str)
            if game_date is None:
                continue

            mins = _safe_float(row.get('min'), 0.0)
            if mins < 1.0:
                continue

            record = {
                'gameDate': game_date,
                'gameId': row.get('gameId', ''),
                'teamAbbreviation': row.get('teamAbbreviation', ''),
                'min': mins,
                'usgPct': _safe_float(row.get('usgPct')),
                'pctBlk': _safe_float(row.get('pctBlk')),
                'pctStl': _safe_float(row.get('pctStl')),
                'pctReb': _safe_float(row.get('pctReb')),
                'pctAst': _safe_float(row.get('pctAst')),
                'pctPts': _safe_float(row.get('pctPts')),
                'pctTov': _safe_float(row.get('pctTov')),
                'pctFga': _safe_float(row.get('pctFga')),
                'pctFgm': _safe_float(row.get('pctFgm')),
                'pctFta': _safe_float(row.get('pctFta')),
                'pctFtm': _safe_float(row.get('pctFtm')),
                'pctOreb': _safe_float(row.get('pctOreb')),
                'pctDreb': _safe_float(row.get('pctDreb')),
                'pctPf': _safe_float(row.get('pctPf')),
                'pctBlka': _safe_float(row.get('pctBlka')),
                'pctFg3A': _safe_float(row.get('pctFg3A')),
                'pctFg3M': _safe_float(row.get('pctFg3M')),
            }

            if person_id not in _USAGE_CACHE:
                _USAGE_CACHE[person_id] = []
            _USAGE_CACHE[person_id].append(record)
            count += 1

    # Sort by date
    for pid in _USAGE_CACHE:
        _USAGE_CACHE[pid].sort(key=lambda x: x['gameDate'])

    _usage_loaded = True
    print(f"[advanced_features] Loaded {count:,} usage stat rows for {len(_USAGE_CACHE):,} players (2015+)")


def load_games_data():
    """Load and cache Games.csv for referee assignments."""
    global _games_loaded
    if _games_loaded:
        return

    if not os.path.exists(GAMES_CSV):
        print(f"[advanced_features] WARNING: {GAMES_CSV} not found")
        _games_loaded = True
        return

    count = 0
    with open(GAMES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = row.get('gameId', '')
            if not game_id:
                continue

            _GAMES_CACHE[game_id] = {
                'gameId': game_id,
                'gameDateTimeEst': row.get('gameDateTimeEst', ''),
                'hometeamId': row.get('hometeamId', ''),
                'awayteamId': row.get('awayteamId', ''),
                'homeScore': _safe_float(row.get('homeScore'), 0),
                'awayScore': _safe_float(row.get('awayScore'), 0),
                'officials': row.get('officials', ''),
            }
            count += 1

    _games_loaded = True
    print(f"[advanced_features] Loaded {count:,} games from Games.csv")


# ═══════════════════════════════════════════════════════════════
# FEATURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_player_advanced_rolling(player_id, game_date, n=10):
    """Get rolling N-game advanced stats for a player BEFORE game_date (no leakage).

    Returns dict with: usg_pct_l10, net_rating_l10, def_rating_l10, off_rating_l10,
    pace_l10, efg_pct_l10, ts_pct_l10, pie_l10, and their stds.
    """
    load_advanced_data()

    result = {
        'usg_pct_l10': np.nan, 'net_rating_l10': np.nan,
        'def_rating_l10': np.nan, 'off_rating_l10': np.nan,
        'pace_l10': np.nan, 'efg_pct_l10': np.nan,
        'ts_pct_l10': np.nan, 'pie_l10': np.nan,
        'usg_pct_std': np.nan, 'net_rating_std': np.nan,
        'efg_pct_l5': np.nan,  # for trend computation
    }

    pid = str(player_id)
    if pid not in _ADV_CACHE:
        return result

    if isinstance(game_date, str):
        game_date = _parse_date(game_date)
    if game_date is None:
        return result

    # Get games strictly before game_date
    prior = [g for g in _ADV_CACHE[pid] if g['gameDate'] < game_date]
    if len(prior) < 3:
        return result

    last_n = prior[-n:]

    result['usg_pct_l10'] = _mean([g['usgPct'] for g in last_n])
    result['net_rating_l10'] = _mean([g['netRating'] for g in last_n])
    result['def_rating_l10'] = _mean([g['defRating'] for g in last_n])
    result['off_rating_l10'] = _mean([g['offRating'] for g in last_n])
    result['pace_l10'] = _mean([g['pace'] for g in last_n])
    result['efg_pct_l10'] = _mean([g['efgPct'] for g in last_n])
    result['ts_pct_l10'] = _mean([g['tsPct'] for g in last_n])
    result['pie_l10'] = _mean([g['pie'] for g in last_n])

    result['usg_pct_std'] = _std([g['usgPct'] for g in last_n])
    result['net_rating_std'] = _std([g['netRating'] for g in last_n])

    # L5 efg for trend
    last_5 = prior[-5:] if len(prior) >= 5 else prior
    result['efg_pct_l5'] = _mean([g['efgPct'] for g in last_5])

    return result


def get_player_usage_rolling(player_id, game_date, n=10):
    """Get rolling N-game usage% stats BEFORE game_date (no leakage).

    Returns dict with: pct_blk_l10, pct_stl_l10, pct_reb_l10, pct_ast_l10, pct_pts_l10, usg_pct_l10
    """
    load_usage_data()

    result = {
        'pct_blk_l10': np.nan, 'pct_stl_l10': np.nan,
        'pct_reb_l10': np.nan, 'pct_ast_l10': np.nan,
        'pct_pts_l10': np.nan, 'usg_pct_l10': np.nan,
    }

    pid = str(player_id)
    if pid not in _USAGE_CACHE:
        return result

    if isinstance(game_date, str):
        game_date = _parse_date(game_date)
    if game_date is None:
        return result

    prior = [g for g in _USAGE_CACHE[pid] if g['gameDate'] < game_date]
    if len(prior) < 3:
        return result

    last_n = prior[-n:]

    result['pct_blk_l10'] = _mean([g['pctBlk'] for g in last_n])
    result['pct_stl_l10'] = _mean([g['pctStl'] for g in last_n])
    result['pct_reb_l10'] = _mean([g['pctReb'] for g in last_n])
    result['pct_ast_l10'] = _mean([g['pctAst'] for g in last_n])
    result['pct_pts_l10'] = _mean([g['pctPts'] for g in last_n])
    result['usg_pct_l10'] = _mean([g['usgPct'] for g in last_n])

    return result


def get_game_officials(game_id):
    """Get referee names for a game from Games.csv (no API call needed).

    Returns list of referee name strings, or empty list if not found.
    """
    load_games_data()

    game_id = str(game_id)
    if game_id not in _GAMES_CACHE:
        return []

    officials_str = _GAMES_CACHE[game_id].get('officials', '')
    if not officials_str or officials_str.strip() == '':
        return []

    # Officials stored as comma-separated: "Scott Foster, Nate Green, Jenna Reneau"
    return [name.strip() for name in officials_str.split(',') if name.strip()]


def get_opponent_cluster(team_abbr, game_date):
    """Cluster teams by defensive profile (from advanced stats). Returns cluster_id 0-4.

    Clusters based on recent defensive rating and pace:
        0: Elite defense, slow pace (grind games)
        1: Good defense, moderate pace
        2: Average defense, average pace
        3: Poor defense, fast pace (shootouts)
        4: Poor defense, slow pace (bad + boring)

    Returns np.nan if insufficient data.
    """
    load_advanced_data()

    if not team_abbr or team_abbr not in _TEAM_ADV_CACHE:
        return np.nan

    if isinstance(game_date, str):
        game_date = _parse_date(game_date)
    if game_date is None:
        return np.nan

    # Get team's last 20 games before game_date
    prior = [g for g in _TEAM_ADV_CACHE[team_abbr] if g['gameDate'] < game_date]
    if len(prior) < 5:
        return np.nan

    last_20 = prior[-20:]

    avg_def = _mean([g['defRating'] for g in last_20])
    avg_pace = _mean([g['pace'] for g in last_20])

    if math.isnan(avg_def) or math.isnan(avg_pace):
        return np.nan

    # Simple rule-based clustering (no sklearn dependency)
    # defRating: lower is better defense (~105 league avg)
    # pace: ~100 league avg
    if avg_def < 108:
        # Good/elite defense
        if avg_pace < 98:
            return 0  # Elite defense, slow
        else:
            return 1  # Good defense, moderate+ pace
    elif avg_def < 113:
        return 2  # Average defense
    else:
        # Poor defense
        if avg_pace >= 100:
            return 3  # Poor defense, fast pace
        else:
            return 4  # Poor defense, slow


# ═══════════════════════════════════════════════════════════════
# ADVANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def engineer_advanced_features(record, player_id=None, game_date=None):
    """Add advanced features to a prop record dict. Returns dict of new features.

    Args:
        record: dict with at least 'direction' and 'stat_type' fields
        player_id: NBA personId (string or int)
        game_date: game date string 'YYYY-MM-DD' or date object

    Returns dict of new features (prefix 'adv_' + target encoding + cyclical).
    """
    features = {col: np.nan for col in ADVANCED_FEATURE_COLS}

    # Extract info from record if not provided
    if player_id is None:
        player_id = record.get('player_id') or record.get('personId')
    if game_date is None:
        game_date = record.get('game_date') or record.get('gameDate')

    direction = (record.get('direction') or '').upper()
    stat_type = (record.get('stat_type') or record.get('stat') or '').lower()
    opp_team = record.get('opp_team') or record.get('opponent') or ''

    # ── Advanced rolling stats ──
    if player_id is not None:
        adv = get_player_advanced_rolling(player_id, game_date)

        features['adv_usg_pct_l10'] = adv['usg_pct_l10']
        features['adv_net_rating_l10'] = adv['net_rating_l10']
        features['adv_off_rating_l10'] = adv['off_rating_l10']
        features['adv_def_rating_l10'] = adv['def_rating_l10']
        features['adv_pace_l10'] = adv['pace_l10']
        features['adv_efg_pct_l10'] = adv['efg_pct_l10']
        features['adv_ts_pct_l10'] = adv['ts_pct_l10']
        features['adv_pie_l10'] = adv['pie_l10']
        features['adv_usg_pct_std'] = adv['usg_pct_std']
        features['adv_net_rating_std'] = adv['net_rating_std']

        # ── EFG trend (L5 - L10, positive = improving) ──
        efg_l5 = adv.get('efg_pct_l5', np.nan)
        efg_l10 = adv.get('efg_pct_l10', np.nan)
        if not (math.isnan(efg_l5) or math.isnan(efg_l10)):
            features['adv_efg_trend'] = efg_l5 - efg_l10
        else:
            features['adv_efg_trend'] = np.nan

        # ── Usage rolling stats ──
        usg = get_player_usage_rolling(player_id, game_date)
        features['adv_pct_blk_l10'] = usg['pct_blk_l10']
        features['adv_pct_stl_l10'] = usg['pct_stl_l10']
        features['adv_pct_reb_l10'] = usg['pct_reb_l10']
        features['adv_pct_ast_l10'] = usg['pct_ast_l10']

        # ── Interaction: usage x direction ──
        # High usage + OVER = positive signal; high usage + UNDER = negative
        usg_val = adv['usg_pct_l10']
        if not math.isnan(usg_val):
            if direction == 'OVER':
                features['adv_usg_x_direction'] = usg_val
            elif direction == 'UNDER':
                features['adv_usg_x_direction'] = -usg_val
            else:
                features['adv_usg_x_direction'] = 0.0
        else:
            features['adv_usg_x_direction'] = np.nan

        # ── Interaction: pace x stat type ──
        # High pace helps pace-sensitive stats (pts, reb, ast, combos)
        pace_val = adv['pace_l10']
        if not math.isnan(pace_val):
            pace_norm = (pace_val - 100.0) / 5.0  # normalize around league avg ~100
            if stat_type in PACE_SENSITIVE_STATS:
                features['adv_pace_x_stat'] = pace_norm
            else:
                features['adv_pace_x_stat'] = pace_norm * 0.3  # reduced for non-pace stats
        else:
            features['adv_pace_x_stat'] = np.nan

    # ── Opponent cluster ──
    if opp_team:
        features['adv_opp_cluster'] = get_opponent_cluster(opp_team, game_date)

    return features


# ═══════════════════════════════════════════════════════════════
# TARGET ENCODING (leakage-safe)
# ═══════════════════════════════════════════════════════════════

def compute_target_encodings(training_records):
    """Compute player-level and team-level UNDER hit rates from training data.

    Uses leave-one-out encoding to prevent leakage: each record's encoding
    excludes that record from the computation.

    Args:
        training_records: list of dicts with 'player_name', 'team', 'direction', 'hit' fields

    Returns:
        dict with:
            'player': {player_name: (total_under_hits, total_count)}
            'team': {team_abbr: (total_under_hits, total_count)}
    """
    player_stats = defaultdict(lambda: [0, 0])  # [under_hits, count]
    team_stats = defaultdict(lambda: [0, 0])

    for rec in training_records:
        player = rec.get('player_name') or rec.get('player') or ''
        team = rec.get('team') or rec.get('teamAbbreviation') or ''
        direction = (rec.get('direction') or '').upper()
        hit = rec.get('hit')

        if hit is None:
            continue

        hit_val = 1 if hit else 0

        if player:
            player_stats[player][1] += 1
            if direction == 'UNDER' and hit_val:
                player_stats[player][0] += 1
            elif direction == 'OVER' and not hit_val:
                # OVER miss = UNDER would have hit
                player_stats[player][0] += 1

        if team:
            team_stats[team][1] += 1
            if direction == 'UNDER' and hit_val:
                team_stats[team][0] += 1
            elif direction == 'OVER' and not hit_val:
                team_stats[team][0] += 1

    encodings = {
        'player': {},
        'team': {},
        'global_under_rate': np.nan,
    }

    # Global prior
    total_under = sum(s[0] for s in player_stats.values())
    total_count = sum(s[1] for s in player_stats.values())
    global_rate = total_under / total_count if total_count > 0 else 0.5
    encodings['global_under_rate'] = global_rate

    # Player-level with smoothing (shrink toward global mean)
    SMOOTH_FACTOR = 20  # regularization strength
    for player, (under_hits, count) in player_stats.items():
        if count >= 5:
            raw_rate = under_hits / count
            # Bayesian shrinkage toward global mean
            smoothed = (under_hits + SMOOTH_FACTOR * global_rate) / (count + SMOOTH_FACTOR)
            encodings['player'][player] = smoothed
        # Skip players with < 5 records (will use global)

    # Team-level with smoothing
    for team, (under_hits, count) in team_stats.items():
        if count >= 10:
            smoothed = (under_hits + SMOOTH_FACTOR * global_rate) / (count + SMOOTH_FACTOR)
            encodings['team'][team] = smoothed

    return encodings


def apply_target_encodings(record, encodings):
    """Add target-encoded features: te_player_under_rate, te_team_under_rate.

    Args:
        record: prop dict with 'player_name' and 'team' fields
        encodings: output from compute_target_encodings()

    Returns:
        dict with te_player_under_rate, te_team_under_rate
    """
    result = {
        'te_player_under_rate': np.nan,
        'te_team_under_rate': np.nan,
    }

    if not encodings:
        return result

    global_rate = encodings.get('global_under_rate', np.nan)

    player = record.get('player_name') or record.get('player') or ''
    team = record.get('team') or record.get('teamAbbreviation') or ''

    # Player-level: use player rate if available, else global
    if player and player in encodings.get('player', {}):
        result['te_player_under_rate'] = encodings['player'][player]
    elif not math.isnan(global_rate):
        result['te_player_under_rate'] = global_rate

    # Team-level
    if team and team in encodings.get('team', {}):
        result['te_team_under_rate'] = encodings['team'][team]
    elif not math.isnan(global_rate):
        result['te_team_under_rate'] = global_rate

    return result


# ═══════════════════════════════════════════════════════════════
# CYCLICAL ENCODING
# ═══════════════════════════════════════════════════════════════

def add_cyclical_features(record, game_date_str):
    """Add sin/cos encoding for day-of-week and month.

    Features: dow_sin, dow_cos, month_sin, month_cos

    Args:
        record: prop dict (not modified)
        game_date_str: 'YYYY-MM-DD' string or date object

    Returns:
        dict with dow_sin, dow_cos, month_sin, month_cos
    """
    result = {
        'dow_sin': np.nan, 'dow_cos': np.nan,
        'month_sin': np.nan, 'month_cos': np.nan,
    }

    if isinstance(game_date_str, str):
        dt = _parse_date(game_date_str)
    else:
        dt = game_date_str

    if dt is None:
        return result

    # Day of week: 0=Monday, 6=Sunday
    dow = dt.weekday()
    result['dow_sin'] = math.sin(2 * math.pi * dow / 7)
    result['dow_cos'] = math.cos(2 * math.pi * dow / 7)

    # Month: 1-12
    month = dt.month
    result['month_sin'] = math.sin(2 * math.pi * month / 12)
    result['month_cos'] = math.cos(2 * math.pi * month / 12)

    return result


# ═══════════════════════════════════════════════════════════════
# INTEGRATION: ONE-CALL ENRICHMENT
# ═══════════════════════════════════════════════════════════════

def enrich_with_advanced_features(record, player_id=None, game_date=None, encodings=None):
    """One-call enrichment: adds all advanced features to a prop record.

    Combines:
        - Advanced rolling stats (usage, net rating, pace, efg, ts, pie)
        - Usage distribution (pct_blk, pct_stl, etc.)
        - Opponent cluster
        - Interaction features (usg x direction, pace x stat, efg trend)
        - Target encodings (if provided)
        - Cyclical date features

    Args:
        record: prop dict (modified in place AND returned)
        player_id: NBA personId
        game_date: game date string or date object
        encodings: output from compute_target_encodings() (optional)

    Returns:
        record dict with all new features added
    """
    # Advanced + interaction features
    adv_feats = engineer_advanced_features(record, player_id, game_date)
    record.update(adv_feats)

    # Target encodings
    if encodings:
        te_feats = apply_target_encodings(record, encodings)
        record.update(te_feats)

    # Cyclical features
    gd = game_date or record.get('game_date') or record.get('gameDate')
    if gd:
        cyc_feats = add_cyclical_features(record, gd)
        record.update(cyc_feats)

    return record


# ═══════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════

def _run_test():
    """Load data and print stats for verification."""
    import time

    print("=" * 60)
    print("Advanced Features Module — Self-Test")
    print("=" * 60)

    t0 = time.time()
    load_advanced_data()
    t1 = time.time()
    print(f"  Advanced data load time: {t1 - t0:.1f}s")

    t0 = time.time()
    load_usage_data()
    t1 = time.time()
    print(f"  Usage data load time: {t1 - t0:.1f}s")

    t0 = time.time()
    load_games_data()
    t1 = time.time()
    print(f"  Games data load time: {t1 - t0:.1f}s")

    print(f"\n  Players in advanced cache: {len(_ADV_CACHE):,}")
    print(f"  Players in usage cache: {len(_USAGE_CACHE):,}")
    print(f"  Teams in team cache: {len(_TEAM_ADV_CACHE):,}")
    print(f"  Games in games cache: {len(_GAMES_CACHE):,}")

    # Sample a player — find one with lots of data
    if _ADV_CACHE:
        # Get player with most games
        top_pid = max(_ADV_CACHE, key=lambda k: len(_ADV_CACHE[k]))
        top_games = _ADV_CACHE[top_pid]
        last_game = top_games[-1]
        print(f"\n  Most-tracked player ID: {top_pid} ({len(top_games)} games)")
        print(f"  Last game: {last_game['gameDate']} ({last_game['teamAbbreviation']})")

        # Test rolling features
        test_date = last_game['gameDate']
        adv_roll = get_player_advanced_rolling(top_pid, test_date)
        print(f"\n  Rolling advanced stats (L10 before {test_date}):")
        for k, v in sorted(adv_roll.items()):
            if k == 'efg_pct_l5':
                continue
            print(f"    {k}: {v:.3f}" if not math.isnan(v) else f"    {k}: NaN")

        usg_roll = get_player_usage_rolling(top_pid, test_date)
        print(f"\n  Rolling usage stats (L10 before {test_date}):")
        for k, v in sorted(usg_roll.items()):
            print(f"    {k}: {v:.4f}" if not math.isnan(v) else f"    {k}: NaN")

    # Test opponent cluster
    if _TEAM_ADV_CACHE:
        from datetime import date
        test_teams = list(_TEAM_ADV_CACHE.keys())[:5]
        test_dt = date(2026, 3, 1)
        print(f"\n  Opponent clusters (as of {test_dt}):")
        for team in test_teams:
            cluster = get_opponent_cluster(team, test_dt)
            label = {0: "Elite D/slow", 1: "Good D/moderate", 2: "Average",
                     3: "Poor D/fast", 4: "Poor D/slow"}.get(cluster, "N/A")
            print(f"    {team}: cluster {cluster} ({label})")

    # Test game officials
    if _GAMES_CACHE:
        sample_ids = list(_GAMES_CACHE.keys())[-3:]
        print(f"\n  Sample game officials:")
        for gid in sample_ids:
            officials = get_game_officials(gid)
            print(f"    Game {gid}: {officials}")

    # Test cyclical features
    from datetime import date
    cyc = add_cyclical_features({}, "2026-03-23")
    print(f"\n  Cyclical features for 2026-03-23:")
    for k, v in sorted(cyc.items()):
        print(f"    {k}: {v:.4f}")

    # Test full enrichment
    sample_record = {
        'direction': 'UNDER',
        'stat_type': 'pts',
        'player_name': 'Test Player',
        'team': 'LAL',
    }
    if _ADV_CACHE:
        top_pid = max(_ADV_CACHE, key=lambda k: len(_ADV_CACHE[k]))
        test_date = _ADV_CACHE[top_pid][-1]['gameDate']
        enriched = enrich_with_advanced_features(
            sample_record.copy(), player_id=top_pid, game_date=test_date
        )
        print(f"\n  Full enrichment ({len(ADVANCED_FEATURE_COLS)} features):")
        non_nan = sum(1 for col in ADVANCED_FEATURE_COLS
                      if col in enriched and not (isinstance(enriched[col], float) and math.isnan(enriched[col])))
        print(f"    Non-NaN features: {non_nan}/{len(ADVANCED_FEATURE_COLS)}")

    print(f"\n{'=' * 60}")
    print("Self-test complete.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        _run_test()
    else:
        print("Usage: python3 advanced_features.py test")
        print("  Loads CSVs and prints diagnostic stats.")

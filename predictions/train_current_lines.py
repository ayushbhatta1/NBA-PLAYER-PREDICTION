#!/usr/bin/env python3
"""
Focused Training Data Generator — Real Sportsbook Lines Applied to Historical Game Logs.

Takes TODAY's actual sportsbook board (real lines), applies them retroactively to each
player's full season of game logs, and labels each game as OVER/UNDER. This gives:
- Player X has PTS line 25.5 today -> check all 60 of their games -> 60 real-labeled samples
- 85 players x ~51 games x ~5 stats each = ~20,000+ focused samples
- ALL with the EXACT line being bet on today
- Rolling features (L10 avg, hit rate, etc.) computed correctly from prior games only

Advantage over existing backfill: uses REAL sportsbook lines instead of synthetic L10-derived
lines. The model trains on the exact lines it will be scoring.

Usage:
    python3 predictions/train_current_lines.py --board predictions/2026-03-19/2026-03-19_full_board.json
    python3 predictions/train_current_lines.py --board predictions/2026-03-19/2026-03-19_full_board.json --stats
    python3 predictions/train_current_lines.py --board predictions/2026-03-19/2026-03-19_full_board.json --compare
"""

import json
import math
import os
import sys
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PREDICTIONS_DIR, 'cache')
FOCUSED_MODEL_PATH = os.path.join(CACHE_DIR, 'focused_xgb_model.pkl')
FOCUSED_DATA_PATH = os.path.join(CACHE_DIR, 'focused_training_data.json')

# Stat column mapping: board stat name -> parquet column(s)
BASE_STATS = {
    'pts': 'PTS', 'reb': 'REB', 'ast': 'AST',
    '3pm': 'FG3M', 'stl': 'STL', 'blk': 'BLK',
}
COMBO_STATS = {
    'pra': ['PTS', 'REB', 'AST'],
    'pr': ['PTS', 'REB'],
    'pa': ['PTS', 'AST'],
    'ra': ['REB', 'AST'],
    'stl_blk': ['STL', 'BLK'],
}

# Tier thresholds (from analyze_v3.py)
def _compute_tier(abs_gap, is_combo=False):
    g = abs_gap
    if is_combo:
        g -= 0.5  # combo penalty before tier grading
    if g >= 4: return 'S'
    if g >= 3: return 'A'
    if g >= 2: return 'B'
    if g >= 1.5: return 'C'
    if g >= 1: return 'D'
    return 'F'


# ═══════════════════════════════════════════════════════════════
# PLAYER NAME -> ID RESOLUTION
# ═══════════════════════════════════════════════════════════════

def _build_name_to_id_map():
    """Build player name -> ID mapping from nba_api static data + parquet filenames.

    Uses both get_players() (all historical) and get_active_players() to maximize
    coverage. Also scans parquet filenames as fallback.
    """
    name_map = {}

    # Method 1: nba_api static player list (includes historical + rookies)
    try:
        from nba_api.stats.static import players as static_players
        all_players = static_players.get_players()
        for p in all_players:
            name_map[p['full_name'].lower()] = p['id']
    except Exception as e:
        print(f"  WARNING: nba_api static lookup failed: {e}")

    # Method 2: scan parquet filenames and read Player_ID from each
    # This catches any IDs that nba_api static might miss
    available_pids = set()
    parquets = [f for f in os.listdir(CACHE_DIR)
                if f.startswith('gamelog_') and f.endswith('.parquet')]
    for fname in parquets:
        try:
            pid = int(fname.split('_')[1])
            available_pids.add(pid)
        except (ValueError, IndexError):
            continue

    return name_map, available_pids


def _resolve_player_id(player_name, name_map, available_pids):
    """Resolve a board player name to a parquet player ID.

    Returns player_id if found and parquet exists, else None.
    """
    # Direct lookup
    pid = name_map.get(player_name.lower())
    if pid and pid in available_pids:
        return pid

    # Try without special characters (De'Aaron -> DeAaron)
    clean_name = player_name.replace("'", "").replace(".", "").lower()
    for stored_name, stored_pid in name_map.items():
        stored_clean = stored_name.replace("'", "").replace(".", "")
        if stored_clean == clean_name and stored_pid in available_pids:
            return stored_pid

    # Last-name fallback: find unique last-name match among available parquets
    last_name = player_name.split()[-1].lower() if ' ' in player_name else player_name.lower()
    last_name_matches = []
    for stored_name, stored_pid in name_map.items():
        stored_last = stored_name.split()[-1] if ' ' in stored_name else stored_name
        if stored_last == last_name and stored_pid in available_pids:
            last_name_matches.append((stored_name, stored_pid))

    if len(last_name_matches) == 1:
        return last_name_matches[0][1]

    # First + last name partial match
    parts = player_name.lower().split()
    if len(parts) >= 2:
        first_lower = parts[0].replace("'", "")
        last_lower = parts[-1]
        for stored_name, stored_pid in name_map.items():
            stored_parts = stored_name.split()
            if len(stored_parts) >= 2:
                sf = stored_parts[0].replace("'", "")
                sl = stored_parts[-1]
                if sf == first_lower and sl == last_lower and stored_pid in available_pids:
                    return stored_pid

    return None


# ═══════════════════════════════════════════════════════════════
# GAME LOG LOADING & STAT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _load_gamelog(player_id):
    """Load a player's cached parquet game log. Returns DataFrame sorted oldest->newest."""
    # Try 2025-26 season first, then any season
    for season_tag in ['2025-26', '2024-25']:
        path = os.path.join(CACHE_DIR, f'gamelog_{player_id}_{season_tag}.parquet')
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if df.empty:
                continue
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed')
            df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
            df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(0)
            # Derive IS_HOME and OPP_ABR from MATCHUP
            if 'MATCHUP' in df.columns:
                df['IS_HOME'] = df['MATCHUP'].apply(
                    lambda x: 1 if 'vs.' in str(x) else 0)
                df['OPP_ABR'] = df['MATCHUP'].apply(
                    lambda x: str(x).split('vs.')[-1].strip() if 'vs.' in str(x)
                    else str(x).split('@')[-1].strip() if '@' in str(x) else '')
            else:
                df['IS_HOME'] = 0
                df['OPP_ABR'] = ''
            return df

    # Fallback: find any matching parquet
    for fname in os.listdir(CACHE_DIR):
        if fname.startswith(f'gamelog_{player_id}_') and fname.endswith('.parquet'):
            path = os.path.join(CACHE_DIR, fname)
            df = pd.read_parquet(path)
            if df.empty:
                continue
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed')
            df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
            df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(0)
            if 'MATCHUP' in df.columns:
                df['IS_HOME'] = df['MATCHUP'].apply(
                    lambda x: 1 if 'vs.' in str(x) else 0)
                df['OPP_ABR'] = df['MATCHUP'].apply(
                    lambda x: str(x).split('vs.')[-1].strip() if 'vs.' in str(x)
                    else str(x).split('@')[-1].strip() if '@' in str(x) else '')
            else:
                df['IS_HOME'] = 0
                df['OPP_ABR'] = ''
            return df

    return None


def _get_stat_value(row, stat_name):
    """Get the stat value from a single DataFrame row."""
    if stat_name in BASE_STATS:
        col = BASE_STATS[stat_name]
        return float(row[col]) if col in row.index else 0.0
    elif stat_name in COMBO_STATS:
        cols = COMBO_STATS[stat_name]
        return float(sum(row[c] for c in cols if c in row.index))
    return 0.0


def _get_stat_series(df, stat_name):
    """Get stat values as a numpy array from DataFrame."""
    if stat_name in BASE_STATS:
        col = BASE_STATS[stat_name]
        if col in df.columns:
            return df[col].astype(float).values
        return np.zeros(len(df))
    elif stat_name in COMBO_STATS:
        cols = COMBO_STATS[stat_name]
        available = [c for c in cols if c in df.columns]
        if available:
            return df[available].astype(float).sum(axis=1).values
        return np.zeros(len(df))
    return np.zeros(len(df))


# ═══════════════════════════════════════════════════════════════
# CONTEXT FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def _load_venue_data():
    """Import venue_data for travel/timezone computations."""
    sys.path.insert(0, PREDICTIONS_DIR)
    from venue_data import VENUE_MAP, TZ_ORDINAL, haversine_miles
    return VENUE_MAP, TZ_ORDINAL, haversine_miles


def _load_team_rankings():
    """Load cached team rankings for matchup adjustments."""
    path = os.path.join(CACHE_DIR, 'team_rankings_2025-26.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    return data.get('teams', {}), data.get('league_avg', {})


# NBA abbreviation -> team rankings key (full team name used in team_rankings.json)
ABR_TO_TEAM_NAME = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards',
}


def _compute_b2b(dates_series, idx):
    """Check if game at idx is back-to-back."""
    if idx == 0:
        return False
    d1 = dates_series.iloc[idx]
    d0 = dates_series.iloc[idx - 1]
    return (d1 - d0).days <= 1


def _compute_travel_features(df, idx, venue_map, tz_ordinal, haversine_fn):
    """Compute travel distance, 7-day travel miles, timezone shifts."""
    travel_dist = np.nan
    travel_7day = np.nan
    tz_shifts = np.nan

    matchup = str(df.iloc[idx].get('MATCHUP', ''))

    if ' vs. ' in matchup:
        parts = matchup.split(' vs. ')
        home_abr = parts[0].strip()
    elif ' @ ' in matchup:
        parts = matchup.split(' @ ')
        home_abr = parts[1].strip()
    else:
        return travel_dist, travel_7day, tz_shifts

    game_venue_abr = home_abr

    # Previous game's venue
    if idx > 0:
        prev_matchup = str(df.iloc[idx - 1].get('MATCHUP', ''))
        prev_venue = None
        if ' vs. ' in prev_matchup:
            prev_venue = prev_matchup.split(' vs. ')[0].strip()
        elif ' @ ' in prev_matchup:
            prev_venue = prev_matchup.split(' @ ')[1].strip()

        if prev_venue and prev_venue in venue_map and game_venue_abr in venue_map:
            pv = venue_map[prev_venue]
            cv = venue_map[game_venue_abr]
            travel_dist = round(haversine_fn(pv['lat'], pv['lng'], cv['lat'], cv['lng']))

    # 7-day travel
    game_date = df.iloc[idx]['GAME_DATE']
    week_ago = game_date - timedelta(days=7)
    total_miles = 0.0
    tz_shift_count = 0
    prev_tz = None

    recent_mask = (df['GAME_DATE'] >= week_ago) & (df['GAME_DATE'] <= game_date)
    recent_games = df[recent_mask].sort_values('GAME_DATE')

    prev_venue_abr = None
    for _, row in recent_games.iterrows():
        m = str(row.get('MATCHUP', ''))
        if ' vs. ' in m:
            cur_venue = m.split(' vs. ')[0].strip()
        elif ' @ ' in m:
            cur_venue = m.split(' @ ')[1].strip()
        else:
            continue

        if prev_venue_abr and prev_venue_abr in venue_map and cur_venue in venue_map:
            pv = venue_map[prev_venue_abr]
            cv = venue_map[cur_venue]
            total_miles += haversine_fn(pv['lat'], pv['lng'], cv['lat'], cv['lng'])

        cur_tz = venue_map.get(cur_venue, {}).get('tz')
        if cur_tz and prev_tz and cur_tz != prev_tz:
            tz_shift_count += 1
        if cur_tz:
            prev_tz = cur_tz
        prev_venue_abr = cur_venue

    travel_7day = round(total_miles)
    tz_shifts = tz_shift_count

    return travel_dist, travel_7day, tz_shifts


def _compute_usage(df, idx):
    """Compute usage rate from game logs up to (not including) idx."""
    prior = df.iloc[:idx]
    prior = prior[prior['MIN'] >= 10]
    if len(prior) < 5:
        return np.nan, np.nan

    def _usage(subset):
        fga = subset['FGA'].astype(float)
        fta = subset['FTA'].astype(float)
        tov = subset['TOV'].astype(float)
        mins = subset['MIN'].astype(float).clip(lower=1)
        return float(((fga + 0.44 * fta + tov) / mins).mean())

    l10 = prior.tail(10)
    l5 = prior.tail(5)
    usage_rate = round(_usage(l10), 3)
    usage_trend = round(_usage(l5) - _usage(l10), 3)
    return usage_rate, usage_trend


def _matchup_adjustment(opp_team, stat_name, team_rankings, league_avg):
    """Compute rate-based matchup adjustment."""
    if not team_rankings or not opp_team or opp_team not in team_rankings:
        return 0.0, np.nan, np.nan

    opp = team_rankings[opp_team]
    stat_to_allowed = {
        'pts': 'avg_pts_allowed', 'reb': 'reb_allowed', 'ast': 'ast_allowed',
        '3pm': 'tpm_allowed', 'stl': 'stl_allowed', 'blk': 'blk_allowed',
    }

    if stat_name in COMBO_STATS:
        components = {
            'pra': ['pts', 'reb', 'ast'], 'pr': ['pts', 'reb'],
            'pa': ['pts', 'ast'], 'ra': ['reb', 'ast'],
            'stl_blk': ['stl', 'blk'],
        }
        adjs = []
        for s in components.get(stat_name, []):
            a, _, _ = _matchup_adjustment(opp_team, s, team_rankings, league_avg)
            adjs.append(a)
        return sum(adjs) / len(adjs) if adjs else 0.0, np.nan, np.nan

    allowed_key = stat_to_allowed.get(stat_name)
    if not allowed_key:
        return 0.0, np.nan, np.nan

    opp_allowed = opp.get(allowed_key, 0)
    lg_avg = league_avg.get(allowed_key, 0) if league_avg else 0

    if lg_avg <= 0:
        return 0.0, opp_allowed, 0.0

    diff = opp_allowed - lg_avg
    rate_vs_avg = diff / lg_avg if lg_avg > 0 else 0
    adjustment = diff / 75.0

    return round(adjustment, 2), round(opp_allowed, 1), round(rate_vs_avg, 3)


# ═══════════════════════════════════════════════════════════════
# MAIN GENERATION LOGIC
# ═══════════════════════════════════════════════════════════════

def generate_focused_training(board_file):
    """Generate focused training data from a board file + historical game logs.

    For each player-stat-line on the board, walks through the player's season
    game logs and creates labeled samples using the EXACT sportsbook line.

    Returns list of record dicts compatible with engineer_features().
    """
    print("=" * 60)
    print("  Focused Training Data Generator")
    print("  Using REAL sportsbook lines from today's board")
    print("=" * 60)

    # Load board
    with open(board_file) as f:
        board = json.load(f)
    print(f"\n  Board: {len(board)} props from {board_file}")

    # Extract unique player-stat-line combos
    prop_map = {}  # {(player_name, stat): line}
    for r in board:
        player = r.get('player', '')
        stat = r.get('stat', '')
        line = r.get('line')
        if player and stat and line is not None:
            key = (player, stat)
            prop_map[key] = float(line)

    unique_players = sorted(set(p for p, _ in prop_map.keys()))
    print(f"  Unique players: {len(unique_players)}")
    print(f"  Unique player-stat-line combos: {len(prop_map)}")

    # Build name -> ID mapping
    print("\n  Building player name -> ID mapping...")
    name_map, available_pids = _build_name_to_id_map()
    print(f"  nba_api static: {len(name_map)} names")
    print(f"  Available parquets: {len(available_pids)} player IDs")

    # Resolve player IDs
    player_ids = {}
    missing = []
    for player_name in unique_players:
        pid = _resolve_player_id(player_name, name_map, available_pids)
        if pid:
            player_ids[player_name] = pid
        else:
            missing.append(player_name)

    print(f"  Resolved: {len(player_ids)}/{len(unique_players)} players")
    if missing:
        print(f"  Missing: {', '.join(missing[:10])}" +
              (f" + {len(missing)-10} more" if len(missing) > 10 else ""))

    # Load context data
    team_rankings, league_avg = _load_team_rankings()
    print(f"  Team rankings: {'loaded' if team_rankings else 'NOT FOUND'}")

    try:
        venue_map, tz_ordinal, haversine_fn = _load_venue_data()
        has_venue = True
        print(f"  Venue data: loaded ({len(venue_map)} arenas)")
    except Exception as e:
        has_venue = False
        venue_map, tz_ordinal, haversine_fn = {}, {}, None
        print(f"  Venue data: FAILED ({e})")

    # Generate records
    all_records = []
    player_stats = {}  # track per-player sample counts
    t0 = time.time()

    for player_name, pid in sorted(player_ids.items()):
        df = _load_gamelog(pid)
        if df is None:
            continue

        # Filter to games with meaningful minutes
        df = df[df['MIN'] >= 10].reset_index(drop=True)
        if len(df) < 11:
            continue

        player_record_count = 0

        # Process each stat-line combo for this player
        for (pname, stat_name), line in prop_map.items():
            if pname != player_name:
                continue

            is_combo = stat_name in COMBO_STATS

            # Verify we can compute this stat from the parquet
            if stat_name in BASE_STATS:
                col = BASE_STATS[stat_name]
                if col not in df.columns:
                    continue
            elif stat_name in COMBO_STATS:
                cols = COMBO_STATS[stat_name]
                if not all(c in df.columns for c in cols):
                    continue
            else:
                continue

            stat_vals = _get_stat_series(df, stat_name)

            # Walk through games starting from game 11 (index 10)
            for idx in range(10, len(df)):
                game_row = df.iloc[idx]
                actual = float(stat_vals[idx])
                game_date = game_row['GAME_DATE']
                game_date_str = game_date.strftime('%Y-%m-%d')

                # Skip pushes
                if actual == line:
                    continue

                # Prior games only (no leakage)
                prior_vals = stat_vals[:idx]
                l10_vals = prior_vals[-10:]
                l5_vals = prior_vals[-5:]
                l3_vals = prior_vals[-3:]

                season_avg = float(np.mean(prior_vals))
                l10_avg = float(np.mean(l10_vals))
                l5_avg = float(np.mean(l5_vals))
                l3_avg = float(np.mean(l3_vals))
                l10_std = float(np.std(l10_vals))

                # Home/away splits from prior games
                prior_df = df.iloc[:idx]
                home_mask = prior_df['IS_HOME'] == 1
                away_mask = prior_df['IS_HOME'] == 0
                prior_stat_series = pd.Series(prior_vals)

                home_vals_s = prior_stat_series[home_mask.values[:idx]] if home_mask.sum() > 3 else None
                away_vals_s = prior_stat_series[away_mask.values[:idx]] if away_mask.sum() > 3 else None
                home_avg = float(home_vals_s.mean()) if home_vals_s is not None and len(home_vals_s) > 0 else season_avg
                away_avg = float(away_vals_s.mean()) if away_vals_s is not None and len(away_vals_s) > 0 else season_avg

                # Hit rates vs the REAL sportsbook line
                l10_hit_rate = float(np.sum(l10_vals > line) / len(l10_vals) * 100)
                l5_hit_rate = float(np.sum(l5_vals > line) / len(l5_vals) * 100)
                season_hit_rate = float(np.sum(prior_vals > line) / len(prior_vals) * 100)

                # Minutes consistency
                prior_mins = prior_df['MIN'].values
                mins_30plus_pct = float(np.sum(prior_mins >= 30) / len(prior_mins) * 100)

                # L10 floor and miss count (vs the REAL line)
                l10_floor = float(np.min(l10_vals))
                l10_miss_count = int(np.sum(l10_vals <= line))

                # Streak detection
                streak_pct = float((l3_avg - l10_avg) / l10_avg * 100) if l10_avg > 0 else 0
                if streak_pct > 15:
                    streak_status = 'HOT'
                elif streak_pct < -15:
                    streak_status = 'COLD'
                else:
                    streak_status = 'NEUTRAL'

                # B2B detection
                is_b2b = _compute_b2b(df['GAME_DATE'], idx)

                # Home/away
                is_home = int(game_row.get('IS_HOME', 0))

                # Projection (analyze_v3 formula)
                projection = 0.4 * l10_avg + 0.3 * l5_avg + 0.3 * season_avg

                # Split adjustment
                if is_home and home_vals_s is not None and len(home_vals_s) > 0:
                    split_adj = round(float(home_avg - season_avg) * 0.5, 2)
                elif not is_home and away_vals_s is not None and len(away_vals_s) > 0:
                    split_adj = round(float(away_avg - season_avg) * 0.5, 2)
                else:
                    split_adj = 0.0

                # Matchup adjustment (translate abbreviation to team name)
                opp_abr = game_row.get('OPP_ABR', '')
                opp_team = ABR_TO_TEAM_NAME.get(opp_abr, opp_abr)
                matchup_adj, opp_allowed_rate, opp_allowed_vs_league = _matchup_adjustment(
                    opp_team, stat_name, team_rankings, league_avg
                )

                # Travel features
                if has_venue:
                    travel_dist, travel_7day, tz_shifts = _compute_travel_features(
                        df, idx, venue_map, tz_ordinal, haversine_fn
                    )
                else:
                    travel_dist, travel_7day, tz_shifts = np.nan, np.nan, np.nan

                # Usage metrics
                usage_rate, usage_trend = _compute_usage(df, idx)

                # Plus/minus and PF from L10
                l10_df = prior_df.tail(10)
                l10_plus_minus = round(float(l10_df['PLUS_MINUS'].mean()), 1) if 'PLUS_MINUS' in l10_df.columns else 0.0
                l10_pf = round(float(l10_df['PF'].mean()), 1) if 'PF' in l10_df.columns else 0.0
                l5_pf = round(float(prior_df.tail(5)['PF'].mean()), 1) if 'PF' in prior_df.columns else 0.0
                foul_trouble_risk = l5_pf >= 4.0

                # B2B travel adjustment
                b2b_adj = 0.0
                if is_b2b and not np.isnan(travel_dist):
                    if travel_dist > 1500:
                        b2b_adj = -0.04
                    elif travel_dist > 500:
                        b2b_adj = -0.02
                    else:
                        b2b_adj = -0.01

                # Gap and tier
                gap = projection - line
                abs_gap = abs(gap)
                effective_gap = gap
                tier = _compute_tier(abs_gap, is_combo)

                # Minutes adjustment
                season_mins = float(prior_df['MIN'].mean())
                l5_mins = float(prior_df.tail(5)['MIN'].mean())
                mins_adj = round((l5_mins - season_mins) / season_mins * 0.5, 2) if season_mins > 0 else 0.0

                # Base record (shared fields for OVER and UNDER)
                rec_base = {
                    'player': player_name,
                    'stat': stat_name,
                    'line': line,
                    'projection': round(projection, 1),
                    'gap': round(gap, 1),
                    'abs_gap': round(abs_gap, 1),
                    'effective_gap': round(effective_gap, 1),
                    'season_avg': round(season_avg, 1),
                    'l10_avg': round(l10_avg, 1),
                    'l5_avg': round(l5_avg, 1),
                    'l3_avg': round(l3_avg, 1),
                    'home_avg': round(home_avg, 1),
                    'away_avg': round(away_avg, 1),
                    'l10_hit_rate': round(l10_hit_rate),
                    'l5_hit_rate': round(l5_hit_rate),
                    'season_hit_rate': round(season_hit_rate),
                    'mins_30plus_pct': round(mins_30plus_pct),
                    'split_adjustment': split_adj,
                    'matchup_adjustment': matchup_adj,
                    'mins_adj': round(mins_adj, 2),
                    'streak_adj': 0.0,
                    'blowout_adj': 0.0,
                    'injury_adjustment': 0,
                    'spread': np.nan,
                    'streak_pct': round(streak_pct, 1),
                    'games_used': len(prior_vals),
                    'is_home': is_home,
                    'is_b2b': is_b2b,
                    'l10_floor': round(l10_floor, 1),
                    'l10_miss_count': l10_miss_count,
                    'l10_std': round(l10_std, 2),
                    'l10_values': [round(float(v), 1) for v in l10_vals],
                    'streak_status': streak_status,
                    'tier': tier,
                    'opponent_history': None,
                    'same_team_out_count': 0,
                    # v4 features
                    'l10_avg_plus_minus': l10_plus_minus,
                    'l10_avg_pf': l10_pf,
                    'foul_trouble_risk': foul_trouble_risk,
                    # v5 features
                    'opp_stat_allowed_rate': _nan_safe(opp_allowed_rate),
                    'opp_stat_allowed_vs_league_avg': _nan_safe(opp_allowed_vs_league),
                    'usage_rate': _nan_safe(usage_rate),
                    'usage_trend': _nan_safe(usage_trend),
                    'dynamic_without_delta': None,
                    'travel_distance': _nan_safe(travel_dist),
                    'travel_miles_7day': _nan_safe(travel_7day),
                    'tz_shifts_7day': _nan_safe(tz_shifts),
                    # v7 features (enrichment -- not available retroactively)
                    'opp_matchup_delta': None,
                    'team_vs_opp_delta': None,
                    'opp_off_pressure': None,
                    'usage_boost': None,
                    'game_total_signal': None,
                    'max_same_game_corr': None,
                    # Metadata
                    '_date': game_date_str,
                    '_source': 'focused',
                    '_data_source': 'focused',
                    '_feature_version': 7,
                    '_board_line': line,  # flag: this is a REAL sportsbook line
                    '_actual': round(actual, 1),
                }

                # OVER record
                over_rec = dict(rec_base)
                over_rec['direction'] = 'OVER'
                over_rec['_hit_label'] = actual > line
                all_records.append(over_rec)

                # UNDER record
                under_rec = dict(rec_base)
                under_rec['direction'] = 'UNDER'
                under_rec['effective_gap'] = round(line - projection, 1)
                under_rec['_hit_label'] = actual < line
                all_records.append(under_rec)

                player_record_count += 2

        # Track per-player stats
        if player_record_count > 0:
            player_stats[player_name] = player_record_count

    elapsed = time.time() - t0
    print(f"\n  Generated {len(all_records)} total records from "
          f"{len(player_stats)} players in {elapsed:.1f}s")

    if all_records:
        _print_summary(all_records, player_stats)

    return all_records


def _nan_safe(val):
    """Convert NaN to None for JSON compatibility."""
    if val is None:
        return None
    try:
        if np.isnan(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _print_summary(records, player_stats=None):
    """Print summary statistics."""
    n = len(records)
    hits = sum(1 for r in records if r['_hit_label'])
    dates = sorted(set(r['_date'] for r in records))

    # By stat
    stat_counts = {}
    stat_hits = {}
    for r in records:
        s = r['stat']
        stat_counts[s] = stat_counts.get(s, 0) + 1
        if r['_hit_label']:
            stat_hits[s] = stat_hits.get(s, 0) + 1

    # By direction
    dir_counts = {}
    dir_hits = {}
    for r in records:
        d = r['direction']
        dir_counts[d] = dir_counts.get(d, 0) + 1
        if r['_hit_label']:
            dir_hits[d] = dir_hits.get(d, 0) + 1

    # By tier
    tier_counts = {}
    for r in records:
        t = r['tier']
        tier_counts[t] = tier_counts.get(t, 0) + 1

    print(f"\n  Summary:")
    print(f"    Records:    {n:,}")
    print(f"    Hit rate:   {hits/n:.1%} ({hits:,}/{n:,})")
    print(f"    Date range: {dates[0]} -> {dates[-1]} ({len(dates)} unique)")

    print(f"\n    By direction:")
    for d in ['OVER', 'UNDER']:
        cnt = dir_counts.get(d, 0)
        h = dir_hits.get(d, 0)
        hr = h / cnt * 100 if cnt > 0 else 0
        print(f"      {d:6s}: {cnt:5,} records, {hr:5.1f}% hit rate")

    print(f"\n    By stat:")
    for s in sorted(stat_counts.keys()):
        cnt = stat_counts[s]
        h = stat_hits.get(s, 0)
        hr = h / cnt * 100 if cnt > 0 else 0
        print(f"      {s:8s}: {cnt:5,} records, {hr:5.1f}% hit rate")

    print(f"\n    By tier:")
    for t in ['S', 'A', 'B', 'C', 'D', 'F']:
        cnt = tier_counts.get(t, 0)
        if cnt > 0:
            print(f"      {t}: {cnt:5,} ({cnt/n*100:.1f}%)")

    if player_stats:
        print(f"\n    Top 10 players by sample count:")
        for name, count in sorted(player_stats.items(), key=lambda x: -x[1])[:10]:
            print(f"      {name:25s}: {count:5,} records")

    # Feature coverage
    context_fields = [
        'matchup_adjustment', 'split_adjustment', 'mins_adj',
        'opp_stat_allowed_rate', 'usage_rate', 'travel_distance',
        'l10_avg_plus_minus', 'l10_avg_pf',
    ]
    print(f"\n    Feature coverage (non-zero/non-null):")
    for field in context_fields:
        non_null = sum(1 for r in records
                       if r.get(field) is not None
                       and r.get(field) != 0
                       and not (isinstance(r.get(field), float) and np.isnan(r.get(field))))
        pct = non_null / n * 100
        print(f"      {field:35s} {pct:5.1f}%")


# ═══════════════════════════════════════════════════════════════
# TRAINING ON FOCUSED DATA
# ═══════════════════════════════════════════════════════════════

def train_on_current_lines(board_file, save_data=True):
    """Generate focused training data and train XGBoost on it.

    Uses walk-forward CV (first 70% train, last 30% test by game date).
    Saves model and training data.
    """
    # Generate focused data
    records = generate_focused_training(board_file)
    if not records:
        print("  ERROR: No training records generated!")
        return None, None

    # Save focused training data
    if save_data:
        _save_records(records, FOCUSED_DATA_PATH)

    # Import XGBoost model utilities
    sys.path.insert(0, PREDICTIONS_DIR)
    from xgb_model import engineer_features, FEATURE_COLS

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  ERROR: xgboost not installed. Run: pip3 install xgboost")
        return None, None

    # Engineer features
    print("\n" + "=" * 60)
    print("  Training Focused XGBoost Model")
    print("=" * 60)

    X, y, dates = engineer_features(records)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Overall hit rate: {y.mean():.1%}")

    # Walk-forward CV: split by game date (first 70% train, last 30% test)
    unique_dates = sorted(set(dates))
    split_idx = int(len(unique_dates) * 0.7)
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])

    dates_arr = np.array(dates)
    train_mask = np.array([d in train_dates for d in dates])
    test_mask = np.array([d in test_dates for d in dates])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"\n  Walk-forward split:")
    print(f"    Train: {len(y_train):,} samples ({unique_dates[0]} -> {unique_dates[split_idx-1]})")
    print(f"    Test:  {len(y_test):,} samples ({unique_dates[split_idx]} -> {unique_dates[-1]})")

    # Class balance
    n_hit = int(y.sum())
    n_miss = len(y) - n_hit
    scale_pos_weight = n_miss / n_hit if n_hit > 0 else 1.0

    # Train XGBoost (same params as xgb_model.py)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'colsample_bylevel': 0.8,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'gamma': 0.3,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbosity': 0,
    }

    model = XGBClassifier(**params)

    # Train with early stopping on test set
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )

    best_n = model.best_iteration + 1 if hasattr(model, 'best_iteration') and model.best_iteration else params['n_estimators']
    print(f"  Best iteration: {best_n}")

    # Evaluate on test set
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    test_acc = float(np.mean(test_preds == y_test))

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(y_test, test_probs)
    except Exception:
        test_auc = _compute_auc_manual(y_test, test_probs)

    print(f"\n  Test Results:")
    print(f"    Accuracy:  {test_acc:.1%}")
    print(f"    AUC:       {test_auc:.3f}")
    print(f"    Hit rate:  {y_test.mean():.1%} (baseline)")

    # Top/bottom decile analysis
    if len(test_probs) >= 20:
        sorted_idx = np.argsort(test_probs)
        decile_size = len(test_probs) // 10

        top_decile = y_test[sorted_idx[-decile_size:]]
        bot_decile = y_test[sorted_idx[:decile_size]]
        print(f"    Top decile: {top_decile.mean():.1%} ({top_decile.sum()}/{len(top_decile)})")
        print(f"    Bot decile: {bot_decile.mean():.1%} ({bot_decile.sum()}/{len(bot_decile)})")

    # Retrain on ALL data with the optimal iteration count
    print(f"\n  Retraining on all {len(y)} samples (n_estimators={min(best_n + 20, 800)})...")
    params['n_estimators'] = min(best_n + 20, 800)
    final_model = XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)

    # Save model
    model_path = os.path.join(CACHE_DIR, 'focused_xgb_model.json')
    final_model.save_model(model_path)
    print(f"  Model saved: {model_path}")

    # Feature importance
    importance = dict(zip(FEATURE_COLS, final_model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"\n  Top 15 features:")
    for name, imp in top_features:
        print(f"    {name:35s} {imp:.4f}")

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'board_file': board_file,
        'n_samples': int(len(y)),
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'hit_rate': float(y.mean()),
        'test_accuracy': test_acc,
        'test_auc': test_auc,
        'top_decile_hr': float(top_decile.mean()) if len(test_probs) >= 20 else None,
        'bot_decile_hr': float(bot_decile.mean()) if len(test_probs) >= 20 else None,
        'best_iteration': best_n,
        'n_features': int(X.shape[1]),
        'top_features': [(name, float(imp)) for name, imp in top_features],
        'model_path': model_path,
    }

    meta_path = model_path.replace('.json', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

    return final_model, metadata


def _compute_auc_manual(y_true, y_prob):
    """Manual AUC computation without sklearn."""
    pairs = list(zip(y_prob, y_true))
    pairs.sort(key=lambda x: x[0], reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0
    fp = 0
    auc = 0.0
    prev_prob = -1
    for prob, label in pairs:
        if prob != prev_prob:
            prev_prob = prob
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)


# ═══════════════════════════════════════════════════════════════
# SCORING WITH FOCUSED MODEL
# ═══════════════════════════════════════════════════════════════

def score_with_focused(results, model_path=None):
    """Score today's props using the focused model.

    Loads the focused XGBoost model, runs engineer_features(),
    and sets focused_prob on each prop dict.
    """
    if model_path is None:
        model_path = os.path.join(CACHE_DIR, 'focused_xgb_model.json')

    if not os.path.exists(model_path):
        print(f"  WARNING: No focused model at {model_path}")
        return 0

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  WARNING: xgboost not installed")
        return 0

    sys.path.insert(0, PREDICTIONS_DIR)
    from xgb_model import engineer_features

    model = XGBClassifier()
    model.load_model(model_path)

    # Build feature matrix
    temp_records = []
    for r in results:
        rec = dict(r)
        rec['_hit_label'] = False  # dummy
        rec['_date'] = ''
        temp_records.append(rec)

    if not temp_records:
        return results

    X, _, _ = engineer_features(temp_records)
    probs = model.predict_proba(X)[:, 1]

    scored = 0
    for r, prob in zip(results, probs):
        r['focused_prob'] = round(float(prob), 4)
        scored += 1

        # Compare with regular xgb_prob if available
        xgb_prob = r.get('xgb_prob')
        if xgb_prob is not None:
            r['focused_vs_xgb_delta'] = round(float(prob) - float(xgb_prob), 4)

    print(f"  Scored {scored} props with focused model")

    # Print comparison stats if xgb_prob available
    both = [(r['focused_prob'], r['xgb_prob']) for r in results
            if 'focused_prob' in r and 'xgb_prob' in r and r['xgb_prob'] is not None]
    if both:
        f_probs, x_probs = zip(*both)
        corr = float(np.corrcoef(f_probs, x_probs)[0, 1]) if len(both) > 2 else 0
        avg_delta = float(np.mean([f - x for f, x in both]))
        print(f"  Focused vs XGBoost: corr={corr:.3f}, avg delta={avg_delta:+.3f}")
        agree = sum(1 for f, x in both if (f >= 0.5) == (x >= 0.5))
        print(f"  Direction agreement: {agree}/{len(both)} ({agree/len(both)*100:.1f}%)")

    return scored


# ═══════════════════════════════════════════════════════════════
# DATA PERSISTENCE
# ═══════════════════════════════════════════════════════════════

def _save_records(records, path=None):
    """Save records to JSON with numpy type conversion."""
    if path is None:
        path = FOCUSED_DATA_PATH

    os.makedirs(os.path.dirname(path), exist_ok=True)

    clean_records = []
    for r in records:
        clean = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            elif isinstance(v, np.ndarray):
                clean[k] = [float(x) for x in v]
            elif isinstance(v, list):
                clean[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            elif isinstance(v, bool):
                clean[k] = v
            else:
                clean[k] = v
        clean_records.append(clean)

    with open(path, 'w') as f:
        json.dump(clean_records, f)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n  Saved {len(clean_records):,} records to {path} ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════
# COMPARISON TOOL
# ═══════════════════════════════════════════════════════════════

def compare_models(board_file):
    """Compare focused model vs baseline XGBoost on the focused test data."""
    print("=" * 60)
    print("  Model Comparison: Focused vs Baseline XGBoost")
    print("=" * 60)

    # Load focused training data
    if not os.path.exists(FOCUSED_DATA_PATH):
        print(f"  No focused data found. Run with --board first.")
        return

    with open(FOCUSED_DATA_PATH) as f:
        records = json.load(f)
    print(f"  Loaded {len(records):,} focused training records")

    sys.path.insert(0, PREDICTIONS_DIR)
    from xgb_model import engineer_features, MODEL_PATH, FEATURE_COLS

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  ERROR: xgboost not installed")
        return

    X, y, dates = engineer_features(records)

    # Walk-forward split (last 30% as test)
    unique_dates = sorted(set(dates))
    split_idx = int(len(unique_dates) * 0.7)
    test_dates = set(unique_dates[split_idx:])
    test_mask = np.array([d in test_dates for d in dates])
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Test set: {len(y_test)} samples")

    results = {}

    # Score with focused model
    focused_path = os.path.join(CACHE_DIR, 'focused_xgb_model.json')
    if os.path.exists(focused_path):
        focused_model = XGBClassifier()
        focused_model.load_model(focused_path)
        f_probs = focused_model.predict_proba(X_test)[:, 1]
        f_preds = (f_probs >= 0.5).astype(int)
        f_acc = float(np.mean(f_preds == y_test))
        f_auc = _compute_auc_manual(y_test, f_probs)
        results['focused'] = {'acc': f_acc, 'auc': f_auc, 'probs': f_probs}
        print(f"\n  Focused model:  Acc={f_acc:.1%}, AUC={f_auc:.3f}")
    else:
        print(f"  WARNING: No focused model found at {focused_path}")

    # Score with baseline model (may have fewer features than current FEATURE_COLS)
    if os.path.exists(MODEL_PATH):
        baseline_model = XGBClassifier()
        baseline_model.load_model(MODEL_PATH)
        # Check feature count mismatch: baseline may expect fewer features
        baseline_n_features = baseline_model.n_features_in_
        if X_test.shape[1] != baseline_n_features:
            print(f"  NOTE: Baseline model expects {baseline_n_features} features, "
                  f"focused data has {X_test.shape[1]}. Truncating to first {baseline_n_features}.")
            X_test_baseline = X_test[:, :baseline_n_features]
        else:
            X_test_baseline = X_test
        b_probs = baseline_model.predict_proba(X_test_baseline)[:, 1]
        b_preds = (b_probs >= 0.5).astype(int)
        b_acc = float(np.mean(b_preds == y_test))
        b_auc = _compute_auc_manual(y_test, b_probs)
        results['baseline'] = {'acc': b_acc, 'auc': b_auc, 'probs': b_probs}
        print(f"  Baseline model: Acc={b_acc:.1%}, AUC={b_auc:.3f}")
    else:
        print(f"  WARNING: No baseline model at {MODEL_PATH}")

    # Comparison
    if 'focused' in results and 'baseline' in results:
        delta_acc = results['focused']['acc'] - results['baseline']['acc']
        delta_auc = results['focused']['auc'] - results['baseline']['auc']
        print(f"\n  Delta (focused - baseline):")
        print(f"    Accuracy: {delta_acc:+.1%}")
        print(f"    AUC:      {delta_auc:+.3f}")

        # Decile comparison
        if len(y_test) >= 20:
            decile_size = len(y_test) // 10

            for name, data in results.items():
                probs = data['probs']
                sorted_idx = np.argsort(probs)
                top = y_test[sorted_idx[-decile_size:]]
                bot = y_test[sorted_idx[:decile_size]]
                print(f"\n  {name.capitalize()} decile analysis:")
                print(f"    Top decile: {top.mean():.1%} ({top.sum()}/{len(top)})")
                print(f"    Bot decile: {bot.mean():.1%} ({bot.sum()}/{len(bot)})")

        # Per-direction comparison
        print(f"\n  Per-direction accuracy:")
        test_records = [r for r, m in zip(records, test_mask.tolist()) if m]
        for direction in ['OVER', 'UNDER']:
            dir_mask = np.array([r.get('direction') == direction for r in test_records])
            if dir_mask.sum() == 0:
                continue
            y_dir = y_test[dir_mask]
            for name, data in results.items():
                p = (data['probs'][dir_mask] >= 0.5).astype(int)
                acc = float(np.mean(p == y_dir))
                print(f"    {name:10s} {direction:5s}: {acc:.1%} ({dir_mask.sum()} samples)")


# ═══════════════════════════════════════════════════════════════
# STATS DISPLAY
# ═══════════════════════════════════════════════════════════════

def print_stats(board_file):
    """Show detailed stats about what the focused data would look like."""
    print("=" * 60)
    print("  Focused Training Data Statistics")
    print("=" * 60)

    with open(board_file) as f:
        board = json.load(f)

    # Build name map
    name_map, available_pids = _build_name_to_id_map()

    # Extract unique player-stat combos
    prop_map = {}
    for r in board:
        player = r.get('player', '')
        stat = r.get('stat', '')
        line = r.get('line')
        if player and stat and line is not None:
            prop_map[(player, stat)] = float(line)

    unique_players = sorted(set(p for p, _ in prop_map.keys()))
    print(f"\n  Board: {len(board)} props, {len(unique_players)} players, "
          f"{len(prop_map)} unique player-stat combos")

    print(f"\n  {'Player':<25s} {'ID':>10s} {'Games':>6s} {'Stats':>6s} {'Est. Samples':>13s}")
    print(f"  {'-'*25} {'-'*10} {'-'*6} {'-'*6} {'-'*13}")

    total_estimated = 0
    resolved_count = 0

    for player_name in unique_players:
        pid = _resolve_player_id(player_name, name_map, available_pids)
        if not pid:
            print(f"  {player_name:<25s} {'MISSING':>10s}")
            continue

        resolved_count += 1
        df = _load_gamelog(pid)
        if df is None:
            print(f"  {player_name:<25s} {pid:>10d} {'NO LOG':>6s}")
            continue

        df = df[df['MIN'] >= 10].reset_index(drop=True)
        n_games = len(df)
        usable_games = max(0, n_games - 10)

        # Count stats for this player
        player_stats = sum(1 for (p, s) in prop_map if p == player_name)

        # Each game produces 2 records (OVER + UNDER) per stat
        est_samples = usable_games * player_stats * 2
        total_estimated += est_samples

        print(f"  {player_name:<25s} {pid:>10d} {n_games:>6d} {player_stats:>6d} {est_samples:>13,d}")

    print(f"\n  Resolved: {resolved_count}/{len(unique_players)} players")
    print(f"  Estimated total samples: {total_estimated:,}")
    print(f"  (actual may be slightly less due to pushes)")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Focused Training Data Generator - uses real sportsbook lines")
    parser.add_argument('--board', required=True,
                        help='Path to parsed board JSON file')
    parser.add_argument('--stats', action='store_true',
                        help='Show stats about expected sample counts (no training)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare focused model vs baseline XGBoost')
    parser.add_argument('--generate-only', action='store_true',
                        help='Generate focused data without training')
    parser.add_argument('--score', type=str, default=None,
                        help='Path to results JSON to score with focused model')
    args = parser.parse_args()

    if not os.path.exists(args.board):
        print(f"  ERROR: Board file not found: {args.board}")
        sys.exit(1)

    if args.stats:
        print_stats(args.board)
    elif args.compare:
        compare_models(args.board)
    elif args.generate_only:
        records = generate_focused_training(args.board)
        if records:
            _save_records(records, FOCUSED_DATA_PATH)
    elif args.score:
        if not os.path.exists(args.score):
            print(f"  ERROR: Results file not found: {args.score}")
            sys.exit(1)
        with open(args.score) as f:
            results = json.load(f)
        results = score_with_focused(results)
        # Save scored results
        out_path = args.score.replace('.json', '_focused_scored.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved scored results to {out_path}")
    else:
        # Default: generate + train
        model, metadata = train_on_current_lines(args.board)
        if model and metadata:
            print(f"\n  Done. Model AUC: {metadata['test_auc']:.3f}")


if __name__ == '__main__':
    main()

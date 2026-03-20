#!/usr/bin/env python3
"""
Coach Tendency Profiling for NBA Prop Betting Pipeline.

Profiles each head coach's rotation patterns, minute distributions, and
blowout behavior from cached player game logs. Features feed into
XGBoost/MLP models and Parlay Engine scoring.

Usage:
    from coach_model import get_coach_features, enrich_with_coach_features

    # Single team lookup
    features = get_coach_features('LAL', GAMES)

    # Bulk enrich prop predictions
    enrich_with_coach_features(results, GAMES)

CLI:
    python3 predictions/coach_model.py --build          # Build all profiles
    python3 predictions/coach_model.py --test LAL       # Test single team
"""

import json
import os
import time
import glob
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

# ── Constants ──

API_DELAY = 0.6  # seconds between nba_api calls

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

# All 30 NBA team abbreviations to nba_api team IDs
TEAM_ABR_TO_ID = {
    'ATL': 1610612737, 'BKN': 1610612751, 'BOS': 1610612738,
    'CHA': 1610612766, 'CHI': 1610612741, 'CLE': 1610612739,
    'DAL': 1610612742, 'DEN': 1610612743, 'DET': 1610612765,
    'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
    'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763,
    'MIA': 1610612748, 'MIL': 1610612749, 'MIN': 1610612750,
    'NOP': 1610612740, 'NYK': 1610612752, 'OKC': 1610612760,
    'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759,
    'TOR': 1610612761, 'UTA': 1610612762, 'WAS': 1610612764,
}

# Reverse: team ID to abbreviation
TEAM_ID_TO_ABR = {v: k for k, v in TEAM_ABR_TO_ID.items()}

# Abbreviation to short name (for MATCHUP string parsing)
TEAM_ABR_TO_SHORT = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards',
}

# League average defaults (fallback when coach not found)
LEAGUE_DEFAULTS = {
    'coach': 'Unknown',
    'rotation_depth': 9.0,
    'star_minutes_share': 0.42,
    'blowout_bench_rate': 0.15,
    'pace_tendency': 99.0,
}

_last_api_call = 0


def _rate_limit():
    """Respect nba_api rate limits."""
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < API_DELAY:
        time.sleep(API_DELAY - elapsed)
    _last_api_call = time.time()


# ═══════════════════════════════════════════════════════════════
# 1. GET COACH FOR A TEAM
# ═══════════════════════════════════════════════════════════════

def get_coach_roster(team_id, season='2025-26'):
    """
    Get head coach name for a team via nba_api CommonTeamRoster.

    Returns coach name string or None if not found.
    Caches results to predictions/cache/coaches_{season}.json.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f'coaches_{season}.json')

    # Check cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            team_id_str = str(team_id)
            if team_id_str in cached:
                return cached[team_id_str]
        except (json.JSONDecodeError, KeyError):
            pass

    # Fetch from API
    try:
        from nba_api.stats.endpoints import CommonTeamRoster
        _rate_limit()
        roster = CommonTeamRoster(team_id=team_id, season=season)
        dfs = roster.get_data_frames()
        if len(dfs) < 2:
            return None

        coaches_df = dfs[1]
        # IS_ASSISTANT=1 is head coach in nba_api (2=assistant, 3=trainer)
        head = coaches_df[coaches_df['COACH_TYPE'] == 'Head Coach']
        if head.empty:
            # Fallback: IS_ASSISTANT == 1
            head = coaches_df[coaches_df['IS_ASSISTANT'] == 1]
        if head.empty:
            return None

        coach_name = head.iloc[0]['COACH_NAME']

        # Update cache
        cached = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    cached = json.load(f)
            except (json.JSONDecodeError, KeyError):
                cached = {}

        cached[str(team_id)] = coach_name
        with open(cache_file, 'w') as f:
            json.dump(cached, f, indent=2)

        return coach_name

    except Exception as e:
        print(f"[WARN] coach_model: Failed to fetch coach for team {team_id}: {e}")
        return None


def get_all_coaches(season='2025-26'):
    """
    Fetch head coaches for all 30 teams. Returns {team_abr: coach_name}.
    Uses cache aggressively -- only calls API for missing teams.
    """
    cache_file = os.path.join(CACHE_DIR, f'coaches_{season}.json')
    cached = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                cached = json.load(f)
        except (json.JSONDecodeError, KeyError):
            cached = {}

    result = {}
    api_calls = 0

    for abr, team_id in TEAM_ABR_TO_ID.items():
        tid_str = str(team_id)
        if tid_str in cached:
            result[abr] = cached[tid_str]
        else:
            coach = get_coach_roster(team_id, season=season)
            if coach:
                result[abr] = coach
                api_calls += 1
            else:
                result[abr] = 'Unknown'

    if api_calls > 0:
        print(f"  [coach_model] Fetched {api_calls} coaches from API, "
              f"{len(result) - api_calls} from cache")

    return result


# ═══════════════════════════════════════════════════════════════
# 2. BUILD COACH PROFILES FROM CACHED GAME LOGS
# ═══════════════════════════════════════════════════════════════

def _load_all_game_logs(season='2025-26'):
    """
    Load all cached player game logs from parquet files.
    Returns list of (player_id, DataFrame) tuples.
    """
    pattern = os.path.join(CACHE_DIR, f'gamelog_*_{season}.parquet')
    files = glob.glob(pattern)

    logs = []
    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
            # Extract player_id from filename: gamelog_{pid}_{season}.parquet
            basename = os.path.basename(fpath)
            pid = int(basename.split('_')[1])
            logs.append((pid, df))
        except Exception:
            continue

    return logs


def _extract_team_abr(matchup):
    """
    Extract player's team abbreviation from MATCHUP string.
    'LAL vs. DEN' -> 'LAL', 'LAL @ DEN' -> 'LAL'
    """
    if not isinstance(matchup, str):
        return None
    parts = matchup.split()
    return parts[0] if parts else None


def _game_margin(row):
    """
    Estimate game margin from player plus_minus.
    This is a proxy -- true margin requires box score data.
    Returns absolute margin or 0 if unavailable.
    """
    pm = row.get('PLUS_MINUS', 0)
    if pd.isna(pm):
        return 0
    return abs(float(pm))


def build_coach_profiles(season='2025-26'):
    """
    Build coach tendency profiles from cached player game logs.

    For each team/coach, computes:
      - rotation_depth: players averaging 15+ minutes
      - star_minutes_share: % of team minutes going to top 3 players
      - blowout_bench_rate: how much star minutes drop in blowouts (15+ margin)
      - pace_tendency: average possessions proxy (FGA + 0.44*FTA + TOV per game)

    Caches to predictions/cache/coach_profiles_{season}.json.
    Returns {team_abr: {coach, rotation_depth, star_minutes_share, ...}}.
    """
    print("\n[coach_model] Building coach profiles from cached game logs...")

    # Step 1: Get coaches
    coaches = get_all_coaches(season=season)
    print(f"  Coaches loaded: {len(coaches)}")

    # Step 2: Load all cached game logs
    logs = _load_all_game_logs(season=season)
    print(f"  Game logs loaded: {len(logs)} players")

    if not logs:
        print("  [WARN] No cached game logs found. Returning defaults.")
        return {}

    # Step 3: Group players by team
    # Each player's team = most common team abbreviation in their MATCHUP strings
    team_players = defaultdict(list)  # team_abr -> [(pid, df), ...]

    for pid, df in logs:
        if df.empty or 'MATCHUP' not in df.columns or 'MIN' not in df.columns:
            continue

        # Determine primary team from most recent games
        recent = df.head(10)
        teams = recent['MATCHUP'].apply(_extract_team_abr).dropna()
        if teams.empty:
            continue

        primary_team = teams.mode().iloc[0] if len(teams) > 0 else None
        if primary_team and primary_team in TEAM_ABR_TO_ID:
            team_players[primary_team].append((pid, df))

    print(f"  Teams with player data: {len(team_players)}")

    # Step 4: Compute profiles per team
    profiles = {}

    for team_abr, player_logs in team_players.items():
        coach_name = coaches.get(team_abr, 'Unknown')

        # --- Rotation depth: how many players average 15+ MIN ---
        player_avg_mins = []
        for pid, df in player_logs:
            df_active = df[pd.to_numeric(df['MIN'], errors='coerce') >= 1]
            if len(df_active) < 5:
                continue
            avg_min = pd.to_numeric(df_active['MIN'], errors='coerce').mean()
            player_avg_mins.append((pid, avg_min))

        rotation_depth = sum(1 for _, avg in player_avg_mins if avg >= 15.0)

        # --- Star minutes share: % of team minutes from top 3 ---
        if player_avg_mins:
            sorted_by_mins = sorted(player_avg_mins, key=lambda x: x[1], reverse=True)
            total_avg_mins = sum(avg for _, avg in sorted_by_mins)
            top3_avg_mins = sum(avg for _, avg in sorted_by_mins[:3])
            star_minutes_share = round(top3_avg_mins / total_avg_mins, 3) if total_avg_mins > 0 else 0.42
        else:
            star_minutes_share = 0.42

        # --- Blowout bench rate ---
        # Compare star (top 3 by avg MIN) minutes in close vs blowout games
        # Blowout proxy: games where WL = W and PLUS_MINUS >= 15, or WL = L and PLUS_MINUS <= -15
        blowout_bench_rate = _compute_blowout_bench_rate(player_logs, player_avg_mins)

        # --- Pace tendency ---
        # Proxy: team-level (FGA + 0.44*FTA + TOV) per 48 minutes
        pace_tendency = _compute_pace_tendency(player_logs)

        profiles[team_abr] = {
            'coach': coach_name,
            'rotation_depth': rotation_depth,
            'star_minutes_share': round(star_minutes_share, 3),
            'blowout_bench_rate': round(blowout_bench_rate, 3),
            'pace_tendency': round(pace_tendency, 1),
        }

    # Step 5: Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f'coach_profiles_{season}.json')
    output = {
        'season': season,
        'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'profiles': profiles,
    }
    with open(cache_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Profiles built: {len(profiles)} teams")
    print(f"  Cached to: {cache_file}")

    # Print summary
    if profiles:
        sorted_depth = sorted(profiles.items(), key=lambda x: x[1]['rotation_depth'], reverse=True)
        print(f"\n  Deepest rotations:")
        for team, p in sorted_depth[:5]:
            print(f"    {team} ({p['coach']}): {p['rotation_depth']} players 15+ min, "
                  f"star share {p['star_minutes_share']:.1%}, "
                  f"blowout bench {p['blowout_bench_rate']:.1%}, "
                  f"pace {p['pace_tendency']}")

        sorted_star = sorted(profiles.items(), key=lambda x: x[1]['star_minutes_share'], reverse=True)
        print(f"\n  Highest star concentration:")
        for team, p in sorted_star[:5]:
            print(f"    {team} ({p['coach']}): star share {p['star_minutes_share']:.1%}, "
                  f"depth {p['rotation_depth']}")

    return profiles


def _compute_blowout_bench_rate(player_logs, player_avg_mins):
    """
    Compute how much star minutes drop in blowout games vs close games.

    Returns a ratio (0.0 = no benching, 0.30 = stars lose 30% of minutes in blowouts).
    """
    if not player_avg_mins or len(player_avg_mins) < 3:
        return 0.15  # league average default

    # Identify top 3 players by average minutes
    sorted_players = sorted(player_avg_mins, key=lambda x: x[1], reverse=True)
    star_pids = {pid for pid, _ in sorted_players[:3]}

    star_mins_close = []
    star_mins_blowout = []

    for pid, df in player_logs:
        if pid not in star_pids:
            continue

        df = df.copy()
        df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
        df['PLUS_MINUS'] = pd.to_numeric(df.get('PLUS_MINUS', 0), errors='coerce').fillna(0)
        df = df[df['MIN'] >= 1]

        if df.empty:
            continue

        # Split into close and blowout games
        # Blowout: absolute plus_minus >= 15 (proxy for game margin)
        close = df[df['PLUS_MINUS'].abs() < 15]
        blowout = df[df['PLUS_MINUS'].abs() >= 15]

        if not close.empty:
            star_mins_close.extend(close['MIN'].tolist())
        if not blowout.empty:
            star_mins_blowout.extend(blowout['MIN'].tolist())

    if not star_mins_close or not star_mins_blowout:
        return 0.15

    avg_close = np.mean(star_mins_close)
    avg_blowout = np.mean(star_mins_blowout)

    if avg_close <= 0:
        return 0.15

    # Bench rate: how much minutes drop in blowouts (positive = benched more)
    bench_rate = max(0.0, (avg_close - avg_blowout) / avg_close)
    return min(bench_rate, 0.50)  # cap at 50%


def _compute_pace_tendency(player_logs):
    """
    Compute team pace proxy from player game logs.

    Pace proxy: sum of (FGA + 0.44*FTA + TOV) across all players per game,
    then average across games. This approximates possessions per game.

    Returns possessions-per-game estimate.
    """
    # Collect per-game possession proxy for the team
    game_poss = defaultdict(float)  # game_id -> total possessions proxy
    game_counts = defaultdict(int)

    for pid, df in player_logs:
        df = df.copy()
        for col in ['FGA', 'FTA', 'TOV', 'MIN']:
            df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

        df = df[df['MIN'] >= 1]

        if 'Game_ID' not in df.columns:
            continue

        for _, row in df.iterrows():
            gid = row['Game_ID']
            poss = float(row['FGA']) + 0.44 * float(row['FTA']) + float(row['TOV'])
            game_poss[gid] += poss
            game_counts[gid] += 1

    if not game_poss:
        return 99.0  # league average default

    # Filter to games with reasonable player counts (at least 5 players)
    valid_poss = [poss for gid, poss in game_poss.items() if game_counts[gid] >= 5]

    if not valid_poss:
        return 99.0

    return np.mean(valid_poss)


# ═══════════════════════════════════════════════════════════════
# 3. GET COACH FEATURES FOR A TEAM
# ═══════════════════════════════════════════════════════════════

def _load_cached_profiles(season='2025-26'):
    """Load coach profiles from cache. Returns profiles dict or empty dict."""
    cache_file = os.path.join(CACHE_DIR, f'coach_profiles_{season}.json')
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file) as f:
            data = json.load(f)
        return data.get('profiles', {})
    except (json.JSONDecodeError, KeyError):
        return {}


def get_coach_features(team_abr, GAMES=None, season='2025-26'):
    """
    Return coach tendency features for a team.

    Args:
        team_abr: Team abbreviation (e.g., 'LAL', 'BOS')
        GAMES: Optional GAMES dict from pipeline (unused currently, reserved for
               future per-game coach adjustments like playoff rotation tightening)
        season: NBA season string

    Returns dict:
        {
            'coach': str,
            'rotation_depth': int,
            'star_minutes_share': float,
            'blowout_bench_rate': float,
            'pace_tendency': float,
        }

    Graceful fallback: returns league averages if team not found.
    """
    profiles = _load_cached_profiles(season=season)

    if team_abr in profiles:
        return profiles[team_abr]

    # Fallback: return league averages
    return dict(LEAGUE_DEFAULTS)


# ═══════════════════════════════════════════════════════════════
# 4. ENRICH PROP PREDICTIONS WITH COACH FEATURES
# ═══════════════════════════════════════════════════════════════

def _get_team_abr_from_pick(pick):
    """
    Extract player's team abbreviation from a pick dict.
    Uses the 'game' field (e.g., 'LAL@DEN') and 'is_home' flag.
    """
    game = pick.get('game', '')
    is_home = pick.get('is_home')

    if not game or '@' not in game:
        return None, None

    parts = game.split('@')
    if len(parts) != 2:
        return None, None

    away_abr = parts[0].strip()
    home_abr = parts[1].strip()

    if is_home is True:
        return home_abr, away_abr
    elif is_home is False:
        return away_abr, home_abr

    # Unknown side -- return both for opponent pace at least
    return None, None


def enrich_with_coach_features(results, GAMES=None, season='2025-26'):
    """
    Add coach tendency features to each prop prediction in results list.

    For each prop, adds:
      - coach_rotation_depth: Player's team rotation depth
      - coach_star_minutes_share: Player's team star concentration
      - coach_blowout_bench_rate: Player's team blowout benching tendency
      - coach_pace_tendency: Player's team pace
      - opp_coach_pace_tendency: Opponent's pace (affects game flow)

    All values default to 0 if unavailable (neutral signal for ML models).

    Returns count of enriched picks.
    """
    profiles = _load_cached_profiles(season=season)
    enriched = 0

    if not profiles:
        # No profiles cached -- set all to 0 (neutral)
        for pick in results:
            pick['coach_rotation_depth'] = 0
            pick['coach_star_minutes_share'] = 0
            pick['coach_blowout_bench_rate'] = 0
            pick['coach_pace_tendency'] = 0
            pick['opp_coach_pace_tendency'] = 0
        return 0

    for pick in results:
        team_abr, opp_abr = _get_team_abr_from_pick(pick)

        # Player's team coach features
        team_profile = profiles.get(team_abr, {}) if team_abr else {}
        pick['coach_rotation_depth'] = team_profile.get('rotation_depth', 0)
        pick['coach_star_minutes_share'] = team_profile.get('star_minutes_share', 0)
        pick['coach_blowout_bench_rate'] = team_profile.get('blowout_bench_rate', 0)
        pick['coach_pace_tendency'] = team_profile.get('pace_tendency', 0)

        # Opponent coach pace (affects game flow for all players)
        opp_profile = profiles.get(opp_abr, {}) if opp_abr else {}
        pick['opp_coach_pace_tendency'] = opp_profile.get('pace_tendency', 0)

        if team_abr and team_profile:
            enriched += 1

    return enriched


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    args = sys.argv[1:]

    if '--build' in args:
        profiles = build_coach_profiles()
        if profiles:
            print(f"\n[OK] Built {len(profiles)} coach profiles")
        else:
            print("\n[WARN] No profiles built -- check cache directory for game logs")

    elif '--test' in args:
        idx = args.index('--test')
        team = args[idx + 1].upper() if idx + 1 < len(args) else 'LAL'

        print(f"\n[TEST] Coach features for {team}:")
        features = get_coach_features(team)
        for k, v in features.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")

        # Also test enrichment on a mock pick
        mock_pick = {
            'player': 'Test Player',
            'game': f'{team}@BOS' if team != 'BOS' else 'LAL@BOS',
            'is_home': False if team != 'BOS' else True,
        }
        enrich_with_coach_features([mock_pick])
        print(f"\n  Enriched mock pick:")
        for k in ['coach_rotation_depth', 'coach_star_minutes_share',
                   'coach_blowout_bench_rate', 'coach_pace_tendency',
                   'opp_coach_pace_tendency']:
            print(f"    {k}: {mock_pick.get(k, 'N/A')}")

    elif '--coaches' in args:
        print("\n[FETCHING] All 30 head coaches...")
        coaches = get_all_coaches()
        for abr in sorted(coaches.keys()):
            print(f"  {abr}: {coaches[abr]}")

    else:
        print("Usage:")
        print("  python3 coach_model.py --build          Build all coach profiles")
        print("  python3 coach_model.py --test LAL       Test features for a team")
        print("  python3 coach_model.py --coaches        Fetch all 30 head coaches")

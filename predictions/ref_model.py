#!/usr/bin/env python3
"""
Referee Tendency Feature Extraction for NBA Prop Pipeline

Extracts referee assignments per game and computes ref-crew-specific features
that predict OVER/UNDER tendencies via foul rates and scoring environments.

Features produced (per game crew):
  - ref_crew_avg_fouls: Average total fouls called by this crew
  - ref_crew_avg_total: Average total points in games this crew officiates
  - ref_crew_over_rate: Historical % of games where total exceeded the line
  - ref_foul_rate_per_48: Fouls per 48 minutes for crew

Data flow:
  1. LeagueGameFinder provides game_ids + team PF/PTS per game (1 API call)
  2. BoxScoreSummaryV2 per game provides the 3 officials (1 call per game)
  3. Build/update ref_database.json incrementally
  4. enrich_with_ref_features() called by run_board_v5.py before ML scoring

Usage:
    python3 predictions/ref_model.py --build                    # Build ref DB from 2024-25 season
    python3 predictions/ref_model.py --build --season 2025-26   # Build from specific season
    python3 predictions/ref_model.py --test 0022400001          # Test single game
    python3 predictions/ref_model.py --stats                    # Print ref leaderboard
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

API_DELAY = 0.6  # seconds between nba_api calls (matches nba_fetcher.py)

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PREDICTIONS_DIR, 'cache')
REF_DB_PATH = os.path.join(CACHE_DIR, 'ref_database.json')

# League averages for neutral fallback (2024-25 approx)
LEAGUE_AVG_FOULS_PER_GAME = 40.0     # ~20 PF per team
LEAGUE_AVG_TOTAL_POINTS = 224.0      # ~112 per team
LEAGUE_AVG_FOUL_RATE_PER_48 = 20.0   # fouls per team per 48 min
LEAGUE_AVG_OVER_RATE = 0.50          # neutral

_last_api_call = 0


def _rate_limit():
    """Respect nba_api rate limits (0.6s between calls)."""
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < API_DELAY:
        time.sleep(API_DELAY - elapsed)
    _last_api_call = time.time()


# ═══════════════════════════════════════════════════════════════
# 1. GET GAME OFFICIALS
# ═══════════════════════════════════════════════════════════════

def get_game_officials(game_id):
    """
    Extract the 3 officials for a specific game using BoxScoreSummaryV2.

    Args:
        game_id: NBA game ID string (e.g., '0022400001')

    Returns:
        List of dicts: [{'official_id': int, 'name': str, 'jersey': str}, ...]
        Empty list on failure.
    """
    try:
        from nba_api.stats.endpoints import boxscoresummaryv2
    except ImportError:
        print("[WARN] nba_api not installed")
        return []

    _rate_limit()
    try:
        bs = boxscoresummaryv2.BoxScoreSummaryV2(game_id=str(game_id))
        dfs = bs.get_data_frames()
        officials_df = dfs[2]  # Officials table is at index 2

        if officials_df.empty:
            return []

        officials = []
        for _, row in officials_df.iterrows():
            officials.append({
                'official_id': int(row['OFFICIAL_ID']),
                'name': f"{row['FIRST_NAME']} {row['LAST_NAME']}".strip(),
                'jersey': str(row.get('JERSEY_NUM', '')).strip(),
            })
        return officials

    except Exception as e:
        # Graceful degradation -- never crash the pipeline
        print(f"[WARN] Failed to get officials for game {game_id}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# 2. BUILD REF DATABASE
# ═══════════════════════════════════════════════════════════════

def _load_ref_db():
    """Load existing ref database from disk, or return empty structure."""
    if os.path.exists(REF_DB_PATH):
        try:
            with open(REF_DB_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {
        'refs': {},
        'games_processed': [],
        'meta': {
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat(),
            'total_games': 0,
            'total_refs': 0,
        }
    }


def _save_ref_db(db):
    """Persist ref database to disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    db['meta']['updated'] = datetime.now().isoformat()
    db['meta']['total_games'] = len(db['games_processed'])
    db['meta']['total_refs'] = len(db['refs'])
    with open(REF_DB_PATH, 'w') as f:
        json.dump(db, f, indent=2)


def _fetch_season_games(season='2024-25'):
    """
    Fetch all completed regular season games for a season using LeagueGameFinder.
    Returns list of dicts with game_id, date, team fouls, team points.
    One API call retrieves all games (each game appears twice, once per team).
    """
    try:
        from nba_api.stats.endpoints import leaguegamefinder
    except ImportError:
        print("[ERROR] nba_api not installed")
        return []

    _rate_limit()
    try:
        lgf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            league_id_nullable='00',
            season_type_nullable='Regular Season',
        )
        df = lgf.get_data_frames()[0]
    except Exception as e:
        print(f"[ERROR] LeagueGameFinder failed for {season}: {e}")
        return []

    if df.empty:
        return []

    # Group by game_id to merge both teams into one record
    games = {}
    for _, row in df.iterrows():
        gid = row['GAME_ID']
        if gid not in games:
            games[gid] = {
                'game_id': gid,
                'date': row['GAME_DATE'],
                'teams': [],
                'total_fouls': 0,
                'total_points': 0,
            }
        # PF and PTS are per-team
        pf = int(row.get('PF', 0) or 0)
        pts = int(row.get('PTS', 0) or 0)
        games[gid]['teams'].append(row.get('TEAM_ABBREVIATION', ''))
        games[gid]['total_fouls'] += pf
        games[gid]['total_points'] += pts

    result = sorted(games.values(), key=lambda g: g['date'])
    print(f"  Found {len(result)} unique games for {season}")
    return result


def build_ref_database(season='2024-25', max_games=None, force=False):
    """
    Build or incrementally update the local ref database from historical games.

    Fetches all game IDs for the season via LeagueGameFinder (1 API call),
    then calls BoxScoreSummaryV2 per game to get officials (0.6s each).

    Args:
        season: NBA season string (e.g., '2024-25')
        max_games: Cap on number of new games to process (for partial builds)
        force: If True, reprocess already-processed games

    Returns:
        dict: Updated ref database
    """
    db = _load_ref_db()
    processed_set = set(db['games_processed'])

    print(f"\n{'='*60}")
    print(f"  REFEREE DATABASE BUILDER")
    print(f"  Season: {season}")
    print(f"  Existing: {len(db['refs'])} refs from {len(processed_set)} games")
    print(f"{'='*60}")

    # Step 1: Get all game IDs and team stats (single API call)
    print("\n[1/2] Fetching season game list...")
    season_games = _fetch_season_games(season)
    if not season_games:
        print("[WARN] No games found. Returning existing database.")
        return db

    # Filter to unprocessed games
    if not force:
        new_games = [g for g in season_games if g['game_id'] not in processed_set]
    else:
        new_games = season_games

    if max_games:
        new_games = new_games[:max_games]

    if not new_games:
        print("  All games already processed. Database is up to date.")
        return db

    print(f"  {len(new_games)} new games to process")

    # Step 2: Fetch officials per game (one API call each, 0.6s rate limit)
    print(f"\n[2/2] Fetching officials for {len(new_games)} games...")
    print(f"  Estimated time: {len(new_games) * 0.7:.0f}s ({len(new_games) * 0.7 / 60:.1f} min)")

    games_added = 0
    games_skipped = 0
    start_time = time.time()

    for i, game in enumerate(new_games):
        gid = game['game_id']

        officials = get_game_officials(gid)
        if not officials:
            games_skipped += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (len(new_games) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(new_games)}] {games_added} added, "
                      f"{games_skipped} skipped ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
            continue

        # Update ref stats for each official in this game
        total_fouls = game['total_fouls']
        total_points = game['total_points']
        game_date = game['date']

        for official in officials:
            ref_id = str(official['official_id'])

            if ref_id not in db['refs']:
                db['refs'][ref_id] = {
                    'name': official['name'],
                    'jersey': official['jersey'],
                    'games_officiated': 0,
                    'total_fouls_sum': 0,
                    'total_points_sum': 0,
                    'game_dates': [],
                }

            ref = db['refs'][ref_id]
            ref['games_officiated'] += 1
            ref['total_fouls_sum'] += total_fouls
            ref['total_points_sum'] += total_points
            ref['game_dates'].append(game_date)
            # Keep name/jersey updated to latest
            ref['name'] = official['name']
            ref['jersey'] = official['jersey']

        db['games_processed'].append(gid)
        games_added += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(new_games) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(new_games)}] {games_added} added, "
                  f"{games_skipped} skipped ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
            # Periodic save in case of interruption
            _save_ref_db(db)

    elapsed = time.time() - start_time

    # Compute derived stats for each ref
    for ref_id, ref in db['refs'].items():
        gp = ref['games_officiated']
        if gp > 0:
            ref['avg_total_fouls'] = round(ref['total_fouls_sum'] / gp, 1)
            ref['avg_total_points'] = round(ref['total_points_sum'] / gp, 1)
            # Fouls per 48 minutes (per team): total_fouls / 2 teams / games
            # Normalize: total fouls across both teams per 48 min game
            ref['foul_rate_per_48'] = round(ref['total_fouls_sum'] / gp, 1)
        else:
            ref['avg_total_fouls'] = LEAGUE_AVG_FOULS_PER_GAME
            ref['avg_total_points'] = LEAGUE_AVG_TOTAL_POINTS
            ref['foul_rate_per_48'] = LEAGUE_AVG_FOUL_RATE_PER_48

    _save_ref_db(db)

    print(f"\n  DONE: {games_added} games added, {games_skipped} skipped in {elapsed:.1f}s")
    print(f"  Database: {len(db['refs'])} refs from {len(db['games_processed'])} total games")

    return db


# ═══════════════════════════════════════════════════════════════
# 3. GET REF FEATURES FOR A GAME
# ═══════════════════════════════════════════════════════════════

def get_ref_features(game_id=None, officials=None, over_under_line=None):
    """
    Given a game's officials, compute crew-level referee tendency features.

    Args:
        game_id: NBA game ID (will fetch officials if not provided directly)
        officials: List of official dicts from get_game_officials()
                   (provide this to avoid an extra API call)
        over_under_line: Game total line for over_rate calculation

    Returns:
        dict with ref features:
          - ref_crew_avg_fouls: float (crew average of each ref's avg_total_fouls)
          - ref_crew_avg_total: float (crew average of each ref's avg_total_points)
          - ref_crew_over_rate: float (fraction of games crew's avg_total > line)
          - ref_foul_rate_per_48: float (crew average foul rate)
          - ref_crew_names: str (comma-separated names for display)
          - ref_crew_games: int (minimum games any crew member has)
          - ref_data_quality: str ('good', 'partial', 'none')
    """
    # Default neutral features (league averages)
    defaults = {
        'ref_crew_avg_fouls': LEAGUE_AVG_FOULS_PER_GAME,
        'ref_crew_avg_total': LEAGUE_AVG_TOTAL_POINTS,
        'ref_crew_over_rate': LEAGUE_AVG_OVER_RATE,
        'ref_foul_rate_per_48': LEAGUE_AVG_FOUL_RATE_PER_48,
        'ref_crew_names': '',
        'ref_crew_games': 0,
        'ref_data_quality': 'none',
    }

    # Get officials if not provided
    if officials is None and game_id:
        officials = get_game_officials(game_id)

    if not officials:
        return defaults

    # Load ref database
    db = _load_ref_db()
    refs_db = db.get('refs', {})

    if not refs_db:
        return defaults

    # Look up each official in the database
    crew_stats = []
    for official in officials:
        ref_id = str(official['official_id'])
        if ref_id in refs_db:
            crew_stats.append(refs_db[ref_id])

    if not crew_stats:
        # Officials assigned but none found in database
        defaults['ref_crew_names'] = ', '.join(o['name'] for o in officials)
        defaults['ref_data_quality'] = 'none'
        return defaults

    # Compute crew-level features (average across the 3 officials)
    crew_avg_fouls = sum(r['avg_total_fouls'] for r in crew_stats) / len(crew_stats)
    crew_avg_total = sum(r['avg_total_points'] for r in crew_stats) / len(crew_stats)
    crew_foul_rate = sum(r['foul_rate_per_48'] for r in crew_stats) / len(crew_stats)
    min_games = min(r['games_officiated'] for r in crew_stats)

    # Over rate: what fraction of games does this crew's environment exceed a line?
    # If no line provided, use league average as baseline
    line = over_under_line if over_under_line else LEAGUE_AVG_TOTAL_POINTS
    crew_over_rate = 1.0 if crew_avg_total > line else 0.0
    # More nuanced: how far above/below the line on average
    if line > 0:
        crew_over_rate = min(max((crew_avg_total - line) / 20.0 + 0.5, 0.0), 1.0)

    data_quality = 'good' if min_games >= 20 else ('partial' if min_games >= 5 else 'none')

    return {
        'ref_crew_avg_fouls': round(crew_avg_fouls, 1),
        'ref_crew_avg_total': round(crew_avg_total, 1),
        'ref_crew_over_rate': round(crew_over_rate, 3),
        'ref_foul_rate_per_48': round(crew_foul_rate, 1),
        'ref_crew_names': ', '.join(r['name'] for r in crew_stats),
        'ref_crew_games': min_games,
        'ref_data_quality': data_quality,
    }


# ═══════════════════════════════════════════════════════════════
# 4. ENRICH PICKS WITH REF FEATURES
# ═══════════════════════════════════════════════════════════════

def enrich_with_ref_features(results, GAMES=None):
    """
    Enrich a list of prop predictions with referee tendency features.
    Called by run_board_v5.py alongside correlation enrichment.

    For each prop, looks up the game's officials from the GAMES dict
    (if officials were pre-fetched) or falls back to neutral values.

    Args:
        results: List of prop prediction dicts (modified in place)
        GAMES: Dict of game contexts from game_researcher.py

    Returns:
        int: Number of picks enriched with real ref data (non-neutral)
    """
    db = _load_ref_db()
    refs_db = db.get('refs', {})

    if not refs_db:
        # No ref database built yet -- set neutral values for all picks
        for pick in results:
            pick['ref_crew_avg_fouls'] = 0.0
            pick['ref_crew_avg_total'] = 0.0
            pick['ref_crew_over_rate'] = 0.0
            pick['ref_foul_rate_per_48'] = 0.0
        return 0

    # Cache ref features per game to avoid recomputation
    game_ref_cache = {}
    enriched_count = 0

    for pick in results:
        game_key = pick.get('game', '')

        if game_key in game_ref_cache:
            ref_feats = game_ref_cache[game_key]
        else:
            # Try to get officials from GAMES dict if available
            officials = None
            over_under = None
            if GAMES and game_key in GAMES:
                gctx = GAMES[game_key]
                officials = gctx.get('officials')  # May not exist yet
                over_under = gctx.get('over_under')

            ref_feats = get_ref_features(
                officials=officials,
                over_under_line=float(over_under) if over_under else None,
            )
            game_ref_cache[game_key] = ref_feats

        # Write features to the pick dict
        pick['ref_crew_avg_fouls'] = ref_feats['ref_crew_avg_fouls']
        pick['ref_crew_avg_total'] = ref_feats['ref_crew_avg_total']
        pick['ref_crew_over_rate'] = ref_feats['ref_crew_over_rate']
        pick['ref_foul_rate_per_48'] = ref_feats['ref_foul_rate_per_48']

        if ref_feats['ref_data_quality'] != 'none':
            enriched_count += 1

    return enriched_count


# ═══════════════════════════════════════════════════════════════
# 5. REF LEADERBOARD / STATS
# ═══════════════════════════════════════════════════════════════

def print_ref_stats(min_games=20, top_n=20):
    """Print referee tendency leaderboard."""
    db = _load_ref_db()
    refs = db.get('refs', {})

    if not refs:
        print("No ref data. Run: python3 predictions/ref_model.py --build")
        return

    # Filter to refs with enough games
    qualified = {rid: r for rid, r in refs.items() if r['games_officiated'] >= min_games}

    print(f"\n{'='*80}")
    print(f"  REFEREE TENDENCY LEADERBOARD (min {min_games} games)")
    print(f"  Database: {len(refs)} refs, {len(qualified)} qualified, "
          f"{db['meta']['total_games']} games processed")
    print(f"{'='*80}")

    # Sort by avg_total_fouls descending (whistle-happy refs)
    sorted_refs = sorted(qualified.values(), key=lambda r: r.get('avg_total_fouls', 0), reverse=True)

    # Compute league averages from the database
    if qualified:
        db_avg_fouls = sum(r.get('avg_total_fouls', 0) for r in qualified.values()) / len(qualified)
        db_avg_pts = sum(r.get('avg_total_points', 0) for r in qualified.values()) / len(qualified)
    else:
        db_avg_fouls = LEAGUE_AVG_FOULS_PER_GAME
        db_avg_pts = LEAGUE_AVG_TOTAL_POINTS

    print(f"\n  League avg from DB: {db_avg_fouls:.1f} fouls/game, {db_avg_pts:.1f} pts/game")

    # Most whistle-happy refs
    print(f"\n  {'─'*75}")
    print(f"  TOP {top_n} MOST WHISTLE-HAPPY REFS (highest fouls/game)")
    print(f"  {'─'*75}")
    print(f"  {'Rank':<5} {'Name':<25} {'GP':<5} {'Avg Fouls':<12} {'Avg Pts':<12} {'vs Avg':<10}")
    print(f"  {'─'*75}")
    for i, ref in enumerate(sorted_refs[:top_n], 1):
        fouls = ref.get('avg_total_fouls', 0)
        pts = ref.get('avg_total_points', 0)
        delta = fouls - db_avg_fouls
        print(f"  {i:<5} {ref['name']:<25} {ref['games_officiated']:<5} "
              f"{fouls:<12.1f} {pts:<12.1f} {delta:+.1f}")

    # Least whistle-happy
    print(f"\n  {'─'*75}")
    print(f"  TOP {top_n} LEAST WHISTLE-HAPPY REFS (lowest fouls/game)")
    print(f"  {'─'*75}")
    print(f"  {'Rank':<5} {'Name':<25} {'GP':<5} {'Avg Fouls':<12} {'Avg Pts':<12} {'vs Avg':<10}")
    print(f"  {'─'*75}")
    sorted_low = sorted(qualified.values(), key=lambda r: r.get('avg_total_fouls', 0))
    for i, ref in enumerate(sorted_low[:top_n], 1):
        fouls = ref.get('avg_total_fouls', 0)
        pts = ref.get('avg_total_points', 0)
        delta = fouls - db_avg_fouls
        print(f"  {i:<5} {ref['name']:<25} {ref['games_officiated']:<5} "
              f"{fouls:<12.1f} {pts:<12.1f} {delta:+.1f}")

    # Highest scoring environments
    print(f"\n  {'─'*75}")
    print(f"  TOP {top_n} HIGHEST-SCORING ENVIRONMENTS")
    print(f"  {'─'*75}")
    print(f"  {'Rank':<5} {'Name':<25} {'GP':<5} {'Avg Pts':<12} {'Avg Fouls':<12} {'vs Avg':<10}")
    print(f"  {'─'*75}")
    sorted_pts = sorted(qualified.values(), key=lambda r: r.get('avg_total_points', 0), reverse=True)
    for i, ref in enumerate(sorted_pts[:top_n], 1):
        fouls = ref.get('avg_total_fouls', 0)
        pts = ref.get('avg_total_points', 0)
        delta = pts - db_avg_pts
        print(f"  {i:<5} {ref['name']:<25} {ref['games_officiated']:<5} "
              f"{pts:<12.1f} {fouls:<12.1f} {delta:+.1f}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    args = sys.argv[1:]

    if '--build' in args:
        season = '2024-25'
        if '--season' in args:
            idx = args.index('--season')
            if idx + 1 < len(args):
                season = args[idx + 1]
        max_games = None
        if '--max' in args:
            idx = args.index('--max')
            if idx + 1 < len(args):
                max_games = int(args[idx + 1])
        force = '--force' in args
        build_ref_database(season=season, max_games=max_games, force=force)

    elif '--test' in args:
        idx = args.index('--test')
        if idx + 1 < len(args):
            game_id = args[idx + 1]
        else:
            game_id = '0022400001'

        print(f"\nTesting game {game_id}...")
        officials = get_game_officials(game_id)
        if officials:
            print(f"  Officials: {', '.join(o['name'] for o in officials)}")
            features = get_ref_features(officials=officials)
            print(f"\n  Ref features:")
            for k, v in features.items():
                print(f"    {k}: {v}")
        else:
            print(f"  No officials found for game {game_id}")
            print(f"  (Officials may not be available for future/unplayed games)")

    elif '--stats' in args:
        min_games = 20
        if '--min-games' in args:
            idx = args.index('--min-games')
            if idx + 1 < len(args):
                min_games = int(args[idx + 1])
        print_ref_stats(min_games=min_games)

    else:
        print("Usage:")
        print("  python3 predictions/ref_model.py --build                    # Build ref DB from 2024-25")
        print("  python3 predictions/ref_model.py --build --season 2025-26   # Specific season")
        print("  python3 predictions/ref_model.py --build --max 50           # Process max 50 games")
        print("  python3 predictions/ref_model.py --build --force            # Reprocess all games")
        print("  python3 predictions/ref_model.py --test 0022400001          # Test single game")
        print("  python3 predictions/ref_model.py --stats                    # Ref leaderboard")
        print("  python3 predictions/ref_model.py --stats --min-games 30     # Custom min games")

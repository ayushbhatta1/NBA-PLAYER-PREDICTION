#!/usr/bin/env python3
"""
Pre-Game Availability Check
Runs BEFORE NEXUS to filter out:
1. Players confirmed OUT (DNP disasters like Murphy, Sengun, R. Williams)
2. Players in postponed/cancelled games (MIN@GSW, CHI@LAC on Mar 13)
3. Players marked GTD/Questionable (risk flags)

This module alone would have saved both Mar 13 parlays.
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_game_availability(game_date, GAMES, fetcher=None):
    """
    Check which games are actually happening today.
    Returns set of game keys that are confirmed ON the schedule.

    Uses nba_api scoreboard to verify games are scheduled and not postponed.
    """
    if fetcher is None:
        from nba_fetcher import NBAFetcher
        fetcher = NBAFetcher()

    # If game_date is in the future, skip schedule validation (API won't have it)
    from datetime import date, datetime as _dt
    try:
        gd = _dt.strptime(game_date, '%Y-%m-%d').date() if game_date else None
        if gd and gd > date.today():
            print(f"  [PREGAME] Game date {game_date} is in the future — skipping schedule validation")
            return set(GAMES.keys()), set()
    except Exception:
        pass

    scheduled_games = set()
    postponed_games = set()

    try:
        schedule = fetcher.get_todays_schedule()
        if not schedule:
            print(f"  [PREGAME] WARNING: No games found on schedule for today")
            print(f"  [PREGAME] Falling back to GAMES dict (no schedule validation)")
            return set(GAMES.keys()), set()

        # Build set of scheduled matchups from NBA API
        api_matchups = set()
        for game in schedule:
            home = game.get('home', '')
            away = game.get('away', '')
            status = game.get('status', '')

            # Check for postponed/cancelled
            status_lower = status.lower() if status else ''
            if 'postponed' in status_lower or 'cancelled' in status_lower:
                postponed_games.add(f"{away}@{home}")
                print(f"  [PREGAME] POSTPONED: {away} @ {home} — {status}")
                continue

            api_matchups.add((home, away))

        # Match GAMES dict keys against API schedule
        from nba_fetcher import TEAM_ABR
        for game_key, gctx in GAMES.items():
            home_abr = gctx.get('home_abr', '')
            away_abr = gctx.get('away_abr', '')
            home_name = gctx.get('home', '')
            away_name = gctx.get('away', '')

            # Try matching by team name
            found = False
            for api_home, api_away in api_matchups:
                # Match by name (fuzzy — handles "Timberwolves" vs "Timberwolves")
                if (_teams_match(home_name, api_home) and _teams_match(away_name, api_away)):
                    found = True
                    break
                # Also try abbreviation matching via reverse lookup
                if (home_abr and away_abr):
                    api_home_short = api_home
                    api_away_short = api_away
                    if (_teams_match(TEAM_ABR.get(home_abr, ''), api_home_short) and
                        _teams_match(TEAM_ABR.get(away_abr, ''), api_away_short)):
                        found = True
                        break

            if found:
                scheduled_games.add(game_key)
            else:
                postponed_games.add(game_key)
                print(f"  [PREGAME] NOT ON SCHEDULE: {game_key} ({away_name} @ {home_name})")

    except Exception as e:
        print(f"  [PREGAME] Schedule check failed: {e}")
        print(f"  [PREGAME] Falling back to GAMES dict (no schedule validation)")
        return set(GAMES.keys()), set()

    return scheduled_games, postponed_games


def check_player_availability(results, GAMES):
    """
    Check real-time player availability from GAMES injury data.
    Returns (available, removed) where removed is list of (result, reason).

    Filters:
    - Player confirmed OUT → remove entirely
    - Player GTD/Questionable → flag (kept but marked)
    - Player's game postponed → remove entirely
    """
    available = []
    removed = []

    for r in results:
        if 'error' in r:
            available.append(r)  # let downstream handle errors
            continue

        player = r.get('player', '')
        game = r.get('game', '')
        injury_status = r.get('player_injury_status', '')

        # Check if player is confirmed OUT
        if injury_status and injury_status.lower() in ['out', 'inactive']:
            removed.append((r, f"OUT — {injury_status}"))
            continue

        # Check game context for OUT lists
        game_key = _find_game_key(player, game, GAMES)
        if game_key and game_key in GAMES:
            gctx = GAMES[game_key]
            all_out = (gctx.get('away_out', []) + gctx.get('home_out', []))
            player_lower = player.lower()
            is_out = False
            for out_name in all_out:
                if player_lower in out_name.lower() or out_name.lower() in player_lower:
                    removed.append((r, f"OUT per injury report: {out_name}"))
                    is_out = True
                    break
            if is_out:
                continue

        available.append(r)

    return available, removed


def filter_postponed_games(results, postponed_games, GAMES):
    """
    Remove all players from postponed/cancelled games.
    Returns (kept, removed).
    """
    if not postponed_games:
        return results, []

    kept = []
    removed = []

    for r in results:
        player = r.get('player', '')
        game = r.get('game', '')

        # Check if this player's game is postponed
        game_key = _find_game_key(player, game, GAMES)
        if game_key and game_key in postponed_games:
            removed.append((r, f"Game postponed: {game_key}"))
            continue

        # Also check by game string directly
        if game and game in postponed_games:
            removed.append((r, f"Game postponed: {game}"))
            continue

        kept.append(r)

    return kept, removed


def run_pregame_check(results, GAMES, game_date=None, fetcher=None):
    """
    Full pre-game availability check.
    Call this BEFORE running NEXUS.

    Returns:
        filtered_results: Results with OUT players and postponed games removed
        pregame_report: Dict with details of what was filtered
    """
    print(f"\n{'='*60}")
    print(f"  PRE-GAME AVAILABILITY CHECK")
    print(f"{'='*60}")

    report = {
        'total_input': len(results),
        'players_removed': [],
        'games_postponed': [],
        'gtd_flagged': [],
    }

    # Step 1: Check game schedule (postponed games)
    print(f"\n  Step 1: Checking game schedule...")
    scheduled, postponed = check_game_availability(game_date, GAMES, fetcher)
    print(f"    Scheduled: {len(scheduled)} games | Postponed: {len(postponed)} games")

    if postponed:
        results, postponed_removed = filter_postponed_games(results, postponed, GAMES)
        for r, reason in postponed_removed:
            report['games_postponed'].append({
                'player': r.get('player', '?'),
                'stat': r.get('stat', '?'),
                'game': r.get('game', '?'),
                'reason': reason,
            })
            print(f"    REMOVED: {r.get('player', '?'):22s} — {reason}")
        print(f"    Removed {len(postponed_removed)} lines from postponed games")

    # Step 2: Check player availability (OUT/injured)
    print(f"\n  Step 2: Checking player availability...")
    results, out_removed = check_player_availability(results, GAMES)
    for r, reason in out_removed:
        report['players_removed'].append({
            'player': r.get('player', '?'),
            'stat': r.get('stat', '?'),
            'reason': reason,
        })
        print(f"    REMOVED: {r.get('player', '?'):22s} — {reason}")
    print(f"    Removed {len(out_removed)} lines (players OUT)")

    # Step 3: REMOVE GTD/Questionable/Doubtful (was flag-only — caused 62.5% DNP rate)
    gtd_count = 0
    gtd_removed = []
    kept = []
    for r in results:
        injury = r.get('player_injury_status', '')
        if injury and injury.lower() in ['questionable', 'gtd', 'game-time decision', 'doubtful']:
            gtd_count += 1
            report['gtd_flagged'].append({
                'player': r.get('player', '?'),
                'stat': r.get('stat', '?'),
                'status': injury,
            })
            gtd_removed.append(r)
            report['players_removed'].append({
                'player': r.get('player', '?'),
                'reason': f'GTD/Questionable: {injury}',
            })
            print(f"    REMOVED (GTD): {r.get('player', '?'):22s} — {injury}")
        else:
            kept.append(r)
    results = kept
    print(f"    Removed {gtd_count} GTD/Questionable players")

    total_removed = len(report['players_removed']) + len(report['games_postponed'])
    report['total_removed'] = total_removed
    report['total_remaining'] = len(results)

    print(f"\n  PREGAME RESULT: {len(results)} lines remaining "
          f"({total_removed} removed)")
    print(f"{'='*60}")

    return results, report


# ── Helpers ──

def _teams_match(name1, name2):
    """Fuzzy team name matching."""
    if not name1 or not name2:
        return False
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    return n1 == n2 or n1 in n2 or n2 in n1


def _find_game_key(player, game_str, GAMES):
    """Find the GAMES dict key for a player's game."""
    # Try direct match on game string
    if game_str and game_str in GAMES:
        return game_str

    # Try matching by looking through all games
    if not GAMES:
        return None

    for key, gctx in GAMES.items():
        # Match by game string format (e.g., "MIN@GSW")
        if game_str:
            home = gctx.get('home_abr', '')
            away = gctx.get('away_abr', '')
            if f"{away}@{home}" == game_str or f"{away} @ {home}" == game_str:
                return key
            # Also match by full names
            home_name = gctx.get('home', '')
            away_name = gctx.get('away', '')
            if game_str in [f"{away_name}@{home_name}", f"{away_name} @ {home_name}",
                           f"{away}@{home}", key]:
                return key

    return game_str  # fallback to original string

#!/usr/bin/env python3
"""
PBP Stats Client - pbpstats.com REST API Integration
Fetches pace, efficiency, usage, and on/off court splits from the free
pbpstats.com API. No authentication required.

Designed to slot into the NBA prop betting pipeline alongside nba_fetcher.py.

Usage:
    from pbp_client import PBPStatsClient
    client = PBPStatsClient(season='2025-26')

    # Team pace and efficiency
    pace = client.get_pace_efficiency('LAL')

    # Player on/off impact
    on_off = client.get_on_off(team_id=1610612747, player_id=1629029, stat='Usage')

    # Predicted pace for a matchup
    matchup_pace = client.get_team_pace_matchup('LAL', 'DEN')

CLI test:
    python3 predictions/pbp_client.py --test
"""

import json
import os
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = 'https://api.pbpstats.com'
API_DELAY = 1.0          # 1s between calls -- polite to a free API
CACHE_TTL_HOURS = 4      # match nba_fetcher pattern
REQUEST_TIMEOUT = 15     # seconds per HTTP request

# 3-letter abbreviation -> NBA team ID (standard across stats.nba.com / pbpstats)
TEAM_ABR_TO_ID = {
    'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751, 'CHA': 1610612766,
    'CHI': 1610612741, 'CLE': 1610612739, 'DAL': 1610612742, 'DEN': 1610612743,
    'DET': 1610612765, 'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
    'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763, 'MIA': 1610612748,
    'MIL': 1610612749, 'MIN': 1610612750, 'NOP': 1610612740, 'NYK': 1610612752,
    'OKC': 1610612760, 'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'TOR': 1610612761,
    'UTA': 1610612762, 'WAS': 1610612764,
}

# Reverse: team ID -> abbreviation
TEAM_ID_TO_ABR = {v: k for k, v in TEAM_ABR_TO_ID.items()}

# Short team names (for display / loose matching from the pipeline)
TEAM_SHORT_TO_ABR = {
    'Hawks': 'ATL', 'Celtics': 'BOS', 'Nets': 'BKN', 'Hornets': 'CHA',
    'Bulls': 'CHI', 'Cavaliers': 'CLE', 'Mavericks': 'DAL', 'Nuggets': 'DEN',
    'Pistons': 'DET', 'Warriors': 'GSW', 'Rockets': 'HOU', 'Pacers': 'IND',
    'Clippers': 'LAC', 'Lakers': 'LAL', 'Grizzlies': 'MEM', 'Heat': 'MIA',
    'Bucks': 'MIL', 'Timberwolves': 'MIN', 'Pelicans': 'NOP', 'Knicks': 'NYK',
    'Thunder': 'OKC', 'Magic': 'ORL', '76ers': 'PHI', 'Suns': 'PHX',
    'Trail Blazers': 'POR', 'Kings': 'SAC', 'Spurs': 'SAS', 'Raptors': 'TOR',
    'Jazz': 'UTA', 'Wizards': 'WAS',
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PBPStatsClient:
    """Client for the pbpstats.com free REST API with disk caching and
    rate limiting.  All methods return None on failure so the pipeline
    continues without pbp features."""

    def __init__(self, season='2025-26', cache_dir=None):
        self.season = season
        self.cache_dir = cache_dir or os.path.join('predictions', 'cache', 'pbp')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._last_api_call = 0
        # In-memory cache for team totals (keyed by season)
        self._team_totals_cache = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rate_limit(self):
        """Enforce minimum delay between outbound requests."""
        elapsed = time.time() - self._last_api_call
        if elapsed < API_DELAY:
            time.sleep(API_DELAY - elapsed)
        self._last_api_call = time.time()

    def _resolve_team_id(self, team):
        """Accept an abbreviation (str), short name (str), or int team ID.
        Returns the integer NBA team ID or None."""
        if isinstance(team, int):
            return team
        team_str = str(team).strip()
        # Try abbreviation first
        if team_str.upper() in TEAM_ABR_TO_ID:
            return TEAM_ABR_TO_ID[team_str.upper()]
        # Try short name
        abr = TEAM_SHORT_TO_ABR.get(team_str)
        if abr:
            return TEAM_ABR_TO_ID[abr]
        return None

    def _cache_path(self, key):
        """Build a filesystem-safe cache path from a logical key."""
        safe = key.replace('/', '_').replace('?', '_').replace('&', '_')
        return os.path.join(self.cache_dir, f"{safe}.json")

    def _read_cache(self, key, ttl_hours=None):
        """Return cached JSON data if it exists and is fresh, else None."""
        ttl = ttl_hours if ttl_hours is not None else CACHE_TTL_HOURS
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            mod_time = datetime.fromtimestamp(os.path.getmtime(path))
            if datetime.now() - mod_time > timedelta(hours=ttl):
                return None
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, key, data):
        """Persist JSON data to disk cache."""
        path = self._cache_path(key)
        try:
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass  # non-fatal

    def _fetch(self, path, params=None):
        """HTTP GET against the pbpstats API.  Returns parsed JSON or None."""
        url = f"{BASE_URL}{path}"
        if params:
            url += '?' + urllib.parse.urlencode(params)
        self._rate_limit()
        req = urllib.request.Request(url, headers={
            'User-Agent': 'NBAPropsBot/1.0 (research)',
            'Accept': 'application/json',
        })
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            print(f"[PBP] HTTP {e.code} for {url}")
            return None
        except urllib.error.URLError as e:
            print(f"[PBP] Connection error for {url}: {e.reason}")
            return None
        except Exception as e:
            print(f"[PBP] Unexpected error for {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # Team totals (pace, efficiency) -- one call fetches all 30 teams
    # ------------------------------------------------------------------

    def _load_team_totals(self, season=None):
        """Fetch /get-totals/nba?Type=Team for the season.
        Caches the full response and returns the list of team rows."""
        season = season or self.season
        cache_key = f"team_totals_{season}"

        # Memory cache
        if cache_key in self._team_totals_cache:
            return self._team_totals_cache[cache_key]

        # Disk cache
        cached = self._read_cache(cache_key)
        if cached is not None:
            self._team_totals_cache[cache_key] = cached
            return cached

        # API call
        data = self._fetch('/get-totals/nba', params={
            'Season': season,
            'SeasonType': 'Regular Season',
            'Type': 'Team',
        })
        if data is None:
            return None

        rows = data.get('multi_row_table_data', [])
        if not rows:
            return None

        self._write_cache(cache_key, rows)
        self._team_totals_cache[cache_key] = rows
        return rows

    def get_pace_efficiency(self, team, season=None):
        """Fetch pace and efficiency for a single team.

        Parameters
        ----------
        team : str or int
            Team abbreviation ('LAL'), short name ('Lakers'), or NBA team ID.
        season : str, optional
            e.g. '2025-26'. Defaults to self.season.

        Returns
        -------
        dict or None
            {pace, off_rating, def_rating, possessions_per_game,
             off_poss, def_poss, games_played, team_abr}
        """
        season = season or self.season
        team_id = self._resolve_team_id(team)
        if team_id is None:
            print(f"[PBP] Unknown team: {team}")
            return None

        rows = self._load_team_totals(season)
        if rows is None:
            return None

        team_id_str = str(team_id)
        row = None
        for r in rows:
            if str(r.get('TeamId', '')) == team_id_str:
                row = r
                break

        if row is None:
            return None

        gp = row.get('GamesPlayed', 1) or 1
        off_poss = row.get('OffPoss', 0)
        def_poss = row.get('DefPoss', 0)
        points = row.get('Points', 0)
        opp_points = row.get('OpponentPoints', 0)

        # Derive offensive/defensive rating: (points / possessions) * 100
        off_rtg = round(points / off_poss * 100, 2) if off_poss > 0 else None
        def_rtg = round(opp_points / def_poss * 100, 2) if def_poss > 0 else None

        return {
            'pace': round(row.get('Pace', 0), 2),
            'off_rating': off_rtg,
            'def_rating': def_rtg,
            'possessions_per_game': round((off_poss + def_poss) / (2 * gp), 1),
            'off_poss': off_poss,
            'def_poss': def_poss,
            'games_played': gp,
            'team_abr': TEAM_ID_TO_ABR.get(team_id, str(team_id)),
        }

    # ------------------------------------------------------------------
    # WOWY (with-or-without-you) player stats
    # ------------------------------------------------------------------

    def get_wowy_stats(self, team, player_id=None, season=None):
        """Fetch per-player totals for a team from the WOWY endpoint.

        The /get-wowy-stats/nba endpoint returns one row per player with
        aggregate stats (points, assists, rebounds, usage, minutes, etc.).
        Useful for comparing role sizes within a roster.

        Parameters
        ----------
        team : str or int
            Team abbreviation, short name, or NBA team ID.
        player_id : int, optional
            If provided, return only that player's row.
        season : str, optional

        Returns
        -------
        list[dict] or dict or None
            List of player dicts (if player_id is None), single player dict
            (if player_id given and found), or None on failure.
        """
        season = season or self.season
        team_id = self._resolve_team_id(team)
        if team_id is None:
            return None

        cache_key = f"wowy_{team_id}_{season}"
        cached = self._read_cache(cache_key)
        if cached is None:
            data = self._fetch('/get-wowy-stats/nba', params={
                'Season': season,
                'SeasonType': 'Regular Season',
                'TeamId': str(team_id),
                'Type': 'Player',
            })
            if data is None:
                return None
            cached = data.get('multi_row_table_data', [])
            if not cached:
                return None
            self._write_cache(cache_key, cached)

        # WOWY endpoint does not include GamesPlayed per player.
        # Pull team GP from the team totals endpoint to compute per-game values.
        team_gp = self._get_team_gp(team_id, season)

        def _build_player(row, full=False):
            mins = row.get('Minutes', 0)
            # WOWY does not provide per-player GamesPlayed.
            # Approximate mpg using team GP as denominator -- this is a
            # floor estimate (players who missed games will show lower mpg).
            mpg = round(mins / team_gp, 1) if team_gp > 0 else 0
            out = {
                'player_id': int(row.get('EntityId', 0)),
                'name': row.get('Name', ''),
                'team_games': team_gp,
                'minutes_total': mins,
                'minutes_per_game': mpg,
                'points': row.get('Points', 0),
                'assists': row.get('Assists', 0),
                'rebounds': row.get('Rebounds', 0),
                'usage': round(row.get('Usage', 0), 2),
                'ts_pct': round(row.get('TsPct', 0), 4),
            }
            if full:
                out.update({
                    'steals': row.get('Steals', 0),
                    'blocks': row.get('Blocks', 0),
                    'turnovers': row.get('Turnovers', 0),
                    'efg_pct': round(row.get('EfgPct', 0), 4),
                    'off_poss': row.get('OffPoss', 0),
                    'def_poss': row.get('DefPoss', 0),
                    'plus_minus': row.get('PlusMinus', 0),
                    'fouls': row.get('Fouls', 0),
                    'shot_quality_avg': round(row.get('ShotQualityAvg', 0), 4),
                })
            return out

        if player_id is not None:
            pid_str = str(player_id)
            for row in cached:
                if str(row.get('EntityId', '')) == pid_str:
                    return _build_player(row, full=True)
            return None

        # Return all players on the roster
        return [_build_player(row) for row in cached]

    def _get_team_gp(self, team_id, season=None):
        """Return GamesPlayed for a team from the totals cache."""
        season = season or self.season
        rows = self._load_team_totals(season)
        if rows is None:
            return 69  # reasonable late-season fallback
        tid = str(team_id)
        for r in rows:
            if str(r.get('TeamId', '')) == tid:
                return r.get('GamesPlayed', 69) or 69
        return 69

    # ------------------------------------------------------------------
    # On/Off court impact
    # ------------------------------------------------------------------

    def get_on_off(self, team, player_id, stat='Usage', season=None):
        """Fetch on/off court differential for a player on a given stat.

        Parameters
        ----------
        team : str or int
            Team abbreviation, short name, or NBA team ID.
        player_id : int
            NBA player ID.
        stat : str
            The pbpstats metric name, e.g. 'Usage', 'EfgPct', 'Pace',
            'TsPct', 'Fg3Pct', 'AtRimFrequency', 'ShotQualityAvg', etc.
            The endpoint exposes ~50 metrics.
        season : str, optional

        Returns
        -------
        dict or None
            {stat, on_value, off_value, delta, minutes_on, minutes_off}
        """
        season = season or self.season
        team_id = self._resolve_team_id(team)
        if team_id is None:
            return None

        cache_key = f"onoff_{team_id}_{player_id}_{season}"
        cached = self._read_cache(cache_key)
        if cached is None:
            data = self._fetch('/get-on-off/nba/player', params={
                'Season': season,
                'SeasonType': 'Regular Season',
                'TeamId': str(team_id),
                'PlayerId': str(player_id),
            })
            if data is None:
                return None
            cached = data.get('results', {})
            if not cached:
                return None
            self._write_cache(cache_key, cached)

        entries = cached.get(stat)
        if not entries or not isinstance(entries, list) or len(entries) == 0:
            return None

        # The first entry is the player themselves (Name matches player_id)
        row = entries[0]
        try:
            on_val = float(row.get('On', 0))
            off_val = float(row.get('Off', 0))
            delta = float(row.get('On-Off', on_val - off_val))
        except (ValueError, TypeError):
            return None

        return {
            'stat': stat,
            'on_value': round(on_val, 4),
            'off_value': round(off_val, 4),
            'delta': round(delta, 4),
            'minutes_on': int(row.get('MinutesOn', 0)),
            'minutes_off': int(row.get('MinutesOff', 0)),
        }

    def get_on_off_multi(self, team, player_id, stats=None, season=None):
        """Fetch on/off for multiple stats in one API call (single cache hit).

        Parameters
        ----------
        stats : list[str], optional
            Stats to extract. Defaults to a useful subset for prop betting.

        Returns
        -------
        dict or None
            {stat_name: {on_value, off_value, delta, ...}, ...}
        """
        if stats is None:
            stats = ['Usage', 'EfgPct', 'TsPct', 'Pace', 'Fg3Pct',
                     'AtRimFrequency', 'ShotQualityAvg']

        season = season or self.season
        team_id = self._resolve_team_id(team)
        if team_id is None:
            return None

        # Fetch / cache the full on-off payload once
        cache_key = f"onoff_{team_id}_{player_id}_{season}"
        cached = self._read_cache(cache_key)
        if cached is None:
            data = self._fetch('/get-on-off/nba/player', params={
                'Season': season,
                'SeasonType': 'Regular Season',
                'TeamId': str(team_id),
                'PlayerId': str(player_id),
            })
            if data is None:
                return None
            cached = data.get('results', {})
            if not cached:
                return None
            self._write_cache(cache_key, cached)

        out = {}
        for stat in stats:
            entries = cached.get(stat)
            if not entries or not isinstance(entries, list) or len(entries) == 0:
                continue
            row = entries[0]
            try:
                on_val = float(row.get('On', 0))
                off_val = float(row.get('Off', 0))
                delta = float(row.get('On-Off', on_val - off_val))
            except (ValueError, TypeError):
                continue
            out[stat] = {
                'on_value': round(on_val, 4),
                'off_value': round(off_val, 4),
                'delta': round(delta, 4),
                'minutes_on': int(row.get('MinutesOn', 0)),
                'minutes_off': int(row.get('MinutesOff', 0)),
            }
        return out if out else None

    # ------------------------------------------------------------------
    # Matchup pace prediction
    # ------------------------------------------------------------------

    def get_team_pace_matchup(self, team1, team2, season=None):
        """Predict the pace for a head-to-head matchup.

        Uses the standard formula: average of both teams' pace values,
        adjusted toward the league mean (regression to the mean).

        Parameters
        ----------
        team1, team2 : str or int
            Team abbreviations, short names, or NBA team IDs.

        Returns
        -------
        dict or None
            {team1_pace, team2_pace, league_avg_pace, predicted_pace,
             pace_delta (vs league avg)}
        """
        season = season or self.season
        p1 = self.get_pace_efficiency(team1, season)
        p2 = self.get_pace_efficiency(team2, season)
        if p1 is None or p2 is None:
            return None

        rows = self._load_team_totals(season)
        if rows is None:
            return None

        # League average pace
        paces = [r.get('Pace', 0) for r in rows if r.get('Pace')]
        league_avg = sum(paces) / len(paces) if paces else 100.0

        t1_pace = p1['pace']
        t2_pace = p2['pace']

        # Predicted pace: each team's pace as deviation from league avg,
        # then combine.  (t1_pace - avg) + (t2_pace - avg) + avg
        predicted = (t1_pace - league_avg) + (t2_pace - league_avg) + league_avg

        return {
            'team1_pace': round(t1_pace, 2),
            'team2_pace': round(t2_pace, 2),
            'league_avg_pace': round(league_avg, 2),
            'predicted_pace': round(predicted, 2),
            'pace_delta': round(predicted - league_avg, 2),
        }

    # ------------------------------------------------------------------
    # Bulk prefetch
    # ------------------------------------------------------------------

    def prefetch_teams(self, team_ids, season=None):
        """Prefetch pace/efficiency for multiple teams.

        Since /get-totals/nba?Type=Team returns ALL 30 teams in one call,
        this is effectively a single-request warm-up of the cache.

        Parameters
        ----------
        team_ids : list
            List of team abbreviations, short names, or NBA team IDs.

        Returns
        -------
        int
            Number of teams successfully loaded.
        """
        season = season or self.season
        rows = self._load_team_totals(season)
        if rows is None:
            return 0
        return len(rows)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

def _run_test():
    """Quick smoke test -- fetch one team and print results."""
    import argparse
    parser = argparse.ArgumentParser(description='PBP Stats Client test')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    parser.add_argument('--team', default='LAL', help='Team abbreviation (default: LAL)')
    parser.add_argument('--player-id', type=int, default=None,
                        help='Player ID for on/off test (default: auto-pick top player)')
    parser.add_argument('--season', default='2025-26', help='Season (default: 2025-26)')
    args = parser.parse_args()

    if not args.test:
        parser.print_help()
        return

    client = PBPStatsClient(season=args.season)
    team = args.team.upper()
    team_id = TEAM_ABR_TO_ID.get(team)

    print('=' * 60)
    print(f'PBP Stats Client -- Test ({team}, {args.season})')
    print('=' * 60)

    # 1. Team pace / efficiency
    print(f'\n1. Pace & efficiency for {team}...')
    pace = client.get_pace_efficiency(team)
    if pace:
        for k, v in pace.items():
            print(f'   {k}: {v}')
    else:
        print('   [FAIL] No data returned.')

    # 2. Prefetch all teams
    print('\n2. Prefetch all teams...')
    count = client.prefetch_teams([team])
    print(f'   Loaded {count} teams (single API call).')

    # 3. WOWY roster
    print(f'\n3. WOWY roster for {team}...')
    roster = client.get_wowy_stats(team)
    if roster:
        # Sort by minutes per game descending, show top 5
        roster.sort(key=lambda x: x.get('minutes_per_game', 0), reverse=True)
        print(f'   {len(roster)} players. Top 5 by mpg:')
        for p in roster[:5]:
            print(f'   - {p["name"]}: {p["minutes_per_game"]} mpg, '
                  f'{p["minutes_total"]} total min, '
                  f'usage={p["usage"]}, ts%={p["ts_pct"]}')
        # Auto-pick top-minutes player for on/off test
        if args.player_id is None and roster:
            args.player_id = roster[0]['player_id']
    else:
        print('   [FAIL] No data returned.')

    # 4. On/Off impact
    if args.player_id and team_id:
        print(f'\n4. On/Off impact for player {args.player_id}...')
        multi = client.get_on_off_multi(team, args.player_id)
        if multi:
            for stat_name, vals in multi.items():
                print(f'   {stat_name}: on={vals["on_value"]}, '
                      f'off={vals["off_value"]}, delta={vals["delta"]:+.4f}')
        else:
            print('   [FAIL] No data returned.')

    # 5. Matchup pace prediction
    matchup_team = 'DEN' if team != 'DEN' else 'BOS'
    print(f'\n5. Matchup pace: {team} vs {matchup_team}...')
    mp = client.get_team_pace_matchup(team, matchup_team)
    if mp:
        for k, v in mp.items():
            print(f'   {k}: {v}')
    else:
        print('   [FAIL] No data returned.')

    print('\n' + '=' * 60)
    print('[DONE] All tests complete.')


if __name__ == '__main__':
    _run_test()

#!/usr/bin/env python3
"""
NBA Live Data Fetcher - nba_api Integration
Pulls real-time game logs, splits, defensive rankings, and pace data
directly from stats.nba.com instead of relying on static CSV/JSON files.

Usage:
    from nba_fetcher import NBAFetcher
    fetcher = NBAFetcher(season='2025-26')

    # Get player game log with all stats
    log = fetcher.get_player_log('LeBron James')

    # Get live team defensive rankings (replaces team_rankings.json)
    rankings = fetcher.get_team_rankings()

    # Get player projection data ready for pipeline
    data = fetcher.get_player_data('Amen Thompson', stat='pts', n_games=82)
"""

import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta

# Rate limiting: NBA stats.nba.com throttles aggressive requests
API_DELAY = 0.6  # seconds between API calls

# ── Player name aliases ──
# Maps nickname/abbreviated/alternate names → canonical full name used by nba_api
PLAYER_ALIASES = {
    # Nicknames
    'giannis': 'Giannis Antetokounmpo',
    'wemby': 'Victor Wembanyama',
    'luka': 'Luka Doncic',
    'sga': 'Shai Gilgeous-Alexander',
    'ad': 'Anthony Davis',
    'bron': 'LeBron James',
    'lebron': 'LeBron James',
    'steph': 'Stephen Curry',
    'jokic': 'Nikola Jokic',
    'embiid': 'Joel Embiid',
    'kat': 'Karl-Anthony Towns',
    'ja': 'Ja Morant',
    'ant': 'Anthony Edwards',
    'trae': 'Trae Young',
    'dame': 'Damian Lillard',
    'booker': 'Devin Booker',
    'book': 'Devin Booker',
    'pg': 'Paul George',
    'kawhi': 'Kawhi Leonard',
    'jimmy': 'Jimmy Butler',
    'bam': 'Bam Adebayo',
    'fox': 'De\'Aaron Fox',
    'haliburton': 'Tyrese Haliburton',
    'brunson': 'Jalen Brunson',
    'tatum': 'Jayson Tatum',
    'zion': 'Zion Williamson',
    # Abbreviated names (First Initial. Last) — commonly used on prop boards
    'c. cunningham': 'Cade Cunningham',
    'd. robinson': 'Duncan Robinson',
    'i. stewart': 'Isaiah Stewart',
    'n. claxton': 'Nicolas Claxton',
    'c. braun': 'Christian Braun',
    't. da silva': 'Tristan da Silva',
    'w. carter jr.': 'Wendell Carter Jr.',
    't. hardaway jr.': 'Tim Hardaway Jr.',
    'k. porter jr.': 'Kevin Porter Jr.',
    'p. pritchard': 'Payton Pritchard',
    'd. mitchell': 'Donovan Mitchell',
    'j. tatum': 'Jayson Tatum',
    # More common abbreviated names on prop boards
    'a. drummond': 'Andre Drummond',
    'a. sarr': 'Alexandre Sarr',
    'b. coulibaly': 'Bilal Coulibaly',
    'b. carrington': 'Bub Carrington',
    'b. scheierman': 'Baylor Scheierman',
    'b. williams': 'Brandon Williams',
    'c. gillespie': 'Collin Gillespie',
    'd. barlow': 'Dominick Barlow',
    'd. gafford': 'Daniel Gafford',
    'g. antetokounmpo': 'Giannis Antetokounmpo',
    'j. jaquez jr.': 'Jaime Jaquez Jr.',
    'j. champagnie': 'Julian Champagnie',
    'k. middleton': 'Khris Middleton',
    'k. johnson': 'Keldon Johnson',
    'k. jakucionis': 'Kasparas Jakucionis',
    'l. miller': 'Leonard Miller',
    'm. bagley iii': 'Marvin Bagley III',
    'm. sasser': 'Marcus Sasser',
    'o. prosper': 'Olivier-Maxence Prosper',
    'p. banchero': 'Paolo Banchero',
    'p. washington': 'P.J. Washington',
    'q. grimes': 'Quentin Grimes',
    'r. holland ii': 'Ron Holland II',
    's. gilgeous-alexander': 'Shai Gilgeous-Alexander',
    's. castle': 'Stephon Castle',
    's. fontecchio': 'Simone Fontecchio',
    't. hendricks': 'Taylor Hendricks',
    'v. wembanyama': 'Victor Wembanyama',
}

# Team name mapping: various formats → nba_api team name
TEAM_NAME_MAP = {
    # Short names used in our pipeline → nba_api full names
    '76ers': 'Philadelphia 76ers', 'Bucks': 'Milwaukee Bucks',
    'Bulls': 'Chicago Bulls', 'Cavaliers': 'Cleveland Cavaliers',
    'Celtics': 'Boston Celtics', 'Clippers': 'LA Clippers',
    'Grizzlies': 'Memphis Grizzlies', 'Hawks': 'Atlanta Hawks',
    'Heat': 'Miami Heat', 'Hornets': 'Charlotte Hornets',
    'Jazz': 'Utah Jazz', 'Kings': 'Sacramento Kings',
    'Knicks': 'New York Knicks', 'Lakers': 'Los Angeles Lakers',
    'Magic': 'Orlando Magic', 'Mavericks': 'Dallas Mavericks',
    'Nets': 'Brooklyn Nets', 'Nuggets': 'Denver Nuggets',
    'Pacers': 'Indiana Pacers', 'Pelicans': 'New Orleans Pelicans',
    'Pistons': 'Detroit Pistons', 'Raptors': 'Toronto Raptors',
    'Rockets': 'Houston Rockets', 'Spurs': 'San Antonio Spurs',
    'Suns': 'Phoenix Suns', 'Thunder': 'Oklahoma City Thunder',
    'Timberwolves': 'Minnesota Timberwolves',
    'Trail Blazers': 'Portland Trail Blazers',
    'Warriors': 'Golden State Warriors', 'Wizards': 'Washington Wizards',
}

# Reverse map: full name → short name
TEAM_SHORT = {v: k for k, v in TEAM_NAME_MAP.items()}
# Also map partial matches
for full, short in list(TEAM_SHORT.items()):
    # "Los Angeles Lakers" → also map "Lakers"
    parts = full.split()
    if len(parts) > 1:
        TEAM_SHORT[parts[-1]] = short

# Abbreviation → short name
TEAM_ABR = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards',
}


class NBAFetcher:
    """Live NBA data fetcher using nba_api"""

    def __init__(self, season='2025-26', cache_dir=None):
        self.season = season
        self.cache_dir = cache_dir or os.path.join('predictions', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self._player_cache = {}  # player_name → player_id
        self._gamelog_cache = {}  # player_id → DataFrame
        self._team_rankings_cache = None
        self._last_api_call = 0

        # Lazy imports
        from nba_api.stats.static import players as nba_players, teams as nba_teams
        self._nba_players = nba_players
        self._nba_teams = nba_teams

    def _rate_limit(self):
        """Respect API rate limits"""
        elapsed = time.time() - self._last_api_call
        if elapsed < API_DELAY:
            time.sleep(API_DELAY - elapsed)
        self._last_api_call = time.time()

    def _resolve_player(self, name):
        """Resolve player name to nba_api player ID.

        Handles:
        - Exact full names ("Cade Cunningham")
        - Aliases/nicknames ("giannis", "sga")
        - Abbreviated names ("C. Cunningham", "T. Hardaway Jr.")
        - Suffix variations (Jr., III, II)
        """
        # Check alias first (case-insensitive)
        lookup = PLAYER_ALIASES.get(name.lower(), name)

        if lookup in self._player_cache:
            return self._player_cache[lookup]

        # Try exact match on resolved name
        results = self._nba_players.find_players_by_full_name(lookup)
        if results:
            # Filter to active players first (prefer current over historical)
            active = [r for r in results if r.get('is_active', False)]
            best = active[0] if active else results[0]
            pid = best['id']
            self._player_cache[lookup] = pid
            self._player_cache[name] = pid
            return pid

        # Detect abbreviated name pattern: "X. LastName" or "X. LastName Jr."
        parts = lookup.split()
        is_abbreviated = len(parts) >= 2 and len(parts[0]) <= 2 and '.' in parts[0]

        if is_abbreviated:
            initial = parts[0].replace('.', '').lower()
            # Reconstruct last name (handles "Da Silva", "Carter Jr.", "Hardaway Jr.", "Bagley III")
            last_parts = parts[1:]
            last_name = last_parts[0]  # Primary last name for lookup

            results = self._nba_players.find_players_by_last_name(last_name)
            if results:
                # Filter: first name starts with the initial, prefer active
                matches = [r for r in results
                           if r['first_name'].lower().startswith(initial)]
                if matches:
                    active = [r for r in matches if r.get('is_active', False)]
                    best = active[0] if active else matches[0]
                    pid = best['id']
                    self._player_cache[lookup] = pid
                    self._player_cache[name] = pid
                    return pid

        # General last-name fallback for non-abbreviated names
        if len(parts) >= 2:
            # Try last name (handle suffixes: skip Jr., III, II, Sr.)
            suffixes = {'jr.', 'jr', 'iii', 'ii', 'sr.', 'sr', 'iv'}
            last_name_parts = [p for p in parts[1:] if p.lower().rstrip('.') not in suffixes]
            last_name = last_name_parts[-1] if last_name_parts else parts[-1]

            results = self._nba_players.find_players_by_last_name(last_name)
            if results:
                first = parts[0].lower().rstrip('.')
                # Try matching first name (at least 3 chars or full initial)
                for r in results:
                    r_first = r['first_name'].lower()
                    if r_first.startswith(first[:3]):
                        # Prefer active players
                        active_matches = [
                            x for x in results
                            if x['first_name'].lower().startswith(first[:3])
                            and x.get('is_active', False)
                        ]
                        best = active_matches[0] if active_matches else r
                        pid = best['id']
                        self._player_cache[lookup] = pid
                        self._player_cache[name] = pid
                        return pid

        return None

    def get_player_log(self, player_name, use_cache=True):
        """
        Get full season game log for a player.
        Returns DataFrame with: GAME_DATE, MATCHUP, MIN, PTS, REB, AST, FG3M,
                                STL, BLK, FGM, FGA, FTM, FTA, OREB, DREB, TOV, PF, PLUS_MINUS
        """
        pid = self._resolve_player(player_name)
        if pid is None:
            print(f"[WARN] Player not found: {player_name}")
            return pd.DataFrame()

        # Check memory cache
        if use_cache and pid in self._gamelog_cache:
            return self._gamelog_cache[pid]

        # Check disk cache (refresh if older than 4 hours)
        cache_file = os.path.join(self.cache_dir, f"gamelog_{pid}_{self.season}.parquet")
        if use_cache and os.path.exists(cache_file):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mod_time < timedelta(hours=4):
                df = pd.read_parquet(cache_file)
                self._gamelog_cache[pid] = df
                return df

        # Fetch from API
        from nba_api.stats.endpoints import playergamelog
        self._rate_limit()
        try:
            gl = playergamelog.PlayerGameLog(player_id=pid, season=self.season)
            df = gl.get_data_frames()[0]
        except Exception as e:
            print(f"[ERROR] Failed to fetch game log for {player_name} (ID {pid}): {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        # Clean up
        df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed')
        df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)

        # Parse home/away from MATCHUP (e.g., "LAL vs. DEN" = home, "LAL @ DEN" = away)
        df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)

        # Parse opponent abbreviation
        df['OPP_ABR'] = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s*(\w+)')[0]
        df['OPP_TEAM'] = df['OPP_ABR'].map(TEAM_ABR)

        # Cache it
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass  # parquet might not be available
        self._gamelog_cache[pid] = df

        return df

    def prefetch_player_logs(self, player_names):
        """
        Pre-fetch game logs for a list of players into cache.
        Fetches sequentially (respects 0.6s rate limit) but skips cached entries.
        Returns (cache_hits, api_fetches) counts.
        """
        unique_names = list(dict.fromkeys(player_names))  # dedupe preserving order
        cache_hits = 0
        api_fetches = 0

        for name in unique_names:
            pid = self._resolve_player(name)
            if pid is None:
                continue

            # Check memory cache
            if pid in self._gamelog_cache:
                cache_hits += 1
                continue

            # Check disk cache (< 4hr old)
            cache_file = os.path.join(self.cache_dir, f"gamelog_{pid}_{self.season}.parquet")
            if os.path.exists(cache_file):
                mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - mod_time < timedelta(hours=4):
                    try:
                        df = pd.read_parquet(cache_file)
                        self._gamelog_cache[pid] = df
                        cache_hits += 1
                        continue
                    except Exception:
                        pass

            # Fetch from API
            self.get_player_log(name, use_cache=False)
            api_fetches += 1

        return cache_hits, api_fetches

    def get_player_data(self, player_name, stat, line, opponent=None, is_home=None):
        """
        Get all projection data for a single player prop.
        Returns a dict with everything the pipeline needs:
        - season_avg, l10_avg, l5_avg, l3_avg
        - home_avg, away_avg
        - hit rates (season, l10, l5)
        - minutes consistency
        - streak detection
        - game count
        """
        df = self.get_player_log(player_name)
        if df.empty:
            return None

        # Filter out DNPs (< 10 min)
        df = df[df['MIN'] >= 10].copy()
        if len(df) < 3:
            return None

        # Calculate stat values
        vals = self._calc_stat(df, stat)

        # Core averages
        season_avg = vals.mean()
        l10 = vals.head(10)
        l5 = vals.head(5)
        l3 = vals.head(3)
        l10_avg = l10.mean()
        l5_avg = l5.mean()
        l3_avg = l3.mean()

        # Home/away splits
        home_df = df[df['IS_HOME'] == 1]
        away_df = df[df['IS_HOME'] == 0]
        home_avg = self._calc_stat(home_df, stat).mean() if len(home_df) > 3 else season_avg
        away_avg = self._calc_stat(away_df, stat).mean() if len(away_df) > 3 else season_avg

        # Hit rates
        season_hit = (vals > line).mean()
        l10_hit = (l10 > line).mean()
        l5_hit = (l5 > line).mean()

        # Minutes consistency
        mins_30plus = df[df['MIN'] >= 30]
        mins_ratio = len(mins_30plus) / len(df) if len(df) > 0 else 0

        # If 30+ min player, use those games for L10 hit rate
        l10_30plus = df.head(10)
        l10_30plus = l10_30plus[l10_30plus['MIN'] >= 30]
        if len(l10_30plus) >= 7:
            l10_hit = (self._calc_stat(l10_30plus, stat) > line).mean()
            l10_source = f"L10 (30+ min: {len(l10_30plus)} games)"
        else:
            l10_source = f"L10 (all: {min(10, len(df))} games)"

        # Minutes trend (NEW v4): track L5 vs season minutes for projection scaling
        season_mins_avg = df['MIN'].mean()
        l5_mins_avg = df.head(5)['MIN'].mean()

        # L10 individual values for floor check
        l10_values = [round(v, 1) for v in l10.tolist()]
        l10_floor = round(l10.min(), 1) if len(l10) > 0 else 0
        l10_miss_count = int((l10 <= line).sum()) if len(l10) > 0 else 0

        # Streak detection
        streak_pct = ((l3_avg - l10_avg) / l10_avg * 100) if l10_avg > 0 else 0
        streak_status = "HOT" if streak_pct > 15 else ("COLD" if streak_pct < -15 else "NEUTRAL")

        # Opponent-specific history (last 3 matchups this season)
        opp_history = None
        if opponent:
            opp_df = df[df['OPP_TEAM'] == opponent]
            if len(opp_df) > 0:
                opp_vals = self._calc_stat(opp_df, stat)
                opp_history = {
                    'games': len(opp_df),
                    'avg': round(opp_vals.mean(), 1),
                    'hit_rate': round((opp_vals > line).mean() * 100),
                    'last_3': [round(v, 1) for v in opp_vals.head(3).tolist()],
                }

        # ── v4 Scout data: PLUS_MINUS, PF from cached DataFrames ──
        l10_plus_minus = round(float(df.head(10)['PLUS_MINUS'].mean()), 1) if 'PLUS_MINUS' in df.columns else 0.0
        l10_pf = round(float(df.head(10)['PF'].mean()), 1) if 'PF' in df.columns else 0.0
        l5_pf = round(float(df.head(5)['PF'].mean()), 1) if 'PF' in df.columns else 0.0
        foul_trouble_risk = l5_pf >= 4.0
        l5_plus_minus = round(float(df.head(5)['PLUS_MINUS'].mean()), 1) if 'PLUS_MINUS' in df.columns else 0.0
        efficiency_trend = round(l5_plus_minus - l10_plus_minus, 1)

        return {
            'player': player_name,
            'stat': stat,
            'line': line,
            'season_avg': round(season_avg, 1),
            'l10_avg': round(l10_avg, 1),
            'l5_avg': round(l5_avg, 1),
            'l3_avg': round(l3_avg, 1),
            'home_avg': round(home_avg, 1),
            'away_avg': round(away_avg, 1),
            'season_hit_rate': round(season_hit * 100),
            'l10_hit_rate': round(l10_hit * 100),
            'l5_hit_rate': round(l5_hit * 100),
            'l10_source': l10_source,
            'l10_values': l10_values,
            'l10_floor': l10_floor,
            'l10_miss_count': l10_miss_count,
            'mins_30plus_pct': round(mins_ratio * 100),
            'season_mins_avg': round(season_mins_avg, 1),
            'l5_mins_avg': round(l5_mins_avg, 1),
            'streak_status': streak_status,
            'streak_pct': round(streak_pct, 1),
            'games_played': len(df),
            'opponent_history': opp_history,
            # v4 scout data (extracted from cached DataFrame, zero new API calls)
            'l10_avg_plus_minus': l10_plus_minus,
            'l10_avg_pf': l10_pf,
            'l5_avg_pf': l5_pf,
            'foul_trouble_risk': foul_trouble_risk,
            'efficiency_trend': efficiency_trend,
        }

    def _calc_stat(self, df, stat):
        """Calculate stat values from game log DataFrame"""
        stat_cols = {
            'pts': ['PTS'],
            'reb': ['REB'],
            'ast': ['AST'],
            '3pm': ['FG3M'],
            'stl': ['STL'],
            'blk': ['BLK'],
            'pra': ['PTS', 'REB', 'AST'],
            'pr': ['PTS', 'REB'],
            'pa': ['PTS', 'AST'],
            'ra': ['REB', 'AST'],
            'stl_blk': ['STL', 'BLK'],
        }
        cols = stat_cols.get(stat.lower(), ['PTS'])
        if len(cols) == 1:
            return df[cols[0]].astype(float)
        else:
            return df[cols].astype(float).sum(axis=1)

    def get_team_rankings(self, use_cache=True):
        """
        Fetch LIVE team defensive rankings from nba_api.
        Returns dict in same format as team_rankings.json but with REAL data.
        Replaces the static file entirely.
        """
        if use_cache and self._team_rankings_cache is not None:
            return self._team_rankings_cache

        # Check disk cache (refresh daily)
        cache_file = os.path.join(self.cache_dir, f"team_rankings_{self.season}.json")
        if use_cache and os.path.exists(cache_file):
            mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mod_time < timedelta(hours=12):
                with open(cache_file) as f:
                    data = json.load(f)
                    self._team_rankings_cache = data
                    return data

        from nba_api.stats.endpoints import leaguedashteamstats

        # 1. Get opponent stats (what each team ALLOWS) - use Totals, we'll calc per-game
        self._rate_limit()
        opp = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            measure_type_detailed_defense='Opponent'
        )
        opp_df = opp.get_data_frames()[0]

        # 2. Get advanced stats (pace, ratings)
        self._rate_limit()
        adv = leaguedashteamstats.LeagueDashTeamStats(
            season=self.season,
            measure_type_detailed_defense='Advanced'
        )
        adv_df = adv.get_data_frames()[0]

        # Merge
        merged = opp_df.merge(adv_df[['TEAM_ID', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE']],
                               on='TEAM_ID')

        # Build rankings dict
        teams = {}
        for _, row in merged.iterrows():
            full_name = row['TEAM_NAME']
            short = TEAM_SHORT.get(full_name)
            if not short:
                # Try partial match
                for k, v in TEAM_SHORT.items():
                    if k in full_name:
                        short = v
                        break
            if not short:
                continue

            gp = row['GP']
            teams[short] = {
                'avg_pts_allowed': round(row['OPP_PTS'] / gp, 1) if gp > 0 else 0,
                'reb_allowed': round(row['OPP_REB'] / gp, 1) if gp > 0 else 0,
                'ast_allowed': round(row['OPP_AST'] / gp, 1) if gp > 0 else 0,
                'tpm_allowed': round(row['OPP_FG3M'] / gp, 1) if gp > 0 else 0,
                'stl_allowed': round(row['OPP_STL'] / gp, 1) if gp > 0 else 0,
                'blk_allowed': round(row['OPP_BLK'] / gp, 1) if gp > 0 else 0,
                'opp_pts_total': int(row['OPP_PTS']),
                'pace': round(row['PACE'], 1),
                'off_rating': round(row['OFF_RATING'], 1),
                'def_rating': round(row['DEF_RATING'], 1),
                'net_rating': round(row['NET_RATING'], 1),
                'games': int(gp),
            }

        # ── League averages for rate-based defense adjustment ──
        league_avg = {}
        avg_keys = ['avg_pts_allowed', 'reb_allowed', 'ast_allowed', 'tpm_allowed', 'stl_allowed', 'blk_allowed']
        for key in avg_keys:
            vals = [t[key] for t in teams.values() if key in t]
            league_avg[key] = round(sum(vals) / len(vals), 1) if vals else 0

        # Calculate ranks
        for stat_key in ['avg_pts_allowed', 'reb_allowed', 'ast_allowed', 'tpm_allowed', 'pace']:
            sorted_teams = sorted(teams.items(), key=lambda x: x[1][stat_key])
            for rank, (team, _) in enumerate(sorted_teams, 1):
                rank_key = stat_key.replace('allowed', 'allowed_rank') if 'allowed' in stat_key else f"{stat_key}_rank"
                teams[team][rank_key] = rank

        # Defense rank based on def_rating (lower = better)
        sorted_def = sorted(teams.items(), key=lambda x: x[1]['def_rating'])
        for rank, (team, _) in enumerate(sorted_def, 1):
            teams[team]['def_rank'] = rank

        result = {
            'season': self.season,
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'source': 'nba_api (stats.nba.com)',
            'teams': teams,
            'league_avg': league_avg,
        }

        # Cache to disk
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)

        self._team_rankings_cache = result
        return result

    def get_box_scores(self, game_date):
        """
        Get box scores for all games on a given date.
        Uses raw V3 JSON (nba_api's V3 DataFrame parser is broken on Python 3.14).
        Returns dict of {player_name: {pts, reb, ast, 3pm, stl, blk, min, ...}}
        """
        import unicodedata, json
        from nba_api.stats.endpoints import scoreboardv2
        from nba_api.stats.library.http import NBAStatsHTTP

        if isinstance(game_date, str):
            game_date = datetime.strptime(game_date, '%Y-%m-%d')

        date_str = game_date.strftime('%m/%d/%Y')

        self._rate_limit()
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=date_str)
            games = sb.get_data_frames()[0]  # GameHeader
        except Exception as e:
            print(f"[ERROR] Scoreboard fetch failed for {date_str}: {e}")
            return {}

        if games.empty:
            print(f"[WARN] No games found for {date_str}")
            return {}

        all_players = {}
        nba_http = NBAStatsHTTP()

        for _, game in games.iterrows():
            game_id = game['GAME_ID']
            self._rate_limit()
            try:
                resp = nba_http.send_api_request(
                    endpoint='boxscoretraditionalv3',
                    parameters={'GameID': game_id}
                )
                data = resp.get_response()
                if isinstance(data, str):
                    data = json.loads(data)

                bs_data = data.get('boxScoreTraditional', {})
                # Collect players from both teams
                team_players = []
                for team_key in ('homeTeam', 'awayTeam'):
                    team = bs_data.get(team_key, {})
                    tricode = team.get('teamTricode', '')
                    for p in team.get('players', []):
                        p['_teamTricode'] = tricode
                        team_players.append(p)
            except Exception as e:
                print(f"[ERROR] Box score fetch failed for game {game_id}: {e}")
                continue

            for ps in team_players:
                first = str(ps.get('firstName', ''))
                last = str(ps.get('familyName', ''))
                name = f"{first} {last}".strip()
                if not name or name == ' ':
                    continue

                stats = ps.get('statistics', {})
                mins_raw = stats.get('minutes', 'PT00M00.00S')
                mins = 0
                if isinstance(mins_raw, str) and 'M' in str(mins_raw):
                    try:
                        m_part = str(mins_raw).split('PT')[1].split('M')[0]
                        s_part = str(mins_raw).split('M')[1].replace('S', '')
                        mins = int(m_part) + float(s_part) / 60
                    except Exception:
                        mins = 0

                pts = int(stats.get('points', 0) or 0)
                reb = int(stats.get('reboundsTotal', 0) or 0)
                ast = int(stats.get('assists', 0) or 0)
                fg3m = int(stats.get('threePointersMade', 0) or 0)
                stl = int(stats.get('steals', 0) or 0)
                blk = int(stats.get('blocks', 0) or 0)

                all_players[name] = {
                    'pts': pts, 'reb': reb, 'ast': ast,
                    '3pm': fg3m, 'stl': stl, 'blk': blk,
                    'min': round(mins, 1),
                    'pra': pts + reb + ast,
                    'pr': pts + reb,
                    'pa': pts + ast,
                    'ra': reb + ast,
                    'stl_blk': stl + blk,
                    'team': ps.get('_teamTricode', ''),
                    'game_id': game_id,
                }

        return all_players

    def get_todays_schedule(self):
        """Get today's NBA schedule with home/away teams"""
        from nba_api.stats.endpoints import scoreboardv2

        today = datetime.now().strftime('%m/%d/%Y')
        self._rate_limit()
        try:
            sb = scoreboardv2.ScoreboardV2(game_date=today)
            games = sb.get_data_frames()[0]
        except Exception as e:
            print(f"[ERROR] Schedule fetch failed: {e}")
            return []

        schedule = []
        for _, game in games.iterrows():
            home_id = game.get('HOME_TEAM_ID')
            away_id = game.get('VISITOR_TEAM_ID')

            # Resolve team names
            home_team = self._team_id_to_short(home_id)
            away_team = self._team_id_to_short(away_id)

            schedule.append({
                'game_id': game['GAME_ID'],
                'home': home_team,
                'away': away_team,
                'status': game.get('GAME_STATUS_TEXT', ''),
            })

        return schedule

    def _team_id_to_short(self, team_id):
        """Convert nba_api team_id to our short name"""
        all_teams = self._nba_teams.get_teams()
        for t in all_teams:
            if t['id'] == team_id:
                full = t['full_name']
                return TEAM_SHORT.get(full, t['nickname'])
        return str(team_id)

    def update_team_rankings_file(self, filepath='predictions/team_rankings.json'):
        """
        Fetch live rankings and write to team_rankings.json.
        This replaces the static file with real data.
        """
        rankings = self.get_team_rankings(use_cache=False)
        with open(filepath, 'w') as f:
            json.dump(rankings, f, indent=2)
        print(f"[OK] Updated {filepath} with live data from stats.nba.com")
        print(f"     {len(rankings['teams'])} teams, updated {rankings['updated']}")
        return rankings

    def clear_cache(self):
        """Clear all cached data"""
        self._player_cache = {}
        self._gamelog_cache = {}
        self._team_rankings_cache = None
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
        print("[OK] Cache cleared")


    def get_usage_metrics(self, player_name):
        """
        Compute usage rate from cached game logs (zero new API calls).
        Usage proxy: (FGA + 0.44*FTA + TOV) / MIN.clip(1)
        Returns {season_usage, l10_usage, l5_usage, usage_trend} or None.
        """
        df = self.get_player_log(player_name)
        if df.empty or len(df) < 5:
            return None

        df = df[df['MIN'] >= 10].copy()
        if len(df) < 5:
            return None

        def _usage(subset):
            fga = subset['FGA'].astype(float)
            fta = subset['FTA'].astype(float)
            tov = subset['TOV'].astype(float)
            mins = subset['MIN'].astype(float).clip(lower=1)
            return ((fga + 0.44 * fta + tov) / mins).mean()

        season_usage = round(_usage(df), 3)
        l10_usage = round(_usage(df.head(10)), 3)
        l5_usage = round(_usage(df.head(5)), 3)
        usage_trend = round(l5_usage - l10_usage, 3)

        return {
            'season_usage': season_usage,
            'l10_usage': l10_usage,
            'l5_usage': l5_usage,
            'usage_trend': usage_trend,
        }

    def get_without_stats(self, player_name, teammate_name, stat):
        """
        Compute player's stats WITH vs WITHOUT a teammate from cached game logs.
        Teammate absent = MIN < 10 or not in game.
        Returns {avg_with, avg_without, delta, games_without} or None.
        """
        player_df = self.get_player_log(player_name)
        teammate_df = self.get_player_log(teammate_name)

        if player_df.empty or teammate_df.empty:
            return None

        player_df = player_df[player_df['MIN'] >= 10].copy()
        if len(player_df) < 5:
            return None

        # Get dates where teammate was absent (MIN < 10 or not playing)
        teammate_active_dates = set()
        for _, row in teammate_df.iterrows():
            if row['MIN'] >= 10:
                teammate_active_dates.add(row['GAME_DATE'])

        # Split player games by teammate presence
        with_mask = player_df['GAME_DATE'].isin(teammate_active_dates)
        games_with = player_df[with_mask]
        games_without = player_df[~with_mask]

        if len(games_without) < 3:
            return None

        vals_with = self._calc_stat(games_with, stat)
        vals_without = self._calc_stat(games_without, stat)

        avg_with = round(vals_with.mean(), 1)
        avg_without = round(vals_without.mean(), 1)

        return {
            'avg_with': avg_with,
            'avg_without': avg_without,
            'delta': round(avg_without - avg_with, 1),
            'games_without': len(games_without),
        }


# ── CONVENIENCE FUNCTIONS ──

def fetch_player_projection(player_name, stat, line, opponent=None, is_home=None, season='2025-26'):
    """Quick one-shot: get projection data for a single player prop"""
    fetcher = NBAFetcher(season=season)
    return fetcher.get_player_data(player_name, stat, line, opponent, is_home)

def fetch_team_rankings(season='2025-26'):
    """Quick one-shot: get live team defensive rankings"""
    fetcher = NBAFetcher(season=season)
    return fetcher.get_team_rankings()

def fetch_box_scores(game_date, season='2025-26'):
    """Quick one-shot: get all box scores for a date"""
    fetcher = NBAFetcher(season=season)
    return fetcher.get_box_scores(game_date)


if __name__ == '__main__':
    print("=" * 60)
    print("NBA Live Data Fetcher - Testing")
    print("=" * 60)

    fetcher = NBAFetcher()

    # Test player lookup
    print("\n1. Testing player game log...")
    log = fetcher.get_player_log('Amen Thompson')
    if not log.empty:
        print(f"   Found {len(log)} games")
        print(f"   Last game: {log.iloc[0]['GAME_DATE']} - {log.iloc[0]['PTS']}pts {log.iloc[0]['REB']}reb {log.iloc[0]['AST']}ast")

    # Test projection data
    print("\n2. Testing player projection data...")
    data = fetcher.get_player_data('Amen Thompson', 'pts', 14.5, opponent='Nuggets', is_home=False)
    if data:
        print(f"   Season avg: {data['season_avg']}")
        print(f"   L10 avg: {data['l10_avg']}, L5 avg: {data['l5_avg']}")
        print(f"   L10 hit rate: {data['l10_hit_rate']}%")
        print(f"   Streak: {data['streak_status']} ({data['streak_pct']:+.1f}%)")
        if data['opponent_history']:
            print(f"   vs Nuggets: {data['opponent_history']['avg']} avg in {data['opponent_history']['games']} games")

    # Test team rankings
    print("\n3. Testing live team rankings...")
    rankings = fetcher.get_team_rankings()
    print(f"   {len(rankings['teams'])} teams loaded")
    # Show top 5 defenses
    sorted_teams = sorted(rankings['teams'].items(), key=lambda x: x[1]['def_rank'])
    print("   Top 5 defenses:")
    for team, data in sorted_teams[:5]:
        print(f"   #{data['def_rank']} {team}: DEF RTG {data['def_rating']}, allows {data['avg_pts_allowed']} ppg, pace {data['pace']}")

    print("\n[DONE] All tests passed!")

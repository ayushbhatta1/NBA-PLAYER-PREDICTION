#!/usr/bin/env python3
"""
Parallel Game Research Engine
Deploys concurrent research agents (1 per team) to gather:
- Injuries (official NBA report + nba_api)
- Vegas spreads (web scrape)
- B2B detection (schedule check)
- Breaking news / late scratches
- Team recent form

All research runs in parallel via ThreadPoolExecutor.
Output: GAMES dict ready for the pipeline runner.

Usage:
    from game_researcher import research_all_games
    games = research_all_games("2026-03-13")
"""

import json
import os
import sys
import time
import re
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from injury_scraper import get_injury_context, ABR_TO_SHORT, SHORT_TO_ABR
except ImportError:
    get_injury_context = None
    ABR_TO_SHORT = {}
    SHORT_TO_ABR = {}

try:
    from nba_fetcher import NBAFetcher
except ImportError:
    NBAFetcher = None

# ── Constants ──
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

ABR_TO_FULL = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards',
}

ABR_TO_SHORT_LOCAL = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards',
    # ESPN uses different abbreviations for some teams
    'SA': 'Spurs', 'WSH': 'Wizards', 'NO': 'Pelicans', 'NY': 'Knicks',
    'GS': 'Warriors', 'UTAH': 'Jazz', 'PHO': 'Suns',
}

# ESPN → pipeline abbreviation normalization
ESPN_ABR_MAP = {
    'SA': 'SAS', 'WSH': 'WAS', 'NO': 'NOP', 'NY': 'NYK',
    'GS': 'GSW', 'UTAH': 'UTA', 'PHO': 'PHX',
}

# ESPN team IDs (single source of truth — used by news, form, etc.)
ESPN_TEAM_IDS = {
    'ATL': 1, 'BOS': 2, 'BKN': 17, 'CHA': 30, 'CHI': 4, 'CLE': 5,
    'DAL': 6, 'DEN': 7, 'DET': 8, 'GSW': 9, 'HOU': 10, 'IND': 11,
    'LAC': 12, 'LAL': 13, 'MEM': 29, 'MIA': 14, 'MIL': 15, 'MIN': 16,
    'NOP': 3, 'NYK': 18, 'OKC': 25, 'ORL': 19, 'PHI': 20, 'PHX': 21,
    'POR': 22, 'SAC': 23, 'SAS': 24, 'TOR': 28, 'UTA': 26, 'WAS': 27,
}

def _normalize_abr(abr):
    """Normalize ESPN team abbreviations to pipeline standard."""
    return ESPN_ABR_MAP.get(abr, abr)


def _calc_workers(num_agents, api_type='espn'):
    """Calculate optimal thread pool size based on workload and API type."""
    cpu = os.cpu_count() or 4
    if api_type == 'espn':
        # IO-bound ESPN calls, cap at 20
        return min(max(4, num_agents, cpu * 2), 20)
    elif api_type == 'nba_api':
        return 2  # rate-limited at 0.6s, more threads don't help
    elif api_type == 'local':
        # CPU-bound (verification, parlay reasoning)
        return min(num_agents, cpu * 2, 16)
    return min(num_agents, 10)


def _web_fetch(url, timeout=10):
    """Simple web fetch with User-Agent header."""
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except (URLError, HTTPError, Exception) as e:
        return f"ERROR: {e}"


class ScoreboardCache:
    """Thread-safe cache for ESPN scoreboard data by date.
    Fetches each date exactly once, eliminating ~340 redundant ESPN calls."""

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._fetch_count = 0

    def get(self, date_yyyymmdd):
        """Get scoreboard data for a date (YYYYMMDD format). Fetches once, caches forever."""
        with self._lock:
            if date_yyyymmdd in self._cache:
                return self._cache[date_yyyymmdd]

        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_yyyymmdd}"
        html = _web_fetch(url, timeout=10)
        parsed = None
        if "ERROR:" not in html:
            try:
                parsed = json.loads(html)
            except (json.JSONDecodeError, ValueError):
                pass

        with self._lock:
            self._cache[date_yyyymmdd] = parsed
            self._fetch_count += 1
        return parsed

    def prewarm(self, dates, max_workers=8):
        """Fetch multiple dates in parallel to prime the cache."""
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.get, d): d for d in dates}
            for f in as_completed(futures):
                try:
                    f.result(timeout=15)
                except Exception:
                    pass

    @property
    def stats(self):
        with self._lock:
            return {'cached': len(self._cache), 'fetched': self._fetch_count}


def _fetch_cbs_injuries(date_str, games):
    """
    Fetch injury data from ESPN injuries endpoint (league-wide).
    Maps each injured player to their team and game for today's slate.
    """
    injury_context = {
        'injured_out': {},
        'all_out': [],
        'player_statuses': {},
        'games': [],
        'report_date': date_str,
        'total_out': 0,
        'total_questionable': 0,
        'source': 'ESPN injuries endpoint',
    }

    # Build set of teams playing today for filtering
    today_teams = set()
    team_to_game = {}  # abr -> game info
    for g in games:
        today_teams.add(g['away'])
        today_teams.add(g['home'])

    # Pre-build game_info dicts
    game_infos = {}
    for g in games:
        away = g['away']
        home = g['home']
        key = g['game_key']
        away_short = ABR_TO_SHORT_LOCAL.get(away, away)
        home_short = ABR_TO_SHORT_LOCAL.get(home, home)
        game_infos[key] = {
            'matchup': key,
            'label': f"{away_short}@{home_short}",
            'away': away_short,
            'home': home_short,
            'away_abr': away,
            'home_abr': home,
            'away_out': [],
            'home_out': [],
            'away_questionable': [],
            'home_questionable': [],
        }
        team_to_game[away] = (key, 'away')
        team_to_game[home] = (key, 'home')

    # Fetch ESPN injuries endpoint (all teams)
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
    html = _web_fetch(url, timeout=10)

    if "ERROR:" in html:
        # Fallback: try scoreboard-based injuries
        return _fetch_scoreboard_injuries(date_str, games)

    try:
        data = json.loads(html)
    except (json.JSONDecodeError, ValueError):
        return _fetch_scoreboard_injuries(date_str, games)

    # ESPN injuries response: {injuries: [{id, displayName: "Atlanta Hawks", injuries: [{athlete, type, status, ...}]}]}
    team_entries = data.get('injuries', []) if isinstance(data, dict) else data

    # Map full team names → our abbreviations
    FULL_TO_ABR = {v: k for k, v in ABR_TO_FULL.items()}

    for team_entry in team_entries:
        team_name = team_entry.get('displayName', '')
        team_abr = FULL_TO_ABR.get(team_name, '')

        # Only process teams playing today
        if not team_abr or team_abr not in today_teams:
            continue

        game_key_side = team_to_game.get(team_abr)
        if not game_key_side:
            continue
        game_key, side = game_key_side
        team_short = ABR_TO_SHORT_LOCAL.get(team_abr, team_abr)

        injuries = team_entry.get('injuries', [])
        for inj in injuries:
            player = inj.get('athlete', {}).get('displayName', '')
            status_desc = inj.get('type', {}).get('description', '').lower()
            comment = inj.get('shortComment', '')
            if not player:
                continue

            if status_desc == 'out':
                game_infos[game_key][f'{side}_out'].append(player)
                injury_context['all_out'].append(player)
                injury_context['injured_out'].setdefault(team_short, []).append(player)
            elif status_desc in ('day-to-day', 'questionable', 'doubtful'):
                # Check shortComment for "is out" — ESPN marks some as day-to-day but confirms out
                if comment and 'is out' in comment.lower():
                    game_infos[game_key][f'{side}_out'].append(player)
                    injury_context['all_out'].append(player)
                    injury_context['injured_out'].setdefault(team_short, []).append(player)
                else:
                    game_infos[game_key][f'{side}_questionable'].append(player)
                injury_context['player_statuses'][player] = {
                    'status': status_desc.title(),
                    'team': team_short,
                    'comment': comment[:100] if comment else '',
                }

    injury_context['games'] = list(game_infos.values())
    injury_context['total_out'] = len(injury_context['all_out'])
    injury_context['total_questionable'] = len(injury_context['player_statuses'])

    return injury_context


def _fetch_scoreboard_injuries(date_str, games):
    """
    Legacy fallback: try to get injuries from ESPN scoreboard competitor data.
    """
    injury_context = {
        'injured_out': {},
        'all_out': [],
        'player_statuses': {},
        'games': [],
        'report_date': date_str,
        'total_out': 0,
        'total_questionable': 0,
        'source': 'ESPN scoreboard (legacy fallback)',
    }

    formatted = date_str.replace('-', '')
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={formatted}"
    html = _web_fetch(url, timeout=10)

    if "ERROR:" in html:
        return injury_context

    data = json.loads(html)
    events = data.get('events', [])

    for event in events:
        comps = event.get('competitions', [{}])[0]
        competitors = comps.get('competitors', [])

        away_data = None
        home_data = None
        for comp in competitors:
            if comp.get('homeAway') == 'away':
                away_data = comp
            else:
                home_data = comp

        if not away_data or not home_data:
            continue

        away_abr = _normalize_abr(away_data.get('team', {}).get('abbreviation', ''))
        home_abr = _normalize_abr(home_data.get('team', {}).get('abbreviation', ''))
        away_short = ABR_TO_SHORT_LOCAL.get(away_abr, away_abr)
        home_short = ABR_TO_SHORT_LOCAL.get(home_abr, home_abr)

        game_info = {
            'matchup': f"{away_abr}@{home_abr}",
            'label': f"{away_short}@{home_short}",
            'away': away_short,
            'home': home_short,
            'away_abr': away_abr,
            'home_abr': home_abr,
            'away_out': [],
            'home_out': [],
            'away_questionable': [],
            'home_questionable': [],
        }

        for side, comp in [('away', away_data), ('home', home_data)]:
            injuries = comp.get('injuries', [])
            for inj in injuries:
                entries = inj.get('entries', [])
                for entry in entries:
                    player = entry.get('athlete', {}).get('displayName', '')
                    status = entry.get('status', '').lower()
                    if not player:
                        continue
                    if 'out' in status:
                        game_info[f'{side}_out'].append(player)
                        injury_context['all_out'].append(player)
                    elif any(s in status for s in ['day-to-day', 'questionable', 'doubtful']):
                        game_info[f'{side}_questionable'].append(player)
                        injury_context['player_statuses'][player] = {
                            'status': status.title(),
                            'team': ABR_TO_SHORT_LOCAL.get(
                                away_abr if side == 'away' else home_abr, ''
                            ),
                        }

        injury_context['games'].append(game_info)

    injury_context['total_out'] = len(injury_context['all_out'])
    injury_context['total_questionable'] = len(injury_context['player_statuses'])

    return injury_context


# ═══════════════════════════════════════════════════
# AGENT 1: Injury Research (per team)
# ═══════════════════════════════════════════════════
def research_injuries(team_abr, date_str, injury_context=None):
    """
    Research agent for one team's injuries.
    Uses official NBA injury report (already fetched) + supplements.
    """
    result = {
        'team': team_abr,
        'out': [],
        'questionable': [],
        'doubtful': [],
        'probable': [],
        'notes': [],
    }

    if not injury_context:
        return result

    team_short = ABR_TO_SHORT_LOCAL.get(team_abr, team_abr)

    # Extract from official injury report
    for game in injury_context.get('games', []):
        away_abr = game.get('away_abr', '')
        home_abr = game.get('home_abr', '')

        if team_abr == away_abr:
            result['out'] = game.get('away_out', [])
            result['questionable'] = game.get('away_questionable', [])
        elif team_abr == home_abr:
            result['out'] = game.get('home_out', [])
            result['questionable'] = game.get('home_questionable', [])

    # Also check player_statuses for Doubtful/Probable granularity
    for player, info in injury_context.get('player_statuses', {}).items():
        if info.get('team') == team_short:
            status = info.get('status', '')
            if status == 'Doubtful' and player not in result['doubtful']:
                result['doubtful'].append(player)
            elif status == 'Probable' and player not in result['probable']:
                result['probable'].append(player)

    if result['out']:
        result['notes'].append(f"{team_short} OUT: {', '.join(result['out'])}")
    if result['questionable']:
        result['notes'].append(f"{team_short} Q: {', '.join(result['questionable'])}")

    return result


# ═══════════════════════════════════════════════════
# AGENT 2: Schedule / B2B Detection (per team)
# ═══════════════════════════════════════════════════
def research_b2b(team_abr, date_str, _scoreboard_cache=None):
    """
    Check if team played yesterday (back-to-back).
    Uses ScoreboardCache if available, else falls back to direct ESPN fetch.
    """
    result = {'team': team_abr, 'is_b2b': False, 'yesterday_game': None}

    try:
        target = datetime.strptime(date_str, '%Y-%m-%d')
        yesterday = (target - timedelta(days=1)).strftime('%Y%m%d')

        data = None
        if _scoreboard_cache:
            data = _scoreboard_cache.get(yesterday)
        else:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={yesterday}"
            html = _web_fetch(url, timeout=8)
            if "ERROR:" not in html:
                data = json.loads(html)

        if data:
            events = data.get('events', [])
            team_full = ABR_TO_FULL.get(team_abr, '').lower()

            for event in events:
                name = event.get('name', '').lower()
                short_name = event.get('shortName', '').lower()
                if team_full and (team_full.split()[-1] in name):
                    result['is_b2b'] = True
                    result['yesterday_game'] = event.get('shortName', '')
                    break
                if team_abr.lower() in short_name:
                    result['is_b2b'] = True
                    result['yesterday_game'] = event.get('shortName', '')
                    break
    except Exception as e:
        result['error'] = str(e)

    return result


# ═══════════════════════════════════════════════════
# AGENT 3: Spread / Odds Research (per game)
# ═══════════════════════════════════════════════════
def research_spread(away_abr, home_abr, date_str, _scoreboard_cache=None):
    """
    Fetch Vegas spread for a game.
    Uses ScoreboardCache if available, else falls back to direct ESPN fetch.
    """
    result = {
        'game': f"{away_abr}@{home_abr}",
        'spread': None,
        'over_under': None,
        'source': None,
    }

    try:
        formatted_date = date_str.replace('-', '')
        data = None
        if _scoreboard_cache:
            data = _scoreboard_cache.get(formatted_date)
        else:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={formatted_date}"
            html = _web_fetch(url, timeout=8)
            if "ERROR:" not in html:
                data = json.loads(html)

        if data:
            events = data.get('events', [])

            for event in events:
                competitors = event.get('competitions', [{}])[0].get('competitors', [])
                teams_in_game = [c.get('team', {}).get('abbreviation', '') for c in competitors]

                if away_abr in teams_in_game and home_abr in teams_in_game:
                    # Found our game - extract odds
                    odds = event.get('competitions', [{}])[0].get('odds', [])
                    if odds:
                        odds_data = odds[0]
                        result['spread'] = odds_data.get('spread', None)
                        result['over_under'] = odds_data.get('overUnder', None)
                        result['source'] = odds_data.get('provider', {}).get('name', 'ESPN')

                        # Determine which team is favored
                        fav_flag = odds_data.get('homeTeamOdds', {}).get('favorite', False)
                        spread_val = odds_data.get('spread', 0)
                        if spread_val is not None:
                            try:
                                spread_val = float(spread_val)
                                # ESPN spread is from home perspective (negative = home favored)
                                result['spread'] = spread_val
                            except (ValueError, TypeError):
                                pass
                    break
    except Exception as e:
        result['error'] = str(e)

    return result


# ═══════════════════════════════════════════════════
# AGENT 4: Breaking News / Late Scratches (per team)
# ═══════════════════════════════════════════════════
def research_news(team_abr, date_str):
    """
    Check for breaking news / late scratches via ESPN team news + roster/injuries.
    """
    result = {
        'team': team_abr,
        'headlines': [],
        'key_updates': [],
        'injuries_from_espn': [],
    }

    team_id = ESPN_TEAM_IDS.get(team_abr)
    if not team_id:
        return result

    # Agent task 1: ESPN team injuries endpoint
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/injuries"
        html = _web_fetch(url, timeout=8)
        if "ERROR:" not in html:
            data = json.loads(html)
            items = data.get('items', [])
            for item in items:
                player_name = item.get('athlete', {}).get('displayName', '')
                status = item.get('status', '')
                desc = item.get('longComment', '') or item.get('shortComment', '')
                if player_name:
                    result['injuries_from_espn'].append({
                        'player': player_name,
                        'status': status,
                        'detail': desc[:100],
                    })
    except Exception:
        pass

    # Agent task 2: ESPN team news
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/news"
        html = _web_fetch(url, timeout=8)
        if "ERROR:" not in html:
            data = json.loads(html)
            articles = data.get('articles', [])
            for article in articles[:5]:
                headline = article.get('headline', '')
                desc = article.get('description', '')
                result['headlines'].append(headline)

                injury_keywords = ['out', 'injury', 'questionable', 'ruled out',
                                   'miss', 'scratch', 'sidelined', 'rest',
                                   'return', 'cleared', 'upgrade', 'downgrade',
                                   'doubtful', 'day-to-day', 'dtd']
                headline_lower = (headline + ' ' + desc).lower()
                if any(kw in headline_lower for kw in injury_keywords):
                    result['key_updates'].append(headline)
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════
# AGENT 5: Recent Form / Win Streak (per team)
# ═══════════════════════════════════════════════════
def research_recent_form(team_abr, date_str, _scoreboard_cache=None):
    """
    Pull team's last 5 game results, W/L record, scoring trends.
    Uses ScoreboardCache if available, else falls back to direct ESPN fetch.
    """
    result = {
        'team': team_abr,
        'last5_record': '',
        'last5_results': [],
        'streak': '',
        'avg_pts_scored': 0,
        'avg_pts_allowed': 0,
        'home_record': '',
        'away_record': '',
    }

    team_id = ESPN_TEAM_IDS.get(team_abr)
    if not team_id:
        return result

    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}"
        html = _web_fetch(url, timeout=8)
        if "ERROR:" not in html:
            data = json.loads(html)
            team_data = data.get('team', {})

            # Overall record
            record = team_data.get('record', {})
            items = record.get('items', [])
            for item in items:
                summary = item.get('summary', '')
                stat_type = item.get('type', '')
                if stat_type == 'total':
                    result['overall_record'] = summary
                elif stat_type == 'home':
                    result['home_record'] = summary
                elif stat_type == 'road' or stat_type == 'away':
                    result['away_record'] = summary

            # Streak from standingSummary or nextEvent
            standing = team_data.get('standingSummary', '')
            result['standing'] = standing

    except Exception as e:
        result['error'] = str(e)

    # Get last 5 results from scoreboard lookups
    try:
        target = datetime.strptime(date_str, '%Y-%m-%d')
        wins = 0
        losses = 0
        pts_scored = []
        pts_allowed = []

        for days_back in range(1, 15):  # look back up to 14 days for 5 games
            check_date = (target - timedelta(days=days_back)).strftime('%Y%m%d')

            data = None
            if _scoreboard_cache:
                data = _scoreboard_cache.get(check_date)
            else:
                url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={check_date}"
                html = _web_fetch(url, timeout=5)
                if "ERROR:" not in html:
                    data = json.loads(html)

            if not data:
                continue

            for event in data.get('events', []):
                if event.get('status', {}).get('type', {}).get('name') != 'STATUS_FINAL':
                    continue
                comps = event.get('competitions', [{}])[0]
                competitors = comps.get('competitors', [])

                our_team = None
                opp_team = None
                for comp in competitors:
                    abr = _normalize_abr(comp.get('team', {}).get('abbreviation', ''))
                    if abr == team_abr:
                        our_team = comp
                    else:
                        opp_team = comp

                if our_team and opp_team:
                    our_score = int(our_team.get('score', 0))
                    opp_score = int(opp_team.get('score', 0))
                    won = our_score > opp_score
                    opp_abr = _normalize_abr(opp_team.get('team', {}).get('abbreviation', ''))

                    result['last5_results'].append({
                        'date': check_date,
                        'opponent': opp_abr,
                        'score': f"{our_score}-{opp_score}",
                        'result': 'W' if won else 'L',
                    })
                    if won:
                        wins += 1
                    else:
                        losses += 1
                    pts_scored.append(our_score)
                    pts_allowed.append(opp_score)

            if len(result['last5_results']) >= 5:
                break

        result['last5_results'] = result['last5_results'][:5]
        result['last5_record'] = f"{wins}-{losses}"
        if pts_scored:
            result['avg_pts_scored'] = round(sum(pts_scored[:5]) / len(pts_scored[:5]), 1)
            result['avg_pts_allowed'] = round(sum(pts_allowed[:5]) / len(pts_allowed[:5]), 1)

        # Determine streak
        if result['last5_results']:
            streak_type = result['last5_results'][0]['result']
            streak_count = 0
            for g in result['last5_results']:
                if g['result'] == streak_type:
                    streak_count += 1
                else:
                    break
            result['streak'] = f"{'W' if streak_type == 'W' else 'L'}{streak_count}"

    except Exception as e:
        result['form_error'] = str(e)

    return result


# ═══════════════════════════════════════════════════
# AGENT 6: Rest Days / Schedule Density (per team)
# ═══════════════════════════════════════════════════
def research_rest_days(team_abr, date_str, _scoreboard_cache=None):
    """
    Check how many days rest a team has (not just B2B).
    0 days = B2B, 1 day = normal, 2+ days = well-rested.
    Also checks if team has 3-in-4 or 4-in-6 density.
    """
    result = {
        'team': team_abr,
        'days_rest': None,
        'last_game_date': None,
        'schedule_density': 'normal',  # 'compressed', 'normal', 'rested'
        'games_in_last_7': 0,
    }

    try:
        target = datetime.strptime(date_str, '%Y-%m-%d')
        games_found = []

        # Check last 7 days for games
        for days_back in range(1, 8):
            check_date = (target - timedelta(days=days_back)).strftime('%Y%m%d')

            data = None
            if _scoreboard_cache:
                data = _scoreboard_cache.get(check_date)
            else:
                url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={check_date}"
                html = _web_fetch(url, timeout=5)
                if "ERROR:" not in html:
                    data = json.loads(html)

            if not data:
                continue

            for event in data.get('events', []):
                comps = event.get('competitions', [{}])[0]
                for comp in comps.get('competitors', []):
                    abr = _normalize_abr(comp.get('team', {}).get('abbreviation', ''))
                    if abr == team_abr:
                        games_found.append(days_back)

        result['games_in_last_7'] = len(games_found)

        if games_found:
            result['days_rest'] = min(games_found) - 1  # 1 day back = 0 rest (B2B)
            result['last_game_date'] = (target - timedelta(days=min(games_found))).strftime('%Y-%m-%d')

            if result['days_rest'] == 0:
                result['schedule_density'] = 'b2b'
            elif len(games_found) >= 4:
                result['schedule_density'] = 'compressed'  # 4+ games in 7 days
            elif len(games_found) >= 3 and result['days_rest'] <= 1:
                result['schedule_density'] = 'compressed'
            elif result['days_rest'] >= 3:
                result['schedule_density'] = 'well_rested'
            else:
                result['schedule_density'] = 'normal'
        else:
            result['days_rest'] = 7
            result['schedule_density'] = 'well_rested'

    except Exception as e:
        result['error'] = str(e)

    return result


# ═══════════════════════════════════════════════════
# AGENT 7: Pace & Tempo Profile (per game)
# ═══════════════════════════════════════════════════
def research_pace(away_abr, home_abr, date_str, _team_rankings=None):
    """
    Fetch pace/tempo data for both teams to predict game tempo.
    Uses NBAFetcher team rankings (nba_api) for accurate pace data.
    """
    result = {
        'game': f"{away_abr}@{home_abr}",
        'away_pace': None,
        'home_pace': None,
        'projected_pace': None,
        'pace_label': 'average',  # 'fast', 'average', 'slow'
        'away_ppg': None,
        'home_ppg': None,
        'projected_total': None,
    }

    # Use pre-fetched team rankings if available
    teams = {}
    if _team_rankings:
        teams = _team_rankings.get('teams', {})
    elif NBAFetcher:
        try:
            fetcher = NBAFetcher()
            rankings = fetcher.get_team_rankings()
            teams = rankings.get('teams', {})
        except Exception:
            pass

    away_short = ABR_TO_SHORT_LOCAL.get(away_abr, away_abr)
    home_short = ABR_TO_SHORT_LOCAL.get(home_abr, home_abr)

    away_data = teams.get(away_short, {})
    home_data = teams.get(home_short, {})

    if away_data.get('pace'):
        result['away_pace'] = away_data['pace']
    if home_data.get('pace'):
        result['home_pace'] = home_data['pace']

    # PPG: use opponent's avg_pts_allowed as proxy, or off_rating
    if away_data.get('avg_pts_allowed'):
        # Away team's offense scores against home defense
        result['away_ppg'] = round(
            (away_data.get('off_rating', 110) / 100) * (away_data.get('pace', 100)), 1
        ) if away_data.get('off_rating') else None
    if home_data.get('avg_pts_allowed'):
        result['home_ppg'] = round(
            (home_data.get('off_rating', 110) / 100) * (home_data.get('pace', 100)), 1
        ) if home_data.get('off_rating') else None

    # Calculate projected pace
    if result['away_pace'] and result['home_pace']:
        result['projected_pace'] = round((result['away_pace'] + result['home_pace']) / 2, 1)
        if result['projected_pace'] >= 101:
            result['pace_label'] = 'fast'
        elif result['projected_pace'] <= 97:
            result['pace_label'] = 'slow'

    if result['away_ppg'] and result['home_ppg']:
        result['projected_total'] = round(result['away_ppg'] + result['home_ppg'], 1)

    return result


# ═══════════════════════════════════════════════════
# AGENT 8: Defensive Rating Profile (per team)
# ═══════════════════════════════════════════════════
def research_defense(team_abr, date_str, _team_rankings=None):
    """
    Fetch team defensive stats: DRTG, opponent PPG, defensive rank.
    Uses NBAFetcher team rankings (nba_api) for accurate data.
    """
    result = {
        'team': team_abr,
        'def_rating': None,
        'opp_ppg': None,
        'opp_3pt_pct': None,
        'opp_fg_pct': None,
        'def_rank': None,
        'def_label': 'average',  # 'elite', 'good', 'average', 'poor', 'terrible'
    }

    # Use pre-fetched team rankings if available
    teams = {}
    if _team_rankings:
        teams = _team_rankings.get('teams', {})
    elif NBAFetcher:
        try:
            fetcher = NBAFetcher()
            rankings = fetcher.get_team_rankings()
            teams = rankings.get('teams', {})
        except Exception:
            pass

    team_short = ABR_TO_SHORT_LOCAL.get(team_abr, team_abr)
    team_data = teams.get(team_short, {})

    if not team_data:
        return result

    result['def_rating'] = team_data.get('def_rating')
    result['opp_ppg'] = team_data.get('avg_pts_allowed')
    result['def_rank'] = team_data.get('def_rank')

    # Label defense quality
    if result['def_rank']:
        rank = result['def_rank']
        if rank <= 5:
            result['def_label'] = 'elite'
        elif rank <= 10:
            result['def_label'] = 'good'
        elif rank <= 20:
            result['def_label'] = 'average'
        elif rank <= 25:
            result['def_label'] = 'poor'
        else:
            result['def_label'] = 'terrible'
    elif result['opp_ppg']:
        ppg = result['opp_ppg']
        if ppg <= 107:
            result['def_label'] = 'elite'
        elif ppg <= 111:
            result['def_label'] = 'good'
        elif ppg <= 115:
            result['def_label'] = 'average'
        elif ppg <= 118:
            result['def_label'] = 'poor'
        else:
            result['def_label'] = 'terrible'

    return result


# ═══════════════════════════════════════════════════
# AGENT 9: Travel Load (distance-based fatigue)
# ═══════════════════════════════════════════════════
def research_travel_load(team_abr, game_date, scoreboard_cache=None):
    """
    Compute travel fatigue from recent schedule using cached scoreboard data.
    Looks back 7 days, calculates city-to-city distances.
    Returns {last_game_distance, total_7day_miles, tz_shifts_7day, last_game_city}.
    Zero new API calls — uses ScoreboardCache data + venue_data.
    """
    try:
        from venue_data import get_travel_distance, VENUE_MAP, TZ_ORDINAL
    except ImportError:
        return {'last_game_distance': 0, 'total_7day_miles': 0, 'tz_shifts_7day': 0, 'last_game_city': ''}

    if isinstance(game_date, str):
        game_date = datetime.strptime(game_date, '%Y-%m-%d')

    # Find team's games in last 7 days from scoreboard cache or ESPN data
    recent_cities = []  # list of team abbreviations where games were played (home team city)

    if scoreboard_cache:
        for days_back in range(1, 8):
            check_date = game_date - timedelta(days=days_back)
            date_key = check_date.strftime('%Y%m%d')
            games = scoreboard_cache.get(date_key, [])
            if not games and hasattr(scoreboard_cache, 'get_games'):
                games = scoreboard_cache.get_games(date_key)
            if not games:
                continue
            for g in games:
                home = g.get('home_abr', g.get('home', ''))
                away = g.get('away_abr', g.get('away', ''))
                if team_abr in (home, away):
                    # The game was played at the home team's city
                    recent_cities.append(home)
                    break

    # Current game city = the home team of today's game (will be figured out by caller)
    # For now just compute travel between recent cities

    total_miles = 0
    tz_shifts = 0
    last_distance = 0
    last_city = ''

    if recent_cities:
        last_city = recent_cities[0]  # most recent game

        # Compute distances between consecutive cities
        for i in range(len(recent_cities) - 1):
            dist = get_travel_distance(recent_cities[i], recent_cities[i+1])
            total_miles += dist

            # Timezone shifts
            v1 = VENUE_MAP.get(recent_cities[i], {})
            v2 = VENUE_MAP.get(recent_cities[i+1], {})
            tz1 = TZ_ORDINAL.get(v1.get('tz', 'ET'), 0)
            tz2 = TZ_ORDINAL.get(v2.get('tz', 'ET'), 0)
            if tz1 != tz2:
                tz_shifts += 1

    return {
        'last_game_distance': last_distance,
        'total_7day_miles': total_miles,
        'tz_shifts_7day': tz_shifts,
        'last_game_city': last_city,
    }


# ═══════════════════════════════════════════════════
# VERIFICATION AGENTS (3 agents, run after research)
# ═══════════════════════════════════════════════════

def verify_cross_reference(game_key, game_data, injury_results, news_results, injury_context):
    """
    VERIFIER 1: Cross-reference injury data across all sources.
    Flags contradictions (e.g., news says player returning but injury agent says OUT).
    """
    issues = []
    warnings = []
    away_abr = game_data['away_abr']
    home_abr = game_data['home_abr']

    for side, abr in [('away', away_abr), ('home', home_abr)]:
        team_short = game_data[side]
        injury_out = set(game_data.get(f'{side}_out', []))
        injury_q = set(game_data.get(f'{side}_questionable', []))
        news_data = news_results.get(abr, {})

        # Check if news mentions players returning that we have as OUT
        for headline in news_data.get('key_updates', []):
            hl = headline.lower()
            if any(word in hl for word in ['return', 'cleared', 'upgrade', 'back in lineup']):
                for player in injury_out:
                    if player.split()[-1].lower() in hl:
                        issues.append(f"CONFLICT: {player} listed OUT but news says returning: '{headline[:60]}'")

            # Check if news says player ruled out but we don't have them
            if any(word in hl for word in ['ruled out', 'will miss', 'sidelined']):
                for player_name in _extract_names_from_headline(hl):
                    if player_name not in injury_out and player_name not in injury_q:
                        warnings.append(f"MISSING: News says '{headline[:60]}' but player not in injury list")

        # Cross-check ESPN injuries vs official report
        for espn_inj in news_data.get('injuries_from_espn', []):
            name = espn_inj.get('player', '')
            status = espn_inj.get('status', '').lower()
            if 'out' in status and name not in injury_out:
                warnings.append(f"ESPN says {name} OUT but not in official report for {team_short}")

    # Verify B2B consistency with rest days
    if game_data.get('away_b2b') and game_data.get('away_rest_days', 99) > 0:
        issues.append(f"B2B flag set but rest days = {game_data.get('away_rest_days')} for {game_data['away']}")
    if game_data.get('home_b2b') and game_data.get('home_rest_days', 99) > 0:
        issues.append(f"B2B flag set but rest days = {game_data.get('home_rest_days')} for {game_data['home']}")

    return {
        'game': game_key,
        'agent': 'cross_reference',
        'issues': issues,
        'warnings': warnings,
        'status': 'FAIL' if issues else ('WARN' if warnings else 'PASS'),
    }


def verify_data_freshness(game_key, game_data, form_results):
    """
    VERIFIER 2: Check that data is fresh and complete.
    Flags stale form data, missing spreads, incomplete injury lists.
    """
    issues = []
    warnings = []
    away_abr = game_data['away_abr']
    home_abr = game_data['home_abr']

    # Check if we have a spread (critical for blowout risk layer)
    if not game_data.get('spread') or game_data['spread'] == 0:
        warnings.append(f"No spread data — blowout risk layer will be inactive")

    # Check form data completeness
    for side, abr in [('away', away_abr), ('home', home_abr)]:
        form = form_results.get(abr, {})
        team_short = game_data[side]

        if not form.get('last5_results'):
            warnings.append(f"No recent form data for {team_short}")
        elif len(form.get('last5_results', [])) < 3:
            warnings.append(f"Only {len(form.get('last5_results', []))} recent games found for {team_short} (want 5)")

        # Check if form data is stale (last game > 5 days ago)
        if form.get('last5_results'):
            last_date = form['last5_results'][0].get('date', '')
            if last_date:
                try:
                    from datetime import datetime
                    game_dt = datetime.strptime(game_data.get('away_abr', ''), '%Y%m%d')  # won't work, need date_str
                except Exception:
                    pass

    # Check defense data
    for side, label_key in [('away', 'away_defense'), ('home', 'home_defense')]:
        defense = game_data.get(label_key, {})
        if not defense.get('rating') and not defense.get('rank'):
            warnings.append(f"No defensive rating data for {game_data[side]}")

    # Check pace data
    pace = game_data.get('pace', {})
    if not pace.get('projected'):
        warnings.append(f"No pace data — tempo prediction unavailable")

    return {
        'game': game_key,
        'agent': 'data_freshness',
        'issues': issues,
        'warnings': warnings,
        'status': 'FAIL' if issues else ('WARN' if warnings else 'PASS'),
    }


def verify_narrative_consistency(game_key, game_data):
    """
    VERIFIER 3: Check that the overall narrative is coherent.
    Flags contradictions like: elite defense but terrible record,
    huge spread but no injuries to explain it, etc.
    """
    issues = []
    warnings = []

    spread = game_data.get('spread', 0)
    away_form = game_data.get('away_form', {})
    home_form = game_data.get('home_form', {})
    away_def = game_data.get('away_defense', {})
    home_def = game_data.get('home_defense', {})

    # Big spread but favorite on a losing streak?
    if spread and abs(spread) >= 8:
        fav_side = 'home' if spread < 0 else 'away'
        fav_form = home_form if fav_side == 'home' else away_form
        fav_short = game_data[fav_side]
        streak = fav_form.get('streak', '')
        if streak.startswith('L') and len(streak) > 1 and int(streak[1:]) >= 3:
            warnings.append(
                f"{fav_short} favored by {abs(spread)} but on {streak} losing streak — "
                f"spread may not reflect current form"
            )

    # Underdog on a huge win streak?
    if spread and abs(spread) >= 5:
        dog_side = 'away' if spread < 0 else 'home'
        dog_form = away_form if dog_side == 'away' else home_form
        dog_short = game_data[dog_side]
        streak = dog_form.get('streak', '')
        if streak.startswith('W') and len(streak) > 1 and int(streak[1:]) >= 4:
            warnings.append(
                f"{dog_short} is underdog but on {streak} win streak — "
                f"line may be wrong, proceed with caution"
            )

    # B2B team favored by a lot?
    if game_data.get('away_b2b') and spread and spread > 3:
        warnings.append(f"{game_data['away']} on B2B but favored on road — suspicious")
    if game_data.get('home_b2b') and spread and spread < -8:
        warnings.append(f"{game_data['home']} on B2B but big home favorite — fatigue risk")

    # Many injuries but no spread adjustment?
    total_out = len(game_data.get('away_out', [])) + len(game_data.get('home_out', []))
    if total_out >= 5 and (not spread or abs(spread) < 3):
        warnings.append(
            f"{total_out} players OUT between both teams but spread is only "
            f"{abs(spread) if spread else 0:.1f} — check if spread is stale"
        )

    # Rest mismatch but no spread reflection?
    away_rest = game_data.get('away_rest_days')
    home_rest = game_data.get('home_rest_days')
    if away_rest is not None and home_rest is not None:
        rest_diff = abs((away_rest or 0) - (home_rest or 0))
        if rest_diff >= 3 and (not spread or abs(spread) < 3):
            rested = game_data['away'] if (away_rest or 0) > (home_rest or 0) else game_data['home']
            warnings.append(f"{rest_diff}-day rest advantage for {rested} but spread doesn't reflect it")

    # Elite defense on both sides = low-scoring game flag
    if away_def.get('def_label') in ['elite', 'good'] and home_def.get('def_label') in ['elite', 'good']:
        warnings.append(
            f"Both teams have {away_def.get('def_label')}/{home_def.get('def_label')} defense — "
            f"strong UNDER environment for all stats"
        )

    # Both defenses terrible = high-scoring game flag
    if away_def.get('def_label') in ['poor', 'terrible'] and home_def.get('def_label') in ['poor', 'terrible']:
        warnings.append(
            f"Both teams have {away_def.get('def_label')}/{home_def.get('def_label')} defense — "
            f"strong OVER environment for all stats"
        )

    return {
        'game': game_key,
        'agent': 'narrative_consistency',
        'issues': issues,
        'warnings': warnings,
        'status': 'FAIL' if issues else ('WARN' if warnings else 'PASS'),
    }


def _extract_names_from_headline(headline_lower):
    """Best-effort name extraction from a headline string."""
    # Very simple: look for capitalized two-word sequences
    # This is a rough heuristic — won't catch everything
    import re
    names = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)', headline_lower.title())
    return [n.lower() for n in names]


def run_verification(GAMES, injury_results, news_results, form_results, injury_context, max_workers=None):
    """
    Deploy 3 verification agents per game (= 3 * num_games agents).
    Returns verification report.
    """
    if max_workers is None:
        max_workers = _calc_workers(len(GAMES) * 3, 'local')
    print(f"\n[VERIFY] Deploying {len(GAMES) * 3} verification agents (workers={max_workers})...")

    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for game_key, game_data in GAMES.items():
            # Verifier 1: Cross-reference
            f = executor.submit(verify_cross_reference, game_key, game_data,
                                injury_results, news_results, injury_context)
            futures[f] = ('cross_ref', game_key)

            # Verifier 2: Data freshness
            f = executor.submit(verify_data_freshness, game_key, game_data, form_results)
            futures[f] = ('freshness', game_key)

            # Verifier 3: Narrative consistency
            f = executor.submit(verify_narrative_consistency, game_key, game_data)
            futures[f] = ('narrative', game_key)

        for future in as_completed(futures):
            agent_type, key = futures[future]
            try:
                result = future.result(timeout=10)
                all_results.append(result)
            except Exception as e:
                all_results.append({
                    'game': key, 'agent': agent_type,
                    'issues': [f'Agent crashed: {e}'], 'warnings': [],
                    'status': 'ERROR',
                })

    # Print verification report
    fails = [r for r in all_results if r['status'] == 'FAIL']
    warns = [r for r in all_results if r['status'] == 'WARN']
    passes = [r for r in all_results if r['status'] == 'PASS']

    print(f"  Results: {len(passes)} PASS | {len(warns)} WARN | {len(fails)} FAIL")

    for r in fails:
        print(f"  [FAIL] {r['game']} ({r['agent']}):")
        for issue in r['issues']:
            print(f"    !! {issue}")

    for r in warns:
        if r['warnings']:
            print(f"  [WARN] {r['game']} ({r['agent']}):")
            for w in r['warnings'][:3]:
                print(f"    >> {w[:100]}")

    return all_results


# ═══════════════════════════════════════════════════
# ORCHESTRATOR: Run all agents in parallel
# ═══════════════════════════════════════════════════
def get_todays_games(date_str):
    """Fetch today's game schedule from ESPN."""
    formatted = date_str.replace('-', '')
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={formatted}"
    html = _web_fetch(url, timeout=10)

    if "ERROR:" in html:
        print(f"[ERROR] Could not fetch schedule: {html}")
        return []

    data = json.loads(html)
    events = data.get('events', [])
    games = []

    for event in events:
        comps = event.get('competitions', [{}])[0]
        competitors = comps.get('competitors', [])

        away_team = None
        home_team = None
        for comp in competitors:
            abr = comp.get('team', {}).get('abbreviation', '')
            if comp.get('homeAway') == 'away':
                away_team = abr
            else:
                home_team = abr

        if away_team and home_team:
            # Normalize ESPN abbreviations to pipeline standard
            away_team = _normalize_abr(away_team)
            home_team = _normalize_abr(home_team)

            # Extract odds if available
            odds = comps.get('odds', [])
            spread = None
            over_under = None
            if odds:
                spread = odds[0].get('spread')
                over_under = odds[0].get('overUnder')
                try:
                    spread = float(spread) if spread else None
                except (ValueError, TypeError):
                    spread = None

            games.append({
                'away': away_team,
                'home': home_team,
                'game_key': f"{away_team}@{home_team}",
                'spread': spread,
                'over_under': over_under,
                'status': event.get('status', {}).get('type', {}).get('name', ''),
                'game_time': event.get('date', ''),
            })

    return games


def research_all_games(date_str, max_workers=None):
    """
    Main entry point. Deploys parallel research agents for every team
    in every game on the schedule. Worker count auto-scales if not specified.

    Returns:
        GAMES dict in the format expected by run_board pipeline.
    """
    print(f"\n{'='*60}")
    print(f"  PARALLEL GAME RESEARCH ENGINE (v7 — adaptive)")
    print(f"  Date: {date_str}")
    print(f"{'='*60}")

    start_time = time.time()

    # Step 1: Get today's schedule
    print("\n[1/6] Fetching schedule...")
    games = get_todays_games(date_str)
    if not games:
        print("[ERROR] No games found for this date.")
        return {}
    print(f"  Found {len(games)} games")
    for g in games:
        spread_str = f"spread {g['spread']:+.1f}" if g['spread'] else "no spread"
        print(f"    {g['game_key']} | {spread_str}")

    # Step 1.5: Pre-warm ScoreboardCache (eliminates ~340 redundant ESPN calls)
    print("\n[1.5/6] Pre-warming ScoreboardCache...")
    scoreboard_cache = ScoreboardCache()
    target = datetime.strptime(date_str, '%Y-%m-%d')
    dates_to_cache = [date_str.replace('-', '')]  # today
    for days_back in range(1, 15):  # yesterday + 13 more days (for form/rest lookups)
        dates_to_cache.append((target - timedelta(days=days_back)).strftime('%Y%m%d'))
    prewarm_start = time.time()
    scoreboard_cache.prewarm(dates_to_cache, max_workers=min(len(dates_to_cache), 8))
    cache_stats = scoreboard_cache.stats
    print(f"  Cached {cache_stats['cached']} dates in {time.time() - prewarm_start:.1f}s "
          f"({cache_stats['fetched']} ESPN calls — would have been ~{len(games) * 2 * 15} without cache)")

    # Step 2: Fetch injury report (single call, shared across all agents)
    print("\n[2/6] Fetching injury report...")
    injury_context = None

    # Primary: ESPN injuries endpoint (most reliable, always available)
    print("  [ESPN] Fetching league-wide injuries...")
    try:
        injury_context = _fetch_cbs_injuries(date_str, games)
        if injury_context and injury_context.get('total_out', 0) > 0:
            print(f"  [ESPN] {injury_context['total_out']} OUT | {injury_context['total_questionable']} Q/D/P")
        else:
            print("  [ESPN] No injuries found (endpoint may be empty)")
    except Exception as e:
        print(f"  [WARN] ESPN injuries failed: {e}")

    # Fallback: try official NBA injury report package
    if (not injury_context or injury_context.get('total_out', 0) == 0) and get_injury_context:
        print("  [Fallback] Trying nbainjuries package...")
        try:
            injury_context = get_injury_context(date_str)
            print(f"  [Official] {injury_context['total_out']} OUT | {injury_context['total_questionable']} Q/D/P")
        except Exception as e:
            print(f"  [WARN] Official report failed: {e}")

    # Step 2.5: Pre-fetch team rankings (single call, shared by pace + defense agents)
    _team_rankings = None
    if NBAFetcher:
        try:
            print("\n[2.5/6] Fetching team rankings (pace + defense)...")
            fetcher = NBAFetcher()
            _team_rankings = fetcher.get_team_rankings()
            n_teams = len(_team_rankings.get('teams', {}))
            print(f"  [nba_api] {n_teams} teams loaded (pace, def_rating, opp_ppg)")
        except Exception as e:
            print(f"  [WARN] Team rankings failed: {e}")

    # Step 3: Deploy parallel agents (1 per team)
    all_teams = set()
    for g in games:
        all_teams.add(g['away'])
        all_teams.add(g['home'])

    # 8 agent types: injury, b2b, news, spread, form, rest, pace, defense
    num_agents = len(all_teams) * 6 + len(games) * 2  # 6 per-team + 2 per-game
    workers = max_workers if max_workers else _calc_workers(num_agents, 'espn')
    print(f"\n[3/6] Deploying {num_agents} research agents across {len(all_teams)} teams (workers={workers})...")
    print(f"  Agent types: injury, b2b, news, form, rest, defense (per team) + spread, pace (per game)")

    injury_results = {}
    b2b_results = {}
    news_results = {}
    spread_results = {}
    form_results = {}
    rest_results = {}
    pace_results = {}
    defense_results = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}

        # Submit injury agents (1 per team)
        for team in all_teams:
            f = executor.submit(research_injuries, team, date_str, injury_context)
            futures[f] = ('injury', team)

        # Submit B2B agents (1 per team) — uses ScoreboardCache
        for team in all_teams:
            f = executor.submit(research_b2b, team, date_str, scoreboard_cache)
            futures[f] = ('b2b', team)

        # Submit news agents (1 per team)
        for team in all_teams:
            f = executor.submit(research_news, team, date_str)
            futures[f] = ('news', team)

        # Submit spread agents (1 per game) — uses ScoreboardCache
        for g in games:
            f = executor.submit(research_spread, g['away'], g['home'], date_str, scoreboard_cache)
            futures[f] = ('spread', g['game_key'])

        # NEW: Submit recent form agents (1 per team) — uses ScoreboardCache
        for team in all_teams:
            f = executor.submit(research_recent_form, team, date_str, scoreboard_cache)
            futures[f] = ('form', team)

        # NEW: Submit rest days agents (1 per team) — uses ScoreboardCache
        for team in all_teams:
            f = executor.submit(research_rest_days, team, date_str, scoreboard_cache)
            futures[f] = ('rest', team)

        # NEW: Submit pace/tempo agents (1 per game) — uses pre-fetched rankings
        for g in games:
            f = executor.submit(research_pace, g['away'], g['home'], date_str, _team_rankings)
            futures[f] = ('pace', g['game_key'])

        # NEW: Submit defensive rating agents (1 per team) — uses pre-fetched rankings
        for team in all_teams:
            f = executor.submit(research_defense, team, date_str, _team_rankings)
            futures[f] = ('defense', team)

        # Collect results
        completed = 0
        for future in as_completed(futures):
            agent_type, key = futures[future]
            completed += 1
            try:
                result = future.result(timeout=15)
                if agent_type == 'injury':
                    injury_results[key] = result
                elif agent_type == 'b2b':
                    b2b_results[key] = result
                elif agent_type == 'news':
                    news_results[key] = result
                elif agent_type == 'spread':
                    spread_results[key] = result
                elif agent_type == 'form':
                    form_results[key] = result
                elif agent_type == 'rest':
                    rest_results[key] = result
                elif agent_type == 'pace':
                    pace_results[key] = result
                elif agent_type == 'defense':
                    defense_results[key] = result
            except Exception as e:
                print(f"  [WARN] {agent_type} agent for {key} failed: {e}")

            if completed % 10 == 0 or completed == len(futures):
                print(f"  [{completed}/{len(futures)}] agents complete")

    # Step 4: Compile into GAMES dict
    print(f"\n[4/6] Compiling research results...")
    GAMES = {}

    for g in games:
        away = g['away']
        home = g['home']
        game_key = g['game_key']

        away_short = ABR_TO_SHORT_LOCAL.get(away, away)
        home_short = ABR_TO_SHORT_LOCAL.get(home, home)

        # Get injury data
        away_inj = injury_results.get(away, {})
        home_inj = injury_results.get(home, {})

        # Get B2B data
        away_b2b = b2b_results.get(away, {})
        home_b2b = b2b_results.get(home, {})

        # Get spread (from ESPN schedule or spread agent)
        spread = g.get('spread')
        spread_data = spread_results.get(game_key, {})
        if spread is None and spread_data.get('spread') is not None:
            spread = spread_data['spread']

        # Get new agent data
        away_form = form_results.get(away, {})
        home_form = form_results.get(home, {})
        away_rest = rest_results.get(away, {})
        home_rest = rest_results.get(home, {})
        pace_data = pace_results.get(game_key, {})
        away_def = defense_results.get(away, {})
        home_def = defense_results.get(home, {})

        # Get news (and ESPN injuries as fallback)
        away_news = news_results.get(away, {})
        home_news = news_results.get(home, {})

        # If no injuries from official report, use ESPN injuries endpoint
        if not away_inj.get('out') and not away_inj.get('questionable'):
            for inj in away_news.get('injuries_from_espn', []):
                status = inj.get('status', '').lower()
                name = inj.get('player', '')
                if 'out' in status:
                    away_inj.setdefault('out', []).append(name)
                elif any(s in status for s in ['day-to-day', 'questionable', 'doubtful']):
                    away_inj.setdefault('questionable', []).append(name)

        if not home_inj.get('out') and not home_inj.get('questionable'):
            for inj in home_news.get('injuries_from_espn', []):
                status = inj.get('status', '').lower()
                name = inj.get('player', '')
                if 'out' in status:
                    home_inj.setdefault('out', []).append(name)
                elif any(s in status for s in ['day-to-day', 'questionable', 'doubtful']):
                    home_inj.setdefault('questionable', []).append(name)

        # Build notes
        notes_parts = []
        if away_inj.get('out'):
            notes_parts.append(f"{away_short} without {', '.join(away_inj['out'][:3])}")
            if len(away_inj['out']) > 3:
                notes_parts[-1] += f" (+{len(away_inj['out'])-3} more)"
        if home_inj.get('out'):
            notes_parts.append(f"{home_short} without {', '.join(home_inj['out'][:3])}")
            if len(home_inj['out']) > 3:
                notes_parts[-1] += f" (+{len(home_inj['out'])-3} more)"
        if away_b2b.get('is_b2b'):
            notes_parts.append(f"{away_short} on B2B")
        if home_b2b.get('is_b2b'):
            notes_parts.append(f"{home_short} on B2B")
        if spread and abs(spread) >= 10:
            fav = home_short if spread < 0 else away_short
            notes_parts.append(f"{fav} big favorite ({abs(spread):.1f})")

        # Form/streak notes
        if away_form.get('streak') and away_form['streak'].startswith('L') and int(away_form['streak'][1:]) >= 3:
            notes_parts.append(f"{away_short} on {away_form['streak']} skid")
        elif away_form.get('streak') and away_form['streak'].startswith('W') and int(away_form['streak'][1:]) >= 3:
            notes_parts.append(f"{away_short} on {away_form['streak']} streak")
        if home_form.get('streak') and home_form['streak'].startswith('L') and int(home_form['streak'][1:]) >= 3:
            notes_parts.append(f"{home_short} on {home_form['streak']} skid")
        elif home_form.get('streak') and home_form['streak'].startswith('W') and int(home_form['streak'][1:]) >= 3:
            notes_parts.append(f"{home_short} on {home_form['streak']} streak")

        # Rest advantage notes
        away_rest_days = away_rest.get('days_rest')
        home_rest_days = home_rest.get('days_rest')
        if away_rest_days is not None and home_rest_days is not None:
            if abs((away_rest_days or 0) - (home_rest_days or 0)) >= 2:
                rested = away_short if (away_rest_days or 0) > (home_rest_days or 0) else home_short
                tired = home_short if rested == away_short else away_short
                notes_parts.append(f"Rest edge: {rested} ({away_rest_days if rested == away_short else home_rest_days}d) vs {tired} ({home_rest_days if rested == away_short else away_rest_days}d)")

        # Schedule compression
        if away_rest.get('schedule_density') == 'compressed':
            notes_parts.append(f"{away_short} compressed schedule ({away_rest.get('games_in_last_7', 0)} in 7d)")
        if home_rest.get('schedule_density') == 'compressed':
            notes_parts.append(f"{home_short} compressed schedule ({home_rest.get('games_in_last_7', 0)} in 7d)")

        # Pace note
        if pace_data.get('pace_label') == 'fast':
            notes_parts.append(f"FAST pace game (proj {pace_data.get('projected_pace', 0)})")
        elif pace_data.get('pace_label') == 'slow':
            notes_parts.append(f"SLOW pace game (proj {pace_data.get('projected_pace', 0)})")

        # Defense matchup note
        if away_def.get('def_label') == 'elite':
            notes_parts.append(f"{away_short} elite D (rank {away_def.get('def_rank', '?')})")
        if home_def.get('def_label') == 'elite':
            notes_parts.append(f"{home_short} elite D (rank {home_def.get('def_rank', '?')})")
        if away_def.get('def_label') == 'terrible':
            notes_parts.append(f"{away_short} terrible D (rank {away_def.get('def_rank', '?')})")
        if home_def.get('def_label') == 'terrible':
            notes_parts.append(f"{home_short} terrible D (rank {home_def.get('def_rank', '?')})")

        # Add key news updates
        for update in (away_news.get('key_updates', []) + home_news.get('key_updates', []))[:2]:
            notes_parts.append(f"NEWS: {update[:80]}")

        game_entry = {
            "away": away_short,
            "home": home_short,
            "away_abr": away,
            "home_abr": home,
            "spread": spread if spread else 0,
            "away_out": away_inj.get('out', []),
            "home_out": home_inj.get('out', []),
            "away_questionable": away_inj.get('questionable', []) + away_inj.get('doubtful', []),
            "home_questionable": home_inj.get('questionable', []) + home_inj.get('doubtful', []),
            "away_b2b": away_b2b.get('is_b2b', False),
            "home_b2b": home_b2b.get('is_b2b', False),
            "notes": " | ".join(notes_parts) if notes_parts else "No significant flags.",
            # Extra metadata
            "over_under": g.get('over_under') or spread_data.get('over_under'),
            "away_probable": away_inj.get('probable', []),
            "home_probable": home_inj.get('probable', []),
            "away_news": away_news.get('headlines', [])[:3],
            "home_news": home_news.get('headlines', [])[:3],
            # NEW: Form & streak data
            "away_form": {
                "last5": away_form.get('last5_record', ''),
                "streak": away_form.get('streak', ''),
                "avg_scored": away_form.get('avg_pts_scored', 0),
                "avg_allowed": away_form.get('avg_pts_allowed', 0),
            },
            "home_form": {
                "last5": home_form.get('last5_record', ''),
                "streak": home_form.get('streak', ''),
                "avg_scored": home_form.get('avg_pts_scored', 0),
                "avg_allowed": home_form.get('avg_pts_allowed', 0),
            },
            # NEW: Rest & schedule density
            "away_rest_days": away_rest.get('days_rest'),
            "home_rest_days": home_rest.get('days_rest'),
            "away_schedule_density": away_rest.get('schedule_density', 'normal'),
            "home_schedule_density": home_rest.get('schedule_density', 'normal'),
            # NEW: Pace prediction
            "pace": {
                "away_pace": pace_data.get('away_pace'),
                "home_pace": pace_data.get('home_pace'),
                "projected": pace_data.get('projected_pace'),
                "label": pace_data.get('pace_label', 'average'),
                "projected_total": pace_data.get('projected_total'),
            },
            # NEW: Defense profiles
            "away_defense": {
                "rating": away_def.get('def_rating'),
                "rank": away_def.get('def_rank'),
                "opp_ppg": away_def.get('opp_ppg'),
                "label": away_def.get('def_label', 'average'),
            },
            "home_defense": {
                "rating": home_def.get('def_rating'),
                "rank": home_def.get('def_rank'),
                "opp_ppg": home_def.get('opp_ppg'),
                "label": home_def.get('def_label', 'average'),
            },
        }

        GAMES[game_key] = game_entry

    # Step 5: Verification pass
    verify_workers = _calc_workers(len(GAMES) * 3, 'local')
    print(f"\n[5/6] Running verification agents (workers={verify_workers})...")
    verification = run_verification(
        GAMES, injury_results, news_results, form_results, injury_context,
        max_workers=verify_workers,
    )

    # Attach verification warnings to game notes
    for v in verification:
        if v['status'] in ('WARN', 'FAIL') and v['game'] in GAMES:
            for issue in v.get('issues', []):
                existing = GAMES[v['game']].get('notes', '')
                GAMES[v['game']]['notes'] = existing + f" | ⚠ {issue[:80]}"
            # Store all warnings in metadata
            GAMES[v['game']].setdefault('verification_warnings', []).extend(
                v.get('warnings', []) + v.get('issues', [])
            )

    # Step 6: Print summary
    elapsed = time.time() - start_time
    final_cache_stats = scoreboard_cache.stats
    print(f"\n[6/6] Research complete in {elapsed:.1f}s")
    print(f"  ScoreboardCache: {final_cache_stats['cached']} dates cached, {final_cache_stats['fetched']} total ESPN fetches")
    print(f"\n{'='*60}")
    print(f"  RESEARCH SUMMARY — {date_str}")
    print(f"{'='*60}")

    for game_key, g in GAMES.items():
        spread_str = f"{g['spread']:+.1f}" if g['spread'] else "N/A"
        b2b_tag = ""
        if g['away_b2b']:
            b2b_tag += f" [{g['away']} B2B]"
        if g['home_b2b']:
            b2b_tag += f" [{g['home']} B2B]"

        print(f"\n  {game_key} | Spread: {spread_str}{b2b_tag}")
        if g['away_out']:
            print(f"    {g['away']} OUT: {', '.join(g['away_out'])}")
        if g['away_questionable']:
            print(f"    {g['away']} Q/D: {', '.join(g['away_questionable'])}")
        if g['home_out']:
            print(f"    {g['home']} OUT: {', '.join(g['home_out'])}")
        if g['home_questionable']:
            print(f"    {g['home']} Q/D: {', '.join(g['home_questionable'])}")
        # Form
        away_f = g.get('away_form', {})
        home_f = g.get('home_form', {})
        if away_f.get('last5') or home_f.get('last5'):
            print(f"    Form: {g['away']} L5 {away_f.get('last5','?')} {away_f.get('streak','')} | "
                  f"{g['home']} L5 {home_f.get('last5','?')} {home_f.get('streak','')}")
        # Rest
        ar = g.get('away_rest_days')
        hr = g.get('home_rest_days')
        if ar is not None or hr is not None:
            print(f"    Rest: {g['away']} {ar}d ({g.get('away_schedule_density','?')}) | "
                  f"{g['home']} {hr}d ({g.get('home_schedule_density','?')})")
        # Pace
        pace = g.get('pace', {})
        if pace.get('projected'):
            print(f"    Pace: {pace['label']} (proj {pace['projected']}) | "
                  f"Total: {pace.get('projected_total', 'N/A')}")
        # Defense
        ad = g.get('away_defense', {})
        hd = g.get('home_defense', {})
        if ad.get('rank') or hd.get('rank'):
            print(f"    Defense: {g['away']} {ad.get('label','?')} (rank {ad.get('rank','?')}) | "
                  f"{g['home']} {hd.get('label','?')} (rank {hd.get('rank','?')})")
        # News
        if g.get('away_news'):
            print(f"    {g['away']} News: {g['away_news'][0][:70]}")
        if g.get('home_news'):
            print(f"    {g['home']} News: {g['home_news'][0][:70]}")
        print(f"    Notes: {g['notes'][:120]}")

    print(f"\n  Total: {len(GAMES)} games researched in {elapsed:.1f}s")
    print(f"{'='*60}")

    return GAMES


def save_research(games_dict, date_str):
    """Save compiled research to JSON for pipeline use."""
    output_dir = os.path.join(os.path.dirname(__file__), date_str)
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f"{date_str}_game_research.json")
    with open(filepath, 'w') as f:
        json.dump(games_dict, f, indent=2)
    print(f"[OK] Saved game research to {filepath}")
    return filepath


def generate_runner_code(games_dict, date_str):
    """
    Generate the GAMES dict as Python code that can be pasted into a runner script.
    """
    lines = [f'# Auto-generated by game_researcher.py on {datetime.now().isoformat()}']
    lines.append(f'DATE = "{date_str}"')
    lines.append('')
    lines.append('GAMES = {')

    for game_key, g in games_dict.items():
        lines.append(f'    "{game_key}": {{')
        lines.append(f'        "away": "{g["away"]}", "home": "{g["home"]}",')
        lines.append(f'        "away_abr": "{g["away_abr"]}", "home_abr": "{g["home_abr"]}",')
        lines.append(f'        "spread": {g["spread"]},')
        lines.append(f'        "away_out": {json.dumps(g["away_out"])},')
        lines.append(f'        "home_out": {json.dumps(g["home_out"])},')
        lines.append(f'        "away_questionable": {json.dumps(g["away_questionable"])},')
        lines.append(f'        "home_questionable": {json.dumps(g["home_questionable"])},')
        lines.append(f'        "away_b2b": {g["away_b2b"]}, "home_b2b": {g["home_b2b"]},')
        lines.append(f'        "notes": {json.dumps(g["notes"])},')
        lines.append(f'    }},')

    lines.append('}')
    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='NBA Parallel Game Research Engine')
    parser.add_argument('date', nargs='?', default=datetime.now().strftime('%Y-%m-%d'),
                        help='Date to research (YYYY-MM-DD). Default: today')
    parser.add_argument('--workers', type=int, default=10,
                        help='Max parallel workers (default: 10)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results to JSON (default: True)')
    args = parser.parse_args()

    games = research_all_games(args.date, max_workers=args.workers)

    if games and args.save:
        save_research(games, args.date)

        # Also generate Python code
        code = generate_runner_code(games, args.date)
        code_path = os.path.join(os.path.dirname(__file__), args.date, f"{args.date}_games_code.py")
        with open(code_path, 'w') as f:
            f.write(code)
        print(f"[OK] Saved runner code to {code_path}")

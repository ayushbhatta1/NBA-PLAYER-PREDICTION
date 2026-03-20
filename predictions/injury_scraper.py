#!/usr/bin/env python3
"""
NBA Injury Report Scraper
Pulls official NBA injury reports and Basketball-Reference data
to auto-populate injury context for the v3 pipeline.

Data sources:
1. Official NBA Injury Report (via nbainjuries package) — primary
2. Basketball-Reference injuries page — fallback

Output format matches what analyze_v3.py expects:
- injured_out: List of player names confirmed OUT
- player_statuses: Dict of {player_name: status} for GTD/Questionable/Probable
- game_contexts: Dict of {game_label: {injuries, b2b, spread, ...}}
"""

import json
import os
from datetime import datetime, timedelta

# ── Team name normalization ──
# Maps various formats to our pipeline's short names
TEAM_TO_SHORT = {
    'Cleveland Cavaliers': 'Cavaliers', 'Orlando Magic': 'Magic',
    'Toronto Raptors': 'Raptors', 'New Orleans Pelicans': 'Pelicans',
    'New York Knicks': 'Knicks', 'Utah Jazz': 'Jazz',
    'Charlotte Hornets': 'Hornets', 'Sacramento Kings': 'Kings',
    'Houston Rockets': 'Rockets', 'Denver Nuggets': 'Nuggets',
    'Minnesota Timberwolves': 'Timberwolves', 'LA Clippers': 'Clippers',
    'Philadelphia 76ers': '76ers', 'Detroit Pistons': 'Pistons',
    'Phoenix Suns': 'Suns', 'Indiana Pacers': 'Pacers',
    'Washington Wizards': 'Wizards', 'Brooklyn Nets': 'Nets',
    'Atlanta Hawks': 'Hawks', 'Milwaukee Bucks': 'Bucks',
    'Miami Heat': 'Heat', 'Dallas Mavericks': 'Mavericks',
    'Memphis Grizzlies': 'Grizzlies', 'Boston Celtics': 'Celtics',
    'Oklahoma City Thunder': 'Thunder', 'Chicago Bulls': 'Bulls',
    'Los Angeles Lakers': 'Lakers', 'San Antonio Spurs': 'Spurs',
    'Golden State Warriors': 'Warriors', 'Portland Trail Blazers': 'Trail Blazers',
}

# Abbreviation → short name
ABR_TO_SHORT = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards',
}

SHORT_TO_ABR = {v: k for k, v in ABR_TO_SHORT.items()}


def _normalize_name(name_str):
    """Convert 'Last, First' format to 'First Last'"""
    if ',' in name_str:
        parts = name_str.split(',', 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name_str.strip()


def _team_short(full_name):
    """Convert full team name to short name"""
    short = TEAM_TO_SHORT.get(full_name)
    if short:
        return short
    # Try partial match
    for k, v in TEAM_TO_SHORT.items():
        if v.lower() in full_name.lower() or full_name.lower() in k.lower():
            return v
    return full_name


def _parse_matchup(matchup_str):
    """Parse 'CLE@ORL' into (away_abr, home_abr)"""
    if '@' in matchup_str:
        parts = matchup_str.split('@')
        return parts[0].strip(), parts[1].strip()
    return matchup_str, matchup_str


def fetch_official_report(game_date=None, hour=17, minute=0):
    """
    Fetch the official NBA injury report for a given date.

    Args:
        game_date: datetime or 'YYYY-MM-DD' string. Defaults to today.
        hour: Hour of the report snapshot (default 17 = 5pm ET)
        minute: Minute of the report snapshot (default 0)

    Returns:
        DataFrame with columns: Game Date, Game Time, Matchup, Team,
                                Player Name, Current Status, Reason
    """
    from nbainjuries import injury

    if game_date is None:
        game_date = datetime.now()
    elif isinstance(game_date, str):
        game_date = datetime.strptime(game_date, '%Y-%m-%d')

    report_dt = datetime(
        year=game_date.year, month=game_date.month, day=game_date.day,
        hour=hour, minute=minute
    )

    try:
        df = injury.get_reportdata(report_dt, return_df=True)
        if df is not None and not df.empty:
            # Drop rows with NaN player names or statuses (e.g., "NOT YET SUBMITTED")
            df = df.dropna(subset=['Player Name', 'Current Status']).copy()
            # Filter to only today's games (report may include tomorrow's)
            target_date = game_date.strftime('%m/%d/%Y')
            if 'Game Date' in df.columns:
                df = df[df['Game Date'] == target_date].copy()
            # Normalize player names from "Last, First" → "First Last"
            df['Player Name'] = df['Player Name'].apply(_normalize_name)
            # Ensure Reason column has no NaN
            df['Reason'] = df['Reason'].fillna('Unknown')
            return df
    except Exception as e:
        print(f"[WARN] Official report failed for {report_dt}: {e}")
        # Try earlier snapshots
        for h in [17, 16, 15, 14, 13, 12]:
            if h == hour:
                continue
            try:
                fallback_dt = datetime(
                    year=game_date.year, month=game_date.month, day=game_date.day,
                    hour=h, minute=0
                )
                df = injury.get_reportdata(fallback_dt, return_df=True)
                if df is not None and not df.empty:
                    df = df.dropna(subset=['Player Name', 'Current Status']).copy()
                    target_date = game_date.strftime('%m/%d/%Y')
                    if 'Game Date' in df.columns:
                        df = df[df['Game Date'] == target_date].copy()
                    df['Player Name'] = df['Player Name'].apply(_normalize_name)
                    df['Reason'] = df['Reason'].fillna('Unknown')
                    print(f"[OK] Got report from {h}:00 snapshot")
                    return df
            except Exception:
                continue

    print("[ERROR] Could not fetch official injury report from any snapshot")
    return None


def get_injury_context(game_date=None):
    """
    Main entry point: fetch injuries and build context dicts for the pipeline.

    Returns:
        dict with:
        - 'injured_out': {team_short: [player_names]}  — confirmed OUT players
        - 'all_out': [player_names]                     — flat list of all OUT players
        - 'player_statuses': {player_name: status}      — GTD/Questionable/Doubtful/Probable
        - 'games': [{matchup, away, home, away_out, home_out, away_questionable, home_questionable}]
        - 'report_date': str
        - 'total_out': int
        - 'total_questionable': int
        - 'source': str
    """
    df = fetch_official_report(game_date)
    if df is None:
        return _empty_context(game_date)

    injured_out = {}    # team → [names]
    all_out = []
    player_statuses = {}
    games_dict = {}     # matchup → game info

    # Filter out G-League / Two-Way assignments (not real injuries)
    real_injuries = df[~df['Reason'].str.contains('G League|Two-Way', case=False, na=False)].copy()

    for _, row in df.iterrows():
        team_full = row['Team']
        team_short = _team_short(team_full)
        player = row['Player Name']
        status = row['Current Status']
        reason = row.get('Reason', '')
        matchup = row.get('Matchup', '')

        # Skip G-League / Two-Way for injury tracking (they don't affect prop analysis)
        is_gleague = 'G League' in str(reason) or 'Two-Way' in str(reason)

        if status == 'Out':
            if team_short not in injured_out:
                injured_out[team_short] = []
            injured_out[team_short].append({
                'name': player,
                'reason': reason,
                'is_gleague': is_gleague,
            })
            if not is_gleague:
                all_out.append(player)

        elif status in ['Questionable', 'Doubtful', 'Probable', 'Game Time Decision']:
            mapped_status = status
            if status == 'Doubtful':
                mapped_status = 'Doubtful'  # treat separately — likely OUT
            player_statuses[player] = {
                'status': mapped_status,
                'team': team_short,
                'reason': reason,
            }

        # Build game context
        if matchup and matchup not in games_dict:
            away_abr, home_abr = _parse_matchup(matchup)
            away_short = ABR_TO_SHORT.get(away_abr, away_abr)
            home_short = ABR_TO_SHORT.get(home_abr, home_abr)
            games_dict[matchup] = {
                'matchup': matchup,
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

        # Populate game injury lists (skip G-League)
        if matchup in games_dict and not is_gleague:
            away_abr, home_abr = _parse_matchup(matchup)
            away_short = ABR_TO_SHORT.get(away_abr, away_abr)

            if team_short == away_short:
                side = 'away'
            else:
                side = 'home'

            if status == 'Out':
                games_dict[matchup][f'{side}_out'].append(player)
            elif status in ['Questionable', 'Doubtful']:
                games_dict[matchup][f'{side}_questionable'].append(player)

    # Build games list
    games = list(games_dict.values())

    date_str = game_date.strftime('%Y-%m-%d') if isinstance(game_date, datetime) else str(game_date) if game_date else datetime.now().strftime('%Y-%m-%d')

    return {
        'injured_out': injured_out,
        'all_out': all_out,
        'player_statuses': player_statuses,
        'games': games,
        'report_date': date_str,
        'total_out': len(all_out),
        'total_questionable': len(player_statuses),
        'source': 'Official NBA Injury Report (nbainjuries)',
    }


def _empty_context(game_date):
    """Return empty context when no report is available"""
    date_str = game_date.strftime('%Y-%m-%d') if isinstance(game_date, datetime) else str(game_date) if game_date else datetime.now().strftime('%Y-%m-%d')
    return {
        'injured_out': {},
        'all_out': [],
        'player_statuses': {},
        'games': [],
        'report_date': date_str,
        'total_out': 0,
        'total_questionable': 0,
        'source': 'empty (report unavailable)',
    }


def get_injured_out_for_game(injury_context, team_short):
    """
    Get list of OUT player names for a specific team from the injury context.
    This is what analyze_v3.py needs for the 'injured_out' parameter.
    Filters out G-League / Two-Way assignments.
    """
    team_injuries = injury_context.get('injured_out', {}).get(team_short, [])
    return [p['name'] for p in team_injuries if not p.get('is_gleague', False)]


def get_player_status(injury_context, player_name):
    """
    Get injury status for a specific player.
    Returns: 'Out', 'Questionable', 'Doubtful', 'Probable', or None
    """
    # Check OUT list first
    for team, players in injury_context.get('injured_out', {}).items():
        for p in players:
            if _name_match(player_name, p['name']):
                return 'Out'

    # Check questionable/probable
    for name, info in injury_context.get('player_statuses', {}).items():
        if _name_match(player_name, name):
            return info['status']

    return None


def _name_match(name1, name2):
    """Simple name matching"""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return True
    # Last name + first initial match
    p1 = n1.split()
    p2 = n2.split()
    if len(p1) >= 2 and len(p2) >= 2:
        if p1[-1] == p2[-1] and p1[0][0] == p2[0][0]:
            return True
    return False


def get_game_injury_summary(injury_context, game_label):
    """
    Get a summary of injuries for a specific game.
    game_label format: "Rockets@Nuggets" or "HOU@DEN"
    """
    for game in injury_context.get('games', []):
        if game['label'] == game_label or game['matchup'] == game_label:
            return game
    return None


def print_injury_report(injury_context):
    """Pretty print the injury report"""
    ctx = injury_context
    print(f"\n{'='*60}")
    print(f"  NBA INJURY REPORT — {ctx['report_date']}")
    print(f"  Source: {ctx['source']}")
    print(f"  {ctx['total_out']} players OUT | {ctx['total_questionable']} Questionable/Doubtful/Probable")
    print(f"{'='*60}\n")

    for game in ctx.get('games', []):
        print(f"  {game['label']}")
        if game['away_out']:
            print(f"    {game['away']} OUT: {', '.join(game['away_out'])}")
        if game['away_questionable']:
            print(f"    {game['away']} Q/D: {', '.join(game['away_questionable'])}")
        if game['home_out']:
            print(f"    {game['home']} OUT: {', '.join(game['home_out'])}")
        if game['home_questionable']:
            print(f"    {game['home']} Q/D: {', '.join(game['home_questionable'])}")
        if not any([game['away_out'], game['away_questionable'], game['home_out'], game['home_questionable']]):
            print(f"    No significant injuries reported")
        print()


def save_injury_report(injury_context, save_dir='predictions'):
    """Save injury context to JSON for pipeline use"""
    date_str = injury_context['report_date']
    date_dir = os.path.join(save_dir, date_str)
    os.makedirs(date_dir, exist_ok=True)

    filepath = os.path.join(date_dir, f'{date_str}_injuries.json')
    with open(filepath, 'w') as f:
        json.dump(injury_context, f, indent=2, default=str)
    print(f"[OK] Saved injury report to {filepath}")
    return filepath


if __name__ == '__main__':
    print("Fetching today's NBA injury report...")
    ctx = get_injury_context()
    print_injury_report(ctx)
    save_injury_report(ctx)

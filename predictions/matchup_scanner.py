#!/usr/bin/env python3
"""
Matchup Scanner v1 -- Defense-First Prop Selection

Flips the pipeline: instead of scoring 400 props and picking the best,
starts with DEFENSIVE WEAKNESSES and finds players who exploit them.

DaftPreviews hits 62% by asking "which defense is weak?" then finding
the player who exploits it. OddsJam says "find mispricing, don't predict."
Sharp bettors say "be selective -- only bet the top 3-5 edges."

Usage:
    from matchup_scanner import scan_tonight, enrich_with_matchup_scan

    # Standalone scan
    picks = scan_tonight(board_props, GAMES, max_picks=5)

    # Pipeline integration
    enrich_with_matchup_scan(results, GAMES)

CLI:
    python3 predictions/matchup_scanner.py --board /path/to/board.json --games /path/to/research.json
    python3 predictions/matchup_scanner.py --test
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from nba_fetcher import NBAFetcher, TEAM_ABR
except ImportError:
    NBAFetcher = None
    TEAM_ABR = {}

try:
    from pbp_client import PBPStatsClient, TEAM_ABR_TO_ID
except ImportError:
    PBPStatsClient = None
    TEAM_ABR_TO_ID = {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map stat category names to the team_rankings keys that measure
# how much a defense ALLOWS in that category.
STAT_TO_DEFENSE_KEY = {
    'pts':  'avg_pts_allowed',
    'reb':  'reb_allowed',
    'ast':  'ast_allowed',
    '3pm':  'tpm_allowed',
    'stl':  'stl_allowed',
    'blk':  'blk_allowed',
}

STAT_TO_RANK_KEY = {
    'pts':  'avg_pts_allowed_rank',
    'reb':  'reb_allowed_rank',
    'ast':  'ast_allowed_rank',
    '3pm':  'tpm_allowed_rank',
    'stl':  None,   # no rank computed for stl/blk in current rankings
    'blk':  None,
}

# Minimum severity (rank 20+ out of 30 = bottom third) to flag as a weakness.
# Lower threshold catches more; higher is more selective.
WEAKNESS_RANK_THRESHOLD = 20   # top-10 worst = rank 21-30

# Line softness: the matchup-adjusted projection must exceed the line
# by at least this fraction to call it soft.
SOFTNESS_MIN_MARGIN_PCT = 0.08  # 8%

# Matchup grade thresholds (projected margin over line as percentage)
GRADE_THRESHOLDS = {
    'S': 0.20,   # 20%+ over line
    'A': 0.15,
    'B': 0.10,
    'C': 0.05,
}

# Blowout minutes discount by spread
BLOWOUT_DISCOUNT = {
    10: 0.05,
    12: 0.08,
    15: 0.12,
    18: 0.17,
    20: 0.20,
}

# Per-minute production baseline context -- used when player data unavailable
LEAGUE_AVG_PER_MIN = {
    'pts': 0.61,
    'reb': 0.27,
    'ast': 0.22,
    '3pm': 0.10,
    'stl': 0.05,
    'blk': 0.04,
}

# Combo stat components
COMBO_COMPONENTS = {
    'pra': ['pts', 'reb', 'ast'],
    'pr':  ['pts', 'reb'],
    'pa':  ['pts', 'ast'],
    'ra':  ['reb', 'ast'],
    'stl_blk': ['stl', 'blk'],
}

COMBO_STATS = set(COMBO_COMPONENTS.keys())
BASE_STATS = {'pts', 'reb', 'ast', '3pm', 'stl', 'blk'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _abr_to_short(abr):
    """Abbreviation (e.g. 'BOS') to short name (e.g. 'Celtics')."""
    return TEAM_ABR.get(abr, abr)


def _short_to_abr(short):
    """Short name (e.g. 'Celtics') to abbreviation (e.g. 'BOS')."""
    for abr, name in TEAM_ABR.items():
        if name == short:
            return abr
    return short


def _game_teams(game_key):
    """Parse 'AWAY@HOME' into (away_abr, home_abr)."""
    if '@' not in game_key:
        return None, None
    parts = game_key.split('@')
    return parts[0].strip().upper(), parts[1].strip().upper() if len(parts) > 1 else (None, None)


def _blowout_minutes_discount(spread):
    """Return fractional discount [0, 0.20] based on absolute spread."""
    abs_spread = abs(spread or 0)
    if abs_spread < 10:
        return 0.0
    for threshold in sorted(BLOWOUT_DISCOUNT.keys(), reverse=True):
        if abs_spread >= threshold:
            return BLOWOUT_DISCOUNT[threshold]
    return 0.0


# ---------------------------------------------------------------------------
# 1. SCAN DEFENSIVE WEAKNESSES
# ---------------------------------------------------------------------------

def scan_defensive_weaknesses(GAMES, fetcher=None):
    """Scan each game tonight for exploitable defensive weaknesses.

    For every team on the slate, identifies which stat categories that
    team's defense is worst at allowing (ranked against the league).

    Parameters
    ----------
    GAMES : dict
        Game research dict from game_researcher (keyed by 'AWAY@HOME').
    fetcher : NBAFetcher, optional
        Reuse an existing fetcher for its cached team_rankings.

    Returns
    -------
    list[dict]
        Sorted by severity (worst defenses first).  Each entry:
        {game, defending_team, defending_team_abr, attacking_team,
         attacking_team_abr, stat, stat_allowed, league_avg,
         delta_vs_avg, rank, severity, pace_factor}
    """
    if fetcher is None:
        fetcher = NBAFetcher()
    rankings_data = fetcher.get_team_rankings()
    if not rankings_data:
        return []

    teams = rankings_data.get('teams', {})
    league_avg = rankings_data.get('league_avg', {})

    # Try to get pace data from pbp_client for more accurate pace factors
    pbp = None
    try:
        if PBPStatsClient:
            pbp = PBPStatsClient()
    except Exception:
        pass

    weaknesses = []

    for game_key, gctx in GAMES.items():
        away_abr = gctx.get('away_abr', '')
        home_abr = gctx.get('home_abr', '')
        away_short = gctx.get('away', '')
        home_short = gctx.get('home', '')
        spread = gctx.get('spread', 0) or 0

        # Compute pace factor for this specific matchup
        pace_factor = 1.0
        pace_info = gctx.get('pace', {})
        projected_pace = pace_info.get('projected') if isinstance(pace_info, dict) else None
        if projected_pace and projected_pace > 0:
            # League avg pace is roughly 100; scale proportionally
            pace_factor = projected_pace / 100.0
        elif pbp:
            matchup_pace = pbp.get_team_pace_matchup(away_abr, home_abr)
            if matchup_pace:
                pace_factor = matchup_pace['predicted_pace'] / 100.0

        # Scan both teams as defenders -- each one's weakness is the
        # other team's opportunity.
        for def_abr, def_short, atk_abr, atk_short in [
            (home_abr, home_short, away_abr, away_short),
            (away_abr, away_short, home_abr, home_short),
        ]:
            def_data = teams.get(def_short) or teams.get(def_abr)
            if not def_data:
                continue

            for stat, def_key in STAT_TO_DEFENSE_KEY.items():
                allowed = def_data.get(def_key, 0)
                avg = league_avg.get(def_key, allowed)
                if avg == 0:
                    continue

                delta = allowed - avg  # positive = allows MORE than league avg
                rank_key = STAT_TO_RANK_KEY.get(stat)
                rank = def_data.get(rank_key, 15) if rank_key else None

                # For stats without a rank, compute one from the delta
                if rank is None:
                    # Bigger delta = worse defense for that stat
                    # Map delta into a pseudo-rank (30 = worst)
                    rank = 15 + int(delta * 3)  # rough heuristic
                    rank = max(1, min(30, rank))

                # Severity: how far above league average, scaled 0-1
                # A team allowing 4 more AST than avg is much worse than 4 more PTS
                stat_scale = {
                    'pts': 5.0, 'reb': 3.0, 'ast': 2.5,
                    '3pm': 1.5, 'stl': 1.0, 'blk': 1.0,
                }
                divisor = stat_scale.get(stat, 3.0)
                severity = delta / divisor  # normalized severity

                weaknesses.append({
                    'game': game_key,
                    'defending_team': def_short,
                    'defending_team_abr': def_abr,
                    'attacking_team': atk_short,
                    'attacking_team_abr': atk_abr,
                    'stat': stat,
                    'stat_allowed': round(allowed, 1),
                    'league_avg': round(avg, 1),
                    'delta_vs_avg': round(delta, 1),
                    'rank': rank,
                    'severity': round(severity, 3),
                    'pace_factor': round(pace_factor, 3),
                    'spread': spread,
                })

    # Sort by severity descending (worst defenses first)
    weaknesses.sort(key=lambda w: w['severity'], reverse=True)
    return weaknesses


# ---------------------------------------------------------------------------
# 2. FIND EXPLOITING PLAYERS
# ---------------------------------------------------------------------------

def find_exploiting_players(weakness, board_props, fetcher=None):
    """Given a defensive weakness, find board players who can exploit it.

    Scans the board for players on the attacking team whose prop stat
    matches the defensive weakness category.

    Parameters
    ----------
    weakness : dict
        One entry from scan_defensive_weaknesses().
    board_props : list[dict]
        The parsed board (player, team, game, stat, line, ...).
    fetcher : NBAFetcher, optional

    Returns
    -------
    list[dict]
        Ranked by matchup quality.  Each entry:
        {player, stat, line, gap, l10_hr, l10_avg, l5_avg, season_avg,
         mins_pct, matchup_grade, per_min_rate, projected_minutes,
         matchup_quality_score, game, direction}
    """
    if fetcher is None:
        fetcher = NBAFetcher()

    atk_abr = weakness['attacking_team_abr']
    atk_short = weakness['attacking_team']
    target_stat = weakness['stat']

    # Find board props for this attacking team AND this stat.
    # Also accept combo stats that include the target base stat.
    matching_props = []
    for prop in board_props:
        prop_team = (prop.get('team') or '').upper()
        prop_game = prop.get('game', '')
        prop_stat = (prop.get('stat') or '').lower()

        # Match team by abbreviation, short name, or game string
        team_match = (
            prop_team == atk_abr or
            prop_team == atk_short.upper() or
            (prop_game == weakness['game'] and _is_attacking_team(prop, weakness))
        )
        if not team_match:
            continue

        # Match stat: exact match on base stat, or combo that includes it
        stat_match = False
        if prop_stat == target_stat:
            stat_match = True
        elif prop_stat in COMBO_COMPONENTS:
            if target_stat in COMBO_COMPONENTS[prop_stat]:
                stat_match = True

        if stat_match:
            matching_props.append(prop)

    if not matching_props:
        return []

    # Score each matching player
    candidates = []
    for prop in matching_props:
        player = prop['player']
        stat = prop.get('stat', target_stat).lower()
        line = prop.get('line', 0)

        # Fetch player data from nba_fetcher (uses cache)
        data = fetcher.get_player_data(player, stat, line)
        if not data:
            continue

        l10_avg = data.get('l10_avg', 0)
        l5_avg = data.get('l5_avg', 0)
        season_avg = data.get('season_avg', 0)
        l10_hr = data.get('l10_hit_rate', 50)
        mins_pct = data.get('mins_30plus_pct', 50)
        season_mins = data.get('season_mins_avg', 0)
        l5_mins = data.get('l5_mins_avg', 0)

        # Gap: how much the average exceeds the line
        gap = l10_avg - line

        # Per-minute production rate
        per_min_rate = 0
        if season_mins > 0:
            per_min_rate = round(season_avg / season_mins, 4)

        # Projected minutes: blend of season and recent
        projected_minutes = season_mins * 0.75 + l5_mins * 0.25
        # Apply blowout discount
        blowout_disc = _blowout_minutes_discount(weakness.get('spread', 0))
        projected_minutes *= (1.0 - blowout_disc)

        # Matchup quality score: composite of multiple factors
        # Higher = better matchup for the player
        mq_score = 0.0

        # 1. Defensive weakness severity (0 to ~0.35)
        mq_score += min(weakness['severity'] * 0.35, 0.35)

        # 2. Gap over line (0 to 0.20)
        if l10_avg > 0 and line > 0:
            gap_pct = gap / line
            mq_score += min(max(gap_pct, 0) * 0.40, 0.20)

        # 3. L10 hit rate (0 to 0.20)
        mq_score += (l10_hr / 100) * 0.20

        # 4. Minutes stability (0 to 0.10)
        mq_score += (mins_pct / 100) * 0.10

        # 5. Recency trend: L5 > L10 = trending up (0 to 0.10)
        if l10_avg > 0 and l5_avg > l10_avg:
            trend = (l5_avg - l10_avg) / l10_avg
            mq_score += min(trend * 0.20, 0.10)

        # 6. Combo stat penalty (-0.10)
        if stat in COMBO_STATS:
            mq_score -= 0.10

        # Matchup grade
        if line > 0:
            margin_pct = (l10_avg - line) / line
        else:
            margin_pct = 0

        grade = 'D'
        for g, thresh in sorted(GRADE_THRESHOLDS.items(), key=lambda x: x[1], reverse=True):
            if margin_pct >= thresh:
                grade = g
                break

        candidates.append({
            'player': player,
            'stat': stat,
            'line': line,
            'gap': round(gap, 1),
            'l10_hr': l10_hr,
            'l10_avg': l10_avg,
            'l5_avg': l5_avg,
            'season_avg': season_avg,
            'mins_pct': mins_pct,
            'matchup_grade': grade,
            'per_min_rate': per_min_rate,
            'projected_minutes': round(projected_minutes, 1),
            'matchup_quality_score': round(mq_score, 4),
            'game': weakness['game'],
            'direction': 'OVER',  # exploiting a bad defense means OVER
            'weakness_stat': target_stat,
            'weakness_severity': weakness['severity'],
            'weakness_rank': weakness['rank'],
            'defending_team': weakness['defending_team'],
            'defending_team_abr': weakness['defending_team_abr'],
            'stat_allowed': weakness['stat_allowed'],
            'league_avg': weakness['league_avg'],
            'delta_vs_avg': weakness['delta_vs_avg'],
            'pace_factor': weakness['pace_factor'],
            'spread': weakness.get('spread', 0),
            'streak_status': data.get('streak_status', 'NEUTRAL'),
            'l10_miss_count': data.get('l10_miss_count', 0),
            'foul_trouble_risk': data.get('foul_trouble_risk', False),
            'efficiency_trend': data.get('efficiency_trend', 0),
        })

    # Sort by matchup quality score descending
    candidates.sort(key=lambda c: c['matchup_quality_score'], reverse=True)
    return candidates


def _is_attacking_team(prop, weakness):
    """Check whether the prop player is on the attacking side of this weakness."""
    game = prop.get('game', '')
    is_home = prop.get('is_home')
    if not game or '@' not in game:
        return False
    away, home = _game_teams(game)
    atk_abr = weakness['attacking_team_abr']
    if is_home is True and home == atk_abr:
        return True
    if is_home is False and away == atk_abr:
        return True
    # Fallback: team field match
    prop_team = (prop.get('team') or '').upper()
    return prop_team == atk_abr


# ---------------------------------------------------------------------------
# 3. CHECK LINE SOFTNESS
# ---------------------------------------------------------------------------

def check_line_softness(candidate, fetcher=None):
    """Is the sportsbook line mispriced for this matchup?

    Computes a matchup-adjusted projection and compares it to the line.
    A "soft" line is one where the defense weakness pushes the realistic
    projection well above what the book is offering.

    Parameters
    ----------
    candidate : dict
        One entry from find_exploiting_players().
    fetcher : NBAFetcher, optional

    Returns
    -------
    dict
        {softness_score (0-1), matchup_adjusted_projection,
         projected_margin, defense_factor, pace_factor,
         per_min_projection}
    """
    line = candidate.get('line', 0)
    if line <= 0:
        return {'softness_score': 0, 'matchup_adjusted_projection': 0,
                'projected_margin': 0, 'defense_factor': 1.0,
                'pace_factor': 1.0, 'per_min_projection': 0}

    stat = candidate.get('stat', 'pts').lower()
    l10_avg = candidate.get('l10_avg', 0)
    season_avg = candidate.get('season_avg', 0)
    per_min_rate = candidate.get('per_min_rate', 0)
    projected_minutes = candidate.get('projected_minutes', 0)
    pace_factor = candidate.get('pace_factor', 1.0)

    # Defense factor: how much MORE this defense allows vs league avg
    # severity > 0 means they allow more, translating to a boost for the player
    severity = candidate.get('weakness_severity', 0)
    # Cap the defense factor to avoid absurd projections
    defense_factor = 1.0 + min(max(severity * 0.10, -0.15), 0.20)

    # Base projection: blend of statistical average and per-minute decomposition
    stat_projection = l10_avg * 0.6 + season_avg * 0.4
    per_min_projection = 0
    if per_min_rate > 0 and projected_minutes > 0:
        per_min_projection = per_min_rate * projected_minutes

    # Use per-minute decomposition if available, otherwise fall back to averages
    if per_min_projection > 0:
        # Blend: 60% per-minute model, 40% pure stat average
        base_projection = per_min_projection * 0.6 + stat_projection * 0.4
    else:
        base_projection = stat_projection

    # Apply matchup adjustments
    matchup_adjusted = base_projection * defense_factor * pace_factor

    # Projected margin over the line
    margin = matchup_adjusted - line
    margin_pct = margin / line if line > 0 else 0

    # Softness score: 0 = line is fair/sharp, 1 = massively soft
    # Based on projected margin percentage
    if margin_pct <= 0:
        softness = 0.0
    elif margin_pct >= 0.25:
        softness = 1.0
    else:
        # Linear scale from 0 to 1 across 0-25% margin
        softness = margin_pct / 0.25

    # Bonus for per-minute projection confirming the same direction
    if per_min_projection > line and stat_projection > line:
        softness = min(softness + 0.10, 1.0)

    return {
        'softness_score': round(softness, 3),
        'matchup_adjusted_projection': round(matchup_adjusted, 1),
        'projected_margin': round(margin, 1),
        'margin_pct': round(margin_pct, 4),
        'defense_factor': round(defense_factor, 4),
        'pace_factor': round(pace_factor, 4),
        'per_min_projection': round(per_min_projection, 1),
        'base_projection': round(base_projection, 1),
    }


# ---------------------------------------------------------------------------
# 4. PER-MINUTE PROJECTION
# ---------------------------------------------------------------------------

def compute_per_minute_projection(player_prop, fetcher=None):
    """Decomposed projection: production_rate * projected_minutes * adjustments.

    This is more mechanistic than averaging past stat totals because it
    separates the rate from the opportunity (minutes).

    Parameters
    ----------
    player_prop : dict
        A result dict from the pipeline (with l10_avg, season_avg,
        season_mins_avg, l5_mins_avg, spread, etc.) OR a board prop
        that we'll look up fresh data for.
    fetcher : NBAFetcher, optional

    Returns
    -------
    dict or None
        {production_rate, projected_minutes, pace_factor, defense_factor,
         projected_stat, blowout_discount, components}
    """
    stat = (player_prop.get('stat') or 'pts').lower()
    line = player_prop.get('line', 0)

    # Try to pull from already-computed fields first
    l10_avg = player_prop.get('l10_avg')
    season_avg = player_prop.get('season_avg')
    season_mins = player_prop.get('season_mins_avg')
    l5_mins = player_prop.get('l5_mins_avg')

    # If missing key fields, fetch fresh
    if l10_avg is None or season_mins is None:
        if fetcher is None:
            fetcher = NBAFetcher()
        data = fetcher.get_player_data(player_prop['player'], stat, line)
        if not data:
            return None
        l10_avg = data.get('l10_avg', 0)
        season_avg = data.get('season_avg', 0)
        season_mins = data.get('season_mins_avg', 0)
        l5_mins = data.get('l5_mins_avg', 0)

    if not season_mins or season_mins < 5:
        return None

    # Production rate: stat per minute played
    production_rate = season_avg / season_mins

    # Projected minutes: weighted blend favoring season stability
    projected_minutes = season_mins * 0.75 + (l5_mins or season_mins) * 0.25

    # Blowout discount
    spread = abs(player_prop.get('spread', 0) or 0)
    blowout_disc = _blowout_minutes_discount(spread)
    projected_minutes *= (1.0 - blowout_disc)

    # Pace factor (from pipeline data or default 1.0)
    pace_factor = 1.0
    pace_impact = player_prop.get('pace_impact')
    if pace_impact is not None and pace_impact != 0:
        pace_factor = 1.0 + pace_impact

    # Defense factor (from opp_stat_allowed_vs_league_avg)
    defense_factor = 1.0
    opp_vs_avg = player_prop.get('opp_stat_allowed_vs_league_avg', 0)
    if opp_vs_avg != 0:
        # Scale: each point above league avg = ~1.5% boost
        stat_divisors = {'pts': 115, 'reb': 44, 'ast': 27, '3pm': 13, 'stl': 8, 'blk': 5}
        divisor = stat_divisors.get(stat, 50)
        defense_factor = 1.0 + (opp_vs_avg / divisor)

    projected_stat = production_rate * projected_minutes * pace_factor * defense_factor

    return {
        'production_rate': round(production_rate, 4),
        'projected_minutes': round(projected_minutes, 1),
        'pace_factor': round(pace_factor, 4),
        'defense_factor': round(defense_factor, 4),
        'projected_stat': round(projected_stat, 1),
        'blowout_discount': round(blowout_disc, 3),
        'components': {
            'base': round(production_rate * projected_minutes, 1),
            'with_pace': round(production_rate * projected_minutes * pace_factor, 1),
            'with_defense': round(projected_stat, 1),
        },
    }


# ---------------------------------------------------------------------------
# 5. SCAN TONIGHT -- THE MAIN FUNCTION
# ---------------------------------------------------------------------------

def scan_tonight(board_props, GAMES, max_picks=5, fetcher=None):
    """Defense-first matchup scanner.  THE core selection function.

    1. Scans all defensive weaknesses across tonight's games.
    2. For the top weaknesses, finds exploiting players on the board.
    3. Checks line softness for each candidate.
    4. Ranks by matchup_grade * softness * confidence.
    5. Applies diversity constraints (max 1 pick per game by default).
    6. Returns top 3-5 picks with full reasoning.

    Parameters
    ----------
    board_props : list[dict]
        Parsed board (player, team, game, stat, line).
    GAMES : dict
        Game research from game_researcher.
    max_picks : int
        Maximum number of picks to return (default 5).
    fetcher : NBAFetcher, optional

    Returns
    -------
    list[dict]
        Top picks sorted by final_score descending.
    """
    if fetcher is None:
        fetcher = NBAFetcher()

    if not GAMES or not board_props:
        return []

    # Step 1: Scan defensive weaknesses
    weaknesses = scan_defensive_weaknesses(GAMES, fetcher=fetcher)
    if not weaknesses:
        return []

    # Step 2: For top weaknesses, find exploiting players
    # Take the top N weaknesses (enough to generate a rich candidate pool)
    top_weaknesses = weaknesses[:min(len(weaknesses), 20)]

    all_candidates = []
    seen_player_stat = set()  # Deduplicate: same player+stat from different weaknesses

    for weakness in top_weaknesses:
        exploiters = find_exploiting_players(weakness, board_props, fetcher=fetcher)
        for cand in exploiters:
            key = (cand['player'], cand['stat'])
            if key in seen_player_stat:
                continue
            seen_player_stat.add(key)
            all_candidates.append(cand)

    if not all_candidates:
        return []

    # Step 3: Check line softness for each candidate
    for cand in all_candidates:
        softness = check_line_softness(cand, fetcher=fetcher)
        cand.update(softness)

        # Also compute per-minute projection
        per_min = compute_per_minute_projection(cand, fetcher=fetcher)
        if per_min:
            cand['per_min_projection_detail'] = per_min
            cand['per_minute_projected'] = per_min['projected_stat']
        else:
            cand['per_minute_projected'] = cand.get('matchup_adjusted_projection', 0)

    # Step 4: Compute final composite score
    for cand in all_candidates:
        mq = cand.get('matchup_quality_score', 0)
        soft = cand.get('softness_score', 0)
        l10_hr = cand.get('l10_hr', 50) / 100.0

        # Confidence: combination of hit rate, minutes stability, not foul-prone
        confidence = l10_hr * 0.50
        confidence += (cand.get('mins_pct', 50) / 100.0) * 0.25
        if not cand.get('foul_trouble_risk', False):
            confidence += 0.10
        if cand.get('streak_status') != 'COLD':
            confidence += 0.05
        # Penalty for high miss count
        if cand.get('l10_miss_count', 0) >= 4:
            confidence -= 0.10
        # Penalty for combo stats (more volatile)
        if cand.get('stat', '') in COMBO_STATS:
            confidence -= 0.08
        confidence = max(0.0, min(1.0, confidence))

        # Final score: weighted product
        final = (mq * 0.40) + (soft * 0.35) + (confidence * 0.25)

        # Bonus: per-minute projection confirms the direction
        per_min_proj = cand.get('per_minute_projected', 0)
        line = cand.get('line', 0)
        if per_min_proj > line * 1.05:
            final += 0.05
        elif per_min_proj < line * 0.90:
            final -= 0.08  # per-minute model disagrees -- red flag

        cand['confidence'] = round(confidence, 3)
        cand['final_score'] = round(final, 4)

    # Sort by final score
    all_candidates.sort(key=lambda c: c['final_score'], reverse=True)

    # Step 5: Apply diversity constraints
    selected = []
    used_games = set()
    used_players = set()

    for cand in all_candidates:
        if len(selected) >= max_picks:
            break

        game = cand['game']
        player = cand['player']

        # Max 2 picks per game (allows exploiting both sides of a bad-defense game)
        game_count = sum(1 for s in selected if s['game'] == game)
        if game_count >= 2:
            continue

        # No duplicate players
        if player in used_players:
            continue

        # Skip if line softness is too low (line is sharp, not exploitable)
        if cand.get('softness_score', 0) < 0.05:
            continue

        # Skip if L10 hit rate is below 50% (too unreliable)
        if cand.get('l10_hr', 0) < 50:
            continue

        used_players.add(player)
        used_games.add(game)

        # Step 6: Build reasoning string
        reasoning = _build_reasoning(cand)
        cand['reasoning'] = reasoning

        # Map matchup_grade to descriptive text
        grade = cand.get('matchup_grade', 'D')
        cand['matchup_exploit'] = (
            f"{cand['defending_team']} allows "
            f"{_rank_suffix(cand['weakness_rank'])} most {cand['weakness_stat'].upper()} "
            f"in the league ({cand.get('stat_allowed', 0)}/g vs {cand.get('league_avg', 0)} avg)"
        )
        cand['defense_weakness_rank'] = cand['weakness_rank']

        selected.append(cand)

    return selected


def _rank_suffix(rank):
    """Convert rank 28 to '3rd' (from bottom), rank 25 to '6th', etc."""
    from_bottom = 31 - rank
    if from_bottom <= 0:
        from_bottom = 1
    suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
    suffix = suffixes.get(from_bottom if from_bottom < 20 else from_bottom % 10, 'th')
    return f"{from_bottom}{suffix}"


def _build_reasoning(cand):
    """Build a human-readable reasoning string for a pick."""
    parts = []

    # Defense weakness
    parts.append(
        f"{cand['defending_team']} bottom-{31 - cand['weakness_rank']} "
        f"{cand['weakness_stat'].upper()} defense "
        f"(allows {cand.get('stat_allowed', 0)}/g, league avg {cand.get('league_avg', 0)})"
    )

    # Per-minute projection
    per_min = cand.get('per_min_rate', 0)
    proj_mins = cand.get('projected_minutes', 0)
    proj = cand.get('matchup_adjusted_projection', 0)
    line = cand.get('line', 0)
    if per_min > 0 and proj_mins > 0:
        parts.append(
            f"{cand['player']} {per_min:.2f} {cand['stat']}/min * "
            f"{proj_mins:.0f} projected min = {proj:.1f} projected vs {line:.1f} line"
        )
    else:
        parts.append(f"Projected {proj:.1f} vs {line:.1f} line")

    # Hit rate context
    parts.append(f"L10 HR {cand['l10_hr']}% (L10 avg {cand['l10_avg']}, L5 avg {cand['l5_avg']})")

    # Pace context
    pf = cand.get('pace_factor', 1.0)
    if pf > 1.02:
        parts.append(f"Fast-paced game (pace factor {pf:.2f})")
    elif pf < 0.98:
        parts.append(f"Slow-paced game (pace factor {pf:.2f})")

    # Blowout risk
    spread = abs(cand.get('spread', 0) or 0)
    if spread >= 10:
        disc = _blowout_minutes_discount(spread)
        parts.append(f"Blowout risk: {spread:.1f}pt spread, {disc*100:.0f}% minutes discount applied")

    # Softness
    soft = cand.get('softness_score', 0)
    if soft >= 0.50:
        parts.append(f"Line appears SOFT (softness {soft:.2f})")
    elif soft >= 0.20:
        parts.append(f"Line slightly soft (softness {soft:.2f})")

    return ' | '.join(parts)


# ---------------------------------------------------------------------------
# 6. PIPELINE INTEGRATION
# ---------------------------------------------------------------------------

def enrich_with_matchup_scan(results, GAMES, fetcher=None):
    """Run the scanner and tag each result in the full board.

    Adds to each result dict:
        matchup_scan_rank : int or None (1-5 if selected, None otherwise)
        is_matchup_exploit : bool
        matchup_scan_score : float (final_score from the scanner, 0 if not selected)

    Parameters
    ----------
    results : list[dict]
        The full board results from the pipeline.
    GAMES : dict
        Game research dict.
    fetcher : NBAFetcher, optional

    Returns
    -------
    int
        Number of results enriched (i.e., flagged by the scanner).
    """
    if not results or not GAMES:
        return 0

    # Build a board-like list from results for the scanner
    board_props = []
    for r in results:
        if 'error' in r:
            continue
        board_props.append({
            'player': r.get('player', ''),
            'team': r.get('team', ''),
            'game': r.get('game', ''),
            'stat': r.get('stat', ''),
            'line': r.get('line', 0),
            'is_home': r.get('is_home'),
            # Pass through existing data so the scanner can skip re-fetching
            'l10_avg': r.get('l10_avg'),
            'l5_avg': r.get('l5_avg'),
            'season_avg': r.get('season_avg'),
            'season_mins_avg': r.get('season_mins_avg'),
            'l5_mins_avg': r.get('l5_mins_avg'),
            'l10_hit_rate': r.get('l10_hit_rate'),
            'mins_30plus_pct': r.get('mins_30plus_pct'),
            'streak_status': r.get('streak_status'),
            'l10_miss_count': r.get('l10_miss_count'),
            'foul_trouble_risk': r.get('foul_trouble_risk'),
            'efficiency_trend': r.get('efficiency_trend'),
            'spread': r.get('spread'),
            'pace_impact': r.get('pace_impact'),
            'opp_stat_allowed_vs_league_avg': r.get('opp_stat_allowed_vs_league_avg'),
        })

    # Run the scanner
    top_picks = scan_tonight(board_props, GAMES, max_picks=5, fetcher=fetcher)

    # Build lookup: (player, stat) -> scanner result
    scan_lookup = {}
    for rank, pick in enumerate(top_picks, 1):
        key = (pick['player'], pick['stat'].lower())
        scan_lookup[key] = {
            'rank': rank,
            'final_score': pick.get('final_score', 0),
            'matchup_exploit': pick.get('matchup_exploit', ''),
            'softness_score': pick.get('softness_score', 0),
            'matchup_adjusted_projection': pick.get('matchup_adjusted_projection', 0),
            'reasoning': pick.get('reasoning', ''),
        }

    # Tag every result
    enriched = 0
    for r in results:
        key = (r.get('player', ''), (r.get('stat') or '').lower())
        scan = scan_lookup.get(key)
        if scan:
            r['matchup_scan_rank'] = scan['rank']
            r['is_matchup_exploit'] = True
            r['matchup_scan_score'] = scan['final_score']
            r['matchup_scan_exploit'] = scan['matchup_exploit']
            r['matchup_scan_softness'] = scan['softness_score']
            r['matchup_scan_projection'] = scan['matchup_adjusted_projection']
            r['matchup_scan_reasoning'] = scan['reasoning']
            enriched += 1
        else:
            r['matchup_scan_rank'] = None
            r['is_matchup_exploit'] = False
            r['matchup_scan_score'] = 0

    return enriched


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def _run_test():
    """Demo with sample data showing the scanner's logic."""
    print('=' * 70)
    print('  MATCHUP SCANNER v1 -- Defense-First Prop Selection (TEST MODE)')
    print('=' * 70)

    # Build a minimal GAMES dict for testing
    sample_games = {
        'MIN@OKC': {
            'away': 'Timberwolves', 'home': 'Thunder',
            'away_abr': 'MIN', 'home_abr': 'OKC',
            'spread': -8.5,
            'over_under': 225,
            'away_out': [], 'home_out': [],
            'away_questionable': [], 'home_questionable': [],
            'away_b2b': False, 'home_b2b': False,
            'notes': 'Test game',
            'pace': {'projected': 100.5, 'label': 'average'},
            'away_defense': {'rating': 110, 'rank': 8, 'label': 'good'},
            'home_defense': {'rating': 105, 'rank': 3, 'label': 'elite'},
        },
        'GSW@DET': {
            'away': 'Warriors', 'home': 'Pistons',
            'away_abr': 'GSW', 'home_abr': 'DET',
            'spread': 4.5,
            'over_under': 217.5,
            'away_out': [], 'home_out': [],
            'away_questionable': [], 'home_questionable': [],
            'away_b2b': False, 'home_b2b': False,
            'notes': 'Test game 2',
            'pace': {'projected': 99.0, 'label': 'average'},
            'away_defense': {'rating': 113, 'rank': 18, 'label': 'poor'},
            'home_defense': {'rating': 115, 'rank': 25, 'label': 'terrible'},
        },
    }

    sample_board = [
        {'player': 'Shai Gilgeous-Alexander', 'team': 'OKC', 'game': 'MIN@OKC',
         'stat': 'pts', 'line': 32.5},
        {'player': 'Shai Gilgeous-Alexander', 'team': 'OKC', 'game': 'MIN@OKC',
         'stat': 'ast', 'line': 6.5},
        {'player': 'Anthony Edwards', 'team': 'MIN', 'game': 'MIN@OKC',
         'stat': 'pts', 'line': 25.5},
        {'player': 'Stephen Curry', 'team': 'GSW', 'game': 'GSW@DET',
         'stat': 'pts', 'line': 23.5},
        {'player': 'Stephen Curry', 'team': 'GSW', 'game': 'GSW@DET',
         'stat': '3pm', 'line': 4.5},
        {'player': 'Jalen Duren', 'team': 'DET', 'game': 'GSW@DET',
         'stat': 'reb', 'line': 9.5},
        {'player': 'Cade Cunningham', 'team': 'DET', 'game': 'GSW@DET',
         'stat': 'ast', 'line': 8.5},
    ]

    print(f'\n  Sample board: {len(sample_board)} props across {len(sample_games)} games')

    # Step 1: Defensive weaknesses
    print('\n' + '-' * 70)
    print('  STEP 1: Scanning defensive weaknesses...')
    print('-' * 70)
    fetcher = NBAFetcher()
    weaknesses = scan_defensive_weaknesses(sample_games, fetcher=fetcher)
    print(f'  Found {len(weaknesses)} weakness entries')
    for w in weaknesses[:10]:
        print(f'    {w["defending_team"]:15s} allows {w["stat"].upper():4s}: '
              f'{w["stat_allowed"]:5.1f}/g (avg {w["league_avg"]:5.1f}, '
              f'rank {w["rank"]}, severity {w["severity"]:+.3f})')

    # Step 2: Full scan
    print('\n' + '-' * 70)
    print('  STEP 2: Running full scan...')
    print('-' * 70)
    picks = scan_tonight(sample_board, sample_games, max_picks=5, fetcher=fetcher)

    if not picks:
        print('  No picks found (player data may not be cached).')
        print('  In production, player logs are pre-fetched by run_board_v5.py.')
        return

    print(f'\n  TOP {len(picks)} MATCHUP EXPLOITS:')
    print('  ' + '=' * 68)
    for i, pick in enumerate(picks, 1):
        print(f'\n  #{i}  {pick["player"]} {pick["stat"].upper()} {pick["direction"]} {pick["line"]}')
        print(f'      Game: {pick["game"]}')
        print(f'      Matchup: {pick.get("matchup_exploit", "N/A")}')
        print(f'      Matchup grade: {pick["matchup_grade"]} | '
              f'Line softness: {pick.get("softness_score", 0):.2f} | '
              f'Confidence: {pick.get("confidence", 0):.2f}')
        print(f'      Projection: {pick.get("matchup_adjusted_projection", 0):.1f} '
              f'(per-min: {pick.get("per_minute_projected", 0):.1f})')
        print(f'      L10 avg: {pick["l10_avg"]} | L10 HR: {pick["l10_hr"]}% | '
              f'Gap: {pick["gap"]:+.1f}')
        print(f'      Final score: {pick["final_score"]:.4f}')
        print(f'      Reasoning: {pick.get("reasoning", "")}')

    print('\n  ' + '=' * 68)
    print(f'  Scanner complete: {len(picks)} high-conviction picks')


def _run_live(board_path, games_path):
    """Run the scanner against a real board and research file."""
    print('=' * 70)
    print('  MATCHUP SCANNER v1 -- Defense-First Prop Selection')
    print('=' * 70)

    # Load board
    with open(board_path) as f:
        board_props = json.load(f)
    print(f'\n  Board: {len(board_props)} props from {board_path}')

    # Load GAMES
    with open(games_path) as f:
        GAMES = json.load(f)
    print(f'  Games: {len(GAMES)} games from {games_path}')

    fetcher = NBAFetcher()

    # Scan
    picks = scan_tonight(board_props, GAMES, max_picks=5, fetcher=fetcher)

    if not picks:
        print('\n  No matchup exploits found on this board.')
        return

    print(f'\n  TOP {len(picks)} MATCHUP EXPLOITS:')
    print('  ' + '=' * 68)
    for i, pick in enumerate(picks, 1):
        print(f'\n  #{i}  {pick["player"]} {pick["stat"].upper()} {pick["direction"]} {pick["line"]}')
        print(f'      Game: {pick["game"]}')
        print(f'      Matchup: {pick.get("matchup_exploit", "N/A")}')
        print(f'      Matchup grade: {pick["matchup_grade"]} | '
              f'Line softness: {pick.get("softness_score", 0):.2f} | '
              f'Confidence: {pick.get("confidence", 0):.2f}')
        print(f'      Projection: {pick.get("matchup_adjusted_projection", 0):.1f} '
              f'(per-min: {pick.get("per_minute_projected", 0):.1f})')
        print(f'      L10 avg: {pick["l10_avg"]} | L10 HR: {pick["l10_hr"]}% | '
              f'Gap: {pick["gap"]:+.1f}')
        print(f'      Final score: {pick["final_score"]:.4f}')
        print(f'      Reasoning: {pick.get("reasoning", "")}')

    print('\n  ' + '=' * 68)
    print(f'  Scanner complete: {len(picks)} high-conviction picks')

    # Also output as JSON
    out_dir = os.path.dirname(board_path)
    out_path = os.path.join(out_dir, 'matchup_scan.json')
    with open(out_path, 'w') as f:
        # Clean for serialization
        clean_picks = []
        for p in picks:
            clean = {k: v for k, v in p.items() if k != 'per_min_projection_detail'}
            clean_picks.append(clean)
        json.dump(clean_picks, f, indent=2, default=str)
    print(f'  Saved to {out_path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Matchup Scanner -- Defense-First Prop Selection')
    parser.add_argument('--board', type=str, help='Path to board JSON')
    parser.add_argument('--games', type=str, help='Path to game research JSON')
    parser.add_argument('--test', action='store_true', help='Run test with sample data')
    parser.add_argument('--max-picks', type=int, default=5, help='Max picks (default 5)')
    args = parser.parse_args()

    if args.test:
        _run_test()
    elif args.board and args.games:
        _run_live(args.board, args.games)
    else:
        parser.print_help()
        print('\nExamples:')
        print('  python3 predictions/matchup_scanner.py --test')
        print('  python3 predictions/matchup_scanner.py --board predictions/2026-03-20/board.json '
              '--games predictions/2026-03-20/2026-03-20_game_research.json')


if __name__ == '__main__':
    main()

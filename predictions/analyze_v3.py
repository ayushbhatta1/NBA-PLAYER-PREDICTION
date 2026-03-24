#!/usr/bin/env python3
"""
NBA Prop Analysis Pipeline v3 - LIVE DATA
Now powered by nba_api instead of static CSV files.

Changes from v2:
- Player game logs pulled LIVE from stats.nba.com via nba_fetcher
- Team defensive rankings pulled LIVE (real DEF_RATING, PACE, opponent stats)
- Opponent-specific history added (how player performs vs this specific team)
- Injury adjustment bug FIXED: pts bump only applied to pts stat, not combos
- Player lookup uses nba_api's full player database instead of CSV fuzzy match
- Caching layer: game logs cached 4hrs, team rankings cached 12hrs
- Combo stat injury adjustments now properly weighted

Pipeline layers (v14: UNDER-dominant direction model):
1. Core Projection — market-anchored (dynamic blend: stats + sportsbook line)
2. Home/Away Split ← LIVE splits from game log
3. Opponent Defense ← LIVE team rankings from stats.nba.com
4. Pace Context ← LIVE pace data
5. Minutes Filter ← LIVE minutes data
6. Player Injury Status (manual input)
7. Teammate Injury Impact (from injury_impacts.json)
8. Opponent Injury (web search)
9. Hot/Cold Streak ← LIVE L3 vs L10
10. B2B Flag (manual input)
11. Blowout Risk (manual spread input)
12. Hit Rate Direction Calibration (L10/season HR adjusts projection)
13. Empirical UNDER Calibration (v14: 5% systematic + stat-specific corrections)
14. UNDER-Dominant Direction (v14: default UNDER, OVER needs gap>3 AND HR>=65)
"""

import json
import os
from datetime import datetime
from nba_fetcher import NBAFetcher

# ── CONFIG ──
# v14: Gap-based tiers (legacy — still used for OVER picks and gap reporting)
TIERS = [
    ("S",  4.0, 999),   # Elite edge — parlay core lock
    ("A",  3.0, 4.0),   # Strong edge — parlay core
    ("B",  2.0, 3.0),   # Solid edge — parlay flex
    ("C",  1.5, 2.0),   # Moderate edge — singles worthy
    ("D",  1.0, 1.5),   # Slight edge — risky
    ("F",  0.0, 1.0),   # Thin edge — tracked with reasoning for learning
]

# v14: UNDER confidence tiers (primary tier system for UNDER picks)
# Backtested: S=73.6%, A=63.7%, B=64.9%, C=61.4%, D=56.4%, F=54.3%
UNDER_CONFIDENCE_TIERS = [
    ("S",  5.0, 999),   # Elite UNDER — 73.6% backtested hit rate
    ("A",  3.5, 5.0),   # Strong UNDER — 63.7%
    ("B",  2.0, 3.5),   # Good UNDER — 64.9%
    ("C",  0.5, 2.0),   # Moderate UNDER — 61.4%
    ("D", -1.0, 0.5),   # Weak UNDER — 56.4%
    ("F", -999, -1.0),  # Very weak — 54.3%
]

# v14: Stat-type UNDER bonuses for confidence scoring
STAT_UNDER_BONUS = {
    'blk': 2.0, 'stl': 1.5, 'stl_blk': 2.5,
    '3pm': 1.0, 'pa': 0.8, 'ast': 0.5, 'ra': 0.3,
}


def under_confidence_score(player_data, stat, gap, streak_status='', is_b2b=False,
                           spread=None, is_home=None):
    """
    Composite UNDER confidence score. Higher = more likely to go UNDER.

    Backtested on 1,090,827 props (10 years, 2016-2026):
      conf >= 5 (S-tier): 64.5-67% UNDER hit rate
      conf >= 7: 70.9%
      conf >= 8: 73.6%
      conf >= 9: 77.8%

    10-year parlay simulation: 698/1639 = 42.6% 3-leg cash rate, 75.5% leg HR.
    With away-game bonus: 718/1639 = 43.8% cash rate (+1.2pp).

    Used for tier assignment on UNDER picks and parlay leg ranking.
    """
    score = 0.0

    # 1. L10 Hit Rate (inverted — low HR = rarely goes OVER = strong UNDER)
    hr = player_data.get('l10_hit_rate', 50)
    if hr < 20: score += 3.0
    elif hr < 35: score += 2.0
    elif hr < 45: score += 1.0
    elif hr < 55: score += 0
    elif hr < 65: score -= 0.5
    elif hr < 80: score -= 1.0
    else: score -= 2.0

    # 2. Season Hit Rate
    shr = player_data.get('season_hit_rate', 50)
    if shr < 30: score += 2.0
    elif shr < 45: score += 0.5
    elif shr < 55: score += 0
    elif shr < 70: score -= 0.5
    else: score -= 1.0

    # 3. Stat type bonus (backtested: BLK 71.1%, STL 68.9%, 3PM 65.3%, AST 64.7%)
    score += STAT_UNDER_BONUS.get(stat, 0)

    # 4. Streak
    if streak_status == 'COLD': score += 1.0
    elif streak_status == 'HOT': score -= 0.5

    # 5. Gap (negative gap = projection below line = stronger UNDER signal)
    if gap < -5: score += 2.0
    elif gap < -3: score += 1.5
    elif gap < -1.5: score += 1.0
    elif gap < 0: score += 0.5
    elif gap < 1.5: score += 0
    elif gap < 3: score -= 0.5
    else: score -= 0.5

    # 6. L10 Miss Count (games player went under the line)
    mc = player_data.get('l10_miss_count', 5)
    if mc >= 9: score += 2.0
    elif mc >= 7: score += 1.0
    elif mc >= 5: score += 0.3
    elif mc < 3: score -= 0.5

    # 7. B2B penalty (backtested: B2B hurts UNDER by ~7pp on 10-day data)
    if is_b2b: score -= 1.0

    # 8. Spread (big favorites → starters sit → UNDER)
    if spread is not None:
        if spread < -10: score += 0.5
        elif spread < -5: score += 0.3

    # 9. Away game bonus (backtested 10yr: away 68.0% vs home 67.1% for S-tier UNDERs)
    # Adds +1.2pp to parlay cash rate (43.8% vs 42.6%)
    if is_home is not None and not is_home:
        score += 0.5

    return round(score, 1)

# Defensive adjustment scaling factor (higher = more aggressive adjustment)
# v3: was /150 (~10% max). v4: /75 (~20% max) based on Mar 11 calibration.
DEF_ADJUSTMENT_DIVISOR = 75

# UNDER penalty: REMOVED in v5
# v3: B/C/D → F. v4: C/D → F. v5: NONE — UNDERs keep their earned tier.
# Mar 11: UNDERs 77.6%. Mar 12: UNDERs 70.7% vs OVERs 42.0%. Penalty was counterproductive.
UNDER_PENALTY_TIERS = set()  # empty = no UNDER downgrade

# Blowout risk adjustments (v4: strengthened from -2%/-3%)
BLOWOUT_FAVORITE_ADJ = -0.05   # big favorite: Q4 minutes risk
BLOWOUT_UNDERDOG_ADJ = -0.03   # big underdog: garbage time stats
BLOWOUT_SPREAD_THRESHOLD = 10  # minimum abs(spread) to trigger

# Streak detection threshold (L3 vs L10 % change)
STREAK_HOT_THRESHOLD = 15
STREAK_COLD_THRESHOLD = -15

# Streak adjustment magnitude
STREAK_ADJ_FACTOR = 0.05

# Combo stat volatility penalty: these stats need higher gap to qualify for tier
COMBO_STATS = {'pra', 'pr', 'pa', 'ra', 'stl_blk'}
COMBO_GAP_PENALTY = 0.5  # subtract from abs_gap before tier grading for combos

# ── v14: UNDER-DOMINANT DIRECTION MODEL ──
# Backtested Mar 13-21 (4,982 lines): Actual OVER rate = 39.2%. UNDER rate = 60.5%.
# No OVER filter beats "always UNDER" (60.7%). Even gap>5 + HR>=70 OVERs hit only 31-43%.
# ParlayPlay lines are systematically set high — UNDER is the structural edge.
# Key stat splits: BLK UNDER 74.8%, STL 68%, 3PM 63.3%, AST 61%, REB 59.7%
MARKET_LINE_WEIGHT = 0.35   # Default blend; dynamically adjusted by hit rate in analyze_player()
OVER_BIAS_CORRECTION = 0.05  # 5% systematic downward shift (was 2.5% — doubled based on backtesting)
THIN_GAP_UNDER_THRESHOLD = 3.0  # Gaps 0-3.0 flip to UNDER (was 1.0 — data shows gap 1.5-3.0 OVERs hit 42%)
OVER_CONFIRMATION_HR = 65  # L10 HR must be >= this to allow OVER call (even above threshold)

# v14: Stat-specific UNDER bias — some stats go UNDER far more than others
# From backtesting: BLK actual OVER rate 25.2%, STL 32%, 3PM 36.7%
STAT_UNDER_EXTRA_CORRECTION = {
    'blk': 0.06,      # BLK goes UNDER 74.8% of the time
    'stl': 0.04,      # STL goes UNDER 68%
    'stl_blk': 0.06,  # STL+BLK combo goes UNDER 86.7%
    '3pm': 0.03,      # 3PM goes UNDER 63.3%
    'pa': 0.02,       # PA goes UNDER 63.5%
    'ast': 0.015,     # AST goes UNDER 61%
}

# Injury status severity weights (for tier downgrade)
INJURY_SEVERITY = {
    'out': 'SKIP',       # don't even analyze
    'dnp': 'SKIP',
    'doubtful': 2,       # downgrade 2 tiers
    'questionable': 1,   # downgrade 1 tier
    'gtd': 1,            # downgrade 1 tier
    'game-time decision': 1,
    'probable': 0,       # no downgrade
    'available': 0,
}

# Which defensive ranking to use for each stat
DEF_RANK_MAP = {
    'pts': 'avg_pts_allowed_rank', '3pm': 'tpm_allowed_rank',
    'reb': 'reb_allowed_rank', 'ast': 'ast_allowed_rank',
    'pra': 'avg_pts_allowed_rank', 'pr': 'avg_pts_allowed_rank',
    'pa': 'avg_pts_allowed_rank', 'ra': 'reb_allowed_rank',
    'stl': 'avg_pts_allowed_rank', 'blk': 'avg_pts_allowed_rank',
}

# Rate-based defense: maps stat → allowed value key in team rankings
DEF_ALLOWED_MAP = {
    'pts': 'avg_pts_allowed', '3pm': 'tpm_allowed',
    'reb': 'reb_allowed', 'ast': 'ast_allowed',
    'stl': 'stl_allowed', 'blk': 'blk_allowed',
}

# Global fetcher instance (reused across all analyze calls)
_fetcher = None
_corrections = None

def get_fetcher(season='2025-26'):
    global _fetcher
    if _fetcher is None:
        _fetcher = NBAFetcher(season=season)
    return _fetcher


def _load_corrections():
    """Load active corrections from self_heal output."""
    global _corrections
    if _corrections is not None:
        return _corrections

    corrections_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'corrections.json')
    if not os.path.exists(corrections_path):
        _corrections = []
        return _corrections

    try:
        with open(corrections_path) as f:
            _corrections = [c for c in json.load(f) if c.get('status') == 'ACTIVE']
    except Exception:
        _corrections = []
    return _corrections


def _apply_corrections(stat, direction, projection, corrections):
    """Apply active corrections to a projection. Returns (adj_proj, notes)."""
    adj = 0
    notes = []
    for c in corrections:
        ctype = c.get('type', c.get('id', ''))
        details = c.get('details', {})

        if 'STAT_BIAS' in ctype:
            # Stat-level bias: if a stat is systematically over/under-projected
            bias_stat = details.get('stat', '')
            if bias_stat == stat:
                bias_dir = details.get('bias_direction', '')  # 'over' or 'under' projected
                severity = details.get('severity', 'LOW')
                factor = {'HIGH': 0.05, 'MEDIUM': 0.03, 'LOW': 0.01}.get(severity, 0.01)
                if bias_dir == 'over':
                    adj -= projection * factor
                    notes.append(f"CORRECTION: {stat} over-projected by {factor*100:.0f}%")
                elif bias_dir == 'under':
                    adj += projection * factor
                    notes.append(f"CORRECTION: {stat} under-projected by {factor*100:.0f}%")

        elif 'DIRECTION_BIAS' in ctype:
            # Direction bias: OVER/UNDER systemically performs differently
            bias_direction = details.get('direction', '')
            if bias_direction == direction:
                accuracy = details.get('accuracy', 50)
                # If UNDERs hit 77%, give slight projection boost toward UNDER
                if accuracy > 65 and direction == 'UNDER':
                    adj -= projection * 0.01  # Lower projection slightly → bigger UNDER gap
                    notes.append(f"CORRECTION: {direction} outperforming ({accuracy}%)")

        elif 'MATCHUP_BIAS' in ctype:
            # Matchup bias: specific opponent causing systematic error
            pass  # Would need opponent context

    return adj, notes


def analyze_player(player_name, stat, line, opponent=None, is_home=None,
                   injury_data=None, injured_out=None, player_injury_status=None,
                   is_b2b=False, spread=None, team_rankings=None, game=None,
                   same_team_out_count=0):
    """
    Full v3 analysis for a single player prop using LIVE data.

    Args:
        player_name: Full player name (e.g., "Amen Thompson")
        stat: Stat category (pts, reb, ast, 3pm, pra, pr, pa, ra, stl, blk)
        line: Sportsbook line (e.g., 14.5)
        opponent: Opponent team short name (e.g., "Nuggets")
        is_home: True if player's team is home, False if away, None if unknown
        injury_data: Dict from injury_impacts.json
        injured_out: List of player names confirmed OUT
        player_injury_status: "OUT", "Questionable", "GTD", "Probable", None
        is_b2b: True if back-to-back game
        spread: Vegas spread for blowout risk calc
        team_rankings: Pre-fetched team rankings dict (optional, will fetch if None)
        game: Game label string (e.g., "HOU@DEN")
    """
    fetcher = get_fetcher()

    # ── PLAYER INJURY CHECK ──
    if player_injury_status and player_injury_status.lower() in ['out', 'dnp']:
        return {
            "player": player_name, "stat": stat, "line": line,
            "error": f"Player is {player_injury_status} — SKIP",
            "tier": "SKIP", "direction": "SKIP", "game": game
        }

    # ── FETCH LIVE DATA ──
    player_data = fetcher.get_player_data(player_name, stat, line, opponent, is_home)
    if player_data is None:
        return {
            "player": player_name, "stat": stat, "line": line,
            "error": "Player not found or < 5 games",
            "tier": "SKIP", "direction": "SKIP", "game": game
        }

    # ── CORE PROJECTION (Layer 1) ──
    # v13: Market-anchored projection — blend statistical avg with sportsbook line
    # The line IS smart money's estimate; pure averages overshoot (mean > median)
    # Dynamic blend: trust stats more when hit rate is decisive, trust market when it's unclear
    raw_proj = 0.4 * player_data['season_avg'] + 0.35 * player_data['l10_avg'] + 0.25 * player_data['l5_avg']
    _pre_hr = 0.6 * player_data['l10_hit_rate'] + 0.4 * player_data.get('season_hit_rate', player_data['l10_hit_rate'])
    if _pre_hr >= 70 or _pre_hr <= 30:
        _mkt_wt = 0.15  # Strong HR signal → trust stats more
    elif _pre_hr >= 60 or _pre_hr <= 40:
        _mkt_wt = 0.25  # Moderate HR signal
    else:
        _mkt_wt = 0.40  # Coin flip zone → lean on market line
    base_proj = (1 - _mkt_wt) * raw_proj + _mkt_wt * line

    # ── HOME/AWAY SPLIT (Layer 2) ──
    split_adj = 0
    if is_home is not None:
        relevant_avg = player_data['home_avg'] if is_home else player_data['away_avg']
        split_adj = (relevant_avg - player_data['season_avg']) * 0.15

    adjusted_proj = base_proj + split_adj

    # ── OPPONENT DEFENSE (Layer 3) - LIVE DATA ──
    matchup_note = ""
    matchup_adj = 0
    matchup_factor = 0
    opp_data = None
    if opponent:
        if team_rankings is None:
            team_rankings = fetcher.get_team_rankings()
        teams = team_rankings.get('teams', team_rankings)
        if isinstance(teams, dict) and 'teams' in teams:
            teams = teams['teams']

        opp_data = teams.get(opponent)
        if opp_data:
            def_rank_key = DEF_RANK_MAP.get(stat, 'avg_pts_allowed_rank')
            # Fallback to old key names for backward compat
            def_rank = opp_data.get(def_rank_key, opp_data.get('def_rank', 15))

            # v5 UPGRADE: Rate-based defense (proportional to actual allowed values)
            allowed_key = DEF_ALLOWED_MAP.get(stat)
            league_avg_data = teams.get('league_avg') if isinstance(team_rankings, dict) else None
            if not league_avg_data:
                league_avg_data = team_rankings.get('league_avg') if isinstance(team_rankings, dict) else None

            if allowed_key and league_avg_data and allowed_key in league_avg_data:
                opp_allowed = opp_data.get(allowed_key, 0)
                lg_avg = league_avg_data.get(allowed_key, 0)
                if lg_avg > 0 and opp_allowed > 0:
                    # Rate-based: proportional to how much this team deviates from league avg
                    matchup_factor = (opp_allowed - lg_avg) / lg_avg
                    matchup_factor = max(-0.20, min(0.20, matchup_factor))  # cap +/-20%
                    matchup_adj = base_proj * matchup_factor
                else:
                    # Fallback to rank-based
                    matchup_factor = (def_rank - 15) / DEF_ADJUSTMENT_DIVISOR
                    matchup_adj = base_proj * matchup_factor
            elif stat in COMBO_STATS:
                # Combo stats: weighted average of component defense rates
                combo_components = {
                    'pra': ['avg_pts_allowed', 'reb_allowed', 'ast_allowed'],
                    'pr': ['avg_pts_allowed', 'reb_allowed'],
                    'pa': ['avg_pts_allowed', 'ast_allowed'],
                    'ra': ['reb_allowed', 'ast_allowed'],
                }
                components = combo_components.get(stat, [])
                if league_avg_data and components:
                    factors = []
                    for comp_key in components:
                        opp_val = opp_data.get(comp_key, 0)
                        lg_val = league_avg_data.get(comp_key, 0)
                        if lg_val > 0 and opp_val > 0:
                            f = (opp_val - lg_val) / lg_val
                            factors.append(max(-0.20, min(0.20, f)))
                    if factors:
                        matchup_factor = sum(factors) / len(factors)
                        matchup_adj = base_proj * matchup_factor
                    else:
                        matchup_factor = (def_rank - 15) / DEF_ADJUSTMENT_DIVISOR
                        matchup_adj = base_proj * matchup_factor
                else:
                    matchup_factor = (def_rank - 15) / DEF_ADJUSTMENT_DIVISOR
                    matchup_adj = base_proj * matchup_factor
            else:
                # Fallback to rank-based
                matchup_factor = (def_rank - 15) / DEF_ADJUSTMENT_DIVISOR
                matchup_adj = base_proj * matchup_factor

            adjusted_proj += matchup_adj

            # Pace context (Layer 4) — now adjusts projection
            pace_rank = opp_data.get('pace_rank', 15)
            pace = opp_data.get('pace', 0)
            league_avg_pace = 100.0  # approximate league average
            if pace > 0:
                # Pace adjustment: proportional to deviation from league avg
                # ~5% max impact (fast game = more possessions = more stats)
                pace_factor = (pace - league_avg_pace) / league_avg_pace
                pace_adj = adjusted_proj * pace_factor * 0.5  # dampen to ~2.5% max
                adjusted_proj += pace_adj
            if pace_rank <= 5:
                matchup_note = f"Pace-UP spot (pace {pace}, rank {pace_rank})"
            elif pace_rank >= 26:
                matchup_note = f"Pace-DOWN spot (pace {pace}, rank {pace_rank})"

            if def_rank >= 25:
                matchup_note += f" | Weak {stat.upper()} D (rank {def_rank}/30)"
            elif def_rank <= 5:
                matchup_note += f" | Elite {stat.upper()} D (rank {def_rank}/30)"

            # Add real def rating
            def_rtg = opp_data.get('def_rating', 0)
            if def_rtg:
                matchup_note += f" | DEF RTG {def_rtg}"

    # Track rate-based defense data for return dict
    opp_stat_allowed_rate = 0.0
    opp_stat_vs_league = 0.0
    if opponent and opp_data:
        _allowed_key = DEF_ALLOWED_MAP.get(stat, '')
        if _allowed_key:
            opp_stat_allowed_rate = opp_data.get(_allowed_key, 0)
        opp_stat_vs_league = round(matchup_factor * 100, 1)

    # ── OPPONENT-SPECIFIC HISTORY (NEW in v3) ──
    opp_note = ""
    if player_data.get('opponent_history'):
        oh = player_data['opponent_history']
        opp_note = f"vs {opponent}: {oh['avg']} avg in {oh['games']}g (HR {oh['hit_rate']}%)"
        if oh.get('last_3'):
            opp_note += f" | Last: {oh['last_3']}"

    # ── INJURY ADJUSTMENTS (Layers 6-8) ──
    injury_adj = 0
    injury_notes = []

    # Player's own status
    if player_injury_status:
        status = player_injury_status.lower()
        if status in ['questionable', 'gtd', 'game-time decision']:
            injury_notes.append(f"INJURY RISK: {player_injury_status} — minutes may be limited")
        elif status in ['probable', 'available']:
            injury_notes.append(f"Listed {player_injury_status} — likely full minutes")

    # Teammate injury impact (Layer 7) — Dynamic WITH/WITHOUT from game logs, static fallback
    # v10 FIX: Diminishing returns for multiple teammates out + cap at ±20% of season avg
    dynamic_without_delta = 0.0
    dynamic_bumps = []  # collect bumps, apply diminishing returns after
    if injured_out:
        for out_player in injured_out:
            # Try dynamic WITH/WITHOUT from cached game logs (zero new API calls)
            dynamic_used = False
            try:
                without_data = fetcher.get_without_stats(player_name, out_player, stat)
                if without_data and without_data['games_without'] >= 3:
                    bump = without_data['delta']
                    if bump != 0:
                        dynamic_bumps.append((out_player, bump, without_data))
                    dynamic_used = True
            except Exception:
                pass

            # Fall back to static injury_impacts.json if dynamic unavailable
            if not dynamic_used and injury_data:
                impacts = injury_data.get('teammate_impacts', {})
                for key, impact in impacts.items():
                    if (impact['star_out'].lower() in out_player.lower() or
                        out_player.lower() in impact['star_out'].lower()):
                        if (impact['player_affected'].lower() in player_name.lower() or
                            player_name.lower() in impact['player_affected'].lower()):

                            # v3 FIX: Only apply the bump to the matching base stat
                            bump = 0
                            if stat in ['pts', 'ast', 'reb']:
                                bump = impact.get(stat, 0)
                            elif stat == 'pra':
                                bump = impact.get('pts', 0) + impact.get('ast', 0) + impact.get('reb', 0)
                            elif stat == 'pr':
                                bump = impact.get('pts', 0) + impact.get('reb', 0)
                            elif stat == 'pa':
                                bump = impact.get('pts', 0) + impact.get('ast', 0)
                            elif stat == 'ra':
                                bump = impact.get('reb', 0) + impact.get('ast', 0)
                            elif stat == '3pm':
                                bump = impact.get('3pm', 0)

                            if bump != 0:
                                injury_adj += bump
                                injury_notes.append(
                                    f"WITH {impact['star_out']} OUT: {stat.upper()} {bump:+.1f} "
                                    f"(based on {impact.get('games_without', '?')}g sample)")
                            break

    # Apply dynamic bumps with diminishing returns (sorted by magnitude)
    if dynamic_bumps:
        dynamic_bumps.sort(key=lambda x: abs(x[1]), reverse=True)
        for i, (out_player, bump, without_data) in enumerate(dynamic_bumps):
            weight = 1.0 / (2 ** i)  # 1.0, 0.5, 0.25, 0.125...
            weighted_bump = bump * weight
            dynamic_without_delta += bump  # raw delta for reporting
            injury_adj += weighted_bump
            injury_notes.append(
                f"DYNAMIC W/O {out_player}: {stat.upper()} {bump:+.1f} (wt {weight:.0%}) "
                f"(avg {without_data['avg_without']} in {without_data['games_without']}g without vs {without_data['avg_with']} with)")

    # Cap injury adjustment at ±20% of season avg (min ±2 for base stats, ±3 for combos)
    _savg = player_data['season_avg']
    if _savg > 0:
        if stat in ['pra', 'pr', 'pa', 'ra']:
            max_inj = max(_savg * 0.20, 3.0)
        else:
            max_inj = max(_savg * 0.20, 2.0)
        injury_adj = max(-max_inj, min(max_inj, injury_adj))

    adjusted_proj += injury_adj

    # ── USAGE RATE (Layer 7b - v5 dynamic) ──
    usage_rate = 0.0
    usage_trend = 0.0
    try:
        usage_data = fetcher.get_usage_metrics(player_name)
        if usage_data:
            usage_rate = usage_data['l10_usage']
            usage_trend = usage_data['usage_trend']
    except Exception:
        pass

    # ── FOUL TROUBLE ADJUSTMENT (Layer 15 - NEW v4) ──
    # Note: applied after direction is computed below (needs direction)
    foul_adj = 0
    foul_note = ""

    # ── MINUTES TREND ADJUSTMENT (Layer 5b - NEW v4) ──
    mins_adj = 0
    mins_note = ""
    if player_data.get('l5_mins_avg') and player_data.get('season_mins_avg'):
        l5_mins = player_data['l5_mins_avg']
        season_mins = player_data['season_mins_avg']
        if season_mins > 0:
            mins_ratio = l5_mins / season_mins
            if mins_ratio < 0.85:
                # Minutes trending down significantly — reduce projection
                mins_adj = base_proj * (mins_ratio - 1.0) * 0.5  # half the proportional reduction
                mins_note = f"MIN_DOWN: L5 {l5_mins:.0f}min vs season {season_mins:.0f}min ({mins_ratio:.0%})"
            elif mins_ratio > 1.10:
                # Minutes trending up — slight boost
                mins_adj = base_proj * (mins_ratio - 1.0) * 0.3
                mins_note = f"MIN_UP: L5 {l5_mins:.0f}min vs season {season_mins:.0f}min ({mins_ratio:.0%})"

    adjusted_proj += mins_adj

    # ── HOT/COLD STREAK (Layer 9) ──
    streak_adj = 0
    streak_note = ""
    streak_status = player_data['streak_status']
    streak_pct = player_data['streak_pct']
    l3_avg = player_data['l3_avg']
    l10_avg = player_data['l10_avg']

    if streak_status == "HOT":
        streak_adj = base_proj * STREAK_ADJ_FACTOR
        streak_note = f"HOT: L3 {l3_avg} vs L10 {l10_avg} ({streak_pct:+.0f}%)"
    elif streak_status == "COLD":
        streak_adj = base_proj * -STREAK_ADJ_FACTOR
        streak_note = f"COLD: L3 {l3_avg} vs L10 {l10_avg} ({streak_pct:+.0f}%)"

    adjusted_proj += streak_adj

    # ── FATIGUE & REST (Layer 10) — B2B + multi-day fatigue ──
    b2b_note = ""
    b2b_adj = 0
    fatigue_adj = 0
    rest_days = None
    games_in_7 = None
    travel_distance = 0
    travel_miles_7day = 0
    tz_shifts_7day = 0

    # Compute multi-day fatigue from game log dates
    df = fetcher.get_player_log(player_name)
    if not df.empty and len(df) >= 2:
        import pandas as pd
        from datetime import datetime as _dt, timedelta as _td
        try:
            today = pd.Timestamp(_dt.now().date())
            game_dates = pd.to_datetime(df['GAME_DATE'], format='mixed').sort_values(ascending=False)
            if len(game_dates) >= 1:
                last_game = game_dates.iloc[0]
                rest_days = (today - last_game).days
            # Games in last 7 days
            week_ago = today - _td(days=7)
            games_in_7 = int((game_dates >= week_ago).sum())
        except Exception:
            pass

        # ── LIVE TRAVEL COMPUTATION ── from game log MATCHUP strings
        try:
            from venue_data import VENUE_MAP, TZ_ORDINAL, haversine_miles as _haversine
            _today_t = pd.Timestamp(_dt.now().date())
            _week_ago_t = _today_t - _td(days=7)
            _df_t = df.copy()
            _df_t['_gd'] = pd.to_datetime(_df_t['GAME_DATE'], format='mixed')
            _recent = _df_t[_df_t['_gd'] >= _week_ago_t].sort_values('_gd')
            _prev_venue = None
            _prev_tz = None
            _tot_miles = 0.0
            _tz_cnt = 0
            for _, _row in _recent.iterrows():
                _m = str(_row.get('MATCHUP', ''))
                if ' vs. ' in _m:
                    _cv = _m.split(' vs. ')[0].strip()
                elif ' @ ' in _m:
                    _cv = _m.split(' @ ')[1].strip()
                else:
                    continue
                if _prev_venue and _prev_venue in VENUE_MAP and _cv in VENUE_MAP:
                    _pv = VENUE_MAP[_prev_venue]
                    _cvd = VENUE_MAP[_cv]
                    _tot_miles += _haversine(_pv['lat'], _pv['lng'], _cvd['lat'], _cvd['lng'])
                _ctz = VENUE_MAP.get(_cv, {}).get('tz')
                if _ctz and _prev_tz and _ctz != _prev_tz:
                    _tz_cnt += 1
                if _ctz:
                    _prev_tz = _ctz
                _prev_venue = _cv
            travel_miles_7day = round(_tot_miles)
            tz_shifts_7day = _tz_cnt
        except Exception:
            pass

    # Multi-day fatigue penalty (stacks with B2B)
    if games_in_7 is not None and games_in_7 >= 4:
        # 4-in-7 = -2%, 5-in-7 = -3%
        fatigue_pct = -0.02 if games_in_7 == 4 else -0.03
        fatigue_adj = base_proj * fatigue_pct
        b2b_note += f" | HEAVY SCHEDULE ({games_in_7} games in 7 days, {fatigue_pct*100:.0f}%)"

    if is_b2b:
        if not b2b_note:
            b2b_note = "B2B game — monitor for rest/minutes cut"
        else:
            b2b_note = "B2B game" + b2b_note
        if player_data['mins_30plus_pct'] > 70:
            b2b_note += " (starter, slight rest risk)"
        # Distance-scaled B2B penalty
        if game and '@' in game:
            try:
                from venue_data import get_travel_distance
                parts = game.split('@')
                away_abr = parts[0]
                home_abr = parts[1] if len(parts) > 1 else ''
                if away_abr and home_abr:
                    travel_distance = get_travel_distance(away_abr, home_abr)
                    if travel_distance > 1500:
                        b2b_adj = base_proj * -0.04
                        b2b_note += f" | LONG TRAVEL ({travel_distance}mi) -4%"
                    elif travel_distance > 500:
                        b2b_adj = base_proj * -0.02
                        b2b_note += f" | MED TRAVEL ({travel_distance}mi) -2%"
                    else:
                        b2b_adj = base_proj * -0.01
                        b2b_note += f" | SHORT TRAVEL ({travel_distance}mi) -1%"
            except ImportError:
                pass

    adjusted_proj += b2b_adj + fatigue_adj

    # ── BLOWOUT RISK (Layer 11) — Minutes-projected ──
    blowout_note = ""
    blowout_adj = 0
    if spread is not None:
        abs_spread = abs(spread)
        if abs_spread >= BLOWOUT_SPREAD_THRESHOLD:
            # Scale penalty with spread magnitude
            # 10-pt spread = ~5% mins reduction, 15-pt = ~10%, 20-pt = ~15%
            mins_reduction_pct = min((abs_spread - 8) * 0.01, 0.15)
            if spread < -BLOWOUT_SPREAD_THRESHOLD:
                # Big favorite: starters sit Q4 in blowout win
                blowout_adj = base_proj * -mins_reduction_pct
                blowout_note = f"Big favorite ({spread}) — est. {mins_reduction_pct*100:.0f}% mins cut"
            elif spread > BLOWOUT_SPREAD_THRESHOLD:
                # Big underdog: starters may sit if blown out
                blowout_adj = base_proj * -(mins_reduction_pct * 0.7)  # less certain
                blowout_note = f"Big underdog (+{spread}) — est. {mins_reduction_pct*70:.0f}% mins risk"

    adjusted_proj += blowout_adj

    # ── SELF-HEAL CORRECTIONS (auto-applied from corrections.json) ──
    corrections = _load_corrections()
    if corrections:
        # Pre-compute direction to inform corrections
        _pre_dir = "OVER" if (adjusted_proj - line) > 0 else "UNDER"
        corr_adj, corr_notes = _apply_corrections(stat, _pre_dir, adjusted_proj, corrections)
        adjusted_proj += corr_adj
        injury_notes.extend(corr_notes)

    # ── v13: HIT RATE DIRECTION CALIBRATION ──
    # L10/season hit rate = how often player ACTUALLY goes OVER this line
    # If HR < 50%, player goes UNDER more often — that's empirical, not theoretical
    l10_hr = player_data['l10_hit_rate']
    season_hr = player_data.get('season_hit_rate', l10_hr)
    blend_hr = 0.6 * l10_hr + 0.4 * season_hr  # weight recent more
    hr_adj = 0
    if blend_hr <= 30:
        hr_adj = adjusted_proj * -0.05   # rarely goes OVER → strong UNDER lean
    elif blend_hr <= 40:
        hr_adj = adjusted_proj * -0.03   # usually goes UNDER
    elif blend_hr <= 50:
        hr_adj = adjusted_proj * -0.015  # slight UNDER lean
    elif blend_hr >= 80:
        hr_adj = adjusted_proj * 0.03    # consistently goes OVER → protect OVER signal
    elif blend_hr >= 70:
        hr_adj = adjusted_proj * 0.015   # often goes OVER
    adjusted_proj += hr_adj

    # ── v14: EMPIRICAL UNDER CALIBRATION ──
    # Cross-day data (4,982 lines): projections run systematically high.
    # Actual OVER rate = 39.2%. 5% correction (was 2.5%).
    adjusted_proj *= (1 - OVER_BIAS_CORRECTION)

    # ── v14: STAT-SPECIFIC UNDER CORRECTION ──
    # Some stats (BLK, STL, 3PM) go UNDER at extreme rates.
    stat_extra_corr = STAT_UNDER_EXTRA_CORRECTION.get(stat, 0)
    if stat_extra_corr > 0:
        adjusted_proj *= (1 - stat_extra_corr)

    # ── GAP & TIER (Layers 12-13) ──
    gap = adjusted_proj - line
    abs_gap = abs(gap)

    # ── v14: UNDER-DOMINANT DIRECTION MODEL ──
    # Default: UNDER. OVER requires: gap > 3.0 AND L10 HR >= 65.
    # Backtested: "always UNDER" = 60.7%. No OVER filter beats this.
    # We allow narrow OVER calls only for the strongest statistical cases,
    # but the structural bet is UNDER.
    if gap > THIN_GAP_UNDER_THRESHOLD and blend_hr >= OVER_CONFIRMATION_HR:
        direction = "OVER"
    else:
        direction = "UNDER"
        if gap > 0:
            gap = -gap  # flip sign for UNDER calls where projection was above line

    # ── FOUL TROUBLE ADJUSTMENT (Layer 15 - applied after direction computed) ──
    # Foul trouble reinforces UNDER (already the default in v14)
    if player_data.get('foul_trouble_risk') and direction == 'OVER':
        foul_adj = base_proj * -0.05
        foul_note = f"FOUL RISK: L5 PF avg {player_data.get('l5_avg_pf', 0):.1f} >= 4.0 — minutes risk"
        adjusted_proj += foul_adj
        gap = adjusted_proj - line
        abs_gap = abs(gap)
        # v14: Re-apply UNDER-dominant direction after foul adjustment
        if gap > THIN_GAP_UNDER_THRESHOLD and blend_hr >= OVER_CONFIRMATION_HR:
            direction = "OVER"
        else:
            direction = "UNDER"
            if gap > 0:
                gap = -gap

    # ── v6: USAGE REDISTRIBUTION RISK FOR UNDERs (Layer 7b) ──
    if direction == 'UNDER' and same_team_out_count >= 2:
        usage_bump = same_team_out_count * 1.5
        if stat in ['pra', 'pr', 'pa', 'ra']:
            usage_bump *= 1.5  # Combo stats amplify usage redistribution
        adjusted_proj += usage_bump
        injury_notes.append(
            f"USAGE RISK: {same_team_out_count} teammates OUT -> +{usage_bump:.1f} usage boost")
        # Recalculate gap after usage bump
        gap = adjusted_proj - line
        abs_gap = abs(gap)
        # v14: Re-apply UNDER-dominant direction after usage bump
        if gap > THIN_GAP_UNDER_THRESHOLD and blend_hr >= OVER_CONFIRMATION_HR:
            direction = "OVER"
        else:
            direction = "UNDER"
            if gap > 0:
                gap = -gap

    # v4: Combo stat penalty — PRA/PR/PA/RA need higher gap due to higher variance
    effective_gap = abs_gap
    combo_penalized = False
    if stat in COMBO_STATS:
        effective_gap = max(0, abs_gap - COMBO_GAP_PENALTY)
        combo_penalized = True

    # ── v14: CONFIDENCE-BASED TIER ASSIGNMENT ──
    # UNDER picks: use composite confidence score (backtested: S=73.6% → F=54.3%)
    # OVER picks: use legacy gap-based tiers (rare — only ~0.3% of picks)
    under_conf_score = 0.0
    if direction == "UNDER":
        under_conf_score = under_confidence_score(
            player_data, stat, gap, streak_status,
            is_b2b=is_b2b, spread=spread, is_home=is_home
        )
        tier = "SKIP"
        for tier_name, lo, hi in UNDER_CONFIDENCE_TIERS:
            if lo <= under_conf_score < hi:
                tier = tier_name
                break
    else:
        # OVER picks (rare) use legacy gap-based tiers
        tier = "SKIP"
        for tier_name, lo, hi in TIERS:
            if lo <= effective_gap < hi:
                tier = tier_name
                break

    # v4: Injury severity tiers (was: all GTD/Q get 1-tier downgrade)
    if player_injury_status:
        status_lower = player_injury_status.lower()
        downgrade_steps = INJURY_SEVERITY.get(status_lower, 0)
        if isinstance(downgrade_steps, int) and downgrade_steps > 0:
            tier_order = ["S", "A", "B", "C", "D", "F"]
            if tier in tier_order:
                idx = tier_order.index(tier)
                new_idx = min(idx + downgrade_steps, len(tier_order) - 1)
                if new_idx != idx:
                    old_tier = tier
                    tier = tier_order[new_idx]
                    injury_notes.append(f"Tier downgraded {old_tier}→{tier} due to {player_injury_status} ({downgrade_steps} step{'s' if downgrade_steps > 1 else ''})")

    # ── PREDICTION REASONING (Layer 15 - NEW v5) ──
    # Every pick gets a logic-based reasoning, even coin-flips, so we can track what we got right/wrong
    reasoning = _build_reasoning(
        player_name, stat, line, direction, tier, round(adjusted_proj, 1),
        round(gap, 1), player_data, matchup_note, streak_status,
        blowout_note, is_b2b, injury_notes, mins_note, opp_note
    )

    return {
        "player": player_name,
        "stat": stat,
        "line": line,
        "projection": round(adjusted_proj, 1),
        "raw_projection": round(raw_proj, 1),
        "base_projection": round(base_proj, 1),
        "hr_calibration_adj": round(hr_adj, 1),
        "blend_hit_rate": round(blend_hr, 1),
        "gap": round(gap, 1),
        "abs_gap": round(abs_gap, 1),
        "effective_gap": round(effective_gap, 1),
        "direction": direction,
        "tier": tier,
        "under_conf_score": round(under_conf_score, 1),
        "reasoning": reasoning,
        "combo_penalized": combo_penalized,
        "season_avg": player_data['season_avg'],
        "l10_avg": player_data['l10_avg'],
        "l5_avg": player_data['l5_avg'],
        "home_avg": player_data['home_avg'],
        "away_avg": player_data['away_avg'],
        "split_adjustment": round(split_adj, 1),
        "matchup_adjustment": round(matchup_adj, 1),
        "opp_stat_allowed_rate": round(opp_stat_allowed_rate, 1),
        "opp_stat_allowed_vs_league_avg": opp_stat_vs_league,
        "l10_hit_rate": player_data['l10_hit_rate'],
        "l5_hit_rate": player_data['l5_hit_rate'],
        "season_hit_rate": player_data['season_hit_rate'],
        "l10_source": player_data['l10_source'],
        "mins_30plus_pct": player_data['mins_30plus_pct'],
        "mins_adj": round(mins_adj, 1),
        "mins_note": mins_note,
        "matchup_note": matchup_note.strip(" |"),
        "opponent_note": opp_note,
        "injury_adjustment": round(injury_adj, 1),
        "injury_notes": injury_notes,
        "player_injury_status": player_injury_status,
        "streak_status": streak_status,
        "streak_pct": round(streak_pct, 1),
        "streak_adj": round(streak_adj, 1),
        "streak_note": streak_note,
        "l3_avg": player_data['l3_avg'],
        "is_b2b": is_b2b,
        "b2b_note": b2b_note,
        "b2b_adj": round(b2b_adj + fatigue_adj, 1),
        "rest_days": rest_days,
        "games_in_7": games_in_7,
        "travel_distance": travel_distance,
        "travel_miles_7day": travel_miles_7day,
        "tz_shifts_7day": tz_shifts_7day,
        "spread": spread,
        "blowout_note": blowout_note,
        "blowout_adj": round(blowout_adj, 1),
        "opponent": opponent,
        "is_home": is_home,
        "games_used": player_data['games_played'],
        "game": game,
        "opponent_history": player_data.get('opponent_history'),
        "l10_values": player_data.get('l10_values', []),
        "l10_floor": player_data.get('l10_floor', 0),
        "l10_miss_count": player_data.get('l10_miss_count', 0),
        "same_team_out_count": same_team_out_count,
        # v4 scout data (from nba_fetcher, zero new API calls)
        "l10_avg_plus_minus": player_data.get('l10_avg_plus_minus', 0),
        "l10_avg_pf": player_data.get('l10_avg_pf', 0),
        "l5_avg_pf": player_data.get('l5_avg_pf', 0),
        "foul_trouble_risk": player_data.get('foul_trouble_risk', False),
        "efficiency_trend": player_data.get('efficiency_trend', 0),
        "foul_adj": round(foul_adj, 1),
        "foul_note": foul_note,
        # v5 dynamic usage + injury data
        "usage_rate": round(usage_rate, 3),
        "usage_trend": round(usage_trend, 3),
        "dynamic_without_delta": round(dynamic_without_delta, 1),
        "data_source": "nba_api_live",
        "pipeline_version": "v14",
    }


def build_parlays(results):
    """
    Build parlay recommendations from graded results.
    v4 upgrades:
    - Team diversity enforcement (max 1 player per team across entire parlay)
    - S-tier UNDERs now eligible (Mar 11: 77.6% UNDER accuracy)
    - BLK/STL anchor bonus
    - Combo stat penalty
    - L5 trend filter
    """
    # S/A tier OVERs
    core_over = [r for r in results if r.get('direction') == 'OVER' and 'error' not in r
                 and r.get('l10_hit_rate', 0) >= 60]
    core_under = [r for r in results if r.get('direction') == 'UNDER' and 'error' not in r
                  and r.get('l10_hit_rate', 0) >= 60]
    flex = [r for r in results if r.get('direction') == 'OVER' and 'error' not in r
            and r.get('l10_hit_rate', 0) >= 50]

    all_candidates = core_over + core_under + flex
    # Deduplicate
    seen = set()
    unique = []
    for r in all_candidates:
        key = (r['player'], r['stat'])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    def parlay_score(r):
        gap = r.get('abs_gap', 0)
        hr = r.get('l10_hit_rate', 50) / 100
        l5_hr = r.get('l5_hit_rate', 50) / 100
        organic = gap - abs(r.get('injury_adjustment', 0))
        score = gap * 0.4 + hr * 0.25 + l5_hr * 0.15 + organic * 0.2

        # BLK/STL bonus (85%+ accuracy historically)
        if r.get('stat') in ['blk', 'stl']:
            score *= 1.2
        # Combo stat penalty
        if r.get('stat') in COMBO_STATS and gap < 4.0:
            score *= 0.7
        # Streak adjustments
        if r.get('streak_status') == 'HOT':
            score *= 1.1
        elif r.get('streak_status') == 'COLD':
            score *= 0.85
        # Injury penalty
        if r.get('player_injury_status') in ['Questionable', 'GTD', 'Doubtful']:
            score *= 0.5
        # v6: Role player variance cap
        mins_pct = r.get('mins_30plus_pct', 100)
        if mins_pct < 50:
            score *= 0.7  # Inconsistent playing time
        # v6: Regression risk — low-volume player with high L10 HR
        s_avg = r.get('season_avg', 999)
        if s_avg < 20 and hr >= 0.80:
            games = r.get('games_used', 60)
            reg_factor = max(0.75, min(1.0, games / 70))
            score *= reg_factor
        # v6: Trend reversal — L10 much higher than L5
        if hr - l5_hr > 0.15:
            score *= 0.85
        # v6: UNDER with same-team stars out = usage redistribution risk
        if r.get('direction') == 'UNDER' and r.get('same_team_out_count', 0) >= 2:
            score *= 0.6  # Heavy penalty (Brown scenario)
        elif r.get('direction') == 'UNDER' and r.get('same_team_out_count', 0) == 1:
            score *= 0.85
        return score

    unique.sort(key=parlay_score, reverse=True)

    # Select legs with constraints
    used_games = set()
    used_teams = set()  # v4: team diversity enforcement
    safe_legs = []
    for r in unique:
        game_key = r.get('game', r.get('opponent', ''))
        # No same-game combos
        if game_key and game_key in used_games:
            continue
        # v4: No two players from same team
        player_team = _extract_player_team(r)
        if player_team and player_team in used_teams:
            continue
        # L10 hit rate >= 60%
        if r.get('l10_hit_rate', 0) < 60:
            continue
        # L5 hit rate >= 40% (trend filter)
        if r.get('l5_hit_rate', 0) < 40:
            continue

        safe_legs.append(r)
        if game_key:
            used_games.add(game_key)
        if player_team:
            used_teams.add(player_team)
        if len(safe_legs) >= 9:
            break

    parlays = {}
    if len(safe_legs) >= 3:
        parlays['conservative_3leg'] = {
            "legs": [_leg_summary(l) for l in safe_legs[:3]],
            "description": "Conservative 3-leg: Top 3 edges, all different games/teams"
        }
    if len(safe_legs) >= 4:
        parlays['main_4leg'] = {
            "legs": [_leg_summary(l) for l in safe_legs[:4]],
            "description": "Main 4-leg: Strong core + 1 flex"
        }
    if len(safe_legs) >= 5:
        parlays['standard_5leg'] = {
            "legs": [_leg_summary(l) for l in safe_legs[:5]],
            "description": "Standard 5-leg: Full parlay with solid edges"
        }
    if len(safe_legs) >= 6:
        parlays['aggressive_6leg'] = {
            "legs": [_leg_summary(l) for l in safe_legs[:6]],
            "description": "Aggressive 6-leg: Higher payout, more risk"
        }

    return parlays


def build_game_locks(results):
    """
    Pick the single best lock from each game.
    Backtested Jan 2021–Mar 2026: 65.9% hit rate across 759 games.

    Tuned filters from 5-year backtest:
    - UNDERs are the edge (66%), OVERs need near-perfect L10 to be viable
    - PRA/PR killed (55-60% = coin flip), PA/RA kept (65-76%)
    - ABSOLUTE = UNDER-dominant with extreme consistency, or OVER with 90%+ L10
    - LEAN = relaxed fallback, combos PA/RA allowed
    """
    ALLOWED_COMBOS = ['pa', 'ra']  # PA=65.6%, RA=76.5% in backtest

    # ABSOLUTE lock: UNDER with bulletproof hit rates, or OVER near-perfect on simple stats
    candidates = [
        r for r in results
        if 'error' not in r
        and r.get('abs_gap', 0) >= 3.0
        and r.get('l10_hit_rate', 0) >= 60
        and r.get('streak_status') != 'COLD'
        and not r.get('player_injury_status')  # no injury status at all
        and (r.get('stat') not in COMBO_STATS or r.get('stat') in ALLOWED_COMBOS)
        and (
            # UNDER path: the proven edge
            (r.get('direction') == 'UNDER'
             and r.get('l10_hit_rate', 0) >= 80
             and r.get('l5_hit_rate', 0) >= 80)
            or
            # OVER path: simple stats only, near-perfect form
            (r.get('direction') == 'OVER'
             and r.get('stat') not in COMBO_STATS
             and r.get('l10_hit_rate', 0) >= 90
             and r.get('l5_hit_rate', 0) >= 100
             and r.get('abs_gap', 0) >= 4.0)
        )
    ]

    def lock_score(r):
        gap = r.get('abs_gap', 0)
        l10 = r.get('l10_hit_rate', 0) / 100
        l5 = r.get('l5_hit_rate', 0) / 100
        score = l10 * 0.30 + l5 * 0.25 + gap * 0.20
        # Organic gap (not inflated by injury adjustments)
        organic = gap - abs(r.get('injury_adjustment', 0))
        score += organic * 0.15
        # HOT streak bonus
        if r.get('streak_status') == 'HOT':
            score *= 1.10
        # BLK/STL historically most reliable
        if r.get('stat') in ['blk', 'stl']:
            score *= 1.25
        # Season avg margin from line — penalize thin margins
        season_avg = r.get('season_avg', 0)
        line = r.get('line', 0)
        if season_avg and line:
            margin = abs(season_avg - line)
            if margin >= 2.0:
                score *= 1.10
            elif margin < 1.5:
                score *= 0.85
        return score

    from collections import defaultdict
    by_game = defaultdict(list)
    for r in candidates:
        by_game[r.get('game', 'UNKNOWN')].append(r)

    locks = {}
    for game, picks in by_game.items():
        picks.sort(key=lock_score, reverse=True)
        best = picks[0]
        locks[game] = _leg_summary(best)
        locks[game]['lock_score'] = round(lock_score(best), 3)
        locks[game]['confidence'] = 'ABSOLUTE'

    # LEAN fallback for uncovered games
    # PA/RA combos allowed. UNDERs: L10>=70%, L5>=60%. OVERs: L10>=80%, L5>=80% simple only.
    all_games = set(r.get('game', '') for r in results if 'error' not in r and r.get('game'))
    missing_games = all_games - set(locks.keys())

    if missing_games:
        fallback = [
            r for r in results
            if 'error' not in r
            and r.get('abs_gap', 0) >= 3.0
            and r.get('l10_hit_rate', 0) >= 60
            and r.get('streak_status') != 'COLD'
            and (r.get('player_injury_status') or '').lower() not in ['doubtful', 'out', 'questionable', 'gtd']
            and r.get('game', '') in missing_games
            and (r.get('stat') not in COMBO_STATS or r.get('stat') in ALLOWED_COMBOS)
            and (
                (r.get('direction') == 'UNDER'
                 and r.get('l10_hit_rate', 0) >= 70
                 and r.get('l5_hit_rate', 0) >= 60)
                or
                (r.get('direction') == 'OVER'
                 and r.get('stat') not in COMBO_STATS
                 and r.get('l10_hit_rate', 0) >= 80
                 and r.get('l5_hit_rate', 0) >= 80)
            )
        ]
        fb_by_game = defaultdict(list)
        for r in fallback:
            fb_by_game[r.get('game', 'UNKNOWN')].append(r)

        for game, picks in fb_by_game.items():
            picks.sort(key=lock_score, reverse=True)
            best = picks[0]
            locks[game] = _leg_summary(best)
            locks[game]['lock_score'] = round(lock_score(best), 3)
            locks[game]['confidence'] = 'STRONG'

    return locks


def _build_reasoning(player, stat, line, direction, tier, projection, gap,
                     player_data, matchup_note, streak_status,
                     blowout_note, is_b2b, injury_notes, mins_note, opp_note):
    """
    Build a human-readable reasoning string for every pick.
    Even coin-flip F-tier picks get logic so we can review what we got right/wrong.
    """
    reasons = []
    confidence = tier

    # Core projection logic
    season_avg = player_data['season_avg']
    l10_avg = player_data['l10_avg']
    l5_avg = player_data['l5_avg']
    l10_hr = player_data['l10_hit_rate']
    l5_hr = player_data['l5_hit_rate']

    if direction == "OVER":
        if season_avg > line:
            reasons.append(f"Season avg {season_avg} already clears {line} line")
        elif l10_avg > line:
            reasons.append(f"Recent form (L10 {l10_avg}) above {line} line despite lower season avg {season_avg}")
        else:
            reasons.append(f"Projection {projection} edges above {line} line via matchup/context factors despite avgs below line (season {season_avg}, L10 {l10_avg})")
    else:
        if season_avg < line:
            reasons.append(f"Season avg {season_avg} already under {line} line")
        elif l10_avg < line:
            reasons.append(f"Recent form (L10 {l10_avg}) trending below {line} line")
        else:
            reasons.append(f"Projection {projection} pulled below {line} line via matchup/context factors despite avgs above line (season {season_avg}, L10 {l10_avg})")

    # Hit rate context
    if l10_hr >= 80:
        reasons.append(f"Strong L10 hit rate {l10_hr}%")
    elif l10_hr >= 60:
        reasons.append(f"Decent L10 hit rate {l10_hr}%")
    elif l10_hr >= 40:
        reasons.append(f"Mediocre L10 hit rate {l10_hr}% — volatile")
    else:
        reasons.append(f"Low L10 hit rate {l10_hr}% — contrarian pick based on projection")

    # Matchup
    if matchup_note:
        reasons.append(f"Matchup: {matchup_note}")

    # Streak
    if streak_status == "HOT":
        reasons.append(f"HOT streak — momentum favors {direction}")
    elif streak_status == "COLD":
        if direction == "UNDER":
            reasons.append("COLD streak supports UNDER")
        else:
            reasons.append("WARNING: COLD streak works against this OVER")

    # Opponent history
    if opp_note:
        reasons.append(f"History: {opp_note}")

    # Blowout
    if blowout_note:
        reasons.append(f"Blowout risk: {blowout_note}")

    # B2B
    if is_b2b:
        reasons.append("Back-to-back game — fatigue/rest risk")

    # Injuries
    for note in injury_notes:
        reasons.append(note)

    # Minutes trend
    if mins_note:
        reasons.append(mins_note)

    # Confidence qualifier for thin edges
    if abs(gap) < 1.0:
        reasons.append(f"THIN EDGE ({gap:+.1f}) — near coin-flip, tracking for learning")
    elif abs(gap) < 2.0:
        reasons.append(f"Moderate edge ({gap:+.1f}) — lean {direction} but not confident")

    return " | ".join(reasons)


def _extract_player_team(result):
    """Extract team abbreviation from game context for team diversity check"""
    game = result.get('game', '')
    is_home = result.get('is_home')
    if not game or '@' not in game:
        return None
    parts = game.split('@')
    if is_home is True:
        return parts[1] if len(parts) > 1 else None
    elif is_home is False:
        return parts[0] if parts else None
    return None


def _leg_summary(r):
    return {
        "player": r['player'],
        "stat": r['stat'],
        "line": r['line'],
        "direction": r['direction'],
        "tier": r['tier'],
        "gap": r['gap'],
        "projection": r.get('projection', 0),
        "l10_hit_rate": r.get('l10_hit_rate', 0),
        "l5_hit_rate": r.get('l5_hit_rate', 0),
        "season_avg": r.get('season_avg', 0),
        "game": r.get('game', ''),
        "streak": r.get('streak_status', 'NEUTRAL'),
        "matchup_note": r.get('matchup_note', ''),
    }


def grade_board(predictions_file, game_date):
    """
    Post-game grading: compare predictions to actual box scores.
    Uses nba_api to pull real box scores instead of manual entry.
    """
    fetcher = get_fetcher()

    # Load predictions
    with open(predictions_file) as f:
        predictions = json.load(f)

    # Fetch actual box scores
    print(f"Fetching box scores for {game_date}...")
    actuals = fetcher.get_box_scores(game_date)

    if not actuals:
        print("[ERROR] No box scores found. Games may not be finished yet.")
        return None

    print(f"Got stats for {len(actuals)} players")

    # Grade each prediction
    results = []
    hits = 0
    misses = 0
    over_hits = 0
    over_misses = 0
    under_hits = 0
    under_misses = 0

    for pred in predictions:
        player = pred.get('player', '')
        stat = pred.get('stat', '')
        line = pred.get('line', 0)
        direction = pred.get('direction', '')
        tier = pred.get('tier', 'SKIP')
        projection = pred.get('projection', 0)

        if direction == 'SKIP' or 'error' in pred:
            results.append({**pred, 'actual': None, 'result': 'SKIP', 'margin': None})
            continue

        # Find player in actuals (fuzzy match)
        actual_stats = None
        for name, stats in actuals.items():
            if _fuzzy_name_match(player, name):
                actual_stats = stats
                break

        if actual_stats is None:
            results.append({**pred, 'actual': None, 'result': 'DNP', 'margin': None})
            continue

        actual_val = actual_stats.get(stat, 0)
        margin = actual_val - line

        if direction == 'OVER':
            hit = actual_val > line
            if hit:
                over_hits += 1
            else:
                over_misses += 1
        else:
            hit = actual_val < line
            if hit:
                under_hits += 1
            else:
                under_misses += 1

        if hit:
            hits += 1
        else:
            misses += 1

        results.append({
            **pred,
            'actual': actual_val,
            'result': 'HIT' if hit else 'MISS',
            'margin': round(margin, 1),
            'projection_error': round(projection - actual_val, 1),
        })

    total = hits + misses
    accuracy = round(hits / total * 100, 1) if total > 0 else 0

    summary = {
        'date': game_date,
        'total_graded': total,
        'hits': hits,
        'misses': misses,
        'accuracy': accuracy,
        'over_accuracy': round(over_hits / (over_hits + over_misses) * 100, 1) if (over_hits + over_misses) > 0 else 0,
        'under_accuracy': round(under_hits / (under_hits + under_misses) * 100, 1) if (under_hits + under_misses) > 0 else 0,
        'by_tier': {},
    }

    # Accuracy by tier
    for tier_name in ['S', 'A', 'B', 'C', 'D', 'F']:
        tier_results = [r for r in results if r.get('tier') == tier_name and r.get('result') in ['HIT', 'MISS']]
        tier_hits = len([r for r in tier_results if r['result'] == 'HIT'])
        tier_total = len(tier_results)
        summary['by_tier'][tier_name] = {
            'total': tier_total,
            'hits': tier_hits,
            'accuracy': round(tier_hits / tier_total * 100, 1) if tier_total > 0 else 0,
        }

    return {
        'summary': summary,
        'results': results,
    }


def _fuzzy_name_match(name1, name2):
    """Fuzzy match two player names (unicode-aware for diacritics like Jokić)"""
    import unicodedata
    def _norm(s):
        nfkd = unicodedata.normalize('NFKD', s)
        return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

    n1 = _norm(name1)
    n2 = _norm(name2)
    if n1 == n2:
        return True
    # Check last name match + first initial
    parts1 = n1.split()
    parts2 = n2.split()
    if len(parts1) >= 2 and len(parts2) >= 2:
        if parts1[-1] == parts2[-1] and parts1[0][0] == parts2[0][0]:
            return True
    return False


if __name__ == '__main__':
    print("NBA Prop Analysis Pipeline v3 - LIVE DATA")
    print("Usage: from analyze_v3 import analyze_player, build_parlays, grade_board")
    print("Functions: analyze_player(), build_parlays(), grade_board()")

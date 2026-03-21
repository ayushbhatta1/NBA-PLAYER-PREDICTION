#!/usr/bin/env python3
"""
Parlay Engine v1.2 — BACKUP Builder (NEXUS v4 is primary)

Data-driven parlay builder based on 2,328-prop cross-day analysis (Mar 11-15):
- UNDER bias: 68.1% vs OVER 47.1%
- COLD+UNDER: 75.3%
- BLK/STL UNDER: 73.3%
- HOT streak trap: 49.2%
- Hardened filters: GTD -0.15 penalty, ensemble_prob >= 0.50 SAFE floor, D/F blocked from AGG

Backup: 1x SAFE 3-leg + 1x AGGRESSIVE 8-leg (saved to engine_parlays.json)
Shadow: 100 diverse 3-leg parlays (30 curated + 70 grid-generated) with consensus tracking
Zero nulls: Every strategy guaranteed to produce output via cascade fallback.
"""

import random
from collections import defaultdict

COMBO_STATS = {'pra', 'pr', 'pa', 'ra', 'stl_blk'}
BASE_STATS = {'pts', 'reb', 'ast', '3pm', 'stl', 'blk'}

# Player trust/blacklist from 484 real Underdog picks
import json as _json, os as _os
_trust_path = _os.path.join(_os.path.dirname(__file__), 'player_trust.json')
_TRUST_DATA = {}
if _os.path.exists(_trust_path):
    with open(_trust_path) as _f:
        _TRUST_DATA = _json.load(_f)
TRUSTED_PLAYERS = set(_TRUST_DATA.get('trust', {}).keys())
BLACKLISTED_PLAYERS = set(_TRUST_DATA.get('blacklist', {}).keys())
DEATH_COMBOS = set(_TRUST_DATA.get('death_combos', {}).keys())


# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════

def _stat_scale(stat):
    """Scale factor for normalizing matchup deltas per stat type."""
    s = stat.lower()
    if s in ('pts', 'pra'):
        return 5.0
    if s in ('reb', 'ast', 'pr', 'pa', 'ra'):
        return 2.0
    if s == '3pm':
        return 1.0
    if s in ('stl', 'blk', 'stl_blk'):
        return 0.5
    return 2.0


def _primary_score(p):
    """Composite score based on cross-day analysis (Mar 11-17).
    v11: 8 new factor groups wired in — opponent intelligence, usage/role,
    efficiency, defense signal, foul trouble, travel, game total. ~±0.15 max."""
    # Use ensemble_prob if available (XGB 60% + MLP 40%), fall back to xgb_prob
    xgb = p.get('ensemble_prob', p.get('xgb_prob', 0.5))
    direction = p.get('direction', 'OVER').upper()
    stat = p.get('stat', '').lower()
    streak = p.get('streak_status', 'NEUTRAL')
    spread = abs(p.get('spread', 0) or 0)

    # UNDER bonus — doubled from 0.15 (UNDER 65% vs OVER 38% across Mar 11/16/17)
    dir_bonus = 0.30 if direction == 'UNDER' else 0

    # OVER penalty for high-spread games (blowout benching risk)
    # Cade scored 6 in a 19.5 spread game, Jokic DNP'd in 15.5 spread
    blowout_pen = 0.0
    if direction == 'OVER' and spread >= 12:
        blowout_pen = -0.05 * (spread - 10) / 5  # -0.02 at 12, -0.10 at 20

    # COLD + UNDER = 75.3% — boosted from 0.08
    streak_adj = 0.0
    if streak == 'HOT':
        streak_adj = -0.08  # HOT trap is real (49.2%)
    elif streak == 'COLD' and direction == 'UNDER':
        streak_adj = 0.12  # COLD+UNDER is the best combo

    # Gap confidence (capped)
    gap_bonus = min(p.get('abs_gap', 0) / 10, 0.10)

    # BLK/STL UNDER = 73.3%
    stat_bonus = 0.05 if stat in ('blk', 'stl') and direction == 'UNDER' else 0

    # Combo penalty
    combo_pen = -0.10 if stat in COMBO_STATS else 0

    # HR floor
    hr_bonus = (p.get('l10_hit_rate', 50) / 100) * 0.10

    # Minutes stability
    mins_bonus = 0.05 if p.get('mins_30plus_pct', 0) >= 70 else 0

    # L5-declining OVER penalty (fading player)
    decline_pen = 0.0
    if direction == 'OVER':
        l5 = p.get('l5_avg', 0)
        l10 = p.get('l10_avg', 0)
        if l10 > 0 and l5 < l10 - 2:
            decline_pen = -0.08

    # Season HR penalty (season HR < 55% = coin flip)
    season_pen = 0.0
    season_hr = p.get('season_hit_rate', 50)
    if season_hr < 55:
        season_pen = -0.05

    # L10 miss count penalty (unreliable prop)
    miss_pen = 0.0
    if p.get('l10_miss_count', 0) >= 5:
        miss_pen = -0.05

    # GTD/Questionable penalty — reduce score for injury-risk players
    gtd_pen = 0.0
    injury_status = (p.get('player_injury_status') or '').upper()
    if 'QUESTIONABLE' in injury_status or 'GTD' in injury_status or 'GAME-TIME' in injury_status:
        gtd_pen = -0.15

    # ── GROUP 1: Opponent Intelligence ──
    opp_delta = p.get('opp_matchup_delta', 0)
    scale = _stat_scale(stat)
    opp_matchup_adj = 0.0
    if opp_delta != 0 and scale > 0:
        norm = opp_delta / scale
        if (direction == 'OVER' and norm > 0) or (direction == 'UNDER' and norm < 0):
            opp_matchup_adj = min(abs(norm) * 0.04, 0.08)
        elif (direction == 'OVER' and norm < 0) or (direction == 'UNDER' and norm > 0):
            opp_matchup_adj = -min(abs(norm) * 0.04, 0.08)

    team_delta = p.get('team_vs_opp_delta', 0)
    team_matchup_adj = 0.03 if ((team_delta > 2 and direction == 'OVER') or (team_delta < -2 and direction == 'UNDER')) else 0.0

    opp_off = p.get('opp_off_pressure', 0)
    opp_off_adj = 0.0
    if (opp_off > 0 and stat in ('pts', 'pra', 'pr', 'pa') and direction == 'OVER'):
        opp_off_adj = 0.03
    elif (opp_off < 0 and stat in ('pts', 'pra', 'pr', 'pa') and direction == 'UNDER'):
        opp_off_adj = 0.03

    # ── GROUP 2: Usage & Role ──
    usage = p.get('usage_rate', 0)
    usage_adj = 0.0
    if usage > 0.30 and direction == 'OVER':
        usage_adj = 0.03
    elif 0 < usage < 0.15 and direction == 'UNDER':
        usage_adj = 0.02

    u_trend = p.get('usage_trend', 0)
    usage_trend_adj = 0.0
    if u_trend > 0.02 and direction == 'OVER':
        usage_trend_adj = 0.03
    elif u_trend < -0.02 and direction == 'UNDER':
        usage_trend_adj = 0.02

    without_delta = p.get('dynamic_without_delta', 0)
    without_adj = 0.04 if ((without_delta >= 2 and direction == 'OVER') or (without_delta <= -2 and direction == 'UNDER')) else 0.0

    usage_boost = p.get('usage_boost', 0)
    usage_boost_adj = 0.0
    if usage_boost > 0 and direction == 'OVER':
        usage_boost_adj = min(usage_boost / 5, 0.04)
    elif usage_boost < 0 and direction == 'UNDER':
        usage_boost_adj = min(abs(usage_boost) / 5, 0.04)

    # ── GROUP 3: Efficiency & Team Context ──
    plus_minus = p.get('l10_avg_plus_minus', 0)
    eff_adj = 0.0
    if plus_minus > 5 and direction == 'OVER':
        eff_adj = 0.02
    elif plus_minus < -5 and direction == 'OVER':
        eff_adj = -0.02

    eff_trend = p.get('efficiency_trend', 0)
    eff_trend_adj = 0.02 if ((eff_trend > 0 and direction == 'OVER') or (eff_trend < 0 and direction == 'UNDER')) else 0.0

    # ── GROUP 4: Defense Signal ──
    opp_allowed = p.get('opp_stat_allowed_vs_league_avg', 0)
    def_adj = 0.03 if ((opp_allowed > 0 and direction == 'OVER') or (opp_allowed < 0 and direction == 'UNDER')) else 0.0

    # ── GROUP 5: Foul Trouble ──
    avg_pf = p.get('l10_avg_pf', 0)
    foul_adj = 0.0
    if avg_pf >= 4.0:
        foul_adj = -0.03 if direction == 'OVER' else 0.03

    # ── GROUP 6: Travel & Fatigue ──
    travel_7d = p.get('travel_miles_7day', 0)
    tz_shifts = p.get('tz_shifts_7day', 0)
    travel_adj = 0.0
    if direction == 'OVER':
        if travel_7d > 5000:
            travel_adj = -0.03
        elif travel_7d > 3000:
            travel_adj = -0.02
        if tz_shifts >= 2:
            travel_adj -= 0.02

    # ── GROUP 7: Game Total ──
    game_total = p.get('game_total_signal', 0)
    game_total_adj = 0.0
    if game_total > 5 and stat in ('pts', 'pra', 'pr', 'pa', 'ast') and direction == 'OVER':
        game_total_adj = 0.03
    elif game_total < -5 and direction == 'UNDER':
        game_total_adj = 0.02

    base = xgb + dir_bonus + streak_adj + gap_bonus + stat_bonus + combo_pen + hr_bonus + mins_bonus + decline_pen + season_pen + blowout_pen + miss_pen + gtd_pen
    new_factors = opp_matchup_adj + team_matchup_adj + opp_off_adj + usage_adj + usage_trend_adj + without_adj + usage_boost_adj + eff_adj + eff_trend_adj + def_adj + foul_adj + travel_adj + game_total_adj
    return base + new_factors


def _sort_fn(sort_key):
    """Return a sort-key lambda for the given sort_key name."""
    if sort_key == 'xgb_prob':
        return lambda p: p.get('xgb_prob', 0) or 0
    elif sort_key == 'mlp_prob':
        return lambda p: p.get('mlp_prob', 0) or 0
    elif sort_key == 'ensemble_prob':
        return lambda p: p.get('ensemble_prob', p.get('xgb_prob', 0)) or 0
    elif sort_key == 'gap':
        return lambda p: p.get('abs_gap', 0)
    elif sort_key == 'hr_weighted':
        return lambda p: p.get('l10_hit_rate', 0) * 0.6 + p.get('l5_hit_rate', 0) * 0.4
    elif sort_key == 'composite_under':
        return _primary_score
    elif sort_key == 'floor':
        return lambda p: (p.get('l10_hit_rate', 0) / 100) * 0.5 + min(p.get('abs_gap', 0) / 5, 1.0) * 0.3 + (min(p.get('mins_30plus_pct', 0) / 100, 1.0)) * 0.2
    elif sort_key == 'season_margin':
        return lambda p: abs(p.get('season_avg', p.get('projection', p.get('line', 0))) - p.get('line', 0))
    else:
        return _primary_score


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_player_team(pick):
    """Extract team abbreviation from game context."""
    game = pick.get('game', '')
    is_home = pick.get('is_home')
    if not game or '@' not in game:
        return None
    parts = game.split('@')
    if is_home is True:
        return parts[1] if len(parts) > 1 else None
    elif is_home is False:
        return parts[0] if parts else None
    return None


def _make_leg(p):
    """Format a prop as a parlay leg dict."""
    return {
        'player': p['player'],
        'stat': p.get('stat', ''),
        'line': p.get('line', 0),
        'direction': p.get('direction', 'OVER'),
        'tier': p.get('tier', '?'),
        'gap': p.get('gap', 0),
        'projection': p.get('projection', 0),
        'l10_hit_rate': p.get('l10_hit_rate', 0),
        'l5_hit_rate': p.get('l5_hit_rate', 0),
        'season_avg': p.get('season_avg', 0),
        'mins_30plus_pct': p.get('mins_30plus_pct', 0),
        'game': p.get('game', ''),
        'is_home': p.get('is_home'),
        'streak': p.get('streak_status', 'NEUTRAL'),
        'xgb_prob': p.get('xgb_prob'),
        'mlp_prob': p.get('mlp_prob'),
        'ensemble_prob': p.get('ensemble_prob'),
        'nexus_score': p.get('nexus_score', 0),
        'l10_miss_count': p.get('l10_miss_count', 0),
        'primary_score': round(_primary_score(p), 4),
        # v11: enrichment fields
        'opp_matchup_delta': p.get('opp_matchup_delta', 0),
        'team_vs_opp_delta': p.get('team_vs_opp_delta', 0),
        'usage_rate': p.get('usage_rate', 0),
        'usage_trend': p.get('usage_trend', 0),
        'efficiency_trend': p.get('efficiency_trend', 0),
        'game_total': p.get('game_total_signal', 0),
        'travel_miles_7day': p.get('travel_miles_7day', 0),
        'opp_off_pressure': p.get('opp_off_pressure', 0),
    }


def _is_eligible(p):
    """Basic eligibility: not error, not injured/GTD/questionable."""
    if 'error' in p or p.get('tier') == 'SKIP':
        return False
    status = (p.get('player_injury_status') or '').upper()
    if any(tag in status for tag in ('OUT', 'DOUBTFUL', 'QUESTIONABLE', 'GTD', 'GAME-TIME', 'DAY-TO-DAY')):
        return False
    return True


def _greedy_select(pool, n_legs, sort_key_fn, excluded_players=None, max_combo=1, max_same_team=1):
    """
    Greedy selection with constraints: no dup player, no same-game, limited same-team,
    limited combos, and correlation penalty (skip highly-correlated same-game pairs).
    Returns list of props (not legs).
    """
    excluded = excluded_players or set()
    sorted_pool = sorted(pool, key=sort_key_fn, reverse=True)

    selected = []
    used_players = set()
    used_games = set()
    team_counts = defaultdict(int)
    combo_count = 0

    for pick in sorted_pool:
        player = pick.get('player', '')
        game = pick.get('game', '')
        team = _get_player_team(pick)
        stat = pick.get('stat', '').lower()

        if player in excluded or player in used_players:
            continue
        if game and game in used_games:
            continue
        if team and team_counts[team] >= max_same_team:
            continue
        if stat in COMBO_STATS:
            if combo_count >= max_combo:
                continue
            combo_count += 1

        # Correlation check: skip if highly correlated (|r| > 0.6) with already-selected player
        corr_list = pick.get('teammate_correlations', [])
        corr_conflict = False
        for c in corr_list:
            if c.get('player', '') in used_players and abs(c.get('correlation', 0)) > 0.6:
                corr_conflict = True
                break
        if corr_conflict:
            continue

        selected.append(pick)
        used_players.add(player)
        if game:
            used_games.add(game)
        if team:
            team_counts[team] += 1

        if len(selected) >= n_legs:
            break

    return selected


def _floor_score(p):
    """Floor safety score — the ONLY thing that matters for parlays.
    Not gap. Not XGB. Does this player's WORST game still hit?
    Plus trust/blacklist from 484 real betting picks."""
    direction = p.get('direction', 'OVER').upper()
    line = p.get('line', 0)
    player = p.get('player', '')
    l10_floor = p.get('l10_floor', 0)  # worst L10 game
    l10_values = p.get('l10_values', [])
    l10_ceiling = max(l10_values) if l10_values else p.get('l10_avg', 0)
    l10_hr = p.get('l10_hit_rate', 0)
    l5_hr = p.get('l5_hit_rate', 0)
    miss_count = p.get('l10_miss_count', 10)
    spread = abs(p.get('spread', 0) or 0)

    score = 0.0

    if direction == 'UNDER':
        # Best case: player's CEILING (best recent game) is still under the line
        ceiling_margin = line - l10_ceiling
        if ceiling_margin > 0:
            score += 0.50  # ceiling doesn't even reach line — near lock
        elif ceiling_margin > -2:
            score += 0.25  # ceiling barely reaches line

        # UNDER base bonus (67% over 10 years)
        score += 0.30

        # HR is king
        score += (l10_hr / 100) * 0.30
        score += (l5_hr / 100) * 0.20

        # Miss count: fewer misses = safer
        score += max(0, (10 - miss_count)) * 0.03

    else:  # OVER
        # Best case: player's FLOOR (worst recent game) still clears
        floor_margin = l10_floor - line
        if floor_margin > 0:
            score += 0.50  # floor clears line — near lock
        elif floor_margin > -2:
            score += 0.20  # floor barely misses

        # HR is king
        score += (l10_hr / 100) * 0.30
        score += (l5_hr / 100) * 0.20

        # Blowout kill for OVERs
        if spread >= 12:
            score -= 0.30
        elif spread >= 8:
            score -= 0.10

        # Miss count
        score += max(0, (10 - miss_count)) * 0.03

    # ── v11: Opponent matchup floor bonus ──
    opp_matchup_delta = p.get('opp_matchup_delta', 0)
    stat = p.get('stat', '').lower()
    if direction == 'OVER' and opp_matchup_delta > 0:
        score += min(opp_matchup_delta / _stat_scale(stat) * 0.05, 0.15)
    elif direction == 'UNDER' and opp_matchup_delta < 0:
        score += min(abs(opp_matchup_delta) / _stat_scale(stat) * 0.05, 0.15)

    # High usage = more consistent touches = higher floor
    usage_rate = p.get('usage_rate', 0)
    if usage_rate > 0.25 and direction == 'OVER':
        score += 0.05

    # Favorable game total = floor bonus
    game_total = p.get('game_total_signal', 0)
    if game_total > 5 and direction == 'OVER' and stat in ('pts', 'pra', 'pr', 'pa', 'ast'):
        score += 0.05
    elif game_total < -5 and direction == 'UNDER':
        score += 0.05

    # Heavy travel = lower floor for OVER
    travel_7d = p.get('travel_miles_7day', 0)
    if direction == 'OVER':
        if travel_7d > 5000:
            score -= 0.10
        elif travel_7d > 3000:
            score -= 0.05

    # Market signal boost — if 30+ sportsbooks agree with our direction, trust it
    market_edge = p.get('market_edge')
    if market_edge is not None:
        if market_edge > 3:
            score += 0.15  # strong edge over market
        elif market_edge > 0:
            score += 0.05  # slight edge
        elif market_edge < -5:
            score -= 0.20  # market disagrees strongly — danger

    # Player trust/blacklist from real betting history
    player = p.get('player', '')
    if player in TRUSTED_PLAYERS:
        score += 0.10
    if player in BLACKLISTED_PLAYERS:
        score -= 0.25  # hard penalty

    # Death combo check (specific player+stat+direction that always loses)
    combo_key = f"{player} {stat.upper()} {direction}"
    if combo_key in DEATH_COMBOS:
        score -= 0.50  # near-kill

    # Line sweet spot: OVER lines 0-16 hit 60%, 16-24 hit 42%
    if direction == 'OVER' and 16 <= line < 24:
        score -= 0.08  # dead zone

    # GTD/Questionable penalty — injury-risk players have lower floor
    injury_status = (p.get('player_injury_status') or '').upper()
    if 'QUESTIONABLE' in injury_status or 'GTD' in injury_status or 'GAME-TIME' in injury_status:
        score -= 0.15

    return score


# ═══════════════════════════════════════════════════════════════
# PRIMARY PARLAYS
# ═══════════════════════════════════════════════════════════════

def build_primary_safe(pool):
    """
    3-Leg SAFE parlay: FLOOR SAFETY approach.
    Not gap chasing. Not XGB ranking. Does the player's WORST game still hit?

    v10.2: Rebuilt around floor safety scoring.
    - UNDERs: player's L10 ceiling near/below line (even best game doesn't kill us)
    - OVERs: player's L10 floor above line (even worst game clears)
    - No combos, no blowout games, base stats only
    - Sorted by _floor_score, not _primary_score
    """
    def _safe_eligible(p):
        if not _is_eligible(p):
            return False
        if p.get('mins_30plus_pct', 0) < 60:
            return False
        if p.get('l10_hit_rate', 0) < 60:
            return False
        if p.get('l10_miss_count', 10) >= 3:
            return False
        # Vig-adjusted probability floor — require 3% edge over break-even
        # At -110 (1.91x), break-even is 52.38%. For 3-leg parlay, effective vig ~14%
        # Require 3% edge over implied probability = 55.4% default
        multiplier = p.get('multiplier')
        if multiplier and multiplier > 1.0:
            implied_prob = 1.0 / multiplier
            min_prob = implied_prob + 0.03  # 3% edge minimum
        else:
            min_prob = 0.554  # default for -110
        ens = p.get('ensemble_prob', p.get('xgb_prob', 0))
        if ens and ens < min_prob:
            return False
        # Hard-block high miss count — unreliable props
        if p.get('l10_miss_count', 10) >= 4:
            return False
        if p.get('stat', '').lower() in COMBO_STATS:
            return False
        # No OVERs in blowout games
        spread = abs(p.get('spread', 0) or 0)
        if p.get('direction', '').upper() == 'OVER' and spread >= 10:
            return False
        return True

    # Level 1: Floor safety — pick 3 safest legs
    filtered = [p for p in pool if _safe_eligible(p)]
    picks = _greedy_select(filtered, 3, _floor_score, max_combo=0)

    if len(picks) >= 3:
        return picks

    # Level 2: Relax miss_count<5, spread<15
    filtered = [p for p in pool if (
        _is_eligible(p) and
        p.get('mins_30plus_pct', 0) >= 55 and
        p.get('l10_hit_rate', 0) >= 50 and
        p.get('l10_miss_count', 10) < 5 and
        p.get('stat', '').lower() not in COMBO_STATS and
        not (p.get('direction', '').upper() == 'OVER' and abs(p.get('spread', 0) or 0) >= 15)
    )]
    picks = _greedy_select(filtered, 3, _floor_score, max_combo=0)

    if len(picks) >= 3:
        return picks

    # Level 3: Survival with primary_score fallback
    filtered = [p for p in pool if (
        _is_eligible(p) and
        p.get('l10_hit_rate', 0) >= 40
    )]
    picks = _greedy_select(filtered, 3, _primary_score, max_combo=1)

    return picks


def build_primary_aggressive(pool, safe_players):
    """
    8-Leg AGGRESSIVE parlay: UNDER-heavy (5+ UNDERs), broader filters.
    v10: OVERs restricted to S-tier only (S-tier OVER 61.5%, all others < 42%).
         Require mins>=55%, xgb_prob>=0.52 for all legs.
         Penalize L5-declining OVERs.
    Excludes players already in SAFE parlay.
    Fallback: minimum 6 legs.
    """
    excluded = set(safe_players)

    # Quality floor for all legs
    # v10.1: Block OVERs in high-spread games (blowout benching)
    filtered = [p for p in pool if (
        _is_eligible(p) and
        p.get('mins_30plus_pct', 0) >= 55 and
        p.get('l10_hit_rate', 0) >= 50 and
        p.get('xgb_prob', 0) >= 0.52 and
        p.get('player', '') not in excluded and
        not (p.get('direction', '').upper() == 'OVER' and abs(p.get('spread', 0) or 0) >= 15)
    )]

    # Split: UNDERs + OVERs (no tier gating)
    unders = [p for p in filtered if p.get('direction', '').upper() == 'UNDER']
    overs = [p for p in filtered if p.get('direction', '').upper() == 'OVER']

    # Target 6 UNDERs (was 5), fill rest with S/A-tier OVERs
    under_picks = _greedy_select(unders, 6, _primary_score, excluded_players=excluded, max_combo=1)
    used_after_unders = excluded | {p['player'] for p in under_picks}

    # Fill OVERs (S-tier only) avoiding used players/games
    used_games = {p.get('game', '') for p in under_picks if p.get('game')}
    over_pool = [p for p in overs if p['player'] not in used_after_unders and p.get('game', '') not in used_games]
    over_picks = _greedy_select(over_pool, 8 - len(under_picks), _primary_score, excluded_players=used_after_unders, max_combo=1)

    picks = under_picks + over_picks

    if len(picks) >= 8:
        return picks[:8]

    # Fallback 1: Fill from remaining OVERs
    used_all = {p['player'] for p in picks} | excluded
    remaining_overs = [p for p in filtered if (
        p['player'] not in used_all and
        p.get('direction', '').upper() == 'OVER'
    )]
    extra = _greedy_select(remaining_overs, 8 - len(picks), _primary_score, excluded_players=used_all, max_combo=1)
    picks.extend(extra)

    if len(picks) >= 8:
        return picks[:8]

    # Fallback 2: Fill from remaining UNDERs with relaxed filters
    used_all = {p['player'] for p in picks} | excluded
    remaining = [p for p in pool if (
        _is_eligible(p) and
        p.get('direction', '').upper() == 'UNDER' and
        p.get('l10_hit_rate', 0) >= 40 and
        p['player'] not in used_all
    )]
    extra = _greedy_select(remaining, 8 - len(picks), _primary_score, excluded_players=used_all, max_combo=2)
    picks.extend(extra)

    if len(picks) >= 6:
        return picks[:8]

    # Level 3: Survival
    filtered = [p for p in pool if (
        _is_eligible(p) and
        p.get('l10_hit_rate', 0) >= 40 and
        p.get('player', '') not in excluded
    )]
    picks = _greedy_select(filtered, 8, _primary_score, excluded_players=excluded, max_combo=2)

    return picks[:8] if len(picks) >= 6 else picks


def _kelly_fraction(legs):
    """Compute Kelly Criterion fraction for a parlay.

    Uses average calibrated xgb_prob across legs as win probability estimate.
    Parlay odds = product of individual leg odds (~1.87x each).
    Returns recommended fraction of bankroll (0.01 = 1%).
    """
    if not legs:
        return 0.0

    # Estimate parlay win probability as product of leg probabilities
    probs = []
    for leg in legs:
        p = leg.get('xgb_prob_calibrated', leg.get('ensemble_prob', leg.get('xgb_prob', 0.5)))
        probs.append(max(0.01, min(0.99, p)))

    win_prob = 1.0
    for p in probs:
        win_prob *= p

    # Parlay odds: each leg ~1.87x, so n-leg parlay = 1.87^n - 1 (net payout)
    n = len(legs)
    payout_mult = 1.87 ** n  # gross payout per $1
    net_odds = payout_mult - 1  # net profit per $1

    # Kelly: f* = (b*p - q) / b where b=net_odds, p=win_prob, q=1-p
    if net_odds <= 0:
        return 0.0
    kelly = (net_odds * win_prob - (1 - win_prob)) / net_odds

    # Fifth-Kelly for safety (research: 1/5 Kelly earned 98% ROI vs full Kelly crashed)
    fifth_kelly = max(0.0, kelly * 0.2)

    # Cap at 3% of bankroll
    return round(min(fifth_kelly, 0.03), 4)


def build_primary_parlays(results):
    """
    Main entry: build 1x SAFE 3-leg + 1x AGGRESSIVE 8-leg.
    Returns dict with both parlays, guaranteed non-empty.
    """
    pool = [p for p in results if _is_eligible(p)]

    safe_picks = build_primary_safe(pool)
    safe_players = [p['player'] for p in safe_picks]
    agg_picks = build_primary_aggressive(pool, safe_players)

    safe_legs = [_make_leg(p) for p in safe_picks]
    agg_legs = [_make_leg(p) for p in agg_picks]

    under_count_safe = sum(1 for l in safe_legs if l.get('direction', '').upper() == 'UNDER')
    under_count_agg = sum(1 for l in agg_legs if l.get('direction', '').upper() == 'UNDER')

    safe_kelly = _kelly_fraction(safe_legs)
    agg_kelly = _kelly_fraction(agg_legs)

    return {
        'safe': {
            'name': 'SAFE 3-LEG',
            'method': 'parlay_engine_v1',
            'legs': safe_legs,
            'legs_total': len(safe_legs),
            'under_count': under_count_safe,
            'kelly_fraction': safe_kelly,
            'suggested_units': round(safe_kelly * 100, 1),
            'description': f'Top 3 by composite score (XGBoost + UNDER bias). {under_count_safe} UNDERs. Suggested: {safe_kelly*100:.1f}% bankroll.',
        },
        'aggressive': {
            'name': 'AGGRESSIVE 8-LEG',
            'method': 'parlay_engine_v1',
            'legs': agg_legs,
            'legs_total': len(agg_legs),
            'under_count': under_count_agg,
            'kelly_fraction': agg_kelly,
            'suggested_units': round(agg_kelly * 100, 1),
            'description': f'8 legs, UNDER-heavy ({under_count_agg} UNDERs). Suggested: {agg_kelly*100:.1f}% bankroll.',
        },
    }


# ═══════════════════════════════════════════════════════════════
# PARAMETRIC SHADOW STRATEGY SYSTEM
# ═══════════════════════════════════════════════════════════════

def _apply_filters(pool, params):
    """Apply strategy parameter filters to a pool. Returns filtered list."""
    filtered = []
    tier_filter = params.get('tier_filter', 'any')
    direction_filter = params.get('direction_filter', 'any')
    stat_filter = params.get('stat_filter', 'any')
    streak_filter = params.get('streak_filter', 'any')
    hr_threshold = params.get('hr_threshold', (0, 0))
    gap_min = params.get('gap_min', 0)
    mins_min = params.get('mins_min', 0)

    # Tier filtering removed — tiers were inversely correlated with accuracy
    for p in pool:
        if not _is_eligible(p):
            continue

        # Direction
        direction = p.get('direction', 'OVER').upper()
        if direction_filter == 'under_only' and direction != 'UNDER':
            continue
        elif direction_filter == 'over_only' and direction != 'OVER':
            continue
        elif direction_filter == 'under_heavy':
            pass  # handled at selection time

        # Stat
        stat = p.get('stat', '').lower()
        if stat_filter == 'base_only' and stat in COMBO_STATS:
            continue
        elif stat_filter == 'no_combo' and stat in COMBO_STATS:
            continue
        elif stat_filter == 'blk_stl' and stat not in ('blk', 'stl'):
            continue

        # Hit rate
        l10_hr = p.get('l10_hit_rate', 0)
        l5_hr = p.get('l5_hit_rate', 0)
        if l10_hr < hr_threshold[0] or l5_hr < hr_threshold[1]:
            continue

        # Gap
        if p.get('abs_gap', 0) < gap_min:
            continue

        # Streak
        streak = p.get('streak_status', 'NEUTRAL')
        if streak_filter == 'cold_only' and streak != 'COLD':
            continue
        elif streak_filter == 'not_hot' and streak == 'HOT':
            continue
        elif streak_filter == 'neutral' and streak != 'NEUTRAL':
            continue

        # Minutes
        if p.get('mins_30plus_pct', 0) < mins_min:
            continue

        filtered.append(p)

    return filtered


def build_parlay_from_params(params, pool, n_legs=3, excluded_players=None):
    """
    Universal builder: interpret param dict, filter, sort, select.
    3-level fallback guarantees output.
    """
    sort_key = params.get('sort_key', 'composite_under')
    combo_max = params.get('combo_max', 1)
    excluded = excluded_players or set()

    # Level 1: Full filters
    filtered = _apply_filters(pool, params)
    filtered = [p for p in filtered if p['player'] not in excluded]

    # For under_heavy: sort UNDERs first
    if params.get('direction_filter') == 'under_heavy':
        unders = [p for p in filtered if p.get('direction', '').upper() == 'UNDER']
        overs = [p for p in filtered if p.get('direction', '').upper() == 'OVER']
        under_picks = _greedy_select(unders, max(n_legs - 1, 2), _sort_fn(sort_key), excluded_players=excluded, max_combo=combo_max)
        used = excluded | {p['player'] for p in under_picks}
        over_picks = _greedy_select(overs, n_legs - len(under_picks), _sort_fn(sort_key), excluded_players=used, max_combo=max(0, combo_max - sum(1 for p in under_picks if p.get('stat', '').lower() in COMBO_STATS)))
        picks = under_picks + over_picks
    else:
        picks = _greedy_select(filtered, n_legs, _sort_fn(sort_key), excluded_players=excluded, max_combo=combo_max)

    if len(picks) >= n_legs:
        return picks[:n_legs]

    # Level 2: Relaxed — drop streak, drop stat, expand tier to SABC, lower HR
    relaxed = {
        'tier_filter': 'SABC',
        'direction_filter': params.get('direction_filter', 'any'),
        'stat_filter': 'any',
        'streak_filter': 'any',
        'hr_threshold': (40, 0),
        'gap_min': 0,
        'mins_min': 0,
        'sort_key': sort_key,
        'combo_max': 2,
    }
    filtered = _apply_filters(pool, relaxed)
    filtered = [p for p in filtered if p['player'] not in excluded]
    picks = _greedy_select(filtered, n_legs, _sort_fn(sort_key), excluded_players=excluded, max_combo=2)

    if len(picks) >= n_legs:
        return picks[:n_legs]

    # Level 3: Survival — any non-D/F/OUT prop, sort by xgb_prob or primary_score
    survival = [p for p in pool if _is_eligible(p) and p['player'] not in excluded]
    picks = _greedy_select(survival, n_legs, _primary_score, excluded_players=excluded, max_combo=3)

    return picks[:n_legs]


# ═══════════════════════════════════════════════════════════════
# 30 CURATED STRATEGIES
# ═══════════════════════════════════════════════════════════════

CURATED_STRATEGIES = [
    # 1-5: COLD + UNDER = 75.3%
    {'name': 'under_cold_xgb', 'description': 'UNDER + COLD streak + XGBoost ranking',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (60, 40), 'gap_min': 0, 'streak_filter': 'cold_only',
     'sort_key': 'xgb_prob', 'mins_min': 50, 'combo_max': 0},
    {'name': 'under_cold_gap', 'description': 'UNDER + COLD streak + gap-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 1.5, 'streak_filter': 'cold_only',
     'sort_key': 'gap', 'mins_min': 50, 'combo_max': 0},
    {'name': 'under_cold_hr', 'description': 'UNDER + COLD streak + hit-rate-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (60, 40), 'gap_min': 0, 'streak_filter': 'cold_only',
     'sort_key': 'hr_weighted', 'mins_min': 50, 'combo_max': 0},
    {'name': 'under_cold_floor', 'description': 'UNDER + COLD streak + floor-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'base_only',
     'hr_threshold': (60, 40), 'gap_min': 0, 'streak_filter': 'cold_only',
     'sort_key': 'floor', 'mins_min': 50, 'combo_max': 0},
    {'name': 'under_cold_composite', 'description': 'UNDER + COLD streak + composite score',
     'direction_filter': 'under_only', 'tier_filter': 'SABC', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 0, 'streak_filter': 'cold_only',
     'sort_key': 'composite_under', 'mins_min': 40, 'combo_max': 1},

    # 6-10: UNDER + gap>=2 = 73.9%
    {'name': 'under_gap2_xgb', 'description': 'UNDER + gap>=2 + XGBoost',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 2.0, 'streak_filter': 'any',
     'sort_key': 'xgb_prob', 'mins_min': 50, 'combo_max': 0},
    {'name': 'under_gap2_hr', 'description': 'UNDER + gap>=2 + hit-rate-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (60, 40), 'gap_min': 2.0, 'streak_filter': 'any',
     'sort_key': 'hr_weighted', 'mins_min': 50, 'combo_max': 0},
    {'name': 'under_gap2_floor', 'description': 'UNDER + gap>=2 + floor-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'base_only',
     'hr_threshold': (60, 40), 'gap_min': 2.0, 'streak_filter': 'any',
     'sort_key': 'floor', 'mins_min': 50, 'combo_max': 0},
    {'name': 'under_gap2_composite', 'description': 'UNDER + gap>=2 + composite',
     'direction_filter': 'under_only', 'tier_filter': 'SABC', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 2.0, 'streak_filter': 'any',
     'sort_key': 'composite_under', 'mins_min': 40, 'combo_max': 1},
    {'name': 'under_gap2_margin', 'description': 'UNDER + gap>=2 + season-margin-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 2.0, 'streak_filter': 'any',
     'sort_key': 'season_margin', 'mins_min': 50, 'combo_max': 0},

    # 11-13: BLK/STL UNDER = 73.3%
    {'name': 'under_blkstl_xgb', 'description': 'BLK/STL UNDER + XGBoost',
     'direction_filter': 'under_only', 'tier_filter': 'SABC', 'stat_filter': 'blk_stl',
     'hr_threshold': (50, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'xgb_prob', 'mins_min': 40, 'combo_max': 0},
    {'name': 'under_blkstl_gap', 'description': 'BLK/STL UNDER + gap-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SABC', 'stat_filter': 'blk_stl',
     'hr_threshold': (50, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'gap', 'mins_min': 40, 'combo_max': 0},
    {'name': 'under_blkstl_hr', 'description': 'BLK/STL UNDER + hit-rate-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SABC', 'stat_filter': 'blk_stl',
     'hr_threshold': (50, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'hr_weighted', 'mins_min': 40, 'combo_max': 0},

    # 14-18: UNDER + tier SAB = 71.4%
    {'name': 'under_sab_xgb', 'description': 'UNDER + tier SAB + XGBoost',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'xgb_prob', 'mins_min': 50, 'combo_max': 1},
    {'name': 'under_sab_gap', 'description': 'UNDER + tier SAB + gap-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 1.0, 'streak_filter': 'any',
     'sort_key': 'gap', 'mins_min': 50, 'combo_max': 1},
    {'name': 'under_sab_hr', 'description': 'UNDER + tier SAB + hit-rate-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'base_only',
     'hr_threshold': (60, 40), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'hr_weighted', 'mins_min': 50, 'combo_max': 0},
    {'name': 'under_sab_floor', 'description': 'UNDER + tier SAB + floor-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (60, 40), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'floor', 'mins_min': 50, 'combo_max': 0},
    {'name': 'under_sab_composite', 'description': 'UNDER + tier SAB + composite',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'composite_under', 'mins_min': 50, 'combo_max': 1},

    # 19-22: Pure XGBoost ranking
    {'name': 'xgb_top_any', 'description': 'Top 3 by XGBoost, any direction',
     'direction_filter': 'any', 'tier_filter': 'SABC', 'stat_filter': 'any',
     'hr_threshold': (40, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'xgb_prob', 'mins_min': 40, 'combo_max': 1},
    {'name': 'xgb_top_under', 'description': 'Top 3 UNDERs by XGBoost',
     'direction_filter': 'under_only', 'tier_filter': 'SABC', 'stat_filter': 'any',
     'hr_threshold': (40, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'xgb_prob', 'mins_min': 40, 'combo_max': 1},
    {'name': 'xgb_top_base', 'description': 'Top 3 base stats by XGBoost',
     'direction_filter': 'any', 'tier_filter': 'SABC', 'stat_filter': 'base_only',
     'hr_threshold': (40, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'xgb_prob', 'mins_min': 40, 'combo_max': 0},
    {'name': 'xgb_top_nothot', 'description': 'Top 3 non-HOT by XGBoost',
     'direction_filter': 'any', 'tier_filter': 'SABC', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 0, 'streak_filter': 'not_hot',
     'sort_key': 'xgb_prob', 'mins_min': 40, 'combo_max': 1},

    # 23-25: Anti-HOT (HOT = 49.2% trap)
    {'name': 'anti_hot_xgb', 'description': 'Non-HOT streaks + XGBoost',
     'direction_filter': 'any', 'tier_filter': 'SAB', 'stat_filter': 'base_only',
     'hr_threshold': (60, 40), 'gap_min': 1.0, 'streak_filter': 'not_hot',
     'sort_key': 'xgb_prob', 'mins_min': 50, 'combo_max': 0},
    {'name': 'anti_hot_under', 'description': 'Non-HOT UNDERs + composite',
     'direction_filter': 'under_only', 'tier_filter': 'SAB', 'stat_filter': 'any',
     'hr_threshold': (50, 0), 'gap_min': 0, 'streak_filter': 'not_hot',
     'sort_key': 'composite_under', 'mins_min': 50, 'combo_max': 1},
    {'name': 'anti_hot_floor', 'description': 'Non-HOT + floor-sorted',
     'direction_filter': 'any', 'tier_filter': 'SAB', 'stat_filter': 'base_only',
     'hr_threshold': (70, 50), 'gap_min': 0, 'streak_filter': 'not_hot',
     'sort_key': 'floor', 'mins_min': 60, 'combo_max': 0},

    # 26-29: Backtest reference strategies (absorbed from backtest_pipelines.py)
    {'name': 'old_pipeline', 'description': 'Old Pipeline (S/A OVER + S UNDER, gap-weighted)',
     'direction_filter': 'any', 'tier_filter': 'SA', 'stat_filter': 'any',
     'hr_threshold': (60, 40), 'gap_min': 1.5, 'streak_filter': 'any',
     'sort_key': 'gap', 'mins_min': 50, 'combo_max': 1},
    {'name': 'hybrid', 'description': 'Hybrid (old filters + XGBoost ranking)',
     'direction_filter': 'any', 'tier_filter': 'SA', 'stat_filter': 'any',
     'hr_threshold': (60, 40), 'gap_min': 1.5, 'streak_filter': 'any',
     'sort_key': 'xgb_prob', 'mins_min': 50, 'combo_max': 1},
    {'name': 'floor_first', 'description': 'Floor-First (L10 HR >= 70%, no combos)',
     'direction_filter': 'any', 'tier_filter': 'SAB', 'stat_filter': 'base_only',
     'hr_threshold': (70, 50), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'floor', 'mins_min': 60, 'combo_max': 0},
    {'name': 'xgb_only', 'description': 'XGBoost-Only (pure ML ranking, minimal filters)',
     'direction_filter': 'any', 'tier_filter': 'SABC', 'stat_filter': 'any',
     'hr_threshold': (40, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'xgb_prob', 'mins_min': 0, 'combo_max': 1},

    # 30: Baseline UNDER
    {'name': 'under_pure', 'description': 'All UNDERs, minimal filters, XGBoost-sorted',
     'direction_filter': 'under_only', 'tier_filter': 'SABC', 'stat_filter': 'any',
     'hr_threshold': (40, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'xgb_prob', 'mins_min': 0, 'combo_max': 1},

    # 31-32: ML ensemble strategies (MLP + ensemble validation)
    {'name': 'mlp_top', 'description': 'Top 3 by MLP neural network probability',
     'direction_filter': 'any', 'tier_filter': 'SABC', 'stat_filter': 'any',
     'hr_threshold': (40, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'mlp_prob', 'mins_min': 40, 'combo_max': 1},
    {'name': 'ensemble_top', 'description': 'Top 3 by ensemble (XGB 60% + MLP 40%)',
     'direction_filter': 'any', 'tier_filter': 'SABC', 'stat_filter': 'any',
     'hr_threshold': (40, 0), 'gap_min': 0, 'streak_filter': 'any',
     'sort_key': 'ensemble_prob', 'mins_min': 40, 'combo_max': 1},
]


# ═══════════════════════════════════════════════════════════════
# 70 GRID-GENERATED STRATEGIES
# ═══════════════════════════════════════════════════════════════

def generate_strategy_grid(n=70, seed=42):
    """
    Generate n strategies via deterministic parameter grid sampling.
    Cartesian product of direction(5) x sort(6) x tier(3) x stat(3) x streak(3) = 810 combos.
    Deterministic shuffle, take first n not in curated set.
    """
    directions = ['any', 'under_only', 'over_only', 'under_heavy']
    sorts = ['xgb_prob', 'gap', 'hr_weighted', 'composite_under', 'floor', 'season_margin']
    tiers = ['SAB', 'SABC', 'any']
    stats = ['any', 'base_only', 'no_combo']
    streaks = ['any', 'not_hot', 'cold_only']

    # Pre-set HR thresholds and gap_min per tier
    tier_hr = {'SAB': (60, 40), 'SABC': (50, 0), 'any': (40, 0)}
    tier_gap = {'SAB': 1.0, 'SABC': 0, 'any': 0}
    tier_mins = {'SAB': 50, 'SABC': 40, 'any': 0}

    curated_names = {s['name'] for s in CURATED_STRATEGIES}
    seen_names = set()

    combos = []
    for d in directions:
        for s in sorts:
            for t in tiers:
                for st in stats:
                    for sk in streaks:
                        # Generate auto-name
                        d_short = {'any': 'any', 'under_only': 'und', 'over_only': 'ovr', 'under_heavy': 'uhv'}[d]
                        s_short = {'xgb_prob': 'xgb', 'gap': 'gap', 'hr_weighted': 'hr', 'composite_under': 'cmp', 'floor': 'flr', 'season_margin': 'mrg'}[s]
                        st_short = {'any': 'any', 'base_only': 'bas', 'no_combo': 'noc'}[st]
                        sk_short = {'any': 'any', 'not_hot': 'noh', 'cold_only': 'col'}[sk]
                        name = f'gen_{d_short}_{s_short}_{t}_{st_short}_{sk_short}'

                        if name in curated_names or name in seen_names:
                            continue
                        seen_names.add(name)

                        combos.append({
                            'name': name,
                            'description': f'Grid: dir={d} sort={s} tier={t} stat={st} streak={sk}',
                            'direction_filter': d,
                            'tier_filter': t,
                            'stat_filter': st,
                            'hr_threshold': tier_hr[t],
                            'gap_min': tier_gap[t],
                            'streak_filter': sk,
                            'sort_key': s,
                            'mins_min': tier_mins[t],
                            'combo_max': 0 if st == 'no_combo' else 1,
                        })

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(combos)

    return combos[:n]


# ═══════════════════════════════════════════════════════════════
# 100 SHADOW PARLAYS
# ═══════════════════════════════════════════════════════════════

def _random_unused_trio(pool, used_trios, sort_fn=None):
    """Pick any 3 valid props whose player trio hasn't been used."""
    if sort_fn is None:
        sort_fn = _primary_score
    eligible = [p for p in pool if _is_eligible(p)]
    sorted_pool = sorted(eligible, key=sort_fn, reverse=True)

    # Try greedy with different starting offsets
    for offset in range(min(len(sorted_pool), 50)):
        sub_pool = sorted_pool[offset:]
        picks = _greedy_select(sub_pool, 3, sort_fn, max_combo=3)
        if len(picks) >= 3:
            trio = frozenset(p['player'] for p in picks[:3])
            if trio not in used_trios:
                return picks[:3]

    # Last resort: random sample
    rng = random.Random(42)
    players = list({p['player']: p for p in eligible}.values())
    rng.shuffle(players)
    for i in range(len(players) - 2):
        trio = frozenset(p['player'] for p in players[i:i+3])
        if trio not in used_trios:
            return players[i:i+3]

    # Absolute fallback: return top 3 even if trio is used
    return sorted_pool[:3] if len(sorted_pool) >= 3 else sorted_pool


def _pick_exclude(trio, used_trios):
    """Pick the player from a trio that appears most in used trios to exclude."""
    counts = {}
    for player in trio:
        counts[player] = sum(1 for t in used_trios if player in t)
    return max(counts, key=counts.get)


def build_100_shadow_parlays(results):
    """
    Build 100 diverse 3-leg shadow parlays.
    30 curated + 70 grid-generated.
    Guaranteed: zero nulls, unique player trios.
    """
    pool = [p for p in results if _is_eligible(p)]

    if len(pool) < 3:
        return []

    all_strategies = CURATED_STRATEGIES + generate_strategy_grid(70, seed=42)
    used_trios = set()
    shadow_parlays = []

    for idx, strategy in enumerate(all_strategies):
        picks = build_parlay_from_params(strategy, pool)

        if len(picks) >= 3:
            trio = frozenset(p['player'] for p in picks[:3])

            # Ensure unique trio
            if trio in used_trios:
                # Retry excluding the most-used player from this trio
                exclude_player = _pick_exclude(trio, used_trios)
                picks = build_parlay_from_params(strategy, pool, excluded_players={exclude_player})
                if len(picks) >= 3:
                    trio = frozenset(p['player'] for p in picks[:3])

            if trio in used_trios:
                # Ultimate fallback: random unused trio
                picks = _random_unused_trio(pool, used_trios, _sort_fn(strategy.get('sort_key', 'composite_under')))
                if len(picks) >= 3:
                    trio = frozenset(p['player'] for p in picks[:3])

            if len(picks) >= 3:
                used_trios.add(trio)
                legs = [_make_leg(p) for p in picks[:3]]
                shadow_parlays.append({
                    'strategy_name': strategy['name'],
                    'strategy_id': f'engine_{strategy["name"]}',
                    'strategy_description': strategy['description'],
                    'legs': legs,
                    'confidence': round(sum(l.get('primary_score', 0) for l in legs) / 3, 2),
                    'legs_total': len(legs),
                    'result': None,
                    'legs_hit': None,
                })

    # ── Consensus tracking: count how many shadows pick each player-stat combo ──
    consensus = defaultdict(int)
    for sp in shadow_parlays:
        for leg in sp.get('legs', []):
            key = f"{leg.get('player', '')}|{leg.get('stat', '')}|{leg.get('direction', '')}"
            consensus[key] += 1

    # Add consensus_count to each leg
    for sp in shadow_parlays:
        for leg in sp.get('legs', []):
            key = f"{leg.get('player', '')}|{leg.get('stat', '')}|{leg.get('direction', '')}"
            leg['consensus_count'] = consensus.get(key, 0)

    # Report top consensus picks
    top_consensus = sorted(consensus.items(), key=lambda x: x[1], reverse=True)[:5]
    if top_consensus:
        print(f"  Top consensus picks: {', '.join(f'{k.split(chr(124))[0]} {k.split(chr(124))[1].upper()} {k.split(chr(124))[2]} ({v})' for k, v in top_consensus)}")

    print(f"  Parlay Engine: {len(shadow_parlays)} shadow parlays built ({len(used_trios)} unique trios)")
    return shadow_parlays

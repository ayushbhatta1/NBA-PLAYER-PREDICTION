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

    # HR regression trap penalty — validated: L10 HR>=70 = 43.2% (WORSE than random)
    # Sportsbooks adjust lines for hot streaks, making high HR anti-predictive
    hr_bonus = 0.0
    l10_hr = p.get('l10_hit_rate', 50) or 50
    if l10_hr >= 70:
        hr_bonus = -0.05  # regression trap
    elif l10_hr < 30:
        hr_bonus = -0.05  # too unreliable

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
        'sniper_score': round(_sniper_score(p), 2),
        'floor_score': round(_floor_score(p), 4),
        'composite_score': round(_composite_safe_score(p), 2),
        'reg_margin': p.get('reg_margin'),
        'sim_prob': p.get('sim_prob'),
        'flow_adj': p.get('flow_adj'),
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

def _sniper_score(p):
    """SNIPER scoring — backtested 79% parlay cash rate on 447K records.

    Key findings from 447K records across 464 days:
    - L5 trending down = 79.9% vs 48.2% trending up
    - + spread >= 10 = 89.1% HR
    - + B2B = 87.8% HR
    - + COLD streak = 87.4% HR
    - + std < 3 = 86.6% HR
    - + floor < line + std < 4 = 85.3% HR
    - Small lines (0-5) = 73.5% HR
    - Stars (avg > 25) blow up and kill parlays
    """
    s = 0.0
    stat = p.get('stat', '').lower()
    line = p.get('line', 0) or 0
    season_avg = p.get('season_avg', 0) or 0
    l10_avg = p.get('l10_avg', 0) or 0
    l5_avg = p.get('l5_avg', 0) or 0

    # #1: margin above season avg
    margin = line - season_avg
    s += min(margin * 2, 12)

    # #2: L10 confirmation
    l10_margin = line - l10_avg
    s += min(l10_margin * 1.5, 8)

    # #3: L5 trending down (79.9% base)
    if l5_avg and l10_avg and l5_avg < l10_avg:
        s += 3

    # #4: Stat reliability for UNDERs
    stat_bonus = {'blk': 6, 'stl': 5, '3pm': 5, 'ast': 4, 'reb': 2, 'pts': 0}
    s += stat_bonus.get(stat, -1)

    # #5: Gap bonus (capped)
    s += min(p.get('abs_gap', 0) or 0, 4)

    # #6: Blowout spread bonus (89.1% with spread >= 10!)
    spread = abs(p.get('spread', 0) or 0)
    if spread >= 10:
        s += 5
    elif spread >= 8:
        s += 4
    elif spread >= 5:
        s += 2

    # #7: SMALL LINE bonus
    if line <= 5:
        s += 5
    elif line <= 10:
        s += 3
    elif line <= 15:
        s += 1

    # #8: Role player bonus
    if season_avg < 5:
        s += 4
    elif season_avg < 10:
        s += 2
    elif season_avg < 15:
        s += 1
    elif season_avg > 25:
        s -= 2  # star blowup risk

    # #9: Margin% bonus
    if line > 0:
        mpct = margin / line
        if mpct >= 0.30:
            s += 4
        elif mpct >= 0.20:
            s += 3
        elif mpct >= 0.15:
            s += 1

    # #10: NEW — COLD streak (87.4% HR with L5↓)
    streak = p.get('streak_status', '')
    if streak == 'COLD':
        s += 4
    elif streak == 'HOT':
        s -= 2  # hot players more likely to stay over

    # #11: NEW — B2B fatigue (87.8% HR)
    if p.get('is_b2b'):
        s += 4

    # #12: NEW — Low L10 std = consistent player = predictable UNDER (86.6% with std<3)
    l10_std = p.get('l10_std')
    if l10_std is not None:
        if l10_std < 3:
            s += 4
        elif l10_std < 4:
            s += 2
        elif l10_std < 5:
            s += 1

    # #13: NEW — L10 floor below line (80.0% HR)
    l10_floor = p.get('l10_floor')
    if l10_floor is not None and line > 0 and l10_floor < line:
        s += 2

    # #14: NEW — High miss count = player consistently goes under (87.2% with miss>=8)
    miss_count = p.get('l10_miss_count', 0) or 0
    if miss_count >= 8:
        s += 4
    elif miss_count >= 7:
        s += 3
    elif miss_count >= 6:
        s += 1

    # ═══ V3: ML & MODEL SIGNALS ═══

    # #15: Regression margin (|margin| >= 3 hits at 95%+)
    reg_margin = p.get('reg_margin')
    if reg_margin is not None and reg_margin < -3:
        s += 5  # regression model strongly predicts UNDER
    elif reg_margin is not None and reg_margin < -1.5:
        s += 3

    # #16: Monte Carlo simulation confirmation
    sim_prob = p.get('sim_prob')
    if sim_prob is not None:
        if sim_prob > 0.70:
            s += 4  # 5000 sims confirm UNDER
        elif sim_prob > 0.60:
            s += 2

    # #17: ML VETO — ensemble_prob < 0.45 means ML strongly disagrees
    ensemble_prob = p.get('ensemble_prob', p.get('xgb_prob'))
    if ensemble_prob is not None and ensemble_prob < 0.45:
        return -100  # hard reject — ML veto gate

    # #18: Game total signal (low-scoring game = UNDER friendly)
    game_total = p.get('game_total_signal', 0) or 0
    if game_total < -5:
        s += 3  # low game total
    elif game_total < -2:
        s += 1

    # #19: Game flow confidence
    flow_adj = p.get('flow_adj', 0) or 0
    if flow_adj < -1.5:
        s += 3  # game script predicts reduced production

    # #20: Opponent matchup delta
    opp_delta = p.get('opp_matchup_delta', 0) or 0
    if opp_delta < -1:
        s += 2  # player underperforms vs this specific opponent

    # #21: Travel fatigue
    travel_7d = p.get('travel_miles_7day', 0) or 0
    if travel_7d > 4000:
        s += 2
    tz_shifts = p.get('tz_shifts_7day', 0) or 0
    if tz_shifts >= 2:
        s += 1

    # #22: Low usage role player (UNDER friendly)
    usage_rate = p.get('usage_rate', 0) or 0
    if 0 < usage_rate < 0.15:
        s += 2

    # #23: L10 median vs line (books price at median)
    l10_median = p.get('l10_median')
    if l10_median is not None and line > 0 and l10_median < line - 0.5:
        s += 2

    # #24: Model consensus (multiple models agree on UNDER)
    model_votes = 0
    if ensemble_prob is not None and ensemble_prob > 0.55:
        model_votes += 1
    if sim_prob is not None and sim_prob > 0.55:
        model_votes += 1
    if reg_margin is not None and reg_margin < -1:
        model_votes += 1
    arena_prob = p.get('arena_prob')
    if arena_prob is not None and arena_prob > 0.55:
        model_votes += 1
    if model_votes >= 3:
        s += 3
    elif model_votes >= 2:
        s += 1

    # ═══ V3: DERIVED FEATURES ═══

    # #25: Ceiling-Line Gap (if best L10 barely reaches line, UNDER is safe)
    l10_values = p.get('l10_values', [])
    if l10_values and line > 0:
        l10_ceiling = max(l10_values) if l10_values else 0
        ceiling_gap = l10_ceiling - line
        if ceiling_gap < 2:
            s += 5  # best game barely reaches line — near lock
        elif ceiling_gap < 4:
            s += 2

    # #26: Model disagreement penalty
    probs = [v for v in [
        p.get('xgb_prob'), p.get('mlp_prob'), sim_prob, arena_prob
    ] if v is not None]
    if len(probs) >= 2:
        disagreement = max(probs) - min(probs)
        if disagreement > 0.25:
            s -= 4  # models disagree strongly — uncertain pick

    # #27: L10 CV (coefficient of variation — low = predictable = safer UNDER)
    l10_cv = p.get('l10_cv')
    if l10_cv is not None:
        if l10_cv < 0.15:
            s += 3  # very predictable
        elif l10_cv < 0.25:
            s += 1

    # #28: Blacklist check
    player = p.get('player', '')
    if player in BLACKLISTED_PLAYERS:
        return -100  # hard reject

    return s


def _composite_safe_score(p):
    """Composite: 60% SNIPER heuristics + 40% floor safety (normalized).
    Floor safety catches hidden downside that heuristics miss."""
    sniper = _sniper_score(p)
    if sniper <= -100:
        return -100  # propagate hard rejects
    floor = _floor_score(p)
    # Normalize: sniper scores range ~0-50, floor ~0-1.5
    return sniper * 0.6 + floor * 40 * 0.4


def should_play_today(pool):
    """Day-level classifier: should we build a parlay today or SKIP?

    From 1M simulation on 445K records:
    - min_hr>=60 + line_above>=1 = 50.8% cash rate (20K sample)
    - Need at least 5 qualifying UNDER props across 3+ games for edge
    - When forced to pick from weak pool (Pass 4+), we lose money

    Returns (should_play: bool, reason: str, qualifying_count: int, game_count: int)
    """
    qualifying = []
    strong_qualifying = []  # Pass 0 validated filter (line_above>=2)
    games = set()
    for p in pool:
        if not _is_eligible(p):
            continue
        if p.get('direction', '').upper() != 'UNDER':
            continue
        hr = p.get('l10_hit_rate', 0) or 0
        line = p.get('line', 0) or 0
        avg = p.get('season_avg', 0) or 0
        line_above = line - avg
        is_hot = p.get('streak_status') == 'HOT'
        if hr >= 60 and line_above >= 0.5 and not is_hot:
            qualifying.append(p)
            g = p.get('game', '')
            if g:
                games.add(g)
            # Strong = line > L10 avg by 2+ (validated 71.6% HR)
            l10_avg = p.get('l10_avg', 0) or 0
            line_vs_l10 = line - l10_avg
            if line_vs_l10 >= 2.0:
                strong_qualifying.append(p)

    n_qual = len(qualifying)
    n_games = len(games)

    n_strong = len(strong_qualifying)

    # Strong play: 5+ qualifying props across 3+ games
    if n_qual >= 5 and n_games >= 3:
        return True, f"PLAY — {n_qual} qualifying ({n_strong} strong) across {n_games} games", n_qual, n_games

    # Strong validated: 3+ strong (line_above>=2) props — validated 82.9% HR
    if n_strong >= 3 and n_games >= 2:
        return True, f"PLAY (VALIDATED) — {n_strong} strong props (line_above>=2) across {n_games} games", n_qual, n_games

    # Marginal play: 3-4 qualifying but still diverse
    if n_qual >= 3 and n_games >= 3:
        return True, f"MARGINAL PLAY — {n_qual} props across {n_games} games (lower confidence)", n_qual, n_games

    # Skip: not enough edge
    return False, f"NO PLAY — only {n_qual} qualifying ({n_strong} strong) across {n_games} games (need 5+ across 3+)", n_qual, n_games


def _sim_sort(p):
    """Sort key for SAFE parlay leg selection.

    Validated signals (46K real-line records, no data leakage):
      line > L10 by 5+:  76.0% HR (1464 picks)
      line > L10 by 3+ + COLD: 72.5% HR (726 picks)
      COLD+UNDER gap3:   63.3% HR (1155 picks)
      L10 HR >= 70:      43.2% HR (ANTI-PREDICTIVE — penalize, don't reward)
    """
    s = 0.0
    line = p.get('line', 0) or 0
    l10 = p.get('l10_avg', 0) or 0
    avg = p.get('season_avg', 0) or 0

    # #1 (DOMINANT): Line vs L10 avg — validated strongest clean signal
    # 76.0% at line>L10+5, 72.5% at line>L10+3+COLD, 71.6% at line>L10+2+COLD
    line_vs_l10 = line - l10 if l10 > 0 else line - avg
    s += min(line_vs_l10 * 3, 21)  # gap of 7 = +21 (dominant signal)

    # #2: COLD streak — validated 63.3% COLD+UNDER gap3 vs 57% baseline UNDER
    streak = p.get('streak_status', 'NEUTRAL')
    if streak == 'COLD':
        s += 10
    elif streak == 'HOT':
        s -= 15  # HOT = 49.2% trap on real data

    # #3: L5 trending down — triple confirmation with line gap + COLD
    l5 = p.get('l5_avg', 0) or 0
    if l5 > 0 and l10 > 0 and l5 < l10:
        s += 5

    # #4: Regression margin — when regression model also confirms UNDER
    reg_margin = p.get('reg_margin', 0) or 0
    if reg_margin < -3:
        s += 8   # strong regression confirmation
    elif reg_margin < -1.5:
        s += 4

    # #5: Multi-model consensus — independent models agreeing
    votes = 0
    ep = p.get('ensemble_prob', p.get('xgb_prob'))
    if ep is not None and ep > 0.55:
        votes += 1
    sp = p.get('sim_prob')
    if sp is not None and sp > 0.55:
        votes += 1
    if reg_margin < -1.0:
        votes += 1
    fp = p.get('focused_prob')
    if fp is not None and fp > 0.55:
        votes += 1
    s += votes * 3  # up to +12 for 4/4 consensus

    # #6: HR regression trap — validated anti-predictive
    l10_hr = p.get('l10_hit_rate', 50) or 50
    if l10_hr >= 70:
        s -= 5  # books already priced in the streak
    elif l10_hr < 30:
        s -= 3  # too unreliable

    # #7: Low std dev (more predictable)
    l10_std = p.get('l10_std', 0) or 0
    if l10_std > 0 and l10_std <= 3:
        s += 2
    elif l10_std > 6:
        s -= 2

    # #8: Stat type bonus
    stat = p.get('stat', '').lower()
    if stat in ('blk', 'stl', 'stl_blk'):
        s += 4
    elif stat in ('pra', 'pr', 'pa', 'ra'):
        s -= 1  # combos less reliable

    # #9: Away preference (small consistent edge)
    if not p.get('is_home'):
        s += 1

    return s


def build_primary_safe(pool):
    """
    3-Leg SAFE parlay: Rebuilt from 1M parlay simulation on 777K valid parlays.

    1M SIMULATION FINDINGS (what separates 27.9% winners from losers):
    #1 Line above season avg: >=2.0 = 42.8%, >=3.0 = 45.2%, >=4.0 = 55.6%
    #2 No stars: max_avg<=5 = 43.5%, <=10 = 38.0%, <=15 = 33.6%
    #3 Small lines: max_line<=5 = 35.1%
    #4 Low minutes players more predictable (43.9% vs 49.8%)
    #5 Miss count >= 7 KILLS (14.7% cash)
    #6 Stat diversity: BLK+REB+STL = 63.7% vs PTS+PTS+PTS = 15.8%
    #7 Away players preferred (0.96 home vs 1.31 in losers)

    HARMFUL (confirmed by simulation):
    - L5<L10: winners 1.39 vs losers 1.51 (losers use it MORE)
    - XGBoost: winners 0.530 vs losers 0.543 (model favors losers!)
    - High miss count: >=7 = 14.7%, >=8 = 7.8%
    """
    used_games = set()
    picks = []

    def _is_hot(p):
        return p.get('streak_status') == 'HOT'

    # ALL passes: UNDER + L10 HR >= 60% + NOT HOT (HOT = trap, 0 HOT = 34.6% vs 14.4%)
    base_filter = lambda p: (
        _is_eligible(p) and
        p.get('direction', '').upper() == 'UNDER' and
        (p.get('l10_hit_rate', 0) or 0) >= 60 and
        not _is_hot(p)
    )

    def _pick_from(candidates, picks, used_games, n_target):
        for p in candidates:
            g = p.get('game', '')
            if g and g in used_games:
                continue
            picks.append(p)
            if g:
                used_games.add(g)
            if len(picks) >= n_target:
                return True
        return len(picks) >= n_target

    # Pass 0A (STRONGEST): line > L10 avg by 3+ AND COLD
    # Validated on 46K real-line records: 72.5% HR (726 picks, no data leakage)
    # Logic: book set line 3+ above recent production + player in cold streak
    p0a = [p for p in pool if (
        _is_eligible(p) and
        p.get('direction', '').upper() == 'UNDER' and
        not _is_hot(p) and
        p.get('streak_status') == 'COLD' and
        ((p.get('line', 0) or 0) - (p.get('l10_avg', 0) or 0)) >= 3.0
    )]
    p0a.sort(key=_sim_sort, reverse=True)
    if _pick_from(p0a, picks, used_games, 3):
        return picks

    # Pass 0B: line > L10 avg by 2+ AND COLD + L5 declining
    # 70.8% HR on 987 picks — triple confirmation (line gap + cold + trend)
    p0b = [p for p in pool if (
        _is_eligible(p) and
        p.get('direction', '').upper() == 'UNDER' and
        not _is_hot(p) and
        p.get('streak_status') == 'COLD' and
        ((p.get('line', 0) or 0) - (p.get('l10_avg', 0) or 0)) >= 2.0 and
        (p.get('l5_avg', 0) or 0) > 0 and (p.get('l10_avg', 0) or 0) > 0 and
        (p.get('l5_avg', 0) or 0) < (p.get('l10_avg', 0) or 0)
    )]
    p0b.sort(key=_sim_sort, reverse=True)
    if _pick_from(p0b, picks, used_games, 3):
        return picks

    # Pass 0C: line > L10 avg by 2+ AND COLD (relax L5 declining)
    # 71.6% HR on 1211 picks
    p0c = [p for p in pool if (
        _is_eligible(p) and
        p.get('direction', '').upper() == 'UNDER' and
        not _is_hot(p) and
        p.get('streak_status') == 'COLD' and
        ((p.get('line', 0) or 0) - (p.get('l10_avg', 0) or 0)) >= 2.0 and
        p not in picks
    )]
    p0c.sort(key=_sim_sort, reverse=True)
    if _pick_from(p0c, picks, used_games, 3):
        return picks

    # Pass 1: COLD + L5<L10 + line above avg >=1 (targets 53-62% cash zone)
    p1 = [p for p in pool if (
        base_filter(p) and
        p.get('streak_status') == 'COLD' and
        (p.get('l5_avg', 0) or 0) > 0 and (p.get('l10_avg', 0) or 0) > 0 and
        (p.get('l5_avg', 0) or 0) < (p.get('l10_avg', 0) or 0) and
        ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 1.0
    )]
    p1.sort(key=_sim_sort, reverse=True)
    if _pick_from(p1, picks, used_games, 3):
        return picks

    # Pass 2: COLD + line above avg >=0.5 (relax L5<L10)
    p2 = [p for p in pool if (
        base_filter(p) and
        p.get('streak_status') == 'COLD' and
        ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 0.5 and
        p not in picks
    )]
    p2.sort(key=_sim_sort, reverse=True)
    if _pick_from(p2, picks, used_games, 3):
        return picks

    # Pass 3: L5<L10 + line above avg >=1 + no HOT (50.8% zone)
    p3 = [p for p in pool if (
        base_filter(p) and
        (p.get('l5_avg', 0) or 0) > 0 and (p.get('l10_avg', 0) or 0) > 0 and
        (p.get('l5_avg', 0) or 0) < (p.get('l10_avg', 0) or 0) and
        ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 1.0 and
        p not in picks
    )]
    p3.sort(key=_sim_sort, reverse=True)
    if _pick_from(p3, picks, used_games, 3):
        return picks

    # Pass 4: Any UNDER with L10 HR >= 60% + line above avg (drop diversity)
    p4 = [p for p in pool if (
        base_filter(p) and
        ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 0.5 and
        p not in picks
    )]
    p4.sort(key=_sim_sort, reverse=True)
    for p in p4:
        g = p.get('game', '')
        if g and g in used_games:
            continue
        picks.append(p)
        if g:
            used_games.add(g)
        if len(picks) >= 3:
            break

    if len(picks) >= 3:
        return picks

    # Pass 5: Any UNDER with L10 HR >= 55% (survival — avoid DNP)
    p5 = [p for p in pool if (
        _is_eligible(p) and
        p.get('direction', '').upper() == 'UNDER' and
        (p.get('l10_hit_rate', 0) or 0) >= 55 and
        p not in picks
    )]
    p5.sort(key=_sim_sort, reverse=True)
    for p in p5:
        g = p.get('game', '')
        if g and g in used_games:
            continue
        picks.append(p)
        if g:
            used_games.add(g)
        if len(picks) >= 3:
            break

    if len(picks) >= 3:
        return picks

    # Pass 6: Any UNDER (absolute survival)
    p6 = [p for p in pool if (
        _is_eligible(p) and
        p.get('direction', '').upper() == 'UNDER' and
        p not in picks
    )]
    p6.sort(key=_sim_sort, reverse=True)
    for p in p6:
        g = p.get('game', '')
        if g and g in used_games:
            continue
        picks.append(p)
        if g:
            used_games.add(g)
        if len(picks) >= 3:
            break

    return picks


def build_2leg_safe(pool):
    """
    2-Leg SAFE parlay: Higher cash rate (64.3%), lower payout (3x).
    From 1M sim: min_hr>=60 + line_above>=1 + not HOT → 64.3% 2-leg cash rate.
    EV = 1.93 per $1 (vs 3.19 for 3-leg). Use when day is marginal.
    Only picks from the strongest COLD+L5↓ players.
    """
    used_games = set()
    picks = []

    def _is_hot(p):
        return p.get('streak_status') == 'HOT'

    base_filter = lambda p: (
        _is_eligible(p) and
        p.get('direction', '').upper() == 'UNDER' and
        (p.get('l10_hit_rate', 0) or 0) >= 60 and
        not _is_hot(p)
    )

    def _pick_from(candidates, picks, used_games, n_target):
        for p in candidates:
            g = p.get('game', '')
            if g and g in used_games:
                continue
            picks.append(p)
            if g:
                used_games.add(g)
            if len(picks) >= n_target:
                return True
        return len(picks) >= n_target

    # Pass 0: line > L10 avg by 2+ AND COLD (72.5% HR validated on 46K real-line records)
    p0 = [p for p in pool if (
        _is_eligible(p) and
        p.get('direction', '').upper() == 'UNDER' and
        not _is_hot(p) and
        p.get('streak_status') == 'COLD' and
        ((p.get('line', 0) or 0) - (p.get('l10_avg', 0) or 0)) >= 2.0
    )]
    p0.sort(key=_sim_sort, reverse=True)
    if _pick_from(p0, picks, used_games, 2):
        return picks

    # Pass 1: COLD + line above season avg >=2 (fallback)
    p1 = [p for p in pool if (
        base_filter(p) and
        p.get('streak_status') == 'COLD' and
        ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 2.0 and
        p not in picks
    )]
    p1.sort(key=_sim_sort, reverse=True)
    if _pick_from(p1, picks, used_games, 2):
        return picks

    # Pass 2: COLD + line_above>=1
    p2 = [p for p in pool if (
        base_filter(p) and
        p.get('streak_status') == 'COLD' and
        ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 1.0 and
        p not in picks
    )]
    p2.sort(key=_sim_sort, reverse=True)
    if _pick_from(p2, picks, used_games, 2):
        return picks

    # Pass 3: Any qualifying UNDER with line_above>=1
    p3 = [p for p in pool if (
        base_filter(p) and
        ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 1.0 and
        p not in picks
    )]
    p3.sort(key=_sim_sort, reverse=True)
    if _pick_from(p3, picks, used_games, 2):
        return picks

    # Pass 4: Survival — any qualifying UNDER
    p4 = [p for p in pool if base_filter(p) and p not in picks]
    p4.sort(key=_sim_sort, reverse=True)
    _pick_from(p4, picks, used_games, 2)

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


def build_triple_safe(results):
    """Triple-SAFE: 3 independent 3-leg parlays with NO player overlap.

    At 78% per-parlay cash rate: P(at least 1 cashes) = 1 - (0.22)^3 = 98.9%

    SAFE #1: Pure SNIPER scoring (proven heuristic baseline)
    SAFE #2: Composite scoring (floor-weighted, catches downside)
    SAFE #3: Correlation-optimized (max independence between legs)
    """
    pool = [p for p in results if _is_eligible(p)]
    used_players = set()
    triple = {}

    # SAFE #1: Pure SNIPER
    safe1_picks = build_primary_safe(pool)
    safe1_legs = [_make_leg(p) for p in safe1_picks]
    used_players.update(p['player'] for p in safe1_picks)
    triple['safe_1_sniper'] = {
        'name': 'SAFE #1 — SNIPER',
        'method': 'sniper_v3_pure',
        'legs': safe1_legs,
        'legs_total': len(safe1_legs),
        'under_count': sum(1 for l in safe1_legs if l.get('direction', '').upper() == 'UNDER'),
        'kelly_fraction': _kelly_fraction(safe1_legs),
        'description': 'Pure SNIPER scoring — proven 79% backtest baseline',
    }

    # SAFE #2: Composite (floor-weighted) — exclude SAFE #1 players
    pool2 = [p for p in pool if p['player'] not in used_players]

    def _l5_trending_down(p):
        l5 = p.get('l5_avg', 0) or 0
        l10 = p.get('l10_avg', 0) or 0
        return l5 > 0 and l10 > 0 and l5 < l10

    # Try strict first, then relax
    safe2_picks = []
    for pass_filters in [
        lambda p: (p.get('direction', '').upper() == 'UNDER' and
                   _l5_trending_down(p) and
                   (p.get('line', 0) or 0) <= 20 and
                   ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 0.5),
        lambda p: (p.get('direction', '').upper() == 'UNDER' and
                   ((p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0)) >= 0.5),
        lambda p: p.get('direction', '').upper() == 'UNDER',
    ]:
        candidates = [p for p in pool2 if pass_filters(p) and p['player'] not in used_players]
        candidates.sort(key=_composite_safe_score, reverse=True)
        used_games = set()
        for p in candidates:
            if _floor_score(p) < 0.3:
                continue
            g = p.get('game', '')
            if g and g in used_games:
                continue
            safe2_picks.append(p)
            if g:
                used_games.add(g)
            if len(safe2_picks) >= 3:
                break
        if len(safe2_picks) >= 3:
            break

    safe2_legs = [_make_leg(p) for p in safe2_picks]
    used_players.update(p['player'] for p in safe2_picks)
    triple['safe_2_floor'] = {
        'name': 'SAFE #2 — FLOOR',
        'method': 'sniper_v3_floor',
        'legs': safe2_legs,
        'legs_total': len(safe2_legs),
        'under_count': sum(1 for l in safe2_legs if l.get('direction', '').upper() == 'UNDER'),
        'kelly_fraction': _kelly_fraction(safe2_legs),
        'description': 'Floor-weighted composite — catches hidden downside',
    }

    # SAFE #3: Correlation-optimized — exclude SAFE #1 and #2 players
    pool3 = [p for p in pool if p['player'] not in used_players]
    try:
        from parlay_optimizer import build_optimal_parlay
        corr_result = build_optimal_parlay(pool3, n_legs=3, mode='safe')
        if corr_result and corr_result.get('legs') and len(corr_result['legs']) >= 3:
            safe3_legs = corr_result['legs']
        else:
            raise ValueError("Not enough corr legs")
    except Exception:
        # Fallback: sort by _composite_safe_score with max game diversity
        candidates = [p for p in pool3 if p.get('direction', '').upper() == 'UNDER']
        candidates.sort(key=_composite_safe_score, reverse=True)
        safe3_picks = []
        used_games = set()
        for p in candidates:
            g = p.get('game', '')
            if g and g in used_games:
                continue
            safe3_picks.append(p)
            if g:
                used_games.add(g)
            if len(safe3_picks) >= 3:
                break
        safe3_legs = [_make_leg(p) for p in safe3_picks]

    triple['safe_3_corr'] = {
        'name': 'SAFE #3 — CORRELATION',
        'method': 'sniper_v3_corr',
        'legs': safe3_legs,
        'legs_total': len(safe3_legs),
        'under_count': sum(1 for l in safe3_legs if l.get('direction', '').upper() == 'UNDER'),
        'kelly_fraction': _kelly_fraction(safe3_legs) if safe3_legs else 0.0,
        'description': 'Correlation-optimized — max independence between legs',
    }

    return triple


def build_primary_parlays(results):
    """
    Main entry: build 2-leg + 3-leg SAFE + AGGRESSIVE.
    Includes play/skip day classifier from 1M sim on 445K records.

    Returns dict with parlays + day_signal metadata.
    2-leg: 64.3% cash × 3x = 1.93 EV (consistency play)
    3-leg: 53.2% cash × 6x = 3.19 EV (max EV play)
    """
    pool = [p for p in results if _is_eligible(p)]

    # Day classifier
    play, play_reason, n_qual, n_games = should_play_today(pool)

    # Always build both — let user decide based on signal
    safe_picks_3 = build_primary_safe(pool)
    safe_picks_2 = build_2leg_safe(pool)
    safe_players = [p['player'] for p in safe_picks_3] + [p['player'] for p in safe_picks_2]
    agg_picks = build_primary_aggressive(pool, safe_players)

    safe_legs_3 = [_make_leg(p) for p in safe_picks_3]
    safe_legs_2 = [_make_leg(p) for p in safe_picks_2]
    agg_legs = [_make_leg(p) for p in agg_picks]

    under_count_safe_3 = sum(1 for l in safe_legs_3 if l.get('direction', '').upper() == 'UNDER')
    under_count_safe_2 = sum(1 for l in safe_legs_2 if l.get('direction', '').upper() == 'UNDER')
    under_count_agg = sum(1 for l in agg_legs if l.get('direction', '').upper() == 'UNDER')

    safe_kelly_3 = _kelly_fraction(safe_legs_3)
    safe_kelly_2 = _kelly_fraction(safe_legs_2)
    agg_kelly = _kelly_fraction(agg_legs)

    result = {
        'day_signal': {
            'should_play': play,
            'reason': play_reason,
            'qualifying_props': n_qual,
            'qualifying_games': n_games,
        },
        'safe_2leg': {
            'name': 'SAFE 2-LEG (CONSISTENCY)',
            'method': 'sim_2leg',
            'legs': safe_legs_2,
            'legs_total': len(safe_legs_2),
            'under_count': under_count_safe_2,
            'kelly_fraction': safe_kelly_2,
            'suggested_units': round(safe_kelly_2 * 100, 1),
            'ev_multiplier': 3.0,
            'sim_cash_rate': 64.3,
            'description': f'2-leg SAFE: 64.3% cash × 3x = 1.93 EV. {under_count_safe_2} UNDERs. Suggested: {safe_kelly_2*100:.1f}% bankroll.',
        },
        'safe': {
            'name': 'SAFE 3-LEG (MAX EV)',
            'method': 'sniper_v3',
            'legs': safe_legs_3,
            'legs_total': len(safe_legs_3),
            'under_count': under_count_safe_3,
            'kelly_fraction': safe_kelly_3,
            'suggested_units': round(safe_kelly_3 * 100, 1),
            'ev_multiplier': 6.0,
            'sim_cash_rate': 53.2,
            'description': f'3-leg SAFE: 53.2% cash × 6x = 3.19 EV. {under_count_safe_3} UNDERs. Suggested: {safe_kelly_3*100:.1f}% bankroll.',
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

    return result


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

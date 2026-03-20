#!/usr/bin/env python3
"""
EV Optimizer — Expected Value Parlay Builder

Philosophy shift: optimize for bankroll GROWTH (EV), not hit rate (accuracy).

A 45% prop at +200 is a BETTER bet than a 65% prop at -150:
  EV(+200) = 0.45 * 2.00 - 0.55 * 1.00 = +0.35
  EV(-150) = 0.65 * 0.67 - 0.35 * 1.00 = +0.09

Core functions:
  1. compute_ev            — single prop EV per unit staked
  2. compute_parlay_ev     — parlay EV accounting for correlation
  3. compute_implied_prob  — decimal odds to implied probability
  4. find_book_edge        — our prob vs book implied = real edge
  5. find_positive_ev_props— filter + rank all props by EV
  6. build_ev_parlay       — build parlay maximizing EV, not accuracy
  7. enrich_with_ev        — pipeline integration (add EV fields to all props)
  8. print_ev_report       — formatted summary
  9. CLI                   — --board, --parlay commands

Uses multiplier field from parse_board.py when available.
Falls back to 1.91 (-110 standard) when no multiplier present.
"""

import json
import math
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
COMBO_STATS = {'pra', 'pr', 'pa', 'ra', 'stl_blk'}
DEFAULT_ODDS = 1.91        # Standard -110 line (decimal)
STANDARD_VIG = 0.0476      # ~4.76% vig on -110/-110 market (1 - 2/2.09)
MIN_EDGE_PCT = 0.02        # 2% minimum edge to consider positive EV
KELLY_FRACTION = 0.25      # Quarter-Kelly for parlay sizing (conservative)
MAX_KELLY_CAP = 0.05       # Cap at 5% of bankroll per bet


# ---------------------------------------------------------------------------
# 0. Multiplier Recovery
# ---------------------------------------------------------------------------

def recover_multipliers(results, parsed_board_path=None):
    """
    Recover multiplier (odds) data from the parsed board and attach to results.

    The pipeline flow is: parse_board.py (has multiplier) -> analyze_v3.py
    (drops multiplier) -> full_board.json (no multiplier). This function
    re-attaches multiplier data by matching player+stat+line.

    If no parsed board path given, attempts to auto-locate it from the same
    date folder as the results.

    Args:
        results: list of prop dicts (full_board output)
        parsed_board_path: path to the parsed_board.json with multiplier data

    Returns:
        count of props that got multiplier data attached
    """
    if not parsed_board_path:
        return 0

    if not os.path.exists(parsed_board_path):
        return 0

    try:
        with open(parsed_board_path) as f:
            board = json.load(f)
    except (json.JSONDecodeError, IOError):
        return 0

    if not isinstance(board, list):
        return 0

    # Build lookup: (player_lower, stat_lower, line) -> multiplier
    mult_lookup = {}
    for entry in board:
        player = entry.get('player', '').strip().lower()
        stat = entry.get('stat', '').strip().lower()
        line = entry.get('line')
        mult = entry.get('multiplier')
        if player and stat and line is not None and mult and mult > 1.0:
            mult_lookup[(player, stat, line)] = mult

    attached = 0
    for prop in results:
        if prop.get('multiplier'):
            continue  # Already has it
        player = prop.get('player', '').strip().lower()
        stat = prop.get('stat', '').strip().lower()
        line = prop.get('line')
        mult = mult_lookup.get((player, stat, line))
        if mult:
            prop['multiplier'] = mult
            attached += 1

    return attached


def _find_parsed_board(date_dir):
    """Locate the parsed board JSON in a date directory."""
    if not os.path.isdir(date_dir):
        return None
    for fname in os.listdir(date_dir):
        if 'parsed_board' in fname and fname.endswith('.json'):
            return os.path.join(date_dir, fname)
    # Fallback: raw board txt that might have been converted
    for fname in os.listdir(date_dir):
        if 'raw_board' in fname and fname.endswith('.json'):
            return os.path.join(date_dir, fname)
    return None


# ---------------------------------------------------------------------------
# 1. Basic EV Calculation
# ---------------------------------------------------------------------------

def compute_ev(prob, odds):
    """
    Expected value per unit staked.

    Args:
        prob: our estimated probability of hitting (0.0 to 1.0)
        odds: decimal odds (e.g., 1.87 for standard prop, 3.00 for +200)

    Returns:
        EV per unit staked. Positive = edge over the book.

    Example:
        compute_ev(0.60, 1.91) = 0.60 * 0.91 - 0.40 = +0.146  (14.6% edge)
        compute_ev(0.45, 3.00) = 0.45 * 2.00 - 0.55 = +0.35   (35% edge)
    """
    prob = max(0.001, min(0.999, prob))
    odds = max(1.01, odds)
    return prob * (odds - 1) - (1 - prob)


# ---------------------------------------------------------------------------
# 2. Parlay EV Calculation
# ---------------------------------------------------------------------------

def compute_parlay_ev(legs, corr_adjustment=1.0):
    """
    Compute expected value for a parlay.

    Each leg needs: {'prob': float, 'odds': float}
    Parlay probability = product of individual probs * correlation adjustment.
    Parlay odds = product of individual odds.

    Args:
        legs: list of dicts with 'prob' and 'odds' keys
        corr_adjustment: multiplier for correlation (< 1.0 for positive
                         correlation penalty, > 1.0 for negative correlation
                         diversification benefit). From parlay_optimizer.

    Returns:
        dict with parlay_ev, parlay_prob, parlay_odds, leg_count, ev_per_unit
    """
    if not legs:
        return {'parlay_ev': 0.0, 'parlay_prob': 0.0, 'parlay_odds': 1.0,
                'leg_count': 0, 'ev_per_unit': 0.0}

    parlay_prob = 1.0
    parlay_odds = 1.0
    for leg in legs:
        p = max(0.001, min(0.999, leg.get('prob', 0.50)))
        o = max(1.01, leg.get('odds', DEFAULT_ODDS))
        parlay_prob *= p
        parlay_odds *= o

    # Apply correlation adjustment to probability
    parlay_prob *= max(0.50, min(1.50, corr_adjustment))
    parlay_prob = max(0.0001, min(0.99, parlay_prob))

    # Parlay EV: prob * (odds - 1) - (1 - prob)
    # Equivalent to: prob * odds - 1 (net of stake)
    ev = parlay_prob * (parlay_odds - 1) - (1 - parlay_prob)

    return {
        'parlay_ev': round(ev, 4),
        'parlay_prob': round(parlay_prob, 6),
        'parlay_odds': round(parlay_odds, 2),
        'parlay_payout': round(parlay_odds, 2),
        'leg_count': len(legs),
        'ev_per_unit': round(ev, 4),
        'is_positive_ev': ev > 0,
    }


# ---------------------------------------------------------------------------
# 3. Implied Probability from Odds
# ---------------------------------------------------------------------------

def compute_implied_prob(odds):
    """
    Convert decimal odds to implied probability.

    Args:
        odds: decimal odds (e.g., 1.91 for -110)

    Returns:
        implied probability (includes vig)

    Standard -110/-110 market:
        implied = 1/1.91 = 52.36% each side = 104.7% total (4.7% vig)
        Fair prob = 52.36% / 104.7% = 50.0%
    """
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def compute_fair_prob(over_odds, under_odds=None):
    """
    Remove vig to get fair probability.

    If both sides available: fair = implied / (implied_over + implied_under)
    If only one side: approximate by assuming standard vig (~4.76%)

    Args:
        over_odds: decimal odds for the OVER
        under_odds: decimal odds for the UNDER (optional)

    Returns:
        fair probability for the OVER side (vig-free)
    """
    implied_over = compute_implied_prob(over_odds)

    if under_odds and under_odds > 1.0:
        implied_under = compute_implied_prob(under_odds)
        total_implied = implied_over + implied_under
        if total_implied > 0:
            return implied_over / total_implied
    else:
        # No counter-side: remove standard vig approximation
        # Standard vig inflates implied prob by ~2.4% per side
        fair = implied_over / (1.0 + STANDARD_VIG)
        return min(fair, 0.99)

    return implied_over


# ---------------------------------------------------------------------------
# 4. Book Edge Detection
# ---------------------------------------------------------------------------

def find_book_edge(our_prob, book_odds):
    """
    Compute the edge we have over the book.

    Args:
        our_prob: our model's estimated probability (e.g., 0.62)
        book_odds: decimal odds offered (e.g., 1.91 for -110)

    Returns:
        dict with edge details

    If our model says 62% and the book implies 52.4% (-110), we have a
    9.6% edge. Only bet when edge > vig (~4.76% for -110/-110).
    """
    book_implied = compute_implied_prob(book_odds)
    # Approximate fair prob by removing half the vig from implied
    book_fair = book_implied / (1.0 + STANDARD_VIG / 2)
    edge = our_prob - book_fair
    ev = compute_ev(our_prob, book_odds)

    return {
        'our_prob': round(our_prob, 4),
        'book_implied': round(book_implied, 4),
        'book_fair_estimate': round(book_fair, 4),
        'edge_pct': round(edge * 100, 2),
        'ev_per_unit': round(ev, 4),
        'beats_vig': edge > STANDARD_VIG,
        'is_value_bet': ev > 0,
    }


# ---------------------------------------------------------------------------
# 5. Kelly Criterion for EV-Based Sizing
# ---------------------------------------------------------------------------

def kelly_stake(prob, odds, fraction=KELLY_FRACTION, max_cap=MAX_KELLY_CAP):
    """
    Kelly criterion bet sizing based on edge.

    Full Kelly: f* = (b*p - q) / b
    where b = net odds (odds - 1), p = win prob, q = 1 - p

    We use fractional Kelly (default 25%) for safety.
    Parlays are high-variance, so conservative sizing is critical.

    Args:
        prob: estimated win probability
        odds: decimal odds
        fraction: Kelly fraction (0.25 = quarter-Kelly)
        max_cap: maximum fraction of bankroll (0.05 = 5%)

    Returns:
        recommended fraction of bankroll to stake (0.0 to max_cap)
    """
    prob = max(0.001, min(0.999, prob))
    b = max(0.01, odds - 1)  # net payout per unit
    q = 1.0 - prob

    kelly = (b * prob - q) / b
    if kelly <= 0:
        return 0.0

    sized = kelly * fraction
    return round(min(sized, max_cap), 4)


def kelly_parlay_stake(parlay_prob, parlay_odds,
                       fraction=KELLY_FRACTION, max_cap=MAX_KELLY_CAP):
    """Kelly sizing for a full parlay."""
    return kelly_stake(parlay_prob, parlay_odds, fraction, max_cap)


# ---------------------------------------------------------------------------
# 6. Extract Odds from Prop Data
# ---------------------------------------------------------------------------

def _get_prop_odds(prop):
    """
    Extract decimal odds for a prop.

    Priority:
    1. 'multiplier' field from parse_board.py
    2. 'over_mult' / 'under_mult' based on direction
    3. Default to 1.91 (-110 standard)

    The multiplier from ParlayPlay/sportsbooks IS the decimal odds
    (e.g., 1.87x means you get $1.87 back per $1 staked, for $0.87 profit).
    """
    direction = prop.get('direction', 'OVER').upper()

    # Direct multiplier from board parsing
    mult = prop.get('multiplier')
    if mult and mult > 1.0:
        return float(mult)

    # Direction-specific multipliers
    if direction == 'OVER' and prop.get('over_mult'):
        return float(prop['over_mult'])
    elif direction == 'UNDER' and prop.get('under_mult'):
        return float(prop['under_mult'])

    # SGO odds if attached
    sgo_odds = prop.get('sgo_odds')
    if sgo_odds and sgo_odds > 1.0:
        return float(sgo_odds)

    return DEFAULT_ODDS


def _get_prop_prob(prop):
    """Best available probability estimate for a prop."""
    return prop.get('ensemble_prob',
                    prop.get('xgb_prob',
                             prop.get('xgb_prob_calibrated',
                                      prop.get('focused_prob', 0.50))))


# ---------------------------------------------------------------------------
# 7. Find Positive EV Props
# ---------------------------------------------------------------------------

def find_positive_ev_props(results, min_ev=MIN_EDGE_PCT):
    """
    From all scored props, find those with positive expected value.

    Args:
        results: list of prop dicts (from full_board.json or analysis output)
        min_ev: minimum EV per unit to include (0.02 = 2% edge)

    Returns:
        list of prop dicts enriched with EV fields, sorted by EV descending
    """
    ev_props = []

    for prop in results:
        # Skip errored or unscored props
        if 'error' in prop or prop.get('tier') == 'SKIP':
            continue

        prob = _get_prop_prob(prop)
        odds = _get_prop_odds(prop)

        if prob <= 0 or prob >= 1.0:
            continue

        ev = compute_ev(prob, odds)
        book_implied = compute_implied_prob(odds)
        edge = prob - book_implied

        # Only include positive EV props above threshold
        if ev < min_ev:
            continue

        enriched = dict(prop)
        enriched['ev_per_unit'] = round(ev, 4)
        enriched['edge_pct'] = round(edge * 100, 2)
        enriched['prop_odds'] = odds
        enriched['book_implied_prob'] = round(book_implied, 4)
        enriched['kelly_stake'] = kelly_stake(prob, odds)
        enriched['is_positive_ev'] = True
        ev_props.append(enriched)

    # Sort by EV descending
    ev_props.sort(key=lambda p: p['ev_per_unit'], reverse=True)
    return ev_props


# ---------------------------------------------------------------------------
# 8. Build EV-Optimized Parlay
# ---------------------------------------------------------------------------

def _is_ev_eligible(prop):
    """Basic eligibility for EV parlay consideration."""
    if 'error' in prop or prop.get('tier') == 'SKIP':
        return False
    status = (prop.get('player_injury_status') or '').upper()
    if any(tag in status for tag in ('OUT', 'DOUBTFUL')):
        return False
    # D/F tier props are unreliable regardless of EV
    if prop.get('tier', 'F') in ('D', 'F'):
        return False
    return True


def _confidence_score(prop):
    """
    Secondary ranking when odds are uniform (all 1.91).

    At uniform odds, pure EV ranking = pure probability ranking.
    This tiebreaker rewards props where we have HIGH CONFIDENCE in
    our probability estimate -- consistent performers over volatile ones.

    A prop with 60% prob from a player who hits 8/10 recent games with
    stable minutes is more trustworthy than 60% from a streaky player.
    """
    prob = _get_prop_prob(prop)
    ev = compute_ev(prob, _get_prop_odds(prop))

    # Floor consistency: fewer misses = more reliable estimate
    miss_count = prop.get('l10_miss_count', 5)
    consistency = max(0, (10 - miss_count)) * 0.02  # 0.00 to 0.20

    # Minutes stability: stable minutes = stable opportunity
    mins_pct = prop.get('mins_30plus_pct', 50)
    mins_bonus = min(mins_pct / 100, 1.0) * 0.10  # 0.00 to 0.10

    # L5 vs L10 agreement: recent trend matches longer trend
    l5_hr = prop.get('l5_hit_rate', 50)
    l10_hr = prop.get('l10_hit_rate', 50)
    trend_agreement = 1.0 - min(abs(l5_hr - l10_hr) / 100, 0.30)
    trend_bonus = trend_agreement * 0.05  # 0.00 to 0.05

    # UNDER reliability bonus (systematic edge in our data)
    direction = prop.get('direction', 'OVER').upper()
    under_bonus = 0.05 if direction == 'UNDER' else 0.0

    return ev + consistency + mins_bonus + trend_bonus + under_bonus


def build_ev_parlay(results, n_legs=3, target='ev', min_ev=0.0,
                    max_same_team=1, max_combo=1):
    """
    Build a parlay optimizing for Expected Value, not hit rate.

    Instead of picking the 3 most likely props, pick the 3 with the
    highest expected value. A 55% prop at +150 beats a 70% prop at -130.

    Targets:
      'ev':         Maximize raw EV per unit
      'edge':       Maximize minimum edge across legs (floor safety)
      'confidence': EV + consistency tiebreaker (for uniform-odds boards)

    Args:
        results: list of scored prop dicts
        n_legs: number of parlay legs
        target: sort strategy
        min_ev: minimum single-leg EV to consider (0.0 = any positive EV)
        max_same_team: max players from same team
        max_combo: max combo stat legs

    Returns:
        dict with parlay details, EV metrics, and Kelly sizing
    """
    # Score all props with EV
    pool = []
    for prop in results:
        if not _is_ev_eligible(prop):
            continue

        prob = _get_prop_prob(prop)
        odds = _get_prop_odds(prop)
        ev = compute_ev(prob, odds)

        if ev < min_ev:
            continue

        enriched = dict(prop)
        enriched['_ev'] = ev
        enriched['_prob'] = prob
        enriched['_odds'] = odds
        enriched['_edge'] = prob - compute_implied_prob(odds)
        enriched['_confidence_ev'] = _confidence_score(prop)
        pool.append(enriched)

    if not pool:
        return {'legs': [], 'error': 'No positive-EV props found',
                'parlay_ev': 0.0, 'strategy': target}

    # Sort by target metric
    if target == 'edge':
        pool.sort(key=lambda p: p['_edge'], reverse=True)
    elif target == 'confidence':
        pool.sort(key=lambda p: p['_confidence_ev'], reverse=True)
    else:
        pool.sort(key=lambda p: p['_ev'], reverse=True)

    # Greedy selection with constraints
    selected = []
    used_players = set()
    used_games = set()
    team_counts = defaultdict(int)
    combo_count = 0

    for prop in pool:
        player = prop.get('player', '')
        game = prop.get('game', '')
        stat = prop.get('stat', '').lower()
        team = _get_player_team(prop)

        if player in used_players:
            continue
        if game and game in used_games:
            continue
        if team and team_counts.get(team, 0) >= max_same_team:
            continue
        if stat in COMBO_STATS:
            if combo_count >= max_combo:
                continue
            combo_count += 1

        selected.append(prop)
        used_players.add(player)
        if game:
            used_games.add(game)
        if team:
            team_counts[team] += 1

        if len(selected) >= n_legs:
            break

    if not selected:
        return {'legs': [], 'error': 'No legs survived constraints',
                'parlay_ev': 0.0, 'strategy': target}

    # Compute parlay metrics
    leg_data = [{'prob': p['_prob'], 'odds': p['_odds']} for p in selected]
    parlay_info = compute_parlay_ev(leg_data)

    # Format legs
    legs = []
    for p in selected:
        legs.append({
            'player': p.get('player', ''),
            'stat': p.get('stat', ''),
            'line': p.get('line', 0),
            'direction': p.get('direction', 'OVER'),
            'tier': p.get('tier', '?'),
            'game': p.get('game', ''),
            'our_prob': round(p['_prob'], 4),
            'prop_odds': round(p['_odds'], 2),
            'leg_ev': round(p['_ev'], 4),
            'edge_pct': round(p['_edge'] * 100, 2),
            'kelly_stake': kelly_stake(p['_prob'], p['_odds']),
            'l10_hit_rate': p.get('l10_hit_rate', 0),
            'gap': p.get('gap', 0),
            'projection': p.get('projection', 0),
            'ensemble_prob': p.get('ensemble_prob'),
            'xgb_prob': p.get('xgb_prob'),
        })

    under_count = sum(1 for l in legs if l['direction'].upper() == 'UNDER')
    games_used = len(set(l['game'] for l in legs if l['game']))

    # Kelly sizing for full parlay
    parlay_kelly = kelly_parlay_stake(
        parlay_info['parlay_prob'], parlay_info['parlay_odds']
    )

    # Compute total edge: average edge across legs
    avg_edge = sum(l['edge_pct'] for l in legs) / len(legs) if legs else 0
    min_edge = min(l['edge_pct'] for l in legs) if legs else 0

    return {
        'name': f'EV-{target.upper()} {n_legs}-LEG',
        'method': 'ev_optimizer_v1',
        'strategy': target,
        'legs': legs,
        'legs_total': len(legs),
        'under_count': under_count,
        'games_used': games_used,
        'parlay_ev': parlay_info['parlay_ev'],
        'parlay_prob': parlay_info['parlay_prob'],
        'parlay_odds': parlay_info['parlay_odds'],
        'parlay_payout': parlay_info['parlay_payout'],
        'is_positive_ev': parlay_info['is_positive_ev'],
        'kelly_fraction': parlay_kelly,
        'suggested_units': round(parlay_kelly * 100, 1),
        'avg_edge_pct': round(avg_edge, 2),
        'min_edge_pct': round(min_edge, 2),
        'description': (
            f'{len(legs)} legs, {under_count} UNDERs, {games_used} games. '
            f'Parlay EV: {parlay_info["parlay_ev"]:+.4f} per unit. '
            f'Prob: {parlay_info["parlay_prob"]*100:.1f}%. '
            f'Payout: {parlay_info["parlay_odds"]:.1f}x. '
            f'Avg edge: {avg_edge:.1f}%. '
            f'Kelly: {parlay_kelly*100:.1f}% bankroll.'
        ),
    }


def build_ev_parlays(results):
    """
    Build a suite of EV-optimized parlays.

    Returns dict with:
      - ev_safe:       3-leg, confidence-weighted EV, conservative
      - ev_aggressive: 6-8 leg, max EV, broader pool
      - ev_edge:       3-leg, max minimum edge (every leg has large edge)
      - ev_longshot:   3-leg, high-odds targets (+EV longshots)
    """
    # Check if we have real odds diversity or uniform default
    odds_set = set()
    for r in results:
        if _is_ev_eligible(r):
            odds_set.add(_get_prop_odds(r))
    uniform_odds = len(odds_set) <= 1

    # When odds are uniform, use confidence tiebreaker for safe parlay
    safe_target = 'confidence' if uniform_odds else 'ev'

    # Standard EV-optimal 3-leg
    ev_safe = build_ev_parlay(results, n_legs=3, target=safe_target,
                              min_ev=0.02, max_combo=0)

    # Use players from safe for exclusion
    safe_players = set(l['player'] for l in ev_safe.get('legs', []))

    # Aggressive 6-leg (broader pool)
    ev_agg_pool = [r for r in results if r.get('player', '') not in safe_players]
    ev_aggressive = build_ev_parlay(ev_agg_pool, n_legs=6, target='ev',
                                    min_ev=0.0, max_combo=1)

    # Max-edge 3-leg (every leg has strong edge, even if total EV is lower)
    ev_edge = build_ev_parlay(results, n_legs=3, target='edge',
                              min_ev=0.03, max_combo=0)

    # Longshot: props with odds >= 2.50 that still have positive EV
    # When all odds are uniform standard, build a high-prob under parlay instead
    longshot_pool = [r for r in results
                     if _get_prop_odds(r) >= 2.50 and _is_ev_eligible(r)]
    if longshot_pool:
        ev_longshot = build_ev_parlay(longshot_pool, n_legs=3, target='ev',
                                      min_ev=0.0, max_combo=1)
    else:
        # No real longshot odds available -- build UNDER-only EV parlay
        under_pool = [r for r in results
                      if r.get('direction', '').upper() == 'UNDER'
                      and _is_ev_eligible(r)]
        ev_longshot = build_ev_parlay(under_pool, n_legs=3, target='confidence',
                                      min_ev=0.0, max_combo=0)
        if ev_longshot.get('legs'):
            ev_longshot['name'] = 'EV-UNDER 3-LEG'
            ev_longshot['strategy'] = 'under_ev'

    output = {
        'ev_safe': ev_safe,
        'ev_aggressive': ev_aggressive,
        'ev_edge': ev_edge,
    }
    # Only include longshot/under if it has different legs than safe
    safe_players_set = set(l['player'] for l in ev_safe.get('legs', []))
    longshot_players = set(l['player'] for l in ev_longshot.get('legs', []))
    if longshot_players != safe_players_set:
        output['ev_longshot' if longshot_pool else 'ev_under'] = ev_longshot

    return output


# ---------------------------------------------------------------------------
# 9. Pipeline Integration: Enrich Props with EV
# ---------------------------------------------------------------------------

def enrich_with_ev(results):
    """
    Add EV fields to every prop in the results list.

    Fields added:
      - ev_per_unit:    EV per $1 staked (positive = edge)
      - edge_pct:       our prob - book implied prob (in %)
      - kelly_stake:    recommended bankroll fraction
      - is_positive_ev: True if EV > 0
      - prop_odds:      decimal odds used for calculation
      - book_implied:   what the book thinks the probability is

    Modifies results in-place and returns the list.
    """
    for prop in results:
        if 'error' in prop:
            continue

        prob = _get_prop_prob(prop)
        odds = _get_prop_odds(prop)

        if prob <= 0 or prob >= 1.0 or odds <= 1.0:
            prop['ev_per_unit'] = 0.0
            prop['edge_pct'] = 0.0
            prop['kelly_stake'] = 0.0
            prop['is_positive_ev'] = False
            prop['prop_odds'] = odds
            prop['book_implied'] = 0.0
            continue

        ev = compute_ev(prob, odds)
        book_implied = compute_implied_prob(odds)
        edge = prob - book_implied

        prop['ev_per_unit'] = round(ev, 4)
        prop['edge_pct'] = round(edge * 100, 2)
        prop['kelly_stake'] = kelly_stake(prob, odds)
        prop['is_positive_ev'] = ev > 0
        prop['prop_odds'] = round(odds, 3)
        prop['book_implied'] = round(book_implied, 4)

    return len([r for r in results if 'error' not in r])


# ---------------------------------------------------------------------------
# 10. Bankroll Allocation
# ---------------------------------------------------------------------------

def allocate_bankroll(parlays, bankroll=100.0):
    """
    Allocate bankroll across multiple parlays using Kelly-weighted distribution.

    Higher-EV parlays get proportionally more of the bankroll.
    Total allocation capped at 15% of bankroll (conservative).

    Args:
        parlays: dict of parlay name -> parlay dict (from build_ev_parlays)
        bankroll: total bankroll in units

    Returns:
        dict of parlay name -> allocation details
    """
    allocations = {}
    total_kelly = 0.0

    for name, parlay in parlays.items():
        if not parlay.get('legs') or not parlay.get('is_positive_ev', False):
            allocations[name] = {
                'stake': 0.0,
                'pct_bankroll': 0.0,
                'reason': 'Negative EV or no legs',
            }
            continue

        k = parlay.get('kelly_fraction', 0.0)
        total_kelly += k
        allocations[name] = {'raw_kelly': k}

    # Cap total allocation at 15% of bankroll
    max_total = 0.15
    if total_kelly > max_total:
        scale = max_total / total_kelly
    else:
        scale = 1.0

    for name, alloc in allocations.items():
        if 'raw_kelly' not in alloc:
            continue
        k = alloc['raw_kelly'] * scale
        alloc['stake'] = round(k * bankroll, 2)
        alloc['pct_bankroll'] = round(k * 100, 2)
        alloc['parlay_ev'] = parlays[name].get('parlay_ev', 0)
        alloc['parlay_odds'] = parlays[name].get('parlay_odds', 1)
        alloc['expected_profit'] = round(
            alloc['stake'] * parlays[name].get('parlay_ev', 0), 2
        )
        del alloc['raw_kelly']

    return allocations


# ---------------------------------------------------------------------------
# 11. EV Report
# ---------------------------------------------------------------------------

def print_ev_report(results, bankroll=100.0):
    """Print formatted EV analysis report."""
    print(f"\n{'='*72}")
    print(f"  EV OPTIMIZER REPORT")
    print(f"  Bankroll: ${bankroll:.0f}")
    print(f"{'='*72}")

    # Enrich all props
    enriched = enrich_with_ev(list(results))

    # Summary stats
    total = len([r for r in enriched if 'ev_per_unit' in r])
    positive = [r for r in enriched if r.get('is_positive_ev', False)]
    negative = [r for r in enriched if r.get('ev_per_unit', 0) < 0 and 'ev_per_unit' in r]

    print(f"\n  PROP UNIVERSE")
    print(f"  Total scored props:    {total}")
    print(f"  Positive EV (+edge):   {len(positive)} ({len(positive)/total*100:.0f}%)" if total else "")
    print(f"  Negative EV (-edge):   {len(negative)} ({len(negative)/total*100:.0f}%)" if total else "")

    if positive:
        avg_ev = sum(r['ev_per_unit'] for r in positive) / len(positive)
        max_ev = max(r['ev_per_unit'] for r in positive)
        avg_edge = sum(r['edge_pct'] for r in positive) / len(positive)
        print(f"  Avg positive EV:       {avg_ev:+.4f} per unit")
        print(f"  Best single-prop EV:   {max_ev:+.4f} per unit")
        print(f"  Avg edge:              {avg_edge:+.1f}%")

    # Top 10 positive EV props
    top_ev = sorted(positive, key=lambda r: r['ev_per_unit'], reverse=True)[:10]
    if top_ev:
        print(f"\n  TOP 10 POSITIVE EV PROPS")
        print(f"  {'Player':22s} {'Stat':6s} {'Dir':5s} {'Line':>5s} {'Prob':>5s} "
              f"{'Odds':>5s} {'EV':>7s} {'Edge':>6s} {'Kelly':>6s} {'Tier':4s}")
        print(f"  {'-'*68}")
        for r in top_ev:
            print(f"  {r.get('player',''):22s} {r.get('stat',''):6s} "
                  f"{r.get('direction',''):5s} {r.get('line',0):5.1f} "
                  f"{_get_prop_prob(r):5.1%} "
                  f"{r.get('prop_odds', DEFAULT_ODDS):5.2f} "
                  f"{r['ev_per_unit']:+7.4f} "
                  f"{r['edge_pct']:+5.1f}% "
                  f"{r['kelly_stake']*100:5.2f}% "
                  f"{r.get('tier','?'):4s}")

    # EV distribution by direction
    over_ev = [r for r in positive if r.get('direction', '').upper() == 'OVER']
    under_ev = [r for r in positive if r.get('direction', '').upper() == 'UNDER']
    print(f"\n  EV BY DIRECTION")
    print(f"  OVER  +EV: {len(over_ev):3d} props" +
          (f"  avg EV={sum(r['ev_per_unit'] for r in over_ev)/len(over_ev):+.4f}" if over_ev else ""))
    print(f"  UNDER +EV: {len(under_ev):3d} props" +
          (f"  avg EV={sum(r['ev_per_unit'] for r in under_ev)/len(under_ev):+.4f}" if under_ev else ""))

    # EV by tier
    print(f"\n  EV BY TIER")
    for tier in ('S', 'A', 'B', 'C', 'D', 'F'):
        tier_props = [r for r in positive if r.get('tier') == tier]
        if tier_props:
            avg = sum(r['ev_per_unit'] for r in tier_props) / len(tier_props)
            print(f"  {tier}-tier: {len(tier_props):3d} +EV props, avg EV={avg:+.4f}")

    # EV by stat
    print(f"\n  EV BY STAT TYPE")
    stat_groups = defaultdict(list)
    for r in positive:
        stat_groups[r.get('stat', '?').lower()].append(r)
    for stat, props in sorted(stat_groups.items(),
                               key=lambda x: sum(r['ev_per_unit'] for r in x[1])/len(x[1]),
                               reverse=True):
        avg = sum(r['ev_per_unit'] for r in props) / len(props)
        print(f"  {stat:8s}: {len(props):3d} +EV props, avg EV={avg:+.4f}")

    # Build EV parlays
    print(f"\n{'='*72}")
    print(f"  EV-OPTIMIZED PARLAYS")
    print(f"{'='*72}")

    parlays = build_ev_parlays(enriched)
    allocations = allocate_bankroll(parlays, bankroll)

    for name, parlay in parlays.items():
        legs = parlay.get('legs', [])
        if not legs:
            print(f"\n  {parlay.get('name', name)}: No eligible legs")
            continue

        alloc = allocations.get(name, {})
        print(f"\n  {parlay.get('name', name)}")
        print(f"  Strategy: {parlay['strategy']}")
        print(f"  Parlay EV:     {parlay['parlay_ev']:+.4f} per unit")
        print(f"  Parlay prob:   {parlay['parlay_prob']*100:.2f}%")
        print(f"  Parlay payout: {parlay['parlay_payout']:.1f}x")
        print(f"  Avg edge:      {parlay['avg_edge_pct']:+.1f}%")
        print(f"  Min edge:      {parlay['min_edge_pct']:+.1f}%")
        print(f"  Kelly stake:   {parlay['kelly_fraction']*100:.2f}% bankroll")
        if alloc.get('stake'):
            print(f"  Allocation:    ${alloc['stake']:.2f} ({alloc['pct_bankroll']:.1f}%)")
            print(f"  Expected +/-:  ${alloc['expected_profit']:+.2f}")
        print(f"  UNDERs:        {parlay['under_count']}/{parlay['legs_total']}")
        print()

        for i, leg in enumerate(legs, 1):
            print(f"    {i}. {leg['player']:22s} {leg['stat']:6s} "
                  f"{leg['direction']:5s} line={leg['line']:>5.1f}  "
                  f"prob={leg['our_prob']:.1%}  odds={leg['prop_odds']:.2f}x  "
                  f"EV={leg['leg_ev']:+.4f}  edge={leg['edge_pct']:+.1f}%  "
                  f"tier={leg['tier']}")

    # Bankroll summary
    total_stake = sum(a.get('stake', 0) for a in allocations.values())
    total_expected = sum(a.get('expected_profit', 0) for a in allocations.values())
    if total_stake > 0:
        print(f"\n  BANKROLL ALLOCATION SUMMARY")
        print(f"  Total staked:      ${total_stake:.2f} ({total_stake/bankroll*100:.1f}% of bankroll)")
        print(f"  Expected profit:   ${total_expected:+.2f}")
        print(f"  Expected ROI:      {total_expected/total_stake*100:+.1f}%" if total_stake > 0 else "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_player_team(pick):
    """Extract team abbreviation from game context."""
    game = pick.get('game', '')
    is_home = pick.get('is_home')
    if not game or '@' not in game:
        return None
    parts = game.split('@')
    if is_home is True:
        return parts[1].strip() if len(parts) > 1 else None
    elif is_home is False:
        return parts[0].strip() if parts else None
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_results(path):
    """Load results from a JSON file (full_board or predictions)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get('results', data.get('picks', []))
    return []


def main():
    args = sys.argv[1:]

    if not args or args[0] in ('-h', '--help'):
        print("EV Optimizer -- Expected Value Parlay Builder")
        print()
        print("Usage:")
        print("  python3 ev_optimizer.py --board <path> [bankroll]  Full EV analysis on a board")
        print("  python3 ev_optimizer.py --parlay <path>            Score an existing parlay for EV")
        print("  python3 ev_optimizer.py --enrich <path>            Add EV fields to results JSON")
        print("  python3 ev_optimizer.py --test                     Demo with latest data")
        print()
        print("Philosophy: Optimize for expected value (bankroll growth), not hit rate.")
        print("  A 45% prop at +200 odds has MORE value than a 65% prop at -150.")
        print("  EV = (win_prob x payout) - (lose_prob x stake)")
        return

    if args[0] == '--board' and len(args) >= 2:
        results = _load_results(args[1])
        if not results:
            print(f"  No results loaded from {args[1]}")
            return

        # Try to recover multipliers from parsed board in same directory
        board_dir = os.path.dirname(os.path.abspath(args[1]))
        parsed_path = _find_parsed_board(board_dir)
        if parsed_path:
            attached = recover_multipliers(results, parsed_path)
            if attached:
                print(f"  Recovered multipliers for {attached}/{len(results)} props")

        bankroll = float(args[2]) if len(args) >= 3 else 100.0
        print_ev_report(results, bankroll)

    elif args[0] == '--parlay' and len(args) >= 2:
        with open(args[1]) as f:
            data = json.load(f)

        # Handle various parlay JSON formats:
        #   1. {parlays: {safe: {legs: [...]}, aggressive: {legs: [...]}}}
        #   2. {safe: {legs: [...]}, aggressive: {legs: [...]}}
        #   3. [{player, stat, ...}, ...]  (bare leg list)
        parlays = []
        if isinstance(data, dict):
            # Check for nested 'parlays' key (primary_parlays.json format)
            inner = data.get('parlays', data)
            if isinstance(inner, dict):
                for key, val in inner.items():
                    if isinstance(val, dict) and 'legs' in val:
                        parlays.append((val.get('name', key), val))
        elif isinstance(data, list):
            parlays = [('parlay', {'legs': data})]

        for name, parlay in parlays:
            legs = parlay.get('legs', [])
            if not legs:
                continue

            leg_data = []
            for leg in legs:
                prob = _get_prop_prob(leg)
                odds = _get_prop_odds(leg)
                ev = compute_ev(prob, odds)
                leg_data.append({'prob': prob, 'odds': odds})

                edge_info = find_book_edge(prob, odds)
                status = '+EV' if ev > 0 else '-EV'
                print(f"  {status} {leg.get('player',''):22s} "
                      f"{leg.get('stat',''):6s} {leg.get('direction',''):5s} "
                      f"prob={prob:.1%}  odds={odds:.2f}x  "
                      f"EV={ev:+.4f}  edge={edge_info['edge_pct']:+.1f}%")

            parlay_info = compute_parlay_ev(leg_data)
            print(f"\n  {name}: {len(legs)} legs")
            print(f"  Parlay EV:     {parlay_info['parlay_ev']:+.4f}")
            print(f"  Parlay prob:   {parlay_info['parlay_prob']*100:.2f}%")
            print(f"  Parlay payout: {parlay_info['parlay_payout']:.1f}x")
            print(f"  Positive EV:   {'YES' if parlay_info['is_positive_ev'] else 'NO'}")
            kelly = kelly_parlay_stake(parlay_info['parlay_prob'],
                                       parlay_info['parlay_odds'])
            print(f"  Kelly stake:   {kelly*100:.2f}% bankroll")
            print()

    elif args[0] == '--enrich' and len(args) >= 2:
        results = _load_results(args[1])
        enriched = enrich_with_ev(results)

        # Write enriched file
        out_path = args[1].replace('.json', '_ev.json')
        with open(out_path, 'w') as f:
            json.dump(enriched, f, indent=2)
        print(f"  Enriched {len(enriched)} props with EV fields")
        print(f"  Saved to: {out_path}")

        positive = sum(1 for r in enriched if r.get('is_positive_ev', False))
        print(f"  Positive EV: {positive}/{len(enriched)} "
              f"({positive/len(enriched)*100:.0f}%)" if enriched else "")

    elif args[0] == '--test':
        _run_test()

    else:
        print(f"  Unknown command: {args[0]}")
        print("  Use --help for usage.")


def _run_test():
    """Run with most recent board data."""
    # Find most recent date folder with a full board
    for d in sorted(os.listdir(PREDICTIONS_DIR), reverse=True):
        if d.startswith('2026-') and os.path.isdir(os.path.join(PREDICTIONS_DIR, d)):
            date_dir = os.path.join(PREDICTIONS_DIR, d)
            board_path = os.path.join(date_dir, f'{d}_full_board.json')
            if not os.path.exists(board_path):
                # Try any full_board variant
                for fname in os.listdir(date_dir):
                    if 'full_board' in fname and fname.endswith('.json'):
                        board_path = os.path.join(date_dir, fname)
                        break
                else:
                    continue

            results = _load_results(board_path)
            if not results:
                continue

            print(f"  Loading: {board_path}")

            # Attempt to recover multiplier data from parsed board
            parsed_path = _find_parsed_board(date_dir)
            if parsed_path:
                attached = recover_multipliers(results, parsed_path)
                if attached:
                    print(f"  Recovered multipliers for {attached}/{len(results)} props from {os.path.basename(parsed_path)}")
                else:
                    print(f"  No multiplier recovery (all props using default {DEFAULT_ODDS}x odds)")
            else:
                print(f"  No parsed board found for multiplier recovery")

            print_ev_report(results, bankroll=100.0)
            return

    print("  No board data found. Run the pipeline first or provide a --board path.")


if __name__ == '__main__':
    main()

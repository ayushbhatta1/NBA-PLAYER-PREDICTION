#!/usr/bin/env python3
"""
Parlay Correlation Optimizer

Problem: Parlays need ALL legs to hit simultaneously. Greedy selection by individual
pick score ignores inter-leg correlations. Positively correlated legs (same game,
same stat direction, teammates) fail together -- one bad game environment kills
multiple legs. Negatively correlated or independent legs give a higher true parlay
hit rate than the naive product of individual probabilities.

Key insight at 65% per leg (3 legs):
  - Independent:             0.65^3 = 27.5%
  - Positively correlated:   ~22%   (10pp worse)
  - Negatively correlated:   ~32%   (10pp better)

Functions:
  1. compute_pairwise_independence  -- L10 game-log correlation matrix
  2. estimate_parlay_prob           -- adjusted parlay probability
  3. select_uncorrelated_legs       -- maximize parlay prob, not individual prob
  4. build_optimal_parlay           -- end-to-end parlay builder
  5. backtest_correlation           -- compare greedy vs optimized on historical
  6. score_parlay_independence      -- flag risky parlays

No new API calls -- uses L10 values already in each prop dict, plus
teammate_correlations from correlations.py enrichment.
"""

import json
import math
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMBO_STATS = {'pra', 'pr', 'pa', 'ra', 'stl_blk'}
PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_key(pick):
    """Unique identifier for a pick: player + stat."""
    return f"{pick.get('player', '')}_{pick.get('stat', '')}".lower()


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


def _pearson(x, y):
    """Pearson correlation coefficient. Returns 0.0 on insufficient data."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[:n], y[:n]
    mx = sum(x) / n
    my = sum(y) / n
    dx = [xi - mx for xi in x]
    dy = [yi - my for yi in y]
    num = sum(a * b for a, b in zip(dx, dy))
    den_x = math.sqrt(sum(a * a for a in dx))
    den_y = math.sqrt(sum(b * b for b in dy))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _pick_prob(pick):
    """Best available individual probability for a pick."""
    return pick.get('ensemble_prob',
                    pick.get('xgb_prob',
                             pick.get('xgb_prob_calibrated', 0.50)))


# ---------------------------------------------------------------------------
# 1. Pairwise Independence Matrix
# ---------------------------------------------------------------------------

def compute_pairwise_independence(results):
    """
    Build a correlation matrix for all pick pairs in the pool.

    For same-game picks:
      - Use pre-computed teammate_correlations if available (from correlations.py)
      - Otherwise compute Pearson on L10 values (proxy -- not same games, but
        shows if players trend together across recent performance)
    For different-game picks:
      - Default correlation = 0.0 (statistically independent events)
      - Exception: same-stat same-direction picks share systemic risk (e.g.
        all PTS OVERs lose on a low-scoring night), so assign 0.05 correlation

    Returns:
      corr_matrix: dict of dicts, corr_matrix[key_a][key_b] = float in [-1, 1]
    """
    keys = [_pick_key(p) for p in results]
    key_to_pick = {}
    for p in results:
        k = _pick_key(p)
        if k not in key_to_pick:
            key_to_pick[k] = p

    # Pre-index teammate_correlations for fast lookup
    # teammate_correlations stores [{player, stat, correlation, ...}, ...]
    tc_lookup = {}
    for p in results:
        tc_list = p.get('teammate_correlations', [])
        my_key = _pick_key(p)
        for tc in tc_list:
            other_player = tc.get('player', '')
            other_stat = tc.get('stat', '')
            # The correlation in teammate_correlations is on the stat field,
            # but we want correlation for the actual prop stat of the other pick.
            # We store the generic Pearson r here and will look up more specific
            # values when both picks are in the pool.
            other_key = f"{other_player}_{other_stat}".lower()
            tc_lookup[(my_key, other_key)] = tc.get('correlation', 0.0)

    corr_matrix = defaultdict(dict)

    unique_keys = list(set(keys))
    for i, key_a in enumerate(unique_keys):
        pick_a = key_to_pick.get(key_a)
        if not pick_a:
            continue

        corr_matrix[key_a][key_a] = 1.0

        for j in range(i + 1, len(unique_keys)):
            key_b = unique_keys[j]
            pick_b = key_to_pick.get(key_b)
            if not pick_b:
                continue

            game_a = pick_a.get('game', '')
            game_b = pick_b.get('game', '')
            same_game = game_a and game_b and game_a == game_b

            if same_game:
                # Priority 1: Pre-computed teammate_correlations from correlations.py
                r = tc_lookup.get((key_a, key_b))
                if r is None:
                    r = tc_lookup.get((key_b, key_a))

                if r is not None:
                    corr = r
                else:
                    # Priority 2: Pearson on L10 values
                    vals_a = pick_a.get('l10_values', [])
                    vals_b = pick_b.get('l10_values', [])
                    if vals_a and vals_b:
                        corr = _pearson(vals_a, vals_b)
                    else:
                        # Same game but no data -- assume moderate positive
                        # correlation (same game environment affects both)
                        corr = 0.15

                # Same-game same-direction amplifier: if both are OVER in a
                # high-total game or both UNDER in a low-total game, the game
                # environment is a shared risk factor
                dir_a = pick_a.get('direction', '').upper()
                dir_b = pick_b.get('direction', '').upper()
                if dir_a == dir_b:
                    # Same direction in same game = extra correlation from
                    # shared game flow (pace, blowout, OT, etc.)
                    corr = corr + 0.10 * (1.0 if corr >= 0 else -1.0)
                    corr = max(-1.0, min(1.0, corr))

                # Teammates on the same team have higher correlation than
                # opponents (shared possessions vs zero-sum)
                team_a = _get_player_team(pick_a)
                team_b = _get_player_team(pick_b)
                if team_a and team_b and team_a == team_b:
                    stat_a = pick_a.get('stat', '').lower()
                    stat_b = pick_b.get('stat', '').lower()
                    # Same team + same stat type = usage competition (negative)
                    if stat_a == stat_b and stat_a in ('pts', 'ast', '3pm'):
                        corr = max(corr, 0.05)  # floor: mild positive
                        # Note: not strongly negative because team scoring is
                        # positively correlated overall

            else:
                # Different games -- statistically independent
                corr = 0.0

                # Systemic risk: same stat + same direction shares exposure
                # to league-wide variance (e.g. all PTS OVERs lose on a
                # slate where games run slow)
                stat_a = pick_a.get('stat', '').lower()
                stat_b = pick_b.get('stat', '').lower()
                dir_a = pick_a.get('direction', '').upper()
                dir_b = pick_b.get('direction', '').upper()
                if stat_a == stat_b and dir_a == dir_b:
                    corr = 0.05

            corr_matrix[key_a][key_b] = round(corr, 4)
            corr_matrix[key_b][key_a] = round(corr, 4)

    return dict(corr_matrix)


# ---------------------------------------------------------------------------
# 2. Parlay Probability Estimation
# ---------------------------------------------------------------------------

def estimate_parlay_prob(legs, corr_matrix):
    """
    Estimate true parlay hit probability accounting for correlations.

    Naive:    product of individual probs (assumes full independence)
    Adjusted: penalize for positive correlation, reward for negative

    The adjustment formula:
      adjusted = naive * (1 - avg_pairwise_corr * penalty_weight)

    Where penalty_weight scales with the number of legs (more legs =
    more compounding of correlated risk).

    Returns:
      dict with naive_prob, adjusted_prob, avg_correlation, independence_score
    """
    if not legs:
        return {'naive_prob': 0.0, 'adjusted_prob': 0.0,
                'avg_correlation': 0.0, 'independence_score': 1.0}

    # Individual probabilities
    probs = [max(0.01, min(0.99, _pick_prob(leg))) for leg in legs]
    naive = 1.0
    for p in probs:
        naive *= p

    # Pairwise correlations
    keys = [_pick_key(leg) for leg in legs]
    pair_corrs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = corr_matrix.get(keys[i], {}).get(keys[j], 0.0)
            pair_corrs.append(r)

    if not pair_corrs:
        return {'naive_prob': round(naive, 6), 'adjusted_prob': round(naive, 6),
                'avg_correlation': 0.0, 'independence_score': 1.0}

    avg_corr = sum(pair_corrs) / len(pair_corrs)
    max_corr = max(pair_corrs) if pair_corrs else 0.0

    # Penalty weight: scales with leg count because correlation compounds.
    # 3 legs: 0.40, 5 legs: 0.50, 8 legs: 0.55
    n = len(legs)
    penalty_weight = min(0.30 + 0.05 * n, 0.60)

    # Core adjustment:
    # - Positive avg_corr -> reduce probability (correlated failures)
    # - Negative avg_corr -> increase probability (diversification benefit)
    # Capped at +/-30% adjustment to stay realistic
    adjustment = 1.0 - avg_corr * penalty_weight
    adjustment = max(0.70, min(1.30, adjustment))

    # Extra penalty for any single very high correlation pair
    # (one bad pair can sink the parlay even if average is low)
    if max_corr > 0.5:
        max_penalty = (max_corr - 0.5) * 0.15
        adjustment -= max_penalty

    adjusted = naive * adjustment
    adjusted = max(0.001, min(0.99, adjusted))

    # Independence score: 0 = fully correlated, 1 = fully independent
    independence = max(0.0, min(1.0, 1.0 - avg_corr))

    return {
        'naive_prob': round(naive, 6),
        'adjusted_prob': round(adjusted, 6),
        'avg_correlation': round(avg_corr, 4),
        'max_correlation': round(max_corr, 4),
        'independence_score': round(independence, 4),
        'adjustment_factor': round(adjustment, 4),
        'pair_count': len(pair_corrs),
    }


# ---------------------------------------------------------------------------
# 3. Uncorrelated Leg Selection
# ---------------------------------------------------------------------------

def select_uncorrelated_legs(pool, n_legs, corr_matrix, min_prob=0.55,
                             max_combo=1, max_same_team=1):
    """
    Select N legs that MAXIMIZE adjusted parlay probability.

    Greedy algorithm with correlation awareness:
      1. Filter to picks above min_prob
      2. Start with the highest-prob pick
      3. For each subsequent slot, score every candidate by the MARGINAL
         increase in adjusted parlay probability (not just individual prob)
      4. A high-prob but correlated pick scores lower than a slightly
         lower-prob but independent pick

    Constraints:
      - No duplicate players
      - No same game (different games are more independent)
      - Max 1 per team (avoids shared game environment risk)
      - Limited combo stats

    Returns list of selected pick dicts.
    """
    # Filter eligible picks
    eligible = []
    for p in pool:
        if 'error' in p or p.get('tier') == 'SKIP':
            continue
        # Injury filter
        status = (p.get('player_injury_status') or '').upper()
        if any(tag in status for tag in ('OUT', 'DOUBTFUL', 'QUESTIONABLE',
                                          'GTD', 'GAME-TIME', 'DAY-TO-DAY')):
            continue
        if _pick_prob(p) < min_prob:
            continue
        eligible.append(p)

    if not eligible:
        return []

    # Sort by individual probability as tiebreaker
    eligible.sort(key=lambda p: _pick_prob(p), reverse=True)

    selected = []
    used_players = set()
    used_games = set()
    team_counts = defaultdict(int)
    combo_count = 0

    # Seed: pick the highest-probability eligible pick
    for candidate in eligible:
        player = candidate.get('player', '')
        game = candidate.get('game', '')
        stat = candidate.get('stat', '').lower()
        if stat in COMBO_STATS:
            if combo_count >= max_combo:
                continue
        selected.append(candidate)
        used_players.add(player)
        if game:
            used_games.add(game)
        team = _get_player_team(candidate)
        if team:
            team_counts[team] += 1
        if stat in COMBO_STATS:
            combo_count += 1
        break

    # Greedy expansion: add the pick that maximizes adjusted parlay prob
    while len(selected) < n_legs:
        best_candidate = None
        best_parlay_prob = -1.0

        for candidate in eligible:
            player = candidate.get('player', '')
            game = candidate.get('game', '')
            team = _get_player_team(candidate)
            stat = candidate.get('stat', '').lower()

            # Constraint checks
            if player in used_players:
                continue
            if game and game in used_games:
                continue
            if team and team_counts.get(team, 0) >= max_same_team:
                continue
            if stat in COMBO_STATS and combo_count >= max_combo:
                continue

            # Compute adjusted parlay prob if we add this candidate
            trial = selected + [candidate]
            prob_info = estimate_parlay_prob(trial, corr_matrix)
            trial_prob = prob_info['adjusted_prob']

            if trial_prob > best_parlay_prob:
                best_parlay_prob = trial_prob
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        player = best_candidate.get('player', '')
        game = best_candidate.get('game', '')
        team = _get_player_team(best_candidate)
        stat = best_candidate.get('stat', '').lower()

        used_players.add(player)
        if game:
            used_games.add(game)
        if team:
            team_counts[team] += 1
        if stat in COMBO_STATS:
            combo_count += 1

    return selected


# ---------------------------------------------------------------------------
# 4. End-to-End Optimal Parlay Builder
# ---------------------------------------------------------------------------

def build_optimal_parlay(results, n_legs=3, mode='safe'):
    """
    Build a parlay using correlation optimization.

    Modes:
      'safe':       S/A/B tier, min_prob=0.55, base stats only, no blowout
      'aggressive': S/A/B/C tier, min_prob=0.50, combos allowed, wider filters

    Returns dict with legs, probabilities, and independence metrics.
    """
    # Filter by mode
    if mode == 'safe':
        pool = [p for p in results if (
            p.get('mins_30plus_pct', 0) >= 60 and
            p.get('l10_hit_rate', 0) >= 60 and
            p.get('l10_miss_count', 10) < 4 and
            p.get('stat', '').lower() not in COMBO_STATS and
            not (p.get('direction', '').upper() == 'OVER' and
                 abs(p.get('spread', 0) or 0) >= 10)
        )]
        min_prob = 0.55
        max_combo = 0
    else:
        pool = [p for p in results if (
            p.get('mins_30plus_pct', 0) >= 50 and
            p.get('l10_hit_rate', 0) >= 45
        )]
        min_prob = 0.50
        max_combo = 1

    if not pool:
        return {'legs': [], 'error': f'No eligible picks for mode={mode}'}

    # Compute correlation matrix
    corr_matrix = compute_pairwise_independence(pool)

    # Select optimal legs
    picks = select_uncorrelated_legs(
        pool, n_legs, corr_matrix,
        min_prob=min_prob,
        max_combo=max_combo,
        max_same_team=1,
    )

    if not picks:
        return {'legs': [], 'error': 'No picks survived constraints'}

    # Probability analysis
    prob_info = estimate_parlay_prob(picks, corr_matrix)

    # Format legs
    legs = []
    for p in picks:
        legs.append({
            'player': p.get('player', ''),
            'stat': p.get('stat', ''),
            'line': p.get('line', 0),
            'direction': p.get('direction', 'OVER'),
            'tier': p.get('tier', '?'),
            'gap': p.get('gap', 0),
            'projection': p.get('projection', 0),
            'l10_hit_rate': p.get('l10_hit_rate', 0),
            'l5_hit_rate': p.get('l5_hit_rate', 0),
            'game': p.get('game', ''),
            'ensemble_prob': p.get('ensemble_prob', p.get('xgb_prob')),
            'individual_prob': round(_pick_prob(p), 4),
        })

    under_count = sum(1 for l in legs if l.get('direction', '').upper() == 'UNDER')
    games_used = len(set(l.get('game', '') for l in legs))

    return {
        'name': f'CORR-{mode.upper()} {n_legs}-LEG',
        'method': 'parlay_optimizer_v1',
        'mode': mode,
        'legs': legs,
        'legs_total': len(legs),
        'under_count': under_count,
        'games_used': games_used,
        'naive_parlay_prob': prob_info['naive_prob'],
        'adjusted_parlay_prob': prob_info['adjusted_prob'],
        'avg_correlation': prob_info['avg_correlation'],
        'max_correlation': prob_info.get('max_correlation', 0.0),
        'independence_score': prob_info['independence_score'],
        'adjustment_factor': prob_info.get('adjustment_factor', 1.0),
        'description': (
            f'{n_legs} legs, {under_count} UNDERs, {games_used} games. '
            f'Independence: {prob_info["independence_score"]:.2f}. '
            f'Naive prob: {prob_info["naive_prob"]*100:.1f}% -> '
            f'Adjusted: {prob_info["adjusted_prob"]*100:.1f}%'
        ),
    }


# ---------------------------------------------------------------------------
# 5. Backtest: Correlation-Optimized vs Greedy
# ---------------------------------------------------------------------------

def backtest_correlation(date_str):
    """
    Compare correlation-optimized parlays vs greedy parlays on graded data.

    Loads graded results for a date, builds parlays both ways, checks which
    would have hit.
    """
    # Find graded file
    date_dir = os.path.join(PREDICTIONS_DIR, date_str)
    if not os.path.isdir(date_dir):
        print(f"  No data directory for {date_str}")
        return None

    graded_file = None
    for fname in sorted(os.listdir(date_dir)):
        if 'graded' in fname and fname.startswith('v4') and fname.endswith('.json'):
            graded_file = os.path.join(date_dir, fname)
            break

    if not graded_file:
        print(f"  No graded results file found in {date_dir}")
        return None

    with open(graded_file) as f:
        data = json.load(f)

    results = data.get('results', []) if isinstance(data, dict) else data
    graded = [p for p in results if p.get('actual') is not None]

    if not graded:
        print(f"  No graded picks with actuals for {date_str}")
        return None

    print(f"\n{'='*70}")
    print(f"  BACKTEST: {date_str} ({len(graded)} graded picks)")
    print(f"{'='*70}")

    total_hits = sum(1 for p in graded if p.get('result') == 'HIT')
    print(f"  Overall accuracy: {total_hits}/{len(graded)} = {total_hits/len(graded)*100:.1f}%\n")

    # Build correlation matrix from the full pool
    corr_matrix = compute_pairwise_independence(graded)

    report = {}

    for mode, n_legs in [('safe', 3), ('aggressive', 5)]:
        print(f"  --- {mode.upper()} {n_legs}-LEG ---")

        # Method A: Correlation-optimized
        opt_parlay = build_optimal_parlay(graded, n_legs=n_legs, mode=mode)
        opt_legs = opt_parlay.get('legs', [])

        # Method B: Greedy by ensemble_prob (simulates current engine)
        greedy_picks = _greedy_build(graded, n_legs=n_legs, mode=mode)

        # Grade both
        opt_hits = _grade_parlay(opt_legs, graded)
        greedy_hits = _grade_parlay(greedy_picks, graded)

        opt_all_hit = opt_hits == len(opt_legs) and len(opt_legs) > 0
        greedy_all_hit = greedy_hits == len(greedy_picks) and len(greedy_picks) > 0

        # Independence scores
        opt_prob = opt_parlay.get('adjusted_parlay_prob', 0)
        opt_independence = opt_parlay.get('independence_score', 0)

        greedy_as_picks = [_find_pick(graded, l) for l in greedy_picks]
        greedy_as_picks = [p for p in greedy_as_picks if p]
        greedy_prob_info = estimate_parlay_prob(greedy_as_picks, corr_matrix)

        print(f"  CORR-OPTIMIZED: {opt_hits}/{len(opt_legs)} legs hit  "
              f"{'CASHED' if opt_all_hit else 'MISSED'}  "
              f"(independence={opt_independence:.2f}, adj_prob={opt_prob*100:.1f}%)")
        for leg in opt_legs:
            status = _leg_result(leg, graded)
            print(f"    {'HIT' if status else 'MISS':4s} {leg['player']:22s} "
                  f"{leg['stat']:6s} {leg['direction']:5s} line={leg['line']} "
                  f"prob={leg.get('individual_prob', 0):.2f}")

        print()
        print(f"  GREEDY (by prob): {greedy_hits}/{len(greedy_picks)} legs hit  "
              f"{'CASHED' if greedy_all_hit else 'MISSED'}  "
              f"(independence={greedy_prob_info.get('independence_score', 0):.2f}, "
              f"adj_prob={greedy_prob_info.get('adjusted_prob', 0)*100:.1f}%)")
        for leg in greedy_picks:
            status = _leg_result(leg, graded)
            prob = leg.get('individual_prob', leg.get('ensemble_prob', 0))
            print(f"    {'HIT' if status else 'MISS':4s} {leg.get('player',''):22s} "
                  f"{leg.get('stat',''):6s} {leg.get('direction',''):5s} "
                  f"line={leg.get('line',0)} prob={prob:.2f}")

        print()
        report[mode] = {
            'optimized_hits': opt_hits,
            'optimized_total': len(opt_legs),
            'optimized_cashed': opt_all_hit,
            'optimized_independence': opt_independence,
            'optimized_adj_prob': opt_prob,
            'greedy_hits': greedy_hits,
            'greedy_total': len(greedy_picks),
            'greedy_cashed': greedy_all_hit,
            'greedy_independence': greedy_prob_info.get('independence_score', 0),
            'greedy_adj_prob': greedy_prob_info.get('adjusted_prob', 0),
        }

    return report


def _greedy_build(results, n_legs=3, mode='safe'):
    """Simulate greedy selection by individual prob (current engine behavior)."""
    if mode == 'safe':
        pool = [p for p in results if (
            p.get('tier', 'F') in ('S', 'A', 'B') and
            p.get('mins_30plus_pct', 0) >= 60 and
            p.get('l10_hit_rate', 0) >= 60 and
            p.get('l10_miss_count', 10) < 4 and
            p.get('stat', '').lower() not in COMBO_STATS and
            not (p.get('direction', '').upper() == 'OVER' and
                 abs(p.get('spread', 0) or 0) >= 10)
        )]
    else:
        pool = [p for p in results if (
            p.get('tier', 'F') in ('S', 'A', 'B', 'C') and
            p.get('mins_30plus_pct', 0) >= 50 and
            p.get('l10_hit_rate', 0) >= 45
        )]

    # Sort by individual probability (greedy)
    pool.sort(key=lambda p: _pick_prob(p), reverse=True)

    selected = []
    used_players = set()
    used_games = set()
    team_counts = defaultdict(int)

    for pick in pool:
        player = pick.get('player', '')
        game = pick.get('game', '')
        team = _get_player_team(pick)

        if player in used_players:
            continue
        if game and game in used_games:
            continue
        if team and team_counts.get(team, 0) >= 1:
            continue

        leg = {
            'player': player,
            'stat': pick.get('stat', ''),
            'line': pick.get('line', 0),
            'direction': pick.get('direction', 'OVER'),
            'tier': pick.get('tier', '?'),
            'game': game,
            'individual_prob': round(_pick_prob(pick), 4),
            'ensemble_prob': pick.get('ensemble_prob', pick.get('xgb_prob')),
        }
        selected.append(leg)
        used_players.add(player)
        if game:
            used_games.add(game)
        if team:
            team_counts[team] += 1

        if len(selected) >= n_legs:
            break

    return selected


def _grade_parlay(legs, graded_results):
    """Count how many parlay legs hit in graded results."""
    hits = 0
    for leg in legs:
        if _leg_result(leg, graded_results):
            hits += 1
    return hits


def _leg_result(leg, graded_results):
    """Check if a single leg hit based on graded results."""
    player = leg.get('player', '')
    stat = leg.get('stat', '').lower()
    for r in graded_results:
        if r.get('player', '') == player and r.get('stat', '').lower() == stat:
            return r.get('result') == 'HIT'
    return False


def _find_pick(results, leg):
    """Find the full pick dict matching a leg."""
    player = leg.get('player', '')
    stat = leg.get('stat', '').lower()
    for r in results:
        if r.get('player', '') == player and r.get('stat', '').lower() == stat:
            return r
    return None


# ---------------------------------------------------------------------------
# 6. Independence Scoring for Existing Parlays
# ---------------------------------------------------------------------------

def score_parlay_independence(parlay_legs, full_results=None):
    """
    Score how independent a set of parlay legs are.

    Returns:
      0.0 = fully correlated (risky -- one miss likely means multiple)
      1.0 = fully independent (each leg is its own coin flip)

    If full_results is provided, uses those to compute the correlation matrix.
    Otherwise builds a minimal matrix from the leg data alone.
    """
    if not parlay_legs or len(parlay_legs) < 2:
        return {'independence_score': 1.0, 'avg_correlation': 0.0,
                'max_correlation': 0.0, 'risk_level': 'LOW',
                'pair_details': []}

    # Use full results for richer correlation data if available
    pool = full_results if full_results else parlay_legs
    corr_matrix = compute_pairwise_independence(pool)

    keys = [_pick_key(leg) for leg in parlay_legs]
    pair_details = []
    pair_corrs = []

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            r = corr_matrix.get(keys[i], {}).get(keys[j], 0.0)
            pair_corrs.append(r)
            pair_details.append({
                'pick_a': keys[i],
                'pick_b': keys[j],
                'correlation': round(r, 4),
                'risk': 'HIGH' if r > 0.3 else ('MODERATE' if r > 0.1 else 'LOW'),
            })

    avg_corr = sum(pair_corrs) / len(pair_corrs) if pair_corrs else 0.0
    max_corr = max(pair_corrs) if pair_corrs else 0.0
    independence = max(0.0, min(1.0, 1.0 - avg_corr))

    # Risk level
    if avg_corr > 0.25 or max_corr > 0.5:
        risk = 'HIGH'
    elif avg_corr > 0.10 or max_corr > 0.3:
        risk = 'MODERATE'
    else:
        risk = 'LOW'

    # Sort pair details by correlation descending (show riskiest first)
    pair_details.sort(key=lambda x: abs(x['correlation']), reverse=True)

    return {
        'independence_score': round(independence, 4),
        'avg_correlation': round(avg_corr, 4),
        'max_correlation': round(max_corr, 4),
        'risk_level': risk,
        'pair_details': pair_details,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _demo_test():
    """Run a demo with the most recent date's data."""
    # Find most recent date folder with a full board
    dates = []
    for d in sorted(os.listdir(PREDICTIONS_DIR), reverse=True):
        if d.startswith('2026-') and os.path.isdir(os.path.join(PREDICTIONS_DIR, d)):
            board_path = os.path.join(PREDICTIONS_DIR, d, f'{d}_full_board.json')
            if os.path.exists(board_path):
                dates.append(d)

    if not dates:
        print("  No date folders with full board data found.")
        return

    date_str = dates[0]
    board_path = os.path.join(PREDICTIONS_DIR, date_str, f'{date_str}_full_board.json')

    print(f"\n{'='*70}")
    print(f"  PARLAY CORRELATION OPTIMIZER -- DEMO")
    print(f"  Data: {date_str} ({board_path})")
    print(f"{'='*70}")

    with open(board_path) as f:
        results = json.load(f)

    print(f"\n  Total picks in pool: {len(results)}")
    eligible = [p for p in results if _pick_prob(p) >= 0.50]
    print(f"  Picks with prob >= 50%: {len(eligible)}")

    games = set(p.get('game', '') for p in results)
    print(f"  Unique games: {len(games)}")

    # Build correlation matrix
    print(f"\n  Computing pairwise independence matrix...")
    corr_matrix = compute_pairwise_independence(results)
    total_pairs = sum(len(v) for v in corr_matrix.values()) // 2
    print(f"  Matrix: {len(corr_matrix)} picks, {total_pairs} pairs computed")

    # Show correlation distribution
    all_corrs = []
    keys = list(corr_matrix.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            r = corr_matrix.get(k1, {}).get(k2, 0.0)
            all_corrs.append(r)

    if all_corrs:
        high_corr = sum(1 for r in all_corrs if abs(r) > 0.3)
        moderate = sum(1 for r in all_corrs if 0.1 < abs(r) <= 0.3)
        low = sum(1 for r in all_corrs if abs(r) <= 0.1)
        print(f"  Correlation distribution:")
        print(f"    High (|r|>0.3): {high_corr} pairs ({high_corr/len(all_corrs)*100:.1f}%)")
        print(f"    Moderate:       {moderate} pairs ({moderate/len(all_corrs)*100:.1f}%)")
        print(f"    Low/Independent: {low} pairs ({low/len(all_corrs)*100:.1f}%)")

    # Build optimal parlays
    for mode, n_legs in [('safe', 3), ('aggressive', 5)]:
        print(f"\n  {'='*60}")
        print(f"  CORR-OPTIMIZED {mode.upper()} {n_legs}-LEG PARLAY")
        print(f"  {'='*60}")

        parlay = build_optimal_parlay(results, n_legs=n_legs, mode=mode)

        if parlay.get('error'):
            print(f"  Error: {parlay['error']}")
            continue

        print(f"  Independence score: {parlay['independence_score']:.2f} "
              f"(1.0 = fully independent)")
        print(f"  Avg correlation:    {parlay['avg_correlation']:.4f}")
        print(f"  Max correlation:    {parlay['max_correlation']:.4f}")
        print(f"  Naive parlay prob:  {parlay['naive_parlay_prob']*100:.1f}%")
        print(f"  Adjusted prob:      {parlay['adjusted_parlay_prob']*100:.1f}%")
        print(f"  Adjustment factor:  {parlay['adjustment_factor']:.4f}")
        print(f"  Games used:         {parlay['games_used']}")
        print(f"  UNDERs:             {parlay['under_count']}/{parlay['legs_total']}")
        print()

        for i, leg in enumerate(parlay['legs'], 1):
            print(f"    {i}. {leg['player']:22s} {leg['stat']:6s} {leg['direction']:5s} "
                  f"line={leg['line']:>5}  tier={leg['tier']}  "
                  f"HR={leg['l10_hit_rate']}%  prob={leg['individual_prob']:.2f}  "
                  f"game={leg['game']}")

    # Compare with greedy
    print(f"\n  {'='*60}")
    print(f"  COMPARISON: Greedy vs Optimized")
    print(f"  {'='*60}")

    greedy_safe = _greedy_build(results, n_legs=3, mode='safe')
    greedy_as_picks = [_find_pick(results, l) for l in greedy_safe]
    greedy_as_picks = [p for p in greedy_as_picks if p]
    greedy_prob = estimate_parlay_prob(greedy_as_picks, corr_matrix)

    opt_safe = build_optimal_parlay(results, n_legs=3, mode='safe')

    print(f"\n  GREEDY SAFE 3-LEG:")
    print(f"    Independence: {greedy_prob.get('independence_score', 0):.2f}")
    print(f"    Adj prob:     {greedy_prob.get('adjusted_prob', 0)*100:.1f}%")
    for leg in greedy_safe:
        print(f"      {leg.get('player',''):22s} {leg.get('stat',''):6s} "
              f"{leg.get('direction',''):5s} prob={leg.get('individual_prob',0):.2f}")

    print(f"\n  OPTIMIZED SAFE 3-LEG:")
    print(f"    Independence: {opt_safe.get('independence_score', 0):.2f}")
    print(f"    Adj prob:     {opt_safe.get('adjusted_parlay_prob', 0)*100:.1f}%")
    for leg in opt_safe.get('legs', []):
        print(f"      {leg['player']:22s} {leg['stat']:6s} "
              f"{leg['direction']:5s} prob={leg.get('individual_prob',0):.2f}")

    improvement = 0
    if greedy_prob.get('adjusted_prob', 0) > 0:
        improvement = ((opt_safe.get('adjusted_parlay_prob', 0) /
                        greedy_prob.get('adjusted_prob', 0.001)) - 1) * 100
    print(f"\n  Adjusted probability improvement: {improvement:+.1f}%")


def main():
    args = sys.argv[1:]

    if not args or args[0] in ('-h', '--help'):
        print("Parlay Correlation Optimizer")
        print()
        print("Usage:")
        print("  python3 parlay_optimizer.py --test                 Demo with latest data")
        print("  python3 parlay_optimizer.py --backtest 2026-03-19  Backtest on graded date")
        print("  python3 parlay_optimizer.py --score <file>         Score a parlay file")
        return

    if args[0] == '--test':
        _demo_test()
    elif args[0] == '--backtest' and len(args) >= 2:
        backtest_correlation(args[1])
    elif args[0] == '--score' and len(args) >= 2:
        # Score independence of an existing parlay file
        with open(args[1]) as f:
            data = json.load(f)
        parlays = data if isinstance(data, list) else data.get('parlays', [data])
        if isinstance(parlays, dict):
            parlays = [v for v in parlays.values() if isinstance(v, dict) and 'legs' in v]

        for parlay in parlays:
            name = parlay.get('name', 'Unknown')
            legs = parlay.get('legs', [])
            if not legs:
                continue
            result = score_parlay_independence(legs)
            print(f"\n  {name}:")
            print(f"    Independence: {result['independence_score']:.2f}  "
                  f"Risk: {result['risk_level']}")
            print(f"    Avg corr: {result['avg_correlation']:.4f}  "
                  f"Max corr: {result['max_correlation']:.4f}")
            for detail in result['pair_details'][:5]:
                print(f"      {detail['pick_a']:30s} <-> {detail['pick_b']:30s}  "
                      f"r={detail['correlation']:+.4f}  {detail['risk']}")
    else:
        print(f"  Unknown command: {args[0]}")
        print("  Use --help for usage.")


if __name__ == '__main__':
    main()

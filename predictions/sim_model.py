#!/usr/bin/env python3
"""
Monte Carlo Simulation Model for NBA Prop Betting Pipeline

Generates sim_prob — a distribution-based probability estimate for OVER/UNDER
that feeds into XGBoost and MLP as an additional feature.

Approach: fit a distribution to a player's recent game log values for a given stat,
draw N simulated games, count what fraction land OVER vs UNDER the sportsbook line.

NOT a full possession simulation. This is a fast, vectorized statistical resample
that captures the shape of a player's output distribution (skew, bimodality, etc.)
better than a point estimate + standard deviation alone.

Usage:
    # As a feature generator inside the pipeline
    from sim_model import enrich_with_sim
    enrich_with_sim(results)   # mutates each prop dict in place

    # Standalone test
    python3 predictions/sim_model.py --test
"""

import numpy as np
import time
import sys

# ── scipy KDE import with numpy fallback ──
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════
# CORE SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_player_stat(player_logs, stat, line, n_sims=5000):
    """
    Monte Carlo simulation of a player stat distribution.

    Args:
        player_logs: list of recent game values for the stat (e.g., [22, 18, 25, 30, 15])
        stat: stat type string (used for floor clamping rules)
        line: sportsbook line to evaluate against
        n_sims: number of simulated games to draw

    Returns:
        dict with over_prob, under_prob, sim_mean, sim_std, sim_median
        Returns neutral 0.5 probabilities when data is insufficient.
    """
    neutral = {
        'over_prob': 0.5,
        'under_prob': 0.5,
        'sim_mean': 0.0,
        'sim_std': 0.0,
        'sim_median': 0.0,
    }

    try:
        # Validate inputs
        if not player_logs or not isinstance(player_logs, (list, tuple)):
            return neutral
        values = np.array([float(v) for v in player_logs if v is not None], dtype=np.float64)
        if len(values) < 3:
            return neutral
        line_f = float(line)
        if np.isnan(line_f) or line_f <= 0:
            return neutral

        # Fit distribution and sample
        samples = _draw_samples(values, n_sims)

        # Floor clamp: NBA stats cannot go below 0
        np.clip(samples, 0.0, None, out=samples)

        # Integer stats (pts, reb, ast, stl, blk, 3pm, pra, pr, pa, ra, stl_blk)
        # get rounded to nearest integer for realistic counting
        _integer_stats = {'pts', 'reb', 'ast', 'stl', 'blk', '3pm',
                          'pra', 'pr', 'pa', 'ra', 'stl_blk'}
        if stat in _integer_stats:
            np.round(samples, out=samples)

        # Count OVER/UNDER
        over_count = np.sum(samples > line_f)
        under_count = np.sum(samples < line_f)
        equal_count = n_sims - over_count - under_count

        # Split ties evenly (player hits exactly the line = push, counts as neither)
        over_prob = float(over_count) / n_sims
        under_prob = float(under_count) / n_sims

        return {
            'over_prob': round(over_prob, 4),
            'under_prob': round(under_prob, 4),
            'sim_mean': round(float(np.mean(samples)), 2),
            'sim_std': round(float(np.std(samples)), 2),
            'sim_median': round(float(np.median(samples)), 2),
        }

    except Exception:
        return neutral


def simulate_with_context(player_logs, stat, line,
                          pace_factor=1.0, fatigue_factor=1.0,
                          matchup_factor=1.0, n_sims=5000):
    """
    Context-adjusted Monte Carlo simulation.

    Applies multiplicative context adjustments to every simulated value
    before counting OVER/UNDER. This shifts the distribution without
    changing its shape (preserves variance structure).

    Args:
        player_logs: list of recent game values
        stat: stat type string
        line: sportsbook line
        pace_factor: >1.0 faster game (scale up), <1.0 slower
        fatigue_factor: <1.0 fatigue penalty (scale down)
        matchup_factor: >1.0 favorable matchup (scale up)
        n_sims: number of simulated games

    Returns:
        Same dict as simulate_player_stat with adjusted probabilities.
    """
    neutral = {
        'over_prob': 0.5,
        'under_prob': 0.5,
        'sim_mean': 0.0,
        'sim_std': 0.0,
        'sim_median': 0.0,
    }

    try:
        if not player_logs or not isinstance(player_logs, (list, tuple)):
            return neutral
        values = np.array([float(v) for v in player_logs if v is not None], dtype=np.float64)
        if len(values) < 3:
            return neutral
        line_f = float(line)
        if np.isnan(line_f) or line_f <= 0:
            return neutral

        # Draw raw samples
        samples = _draw_samples(values, n_sims)

        # Apply context multiplier to every sample
        context_mult = float(pace_factor) * float(fatigue_factor) * float(matchup_factor)
        # Clamp multiplier to sane range to avoid degenerate distributions
        context_mult = max(0.5, min(2.0, context_mult))
        samples *= context_mult

        # Floor clamp
        np.clip(samples, 0.0, None, out=samples)

        # Integer rounding for counting stats
        _integer_stats = {'pts', 'reb', 'ast', 'stl', 'blk', '3pm',
                          'pra', 'pr', 'pa', 'ra', 'stl_blk'}
        if stat in _integer_stats:
            np.round(samples, out=samples)

        over_count = np.sum(samples > line_f)
        under_count = np.sum(samples < line_f)

        over_prob = float(over_count) / n_sims
        under_prob = float(under_count) / n_sims

        return {
            'over_prob': round(over_prob, 4),
            'under_prob': round(under_prob, 4),
            'sim_mean': round(float(np.mean(samples)), 2),
            'sim_std': round(float(np.std(samples)), 2),
            'sim_median': round(float(np.median(samples)), 2),
        }

    except Exception:
        return neutral


# ═══════════════════════════════════════════════════════════════
# DISTRIBUTION FITTING & SAMPLING
# ═══════════════════════════════════════════════════════════════

def _draw_samples(values, n_sims):
    """
    Draw n_sims samples from a distribution fitted to observed values.

    Strategy:
        10+ games: KDE (captures skew, bimodality, heavy tails)
        5-9 games: Normal distribution (mean, std)
        3-4 games: Normal with inflated std (uncertainty penalty)

    All paths are fully vectorized — no Python loops over sims.
    """
    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 1.0

    # Minimum std floor: prevents degenerate distributions when a player
    # posts identical stat lines (e.g., 5 games of exactly 3 rebounds)
    std = max(std, mean * 0.05 + 0.5)

    if n >= 10 and HAS_SCIPY:
        # KDE: non-parametric density estimation.
        # Captures real distribution shape — skew for scorers,
        # bimodality for streaky players, heavy tails for volatile stats.
        try:
            kde = gaussian_kde(values, bw_method='scott')
            samples = kde.resample(n_sims).flatten()
            return samples
        except Exception:
            # KDE can fail on degenerate data (all identical values).
            # Fall through to normal distribution.
            pass

    if n >= 5:
        # Enough data for normal approximation
        samples = np.random.normal(mean, std, size=n_sims)
    else:
        # 3-4 games: inflate std by 50% to reflect high uncertainty
        samples = np.random.normal(mean, std * 1.5, size=n_sims)

    return samples


# ═══════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════

def enrich_with_sim(results, n_sims=5000):
    """
    Add sim_prob, sim_mean, sim_std to each prop prediction in results.

    Called after analyze_v3 produces the results list, before XGBoost/MLP scoring.
    Mutates each prop dict in place.

    Args:
        results: list of prop dicts from analyze_v3.analyze_player()
        n_sims: simulations per prop (default 5000)

    Returns:
        int: count of props successfully enriched with simulation data
    """
    enriched = 0

    for prop in results:
        try:
            l10_values = prop.get('l10_values', [])
            stat = prop.get('stat', '')
            line = prop.get('line', 0)
            direction = prop.get('direction', '')

            if not l10_values or not line or not direction:
                prop['sim_prob'] = 0.5
                prop['sim_mean'] = 0.0
                prop['sim_std'] = 0.0
                prop['sim_median'] = 0.0
                prop['sim_over_prob'] = 0.5
                prop['sim_under_prob'] = 0.5
                continue

            sim = simulate_player_stat(l10_values, stat, line, n_sims=n_sims)

            # sim_prob: probability aligned with the predicted direction
            if direction == 'OVER':
                prop['sim_prob'] = sim['over_prob']
            else:
                prop['sim_prob'] = sim['under_prob']

            prop['sim_mean'] = sim['sim_mean']
            prop['sim_std'] = sim['sim_std']
            prop['sim_median'] = sim['sim_median']
            prop['sim_over_prob'] = sim['over_prob']
            prop['sim_under_prob'] = sim['under_prob']
            enriched += 1

        except Exception:
            prop['sim_prob'] = 0.5
            prop['sim_mean'] = 0.0
            prop['sim_std'] = 0.0
            prop['sim_median'] = 0.0
            prop['sim_over_prob'] = 0.5
            prop['sim_under_prob'] = 0.5

    return enriched


def enrich_with_context_sim(results, games_dict=None, n_sims=5000):
    """
    Context-aware simulation enrichment using pace, fatigue, and matchup data.

    Pulls context factors from each prop dict and the GAMES dict to adjust
    the simulation distribution. Falls back to basic simulation when context
    is unavailable.

    Args:
        results: list of prop dicts from analyze_v3
        games_dict: GAMES dict from run_board_v5 (optional)
        n_sims: simulations per prop

    Returns:
        int: count of props enriched
    """
    enriched = 0

    for prop in results:
        try:
            l10_values = prop.get('l10_values', [])
            stat = prop.get('stat', '')
            line = prop.get('line', 0)
            direction = prop.get('direction', '')

            if not l10_values or not line or not direction:
                prop['sim_prob'] = 0.5
                prop['sim_mean'] = 0.0
                prop['sim_std'] = 0.0
                prop['sim_median'] = 0.0
                prop['sim_over_prob'] = 0.5
                prop['sim_under_prob'] = 0.5
                continue

            # Extract context factors from prop and GAMES dicts
            pace_factor = _get_pace_factor(prop, games_dict)
            fatigue_factor = _get_fatigue_factor(prop)
            matchup_factor = _get_matchup_factor(prop)

            sim = simulate_with_context(
                l10_values, stat, line,
                pace_factor=pace_factor,
                fatigue_factor=fatigue_factor,
                matchup_factor=matchup_factor,
                n_sims=n_sims,
            )

            if direction == 'OVER':
                prop['sim_prob'] = sim['over_prob']
            else:
                prop['sim_prob'] = sim['under_prob']

            prop['sim_mean'] = sim['sim_mean']
            prop['sim_std'] = sim['sim_std']
            prop['sim_median'] = sim['sim_median']
            prop['sim_over_prob'] = sim['over_prob']
            prop['sim_under_prob'] = sim['under_prob']
            enriched += 1

        except Exception:
            prop['sim_prob'] = 0.5
            prop['sim_mean'] = 0.0
            prop['sim_std'] = 0.0
            prop['sim_median'] = 0.0
            prop['sim_over_prob'] = 0.5
            prop['sim_under_prob'] = 0.5

    return enriched


# ═══════════════════════════════════════════════════════════════
# CONTEXT FACTOR EXTRACTION
# ═══════════════════════════════════════════════════════════════

def _get_pace_factor(prop, games_dict):
    """
    Extract pace adjustment factor from prop/GAMES data.

    Pace data comes from analyze_v3 Layer 4 via nba_fetcher team rankings.
    A game between two fast teams inflates counting stats; two slow teams deflate.
    """
    pace_adj = prop.get('pace_adjustment')
    if pace_adj is not None:
        try:
            # pace_adjustment is a percentage of projection (e.g., 0.03 = +3%)
            return 1.0 + float(pace_adj)
        except (ValueError, TypeError):
            pass

    # Try GAMES dict for pace data
    if games_dict:
        game = prop.get('game', '')
        game_info = games_dict.get(game, {})
        pace = game_info.get('pace_factor')
        if pace is not None:
            try:
                return float(pace)
            except (ValueError, TypeError):
                pass

    return 1.0


def _get_fatigue_factor(prop):
    """
    Extract fatigue adjustment from B2B, travel, and multi-day schedule data.
    """
    factor = 1.0

    # Back-to-back penalty
    if prop.get('is_b2b'):
        travel_dist = prop.get('travel_distance', 0)
        try:
            td = float(travel_dist)
        except (ValueError, TypeError):
            td = 0
        if td > 1500:
            factor *= 0.96   # long-distance B2B: -4%
        elif td > 500:
            factor *= 0.98   # medium B2B: -2%
        else:
            factor *= 0.99   # local B2B: -1%

    # Multi-day travel fatigue
    travel_7day = prop.get('travel_miles_7day', 0)
    try:
        t7 = float(travel_7day)
    except (ValueError, TypeError):
        t7 = 0
    if t7 > 5000:
        factor *= 0.97    # heavy travel week: -3%
    elif t7 > 3000:
        factor *= 0.99    # moderate travel: -1%

    return factor


def _get_matchup_factor(prop):
    """
    Extract matchup adjustment from opponent defense data.

    Uses opp_stat_allowed_vs_league_avg: positive means opponent gives up
    MORE than average (favorable), negative means stingy defense.
    """
    opp_vs_league = prop.get('opp_stat_allowed_vs_league_avg')
    if opp_vs_league is not None:
        try:
            delta = float(opp_vs_league)
            # Scale: +5 allowed vs avg -> ~5% boost; -5 -> ~5% penalty
            # Cap at +/-15% to avoid extreme adjustments
            adj = delta / 100.0
            adj = max(-0.15, min(0.15, adj))
            return 1.0 + adj
        except (ValueError, TypeError):
            pass

    # Fallback: use matchup_adjustment from analyze_v3
    matchup_adj = prop.get('matchup_adjustment')
    if matchup_adj is not None:
        try:
            # matchup_adjustment is absolute (e.g., +2.5 points)
            # Convert to relative factor using line as base
            line = prop.get('line', 0)
            if line and float(line) > 0:
                return 1.0 + float(matchup_adj) / float(line)
        except (ValueError, TypeError):
            pass

    return 1.0


# ═══════════════════════════════════════════════════════════════
# CLI TEST MODE
# ═══════════════════════════════════════════════════════════════

def _run_test():
    """Test simulation with sample player data."""
    print("=" * 65)
    print("  Monte Carlo Simulation Model — Test Suite")
    print("=" * 65)

    np.random.seed(42)

    # Test 1: High-volume scorer with consistent output (should favor OVER)
    print("\n--- Test 1: Consistent Scorer ---")
    print("  Player: LeBron-type | Stat: PTS | Line: 24.5")
    print("  L10 values: [28, 25, 30, 22, 27, 26, 31, 24, 29, 25]")
    logs_consistent = [28, 25, 30, 22, 27, 26, 31, 24, 29, 25]
    result = simulate_player_stat(logs_consistent, 'pts', 24.5)
    print(f"  OVER prob: {result['over_prob']:.1%}")
    print(f"  UNDER prob: {result['under_prob']:.1%}")
    print(f"  Sim mean: {result['sim_mean']:.1f}  std: {result['sim_std']:.1f}  median: {result['sim_median']:.1f}")

    # Test 2: Volatile role player (BLK — low volume, high variance)
    print("\n--- Test 2: Volatile Blocker ---")
    print("  Player: rim-protector | Stat: BLK | Line: 1.5")
    print("  L10 values: [3, 0, 1, 4, 0, 2, 0, 5, 1, 0]")
    logs_volatile = [3, 0, 1, 4, 0, 2, 0, 5, 1, 0]
    result = simulate_player_stat(logs_volatile, 'blk', 1.5)
    print(f"  OVER prob: {result['over_prob']:.1%}")
    print(f"  UNDER prob: {result['under_prob']:.1%}")
    print(f"  Sim mean: {result['sim_mean']:.1f}  std: {result['sim_std']:.1f}  median: {result['sim_median']:.1f}")

    # Test 3: Short sample (5 games — normal fallback)
    print("\n--- Test 3: Short Sample (5 games) ---")
    print("  Player: recent trade | Stat: AST | Line: 5.5")
    print("  L5 values: [7, 4, 8, 6, 5]")
    logs_short = [7, 4, 8, 6, 5]
    result = simulate_player_stat(logs_short, 'ast', 5.5)
    print(f"  OVER prob: {result['over_prob']:.1%}")
    print(f"  UNDER prob: {result['under_prob']:.1%}")
    print(f"  Sim mean: {result['sim_mean']:.1f}  std: {result['sim_std']:.1f}  median: {result['sim_median']:.1f}")

    # Test 4: Combo stat (PRA)
    print("\n--- Test 4: Combo Stat (PRA) ---")
    print("  Player: all-around | Stat: PRA | Line: 35.5")
    print("  L10 values: [38, 32, 40, 35, 28, 42, 36, 30, 39, 34]")
    logs_pra = [38, 32, 40, 35, 28, 42, 36, 30, 39, 34]
    result = simulate_player_stat(logs_pra, 'pra', 35.5)
    print(f"  OVER prob: {result['over_prob']:.1%}")
    print(f"  UNDER prob: {result['under_prob']:.1%}")
    print(f"  Sim mean: {result['sim_mean']:.1f}  std: {result['sim_std']:.1f}  median: {result['sim_median']:.1f}")

    # Test 5: Context-adjusted simulation
    print("\n--- Test 5: Context Adjustment (fast pace + favorable matchup) ---")
    print("  Same scorer as Test 1, but pace=1.05, fatigue=0.98, matchup=1.03")
    result_ctx = simulate_with_context(
        logs_consistent, 'pts', 24.5,
        pace_factor=1.05, fatigue_factor=0.98, matchup_factor=1.03,
    )
    print(f"  OVER prob: {result_ctx['over_prob']:.1%} (was {result['over_prob']:.1%} without context)")
    print(f"  UNDER prob: {result_ctx['under_prob']:.1%}")
    print(f"  Sim mean: {result_ctx['sim_mean']:.1f}  std: {result_ctx['sim_std']:.1f}")

    # Test 6: Edge cases
    print("\n--- Test 6: Edge Cases ---")
    print("  Empty logs:", simulate_player_stat([], 'pts', 20.5))
    print("  2 games:", simulate_player_stat([15, 20], 'pts', 17.5))
    print("  None in logs:", simulate_player_stat([10, None, 15, 12, None, 18, 20, 14, 16, 11], 'pts', 14.5))
    print("  Negative line:", simulate_player_stat([10, 15, 12], 'pts', -1))

    # Test 7: enrich_with_sim pipeline integration
    print("\n--- Test 7: Pipeline Integration (enrich_with_sim) ---")
    mock_results = [
        {
            'player': 'Test Player A', 'stat': 'pts', 'line': 24.5,
            'direction': 'OVER', 'l10_values': [28, 25, 30, 22, 27, 26, 31, 24, 29, 25],
        },
        {
            'player': 'Test Player B', 'stat': 'reb', 'line': 8.5,
            'direction': 'UNDER', 'l10_values': [7, 9, 6, 8, 10, 7, 5, 9, 8, 6],
        },
        {
            'player': 'Test Player C', 'stat': 'ast', 'line': 5.5,
            'direction': 'OVER', 'l10_values': [],  # no data
        },
    ]
    count = enrich_with_sim(mock_results)
    print(f"  Enriched {count}/{len(mock_results)} props")
    for r in mock_results:
        print(f"  {r['player']} ({r['stat']} {r['direction']} {r['line']}): "
              f"sim_prob={r.get('sim_prob', 'N/A'):.3f}, "
              f"mean={r.get('sim_mean', 0):.1f}, std={r.get('sim_std', 0):.1f}")

    # Test 8: Performance benchmark
    print("\n--- Test 8: Performance Benchmark ---")
    n_props = 400
    mock_bulk = [
        {
            'player': f'Player_{i}', 'stat': 'pts', 'line': 20 + (i % 15),
            'direction': 'OVER' if i % 3 else 'UNDER',
            'l10_values': list(np.random.normal(22, 5, 10)),
        }
        for i in range(n_props)
    ]
    t0 = time.time()
    enrich_with_sim(mock_bulk, n_sims=5000)
    elapsed = time.time() - t0
    total_sims = n_props * 5000
    print(f"  {n_props} props x 5,000 sims = {total_sims:,} total samples")
    print(f"  Elapsed: {elapsed:.2f}s ({elapsed / n_props * 1000:.1f}ms per prop)")
    target = "PASS" if elapsed < 5.0 else "FAIL"
    print(f"  Target <5s: {target}")

    print("\n" + "=" * 65)
    print(f"  Distribution method: {'scipy KDE (10+ games)' if HAS_SCIPY else 'numpy normal (scipy unavailable)'}")
    print("=" * 65)


if __name__ == '__main__':
    if '--test' in sys.argv:
        _run_test()
    else:
        print("Usage: python3 sim_model.py --test")
        print("  Or import: from sim_model import enrich_with_sim")

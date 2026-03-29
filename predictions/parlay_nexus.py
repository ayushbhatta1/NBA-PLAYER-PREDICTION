#!/usr/bin/env python3
"""
NEXUS Parlay Builder v4 — PRIMARY BUILDER
38-agent 5-tier system: 3 Scouts → 4 Evaluators → 11 Constructors → 15 Devils → 5 Judges.
Soft screen (CORE/FLEX/REACH/KILL). Borda count consensus. Cascade fallback guarantees output.
Promoted to primary (Mar 19): NEXUS SAFE 3/3 on Mar 17 + Mar 19 vs Engine 0 cashed parlays.

v2 Fixes from March 13 failures:
- DNP blindness: Murphy (0), Sengun (0) killed parlays → pregame_check.py filters OUT players
- Wrong players selected: KD (10/10), Clingan (10/10) weren't in parlay → scoring reweight
- Grant 9pts with 100% L10 HR: floor check catches players with invisible downside
- Harden PA missed by 0.5: combo stat penalty prefers base stats in parlays
- L5 > L10 weighting: recency matters more (L5 100% + L10 75% > L10 100% + L5 60%)

v1 Fixes (March 12):
- Gillespie (44% mins) was in 4/6 parlays → Hard Screener kills mins < 60%
- Bam thin margin (season avg ≈ line) → Profile Scorer penalizes, Reality Checker catches
- Same bad pick in multiple parlays → Single sequential pipeline, one screen

Architecture:
  SCREEN → Gate1 → SCORE → Gate2 → CORRELATE → Gate3 → ARCHITECT → Gate4 →
  HISTORY → REALITY CHECK (3 retries) → FINALIZE
"""

import json
import os
import math
from datetime import datetime
from collections import defaultdict

# Combo stats reference
COMBO_STATS = {'pra', 'pr', 'pa', 'ra'}

# Base stats that are safer for parlays
BASE_STATS = {'pts', 'reb', 'ast', '3pm', 'stl', 'blk'}


# ═══════════════════════════════════════════════════════════════
# PHASE 1: HARD SCREENER — Binary PASS/REJECT
# ═══════════════════════════════════════════════════════════════

def hard_screen(results, relaxed=False):
    """
    Binary PASS/REJECT. No scoring. Kill bad picks before anything else.

    v2 Hard filters (ALL must pass):
    - mins_30plus_pct >= 60 (kills rotation players)
    - abs_gap >= 1.5 base stats, >= 3.0 combo stats
    - No injured/GTD/OUT players
    - l10_hit_rate >= 60 AND l5_hit_rate >= 40
    - Tier S, A, or B (relaxed from S/A only — B was 61.1% clean on Mar 13)
    - Season avg margin check (the Bam filter)
    - NEW: Floor check — l10_miss_count < 3 (Grant filter)
    - NEW: L10 min must clear line for OVER picks (floor safety)

    If relaxed=True (Gate 1 fallback):
    - mins >= 50%, L10 HR >= 55%, tier S/A/B/C
    """
    if relaxed:
        min_mins = 50
        min_l10_hr = 55
    else:
        min_mins = 60
        min_l10_hr = 60

    min_l5_hr = 40  # same either way

    passed = []
    rejected = []

    for r in results:
        if 'error' in r or r.get('tier') == 'SKIP':
            rejected.append((r, 'ERROR/SKIP'))
            continue

        reasons = []

        # Minutes stability
        mins_pct = r.get('mins_30plus_pct', 0)
        if mins_pct < min_mins:
            reasons.append(f"mins_30plus={mins_pct}% (need >={min_mins}%)")

        # Gap minimum
        abs_gap = r.get('abs_gap', 0)
        stat = r.get('stat', '').lower()
        if stat in COMBO_STATS:
            if abs_gap < 3.0:
                reasons.append(f"combo gap={abs_gap} (need >=3.0)")
        else:
            if abs_gap < 1.5:
                reasons.append(f"gap={abs_gap} (need >=1.5)")

        # Injury gate
        injury = r.get('player_injury_status', '')
        if injury and injury.lower() in ['questionable', 'gtd', 'game-time decision', 'doubtful', 'out']:
            reasons.append(f"injury={injury}")

        # Hit rate gates
        l10_hr = r.get('l10_hit_rate', 0)
        l5_hr = r.get('l5_hit_rate', 0)
        if l10_hr < min_l10_hr:
            reasons.append(f"L10HR={l10_hr}% (need >={min_l10_hr}%)")
        if l5_hr < min_l5_hr:
            reasons.append(f"L5HR={l5_hr}% (need >={min_l5_hr}%)")

        # Season avg margin check (the Bam filter)
        season_avg = r.get('season_avg', 0)
        line = r.get('line', 0)
        direction = r.get('direction', '')
        if direction == 'OVER' and season_avg < line - 1.0:
            reasons.append(f"OVER but season_avg={season_avg} < line-1={line-1.0}")
        elif direction == 'UNDER' and season_avg > line + 1.0:
            reasons.append(f"UNDER but season_avg={season_avg} > line+1={line+1.0}")

        # NEW v2: Floor check — the Grant filter
        # If 3+ of L10 games missed the line, don't trust this pick for parlays
        l10_miss_count = r.get('l10_miss_count', 0)
        if l10_miss_count >= 3 and not relaxed:
            reasons.append(f"FLOOR FAIL: {l10_miss_count}/10 games missed line (need <3)")

        # NEW v2: L10 floor safety — worst L10 game vs line
        l10_floor = r.get('l10_floor', 0)
        if direction == 'OVER' and l10_floor > 0 and line > 0:
            floor_gap = l10_floor - line
            # If worst game was WAY below line, that's a hidden risk
            if floor_gap < -5 and not relaxed:
                reasons.append(f"FLOOR RISK: worst L10 game={l10_floor} vs line={line} (gap={floor_gap:.1f})")

        if reasons:
            rejected.append((r, '; '.join(reasons)))
        else:
            passed.append(r)

    return passed, rejected


def gate1(results):
    """Gate 1: Min 8 picks survive. If <8, relax and re-screen."""
    passed, rejected = hard_screen(results, relaxed=False)

    if len(passed) >= 8:
        return passed, rejected, False

    # Relax and re-screen
    passed_relaxed, rejected_relaxed = hard_screen(results, relaxed=True)
    return passed_relaxed, rejected_relaxed, True


# ═══════════════════════════════════════════════════════════════
# PHASE 2: PROFILE SCORER — Score each pick 0-100
# ═══════════════════════════════════════════════════════════════

def profile_score(pick):
    """
    v2 Score a screened pick 0-100:
    - Gap size (30%): 8+ = 30, 6+ = 24, 4+ = 18, 3+ = 10
    - Season avg vs line margin (20%): 4+ = 20, 2+ = 14, 0+ = 8, negative = 0
    - Consistency L5+L10 combined (25%) — v2: L5 weighted MORE than L10
    - Minutes stability (10%)
    - Floor safety (5%): NEW — penalizes invisible downside risk
    - Context factors (10%): BLK/STL +4, combo -4, streaks, B2B, home +1
    """
    score = 0.0

    # ── Gap size (30%) ──
    abs_gap = pick.get('abs_gap', 0)
    if abs_gap >= 8:
        score += 30
    elif abs_gap >= 6:
        score += 24
    elif abs_gap >= 4:
        score += 18
    elif abs_gap >= 3:
        score += 10
    elif abs_gap >= 2:
        score += 5
    else:
        score += 2

    # ── Season avg margin (20%) — the Bam killer ──
    season_avg = pick.get('season_avg', 0)
    line = pick.get('line', 0)
    direction = pick.get('direction', '')
    if direction == 'OVER':
        margin = season_avg - line
    else:
        margin = line - season_avg

    if margin >= 4:
        score += 20
    elif margin >= 2:
        score += 14
    elif margin >= 0:
        score += 8
    else:
        score += 0  # season avg on wrong side of line

    # ── Consistency (25%) — v2: L5 weighted MORE than L10 ──
    l10_hr = pick.get('l10_hit_rate', 50)
    l5_hr = pick.get('l5_hit_rate', 50)
    # v2: L5 worth 15%, L10 worth 10% (recency > history)
    consistency = (l5_hr / 100) * 15 + (l10_hr / 100) * 10
    score += consistency

    # ── Minutes stability (10%) ──
    mins_pct = pick.get('mins_30plus_pct', 50)
    if mins_pct >= 90:
        score += 10
    elif mins_pct >= 80:
        score += 8
    elif mins_pct >= 70:
        score += 6
    elif mins_pct >= 60:
        score += 4
    else:
        score += 2

    # ── Floor safety (5%) — NEW v2 ──
    l10_miss_count = pick.get('l10_miss_count', 0)
    l10_floor = pick.get('l10_floor', 0)
    if l10_miss_count == 0:
        score += 5  # perfect floor
    elif l10_miss_count == 1:
        score += 3
    elif l10_miss_count == 2:
        score += 1
    else:
        score += 0  # 3+ misses = no floor bonus

    # Floor gap penalty (for OVER picks: how bad was worst game?)
    if direction == 'OVER' and l10_floor > 0 and line > 0:
        floor_gap = l10_floor - line
        if floor_gap < -3:
            score -= 2  # worst game was 3+ below line

    # ── Context factors (10%) ──
    ctx = 5  # baseline
    stat = pick.get('stat', '').lower()
    if stat in ['blk', 'stl']:
        ctx += 4  # historically 85%+ accuracy
    if stat in COMBO_STATS:
        ctx -= 4  # combo volatility — v2: increased penalty from -3 to -4
    if pick.get('streak_status') == 'HOT':
        ctx += 2
    elif pick.get('streak_status') == 'COLD':
        ctx -= 2
    if pick.get('is_b2b'):
        ctx -= 1
    # v2: slight home advantage
    if pick.get('is_home') is True:
        ctx += 1

    # v2: Trend reversal penalty (L10 much higher than L5 = cooling)
    if l10_hr > 0 and l5_hr < l10_hr - 20:
        ctx -= 2  # cooling trend

    score += max(0, min(10, ctx))

    return round(score, 1)


def gate2(scored_picks):
    """Gate 2: Top pick must score >= 50, else NO PARLAY TODAY. (v2: lowered from 55 since B-tier now included)"""
    if not scored_picks:
        return False, "No picks survived screening"
    top = max(p['nexus_score'] for p in scored_picks)
    if top < 50:
        return False, f"Top score {top} < 50 threshold — NO PARLAY TODAY"
    return True, f"Top score {top} — proceeding"


# ═══════════════════════════════════════════════════════════════
# PHASE 3: CORRELATION ANALYZER
# ═══════════════════════════════════════════════════════════════

def analyze_correlations(scored_picks, GAMES):
    """
    Build conflict matrix:
    - Same-game = CONFLICT_HIGH (blocked)
    - Same-team = CONFLICT_MEDIUM (max 1 pair)
    - Same-team same-stat = USAGE_CONFLICT (two guards chasing pts on same team)
    - Blowout cluster: 3+ OVERs in blowout-risk games = CLUSTER_RISK
    - Direction imbalance: 4+ same direction = flagged
    - Stat concentration: 3+ same stat = flagged
    """
    n = len(scored_picks)
    conflicts = {}
    flags = []

    # Build pairwise conflict matrix
    for i in range(n):
        for j in range(i + 1, n):
            a = scored_picks[i]
            b = scored_picks[j]
            key = (i, j)

            game_a = a.get('game', '')
            game_b = b.get('game', '')
            team_a = _get_player_team(a)
            team_b = _get_player_team(b)

            if game_a and game_b and game_a == game_b:
                conflicts[key] = 'CONFLICT_HIGH'
            elif team_a and team_b and team_a == team_b:
                # v2: same-team + same stat = usage conflict (stronger than medium)
                if a.get('stat', '').lower() == b.get('stat', '').lower() and a.get('stat', '').lower() in ['pts', 'ast']:
                    conflicts[key] = 'CONFLICT_HIGH'  # anti-correlation: competing for usage
                else:
                    conflicts[key] = 'CONFLICT_MEDIUM'

    # Blowout cluster check
    blowout_overs = []
    for i, p in enumerate(scored_picks):
        spread = p.get('spread')
        if spread is not None and abs(spread) >= 10 and p.get('direction') == 'OVER':
            blowout_overs.append(i)
    if len(blowout_overs) >= 3:
        flags.append({
            'type': 'CLUSTER_RISK',
            'detail': f'{len(blowout_overs)} OVERs in blowout-risk games',
            'indices': blowout_overs,
        })

    # v2: Blowout minutes risk — star in blowout might sit Q4
    for i, p in enumerate(scored_picks):
        spread = p.get('spread')
        if spread is not None and abs(spread) >= 10 and p.get('direction') == 'OVER':
            # If the line requires near-max output, flag it
            season_avg = p.get('season_avg', 0)
            line = p.get('line', 0)
            if season_avg > 0 and line > season_avg * 0.85:
                flags.append({
                    'type': 'BLOWOUT_MINUTES',
                    'detail': f'{p.get("player","?")} needs {line} (near season avg {season_avg}) in blowout-risk game',
                    'indices': [i],
                })

    # Direction imbalance
    over_count = sum(1 for p in scored_picks if p.get('direction') == 'OVER')
    under_count = sum(1 for p in scored_picks if p.get('direction') == 'UNDER')
    if over_count >= 4 and under_count == 0:
        flags.append({'type': 'DIRECTION_IMBALANCE', 'detail': f'{over_count} OVERs, 0 UNDERs'})
    if under_count >= 4 and over_count == 0:
        flags.append({'type': 'DIRECTION_IMBALANCE', 'detail': f'{under_count} UNDERs, 0 OVERs'})

    # Stat concentration
    stat_counts = defaultdict(list)
    for i, p in enumerate(scored_picks):
        stat_counts[p.get('stat', '')].append(i)
    for stat, indices in stat_counts.items():
        if len(indices) >= 3:
            flags.append({'type': 'STAT_CONCENTRATION', 'detail': f'{len(indices)}x {stat.upper()}', 'indices': indices})

    return conflicts, flags


# ═══════════════════════════════════════════════════════════════
# PHASE 4: PARLAY ARCHITECT — Greedy build with constraints
# ═══════════════════════════════════════════════════════════════

def build_nexus_parlays(scored_picks, conflicts, flags):
    """
    v2 Greedy build respecting correlation constraints:
    - No same-game picks
    - No same-team picks (unless different stats)
    - Max 1 combo stat per parlay
    - At least 1 UNDER in 5-leg parlays
    - Max 2 picks from blowout-risk games
    - v2: Max 2 legs from same game (enforced harder)
    - v2: Prefer base stats over combo stats
    - v2: Slight home player preference

    Builds 3 candidates: Safe (3-leg), Main (5-leg), Aggressive (5-leg different pool)
    """
    # Get blowout-risk indices
    blowout_indices = set()
    blowout_minutes_indices = set()
    for f in flags:
        if f['type'] == 'CLUSTER_RISK':
            blowout_indices.update(f.get('indices', []))
        if f['type'] == 'BLOWOUT_MINUTES':
            blowout_minutes_indices.update(f.get('indices', []))

    # Sort by nexus_score descending
    indexed = list(enumerate(scored_picks))
    indexed.sort(key=lambda x: x[1]['nexus_score'], reverse=True)

    def greedy_select(pool, target_legs, require_under=False):
        """Select legs greedily from pool respecting all constraints."""
        selected = []
        selected_indices = set()
        used_games = set()
        used_teams = set()
        combo_count = 0
        blowout_count = 0
        has_under = False

        for idx, pick in pool:
            if idx in selected_indices:
                continue

            game = pick.get('game', '')
            team = _get_player_team(pick)
            stat = pick.get('stat', '').lower()
            direction = pick.get('direction', '')

            # Same-game block
            if game and game in used_games:
                continue
            # Same-team block
            if team and team in used_teams:
                continue
            # Combo limit — v2: max 1 combo stat per parlay
            if stat in COMBO_STATS:
                if combo_count >= 1:
                    continue
                combo_count += 1
            # Blowout limit
            if idx in blowout_indices:
                if blowout_count >= 2:
                    continue
                blowout_count += 1
            # v2: blowout minutes risk — skip if star needs high output in blowout
            if idx in blowout_minutes_indices:
                if blowout_count >= 1:  # tighter limit for minutes-risk
                    continue

            # Check pairwise conflicts
            blocked = False
            for sel_idx in selected_indices:
                pair = (min(idx, sel_idx), max(idx, sel_idx))
                if conflicts.get(pair) == 'CONFLICT_HIGH':
                    blocked = True
                    break
            if blocked:
                continue

            selected.append(pick)
            selected_indices.add(idx)
            if game:
                used_games.add(game)
            if team:
                used_teams.add(team)
            if direction == 'UNDER':
                has_under = True

            if len(selected) >= target_legs:
                break

        # For 5-leg: try to ensure at least 1 UNDER
        if require_under and not has_under and len(selected) >= target_legs:
            # Find best UNDER not yet selected
            for idx, pick in pool:
                if idx in selected_indices:
                    continue
                if pick.get('direction') != 'UNDER':
                    continue
                game = pick.get('game', '')
                team = _get_player_team(pick)
                if game and game in used_games:
                    continue
                if team and team in used_teams:
                    continue
                # Swap out the weakest leg
                weakest_idx = min(range(len(selected)), key=lambda i: selected[i]['nexus_score'])
                if pick['nexus_score'] >= selected[weakest_idx]['nexus_score'] * 0.7:
                    selected[weakest_idx] = pick
                    has_under = True
                break

        return selected

    parlays = {}

    # Safe 3-leg: top 3
    safe = greedy_select(indexed, 3)
    if len(safe) >= 3:
        conf = _geometric_mean([p['nexus_score'] for p in safe])
        parlays['nexus_safe_3leg'] = {
            'legs': [_nexus_leg(p) for p in safe],
            'confidence': round(conf, 1),
            'description': 'NEXUS v2 Safe 3-leg: Floor-checked, base-stat preferred',
            'method': 'nexus-v2 sequential pipeline',
        }

    # Main 5-leg: top 5 with UNDER requirement
    main = greedy_select(indexed, 5, require_under=True)
    if len(main) >= 5:
        conf = _geometric_mean([p['nexus_score'] for p in main])
        parlays['nexus_main_5leg'] = {
            'legs': [_nexus_leg(p) for p in main],
            'confidence': round(conf, 1),
            'description': 'NEXUS v2 Main 5-leg: Full pipeline with floor check + UNDER diversity',
            'method': 'nexus-v2 sequential pipeline',
        }

    # Aggressive 5-leg: skip top 2, build from rest (different pool)
    aggressive_pool = indexed[2:]  # skip top 2 picks
    aggressive = greedy_select(aggressive_pool, 5, require_under=True)
    if len(aggressive) >= 5:
        conf = _geometric_mean([p['nexus_score'] for p in aggressive])
        parlays['nexus_aggressive_5leg'] = {
            'legs': [_nexus_leg(p) for p in aggressive],
            'confidence': round(conf, 1),
            'description': 'NEXUS v2 Aggressive 5-leg: Alternative pool for diversification',
            'method': 'nexus-v2 sequential pipeline',
        }

    return parlays


def gate4(parlays):
    """Gate 4: parlay_confidence >= 50 required for each parlay."""
    viable = {}
    rejected = {}
    for name, parlay in parlays.items():
        conf = parlay.get('confidence', 0)
        if conf >= 50:
            viable[name] = parlay
        else:
            rejected[name] = parlay
    return viable, rejected


# ═══════════════════════════════════════════════════════════════
# PHASE 5: HISTORICAL PATTERN MATCHER
# ═══════════════════════════════════════════════════════════════

def match_historical_patterns(parlays, historical_dir=None):
    """
    Profile-match each leg against March 10-13 graded data.
    Build profile: {stat_type, direction, gap_bucket, hr_bucket, mins_bucket}
    Rate as STRONG / NEUTRAL / WEAK / DANGEROUS
    """
    # Load historical graded data if available
    historical_profiles = _load_historical_data(historical_dir)

    for name, parlay in parlays.items():
        for leg in parlay.get('legs', []):
            profile = _build_profile(leg)
            rating = _rate_profile(profile, historical_profiles)
            leg['historical_rating'] = rating['rating']
            leg['historical_detail'] = rating['detail']

    return parlays


def _load_historical_data(historical_dir):
    """Load graded results from past days to build accuracy profiles."""
    profiles = defaultdict(lambda: {'hits': 0, 'total': 0})

    if not historical_dir:
        # Try default paths — v2: include Mar 13
        base = os.path.dirname(os.path.abspath(__file__))
        for date in ['2026-03-10', '2026-03-11', '2026-03-12', '2026-03-13']:
            day_dir = os.path.join(base, date)
            _load_day(day_dir, profiles)
    else:
        _load_day(historical_dir, profiles)

    return profiles


def _load_day(day_dir, profiles):
    """Load a single day's graded results into profiles."""
    graded_files = [
        'graded_results.json', 'results_graded.json',
        'graded_full_board.json', 'graded_results_mar12.json',
        'v3_graded_full.json',
    ]
    for fname in graded_files:
        fpath = os.path.join(day_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    data = json.load(f)
                # Handle both formats: list or dict with 'results' key
                if isinstance(data, dict) and 'results' in data:
                    results = data['results']
                elif isinstance(data, list):
                    results = data
                else:
                    continue

                for r in results:
                    if r.get('result') not in ['HIT', 'MISS']:
                        continue
                    key = _profile_key(r)
                    profiles[key]['total'] += 1
                    if r['result'] == 'HIT':
                        profiles[key]['hits'] += 1
                return  # only load first found file per day
            except (json.JSONDecodeError, KeyError):
                continue


def _profile_key(r):
    """Build a profile key for historical matching."""
    stat = r.get('stat', '?')
    direction = r.get('direction', '?')

    gap = abs(r.get('gap', 0))
    if gap >= 6:
        gap_bucket = 'huge'
    elif gap >= 4:
        gap_bucket = 'large'
    elif gap >= 2:
        gap_bucket = 'medium'
    else:
        gap_bucket = 'small'

    hr = r.get('l10_hit_rate', 50)
    if hr >= 80:
        hr_bucket = 'elite'
    elif hr >= 60:
        hr_bucket = 'solid'
    else:
        hr_bucket = 'weak'

    mins = r.get('mins_30plus_pct', 50)
    mins_bucket = 'starter' if mins >= 70 else 'rotation'

    return f"{stat}|{direction}|{gap_bucket}|{hr_bucket}|{mins_bucket}"


def _build_profile(leg):
    """Build profile dict from a parlay leg."""
    return {
        'stat': leg.get('stat', '?'),
        'direction': leg.get('direction', '?'),
        'gap': abs(leg.get('gap', 0)),
        'l10_hit_rate': leg.get('l10_hit_rate', 50),
        'l5_hit_rate': leg.get('l5_hit_rate', 50),
        'mins_30plus_pct': leg.get('mins_30plus_pct', 50),
        'season_avg': leg.get('season_avg', 0),
        'line': leg.get('line', 0),
        'l10_floor': leg.get('l10_floor', 0),
        'l10_miss_count': leg.get('l10_miss_count', 0),
    }


def _rate_profile(profile, historical_profiles):
    """Rate a profile as STRONG/NEUTRAL/WEAK/DANGEROUS."""
    stat = profile['stat']
    direction = profile['direction']
    gap = profile['gap']
    hr = profile['l10_hit_rate']
    l5_hr = profile.get('l5_hit_rate', 50)
    mins = profile.get('mins_30plus_pct', 50)

    # Known dangerous patterns from March 10-13
    # Combo OVER with gap < 5 ≈ 55% hit rate
    if stat in COMBO_STATS and direction == 'OVER' and gap < 5:
        return {'rating': 'DANGEROUS', 'detail': f'combo OVER gap={gap:.1f} (<5) — ~55% historical'}
    # Role player OVERs ≈ 40%
    if mins < 60 and direction == 'OVER':
        return {'rating': 'DANGEROUS', 'detail': f'role player OVER (mins={mins}%) — ~40% historical'}
    # v2: Cooling trend (L10 high but L5 dropping)
    if hr >= 80 and l5_hr < 60:
        return {'rating': 'WEAK', 'detail': f'cooling trend: L10={hr}% but L5={l5_hr}%'}

    # Check historical data
    gap_bucket = 'huge' if gap >= 6 else ('large' if gap >= 4 else ('medium' if gap >= 2 else 'small'))
    hr_bucket = 'elite' if hr >= 80 else ('solid' if hr >= 60 else 'weak')
    mins_bucket = 'starter' if mins >= 70 else 'rotation'
    key = f"{stat}|{direction}|{gap_bucket}|{hr_bucket}|{mins_bucket}"

    hist = historical_profiles.get(key)
    if hist and hist['total'] >= 3:
        accuracy = hist['hits'] / hist['total'] * 100
        if accuracy >= 75:
            return {'rating': 'STRONG', 'detail': f'historical {accuracy:.0f}% ({hist["hits"]}/{hist["total"]})'}
        elif accuracy >= 55:
            return {'rating': 'NEUTRAL', 'detail': f'historical {accuracy:.0f}% ({hist["hits"]}/{hist["total"]})'}
        elif accuracy >= 40:
            return {'rating': 'WEAK', 'detail': f'historical {accuracy:.0f}% ({hist["hits"]}/{hist["total"]})'}
        else:
            return {'rating': 'DANGEROUS', 'detail': f'historical {accuracy:.0f}% ({hist["hits"]}/{hist["total"]})'}

    # No historical data — rate by profile characteristics
    if gap >= 6 and hr >= 80 and mins >= 70:
        return {'rating': 'STRONG', 'detail': 'strong profile (high gap + HR + mins)'}
    elif gap >= 3 and hr >= 60:
        return {'rating': 'NEUTRAL', 'detail': 'decent profile'}
    else:
        return {'rating': 'WEAK', 'detail': f'thin profile (gap={gap:.1f}, HR={hr}%)'}


# ═══════════════════════════════════════════════════════════════
# PHASE 6: REALITY CHECKER — Default NEEDS WORK
# ═══════════════════════════════════════════════════════════════

def reality_check(parlay):
    """
    v2 Reality Checker. Default: NEEDS WORK. Runs 8 kill tests.
    Returns (passed, issues, flagged_legs)
    """
    legs = parlay.get('legs', [])
    issues = []
    flagged_legs = set()

    # Test 1: Weakest Link — lowest nexus_score < 45 (v2: lowered from 50 since B-tier now allowed)
    scores = [(i, leg.get('nexus_score', 0)) for i, leg in enumerate(legs)]
    weakest_idx, weakest_score = min(scores, key=lambda x: x[1])
    if weakest_score < 45:
        issues.append(f"WEAKEST LINK: {legs[weakest_idx]['player']} score={weakest_score} (<45)")
        flagged_legs.add(weakest_idx)

    # Test 2: Plausible Failure — 3+ legs with L5 HR < 60%
    volatile_count = 0
    for i, leg in enumerate(legs):
        l5_hr = leg.get('l5_hit_rate', 50)
        if l5_hr < 60:
            volatile_count += 1
            if l5_hr < 40:
                flagged_legs.add(i)
    if volatile_count >= 3:
        issues.append(f"PLAUSIBLE FAILURE: {volatile_count} legs with L5 HR < 60%")

    # Test 3: Margin of Safety — avg season_avg margin < 2.0
    margins = []
    for i, leg in enumerate(legs):
        season_avg = leg.get('season_avg', 0)
        line = leg.get('line', 0)
        direction = leg.get('direction', '')
        if direction == 'OVER':
            margin = season_avg - line
        else:
            margin = line - season_avg
        margins.append(margin)
        if margin < 0:
            flagged_legs.add(i)

    avg_margin = sum(margins) / len(margins) if margins else 0
    if avg_margin < 2.0:
        issues.append(f"MARGIN OF SAFETY: avg margin={avg_margin:.1f} (<2.0)")

    # Test 4: Profile Check — any DANGEROUS = fail, 2+ WEAK = fail
    dangerous = []
    weak = []
    for i, leg in enumerate(legs):
        rating = leg.get('historical_rating', 'NEUTRAL')
        if rating == 'DANGEROUS':
            dangerous.append(i)
            flagged_legs.add(i)
        elif rating == 'WEAK':
            weak.append(i)
    if dangerous:
        issues.append(f"DANGEROUS PROFILE: {len(dangerous)} legs flagged")
    if len(weak) >= 2:
        issues.append(f"WEAK PROFILES: {len(weak)} legs rated WEAK")
        for idx in weak:
            flagged_legs.add(idx)

    # Test 5: Correlation Sanity — check for same-game leaks
    games = [leg.get('game', '') for leg in legs]
    game_counts = defaultdict(int)
    for g in games:
        if g:
            game_counts[g] += 1
    for g, count in game_counts.items():
        if count > 1:
            issues.append(f"CORRELATION: {count} legs from {g}")
            for i, leg in enumerate(legs):
                if leg.get('game') == g:
                    flagged_legs.add(i)

    # Test 6 (NEW v2): GTD/Questionable check
    for i, leg in enumerate(legs):
        injury = leg.get('player_injury_status', '')
        if injury and injury.lower() in ['questionable', 'gtd', 'game-time decision', 'doubtful']:
            issues.append(f"GTD RISK: {leg['player']} status={injury}")
            flagged_legs.add(i)

    # Test 7 (NEW v2): Recent miss check — has player posted below line in last 3 games?
    for i, leg in enumerate(legs):
        l10_values = leg.get('l10_values', [])
        line = leg.get('line', 0)
        direction = leg.get('direction', '')
        if l10_values and len(l10_values) >= 3:
            last_3 = l10_values[:3]
            if direction == 'OVER':
                misses_in_3 = sum(1 for v in last_3 if v <= line)
            else:
                misses_in_3 = sum(1 for v in last_3 if v >= line)
            if misses_in_3 >= 2:
                issues.append(f"RECENT MISS: {leg['player']} missed {misses_in_3}/3 recent games")
                flagged_legs.add(i)

    # Test 8 (NEW v2): Combo stat when base stat alternative might be safer
    combo_legs = [i for i, leg in enumerate(legs) if leg.get('stat', '').lower() in COMBO_STATS]
    if len(combo_legs) >= 2:
        issues.append(f"COMBO OVERLOAD: {len(combo_legs)} combo stat legs (max 1 recommended)")
        # Flag the weakest combo leg
        weakest_combo = min(combo_legs, key=lambda i: legs[i].get('nexus_score', 0))
        flagged_legs.add(weakest_combo)

    # Test 9 (NEW v2): Thin margin extended (the Bam filter v2)
    for i, leg in enumerate(legs):
        season_avg = leg.get('season_avg', 0)
        line = leg.get('line', 0)
        direction = leg.get('direction', '')
        if direction == 'OVER':
            m = season_avg - line
        else:
            m = line - season_avg
        if abs(m) < 1.0:
            issues.append(f"THIN MARGIN: {leg['player']} season_avg={season_avg} vs line={line} (margin={m:+.1f})")
            flagged_legs.add(i)

    passed = len(issues) == 0
    return passed, issues, flagged_legs


def reality_check_with_retry(parlay_name, parlay, scored_picks, conflicts, flags, max_retries=3):
    """
    Reality Checker retry loop.
    On NEEDS_WORK: rebuild excluding flagged legs. Max 3 attempts.
    On 3rd fail: return None (no viable parlay).
    """
    current = parlay
    excluded_players = set()

    for attempt in range(1, max_retries + 1):
        passed, issues, flagged_legs = reality_check(current)

        if passed:
            current['reality_check'] = {
                'status': 'APPROVED',
                'attempt': attempt,
                'issues': [],
            }
            return current

        print(f"    [{parlay_name}] Attempt {attempt}: NEEDS WORK — {len(issues)} issues")
        for issue in issues:
            print(f"      !! {issue}")

        if attempt >= max_retries:
            current['reality_check'] = {
                'status': 'REJECTED',
                'attempt': attempt,
                'issues': issues,
            }
            return None

        # Exclude flagged players and rebuild
        for idx in flagged_legs:
            legs = current.get('legs', [])
            if idx < len(legs):
                excluded_players.add(legs[idx]['player'])

        # Rebuild from scored_picks excluding flagged players
        filtered = [p for p in scored_picks if p['player'] not in excluded_players]
        target = len(current.get('legs', []))
        require_under = target >= 5

        # Re-correlate with filtered pool
        new_conflicts, new_flags = analyze_correlations(filtered, {})

        # Re-select with indexed pool
        indexed = list(enumerate(filtered))
        indexed.sort(key=lambda x: x[1]['nexus_score'], reverse=True)

        # Blowout indices for filtered pool
        blowout_indices = set()
        for f in new_flags:
            if f['type'] == 'CLUSTER_RISK':
                blowout_indices.update(f.get('indices', []))

        selected = _greedy_select_from_indexed(indexed, new_conflicts, blowout_indices, target, require_under)

        if len(selected) < target:
            current['reality_check'] = {
                'status': 'REJECTED',
                'attempt': attempt,
                'issues': issues + [f'Could not fill {target} legs after excluding {excluded_players}'],
            }
            return None

        conf = _geometric_mean([p['nexus_score'] for p in selected])
        current = {
            'legs': [_nexus_leg(p) for p in selected],
            'confidence': round(conf, 1),
            'description': current['description'] + f' (rebuilt attempt {attempt + 1})',
            'method': current['method'],
        }

    return None


def _greedy_select_from_indexed(indexed, conflicts, blowout_indices, target, require_under):
    """Shared greedy selection logic."""
    selected = []
    selected_indices = set()
    used_games = set()
    used_teams = set()
    combo_count = 0
    blowout_count = 0

    for idx, pick in indexed:
        if idx in selected_indices:
            continue

        game = pick.get('game', '')
        team = _get_player_team(pick)
        stat = pick.get('stat', '').lower()

        if game and game in used_games:
            continue
        if team and team in used_teams:
            continue
        if stat in COMBO_STATS:
            if combo_count >= 1:
                continue
            combo_count += 1
        if idx in blowout_indices:
            if blowout_count >= 2:
                continue
            blowout_count += 1

        blocked = False
        for sel_idx in selected_indices:
            pair = (min(idx, sel_idx), max(idx, sel_idx))
            if conflicts.get(pair) == 'CONFLICT_HIGH':
                blocked = True
                break
        if blocked:
            continue

        selected.append(pick)
        selected_indices.add(idx)
        if game:
            used_games.add(game)
        if team:
            used_teams.add(team)

        if len(selected) >= target:
            break

    return selected


# ═══════════════════════════════════════════════════════════════
# PHASE 7: FINALIZER
# ═══════════════════════════════════════════════════════════════

def finalize_parlays(viable_parlays, all_rejected):
    """
    Format approved parlays, compute implied probability, rank.
    Include rejection reasoning for transparency.
    """
    final = {}

    for name, parlay in viable_parlays.items():
        legs = parlay.get('legs', [])
        # Implied probability: product of individual implied HRs
        implied_probs = []
        for leg in legs:
            # v2: Use weighted L5/L10 HR as base (recency bias)
            l10_hr = leg.get('l10_hit_rate', 60)
            l5_hr = leg.get('l5_hit_rate', 60)
            base_prob = (l5_hr * 0.6 + l10_hr * 0.4) / 100  # v2: L5-weighted

            margin = leg.get('season_avg', 0) - leg.get('line', 0)
            if leg.get('direction') == 'UNDER':
                margin = leg.get('line', 0) - leg.get('season_avg', 0)
            # Margin adjustment: thin margin reduces confidence
            if margin < 0:
                base_prob *= 0.85
            elif margin < 1:
                base_prob *= 0.92
            implied_probs.append(min(0.95, max(0.3, base_prob)))

        parlay_prob = 1.0
        for p in implied_probs:
            parlay_prob *= p

        parlay['implied_probability'] = round(parlay_prob * 100, 1)
        parlay['implied_odds'] = f"+{round((1/parlay_prob - 1) * 100)}" if parlay_prob > 0 else "N/A"
        parlay['leg_count'] = len(legs)
        final[name] = parlay

    # Rank by confidence
    ranked = sorted(final.items(), key=lambda x: x[1].get('confidence', 0), reverse=True)
    for rank, (name, parlay) in enumerate(ranked, 1):
        final[name]['rank'] = rank

    # Add rejection log
    final['_rejection_log'] = {
        'screened_out': len(all_rejected),
        'sample_rejections': [
            {'player': r[0].get('player', '?'), 'stat': r[0].get('stat', '?'), 'reason': r[1]}
            for r in all_rejected[:10]
        ]
    }

    return final


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


def _geometric_mean(scores):
    """Geometric mean of a list of scores."""
    if not scores:
        return 0
    product = 1.0
    for s in scores:
        product *= max(s, 0.01)
    return product ** (1.0 / len(scores))


def _nexus_leg(pick):
    """Format a scored pick as a parlay leg."""
    return {
        'player': pick['player'],
        'stat': pick['stat'],
        'line': pick['line'],
        'direction': pick['direction'],
        'tier': pick['tier'],
        'gap': pick.get('gap', 0),
        'projection': pick.get('projection', 0),
        'l10_hit_rate': pick.get('l10_hit_rate', 0),
        'l5_hit_rate': pick.get('l5_hit_rate', 0),
        'season_avg': pick.get('season_avg', 0),
        'mins_30plus_pct': pick.get('mins_30plus_pct', 0),
        'game': pick.get('game', ''),
        'is_home': pick.get('is_home'),
        'streak': pick.get('streak_status', 'NEUTRAL'),
        'matchup_note': pick.get('matchup_note', ''),
        'nexus_score': pick.get('nexus_score', 0),
        'historical_rating': pick.get('historical_rating', 'NEUTRAL'),
        'historical_detail': pick.get('historical_detail', ''),
        # v2: new fields for reality checker
        'l10_values': pick.get('l10_values', []),
        'l10_floor': pick.get('l10_floor', 0),
        'l10_miss_count': pick.get('l10_miss_count', 0),
        'player_injury_status': pick.get('player_injury_status', ''),
        # v4: scout data + screen tier
        'scout_venue': pick.get('scout_venue', {}),
        'scout_efficiency': pick.get('scout_efficiency', {}),
        'screen_tier': pick.get('screen_tier', ''),
        'screen_multiplier': pick.get('screen_multiplier', 1.0),
        'spread': pick.get('spread'),
        'is_b2b': pick.get('is_b2b', False),
    }


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR — nexus_parlay_pipeline()
# ═══════════════════════════════════════════════════════════════

def nexus_parlay_pipeline(results, GAMES, historical_dir=None):
    """
    Main entry point. Sequential 8-phase pipeline.
    v2: Includes floor check, L5>L10 weighting, combo penalties, usage conflicts.

    Args:
        results: List of analyze_v3 result dicts (full board)
        GAMES: Game context dict
        historical_dir: Optional path to graded results for pattern matching

    Returns:
        Dict of parlay recommendations (or empty dict if no viable parlay)
    """
    import time
    start = time.time()

    print(f"\n{'='*60}")
    print(f"  NEXUS-MICRO v2 PARLAY BUILDER")
    print(f"{'='*60}")
    print(f"  Input: {len(results)} prop lines")

    # ── PHASE 1: HARD SCREENER ──
    print(f"\n  PHASE 1: Hard Screener (v2: S/A/B + floor check)")
    passed, rejected, was_relaxed = gate1(results)
    if was_relaxed:
        print(f"    Gate 1 RELAXED: only {len(passed)} picks at strict, relaxed to pass")
    print(f"    PASSED: {len(passed)} | REJECTED: {len(rejected)}")

    # Show key rejections
    notable_rejects = [(r, reason) for r, reason in rejected
                       if r.get('l10_hit_rate', 0) >= 60 and 'error' not in r]
    for r, reason in notable_rejects[:8]:
        print(f"    KILLED: {r.get('player','?'):20s} {r.get('stat','?').upper():4s} "
              f"{r.get('direction','?'):5s} — {reason}")

    if len(passed) < 3:
        print(f"\n  ABORT: Only {len(passed)} picks survived — need at least 3")
        return {'_rejection_log': {'screened_out': len(rejected), 'reason': 'insufficient picks after screening'}}

    # ── PHASE 2: PROFILE SCORER ──
    print(f"\n  PHASE 2: Profile Scorer (v2: L5>L10, floor safety, combo penalty)")
    for pick in passed:
        pick['nexus_score'] = profile_score(pick)

    passed.sort(key=lambda x: x['nexus_score'], reverse=True)

    viable, gate2_msg = gate2(passed)
    print(f"    Gate 2: {gate2_msg}")
    if not viable:
        print(f"\n  ABORT: {gate2_msg}")
        return {'_rejection_log': {'screened_out': len(rejected), 'reason': gate2_msg}}

    # Print scored picks
    print(f"\n    SCORED PICKS (top 15):")
    for p in passed[:15]:
        season_margin = p.get('season_avg', 0) - p.get('line', 0)
        if p.get('direction') == 'UNDER':
            season_margin = p.get('line', 0) - p.get('season_avg', 0)
        floor_tag = f"floor={p.get('l10_floor',0)}" if p.get('l10_floor') else ""
        miss_tag = f"miss={p.get('l10_miss_count',0)}/10" if p.get('l10_miss_count', 0) > 0 else ""
        extra = f"  {floor_tag} {miss_tag}".rstrip()
        print(f"    [{p['nexus_score']:5.1f}] {p['player']:22s} {p['stat'].upper():4s} "
              f"{p['direction']:5s} {p['line']:5.1f}  gap={p.get('gap',0):+5.1f}  "
              f"L10={p.get('l10_hit_rate',0):3.0f}% L5={p.get('l5_hit_rate',0):3.0f}%  "
              f"margin={season_margin:+.1f}  mins={p.get('mins_30plus_pct',0):.0f}%{extra}")

    # ── PHASE 3: CORRELATION ANALYZER ──
    print(f"\n  PHASE 3: Correlation Analyzer (v2: usage conflicts)")
    conflicts, flags = analyze_correlations(passed, GAMES)

    conflict_count = len([v for v in conflicts.values() if v == 'CONFLICT_HIGH'])
    print(f"    Conflicts: {conflict_count} HIGH, {len([v for v in conflicts.values() if v == 'CONFLICT_MEDIUM'])} MEDIUM")
    for f in flags:
        print(f"    FLAG: {f['type']} — {f['detail']}")

    # ── PHASE 4: PARLAY ARCHITECT ──
    print(f"\n  PHASE 4: Parlay Architect (v2: diversification rules)")
    parlays = build_nexus_parlays(passed, conflicts, flags)

    viable_parlays, rejected_parlays = gate4(parlays)
    print(f"    Built: {len(parlays)} parlays | Viable (conf>=50): {len(viable_parlays)}")
    for name, p in rejected_parlays.items():
        print(f"    REJECTED: {name} (confidence={p.get('confidence', 0)})")

    if not viable_parlays:
        print(f"\n  ABORT: No parlays passed Gate 4 (confidence >= 50)")
        return {'_rejection_log': {'screened_out': len(rejected), 'reason': 'no parlays passed confidence gate'}}

    # ── PHASE 5: HISTORICAL PATTERN MATCHER ──
    print(f"\n  PHASE 5: Historical Pattern Matcher")
    viable_parlays = match_historical_patterns(viable_parlays, historical_dir)
    for name, parlay in viable_parlays.items():
        ratings = [leg.get('historical_rating', '?') for leg in parlay.get('legs', [])]
        print(f"    {name}: {ratings}")

    # ── PHASE 6: REALITY CHECKER ──
    print(f"\n  PHASE 6: Reality Checker v2 (9 tests, max 3 retries)")
    approved = {}
    for name, parlay in viable_parlays.items():
        print(f"\n    Checking {name}...")
        result = reality_check_with_retry(name, parlay, passed, conflicts, flags)
        if result:
            approved[name] = result
            status = result.get('reality_check', {}).get('status', '?')
            attempt = result.get('reality_check', {}).get('attempt', '?')
            print(f"    {name}: {status} (attempt {attempt})")
        else:
            print(f"    {name}: REJECTED after 3 attempts")

    if not approved:
        print(f"\n  ABORT: No parlays survived Reality Checker")
        return {'_rejection_log': {
            'screened_out': len(rejected),
            'reason': 'all parlays rejected by Reality Checker',
        }}

    # ── PHASE 7: FINALIZER ──
    print(f"\n  PHASE 7: Finalizer")
    final = finalize_parlays(approved, rejected)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  NEXUS v2 COMPLETE — {len(approved)} parlays approved in {elapsed:.1f}s")
    print(f"{'='*60}")

    # Print final parlays
    for name, parlay in final.items():
        if name.startswith('_'):
            continue
        rank = parlay.get('rank', '?')
        conf = parlay.get('confidence', 0)
        prob = parlay.get('implied_probability', 0)
        rc = parlay.get('reality_check', {}).get('status', '?')
        print(f"\n  #{rank} {name} [conf={conf} prob={prob}% RC={rc}]")
        print(f"  {parlay['description']}")
        for leg in parlay.get('legs', []):
            hr_tag = f"L10={leg['l10_hit_rate']:3.0f}% L5={leg.get('l5_hit_rate',0):3.0f}%"
            hist = leg.get('historical_rating', '?')
            ns = leg.get('nexus_score', 0)
            floor = leg.get('l10_floor', 0)
            print(f"    [{ns:5.1f}] {leg['player']:22s} {leg['stat'].upper():4s} "
                  f"{leg['direction']:5s} {leg['line']:5.1f}  gap={leg['gap']:+5.1f}  "
                  f"{hr_tag}  floor={floor}  hist={hist}")

    return final


# ═══════════════════════════════════════════════════════════════
# NEXUS v3 — 50-Agent Parlay Builder
# 4 Tiers: 15 Evaluators → 10 Constructors → 15 Devils → 10 Judges
# Output: 1x 3-leg SAFE + 1x 8-leg AGGRESSIVE (non-overlapping)
# ═══════════════════════════════════════════════════════════════

from concurrent.futures import ThreadPoolExecutor, as_completed


def hard_screen_8leg(results):
    """
    Stricter hard screen for 8-leg aggressive pool.
    S/A tier only, base stats only, no B2B, no injuries, high HR thresholds.
    """
    passed = []
    rejected = []

    for r in results:
        if 'error' in r or r.get('tier') == 'SKIP':
            rejected.append((r, 'ERROR/SKIP'))
            continue

        reasons = []

        # Base stats only
        stat = r.get('stat', '').lower()
        if stat in COMBO_STATS:
            reasons.append(f"combo stat {stat} (base only)")

        # High HR thresholds
        l10_hr = r.get('l10_hit_rate', 0)
        l5_hr = r.get('l5_hit_rate', 0)
        if l10_hr < 80:
            reasons.append(f"L10 HR={l10_hr}% (need >=80%)")
        if l5_hr < 60:
            reasons.append(f"L5 HR={l5_hr}% (need >=60%)")

        # Minutes stability
        mins_pct = r.get('mins_30plus_pct', 0)
        if mins_pct < 60:
            reasons.append(f"mins_30plus={mins_pct}% (need >=60%)")

        # Miss count strict
        l10_miss_count = r.get('l10_miss_count', 0)
        if l10_miss_count >= 2:
            reasons.append(f"l10_miss_count={l10_miss_count} (need <2)")

        # Spread < 10
        spread = r.get('spread')
        if spread is not None and abs(spread) >= 10:
            reasons.append(f"spread={spread} (need <10)")

        # No B2B
        if r.get('is_b2b'):
            reasons.append("B2B player")

        # No injury flags
        injury = r.get('player_injury_status', '')
        if injury and injury.strip():
            reasons.append(f"injury={injury}")

        # Season avg clears line by >= 2.0
        season_avg = r.get('season_avg', 0)
        line = r.get('line', 0)
        direction = r.get('direction', '')
        if direction == 'OVER':
            margin = season_avg - line
        else:
            margin = line - season_avg
        if margin < 2.0:
            reasons.append(f"margin={margin:.1f} (need >=2.0)")

        if reasons:
            rejected.append((r, '; '.join(reasons)))
        else:
            passed.append(r)

    return passed, rejected


def _build_with_constraints(pool, target, sort_key, excluded_pairs, max_combo=1):
    """
    Shared builder with diversification constraints.
    No same-game, max 1 player per parlay, max max_combo combo stats, no same-team.
    excluded_pairs: set of (player, stat) to exclude for non-overlap.
    """
    if not pool:
        return []

    sorted_pool = sorted(pool, key=sort_key, reverse=True)
    selected = []
    used_players = set()
    used_games = set()
    used_teams = set()
    combo_count = 0

    for pick in sorted_pool:
        player = pick.get('player', '')
        game = pick.get('game', '')
        team = _get_player_team(pick)
        stat = pick.get('stat', '').lower()

        # Excluded pairs check
        if (player, stat) in excluded_pairs:
            continue

        # No duplicate players
        if player in used_players:
            continue

        # No same-game
        if game and game in used_games:
            continue

        # No same-team
        if team and team in used_teams:
            continue

        # Combo limit
        if stat in COMBO_STATS:
            if combo_count >= max_combo:
                continue
            combo_count += 1

        selected.append(pick)
        used_players.add(player)
        if game:
            used_games.add(game)
        if team:
            used_teams.add(team)

        if len(selected) >= target:
            break

    return selected


# ───────────────────────────────────────────────────────────────
# TIER 1: EVALUATOR AGENTS (15)
# ───────────────────────────────────────────────────────────────

def _v3_evaluate_leg(pick, idx):
    """
    Simplified evaluator agent for v3 pipeline.
    Returns verdict dict with confidence and parlay_worthy.
    """
    verdict = {
        'agent_id': f'v3_eval_{idx}',
        'player': pick.get('player', ''),
        'stat': pick.get('stat', ''),
        'confidence': 50,
        'parlay_worthy': False,
        'reasons_for': [],
        'reasons_against': [],
    }

    score = 50
    abs_gap = pick.get('abs_gap', 0)
    l10_hr = pick.get('l10_hit_rate', 0)
    l5_hr = pick.get('l5_hit_rate', 0)
    season_avg = pick.get('season_avg', 0)
    line = pick.get('line', 0)
    direction = pick.get('direction', '')
    stat = pick.get('stat', '').lower()
    streak = pick.get('streak_status', 'NEUTRAL')

    # Gap analysis
    if abs_gap >= 6:
        score += 18
        verdict['reasons_for'].append(f"Massive gap ({abs_gap:.1f})")
    elif abs_gap >= 4:
        score += 10
        verdict['reasons_for'].append(f"Strong gap ({abs_gap:.1f})")
    elif abs_gap >= 2:
        score += 4
    else:
        score -= 5
        verdict['reasons_against'].append(f"Thin gap ({abs_gap:.1f})")

    # Hit rate
    if l10_hr >= 80:
        score += 14
        verdict['reasons_for'].append(f"Elite L10 HR ({l10_hr}%)")
    elif l10_hr >= 70:
        score += 8
    elif l10_hr < 60:
        score -= 8
        verdict['reasons_against'].append(f"Low L10 HR ({l10_hr}%)")

    if l5_hr >= 80:
        score += 8
        verdict['reasons_for'].append(f"Strong L5 ({l5_hr}%)")
    elif l5_hr < 40:
        score -= 12
        verdict['reasons_against'].append(f"L5 crash ({l5_hr}%)")

    # Margin
    margin = (season_avg - line) if direction == 'OVER' else (line - season_avg)
    if margin >= 3:
        score += 8
    elif margin < 0:
        score -= 10
        verdict['reasons_against'].append(f"Season avg on wrong side (margin={margin:.1f})")

    # Streak alignment
    if (streak == 'HOT' and direction == 'OVER') or (streak == 'COLD' and direction == 'UNDER'):
        score += 4
    elif (streak == 'HOT' and direction == 'UNDER') or (streak == 'COLD' and direction == 'OVER'):
        score -= 6
        verdict['reasons_against'].append(f"Streak contradicts direction")

    # Combo penalty
    if stat in COMBO_STATS:
        score -= 5
        verdict['reasons_against'].append(f"Combo stat ({stat})")

    # Injury
    injury = pick.get('player_injury_status', '')
    if injury and injury.lower() in ['questionable', 'gtd', 'game-time decision', 'doubtful']:
        score -= 18
        verdict['reasons_against'].append(f"Injury status: {injury}")

    # B2B fatigue
    if pick.get('is_b2b'):
        score -= 4

    # Floor check
    l10_floor = pick.get('l10_floor', 0)
    if direction == 'OVER' and l10_floor > 0 and line > 0:
        if l10_floor < line - 5:
            score -= 6
            verdict['reasons_against'].append(f"Floor risk (worst={l10_floor} vs line={line})")

    score = max(0, min(100, score))
    verdict['confidence'] = score
    verdict['parlay_worthy'] = score >= 55
    return verdict


# ───────────────────────────────────────────────────────────────
# TIER 2: CONSTRUCTOR AGENTS (10)
# ───────────────────────────────────────────────────────────────

def _expand_parlay_tiers(name, sort_key, safe_pool, agg_pool, safe_3=None, agg_8=None, max_combo_safe=1, max_combo_agg=0):
    """Build all 4 parlay tiers from a sort_key and pools.
    LOCK 3-leg: highest floor, all CORE, no combos, no B2B, spread < 8
    Main 5-leg: balanced, max 1 combo, requires 1+ UNDER
    Value 4-leg: best profile_score, allows B-tier
    Aggressive 6-8: relaxed, allows FLEX/REACH
    """
    if safe_3 is None:
        safe_3 = _build_with_constraints(safe_pool, 3, sort_key, set(), max_combo=max_combo_safe)
    if agg_8 is None:
        excluded = {(p['player'], p['stat']) for p in safe_3}
        agg_8 = _build_with_constraints(agg_pool, 8, sort_key, excluded, max_combo=max_combo_agg)

    # Main 5-leg: from safe pool, balanced risk, max 1 combo, prefer 1+ UNDER
    excluded_3 = {(p['player'], p['stat']) for p in safe_3}
    main_5_pool = list(safe_pool)
    main_5 = _build_with_constraints(main_5_pool, 5, sort_key, set(), max_combo=1)
    # If no UNDER in main 5, try to swap last leg for best UNDER
    if main_5 and not any(p.get('direction') == 'UNDER' for p in main_5):
        unders = [p for p in safe_pool
                  if p.get('direction') == 'UNDER'
                  and (p['player'], p['stat']) not in {(m['player'], m['stat']) for m in main_5[:-1]}
                  and p.get('game', '') not in {m.get('game', '') for m in main_5[:-1]}]
        if unders:
            unders.sort(key=sort_key, reverse=True)
            main_5[-1] = unders[0]

    # Value 4-leg: best edge/gap picks that didn't make safe, allows B-tier
    value_sort = lambda p: p.get('abs_gap', 0) * 0.5 + p.get('nexus_score', 0) * 0.5
    value_4 = _build_with_constraints(safe_pool, 4, value_sort, excluded_3, max_combo=1)

    return {
        'name': name,
        'safe_3leg': safe_3,
        'main_5leg': main_5,
        'value_4leg': value_4,
        'aggressive_8leg': agg_8,
    }


def constructor_nexus_score(safe_pool, agg_pool):
    """Strategy 1: Sort by nexus_score desc (greedy baseline)."""
    sort_key = lambda p: p.get('nexus_score', 0)
    return _expand_parlay_tiers('nexus_score', sort_key, safe_pool, agg_pool)


def constructor_hit_rate(safe_pool, agg_pool):
    """Strategy 2: Sort by weighted (L5*0.6 + L10*0.4) HR."""
    sort_key = lambda p: p.get('l5_hit_rate', 0) * 0.6 + p.get('l10_hit_rate', 0) * 0.4
    return _expand_parlay_tiers('hit_rate', sort_key, safe_pool, agg_pool)


def constructor_floor_safety(safe_pool, agg_pool):
    """Strategy 3: Sort by floor clearance — worst case still clears."""
    def sort_key(p):
        line = p.get('line', 0)
        direction = p.get('direction', '')
        l10_floor = p.get('l10_floor', 0)
        l10_values = p.get('l10_values', [])
        if direction == 'OVER':
            return l10_floor - line if l10_floor else -100
        else:
            # For UNDER, ceiling is max of L10 values
            ceiling = max(l10_values) if l10_values else line + 100
            return line - ceiling
    return _expand_parlay_tiers('floor_safety', sort_key, safe_pool, agg_pool)


def constructor_game_spread(safe_pool, agg_pool):
    """Strategy 4: 1 best pick per game, maximize game diversity."""
    # Group by game, take best per game
    def _best_per_game(pool):
        by_game = defaultdict(list)
        for p in pool:
            by_game[p.get('game', f'unknown_{id(p)}')].append(p)
        best = []
        for game, picks in by_game.items():
            picks.sort(key=lambda x: x.get('nexus_score', 0), reverse=True)
            best.append(picks[0])
        best.sort(key=lambda x: x.get('nexus_score', 0), reverse=True)
        return best

    game_safe = _best_per_game(safe_pool)
    sort_key = lambda p: p.get('nexus_score', 0)
    game_agg = _best_per_game(agg_pool)
    return _expand_parlay_tiers('game_spread', sort_key, game_safe, game_agg)


def constructor_stat_diversity(safe_pool, agg_pool):
    """Strategy 5: No two legs with same stat type."""
    def _diverse_build(pool, target, excluded):
        sorted_pool = sorted(pool, key=lambda p: p.get('nexus_score', 0), reverse=True)
        selected = []
        used_stats = set()
        used_players = set()
        used_games = set()
        used_teams = set()
        for pick in sorted_pool:
            player = pick.get('player', '')
            game = pick.get('game', '')
            team = _get_player_team(pick)
            stat = pick.get('stat', '')
            if (player, stat) in excluded:
                continue
            if player in used_players or (game and game in used_games) or (team and team in used_teams):
                continue
            if stat in used_stats:
                continue
            selected.append(pick)
            used_stats.add(stat)
            used_players.add(player)
            if game:
                used_games.add(game)
            if team:
                used_teams.add(team)
            if len(selected) >= target:
                break
        return selected

    safe_3 = _diverse_build(safe_pool, 3, set())
    excluded_3 = {(p['player'], p['stat']) for p in safe_3}
    agg_8 = _diverse_build(agg_pool, 8, excluded_3)
    main_5 = _diverse_build(safe_pool, 5, set())
    value_4 = _diverse_build(safe_pool, 4, excluded_3)
    return {'name': 'stat_diversity', 'safe_3leg': safe_3, 'main_5leg': main_5, 'value_4leg': value_4, 'aggressive_8leg': agg_8}


def constructor_under_heavy(safe_pool, agg_pool):
    """Strategy 6: At least 50% UNDER legs."""
    def _under_build(pool, target, excluded):
        unders = [p for p in pool if p.get('direction') == 'UNDER' and (p['player'], p['stat']) not in excluded]
        overs = [p for p in pool if p.get('direction') == 'OVER' and (p['player'], p['stat']) not in excluded]
        unders.sort(key=lambda p: p.get('nexus_score', 0), reverse=True)
        overs.sort(key=lambda p: p.get('nexus_score', 0), reverse=True)
        # Start with unders to fill at least 50%
        under_target = (target + 1) // 2
        combined = unders[:under_target * 2] + overs[:target]
        sort_key = lambda p: p.get('nexus_score', 0)
        return _build_with_constraints(combined, target, sort_key, excluded, max_combo=1 if target <= 3 else 0)

    safe_3 = _under_build(safe_pool, 3, set())
    excluded_3 = {(p['player'], p['stat']) for p in safe_3}
    main_5 = _under_build(safe_pool, 5, set())
    value_4 = _under_build(safe_pool, 4, excluded_3)
    agg_8 = _under_build(agg_pool, 8, excluded_3)
    return {'name': 'under_heavy', 'safe_3leg': safe_3, 'main_5leg': main_5, 'value_4leg': value_4, 'aggressive_8leg': agg_8}


def constructor_anti_blowout(safe_pool, agg_pool):
    """Strategy 7: Only spread < 8 games."""
    def _filter_blowout(pool):
        return [p for p in pool if p.get('spread') is None or abs(p.get('spread', 0)) < 8]

    filtered_safe = _filter_blowout(safe_pool)
    filtered_agg = _filter_blowout(agg_pool)
    sort_key = lambda p: p.get('nexus_score', 0)
    return _expand_parlay_tiers('anti_blowout', sort_key, filtered_safe, filtered_agg)


def constructor_streak_aligned(safe_pool, agg_pool):
    """Strategy 8: Only HOT+OVER or COLD+UNDER (streak-aligned)."""
    def _filter_streaks(pool):
        return [p for p in pool
                if (p.get('streak_status') == 'HOT' and p.get('direction') == 'OVER')
                or (p.get('streak_status') == 'COLD' and p.get('direction') == 'UNDER')]

    streaky_safe = _filter_streaks(safe_pool)
    streaky_agg = _filter_streaks(agg_pool)
    sort_key = lambda p: p.get('nexus_score', 0)
    # Fallback to full pool if not enough streak-aligned picks
    if len(streaky_safe) < 3:
        streaky_safe = safe_pool
    if len(streaky_agg) < 5:
        streaky_agg = agg_pool
    return _expand_parlay_tiers('streak_aligned', sort_key, streaky_safe, streaky_agg)


def constructor_home_focused(safe_pool, agg_pool):
    """Strategy 9: Prefer home players (sort home first, then by score)."""
    sort_key = lambda p: (1 if p.get('is_home') else 0, p.get('nexus_score', 0))
    return _expand_parlay_tiers('home_focused', sort_key, safe_pool, agg_pool)


def constructor_matchup_exploit(safe_pool, agg_pool):
    """Strategy 10: Prioritize picks with favorable matchup."""
    def _matchup_score(p):
        note = (p.get('matchup_note', '') or '').lower()
        score = p.get('nexus_score', 0)
        # Boost for favorable matchup keywords
        for keyword in ['weak', 'poor', 'bottom', 'worst', 'allows', 'gives up', 'vulnerable']:
            if keyword in note:
                score += 5
                break
        return score

    sort_key = _matchup_score
    return _expand_parlay_tiers('matchup_exploit', sort_key, safe_pool, agg_pool)


def constructor_xgb_prob(safe_pool, agg_pool=None):
    """Strategy 11 (main): Sort by XGBoost probability (ML-based hit likelihood).
    Confidence-gated: returns empty if model CV AUC < 0.58."""
    # Confidence gate — check model metadata
    try:
        meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xgb_model_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            cv_auc = meta.get('cv_avg_auc')
            if cv_auc is not None and cv_auc < 0.58:
                # Model too weak — return empty so it doesn't pollute Borda count
                return {'name': 'xgb_prob', 'safe_3leg': [], 'aggressive_8leg': []}
    except Exception:
        pass

    # Only use props that have been scored by XGBoost
    scored = [p for p in safe_pool if p.get('xgb_prob') is not None]
    if len(scored) < 3:
        return {'name': 'xgb_prob', 'safe_3leg': [], 'aggressive_8leg': []}

    sort_key = lambda p: p.get('xgb_prob', 0)
    agg_scored = [p for p in (agg_pool or []) if p.get('xgb_prob') is not None]
    return _expand_parlay_tiers('xgb_prob', sort_key, scored, agg_scored or (agg_pool or []))


ALL_CONSTRUCTORS = [
    constructor_nexus_score, constructor_hit_rate, constructor_floor_safety,
    constructor_game_spread, constructor_stat_diversity, constructor_under_heavy,
    constructor_anti_blowout, constructor_streak_aligned, constructor_home_focused,
    constructor_matchup_exploit, constructor_xgb_prob,
]


# ───────────────────────────────────────────────────────────────
# SHADOW CONSTRUCTORS (strategies 11-20) — backtesting only
# ───────────────────────────────────────────────────────────────

def shadow_blk_stl_anchor(safe_pool, agg_pool=None):
    """Strategy 11: Require at least 1 BLK/STL leg (85%+ historic accuracy)."""
    blk_stl = [p for p in safe_pool if p.get('stat', '') in ('blk', 'stl')]
    others = [p for p in safe_pool if p.get('stat', '') not in ('blk', 'stl')]
    if not blk_stl:
        return {'name': 'blk_stl_anchor', 'safe_3leg': [], 'aggressive_8leg': []}
    # Take best BLK/STL, fill rest from others
    blk_stl.sort(key=lambda p: p.get('nexus_score', 0), reverse=True)
    anchor = [blk_stl[0]]
    excluded = {(blk_stl[0]['player'], blk_stl[0]['stat'])}
    sort_key = lambda p: p.get('nexus_score', 0)
    rest = _build_with_constraints(others, 2, sort_key, excluded)
    return {'name': 'blk_stl_anchor', 'safe_3leg': anchor + rest, 'aggressive_8leg': []}


def shadow_base_stats_only(safe_pool, agg_pool=None):
    """Strategy 12: Zero combo stats — pts/reb/ast/3pm/stl/blk only."""
    filtered = [p for p in safe_pool if p.get('stat', '') in BASE_STATS]
    sort_key = lambda p: p.get('nexus_score', 0)
    safe_3 = _build_with_constraints(filtered, 3, sort_key, set())
    return {'name': 'base_stats_only', 'safe_3leg': safe_3, 'aggressive_8leg': []}


def shadow_perfect_l10(safe_pool, agg_pool=None):
    """Strategy 13: Only picks where L10 HR = 100%."""
    perfect = [p for p in safe_pool if p.get('l10_hit_rate', 0) >= 100]
    sort_key = lambda p: p.get('nexus_score', 0)
    safe_3 = _build_with_constraints(perfect, 3, sort_key, set())
    return {'name': 'perfect_l10', 'safe_3leg': safe_3, 'aggressive_8leg': []}


def shadow_high_floor(safe_pool, agg_pool=None):
    """Strategy 14: Sort by floor clearance (l10_floor - line for OVER, line - l10_ceiling for UNDER)."""
    def sort_key(p):
        line = p.get('line', 0)
        direction = p.get('direction', '')
        if direction == 'OVER':
            floor = p.get('l10_floor', 0)
            return floor - line if floor else -100
        else:
            l10_values = p.get('l10_values', [])
            ceiling = max(l10_values) if l10_values else line + 100
            return line - ceiling
    safe_3 = _build_with_constraints(safe_pool, 3, sort_key, set())
    return {'name': 'high_floor', 'safe_3leg': safe_3, 'aggressive_8leg': []}


def shadow_close_game(safe_pool, agg_pool=None):
    """Strategy 15: Only games with spread <= 4 (starters play full minutes)."""
    close = [p for p in safe_pool if p.get('spread') is not None and abs(p.get('spread', 99)) <= 4]
    if len(close) < 3:
        return {'name': 'close_game', 'safe_3leg': [], 'aggressive_8leg': []}
    sort_key = lambda p: p.get('nexus_score', 0)
    safe_3 = _build_with_constraints(close, 3, sort_key, set())
    return {'name': 'close_game', 'safe_3leg': safe_3, 'aggressive_8leg': []}


def shadow_season_margin(safe_pool, agg_pool=None):
    """Strategy 16: Sort by season avg margin over/under line."""
    def sort_key(p):
        avg = p.get('season_avg', 0)
        line = p.get('line', 0)
        direction = p.get('direction', '')
        if direction == 'OVER':
            return avg - line
        else:
            return line - avg
    safe_3 = _build_with_constraints(safe_pool, 3, sort_key, set())
    return {'name': 'season_margin', 'safe_3leg': safe_3, 'aggressive_8leg': []}


def shadow_under_only(safe_pool, agg_pool=None):
    """Strategy 17: All 3 legs UNDER."""
    unders = [p for p in safe_pool if p.get('direction') == 'UNDER']
    if len(unders) < 3:
        return {'name': 'under_only', 'safe_3leg': [], 'aggressive_8leg': []}
    sort_key = lambda p: p.get('nexus_score', 0)
    safe_3 = _build_with_constraints(unders, 3, sort_key, set())
    return {'name': 'under_only', 'safe_3leg': safe_3, 'aggressive_8leg': []}


def shadow_minutes_iron(safe_pool, agg_pool=None):
    """Strategy 18: Sort by mins_30plus_pct desc (most stable minutes)."""
    sort_key = lambda p: (p.get('mins_30plus_pct', 0), p.get('nexus_score', 0))
    safe_3 = _build_with_constraints(safe_pool, 3, sort_key, set())
    return {'name': 'minutes_iron', 'safe_3leg': safe_3, 'aggressive_8leg': []}


def shadow_anti_trend_reversal(safe_pool, agg_pool=None):
    """Strategy 19: Filter out cooling trends (L10 HR > L5 HR + 20)."""
    stable = [p for p in safe_pool
              if not (p.get('l10_hit_rate', 0) > p.get('l5_hit_rate', 0) + 20)]
    sort_key = lambda p: p.get('nexus_score', 0)
    safe_3 = _build_with_constraints(stable, 3, sort_key, set())
    return {'name': 'anti_trend_reversal', 'safe_3leg': safe_3, 'aggressive_8leg': []}


def shadow_gap_monster(safe_pool, agg_pool=None):
    """Strategy 20: Sort by absolute gap only (biggest projection edge)."""
    sort_key = lambda p: p.get('abs_gap', 0)
    safe_3 = _build_with_constraints(safe_pool, 3, sort_key, set())
    return {'name': 'gap_monster', 'safe_3leg': safe_3, 'aggressive_8leg': []}


def shadow_xgb_prob(safe_pool, agg_pool=None):
    """Shadow strategy: Sort by XGBoost probability (ML-based hit likelihood)."""
    scored = [p for p in safe_pool if p.get('xgb_prob') is not None]
    if len(scored) < 3:
        return {'name': 'xgb_prob_shadow', 'safe_3leg': [], 'aggressive_8leg': []}
    sort_key = lambda p: p.get('xgb_prob', 0)
    safe_3 = _build_with_constraints(scored, 3, sort_key, set())
    return {'name': 'xgb_prob_shadow', 'safe_3leg': safe_3, 'aggressive_8leg': []}


SHADOW_CONSTRUCTORS = [
    shadow_blk_stl_anchor, shadow_base_stats_only, shadow_perfect_l10,
    shadow_high_floor, shadow_close_game, shadow_season_margin,
    shadow_under_only, shadow_minutes_iron, shadow_anti_trend_reversal,
    shadow_gap_monster, shadow_xgb_prob,
]

STRATEGY_DESCRIPTIONS = {
    'nexus_score': 'Greedy by nexus_score (baseline)',
    'hit_rate': 'Sort by weighted L5/L10 HR',
    'floor_safety': 'Worst L10 game still clears line',
    'game_spread': '1 best per game, max diversity',
    'stat_diversity': 'No two legs same stat type',
    'under_heavy': 'At least 50% UNDER legs',
    'anti_blowout': 'Only spread < 8 games',
    'streak_aligned': 'HOT+OVER or COLD+UNDER only',
    'home_focused': 'Prefer home players',
    'matchup_exploit': 'Favorable matchup keywords',
    'blk_stl_anchor': 'Require at least 1 BLK/STL leg (85%+ historic accuracy)',
    'base_stats_only': 'Zero combo stats — pts/reb/ast/3pm/stl/blk only',
    'perfect_l10': 'Only picks where L10 HR = 100%',
    'high_floor': 'Sort by floor clearance (worst case still clears)',
    'close_game': 'Only games with spread <= 4 (starters play full minutes)',
    'season_margin': 'Sort by season avg margin over line',
    'under_only': 'All 3 legs UNDER',
    'minutes_iron': 'Sort by mins_30plus_pct desc (most stable minutes)',
    'anti_trend_reversal': 'Filter out cooling trends (L10 HR > L5 HR + 20)',
    'gap_monster': 'Sort by absolute gap only (biggest projection edge)',
    'xgb_prob': 'Sort by XGBoost ML probability (trained on graded data)',
    'xgb_prob_shadow': 'Shadow: Sort by XGBoost ML probability',
}


# ───────────────────────────────────────────────────────────────
# TIER 3: DEVIL'S ADVOCATE AGENTS (15)
# ───────────────────────────────────────────────────────────────

def _devil_check_legs(legs, test_fn):
    """Helper: run test_fn on each leg, return weakest_leg_idx and list of issues."""
    issues = []
    weakest_idx = None
    weakest_severity = 0
    for i, leg in enumerate(legs):
        severity = test_fn(leg)
        if severity > 0:
            issues.append((i, severity))
            if severity > weakest_severity:
                weakest_severity = severity
                weakest_idx = i
    return issues, weakest_idx


def _devil_evaluate_proposals(proposals, test_fn, test_name):
    """Run a devil test across all proposals, for both 3leg and 8leg."""
    results = {}
    for prop in proposals:
        name = prop['name']
        prop_result = {}
        for parlay_type in ['safe_3leg', 'main_5leg', 'value_4leg', 'aggressive_8leg']:
            legs = prop.get(parlay_type, [])
            if not legs:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
                continue
            issues, weakest_idx = _devil_check_legs(legs, test_fn)
            if issues:
                prop_result[parlay_type] = {
                    'passed': False,
                    'kill_reason': f"{test_name}: {len(issues)} leg(s) flagged",
                    'weakest_leg_idx': weakest_idx if weakest_idx is not None else -1,
                    'flagged_count': len(issues),
                }
            else:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
        results[name] = prop_result
    return results


def devil_blowout(proposals):
    """Flag legs in spread >= 7 games (OVER only)."""
    def test(leg):
        spread = leg.get('spread') if 'spread' in leg else None
        # Legs from _nexus_leg don't have spread, check from source
        if spread is None:
            return 0
        if abs(spread) >= 7 and leg.get('direction') == 'OVER':
            return abs(spread) - 6
        return 0
    return _devil_evaluate_proposals(proposals, test, 'BLOWOUT_RISK')


def devil_fatigue(proposals):
    """Flag B2B players with heavy minutes."""
    def test(leg):
        if leg.get('is_b2b'):
            mins = leg.get('mins_30plus_pct', 0)
            if mins >= 80:
                return 3
            return 1
        return 0
    return _devil_evaluate_proposals(proposals, test, 'FATIGUE_B2B')


def devil_floor_test(proposals):
    """If every player posts worst L10 game, how many legs survive?"""
    results = {}
    for prop in proposals:
        name = prop['name']
        prop_result = {}
        for parlay_type in ['safe_3leg', 'main_5leg', 'value_4leg', 'aggressive_8leg']:
            legs = prop.get(parlay_type, [])
            if not legs:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
                continue
            survivors = 0
            worst_idx = -1
            worst_gap = 999
            for i, leg in enumerate(legs):
                floor = leg.get('l10_floor', 0)
                line = leg.get('line', 0)
                direction = leg.get('direction', '')
                if direction == 'OVER':
                    gap = floor - line
                else:
                    l10_vals = leg.get('l10_values', [])
                    ceiling = max(l10_vals) if l10_vals else line
                    gap = line - ceiling
                if gap >= 0:
                    survivors += 1
                if gap < worst_gap:
                    worst_gap = gap
                    worst_idx = i
            fail_count = len(legs) - survivors
            if fail_count > len(legs) // 2:
                prop_result[parlay_type] = {
                    'passed': False,
                    'kill_reason': f"FLOOR_TEST: only {survivors}/{len(legs)} survive worst case",
                    'weakest_leg_idx': worst_idx,
                }
            else:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
        results[name] = prop_result
    return results


def devil_combo_killer(proposals):
    """Flag any combo stat legs."""
    def test(leg):
        if leg.get('stat', '').lower() in COMBO_STATS:
            return 3
        return 0
    return _devil_evaluate_proposals(proposals, test, 'COMBO_STAT')


def devil_minutes_risk(proposals):
    """Flag if mins_30plus_pct < 70."""
    def test(leg):
        mins = leg.get('mins_30plus_pct', 0)
        if mins < 70:
            return 70 - mins
        return 0
    return _devil_evaluate_proposals(proposals, test, 'MINUTES_RISK')


def devil_injury_cascade(proposals):
    """Flag players with any injury status."""
    def test(leg):
        injury = leg.get('player_injury_status', '')
        if injury and injury.strip():
            return 5
        return 0
    return _devil_evaluate_proposals(proposals, test, 'INJURY_CASCADE')


def devil_opponent_history(proposals):
    """Flag weak matchup notes."""
    def test(leg):
        note = (leg.get('matchup_note', '') or '').lower()
        for keyword in ['strong', 'elite', 'best', 'top', 'locks down']:
            if keyword in note:
                return 3
        return 0
    return _devil_evaluate_proposals(proposals, test, 'TOUGH_MATCHUP')


def devil_thin_margin(proposals):
    """Season avg within 1.0 of line."""
    def test(leg):
        season_avg = leg.get('season_avg', 0)
        line = leg.get('line', 0)
        direction = leg.get('direction', '')
        margin = (season_avg - line) if direction == 'OVER' else (line - season_avg)
        if abs(margin) < 1.0:
            return 3
        return 0
    return _devil_evaluate_proposals(proposals, test, 'THIN_MARGIN')


def devil_correlation_leak(proposals):
    """Check for hidden same-team/same-game correlations in legs."""
    results = {}
    for prop in proposals:
        name = prop['name']
        prop_result = {}
        for parlay_type in ['safe_3leg', 'main_5leg', 'value_4leg', 'aggressive_8leg']:
            legs = prop.get(parlay_type, [])
            if not legs:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
                continue
            games = [leg.get('game', '') for leg in legs]
            teams = [_get_player_team(leg) for leg in legs]
            game_dupes = len(games) - len(set(g for g in games if g))
            team_dupes = len([t for t in teams if t]) - len(set(t for t in teams if t))
            if game_dupes > 0 or team_dupes > 0:
                prop_result[parlay_type] = {
                    'passed': False,
                    'kill_reason': f"CORRELATION: {game_dupes} game dupes, {team_dupes} team dupes",
                    'weakest_leg_idx': 0,
                }
            else:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
        results[name] = prop_result
    return results


def devil_recent_miss(proposals):
    """Check l10_values last 3, flag if 2+ misses."""
    def test(leg):
        l10_values = leg.get('l10_values', [])
        line = leg.get('line', 0)
        direction = leg.get('direction', '')
        if l10_values and len(l10_values) >= 3:
            last_3 = l10_values[:3]
            if direction == 'OVER':
                misses = sum(1 for v in last_3 if v <= line)
            else:
                misses = sum(1 for v in last_3 if v >= line)
            if misses >= 2:
                return misses
        return 0
    return _devil_evaluate_proposals(proposals, test, 'RECENT_MISS')


def devil_trend_reversal(proposals):
    """L10 HR >> L5 HR means cooling trend."""
    def test(leg):
        l10_hr = leg.get('l10_hit_rate', 0)
        l5_hr = leg.get('l5_hit_rate', 0)
        if l10_hr > 0 and l5_hr < l10_hr - 20:
            return l10_hr - l5_hr
        return 0
    return _devil_evaluate_proposals(proposals, test, 'TREND_REVERSAL')


def devil_gtd_cascade(proposals):
    """GTD teammate impact — flag questionable/GTD players."""
    def test(leg):
        injury = leg.get('player_injury_status', '')
        if injury and injury.lower() in ['questionable', 'gtd', 'game-time decision', 'doubtful']:
            return 5
        return 0
    return _devil_evaluate_proposals(proposals, test, 'GTD_CASCADE')


def devil_usage_conflict(proposals):
    """Two players same team chasing same stat."""
    results = {}
    for prop in proposals:
        name = prop['name']
        prop_result = {}
        for parlay_type in ['safe_3leg', 'main_5leg', 'value_4leg', 'aggressive_8leg']:
            legs = prop.get(parlay_type, [])
            if not legs:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
                continue
            # Check for same-team + same-stat combinations
            team_stat_map = {}
            conflict_found = False
            weakest_idx = -1
            for i, leg in enumerate(legs):
                team = _get_player_team(leg)
                stat = leg.get('stat', '')
                if team and stat:
                    key = (team, stat)
                    if key in team_stat_map:
                        conflict_found = True
                        # Weakest is the one with lower nexus_score
                        prev_idx = team_stat_map[key]
                        if leg.get('nexus_score', 0) < legs[prev_idx].get('nexus_score', 0):
                            weakest_idx = i
                        else:
                            weakest_idx = prev_idx
                    else:
                        team_stat_map[key] = i
            if conflict_found:
                prop_result[parlay_type] = {
                    'passed': False,
                    'kill_reason': 'USAGE_CONFLICT: same team + same stat',
                    'weakest_leg_idx': weakest_idx,
                }
            else:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
        results[name] = prop_result
    return results


def devil_line_trap(proposals):
    """Season avg barely clears line (within 0.5)."""
    def test(leg):
        season_avg = leg.get('season_avg', 0)
        line = leg.get('line', 0)
        direction = leg.get('direction', '')
        margin = (season_avg - line) if direction == 'OVER' else (line - season_avg)
        if 0 <= margin < 0.5:
            return 3
        return 0
    return _devil_evaluate_proposals(proposals, test, 'LINE_TRAP')


def devil_consensus(proposals):
    """How many constructors picked each leg? < 3 appearances = marginal."""
    # Build a frequency map of (player, stat) across all proposals
    leg_freq = defaultdict(int)
    for prop in proposals:
        for parlay_type in ['safe_3leg', 'main_5leg', 'value_4leg', 'aggressive_8leg']:
            for leg in prop.get(parlay_type, []):
                key = (leg.get('player', ''), leg.get('stat', ''))
                leg_freq[key] += 1

    results = {}
    for prop in proposals:
        name = prop['name']
        prop_result = {}
        for parlay_type in ['safe_3leg', 'main_5leg', 'value_4leg', 'aggressive_8leg']:
            legs = prop.get(parlay_type, [])
            if not legs:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
                continue
            marginal = []
            weakest_idx = -1
            min_freq = 999
            for i, leg in enumerate(legs):
                key = (leg.get('player', ''), leg.get('stat', ''))
                freq = leg_freq.get(key, 0)
                if freq < 3:
                    marginal.append(i)
                if freq < min_freq:
                    min_freq = freq
                    weakest_idx = i
            if len(marginal) > len(legs) // 2:
                prop_result[parlay_type] = {
                    'passed': False,
                    'kill_reason': f"LOW_CONSENSUS: {len(marginal)}/{len(legs)} legs in <3 proposals",
                    'weakest_leg_idx': weakest_idx,
                }
            else:
                prop_result[parlay_type] = {'passed': True, 'kill_reason': '', 'weakest_leg_idx': -1}
        results[name] = prop_result
    return results


ALL_DEVILS = [
    devil_blowout, devil_fatigue, devil_floor_test, devil_combo_killer,
    devil_minutes_risk, devil_injury_cascade, devil_opponent_history,
    devil_thin_margin, devil_correlation_leak, devil_recent_miss,
    devil_trend_reversal, devil_gtd_cascade, devil_usage_conflict,
    devil_line_trap, devil_consensus,
]


# ───────────────────────────────────────────────────────────────
# TIER 4: JUDGE AGENTS (10)
# ───────────────────────────────────────────────────────────────

def _count_devil_flags(devil_results, proposal_name, parlay_type):
    """Count how many devil tests flagged this proposal's parlay_type."""
    flagged = 0
    for devil_name, devil_result in devil_results.items():
        prop_data = devil_result.get(proposal_name, {})
        pt_data = prop_data.get(parlay_type, {})
        if not pt_data.get('passed', True):
            flagged += 1
    return flagged


def _judge_score_proposals(proposals, devil_results, parlay_type, score_fn):
    """Generic judge: apply score_fn to each proposal, return ranked list."""
    scored = []
    for prop in proposals:
        name = prop['name']
        legs = prop.get(parlay_type, [])
        if not legs:
            scored.append((name, -100))
            continue
        flags = _count_devil_flags(devil_results, name, parlay_type)
        score = score_fn(prop, legs, flags, devil_results, parlay_type)
        scored.append((name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def judge_conservative(proposals, devil_results, parlay_type='safe_3leg'):
    """50% devil survival rate, 30% min HR across legs, 20% avg nexus score."""
    def score_fn(prop, legs, flags, dr, pt):
        total_devils = len(ALL_DEVILS)
        survival_rate = (total_devils - flags) / total_devils if total_devils > 0 else 0
        min_hr = min((leg.get('l5_hit_rate', 0) * 0.6 + leg.get('l10_hit_rate', 0) * 0.4) for leg in legs) if legs else 0
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        return survival_rate * 50 + (min_hr / 100) * 30 + (avg_ns / 100) * 20
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_aggressive(proposals, devil_results, parlay_type='aggressive_8leg'):
    """40% avg nexus score, 30% avg gap size, 30% devil survival."""
    def score_fn(prop, legs, flags, dr, pt):
        total_devils = len(ALL_DEVILS)
        survival_rate = (total_devils - flags) / total_devils if total_devils > 0 else 0
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        avg_gap = sum(abs(leg.get('gap', 0)) for leg in legs) / len(legs) if legs else 0
        return (avg_ns / 100) * 40 + min(avg_gap / 10, 1.0) * 30 + survival_rate * 30
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_consistency(proposals, devil_results, parlay_type='safe_3leg'):
    """50% avg L5 HR, 30% avg L10 HR, 20% floor safety."""
    def score_fn(prop, legs, flags, dr, pt):
        avg_l5 = sum(leg.get('l5_hit_rate', 0) for leg in legs) / len(legs) if legs else 0
        avg_l10 = sum(leg.get('l10_hit_rate', 0) for leg in legs) / len(legs) if legs else 0
        # Floor: how many legs have floor > line (for OVER) or ceiling < line (for UNDER)
        floor_pass = 0
        for leg in legs:
            floor = leg.get('l10_floor', 0)
            line = leg.get('line', 0)
            direction = leg.get('direction', '')
            if direction == 'OVER' and floor > line:
                floor_pass += 1
            elif direction == 'UNDER':
                vals = leg.get('l10_values', [])
                if vals and max(vals) < line:
                    floor_pass += 1
        floor_pct = floor_pass / len(legs) if legs else 0
        return (avg_l5 / 100) * 50 + (avg_l10 / 100) * 30 + floor_pct * 20
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_diversity(proposals, devil_results, parlay_type='safe_3leg'):
    """Penalize stat/direction concentration."""
    def score_fn(prop, legs, flags, dr, pt):
        stats = [leg.get('stat', '') for leg in legs]
        directions = [leg.get('direction', '') for leg in legs]
        unique_stats = len(set(stats))
        unique_dirs = len(set(directions))
        stat_diversity = unique_stats / len(legs) if legs else 0
        dir_diversity = unique_dirs / max(len(set(directions)), 1)
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        return stat_diversity * 40 + dir_diversity * 30 + (avg_ns / 100) * 30
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_historical(proposals, devil_results, parlay_type='safe_3leg'):
    """Weight by historical_rating (STRONG=4, NEUTRAL=2, WEAK=0, DANGEROUS=-2)."""
    rating_map = {'STRONG': 4, 'NEUTRAL': 2, 'WEAK': 0, 'DANGEROUS': -2}
    def score_fn(prop, legs, flags, dr, pt):
        total = sum(rating_map.get(leg.get('historical_rating', 'NEUTRAL'), 2) for leg in legs)
        max_possible = 4 * len(legs) if legs else 1
        hist_score = (total + 2 * len(legs)) / (max_possible + 2 * len(legs)) if legs else 0.5
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        return hist_score * 60 + (avg_ns / 100) * 40
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_risk_adjusted(proposals, devil_results, parlay_type='safe_3leg'):
    """Penalize number of devil flags per proposal."""
    def score_fn(prop, legs, flags, dr, pt):
        total_devils = len(ALL_DEVILS)
        penalty = flags / total_devils if total_devils > 0 else 0
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        return (avg_ns / 100) * 60 - penalty * 40
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_matchup(proposals, devil_results, parlay_type='safe_3leg'):
    """Weight opponent defense quality from matchup notes."""
    def score_fn(prop, legs, flags, dr, pt):
        matchup_score = 0
        for leg in legs:
            note = (leg.get('matchup_note', '') or '').lower()
            if any(w in note for w in ['weak', 'poor', 'bottom', 'worst', 'allows', 'gives up']):
                matchup_score += 3
            elif any(w in note for w in ['strong', 'elite', 'best', 'top']):
                matchup_score -= 2
            else:
                matchup_score += 1  # neutral
        max_ms = 3 * len(legs) if legs else 1
        normalized = (matchup_score + 2 * len(legs)) / (max_ms + 2 * len(legs)) if legs else 0.5
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        return normalized * 50 + (avg_ns / 100) * 50
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_momentum(proposals, devil_results, parlay_type='safe_3leg'):
    """Weight streak alignment (HOT+OVER or COLD+UNDER = good)."""
    def score_fn(prop, legs, flags, dr, pt):
        aligned = 0
        misaligned = 0
        for leg in legs:
            streak = leg.get('streak', leg.get('streak_status', 'NEUTRAL'))
            direction = leg.get('direction', '')
            if (streak == 'HOT' and direction == 'OVER') or (streak == 'COLD' and direction == 'UNDER'):
                aligned += 1
            elif (streak == 'HOT' and direction == 'UNDER') or (streak == 'COLD' and direction == 'OVER'):
                misaligned += 1
        momentum = (aligned - misaligned) / len(legs) if legs else 0
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        return (momentum + 1) / 2 * 50 + (avg_ns / 100) * 50  # normalize -1..1 to 0..1
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_floor(proposals, devil_results, parlay_type='safe_3leg'):
    """Weight worst-case floor scenario (l10_floor vs line)."""
    def score_fn(prop, legs, flags, dr, pt):
        floor_gaps = []
        for leg in legs:
            floor = leg.get('l10_floor', 0)
            line = leg.get('line', 0)
            direction = leg.get('direction', '')
            if direction == 'OVER':
                floor_gaps.append(floor - line)
            else:
                vals = leg.get('l10_values', [])
                ceiling = max(vals) if vals else line
                floor_gaps.append(line - ceiling)
        avg_floor_gap = sum(floor_gaps) / len(floor_gaps) if floor_gaps else -10
        # Normalize: +5 or better = 1.0, -5 or worse = 0.0
        normalized = max(0, min(1.0, (avg_floor_gap + 5) / 10))
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        return normalized * 60 + (avg_ns / 100) * 40
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_consensus(proposals, devil_results, parlay_type='safe_3leg'):
    """Wisdom of crowds: how many constructors' legs overlap with this proposal."""
    # Build frequency map
    leg_freq = defaultdict(int)
    for prop in proposals:
        for leg in prop.get(parlay_type, []):
            key = (leg.get('player', ''), leg.get('stat', ''))
            leg_freq[key] += 1

    def score_fn(prop, legs, flags, dr, pt):
        total_freq = sum(leg_freq.get((leg.get('player', ''), leg.get('stat', '')), 0) for leg in legs)
        max_freq = len(proposals) * len(legs) if legs else 1
        consensus_pct = total_freq / max_freq if max_freq > 0 else 0
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        return consensus_pct * 50 + (avg_ns / 100) * 50
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


# ───────────────────────────────────────────────────────────────
# BORDA COUNT CONSENSUS
# ───────────────────────────────────────────────────────────────

def borda_count_consensus(judge_results):
    """
    Aggregate judge rankings via Borda count.
    Each judge ranks proposals 1..N (best=1).
    Sum ranks across all judges. Lowest sum wins.
    Returns sorted list of (proposal_name, rank_sum).
    """
    rank_sums = defaultdict(float)
    for judge_name, rankings in judge_results.items():
        for rank, (proposal_name, score) in enumerate(rankings, 1):
            rank_sums[proposal_name] += rank

    sorted_results = sorted(rank_sums.items(), key=lambda x: x[1])
    return sorted_results


# ───────────────────────────────────────────────────────────────
# NEXUS v3 MAIN ORCHESTRATOR
# ───────────────────────────────────────────────────────────────

def nexus_v3_pipeline(results, GAMES, historical_dir=None):
    """
    NEXUS v3 — 50-Agent Parlay Builder
    4 tiers: 15 Evaluators -> 10 Constructors -> 15 Devils -> 10 Judges
    Output: 1x 3-leg SAFE + 1x 8-leg AGGRESSIVE (non-overlapping)
    """
    import time
    start = time.time()

    print(f"\n{'='*65}")
    print(f"  NEXUS v3 — 50-AGENT PARLAY BUILDER")
    print(f"{'='*65}")
    print(f"  Input: {len(results)} prop lines")
    print(f"  Architecture: 15 Evaluators -> 10 Constructors -> 15 Devils -> 10 Judges")

    # ── PHASE 1: DUAL HARD SCREEN ──
    print(f"\n  PHASE 1: Dual Hard Screen")
    safe_passed, safe_rejected, was_relaxed = gate1(results)
    if was_relaxed:
        print(f"    Safe pool RELAXED: strict screen too tight")
    print(f"    Safe pool: {len(safe_passed)} passed | {len(safe_rejected)} rejected")

    agg_passed, agg_rejected = hard_screen_8leg(results)
    print(f"    Aggressive pool: {len(agg_passed)} passed | {len(agg_rejected)} rejected")

    if len(safe_passed) < 3:
        print(f"\n  ABORT: Only {len(safe_passed)} picks in safe pool — need at least 3")
        return {
            'nexus_v3_safe_3leg': None,
            'nexus_v3_aggressive_8leg': None,
            '_rejection_log': {'screened_out': len(safe_rejected), 'reason': 'insufficient picks after screening'},
        }

    # ── PHASE 2: PROFILE SCORING ──
    print(f"\n  PHASE 2: Profile Scoring")
    all_picks = list({id(p): p for p in safe_passed + agg_passed}.values())
    for pick in all_picks:
        if 'nexus_score' not in pick or pick['nexus_score'] == 0:
            pick['nexus_score'] = profile_score(pick)

    safe_passed.sort(key=lambda x: x.get('nexus_score', 0), reverse=True)
    agg_passed.sort(key=lambda x: x.get('nexus_score', 0), reverse=True)

    viable, gate2_msg = gate2(safe_passed)
    print(f"    Gate 2: {gate2_msg}")
    if not viable:
        print(f"\n  ABORT: {gate2_msg}")
        return {
            'nexus_v3_safe_3leg': None,
            'nexus_v3_aggressive_8leg': None,
            '_rejection_log': {'screened_out': len(safe_rejected), 'reason': gate2_msg},
        }

    # Print top candidates
    print(f"\n    TOP CANDIDATES (safe pool):")
    for p in safe_passed[:12]:
        margin = (p.get('season_avg', 0) - p.get('line', 0)) if p.get('direction') == 'OVER' else (p.get('line', 0) - p.get('season_avg', 0))
        print(f"    [{p.get('nexus_score',0):5.1f}] {p.get('player','?'):22s} {p.get('stat','?').upper():4s} "
              f"{p.get('direction','?'):5s} {p.get('line',0):5.1f}  "
              f"L10={p.get('l10_hit_rate',0):3.0f}% L5={p.get('l5_hit_rate',0):3.0f}%  margin={margin:+.1f}")

    # ── TIER 1: EVALUATOR AGENTS (15) ──
    print(f"\n  TIER 1: Evaluator Agents (15)")
    eval_candidates = safe_passed[:15]
    verdicts = {}
    workers = min(15, (os.cpu_count() or 4) + 4)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for idx, pick in enumerate(eval_candidates):
            f = executor.submit(_v3_evaluate_leg, pick, idx)
            futures[f] = pick

        for f in as_completed(futures):
            pick = futures[f]
            try:
                verdict = f.result()
                key = (pick.get('player', ''), pick.get('stat', ''))
                verdicts[key] = verdict
            except Exception as e:
                print(f"    Evaluator error: {e}")

    # Filter based on evaluator verdicts
    worthy_count = sum(1 for v in verdicts.values() if v.get('parlay_worthy'))
    print(f"    Evaluated {len(verdicts)} candidates: {worthy_count} parlay-worthy")

    # Remove picks that evaluators reject with low confidence
    for pick in list(safe_passed):
        key = (pick.get('player', ''), pick.get('stat', ''))
        if key in verdicts:
            v = verdicts[key]
            if not v.get('parlay_worthy') and v.get('confidence', 50) < 40:
                safe_passed.remove(pick)
                print(f"    EVAL KILL: {pick.get('player','?')} {pick.get('stat','?').upper()} (conf={v['confidence']})")

    for pick in list(agg_passed):
        key = (pick.get('player', ''), pick.get('stat', ''))
        if key in verdicts:
            v = verdicts[key]
            if not v.get('parlay_worthy') and v.get('confidence', 50) < 40:
                agg_passed.remove(pick)

    print(f"    Post-eval: {len(safe_passed)} safe, {len(agg_passed)} aggressive")

    if len(safe_passed) < 3:
        print(f"\n  ABORT: Only {len(safe_passed)} picks after evaluation")
        return {
            'nexus_v3_safe_3leg': None,
            'nexus_v3_aggressive_8leg': None,
            '_rejection_log': {'screened_out': len(safe_rejected), 'reason': 'insufficient picks after evaluation'},
        }

    # ── TIER 2: CONSTRUCTOR AGENTS (10) ──
    print(f"\n  TIER 2: Constructor Agents (10)")
    proposals = []
    workers = min(10, (os.cpu_count() or 4) + 4)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for constructor in ALL_CONSTRUCTORS:
            f = executor.submit(constructor, safe_passed, agg_passed)
            futures[f] = constructor.__name__

        for f in as_completed(futures):
            name = futures[f]
            try:
                result = f.result()
                proposals.append(result)
                safe_count = len(result.get('safe_3leg', []))
                agg_count = len(result.get('aggressive_8leg', []))
                print(f"    {name}: safe={safe_count} legs, aggressive={agg_count} legs")
            except Exception as e:
                print(f"    Constructor {name} error: {e}")

    if not proposals:
        print(f"\n  ABORT: No constructors produced proposals")
        return {
            'nexus_v3_safe_3leg': None,
            'nexus_v3_aggressive_8leg': None,
            '_rejection_log': {'screened_out': len(safe_rejected), 'reason': 'no constructor proposals'},
        }

    # ── TIER 3: DEVIL'S ADVOCATE AGENTS (15) ──
    print(f"\n  TIER 3: Devil's Advocate Agents (15)")
    devil_results = {}
    workers = min(15, (os.cpu_count() or 4) + 4)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for devil_fn in ALL_DEVILS:
            f = executor.submit(devil_fn, proposals)
            futures[f] = devil_fn.__name__

        for f in as_completed(futures):
            name = futures[f]
            try:
                result = f.result()
                devil_results[name] = result
                # Count how many proposals got flagged
                flagged = sum(1 for prop_name, prop_data in result.items()
                              if any(not prop_data.get(pt, {}).get('passed', True)
                                     for pt in ['safe_3leg', 'main_5leg', 'value_4leg', 'aggressive_8leg']))
                if flagged > 0:
                    print(f"    {name}: flagged {flagged}/{len(proposals)} proposals")
            except Exception as e:
                print(f"    Devil {name} error: {e}")

    # ── TIER 4: JUDGE AGENTS (10) ──
    print(f"\n  TIER 4: Judge Agents (10)")

    # Judge functions for safe (3-leg)
    safe_judges = [
        ('conservative', judge_conservative),
        ('consistency', judge_consistency),
        ('diversity', judge_diversity),
        ('historical', judge_historical),
        ('risk_adjusted', judge_risk_adjusted),
        ('matchup', judge_matchup),
        ('momentum', judge_momentum),
        ('floor', judge_floor),
        ('consensus', judge_consensus),
        ('aggressive_as_safe', judge_aggressive),  # 10th: aggressive judge evaluating safe parlays
    ]

    # Judge functions for aggressive (8-leg)
    agg_judges = [
        ('aggressive', judge_aggressive),
        ('conservative', judge_conservative),
        ('consistency', judge_consistency),
        ('diversity', judge_diversity),
        ('historical', judge_historical),
        ('risk_adjusted', judge_risk_adjusted),
        ('matchup', judge_matchup),
        ('momentum', judge_momentum),
        ('floor', judge_floor),
        ('consensus', judge_consensus),
    ]

    safe_judge_results = {}
    agg_judge_results = {}
    workers = min(20, (os.cpu_count() or 4) + 4)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for judge_name, judge_fn in safe_judges:
            f = executor.submit(judge_fn, proposals, devil_results, 'safe_3leg')
            futures[f] = ('safe', judge_name)
        for judge_name, judge_fn in agg_judges:
            f = executor.submit(judge_fn, proposals, devil_results, 'aggressive_8leg')
            futures[f] = ('agg', judge_name)

        for f in as_completed(futures):
            pool_type, judge_name = futures[f]
            try:
                result = f.result()
                if pool_type == 'safe':
                    safe_judge_results[judge_name] = result
                else:
                    agg_judge_results[judge_name] = result
                # Print top pick from each judge
                if result:
                    top_name, top_score = result[0]
                    print(f"    [{pool_type:4s}] {judge_name:20s} -> {top_name} (score={top_score:.1f})")
            except Exception as e:
                print(f"    Judge {judge_name} error: {e}")

    # ── BORDA COUNT CONSENSUS ──
    print(f"\n  CONSENSUS: Borda Count Aggregation")
    safe_ranking = borda_count_consensus(safe_judge_results)
    agg_ranking = borda_count_consensus(agg_judge_results)

    print(f"    SAFE 3-leg ranking:")
    for name, rank_sum in safe_ranking:
        print(f"      {name:20s} rank_sum={rank_sum:.0f}")
    print(f"    AGGRESSIVE 8-leg ranking:")
    for name, rank_sum in agg_ranking:
        print(f"      {name:20s} rank_sum={rank_sum:.0f}")

    # ── BUILD WINNING PARLAYS ──
    print(f"\n  BUILDING FINAL PARLAYS")

    # Find winning proposals
    def _find_proposal(name, proposals):
        for p in proposals:
            if p['name'] == name:
                return p
        return None

    # Safe 3-leg: try winner, then runner-ups
    safe_parlay = None
    for candidate_name, _ in safe_ranking:
        prop = _find_proposal(candidate_name, proposals)
        if not prop or not prop.get('safe_3leg'):
            continue
        legs = prop['safe_3leg']
        if len(legs) < 3:
            continue
        # Build parlay dict for reality check
        leg_dicts = [_nexus_leg(p) for p in legs]
        conf = _geometric_mean([p.get('nexus_score', 50) for p in legs])
        candidate_parlay = {
            'legs': leg_dicts,
            'confidence': round(conf, 1),
            'description': f'NEXUS v3 SAFE 3-Leg (strategy: {candidate_name})',
            'method': f'nexus_v3_{candidate_name}',
        }
        print(f"\n    Trying safe 3-leg from '{candidate_name}'...")

        # Run reality check
        result = reality_check_with_retry(
            f'nexus_v3_safe_{candidate_name}', candidate_parlay,
            safe_passed, {}, []
        )
        if result:
            safe_parlay = result
            safe_parlay['constructor'] = candidate_name
            print(f"    Safe 3-leg APPROVED (strategy: {candidate_name})")
            break
        else:
            print(f"    Safe 3-leg from '{candidate_name}' REJECTED, trying next...")

    # Aggressive 8-leg: try winner, then runner-ups
    agg_parlay = None
    # Get players already used in safe parlay for exclusion
    safe_players = set()
    if safe_parlay:
        for leg in safe_parlay.get('legs', []):
            safe_players.add(leg.get('player', ''))

    for candidate_name, _ in agg_ranking:
        prop = _find_proposal(candidate_name, proposals)
        if not prop or not prop.get('aggressive_8leg'):
            continue
        legs = prop['aggressive_8leg']
        # Filter out players already in safe parlay
        legs = [l for l in legs if l.get('player', '') not in safe_players]
        if len(legs) < 5:  # fallback minimum
            continue
        # Take up to 8
        legs = legs[:8]
        leg_dicts = [_nexus_leg(p) for p in legs]
        conf = _geometric_mean([p.get('nexus_score', 50) for p in legs])
        candidate_parlay = {
            'legs': leg_dicts,
            'confidence': round(conf, 1),
            'description': f'NEXUS v3 AGGRESSIVE {len(legs)}-Leg (strategy: {candidate_name})',
            'method': f'nexus_v3_{candidate_name}',
        }
        print(f"\n    Trying aggressive {len(legs)}-leg from '{candidate_name}'...")

        result = reality_check_with_retry(
            f'nexus_v3_agg_{candidate_name}', candidate_parlay,
            agg_passed, {}, []
        )
        if result:
            agg_parlay = result
            agg_parlay['constructor'] = candidate_name
            print(f"    Aggressive {len(legs)}-leg APPROVED (strategy: {candidate_name})")
            break
        else:
            print(f"    Aggressive from '{candidate_name}' REJECTED, trying next...")

    # ── HISTORICAL PATTERNS ──
    if safe_parlay or agg_parlay:
        print(f"\n  HISTORICAL PATTERN MATCHING")
        temp_parlays = {}
        if safe_parlay:
            temp_parlays['nexus_v3_safe_3leg'] = safe_parlay
        if agg_parlay:
            temp_parlays['nexus_v3_aggressive_8leg'] = agg_parlay
        temp_parlays = match_historical_patterns(temp_parlays, historical_dir)
        if safe_parlay:
            safe_parlay = temp_parlays.get('nexus_v3_safe_3leg', safe_parlay)
        if agg_parlay:
            agg_parlay = temp_parlays.get('nexus_v3_aggressive_8leg', agg_parlay)

    # ── SHADOW PARLAYS (20 strategies for backtesting) ──
    print(f"\n  SHADOW PARLAYS: Collecting 20 strategy proposals")
    shadow_parlays = []

    # Strategies 1-10: pull safe_3leg from existing constructor proposals
    for prop in proposals:
        raw_name = prop.get('name', 'unknown')
        strategy_name = f'nexus_{raw_name}'
        legs = prop.get('safe_3leg', [])
        if len(legs) >= 3:
            leg_dicts = [_nexus_leg(p) for p in legs[:3]]
            conf = _geometric_mean([p.get('nexus_score', 50) for p in legs[:3]])
            shadow_parlays.append({
                'strategy_name': strategy_name,
                'strategy_id': f'nexus_{raw_name}',
                'strategy_description': STRATEGY_DESCRIPTIONS.get(raw_name, ''),
                'legs': leg_dicts,
                'confidence': round(conf, 1),
                'legs_total': 3,
                'result': None,
                'legs_hit': None,
            })
        else:
            shadow_parlays.append({
                'strategy_name': strategy_name,
                'strategy_id': f'nexus_{raw_name}',
                'strategy_description': STRATEGY_DESCRIPTIONS.get(raw_name, ''),
                'legs': [_nexus_leg(p) for p in legs] if legs else [],
                'confidence': 0,
                'legs_total': len(legs),
                'result': 'no_build',
                'legs_hit': None,
            })

    # Strategies 11-20: run shadow constructors against safe_passed pool
    for shadow_fn in SHADOW_CONSTRUCTORS:
        try:
            result = shadow_fn(safe_passed)
            raw_name = result.get('name', 'unknown')
            strategy_name = f'nexus_{raw_name}'
            legs = result.get('safe_3leg', [])
            if len(legs) >= 3:
                leg_dicts = [_nexus_leg(p) for p in legs[:3]]
                conf = _geometric_mean([p.get('nexus_score', 50) for p in legs[:3]])
                shadow_parlays.append({
                    'strategy_name': strategy_name,
                    'strategy_id': f'nexus_{raw_name}',
                    'strategy_description': STRATEGY_DESCRIPTIONS.get(raw_name, ''),
                    'legs': leg_dicts,
                    'confidence': round(conf, 1),
                    'legs_total': 3,
                    'result': None,
                    'legs_hit': None,
                })
            else:
                shadow_parlays.append({
                    'strategy_name': strategy_name,
                    'strategy_id': f'nexus_{raw_name}',
                    'strategy_description': STRATEGY_DESCRIPTIONS.get(raw_name, ''),
                    'legs': [_nexus_leg(p) for p in legs] if legs else [],
                    'confidence': 0,
                    'legs_total': len(legs),
                    'result': 'no_build',
                    'legs_hit': None,
                })
        except Exception as e:
            print(f"    Shadow constructor {shadow_fn.__name__} error: {e}")

    built = sum(1 for s in shadow_parlays if s.get('result') != 'no_build')
    print(f"    Collected {len(shadow_parlays)} shadow parlays ({built} built, {len(shadow_parlays) - built} no_build)")

    # ── FINALIZE ──
    print(f"\n  FINALIZING")
    output_parlays = {}
    if safe_parlay:
        output_parlays['nexus_v3_safe_3leg'] = safe_parlay
    if agg_parlay:
        output_parlays['nexus_v3_aggressive_8leg'] = agg_parlay

    final = finalize_parlays(output_parlays, safe_rejected + agg_rejected)

    # Attach shadow parlays to output
    final['_shadow_parlays'] = shadow_parlays

    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print(f"  NEXUS v3 COMPLETE — {len(output_parlays)} parlays + {len(shadow_parlays)} shadows in {elapsed:.1f}s")
    print(f"  Agents deployed: 15 evaluators + 10 constructors + 15 devils + 10 judges = 50")
    print(f"{'='*65}")

    # Print final parlays
    for name, parlay in final.items():
        if name.startswith('_'):
            continue
        rank = parlay.get('rank', '?')
        conf = parlay.get('confidence', 0)
        prob = parlay.get('implied_probability', 0)
        rc = parlay.get('reality_check', {}).get('status', '?')
        constructor = parlay.get('constructor', '?')
        print(f"\n  #{rank} {name} [conf={conf} prob={prob}% RC={rc} strategy={constructor}]")
        print(f"  {parlay.get('description', '')}")
        for leg in parlay.get('legs', []):
            hr_tag = f"L10={leg.get('l10_hit_rate',0):3.0f}% L5={leg.get('l5_hit_rate',0):3.0f}%"
            ns = leg.get('nexus_score', 0)
            floor = leg.get('l10_floor', 0)
            hist = leg.get('historical_rating', '?')
            print(f"    [{ns:5.1f}] {leg['player']:22s} {leg['stat'].upper():4s} "
                  f"{leg['direction']:5s} {leg['line']:5.1f}  gap={leg['gap']:+5.1f}  "
                  f"{hr_tag}  floor={floor}  hist={hist}")

    return final


# ═══════════════════════════════════════════════════════════════
# NEXUS v4 — 27 Meaningful Agents + Guaranteed Output
# Soft Screen + 3 Scouts + 4 Evaluators + 11 Constructors + 15 Devils + 5 Judges
# ═══════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────
# SOFT SCREEN (replaces binary hard screen)
# ───────────────────────────────────────────────────────────────

def soft_screen(results):
    """
    Tiered penalty system replacing binary hard screen.
    CORE (1.0x): passes all filters
    FLEX (0.85x): fails exactly 1 soft filter
    REACH (0.70x): fails exactly 2 soft filters
    KILL: fails 3+ soft filters OR any hard kill
    """
    screened = []
    killed = []

    for r in results:
        if 'error' in r or r.get('tier') == 'SKIP':
            killed.append((r, 'ERROR/SKIP'))
            continue

        hard_kills = []
        soft_fails = []

        # ── HARD KILLS (non-negotiable) ──
        injury = r.get('player_injury_status', '')
        if injury and injury.lower() in ['out', 'doubtful']:
            hard_kills.append(f"injury={injury}")

        mins_pct = r.get('mins_30plus_pct', 0)
        if mins_pct < 40:
            hard_kills.append(f"mins_30plus={mins_pct}% (<40)")

        l10_hr = r.get('l10_hit_rate', 0)
        if l10_hr < 40:
            hard_kills.append(f"L10HR={l10_hr}% (<40)")

        if hard_kills:
            killed.append((r, 'HARD KILL: ' + '; '.join(hard_kills)))
            continue

        # ── SOFT FILTERS (failing 1-2 is OK) ──
        if 50 <= mins_pct < 60:
            soft_fails.append(f"mins_30plus={mins_pct}% (50-59)")

        if 55 <= l10_hr < 60:
            soft_fails.append(f"L10HR={l10_hr}% (55-59)")

        l5_hr = r.get('l5_hit_rate', 0)
        if 30 <= l5_hr < 40:
            soft_fails.append(f"L5HR={l5_hr}% (30-39)")

        stat = r.get('stat', '').lower()
        abs_gap = r.get('abs_gap', 0)
        if stat not in COMBO_STATS and 1.0 <= abs_gap < 1.5:
            soft_fails.append(f"gap={abs_gap:.1f} (1.0-1.49)")

        l10_miss_count = r.get('l10_miss_count', 0)
        if l10_miss_count == 3:
            soft_fails.append(f"miss_count={l10_miss_count}")

        # GTD/Questionable as soft filter (not hard kill)
        if injury and injury.lower() in ['questionable', 'gtd', 'game-time decision']:
            soft_fails.append(f"injury={injury}")

        # Season avg margin check (soft)
        season_avg = r.get('season_avg', 0)
        line = r.get('line', 0)
        direction = r.get('direction', '')
        if direction == 'OVER' and season_avg < line - 1.0:
            soft_fails.append(f"OVER but season_avg={season_avg} < line-1")
        elif direction == 'UNDER' and season_avg > line + 1.0:
            soft_fails.append(f"UNDER but season_avg={season_avg} > line+1")

        # L10 floor safety (soft)
        l10_floor = r.get('l10_floor', 0)
        if direction == 'OVER' and l10_floor > 0 and line > 0:
            floor_gap = l10_floor - line
            if floor_gap < -5:
                soft_fails.append(f"floor_risk: worst={l10_floor} vs line={line}")

        n_soft = len(soft_fails)

        if n_soft >= 3:
            killed.append((r, f'SOFT KILL ({n_soft} fails): ' + '; '.join(soft_fails)))
            continue

        # Assign tier and multiplier
        if n_soft == 0:
            screen_tier = 'CORE'
            screen_multiplier = 1.0
        elif n_soft == 1:
            screen_tier = 'FLEX'
            screen_multiplier = 0.85
        else:  # n_soft == 2
            screen_tier = 'REACH'
            screen_multiplier = 0.70

        r['screen_tier'] = screen_tier
        r['screen_multiplier'] = screen_multiplier
        r['screen_soft_fails'] = soft_fails
        screened.append(r)

    return screened, killed


def soft_screen_aggressive(results):
    """
    Soft screen for aggressive pool. Higher baselines but allows B-tier and combos with gap >= 5.
    """
    screened = []
    killed = []

    for r in results:
        if 'error' in r or r.get('tier') == 'SKIP':
            killed.append((r, 'ERROR/SKIP'))
            continue

        hard_kills = []
        soft_fails = []

        # Hard kills
        injury = r.get('player_injury_status', '')
        if injury and injury.lower() in ['out', 'doubtful', 'questionable', 'gtd', 'game-time decision']:
            hard_kills.append(f"injury={injury}")

        mins_pct = r.get('mins_30plus_pct', 0)
        if mins_pct < 50:
            hard_kills.append(f"mins={mins_pct}%")

        l10_hr = r.get('l10_hit_rate', 0)
        if l10_hr < 60:
            hard_kills.append(f"L10HR={l10_hr}%")

        l5_hr = r.get('l5_hit_rate', 0)
        if l5_hr < 40:
            hard_kills.append(f"L5HR={l5_hr}%")

        # Allow combos with gap >= 5.0 (relaxed from base-stats-only)
        stat = r.get('stat', '').lower()
        abs_gap = r.get('abs_gap', 0)
        if stat in COMBO_STATS and abs_gap < 5.0:
            hard_kills.append(f"combo gap={abs_gap:.1f} (<5.0)")

        if r.get('is_b2b'):
            soft_fails.append("B2B")

        spread = r.get('spread')
        if spread is not None and abs(spread) >= 10:
            soft_fails.append(f"spread={spread}")

        l10_miss_count = r.get('l10_miss_count', 0)
        if l10_miss_count >= 2:
            soft_fails.append(f"miss_count={l10_miss_count}")

        season_avg = r.get('season_avg', 0)
        line = r.get('line', 0)
        direction = r.get('direction', '')
        if direction == 'OVER':
            margin = season_avg - line
        else:
            margin = line - season_avg
        if margin < 2.0:
            soft_fails.append(f"margin={margin:.1f}")

        if hard_kills:
            killed.append((r, 'HARD KILL: ' + '; '.join(hard_kills)))
            continue

        n_soft = len(soft_fails)
        if n_soft >= 3:
            killed.append((r, f'SOFT KILL ({n_soft}): ' + '; '.join(soft_fails)))
            continue

        if n_soft == 0:
            r['screen_tier'] = 'CORE'
            r['screen_multiplier'] = 1.0
        elif n_soft == 1:
            r['screen_tier'] = 'FLEX'
            r['screen_multiplier'] = 0.85
        else:
            r['screen_tier'] = 'REACH'
            r['screen_multiplier'] = 0.70

        r['screen_soft_fails'] = soft_fails
        screened.append(r)

    return screened, killed


# ───────────────────────────────────────────────────────────────
# SCOUT AGENTS (3) — Enrich picks with new data, zero API calls
# ───────────────────────────────────────────────────────────────

def scout_efficiency(picks):
    """
    Scout 1: Extract efficiency data from cached game log DataFrames.
    PLUS_MINUS, PF columns are fetched but were never used before v4.
    """
    for pick in picks:
        pick['scout_efficiency'] = {
            'l10_avg_plus_minus': pick.get('l10_avg_plus_minus', 0),
            'l10_avg_pf': pick.get('l10_avg_pf', 0),
            'foul_trouble_risk': pick.get('foul_trouble_risk', False),
            'efficiency_trend': pick.get('efficiency_trend', 0),
        }
    return picks


def scout_venue(picks, GAMES):
    """
    Scout 2: Static venue lookup — altitude + timezone travel.
    Zero API calls.
    """
    try:
        from venue_data import VENUE_MAP, TZ_ORDINAL, get_travel_distance
    except ImportError:
        # Fallback if venue_data.py not available
        for pick in picks:
            pick['scout_venue'] = {'venue_altitude': 0, 'travel_zone_diff': 0, 'travel_distance': 0}
        return picks

    for pick in picks:
        game = pick.get('game', '')
        is_home = pick.get('is_home')
        altitude = 0
        zone_diff = 0
        travel_dist = 0

        if game and '@' in game:
            parts = game.split('@')
            away_abr = parts[0] if len(parts) > 1 else ''
            home_abr = parts[1] if len(parts) > 1 else ''

            home_venue = VENUE_MAP.get(home_abr, {})
            altitude = home_venue.get('altitude', 0)

            away_venue = VENUE_MAP.get(away_abr, {})
            if away_venue and home_venue:
                away_tz = TZ_ORDINAL.get(away_venue.get('tz', 'ET'), 0)
                home_tz = TZ_ORDINAL.get(home_venue.get('tz', 'ET'), 0)
                zone_diff = abs(away_tz - home_tz)

            # Travel distance
            try:
                travel_dist = get_travel_distance(away_abr, home_abr)
            except Exception:
                travel_dist = 0

        pick['scout_venue'] = {
            'venue_altitude': altitude,
            'travel_zone_diff': zone_diff,
            'travel_distance': travel_dist,
        }
    return picks


def scout_context(picks, GAMES):
    """
    Scout 3: Derive context from existing board + research data.
    Rest advantage, clinch status from GAMES dict.
    """
    for pick in picks:
        game_key = pick.get('game', '')
        rest_advantage = 0
        clinch_status = 'fighting'

        if game_key and game_key in GAMES:
            gctx = GAMES[game_key]
            # Rest advantage from research data
            is_home = pick.get('is_home')
            if is_home is True:
                home_rest = gctx.get('home_rest_days', 1)
                away_rest = gctx.get('away_rest_days', 1)
                rest_advantage = home_rest - away_rest
            elif is_home is False:
                home_rest = gctx.get('home_rest_days', 1)
                away_rest = gctx.get('away_rest_days', 1)
                rest_advantage = away_rest - home_rest

            # Clinch status if available in research
            if gctx.get('clinched'):
                clinch_status = 'clinched'
            elif gctx.get('eliminated'):
                clinch_status = 'eliminated'

        pick['scout_context'] = {
            'rest_advantage': rest_advantage,
            'clinch_status': clinch_status,
        }
    return picks


# ───────────────────────────────────────────────────────────────
# v4 EVALUATOR AGENTS (4 competing philosophies)
# ───────────────────────────────────────────────────────────────

def eval_statistician(pick):
    """Evaluator A: Pure numbers — gap, HR consistency, margin, stddev."""
    score = 0
    abs_gap = pick.get('abs_gap', 0)
    l10_hr = pick.get('l10_hit_rate', 0)
    l5_hr = pick.get('l5_hit_rate', 0)
    season_avg = pick.get('season_avg', 0)
    line = pick.get('line', 0)
    direction = pick.get('direction', '')

    # Gap magnitude (35%)
    score += min(35, abs_gap * 5)

    # L10/L5 HR consistency (30%)
    hr_avg = (l10_hr + l5_hr) / 2
    score += min(30, hr_avg * 0.3)

    # Season avg margin (25%)
    margin = (season_avg - line) if direction == 'OVER' else (line - season_avg)
    if margin >= 4:
        score += 25
    elif margin >= 2:
        score += 18
    elif margin >= 0:
        score += 10
    else:
        score += 0

    # L10 standard deviation — lower = better (10%)
    l10_values = pick.get('l10_values', [])
    if l10_values and len(l10_values) >= 5:
        import numpy as np
        std = float(np.std(l10_values))
        # Lower std = higher score (max 10 if std < 2)
        score += max(0, min(10, 10 - std))

    score = max(0, min(100, score))
    parlay_worthy = score >= 55
    return {'agent': 'statistician', 'confidence': round(score, 1), 'parlay_worthy': parlay_worthy}


def eval_matchup_hunter(pick):
    """Evaluator B: Context-driven — defense, pace, opponent history, spread."""
    score = 50  # baseline

    # Opponent defensive rank (35%)
    matchup_note = (pick.get('matchup_note', '') or '').lower()
    if any(w in matchup_note for w in ['weak', 'poor', 'bottom', 'worst']):
        score += 15
    elif any(w in matchup_note for w in ['elite', 'strong', 'best', 'top']):
        score -= 15

    # Pace context (20%)
    if 'pace-up' in matchup_note:
        if pick.get('direction') == 'OVER':
            score += 10
        else:
            score -= 5
    elif 'pace-down' in matchup_note:
        if pick.get('direction') == 'UNDER':
            score += 10
        else:
            score -= 5

    # Opponent-specific history (20%)
    opp_hist = pick.get('opponent_history')
    if isinstance(opp_hist, dict):
        opp_hr = opp_hist.get('hit_rate', 50)
        if opp_hr >= 75:
            score += 12
        elif opp_hr >= 50:
            score += 5
        elif opp_hr < 30:
            score -= 10

    # Spread/blowout risk (15%)
    spread = pick.get('spread')
    if spread is not None:
        if abs(spread) >= 10 and pick.get('direction') == 'OVER':
            score -= 8
        elif abs(spread) <= 4:
            score += 5

    # Home/away (10%)
    if pick.get('is_home') is True:
        score += 5
    elif pick.get('is_home') is False:
        score -= 2

    score = max(0, min(100, score))
    return {'agent': 'matchup_hunter', 'confidence': round(score, 1), 'parlay_worthy': score >= 55}


def eval_floor_master(pick):
    """Evaluator C: Worst-case focused — floor, miss count, minutes consistency."""
    score = 50

    l10_floor = pick.get('l10_floor', 0)
    line = pick.get('line', 0)
    direction = pick.get('direction', '')
    l10_miss_count = pick.get('l10_miss_count', 0)

    # L10 floor vs line gap (40%)
    if direction == 'OVER':
        floor_gap = l10_floor - line
        if floor_gap >= 0:
            score += 25  # even worst game clears
        elif floor_gap >= -2:
            score += 10
        elif floor_gap >= -3:
            score += 0
        else:
            score -= 15  # worst game WAY below line — REJECT
            return {'agent': 'floor_master', 'confidence': max(0, score), 'parlay_worthy': False}
    else:
        l10_values = pick.get('l10_values', [])
        if l10_values:
            ceiling = max(l10_values)
            ceil_gap = line - ceiling
            if ceil_gap >= 0:
                score += 25
            elif ceil_gap >= -2:
                score += 10
            else:
                score -= 10

    # L10 miss count (25%)
    if l10_miss_count == 0:
        score += 20
    elif l10_miss_count == 1:
        score += 12
    elif l10_miss_count == 2:
        score += 5
    else:
        score -= 10

    # Minutes consistency (20%): min(L10 mins) vs season avg
    mins_pct = pick.get('mins_30plus_pct', 50)
    if mins_pct >= 90:
        score += 15
    elif mins_pct >= 75:
        score += 10
    elif mins_pct >= 60:
        score += 5
    else:
        score -= 5

    # Foul trouble risk from scout (15%)
    if pick.get('foul_trouble_risk'):
        score -= 8
    else:
        score += 5

    score = max(0, min(100, score))
    return {'agent': 'floor_master', 'confidence': round(score, 1), 'parlay_worthy': score >= 55}


def eval_momentum(pick):
    """Evaluator D: Trend/recency — L3 vs L10, L5 vs season, plus_minus trend."""
    score = 50

    l3_avg = pick.get('l3_avg', 0)
    l5_avg = pick.get('l5_avg', 0)
    l10_avg = pick.get('l10_avg', 0)
    season_avg = pick.get('season_avg', 0)
    direction = pick.get('direction', '')

    # L3 vs L10 trajectory (30%)
    if l10_avg > 0:
        l3_delta = (l3_avg - l10_avg) / l10_avg * 100
        if direction == 'OVER':
            if l3_delta > 10:
                score += 18
            elif l3_delta > 0:
                score += 8
            elif l3_delta < -15:
                score -= 12
        else:  # UNDER
            if l3_delta < -10:
                score += 18
            elif l3_delta < 0:
                score += 8
            elif l3_delta > 15:
                score -= 12

    # L5 vs season trajectory (25%)
    if season_avg > 0:
        l5_delta = (l5_avg - season_avg) / season_avg * 100
        if direction == 'OVER':
            if l5_delta > 5:
                score += 12
            elif l5_delta < -10:
                score -= 8
        else:
            if l5_delta < -5:
                score += 12
            elif l5_delta > 10:
                score -= 8

    # Plus/minus trend from scout (20%)
    eff_trend = pick.get('efficiency_trend', 0)
    if eff_trend > 3:
        score += 10
    elif eff_trend < -3:
        score -= 8

    # Minutes trend (15%)
    season_mins = pick.get('season_mins_avg', 0)
    l5_mins = pick.get('l5_mins_avg', 0)
    if season_mins > 0 and l5_mins > 0:
        mins_ratio = l5_mins / season_mins
        if mins_ratio >= 1.05:
            score += 8
        elif mins_ratio < 0.85:
            score -= 10

    # Streak alignment (10%)
    streak = pick.get('streak_status', 'NEUTRAL')
    if (streak == 'HOT' and direction == 'OVER') or (streak == 'COLD' and direction == 'UNDER'):
        score += 6
    elif (streak == 'HOT' and direction == 'UNDER') or (streak == 'COLD' and direction == 'OVER'):
        score -= 6

    score = max(0, min(100, score))
    return {'agent': 'momentum', 'confidence': round(score, 1), 'parlay_worthy': score >= 55}


V4_EVALUATORS = [eval_statistician, eval_matchup_hunter, eval_floor_master, eval_momentum]


# ───────────────────────────────────────────────────────────────
# v4 JUDGE AGENTS (5 merged from 10)
# ───────────────────────────────────────────────────────────────

def judge_v4_safety(proposals, devil_results, parlay_type='safe_3leg'):
    """Judge 1: merges conservative + risk_adjusted + floor."""
    def score_fn(prop, legs, flags, dr, pt):
        total_devils = len(ALL_DEVILS)
        survival_rate = (total_devils - flags) / total_devils if total_devils > 0 else 0
        min_l5 = min((leg.get('l5_hit_rate', 0) for leg in legs), default=0)
        floor_gaps = []
        for leg in legs:
            floor = leg.get('l10_floor', 0)
            line = leg.get('line', 0)
            direction = leg.get('direction', '')
            if direction == 'OVER':
                floor_gaps.append(floor - line)
            else:
                vals = leg.get('l10_values', [])
                ceiling = max(vals) if vals else line
                floor_gaps.append(line - ceiling)
        min_floor = min(floor_gaps) if floor_gaps else -10
        norm_floor = max(0, min(1.0, (min_floor + 5) / 10))
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        return survival_rate * 40 + (min_l5 / 100) * 30 + norm_floor * 20 + (avg_ns / 100) * 10
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_v4_edge(proposals, devil_results, parlay_type='safe_3leg'):
    """Judge 2: merges aggressive — gap size + nexus_score + HR."""
    def score_fn(prop, legs, flags, dr, pt):
        avg_gap = sum(abs(leg.get('gap', 0)) for leg in legs) / len(legs) if legs else 0
        avg_ns = sum(leg.get('nexus_score', 0) for leg in legs) / len(legs) if legs else 0
        avg_hr = sum(leg.get('l10_hit_rate', 0) for leg in legs) / len(legs) if legs else 0
        stats = set(leg.get('stat', '') for leg in legs)
        diversity_bonus = len(stats) / len(legs) if legs else 0
        return min(avg_gap / 10, 1.0) * 40 + (avg_ns / 100) * 30 + (avg_hr / 100) * 20 + diversity_bonus * 10
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_v4_context(proposals, devil_results, parlay_type='safe_3leg'):
    """Judge 3: merges matchup + momentum — matchup quality, streaks, home advantage, venue."""
    def score_fn(prop, legs, flags, dr, pt):
        matchup_score = 0
        streak_score = 0
        home_score = 0
        for leg in legs:
            note = (leg.get('matchup_note', '') or '').lower()
            if any(w in note for w in ['weak', 'poor', 'bottom', 'worst']):
                matchup_score += 3
            elif any(w in note for w in ['strong', 'elite', 'best', 'top']):
                matchup_score -= 2
            else:
                matchup_score += 1
            streak = leg.get('streak', leg.get('streak_status', 'NEUTRAL'))
            direction = leg.get('direction', '')
            if (streak == 'HOT' and direction == 'OVER') or (streak == 'COLD' and direction == 'UNDER'):
                streak_score += 1
            elif (streak == 'HOT' and direction == 'UNDER') or (streak == 'COLD' and direction == 'OVER'):
                streak_score -= 1
            if leg.get('is_home') is True:
                home_score += 1
            # Venue: favor picks away from altitude
            scout = leg.get('scout_venue', {})
            if scout.get('venue_altitude', 0) > 4000 and not leg.get('is_home'):
                matchup_score -= 1  # visiting at altitude is harder

        n = len(legs) if legs else 1
        spread_score = 0
        for leg in legs:
            sp = leg.get('spread')
            if sp is not None and abs(sp) <= 5:
                spread_score += 1
        norm_matchup = (matchup_score + 2 * n) / (5 * n) if n else 0.5
        norm_streak = (streak_score + n) / (2 * n) if n else 0.5
        norm_home = home_score / n if n else 0.5
        norm_spread = spread_score / n if n else 0.5
        return norm_matchup * 30 + norm_streak * 25 + norm_home * 20 + norm_spread * 15 + 10 * 0.5
    return _judge_score_proposals(proposals, devil_results, parlay_type, score_fn)


def judge_v4_historical(proposals, devil_results, parlay_type='safe_3leg'):
    """Judge 4: kept as-is — 60% historical profile rating, 40% avg nexus_score."""
    return judge_historical(proposals, devil_results, parlay_type)


def judge_v4_consensus(proposals, devil_results, parlay_type='safe_3leg'):
    """Judge 5: kept as-is — constructor agreement + avg nexus_score."""
    return judge_consensus(proposals, devil_results, parlay_type)


V4_JUDGES = [
    ('safety', judge_v4_safety),
    ('edge', judge_v4_edge),
    ('context', judge_v4_context),
    ('historical', judge_v4_historical),
    ('consensus', judge_v4_consensus),
]


# ───────────────────────────────────────────────────────────────
# NEXUS v4 MAIN ORCHESTRATOR — GUARANTEED OUTPUT
# ───────────────────────────────────────────────────────────────

def nexus_v4_pipeline(results, GAMES, historical_dir=None):
    """
    NEXUS v4 — 27 Meaningful Agents + Guaranteed Output
    Soft Screen + 3 Scouts + 4 Evaluators + 11 Constructors + 15 Devils + 5 Judges
    Cascade fallback: full v4 → relaxed v4 → survival build
    """
    import time
    start = time.time()

    print(f"\n{'='*65}")
    print(f"  NEXUS v4 — 27-AGENT PARLAY BUILDER (Guaranteed Output)")
    print(f"{'='*65}")
    print(f"  Input: {len(results)} prop lines")
    print(f"  Architecture: 3 Scouts + 4 Evaluators + 11 Constructors + 15 Devils + 5 Judges")

    # ── PHASE 1: SOFT SCREEN (replaces binary hard screen) ──
    print(f"\n  PHASE 1: Soft Screen (CORE/FLEX/REACH/KILL)")
    safe_passed, safe_killed = soft_screen(results)
    agg_passed, agg_killed = soft_screen_aggressive(results)

    core_count = sum(1 for p in safe_passed if p.get('screen_tier') == 'CORE')
    flex_count = sum(1 for p in safe_passed if p.get('screen_tier') == 'FLEX')
    reach_count = sum(1 for p in safe_passed if p.get('screen_tier') == 'REACH')
    print(f"    Safe pool: {len(safe_passed)} passed (CORE={core_count}, FLEX={flex_count}, REACH={reach_count}) | {len(safe_killed)} killed")
    print(f"    Aggressive pool: {len(agg_passed)} passed | {len(agg_killed)} killed")

    # Show notable kills
    for r, reason in safe_killed[:5]:
        if r.get('l10_hit_rate', 0) >= 60 and 'error' not in r:
            print(f"    KILLED: {r.get('player','?'):20s} {r.get('stat','?').upper():4s} — {reason}")

    # ── PHASE 2: SCOUT AGENTS (3) ──
    print(f"\n  PHASE 2: Scout Agents (3) — enriching data")
    all_picks = list({id(p): p for p in safe_passed + agg_passed}.values())
    all_picks = scout_efficiency(all_picks)
    all_picks = scout_venue(all_picks, GAMES)
    all_picks = scout_context(all_picks, GAMES)
    print(f"    Enriched {len(all_picks)} picks with efficiency, venue, context data")

    # ── PHASE 3: PROFILE SCORING (with screen_multiplier applied) ──
    print(f"\n  PHASE 3: Profile Scoring (screen_multiplier applied)")
    for pick in all_picks:
        base_score = profile_score(pick)
        multiplier = pick.get('screen_multiplier', 1.0)
        pick['nexus_score'] = round(base_score * multiplier, 1)

    safe_passed.sort(key=lambda x: x.get('nexus_score', 0), reverse=True)
    agg_passed.sort(key=lambda x: x.get('nexus_score', 0), reverse=True)

    # Print top candidates
    print(f"\n    TOP CANDIDATES (safe pool):")
    for p in safe_passed[:12]:
        margin = (p.get('season_avg', 0) - p.get('line', 0)) if p.get('direction') == 'OVER' else (p.get('line', 0) - p.get('season_avg', 0))
        tier_tag = p.get('screen_tier', '?')
        print(f"    [{p.get('nexus_score',0):5.1f}] {p.get('player','?'):22s} {p.get('stat','?').upper():4s} "
              f"{p.get('direction','?'):5s} {p.get('line',0):5.1f}  "
              f"L10={p.get('l10_hit_rate',0):3.0f}% L5={p.get('l5_hit_rate',0):3.0f}%  "
              f"margin={margin:+.1f}  [{tier_tag}]")

    # ── PHASE 4: EVALUATOR AGENTS (4 competing) ──
    print(f"\n  PHASE 4: Evaluator Agents (4 competing philosophies)")
    eval_results = {}
    for pick in safe_passed:
        key = (pick.get('player', ''), pick.get('stat', ''))
        verdicts = [ev(pick) for ev in V4_EVALUATORS]
        worthy_count = sum(1 for v in verdicts if v['parlay_worthy'])
        avg_conf = sum(v['confidence'] for v in verdicts) / len(verdicts)
        eval_results[key] = {
            'verdicts': verdicts,
            'worthy_count': worthy_count,
            'avg_confidence': round(avg_conf, 1),
        }
        pick['eval_agreement_count'] = worthy_count
        pick['eval_max_confidence'] = max(v['confidence'] for v in verdicts)

    # Determine evaluator gating threshold
    min_worthy = 2  # majority: at least 2 of 4
    eval_passed = [p for p in safe_passed
                   if eval_results.get((p.get('player',''), p.get('stat','')), {}).get('worthy_count', 0) >= min_worthy]

    if len(eval_passed) < 8:
        # Relax to 1 of 4
        min_worthy = 1
        eval_passed = [p for p in safe_passed
                       if eval_results.get((p.get('player',''), p.get('stat','')), {}).get('worthy_count', 0) >= min_worthy]
        print(f"    Relaxed evaluator gate to 1/4 (only {len([p for p in safe_passed if eval_results.get((p.get('player',''), p.get('stat','')), {}).get('worthy_count', 0) >= 2])} at 2/4)")

    worthy_total = len(eval_passed)
    killed_by_eval = len(safe_passed) - worthy_total
    print(f"    {worthy_total} parlay-worthy (>= {min_worthy}/4 evaluators) | {killed_by_eval} killed")

    # Show evaluator disagreements
    for p in safe_passed[:8]:
        key = (p.get('player', ''), p.get('stat', ''))
        er = eval_results.get(key, {})
        verdicts = er.get('verdicts', [])
        if verdicts:
            tags = [f"{v['agent'][0].upper()}:{v['confidence']:.0f}" for v in verdicts]
            worthy = er.get('worthy_count', 0)
            print(f"    {p.get('player','?'):22s} {p.get('stat','?').upper():4s} "
                  f"votes={worthy}/4  [{' '.join(tags)}]")

    # Use eval_passed for constructors, but keep full safe_passed for fallback
    build_pool = eval_passed if len(eval_passed) >= 3 else safe_passed

    # ── ATTEMPT 1: Full v4 pipeline ──
    output = _v4_build_parlays(build_pool, agg_passed, GAMES, historical_dir, safe_killed + agg_killed, 'nexus_v4')

    if output and any(v is not None for k, v in output.items() if not k.startswith('_')):
        elapsed = time.time() - start
        print(f"\n{'='*65}")
        print(f"  NEXUS v4 COMPLETE (full pipeline) — {elapsed:.1f}s")
        print(f"  Agents: 3 scouts + 4 evaluators + 11 constructors + 15 devils + 5 judges = 27")
        print(f"{'='*65}")
        _v4_print_output(output)
        return output

    # ── ATTEMPT 2: Relaxed v4 — drop evaluator gating, drop devils, widen screen ──
    print(f"\n  FALLBACK 1: Relaxed v4 (no evaluator gate, no devils)")
    relaxed_pool = safe_passed  # all soft-screened picks, no evaluator filter
    if len(relaxed_pool) < 3:
        # Re-screen with even wider tolerance (allow 3 soft fails)
        relaxed_pool = []
        for r in results:
            if 'error' in r or r.get('tier') == 'SKIP':
                continue
            tier = r.get('tier', 'F')
            if tier in ('D', 'F'):
                continue
            injury = r.get('player_injury_status', '')
            if injury and injury.lower() in ['out', 'doubtful']:
                continue
            if r.get('mins_30plus_pct', 0) < 40:
                continue
            if r.get('l10_hit_rate', 0) < 40:
                continue
            r['screen_tier'] = 'REACH'
            r['screen_multiplier'] = 0.70
            r['nexus_score'] = round(profile_score(r) * 0.70, 1)
            relaxed_pool.append(r)

    output = _v4_build_parlays_relaxed(relaxed_pool, safe_killed, 'nexus_v4_relaxed')

    if output and any(v is not None for k, v in output.items() if not k.startswith('_')):
        elapsed = time.time() - start
        print(f"\n{'='*65}")
        print(f"  NEXUS v4 COMPLETE (relaxed) — {elapsed:.1f}s")
        print(f"{'='*65}")
        _v4_print_output(output)
        return output

    # ── ATTEMPT 3: Survival build — top 3 by raw profile_score ──
    print(f"\n  FALLBACK 2: Survival build (top 3 by profile_score)")
    survival_pool = []
    for r in results:
        if 'error' in r or r.get('tier') == 'SKIP':
            continue
        tier = r.get('tier', 'F')
        if tier in ('D', 'F'):
            continue
        injury = r.get('player_injury_status', '')
        if injury and injury.lower() in ['out', 'doubtful']:
            continue
        r['nexus_score'] = profile_score(r)
        survival_pool.append(r)

    survival_pool.sort(key=lambda x: x.get('nexus_score', 0), reverse=True)

    # Build top 3 with diversity constraints
    safe_3 = _build_with_constraints(survival_pool, 3, lambda p: p.get('nexus_score', 0), set())

    if len(safe_3) >= 3:
        leg_dicts = [_nexus_leg(p) for p in safe_3]
        conf = _geometric_mean([p.get('nexus_score', 50) for p in safe_3])
        safe_parlay = {
            'legs': leg_dicts,
            'confidence': round(conf, 1),
            'description': 'NEXUS v4 SURVIVAL 3-Leg (top picks by profile score, no screening)',
            'method': 'nexus_v4_survival',
            'reality_check': {'status': 'SURVIVAL_MODE', 'attempt': 0, 'issues': ['Used survival fallback']},
        }
        output = {
            'nexus_v4_safe_3leg': safe_parlay,
            'nexus_v4_main_5leg': None,
            'nexus_v4_value_4leg': None,
            'nexus_v4_aggressive_8leg': None,
            '_rejection_log': {'screened_out': len(results) - len(survival_pool), 'reason': 'survival fallback'},
        }

        # Still build shadow parlays
        shadow = _build_shadow_parlays(survival_pool)
        output['_shadow_parlays'] = shadow

        elapsed = time.time() - start
        print(f"\n{'='*65}")
        print(f"  NEXUS v4 COMPLETE (survival) — {elapsed:.1f}s")
        print(f"{'='*65}")
        _v4_print_output(output)
        return output

    # Absolute last resort — should never get here
    elapsed = time.time() - start
    print(f"\n  NEXUS v4 FAILED — no viable picks in entire board ({elapsed:.1f}s)")
    return {
        'nexus_v4_safe_3leg': None,
        'nexus_v4_main_5leg': None,
        'nexus_v4_value_4leg': None,
        'nexus_v4_aggressive_8leg': None,
        '_rejection_log': {'screened_out': len(results), 'reason': 'no viable picks in entire board'},
    }


def _v4_build_parlays(safe_pool, agg_pool, GAMES, historical_dir, all_killed, method_tag):
    """Full v4 pipeline: constructors → devils → judges → Borda consensus."""
    if len(safe_pool) < 3:
        return None

    # ── CONSTRUCTORS (11) ──
    print(f"\n  TIER 2: Constructor Agents (11)")
    proposals = []
    workers = min(11, (os.cpu_count() or 4) + 4)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for constructor in ALL_CONSTRUCTORS:
            f = executor.submit(constructor, safe_pool, agg_pool)
            futures[f] = constructor.__name__

        for f in as_completed(futures):
            name = futures[f]
            try:
                result = f.result()
                proposals.append(result)
                safe_count = len(result.get('safe_3leg', []))
                main_count = len(result.get('main_5leg', []))
                value_count = len(result.get('value_4leg', []))
                agg_count = len(result.get('aggressive_8leg', []))
                print(f"    {name}: lock3={safe_count}, main5={main_count}, value4={value_count}, agg={agg_count}")
            except Exception as e:
                print(f"    Constructor {name} error: {e}")

    if not proposals:
        return None

    # ── DEVILS (15) ──
    print(f"\n  TIER 3: Devil's Advocate Agents (15)")
    devil_results = {}
    workers = min(15, (os.cpu_count() or 4) + 4)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for devil_fn in ALL_DEVILS:
            f = executor.submit(devil_fn, proposals)
            futures[f] = devil_fn.__name__

        for f in as_completed(futures):
            name = futures[f]
            try:
                result = f.result()
                devil_results[name] = result
                flagged = sum(1 for prop_name, prop_data in result.items()
                              if any(not prop_data.get(pt, {}).get('passed', True)
                                     for pt in ['safe_3leg', 'main_5leg', 'value_4leg', 'aggressive_8leg']))
                if flagged > 0:
                    print(f"    {name}: flagged {flagged}/{len(proposals)}")
            except Exception as e:
                print(f"    Devil {name} error: {e}")

    # ── JUDGES (5 merged) ──
    print(f"\n  TIER 4: Judge Agents (5)")
    safe_judge_results = {}
    agg_judge_results = {}
    workers = min(10, (os.cpu_count() or 4) + 4)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for judge_name, judge_fn in V4_JUDGES:
            f = executor.submit(judge_fn, proposals, devil_results, 'safe_3leg')
            futures[f] = ('safe', judge_name)
        for judge_name, judge_fn in V4_JUDGES:
            f = executor.submit(judge_fn, proposals, devil_results, 'aggressive_8leg')
            futures[f] = ('agg', judge_name)

        for f in as_completed(futures):
            pool_type, judge_name = futures[f]
            try:
                result = f.result()
                if pool_type == 'safe':
                    safe_judge_results[judge_name] = result
                else:
                    agg_judge_results[judge_name] = result
                if result:
                    top_name, top_score = result[0]
                    print(f"    [{pool_type:4s}] {judge_name:20s} -> {top_name} ({top_score:.1f})")
            except Exception as e:
                print(f"    Judge {judge_name} error: {e}")

    # ── BORDA COUNT CONSENSUS ──
    print(f"\n  CONSENSUS: Borda Count")
    safe_ranking = borda_count_consensus(safe_judge_results)
    agg_ranking = borda_count_consensus(agg_judge_results)

    for name, rank_sum in safe_ranking:
        print(f"    safe:  {name:20s} rank_sum={rank_sum:.0f}")
    for name, rank_sum in agg_ranking:
        print(f"    agg:   {name:20s} rank_sum={rank_sum:.0f}")

    # ── BUILD WINNING PARLAYS ──
    def _find_proposal(name, proposals):
        for p in proposals:
            if p['name'] == name:
                return p
        return None

    # Helper to build a parlay from a proposal's tier
    def _build_tier(tier_key, min_legs, ranking, used_players, pool, label):
        for candidate_name, _ in ranking:
            prop = _find_proposal(candidate_name, proposals)
            if not prop or not prop.get(tier_key):
                continue
            legs = [l for l in prop[tier_key] if l.get('player', '') not in used_players]
            if len(legs) < min_legs:
                continue
            leg_dicts = [_nexus_leg(p) for p in legs]
            conf = _geometric_mean([p.get('nexus_score', 50) for p in legs])
            candidate = {
                'legs': leg_dicts,
                'confidence': round(conf, 1),
                'description': f'NEXUS v4 {label} {len(legs)}-Leg (strategy: {candidate_name})',
                'method': method_tag,
            }
            result = reality_check_with_retry(f'v4_{tier_key}_{candidate_name}', candidate, pool, {}, [])
            if result:
                result['constructor'] = candidate_name
                print(f"    {label} {len(legs)}-leg APPROVED (strategy: {candidate_name})")
                return result
        return None

    # LOCK 3-leg (highest floor, all CORE)
    safe_parlay = _build_tier('safe_3leg', 3, safe_ranking, set(), safe_pool, 'LOCK')

    # Track used players across tiers to avoid overlap
    all_used = set()
    if safe_parlay:
        for leg in safe_parlay.get('legs', []):
            all_used.add(leg.get('player', ''))

    # Main 5-leg (user's preferred size, balanced risk/reward)
    main_parlay = _build_tier('main_5leg', 5, safe_ranking, set(), safe_pool, 'MAIN')
    if main_parlay:
        for leg in main_parlay.get('legs', []):
            all_used.add(leg.get('player', ''))

    # Value 4-leg (best edge picks that didn't make safe cut)
    value_parlay = _build_tier('value_4leg', 4, safe_ranking, all_used, safe_pool, 'VALUE')

    # Aggressive 6-8 leg
    agg_parlay = _build_tier('aggressive_8leg', 5, agg_ranking, all_used, agg_pool, 'AGGRESSIVE')

    # Historical patterns
    temp = {}
    if safe_parlay:
        temp['nexus_v4_safe_3leg'] = safe_parlay
    if main_parlay:
        temp['nexus_v4_main_5leg'] = main_parlay
    if value_parlay:
        temp['nexus_v4_value_4leg'] = value_parlay
    if agg_parlay:
        temp['nexus_v4_aggressive_8leg'] = agg_parlay
    if temp:
        temp = match_historical_patterns(temp, historical_dir)
        safe_parlay = temp.get('nexus_v4_safe_3leg', safe_parlay)
        main_parlay = temp.get('nexus_v4_main_5leg', main_parlay)
        value_parlay = temp.get('nexus_v4_value_4leg', value_parlay)
        agg_parlay = temp.get('nexus_v4_aggressive_8leg', agg_parlay)

    # Build output
    output_parlays = {}
    if safe_parlay:
        output_parlays['nexus_v4_safe_3leg'] = safe_parlay
    if main_parlay:
        output_parlays['nexus_v4_main_5leg'] = main_parlay
    if value_parlay:
        output_parlays['nexus_v4_value_4leg'] = value_parlay
    if agg_parlay:
        output_parlays['nexus_v4_aggressive_8leg'] = agg_parlay

    if not output_parlays:
        return None

    final = finalize_parlays(output_parlays, all_killed)

    # Shadow parlays
    shadow = _build_shadow_parlays(safe_pool)
    final['_shadow_parlays'] = shadow

    return final


def _v4_build_parlays_relaxed(pool, all_killed, method_tag):
    """Relaxed v4: no evaluator gate, no devils, simple greedy build."""
    if len(pool) < 3:
        return None

    pool.sort(key=lambda x: x.get('nexus_score', 0), reverse=True)

    # Simple greedy 3-leg
    safe_3 = _build_with_constraints(pool, 3, lambda p: p.get('nexus_score', 0), set())
    if len(safe_3) < 3:
        return None

    leg_dicts = [_nexus_leg(p) for p in safe_3]
    conf = _geometric_mean([p.get('nexus_score', 50) for p in safe_3])
    safe_parlay = {
        'legs': leg_dicts,
        'confidence': round(conf, 1),
        'description': 'NEXUS v4 RELAXED 3-Leg (no evaluator gate, no devils)',
        'method': method_tag,
        'reality_check': {'status': 'RELAXED', 'attempt': 0, 'issues': ['Used relaxed fallback']},
    }

    output = {}
    output['nexus_v4_safe_3leg'] = safe_parlay
    final = finalize_parlays(output, all_killed)

    shadow = _build_shadow_parlays(pool)
    final['_shadow_parlays'] = shadow

    return final


def _build_shadow_parlays(pool):
    """Build shadow parlays from pool using all constructors + shadow constructors."""
    shadow_parlays = []

    # Main constructors (1-11)
    for constructor in ALL_CONSTRUCTORS:
        try:
            result = constructor(pool, pool)
            raw_name = result.get('name', 'unknown')
            strategy_name = f'nexus_{raw_name}'
            legs = result.get('safe_3leg', [])
            if len(legs) >= 3:
                leg_dicts = [_nexus_leg(p) for p in legs[:3]]
                conf = _geometric_mean([p.get('nexus_score', 50) for p in legs[:3]])
                shadow_parlays.append({
                    'strategy_name': strategy_name,
                    'strategy_id': f'nexus_{raw_name}',
                    'strategy_description': STRATEGY_DESCRIPTIONS.get(raw_name, ''),
                    'legs': leg_dicts,
                    'confidence': round(conf, 1),
                    'legs_total': 3,
                    'result': None,
                    'legs_hit': None,
                })
            else:
                shadow_parlays.append({
                    'strategy_name': strategy_name,
                    'strategy_id': f'nexus_{raw_name}',
                    'strategy_description': STRATEGY_DESCRIPTIONS.get(raw_name, ''),
                    'legs': [_nexus_leg(p) for p in legs] if legs else [],
                    'confidence': 0,
                    'legs_total': len(legs),
                    'result': 'no_build',
                    'legs_hit': None,
                })
        except Exception:
            pass

    # Shadow constructors (12-22)
    for shadow_fn in SHADOW_CONSTRUCTORS:
        try:
            result = shadow_fn(pool)
            raw_name = result.get('name', 'unknown')
            strategy_name = f'nexus_{raw_name}'
            legs = result.get('safe_3leg', [])
            if len(legs) >= 3:
                leg_dicts = [_nexus_leg(p) for p in legs[:3]]
                conf = _geometric_mean([p.get('nexus_score', 50) for p in legs[:3]])
                shadow_parlays.append({
                    'strategy_name': strategy_name,
                    'strategy_id': f'nexus_{raw_name}',
                    'strategy_description': STRATEGY_DESCRIPTIONS.get(raw_name, ''),
                    'legs': leg_dicts,
                    'confidence': round(conf, 1),
                    'legs_total': 3,
                    'result': None,
                    'legs_hit': None,
                })
            else:
                shadow_parlays.append({
                    'strategy_name': strategy_name,
                    'strategy_id': f'nexus_{raw_name}',
                    'strategy_description': STRATEGY_DESCRIPTIONS.get(raw_name, ''),
                    'legs': [_nexus_leg(p) for p in legs] if legs else [],
                    'confidence': 0,
                    'legs_total': len(legs),
                    'result': 'no_build',
                    'legs_hit': None,
                })
        except Exception:
            pass

    return shadow_parlays


def _v4_print_output(output):
    """Print final v4 parlays."""
    for name, parlay in output.items():
        if name.startswith('_') or parlay is None:
            continue
        rank = parlay.get('rank', '?')
        conf = parlay.get('confidence', 0)
        prob = parlay.get('implied_probability', 0)
        rc = parlay.get('reality_check', {}).get('status', '?')
        method = parlay.get('method', '?')
        constructor = parlay.get('constructor', '')
        print(f"\n  #{rank} {name} [conf={conf} prob={prob}% RC={rc} method={method}]")
        if constructor:
            print(f"  Strategy: {constructor}")
        print(f"  {parlay.get('description', '')}")
        for leg in parlay.get('legs', []):
            hr_tag = f"L10={leg.get('l10_hit_rate',0):3.0f}% L5={leg.get('l5_hit_rate',0):3.0f}%"
            ns = leg.get('nexus_score', 0)
            floor = leg.get('l10_floor', 0)
            hist = leg.get('historical_rating', '?')
            print(f"    [{ns:5.1f}] {leg['player']:22s} {leg['stat'].upper():4s} "
                  f"{leg['direction']:5s} {leg['line']:5.1f}  gap={leg['gap']:+5.1f}  "
                  f"{hr_tag}  floor={floor}  hist={hist}")


if __name__ == '__main__':
    print("NEXUS v4 Parlay Builder (27-Agent System + Guaranteed Output)")
    print("Usage: from parlay_nexus import nexus_v4_pipeline")
    print("  nexus_v4_pipeline(results, GAMES, historical_dir=None)")
    print("\nFallback chain: nexus_v3_pipeline, nexus_parlay_pipeline also available")

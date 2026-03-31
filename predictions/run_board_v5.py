#!/usr/bin/env python3
"""
NBA Prop Board Runner v15 - Arena Ensemble + Matchup Scanner + NEXUS Primary
Auto-deploys parallel research agents (1 per team) before running pipeline.
Model Arena (multi-model ensemble) is primary scoring. Matchup Scanner for defense exploits.
Engine v1.2 (validated UNDER bias + line_above_avg) is primary. NEXUS v4 is secondary/shadow.
Ref/coach/sim enrichment subsystems (v8 features). 104-feature XGBoost + MLP ensemble.

Usage:
    python3 run_board_v5.py 2026-03-13 /path/to/parsed_board.json
    python3 run_board_v5.py  # uses today's date, looks for board in /tmp/
"""

import json
import os
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_v3 import analyze_player, build_game_locks, get_fetcher, TIERS, COMBO_STATS
from nba_fetcher import NBAFetcher
from game_researcher import research_all_games, save_research, generate_runner_code, _calc_workers
from self_heal import load_corrections, apply_corrections, analyze_misses, save_corrections
from parlay_nexus import nexus_parlay_pipeline, nexus_v3_pipeline, nexus_v4_pipeline
from pregame_check import run_pregame_check
from backtest_pipelines import (
    old_pipeline_select, hybrid_select, floor_first_select, xgb_select
)
from parlay_engine import build_primary_parlays, build_100_shadow_parlays, build_triple_safe

try:
    from injury_scraper import get_injury_context, get_player_status
except ImportError:
    get_injury_context = None
    get_player_status = None


def _nexus_to_primary(nexus_output):
    """Convert NEXUS v4 output to primary_parlays.json format (safe/aggressive keys)."""
    primary = {}

    # SAFE: use nexus_v4_safe_3leg
    safe = nexus_output.get('nexus_v4_safe_3leg')
    if safe:
        primary['safe'] = {
            'name': 'NEXUS SAFE 3-LEG',
            'method': 'nexus_v4',
            'legs': safe.get('legs', []),
            'legs_total': len(safe.get('legs', [])),
            'confidence': safe.get('confidence', 0),
            'description': safe.get('description', 'NEXUS v4 SAFE 3-Leg (multi-agent consensus)'),
            'constructor': safe.get('constructor', ''),
            'reality_check': safe.get('reality_check', {}),
        }

    # AGGRESSIVE: use nexus_v4_aggressive_8leg, fall back to nexus_v4_main_5leg
    agg = nexus_output.get('nexus_v4_aggressive_8leg') or nexus_output.get('nexus_v4_main_5leg')
    if agg:
        n_legs = len(agg.get('legs', []))
        primary['aggressive'] = {
            'name': f'NEXUS AGGRESSIVE {n_legs}-LEG',
            'method': 'nexus_v4',
            'legs': agg.get('legs', []),
            'legs_total': n_legs,
            'confidence': agg.get('confidence', 0),
            'description': agg.get('description', 'NEXUS v4 Aggressive (multi-agent consensus)'),
            'constructor': agg.get('constructor', ''),
            'reality_check': agg.get('reality_check', {}),
        }

    return primary


def resolve_player_context(player_team_abr, game_key, GAMES):
    """Given player's team abbreviation and game key, return opponent/is_home/spread/is_b2b/injured_out"""
    if game_key not in GAMES:
        return None, None, None, False, [], 0

    gctx = GAMES[game_key]
    away_abr = gctx['away_abr']
    home_abr = gctx['home_abr']

    if player_team_abr == away_abr:
        opponent = gctx['home']
        is_home = False
        spread = gctx['spread']
        is_b2b = gctx.get('away_b2b', False)
        injured_out = gctx.get('away_out', [])  # same-team only for WITH/WITHOUT
        same_team_out_count = len(gctx.get('away_out', []))
    elif player_team_abr == home_abr:
        opponent = gctx['away']
        is_home = True
        spread = -gctx['spread'] if gctx['spread'] else 0
        is_b2b = gctx.get('home_b2b', False)
        injured_out = gctx.get('home_out', [])  # same-team only for WITH/WITHOUT
        same_team_out_count = len(gctx.get('home_out', []))
    else:
        return None, None, None, False, [], 0

    return opponent, is_home, spread, is_b2b, injured_out, same_team_out_count


def get_player_injury_status(player_name, game_key, GAMES):
    """Check if player is in any injury list. Falls back to searching ALL games if game_key missing."""
    name_lower = player_name.lower()

    # Try specific game first
    games_to_check = []
    if game_key and game_key in GAMES:
        games_to_check = [GAMES[game_key]]
    else:
        # Fallback: search ALL games (fixes empty game field causing missed injury lookups)
        games_to_check = list(GAMES.values())

    for gctx in games_to_check:
        for out_player in gctx.get('away_out', []) + gctx.get('home_out', []):
            if name_lower in out_player.lower() or out_player.lower() in name_lower:
                return "OUT"

        for q_player in gctx.get('away_questionable', []) + gctx.get('home_questionable', []):
            if name_lower in q_player.lower() or q_player.lower() in name_lower:
                return "Questionable"

    return None


def run_pipeline(picks, GAMES, pass_num=1):
    """Run the full 14-layer pipeline on all picks"""
    fetcher = get_fetcher()

    injury_data = {}
    injury_file = os.path.join(os.path.dirname(__file__), 'injury_impacts.json')
    if os.path.exists(injury_file):
        with open(injury_file) as f:
            injury_data = json.load(f)

    print(f"\n[Pass {pass_num}] Fetching team rankings...")
    team_rankings = fetcher.get_team_rankings()
    print(f"  {len(team_rankings.get('teams', {}))} teams loaded")

    # Pre-fetch all player logs into cache (cuts analysis from ~7min to ~1.5min)
    unique_players = list(set(p['player'] for p in picks))
    print(f"\n  Pre-fetching game logs for {len(unique_players)} unique players...")
    prefetch_start = time.time()
    hits, fetched = fetcher.prefetch_player_logs(unique_players)
    prefetch_time = time.time() - prefetch_start
    print(f"  Pre-fetched: {hits} cached, {fetched} from API ({prefetch_time:.1f}s)")

    # ═══ FIX: Resolve team+game for every prop ═══
    # Build team abbreviation → game key map from today's GAMES dict
    _team_to_game = {}
    for _gk, _gv in GAMES.items():
        _team_to_game[_gv.get('away_abr', '')] = _gk
        _team_to_game[_gv.get('home_abr', '')] = _gk

    # Step 1: Resolve team from game logs when board lacks it
    needs_team = [p for p in picks if not p.get('team') or p['team'] == '?']
    if needs_team:
        _player_team_cache = {}
        for _pname in list(dict.fromkeys(p['player'] for p in needs_team)):
            try:
                _pid = fetcher._resolve_player(_pname)
                if _pid and _pid in fetcher._gamelog_cache:
                    _df = fetcher._gamelog_cache[_pid]
                    if len(_df) > 0:
                        _matchup = str(_df.iloc[0].get('MATCHUP', ''))
                        if _matchup:
                            _player_team_cache[_pname] = _matchup.split(' ')[0]
            except Exception:
                pass

        team_resolved = 0
        for p in needs_team:
            _t = _player_team_cache.get(p['player'])
            if _t:
                p['team'] = _t
                team_resolved += 1
        if team_resolved:
            print(f"  [FIX] Resolved team from game logs for {team_resolved}/{len(picks)} picks")

    # Step 2: Resolve game key for ALL props that have team but no game
    # MCP scrapers provide team but omit game — this maps team→game via GAMES dict
    game_resolved = 0
    for p in picks:
        if not p.get('game') and p.get('team') and p['team'] in _team_to_game:
            p['game'] = _team_to_game[p['team']]
            game_resolved += 1
    if game_resolved:
        print(f"  [FIX] Resolved game key from team for {game_resolved}/{len(picks)} picks")

    # Filter out line=0 props (scraper artifacts — no real sportsbook line)
    valid_picks = [p for p in picks if p.get('line', 0) > 0]
    if len(valid_picks) < len(picks):
        print(f"  Filtered {len(picks) - len(valid_picks)} line=0 props (scraper artifacts)")
    picks = valid_picks

    results = []
    total = len(picks)
    start_time = time.time()

    for i, pick in enumerate(picks):
        player = pick['player']
        stat = pick['stat'].lower()
        pick['stat'] = stat
        line = pick['line']
        game = pick.get('game', '')
        team_abr = pick.get('team', '')

        opponent, is_home, spread, is_b2b, injured_out, same_team_out_count = resolve_player_context(team_abr, game, GAMES)
        player_status = get_player_injury_status(player, game, GAMES)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate
            print(f"  [{i+1}/{total}] {player} {stat} {line} | {rate:.1f} lines/sec | ETA {eta:.0f}s")

        try:
            result = analyze_player(
                player_name=player,
                stat=stat,
                line=line,
                opponent=opponent,
                is_home=is_home,
                injury_data=injury_data,
                injured_out=injured_out,
                player_injury_status=player_status,
                is_b2b=is_b2b,
                spread=spread,
                team_rankings=team_rankings,
                game=game,
                same_team_out_count=same_team_out_count,
            )
            result['team'] = team_abr
            results.append(result)
        except Exception as e:
            results.append({
                'player': player, 'stat': stat, 'line': line,
                'error': str(e), 'tier': 'SKIP', 'direction': 'SKIP',
                'game': game, 'team': team_abr,
            })

    elapsed = time.time() - start_time
    print(f"\n  Pass {pass_num} complete: {total} lines in {elapsed:.1f}s ({total/elapsed:.1f} lines/sec)")
    return results


def backtest_pass2(results):
    """Pass 2: L5 trend validation"""
    print("\n[Pass 2] Applying L5 trend validation...")
    adjusted = 0
    for r in results:
        if 'error' in r:
            continue
        l5 = r.get('l5_avg', 0)
        proj = r.get('projection', 0)
        direction = r.get('direction', '')
        if proj == 0:
            continue

        # L5 trend warnings (logged but no tier downgrade — tiers removed)
        if direction == 'OVER' and l5 < proj * 0.85:
            r['backtest_note'] = f"L5 trend warning: L5={l5:.1f} vs proj={proj:.1f}"
            adjusted += 1

        if direction == 'UNDER':
            line = r.get('line', 0)
            if l5 > line * 1.1:
                r['backtest_note'] = r.get('backtest_note', '') + f" | L5 UNDER risk: L5={l5:.1f} > line={line:.1f}"
                adjusted += 1

    print(f"  Adjusted {adjusted} picks based on L5 trends")
    return results


def backtest_pass3(results):
    """Pass 3: Opponent matchup history cross-check"""
    print("\n[Pass 3] Cross-checking opponent history...")
    adjusted = 0
    for r in results:
        if 'error' in r:
            continue
        opp_hist = r.get('opponent_history')
        if not opp_hist or opp_hist.get('games', 0) < 1:
            continue

        opp_avg = opp_hist.get('avg', 0)
        line = r.get('line', 0)
        direction = r.get('direction', '')
        opp_hr = opp_hist.get('hit_rate', 50)

        # Opponent history warnings (logged but no tier changes)
        if direction == 'OVER' and opp_avg < line * 0.9 and opp_hr < 40:
            r['backtest_note'] = r.get('backtest_note', '') + f" | Opp weak: {opp_avg} ({opp_hr}% HR)"
            adjusted += 1

        if direction == 'OVER' and opp_avg > line * 1.15 and opp_hr >= 70:
            r['backtest_note'] = r.get('backtest_note', '') + f" | Opp strong: {opp_avg} ({opp_hr}% HR)"
            adjusted += 1

    print(f"  Adjusted {adjusted} picks based on opponent history")
    return results


def improved_build_parlays(results):
    """Improved parlay builder (same logic from run_march12.py)"""
    core_over = [r for r in results if r.get('direction') == 'OVER' and 'error' not in r
                 and r.get('l10_hit_rate', 0) >= 60]
    core_under = [r for r in results if r.get('direction') == 'UNDER' and 'error' not in r
                  and r.get('l10_hit_rate', 0) >= 60]
    flex = [r for r in results if r.get('direction') == 'OVER' and 'error' not in r
            and r.get('l10_hit_rate', 0) >= 50]
    anchors = [r for r in results if r.get('stat', '').lower() in ['blk', 'stl']
               and 'error' not in r and r.get('l10_hit_rate', 0) >= 60]

    all_candidates = core_over + core_under + flex + anchors

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

        if r.get('stat', '').lower() in ['pra', 'pr', 'pa', 'ra'] and gap < 4.0:
            score *= 0.7
        if r.get('stat', '').lower() in ['blk', 'stl']:
            score *= 1.2
        if r.get('streak_status') == 'HOT':
            score *= 1.1
        if r.get('streak_status') == 'COLD':
            score *= 0.85
        if r.get('player_injury_status') in ['Questionable', 'GTD']:
            score *= 0.6
        return score

    unique.sort(key=parlay_score, reverse=True)

    used_games = set()
    used_teams = set()
    selected = []
    for r in unique:
        game_key = r.get('game', '')
        if game_key and game_key in used_games:
            continue

        player_team = _get_player_team(r, game_key)
        if player_team and player_team in used_teams:
            continue

        if r.get('l10_hit_rate', 0) < 60:
            continue
        if r.get('l5_hit_rate', 0) < 40:
            continue

        selected.append(r)
        if game_key:
            used_games.add(game_key)
        if player_team:
            used_teams.add(player_team)
        if len(selected) >= 9:
            break

    def leg_summary(r):
        return {
            'player': r['player'], 'stat': r['stat'], 'line': r['line'],
            'direction': r['direction'], 'tier': r['tier'],
            'gap': r.get('gap', 0), 'projection': r.get('projection', 0),
            'l10_hit_rate': r.get('l10_hit_rate', 0),
            'l5_hit_rate': r.get('l5_hit_rate', 0),
            'season_avg': r.get('season_avg', 0),
            'game': r.get('game', ''),
            'streak': r.get('streak_status', 'NEUTRAL'),
            'matchup_note': r.get('matchup_note', ''),
            'reasoning': r.get('reasoning', ''),
            'score': round(parlay_score(r), 3),
        }

    parlays = {}
    if len(selected) >= 3:
        parlays['conservative_3leg'] = {
            'legs': [leg_summary(l) for l in selected[:3]],
            'description': 'Conservative 3-leg: Top 3 edges, all different games',
        }
    if len(selected) >= 4:
        parlays['main_4leg'] = {
            'legs': [leg_summary(l) for l in selected[:4]],
            'description': 'Main 4-leg: Strong core + 1 flex',
        }
    if len(selected) >= 5:
        parlays['standard_5leg'] = {
            'legs': [leg_summary(l) for l in selected[:5]],
            'description': 'Standard 5-leg: Full parlay with solid edges',
        }
    if len(selected) >= 6:
        parlays['aggressive_6leg'] = {
            'legs': [leg_summary(l) for l in selected[:6]],
            'description': 'Aggressive 6-leg: Higher payout, more risk',
        }
    return parlays


def _get_player_team(result, game_key):
    is_home = result.get('is_home')
    if not game_key or '@' not in game_key:
        return None
    parts = game_key.split('@')
    if is_home is True:
        return parts[1] if len(parts) > 1 else None
    elif is_home is False:
        return parts[0] if parts else None
    return None


# ═══════════════════════════════════════════════════════════════
# 20 PARLAY REASONING AGENTS
# Each agent evaluates candidates from a unique angle.
# All run in parallel, then their verdicts compile into final picks.
# ═══════════════════════════════════════════════════════════════

def _leg_key(r):
    return f"{r['player']}|{r['stat']}|{r['direction']}"


def _get_candidates(results, top_n=20):
    """Extract top parlay candidates (S/A tier + B flex + S UNDER)."""
    pool = [r for r in results if 'error' not in r and r.get('l10_hit_rate', 0) >= 50]
    seen = set()
    unique = []
    for r in pool:
        k = (r['player'], r['stat'])
        if k not in seen:
            seen.add(k)
            unique.append(r)
    unique.sort(key=lambda x: x.get('abs_gap', 0), reverse=True)
    return unique[:top_n]


# ── 12 Leg Evaluator Agents (1 per top candidate) ──

def agent_evaluate_leg(candidate, idx, GAMES):
    """
    Deep evaluation of a single parlay leg candidate.
    Reasons through: gap reliability, matchup, form, risk factors.
    Returns a verdict with confidence score 0-100.
    """
    verdict = {
        'agent_id': f'leg_eval_{idx}',
        'leg': _leg_key(candidate),
        'player': candidate['player'],
        'stat': candidate['stat'],
        'direction': candidate['direction'],
        'line': candidate['line'],
        'tier': candidate['tier'],
        'reasons_for': [],
        'reasons_against': [],
        'confidence': 50,
        'parlay_worthy': False,
    }

    gap = candidate.get('gap', 0)
    abs_gap = candidate.get('abs_gap', 0)
    l10_hr = candidate.get('l10_hit_rate', 0)
    l5_hr = candidate.get('l5_hit_rate', 0)
    season_avg = candidate.get('season_avg', 0)
    l10_avg = candidate.get('l10_avg', 0)
    l5_avg = candidate.get('l5_avg', 0)
    line = candidate['line']
    direction = candidate['direction']
    streak = candidate.get('streak_status', 'NEUTRAL')
    stat = candidate['stat']

    score = 50  # Start neutral

    # ── GAP ANALYSIS ──
    if abs_gap >= 6:
        verdict['reasons_for'].append(f"Massive gap ({gap:+.1f}) — line significantly mispriced")
        score += 20
    elif abs_gap >= 4:
        verdict['reasons_for'].append(f"Strong gap ({gap:+.1f}) — solid edge")
        score += 12
    elif abs_gap >= 3:
        verdict['reasons_for'].append(f"Good gap ({gap:+.1f}) — real but not huge")
        score += 7
    else:
        verdict['reasons_against'].append(f"Modest gap ({gap:+.1f}) — thin margin for parlay")
        score -= 5

    # ── HIT RATE ANALYSIS ──
    if l10_hr >= 80:
        verdict['reasons_for'].append(f"Elite L10 hit rate ({l10_hr}%) — extremely consistent")
        score += 15
    elif l10_hr >= 70:
        verdict['reasons_for'].append(f"Strong L10 hit rate ({l10_hr}%) — reliable")
        score += 10
    elif l10_hr >= 60:
        score += 3
    else:
        verdict['reasons_against'].append(f"Low L10 hit rate ({l10_hr}%) — inconsistent")
        score -= 10

    # L5 trend check
    if l5_hr >= 80:
        verdict['reasons_for'].append(f"L5 confirms trend ({l5_hr}%) — current form strong")
        score += 8
    elif l5_hr < 40:
        verdict['reasons_against'].append(f"L5 drop-off ({l5_hr}%) — recent form contradicts")
        score -= 15

    # ── DIRECTION CONSISTENCY ──
    if direction == 'OVER':
        if season_avg > line and l10_avg > line and l5_avg > line:
            verdict['reasons_for'].append(
                f"All averages clear line: season {season_avg:.1f}, L10 {l10_avg:.1f}, L5 {l5_avg:.1f} > {line}")
            score += 10
        elif l5_avg < line:
            verdict['reasons_against'].append(
                f"L5 avg ({l5_avg:.1f}) below line ({line}) — recent dip")
            score -= 8
    else:  # UNDER
        if season_avg < line and l10_avg < line and l5_avg < line:
            verdict['reasons_for'].append(
                f"All averages below line: season {season_avg:.1f}, L10 {l10_avg:.1f}, L5 {l5_avg:.1f} < {line}")
            score += 10
        elif l5_avg > line:
            verdict['reasons_against'].append(
                f"L5 avg ({l5_avg:.1f}) above line ({line}) — recent spike")
            score -= 8

    # ── STREAK ANALYSIS ──
    if streak == 'HOT' and direction == 'OVER':
        verdict['reasons_for'].append("HOT streak aligns with OVER direction")
        score += 5
    elif streak == 'COLD' and direction == 'UNDER':
        verdict['reasons_for'].append("COLD streak aligns with UNDER direction")
        score += 5
    elif streak == 'HOT' and direction == 'UNDER':
        verdict['reasons_against'].append("HOT streak contradicts UNDER pick")
        score -= 8
    elif streak == 'COLD' and direction == 'OVER':
        verdict['reasons_against'].append("COLD streak contradicts OVER pick")
        score -= 8

    # ── COMBO STAT RISK ──
    if stat.lower() in ['pra', 'pr', 'pa', 'ra']:
        verdict['reasons_against'].append(f"Combo stat ({stat.upper()}) — higher variance than base stats")
        score -= 5
        if abs_gap < 4:
            verdict['reasons_against'].append(f"Combo stat with small gap ({gap:+.1f}) — dangerous for parlay")
            score -= 8

    # ── GAME CONTEXT ──
    game_key = candidate.get('game', '')
    if game_key and game_key in GAMES:
        gctx = GAMES[game_key]
        spread = gctx.get('spread', 0)

        # Blowout risk
        if abs(spread) >= 12:
            if direction == 'OVER':
                verdict['reasons_against'].append(
                    f"Blowout game (spread {spread:+.1f}) — starters may sit Q4, capping upside")
                score -= 7
            else:
                verdict['reasons_for'].append(
                    f"Blowout game (spread {spread:+.1f}) — reduced minutes supports UNDER")
                score += 3

        # B2B fatigue
        is_home = candidate.get('is_home')
        if is_home is True and gctx.get('home_b2b'):
            verdict['reasons_against'].append("Player's team on B2B — fatigue risk")
            score -= 5
        elif is_home is False and gctx.get('away_b2b'):
            verdict['reasons_against'].append("Player's team on B2B — fatigue risk")
            score -= 5

        # Verification warnings
        for warn in gctx.get('verification_warnings', [])[:2]:
            verdict['reasons_against'].append(f"Verification flag: {warn[:60]}")
            score -= 3

    # ── INJURY STATUS ──
    if candidate.get('player_injury_status') in ['Questionable', 'GTD']:
        verdict['reasons_against'].append("Player is Questionable/GTD — may not play or be limited")
        score -= 20

    # ── FINAL VERDICT ──
    score = max(0, min(100, score))
    verdict['confidence'] = score
    verdict['parlay_worthy'] = score >= 60

    return verdict


# ── 3 Cross-Analysis Agents ──

def agent_correlation_check(selected_legs, GAMES):
    """
    Check for hidden correlations between parlay legs.
    Same-game, same-team, and game-environment correlations.
    """
    issues = []
    warnings = []

    # Same game check (should be caught already, but verify)
    games_used = {}
    for leg in selected_legs:
        g = leg.get('game', '')
        if g in games_used:
            issues.append(f"SAME GAME: {games_used[g]} and {leg['player']} both in {g}")
        games_used[g] = leg['player']

    # Same team check
    teams_used = {}
    for leg in selected_legs:
        team = _get_player_team(leg, leg.get('game', ''))
        if team and team in teams_used:
            warnings.append(
                f"Same team: {teams_used[team]} and {leg['player']} both on {team} — "
                f"correlated outcomes")
        if team:
            teams_used[team] = leg['player']

    # Opposing player check (if one goes OVER, opponent might go UNDER)
    for i, leg1 in enumerate(selected_legs):
        for leg2 in selected_legs[i+1:]:
            if leg1.get('game') == leg2.get('game') and leg1.get('game'):
                # Same game, opposing teams
                t1 = _get_player_team(leg1, leg1.get('game', ''))
                t2 = _get_player_team(leg2, leg2.get('game', ''))
                if t1 and t2 and t1 != t2:
                    warnings.append(
                        f"Opposing teams: {leg1['player']} ({t1}) vs {leg2['player']} ({t2}) in same game — "
                        f"negatively correlated")

    # All OVERs check — pace-dependent
    all_over = all(leg.get('direction') == 'OVER' for leg in selected_legs)
    if all_over and len(selected_legs) >= 4:
        warnings.append("All legs are OVER — vulnerable to slow-pace games across the board")

    # All UNDERs check
    all_under = all(leg.get('direction') == 'UNDER' for leg in selected_legs)
    if all_under and len(selected_legs) >= 4:
        warnings.append("All legs are UNDER — vulnerable to high-pace shootouts")

    return {
        'agent': 'correlation_check',
        'issues': issues,
        'warnings': warnings,
        'risk_level': 'HIGH' if issues else ('MEDIUM' if len(warnings) >= 2 else 'LOW'),
    }


def agent_blowout_scenario(selected_legs, GAMES):
    """
    Model what happens to each leg if its game becomes a blowout.
    Blowouts compress Q4 minutes and distort stats.
    """
    leg_risks = []

    for leg in selected_legs:
        game_key = leg.get('game', '')
        if not game_key or game_key not in GAMES:
            continue

        gctx = GAMES[game_key]
        spread = abs(gctx.get('spread', 0))
        risk = {
            'player': leg['player'],
            'game': game_key,
            'spread': spread,
            'blowout_impact': 'NONE',
            'reasoning': '',
        }

        if spread >= 15:
            if leg['direction'] == 'OVER':
                risk['blowout_impact'] = 'HIGH_RISK'
                risk['reasoning'] = (
                    f"Spread {spread:.1f} = likely blowout. Starters may sit Q4. "
                    f"OVER {leg['line']} becomes harder if player only gets 28-30 mins.")
            else:
                risk['blowout_impact'] = 'FAVORABLE'
                risk['reasoning'] = (
                    f"Spread {spread:.1f} = likely blowout. Reduced minutes = fewer stats. "
                    f"UNDER {leg['line']} benefits from early bench.")
        elif spread >= 10:
            if leg['direction'] == 'OVER':
                risk['blowout_impact'] = 'MODERATE_RISK'
                risk['reasoning'] = (
                    f"Spread {spread:.1f} = possible blowout. Some Q4 minute risk for OVER.")
            else:
                risk['blowout_impact'] = 'SLIGHT_FAVOR'
                risk['reasoning'] = f"Spread {spread:.1f} = possible blowout. Slightly helps UNDER."
        else:
            risk['blowout_impact'] = 'NONE'
            risk['reasoning'] = f"Spread {spread:.1f} = competitive game expected. No blowout concern."

        leg_risks.append(risk)

    high_risks = sum(1 for r in leg_risks if r['blowout_impact'] == 'HIGH_RISK')

    return {
        'agent': 'blowout_scenario',
        'leg_risks': leg_risks,
        'summary': f"{high_risks} legs at high blowout risk" if high_risks else "No significant blowout risks",
        'recommendation': 'SWAP_NEEDED' if high_risks >= 2 else 'OK',
    }


def agent_fatigue_analysis(selected_legs, GAMES):
    """
    Assess fatigue and rest factors for each leg.
    B2B, compressed schedule, travel, rest advantage.
    """
    leg_fatigue = []

    for leg in selected_legs:
        game_key = leg.get('game', '')
        if not game_key or game_key not in GAMES:
            continue

        gctx = GAMES[game_key]
        is_home = leg.get('is_home')
        side = 'home' if is_home else 'away'

        fatigue = {
            'player': leg['player'],
            'b2b': gctx.get(f'{side}_b2b', False),
            'rest_days': gctx.get(f'{side}_rest_days'),
            'schedule_density': gctx.get(f'{side}_schedule_density', 'normal'),
            'impact': 'NONE',
            'reasoning': '',
        }

        if fatigue['b2b']:
            if leg['direction'] == 'OVER':
                fatigue['impact'] = 'NEGATIVE'
                fatigue['reasoning'] = "B2B = fatigue. OVER picks harder on tired legs."
            else:
                fatigue['impact'] = 'FAVORABLE'
                fatigue['reasoning'] = "B2B = fatigue. UNDER picks benefit from tired legs."
        elif fatigue['schedule_density'] == 'compressed':
            if leg['direction'] == 'OVER':
                fatigue['impact'] = 'SLIGHT_NEGATIVE'
                fatigue['reasoning'] = "Compressed schedule — accumulated fatigue may limit upside."
            else:
                fatigue['impact'] = 'SLIGHT_FAVOR'
                fatigue['reasoning'] = "Compressed schedule — fatigue helps UNDER."
        elif fatigue['schedule_density'] == 'well_rested':
            if leg['direction'] == 'OVER':
                fatigue['impact'] = 'FAVORABLE'
                fatigue['reasoning'] = "Well-rested — fresh legs boost OVER potential."

        leg_fatigue.append(fatigue)

    b2b_overs = sum(1 for f in leg_fatigue
                    if f['b2b'] and f['impact'] == 'NEGATIVE')

    return {
        'agent': 'fatigue_analysis',
        'leg_fatigue': leg_fatigue,
        'b2b_over_count': b2b_overs,
        'recommendation': 'SWAP_B2B_OVERS' if b2b_overs >= 2 else 'OK',
    }


# ── 5 Strategy Agents ──

def agent_base_stat_only(results, GAMES):
    """Build a parlay using ONLY base stats (PTS, REB, AST, 3PM, BLK, STL) — no combos."""
    candidates = [r for r in results if r.get('stat') not in ['pra', 'pr', 'pa', 'ra']
                  and 'error' not in r and r.get('l10_hit_rate', 0) >= 60
                  and r.get('l5_hit_rate', 0) >= 40]
    candidates.sort(key=lambda x: x.get('abs_gap', 0), reverse=True)

    used_games = set()
    used_teams = set()
    selected = []
    for r in candidates:
        g = r.get('game', '')
        t = _get_player_team(r, g)
        if g and g in used_games:
            continue
        if t and t in used_teams:
            continue
        selected.append(r)
        if g: used_games.add(g)
        if t: used_teams.add(t)
        if len(selected) >= 6:
            break

    return {
        'agent': 'base_stat_only',
        'strategy': 'Base stats only — no combo stat variance',
        'legs': [{
            'player': r['player'], 'stat': r['stat'], 'direction': r['direction'],
            'line': r['line'], 'tier': r['tier'], 'gap': r.get('gap', 0),
            'l10_hr': r.get('l10_hit_rate', 0), 'game': r.get('game', ''),
        } for r in selected],
    }


def agent_under_focused(results, GAMES):
    """Build a parlay focused on UNDER plays (historically 70%+ accuracy)."""
    candidates = [r for r in results if r.get('direction') == 'UNDER' and 'error' not in r
                  and r.get('l10_hit_rate', 0) >= 60]
    candidates.sort(key=lambda x: x.get('abs_gap', 0), reverse=True)

    used_games = set()
    used_teams = set()
    selected = []
    for r in candidates:
        g = r.get('game', '')
        t = _get_player_team(r, g)
        if g and g in used_games:
            continue
        if t and t in used_teams:
            continue
        selected.append(r)
        if g: used_games.add(g)
        if t: used_teams.add(t)
        if len(selected) >= 6:
            break

    return {
        'agent': 'under_focused',
        'strategy': 'UNDER-heavy — leveraging 70%+ UNDER accuracy from backtests',
        'legs': [{
            'player': r['player'], 'stat': r['stat'], 'direction': r['direction'],
            'line': r['line'], 'tier': r['tier'], 'gap': r.get('gap', 0),
            'l10_hr': r.get('l10_hit_rate', 0), 'game': r.get('game', ''),
        } for r in selected],
    }


def agent_chalk_safe(results, GAMES):
    """Build the safest possible parlay — max hit rates, ignore gap size."""
    candidates = [r for r in results if 'error' not in r and r.get('l10_hit_rate', 0) >= 70
                  and r.get('l5_hit_rate', 0) >= 60]
    # Sort by hit rate, not gap
    candidates.sort(key=lambda x: (x.get('l10_hit_rate', 0) + x.get('l5_hit_rate', 0)), reverse=True)

    used_games = set()
    used_teams = set()
    selected = []
    for r in candidates:
        g = r.get('game', '')
        t = _get_player_team(r, g)
        if g and g in used_games:
            continue
        if t and t in used_teams:
            continue
        selected.append(r)
        if g: used_games.add(g)
        if t: used_teams.add(t)
        if len(selected) >= 5:
            break

    return {
        'agent': 'chalk_safe',
        'strategy': 'Maximum hit rate parlay — sacrifice gap for consistency',
        'legs': [{
            'player': r['player'], 'stat': r['stat'], 'direction': r['direction'],
            'line': r['line'], 'tier': r['tier'], 'gap': r.get('gap', 0),
            'l10_hr': r.get('l10_hit_rate', 0), 'l5_hr': r.get('l5_hit_rate', 0),
            'game': r.get('game', ''),
        } for r in selected],
    }


def agent_contrarian(results, GAMES):
    """Build a parlay from strong UNDERs + COLD streak OVERs (mean reversion plays)."""
    # Strong UNDERs on HOT players (they'll regress)
    under_hot = [r for r in results if r.get('direction') == 'UNDER' and r.get('streak_status') == 'HOT'
                 and 'error' not in r and r.get('l10_hit_rate', 0) >= 60]
    # OVERs on COLD players (mean reversion bounce)
    over_cold = [r for r in results if r.get('direction') == 'OVER' and r.get('abs_gap', 0) >= 5
                 and 'error' not in r and r.get('l10_hit_rate', 0) >= 60]
    # Defensive mismatches
    defense_plays = [r for r in results if 'error' not in r and r.get('abs_gap', 0) >= 4
                     and r.get('matchup_note', '') and r.get('l10_hit_rate', 0) >= 60]

    pool = under_hot + over_cold + defense_plays
    seen = set()
    unique = []
    for r in pool:
        k = (r['player'], r['stat'])
        if k not in seen:
            seen.add(k)
            unique.append(r)
    unique.sort(key=lambda x: x.get('abs_gap', 0), reverse=True)

    used_games = set()
    used_teams = set()
    selected = []
    for r in unique:
        g = r.get('game', '')
        t = _get_player_team(r, g)
        if g and g in used_games:
            continue
        if t and t in used_teams:
            continue
        selected.append(r)
        if g: used_games.add(g)
        if t: used_teams.add(t)
        if len(selected) >= 5:
            break

    return {
        'agent': 'contrarian',
        'strategy': 'Contrarian — mean reversion + defensive mismatch exploitation',
        'legs': [{
            'player': r['player'], 'stat': r['stat'], 'direction': r['direction'],
            'line': r['line'], 'tier': r['tier'], 'gap': r.get('gap', 0),
            'l10_hr': r.get('l10_hit_rate', 0), 'streak': r.get('streak_status', ''),
            'game': r.get('game', ''),
        } for r in selected],
    }


def agent_blk_stl_anchor(results, GAMES):
    """Build a parlay anchored by BLK/STL picks (85%+ accuracy historically)."""
    anchors = [r for r in results if r.get('stat', '').lower() in ['blk', 'stl']
               and 'error' not in r
               and r.get('l10_hit_rate', 0) >= 60]
    others = [r for r in results if r.get('stat', '').lower() not in ['blk', 'stl'] and 'error' not in r
              and r.get('l10_hit_rate', 0) >= 70]
    anchors.sort(key=lambda x: x.get('abs_gap', 0), reverse=True)
    others.sort(key=lambda x: x.get('abs_gap', 0), reverse=True)

    used_games = set()
    used_teams = set()
    selected = []

    # Pick up to 2 anchors first
    for r in anchors[:4]:
        g = r.get('game', '')
        t = _get_player_team(r, g)
        if g and g in used_games:
            continue
        if t and t in used_teams:
            continue
        selected.append(r)
        if g: used_games.add(g)
        if t: used_teams.add(t)
        if len(selected) >= 2:
            break

    # Fill with best remaining
    for r in others:
        g = r.get('game', '')
        t = _get_player_team(r, g)
        if g and g in used_games:
            continue
        if t and t in used_teams:
            continue
        selected.append(r)
        if g: used_games.add(g)
        if t: used_teams.add(t)
        if len(selected) >= 5:
            break

    return {
        'agent': 'blk_stl_anchor',
        'strategy': 'BLK/STL anchored — leveraging 85%+ accuracy on defense stats',
        'legs': [{
            'player': r['player'], 'stat': r['stat'], 'direction': r['direction'],
            'line': r['line'], 'tier': r['tier'], 'gap': r.get('gap', 0),
            'l10_hr': r.get('l10_hit_rate', 0), 'game': r.get('game', ''),
        } for r in selected],
    }


# ── MASTER COMPILER ──

def parlay_reasoning_engine(results, GAMES, max_workers=None):
    """
    Deploy 20 parlay reasoning agents in parallel:
    - 12 leg evaluators (1 per top candidate)
    - 3 cross-analysis agents (correlation, blowout, fatigue)
    - 5 strategy agents (base-stat, under-focused, chalk, contrarian, blk/stl)

    Returns compiled parlay recommendations with full reasoning.
    """
    if max_workers is None:
        max_workers = _calc_workers(20, 'local')
    print(f"\n{'='*60}")
    print(f"  PARLAY REASONING ENGINE — 20 AGENTS (workers={max_workers})")
    print(f"{'='*60}")

    start = time.time()
    candidates = _get_candidates(results, top_n=12)
    print(f"  Top {len(candidates)} candidates identified for deep evaluation")

    # Also build the standard parlay first (as baseline)
    standard_parlays = improved_build_parlays(results)
    standard_legs = []
    if 'standard_5leg' in standard_parlays:
        standard_legs = standard_parlays['standard_5leg']['legs']
    elif 'main_4leg' in standard_parlays:
        standard_legs = standard_parlays['main_4leg']['legs']
    elif 'conservative_3leg' in standard_parlays:
        standard_legs = standard_parlays['conservative_3leg']['legs']

    # Need full result objects for cross-analysis
    standard_leg_results = []
    for leg in standard_legs:
        for r in results:
            if r['player'] == leg['player'] and r['stat'] == leg['stat']:
                standard_leg_results.append(r)
                break

    all_verdicts = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # 12 Leg Evaluator Agents
        for i, cand in enumerate(candidates[:12]):
            f = executor.submit(agent_evaluate_leg, cand, i, GAMES)
            futures[f] = ('leg_eval', i)

        # 3 Cross-Analysis Agents (run on standard parlay legs)
        if standard_leg_results:
            f = executor.submit(agent_correlation_check, standard_leg_results, GAMES)
            futures[f] = ('cross', 'correlation')

            f = executor.submit(agent_blowout_scenario, standard_leg_results, GAMES)
            futures[f] = ('cross', 'blowout')

            f = executor.submit(agent_fatigue_analysis, standard_leg_results, GAMES)
            futures[f] = ('cross', 'fatigue')

        # 5 Strategy Agents
        f = executor.submit(agent_base_stat_only, results, GAMES)
        futures[f] = ('strategy', 'base_stat')

        f = executor.submit(agent_under_focused, results, GAMES)
        futures[f] = ('strategy', 'under_focused')

        f = executor.submit(agent_chalk_safe, results, GAMES)
        futures[f] = ('strategy', 'chalk_safe')

        f = executor.submit(agent_contrarian, results, GAMES)
        futures[f] = ('strategy', 'contrarian')

        f = executor.submit(agent_blk_stl_anchor, results, GAMES)
        futures[f] = ('strategy', 'blk_stl')

        print(f"  Deployed {len(futures)} reasoning agents...")

        # Collect
        leg_verdicts = []
        cross_verdicts = []
        strategy_verdicts = []

        for future in as_completed(futures):
            agent_type, key = futures[future]
            try:
                result = future.result(timeout=15)
                if agent_type == 'leg_eval':
                    leg_verdicts.append(result)
                elif agent_type == 'cross':
                    cross_verdicts.append(result)
                elif agent_type == 'strategy':
                    strategy_verdicts.append(result)
            except Exception as e:
                print(f"  [WARN] Agent {agent_type}/{key} failed: {e}")

    elapsed = time.time() - start
    print(f"  All {len(futures)} agents complete in {elapsed:.1f}s")

    # ── PRINT LEG EVALUATIONS ──
    leg_verdicts.sort(key=lambda x: x['confidence'], reverse=True)
    print(f"\n  LEG EVALUATIONS (top candidates):")
    for v in leg_verdicts:
        worthy = "YES" if v['parlay_worthy'] else "NO"
        print(f"    [{v['confidence']:3d}%] {worthy:3s} | {v['player']:20s} {v['stat'].upper():4s} "
              f"{v['direction']:5s} {v['line']:5.1f} ({v['tier']})")
        for r in v['reasons_for'][:2]:
            print(f"         + {r[:80]}")
        for r in v['reasons_against'][:2]:
            print(f"         - {r[:80]}")

    # ── PRINT CROSS-ANALYSIS ──
    print(f"\n  CROSS-ANALYSIS:")
    for v in cross_verdicts:
        agent_name = v.get('agent', '?')
        if v.get('issues'):
            for issue in v['issues']:
                print(f"    [{agent_name}] !! {issue[:90]}")
        if v.get('warnings'):
            for w in v['warnings'][:3]:
                print(f"    [{agent_name}] >> {w[:90]}")
        if v.get('recommendation') and v['recommendation'] != 'OK':
            print(f"    [{agent_name}] ACTION: {v['recommendation']}")
        if v.get('summary'):
            print(f"    [{agent_name}] {v['summary']}")

    # ── PRINT STRATEGY ALTERNATIVES ──
    print(f"\n  STRATEGY PARLAYS:")
    for v in strategy_verdicts:
        agent_name = v.get('agent', '?')
        print(f"\n    [{agent_name}] {v.get('strategy', '')}")
        for leg in v.get('legs', [])[:5]:
            print(f"      {leg['player']:20s} {leg['stat'].upper():4s} {leg['direction']:5s} "
                  f"{leg['line']:5.1f}  gap={leg.get('gap',0):+.1f}  "
                  f"L10={leg.get('l10_hr', leg.get('l10_hit_rate',0)):3.0f}%")

    # ── BUILD FINAL RECOMMENDATION ──
    # Use leg verdicts to re-rank the standard parlay
    verdict_map = {v['leg']: v for v in leg_verdicts}

    # Get all parlay-worthy legs, sorted by confidence
    worthy_legs = [v for v in leg_verdicts if v['parlay_worthy']]
    worthy_legs.sort(key=lambda x: x['confidence'], reverse=True)

    # Build reasoned parlay from worthy legs
    reasoned_selected = []
    used_games = set()
    used_teams = set()

    for v in worthy_legs:
        # Find the full result
        for r in results:
            if _leg_key(r) == v['leg']:
                g = r.get('game', '')
                t = _get_player_team(r, g)
                if g and g in used_games:
                    continue
                if t and t in used_teams:
                    continue
                r_copy = dict(r)
                r_copy['reasoning_confidence'] = v['confidence']
                r_copy['reasoning_for'] = v['reasons_for']
                r_copy['reasoning_against'] = v['reasons_against']
                reasoned_selected.append(r_copy)
                if g: used_games.add(g)
                if t: used_teams.add(t)
                break
        if len(reasoned_selected) >= 6:
            break

    # Build final parlays
    def leg_summary_reasoned(r):
        return {
            'player': r['player'], 'stat': r['stat'], 'line': r['line'],
            'direction': r['direction'], 'tier': r['tier'],
            'gap': r.get('gap', 0), 'projection': r.get('projection', 0),
            'l10_hit_rate': r.get('l10_hit_rate', 0),
            'l5_hit_rate': r.get('l5_hit_rate', 0),
            'season_avg': r.get('season_avg', 0),
            'game': r.get('game', ''),
            'streak': r.get('streak_status', 'NEUTRAL'),
            'matchup_note': r.get('matchup_note', ''),
            'reasoning': r.get('reasoning', ''),
            'confidence': r.get('reasoning_confidence', 0),
            'reasons_for': r.get('reasoning_for', []),
            'reasons_against': r.get('reasoning_against', []),
        }

    final_parlays = {}

    if len(reasoned_selected) >= 3:
        final_parlays['reasoned_3leg'] = {
            'legs': [leg_summary_reasoned(l) for l in reasoned_selected[:3]],
            'description': 'REASONED 3-leg: Top 3 agent-validated picks',
            'method': '20-agent reasoning engine',
        }
    if len(reasoned_selected) >= 4:
        final_parlays['reasoned_4leg'] = {
            'legs': [leg_summary_reasoned(l) for l in reasoned_selected[:4]],
            'description': 'REASONED 4-leg: Agent consensus top 4',
            'method': '20-agent reasoning engine',
        }
    if len(reasoned_selected) >= 5:
        final_parlays['reasoned_5leg'] = {
            'legs': [leg_summary_reasoned(l) for l in reasoned_selected[:5]],
            'description': 'REASONED 5-leg: Full agent-optimized parlay',
            'method': '20-agent reasoning engine',
        }

    # Also include strategy alternatives
    for v in strategy_verdicts:
        agent_name = v.get('agent', 'unknown')
        if v.get('legs') and len(v['legs']) >= 3:
            final_parlays[f'alt_{agent_name}'] = {
                'legs': v['legs'][:5],
                'description': f"ALT ({agent_name}): {v.get('strategy', '')}",
                'method': f'{agent_name} strategy agent',
            }

    # Include standard parlays too for comparison
    for name, parlay in standard_parlays.items():
        final_parlays[f'standard_{name}'] = parlay
        final_parlays[f'standard_{name}']['method'] = 'standard pipeline (baseline)'

    print(f"\n  FINAL: {len(final_parlays)} parlay builds generated")
    print(f"  Reasoning engine complete in {elapsed:.1f}s")

    return final_parlays


def print_summary(results, pass_num):
    tiers = {}
    for r in results:
        t = r.get('tier', 'SKIP')
        tiers[t] = tiers.get(t, 0) + 1

    errors = sum(1 for r in results if 'error' in r)
    overs = sum(1 for r in results if r.get('direction') == 'OVER' and 'error' not in r)
    unders = sum(1 for r in results if r.get('direction') == 'UNDER' and 'error' not in r)

    print(f"\n{'='*60}")
    print(f"  PASS {pass_num} RESULTS")
    print(f"{'='*60}")
    print(f"  Total: {len(results)} | Errors: {errors}")
    print(f"  Direction: {overs} OVER / {unders} UNDER")
    print(f"  Tiers (legacy): {json.dumps(tiers)}")

    # Show top picks by hit rate + ensemble_prob (replacing tier-based ranking)
    top_picks = [r for r in results if 'error' not in r and r.get('l10_hit_rate', 0) >= 70]
    top_picks.sort(key=lambda x: (x.get('ensemble_prob', x.get('xgb_prob', 0)) or 0), reverse=True)
    if top_picks:
        print(f"\n  TOP PICKS by ensemble_prob (L10 HR >= 70%, {len(top_picks)} total):")
        for r in top_picks[:15]:
            ens = r.get('ensemble_prob', r.get('xgb_prob', 0)) or 0
            print(f"    {r['player']:22s} {r['stat'].upper():4s} {r['direction']:5s} {r['line']:5.1f}  "
                  f"proj={r.get('projection',0):5.1f}  gap={r.get('gap',0):+5.1f}  "
                  f"L10HR={r.get('l10_hit_rate',0):3.0f}%  ens={ens:.3f}  {r.get('streak_status','')}")


def main():
    # Parse args
    date_str = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y-%m-%d')
    board_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 60)
    print(f"  NBA PROP ANALYSIS — {date_str}")
    print(f"  Pipeline v15 (Arena Ensemble + Matchup Scanner + NEXUS Primary)")
    print(f"  Auto research agents | No manual context needed")
    print("=" * 60)

    # ═══ PHASE 1: PARALLEL RESEARCH AGENTS ═══
    print("\n" + "=" * 60)
    print("  PHASE 1: DEPLOYING RESEARCH AGENTS")
    print("=" * 60)
    GAMES = research_all_games(date_str)  # auto-scales workers

    if not GAMES:
        print("[ERROR] No games found. Exiting.")
        return None, None

    # Save research
    save_research(GAMES, date_str)

    # ═══ PHASE 2: LOAD BOARD ═══
    if not board_path:
        # Try standard locations
        candidates = [
            '/tmp/parsed_board.json',
            os.path.join(os.path.dirname(__file__), date_str, f'{date_str}_parsed_board.json'),
        ]
        for c in candidates:
            if os.path.exists(c):
                board_path = c
                break

    if not board_path or not os.path.exists(board_path):
        print(f"\n[ERROR] No parsed board found. Provide path as 2nd argument.")
        print(f"  Tried: /tmp/parsed_board.json, predictions/{date_str}/{date_str}_parsed_board.json")
        print(f"\n  Research saved — run again with board path when ready.")
        # Still save the generated runner code
        code = generate_runner_code(GAMES, date_str)
        code_path = os.path.join(os.path.dirname(__file__), date_str, f"{date_str}_games_code.py")
        os.makedirs(os.path.dirname(code_path), exist_ok=True)
        with open(code_path, 'w') as f:
            f.write(code)
        print(f"  Generated runner code: {code_path}")
        return None, None

    with open(board_path) as f:
        picks = json.load(f)

    # ═══ FIX: Extract team from game field when missing ═══
    import re as _re
    fixed_teams = 0
    for p in picks:
        if not p.get('team') or p['team'] == '?':
            game = p.get('game', '')
            # "ORL @ ATL" → player team is ORL (first team)
            # "ATL vs ORL" → player team is ATL (first team)
            m = _re.match(r'^([A-Z]{2,4})\s*[@vs]+\s*([A-Z]{2,4})', game)
            if m:
                p['team'] = m.group(1)
                # Normalize game key to AWAY@HOME format for GAMES dict matching
                if ' vs ' in game or 'vs' in game.replace(m.group(1), '', 1):
                    # "ATL vs ORL" = ATL is home, ORL is away → game key = "ORL@ATL"
                    p['game'] = f"{m.group(2)}@{m.group(1)}"
                else:
                    # "ORL @ ATL" = ORL is away, ATL is home → game key = "ORL@ATL"
                    p['game'] = f"{m.group(1)}@{m.group(2)}"
                fixed_teams += 1
    if fixed_teams:
        print(f"  [FIX] Extracted team from game field for {fixed_teams}/{len(picks)} picks")

    # ═══ FIX: Map team→game when board has team but no game (MCP scrapers) ═══
    _pre_team_to_game = {}
    for _gk, _gv in GAMES.items():
        _pre_team_to_game[_gv.get('away_abr', '')] = _gk
        _pre_team_to_game[_gv.get('home_abr', '')] = _gk
    pre_game_resolved = 0
    for p in picks:
        if not p.get('game') and p.get('team') and p['team'] in _pre_team_to_game:
            p['game'] = _pre_team_to_game[p['team']]
            pre_game_resolved += 1
    if pre_game_resolved:
        print(f"  [FIX] Resolved game key from team for {pre_game_resolved}/{len(picks)} picks")

    # ═══ FIX: Deduplicate picks (parse_board sometimes produces dupes) ═══
    seen = set()
    deduped = []
    for p in picks:
        key = (p['player'], p['stat'], p['line'])
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    if len(deduped) < len(picks):
        print(f"  [FIX] Deduplicated: {len(picks)} → {len(deduped)} picks")
        picks = deduped

    print(f"\nLoaded {len(picks)} prop lines across {len(set(p['player'] for p in picks))} players")

    # ═══ SGO FEATURE ATTACHMENT ═══
    try:
        from sgo_client import SGOClient
        sgo = SGOClient()
        sgo_props = sgo.load_cached_props(date_str)
        if sgo_props:
            # Build lookup: (normalized_player, stat) → SGO data
            sgo_lookup = {}
            for sp in sgo_props:
                key = (sp['player'].lower().strip(), sp['stat'].lower().strip())
                sgo_lookup[key] = sp
            attached = 0
            for p in picks:
                key = (p['player'].lower().strip(), p.get('stat', '').lower().strip())
                if key in sgo_lookup:
                    sp = sgo_lookup[key]
                    book_lines = sp.get('book_lines', {})
                    if book_lines:
                        line_vals = [float(v) for v in book_lines.values()]
                        p['book_line_spread'] = max(line_vals) - min(line_vals)
                        fair_line = sp.get('line', p.get('line'))
                        p['line_vs_consensus'] = p.get('line', 0) - fair_line
                        p['n_books'] = len(book_lines)
                        attached += 1
            print(f"  [SGO] Attached sportsbook features to {attached}/{len(picks)} props")
        else:
            print(f"  [SGO] No cached props for {date_str}")
    except Exception as e:
        print(f"  [SGO] Feature attachment skipped: {e}")

    # ═══ PHASE 3: SELF-HEALING ═══
    corrections = load_corrections()
    if corrections:
        active = [c for c in corrections if c.get('status') == 'ACTIVE']
        if active:
            print(f"\n  [SELF-HEAL] {len(active)} active corrections loaded")

    # ═══ PHASE 4: PIPELINE ═══
    print("\n" + "=" * 60)
    print("  PHASE 4: PASS 1 — Full 14-Layer Pipeline")
    print("=" * 60)
    results = run_pipeline(picks, GAMES, pass_num=1)
    print_summary(results, 1)

    print("\n" + "=" * 60)
    print("  PASS 2: L5 Trend Validation")
    print("=" * 60)
    results = backtest_pass2(results)
    print_summary(results, 2)

    print("\n" + "=" * 60)
    print("  PASS 3: Opponent Matchup History")
    print("=" * 60)
    results = backtest_pass3(results)
    print_summary(results, 3)

    # ═══ PHASE 4c-CORR: CORRELATION ENRICHMENT (before ML scoring so models see enriched features) ═══
    try:
        from correlations import enrich_picks as corr_enrich
        corr_count = corr_enrich(results, GAMES=GAMES)
        print(f"\n  CORRELATIONS: Enriched {corr_count}/{len(results)} picks with matchup + same-game correlation data")
    except Exception as e:
        print(f"\n  CORRELATIONS: skipped ({e})")

    # (REMOVED: matchup_scanner, nn_embedder, market_signal, ref_model, coach_model — dead features, never used by XGBoost or parlay builders)

    try:
        from sim_model import enrich_with_sim
        sim_count = enrich_with_sim(results)
        print(f"  SIM MODEL: Enriched {sim_count}/{len(results)} picks with Monte Carlo sim_prob")
    except Exception as e:
        print(f"  SIM MODEL: skipped ({e})")

    # (REMOVED: line_movement — line_edge not consumed by any model or parlay builder)

    # ═══ PHASE 4c-FLOW: GAME FLOW MODELING (game script → player impact) ═══
    try:
        from game_flow import enrich_with_game_flow
        flow_count = enrich_with_game_flow(results, GAMES)
        flips = sum(1 for r in results if r.get('flow_adj', 0) != 0)
        print(f"  GAME FLOW: {flow_count} enriched, {flips} projections adjusted")
    except Exception as e:
        print(f"  GAME FLOW: skipped ({e})")

    # ═══ PHASE 4c-REG: REGRESSION SCORING (margin predictions) ═══
    try:
        from regression_model import score_regression
        reg_count = score_regression(results)
        if reg_count > 0:
            print(f"\n  REGRESSION: {reg_count}/{len(results)} props with margin predictions")
            # Show top margin picks
            margin_picks = sorted([r for r in results if r.get('reg_margin') is not None],
                                  key=lambda r: abs(r.get('reg_margin', 0)), reverse=True)[:5]
            for mp in margin_picks:
                print(f"    {mp['player']:22s} {mp['stat'].upper():4s} {mp['direction']:5s} "
                      f"line={mp['line']:5.1f}  pred={mp.get('reg_predicted',0):5.1f}  margin={mp.get('reg_margin',0):+5.1f}")
    except Exception as e:
        print(f"\n  REGRESSION: skipped ({e})")

    # (REMOVED: advanced_features — 100% NaN, never populated)

    # ═══ PHASE 4c-ML: XGBoost ML Scoring ═══
    try:
        from xgb_model import score_props
        results = score_props(results)
        scored = sum(1 for r in results if 'xgb_prob' in r)
        print(f"\n  XGBoost scoring: {scored}/{len(results)} props scored")
    except (ImportError, FileNotFoundError) as e:
        print(f"\n  XGBoost not available: {e}")
    except Exception as e:
        import traceback
        print(f"\n  XGBoost scoring failed: {e}")
        traceback.print_exc()

    # ═══ PHASE 4c-ML: MLP Neural Network Scoring ═══
    try:
        from mlp_model import score_props as mlp_score_props
        results = mlp_score_props(results)
        mlp_scored = sum(1 for r in results if r.get('mlp_prob') is not None)
        print(f"\n  MLP scoring: {mlp_scored}/{len(results)} props scored")
    except (ImportError, FileNotFoundError) as e:
        print(f"\n  MLP not available: {e}")
    except Exception as e:
        import traceback
        print(f"\n  MLP scoring failed: {e}")
        traceback.print_exc()

    # ═══ Ensemble: 60% XGBoost + 40% Sim (v16: replaced MLP — sim AUC 0.564 > MLP dead/redundant) ═══
    # MLP uses same 136 features as XGB → redundant. Sim uses Monte Carlo on L10 distributions → real diversity.
    for r in results:
        xgb = r.get('xgb_prob')
        sim = r.get('sim_prob')
        mlp = r.get('mlp_prob')
        if xgb is not None and sim is not None:
            r['ensemble_prob'] = round(0.6 * xgb + 0.4 * sim, 4)
            r['models_used'] = 2
        elif xgb is not None and mlp is not None:
            # Fallback to MLP if sim unavailable
            r['ensemble_prob'] = round(0.6 * xgb + 0.4 * mlp, 4)
            r['models_used'] = 2
        elif xgb is not None:
            r['ensemble_prob'] = xgb
            r['models_used'] = 1

    # ═══ PHASE 4b: PRE-GAME AVAILABILITY CHECK ═══
    filtered_results, pregame_report = run_pregame_check(results, GAMES, game_date=date_str)

    # Save pregame report
    pregame_file = os.path.join(os.path.dirname(__file__), date_str, "pregame_report.json")
    os.makedirs(os.path.dirname(pregame_file), exist_ok=True)
    with open(pregame_file, 'w') as f:
        json.dump(pregame_report, f, indent=2)

    # ═══ PHASE 5: PRIMARY PARLAYS (Engine v1.2 — validated UNDER bias) ═══
    engine_parlays = build_primary_parlays(filtered_results)

    # Day signal from 1M simulation classifier
    day_signal = engine_parlays.get('day_signal', {})
    print(f"\n{'='*60}")
    print(f"  DAY SIGNAL: {day_signal.get('reason', 'N/A')}")
    print(f"  Qualifying props: {day_signal.get('qualifying_props', '?')} | Games: {day_signal.get('qualifying_games', '?')}")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"  PRIMARY PICKS (Engine v1.2 — Validated UNDER Bias)")
    print(f"{'='*60}")
    for pname, parlay in engine_parlays.items():
        if pname == 'day_signal':
            continue
        print(f"\n  {parlay['name']}:")
        print(f"  {parlay.get('description', '')}")
        for leg in parlay.get('legs', []):
            gap = leg.get('gap', 0)
            hr = leg.get('l10_hit_rate', 0)
            l5 = leg.get('l5_hit_rate', 0)
            xgb = leg.get('xgb_prob')
            xgb_tag = f"  xgb={xgb:.3f}" if xgb is not None else ''
            ps = leg.get('primary_score', 0)
            print(f"    {leg['player']:22s} {leg.get('stat','?').upper():4s} "
                  f"{leg.get('direction','?'):5s} {leg.get('line',0):5.1f}  "
                  f"gap={gap:+.1f}  L10={hr:3.0f}% L5={l5:3.0f}%{xgb_tag}  ps={ps:.3f}")

    # ═══ PHASE 5b: NEXUS SECONDARY (shadow — multi-agent consensus) ═══
    nexus_parlays = nexus_v4_pipeline(filtered_results, GAMES)

    # Collect NEXUS parlays for reference
    nexus_named = {}
    for name, parlay in nexus_parlays.items():
        if not name.startswith('_') and parlay is not None:
            nexus_named[name] = parlay

    # Convert NEXUS output to named format
    primary_parlays = _nexus_to_primary(nexus_parlays)

    print(f"\n{'='*60}")
    print(f"  NEXUS SECONDARY (Multi-Agent Consensus — shadow)")
    print(f"{'='*60}")
    for pname, parlay in primary_parlays.items():
        print(f"\n  {parlay['name']}:")
        print(f"  {parlay.get('description', '')}")
        for leg in parlay.get('legs', []):
            gap = leg.get('gap', 0)
            hr = leg.get('l10_hit_rate', 0)
            l5 = leg.get('l5_hit_rate', 0)
            ns = leg.get('nexus_score', 0)
            tier_tag = leg.get('screen_tier', '')
            print(f"    {leg['player']:22s} {leg.get('stat','?').upper():4s} "
                  f"{leg.get('direction','?'):5s} {leg.get('line',0):5.1f}  "
                  f"gap={gap:+.1f}  L10={hr:3.0f}% L5={l5:3.0f}%  ns={ns:.1f}  [{tier_tag}]")

    # ═══ PHASE 5b-TRIPLE: TRIPLE-SAFE (3 independent SAFE parlays) ═══
    triple_safe = build_triple_safe(filtered_results)

    print(f"\n{'='*60}")
    print(f"  TRIPLE-SAFE (3 independent SAFE parlays — 98.9% at-least-1 rate)")
    print(f"{'='*60}")
    for pname, parlay in triple_safe.items():
        print(f"\n  {parlay['name']}:")
        print(f"  {parlay.get('description', '')}")
        for leg in parlay.get('legs', []):
            gap = leg.get('gap', 0)
            hr = leg.get('l10_hit_rate', 0)
            l5 = leg.get('l5_hit_rate', 0)
            ss = leg.get('sniper_score', 0)
            fs = leg.get('floor_score', 0)
            cs = leg.get('composite_score', 0)
            print(f"    {leg['player']:22s} {leg.get('stat','?').upper():4s} "
                  f"{leg.get('direction','?'):5s} {leg.get('line',0):5.1f}  "
                  f"gap={gap:+.1f}  L10={hr:3.0f}% L5={l5:3.0f}%  "
                  f"sniper={ss:.0f} floor={fs:.2f} comp={cs:.0f}")

    # ═══ PHASE 5c: EV ANALYSIS (expected value, not just accuracy) ═══
    try:
        from ev_optimizer import enrich_with_ev, print_ev_report
        ev_count = enrich_with_ev(filtered_results)
        if ev_count > 0:
            pos_ev = sum(1 for r in filtered_results if r.get('is_positive_ev'))
            print(f"\n  EV ANALYSIS: {pos_ev}/{ev_count} props are +EV")
            print_ev_report(filtered_results)
    except Exception as e:
        print(f"\n  EV ANALYSIS: skipped ({e})")

    # ═══ PHASE 5d: CORRELATION-OPTIMIZED PARLAY ═══
    try:
        from parlay_optimizer import build_optimal_parlay, score_parlay_independence
        corr_safe = build_optimal_parlay(filtered_results, n_legs=3, mode='safe')
        if corr_safe and corr_safe.get('legs'):
            adj_prob = corr_safe.get('adjusted_parlay_prob', 0)
            indep = corr_safe.get('independence_score', 0)
            print(f"\n{'='*60}")
            print(f"  CORRELATION-OPTIMIZED 3-LEG (independence={indep:.2f}, adj_prob={adj_prob:.1%})")
            print(f"{'='*60}")
            for leg in corr_safe['legs']:
                print(f"    {leg.get('player','?'):22s} {leg.get('stat','?').upper():4s} "
                      f"{leg.get('direction','?'):5s} {leg.get('line',0):5.1f}  "
                      f"prob={leg.get('prob',0):.3f}  margin={leg.get('reg_margin',0):+.1f}")
    except Exception as e:
        corr_safe = None
        print(f"\n  CORR OPTIMIZER: skipped ({e})")

    # ═══ PHASE 5d: 100 SHADOW PARLAYS ═══
    shadow_parlays = build_100_shadow_parlays(filtered_results)
    # Append NEXUS shadow parlays too (existing strategies)
    nexus_shadows = nexus_parlays.get('_shadow_parlays', [])
    shadow_parlays.extend(nexus_shadows)

    # Append Triple-SAFE as shadow strategies
    for sname, sparlay in triple_safe.items():
        shadow_parlays.append({
            'strategy_name': sname,
            'strategy_id': sname,
            'strategy_description': sparlay.get('description', ''),
            'legs': sparlay.get('legs', []),
            'confidence': 78.0,
            'legs_total': sparlay.get('legs_total', 3),
            'result': None,
            'legs_hit': None,
        })

    # Append correlation-optimized as a shadow strategy
    if corr_safe and corr_safe.get('legs'):
        shadow_parlays.append({
            'strategy_name': 'corr_optimized',
            'strategy_id': 'corr_optimized',
            'strategy_description': 'Correlation-optimized: maximizes parlay prob via uncorrelated legs',
            'legs': corr_safe['legs'],
            'confidence': round(corr_safe.get('adjusted_parlay_prob', 0) * 100, 1),
            'legs_total': len(corr_safe['legs']),
            'result': None,
            'legs_hit': None,
        })

    print(f"  Total shadow parlays: {len(shadow_parlays)} ({len(shadow_parlays) - len(nexus_shadows)} engine + {len(nexus_shadows)} NEXUS)")
    print(f"  Leaderboard tracked at shadow_parlay_tracker.json")

    # ═══ GAME LOCKS ═══
    game_locks = build_game_locks(results)

    print("\n" + "=" * 60)
    print("  GAME LOCKS (Best single pick per game)")
    print("=" * 60)
    for game, lock in sorted(game_locks.items()):
        conf = lock.get('confidence', 'STRONG')
        tag = "LOCK" if conf == 'ABSOLUTE' else "LEAN"
        print(f"  [{tag:4s}] {game:12s} | {lock['player']:20s} | {lock['direction']} {lock['stat'].upper()} {lock['line']} "
              f"| Tier {lock['tier']} | Gap {lock['gap']:+.1f} | L10 {lock['l10_hit_rate']}% | L5 {lock['l5_hit_rate']}%")

    # ═══ SAVE ═══
    output_dir = os.path.join(os.path.dirname(__file__), date_str)
    os.makedirs(output_dir, exist_ok=True)

    board_file = os.path.join(output_dir, f"{date_str}_full_board.json")
    with open(board_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved full board: {board_file}")

    # Save primary parlays (Engine — what user bets on)
    primary_file = os.path.join(output_dir, "primary_parlays.json")
    with open(primary_file, 'w') as f:
        json.dump({
            'date': date_str,
            'generated_at': datetime.now().isoformat(),
            'method': 'parlay_engine_v1',
            'parlays': engine_parlays,
        }, f, indent=2)
    print(f"  Saved primary parlays: {primary_file}")

    # Save NEXUS secondary parlays
    engine_file = os.path.join(output_dir, "engine_parlays.json")
    nexus_secondary = _nexus_to_primary(nexus_parlays)
    with open(engine_file, 'w') as f:
        json.dump({
            'date': date_str,
            'generated_at': datetime.now().isoformat(),
            'method': 'nexus_v4_secondary',
            'parlays': nexus_secondary,
        }, f, indent=2)
    print(f"  Saved NEXUS secondary: {engine_file}")

    # Save Triple-SAFE parlays
    triple_file = os.path.join(output_dir, "triple_safe_parlays.json")
    with open(triple_file, 'w') as f:
        json.dump({
            'date': date_str,
            'generated_at': datetime.now().isoformat(),
            'method': 'sniper_v3_triple_safe',
            'parlays': triple_safe,
        }, f, indent=2)
    print(f"  Saved triple-safe: {triple_file}")

    # Save raw NEXUS parlays (full reference with all tiers)
    nexus_file = os.path.join(output_dir, "nexus_parlays.json")
    with open(nexus_file, 'w') as f:
        json.dump(nexus_parlays, f, indent=2)
    print(f"  Saved NEXUS reference: {nexus_file}")

    # Save all shadow parlays (100 engine + NEXUS shadows)
    if shadow_parlays:
        shadow_file = os.path.join(output_dir, "shadow_parlays.json")
        shadow_output = {
            'date': date_str,
            'generated_at': datetime.now().isoformat(),
            'total_strategies': len(shadow_parlays),
            'shadow_parlays': shadow_parlays,
        }
        with open(shadow_file, 'w') as f:
            json.dump(shadow_output, f, indent=2)
        print(f"  Saved shadow parlays: {shadow_file} ({len(shadow_parlays)} strategies)")

    # Save combined (NEXUS for backward compat)
    parlay_file = os.path.join(output_dir, "final_parlays.json")
    with open(parlay_file, 'w') as f:
        json.dump(nexus_named, f, indent=2)
    print(f"  Saved NEXUS combined: {parlay_file}")

    locks_file = os.path.join(output_dir, "game_locks.json")
    with open(locks_file, 'w') as f:
        json.dump(game_locks, f, indent=2)
    print(f"  Saved game locks: {locks_file}")

    # Save summary
    tiers = {}
    for r in results:
        t = r.get('tier', 'SKIP')
        tiers[t] = tiers.get(t, 0) + 1

    summary = {
        'date': date_str,
        'total_lines': len(results),
        'errors': sum(1 for r in results if 'error' in r),
        'tier_distribution': tiers,
        'pipeline_version': 'v15+arena+scanner',
        'backtest_passes': 3,
        'research_method': 'parallel_agents',
        'games_researched': len(GAMES),
        'v5_features': [
            'Parallel research agents (1 per team, concurrent)',
            'Auto injury report fetch',
            'Auto spread/odds from ESPN',
            'Auto B2B detection from schedule',
            'Auto breaking news scan',
            'No manual game context required',
            'UNDER penalty removed',
            'Reasoning on every pick',
        ],
        'game_contexts': {k: v['notes'] for k, v in GAMES.items()},
        'timestamp': datetime.now().isoformat(),
    }
    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print board by game
    print("\n" + "=" * 60)
    print("  FULL BOARD BY GAME")
    print("=" * 60)

    for game_key in GAMES:
        game_results = [r for r in results if r.get('game') == game_key and 'error' not in r]
        if not game_results:
            continue

        gctx = GAMES[game_key]
        print(f"\n  {'─'*55}")
        spread_str = f"{gctx['spread']:+.1f}" if gctx['spread'] else "N/A"
        print(f"  {game_key} | Spread: {spread_str} | {gctx['notes'][:80]}")
        print(f"  {'─'*55}")

        # Sort by ensemble_prob (or xgb_prob) descending — no tier gating
        game_results.sort(key=lambda x: (x.get('ensemble_prob', x.get('xgb_prob', 0)) or 0), reverse=True)

        for r in game_results:
            streak_tag = f" [{r.get('streak_status', '')}]" if r.get('streak_status', 'NEUTRAL') != 'NEUTRAL' else ''
            ens = r.get('ensemble_prob', r.get('xgb_prob', 0)) or 0
            print(f"    {r['player']:22s} {r['stat'].upper():4s} {r['direction']:5s} {r['line']:5.1f}  "
                  f"proj={r.get('projection',0):5.1f}  gap={r.get('gap',0):+5.1f}  "
                  f"L10={r.get('l10_hit_rate',0):3.0f}%  ens={ens:.3f}{streak_tag}")

    print(f"\n\nDONE! All results saved to predictions/{date_str}/")
    return results, engine_parlays


if __name__ == '__main__':
    results, parlays = main()

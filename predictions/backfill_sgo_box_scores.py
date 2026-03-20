#!/usr/bin/env python3
"""
SGO Box Score Backfill — Training data reconstruction from 342K SGO box score records.

Different data source than nba_api backfill (cross-source diversity).
Has real plus_minus, covers 2+ seasons (2024-02-01 to 2026-03-18).

For each of ~795 players with 11+ games:
1. Infer player team from game strings
2. Compute rolling stats from PRIOR games only (no data leakage)
3. Generate prop lines from L10 avg (snap to 0.5, small noise)
4. Label from actuals (HIT if actual > line for OVER, actual < line for UNDER)
5. Compute context features: home/away, B2B, matchup, travel, streak, plus_minus
6. Generate OVER + UNDER records per prop (2x volume)
7. Join Phase 1 spreads/totals if available

Output: predictions/cache/sgo_backfill_training_data.json

Usage:
    python3 predictions/backfill_sgo_box_scores.py              # Generate SGO backfill
    python3 predictions/backfill_sgo_box_scores.py --stats      # Print stats only
"""

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PREDICTIONS_DIR, 'cache')
SGO_BOX_PATH = os.path.join(CACHE_DIR, 'sgo', 'season_box_scores.json')
HISTORICAL_EVENTS_PATH = os.path.join(CACHE_DIR, 'sgo', 'historical_events.json')
OUTPUT_PATH = os.path.join(CACHE_DIR, 'sgo_backfill_training_data.json')

# Stats available in SGO box scores
BASE_STATS = {'pts': 'pts', 'reb': 'reb', 'ast': 'ast', '3pm': '3pm', 'stl': 'stl', 'blk': 'blk'}
COMBO_STATS = {
    'pra': ['pts', 'reb', 'ast'],
    'pr': ['pts', 'reb'],
    'pa': ['pts', 'ast'],
    'ra': ['reb', 'ast'],
}

# SGO abbreviation → team_rankings nickname
ABR_TO_NICKNAME = {
    'ATL': 'Hawks', 'BOS': 'Celtics', 'BKN': 'Nets', 'CHA': 'Hornets',
    'CHI': 'Bulls', 'CLE': 'Cavaliers', 'DAL': 'Mavericks', 'DEN': 'Nuggets',
    'DET': 'Pistons', 'GSW': 'Warriors', 'HOU': 'Rockets', 'IND': 'Pacers',
    'LAC': 'Clippers', 'LAL': 'Lakers', 'MEM': 'Grizzlies', 'MIA': 'Heat',
    'MIL': 'Bucks', 'MIN': 'Timberwolves', 'NOP': 'Pelicans', 'NYK': 'Knicks',
    'OKC': 'Thunder', 'ORL': 'Magic', 'PHI': '76ers', 'PHX': 'Suns',
    'POR': 'Trail Blazers', 'SAC': 'Kings', 'SAS': 'Spurs', 'TOR': 'Raptors',
    'UTA': 'Jazz', 'WAS': 'Wizards',
}


# Tier thresholds from analyze_v3.py
def _compute_tier(abs_gap, is_combo=False):
    g = abs_gap
    if is_combo:
        g -= 0.5
    if g >= 4: return 'S'
    if g >= 3: return 'A'
    if g >= 2: return 'B'
    if g >= 1.5: return 'C'
    if g >= 1: return 'D'
    return 'F'


def _snap_line(val):
    """Snap to nearest 0.5 like sportsbooks."""
    return round(val * 2) / 2


def _load_sgo_box_scores():
    """Load 342K SGO box scores."""
    if not os.path.exists(SGO_BOX_PATH):
        print(f"  ERROR: No SGO box scores at {SGO_BOX_PATH}")
        return []
    with open(SGO_BOX_PATH) as f:
        return json.load(f)


def _load_historical_events():
    """Load historical events (spreads, totals) from Phase 1 cache."""
    if not os.path.exists(HISTORICAL_EVENTS_PATH):
        return {}
    with open(HISTORICAL_EVENTS_PATH) as f:
        data = json.load(f)
    events = data.get('events', [])
    # Index by date+game for quick lookup
    lookup = {}
    for e in events:
        key = f"{e['date']}_{e['game']}"
        lookup[key] = e
    return lookup


def _load_team_rankings():
    """Load cached team rankings for matchup adjustments."""
    path = os.path.join(CACHE_DIR, 'team_rankings_2025-26.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    return data.get('teams', {}), data.get('league_avg', {})


def _load_venue_data():
    """Import venue_data for travel/timezone computations."""
    sys.path.insert(0, PREDICTIONS_DIR)
    from venue_data import VENUE_MAP, TZ_ORDINAL, haversine_miles
    return VENUE_MAP, TZ_ORDINAL, haversine_miles


def _infer_player_teams(records_by_player):
    """Infer each player's team from game strings.

    The player's team appears in every game they play. Count team abbreviation
    frequency — the one matching total games is their team.
    For traded players, use most frequent team per date range.
    """
    player_teams = {}
    for player, games in records_by_player.items():
        team_counts = Counter()
        for g in games:
            game_str = g['game']
            parts = game_str.split('@')
            if len(parts) == 2:
                team_counts[parts[0]] += 1
                team_counts[parts[1]] += 1

        n_games = len(games)
        # Team appearing in all games = player's team
        candidates = [(t, c) for t, c in team_counts.items() if c == n_games]
        if candidates:
            player_teams[player] = candidates[0][0]
        elif team_counts:
            # Traded player — use most frequent
            player_teams[player] = team_counts.most_common(1)[0][0]
        else:
            player_teams[player] = '?'

    return player_teams


def _matchup_adjustment(opp_team, stat_name, team_rankings, league_avg):
    """Compute rate-based matchup adjustment like analyze_v3 v5."""
    # Convert SGO abbreviation to team_rankings nickname
    opp_team = ABR_TO_NICKNAME.get(opp_team, opp_team)
    if not team_rankings or not opp_team or opp_team not in team_rankings:
        return 0.0, None, None

    opp = team_rankings[opp_team]
    stat_to_allowed = {
        'pts': 'avg_pts_allowed', 'reb': 'reb_allowed', 'ast': 'ast_allowed',
        '3pm': 'tpm_allowed', 'stl': 'stl_allowed', 'blk': 'blk_allowed',
    }

    if stat_name in COMBO_STATS:
        components = {'pra': ['pts', 'reb', 'ast'], 'pr': ['pts', 'reb'],
                      'pa': ['pts', 'ast'], 'ra': ['reb', 'ast']}
        adjs = [_matchup_adjustment(opp_team, s, team_rankings, league_avg)[0]
                for s in components[stat_name]]
        return sum(adjs) / len(adjs) if adjs else 0.0, None, None

    allowed_key = stat_to_allowed.get(stat_name)
    if not allowed_key:
        return 0.0, None, None

    opp_allowed = opp.get(allowed_key, 0)
    lg_avg = league_avg.get(allowed_key, 0) if league_avg else 0

    if lg_avg <= 0:
        return 0.0, opp_allowed, 0.0

    diff = opp_allowed - lg_avg
    rate_vs_avg = diff / lg_avg if lg_avg > 0 else 0
    adjustment = diff / 75.0

    return round(adjustment, 2), round(opp_allowed, 1), round(rate_vs_avg, 3)


def _get_stat_val(record, stat_name):
    """Get stat value from an SGO box score record."""
    if stat_name in BASE_STATS:
        return float(record.get(BASE_STATS[stat_name], 0))
    elif stat_name in COMBO_STATS:
        return float(sum(record.get(col, 0) for col in COMBO_STATS[stat_name]))
    return 0.0


def generate_sgo_backfill():
    """Main backfill generation from SGO box scores."""
    print("=" * 60)
    print("  SGO Box Score Training Data Backfill")
    print("=" * 60)

    # Load all resources
    print("\n  Loading SGO box scores...")
    raw_records = _load_sgo_box_scores()
    if not raw_records:
        return []
    print(f"  Loaded {len(raw_records):,} box score records")

    # Load historical events for spread/total enrichment
    events_lookup = _load_historical_events()
    print(f"  Historical events: {len(events_lookup)} games" if events_lookup else "  Historical events: not available (run --history first)")

    team_rankings, league_avg = _load_team_rankings()
    print(f"  Team rankings: {'loaded' if team_rankings else 'NOT FOUND'}")

    try:
        venue_map, tz_ordinal, haversine_fn = _load_venue_data()
        has_venue = True
        print(f"  Venue data: loaded ({len(venue_map)} arenas)")
    except Exception as e:
        has_venue = False
        venue_map, tz_ordinal, haversine_fn = {}, {}, None
        print(f"  Venue data: FAILED ({e})")

    # Group by player, sorted by date
    print("\n  Grouping by player...")
    by_player = defaultdict(list)
    for r in raw_records:
        by_player[r['player']].append(r)

    # Sort each player's games by date
    for player in by_player:
        by_player[player].sort(key=lambda x: x['date'])

    # Infer player teams
    player_teams = _infer_player_teams(by_player)
    print(f"  Players: {len(by_player):,} total, {sum(1 for p, g in by_player.items() if len(g) >= 11)} with 11+ games")

    all_stats = list(BASE_STATS.keys()) + list(COMBO_STATS.keys())
    all_records = []
    rng = np.random.RandomState(43)  # different seed from nba_api backfill

    t0 = time.time()
    player_count = 0

    for player, games in by_player.items():
        # Need at least 15 games (11 for L10 window + a few)
        if len(games) < 15:
            continue

        player_team = player_teams.get(player, '?')
        player_count += 1

        # Parse game dates
        game_dates = []
        for g in games:
            try:
                game_dates.append(datetime.strptime(g['date'], '%Y-%m-%d'))
            except (ValueError, TypeError):
                game_dates.append(None)

        for stat_name in all_stats:
            is_combo = stat_name in COMBO_STATS

            # Build stat values array
            stat_vals = np.array([_get_stat_val(g, stat_name) for g in games])

            # Start from game 11 (index 10) so we have L10 history
            for idx in range(10, len(games)):
                game = games[idx]
                actual = float(stat_vals[idx])
                game_date_str = game['date']
                game_dt = game_dates[idx]
                mins = float(game.get('min', 0))

                # Skip low-minute games
                if mins < 10:
                    continue

                # Prior games only (no future leakage)
                prior_vals = stat_vals[:idx]
                # Filter prior games with meaningful minutes
                prior_mins = np.array([float(games[i].get('min', 0)) for i in range(idx)])
                valid_mask = prior_mins >= 10
                if valid_mask.sum() < 10:
                    continue

                # Use all prior values for stats (even low-min games contribute to averages)
                l10_vals = prior_vals[-10:]
                l5_vals = prior_vals[-5:]
                l3_vals = prior_vals[-3:]

                season_avg = float(np.mean(prior_vals))
                l10_avg = float(np.mean(l10_vals))
                l5_avg = float(np.mean(l5_vals))
                l3_avg = float(np.mean(l3_vals))

                # Generate realistic prop line
                l10_std = float(np.std(l10_vals))
                noise = rng.normal(0, max(l10_std * 0.2, 0.25))
                line = _snap_line(l10_avg + noise)
                if line <= 0:
                    continue

                # Skip pushes
                if actual == line:
                    continue

                # Home/away detection
                game_str = game['game']
                parts = game_str.split('@')
                if len(parts) == 2:
                    away_team = parts[0]
                    home_team = parts[1]
                    is_home = 1 if player_team == home_team else 0
                    opp_team = away_team if is_home else home_team
                else:
                    is_home = 0
                    opp_team = '?'

                # Home/away splits from prior games
                home_vals = []
                away_vals = []
                for i in range(idx):
                    g_str = games[i]['game']
                    g_parts = g_str.split('@')
                    if len(g_parts) == 2:
                        if player_team == g_parts[1]:
                            home_vals.append(stat_vals[i])
                        else:
                            away_vals.append(stat_vals[i])

                home_avg = float(np.mean(home_vals)) if len(home_vals) > 3 else season_avg
                away_avg = float(np.mean(away_vals)) if len(away_vals) > 3 else season_avg

                # Hit rates vs line
                l10_hit_rate = float(np.sum(l10_vals > line) / len(l10_vals) * 100)
                l5_hit_rate = float(np.sum(l5_vals > line) / len(l5_vals) * 100)
                season_hit_rate = float(np.sum(prior_vals > line) / len(prior_vals) * 100)

                # Minutes consistency
                valid_prior_mins = prior_mins[prior_mins >= 10]
                mins_30plus_pct = float(np.sum(valid_prior_mins >= 30) / len(valid_prior_mins) * 100) if len(valid_prior_mins) > 0 else 0

                # L10 floor & miss count
                l10_floor = float(np.min(l10_vals))
                l10_miss_count = int(np.sum(l10_vals <= line))

                # Streak detection
                streak_pct = float((l3_avg - l10_avg) / l10_avg * 100) if l10_avg > 0 else 0
                if streak_pct > 15:
                    streak_status = 'HOT'
                elif streak_pct < -15:
                    streak_status = 'COLD'
                else:
                    streak_status = 'NEUTRAL'

                # B2B detection
                is_b2b = False
                if idx > 0 and game_dt and game_dates[idx - 1]:
                    is_b2b = (game_dt - game_dates[idx - 1]).days <= 1

                # Projection (same formula as analyze_v3)
                projection = 0.4 * l10_avg + 0.3 * l5_avg + 0.3 * season_avg

                # Split adjustment
                if is_home and len(home_vals) > 3:
                    split_adj = round(float(home_avg - season_avg) * 0.5, 2)
                elif not is_home and len(away_vals) > 3:
                    split_adj = round(float(away_avg - season_avg) * 0.5, 2)
                else:
                    split_adj = 0.0

                # Matchup adjustment
                matchup_adj, opp_allowed_rate, opp_allowed_vs_league = _matchup_adjustment(
                    opp_team, stat_name, team_rankings, league_avg
                )

                # Plus/minus from L10 (directly available in SGO data)
                l10_games = games[max(0, idx-10):idx]
                l10_plus_minus = round(float(np.mean([
                    float(g.get('plus_minus', 0)) for g in l10_games
                ])), 1)

                # Minutes stats
                season_mins = float(np.mean(valid_prior_mins)) if len(valid_prior_mins) > 0 else 0
                l5_games_mins = [float(games[i].get('min', 0)) for i in range(max(0, idx-5), idx)]
                l5_mins = float(np.mean(l5_games_mins)) if l5_games_mins else 0
                mins_adj = round((l5_mins - season_mins) / season_mins * 0.5, 2) if season_mins > 0 else 0.0

                # Travel features
                travel_dist = None
                travel_7day = None
                tz_shifts = None
                if has_venue and idx > 0:
                    # Current game venue
                    game_venue = home_team if len(parts) == 2 else '?'
                    # Previous game venue
                    prev_game = games[idx - 1]['game']
                    prev_parts = prev_game.split('@')
                    prev_venue = prev_parts[1] if len(prev_parts) == 2 else '?'

                    if prev_venue in venue_map and game_venue in venue_map:
                        pv = venue_map[prev_venue]
                        cv = venue_map[game_venue]
                        travel_dist = round(haversine_fn(pv['lat'], pv['lng'], cv['lat'], cv['lng']))

                    # 7-day travel
                    if game_dt:
                        week_ago = game_dt - timedelta(days=7)
                        total_miles = 0.0
                        tz_shift_count = 0
                        prev_tz = None
                        prev_v = None

                        for j in range(max(0, idx - 10), idx + 1):
                            if game_dates[j] and game_dates[j] >= week_ago:
                                g_str = games[j]['game']
                                g_parts = g_str.split('@')
                                cur_v = g_parts[1] if len(g_parts) == 2 else None
                                if cur_v and prev_v and cur_v in venue_map and prev_v in venue_map:
                                    total_miles += haversine_fn(
                                        venue_map[prev_v]['lat'], venue_map[prev_v]['lng'],
                                        venue_map[cur_v]['lat'], venue_map[cur_v]['lng']
                                    )
                                if cur_v:
                                    cur_tz = venue_map.get(cur_v, {}).get('tz')
                                    if cur_tz and prev_tz and cur_tz != prev_tz:
                                        tz_shift_count += 1
                                    if cur_tz:
                                        prev_tz = cur_tz
                                    prev_v = cur_v

                        travel_7day = round(total_miles)
                        tz_shifts = tz_shift_count

                # B2B travel adjustment
                b2b_adj = 0.0
                if is_b2b and travel_dist is not None:
                    if travel_dist > 1500:
                        b2b_adj = -0.04
                    elif travel_dist > 500:
                        b2b_adj = -0.02
                    else:
                        b2b_adj = -0.01

                # Usage rate (from prior games)
                usage_rate = None
                usage_trend = None
                l10_fga = [float(games[i].get('fga', 0)) for i in range(max(0, idx-10), idx) if float(games[i].get('min', 0)) >= 10]
                # SGO box scores don't have FGA/FTA/TOV — skip usage computation
                # (this is a limitation vs nba_api backfill)

                # Spread and game_total from historical events
                spread = None
                game_total_val = None
                event_key = f"{game_date_str}_{game_str}"
                event_data = events_lookup.get(event_key)
                if event_data:
                    spread = event_data.get('spread')
                    game_total_val = event_data.get('game_total')

                # Gap and tier
                gap = projection - line
                abs_gap = abs(gap)
                tier = _compute_tier(abs_gap, is_combo)

                # Build base record
                rec_base = {
                    'player': player,
                    'stat': stat_name,
                    'line': line,
                    'projection': round(projection, 1),
                    'gap': round(gap, 1),
                    'abs_gap': round(abs_gap, 1),
                    'effective_gap': round(gap, 1),
                    'season_avg': round(season_avg, 1),
                    'l10_avg': round(l10_avg, 1),
                    'l5_avg': round(l5_avg, 1),
                    'l3_avg': round(l3_avg, 1),
                    'home_avg': round(home_avg, 1),
                    'away_avg': round(away_avg, 1),
                    'l10_hit_rate': round(l10_hit_rate),
                    'l5_hit_rate': round(l5_hit_rate),
                    'season_hit_rate': round(season_hit_rate),
                    'mins_30plus_pct': round(mins_30plus_pct),
                    'split_adjustment': split_adj,
                    'matchup_adjustment': matchup_adj,
                    'mins_adj': round(mins_adj, 2),
                    'streak_adj': 0.0,
                    'blowout_adj': 0.0,
                    'injury_adjustment': 0,
                    'spread': spread,
                    'streak_pct': round(streak_pct, 1),
                    'games_used': int(idx),
                    'is_home': is_home,
                    'is_b2b': is_b2b,
                    'l10_floor': round(l10_floor, 1),
                    'l10_miss_count': l10_miss_count,
                    'l10_std': round(l10_std, 2),
                    'l10_values': [round(float(v), 1) for v in l10_vals],
                    'streak_status': streak_status,
                    'tier': tier,
                    'opponent_history': None,
                    'same_team_out_count': 0,
                    # v4 features
                    'l10_avg_plus_minus': l10_plus_minus,
                    'l10_avg_pf': 0.0,  # PF not in SGO box scores
                    'foul_trouble_risk': False,
                    # v5 features
                    'opp_stat_allowed_rate': opp_allowed_rate,
                    'opp_stat_allowed_vs_league_avg': opp_allowed_vs_league,
                    'usage_rate': usage_rate,
                    'usage_trend': usage_trend,
                    'dynamic_without_delta': None,
                    'travel_distance': travel_dist,
                    'travel_miles_7day': travel_7day,
                    'tz_shifts_7day': tz_shifts,
                    # v7 features
                    'opp_matchup_delta': None,
                    'team_vs_opp_delta': None,
                    'opp_off_pressure': None,
                    'usage_boost': None,
                    'game_total_signal': None,
                    'max_same_game_corr': None,
                    # Metadata
                    '_date': game_date_str,
                    '_source': 'sgo_backfill',
                    '_data_source': 'sgo_backfill',
                    '_feature_version': 7,
                }

                # Compute game_total_signal if we have the data
                if game_total_val is not None:
                    if game_total_val >= 235:
                        rec_base['game_total_signal'] = 1.0
                    elif game_total_val <= 215:
                        rec_base['game_total_signal'] = -1.0
                    else:
                        rec_base['game_total_signal'] = 0.0

                # OVER record
                over_rec = dict(rec_base)
                over_rec['direction'] = 'OVER'
                over_rec['_hit_label'] = actual > line
                all_records.append(over_rec)

                # UNDER record
                under_rec = dict(rec_base)
                under_rec['direction'] = 'UNDER'
                under_rec['effective_gap'] = round(line - projection, 1)
                under_rec['_hit_label'] = actual < line
                all_records.append(under_rec)

        if player_count % 50 == 0:
            elapsed = time.time() - t0
            print(f"  ... {player_count} players, {len(all_records):,} records ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Generated {len(all_records):,} total records from {player_count} players in {elapsed:.1f}s")

    _print_stats(all_records)
    return all_records


def _print_stats(records):
    """Print summary statistics."""
    if not records:
        print("  No records generated.")
        return

    n = len(records)
    hits = sum(1 for r in records if r.get('_hit_label'))
    dates = sorted(set(r['_date'] for r in records))
    stats = Counter(r['stat'] for r in records)
    tiers = Counter(r['tier'] for r in records)
    directions = Counter(r['direction'] for r in records)

    # Feature coverage
    context_fields = [
        'matchup_adjustment', 'split_adjustment', 'mins_adj',
        'opp_stat_allowed_rate', 'travel_distance', 'l10_avg_plus_minus',
        'spread', 'game_total_signal',
    ]
    coverage = {}
    for field in context_fields:
        non_null = sum(1 for r in records
                       if r.get(field) is not None
                       and r.get(field) != 0
                       and not (isinstance(r.get(field), float) and np.isnan(r.get(field))))
        coverage[field] = round(non_null / n * 100, 1)

    print(f"\n  Summary:")
    print(f"    Records:     {n:,}")
    print(f"    Hit rate:    {hits/n:.1%}")
    print(f"    Dates:       {dates[0]} → {dates[-1]} ({len(dates)} unique)")
    print(f"    Directions:  {', '.join(f'{k}={v}' for k, v in sorted(directions.items()))}")
    print(f"    Stats:       {', '.join(f'{k}={v}' for k, v in sorted(stats.items()))}")
    print(f"    Tiers:       {', '.join(f'{k}={v}' for k, v in sorted(tiers.items()))}")
    print(f"\n  Feature coverage (non-zero/non-null):")
    for field, pct in sorted(coverage.items(), key=lambda x: -x[1]):
        print(f"    {field:35s} {pct:5.1f}%")


def save_backfill_data(records, path=None):
    """Save backfill records to JSON."""
    if path is None:
        path = OUTPUT_PATH

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert numpy types for JSON serialization
    clean_records = []
    for r in records:
        clean = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            elif isinstance(v, np.ndarray):
                clean[k] = [float(x) for x in v]
            elif isinstance(v, list):
                clean[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            elif isinstance(v, bool):
                clean[k] = v
            else:
                clean[k] = v
        clean_records.append(clean)

    with open(path, 'w') as f:
        json.dump(clean_records, f)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n  Saved {len(clean_records):,} records to {path} ({size_mb:.1f} MB)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--stats':
        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH) as f:
                records = json.load(f)
            print(f"  Loaded {len(records):,} records from {OUTPUT_PATH}")
            _print_stats(records)
        else:
            print(f"  No SGO backfill data at {OUTPUT_PATH}")
            print(f"  Run: python3 predictions/backfill_sgo_box_scores.py")
        return

    records = generate_sgo_backfill()
    if not records:
        print("  ERROR: No records generated!")
        sys.exit(1)

    # Pre-sample to manageable size (200K) before saving to avoid multi-GB JSON
    SAVE_CAP = 200000
    if len(records) > SAVE_CAP:
        print(f"\n  Pre-sampling {len(records):,} → {SAVE_CAP:,} records...")
        rng = np.random.RandomState(43)
        weights = np.ones(len(records))
        for i, r in enumerate(records):
            # Recency: 2026 = 3x, 2025 = 2x
            date = r.get('_date', '')
            if date >= '2026-':
                weights[i] *= 3.0
            elif date >= '2025-':
                weights[i] *= 2.0
            # Higher tiers: S/A/B get 2x
            tier = r.get('tier', 'F')
            if tier in ('S', 'A', 'B'):
                weights[i] *= 2.0
            elif tier in ('C', 'D'):
                weights[i] *= 1.3
            # Base stats slightly preferred
            if r.get('stat', '') not in ('pra', 'pr', 'pa', 'ra'):
                weights[i] *= 1.2
            # Prefer records with spread data
            if r.get('spread') is not None:
                weights[i] *= 1.3
        probs = weights / weights.sum()
        indices = rng.choice(len(records), size=SAVE_CAP, replace=False, p=probs)
        records = [records[i] for i in indices]
        print(f"  Sampled to {len(records):,} records")
        _print_stats(records)

    save_backfill_data(records)


if __name__ == '__main__':
    main()

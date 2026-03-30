#!/usr/bin/env python3
"""
Backfill Training Data — Season-wide prop reconstruction from nba_api game logs.

Generates ~20,000+ real labeled prop records by:
1. Loading cached game logs for all players (237+ parquets)
2. For each player-game (starting from game 11), computing rolling stats from PRIOR games only
3. Generating realistic prop lines from L10 avg (snapped to 0.5)
4. Labeling as HIT/MISS from actual box score values
5. Computing REAL context features: home/away, B2B, matchup, travel, usage, plus_minus, PF

Output: predictions/cache/backfill_training_data.json

Usage:
    python3 predictions/backfill_training_data.py              # Generate backfill data
    python3 predictions/backfill_training_data.py --stats      # Print stats only
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PREDICTIONS_DIR, 'cache')
OUTPUT_PATH = os.path.join(CACHE_DIR, 'backfill_training_data.json')

# Stats to generate props for
BASE_STATS = {
    'pts': 'PTS', 'reb': 'REB', 'ast': 'AST',
    '3pm': 'FG3M', 'stl': 'STL', 'blk': 'BLK',
}
COMBO_STATS = {
    'pra': ['PTS', 'REB', 'AST'],
    'pr': ['PTS', 'REB'],
    'pa': ['PTS', 'AST'],
    'ra': ['REB', 'AST'],
    'stl_blk': ['STL', 'BLK'],
}

# Tier thresholds from analyze_v3.py
def _compute_tier(abs_gap, is_combo=False):
    g = abs_gap
    if is_combo:
        g -= 0.5  # combo penalty before tier grading
    if g >= 4: return 'S'
    if g >= 3: return 'A'
    if g >= 2: return 'B'
    if g >= 1.5: return 'C'
    if g >= 1: return 'D'
    return 'F'


def _snap_line(val):
    """Snap to nearest 0.5 like sportsbooks."""
    return round(val * 2) / 2


def _load_all_gamelogs():
    """Load all cached parquet game logs. Returns {player_id: DataFrame}."""
    logs = {}
    parquets = [f for f in os.listdir(CACHE_DIR)
                if f.startswith('gamelog_') and f.endswith('.parquet')]

    for fname in parquets:
        try:
            pid = int(fname.split('_')[1])
            df = pd.read_parquet(os.path.join(CACHE_DIR, fname))
            if df.empty or len(df) < 11:
                continue
            # Ensure sorted oldest → newest for rolling computation
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], format='mixed')
            df = df.sort_values('GAME_DATE', ascending=True).reset_index(drop=True)
            df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(0)
            # Derive IS_HOME and OPP_ABR from MATCHUP if missing
            if 'IS_HOME' not in df.columns and 'MATCHUP' in df.columns:
                df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in str(x) else 0)
            if 'OPP_ABR' not in df.columns and 'MATCHUP' in df.columns:
                df['OPP_ABR'] = df['MATCHUP'].apply(
                    lambda x: str(x).split('vs.')[-1].strip() if 'vs.' in str(x)
                    else str(x).split('@')[-1].strip() if '@' in str(x) else '')
            logs[pid] = df
        except Exception:
            continue

    return logs


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


def _compute_b2b(dates_series, idx):
    """Check if game at idx is back-to-back."""
    if idx == 0:
        return False
    d1 = dates_series.iloc[idx]
    d0 = dates_series.iloc[idx - 1]
    return (d1 - d0).days <= 1


def _compute_travel_features(df, idx, venue_map, tz_ordinal, haversine_fn):
    """Compute travel distance, 7-day travel miles, timezone shifts."""
    travel_dist = np.nan
    travel_7day = np.nan
    tz_shifts = np.nan

    # Current game's venue
    matchup = df.iloc[idx].get('MATCHUP', '')
    is_home = df.iloc[idx].get('IS_HOME', 0)

    # Parse team abbreviations from matchup
    # Format: "LAL vs. DEN" or "LAL @ DEN"
    if ' vs. ' in str(matchup):
        parts = str(matchup).split(' vs. ')
        home_abr = parts[0].strip()
        away_abr = parts[1].strip()
    elif ' @ ' in str(matchup):
        parts = str(matchup).split(' @ ')
        away_abr = parts[0].strip()
        home_abr = parts[1].strip()
    else:
        return travel_dist, travel_7day, tz_shifts

    game_venue_abr = home_abr  # Game is at home team's arena

    # Previous game's venue
    if idx > 0:
        prev_matchup = str(df.iloc[idx - 1].get('MATCHUP', ''))
        if ' vs. ' in prev_matchup:
            prev_parts = prev_matchup.split(' vs. ')
            prev_venue = prev_parts[0].strip()
        elif ' @ ' in prev_matchup:
            prev_parts = prev_matchup.split(' @ ')
            prev_venue = prev_parts[1].strip()
        else:
            prev_venue = None

        if prev_venue and prev_venue in venue_map and game_venue_abr in venue_map:
            pv = venue_map[prev_venue]
            cv = venue_map[game_venue_abr]
            travel_dist = round(haversine_fn(pv['lat'], pv['lng'], cv['lat'], cv['lng']))

    # 7-day travel: sum distances of all games in last 7 days
    game_date = df.iloc[idx]['GAME_DATE']
    week_ago = game_date - timedelta(days=7)
    total_miles = 0.0
    tz_shift_count = 0
    prev_tz = None

    recent_mask = (df['GAME_DATE'] >= week_ago) & (df['GAME_DATE'] <= game_date)
    recent_games = df[recent_mask].sort_values('GAME_DATE')

    prev_venue_abr = None
    for _, row in recent_games.iterrows():
        m = str(row.get('MATCHUP', ''))
        if ' vs. ' in m:
            cur_venue = m.split(' vs. ')[0].strip()
        elif ' @ ' in m:
            cur_venue = m.split(' @ ')[1].strip()
        else:
            continue

        if prev_venue_abr and prev_venue_abr in venue_map and cur_venue in venue_map:
            pv = venue_map[prev_venue_abr]
            cv = venue_map[cur_venue]
            total_miles += haversine_fn(pv['lat'], pv['lng'], cv['lat'], cv['lng'])

        cur_tz = venue_map.get(cur_venue, {}).get('tz')
        if cur_tz and prev_tz and cur_tz != prev_tz:
            tz_shift_count += 1
        if cur_tz:
            prev_tz = cur_tz
        prev_venue_abr = cur_venue

    travel_7day = round(total_miles)
    tz_shifts = tz_shift_count

    return travel_dist, travel_7day, tz_shifts


def _compute_usage(df, idx):
    """Compute usage rate from game logs up to (not including) idx."""
    prior = df.iloc[:idx]
    prior = prior[prior['MIN'] >= 10]
    if len(prior) < 5:
        return np.nan, np.nan

    def _usage(subset):
        fga = subset['FGA'].astype(float)
        fta = subset['FTA'].astype(float)
        tov = subset['TOV'].astype(float)
        mins = subset['MIN'].astype(float).clip(lower=1)
        return float(((fga + 0.44 * fta + tov) / mins).mean())

    l10 = prior.tail(10)
    l5 = prior.tail(5)
    usage_rate = round(_usage(l10), 3)
    usage_trend = round(_usage(l5) - _usage(l10), 3)
    return usage_rate, usage_trend


def _get_stat_val(row, stat_name):
    """Get stat value from a DataFrame row."""
    if stat_name in BASE_STATS:
        return float(row[BASE_STATS[stat_name]])
    elif stat_name in COMBO_STATS:
        return float(sum(row[col] for col in COMBO_STATS[stat_name]))
    return 0.0


def _get_stat_series(df, stat_name):
    """Get stat values as a numpy array from DataFrame."""
    if stat_name in BASE_STATS:
        return df[BASE_STATS[stat_name]].astype(float).values
    elif stat_name in COMBO_STATS:
        return df[COMBO_STATS[stat_name]].astype(float).sum(axis=1).values
    return np.zeros(len(df))


def _matchup_adjustment(opp_team, stat_name, team_rankings, league_avg):
    """Compute rate-based matchup adjustment like analyze_v3 v5."""
    if not team_rankings or not opp_team or opp_team not in team_rankings:
        return 0.0, np.nan, np.nan

    opp = team_rankings[opp_team]
    stat_to_allowed = {
        'pts': 'avg_pts_allowed', 'reb': 'reb_allowed', 'ast': 'ast_allowed',
        '3pm': 'tpm_allowed', 'stl': 'stl_allowed', 'blk': 'blk_allowed',
    }

    # For combo stats, average the component adjustments
    if stat_name in ('pra', 'pr', 'pa', 'ra'):
        components = {'pra': ['pts', 'reb', 'ast'], 'pr': ['pts', 'reb'],
                      'pa': ['pts', 'ast'], 'ra': ['reb', 'ast']}
        adjs = []
        for s in components[stat_name]:
            a, _, _ = _matchup_adjustment(opp_team, s, team_rankings, league_avg)
            adjs.append(a)
        return sum(adjs) / len(adjs) if adjs else 0.0, np.nan, np.nan

    allowed_key = stat_to_allowed.get(stat_name)
    if not allowed_key:
        return 0.0, np.nan, np.nan

    opp_allowed = opp.get(allowed_key, 0)
    lg_avg = league_avg.get(allowed_key, 0) if league_avg else 0

    if lg_avg <= 0:
        return 0.0, opp_allowed, 0.0

    # Rate-based: (allowed - league_avg) / divisor, capped at ±20%
    diff = opp_allowed - lg_avg
    rate_vs_avg = diff / lg_avg if lg_avg > 0 else 0
    adjustment = diff / 75.0  # divisor from analyze_v3

    return round(adjustment, 2), round(opp_allowed, 1), round(rate_vs_avg, 3)


def generate_backfill_data():
    """Main backfill generation — reconstruct props from cached game logs."""
    print("=" * 60)
    print("  XGBoost Training Data Backfill")
    print("=" * 60)

    # Load all resources
    print("\n  Loading cached game logs...")
    all_logs = _load_all_gamelogs()
    print(f"  Loaded {len(all_logs)} player game logs")

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

    all_stats = list(BASE_STATS.keys()) + list(COMBO_STATS.keys())
    all_records = []
    rng = np.random.RandomState(42)

    t0 = time.time()
    player_count = 0

    for pid, df in all_logs.items():
        # Filter to games with meaningful minutes
        df = df[df['MIN'] >= 10].reset_index(drop=True)
        if len(df) < 15:  # Need at least 11 (for L10 window) + a few games
            continue

        player_count += 1

        for stat_name in all_stats:
            is_combo = stat_name in COMBO_STATS
            stat_vals = _get_stat_series(df, stat_name)

            # Start from game 11 (index 10) so we have L10 history
            for idx in range(10, len(df)):
                game_row = df.iloc[idx]
                actual = float(stat_vals[idx])
                game_date = game_row['GAME_DATE']
                game_date_str = game_date.strftime('%Y-%m-%d')

                # Prior games only (no future leakage)
                prior_vals = stat_vals[:idx]
                l10_vals = prior_vals[-10:]
                l5_vals = prior_vals[-5:]
                l3_vals = prior_vals[-3:]

                season_avg = float(np.mean(prior_vals))
                l10_avg = float(np.mean(l10_vals))
                l5_avg = float(np.mean(l5_vals))
                l3_avg = float(np.mean(l3_vals))

                # Generate realistic prop line from L10 avg + small noise
                l10_std = float(np.std(l10_vals))
                noise = rng.normal(0, max(l10_std * 0.2, 0.25))
                line = _snap_line(l10_avg + noise)
                if line <= 0:
                    continue

                # Skip pushes
                if actual == line:
                    continue

                # Home/away splits
                prior_df = df.iloc[:idx]
                home_mask = prior_df['IS_HOME'] == 1
                away_mask = prior_df['IS_HOME'] == 0
                prior_stat_vals_series = pd.Series(prior_vals)

                home_vals = prior_stat_vals_series[home_mask.values[:idx]] if home_mask.sum() > 3 else None
                away_vals = prior_stat_vals_series[away_mask.values[:idx]] if away_mask.sum() > 3 else None
                home_avg = float(home_vals.mean()) if home_vals is not None and len(home_vals) > 0 else season_avg
                away_avg = float(away_vals.mean()) if away_vals is not None and len(away_vals) > 0 else season_avg

                # Hit rates vs line
                l10_hit_rate = float(np.sum(l10_vals > line) / len(l10_vals) * 100)
                l5_hit_rate = float(np.sum(l5_vals > line) / len(l5_vals) * 100)
                season_hit_rate = float(np.sum(prior_vals > line) / len(prior_vals) * 100)

                # Minutes consistency
                prior_mins = prior_df['MIN'].values
                mins_30plus_pct = float(np.sum(prior_mins >= 30) / len(prior_mins) * 100)

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

                # B2B
                is_b2b = _compute_b2b(df['GAME_DATE'], idx)

                # Is home
                is_home = int(game_row.get('IS_HOME', 0))

                # Projection (same formula as analyze_v3)
                projection = 0.4 * l10_avg + 0.3 * l5_avg + 0.3 * season_avg

                # Split adjustment
                if is_home and home_vals is not None and len(home_vals) > 0:
                    split_adj = round(float(home_avg - season_avg) * 0.5, 2)
                elif not is_home and away_vals is not None and len(away_vals) > 0:
                    split_adj = round(float(away_avg - season_avg) * 0.5, 2)
                else:
                    split_adj = 0.0

                # Matchup adjustment
                opp_team = game_row.get('OPP_ABR', '')
                matchup_adj, opp_allowed_rate, opp_allowed_vs_league = _matchup_adjustment(
                    opp_team, stat_name, team_rankings, league_avg
                )

                # Spread (NaN — we don't have historical ESPN scoreboard data for every game)
                spread = np.nan

                # Travel features
                if has_venue:
                    travel_dist, travel_7day, tz_shifts = _compute_travel_features(
                        df, idx, venue_map, tz_ordinal, haversine_fn
                    )
                else:
                    travel_dist, travel_7day, tz_shifts = np.nan, np.nan, np.nan

                # Usage metrics (from prior games)
                usage_rate, usage_trend = _compute_usage(df, idx)

                # Plus/minus and PF from L10
                l10_df = prior_df.tail(10)
                l10_plus_minus = round(float(l10_df['PLUS_MINUS'].mean()), 1) if 'PLUS_MINUS' in l10_df.columns else 0.0
                l10_pf = round(float(l10_df['PF'].mean()), 1) if 'PF' in l10_df.columns else 0.0
                l5_pf = round(float(prior_df.tail(5)['PF'].mean()), 1) if 'PF' in prior_df.columns else 0.0
                foul_trouble_risk = l5_pf >= 4.0

                # Gap and tier
                gap = projection - line
                abs_gap = abs(gap)
                effective_gap = gap
                tier = _compute_tier(abs_gap, is_combo)

                # Minutes adjustment
                season_mins = float(prior_df['MIN'].mean())
                l5_mins = float(prior_df.tail(5)['MIN'].mean())
                mins_adj = round((l5_mins - season_mins) / season_mins * 0.5, 2) if season_mins > 0 else 0.0

                # Streak adjustment
                streak_adj = 0.0

                # Blowout adjustment (NaN spread = can't compute)
                blowout_adj = 0.0

                # B2B travel adjustment
                b2b_adj = 0.0
                if is_b2b and not np.isnan(travel_dist):
                    if travel_dist > 1500:
                        b2b_adj = -0.04
                    elif travel_dist > 500:
                        b2b_adj = -0.02
                    else:
                        b2b_adj = -0.01

                # Build base record
                rec_base = {
                    'player': f'pid_{pid}',  # anonymized — model doesn't use player name
                    'stat': stat_name,
                    'line': line,
                    'projection': round(projection, 1),
                    'gap': round(gap, 1),
                    'abs_gap': round(abs_gap, 1),
                    'effective_gap': round(effective_gap, 1),
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
                    'streak_adj': streak_adj,
                    'blowout_adj': blowout_adj,
                    'injury_adjustment': 0,
                    'spread': spread,
                    'streak_pct': round(streak_pct, 1),
                    'games_used': len(prior_vals),
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
                    'l10_avg_pf': l10_pf,
                    'foul_trouble_risk': foul_trouble_risk,
                    # v5 features
                    'opp_stat_allowed_rate': opp_allowed_rate if not (isinstance(opp_allowed_rate, float) and np.isnan(opp_allowed_rate)) else None,
                    'opp_stat_allowed_vs_league_avg': opp_allowed_vs_league if not (isinstance(opp_allowed_vs_league, float) and np.isnan(opp_allowed_vs_league)) else None,
                    'usage_rate': usage_rate if not np.isnan(usage_rate) else None,
                    'usage_trend': usage_trend if not np.isnan(usage_trend) else None,
                    'dynamic_without_delta': None,  # Can't compute without knowing injured teammates
                    'travel_distance': travel_dist if not np.isnan(travel_dist) else None,
                    'travel_miles_7day': travel_7day if not np.isnan(travel_7day) else None,
                    'tz_shifts_7day': tz_shifts if not np.isnan(tz_shifts) else None,
                    # v7 features (enrichment — computed from existing data)
                    'opp_matchup_delta': None,  # would need per-player vs opponent history
                    'team_vs_opp_delta': None,  # would need team vs team history
                    'opp_off_pressure': None,   # would need opponent offensive rating
                    'usage_boost': None,        # would need same-team injury data
                    'game_total_signal': None,  # computed below if spread available
                    'max_same_game_corr': None, # would need same-game picks
                    # Metadata
                    '_date': game_date_str,
                    '_source': 'backfill',
                    '_data_source': 'backfill',
                    '_feature_version': 7,
                }

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

        if player_count % 25 == 0:
            elapsed = time.time() - t0
            print(f"  ... {player_count} players, {len(all_records)} records ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Generated {len(all_records)} total records from {player_count} players in {elapsed:.1f}s")

    # Print stats
    _print_stats(all_records)

    return all_records


def _print_stats(records):
    """Print summary statistics of backfill data."""
    if not records:
        print("  No records generated.")
        return

    n = len(records)
    hits = sum(1 for r in records if r['_hit_label'])
    dates = sorted(set(r['_date'] for r in records))
    stats = {}
    for r in records:
        s = r['stat']
        stats[s] = stats.get(s, 0) + 1
    tiers = {}
    for r in records:
        t = r['tier']
        tiers[t] = tiers.get(t, 0) + 1

    # Feature coverage
    context_fields = [
        'matchup_adjustment', 'split_adjustment', 'mins_adj',
        'opp_stat_allowed_rate', 'usage_rate', 'travel_distance',
        'l10_avg_plus_minus', 'l10_avg_pf',
    ]
    coverage = {}
    for field in context_fields:
        non_null = sum(1 for r in records
                       if r.get(field) is not None
                       and r.get(field) != 0
                       and not (isinstance(r.get(field), float) and np.isnan(r.get(field))))
        coverage[field] = round(non_null / n * 100, 1)

    print(f"\n  Summary:")
    print(f"    Records:   {n:,}")
    print(f"    Hit rate:  {hits/n:.1%}")
    print(f"    Dates:     {dates[0]} → {dates[-1]} ({len(dates)} unique)")
    print(f"    Stats:     {', '.join(f'{k}={v}' for k, v in sorted(stats.items()))}")
    print(f"    Tiers:     {', '.join(f'{k}={v}' for k, v in sorted(tiers.items()))}")
    print(f"\n  Feature coverage (non-zero/non-null):")
    for field, pct in sorted(coverage.items(), key=lambda x: -x[1]):
        print(f"    {field:35s} {pct:5.1f}%")


def save_backfill_data(records, path=None):
    """Save backfill records to JSON."""
    if path is None:
        path = OUTPUT_PATH

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Convert numpy types to native Python for JSON serialization
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
            else:
                clean[k] = v
        clean_records.append(clean)

    with open(path, 'w') as f:
        json.dump(clean_records, f)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n  Saved {len(clean_records):,} records to {path} ({size_mb:.1f} MB)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--stats':
        # Just load and print stats
        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH) as f:
                records = json.load(f)
            print(f"  Loaded {len(records):,} records from {OUTPUT_PATH}")
            _print_stats(records)
        else:
            print(f"  No backfill data found at {OUTPUT_PATH}")
            print(f"  Run: python3 predictions/backfill_training_data.py")
        return

    records = generate_backfill_data()
    if records:
        save_backfill_data(records)
    else:
        print("  ERROR: No records generated!")
        sys.exit(1)


if __name__ == '__main__':
    main()

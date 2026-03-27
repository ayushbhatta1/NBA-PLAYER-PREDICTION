#!/usr/bin/env python3
"""
Pre-compute ALL training features into a single cached .npz file.

Processes every data source we have (1946-2026), engineers features once,
saves as numpy arrays. All models then load instantly — no JSON parsing,
no feature engineering loop, no memory spikes.

Sources:
  1. Full CSV (1946-2026): 1.66M player-games → reconstruct props
  2. nba_api backfill (2025-26): 242K props
  3. SGO backfill (2024-26): 200K props with real sportsbook data
  4. Graded predictions (Mar 2026): 9K real predictions with outcomes

Output: predictions/cache/all_features.npz (~500MB)
  - X: feature matrix (N x 77)
  - y: labels (N,)
  - dates: date strings (N,)
  - sources: source tags (N,)
  - feature_names: column names (77,)

Usage:
    python3 predictions/precompute_features.py          # Build cache
    python3 predictions/precompute_features.py --info    # Show cache info
"""

import json
import os
import sys
import gc
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRED_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(PRED_DIR, 'cache', 'all_features.npz')
CSV_PATH = os.path.join(PRED_DIR, '..', 'NBA Database (1947 - Present)', 'PlayerStatistics.csv')

# Stat types we can reconstruct from box scores
STAT_MAP = {
    'pts': 'points',
    'reb': 'reboundsTotal',
    'ast': 'assists',
    '3pm': 'threePointersMade',
    'stl': 'steals',
    'blk': 'blocks',
}

COMBO_MAP = {
    'pra': ['points', 'reboundsTotal', 'assists'],
    'pr': ['points', 'reboundsTotal'],
    'pa': ['points', 'assists'],
    'ra': ['reboundsTotal', 'assists'],
}


def reconstruct_props_from_csv(chunk_size=50000):
    """
    Stream the full CSV and reconstruct prop-like records.
    Processes in chunks to limit memory. Uses rolling stats
    from prior games only (no data leakage).
    """
    print(f"  Loading CSV: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"  ERROR: CSV not found at {CSV_PATH}")
        return []

    # Read CSV in chunks
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"  Full CSV: {len(df):,} rows")

    # Clean numeric columns
    # Normalize column names (CSV uses different names than nba_api)
    col_renames = {
        'numMinutes': 'minutes',
        'foulsPersonal': 'personalFouls',
        'plusMinusPoints': 'plusMinus',
    }
    df = df.rename(columns=col_renames)

    for col in ['points', 'reboundsTotal', 'assists', 'threePointersMade',
                'steals', 'blocks', 'minutes', 'plusMinus', 'personalFouls']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Parse dates
    df['date'] = pd.to_datetime(df['gameDateTimeEst'], errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['date'])
    df = df.sort_values(['firstName', 'lastName', 'date']).reset_index(drop=True)

    # Group by player
    df['player_key'] = df['firstName'].astype(str) + ' ' + df['lastName'].astype(str)

    all_records = []
    players = df.groupby('player_key')
    total_players = len(players)
    processed = 0

    for player_name, player_df in players:
        processed += 1
        if processed % 500 == 0:
            print(f"    Processing player {processed:,}/{total_players:,} ({len(all_records):,} props so far)")

        player_df = player_df.sort_values('date').reset_index(drop=True)

        if len(player_df) < 11:  # need 10 prior games for rolling stats
            continue

        for i in range(10, len(player_df)):
            row = player_df.iloc[i]
            prior = player_df.iloc[max(0, i-10):i]
            prior_5 = player_df.iloc[max(0, i-5):i]
            prior_season = player_df.iloc[:i]
            date = row['date']

            # Skip very old data where stats are unreliable
            if date < '1980-01-01':
                continue

            # For each stat type, create a prop record
            for stat, csv_col in STAT_MAP.items():
                actual_val = float(row.get(csv_col, 0))
                l10_vals = [float(x) for x in prior[csv_col].values]
                l5_vals = [float(x) for x in prior_5[csv_col].values]
                season_vals = [float(x) for x in prior_season[csv_col].values[-60:]]  # last 60 games

                if len(l10_vals) < 5:
                    continue

                l10_avg = np.mean(l10_vals)
                l5_avg = np.mean(l5_vals) if l5_vals else l10_avg
                season_avg = np.mean(season_vals) if season_vals else l10_avg
                l10_std = np.std(l10_vals) if len(l10_vals) >= 3 else 0

                # Simulate a sportsbook line (median of L10 ± small noise)
                line = round(np.median(l10_vals) * 2) / 2  # round to nearest 0.5
                if line <= 0:
                    line = 0.5

                gap = l10_avg - line
                direction = 'OVER' if gap > 0 else 'UNDER'
                hit = (actual_val > line) if direction == 'OVER' else (actual_val < line)

                # Hit rates
                l10_hr = sum(1 for v in l10_vals if (v > line if direction == 'OVER' else v < line)) / len(l10_vals) * 100
                l5_hr = sum(1 for v in l5_vals if (v > line if direction == 'OVER' else v < line)) / len(l5_vals) * 100 if l5_vals else l10_hr
                season_hr = sum(1 for v in season_vals if (v > line if direction == 'OVER' else v < line)) / len(season_vals) * 100 if season_vals else l10_hr

                # Minutes
                mins_vals = [float(x) for x in prior['minutes'].values if x > 0]
                avg_mins = np.mean(mins_vals) if mins_vals else 0
                mins_30plus = sum(1 for m in mins_vals if m >= 30) / len(mins_vals) * 100 if mins_vals else 0

                # Plus/minus and PF
                pm_vals = [float(x) for x in prior['plusMinus'].values]
                pf_vals = [float(x) for x in prior['personalFouls'].values]

                # B2B detection
                if i > 0:
                    prev_date = player_df.iloc[i-1]['date']
                    try:
                        d1 = datetime.strptime(date, '%Y-%m-%d')
                        d2 = datetime.strptime(prev_date, '%Y-%m-%d')
                        is_b2b = (d1 - d2).days <= 1
                    except:
                        is_b2b = False
                else:
                    is_b2b = False

                # Miss streak
                miss_streak = 0
                for v in reversed(l10_vals):
                    if (direction == 'OVER' and v <= line) or (direction == 'UNDER' and v >= line):
                        miss_streak += 1
                    else:
                        break

                # Streak status
                if miss_streak >= 3:
                    streak_status = 'COLD'
                elif miss_streak == 0 and l5_hr >= 80:
                    streak_status = 'HOT'
                else:
                    streak_status = 'NEUTRAL'

                record = {
                    'player': player_name,
                    'stat': stat,
                    'line': line,
                    'projection': round(l10_avg * 0.7 + season_avg * 0.3, 1),
                    'gap': round(gap, 2),
                    'abs_gap': round(abs(gap), 2),
                    'effective_gap': round(abs(gap), 2),
                    'direction': direction,
                    'tier': 'C',  # default — not computed for historical
                    'season_avg': round(season_avg, 1),
                    'l10_avg': round(l10_avg, 1),
                    'l5_avg': round(l5_avg, 1),
                    'l3_avg': round(np.mean(l10_vals[-3:]), 1) if len(l10_vals) >= 3 else l10_avg,
                    'home_avg': round(season_avg, 1),  # approximate
                    'away_avg': round(season_avg, 1),
                    'l10_hit_rate': round(l10_hr, 1),
                    'l5_hit_rate': round(l5_hr, 1),
                    'season_hit_rate': round(season_hr, 1),
                    'mins_30plus_pct': round(mins_30plus, 1),
                    'split_adjustment': 0,
                    'matchup_adjustment': 0,
                    'mins_adj': 0,
                    'streak_adj': 0,
                    'blowout_adj': 0,
                    'injury_adjustment': 0,
                    'spread': None,
                    'streak_pct': 0,
                    'games_used': len(prior_season),
                    'l10_floor': min(l10_vals),
                    'l10_miss_count': miss_streak,
                    'l10_values': l10_vals,
                    'same_team_out_count': 0,
                    'is_b2b': is_b2b,
                    'is_home': None,
                    'streak_status': streak_status,
                    'l10_avg_plus_minus': round(np.mean(pm_vals), 1) if pm_vals else 0,
                    'l10_avg_pf': round(np.mean(pf_vals), 1) if pf_vals else 0,
                    'foul_trouble_risk': np.mean(pf_vals) >= 4.0 if pf_vals else False,
                    'usage_rate': 0,
                    'usage_trend': 0,
                    'game_total_signal': 0,
                    'travel_distance': 0,
                    'travel_miles_7day': 0,
                    'tz_shifts_7day': 0,
                    'opponent_history': None,
                    '_hit_label': hit,
                    '_date': date,
                    '_data_source': 'csv_full',
                }
                all_records.append(record)

        # Periodic memory cleanup
        if processed % 200 == 0:
            gc.collect()

    print(f"  CSV reconstruction: {len(all_records):,} props from {processed:,} players")
    return all_records


def load_json_source(path, source_name, max_records=None):
    """Load a JSON training data file."""
    if not os.path.exists(path):
        print(f"  {source_name}: not found at {path}")
        return []

    print(f"  Loading {source_name}...")
    with open(path) as f:
        records = json.load(f)

    for r in records:
        r['_data_source'] = source_name
        if '_hit_label' not in r:
            if 'hit' in r:
                r['_hit_label'] = bool(r['hit'])
            elif 'result' in r:
                r['_hit_label'] = r.get('result', '').upper() == 'HIT'

    valid = [r for r in records if r.get('_hit_label') is not None]

    if max_records and len(valid) > max_records:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(valid), size=max_records, replace=False)
        valid = [valid[i] for i in sorted(indices)]

    print(f"  {source_name}: {len(valid):,} records")
    del records
    gc.collect()
    return valid


def load_graded():
    """Load all graded prediction files."""
    from xgb_model import collect_training_data
    records = collect_training_data()
    for r in records:
        r['_data_source'] = 'graded'
    print(f"  Graded: {len(records):,} records")
    return records


def build_cache():
    """Build the full feature cache from ALL data sources."""
    from xgb_model import engineer_features, FEATURE_COLS

    print("=" * 70)
    print("  PRE-COMPUTING ALL FEATURES")
    print("=" * 70)

    # ── Source 1: Graded predictions (highest quality) ──
    graded = load_graded()

    # ── Source 2: nba_api backfill ──
    backfill_path = os.path.join(PRED_DIR, 'cache', 'backfill_training_data.json')
    backfill = load_json_source(backfill_path, 'backfill')

    # ── Source 3: SGO backfill ──
    sgo_path = os.path.join(PRED_DIR, 'cache', 'sgo_backfill_training_data.json')
    sgo = load_json_source(sgo_path, 'sgo_backfill')

    # ── Source 4: Full CSV reconstruction (1980-2026) ──
    # This is the BIG one — processes the entire PlayerStatistics.csv
    csv_records = reconstruct_props_from_csv()

    # Combine all
    all_records = graded + backfill + sgo + csv_records
    print(f"\n  TOTAL: {len(all_records):,} records from all sources")
    print(f"    Graded:     {len(graded):,}")
    print(f"    Backfill:   {len(backfill):,}")
    print(f"    SGO:        {len(sgo):,}")
    print(f"    CSV (full): {len(csv_records):,}")

    # Free source lists
    del graded, backfill, sgo, csv_records
    gc.collect()

    # ── Engineer features ──
    print(f"\n  Engineering {len(all_records):,} records into {len(FEATURE_COLS)} features...")

    # Process in chunks to manage memory
    CHUNK = 200000
    X_chunks = []
    y_chunks = []
    date_chunks = []
    source_chunks = []

    for start in range(0, len(all_records), CHUNK):
        end = min(start + CHUNK, len(all_records))
        chunk = all_records[start:end]
        print(f"    Chunk {start:,}-{end:,} ({len(chunk):,} records)...")

        X_c, y_c, dates_c = engineer_features(chunk)
        sources_c = [r.get('_data_source', 'unknown') for r in chunk]

        X_chunks.append(X_c)
        y_chunks.append(y_c)
        date_chunks.extend(dates_c)
        source_chunks.extend(sources_c)

        del chunk
        gc.collect()

    # Stack all chunks
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    dates = np.array(date_chunks, dtype='U10')
    sources = np.array(source_chunks, dtype='U20')

    del X_chunks, y_chunks, date_chunks, source_chunks, all_records
    gc.collect()

    print(f"\n  Final matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"  Hit rate: {y.mean():.1%} ({y.sum():,}/{len(y):,})")
    print(f"  NaN rate: {np.isnan(X).mean():.1%}")

    # Source breakdown
    for src in sorted(set(sources)):
        mask = sources == src
        print(f"    {src:15s}: {mask.sum():>10,} records, HR={y[mask].mean():.1%}")

    # ── Save ──
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        X=X, y=y, dates=dates, sources=sources,
        feature_names=np.array(FEATURE_COLS, dtype='U40'),
    )
    file_size = os.path.getsize(CACHE_PATH) / 1e6
    print(f"\n  Saved: {CACHE_PATH} ({file_size:.0f} MB)")
    print(f"  Features: {len(FEATURE_COLS)}")

    return X, y, dates, sources


def load_cache():
    """Load pre-computed features."""
    if not os.path.exists(CACHE_PATH):
        print(f"No cache at {CACHE_PATH}. Run: python3 precompute_features.py")
        return None, None, None, None

    data = np.load(CACHE_PATH, allow_pickle=True)
    return data['X'], data['y'], data['dates'], data['sources']


def show_info():
    """Show cache info."""
    if not os.path.exists(CACHE_PATH):
        print(f"No cache found. Run: python3 predictions/precompute_features.py")
        return

    data = np.load(CACHE_PATH, allow_pickle=True)
    X = data['X']
    y = data['y']
    dates = data['dates']
    sources = data['sources']
    features = data['feature_names']

    print(f"{'=' * 60}")
    print(f"  ALL FEATURES CACHE")
    print(f"{'=' * 60}")
    print(f"  File: {CACHE_PATH}")
    print(f"  Size: {os.path.getsize(CACHE_PATH) / 1e6:.0f} MB")
    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Hit rate: {y.mean():.1%}")
    print(f"  Date range: {dates.min()} to {dates.max()}")
    print(f"  NaN rate: {np.isnan(X).mean():.1%}")
    print(f"\n  By source:")
    for src in sorted(set(sources)):
        mask = sources == src
        print(f"    {src:15s}: {mask.sum():>10,}  HR={y[mask].mean():.1%}")


if __name__ == '__main__':
    if '--info' in sys.argv:
        show_info()
    else:
        build_cache()

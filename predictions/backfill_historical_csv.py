#!/usr/bin/env python3
"""
Reconstruct massive training dataset from 10 years of CSV game logs.
Same logic as backfill_training_data.py but uses the historical CSV database.
Generates labeled prop records with rolling stats — no API calls needed.
"""
import pandas as pd
import numpy as np
import json
import os
import time
from collections import defaultdict

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'NBA Database (1947 - Present)', 'PlayerStatistics.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'cache', 'historical_10yr_training.json')

STAT_COLS = {
    'pts': 'points',
    'reb': 'reboundsTotal',
    'ast': 'assists',
    '3pm': 'threePointersMade',
    'stl': 'steals',
    'blk': 'blocks',
}

COMBO_STATS = {
    'pra': ['pts', 'reb', 'ast'],
    'pr': ['pts', 'reb'],
    'pa': ['pts', 'ast'],
    'ra': ['reb', 'ast'],
    'stl_blk': ['stl', 'blk'],
}

MIN_GAMES = 11  # Need 10 prior games for rolling stats


def load_data(start_year=2016):
    """Load and prep the CSV data."""
    print(f"Loading CSV from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'], errors='coerce')
    df = df.dropna(subset=['gameDateTimeEst'])
    df = df[df['gameDateTimeEst'].dt.year >= start_year].copy()

    # Only regular season (gameType might indicate this)
    # Build player name
    df['player'] = df['firstName'].astype(str) + ' ' + df['lastName'].astype(str)

    # Convert stats to numeric
    for stat_name, col in STAT_COLS.items():
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df['numMinutes'] = pd.to_numeric(df['numMinutes'], errors='coerce').fillna(0)
    df['plusMinusPoints'] = pd.to_numeric(df['plusMinusPoints'], errors='coerce').fillna(0)
    df['foulsPersonal'] = pd.to_numeric(df['foulsPersonal'], errors='coerce').fillna(0)
    df['home'] = df['home'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)

    # Compute combo stats
    df['pts'] = df['points']
    df['reb'] = df['reboundsTotal']
    df['ast'] = df['assists']
    df['3pm_val'] = df['threePointersMade']
    df['stl_val'] = df['steals']
    df['blk_val'] = df['blocks']
    df['pra'] = df['pts'] + df['reb'] + df['ast']
    df['pr'] = df['pts'] + df['reb']
    df['pa'] = df['pts'] + df['ast']
    df['ra'] = df['reb'] + df['ast']
    df['stl_blk'] = df['stl_val'] + df['blk_val']

    # Sort by player and date
    df = df.sort_values(['player', 'gameDateTimeEst']).reset_index(drop=True)

    print(f"  Loaded {len(df):,} game logs for {df['player'].nunique():,} players ({start_year}-present)")
    return df


def build_training_records(df, stats_to_process=None):
    """Build prop records with rolling stats from prior games (no data leakage)."""
    if stats_to_process is None:
        stats_to_process = ['pts', 'reb', 'ast', '3pm_val', 'stl_val', 'blk_val', 'pra', 'pr', 'pa', 'ra', 'stl_blk']

    stat_key_map = {
        'pts': 'pts', 'reb': 'reb', 'ast': 'ast', '3pm_val': '3pm',
        'stl_val': 'stl', 'blk_val': 'blk',
        'pra': 'pra', 'pr': 'pr', 'pa': 'pa', 'ra': 'ra', 'stl_blk': 'stl_blk',
    }

    records = []
    players = df.groupby('player')
    total_players = len(players)
    processed = 0
    start_time = time.time()

    for player_name, player_df in players:
        processed += 1
        if processed % 200 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (total_players - processed) / rate
            print(f"  [{processed}/{total_players}] {rate:.0f} players/sec | ETA {eta:.0f}s | {len(records):,} records")

        games = player_df.reset_index(drop=True)
        if len(games) < MIN_GAMES:
            continue

        for i in range(10, len(games)):  # Start from game 11 (need 10 prior)
            current = games.iloc[i]
            prior = games.iloc[max(0, i-82):i]  # Up to ~1 season of prior games
            prior_l10 = games.iloc[max(0, i-10):i]
            prior_l5 = games.iloc[max(0, i-5):i]
            prior_l3 = games.iloc[max(0, i-3):i]

            mins = current['numMinutes']
            if mins < 5:  # Skip DNP/garbage time
                continue

            game_date = current['gameDateTimeEst'].strftime('%Y-%m-%d')
            is_home = int(current['home'])

            # Home/away splits from prior games
            prior_home = prior[prior['home'] == 1]
            prior_away = prior[prior['home'] == 0]

            for stat_col in stats_to_process:
                stat_key = stat_key_map[stat_col]
                actual = float(current[stat_col])

                season_avg = float(prior[stat_col].mean())
                l10_avg = float(prior_l10[stat_col].mean())
                l5_avg = float(prior_l5[stat_col].mean())
                l3_avg = float(prior_l3[stat_col].mean())

                home_avg = float(prior_home[stat_col].mean()) if len(prior_home) >= 3 else season_avg
                away_avg = float(prior_away[stat_col].mean()) if len(prior_away) >= 3 else season_avg

                # Simulate a prop line: L10 average (how books roughly set lines)
                line = round(l10_avg * 2) / 2  # Round to nearest 0.5
                if line == actual:
                    line = round(l10_avg * 2 + 1) / 2  # Avoid push

                gap = season_avg - line

                # L10 hit rates
                l10_values = prior_l10[stat_col].values
                l10_over_hr = int(np.mean(l10_values > line) * 100)
                l10_under_hr = int(np.mean(l10_values < line) * 100)

                # L5 hit rate
                l5_values = prior_l5[stat_col].values
                l5_over_hr = int(np.mean(l5_values > line) * 100)

                # Minutes
                mins_30plus = int(np.mean(prior_l10['numMinutes'].values >= 30) * 100)

                # Determine direction and hit
                # OVER if projection > line
                projection = 0.4 * season_avg + 0.35 * l10_avg + 0.25 * l5_avg
                direction = 'OVER' if projection > line else 'UNDER'
                hit = (actual > line) if direction == 'OVER' else (actual < line)

                # Floor stats
                l10_floor = float(prior_l10[stat_col].min())
                l10_miss_count = int(np.sum(l10_values <= line)) if direction == 'OVER' else int(np.sum(l10_values >= line))

                records.append({
                    'date': game_date,
                    'player': player_name,
                    'stat': stat_key,
                    'line': line,
                    'actual': actual,
                    'direction': direction,
                    'hit': hit,
                    'projection': round(projection, 1),
                    'season_avg': round(season_avg, 1),
                    'l10_avg': round(l10_avg, 1),
                    'l5_avg': round(l5_avg, 1),
                    'l3_avg': round(l3_avg, 1),
                    'home_avg': round(home_avg, 1),
                    'away_avg': round(away_avg, 1),
                    'gap': round(gap, 1),
                    'abs_gap': round(abs(gap), 1),
                    'effective_gap': round(gap, 1),
                    'l10_hit_rate': l10_over_hr if direction == 'OVER' else l10_under_hr,
                    'l5_hit_rate': l5_over_hr if direction == 'OVER' else (100 - l5_over_hr),
                    'season_hit_rate': l10_over_hr,  # approximate
                    'mins_30plus_pct': mins_30plus,
                    'l10_floor': round(l10_floor, 1),
                    'l10_miss_count': l10_miss_count,
                    'is_home': is_home,
                    'is_b2b': 0,  # can't easily compute from CSV
                    'spread': 0,
                    'plus_minus': float(current['plusMinusPoints']),
                    'pf': float(current['foulsPersonal']),
                    'minutes': float(mins),
                    '_data_source': 'historical_csv',
                })

    return records


def main():
    start = time.time()
    df = load_data(start_year=2016)  # 10 years

    print(f"\nBuilding training records (11 stat types × ~{len(df):,} games)...")
    records = build_training_records(df)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  HISTORICAL BACKFILL COMPLETE")
    print(f"{'='*60}")
    print(f"  Records: {len(records):,}")
    print(f"  Players: {len(set(r['player'] for r in records)):,}")
    print(f"  Date range: {min(r['date'] for r in records)} to {max(r['date'] for r in records)}")
    print(f"  Hit rate: {100*sum(r['hit'] for r in records)/len(records):.1f}%")
    print(f"  Time: {elapsed:.0f}s")

    # Stats breakdown
    from collections import Counter
    stats = Counter(r['stat'] for r in records)
    print(f"  By stat: {dict(stats)}")

    # Direction breakdown
    over = [r for r in records if r['direction'] == 'OVER']
    under = [r for r in records if r['direction'] == 'UNDER']
    print(f"  OVER: {sum(r['hit'] for r in over)}/{len(over)} ({100*sum(r['hit'] for r in over)/len(over):.1f}%)")
    print(f"  UNDER: {sum(r['hit'] for r in under)}/{len(under)} ({100*sum(r['hit'] for r in under)/len(under):.1f}%)")

    print(f"\n  Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(records, f)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024*1024)
    print(f"  Saved: {size_mb:.1f} MB")
    print(f"  Done!")


if __name__ == '__main__':
    main()

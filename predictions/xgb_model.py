#!/usr/bin/env python3
"""
XGBoost Prop Classifier — ML Layer for NEXUS Pipeline

Trains a gradient-boosted model on graded prop data to produce xgb_prob (0-1)
for each prop line. Supplements (not replaces) the rule-based nexus_score.

Usage:
    python3 predictions/xgb_model.py train                # Train on graded + historical (default)
    python3 predictions/xgb_model.py train --no-historical # Train on graded data only
    python3 predictions/xgb_model.py eval                 # Walk-forward CV (graded + historical)
    python3 predictions/xgb_model.py eval --no-historical  # Walk-forward CV (graded only)
    python3 predictions/xgb_model.py score <file>         # Score a board/results file
    python3 predictions/xgb_model.py importance           # Feature importance report
    python3 predictions/xgb_model.py historical-stats     # Print historical data summary
"""

import json
import os
import sys
import glob
import csv
import warnings
from datetime import datetime, timedelta

import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: xgboost not installed. Run: pip3 install xgboost")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PREDICTIONS_DIR, 'xgb_model.json')
META_PATH = os.path.join(PREDICTIONS_DIR, 'xgb_model_meta.json')

COMBO_STATS = {'pra', 'pr', 'pa', 'ra'}

# Stat type ordinal encoding by historical hit rate (higher = better)
STAT_ORDINAL = {
    'blk': 6, 'stl': 5, 'ast': 4, 'reb': 3, '3pm': 2, 'pts': 1,
    'pra': 0, 'pr': 0, 'pa': 0, 'ra': 0,
}

TIER_ORDINAL = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}

STREAK_ORDINAL = {'HOT': 1, 'NEUTRAL': 0, 'COLD': -1}

# All feature columns in order (model expects this exact order)
FEATURE_COLS = [
    # Direct from analyze_v3 output
    'line', 'projection', 'gap', 'abs_gap', 'effective_gap',
    'season_avg', 'l10_avg', 'l5_avg', 'l3_avg',
    'home_avg', 'away_avg',
    'l10_hit_rate', 'l5_hit_rate', 'season_hit_rate',
    'mins_30plus_pct',
    'split_adjustment', 'matchup_adjustment', 'mins_adj',
    'streak_adj', 'blowout_adj', 'injury_adjustment',
    'spread', 'streak_pct', 'games_used',
    # Engineered
    'gap_over_line', 'season_vs_line', 'l5_vs_l10',
    'l5_l10_hr_delta', 'margin_over_line', 'projection_confidence',
    'total_adjustment', 'gap_x_hr', 'mins_x_gap',
    # Floor/consistency signals
    'l10_floor', 'l10_miss_count', 'floor_gap', 'l10_std',
    # Opponent context
    'opp_hit_rate', 'opp_avg_vs_line',
    # Usage/context
    'same_team_out_count', 'miss_streak',
    # Categorical (encoded)
    'stat_ordinal', 'direction_binary', 'is_home_binary',
    'is_b2b_binary', 'is_combo', 'streak_ordinal',
    # v2 features
    'gap_squared', 'is_pts', 'is_blk_stl', 'is_ast',
    'l5_l10_agreement', 'hr_confidence', 'line_relative_avg',
    # v4 features (scout data + screen data) — kept: plus_minus, PF, foul, screen_tier
    'l10_avg_plus_minus', 'l10_avg_pf', 'foul_trouble_risk_binary',
    'screen_tier_ordinal',
    # v5 features (rate-based defense, travel — kept populated ones)
    'travel_distance_last_game', 'travel_miles_7day', 'timezone_shifts_7day',
    # v6 features (interaction terms from cross-day analysis)
    'under_x_cold',           # UNDER + COLD streak = 75.3%
    'under_x_blkstl',         # UNDER + BLK/STL = 73.3%
    'hot_x_over',             # HOT + OVER = 49.2% trap
    'under_x_gap',            # UNDER * abs_gap (73.9% when gap>=2)
    'under_x_hr',             # UNDER * l10_hr / 100
    'combo_x_over',           # combo + OVER = volatile
    'directional_gap',        # gap if OVER, -gap if UNDER (favorable = positive)
    'hr_x_gap',               # l10_hr * abs_gap (high conviction)
    'cold_under_x_gap',       # COLD+UNDER+gap triple interaction
    'streak_x_direction',     # streak_ordinal * direction_binary
    # v7 features — kept: game_total_signal (66% populated, was #2 feature)
    'game_total_signal',          # game total vs 225 league avg
    'game_total_x_direction',     # high total + OVER alignment interaction
    # v10 features — kept: EWMA + median (computed from l10_values, always available)
    'l10_ewma',              # Exponentially Weighted Moving Average (alpha=0.15)
    'l10_median',            # L10 median (books price at median, not mean)
    'mean_median_gap',       # (l10_avg - l10_median) / line — skewness signal
    'l10_cv',                # Coefficient of variation: l10_std / l10_avg
]

# ── DEAD FEATURES (removed in v15 cleanup, 2026-03-27) ──
# 100% NaN (never populated): v11 advanced CSV (24), SGO book data (3),
#   l20_avg, l20_hit_rate, production_rate, usage_boost
# 99.7% NaN: nn_emb_0..nn_emb_31 (32 — only in live inference, never in training)
# 98-99% NaN: ref/coach (9), v7 enrichment (5 of 8), eval_agreement (2),
#   venue_altitude, travel_zone_diff
# 68-97% NaN: opp_stat_allowed (2), usage_rate/trend (2), dynamic_without,
#   sim_prob/mean/std (3), opp_hit_rate, opp_avg_vs_line
# Total removed: 91 features. Remaining: 75 clean features.


# ═══════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════════════

def _extract_hit_label(record):
    """Extract hit/miss label from a graded record. Returns True/False/None."""
    # Try direct 'hit' field (Mar 11 format)
    if 'hit' in record and record['hit'] is not None:
        return bool(record['hit'])

    # Try 'result' field (Mar 12/13 format: 'HIT' or 'MISS')
    result = record.get('result', '')
    if isinstance(result, str):
        if result.upper() == 'HIT':
            return True
        elif result.upper() == 'MISS':
            return False

    # Try computing from actual + line + direction
    actual = record.get('actual')
    line = record.get('line')
    direction = record.get('direction', '')
    if actual is not None and line is not None and direction:
        if direction == 'OVER':
            return actual > line
        else:
            return actual < line

    return None


def collect_training_data(predictions_dir=None):
    """Scan predictions/*graded*.json files and build a training DataFrame-like structure."""
    if predictions_dir is None:
        predictions_dir = PREDICTIONS_DIR

    graded_files = sorted(glob.glob(os.path.join(predictions_dir, '*', '*graded*.json')))

    all_records = []
    file_dates = {}

    for fpath in graded_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Extract records list
        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and 'results' in data:
            records = data['results']
        else:
            continue

        # Extract date from path
        date_dir = os.path.basename(os.path.dirname(fpath))
        file_dates[fpath] = date_dir

        for r in records:
            label = _extract_hit_label(r)
            if label is None:
                continue
            r['_hit_label'] = label
            r['_date'] = date_dir
            r['_source'] = fpath
            all_records.append(r)

    print(f"  Collected {len(all_records)} labeled records from {len(graded_files)} files")

    # Deduplicate (same player+stat+line+date)
    seen = set()
    deduped = []
    for r in all_records:
        key = (r.get('player', ''), r.get('stat', ''), r.get('line', 0), r.get('_date', ''))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    if len(deduped) < len(all_records):
        print(f"  Deduped: {len(all_records)} → {len(deduped)}")

    return deduped


# ═══════════════════════════════════════════════════════════════
# HISTORICAL DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════

CSV_PATH = os.path.join(os.path.dirname(PREDICTIONS_DIR), 'NBA Database (1947 - Present)', 'PlayerStatistics.csv')
HISTORICAL_CACHE_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'historical_features.npz')
BACKFILL_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'backfill_training_data.json')
SGO_BACKFILL_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'sgo_backfill_training_data.json')

# CSV column → model stat name
HIST_STAT_MAP = {
    'pts': 'points', 'reb': 'reboundsTotal', 'ast': 'assists',
    'stl': 'steals', 'blk': 'blocks', '3pm': 'threePointersMade',
}
HIST_COMBO_MAP = {
    'pra': ('points', 'reboundsTotal', 'assists'),
    'pr': ('points', 'reboundsTotal'),
    'pa': ('points', 'assists'),
    'ra': ('reboundsTotal', 'assists'),
}


def _load_historical_csv(seasons_from=2020):
    """Load and filter PlayerStatistics.csv. Returns {personId: [sorted game dicts]}."""
    cutoff = f'{seasons_from}-10-01'
    player_games = {}

    with open(CSV_PATH, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = row.get('gameDateTimeEst', '')[:10]
            if dt < cutoff:
                continue
            if row.get('gameType', '') != 'Regular Season':
                continue
            try:
                mins = float(row.get('numMinutes', 0))
            except (ValueError, TypeError):
                continue
            if mins < 20:
                continue

            pid = row['personId']
            game = {
                'date': dt,
                'home': int(row.get('home', 0)),
                'numMinutes': mins,
                'points': float(row.get('points', 0)),
                'reboundsTotal': float(row.get('reboundsTotal', 0)),
                'assists': float(row.get('assists', 0)),
                'steals': float(row.get('steals', 0)),
                'blocks': float(row.get('blocks', 0)),
                'threePointersMade': float(row.get('threePointersMade', 0)),
            }
            player_games.setdefault(pid, []).append(game)

    # Sort each player's games by date
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g['date'])

    total_games = sum(len(v) for v in player_games.values())
    print(f"  Historical CSV: {len(player_games)} players, {total_games} games (since {seasons_from})")
    return player_games


def _compute_rolling(games, stat_col, idx, line):
    """Compute rolling stats for a player's game list at index idx.

    Returns dict of features or None if insufficient history.
    """
    if idx < 10:
        return None

    prior = games[:idx]
    vals = [g[stat_col] for g in prior]

    season_avg = np.mean(vals)
    l10_vals = vals[-10:]
    l5_vals = vals[-5:]
    l3_vals = vals[-3:]
    l10_avg = np.mean(l10_vals)
    l5_avg = np.mean(l5_vals)
    l3_avg = np.mean(l3_vals)

    home_vals = [g[stat_col] for g in prior if g['home'] == 1]
    away_vals = [g[stat_col] for g in prior if g['home'] == 0]
    home_avg = np.mean(home_vals) if home_vals else season_avg
    away_avg = np.mean(away_vals) if away_vals else season_avg

    # Hit rates vs line
    l10_hits = sum(1 for v in l10_vals if v > line)
    l5_hits = sum(1 for v in l5_vals if v > line)
    season_hits = sum(1 for v in vals if v > line)
    l10_hit_rate = l10_hits / len(l10_vals) * 100
    l5_hit_rate = l5_hits / len(l5_vals) * 100
    season_hit_rate = season_hits / len(vals) * 100

    # Minutes
    prior_mins = [g['numMinutes'] for g in prior]
    mins_30plus = sum(1 for m in prior_mins if m >= 30)
    mins_30plus_pct = mins_30plus / len(prior_mins) * 100

    return {
        'season_avg': season_avg, 'l10_avg': l10_avg, 'l5_avg': l5_avg, 'l3_avg': l3_avg,
        'home_avg': home_avg, 'away_avg': away_avg,
        'l10_hit_rate': l10_hit_rate, 'l5_hit_rate': l5_hit_rate, 'season_hit_rate': season_hit_rate,
        'mins_30plus_pct': mins_30plus_pct, 'games_used': len(vals),
        'l10_floor': float(np.min(l10_vals)),
        'l10_miss_count': 0,  # recomputed after line is known
        'l10_std': float(np.std(l10_vals)),
    }


def _generate_synthetic_records(player_games, sample_cap=50000):
    """Generate synthetic OVER/UNDER prop records from historical game data."""
    all_records = []

    for pid, games in player_games.items():
        for idx in range(10, len(games)):
            game = games[idx]
            game_date = game['date']
            is_home = game['home']

            # B2B detection
            prev_date = games[idx - 1]['date']
            try:
                d1 = datetime.strptime(game_date, '%Y-%m-%d')
                d0 = datetime.strptime(prev_date, '%Y-%m-%d')
                is_b2b = (d1 - d0).days <= 1
            except ValueError:
                is_b2b = False

            # Season weighting for sampling priority
            year = int(game_date[:4])
            month = int(game_date[5:7])
            season_year = year if month >= 10 else year - 1
            if season_year >= 2025:
                weight = 3
            elif season_year >= 2024:
                weight = 2
            else:
                weight = 1

            # Base stats
            for stat_name, csv_col in HIST_STAT_MAP.items():
                actual = game[csv_col]
                rolling = _compute_rolling(games, csv_col, idx, 0)  # temp line=0
                if rolling is None:
                    continue

                line = round(rolling['l10_avg'] * 2) / 2  # nearest 0.5
                # Add line noise — real sportsbook lines aren't exactly at averages
                noise = np.random.normal(0, rolling['l10_std'] * 0.3) if rolling['l10_std'] > 0 else 0
                line = round((line + noise) * 2) / 2  # re-snap to 0.5
                if line <= 0:
                    continue

                # Recompute hit rates against actual line
                prior_vals = [g[csv_col] for g in games[:idx]]
                l10_vals = prior_vals[-10:]
                l5_vals = prior_vals[-5:]
                rolling['l10_hit_rate'] = sum(1 for v in l10_vals if v > line) / len(l10_vals) * 100
                rolling['l5_hit_rate'] = sum(1 for v in l5_vals if v > line) / len(l5_vals) * 100
                rolling['season_hit_rate'] = sum(1 for v in prior_vals if v > line) / len(prior_vals) * 100
                rolling['l10_miss_count'] = sum(1 for v in l10_vals if v <= line)

                if actual == line:  # push — skip
                    continue

                # Generate OVER record
                projection = 0.4 * rolling['l10_avg'] + 0.3 * rolling['l5_avg'] + 0.3 * rolling['season_avg']
                gap = projection - line
                # Compute tier from gap using S/A/B/C/D/F thresholds
                _abs_g = abs(gap)
                if _abs_g >= 4: _tier = 'S'
                elif _abs_g >= 3: _tier = 'A'
                elif _abs_g >= 2: _tier = 'B'
                elif _abs_g >= 1.5: _tier = 'C'
                elif _abs_g >= 1: _tier = 'D'
                else: _tier = 'F'

                rec_base = {
                    'line': line, 'projection': projection,
                    'gap': gap, 'abs_gap': abs(gap), 'effective_gap': gap,
                    'stat': stat_name,
                    'is_home': is_home, 'is_b2b': is_b2b,
                    '_date': game_date, '_source': 'historical_synthetic',
                    '_data_source': 'historical', '_weight': weight,
                    '_feature_version': 6,
                    'same_team_out_count': 0,
                    'opponent_history': None,
                    'l10_values': list(l10_vals),
                    'streak_status': 'NEUTRAL',
                    'tier': _tier,
                    'spread': np.nan,
                    'streak_pct': np.nan,
                }
                rec_base.update(rolling)

                # OVER
                over_rec = dict(rec_base)
                over_rec['direction'] = 'OVER'
                over_rec['_hit_label'] = actual > line
                all_records.append(over_rec)

                # UNDER
                under_rec = dict(rec_base)
                under_rec['direction'] = 'UNDER'
                under_rec['effective_gap'] = line - projection
                under_rec['_hit_label'] = actual < line
                under_rec['_weight'] = weight
                all_records.append(under_rec)

            # Combo stats
            for combo_name, cols in HIST_COMBO_MAP.items():
                actual = sum(game[c] for c in cols)
                # Use first col for rolling base, but compute combo rolling manually
                prior_combo = [sum(g[c] for c in cols) for g in games[:idx]]
                if len(prior_combo) < 10:
                    continue

                l10_avg = np.mean(prior_combo[-10:])
                line = round(l10_avg * 2) / 2
                if line <= 0 or actual == line:
                    continue

                l5_avg = np.mean(prior_combo[-5:])
                l3_avg = np.mean(prior_combo[-3:])
                season_avg = np.mean(prior_combo)
                home_combo = [sum(g[c] for c in cols) for g in games[:idx] if g['home'] == 1]
                away_combo = [sum(g[c] for c in cols) for g in games[:idx] if g['home'] == 0]

                projection = 0.4 * l10_avg + 0.3 * l5_avg + 0.3 * season_avg
                gap = projection - line

                l10_combo = prior_combo[-10:]
                # Add line noise for combos too
                combo_std = float(np.std(l10_combo))
                combo_noise = np.random.normal(0, combo_std * 0.3) if combo_std > 0 else 0
                line = round((line + combo_noise) * 2) / 2
                if line <= 0 or actual == line:
                    continue

                # Recompute gap after noise
                gap = projection - line

                # Compute tier from gap
                _abs_g = abs(gap)
                if _abs_g >= 4: _tier = 'S'
                elif _abs_g >= 3: _tier = 'A'
                elif _abs_g >= 2: _tier = 'B'
                elif _abs_g >= 1.5: _tier = 'C'
                elif _abs_g >= 1: _tier = 'D'
                else: _tier = 'F'

                rec_base = {
                    'line': line, 'projection': projection,
                    'gap': gap, 'abs_gap': abs(gap), 'effective_gap': gap,
                    'season_avg': season_avg, 'l10_avg': l10_avg, 'l5_avg': l5_avg, 'l3_avg': l3_avg,
                    'home_avg': np.mean(home_combo) if home_combo else season_avg,
                    'away_avg': np.mean(away_combo) if away_combo else season_avg,
                    'l10_hit_rate': sum(1 for v in l10_combo if v > line) / 10 * 100,
                    'l5_hit_rate': sum(1 for v in prior_combo[-5:] if v > line) / 5 * 100,
                    'season_hit_rate': sum(1 for v in prior_combo if v > line) / len(prior_combo) * 100,
                    'mins_30plus_pct': sum(1 for g in games[:idx] if g['numMinutes'] >= 30) / idx * 100,
                    'games_used': len(prior_combo),
                    'stat': combo_name,
                    'is_home': is_home, 'is_b2b': is_b2b,
                    '_date': game_date, '_source': 'historical_synthetic',
                    '_data_source': 'historical', '_weight': weight,
                    '_feature_version': 6,
                    'l10_floor': float(np.min(l10_combo)),
                    'l10_miss_count': sum(1 for v in l10_combo if v <= line),
                    'l10_std': float(np.std(l10_combo)),
                    'same_team_out_count': 0,
                    'opponent_history': None,
                    'l10_values': list(l10_combo),
                    'streak_status': 'NEUTRAL',
                    'tier': _tier,
                    'spread': np.nan,
                    'streak_pct': np.nan,
                }

                over_rec = dict(rec_base)
                over_rec['direction'] = 'OVER'
                over_rec['_hit_label'] = actual > line
                all_records.append(over_rec)

                under_rec = dict(rec_base)
                under_rec['direction'] = 'UNDER'
                under_rec['effective_gap'] = line - projection
                under_rec['_hit_label'] = actual < line
                under_rec['_weight'] = weight
                all_records.append(under_rec)

    print(f"  Generated {len(all_records)} raw synthetic records")

    # Weighted sampling to cap
    if len(all_records) > sample_cap:
        rng = np.random.RandomState(42)
        weights = np.array([r['_weight'] for r in all_records], dtype=np.float64)
        # Boost base stats over combos
        for i, r in enumerate(all_records):
            if r['stat'] not in COMBO_STATS:
                weights[i] *= 1.5
        probs = weights / weights.sum()
        indices = rng.choice(len(all_records), size=sample_cap, replace=False, p=probs)
        all_records = [all_records[i] for i in indices]
        print(f"  Sampled down to {len(all_records)} records (cap={sample_cap})")

    return all_records


def collect_historical_data(sample_cap=50000):
    """Orchestrate historical CSV → synthetic prop records with caching."""
    # Check cache — invalidate if schema changed (missing new features)
    if os.path.exists(HISTORICAL_CACHE_PATH):
        try:
            cached = np.load(HISTORICAL_CACHE_PATH, allow_pickle=True)
            sample = cached['records'].tolist()[0]
            if 'l10_floor' not in sample or 'tier' not in sample or '_feature_version' not in sample or sample.get('_feature_version', 0) < 7:
                print("  Cache outdated (missing new features), regenerating...")
                os.remove(HISTORICAL_CACHE_PATH)
        except Exception:
            pass

    if os.path.exists(HISTORICAL_CACHE_PATH) and os.path.exists(CSV_PATH):
        csv_mtime = os.path.getmtime(CSV_PATH)
        cache_mtime = os.path.getmtime(HISTORICAL_CACHE_PATH)
        if cache_mtime > csv_mtime:
            print("  Loading cached historical features...")
            data = np.load(HISTORICAL_CACHE_PATH, allow_pickle=True)
            records = data['records'].tolist()
            print(f"  Loaded {len(records)} cached historical records")
            # Re-sample if cap changed
            if len(records) > sample_cap:
                rng = np.random.RandomState(42)
                records = [records[i] for i in rng.choice(len(records), sample_cap, replace=False)]
            return records

    print("  Processing historical CSV (this may take 15-30s)...")
    player_games = _load_historical_csv(seasons_from=2020)
    records = _generate_synthetic_records(player_games, sample_cap=sample_cap)

    # Cache
    np.savez_compressed(HISTORICAL_CACHE_PATH, records=np.array(records, dtype=object))
    print(f"  Cached to {HISTORICAL_CACHE_PATH}")

    return records


def collect_backfill_data(sample_cap=999999999):
    """Load backfill training data from nba_api season reconstruction.

    Returns list of record dicts ready for engineer_features().
    Run `python3 predictions/backfill_training_data.py` first to generate.
    """
    if not os.path.exists(BACKFILL_PATH):
        print(f"  No backfill data at {BACKFILL_PATH}")
        print(f"  Run: python3 predictions/backfill_training_data.py")
        return []

    with open(BACKFILL_PATH) as f:
        records = json.load(f)

    # Ensure required fields
    for r in records:
        r['_data_source'] = 'backfill'
        if '_hit_label' not in r:
            label = _extract_hit_label(r)
            if label is None:
                continue
            r['_hit_label'] = label

    valid = [r for r in records if '_hit_label' in r and r['_hit_label'] is not None]

    # Weighted sampling to cap — prefer recent dates and higher-tier props
    if len(valid) > sample_cap:
        rng = np.random.RandomState(42)
        weights = np.ones(len(valid))
        for i, r in enumerate(valid):
            # Recency: 2026 dates get 2x weight
            if r.get('_date', '') >= '2026-':
                weights[i] *= 2.0
            # Base stats slightly preferred over combos
            if r.get('stat', '') not in ('pra', 'pr', 'pa', 'ra'):
                weights[i] *= 1.2
        probs = weights / weights.sum()
        indices = rng.choice(len(valid), size=sample_cap, replace=False, p=probs)
        valid = [valid[i] for i in indices]

    print(f"  Backfill data: {len(valid)} records from {BACKFILL_PATH}")
    return valid


def collect_sgo_backfill_data(sample_cap=999999999):
    """Load SGO box score backfill training data.

    Different data source than nba_api backfill (cross-source diversity).
    Has real plus_minus, covers 2+ seasons.
    Run `python3 predictions/backfill_sgo_box_scores.py` first to generate.
    """
    if not os.path.exists(SGO_BACKFILL_PATH):
        print(f"  No SGO backfill data at {SGO_BACKFILL_PATH}")
        print(f"  Run: python3 predictions/backfill_sgo_box_scores.py")
        return []

    with open(SGO_BACKFILL_PATH) as f:
        records = json.load(f)

    for r in records:
        r['_data_source'] = 'sgo_backfill'
        if '_hit_label' not in r:
            label = _extract_hit_label(r)
            if label is None:
                continue
            r['_hit_label'] = label

    valid = [r for r in records if '_hit_label' in r and r['_hit_label'] is not None]

    if len(valid) > sample_cap:
        rng = np.random.RandomState(44)  # different seed
        weights = np.ones(len(valid))
        for i, r in enumerate(valid):
            # Recency: 2026 = 2x, 2025 = 1.5x
            date = r.get('_date', '')
            if date >= '2026-':
                weights[i] *= 2.0
            elif date >= '2025-':
                weights[i] *= 1.5
            # Base stats slightly preferred
            if r.get('stat', '') not in ('pra', 'pr', 'pa', 'ra'):
                weights[i] *= 1.2
        probs = weights / weights.sum()
        indices = rng.choice(len(valid), size=sample_cap, replace=False, p=probs)
        valid = [valid[i] for i in indices]

    print(f"  SGO backfill data: {len(valid)} records from {SGO_BACKFILL_PATH}")
    return valid


HISTORICAL_10YR_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'historical_10yr_training.json')


def collect_10yr_data(sample_cap=500000):
    """Load 10-year historical backfill from CSV reconstruction.
    2.9M records — pre-samples during load to avoid OOM on large files."""
    if not os.path.exists(HISTORICAL_10YR_PATH):
        print(f"  No 10yr data at {HISTORICAL_10YR_PATH}")
        print(f"  Run: python3 predictions/backfill_historical_csv.py")
        return []

    file_size_gb = os.path.getsize(HISTORICAL_10YR_PATH) / 1e9
    print(f"  Loading 10yr historical data ({file_size_gb:.1f}GB, sampling {sample_cap:,})...")

    # Stream-sample: read all but reservoir-sample to cap memory
    # First pass: count records and collect dates for weighting
    import random as _rand
    _rand.seed(42)

    valid = []
    chunk_size = 500000  # process in chunks to limit peak memory
    with open(HISTORICAL_10YR_PATH) as f:
        all_records = json.load(f)

    print(f"  Loaded {len(all_records):,} raw records, sampling to {sample_cap:,}...")

    # Tag and filter
    for r in all_records:
        r['_data_source'] = 'historical_10yr'
        if '_hit_label' not in r:
            r['_hit_label'] = 1 if r.get('hit') else 0
        if r.get('_hit_label') is not None:
            valid.append(r)

    # Free raw memory immediately
    del all_records
    import gc; gc.collect()

    if len(valid) > sample_cap:
        rng = np.random.RandomState(42)
        weights = np.ones(len(valid))
        for i, r in enumerate(valid):
            date = r.get('date', '')
            if date >= '2025-':
                weights[i] *= 5.0
            elif date >= '2024-':
                weights[i] *= 3.0
            elif date >= '2023-':
                weights[i] *= 2.0
            if r.get('stat', '') not in ('pra', 'pr', 'pa', 'ra', 'stl_blk'):
                weights[i] *= 1.3
        probs = weights / weights.sum()
        indices = rng.choice(len(valid), size=sample_cap, replace=False, p=probs)
        valid = [valid[i] for i in sorted(indices)]
        # Free weight arrays
        del weights, probs, indices

    print(f"  10yr data: {len(valid):,} records")
    return valid


def collect_all_training_data(use_historical=True, sample_cap=15000):
    """Combine graded + backfill + sgo_backfill + 10yr historical + legacy historical."""
    graded = collect_training_data()
    for r in graded:
        r['_data_source'] = 'graded'

    # Load backfill data (nba_api season reconstruction — high quality)
    backfill = collect_backfill_data()

    # Load SGO box score backfill (cross-source diversity, real plus_minus)
    sgo_backfill = collect_sgo_backfill_data()

    # Load 10-year CSV historical (massive dataset, sampled)
    hist_10yr = collect_10yr_data(sample_cap=100000)  # 100K from 3M (recency weighted, memory safe for all models)

    if not use_historical:
        all_data = graded + backfill + sgo_backfill + hist_10yr
        print(f"  Combined: {len(graded)} graded + {len(backfill)} backfill + {len(sgo_backfill)} sgo_backfill + {len(hist_10yr)} 10yr = {len(all_data)} total")
        return all_data

    historical = collect_historical_data(sample_cap=50000)

    all_data = graded + backfill + sgo_backfill + hist_10yr + historical
    print(f"  Combined: {len(graded)} graded + {len(backfill)} backfill + {len(sgo_backfill)} sgo_backfill + {len(hist_10yr)} 10yr + {len(historical)} legacy = {len(all_data)} total")
    return all_data


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def _safe_float(val, default=np.nan):
    """Convert to float, return NaN for None/invalid."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def engineer_features(records):
    """Convert list of record dicts into (feature_matrix, labels, feature_names).

    Returns:
        X: numpy array (n_samples, n_features) — NaN for missing values (XGBoost handles natively)
        y: numpy array (n_samples,) — 0/1
        dates: list of date strings for walk-forward splits
    """
    X_rows = []
    y_rows = []
    dates = []

    for r in records:
        line = _safe_float(r.get('line'))
        projection = _safe_float(r.get('projection'))
        gap = _safe_float(r.get('gap'))
        abs_gap = _safe_float(r.get('abs_gap'))
        effective_gap = _safe_float(r.get('effective_gap'))
        season_avg = _safe_float(r.get('season_avg'))
        l10_avg = _safe_float(r.get('l10_avg'))
        l5_avg = _safe_float(r.get('l5_avg'))
        l3_avg = _safe_float(r.get('l3_avg'))
        home_avg = _safe_float(r.get('home_avg'))
        away_avg = _safe_float(r.get('away_avg'))
        l10_hr = _safe_float(r.get('l10_hit_rate'))
        l5_hr = _safe_float(r.get('l5_hit_rate'))
        season_hr = _safe_float(r.get('season_hit_rate'))
        mins_pct = _safe_float(r.get('mins_30plus_pct'))
        split_adj = _safe_float(r.get('split_adjustment'), 0)
        matchup_adj = _safe_float(r.get('matchup_adjustment'), 0)
        mins_adj = _safe_float(r.get('mins_adj'), 0)
        streak_adj = _safe_float(r.get('streak_adj'), 0)
        blowout_adj = _safe_float(r.get('blowout_adj'), 0)
        injury_adj = _safe_float(r.get('injury_adjustment'), 0)
        spread = _safe_float(r.get('spread'))
        streak_pct = _safe_float(r.get('streak_pct'))
        games_used = _safe_float(r.get('games_used'))

        # Engineered features
        gap_over_line = abs_gap / line if line and not np.isnan(line) and line > 0 else np.nan
        season_vs_line = (season_avg - line) / line if line and not np.isnan(line) and not np.isnan(season_avg) and line > 0 else np.nan
        l5_vs_l10 = l5_avg - l10_avg if not np.isnan(l5_avg) and not np.isnan(l10_avg) else np.nan
        l5_l10_hr_delta = l5_hr - l10_hr if not np.isnan(l5_hr) and not np.isnan(l10_hr) else np.nan

        # Margin over line (directional)
        direction = r.get('direction', '')
        if direction == 'OVER' and not np.isnan(season_avg) and line and not np.isnan(line) and line > 0:
            margin_over_line = (season_avg - line) / line
        elif direction == 'UNDER' and not np.isnan(season_avg) and line and not np.isnan(line) and line > 0:
            margin_over_line = (line - season_avg) / line
        else:
            margin_over_line = np.nan

        projection_confidence = min(games_used, 60) / 60.0 if not np.isnan(games_used) else np.nan
        total_adjustment = split_adj + matchup_adj + mins_adj + streak_adj + blowout_adj + injury_adj
        gap_x_hr = abs_gap * l10_hr / 100.0 if not np.isnan(abs_gap) and not np.isnan(l10_hr) else np.nan
        mins_x_gap = mins_pct * abs_gap / 100.0 if not np.isnan(mins_pct) and not np.isnan(abs_gap) else np.nan

        # Floor/consistency (from analyze_v3 pipeline)
        l10_floor_val = _safe_float(r.get('l10_floor'))
        l10_miss_count_val = _safe_float(r.get('l10_miss_count'))
        floor_gap = line - l10_floor_val if not np.isnan(line) and not np.isnan(l10_floor_val) else np.nan

        l10_values = r.get('l10_values', [])
        l10_std = float(np.std(l10_values)) if isinstance(l10_values, list) and len(l10_values) >= 3 else np.nan

        # Opponent history
        opp_hist = r.get('opponent_history')
        if isinstance(opp_hist, dict):
            opp_hit_rate = _safe_float(opp_hist.get('hit_rate'))
            opp_avg = _safe_float(opp_hist.get('avg'))
            opp_avg_vs_line = opp_avg - line if not np.isnan(opp_avg) and not np.isnan(line) else np.nan
        else:
            opp_hit_rate = np.nan
            opp_avg_vs_line = np.nan

        same_team_out = _safe_float(r.get('same_team_out_count'), 0)

        # Miss streak: consecutive misses from end of L10
        miss_streak = 0
        if isinstance(l10_values, list) and len(l10_values) > 0 and not np.isnan(line):
            for v in reversed(l10_values):
                try:
                    v_f = float(v)
                except (ValueError, TypeError):
                    break
                if (direction == 'OVER' and v_f <= line) or (direction == 'UNDER' and v_f >= line):
                    miss_streak += 1
                else:
                    break

        # Categorical encodings
        stat = r.get('stat', '')
        stat_ordinal = STAT_ORDINAL.get(stat, 0)
        direction_binary = 1 if direction == 'OVER' else 0
        is_home_binary = 1 if r.get('is_home') else 0
        is_b2b_binary = 1 if r.get('is_b2b') else 0
        is_combo = 1 if stat in COMBO_STATS else 0
        streak_status = r.get('streak_status', r.get('streak', 'NEUTRAL'))
        streak_ord = STREAK_ORDINAL.get(streak_status, 0)
        tier = r.get('tier', 'F')
        tier_ord = TIER_ORDINAL.get(tier, 0)

        # v2 features
        gap_squared = gap * abs(gap) if not np.isnan(gap) else np.nan  # sign-preserving
        is_pts = 1.0 if stat == 'pts' else 0.0
        is_blk_stl = 1.0 if stat in ('blk', 'stl') else 0.0
        is_ast = 1.0 if stat == 'ast' else 0.0
        l5_l10_agreement = np.nan
        if not np.isnan(l5_avg) and not np.isnan(l10_avg) and not np.isnan(line):
            l5_l10_agreement = 1.0 if (l5_avg > line) == (l10_avg > line) else 0.0
        hr_confidence = np.nan
        if not np.isnan(l10_hr) and not np.isnan(l5_hr) and not np.isnan(season_hr):
            hr_confidence = (l10_hr + l5_hr + season_hr) / 300.0
        line_relative_avg = np.nan
        if not np.isnan(line) and not np.isnan(season_avg) and season_avg > 0:
            line_relative_avg = line / season_avg

        # SGO-derived features (NaN when unavailable — XGBoost handles natively)
        book_line_spread_val = _safe_float(r.get('book_line_spread'))
        line_vs_consensus_val = _safe_float(r.get('line_vs_consensus'))
        n_books_val = _safe_float(r.get('n_books'))

        # v4 features (scout data + screen data)
        l10_avg_pm = _safe_float(r.get('l10_avg_plus_minus'))
        l10_avg_pf_val = _safe_float(r.get('l10_avg_pf'))
        foul_trouble_binary = 1.0 if r.get('foul_trouble_risk') else 0.0
        venue_altitude = _safe_float(r.get('scout_venue', {}).get('venue_altitude') if isinstance(r.get('scout_venue'), dict) else None)
        travel_zone_diff_val = _safe_float(r.get('scout_venue', {}).get('travel_zone_diff') if isinstance(r.get('scout_venue'), dict) else None)
        screen_tier_map = {'CORE': 3, 'FLEX': 2, 'REACH': 1}
        screen_tier_ord = float(screen_tier_map.get(r.get('screen_tier', ''), 0))
        eval_agree = _safe_float(r.get('eval_agreement_count'))
        eval_max_conf = _safe_float(r.get('eval_max_confidence'))

        # v5 features (rate-based defense, usage, travel)
        opp_allowed_rate = _safe_float(r.get('opp_stat_allowed_rate'))
        opp_allowed_vs_league = _safe_float(r.get('opp_stat_allowed_vs_league_avg'))
        usage_rate_val = _safe_float(r.get('usage_rate'))
        usage_trend_val = _safe_float(r.get('usage_trend'))
        dynamic_without = _safe_float(r.get('dynamic_without_delta'))
        travel_dist = _safe_float(r.get('travel_distance'))
        travel_7day = _safe_float(r.get('travel_miles_7day'))
        tz_shifts = _safe_float(r.get('tz_shifts_7day'))

        # v6 interaction features
        under_x_cold = 1.0 if direction == 'UNDER' and streak_status == 'COLD' else 0.0
        under_x_blkstl = 1.0 if direction == 'UNDER' and stat in ('blk', 'stl') else 0.0
        hot_x_over = 1.0 if streak_status == 'HOT' and direction == 'OVER' else 0.0
        under_x_gap_val = abs_gap if direction == 'UNDER' and not np.isnan(abs_gap) else 0.0
        under_x_hr_val = (l10_hr / 100.0) if direction == 'UNDER' and not np.isnan(l10_hr) else 0.0
        combo_x_over = 1.0 if is_combo and direction == 'OVER' else 0.0
        directional_gap_val = gap if direction == 'OVER' and not np.isnan(gap) else (-gap if direction == 'UNDER' and not np.isnan(gap) else np.nan)
        hr_x_gap_val = (l10_hr / 100.0) * abs_gap if not np.isnan(l10_hr) and not np.isnan(abs_gap) else np.nan
        cold_under_x_gap = abs_gap if direction == 'UNDER' and streak_status == 'COLD' and not np.isnan(abs_gap) else 0.0
        streak_x_dir = streak_ord * direction_binary

        # v7 features (v11 enrichment — now available because enrichment runs before scoring)
        opp_matchup_delta_val = _safe_float(r.get('opp_matchup_delta'))
        team_vs_opp_delta_val = _safe_float(r.get('team_vs_opp_delta'))
        opp_off_pressure_val = _safe_float(r.get('opp_off_pressure'))
        usage_boost_val = _safe_float(r.get('usage_boost'))
        game_total_signal_val = _safe_float(r.get('game_total_signal'))
        max_same_game_corr_val = _safe_float(r.get('max_same_game_corr'))
        # Interaction: UNDER + stingy defense (opp allows less than avg)
        under_x_opp_favorable = 0.0
        if direction == 'UNDER' and not np.isnan(opp_allowed_vs_league) and opp_allowed_vs_league < 0:
            under_x_opp_favorable = abs(opp_allowed_vs_league)
        # Interaction: game total aligned with direction
        game_total_x_dir = 0.0
        if not np.isnan(game_total_signal_val):
            if direction == 'OVER' and game_total_signal_val > 0:
                game_total_x_dir = game_total_signal_val
            elif direction == 'UNDER' and game_total_signal_val < 0:
                game_total_x_dir = abs(game_total_signal_val)

        # (REMOVED: ref/coach/sim feature reads — 98-99% NaN, never in training row)

        # v10 features (research-backed improvements: EWMA, median, production rate, L20)
        l10_values_raw = r.get('l10_values', [])
        if isinstance(l10_values_raw, list) and len(l10_values_raw) >= 3:
            _vals = np.array([float(v) for v in l10_values_raw if v is not None], dtype=np.float64)
            if len(_vals) >= 3:
                # EWMA with alpha=0.15 (most recent game gets highest weight)
                _weights = np.array([(1 - 0.15) ** i for i in range(len(_vals) - 1, -1, -1)])
                l10_ewma_val = float(np.average(_vals, weights=_weights))
                l10_median_val = float(np.median(_vals))
                mean_median_gap_val = (l10_avg - l10_median_val) / line if not np.isnan(l10_avg) and not np.isnan(line) and line > 0 else np.nan
                _mean_v = float(np.mean(_vals))
                l10_cv_val = float(np.std(_vals) / _mean_v) if _mean_v > 0 else np.nan
            else:
                l10_ewma_val = np.nan
                l10_median_val = np.nan
                mean_median_gap_val = np.nan
                l10_cv_val = np.nan
        else:
            l10_ewma_val = np.nan
            l10_median_val = np.nan
            mean_median_gap_val = np.nan
            l10_cv_val = np.nan

        # (REMOVED: production_rate, l20 stats, v11 advanced CSV (24 fields), v9 NN embeddings (32 dims) — all dead/NaN)

        row = [
            line, projection, gap, abs_gap, effective_gap,
            season_avg, l10_avg, l5_avg, l3_avg,
            home_avg, away_avg,
            l10_hr, l5_hr, season_hr,
            mins_pct,
            split_adj, matchup_adj, mins_adj,
            streak_adj, blowout_adj, injury_adj,
            spread, streak_pct, games_used,
            gap_over_line, season_vs_line, l5_vs_l10,
            l5_l10_hr_delta, margin_over_line, projection_confidence,
            total_adjustment, gap_x_hr, mins_x_gap,
            l10_floor_val, l10_miss_count_val, floor_gap, l10_std,
            opp_hit_rate, opp_avg_vs_line,
            same_team_out, float(miss_streak),
            stat_ordinal, direction_binary, is_home_binary,
            is_b2b_binary, is_combo, streak_ord,
            # v2 features
            gap_squared, is_pts, is_blk_stl, is_ast,
            l5_l10_agreement, hr_confidence, line_relative_avg,
            # v4 features (kept clean ones)
            l10_avg_pm, l10_avg_pf_val, foul_trouble_binary,
            screen_tier_ord,
            # v5 features (kept travel — always populated)
            travel_dist, travel_7day, tz_shifts,
            # v6 interaction features
            under_x_cold, under_x_blkstl, hot_x_over,
            under_x_gap_val, under_x_hr_val, combo_x_over,
            directional_gap_val, hr_x_gap_val, cold_under_x_gap,
            float(streak_x_dir),
            # v7 features (kept game_total only — rest 98%+ NaN)
            game_total_signal_val, game_total_x_dir,
            # v10 features (EWMA, median — computed from l10_values, always available)
            l10_ewma_val, l10_median_val, mean_median_gap_val, l10_cv_val,
        ]

        X_rows.append(row)
        y_rows.append(1 if r['_hit_label'] else 0)
        dates.append(r.get('_date', ''))

    X = np.array(X_rows, dtype=np.float64)
    y = np.array(y_rows, dtype=np.int32)

    print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Hit rate: {y.mean():.1%} ({y.sum()}/{len(y)})")

    return X, y, dates


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def _get_model_params(y):
    """Return XGBoost params with auto-computed scale_pos_weight."""
    n_hit = int(y.sum())
    n_miss = len(y) - n_hit
    scale_pos_weight = n_miss / n_hit if n_hit > 0 else 1.0

    # Build monotonic constraints: (1)=increasing, (-1)=decreasing, (0)=none
    monotone = [0] * len(FEATURE_COLS)
    mono_map = {
        'abs_gap': 1,          # bigger gap → more likely to hit
        'l10_hit_rate': 1,     # higher HR → more likely to hit
        'l5_hit_rate': 1,      # higher HR → more likely to hit
        'season_hit_rate': 1,  # higher HR → more likely to hit
        'mins_30plus_pct': 1,  # more minutes stability → more likely
        'gap_x_hr': 1,         # higher conviction → more likely
        'hr_confidence': 1,    # higher overall confidence → more likely
        'l10_miss_count': -1,  # more misses → less likely
        'miss_streak': -1,     # longer miss streak → less likely
        'under_x_cold': 1,     # UNDER+COLD proven positive
        'under_x_blkstl': 1,   # UNDER+BLK/STL proven positive
        'hot_x_over': -1,      # HOT+OVER proven negative (49.2% trap)
        'combo_x_over': -1,    # combo+OVER proven volatile/negative
        'under_x_opp_favorable': 1,  # UNDER + stingy defense = positive
        'game_total_x_direction': 1, # aligned game total = positive
    }
    for feat, direction in mono_map.items():
        if feat in FEATURE_COLS:
            monotone[FEATURE_COLS.index(feat)] = direction

    return {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 4,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'colsample_bylevel': 0.8,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'gamma': 0.3,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbosity': 0,
        'monotone_constraints': tuple(monotone),
    }


def walk_forward_cv(X, y, dates, sample_weights=None, sources=None):
    """Leave-one-day-out walk-forward cross-validation.

    Train on earlier days, test on next day. Returns per-fold metrics.
    Historical data (date < '2026-') is included in ALL training folds as background.
    Backfill data is included chronologically in training (never in test set).
    CV test splits only on graded data dates.
    """
    dates_arr = np.array(dates)
    sources_arr = np.array(sources) if sources is not None else np.array(['graded'] * len(dates))

    # Only graded records define test folds
    graded_mask = sources_arr == 'graded'
    graded_dates = sorted(set(d for d, s in zip(dates, sources_arr) if s == 'graded' and d >= '2026-'))
    # Historical = CSV synthetic (always in training)
    historical_mask = np.array([d < '2026-' and s == 'historical' for d, s in zip(dates, sources_arr)])
    # Backfill = nba_api reconstruction (included chronologically in training, never in test)
    backfill_mask = np.isin(sources_arr, ['backfill', 'sgo_backfill'])

    if len(graded_dates) < 2:
        print("  WARNING: Need at least 2 graded days for walk-forward CV")
        return []

    folds = []
    for i in range(1, len(graded_dates)):
        train_graded_dates = set(graded_dates[:i])
        test_date = graded_dates[i]

        # Train = historical (always) + backfill before test date + graded before test date
        train_graded_mask = np.array([d in train_graded_dates and s == 'graded' for d, s in zip(dates, sources_arr)])
        train_backfill_mask = np.array([d < test_date and s in ('backfill', 'sgo_backfill') for d, s in zip(dates, sources_arr)])
        train_mask = historical_mask | train_graded_mask | train_backfill_mask
        # Test = graded records on this date only
        test_mask = (dates_arr == test_date) & graded_mask

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        sw_train = sample_weights[train_mask] if sample_weights is not None else None

        params = _get_model_params(y_train)
        model = XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
            early_stopping_rounds=50,
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        accuracy = (y_pred == y_test).mean()
        auc = _compute_auc(y_test, y_prob)
        logloss = _compute_logloss(y_test, y_prob)
        brier = _compute_brier(y_test, y_prob)

        # Calibration: top/bottom decile
        top_mask = y_prob >= np.percentile(y_prob, 90)
        bot_mask = y_prob <= np.percentile(y_prob, 10)
        top_hr = y_test[top_mask].mean() if top_mask.sum() > 0 else np.nan
        bot_hr = y_test[bot_mask].mean() if bot_mask.sum() > 0 else np.nan

        fold = {
            'train_graded_dates': list(train_graded_dates),
            'test_date': test_date,
            'train_size': int(train_mask.sum()),
            'test_size': int(test_mask.sum()),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'logloss': float(logloss),
            'brier': float(brier),
            'top_decile_hr': float(top_hr) if not np.isnan(top_hr) else None,
            'bot_decile_hr': float(bot_hr) if not np.isnan(bot_hr) else None,
            'y_test': y_test,
            'y_prob': y_prob,
        }
        folds.append(fold)

        print(f"    Fold: train {sorted(train_graded_dates)} → test {test_date}")
        print(f"      N={fold['test_size']}, Acc={accuracy:.3f}, AUC={auc:.3f}, Brier={brier:.4f}, "
              f"Top10%={top_hr:.1%}, Bot10%={bot_hr:.1%}")

    # Pooled CV metrics across all folds
    if len(folds) >= 2:
        all_y_test = np.concatenate([f['y_test'] for f in folds])
        all_y_prob = np.concatenate([f['y_prob'] for f in folds])
        pooled_auc = _compute_auc(all_y_test, all_y_prob)
        pooled_brier = _compute_brier(all_y_test, all_y_prob)
        p90 = np.percentile(all_y_prob, 90)
        p10 = np.percentile(all_y_prob, 10)
        pooled_top10 = all_y_test[all_y_prob >= p90].mean() if (all_y_prob >= p90).sum() > 0 else np.nan
        pooled_bot10 = all_y_test[all_y_prob <= p10].mean() if (all_y_prob <= p10).sum() > 0 else np.nan
        print(f"\n    Pooled CV: AUC={pooled_auc:.3f}, Brier={pooled_brier:.4f}, Top10%={pooled_top10:.1%}, Bot10%={pooled_bot10:.1%}")
        print(f"    Pooled N={len(all_y_test)}, Hit rate={all_y_test.mean():.1%}")

    # Strip numpy arrays before returning (not JSON-serializable)
    for fold in folds:
        fold.pop('y_test', None)
        fold.pop('y_prob', None)

    return folds


def _compute_auc(y_true, y_prob):
    """Simple AUC computation without sklearn dependency."""
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U statistic
    auc = 0.0
    for p in pos:
        auc += (neg < p).sum() + 0.5 * (neg == p).sum()
    return auc / (len(pos) * len(neg))


def _build_calibration_map(folds, X, y, dates, sample_weights, sources, params):
    """Build Platt-style calibration from CV out-of-fold predictions.

    Creates a bin-based mapping: raw_prob_bin → actual_hit_rate.
    Stored in metadata and applied during scoring.
    """
    if not folds:
        return None

    # Rebuild out-of-fold predictions for graded test data
    dates_arr = np.array(dates)
    sources_arr = np.array(sources) if sources is not None else np.array(['graded'] * len(dates))
    graded_mask = sources_arr == 'graded'
    graded_dates = sorted(set(d for d, s in zip(dates, sources_arr) if s == 'graded' and d >= '2026-'))
    historical_mask = np.array([d < '2026-' and s == 'historical' for d, s in zip(dates, sources_arr)])
    backfill_mask = np.isin(sources_arr, ['backfill', 'sgo_backfill'])

    all_probs = []
    all_actuals = []

    for i in range(1, len(graded_dates)):
        train_graded_dates = set(graded_dates[:i])
        test_date = graded_dates[i]

        train_graded_m = np.array([d in train_graded_dates and s == 'graded' for d, s in zip(dates, sources_arr)])
        train_backfill_m = np.array([d < test_date and s in ('backfill', 'sgo_backfill') for d, s in zip(dates, sources_arr)])
        train_mask = historical_mask | train_graded_m | train_backfill_m
        test_mask = (dates_arr == test_date) & graded_mask

        if train_mask.sum() < 50 or test_mask.sum() < 5:
            continue

        sw_train = sample_weights[train_mask] if sample_weights is not None else None
        m = XGBClassifier(**params)
        m.fit(X[train_mask], y[train_mask], sample_weight=sw_train, verbose=False)
        probs = m.predict_proba(X[test_mask])[:, 1]
        all_probs.extend(probs.tolist())
        all_actuals.extend(y[test_mask].tolist())

    if len(all_probs) < 50:
        return None

    # Build 10-bin calibration map
    probs_arr = np.array(all_probs)
    actuals_arr = np.array(all_actuals)
    bins = np.linspace(0, 1, 11)  # 10 bins: 0-0.1, 0.1-0.2, ... 0.9-1.0
    cal_map = []
    for j in range(len(bins) - 1):
        mask = (probs_arr >= bins[j]) & (probs_arr < bins[j + 1])
        if mask.sum() >= 5:
            actual_hr = float(actuals_arr[mask].mean())
            cal_map.append({
                'bin_low': round(float(bins[j]), 2),
                'bin_high': round(float(bins[j + 1]), 2),
                'raw_avg': round(float(probs_arr[mask].mean()), 3),
                'actual_hr': round(actual_hr, 3),
                'n': int(mask.sum()),
            })

    if cal_map:
        print(f"\n  Calibration Map (raw_prob → actual_hit_rate):")
        for b in cal_map:
            bar = '#' * int(b['actual_hr'] * 30)
            print(f"    [{b['bin_low']:.1f}-{b['bin_high']:.1f}] raw={b['raw_avg']:.3f} → actual={b['actual_hr']:.3f} (n={b['n']:3d}) {bar}")

    return cal_map


def _calibrate_prob(raw_prob, cal_map):
    """Apply calibration map to a raw xgb_prob. Returns calibrated probability."""
    if not cal_map:
        return raw_prob
    for b in cal_map:
        if b['bin_low'] <= raw_prob < b['bin_high']:
            return b['actual_hr']
    # Edge case: prob exactly 1.0 or outside bins
    return raw_prob


def _compute_logloss(y_true, y_prob):
    """Binary cross-entropy."""
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def _compute_brier(y_true, y_prob):
    """Brier score: mean squared error between predicted probability and actual outcome.
    Range 0-1, lower is better. 0.25 = random baseline for balanced classes."""
    return float(np.mean((y_prob - y_true) ** 2))


def train_model(X, y, dates, save_path=None, sample_weights=None, sources=None):
    """Train final model on all data, run walk-forward CV, save model + metadata.
    For large datasets (>1M), subsamples non-graded data for CV speed, then
    trains the final model on ALL data."""
    if save_path is None:
        save_path = MODEL_PATH

    # For large datasets, subsample for CV (keep all graded + sample historical)
    CV_MAX = 600000
    if len(y) > CV_MAX and sources is not None:
        sources_arr = np.array(sources)
        graded_mask = sources_arr == 'graded'
        other_mask = ~graded_mask

        n_graded = graded_mask.sum()
        n_other_needed = min(CV_MAX - n_graded, other_mask.sum())

        rng = np.random.RandomState(42)
        other_indices = np.where(other_mask)[0]
        sampled_other = rng.choice(other_indices, size=n_other_needed, replace=False)
        cv_indices = np.sort(np.concatenate([np.where(graded_mask)[0], sampled_other]))

        print(f"\n  CV subsample: {len(cv_indices):,} / {len(y):,} "
              f"(all {n_graded:,} graded + {n_other_needed:,} sampled historical)")

        X_cv = X[cv_indices]
        y_cv = y[cv_indices]
        dates_cv = np.array(dates)[cv_indices] if isinstance(dates, list) else dates[cv_indices]
        sw_cv = sample_weights[cv_indices] if sample_weights is not None else None
        sources_cv = [sources[i] for i in cv_indices] if isinstance(sources, list) else sources[cv_indices]

        print("\n  Walk-Forward Cross-Validation (on subsample):")
        folds = walk_forward_cv(X_cv, y_cv, dates_cv, sample_weights=sw_cv, sources=sources_cv)
        del X_cv, y_cv, dates_cv, sw_cv, sources_cv
    else:
        print("\n  Walk-Forward Cross-Validation:")
        folds = walk_forward_cv(X, y, dates, sample_weights=sample_weights, sources=sources)

    if folds:
        avg_auc = np.mean([f['auc'] for f in folds])
        avg_acc = np.mean([f['accuracy'] for f in folds])
        print(f"\n  CV Summary: Avg AUC={avg_auc:.3f}, Avg Acc={avg_acc:.3f}")

        if avg_auc < 0.58:
            print(f"  WARNING: CV AUC {avg_auc:.3f} < 0.58 — model may not be reliable")
    else:
        avg_auc = None
        avg_acc = None

    # Train final model on all data
    print(f"\n  Training final model on {len(y)} samples...")
    params = _get_model_params(y)
    model = XGBClassifier(**params)

    # Use last day as eval set for early stopping
    unique_dates = sorted(set(dates))
    dates_arr = np.array(dates)
    if len(unique_dates) >= 2:
        last_date = unique_dates[-1]
        train_mask = dates_arr != last_date
        eval_mask = dates_arr == last_date
        sw_train = sample_weights[train_mask] if sample_weights is not None else None
        model.fit(
            X[train_mask], y[train_mask],
            sample_weight=sw_train,
            eval_set=[(X[eval_mask], y[eval_mask])],
            verbose=False,
            early_stopping_rounds=50,
        )
        # Refit on all data with the best iteration count
        best_n = model.best_iteration + 1 if hasattr(model, 'best_iteration') and model.best_iteration else params['n_estimators']
        params['n_estimators'] = min(best_n + 20, params['n_estimators'])  # small buffer

    model = XGBClassifier(**params)
    model.fit(X, y, sample_weight=sample_weights, verbose=False)
    # Save via raw Booster to avoid Python 3.14 sklearn wrapper bug
    try:
        model.save_model(save_path)
    except TypeError:
        model.get_booster().save_model(save_path)
    print(f"  Model saved: {save_path}")

    # ── Platt Scaling Calibration ──
    # Build calibration map from pooled CV predictions
    calibration_map = _build_calibration_map(folds, X, y, dates, sample_weights, sources, params)

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'n_samples': int(len(y)),
        'hit_rate': float(y.mean()),
        'n_features': int(X.shape[1]),
        'unique_dates': sorted(set(dates)),
        'cv_folds': folds,
        'cv_avg_auc': float(avg_auc) if avg_auc is not None else None,
        'cv_avg_accuracy': float(avg_acc) if avg_acc is not None else None,
        'top_features': [(name, float(imp)) for name, imp in top_features],
        'calibration_map': calibration_map,
        'model_path': save_path,
        'params': {k: v for k, v in params.items() if k != 'random_state'},
    }

    meta_path = save_path.replace('.json', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

    return model, metadata


# ═══════════════════════════════════════════════════════════════
# SCORING (used by run_board_v5.py)
# ═══════════════════════════════════════════════════════════════

def score_props(results, model_path=None):
    """Load trained model and add xgb_prob to each result dict.

    Args:
        results: list of prop dicts from analyze_v3
        model_path: path to saved model (default: predictions/xgb_model.json)

    Returns:
        results with xgb_prob added to each dict
    """
    if model_path is None:
        model_path = MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. Run: python3 xgb_model.py train")

    # Try sklearn wrapper first, fall back to raw Booster for compatibility
    try:
        model = XGBClassifier()
        model.load_model(model_path)
        use_booster = False
    except (TypeError, Exception):
        # Python 3.14+ or version mismatch — use raw Booster
        model = xgb.Booster()
        model.load_model(model_path)
        use_booster = True

    # Load metadata for confidence gating + calibration
    meta_path = model_path.replace('.json', '_meta.json')
    cal_map = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        cv_auc = meta.get('cv_avg_auc')
        if cv_auc is not None and cv_auc < 0.58:
            print(f"  WARNING: Model CV AUC={cv_auc:.3f} < 0.58 — xgb_prob may be unreliable")
        cal_map = meta.get('calibration_map')

    # Build feature matrix for all results
    temp_records = []
    for r in results:
        rec = dict(r)
        rec['_hit_label'] = False  # dummy — not used for scoring
        rec['_date'] = ''
        temp_records.append(rec)

    if not temp_records:
        return results

    X, _, _ = engineer_features(temp_records)

    # Score
    if use_booster:
        import numpy as np
        dmat = xgb.DMatrix(X)
        probs = model.predict(dmat)
    else:
        probs = model.predict_proba(X)[:, 1]

    scored = 0
    for r, prob in zip(results, probs):
        raw = round(float(prob), 4)
        r['xgb_prob'] = raw
        # Add calibrated probability if calibration map available
        if cal_map:
            r['xgb_prob_calibrated'] = round(_calibrate_prob(raw, cal_map), 4)
        scored += 1

    return results


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def print_report(metadata):
    """Print formatted CV and feature importance report."""
    print("\n" + "=" * 60)
    print("  XGBoost Model Report")
    print("=" * 60)

    print(f"\n  Training data: {metadata['n_samples']} samples, "
          f"hit rate {metadata['hit_rate']:.1%}")
    print(f"  Dates: {', '.join(metadata['unique_dates'])}")

    if metadata.get('cv_avg_auc') is not None:
        print(f"\n  Walk-Forward CV:")
        print(f"    Average AUC:      {metadata['cv_avg_auc']:.3f}")
        print(f"    Average Accuracy: {metadata['cv_avg_accuracy']:.3f}")

        for fold in metadata.get('cv_folds', []):
            top = f"{fold['top_decile_hr']:.1%}" if fold.get('top_decile_hr') is not None else 'N/A'
            bot = f"{fold['bot_decile_hr']:.1%}" if fold.get('bot_decile_hr') is not None else 'N/A'
            print(f"    {fold['test_date']}: AUC={fold['auc']:.3f} Acc={fold['accuracy']:.3f} "
                  f"Top10%={top} Bot10%={bot} (N={fold['test_size']})")

    print(f"\n  Top Features:")
    for name, imp in metadata.get('top_features', []):
        bar = "█" * int(imp * 200)
        print(f"    {name:25s} {imp:.4f} {bar}")

    # Confidence gate
    auc = metadata.get('cv_avg_auc')
    if auc is not None:
        if auc >= 0.65:
            print(f"\n  ✓ Model is STRONG (AUC {auc:.3f} >= 0.65)")
        elif auc >= 0.58:
            print(f"\n  ~ Model is USABLE (AUC {auc:.3f} >= 0.58)")
        else:
            print(f"\n  ✗ Model is WEAK (AUC {auc:.3f} < 0.58) — constructor_xgb should be disabled")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    # Historical enabled by default (--no-historical to disable)
    no_historical = '--no-historical' in sys.argv
    use_historical = not no_historical

    if command == 'train':
        use_precomputed = '--precomputed' in sys.argv or '--all' in sys.argv
        precomputed_path = os.path.join(PREDICTIONS_DIR, 'cache', 'all_features.npz')

        if use_precomputed and os.path.exists(precomputed_path):
            print("=" * 60)
            print(f"  XGBoost Prop Classifier — Training on ALL pre-computed data")
            print("=" * 60)

            data = np.load(precomputed_path, allow_pickle=True)
            X = data['X']
            y = data['y']
            dates = data['dates'].astype(str)
            sources_arr = data['sources'].astype(str)

            print(f"  Loaded: {X.shape[0]:,} samples × {X.shape[1]} features")
            print(f"  Hit rate: {y.mean():.1%}")
            for src in sorted(set(sources_arr)):
                mask = sources_arr == src
                print(f"    {src:15s}: {mask.sum():>10,}")

            # Build sample weights from source tags
            weight_map = {'graded': 25.0, 'backfill': 10.0, 'sgo_backfill': 8.0, 'csv_full': 1.0}
            sample_weights = np.array([weight_map.get(s, 1.0) for s in sources_arr])
            sources = list(sources_arr)

        else:
            print("=" * 60)
            print(f"  XGBoost Prop Classifier — Training {'(+historical)' if use_historical else '(graded only)'}")
            print("=" * 60)

            if use_historical:
                records = collect_all_training_data(use_historical=True)
            else:
                records = collect_training_data()

            if len(records) < 100:
                print(f"  ERROR: Only {len(records)} records — need at least 100 for training")
                sys.exit(1)

            X, y, dates = engineer_features(records)

            # Build sample weights
            sample_weights = None
            if use_historical or any(r.get('_data_source') in ('backfill', 'sgo_backfill') for r in records):
                def _weight(r):
                    src = r.get('_data_source', '')
                    if src == 'graded': return 25.0
                    if src == 'backfill': return 10.0
                    if src == 'sgo_backfill': return 8.0
                    return 1.0
                sample_weights = np.array([_weight(r) for r in records])

            sources = [r.get('_data_source', 'graded') for r in records]
        model, metadata = train_model(X, y, dates, sample_weights=sample_weights, sources=sources)
        metadata['use_historical'] = use_historical

        # Count sources — from records if available, else from sources list
        if 'records' in dir() and records:
            n_graded = sum(1 for r in records if r.get('_data_source') == 'graded')
            n_hist = sum(1 for r in records if r.get('_data_source') == 'historical')
            n_backfill = sum(1 for r in records if r.get('_data_source') == 'backfill')
            n_sgo_backfill = sum(1 for r in records if r.get('_data_source') == 'sgo_backfill')
        else:
            # Precomputed path — count from sources list
            n_graded = sum(1 for s in sources if s == 'graded')
            n_hist = sum(1 for s in sources if s in ('historical', 'historical_10yr', 'csv_full'))
            n_backfill = sum(1 for s in sources if s == 'backfill')
            n_sgo_backfill = sum(1 for s in sources if s == 'sgo_backfill')
        metadata['n_graded'] = n_graded
        metadata['n_backfill'] = n_backfill
        metadata['n_sgo_backfill'] = n_sgo_backfill
        metadata['n_historical'] = n_hist
        # Re-save metadata with historical info
        meta_path = MODEL_PATH.replace('.json', '_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print_report(metadata)

    elif command == 'eval':
        print("=" * 60)
        print(f"  XGBoost Prop Classifier — Evaluation {'(+historical)' if use_historical else '(graded only)'}")
        print("=" * 60)

        if use_historical:
            records = collect_all_training_data(use_historical=True)
        else:
            records = collect_training_data()

        X, y, dates = engineer_features(records)

        sample_weights = None
        if use_historical or any(r.get('_data_source') in ('backfill', 'sgo_backfill') for r in records):
            def _weight(r):
                src = r.get('_data_source', '')
                if src == 'graded': return 25.0
                if src == 'backfill': return 10.0
                if src == 'sgo_backfill': return 8.0
                return 1.0
            sample_weights = np.array([_weight(r) for r in records])

        sources = [r.get('_data_source', 'graded') for r in records]
        print("\n  Walk-Forward CV:")
        folds = walk_forward_cv(X, y, dates, sample_weights=sample_weights, sources=sources)

        if folds:
            avg_auc = np.mean([f['auc'] for f in folds])
            avg_acc = np.mean([f['accuracy'] for f in folds])
            print(f"\n  Summary: Avg AUC={avg_auc:.3f}, Avg Acc={avg_acc:.3f}")

    elif command == 'score':
        if len(sys.argv) < 3:
            print("Usage: python3 xgb_model.py score <board_or_results.json>")
            sys.exit(1)

        filepath = sys.argv[2]
        print(f"  Scoring: {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        if isinstance(data, list):
            results = data
        elif isinstance(data, dict) and 'results' in data:
            results = data['results']
        else:
            print("  ERROR: Unrecognized file format")
            sys.exit(1)

        results = score_props(results)
        scored = sum(1 for r in results if 'xgb_prob' in r)
        print(f"  Scored {scored}/{len(results)} props")

        # Print top/bottom 10
        with_prob = [r for r in results if 'xgb_prob' in r]
        with_prob.sort(key=lambda r: r['xgb_prob'], reverse=True)

        print(f"\n  Top 10 (highest xgb_prob):")
        for r in with_prob[:10]:
            print(f"    {r['xgb_prob']:.3f}  {r.get('tier','?'):1s}  "
                  f"{r['player']:20s} {r.get('stat',''):5s} "
                  f"{r.get('direction',''):5s} {r.get('line',0):.1f}  "
                  f"L10 HR={r.get('l10_hit_rate',0)}%")

        print(f"\n  Bottom 10 (lowest xgb_prob):")
        for r in with_prob[-10:]:
            print(f"    {r['xgb_prob']:.3f}  {r.get('tier','?'):1s}  "
                  f"{r['player']:20s} {r.get('stat',''):5s} "
                  f"{r.get('direction',''):5s} {r.get('line',0):.1f}  "
                  f"L10 HR={r.get('l10_hit_rate',0)}%")

    elif command == 'importance':
        if not os.path.exists(META_PATH):
            print("  No model metadata found. Run: python3 xgb_model.py train")
            sys.exit(1)

        with open(META_PATH) as f:
            metadata = json.load(f)
        print_report(metadata)

    elif command == 'historical-stats':
        print("=" * 60)
        print("  Historical Data Summary")
        print("=" * 60)

        records = collect_historical_data()
        print(f"\n  Total records: {len(records)}")

        # By season
        from collections import Counter
        season_counts = Counter()
        stat_counts = Counter()
        dir_counts = Counter()
        hit_by_stat = {}

        for r in records:
            d = r.get('_date', '')
            year = int(d[:4]) if d else 0
            month = int(d[5:7]) if d else 0
            season = f"{year}-{year+1}" if month >= 10 else f"{year-1}-{year}"
            season_counts[season] += 1
            stat_counts[r.get('stat', '')] += 1
            dir_counts[r.get('direction', '')] += 1
            stat = r.get('stat', '')
            hit_by_stat.setdefault(stat, []).append(1 if r.get('_hit_label') else 0)

        print(f"\n  By season:")
        for s in sorted(season_counts):
            print(f"    {s}: {season_counts[s]:,}")

        print(f"\n  By stat:")
        for s in sorted(stat_counts):
            hr = np.mean(hit_by_stat[s]) * 100 if s in hit_by_stat else 0
            print(f"    {s:5s}: {stat_counts[s]:>7,}  (hit rate: {hr:.1f}%)")

        print(f"\n  By direction:")
        for d in sorted(dir_counts):
            print(f"    {d}: {dir_counts[d]:,}")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()

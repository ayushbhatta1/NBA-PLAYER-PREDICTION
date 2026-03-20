#!/usr/bin/env python3
"""
XGBoost Regression Model -- Predicts actual stat values (not hit/miss).

Instead of P(OVER), predicts the actual stat value. This gives:
  - "Player will score 25 +/- 3" rather than "55% OVER 22.5"
  - Derived OVER/UNDER probability from predicted distribution
  - Margin estimates: HOW MUCH a pick will clear the line (or miss by)
  - Better parlay selection via large predicted margins

Uses same FEATURE_COLS (92 features) as classification XGBoost.
Target: actual stat value (float) instead of binary hit/miss.

Usage:
    python3 predictions/regression_model.py train          # Train and evaluate
    python3 predictions/regression_model.py score <file>   # Score a board file
    python3 predictions/regression_model.py compare        # Compare vs classification probs
"""

import json
import os
import sys
import glob
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
except ImportError:
    print("ERROR: xgboost not installed. Run: pip3 install xgboost")
    sys.exit(1)

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed. Derived OVER/UNDER probs will use fallback.")

# ===============================================================
# CONSTANTS
# ===============================================================

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'regression_model.pkl')
META_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'regression_model_meta.json')

# Import shared feature engineering from xgb_model
sys.path.insert(0, PREDICTIONS_DIR)
from xgb_model import (
    FEATURE_COLS, COMBO_STATS, STAT_ORDINAL, TIER_ORDINAL, STREAK_ORDINAL,
    engineer_features, _safe_float,
    collect_training_data, collect_backfill_data, collect_sgo_backfill_data,
    collect_10yr_data, collect_historical_data,
)

# Stat type grouping for per-stat residual std estimation
STAT_GROUPS = {
    'pts': 'pts',
    'reb': 'reb',
    'ast': 'ast',
    '3pm': '3pm',
    'stl': 'stl',
    'blk': 'blk',
    'pra': 'combo',
    'pr': 'combo',
    'pa': 'combo',
    'ra': 'combo',
}


# ===============================================================
# DATA COLLECTION (regression-specific: needs actual stat values)
# ===============================================================

def _extract_actual_value(record):
    """Extract actual stat value from a record. Returns float or None."""
    actual = record.get('actual')
    if actual is None:
        return None
    try:
        val = float(actual)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except (ValueError, TypeError):
        return None


def collect_regression_data():
    """Collect records that have actual stat values for regression training.

    Sources with actual values:
      - Graded records (predictions/YYYY-MM-DD/graded*.json) -- 'actual' field
      - 10yr historical CSV reconstruction -- 'actual' field
    Sources WITHOUT actual values (excluded):
      - nba_api backfill -- only has _hit_label, no actual stored
      - SGO backfill -- only has _hit_label, no actual stored
      - Legacy historical synthetic -- only has _hit_label, no actual stored
    """
    all_records = []
    source_counts = defaultdict(int)

    # 1. Graded records -- highest quality, have actual from box scores
    print("  Loading graded records...")
    graded = collect_training_data()
    graded_with_actual = []
    for r in graded:
        actual = _extract_actual_value(r)
        if actual is not None:
            r['_actual_value'] = actual
            r['_data_source'] = 'graded'
            graded_with_actual.append(r)
    print(f"    Graded: {len(graded_with_actual)}/{len(graded)} have actual values")
    source_counts['graded'] = len(graded_with_actual)
    all_records.extend(graded_with_actual)

    # 2. 10yr historical CSV -- has actual field from reconstructed box scores
    print("  Loading 10yr historical data...")
    hist_10yr = collect_10yr_data(sample_cap=50000)
    hist_with_actual = []
    for r in hist_10yr:
        actual = _extract_actual_value(r)
        if actual is not None:
            r['_actual_value'] = actual
            r['_data_source'] = 'historical_10yr'
            hist_with_actual.append(r)
    print(f"    10yr historical: {len(hist_with_actual)}/{len(hist_10yr)} have actual values")
    source_counts['historical_10yr'] = len(hist_with_actual)
    all_records.extend(hist_with_actual)

    # 3. nba_api backfill -- reconstruct actual from l10_values context
    # The backfill generates OVER/UNDER pairs from the same game.
    # For the OVER record at game index i, the actual value was the real box score.
    # We can approximate: if _hit_label=True and direction=OVER, actual > line
    # But we don't know the exact value. Skip these -- not reliable for regression.

    # 4. SGO backfill -- same issue, no actual stored
    # Skip.

    print(f"\n  Regression data summary:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src}: {cnt:,}")
    print(f"    TOTAL: {len(all_records):,}")

    return all_records


def engineer_regression_features(records):
    """Convert records to (X, y_actual, dates, stats) for regression.

    Same feature matrix as classification, but y is actual stat value (float)
    instead of binary hit/miss.

    Returns:
        X: feature matrix (n_samples, n_features)
        y: actual stat values (n_samples,)
        dates: list of date strings
        stats: list of stat type strings (for per-stat residual estimation)
    """
    # We need _hit_label set for engineer_features (it extracts y but we ignore it)
    for r in records:
        if '_hit_label' not in r:
            # Set a dummy -- we won't use the classification label
            actual = r.get('_actual_value', 0)
            line = r.get('line', 0)
            direction = r.get('direction', 'OVER')
            if direction == 'OVER':
                r['_hit_label'] = actual > line
            else:
                r['_hit_label'] = actual < line

    # Use shared engineer_features to build X
    X, _y_binary, dates = engineer_features(records)

    # Build regression target: actual stat value
    y_actual = np.array([r['_actual_value'] for r in records], dtype=np.float64)
    stats = [r.get('stat', 'pts') for r in records]

    print(f"  Regression target stats:")
    print(f"    Mean actual: {y_actual.mean():.2f}")
    print(f"    Median actual: {np.median(y_actual):.2f}")
    print(f"    Std actual: {y_actual.std():.2f}")
    print(f"    Range: [{y_actual.min():.1f}, {y_actual.max():.1f}]")

    return X, y_actual, dates, stats


# ===============================================================
# MODEL TRAINING
# ===============================================================

def _get_regression_params():
    """Return XGBRegressor hyperparameters.

    Same structure as classifier but with reg:squarederror objective.
    No scale_pos_weight (not applicable to regression).
    No monotonic constraints tied to hit probability.
    """
    return {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 5,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'colsample_bylevel': 0.8,
        'learning_rate': 0.05,
        'n_estimators': 800,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'gamma': 0.3,
        'random_state': 42,
        'verbosity': 0,
    }


def walk_forward_cv_regression(X, y, dates, stats, sample_weights=None, sources=None):
    """Walk-forward CV for regression. Train on earlier days, test on next day.

    Returns per-fold metrics: MAE, RMSE, R-squared, plus per-stat residual std.
    """
    dates_arr = np.array(dates)
    sources_arr = np.array(sources) if sources is not None else np.array(['graded'] * len(dates))
    stats_arr = np.array(stats)

    # Only graded records define test folds
    graded_mask = sources_arr == 'graded'
    graded_dates = sorted(set(d for d, s in zip(dates, sources_arr) if s == 'graded' and d >= '2026-'))
    # Historical data always in training
    historical_mask = np.array([s in ('historical_10yr', 'historical') for s in sources_arr])

    if len(graded_dates) < 2:
        print("  WARNING: Need at least 2 graded days for walk-forward CV")
        return [], {}

    folds = []
    all_residuals = defaultdict(list)
    all_y_test = []
    all_y_pred = []

    for i in range(1, len(graded_dates)):
        train_graded_dates = set(graded_dates[:i])
        test_date = graded_dates[i]

        # Train = historical (always) + graded before test date
        train_graded_mask = np.array([d in train_graded_dates and s == 'graded'
                                       for d, s in zip(dates, sources_arr)])
        train_mask = historical_mask | train_graded_mask
        test_mask = (dates_arr == test_date) & graded_mask

        if train_mask.sum() < 50 or test_mask.sum() < 5:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        sw_train = sample_weights[train_mask] if sample_weights is not None else None
        test_stats = stats_arr[test_mask]

        params = _get_regression_params()
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test)

        # Metrics
        residuals = y_test - y_pred
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Per-stat residuals for this fold
        for stat_type, resid in zip(test_stats, residuals):
            group = STAT_GROUPS.get(stat_type, stat_type)
            all_residuals[group].append(float(resid))

        all_y_test.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        fold = {
            'train_dates': sorted(train_graded_dates),
            'test_date': test_date,
            'train_size': int(train_mask.sum()),
            'test_size': int(test_mask.sum()),
            'mae': round(float(mae), 3),
            'rmse': round(float(rmse), 3),
            'r2': round(float(r2), 4),
        }
        folds.append(fold)

        print(f"    Fold: train {sorted(train_graded_dates)} -> test {test_date}")
        print(f"      N={fold['test_size']}, MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")

    # Pooled metrics
    if len(all_y_test) >= 10:
        all_y_test = np.array(all_y_test)
        all_y_pred = np.array(all_y_pred)
        pooled_residuals = all_y_test - all_y_pred
        pooled_mae = np.mean(np.abs(pooled_residuals))
        pooled_rmse = np.sqrt(np.mean(pooled_residuals ** 2))
        ss_res = np.sum(pooled_residuals ** 2)
        ss_tot = np.sum((all_y_test - all_y_test.mean()) ** 2)
        pooled_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        print(f"\n    Pooled CV: MAE={pooled_mae:.2f}, RMSE={pooled_rmse:.2f}, R2={pooled_r2:.4f}")
        print(f"    Pooled N={len(all_y_test)}")

    # Per-stat residual std (used for derived OVER/UNDER probability)
    residual_stds = {}
    print(f"\n    Per-stat residual std (for probability derivation):")
    for group, resids in sorted(all_residuals.items()):
        resids = np.array(resids)
        std = float(np.std(resids))
        bias = float(np.mean(resids))
        residual_stds[group] = {'std': round(std, 3), 'bias': round(bias, 3), 'n': len(resids)}
        print(f"      {group:6s}: std={std:.2f}, bias={bias:+.2f}, n={len(resids)}")

    # Global residual std as fallback
    all_r = [r for resids in all_residuals.values() for r in resids]
    if all_r:
        global_std = float(np.std(all_r))
        residual_stds['_global'] = {'std': round(global_std, 3), 'bias': 0.0, 'n': len(all_r)}
        print(f"      {'GLOBAL':6s}: std={global_std:.2f}, n={len(all_r)}")

    return folds, residual_stds


def train_regression(records):
    """Train XGBRegressor on records with actual values. Run CV and save model.

    Returns (model, metadata).
    """
    print(f"\n  Engineering features for {len(records)} records...")
    X, y, dates, stats = engineer_regression_features(records)

    # Build sample weights: graded=25.0, historical=1.0
    sources = [r.get('_data_source', 'graded') for r in records]
    sample_weights = np.array([25.0 if s == 'graded' else 1.0 for s in sources])

    # Walk-forward CV
    print("\n  Walk-Forward Cross-Validation (Regression):")
    folds, residual_stds = walk_forward_cv_regression(
        X, y, dates, stats,
        sample_weights=sample_weights,
        sources=sources,
    )

    cv_mae = np.mean([f['mae'] for f in folds]) if folds else None
    cv_rmse = np.mean([f['rmse'] for f in folds]) if folds else None
    cv_r2 = np.mean([f['r2'] for f in folds]) if folds else None

    if folds:
        print(f"\n  CV Summary: Avg MAE={cv_mae:.2f}, Avg RMSE={cv_rmse:.2f}, Avg R2={cv_r2:.4f}")

    # Train final model on all data
    print(f"\n  Training final regression model on {len(y)} samples...")
    params = _get_regression_params()

    # Use last day as eval set for early stopping
    unique_dates = sorted(set(dates))
    dates_arr = np.array(dates)
    best_n = params['n_estimators']

    if len(unique_dates) >= 2:
        last_date = unique_dates[-1]
        train_mask = dates_arr != last_date
        eval_mask = dates_arr == last_date
        sw_train = sample_weights[train_mask]

        probe_model = XGBRegressor(**params)
        probe_model.fit(
            X[train_mask], y[train_mask],
            sample_weight=sw_train,
            eval_set=[(X[eval_mask], y[eval_mask])],
            verbose=False,
        )
        if hasattr(probe_model, 'best_iteration') and probe_model.best_iteration:
            best_n = min(probe_model.best_iteration + 20, params['n_estimators'])

    params['n_estimators'] = best_n
    model = XGBRegressor(**params)
    model.fit(X, y, sample_weight=sample_weights, verbose=False)

    # Compute final training residual stds per stat (on full data)
    y_pred_train = model.predict(X)
    train_residuals = y - y_pred_train
    final_residual_stds = {}
    stat_residuals = defaultdict(list)
    for stat_type, resid in zip(stats, train_residuals):
        group = STAT_GROUPS.get(stat_type, stat_type)
        stat_residuals[group].append(float(resid))
    for group, resids in stat_residuals.items():
        resids = np.array(resids)
        final_residual_stds[group] = {
            'std': round(float(np.std(resids)), 3),
            'bias': round(float(np.mean(resids)), 3),
            'n': len(resids),
        }
    all_resids = train_residuals
    final_residual_stds['_global'] = {
        'std': round(float(np.std(all_resids)), 3),
        'bias': round(float(np.mean(all_resids)), 3),
        'n': len(all_resids),
    }

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    # Save model via pickle (XGBRegressor supports save_model for .json but pickle
    # is simpler for bundling with metadata)
    import pickle
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved: {MODEL_PATH}")

    # Metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'model_type': 'XGBRegressor',
        'objective': 'reg:squarederror',
        'n_samples': int(len(y)),
        'n_features': int(X.shape[1]),
        'target_mean': round(float(y.mean()), 2),
        'target_std': round(float(y.std()), 2),
        'unique_dates': sorted(set(dates)),
        'cv_folds': folds,
        'cv_avg_mae': round(float(cv_mae), 3) if cv_mae is not None else None,
        'cv_avg_rmse': round(float(cv_rmse), 3) if cv_rmse is not None else None,
        'cv_avg_r2': round(float(cv_r2), 4) if cv_r2 is not None else None,
        'residual_stds_cv': residual_stds,
        'residual_stds_train': final_residual_stds,
        'top_features': [(name, round(float(imp), 5)) for name, imp in top_features],
        'model_path': MODEL_PATH,
        'n_estimators_used': best_n,
        'source_counts': dict(defaultdict(int, {
            src: sum(1 for s in sources if s == src) for src in set(sources)
        })),
    }

    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {META_PATH}")

    return model, metadata


# ===============================================================
# SCORING
# ===============================================================

def _load_regression_model():
    """Load trained regression model and metadata. Returns (model, meta) or (None, None)."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    if not os.path.exists(META_PATH):
        return None, None

    import pickle
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(META_PATH) as f:
        meta = json.load(f)

    return model, meta


def _get_residual_std(stat_type, meta):
    """Get residual std for a stat type from model metadata.

    Prefers CV residual stds (honest out-of-sample error) for probability derivation.
    Falls back to training residuals if CV not available, then global, then hardcoded.
    """
    group = STAT_GROUPS.get(stat_type, stat_type)

    # Prefer CV residuals -- honest out-of-sample estimate
    cv_stds = meta.get('residual_stds_cv', {})
    if group in cv_stds and cv_stds[group].get('n', 0) >= 10:
        return cv_stds[group]['std']

    # Fall back to training residuals (more data but overfitted)
    train_stds = meta.get('residual_stds_train', {})
    if group in train_stds:
        return train_stds[group]['std']

    # Global fallbacks
    if '_global' in cv_stds and cv_stds['_global'].get('n', 0) >= 10:
        return cv_stds['_global']['std']
    if '_global' in train_stds:
        return train_stds['_global']['std']

    # Last resort hardcoded defaults by stat scale
    defaults = {'pts': 8.0, 'reb': 3.5, 'ast': 3.0, '3pm': 1.8, 'stl': 1.3, 'blk': 1.4,
                'combo': 10.0}
    return defaults.get(group, 5.0)


def _compute_over_prob(predicted, line, residual_std):
    """Compute P(actual > line) given predicted value and residual std.

    Assumes prediction errors are approximately normal.
    Uses scipy.stats.norm if available, otherwise a simple sigmoid approximation.
    """
    if residual_std <= 0:
        # Degenerate case: point prediction
        return 1.0 if predicted > line else 0.0

    if HAS_SCIPY:
        # P(actual > line) = 1 - CDF(line, loc=predicted, scale=residual_std)
        prob = 1.0 - norm.cdf(line, loc=predicted, scale=residual_std)
    else:
        # Sigmoid approximation: P(X > line) ~ sigmoid((predicted - line) / std * 1.7)
        z = (predicted - line) / residual_std
        prob = 1.0 / (1.0 + np.exp(-z * 1.7))

    return float(np.clip(prob, 0.001, 0.999))


def score_regression(results, model_path=None):
    """Score props with regression model. Adds predicted stat value and derived metrics.

    Sets on each prop dict:
        reg_predicted: predicted stat value
        reg_margin: predicted - line (positive = favors OVER)
        reg_confidence: |margin| / line (normalized distance from line)
        reg_over_prob: P(actual > line) from normal distribution
        reg_under_prob: P(actual < line)
        reg_residual_std: residual std used for this stat type

    Returns results with regression fields added. Graceful no-op if model not trained.
    """
    model, meta = _load_regression_model()
    if model is None:
        print("  WARNING: No regression model found. Run: python3 regression_model.py train")
        for r in results:
            r['reg_predicted'] = None
            r['reg_margin'] = None
            r['reg_confidence'] = None
            r['reg_over_prob'] = None
            r['reg_under_prob'] = None
        return results

    # Build feature matrix
    temp_records = []
    for r in results:
        rec = dict(r)
        rec['_hit_label'] = False  # dummy
        rec['_date'] = rec.get('_date', '')
        rec['_actual_value'] = 0.0  # dummy
        temp_records.append(rec)

    if not temp_records:
        return results

    X, _y_binary, _dates = engineer_features(temp_records)

    # Predict
    y_pred = model.predict(X)

    scored = 0
    for r, pred in zip(results, y_pred):
        predicted = round(float(pred), 2)
        line = float(r.get('line', 0))
        stat_type = r.get('stat', 'pts')
        residual_std = _get_residual_std(stat_type, meta)

        margin = round(predicted - line, 2)
        confidence = round(abs(margin) / line, 4) if line > 0 else 0.0
        over_prob = round(_compute_over_prob(predicted, line, residual_std), 4)
        under_prob = round(1.0 - over_prob, 4)

        r['reg_predicted'] = predicted
        r['reg_margin'] = margin
        r['reg_confidence'] = confidence
        r['reg_over_prob'] = over_prob
        r['reg_under_prob'] = under_prob
        r['reg_residual_std'] = round(residual_std, 3)
        scored += 1

    print(f"  Regression scored {scored}/{len(results)} props")
    return results


# ===============================================================
# COMPARISON: regression-derived probs vs classification probs
# ===============================================================

def compare_probs():
    """Compare regression-derived OVER probs vs XGBoost classification probs on graded data.

    For each graded record with an actual value:
      - Compute regression-derived P(OVER) from predicted value + residual std
      - Compare against classification xgb_prob (if available)
      - Evaluate which probability better predicts actual outcomes
    """
    print("=" * 60)
    print("  Regression vs Classification Probability Comparison")
    print("=" * 60)

    # Load graded records
    graded_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, '*', '*graded*.json')))
    all_records = []
    for fpath in graded_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        if isinstance(data, dict) and 'results' in data:
            records = data['results']
        elif isinstance(data, list):
            records = data
        else:
            continue
        for r in records:
            if not isinstance(r, dict):
                continue
            actual = _extract_actual_value(r)
            if actual is None:
                continue
            r['_actual_value'] = actual
            all_records.append(r)

    if not all_records:
        print("  No graded records with actual values found.")
        return

    print(f"\n  Graded records with actuals: {len(all_records)}")

    # Score with regression model
    model, meta = _load_regression_model()
    if model is None:
        print("  No regression model found. Run: python3 regression_model.py train")
        return

    # Build features and predict
    temp_records = []
    for r in all_records:
        rec = dict(r)
        rec['_hit_label'] = False
        rec['_date'] = rec.get('_date', '')
        temp_records.append(rec)

    X, _y_binary, _dates = engineer_features(temp_records)
    y_pred = model.predict(X)

    # Compare
    reg_correct = 0
    cls_correct = 0
    both_have = 0
    reg_only = 0

    # Binned accuracy for regression-derived probs
    reg_bins = defaultdict(lambda: {'correct': 0, 'total': 0})
    cls_bins = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Per-stat metrics
    stat_metrics = defaultdict(lambda: {
        'mae': [], 'reg_correct': 0, 'cls_correct': 0, 'total': 0,
        'margin_when_correct': [], 'margin_when_wrong': [],
    })

    for r, pred in zip(all_records, y_pred):
        actual = r['_actual_value']
        line = float(r.get('line', 0))
        direction = r.get('direction', 'OVER')
        stat_type = r.get('stat', 'pts')
        residual_std = _get_residual_std(stat_type, meta)

        # Did the actual outcome hit?
        if direction == 'OVER':
            actual_hit = actual > line
        else:
            actual_hit = actual < line

        # Regression prediction
        margin = pred - line
        reg_over_prob = _compute_over_prob(pred, line, residual_std)

        if direction == 'OVER':
            reg_says_hit = reg_over_prob > 0.5
        else:
            reg_says_hit = reg_over_prob < 0.5

        reg_is_correct = reg_says_hit == actual_hit

        # MAE
        prediction_error = abs(pred - actual)
        stat_metrics[stat_type]['mae'].append(prediction_error)
        stat_metrics[stat_type]['total'] += 1

        if reg_is_correct:
            reg_correct += 1
            stat_metrics[stat_type]['reg_correct'] += 1
            stat_metrics[stat_type]['margin_when_correct'].append(abs(margin))
        else:
            stat_metrics[stat_type]['margin_when_wrong'].append(abs(margin))

        # Bin regression-derived prob
        prob_for_dir = reg_over_prob if direction == 'OVER' else (1 - reg_over_prob)
        bin_key = round(prob_for_dir * 10) / 10  # 0.0, 0.1, ..., 1.0
        bin_key = min(0.9, max(0.0, bin_key))
        reg_bins[bin_key]['total'] += 1
        if actual_hit:
            reg_bins[bin_key]['correct'] += 1

        # Classification comparison
        xgb_prob = r.get('xgb_prob')
        if xgb_prob is not None:
            both_have += 1
            # For classification, xgb_prob > 0.5 means "hit"
            cls_says_hit = xgb_prob > 0.5
            if cls_says_hit == actual_hit:
                cls_correct += 1
                stat_metrics[stat_type]['cls_correct'] += 1

            # Bin classification prob
            cls_bin = round(xgb_prob * 10) / 10
            cls_bin = min(0.9, max(0.0, cls_bin))
            cls_bins[cls_bin]['total'] += 1
            if actual_hit:
                cls_bins[cls_bin]['correct'] += 1
        else:
            reg_only += 1

    total = len(all_records)
    print(f"\n  Overall accuracy:")
    print(f"    Regression-derived:  {reg_correct}/{total} = {reg_correct/total:.1%}")
    if both_have > 0:
        print(f"    Classification:      {cls_correct}/{both_have} = {cls_correct/both_have:.1%}")
    print(f"    Records with both:   {both_have}")

    # Per-stat breakdown
    print(f"\n  Per-stat regression performance:")
    print(f"    {'Stat':6s} {'MAE':>6s} {'Reg Acc':>8s} {'Cls Acc':>8s} {'N':>5s} {'Avg|Margin|':>12s}")
    for stat_type in sorted(stat_metrics.keys()):
        m = stat_metrics[stat_type]
        mae = np.mean(m['mae']) if m['mae'] else 0
        reg_acc = m['reg_correct'] / m['total'] if m['total'] > 0 else 0
        cls_acc = m['cls_correct'] / m['total'] if m['total'] > 0 else 0
        all_margins = m['margin_when_correct'] + m['margin_when_wrong']
        avg_margin = np.mean(all_margins) if all_margins else 0
        print(f"    {stat_type:6s} {mae:6.2f} {reg_acc:8.1%} {cls_acc:8.1%} {m['total']:5d} {avg_margin:12.2f}")

    # Calibration: binned accuracy
    print(f"\n  Regression-derived probability calibration:")
    print(f"    {'Bin':>5s} {'Actual HR':>10s} {'N':>6s}")
    for bin_key in sorted(reg_bins.keys()):
        b = reg_bins[bin_key]
        if b['total'] >= 3:
            hr = b['correct'] / b['total']
            print(f"    {bin_key:.1f}   {hr:10.1%} {b['total']:6d}")

    if cls_bins:
        print(f"\n  Classification probability calibration:")
        print(f"    {'Bin':>5s} {'Actual HR':>10s} {'N':>6s}")
        for bin_key in sorted(cls_bins.keys()):
            b = cls_bins[bin_key]
            if b['total'] >= 3:
                hr = b['correct'] / b['total']
                print(f"    {bin_key:.1f}   {hr:10.1%} {b['total']:6d}")

    # Margin analysis: do large regression margins predict better?
    print(f"\n  Margin analysis (does |reg_margin| predict accuracy?):")
    margin_bins = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r, pred in zip(all_records, y_pred):
        actual = r['_actual_value']
        line = float(r.get('line', 0))
        direction = r.get('direction', 'OVER')
        margin = abs(pred - line)

        if direction == 'OVER':
            actual_hit = actual > line
            pred_hit = pred > line
        else:
            actual_hit = actual < line
            pred_hit = pred < line

        # Bin by margin
        if margin < 1:
            mbin = '<1'
        elif margin < 2:
            mbin = '1-2'
        elif margin < 3:
            mbin = '2-3'
        elif margin < 5:
            mbin = '3-5'
        else:
            mbin = '5+'

        margin_bins[mbin]['total'] += 1
        if pred_hit == actual_hit:
            margin_bins[mbin]['correct'] += 1

    print(f"    {'Margin':>8s} {'Accuracy':>10s} {'N':>6s}")
    for mbin in ['<1', '1-2', '2-3', '3-5', '5+']:
        if mbin in margin_bins and margin_bins[mbin]['total'] >= 3:
            b = margin_bins[mbin]
            acc = b['correct'] / b['total']
            print(f"    {mbin:>8s} {acc:10.1%} {b['total']:6d}")


# ===============================================================
# REPORTING
# ===============================================================

def print_report(metadata):
    """Print formatted regression model report."""
    print("\n" + "=" * 60)
    print("  Regression Model Report")
    print("=" * 60)

    print(f"\n  Training data: {metadata['n_samples']:,} samples")
    print(f"  Target: actual stat value (mean={metadata['target_mean']:.2f}, std={metadata['target_std']:.2f})")

    src = metadata.get('source_counts', {})
    if src:
        print(f"  Sources: {', '.join(f'{k}={v:,}' for k, v in sorted(src.items()))}")

    if metadata.get('cv_avg_mae') is not None:
        print(f"\n  Walk-Forward CV:")
        print(f"    Average MAE:  {metadata['cv_avg_mae']:.3f}")
        print(f"    Average RMSE: {metadata['cv_avg_rmse']:.3f}")
        print(f"    Average R2:   {metadata['cv_avg_r2']:.4f}")

        for fold in metadata.get('cv_folds', []):
            print(f"    {fold['test_date']}: MAE={fold['mae']:.2f} RMSE={fold['rmse']:.2f} "
                  f"R2={fold['r2']:.4f} (N={fold['test_size']})")

    # Per-stat residual std
    stds = metadata.get('residual_stds_cv', metadata.get('residual_stds_train', {}))
    if stds:
        print(f"\n  Residual Std by Stat (for probability derivation):")
        for group in sorted(stds.keys()):
            if group.startswith('_'):
                continue
            s = stds[group]
            print(f"    {group:6s}: std={s['std']:.2f}, bias={s['bias']:+.2f} (n={s['n']})")
        if '_global' in stds:
            g = stds['_global']
            print(f"    {'GLOBAL':6s}: std={g['std']:.2f} (n={g['n']})")

    print(f"\n  Top Features (importance):")
    for name, imp in metadata.get('top_features', [])[:15]:
        bar = '#' * int(imp * 200)
        print(f"    {name:30s} {imp:.5f} {bar}")


# ===============================================================
# CLI
# ===============================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == 'train':
        print("=" * 60)
        print("  XGBoost Regression Model -- Training")
        print("=" * 60)

        records = collect_regression_data()

        if len(records) < 100:
            print(f"  ERROR: Only {len(records)} records with actual values -- need at least 100")
            sys.exit(1)

        model, metadata = train_regression(records)
        print_report(metadata)

    elif command == 'score':
        if len(sys.argv) < 3:
            print("Usage: python3 regression_model.py score <board_or_results.json>")
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

        results = score_regression(results)

        # Print top predictions by margin
        with_pred = [r for r in results if r.get('reg_predicted') is not None]
        if not with_pred:
            print("  No predictions generated (model not trained?)")
            sys.exit(1)

        # Sort by confidence (absolute margin / line)
        with_pred.sort(key=lambda r: abs(r.get('reg_margin', 0)), reverse=True)

        print(f"\n  Top 15 by predicted margin (largest |predicted - line|):")
        print(f"  {'Player':20s} {'Stat':5s} {'Line':>6s} {'Pred':>6s} {'Margin':>7s} "
              f"{'OvProb':>7s} {'Dir':>5s} {'Tier':>4s}")
        print(f"  {'-'*20} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*5} {'-'*4}")
        for r in with_pred[:15]:
            player = r.get('player', '?')[:20]
            stat = r.get('stat', '?')
            line = r.get('line', 0)
            pred = r.get('reg_predicted', 0)
            margin = r.get('reg_margin', 0)
            over_prob = r.get('reg_over_prob', 0)
            direction = r.get('direction', '?')
            tier = r.get('tier', '?')
            print(f"  {player:20s} {stat:5s} {line:6.1f} {pred:6.1f} {margin:+7.2f} "
                  f"{over_prob:7.1%} {direction:>5s} {tier:>4s}")

        # Print favored picks: where regression strongly agrees with direction
        print(f"\n  Strongest regression signals (margin aligns with direction):")
        favored = []
        for r in with_pred:
            margin = r.get('reg_margin', 0)
            direction = r.get('direction', '')
            if (direction == 'OVER' and margin > 0) or (direction == 'UNDER' and margin < 0):
                favored.append(r)

        favored.sort(key=lambda r: abs(r.get('reg_margin', 0)), reverse=True)
        print(f"  {'Player':20s} {'Stat':5s} {'Line':>6s} {'Pred':>6s} {'Margin':>7s} "
              f"{'DirProb':>8s} {'Dir':>5s}")
        print(f"  {'-'*20} {'-'*5} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*5}")
        for r in favored[:15]:
            player = r.get('player', '?')[:20]
            stat = r.get('stat', '?')
            line = r.get('line', 0)
            pred = r.get('reg_predicted', 0)
            margin = r.get('reg_margin', 0)
            direction = r.get('direction', '?')
            if direction == 'OVER':
                dir_prob = r.get('reg_over_prob', 0)
            else:
                dir_prob = r.get('reg_under_prob', 0)
            print(f"  {player:20s} {stat:5s} {line:6.1f} {pred:6.1f} {margin:+7.2f} "
                  f"{dir_prob:8.1%} {direction:>5s}")

    elif command == 'compare':
        compare_probs()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()

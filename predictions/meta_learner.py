#!/usr/bin/env python3
"""
Meta-Learner — Stacked ensemble that learns optimal blending of sub-model predictions.

Currently: ensemble_prob = 0.6 * xgb_prob + 0.4 * mlp_prob (hardcoded).
Meta-learner: LEARNS the optimal combination from data, including nonlinear interactions.

Inputs (12 meta-features):
    1. xgb_prob          — XGBoost hit probability
    2. mlp_prob          — MLP neural net hit probability
    3. sim_prob          — Monte Carlo simulation probability
    4. xgb_prob * mlp_prob         — agreement interaction
    5. abs(xgb_prob - mlp_prob)    — disagreement signal
    6. max(xgb_prob, mlp_prob)     — optimistic signal
    7. min(xgb_prob, mlp_prob)     — pessimistic signal
    8. (xgb + mlp + sim) / 3      — simple average
    9. tier_ordinal       — tier context (S=5..F=0)
   10. direction_binary   — OVER=1, UNDER=0
   11. is_combo           — combo stat flag
   12. abs_gap            — gap magnitude

Architecture: LogisticRegression + isotonic calibration.
Intentionally simple to avoid overfitting on small graded data (~3-4K records).

Training data: graded daily predictions from predictions/YYYY-MM-DD/v4_graded_*_lines.json.
These are genuine out-of-sample predictions (model was trained before that day's games).

Usage:
    python3 predictions/meta_learner.py train      # Train from graded daily predictions
    python3 predictions/meta_learner.py eval       # Evaluate on held-out days
    python3 predictions/meta_learner.py weights    # Show learned model weights
"""

import json
import os
import sys
import glob
import pickle
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip3 install scikit-learn")
    sys.exit(1)

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'meta_learner.pkl')
META_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'meta_learner_meta.json')

TIER_ORDINAL = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
COMBO_STATS = {'pra', 'pr', 'pa', 'ra'}

META_FEATURE_NAMES = [
    'xgb_prob',
    'mlp_prob',
    'sim_prob',
    'agreement',          # xgb * mlp
    'disagreement',       # |xgb - mlp|
    'optimistic',         # max(xgb, mlp)
    'pessimistic',        # min(xgb, mlp)
    'simple_avg',         # (xgb + mlp + sim) / 3
    'tier_ordinal',
    'direction_binary',
    'is_combo',
    'abs_gap',
]


# ===============================================================
# HELPERS
# ===============================================================

def _safe_float(val, default=0.5):
    """Convert to float, return default for None/invalid."""
    if val is None:
        return default
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (ValueError, TypeError):
        return default


def _extract_hit_label(record):
    """Extract hit/miss label from a graded record. Returns True/False/None."""
    if 'hit' in record and record['hit'] is not None:
        return bool(record['hit'])
    result = record.get('result', '')
    if isinstance(result, str):
        if result.upper() == 'HIT':
            return True
        elif result.upper() == 'MISS':
            return False
    actual = record.get('actual')
    line = record.get('line')
    direction = record.get('direction', '')
    if actual is not None and line is not None and direction:
        try:
            if direction == 'OVER':
                return float(actual) > float(line)
            else:
                return float(actual) < float(line)
        except (ValueError, TypeError):
            pass
    return None


def _reconstruct_sim_prob(record):
    """Reconstruct sim_prob from l10_values when it was not persisted.

    This uses the same logic as sim_model.simulate_player_stat() but
    is a lightweight inline version to avoid circular imports during training.
    Returns 0.5 (neutral) when data is insufficient.
    """
    # If sim_prob was already computed and stored, use it directly
    existing = record.get('sim_prob')
    if existing is not None:
        return _safe_float(existing, 0.5)

    l10_values = record.get('l10_values', [])
    line = record.get('line')
    direction = record.get('direction', '')

    if not l10_values or not line or not direction:
        return 0.5

    try:
        values = np.array([float(v) for v in l10_values if v is not None], dtype=np.float64)
        if len(values) < 3:
            return 0.5
        line_f = float(line)
        if np.isnan(line_f) or line_f <= 0:
            return 0.5

        # Simple simulation: draw from normal distribution fitted to recent data
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
        std = max(std, mean * 0.05 + 0.5)  # floor to prevent degenerate distributions

        np.random.seed(hash((record.get('player', ''), record.get('stat', ''))) % (2**31))
        samples = np.random.normal(mean, std, size=5000)
        np.clip(samples, 0.0, None, out=samples)

        over_count = np.sum(samples > line_f)
        under_count = np.sum(samples < line_f)

        if direction == 'OVER':
            return round(float(over_count) / 5000, 4)
        else:
            return round(float(under_count) / 5000, 4)

    except Exception:
        return 0.5


# ===============================================================
# META-FEATURE ENGINEERING
# ===============================================================

def build_meta_features(record):
    """Build the 12-element meta-feature vector from a single prop dict.

    Handles missing sub-model predictions gracefully:
    - If mlp_prob is missing, uses xgb_prob as stand-in (since they are correlated)
    - If sim_prob is missing, reconstructs from l10_values via lightweight simulation
    - If xgb_prob is missing, returns None (record unusable)
    """
    xgb_p = record.get('xgb_prob')
    if xgb_p is None:
        return None  # cannot build meta-features without at least xgb_prob

    xgb_p = _safe_float(xgb_p, None)
    if xgb_p is None:
        return None

    # MLP: use actual if available, fall back to xgb_prob
    mlp_p = record.get('mlp_prob')
    if mlp_p is not None:
        mlp_p = _safe_float(mlp_p, xgb_p)
    else:
        mlp_p = xgb_p  # correlated stand-in

    # Sim: use actual if available, reconstruct from l10_values, or default 0.5
    sim_p = _reconstruct_sim_prob(record)

    # Derived interaction features
    agreement = xgb_p * mlp_p
    disagreement = abs(xgb_p - mlp_p)
    optimistic = max(xgb_p, mlp_p)
    pessimistic = min(xgb_p, mlp_p)
    simple_avg = (xgb_p + mlp_p + sim_p) / 3.0

    # Context features
    tier = record.get('tier', 'F')
    tier_ord = TIER_ORDINAL.get(tier, 0)

    direction = record.get('direction', '')
    dir_bin = 1 if direction == 'OVER' else 0

    stat = record.get('stat', '')
    is_combo = 1 if stat in COMBO_STATS else 0

    abs_gap = _safe_float(record.get('abs_gap', record.get('gap', 0)), 0.0)
    abs_gap = abs(abs_gap)

    return np.array([
        xgb_p,
        mlp_p,
        sim_p,
        agreement,
        disagreement,
        optimistic,
        pessimistic,
        simple_avg,
        tier_ord,
        dir_bin,
        is_combo,
        abs_gap,
    ], dtype=np.float64)


# ===============================================================
# DATA COLLECTION
# ===============================================================

def collect_graded_records():
    """Scan graded prediction files and collect records with xgb_prob + labels.

    These are genuine out-of-sample predictions: the XGBoost/MLP models were
    trained before each day's games were played.
    """
    graded_files = sorted(glob.glob(
        os.path.join(PREDICTIONS_DIR, '20*', '*graded*.json')
    ))

    all_records = []
    files_used = 0

    for fpath in graded_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict) and 'results' in data:
            records = data['results']
        else:
            continue

        date_dir = os.path.basename(os.path.dirname(fpath))
        file_had_useful = False

        for r in records:
            label = _extract_hit_label(r)
            if label is None:
                continue
            if r.get('xgb_prob') is None:
                continue  # need at least xgb_prob

            r['_hit_label'] = label
            r['_date'] = date_dir
            all_records.append(r)
            file_had_useful = True

        if file_had_useful:
            files_used += 1

    # Deduplicate (same player+stat+line+date)
    seen = set()
    deduped = []
    for r in all_records:
        key = (r.get('player', ''), r.get('stat', ''), r.get('line', 0), r.get('_date', ''))
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    print(f"  Collected {len(deduped)} labeled records with xgb_prob from {files_used} graded files")

    # Summary by date
    date_counts = {}
    for r in deduped:
        d = r['_date']
        date_counts[d] = date_counts.get(d, 0) + 1
    for d in sorted(date_counts):
        print(f"    {d}: {date_counts[d]} records")

    return deduped


def _build_training_data(records):
    """Convert records into X (meta-features), y (labels), dates arrays."""
    X_list = []
    y_list = []
    dates_list = []
    skipped = 0

    for r in records:
        features = build_meta_features(r)
        if features is None:
            skipped += 1
            continue
        X_list.append(features)
        y_list.append(1.0 if r['_hit_label'] else 0.0)
        dates_list.append(r['_date'])

    if skipped > 0:
        print(f"  Skipped {skipped} records (missing xgb_prob)")

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    return X, y, dates_list


# ===============================================================
# AUC + LOGLOSS (inline to avoid import dependency)
# ===============================================================

def _compute_auc(y_true, y_prob):
    """AUC via Mann-Whitney U statistic."""
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    auc = 0.0
    for p in pos:
        auc += (neg < p).sum() + 0.5 * (neg == p).sum()
    return auc / (len(pos) * len(neg))


def _compute_logloss(y_true, y_prob):
    """Binary cross-entropy."""
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


# ===============================================================
# TRAINING
# ===============================================================

def train_meta(records=None):
    """Train the meta-learner from graded daily predictions.

    Uses LogisticRegression with isotonic calibration.
    Walk-forward CV: for each graded date, train on all prior dates, test on that date.
    Final model is trained on all data.
    """
    if records is None:
        records = collect_graded_records()

    if len(records) < 50:
        print(f"  ERROR: Only {len(records)} usable records -- need at least 50")
        return None, None

    X, y, dates = _build_training_data(records)
    print(f"\n  Meta-learner training data: {len(y)} samples, {X.shape[1]} features")
    print(f"  Hit rate: {y.mean():.1%}")
    print(f"  Feature means: {', '.join(f'{n}={v:.3f}' for n, v in zip(META_FEATURE_NAMES, X.mean(axis=0)))}")

    # Walk-forward CV
    unique_dates = sorted(set(dates))
    print(f"\n  Walk-Forward CV ({len(unique_dates)} dates):")

    folds = []
    all_oof_probs = []
    all_oof_actuals = []

    for i in range(1, len(unique_dates)):
        train_dates = set(unique_dates[:i])
        test_date = unique_dates[i]

        dates_arr = np.array(dates)
        train_mask = np.array([d in train_dates for d in dates])
        test_mask = dates_arr == test_date

        if train_mask.sum() < 30 or test_mask.sum() < 5:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train LogisticRegression (simple, low-parameter model)
        lr = LogisticRegression(
            C=1.0,            # regularization strength (1.0 = moderate)
            penalty='l2',     # L2 regularization prevents overfitting
            max_iter=1000,
            solver='lbfgs',
            random_state=42,
        )
        lr.fit(X_train_s, y_train)
        y_prob = lr.predict_proba(X_test_s)[:, 1]

        all_oof_probs.extend(y_prob.tolist())
        all_oof_actuals.extend(y_test.tolist())

        y_pred = (y_prob >= 0.5).astype(int)
        accuracy = (y_pred == y_test).mean()
        auc = _compute_auc(y_test, y_prob)
        logloss = _compute_logloss(y_test, y_prob)

        # Top/bottom decile analysis
        top_mask = y_prob >= np.percentile(y_prob, 90) if len(y_prob) >= 10 else np.ones(len(y_prob), dtype=bool)
        bot_mask = y_prob <= np.percentile(y_prob, 10) if len(y_prob) >= 10 else np.zeros(len(y_prob), dtype=bool)
        top_hr = y_test[top_mask].mean() if top_mask.sum() > 0 else float('nan')
        bot_hr = y_test[bot_mask].mean() if bot_mask.sum() > 0 else float('nan')

        fold = {
            'train_dates': sorted(train_dates),
            'test_date': test_date,
            'train_size': int(train_mask.sum()),
            'test_size': int(test_mask.sum()),
            'accuracy': round(float(accuracy), 4),
            'auc': round(float(auc), 4),
            'logloss': round(float(logloss), 4),
            'top_decile_hr': round(float(top_hr), 4) if not np.isnan(top_hr) else None,
            'bot_decile_hr': round(float(bot_hr), 4) if not np.isnan(bot_hr) else None,
        }
        folds.append(fold)

        top_str = f"{top_hr:.1%}" if not np.isnan(top_hr) else "N/A"
        bot_str = f"{bot_hr:.1%}" if not np.isnan(bot_hr) else "N/A"
        print(f"    {test_date}: N={fold['test_size']}, Acc={accuracy:.3f}, "
              f"AUC={auc:.3f}, LogLoss={logloss:.3f}, Top10%={top_str}, Bot10%={bot_str}")

    # Pooled CV metrics
    pooled_auc = None
    pooled_top10 = None
    pooled_bot10 = None

    if len(all_oof_probs) >= 50:
        all_y = np.array(all_oof_actuals)
        all_p = np.array(all_oof_probs)
        pooled_auc = _compute_auc(all_y, all_p)
        pooled_logloss = _compute_logloss(all_y, all_p)
        pooled_acc = ((all_p >= 0.5).astype(int) == all_y).mean()

        p90 = np.percentile(all_p, 90)
        p10 = np.percentile(all_p, 10)
        pooled_top10 = all_y[all_p >= p90].mean() if (all_p >= p90).sum() > 0 else float('nan')
        pooled_bot10 = all_y[all_p <= p10].mean() if (all_p <= p10).sum() > 0 else float('nan')

        top_str = f"{pooled_top10:.1%}" if not np.isnan(pooled_top10) else "N/A"
        bot_str = f"{pooled_bot10:.1%}" if not np.isnan(pooled_bot10) else "N/A"
        print(f"\n    Pooled CV: AUC={pooled_auc:.3f}, Acc={pooled_acc:.3f}, "
              f"LogLoss={pooled_logloss:.3f}")
        print(f"    Top10%={top_str}, Bot10%={bot_str}, N={len(all_y)}")

    # Compare to hardcoded 0.6/0.4 baseline
    if len(all_oof_probs) >= 50:
        # Baseline: the xgb_prob that was already in the data (column 0)
        xgb_only_auc = _compute_auc(all_y, X[np.isin(dates, [f['test_date'] for f in folds]), 0])
        if len(X[np.isin(dates, [f['test_date'] for f in folds]), 0]) == len(all_y):
            print(f"    Baseline (xgb_prob only): AUC={xgb_only_auc:.3f}")
            improvement = pooled_auc - xgb_only_auc
            print(f"    Meta-learner improvement: {'+' if improvement >= 0 else ''}{improvement:.3f} AUC")

    # Train final model on all data
    print(f"\n  Training final meta-learner on {len(y)} samples...")
    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X)

    # Base model: Logistic Regression
    base_lr = LogisticRegression(
        C=1.0,
        penalty='l2',
        max_iter=1000,
        solver='lbfgs',
        random_state=42,
    )

    # Wrap with isotonic calibration for well-calibrated probabilities
    # cv=3 for small datasets, cv=5 if we have enough data
    cv_folds = 5 if len(y) >= 500 else 3
    calibrated_model = CalibratedClassifierCV(
        estimator=base_lr,
        method='isotonic',
        cv=cv_folds,
    )
    calibrated_model.fit(X_scaled, y)

    # Also train a plain LR to extract weights for interpretability
    plain_lr = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, solver='lbfgs', random_state=42)
    plain_lr.fit(X_scaled, y)

    # Save model bundle
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bundle = {
        'model': calibrated_model,
        'scaler': final_scaler,
        'plain_lr': plain_lr,
        'n_features': X.shape[1],
        'feature_names': META_FEATURE_NAMES,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"  Meta-learner saved: {MODEL_PATH}")

    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'n_samples': int(len(y)),
        'hit_rate': round(float(y.mean()), 4),
        'n_features': int(X.shape[1]),
        'feature_names': META_FEATURE_NAMES,
        'unique_dates': sorted(set(dates)),
        'n_dates': len(set(dates)),
        'cv_folds': folds,
        'cv_pooled_auc': round(float(pooled_auc), 4) if pooled_auc is not None else None,
        'cv_pooled_top10': round(float(pooled_top10), 4) if pooled_top10 is not None and not np.isnan(pooled_top10) else None,
        'cv_pooled_bot10': round(float(pooled_bot10), 4) if pooled_bot10 is not None and not np.isnan(pooled_bot10) else None,
        'model_type': 'LogisticRegression + isotonic calibration',
        'regularization': 'L2 (C=1.0)',
        'calibration_cv': cv_folds,
        'weights': _extract_weights(plain_lr),
    }

    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {META_PATH}")

    return calibrated_model, metadata


def _extract_weights(lr_model):
    """Extract and label the logistic regression coefficients."""
    if not hasattr(lr_model, 'coef_'):
        return None
    coefs = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]

    weights = {'intercept': round(float(intercept), 4)}
    for name, coef in zip(META_FEATURE_NAMES, coefs):
        weights[name] = round(float(coef), 4)

    return weights


# ===============================================================
# SCORING (used by run_board_v5.py)
# ===============================================================

def score_meta(results, model_path=None):
    """Score props with the trained meta-learner.

    Takes list of prop dicts that already have xgb_prob (and optionally
    mlp_prob, sim_prob). Builds meta-features, runs through the calibrated
    meta-learner, and sets meta_prob + updates ensemble_prob.

    Graceful fallback: if meta-learner not trained or scoring fails,
    falls back to the current 0.6*xgb + 0.4*mlp blend.

    Args:
        results: list of prop dicts with xgb_prob already set
        model_path: path to saved meta-learner pickle

    Returns:
        results with meta_prob and updated ensemble_prob
    """
    if model_path is None:
        model_path = MODEL_PATH

    if not os.path.exists(model_path):
        _fallback_ensemble(results)
        return results

    try:
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)

        model = bundle['model']
        scaler = bundle['scaler']
        expected_features = bundle.get('n_features', 12)

        # Build meta-features for all scoreable records
        scoreable_indices = []
        X_list = []

        for i, r in enumerate(results):
            features = build_meta_features(r)
            if features is not None and len(features) == expected_features:
                scoreable_indices.append(i)
                X_list.append(features)

        if not X_list:
            _fallback_ensemble(results)
            return results

        X = np.array(X_list, dtype=np.float64)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]

        scored = 0
        for idx, prob in zip(scoreable_indices, probs):
            results[idx]['meta_prob'] = round(float(prob), 4)
            results[idx]['ensemble_prob'] = round(float(prob), 4)
            scored += 1

        # For records that couldn't be scored by meta-learner, use fallback
        meta_scored = set(scoreable_indices)
        for i, r in enumerate(results):
            if i not in meta_scored:
                _fallback_single(r)

        print(f"  META-LEARNER: Scored {scored}/{len(results)} props")
        return results

    except Exception as e:
        print(f"  META-LEARNER: Failed ({e}), falling back to 0.6/0.4 blend")
        _fallback_ensemble(results)
        return results


def _fallback_ensemble(results):
    """Apply the hardcoded 0.6*xgb + 0.4*mlp blend as fallback."""
    for r in results:
        _fallback_single(r)


def _fallback_single(r):
    """Apply fallback ensemble to a single prop dict."""
    xgb_p = r.get('xgb_prob')
    mlp_p = r.get('mlp_prob')
    if xgb_p is not None and mlp_p is not None:
        r['ensemble_prob'] = round(0.6 * float(xgb_p) + 0.4 * float(mlp_p), 4)
    elif xgb_p is not None:
        r['ensemble_prob'] = float(xgb_p)


# ===============================================================
# EVALUATION
# ===============================================================

def eval_meta(records=None):
    """Evaluate meta-learner vs baseline via walk-forward CV.

    Compares:
    1. xgb_prob alone (current baseline for many records)
    2. hardcoded 0.6*xgb + 0.4*mlp blend
    3. meta-learner (learned blend)
    """
    if records is None:
        records = collect_graded_records()

    if len(records) < 50:
        print(f"  ERROR: Only {len(records)} usable records -- need at least 50")
        return

    X, y, dates = _build_training_data(records)
    unique_dates = sorted(set(dates))
    dates_arr = np.array(dates)

    print(f"\n  Evaluation: {len(y)} samples, {len(unique_dates)} dates")
    print(f"  Hit rate: {y.mean():.1%}")

    # Walk-forward with all three methods
    results_xgb = []
    results_blend = []
    results_meta = []

    for i in range(1, len(unique_dates)):
        train_dates = set(unique_dates[:i])
        test_date = unique_dates[i]

        train_mask = np.array([d in train_dates for d in dates])
        test_mask = dates_arr == test_date

        if train_mask.sum() < 30 or test_mask.sum() < 5:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Method 1: XGBoost alone (column 0)
        xgb_probs = X_test[:, 0]
        results_xgb.append((y_test, xgb_probs, test_date))

        # Method 2: Hardcoded 0.6/0.4 blend (columns 0 and 1)
        xgb_col = X_test[:, 0]
        mlp_col = X_test[:, 1]
        blend_probs = 0.6 * xgb_col + 0.4 * mlp_col
        results_blend.append((y_test, blend_probs, test_date))

        # Method 3: Meta-learner
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lr = LogisticRegression(C=1.0, penalty='l2', max_iter=1000, solver='lbfgs', random_state=42)
        lr.fit(X_train_s, y_train)
        meta_probs = lr.predict_proba(X_test_s)[:, 1]
        results_meta.append((y_test, meta_probs, test_date))

    if not results_meta:
        print("  Not enough data for walk-forward evaluation")
        return

    # Aggregate and compare
    def _aggregate(results_list, name):
        all_y = np.concatenate([r[0] for r in results_list])
        all_p = np.concatenate([r[1] for r in results_list])
        auc = _compute_auc(all_y, all_p)
        logloss = _compute_logloss(all_y, all_p)
        acc = ((all_p >= 0.5).astype(int) == all_y).mean()

        p90 = np.percentile(all_p, 90)
        p10 = np.percentile(all_p, 10)
        top10 = all_y[all_p >= p90].mean() if (all_p >= p90).sum() > 0 else float('nan')
        bot10 = all_y[all_p <= p10].mean() if (all_p <= p10).sum() > 0 else float('nan')

        top_str = f"{top10:.1%}" if not np.isnan(top10) else "N/A"
        bot_str = f"{bot10:.1%}" if not np.isnan(bot10) else "N/A"
        print(f"    {name:25s}: AUC={auc:.4f}, Acc={acc:.3f}, LogLoss={logloss:.4f}, "
              f"Top10%={top_str}, Bot10%={bot_str}, N={len(all_y)}")
        return auc

    print(f"\n  Walk-Forward Comparison ({len(results_meta)} test folds):")
    print(f"  {'=' * 90}")
    auc_xgb = _aggregate(results_xgb, "XGBoost only")
    auc_blend = _aggregate(results_blend, "Hardcoded 0.6/0.4 blend")
    auc_meta = _aggregate(results_meta, "Meta-learner (learned)")
    print(f"  {'=' * 90}")

    # Improvement summary
    imp_vs_xgb = auc_meta - auc_xgb
    imp_vs_blend = auc_meta - auc_blend
    print(f"\n  Meta-learner vs XGBoost only: {'+' if imp_vs_xgb >= 0 else ''}{imp_vs_xgb:.4f} AUC")
    print(f"  Meta-learner vs 0.6/0.4 blend: {'+' if imp_vs_blend >= 0 else ''}{imp_vs_blend:.4f} AUC")

    # Per-date breakdown
    print(f"\n  Per-Date AUC Breakdown:")
    print(f"  {'Date':15s} {'N':>5s} {'XGB':>8s} {'Blend':>8s} {'Meta':>8s} {'Winner':>10s}")
    print(f"  {'-'*55}")

    for (y_x, p_x, d_x), (y_b, p_b, _), (y_m, p_m, _) in zip(results_xgb, results_blend, results_meta):
        a_x = _compute_auc(y_x, p_x)
        a_b = _compute_auc(y_b, p_b)
        a_m = _compute_auc(y_m, p_m)
        best = max(a_x, a_b, a_m)
        winner = 'XGB' if a_x == best else ('Blend' if a_b == best else 'Meta')
        print(f"  {d_x:15s} {len(y_x):5d} {a_x:8.4f} {a_b:8.4f} {a_m:8.4f} {winner:>10s}")


# ===============================================================
# WEIGHTS DISPLAY
# ===============================================================

def show_weights():
    """Display the learned meta-learner weights to understand model trust allocation."""
    if not os.path.exists(META_PATH):
        print("  ERROR: No trained meta-learner. Run: python3 meta_learner.py train")
        return

    with open(META_PATH) as f:
        meta = json.load(f)

    weights = meta.get('weights')
    if not weights:
        print("  ERROR: No weights stored in metadata")
        return

    print("=" * 65)
    print("  Meta-Learner Weights (Logistic Regression Coefficients)")
    print("=" * 65)
    print(f"\n  Trained: {meta.get('trained_at', 'unknown')}")
    print(f"  Samples: {meta.get('n_samples', '?')}")
    print(f"  Dates:   {meta.get('n_dates', '?')}")
    print(f"  CV AUC:  {meta.get('cv_pooled_auc', 'N/A')}")

    print(f"\n  {'Feature':25s} {'Coef':>8s} {'Interpretation'}")
    print(f"  {'-'*70}")

    # Sort by absolute coefficient magnitude
    intercept = weights.pop('intercept', 0)
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)

    for name, coef in sorted_weights:
        if abs(coef) > 0.5:
            strength = "STRONG"
        elif abs(coef) > 0.2:
            strength = "moderate"
        elif abs(coef) > 0.05:
            strength = "weak"
        else:
            strength = "negligible"

        direction = "+" if coef > 0 else "-"
        interpretation = _interpret_weight(name, coef)
        print(f"  {name:25s} {coef:>+8.4f} {strength:12s} {interpretation}")

    print(f"\n  {'intercept':25s} {intercept:>+8.4f}")

    # Trust allocation summary
    print(f"\n  Sub-Model Trust Allocation:")
    print(f"  {'-'*45}")

    xgb_w = weights.get('xgb_prob', 0)
    mlp_w = weights.get('mlp_prob', 0)
    sim_w = weights.get('sim_prob', 0)
    total_sub = abs(xgb_w) + abs(mlp_w) + abs(sim_w)

    if total_sub > 0:
        print(f"    XGBoost:     {abs(xgb_w)/total_sub:6.1%} (coef={xgb_w:+.4f})")
        print(f"    MLP:         {abs(mlp_w)/total_sub:6.1%} (coef={mlp_w:+.4f})")
        print(f"    Simulation:  {abs(sim_w)/total_sub:6.1%} (coef={sim_w:+.4f})")
    else:
        print("    (sub-model weights are near zero -- interactions dominate)")

    # Context feature summary
    print(f"\n  Context Feature Effects:")
    context_features = ['tier_ordinal', 'direction_binary', 'is_combo', 'abs_gap']
    for feat in context_features:
        coef = weights.get(feat, 0)
        print(f"    {feat:25s} {coef:>+8.4f}")


def _interpret_weight(name, coef):
    """Human-readable interpretation of a meta-feature weight."""
    interpretations = {
        'xgb_prob': "higher XGBoost confidence -> higher meta prediction" if coef > 0 else "meta discounts XGBoost",
        'mlp_prob': "higher MLP confidence -> higher meta prediction" if coef > 0 else "meta discounts MLP",
        'sim_prob': "simulation agreement boosts confidence" if coef > 0 else "simulation disagreement is informative",
        'agreement': "model consensus amplifies signal" if coef > 0 else "model agreement is overconfident",
        'disagreement': "model disagreement -> uncertainty penalty" if coef < 0 else "disagreement signals opportunity",
        'optimistic': "optimistic model often right" if coef > 0 else "optimistic model is overfit",
        'pessimistic': "floor model is reliable" if coef > 0 else "pessimistic model undershoots",
        'simple_avg': "average is useful baseline" if coef > 0 else "average is misleading",
        'tier_ordinal': "higher tier -> more reliable" if coef > 0 else "tier grades are miscalibrated",
        'direction_binary': "OVERs more reliable" if coef > 0 else "UNDERs more reliable",
        'is_combo': "combo stats more reliable" if coef > 0 else "combo stats less reliable",
        'abs_gap': "larger gaps more reliable" if coef > 0 else "large gaps are traps",
    }
    return interpretations.get(name, "")


# ===============================================================
# CLI
# ===============================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == 'train':
        print("=" * 65)
        print("  Meta-Learner -- Training from Graded Predictions")
        print("  Architecture: LogisticRegression + Isotonic Calibration")
        print("=" * 65)
        model, metadata = train_meta()
        if model is not None and metadata is not None:
            print(f"\n  Training complete: {metadata['n_samples']} samples, "
                  f"{metadata['n_dates']} dates")
            if metadata.get('cv_pooled_auc'):
                print(f"  Pooled CV AUC: {metadata['cv_pooled_auc']:.4f}")

    elif command == 'eval':
        print("=" * 65)
        print("  Meta-Learner -- Walk-Forward Evaluation")
        print("=" * 65)
        eval_meta()

    elif command == 'weights':
        show_weights()

    else:
        print(f"Unknown command: {command}")
        print("Usage: python3 meta_learner.py [train|eval|weights]")
        sys.exit(1)


if __name__ == '__main__':
    main()

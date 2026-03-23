#!/usr/bin/env python3
"""
K-Nearest Neighbors Prop Classifier — Ensemble partner for XGBoost.

Distance-weighted KNN (k=50) with StandardScaler + median imputation.
KNN is distance-based and cannot handle NaN or different feature scales.
Subsamples to 30K most recent records when total > 50K for speed.

Reuses exact same features as XGBoost (engineer_features from xgb_model.py).

Usage:
    python3 predictions/knn_model.py train          # Train on all data
    python3 predictions/knn_model.py eval           # Walk-forward CV
    python3 predictions/knn_model.py score <file>   # Score a board/results file
"""

import json
import os
import sys
import warnings
import pickle
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip3 install scikit-learn")
    sys.exit(1)

# Reuse feature engineering from XGBoost
PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PREDICTIONS_DIR)

from xgb_model import (
    engineer_features, collect_all_training_data, collect_training_data,
    FEATURE_COLS, _compute_auc, _compute_logloss, _safe_float,
)

MODEL_PATH = os.path.join(PREDICTIONS_DIR, 'knn_model.pkl')
META_PATH = os.path.join(PREDICTIONS_DIR, 'knn_model_meta.json')

# ── HYPERPARAMETERS ──
N_NEIGHBORS = 50
WEIGHTS = 'distance'
METRIC = 'minkowski'
MAX_TRAIN_SAMPLES = 30000  # Subsample for speed (KNN is O(n) at predict time)


# ===============================================================
# IMPUTATION + SCALING
# ===============================================================

def _impute_and_scale(X_train, X_test=None, medians=None, scaler=None):
    """Replace NaN with column medians, then StandardScale."""
    if medians is None:
        medians = np.nanmedian(X_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)

    X_train_imp = X_train.copy()
    for j in range(X_train_imp.shape[1]):
        mask = np.isnan(X_train_imp[:, j])
        X_train_imp[mask, j] = medians[j]

    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
    else:
        X_train_scaled = scaler.transform(X_train_imp)

    X_test_scaled = None
    if X_test is not None:
        X_test_imp = X_test.copy()
        for j in range(X_test_imp.shape[1]):
            mask = np.isnan(X_test_imp[:, j])
            X_test_imp[mask, j] = medians[j]
        X_test_scaled = scaler.transform(X_test_imp)

    return X_train_scaled, X_test_scaled, medians, scaler


# ===============================================================
# SUBSAMPLING
# ===============================================================

def _subsample_recent(X, y, dates, sample_weights=None, sources=None, max_n=MAX_TRAIN_SAMPLES):
    """Subsample to max_n most recent records if total > max_n.

    Prioritizes graded data, then most recent backfill/historical.
    """
    if len(y) <= max_n:
        return X, y, dates, sample_weights, sources

    dates_arr = np.array(dates)
    sources_arr = np.array(sources) if sources is not None else np.array(['graded'] * len(dates))

    # Always keep all graded data
    graded_mask = sources_arr == 'graded'
    n_graded = graded_mask.sum()

    if n_graded >= max_n:
        # Even graded data exceeds limit — take most recent graded
        graded_idx = np.where(graded_mask)[0]
        graded_dates_vals = dates_arr[graded_idx]
        sorted_order = np.argsort(graded_dates_vals)[::-1]
        keep_idx = graded_idx[sorted_order[:max_n]]
    else:
        # Keep all graded, fill remainder with most recent non-graded
        non_graded_idx = np.where(~graded_mask)[0]
        non_graded_dates = dates_arr[non_graded_idx]
        sorted_order = np.argsort(non_graded_dates)[::-1]
        n_fill = max_n - n_graded
        fill_idx = non_graded_idx[sorted_order[:n_fill]]
        keep_idx = np.concatenate([np.where(graded_mask)[0], fill_idx])

    keep_idx = np.sort(keep_idx)

    X_sub = X[keep_idx]
    y_sub = y[keep_idx]
    dates_sub = [dates[i] for i in keep_idx]
    sw_sub = sample_weights[keep_idx] if sample_weights is not None else None
    src_sub = [sources[i] for i in keep_idx] if sources is not None else None

    return X_sub, y_sub, dates_sub, sw_sub, src_sub


# ===============================================================
# WALK-FORWARD CV
# ===============================================================

def walk_forward_cv(X, y, dates, sample_weights=None, sources=None):
    """Walk-forward CV matching XGBoost's fold structure."""
    dates_arr = np.array(dates)
    sources_arr = np.array(sources) if sources is not None else np.array(['graded'] * len(dates))

    graded_mask = sources_arr == 'graded'
    graded_dates = sorted(set(d for d, s in zip(dates, sources_arr) if s == 'graded' and d >= '2026-'))
    historical_mask = np.array([d < '2026-' and s == 'historical' for d, s in zip(dates, sources_arr)])
    backfill_mask = np.isin(sources_arr, ['backfill', 'sgo_backfill'])

    if len(graded_dates) < 2:
        print("  WARNING: Need at least 2 graded days for walk-forward CV")
        return [], [], []

    folds = []
    oof_probs = []
    oof_actuals = []

    for i in range(1, len(graded_dates)):
        train_graded_dates = set(graded_dates[:i])
        test_date = graded_dates[i]

        train_graded_mask = np.array([d in train_graded_dates and s == 'graded' for d, s in zip(dates, sources_arr)])
        train_backfill_mask = np.array([d < test_date and s in ('backfill', 'sgo_backfill') for d, s in zip(dates, sources_arr)])
        train_mask = historical_mask | train_graded_mask | train_backfill_mask
        test_mask = (dates_arr == test_date) & graded_mask

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        X_train_raw, y_train_raw = X[train_mask], y[train_mask]
        X_test_raw, y_test = X[test_mask], y[test_mask]
        sw_train = sample_weights[train_mask] if sample_weights is not None else None

        # Subsample training data for speed
        if X_train_raw.shape[0] > MAX_TRAIN_SAMPLES:
            train_dates_fold = [dates[j] for j in range(len(dates)) if train_mask[j]]
            train_sources_fold = [sources[j] for j in range(len(dates)) if train_mask[j]] if sources else None
            X_train_raw, y_train_raw, _, sw_train, _ = _subsample_recent(
                X_train_raw, y_train_raw, train_dates_fold, sw_train, train_sources_fold
            )

        # Impute NaN + scale
        X_train_s, X_test_s, _, _ = _impute_and_scale(X_train_raw, X_test_raw)

        # Train KNN
        model = KNeighborsClassifier(
            n_neighbors=min(N_NEIGHBORS, len(y_train_raw) - 1),
            weights=WEIGHTS,
            metric=METRIC,
            n_jobs=-1,
        )
        if sw_train is not None:
            # KNN fit doesn't support sample_weight in predict, but we can
            # still pass it — sklearn ignores it for KNN anyway
            model.fit(X_train_s, y_train_raw)
        else:
            model.fit(X_train_s, y_train_raw)

        y_prob = model.predict_proba(X_test_s)[:, 1]

        oof_probs.extend(y_prob.tolist())
        oof_actuals.extend(y_test.tolist())

        y_pred = (y_prob >= 0.5).astype(int)
        accuracy = (y_pred == y_test).mean()
        auc = _compute_auc(y_test, y_prob)
        logloss = _compute_logloss(y_test, y_prob)

        top_mask = y_prob >= np.percentile(y_prob, 90)
        bot_mask = y_prob <= np.percentile(y_prob, 10)
        top_hr = y_test[top_mask].mean() if top_mask.sum() > 0 else np.nan
        bot_hr = y_test[bot_mask].mean() if bot_mask.sum() > 0 else np.nan

        fold = {
            'train_graded_dates': list(train_graded_dates),
            'test_date': test_date,
            'train_size': int(X_train_raw.shape[0]),
            'test_size': int(test_mask.sum()),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'logloss': float(logloss),
            'top_decile_hr': float(top_hr) if not np.isnan(top_hr) else None,
            'bot_decile_hr': float(bot_hr) if not np.isnan(bot_hr) else None,
            'y_test': y_test,
            'y_prob': y_prob,
        }
        folds.append(fold)

        print(f"    Fold: train {sorted(train_graded_dates)} -> test {test_date}")
        print(f"      N={fold['test_size']}, Acc={accuracy:.3f}, AUC={auc:.3f}, "
              f"Top10%={top_hr:.1%}, Bot10%={bot_hr:.1%}")

    # Pooled metrics
    if len(folds) >= 2:
        all_y_test = np.concatenate([f['y_test'] for f in folds])
        all_y_prob = np.concatenate([f['y_prob'] for f in folds])
        pooled_auc = _compute_auc(all_y_test, all_y_prob)
        p90 = np.percentile(all_y_prob, 90)
        p10 = np.percentile(all_y_prob, 10)
        pooled_top10 = all_y_test[all_y_prob >= p90].mean() if (all_y_prob >= p90).sum() > 0 else np.nan
        pooled_bot10 = all_y_test[all_y_prob <= p10].mean() if (all_y_prob <= p10).sum() > 0 else np.nan
        print(f"\n    Pooled CV: AUC={pooled_auc:.3f}, Top10%={pooled_top10:.1%}, Bot10%={pooled_bot10:.1%}")
        print(f"    Pooled N={len(all_y_test)}, Hit rate={all_y_test.mean():.1%}")

    # Strip numpy arrays for JSON serialization
    for fold in folds:
        fold.pop('y_test', None)
        fold.pop('y_prob', None)

    return folds, oof_probs, oof_actuals


# ===============================================================
# CALIBRATION
# ===============================================================

def _build_calibration_map(oof_probs, oof_actuals):
    """Build bin-based calibration from OOF predictions."""
    if len(oof_probs) < 50:
        return None

    probs_arr = np.array(oof_probs)
    actuals_arr = np.array(oof_actuals)
    bins = np.linspace(0, 1, 11)
    cal_map = []
    for j in range(len(bins) - 1):
        mask = (probs_arr >= bins[j]) & (probs_arr < bins[j + 1])
        if mask.sum() >= 5:
            cal_map.append({
                'bin_low': round(float(bins[j]), 2),
                'bin_high': round(float(bins[j + 1]), 2),
                'raw_avg': round(float(probs_arr[mask].mean()), 3),
                'actual_hr': round(float(actuals_arr[mask].mean()), 3),
                'n': int(mask.sum()),
            })
    return cal_map if cal_map else None


def _calibrate_prob(raw_prob, cal_map):
    """Apply calibration map to a raw knn_prob."""
    if not cal_map:
        return raw_prob
    for b in cal_map:
        if b['bin_low'] <= raw_prob < b['bin_high']:
            return b['actual_hr']
    return raw_prob


# ===============================================================
# FULL TRAINING PIPELINE
# ===============================================================

def train_and_evaluate(use_historical=True):
    """Train KNN on all data, run walk-forward CV, save model."""
    print("=" * 60)
    print(f"  KNN Classifier — Training {'(+historical)' if use_historical else '(graded only)'}")
    print(f"  k={N_NEIGHBORS}, weights={WEIGHTS}, metric={METRIC}")
    print(f"  Max training samples: {MAX_TRAIN_SAMPLES}")
    print("=" * 60)

    if use_historical:
        records = collect_all_training_data(use_historical=True)
    else:
        records = collect_training_data()

    if len(records) < 100:
        print(f"  ERROR: Only {len(records)} records -- need at least 100")
        sys.exit(1)

    X, y, dates = engineer_features(records)

    sample_weights = None
    if use_historical or any(r.get('_data_source') == 'backfill' for r in records):
        def _weight(r):
            src = r.get('_data_source', '')
            if src == 'graded': return 25.0
            if src == 'backfill': return 10.0
            if src == 'sgo_backfill': return 8.0
            return 1.0
        sample_weights = np.array([_weight(r) for r in records])

    sources = [r.get('_data_source', 'graded') for r in records]

    # Walk-forward CV
    print("\n  Walk-Forward Cross-Validation (KNN):")
    folds, oof_probs, oof_actuals = walk_forward_cv(
        X, y, dates, sample_weights=sample_weights, sources=sources
    )

    if folds:
        avg_auc = np.mean([f['auc'] for f in folds])
        avg_acc = np.mean([f['accuracy'] for f in folds])
        print(f"\n  KNN CV Summary: Avg AUC={avg_auc:.3f}, Avg Acc={avg_acc:.3f}")
    else:
        avg_auc = None
        avg_acc = None

    # Subsample for final model training
    X_sub, y_sub, dates_sub, sw_sub, src_sub = _subsample_recent(
        X, y, dates, sample_weights, sources
    )
    print(f"\n  Training final KNN on {len(y_sub)} samples ({len(y)} total, subsampled to {MAX_TRAIN_SAMPLES} max)...")

    # Impute + scale
    X_scaled, _, medians, scaler = _impute_and_scale(X_sub)

    # Train final model
    model = KNeighborsClassifier(
        n_neighbors=min(N_NEIGHBORS, len(y_sub) - 1),
        weights=WEIGHTS,
        metric=METRIC,
        n_jobs=-1,
    )
    model.fit(X_scaled, y_sub)

    # Save model bundle
    bundle = {
        'model': model,
        'scaler': scaler,
        'medians': medians,
        'n_features': X.shape[1],
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"  KNN model saved: {MODEL_PATH}")

    # Build calibration map
    calibration_map = _build_calibration_map(oof_probs, oof_actuals)

    metadata = {
        'trained_at': datetime.now().isoformat(),
        'n_samples_total': int(len(y)),
        'n_samples_trained': int(len(y_sub)),
        'hit_rate': float(y.mean()),
        'n_features': int(X.shape[1]),
        'unique_dates': sorted(set(dates)),
        'cv_folds': folds,
        'cv_avg_auc': float(avg_auc) if avg_auc is not None else None,
        'cv_avg_accuracy': float(avg_acc) if avg_acc is not None else None,
        'calibration_map': calibration_map,
        'model_path': MODEL_PATH,
        'hyperparameters': {
            'n_neighbors': N_NEIGHBORS,
            'weights': WEIGHTS,
            'metric': METRIC,
            'max_train_samples': MAX_TRAIN_SAMPLES,
        },
    }

    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {META_PATH}")

    print(f"\n  Training: {metadata['n_samples_trained']} samples (of {metadata['n_samples_total']}), {metadata['n_features']} features")
    if metadata.get('cv_avg_auc'):
        print(f"  CV AUC: {metadata['cv_avg_auc']:.3f}")
        print(f"  CV Acc: {metadata['cv_avg_accuracy']:.3f}")

    return model, metadata


# ===============================================================
# SCORING (used by run_board_v5.py)
# ===============================================================

def score_props(results, model_path=None):
    """Load trained KNN and add knn_prob to each result dict."""
    if model_path is None:
        model_path = MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. Run: python3 knn_model.py train")

    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)

    model = bundle['model']
    scaler = bundle['scaler']
    medians = bundle['medians']

    # Load calibration map
    cal_map = None
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        cal_map = meta.get('calibration_map')

    # Build feature matrix
    temp_records = []
    for r in results:
        rec = dict(r)
        rec['_hit_label'] = False
        rec['_date'] = ''
        temp_records.append(rec)

    if not temp_records:
        return results

    X, _, _ = engineer_features(temp_records)
    X_scaled, _, _, _ = _impute_and_scale(X, medians=medians, scaler=scaler)

    probs = model.predict_proba(X_scaled)[:, 1]

    for r, prob in zip(results, probs):
        raw = round(float(prob), 4)
        r['knn_prob'] = raw
        if cal_map:
            r['knn_prob_calibrated'] = round(_calibrate_prob(raw, cal_map), 4)

    return results


# ===============================================================
# CLI
# ===============================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    no_historical = '--no-historical' in sys.argv
    use_historical = not no_historical

    if command == 'train':
        train_and_evaluate(use_historical=use_historical)

    elif command == 'eval':
        print("=" * 60)
        print(f"  KNN Classifier — Evaluation Only")
        print("=" * 60)

        if use_historical:
            records = collect_all_training_data(use_historical=True)
        else:
            records = collect_training_data()

        X, y, dates = engineer_features(records)

        sample_weights = None
        if use_historical or any(r.get('_data_source') == 'backfill' for r in records):
            def _weight(r):
                src = r.get('_data_source', '')
                if src == 'graded': return 25.0
                if src == 'backfill': return 10.0
                if src == 'sgo_backfill': return 8.0
                return 1.0
            sample_weights = np.array([_weight(r) for r in records])

        sources = [r.get('_data_source', 'graded') for r in records]
        print("\n  Walk-Forward CV (KNN):")
        folds, _, _ = walk_forward_cv(X, y, dates, sample_weights=sample_weights, sources=sources)

        if folds:
            avg_auc = np.mean([f['auc'] for f in folds])
            avg_acc = np.mean([f['accuracy'] for f in folds])
            print(f"\n  Summary: Avg AUC={avg_auc:.3f}, Avg Acc={avg_acc:.3f}")

    elif command == 'score':
        if len(sys.argv) < 3:
            print("Usage: python3 knn_model.py score <board_or_results.json>")
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
        scored = sum(1 for r in results if 'knn_prob' in r)
        print(f"  Scored {scored}/{len(results)} props")

        knn_probs = [r['knn_prob'] for r in results if 'knn_prob' in r]
        if knn_probs:
            arr = np.array(knn_probs)
            print(f"  knn_prob: mean={arr.mean():.3f}, std={arr.std():.3f}, "
                  f"min={arr.min():.3f}, max={arr.max():.3f}")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()

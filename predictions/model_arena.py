#!/usr/bin/env python3
"""
Model Arena — Multi-Model Training, Evaluation, and Stacking Ensemble.

Tests 10 diverse ML model types on the same training data, evaluates by
Brier score (calibration), keeps the top performers, and creates a stacking
meta-learner ensemble. Replaces the hardcoded 60/40 XGB+MLP blend.

Models tested:
  xgboost, lightgbm, random_forest, extra_trees, catboost,
  ridge, logistic, elastic_net, knn, naive_bayes

Primary metric: Brier score (lower = better calibrated for betting)
Secondary: AUC, accuracy, calibration error (per-bin predicted vs actual)

Usage:
    python3 predictions/model_arena.py train                         # Train all, evaluate, select, ensemble
    python3 predictions/model_arena.py train --board /path/board.json # Focused training (real lines)
    python3 predictions/model_arena.py eval                          # Evaluation report only
    python3 predictions/model_arena.py leaderboard                   # Current model rankings
"""

import json
import os
import sys
import pickle
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =====================================================================
# PATHS
# =====================================================================

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PREDICTIONS_DIR)

CACHE_DIR = os.path.join(PREDICTIONS_DIR, 'cache')
ARENA_DIR = os.path.join(CACHE_DIR, 'arena')
ARENA_MODELS_PATH = os.path.join(ARENA_DIR, 'arena_models.pkl')
ARENA_META_PATH = os.path.join(ARENA_DIR, 'arena_meta.json')
ARENA_ENSEMBLE_PATH = os.path.join(ARENA_DIR, 'arena_ensemble.pkl')

# =====================================================================
# IMPORTS — reuse feature engineering from xgb_model.py
# =====================================================================

from xgb_model import (
    engineer_features, collect_all_training_data,
    FEATURE_COLS, _compute_auc, _compute_logloss, _safe_float,
)

# =====================================================================
# MODEL DEFINITIONS — graceful import handling
# =====================================================================

def _build_model_registry():
    """Build the registry of 10 models, skipping any with missing dependencies.

    Returns dict of {name: model_instance} for all available models,
    plus a list of skipped model names with reasons.
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV

    models = {}
    skipped = []

    # --- Tree-based models (handle NaN natively) ---

    # XGBoost
    try:
        from xgboost import XGBClassifier
        models['xgboost'] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            eval_metric='logloss', verbosity=0, random_state=42,
        )
    except (ImportError, OSError) as e:
        skipped.append(('xgboost', str(e).split('\n')[0]))

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        models['lightgbm'] = LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6,
            verbose=-1, random_state=42,
        )
    except (ImportError, OSError) as e:
        skipped.append(('lightgbm', str(e).split('\n')[0]))

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        models['catboost'] = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05,
            verbose=0, random_seed=42,
        )
    except (ImportError, OSError) as e:
        skipped.append(('catboost', str(e).split('\n')[0]))

    # --- Sklearn tree ensembles (handle NaN in recent sklearn, but impute for safety) ---

    models['random_forest'] = make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        ),
    )

    models['extra_trees'] = make_pipeline(
        SimpleImputer(strategy='median'),
        ExtraTreesClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        ),
    )

    # --- Linear models (require imputation + scaling) ---

    # RidgeClassifier has no predict_proba, so wrap with CalibratedClassifierCV
    models['ridge'] = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        CalibratedClassifierCV(
            RidgeClassifier(alpha=1.0),
            cv=3,
            method='sigmoid',
        ),
    )

    models['logistic'] = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    )

    models['elastic_net'] = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        SGDClassifier(
            loss='log_loss', penalty='elasticnet', l1_ratio=0.5,
            max_iter=1000, random_state=42,
        ),
    )

    # --- Instance-based / probabilistic ---

    models['knn'] = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=50, weights='distance', n_jobs=-1),
    )

    models['naive_bayes'] = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        GaussianNB(),
    )

    return models, skipped


def _predict_proba_safe(model, X, model_name):
    """Get probability predictions from any model type, handling edge cases.

    Some models (SGDClassifier) need special handling for predict_proba.
    Falls back to decision_function -> sigmoid if predict_proba unavailable.
    """
    try:
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]
        return probs.ravel()
    except AttributeError:
        pass

    # Fallback: decision_function -> sigmoid
    try:
        decision = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-decision))
    except AttributeError:
        print(f"    WARNING: {model_name} has no predict_proba or decision_function")
        return np.full(X.shape[0], 0.5)


# =====================================================================
# DATA LOADING
# =====================================================================

def _load_data(board_file=None):
    """Load training data — focused (real lines) if board provided, else standard pipeline.

    Returns (records, source_label).
    """
    if board_file:
        from train_current_lines import generate_focused_training
        records = generate_focused_training(board_file)
        if not records:
            print("  ERROR: No focused training records generated")
            return None, 'focused'
        return records, 'focused'
    else:
        records = collect_all_training_data(use_historical=True)
        if not records:
            print("  ERROR: No training records found")
            return None, 'standard'
        return records, 'standard'


def _prepare_data(records, source_label):
    """Engineer features, compute sample weights, build train/test split.

    Uses walk-forward: first 70% of dates = train, last 30% = test.

    Returns (X_train, y_train, X_test, y_test, dates_train, dates_test, sample_weights_train).
    """
    X, y, dates = engineer_features(records)

    # Sample weights by data source
    sample_weights = np.ones(len(y))
    if source_label == 'standard':
        for i, r in enumerate(records):
            src = r.get('_data_source', '')
            if src == 'graded':
                sample_weights[i] = 25.0
            elif src == 'backfill':
                sample_weights[i] = 10.0
            elif src == 'sgo_backfill':
                sample_weights[i] = 8.0
            else:
                sample_weights[i] = 1.0

    # Walk-forward split: first 70% dates -> train, last 30% -> test
    unique_dates = sorted(set(d for d in dates if d))
    if len(unique_dates) < 3:
        print("  WARNING: Very few unique dates, using 50/50 split")
        split_idx = max(1, len(unique_dates) // 2)
    else:
        split_idx = int(len(unique_dates) * 0.7)

    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])

    train_mask = np.array([d in train_dates for d in dates])
    test_mask = np.array([d in test_dates for d in dates])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    sw_train = sample_weights[train_mask]

    print(f"\n  Walk-forward split:")
    print(f"    Train: {len(y_train):,} samples ({sorted(train_dates)[0] if train_dates else '?'} -> "
          f"{sorted(train_dates)[-1] if train_dates else '?'})")
    print(f"    Test:  {len(y_test):,} samples ({sorted(test_dates)[0] if test_dates else '?'} -> "
          f"{sorted(test_dates)[-1] if test_dates else '?'})")
    print(f"    Train hit rate: {y_train.mean():.1%}, Test hit rate: {y_test.mean():.1%}")

    return X_train, y_train, X_test, y_test, sw_train


# =====================================================================
# EVALUATION
# =====================================================================

def _brier_score(y_true, y_prob):
    """Brier score: mean squared error of probability predictions.

    Lower is better. Perfect = 0.0, coin flip on 50/50 data = 0.25.
    """
    y_prob = np.clip(y_prob, 0.0, 1.0)
    return float(np.mean((y_prob - y_true) ** 2))


def _calibration_error(y_true, y_prob, n_bins=10):
    """Expected calibration error: mean absolute difference between
    predicted probability and actual hit rate per bin.

    Lower is better. Reports per-bin breakdown and weighted ECE.
    """
    y_prob = np.clip(y_prob, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_details = []
    total_ece = 0.0
    total_n = 0

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:  # include 1.0 in last bin
            mask = mask | (y_prob == bins[i + 1])

        n = int(mask.sum())
        if n == 0:
            continue

        avg_predicted = float(y_prob[mask].mean())
        avg_actual = float(y_true[mask].mean())
        abs_diff = abs(avg_predicted - avg_actual)

        bin_details.append({
            'bin_low': round(float(bins[i]), 2),
            'bin_high': round(float(bins[i + 1]), 2),
            'n': n,
            'avg_predicted': round(avg_predicted, 4),
            'avg_actual': round(avg_actual, 4),
            'abs_error': round(abs_diff, 4),
        })

        total_ece += abs_diff * n
        total_n += n

    ece = total_ece / total_n if total_n > 0 else 0.0
    return round(ece, 4), bin_details


def evaluate_models(models, X_train, y_train, X_test, y_test, sample_weights_train=None):
    """Train each model on training set, evaluate on test set.

    Returns list of result dicts sorted by Brier score (best first).
    Each result: {name, brier, auc, accuracy, cal_error, y_prob, model, cal_bins, train_time}
    """
    results = []

    for name, model in models.items():
        print(f"\n  Training {name}...", end=' ', flush=True)
        t0 = time.time()

        try:
            # Tree-based models that support sample_weight directly
            if name in ('xgboost', 'lightgbm', 'catboost'):
                if name == 'xgboost':
                    model.fit(
                        X_train, y_train,
                        sample_weight=sample_weights_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False,
                    )
                elif name == 'lightgbm':
                    model.fit(
                        X_train, y_train,
                        sample_weight=sample_weights_train,
                        eval_set=[(X_test, y_test)],
                    )
                elif name == 'catboost':
                    model.fit(
                        X_train, y_train,
                        sample_weight=sample_weights_train,
                        eval_set=(X_test, y_test),
                        verbose=False,
                    )
            else:
                # Pipeline models -- pass sample_weight through the pipeline
                # sklearn pipelines route sample_weight to the final estimator
                # via the <step_name>__sample_weight parameter
                # For simplicity, just fit without sample_weight since these
                # are secondary models and the data is already weighted by
                # over-sampling in collection
                model.fit(X_train, y_train)

            train_time = time.time() - t0

            # Get probabilities
            y_prob = _predict_proba_safe(model, X_test, name)

            # Compute metrics
            brier = _brier_score(y_test, y_prob)
            auc = _compute_auc(y_test, y_prob)
            preds = (y_prob >= 0.5).astype(int)
            accuracy = float(np.mean(preds == y_test))
            cal_err, cal_bins = _calibration_error(y_test, y_prob)

            results.append({
                'name': name,
                'brier': brier,
                'auc': auc,
                'accuracy': accuracy,
                'cal_error': cal_err,
                'y_prob': y_prob,
                'model': model,
                'cal_bins': cal_bins,
                'train_time': train_time,
            })

            print(f"done ({train_time:.1f}s) -- Brier={brier:.4f} AUC={auc:.3f}")

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Sort by Brier score (lower is better)
    results.sort(key=lambda r: r['brier'])
    return results


def print_results_table(results, y_test):
    """Print formatted results table."""
    print("\n")
    print("=" * 72)
    print("  MODEL ARENA RESULTS (Brier Score -- lower is better)")
    print("=" * 72)

    baseline_brier = _brier_score(y_test, np.full(len(y_test), y_test.mean()))
    print(f"  Baseline (predict mean): Brier={baseline_brier:.4f}")
    print(f"  Test set: {len(y_test):,} samples, hit rate {y_test.mean():.1%}")
    print("-" * 72)

    for i, r in enumerate(results):
        status = r.get('status', '')
        rank = f"#{i+1}"
        line = (f"  {rank:4s} {r['name']:18s} "
                f"Brier={r['brier']:.4f}  AUC={r['auc']:.3f}  "
                f"Acc={r['accuracy']:.1%}  CalErr={r['cal_error']:.3f}  "
                f"({r['train_time']:.1f}s)  [{status}]")
        print(line)

    print("-" * 72)


# =====================================================================
# MODEL SELECTION
# =====================================================================

def select_top_models(results, max_models=5):
    """Select top models by Brier score with AUC floor.

    Requirements to survive:
      - AUC > 0.52 (must be better than random)
      - Top max_models by Brier score among those passing AUC floor

    Mutates results in place to add 'status' field.
    Returns list of surviving result dicts.
    """
    AUC_FLOOR = 0.52

    survivors = []
    for r in results:
        if r['auc'] <= AUC_FLOOR:
            r['status'] = 'KILL (AUC)'
        elif len(survivors) >= max_models:
            r['status'] = 'KILL (rank)'
        else:
            r['status'] = 'KEEP'
            survivors.append(r)

    print(f"\n  Selection: {len(survivors)}/{len(results)} models survived")
    for r in results:
        marker = '>>' if r['status'] == 'KEEP' else '  '
        print(f"    {marker} {r['name']:18s} Brier={r['brier']:.4f}  AUC={r['auc']:.3f}  [{r['status']}]")

    return survivors


# =====================================================================
# STACKING ENSEMBLE
# =====================================================================

def build_ensemble(top_results, X_train, y_train, X_test, y_test, sample_weights_train=None):
    """Build a stacking meta-learner from out-of-fold predictions.

    Process:
      1. Split training data into K folds
      2. For each fold: train all top models on other folds, predict on held-out
      3. Stack out-of-fold predictions as features for a LogisticRegression meta-learner
      4. Evaluate ensemble on test set

    Returns (meta_learner, top_models_dict, ensemble_metrics).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    n_models = len(top_results)
    model_names = [r['name'] for r in top_results]
    print(f"\n  Building stacking ensemble from {n_models} models: {model_names}")

    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    # Out-of-fold prediction matrix
    oof_preds = np.zeros((len(y_train), n_models))

    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(kf.split(X_train)):
        X_fold_train = X_train[fold_train_idx]
        y_fold_train = y_train[fold_train_idx]
        X_fold_val = X_train[fold_val_idx]
        sw_fold = sample_weights_train[fold_train_idx] if sample_weights_train is not None else None

        for m_idx, result in enumerate(top_results):
            name = result['name']
            # Clone the model (re-instantiate to avoid state leakage)
            model = _clone_model(result['model'])

            try:
                if name in ('xgboost', 'lightgbm', 'catboost'):
                    model.fit(X_fold_train, y_fold_train, sample_weight=sw_fold, verbose=False)
                else:
                    model.fit(X_fold_train, y_fold_train)

                fold_probs = _predict_proba_safe(model, X_fold_val, name)
                oof_preds[fold_val_idx, m_idx] = fold_probs
            except Exception as e:
                print(f"    WARNING: {name} failed on fold {fold_idx}: {e}")
                oof_preds[fold_val_idx, m_idx] = 0.5

        print(f"    Fold {fold_idx+1}/{K} complete")

    # Handle any NaN in oof_preds
    nan_mask = np.isnan(oof_preds)
    if nan_mask.any():
        print(f"    Imputing {nan_mask.sum()} NaN values in OOF predictions")
        oof_preds = np.nan_to_num(oof_preds, nan=0.5)

    # Train meta-learner on out-of-fold predictions
    print(f"    Training meta-learner (LogisticRegression) on OOF matrix {oof_preds.shape}...")
    meta_scaler = StandardScaler()
    oof_scaled = meta_scaler.fit_transform(oof_preds)

    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_learner.fit(oof_scaled, y_train)

    # Print meta-learner weights (which models contribute most)
    print(f"\n    Meta-learner coefficients (model importance):")
    coefs = meta_learner.coef_[0]
    for name, coef in sorted(zip(model_names, coefs), key=lambda x: -abs(x[1])):
        bar = '+' * int(abs(coef) * 10) if coef > 0 else '-' * int(abs(coef) * 10)
        print(f"      {name:18s} {coef:+.4f}  {bar}")

    # Evaluate ensemble on test set
    print(f"\n    Evaluating ensemble on test set...")
    test_model_preds = np.zeros((len(y_test), n_models))

    # Retrain all models on full training set for final predictions
    final_models = {}
    for m_idx, result in enumerate(top_results):
        name = result['name']
        model = _clone_model(result['model'])

        try:
            if name in ('xgboost', 'lightgbm', 'catboost'):
                model.fit(X_train, y_train, sample_weight=sample_weights_train, verbose=False)
            else:
                model.fit(X_train, y_train)

            test_probs = _predict_proba_safe(model, X_test, name)
            test_model_preds[:, m_idx] = test_probs
            final_models[name] = model
        except Exception as e:
            print(f"    WARNING: {name} failed in final training: {e}")
            test_model_preds[:, m_idx] = 0.5

    # Meta-learner prediction
    test_scaled = meta_scaler.transform(test_model_preds)
    ensemble_probs = meta_learner.predict_proba(test_scaled)[:, 1]

    # Ensemble metrics
    ensemble_brier = _brier_score(y_test, ensemble_probs)
    ensemble_auc = _compute_auc(y_test, ensemble_probs)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)
    ensemble_acc = float(np.mean(ensemble_preds == y_test))
    ensemble_cal_err, ensemble_cal_bins = _calibration_error(y_test, ensemble_probs)

    ensemble_metrics = {
        'brier': ensemble_brier,
        'auc': ensemble_auc,
        'accuracy': ensemble_acc,
        'cal_error': ensemble_cal_err,
        'cal_bins': ensemble_cal_bins,
        'model_names': model_names,
        'meta_coefs': {name: round(float(c), 4) for name, c in zip(model_names, coefs)},
    }

    # Compare with best single model
    best_single = top_results[0]
    print(f"\n    ENSEMBLE:     Brier={ensemble_brier:.4f}  AUC={ensemble_auc:.3f}  "
          f"Acc={ensemble_acc:.1%}  CalErr={ensemble_cal_err:.3f}")
    print(f"    BEST SINGLE:  Brier={best_single['brier']:.4f}  AUC={best_single['auc']:.3f}  "
          f"Acc={best_single['accuracy']:.1%}  CalErr={best_single['cal_error']:.3f}  "
          f"({best_single['name']})")

    improvement = best_single['brier'] - ensemble_brier
    if improvement > 0:
        print(f"    Ensemble WINS by {improvement:.4f} Brier points")
    else:
        print(f"    Best single model wins by {abs(improvement):.4f} Brier points")
        print(f"    (Ensemble still used for diversity and robustness)")

    return meta_learner, meta_scaler, final_models, ensemble_metrics


def _clone_model(model):
    """Clone a model by reconstructing it with the same parameters.

    sklearn's clone() works for most, but we handle special cases.
    """
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        # Fallback: pickle round-trip (works for all serializable models)
        import io
        buf = io.BytesIO()
        pickle.dump(model, buf)
        buf.seek(0)
        return pickle.load(buf)


# =====================================================================
# PERSISTENCE
# =====================================================================

def _save_arena(meta_learner, meta_scaler, final_models, ensemble_metrics, eval_results):
    """Save all arena artifacts to predictions/cache/arena/."""
    os.makedirs(ARENA_DIR, exist_ok=True)

    # Save models
    arena_bundle = {
        'meta_learner': meta_learner,
        'meta_scaler': meta_scaler,
        'models': final_models,
        'model_names': list(final_models.keys()),
    }
    with open(ARENA_MODELS_PATH, 'wb') as f:
        pickle.dump(arena_bundle, f)
    print(f"\n  Models saved: {ARENA_MODELS_PATH}")

    # Save ensemble separately (for scoring)
    ensemble_bundle = {
        'meta_learner': meta_learner,
        'meta_scaler': meta_scaler,
        'model_names': list(final_models.keys()),
    }
    with open(ARENA_ENSEMBLE_PATH, 'wb') as f:
        pickle.dump(ensemble_bundle, f)

    # Save metadata (JSON-serializable)
    meta = {
        'trained_at': datetime.now().isoformat(),
        'model_names': list(final_models.keys()),
        'ensemble_metrics': {
            k: v for k, v in ensemble_metrics.items()
            if k not in ('cal_bins',)  # cal_bins is already JSON-safe
        },
        'ensemble_cal_bins': ensemble_metrics.get('cal_bins', []),
        'individual_results': [
            {
                'name': r['name'],
                'brier': round(r['brier'], 4),
                'auc': round(r['auc'], 4),
                'accuracy': round(r['accuracy'], 4),
                'cal_error': round(r['cal_error'], 4),
                'status': r.get('status', ''),
                'train_time': round(r['train_time'], 2),
            }
            for r in eval_results
        ],
        'n_features': len(FEATURE_COLS),
        'feature_cols': FEATURE_COLS,
    }
    with open(ARENA_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved: {ARENA_META_PATH}")


def _load_arena():
    """Load saved arena models and ensemble.

    Returns (arena_bundle, meta) or (None, None) if not found.
    """
    if not os.path.exists(ARENA_MODELS_PATH):
        return None, None

    try:
        with open(ARENA_MODELS_PATH, 'rb') as f:
            bundle = pickle.load(f)

        meta = None
        if os.path.exists(ARENA_META_PATH):
            with open(ARENA_META_PATH) as f:
                meta = json.load(f)

        return bundle, meta
    except Exception as e:
        print(f"  WARNING: Failed to load arena: {e}")
        return None, None


# =====================================================================
# SCORING — called by the pipeline
# =====================================================================

def score_arena(results):
    """Score props using the arena ensemble.

    Loads all saved models, runs each prop through all top models,
    blends via meta-learner, and sets arena_prob on each prop dict.

    Also sets individual model probs: xgb_arena_prob, lgbm_arena_prob, etc.

    Args:
        results: list of prop dicts from analyze_v3

    Returns:
        Number of props scored (0 if arena not trained).
    """
    bundle, meta = _load_arena()
    if bundle is None:
        return 0

    meta_learner = bundle['meta_learner']
    meta_scaler = bundle['meta_scaler']
    models = bundle['models']
    model_names = bundle['model_names']

    # Build feature matrix
    temp_records = []
    for r in results:
        rec = dict(r)
        rec['_hit_label'] = False  # dummy
        rec['_date'] = ''
        temp_records.append(rec)

    if not temp_records:
        return 0

    X, _, _ = engineer_features(temp_records)

    # Get predictions from each model
    n = X.shape[0]
    model_preds = np.zeros((n, len(model_names)))

    for m_idx, name in enumerate(model_names):
        if name not in models:
            model_preds[:, m_idx] = 0.5
            continue

        try:
            probs = _predict_proba_safe(models[name], X, name)
            model_preds[:, m_idx] = probs

            # Set individual model probability on each prop
            short_name = name.replace('_', '')
            key = f'{name}_arena_prob'
            for r, p in zip(results, probs):
                r[key] = round(float(p), 4)

        except Exception as e:
            print(f"  WARNING: Arena model {name} scoring failed: {e}")
            model_preds[:, m_idx] = 0.5

    # Meta-learner blend
    try:
        scaled = meta_scaler.transform(model_preds)
        ensemble_probs = meta_learner.predict_proba(scaled)[:, 1]
    except Exception as e:
        print(f"  WARNING: Arena meta-learner failed: {e}")
        # Fallback: simple average
        ensemble_probs = model_preds.mean(axis=1)

    scored = 0
    for r, prob in zip(results, ensemble_probs):
        r['arena_prob'] = round(float(prob), 4)
        scored += 1

    return scored


# =====================================================================
# TRAINING PIPELINE
# =====================================================================

def train_arena(board_file=None, max_models=5):
    """Full arena pipeline: load data, train 10 models, evaluate, select, build ensemble.

    Args:
        board_file: path to board JSON for focused training (optional)
        max_models: max number of models to keep in ensemble (default 5)

    Returns:
        (ensemble_metrics, eval_results) or (None, None) on failure.
    """
    t_start = time.time()
    print("=" * 72)
    print("  MODEL ARENA -- Multi-Model Training and Evaluation")
    print("=" * 72)

    # Load data
    print("\n  Loading training data...")
    records, source_label = _load_data(board_file)
    if records is None:
        return None, None

    print(f"  Source: {source_label}, {len(records):,} records")

    # Prepare features and split
    X_train, y_train, X_test, y_test, sw_train = _prepare_data(records, source_label)

    if len(y_test) < 20:
        print(f"  ERROR: Test set too small ({len(y_test)} samples). Need at least 20.")
        return None, None

    # Build model registry
    print("\n  Building model registry...")
    models, skipped = _build_model_registry()
    print(f"  Available: {len(models)} models")
    if skipped:
        print(f"  Skipped:")
        for name, reason in skipped:
            print(f"    - {name}: {reason}")

    # Evaluate all models
    print("\n  Evaluating all models on test set...")
    eval_results = evaluate_models(models, X_train, y_train, X_test, y_test, sw_train)

    if not eval_results:
        print("  ERROR: No models trained successfully")
        return None, None

    # Select top models
    top_results = select_top_models(eval_results, max_models=max_models)

    if len(top_results) < 2:
        print(f"  WARNING: Only {len(top_results)} model(s) survived. "
              f"Need at least 2 for stacking. Using simple average.")

    # Print results table
    print_results_table(eval_results, y_test)

    # Build stacking ensemble
    if len(top_results) >= 2:
        meta_learner, meta_scaler, final_models, ensemble_metrics = build_ensemble(
            top_results, X_train, y_train, X_test, y_test, sw_train
        )
    else:
        # Fallback: single model, no stacking
        final_models = {top_results[0]['name']: top_results[0]['model']}
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler as SS
        meta_learner = LogisticRegression(random_state=42)
        meta_scaler = SS()
        # Fit trivial meta-learner on single model probs
        single_probs = top_results[0]['y_prob'].reshape(-1, 1)
        meta_scaler.fit(single_probs)
        meta_learner.fit(meta_scaler.transform(single_probs), y_test)
        ensemble_metrics = {
            'brier': top_results[0]['brier'],
            'auc': top_results[0]['auc'],
            'accuracy': top_results[0]['accuracy'],
            'cal_error': top_results[0]['cal_error'],
            'cal_bins': top_results[0].get('cal_bins', []),
            'model_names': [top_results[0]['name']],
            'meta_coefs': {top_results[0]['name']: 1.0},
        }

    # Save everything
    _save_arena(meta_learner, meta_scaler, final_models, ensemble_metrics, eval_results)

    # Final summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"  ARENA COMPLETE ({elapsed:.1f}s)")
    print(f"  Ensemble: {len(final_models)} models -> Brier={ensemble_metrics['brier']:.4f} "
          f"AUC={ensemble_metrics['auc']:.3f}")
    print(f"  Models: {list(final_models.keys())}")
    print(f"  Saved to: {ARENA_DIR}/")
    print("=" * 72)

    return ensemble_metrics, eval_results


# =====================================================================
# EVALUATION ONLY (no retrain)
# =====================================================================

def eval_arena(board_file=None):
    """Run evaluation only -- train all models but don't save ensemble.

    Useful for comparing models without overwriting the production ensemble.
    """
    print("=" * 72)
    print("  MODEL ARENA -- Evaluation Only")
    print("=" * 72)

    records, source_label = _load_data(board_file)
    if records is None:
        return None

    X_train, y_train, X_test, y_test, sw_train = _prepare_data(records, source_label)

    if len(y_test) < 20:
        print(f"  ERROR: Test set too small ({len(y_test)} samples)")
        return None

    models, skipped = _build_model_registry()
    if skipped:
        print(f"  Skipped: {[s[0] for s in skipped]}")

    eval_results = evaluate_models(models, X_train, y_train, X_test, y_test, sw_train)

    # Mark top 5 for reference
    for i, r in enumerate(eval_results):
        if r['auc'] > 0.52 and i < 5:
            r['status'] = 'KEEP'
        else:
            r['status'] = 'KILL' + (' (AUC)' if r['auc'] <= 0.52 else ' (rank)')

    print_results_table(eval_results, y_test)

    # Print calibration details for top 3
    print("\n  Calibration Detail (top 3 models):")
    for r in eval_results[:3]:
        print(f"\n    {r['name']}:")
        for b in r.get('cal_bins', []):
            bar = '#' * int(b['avg_actual'] * 40)
            print(f"      [{b['bin_low']:.1f}-{b['bin_high']:.1f}] "
                  f"pred={b['avg_predicted']:.3f} actual={b['avg_actual']:.3f} "
                  f"err={b['abs_error']:.3f} n={b['n']:4d}  {bar}")

    return eval_results


# =====================================================================
# LEADERBOARD
# =====================================================================

def show_leaderboard():
    """Display current arena leaderboard from saved metadata."""
    if not os.path.exists(ARENA_META_PATH):
        print("  No arena metadata found. Run: python3 model_arena.py train")
        return

    with open(ARENA_META_PATH) as f:
        meta = json.load(f)

    print("\n" + "=" * 72)
    print("  MODEL ARENA LEADERBOARD")
    print("=" * 72)
    print(f"  Trained: {meta.get('trained_at', '?')}")
    print(f"  Features: {meta.get('n_features', '?')}")

    # Ensemble metrics
    em = meta.get('ensemble_metrics', {})
    print(f"\n  ENSEMBLE ({len(meta.get('model_names', []))} models):")
    print(f"    Brier:    {em.get('brier', '?')}")
    print(f"    AUC:      {em.get('auc', '?')}")
    print(f"    Accuracy: {em.get('accuracy', '?')}")
    print(f"    CalErr:   {em.get('cal_error', '?')}")

    # Meta-learner weights
    coefs = em.get('meta_coefs', {})
    if coefs:
        print(f"\n    Meta-learner weights:")
        for name, coef in sorted(coefs.items(), key=lambda x: -abs(x[1])):
            print(f"      {name:18s} {coef:+.4f}")

    # Individual models
    individual = meta.get('individual_results', [])
    if individual:
        print(f"\n  Individual Model Rankings:")
        print(f"  {'#':>4s}  {'Model':18s}  {'Brier':>7s}  {'AUC':>6s}  {'Acc':>6s}  "
              f"{'CalErr':>7s}  {'Time':>6s}  Status")
        print(f"  {'-'*4}  {'-'*18}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*10}")
        for i, r in enumerate(individual):
            print(f"  {i+1:>4d}  {r['name']:18s}  {r['brier']:.4f}  {r['auc']:.3f}  "
                  f"{r['accuracy']:.1%}  {r['cal_error']:.4f}  {r['train_time']:5.1f}s  "
                  f"{r.get('status', '')}")

    # Calibration bins for ensemble
    cal_bins = meta.get('ensemble_cal_bins', [])
    if cal_bins:
        print(f"\n  Ensemble Calibration:")
        for b in cal_bins:
            bar = '#' * int(b['avg_actual'] * 40)
            print(f"    [{b['bin_low']:.1f}-{b['bin_high']:.1f}] "
                  f"pred={b['avg_predicted']:.3f} actual={b['avg_actual']:.3f} "
                  f"err={b['abs_error']:.3f} n={b['n']:4d}  {bar}")

    print("\n" + "=" * 72)


# =====================================================================
# CLI
# =====================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    # Parse --board flag
    board_file = None
    for i, arg in enumerate(sys.argv):
        if arg == '--board' and i + 1 < len(sys.argv):
            board_file = sys.argv[i + 1]

    if command == 'train':
        ensemble_metrics, eval_results = train_arena(board_file=board_file)
        if ensemble_metrics is None:
            sys.exit(1)

    elif command == 'eval':
        eval_results = eval_arena(board_file=board_file)
        if eval_results is None:
            sys.exit(1)

    elif command == 'leaderboard':
        show_leaderboard()

    else:
        print(f"  Unknown command: {command}")
        print("  Available: train, eval, leaderboard")
        sys.exit(1)


if __name__ == '__main__':
    main()

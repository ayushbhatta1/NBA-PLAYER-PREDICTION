#!/usr/bin/env python3
"""
Secondary Model Suite — RF, CatBoost, KNN, LogReg
Trained on same data + features as XGBoost, with walk-forward CV.

Each model has a specific role:
  - RF: Non-linear pattern ensemble (pruned, max 20 features)
  - CatBoost: Gradient boost with native categorical handling
  - KNN: Similarity-based with recency weighting
  - LogReg: Calibration layer for probability outputs

Usage:
    python3 predictions/secondary_models.py train          # Train all 4 models
    python3 predictions/secondary_models.py train rf       # Train one model
    python3 predictions/secondary_models.py eval           # Walk-forward CV only
    python3 predictions/secondary_models.py score <file>   # Score a board/results file
"""

import json
import os
import sys
import pickle
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore')

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PREDICTIONS_DIR)

from xgb_model import (
    collect_all_training_data,
    engineer_features,
    FEATURE_COLS,
    _compute_auc,
    _compute_logloss,
    _compute_brier,
)

CACHE_DIR = os.path.join(PREDICTIONS_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD CV (shared across all models)
# ═══════════════════════════════════════════════════════════════

def walk_forward_cv(X, y, dates, sources, model_factory, model_name, sample_weights=None, impute=False):
    """Walk-forward CV: train on past, test on each graded day.

    No data leakage: historical/backfill always in train, graded splits chronologically.
    """
    dates_arr = np.array(dates)
    sources_arr = np.array(sources)

    graded_mask = sources_arr == 'graded'
    graded_dates = sorted(set(d for d, s in zip(dates, sources_arr) if s == 'graded' and d >= '2026-'))
    historical_mask = np.array([d < '2026-' and s == 'historical' for d, s in zip(dates, sources_arr)])
    backfill_mask = np.isin(sources_arr, ['backfill', 'sgo_backfill'])

    if len(graded_dates) < 2:
        print(f"  {model_name}: Need 2+ graded days for CV")
        return []

    # Impute NaN if model can't handle it
    X_clean = X.copy()
    if impute:
        col_medians = np.nanmedian(X_clean, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0, col_medians)
        for col in range(X_clean.shape[1]):
            mask = np.isnan(X_clean[:, col])
            X_clean[mask, col] = col_medians[col]

    folds = []
    for i in range(1, len(graded_dates)):
        train_graded_dates = set(graded_dates[:i])
        test_date = graded_dates[i]

        train_graded_mask = np.array([d in train_graded_dates and s == 'graded' for d, s in zip(dates, sources_arr)])
        train_backfill_mask = np.array([d < test_date and s in ('backfill', 'sgo_backfill') for d, s in zip(dates, sources_arr)])
        train_mask = historical_mask | train_graded_mask | train_backfill_mask
        test_mask = (dates_arr == test_date) & graded_mask

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        X_train, y_train = X_clean[train_mask], y[train_mask]
        X_test, y_test = X_clean[test_mask], y[test_mask]
        sw_train = sample_weights[train_mask] if sample_weights is not None else None

        try:
            model = model_factory()
            if sw_train is not None and hasattr(model, 'fit'):
                # Not all models support sample_weight
                try:
                    model.fit(X_train, y_train, sample_weight=sw_train)
                except TypeError:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            accuracy = (y_pred == y_test).mean()
            auc = _compute_auc(y_test, y_prob)

            top_mask = y_prob >= np.percentile(y_prob, 90)
            bot_mask = y_prob <= np.percentile(y_prob, 10)
            top_hr = y_test[top_mask].mean() if top_mask.sum() > 0 else np.nan
            bot_hr = y_test[bot_mask].mean() if bot_mask.sum() > 0 else np.nan

            folds.append({
                'test_date': test_date,
                'test_size': int(test_mask.sum()),
                'accuracy': float(accuracy),
                'auc': float(auc),
                'top_decile_hr': float(top_hr) if not np.isnan(top_hr) else None,
                'bot_decile_hr': float(bot_hr) if not np.isnan(bot_hr) else None,
                'y_test': y_test,
                'y_prob': y_prob,
            })

            print(f"    {test_date}: AUC={auc:.3f} Acc={accuracy:.3f} "
                  f"Top10%={top_hr:.1%} Bot10%={bot_hr:.1%} (N={test_mask.sum()})")

        except Exception as e:
            print(f"    {test_date}: FAILED — {e}")
            continue

    if len(folds) >= 2:
        all_y = np.concatenate([f['y_test'] for f in folds])
        all_p = np.concatenate([f['y_prob'] for f in folds])
        pooled_auc = _compute_auc(all_y, all_p)
        p90 = np.percentile(all_p, 90)
        p10 = np.percentile(all_p, 10)
        top10 = all_y[all_p >= p90].mean() if (all_p >= p90).sum() > 0 else np.nan
        bot10 = all_y[all_p <= p10].mean() if (all_p <= p10).sum() > 0 else np.nan
        print(f"\n    Pooled: AUC={pooled_auc:.3f}, Top10%={top10:.1%}, Bot10%={bot10:.1%}, N={len(all_y)}")

    return folds


# ═══════════════════════════════════════════════════════════════
# MODEL 1: RANDOM FOREST (pruned, top features only)
# ═══════════════════════════════════════════════════════════════

def get_rf_top_features(n=30):
    """Get top N features from XGBoost importance for RF pruning."""
    meta_path = os.path.join(PREDICTIONS_DIR, 'xgb_model_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        top = [name for name, _ in meta.get('top_features', [])[:n]]
        if top:
            return top
    # Fallback: use known top features
    return [
        'miss_streak', 'directional_gap', 'l10_avg_pf', 'effective_gap',
        'is_b2b_binary', 'games_used', 'cold_under_x_gap', 'travel_distance_last_game',
        'game_total_signal', 'streak_x_direction', 'usage_trend', 'l10_std',
        'under_x_cold', 'direction_binary', 'hr_confidence', 'abs_gap',
        'l10_hit_rate', 'season_hit_rate', 'l5_hit_rate', 'mins_30plus_pct',
        'gap_x_hr', 'l10_miss_count', 'spread', 'stat_ordinal',
        'margin_over_line', 'season_vs_line', 'l5_vs_l10', 'projection_confidence',
        'l10_median', 'total_adjustment',
    ]


def train_rf(X, y, dates, sources, sample_weights=None, feature_indices=None):
    """Train Random Forest — pruned to top features, limited depth."""
    from sklearn.ensemble import RandomForestClassifier

    X_rf = X[:, feature_indices] if feature_indices is not None else X

    # Impute NaN (RF can't handle it)
    col_medians = np.nanmedian(X_rf, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0, col_medians)
    X_clean = X_rf.copy()
    for col in range(X_clean.shape[1]):
        mask = np.isnan(X_clean[:, col])
        X_clean[mask, col] = col_medians[col]

    print(f"\n  Random Forest — {X_clean.shape[1]} features (pruned from {X.shape[1]})")

    def rf_factory():
        return RandomForestClassifier(
            n_estimators=300,       # Enough trees for stability
            max_depth=8,            # Prevent overfitting (was unlimited = 72MB)
            min_samples_leaf=20,    # Regularization
            max_features='sqrt',    # Classic RF — sqrt(n_features)
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )

    # Walk-forward CV
    folds = walk_forward_cv(X_clean, y, dates, sources, rf_factory, "RF",
                           sample_weights=sample_weights, impute=False)

    # Train final model
    model = rf_factory()
    if sample_weights is not None:
        model.fit(X_clean, y, sample_weight=sample_weights)
    else:
        model.fit(X_clean, y)

    # Feature importance
    if feature_indices is not None:
        feat_names = [FEATURE_COLS[i] for i in feature_indices]
    else:
        feat_names = FEATURE_COLS
    importance = sorted(zip(feat_names, model.feature_importances_), key=lambda x: x[1], reverse=True)

    # Save
    save_path = os.path.join(CACHE_DIR, 'rf_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_indices': feature_indices,
            'col_medians': col_medians,
            'feature_names': feat_names,
        }, f)

    avg_auc = np.mean([f['auc'] for f in folds]) if folds else 0
    avg_acc = np.mean([f['accuracy'] for f in folds]) if folds else 0

    meta = {
        'trained_at': datetime.now().isoformat(),
        'n_samples': int(len(y)),
        'n_features': int(X_clean.shape[1]),
        'cv_avg_auc': float(avg_auc),
        'cv_avg_accuracy': float(avg_acc),
        'top_features': [(n, float(v)) for n, v in importance[:15]],
        'model_size_kb': os.path.getsize(save_path) / 1024,
    }
    with open(save_path.replace('.pkl', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  RF saved: {save_path} ({meta['model_size_kb']:.0f} KB)")
    print(f"  CV: AUC={avg_auc:.3f}, Acc={avg_acc:.3f}")
    print(f"  Top 5 features: {', '.join(n for n, _ in importance[:5])}")

    return model, meta


# ═══════════════════════════════════════════════════════════════
# MODEL 2: CATBOOST (native categorical handling)
# ═══════════════════════════════════════════════════════════════

def train_catboost(X, y, dates, sources, sample_weights=None):
    """Train CatBoost — uses different boosting strategy than XGBoost."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  CatBoost not installed. Run: pip3 install catboost")
        return None, None

    # Impute NaN
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0, col_medians)
    X_clean = X.copy()
    for col in range(X_clean.shape[1]):
        mask = np.isnan(X_clean[:, col])
        X_clean[mask, col] = col_medians[col]

    # Identify categorical feature indices (stat_ordinal, direction_binary, etc)
    cat_features_names = ['stat_ordinal', 'direction_binary', 'is_b2b_binary', 'is_combo', 'is_home']
    cat_indices = [FEATURE_COLS.index(f) for f in cat_features_names if f in FEATURE_COLS]

    print(f"\n  CatBoost — {X_clean.shape[1]} features, {len(cat_indices)} categorical")

    def cb_factory():
        return CatBoostClassifier(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=0,
            eval_metric='AUC',
            early_stopping_rounds=50,
        )

    # Walk-forward CV
    folds = walk_forward_cv(X_clean, y, dates, sources, cb_factory, "CatBoost",
                           sample_weights=sample_weights, impute=False)

    # Train final model
    model = cb_factory()
    if sample_weights is not None:
        model.fit(X_clean, y, sample_weight=sample_weights, verbose=0)
    else:
        model.fit(X_clean, y, verbose=0)

    # Feature importance
    importance = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: x[1], reverse=True)

    # Save
    save_path = os.path.join(CACHE_DIR, 'catboost_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'col_medians': col_medians,
            'cat_indices': cat_indices,
        }, f)

    avg_auc = np.mean([f['auc'] for f in folds]) if folds else 0
    avg_acc = np.mean([f['accuracy'] for f in folds]) if folds else 0

    meta = {
        'trained_at': datetime.now().isoformat(),
        'n_samples': int(len(y)),
        'n_features': int(X_clean.shape[1]),
        'cv_avg_auc': float(avg_auc),
        'cv_avg_accuracy': float(avg_acc),
        'top_features': [(n, float(v)) for n, v in importance[:15]],
        'model_size_kb': os.path.getsize(save_path) / 1024,
    }
    with open(save_path.replace('.pkl', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  CatBoost saved: {save_path} ({meta['model_size_kb']:.0f} KB)")
    print(f"  CV: AUC={avg_auc:.3f}, Acc={avg_acc:.3f}")
    print(f"  Top 5 features: {', '.join(n for n, _ in importance[:5])}")

    return model, meta


# ═══════════════════════════════════════════════════════════════
# MODEL 3: KNN (recency-weighted, compact)
# ═══════════════════════════════════════════════════════════════

def train_knn(X, y, dates, sources, sample_weights=None):
    """Train KNN — distance-weighted with recency bias via StandardScaler on recent data."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    # Impute + scale (critical for KNN — distance-based)
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0, col_medians)
    X_clean = X.copy()
    for col in range(X_clean.shape[1]):
        mask = np.isnan(X_clean[:, col])
        X_clean[mask, col] = col_medians[col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # Use only top features (KNN suffers from curse of dimensionality)
    top_feats = get_rf_top_features(20)
    feat_indices = [FEATURE_COLS.index(f) for f in top_feats if f in FEATURE_COLS]
    X_knn = X_scaled[:, feat_indices]

    print(f"\n  KNN — {X_knn.shape[1]} features (pruned from {X.shape[1]}), scaled")

    def knn_factory():
        return KNeighborsClassifier(
            n_neighbors=31,          # Odd number, enough for stable vote
            weights='distance',       # Closer neighbors have more influence
            metric='euclidean',
            algorithm='ball_tree',    # Compact index (not brute force)
            n_jobs=-1,
        )

    # Walk-forward CV with scaled+pruned features
    dates_arr = np.array(dates)
    sources_arr = np.array(sources)
    graded_dates = sorted(set(d for d, s in zip(dates, sources_arr) if s == 'graded' and d >= '2026-'))
    historical_mask = np.array([d < '2026-' and s == 'historical' for d, s in zip(dates, sources_arr)])
    graded_mask = sources_arr == 'graded'

    folds = []
    for i in range(1, len(graded_dates)):
        train_graded_dates = set(graded_dates[:i])
        test_date = graded_dates[i]

        train_graded_m = np.array([d in train_graded_dates and s == 'graded' for d, s in zip(dates, sources_arr)])
        train_backfill_m = np.array([d < test_date and s in ('backfill', 'sgo_backfill') for d, s in zip(dates, sources_arr)])
        train_mask = historical_mask | train_graded_m | train_backfill_m
        test_mask = (dates_arr == test_date) & graded_mask

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        # Recency weighting for KNN: recent samples get higher weight
        # by duplicating/upweighting recent data in the training set
        X_tr, y_tr = X_knn[train_mask], y[train_mask]

        # Subsample to max 30K for KNN speed (prefer recent)
        if len(y_tr) > 30000:
            # Keep all graded, sample from rest
            tr_sources = sources_arr[train_mask]
            graded_idx = np.where(tr_sources == 'graded')[0]
            other_idx = np.where(tr_sources != 'graded')[0]
            n_other = min(len(other_idx), 30000 - len(graded_idx))
            rng = np.random.RandomState(42)
            sampled_other = rng.choice(other_idx, n_other, replace=False)
            keep_idx = np.concatenate([graded_idx, sampled_other])
            X_tr, y_tr = X_tr[keep_idx], y_tr[keep_idx]

        try:
            model = knn_factory()
            model.fit(X_tr, y_tr)

            y_prob = model.predict_proba(X_knn[test_mask])[:, 1]
            y_test = y[test_mask]
            accuracy = ((y_prob >= 0.5).astype(int) == y_test).mean()
            auc = _compute_auc(y_test, y_prob)

            top_m = y_prob >= np.percentile(y_prob, 90)
            bot_m = y_prob <= np.percentile(y_prob, 10)
            top_hr = y_test[top_m].mean() if top_m.sum() > 0 else np.nan
            bot_hr = y_test[bot_m].mean() if bot_m.sum() > 0 else np.nan

            folds.append({
                'test_date': test_date,
                'test_size': int(test_mask.sum()),
                'accuracy': float(accuracy),
                'auc': float(auc),
                'top_decile_hr': float(top_hr) if not np.isnan(top_hr) else None,
                'bot_decile_hr': float(bot_hr) if not np.isnan(bot_hr) else None,
            })
            print(f"    {test_date}: AUC={auc:.3f} Acc={accuracy:.3f} "
                  f"Top10%={top_hr:.1%} Bot10%={bot_hr:.1%} (N={test_mask.sum()})")
        except Exception as e:
            print(f"    {test_date}: FAILED — {e}")

    # Train final on all data (subsampled)
    if len(X_knn) > 30000:
        rng = np.random.RandomState(42)
        graded_idx = np.where(sources_arr == 'graded')[0]
        other_idx = np.where(sources_arr != 'graded')[0]
        n_other = min(len(other_idx), 30000 - len(graded_idx))
        sampled = rng.choice(other_idx, n_other, replace=False)
        keep = np.concatenate([graded_idx, sampled])
        X_final, y_final = X_knn[keep], y[keep]
    else:
        X_final, y_final = X_knn, y

    model = knn_factory()
    model.fit(X_final, y_final)

    save_path = os.path.join(CACHE_DIR, 'knn_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'col_medians': col_medians,
            'feature_indices': feat_indices,
        }, f)

    avg_auc = np.mean([f['auc'] for f in folds]) if folds else 0
    avg_acc = np.mean([f['accuracy'] for f in folds]) if folds else 0

    if folds:
        print(f"\n    Pooled: AUC={avg_auc:.3f}, Acc={avg_acc:.3f}")

    meta = {
        'trained_at': datetime.now().isoformat(),
        'n_samples': int(len(y_final)),
        'n_features': int(X_knn.shape[1]),
        'cv_avg_auc': float(avg_auc),
        'cv_avg_accuracy': float(avg_acc),
        'model_size_kb': os.path.getsize(save_path) / 1024,
    }
    with open(save_path.replace('.pkl', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  KNN saved: {save_path} ({meta['model_size_kb']:.0f} KB)")
    print(f"  CV: AUC={avg_auc:.3f}, Acc={avg_acc:.3f}")

    return model, meta


# ═══════════════════════════════════════════════════════════════
# MODEL 4: LOGISTIC REGRESSION (calibration layer)
# ═══════════════════════════════════════════════════════════════

def train_logreg(X, y, dates, sources, sample_weights=None):
    """Train LogReg — meant as calibration layer, not standalone predictor."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Impute + scale
    col_medians = np.nanmedian(X, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0, col_medians)
    X_clean = X.copy()
    for col in range(X_clean.shape[1]):
        mask = np.isnan(X_clean[:, col])
        X_clean[mask, col] = col_medians[col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    print(f"\n  LogReg — {X_scaled.shape[1]} features, scaled")

    def lr_factory():
        return LogisticRegression(
            C=0.1,                   # Strong regularization
            penalty='l1',            # L1 for feature selection
            solver='saga',
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
        )

    folds = walk_forward_cv(X_scaled, y, dates, sources, lr_factory, "LogReg",
                           sample_weights=sample_weights, impute=False)

    # Train final
    model = lr_factory()
    if sample_weights is not None:
        model.fit(X_scaled, y, sample_weight=sample_weights)
    else:
        model.fit(X_scaled, y)

    # Feature coefficients (L1 gives sparse)
    nonzero = np.sum(model.coef_[0] != 0)
    coefs = sorted(zip(FEATURE_COLS, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True)

    save_path = os.path.join(CACHE_DIR, 'logreg_model.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'col_medians': col_medians,
        }, f)

    avg_auc = np.mean([f['auc'] for f in folds]) if folds else 0
    avg_acc = np.mean([f['accuracy'] for f in folds]) if folds else 0

    meta = {
        'trained_at': datetime.now().isoformat(),
        'n_samples': int(len(y)),
        'n_features': int(X_scaled.shape[1]),
        'n_nonzero_features': int(nonzero),
        'cv_avg_auc': float(avg_auc),
        'cv_avg_accuracy': float(avg_acc),
        'top_features': [(n, float(v)) for n, v in coefs[:15]],
        'model_size_kb': os.path.getsize(save_path) / 1024,
    }
    with open(save_path.replace('.pkl', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  LogReg saved: {save_path} ({meta['model_size_kb']:.0f} KB)")
    print(f"  CV: AUC={avg_auc:.3f}, Acc={avg_acc:.3f}")
    print(f"  Non-zero features: {nonzero}/{X_scaled.shape[1]}")
    print(f"  Top 5: {', '.join(f'{n}({v:+.3f})' for n, v in coefs[:5])}")

    return model, meta


# ═══════════════════════════════════════════════════════════════
# SCORING (used by run_board_v5.py)
# ═══════════════════════════════════════════════════════════════

def score_props(results, models=None):
    """Score all props with all 4 secondary models.

    Args:
        results: list of prop dicts (already has features from analyze_v3)
        models: list of model names to score (default: all available)

    Returns:
        results with rf_prob, catboost_prob, knn_prob, logreg_prob added
    """
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

    model_configs = {
        'rf': ('rf_model.pkl', 'rf_prob'),
        'catboost': ('catboost_model.pkl', 'catboost_prob'),
        'knn': ('knn_model.pkl', 'knn_prob'),
        'logreg': ('logreg_model.pkl', 'logreg_prob'),
    }

    if models is None:
        models = list(model_configs.keys())

    for name in models:
        if name not in model_configs:
            continue

        filename, prob_key = model_configs[name]
        model_path = os.path.join(CACHE_DIR, filename)

        if not os.path.exists(model_path):
            continue

        try:
            with open(model_path, 'rb') as f:
                bundle = pickle.load(f)

            model = bundle['model']
            col_medians = bundle.get('col_medians')

            # Prepare features
            X_score = X.copy()

            # Feature pruning (RF, KNN)
            feat_indices = bundle.get('feature_indices')

            # Impute NaN
            if col_medians is not None:
                if feat_indices is not None:
                    X_score = X_score[:, feat_indices]
                    for col in range(X_score.shape[1]):
                        mask = np.isnan(X_score[:, col])
                        if col < len(col_medians):
                            X_score[mask, col] = col_medians[col]
                else:
                    for col in range(X_score.shape[1]):
                        mask = np.isnan(X_score[:, col])
                        if col < len(col_medians):
                            X_score[mask, col] = col_medians[col]

            # Scale (KNN, LogReg)
            scaler = bundle.get('scaler')
            if scaler is not None:
                X_score = scaler.transform(X_score)

            probs = model.predict_proba(X_score)[:, 1]

            scored = 0
            for r, prob in zip(results, probs):
                r[prob_key] = round(float(prob), 4)
                scored += 1

            print(f"  {name.upper()}: {scored}/{len(results)} scored")

        except Exception as e:
            print(f"  {name.upper()}: Failed — {e}")

    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else 'all'

    if command in ('train', 'eval'):
        print("=" * 60)
        print(f"  Secondary Models — {'Training' if command == 'train' else 'Evaluation'}")
        print("=" * 60)

        # Load all training data (same as XGBoost)
        records = collect_all_training_data(use_historical=True)
        print(f"\n  Total records: {len(records)}")

        X, y, dates = engineer_features(records)
        print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
        print(f"  Hit rate: {y.mean():.1%}")

        # Sample weights (same as XGBoost)
        def _weight(r):
            src = r.get('_data_source', '')
            if src == 'graded': return 25.0
            if src == 'backfill': return 10.0
            if src == 'sgo_backfill': return 8.0
            return 1.0
        sample_weights = np.array([_weight(r) for r in records])
        sources = [r.get('_data_source', 'graded') for r in records]

        results = {}

        # ── Random Forest ──
        if target in ('all', 'rf'):
            top_feats = get_rf_top_features(30)
            feat_indices = [FEATURE_COLS.index(f) for f in top_feats if f in FEATURE_COLS]
            _, meta = train_rf(X, y, dates, sources, sample_weights, feat_indices)
            if meta:
                results['rf'] = meta

        # ── CatBoost ──
        if target in ('all', 'catboost', 'cb'):
            _, meta = train_catboost(X, y, dates, sources, sample_weights)
            if meta:
                results['catboost'] = meta

        # ── KNN ──
        if target in ('all', 'knn'):
            _, meta = train_knn(X, y, dates, sources, sample_weights)
            if meta:
                results['knn'] = meta

        # ── Logistic Regression ──
        if target in ('all', 'logreg', 'lr'):
            _, meta = train_logreg(X, y, dates, sources, sample_weights)
            if meta:
                results['logreg'] = meta

        # ── Summary ──
        if results:
            print("\n" + "=" * 60)
            print("  SUMMARY — Secondary Model Comparison")
            print("=" * 60)
            print(f"  {'Model':<12} {'AUC':>6} {'Acc':>6} {'Size':>8} {'Features':>8}")
            print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
            for name, m in sorted(results.items(), key=lambda x: x[1].get('cv_avg_auc', 0), reverse=True):
                print(f"  {name:<12} {m.get('cv_avg_auc',0):>6.3f} {m.get('cv_avg_accuracy',0):>6.3f} "
                      f"{m.get('model_size_kb',0):>7.0f}K {m.get('n_features',0):>8}")

    elif command == 'score':
        if len(sys.argv) < 3:
            print("Usage: python3 secondary_models.py score <file>")
            sys.exit(1)

        filepath = sys.argv[2]
        with open(filepath) as f:
            data = json.load(f)

        if isinstance(data, list):
            props = data
        elif isinstance(data, dict) and 'results' in data:
            props = data['results']
        else:
            print("  ERROR: Unrecognized format")
            sys.exit(1)

        props = score_props(props)
        scored = {
            'rf': sum(1 for r in props if 'rf_prob' in r),
            'catboost': sum(1 for r in props if 'catboost_prob' in r),
            'knn': sum(1 for r in props if 'knn_prob' in r),
            'logreg': sum(1 for r in props if 'logreg_prob' in r),
        }
        for name, cnt in scored.items():
            if cnt > 0:
                print(f"  {name}: {cnt}/{len(props)} scored")


if __name__ == '__main__':
    main()

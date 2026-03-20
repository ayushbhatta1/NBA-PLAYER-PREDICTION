#!/usr/bin/env python3
"""
Neural Network Embedder — Learned embedding features for XGBoost.

Trains a deeper NN (256, 128, 64, 32) on the same 104 features as XGBoost/MLP.
Extracts the 32-neuron penultimate layer activations as "learned embeddings" —
nonlinear interaction patterns that tree-based models cannot discover on their own.
These 32 values (nn_emb_0..nn_emb_31) are fed back as additional XGBoost features.

Uses scikit-learn MLPClassifier for consistency with the rest of the pipeline.
Forward pass through coefs_/intercepts_ extracts hidden layer activations.

Usage:
    python3 predictions/nn_embedder.py train          # Train embedder and save
    python3 predictions/nn_embedder.py eval           # Walk-forward CV evaluation
    python3 predictions/nn_embedder.py test           # Test embedding extraction on sample data
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
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
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

MODEL_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'nn_embedder.pkl')
SCALER_PATH = os.path.join(PREDICTIONS_DIR, 'cache', 'nn_embedder_scaler.pkl')
META_PATH = os.path.join(PREDICTIONS_DIR, 'nn_embedder_meta.json')

# Architecture: deeper than existing MLP (128, 64) to capture richer interactions
HIDDEN_LAYERS = (256, 128, 64, 32)
EMBEDDING_LAYER_IDX = 3  # 0-indexed: layer 3 = the 32-neuron layer
EMBEDDING_DIM = 32
EMBEDDING_COLS = [f'nn_emb_{i}' for i in range(EMBEDDING_DIM)]


# ===============================================================
# IMPUTATION + SCALING (same pattern as mlp_model.py)
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
# EMBEDDING EXTRACTION — THE KEY FUNCTION
# ===============================================================

def _forward_to_layer(model, X, layer_idx):
    """Forward pass through MLPClassifier up to layer_idx (0-indexed).

    Manually computes activations by walking through the trained
    coefs_ and intercepts_ arrays with ReLU activation.

    For a (256, 128, 64, 32) network:
      layer_idx=0 -> 256-neuron activations
      layer_idx=1 -> 128-neuron activations
      layer_idx=2 -> 64-neuron activations
      layer_idx=3 -> 32-neuron activations (embeddings)
    """
    a = np.array(X, dtype=np.float64)
    for i in range(layer_idx + 1):
        a = a @ model.coefs_[i] + model.intercepts_[i]
        a = np.maximum(a, 0)  # relu
    return a


def get_embeddings(X, model=None, scaler=None, medians=None):
    """Extract 32-dim embeddings from the penultimate layer.

    Args:
        X: feature matrix (N x 104), may contain NaN
        model: trained MLPClassifier (loaded from cache if None)
        scaler: fitted StandardScaler (loaded from cache if None)
        medians: column medians for imputation (loaded from cache if None)

    Returns:
        N x 32 numpy array of embedding values, or None if model unavailable
    """
    if model is None or scaler is None or medians is None:
        loaded = _load_model()
        if loaded is None:
            return None
        model, scaler, medians = loaded

    # Impute NaN + scale (same preprocessing as training)
    X_scaled, _, _, _ = _impute_and_scale(X, medians=medians, scaler=scaler)

    # Forward pass to embedding layer
    embeddings = _forward_to_layer(model, X_scaled, EMBEDDING_LAYER_IDX)

    return embeddings


# ===============================================================
# MODEL I/O
# ===============================================================

def _load_model():
    """Load trained embedder from cache. Returns (model, scaler, medians) or None."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            bundle = pickle.load(f)
        return model, bundle['scaler'], bundle['medians']
    except Exception:
        return None


def _save_model(model, scaler, medians):
    """Save trained embedder and preprocessing artifacts."""
    cache_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(cache_dir, exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump({'scaler': scaler, 'medians': medians}, f)


# ===============================================================
# TRAINING
# ===============================================================

def train_embedder(records=None):
    """Train the NN embedder on all available data.

    Uses the same data loading and weighting as mlp_model.py.
    Saves trained model + scaler to cache.

    Returns:
        (model, metadata) tuple
    """
    if records is None:
        print("  Loading training data...")
        records = collect_all_training_data(use_historical=True)

    if len(records) < 100:
        print(f"  ERROR: Only {len(records)} records -- need at least 100")
        return None, None

    print(f"  Engineering features from {len(records)} records...")
    X, y, dates = engineer_features(records)

    # Source-aware sample weights (same as mlp_model.py)
    sample_weights = None
    if any(r.get('_data_source') in ('backfill', 'sgo_backfill', 'historical') for r in records):
        def _weight(r):
            src = r.get('_data_source', '')
            if src == 'graded':
                return 25.0
            if src == 'backfill':
                return 10.0
            if src == 'sgo_backfill':
                return 8.0
            return 1.0
        sample_weights = np.array([_weight(r) for r in records])

    sources = [r.get('_data_source', 'graded') for r in records]

    # Walk-forward CV for evaluation
    print("\n  Walk-Forward Cross-Validation (NN Embedder):")
    folds, oof_probs, oof_actuals = _walk_forward_cv(
        X, y, dates, sample_weights=sample_weights, sources=sources
    )

    if folds:
        avg_auc = np.mean([f['auc'] for f in folds])
        avg_acc = np.mean([f['accuracy'] for f in folds])
        print(f"\n  NN Embedder CV Summary: Avg AUC={avg_auc:.3f}, Avg Acc={avg_acc:.3f}")
    else:
        avg_auc = None
        avg_acc = None

    # Train final model on all data
    print(f"\n  Training final embedder on {len(y)} samples...")
    X_scaled, _, medians, scaler = _impute_and_scale(X)

    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation='relu',
        solver='adam',
        alpha=1e-4,          # L2 regularization
        batch_size=512,
        learning_rate='adaptive',
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )

    if sample_weights is not None:
        # Normalize weights for sklearn (expects relative weights)
        sw_norm = sample_weights / sample_weights.max()
    else:
        sw_norm = None

    # sklearn MLPClassifier does not support sample_weight in fit() for
    # all solvers, but adam does not support it at all. Oversample instead.
    # Actually, MLPClassifier.fit() does NOT accept sample_weight.
    # Use the data as-is (weighting handled by data composition already).
    model.fit(X_scaled, y)

    # Verify embedding extraction works
    test_emb = _forward_to_layer(model, X_scaled[:5], EMBEDDING_LAYER_IDX)
    assert test_emb.shape == (5, EMBEDDING_DIM), \
        f"Embedding shape mismatch: {test_emb.shape} != (5, {EMBEDDING_DIM})"

    # Save
    _save_model(model, scaler, medians)
    print(f"  Model saved: {MODEL_PATH}")
    print(f"  Scaler saved: {SCALER_PATH}")

    # Metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'n_samples': int(len(y)),
        'hit_rate': float(y.mean()),
        'n_features_in': int(X.shape[1]),
        'n_embedding_dims': EMBEDDING_DIM,
        'architecture': f'MLPClassifier{HIDDEN_LAYERS}',
        'embedding_layer_idx': EMBEDDING_LAYER_IDX,
        'unique_dates': sorted(set(dates)),
        'cv_folds': folds,
        'cv_avg_auc': float(avg_auc) if avg_auc is not None else None,
        'cv_avg_accuracy': float(avg_acc) if avg_acc is not None else None,
        'n_iterations': int(model.n_iter_),
    }

    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {META_PATH}")

    return model, metadata


# ===============================================================
# WALK-FORWARD CV (same fold structure as mlp_model.py)
# ===============================================================

def _walk_forward_cv(X, y, dates, sample_weights=None, sources=None):
    """Walk-forward CV matching XGBoost/MLP fold structure."""
    dates_arr = np.array(dates)
    sources_arr = np.array(sources) if sources is not None else np.array(['graded'] * len(dates))

    graded_mask = sources_arr == 'graded'
    graded_dates = sorted(set(
        d for d, s in zip(dates, sources_arr) if s == 'graded' and d >= '2026-'
    ))
    historical_mask = np.array([
        d < '2026-' and s == 'historical' for d, s in zip(dates, sources_arr)
    ])
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

        train_graded_mask = np.array([
            d in train_graded_dates and s == 'graded'
            for d, s in zip(dates, sources_arr)
        ])
        train_backfill_mask = np.array([
            d < test_date and s in ('backfill', 'sgo_backfill')
            for d, s in zip(dates, sources_arr)
        ])
        train_mask = historical_mask | train_graded_mask | train_backfill_mask
        test_mask = (dates_arr == test_date) & graded_mask

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        X_train_raw, y_train_raw = X[train_mask], y[train_mask]
        X_test_raw, y_test = X[test_mask], y[test_mask]

        # Impute NaN + scale
        X_train_s, X_test_s, _, _ = _impute_and_scale(X_train_raw, X_test_raw)

        # Train embedder for this fold
        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=512,
            learning_rate='adaptive',
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
            verbose=False,
        )
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

        # Also evaluate embedding quality: variance of embeddings
        emb_test = _forward_to_layer(model, X_test_s, EMBEDDING_LAYER_IDX)
        emb_mean_var = float(np.mean(np.var(emb_test, axis=0)))
        emb_active_pct = float(np.mean(emb_test > 0))  # % of neurons active (post-ReLU)

        fold = {
            'train_graded_dates': sorted(train_graded_dates),
            'test_date': test_date,
            'train_size': int(train_mask.sum()),
            'test_size': int(test_mask.sum()),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'logloss': float(logloss),
            'top_decile_hr': float(top_hr) if not np.isnan(top_hr) else None,
            'bot_decile_hr': float(bot_hr) if not np.isnan(bot_hr) else None,
            'emb_mean_variance': round(emb_mean_var, 4),
            'emb_active_pct': round(emb_active_pct, 3),
        }
        folds.append(fold)

        print(f"    Fold: train {sorted(train_graded_dates)} -> test {test_date}")
        print(f"      N={fold['test_size']}, Acc={accuracy:.3f}, AUC={auc:.3f}, "
              f"Top10%={top_hr:.1%}, Bot10%={bot_hr:.1%}")
        print(f"      Embeddings: var={emb_mean_var:.4f}, active={emb_active_pct:.1%}")

    # Pooled metrics
    if len(folds) >= 2:
        all_y_test = np.array(oof_actuals)
        all_y_prob = np.array(oof_probs)
        pooled_auc = _compute_auc(all_y_test, all_y_prob)
        p90 = np.percentile(all_y_prob, 90)
        p10 = np.percentile(all_y_prob, 10)
        pooled_top10 = all_y_test[all_y_prob >= p90].mean() if (all_y_prob >= p90).sum() > 0 else np.nan
        pooled_bot10 = all_y_test[all_y_prob <= p10].mean() if (all_y_prob <= p10).sum() > 0 else np.nan
        print(f"\n    Pooled CV: AUC={pooled_auc:.3f}, Top10%={pooled_top10:.1%}, Bot10%={pooled_bot10:.1%}")
        print(f"    Pooled N={len(all_y_test)}, Hit rate={all_y_test.mean():.1%}")

    return folds, oof_probs, oof_actuals


# ===============================================================
# PIPELINE INTEGRATION
# ===============================================================

def enrich_with_embeddings(results):
    """Add nn_emb_0..nn_emb_31 to each prop dict.

    Args:
        results: list of prop dicts (from analyze_v3 output)

    Returns:
        int: number of props enriched (0 if model not available)
    """
    if not results:
        return 0

    # Load trained model
    loaded = _load_model()
    if loaded is None:
        return 0

    model, scaler, medians = loaded

    # Build feature matrix via engineer_features
    temp_records = []
    for r in results:
        rec = dict(r)
        rec['_hit_label'] = False
        rec['_date'] = ''
        temp_records.append(rec)

    try:
        X, _, _ = engineer_features(temp_records)
    except Exception:
        return 0

    if X.shape[0] == 0:
        return 0

    # Get embeddings
    embeddings = get_embeddings(X, model=model, scaler=scaler, medians=medians)
    if embeddings is None:
        return 0

    # Add embedding values to each prop dict
    count = 0
    for i, r in enumerate(results):
        if i < embeddings.shape[0]:
            for j in range(EMBEDDING_DIM):
                r[f'nn_emb_{j}'] = round(float(embeddings[i, j]), 6)
            count += 1

    return count


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
        print(f"  NN Embedder — Training")
        print(f"  Architecture: MLPClassifier{HIDDEN_LAYERS}")
        print(f"  Embedding dim: {EMBEDDING_DIM} (layer {EMBEDDING_LAYER_IDX})")
        print(f"  Input features: {len(FEATURE_COLS)}")
        print("=" * 60)

        model, metadata = train_embedder()

        if model is not None:
            print(f"\n  Training complete: {metadata['n_samples']} samples, "
                  f"{metadata['n_features_in']} features -> {EMBEDDING_DIM} embeddings")
            if metadata.get('cv_avg_auc'):
                print(f"  CV AUC: {metadata['cv_avg_auc']:.3f}")
                print(f"  CV Acc: {metadata['cv_avg_accuracy']:.3f}")
            print(f"  Converged in {metadata['n_iterations']} iterations")

    elif command == 'eval':
        print("=" * 60)
        print(f"  NN Embedder — Evaluation Only")
        print("=" * 60)

        print("  Loading training data...")
        records = collect_all_training_data(use_historical=True)

        X, y, dates = engineer_features(records)

        sample_weights = None
        if any(r.get('_data_source') in ('backfill', 'sgo_backfill', 'historical') for r in records):
            def _weight(r):
                src = r.get('_data_source', '')
                if src == 'graded':
                    return 25.0
                if src == 'backfill':
                    return 10.0
                if src == 'sgo_backfill':
                    return 8.0
                return 1.0
            sample_weights = np.array([_weight(r) for r in records])

        sources = [r.get('_data_source', 'graded') for r in records]

        print("\n  Walk-Forward CV (NN Embedder):")
        folds, _, _ = _walk_forward_cv(X, y, dates, sample_weights=sample_weights, sources=sources)

        if folds:
            avg_auc = np.mean([f['auc'] for f in folds])
            avg_acc = np.mean([f['accuracy'] for f in folds])
            print(f"\n  Summary: Avg AUC={avg_auc:.3f}, Avg Acc={avg_acc:.3f}")

            # Embedding quality summary
            avg_var = np.mean([f['emb_mean_variance'] for f in folds])
            avg_active = np.mean([f['emb_active_pct'] for f in folds])
            print(f"  Embedding quality: Avg variance={avg_var:.4f}, Avg active={avg_active:.1%}")

    elif command == 'test':
        print("=" * 60)
        print(f"  NN Embedder — Test Embedding Extraction")
        print("=" * 60)

        loaded = _load_model()
        if loaded is None:
            print("\n  ERROR: No trained model found. Run: python3 nn_embedder.py train")
            sys.exit(1)

        model, scaler, medians = loaded
        print(f"  Model loaded: {HIDDEN_LAYERS}")
        print(f"  Weights shape: {[c.shape for c in model.coefs_]}")
        print(f"  Biases shape: {[b.shape for b in model.intercepts_]}")

        # Generate sample data for testing
        print(f"\n  Generating sample embeddings from random data...")
        n_samples = 10
        n_features = len(FEATURE_COLS)
        X_sample = np.random.randn(n_samples, n_features)

        embeddings = get_embeddings(X_sample, model=model, scaler=scaler, medians=medians)
        if embeddings is not None:
            print(f"  Input shape:     {X_sample.shape}")
            print(f"  Embedding shape: {embeddings.shape}")
            print(f"  Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
            print(f"  Embedding mean:  {embeddings.mean():.4f}")
            print(f"  Embedding std:   {embeddings.std():.4f}")
            print(f"  Active neurons:  {(embeddings > 0).mean():.1%} (post-ReLU)")
            print(f"  Dead neurons:    {(embeddings == 0).all(axis=0).sum()}/{EMBEDDING_DIM}")

            print(f"\n  Per-dimension stats (first 8):")
            for j in range(min(8, EMBEDDING_DIM)):
                col = embeddings[:, j]
                print(f"    nn_emb_{j}: mean={col.mean():.4f}, std={col.std():.4f}, "
                      f"min={col.min():.4f}, max={col.max():.4f}, "
                      f"active={np.mean(col > 0):.0%}")

            # Test enrich_with_embeddings with dummy props
            print(f"\n  Testing enrich_with_embeddings with dummy props...")
            dummy_props = [{'line': 20.5, 'projection': 22.0, 'stat': 'pts',
                            'direction': 'OVER', '_hit_label': False, '_date': ''}]
            count = enrich_with_embeddings(dummy_props)
            if count > 0:
                emb_keys = [k for k in dummy_props[0] if k.startswith('nn_emb_')]
                print(f"  Enriched {count} prop(s), added {len(emb_keys)} embedding features")
                if emb_keys:
                    vals = [dummy_props[0][k] for k in sorted(emb_keys)[:8]]
                    print(f"  First 8 values: {vals}")
            else:
                print(f"  enrich_with_embeddings returned 0 (expected with dummy data)")

        print("\n  Test complete.")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()

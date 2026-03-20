#!/usr/bin/env python3
"""
Deep Neural Network Prop Classifier — Ensemble partner for XGBoost.

PyTorch deep net with techniques from Nielsen's "Neural Networks and Deep Learning":
  - He weight initialization (Ch 3) for ReLU layers
  - Dropout regularization (Ch 3) to prevent overfitting
  - Batch normalization for stable training
  - L2 regularization via weight decay (Ch 3)
  - Learning rate scheduling (ReduceLROnPlateau)
  - Early stopping with patience
  - Mini-ensemble of 3 networks (Ch 6) — average predictions
  - MPS (Apple Silicon GPU) acceleration

Reuses exact same 92 features as XGBoost (engineer_features from xgb_model.py).

Usage:
    python3 predictions/mlp_model.py train          # Train on all data
    python3 predictions/mlp_model.py eval           # Walk-forward CV
    python3 predictions/mlp_model.py score <file>   # Score a board/results file
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
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip3 install torch")
    sys.exit(1)

try:
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

MODEL_PATH = os.path.join(PREDICTIONS_DIR, 'mlp_model.pkl')
META_PATH = os.path.join(PREDICTIONS_DIR, 'mlp_model_meta.json')

# ── DEVICE ──
def _get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# ── HYPERPARAMETERS ──
N_ENSEMBLE = 3          # Number of models in mini-ensemble
HIDDEN_DIMS = [256, 128, 64]  # Hidden layer sizes
DROPOUT_RATES = [0.3, 0.3, 0.2]  # Dropout per layer
BATCH_SIZE = 512
LR_INIT = 1e-3
WEIGHT_DECAY = 1e-4     # L2 regularization (Ch 3)
MAX_EPOCHS = 300
PATIENCE = 20           # Early stopping patience
LR_PATIENCE = 8         # LR scheduler patience
VAL_FRACTION = 0.15


# ===============================================================
# NETWORK ARCHITECTURE
# ===============================================================

class PropNet(nn.Module):
    """Deep feedforward network with BatchNorm + Dropout + He init.

    Architecture from Nielsen Ch 3 (regularization) and Ch 6 (deep nets):
      Input → [BatchNorm → Linear → ReLU → Dropout] × N → Linear → Sigmoid
    """
    def __init__(self, input_dim, hidden_dims=None, dropout_rates=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = HIDDEN_DIMS
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES

        layers = []
        prev_dim = input_dim
        for i, (h_dim, drop) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.BatchNorm1d(prev_dim))
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        # He initialization (Ch 3) — optimal for ReLU activations
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


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
# TRAINING SINGLE MODEL
# ===============================================================

def _train_single(X_train, y_train, sample_weights=None, seed=42, device=None):
    """Train one PropNet with early stopping + LR scheduling.

    Applies Nielsen Ch 3 techniques:
      - Cross-entropy loss (BCEWithLogitsLoss)
      - L2 regularization via weight_decay
      - Early stopping to prevent overfitting
      - Learning rate reduction on plateau
    """
    if device is None:
        device = _get_device()

    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(X_train)
    n_val = max(int(n * VAL_FRACTION), 50)
    perm = np.random.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_t = torch.FloatTensor(X_train[train_idx]).to(device)
    y_t = torch.FloatTensor(y_train[train_idx]).to(device)
    X_v = torch.FloatTensor(X_train[val_idx]).to(device)
    y_v = torch.FloatTensor(y_train[val_idx]).to(device)

    # Sample weights for DataLoader
    if sample_weights is not None:
        w_t = torch.FloatTensor(sample_weights[train_idx]).to(device)
    else:
        w_t = torch.ones(len(train_idx)).to(device)

    model = PropNet(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=LR_PATIENCE, factor=0.5
    )
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        # Mini-batch training
        perm_t = torch.randperm(len(X_t))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(X_t), BATCH_SIZE):
            idx = perm_t[start:start + BATCH_SIZE]
            batch_X = X_t[idx]
            batch_y = y_t[idx]
            batch_w = w_t[idx]

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = (criterion(logits, batch_y) * batch_w).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).mean().item()

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    model.eval()
    return model


def _predict_single(model, X, device=None):
    """Get probabilities from a single trained PropNet."""
    if device is None:
        device = _get_device()
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ===============================================================
# MINI-ENSEMBLE (Ch 6)
# ===============================================================

def _train_ensemble(X_train, y_train, sample_weights=None, n_models=None):
    """Train N_ENSEMBLE models with different seeds, return list of models.

    Nielsen Ch 6: "Ensembles of networks" — multiple models trained
    independently capture different patterns, averaging reduces variance.
    """
    if n_models is None:
        n_models = N_ENSEMBLE
    device = _get_device()
    models = []
    for i in range(n_models):
        model = _train_single(X_train, y_train, sample_weights, seed=42 + i, device=device)
        models.append(model)
    return models


def _predict_ensemble(models, X):
    """Average predictions across ensemble members."""
    device = _get_device()
    all_probs = []
    for model in models:
        model = model.to(device)
        probs = _predict_single(model, X, device)
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


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

        # Impute NaN + scale
        X_train_s, X_test_s, _, _ = _impute_and_scale(X_train_raw, X_test_raw)

        # Normalize weights for BCE loss weighting
        if sw_train is not None:
            sw_norm = sw_train / sw_train.max()
        else:
            sw_norm = None

        # Train mini-ensemble for this fold (use fewer models in CV for speed)
        models = _train_ensemble(X_train_s, y_train_raw, sw_norm, n_models=2)
        y_prob = _predict_ensemble(models, X_test_s)

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
            'train_size': int(train_mask.sum()),
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
    """Apply calibration map to a raw mlp_prob."""
    if not cal_map:
        return raw_prob
    for b in cal_map:
        if b['bin_low'] <= raw_prob < b['bin_high']:
            return b['actual_hr']
    return raw_prob


# ===============================================================
# FULL TRAINING PIPELINE
# ===============================================================

def train_model(X, y, dates, save_path=None, sample_weights=None, sources=None):
    """Train deep ensemble on all data, run walk-forward CV, save."""
    if save_path is None:
        save_path = MODEL_PATH

    device = _get_device()
    print(f"\n  Device: {device}")

    print("\n  Walk-Forward Cross-Validation (Deep Net):")
    folds, oof_probs, oof_actuals = walk_forward_cv(
        X, y, dates, sample_weights=sample_weights, sources=sources
    )

    if folds:
        avg_auc = np.mean([f['auc'] for f in folds])
        avg_acc = np.mean([f['accuracy'] for f in folds])
        print(f"\n  Deep Net CV Summary: Avg AUC={avg_auc:.3f}, Avg Acc={avg_acc:.3f}")
    else:
        avg_auc = None
        avg_acc = None

    # Train final ensemble on all data
    print(f"\n  Training final {N_ENSEMBLE}-model ensemble on {len(y)} samples...")
    X_scaled, _, medians, scaler = _impute_and_scale(X)

    if sample_weights is not None:
        sw_norm = sample_weights / sample_weights.max()
    else:
        sw_norm = None

    models = _train_ensemble(X_scaled, y, sw_norm, n_models=N_ENSEMBLE)

    # Move models to CPU for saving
    cpu_models = []
    for m in models:
        m = m.cpu()
        cpu_models.append(m)

    bundle = {
        'models': cpu_models,
        'scaler': scaler,
        'medians': medians,
        'n_features': X.shape[1],
    }
    with open(save_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"  Deep Net ensemble saved: {save_path}")

    # Build calibration map
    calibration_map = _build_calibration_map(oof_probs, oof_actuals)

    metadata = {
        'trained_at': datetime.now().isoformat(),
        'n_samples': int(len(y)),
        'hit_rate': float(y.mean()),
        'n_features': int(X.shape[1]),
        'unique_dates': sorted(set(dates)),
        'cv_folds': folds,
        'cv_avg_auc': float(avg_auc) if avg_auc is not None else None,
        'cv_avg_accuracy': float(avg_acc) if avg_acc is not None else None,
        'calibration_map': calibration_map,
        'model_path': save_path,
        'architecture': f'{N_ENSEMBLE}x PropNet({HIDDEN_DIMS}, drop={DROPOUT_RATES}, wd={WEIGHT_DECAY})',
        'device': str(device),
    }

    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {META_PATH}")

    return models, metadata


# ===============================================================
# SCORING (used by run_board_v5.py)
# ===============================================================

def score_props(results, model_path=None):
    """Load trained ensemble and add mlp_prob to each result dict."""
    if model_path is None:
        model_path = MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}. Run: python3 mlp_model.py train")

    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)

    # Handle both old sklearn format and new PyTorch format
    if 'models' in bundle:
        models = bundle['models']
    elif 'model' in bundle:
        # Legacy sklearn model — fall back to old scoring
        return _score_props_legacy(results, bundle)
    else:
        raise ValueError("Unrecognized model bundle format")

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

    # Ensemble prediction
    device = _get_device()
    models_on_device = [m.to(device) for m in models]
    probs = _predict_ensemble(models_on_device, X_scaled)

    for r, prob in zip(results, probs):
        raw = round(float(prob), 4)
        r['mlp_prob'] = raw
        if cal_map:
            r['mlp_prob_calibrated'] = round(_calibrate_prob(raw, cal_map), 4)

    return results


def _score_props_legacy(results, bundle):
    """Score using old sklearn MLPClassifier format for backward compat."""
    model = bundle['model']
    scaler = bundle['scaler']
    medians = bundle['medians']

    cal_map = None
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        cal_map = meta.get('calibration_map')

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
        r['mlp_prob'] = raw
        if cal_map:
            r['mlp_prob_calibrated'] = round(_calibrate_prob(raw, cal_map), 4)

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
        print("=" * 60)
        print(f"  Deep Neural Network — Training {'(+historical)' if use_historical else '(graded only)'}")
        print(f"  Architecture: {N_ENSEMBLE}x PropNet({HIDDEN_DIMS})")
        print(f"  Dropout={DROPOUT_RATES}, WD={WEIGHT_DECAY}, LR={LR_INIT}")
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
        models, metadata = train_model(X, y, dates, sample_weights=sample_weights, sources=sources)

        print(f"\n  Training: {metadata['n_samples']} samples, {metadata['n_features']} features")
        if metadata.get('cv_avg_auc'):
            print(f"  CV AUC: {metadata['cv_avg_auc']:.3f}")
            print(f"  CV Acc: {metadata['cv_avg_accuracy']:.3f}")

    elif command == 'eval':
        print("=" * 60)
        print(f"  Deep Neural Network — Evaluation")
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
        print("\n  Walk-Forward CV (Deep Net):")
        folds, _, _ = walk_forward_cv(X, y, dates, sample_weights=sample_weights, sources=sources)

        if folds:
            avg_auc = np.mean([f['auc'] for f in folds])
            avg_acc = np.mean([f['accuracy'] for f in folds])
            print(f"\n  Summary: Avg AUC={avg_auc:.3f}, Avg Acc={avg_acc:.3f}")

    elif command == 'score':
        if len(sys.argv) < 3:
            print("Usage: python3 mlp_model.py score <board_or_results.json>")
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
        scored = sum(1 for r in results if 'mlp_prob' in r)
        print(f"  Scored {scored}/{len(results)} props")

        mlp_probs = [r['mlp_prob'] for r in results if 'mlp_prob' in r]
        if mlp_probs:
            arr = np.array(mlp_probs)
            print(f"  mlp_prob: mean={arr.mean():.3f}, std={arr.std():.3f}, "
                  f"min={arr.min():.3f}, max={arr.max():.3f}")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()

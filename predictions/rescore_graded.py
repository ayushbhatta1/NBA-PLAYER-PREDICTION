#!/usr/bin/env python3
"""
Retroactively score ALL graded data with ALL models.

Problem: Meta-learner was trained with only XGB probs (other models = 0.5 default).
Solution: Score all 6,376 graded records with all 6 models, then retrain meta-learner
on fully-populated data so it learns optimal blend weights.

Usage:
    python3 predictions/rescore_graded.py              # Rescore + retrain meta-learner
    python3 predictions/rescore_graded.py --score-only  # Just rescore, don't retrain
"""

import json
import os
import sys
import glob
import pickle
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings('ignore')

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PREDICTIONS_DIR)

from xgb_model import engineer_features, FEATURE_COLS, _extract_hit_label


def load_all_graded():
    """Load all graded records with labels."""
    graded_files = sorted(
        glob.glob(os.path.join(PREDICTIONS_DIR, '*', '*graded*.json'))
    )

    all_records = []
    seen = set()

    for fpath in graded_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        records = data if isinstance(data, list) else data.get('results', [])
        date_dir = os.path.basename(os.path.dirname(fpath))

        for r in records:
            label = _extract_hit_label(r)
            if label is None:
                continue

            # Dedup
            key = (r.get('player', ''), r.get('stat', ''), r.get('line', 0), date_dir)
            if key in seen:
                continue
            seen.add(key)

            r['_hit_label'] = label
            r['_date'] = date_dir
            all_records.append(r)

    return all_records


def score_with_all_models(records):
    """Score records with all 6 models + sim + regression."""

    # Build feature matrix
    temp = [dict(r) for r in records]
    X, y, dates = engineer_features(temp)
    print(f"  Feature matrix: {X.shape[0]} x {X.shape[1]}")

    # ── 1. XGBoost ──
    try:
        from xgboost import XGBClassifier
        model_path = os.path.join(PREDICTIONS_DIR, 'xgb_model.json')
        if os.path.exists(model_path):
            model = XGBClassifier()
            model.load_model(model_path)
            probs = model.predict_proba(X)[:, 1]
            for r, p in zip(records, probs):
                r['xgb_prob'] = round(float(p), 4)
            print(f"  XGB: {len(records)} scored")
    except Exception as e:
        print(f"  XGB: failed ({e})")

    # ── 2. MLP ──
    try:
        from mlp_model import PropNet
        mlp_path = os.path.join(PREDICTIONS_DIR, 'mlp_model.pkl')
        if os.path.exists(mlp_path):
            import torch

            class PropNetUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if name == 'PropNet':
                        return PropNet
                    return super().find_class(module, name)

            with open(mlp_path, 'rb') as f:
                bundle = PropNetUnpickler(f).load()

            models = bundle['models']
            scaler = bundle['scaler']
            medians = bundle.get('medians')
            n_feat = bundle.get('n_features', X.shape[1])

            X_mlp = X.copy()
            # Impute NaN with medians
            if medians is not None and len(medians) == X_mlp.shape[1]:
                for col in range(X_mlp.shape[1]):
                    mask = np.isnan(X_mlp[:, col])
                    X_mlp[mask, col] = medians[col]
            else:
                col_med = np.nanmedian(X_mlp, axis=0)
                col_med = np.where(np.isnan(col_med), 0, col_med)
                for col in range(X_mlp.shape[1]):
                    mask = np.isnan(X_mlp[:, col])
                    X_mlp[mask, col] = col_med[col]

            X_scaled = scaler.transform(X_mlp)
            X_tensor = torch.FloatTensor(X_scaled)

            all_probs = []
            for net in models:
                net.eval()
                with torch.no_grad():
                    out = net(X_tensor)
                    prob = torch.sigmoid(out).numpy().flatten()
                    all_probs.append(prob)

            avg_probs = np.mean(all_probs, axis=0)
            for r, p in zip(records, avg_probs):
                r['mlp_prob'] = round(float(p), 4)
            print(f"  MLP: {len(records)} scored")
    except Exception as e:
        print(f"  MLP: failed ({e})")

    # ── 3-6. Secondary Models (RF, CatBoost, KNN, LogReg) ──
    model_configs = {
        'rf': ('cache/rf_model.pkl', 'rf_prob'),
        'catboost': ('cache/catboost_model.pkl', 'catboost_prob'),
        'knn': ('cache/knn_model.pkl', 'knn_prob'),
        'logreg': ('cache/logreg_model.pkl', 'logreg_prob'),
    }

    for name, (rel_path, prob_key) in model_configs.items():
        model_path = os.path.join(PREDICTIONS_DIR, rel_path)
        if not os.path.exists(model_path):
            print(f"  {name.upper()}: no model file")
            continue

        try:
            with open(model_path, 'rb') as f:
                bundle = pickle.load(f)

            model = bundle['model']
            col_medians = bundle.get('col_medians')
            feat_indices = bundle.get('feature_indices')
            scaler = bundle.get('scaler')

            X_score = X.copy()

            # Impute NaN on full feature set first
            if col_medians is not None:
                for col in range(min(X_score.shape[1], len(col_medians))):
                    mask = np.isnan(X_score[:, col])
                    X_score[mask, col] = col_medians[col]

            # Scale on full feature set (KNN scales before pruning)
            if scaler is not None:
                X_score = scaler.transform(X_score)

            # Feature pruning AFTER scaling (matches training order)
            if feat_indices is not None:
                X_score = X_score[:, feat_indices]

            probs = model.predict_proba(X_score)[:, 1]
            for r, p in zip(records, probs):
                r[prob_key] = round(float(p), 4)
            print(f"  {name.upper()}: {len(records)} scored")

        except Exception as e:
            print(f"  {name.upper()}: failed ({e})")

    # ── 7. Sim Model ──
    try:
        from sim_model import enrich_with_sim
        sim_count = enrich_with_sim(records)
        print(f"  SIM: {sim_count} scored")
    except Exception as e:
        print(f"  SIM: failed ({e})")

    # ── 8. Regression Model ──
    try:
        from regression_model import score_regression
        reg_count = score_regression(records)
        print(f"  REG: {reg_count} scored")
    except Exception as e:
        print(f"  REG: failed ({e})")

    return records


def analyze_model_agreement(records):
    """Show how models agree/disagree and where each adds value."""
    model_keys = ['xgb_prob', 'mlp_prob', 'rf_prob', 'catboost_prob', 'knn_prob', 'logreg_prob']

    # Per-model accuracy when confident (prob >= 0.55 or <= 0.45)
    print("\n  ── Per-Model Accuracy (confident picks, prob >= 0.55) ──")
    for mk in model_keys:
        confident = [(r[mk], r['_hit_label']) for r in records if mk in r and r[mk] >= 0.55]
        if confident:
            correct = sum(1 for p, h in confident if h == True)
            print(f"    {mk:16s}: {correct}/{len(confident)} = {correct/len(confident):.1%}")

    # Agreement analysis
    print("\n  ── Model Agreement vs Accuracy ──")
    for r in records:
        probs = [r.get(mk) for mk in model_keys if r.get(mk) is not None]
        if len(probs) >= 4:
            above_50 = sum(1 for p in probs if p > 0.5)
            r['_agreement'] = above_50 / len(probs)  # 0-1 scale

    for threshold_name, lo, hi in [
        ('Strong agree (>80%)', 0.8, 1.01),
        ('Moderate (60-80%)', 0.6, 0.8),
        ('Split (40-60%)', 0.4, 0.6),
        ('Disagree (<40%)', 0.0, 0.4),
    ]:
        subset = [r for r in records if lo <= r.get('_agreement', -1) < hi]
        if subset:
            hits = sum(1 for r in subset if r['_hit_label'])
            print(f"    {threshold_name:25s}: {hits}/{len(subset)} = {hits/len(subset):.1%}")

    # Per-stat-type: which model is best
    print("\n  ── Best Model by Stat Type ──")
    from collections import defaultdict
    stat_model_acc = defaultdict(lambda: defaultdict(list))
    for r in records:
        stat = r.get('stat', '?')
        hit = r['_hit_label']
        for mk in model_keys:
            if r.get(mk) is not None:
                pred_hit = r[mk] >= 0.5
                stat_model_acc[stat][mk].append(1 if pred_hit == hit else 0)

    for stat in sorted(stat_model_acc.keys()):
        best_model = max(stat_model_acc[stat].items(), key=lambda x: np.mean(x[1]) if x[1] else 0)
        best_acc = np.mean(best_model[1]) if best_model[1] else 0
        n = len(best_model[1])
        print(f"    {stat:6s}: {best_model[0]:16s} {best_acc:.1%} ({n} props)")


def save_rescored(records, path=None):
    """Save rescored graded data for meta-learner training."""
    if path is None:
        path = os.path.join(PREDICTIONS_DIR, 'cache', 'rescored_graded.json')

    # Strip numpy types for JSON
    clean = []
    for r in records:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, np.int64)):
                cr[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                cr[k] = float(v)
            elif isinstance(v, np.bool_):
                cr[k] = bool(v)
            elif isinstance(v, np.ndarray):
                cr[k] = v.tolist()
            else:
                cr[k] = v
        clean.append(cr)

    with open(path, 'w') as f:
        json.dump(clean, f)

    print(f"\n  Saved {len(clean)} rescored records to {path}")
    return path


def main():
    score_only = '--score-only' in sys.argv

    print("=" * 60)
    print("  Retroactive Model Scoring — All Models x All Graded Data")
    print("=" * 60)

    # 1. Load graded data
    records = load_all_graded()
    print(f"\n  Loaded {len(records)} graded records")

    dates = sorted(set(r['_date'] for r in records))
    print(f"  Dates: {', '.join(dates)}")

    # 2. Score with all models
    print(f"\n  Scoring with all models...")
    records = score_with_all_models(records)

    # 3. Coverage report
    model_keys = ['xgb_prob', 'mlp_prob', 'rf_prob', 'catboost_prob', 'knn_prob', 'logreg_prob', 'sim_prob', 'reg_margin']
    print(f"\n  ── Coverage Report ──")
    for mk in model_keys:
        count = sum(1 for r in records if r.get(mk) is not None)
        print(f"    {mk:16s}: {count}/{len(records)} ({count/len(records):.0%})")

    # 4. Agreement analysis
    analyze_model_agreement(records)

    # 5. Save rescored data
    save_path = save_rescored(records)

    # 6. Retrain meta-learner (unless --score-only)
    if not score_only:
        print(f"\n{'=' * 60}")
        print(f"  Retraining Meta-Learner on Fully-Scored Data")
        print(f"{'=' * 60}")

        # The meta-learner's collect_graded_records looks at the graded files.
        # We need to temporarily replace with our rescored data.
        # Instead, let's call the meta-learner train with the rescored data injected.

        from meta_learner import (
            build_meta_features, calibrate_base_probs,
            BASE_MODEL_NAMES, META_FEATURE_NAMES, N_META_FEATURES,
            MODEL_PATH as META_MODEL_PATH,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.isotonic import IsotonicRegression

        # Phase 1: Calibrate base model probs
        print("\n  Phase 1: Calibrating base model probabilities...")
        calibrators = {}
        for model_name in BASE_MODEL_NAMES:
            y_true = []
            y_pred = []
            for r in records:
                p = r.get(model_name)
                if p is not None and r.get('_hit_label') is not None:
                    y_true.append(int(r['_hit_label']))
                    y_pred.append(float(p))

            if len(y_true) >= 100:
                try:
                    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
                    iso.fit(y_pred, y_true)
                    calibrators[model_name] = iso
                    # Apply calibration
                    cal_preds = iso.predict(y_pred)
                    orig_acc = sum(1 for yt, yp in zip(y_true, y_pred) if (yp >= 0.5) == yt) / len(y_true)
                    cal_acc = sum(1 for yt, cp in zip(y_true, cal_preds) if (cp >= 0.5) == yt) / len(y_true)
                    print(f"    {model_name:16s}: {len(y_true)} samples, "
                          f"raw acc {orig_acc:.1%} → calibrated {cal_acc:.1%}")
                except Exception as e:
                    print(f"    {model_name:16s}: calibration failed ({e})")

        # Phase 2: Build meta-features
        print(f"\n  Phase 2: Building {N_META_FEATURES}-dim meta-features...")
        X_list = []
        y_list = []
        dates_list = []

        for r in records:
            features = build_meta_features(r, calibrators=calibrators)
            if features is not None and len(features) == N_META_FEATURES:
                X_list.append(features)
                y_list.append(int(r['_hit_label']))
                dates_list.append(r.get('_date', ''))

        X_meta = np.array(X_list, dtype=np.float64)
        y_meta = np.array(y_list)
        print(f"    {len(y_meta)} samples with complete meta-features")

        # Phase 3: Walk-forward CV on graded dates
        print(f"\n  Phase 3: Walk-forward CV...")
        graded_dates = sorted(set(d for d in dates_list if d >= '2026-'))
        dates_arr = np.array(dates_list)

        folds = []
        for i in range(1, len(graded_dates)):
            train_dates = set(graded_dates[:i])
            test_date = graded_dates[i]

            train_mask = np.array([d in train_dates or d < '2026-' for d in dates_list])
            test_mask = dates_arr == test_date

            if train_mask.sum() < 30 or test_mask.sum() < 10:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_meta[train_mask])
            X_te = scaler.transform(X_meta[test_mask])
            y_tr = y_meta[train_mask]
            y_te = y_meta[test_mask]

            lr = LogisticRegression(C=0.5, penalty='l2', max_iter=1000, random_state=42)
            lr.fit(X_tr, y_tr)

            y_prob = lr.predict_proba(X_te)[:, 1]
            acc = ((y_prob >= 0.5).astype(int) == y_te).mean()
            # AUC
            from xgb_model import _compute_auc
            auc = _compute_auc(y_te, y_prob)

            top_m = y_prob >= np.percentile(y_prob, 90)
            bot_m = y_prob <= np.percentile(y_prob, 10)
            top_hr = y_te[top_m].mean() if top_m.sum() > 0 else np.nan
            bot_hr = y_te[bot_m].mean() if bot_m.sum() > 0 else np.nan

            folds.append({
                'test_date': test_date,
                'test_size': int(test_mask.sum()),
                'auc': float(auc),
                'accuracy': float(acc),
                'top_decile_hr': float(top_hr) if not np.isnan(top_hr) else None,
                'bot_decile_hr': float(bot_hr) if not np.isnan(bot_hr) else None,
            })
            print(f"    {test_date}: AUC={auc:.3f} Acc={acc:.3f} "
                  f"Top10%={top_hr:.1%} Bot10%={bot_hr:.1%} (N={test_mask.sum()})")

        if folds:
            avg_auc = np.mean([f['auc'] for f in folds])
            avg_acc = np.mean([f['accuracy'] for f in folds])
            print(f"\n    CV Summary: AUC={avg_auc:.3f}, Acc={avg_acc:.3f}")

        # Phase 4: Train final meta-learner on all data
        print(f"\n  Phase 4: Training final meta-learner on {len(y_meta)} samples...")
        final_scaler = StandardScaler()
        X_final = final_scaler.fit_transform(X_meta)

        final_lr = LogisticRegression(C=0.5, penalty='l2', max_iter=1000, random_state=42)
        final_lr.fit(X_final, y_meta)

        # Save
        calibrator_data = {}
        for name, iso in calibrators.items():
            calibrator_data[name] = {
                'X_thresholds': iso.X_thresholds_.tolist() if hasattr(iso, 'X_thresholds_') else [],
                'y_thresholds': iso.y_thresholds_.tolist() if hasattr(iso, 'y_thresholds_') else [],
            }

        bundle = {
            'model': final_lr,
            'scaler': final_scaler,
            'calibrators': calibrators,
            'n_features': N_META_FEATURES,
            'version': 2,
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(y_meta),
            'cv_folds': [{k: v for k, v in f.items() if k not in ('y_test', 'y_prob')} for f in folds],
        }

        meta_path = os.path.join(PREDICTIONS_DIR, 'cache', 'meta_learner.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(bundle, f)

        # Model weights analysis
        print(f"\n  ── Model Weights (meta-learner coefficients) ──")
        coefs = dict(zip(META_FEATURE_NAMES, final_lr.coef_[0]))
        for mk in BASE_MODEL_NAMES + ['sim_prob', 'reg_margin']:
            w = coefs.get(mk, 0)
            bar = '█' * int(abs(w) * 10)
            sign = '+' if w >= 0 else '-'
            print(f"    {mk:16s}: {sign}{abs(w):.3f} {bar}")

        # Agreement features
        for mk in ['xgb_mlp_agree', 'tree_agree', 'all_agree', 'model_std', 'models_above_50']:
            w = coefs.get(mk, 0)
            if abs(w) > 0.01:
                bar = '█' * int(abs(w) * 10)
                sign = '+' if w >= 0 else '-'
                print(f"    {mk:16s}: {sign}{abs(w):.3f} {bar}")

        avg_auc_final = np.mean([f['auc'] for f in folds]) if folds else 0
        print(f"\n  Meta-learner saved: {meta_path}")
        print(f"  Final CV AUC: {avg_auc_final:.4f}")
        print(f"  Samples: {len(y_meta)} (was 3,109 before, now fully scored)")

        # Save metadata
        meta_info = {
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(y_meta),
            'n_features': N_META_FEATURES,
            'cv_avg_auc': float(avg_auc_final),
            'cv_avg_accuracy': float(np.mean([f['accuracy'] for f in folds])) if folds else 0,
            'cv_folds': [{k: v for k, v in f.items()} for f in folds],
            'model_weights': {k: float(v) for k, v in coefs.items()},
            'calibrators': list(calibrators.keys()),
            'rescored': True,
        }
        with open(meta_path.replace('.pkl', '_meta.json'), 'w') as f:
            json.dump(meta_info, f, indent=2)


if __name__ == '__main__':
    main()

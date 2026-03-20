#!/usr/bin/env python3
"""
Feature Measurement Script — Evaluates whether v8/v9 features (92 -> 136) improve XGBoost.

Runs three models through walk-forward CV with identical data/folds/params:
  A) Baseline (92 features) — original v7 feature set
  B) Full (136 features) — v7 + v8 ref/coach/sim + v9 nn embeddings
  C) Pruned — features with importance > median only

Outputs:
  - Comparison table (AUC, accuracy, top-decile)
  - Full 136-feature importance ranking
  - Pruned feature list saved to cache/pruned_features.json
"""

import json
import os
import sys
import time
import numpy as np

# Add predictions dir to path
PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PREDICTIONS_DIR)

from xgb_model import (
    FEATURE_COLS,
    collect_all_training_data,
    engineer_features,
    _get_model_params,
    _compute_auc,
    _compute_logloss,
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------
# Feature set definitions
# ---------------------------------------------------------------
BASELINE_COUNT = 92   # v7 features (indices 0-91)
V8_START = 92         # ref/coach/sim features (indices 92-103)
V8_END = 104          # 12 features
V9_START = 104        # nn embeddings (indices 104-135)
V9_END = 136          # 32 features

BASELINE_COLS = FEATURE_COLS[:BASELINE_COUNT]
V8_COLS = FEATURE_COLS[V8_START:V8_END]
V9_COLS = FEATURE_COLS[V9_START:V9_END]
NEW_COLS = V8_COLS + V9_COLS  # 44 new features


def walk_forward_cv_with_features(X_full, y, dates, sample_weights, sources, feature_indices, label=""):
    """Run walk-forward CV on a subset of features. Returns metrics dict.

    Mirrors xgb_model.walk_forward_cv exactly:
    - Historical data always in training
    - Backfill included chronologically
    - Only graded records in test folds
    """
    dates_arr = np.array(dates)
    sources_arr = np.array(sources)

    # Subset features
    X = X_full[:, feature_indices]
    n_features = len(feature_indices)

    # Only graded records define test folds
    graded_mask = sources_arr == 'graded'
    graded_dates = sorted(set(d for d, s in zip(dates, sources_arr) if s == 'graded' and d >= '2026-'))
    historical_mask = np.array([d < '2026-' and s == 'historical' for d, s in zip(dates, sources_arr)])

    if len(graded_dates) < 2:
        print(f"  [{label}] WARNING: Need at least 2 graded days for CV")
        return None

    all_y_test = []
    all_y_prob = []
    fold_metrics = []

    for i in range(1, len(graded_dates)):
        train_graded_dates = set(graded_dates[:i])
        test_date = graded_dates[i]

        train_graded_m = np.array([d in train_graded_dates and s == 'graded' for d, s in zip(dates, sources_arr)])
        train_backfill_m = np.array([d < test_date and s in ('backfill', 'sgo_backfill') for d, s in zip(dates, sources_arr)])
        train_mask = historical_mask | train_graded_m | train_backfill_m
        test_mask = (dates_arr == test_date) & graded_mask

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        sw_train = sample_weights[train_mask] if sample_weights is not None else None

        # Build params — need to adjust monotonic constraints for feature subset
        params = _get_model_params(y_train)
        # Rebuild monotonic constraints for the feature subset
        full_mono = list(params['monotone_constraints'])
        subset_mono = tuple(full_mono[idx] for idx in feature_indices)
        params['monotone_constraints'] = subset_mono

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

        accuracy = (y_pred == y_test).mean()
        auc = _compute_auc(y_test, y_prob)

        fold_metrics.append({
            'test_date': test_date,
            'test_size': int(test_mask.sum()),
            'accuracy': float(accuracy),
            'auc': float(auc),
        })

        all_y_test.append(y_test)
        all_y_prob.append(y_prob)

    if not fold_metrics:
        return None

    # Pooled metrics
    all_y_test = np.concatenate(all_y_test)
    all_y_prob = np.concatenate(all_y_prob)
    pooled_auc = _compute_auc(all_y_test, all_y_prob)
    pooled_acc = (( all_y_prob >= 0.5).astype(int) == all_y_test).mean()
    pooled_logloss = _compute_logloss(all_y_test, all_y_prob)

    # Top/bottom decile
    p90 = np.percentile(all_y_prob, 90)
    p10 = np.percentile(all_y_prob, 10)
    top_mask = all_y_prob >= p90
    bot_mask = all_y_prob <= p10
    top_decile = float(all_y_test[top_mask].mean()) if top_mask.sum() > 0 else float('nan')
    bot_decile = float(all_y_test[bot_mask].mean()) if bot_mask.sum() > 0 else float('nan')

    avg_auc = np.mean([f['auc'] for f in fold_metrics])
    avg_acc = np.mean([f['accuracy'] for f in fold_metrics])

    return {
        'label': label,
        'n_features': n_features,
        'n_folds': len(fold_metrics),
        'n_test_total': len(all_y_test),
        'pooled_auc': pooled_auc,
        'pooled_accuracy': pooled_acc,
        'pooled_logloss': pooled_logloss,
        'avg_auc': avg_auc,
        'avg_accuracy': avg_acc,
        'top_decile_hr': top_decile,
        'bot_decile_hr': bot_decile,
        'folds': fold_metrics,
    }


def get_feature_importance(X_full, y, dates, sample_weights, sources, feature_indices, feature_names):
    """Train a full model and return feature importances sorted descending."""
    X = X_full[:, feature_indices]

    params = _get_model_params(y)
    full_mono = list(params['monotone_constraints'])
    subset_mono = tuple(full_mono[idx] for idx in feature_indices)
    params['monotone_constraints'] = subset_mono

    model = XGBClassifier(**params)
    model.fit(X, y, sample_weight=sample_weights, verbose=False)

    importances = dict(zip(feature_names, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return sorted_imp


def main():
    print("=" * 70)
    print("  FEATURE MEASUREMENT: 92 (baseline) vs 136 (full) vs pruned")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: Load all training data (same as xgb_model.py train)
    # ---------------------------------------------------------------
    print("\n[1/5] Loading training data...")
    t0 = time.time()
    records = collect_all_training_data(use_historical=True)

    if len(records) < 100:
        print(f"  ERROR: Only {len(records)} records -- need at least 100")
        sys.exit(1)

    X_full, y, dates = engineer_features(records)

    # Build sample weights (exact same logic as xgb_model.py)
    def _weight(r):
        src = r.get('_data_source', '')
        if src == 'graded': return 25.0
        if src == 'backfill': return 10.0
        if src == 'sgo_backfill': return 8.0
        return 1.0
    sample_weights = np.array([_weight(r) for r in records])
    sources = [r.get('_data_source', 'graded') for r in records]

    n_graded = sum(1 for s in sources if s == 'graded')
    n_backfill = sum(1 for s in sources if s == 'backfill')
    n_sgo = sum(1 for s in sources if s == 'sgo_backfill')
    n_hist = sum(1 for s in sources if s == 'historical')
    n_10yr = sum(1 for s in sources if s == 'historical_10yr')

    print(f"\n  Data loaded in {time.time()-t0:.1f}s")
    print(f"  Total: {len(records):,} records ({X_full.shape[1]} features)")
    print(f"    Graded:      {n_graded:>7,}")
    print(f"    Backfill:    {n_backfill:>7,}")
    print(f"    SGO backfill:{n_sgo:>7,}")
    print(f"    10yr:        {n_10yr:>7,}")
    print(f"    Legacy hist: {n_hist:>7,}")

    # Check NaN rates for new features to understand data coverage
    print(f"\n  NaN rates for new v8/v9 features:")
    for i in range(V8_START, V9_END):
        col = FEATURE_COLS[i]
        nan_rate = np.isnan(X_full[:, i]).mean()
        # Also check NaN rate for graded data only
        graded_idx = [j for j, s in enumerate(sources) if s == 'graded']
        if graded_idx:
            graded_nan = np.isnan(X_full[graded_idx, i]).mean()
        else:
            graded_nan = float('nan')
        tag = "v8" if i < V9_START else "v9"
        print(f"    [{tag}] {col:30s}  all={nan_rate:5.1%}  graded={graded_nan:5.1%}")

    # ---------------------------------------------------------------
    # Step 2: Run walk-forward CV for each feature set
    # ---------------------------------------------------------------
    baseline_idx = list(range(BASELINE_COUNT))
    full_idx = list(range(len(FEATURE_COLS)))

    # A) Baseline (92 features)
    print(f"\n{'='*70}")
    print(f"[2/5] Baseline model (92 features -- v7)")
    print(f"{'='*70}")
    t1 = time.time()
    baseline_results = walk_forward_cv_with_features(
        X_full, y, dates, sample_weights, sources,
        feature_indices=baseline_idx, label="Baseline (v7)"
    )
    print(f"  Completed in {time.time()-t1:.1f}s")
    if baseline_results:
        print(f"  Pooled AUC={baseline_results['pooled_auc']:.4f}, "
              f"Acc={baseline_results['pooled_accuracy']:.1%}, "
              f"Top10%={baseline_results['top_decile_hr']:.1%}")

    # B) Full (136 features)
    print(f"\n{'='*70}")
    print(f"[3/5] Full model (136 features -- v7 + v8 + v9)")
    print(f"{'='*70}")
    t2 = time.time()
    full_results = walk_forward_cv_with_features(
        X_full, y, dates, sample_weights, sources,
        feature_indices=full_idx, label="Full (v9)"
    )
    print(f"  Completed in {time.time()-t2:.1f}s")
    if full_results:
        print(f"  Pooled AUC={full_results['pooled_auc']:.4f}, "
              f"Acc={full_results['pooled_accuracy']:.1%}, "
              f"Top10%={full_results['top_decile_hr']:.1%}")

    # ---------------------------------------------------------------
    # Step 3: Feature importance on full model
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[4/5] Feature importance (full 136-feature model)")
    print(f"{'='*70}")

    sorted_importance = get_feature_importance(
        X_full, y, dates, sample_weights, sources,
        feature_indices=full_idx, feature_names=FEATURE_COLS
    )

    # Compute median importance
    imp_values = [imp for _, imp in sorted_importance]
    median_imp = float(np.median(imp_values))
    mean_imp = float(np.mean(imp_values))

    print(f"\n  Importance stats: median={median_imp:.5f}, mean={mean_imp:.5f}")
    print(f"\n  ALL 136 features ranked by importance:")
    print(f"  {'Rank':>4s}  {'Feature':35s}  {'Importance':>10s}  {'Version':>7s}  {'Status':>8s}")
    print(f"  {'-'*4}  {'-'*35}  {'-'*10}  {'-'*7}  {'-'*8}")

    new_in_top_50 = []
    features_above_median = []

    for rank, (name, imp) in enumerate(sorted_importance, 1):
        idx = FEATURE_COLS.index(name)
        if idx < BASELINE_COUNT:
            version = "v7"
        elif idx < V8_END:
            version = "v8"
        else:
            version = "v9"

        above = "KEEP" if imp > median_imp else "DROP"
        if imp > median_imp:
            features_above_median.append(name)

        is_new = version in ("v8", "v9")
        marker = " ***" if is_new and rank <= 50 else ""
        if is_new and rank <= 50:
            new_in_top_50.append((rank, name, imp, version))

        print(f"  {rank:>4d}  {name:35s}  {imp:10.5f}  {version:>7s}  {above:>8s}{marker}")

    # Summarize new features performance
    print(f"\n  --- New features (v8+v9) in top 50: {len(new_in_top_50)}/44 ---")
    if new_in_top_50:
        for rank, name, imp, ver in new_in_top_50:
            print(f"    #{rank:2d}  {name:35s}  {imp:.5f}  ({ver})")
    else:
        print(f"    NONE -- no v8/v9 feature ranked in top 50")

    # Count new features above median
    new_above = [(name, imp) for name, imp in sorted_importance
                 if FEATURE_COLS.index(name) >= BASELINE_COUNT and imp > median_imp]
    new_below = [(name, imp) for name, imp in sorted_importance
                 if FEATURE_COLS.index(name) >= BASELINE_COUNT and imp <= median_imp]

    print(f"\n  New features above median importance: {len(new_above)}/44")
    print(f"  New features below median importance: {len(new_below)}/44")

    if new_above:
        print(f"  Worth keeping:")
        for name, imp in new_above:
            print(f"    {name:35s}  {imp:.5f}")
    if new_below:
        print(f"  Candidates for removal:")
        for name, imp in new_below:
            print(f"    {name:35s}  {imp:.5f}")

    # ---------------------------------------------------------------
    # Step 4: Pruned model (features above median importance)
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[5/5] Pruned model ({len(features_above_median)} features -- importance > median)")
    print(f"{'='*70}")

    pruned_idx = [FEATURE_COLS.index(f) for f in features_above_median]
    pruned_idx.sort()  # maintain original order
    pruned_names = [FEATURE_COLS[i] for i in pruned_idx]

    print(f"  Pruned feature set: {len(pruned_idx)} features")
    n_baseline_kept = sum(1 for i in pruned_idx if i < BASELINE_COUNT)
    n_v8_kept = sum(1 for i in pruned_idx if V8_START <= i < V8_END)
    n_v9_kept = sum(1 for i in pruned_idx if V9_START <= i < V9_END)
    print(f"    From v7 baseline: {n_baseline_kept}/92")
    print(f"    From v8 (ref/coach/sim): {n_v8_kept}/12")
    print(f"    From v9 (nn embeddings): {n_v9_kept}/32")

    t3 = time.time()
    pruned_results = walk_forward_cv_with_features(
        X_full, y, dates, sample_weights, sources,
        feature_indices=pruned_idx, label="Pruned"
    )
    print(f"  Completed in {time.time()-t3:.1f}s")
    if pruned_results:
        print(f"  Pooled AUC={pruned_results['pooled_auc']:.4f}, "
              f"Acc={pruned_results['pooled_accuracy']:.1%}, "
              f"Top10%={pruned_results['top_decile_hr']:.1%}")

    # ---------------------------------------------------------------
    # BONUS: Also test baseline + v8-only (no v9 nn embeddings)
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"[BONUS] Baseline + v8 only (104 features -- no nn embeddings)")
    print(f"{'='*70}")

    v8_only_idx = list(range(V8_END))  # 0-103
    t4 = time.time()
    v8_only_results = walk_forward_cv_with_features(
        X_full, y, dates, sample_weights, sources,
        feature_indices=v8_only_idx, label="Baseline + v8"
    )
    print(f"  Completed in {time.time()-t4:.1f}s")
    if v8_only_results:
        print(f"  Pooled AUC={v8_only_results['pooled_auc']:.4f}, "
              f"Acc={v8_only_results['pooled_accuracy']:.1%}, "
              f"Top10%={v8_only_results['top_decile_hr']:.1%}")

    # ---------------------------------------------------------------
    # Final comparison table
    # ---------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")

    models = []
    if baseline_results:
        models.append(baseline_results)
    if full_results:
        models.append(full_results)
    if v8_only_results:
        models.append(v8_only_results)
    if pruned_results:
        models.append(pruned_results)

    header = f"  {'Model':<22s} | {'Features':>8s} | {'Pooled AUC':>10s} | {'Accuracy':>8s} | {'Top-Decile':>10s} | {'Bot-Decile':>10s} | {'LogLoss':>8s}"
    print(header)
    print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    best_auc = max(m['pooled_auc'] for m in models)

    for m in models:
        is_best = " *" if m['pooled_auc'] == best_auc else "  "
        print(f"  {m['label']:<22s} | {m['n_features']:>8d} | "
              f"{m['pooled_auc']:>10.4f} | {m['pooled_accuracy']:>7.1%} | "
              f"{m['top_decile_hr']:>9.1%} | {m['bot_decile_hr']:>9.1%} | "
              f"{m['pooled_logloss']:>8.4f}{is_best}")

    print(f"\n  * = best pooled AUC")

    # Delta analysis
    if baseline_results and full_results:
        auc_delta = full_results['pooled_auc'] - baseline_results['pooled_auc']
        acc_delta = full_results['pooled_accuracy'] - baseline_results['pooled_accuracy']
        top_delta = full_results['top_decile_hr'] - baseline_results['top_decile_hr']

        direction = "IMPROVEMENT" if auc_delta > 0 else "REGRESSION" if auc_delta < 0 else "NO CHANGE"
        print(f"\n  Full vs Baseline delta:")
        print(f"    AUC:       {auc_delta:+.4f}  ({direction})")
        print(f"    Accuracy:  {acc_delta:+.1%}")
        print(f"    Top-10%:   {top_delta:+.1%}")

        if auc_delta < -0.005:
            print(f"\n  VERDICT: v8/v9 features HURT performance. Recommend reverting to 92 baseline.")
        elif auc_delta < 0.005:
            print(f"\n  VERDICT: v8/v9 features make NO meaningful difference. Extra complexity not justified.")
        else:
            print(f"\n  VERDICT: v8/v9 features HELP. Keep the full 136-feature model.")

    if baseline_results and pruned_results:
        prune_delta = pruned_results['pooled_auc'] - baseline_results['pooled_auc']
        print(f"\n  Pruned vs Baseline delta:")
        print(f"    AUC: {prune_delta:+.4f}")
        if pruned_results['pooled_auc'] >= best_auc - 0.002:
            print(f"    Pruned model is competitive -- recommend using {len(features_above_median)} features")

    # Per-fold breakdown
    print(f"\n\n{'='*70}")
    print(f"  PER-FOLD BREAKDOWN")
    print(f"{'='*70}")

    print(f"\n  {'Date':<12s} | ", end="")
    for m in models:
        print(f"  {m['label'][:15]:>15s}", end="")
    print()
    print(f"  {'-'*12}-+-", end="")
    for _ in models:
        print(f"-{'-'*15}", end="")
    print()

    # Collect all test dates
    all_test_dates = set()
    for m in models:
        for f in m['folds']:
            all_test_dates.add(f['test_date'])

    for date in sorted(all_test_dates):
        print(f"  {date:<12s} | ", end="")
        for m in models:
            fold = next((f for f in m['folds'] if f['test_date'] == date), None)
            if fold:
                print(f"  AUC={fold['auc']:.3f} N={fold['test_size']:>3d}", end="")
            else:
                print(f"  {'---':>15s}", end="")
        print()

    # ---------------------------------------------------------------
    # Save pruned feature list
    # ---------------------------------------------------------------
    cache_dir = os.path.join(PREDICTIONS_DIR, 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    output = {
        'pruned_features': pruned_names,
        'n_features': len(pruned_names),
        'n_baseline_kept': n_baseline_kept,
        'n_v8_kept': n_v8_kept,
        'n_v9_kept': n_v9_kept,
        'baseline_auc': baseline_results['pooled_auc'] if baseline_results else None,
        'full_auc': full_results['pooled_auc'] if full_results else None,
        'pruned_auc': pruned_results['pooled_auc'] if pruned_results else None,
        'v8_only_auc': v8_only_results['pooled_auc'] if v8_only_results else None,
        'recommendation': '',
        'feature_importance': {name: float(imp) for name, imp in sorted_importance},
        'dropped_features': [FEATURE_COLS[i] for i in range(len(FEATURE_COLS)) if i not in set(pruned_idx)],
    }

    # Set recommendation
    results_map = {}
    if baseline_results: results_map['baseline'] = baseline_results['pooled_auc']
    if full_results: results_map['full'] = full_results['pooled_auc']
    if pruned_results: results_map['pruned'] = pruned_results['pooled_auc']
    if v8_only_results: results_map['v8_only'] = v8_only_results['pooled_auc']

    best_name = max(results_map, key=results_map.get)
    output['recommendation'] = f"Use {best_name} model (AUC={results_map[best_name]:.4f})"

    output_path = os.path.join(cache_dir, 'pruned_features.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Pruned feature list saved to: {output_path}")
    print(f"  Recommendation: {output['recommendation']}")

    print(f"\n  Total runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()

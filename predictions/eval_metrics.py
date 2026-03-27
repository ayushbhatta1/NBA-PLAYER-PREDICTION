#!/usr/bin/env python3
"""
Evaluation Metrics Module — Comprehensive statistical evaluation for NBA prop predictions.

Computes classification, probabilistic, calibration, and profitability metrics
from graded daily predictions. Designed for sports betting where calibration
and profit matter more than raw accuracy.

Usage:
    python3 eval_metrics.py 2026-03-26              # Single day
    python3 eval_metrics.py 2026-03-26 2026-03-20   # Date range
    python3 eval_metrics.py --all                    # All graded days
    python3 eval_metrics.py --summary                # Cross-day summary from tracker

Outputs:
    - Terminal report with all metrics
    - predictions/YYYY-MM-DD/eval_metrics.json (per-day)
    - predictions/logs/eval_tracker.json (cumulative)
"""

import sys
import os
import json
import math
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRED_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_PATH = os.path.join(PRED_DIR, "logs", "eval_tracker.json")

# ─────────────────────────────────────────────────────────────────────
# 1. CLASSIFICATION METRICS (accuracy, precision, recall, F1)
# ─────────────────────────────────────────────────────────────────────

def compute_classification_metrics(results):
    """
    Compute precision, recall, F1, accuracy from graded predictions.

    In our context:
      - Positive class = HIT (prediction was correct)
      - Negative class = MISS (prediction was wrong)

    For per-direction breakdown (OVER/UNDER), we compute metrics separately
    since the pipeline is UNDER-dominant and class balance matters.

    Returns dict with overall + per-direction + per-stat + per-tier metrics.
    """
    graded = [r for r in results if r.get("result") in ("HIT", "MISS")]
    if not graded:
        return {"error": "no graded results"}

    # Overall
    overall = _classification_from_list(graded)

    # Per direction
    by_direction = {}
    for direction in ("OVER", "UNDER"):
        subset = [r for r in graded if r.get("direction") == direction]
        if subset:
            by_direction[direction] = _classification_from_list(subset)

    # Per stat type
    by_stat = {}
    for stat in sorted(set(r.get("stat", "?") for r in graded)):
        subset = [r for r in graded if r.get("stat") == stat]
        if len(subset) >= 5:  # minimum sample for meaningful metrics
            by_stat[stat] = _classification_from_list(subset)

    # Per tier
    by_tier = {}
    for tier in ["S", "A", "B", "C", "D", "F"]:
        subset = [r for r in graded if r.get("tier") == tier]
        if len(subset) >= 5:
            by_tier[tier] = _classification_from_list(subset)

    return {
        "overall": overall,
        "by_direction": by_direction,
        "by_stat": by_stat,
        "by_tier": by_tier,
    }


def _classification_from_list(results):
    """Compute TP/FP/FN/TN, precision, recall, F1, accuracy from a list of graded results."""
    tp = sum(1 for r in results if r["result"] == "HIT")
    fn = sum(1 for r in results if r["result"] == "MISS")
    total = tp + fn

    # In binary HIT/MISS:
    #   precision = TP / (TP + FP). Since every prediction is a "positive" claim
    #   (we predict HIT), FP = MISS, so precision = accuracy.
    #   recall = TP / (TP + FN) = same thing.
    #
    # More useful: treat OVER and UNDER as separate classes for macro F1.
    # But for single-direction analysis, we use a different framing:
    #   For UNDER predictions: TP = actual < line, FP = actual >= line
    #   For OVER predictions: TP = actual > line, FP = actual <= line

    accuracy = tp / total if total > 0 else 0

    # For single-class (all predictions are positive claims), F1 = accuracy
    # More meaningful: compute F1 treating HIT as positive class
    precision = tp / total if total > 0 else 0  # all predictions are "positive"
    recall = 1.0  # we predict on every line, so recall = 1 trivially

    # Binary F1 with actual class balance
    # Reframe: for each prediction, was the predicted direction correct?
    # This is just accuracy. F1 becomes interesting in macro-average across slices.
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "n": total,
        "hits": tp,
        "misses": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def compute_macro_f1(results):
    """
    Macro F1 across stat types — the useful F1 for this pipeline.

    Averages F1 per stat type equally, regardless of how many predictions
    each stat has. This penalizes poor performance on rare stats (BLK, STL)
    equally to common stats (PTS, REB).
    """
    graded = [r for r in results if r.get("result") in ("HIT", "MISS")]
    stat_groups = defaultdict(list)
    for r in graded:
        stat_groups[r.get("stat", "?")].append(r)

    f1_scores = []
    details = {}
    for stat, group in sorted(stat_groups.items()):
        if len(group) < 3:
            continue
        hits = sum(1 for r in group if r["result"] == "HIT")
        total = len(group)
        acc = hits / total
        # F1 in binary: 2*p*r/(p+r), with p=acc, r=1 → F1 = 2*acc/(acc+1)
        f1 = 2 * acc / (acc + 1) if acc > 0 else 0
        f1_scores.append(f1)
        details[stat] = {"n": total, "accuracy": round(acc, 4), "f1": round(f1, 4)}

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return {"macro_f1": round(macro_f1, 4), "per_stat": details}


# ─────────────────────────────────────────────────────────────────────
# 2. PROBABILISTIC METRICS (Brier, Log Loss, ROC-AUC)
# ─────────────────────────────────────────────────────────────────────

def compute_probabilistic_metrics(results):
    """
    Compute Brier score, log loss, and ROC-AUC from ensemble probabilities.

    These require model-assigned probabilities (ensemble_prob, xgb_prob, etc.)
    and are the gold standard for evaluating probabilistic predictions in betting.

    Brier score: MSE of probabilities. Range [0, 1], lower is better.
        - 0.25 = random baseline (predicting 0.5 always)
        - < 0.25 = model has skill
    Log loss: Cross-entropy. Heavily penalizes confident wrong predictions.
        - 0.693 = random baseline
        - < 0.693 = model has skill
    ROC-AUC: Discrimination ability. 0.5 = random, 1.0 = perfect.
    """
    # Collect (predicted_prob, actual_binary) pairs
    pairs = []
    xgb_pairs = []
    mlp_pairs = []
    ensemble_pairs = []
    sim_pairs = []
    reg_pairs = []

    for r in results:
        if r.get("result") not in ("HIT", "MISS"):
            continue

        actual = 1 if r["result"] == "HIT" else 0

        # Ensemble probability (primary)
        ens_prob = r.get("ensemble_prob")
        if ens_prob is not None and not math.isnan(ens_prob):
            ensemble_pairs.append((ens_prob, actual))
            pairs.append((ens_prob, actual))

        # XGBoost
        xgb_prob = r.get("xgb_prob")
        if xgb_prob is not None and not math.isnan(xgb_prob):
            xgb_pairs.append((xgb_prob, actual))
            if not pairs or pairs[-1][1] != actual:  # avoid double-counting
                pass  # already added via ensemble

        # MLP
        mlp_prob = r.get("mlp_prob")
        if mlp_prob is not None and not math.isnan(mlp_prob):
            mlp_pairs.append((mlp_prob, actual))

        # Simulation
        sim_prob = r.get("sim_prob")
        if sim_prob is not None and not math.isnan(sim_prob):
            sim_pairs.append((sim_prob, actual))

        # Regression-derived
        reg_prob = r.get("reg_over_prob")
        if reg_prob is not None and not math.isnan(reg_prob):
            # reg_over_prob is P(OVER). If direction=UNDER, our "hit prob" = 1 - reg_over_prob
            direction = r.get("direction", "UNDER")
            if direction == "UNDER":
                reg_pairs.append((1 - reg_prob, actual))
            else:
                reg_pairs.append((reg_prob, actual))

    output = {}

    # Compute for each model source
    for name, model_pairs in [
        ("ensemble", ensemble_pairs),
        ("xgb", xgb_pairs),
        ("mlp", mlp_pairs),
        ("sim", sim_pairs),
        ("regression", reg_pairs),
    ]:
        if len(model_pairs) < 10:
            continue
        probs = [p[0] for p in model_pairs]
        actuals = [p[1] for p in model_pairs]

        brier = _brier_score(probs, actuals)
        logloss = _log_loss(probs, actuals)
        auc = _roc_auc(probs, actuals)

        output[name] = {
            "n": len(model_pairs),
            "brier_score": round(brier, 4),
            "log_loss": round(logloss, 4),
            "roc_auc": round(auc, 4),
            "brier_skill": round(1 - brier / 0.25, 4),  # skill score vs random
            "logloss_skill": round(1 - logloss / 0.6931, 4),  # skill vs random
        }

    return output


def _brier_score(probs, actuals):
    """Brier score = mean((prob - actual)^2). Lower is better."""
    return sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(probs)


def _log_loss(probs, actuals, eps=1e-15):
    """Binary cross-entropy log loss. Lower is better."""
    total = 0
    for p, a in zip(probs, actuals):
        p = max(eps, min(1 - eps, p))  # clip to avoid log(0)
        total += a * math.log(p) + (1 - a) * math.log(1 - p)
    return -total / len(probs)


def _roc_auc(probs, actuals):
    """
    ROC-AUC via Mann-Whitney U statistic. No sklearn dependency.
    Equivalent to: P(score(positive) > score(negative)).
    """
    pos_scores = [p for p, a in zip(probs, actuals) if a == 1]
    neg_scores = [p for p, a in zip(probs, actuals) if a == 0]

    if not pos_scores or not neg_scores:
        return 0.5  # undefined, return random baseline

    # Count concordant pairs
    concordant = 0
    tied = 0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                concordant += 1
            elif ps == ns:
                tied += 0.5

    total_pairs = len(pos_scores) * len(neg_scores)
    return (concordant + tied) / total_pairs if total_pairs > 0 else 0.5


def compute_roc_curve(results, model_key="ensemble_prob"):
    """
    Compute ROC curve points for terminal ASCII rendering.
    Returns list of (fpr, tpr, threshold) sorted by threshold descending.
    """
    pairs = []
    for r in results:
        if r.get("result") not in ("HIT", "MISS"):
            continue
        prob = r.get(model_key)
        if prob is None or (isinstance(prob, float) and math.isnan(prob)):
            continue
        actual = 1 if r["result"] == "HIT" else 0
        pairs.append((prob, actual))

    if len(pairs) < 20:
        return []

    # Sort by probability descending
    pairs.sort(key=lambda x: x[0], reverse=True)
    total_pos = sum(a for _, a in pairs)
    total_neg = len(pairs) - total_pos

    if total_pos == 0 or total_neg == 0:
        return []

    curve = []
    tp = 0
    fp = 0

    # Walk through thresholds
    for i, (prob, actual) in enumerate(pairs):
        if actual == 1:
            tp += 1
        else:
            fp += 1

        # Record point at each unique threshold
        if i == len(pairs) - 1 or pairs[i + 1][0] != prob:
            tpr = tp / total_pos
            fpr = fp / total_neg
            curve.append({"fpr": round(fpr, 4), "tpr": round(tpr, 4), "threshold": round(prob, 3)})

    return curve


# ─────────────────────────────────────────────────────────────────────
# 3. CALIBRATION METRICS (ECE, reliability diagram data)
# ─────────────────────────────────────────────────────────────────────

def compute_calibration_metrics(results, n_bins=10, model_key="ensemble_prob"):
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.

    ECE = weighted average of |predicted_prob - actual_hit_rate| per bin.
    This is THE metric for sports betting: if you say 70%, it should hit 70%.

    Also computes Maximum Calibration Error (MCE) — worst single bin.
    """
    pairs = []
    for r in results:
        if r.get("result") not in ("HIT", "MISS"):
            continue
        prob = r.get(model_key)
        if prob is None or (isinstance(prob, float) and math.isnan(prob)):
            continue
        actual = 1 if r["result"] == "HIT" else 0
        pairs.append((prob, actual))

    if len(pairs) < 20:
        return {"error": "insufficient data", "n": len(pairs)}

    # Build bins
    bins = []
    bin_edges = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]

    ece = 0
    mce = 0
    total_n = len(pairs)

    for low, high in bin_edges:
        bin_pairs = [(p, a) for p, a in pairs if low <= p < high]
        if not bin_pairs:
            bins.append({
                "range": f"{low:.1f}-{high:.1f}",
                "n": 0,
                "avg_predicted": 0,
                "avg_actual": 0,
                "gap": 0,
            })
            continue

        avg_pred = sum(p for p, _ in bin_pairs) / len(bin_pairs)
        avg_actual = sum(a for _, a in bin_pairs) / len(bin_pairs)
        gap = abs(avg_pred - avg_actual)

        ece += gap * len(bin_pairs) / total_n
        mce = max(mce, gap)

        bins.append({
            "range": f"{low:.1f}-{high:.1f}",
            "n": len(bin_pairs),
            "avg_predicted": round(avg_pred, 4),
            "avg_actual": round(avg_actual, 4),
            "gap": round(gap, 4),
        })

    return {
        "n": total_n,
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "bins": bins,
        "model": model_key,
    }


# ─────────────────────────────────────────────────────────────────────
# 4. PROFIT / EV METRICS (ROI, profit curve, Precision@K)
# ─────────────────────────────────────────────────────────────────────

def compute_profit_metrics(results, odds=1.91):
    """
    Compute profitability metrics at various confidence thresholds.

    Uses standard -110 odds (1.91 decimal) unless overridden.
    Flat $1 unit bets. Calculates:
      - ROI at each confidence bin
      - Cumulative profit curve (sorted by confidence descending)
      - Optimal threshold (max ROI)
      - Precision@K (top-K most confident picks)
    """
    graded = []
    for r in results:
        if r.get("result") not in ("HIT", "MISS"):
            continue
        prob = r.get("ensemble_prob", r.get("xgb_prob"))
        if prob is None or (isinstance(prob, float) and math.isnan(prob)):
            continue
        graded.append({
            "prob": prob,
            "hit": r["result"] == "HIT",
            "ev": r.get("ev_per_unit", 0),
            "kelly": r.get("kelly_stake", 0),
            "tier": r.get("tier", "?"),
            "stat": r.get("stat", "?"),
        })

    if not graded:
        return {"error": "no graded results with probabilities"}

    # Sort by confidence descending
    graded.sort(key=lambda x: x["prob"], reverse=True)

    # ── Precision@K ──
    precision_at_k = {}
    for k in [5, 10, 20, 50, 100]:
        if k > len(graded):
            continue
        top_k = graded[:k]
        hits = sum(1 for g in top_k if g["hit"])
        precision_at_k[f"p@{k}"] = round(hits / k, 4)

    # ── Profit curve (cumulative) ──
    profit_curve = []
    cumulative_profit = 0
    cumulative_bets = 0
    for g in graded:
        cumulative_bets += 1
        if g["hit"]:
            cumulative_profit += odds - 1  # win payout
        else:
            cumulative_profit -= 1  # lose stake

        if cumulative_bets in (5, 10, 20, 50, 100, 200, len(graded)) or cumulative_bets % 50 == 0:
            profit_curve.append({
                "bets": cumulative_bets,
                "profit": round(cumulative_profit, 2),
                "roi": round(cumulative_profit / cumulative_bets * 100, 2),
                "min_prob": round(g["prob"], 3),
            })

    # ── ROI by confidence bins ──
    roi_by_bin = []
    bin_edges = [(0.3, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55),
                 (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 1.0)]
    for low, high in bin_edges:
        bin_picks = [g for g in graded if low <= g["prob"] < high]
        if not bin_picks:
            continue
        wins = sum(1 for g in bin_picks if g["hit"])
        pnl = wins * (odds - 1) - (len(bin_picks) - wins)
        roi_by_bin.append({
            "range": f"{low:.2f}-{high:.2f}",
            "n": len(bin_picks),
            "wins": wins,
            "accuracy": round(wins / len(bin_picks), 4),
            "pnl": round(pnl, 2),
            "roi": round(pnl / len(bin_picks) * 100, 2),
        })

    # ── Optimal threshold ──
    best_roi = -999
    best_threshold = 0.5
    for threshold in [i / 100 for i in range(40, 85, 1)]:
        above = [g for g in graded if g["prob"] >= threshold]
        if len(above) < 5:
            continue
        wins = sum(1 for g in above if g["hit"])
        pnl = wins * (odds - 1) - (len(above) - wins)
        roi = pnl / len(above) * 100
        if roi > best_roi:
            best_roi = roi
            best_threshold = threshold

    # ── Overall P&L ──
    total_wins = sum(1 for g in graded if g["hit"])
    total_pnl = total_wins * (odds - 1) - (len(graded) - total_wins)
    total_roi = total_pnl / len(graded) * 100 if graded else 0

    return {
        "total_bets": len(graded),
        "total_wins": total_wins,
        "total_pnl": round(total_pnl, 2),
        "total_roi": round(total_roi, 2),
        "precision_at_k": precision_at_k,
        "profit_curve": profit_curve,
        "roi_by_confidence": roi_by_bin,
        "optimal_threshold": round(best_threshold, 2),
        "optimal_roi": round(best_roi, 2),
    }


# ─────────────────────────────────────────────────────────────────────
# 5. BETTING-SPECIFIC METRICS (CLV proxy, edge retention)
# ─────────────────────────────────────────────────────────────────────

def compute_betting_metrics(results):
    """
    Betting-specific analytics:
      - Edge distribution: how many picks have positive expected value
      - Confidence vs outcome correlation
      - Model agreement analysis (when multiple models agree, does accuracy improve?)
      - Tier efficiency: ROI per tier to identify profitable segments
    """
    graded = [r for r in results if r.get("result") in ("HIT", "MISS")]
    if not graded:
        return {}

    # ── Positive EV rate ──
    ev_picks = [r for r in graded if r.get("ev_per_unit") is not None]
    pos_ev = sum(1 for r in ev_picks if r.get("ev_per_unit", 0) > 0)
    pos_ev_hits = sum(1 for r in ev_picks if r.get("ev_per_unit", 0) > 0 and r["result"] == "HIT")
    neg_ev_hits = sum(1 for r in ev_picks if r.get("ev_per_unit", 0) <= 0 and r["result"] == "HIT")

    ev_analysis = {
        "total_with_ev": len(ev_picks),
        "positive_ev_count": pos_ev,
        "positive_ev_accuracy": round(pos_ev_hits / pos_ev, 4) if pos_ev > 0 else 0,
        "negative_ev_accuracy": round(neg_ev_hits / (len(ev_picks) - pos_ev), 4) if (len(ev_picks) - pos_ev) > 0 else 0,
    }

    # ── Model agreement analysis ──
    agreement_buckets = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in graded:
        agree_pct = r.get("model_agree_pct")
        if agree_pct is None:
            continue
        bucket = f"{int(agree_pct // 20) * 20}-{int(agree_pct // 20) * 20 + 20}%"
        agreement_buckets[bucket]["total"] += 1
        if r["result"] == "HIT":
            agreement_buckets[bucket]["hits"] += 1

    model_agreement = {}
    for bucket, stats in sorted(agreement_buckets.items()):
        if stats["total"] >= 3:
            model_agreement[bucket] = {
                "n": stats["total"],
                "accuracy": round(stats["hits"] / stats["total"], 4),
            }

    # ── Confidence-outcome correlation ──
    probs = []
    outcomes = []
    for r in graded:
        p = r.get("ensemble_prob", r.get("xgb_prob"))
        if p is not None and not (isinstance(p, float) and math.isnan(p)):
            probs.append(p)
            outcomes.append(1 if r["result"] == "HIT" else 0)

    corr = _pearson_correlation(probs, outcomes) if len(probs) >= 10 else None

    return {
        "ev_analysis": ev_analysis,
        "model_agreement": model_agreement,
        "confidence_outcome_correlation": round(corr, 4) if corr is not None else None,
        "n_graded": len(graded),
    }


def _pearson_correlation(x, y):
    """Pearson correlation coefficient. No numpy/scipy dependency."""
    n = len(x)
    if n < 2:
        return 0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5
    if std_x == 0 or std_y == 0:
        return 0
    return cov / (std_x * std_y)


# ─────────────────────────────────────────────────────────────────────
# 6. ASCII RENDERING (terminal-friendly output)
# ─────────────────────────────────────────────────────────────────────

def render_ascii_roc(curve_points, width=50, height=20):
    """Render an ASCII ROC curve for terminal display."""
    if not curve_points:
        return "  (insufficient data for ROC curve)"

    # Build canvas
    canvas = [[" "] * (width + 2) for _ in range(height + 2)]

    # Draw axes
    for y in range(height + 1):
        canvas[y][0] = "|"
    for x in range(width + 1):
        canvas[height][x + 1] = "-"
    canvas[height][0] = "+"

    # Draw diagonal (random baseline)
    for i in range(min(width, height) + 1):
        x = int(i * width / max(width, height))
        y = height - int(i * height / max(width, height))
        if 0 <= y < height and 0 < x <= width:
            canvas[y][x + 1] = "."

    # Plot ROC curve
    for point in curve_points:
        x = int(point["fpr"] * width)
        y = height - int(point["tpr"] * height)
        if 0 <= y < height and 0 < x <= width:
            canvas[y][x + 1] = "*"

    # Render
    lines = []
    lines.append("  TPR")
    lines.append("  1.0" + " " * (width - 3) + "|")
    for y in range(height + 1):
        row = "".join(canvas[y])
        if y == 0:
            lines.append(f"   {row}")
        elif y == height:
            lines.append(f"  0{row}")
        elif y == height // 2:
            lines.append(f" .5{row}")
        else:
            lines.append(f"   {row}")
    lines.append("    0" + " " * (width // 2 - 1) + "0.5" + " " * (width // 2 - 2) + "1.0  FPR")
    lines.append("    (. = random baseline, * = model)")
    return "\n".join(lines)


def render_calibration_plot(bins, width=50):
    """Render ASCII calibration reliability diagram."""
    if not bins:
        return "  (no calibration data)"

    lines = []
    lines.append("  Predicted  Actual   N     |  Calibration Plot")
    lines.append("  " + "-" * 60)

    for b in bins:
        if b["n"] == 0:
            continue
        pred = b["avg_predicted"]
        actual = b["avg_actual"]
        n = b["n"]
        gap = b["gap"]

        # Bar chart
        bar_pred = int(pred * width)
        bar_actual = int(actual * width)

        pred_bar = "P" * bar_pred
        actual_bar = "A" * bar_actual

        # Color indicator
        quality = "=" if gap < 0.05 else ("~" if gap < 0.10 else "!")

        lines.append(f"  {pred:6.3f}   {actual:6.3f}  {n:4d}  {quality}  P|{'#' * bar_pred}")
        lines.append(f"  {'':6s}   {'':6s}  {'':4s}  {quality}  A|{'=' * bar_actual}")

    lines.append("")
    lines.append("  = well-calibrated (<5%)  ~ moderate (5-10%)  ! poor (>10%)")
    return "\n".join(lines)


def render_profit_curve(profit_data, width=50):
    """Render ASCII cumulative profit curve."""
    curve = profit_data.get("profit_curve", [])
    if not curve:
        return "  (no profit data)"

    max_profit = max(abs(p["profit"]) for p in curve) or 1
    max_bets = curve[-1]["bets"]

    lines = []
    lines.append(f"  Cumulative P&L (flat $1 units, {profit_data['total_bets']} bets)")
    lines.append("  " + "-" * 60)

    for p in curve:
        x_pos = int(p["bets"] / max_bets * width) if max_bets > 0 else 0
        bar_len = int(abs(p["profit"]) / max_profit * 20) if max_profit > 0 else 0
        direction = "+" if p["profit"] >= 0 else "-"
        bar = direction * bar_len

        lines.append(
            f"  Bets {p['bets']:>4d} (p>={p['min_prob']:.2f}): "
            f"${p['profit']:>7.2f} ({p['roi']:>+6.1f}% ROI) {bar}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# 7. MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────

def evaluate_date(date_str, verbose=True):
    """
    Run full evaluation on a single date's graded predictions.
    Returns complete metrics dict. Optionally prints report.
    """
    graded_file = _find_graded_file(date_str)
    if not graded_file:
        print(f"[ERROR] No graded file for {date_str}")
        return None

    with open(graded_file) as f:
        data = json.load(f)

    if isinstance(data, dict):
        results = data.get("results", data.get("line_predictions", []))
    elif isinstance(data, list):
        results = data
    else:
        results = []

    # Only use graded results (skip non-dict entries from malformed files)
    graded = [r for r in results if isinstance(r, dict) and r.get("result") in ("HIT", "MISS")]
    if not graded:
        print(f"[WARN] No graded results for {date_str}")
        return None

    # Compute all metrics
    classification = compute_classification_metrics(graded)
    macro_f1 = compute_macro_f1(graded)
    probabilistic = compute_probabilistic_metrics(graded)
    calibration = compute_calibration_metrics(graded)
    roc_curve = compute_roc_curve(graded)
    profit = compute_profit_metrics(graded)
    betting = compute_betting_metrics(graded)

    metrics = {
        "date": date_str,
        "evaluated_at": datetime.now().isoformat(),
        "n_graded": len(graded),
        "classification": classification,
        "macro_f1": macro_f1,
        "probabilistic": probabilistic,
        "calibration": calibration,
        "roc_curve": roc_curve,
        "profit": profit,
        "betting": betting,
    }

    # Save per-day metrics
    day_dir = os.path.join(PRED_DIR, date_str)
    if os.path.isdir(day_dir):
        out_path = os.path.join(day_dir, "eval_metrics.json")
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Update tracker
    _update_tracker(date_str, metrics)

    # Print report
    if verbose:
        _print_report(date_str, metrics)

    return metrics


def evaluate_range(start_date, end_date, verbose=True):
    """Evaluate all dates in a range."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_metrics = []
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        m = evaluate_date(ds, verbose=verbose)
        if m:
            all_metrics.append(m)
        d += timedelta(days=1)

    if verbose and len(all_metrics) > 1:
        _print_cross_day_summary(all_metrics)

    return all_metrics


def _find_graded_file(date_str):
    """Find the main graded predictions file for a date (not parlay grading)."""
    day_dir = os.path.join(PRED_DIR, date_str)
    if not os.path.isdir(day_dir):
        return None

    # Prefer v4_graded_*_lines.json (main grading output)
    for f in sorted(os.listdir(day_dir)):
        if f.startswith("v4_graded") and f.endswith(".json"):
            return os.path.join(day_dir, f)

    # Fallback: any graded file that's not parlay-specific
    for f in sorted(os.listdir(day_dir)):
        if "graded" in f and f.endswith(".json") and "parlay" not in f:
            return os.path.join(day_dir, f)

    return None


def _update_tracker(date_str, metrics):
    """Update cumulative eval tracker."""
    os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)

    tracker = {}
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH) as f:
            tracker = json.load(f)

    # Extract summary-level metrics for tracker
    cls = metrics.get("classification", {}).get("overall", {})
    prob = metrics.get("probabilistic", {}).get("ensemble", {})
    cal = metrics.get("calibration", {})
    pft = metrics.get("profit", {})
    mf1 = metrics.get("macro_f1", {})

    tracker[date_str] = {
        "n": cls.get("n", 0),
        "accuracy": cls.get("accuracy", 0),
        "f1": cls.get("f1", 0),
        "macro_f1": mf1.get("macro_f1", 0),
        "brier": prob.get("brier_score", None),
        "log_loss": prob.get("log_loss", None),
        "roc_auc": prob.get("roc_auc", None),
        "ece": cal.get("ece", None),
        "roi": pft.get("total_roi", None),
        "pnl": pft.get("total_pnl", None),
        "optimal_threshold": pft.get("optimal_threshold", None),
    }

    with open(TRACKER_PATH, "w") as f:
        json.dump(tracker, f, indent=2)


def _print_report(date_str, metrics):
    """Print comprehensive evaluation report to terminal."""
    n = metrics["n_graded"]

    print(f"\n{'=' * 70}")
    print(f"  EVALUATION METRICS — {date_str} ({n} graded predictions)")
    print(f"{'=' * 70}")

    # ── Section 1: Classification ──
    cls = metrics["classification"]
    overall = cls.get("overall", {})
    mf1 = metrics["macro_f1"]

    print(f"\n  ┌─ CLASSIFICATION ─────────────────────────────────────────────┐")
    print(f"  │ Accuracy:  {overall.get('accuracy', 0):.1%}  ({overall.get('hits', 0)}/{overall.get('n', 0)})")
    print(f"  │ F1 Score:  {overall.get('f1', 0):.4f}")
    print(f"  │ Macro F1:  {mf1.get('macro_f1', 0):.4f}  (averaged across stat types)")

    # Per-direction
    by_dir = cls.get("by_direction", {})
    for d in ("OVER", "UNDER"):
        if d in by_dir:
            dd = by_dir[d]
            print(f"  │   {d:5s}:  {dd['accuracy']:.1%} acc, F1={dd['f1']:.4f}  (n={dd['n']})")

    # Per-stat F1
    stat_f1 = mf1.get("per_stat", {})
    if stat_f1:
        print(f"  │")
        print(f"  │ Per-Stat F1:")
        for stat, info in sorted(stat_f1.items(), key=lambda x: x[1]["f1"], reverse=True):
            bar = "#" * int(info["f1"] * 20)
            print(f"  │   {stat:5s}: F1={info['f1']:.3f}  acc={info['accuracy']:.1%}  n={info['n']:3d}  {bar}")
    print(f"  └────────────────────────────────────────────────────────────────┘")

    # ── Section 2: Probabilistic ──
    prob = metrics.get("probabilistic", {})
    if prob:
        print(f"\n  ┌─ PROBABILISTIC (lower Brier/LogLoss = better, higher AUC = better) ┐")
        print(f"  │ {'Model':<12s} {'Brier':>7s} {'Skill':>7s} {'LogLoss':>8s} {'AUC':>6s}  {'N':>5s}  │")
        print(f"  │ {'-'*12} {'-'*7} {'-'*7} {'-'*8} {'-'*6}  {'-'*5}  │")

        # Random baseline reference
        print(f"  │ {'(random)':<12s} {'0.2500':>7s} {'0.0%':>7s} {'0.6931':>8s} {'0.500':>6s}  {'':>5s}  │")

        for model, m in sorted(prob.items(), key=lambda x: x[1].get("brier_score", 1)):
            skill_pct = f"{m['brier_skill']:.1%}"
            print(
                f"  │ {model:<12s} {m['brier_score']:>7.4f} {skill_pct:>7s} "
                f"{m['log_loss']:>8.4f} {m['roc_auc']:>6.3f}  {m['n']:>5d}  │"
            )
        print(f"  └────────────────────────────────────────────────────────────────────┘")

    # ── Section 3: Calibration ──
    cal = metrics.get("calibration", {})
    if cal and "ece" in cal:
        print(f"\n  ┌─ CALIBRATION ──────────────────────────────────────────────┐")
        print(f"  │ ECE (Expected Calibration Error): {cal['ece']:.4f}")
        print(f"  │ MCE (Maximum Calibration Error):  {cal['mce']:.4f}")
        print(f"  │ Model: {cal.get('model', 'ensemble_prob')}")
        print(f"  │")

        bins = cal.get("bins", [])
        if bins:
            print(f"  │ {'Bin':>9s}  {'Pred':>6s}  {'Actual':>6s}  {'Gap':>5s}  {'N':>4s}  Reliability")
            print(f"  │ {'-'*9}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*4}  {'-'*20}")
            for b in bins:
                if b["n"] == 0:
                    continue
                gap_bar = "|" + "!" * min(int(b["gap"] * 100), 20)
                quality = "OK" if b["gap"] < 0.05 else ("~~" if b["gap"] < 0.10 else "!!")
                print(
                    f"  │ {b['range']:>9s}  {b['avg_predicted']:>6.3f}  {b['avg_actual']:>6.3f}  "
                    f"{b['gap']:>5.3f}  {b['n']:>4d}  {quality} {gap_bar}"
                )
        print(f"  └────────────────────────────────────────────────────────────┘")

    # ── Section 4: ROC Curve ──
    roc = metrics.get("roc_curve", [])
    if roc:
        print(f"\n  ┌─ ROC CURVE ────────────────────────────────────────────────┐")
        auc_val = prob.get("ensemble", {}).get("roc_auc", "?")
        print(f"  │ AUC = {auc_val}")
        print(render_ascii_roc(roc))
        print(f"  └────────────────────────────────────────────────────────────┘")

    # ── Section 5: Profit ──
    pft = metrics.get("profit", {})
    if pft and "total_bets" in pft:
        print(f"\n  ┌─ PROFITABILITY ─────────────────────────────────────────────┐")
        print(f"  │ Total bets: {pft['total_bets']}  Wins: {pft['total_wins']}  "
              f"P&L: ${pft['total_pnl']:+.2f}  ROI: {pft['total_roi']:+.1f}%")
        print(f"  │ Optimal threshold: {pft['optimal_threshold']:.2f} → ROI: {pft['optimal_roi']:+.1f}%")
        print(f"  │")

        # Precision@K
        pak = pft.get("precision_at_k", {})
        if pak:
            pak_str = "  ".join(f"{k}={v:.1%}" for k, v in sorted(pak.items()))
            print(f"  │ Precision@K: {pak_str}")

        # ROI by confidence
        roi_bins = pft.get("roi_by_confidence", [])
        if roi_bins:
            print(f"  │")
            print(f"  │ {'Confidence':>12s}  {'N':>4s}  {'Acc':>6s}  {'P&L':>7s}  {'ROI':>7s}  Bar")
            print(f"  │ {'-'*12}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*15}")
            for rb in roi_bins:
                bar_len = min(abs(int(rb["roi"] / 5)), 15)
                bar = ("+" if rb["roi"] >= 0 else "-") * bar_len
                print(
                    f"  │ {rb['range']:>12s}  {rb['n']:>4d}  {rb['accuracy']:>6.1%}  "
                    f"${rb['pnl']:>+6.2f}  {rb['roi']:>+6.1f}%  {bar}"
                )
        print(f"  └────────────────────────────────────────────────────────────┘")

    # ── Section 6: Betting Analytics ──
    bet = metrics.get("betting", {})
    ev = bet.get("ev_analysis", {})
    if ev.get("total_with_ev", 0) > 0:
        print(f"\n  ┌─ BETTING ANALYTICS ──────────────────────────────────────────┐")
        print(f"  │ +EV picks: {ev['positive_ev_count']}/{ev['total_with_ev']} "
              f"({ev['positive_ev_count']/ev['total_with_ev']:.1%})")
        print(f"  │ +EV accuracy: {ev['positive_ev_accuracy']:.1%}  "
              f"vs  -EV accuracy: {ev['negative_ev_accuracy']:.1%}")

        corr = bet.get("confidence_outcome_correlation")
        if corr is not None:
            print(f"  │ Confidence-outcome correlation: {corr:+.4f}")

        agreement = bet.get("model_agreement", {})
        if agreement:
            print(f"  │")
            print(f"  │ Model Agreement → Accuracy:")
            for bucket, stats in sorted(agreement.items()):
                bar = "#" * int(stats["accuracy"] * 20)
                print(f"  │   {bucket:>8s}: {stats['accuracy']:.1%} (n={stats['n']:3d})  {bar}")
        print(f"  └────────────────────────────────────────────────────────────┘")


def _print_cross_day_summary(all_metrics):
    """Print summary across multiple days."""
    print(f"\n{'=' * 70}")
    print(f"  CROSS-DAY SUMMARY ({len(all_metrics)} days)")
    print(f"{'=' * 70}")

    print(f"\n  {'Date':<12s} {'N':>5s} {'Acc':>6s} {'F1':>6s} {'MF1':>6s} "
          f"{'Brier':>7s} {'AUC':>6s} {'ECE':>6s} {'ROI':>7s}")
    print(f"  {'-'*12} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")

    for m in all_metrics:
        cls = m["classification"]["overall"]
        prob = m.get("probabilistic", {}).get("ensemble", {})
        cal = m.get("calibration", {})
        pft = m.get("profit", {})
        mf1 = m.get("macro_f1", {})

        brier_str = f"{prob['brier_score']:.4f}" if prob.get("brier_score") else "  --  "
        auc_str = f"{prob['roc_auc']:.3f}" if prob.get("roc_auc") else " -- "
        ece_str = f"{cal['ece']:.4f}" if cal.get("ece") else "  -- "
        roi_str = f"{pft['total_roi']:+.1f}%" if pft.get("total_roi") is not None else "  -- "

        print(
            f"  {m['date']:<12s} {cls['n']:>5d} {cls['accuracy']:>6.1%} "
            f"{cls['f1']:>6.4f} {mf1.get('macro_f1', 0):>6.4f} "
            f"{brier_str:>7s} {auc_str:>6s} {ece_str:>6s} {roi_str:>7s}"
        )

    # Averages
    avg_acc = sum(m["classification"]["overall"]["accuracy"] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m["classification"]["overall"]["f1"] for m in all_metrics) / len(all_metrics)
    avg_mf1 = sum(m["macro_f1"].get("macro_f1", 0) for m in all_metrics) / len(all_metrics)

    briers = [m["probabilistic"]["ensemble"]["brier_score"]
              for m in all_metrics if m.get("probabilistic", {}).get("ensemble")]
    avg_brier = sum(briers) / len(briers) if briers else None

    print(f"  {'-'*12} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*7}")
    brier_avg_str = f"{avg_brier:.4f}" if avg_brier else "  --  "
    print(f"  {'AVERAGE':<12s} {'':>5s} {avg_acc:>6.1%} {avg_f1:>6.4f} {avg_mf1:>6.4f} {brier_avg_str:>7s}")


def print_tracker_summary():
    """Print summary from the cumulative tracker."""
    if not os.path.exists(TRACKER_PATH):
        print("[INFO] No eval tracker found. Run evaluations first.")
        return

    with open(TRACKER_PATH) as f:
        tracker = json.load(f)

    if not tracker:
        print("[INFO] Tracker is empty.")
        return

    print(f"\n{'=' * 70}")
    print(f"  EVALUATION TRACKER — {len(tracker)} days")
    print(f"{'=' * 70}")
    print(f"\n  {'Date':<12s} {'N':>5s} {'Acc':>6s} {'F1':>6s} {'MF1':>6s} "
          f"{'Brier':>7s} {'AUC':>6s} {'ECE':>6s} {'ROI':>7s} {'P&L':>7s}")
    print(f"  {'-'*12} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")

    for date_str in sorted(tracker.keys()):
        t = tracker[date_str]
        n = t.get("n", 0)
        acc = t.get("accuracy", 0)
        f1 = t.get("f1", 0)
        mf1 = t.get("macro_f1", 0)

        brier = f"{t['brier']:.4f}" if t.get("brier") else "  --  "
        auc = f"{t['roc_auc']:.3f}" if t.get("roc_auc") else " -- "
        ece = f"{t['ece']:.4f}" if t.get("ece") else "  -- "
        roi = f"{t['roi']:+.1f}%" if t.get("roi") is not None else "  -- "
        pnl = f"${t['pnl']:+.0f}" if t.get("pnl") is not None else "  -- "

        print(
            f"  {date_str:<12s} {n:>5d} {acc:>6.1%} {f1:>6.4f} {mf1:>6.4f} "
            f"{brier:>7s} {auc:>6s} {ece:>6s} {roi:>7s} {pnl:>7s}"
        )


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python3 eval_metrics.py 2026-03-26              # Single day")
        print("  python3 eval_metrics.py 2026-03-26 2026-03-20   # Date range (newest first)")
        print("  python3 eval_metrics.py --all                   # All graded days")
        print("  python3 eval_metrics.py --summary               # Tracker summary")
        sys.exit(0)

    if args[0] == "--summary":
        print_tracker_summary()

    elif args[0] == "--all":
        # Find all graded days
        dates = []
        for entry in sorted(os.listdir(PRED_DIR)):
            if len(entry) == 10 and entry[4] == "-" and entry[7] == "-":
                graded = _find_graded_file(entry)
                if graded:
                    dates.append(entry)
        if dates:
            evaluate_range(dates[0], dates[-1])
        else:
            print("[INFO] No graded predictions found.")

    elif len(args) == 2:
        # Date range
        date1, date2 = args
        start = min(date1, date2)
        end = max(date1, date2)
        evaluate_range(start, end)

    elif len(args) == 1:
        evaluate_date(args[0])

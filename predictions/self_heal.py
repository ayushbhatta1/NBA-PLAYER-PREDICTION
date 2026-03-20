#!/usr/bin/env python3
"""
NBA Pipeline Self-Healing Module — v4
Analyzes grading results to detect systematic biases and generate corrections.

Similar to the esports corrections.md system but adapted for NBA props.

Correction types:
- STAT_BIAS: Systematic over/under-projection for a stat category
- MATCHUP_BIAS: Consistent misses against certain defensive profiles
- COMBO_PENALTY: Combo stats (PRA/PR/PA/RA) needing wider gaps
- STREAK_FALSE: Streak signals that don't predict outcomes
- BLOWOUT_MISS: Minutes reduction in blowout games
- INJURY_IMPACT: Incorrect teammate injury adjustments

Usage:
    from self_heal import analyze_misses, load_corrections, apply_corrections
"""

import json
import os
from datetime import datetime
from collections import defaultdict

CORRECTIONS_FILE = os.path.join(os.path.dirname(__file__), 'corrections.json')


def analyze_misses(graded_results):
    """
    Analyze graded results to detect systematic patterns in misses.
    Returns list of detected patterns with suggested corrections.
    """
    if not graded_results:
        return []

    results = graded_results if isinstance(graded_results, list) else graded_results.get('results', [])

    patterns = []

    # Normalize result format: handle both 'result'='HIT'/'MISS' and 'hit'=True/False
    for r in results:
        if 'result' not in r and 'hit' in r and r.get('actual') is not None:
            r['result'] = 'HIT' if r['hit'] else 'MISS'
        if 'projection_error' not in r and 'proj_error' in r:
            r['projection_error'] = r['proj_error']

    # Filter to graded picks only
    graded = [r for r in results if r.get('result') in ['HIT', 'MISS']]
    if len(graded) < 10:
        return []

    misses = [r for r in graded if r['result'] == 'MISS']
    hits = [r for r in graded if r['result'] == 'HIT']

    # ── PATTERN 1: Stat-specific bias ──
    stat_performance = defaultdict(lambda: {'hits': 0, 'misses': 0, 'errors': []})
    for r in graded:
        stat = r.get('stat', '')
        is_hit = r['result'] == 'HIT'
        stat_performance[stat]['hits' if is_hit else 'misses'] += 1
        if r.get('projection_error') is not None:
            stat_performance[stat]['errors'].append(r['projection_error'])

    for stat, perf in stat_performance.items():
        total = perf['hits'] + perf['misses']
        if total < 5:
            continue
        accuracy = perf['hits'] / total
        avg_error = sum(perf['errors']) / len(perf['errors']) if perf['errors'] else 0

        # If accuracy < 50% or systematic directional error
        if accuracy < 0.50 and total >= 8:
            patterns.append({
                'type': 'STAT_BIAS',
                'stat': stat,
                'accuracy': round(accuracy * 100, 1),
                'avg_error': round(avg_error, 1),
                'sample': total,
                'suggestion': f"{'Reduce' if avg_error > 0 else 'Increase'} {stat} projections by ~{abs(avg_error):.1f}",
                'severity': 'HIGH' if accuracy < 0.40 else 'MEDIUM',
            })

    # ── PATTERN 2: Direction bias ──
    over_graded = [r for r in graded if r.get('direction') == 'OVER']
    under_graded = [r for r in graded if r.get('direction') == 'UNDER']

    if len(over_graded) >= 10:
        over_acc = sum(1 for r in over_graded if r['result'] == 'HIT') / len(over_graded)
        if over_acc < 0.50:
            patterns.append({
                'type': 'DIRECTION_BIAS',
                'direction': 'OVER',
                'accuracy': round(over_acc * 100, 1),
                'sample': len(over_graded),
                'suggestion': 'OVER picks underperforming — consider tightening OVER tier thresholds',
                'severity': 'HIGH' if over_acc < 0.45 else 'MEDIUM',
            })

    if len(under_graded) >= 10:
        under_acc = sum(1 for r in under_graded if r['result'] == 'HIT') / len(under_graded)
        if under_acc > 0.75:
            patterns.append({
                'type': 'DIRECTION_BIAS',
                'direction': 'UNDER',
                'accuracy': round(under_acc * 100, 1),
                'sample': len(under_graded),
                'suggestion': 'UNDER picks outperforming — consider relaxing UNDER penalty further',
                'severity': 'LOW',
            })

    # ── PATTERN 3: Tier calibration ──
    tier_performance = defaultdict(lambda: {'hits': 0, 'total': 0})
    for r in graded:
        tier = r.get('tier', 'F')
        tier_performance[tier]['total'] += 1
        if r['result'] == 'HIT':
            tier_performance[tier]['hits'] += 1

    # S-tier should be >80%, A-tier >65%
    expected = {'S': 0.80, 'A': 0.65, 'B': 0.55, 'C': 0.50}
    for tier, expected_acc in expected.items():
        tp = tier_performance[tier]
        if tp['total'] < 3:
            continue
        actual_acc = tp['hits'] / tp['total']
        if actual_acc < expected_acc - 0.15:
            patterns.append({
                'type': 'TIER_MISCALIBRATION',
                'tier': tier,
                'expected': round(expected_acc * 100, 1),
                'actual': round(actual_acc * 100, 1),
                'sample': tp['total'],
                'suggestion': f"{tier}-tier threshold may be too loose — increase gap requirement",
                'severity': 'HIGH',
            })

    # ── PATTERN 4: Blowout game misses ──
    blowout_misses = [r for r in misses if r.get('spread') and abs(r.get('spread', 0)) >= 10]
    blowout_all = [r for r in graded if r.get('spread') and abs(r.get('spread', 0)) >= 10]
    if len(blowout_all) >= 5:
        blowout_acc = (len(blowout_all) - len(blowout_misses)) / len(blowout_all)
        if blowout_acc < 0.50:
            patterns.append({
                'type': 'BLOWOUT_MISS',
                'accuracy': round(blowout_acc * 100, 1),
                'sample': len(blowout_all),
                'suggestion': 'Blowout games underperforming — strengthen blowout adjustment or avoid',
                'severity': 'MEDIUM',
            })

    # ── PATTERN 5: Combo stat volatility ──
    combo_misses = [r for r in misses if r.get('stat') in ['pra', 'pr', 'pa', 'ra']]
    combo_all = [r for r in graded if r.get('stat') in ['pra', 'pr', 'pa', 'ra']]
    if len(combo_all) >= 8:
        combo_acc = (len(combo_all) - len(combo_misses)) / len(combo_all)
        if combo_acc < 0.55:
            patterns.append({
                'type': 'COMBO_PENALTY',
                'accuracy': round(combo_acc * 100, 1),
                'sample': len(combo_all),
                'suggestion': 'Combo stats underperforming — increase COMBO_GAP_PENALTY',
                'severity': 'MEDIUM',
            })

    return patterns


def save_corrections(patterns, date_str=None):
    """Save detected patterns as corrections."""
    if not patterns:
        return

    corrections = load_corrections()
    date_str = date_str or datetime.now().strftime('%Y-%m-%d')

    for p in patterns:
        correction = {
            'id': f"{p['type']}_{p.get('stat', p.get('tier', p.get('direction', 'general')))}",
            'detected_on': date_str,
            'type': p['type'],
            'details': p,
            'status': 'ACTIVE',
            'applied': False,
        }

        # Update existing or add new
        existing_idx = None
        for i, c in enumerate(corrections):
            if c['id'] == correction['id']:
                existing_idx = i
                break

        if existing_idx is not None:
            corrections[existing_idx] = correction
        else:
            corrections.append(correction)

    with open(CORRECTIONS_FILE, 'w') as f:
        json.dump(corrections, f, indent=2)

    print(f"[SELF-HEAL] Saved {len(patterns)} corrections to {CORRECTIONS_FILE}")


def load_corrections():
    """Load active corrections."""
    if os.path.exists(CORRECTIONS_FILE):
        with open(CORRECTIONS_FILE) as f:
            return json.load(f)
    return []


def apply_corrections(result):
    """
    Apply active corrections to a single pipeline result.
    Modifies projection and tier based on known biases.
    Returns the modified result.
    """
    corrections = load_corrections()
    if not corrections:
        return result

    applied = []
    for c in corrections:
        if c.get('status') != 'ACTIVE':
            continue

        details = c.get('details', {})

        # Apply stat bias correction
        if c['type'] == 'STAT_BIAS' and details.get('stat') == result.get('stat'):
            avg_error = details.get('avg_error', 0)
            if abs(avg_error) > 1.0:
                adjustment = -avg_error * 0.5  # apply half the correction
                result['projection'] = round(result.get('projection', 0) + adjustment, 1)
                result['gap'] = round(result['projection'] - result.get('line', 0), 1)
                result['abs_gap'] = round(abs(result['gap']), 1)
                applied.append(f"STAT_BIAS({details['stat']}): adj {adjustment:+.1f}")

    if applied:
        result['corrections_applied'] = applied

    return result


def print_corrections_report():
    """Print summary of all active corrections."""
    corrections = load_corrections()
    if not corrections:
        print("[SELF-HEAL] No corrections on file.")
        return

    active = [c for c in corrections if c.get('status') == 'ACTIVE']
    print(f"\n{'='*60}")
    print(f"  ACTIVE CORRECTIONS ({len(active)})")
    print(f"{'='*60}")

    for c in active:
        d = c.get('details', {})
        severity = d.get('severity', 'UNKNOWN')
        print(f"\n  [{severity}] {c['id']}")
        print(f"    Detected: {c.get('detected_on', 'unknown')}")
        print(f"    Issue: {d.get('suggestion', 'N/A')}")
        if 'accuracy' in d:
            print(f"    Accuracy: {d['accuracy']}% (n={d.get('sample', '?')})")


if __name__ == '__main__':
    print("NBA Pipeline Self-Healing Module")
    print("Usage: from self_heal import analyze_misses, save_corrections")
    print_corrections_report()

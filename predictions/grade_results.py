#!/usr/bin/env python3
"""
Auto-grading script for NBA prop predictions — v4.
Now uses nba_api live box scores instead of static CSV database.

Usage: python3 grade_results.py 2026-03-12
"""
import sys, json, os
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def grade_date(date_str, predictions_file=None):
    """Grade all predictions for a given date using live box score data."""
    pred_dir = os.path.join(os.path.dirname(__file__), date_str)

    # Find predictions file
    if predictions_file is None:
        candidates = [
            os.path.join(pred_dir, f"{date_str}_full_board.json"),
            os.path.join(pred_dir, f"{date_str}_all_lines_predictions.json"),
            os.path.join(pred_dir, "v3_full_predictions.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                predictions_file = c
                break

    if not predictions_file or not os.path.exists(predictions_file):
        print(f"No predictions found for {date_str}")
        return None

    # Load predictions
    with open(predictions_file) as f:
        data = json.load(f)

    # Handle both formats: list of predictions or dict with 'line_predictions'
    if isinstance(data, list):
        preds = data
    elif isinstance(data, dict) and 'line_predictions' in data:
        preds = data['line_predictions']
    else:
        preds = data

    # Fetch actuals via nba_api (replaces CSV dependency)
    print(f"Fetching box scores for {date_str} via nba_api...")
    try:
        from analyze_v3 import get_fetcher
        fetcher = get_fetcher()
        actuals = fetcher.get_box_scores(date_str)
        if not actuals:
            print(f"[WARN] No box scores from nba_api for {date_str}. Games may not be finished.")
            # Fallback to CSV if available
            actuals = _fallback_csv_actuals(date_str)
            if not actuals:
                print("[ERROR] No actuals available from any source.")
                return None
        else:
            print(f"  Got stats for {len(actuals)} players from nba_api")
    except Exception as e:
        print(f"[WARN] nba_api fetch failed: {e}. Trying CSV fallback...")
        actuals = _fallback_csv_actuals(date_str)
        if not actuals:
            return None

    # Grade each prediction
    results = []
    hits = 0
    misses = 0
    over_hits = 0
    over_total = 0
    under_hits = 0
    under_total = 0
    proj_errors = []

    tier_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
    stat_stats = defaultdict(lambda: {'hits': 0, 'total': 0})

    for p in preds:
        player = p.get('player', '')
        stat = p.get('stat', '')
        line = p.get('line', 0)
        direction = p.get('direction', p.get('pick', 'SKIP'))
        tier = p.get('tier', 'SKIP')
        proj = p.get('projection', 0)

        if direction == 'SKIP' or tier == 'SKIP' or 'error' in p:
            results.append({**p, 'actual': None, 'result': 'SKIP', 'margin': None})
            continue

        # Find player in actuals (fuzzy match)
        actual_val = _find_actual(player, stat, actuals, team=p.get('team', ''))

        if actual_val is None:
            results.append({**p, 'actual': None, 'result': 'DNP', 'margin': None})
            continue

        margin = actual_val - line
        proj_error = proj - actual_val if proj else 0

        # Push = actual equals line (void, neither hit nor miss)
        if actual_val == line:
            results.append({
                **p,
                'actual': actual_val,
                'result': 'PUSH',
                'margin': 0.0,
                'projection_error': round(proj_error, 1),
            })
            # Don't include pushes in proj_errors — they're voided picks
            continue

        if direction == 'OVER':
            hit = actual_val > line
            over_total += 1
            if hit:
                over_hits += 1
        else:
            hit = actual_val < line
            under_total += 1
            if hit:
                under_hits += 1

        if hit:
            hits += 1
        else:
            misses += 1

        # Track by tier and stat
        tier_stats[tier]['total'] += 1
        if hit:
            tier_stats[tier]['hits'] += 1
        stat_stats[stat]['total'] += 1
        if hit:
            stat_stats[stat]['hits'] += 1

        proj_errors.append(abs(proj_error))

        results.append({
            **p,
            'actual': actual_val,
            'result': 'HIT' if hit else 'MISS',
            'margin': round(margin, 1),
            'projection_error': round(proj_error, 1),
        })

    total = hits + misses
    accuracy = round(hits / total * 100, 1) if total > 0 else 0
    mae = round(sum(proj_errors) / len(proj_errors), 1) if proj_errors else 0

    # Print report
    print(f"\n{'='*70}")
    print(f"  GRADING REPORT — {date_str}")
    print(f"{'='*70}")
    print(f"  Total lines analyzed: {len(preds)}")
    print(f"  Graded (matched): {total}")
    print(f"  Overall accuracy: {hits}/{total} ({accuracy}%)")
    print(f"  OVER accuracy: {over_hits}/{over_total} ({round(over_hits/over_total*100,1) if over_total else 0}%)")
    print(f"  UNDER accuracy: {under_hits}/{under_total} ({round(under_hits/under_total*100,1) if under_total else 0}%)")
    print(f"  Projection MAE: {mae}")

    print(f"\n  By tier:")
    for tier in ['S', 'A', 'B', 'C', 'D', 'F']:
        ts = tier_stats[tier]
        if ts['total'] > 0:
            pct = round(ts['hits'] / ts['total'] * 100, 1)
            print(f"    {tier}: {ts['hits']}/{ts['total']} ({pct}%)")

    print(f"\n  By stat:")
    for s in sorted(stat_stats.keys()):
        ss = stat_stats[s]
        if ss['total'] > 0:
            pct = round(ss['hits'] / ss['total'] * 100, 1)
            print(f"    {s:5s}: {ss['hits']}/{ss['total']} ({pct}%)")

    # Build summary
    summary = {
        'date': date_str,
        'graded_at': datetime.now().isoformat(),
        'pipeline_version': 'v4',
        'total_lines': len(preds),
        'graded': total,
        'hits': hits,
        'misses': misses,
        'accuracy': accuracy,
        'over_accuracy': round(over_hits / over_total * 100, 1) if over_total else 0,
        'under_accuracy': round(under_hits / under_total * 100, 1) if under_total else 0,
        'mae': mae,
        'by_tier': {t: {'hits': v['hits'], 'total': v['total'],
                        'accuracy': round(v['hits']/v['total']*100, 1) if v['total'] else 0}
                    for t, v in tier_stats.items()},
        'by_stat': {s: {'hits': v['hits'], 'total': v['total'],
                        'accuracy': round(v['hits']/v['total']*100, 1) if v['total'] else 0}
                    for s, v in stat_stats.items()},
    }

    # Save grading results
    os.makedirs(pred_dir, exist_ok=True)
    outfile = os.path.join(pred_dir, f"v4_graded_{total}_lines.json")
    with open(outfile, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)

    # Update accuracy log
    _update_accuracy_log(date_str, summary)

    print(f"  Updated accuracy log")
    print(f"\n  Saved to {outfile}")

    # Run evaluation metrics automatically
    try:
        from eval_metrics import evaluate_date
        evaluate_date(date_str, verbose=True)
    except Exception as e:
        print(f"  [WARN] Eval metrics skipped: {e}")

    return {'summary': summary, 'results': results}


def _find_actual(player_name, stat, actuals, team=None):
    """Find a player's actual stat value with fuzzy matching.

    When team is provided and fuzzy matching (not exact), prefer the match
    whose team matches to disambiguate players with similar names.
    """
    import unicodedata
    def _norm(s):
        nfkd = unicodedata.normalize('NFKD', s)
        return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

    def _get_val(stats_dict):
        val = stats_dict.get(stat)
        if val is None:
            val = stats_dict.get(stat.lower())
        return val

    p_norm = _norm(player_name)
    p_parts = p_norm.split()

    # Collect fuzzy matches for disambiguation
    fuzzy_matches = []

    for name, stats in actuals.items():
        n_norm = _norm(name)
        # Exact match — return immediately
        if p_norm == n_norm:
            return _get_val(stats)
        # Last name + first initial
        n_parts = n_norm.split()
        if len(p_parts) >= 2 and len(n_parts) >= 2:
            if p_parts[-1] == n_parts[-1] and p_parts[0][0] == n_parts[0][0]:
                fuzzy_matches.append((name, stats))

    if not fuzzy_matches:
        return None

    # If team provided, prefer match with matching team
    if team:
        team_upper = team.upper()
        for name, stats in fuzzy_matches:
            if stats.get('team', '').upper() == team_upper:
                return _get_val(stats)

    # Fallback to first fuzzy match
    return _get_val(fuzzy_matches[0][1])


def _fallback_csv_actuals(date_str):
    """Fallback: load actuals from CSV database if nba_api fails."""
    try:
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'NBA Database (1947 - Present)', 'PlayerStatistics.csv')
        if not os.path.exists(csv_path):
            return None

        df = pd.read_csv(csv_path, low_memory=False)
        df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'], errors='coerce')
        game_day = df[df['gameDateTimeEst'].dt.date.astype(str) == date_str].copy()

        if game_day.empty:
            return None

        for c in ['points', 'assists', 'reboundsTotal', 'steals', 'blocks', 'threePointersMade']:
            game_day[c] = pd.to_numeric(game_day[c], errors='coerce').fillna(0)

        actuals = {}
        for _, row in game_day.iterrows():
            name = f"{row['firstName']} {row['lastName']}"
            pts = int(row['points'])
            reb = int(row['reboundsTotal'])
            ast = int(row['assists'])
            actuals[name] = {
                'pts': pts, 'reb': reb, 'ast': ast,
                '3pm': int(row['threePointersMade']),
                'stl': int(row['steals']), 'blk': int(row['blocks']),
                'pra': pts + reb + ast,
                'pr': pts + reb, 'pa': pts + ast, 'ra': reb + ast,
                'stl_blk': int(row['steals']) + int(row['blocks']),
            }
        print(f"  CSV fallback: {len(actuals)} players from database")
        return actuals
    except Exception as e:
        print(f"  CSV fallback failed: {e}")
        return None


def _update_accuracy_log(date_str, summary):
    """Update the master accuracy log."""
    log_file = os.path.join(os.path.dirname(__file__), 'logs', 'accuracy_log.json')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = {"dates": {}, "entries": []}

    # Update/add entry
    entry = {
        'date': date_str,
        'pipeline': 'v4',
        'total_lines': summary['total_lines'],
        'hits': summary['hits'],
        'accuracy_pct': summary['accuracy'],
        'over_accuracy': summary['over_accuracy'],
        'under_accuracy': summary['under_accuracy'],
        'tier_accuracy': summary['by_tier'],
        'mae': summary['mae'],
        'data_source': 'nba_api live',
        'timestamp': summary['graded_at'],
    }

    # Update entries list (replace if same date exists)
    log['entries'] = [e for e in log.get('entries', []) if e.get('date') != date_str]
    log['entries'].append(entry)
    log['entries'].sort(key=lambda x: x.get('date', ''))

    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  Updated {log_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        grade_date(sys.argv[1])
    else:
        print("Usage: python3 grade_results.py YYYY-MM-DD")

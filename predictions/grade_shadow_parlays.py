#!/usr/bin/env python3
"""
Shadow Parlay Grader — Strategy Backtesting Lab

Grades 20 shadow parlays against actual box scores and maintains
a cumulative tracker/leaderboard across days.

Usage: python3 grade_shadow_parlays.py 2026-03-13
"""
import sys
import json
import os
import unicodedata
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_FILE = os.path.join(PREDICTIONS_DIR, 'shadow_parlay_tracker.json')


def _norm(s):
    nfkd = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _find_actual(player_name, stat, actuals, team=None):
    """Find a player's actual stat value with fuzzy matching.

    When team is provided and fuzzy matching (not exact), prefer the match
    whose team matches to disambiguate players with similar names.
    """
    p_norm = _norm(player_name)
    p_parts = p_norm.split()

    def _get_val(stats_dict):
        val = stats_dict.get(stat)
        if val is None:
            val = stats_dict.get(stat.lower())
        return val

    fuzzy_matches = []

    for name, stats in actuals.items():
        n_norm = _norm(name)
        if p_norm == n_norm:
            return _get_val(stats)
        n_parts = n_norm.split()
        if len(p_parts) >= 2 and len(n_parts) >= 2:
            if p_parts[-1] == n_parts[-1] and p_parts[0][0] == n_parts[0][0]:
                fuzzy_matches.append((name, stats))

    if not fuzzy_matches:
        return None

    if team:
        team_upper = team.upper()
        for name, stats in fuzzy_matches:
            if stats.get('team', '').upper() == team_upper:
                return _get_val(stats)

    return _get_val(fuzzy_matches[0][1])


def grade_shadow_parlays(date_str):
    """Grade all shadow parlays for a given date."""
    shadow_file = os.path.join(PREDICTIONS_DIR, date_str, 'shadow_parlays.json')
    if not os.path.exists(shadow_file):
        print(f"No shadow parlays found for {date_str}")
        return None

    with open(shadow_file) as f:
        data = json.load(f)

    shadow_parlays = data.get('shadow_parlays', [])
    if not shadow_parlays:
        print(f"Empty shadow parlays for {date_str}")
        return None

    # Fetch actuals via nba_api
    print(f"Fetching box scores for {date_str} via nba_api...")
    try:
        from analyze_v3 import get_fetcher
        fetcher = get_fetcher()
        actuals = fetcher.get_box_scores(date_str)
        if not actuals:
            print(f"[WARN] No box scores for {date_str}. Games may not be finished.")
            return None
        print(f"  Got stats for {len(actuals)} players")
    except Exception as e:
        print(f"[ERROR] Failed to fetch box scores: {e}")
        return None

    # Grade each shadow parlay
    graded = []
    for sp in shadow_parlays:
        strategy = sp['strategy_name']
        legs = sp.get('legs', [])

        if sp.get('result') == 'no_build' or len(legs) < 3:
            graded.append({
                **sp,
                'result': 'no_build',
                'legs_hit': 0,
                'leg_results': [],
            })
            continue

        leg_results = []
        hits = 0
        has_dnp = False

        for leg in legs:
            player = leg.get('player', '')
            stat = leg.get('stat', '')
            line = leg.get('line', 0)
            direction = leg.get('direction', '')

            actual_val = _find_actual(player, stat, actuals, team=leg.get('team', ''))

            if actual_val is None:
                leg_results.append({
                    'player': player, 'stat': stat, 'line': line,
                    'direction': direction, 'actual': None, 'result': 'DNP',
                })
                has_dnp = True
                continue

            if direction == 'OVER':
                hit = actual_val > line
            else:
                hit = actual_val < line

            if hit:
                hits += 1

            leg_results.append({
                'player': player, 'stat': stat, 'line': line,
                'direction': direction, 'actual': actual_val,
                'result': 'HIT' if hit else 'MISS',
                'margin': round(actual_val - line, 1),
            })

        if has_dnp:
            parlay_result = 'DNP'
        elif hits == len(legs):
            parlay_result = 'HIT'
        else:
            parlay_result = 'MISS'

        graded.append({
            **sp,
            'result': parlay_result,
            'legs_hit': hits,
            'leg_results': leg_results,
        })

    # Print report
    print(f"\n{'='*65}")
    print(f"  SHADOW PARLAY GRADING — {date_str}")
    print(f"{'='*65}")

    for sp in graded:
        name = sp['strategy_name']
        result = sp['result']
        legs_hit = sp.get('legs_hit', 0)
        legs_total = sp.get('legs_total', 0)
        tag = {'HIT': 'HIT ', 'MISS': 'MISS', 'DNP': 'DNP ', 'no_build': 'SKIP'}.get(result, '????')
        print(f"  [{tag}] {name:25s} {legs_hit}/{legs_total} legs", end='')
        if result in ('HIT', 'MISS'):
            details = []
            for lr in sp.get('leg_results', []):
                if lr['result'] == 'DNP':
                    details.append(f"{lr['player'][:12]}=DNP")
                else:
                    details.append(f"{lr['player'][:12]}={lr.get('actual',0)}")
            print(f"  ({', '.join(details)})", end='')
        print()

    hits = sum(1 for s in graded if s['result'] == 'HIT')
    misses = sum(1 for s in graded if s['result'] == 'MISS')
    total = hits + misses
    print(f"\n  Summary: {hits}/{total} strategies HIT ({round(hits/total*100,1) if total else 0}%)")

    # Save graded results
    graded_file = os.path.join(PREDICTIONS_DIR, date_str, 'shadow_parlays_graded.json')
    with open(graded_file, 'w') as f:
        json.dump({
            'date': date_str,
            'graded_at': datetime.now().isoformat(),
            'total_strategies': len(graded),
            'hits': hits,
            'misses': misses,
            'shadow_parlays': graded,
        }, f, indent=2)
    print(f"\n  Saved graded: {graded_file}")

    # Update tracker
    update_tracker(date_str, graded)

    return graded


def update_tracker(date_str, graded):
    """Append graded results to cumulative tracker and rebuild leaderboard."""
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE) as f:
            tracker = json.load(f)
    else:
        tracker = {
            'last_updated': None,
            'total_days': 0,
            'strategies': {},
            'leaderboard': [],
        }

    # Remove existing entry for this date (re-grade support)
    for strat_name, strat_data in tracker['strategies'].items():
        strat_data['daily_results'] = [
            d for d in strat_data.get('daily_results', []) if d.get('date') != date_str
        ]

    # Add today's results
    dates_seen = set()
    for sp in graded:
        strategy = sp['strategy_name']
        result = sp['result']

        if strategy not in tracker['strategies']:
            tracker['strategies'][strategy] = {
                'description': sp.get('strategy_description', ''),
                'days': 0, 'wins': 0, 'losses': 0,
                'win_rate': 0.0, 'avg_legs_hit': 0.0,
                'current_streak': 0, 'streak_type': None,
                'daily_results': [],
            }

        strat = tracker['strategies'][strategy]
        strat['description'] = sp.get('strategy_description', strat.get('description', ''))

        daily_entry = {
            'date': date_str,
            'result': result,
            'legs_hit': sp.get('legs_hit', 0),
            'legs_total': sp.get('legs_total', 0),
        }
        strat['daily_results'].append(daily_entry)
        dates_seen.add(date_str)

    # Recalculate stats for each strategy
    all_dates = set()
    for strat_name, strat_data in tracker['strategies'].items():
        results_list = strat_data.get('daily_results', [])
        for d in results_list:
            all_dates.add(d['date'])

        # Only count HIT/MISS (exclude DNP, no_build)
        countable = [d for d in results_list if d['result'] in ('HIT', 'MISS')]
        wins = sum(1 for d in countable if d['result'] == 'HIT')
        losses = sum(1 for d in countable if d['result'] == 'MISS')
        total = wins + losses

        strat_data['days'] = len(countable)
        strat_data['wins'] = wins
        strat_data['losses'] = losses
        strat_data['win_rate'] = round(wins / total * 100, 1) if total > 0 else 0.0

        # Avg legs hit (across countable days)
        legs_hit_sum = sum(d.get('legs_hit', 0) for d in countable)
        strat_data['avg_legs_hit'] = round(legs_hit_sum / len(countable), 1) if countable else 0.0

        # Current streak
        sorted_results = sorted(countable, key=lambda d: d['date'], reverse=True)
        if sorted_results:
            streak_type = sorted_results[0]['result']
            streak = 0
            for d in sorted_results:
                if d['result'] == streak_type:
                    streak += 1
                else:
                    break
            strat_data['current_streak'] = streak
            strat_data['streak_type'] = 'W' if streak_type == 'HIT' else 'L'
        else:
            strat_data['current_streak'] = 0
            strat_data['streak_type'] = None

    tracker['total_days'] = len(all_dates)
    tracker['last_updated'] = datetime.now().isoformat()

    # Build leaderboard (sorted by win_rate desc, then wins desc)
    leaderboard = []
    for strat_name, strat_data in tracker['strategies'].items():
        if strat_data['days'] > 0:
            leaderboard.append({
                'strategy': strat_name,
                'win_rate': strat_data['win_rate'],
                'wins': strat_data['wins'],
                'losses': strat_data['losses'],
                'days': strat_data['days'],
                'avg_legs_hit': strat_data['avg_legs_hit'],
                'streak': f"{strat_data['streak_type']}{strat_data['current_streak']}",
            })

    leaderboard.sort(key=lambda x: (x['win_rate'], x['wins']), reverse=True)
    for i, entry in enumerate(leaderboard):
        entry['rank'] = i + 1

    tracker['leaderboard'] = leaderboard

    # Save
    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)

    # Print leaderboard
    print(f"\n{'='*65}")
    print(f"  STRATEGY LEADERBOARD ({tracker['total_days']} days)")
    print(f"{'='*65}")
    print(f"  {'Rank':4s} {'Strategy':25s} {'W-L':8s} {'Win%':7s} {'AvgLegs':8s} {'Streak':7s}")
    print(f"  {'-'*60}")
    for entry in leaderboard:
        wl = f"{entry['wins']}-{entry['losses']}"
        print(f"  {entry['rank']:4d} {entry['strategy']:25s} "
              f"{wl:<8s} "
              f"{entry['win_rate']:5.1f}%  "
              f"{entry['avg_legs_hit']:5.1f}    "
              f"{entry['streak']:5s}")

    print(f"\n  Saved tracker: {TRACKER_FILE}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        grade_shadow_parlays(sys.argv[1])
    else:
        print("Usage: python3 grade_shadow_parlays.py YYYY-MM-DD")

#!/usr/bin/env python3
"""
MEGA BACKTEST: Explore 1000+ strategies across 10 years of NBA data.
Tests every dimension: variance, floor, streaks, day-of-week, season phase,
opponent, rest days, line shapes, parlay sizes, selectivity thresholds,
multi-factor combos, and more.

Goal: Find a strategy that can win 10-20 days in a row.
"""

import csv, json, sys, os, math, random
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "NBA Database (1947 - Present)", "PlayerStatistics.csv")

STAT_MAP = {
    'pts': 'points',
    'reb': 'reboundsTotal',
    'ast': 'assists',
    '3pm': 'threePointersMade',
    'blk': 'blocks',
    'stl': 'steals',
}
COMBO_MAP = {
    'pra': ['points', 'reboundsTotal', 'assists'],
    'pr': ['points', 'reboundsTotal'],
    'pa': ['points', 'assists'],
    'ra': ['reboundsTotal', 'assists'],
}

MIN_LINES = {
    'pts': 5, 'reb': 2, 'ast': 1, '3pm': 0.5, 'blk': 0.5, 'stl': 0.5,
    'pra': 10, 'pr': 7, 'pa': 6, 'ra': 3,
}

def get_stat_val(row, stat):
    if stat in STAT_MAP:
        return float(row.get(STAT_MAP[stat], 0) or 0)
    if stat in COMBO_MAP:
        return sum(float(row.get(c, 0) or 0) for c in COMBO_MAP[stat])
    return 0

def parse_minutes(s):
    if not s: return 0
    try:
        if ':' in str(s): p = str(s).split(':'); return float(p[0]) + float(p[1])/60
        return float(s)
    except: return 0

def load_data():
    """Load all player games from CSV, sorted by date."""
    print("Loading CSV...", file=sys.stderr)
    player_games = defaultdict(list)
    count = 0
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            date = row.get('gameDateTimeEst', '')
            if not date or date < '2016-01-01': continue
            if row.get('gameType', '') != 'Regular Season': continue
            mins = parse_minutes(row.get('numMinutes', 0))
            if mins < 10: continue
            pid = row.get('personId', '')
            if not pid: continue

            home_val = row.get('home', '')
            is_home = True if home_val in ('True', 'true', '1') else False if home_val in ('False', 'false', '0') else None
            win_val = row.get('win', '')
            is_win = True if win_val in ('True', 'true', '1') else False

            player_games[pid].append({
                'date': date[:10],
                'name': f"{row.get('firstName','')} {row.get('lastName','')}",
                'team': row.get('playerteamName', ''),
                'opp': row.get('opponentteamName', ''),
                'home': is_home,
                'win': is_win,
                'mins': mins,
                'pts': float(row.get('points', 0) or 0),
                'reb': float(row.get('reboundsTotal', 0) or 0),
                'ast': float(row.get('assists', 0) or 0),
                '3pm': float(row.get('threePointersMade', 0) or 0),
                'blk': float(row.get('blocks', 0) or 0),
                'stl': float(row.get('steals', 0) or 0),
                'pf': float(row.get('foulsPersonal', 0) or 0),
                'to': float(row.get('turnovers', 0) or 0),
                'pm': float(row.get('plusMinusPoints', 0) or 0),
                'fga': float(row.get('fieldGoalsAttempted', 0) or 0),
                'fgm': float(row.get('fieldGoalsMade', 0) or 0),
                'fta': float(row.get('freeThrowsAttempted', 0) or 0),
                'ftm': float(row.get('freeThrowsMade', 0) or 0),
                'pra': float(row.get('points',0) or 0)+float(row.get('reboundsTotal',0) or 0)+float(row.get('assists',0) or 0),
            })
            count += 1

    print(f"Loaded {count:,} games for {len(player_games)} players", file=sys.stderr)
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g['date'])
    return player_games


def compute_props(player_games, stats=['pts','reb','ast','3pm','blk','stl','pra'],
                  inflation=4):
    """Generate all testable props from player history.
    Returns list of dicts with features + actual outcome."""

    props = []
    for pid, games in player_games.items():
        if len(games) < 15: continue  # need enough history

        for i in range(15, len(games)):
            current = games[i]
            prior = games[:i]

            for stat in stats:
                l10 = prior[-10:]
                l10_vals = [g[stat] for g in l10]
                l10_avg = sum(l10_vals) / len(l10_vals)

                # Skip unrealistic props
                if l10_avg < MIN_LINES.get(stat, 0.5): continue

                # Simulate sportsbook line (L10 avg + markup)
                line = round((l10_avg * (1 + inflation/100)) * 2) / 2
                actual = current[stat]
                if actual == line: continue  # push

                actual_under = actual < line

                # === COMPUTE ALL FEATURES ===
                l5 = prior[-5:]
                l3 = prior[-3:]
                l20 = prior[-20:] if len(prior) >= 20 else prior[-10:]
                season_games = [g for g in prior if g['date'][:4] == current['date'][:4]] or prior[-30:]

                l5_vals = [g[stat] for g in l5]
                l3_vals = [g[stat] for g in l3]
                l20_vals = [g[stat] for g in l20]
                season_vals = [g[stat] for g in season_games]

                l5_avg = sum(l5_vals)/len(l5_vals)
                l3_avg = sum(l3_vals)/len(l3_vals)
                l20_avg = sum(l20_vals)/len(l20_vals)
                season_avg = sum(season_vals)/len(season_vals)

                gap = l10_avg - line  # negative = line above average

                l10_hr = sum(1 for v in l10_vals if v > line) / 10 * 100
                l5_hr = sum(1 for v in l5_vals if v > line) / 5 * 100
                season_hr = sum(1 for v in season_vals if v > line) / len(season_vals) * 100

                l10_miss = sum(1 for v in l10_vals if v < line)
                l5_miss = sum(1 for v in l5_vals if v < line)

                # Variance/consistency
                l10_std = (sum((v - l10_avg)**2 for v in l10_vals) / len(l10_vals)) ** 0.5
                l10_cv = l10_std / max(l10_avg, 0.1)  # coefficient of variation
                l10_min = min(l10_vals)
                l10_max = max(l10_vals)
                l10_range = l10_max - l10_min
                l10_floor = sorted(l10_vals)[1] if len(l10_vals) > 1 else l10_min  # 2nd lowest

                # Floor analysis: how often does player hit their floor?
                floor_pct = sum(1 for v in l10_vals if v >= l10_floor) / 10 * 100

                # Streak detection
                if l3_avg > l10_avg * 1.15: streak = 'HOT'
                elif l3_avg < l10_avg * 0.85: streak = 'COLD'
                else: streak = 'NEUTRAL'

                # Trend (L5 vs L10)
                trend = (l5_avg - l10_avg) / max(l10_avg, 0.1) * 100

                # Minutes stability
                l10_mins = [g['mins'] for g in l10]
                mins_avg = sum(l10_mins)/len(l10_mins)
                mins_std = (sum((m - mins_avg)**2 for m in l10_mins)/len(l10_mins))**0.5
                mins_stable = mins_std < 3  # low variance in minutes

                # Plus/minus trend
                l5_pm = sum(g['pm'] for g in l5) / 5
                l10_pm = sum(g['pm'] for g in l10) / 10

                # Day of week
                try:
                    dow = datetime.strptime(current['date'], '%Y-%m-%d').weekday()
                except: dow = -1

                # Month / season phase
                month = int(current['date'][5:7])

                # Rest days (gap between games)
                if i > 0:
                    try:
                        d1 = datetime.strptime(prior[-1]['date'], '%Y-%m-%d')
                        d2 = datetime.strptime(current['date'], '%Y-%m-%d')
                        rest_days = (d2 - d1).days - 1
                    except: rest_days = 1
                else:
                    rest_days = 1
                is_b2b = rest_days == 0

                # Foul trouble tendency
                l10_pf = sum(g['pf'] for g in l10) / 10
                foul_prone = l10_pf >= 3.5

                # Usage proxy (FGA + FTA) / mins
                l10_usage = sum(g['fga'] + 0.44*g['fta'] for g in l10) / max(sum(g['mins'] for g in l10), 1) * 36

                # Consecutive unders (how many of last N went under?)
                consec_under = 0
                for v in reversed(l10_vals):
                    if v < line: consec_under += 1
                    else: break

                props.append({
                    'date': current['date'],
                    'name': current['name'],
                    'stat': stat,
                    'line': line,
                    'actual': actual,
                    'under': actual_under,
                    'home': current['home'],
                    'is_away': not current['home'] if current['home'] is not None else None,
                    'opp': current['opp'],
                    # Core stats
                    'l10_avg': l10_avg, 'l5_avg': l5_avg, 'l3_avg': l3_avg,
                    'l20_avg': l20_avg, 'season_avg': season_avg,
                    'gap': gap,
                    # Hit rates
                    'l10_hr': l10_hr, 'l5_hr': l5_hr, 'season_hr': season_hr,
                    'l10_miss': l10_miss, 'l5_miss': l5_miss,
                    # Variance
                    'l10_std': l10_std, 'l10_cv': l10_cv,
                    'l10_min': l10_min, 'l10_max': l10_max,
                    'l10_range': l10_range, 'l10_floor': l10_floor,
                    'floor_pct': floor_pct,
                    # Streaks & trends
                    'streak': streak, 'trend': trend,
                    'consec_under': consec_under,
                    # Context
                    'mins_avg': mins_avg, 'mins_std': mins_std, 'mins_stable': mins_stable,
                    'l5_pm': l5_pm, 'l10_pm': l10_pm,
                    'dow': dow, 'month': month,
                    'rest_days': rest_days, 'is_b2b': is_b2b,
                    'foul_prone': foul_prone, 'l10_pf': l10_pf,
                    'usage': l10_usage,
                })

    return props


def run_strategy(props, filter_fn, sort_fn, legs=3, name="unnamed"):
    """Run a parlay strategy across all dates.
    filter_fn: prop -> bool (which props to consider)
    sort_fn: prop -> float (higher = better, for selection)
    legs: number of legs per parlay
    Returns: {wins, total, leg_hits, leg_total, max_streak, streaks, daily_results}
    """
    # Group by date
    by_date = defaultdict(list)
    for p in props:
        if filter_fn(p):
            by_date[p['date']].append(p)

    wins = 0
    total = 0
    leg_hits = 0
    leg_total = 0
    streak = 0
    max_streak = 0
    streaks = []
    daily = []

    for date in sorted(by_date.keys()):
        candidates = by_date[date]
        # Sort by strategy preference
        candidates.sort(key=sort_fn, reverse=True)

        # Pick top N unique players
        selected = []
        used_names = set()
        for p in candidates:
            if p['name'] in used_names: continue
            selected.append(p)
            used_names.add(p['name'])
            if len(selected) >= legs: break

        if len(selected) < legs: continue

        all_hit = all(p['under'] for p in selected)
        n_hit = sum(p['under'] for p in selected)
        total += 1
        leg_total += legs
        leg_hits += n_hit

        if all_hit:
            wins += 1
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            if streak > 0:
                streaks.append(streak)
            streak = 0

        daily.append({'date': date, 'hit': all_hit, 'legs_hit': n_hit})

    if streak > 0:
        streaks.append(streak)

    return {
        'name': name,
        'wins': wins,
        'total': total,
        'rate': wins / max(total, 1),
        'leg_hits': leg_hits,
        'leg_total': leg_total,
        'leg_rate': leg_hits / max(leg_total, 1),
        'max_streak': max_streak,
        'streaks_5plus': sum(1 for s in streaks if s >= 5),
        'streaks_10plus': sum(1 for s in streaks if s >= 10),
        'avg_streak': sum(streaks) / max(len(streaks), 1),
        'daily': daily,
    }


# =========================================================================
# STRATEGY DEFINITIONS: Each returns (filter_fn, sort_fn, legs, name)
# =========================================================================

def gen_strategies():
    strategies = []

    # === DIMENSION 1: CONFIDENCE SCORE VARIANTS ===
    def conf_score(p, away_bonus=0.5, b2b_bonus=0.3):
        s = 0.0
        hr = p['l10_hr']
        if hr < 20: s += 3.0
        elif hr < 35: s += 2.0
        elif hr < 45: s += 1.0
        elif hr >= 80: s -= 2.0
        elif hr >= 65: s -= 1.0
        elif hr >= 55: s -= 0.5

        shr = p['season_hr']
        if shr < 30: s += 2.0
        elif shr < 45: s += 0.5
        elif shr >= 70: s -= 1.0
        elif shr >= 55: s -= 0.5

        stat_bonus = {'blk': 2.0, 'stl': 1.5, '3pm': 1.0, 'ast': 0.5}
        s += stat_bonus.get(p['stat'], 0)

        if p['streak'] == 'COLD': s += 1.0
        elif p['streak'] == 'HOT': s -= 0.5

        gap = p['gap']
        if gap < -5: s += 2.0
        elif gap < -3: s += 1.5
        elif gap < -1.5: s += 1.0
        elif gap < 0: s += 0.5
        elif gap < 3: s -= 0.5
        else: s -= 0.5

        mc = p['l10_miss']
        if mc >= 9: s += 2.0
        elif mc >= 7: s += 1.0
        elif mc >= 5: s += 0.3
        elif mc < 3: s -= 0.5

        if p.get('is_away'): s += away_bonus
        if p.get('is_b2b'): s += b2b_bonus

        return s

    # --- Baseline strategies with different thresholds ---
    for thresh in [4, 5, 6, 7]:
        for legs in [2, 3]:
            name = f"conf>={thresh}_{legs}leg"
            t, l = thresh, legs
            strategies.append((
                lambda p, t=t: conf_score(p) >= t,
                lambda p: conf_score(p),
                l, name
            ))

    # === DIMENSION 2: VARIANCE / CONSISTENCY ===
    # Low-variance players are more predictable
    for cv_max in [0.3, 0.4, 0.5]:
        for thresh in [4, 5]:
            for legs in [2, 3]:
                name = f"low_var_cv{cv_max}_conf{thresh}_{legs}leg"
                strategies.append((
                    lambda p, t=thresh, cv=cv_max: conf_score(p) >= t and p['l10_cv'] <= cv,
                    lambda p: conf_score(p) + (0.5 - p['l10_cv']),  # prefer lower variance
                    legs, name
                ))

    # High floor: player's 2nd-lowest game still under line
    for thresh in [4, 5]:
        for legs in [2, 3]:
            name = f"high_floor_conf{thresh}_{legs}leg"
            strategies.append((
                lambda p, t=thresh: conf_score(p) >= t and p['l10_floor'] < p['line'],
                lambda p: conf_score(p) + (p['line'] - p['l10_floor']) / max(p['line'], 1) * 2,
                legs, name
            ))

    # === DIMENSION 3: STREAK-BASED ===
    # Only cold players
    for thresh in [3, 4, 5]:
        for legs in [2, 3]:
            name = f"cold_only_conf{thresh}_{legs}leg"
            strategies.append((
                lambda p, t=thresh: conf_score(p) >= t and p['streak'] == 'COLD',
                lambda p: conf_score(p),
                legs, name
            ))

    # Consecutive unders (momentum)
    for min_consec in [2, 3, 4, 5]:
        for legs in [2, 3]:
            name = f"consec_under{min_consec}_{legs}leg"
            strategies.append((
                lambda p, mc=min_consec: p['consec_under'] >= mc and conf_score(p) >= 3,
                lambda p: p['consec_under'] * 2 + conf_score(p),
                legs, name
            ))

    # Anti-HOT (avoid regression traps)
    for thresh in [4, 5]:
        for legs in [2, 3]:
            name = f"anti_hot_conf{thresh}_{legs}leg"
            strategies.append((
                lambda p, t=thresh: conf_score(p) >= t and p['streak'] != 'HOT',
                lambda p: conf_score(p),
                legs, name
            ))

    # === DIMENSION 4: STAT-SPECIFIC ===
    for stat_group, stats in [
        ('blk_stl', ['blk', 'stl']),
        ('blk_only', ['blk']),
        ('stl_only', ['stl']),
        ('3pm_only', ['3pm']),
        ('ast_blk_stl', ['ast', 'blk', 'stl']),
        ('no_pts', ['reb', 'ast', '3pm', 'blk', 'stl']),
        ('singles_only', ['pts', 'reb', 'ast', '3pm', 'blk', 'stl']),
    ]:
        for thresh in [3, 4, 5]:
            for legs in [2, 3]:
                name = f"stat_{stat_group}_conf{thresh}_{legs}leg"
                strategies.append((
                    lambda p, t=thresh, ss=stats: conf_score(p) >= t and p['stat'] in ss,
                    lambda p: conf_score(p),
                    legs, name
                ))

    # === DIMENSION 5: HOME/AWAY ===
    for legs in [2, 3]:
        # Away only
        strategies.append((
            lambda p: conf_score(p) >= 5 and p.get('is_away') == True,
            lambda p: conf_score(p),
            legs, f"away_only_conf5_{legs}leg"
        ))
        # Home only
        strategies.append((
            lambda p: conf_score(p) >= 5 and p.get('home') == True,
            lambda p: conf_score(p),
            legs, f"home_only_conf5_{legs}leg"
        ))

    # === DIMENSION 6: REST / B2B ===
    for legs in [2, 3]:
        # B2B games only (fatigue → under)
        strategies.append((
            lambda p: conf_score(p) >= 4 and p['is_b2b'],
            lambda p: conf_score(p),
            legs, f"b2b_only_conf4_{legs}leg"
        ))
        # Well-rested (2+ days rest — performance stabilizes)
        strategies.append((
            lambda p: conf_score(p) >= 5 and p['rest_days'] >= 2,
            lambda p: conf_score(p),
            legs, f"rested_2d_conf5_{legs}leg"
        ))

    # === DIMENSION 7: MINUTES STABILITY ===
    for legs in [2, 3]:
        strategies.append((
            lambda p: conf_score(p) >= 5 and p['mins_stable'] and p['mins_avg'] >= 25,
            lambda p: conf_score(p) + (p['mins_avg'] / 48) * 2,
            legs, f"stable_mins_conf5_{legs}leg"
        ))

    # === DIMENSION 8: HIT RATE EXTREMES ===
    for max_hr in [10, 20, 30]:
        for legs in [2, 3]:
            name = f"l10hr_max{max_hr}_{legs}leg"
            strategies.append((
                lambda p, mh=max_hr: p['l10_hr'] <= mh and conf_score(p) >= 3,
                lambda p: -p['l10_hr'] + conf_score(p),  # lower HR = better for UNDER
                legs, name
            ))

    for min_miss in [7, 8, 9]:
        for legs in [2, 3]:
            name = f"l10miss_min{min_miss}_{legs}leg"
            strategies.append((
                lambda p, mm=min_miss: p['l10_miss'] >= mm,
                lambda p: p['l10_miss'] + conf_score(p) / 5,
                legs, name
            ))

    # === DIMENSION 9: GAP-BASED (line far above average) ===
    for min_gap in [1, 2, 3, 5]:
        for legs in [2, 3]:
            name = f"gap_neg{min_gap}_{legs}leg"
            strategies.append((
                lambda p, mg=min_gap: p['gap'] <= -mg and conf_score(p) >= 3,
                lambda p: -p['gap'] + conf_score(p) / 3,
                legs, name
            ))

    # === DIMENSION 10: TREND-BASED (declining players) ===
    for max_trend in [-10, -15, -20]:
        for legs in [2, 3]:
            name = f"declining_{-max_trend}pct_{legs}leg"
            strategies.append((
                lambda p, mt=max_trend: p['trend'] <= mt and conf_score(p) >= 3,
                lambda p: -p['trend'] + conf_score(p),
                legs, name
            ))

    # === DIMENSION 11: MULTI-FACTOR COMBOS ===
    # The "perfect storm" — multiple UNDER signals aligned
    for legs in [2, 3]:
        # Cold + away + high miss count
        strategies.append((
            lambda p: p['streak'] == 'COLD' and p.get('is_away') and p['l10_miss'] >= 7 and conf_score(p) >= 4,
            lambda p: conf_score(p) + p['l10_miss'],
            legs, f"cold_away_miss7_{legs}leg"
        ))

        # Low variance + high miss + away
        strategies.append((
            lambda p: p['l10_cv'] <= 0.4 and p['l10_miss'] >= 7 and p.get('is_away'),
            lambda p: p['l10_miss'] - p['l10_cv'] * 10 + conf_score(p) / 3,
            legs, f"lowvar_miss7_away_{legs}leg"
        ))

        # Floor below line + cold + big gap
        strategies.append((
            lambda p: p['l10_floor'] < p['line'] and p['streak'] == 'COLD' and p['gap'] <= -2,
            lambda p: (p['line'] - p['l10_floor']) + conf_score(p),
            legs, f"floor_cold_gap2_{legs}leg"
        ))

        # Mega filter: miss >= 8 + gap <= -2 + floor < line
        strategies.append((
            lambda p: p['l10_miss'] >= 8 and p['gap'] <= -2 and p['l10_floor'] < p['line'],
            lambda p: p['l10_miss'] + abs(p['gap']) + (p['line'] - p['l10_floor']),
            legs, f"mega_miss8_gap2_floor_{legs}leg"
        ))

        # Ultra: miss >= 9 + l10_hr <= 10 + gap <= -3
        strategies.append((
            lambda p: p['l10_miss'] >= 9 and p['l10_hr'] <= 10 and p['gap'] <= -3,
            lambda p: p['l10_miss'] + abs(p['gap']),
            legs, f"ultra_miss9_hr10_gap3_{legs}leg"
        ))

        # Stat specialists: BLK/STL + cold + miss >= 6
        strategies.append((
            lambda p: p['stat'] in ['blk', 'stl'] and p['streak'] == 'COLD' and p['l10_miss'] >= 6,
            lambda p: conf_score(p) + p['l10_miss'],
            legs, f"blkstl_cold_miss6_{legs}leg"
        ))

        # High minutes + under momentum
        strategies.append((
            lambda p: p['mins_avg'] >= 30 and p['consec_under'] >= 3 and conf_score(p) >= 4,
            lambda p: p['consec_under'] * 3 + conf_score(p),
            legs, f"highmins_consec3_{legs}leg"
        ))

    # === DIMENSION 12: SELECTIVITY (only bet on days with many options) ===
    # This is handled differently — we'll add a min_candidates filter
    for min_cand in [10, 15, 20, 30]:
        for legs in [2, 3]:
            name = f"selective_{min_cand}cand_conf5_{legs}leg"
            # We'll mark these specially and handle in the runner
            strategies.append((
                lambda p: conf_score(p) >= 5,  # base filter
                lambda p: conf_score(p),
                legs, name
            ))

    # === DIMENSION 13: SEASON PHASE ===
    for months, label in [
        ([10, 11, 12], 'early'),
        ([1, 2], 'mid'),
        ([3, 4], 'late'),
    ]:
        for thresh in [4, 5]:
            for legs in [2, 3]:
                name = f"season_{label}_conf{thresh}_{legs}leg"
                strategies.append((
                    lambda p, t=thresh, ms=months: conf_score(p) >= t and p['month'] in ms,
                    lambda p: conf_score(p),
                    legs, name
                ))

    # === DIMENSION 14: DAY OF WEEK ===
    for dow_group, label in [
        ([0, 1, 2, 3, 4], 'weekday'),
        ([5, 6], 'weekend'),
    ]:
        for legs in [2, 3]:
            name = f"dow_{label}_conf5_{legs}leg"
            strategies.append((
                lambda p, ds=dow_group: conf_score(p) >= 5 and p['dow'] in ds,
                lambda p: conf_score(p),
                legs, name
            ))

    # === DIMENSION 15: PLUS/MINUS CONTEXT ===
    for legs in [2, 3]:
        # Players on losing teams (negative plus/minus) → fatigue/morale
        strategies.append((
            lambda p: conf_score(p) >= 5 and p['l10_pm'] < -3,
            lambda p: conf_score(p) - p['l10_pm'] / 10,
            legs, f"neg_pm_conf5_{legs}leg"
        ))

    # === DIMENSION 16: FOUL TROUBLE ===
    for legs in [2, 3]:
        strategies.append((
            lambda p: conf_score(p) >= 4 and p['foul_prone'],
            lambda p: conf_score(p) + p['l10_pf'],
            legs, f"foul_prone_conf4_{legs}leg"
        ))

    # === DIMENSION 17: PURE STATISTICAL SORTS (no conf_score) ===
    for legs in [2, 3]:
        # Pure miss count sort
        strategies.append((
            lambda p: p['l10_miss'] >= 7,
            lambda p: p['l10_miss'] * 10 - p['l10_hr'],
            legs, f"pure_miss7_{legs}leg"
        ))
        # Pure HR sort (lowest HR)
        strategies.append((
            lambda p: p['l10_hr'] <= 20,
            lambda p: -p['l10_hr'],
            legs, f"pure_hr20_{legs}leg"
        ))
        # Pure gap sort (biggest gap below line)
        strategies.append((
            lambda p: p['gap'] <= -3,
            lambda p: -p['gap'],
            legs, f"pure_gap3_{legs}leg"
        ))
        # Pure consecutive under
        strategies.append((
            lambda p: p['consec_under'] >= 5,
            lambda p: p['consec_under'],
            legs, f"pure_consec5_{legs}leg"
        ))

    # === DIMENSION 18: COMPOSITE SCORING VARIANTS ===
    for legs in [2, 3]:
        # Score: weighted miss + gap + var
        strategies.append((
            lambda p: p['l10_miss'] >= 6 and p['gap'] <= -1,
            lambda p: p['l10_miss'] * 3 + abs(p['gap']) * 2 + (1 - p['l10_cv']) * 5,
            legs, f"composite_miss_gap_var_{legs}leg"
        ))

        # Score: HR inverse + miss + consec
        strategies.append((
            lambda p: p['l10_hr'] <= 40,
            lambda p: (100 - p['l10_hr']) / 10 + p['l10_miss'] * 2 + p['consec_under'] * 3,
            legs, f"composite_hr_miss_consec_{legs}leg"
        ))

        # Score: Bayesian-style (season + L10 blend)
        strategies.append((
            lambda p: p['l10_hr'] <= 30 and p['season_hr'] <= 45,
            lambda p: (100 - p['l10_hr']) * 0.6 + (100 - p['season_hr']) * 0.4,
            legs, f"bayesian_hr_blend_{legs}leg"
        ))

    # === DIMENSION 19: ADAPTIVE THRESHOLD ===
    # Only bet when confidence is MUCH higher than usual
    for top_n in [3, 5, 8]:
        for legs in [2, 3]:
            name = f"top{top_n}_daily_{legs}leg"
            strategies.append((
                lambda p: True,  # no filter — just sort
                lambda p: conf_score(p),
                legs, name
            ))

    # === DIMENSION 20: "NUCLEAR" COMBOS ===
    for legs in [2, 3]:
        # Every possible strong signal aligned
        strategies.append((
            lambda p: (p['l10_miss'] >= 8 and p['l10_hr'] <= 20 and
                       p['gap'] <= -2 and p['l10_floor'] < p['line'] and
                       p['streak'] != 'HOT' and p.get('is_away')),
            lambda p: conf_score(p) + p['l10_miss'] + abs(p['gap']),
            legs, f"nuclear_all_signals_{legs}leg"
        ))

        # Miss >= 9 + BLK/STL + away
        strategies.append((
            lambda p: (p['l10_miss'] >= 9 and p['stat'] in ['blk', 'stl'] and
                       p.get('is_away')),
            lambda p: p['l10_miss'] + conf_score(p),
            legs, f"nuclear_blkstl_miss9_away_{legs}leg"
        ))

    print(f"Generated {len(strategies)} strategies", file=sys.stderr)
    return strategies


def main():
    player_games = load_data()

    print("Computing props (4% inflation)...", file=sys.stderr)
    props = compute_props(player_games, inflation=4)
    print(f"Generated {len(props):,} props", file=sys.stderr)

    strategies = gen_strategies()

    results = []
    print(f"\nTesting {len(strategies)} strategies...", file=sys.stderr)

    for i, (filt, sort, legs, name) in enumerate(strategies):
        try:
            r = run_strategy(props, filt, sort, legs, name)
            if r['total'] >= 50:  # need minimum sample
                results.append(r)
        except Exception as e:
            pass

        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(strategies)} tested...", file=sys.stderr)

    # Sort by parlay win rate
    results.sort(key=lambda r: r['rate'], reverse=True)

    print(f"\n{'='*90}")
    print(f"MEGA BACKTEST RESULTS: {len(results)} strategies with 50+ parlays")
    print(f"{'='*90}")

    # Top 50 by win rate
    print(f"\n{'='*90}")
    print(f"TOP 50 BY PARLAY CASH RATE")
    print(f"{'='*90}")
    print(f"{'#':>3} {'Strategy':<50} {'W/L':>10} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4}")
    for i, r in enumerate(results[:50]):
        print(f"{i+1:>3} {r['name']:<50} {r['wins']}/{r['total']:>5} {r['rate']:>6.1%} {r['leg_rate']:>6.1%} {r['max_streak']:>6} {r['streaks_5plus']:>4} {r['streaks_10plus']:>4}")

    # Top 20 by max streak
    results_by_streak = sorted(results, key=lambda r: r['max_streak'], reverse=True)
    print(f"\n{'='*90}")
    print(f"TOP 20 BY MAX WIN STREAK")
    print(f"{'='*90}")
    print(f"{'#':>3} {'Strategy':<50} {'W/L':>10} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4}")
    for i, r in enumerate(results_by_streak[:20]):
        print(f"{i+1:>3} {r['name']:<50} {r['wins']}/{r['total']:>5} {r['rate']:>6.1%} {r['leg_rate']:>6.1%} {r['max_streak']:>6} {r['streaks_5plus']:>4} {r['streaks_10plus']:>4}")

    # Top 20 by number of 5+ streaks
    results_by_s5 = sorted(results, key=lambda r: (r['streaks_5plus'], r['rate']), reverse=True)
    print(f"\n{'='*90}")
    print(f"TOP 20 BY COUNT OF 5+ WIN STREAKS")
    print(f"{'='*90}")
    print(f"{'#':>3} {'Strategy':<50} {'W/L':>10} {'Rate':>7} {'LegHR':>7} {'MaxStr':>7} {'5+':>4} {'10+':>4}")
    for i, r in enumerate(results_by_s5[:20]):
        print(f"{i+1:>3} {r['name']:<50} {r['wins']}/{r['total']:>5} {r['rate']:>6.1%} {r['leg_rate']:>6.1%} {r['max_streak']:>6} {r['streaks_5plus']:>4} {r['streaks_10plus']:>4}")

    # === DEEP DIVE: Best strategies ===
    if results:
        best = results[0]
        print(f"\n{'='*90}")
        print(f"DEEP DIVE: {best['name']}")
        print(f"{'='*90}")
        # Yearly breakdown
        yearly = defaultdict(lambda: [0, 0])
        for d in best['daily']:
            yearly[d['date'][:4]][1] += 1
            if d['hit']: yearly[d['date'][:4]][0] += 1
        for y in sorted(yearly.keys()):
            w, t = yearly[y]
            print(f"  {y}: {w}/{t} = {w/max(t,1):.1%}")

    # Save results
    output = []
    for r in results[:100]:
        output.append({
            'name': r['name'],
            'wins': r['wins'],
            'total': r['total'],
            'rate': round(r['rate'], 4),
            'leg_rate': round(r['leg_rate'], 4),
            'max_streak': r['max_streak'],
            'streaks_5plus': r['streaks_5plus'],
            'streaks_10plus': r['streaks_10plus'],
        })

    out_path = os.path.join(os.path.dirname(__file__), 'mega_backtest_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved top 100 to {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()

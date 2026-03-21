#!/usr/bin/env python3
"""
1M parlay simulation on ALL available data (442K+ records, 580+ dates).
Uses backfill + SGO + graded data for maximum statistical power.
"""
import json, glob, os, sys, random
import numpy as np
from collections import Counter, defaultdict

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading data...")

# Source 1: backfill (242K records, 120 dates)
all_records = []
try:
    data = json.load(open(os.path.join(PREDICTIONS_DIR, 'cache/backfill_training_data.json')))
    for r in data:
        if r.get('_hit_label') is not None and r.get('direction'):
            r['_hit'] = bool(r['_hit_label'])
            r['_src'] = 'backfill'
            all_records.append(r)
    print(f"  Backfill: {len(data)} → {sum(1 for r in all_records if r['_src']=='backfill')} with labels")
except Exception as e:
    print(f"  Backfill: {e}")

# Source 2: SGO backfill (200K records, 462 dates)
try:
    data = json.load(open(os.path.join(PREDICTIONS_DIR, 'cache/sgo_backfill_training_data.json')))
    n = 0
    for r in data:
        if r.get('_hit_label') is not None and r.get('direction'):
            r['_hit'] = bool(r['_hit_label'])
            r['_src'] = 'sgo'
            all_records.append(r)
            n += 1
    print(f"  SGO: {len(data)} → {n} with labels")
except Exception as e:
    print(f"  SGO: {e}")

# Source 3: graded daily data (3.7K records, 9 dates)
def extract_hit(r):
    hit = r.get('hit')
    if hit is None:
        res = r.get('result', '')
        if isinstance(res, str):
            if res.upper() == 'HIT': return True
            if res.upper() == 'MISS': return False
    return hit if hit is not None else None

for d in sorted(glob.glob(os.path.join(PREDICTIONS_DIR, '2026-03-*'))):
    date = os.path.basename(d)
    for f in sorted(os.listdir(d)):
        if f.startswith('v4_graded') or f == 'v3_graded_full.json':
            fp = os.path.join(d, f)
            try:
                data = json.load(open(fp))
                records = data.get('results', data) if isinstance(data, dict) else data
                if isinstance(records, list):
                    for r in records:
                        h = extract_hit(r)
                        if h is not None and r.get('direction'):
                            r['_hit'] = h
                            r['_date'] = date
                            r['_src'] = 'graded'
                            all_records.append(r)
                    break
            except:
                pass

graded_n = sum(1 for r in all_records if r.get('_src') == 'graded')
print(f"  Graded: {graded_n}")
print(f"  TOTAL: {len(all_records)} records")

# Group by date
days = defaultdict(list)
for r in all_records:
    date = r.get('_date', '')
    if date:
        days[date].append(r)

# Filter to UNDER props per day (need at least 3)
under_by_day = {}
for date, props in days.items():
    unders = [p for p in props if p.get('direction', '').upper() == 'UNDER']
    if len(unders) >= 5:  # need enough for game diversity
        under_by_day[date] = unders

print(f"  Days with 5+ UNDERs: {len(under_by_day)}")
print(f"  Total UNDER props in pool: {sum(len(v) for v in under_by_day.values()):,}")

# ═══ SIMULATE 1M PARLAYS ═══
N_SIMS = 1_000_000
random.seed(42)
date_list = list(under_by_day.keys())

print(f"\nSimulating {N_SIMS:,} parlays...")

winners = []
losers = []

for i in range(N_SIMS):
    date = random.choice(date_list)
    pool = under_by_day[date]

    # Pick 3 random UNDERs from different games (if game info available)
    random.shuffle(pool)
    picks = []
    used_games = set()
    for p in pool:
        g = p.get('game', '')
        if g and g in used_games:
            continue
        picks.append(p)
        if g:
            used_games.add(g)
        if len(picks) >= 3:
            break

    if len(picks) < 3:
        # No game info — just pick 3 random
        picks = random.sample(pool, min(3, len(pool)))

    if len(picks) < 3:
        continue

    hits = sum(1 for p in picks if p['_hit'])
    cashed = (hits == 3)

    line_vals = [p.get('line', 0) or 0 for p in picks]
    avg_vals = [p.get('season_avg', 0) or 0 for p in picks]

    parlay = {
        'cashed': cashed,
        'hits': hits,
        # Hit rate features
        'avg_l10_hr': np.mean([p.get('l10_hit_rate', 0) or 0 for p in picks]),
        'min_l10_hr': min(p.get('l10_hit_rate', 0) or 0 for p in picks),
        'avg_l5_hr': np.mean([p.get('l5_hit_rate', 0) or 0 for p in picks]),
        'min_l5_hr': min(p.get('l5_hit_rate', 0) or 0 for p in picks),
        # Line features
        'avg_line': np.mean(line_vals),
        'max_line': max(line_vals),
        'min_line': min(line_vals),
        # Season avg (star vs role player)
        'avg_season_avg': np.mean(avg_vals),
        'max_season_avg': max(avg_vals),
        # Line above avg (the money feature from graded data)
        'avg_line_above': np.mean([l - a for l, a in zip(line_vals, avg_vals)]),
        'min_line_above': min(l - a for l, a in zip(line_vals, avg_vals)),
        # Gap
        'avg_gap': np.mean([abs(p.get('gap', 0) or p.get('abs_gap', 0) or 0) for p in picks]),
        # Spread
        'avg_spread': np.mean([abs(p.get('spread', 0) or 0) for p in picks]),
        'max_spread': max(abs(p.get('spread', 0) or 0) for p in picks),
        # Consistency
        'avg_l10_std': np.mean([p.get('l10_std', 0) or 0 for p in picks]),
        'max_l10_std': max(p.get('l10_std', 0) or 0 for p in picks),
        # Miss count
        'avg_miss_count': np.mean([p.get('l10_miss_count', 0) or 0 for p in picks]),
        'max_miss_count': max(p.get('l10_miss_count', 0) or 0 for p in picks),
        # Minutes
        'avg_mins_pct': np.mean([p.get('mins_30plus_pct', 0) or 0 for p in picks]),
        # Stat types
        'n_combo': sum(1 for p in picks if p.get('stat', '').lower() in ('pra', 'pr', 'pa', 'ra', 'stl_blk')),
        'stats': tuple(sorted(p.get('stat', '').lower() for p in picks)),
        # Context
        'n_home': sum(1 for p in picks if p.get('is_home')),
        'n_b2b': sum(1 for p in picks if p.get('is_b2b')),
        'n_l5_down': sum(1 for p in picks if (p.get('l5_avg', 0) or 0) > 0 and (p.get('l10_avg', 0) or 0) > 0 and (p.get('l5_avg', 0) or 0) < (p.get('l10_avg', 0) or 0)),
        # Streak
        'n_cold': sum(1 for p in picks if p.get('streak_status') == 'COLD'),
        'n_hot': sum(1 for p in picks if p.get('streak_status') == 'HOT'),
    }

    if cashed:
        winners.append(parlay)
    else:
        losers.append(parlay)

total = len(winners) + len(losers)
print(f"\n{'='*70}")
print(f"  1M PARLAY SIMULATION — ALL DATA ({len(all_records):,} records, {len(under_by_day)} days)")
print(f"{'='*70}")
print(f"  Total parlays: {total:,}")
print(f"  Winners: {len(winners):,} ({len(winners)/total*100:.1f}%)")
print(f"  Losers: {len(losers):,}")

# ═══ PROFILE ═══
print(f"\n{'='*70}")
print(f"  WINNER vs LOSER PROFILES")
print(f"{'='*70}")

features = [
    ('avg_l10_hr', 'Avg L10 HR', '.1f'),
    ('min_l10_hr', 'MIN L10 HR (weakest leg)', '.1f'),
    ('avg_l5_hr', 'Avg L5 HR', '.1f'),
    ('min_l5_hr', 'MIN L5 HR', '.1f'),
    ('avg_line', 'Avg Line', '.1f'),
    ('max_line', 'MAX Line', '.1f'),
    ('avg_season_avg', 'Avg Season Avg', '.1f'),
    ('max_season_avg', 'MAX Season Avg (star risk)', '.1f'),
    ('avg_line_above', 'Avg Line Above Avg', '.2f'),
    ('min_line_above', 'MIN Line Above Avg', '.2f'),
    ('avg_gap', 'Avg Abs Gap', '.2f'),
    ('avg_spread', 'Avg Spread', '.1f'),
    ('avg_l10_std', 'Avg L10 Std Dev', '.2f'),
    ('max_l10_std', 'MAX L10 Std Dev', '.2f'),
    ('avg_miss_count', 'Avg Miss Count', '.2f'),
    ('max_miss_count', 'MAX Miss Count', '.2f'),
    ('avg_mins_pct', 'Avg Mins 30+ Pct', '.1f'),
    ('n_combo', 'Combo stat count', '.2f'),
    ('n_home', 'Home count', '.2f'),
    ('n_b2b', 'B2B count', '.2f'),
    ('n_l5_down', 'L5<L10 count', '.2f'),
    ('n_cold', 'COLD streak count', '.2f'),
    ('n_hot', 'HOT streak count', '.2f'),
]

print(f"\n  {'Feature':<35s} {'Winners':>10s} {'Losers':>10s} {'Delta':>10s} {'Edge':>6s}")
print(f"  {'-'*75}")
for feat, label, fmt in features:
    w_val = np.mean([w[feat] for w in winners])
    l_val = np.mean([l[feat] for l in losers])
    delta = w_val - l_val
    d_sign = '+' if delta > 0 else ''
    # Edge direction
    if abs(delta) < 0.01:
        edge = '  ~'
    elif delta > 0:
        edge = '  WIN↑'
    else:
        edge = '  WIN↓'
    print(f"  {label:<35s} {w_val:>10{fmt}} {l_val:>10{fmt}} {d_sign}{delta:>9{fmt}} {edge}")

# ═══ OPTIMAL THRESHOLDS ═══
print(f"\n{'='*70}")
print(f"  OPTIMAL THRESHOLDS")
print(f"{'='*70}")

all_parlays = winners + losers

for feat, label, thresholds, higher_better in [
    ('min_l10_hr', 'MIN L10 HR >=', [30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90], True),
    ('avg_l10_hr', 'AVG L10 HR >=', [40, 50, 55, 60, 65, 70, 75, 80], True),
    ('max_line', 'MAX Line <=', [3, 5, 8, 10, 15, 20, 25, 30], False),
    ('max_season_avg', 'MAX Season Avg <=', [5, 8, 10, 15, 20, 25, 30], False),
    ('avg_line_above', 'Line Above Avg >=', [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], True),
    ('min_line_above', 'MIN Line Above >=', [-2, -1, 0, 0.5, 1.0, 1.5, 2.0], True),
    ('max_l10_std', 'MAX Std Dev <=', [2, 3, 4, 5, 6, 8, 10], False),
    ('avg_miss_count', 'AVG Miss <=', [2, 3, 4, 5, 6, 7], False),
    ('n_combo', 'Combo Count <=', [0, 1, 2, 3], False),
    ('avg_mins_pct', 'Mins Pct <=', [20, 30, 40, 50, 60, 70, 80], False),
    ('n_home', 'Home Count <=', [0, 1, 2, 3], False),
    ('n_cold', 'COLD Count >=', [0, 1, 2, 3], True),
    ('n_hot', 'HOT Count <=', [0, 1, 2, 3], False),
]:
    print(f"\n  {label}")
    for t in thresholds:
        if higher_better:
            subset = [p for p in all_parlays if p[feat] >= t]
        else:
            subset = [p for p in all_parlays if p[feat] <= t]
        if len(subset) >= 500:
            cash = sum(1 for p in subset if p['cashed'])
            rate = cash / len(subset) * 100
            bar = '█' * int(rate)
            print(f"    {t:6.1f}: {cash:7,}/{len(subset):8,} = {rate:5.1f}%  {bar}")
        elif len(subset) >= 50:
            cash = sum(1 for p in subset if p['cashed'])
            rate = cash / len(subset) * 100
            print(f"    {t:6.1f}: {cash:7,}/{len(subset):8,} = {rate:5.1f}%  (small sample)")

# ═══ STAT TYPE COMBOS ═══
print(f"\n{'='*70}")
print(f"  BEST/WORST STAT COMBINATIONS (min 500 samples)")
print(f"{'='*70}")

stat_combos = Counter()
stat_combos_win = Counter()
for p in winners:
    stat_combos_win[p['stats']] += 1
    stat_combos[p['stats']] += 1
for p in losers:
    stat_combos[p['stats']] += 1

combo_rates = []
for combo, total_c in stat_combos.items():
    if total_c >= 500:
        wins_c = stat_combos_win.get(combo, 0)
        rate = wins_c / total_c * 100
        combo_rates.append((rate, combo, wins_c, total_c))

combo_rates.sort(reverse=True)
print(f"\n  TOP 15:")
print(f"  {'Stat Combo':<30s} {'Win':>7s} {'Total':>8s} {'Cash%':>7s}")
print(f"  {'-'*55}")
for rate, combo, w, t in combo_rates[:15]:
    combo_str = ' + '.join(s.upper() for s in combo)
    print(f"  {combo_str:<30s} {w:7,} {t:8,} {rate:6.1f}%")

print(f"\n  BOTTOM 10:")
for rate, combo, w, t in combo_rates[-10:]:
    combo_str = ' + '.join(s.upper() for s in combo)
    print(f"  {combo_str:<30s} {w:7,} {t:8,} {rate:6.1f}%")

# ═══ COMPOUND FILTERS ═══
print(f"\n{'='*70}")
print(f"  COMPOUND FILTERS (min 500 samples)")
print(f"{'='*70}")

filters = [
    ('BASELINE: random UNDER', lambda p: True),
    ('min_hr>=60', lambda p: p['min_l10_hr'] >= 60),
    ('min_hr>=70', lambda p: p['min_l10_hr'] >= 70),
    ('min_hr>=80', lambda p: p['min_l10_hr'] >= 80),
    ('min_hr>=60 + max_avg<=15', lambda p: p['min_l10_hr'] >= 60 and p['max_season_avg'] <= 15),
    ('min_hr>=60 + max_avg<=10', lambda p: p['min_l10_hr'] >= 60 and p['max_season_avg'] <= 10),
    ('min_hr>=70 + max_avg<=15', lambda p: p['min_l10_hr'] >= 70 and p['max_season_avg'] <= 15),
    ('min_hr>=70 + max_avg<=10', lambda p: p['min_l10_hr'] >= 70 and p['max_season_avg'] <= 10),
    ('min_hr>=60 + line_above>=1', lambda p: p['min_l10_hr'] >= 60 and p['avg_line_above'] >= 1),
    ('min_hr>=60 + line_above>=2', lambda p: p['min_l10_hr'] >= 60 and p['avg_line_above'] >= 2),
    ('min_hr>=70 + line_above>=1', lambda p: p['min_l10_hr'] >= 70 and p['avg_line_above'] >= 1),
    ('min_hr>=60 + max_line<=10', lambda p: p['min_l10_hr'] >= 60 and p['max_line'] <= 10),
    ('min_hr>=60 + max_line<=15', lambda p: p['min_l10_hr'] >= 60 and p['max_line'] <= 15),
    ('min_hr>=70 + max_line<=15', lambda p: p['min_l10_hr'] >= 70 and p['max_line'] <= 15),
    ('min_hr>=60 + max_std<=4', lambda p: p['min_l10_hr'] >= 60 and p['max_l10_std'] <= 4),
    ('min_hr>=60 + max_std<=5', lambda p: p['min_l10_hr'] >= 60 and p['max_l10_std'] <= 5),
    ('min_hr>=60 + miss<=4', lambda p: p['min_l10_hr'] >= 60 and p['avg_miss_count'] <= 4),
    ('min_hr>=60 + combo<=1', lambda p: p['min_l10_hr'] >= 60 and p['n_combo'] <= 1),
    ('min_hr>=60 + away_only', lambda p: p['min_l10_hr'] >= 60 and p['n_home'] == 0),
    ('min_hr>=60 + cold>=1', lambda p: p['min_l10_hr'] >= 60 and p['n_cold'] >= 1),
    ('max_avg<=5', lambda p: p['max_season_avg'] <= 5),
    ('max_avg<=5 + line_above>=1', lambda p: p['max_season_avg'] <= 5 and p['avg_line_above'] >= 1),
    ('max_avg<=5 + line_above>=2', lambda p: p['max_season_avg'] <= 5 and p['avg_line_above'] >= 2),
    ('line_above>=2 + max_std<=5', lambda p: p['avg_line_above'] >= 2 and p['max_l10_std'] <= 5),
    ('line_above>=2 + max_avg<=15', lambda p: p['avg_line_above'] >= 2 and p['max_season_avg'] <= 15),
    ('min_hr>=60 + line_above>=1 + max_avg<=15', lambda p: p['min_l10_hr'] >= 60 and p['avg_line_above'] >= 1 and p['max_season_avg'] <= 15),
    ('min_hr>=60 + line_above>=1 + max_std<=5', lambda p: p['min_l10_hr'] >= 60 and p['avg_line_above'] >= 1 and p['max_l10_std'] <= 5),
    ('min_hr>=60 + line_above>=1 + max_avg<=15 + max_std<=5', lambda p: p['min_l10_hr'] >= 60 and p['avg_line_above'] >= 1 and p['max_season_avg'] <= 15 and p['max_l10_std'] <= 5),
    ('min_hr>=70 + line_above>=1 + max_avg<=15', lambda p: p['min_l10_hr'] >= 70 and p['avg_line_above'] >= 1 and p['max_season_avg'] <= 15),
    ('min_hr>=70 + line_above>=1 + max_std<=5', lambda p: p['min_l10_hr'] >= 70 and p['avg_line_above'] >= 1 and p['max_l10_std'] <= 5),
    ('GOLDEN: min_hr>=60 + max_avg<=10 + line_above>=1 + max_std<=5', lambda p: p['min_l10_hr'] >= 60 and p['max_season_avg'] <= 10 and p['avg_line_above'] >= 1 and p['max_l10_std'] <= 5),
]

results = []
for label, filt in filters:
    subset = [p for p in all_parlays if filt(p)]
    if len(subset) >= 100:
        cash = sum(1 for p in subset if p['cashed'])
        rate = cash / len(subset) * 100
        results.append((rate, label, cash, len(subset)))

results.sort(reverse=True)
print(f"\n  {'Filter':<60s} {'Win':>7s} {'Total':>8s} {'Cash%':>7s}")
print(f"  {'-'*85}")
for rate, label, w, t in results:
    bar = '█' * int(rate / 2)
    marker = ' ← BEST' if rate == results[0][0] else ''
    print(f"  {label:<60s} {w:7,} {t:8,} {rate:6.1f}%  {bar}{marker}")

print(f"\n  Done.")

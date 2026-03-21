#!/usr/bin/env python3
"""
Simulate 1M 3-leg UNDER parlays from graded data.
Profile winners vs losers to find what actually matters.
"""
import json, glob, os, sys, random
import numpy as np
from collections import Counter, defaultdict

PREDICTIONS_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_hit(r):
    hit = r.get('hit')
    if hit is None:
        res = r.get('result', '')
        if isinstance(res, str):
            if res.upper() == 'HIT': return True
            if res.upper() == 'MISS': return False
    return hit if hit is not None else None

# Load graded data
all_props = []
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
                        if h is not None:
                            r['_hit'] = h
                            r['_date'] = date
                            all_props.append(r)
                    break
            except:
                pass

# Split by day
days = defaultdict(list)
for p in all_props:
    days[p['_date']].append(p)

# Get UNDER props per day
under_by_day = {}
for date, props in days.items():
    unders = [p for p in props if p.get('direction', '').upper() == 'UNDER']
    if len(unders) >= 3:
        under_by_day[date] = unders

print(f"Loaded {len(all_props)} graded props across {len(days)} days")
print(f"Days with 3+ UNDERs: {len(under_by_day)}")
print(f"Total UNDER props: {sum(len(v) for v in under_by_day.values())}")

# ═══ SIMULATE 1M PARLAYS ═══
N_SIMS = 1_000_000
random.seed(42)

date_list = list(under_by_day.keys())
winners = []
losers = []
win_legs = []
lose_legs = []

for i in range(N_SIMS):
    date = random.choice(date_list)
    pool = under_by_day[date]
    
    # Pick 3 random UNDERs from different games
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
        continue
    
    hits = sum(1 for p in picks if p['_hit'])
    cashed = (hits == 3)
    
    # Extract features for analysis
    parlay_features = {
        'cashed': cashed,
        'hits': hits,
        'date': date,
        'avg_l10_hr': np.mean([p.get('l10_hit_rate', 0) or 0 for p in picks]),
        'min_l10_hr': min(p.get('l10_hit_rate', 0) or 0 for p in picks),
        'avg_l5_hr': np.mean([p.get('l5_hit_rate', 0) or 0 for p in picks]),
        'min_l5_hr': min(p.get('l5_hit_rate', 0) or 0 for p in picks),
        'avg_line': np.mean([p.get('line', 0) or 0 for p in picks]),
        'max_line': max(p.get('line', 0) or 0 for p in picks),
        'min_line': min(p.get('line', 0) or 0 for p in picks),
        'avg_gap': np.mean([p.get('abs_gap', 0) or 0 for p in picks]),
        'avg_spread': np.mean([abs(p.get('spread', 0) or 0) for p in picks]),
        'max_spread': max(abs(p.get('spread', 0) or 0) for p in picks),
        'avg_season_avg': np.mean([p.get('season_avg', 0) or 0 for p in picks]),
        'max_season_avg': max(p.get('season_avg', 0) or 0 for p in picks),
        'n_combo': sum(1 for p in picks if p.get('stat', '').lower() in ('pra','pr','pa','ra','stl_blk')),
        'n_base': sum(1 for p in picks if p.get('stat', '').lower() in ('pts','reb','ast','3pm','stl','blk')),
        'avg_mins_pct': np.mean([p.get('mins_30plus_pct', 0) or 0 for p in picks]),
        'avg_miss_count': np.mean([p.get('l10_miss_count', 0) or 0 for p in picks]),
        'min_miss_count': min(p.get('l10_miss_count', 0) or 0 for p in picks),
        'avg_xgb': np.mean([p.get('xgb_prob', 0.5) or 0.5 for p in picks]),
        'min_xgb': min(p.get('xgb_prob', 0.5) or 0.5 for p in picks),
        'avg_ensemble': np.mean([p.get('ensemble_prob', p.get('xgb_prob', 0.5)) or 0.5 for p in picks]),
        'n_b2b': sum(1 for p in picks if p.get('is_b2b')),
        'n_home': sum(1 for p in picks if p.get('is_home')),
        'avg_l10_std': np.mean([p.get('l10_std', 0) or 0 for p in picks]),
        'max_l10_std': max(p.get('l10_std', 0) or 0 for p in picks),
        # Stat types
        'stats': tuple(sorted(p.get('stat', '').lower() for p in picks)),
        # L5 vs L10 trend
        'n_l5_down': sum(1 for p in picks if (p.get('l5_avg',0) or 0) > 0 and (p.get('l10_avg',0) or 0) > 0 and (p.get('l5_avg',0) or 0) < (p.get('l10_avg',0) or 0)),
        'all_l5_down': all((p.get('l5_avg',0) or 0) > 0 and (p.get('l10_avg',0) or 0) > 0 and (p.get('l5_avg',0) or 0) < (p.get('l10_avg',0) or 0) for p in picks),
        # Floor analysis
        'avg_floor': np.mean([p.get('l10_floor', 0) or 0 for p in picks]),
        'line_above_avg': np.mean([(p.get('line', 0) or 0) - (p.get('season_avg', 0) or 0) for p in picks]),
        'avg_l10_cv': np.mean([p.get('l10_cv', 0) or 0 for p in picks]) if any(p.get('l10_cv') for p in picks) else 0,
    }
    
    if cashed:
        winners.append(parlay_features)
        win_legs.extend(picks)
    else:
        losers.append(parlay_features)
        lose_legs.extend(picks)

total = len(winners) + len(losers)
print(f"\n{'='*70}")
print(f"  1M PARLAY SIMULATION RESULTS")
print(f"{'='*70}")
print(f"  Total parlays: {total:,}")
print(f"  Winners: {len(winners):,} ({len(winners)/total*100:.1f}%)")
print(f"  Losers: {len(losers):,}")

# ═══ PROFILE WINNERS vs LOSERS ═══
print(f"\n{'='*70}")
print(f"  WINNER vs LOSER PROFILES")
print(f"{'='*70}")

features_to_compare = [
    ('avg_l10_hr', 'Avg L10 HR', '.1f'),
    ('min_l10_hr', 'MIN L10 HR (weakest leg)', '.1f'),
    ('avg_l5_hr', 'Avg L5 HR', '.1f'),
    ('min_l5_hr', 'MIN L5 HR', '.1f'),
    ('avg_line', 'Avg Line', '.1f'),
    ('max_line', 'MAX Line', '.1f'),
    ('min_line', 'Min Line', '.1f'),
    ('avg_gap', 'Avg Abs Gap', '.2f'),
    ('avg_spread', 'Avg Spread', '.1f'),
    ('max_spread', 'Max Spread', '.1f'),
    ('avg_season_avg', 'Avg Season Avg', '.1f'),
    ('max_season_avg', 'MAX Season Avg (star risk)', '.1f'),
    ('n_combo', 'Combo stats in parlay', '.2f'),
    ('avg_mins_pct', 'Avg Mins 30+ %', '.1f'),
    ('avg_miss_count', 'Avg L10 Miss Count', '.2f'),
    ('min_miss_count', 'MIN Miss Count', '.2f'),
    ('avg_xgb', 'Avg XGBoost Prob', '.3f'),
    ('min_xgb', 'MIN XGBoost Prob', '.3f'),
    ('avg_ensemble', 'Avg Ensemble Prob', '.3f'),
    ('n_b2b', 'B2B count', '.2f'),
    ('n_home', 'Home count', '.2f'),
    ('avg_l10_std', 'Avg L10 Std Dev', '.2f'),
    ('max_l10_std', 'MAX L10 Std Dev', '.2f'),
    ('n_l5_down', 'L5<L10 count', '.2f'),
    ('line_above_avg', 'Avg Line - Season Avg', '.2f'),
]

print(f"\n  {'Feature':<35s} {'Winners':>10s} {'Losers':>10s} {'Delta':>10s}")
print(f"  {'-'*70}")
for feat, label, fmt in features_to_compare:
    w_val = np.mean([w[feat] for w in winners])
    l_val = np.mean([l[feat] for l in losers])
    delta = w_val - l_val
    d_sign = '+' if delta > 0 else ''
    print(f"  {label:<35s} {w_val:>10{fmt}} {l_val:>10{fmt}} {d_sign}{delta:>9{fmt}}")

# ═══ FIND THRESHOLD THAT MAXIMIZES CASH RATE ═══
print(f"\n{'='*70}")
print(f"  OPTIMAL THRESHOLDS (cash rate at each cutoff)")
print(f"{'='*70}")

for feat, label, thresholds in [
    ('min_l10_hr', 'MIN L10 HR >=', [40, 50, 55, 60, 65, 70, 75, 80]),
    ('avg_l10_hr', 'AVG L10 HR >=', [50, 55, 60, 65, 70, 75, 80]),
    ('max_line', 'MAX Line <=', [5, 8, 10, 15, 20, 25, 30, 40]),
    ('max_season_avg', 'MAX Season Avg <=', [5, 10, 15, 20, 25, 30]),
    ('avg_line', 'AVG Line <=', [3, 5, 8, 10, 15, 20]),
    ('n_combo', 'Combo Count <=', [0, 1, 2]),
    ('max_l10_std', 'MAX L10 Std <=', [2, 3, 4, 5, 6, 8]),
    ('avg_gap', 'AVG Gap >=', [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    ('max_spread', 'MAX Spread >=', [0, 5, 8, 10, 12, 15]),
    ('avg_miss_count', 'AVG Miss Count >=', [3, 4, 5, 6, 7, 8]),
    ('line_above_avg', 'Line-Avg >=', [0, 0.5, 1.0, 1.5, 2.0, 3.0]),
]:
    print(f"\n  {label}")
    all_parlays = winners + losers
    for t in thresholds:
        if '<=' in label:
            subset = [p for p in all_parlays if p[feat] <= t]
        else:
            subset = [p for p in all_parlays if p[feat] >= t]
        if len(subset) >= 100:
            cash = sum(1 for p in subset if p['cashed'])
            rate = cash / len(subset) * 100
            bar = '█' * int(rate)
            print(f"    {t:6.1f}: {cash:6,}/{len(subset):7,} = {rate:5.1f}%  {bar}")

# ═══ BEST STAT COMBINATIONS ═══
print(f"\n{'='*70}")
print(f"  BEST STAT TYPE COMBINATIONS")
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
    if total_c >= 200:  # enough samples
        wins_c = stat_combos_win.get(combo, 0)
        rate = wins_c / total_c * 100
        combo_rates.append((rate, combo, wins_c, total_c))

combo_rates.sort(reverse=True)
print(f"\n  {'Stat Combo':<30s} {'Win':>6s} {'Total':>7s} {'Cash%':>7s}")
print(f"  {'-'*55}")
for rate, combo, w, t in combo_rates[:20]:
    combo_str = ' + '.join(s.upper() for s in combo)
    print(f"  {combo_str:<30s} {w:6,} {t:7,} {rate:6.1f}%")

print(f"\n  WORST:")
for rate, combo, w, t in combo_rates[-10:]:
    combo_str = ' + '.join(s.upper() for s in combo)
    print(f"  {combo_str:<30s} {w:6,} {t:7,} {rate:6.1f}%")

# ═══ COMPOUND FILTERS (best combinations) ═══
print(f"\n{'='*70}")
print(f"  COMPOUND FILTERS — BEST COMBINATIONS")
print(f"{'='*70}")

all_parlays = winners + losers
filters = [
    ('min_l10_hr>=60 + max_line<=15', lambda p: p['min_l10_hr'] >= 60 and p['max_line'] <= 15),
    ('min_l10_hr>=60 + combo==0', lambda p: p['min_l10_hr'] >= 60 and p['n_combo'] == 0),
    ('min_l10_hr>=70 + combo==0', lambda p: p['min_l10_hr'] >= 70 and p['n_combo'] == 0),
    ('min_l10_hr>=60 + max_line<=10', lambda p: p['min_l10_hr'] >= 60 and p['max_line'] <= 10),
    ('min_l10_hr>=70 + max_line<=10', lambda p: p['min_l10_hr'] >= 70 and p['max_line'] <= 10),
    ('min_l10_hr>=60 + max_std<=4', lambda p: p['min_l10_hr'] >= 60 and p['max_l10_std'] <= 4),
    ('min_l10_hr>=70 + max_std<=4', lambda p: p['min_l10_hr'] >= 70 and p['max_l10_std'] <= 4),
    ('min_l10_hr>=60 + max_line<=10 + combo==0', lambda p: p['min_l10_hr'] >= 60 and p['max_line'] <= 10 and p['n_combo'] == 0),
    ('min_l10_hr>=70 + max_line<=10 + combo==0', lambda p: p['min_l10_hr'] >= 70 and p['max_line'] <= 10 and p['n_combo'] == 0),
    ('min_l10_hr>=60 + max_avg<=15 + combo==0', lambda p: p['min_l10_hr'] >= 60 and p['max_season_avg'] <= 15 and p['n_combo'] == 0),
    ('min_l10_hr>=70 + max_avg<=20', lambda p: p['min_l10_hr'] >= 70 and p['max_season_avg'] <= 20),
    ('min_l10_hr>=60 + gap>=1.5', lambda p: p['min_l10_hr'] >= 60 and p['avg_gap'] >= 1.5),
    ('min_l10_hr>=70 + gap>=1.5', lambda p: p['min_l10_hr'] >= 70 and p['avg_gap'] >= 1.5),
    ('min_l10_hr>=60 + max_line<=10 + max_std<=5', lambda p: p['min_l10_hr'] >= 60 and p['max_line'] <= 10 and p['max_l10_std'] <= 5),
    ('min_l10_hr>=70 + max_line<=15 + combo==0', lambda p: p['min_l10_hr'] >= 70 and p['max_line'] <= 15 and p['n_combo'] == 0),
    ('min_l10_hr>=60 + line_above>=1', lambda p: p['min_l10_hr'] >= 60 and p['line_above_avg'] >= 1),
    ('min_l10_hr>=70 + line_above>=1', lambda p: p['min_l10_hr'] >= 70 and p['line_above_avg'] >= 1),
    ('avg_l10_hr>=75 + combo==0', lambda p: p['avg_l10_hr'] >= 75 and p['n_combo'] == 0),
    ('min_l10_hr>=60 + max_line<=8 + combo==0', lambda p: p['min_l10_hr'] >= 60 and p['max_line'] <= 8 and p['n_combo'] == 0),
    ('base_only + max_line<=10 + min_hr>=60', lambda p: p['n_combo'] == 0 and p['n_base'] == 3 and p['max_line'] <= 10 and p['min_l10_hr'] >= 60),
]

results = []
for label, filt in filters:
    subset = [p for p in all_parlays if filt(p)]
    if len(subset) >= 50:
        cash = sum(1 for p in subset if p['cashed'])
        rate = cash / len(subset) * 100
        results.append((rate, label, cash, len(subset)))

results.sort(reverse=True)
print(f"\n  {'Filter':<50s} {'Win':>6s} {'Total':>7s} {'Cash%':>7s}")
print(f"  {'-'*75}")
for rate, label, w, t in results:
    bar = '█' * int(rate / 2)
    print(f"  {label:<50s} {w:6,} {t:7,} {rate:6.1f}%  {bar}")

print(f"\n  Done.")

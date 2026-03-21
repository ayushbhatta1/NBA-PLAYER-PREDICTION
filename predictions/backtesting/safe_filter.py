#!/usr/bin/env python3
"""
SAFE Parlay Filter — Backtested optimal filter for parlay construction.

Based on mass backtest across 200K SGO records (462 dates, 684 players):
  - Individual HR: 93.6%
  - 2-leg parlay WR: 91.0%
  - 3-leg parlay WR: 88.0%

Core formula: UNDER + gap >= 5 + L10 HR [60,70) + line >= 15

Usage:
    from backtesting.safe_filter import is_safe_pick, filter_safe_picks

    # Check single pick
    if is_safe_pick(pick):
        print("SAFE pick")

    # Filter a list of picks
    safe_picks = filter_safe_picks(all_picks)
"""

# Minimum thresholds (backtested optimal)
MIN_GAP = 5.0
MIN_L10_HR = 60
MAX_L10_HR = 70  # L10 HR 60-69% is the sweet spot
MIN_LINE = 15.0
REQUIRED_DIRECTION = 'UNDER'

# Relaxed version for higher volume
RELAXED_MIN_GAP = 4.0
RELAXED_MIN_LINE = 15.0

# Ultra-safe version
ULTRA_MIN_GAP = 5.0
ULTRA_MIN_LINE = 20.0


def is_safe_pick(pick, mode='balanced'):
    """
    Check if a pick passes the SAFE filter.

    Args:
        pick: dict with keys: direction, gap (or abs_gap), l10_hit_rate, line
        mode: 'balanced' (default), 'relaxed' (more volume), 'ultra' (highest accuracy)

    Returns:
        bool: True if pick passes the SAFE filter
    """
    direction = (pick.get('direction', '') or '').upper()
    if direction != REQUIRED_DIRECTION:
        return False

    gap = abs(float(pick.get('gap', 0) or pick.get('abs_gap', 0) or 0))
    l10_hr = float(pick.get('l10_hit_rate', 0) or 0)
    line = float(pick.get('line', 0) or 0)

    if mode == 'relaxed':
        min_gap = RELAXED_MIN_GAP
        min_line = RELAXED_MIN_LINE
    elif mode == 'ultra':
        min_gap = ULTRA_MIN_GAP
        min_line = ULTRA_MIN_LINE
    else:
        min_gap = MIN_GAP
        min_line = MIN_LINE

    if gap < min_gap:
        return False
    if not (MIN_L10_HR <= l10_hr < MAX_L10_HR):
        return False
    if line < min_line:
        return False

    return True


def filter_safe_picks(picks, mode='balanced'):
    """
    Filter a list of picks to only SAFE ones.

    Returns list sorted by gap (highest first = strongest edge).
    """
    safe = [p for p in picks if is_safe_pick(p, mode)]
    safe.sort(key=lambda p: abs(float(p.get('gap', 0) or 0)), reverse=True)
    return safe


def get_safe_summary():
    """Return backtested performance summary."""
    return {
        'filter': f'UNDER + gap>={MIN_GAP} + L10HR [{MIN_L10_HR},{MAX_L10_HR}) + line>={MIN_LINE}',
        'backtested_on': '200K SGO records, 462 dates',
        'individual_hr': '93.6%',
        '2leg_wr': '91.0%',
        '3leg_wr': '88.0%',
        'avg_picks_per_day': 2.9,
    }


if __name__ == '__main__':
    import json
    print(json.dumps(get_safe_summary(), indent=2))

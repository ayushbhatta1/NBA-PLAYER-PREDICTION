#!/usr/bin/env python3
"""
Line Movement & Book Disagreement Analyzer

Analyzes line disagreements across sportsbooks to find mathematical edges.
When 5+ books set different lines for the same player prop, someone is wrong.

Core idea: if FanDuel says PTS OVER 25.5 and DraftKings says OVER 23.5,
the consensus is ~24.5. Betting OVER 23.5 on DK has a mathematical edge.
Conversely, when most books cluster at 25.5 but one outlier says 23.5,
the outlier is probably wrong -- bet against it.

No API calls. Pure analysis on already-cached SGO props files.

Usage:
    python3 line_movement.py --date 2026-03-19
    python3 line_movement.py --date 2026-03-18 --edges-only
    python3 line_movement.py --date 2026-03-18 --min-books 5
"""

import json
import os
import sys
import unicodedata
from collections import defaultdict
from statistics import median, stdev

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache', 'sgo')

# Major sportsbooks -- these set sharp lines and move markets.
# Outliers from fringe books (prophetexchange, unknown, bookmakereu) are noise.
SHARP_BOOKS = {
    'fanduel', 'draftkings', 'betmgm', 'caesars', 'espnbet',
    'pinnacle', 'bet365', 'bovada',
}

# Books that frequently post stale or erroneous lines.
# Excluded from consensus calculation but flagged when they disagree.
NOISE_BOOKS = {'unknown', 'prophetexchange', 'bookmakereu'}

# Minimum books required for a reliable consensus
MIN_BOOKS_FOR_CONSENSUS = 3

# Edge thresholds
EDGE_THRESHOLD = 1.0       # Points -- our line vs consensus to flag an edge
HIGH_DISAGREEMENT = 2.0    # Points -- spread across books to flag volatility
VOLATILE_THRESHOLD = 3.0   # Points -- extreme disagreement, avoid this prop


# ═══════════════════════════════════════════════════════════════
# NAME NORMALIZATION
# ═══════════════════════════════════════════════════════════════

def _norm(s):
    """Normalize player name for matching: lowercase, strip accents."""
    nfkd = unicodedata.normalize('NFKD', str(s))
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _name_key(player_name):
    """Create a fuzzy-matchable key from a player name.
    'Jayson Tatum' -> 'tatum_j'  (last name + first initial)
    """
    parts = _norm(player_name).split()
    if len(parts) >= 2:
        return f"{parts[-1]}_{parts[0][0]}"
    return _norm(player_name)


# ═══════════════════════════════════════════════════════════════
# 1. LOAD SGO PROPS
# ═══════════════════════════════════════════════════════════════

def load_sgo_props(date_str):
    """Load cached SGO props for a date. Return structured dict keyed by (player_norm, stat).

    Each value is a dict:
        {
            'player': original name,
            'stat': stat type,
            'sgo_line': SGO fair/consensus line,
            'book_lines': {book: line, ...},  -- all books
            'sharp_lines': {book: line, ...},  -- sharp books only
            'game': game string,
        }
    """
    path = os.path.join(_CACHE_DIR, f'props_{date_str}.json')
    if not os.path.exists(path):
        return {}

    with open(path) as f:
        data = json.load(f)

    raw_props = data.get('props', data) if isinstance(data, dict) else data

    # Group by (player, stat) -- some props appear with different SGO lines
    grouped = defaultdict(list)
    for p in raw_props:
        player = p.get('player', '')
        stat = p.get('stat', '')
        if not player or not stat:
            continue
        key = (_norm(player), stat)
        grouped[key].append(p)

    result = {}
    for key, entries in grouped.items():
        # Take the entry with the most book_lines (richest data)
        best = max(entries, key=lambda e: len(e.get('book_lines', {})))

        all_books = best.get('book_lines', {})
        sharp = {b: v for b, v in all_books.items() if b in SHARP_BOOKS}

        result[key] = {
            'player': best.get('player', ''),
            'stat': best.get('stat', ''),
            'sgo_line': best.get('line', 0),
            'book_lines': all_books,
            'sharp_lines': sharp,
            'game': best.get('game', ''),
            'fair_odds': best.get('fair_odds', ''),
        }

    return result


# ═══════════════════════════════════════════════════════════════
# 2. COMPUTE LINE SPREAD (disagreement analysis)
# ═══════════════════════════════════════════════════════════════

def compute_line_spread(sgo_props):
    """For each player-stat combo, compute cross-book disagreement metrics.

    Returns dict keyed by (player_norm, stat) with:
        consensus_line:     median line across sharp books (or all books)
        line_spread:        max - min across sharp books
        line_spread_all:    max - min across ALL books (including noise)
        n_books:            total books offering this prop
        n_sharp:            sharp books offering this prop
        stdev:              standard deviation of lines (0 = perfect agreement)
        outlier_book:       which book deviates most from consensus
        outlier_deviation:  how far the outlier is from consensus
        book_cluster:       the line most books agree on (mode)
        cluster_pct:        what percentage of books agree on cluster line
    """
    results = {}

    for key, prop in sgo_props.items():
        all_books = prop['book_lines']
        sharp = prop['sharp_lines']

        # Use sharp books for consensus if we have enough, else fall back to all
        consensus_books = sharp if len(sharp) >= MIN_BOOKS_FOR_CONSENSUS else {
            b: v for b, v in all_books.items() if b not in NOISE_BOOKS
        }
        if len(consensus_books) < 2:
            consensus_books = all_books

        if not consensus_books:
            continue

        values = list(consensus_books.values())
        all_values = list(all_books.values())

        consensus_line = median(values)
        line_spread = max(values) - min(values)
        line_spread_all = max(all_values) - min(all_values) if all_values else 0
        sd = stdev(values) if len(values) >= 2 else 0.0

        # Find outlier -- the book furthest from consensus
        max_dev = 0
        outlier_book = None
        outlier_deviation = 0
        for book, val in all_books.items():
            dev = abs(val - consensus_line)
            if dev > max_dev:
                max_dev = dev
                outlier_book = book
                outlier_deviation = val - consensus_line  # positive = book higher

        # Find cluster -- the most common line value
        line_counts = defaultdict(int)
        for val in all_values:
            line_counts[val] += 1
        if line_counts:
            cluster_line = max(line_counts, key=line_counts.get)
            cluster_pct = round(line_counts[cluster_line] / len(all_values) * 100, 1)
        else:
            cluster_line = consensus_line
            cluster_pct = 100.0

        results[key] = {
            'player': prop['player'],
            'stat': prop['stat'],
            'game': prop['game'],
            'sgo_line': prop['sgo_line'],
            'consensus_line': consensus_line,
            'line_spread': round(line_spread, 1),
            'line_spread_all': round(line_spread_all, 1),
            'n_books': len(all_books),
            'n_sharp': len(sharp),
            'stdev': round(sd, 2),
            'outlier_book': outlier_book,
            'outlier_deviation': round(outlier_deviation, 1),
            'book_cluster': cluster_line,
            'cluster_pct': cluster_pct,
            'sharp_lines': prop['sharp_lines'],
        }

    return results


# ═══════════════════════════════════════════════════════════════
# 3. FIND EDGES (board vs multi-book consensus)
# ═══════════════════════════════════════════════════════════════

def find_edges(board_props, sgo_spreads):
    """Compare today's board lines vs SGO multi-book consensus.

    Args:
        board_props: list of dicts with 'player', 'stat', 'line' from our board
        sgo_spreads: output of compute_line_spread()

    Returns list of edge dicts sorted by |edge| descending:
        {
            player, stat, line (ours), consensus_line, edge, edge_direction,
            line_spread, n_books, outlier_book, outlier_deviation,
            confidence: 'HIGH' / 'MEDIUM' / 'LOW',
            signal: human-readable description
        }
    """
    edges = []

    for prop in board_props:
        player = prop.get('player', '')
        stat = prop.get('stat', '')
        our_line = prop.get('line', 0)

        if not player or not stat:
            continue

        # Match against SGO data
        key = (_norm(player), stat)
        spread_data = sgo_spreads.get(key)

        if not spread_data:
            # Try fuzzy match via name key
            nk = _name_key(player)
            for k, v in sgo_spreads.items():
                if k[1] == stat and _name_key(v['player']) == nk:
                    spread_data = v
                    break

        if not spread_data:
            continue

        consensus = spread_data['consensus_line']
        line_spread = spread_data['line_spread']
        n_books = spread_data['n_books']

        # Edge: how far is OUR line from the market consensus
        edge = consensus - our_line  # positive = our line is BELOW consensus (OVER edge)

        # Determine direction
        if edge > 0:
            edge_direction = 'OVER'   # our line below consensus -> OVER has edge
        elif edge < 0:
            edge_direction = 'UNDER'  # our line above consensus -> UNDER has edge
        else:
            edge_direction = 'NONE'

        abs_edge = abs(edge)

        # Confidence based on book agreement and edge magnitude
        if abs_edge >= EDGE_THRESHOLD and line_spread <= HIGH_DISAGREEMENT and n_books >= 5:
            confidence = 'HIGH'
        elif abs_edge >= 0.5 and line_spread <= VOLATILE_THRESHOLD and n_books >= 3:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        # Volatile flag -- high disagreement means the market itself is unsure
        volatile = line_spread >= VOLATILE_THRESHOLD

        # Build signal description
        signal_parts = []
        if abs_edge >= EDGE_THRESHOLD:
            signal_parts.append(
                f"Line edge: our {our_line} vs consensus {consensus} "
                f"({edge_direction} +{abs_edge:.1f}pt edge)"
            )
        if volatile:
            signal_parts.append(
                f"VOLATILE: {line_spread:.1f}pt spread across books -- market unsure"
            )
        if spread_data['outlier_book'] and abs(spread_data['outlier_deviation']) >= 2.0:
            ob = spread_data['outlier_book']
            od = spread_data['outlier_deviation']
            direction = 'higher' if od > 0 else 'lower'
            signal_parts.append(
                f"Outlier: {ob} is {abs(od):.1f}pt {direction} than consensus"
            )
        if spread_data['cluster_pct'] >= 80:
            signal_parts.append(
                f"Strong cluster: {spread_data['cluster_pct']:.0f}% of books at {spread_data['book_cluster']}"
            )

        edges.append({
            'player': player,
            'stat': stat,
            'our_line': our_line,
            'consensus_line': consensus,
            'edge': round(edge, 1),
            'abs_edge': round(abs_edge, 1),
            'edge_direction': edge_direction,
            'line_spread': line_spread,
            'n_books': n_books,
            'n_sharp': spread_data['n_sharp'],
            'stdev': spread_data['stdev'],
            'outlier_book': spread_data['outlier_book'],
            'outlier_deviation': spread_data['outlier_deviation'],
            'cluster_line': spread_data['book_cluster'],
            'cluster_pct': spread_data['cluster_pct'],
            'confidence': confidence,
            'volatile': volatile,
            'signal': ' | '.join(signal_parts) if signal_parts else 'No significant edge',
            'game': spread_data['game'],
        })

    # Sort by absolute edge descending
    edges.sort(key=lambda x: x['abs_edge'], reverse=True)
    return edges


# ═══════════════════════════════════════════════════════════════
# 4. PIPELINE INTEGRATION (enrich results)
# ═══════════════════════════════════════════════════════════════

def enrich_with_line_edges(results, date_str):
    """Enrich analyzed picks with line disagreement data from SGO.

    Called by the pipeline after analysis, before parlay building.
    Adds to each result dict:
        line_edge:          signed edge vs consensus (positive = favorable for our direction)
        book_disagreement:  line spread across sharp books
        consensus_line:     median line from sharp books
        line_confidence:    'HIGH' / 'MEDIUM' / 'LOW'
        line_volatile:      True if books disagree by 3+ points

    Args:
        results: list of analyzed pick dicts (from analyze_v3.py)
        date_str: date string for SGO cache lookup

    Returns:
        int: number of picks enriched
    """
    sgo_props = load_sgo_props(date_str)
    if not sgo_props:
        return 0

    spreads = compute_line_spread(sgo_props)
    if not spreads:
        return 0

    enriched = 0

    for pick in results:
        player = pick.get('player', '')
        stat = pick.get('stat', '')
        our_line = pick.get('line', 0)
        direction = pick.get('direction', 'OVER').upper()

        if not player or not stat:
            continue

        # Match against SGO spread data
        key = (_norm(player), stat)
        spread_data = spreads.get(key)

        if not spread_data:
            nk = _name_key(player)
            for k, v in spreads.items():
                if k[1] == stat and _name_key(v['player']) == nk:
                    spread_data = v
                    break

        if not spread_data:
            continue

        consensus = spread_data['consensus_line']
        line_spread = spread_data['line_spread']
        n_books = spread_data['n_books']

        # Raw edge: consensus - our_line
        # Positive means our line is below consensus (OVER edge)
        raw_edge = consensus - our_line

        # Directional edge: positive = favorable for our picked direction
        if direction == 'OVER':
            # OVER benefits when our line is BELOW consensus (raw_edge > 0)
            line_edge = raw_edge
        else:
            # UNDER benefits when our line is ABOVE consensus (raw_edge < 0)
            line_edge = -raw_edge

        # Confidence
        abs_edge = abs(raw_edge)
        if abs_edge >= EDGE_THRESHOLD and line_spread <= HIGH_DISAGREEMENT and n_books >= 5:
            confidence = 'HIGH'
        elif abs_edge >= 0.5 and line_spread <= VOLATILE_THRESHOLD and n_books >= 3:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        volatile = line_spread >= VOLATILE_THRESHOLD

        pick['line_edge'] = round(line_edge, 1)
        pick['book_disagreement'] = round(line_spread, 1)
        pick['consensus_line'] = consensus
        pick['line_confidence'] = confidence
        pick['line_volatile'] = volatile
        pick['line_n_books'] = n_books
        enriched += 1

    return enriched


# ═══════════════════════════════════════════════════════════════
# 5. CLI -- show all edges for a date
# ═══════════════════════════════════════════════════════════════

def _print_edges_report(date_str, edges_only=False, min_books=3):
    """Print a formatted report of line disagreements and edges."""
    sgo_props = load_sgo_props(date_str)
    if not sgo_props:
        print(f"No SGO data for {date_str}")
        print(f"  Looked in: {_CACHE_DIR}/props_{date_str}.json")
        return

    spreads = compute_line_spread(sgo_props)

    print(f"\n{'=' * 72}")
    print(f"  LINE DISAGREEMENT ANALYSIS -- {date_str}")
    print(f"{'=' * 72}")
    print(f"  SGO props loaded: {len(sgo_props)}")
    print(f"  Props with spread data: {len(spreads)}")

    # Summary stats
    all_spreads = [v['line_spread'] for v in spreads.values() if v['n_books'] >= min_books]
    if all_spreads:
        avg_spread = sum(all_spreads) / len(all_spreads)
        high_disagree = sum(1 for s in all_spreads if s >= HIGH_DISAGREEMENT)
        volatile = sum(1 for s in all_spreads if s >= VOLATILE_THRESHOLD)
        tight = sum(1 for s in all_spreads if s == 0)
        print(f"  Props with {min_books}+ books: {len(all_spreads)}")
        print(f"  Average sharp spread: {avg_spread:.2f} pts")
        print(f"  Tight (0 spread): {tight} ({tight/len(all_spreads)*100:.0f}%)")
        print(f"  High disagreement (2+ pts): {high_disagree}")
        print(f"  Volatile (3+ pts): {volatile}")

    # ─── BIGGEST DISAGREEMENTS ───
    print(f"\n{'─' * 72}")
    print(f"  TOP DISAGREEMENTS (sharp book spread, {min_books}+ books)")
    print(f"{'─' * 72}")

    sorted_by_spread = sorted(
        [v for v in spreads.values() if v['n_books'] >= min_books],
        key=lambda x: x['line_spread'],
        reverse=True
    )

    if edges_only:
        sorted_by_spread = [s for s in sorted_by_spread if s['line_spread'] >= EDGE_THRESHOLD]

    for i, s in enumerate(sorted_by_spread[:25]):
        flag = ''
        if s['line_spread'] >= VOLATILE_THRESHOLD:
            flag = ' [VOLATILE]'
        elif s['line_spread'] >= HIGH_DISAGREEMENT:
            flag = ' [HIGH]'

        print(f"\n  {i+1:2d}. {s['player']:25s} {s['stat'].upper():5s}  "
              f"consensus={s['consensus_line']:.1f}  "
              f"spread={s['line_spread']:.1f}{flag}")
        print(f"      Game: {s['game']}  |  Books: {s['n_books']} total, {s['n_sharp']} sharp  |  "
              f"stdev={s['stdev']:.2f}")

        if s['outlier_book'] and abs(s['outlier_deviation']) >= 1.0:
            direction = 'higher' if s['outlier_deviation'] > 0 else 'lower'
            print(f"      Outlier: {s['outlier_book']} is {abs(s['outlier_deviation']):.1f}pt "
                  f"{direction} than consensus")

        if s['cluster_pct'] >= 60:
            print(f"      Cluster: {s['cluster_pct']:.0f}% of books at {s['book_cluster']}")

        # Show sharp book lines
        if s['sharp_lines']:
            sharp_str = ', '.join(
                f"{b}={v:.1f}" for b, v in sorted(s['sharp_lines'].items(), key=lambda x: x[1])
            )
            print(f"      Sharp: {sharp_str}")

    # ─── OUTLIER BOOKS ───
    print(f"\n{'─' * 72}")
    print(f"  OUTLIER BOOK FREQUENCY")
    print(f"{'─' * 72}")

    outlier_counts = defaultdict(int)
    outlier_total_dev = defaultdict(float)
    for s in spreads.values():
        if s['outlier_book'] and abs(s['outlier_deviation']) >= 1.0:
            outlier_counts[s['outlier_book']] += 1
            outlier_total_dev[s['outlier_book']] += abs(s['outlier_deviation'])

    for book, count in sorted(outlier_counts.items(), key=lambda x: -x[1])[:10]:
        avg_dev = outlier_total_dev[book] / count
        tag = ' [NOISE]' if book in NOISE_BOOKS else ''
        print(f"  {book:25s} {count:3d} outlier props  avg deviation {avg_dev:.1f}pt{tag}")


def _print_board_edges(date_str, board_path, min_books=3):
    """Print edges between our board and SGO consensus."""
    sgo_props = load_sgo_props(date_str)
    if not sgo_props:
        print(f"No SGO data for {date_str}")
        return

    spreads = compute_line_spread(sgo_props)

    # Load board
    with open(board_path) as f:
        board = json.load(f)
    if isinstance(board, dict):
        board = board.get('props', board.get('results', []))

    edges = find_edges(board, spreads)
    edges = [e for e in edges if e['n_books'] >= min_books]

    print(f"\n{'=' * 72}")
    print(f"  BOARD vs CONSENSUS EDGES -- {date_str}")
    print(f"{'=' * 72}")
    print(f"  Board props: {len(board)}")
    print(f"  Matched to SGO: {len(edges)}")

    significant = [e for e in edges if e['abs_edge'] >= EDGE_THRESHOLD]
    favorable = [e for e in significant if e['edge'] > 0]
    unfavorable = [e for e in significant if e['edge'] < 0]

    print(f"  Significant edges (1+ pt): {len(significant)}")
    print(f"    OVER edges: {sum(1 for e in significant if e['edge_direction'] == 'OVER')}")
    print(f"    UNDER edges: {sum(1 for e in significant if e['edge_direction'] == 'UNDER')}")

    # Show top edges
    print(f"\n{'─' * 72}")
    print(f"  STRONGEST EDGES")
    print(f"{'─' * 72}")

    for i, e in enumerate(edges[:30]):
        if e['abs_edge'] < 0.5:
            break

        flag = ''
        if e['volatile']:
            flag = ' [VOLATILE-AVOID]'
        elif e['confidence'] == 'HIGH':
            flag = ' [HIGH CONFIDENCE]'
        elif e['confidence'] == 'MEDIUM':
            flag = ' [MEDIUM]'

        dir_str = e['edge_direction']
        print(f"\n  {i+1:2d}. {e['player']:25s} {e['stat'].upper():5s}  "
              f"ours={e['our_line']:.1f}  consensus={e['consensus_line']:.1f}  "
              f"edge={e['edge']:+.1f} {dir_str}{flag}")
        print(f"      {e['game']}  |  {e['n_books']} books, {e['n_sharp']} sharp  |  "
              f"spread={e['line_spread']:.1f}")
        if e['signal']:
            print(f"      {e['signal']}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NBA Line Disagreement Analyzer')
    parser.add_argument('--date', required=True, help='Date to analyze (YYYY-MM-DD)')
    parser.add_argument('--board', help='Path to board JSON for edge comparison')
    parser.add_argument('--edges-only', action='store_true',
                        help='Only show props with significant disagreement')
    parser.add_argument('--min-books', type=int, default=3,
                        help='Minimum books required for analysis (default: 3)')
    args = parser.parse_args()

    if args.board:
        _print_board_edges(args.date, args.board, min_books=args.min_books)
    else:
        _print_edges_report(args.date, edges_only=args.edges_only,
                            min_books=args.min_books)

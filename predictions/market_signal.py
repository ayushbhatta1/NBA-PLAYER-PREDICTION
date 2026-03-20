#!/usr/bin/env python3
"""
Market Signal Module — ported from TennisPredict.

Extracts fair odds, bookmaker consensus, line movement, and value edges
from SGO cached player prop data. Gives the parlay engine what 30+ sportsbooks
already know.

Usage:
    signals = MarketSignal('2026-03-18')
    sig = signals.get_signal('Jayson Tatum', 'pts', 24.5)
    # → {'fair_over_prob': 0.52, 'consensus_over': 0.51, 'movement': +1.2,
    #    'best_book_over': 'fanduel', 'best_over_odds': '-108', 'vig': 2.1,
    #    'edge_over': 3.5, 'edge_under': -3.5}
"""
import json
import os
import unicodedata


def american_to_prob(odds_str):
    """Convert American odds string to implied probability (0-1)."""
    try:
        odds = int(str(odds_str).replace('+', ''))
    except (ValueError, TypeError):
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    return 0.5


def _norm(s):
    nfkd = unicodedata.normalize('NFKD', str(s))
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


class MarketSignal:
    def __init__(self, date_str):
        self.date = date_str
        self.props = {}  # keyed by (player_norm, stat, line)
        self._load(date_str)

    def _load(self, date_str):
        """Load SGO cached props for a date."""
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache', 'sgo')
        path = os.path.join(cache_dir, f'props_{date_str}.json')
        if not os.path.exists(path):
            return

        with open(path) as f:
            data = json.load(f)

        raw_props = data.get('props', data) if isinstance(data, dict) else data

        for p in raw_props:
            player = _norm(p.get('player', ''))
            stat = p.get('stat', '')
            line = p.get('line', 0)
            key = (player, stat, line)

            # Store raw SGO data including book_lines
            self.props[key] = p

    def get_signal(self, player_name, stat, line, our_prob=None):
        """
        Get market signal for a specific prop.

        Args:
            player_name: Player name
            stat: Stat type (pts, reb, ast, etc.)
            line: Prop line value
            our_prob: Our model's win probability (0-1) for edge calculation

        Returns dict with fair_over_prob, consensus, movement, vig, edge, etc.
        """
        key = (_norm(player_name), stat, line)
        p = self.props.get(key)

        if not p:
            # Try fuzzy match on player name
            player_parts = _norm(player_name).split()
            for k, v in self.props.items():
                if k[1] == stat and k[2] == line:
                    k_parts = k[0].split()
                    if (len(player_parts) >= 2 and len(k_parts) >= 2 and
                        player_parts[-1] == k_parts[-1] and player_parts[0][0] == k_parts[0][0]):
                        p = v
                        break

        if not p:
            return None

        result = {
            'has_market': True,
            'fair_over_prob': None,
            'consensus_over_prob': None,
            'movement': None,
            'best_book': None,
            'best_book_odds': None,
            'vig': None,
            'edge_over': None,
            'edge_under': None,
            'num_books': 0,
        }

        # Fair odds (vig-removed — SGO's computed fair value)
        fair_odds = p.get('fair_odds')
        if fair_odds:
            fair_prob = american_to_prob(fair_odds)
            if fair_prob:
                result['fair_over_prob'] = fair_prob

        # Bookmaker consensus — average implied prob across all books
        book_lines = p.get('book_lines', {})
        if book_lines:
            over_probs = []
            best_odds_val = None
            best_book = None

            for book, odds_str in book_lines.items():
                prob = american_to_prob(str(odds_str))
                if prob:
                    over_probs.append(prob)
                    # Track best odds (highest implied prob = worst for bettor...
                    # Actually for OVER, best odds = lowest implied prob = best payout)
                    odds_val = int(str(odds_str).replace('+', ''))
                    if best_odds_val is None or odds_val > best_odds_val:
                        best_odds_val = odds_val
                        best_book = book

            if over_probs:
                consensus = sum(over_probs) / len(over_probs)
                result['consensus_over_prob'] = consensus
                result['num_books'] = len(over_probs)

                if best_book:
                    result['best_book'] = best_book
                    result['best_book_odds'] = str(best_odds_val)

                # Vig = difference between book consensus and fair odds
                if result['fair_over_prob']:
                    result['vig'] = round((consensus - result['fair_over_prob']) * 100, 1)

        # Edge calculation: our model prob vs market (consensus or fair odds)
        market_prob = result['consensus_over_prob'] or result['fair_over_prob']
        if our_prob is not None and market_prob:
            result['edge_over'] = round((our_prob - market_prob) * 100, 1)
            result['edge_under'] = round(((1 - our_prob) - (1 - market_prob)) * 100, 1)

        return result

    def enrich_picks(self, picks):
        """Add market signal data to a list of analyzed picks."""
        enriched = 0
        for p in picks:
            player = p.get('player', '')
            stat = p.get('stat', '')
            line = p.get('line', 0)

            # Our model's probability for the OVER direction
            xgb = p.get('xgb_prob', 0.5)
            direction = p.get('direction', 'OVER').upper()
            our_over_prob = xgb if direction == 'OVER' else (1 - xgb)

            sig = self.get_signal(player, stat, line, our_prob=our_over_prob)
            if sig:
                p['market_fair_over'] = sig.get('fair_over_prob')
                p['market_consensus_over'] = sig.get('consensus_over_prob')
                p['market_vig'] = sig.get('vig')
                p['market_edge'] = sig.get('edge_over') if direction == 'OVER' else sig.get('edge_under')
                p['market_num_books'] = sig.get('num_books', 0)
                p['market_best_book'] = sig.get('best_book')
                enriched += 1

        return enriched

    def __len__(self):
        return len(self.props)


if __name__ == '__main__':
    import sys
    date = sys.argv[1] if len(sys.argv) > 1 else '2026-03-18'
    ms = MarketSignal(date)
    print(f"Loaded {len(ms)} props for {date}")

    # Show a sample
    for key, p in list(ms.props.items())[:5]:
        sig = ms.get_signal(p['player'], p['stat'], p['line'])
        if sig:
            print(f"  {p['player']:25s} {p['stat']:5s} {p['line']:5.1f} | "
                  f"fair={sig['fair_over_prob']:.1%} consensus={sig['consensus_over_prob']:.1%} "
                  f"books={sig['num_books']} vig={sig['vig']}")

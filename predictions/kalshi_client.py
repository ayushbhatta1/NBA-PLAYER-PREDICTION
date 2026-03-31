#!/usr/bin/env python3
"""
Kalshi API client for NBA player prop market data.
Pulls exchange odds (implied probabilities) for calibration + CLV tracking.

Usage:
    python3 kalshi_client.py --markets          # List today's NBA markets
    python3 kalshi_client.py --props            # Pull player prop prices
    python3 kalshi_client.py --compare <board>  # Compare model vs Kalshi odds
"""

import json, os, sys, time, re
from datetime import datetime, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_URL = "https://demo-trading-api.kalshi.com/trade-api/v2"

# Rate limit: 10 req/s
RATE_LIMIT = 0.15

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "kalshi")


class KalshiClient:
    """Kalshi exchange API client."""

    def __init__(self, api_key=None, email=None, password=None, demo=False):
        self.base = DEMO_URL if demo else BASE_URL
        self.token = None
        self.api_key = api_key
        self.last_call = 0
        os.makedirs(CACHE_DIR, exist_ok=True)

        if email and password:
            self._login(email, password)
        elif api_key:
            self.token = api_key

    def _login(self, email, password):
        """Login with email/password to get bearer token."""
        data = json.dumps({"email": email, "password": password}).encode()
        req = Request(f"{self.base}/login", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            resp = urlopen(req)
            result = json.loads(resp.read())
            self.token = result.get("token")
            print(f"  [Kalshi] Logged in, token obtained")
        except HTTPError as e:
            body = e.read().decode()
            print(f"  [Kalshi] Login failed: {e.code} {body}")

    def _rate_limit(self):
        elapsed = time.time() - self.last_call
        if elapsed < RATE_LIMIT:
            time.sleep(RATE_LIMIT - elapsed)
        self.last_call = time.time()

    def _get(self, path, params=None):
        """Make authenticated GET request."""
        self._rate_limit()
        url = f"{self.base}{path}"
        if params:
            url += "?" + urlencode({k: v for k, v in params.items() if v is not None})

        req = Request(url, method="GET")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/json")

        try:
            resp = urlopen(req)
            return json.loads(resp.read())
        except HTTPError as e:
            body = e.read().decode()
            print(f"  [Kalshi] GET {path} failed: {e.code} {body[:200]}")
            return None
        except URLError as e:
            print(f"  [Kalshi] Connection error: {e}")
            return None

    def get_exchange_status(self):
        """Check if exchange is online."""
        return self._get("/exchange/status")

    def get_events(self, series_ticker=None, status="open", limit=200, cursor=None):
        """List events, optionally filtered by series."""
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return self._get("/events", params)

    def get_markets(self, event_ticker=None, series_ticker=None, status="open",
                    limit=200, cursor=None, ticker=None):
        """List markets with filters."""
        params = {"status": status, "limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        if ticker:
            params["ticker"] = ticker
        return self._get("/markets", params)

    def get_market(self, ticker):
        """Get single market details."""
        return self._get(f"/markets/{ticker}")

    def get_orderbook(self, ticker):
        """Get order book for a market."""
        return self._get(f"/markets/{ticker}/orderbook")

    def search_nba_events(self):
        """Find all NBA-related events by searching common series tickers."""
        nba_events = []
        # Try common NBA series ticker patterns
        search_terms = ["NBA", "KXNBA", "NBAPROPS", "KXNBAPROPS"]

        # First try: get all open events and filter
        print("  [Kalshi] Searching for NBA events...")
        result = self._get("/events", {"status": "open", "limit": 200})
        if result and "events" in result:
            for evt in result["events"]:
                title = (evt.get("title", "") + " " + evt.get("sub_title", "")).upper()
                ticker = evt.get("event_ticker", "").upper()
                series = evt.get("series_ticker", "").upper()
                cat = evt.get("category", "").upper()
                if any(term in title or term in ticker or term in series or term in cat
                       for term in ["NBA", "BASKETBALL", "LAKERS", "CELTICS", "KNICKS",
                                    "CAVALIERS", "BUCKS", "SUNS", "MAGIC", "NETS",
                                    "ROCKETS", "MAVERICKS", "CLIPPERS", "BLAZERS",
                                    "HORNETS", "PISTONS", "RAPTORS"]):
                    nba_events.append(evt)

            # If no NBA events found in first page, try with cursor
            cursor = result.get("cursor")
            pages = 1
            while cursor and pages < 5 and not nba_events:
                result = self._get("/events", {"status": "open", "limit": 200, "cursor": cursor})
                if not result or "events" not in result:
                    break
                for evt in result["events"]:
                    title = (evt.get("title", "") + " " + evt.get("sub_title", "")).upper()
                    if "NBA" in title or "BASKETBALL" in title:
                        nba_events.append(evt)
                cursor = result.get("cursor")
                pages += 1

        print(f"  [Kalshi] Found {len(nba_events)} NBA events")
        return nba_events

    def get_nba_markets(self):
        """Get all NBA player prop markets."""
        events = self.search_nba_events()
        all_markets = []

        for evt in events:
            event_ticker = evt.get("event_ticker", "")
            result = self._get("/markets", {
                "event_ticker": event_ticker,
                "status": "open",
                "limit": 200,
            })
            if result and "markets" in result:
                for mkt in result["markets"]:
                    mkt["_event"] = evt  # attach parent event
                    all_markets.append(mkt)

        print(f"  [Kalshi] Found {len(all_markets)} NBA markets")
        return all_markets

    def parse_player_props(self, markets):
        """Parse raw markets into standardized player prop format.
        Returns list of dicts with player, stat, line, kalshi_yes, kalshi_no, implied_prob."""
        props = []

        for mkt in markets:
            title = mkt.get("title", "")
            subtitle = mkt.get("subtitle", "")

            # Try to extract player name, stat, and threshold from title
            # Common formats:
            # "Will LeBron James score 25+ points?"
            # "LeBron James Over 24.5 Points"
            prop = self._parse_market_title(title, subtitle, mkt)
            if prop:
                props.append(prop)

        return props

    def _parse_market_title(self, title, subtitle, mkt):
        """Extract player/stat/line from market title."""
        title_lower = title.lower()

        # Pattern: "Will [Player] [get/score/record] [X]+ [stat]?"
        m = re.match(
            r"will\s+(.+?)\s+(?:score|get|record|have)\s+(\d+\.?\d*)\+?\s+(points?|rebounds?|assists?|steals?|blocks?|three.?pointers?|3.?pointers?|3pm)",
            title_lower
        )
        if not m:
            # Pattern: "[Player] Over/Under [X] [stat]"
            m = re.match(
                r"(.+?)\s+(?:over|under)\s+(\d+\.?\d*)\s+(points?|rebounds?|assists?|steals?|blocks?|three.?pointers?|3.?pointers?|3pm)",
                title_lower
            )
        if not m:
            return None

        player_name = m.group(1).strip().title()
        threshold = float(m.group(2))
        stat_raw = m.group(3).lower()

        # Map to our stat keys
        stat_map = {
            'point': 'pts', 'points': 'pts',
            'rebound': 'reb', 'rebounds': 'reb',
            'assist': 'ast', 'assists': 'ast',
            'steal': 'stl', 'steals': 'stl',
            'block': 'blk', 'blocks': 'blk',
            'three pointer': '3pm', 'three pointers': '3pm',
            'three-pointer': '3pm', 'three-pointers': '3pm',
            '3 pointer': '3pm', '3 pointers': '3pm',
            '3pm': '3pm',
        }
        stat = stat_map.get(stat_raw, stat_raw)

        # Extract prices
        yes_bid = mkt.get("yes_bid", 0) or 0
        yes_ask = mkt.get("yes_ask", 0) or 0
        no_bid = mkt.get("no_bid", 0) or 0
        no_ask = mkt.get("no_ask", 0) or 0
        last_price = mkt.get("last_price", 0) or 0
        volume = mkt.get("volume", 0) or 0

        # Mid price as implied probability
        if yes_bid and yes_ask:
            implied_over = (yes_bid + yes_ask) / 200  # cents to probability
        elif last_price:
            implied_over = last_price / 100
        else:
            implied_over = 0.5

        return {
            "player": player_name,
            "stat": stat,
            "line": threshold - 0.5,  # Convert "25+" to line 24.5
            "kalshi_ticker": mkt.get("ticker", ""),
            "kalshi_yes_bid": yes_bid,
            "kalshi_yes_ask": yes_ask,
            "kalshi_no_bid": no_bid,
            "kalshi_no_ask": no_ask,
            "kalshi_last": last_price,
            "kalshi_volume": volume,
            "kalshi_implied_over": round(implied_over, 4),
            "kalshi_implied_under": round(1 - implied_over, 4),
            "title": title,
        }

    def compare_with_model(self, kalshi_props, model_results):
        """Compare Kalshi implied probs with model predictions.
        Returns list of comparisons with edge calculations."""
        comparisons = []

        # Build lookup from model results
        model_lookup = {}
        for r in model_results:
            key = (r.get("player", "").lower(), r.get("stat", "").lower())
            model_lookup[key] = r

        for kp in kalshi_props:
            key = (kp["player"].lower(), kp["stat"].lower())
            model = model_lookup.get(key)

            if not model:
                continue

            # Our model's probability
            model_prob = model.get("ensemble_prob") or model.get("xgb_prob") or 0.5
            direction = model.get("direction", "OVER").upper()

            # Compare
            if direction == "OVER":
                model_dir_prob = model_prob
                kalshi_dir_prob = kp["kalshi_implied_over"]
            else:
                model_dir_prob = model_prob
                kalshi_dir_prob = kp["kalshi_implied_under"]

            edge = model_dir_prob - kalshi_dir_prob

            comparisons.append({
                "player": kp["player"],
                "stat": kp["stat"],
                "line": kp["line"],
                "model_line": model.get("line", 0),
                "direction": direction,
                "model_prob": round(model_dir_prob, 4),
                "kalshi_prob": round(kalshi_dir_prob, 4),
                "edge": round(edge, 4),
                "edge_pct": round(edge * 100, 2),
                "kalshi_volume": kp["kalshi_volume"],
                "tier": model.get("tier", "?"),
                "reg_margin": model.get("reg_margin", 0),
                "kalshi_ticker": kp["kalshi_ticker"],
            })

        # Sort by absolute edge
        comparisons.sort(key=lambda x: abs(x["edge"]), reverse=True)
        return comparisons

    def cache_markets(self, date_str=None):
        """Fetch and cache today's NBA markets."""
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        markets = self.get_nba_markets()
        props = self.parse_player_props(markets)

        cache_file = os.path.join(CACHE_DIR, f"kalshi_{date_str}.json")
        with open(cache_file, "w") as f:
            json.dump({
                "date": date_str,
                "fetched_at": datetime.now().isoformat(),
                "raw_markets": len(markets),
                "parsed_props": len(props),
                "markets": markets,
                "props": props,
            }, f, indent=2)

        print(f"  [Kalshi] Cached {len(props)} props to {cache_file}")
        return props


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kalshi NBA market data")
    parser.add_argument("--key", default=os.environ.get("KALSHI_API_KEY"), help="API key")
    parser.add_argument("--markets", action="store_true", help="List NBA markets")
    parser.add_argument("--props", action="store_true", help="Pull player props")
    parser.add_argument("--compare", metavar="BOARD", help="Compare model vs Kalshi")
    parser.add_argument("--cache", action="store_true", help="Cache today's markets")
    parser.add_argument("--status", action="store_true", help="Check exchange status")
    parser.add_argument("--demo", action="store_true", help="Use demo API")
    args = parser.parse_args()

    key = args.key or "3d599518-2433-4635-bcee-1ffc8823ec59"
    client = KalshiClient(api_key=key, demo=args.demo)

    if args.status:
        status = client.get_exchange_status()
        print(json.dumps(status, indent=2))
        return

    if args.markets or args.props or args.cache:
        markets = client.get_nba_markets()

        if args.props or args.cache:
            props = client.parse_player_props(markets)
            print(f"\n  Parsed {len(props)} player props:")
            for p in props[:20]:
                print(f"    {p['player']:22s} {p['stat']:5s} line={p['line']:5.1f}  "
                      f"over={p['kalshi_implied_over']:.1%}  under={p['kalshi_implied_under']:.1%}  "
                      f"vol={p['kalshi_volume']}")

            if args.cache:
                client.cache_markets()
        else:
            print(f"\n  {len(markets)} markets:")
            for m in markets[:30]:
                print(f"    [{m.get('ticker','')}] {m.get('title','')}")
                print(f"      yes={m.get('yes_bid',0)}/{m.get('yes_ask',0)}  "
                      f"last={m.get('last_price',0)}  vol={m.get('volume',0)}")

    if args.compare:
        with open(args.compare) as f:
            data = json.load(f)
        model_results = data if isinstance(data, list) else data.get("results", [])

        props = client.cache_markets()
        comparisons = client.compare_with_model(props, model_results)

        print(f"\n{'='*80}")
        print(f"  MODEL vs KALSHI — {len(comparisons)} matched props")
        print(f"{'='*80}")
        print(f"  {'Player':22s} {'Stat':5s} {'Dir':5s} {'Model':7s} {'Kalshi':7s} {'Edge':7s} {'Vol':6s} {'Tier':4s}")
        print(f"  {'─'*70}")
        for c in comparisons[:30]:
            edge_marker = "+++" if c["edge"] > 0.10 else "++" if c["edge"] > 0.05 else "+" if c["edge"] > 0 else "---"
            print(f"  {c['player']:22s} {c['stat']:5s} {c['direction']:5s} "
                  f"{c['model_prob']:6.1%} {c['kalshi_prob']:6.1%} {c['edge']:+6.1%} "
                  f"{c['kalshi_volume']:6d} {c['tier']:4s} {edge_marker}")


if __name__ == "__main__":
    main()

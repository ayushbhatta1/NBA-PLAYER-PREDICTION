#!/usr/bin/env python3
"""
Sports Game Odds API client for NBA player props.
Fetches real sportsbook lines (FanDuel, DraftKings, BetMGM, Caesars, ESPN BET).

API: https://api.sportsgameodds.com/v2/
Auth: apiKey query parameter
Counting: 1 credit = 1 event (game), not per prop line

NOTE: Player props are only available on upcoming/live games.
      Completed games only retain team-level odds.
      To backtest with real lines, we must collect daily BEFORE games start.
"""

import json
import os
import time
import requests
from datetime import datetime, timedelta
from collections import defaultdict

BASE_URL = "https://api.sportsgameodds.com/v2"

# Priority bookmakers for consensus line
PRIORITY_BOOKS = ["fanduel", "draftkings", "betmgm", "caesars", "espnbet"]

# Map SGO stat IDs to our pipeline stat names
STAT_MAP = {
    "points": "pts",
    "assists": "ast",
    "rebounds": "reb",
    "threePointersMade": "3pm",
    "blocks": "blk",
    "steals": "stl",
    "points+rebounds+assists": "pra",
    "points+assists": "pa",
    "points+rebounds": "pr",
    "rebounds+assists": "ra",
}


class SGOClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("SGO_API_KEY") or self._load_key()
        if not self.api_key:
            raise ValueError("No SGO API key. Set SGO_API_KEY env var or add to .env")
        self.session = requests.Session()
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache", "sgo")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _load_key(self):
        """Try loading from .env file."""
        for env_path in [
            os.path.join(os.path.dirname(__file__), ".env"),
            os.path.join(os.path.dirname(__file__), "..", ".env"),
        ]:
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        if line.strip().startswith("SGO_API_KEY="):
                            return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
        return None

    def _get(self, endpoint, params=None):
        """Make authenticated GET request."""
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        url = f"{BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get_all_pages(self, endpoint, params=None):
        """Paginate through all results."""
        if params is None:
            params = {}
        params["limit"] = 100
        all_data = []
        cursor = None

        while True:
            if cursor:
                params["cursor"] = cursor
            result = self._get(endpoint, params)
            data = result.get("data", [])
            all_data.extend(data)
            cursor = result.get("nextCursor")
            if not cursor or not data:
                break
            time.sleep(0.2)

        return all_data

    def get_upcoming_events(self):
        """Get all upcoming NBA games with live odds (including player props)."""
        return self._get_all_pages("events", {
            "leagueID": "NBA",
            "oddsAvailable": "true",
        })

    def extract_player_props(self, events):
        """
        Extract player prop lines from event data.
        Returns list of dicts matching our pipeline format.
        """
        props = []

        for event in events:
            teams = event.get("teams", {})
            home = teams.get("home", {}).get("names", {}).get("short", "?")
            away = teams.get("away", {}).get("names", {}).get("short", "?")
            game_str = f"{away}@{home}"
            start_time = event.get("status", {}).get("startsAt", "")

            odds = event.get("odds", {})

            for odd_id, odd_data in odds.items():
                # Only player prop over/unders
                if "game-ou-over" not in odd_id:
                    continue
                player_id = odd_data.get("playerID")
                if not player_id:
                    continue

                stat_id = odd_data.get("statID", "")
                if stat_id not in STAT_MAP:
                    continue

                # Get the line (fairOverUnder is the consensus/fair line)
                fair_line = odd_data.get("fairOverUnder")
                book_line = odd_data.get("bookOverUnder")
                line = float(fair_line) if fair_line else (float(book_line) if book_line else None)
                if line is None:
                    continue

                # Extract per-bookmaker lines
                book_lines = {}
                for book_id, book_data in odd_data.get("byBookmaker", {}).items():
                    ou = book_data.get("overUnder")
                    if ou is not None:
                        book_lines[book_id] = float(ou)

                # Get player name
                player_name = self._resolve_player_name(event, player_id)

                # Fair odds (juice indicator)
                fair_odds = odd_data.get("fairOdds", "")

                props.append({
                    "player": player_name,
                    "player_id": player_id,
                    "stat": STAT_MAP[stat_id],
                    "line": line,
                    "book_lines": book_lines,
                    "fair_odds": fair_odds,
                    "game": game_str,
                    "event_id": event.get("eventID"),
                    "start_time": start_time,
                    "source": "sportsgameodds",
                })

        return props

    def _resolve_player_name(self, event, player_id):
        """Convert player ID like JALEN_BRUNSON_1_NBA to display name."""
        # Remove _NBA suffix and trailing number
        clean = player_id.replace("_NBA", "")
        parts = clean.split("_")
        if parts and parts[-1].isdigit():
            parts = parts[:-1]
        return " ".join(p.capitalize() for p in parts)

    def fetch_and_cache_today(self):
        """
        Fetch today's props and cache them. Call this BEFORE games start
        to preserve lines for backtesting (props disappear after games end).
        """
        # Use Pacific time for date — user is in PT
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        now_pt = datetime.now(ZoneInfo("America/Los_Angeles"))
        today = now_pt.strftime("%Y-%m-%d")
        tomorrow = (now_pt + timedelta(days=1)).strftime("%Y-%m-%d")
        cache_file = os.path.join(self.cache_dir, f"props_{today}.json")

        print(f"[SGO] Fetching NBA props for {today} (Pacific)...")
        events = self.get_upcoming_events()

        # SGO uses UTC. Convert each event start to PT and match today's date.
        today_events = []
        for e in events:
            starts_at = e.get("status", {}).get("startsAt", "")
            if not starts_at:
                continue
            try:
                utc_dt = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
                pt_dt = utc_dt.astimezone(ZoneInfo("America/Los_Angeles"))
                if pt_dt.strftime("%Y-%m-%d") == today:
                    today_events.append(e)
            except (ValueError, TypeError):
                continue

        props = self.extract_player_props(today_events)
        print(f"[SGO] Got {len(props)} prop lines from {len(today_events)} games")

        # Save with timestamp
        cache_data = {
            "date": today,
            "fetched_at": datetime.now().isoformat(),
            "games": len(today_events),
            "total_props": len(props),
            "props": props,
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(f"[SGO] Cached to {cache_file}")
        return props

    def load_cached_props(self, date_str):
        """Load previously cached props for a date."""
        cache_file = os.path.join(self.cache_dir, f"props_{date_str}.json")
        if not os.path.exists(cache_file):
            return None
        with open(cache_file) as f:
            data = json.load(f)
        return data.get("props", data) if isinstance(data, dict) else data

    def get_board_for_pipeline(self, date_str=None):
        """
        Get props formatted for the v4 pipeline.
        If date_str is provided, loads from cache. Otherwise fetches live.
        Returns list of {player, stat, line, game, book_lines} dicts.
        """
        if date_str:
            props = self.load_cached_props(date_str)
            if not props:
                print(f"[SGO] No cached props for {date_str}")
                return []
        else:
            props = self.fetch_and_cache_today()

        board = []
        for p in props:
            board.append({
                "player": p["player"],
                "stat": p["stat"],
                "line": p["line"],
                "game": p["game"],
                "book_lines": p.get("book_lines", {}),
                "fair_odds": p.get("fair_odds", ""),
                "source": "sportsgameodds",
            })

        return board


    def fetch_historical_events(self, starts_after, starts_before, batch_days=30):
        """Fetch completed NBA events with team-level odds (spreads, totals, moneylines).

        Player props vanish after completion, but team-level odds persist.
        Paginates through date ranges in batches to avoid API limits.

        Returns list of event dicts with extracted odds.
        """
        from datetime import datetime as dt

        all_events = []
        current = dt.fromisoformat(starts_after)
        end = dt.fromisoformat(starts_before)

        batch_num = 0
        while current < end:
            batch_end = min(current + timedelta(days=batch_days), end)
            batch_num += 1
            sa = current.strftime("%Y-%m-%dT00:00:00Z")
            sb = batch_end.strftime("%Y-%m-%dT23:59:59Z")
            print(f"  [SGO] Batch {batch_num}: {current.strftime('%Y-%m-%d')} → {batch_end.strftime('%Y-%m-%d')}...", end="", flush=True)

            try:
                events = self._get_all_pages("events", {
                    "leagueID": "NBA",
                    "startsAfter": sa,
                    "startsBefore": sb,
                    "ended": "true",
                })
                all_events.extend(events)
                print(f" {len(events)} events")
            except Exception as e:
                print(f" ERROR: {e}")

            current = batch_end + timedelta(days=1)
            time.sleep(0.3)

        return all_events

    def extract_team_odds(self, events):
        """Extract spread, game total, and moneyline from completed events.

        Returns list of dicts: {date, home, away, spread, game_total, home_ml, away_ml, home_score, away_score}
        """
        results = []
        for event in events:
            teams = event.get("teams", {})
            home = teams.get("home", {}).get("names", {}).get("short", "?")
            away = teams.get("away", {}).get("names", {}).get("short", "?")
            home_score = teams.get("home", {}).get("score")
            away_score = teams.get("away", {}).get("score")
            start_time = event.get("status", {}).get("startsAt", "")

            # Parse date from startsAt
            game_date = ""
            if start_time:
                try:
                    game_date = start_time[:10]
                except Exception:
                    pass

            odds = event.get("odds", {})

            # Extract spread, total, moneyline from odds keys
            # Spreads: "points-home-game-sp-home" → fairSpread/bookSpread
            # Totals: "points-all-game-ou-over" → fairOverUnder/bookOverUnder
            # Moneylines: "points-home-game-ml-home" → fairOdds/bookOdds
            spread = None
            game_total = None
            home_ml = None
            away_ml = None

            for odd_id, odd_data in odds.items():
                # Home spread (negative = home favored)
                if "game-sp-home" in odd_id:
                    fair = odd_data.get("fairSpread") or odd_data.get("bookSpread")
                    if fair is not None:
                        try:
                            spread = float(fair)
                        except (ValueError, TypeError):
                            pass

                # Game total (over/under line)
                if "game-ou-over" in odd_id and "all-" in odd_id:
                    fair = odd_data.get("fairOverUnder") or odd_data.get("bookOverUnder")
                    if fair is not None:
                        try:
                            game_total = float(fair)
                        except (ValueError, TypeError):
                            pass

                # Moneylines
                if "game-ml-home" in odd_id:
                    fair_odds = odd_data.get("fairOdds") or odd_data.get("bookOdds")
                    if fair_odds:
                        try:
                            home_ml = float(str(fair_odds).replace("+", ""))
                        except (ValueError, TypeError):
                            pass
                elif "game-ml-away" in odd_id:
                    fair_odds = odd_data.get("fairOdds") or odd_data.get("bookOdds")
                    if fair_odds:
                        try:
                            away_ml = float(str(fair_odds).replace("+", ""))
                        except (ValueError, TypeError):
                            pass

            results.append({
                "date": game_date,
                "home": home,
                "away": away,
                "game": f"{away}@{home}",
                "spread": spread,
                "game_total": game_total,
                "home_ml": home_ml,
                "away_ml": away_ml,
                "home_score": home_score,
                "away_score": away_score,
                "event_id": event.get("eventID"),
            })

        return results

    def fetch_all_historical(self):
        """Fetch all historical NBA events and cache team-level odds.

        Covers 2024-02-01 to today. Extracts spreads, totals, moneylines.
        Saves to predictions/cache/sgo/historical_events.json
        """
        cache_path = os.path.join(self.cache_dir, "historical_events.json")

        print("=" * 60)
        print("  SGO Historical Events Fetch")
        print("=" * 60)
        print(f"  Range: 2024-02-01 → 2026-03-18")

        raw_events = self.fetch_historical_events("2024-02-01", "2026-03-18")
        print(f"\n  Total raw events: {len(raw_events)}")

        team_odds = self.extract_team_odds(raw_events)
        print(f"  Extracted team odds: {len(team_odds)}")

        # Stats
        has_spread = sum(1 for r in team_odds if r["spread"] is not None)
        has_total = sum(1 for r in team_odds if r["game_total"] is not None)
        has_score = sum(1 for r in team_odds if r["home_score"] is not None)
        dates = sorted(set(r["date"] for r in team_odds if r["date"]))
        print(f"  With spread: {has_spread} ({has_spread/len(team_odds)*100:.1f}%)" if team_odds else "")
        print(f"  With game total: {has_total} ({has_total/len(team_odds)*100:.1f}%)" if team_odds else "")
        print(f"  With scores: {has_score}")
        print(f"  Date range: {dates[0]} → {dates[-1]} ({len(dates)} days)" if dates else "  No dates")

        cache_data = {
            "fetched_at": datetime.now().isoformat(),
            "total_events": len(raw_events),
            "total_odds": len(team_odds),
            "date_range": [dates[0], dates[-1]] if dates else [],
            "events": team_odds,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"  Saved to {cache_path} ({size_mb:.1f} MB)")

        return team_odds

    def probe_metadata(self):
        """Quick exploratory calls for player roster, team metadata, stat types."""
        results = {}
        for endpoint in ["leagues/NBA", "teams?leagueID=NBA", "statTypes?leagueID=NBA"]:
            try:
                data = self._get(endpoint)
                results[endpoint] = data
                print(f"  [SGO] {endpoint}: {type(data).__name__}, {len(data) if isinstance(data, (list, dict)) else 'N/A'} items")
            except Exception as e:
                print(f"  [SGO] {endpoint}: ERROR {e}")
                results[endpoint] = str(e)
        return results


if __name__ == "__main__":
    import sys
    client = SGOClient()

    if len(sys.argv) > 1 and sys.argv[1] == "--cache":
        # Cache mode: save today's props for later backtesting
        client.fetch_and_cache_today()
    elif len(sys.argv) > 1 and sys.argv[1] == "--history":
        # Fetch all historical events (spreads, totals, moneylines)
        client.fetch_all_historical()
    elif len(sys.argv) > 1 and sys.argv[1] == "--probe":
        # Probe metadata endpoints
        client.probe_metadata()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-completed":
        # Test fetching a specific completed date
        test_date = sys.argv[2] if len(sys.argv) > 2 else "2026-03-12"
        print(f"  Testing completed event fetch for {test_date}...")
        events = client.fetch_historical_events(test_date, test_date)
        print(f"  Got {len(events)} events")
        if events:
            odds = client.extract_team_odds(events)
            for o in odds:
                print(f"    {o['game']} spread={o['spread']} total={o['game_total']} score={o['home_score']}-{o['away_score']}")
    else:
        # Display mode: show today's board
        props = client.fetch_and_cache_today()

        print(f"\n{'='*60}")
        print(f"  NBA Player Props (via Sports Game Odds)")
        print(f"{'='*60}")
        print(f"  Total props: {len(props)}")

        by_game = defaultdict(list)
        for p in props:
            by_game[p["game"]].append(p)

        for game, game_props in sorted(by_game.items()):
            print(f"\n  {game} ({len(game_props)} props)")
            for p in sorted(game_props, key=lambda x: (x["stat"], x["player"])):
                books = len(p.get("book_lines", {}))
                print(f"    {p['player']:22s} {p['stat'].upper():4s} {p['line']:5.1f}  ({books} books)")

"""Merge ParlayPlay + Underdog board JSONs into a single board.

For each player+stat combo that appears on both platforms, picks the line
that is better for UNDER bets (higher line = easier UNDER).
Keeps all props that only appear on one platform.

Usage:
    python3 merge_boards.py --pp board_pp.json --ud board_ud.json --out board_merged.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _normalize_name(name: str) -> str:
    """Normalize player name for matching across platforms."""
    # Lowercase, strip suffixes like Jr./III/II, collapse whitespace
    n = name.lower().strip()
    for suffix in [" jr.", " jr", " iii", " ii", " iv", " sr.", " sr"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    # Collapse multiple spaces
    n = " ".join(n.split())
    return n


def _normalize_stat(stat: str) -> str:
    """Normalize stat name for matching."""
    return stat.lower().strip().replace("-", "_")


def merge_boards(pp_props: list[dict], ud_props: list[dict]) -> list[dict]:
    """Merge two boards, picking the higher line (better for UNDER) when both have a prop.

    Args:
        pp_props: ParlayPlay board props.
        ud_props: Underdog board props.

    Returns:
        Merged list of props.
    """
    # Index ParlayPlay props by (normalized_name, stat)
    pp_index: dict[tuple[str, str], dict] = {}
    for p in pp_props:
        key = (_normalize_name(p["player"]), _normalize_stat(p["stat"]))
        pp_index[key] = p

    # Index Underdog props by (normalized_name, stat)
    ud_index: dict[tuple[str, str], dict] = {}
    for p in ud_props:
        key = (_normalize_name(p["player"]), _normalize_stat(p["stat"]))
        ud_index[key] = p

    merged = []
    used_keys = set()

    # Process all keys from both platforms
    all_keys = set(pp_index.keys()) | set(ud_index.keys())

    overlap_count = 0
    pp_only_count = 0
    ud_only_count = 0
    pp_picked = 0
    ud_picked = 0

    for key in sorted(all_keys):
        pp_prop = pp_index.get(key)
        ud_prop = ud_index.get(key)

        if pp_prop and ud_prop:
            # Both platforms have this prop -- pick higher line (better for UNDER)
            overlap_count += 1
            pp_line = pp_prop.get("line", 0)
            ud_line = ud_prop.get("line", 0)

            if ud_line >= pp_line:
                winner = dict(ud_prop)
                winner["source"] = "underdog"
                winner["alt_line"] = pp_line
                winner["alt_source"] = "parlayplay"
                ud_picked += 1
            else:
                winner = dict(pp_prop)
                winner["source"] = "parlayplay"
                winner["alt_line"] = ud_line
                winner["alt_source"] = "underdog"
                pp_picked += 1

            merged.append(winner)

        elif pp_prop:
            # Only on ParlayPlay
            pp_only_count += 1
            entry = dict(pp_prop)
            entry["source"] = "parlayplay"
            merged.append(entry)

        elif ud_prop:
            # Only on Underdog
            ud_only_count += 1
            entry = dict(ud_prop)
            entry["source"] = "underdog"
            merged.append(entry)

    # Print summary
    # Filter out line<=0 props (scraper artifacts)
    merged = [p for p in merged if p.get("line", 0) > 0]

    total = len(merged)
    print(f"\n=== Board Merge Summary ===")
    print(f"ParlayPlay props:  {len(pp_props)}")
    print(f"Underdog props:    {len(ud_props)}")
    print(f"Overlap (both):    {overlap_count}")
    print(f"  -> PP line picked: {pp_picked}")
    print(f"  -> UD line picked: {ud_picked}")
    print(f"PP only:           {pp_only_count}")
    print(f"UD only:           {ud_only_count}")
    print(f"Merged total:      {total}")
    print()

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge ParlayPlay + Underdog boards")
    parser.add_argument("--pp", required=True, help="ParlayPlay board JSON path")
    parser.add_argument("--ud", required=True, help="Underdog board JSON path")
    parser.add_argument("--out", required=True, help="Output merged board JSON path")
    args = parser.parse_args()

    pp_path = Path(args.pp)
    ud_path = Path(args.ud)
    out_path = Path(args.out)

    if not pp_path.exists():
        print(f"ERROR: ParlayPlay board not found: {pp_path}")
        sys.exit(1)
    if not ud_path.exists():
        print(f"ERROR: Underdog board not found: {ud_path}")
        sys.exit(1)

    pp_props = json.loads(pp_path.read_text())
    ud_props = json.loads(ud_path.read_text())

    merged = merge_boards(pp_props, ud_props)

    # Save -- strip source/alt fields for pipeline compatibility if desired,
    # but keep them for transparency
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"Saved merged board to {out_path}")


if __name__ == "__main__":
    main()

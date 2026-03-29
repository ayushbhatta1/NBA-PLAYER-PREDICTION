#!/usr/bin/env python3
"""
Parse raw sportsbook board text into structured prop lines.
Handles the format: Player Name, stat categories with lines and multipliers.
"""

import re
import json

def parse_board_tsv(raw_text):
    """Parse tab-delimited board: Player\\tGAME\\tstat\\tline\\tHigher [mult]\\tLower [mult]"""
    stat_map = {
        'pts': 'pts', 'reb': 'reb', 'ast': 'ast', '3pm': '3pm',
        'stl': 'stl', 'blk': 'blk', 'pra': 'pra', 'pr': 'pr', 'pa': 'pa', 'ra': 'ra',
    }
    props = []
    for line in raw_text.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        player = parts[0].strip()
        game_raw = parts[1].strip()
        stat = parts[2].strip().lower()
        if stat not in stat_map:
            continue
        try:
            line_val = float(parts[3].strip())
        except ValueError:
            continue

        # Extract team and normalize game key to AWAY@HOME
        m = re.match(r'^([A-Z]{2,4})\s*(@|vs)\s*([A-Z]{2,4})', game_raw)
        if m:
            team1, sep, team2 = m.group(1), m.group(2), m.group(3)
            if sep == 'vs':
                # "ATL vs ORL" = ATL is home → game = ORL@ATL, team = ATL
                team = team1
                game = f"{team2}@{team1}"
            else:
                # "ORL @ ATL" = ORL is away → game = ORL@ATL, team = ORL
                team = team1
                game = f"{team1}@{team2}"
        else:
            team = '?'
            game = game_raw

        # Parse multipliers from Higher/Lower columns
        mult = None
        for col in parts[4:]:
            mult_match = re.search(r'(\d+\.?\d*)x', col)
            if mult_match:
                m_val = float(mult_match.group(1))
                if mult is None or abs(m_val - 1.87) < abs(mult - 1.87):
                    mult = m_val

        props.append({
            'player': player, 'team': team, 'game': game,
            'stat': stat, 'line': line_val, 'multiplier': mult,
        })
    return props


def parse_board_web(raw_text):
    """Parse ParlayPlay web copy-paste format ('athlete or team avatar' markers)."""
    stat_map = {
        'Points': 'pts', 'Rebounds': 'reb', 'Assists': 'ast',
        '3PT Made': '3pm', '3-Pointers Made': '3pm',
        'Pts + Reb + Ast': 'pra', 'Pts + Rebs + Asts': 'pra',
        'Pts + Reb': 'pr', 'Points + Rebounds': 'pr',
        'Pts + Ast': 'pa', 'Points + Assists': 'pa',
        'Reb + Ast': 'ra', 'Rebounds + Assists': 'ra',
        'Steals': 'stl', 'Blocks': 'blk',
        'Steals + Blocks': 'stl_blk', 'Blocks + Steals': 'stl_blk',
    }
    skip_stats = {
        'Double Double', 'Double Doubles', 'Triple Double', 'Triple Doubles',
        'Fantasy Score', 'Fantasy Points', 'Turnovers', 'Free Throws Made',
        'FT Made', 'Personal Fouls', 'Minutes', 'Defensive Rebounds',
        'Offensive Rebounds', 'FG Made', 'FG Attempted', '3s Attempted',
        'First Point Scorer', 'Payout Boost Special*', 'Fantasy',
    }

    blocks = raw_text.split('athlete or team avatar')
    props = []

    for block in blocks[1:]:
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 3:
            continue

        player = lines[0]
        if player.startswith('$') or player in ('Champions', 'Drafts', 'Add picks'):
            continue

        # Find game info
        team = game = None
        game_idx = 0
        for idx, l in enumerate(lines[1:], 1):
            gm = re.match(r'^([A-Z]{2,4})\s*(@|vs)\s*([A-Z]{2,4})', l)
            if gm:
                game_idx = idx
                t1, sep, t2 = gm.group(1), gm.group(2), gm.group(3)
                if sep == 'vs':
                    team, game = t1, f"{t2}@{t1}"
                else:
                    team, game = t1, f"{t1}@{t2}"
                break

        if not game:
            continue

        # Find line value and stat
        line_val = stat_key = None
        mults = []

        for l in lines[game_idx + 1:]:
            if l in ('Higher', 'Lower', 'Add picks', 'Play', 'Standard', 'Flex',
                      'Enter amount', 'Rewards', 'Champions entry amount'):
                continue

            mult_m = re.match(r'^(\d+\.?\d*)x$', l)
            if mult_m:
                mults.append(float(mult_m.group(1)))
                continue

            if line_val is None:
                try:
                    line_val = float(l)
                    continue
                except ValueError:
                    pass

            if line_val is not None and stat_key is None:
                if l in skip_stats:
                    break
                if l in stat_map:
                    stat_key = stat_map[l]
                    continue

        if line_val is not None and stat_key is not None:
            mult = min(mults, key=lambda m: abs(m - 1.87)) if mults else None
            props.append({
                'player': player, 'team': team, 'game': game,
                'stat': stat_key, 'line': line_val, 'multiplier': mult,
            })

    return props


def parse_board(raw_text):
    """Parse raw board text into list of {player, team, game, stat, line, over_mult, under_mult}.
    Auto-detects web, tab-delimited, and ParlayPlay multi-line formats."""

    # Auto-detect: ParlayPlay web format (copy-paste from browser)
    if 'athlete or team avatar' in raw_text:
        props = parse_board_web(raw_text)
        return [p for p in props if p.get('line', 0) > 0]

    # Auto-detect: if first non-empty line has tabs and looks like TSV format, use TSV parser
    first_line = raw_text.strip().split('\n')[0]
    if '\t' in first_line and re.match(r'^[^@]+\t[A-Z]{2,4}\s*[@vs]', first_line):
        props = parse_board_tsv(raw_text)
        return [p for p in props if p.get('line', 0) > 0]

    lines = raw_text.strip().split('\n')

    props = []
    current_player = None
    current_team = None
    current_game = None
    current_stat = None

    # Stat name mapping
    stat_map = {
        'Points': 'pts',
        'Rebounds': 'reb',
        'Assists': 'ast',
        '3PT Made': '3pm',
        'Pts + Reb + Ast': 'pra',
        'Pts + Reb': 'pr',
        'Pts + Ast': 'pa',
        'Reb + Ast': 'ra',
        'Steals': 'stl',
        'Blocks': 'blk',
        'Steals + Blocks': 'stl_blk',
    }

    # Skip these stat types
    skip_stats = {'Double Double', 'First Point Scorer', 'Triple Double',
                  'Fantasy Score', 'Fantasy Points', 'Turnovers', 'Free Throws Made',
                  'Personal Fouls', 'Minutes', 'Defensive Rebounds',
                  'Offensive Rebounds', 'FG Made', 'Payout Boost Special*'}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and UI elements
        if not line or line in ['Less', 'More', 'Show more', 'expert-opinion']:
            i += 1
            continue

        # Skip "Show more X lines (N)" patterns
        if line.startswith('Show more') or line.startswith('Payout Boost'):
            i += 1
            continue

        # Skip timer patterns (HH:MM:SS)
        if re.match(r'^\d{2}:\d{2}:\d{2}$', line):
            i += 1
            continue

        # Detect player line: position - team pattern on next line
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            pos_team_match = re.match(r'^(PG|SG|SF|PF|C)\s*-\s*([A-Z]{2,3})$', next_line)
            if pos_team_match and not re.match(r'^\d', line):
                # Check if previous line has the full name (ParlayPlay format:
                # "Full Name" then "Abbreviated Name" then "POS - TEAM")
                if i > 0:
                    prev_line = lines[i - 1].strip()
                    # If current line is abbreviated (e.g. "C. Cunningham") and prev line
                    # looks like a full name (no dots/numbers), use prev line
                    if (re.match(r'^[A-Z]\.', line) and prev_line
                        and not re.match(r'^\d', prev_line)
                        and prev_line not in ['Less', 'More', '']
                        and not re.match(r'^(\d+\.?\d*)x$', prev_line)
                        and not any(prev_line == s for s in stat_map.keys())
                        and not any(prev_line == s for s in skip_stats)):
                        current_player = prev_line
                    else:
                        current_player = line
                else:
                    current_player = line
                current_team = pos_team_match.group(2)
                # Game info should be on line after that
                if i + 2 < len(lines):
                    game_line = lines[i + 2].strip()
                    game_match = re.match(r'^([A-Z]{2,3})\s*@\s*([A-Z]{2,3})', game_line)
                    if game_match:
                        current_game = f"{game_match.group(1)}@{game_match.group(2)}"
                        i += 3
                    else:
                        i += 2
                else:
                    i += 2
                continue

        # Detect stat category header
        stat_found = False
        for stat_name in sorted(stat_map.keys(), key=len, reverse=True):
            if line == stat_name:
                current_stat = stat_map[stat_name]
                stat_found = True
                break

        # Also check skip stats
        if not stat_found:
            for skip in skip_stats:
                if line == skip:
                    current_stat = None  # Reset to skip these lines
                    stat_found = True
                    break

        if stat_found:
            i += 1
            continue

        # Detect prop line: "N.5 StatName" pattern with multiplier
        if current_player and current_stat:
            # Match patterns like "14.5 Points", "2.5 Rebounds", etc.
            prop_match = re.match(r'^(\d+\.?\d*)\s+(.+)$', line)
            if prop_match:
                line_val = float(prop_match.group(1))
                stat_label = prop_match.group(2).strip()

                # Verify stat label matches current stat
                matched_stat = None
                for stat_name, stat_key in stat_map.items():
                    if stat_label == stat_name:
                        matched_stat = stat_key
                        break

                if matched_stat == current_stat or matched_stat is None:
                    # Look for multipliers after this line (up to 2)
                    mults_found = []
                    for offset in [1, 2]:
                        if i + offset < len(lines):
                            cleaned = re.sub(r'[^\dx.]', '', lines[i + offset].strip())
                            after_match = re.match(r'^(\d+\.?\d*)x$', cleaned)
                            if after_match:
                                mults_found.append(float(after_match.group(1)))
                            else:
                                break  # Stop if next line isn't a multiplier

                    # Pick multiplier closest to 1.87
                    mult = None
                    if mults_found:
                        mult = min(mults_found, key=lambda m: abs(m - 1.87))

                    props.append({
                        'player': current_player,
                        'team': current_team,
                        'game': current_game,
                        'stat': current_stat,
                        'line': line_val,
                        'multiplier': mult,
                    })

            # Also match standalone multiplier lines like "1.74x" that represent OVER/UNDER
            mult_only = re.match(r'^(\d+\.?\d*)x$', line)
            if mult_only:
                # This is a multiplier without a line number - skip
                pass

        i += 1

    return [p for p in props if p.get('line', 0) > 0]


def deduplicate_props(props):
    """Remove duplicate lines, keeping the one closest to the key line for each player/stat"""
    # Group by player + stat
    from collections import defaultdict
    groups = defaultdict(list)
    for p in props:
        key = (p['player'], p['stat'])
        groups[key].append(p)

    # For each group, pick the PRIMARY line (most common/middle line from sportsbook)
    # Usually the one with multiplier closest to 1.85-1.95 (standard -110 juice)
    deduped = []
    for key, group in groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Pick the line with multiplier closest to 1.87 (standard juice)
            best = None
            best_dist = float('inf')
            for p in group:
                if p['multiplier']:
                    dist = abs(p['multiplier'] - 1.87)
                    if dist < best_dist:
                        best_dist = dist
                        best = p
            if best is None:
                # No multipliers, pick middle line
                group.sort(key=lambda x: x['line'])
                best = group[len(group) // 2]
            deduped.append(best)

    return deduped


if __name__ == '__main__':
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/full_board_raw.txt'
    output_file = sys.argv[2] if len(sys.argv) > 2 else '/tmp/parsed_board.json'

    with open(input_file, 'r') as f:
        raw = f.read()

    props = parse_board(raw)
    print(f"Total raw prop lines extracted: {len(props)}")

    # v4: Validation - warn if extraction rate seems low
    raw_line_count = len([l for l in raw.split('\n') if l.strip()])
    extraction_rate = len(props) / raw_line_count * 100 if raw_line_count > 0 else 0
    print(f"Raw input lines: {raw_line_count} | Extraction rate: {extraction_rate:.1f}%")
    if extraction_rate < 5:
        print("[WARN] Extraction rate very low (<5%). Board format may not match parser expectations.")
    elif extraction_rate < 15:
        print("[WARN] Extraction rate low (<15%). Some lines may have been missed.")

    # Filter out line<=0 props (scraper artifacts — no real sportsbook line)
    props = [p for p in props if p.get('line', 0) > 0]
    print(f"After filtering line<=0 artifacts: {len(props)}")

    deduped = deduplicate_props(props)
    print(f"After deduplication (1 line per player/stat): {len(deduped)}")

    # Validation: check for missing fields
    missing_game = sum(1 for p in deduped if not p.get('game'))
    missing_team = sum(1 for p in deduped if not p.get('team'))
    if missing_game > 0:
        print(f"[WARN] {missing_game} picks missing game context")
    if missing_team > 0:
        print(f"[WARN] {missing_team} picks missing team assignment")

    # Count unique players
    players = set(p['player'] for p in deduped)
    print(f"Unique players: {len(players)}")

    # Count by stat
    from collections import Counter
    stat_counts = Counter(p['stat'] for p in deduped)
    print(f"By stat: {dict(stat_counts)}")

    # Count by game
    game_counts = Counter(p['game'] for p in deduped)
    print(f"By game: {dict(game_counts)}")

    # Save parsed board
    with open(output_file, 'w') as f:
        json.dump(deduped, f, indent=2)

    print(f"\nSaved to {output_file}")

    # Show first 5
    for p in deduped[:5]:
        print(f"  {p['player']} | {p['stat']} | {p['line']} | {p['multiplier']}x | {p['game']}")

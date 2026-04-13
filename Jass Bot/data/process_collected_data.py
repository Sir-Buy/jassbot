"""
Process collected Swisslos game data into clean training-ready format.

Reads raw JSON exports from the collector, validates, deduplicates, merges
partial matches, and outputs:
  1. cleaned_games.json  — Clean merged data for analysis
  2. training_rounds.json — Flat list of complete rounds with all fields needed
                            for neural network feature encoding
  3. summary report to stdout

Usage:
    python process_collected_data.py
    python process_collected_data.py --input-dir "Export Games" --output-dir "Processed"
"""

import json
import os
import sys
import glob
from collections import defaultdict
from pathlib import Path

# Card system (must match engine/cards.py)
SUIT_ID = {'H': 0, 'D': 1, 'S': 2, 'C': 3}
SUIT_NAME = {0: 'herz', 1: 'ecke', 2: 'schaufel', 3: 'kreuz'}
VAL_ID = {'6': 0, '7': 1, '8': 2, '9': 3, '10': 4, 'J': 5, 'Q': 6, 'K': 7, 'A': 8}
BASE_PTS = [0, 0, 0, 0, 10, 2, 3, 4, 11]
TRUMP_PTS = [0, 0, 0, 14, 10, 20, 3, 4, 11]

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "Export Games"
DEFAULT_OUTPUT = SCRIPT_DIR / "Processed"

# Bot detection patterns
BOT_NAME_PATTERNS = ['Computer', 'Bot', 'Compi']
BOT_UID_RANGE = range(-100, 0)  # Swisslos bots have negative UIDs like -1, -2, -3


def classify_match(match):
    """Classify match as 'bot_training', 'human', or 'mixed'."""
    players = match.get('players', [])
    if not players:
        # Check match_id pattern: training games use "T-SP-D-" prefix
        mid = match.get('match_id', '')
        if mid.startswith('T-SP-'):
            return 'bot_training'
        return 'unknown'

    bot_count = 0
    human_count = 0
    for p in players:
        name = p.get('name', '') if isinstance(p, dict) else str(p)
        uid = p.get('uid', None) if isinstance(p, dict) else None

        is_bot = False
        # Check name patterns
        for pat in BOT_NAME_PATTERNS:
            if pat.lower() in name.lower():
                is_bot = True
                break
        # Check negative UIDs (Swisslos bot convention)
        if uid is not None and isinstance(uid, int) and uid < 0:
            is_bot = True

        if is_bot:
            bot_count += 1
        else:
            human_count += 1

    if bot_count > 0 and human_count <= 1:
        return 'bot_training'
    elif bot_count == 0:
        return 'human'
    else:
        return 'mixed'


def load_all_exports(input_dir):
    """Load all JSON export files from directory."""
    all_matches = []
    files = sorted(glob.glob(str(input_dir / "jass_data_*.json")) + glob.glob(str(input_dir / "round_*.json")))
    print(f"Found {len(files)} export file(s) in {input_dir}")

    for fpath in files:
        with open(fpath, encoding='utf-8') as f:
            data = json.load(f)
        # Handle both formats: {matches: [...]} (extension export) and raw match objects (server save)
        if 'matches' in data:
            matches = data['matches']
        elif 'match_id' in data:
            # Raw match object from server /save endpoint
            matches = [data]
        else:
            matches = []
        print(f"  {os.path.basename(fpath)}: {len(matches)} matches")
        all_matches.extend(matches)

    return all_matches


def merge_partial_matches(matches):
    """Merge matches with the same match_id (split by page reload/rejoin)."""
    by_id = defaultdict(list)
    for m in matches:
        mid = m.get('match_id', 'unknown')
        by_id[mid].append(m)

    merged = []
    merge_count = 0

    for mid, parts in by_id.items():
        if len(parts) == 1:
            merged.append(parts[0])
            continue

        # Merge: combine rounds from all parts, deduplicate by round_number
        base = parts[0].copy()
        seen_rounds = set()
        all_rounds = []

        for part in parts:
            for r in part.get('rounds', []):
                rn = r.get('round_number', 0)
                # Deduplicate by round content (same round_number + same trump)
                key = (rn, r.get('trump', ''))
                if key not in seen_rounds:
                    seen_rounds.add(key)
                    all_rounds.append(r)

        all_rounds.sort(key=lambda r: r.get('round_number', 0))
        base['rounds'] = all_rounds
        base['max_rounds'] = max(p.get('max_rounds', 4) for p in parts)
        base['complete'] = len(all_rounds) >= base['max_rounds']

        # Recompute standings — use last round's cumulative 'total' (already cumulative)
        standings = {}
        last_r = all_rounds[-1] if all_rounds else None
        if last_r and last_r.get('results') and last_r['results'].get('results'):
            for res in last_r['results']['results']:
                pos = str(res['pos'])
                standings[pos] = {
                    'total_deviation': abs(res.get('total', 0)),
                    'rounds_played': len(all_rounds),
                    'last_rank': res.get('rank'),
                }
        base['final_standings'] = standings

        merged.append(base)
        merge_count += 1

    if merge_count:
        print(f"Merged {merge_count} partial match(es)")

    return merged


def validate_round(r):
    """Validate a round's data integrity. Returns list of issues."""
    issues = []
    tricks = r.get('tricks', [])

    if len(tricks) == 0:
        issues.append("no tricks")
        return issues

    # Check card count
    all_cards = []
    all_card_ids = set()
    for t in tricks:
        for c in t.get('cards', []):
            all_cards.append(c)
            if c.get('card_id') is not None:
                all_card_ids.add(c['card_id'])

    if len(tricks) == 9:
        if len(all_cards) != 36:
            issues.append(f"{len(all_cards)} cards (expected 36)")
        if len(all_card_ids) != 36:
            issues.append(f"{len(all_card_ids)} unique card IDs (expected 36)")

    # Check total points
    total = r.get('total_points', 0)
    if len(tricks) == 9 and total != 157:
        issues.append(f"total_points={total} (expected 157)")

    # Check results exist
    if not r.get('results') or not r['results'].get('results'):
        issues.append("no results")

    # Verify computed vs server points
    if r.get('results') and r['results'].get('results') and r.get('player_points'):
        pp = r['player_points']
        for res in r['results']['results']:
            pos = str(res['pos'])
            srv = res.get('points', 0)
            comp = pp.get(pos, pp.get(res['pos'], -1))
            if comp != srv:
                issues.append(f"seat {pos}: computed={comp} server={srv}")

    return issues


def extract_training_round(r, match_meta):
    """Extract a clean training-ready round from raw data."""
    trump = r.get('trump', '')
    trump_id = SUIT_ID.get(trump)
    if trump_id is None:
        return None

    tricks = r.get('tricks', [])
    if len(tricks) != 9:
        return None

    # Verify 36 cards
    card_ids = set()
    for t in tricks:
        for c in t.get('cards', []):
            if c.get('card_id') is not None:
                card_ids.add(c['card_id'])
    if len(card_ids) != 36:
        return None

    # Must have results
    results = r.get('results', {})
    if not results or not results.get('results'):
        return None

    # Build declarations dict
    declarations = {}
    for res in results['results']:
        declarations[res['pos']] = res.get('callP', 0)

    # Build player results
    # NOTE: server 'total' field is CUMULATIVE across the match, not per-round!
    # Per-round deviation = |callP - points|
    player_results = {}
    for res in results['results']:
        callP = res.get('callP', 0)
        points = res.get('points', 0)
        player_results[res['pos']] = {
            'declared': callP,
            'scored': points,
            'deviation': abs(callP - points),            # per-round deviation
            'cumulative_deviation': res.get('total', 0),  # running match total from server
            'rank': res.get('rank', 0),
        }

    # Build clean trick list
    clean_tricks = []
    for t in tricks:
        cards = []
        for c in t.get('cards', []):
            cards.append({
                'player': c['player'],
                'card_id': c['card_id'],
                'card_code': c.get('card', ''),
                'points': c.get('points', 0),
                'is_trump': c.get('is_trump', False),
            })
        clean_tricks.append({
            'trick_number': t.get('trick_number', 0),
            'leader': t.get('leader'),
            'cards': cards,
            'winner': t.get('winner'),
            'points': t.get('points', 0),
        })

    # Reconstruct each player's hand from the tricks
    hands = {0: [], 1: [], 2: [], 3: []}
    for t in clean_tricks:
        for c in t['cards']:
            hands[c['player']].append(c['card_id'])

    # Identify our bot's seat (Gast_* with uid=None, or non-Computer player in bot games)
    players = match_meta.get('players', [])
    our_seat = None
    for j, p in enumerate(players):
        name = p.get('name', '') if isinstance(p, dict) else str(p)
        uid = p.get('uid', 'missing') if isinstance(p, dict) else 'missing'
        if 'Gast' in name or uid is None:
            our_seat = j
            break

    return {
        'trump': trump_id,
        'trump_code': trump,
        'declarations': declarations,
        'hands': hands,
        'tricks': clean_tricks,
        'results': player_results,
        'total_points': r.get('total_points', 157),
        # Match context
        'match_id': match_meta.get('match_id'),
        'room': match_meta.get('room', {}).get('title'),
        'stake': match_meta.get('room', {}).get('stake'),
        'prize_pool': match_meta.get('room', {}).get('prize_pool'),
        'players': [p.get('name', '?') for p in match_meta.get('players', [])],
        'player_uids': [p.get('uid') for p in match_meta.get('players', [])],
        'our_seat': our_seat,  # Swisslos seat of our bot (None for human-only games)
        'round_number': r.get('round_number', 0),
        'max_rounds': match_meta.get('max_rounds', 4),
        'game_type': classify_match(match_meta),  # 'bot_training', 'human', or 'mixed'
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process Swisslos Jass collected data")
    parser.add_argument('--input-dir', default=str(DEFAULT_INPUT))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    matches = load_all_exports(input_dir)
    print(f"\nLoaded {len(matches)} raw matches")

    # Merge partial
    matches = merge_partial_matches(matches)
    print(f"After merge: {len(matches)} matches")

    # Validate and extract
    valid_rounds = 0
    invalid_rounds = 0
    training_rounds = []
    clean_matches = []

    for m in matches:
        game_type = classify_match(m)
        clean_m = {
            'match_id': m.get('match_id'),
            'game_type': game_type,
            'room': m.get('room'),
            'players': m.get('players'),
            'max_rounds': m.get('max_rounds', 4),
            'complete': m.get('complete', False),
            'final_standings': m.get('final_standings'),
            'rounds': [],
        }

        for r in m.get('rounds', []):
            issues = validate_round(r)
            if issues:
                print(f"  SKIP {m.get('match_id')} R{r.get('round_number')}: {', '.join(issues)}")
                invalid_rounds += 1
                continue

            valid_rounds += 1
            clean_m['rounds'].append(r)

            # Extract training format
            tr = extract_training_round(r, m)
            if tr:
                training_rounds.append(tr)

        if clean_m['rounds']:
            clean_matches.append(clean_m)

    # Separate by game type
    human_rounds = [r for r in training_rounds if r.get('game_type') == 'human']
    bot_rounds = [r for r in training_rounds if r.get('game_type') == 'bot_training']
    other_rounds = [r for r in training_rounds if r.get('game_type') not in ('human', 'bot_training')]

    human_matches = [m for m in clean_matches if m.get('game_type') == 'human']
    bot_matches = [m for m in clean_matches if m.get('game_type') == 'bot_training']

    card_encoding = {
        'formula': 'card_id = suit * 9 + value',
        'suits': {'herz': 0, 'ecke': 1, 'schaufel': 2, 'kreuz': 3},
        'values': {'6': 0, '7': 1, '8': 2, '9': 3, '10': 4, 'U': 5, 'O': 6, 'K': 7, 'A': 8},
        'base_points': BASE_PTS,
        'trump_points': TRUMP_PTS,
    }

    # Save all data
    def save_rounds_file(path, rounds, label):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'total_rounds': len(rounds), 'card_encoding': card_encoding, 'rounds': rounds},
                      f, indent=2, ensure_ascii=False)
        print(f"  {path} ({len(rounds)} rounds)")

    def save_matches_file(path, matches, label):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'total_matches': len(matches), 'total_rounds': sum(len(m['rounds']) for m in matches),
                        'matches': matches}, f, indent=2, ensure_ascii=False)
        print(f"  {path} ({len(matches)} matches)")

    print(f"\n{'='*60}")
    print(f"SAVING")
    print(f"{'='*60}")

    # All data combined
    save_matches_file(output_dir / "cleaned_games.json", clean_matches, "all")
    save_rounds_file(output_dir / "training_rounds.json", training_rounds, "all")

    # Human only
    if human_rounds:
        save_rounds_file(output_dir / "training_rounds_human.json", human_rounds, "human")
        save_matches_file(output_dir / "cleaned_games_human.json", human_matches, "human")

    # Bot training only
    if bot_rounds:
        save_rounds_file(output_dir / "training_rounds_bot.json", bot_rounds, "bot")
        save_matches_file(output_dir / "cleaned_games_bot.json", bot_matches, "bot")

    # Summary
    def print_stats(label, rounds_list, matches_list):
        if not rounds_list:
            print(f"  (no data)")
            return
        round_devs = []
        decl_dist = []
        perfect = 0
        total_pr = 0
        for tr in rounds_list:
            for pos, res in tr['results'].items():
                round_devs.append(res['deviation'])
                decl_dist.append(res['declared'])
                total_pr += 1
                if res['deviation'] == 0:
                    perfect += 1

        match_devs = []
        match_winners = []
        for m in matches_list:
            last_r = m.get('rounds', [])[-1] if m.get('rounds') else None
            if last_r and last_r.get('results') and last_r['results'].get('results'):
                ptd = {}
                for res in last_r['results']['results']:
                    ptd[res['pos']] = abs(res.get('total', 0))
                if ptd:
                    match_devs.extend(ptd.values())
                    match_winners.append(min(ptd.values()))

        avg_dev = sum(round_devs) / len(round_devs) if round_devs else 0
        avg_decl = sum(decl_dist) / len(decl_dist) if decl_dist else 0
        avg_match = sum(match_devs) / len(match_devs) if match_devs else 0
        avg_win = sum(match_winners) / len(match_winners) if match_winners else 0

        print(f"  Matches: {len(matches_list)}  Rounds: {len(rounds_list)}  Complete: {sum(1 for m in matches_list if m.get('complete'))}")
        print(f"  Avg round dev: {avg_dev:.1f}  Perfect: {perfect}/{total_pr} ({100*perfect/max(total_pr,1):.0f}%)  Avg decl: {avg_decl:.1f}")
        if match_devs:
            print(f"  Avg match dev: {avg_match:.1f}  Winner: {avg_win:.1f}  Best: {min(match_devs)}  Worst: {max(match_devs)}")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total: {len(clean_matches)} matches, {valid_rounds} valid rounds, {invalid_rounds} skipped")

    trump_dist = [0, 0, 0, 0]
    for tr in training_rounds:
        trump_dist[tr['trump']] += 1
    print(f"Trump: H={trump_dist[0]} D={trump_dist[1]} S={trump_dist[2]} C={trump_dist[3]}")
    print()

    print(f"--- HUMAN GAMES ---")
    print_stats("human", human_rounds, human_matches)
    print()
    print(f"--- BOT TRAINING ---")
    print_stats("bot", bot_rounds, bot_matches)
    if other_rounds:
        print()
        print(f"--- OTHER/MIXED ({len(other_rounds)} rounds) ---")
        print_stats("other", other_rounds, [m for m in clean_matches if m.get('game_type') not in ('human', 'bot_training')])


if __name__ == '__main__':
    main()

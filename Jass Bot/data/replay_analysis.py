"""
Replay Analysis: Run the bot on real Swisslos hands.

For each collected round, seats the bot in every position and compares:
  - Bot declaration vs human declaration
  - Bot card choices vs human card choices (trick by trick)
  - Bot final deviation vs human final deviation

Usage:
    python replay_analysis.py
"""

import json
import sys
import os

# Add Jass Bot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Jass Bot'))

from engine.cards import (
    card_name, card_short, CARD_SUIT, points as card_points,
    TOTAL_ROUND_POINTS, LAST_TRICK_BONUS, NUM_VALUES,
)
from engine.game_state import GameState
from engine.rules import legal_moves, trick_winner, trick_points
from bot.strategy import StrategyEngine
from bot.declaration import DeclarationModel
from bot.belief_tracker import BeliefTracker

# Swisslos card code -> bot card ID
SWISSLOS_SUIT = {'H': 0, 'D': 1, 'S': 2, 'C': 3}
SWISSLOS_VAL = {'6': 0, '7': 1, '8': 2, '9': 3, '10': 4, 'J': 5, 'Q': 6, 'K': 7, 'A': 8}

def swisslos_to_id(code):
    """Convert 'H6', 'DJ', 'SA' etc to card ID 0-35."""
    s = SWISSLOS_SUIT[code[0]]
    v = SWISSLOS_VAL[code[1:]]
    return s * NUM_VALUES + v


def load_training_rounds():
    path = os.path.join(os.path.dirname(__file__), 'Processed', 'training_rounds.json')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data['rounds']


def create_bot(player_id=0):
    """Create the best PIMC bot configuration."""
    engine = StrategyEngine(
        num_worlds=200,
        ansage_worlds=200,
        use_target_aware_opponents=True,
        endgame_depth=4,
    )
    decl_model = DeclarationModel(use_target_aware_opponents=True)
    return engine, decl_model


def replay_round(round_data, verbose=True):
    """Replay a single round: bot vs human in each seat."""
    trump = round_data['trump']
    hands = {int(k): v for k, v in round_data['hands'].items()}
    tricks_data = round_data['tricks']
    human_decls = round_data['declarations']
    human_results = round_data['results']
    players = round_data.get('players', ['P0', 'P1', 'P2', 'P3'])

    results = []

    for bot_seat in range(4):
        bot_hand = list(hands[bot_seat])
        engine, decl_model = create_bot(bot_seat)

        # --- Bot declaration ---
        engine.reset()
        bot_decl = decl_model.declare(bot_hand, trump, voids={1: set(), 2: set(), 3: set()}, num_worlds=200)

        human_decl = int(human_decls.get(str(bot_seat), human_decls.get(bot_seat, 0)))

        # --- Replay tricks with bot playing in bot_seat ---
        # Build game state incrementally, let bot choose cards
        state = GameState(
            hands={p: list(h) for p, h in hands.items()},
            trump=trump,
            trick_number=1,
            leader=tricks_data[0]['leader'] if tricks_data else 0,
        )
        state.declarations = {p: int(human_decls.get(str(p), human_decls.get(p, 0))) for p in range(4)}
        state.declarations[bot_seat] = bot_decl

        bot_points = 0
        bot_cards_played = []
        human_cards_played = []
        agreements = 0
        belief_tracker = BeliefTracker()
        belief_tracker.init_round(trump)
        trick_number = 0

        for t_idx, trick_data in enumerate(tricks_data):
            state.trick_number = t_idx + 1
            state.current_trick = []
            is_last = (t_idx == len(tricks_data) - 1)
            leader = trick_data['leader']
            state.leader = leader

            trick_cards = []

            for c_idx, card_entry in enumerate(trick_data['cards']):
                pid = card_entry['player']
                human_card_code = card_entry.get('card_code') or card_entry.get('card', '')
                if human_card_code and human_card_code[0] in SWISSLOS_SUIT:
                    human_card = swisslos_to_id(human_card_code)
                else:
                    human_card = card_entry['card_id']

                if pid == bot_seat:
                    # Bot decides
                    hand = state.hands[bot_seat]
                    legal = legal_moves(hand, state.lead_suit, trump, state.current_trick if state.current_trick else None)
                    if not legal:
                        legal = list(hand)

                    if len(legal) == 1:
                        bot_card = legal[0]
                    else:
                        played = list(state.played_cards)
                        table = state.cards_on_table()

                        bot_card = engine.zug(
                            hand=list(hand),
                            trump=trump,
                            target=bot_decl,
                            current_points=bot_points,
                            table_cards=table,
                            leader=leader,
                            played_cards=played,
                            trick_number=trick_number,
                            belief_tracker=belief_tracker,
                            opp_points=list(state.points),
                        )
                        if bot_card is None or bot_card not in legal:
                            bot_card = legal[0]

                    bot_cards_played.append(bot_card)
                    human_cards_played.append(human_card)
                    if bot_card == human_card:
                        agreements += 1

                    # Use the HUMAN card to continue the actual game replay
                    # (so other positions see the real game state)
                    actual_card = human_card
                else:
                    actual_card = human_card

                # Detect voids
                if state.current_trick:
                    lead_s = CARD_SUIT[state.current_trick[0][1]]
                    if CARD_SUIT[actual_card] != lead_s:
                        state.voids[pid].add(lead_s)
                        engine.record_void(pid, lead_s)

                state.current_trick.append((pid, actual_card))
                state.hands[pid].remove(actual_card)
                trick_cards.append((pid, actual_card))

            # Resolve trick
            winner = trick_winner(state.current_trick, CARD_SUIT[state.current_trick[0][1]], trump)
            pts = trick_points(state.current_trick, trump)
            if is_last:
                pts += LAST_TRICK_BONUS
            state.points[winner] += pts

            if winner == bot_seat:
                bot_points += pts

            # Update beliefs
            lead_suit = CARD_SUIT[trick_cards[0][1]]
            cards_before = []
            for player, card in trick_cards:
                if player != bot_seat and player in [1, 2, 3]:
                    belief_tracker.update(
                        player=player, action=card,
                        player_current_points=state.points[player] - (pts if winner == player else 0),
                        trick_number=trick_number,
                        table_cards_before=list(cards_before),
                        lead_suit=lead_suit if cards_before else None,
                        trump=trump,
                    )
                cards_before.append((player, card))

            for _, cid in state.current_trick:
                state.played_cards.add(cid)
            state.leader = winner
            trick_number += 1

        # Compute bot deviation using human's actual points
        # (bot played in human's game, receiving same tricks)
        human_scored = int(human_results[str(bot_seat)]['scored'])
        human_dev = int(human_results[str(bot_seat)]['deviation'])
        bot_dev = abs(bot_decl - human_scored)

        n_cards = len(bot_cards_played)
        agree_pct = (agreements / n_cards * 100) if n_cards > 0 else 0

        results.append({
            'seat': bot_seat,
            'player': players[bot_seat] if bot_seat < len(players) else f'P{bot_seat}',
            'human_decl': human_decl,
            'bot_decl': bot_decl,
            'scored': human_scored,
            'human_dev': human_dev,
            'bot_dev': bot_dev,
            'improvement': human_dev - bot_dev,
            'card_agreements': agreements,
            'total_cards': n_cards,
            'agree_pct': agree_pct,
        })

    return results


def main():
    rounds = load_training_rounds()
    print(f"Loaded {len(rounds)} rounds for replay analysis")
    print(f"Bot config: PIMC 200w + target-aware + beliefs + endgame depth 4")
    print()

    all_results = []
    total_human_dev = 0
    total_bot_dev = 0
    total_improvements = 0
    total_agreements = 0
    total_cards = 0

    for ri, r in enumerate(rounds):
        trump_names = ['Herz', 'Ecke', 'Schaufel', 'Kreuz']
        trump_sym = ['H', 'D', 'S', 'C']
        t = r['trump']
        players = r.get('players', ['P0', 'P1', 'P2', 'P3'])

        print(f"{'='*70}")
        print(f"Round {ri+1}: Trump={trump_names[t]}  Room={r.get('room', '?')}  R{r.get('round_number', '?')}/{r.get('max_rounds', 4)}")
        print(f"Players: {', '.join(players)}")
        print(f"{'='*70}")

        results = replay_round(r)
        all_results.extend(results)

        print(f"{'Seat':<6} {'Player':<14} {'H.Decl':>6} {'B.Decl':>6} {'Scored':>6} {'H.Dev':>5} {'B.Dev':>5} {'Impr':>5} {'Cards':>6}")
        print(f"{'-'*60}")

        for res in results:
            impr = res['improvement']
            impr_str = f"+{impr}" if impr > 0 else str(impr)
            print(f"{res['seat']:<6} {res['player']:<14} {res['human_decl']:>6} {res['bot_decl']:>6} "
                  f"{res['scored']:>6} {res['human_dev']:>5} {res['bot_dev']:>5} {impr_str:>5} "
                  f"{res['card_agreements']}/{res['total_cards']}")

            total_human_dev += res['human_dev']
            total_bot_dev += res['bot_dev']
            total_improvements += res['improvement']
            total_agreements += res['card_agreements']
            total_cards += res['total_cards']

        print()

    n = len(all_results)
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY ({len(rounds)} rounds, {n} player-seats)")
    print(f"{'='*70}")
    print(f"Human avg deviation:  {total_human_dev / n:.1f}")
    print(f"Bot avg deviation:    {total_bot_dev / n:.1f}")
    print(f"Avg improvement:      {total_improvements / n:+.1f} per round")
    print(f"Bot better:           {sum(1 for r in all_results if r['improvement'] > 0)}/{n} seats")
    print(f"Bot same:             {sum(1 for r in all_results if r['improvement'] == 0)}/{n} seats")
    print(f"Bot worse:            {sum(1 for r in all_results if r['improvement'] < 0)}/{n} seats")
    print(f"Card agreement:       {total_agreements}/{total_cards} ({100*total_agreements/max(total_cards,1):.0f}%)")

    # Best/worst improvements
    best = max(all_results, key=lambda r: r['improvement'])
    worst = min(all_results, key=lambda r: r['improvement'])
    print(f"\nBest improvement:     seat {best['seat']} ({best['player']}): human={best['human_dev']} bot={best['bot_dev']} (+{best['improvement']})")
    print(f"Worst regression:     seat {worst['seat']} ({worst['player']}): human={worst['human_dev']} bot={worst['bot_dev']} ({worst['improvement']})")


if __name__ == '__main__':
    main()

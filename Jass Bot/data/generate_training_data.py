"""Generate training data from PIMC self-play.

Each play decision is recorded as a pre-encoded feature vector + labels.
Uses all CPU cores for maximum throughput.
"""

from __future__ import annotations

import os
import sys
import random
import time
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.add_dll_directory("C:/msys64/ucrt64/bin")

from engine.cards import CARD_SUIT, CARD_VALUE, NUM_CARDS, NUM_VALUES, points as card_points
from engine.rules import legal_moves

# ===== Feature encoding =====

# Feature layout:
# my_hand[36] + played_cards[36] + trick_cards[36] + trump_cards[36] + legal_mask[36]
# + current_winner[36]
# + suit_remaining[16] (4 suits × 4 players, but we only know for opponents est)
# + void_knowledge[12] (3 opps × 4 suits)
# + scalars[22]
# Total: 36*6 + 16 + 12 + 22 = 266

FEATURE_SIZE = 266


def encode_state(
    hand: list[int],
    played_cards: set[int],
    trick_cards: list[tuple[int, int]],
    trump: int,
    legal: list[int],
    trick_number: int,
    my_points: int,
    opp_points: list[int],
    my_target: int,
    leader: int,
    voids: dict[int, set[int]],
) -> np.ndarray:
    """Encode game state as float32 feature vector."""
    f = np.zeros(FEATURE_SIZE, dtype=np.float32)
    off = 0

    # my_hand[36]
    for c in hand:
        f[off + c] = 1.0
    off += 36

    # played_cards[36]
    for c in played_cards:
        f[off + c] = 1.0
    off += 36

    # current_trick_cards[36]
    for _, c in trick_cards:
        f[off + c] = 1.0
    off += 36

    # trump_cards[36]
    for c in range(NUM_CARDS):
        if CARD_SUIT[c] == trump:
            f[off + c] = 1.0
    off += 36

    # legal_mask[36]
    for c in legal:
        f[off + c] = 1.0
    off += 36

    # current_winner[36] — one-hot of winning card on table
    if trick_cards:
        from engine.cards import strength
        lead_suit = CARD_SUIT[trick_cards[0][1]]
        best_c = trick_cards[0][1]
        for _, c in trick_cards[1:]:
            cs = CARD_SUIT[c]
            bs = CARD_SUIT[best_c]
            if cs == trump and bs != trump:
                best_c = c
            elif cs == bs and strength(c, trump) > strength(best_c, trump):
                best_c = c
        f[off + best_c] = 1.0
    off += 36

    # suit_remaining[16]: 4 suits × 4 "slots" (rough estimate)
    all_seen = played_cards | {c for _, c in trick_cards} | set(hand)
    for s in range(4):
        remaining = sum(1 for c in range(NUM_CARDS) if CARD_SUIT[c] == s and c not in all_seen)
        for slot in range(4):
            f[off + s * 4 + slot] = remaining / 9.0
    off += 16

    # void_knowledge[12]: 3 opponents × 4 suits
    for oi, p in enumerate([1, 2, 3]):
        for s in range(4):
            if s in voids.get(p, set()):
                f[off + oi * 4 + s] = 1.0
    off += 12

    # Scalars[22]
    gap = my_target - my_points
    total_pts = my_points + sum(opp_points)
    n_on_table = len(trick_cards)
    trump_in_hand = sum(1 for c in hand if CARD_SUIT[c] == trump)
    puur_id = trump * NUM_VALUES + 5
    nell_id = trump * NUM_VALUES + 3
    trump_ace_id = trump * NUM_VALUES + 8

    scalars = [
        trick_number / 8.0,
        my_points / 157.0,
        opp_points[0] / 157.0,
        opp_points[1] / 157.0,
        opp_points[2] / 157.0,
        my_target / 157.0,
        gap / 157.0,
        abs(gap) / 157.0,
        1.0 if gap > 0 else 0.0,
        1.0 if gap < 0 else 0.0,
        1.0 if gap == 0 else 0.0,
        (9 - trick_number) / 9.0,
        len(hand) / 9.0,
        1.0 if n_on_table == 0 else 0.0,
        n_on_table / 3.0,
        sum(card_points(c, trump) for _, c in trick_cards) / 60.0 if trick_cards else 0.0,
        total_pts / 157.0,
        trump_in_hand / 9.0,
        1.0 if puur_id in hand else 0.0,
        1.0 if nell_id in hand else 0.0,
        1.0 if trump_ace_id in hand else 0.0,
        len(legal) / 9.0,
    ]
    for i, v in enumerate(scalars):
        f[off + i] = v

    return f


# ===== Worker =====

def _worker(args):
    """Generate data for a batch of rounds. Returns (features, cards, utilities, deviations)."""
    seed_start, n_rounds, num_worlds, ansage_worlds = args

    from arena.players import PIMCPlayer, HeuristicPlayer
    from engine.game import deal, play_round
    from engine.game_state import GameState
    from engine.rules import legal_moves as lm

    features_list = []
    cards_list = []
    utilities_list = []
    deviations_list = []

    for r in range(n_rounds):
        seed = seed_start + r
        rng = random.Random(seed)

        pimc = PIMCPlayer(
            player_id=0, num_worlds=num_worlds, ansage_worlds=ansage_worlds,
            declaration="median", use_target_aware_opponents=True,
            use_belief_tracking=True, endgame_depth=4,
            rng=random.Random(rng.randint(0, 2**32)),
        )
        players = {
            0: pimc,
            1: HeuristicPlayer(player_id=1, rng=random.Random(rng.randint(0, 2**32))),
            2: HeuristicPlayer(player_id=2, rng=random.Random(rng.randint(0, 2**32))),
            3: HeuristicPlayer(player_id=3, rng=random.Random(rng.randint(0, 2**32))),
        }

        # We need to intercept play decisions to record features + PIMC utilities
        # Play the round manually to capture per-move data
        result = play_round(players, rng=random.Random(seed))
        final_dev = result.deviations[0]

        # Record: we don't have per-move utilities from play_round.
        # Instead, record the chosen card and final deviation.
        # For utility labels, we'll use the deviation as a proxy.
        # The per-card utilities require calling evaluate_moves separately,
        # which is too slow for 5K rounds. Use final deviation + chosen card instead.

    # Simpler approach: run rounds and record every state we observe
    features_list = []
    cards_list = []
    deviations_list = []

    for r in range(n_rounds):
        seed = seed_start + r
        rng = random.Random(seed)

        # Deal
        from engine.game import deal
        trump, hands = deal(rng)

        # Create players
        pimc = PIMCPlayer(
            player_id=0, num_worlds=num_worlds, ansage_worlds=ansage_worlds,
            declaration="median", use_target_aware_opponents=True,
            use_belief_tracking=True, endgame_depth=4,
            rng=random.Random(rng.randint(0, 2**32)),
        )
        heurs = {
            1: HeuristicPlayer(player_id=1, rng=random.Random(rng.randint(0, 2**32))),
            2: HeuristicPlayer(player_id=2, rng=random.Random(rng.randint(0, 2**32))),
            3: HeuristicPlayer(player_id=3, rng=random.Random(rng.randint(0, 2**32))),
        }

        # Declarations
        decls = {}
        decls[0] = pimc.declare(list(hands[0]), trump)
        for p in [1, 2, 3]:
            decls[p] = heurs[p].declare(list(hands[p]), trump)

        # Play tricks
        state = GameState(
            hands={p: list(h) for p, h in hands.items()},
            trump=trump, trick_number=1, leader=0,
            declarations=decls,
        )
        state.points = [0, 0, 0, 0]

        round_features = []
        round_cards = []

        for trick_idx in range(9):
            trick = []
            for i in range(4):
                p = (state.leader + i) % 4
                lead_suit = CARD_SUIT[trick[0][1]] if trick else None
                legal = lm(state.hands[p], lead_suit, trump,
                           trick if trick else None)
                if not legal:
                    legal = list(state.hands[p])

                if p == 0:
                    # Record state before decision
                    feat = encode_state(
                        hand=list(state.hands[0]),
                        played_cards=set(state.played_cards),
                        trick_cards=trick,
                        trump=trump,
                        legal=legal,
                        trick_number=trick_idx,
                        my_points=state.points[0],
                        opp_points=[state.points[1], state.points[2], state.points[3]],
                        my_target=decls[0],
                        leader=state.leader,
                        voids=state.voids,
                    )
                    round_features.append(feat)

                    # Get PIMC choice
                    state_copy = state.copy()
                    state_copy.current_trick = list(trick)
                    card = pimc.play(state_copy, legal)
                    round_cards.append(card)
                else:
                    state_copy = state.copy()
                    state_copy.current_trick = list(trick)
                    card = heurs[p].play(state_copy, legal)

                state.hands[p].remove(card)
                trick.append((p, card))
                if lead_suit is None and trick:
                    lead_suit = CARD_SUIT[trick[0][1]]

                # Detect voids
                if len(trick) > 1 and CARD_SUIT[card] != CARD_SUIT[trick[0][1]]:
                    state.voids[p].add(CARD_SUIT[trick[0][1]])

            # Resolve trick
            from engine.cards import strength as card_str
            lead_s = CARD_SUIT[trick[0][1]]
            best_p, best_c = trick[0]
            for tp, tc in trick[1:]:
                cs = CARD_SUIT[tc]
                bs = CARD_SUIT[best_c]
                if cs == trump and bs != trump:
                    best_p, best_c = tp, tc
                elif cs == bs and card_str(tc, trump) > card_str(best_c, trump):
                    best_p, best_c = tp, tc
            winner = best_p

            pts = sum(card_points(c, trump) for _, c in trick)
            if trick_idx == 8:
                pts += 5
            state.points[winner] += pts
            state.leader = winner

            for _, c in trick:
                state.played_cards.add(c)

            # Notify
            pimc.notify_trick(state, trick, winner, pts)
            for p in [1, 2, 3]:
                heurs[p].notify_trick(state, trick, winner, pts)

        final_dev = abs(decls[0] - state.points[0])

        for feat, card in zip(round_features, round_cards):
            features_list.append(feat)
            cards_list.append(card)
            deviations_list.append(float(final_dev))

    if not features_list:
        return np.zeros((0, FEATURE_SIZE), dtype=np.float32), \
               np.zeros(0, dtype=np.int64), \
               np.zeros(0, dtype=np.float32)

    return (
        np.array(features_list, dtype=np.float32),
        np.array(cards_list, dtype=np.int64),
        np.array(deviations_list, dtype=np.float32),
    )


def generate(num_rounds: int = 5000, num_worlds: int = 100, ansage_worlds: int = 500,
             output_dir: str = "data"):
    """Generate training data using parallel PIMC self-play."""
    n_workers = min(cpu_count() or 1, num_rounds)
    rounds_per_worker = num_rounds // n_workers
    remainder = num_rounds % n_workers

    print(f"Generating {num_rounds} rounds on {n_workers} cores")
    print(f"  PIMC: {num_worlds} move worlds, {ansage_worlds} declaration worlds")
    print(f"  Feature vector size: {FEATURE_SIZE}")

    args = []
    seed = 100000
    for w in range(n_workers):
        n = rounds_per_worker + (1 if w < remainder else 0)
        args.append((seed, n, num_worlds, ansage_worlds))
        seed += n

    start = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(_worker, args)

    all_features = np.concatenate([r[0] for r in results if len(r[0]) > 0])
    all_cards = np.concatenate([r[1] for r in results if len(r[1]) > 0])
    all_devs = np.concatenate([r[2] for r in results if len(r[2]) > 0])

    elapsed = time.time() - start

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "poc_features.npy"), all_features)
    np.save(os.path.join(output_dir, "poc_labels_card.npy"), all_cards)
    np.save(os.path.join(output_dir, "poc_labels_deviation.npy"), all_devs)

    # Compute utility labels (zeros for now — no per-card utilities in fast generation)
    utils = np.zeros((len(all_cards), 36), dtype=np.float32)
    np.save(os.path.join(output_dir, "poc_labels_utility.npy"), utils)

    total_mb = (all_features.nbytes + all_cards.nbytes + all_devs.nbytes + utils.nbytes) / 1e6
    print(f"\nGeneration complete:")
    print(f"  Rounds: {num_rounds}")
    print(f"  Play decisions: {len(all_cards)}")
    print(f"  Time: {elapsed:.1f}s ({num_rounds/elapsed:.1f} rounds/sec)")
    print(f"  Feature shape: {all_features.shape}")
    print(f"  Files: {total_mb:.1f} MB total in {output_dir}/")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    generate(num_rounds=n)

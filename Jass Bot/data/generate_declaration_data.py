"""Generate training data for neural declaration network.

For each random deal, simulates the game N times with different opponent
configurations and records (hand, trump, score_distribution) for each player.

Uses the C engine for maximum throughput.
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

from engine.cards import CARD_SUIT, NUM_CARDS
from bot.heuristics import estimate_hand_score


def _deal(rng: random.Random) -> tuple[int, dict[int, list[int]]]:
    """Deal 36 cards to 4 players, pick random trump."""
    cards = list(range(NUM_CARDS))
    rng.shuffle(cards)
    hands = {
        0: sorted(cards[0:9]),
        1: sorted(cards[9:18]),
        2: sorted(cards[18:27]),
        3: sorted(cards[27:36]),
    }
    trump = rng.randint(0, 3)
    return trump, hands


def encode_hand(hand: list[int], trump: int) -> np.ndarray:
    """Encode hand + trump as feature vector.

    Layout (40 features):
      hand_binary[36]: 1.0 for each card in hand
      trump_onehot[4]: 1.0 for trump suit
    """
    f = np.zeros(40, dtype=np.float32)
    for c in hand:
        f[c] = 1.0
    f[36 + trump] = 1.0
    return f


def encode_hand_rich(hand: list[int], trump: int) -> np.ndarray:
    """Rich feature encoding for better learning.

    Layout (72 features):
      hand_binary[36]: 1.0 for each card in hand
      trump_onehot[4]: 1.0 for trump suit
      trump_cards[9]: 1.0 for each trump card in hand (by value index)
      suit_lengths[4]: number of cards per suit / 9
      suit_points[4]: points per suit / 62 (normalized)
      has_puur[1]: 1.0 if hand contains Puur
      has_nell[1]: 1.0 if hand contains Nell
      has_trump_ace[1]: 1.0 if hand contains trump Ace
      trump_count[1]: number of trumps / 9
      void_count[1]: number of void side suits / 3
      total_points[1]: sum of all card points / 157
      trump_points[1]: sum of trump card points / 62
      side_aces[1]: number of non-trump aces / 3
      high_trump_count[1]: number of trumps with strength >= 6 (A,9,U) / 3
      protected_aces[1]: number of aces in suits with 2+ cards / 3
      long_suits[1]: number of suits with 4+ cards / 4
    """
    from engine.cards import CARD_STRENGTH, points as card_points

    f = np.zeros(72, dtype=np.float32)

    # hand_binary[36]
    for c in hand:
        f[c] = 1.0

    # trump_onehot[4]
    f[36 + trump] = 1.0

    # trump_cards[9] — by value index within trump suit
    trump_base = trump * 9
    for c in hand:
        if CARD_SUIT[c] == trump:
            f[40 + (c - trump_base)] = 1.0

    # Suit analysis
    suits: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for c in hand:
        suits[CARD_SUIT[c]].append(c)

    # suit_lengths[4]
    for s in range(4):
        f[49 + s] = len(suits[s]) / 9.0

    # suit_points[4]
    for s in range(4):
        pts = sum(card_points(c, trump) for c in suits[s])
        f[53 + s] = pts / 62.0

    trump_cards = suits[trump]
    puur_id = trump * 9 + 5
    nell_id = trump * 9 + 3
    trump_ace_id = trump * 9 + 8

    # has_puur, has_nell, has_trump_ace
    f[57] = 1.0 if puur_id in hand else 0.0
    f[58] = 1.0 if nell_id in hand else 0.0
    f[59] = 1.0 if trump_ace_id in hand else 0.0

    # trump_count
    f[60] = len(trump_cards) / 9.0

    # void_count (side suits only)
    void_count = sum(1 for s in range(4) if s != trump and len(suits[s]) == 0)
    f[61] = void_count / 3.0

    # total_points
    f[62] = sum(card_points(c, trump) for c in hand) / 157.0

    # trump_points
    f[63] = sum(card_points(c, trump) for c in trump_cards) / 62.0

    # side_aces
    side_aces = sum(1 for c in hand if CARD_SUIT[c] != trump and CARD_STRENGTH[c][0] == 8)
    f[64] = side_aces / 3.0

    # high_trump_count (A=6, 9=7, U=8 in trump strength)
    high_trumps = sum(1 for c in trump_cards if CARD_STRENGTH[c][1] >= 6)
    f[65] = high_trumps / 3.0

    # protected_aces (aces in suits with 2+ cards)
    protected = 0
    for s in range(4):
        if s == trump:
            continue
        if len(suits[s]) >= 2 and any(CARD_STRENGTH[c][0] == 8 for c in suits[s]):
            protected += 1
    f[66] = protected / 3.0

    # long_suits (4+ cards)
    long_suits = sum(1 for s in range(4) if len(suits[s]) >= 4)
    f[67] = long_suits / 4.0

    return f


FEATURE_SIZE_SIMPLE = 40
FEATURE_SIZE_RICH = 72  # Updated to match actual feature count


def _worker_c_engine(args):
    """Generate declaration data using C engine batch simulation."""
    seed_start, n_deals, sims_per_hand = args

    from engine_c.fast_engine import FastEngine, hand_to_mask
    from ctypes import c_uint64, c_int

    engine = FastEngine()
    rng = random.Random(seed_start)

    features_list = []
    scores_list = []

    for d in range(n_deals):
        trump, hands = _deal(rng)

        # For each player, simulate their score across multiple opponent configurations
        for player in range(4):
            my_hand = hands[player]
            feat = encode_hand_rich(my_hand, trump)

            # Generate sims_per_hand random opponent configurations
            unknown = [c for c in range(36) if c not in my_hand]
            sim_scores = []

            for _ in range(sims_per_hand):
                rng.shuffle(unknown)
                opp_hands = {
                    (player + 1) % 4: unknown[0:9],
                    (player + 2) % 4: unknown[9:18],
                    (player + 3) % 4: unknown[18:27],
                }
                # Remap to internal IDs (0=us, 1,2,3=opponents)
                world = {1: opp_hands[(player + 1) % 4],
                         2: opp_hands[(player + 2) % 4],
                         3: opp_hands[(player + 3) % 4]}

                # Get targets for all players
                my_target = estimate_hand_score(my_hand, trump)
                opp_targets = {p: estimate_hand_score(world[p], trump) for p in [1, 2, 3]}

                # Use C engine for simulation
                worlds_list = [world]
                opp_tgt_list = [opp_targets]
                result = engine.simulate_for_distribution(
                    my_hand, worlds_list, trump, my_target, opp_tgt_list
                )
                sim_scores.append(result[0])

            # Record median score as the label
            sim_scores.sort()
            median_score = sim_scores[len(sim_scores) // 2]

            features_list.append(feat)
            scores_list.append(float(median_score))

    return (
        np.array(features_list, dtype=np.float32),
        np.array(scores_list, dtype=np.float32),
    )


def _worker_fast(args):
    """Faster worker: 1 simulation per deal, rely on data volume for accuracy."""
    seed_start, n_deals, _ = args

    from engine_c.fast_engine import FastEngine
    engine = FastEngine()
    rng = random.Random(seed_start)

    features_list = []
    scores_list = []

    for d in range(n_deals):
        trump, hands = _deal(rng)

        # Simulate one game with all 4 players using gap heuristic
        targets = {p: estimate_hand_score(hands[p], trump) for p in range(4)}

        # Use C engine: simulate from player 0's perspective
        # But we want ALL players' scores. Simulate 4 times, once per "perspective"
        for player in range(4):
            my_hand = hands[player]
            feat = encode_hand_rich(my_hand, trump)

            # Remap opponents
            opp_ids = [(player + i) % 4 for i in range(1, 4)]
            world = {i + 1: hands[opp_ids[i]] for i in range(3)}
            opp_tgts = {i + 1: targets[opp_ids[i]] for i in range(3)}

            result = engine.simulate_for_distribution(
                my_hand, [world], trump, targets[player], [opp_tgts]
            )
            score = result[0]

            features_list.append(feat)
            scores_list.append(float(score))

    return (
        np.array(features_list, dtype=np.float32),
        np.array(scores_list, dtype=np.float32),
    )


def generate(
    num_deals: int = 50000,
    sims_per_hand: int = 1,
    output_dir: str = "data",
    mode: str = "fast",
):
    """Generate declaration training data.

    Args:
        num_deals: number of random deals
        sims_per_hand: simulations per hand (1=fast, 20+=accurate medians)
        output_dir: output directory
        mode: 'fast' (1 sim/deal, high volume) or 'accurate' (N sims/hand, median labels)
    """
    n_workers = min(cpu_count() or 1, num_deals)
    deals_per_worker = num_deals // n_workers
    remainder = num_deals % n_workers

    total_samples = num_deals * 4  # 4 players per deal
    print(f"Generating declaration data: {num_deals} deals = {total_samples} samples")
    print(f"  Mode: {mode}, sims_per_hand: {sims_per_hand}")
    print(f"  Workers: {n_workers}")
    print(f"  Feature size: {FEATURE_SIZE_RICH}")

    args = []
    seed = 200000
    for w in range(n_workers):
        n = deals_per_worker + (1 if w < remainder else 0)
        args.append((seed, n, sims_per_hand))
        seed += n * 100

    worker_fn = _worker_fast if mode == "fast" else _worker_c_engine

    start = time.time()
    with Pool(processes=n_workers) as pool:
        results = pool.map(worker_fn, args)

    all_features = np.concatenate([r[0] for r in results if len(r[0]) > 0])
    all_scores = np.concatenate([r[1] for r in results if len(r[1]) > 0])

    elapsed = time.time() - start

    os.makedirs(output_dir, exist_ok=True)
    feat_path = os.path.join(output_dir, "decl_features.npy")
    score_path = os.path.join(output_dir, "decl_scores.npy")
    np.save(feat_path, all_features)
    np.save(score_path, all_scores)

    total_mb = (all_features.nbytes + all_scores.nbytes) / 1e6
    print(f"\nGeneration complete:")
    print(f"  Deals: {num_deals}, Samples: {len(all_scores)}")
    print(f"  Time: {elapsed:.1f}s ({num_deals / elapsed:.0f} deals/sec)")
    print(f"  Score stats: mean={all_scores.mean():.1f}, std={all_scores.std():.1f}, "
          f"min={all_scores.min():.0f}, max={all_scores.max():.0f}")
    print(f"  Files: {feat_path} ({all_features.nbytes/1e6:.1f}MB), "
          f"{score_path} ({all_scores.nbytes/1e6:.1f}MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate declaration training data")
    parser.add_argument("--deals", type=int, default=50000, help="Number of deals")
    parser.add_argument("--sims", type=int, default=20, help="Sims per hand (accurate mode)")
    parser.add_argument("--mode", choices=["fast", "accurate"], default="accurate",
                        help="fast=1sim/deal, accurate=N sims/hand with median")
    parser.add_argument("--output", default="data", help="Output directory")
    args = parser.parse_args()

    generate(num_deals=args.deals, sims_per_hand=args.sims,
             output_dir=args.output, mode=args.mode)

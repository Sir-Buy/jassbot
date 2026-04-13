"""Fast vectorized engine operations using numpy.

Provides batch simulation for declaration evaluation and move selection.
All card operations use integer IDs (0-35) in numpy arrays.
"""

from __future__ import annotations

import numpy as np

# --- Precomputed tables as numpy arrays ---

# Points: shape (36, 2) — [card_id][0=non-trump, 1=trump]
_PTS_NORMAL = np.array([0, 0, 0, 0, 10, 2, 3, 4, 11] * 4, dtype=np.int32)
_PTS_TRUMP = np.array([0, 0, 0, 14, 10, 20, 3, 4, 11] * 4, dtype=np.int32)
POINTS_TABLE = np.stack([_PTS_NORMAL, _PTS_TRUMP], axis=1)  # (36, 2)

# Strength: shape (36, 2) — [card_id][0=non-trump, 1=trump]
_STR_NORMAL = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8] * 4, dtype=np.int32)
_STR_TRUMP = np.array([0, 1, 2, 7, 3, 8, 4, 5, 6] * 4, dtype=np.int32)
STRENGTH_TABLE = np.stack([_STR_NORMAL, _STR_TRUMP], axis=1)  # (36, 2)

# Suit lookup
SUIT_TABLE = np.arange(36, dtype=np.int32) // 9


def fast_points(cards: np.ndarray, trump: int) -> np.ndarray:
    """Points for an array of card IDs."""
    is_trump = (SUIT_TABLE[cards] == trump).astype(np.int32)
    return POINTS_TABLE[cards, is_trump]


def fast_strength(cards: np.ndarray, trump: int) -> np.ndarray:
    """Strength for an array of card IDs. Trump cards get +100."""
    is_trump = (SUIT_TABLE[cards] == trump).astype(np.int32)
    base = STRENGTH_TABLE[cards, is_trump]
    return base + is_trump * 100


def fast_trick_winner(
    player_ids: np.ndarray,
    card_ids: np.ndarray,
    trump: int,
) -> int:
    """Fast trick winner for a single trick.

    Args:
        player_ids: (4,) array of player IDs in play order
        card_ids: (4,) array of card IDs in play order
        trump: trump suit
    Returns:
        winning player ID
    """
    strengths = fast_strength(card_ids, trump)
    suits = SUIT_TABLE[card_ids]
    lead_suit = suits[0]

    # Trump cards always eligible to win
    # Non-trump cards only win if they match the current best card's suit
    best_idx = 0
    best_str = strengths[0]
    best_is_trump = (suits[0] == trump)

    for i in range(1, len(card_ids)):
        is_trump_i = (suits[i] == trump)
        if is_trump_i and not best_is_trump:
            best_idx = i
            best_str = strengths[i]
            best_is_trump = True
        elif is_trump_i and best_is_trump:
            if strengths[i] > best_str:
                best_idx = i
                best_str = strengths[i]
        elif not is_trump_i and not best_is_trump:
            if suits[i] == suits[best_idx] and strengths[i] > best_str:
                best_idx = i
                best_str = strengths[i]

    return int(player_ids[best_idx])


# =========================================================================
# Batch simulation for declaration evaluation
# =========================================================================

def batch_sim_games(
    my_hand: np.ndarray,
    worlds: np.ndarray,
    trump: int,
    target: int,
    opp_targets: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate games across multiple worlds in batch.

    Args:
        my_hand: (num_cards,) my card IDs
        worlds: (num_worlds, 3, num_cards) opponent hands
        trump: trump suit
        target: my target (used for target-aware self-play)
        opp_targets: (num_worlds, 3) opponent targets, or None for greedy

    Returns:
        (num_worlds,) array of player 0's final points per world
    """
    num_worlds = worlds.shape[0]
    num_cards = my_hand.shape[0]
    results = np.zeros(num_worlds, dtype=np.int32)

    for w in range(num_worlds):
        results[w] = _sim_one_game(
            my_hand.copy(), worlds[w].copy(), trump, target,
            opp_targets[w] if opp_targets is not None else None,
        )

    return results


def _sim_one_game(
    my_hand: np.ndarray,
    opp_hands: np.ndarray,
    trump: int,
    target: int,
    opp_targets: np.ndarray | None,
) -> int:
    """Simulate one game. opp_hands shape: (3, num_cards)."""
    num_cards = len(my_hand)
    my_pts = 0
    opp_pts = np.zeros(3, dtype=np.int32)
    leader = 0

    # Track hand sizes (cards get "removed" by setting to -1)
    my_mask = np.ones(num_cards, dtype=bool)
    opp_mask = np.ones((3, num_cards), dtype=bool)

    for trick_num in range(num_cards):
        trick_players = np.empty(4, dtype=np.int32)
        trick_cards = np.empty(4, dtype=np.int32)
        tc = 0
        lead_suit = -1

        for i in range(4):
            p = (leader + i) % 4

            if p == 0:
                card, idx = _pick_card_fast(
                    my_hand, my_mask, lead_suit, trump, target - my_pts
                )
                my_mask[idx] = False
            else:
                oi = p - 1  # opponent index 0-2
                gap = (opp_targets[oi] - opp_pts[oi]) if opp_targets is not None else 20
                card, idx = _pick_card_fast(
                    opp_hands[oi], opp_mask[oi], lead_suit, trump, gap
                )
                opp_mask[oi][idx] = False

            trick_players[tc] = p
            trick_cards[tc] = card
            tc += 1

            if lead_suit < 0:
                lead_suit = int(SUIT_TABLE[card])

        winner = fast_trick_winner(trick_players[:tc], trick_cards[:tc], trump)
        pts = int(np.sum(fast_points(trick_cards[:tc], trump)))

        if winner == 0:
            my_pts += pts
        else:
            opp_pts[winner - 1] += pts

        leader = winner

    return my_pts


def _pick_card_fast(
    hand: np.ndarray,
    mask: np.ndarray,
    lead_suit: int,
    trump: int,
    gap: int,
) -> tuple[int, int]:
    """Pick a card using heuristic. Returns (card_id, index_in_hand)."""
    available = np.where(mask)[0]
    if len(available) == 0:
        return 0, 0

    cards = hand[available]

    # Follow suit
    if lead_suit >= 0:
        suits = SUIT_TABLE[cards]
        follow = available[suits == lead_suit]
        if len(follow) > 0:
            available = follow
            cards = hand[available]

    pts = fast_points(cards, trump)
    strs = fast_strength(cards, trump)

    if gap <= 0:
        # Shed: lowest points, then lowest strength
        score = pts * 1000 + strs
        best = np.argmin(score)
    else:
        # Gain: highest points, then highest strength
        score = pts * 1000 + strs
        best = np.argmax(score)

    return int(cards[best]), int(available[best])


# =========================================================================
# Fast declaration evaluation
# =========================================================================

def fast_eval_declarations(
    my_hand: list[int],
    worlds_list: list[dict[int, list[int]]],
    trump: int,
    opp_decls_list: list[dict[int, int]] | None = None,
) -> list[int]:
    """Simulate all worlds and return score distribution.

    Returns list of player 0's final points for each world.
    """
    num_cards = len(my_hand)
    num_worlds = len(worlds_list)

    hand_arr = np.array(my_hand, dtype=np.int32)

    # Pack worlds into numpy array
    worlds_arr = np.zeros((num_worlds, 3, num_cards), dtype=np.int32)
    opp_tgt_arr = None

    for w, world in enumerate(worlds_list):
        for oi, p in enumerate([1, 2, 3]):
            cards = world[p]
            n = min(len(cards), num_cards)
            worlds_arr[w, oi, :n] = cards[:n]

    if opp_decls_list is not None:
        opp_tgt_arr = np.zeros((num_worlds, 3), dtype=np.int32)
        for w, decls in enumerate(opp_decls_list):
            for oi, p in enumerate([1, 2, 3]):
                opp_tgt_arr[w, oi] = decls[p]

    # Use a moderate target estimate for self-play
    from bot.heuristics import estimate_hand_score
    est_target = estimate_hand_score(my_hand, trump)

    results = batch_sim_games(hand_arr, worlds_arr, trump, est_target, opp_tgt_arr)
    return results.tolist()

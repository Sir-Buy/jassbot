"""Heuristic play and declaration functions for opponents and simulation rollouts.

All functions work with integer card IDs.
"""

from __future__ import annotations

from engine.cards import CARD_SUIT, CARD_STRENGTH, CARD_POINTS, points, strength


# =========================================================================
# Greedy opponent (original, for backward compat)
# =========================================================================

def pick_opponent_greedy(
    hand: list[int],
    table_cards: list[tuple[int, int]],
    lead_suit: int | None,
    trump_suit: int,
) -> int:
    """Greedy opponent: tries to win valuable tricks, dumps low cards otherwise."""
    if not hand:
        raise ValueError("Empty hand")

    legal = _follow_suit(hand, lead_suit) if lead_suit is not None else list(hand)
    if not legal:
        legal = list(hand)

    table_pts = sum(points(c, trump_suit) for _, c in table_cards)

    if table_pts >= 10:
        winners = [c for c in legal if _beats_table(c, table_cards, lead_suit, trump_suit)]
        if winners:
            return min(winners, key=lambda c: strength(c, trump_suit))

    return min(legal, key=lambda c: (points(c, trump_suit), strength(c, trump_suit)))


# =========================================================================
# Target-aware opponent (Phase 2)
# =========================================================================

def pick_opponent_target_aware(
    hand: list[int],
    table_cards: list[tuple[int, int]],
    lead_suit: int | None,
    trump_suit: int,
    target: int,
    current_points: int,
    trick_number: int,
) -> int:
    """Target-tracking opponent: plays to hit their own declaration target.

    Args:
        hand: opponent's cards
        table_cards: (player, card_id) on table so far
        lead_suit: led suit or None
        trump_suit: trump suit index
        target: opponent's declared target
        current_points: opponent's current points
        trick_number: 0-indexed trick number (0 = first trick)

    Returns:
        card_id to play
    """
    if not hand:
        raise ValueError("Empty hand")

    legal = _follow_suit(hand, lead_suit) if lead_suit is not None else list(hand)
    if not legal:
        legal = list(hand)

    if len(legal) == 1:
        return legal[0]

    gap = target - current_points
    remaining_tricks = max(1, 9 - trick_number)

    if gap <= 0:
        # At or over target: shed points, avoid winning
        # Play lowest card that doesn't win, if possible
        non_winners = [c for c in legal if not _beats_table(c, table_cards, lead_suit, trump_suit)]
        if non_winners:
            return min(non_winners, key=lambda c: (points(c, trump_suit), strength(c, trump_suit)))
        # All cards win — play the one that wins least points
        return min(legal, key=lambda c: (points(c, trump_suit), strength(c, trump_suit)))

    avg_needed = gap / remaining_tricks

    if avg_needed > 10:
        # Far behind: play aggressively to win points
        table_pts = sum(points(c, trump_suit) for _, c in table_cards)
        winners = [c for c in legal if _beats_table(c, table_cards, lead_suit, trump_suit)]
        if winners and table_pts >= 5:
            # Win with the strongest card to secure points
            return max(winners, key=lambda c: strength(c, trump_suit))
        if winners:
            return max(winners, key=lambda c: points(c, trump_suit))
        # Can't win — dump lowest
        return min(legal, key=lambda c: (points(c, trump_suit), strength(c, trump_suit)))

    # Moderate need: win only when it brings us closer to target
    table_pts = sum(points(c, trump_suit) for _, c in table_cards)
    winners = [c for c in legal if _beats_table(c, table_cards, lead_suit, trump_suit)]

    if winners and table_pts > 0:
        # Would winning overshoot? Pick cheapest winner if not
        cheapest = min(winners, key=lambda c: strength(c, trump_suit))
        trick_value = table_pts + points(cheapest, trump_suit)
        if current_points + trick_value <= target + 5:
            return cheapest

    # Play middle-strength card (cautious)
    by_str = sorted(legal, key=lambda c: strength(c, trump_suit))
    return by_str[len(by_str) // 2]


# =========================================================================
# Self-play rollout heuristic (for player 0 during simulation)
# =========================================================================

def pick_sim_target_aware(
    hand: list[int],
    table_cards: list[tuple[int, int]],
    lead_suit: int | None,
    trump_suit: int,
    target: int,
    current_points: int,
) -> int:
    """Target-aware heuristic for simulation rollouts.

    Plays to approach target: gains points when behind, sheds when ahead.
    """
    if not hand:
        raise ValueError("Empty hand")

    legal = _follow_suit(hand, lead_suit) if lead_suit is not None else list(hand)
    if not legal:
        legal = list(hand)

    diff = target - current_points

    if diff <= 0:
        return min(legal, key=lambda c: (points(c, trump_suit), strength(c, trump_suit)))

    table_pts = sum(points(c, trump_suit) for _, c in table_cards)

    if table_pts >= 10 or diff > 20:
        winners = [c for c in legal if _beats_table(c, table_cards, lead_suit, trump_suit)]
        if winners:
            return max(winners, key=lambda c: points(c, trump_suit))

    by_str = sorted(legal, key=lambda c: strength(c, trump_suit))
    return by_str[len(by_str) // 2]


# =========================================================================
# Hand score estimation (for opponent declaration inference)
# =========================================================================

def estimate_hand_score(hand: list[int], trump_suit: int) -> int:
    """Expert-calibrated hand score estimate for Differenzler.

    Based on the Swiss Jass expert rule: "Trump face values × 2 + side Aces × 11"
    enhanced with trump count, void suit, and card protection adjustments.

    Used for opponent declaration inference and declaration simulation targets.
    """
    # Separate cards by suit
    suits: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    for c in hand:
        suits[CARD_SUIT[c]].append(c)

    trump_cards = suits[trump_suit]
    trump_count = len(trump_cards)

    # ── Expert base: trump face value × multiplier ──
    # Expert rule is "trump face × 2", but calibrated to 1.7 against actual
    # game data (the rollout heuristic doesn't extract full expert value).
    trump_face = sum(points(c, trump_suit) for c in trump_cards)
    base = int(trump_face * 1.7)

    # ── Trump count bonus ──
    # Many low trumps (face value near 0) still win tricks worth ~12 pts each.
    # Expert insight: 5 low trumps ≈ 35 pts even if face value is 0.
    if trump_count >= 5:
        base = max(base, 25 + trump_count * 5)
    elif trump_count >= 3:
        count_bonus = (trump_count - 2) * 3
        base += count_bonus

    # ── Puur and Nell extra value ──
    # Puur: strongest card, guaranteed trick winner. Small bonus beyond the multiplier.
    # Nell: second strongest. Modest extra.
    has_puur = False
    has_nell = False
    for c in trump_cards:
        trump_str = CARD_STRENGTH[c][1]
        if trump_str == 8:  # Puur
            has_puur = True
            base += 4
        elif trump_str == 7:  # Nell
            has_nell = True
            base += 2

    # ── Side suits analysis ──
    side_bonus = 0
    void_count = 0

    for s in range(4):
        if s == trump_suit:
            continue
        suit_cards = suits[s]

        if not suit_cards:
            void_count += 1
            continue

        suit_len = len(suit_cards)
        has_ace = False
        has_king = False
        has_ten = False
        for c in suit_cards:
            ns = CARD_STRENGTH[c][0]
            if ns == 8:
                has_ace = True
            elif ns == 7:
                has_king = True
            elif ns == 4:  # 10/Banner
                has_ten = True

        # Ace: likely wins its trick (11 pts)
        if has_ace:
            if suit_len >= 2:
                side_bonus += 9   # Protected — likely to take trick
            else:
                side_bonus += 5   # Singleton — may be trumped (~55% success)

        # King with protection: sometimes wins
        if has_king and suit_len >= 3:
            side_bonus += 2

        # Ten/Banner: valuable but vulnerable
        if has_ten and has_ace:
            side_bonus += 4  # Ace protects it

    # ── Void suit bonus ──
    # Being void in a side suit lets you trump in on opponents' aces/tens.
    # Each void + available trump ≈ one extra trick won (~10 pts average).
    if trump_count >= 1 and void_count >= 1:
        usable_trumps = trump_count
        if has_puur:
            usable_trumps -= 1
        if has_nell:
            usable_trumps -= 1
        effective_voids = min(void_count, max(0, usable_trumps))
        side_bonus += effective_voids * 8

    return max(0, min(157, base + side_bonus))


# =========================================================================
# Shared helpers
# =========================================================================

def _follow_suit(hand: list[int], lead_suit: int) -> list[int]:
    return [c for c in hand if CARD_SUIT[c] == lead_suit]


def _beats_table(
    card: int,
    table_cards: list[tuple[int, int]],
    lead_suit: int | None,
    trump_suit: int,
) -> bool:
    """Would this card currently beat everything on the table?"""
    if not table_cards:
        return True
    card_suit = CARD_SUIT[card]
    if lead_suit is not None and card_suit != lead_suit and card_suit != trump_suit:
        return False
    card_str = strength(card, trump_suit)
    for _, tc in table_cards:
        tc_suit = CARD_SUIT[tc]
        if tc_suit == trump_suit and card_suit != trump_suit:
            return False
        if tc_suit == card_suit and strength(tc, trump_suit) >= card_str:
            return False
    return True

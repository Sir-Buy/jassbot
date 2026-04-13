"""Jass Differenzler rules: legal moves, trick winner, scoring.

All functions operate on integer card IDs for speed.
"""

from __future__ import annotations

from engine.cards import (
    CARD_SUIT,
    CARD_STRENGTH,
    CARD_POINTS,
    CARD_VALUE,
    LAST_TRICK_BONUS,
    NUM_VALUES,
    VALUE_INDEX,
    points,
    strength,
)


# --- Legal moves ---

def legal_moves(
    hand: list[int],
    lead_suit: int | None,
    trump_suit: int,
    table_cards: list[tuple[int, int]] | None = None,
) -> list[int]:
    """Return legal card IDs from hand given the current trick state.

    Args:
        hand: cards in player's hand (card IDs)
        lead_suit: suit index of the led card, or None if leading
        trump_suit: suit index of trump
        table_cards: list of (player, card_id) already on the table, in play order

    Rules:
        1. If leading: any card is legal.
        2. Must follow led suit if possible.
        3. If trump is led: must play trump AND must overtrump if possible.
           Exception (Puur privilege): if your only trump that could overtrump
           is the Puur (trump Under/Jack), you may keep it and undertrump.
        4. If you can't follow suit: may play anything, BUT cannot undertrump
           (play a trump lower than the highest trump on the table) unless
           you have no other option.
    """
    if not hand:
        return []

    # Leading: anything goes
    if lead_suit is None:
        return list(hand)

    # Cards of the led suit in hand
    follow = [c for c in hand if CARD_SUIT[c] == lead_suit]

    if lead_suit == trump_suit:
        # Trump was led
        return _legal_trump_led(hand, follow, trump_suit, table_cards)

    if follow:
        # Can follow suit (non-trump led)
        return follow

    # Can't follow suit: may play anything, but undertrump restriction applies
    return _legal_cant_follow(hand, trump_suit, table_cards)


def _legal_trump_led(
    hand: list[int],
    trumps_in_hand: list[int],
    trump_suit: int,
    table_cards: list[tuple[int, int]] | None,
) -> list[int]:
    """When trump is led, must play trump and overtrump if possible."""
    if not trumps_in_hand:
        # No trump at all: play anything
        return list(hand)

    # Find highest trump strength on table
    highest_trump_str = _highest_trump_strength_on_table(table_cards, trump_suit)

    # Cards that overtrump
    over = [c for c in trumps_in_hand
            if CARD_STRENGTH[c][1] > highest_trump_str]

    if over:
        # Must overtrump. But Puur privilege: if the ONLY card that can
        # overtrump is the Puur, you may play any trump instead.
        puur_id = trump_suit * NUM_VALUES + VALUE_INDEX["U"]
        if len(over) == 1 and over[0] == puur_id:
            # Puur privilege: can play any trump
            return trumps_in_hand
        return over

    # Can't overtrump: play any trump (undertrump is allowed when trump is led
    # and you can't overtrump)
    return trumps_in_hand


def _legal_cant_follow(
    hand: list[int],
    trump_suit: int,
    table_cards: list[tuple[int, int]] | None,
) -> list[int]:
    """When you can't follow suit: anything goes, but no undertrumping unless forced."""
    highest_trump_str = _highest_trump_strength_on_table(table_cards, trump_suit)

    if highest_trump_str < 0:
        # No trump on table: anything is fine
        return list(hand)

    # There's trump on the table. Can't undertrump unless that's all we have.
    non_trump = [c for c in hand if CARD_SUIT[c] != trump_suit]
    over_trump = [c for c in hand if CARD_SUIT[c] == trump_suit
                  and CARD_STRENGTH[c][1] > highest_trump_str]

    allowed = non_trump + over_trump
    if allowed:
        return allowed

    # Only have undertrumps: forced to play them
    return list(hand)


def _highest_trump_strength_on_table(
    table_cards: list[tuple[int, int]] | None,
    trump_suit: int,
) -> int:
    """Return the highest trump strength on the table, or -1 if no trump played."""
    if not table_cards:
        return -1
    best = -1
    for _, cid in table_cards:
        if CARD_SUIT[cid] == trump_suit:
            s = CARD_STRENGTH[cid][1]
            if s > best:
                best = s
    return best


# --- Trick winner ---

def trick_winner(
    trick: list[tuple[int, int]],
    lead_suit: int,
    trump_suit: int,
) -> int:
    """Determine which player wins the trick.

    Args:
        trick: list of (player_id, card_id) in play order
        lead_suit: suit of the first card played
        trump_suit: trump suit index

    Returns:
        player_id of the winner
    """
    best_player = trick[0][0]
    best_card = trick[0][1]
    best_str = strength(best_card, trump_suit)
    best_is_trump = CARD_SUIT[best_card] == trump_suit

    for player, card in trick[1:]:
        card_suit = CARD_SUIT[card]
        is_trump = card_suit == trump_suit

        if is_trump and not best_is_trump:
            # Trump beats non-trump
            best_player, best_card = player, card
            best_str = strength(card, trump_suit)
            best_is_trump = True
        elif is_trump and best_is_trump:
            # Both trump: higher strength wins
            s = strength(card, trump_suit)
            if s > best_str:
                best_player, best_card = player, card
                best_str = s
        elif not is_trump and not best_is_trump:
            # Neither trump: must be same suit as current best to beat it
            if card_suit == CARD_SUIT[best_card]:
                s = strength(card, trump_suit)
                if s > best_str:
                    best_player, best_card = player, card
                    best_str = s
        # else: non-trump can't beat trump — skip

    return best_player


# --- Scoring ---

def trick_points(trick: list[tuple[int, int]], trump_suit: int) -> int:
    """Sum of points for cards in a trick (excludes last-trick bonus)."""
    return sum(points(card, trump_suit) for _, card in trick)


def trick_points_with_bonus(
    trick: list[tuple[int, int]],
    trump_suit: int,
    is_last_trick: bool,
) -> int:
    """Points for a trick, including last-trick bonus if applicable."""
    pts = trick_points(trick, trump_suit)
    if is_last_trick:
        pts += LAST_TRICK_BONUS
    return pts


def round_card_points(trump_suit: int) -> int:
    """Total card points in a round (without last-trick bonus). Should be 152."""
    total = 0
    for cid in range(36):
        total += points(cid, trump_suit)
    return total

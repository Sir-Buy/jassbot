"""Exhaustive tests for Jass Differenzler rules.

Covers: card points, strength rankings, legal moves (follow suit,
undertrump, Puur privilege), trick winner, scoring, and 157 total.
"""

from __future__ import annotations

import random

import pytest

from engine.cards import (
    SUITS, VALUES, SUIT_INDEX, VALUE_INDEX, NUM_VALUES,
    card_id, points, strength, CARD_SUIT, CARD_POINTS, CARD_STRENGTH,
    LAST_TRICK_BONUS, TOTAL_ROUND_POINTS, ALL_CARD_IDS,
)
from engine.rules import legal_moves, trick_winner, trick_points, trick_points_with_bonus, round_card_points


# =========================================================================
# Helpers
# =========================================================================

def cid(suit: str, value: str) -> int:
    return card_id(suit, value)


H6, H7, H8, H9, H10, HU, HO, HK, HA = [cid("herz", v) for v in VALUES]
E6, E7, E8, E9, E10, EU, EO, EK, EA = [cid("ecke", v) for v in VALUES]
S6, S7, S8, S9, S10, SU, SO, SK, SA = [cid("schaufel", v) for v in VALUES]
K6, K7, K8, K9, K10, KU, KO, KK, KA = [cid("kreuz", v) for v in VALUES]

HERZ = SUIT_INDEX["herz"]
ECKE = SUIT_INDEX["ecke"]
SCHAUFEL = SUIT_INDEX["schaufel"]
KREUZ = SUIT_INDEX["kreuz"]


# =========================================================================
# Card Points
# =========================================================================

class TestCardPoints:
    """Every card's points in trump and non-trump."""

    def test_non_trump_points(self):
        expected = {"6": 0, "7": 0, "8": 0, "9": 0, "10": 10, "U": 2, "O": 3, "K": 4, "A": 11}
        for suit in SUITS:
            trump = SUIT_INDEX["ecke" if suit != "ecke" else "herz"]  # different suit
            for val, exp_pts in expected.items():
                c = cid(suit, val)
                assert points(c, trump) == exp_pts, f"{suit} {val} non-trump should be {exp_pts}"

    def test_trump_points(self):
        expected = {"6": 0, "7": 0, "8": 0, "9": 14, "10": 10, "U": 20, "O": 3, "K": 4, "A": 11}
        for suit in SUITS:
            trump = SUIT_INDEX[suit]
            for val, exp_pts in expected.items():
                c = cid(suit, val)
                assert points(c, trump) == exp_pts, f"{suit} {val} trump should be {exp_pts}"

    def test_round_card_points_152(self):
        """Card points per round (without last-trick bonus) = 152 for any trump."""
        for trump in range(4):
            assert round_card_points(trump) == 152

    def test_total_with_bonus_157(self):
        assert 152 + LAST_TRICK_BONUS == TOTAL_ROUND_POINTS == 157


# =========================================================================
# Strength Rankings
# =========================================================================

class TestStrengthRankings:
    """Non-trump: 6<7<8<9<10<U<O<K<A. Trump: 6<7<8<10<O<K<A<9<U."""

    def test_non_trump_order(self):
        order = ["6", "7", "8", "9", "10", "U", "O", "K", "A"]
        for suit in SUITS:
            trump = SUIT_INDEX["ecke" if suit != "ecke" else "herz"]
            strengths = [strength(cid(suit, v), trump) for v in order]
            for i in range(len(strengths) - 1):
                assert strengths[i] < strengths[i + 1], (
                    f"Non-trump {suit}: {order[i]}({strengths[i]}) should be < {order[i+1]}({strengths[i+1]})"
                )

    def test_trump_order(self):
        order = ["6", "7", "8", "10", "O", "K", "A", "9", "U"]
        for suit in SUITS:
            trump = SUIT_INDEX[suit]
            strengths = [strength(cid(suit, v), trump) for v in order]
            for i in range(len(strengths) - 1):
                assert strengths[i] < strengths[i + 1], (
                    f"Trump {suit}: {order[i]}({strengths[i]}) should be < {order[i+1]}({strengths[i+1]})"
                )

    def test_trump_beats_non_trump(self):
        """Any trump card has higher strength than any non-trump card."""
        for trump_suit in range(4):
            for c in ALL_CARD_IDS:
                s = strength(c, trump_suit)
                if CARD_SUIT[c] == trump_suit:
                    assert s >= 100, f"Trump card {c} should have strength >= 100"
                else:
                    assert s < 100, f"Non-trump card {c} should have strength < 100"

    def test_trump_9_and_U_are_top(self):
        """Trump 9 (Nell) and Trump U (Puur) are the two strongest cards."""
        for suit in SUITS:
            trump = SUIT_INDEX[suit]
            nell = cid(suit, "9")
            puur = cid(suit, "U")
            for val in VALUES:
                c = cid(suit, val)
                if val not in ("9", "U"):
                    assert strength(c, trump) < strength(nell, trump)
                    assert strength(c, trump) < strength(puur, trump)
            assert strength(nell, trump) < strength(puur, trump)


# =========================================================================
# Legal Moves
# =========================================================================

class TestLegalMoves:
    """Legal move rules including follow suit, undertrump, Puur privilege."""

    def test_leading_any_card(self):
        hand = [H6, E10, SA, KU]
        legal = legal_moves(hand, None, HERZ)
        assert set(legal) == set(hand)

    def test_must_follow_suit(self):
        hand = [H6, H10, E7, SA]
        legal = legal_moves(hand, HERZ, ECKE)  # herz led, ecke trump
        assert set(legal) == {H6, H10}

    def test_cant_follow_suit_anything_goes(self):
        hand = [E7, SA, KU]  # no herz
        legal = legal_moves(hand, HERZ, ECKE)  # herz led, ecke trump, no trump on table
        assert set(legal) == set(hand)

    def test_trump_led_must_play_trump(self):
        hand = [H6, H9, E7, SA]  # herz is trump, have H6 and H9
        legal = legal_moves(hand, HERZ, HERZ, [(1, H7)])  # herz led
        # Must play herz, must overtrump H7 if possible
        # H6 str=100, H7 str=101, H9 str=107, HU would be 108
        # H9 overtrumps H7, H6 does not
        assert H9 in legal
        # H6 cannot overtrump, but we have H9 which can, so H6 is excluded
        # Unless Puur privilege applies — it doesn't here since H9 isn't the Puur
        # Actually, we need to overtrump: H9 can, so only overtrumping cards are legal
        # H6 strength(trump)=100, H7 on table strength=101, H6 < 101, can't overtrump
        # Wait: H6 trump strength is 0+100=100, H7 trump strength is 1+100=101
        # H9 trump strength is 7+100=107 > 101, so it overtrumps
        # Only overtrumping cards are returned (unless Puur privilege)
        assert set(legal) == {H9}

    def test_trump_led_cant_overtrump(self):
        """If you can't overtrump, play any trump."""
        hand = [H6, H7, E7, SA]  # herz trump
        # Table has H9 (strength 107)
        legal = legal_moves(hand, HERZ, HERZ, [(1, H9)])
        # H6 (100) and H7 (101) can't beat H9 (107)
        assert set(legal) == {H6, H7}

    def test_puur_privilege(self):
        """If only the Puur can overtrump, you may play any trump."""
        hand = [H6, H7, HU, E7]  # herz trump, have Puur
        # Table has HA (strength 106) — only HU (108) can overtrump
        legal = legal_moves(hand, HERZ, HERZ, [(1, HA)])
        # Puur privilege: since only the Puur can overtrump, can play any trump
        assert set(legal) == {H6, H7, HU}

    def test_no_undertrump_when_cant_follow(self):
        """Can't undertrump unless forced."""
        hand = [H6, E7, E8]  # herz trump, can't follow schaufel
        # Schaufel led, herz trump on table: H10 (strength 103+100? no)
        # Let's say: schaufel led, herz is trump, table has H10 as trump
        # H10 trump strength = 3+100 = 103
        # H6 trump strength = 0+100 = 100 < 103, this is undertrumping
        # E7 and E8 are non-trump non-lead, they're fine
        legal = legal_moves(hand, SCHAUFEL, HERZ, [(1, S7), (2, H10)])
        # Can't follow schaufel. Have E7, E8 (non-trump) and H6 (undertrump).
        # Non-trump cards are allowed, undertrump H6 is not (unless forced).
        assert set(legal) == {E7, E8}

    def test_forced_undertrump(self):
        """Must undertrump if only option."""
        hand = [H6]  # only card is an undertrump
        legal = legal_moves(hand, SCHAUFEL, HERZ, [(1, S7), (2, H10)])
        assert set(legal) == {H6}

    def test_overtrump_allowed_when_cant_follow(self):
        """Can play higher trump when can't follow suit."""
        hand = [H6, HA, E7]  # herz trump
        # Schaufel led, H6 on table (strength 100)
        # HA strength = 106, overtrumps
        legal = legal_moves(hand, SCHAUFEL, HERZ, [(1, S7), (2, H6)])
        # Can't follow schaufel. HA overtrumps (106 > 100). E7 is non-trump.
        # H6 would be undertrump... wait, H6 strength 100 = table H6 100, not strictly greater
        # So HA is overtrump, E7 is fine, H6 is equal not over, so it's undertrump
        # Actually the table has H6 at strength 100, our H6... wait, we can't have the same card
        # Let me fix: table has H7 (101)
        legal = legal_moves(hand, SCHAUFEL, HERZ, [(1, S7), (2, H7)])
        # H6 (100) < H7 (101) = undertrump, not allowed
        # HA (106) > H7 (101) = overtrump, allowed
        # E7 = non-trump, allowed
        assert set(legal) == {HA, E7}

    def test_no_trump_on_table_free_play(self):
        """When can't follow and no trump on table, all cards legal."""
        hand = [H6, E7, KA]
        legal = legal_moves(hand, SCHAUFEL, HERZ, [(1, S7)])
        # No trump on table, can't follow schaufel: anything goes
        assert set(legal) == set(hand)

    def test_empty_hand(self):
        assert legal_moves([], HERZ, ECKE) == []


# =========================================================================
# Trick Winner
# =========================================================================

class TestTrickWinner:
    """Trick winner for various scenarios."""

    def test_same_suit_highest_wins(self):
        trick = [(0, H6), (1, H10), (2, HK), (3, HA)]
        assert trick_winner(trick, HERZ, ECKE) == 3  # HA highest non-trump

    def test_trump_beats_non_trump(self):
        trick = [(0, SA), (1, E6), (2, SK), (3, S10)]  # ecke is trump
        assert trick_winner(trick, SCHAUFEL, ECKE) == 1  # E6 is trump

    def test_higher_trump_wins(self):
        trick = [(0, H6), (1, H9), (2, HU), (3, HA)]  # herz trump
        assert trick_winner(trick, HERZ, HERZ) == 2  # HU (Puur) is highest

    def test_nell_beats_ace_in_trump(self):
        trick = [(0, HA), (1, H9)]  # herz trump
        # H9 (Nell, strength 107) > HA (strength 106)
        assert trick_winner([(0, HA), (1, H9)], HERZ, HERZ) == 1

    def test_off_suit_cannot_win(self):
        # herz led, ecke is trump. E10 is trump so it wins.
        # For a true "off-suit can't win" test: herz led, schaufel trump
        trick = [(0, H6), (1, E10), (2, K7), (3, K8)]  # herz led, schaufel trump
        # Only H6 follows suit. E10, K7, K8 are off-suit non-trump.
        assert trick_winner(trick, HERZ, SCHAUFEL) == 0  # H6 only herz card

    def test_multiple_trumps(self):
        trick = [(0, S6), (1, H7), (2, H10), (3, HU)]  # herz trump, schaufel led
        assert trick_winner(trick, SCHAUFEL, HERZ) == 3  # HU highest trump

    def test_leader_wins_if_no_one_follows(self):
        # herz led, ecke trump, but no one has herz or ecke
        trick = [(0, H6), (1, S7), (2, K8), (3, E6)]  # E6 is trump!
        # Wait, ecke is trump, so E6 is trump and beats H6
        assert trick_winner(trick, HERZ, ECKE) == 3

    def test_single_card_trick(self):
        # Edge case: shouldn't normally happen but test robustness
        trick = [(2, SA)]
        assert trick_winner(trick, SCHAUFEL, HERZ) == 2


# =========================================================================
# Scoring
# =========================================================================

class TestScoring:
    """Point calculation and last-trick bonus."""

    def test_trick_points_simple(self):
        trick = [(0, H10), (1, HA), (2, H6), (3, H7)]  # herz not trump
        assert trick_points(trick, ECKE) == 10 + 11 + 0 + 0

    def test_trick_points_with_trump(self):
        trick = [(0, H9), (1, H10)]  # herz is trump
        assert trick_points(trick, HERZ) == 14 + 10

    def test_last_trick_bonus(self):
        trick = [(0, H6), (1, E6), (2, S6), (3, K6)]
        pts_no_bonus = trick_points_with_bonus(trick, HERZ, is_last_trick=False)
        pts_bonus = trick_points_with_bonus(trick, HERZ, is_last_trick=True)
        assert pts_bonus == pts_no_bonus + 5

    def test_full_round_157_random(self):
        """Play 10000 random rounds and verify each totals 157."""
        rng = random.Random(42)

        for _ in range(10000):
            deck = list(ALL_CARD_IDS)
            rng.shuffle(deck)
            trump = rng.randint(0, 3)

            hands = [deck[i * 9:(i + 1) * 9] for i in range(4)]
            total = 0

            for trick_num in range(9):
                trick_cards = []
                for p in range(4):
                    if hands[p]:
                        card = hands[p].pop(0)
                        trick_cards.append((p, card))

                pts = trick_points(trick_cards, trump)
                if trick_num == 8:  # last trick
                    pts += LAST_TRICK_BONUS
                total += pts

            assert total == 157, f"Round total was {total}, expected 157"

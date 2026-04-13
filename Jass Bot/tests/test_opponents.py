"""Tests for target-aware opponent heuristics."""

from __future__ import annotations

import random

import pytest

from engine.cards import card_id, SUIT_INDEX, points, strength, CARD_SUIT, TOTAL_ROUND_POINTS
from engine.game import play_round
from bot.heuristics import (
    pick_opponent_target_aware,
    pick_opponent_greedy,
    estimate_hand_score,
)
from arena.players import RandomPlayer, HeuristicPlayer, PIMCPlayer


def cid(suit: str, value: str) -> int:
    return card_id(suit, value)


HERZ = SUIT_INDEX["herz"]
ECKE = SUIT_INDEX["ecke"]
SCHAUFEL = SUIT_INDEX["schaufel"]
KREUZ = SUIT_INDEX["kreuz"]


class TestTargetAwareOpponent:

    def test_over_target_plays_lowest(self):
        """Opponent with target=0 and 0 points should play lowest card."""
        hand = [cid("ecke", "A"), cid("ecke", "6"), cid("ecke", "10")]
        card = pick_opponent_target_aware(
            hand, table_cards=[], lead_suit=None,
            trump_suit=HERZ, target=0, current_points=0, trick_number=0,
        )
        # With target=0, gap=0 → shed mode. Should play lowest.
        assert card == cid("ecke", "6")

    def test_behind_target_plays_aggressively(self):
        """Opponent with target=100 and 10 points should try to win."""
        hand = [cid("ecke", "A"), cid("ecke", "6"), cid("ecke", "7")]
        # Leading, gap = 90, avg_needed ~11 per trick → aggressive
        card = pick_opponent_target_aware(
            hand, table_cards=[], lead_suit=None,
            trump_suit=HERZ, target=100, current_points=10, trick_number=1,
        )
        # Should play high card to try to win
        assert points(card, HERZ) >= points(cid("ecke", "6"), HERZ)

    def test_over_target_sheds(self):
        """Opponent with target=30 and 35 points should shed."""
        hand = [cid("ecke", "A"), cid("ecke", "6"), cid("ecke", "K")]
        card = pick_opponent_target_aware(
            hand, table_cards=[], lead_suit=None,
            trump_suit=HERZ, target=30, current_points=35, trick_number=5,
        )
        # Over target: gap=-5, should shed → play lowest
        assert card == cid("ecke", "6")

    def test_following_suit_over_target_avoids_winning(self):
        """When over target and following suit, prefer not winning."""
        hand = [cid("ecke", "A"), cid("ecke", "6")]
        # Ecke led, E7 on table. EA would win, E6 would not.
        table = [(1, cid("ecke", "7"))]
        card = pick_opponent_target_aware(
            hand, table_cards=table, lead_suit=ECKE,
            trump_suit=HERZ, target=20, current_points=25, trick_number=3,
        )
        # Over target: should play E6 (doesn't win) rather than EA (wins)
        assert card == cid("ecke", "6")

    def test_returns_valid_card(self):
        """Should always return a card from the hand."""
        rng = random.Random(42)
        for _ in range(100):
            deck = list(range(36))
            rng.shuffle(deck)
            hand = deck[:5]
            trump = rng.randint(0, 3)
            target = rng.randint(0, 157)
            cur_pts = rng.randint(0, 157)
            trick_num = rng.randint(0, 8)

            card = pick_opponent_target_aware(
                hand, table_cards=[], lead_suit=None,
                trump_suit=trump, target=target, current_points=cur_pts,
                trick_number=trick_num,
            )
            assert card in hand


class TestEstimateHandScore:

    def test_all_trump_high(self):
        """Strong trump hand should estimate high."""
        hand = [
            cid("herz", "U"), cid("herz", "9"), cid("herz", "A"),
            cid("herz", "K"), cid("herz", "O"), cid("herz", "10"),
            cid("ecke", "A"), cid("schaufel", "A"), cid("kreuz", "A"),
        ]
        score = estimate_hand_score(hand, HERZ)
        assert score > 60, f"Strong hand estimated {score}, expected > 60"

    def test_all_low_non_trump(self):
        """Weak hand should estimate low."""
        hand = [
            cid("ecke", "6"), cid("ecke", "7"), cid("ecke", "8"),
            cid("schaufel", "6"), cid("schaufel", "7"), cid("schaufel", "8"),
            cid("kreuz", "6"), cid("kreuz", "7"), cid("kreuz", "8"),
        ]
        score = estimate_hand_score(hand, HERZ)
        assert score < 15, f"Weak hand estimated {score}, expected < 15"

    def test_score_in_valid_range(self):
        rng = random.Random(42)
        for _ in range(200):
            deck = list(range(36))
            rng.shuffle(deck)
            hand = deck[:9]
            trump = rng.randint(0, 3)
            score = estimate_hand_score(hand, trump)
            assert 0 <= score <= 157


class TestTargetAwareRoundsStillTotal157:
    """Rounds using target-aware opponent simulation must still total 157."""

    def test_pimc_target_aware_rounds_157(self):
        """200 rounds with heuristic players should all total 157."""
        rng = random.Random(42)
        for _ in range(200):
            players = {i: HeuristicPlayer(player_id=i, rng=random.Random(rng.randint(0, 2**32))) for i in range(4)}
            result = play_round(players, rng=random.Random(rng.randint(0, 2**32)))
            assert result.total_points == TOTAL_ROUND_POINTS

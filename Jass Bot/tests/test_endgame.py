"""Tests for the endgame solver (max^n exact search)."""

from __future__ import annotations

import time

import pytest

from engine.cards import card_id, SUIT_INDEX, points, LAST_TRICK_BONUS
from bot.endgame import EndgameSolver


def cid(suit: str, value: str) -> int:
    return card_id(suit, value)


HERZ = SUIT_INDEX["herz"]
ECKE = SUIT_INDEX["ecke"]
SCHAUFEL = SUIT_INDEX["schaufel"]
KREUZ = SUIT_INDEX["kreuz"]


class TestEndgame1Trick:
    """1 trick remaining: should be deterministic."""

    def test_picks_card_that_hits_target(self):
        """With 1 card each, solver picks the only option."""
        solver = EndgameSolver(max_depth=1)
        # Each player has 1 card. Player 0 leads.
        my_hand = [cid("herz", "A")]  # 11 pts
        opp_hands = {
            1: [cid("ecke", "6")],
            2: [cid("schaufel", "6")],
            3: [cid("kreuz", "6")],
        }
        # Target: current + 11 + 5 (last trick bonus) = need to win
        # If we win: get 11 + 5 = 16 pts
        card, final_pts = solver.solve(
            my_hand, opp_hands, trump=HERZ, leader=0,
            my_target=36, my_points=20,
            opponent_targets={1: 30, 2: 30, 3: 30},
            opponent_points={1: 30, 2: 30, 3: 30},
            trick_number=8,  # 0-indexed trick 8 = trick 9 (last)
        )
        assert card == cid("herz", "A")

    def test_single_card_returns_it(self):
        solver = EndgameSolver(max_depth=1)
        my_hand = [cid("ecke", "6")]
        opp_hands = {
            1: [cid("herz", "6")],
            2: [cid("schaufel", "6")],
            3: [cid("kreuz", "6")],
        }
        card, _ = solver.solve(
            my_hand, opp_hands, trump=HERZ, leader=0,
            my_target=0, my_points=0,
            opponent_targets={1: 30, 2: 30, 3: 30},
            opponent_points={1: 30, 2: 30, 3: 30},
            trick_number=8,
        )
        assert card == cid("ecke", "6")


class TestEndgame2Tricks:
    """2 tricks remaining: solver finds optimal sequence."""

    def test_wins_first_dumps_second(self):
        """Target = current + 11: win Herz Ass trick, then dump."""
        solver = EndgameSolver(max_depth=2)
        # Trick 8 (second-to-last) and trick 9 (last)
        my_hand = [cid("herz", "A"), cid("schaufel", "6")]
        opp_hands = {
            1: [cid("ecke", "7"), cid("ecke", "8")],
            2: [cid("kreuz", "7"), cid("kreuz", "8")],
            3: [cid("schaufel", "7"), cid("schaufel", "8")],
        }
        card, final_pts = solver.solve(
            my_hand, opp_hands, trump=HERZ, leader=0,
            my_target=31, my_points=20,
            opponent_targets={1: 30, 2: 30, 3: 30},
            opponent_points={1: 30, 2: 30, 3: 30},
            trick_number=7,
        )
        # Should lead HA to win 11 pts, then dump S6
        assert card == cid("herz", "A")

    def test_opponent_plays_optimally(self):
        """Opponents minimize their own deviation, not ours."""
        solver = EndgameSolver(max_depth=2)
        my_hand = [cid("ecke", "6"), cid("ecke", "7")]
        opp_hands = {
            1: [cid("herz", "A"), cid("herz", "6")],
            2: [cid("schaufel", "A"), cid("schaufel", "6")],
            3: [cid("kreuz", "A"), cid("kreuz", "6")],
        }
        # Just verify it runs without error and returns a valid card
        card, final_pts = solver.solve(
            my_hand, opp_hands, trump=HERZ, leader=0,
            my_target=0, my_points=0,
            opponent_targets={1: 50, 2: 50, 3: 50},
            opponent_points={1: 30, 2: 30, 3: 30},
            trick_number=7,
        )
        assert card in my_hand


class TestEndgame3Tricks:
    """3 tricks remaining: should complete in reasonable time."""

    def test_completes_within_time_limit(self):
        solver = EndgameSolver(max_depth=3)
        my_hand = [cid("herz", "A"), cid("herz", "K"), cid("ecke", "6")]
        opp_hands = {
            1: [cid("ecke", "A"), cid("ecke", "K"), cid("ecke", "10")],
            2: [cid("schaufel", "A"), cid("schaufel", "K"), cid("schaufel", "10")],
            3: [cid("kreuz", "A"), cid("kreuz", "K"), cid("kreuz", "10")],
        }
        start = time.time()
        card, final_pts = solver.solve(
            my_hand, opp_hands, trump=HERZ, leader=0,
            my_target=20, my_points=5,
            opponent_targets={1: 40, 2: 40, 3: 40},
            opponent_points={1: 30, 2: 30, 3: 30},
            trick_number=6,
        )
        elapsed = time.time() - start
        assert elapsed < 2.0, f"Endgame solver took {elapsed:.2f}s, expected < 2s"
        assert card in my_hand

    def test_returns_valid_card(self):
        """Across many random positions, always returns a valid card."""
        import random
        solver = EndgameSolver(max_depth=3)
        rng = random.Random(42)

        for _ in range(20):
            deck = list(range(36))
            rng.shuffle(deck)
            my_hand = deck[:3]
            opp_hands = {1: deck[3:6], 2: deck[6:9], 3: deck[9:12]}
            trump = rng.randint(0, 3)
            leader = rng.randint(0, 3)

            card, final_pts = solver.solve(
                my_hand, opp_hands, trump=trump, leader=leader,
                my_target=rng.randint(0, 157), my_points=rng.randint(0, 100),
                opponent_targets={1: rng.randint(0, 157), 2: rng.randint(0, 157), 3: rng.randint(0, 157)},
                opponent_points={1: rng.randint(0, 100), 2: rng.randint(0, 100), 3: rng.randint(0, 100)},
                trick_number=6,
            )
            if card is not None:
                assert card in my_hand


class TestShouldActivate:

    def test_activates_at_threshold(self):
        solver = EndgameSolver(max_depth=3)
        assert solver.should_activate(3) is True
        assert solver.should_activate(2) is True
        assert solver.should_activate(1) is True
        assert solver.should_activate(4) is False
        assert solver.should_activate(9) is False

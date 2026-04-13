"""Tests for the game engine: full round play, player interface, arena."""

from __future__ import annotations

import random

import pytest

from engine.cards import TOTAL_ROUND_POINTS, ALL_CARD_IDS
from engine.game import play_round, deal, RoundResult
from arena.players import RandomPlayer, HeuristicPlayer


class TestDeal:
    def test_deal_9_cards_each(self):
        trump, hands = deal(random.Random(42))
        for p in range(4):
            assert len(hands[p]) == 9

    def test_deal_no_duplicates(self):
        trump, hands = deal(random.Random(42))
        all_cards = []
        for h in hands.values():
            all_cards.extend(h)
        assert len(set(all_cards)) == 36

    def test_deal_trump_valid(self):
        for seed in range(100):
            trump, _ = deal(random.Random(seed))
            assert 0 <= trump <= 3


class TestPlayRound:
    def test_round_totals_157_random(self):
        """Random players: every round must total 157."""
        rng = random.Random(42)
        for _ in range(500):
            players = {i: RandomPlayer(rng=random.Random(rng.randint(0, 2**32))) for i in range(4)}
            result = play_round(players, rng=random.Random(rng.randint(0, 2**32)))
            assert result.total_points == TOTAL_ROUND_POINTS, (
                f"Total {result.total_points} != 157\n{result}"
            )

    def test_round_totals_157_heuristic(self):
        """Heuristic players: every round must total 157."""
        rng = random.Random(123)
        for _ in range(500):
            players = {i: HeuristicPlayer(player_id=i, rng=random.Random(rng.randint(0, 2**32))) for i in range(4)}
            result = play_round(players, rng=random.Random(rng.randint(0, 2**32)))
            assert result.total_points == TOTAL_ROUND_POINTS

    def test_deviations_non_negative(self):
        rng = random.Random(42)
        players = {i: RandomPlayer(rng=random.Random(rng.randint(0, 2**32))) for i in range(4)}
        result = play_round(players, rng=random.Random(42))
        for p in range(4):
            assert result.deviations[p] >= 0

    def test_points_non_negative(self):
        rng = random.Random(42)
        for _ in range(100):
            players = {i: RandomPlayer(rng=random.Random(rng.randint(0, 2**32))) for i in range(4)}
            result = play_round(players, rng=random.Random(rng.randint(0, 2**32)))
            for p in range(4):
                assert result.points[p] >= 0

    def test_all_cards_played(self):
        """After a round, 36 cards should have been played."""
        rng = random.Random(42)
        players = {i: RandomPlayer(rng=random.Random(rng.randint(0, 2**32))) for i in range(4)}
        # We can verify indirectly: total points = 157 means all cards accounted for
        result = play_round(players, rng=random.Random(42))
        assert result.total_points == 157

    def test_predetermined_hands(self):
        """Can play with predetermined hands and trump."""
        rng = random.Random(42)
        _, hands = deal(rng)
        players = {i: RandomPlayer(rng=random.Random(42)) for i in range(4)}
        result = play_round(players, trump=0, hands=hands, rng=random.Random(42))
        assert result.total_points == 157
        assert result.trump == 0

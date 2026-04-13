"""Tests for ISMCTS implementation."""

from __future__ import annotations

import random
import time

import pytest

from engine.cards import card_id, CARD_SUIT
from engine.rules import legal_moves
from bot.ismcts import Node, ISMCTSEngine, _normalize_utility


# ===== Node tests =====

class TestNode:

    def test_ucb_unvisited_is_inf(self):
        parent = Node(action=None, player=0)
        parent.visits = 10
        child = Node(action=0, player=1, parent=parent)
        child.visits = 0
        import math
        assert child.ucb(0.7, math.log(10)) == float('inf')

    def test_ucb_calculation(self):
        """Manual UCB check: exploit + explore."""
        import math
        parent = Node(action=None, player=0)
        parent.visits = 100
        child = Node(action=0, player=1, parent=parent)
        child.visits = 20
        child.total_utility = 14.0  # avg = 0.7
        c = 0.7
        log_parent = math.log(100)
        expected = 14.0 / 20 + c * math.sqrt(log_parent / 20)
        assert abs(child.ucb(c, log_parent) - expected) < 1e-9

    def test_expand_creates_child(self):
        root = Node(action=None, player=0)
        child = root.expand_one(5, 1)
        assert 5 in root.children
        assert child.action == 5
        assert child.player == 1
        assert child.parent is root

    def test_best_child_ucb_filters_compatible(self):
        root = Node(action=None, player=0)
        root.visits = 30
        c1 = root.expand_one(1, 1)
        c1.visits = 10
        c1.total_utility = 5.0
        c2 = root.expand_one(2, 1)
        c2.visits = 10
        c2.total_utility = 8.0
        c3 = root.expand_one(3, 1)
        c3.visits = 10
        c3.total_utility = 3.0

        # Only cards 1 and 3 are compatible
        best = root.best_child_ucb([1, 3], 0.7)
        assert best is c1  # 5/10 > 3/10


class TestNormalize:

    def test_perfect_hit(self):
        assert _normalize_utility(0) == pytest.approx(157 / 157)

    def test_worst_case(self):
        assert _normalize_utility(-157) == pytest.approx(0.0)

    def test_mid(self):
        assert _normalize_utility(-78) == pytest.approx(79 / 157)


# ===== Engine tests =====

class TestISMCTSEngine:

    def test_returns_legal_move(self):
        """ISMCTS must return a legal card."""
        eng = ISMCTSEngine(time_limit_ms=200, exploration=0.7)
        rng = random.Random(42)

        hand = [card_id("herz", v) for v in ["A", "K", "O", "U", "9", "10", "8", "7", "6"]]
        trump = 0  # herz

        card, stats = eng.search(
            my_hand=hand, trump=trump, target=80, my_points=0,
            opp_points=[0, 0, 0, 0], table_cards=[], leader=0,
            trick_number=0, played_cards=[], voids={1: set(), 2: set(), 3: set()},
            rng=rng,
        )
        assert card in hand

    def test_visits_equal_iterations(self):
        """Root visits should equal number of iterations."""
        eng = ISMCTSEngine(time_limit_ms=300, exploration=0.7)
        rng = random.Random(42)

        hand = [card_id("herz", "A"), card_id("ecke", "6"), card_id("schaufel", "7")]
        trump = 0

        card, stats = eng.search(
            my_hand=hand, trump=trump, target=15, my_points=0,
            opp_points=[0, 0, 0, 0], table_cards=[], leader=0,
            trick_number=6, played_cards=list(range(24)),
            voids={1: set(), 2: set(), 3: set()}, rng=rng,
        )
        total_visits = sum(v for v, _ in stats.values())
        assert total_visits == eng.last_iterations

    def test_time_budget_respected(self):
        """Should run close to the time limit."""
        eng = ISMCTSEngine(time_limit_ms=500, exploration=0.7)
        rng = random.Random(42)

        hand = [card_id("herz", v) for v in ["A", "K", "O", "U", "9"]]
        trump = 0

        start = time.perf_counter()
        eng.search(
            my_hand=hand, trump=trump, target=50, my_points=0,
            opp_points=[0, 0, 0, 0], table_cards=[], leader=0,
            trick_number=4, played_cards=list(range(16)),
            voids={1: set(), 2: set(), 3: set()}, rng=rng,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert 400 < elapsed_ms < 800, f"Elapsed {elapsed_ms:.0f}ms, expected ~500ms"

    def test_prefers_undershoot_over_overshoot(self):
        """Given a big overshoot card vs small undershoot, prefer undershoot."""
        eng = ISMCTSEngine(time_limit_ms=500, exploration=0.7)
        rng = random.Random(42)

        # 2 cards left, trick 7 (of 9). target=50, points=48.
        # Card A: herz A (11 pts non-trump) — overshoot to 59+ (dev 9+)
        # Card B: ecke 6 (0 pts) — undershoot at 48 (dev 2)
        hand = [card_id("herz", "A"), card_id("ecke", "6")]
        trump = 2  # schaufel

        card, stats = eng.search(
            my_hand=hand, trump=trump, target=50, my_points=48,
            opp_points=[0, 50, 50, 50], table_cards=[], leader=0,
            trick_number=7, played_cards=list(range(28)),
            voids={1: set(), 2: set(), 3: set()}, rng=rng,
        )
        # ecke 6 should be preferred (undershoot by 2 vs overshoot by 9+)
        ecke_6 = card_id("ecke", "6")
        if ecke_6 in stats:
            herz_a = card_id("herz", "A")
            if herz_a in stats:
                assert stats[ecke_6][0] > stats[herz_a][0], \
                    f"ecke 6 visits {stats[ecke_6][0]} should exceed herz A visits {stats[herz_a][0]}"

    def test_single_card_returns_immediately(self):
        eng = ISMCTSEngine(time_limit_ms=2000)
        rng = random.Random(42)
        hand = [card_id("herz", "A")]
        card, stats = eng.search(
            my_hand=hand, trump=0, target=50, my_points=0,
            opp_points=[0, 0, 0, 0], table_cards=[], leader=0,
            trick_number=8, played_cards=list(range(32)),
            voids={1: set(), 2: set(), 3: set()}, rng=rng,
        )
        assert card == hand[0]


# ===== ISMCTSPlayer integration tests =====

class TestISMCTSPlayer:

    def test_full_round_no_crash(self):
        """ISMCTSPlayer completes a full 9-trick round."""
        from arena.players import ISMCTSPlayer, HeuristicPlayer
        from engine.game import play_round

        rng = random.Random(42)
        players = {
            0: ISMCTSPlayer(player_id=0, time_limit_ms=100, rng=random.Random(42)),
            1: HeuristicPlayer(player_id=1, rng=random.Random(43)),
            2: HeuristicPlayer(player_id=2, rng=random.Random(44)),
            3: HeuristicPlayer(player_id=3, rng=random.Random(45)),
        }
        result = play_round(players, rng=rng)
        assert result.total_points == 157
        assert all(d >= 0 for d in result.deviations.values())

    def test_produces_legal_moves(self):
        """All moves in a round should be legal."""
        from arena.players import ISMCTSPlayer, HeuristicPlayer
        from engine.game import play_round

        rng = random.Random(99)
        players = {
            0: ISMCTSPlayer(player_id=0, time_limit_ms=50, rng=random.Random(99)),
            1: HeuristicPlayer(player_id=1, rng=random.Random(100)),
            2: HeuristicPlayer(player_id=2, rng=random.Random(101)),
            3: HeuristicPlayer(player_id=3, rng=random.Random(102)),
        }
        # play_round validates legal moves internally
        result = play_round(players, rng=rng)
        assert result.total_points == 157

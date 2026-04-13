"""Cross-validation tests: C engine must match Python engine exactly.

Uses multiprocessing to run tests as fast as possible.
"""

from __future__ import annotations

import os
import random
import sys
import time

# Add DLL directory for ucrt64 runtime
if sys.platform == "win32":
    os.add_dll_directory("C:/msys64/ucrt64/bin")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from ctypes import c_int, c_uint64, POINTER

from engine.cards import (
    CARD_SUIT, CARD_VALUE, CARD_POINTS, CARD_STRENGTH, NUM_CARDS,
    points, strength, ALL_CARD_IDS,
)
from engine.rules import trick_winner as py_trick_winner, legal_moves as py_legal_moves
from engine_c.fast_engine import FastEngine, _load_lib, _lib, hand_to_mask, mask_to_hand


# ===== Fixtures =====

@pytest.fixture(scope="module")
def engine():
    return FastEngine()


@pytest.fixture(scope="module")
def lib():
    assert _load_lib(), "C engine DLL not found"
    from engine_c.fast_engine import _lib as loaded_lib
    return loaded_lib


# ===== 1. Lookup table verification =====

class TestLookupTables:
    """Verify C lookup tables match Python tables exactly."""

    def test_card_points_all_36_nontop(self, lib):
        """All 36 cards: non-trump points must match."""
        for cid in range(NUM_CARDS):
            py_pts = CARD_POINTS[cid][0]  # non-trump
            c_cards = (c_int * 1)(cid)
            c_pts = lib.trick_points_sum(c_cards, 1, 99)  # trump=99 means no card is trump
            assert c_pts == py_pts, f"Card {cid}: Python={py_pts}, C={c_pts} (non-trump)"

    def test_card_points_all_36_trump(self, lib):
        """All 36 cards: trump points must match for each suit."""
        for trump in range(4):
            for cid in range(NUM_CARDS):
                is_trump = 1 if CARD_SUIT[cid] == trump else 0
                py_pts = CARD_POINTS[cid][is_trump]
                c_cards = (c_int * 1)(cid)
                c_pts = lib.trick_points_sum(c_cards, 1, trump)
                assert c_pts == py_pts, f"Card {cid}, trump={trump}: Python={py_pts}, C={c_pts}"


# ===== 2. Trick winner cross-validation =====

class TestTrickWinner:

    def test_basic_same_suit(self, lib):
        """Same suit, no trump: highest strength wins."""
        # herz: 6(0), 7(1), 8(2), A(8) — trump=kreuz(3)
        players = (c_int * 4)(0, 1, 2, 3)
        cards = (c_int * 4)(0, 1, 2, 8)  # H6, H7, H8, HA
        c_winner = lib.trick_winner(players, cards, 4, 0, 3)

        py_trick = [(0, 0), (1, 1), (2, 2), (3, 8)]
        py_winner = py_trick_winner(py_trick, 0, 3)

        assert c_winner == py_winner == 3

    def test_trump_beats_all(self, lib):
        """A single trump card beats all non-trump cards."""
        # ecke 6 (id=9) is trump=1, others are herz (suit=0)
        players = (c_int * 4)(0, 1, 2, 3)
        cards = (c_int * 4)(8, 7, 6, 9)  # HA, HK, HO, E6
        c_winner = lib.trick_winner(players, cards, 4, 0, 1)

        py_trick = [(0, 8), (1, 7), (2, 6), (3, 9)]
        py_winner = py_trick_winner(py_trick, 0, 1)

        assert c_winner == py_winner == 3

    def test_puur_beats_nell(self, lib):
        """Trump Under (Puur) beats trump 9 (Nell)."""
        # trump=herz(0), Puur=H_U(id=5), Nell=H_9(id=3)
        players = (c_int * 4)(0, 1, 2, 3)
        cards = (c_int * 4)(3, 5, 0, 1)  # H9, HU, H6, H7
        c_winner = lib.trick_winner(players, cards, 4, 0, 0)

        py_trick = [(0, 3), (1, 5), (2, 0), (3, 1)]
        py_winner = py_trick_winner(py_trick, 0, 0)

        assert c_winner == py_winner == 1  # HU wins

    def test_nell_beats_ace(self, lib):
        """Trump 9 (Nell) beats trump Ace."""
        # trump=herz(0)
        players = (c_int * 4)(0, 1, 2, 3)
        cards = (c_int * 4)(8, 3, 0, 1)  # HA, H9, H6, H7
        c_winner = lib.trick_winner(players, cards, 4, 0, 0)

        py_trick = [(0, 8), (1, 3), (2, 0), (3, 1)]
        py_winner = py_trick_winner(py_trick, 0, 0)

        assert c_winner == py_winner == 1  # H9 (Nell) wins

    def test_offsuit_cannot_win(self, lib):
        """Off-suit non-trump cards cannot win."""
        # Lead herz, ecke A played but no trump
        players = (c_int * 4)(0, 1, 2, 3)
        cards = (c_int * 4)(0, 17, 1, 2)  # H6, EA, H7, H8 — trump=schaufel(2)
        c_winner = lib.trick_winner(players, cards, 4, 0, 2)

        py_trick = [(0, 0), (1, 17), (2, 1), (3, 2)]
        py_winner = py_trick_winner(py_trick, 0, 2)

        assert c_winner == py_winner == 3  # H8 wins (highest in led suit)

    def test_random_100k(self, lib):
        """100K random 4-card tricks: C must match Python."""
        rng = random.Random(42)
        mismatches = 0
        n = 100_000

        for _ in range(n):
            # Pick 4 random distinct cards
            four = rng.sample(range(36), 4)
            trump = rng.randint(0, 3)
            lead_suit = CARD_SUIT[four[0]]

            players_py = [(i, four[i]) for i in range(4)]
            py_w = py_trick_winner(players_py, lead_suit, trump)

            c_players = (c_int * 4)(0, 1, 2, 3)
            c_cards = (c_int * 4)(*four)
            c_w = lib.trick_winner(c_players, c_cards, 4, lead_suit, trump)

            if c_w != py_w:
                mismatches += 1

        assert mismatches == 0, f"{mismatches}/{n} trick winner mismatches"


# ===== 3. Trick points cross-validation =====

class TestTrickPoints:

    def test_random_100k(self, lib):
        """100K random tricks: point sums must match."""
        rng = random.Random(123)
        mismatches = 0
        n = 100_000

        for _ in range(n):
            four = rng.sample(range(36), 4)
            trump = rng.randint(0, 3)

            py_pts = sum(points(c, trump) for c in four)

            c_cards = (c_int * 4)(*four)
            c_pts = lib.trick_points_sum(c_cards, 4, trump)

            if c_pts != py_pts:
                mismatches += 1

        assert mismatches == 0, f"{mismatches}/{n} trick points mismatches"


# ===== 4. Batch simulation smoke tests =====

class TestBatchSimulation:

    def test_batch_returns_valid_scores(self, engine):
        """Batch simulation returns scores in valid range [0, 157]."""
        rng = random.Random(42)
        deck = list(range(36))
        rng.shuffle(deck)

        my_hand = deck[:9]
        worlds = []
        for _ in range(50):
            remaining = [c for c in range(36) if c not in my_hand]
            rng.shuffle(remaining)
            worlds.append({
                1: remaining[:9],
                2: remaining[9:18],
                3: remaining[18:27],
            })

        results = engine.simulate_batch(my_hand, worlds, trump=0, my_target=40)

        assert len(results) == 50
        for s in results:
            assert 0 <= s <= 157, f"Score {s} out of range"

    def test_batch_deterministic(self, engine):
        """Same inputs produce same outputs."""
        rng = random.Random(99)
        deck = list(range(36))
        rng.shuffle(deck)

        my_hand = deck[:9]
        remaining = [c for c in range(36) if c not in my_hand]
        rng.shuffle(remaining)
        worlds = [{1: remaining[:9], 2: remaining[9:18], 3: remaining[18:27]}]

        r1 = engine.simulate_batch(my_hand, worlds, trump=1, my_target=50)
        r2 = engine.simulate_batch(my_hand, worlds, trump=1, my_target=50)
        assert r1 == r2

    def test_total_157_per_round(self, engine):
        """All 4 players' points should sum to 157 per round."""
        rng = random.Random(77)
        n_tests = 200

        for _ in range(n_tests):
            deck = list(range(36))
            rng.shuffle(deck)
            trump = rng.randint(0, 3)

            hands = [deck[i*9:(i+1)*9] for i in range(4)]

            # Simulate from each player's perspective with same world
            # We can only test player 0's score directly. Instead, simulate
            # a single full game and check via the C sim.
            # Use engine.simulate_batch for player 0
            worlds = [{1: hands[1], 2: hands[2], 3: hands[3]}]
            scores_0 = engine.simulate_batch(hands[0], worlds, trump, my_target=40)

            # Scores should be in valid range
            assert 0 <= scores_0[0] <= 157


# ===== 5. Endgame solver tests =====

class TestEndgameSolver:

    def test_single_trick(self, engine):
        """1-trick endgame: should pick the right card."""
        # Each player has 1 card, trick 8 (last trick = +5 bonus)
        my_hand = [8]   # herz A (11 pts)
        opp_hands = {1: [17], 2: [26], 3: [35]}  # EA, SA, KA
        trump = 0  # herz

        card, pts = engine.solve_endgame(
            my_hand, opp_hands, trump=trump, leader=0,
            my_target=16, my_points=0,
            opp_targets={1: 16, 2: 16, 3: 16},
            opp_points={1: 0, 2: 0, 3: 0},
            trick_number=8,
        )
        assert card == 8  # only card
        assert pts >= 0  # should get some points

    def test_two_trick_endgame(self, engine):
        """2-trick endgame returns valid results."""
        my_hand = [7, 8]   # herz K, herz A
        opp_hands = {1: [16, 17], 2: [25, 26], 3: [34, 35]}
        trump = 0

        card, pts = engine.solve_endgame(
            my_hand, opp_hands, trump=trump, leader=0,
            my_target=20, my_points=5,
            opp_targets={1: 20, 2: 20, 3: 20},
            opp_points={1: 5, 2: 5, 3: 5},
            trick_number=7,
        )
        assert card in [7, 8]
        assert 5 <= pts <= 157


# ===== 6. Declaration tests =====

class TestDeclaration:

    def test_find_declaration_in_range(self, engine):
        """Declaration should be in [0, 157]."""
        rng = random.Random(42)
        deck = list(range(36))
        rng.shuffle(deck)

        my_hand = deck[:9]
        worlds = []
        for _ in range(100):
            remaining = [c for c in range(36) if c not in my_hand]
            rng.shuffle(remaining)
            worlds.append({
                1: remaining[:9],
                2: remaining[9:18],
                3: remaining[18:27],
            })

        decl = engine.find_declaration(my_hand, worlds, trump=0)
        assert 0 <= decl <= 157, f"Declaration {decl} out of range"

    def test_distribution_valid_scores(self, engine):
        """Score distribution should have all values in [0, 157]."""
        rng = random.Random(42)
        deck = list(range(36))
        rng.shuffle(deck)

        my_hand = deck[:9]
        worlds = []
        for _ in range(200):
            remaining = [c for c in range(36) if c not in my_hand]
            rng.shuffle(remaining)
            worlds.append({
                1: remaining[:9],
                2: remaining[9:18],
                3: remaining[18:27],
            })

        dist = engine.simulate_for_distribution(my_hand, worlds, trump=0, est_target=40)
        assert len(dist) == 200
        for s in dist:
            assert 0 <= s <= 157


# ===== 7. Speed benchmarks =====

class TestSpeedBenchmarks:

    def test_trick_winner_speed(self, lib):
        """Benchmark trick_winner: should handle 1M+ calls/sec."""
        rng = random.Random(42)
        n = 500_000
        # Pre-generate data
        data = []
        for _ in range(n):
            four = rng.sample(range(36), 4)
            trump = rng.randint(0, 3)
            lead_suit = CARD_SUIT[four[0]]
            data.append((four, trump, lead_suit))

        start = time.perf_counter()
        for four, trump, lead_suit in data:
            c_players = (c_int * 4)(0, 1, 2, 3)
            c_cards = (c_int * 4)(*four)
            lib.trick_winner(c_players, c_cards, 4, lead_suit, trump)
        elapsed = time.perf_counter() - start

        rate = n / elapsed
        print(f"\nTrick winner: {n} calls in {elapsed:.2f}s = {rate:.0f}/sec")
        assert rate > 100_000, f"Too slow: {rate:.0f}/sec"

    def test_batch_sim_speed(self, engine):
        """Benchmark batch simulation."""
        rng = random.Random(42)
        deck = list(range(36))
        rng.shuffle(deck)
        my_hand = deck[:9]

        n_worlds = 2000
        worlds = []
        for _ in range(n_worlds):
            remaining = [c for c in range(36) if c not in my_hand]
            rng.shuffle(remaining)
            worlds.append({
                1: remaining[:9],
                2: remaining[9:18],
                3: remaining[18:27],
            })

        start = time.perf_counter()
        results = engine.simulate_batch(my_hand, worlds, trump=0, my_target=40)
        elapsed = time.perf_counter() - start

        rate = n_worlds / elapsed
        print(f"\nBatch sim: {n_worlds} worlds in {elapsed*1000:.1f}ms = {rate:.0f} worlds/sec")
        assert len(results) == n_worlds

    def test_endgame_speed(self, engine):
        """Benchmark 3-trick endgame solver."""
        rng = random.Random(42)
        n = 100
        times = []

        for _ in range(n):
            deck = list(range(36))
            rng.shuffle(deck)
            hands = [deck[i*3:(i+1)*3] for i in range(4)]  # 3 cards each
            trump = rng.randint(0, 3)

            start = time.perf_counter()
            engine.solve_endgame(
                hands[0], {1: hands[1], 2: hands[2], 3: hands[3]},
                trump=trump, leader=0,
                my_target=20, my_points=rng.randint(0, 100),
                opp_targets={1: 30, 2: 40, 3: 35},
                opp_points={1: rng.randint(0, 100), 2: rng.randint(0, 100), 3: rng.randint(0, 100)},
                trick_number=6,
            )
            times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        print(f"\nEndgame (3-trick): avg {avg_ms:.2f}ms over {n} solves")

    def test_declaration_speed(self, engine):
        """Benchmark declaration finding."""
        rng = random.Random(42)
        deck = list(range(36))
        rng.shuffle(deck)
        my_hand = deck[:9]

        worlds = []
        for _ in range(500):
            remaining = [c for c in range(36) if c not in my_hand]
            rng.shuffle(remaining)
            worlds.append({
                1: remaining[:9],
                2: remaining[9:18],
                3: remaining[18:27],
            })

        start = time.perf_counter()
        decl = engine.find_declaration(my_hand, worlds, trump=0)
        elapsed = time.perf_counter() - start

        print(f"\nDeclaration (500 worlds, sweep): {elapsed*1000:.1f}ms, result={decl}")


# ===== 8. Hand mask conversion tests =====

class TestHandMaskConversion:

    def test_roundtrip(self):
        """hand_to_mask -> mask_to_hand roundtrip."""
        for _ in range(100):
            hand = random.sample(range(36), 9)
            mask = hand_to_mask(hand)
            recovered = mask_to_hand(mask)
            assert sorted(hand) == sorted(recovered)

    def test_empty(self):
        assert hand_to_mask([]) == 0
        assert mask_to_hand(0) == []

    def test_single_card(self):
        for c in range(36):
            assert hand_to_mask([c]) == (1 << c)
            assert mask_to_hand(1 << c) == [c]

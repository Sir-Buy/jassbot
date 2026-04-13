"""Python wrapper for the C Jass engine via ctypes.

Provides a drop-in accelerated engine for PIMC simulation and endgame solving.
Falls back gracefully if the DLL is not available.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import c_int, c_uint64, POINTER, Structure

# --- Load library ---

_lib = None
_LIB_LOADED = False

def _load_lib():
    global _lib, _LIB_LOADED
    if _LIB_LOADED:
        return _lib is not None

    lib_dir = os.path.dirname(__file__)
    if sys.platform == "win32":
        lib_path = os.path.join(lib_dir, "jass_engine.dll")
    else:
        lib_path = os.path.join(lib_dir, "jass_engine.so")

    try:
        _lib = ctypes.CDLL(lib_path)
        _lib.init_tables()
        _setup_signatures()
        _LIB_LOADED = True
        return True
    except OSError:
        _lib = None
        _LIB_LOADED = True
        return False


class EndgameResult(Structure):
    _fields_ = [("best_card", c_int), ("final_points", c_int)]


def _setup_signatures():
    """Set up ctypes function signatures."""
    _lib.init_tables.restype = None

    _lib.trick_winner.argtypes = [POINTER(c_int), POINTER(c_int), c_int, c_int, c_int]
    _lib.trick_winner.restype = c_int

    _lib.trick_points_sum.argtypes = [POINTER(c_int), c_int, c_int]
    _lib.trick_points_sum.restype = c_int

    _lib.simulate_games_batch.argtypes = [
        c_int, c_uint64, POINTER(c_uint64), c_int, c_int, POINTER(c_int), POINTER(c_int)
    ]
    _lib.simulate_games_batch.restype = None

    _lib.solve_endgame.argtypes = [
        POINTER(c_uint64), c_int, c_int, POINTER(c_int), POINTER(c_int), c_int
    ]
    _lib.solve_endgame.restype = EndgameResult

    _lib.find_best_declaration.argtypes = [
        c_uint64, POINTER(c_uint64), POINTER(c_int), c_int, c_int
    ]
    _lib.find_best_declaration.restype = c_int

    _lib.simulate_for_distribution.argtypes = [
        c_uint64, POINTER(c_uint64), POINTER(c_int), c_int, c_int, c_int, POINTER(c_int)
    ]
    _lib.simulate_for_distribution.restype = None

    _lib.rollout_game.argtypes = [
        POINTER(c_uint64),  # hands (4)
        POINTER(c_int),     # pts (4)
        POINTER(c_int),     # targets (4)
        c_int,              # trump
        c_int,              # leader
        c_int,              # trick_number
        c_int,              # total_tricks
        POINTER(c_int),     # tc_players
        POINTER(c_int),     # tc_cards
        c_int,              # n_tc
    ]
    _lib.rollout_game.restype = c_int

    _lib.evaluate_moves_batch.argtypes = [
        c_int,              # n_worlds
        c_int,              # n_candidates
        POINTER(c_int),     # candidates
        c_uint64,           # my_hand
        POINTER(c_uint64),  # opp_hands
        c_int,              # trump
        c_int,              # my_target
        POINTER(c_int),     # opp_targets
        c_int,              # my_points
        POINTER(c_int),     # opp_points
        c_int,              # leader
        c_int,              # trick_number
        c_int,              # total_tricks
        POINTER(c_int),     # table_players
        POINTER(c_int),     # table_cards
        c_int,              # n_table
        POINTER(c_int),     # results
    ]
    _lib.evaluate_moves_batch.restype = None

    # Single-agent optimal solver
    _lib.solve_single_agent.argtypes = [
        c_uint64,           # my_hand
        POINTER(c_uint64),  # opp_hands (3)
        c_int,              # trump
        c_int,              # target
        c_int,              # leader
        POINTER(c_int),     # points_so_far (4)
        c_int,              # trick_number
        c_int,              # total_tricks
        POINTER(c_int),     # opp_targets (3)
    ]
    _lib.solve_single_agent.restype = c_int

    _lib.optimal_declaration.argtypes = [
        c_uint64,           # my_hand
        POINTER(c_uint64),  # opp_hands (n*3)
        POINTER(c_int),     # opp_targets (n*3)
        c_int,              # n_worlds
        c_int,              # trump
    ]
    _lib.optimal_declaration.restype = c_int

    _lib.optimal_play.argtypes = [
        c_int,              # n_worlds
        c_int,              # n_candidates
        POINTER(c_int),     # candidates
        c_uint64,           # my_hand
        POINTER(c_uint64),  # opp_hands
        c_int,              # trump
        c_int,              # my_target
        POINTER(c_int),     # opp_targets
        c_int,              # my_points
        POINTER(c_int),     # opp_points
        c_int,              # leader
        c_int,              # trick_number
        c_int,              # total_tricks
        POINTER(c_int),     # table_players
        POINTER(c_int),     # table_cards
        c_int,              # n_table
        POINTER(c_int),     # results
    ]
    _lib.optimal_play.restype = None


# --- Helper: convert hand list to bitmask ---

def hand_to_mask(cards: list[int]) -> int:
    mask = 0
    for c in cards:
        mask |= (1 << c)
    return mask


def mask_to_hand(mask: int) -> list[int]:
    cards = []
    for i in range(36):
        if mask & (1 << i):
            cards.append(i)
    return cards


# --- Public API ---

class FastEngine:
    """C-accelerated game engine. Drop-in for Python engine."""

    def __init__(self):
        if not _load_lib():
            raise OSError("C engine library not found")

    def simulate_batch(
        self,
        my_hand: list[int],
        worlds: list[dict[int, list[int]]],
        trump: int,
        my_target: int,
        opp_targets_list: list[dict[int, int]] | None = None,
    ) -> list[int]:
        """Run N game simulations entirely in C. Returns list of final scores."""
        n = len(worlds)
        my_mask = c_uint64(hand_to_mask(my_hand))

        # Pack opponent hands as flat array of bitmasks
        opp_arr = (c_uint64 * (n * 3))()
        opp_tgt_arr = (c_int * (n * 3))()

        for w, world in enumerate(worlds):
            base = w * 3
            for oi, p in enumerate([1, 2, 3]):
                opp_arr[base + oi] = hand_to_mask(world[p])
                if opp_targets_list is not None:
                    opp_tgt_arr[base + oi] = opp_targets_list[w][p]
                else:
                    opp_tgt_arr[base + oi] = 40  # default

        results = (c_int * n)()
        _lib.simulate_games_batch(n, my_mask, opp_arr, trump, my_target, opp_tgt_arr, results)
        return list(results)

    def solve_endgame(
        self,
        my_hand: list[int],
        opp_hands: dict[int, list[int]],
        trump: int,
        leader: int,
        my_target: int,
        my_points: int,
        opp_targets: dict[int, int],
        opp_points: dict[int, int],
        trick_number: int,
    ) -> tuple[int | None, int]:
        """Exact endgame solve in C."""
        hands = (c_uint64 * 4)(
            hand_to_mask(my_hand),
            hand_to_mask(opp_hands.get(1, [])),
            hand_to_mask(opp_hands.get(2, [])),
            hand_to_mask(opp_hands.get(3, [])),
        )
        targets = (c_int * 4)(
            my_target,
            opp_targets.get(1, 40),
            opp_targets.get(2, 40),
            opp_targets.get(3, 40),
        )
        points = (c_int * 4)(
            my_points,
            opp_points.get(1, 0),
            opp_points.get(2, 0),
            opp_points.get(3, 0),
        )

        res = _lib.solve_endgame(hands, leader, trump, targets, points, trick_number)
        card = res.best_card if res.best_card >= 0 else None
        return card, res.final_points

    def find_declaration(
        self,
        my_hand: list[int],
        worlds: list[dict[int, list[int]]],
        trump: int,
        opp_targets_list: list[dict[int, int]] | None = None,
    ) -> int:
        """Find best declaration via C sweep."""
        n = len(worlds)
        my_mask = c_uint64(hand_to_mask(my_hand))

        opp_arr = (c_uint64 * (n * 3))()
        opp_tgt_arr = (c_int * (n * 3))()

        for w, world in enumerate(worlds):
            base = w * 3
            for oi, p in enumerate([1, 2, 3]):
                opp_arr[base + oi] = hand_to_mask(world[p])
                if opp_targets_list is not None:
                    opp_tgt_arr[base + oi] = opp_targets_list[w][p]
                else:
                    opp_tgt_arr[base + oi] = 40

        return _lib.find_best_declaration(my_mask, opp_arr, opp_tgt_arr, n, trump)

    def rollout(
        self,
        hands: dict[int, list[int]],
        points: list[int],
        targets: list[int],
        trump: int,
        leader: int,
        trick_number: int,
        total_tricks: int = 9,
        trick_cards: list[tuple[int, int]] | None = None,
    ) -> int:
        """Single heuristic rollout from mid-game state. Returns player 0's final points."""
        h = (c_uint64 * 4)(
            hand_to_mask(hands.get(0, [])),
            hand_to_mask(hands.get(1, [])),
            hand_to_mask(hands.get(2, [])),
            hand_to_mask(hands.get(3, [])),
        )
        pts = (c_int * 4)(*points)
        tgts = (c_int * 4)(*targets)

        tc = trick_cards or []
        n_tc = len(tc)
        tcp = (c_int * 4)()
        tcc = (c_int * 4)()
        for i, (p, c) in enumerate(tc):
            tcp[i] = p
            tcc[i] = c

        return _lib.rollout_game(h, pts, tgts, trump, leader, trick_number,
                                 total_tricks, tcp, tcc, n_tc)

    def evaluate_moves(
        self,
        candidates: list[int],
        my_hand: list[int],
        worlds: list[dict[int, list[int]]],
        trump: int,
        my_target: int,
        my_points: int,
        opp_targets_list: list[dict[int, int]] | None = None,
        opp_points_list: list[dict[int, int]] | None = None,
        leader: int = 0,
        trick_number: int = 0,
        total_tricks: int = 9,
        table_cards: list[tuple[int, int]] | None = None,
    ) -> dict[int, float]:
        """Evaluate candidate moves via C batch simulation.

        Returns {card_id: avg_utility} for each candidate.
        """
        n = len(worlds)
        nc = len(candidates)
        my_mask = c_uint64(hand_to_mask(my_hand))

        cand_arr = (c_int * nc)(*candidates)

        opp_arr = (c_uint64 * (n * 3))()
        opp_tgt_arr = (c_int * (n * 3))()
        opp_pts_arr = (c_int * (n * 3))()

        for w, world in enumerate(worlds):
            base = w * 3
            for oi, p in enumerate([1, 2, 3]):
                opp_arr[base + oi] = hand_to_mask(world[p])
                if opp_targets_list is not None:
                    opp_tgt_arr[base + oi] = opp_targets_list[w].get(p, 40)
                else:
                    opp_tgt_arr[base + oi] = 40
                if opp_points_list is not None:
                    opp_pts_arr[base + oi] = opp_points_list[w].get(p, 0)

        # Table cards
        tc = table_cards or []
        n_table = len(tc)
        tbl_players = (c_int * 4)()
        tbl_cards = (c_int * 4)()
        for i, (p, c) in enumerate(tc):
            tbl_players[i] = p
            tbl_cards[i] = c

        results = (c_int * nc)()
        _lib.evaluate_moves_batch(
            n, nc, cand_arr, my_mask, opp_arr, trump, my_target,
            opp_tgt_arr, my_points, opp_pts_arr, leader, trick_number,
            total_tricks, tbl_players, tbl_cards, n_table, results,
        )

        return {candidates[i]: results[i] / n for i in range(nc)}

    def simulate_for_distribution(
        self,
        my_hand: list[int],
        worlds: list[dict[int, list[int]]],
        trump: int,
        est_target: int,
        opp_targets_list: list[dict[int, int]] | None = None,
    ) -> list[int]:
        """Simulate all worlds once, return score distribution."""
        n = len(worlds)
        my_mask = c_uint64(hand_to_mask(my_hand))

        opp_arr = (c_uint64 * (n * 3))()
        opp_tgt_arr = (c_int * (n * 3))()

        for w, world in enumerate(worlds):
            base = w * 3
            for oi, p in enumerate([1, 2, 3]):
                opp_arr[base + oi] = hand_to_mask(world[p])
                if opp_targets_list is not None:
                    opp_tgt_arr[base + oi] = opp_targets_list[w][p]
                else:
                    opp_tgt_arr[base + oi] = 40

        results = (c_int * n)()
        _lib.simulate_for_distribution(my_mask, opp_arr, opp_tgt_arr, n, trump, est_target, results)
        return list(results)

    # ── Single-Agent Optimal Solver ──────────────────────────────────

    def optimal_declaration(
        self,
        my_hand: list[int],
        worlds: list[dict[int, list[int]]],
        trump: int,
        opp_targets_list: list[dict[int, int]] | None = None,
    ) -> int:
        """Find the mathematically optimal declaration via tree search.

        Solves each world with a single-agent backward induction solver.
        Branches only on our legal moves; opponents play pick_card heuristic.
        Uses coarse+fine sweep to find the target minimizing avg deviation.
        """
        n = len(worlds)
        my_mask = c_uint64(hand_to_mask(my_hand))

        opp_arr = (c_uint64 * (n * 3))()
        opp_tgt_arr = (c_int * (n * 3))()

        for w, world in enumerate(worlds):
            base = w * 3
            for oi, p in enumerate([1, 2, 3]):
                opp_arr[base + oi] = hand_to_mask(world[p])
                if opp_targets_list is not None:
                    opp_tgt_arr[base + oi] = opp_targets_list[w].get(p, 40)
                else:
                    opp_tgt_arr[base + oi] = 40

        return _lib.optimal_declaration(my_mask, opp_arr, opp_tgt_arr, n, trump)

    def optimal_play(
        self,
        candidates: list[int],
        my_hand: list[int],
        worlds: list[dict[int, list[int]]],
        trump: int,
        my_target: int,
        my_points: int,
        opp_targets_list: list[dict[int, int]] | None = None,
        opp_points_list: list[dict[int, int]] | None = None,
        leader: int = 0,
        trick_number: int = 0,
        total_tricks: int = 9,
        table_cards: list[tuple[int, int]] | None = None,
    ) -> dict[int, float]:
        """Optimal move selection via single-agent tree search.

        Same interface as evaluate_moves but uses exact backward induction
        instead of heuristic rollout. Returns {card_id: avg_utility}.
        """
        n = len(worlds)
        nc = len(candidates)
        my_mask = c_uint64(hand_to_mask(my_hand))

        cand_arr = (c_int * nc)(*candidates)

        opp_arr = (c_uint64 * (n * 3))()
        opp_tgt_arr = (c_int * (n * 3))()
        opp_pts_arr = (c_int * (n * 3))()

        for w, world in enumerate(worlds):
            base = w * 3
            for oi, p in enumerate([1, 2, 3]):
                opp_arr[base + oi] = hand_to_mask(world[p])
                if opp_targets_list is not None:
                    opp_tgt_arr[base + oi] = opp_targets_list[w].get(p, 40)
                else:
                    opp_tgt_arr[base + oi] = 40
                if opp_points_list is not None:
                    opp_pts_arr[base + oi] = opp_points_list[w].get(p, 0)

        tc = table_cards or []
        n_table = len(tc)
        tbl_players = (c_int * 4)()
        tbl_cards = (c_int * 4)()
        for i, (p, c) in enumerate(tc):
            tbl_players[i] = p
            tbl_cards[i] = c

        results = (c_int * nc)()
        _lib.optimal_play(
            n, nc, cand_arr, my_mask, opp_arr, trump, my_target,
            opp_tgt_arr, my_points, opp_pts_arr, leader, trick_number,
            total_tricks, tbl_players, tbl_cards, n_table, results,
        )

        return {candidates[i]: results[i] / n for i in range(nc)}


# --- Quick availability check ---

def is_available() -> bool:
    """Check if C engine is available."""
    return _load_lib()

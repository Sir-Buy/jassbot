"""Integration tests for Phase 3: belief tracking + endgame solver + full rounds."""

from __future__ import annotations

import random

import pytest

from engine.cards import TOTAL_ROUND_POINTS
from engine.game import play_round
from arena.players import PIMCPlayer, HeuristicPlayer, RandomPlayer


class TestFullRoundWithBeliefs:
    """Full rounds with belief tracking enabled."""

    def test_belief_tracking_no_crash(self):
        """10 rounds with belief tracking: no crashes, totals 157."""
        rng = random.Random(42)
        for _ in range(10):
            players = {
                0: PIMCPlayer(
                    player_id=0, num_worlds=5, ansage_worlds=10,
                    declaration="median", use_target_aware_opponents=True,
                    use_belief_tracking=True,
                    rng=random.Random(rng.randint(0, 2**32)),
                ),
                1: HeuristicPlayer(player_id=1, rng=random.Random(rng.randint(0, 2**32))),
                2: HeuristicPlayer(player_id=2, rng=random.Random(rng.randint(0, 2**32))),
                3: HeuristicPlayer(player_id=3, rng=random.Random(rng.randint(0, 2**32))),
            }
            result = play_round(players, rng=random.Random(rng.randint(0, 2**32)))
            assert result.total_points == TOTAL_ROUND_POINTS


class TestFullRoundWithEndgame:
    """Full rounds with endgame solver enabled."""

    def test_endgame_solver_no_crash(self):
        """10 rounds with endgame solver depth 2: no crashes, totals 157."""
        rng = random.Random(42)
        for _ in range(10):
            players = {
                0: PIMCPlayer(
                    player_id=0, num_worlds=5, ansage_worlds=10,
                    declaration="median", use_target_aware_opponents=True,
                    endgame_depth=2,
                    rng=random.Random(rng.randint(0, 2**32)),
                ),
                1: HeuristicPlayer(player_id=1, rng=random.Random(rng.randint(0, 2**32))),
                2: HeuristicPlayer(player_id=2, rng=random.Random(rng.randint(0, 2**32))),
                3: HeuristicPlayer(player_id=3, rng=random.Random(rng.randint(0, 2**32))),
            }
            result = play_round(players, rng=random.Random(rng.randint(0, 2**32)))
            assert result.total_points == TOTAL_ROUND_POINTS


class TestFullPhase3:
    """Full Phase 3 config: beliefs + endgame + median + target-aware."""

    def test_full_phase3_no_crash(self):
        """10 rounds with all Phase 3 features: no crashes, totals 157."""
        rng = random.Random(42)
        for _ in range(10):
            players = {
                0: PIMCPlayer(
                    player_id=0, num_worlds=5, ansage_worlds=10,
                    declaration="median", use_target_aware_opponents=True,
                    use_belief_tracking=True, endgame_depth=3,
                    rng=random.Random(rng.randint(0, 2**32)),
                ),
                1: HeuristicPlayer(player_id=1, rng=random.Random(rng.randint(0, 2**32))),
                2: HeuristicPlayer(player_id=2, rng=random.Random(rng.randint(0, 2**32))),
                3: HeuristicPlayer(player_id=3, rng=random.Random(rng.randint(0, 2**32))),
            }
            result = play_round(players, rng=random.Random(rng.randint(0, 2**32)))
            assert result.total_points == TOTAL_ROUND_POINTS

    def test_all_features_configurable(self):
        """Verify all feature flags work independently."""
        rng = random.Random(42)
        configs = [
            {"declaration": "sweep", "use_target_aware_opponents": False, "use_belief_tracking": False, "endgame_depth": 0},
            {"declaration": "median", "use_target_aware_opponents": False, "use_belief_tracking": False, "endgame_depth": 0},
            {"declaration": "median", "use_target_aware_opponents": True, "use_belief_tracking": False, "endgame_depth": 0},
            {"declaration": "median", "use_target_aware_opponents": True, "use_belief_tracking": True, "endgame_depth": 0},
            {"declaration": "median", "use_target_aware_opponents": True, "use_belief_tracking": False, "endgame_depth": 2},
            {"declaration": "median", "use_target_aware_opponents": True, "use_belief_tracking": True, "endgame_depth": 3},
        ]
        for cfg in configs:
            players = {
                0: PIMCPlayer(
                    player_id=0, num_worlds=3, ansage_worlds=5,
                    rng=random.Random(rng.randint(0, 2**32)),
                    **cfg,
                ),
                1: HeuristicPlayer(player_id=1, rng=random.Random(rng.randint(0, 2**32))),
                2: HeuristicPlayer(player_id=2, rng=random.Random(rng.randint(0, 2**32))),
                3: HeuristicPlayer(player_id=3, rng=random.Random(rng.randint(0, 2**32))),
            }
            result = play_round(players, rng=random.Random(rng.randint(0, 2**32)))
            assert result.total_points == TOTAL_ROUND_POINTS, f"Failed with config: {cfg}"

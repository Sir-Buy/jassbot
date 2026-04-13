"""Tests for Bayesian belief tracking over opponent declarations."""

from __future__ import annotations

import random
import math

import pytest

from engine.cards import card_id, SUIT_INDEX, points
from bot.belief_tracker import BeliefTracker


HERZ = SUIT_INDEX["herz"]
ECKE = SUIT_INDEX["ecke"]


def cid(suit: str, value: str) -> int:
    return card_id(suit, value)


class TestPrior:

    def test_prior_sums_to_one(self):
        bt = BeliefTracker()
        bt.init_round(HERZ)
        for p in [1, 2, 3]:
            total = sum(bt.beliefs[p])
            assert abs(total - 1.0) < 1e-10

    def test_prior_covers_full_range(self):
        bt = BeliefTracker()
        bt.init_round(HERZ)
        belief = bt.beliefs[1]
        assert len(belief) == 158  # 0..157
        # All probabilities are non-negative
        assert all(p >= 0 for p in belief)
        # Some probability at extremes (even if very small)
        assert belief[0] > 0
        assert belief[157] > 0

    def test_prior_peaked_around_mean(self):
        bt = BeliefTracker()
        bt.init_round(HERZ)
        belief = bt.beliefs[1]
        # Max probability should be near 39 (the prior mean)
        max_idx = belief.index(max(belief))
        assert 30 <= max_idx <= 50


class TestUpdates:

    def test_shedding_shifts_beliefs_downward(self):
        bt = BeliefTracker()
        bt.init_round(HERZ)

        mean_before = bt.get_expected_declaration(1)

        # Opponent plays a low card (shedding signal) multiple times
        for _ in range(3):
            bt.update(
                player=1,
                action=cid("ecke", "6"),  # 0-point card, shedding
                player_current_points=30,
                trick_number=3,
                table_cards_before=[(0, cid("ecke", "A"))],  # high card on table
                lead_suit=ECKE,
                trump=HERZ,
            )

        mean_after = bt.get_expected_declaration(1)
        # Shedding → declaration likely lower (at or over target)
        assert mean_after < mean_before, (
            f"After shedding, expected decl should decrease: {mean_before:.1f} → {mean_after:.1f}"
        )

    def test_grabbing_shifts_beliefs_upward(self):
        bt = BeliefTracker()
        bt.init_round(HERZ)

        mean_before = bt.get_expected_declaration(1)

        # Opponent trumps in on a valuable trick (grabbing signal)
        for _ in range(3):
            bt.update(
                player=1,
                action=cid("herz", "U"),  # trump Puur, aggressive
                player_current_points=10,
                trick_number=2,
                table_cards_before=[(0, cid("ecke", "A"))],
                lead_suit=ECKE,
                trump=HERZ,
            )

        mean_after = bt.get_expected_declaration(1)
        assert mean_after > mean_before, (
            f"After grabbing, expected decl should increase: {mean_before:.1f} → {mean_after:.1f}"
        )

    def test_multiple_updates_narrow_distribution(self):
        bt = BeliefTracker()
        bt.init_round(HERZ)

        std_before = bt.get_std(1)

        # Multiple consistent shedding signals
        for i in range(5):
            bt.update(
                player=1,
                action=cid("ecke", "6"),
                player_current_points=30 + i * 5,
                trick_number=i,
                table_cards_before=[(0, cid("ecke", "10"))],
                lead_suit=ECKE,
                trump=HERZ,
            )

        std_after = bt.get_std(1)
        assert std_after < std_before, (
            f"Std should decrease with updates: {std_before:.1f} → {std_after:.1f}"
        )

    def test_neutral_action_minimal_change(self):
        bt = BeliefTracker()
        bt.init_round(HERZ)

        mean_before = bt.get_expected_declaration(1)

        # Play a mid card following suit — neutral signal
        bt.update(
            player=1,
            action=cid("ecke", "8"),  # 0-point following suit
            player_current_points=20,
            trick_number=1,
            table_cards_before=[(0, cid("ecke", "7"))],
            lead_suit=ECKE,
            trump=HERZ,
        )

        mean_after = bt.get_expected_declaration(1)
        # Should barely change
        assert abs(mean_after - mean_before) < 5.0


class TestQueries:

    def test_declaration_range_contains_mass(self):
        bt = BeliefTracker()
        bt.init_round(HERZ)
        low, high = bt.get_declaration_range(1, confidence=0.8)
        belief = bt.beliefs[1]
        mass = sum(belief[low:high + 1])
        assert mass >= 0.79, f"Range [{low}, {high}] contains {mass:.3f}, expected >= 0.8"

    def test_sample_declaration_in_range(self):
        bt = BeliefTracker()
        bt.init_round(HERZ)
        rng = random.Random(42)
        for _ in range(100):
            d = bt.sample_declaration(1, rng)
            assert 0 <= d <= 157

    def test_beliefs_dont_crash_trick_1(self):
        """Edge case: minimal info at start of round."""
        bt = BeliefTracker()
        bt.init_round(HERZ)
        bt.update(
            player=1,
            action=cid("ecke", "7"),
            player_current_points=0,
            trick_number=0,
            table_cards_before=[],
            lead_suit=None,
            trump=HERZ,
        )
        # Should not crash, beliefs should still be valid
        total = sum(bt.beliefs[1])
        assert abs(total - 1.0) < 1e-10
        assert 0 <= bt.get_expected_declaration(1) <= 157

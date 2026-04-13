"""Tests for the distribution-based declaration model."""

from __future__ import annotations

import random

import pytest

from engine.cards import card_id, SUIT_INDEX, points, CARD_SUIT
from bot.declaration import DeclarationModel


HERZ = SUIT_INDEX["herz"]
ECKE = SUIT_INDEX["ecke"]
SCHAUFEL = SUIT_INDEX["schaufel"]
KREUZ = SUIT_INDEX["kreuz"]


def cid(suit: str, value: str) -> int:
    return card_id(suit, value)


class TestDeclarationModel:

    def test_distribution_length_matches_num_worlds(self):
        hand = [cid("herz", v) for v in ["6", "7", "8", "9", "10", "U", "O", "K", "A"]]
        model = DeclarationModel()
        info = model.declare_with_info(hand, HERZ, num_worlds=50, rng=random.Random(42))
        assert len(info["distribution"]) == 50

    def test_distribution_is_sorted(self):
        hand = [cid("herz", v) for v in ["6", "7", "8", "9", "10", "U", "O", "K", "A"]]
        model = DeclarationModel()
        info = model.declare_with_info(hand, HERZ, num_worlds=100, rng=random.Random(42))
        assert info["distribution"] == sorted(info["distribution"])

    def test_calibrated_percentile_is_picked(self):
        """Declaration should use calibrated percentile + offset."""
        hand = [cid("herz", v) for v in ["6", "7", "8", "9", "10", "U", "O", "K", "A"]]
        model = DeclarationModel()  # defaults: percentile=50, offset=0
        info = model.declare_with_info(hand, HERZ, num_worlds=100, rng=random.Random(42))
        dist = info["distribution"]
        pct_idx = max(0, min(len(dist) - 1, int(len(dist) * 50 / 100)))
        expected = max(0, min(157, dist[pct_idx] + 0))
        assert info["declaration"] == expected
        # Median field should still be the actual median
        assert info["median"] == dist[len(dist) // 2]

    def test_high_trump_hand_declares_high(self):
        """Hand full of high trump cards should declare > 100."""
        # All herz (trump): U(20), 9(14), A(11), K(4), O(3), 10(10) = 62 trump pts
        # Plus some extras
        hand = [
            cid("herz", "U"), cid("herz", "9"), cid("herz", "A"),
            cid("herz", "K"), cid("herz", "O"), cid("herz", "10"),
            cid("ecke", "A"), cid("schaufel", "A"), cid("kreuz", "A"),
        ]
        model = DeclarationModel()
        decl = model.declare(hand, HERZ, num_worlds=100, rng=random.Random(42))
        assert decl > 100, f"Strong trump hand declared {decl}, expected > 100"

    def test_low_non_trump_hand_declares_low(self):
        """Hand of all low non-trump cards should declare < 20."""
        hand = [
            cid("ecke", "6"), cid("ecke", "7"), cid("ecke", "8"),
            cid("schaufel", "6"), cid("schaufel", "7"), cid("schaufel", "8"),
            cid("kreuz", "6"), cid("kreuz", "7"), cid("kreuz", "8"),
        ]
        model = DeclarationModel()
        decl = model.declare(hand, HERZ, num_worlds=100, rng=random.Random(42))
        assert decl < 20, f"Weak hand declared {decl}, expected < 20"

    def test_empty_hand(self):
        model = DeclarationModel()
        info = model.declare_with_info([], HERZ)
        assert info["declaration"] == 0
        assert info["distribution"] == []

    def test_diagnostic_fields_present(self):
        hand = [cid("herz", v) for v in ["6", "7", "8", "9", "10", "U", "O", "K", "A"]]
        model = DeclarationModel()
        info = model.declare_with_info(hand, HERZ, num_worlds=50, rng=random.Random(42))
        for key in ["declaration", "median", "mean", "std", "min", "max", "distribution"]:
            assert key in info, f"Missing key: {key}"
        assert info["min"] <= info["median"] <= info["max"]
        assert info["min"] <= info["mean"] <= info["max"]
        assert info["std"] >= 0

    def test_target_aware_mode(self):
        """Should work with target-aware opponents too."""
        hand = [cid("herz", v) for v in ["6", "7", "8", "9", "10", "U", "O", "K", "A"]]
        model = DeclarationModel(use_target_aware_opponents=True)
        info = model.declare_with_info(hand, HERZ, num_worlds=50, rng=random.Random(42))
        assert len(info["distribution"]) == 50
        assert 0 <= info["declaration"] <= 157

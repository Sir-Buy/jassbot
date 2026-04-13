"""Tests for neural network and neural players."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import pytest

from bot.neural_net import DifferenzlerNet, FEATURE_SIZE
from bot.neural_player import encode_state_tensor


class TestEncoding:

    def test_shape(self):
        feat = torch.zeros(FEATURE_SIZE)
        encode_state_tensor(feat, [0, 1, 2], set(), [], 0, [0, 1, 2],
                            0, 0, [0, 0, 0], 40, 0, {1: set(), 2: set(), 3: set()})
        assert feat.shape == (FEATURE_SIZE,)

    def test_deterministic(self):
        f1 = torch.zeros(FEATURE_SIZE)
        f2 = torch.zeros(FEATURE_SIZE)
        args = ([0, 5, 10], {15, 20}, [(1, 25)], 0, [0, 5, 10],
                3, 30, [20, 25, 15], 40, 1, {1: {2}, 2: set(), 3: set()})
        encode_state_tensor(f1, *args)
        encode_state_tensor(f2, *args)
        assert torch.equal(f1, f2)

    def test_no_nan_inf(self):
        rng = random.Random(42)
        for _ in range(100):
            hand = rng.sample(range(36), 5)
            played = set(rng.sample([c for c in range(36) if c not in hand], 10))
            trump = rng.randint(0, 3)
            feat = torch.zeros(FEATURE_SIZE)
            encode_state_tensor(feat, hand, played, [], trump, hand,
                                rng.randint(0, 8), rng.randint(0, 100),
                                [rng.randint(0, 80)] * 3, rng.randint(0, 157),
                                0, {1: set(), 2: set(), 3: set()})
            assert not torch.isnan(feat).any()
            assert not torch.isinf(feat).any()

    def test_legal_mask_position(self):
        feat = torch.zeros(FEATURE_SIZE)
        legal = [5, 10, 20]
        encode_state_tensor(feat, [5, 10, 20, 30], set(), [], 0, legal,
                            0, 0, [0, 0, 0], 40, 0, {1: set(), 2: set(), 3: set()})
        for c in legal:
            assert feat[144 + c] == 1.0
        assert feat[144 + 0] == 0.0


class TestModel:

    def test_forward_no_nan(self):
        model = DifferenzlerNet()
        model.eval()
        x = torch.randn(4, FEATURE_SIZE)
        with torch.no_grad():
            logits, values = model(x)
        assert not torch.isnan(logits).any()
        assert not torch.isnan(values).any()

    def test_output_shapes(self):
        model = DifferenzlerNet()
        model.eval()
        x = torch.randn(8, FEATURE_SIZE)
        with torch.no_grad():
            logits, values = model(x)
        assert logits.shape == (8, 36)
        assert values.shape == (8, 1)

    def test_legal_mask(self):
        model = DifferenzlerNet()
        model.eval()
        x = torch.randn(1, FEATURE_SIZE)
        mask = torch.zeros(1, 36, dtype=torch.bool)
        mask[0, [0, 5, 10]] = True
        with torch.no_grad():
            logits, _ = model(x, mask)
        assert logits[0, 1].item() < -1e8
        assert logits[0, 0].item() > -1e8

    def test_save_load_identical(self, tmp_path):
        model = DifferenzlerNet()
        model.eval()
        x = torch.randn(2, FEATURE_SIZE)
        with torch.no_grad():
            out1, v1 = model(x)

        path = str(tmp_path / "test_model.pt")
        torch.save(model.state_dict(), path)

        model2 = DifferenzlerNet()
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()
        with torch.no_grad():
            out2, v2 = model2(x)

        assert torch.allclose(out1, out2)
        assert torch.allclose(v1, v2)

    def test_softmax_sums_to_one(self):
        model = DifferenzlerNet()
        model.eval()
        x = torch.randn(1, FEATURE_SIZE)
        mask = torch.ones(1, 36, dtype=torch.bool)
        with torch.no_grad():
            logits, _ = model(x, mask)
        probs = torch.softmax(logits, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestPlayers:

    @pytest.fixture
    def model_path(self):
        p = "data/model_poc.pt"
        if not os.path.exists(p):
            pytest.skip("No trained model")
        return p

    def test_direct_neural_round(self, model_path):
        from bot.neural_player import DirectNeuralPlayer
        from arena.players import HeuristicPlayer
        from engine.game import play_round

        players = {
            0: DirectNeuralPlayer(model_path, rng=random.Random(42)),
            1: HeuristicPlayer(player_id=1, rng=random.Random(43)),
            2: HeuristicPlayer(player_id=2, rng=random.Random(44)),
            3: HeuristicPlayer(player_id=3, rng=random.Random(45)),
        }
        result = play_round(players, rng=random.Random(42))
        assert result.total_points == 157

    def test_neural_pimc_round(self, model_path):
        from bot.neural_player import NeuralPIMCPlayer
        from arena.players import HeuristicPlayer
        from engine.game import play_round

        players = {
            0: NeuralPIMCPlayer(model_path, num_worlds=10, rng=random.Random(42)),
            1: HeuristicPlayer(player_id=1, rng=random.Random(43)),
            2: HeuristicPlayer(player_id=2, rng=random.Random(44)),
            3: HeuristicPlayer(player_id=3, rng=random.Random(45)),
        }
        result = play_round(players, rng=random.Random(42))
        assert result.total_points == 157

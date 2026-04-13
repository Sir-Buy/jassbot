"""Neural network-based players for Differenzler.

Three player types:
  - DirectNeuralPlayer: pure network, no search
  - NeuralPIMCPlayer: PIMC with neural rollouts (batched)
  - ValueEnhancedPIMCPlayer: C engine rollouts + neural value at cutoff
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

from engine.cards import CARD_SUIT, NUM_CARDS, NUM_VALUES, points as card_points
from engine.rules import legal_moves
from bot.neural_net import DifferenzlerNet, FEATURE_SIZE
from bot.declaration import DeclarationModel
from bot.belief_tracker import BeliefTracker

# ===== Model loading =====

_model_cache: dict[str, torch.nn.Module] = {}


def load_model(path: str) -> torch.nn.Module:
    if path in _model_cache:
        return _model_cache[path]
    model = DifferenzlerNet(FEATURE_SIZE)
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    torch.set_num_threads(4)
    _model_cache[path] = model
    return model


# ===== Fast feature encoding =====

def encode_state_tensor(
    feat: torch.Tensor,
    hand: list[int],
    played_cards: set[int] | list[int],
    trick_cards: list[tuple[int, int]],
    trump: int,
    legal: list[int],
    trick_number: int,
    my_points: int,
    opp_points: list[int],
    my_target: int,
    leader: int,
    voids: dict[int, set[int]],
) -> None:
    """Encode state into pre-allocated tensor. Zero-allocation."""
    feat.zero_()
    played = set(played_cards) if not isinstance(played_cards, set) else played_cards

    for c in hand:
        feat[c] = 1.0
    for c in played:
        feat[36 + c] = 1.0
    for _, c in trick_cards:
        feat[72 + c] = 1.0
    for c in range(NUM_CARDS):
        if CARD_SUIT[c] == trump:
            feat[108 + c] = 1.0
    for c in legal:
        feat[144 + c] = 1.0

    # Current winner
    if trick_cards:
        from engine.cards import strength
        best_c = trick_cards[0][1]
        for _, c in trick_cards[1:]:
            cs = CARD_SUIT[c]
            bs = CARD_SUIT[best_c]
            if cs == trump and bs != trump:
                best_c = c
            elif cs == bs and strength(c, trump) > strength(best_c, trump):
                best_c = c
        feat[180 + best_c] = 1.0

    # Suit remaining
    all_seen = played | {c for _, c in trick_cards} | set(hand)
    off = 216
    for s in range(4):
        rem = sum(1 for c in range(NUM_CARDS) if CARD_SUIT[c] == s and c not in all_seen)
        for slot in range(4):
            feat[off + s * 4 + slot] = rem / 9.0
    off += 16

    # Voids
    for oi, p in enumerate([1, 2, 3]):
        for s in range(4):
            if s in voids.get(p, set()):
                feat[off + oi * 4 + s] = 1.0
    off += 12

    # Scalars
    gap = my_target - my_points
    total_pts = my_points + sum(opp_points)
    n_on_table = len(trick_cards)
    trump_in_hand = sum(1 for c in hand if CARD_SUIT[c] == trump)
    puur_id = trump * NUM_VALUES + 5
    nell_id = trump * NUM_VALUES + 3
    trump_ace_id = trump * NUM_VALUES + 8

    feat[off] = trick_number / 8.0
    feat[off+1] = my_points / 157.0
    feat[off+2] = opp_points[0] / 157.0
    feat[off+3] = opp_points[1] / 157.0
    feat[off+4] = opp_points[2] / 157.0
    feat[off+5] = my_target / 157.0
    feat[off+6] = gap / 157.0
    feat[off+7] = abs(gap) / 157.0
    feat[off+8] = 1.0 if gap > 0 else 0.0
    feat[off+9] = 1.0 if gap < 0 else 0.0
    feat[off+10] = 1.0 if gap == 0 else 0.0
    feat[off+11] = (9 - trick_number) / 9.0
    feat[off+12] = len(hand) / 9.0
    feat[off+13] = 1.0 if n_on_table == 0 else 0.0
    feat[off+14] = n_on_table / 3.0
    feat[off+15] = sum(card_points(c, trump) for _, c in trick_cards) / 60.0 if trick_cards else 0.0
    feat[off+16] = total_pts / 157.0
    feat[off+17] = trump_in_hand / 9.0
    feat[off+18] = 1.0 if puur_id in hand else 0.0
    feat[off+19] = 1.0 if nell_id in hand else 0.0
    feat[off+20] = 1.0 if trump_ace_id in hand else 0.0
    feat[off+21] = len(legal) / 9.0


# ===== DirectNeuralPlayer =====

class DirectNeuralPlayer:
    """Pure neural network player — no search."""

    def __init__(self, model_path: str, ansage_worlds: int = 2000,
                 rng=None, player_id: int = 0):
        self.player_id = player_id
        self.rng = rng or __import__("random").Random()
        self.model = load_model(model_path)
        self._feat = torch.zeros(1, FEATURE_SIZE)
        self._decl = DeclarationModel(use_target_aware_opponents=True)
        self._belief = BeliefTracker()
        self._target = 0
        self._trick_number = 0
        self._trump = 0
        self._ansage_worlds = ansage_worlds
        self._voids: dict[int, set[int]] = {1: set(), 2: set(), 3: set()}

    def declare(self, hand, trump):
        self._trick_number = 0
        self._trump = trump
        self._voids = {1: set(), 2: set(), 3: set()}
        self._belief.init_round(trump)
        self._target = self._decl.declare(hand, trump, num_worlds=self._ansage_worlds, rng=self.rng)
        return self._target

    def play(self, state, legal):
        if len(legal) == 1:
            return legal[0]
        for p in [1, 2, 3]:
            for s in state.voids.get(p, set()):
                self._voids.setdefault(p, set()).add(s)

        encode_state_tensor(
            self._feat[0],
            list(state.hands[self.player_id]),
            state.played_cards,
            state.cards_on_table(),
            state.trump, legal, self._trick_number,
            state.points[self.player_id],
            [state.points[1], state.points[2], state.points[3]],
            self._target, state.leader, self._voids,
        )
        with torch.no_grad():
            logits, _ = self.model(self._feat)
        # Mask illegal
        mask = torch.zeros(36)
        for c in legal:
            mask[c] = 1.0
        logits[0][mask < 0.5] = -1e9
        return legal[logits[0][legal].argmax().item()]

    def notify_trick(self, state, trick, winner, pts):
        if self._belief is not None:
            lead_suit = CARD_SUIT[trick[0][1]] if trick else None
            cards_before = []
            for player, card in trick:
                if player != self.player_id and player in [1, 2, 3]:
                    self._belief.update(
                        player=player, action=card,
                        player_current_points=state.points[player] - (pts if winner == player else 0),
                        trick_number=self._trick_number,
                        table_cards_before=list(cards_before),
                        lead_suit=lead_suit if cards_before else None,
                        trump=state.trump,
                    )
                cards_before.append((player, card))
        self._trick_number += 1


# ===== NeuralPIMCPlayer =====

class NeuralPIMCPlayer:
    """PIMC with neural network as rollout policy for player 0.

    Uses neural_rollout.NeuralRolloutEvaluator for move evaluation.
    Player 0 plays via neural network, opponents use gap heuristic.
    Endgame positions use exact solver.
    """

    def __init__(self, model_path: str, num_worlds: int = 50,
                 ansage_worlds: int = 2000, endgame_depth: int = 4,
                 rng=None, player_id: int = 0):
        self.player_id = player_id
        self.rng = rng or __import__("random").Random()
        self._decl = DeclarationModel(use_target_aware_opponents=True)
        self._belief = BeliefTracker()
        self._target = 0
        self._trick_number = 0
        self._trump = 0
        self._ansage_worlds = ansage_worlds
        self._num_worlds = num_worlds
        self._voids: dict[int, set[int]] = {1: set(), 2: set(), 3: set()}

        from bot.neural_rollout import NeuralRolloutEvaluator
        self._evaluator = NeuralRolloutEvaluator(model_path, endgame_depth=endgame_depth)

    def declare(self, hand, trump):
        self._trick_number = 0
        self._trump = trump
        self._voids = {1: set(), 2: set(), 3: set()}
        self._belief.init_round(trump)
        self._target = self._decl.declare(hand, trump, num_worlds=self._ansage_worlds, rng=self.rng)
        return self._target

    def play(self, state, legal):
        if len(legal) == 1:
            return legal[0]

        for p in [1, 2, 3]:
            for s in state.voids.get(p, set()):
                self._voids.setdefault(p, set()).add(s)

        from bot.world_sampler import sample_worlds_with_declarations

        hand = list(state.hands[self.player_id])
        trump = state.trump
        played = list(state.played_cards)
        table = state.cards_on_table()
        on_table = [c for _, c in table]
        unknown = [c for c in range(36) if c not in set(hand) and c not in set(played) and c not in set(on_table)]

        raw_worlds = sample_worlds_with_declarations(
            unknown, len(hand), self._voids, self._num_worlds,
            trump, self.rng, belief_tracker=self._belief,
        )
        if not raw_worlds:
            return legal[0]

        worlds_hands = [wh for wh, _ in raw_worlds]
        opp_tgts = [wd for _, wd in raw_worlds]

        total_tricks = len(hand) + self._trick_number
        card_utils = self._evaluator.evaluate_moves(
            candidates=legal,
            my_hand=hand,
            worlds=worlds_hands,
            trump=trump,
            my_target=self._target,
            my_points=state.points[self.player_id],
            opp_targets_list=opp_tgts,
            opp_points=list(state.points),
            leader=state.leader,
            trick_number=self._trick_number,
            total_tricks=total_tricks,
            table_cards=table if table else None,
            voids=self._voids,
        )
        return max(legal, key=lambda c: card_utils.get(c, float("-inf")))

    def notify_trick(self, state, trick, winner, pts):
        if self._belief is not None:
            lead_suit = CARD_SUIT[trick[0][1]] if trick else None
            cards_before = []
            for player, card in trick:
                if player != self.player_id and player in [1, 2, 3]:
                    self._belief.update(
                        player=player, action=card,
                        player_current_points=state.points[player] - (pts if winner == player else 0),
                        trick_number=self._trick_number,
                        table_cards_before=list(cards_before),
                        lead_suit=lead_suit if cards_before else None,
                        trump=state.trump,
                    )
                cards_before.append((player, card))
        self._trick_number += 1

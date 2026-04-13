"""Best player: Expert declaration + RL neural card play.

Declaration: human formula (trump*2 + aces) enhanced with expert adjustments.
Card play: RL neural net (ONNX, fast inference).

This is the deployment-ready player for Swisslos.
"""

from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import onnxruntime as ort
from engine.cards import CARD_SUIT, points, strength
from engine.rules import legal_moves
from engine.game_state import GameState
from bot.heuristics import estimate_hand_score


SUIT_NAME = {0: "Herz", 1: "Ecke", 2: "Schaufel", 3: "Kreuz"}
SUIT_SYM = {0: "H", 1: "D", 2: "S", 3: "C"}
VAL_NAME = {0: "6", 1: "7", 2: "8", 3: "9", 4: "10", 5: "U", 6: "O", 7: "K", 8: "A"}


def card_name(c: int) -> str:
    return f"{SUIT_SYM[c // 9]}{VAL_NAME[c % 9]}"


class BestPlayer:
    """Expert declaration + RL neural card play."""

    def __init__(
        self,
        player_id: int = 0,
        onnx_path: str = "models/play_net.onnx",
        verbose: bool = False,
    ):
        self.player_id = player_id
        self.verbose = verbose

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

        self._target = 0
        self._trump = 0
        self._my_points = 0
        self._opp_points = {1: 0, 2: 0, 3: 0}
        self._played_cards = []
        self._trick_number = 0
        self._voids = [[False]*4 for _ in range(4)]

    def declare(self, hand: list[int], trump: int) -> int:
        self._trump = trump
        self._my_points = 0
        self._opp_points = {1: 0, 2: 0, 3: 0}
        self._played_cards = []
        self._trick_number = 0
        self._voids = [[False]*4 for _ in range(4)]

        self._target = estimate_hand_score(hand, trump)

        if self.verbose:
            hand_str = " ".join(card_name(c) for c in sorted(hand))
            print(f"\n[DECLARE] Trump={SUIT_NAME[trump]} Hand={hand_str}")
            print(f"[DECLARE] → {self._target} points")

        return self._target

    def play(self, state: GameState, legal: list[int]) -> int:
        obs = self._encode(state, legal)
        mask = np.zeros((1, 36), dtype=np.float32)
        for c in legal:
            mask[0, c] = 1.0

        logits, value = self.sess.run(None, {
            'obs': obs.reshape(1, -1),
            'legal_mask': mask,
        })

        card = int(np.argmax(logits[0]))
        if card not in legal:
            card = legal[0]

        if self.verbose:
            gap = self._target - state.points[self.player_id]
            legal_str = " ".join(card_name(c) for c in legal)
            trick_str = " ".join(f"{card_name(c)}" for _, c in (state.current_trick or []))
            print(f"\n[PLAY] Trick {self._trick_number+1} | Gap={gap:+d} | Table: {trick_str}")
            print(f"[PLAY] Legal: {legal_str}")
            print(f"[PLAY] → {card_name(card)} (value={value[0]:.1f})")

        return card

    def notify_trick(self, state: GameState, trick: list[tuple[int, int]], winner: int, pts: int) -> None:
        for p, c in trick:
            self._played_cards.append(c)
            if trick:
                lead_suit = CARD_SUIT[trick[0][1]]
                if CARD_SUIT[c] != lead_suit and p != self.player_id:
                    rp = (p - self.player_id) % 4
                    if 0 <= rp < 4:
                        self._voids[rp][lead_suit] = True

        if winner == self.player_id:
            self._my_points += pts
        elif winner in self._opp_points:
            self._opp_points[winner] += pts

        self._trick_number += 1

        if self.verbose:
            trick_str = " ".join(f"{card_name(c)}" for _, c in trick)
            w = "US" if winner == self.player_id else f"P{winner}"
            gap = self._target - self._my_points
            print(f"[TRICK] {trick_str} → {w} +{pts}pts | Our pts={self._my_points} gap={gap:+d}")

    def _encode(self, state: GameState, legal: list[int]) -> np.ndarray:
        obs = np.zeros(204, dtype=np.float32)
        trump = state.trump
        pid = self.player_id

        for c in state.hands[pid]: obs[c] = 1.0

        current_tc = set()
        if state.current_trick:
            for _, c in state.current_trick:
                current_tc.add(c)
        for c in self._played_cards:
            if c not in current_tc:
                obs[36 + c] = 1.0

        for c in range(trump*9, trump*9+9): obs[72 + c] = 1.0
        for c in legal: obs[108 + c] = 1.0
        if state.current_trick:
            for _, c in state.current_trick: obs[144 + c] = 1.0

        for i in range(3):
            opp = (pid + 1 + i) % 4
            rp = (opp - pid) % 4
            for s in range(4):
                if 0 <= rp < 4:
                    obs[180 + i*4 + s] = 1.0 if self._voids[rp][s] else 0.0

        obs[192] = self._trick_number / 8.0
        obs[193] = state.points[pid] / 157.0
        gap = (self._target - state.points[pid]) / 157.0
        obs[194] = max(-1.0, min(1.0, gap))
        obs[195] = len(state.current_trick or []) / 3.0
        obs[196] = 1.0 if not state.current_trick else 0.0
        for i in range(3):
            opp = (pid + 1 + i) % 4
            obs[197 + i] = state.points[opp] / 157.0
        obs[203] = self._target / 157.0

        return obs

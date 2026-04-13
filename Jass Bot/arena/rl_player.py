"""RL Neural Network player for arena evaluation.

Implements the Player protocol (declare, play, notify_trick).
Loads trained PlayNet + DeclNet from checkpoint.
"""

from __future__ import annotations

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.cards import CARD_SUIT, CARD_POINTS, points, strength
from engine.rules import legal_moves
from engine.game_state import GameState


class RLPlayer:
    """Arena player using trained RL neural networks."""

    def __init__(self, player_id: int = 0, checkpoint_path: str = None, device: str = "cuda"):
        self.player_id = player_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._target = 0
        self._trump = 0
        self._hand = []
        self._played_cards = set()
        self._trick_cards = []
        self._my_points = 0
        self._opp_points = [0, 0, 0, 0]
        self._opp_declarations = [0, 0, 0, 0]
        self._voids = [[False]*4 for _ in range(4)]  # [player][suit]
        self._trick_count = 0

        # Load networks
        from bot.train_v5 import PlayNet, DeclNet
        self.play_net = PlayNet(204, 512).to(self.device)
        self.decl_net = DeclNet(72, 256).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.play_net.load_state_dict(ckpt["play_net"])
            self.decl_net.load_state_dict(ckpt["decl_net"])

        self.play_net.eval()
        self.decl_net.eval()

    def declare(self, hand: list[int], trump: int) -> int:
        self._trump = trump
        self._hand = list(hand)
        self._played_cards = set()
        self._trick_cards = []
        self._my_points = 0
        self._opp_points = [0, 0, 0, 0]
        self._voids = [[False]*4 for _ in range(4)]
        self._trick_count = 0

        feat = self._encode_decl(hand, trump)
        x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            pred = self.decl_net(x).item()
        self._target = max(0, min(157, round(pred)))
        return self._target

    def play(self, state: GameState, legal: list[int]) -> int:
        obs = self._encode_play(state, legal)
        legal_mask = np.zeros(36, dtype=np.float32)
        for c in legal:
            legal_mask[c] = 1.0

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(legal_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.play_net(obs_t, mask_t)
            # Greedy (no sampling during evaluation)
            card = logits.argmax(dim=-1).item()

        # Fallback if somehow illegal
        if card not in legal:
            card = legal[0]

        return card

    def notify_trick(self, state: GameState, trick: list[tuple[int, int]], winner: int, pts: int) -> None:
        for p, c in trick:
            self._played_cards.add(c)
            # Detect voids
            if len(trick) > 0:
                lead_suit = CARD_SUIT[trick[0][1]]
                if CARD_SUIT[c] != lead_suit:
                    remap_p = self._remap(p)
                    if 0 <= remap_p < 4:
                        self._voids[remap_p][lead_suit] = True

        if winner == self.player_id:
            self._my_points += pts
        else:
            remap_w = self._remap(winner)
            if 0 <= remap_w < 4:
                self._opp_points[remap_w] += pts

        self._trick_count += 1
        self._trick_cards = []

    def notify_declarations(self, declarations: dict[int, int]) -> None:
        """Called by arena to tell us what everyone declared."""
        for p, d in declarations.items():
            rp = self._remap(p)
            if 0 <= rp < 4:
                self._opp_declarations[rp] = d

    def _remap(self, player: int) -> int:
        if player == self.player_id:
            return 0
        offset = (player - self.player_id) % 4
        return offset

    def _encode_decl(self, hand: list[int], trump: int) -> np.ndarray:
        """72-feature declaration encoding matching C engine."""
        obs = np.zeros(72, dtype=np.float32)

        # [0..35] hand binary
        for c in hand:
            obs[c] = 1.0

        # [36..39] trump one-hot
        obs[36 + trump] = 1.0

        # [40..48] trump card positions
        for c in hand:
            if CARD_SUIT[c] == trump:
                obs[40 + c % 9] = 1.0

        # [49..52] suit lengths / 9
        suit_lens = [0]*4
        for c in hand:
            suit_lens[CARD_SUIT[c]] += 1
        for i in range(4):
            obs[49+i] = suit_lens[i] / 9.0

        # [53..56] suit points / 62
        suit_pts = [0]*4
        for c in hand:
            suit_pts[CARD_SUIT[c]] += points(c, trump)
        for i in range(4):
            obs[53+i] = suit_pts[i] / 62.0

        # [57..59] has_puur, has_nell, has trump ace
        trump_base = trump * 9
        obs[57] = 1.0 if (trump_base + 5) in hand else 0.0  # Puur
        obs[58] = 1.0 if (trump_base + 3) in hand else 0.0  # Nell
        obs[59] = 1.0 if (trump_base + 8) in hand else 0.0  # Trump Ace

        # [60] trump_count / 9
        obs[60] = suit_lens[trump] / 9.0

        # [61] void_count / 3
        void_count = sum(1 for i in range(4) if i != trump and suit_lens[i] == 0)
        obs[61] = void_count / 3.0

        # [62] total_points / 157
        obs[62] = sum(suit_pts) / 157.0

        # [63] trump_points / 62
        obs[63] = suit_pts[trump] / 62.0

        # [64] side_aces / 3
        side_aces = sum(1 for i in range(4) if i != trump and (i*9+8) in hand)
        obs[64] = side_aces / 3.0

        # [65] high_trump_count / 3
        high = sum(1 for cid in [trump_base+5, trump_base+3, trump_base+8] if cid in hand)
        obs[65] = high / 3.0

        # [66] protected_aces / 3
        prot = sum(1 for i in range(4) if i != trump and (i*9+8) in hand and suit_lens[i] >= 2)
        obs[66] = prot / 3.0

        # [67] long_suits / 3
        long_s = sum(1 for i in range(4) if i != trump and suit_lens[i] >= 4)
        obs[67] = long_s / 3.0

        return obs

    def _encode_play(self, state: GameState, legal: list[int]) -> np.ndarray:
        """204-feature play encoding matching C engine."""
        obs = np.zeros(204, dtype=np.float32)
        trump = state.trump
        pid = self.player_id

        # [0..35] my hand
        for c in state.hands[pid]:
            obs[c] = 1.0

        # [36..71] played cards (previous tricks, not current)
        current_trick_cards = set()
        if state.current_trick:
            for p, c in state.current_trick:
                current_trick_cards.add(c)
        for c in self._played_cards:
            if c not in current_trick_cards:
                obs[36 + c] = 1.0

        # [72..107] is_trump
        for c in range(trump*9, trump*9+9):
            obs[72 + c] = 1.0

        # [108..143] legal mask
        for c in legal:
            obs[108 + c] = 1.0

        # [144..179] current trick cards
        if state.current_trick:
            for p, c in state.current_trick:
                obs[144 + c] = 1.0

        # [180..191] void knowledge (3 opponents × 4 suits)
        for i in range(3):
            opp = (pid + 1 + i) % 4
            rp = self._remap(opp)
            for s in range(4):
                if 0 <= rp < 4:
                    obs[180 + i*4 + s] = 1.0 if self._voids[rp][s] else 0.0

        # Scalars
        obs[192] = self._trick_count / 8.0
        obs[193] = state.points[pid] / 157.0
        gap = (self._target - state.points[pid]) / 157.0
        obs[194] = max(-1.0, min(1.0, gap))
        obs[195] = len(state.current_trick) / 3.0 if state.current_trick else 0.0
        obs[196] = 1.0 if (not state.current_trick or len(state.current_trick) == 0) else 0.0

        # Opponent points and declarations
        for i in range(3):
            opp = (pid + 1 + i) % 4
            obs[197 + i] = state.points[opp] / 157.0
            obs[200 + i] = self._opp_declarations[self._remap(opp)] / 157.0
        obs[203] = self._target / 157.0

        return obs

"""Batched neural game simulation for declaration.

Plays N complete games in parallel using the V7 PlayNet on GPU.
All 4 players use the neural network (or a mix of neural + heuristic).
Returns the score distribution for player 0.

This gives REALISTIC score predictions — the neural net plays like
the actual bot, not like the braindead pick_card heuristic.
"""

from __future__ import annotations

import os
import sys
import random
import torch
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.cards import CARD_SUIT, CARD_POINTS, points as card_points


class NeuralSimulator:
    """Batched neural game simulator on GPU."""

    def __init__(self, checkpoint_path: str = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load V7 PlayNet
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), "..", "checkpoints_v7", "best.pt"
            )

        from bot.train_v7_league import PlayNet
        self.net = PlayNet(204, 512).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["play_net"])
        self.net.eval()

        # Precompute card properties
        self._suit = np.array([c // 9 for c in range(36)], dtype=np.int32)
        self._pts_normal = np.array([card_points(c, -1) for c in range(36)], dtype=np.int32)
        # Points per card per trump suit
        self._pts_trump = np.zeros((4, 36), dtype=np.int32)
        for t in range(4):
            for c in range(36):
                self._pts_trump[t, c] = card_points(c, t)
        # Strength for trick winner determination
        _str_normal = [0,1,2,3,4,5,6,7,8]
        _str_trump = [0,1,2,7,3,8,4,5,6]
        self._str = np.zeros((2, 36), dtype=np.int32)  # [is_trump][card]
        for c in range(36):
            self._str[0, c] = _str_normal[c % 9]
            self._str[1, c] = _str_trump[c % 9]

        self.available = True

    @torch.no_grad()
    def simulate_games(
        self,
        my_hand: list[int],
        worlds: list[dict[int, list[int]]],
        trump: int,
        targets: list[dict[int, int]] | None = None,
    ) -> list[int]:
        """Simulate N games with neural play, return player 0's scores.

        Args:
            my_hand: player 0's cards (list of card IDs)
            worlds: [{1: [cards], 2: [cards], 3: [cards]}, ...] opponent hands
            trump: trump suit (0-3)
            targets: [{1: t, 2: t, 3: t}, ...] opponent targets (for obs encoding)

        Returns:
            List of player 0's final scores, one per world.
        """
        N = len(worlds)
        if N == 0:
            return []

        total_tricks = len(my_hand)
        device = self.device

        # Initialize hands as bitmasks [N, 4] on CPU (numpy)
        hands = np.zeros((N, 4), dtype=np.int64)
        for w in range(N):
            mask = 0
            for c in my_hand:
                mask |= (1 << c)
            hands[w, 0] = mask
            for pi, p in enumerate([1, 2, 3]):
                mask = 0
                for c in worlds[w][p]:
                    mask |= (1 << c)
                hands[w, pi + 1] = mask

        # Points and targets
        pts = np.zeros((N, 4), dtype=np.int32)
        tgts = np.zeros((N, 4), dtype=np.int32)
        for w in range(N):
            tgts[w, 0] = 40  # dummy target for player 0 during sim
            if targets:
                for pi, p in enumerate([1, 2, 3]):
                    tgts[w, pi + 1] = targets[w].get(p, 39)
            else:
                tgts[w, 1:] = 39

        leaders = np.zeros(N, dtype=np.int32)  # who leads each trick
        played = np.zeros((N, 36), dtype=np.float32)  # played cards tracking

        for trick in range(total_tricks):
            trick_cards = np.full((N, 4), -1, dtype=np.int32)  # cards played this trick
            trick_players = np.full((N, 4), -1, dtype=np.int32)
            lead_suits = np.full(N, -1, dtype=np.int32)
            n_played = np.zeros(N, dtype=np.int32)

            for seat_offset in range(4):
                # Which player plays at this position in each game
                current_p = (leaders + seat_offset) % 4  # [N]

                # Build observation batch for current players
                obs = np.zeros((N, 204), dtype=np.float32)
                mask = np.zeros((N, 36), dtype=np.float32)

                for w in range(N):
                    p = current_p[w]
                    hand = hands[w, p]

                    # Legal moves
                    ls = lead_suits[w]
                    if ls >= 0:
                        suit_mask_val = 0
                        for c in range(ls * 9, ls * 9 + 9):
                            suit_mask_val |= (1 << c)
                        follow = hand & suit_mask_val
                        legal = follow if follow else hand
                    else:
                        legal = hand

                    # Encode obs[0:36] = hand
                    for c in range(36):
                        if hand & (1 << c):
                            obs[w, c] = 1.0
                        # played cards
                        if played[w, c] and not (trick_cards[w] == c).any():
                            obs[w, 36 + c] = 1.0

                    # Trump
                    for c in range(trump * 9, trump * 9 + 9):
                        obs[w, 72 + c] = 1.0

                    # Legal mask
                    for c in range(36):
                        if legal & (1 << c):
                            obs[w, 108 + c] = 1.0
                            mask[w, c] = 1.0

                    # Current trick cards
                    for i in range(n_played[w]):
                        tc = trick_cards[w, i]
                        if tc >= 0:
                            obs[w, 144 + tc] = 1.0

                    # Scalars
                    obs[w, 192] = trick / 8.0
                    obs[w, 193] = pts[w, p] / 157.0
                    gap = (tgts[w, p] - pts[w, p]) / 157.0
                    obs[w, 194] = max(-1.0, min(1.0, gap))
                    obs[w, 195] = n_played[w] / 3.0
                    obs[w, 196] = 1.0 if n_played[w] == 0 else 0.0
                    obs[w, 203] = tgts[w, p] / 157.0

                # Batch inference
                obs_t = torch.from_numpy(obs).to(device)
                mask_t = torch.from_numpy(mask).to(device)
                logits, _ = self.net(obs_t, mask_t)
                choices = logits.argmax(dim=-1).cpu().numpy()  # [N]

                # Apply choices
                for w in range(N):
                    card = int(choices[w])
                    p = current_p[w]

                    # Validate legal
                    if not (mask[w, card] > 0):
                        # Fallback: pick first legal
                        for c in range(36):
                            if mask[w, c] > 0:
                                card = c
                                break

                    hands[w, p] &= ~(1 << card)
                    trick_cards[w, n_played[w]] = card
                    trick_players[w, n_played[w]] = p
                    if lead_suits[w] < 0:
                        lead_suits[w] = self._suit[card]
                    n_played[w] += 1
                    played[w, card] = 1.0

            # Resolve tricks
            for w in range(N):
                # Find winner
                best_p = trick_players[w, 0]
                best_c = trick_cards[w, 0]
                best_is_trump = (self._suit[best_c] == trump)
                best_str = self._str[1 if best_is_trump else 0, best_c]

                for i in range(1, 4):
                    c = trick_cards[w, i]
                    if c < 0:
                        break
                    c_suit = self._suit[c]
                    is_trump = (c_suit == trump)
                    c_str = self._str[1 if is_trump else 0, c]

                    if is_trump and not best_is_trump:
                        best_p = trick_players[w, i]
                        best_c = c
                        best_str = c_str
                        best_is_trump = True
                    elif is_trump and best_is_trump and c_str > best_str:
                        best_p = trick_players[w, i]
                        best_c = c
                        best_str = c_str
                    elif not is_trump and not best_is_trump and c_suit == self._suit[best_c] and c_str > best_str:
                        best_p = trick_players[w, i]
                        best_c = c
                        best_str = c_str

                # Points
                tpts = 0
                for i in range(4):
                    c = trick_cards[w, i]
                    if c >= 0:
                        tpts += self._pts_trump[trump, c]
                if trick == total_tricks - 1:
                    tpts += 5  # last trick bonus

                pts[w, best_p] += tpts
                leaders[w] = best_p

        return [int(pts[w, 0]) for w in range(N)]

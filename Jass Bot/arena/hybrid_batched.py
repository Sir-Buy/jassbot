"""Batched Hybrid PIMC+Neural player.

KEY OPTIMIZATION: All worlds simulated in parallel. Each trick step makes
ONE batched GPU call for all worlds, not one call per world.

For N worlds and C candidate cards:
  Old: N * C * 9 forward passes = 450*C calls
  New: C * 9 forward passes (batch size N) = 9*C calls
  Speedup: ~50x for N=50, ~200x for N=200

Usage in arena:
    player = BatchedHybridPlayer(checkpoint_path="checkpoints_league/latest.pt", num_worlds=200)
"""

from __future__ import annotations

import os
import sys
import random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.cards import CARD_SUIT, points, strength
from engine.rules import legal_moves
from engine.game_state import GameState
from bot.heuristics import pick_opponent_target_aware, pick_opponent_greedy
from bot.world_sampler import sample_worlds_with_declarations
from bot.belief_tracker import BeliefTracker


class BatchedHybridPlayer:
    """PIMC search with batched neural rollouts. All worlds evaluated in parallel on GPU."""

    def __init__(
        self,
        player_id: int = 0,
        checkpoint_path: str = None,
        num_worlds: int = 200,
        device: str = "cuda",
        rng: random.Random = None,
    ):
        self.player_id = player_id
        self.num_worlds = num_worlds
        self.rng = rng or random.Random()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        from bot.train_v5 import PlayNet, DeclNet
        self.play_net = PlayNet(204, 512).to(self.device)
        self.decl_net = DeclNet(72, 256).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.play_net.load_state_dict(ckpt["play_net"])
            self.decl_net.load_state_dict(ckpt["decl_net"])

        self.play_net.eval()
        self.decl_net.eval()

        self._target = 0
        self._trump = 0
        self._my_points = 0
        self._opp_points = {1: 0, 2: 0, 3: 0}
        self._played_cards = []
        self._trick_number = 0
        self._voids = {1: set(), 2: set(), 3: set()}
        self._belief_tracker = BeliefTracker()

        # Pre-allocate GPU tensors for batched inference
        self._obs_buf = torch.zeros(num_worlds, 204, device=self.device)
        self._mask_buf = torch.zeros(num_worlds, 36, device=self.device)

    def declare(self, hand: list[int], trump: int) -> int:
        self._trump = trump
        self._my_points = 0
        self._opp_points = {1: 0, 2: 0, 3: 0}
        self._played_cards = []
        self._trick_number = 0
        self._voids = {1: set(), 2: set(), 3: set()}
        self._belief_tracker.init_round(trump)

        feat = self._encode_decl(hand, trump)
        x = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            pred = self.decl_net(x).item()
        self._target = max(0, min(157, round(pred)))
        return self._target

    def play(self, state: GameState, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]

        hand = list(state.hands[self.player_id])
        trump = state.trump
        table_cards = list(state.current_trick) if state.current_trick else []

        on_table = [c for _, c in table_cards]
        unknown = [c for c in range(36) if c not in hand and c not in self._played_cards and c not in on_table]

        worlds = sample_worlds_with_declarations(
            unknown, len(hand), self._voids, self.num_worlds,
            trump, self.rng, belief_tracker=self._belief_tracker,
        )
        if not worlds:
            return self._neural_pick_single(hand, trump, legal)

        leader = table_cards[0][0] if table_cards else (state.leader if hasattr(state, 'leader') else 0)

        # Evaluate each candidate
        best_card = legal[0]
        best_util = float("-inf")

        for card in legal:
            util = self._eval_card_batched(
                hand, trump, self._target, self._my_points,
                table_cards, leader, card, worlds,
                self._trick_number, self._opp_points,
            )
            if util > best_util:
                best_util = util
                best_card = card

        return best_card

    def _eval_card_batched(self, hand, trump, target, my_points,
                           table_cards, leader, my_card, worlds,
                           trick_number, opp_points) -> float:
        """Evaluate a candidate card across all worlds with batched neural inference."""
        W = len(worlds)
        my_hand = [c for c in hand if c != my_card]
        total_tricks = trick_number + len(hand)

        # Initialize per-world state
        w_hands = []  # [W][player] -> list of cards
        w_opp_pts = []
        w_my_pts = np.full(W, my_points, dtype=np.float32)
        w_leaders = np.full(W, leader, dtype=np.int32)
        w_played = [set(self._played_cards) for _ in range(W)]
        w_my_hands = [list(my_hand) for _ in range(W)]
        w_decls = []

        for wi, (wh, wd) in enumerate(worlds):
            w_hands.append({p: list(cards) for p, cards in wh.items()})
            w_opp_pts.append({p: opp_points.get(p, 0) for p in [1, 2, 3]})
            w_decls.append(wd)
            w_played[wi].add(my_card)

        # Complete current trick (opponents use heuristic — fast, no neural)
        for wi in range(W):
            trick = list(table_cards) + [(self.player_id, my_card)]
            lead_suit = CARD_SUIT[trick[0][1]]

            # Remove table cards from opponent hands
            for p, c in table_cards:
                if p in w_hands[wi] and c in w_hands[wi][p]:
                    w_hands[wi][p].remove(c)

            played_set = set(p for p, _ in trick)
            for i in range(4):
                p = (leader + i) % 4
                if p in played_set or p == self.player_id:
                    continue
                if p in w_hands[wi] and w_hands[wi][p]:
                    if w_decls[wi] is not None:
                        card = pick_opponent_target_aware(
                            w_hands[wi][p], trick, lead_suit, trump,
                            w_decls[wi][p], w_opp_pts[wi].get(p, 0), trick_number)
                    else:
                        card = pick_opponent_greedy(w_hands[wi][p], trick, lead_suit, trump)
                    trick.append((p, card))
                    w_hands[wi][p].remove(card)
                    w_played[wi].add(card)

            # Resolve trick
            if len(trick) >= 4:
                winner = self._trick_winner(trick, lead_suit, trump)
                pts = sum(points(c, trump) for _, c in trick)
                if trick_number == total_tricks - 1:
                    pts += 5
                if winner == self.player_id:
                    w_my_pts[wi] += pts
                elif winner in w_opp_pts[wi]:
                    w_opp_pts[wi][winner] += pts
                w_leaders[wi] = winner

        # Remaining tricks: BATCHED neural for player 0
        for t in range(trick_number + 1, total_tricks):
            # For each world, play opponents until it's player 0's turn,
            # then batch all player 0 decisions

            # Step 1: play opponents before player 0 in trick order
            w_tricks = [[] for _ in range(W)]
            w_lead_suits = [None] * W

            for wi in range(W):
                for i in range(4):
                    p = (w_leaders[wi] + i) % 4
                    if p == self.player_id:
                        break  # player 0's turn — handle batched below
                    if p in w_hands[wi] and w_hands[wi][p]:
                        ls = CARD_SUIT[w_tricks[wi][0][1]] if w_tricks[wi] else None
                        if w_decls[wi] is not None:
                            card = pick_opponent_target_aware(
                                w_hands[wi][p], w_tricks[wi], ls, trump,
                                w_decls[wi][p], w_opp_pts[wi].get(p, 0), t)
                        else:
                            card = pick_opponent_greedy(w_hands[wi][p], w_tricks[wi], ls, trump)
                        w_tricks[wi].append((p, card))
                        w_hands[wi][p].remove(card)
                        w_played[wi].add(card)
                w_lead_suits[wi] = CARD_SUIT[w_tricks[wi][0][1]] if w_tricks[wi] else None

            # Step 2: BATCHED neural inference for player 0
            active_worlds = [wi for wi in range(W) if w_my_hands[wi]]
            if not active_worlds:
                break

            # Build batched observation
            n_active = len(active_worlds)
            obs_batch = self._obs_buf[:n_active]
            mask_batch = self._mask_buf[:n_active]
            obs_batch.zero_()
            mask_batch.zero_()

            legal_lists = []
            for bi, wi in enumerate(active_worlds):
                ls = w_lead_suits[wi]
                tc = w_tricks[wi] if w_tricks[wi] else None
                leg = legal_moves(w_my_hands[wi], ls, trump, tc)
                legal_lists.append(leg)

                # Encode observation directly into pre-allocated buffer
                obs = obs_batch[bi]
                for c in w_my_hands[wi]: obs[c] = 1.0
                for c in w_played[wi]:
                    tc_set = set(cc for _, cc in (w_tricks[wi] or []))
                    if c not in tc_set: obs[36 + c] = 1.0
                for c in range(trump*9, trump*9+9): obs[72 + c] = 1.0
                for c in leg: obs[108 + c] = 1.0
                for _, c in (w_tricks[wi] or []): obs[144 + c] = 1.0
                obs[192] = t / 8.0
                obs[193] = float(w_my_pts[wi]) / 157.0
                obs[194] = max(-1.0, min(1.0, float(target - w_my_pts[wi]) / 157.0))
                obs[195] = len(w_tricks[wi] or []) / 3.0
                obs[203] = float(target) / 157.0

                for c in leg: mask_batch[bi][c] = 1.0

            # ONE batched GPU call for all worlds
            with torch.no_grad():
                logits, _ = self.play_net(obs_batch[:n_active], mask_batch[:n_active])
                cards_tensor = logits.argmax(dim=-1)
                chosen_cards = cards_tensor.cpu().numpy()

            # Apply chosen cards
            for bi, wi in enumerate(active_worlds):
                card = int(chosen_cards[bi])
                if card not in legal_lists[bi]:
                    card = legal_lists[bi][0]
                w_tricks[wi].append((self.player_id, card))
                w_my_hands[wi].remove(card)
                w_played[wi].add(card)
                if w_lead_suits[wi] is None:
                    w_lead_suits[wi] = CARD_SUIT[card]

            # Step 3: opponents after player 0
            for wi in range(W):
                my_pos = None
                for i in range(4):
                    p = (w_leaders[wi] + i) % 4
                    if p == self.player_id:
                        my_pos = i
                        break

                if my_pos is None:
                    continue

                for i in range(my_pos + 1, 4):
                    p = (w_leaders[wi] + i) % 4
                    if p == self.player_id:
                        continue
                    if p in w_hands[wi] and w_hands[wi][p]:
                        ls = w_lead_suits[wi]
                        if w_decls[wi] is not None:
                            card = pick_opponent_target_aware(
                                w_hands[wi][p], w_tricks[wi], ls, trump,
                                w_decls[wi][p], w_opp_pts[wi].get(p, 0), t)
                        else:
                            card = pick_opponent_greedy(w_hands[wi][p], w_tricks[wi], ls, trump)
                        w_tricks[wi].append((p, card))
                        w_hands[wi][p].remove(card)
                        w_played[wi].add(card)

            # Step 4: resolve tricks
            for wi in range(W):
                if len(w_tricks[wi]) >= 4:
                    ls = CARD_SUIT[w_tricks[wi][0][1]]
                    winner = self._trick_winner(w_tricks[wi], ls, trump)
                    pts = sum(points(c, trump) for _, c in w_tricks[wi])
                    if t == total_tricks - 1:
                        pts += 5
                    if winner == self.player_id:
                        w_my_pts[wi] += pts
                    elif winner in w_opp_pts[wi]:
                        w_opp_pts[wi][winner] += pts
                    w_leaders[wi] = winner

        # Compute mean utility
        utilities = -(np.abs(target - w_my_pts))
        return float(utilities.mean())

    def notify_trick(self, state, trick, winner, pts):
        for p, c in trick:
            self._played_cards.append(c)
            if len(trick) > 0:
                lead_suit = CARD_SUIT[trick[0][1]]
                if CARD_SUIT[c] != lead_suit and p != self.player_id:
                    self._voids.setdefault(p, set()).add(lead_suit)
        if winner == self.player_id:
            self._my_points += pts
        elif winner in self._opp_points:
            self._opp_points[winner] += pts
        self._trick_number += 1

    def _neural_pick_single(self, hand, trump, legal):
        obs = np.zeros(204, dtype=np.float32)
        for c in hand: obs[c] = 1.0
        for c in range(trump*9, trump*9+9): obs[72+c] = 1.0
        for c in legal: obs[108+c] = 1.0
        obs[203] = self._target / 157.0
        mask = np.zeros(36, dtype=np.float32)
        for c in legal: mask[c] = 1.0
        with torch.no_grad():
            l, _ = self.play_net(
                torch.tensor(obs, device=self.device).unsqueeze(0),
                torch.tensor(mask, device=self.device).unsqueeze(0))
            return l.argmax(-1).item()

    def _trick_winner(self, trick, lead_suit, trump):
        best_p, best_c = trick[0]
        for p, c in trick[1:]:
            cs = CARD_SUIT[c]
            bs = CARD_SUIT[best_c]
            if cs == trump and bs != trump:
                best_p, best_c = p, c
            elif cs == bs and strength(c, trump) > strength(best_c, trump):
                best_p, best_c = p, c
        return best_p

    def _encode_decl(self, hand, trump):
        obs = np.zeros(72, dtype=np.float32)
        for c in hand: obs[c] = 1.0
        obs[36+trump] = 1.0
        for c in hand:
            if CARD_SUIT[c] == trump: obs[40+c%9] = 1.0
        sl = [0]*4
        for c in hand: sl[CARD_SUIT[c]] += 1
        for i in range(4): obs[49+i] = sl[i]/9.0
        sp = [0]*4
        for c in hand: sp[CARD_SUIT[c]] += points(c, trump)
        for i in range(4): obs[53+i] = sp[i]/62.0
        tb = trump*9
        obs[57] = 1.0 if (tb+5) in hand else 0.0
        obs[58] = 1.0 if (tb+3) in hand else 0.0
        obs[59] = 1.0 if (tb+8) in hand else 0.0
        obs[60] = sl[trump]/9.0
        obs[61] = sum(1 for i in range(4) if i!=trump and sl[i]==0)/3.0
        obs[62] = sum(sp)/157.0
        obs[63] = sp[trump]/62.0
        obs[64] = sum(1 for i in range(4) if i!=trump and (i*9+8) in hand)/3.0
        h = sum(1 for c in [tb+5,tb+3,tb+8] if c in hand)
        obs[65] = h/3.0
        obs[66] = sum(1 for i in range(4) if i!=trump and (i*9+8) in hand and sl[i]>=2)/3.0
        obs[67] = sum(1 for i in range(4) if i!=trump and sl[i]>=4)/3.0
        return obs

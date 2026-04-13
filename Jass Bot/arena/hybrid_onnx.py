"""ONNX Hybrid PIMC+Neural player — maximum speed.

Uses ONNX Runtime (GPU) for player 0 neural inference.
Uses C engine pick_card for opponent rollouts (microsecond speed).
All opponent simulation in C, neural batched on GPU.

Speed target: <100ms per move with 200 worlds.
"""

from __future__ import annotations

import os
import sys
import random
import ctypes
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import onnxruntime as ort
from engine.cards import CARD_SUIT, points, strength
from engine.rules import legal_moves
from engine.game_state import GameState
from bot.world_sampler import sample_worlds_with_declarations
from bot.belief_tracker import BeliefTracker
from bot.heuristics import estimate_hand_score


class OnnxHybridPlayer:
    """PIMC search with ONNX neural rollouts — max throughput."""

    def __init__(
        self,
        player_id: int = 0,
        onnx_path: str = "models/play_net.onnx",
        decl_checkpoint: str = None,
        num_worlds: int = 200,
        rng: random.Random = None,
    ):
        self.player_id = player_id
        self.num_worlds = num_worlds
        self.rng = rng or random.Random()

        # ONNX session for play net
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)

        # Declaration net (PyTorch, only called once per round)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from bot.train_v5 import DeclNet
        self.decl_net = DeclNet(72, 256).to(self.device)
        if decl_checkpoint and os.path.exists(decl_checkpoint):
            ckpt = torch.load(decl_checkpoint, map_location=self.device, weights_only=False)
            self.decl_net.load_state_dict(ckpt["decl_net"])
        self.decl_net.eval()

        # C engine for fast opponent play
        lib_dir = os.path.join(os.path.dirname(__file__), "..", "engine_c")
        # Use v2 DLL (has exported pick_card)
        lib_path = os.path.join(lib_dir, "jass_engine_v2.dll")
        if not os.path.exists(lib_path):
            lib_path = os.path.join(lib_dir, "jass_engine.dll")
        self._lib = ctypes.CDLL(lib_path)
        self._lib.init_tables()

        # State
        self._target = 0
        self._trump = 0
        self._my_points = 0
        self._opp_points = {1: 0, 2: 0, 3: 0}
        self._played_cards = []
        self._trick_number = 0
        self._voids = {1: set(), 2: set(), 3: set()}
        self._belief_tracker = BeliefTracker()

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
            return self._onnx_pick_single(hand, trump, legal)

        leader = table_cards[0][0] if table_cards else (state.leader if hasattr(state, 'leader') else 0)

        best_card = legal[0]
        best_util = float("-inf")

        for card in legal:
            util = self._eval_card(
                hand, trump, self._target, self._my_points,
                table_cards, leader, card, worlds,
                self._trick_number, self._opp_points,
            )
            if util > best_util:
                best_util = util
                best_card = card

        return best_card

    def _eval_card(self, hand, trump, target, my_points, table_cards, leader,
                   my_card, worlds, trick_number, opp_points) -> float:
        """Evaluate candidate card. Opponents in Python (C pick_card), player 0 batched ONNX."""
        W = len(worlds)
        my_hand = [c for c in hand if c != my_card]
        total_tricks = trick_number + len(hand)

        # Init per-world state as numpy arrays for speed
        w_my_pts = np.full(W, my_points, dtype=np.float32)
        w_leaders = np.full(W, leader, dtype=np.int32)
        w_my_hands = [list(my_hand) for _ in range(W)]
        w_opp_hands = []
        w_opp_pts = []
        w_decls = []
        w_played = [set(self._played_cards) | {my_card} for _ in range(W)]

        for wi, (wh, wd) in enumerate(worlds):
            w_opp_hands.append({p: list(cards) for p, cards in wh.items()})
            w_opp_pts.append({p: opp_points.get(p, 0) for p in [1, 2, 3]})
            w_decls.append(wd)

        # Complete current trick (all heuristic — fast)
        for wi in range(W):
            trick = list(table_cards) + [(self.player_id, my_card)]
            lead_suit = CARD_SUIT[trick[0][1]]

            for p, c in table_cards:
                if p in w_opp_hands[wi] and c in w_opp_hands[wi][p]:
                    w_opp_hands[wi][p].remove(c)

            played_set = {p for p, _ in trick}
            for i in range(4):
                p = (leader + i) % 4
                if p in played_set or p == self.player_id:
                    continue
                if p in w_opp_hands[wi] and w_opp_hands[wi][p]:
                    card = self._c_pick(w_opp_hands[wi][p], lead_suit, trump,
                                        w_decls[wi][p] if w_decls[wi] else 40,
                                        w_opp_pts[wi].get(p, 0), trick_number == total_tricks - 1)
                    trick.append((p, card))
                    w_opp_hands[wi][p].remove(card)
                    w_played[wi].add(card)

            if len(trick) >= 4:
                winner = self._trick_winner_fast(trick, lead_suit, trump)
                pts = sum(points(c, trump) for _, c in trick)
                if trick_number == total_tricks - 1:
                    pts += 5
                if winner == self.player_id:
                    w_my_pts[wi] += pts
                elif winner in w_opp_pts[wi]:
                    w_opp_pts[wi][winner] += pts
                w_leaders[wi] = winner

        # Remaining tricks: batched ONNX for player 0, heuristic for opponents
        for t in range(trick_number + 1, total_tricks):
            w_tricks = [[] for _ in range(W)]
            w_lead_suits = [None] * W

            # Opponents before player 0
            for wi in range(W):
                for i in range(4):
                    p = (w_leaders[wi] + i) % 4
                    if p == self.player_id:
                        break
                    if p in w_opp_hands[wi] and w_opp_hands[wi][p]:
                        ls = CARD_SUIT[w_tricks[wi][0][1]] if w_tricks[wi] else -1
                        card = self._c_pick(w_opp_hands[wi][p], ls, trump,
                                            w_decls[wi][p] if w_decls[wi] else 40,
                                            w_opp_pts[wi].get(p, 0), t == total_tricks - 1)
                        w_tricks[wi].append((p, card))
                        w_opp_hands[wi][p].remove(card)
                        w_played[wi].add(card)
                w_lead_suits[wi] = CARD_SUIT[w_tricks[wi][0][1]] if w_tricks[wi] else None

            # Batched ONNX for player 0
            active = [wi for wi in range(W) if w_my_hands[wi]]
            if not active:
                break

            n = len(active)
            obs_batch = np.zeros((n, 204), dtype=np.float32)
            mask_batch = np.zeros((n, 36), dtype=np.float32)
            legal_lists = []

            for bi, wi in enumerate(active):
                ls = w_lead_suits[wi]
                tc = w_tricks[wi] if w_tricks[wi] else None
                leg = legal_moves(w_my_hands[wi], ls, trump, tc)
                legal_lists.append(leg)

                obs = obs_batch[bi]
                for c in w_my_hands[wi]: obs[c] = 1.0
                for c in w_played[wi]:
                    tc_set = {cc for _, cc in (w_tricks[wi] or [])}
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

            # ONE ONNX call for all worlds
            logits, _ = self.sess.run(None, {'obs': obs_batch, 'legal_mask': mask_batch})
            chosen = np.argmax(logits, axis=-1)

            for bi, wi in enumerate(active):
                card = int(chosen[bi])
                if card not in legal_lists[bi]:
                    card = legal_lists[bi][0]
                w_tricks[wi].append((self.player_id, card))
                w_my_hands[wi].remove(card)
                w_played[wi].add(card)
                if w_lead_suits[wi] is None:
                    w_lead_suits[wi] = CARD_SUIT[card]

            # Opponents after player 0
            for wi in range(W):
                my_pos = None
                for i in range(4):
                    if (w_leaders[wi] + i) % 4 == self.player_id:
                        my_pos = i
                        break
                if my_pos is None:
                    continue
                for i in range(my_pos + 1, 4):
                    p = (w_leaders[wi] + i) % 4
                    if p == self.player_id:
                        continue
                    if p in w_opp_hands[wi] and w_opp_hands[wi][p]:
                        ls = w_lead_suits[wi]
                        card = self._c_pick(w_opp_hands[wi][p], ls if ls is not None else -1, trump,
                                            w_decls[wi][p] if w_decls[wi] else 40,
                                            w_opp_pts[wi].get(p, 0), t == total_tricks - 1)
                        w_tricks[wi].append((p, card))
                        w_opp_hands[wi][p].remove(card)
                        w_played[wi].add(card)

            # Resolve tricks
            for wi in range(W):
                if len(w_tricks[wi]) >= 4:
                    ls = CARD_SUIT[w_tricks[wi][0][1]]
                    winner = self._trick_winner_fast(w_tricks[wi], ls, trump)
                    pts = sum(points(c, trump) for _, c in w_tricks[wi])
                    if t == total_tricks - 1:
                        pts += 5
                    if winner == self.player_id:
                        w_my_pts[wi] += pts
                    elif winner in w_opp_pts[wi]:
                        w_opp_pts[wi][winner] += pts
                    w_leaders[wi] = winner

        return float(-(np.abs(target - w_my_pts)).mean())

    def _c_pick(self, hand, lead_suit, trump, target, current_pts, is_last):
        """Use C engine pick_card for fast heuristic play."""
        hand_mask = 0
        for c in hand:
            hand_mask |= (1 << c)
        gap = target - current_pts
        card = self._lib.pick_card(
            ctypes.c_uint64(hand_mask),
            ctypes.c_int(lead_suit if lead_suit is not None else -1),
            ctypes.c_int(trump),
            ctypes.c_int(gap),
            ctypes.c_int(1 if is_last else 0),
        )
        if card < 0 or card not in hand:
            return hand[0]
        return card

    def _trick_winner_fast(self, trick, lead_suit, trump):
        best_p, best_c = trick[0]
        for p, c in trick[1:]:
            cs = CARD_SUIT[c]
            bs = CARD_SUIT[best_c]
            if cs == trump and bs != trump:
                best_p, best_c = p, c
            elif cs == bs and strength(c, trump) > strength(best_c, trump):
                best_p, best_c = p, c
        return best_p

    def _onnx_pick_single(self, hand, trump, legal):
        obs = np.zeros((1, 204), dtype=np.float32)
        mask = np.zeros((1, 36), dtype=np.float32)
        for c in hand: obs[0, c] = 1.0
        for c in range(trump*9, trump*9+9): obs[0, 72+c] = 1.0
        for c in legal: obs[0, 108+c] = 1.0; mask[0, c] = 1.0
        obs[0, 203] = self._target / 157.0
        logits, _ = self.sess.run(None, {'obs': obs, 'legal_mask': mask})
        return int(np.argmax(logits[0]))

    def notify_trick(self, state, trick, winner, pts):
        for p, c in trick:
            self._played_cards.append(c)
            if trick:
                ls = CARD_SUIT[trick[0][1]]
                if CARD_SUIT[c] != ls and p != self.player_id:
                    self._voids.setdefault(p, set()).add(ls)
        if winner == self.player_id:
            self._my_points += pts
        elif winner in self._opp_points:
            self._opp_points[winner] += pts
        self._trick_number += 1

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

"""Hybrid PIMC+Neural player.

Uses PIMC search structure (sample worlds, evaluate candidates) but with
the RL neural net for player 0's move selection during rollouts.

For each candidate card:
  1. Play candidate in current trick
  2. Opponents complete trick using heuristic
  3. Remaining tricks: player 0 uses neural net, opponents use heuristic
  4. Collect final score, compute utility

This gives PIMC's multi-world evaluation quality with neural play quality.
"""

from __future__ import annotations

import os
import sys
import random
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.cards import CARD_SUIT, points, strength
from engine.rules import legal_moves
from engine.game_state import GameState
from bot.heuristics import pick_opponent_target_aware, pick_opponent_greedy, estimate_hand_score
from bot.world_sampler import sample_worlds_with_declarations
from bot.declaration import DeclarationModel
from bot.belief_tracker import BeliefTracker


class HybridPlayer:
    """PIMC search with neural net rollouts for player 0."""

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

        # Load RL networks
        from bot.train_v5 import PlayNet, DeclNet
        self.play_net = PlayNet(204, 512).to(self.device)
        self.decl_net = DeclNet(72, 256).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.play_net.load_state_dict(ckpt["play_net"])
            self.decl_net.load_state_dict(ckpt["decl_net"])

        self.play_net.eval()
        self.decl_net.eval()

        # State tracking
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

        # Use RL declaration net
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

        # Sample worlds
        on_table = [c for _, c in table_cards]
        unknown = [c for c in range(36) if c not in hand and c not in self._played_cards and c not in on_table]
        cards_per_player = len(hand)

        worlds = sample_worlds_with_declarations(
            unknown, cards_per_player, self._voids, self.num_worlds,
            trump, self.rng, belief_tracker=self._belief_tracker,
        )
        if not worlds:
            return self._neural_pick(state, legal)

        # Evaluate each candidate card across worlds
        best_card = legal[0]
        best_util = float("-inf")

        leader = 0
        if table_cards:
            leader = table_cards[0][0]
        elif hasattr(state, 'leader'):
            leader = state.leader

        for card in legal:
            total_util = 0
            for world_hands, world_decls in worlds:
                score = self._simulate_with_neural(
                    hand, trump, self._target, self._my_points,
                    table_cards, leader, card, world_hands, world_decls,
                    self._trick_number, self._opp_points,
                )
                util = -abs(self._target - score)
                total_util += util

            avg_util = total_util / len(worlds)
            if avg_util > best_util:
                best_util = avg_util
                best_card = card

        return best_card

    def notify_trick(self, state: GameState, trick: list[tuple[int, int]], winner: int, pts: int) -> None:
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

    def _neural_pick(self, state: GameState, legal: list[int]) -> int:
        """Single neural net forward pass for fallback."""
        obs = self._encode_play_from_state(state, legal)
        legal_mask = np.zeros(36, dtype=np.float32)
        for c in legal:
            legal_mask[c] = 1.0

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(legal_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.play_net(obs_t, mask_t)
            card = logits.argmax(dim=-1).item()

        return card if card in legal else legal[0]

    def _simulate_with_neural(
        self, hand, trump, target, my_points,
        table_cards, leader, my_card,
        world_hands, world_decls,
        trick_number, opp_points,
    ) -> int:
        """Simulate game: player 0 uses neural net, opponents use heuristic.

        Returns player 0's final points.
        """
        my_hand = [c for c in hand if c != my_card]
        opp = {p: list(cards) for p, cards in world_hands.items()}
        opp_pts = {p: opp_points.get(p, 0) for p in [1, 2, 3]}
        my_pts = my_points
        cur_leader = leader
        played = set(self._played_cards)

        # Complete current trick first
        trick = list(table_cards)
        lead_suit = CARD_SUIT[trick[0][1]] if trick else CARD_SUIT[my_card]
        trick.append((self.player_id, my_card))
        played.add(my_card)

        # Remove table cards from opponent hands
        for p, c in table_cards:
            if p in opp and c in opp[p]:
                opp[p].remove(c)

        # Opponents who haven't played complete the trick
        played_mask = set(p for p, _ in trick)
        for i in range(4):
            p = (cur_leader + i) % 4
            if p in played_mask or p == self.player_id:
                continue
            if p in opp and opp[p]:
                if world_decls is not None:
                    card = pick_opponent_target_aware(
                        opp[p], trick, lead_suit, trump,
                        world_decls[p], opp_pts.get(p, 0), trick_number,
                    )
                else:
                    card = pick_opponent_greedy(opp[p], trick, lead_suit, trump)
                trick.append((p, card))
                opp[p].remove(card)
                played.add(card)

        # Resolve current trick
        if len(trick) >= 4:
            winner = self._trick_winner(trick, lead_suit, trump)
            pts = sum(points(c, trump) for _, c in trick)
            if trick_number == 8:
                pts += 5
            if winner == self.player_id:
                my_pts += pts
            elif winner in opp_pts:
                opp_pts[winner] += pts
            cur_leader = winner
            start_trick = trick_number + 1
        else:
            start_trick = trick_number

        # Remaining tricks: neural for player 0, heuristic for opponents
        total_tricks = trick_number + len(hand)  # total tricks in the round
        for t in range(start_trick, total_tricks):
            trick = []
            lead_suit = None

            for i in range(4):
                p = (cur_leader + i) % 4
                if p == self.player_id:
                    if not my_hand:
                        continue
                    # Neural pick for player 0
                    card = self._neural_pick_sim(
                        my_hand, trick, lead_suit, trump, target, my_pts, t, played
                    )
                    trick.append((p, card))
                    my_hand.remove(card)
                    played.add(card)
                else:
                    if p not in opp or not opp[p]:
                        continue
                    if world_decls is not None:
                        card = pick_opponent_target_aware(
                            opp[p], trick, lead_suit, trump,
                            world_decls[p], opp_pts.get(p, 0), t,
                        )
                    else:
                        card = pick_opponent_greedy(opp[p], trick, lead_suit, trump)
                    trick.append((p, card))
                    opp[p].remove(card)
                    played.add(card)

                if lead_suit is None and trick:
                    lead_suit = CARD_SUIT[trick[0][1]]

            if trick:
                winner = self._trick_winner(trick, lead_suit, trump)
                pts = sum(points(c, trump) for _, c in trick)
                if t == total_tricks - 1:
                    pts += 5
                if winner == self.player_id:
                    my_pts += pts
                elif winner in opp_pts:
                    opp_pts[winner] += pts
                cur_leader = winner

        return my_pts

    def _neural_pick_sim(self, hand, trick, lead_suit, trump, target, my_pts, trick_num, played):
        """Neural net card selection during simulation rollout."""
        legal = legal_moves(hand, lead_suit, trump, trick if trick else None)
        if len(legal) == 1:
            return legal[0]

        # Build minimal observation
        obs = np.zeros(204, dtype=np.float32)
        for c in hand: obs[c] = 1.0
        for c in played:
            if c not in set(cc for _, cc in trick):
                obs[36 + c] = 1.0
        for c in range(trump*9, trump*9+9): obs[72 + c] = 1.0
        for c in legal: obs[108 + c] = 1.0
        for _, c in trick: obs[144 + c] = 1.0
        obs[192] = trick_num / 8.0
        obs[193] = my_pts / 157.0
        obs[194] = max(-1, min(1, (target - my_pts) / 157.0))
        obs[195] = len(trick) / 3.0
        obs[203] = target / 157.0

        legal_mask = np.zeros(36, dtype=np.float32)
        for c in legal: legal_mask[c] = 1.0

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(legal_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.play_net(obs_t, mask_t)
            card = logits.argmax(dim=-1).item()

        return card if card in legal else legal[0]

    def _trick_winner(self, trick, lead_suit, trump):
        if not trick:
            return 0
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
        """72-feature declaration encoding."""
        obs = np.zeros(72, dtype=np.float32)
        for c in hand: obs[c] = 1.0
        obs[36 + trump] = 1.0
        for c in hand:
            if CARD_SUIT[c] == trump: obs[40 + c % 9] = 1.0
        suit_lens = [0]*4
        for c in hand: suit_lens[CARD_SUIT[c]] += 1
        for i in range(4): obs[49+i] = suit_lens[i]/9.0
        suit_pts = [0]*4
        for c in hand: suit_pts[CARD_SUIT[c]] += points(c, trump)
        for i in range(4): obs[53+i] = suit_pts[i]/62.0
        tb = trump*9
        obs[57] = 1.0 if (tb+5) in hand else 0.0
        obs[58] = 1.0 if (tb+3) in hand else 0.0
        obs[59] = 1.0 if (tb+8) in hand else 0.0
        obs[60] = suit_lens[trump]/9.0
        obs[61] = sum(1 for i in range(4) if i!=trump and suit_lens[i]==0)/3.0
        obs[62] = sum(suit_pts)/157.0
        obs[63] = suit_pts[trump]/62.0
        obs[64] = sum(1 for i in range(4) if i!=trump and (i*9+8) in hand)/3.0
        high = sum(1 for c in [tb+5,tb+3,tb+8] if c in hand)
        obs[65] = high/3.0
        obs[66] = sum(1 for i in range(4) if i!=trump and (i*9+8) in hand and suit_lens[i]>=2)/3.0
        obs[67] = sum(1 for i in range(4) if i!=trump and suit_lens[i]>=4)/3.0
        return obs

    def _encode_play_from_state(self, state, legal):
        """Encode from GameState for fallback."""
        obs = np.zeros(204, dtype=np.float32)
        trump = state.trump
        pid = self.player_id
        for c in state.hands[pid]: obs[c] = 1.0
        for c in self._played_cards:
            tc = set(cc for _, cc in (state.current_trick or []))
            if c not in tc: obs[36+c] = 1.0
        for c in range(trump*9, trump*9+9): obs[72+c] = 1.0
        for c in legal: obs[108+c] = 1.0
        if state.current_trick:
            for _, c in state.current_trick: obs[144+c] = 1.0
        obs[192] = self._trick_number/8.0
        obs[193] = state.points[pid]/157.0
        obs[194] = max(-1, min(1, (self._target-state.points[pid])/157.0))
        obs[195] = len(state.current_trick or [])/3.0
        obs[203] = self._target/157.0
        return obs

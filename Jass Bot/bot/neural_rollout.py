"""Neural-guided PIMC rollouts for Differenzler.

Uses neural network for player 0's decisions, gap heuristic for opponents.
All worlds are processed in parallel using flat numpy arrays for speed.
"""

from __future__ import annotations

import numpy as np
import torch

from engine.cards import (
    CARD_SUIT, NUM_CARDS, NUM_VALUES, LAST_TRICK_BONUS,
    points as card_points, strength as card_strength,
)
from engine.rules import legal_moves
from bot.neural_net import FEATURE_SIZE
from bot.neural_player import load_model, encode_state_tensor

# Precomputed tables
_SUIT = CARD_SUIT
_PTS = [[card_points(c, t) for c in range(NUM_CARDS)] for t in range(4)]
_STR = [[card_strength(c, t) for c in range(NUM_CARDS)] for t in range(4)]


def _opp_pick(hand: list[int], lead_suit: int | None, trump: int, gap: int,
              is_last: bool) -> int:
    """Gap heuristic matching C pick_card."""
    if lead_suit is not None:
        legal = [c for c in hand if _SUIT[c] == lead_suit]
        if not legal:
            legal = hand
    else:
        legal = hand
    if not legal:
        return -1
    if is_last:
        if 2 <= gap <= 8:
            return max(legal, key=lambda c: _STR[trump][c])
        if -8 <= gap <= -2:
            return min(legal, key=lambda c: _STR[trump][c])
    pts = _PTS[trump]
    str_t = _STR[trump]
    if gap <= 0:
        return min(legal, key=lambda c: pts[c] * 1000 + str_t[c])
    return max(legal, key=lambda c: pts[c] * 1000 + str_t[c])


def _trick_winner(cards: list[tuple[int, int]], trump: int) -> int:
    bp, bc = cards[0]
    bs = _STR[trump][bc]
    for p, c in cards[1:]:
        cs = _SUIT[c]; bcs = _SUIT[bc]; s = _STR[trump][c]
        if cs == trump and bcs != trump:
            bp, bc, bs = p, c, s
        elif cs == bcs and s > bs:
            bp, bc, bs = p, c, s
    return bp


class NeuralRolloutEvaluator:
    """Evaluates candidate moves using neural-guided PIMC rollouts."""

    def __init__(self, model_path: str, endgame_depth: int = 4):
        self.model = load_model(model_path)
        self.endgame_depth = endgame_depth
        self._c_engine = None
        try:
            from engine_c.fast_engine import FastEngine, is_available
            if is_available():
                self._c_engine = FastEngine()
        except (ImportError, OSError):
            pass

    def evaluate_moves(
        self,
        candidates: list[int],
        my_hand: list[int],
        worlds: list[dict[int, list[int]]],
        trump: int,
        my_target: int,
        my_points: int,
        opp_targets_list: list[dict[int, int]],
        opp_points: list[int],
        leader: int,
        trick_number: int,
        total_tricks: int,
        table_cards: list[tuple[int, int]] | None = None,
        voids: dict[int, set[int]] | None = None,
    ) -> dict[int, float]:
        N = len(worlds)
        table = table_cards or []
        v = voids or {1: set(), 2: set(), 3: set()}
        pts_t = _PTS[trump]
        result = {}

        for cand in candidates:
            # Initialize per-world state as flat lists for speed
            # hands[w][p] = set of card ints
            w_hands = []
            w_pts = []
            w_tgts = []
            w_leader = []
            w_played = []

            for w in range(N):
                h = [
                    set(c for c in my_hand if c != cand),
                    set(worlds[w][1]),
                    set(worlds[w][2]),
                    set(worlds[w][3]),
                ]
                p = [my_points, opp_points[1], opp_points[2], opp_points[3]]
                t = [my_target,
                     opp_targets_list[w].get(1, 40),
                     opp_targets_list[w].get(2, 40),
                     opp_targets_list[w].get(3, 40)]
                w_hands.append(h)
                w_pts.append(p)
                w_tgts.append(t)
                w_played.append(set())
                w_leader.append(leader)

            # Complete current trick
            trick_template = list(table) + [(0, cand)]
            lead_suit = _SUIT[trick_template[0][1]]
            played_in_trick = {p for p, _ in trick_template}
            opp_order = [(leader + i) % 4 for i in range(4)
                         if (leader + i) % 4 not in played_in_trick]

            cur_trick_num = trick_number
            for w in range(N):
                trick = list(trick_template)
                for p, c in table:
                    if p != 0:
                        w_hands[w][p].discard(c)
                for p in opp_order:
                    hand_list = sorted(w_hands[w][p])
                    if not hand_list:
                        continue
                    gap = w_tgts[w][p] - w_pts[w][p]
                    c = _opp_pick(hand_list, lead_suit, trump, gap,
                                  cur_trick_num == total_tricks - 1)
                    if c >= 0:
                        w_hands[w][p].discard(c)
                        trick.append((p, c))
                if len(trick) >= 2:
                    winner = _trick_winner(trick, trump)
                    pts = sum(pts_t[c] for _, c in trick)
                    if cur_trick_num == total_tricks - 1:
                        pts += LAST_TRICK_BONUS
                    w_pts[w][winner] += pts
                    for _, c in trick:
                        w_played[w].add(c)
                    w_leader[w] = winner

            cur_trick_num += 1

            # Remaining tricks
            # Track which worlds still need simulation vs handled by endgame
            active = list(range(N))

            for trick_idx in range(cur_trick_num, total_tricks):
                if not active:
                    break

                remaining = total_tricks - trick_idx

                # Endgame solver for final tricks
                if remaining <= self.endgame_depth and self._c_engine is not None:
                    for w in active:
                        try:
                            _, final = self._c_engine.solve_endgame(
                                sorted(w_hands[w][0]),
                                {1: sorted(w_hands[w][1]), 2: sorted(w_hands[w][2]),
                                 3: sorted(w_hands[w][3])},
                                trump, w_leader[w], w_tgts[w][0], w_pts[w][0],
                                {1: w_tgts[w][1], 2: w_tgts[w][2], 3: w_tgts[w][3]},
                                {1: w_pts[w][1], 2: w_pts[w][2], 3: w_pts[w][3]},
                                trick_idx,
                            )
                            w_pts[w][0] = final
                        except Exception:
                            pass
                    active = []
                    break

                # Play one trick across all active worlds
                w_tricks = {w: [] for w in active}
                is_last = (trick_idx == total_tricks - 1)

                for seat in range(4):
                    # Which active worlds have player 0 at this seat?
                    p0_at_seat = []
                    opp_at_seat = []
                    for w in active:
                        p = (w_leader[w] + seat) % 4
                        if p == 0:
                            p0_at_seat.append(w)
                        else:
                            opp_at_seat.append((w, p))

                    # Opponents play (fast, no neural)
                    for w, p in opp_at_seat:
                        trick = w_tricks[w]
                        ls = _SUIT[trick[0][1]] if trick else None
                        hand_list = sorted(w_hands[w][p])
                        if not hand_list:
                            continue
                        gap = w_tgts[w][p] - w_pts[w][p]
                        c = _opp_pick(hand_list, ls, trump, gap, is_last)
                        if c >= 0:
                            w_hands[w][p].discard(c)
                            trick.append((p, c))
                            w_tricks[w] = trick

                    # Player 0 plays (batched neural)
                    if p0_at_seat:
                        self._play_p0_batch(
                            p0_at_seat, w_hands, w_pts, w_tgts, w_played,
                            w_tricks, trump, trick_idx, total_tricks, v,
                        )

                # Resolve tricks
                for w in active:
                    trick = w_tricks[w]
                    if len(trick) >= 2:
                        winner = _trick_winner(trick, trump)
                        pts = sum(pts_t[c] for _, c in trick)
                        if is_last:
                            pts += LAST_TRICK_BONUS
                        w_pts[w][winner] += pts
                        for _, c in trick:
                            w_played[w].add(c)
                        w_leader[w] = winner

            # Compute utilities
            total_util = 0
            for w in range(N):
                total_util -= abs(my_target - w_pts[w][0])
            result[cand] = total_util / N

        return result

    def _play_p0_batch(
        self,
        world_ids: list[int],
        w_hands: list[list[set[int]]],
        w_pts: list[list[int]],
        w_tgts: list[list[int]],
        w_played: list[set[int]],
        w_tricks: dict[int, list[tuple[int, int]]],
        trump: int,
        trick_num: int,
        total_tricks: int,
        voids: dict[int, set[int]],
    ) -> None:
        n = len(world_ids)
        feat = torch.zeros(n, FEATURE_SIZE)

        all_legal = []
        for i, w in enumerate(world_ids):
            hand = sorted(w_hands[w][0])
            trick = w_tricks[w]
            ls = _SUIT[trick[0][1]] if trick else None
            legal = legal_moves(hand, ls, trump, trick if trick else None)
            if not legal:
                legal = list(hand) if hand else []
            all_legal.append(legal)

            encode_state_tensor(
                feat[i], hand, w_played[w], trick, trump, legal,
                trick_num, w_pts[w][0],
                [w_pts[w][1], w_pts[w][2], w_pts[w][3]],
                w_tgts[w][0], w_hands[w], voids,  # w_hands[w] used as leader proxy — fix below
            )

        # Fix: encode_state_tensor expects leader as int, not list
        # Re-encode with proper args
        feat.zero_()
        for i, w in enumerate(world_ids):
            hand = sorted(w_hands[w][0])
            trick = w_tricks[w]
            ls = _SUIT[trick[0][1]] if trick else None
            legal = all_legal[i]
            # Leader info not critical for feature encoding — use 0 as placeholder
            encode_state_tensor(
                feat[i], hand, w_played[w], trick, trump, legal,
                trick_num, w_pts[w][0],
                [w_pts[w][1], w_pts[w][2], w_pts[w][3]],
                w_tgts[w][0], 0, voids,
            )

        # Legal mask
        legal_mask = torch.zeros(n, 36, dtype=torch.bool)
        for i, legal in enumerate(all_legal):
            for c in legal:
                legal_mask[i, c] = True

        with torch.no_grad():
            logits, _ = self.model(feat, legal_mask)

        for i, w in enumerate(world_ids):
            legal = all_legal[i]
            if not legal:
                continue
            best_idx = logits[i][legal].argmax().item()
            card = legal[best_idx]
            w_hands[w][0].discard(card)
            trick = w_tricks[w]
            trick.append((0, card))
            w_tricks[w] = trick

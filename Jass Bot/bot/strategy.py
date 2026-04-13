"""PIMC-based strategy engine for Differenzler.

Operates on integer card IDs. Provides declaration (ansage) and
move selection (zug) using Perfect Information Monte Carlo search
with full game tree solve.

Supports two modes:
- Greedy opponents (original): opponents maximize points
- Target-aware opponents: opponents play to hit their own targets
"""

from __future__ import annotations

import random
from typing import Any

from engine.cards import CARD_SUIT, points, strength
from engine.rules import legal_moves
from bot.heuristics import (
    pick_opponent_greedy,
    pick_opponent_target_aware,
    pick_sim_target_aware,
    estimate_hand_score,
)
from bot.world_sampler import sample_worlds, sample_worlds_with_declarations
from bot.endgame import EndgameSolver

# Try to load C engine for accelerated move selection
_USE_C_MOVES = False
_c_engine = None
try:
    from engine_c.fast_engine import FastEngine, is_available
    if is_available():
        _c_engine = FastEngine()
        _USE_C_MOVES = True
except (ImportError, OSError):
    pass


class StrategyEngine:
    """PIMC strategy engine for Differenzler.

    Objective: minimize E[|target - final_points|]
    Same utility drives both declaration search and move selection.
    """

    def __init__(
        self,
        num_worlds: int = 25,
        ansage_worlds: int = 40,
        use_target_aware_opponents: bool = False,
        endgame_depth: int = 0,
    ):
        self.num_worlds = num_worlds
        self.ansage_worlds = ansage_worlds
        self.use_target_aware_opponents = use_target_aware_opponents
        self.endgame_depth = endgame_depth
        self._endgame_solver: EndgameSolver | None = None
        if endgame_depth > 0:
            self._endgame_solver = EndgameSolver(max_depth=endgame_depth)
        self.player_voids: dict[int, set[int]] = {1: set(), 2: set(), 3: set()}
        self.tt: dict[Any, int] = {}

    def reset(self) -> None:
        self.player_voids = {1: set(), 2: set(), 3: set()}
        self.tt = {}

    def record_void(self, player: int, suit: int) -> None:
        if player > 0:
            self.player_voids[player].add(suit)

    @staticmethod
    def utility(target: int, final_points: int) -> int:
        return -abs(target - final_points)

    # =========================================================================
    # ANSAGE (declaration) — sweep-based (original)
    # =========================================================================

    def ansage(
        self,
        hand: list[int],
        trump: int,
        rng: random.Random | None = None,
    ) -> int:
        """Find the declaration that maximizes E[utility] under PIMC (sweep-based)."""
        if not hand:
            return 0

        self.reset()
        unknown = self._unknown_cards(hand, [], [])

        if self.use_target_aware_opponents:
            raw_worlds = sample_worlds_with_declarations(
                unknown, len(hand), self.player_voids, self.ansage_worlds, trump, rng
            )
            worlds_for_sweep = []
            for w_hands, w_decls in raw_worlds:
                worlds_for_sweep.append((w_hands, w_decls))
        else:
            plain_worlds = sample_worlds(
                unknown, len(hand), self.player_voids, self.ansage_worlds, rng
            )
            worlds_for_sweep = [(w, None) for w in plain_worlds]

        if not worlds_for_sweep:
            return 0

        # Coarse search: 0-157 in steps of 5
        best_target, best_eu = 0, float("-inf")
        for t in range(0, 158, 5):
            eu = self._eval_target_sweep(hand, trump, t, worlds_for_sweep)
            if eu > best_eu:
                best_eu, best_target = eu, t

        # Fine search: ±6 around best
        for t in range(max(0, best_target - 6), min(158, best_target + 7)):
            eu = self._eval_target_sweep(hand, trump, t, worlds_for_sweep)
            if eu > best_eu:
                best_eu, best_target = eu, t

        return best_target

    def _eval_target_sweep(
        self,
        hand: list[int],
        trump: int,
        target: int,
        worlds: list[tuple[dict[int, list[int]], dict[int, int] | None]],
    ) -> float:
        total = sum(
            self.utility(target, self._sim_game(hand, trump, target, w_hands, w_decls))
            for w_hands, w_decls in worlds
        )
        return total / len(worlds)

    def _sim_game(
        self,
        hand: list[int],
        trump: int,
        target: int,
        world: dict[int, list[int]],
        opp_decls: dict[int, int] | None = None,
    ) -> int:
        """Simulate full game with heuristic play. Returns player 0's final points."""
        my_hand = list(hand)
        opp = {p: list(cards) for p, cards in world.items()}
        opp_pts = {1: 0, 2: 0, 3: 0}
        my_pts = 0
        leader = 0

        for trick_num in range(len(hand)):
            if not my_hand:
                break
            trick: list[tuple[int, int]] = []
            lead_suit: int | None = None

            for i in range(4):
                p = (leader + i) % 4
                if p == 0:
                    card = pick_sim_target_aware(
                        my_hand, trick, lead_suit, trump, target, my_pts
                    )
                    trick.append((0, card))
                    my_hand.remove(card)
                elif opp[p]:
                    if self.use_target_aware_opponents and opp_decls is not None:
                        card = pick_opponent_target_aware(
                            opp[p], trick, lead_suit, trump,
                            opp_decls[p], opp_pts[p], trick_num,
                        )
                    else:
                        card = pick_opponent_greedy(opp[p], trick, lead_suit, trump)
                    trick.append((p, card))
                    opp[p].remove(card)

                if lead_suit is None and trick:
                    lead_suit = CARD_SUIT[trick[0][1]]

            winner = self._trick_winner(trick, lead_suit, trump)
            pts = sum(points(c, trump) for _, c in trick)
            if winner == 0:
                my_pts += pts
            elif winner in opp_pts:
                opp_pts[winner] += pts
            leader = winner

        return my_pts

    # =========================================================================
    # ZUG (move selection)
    # =========================================================================

    def zug(
        self,
        hand: list[int],
        trump: int,
        target: int,
        current_points: int,
        table_cards: list[tuple[int, int]],
        leader: int,
        played_cards: list[int],
        trick_number: int = 0,
        rng: random.Random | None = None,
        belief_tracker: object | None = None,
        opp_points: list[int] | None = None,
    ) -> int | None:
        """Select the card maximizing E[utility]."""
        lead_suit = CARD_SUIT[table_cards[0][1]] if table_cards else None
        playable = legal_moves(hand, lead_suit, trump, table_cards if table_cards else None)
        if not playable:
            return None
        if len(playable) == 1:
            return playable[0]

        self.tt = {}
        on_table = [c for _, c in table_cards]
        unknown = self._unknown_cards(hand, played_cards, on_table)
        cards_per_player = len(hand)

        if self.use_target_aware_opponents:
            raw_worlds = sample_worlds_with_declarations(
                unknown, cards_per_player, self.player_voids, self.num_worlds,
                trump, rng, belief_tracker=belief_tracker,
            )
            worlds: list[tuple[dict[int, list[int]], dict[int, int] | None]] = [
                (w_h, w_d) for w_h, w_d in raw_worlds
            ]
        else:
            plain_worlds = sample_worlds(
                unknown, cards_per_player, self.player_voids, self.num_worlds, rng
            )
            worlds = [(w, None) for w in plain_worlds]

        if not worlds:
            return playable[0]

        # C-accelerated move evaluation: batch all candidates × all worlds in one call
        if _USE_C_MOVES and _c_engine is not None:
            worlds_hands = [w_h for w_h, _ in worlds]
            opp_tgts = [w_d for _, w_d in worlds] if worlds[0][1] is not None else None
            total_tricks = len(hand) + trick_number

            # Pass actual opponent points (same for all worlds — it's observed state)
            opp_pts_per_world = None
            if opp_points is not None:
                opp_pts_dict = {1: opp_points[1], 2: opp_points[2], 3: opp_points[3]}
                opp_pts_per_world = [opp_pts_dict] * len(worlds)

            card_utils = _c_engine.evaluate_moves(
                candidates=playable,
                my_hand=hand,
                worlds=worlds_hands,
                trump=trump,
                my_target=target,
                my_points=current_points,
                opp_targets_list=opp_tgts,
                opp_points_list=opp_pts_per_world,
                leader=leader,
                trick_number=trick_number,
                total_tricks=total_tricks,
                table_cards=table_cards if table_cards else None,
            )
            return max(playable, key=lambda c: card_utils.get(c, float("-inf")))

        # Python fallback
        best_card = playable[0]
        best_eu = float("-inf")

        for card in playable:
            total = sum(
                self.utility(
                    target,
                    self._solve(hand, trump, target, current_points,
                                table_cards, leader, card, w_h, w_d, trick_number),
                )
                for w_h, w_d in worlds
            )
            eu = total / len(worlds)
            if eu > best_eu:
                best_eu = eu
                best_card = card

        return best_card

    def _solve(
        self,
        hand: list[int],
        trump: int,
        target: int,
        current_points: int,
        table_cards: list[tuple[int, int]],
        leader: int,
        my_card: int,
        world: dict[int, list[int]],
        opp_decls: dict[int, int] | None,
        trick_number: int,
    ) -> int:
        """Solve game tree after playing my_card. Returns final points."""
        trick = list(table_cards) + [(0, my_card)]
        lead_suit = CARD_SUIT[table_cards[0][1]] if table_cards else CARD_SUIT[my_card]

        opp = {p: list(cards) for p, cards in world.items()}
        opp_pts = {1: 0, 2: 0, 3: 0}  # approximate: we don't track mid-game opp points in solve

        played_in_trick = {p for p, _ in trick}
        for i in range(4):
            p = (leader + i) % 4
            if p not in played_in_trick and opp[p]:
                if self.use_target_aware_opponents and opp_decls is not None:
                    card = pick_opponent_target_aware(
                        opp[p], trick, lead_suit, trump,
                        opp_decls[p], opp_pts[p], trick_number,
                    )
                else:
                    card = pick_opponent_greedy(opp[p], trick, lead_suit, trump)
                trick.append((p, card))
                opp[p].remove(card)

        winner = self._trick_winner(trick, lead_suit, trump)
        pts = sum(points(c, trump) for _, c in trick)
        my_pts = current_points + (pts if winner == 0 else 0)

        my_rem = [c for c in hand if c != my_card]
        my_pts += self._solve_remaining(
            my_rem, opp, trump, target, my_pts, winner, opp_decls, opp_pts, trick_number + 1
        )
        return my_pts

    def _solve_remaining(
        self,
        my_hand: list[int],
        opp: dict[int, list[int]],
        trump: int,
        target: int,
        cur_pts: int,
        leader: int,
        opp_decls: dict[int, int] | None = None,
        opp_pts: dict[int, int] | None = None,
        trick_number: int = 0,
        depth: int = 0,
    ) -> int:
        if not my_hand or depth > 8:
            return 0

        remaining_tricks = len(my_hand)

        # Endgame solver: exact search for final tricks
        if (self._endgame_solver is not None
                and remaining_tricks <= self.endgame_depth):
            o_targets = opp_decls or {p: 40 for p in [1, 2, 3]}
            o_pts = opp_pts or {1: 0, 2: 0, 3: 0}
            _, final_pts = self._endgame_solver.solve(
                my_hand, opp, trump, leader, target, cur_pts,
                o_targets, o_pts, trick_number,
            )
            return final_pts - cur_pts

        h = self._state_hash(my_hand, opp, leader, cur_pts)
        if h in self.tt:
            return self.tt[h]

        if leader == 0:
            result = self._solve_as_leader(
                my_hand, opp, trump, target, cur_pts, depth,
                opp_decls, opp_pts, trick_number,
            )
        else:
            result = self._solve_as_follower(
                my_hand, opp, trump, target, cur_pts, leader, depth,
                opp_decls, opp_pts, trick_number,
            )

        self.tt[h] = result
        return result

    def _opp_play(
        self, hand: list[int], trick: list[tuple[int, int]],
        lead_suit: int | None, trump: int, player: int,
        opp_decls: dict[int, int] | None, opp_pts: dict[int, int] | None,
        trick_number: int,
    ) -> int:
        """Dispatch to greedy or target-aware opponent play."""
        if self.use_target_aware_opponents and opp_decls is not None and opp_pts is not None:
            return pick_opponent_target_aware(
                hand, trick, lead_suit, trump,
                opp_decls[player], opp_pts.get(player, 0), trick_number,
            )
        return pick_opponent_greedy(hand, trick, lead_suit, trump)

    def _solve_as_leader(
        self, my_hand: list[int], opp: dict[int, list[int]],
        trump: int, target: int, cur_pts: int, depth: int,
        opp_decls: dict[int, int] | None, opp_pts: dict[int, int] | None,
        trick_number: int,
    ) -> int:
        best_u = None
        best_add = 0

        for card in my_hand:
            trick: list[tuple[int, int]] = [(0, card)]
            lead_suit = CARD_SUIT[card]
            opp2 = {p: list(h) for p, h in opp.items()}
            opp_pts2 = dict(opp_pts) if opp_pts else {1: 0, 2: 0, 3: 0}

            for p in [1, 2, 3]:
                if opp2[p]:
                    c = self._opp_play(
                        opp2[p], trick, lead_suit, trump, p,
                        opp_decls, opp_pts2, trick_number,
                    )
                    trick.append((p, c))
                    opp2[p].remove(c)

            winner = self._trick_winner(trick, lead_suit, trump)
            pts = sum(points(c, trump) for _, c in trick)
            gain = pts if winner == 0 else 0

            # Update opp points
            if winner in opp_pts2 and winner != 0:
                opp_pts2[winner] += pts

            my_rem = [c for c in my_hand if c != card]
            if not my_rem:
                add = gain
            else:
                add = gain + self._solve_remaining(
                    my_rem, opp2, trump, target, cur_pts + gain, winner,
                    opp_decls, opp_pts2, trick_number + 1, depth + 1,
                )

            u = self.utility(target, cur_pts + add)
            if best_u is None or u > best_u:
                best_u = u
                best_add = add

        return best_add

    def _solve_as_follower(
        self, my_hand: list[int], opp: dict[int, list[int]],
        trump: int, target: int, cur_pts: int, leader: int, depth: int,
        opp_decls: dict[int, int] | None, opp_pts: dict[int, int] | None,
        trick_number: int,
    ) -> int:
        trick: list[tuple[int, int]] = []
        lead_suit: int | None = None
        opp2 = {p: list(h) for p, h in opp.items()}
        opp_pts2 = dict(opp_pts) if opp_pts else {1: 0, 2: 0, 3: 0}

        for i in range(4):
            p = (leader + i) % 4
            if p == 0:
                break
            if opp2[p]:
                c = self._opp_play(
                    opp2[p], trick, lead_suit, trump, p,
                    opp_decls, opp_pts2, trick_number,
                )
                trick.append((p, c))
                opp2[p].remove(c)
                if lead_suit is None:
                    lead_suit = CARD_SUIT[c]

        playable = legal_moves(my_hand, lead_suit, trump, trick if trick else None)
        if not playable:
            playable = list(my_hand)

        best_u = None
        best_add = 0

        for card in playable:
            t2 = list(trick) + [(0, card)]
            opp3 = {p: list(h) for p, h in opp2.items()}
            opp_pts3 = dict(opp_pts2)

            played_in = {p for p, _ in t2}
            for i in range(4):
                p = (leader + i) % 4
                if p not in played_in and opp3[p]:
                    c = self._opp_play(
                        opp3[p], t2, lead_suit, trump, p,
                        opp_decls, opp_pts3, trick_number,
                    )
                    t2.append((p, c))
                    opp3[p].remove(c)

            winner = self._trick_winner(t2, lead_suit, trump)
            pts = sum(points(c, trump) for _, c in t2)
            gain = pts if winner == 0 else 0

            if winner in opp_pts3 and winner != 0:
                opp_pts3[winner] += pts

            my_rem = [c for c in my_hand if c != card]
            if not my_rem:
                add = gain
            else:
                add = gain + self._solve_remaining(
                    my_rem, opp3, trump, target, cur_pts + gain, winner,
                    opp_decls, opp_pts3, trick_number + 1, depth + 1,
                )

            u = self.utility(target, cur_pts + add)
            if best_u is None or u > best_u:
                best_u = u
                best_add = add

        return best_add

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _trick_winner(
        trick: list[tuple[int, int]], lead_suit: int | None, trump: int,
    ) -> int:
        if not trick:
            return 0
        best_p, best_c = trick[0]
        for p, c in trick[1:]:
            c_suit = CARD_SUIT[c]
            b_suit = CARD_SUIT[best_c]
            if c_suit == trump and b_suit != trump:
                best_p, best_c = p, c
            elif c_suit == b_suit and strength(c, trump) > strength(best_c, trump):
                best_p, best_c = p, c
        return best_p

    @staticmethod
    def _unknown_cards(
        hand: list[int], played: list[int], on_table: list[int],
    ) -> list[int]:
        known = set(hand) | set(played) | set(on_table)
        return [c for c in range(36) if c not in known]

    @staticmethod
    def _state_hash(
        my_hand: list[int], opp: dict[int, list[int]], leader: int, pts: int,
    ) -> int:
        h = tuple(sorted(my_hand))
        o = tuple(tuple(sorted(opp.get(p, []))) for p in [1, 2, 3])
        return hash((h, o, leader, pts))

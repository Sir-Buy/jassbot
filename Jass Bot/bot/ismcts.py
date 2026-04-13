"""SO-ISMCTS (Single Observer Information Set Monte Carlo Tree Search) for Differenzler.

Each iteration:
  1. Determinize: sample opponent hands + declarations
  2. Select: UCB1 walk down tree, filtering by compatibility
  3. Expand: add one new child
  4. Rollout: C engine heuristic to game end
  5. Backpropagate: update visit counts and utility sums
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from engine.cards import CARD_SUIT, NUM_CARDS, strength as card_strength, points as card_points
from engine.rules import legal_moves
from engine_c.fast_engine import FastEngine

_engine: FastEngine | None = None

def _get_engine() -> FastEngine:
    global _engine
    if _engine is None:
        _engine = FastEngine()
    return _engine


class Node:
    __slots__ = ('action', 'player', 'parent', 'children', 'visits', 'total_utility')

    def __init__(self, action: int | None, player: int, parent: Node | None = None):
        self.action = action
        self.player = player
        self.parent = parent
        self.children: dict[int, Node] = {}
        self.visits = 0
        self.total_utility = 0.0

    def ucb(self, exploration: float, log_parent: float) -> float:
        if self.visits == 0:
            return float('inf')
        return self.total_utility / self.visits + exploration * math.sqrt(log_parent / self.visits)

    def best_child_ucb(self, compatible: list[int], exploration: float) -> Node | None:
        best = None
        best_val = -1.0
        lp = math.log(self.visits) if self.visits > 0 else 0.0
        for a in compatible:
            child = self.children.get(a)
            if child is not None:
                val = child.ucb(exploration, lp)
                if val > best_val:
                    best_val = val
                    best = child
        return best

    def expand_one(self, action: int, next_player: int) -> Node:
        child = Node(action=action, player=next_player, parent=self)
        self.children[action] = child
        return child


def _normalize_utility(utility: int) -> float:
    return (utility + 157) / 157.0


def _trick_winner_inline(trick: list[tuple[int, int]], trump: int) -> int:
    best_p, best_c = trick[0]
    best_s = card_strength(best_c, trump)
    for p, c in trick[1:]:
        cs = CARD_SUIT[c]
        bs = CARD_SUIT[best_c]
        if cs == trump and bs != trump:
            best_p, best_c = p, c
            best_s = card_strength(c, trump)
        elif cs == bs and card_strength(c, trump) > best_s:
            best_p, best_c = p, c
            best_s = card_strength(c, trump)
    return best_p


class ISMCTSEngine:
    def __init__(self, time_limit_ms: int = 2000, exploration: float = 0.7,
                 endgame_depth: int = 4):
        self.time_limit_ms = time_limit_ms
        self.exploration = exploration
        self.endgame_depth = endgame_depth
        self.last_iterations = 0
        self.last_iter_per_sec = 0.0

    def search(
        self,
        my_hand: list[int],
        trump: int,
        target: int,
        my_points: int,
        opp_points: list[int],
        table_cards: list[tuple[int, int]],
        leader: int,
        trick_number: int,
        played_cards: list[int],
        voids: dict[int, set[int]],
        rng: random.Random,
        belief_tracker: Any | None = None,
    ) -> tuple[int, dict[int, tuple[int, float]]]:
        lead_suit = CARD_SUIT[table_cards[0][1]] if table_cards else None
        my_legal = legal_moves(my_hand, lead_suit, trump,
                               table_cards if table_cards else None)
        if not my_legal:
            return my_hand[0] if my_hand else -1, {}
        if len(my_legal) == 1:
            return my_legal[0], {my_legal[0]: (1, 0.0)}

        root = Node(action=None, player=0)
        total_tricks = len(my_hand) + trick_number
        unknown = [c for c in range(NUM_CARDS)
                   if c not in set(my_hand) and c not in set(played_cards)
                   and c not in {cc for _, cc in table_cards}]
        cards_per_opp = len(my_hand)

        eng = _get_engine()
        exploration = self.exploration
        endgame_depth = self.endgame_depth

        # Pre-import for speed
        from bot.world_sampler import _generate_one_world
        from bot.heuristics import estimate_hand_score
        _belief_sample = belief_tracker.sample_declaration if belief_tracker else None

        # Cache initial state
        init_opp_pts = [opp_points[1], opp_points[2], opp_points[3]]
        init_table = list(table_cards)

        iterations = 0
        start = time.perf_counter()
        deadline = start + self.time_limit_ms / 1000.0

        while time.perf_counter() < deadline:
            # 1. DETERMINIZE
            det = _generate_one_world(unknown, cards_per_opp, voids, rng)
            if det is None:
                cards = list(unknown)
                rng.shuffle(cards)
                det = {1: cards[:cards_per_opp], 2: cards[cards_per_opp:2*cards_per_opp],
                       3: cards[2*cards_per_opp:3*cards_per_opp]}

            if _belief_sample is not None:
                det_tgt = {p: _belief_sample(p, rng) for p in (1, 2, 3)}
            else:
                det_tgt = {p: estimate_hand_score(det[p], trump) for p in (1, 2, 3)}

            # 2-3. SELECT + EXPAND
            # Build game state by replaying actions down the tree
            hands = [list(my_hand), list(det[1]), list(det[2]), list(det[3])]
            pts = [my_points, init_opp_pts[0], init_opp_pts[1], init_opp_pts[2]]
            trick = list(init_table)
            cur_leader = leader
            cur_trick_num = trick_number

            node = root
            expanded = False

            while cur_trick_num < total_tricks:
                if not any(hands):
                    break

                player = node.player
                ls = CARD_SUIT[trick[0][1]] if trick else None
                det_legal = legal_moves(hands[player], ls, trump,
                                        trick if trick else None)
                if not det_legal:
                    break

                # Find untried compatible actions
                compatible_existing = []
                compatible_untried = []
                for a in det_legal:
                    if a in node.children:
                        compatible_existing.append(a)
                    else:
                        compatible_untried.append(a)

                if compatible_untried:
                    # EXPAND
                    action = compatible_untried[0]
                    hands, pts, trick, cur_leader, cur_trick_num, np = \
                        _advance_state(hands, pts, trick, cur_leader,
                                       cur_trick_num, total_tricks, trump, player, action)
                    node = node.expand_one(action, np)
                    expanded = True
                    break

                if not compatible_existing:
                    break

                # SELECT via UCB
                best_child = node.best_child_ucb(compatible_existing, exploration)
                if best_child is None:
                    break

                action = best_child.action
                hands, pts, trick, cur_leader, cur_trick_num, np = \
                    _advance_state(hands, pts, trick, cur_leader,
                                   cur_trick_num, total_tricks, trump, player, action)
                node = best_child

            # 4. ROLLOUT
            targets = [target, det_tgt.get(1, 40), det_tgt.get(2, 40), det_tgt.get(3, 40)]
            remaining = total_tricks - cur_trick_num

            if remaining > 0 and remaining <= endgame_depth and not trick:
                try:
                    _, final = eng.solve_endgame(
                        hands[0],
                        {1: hands[1], 2: hands[2], 3: hands[3]},
                        trump, cur_leader, target, pts[0],
                        {1: targets[1], 2: targets[2], 3: targets[3]},
                        {1: pts[1], 2: pts[2], 3: pts[3]},
                        cur_trick_num,
                    )
                except Exception:
                    final = eng.rollout(
                        {i: hands[i] for i in range(4)}, pts, targets,
                        trump, cur_leader, cur_trick_num, total_tricks,
                        trick if trick else None,
                    )
            elif remaining > 0 or trick:
                final = eng.rollout(
                    {i: hands[i] for i in range(4)}, pts, targets,
                    trump, cur_leader, cur_trick_num, total_tricks,
                    trick if trick else None,
                )
            else:
                final = pts[0]

            # 5. BACKPROPAGATE
            norm_u = _normalize_utility(-abs(target - final))
            cur = node
            while cur is not None:
                cur.visits += 1
                cur.total_utility += norm_u
                cur = cur.parent

            iterations += 1

        elapsed = time.perf_counter() - start
        self.last_iterations = iterations
        self.last_iter_per_sec = iterations / elapsed if elapsed > 0 else 0

        card_stats = {}
        for card, child in root.children.items():
            avg_u = (child.total_utility / child.visits * 157 - 157) if child.visits > 0 else 0
            card_stats[card] = (child.visits, avg_u)

        if not card_stats:
            return my_legal[0], {}

        best = max(card_stats, key=lambda c: card_stats[c][0])
        return best, card_stats


def _advance_state(
    hands: list[list[int]],
    points: list[int],
    trick: list[tuple[int, int]],
    leader: int,
    trick_num: int,
    total_tricks: int,
    trump: int,
    player: int,
    action: int,
) -> tuple[list[list[int]], list[int], list[tuple[int, int]], int, int, int]:
    """Play a card, advance game state. Returns (hands, points, trick, leader, trick_num, next_player).

    Mutates hands and points in-place for speed (caller must copy if needed).
    """
    hands[player] = [c for c in hands[player] if c != action]
    trick = list(trick)  # shallow copy — just tuples
    trick.append((player, action))

    if len(trick) == 4:
        winner = _trick_winner_inline(trick, trump)
        pts = sum(card_points(c, trump) for _, c in trick)
        if trick_num == total_tricks - 1:
            pts += 5
        points[winner] += pts
        trick_num += 1
        return hands, points, [], winner, trick_num, winner
    else:
        next_player = (leader + len(trick)) % 4
        return hands, points, trick, leader, trick_num, next_player

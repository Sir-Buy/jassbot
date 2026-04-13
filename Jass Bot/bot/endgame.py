"""Exact endgame solver for the last 2-3 tricks using max^n search.

Each player independently maximizes their own utility:
  U_i = -|target_i - final_points_i|

Optimized: uses flat tuples and integer-packed state for fast hashing.
No object creation in the hot loop.
"""

from __future__ import annotations

from engine.cards import (
    CARD_SUIT, CARD_STRENGTH, CARD_POINTS, LAST_TRICK_BONUS,
    NUM_VALUES,
)

# Precompute arrays for direct indexing (avoid function call overhead)
_SUIT = CARD_SUIT
_PTS_NT = [CARD_POINTS[c][0] for c in range(36)]
_PTS_T = [CARD_POINTS[c][1] for c in range(36)]
_STR_NT = [CARD_STRENGTH[c][0] for c in range(36)]
_STR_T = [CARD_STRENGTH[c][1] for c in range(36)]


_USE_C = False
_c_engine = None

try:
    from engine_c.fast_engine import FastEngine, is_available
    if is_available():
        _c_engine = FastEngine()
        _USE_C = True
except (ImportError, OSError):
    pass


class EndgameSolver:
    """Exact solver for endgame positions (last 2-3 tricks).

    Automatically uses C engine when available for ~5x speedup.
    """

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self._tt: dict[int, tuple[float, float, float, float]] = {}

    def should_activate(self, remaining_tricks: int) -> bool:
        return remaining_tricks <= self.max_depth

    def solve(
        self,
        my_hand: list[int],
        opponent_hands: dict[int, list[int]],
        trump: int,
        leader: int,
        my_target: int,
        my_points: int,
        opponent_targets: dict[int, int],
        opponent_points: dict[int, int],
        trick_number: int,
    ) -> tuple[int | None, int]:
        """Solve endgame position exactly.

        Returns (best_card, expected_final_points) for player 0.
        """
        # Try C engine first
        if _USE_C and _c_engine is not None:
            return _c_engine.solve_endgame(
                my_hand, opponent_hands, trump, leader,
                my_target, my_points, opponent_targets, opponent_points,
                trick_number,
            )

        self._tt = {}
        self._trump = trump

        # Precompute per-card points and strength for this trump
        if trump >= 0:
            self._pts = _PTS_T if True else _PTS_NT  # we need per-card
            self._card_pts = [_PTS_T[c] if _SUIT[c] == trump else _PTS_NT[c] for c in range(36)]
            self._card_str = [_STR_T[c] + 100 if _SUIT[c] == trump else _STR_NT[c] for c in range(36)]

        # Pack hands as frozensets for fast manipulation
        hands = (
            tuple(sorted(my_hand)),
            tuple(sorted(opponent_hands.get(1, []))),
            tuple(sorted(opponent_hands.get(2, []))),
            tuple(sorted(opponent_hands.get(3, []))),
        )
        targets = (my_target, opponent_targets.get(1, 40),
                   opponent_targets.get(2, 40), opponent_targets.get(3, 40))
        pts = (my_points, opponent_points.get(1, 0),
               opponent_points.get(2, 0), opponent_points.get(3, 0))

        # Find best card for player 0
        if not my_hand:
            return None, my_points

        best_card, best_pts = self._find_best(hands, pts, targets, leader, trick_number, ())
        return best_card, best_pts

    def _find_best(self, hands, pts, targets, leader, trick_num, trick_cards):
        """Find best card for player 0 when it's their turn."""
        active = (leader + len(trick_cards)) % 4
        hand = hands[active]
        if not hand:
            return None, pts[0]

        lead_suit = _SUIT[trick_cards[0][1]] if trick_cards else -1
        legal = self._legal_fast(hand, lead_suit)

        if active != 0:
            # Not our turn — solve with max^n
            utils = self._solve(hands, pts, targets, leader, trick_num, trick_cards)
            return None, utils[0]

        if len(legal) == 1:
            new_tc = trick_cards + ((0, legal[0]),)
            if len(new_tc) == 4:
                new_hands, new_pts, new_leader, new_tn = self._resolve_trick(
                    hands, pts, 0, legal[0], leader, trick_num, trick_cards
                )
                utils = self._solve(new_hands, new_pts, targets, new_leader, new_tn, ())
            else:
                new_hands = (
                    tuple(c for c in hands[0] if c != legal[0]),
                    hands[1], hands[2], hands[3],
                )
                utils = self._solve(new_hands, pts, targets, leader, trick_num, new_tc)
            return legal[0], utils[0]

        best_card = legal[0]
        best_u = -999

        for card in legal:
            new_tc = trick_cards + ((0, card),)
            new_h0 = tuple(c for c in hands[0] if c != card)

            if len(new_tc) == 4:
                new_hands, new_pts, new_leader, new_tn = self._resolve_trick_from_tc(
                    (new_h0, hands[1], hands[2], hands[3]),
                    pts, leader, trick_num, new_tc
                )
                utils = self._solve(new_hands, new_pts, targets, new_leader, new_tn, ())
            else:
                new_hands = (new_h0, hands[1], hands[2], hands[3])
                utils = self._solve(new_hands, pts, targets, leader, trick_num, new_tc)

            u = -abs(targets[0] - utils[0])
            if u > best_u:
                best_u = u
                best_card = card
                best_final = utils[0]
                if u == 0:
                    break

        return best_card, best_final

    def _solve(self, hands, pts, targets, leader, trick_num, trick_cards):
        """Max^n solve. Returns (p0_pts, p1_pts, p2_pts, p3_pts) final points."""
        # Terminal check
        if not any(hands) and not trick_cards:
            return pts

        # TT lookup
        h = hash((hands, pts, leader, trick_num, trick_cards))
        cached = self._tt.get(h)
        if cached is not None:
            return cached

        active = (leader + len(trick_cards)) % 4
        hand = hands[active]
        if not hand:
            return pts

        lead_suit = _SUIT[trick_cards[0][1]] if trick_cards else -1
        legal = self._legal_fast(hand, lead_suit)
        if not legal:
            return pts

        best_utils = None
        card_pts = self._card_pts
        card_str = self._card_str

        for card in legal:
            # Remove card from active player's hand
            new_hand = tuple(c for c in hand if c != card)
            new_tc = trick_cards + ((active, card),)

            if len(new_tc) == 4:
                # Resolve trick
                new_hands_list = list(hands)
                new_hands_list[active] = new_hand
                new_hands = tuple(new_hands_list)
                new_hands, new_pts, new_leader, new_tn = self._resolve_trick_from_tc(
                    new_hands, pts, leader, trick_num, new_tc
                )
                utils = self._solve(new_hands, new_pts, targets, new_leader, new_tn, ())
            else:
                new_hands_list = list(hands)
                new_hands_list[active] = new_hand
                utils = self._solve(tuple(new_hands_list), pts, targets, leader, trick_num, new_tc)

            if best_utils is None:
                best_utils = utils
            else:
                # Active player picks action maximizing own utility
                my_u_new = -abs(targets[active] - utils[active])
                my_u_old = -abs(targets[active] - best_utils[active])
                if my_u_new > my_u_old:
                    best_utils = utils
                    if my_u_new == 0:
                        break

        result = best_utils if best_utils is not None else pts
        self._tt[h] = result
        return result

    def _resolve_trick_from_tc(self, hands, pts, leader, trick_num, trick_cards):
        """Resolve a complete trick (4 cards). Returns (new_hands, new_pts, winner, new_trick_num)."""
        lead_suit = _SUIT[trick_cards[0][1]]
        card_str = self._card_str

        # Find winner
        best_p = trick_cards[0][0]
        best_c = trick_cards[0][1]
        best_s = card_str[best_c]

        for i in range(1, 4):
            p, c = trick_cards[i]
            cs = _SUIT[c]
            bs = _SUIT[best_c]
            s = card_str[c]

            if cs == self._trump and bs != self._trump:
                best_p, best_c, best_s = p, c, s
            elif cs == bs and s > best_s:
                best_p, best_c, best_s = p, c, s

        # Sum points
        card_pts = self._card_pts
        trick_pts = sum(card_pts[c] for _, c in trick_cards)
        if trick_num == 8:  # last trick (0-indexed)
            trick_pts += LAST_TRICK_BONUS

        new_pts = list(pts)
        new_pts[best_p] += trick_pts

        return hands, tuple(new_pts), best_p, trick_num + 1

    def _resolve_trick(self, hands, pts, active, card, leader, trick_num, existing_tc):
        """Legacy resolve for partial trick completion."""
        new_hands_list = list(hands)
        new_hands_list[active] = tuple(c for c in hands[active] if c != card)
        new_tc = existing_tc + ((active, card),)
        return self._resolve_trick_from_tc(tuple(new_hands_list), pts, leader, trick_num, new_tc)

    def _legal_fast(self, hand: tuple[int, ...], lead_suit: int) -> list[int]:
        """Fast legal moves — just follow suit for endgame (simplified)."""
        if lead_suit < 0:
            return list(hand)
        follow = [c for c in hand if _SUIT[c] == lead_suit]
        return follow if follow else list(hand)


def solve_endgame_for_pimc(
    my_hand: list[int],
    opp_hands: dict[int, list[int]],
    trump: int,
    leader: int,
    my_target: int,
    my_points: int,
    opp_targets: dict[int, int],
    opp_points: dict[int, int],
    trick_number: int,
) -> int:
    """Convenience: solve endgame, return player 0's final points."""
    solver = EndgameSolver(max_depth=3)
    _, final_pts = solver.solve(
        my_hand, opp_hands, trump, leader, my_target, my_points,
        opp_targets, opp_points, trick_number,
    )
    return final_pts

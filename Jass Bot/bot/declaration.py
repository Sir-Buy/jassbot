"""Distribution-based declaration model for Differenzler.

Simulates N worlds, collects the score distribution, and picks
the median (Bayes-optimal for minimizing absolute error).
"""

from __future__ import annotations

import random
import statistics

from engine.cards import CARD_SUIT, points
from bot.heuristics import pick_opponent_greedy, pick_sim_target_aware, pick_opponent_target_aware, estimate_hand_score
from bot.world_sampler import sample_worlds_with_declarations


class DeclarationModel:
    """Distribution-based declaration using score distribution from PIMC simulation.

    By default picks the median (percentile=50, offset=0).
    Calibration can shift these for better performance.
    """

    def __init__(
        self,
        use_target_aware_opponents: bool = False,
        percentile: int = 50,
        offset: int = 0,
    ):
        self.use_target_aware_opponents = use_target_aware_opponents
        self._percentile = percentile
        self._offset = offset

    def declare(
        self,
        hand: list[int],
        trump: int,
        voids: dict[int, set[int]] | None = None,
        num_worlds: int = 200,
        rng: random.Random | None = None,
    ) -> int:
        """Returns the optimal declaration (median of simulated score distribution)."""
        info = self.declare_with_info(hand, trump, voids, num_worlds, rng)
        return info["declaration"]

    def declare_with_info(
        self,
        hand: list[int],
        trump: int,
        voids: dict[int, set[int]] | None = None,
        num_worlds: int = 200,
        rng: random.Random | None = None,
    ) -> dict:
        """Returns declaration plus diagnostic info."""
        if not hand:
            return {
                "declaration": 0, "median": 0, "mean": 0.0,
                "std": 0.0, "min": 0, "max": 0, "distribution": [],
            }

        v = voids or {1: set(), 2: set(), 3: set()}
        unknown = [c for c in range(36) if c not in set(hand)]
        worlds = sample_worlds_with_declarations(
            unknown, len(hand), v, num_worlds, trump, rng
        )
        if not worlds:
            return {
                "declaration": 0, "median": 0, "mean": 0.0,
                "std": 0.0, "min": 0, "max": 0, "distribution": [],
            }

        # Try C engine for massive speedup
        try:
            from engine_c.fast_engine import FastEngine
            _c = FastEngine()
            worlds_hands = [wh for wh, _ in worlds]
            opp_tgts = [wd for _, wd in worlds] if self.use_target_aware_opponents else None
            est = estimate_hand_score(hand, trump)
            scores = _c.simulate_for_distribution(hand, worlds_hands, trump, est, opp_tgts)
        except (ImportError, OSError):
            scores = []
            for world_hands, world_decls in worlds:
                score = self._sim_game(hand, trump, world_hands, world_decls)
                scores.append(score)

        scores.sort()
        median = scores[len(scores) // 2]
        # Use calibrated percentile + offset
        pct_idx = max(0, min(len(scores) - 1, int(len(scores) * self._percentile / 100)))
        declaration = max(0, min(157, scores[pct_idx] + self._offset))
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0

        return {
            "declaration": declaration,
            "median": median,
            "mean": mean,
            "std": std,
            "min": scores[0],
            "max": scores[-1],
            "distribution": scores,
        }

    def _sim_game(
        self,
        hand: list[int],
        trump: int,
        world_hands: dict[int, list[int]],
        world_decls: dict[int, int],
    ) -> int:
        """Simulate full game with heuristic play. Returns player 0's final points.

        Uses a moderate target guess for player 0 (midpoint of hand estimate)
        so the self-play rollout is somewhat target-aware.
        """
        my_hand = list(hand)
        opp = {p: list(cards) for p, cards in world_hands.items()}
        opp_pts = {1: 0, 2: 0, 3: 0}
        my_pts = 0
        leader = 0
        # Use a rough estimate as our own "target" during simulation
        my_est_target = estimate_hand_score(hand, trump)

        for trick_num in range(len(hand)):
            if not my_hand:
                break
            trick: list[tuple[int, int]] = []
            lead_suit: int | None = None

            for i in range(4):
                p = (leader + i) % 4
                if p == 0:
                    card = pick_sim_target_aware(
                        my_hand, trick, lead_suit, trump, my_est_target, my_pts
                    )
                    trick.append((0, card))
                    my_hand.remove(card)
                elif opp[p]:
                    if self.use_target_aware_opponents:
                        card = pick_opponent_target_aware(
                            opp[p], trick, lead_suit, trump,
                            world_decls[p], opp_pts[p], trick_num,
                        )
                    else:
                        card = pick_opponent_greedy(opp[p], trick, lead_suit, trump)
                    trick.append((p, card))
                    opp[p].remove(card)

                if lead_suit is None and trick:
                    lead_suit = CARD_SUIT[trick[0][1]]

            winner = _trick_winner_fast(trick, lead_suit, trump)
            pts = sum(points(c, trump) for _, c in trick)
            if winner == 0:
                my_pts += pts
            elif winner in opp_pts:
                opp_pts[winner] += pts
            leader = winner

        return my_pts


def _trick_winner_fast(
    trick: list[tuple[int, int]], lead_suit: int | None, trump: int,
) -> int:
    """Fast trick winner without importing strategy module."""
    from engine.cards import strength as card_strength
    if not trick:
        return 0
    best_p, best_c = trick[0]
    for p, c in trick[1:]:
        c_suit = CARD_SUIT[c]
        b_suit = CARD_SUIT[best_c]
        if c_suit == trump and b_suit != trump:
            best_p, best_c = p, c
        elif c_suit == b_suit and card_strength(c, trump) > card_strength(best_c, trump):
            best_p, best_c = p, c
    return best_p

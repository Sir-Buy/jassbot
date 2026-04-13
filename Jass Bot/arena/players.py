"""Player implementations for arena play.

All players implement the Player protocol from engine.game.
"""

from __future__ import annotations

import random

from engine.cards import CARD_SUIT, points, strength
from engine.game_state import GameState
from engine.rules import legal_moves
from bot.heuristics import pick_opponent_greedy, pick_sim_target_aware
from bot.strategy import StrategyEngine
from bot.declaration import DeclarationModel
from bot.belief_tracker import BeliefTracker
from bot.declaration_net import NeuralDeclarationModel


class RandomPlayer:
    """Plays random legal moves, declares random target."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def declare(self, hand: list[int], trump: int) -> int:
        return self.rng.randint(0, 157)

    def play(self, state: GameState, legal: list[int]) -> int:
        return self.rng.choice(legal)

    def notify_trick(self, state: GameState, trick: list[tuple[int, int]], winner: int, pts: int) -> None:
        pass


class HeuristicPlayer:
    """Simple rule-based player: plays highest when behind target, lowest when ahead."""

    def __init__(self, player_id: int = 0, rng: random.Random | None = None):
        self.player_id = player_id
        self.rng = rng or random.Random()
        self._target = 0

    def declare(self, hand: list[int], trump: int) -> int:
        total = sum(points(c, trump) for c in hand)
        trump_cards = [c for c in hand if CARD_SUIT[c] == trump]
        trump_bonus = len(trump_cards) * 5
        self._target = min(157, max(0, total // 2 + trump_bonus))
        return self._target

    def play(self, state: GameState, legal: list[int]) -> int:
        trump = state.trump
        my_pts = state.points[self.player_id]
        diff = self._target - my_pts

        if diff <= 0:
            return min(legal, key=lambda c: (points(c, trump), strength(c, trump)))
        return max(legal, key=lambda c: (points(c, trump), strength(c, trump)))

    def notify_trick(self, state: GameState, trick: list[tuple[int, int]], winner: int, pts: int) -> None:
        pass


class PIMCPlayer:
    """Full PIMC strategy engine player.

    Args:
        declaration: "sweep" (original) or "median" (distribution-based)
        use_target_aware_opponents: target-aware opponent rollout
        use_belief_tracking: Bayesian opponent declaration inference
        endgame_depth: endgame solver activation depth (0 = off)
    """

    def __init__(
        self,
        player_id: int = 0,
        num_worlds: int = 25,
        ansage_worlds: int = 40,
        declaration: str = "sweep",
        use_target_aware_opponents: bool = False,
        use_belief_tracking: bool = False,
        endgame_depth: int = 0,
        rng: random.Random | None = None,
    ):
        self.player_id = player_id
        self.rng = rng or random.Random()
        self.declaration_mode = declaration
        self.use_target_aware_opponents = use_target_aware_opponents
        self.use_belief_tracking = use_belief_tracking
        self.engine = StrategyEngine(
            num_worlds=num_worlds,
            ansage_worlds=ansage_worlds,
            use_target_aware_opponents=use_target_aware_opponents,
            endgame_depth=endgame_depth,
        )
        self._decl_model: DeclarationModel | None = None
        self._neural_decl: NeuralDeclarationModel | None = None
        if declaration == "neural":
            self._neural_decl = NeuralDeclarationModel()
            if not self._neural_decl.available:
                print("[WARN] Neural declaration model not found, falling back to median")
                self._neural_decl = None
                self._decl_model = DeclarationModel(
                    use_target_aware_opponents=use_target_aware_opponents,
                )
        elif declaration == "median":
            self._decl_model = DeclarationModel(
                use_target_aware_opponents=use_target_aware_opponents,
            )
        self._belief_tracker: BeliefTracker | None = None
        if use_belief_tracking:
            self._belief_tracker = BeliefTracker()
        self._ansage_worlds = ansage_worlds
        self._target = 0
        self._my_points = 0
        self._trick_number = 0
        self._trump = 0

    def declare(self, hand: list[int], trump: int) -> int:
        self.engine.reset()
        self._my_points = 0
        self._trick_number = 0
        self._trump = trump

        if self._belief_tracker is not None:
            self._belief_tracker.init_round(trump)

        if self._neural_decl is not None:
            self._target = self._neural_decl.declare(hand, trump)
        elif self._decl_model is not None:
            self._target = self._decl_model.declare(
                hand, trump,
                voids=self.engine.player_voids,
                num_worlds=self._ansage_worlds,
                rng=self.rng,
            )
        else:
            self._target = self.engine.ansage(hand, trump, self.rng)

        return self._target

    def play(self, state: GameState, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]

        for p in [1, 2, 3]:
            if p != self.player_id:
                for suit in state.voids.get(p, set()):
                    self.engine.record_void(p, suit)

        played = list(state.played_cards)
        table = state.cards_on_table()

        card = self.engine.zug(
            hand=list(state.hands[self.player_id]),
            trump=state.trump,
            target=self._target,
            current_points=state.points[self.player_id],
            table_cards=table,
            leader=state.leader,
            played_cards=played,
            trick_number=self._trick_number,
            rng=self.rng,
            belief_tracker=self._belief_tracker,
            opp_points=list(state.points),
        )
        return card if card is not None else legal[0]

    def notify_trick(self, state: GameState, trick: list[tuple[int, int]], winner: int, pts: int) -> None:
        if winner == self.player_id:
            self._my_points += pts

        # Update beliefs from observed opponent actions
        if self._belief_tracker is not None:
            lead_suit = CARD_SUIT[trick[0][1]] if trick else None
            cards_before: list[tuple[int, int]] = []
            for player, card in trick:
                if player != self.player_id and player in [1, 2, 3]:
                    self._belief_tracker.update(
                        player=player,
                        action=card,
                        player_current_points=state.points[player] - (pts if winner == player else 0),
                        trick_number=self._trick_number,
                        table_cards_before=list(cards_before),
                        lead_suit=lead_suit if cards_before else None,
                        trump=state.trump,
                    )
                cards_before.append((player, card))

        self._trick_number += 1


class ISMCTSPlayer:
    """ISMCTS-based player for Differenzler.

    Uses SO-ISMCTS for card play, DeclarationModel for declaration.
    """

    def __init__(
        self,
        player_id: int = 0,
        time_limit_ms: int = 2000,
        exploration: float = 0.7,
        use_belief_tracking: bool = True,
        endgame_depth: int = 4,
        ansage_worlds: int = 2000,
        declaration_percentile: int = 45,
        declaration_offset: int = -1,
        rng: random.Random | None = None,
        verbose: bool = False,
    ):
        self.player_id = player_id
        self.rng = rng or random.Random()
        self.verbose = verbose

        from bot.ismcts import ISMCTSEngine
        self._ismcts = ISMCTSEngine(
            time_limit_ms=time_limit_ms,
            exploration=exploration,
            endgame_depth=endgame_depth,
        )
        self._decl_model = DeclarationModel(
            use_target_aware_opponents=True,
            percentile=declaration_percentile,
            offset=declaration_offset,
        )
        self._belief_tracker = None
        if use_belief_tracking:
            from bot.belief_tracker import BeliefTracker
            self._belief_tracker = BeliefTracker()

        self._ansage_worlds = ansage_worlds
        self._target = 0
        self._my_points = 0
        self._trick_number = 0
        self._trump = 0
        self._voids: dict[int, set[int]] = {1: set(), 2: set(), 3: set()}

    def declare(self, hand: list[int], trump: int) -> int:
        self._my_points = 0
        self._trick_number = 0
        self._trump = trump
        self._voids = {1: set(), 2: set(), 3: set()}

        if self._belief_tracker is not None:
            self._belief_tracker.init_round(trump)

        self._target = self._decl_model.declare(
            hand, trump, voids=self._voids,
            num_worlds=self._ansage_worlds, rng=self.rng,
        )
        return self._target

    def play(self, state: GameState, legal: list[int]) -> int:
        if len(legal) == 1:
            return legal[0]

        for p in [1, 2, 3]:
            for suit in state.voids.get(p, set()):
                self._voids.setdefault(p, set()).add(suit)

        played = list(state.played_cards)
        table = state.cards_on_table()

        card, stats = self._ismcts.search(
            my_hand=list(state.hands[self.player_id]),
            trump=state.trump,
            target=self._target,
            my_points=state.points[self.player_id],
            opp_points=list(state.points),
            table_cards=table,
            leader=state.leader,
            trick_number=self._trick_number,
            played_cards=played,
            voids=self._voids,
            rng=self.rng,
            belief_tracker=self._belief_tracker,
        )

        if self.verbose and stats:
            top = sorted(stats.items(), key=lambda x: -x[1][0])[:3]
            from engine.cards import card_short
            info = ", ".join(f"{card_short(c)}:{v}v/{u:.1f}u" for c, (v, u) in top)
            print(f"  ISMCTS: {self._ismcts.last_iterations} iter "
                  f"({self._ismcts.last_iter_per_sec:.0f}/s) -> {info}")

        return card if card in legal else legal[0]

    def notify_trick(self, state: GameState, trick: list[tuple[int, int]], winner: int, pts: int) -> None:
        if winner == self.player_id:
            self._my_points += pts

        if self._belief_tracker is not None:
            lead_suit = CARD_SUIT[trick[0][1]] if trick else None
            cards_before: list[tuple[int, int]] = []
            for player, card in trick:
                if player != self.player_id and player in [1, 2, 3]:
                    self._belief_tracker.update(
                        player=player,
                        action=card,
                        player_current_points=state.points[player] - (pts if winner == player else 0),
                        trick_number=self._trick_number,
                        table_cards_before=list(cards_before),
                        lead_suit=lead_suit if cards_before else None,
                        trump=state.trump,
                    )
                cards_before.append((player, card))

        self._trick_number += 1

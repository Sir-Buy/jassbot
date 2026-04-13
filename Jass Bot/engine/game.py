"""Game runner for Differenzler rounds.

Manages trick flow, enforces rules, tracks state.
Players implement the Player protocol.
"""

from __future__ import annotations

import random
from typing import Protocol

from engine.cards import (
    ALL_CARD_IDS,
    CARD_SUIT,
    LAST_TRICK_BONUS,
    NUM_SUITS,
    NUM_VALUES,
    TOTAL_ROUND_POINTS,
    points,
)
from engine.game_state import GameState
from engine.rules import legal_moves, trick_points, trick_winner


class Player(Protocol):
    """Interface that all players must implement."""

    def declare(self, hand: list[int], trump: int) -> int:
        """Predict how many points you'll score. Returns 0-157."""
        ...

    def play(self, state: GameState, legal: list[int]) -> int:
        """Choose a card to play from the legal moves."""
        ...

    def notify_trick(self, state: GameState, trick: list[tuple[int, int]], winner: int, pts: int) -> None:
        """Called after each trick completes (optional, for tracking)."""
        ...


def deal(rng: random.Random | None = None) -> tuple[int, dict[int, list[int]]]:
    """Deal 9 cards to each of 4 players, choose random trump.

    Returns:
        (trump_suit, {player_id: [card_ids]})
    """
    r = rng or random.Random()
    deck = list(ALL_CARD_IDS)
    r.shuffle(deck)
    hands = {
        0: sorted(deck[0:9]),
        1: sorted(deck[9:18]),
        2: sorted(deck[18:27]),
        3: sorted(deck[27:36]),
    }
    trump = r.randint(0, NUM_SUITS - 1)
    return trump, hands


def play_round(
    players: dict[int, Player],
    trump: int | None = None,
    hands: dict[int, list[int]] | None = None,
    rng: random.Random | None = None,
) -> RoundResult:
    """Play a complete 9-trick Differenzler round.

    Args:
        players: {player_id: Player} for all 4 players
        trump: trump suit index (random if None)
        hands: pre-dealt hands (dealt randomly if None)
        rng: random source

    Returns:
        RoundResult with final scores and deviations
    """
    r = rng or random.Random()

    if trump is None or hands is None:
        t, h = deal(r)
        if trump is None:
            trump = t
        if hands is None:
            hands = h

    state = GameState(
        hands={p: list(h) for p, h in hands.items()},
        trump=trump,
        trick_number=1,
        leader=0,
    )

    # Declaration phase
    for pid, player in players.items():
        decl = player.declare(list(state.hands[pid]), trump)
        state.declarations[pid] = max(0, min(157, decl))

    # Play 9 tricks
    for trick_num in range(1, 10):
        state.trick_number = trick_num
        state.current_trick = []
        is_last = trick_num == 9

        for i in range(4):
            pid = (state.leader + i) % 4
            hand = state.hands[pid]
            legal = legal_moves(
                hand,
                state.lead_suit,
                trump,
                state.current_trick if state.current_trick else None,
            )
            if not legal:
                legal = list(hand)

            card = players[pid].play(state.copy(), legal)

            # Validate
            if card not in legal:
                card = legal[0]

            # Auto-detect voids
            if state.current_trick:
                lead_s = CARD_SUIT[state.current_trick[0][1]]
                if CARD_SUIT[card] != lead_s:
                    state.voids[pid].add(lead_s)

            state.current_trick.append((pid, card))
            hand.remove(card)

        # Resolve trick
        winner = trick_winner(state.current_trick, state.lead_suit, trump)
        pts = trick_points(state.current_trick, trump)
        if is_last:
            pts += LAST_TRICK_BONUS

        state.points[winner] += pts

        # Notify players
        for pid, player in players.items():
            try:
                player.notify_trick(state, list(state.current_trick), winner, pts)
            except (AttributeError, TypeError):
                pass

        # Move played cards to history
        for _, cid in state.current_trick:
            state.played_cards.add(cid)

        state.leader = winner

    # Build result
    return RoundResult(
        trump=trump,
        declarations={p: state.declarations[p] for p in range(4)},
        points={p: state.points[p] for p in range(4)},
    )


class RoundResult:
    """Result of a single Differenzler round."""

    def __init__(
        self,
        trump: int,
        declarations: dict[int, int],
        points: dict[int, int],
    ):
        self.trump = trump
        self.declarations = declarations
        self.points = points
        self.deviations = {
            p: abs(declarations[p] - points[p]) for p in range(4)
        }

    @property
    def total_points(self) -> int:
        return sum(self.points.values())

    def __repr__(self) -> str:
        lines = []
        for p in range(4):
            lines.append(
                f"P{p}: declared={self.declarations[p]:3d}  "
                f"actual={self.points[p]:3d}  "
                f"dev={self.deviations[p]:3d}"
            )
        lines.append(f"Total points: {self.total_points} (should be 157)")
        return "\n".join(lines)

"""Immutable-style game state for simulation and search.

All card references are integer IDs (0-35).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from engine.cards import CARD_SUIT, NUM_SUITS


@dataclass
class GameState:
    """Snapshot of a Differenzler round, cheaply copyable for simulation."""

    # Per-player hands: player_id → list of card IDs
    hands: dict[int, list[int]] = field(default_factory=lambda: {0: [], 1: [], 2: [], 3: []})

    trump: int = 0                    # Trump suit index (0-3)
    trick_number: int = 1             # Current trick (1-9)
    leader: int = 0                   # Who leads the current trick

    # Current trick: list of (player_id, card_id) in play order
    current_trick: list[tuple[int, int]] = field(default_factory=list)

    # Points accumulated per player
    points: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    # All cards played so far (not including current trick)
    played_cards: set[int] = field(default_factory=set)

    # Known voids: player_id → set of suit indices they can't have
    voids: dict[int, set[int]] = field(
        default_factory=lambda: {0: set(), 1: set(), 2: set(), 3: set()}
    )

    # Declarations: player_id → declared target (-1 if unknown)
    declarations: dict[int, int] = field(
        default_factory=lambda: {0: -1, 1: -1, 2: -1, 3: -1}
    )

    def copy(self) -> GameState:
        """Deep copy for simulation branching."""
        return GameState(
            hands={p: list(h) for p, h in self.hands.items()},
            trump=self.trump,
            trick_number=self.trick_number,
            leader=self.leader,
            current_trick=list(self.current_trick),
            points=list(self.points),
            played_cards=set(self.played_cards),
            voids={p: set(v) for p, v in self.voids.items()},
            declarations=dict(self.declarations),
        )

    @property
    def lead_suit(self) -> int | None:
        """Suit of the first card in the current trick, or None."""
        if self.current_trick:
            return CARD_SUIT[self.current_trick[0][1]]
        return None

    @property
    def next_player(self) -> int:
        """Who plays next in the current trick."""
        return (self.leader + len(self.current_trick)) % 4

    @property
    def trick_complete(self) -> bool:
        return len(self.current_trick) == 4

    @property
    def round_over(self) -> bool:
        return self.trick_number > 9

    def cards_on_table(self) -> list[tuple[int, int]]:
        """Cards currently on the table as (player, card_id) pairs."""
        return list(self.current_trick)

    def remaining_cards(self, exclude_hands: bool = False) -> set[int]:
        """Cards not yet played and not on the table."""
        on_table = {c for _, c in self.current_trick}
        used = self.played_cards | on_table
        if not exclude_hands:
            for h in self.hands.values():
                used |= set(h)
        return set(range(36)) - used

    def detect_void(self, player: int, card: int) -> None:
        """If player didn't follow suit, record the void."""
        if not self.current_trick:
            return
        lead = CARD_SUIT[self.current_trick[0][1]]
        if CARD_SUIT[card] != lead:
            self.voids[player].add(lead)

"""Bayesian belief tracking over opponent declarations.

Maintains a probability distribution over each opponent's likely
declaration (0-157) and updates it after every observed action.
"""

from __future__ import annotations

import math
import random

from engine.cards import CARD_SUIT, CARD_STRENGTH, CARD_POINTS, points, strength

# --- Prior distribution ---
# Empirically, declarations cluster around 157/4 ≈ 39 with std ~25.
_PRIOR_MEAN = 39.0
_PRIOR_STD = 25.0


def _build_prior() -> list[float]:
    """Gaussian prior over [0, 157], normalized."""
    raw = []
    for d in range(158):
        z = (d - _PRIOR_MEAN) / _PRIOR_STD
        raw.append(math.exp(-0.5 * z * z))
    total = sum(raw)
    return [p / total for p in raw]


_DEFAULT_PRIOR = _build_prior()


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


class BeliefTracker:
    """Tracks posterior beliefs over opponent declarations."""

    def __init__(self):
        self.beliefs: dict[int, list[float]] = {}

    def init_round(self, trump: int) -> None:
        """Initialize shaped priors at start of round."""
        for p in [1, 2, 3]:
            self.beliefs[p] = list(_DEFAULT_PRIOR)

    def update(
        self,
        player: int,
        action: int,
        player_current_points: int,
        trick_number: int,
        table_cards_before: list[tuple[int, int]],
        lead_suit: int | None,
        trump: int,
    ) -> None:
        """Update beliefs after observing a player's card choice.

        Args:
            player: opponent player id (1, 2, 3)
            action: card_id they played
            player_current_points: their points before this trick resolves
            trick_number: 0-indexed trick number
            table_cards_before: cards on table before this player acted
            lead_suit: suit of the led card, or None if this player led
            trump: trump suit index
        """
        if player not in self.beliefs:
            return

        signal = _classify_action(
            action, table_cards_before, lead_suit, trump
        )

        belief = self.beliefs[player]
        for d in range(158):
            gap = d - player_current_points
            likelihood = _action_likelihood(signal, gap, trick_number)
            belief[d] *= likelihood

        # Renormalize
        total = sum(belief)
        if total > 1e-30:
            self.beliefs[player] = [p / total for p in belief]
        else:
            # Degenerate — reset to prior
            self.beliefs[player] = list(_DEFAULT_PRIOR)

    def sample_declaration(self, player: int, rng: random.Random | None = None) -> int:
        """Sample a declaration from the posterior distribution."""
        r = rng or random.Random()
        if player not in self.beliefs:
            return int(_PRIOR_MEAN)
        belief = self.beliefs[player]
        # Weighted random choice
        roll = r.random()
        cumulative = 0.0
        for d in range(158):
            cumulative += belief[d]
            if cumulative >= roll:
                return d
        return 157

    def set_known_declaration(self, player: int, declaration: int) -> None:
        """Set a known declaration value — collapses belief to a spike.

        Called when we intercept the actual opponent declaration from the
        game protocol. This is the strongest possible belief update.
        """
        if player not in self.beliefs:
            return
        belief = [0.0] * 158
        # Tight gaussian around known value (allow ±1 for rounding)
        declaration = max(0, min(157, declaration))
        for d in range(max(0, declaration - 1), min(158, declaration + 2)):
            belief[d] = 1.0
        total = sum(belief)
        self.beliefs[player] = [p / total for p in belief]

    def get_expected_declaration(self, player: int) -> float:
        """Return expected value of declaration distribution."""
        if player not in self.beliefs:
            return _PRIOR_MEAN
        return sum(d * p for d, p in enumerate(self.beliefs[player]))

    def get_declaration_range(
        self, player: int, confidence: float = 0.8
    ) -> tuple[int, int]:
        """Return (low, high) range containing `confidence` probability mass."""
        if player not in self.beliefs:
            return (0, 157)
        belief = self.beliefs[player]
        # Find shortest interval containing `confidence` mass
        best_low, best_high = 0, 157
        best_width = 158

        cumsum = [0.0] * 159
        for i in range(158):
            cumsum[i + 1] = cumsum[i] + belief[i]

        for low in range(158):
            # Binary search for high
            target_mass = cumsum[low] + confidence
            for high in range(low, 158):
                if cumsum[high + 1] >= target_mass - 1e-10:
                    width = high - low
                    if width < best_width:
                        best_width = width
                        best_low, best_high = low, high
                    break

        return (best_low, best_high)

    def get_std(self, player: int) -> float:
        """Standard deviation of the belief distribution."""
        if player not in self.beliefs:
            return _PRIOR_STD
        mean = self.get_expected_declaration(player)
        var = sum((d - mean) ** 2 * p for d, p in enumerate(self.beliefs[player]))
        return math.sqrt(var)


# --- Action classification ---

_SIGNAL_SHED = -1
_SIGNAL_NEUTRAL = 0
_SIGNAL_GRAB = 1


def _classify_action(
    action: int,
    table_cards_before: list[tuple[int, int]],
    lead_suit: int | None,
    trump: int,
) -> int:
    """Classify an opponent's action as shedding, grabbing, or neutral.

    Returns _SIGNAL_SHED, _SIGNAL_NEUTRAL, or _SIGNAL_GRAB.
    """
    action_suit = CARD_SUIT[action]
    action_pts = points(action, trump)
    action_str = strength(action, trump)
    is_trump_play = action_suit == trump

    if not table_cards_before:
        # Player is leading
        if action_pts >= 10 or is_trump_play:
            return _SIGNAL_GRAB  # Leading high or leading trump = aggressive
        if action_pts == 0 and not is_trump_play:
            return _SIGNAL_SHED  # Leading a 0-point non-trump = passive
        return _SIGNAL_NEUTRAL

    # Player is following or playing off-suit
    # Check if this card beats the current table
    beats_all = _would_beat_table(action, table_cards_before, lead_suit, trump)
    table_pts = sum(points(c, trump) for _, c in table_cards_before)

    if beats_all and table_pts >= 10:
        # Winning a valuable trick
        return _SIGNAL_GRAB

    if not beats_all and action_pts <= 3:
        # Losing with a low card
        return _SIGNAL_SHED

    # Trumping in when can't follow
    if lead_suit is not None and action_suit != lead_suit and is_trump_play:
        return _SIGNAL_GRAB

    # Dumping a 0-pointer off-suit
    if lead_suit is not None and action_suit != lead_suit and not is_trump_play and action_pts == 0:
        return _SIGNAL_SHED

    return _SIGNAL_NEUTRAL


def _would_beat_table(
    card: int,
    table_cards: list[tuple[int, int]],
    lead_suit: int | None,
    trump: int,
) -> bool:
    """Would this card beat everything on the table?"""
    card_suit = CARD_SUIT[card]
    card_str = strength(card, trump)
    for _, tc in table_cards:
        tc_suit = CARD_SUIT[tc]
        if tc_suit == trump and card_suit != trump:
            return False
        if tc_suit == card_suit and strength(tc, trump) >= card_str:
            return False
        if lead_suit is not None and card_suit != lead_suit and card_suit != trump:
            return False
    return True


def _action_likelihood(signal: int, gap: float, trick_number: int) -> float:
    """Compute likelihood of this signal given a particular gap.

    gap = declaration - current_points
    Positive gap = behind target, negative = over target.
    """
    # Scale factor decreases with trick number (later actions are stronger signals)
    scale = max(5.0, 15.0 - trick_number * 1.5)

    if signal == _SIGNAL_SHED:
        # Shedding is likely when gap <= 0 (over target)
        return 0.1 + 0.9 * _sigmoid(-gap / scale)
    elif signal == _SIGNAL_GRAB:
        # Grabbing is likely when gap > 0 (behind target)
        return 0.1 + 0.9 * _sigmoid(gap / scale)
    else:
        # Neutral: weak update
        return 1.0

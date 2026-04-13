"""Constrained world generation for PIMC search.

Generates plausible opponent hand assignments consistent with
known constraints (voids, played cards, own hand).
"""

from __future__ import annotations

import random

from engine.cards import CARD_SUIT


def sample_worlds(
    unknown_cards: list[int],
    cards_per_player: int,
    voids: dict[int, set[int]],
    num_worlds: int,
    rng: random.Random | None = None,
) -> list[dict[int, list[int]]]:
    """Generate multiple plausible worlds (opponent hand assignments).

    Args:
        unknown_cards: card IDs not in our hand, not played, not on table
        cards_per_player: how many cards each opponent should have
        voids: {player_id: set of suit indices they're void in}
        num_worlds: number of worlds to generate
        rng: random source

    Returns:
        List of {player_id: [card_ids]} for players 1, 2, 3
    """
    r = rng or random.Random()
    worlds = []
    for _ in range(num_worlds):
        w = _generate_one_world(unknown_cards, cards_per_player, voids, r)
        if w is not None:
            worlds.append(w)
    return worlds


def sample_worlds_with_declarations(
    unknown_cards: list[int],
    cards_per_player: int,
    voids: dict[int, set[int]],
    num_worlds: int,
    trump: int,
    rng: random.Random | None = None,
    belief_tracker: object | None = None,
) -> list[tuple[dict[int, list[int]], dict[int, int]]]:
    """Generate worlds with plausible opponent declarations.

    If a belief_tracker is provided, sample declarations from the posterior.
    Otherwise, estimate from the dealt hand.

    Returns:
        List of (hands_dict, declarations_dict) tuples.
    """
    from bot.heuristics import estimate_hand_score

    r = rng or random.Random()
    worlds = []
    for _ in range(num_worlds):
        w = _generate_one_world(unknown_cards, cards_per_player, voids, r)
        if w is not None:
            decls = {}
            for p in [1, 2, 3]:
                if belief_tracker is not None:
                    decls[p] = belief_tracker.sample_declaration(p, r)
                else:
                    decls[p] = estimate_hand_score(w[p], trump)
            worlds.append((w, decls))
    return worlds


def _generate_one_world(
    unknown_cards: list[int],
    cards_per_player: int,
    voids: dict[int, set[int]],
    rng: random.Random,
) -> dict[int, list[int]] | None:
    """Generate a single valid world using greedy assignment with fallback."""
    cards = list(unknown_cards)
    rng.shuffle(cards)

    world: dict[int, list[int]] = {1: [], 2: [], 3: []}
    rest: list[int] = []

    for card in cards:
        suit = CARD_SUIT[card]
        placed = False
        players = [1, 2, 3]
        rng.shuffle(players)
        for p in players:
            if len(world[p]) < cards_per_player and suit not in voids.get(p, set()):
                world[p].append(card)
                placed = True
                break
        if not placed:
            rest.append(card)

    for card in rest:
        for p in [1, 2, 3]:
            if len(world[p]) < cards_per_player:
                world[p].append(card)
                break

    return world

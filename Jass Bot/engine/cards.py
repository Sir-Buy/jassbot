"""Card representation and precomputed lookup tables for Swiss Jass (36-card deck).

Cards are identified by integer IDs 0-35 for speed.
Card objects are available for display/debugging.

Layout: card_id = suit_index * 9 + value_index
  Suits: herz=0, ecke=1, schaufel=2, kreuz=3
  Values: 6=0, 7=1, 8=2, 9=3, 10=4, U=5, O=6, K=7, A=8
"""

from __future__ import annotations

from typing import Final

# --- Suit and value constants ---

SUITS: Final[list[str]] = ["herz", "ecke", "schaufel", "kreuz"]
VALUES: Final[list[str]] = ["6", "7", "8", "9", "10", "U", "O", "K", "A"]

SUIT_INDEX: Final[dict[str, int]] = {s: i for i, s in enumerate(SUITS)}
VALUE_INDEX: Final[dict[str, int]] = {v: i for i, v in enumerate(VALUES)}

NUM_SUITS: Final[int] = 4
NUM_VALUES: Final[int] = 9
NUM_CARDS: Final[int] = 36

SUIT_SYMBOLS: Final[dict[str, str]] = {
    "herz": "♥", "ecke": "♦", "schaufel": "♠", "kreuz": "♣",
}
SUIT_NAMES: Final[dict[str, str]] = {
    "herz": "Herz", "ecke": "Ecke", "schaufel": "Schaufel", "kreuz": "Kreuz",
}
VALUE_NAMES: Final[dict[str, str]] = {
    "6": "6", "7": "7", "8": "8", "9": "9", "10": "10",
    "U": "Under", "O": "Ober", "K": "König", "A": "Ass",
}


# --- Card ID helpers ---

def card_id(suit: str, value: str) -> int:
    return SUIT_INDEX[suit] * NUM_VALUES + VALUE_INDEX[value]


def card_id_from_ints(suit_idx: int, value_idx: int) -> int:
    return suit_idx * NUM_VALUES + value_idx


# --- Precomputed lookup tables (indexed by card_id 0-35) ---

CARD_SUIT: Final[list[int]] = [cid // NUM_VALUES for cid in range(NUM_CARDS)]
CARD_VALUE: Final[list[int]] = [cid % NUM_VALUES for cid in range(NUM_CARDS)]
CARD_SUIT_NAME: Final[list[str]] = [SUITS[cid // NUM_VALUES] for cid in range(NUM_CARDS)]
CARD_VALUE_NAME: Final[list[str]] = [VALUES[cid % NUM_VALUES] for cid in range(NUM_CARDS)]

# Points: CARD_POINTS[card_id][is_trump]  (0=non-trump, 1=trump)
_POINTS_NORMAL: Final[dict[str, int]] = {
    "6": 0, "7": 0, "8": 0, "9": 0, "10": 10, "U": 2, "O": 3, "K": 4, "A": 11,
}
_POINTS_TRUMP: Final[dict[str, int]] = {
    "6": 0, "7": 0, "8": 0, "9": 14, "10": 10, "U": 20, "O": 3, "K": 4, "A": 11,
}

CARD_POINTS: Final[list[tuple[int, int]]] = []
for _cid in range(NUM_CARDS):
    _val = VALUES[_cid % NUM_VALUES]
    CARD_POINTS.append((_POINTS_NORMAL[_val], _POINTS_TRUMP[_val]))

# Strength: CARD_STRENGTH[card_id][is_trump]  (0=non-trump, 1=trump)
# Non-trump: 6<7<8<9<10<U<O<K<A  → 0..8
# Trump:     6<7<8<10<O<K<A<9<U  → 0,1,2,3,4,5,6,7,8
_STRENGTH_NORMAL: Final[dict[str, int]] = {
    "6": 0, "7": 1, "8": 2, "9": 3, "10": 4, "U": 5, "O": 6, "K": 7, "A": 8,
}
_STRENGTH_TRUMP: Final[dict[str, int]] = {
    "6": 0, "7": 1, "8": 2, "10": 3, "O": 4, "K": 5, "A": 6, "9": 7, "U": 8,
}

CARD_STRENGTH: Final[list[tuple[int, int]]] = []
for _cid in range(NUM_CARDS):
    _val = VALUES[_cid % NUM_VALUES]
    CARD_STRENGTH.append((_STRENGTH_NORMAL[_val], _STRENGTH_TRUMP[_val]))


def points(card: int, trump_suit: int) -> int:
    """Points for a card given the trump suit index."""
    return CARD_POINTS[card][1 if CARD_SUIT[card] == trump_suit else 0]


def strength(card: int, trump_suit: int) -> int:
    """Strength for a card given the trump suit index. Trump cards get +100."""
    is_trump = CARD_SUIT[card] == trump_suit
    base = CARD_STRENGTH[card][1 if is_trump else 0]
    return base + 100 if is_trump else base


# --- Card display helpers ---

def card_name(card: int) -> str:
    """Short display name like ♥Under."""
    return f"{SUIT_SYMBOLS[CARD_SUIT_NAME[card]]}{VALUE_NAMES[CARD_VALUE_NAME[card]]}"


def card_full_name(card: int) -> str:
    """Full display name like Herz Under."""
    return f"{SUIT_NAMES[CARD_SUIT_NAME[card]]} {VALUE_NAMES[CARD_VALUE_NAME[card]]}"


def card_short(card: int) -> str:
    """Compact name like H6, EU, SA."""
    suit_letter = CARD_SUIT_NAME[card][0].upper()
    return f"{suit_letter}{CARD_VALUE_NAME[card]}"


# --- Full deck ---

ALL_CARD_IDS: Final[list[int]] = list(range(NUM_CARDS))


# --- Named card IDs for convenience ---

def _make_named() -> dict[str, int]:
    """Generate constants like HERZ_6, ECKE_U, etc."""
    named = {}
    for s in SUITS:
        for v in VALUES:
            named[f"{s.upper()}_{v}"] = card_id(s, v)
    return named


NAMED_CARDS: Final[dict[str, int]] = _make_named()

# Expose commonly used trump cards
HERZ_U: Final[int] = card_id("herz", "U")
HERZ_9: Final[int] = card_id("herz", "9")
ECKE_U: Final[int] = card_id("ecke", "U")
ECKE_9: Final[int] = card_id("ecke", "9")
SCHAUFEL_U: Final[int] = card_id("schaufel", "U")
SCHAUFEL_9: Final[int] = card_id("schaufel", "9")
KREUZ_U: Final[int] = card_id("kreuz", "U")
KREUZ_9: Final[int] = card_id("kreuz", "9")

# Last trick bonus
LAST_TRICK_BONUS: Final[int] = 5
TOTAL_ROUND_POINTS: Final[int] = 157

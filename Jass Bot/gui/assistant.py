"""Pygame GUI assistant for Jass Differenzler.

Refactored from BOT_working.py to use the new engine modules.
The GUI still uses string-based Card objects for display, with a bridge
to the integer-based engine for strategy calculations.
"""

from __future__ import annotations

import pygame

from engine.cards import (
    SUITS, VALUES, SUIT_SYMBOLS, SUIT_NAMES, VALUE_NAMES,
    SUIT_INDEX, VALUE_INDEX, NUM_VALUES,
    card_id, card_name, points as card_points, strength as card_strength,
    CARD_SUIT, CARD_POINTS, CARD_STRENGTH,
)
from engine.rules import legal_moves, trick_winner as engine_trick_winner
from bot.strategy import StrategyEngine
from bot.declaration import DeclarationModel

# --- Colors ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
DARK_GREEN = (20, 80, 20)
RED = (220, 20, 60)
GOLD = (255, 215, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (180, 180, 180)
BLUE = (70, 130, 180)
HIGHLIGHT = (0, 255, 100)

SUIT_COLORS = {"herz": RED, "ecke": RED, "schaufel": BLACK, "kreuz": BLACK}


class Card:
    """Display-oriented card object bridging to integer IDs."""

    def __init__(self, suit: str, value: str):
        self.suit = suit
        self.value = value
        self.id = card_id(suit, value)

    def points(self, trump: str | None = None) -> int:
        if trump is None:
            return CARD_POINTS[self.id][0]
        return card_points(self.id, SUIT_INDEX[trump])

    def strength(self, trump: str | None = None) -> int:
        if trump is None:
            return CARD_STRENGTH[self.id][0]
        return card_strength(self.id, SUIT_INDEX[trump])

    def name(self) -> str:
        return f"{SUIT_SYMBOLS[self.suit]}{VALUE_NAMES[self.value]}"

    def full_name(self) -> str:
        return f"{SUIT_NAMES[self.suit]} {VALUE_NAMES[self.value]}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Card) and self.suit == other.suit and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.suit, self.value))


# Lookup: card_id → Card object
_ID_TO_CARD: dict[int, Card] = {}
for _s in SUITS:
    for _v in VALUES:
        _c = Card(_s, _v)
        _ID_TO_CARD[_c.id] = _c


class GameLogic:
    """Game state tracker for the GUI assistant."""

    PLAYERS = ["ICH", "RECHTS", "OBEN", "LINKS"]

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.all_cards = [Card(s, v) for s in SUITS for v in VALUES]
        self.hand: list[Card] = []
        self.trump: str | None = None
        self.ziel: int = 0
        self.punkte: int = 0
        self.stich: int = 1
        self.gespielt: list[Card] = []
        self.tisch: dict[int, Card] = {}
        self.ausspieler: int = 0
        self.spieler_punkte: list[int] = [0, 0, 0, 0]
        self.history: list[dict] = []

    def get_remaining_cards(self) -> list[Card]:
        used = set(self.gespielt + self.hand + list(self.tisch.values()))
        return [c for c in self.all_cards if c not in used]

    def get_tisch_karten(self) -> list[tuple[int, Card]]:
        karten = []
        for i in range(4):
            spieler = (self.ausspieler + i) % 4
            if spieler in self.tisch:
                karten.append((spieler, self.tisch[spieler]))
        return karten

    def get_lead_suit(self) -> str | None:
        if not self.tisch:
            return None
        if self.ausspieler in self.tisch:
            return self.tisch[self.ausspieler].suit
        karten = self.get_tisch_karten()
        return karten[0][1].suit if karten else None

    def get_stich_gewinner(self) -> int | None:
        if not self.tisch:
            return None
        lead_suit = self.get_lead_suit()
        if not lead_suit:
            return None

        gewinner = None
        hoechste = None

        for spieler, karte in self.tisch.items():
            if hoechste is None:
                hoechste = karte
                gewinner = spieler
            else:
                if karte.suit == self.trump and hoechste.suit != self.trump:
                    hoechste = karte
                    gewinner = spieler
                elif karte.suit == hoechste.suit:
                    if karte.strength(self.trump) > hoechste.strength(self.trump):
                        hoechste = karte
                        gewinner = spieler

        return gewinner

    def get_stich_punkte(self) -> int:
        return sum(c.points(self.trump) for c in self.tisch.values())

    def get_playable_cards(self) -> list[Card]:
        if not self.hand:
            return []
        lead_suit = self.get_lead_suit()
        if not lead_suit:
            return self.hand.copy()
        same_suit = [c for c in self.hand if c.suit == lead_suit]
        return same_suit if same_suit else self.hand.copy()

    def karte_spielen(self, spieler: int, karte: Card) -> None:
        self.tisch[spieler] = karte
        if spieler == 0 and karte in self.hand:
            self.hand.remove(karte)

    def stich_beenden(self) -> int | None:
        if len(self.tisch) < 2:
            return None

        gewinner = self.get_stich_gewinner()
        punkte = self.get_stich_punkte()

        if gewinner is not None:
            self.spieler_punkte[gewinner] += punkte
            if gewinner == 0:
                self.punkte += punkte

        self.history.append({
            "stich": self.stich,
            "karten": self.tisch.copy(),
            "gewinner": gewinner,
            "punkte": punkte,
        })

        self.gespielt.extend(self.tisch.values())
        self.tisch = {}
        self.stich += 1

        if gewinner is not None:
            self.ausspieler = gewinner

        return gewinner

    def get_max_hand_size(self) -> int:
        return 10 - self.stich


class _EngineAdapter:
    """Bridges the GUI's Card-based GameLogic to the integer-based StrategyEngine."""

    def __init__(self) -> None:
        self.engine = StrategyEngine()
        self._decl_model = DeclarationModel(use_target_aware_opponents=False)
        self.last_decl_info: dict | None = None

    def reset(self) -> None:
        self.engine.reset()
        self.last_decl_info = None

    def record_void(self, player: int, suit_name: str) -> None:
        self.engine.record_void(player, SUIT_INDEX[suit_name])

    def ansage(self, game: GameLogic) -> int:
        if not game.hand or not game.trump:
            return 0
        hand_ids = [c.id for c in game.hand]
        trump_idx = SUIT_INDEX[game.trump]
        info = self._decl_model.declare_with_info(
            hand_ids, trump_idx,
            voids=self.engine.player_voids,
            num_worlds=200,
        )
        self.last_decl_info = info
        return info["declaration"]

    def zug(self, game: GameLogic) -> Card | None:
        if not game.hand or not game.trump:
            return None

        hand_ids = [c.id for c in game.hand]
        trump_idx = SUIT_INDEX[game.trump]

        # Build table cards as (player, card_id)
        table_cards: list[tuple[int, int]] = []
        for i in range(4):
            p = (game.ausspieler + i) % 4
            if p in game.tisch:
                table_cards.append((p, game.tisch[p].id))

        played_ids = [c.id for c in game.gespielt]

        # Sync voids from history
        for entry in game.history:
            lead_card = entry["karten"].get(
                list(entry["karten"].keys())[0] if entry["karten"] else 0
            )
            if lead_card:
                lead_s = lead_card.suit
                for p, card in entry["karten"].items():
                    if p > 0 and card.suit != lead_s:
                        self.engine.record_void(p, SUIT_INDEX[lead_s])

        card_id_result = self.engine.zug(
            hand=hand_ids,
            trump=trump_idx,
            target=game.ziel,
            current_points=game.punkte,
            table_cards=table_cards,
            leader=game.ausspieler,
            played_cards=played_ids,
        )

        if card_id_result is not None:
            return _ID_TO_CARD.get(card_id_result)
        return None


class JassAssistent:
    def __init__(self) -> None:
        pygame.init()

        info = pygame.display.Info()
        self.W = info.current_w // 2
        self.H = info.current_h - 100

        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{info.current_w // 2},50"

        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Jass")

        self.font_xl = pygame.font.SysFont("Arial", 32)
        self.font_lg = pygame.font.SysFont("Arial", 24)
        self.font_md = pygame.font.SysFont("Arial", 20)
        self.font_sm = pygame.font.SysFont("Arial", 16)

        self.game = GameLogic()
        self.adapter = _EngineAdapter()
        print("Strategy Engine loaded")

        self.msg = "Trumpf → Karten → ANSAGE"
        self.msg_color = WHITE
        self.popup = False
        self.popup_target: str | None = None
        self.recommended: Card | None = None

        self.card_w, self.card_h = 45, 65
        self.setup_layout()

    def setup_layout(self) -> None:
        self.trump_btns = [(s, pygame.Rect(10 + i * 50, 10, 45, 45)) for i, s in enumerate(SUITS)]

        cx, cy = self.W // 2, self.H // 2 - 20

        self.player_slots = {
            0: ("ICH", pygame.Rect(cx - 22, cy + 50, 45, 65), GOLD),
            1: ("RECHTS", pygame.Rect(cx + 50, cy - 20, 45, 65), LIGHT_GRAY),
            2: ("OBEN", pygame.Rect(cx - 22, cy - 90, 45, 65), LIGHT_GRAY),
            3: ("LINKS", pygame.Rect(cx - 95, cy - 20, 45, 65), LIGHT_GRAY),
        }

        bx = self.W - 85
        self.btns = {
            "ansage": pygame.Rect(bx, 10, 75, 32),
            "zug": pygame.Rect(bx, 47, 75, 32),
            "stich": pygame.Rect(bx, 95, 75, 28),
            "neu": pygame.Rect(bx, 138, 75, 28),
            "x": pygame.Rect(bx, self.H - 35, 75, 28),
        }

        self.hand_y = self.H - 90
        self.popup_rect = pygame.Rect(15, 15, self.W - 30, self.H - 30)

    def run(self) -> None:
        clock = pygame.time.Clock()
        running = True

        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        if self.popup:
                            self.popup = False
                        else:
                            running = False
                    elif e.key == pygame.K_RETURN and self.popup:
                        self.popup = False
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    self.click(e.pos, e.button)

            self.draw()
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def click(self, pos: tuple[int, int], btn: int) -> None:
        if self.popup:
            self.popup_click(pos)
            return

        for suit, rect in self.trump_btns:
            if rect.collidepoint(pos):
                self.game.trump = suit
                self.msg = f"Trumpf: {SUIT_SYMBOLS[suit]}"
                return

        if self.btns["ansage"].collidepoint(pos):
            if self.game.trump and self.game.hand:
                self.game.ziel = self.adapter.ansage(self.game)
                info = self.adapter.last_decl_info
                if info:
                    self.msg = f"ANSAGE:{self.game.ziel} σ={info['std']:.0f} [{info['min']}-{info['max']}]"
                else:
                    self.msg = f"ANSAGE: {self.game.ziel}"
            else:
                self.msg = "Trumpf + Karten!"
                self.msg_color = RED
            return

        if self.btns["zug"].collidepoint(pos):
            if self.game.trump and self.game.hand:
                card = self.adapter.zug(self.game)
                if card and card in self.game.hand:
                    self.recommended = card
                    self.game.karte_spielen(0, card)
                    self.msg = f"► {card.name()}"
                else:
                    self.msg = "?"
            return

        if self.btns["stich"].collidepoint(pos):
            if self.game.tisch:
                gewinner = self.game.stich_beenden()
                if gewinner is not None:
                    self.msg = f"{self.game.PLAYERS[gewinner]} gewinnt! Ich:{self.game.punkte}P"
                else:
                    self.msg = "Stich beendet"
                self.recommended = None
            return

        if self.btns["neu"].collidepoint(pos):
            self.game.reset()
            self.adapter.reset()
            self.msg = "Neu"
            self.recommended = None
            return

        if self.btns["x"].collidepoint(pos):
            pygame.quit()
            exit()

        for i, c in enumerate(self.game.hand):
            x = 10 + i * (self.card_w + 4)
            if pygame.Rect(x, self.hand_y, self.card_w, self.card_h).collidepoint(pos):
                if btn == 3:
                    self.game.hand.remove(c)
                else:
                    self.game.karte_spielen(0, c)
                    self.msg = f"Gespielt: {c.name()}"
                self.recommended = None
                return

        max_hand = self.game.get_max_hand_size()
        if pygame.Rect(10, self.hand_y, 9 * (self.card_w + 4), self.card_h).collidepoint(pos):
            if len(self.game.hand) < max_hand:
                self.popup = True
                self.popup_target = "hand"
            return

        for spieler, (name, rect, col) in self.player_slots.items():
            if rect.collidepoint(pos):
                if spieler in self.game.tisch:
                    karte = self.game.tisch.pop(spieler)
                    if spieler == 0:
                        self.game.hand.append(karte)
                    self.msg = f"{name}: entfernt"
                else:
                    if spieler != 0:
                        self.popup = True
                        self.popup_target = f"spieler_{spieler}"
                        self.msg = f"{name}: Karte wählen"
                return

    def popup_click(self, pos: tuple[int, int]) -> None:
        if not self.popup_rect.collidepoint(pos):
            self.popup = False
            return

        done = pygame.Rect(self.popup_rect.right - 80, self.popup_rect.bottom - 35, 70, 28)
        if done.collidepoint(pos):
            self.popup = False
            return

        gx, gy = self.popup_rect.x + 45, self.popup_rect.y + 45
        cw, ch = 50, 70
        gapx, gapy = 55, 78

        used = set(self.game.hand + list(self.game.tisch.values()) + self.game.gespielt)

        for si, suit in enumerate(SUITS):
            for vi, val in enumerate(VALUES):
                x, y = gx + vi * gapx, gy + si * gapy
                if pygame.Rect(x, y, cw, ch).collidepoint(pos):
                    card = Card(suit, val)
                    if card not in used:
                        if self.popup_target == "hand":
                            max_hand = self.game.get_max_hand_size()
                            if len(self.game.hand) < max_hand:
                                self.game.hand.append(card)
                                self.msg = f"+{card.name()} ({len(self.game.hand)}/{max_hand})"
                        elif self.popup_target and self.popup_target.startswith("spieler_"):
                            spieler = int(self.popup_target.split("_")[1])
                            self.game.karte_spielen(spieler, card)
                            self.msg = f"{self.game.PLAYERS[spieler]}: {card.name()}"
                            self.popup = False
                    return

    def draw(self) -> None:
        self.screen.fill(DARK_GREEN)

        # Table
        pygame.draw.rect(self.screen, GREEN, (self.W // 2 - 100, self.H // 2 - 110, 200, 200), border_radius=8)

        # Trump buttons
        for suit, rect in self.trump_btns:
            col = GOLD if self.game.trump == suit else GRAY
            pygame.draw.rect(self.screen, col, rect, border_radius=4)
            self.screen.blit(self.font_xl.render(SUIT_SYMBOLS[suit], True, SUIT_COLORS[suit]), (rect.x + 8, rect.y + 5))

        # Info panel
        x = self.W - 85
        diff = self.game.ziel - self.game.punkte
        self.screen.blit(self.font_sm.render(f"Stich {self.game.stich}/9", True, WHITE), (x, 180))
        self.screen.blit(self.font_sm.render(f"Ziel:{self.game.ziel}", True, WHITE), (x, 198))
        self.screen.blit(self.font_sm.render(f"Habe:{self.game.punkte}", True, WHITE), (x, 216))
        self.screen.blit(self.font_lg.render(f"{diff:+d}", True, WHITE if diff >= 0 else RED), (x, 235))

        gewinner = self.game.get_stich_gewinner()
        if gewinner is not None:
            punkte = self.game.get_stich_punkte()
            self.screen.blit(self.font_sm.render(f"Gewinnt:{self.game.PLAYERS[gewinner]}", True, GOLD), (x, 265))
            self.screen.blit(self.font_sm.render(f"({punkte}P)", True, GOLD), (x, 283))

        # Player slots
        for spieler, (name, rect, col) in self.player_slots.items():
            name_txt = self.font_sm.render(name, True, col)
            self.screen.blit(name_txt, (rect.x, rect.y - 15))

            if spieler > 0:
                pts_txt = self.font_sm.render(f"{self.game.spieler_punkte[spieler]}P", True, LIGHT_GRAY)
                self.screen.blit(pts_txt, (rect.x, rect.y + rect.h + 2))

            if spieler in self.game.tisch:
                karte = self.game.tisch[spieler]
                is_winner = (gewinner == spieler)
                self.draw_card(karte, rect.x, rect.y, hl=is_winner)
            else:
                pygame.draw.rect(self.screen, DARK_GREEN, rect, border_radius=3)
                pygame.draw.rect(self.screen, col, rect, 2 if spieler == 0 else 1, border_radius=3)
                self.screen.blit(self.font_md.render("+", True, col), (rect.centerx - 5, rect.centery - 8))

        # Table points
        if self.game.tisch:
            tisch_punkte = self.game.get_stich_punkte()
            self.screen.blit(self.font_md.render(f"{tisch_punkte}P", True, GOLD), (self.W // 2 - 12, self.H // 2 - 10))

        # Hand
        max_hand = self.game.get_max_hand_size()
        self.screen.blit(self.font_sm.render(f"Hand ({len(self.game.hand)}/{max_hand})", True, WHITE), (10, self.hand_y - 18))
        for i in range(max_hand):
            x = 10 + i * (self.card_w + 4)
            if i < len(self.game.hand):
                card = self.game.hand[i]
                self.draw_card(card, x, self.hand_y, hl=(card == self.recommended))
            else:
                pygame.draw.rect(self.screen, DARK_GREEN, (x, self.hand_y, self.card_w, self.card_h), border_radius=3)
                pygame.draw.rect(self.screen, GRAY, (x, self.hand_y, self.card_w, self.card_h), 1, border_radius=3)

        # Buttons
        for name, (txt, col) in [
            ("ansage", ("ANSAGE", BLUE)),
            ("zug", ("ZUG?", BLUE)),
            ("stich", ("STICH", GREEN)),
            ("neu", ("NEU", GOLD)),
            ("x", ("X", GRAY)),
        ]:
            r = self.btns[name]
            pygame.draw.rect(self.screen, col, r, border_radius=4)
            t = self.font_sm.render(txt, True, WHITE)
            self.screen.blit(t, t.get_rect(center=r.center))

        # Message bar
        pygame.draw.rect(self.screen, BLACK, (10, self.H - 30, self.W - 100, 24), border_radius=3)
        self.screen.blit(self.font_md.render(self.msg[:35], True, self.msg_color), (15, self.H - 28))

        if self.popup:
            self.draw_popup()

    def draw_card(self, c: Card, x: int, y: int, hl: bool = False) -> None:
        r = pygame.Rect(x, y, self.card_w, self.card_h)
        pygame.draw.rect(self.screen, WHITE, r, border_radius=3)
        if hl:
            pygame.draw.rect(self.screen, HIGHLIGHT, r, 3, border_radius=3)
        elif c.suit == self.game.trump:
            pygame.draw.rect(self.screen, GOLD, r, 2, border_radius=3)
        else:
            pygame.draw.rect(self.screen, BLACK, r, 1, border_radius=3)

        col = SUIT_COLORS[c.suit]
        self.screen.blit(self.font_sm.render(SUIT_SYMBOLS[c.suit], True, col), (x + 2, y + 1))
        self.screen.blit(self.font_lg.render(c.value, True, col), (x + self.card_w // 2 - 8, y + self.card_h // 2 - 5))

        p = c.points(self.game.trump)
        if p:
            self.screen.blit(self.font_sm.render(str(p), True, GRAY), (x + self.card_w - 14, y + self.card_h - 16))

    def draw_popup(self) -> None:
        pygame.draw.rect(self.screen, (0, 0, 0, 200), (0, 0, self.W, self.H))
        pygame.draw.rect(self.screen, DARK_GREEN, self.popup_rect, border_radius=6)
        pygame.draw.rect(self.screen, GOLD, self.popup_rect, 2, border_radius=6)

        used = set(self.game.hand + list(self.game.tisch.values()) + self.game.gespielt)

        title = f"Karten wählen ({len(used)} verwendet) → FERTIG"
        if self.popup_target and self.popup_target.startswith("spieler_"):
            spieler = int(self.popup_target.split("_")[1])
            title = f"{self.game.PLAYERS[spieler]}: Karte wählen"
        self.screen.blit(self.font_lg.render(title, True, WHITE), (self.popup_rect.x + 10, self.popup_rect.y + 8))

        done = pygame.Rect(self.popup_rect.right - 80, self.popup_rect.bottom - 35, 70, 28)
        pygame.draw.rect(self.screen, BLUE, done, border_radius=4)
        self.screen.blit(self.font_md.render("FERTIG", True, WHITE), (done.x + 8, done.y + 4))

        gx, gy = self.popup_rect.x + 45, self.popup_rect.y + 45
        cw, ch = 50, 70

        for si, suit in enumerate(SUITS):
            self.screen.blit(self.font_xl.render(SUIT_SYMBOLS[suit], True, SUIT_COLORS[suit]), (gx - 30, gy + si * 78 + 18))
            for vi, val in enumerate(VALUES):
                x, y = gx + vi * 55, gy + si * 78
                card = Card(suit, val)
                if card in used:
                    pygame.draw.rect(self.screen, GRAY, (x, y, cw, ch), border_radius=3)
                    self.screen.blit(self.font_lg.render("X", True, RED), (x + 18, y + 22))
                else:
                    self.draw_card(card, x, y)

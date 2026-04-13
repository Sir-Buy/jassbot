"""
Jass Bot Server — Flask API for Tampermonkey extension.

Wraps StrategyEngine + DeclarationModel + BeliefTracker.
Maintains round state between calls. Saves game data to disk.

Usage:
    python server.py
    python server.py --port 5000 --worlds 200
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure engine is importable
sys.path.insert(0, os.path.dirname(__file__))

from engine.cards import (
    card_short, card_name, CARD_SUIT, NUM_VALUES,
    points as card_points, SUIT_SYMBOLS, SUITS, VALUES,
)
from engine.game_state import GameState
from engine.rules import legal_moves, trick_winner, trick_points
from bot.strategy import StrategyEngine
from bot.declaration import DeclarationModel
from bot.declaration_net import NeuralDeclarationModel
from bot.belief_tracker import BeliefTracker
from bot.heuristics import estimate_hand_score

# RL mode support
_RL_MODE = False
_RL_SESS = None
_RL_NET = None  # PyTorch V7 PlayNet
_RL_DEVICE = None

def init_rl_mode():
    global _RL_MODE, _RL_SESS, _RL_NET, _RL_DEVICE
    import torch

    # Try V7 PyTorch checkpoint first (best engine)
    v7_path = os.path.join(os.path.dirname(__file__), "checkpoints_v7", "best.pt")
    if os.path.exists(v7_path):
        _RL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sys.path.insert(0, os.path.dirname(__file__))
        from bot.train_v7_league import PlayNet
        _RL_NET = PlayNet(204, 512).to(_RL_DEVICE)
        ckpt = torch.load(v7_path, map_location=_RL_DEVICE, weights_only=False)
        _RL_NET.load_state_dict(ckpt['play_net'])
        _RL_NET.eval()
        _RL_MODE = True
        dev = ckpt.get('best_dev', '?')
        print(f"[RL] V7 PlayNet loaded: {v7_path} (dev={dev}, {_RL_DEVICE})")
        return

    # Fallback: ONNX
    try:
        import onnxruntime as ort
        onnx_path = os.path.join(os.path.dirname(__file__), "models", "play_net.onnx")
        if os.path.exists(onnx_path):
            _RL_SESS = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            _RL_MODE = True
            print(f"[RL] ONNX loaded: {onnx_path}")
            return
    except ImportError:
        pass

    print(f"[RL] No model found, falling back to PIMC")

app = Flask(__name__)
CORS(app)  # Allow requests from swisslos.ch

# ── Config ──────────────────────────────────────────────────────────────

LIVE_DIR = Path(__file__).parent.parent / 'Jass Live'
EXPORT_DIR = LIVE_DIR / 'games_solver_v3' if LIVE_DIR.exists() else Path(__file__).parent / 'data' / 'export_games_v3'
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
GAMES_DIR = Path(__file__).parent / 'data' / 'games'
GAMES_DIR.mkdir(parents=True, exist_ok=True)
SESSION_FILE = LIVE_DIR / 'sessions' / 'session_pnl.json' if LIVE_DIR.exists() else Path(__file__).parent / 'data' / 'session_pnl.json'
SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)

DEFAULT_WORLDS = 2000
DEFAULT_ANSAGE_WORLDS = 10000
DEFAULT_ENDGAME_DEPTH = 5

# ── Round State ─────────────────────────────────────────────────────────
# Single-game state (one round at a time)

class RoundState:
    """Persistent state for one round of play."""

    def __init__(self, num_worlds=DEFAULT_WORLDS, endgame_depth=DEFAULT_ENDGAME_DEPTH):
        self.engine = StrategyEngine(
            num_worlds=num_worlds,
            ansage_worlds=DEFAULT_ANSAGE_WORLDS,
            use_target_aware_opponents=True,
            endgame_depth=endgame_depth,
        )
        self.neural_decl = NeuralDeclarationModel()
        self.decl_model = DeclarationModel(use_target_aware_opponents=True, percentile=50, offset=0)
        self.belief_tracker = BeliefTracker()
        self.target = 0
        self.trump = 0
        self.hand = []
        self.my_points = 0
        self.opp_points = [0, 0, 0, 0]
        self.trick_number = 0
        self.played_cards = []
        self.active = False
        self.my_seat = 0  # Swisslos seat number (set by /new-round)

    def _remap(self, swisslos_seat: int) -> int:
        """Remap Swisslos seat number to internal ID (0=us, 1/2/3=opponents).

        The engine always treats player 0 as 'us'. This maps the real
        Swisslos seat numbers so that our seat becomes 0, and the three
        opponents become 1, 2, 3 in clockwise order.
        """
        if swisslos_seat == self.my_seat:
            return 0
        # Opponents: clockwise from our seat → internal 1, 2, 3
        offset = (swisslos_seat - self.my_seat) % 4
        return offset

    def reset(self, hand, trump, my_seat=0):
        """Start a new round."""
        self.engine.reset()
        self.belief_tracker.init_round(trump)
        self.trump = trump
        self.hand = list(hand)
        self.my_points = 0
        self.opp_points = [0, 0, 0, 0]
        self.trick_number = 0
        self.played_cards = []
        self.active = True
        self.my_seat = my_seat

        # Declaration — use optimal single-agent solver
        t0 = time.perf_counter()

        decl_method = "?"
        try:
            from engine_c.fast_engine import FastEngine
            from ctypes import c_int, c_uint64
            from engine_c.fast_engine import hand_to_mask
            from bot.world_sampler import sample_worlds_with_declarations
            _solver = FastEngine()
            _dll = _solver._lib if hasattr(_solver, '_lib') else None

            unknown = [c for c in range(36) if c not in set(hand)]
            n_worlds = 800
            worlds_raw = sample_worlds_with_declarations(
                unknown, len(hand), {1: set(), 2: set(), 3: set()},
                n_worlds, trump,
            )
            worlds = [{1: wh[1], 2: wh[2], 3: wh[3]} for wh, _ in worlds_raw]
            opp_tgts = [{1: wd[1], 2: wd[2], 3: wd[3]} for _, wd in worlds_raw]

            # Step 1: Find min and max achievable scores per world
            # This tells us our CONTROL range
            my_mask = hand_to_mask(list(hand))
            min_scores = []
            max_scores = []

            import ctypes as _ct
            for w in range(min(n_worlds, 300)):
                wh = worlds[w]
                wt = opp_tgts[w]
                oh = (c_uint64 * 3)(hand_to_mask(wh[1]), hand_to_mask(wh[2]), hand_to_mask(wh[3]))
                ot = (c_int * 3)(wt[1], wt[2], wt[3])
                pts = (c_int * 4)(0, 0, 0, 0)

                from engine_c.fast_engine import _lib
                _lib.solve_single_agent.restype = c_int
                p_min = _lib.solve_single_agent(c_uint64(my_mask), oh, trump, 0, 0, pts, 0, len(hand), ot)
                p_max = _lib.solve_single_agent(c_uint64(my_mask), oh, trump, 157, 0, pts, 0, len(hand), ot)
                min_scores.append(p_min >> 8)
                max_scores.append(p_max >> 8)

            avg_min = sum(min_scores) / len(min_scores)
            avg_max = sum(max_scores) / len(max_scores)
            control_range = avg_max - avg_min

            # Step 2: Get solver's optimal declaration (now uses human-like opponent model)
            solver_decl = _solver.optimal_declaration(list(hand), worlds, trump, opp_tgts)

            # Step 3: Get expert heuristic as reality check
            expert_decl = estimate_hand_score(list(hand), trump)

            # Step 4: Blend based on declaration range
            # Data-driven calibration (36 live solver rounds):
            #   Dodge (0):     solver dev=0.6 — excellent, trust it
            #   Low (1-30):    solver dev=7.5 — good, trust it
            #   Mid (31-60):   solver dev=11.4, over-scores by +9.5
            #   High (61-90):  solver dev=18.7, under-scores by -18.7
            #   V.high (91+):  solver dev=23.7, under-scores by -23.7
            # Expert is much better on catastrophic rounds (avg dev 12.6 vs 30.3)
            # but worse on normal rounds (10.0 vs 1.4).
            # Strategy: trust solver for low, blend with expert for high.
            if solver_decl <= 10 and expert_decl - solver_decl > 15:
                # Solver says dodge but expert strongly disagrees (>15pt gap).
                # v2 data: solver declared 0 on a hand with DA+HA → actual=29, dev=29.
                # Solver overestimates ability to shed aces; expert is more realistic.
                self.target = round(solver_decl * 0.5 + expert_decl * 0.5)
                decl_method = f"dodge_sanity(solver={solver_decl},expert={expert_decl})"
            elif solver_decl <= 30:
                # Solver excels here (dodge + low)
                self.target = solver_decl
                decl_method = f"solver(decl={solver_decl},expert={expert_decl})"
            elif solver_decl <= 60:
                # Mid range: solver slightly under-declares, blend lightly with expert
                # Both have comparable dev here, average them
                self.target = round(solver_decl * 0.7 + expert_decl * 0.3)
                decl_method = f"blend70(solver={solver_decl},expert={expert_decl})"
            else:
                # High range: solver systematically over-declares
                # Expert is dramatically better on catastrophic rounds
                # Heavy blend toward expert
                self.target = round(solver_decl * 0.4 + expert_decl * 0.6)
                decl_method = f"blend40(solver={solver_decl},expert={expert_decl})"

            self.target = max(0, min(157, self.target))
            print(f"[Decl] min={avg_min:.0f} max={avg_max:.0f} range={control_range:.0f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.target = estimate_hand_score(list(hand), trump)
            decl_method = f"expert(fallback: {e})"

        decl_ms = (time.perf_counter() - t0) * 1000

        print(f"[Round] Trump={SUITS[trump]} Hand={[card_short(c) for c in hand]} Seat={my_seat}")
        print(f"[Round] Declaration: {self.target} ({decl_ms:.0f}ms, {decl_method})")

        return self.target, decl_ms

    def play(self, hand, trick_cards, leader, legal, opp_points=None):
        """Choose a card to play.

        trick_cards and leader use INTERNAL seat IDs (already remapped).
        opp_points is [opp1_pts, opp2_pts, opp3_pts] for internal IDs 1,2,3.
        """
        if not self.active:
            return legal[0] if legal else 0, 0

        if opp_points:
            self.opp_points = list(opp_points)

        t0 = time.perf_counter()

        if len(legal) == 1:
            card = legal[0]
        else:
            # Try optimal solver first, fall back to PIMC
            try:
                from engine_c.fast_engine import FastEngine
                from bot.world_sampler import sample_worlds_with_declarations
                _solver = FastEngine()

                unknown = [c for c in range(36) if c not in set(hand)
                           and c not in set(self.played_cards)]
                # Remove table cards from unknown
                for _, tc in trick_cards:
                    if tc in unknown:
                        unknown.remove(tc)

                n_worlds = min(500, max(200, 3000 // max(1, len(hand))))
                worlds_raw = sample_worlds_with_declarations(
                    unknown, len(hand), self.engine.player_voids,
                    n_worlds, self.trump,
                    belief_tracker=self.belief_tracker,
                )
                worlds = [{1: wh[1], 2: wh[2], 3: wh[3]} for wh, _ in worlds_raw]
                opp_tgts = [{1: wd[1], 2: wd[2], 3: wd[3]} for _, wd in worlds_raw]
                opp_pts = [{1: self.opp_points[1], 2: self.opp_points[2],
                            3: self.opp_points[3]}] * len(worlds)

                total_tricks = self.trick_number + len(hand)

                utils = _solver.optimal_play(
                    legal, list(hand), worlds, self.trump, self.target,
                    self.my_points, opp_tgts, opp_pts,
                    leader, self.trick_number, total_tricks, trick_cards,
                )
                card = max(legal, key=lambda c: utils.get(c, float('-inf')))

                # ── Safety rails ──────────────────────────────────────
                from engine.cards import strength as c_strength
                gap = self.target - self.my_points

                # Rail 1: Shed lead override
                # When at/over target (gap<=0) and we lead, always play weakest.
                # Avoid trump strongly — leading trump when shedding wins tricks.
                # Data: 9 overshoot events (157pts) caused by leading strong when over target.
                if gap <= 0 and not trick_cards and len(legal) > 1:
                    def _shed_key(c):
                        is_trump = 1 if CARD_SUIT[c] == self.trump else 0
                        return (is_trump * 1000
                                + card_points(c, self.trump) * 100
                                + c_strength(c, self.trump))
                    weakest = min(legal, key=_shed_key)
                    if card != weakest:
                        print(f"[Safety] Shed lead override: {card_short(card)} -> {card_short(weakest)} (gap={gap})")
                        card = weakest

                # Rail 2: Shed follow override
                # When at/over target and following, prefer non-winning cards.
                # The solver's opponent model may not predict real outcomes perfectly,
                # so explicitly prefer ducking when we don't need points.
                if gap <= 0 and trick_cards and len(legal) > 1:
                    from engine.rules import trick_winner as tw_fn

                    lead_suit_id = CARD_SUIT[trick_cards[0][1]]

                    def _would_win(c):
                        """Check if playing card c would currently win the trick."""
                        test_cards = list(trick_cards) + [(0, c)]
                        try:
                            winner = tw_fn(test_cards, lead_suit_id, self.trump)
                            return winner == 0  # player 0 is us
                        except Exception:
                            return False

                    non_winners = [c for c in legal if not _would_win(c)]
                    if non_winners:
                        # Among non-winners, pick the one with lowest point value
                        best_duck = min(non_winners, key=lambda c: card_points(c, self.trump) * 100 + c_strength(c, self.trump))
                        if card != best_duck and _would_win(card):
                            print(f"[Safety] Shed follow override: {card_short(card)} -> {card_short(best_duck)} (gap={gap})")
                            card = best_duck

                # Rail 3: Follow play management (cheapest winner + duck)
                # v3 data: HA(11pts) instead of HJ(2pts) at gap=12; SA(32pt trick) at gap=14.
                if 0 < gap and trick_cards and len(legal) > 1:
                    from engine.rules import trick_winner as tw_fn2

                    lead_suit_id2 = CARD_SUIT[trick_cards[0][1]]

                    def _would_win2(c):
                        test_cards = list(trick_cards) + [(0, c)]
                        try:
                            winner = tw_fn2(test_cards, lead_suit_id2, self.trump)
                            return winner == 0
                        except Exception:
                            return False

                    def _card_cost(c):
                        """Points this card adds if we win."""
                        return card_points(c, self.trump)

                    winners = [c for c in legal if _would_win2(c)]
                    non_winners = [c for c in legal if not _would_win2(c)]

                    if _would_win2(card):
                        # 3a: Cheapest winner — when multiple cards win,
                        # use the one that adds fewest points to the trick.
                        # Fixes: HA(11pts) vs HJ(2pts) at gap=12.
                        if gap < 20 and len(winners) > 1:
                            cheapest_win = min(winners, key=lambda c: _card_cost(c))
                            if _card_cost(card) > _card_cost(cheapest_win) + 3:
                                print(f"[Safety] Cheapest winner: {card_short(card)} ({_card_cost(card)}pts) -> {card_short(cheapest_win)} ({_card_cost(cheapest_win)}pts) (gap={gap})")
                                card = cheapest_win

                        # 3b: Duck strong cards when gap is small.
                        # Strong trump/aces guarantee winning but risk huge tricks
                        # from opponents' cards played after us.
                        # Fixes: SA on 32pt trick at gap=14.
                        if non_winners and _would_win2(card):
                            is_strong = (c_strength(card, self.trump) >= 100  # any trump
                                         or _card_cost(card) >= 10)  # ace or 10
                            if gap < 20 and is_strong:
                                best_duck = min(non_winners, key=lambda c: _card_cost(c) * 100 + c_strength(c, self.trump))
                                print(f"[Safety] Duck strong: {card_short(card)} -> {card_short(best_duck)} (gap={gap}, strong card risk)")
                                card = best_duck

                # Rail 4: Small-gap lead override
                # When gap is small (1-10) and leading, play weakest to avoid
                # winning a big trick. v3 data: led HJ at gap=1 → won 12pts.
                if 0 < gap <= 10 and not trick_cards and len(legal) > 1:
                    def _lead_risk(c):
                        is_trump = 1 if CARD_SUIT[c] == self.trump else 0
                        return (is_trump * 1000
                                + card_points(c, self.trump) * 100
                                + c_strength(c, self.trump))
                    weakest = min(legal, key=_lead_risk)
                    if card != weakest and _lead_risk(card) > _lead_risk(weakest) + 200:
                        print(f"[Safety] Small-gap lead: {card_short(card)} -> {card_short(weakest)} (gap={gap})")
                        card = weakest

                # Rail 5: Overshoot protection (small positive gap)
                # When gap is small positive (10 < gap < 25) and we lead,
                # don't lead with strong trump — risk of winning a huge trick.
                if 10 < gap < 25 and not trick_cards and len(legal) > 1:
                    if CARD_SUIT[card] == self.trump and c_strength(card, self.trump) >= 100:
                        non_trump = [c for c in legal if CARD_SUIT[c] != self.trump]
                        if non_trump:
                            best_nt = max(non_trump, key=lambda c: utils.get(c, float('-inf')))
                            print(f"[Safety] Overshoot guard: {card_short(card)} -> {card_short(best_nt)} (gap={gap})")
                            card = best_nt

            except Exception as e:
                print(f"[Play] Solver failed ({e}), falling back to PIMC")
                card = self.engine.zug(
                    hand=list(hand),
                    trump=self.trump,
                    target=self.target,
                    current_points=self.my_points,
                    table_cards=trick_cards,
                    leader=leader,
                    played_cards=list(self.played_cards),
                    trick_number=self.trick_number,
                    belief_tracker=self.belief_tracker,
                    opp_points=self.opp_points,
                )
                if card is None or card not in legal:
                    card = legal[0]

        play_ms = (time.perf_counter() - t0) * 1000
        print(f"[Play] Trick {self.trick_number}: {card_short(card)} ({play_ms:.0f}ms) "
              f"from {[card_short(c) for c in legal]}")

        return card, play_ms

    def _rl_play(self, hand, trick_cards, legal):
        """RL inference for card selection (V7 PyTorch or ONNX fallback)."""
        import numpy as np
        obs = np.zeros(204, dtype=np.float32)
        trump = self.trump

        for c in hand: obs[c] = 1.0
        tc_set = set(c for _, c in trick_cards)
        for c in self.played_cards:
            if c not in tc_set: obs[36 + c] = 1.0
        for c in range(trump*9, trump*9+9): obs[72 + c] = 1.0
        for c in legal: obs[108 + c] = 1.0
        for _, c in trick_cards: obs[144 + c] = 1.0
        obs[192] = self.trick_number / 8.0
        obs[193] = self.my_points / 157.0
        obs[194] = max(-1.0, min(1.0, (self.target - self.my_points) / 157.0))
        obs[195] = len(trick_cards) / 3.0
        obs[196] = 1.0 if not trick_cards else 0.0
        obs[203] = self.target / 157.0

        mask = np.zeros(36, dtype=np.float32)
        for c in legal: mask[c] = 1.0

        if _RL_NET is not None:
            # V7 PyTorch
            import torch
            obs_t = torch.tensor(obs, dtype=torch.float32, device=_RL_DEVICE).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32, device=_RL_DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits, _ = _RL_NET(obs_t, mask_t)
            card = int(logits[0].argmax().item())
        elif _RL_SESS is not None:
            # ONNX fallback
            logits, _ = _RL_SESS.run(None, {
                'obs': obs.reshape(1, -1),
                'legal_mask': mask.reshape(1, -1),
            })
            card = int(np.argmax(logits[0]))
        else:
            card = legal[0]

        if card not in legal:
            card = legal[0]
        return card

    def notify_trick(self, trick, winner, pts, lead_suit):
        """Update state after a trick completes.

        trick uses INTERNAL seat IDs (already remapped). winner is internal.
        """
        # Update points
        if winner == 0:
            self.my_points += pts
        self.opp_points[winner] = self.opp_points[winner] + pts

        # Update beliefs
        cards_before = []
        for player, card in trick:
            if player != 0 and player in [1, 2, 3]:
                self.belief_tracker.update(
                    player=player,
                    action=card,
                    player_current_points=self.opp_points[player] - (pts if winner == player else 0),
                    trick_number=self.trick_number,
                    table_cards_before=list(cards_before),
                    lead_suit=lead_suit if cards_before else None,
                    trump=self.trump,
                )
            cards_before.append((player, card))

            # Record voids
            if cards_before and len(cards_before) > 1:
                first_suit = CARD_SUIT[cards_before[0][1]]
                if CARD_SUIT[card] != first_suit and player != 0:
                    self.engine.record_void(player, first_suit)

        # Track played cards
        for _, card in trick:
            self.played_cards.append(card)

        # Remove played cards from hand
        for player, card in trick:
            if player == 0 and card in self.hand:
                self.hand.remove(card)

        self.trick_number += 1
        print(f"[Trick {self.trick_number - 1}] Winner: internal {winner} ({pts}pts) "
              f"My points: {self.my_points}")


# ── Session PNL Tracker ────────────────────────────────────────────────

class SessionPNL:
    """Tracks profit/loss across the session."""

    def __init__(self):
        self.start_time = datetime.now().isoformat()
        self.games: list[dict] = []
        self.total_rounds = 0
        self.total_dev = 0
        self.perfect_rounds = 0
        self.wins = 0  # 1st place
        self.losses = 0  # not 1st
        self._load()

    def _load(self):
        if SESSION_FILE.exists():
            try:
                data = json.loads(SESSION_FILE.read_text())
                self.games = data.get('games', [])
                self.total_rounds = data.get('total_rounds', 0)
                self.total_dev = data.get('total_dev', 0)
                self.perfect_rounds = data.get('perfect_rounds', 0)
                self.wins = data.get('wins', 0)
                self.losses = data.get('losses', 0)
                self.start_time = data.get('start_time', self.start_time)
            except Exception:
                pass

    def _save(self):
        SESSION_FILE.write_text(json.dumps(self.to_dict(), indent=2))

    def record_round(self, target: int, scored: int, rank: int,
                     trump: int, opponent_names: list[str] = None,
                     match_id: str = ''):
        dev = abs(target - scored)
        game = {
            'timestamp': datetime.now().isoformat(),
            'target': target,
            'scored': scored,
            'deviation': dev,
            'rank': rank,
            'trump': ['H', 'D', 'S', 'C'][trump] if trump < 4 else '?',
            'opponents': opponent_names or [],
            'match_id': match_id,
        }
        self.games.append(game)
        self.total_rounds += 1
        self.total_dev += dev
        if dev == 0:
            self.perfect_rounds += 1
        if rank == 1:
            self.wins += 1
        else:
            self.losses += 1
        self._save()
        return game

    def to_dict(self) -> dict:
        avg_dev = self.total_dev / max(self.total_rounds, 1)
        win_rate = self.wins / max(self.total_rounds, 1) * 100
        return {
            'start_time': self.start_time,
            'total_rounds': self.total_rounds,
            'total_dev': self.total_dev,
            'avg_dev': round(avg_dev, 1),
            'perfect_rounds': self.perfect_rounds,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': round(win_rate, 1),
            'games': self.games[-50:],  # last 50 for API response
            'last_10_avg_dev': round(
                sum(g['deviation'] for g in self.games[-10:]) / max(len(self.games[-10:]), 1), 1
            ) if self.games else 0,
        }

    def reset(self):
        self.games = []
        self.total_rounds = 0
        self.total_dev = 0
        self.perfect_rounds = 0
        self.wins = 0
        self.losses = 0
        self.start_time = datetime.now().isoformat()
        self._save()


session_pnl = SessionPNL()


# Global round state
round_state = None
num_worlds_cfg = DEFAULT_WORLDS
endgame_depth_cfg = DEFAULT_ENDGAME_DEPTH


def get_round_state():
    global round_state
    if round_state is None:
        round_state = RoundState(num_worlds_cfg, endgame_depth_cfg)
    return round_state


# ── Swisslos card code conversion ──────────────────────────────────────

SWISSLOS_SUIT = {'H': 0, 'D': 1, 'S': 2, 'C': 3}
SWISSLOS_VAL = {'6': 0, '7': 1, '8': 2, '9': 3, '10': 4, 'J': 5, 'Q': 6, 'K': 7, 'A': 8}

def swisslos_to_id(code):
    """Convert 'H6', 'DJ', 'SA' to card ID 0-35."""
    if not code or len(code) < 2:
        return None
    s = SWISSLOS_SUIT.get(code[0])
    v = SWISSLOS_VAL.get(code[1:])
    if s is None or v is None:
        return None
    return s * NUM_VALUES + v


def id_to_swisslos(card_id):
    """Convert card ID 0-35 to 'H6', 'DJ', 'SA'."""
    suit = card_id // NUM_VALUES
    val = card_id % NUM_VALUES
    suit_chars = ['H', 'D', 'S', 'C']
    val_chars = ['6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
    return suit_chars[suit] + val_chars[val]


# ── API Endpoints ──────────────────────────────────────────────────────

@app.route('/status', methods=['GET'])
def status():
    rs = get_round_state()
    return jsonify({
        'ok': True,
        'version': '2.0',
        'engine': 'V7-RL' if _RL_NET else ('RL-ONNX' if _RL_SESS else 'PIMC'),
        'worlds': num_worlds_cfg,
        'endgame_depth': endgame_depth_cfg,
        'round_active': rs.active,
        'trick_number': rs.trick_number,
        'target': rs.target,
        'my_points': rs.my_points,
        'my_seat': rs.my_seat,
        'session': session_pnl.to_dict(),
    })


@app.route('/new-round', methods=['POST'])
def new_round():
    """Start a new round. Expects: {hand: [card_codes], trump: "H"/"D"/"S"/"C", my_pos: N}"""
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    # Parse hand — accept both card codes ("H6") and card IDs (0-35)
    raw_hand = data.get('hand', [])
    hand = []
    for c in raw_hand:
        if isinstance(c, int):
            hand.append(c)
        elif isinstance(c, str):
            cid = swisslos_to_id(c)
            if cid is not None:
                hand.append(cid)

    # Parse trump
    trump_raw = data.get('trump', 0)
    if isinstance(trump_raw, str):
        trump = SWISSLOS_SUIT.get(trump_raw, 0)
    else:
        trump = int(trump_raw)

    # Parse our Swisslos seat position (critical for correct seat mapping)
    my_seat = data.get('my_pos', 0)
    if my_seat < 0:
        print(f"[WARN] my_pos={my_seat}, defaulting to 0. Extension may need update.")
        my_seat = 0

    if not hand:
        return jsonify({'error': 'Empty hand'}), 400

    rs = get_round_state()
    target, decl_ms = rs.reset(hand, trump, my_seat=my_seat)

    return jsonify({
        'declaration': target,
        'time_ms': round(decl_ms, 1),
        'hand': [id_to_swisslos(c) for c in hand],
        'trump': ['H', 'D', 'S', 'C'][trump],
        'my_seat': my_seat,
    })


@app.route('/play', methods=['POST'])
def play():
    """Choose a card. Expects game state with Swisslos seat numbers."""
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    rs = get_round_state()

    # Parse hand
    raw_hand = data.get('hand', [])
    hand = []
    for c in raw_hand:
        if isinstance(c, int):
            hand.append(c)
        elif isinstance(c, str):
            cid = swisslos_to_id(c)
            if cid is not None:
                hand.append(cid)

    # Parse trick cards on table: [{player: N, card: "H6"}, ...]
    # Remap Swisslos seat numbers to internal IDs
    trick_cards = []
    for tc in data.get('trick_cards', []):
        sw_player = tc.get('player', 0)
        card_raw = tc.get('card', tc.get('card_id', 0))
        if isinstance(card_raw, str):
            card_raw = swisslos_to_id(card_raw)
        if card_raw is not None:
            trick_cards.append((rs._remap(sw_player), card_raw))

    # Parse legal moves
    raw_legal = data.get('legal_moves', data.get('legal', []))
    legal = []
    for c in raw_legal:
        if isinstance(c, int):
            legal.append(c)
        elif isinstance(c, str):
            cid = swisslos_to_id(c)
            if cid is not None:
                legal.append(cid)

    if not legal:
        # Compute legal moves ourselves
        lead_suit = CARD_SUIT[trick_cards[0][1]] if trick_cards else None
        legal = legal_moves(hand, lead_suit, rs.trump, trick_cards if trick_cards else None)
        if not legal:
            legal = list(hand)

    # Remap leader from Swisslos seat to internal ID
    raw_leader = data.get('leader', trick_cards[0][0] if trick_cards else 0)
    # If trick_cards is non-empty, leader was already remapped via trick_cards[0][0]
    # Only remap if leader came from data directly
    if data.get('leader') is not None:
        leader = rs._remap(raw_leader)
    else:
        leader = trick_cards[0][0] if trick_cards else 0

    # Remap opponent points: extension sends [pts_for_each_seat_except_ours]
    # We need to convert to [0, opp1_pts, opp2_pts, opp3_pts] indexed by internal ID
    raw_opp_points = data.get('opp_points', None)
    opp_points = None
    if raw_opp_points is not None:
        # Extension sends oppPoints as array of 3 values for seats != myPos, in seat order
        opp_points = [0, 0, 0, 0]  # indexed by internal ID
        opp_idx = 0
        for sw_seat in range(4):
            if sw_seat == rs.my_seat:
                continue
            internal = rs._remap(sw_seat)
            if opp_idx < len(raw_opp_points):
                opp_points[internal] = raw_opp_points[opp_idx]
            opp_idx += 1

    card, play_ms = rs.play(hand, trick_cards, leader, legal, opp_points)

    gap = rs.target - rs.my_points
    # Void info for UI
    voids_info = {}
    for p in [1, 2, 3]:
        v = rs.engine.player_voids.get(p, set())
        if v:
            voids_info[p] = [['H','D','S','C'][s] for s in v]

    return jsonify({
        'card': card,
        'card_code': id_to_swisslos(card),
        'card_name': card_name(card),
        'time_ms': round(play_ms, 1),
        'target': rs.target,
        'my_points': rs.my_points,
        'gap': gap,
        'trick_number': rs.trick_number,
        'hand': [id_to_swisslos(c) for c in hand],
        'legal': [id_to_swisslos(c) for c in legal],
        'opp_points': rs.opp_points,
        'voids': voids_info,
        'engine': 'solver',
    })


@app.route('/notify-trick', methods=['POST'])
def notify_trick():
    """Notify the bot that a trick completed. Seat numbers are remapped."""
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    rs = get_round_state()

    # Parse trick — remap Swisslos seat numbers to internal IDs
    trick = []
    for tc in data.get('trick', []):
        sw_player = tc.get('player', 0)
        card_raw = tc.get('card', tc.get('card_id', 0))
        if isinstance(card_raw, str):
            card_raw = swisslos_to_id(card_raw)
        if card_raw is not None:
            trick.append((rs._remap(sw_player), card_raw))

    # Remap winner
    winner = rs._remap(data.get('winner', 0))
    pts = data.get('points', 0)

    # Determine lead suit
    lead_suit = CARD_SUIT[trick[0][1]] if trick else None

    rs.notify_trick(trick, winner, pts, lead_suit)

    return jsonify({
        'ok': True,
        'trick_number': rs.trick_number,
        'my_points': rs.my_points,
    })


@app.route('/opponent-info', methods=['POST'])
def opponent_info():
    """Receive opponent declarations or any leaked game info.

    Called by the userscript when it captures opponent data from WebSocket messages.
    This data is gold for belief tracking and PIMC evaluation.
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    rs = get_round_state()

    # Opponent declarations: {pos: declared_points, ...}
    declarations = data.get('declarations', {})
    for sw_seat_str, decl_pts in declarations.items():
        sw_seat = int(sw_seat_str)
        internal = rs._remap(sw_seat)
        if internal in [1, 2, 3]:
            rs.belief_tracker.set_known_declaration(internal, decl_pts)
            print(f"[Intel] Opponent seat {sw_seat} (internal {internal}) declared {decl_pts}")

    # Opponent cumulative scores from match (total field in results)
    cumulative = data.get('cumulative_devs', {})
    # Store for meta-game awareness
    rs.opp_cumulative = {int(k): v for k, v in cumulative.items()} if cumulative else {}

    return jsonify({'ok': True, 'received': len(declarations)})


@app.route('/save', methods=['POST'])
def save_data():
    """Save match/round data to Export Games folder.

    Filename format: round_TRUMP_dDEV_pSCORED_MATCHID_TIMESTAMP.json
    Example: round_H_d3_p45_T-SP-D-3468375_2026-04-07T01-40-28.json
    """
    data = request.json
    if not data:
        return jsonify({'error': 'No JSON body'}), 400

    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%S')

    # Extract round info for descriptive filename
    tag = 'round'
    trump_code = ''
    dev_str = ''
    pts_str = ''
    match_id = data.get('match_id', '')

    rounds_list = data.get('rounds', [])
    if rounds_list:
        last_round = rounds_list[-1]
        trump_code = last_round.get('trump', '')

        # Find our bot's result (Gast_* or uid=None player)
        players = data.get('players', [])
        our_pos = None
        for p in players:
            name = p.get('name', '') if isinstance(p, dict) else str(p)
            uid = p.get('uid', 'x') if isinstance(p, dict) else 'x'
            if 'Gast' in name or uid is None:
                our_pos = p.get('pos') if isinstance(p, dict) else None
                break

        results = last_round.get('results', {})
        res_list = results.get('results', []) if isinstance(results, dict) else []
        for res in res_list:
            if our_pos is not None and res.get('pos') == our_pos:
                scored = res.get('points', 0)
                declared = res.get('callP', 0)
                dev = abs(declared - scored)
                dev_str = f'd{dev}'
                pts_str = f'p{scored}'
                break

    # Detect game type from players
    players = data.get('players', [])
    bot_count = sum(1 for p in players if isinstance(p, dict) and (
        (isinstance(p.get('uid'), int) and p['uid'] < 0) or
        any(pat in (p.get('name', '') or '').lower() for pat in ['computer', 'bot', 'compi'])
    ))
    game_type = 'bot' if bot_count >= 2 else 'human'

    # Build filename: round_bot_H_d3_p45_3468375_2026-04-07T01-40-28.json
    parts = [tag, game_type]
    if trump_code:
        parts.append(trump_code)
    if dev_str:
        parts.append(dev_str)
    if pts_str:
        parts.append(pts_str)
    # Shorten match_id: T-SP-D-3468375 -> 3468375
    if match_id:
        short_id = match_id.split('-')[-1] if '-' in match_id else match_id
        parts.append(short_id)
    parts.append(timestamp)

    filename = '_'.join(parts) + '.json'
    filepath = EXPORT_DIR / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Also save to data/games/ for analytics
    games_filepath = GAMES_DIR / filename
    with open(games_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(filepath) / 1024
    print(f"[Save] {filename} ({size_kb:.1f} KB) -> export + games/")

    return jsonify({
        'ok': True,
        'filename': filename,
        'path': str(filepath),
        'size_kb': round(size_kb, 1),
    })


@app.route('/end-round', methods=['POST'])
def end_round():
    """Signal that the round is over. Records PNL and resets state."""
    data = request.json or {}
    rs = get_round_state()

    dev = abs(rs.target - rs.my_points)
    rank = data.get('rank', 0)
    opponent_names = data.get('opponents', [])
    match_id = data.get('match_id', '')

    # Record in session PNL
    game = session_pnl.record_round(
        target=rs.target, scored=rs.my_points, rank=rank,
        trump=rs.trump, opponent_names=opponent_names, match_id=match_id,
    )

    result = {
        'target': rs.target,
        'my_points': rs.my_points,
        'deviation': dev,
        'rank': rank,
        'tricks_played': rs.trick_number,
        'session': session_pnl.to_dict(),
    }
    rs.active = False
    rank_str = f" Rank={rank}" if rank else ""
    print(f"[Round End] Target={rs.target} Scored={rs.my_points} "
          f"Dev={dev}{rank_str} | Session: {session_pnl.total_rounds} rounds, "
          f"avg dev={session_pnl.to_dict()['avg_dev']}, "
          f"W/L={session_pnl.wins}/{session_pnl.losses}")

    return jsonify(result)


@app.route('/session', methods=['GET'])
def get_session():
    """Get current session PNL stats."""
    return jsonify(session_pnl.to_dict())


@app.route('/session/reset', methods=['POST'])
def reset_session():
    """Reset session PNL tracking."""
    session_pnl.reset()
    print("[Session] PNL reset")
    return jsonify({'ok': True})


# ── Script Serving (auto-update for Tampermonkey) ──────────────────────

SCRIPT_PATH = Path(__file__).parent / 'swisslos_jass_collector.user.js'

@app.route('/script', methods=['GET'])
def serve_script():
    """Serve the latest userscript for Tampermonkey auto-update."""
    if SCRIPT_PATH.exists():
        with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'application/javascript; charset=utf-8'}
    return 'Script not found', 404


# ── Main ───────────────────────────────────────────────────────────────

def main():
    global num_worlds_cfg, endgame_depth_cfg

    parser = argparse.ArgumentParser(description='Jass Bot Server')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--worlds', type=int, default=DEFAULT_WORLDS)
    parser.add_argument('--endgame-depth', type=int, default=DEFAULT_ENDGAME_DEPTH)
    parser.add_argument('--mode', choices=['pimc', 'rl'], default='rl', help='rl (V7 league net, 44% win rate) or pimc (500w search)')
    args = parser.parse_args()

    num_worlds_cfg = args.worlds
    endgame_depth_cfg = args.endgame_depth

    if args.mode == 'rl':
        init_rl_mode()

    # Warm up engine
    print(f"Jass Bot Server starting...")
    engine_name = 'V7-RL (PyTorch)' if _RL_NET else ('RL-ONNX' if _RL_SESS else 'PIMC')
    print(f"  Mode: {engine_name}")
    print(f"  PIMC worlds: {args.worlds}")
    print(f"  Endgame depth: {args.endgame_depth}")
    print(f"  Export dir: {EXPORT_DIR}")
    print(f"  URL: http://{args.host}:{args.port}")

    # Test engine availability
    try:
        from engine_c.fast_engine import FastEngine
        fe = FastEngine()
        print(f"  C engine: available ({type(fe).__name__})")
    except Exception as e:
        print(f"  C engine: NOT available ({e})")

    print()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

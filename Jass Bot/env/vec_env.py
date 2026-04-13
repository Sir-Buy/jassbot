"""Vectorized Jass environment backed by C engine.

Runs N parallel games simultaneously. All heavy lifting in C.
Python side just manages the ctypes interface and numpy arrays.

Usage:
    env = VecJassEnv(n_envs=256)
    obs_decl = env.reset()                    # (N, 4, 72) declaration obs
    env.set_declarations(decl_actions)         # (N, 4) int array
    obs_play, legal_masks = env.get_play_obs() # (N, 204), (N, 36)

    # Game loop
    while not all_done:
        actions = agent.select(obs_play, legal_masks)
        obs_play, legal_masks, rewards, dones, current_players = env.step(actions)
"""

from __future__ import annotations

import ctypes
import os
import sys
import numpy as np
from ctypes import c_int, c_uint64, c_float, c_void_p, POINTER

# ── Load C library ──

_lib = None

def _load():
    global _lib
    if _lib is not None:
        return _lib

    lib_dir = os.path.join(os.path.dirname(__file__), "..", "engine_c")
    if sys.platform == "win32":
        # Add UCRT64 DLL directory
        ucrt = r"C:\msys64\ucrt64\bin"
        if os.path.isdir(ucrt):
            os.add_dll_directory(ucrt)
        lib_path = os.path.join(lib_dir, "jass_engine.dll")
    else:
        lib_path = os.path.join(lib_dir, "jass_engine.so")

    _lib = ctypes.CDLL(lib_path)
    _lib.init_tables()
    _setup_sigs(_lib)
    return _lib


def _setup_sigs(lib):
    lib.rl_gamestate_size.restype = c_int
    lib.rl_obs_size_play.restype = c_int
    lib.rl_obs_size_decl.restype = c_int

    lib.rl_batch_init.argtypes = [c_void_p, c_int, c_uint64]
    lib.rl_batch_legal_moves.argtypes = [c_void_p, c_int, POINTER(c_uint64)]
    lib.rl_batch_step.argtypes = [c_void_p, c_int, POINTER(c_int), POINTER(c_int)]
    lib.rl_batch_set_declarations.argtypes = [c_void_p, c_int, POINTER(c_int)]
    lib.rl_batch_get_rewards.argtypes = [c_void_p, c_int, POINTER(c_float)]
    lib.rl_batch_get_obs_play.argtypes = [c_void_p, c_int, POINTER(c_float)]
    lib.rl_batch_get_obs_decl.argtypes = [c_void_p, c_int, c_int, POINTER(c_float)]
    lib.rl_batch_get_current_player.argtypes = [c_void_p, c_int, POINTER(c_int)]
    lib.rl_batch_get_phase.argtypes = [c_void_p, c_int, POINTER(c_int)]
    lib.rl_batch_get_points.argtypes = [c_void_p, c_int, POINTER(c_int)]
    lib.rl_batch_get_declarations.argtypes = [c_void_p, c_int, POINTER(c_int)]
    lib.rl_batch_rank_rewards.argtypes = [POINTER(c_int), c_int, POINTER(c_float)]
    lib.rl_batch_expert_declare.argtypes = [c_void_p, c_int, POINTER(c_int)]
    lib.rl_batch_expert_declare_noisy.argtypes = [c_void_p, c_int, POINTER(c_int), c_int, c_float, c_uint64]
    lib.rl_batch_step_opponents.argtypes = [c_void_p, c_int, POINTER(c_int), c_int]
    lib.rl_batch_step_opponents.restype = c_int
    lib.rl_batch_pick_opponents.argtypes = [c_void_p, c_int, POINTER(c_int), c_int, POINTER(c_int)]
    lib.rl_batch_step_actions.argtypes = [c_void_p, c_int, POINTER(c_int)]

    lib.rl_game_init.argtypes = [c_void_p, c_uint64]
    lib.rl_legal_moves.argtypes = [c_void_p]
    lib.rl_legal_moves.restype = c_uint64
    lib.rl_step_opponent.argtypes = [c_void_p, c_int]
    lib.rl_step_opponent.restype = c_int
    lib.rl_pick_heuristic.argtypes = [c_void_p]
    lib.rl_pick_heuristic.restype = c_int


# ── Bitmask ↔ numpy conversion (vectorized) ──

# Precompute bit positions for fast mask→array
_BIT_POSITIONS = np.arange(36, dtype=np.uint64)

def masks_to_arrays(masks: np.ndarray) -> np.ndarray:
    """Convert (N,) uint64 bitmasks to (N, 36) float32 binary arrays."""
    # Broadcast: (N, 1) >> (36,) → (N, 36), then & 1
    expanded = (masks[:, None].astype(np.uint64) >> _BIT_POSITIONS[None, :]) & np.uint64(1)
    return expanded.astype(np.float32)


class VecJassEnv:
    """Vectorized Jass Differenzler environment.

    Manages N parallel games. All game logic runs in C.
    """

    PHASE_DECLARE = 0
    PHASE_PLAY = 1
    PHASE_DONE = 2

    def __init__(self, n_envs: int = 256, seed: int = 0):
        self.lib = _load()
        self.n = n_envs
        self._base_seed = seed
        self._seed_counter = seed

        # Sizes
        self._gs_size = self.lib.rl_gamestate_size()
        self.obs_play_size = self.lib.rl_obs_size_play()
        self.obs_decl_size = self.lib.rl_obs_size_decl()

        # Allocate game states as contiguous byte buffer
        self._buf = (ctypes.c_byte * (self._gs_size * n_envs))()
        self._ptr = ctypes.cast(self._buf, c_void_p)

        # Pre-allocate numpy output arrays (reused every step)
        self._obs_play = np.zeros((n_envs, self.obs_play_size), dtype=np.float32)
        self._obs_decl = np.zeros((n_envs, self.obs_decl_size), dtype=np.float32)
        self._legal_u64 = np.zeros(n_envs, dtype=np.uint64)
        self._phases = np.zeros(n_envs, dtype=np.int32)
        self._cur_players = np.zeros(n_envs, dtype=np.int32)
        self._rewards = np.zeros((n_envs, 4), dtype=np.float32)
        self._status = np.zeros(n_envs, dtype=np.int32)
        self._points = np.zeros((n_envs, 4), dtype=np.int32)
        self._declarations = np.zeros((n_envs, 4), dtype=np.int32)

    def _ptr_at(self, game_idx: int) -> c_void_p:
        """Pointer to a specific game state."""
        return c_void_p(ctypes.addressof(self._buf) + game_idx * self._gs_size)

    def reset(self) -> np.ndarray:
        """Reset all games. Returns declaration observations (N, 4, obs_decl_size)."""
        self._seed_counter += self.n
        self.lib.rl_batch_init(self._ptr, self.n, c_uint64(self._seed_counter))

        # Get declaration obs for all 4 players
        all_decl = np.zeros((self.n, 4, self.obs_decl_size), dtype=np.float32)
        for p in range(4):
            self.lib.rl_batch_get_obs_decl(
                self._ptr, self.n, p,
                all_decl[:, p, :].ctypes.data_as(POINTER(c_float))
            )
        return all_decl

    def set_declarations(self, declarations: np.ndarray):
        """Set declarations for all games. declarations: (N, 4) int32 array."""
        decls = np.ascontiguousarray(declarations.astype(np.int32).reshape(-1))
        self.lib.rl_batch_set_declarations(
            self._ptr, self.n,
            decls.ctypes.data_as(POINTER(c_int))
        )

    def get_play_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get play observations, legal masks, and current players.

        Returns:
            obs: (N, obs_play_size) float32
            legal: (N, 36) float32 binary
            current_players: (N,) int32
        """
        self.lib.rl_batch_get_obs_play(
            self._ptr, self.n,
            self._obs_play.ctypes.data_as(POINTER(c_float))
        )
        self.lib.rl_batch_legal_moves(
            self._ptr, self.n,
            self._legal_u64.ctypes.data_as(POINTER(c_uint64))
        )
        self.lib.rl_batch_get_current_player(
            self._ptr, self.n,
            self._cur_players.ctypes.data_as(POINTER(c_int))
        )
        legal_arr = masks_to_arrays(self._legal_u64)
        return self._obs_play.copy(), legal_arr, self._cur_players.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step all games with given actions.

        Args:
            actions: (N,) int32 — card ID to play per game

        Returns:
            obs: (N, obs_play_size)
            legal: (N, 36)
            rewards: (N, 4) — only meaningful for done games
            dones: (N,) bool
            current_players: (N,) int32
        """
        acts = np.ascontiguousarray(actions.astype(np.int32))
        self.lib.rl_batch_step(
            self._ptr, self.n,
            acts.ctypes.data_as(POINTER(c_int)),
            self._status.ctypes.data_as(POINTER(c_int))
        )

        # Get new state
        self.lib.rl_batch_get_phase(
            self._ptr, self.n,
            self._phases.ctypes.data_as(POINTER(c_int))
        )
        dones = (self._phases == self.PHASE_DONE)

        # Rewards
        self.lib.rl_batch_get_rewards(
            self._ptr, self.n,
            self._rewards.ctypes.data_as(POINTER(c_float))
        )

        # Obs + legal for next step
        self.lib.rl_batch_get_obs_play(
            self._ptr, self.n,
            self._obs_play.ctypes.data_as(POINTER(c_float))
        )
        self.lib.rl_batch_legal_moves(
            self._ptr, self.n,
            self._legal_u64.ctypes.data_as(POINTER(c_uint64))
        )
        self.lib.rl_batch_get_current_player(
            self._ptr, self.n,
            self._cur_players.ctypes.data_as(POINTER(c_int))
        )
        legal_arr = masks_to_arrays(self._legal_u64)

        return self._obs_play.copy(), legal_arr, self._rewards.copy(), dones, self._cur_players.copy()

    def reset_done_games(self):
        """Reset only the games that are done. Returns indices of reset games."""
        self.lib.rl_batch_get_phase(
            self._ptr, self.n,
            self._phases.ctypes.data_as(POINTER(c_int))
        )
        done_mask = self._phases == self.PHASE_DONE
        done_indices = np.where(done_mask)[0]

        for idx in done_indices:
            self._seed_counter += 1
            self.lib.rl_game_init(self._ptr_at(int(idx)), c_uint64(self._seed_counter))

        return done_indices

    def get_phases(self) -> np.ndarray:
        """Get current phase for all games."""
        self.lib.rl_batch_get_phase(
            self._ptr, self.n,
            self._phases.ctypes.data_as(POINTER(c_int))
        )
        return self._phases.copy()

    def get_points(self) -> np.ndarray:
        """Get current points for all players in all games. Returns (N, 4) int32."""
        self.lib.rl_batch_get_points(
            self._ptr, self.n,
            self._points.ctypes.data_as(POINTER(c_int))
        )
        return self._points.copy()

    def get_declarations(self) -> np.ndarray:
        """Get declarations for all players in all games. Returns (N, 4) int32."""
        self.lib.rl_batch_get_declarations(
            self._ptr, self.n,
            self._declarations.ctypes.data_as(POINTER(c_int))
        )
        return self._declarations.copy()

    def step_opponent(self, game_idx: int, opp_mode: int = 0) -> int:
        """Step a single game with heuristic (0) or random (1) play."""
        return self.lib.rl_step_opponent(self._ptr_at(game_idx), opp_mode)

    def step_opponents_batch(self, opp_modes: np.ndarray, current_players: np.ndarray, learning_seat: int = 0):
        """Auto-play all opponents in games where it's not the learning seat's turn.
        opp_modes: (N,) — 0=heuristic, 1=random per game.
        Returns number of steps taken."""
        count = 0
        phases = self.get_phases()
        for i in range(self.n):
            if phases[i] == self.PHASE_DONE:
                continue
            if current_players[i] != learning_seat:
                self.lib.rl_step_opponent(self._ptr_at(i), int(opp_modes[i]))
                count += 1
        return count

    def expert_declare(self) -> np.ndarray:
        """Compute expert declarations for all players in all games via C engine.
        Returns (N, 4) int32 array."""
        out = np.zeros(self.n * 4, dtype=np.int32)
        self.lib.rl_batch_expert_declare(
            self._ptr, self.n,
            out.ctypes.data_as(POINTER(c_int)),
        )
        return out.reshape(self.n, 4)

    def expert_declare_noisy(self, noise_range: int = 0, scale: float = 1.0,
                              seed: int = 0) -> np.ndarray:
        """Expert declarations with optional noise and scaling. Returns (N, 4) int32."""
        out = np.zeros(self.n * 4, dtype=np.int32)
        self.lib.rl_batch_expert_declare_noisy(
            self._ptr, self.n,
            out.ctypes.data_as(POINTER(c_int)),
            c_int(noise_range), c_float(scale), c_uint64(seed),
        )
        return out.reshape(self.n, 4)

    def step_opponents_c(self, opp_modes: np.ndarray, seat: int) -> int:
        """Step all games where current_player != seat using C opponent modes.
        opp_modes: (N,) int32 — 0=heuristic, 1=random, 2=aggressive, 3=passive, 4=human-sim.
        Returns number of games stepped."""
        modes = np.ascontiguousarray(opp_modes.astype(np.int32))
        return self.lib.rl_batch_step_opponents(
            self._ptr, self.n,
            modes.ctypes.data_as(POINTER(c_int)), seat,
        )

    def pick_opponents_c(self, opp_modes: np.ndarray, seat: int) -> np.ndarray:
        """Pick actions for C opponents without stepping. Returns (N,) int32.
        Games where current_player == seat get -1."""
        modes = np.ascontiguousarray(opp_modes.astype(np.int32))
        actions = np.full(self.n, -1, dtype=np.int32)
        self.lib.rl_batch_pick_opponents(
            self._ptr, self.n,
            modes.ctypes.data_as(POINTER(c_int)), seat,
            actions.ctypes.data_as(POINTER(c_int)),
        )
        return actions

    def step_actions(self, actions: np.ndarray):
        """Step all games where actions[i] >= 0. Skip games with -1."""
        acts = np.ascontiguousarray(actions.astype(np.int32))
        self.lib.rl_batch_step_actions(
            self._ptr, self.n,
            acts.ctypes.data_as(POINTER(c_int)),
        )

    def get_rank_rewards(self, declarations: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Compute rank-based rewards. Returns (N, 4) float32.
        1st=+3, 2nd=+1, 3rd=-1, 4th=-3. Ties share average."""
        deviations = np.abs(declarations - points).astype(np.int32)
        dev_flat = np.ascontiguousarray(deviations.reshape(-1))
        rewards = np.zeros(self.n * 4, dtype=np.float32)
        self.lib.rl_batch_rank_rewards(
            dev_flat.ctypes.data_as(POINTER(c_int)),
            self.n,
            rewards.ctypes.data_as(POINTER(c_float)),
        )
        return rewards.reshape(self.n, 4)

"""Arena: run bot vs bot matches and collect statistics.

Supports parallel execution via multiprocessing for multi-core speedup.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count

from engine.game import play_round, RoundResult


@dataclass
class ArenaStats:
    """Accumulated statistics across multiple rounds."""
    rounds_played: int = 0
    total_deviations: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    total_points: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    perfect_rounds: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    results: list[RoundResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def avg_deviation(self, player: int) -> float:
        if self.rounds_played == 0:
            return 0.0
        return self.total_deviations[player] / self.rounds_played

    def avg_points(self, player: int) -> float:
        if self.rounds_played == 0:
            return 0.0
        return self.total_points[player] / self.rounds_played

    def summary(self, player_names: dict[int, str] | None = None) -> str:
        names = player_names or {i: f"Player {i}" for i in range(4)}
        cores = cpu_count() or 1
        lines = [f"Arena: {self.rounds_played} rounds in {self.elapsed_seconds:.1f}s ({cores} cores)"]
        lines.append("-" * 55)
        for p in range(4):
            lines.append(
                f"  {names[p]:15s}  "
                f"avg_dev={self.avg_deviation(p):5.1f}  "
                f"avg_pts={self.avg_points(p):5.1f}  "
                f"perfect={self.perfect_rounds[p]}"
            )
        return "\n".join(lines)


def run_arena(
    players: dict[int, object],
    num_rounds: int = 100,
    player_names: dict[int, str] | None = None,
    rng: random.Random | None = None,
    verbose: bool = False,
    parallel: bool = False,
    player_config: dict | None = None,
) -> ArenaStats:
    """Run multiple rounds and collect stats.

    Args:
        players: {player_id: Player} for serial mode
        num_rounds: how many rounds to play
        player_names: optional display names
        rng: random source
        verbose: print each round result
        parallel: use multiprocessing for multi-core speedup
        player_config: dict config for PIMCPlayer (parallel mode).
            Keys: declaration, use_target_aware_opponents, use_belief_tracking,
            endgame_depth, num_worlds, ansage_worlds

    Returns:
        ArenaStats with accumulated results
    """
    if parallel and player_config is not None:
        return _run_parallel(player_config, num_rounds, player_names, rng, verbose)
    return _run_serial(players, num_rounds, player_names, rng, verbose)


def _run_serial(players, num_rounds, player_names, rng, verbose):
    r = rng or random.Random()
    stats = ArenaStats()
    start = time.time()

    for i in range(num_rounds):
        result = play_round(players, rng=r)
        _accumulate(stats, result)

        if verbose:
            names = player_names or {j: f"P{j}" for j in range(4)}
            devs = " ".join(f"{names[p]}:{result.deviations[p]:3d}" for p in range(4))
            print(f"Round {i+1:4d}: {devs}  total={result.total_points}")

    stats.elapsed_seconds = time.time() - start
    return stats


def _run_parallel(player_config, num_rounds, player_names, rng, verbose):
    r = rng or random.Random()
    seeds = [r.randint(0, 2**32) for _ in range(num_rounds)]

    n_workers = min(cpu_count() or 1, num_rounds)
    start = time.time()

    args = [(player_config, seed) for seed in seeds]

    with Pool(processes=n_workers) as pool:
        results = pool.map(_play_round_worker, args)

    stats = ArenaStats()
    for i, result in enumerate(results):
        _accumulate(stats, result)

        if verbose:
            names = player_names or {j: f"P{j}" for j in range(4)}
            devs = " ".join(f"{names[p]}:{result.deviations[p]:3d}" for p in range(4))
            print(f"Round {i+1:4d}: {devs}  total={result.total_points}")

    stats.elapsed_seconds = time.time() - start
    return stats


def _play_round_worker(args) -> RoundResult:
    """Top-level worker function for multiprocessing (must be picklable)."""
    player_config, seed = args
    from arena.players import HeuristicPlayer, PIMCPlayer

    rng = random.Random(seed)

    player_type = player_config.get("player_type", "pimc")

    if player_type == "ismcts":
        from arena.players import ISMCTSPlayer
        p0 = ISMCTSPlayer(
            player_id=0,
            time_limit_ms=player_config.get("time_limit_ms", 2000),
            exploration=player_config.get("exploration", 0.7),
            use_belief_tracking=player_config.get("use_belief_tracking", True),
            endgame_depth=player_config.get("endgame_depth", 4),
            ansage_worlds=player_config.get("ansage_worlds", 2000),
            rng=random.Random(rng.randint(0, 2**32)),
        )
    else:
        p0 = PIMCPlayer(
            player_id=0,
            num_worlds=player_config.get("num_worlds", 25),
            ansage_worlds=player_config.get("ansage_worlds", 100),
            declaration=player_config.get("declaration", "sweep"),
            use_target_aware_opponents=player_config.get("use_target_aware_opponents", False),
            use_belief_tracking=player_config.get("use_belief_tracking", False),
            endgame_depth=player_config.get("endgame_depth", 0),
            rng=random.Random(rng.randint(0, 2**32)),
        )
        if p0._decl_model is not None:
            if "decl_percentile" in player_config:
                p0._decl_model._percentile = player_config["decl_percentile"]
            if "decl_offset" in player_config:
                p0._decl_model._offset = player_config["decl_offset"]

    players = {
        0: p0,
        1: HeuristicPlayer(player_id=1, rng=random.Random(rng.randint(0, 2**32))),
        2: HeuristicPlayer(player_id=2, rng=random.Random(rng.randint(0, 2**32))),
        3: HeuristicPlayer(player_id=3, rng=random.Random(rng.randint(0, 2**32))),
    }
    return play_round(players, rng=random.Random(seed))


def _accumulate(stats: ArenaStats, result: RoundResult) -> None:
    stats.rounds_played += 1
    stats.results.append(result)
    for p in range(4):
        stats.total_deviations[p] += result.deviations[p]
        stats.total_points[p] += result.points[p]
        if result.deviations[p] == 0:
            stats.perfect_rounds[p] += 1

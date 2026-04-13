"""Entry point for Jass Differenzler bot.

Usage:
    python main.py --gui          Launch Pygame GUI assistant
    python main.py --arena N      Run N rounds of bot vs bot
    python main.py --benchmark    Quick benchmark of all player types
    python main.py --compare      Compare old vs new declaration/opponent models
"""

from __future__ import annotations

import argparse
import random
import sys


def run_gui() -> None:
    from gui.assistant import JassAssistent
    app = JassAssistent()
    app.run()


def run_arena(num_rounds: int, verbose: bool = False) -> None:
    from arena.arena import run_arena as arena_run
    from arena.players import RandomPlayer, HeuristicPlayer, PIMCPlayer

    rng = random.Random(42)

    players = {
        0: PIMCPlayer(player_id=0, num_worlds=25, ansage_worlds=40, rng=random.Random(rng.randint(0, 2**32))),
        1: HeuristicPlayer(player_id=1, rng=random.Random(rng.randint(0, 2**32))),
        2: HeuristicPlayer(player_id=2, rng=random.Random(rng.randint(0, 2**32))),
        3: HeuristicPlayer(player_id=3, rng=random.Random(rng.randint(0, 2**32))),
    }
    names = {0: "PIMC", 1: "Heuristic1", 2: "Heuristic2", 3: "Heuristic3"}

    stats = arena_run(players, num_rounds=num_rounds, player_names=names, rng=rng, verbose=verbose)
    print()
    print(stats.summary(names))


def run_benchmark() -> None:
    from arena.arena import run_arena as arena_run
    from arena.players import RandomPlayer, HeuristicPlayer

    rng = random.Random(42)
    num_rounds = 200

    print("=== Benchmark: 4x Random ===")
    players_r = {i: RandomPlayer(rng=random.Random(rng.randint(0, 2**32))) for i in range(4)}
    names_r = {i: f"Random{i}" for i in range(4)}
    stats_r = arena_run(players_r, num_rounds=num_rounds, player_names=names_r, rng=rng)
    print(stats_r.summary(names_r))

    print()
    print("=== Benchmark: 4x Heuristic ===")
    players_h = {i: HeuristicPlayer(player_id=i, rng=random.Random(rng.randint(0, 2**32))) for i in range(4)}
    names_h = {i: f"Heuristic{i}" for i in range(4)}
    stats_h = arena_run(players_h, num_rounds=num_rounds, player_names=names_h, rng=rng)
    print(stats_h.summary(names_h))

    print()
    print("=== Benchmark: PIMC vs 3x Heuristic (50 rounds) ===")
    from arena.players import PIMCPlayer
    players_p = {
        0: PIMCPlayer(player_id=0, rng=random.Random(rng.randint(0, 2**32))),
        1: HeuristicPlayer(player_id=1, rng=random.Random(rng.randint(0, 2**32))),
        2: HeuristicPlayer(player_id=2, rng=random.Random(rng.randint(0, 2**32))),
        3: HeuristicPlayer(player_id=3, rng=random.Random(rng.randint(0, 2**32))),
    }
    names_p = {0: "PIMC", 1: "Heuristic1", 2: "Heuristic2", 3: "Heuristic3"}
    stats_p = arena_run(players_p, num_rounds=50, player_names=names_p, rng=rng)
    print(stats_p.summary(names_p))


def run_compare(num_rounds: int = 200) -> None:
    """Compare configurations with parallel multi-core execution."""
    from arena.arena import run_arena as arena_run
    from multiprocessing import cpu_count

    cores = cpu_count() or 1
    print(f"Running comparisons on {cores} CPU cores (parallel)")

    # Check C engine availability
    c_available = False
    try:
        from engine_c.fast_engine import is_available
        c_available = is_available()
    except (ImportError, OSError):
        pass

    configs = [
        {
            "name": "Sweep + Greedy (Phase 1)",
            "declaration": "sweep",
            "target_aware": False,
            "beliefs": False,
            "endgame": 0,
            "num_worlds": 25,
            "ansage_worlds": 40,
        },
        {
            "name": "Median + Target-Aware (Phase 2)",
            "declaration": "median",
            "target_aware": True,
            "beliefs": False,
            "endgame": 0,
            "num_worlds": 25,
            "ansage_worlds": 100,
        },
        {
            "name": "Full Phase 3 (Python)",
            "declaration": "median",
            "target_aware": True,
            "beliefs": True,
            "endgame": 3,
            "num_worlds": 25,
            "ansage_worlds": 100,
        },
    ]

    if c_available:
        configs.append({
            "name": "C Engine (500w, eg4)",
            "declaration": "median",
            "target_aware": True,
            "beliefs": True,
            "endgame": 4,
            "num_worlds": 500,
            "ansage_worlds": 2000,
        })
        configs.append({
            "name": "C Engine + Calibrated",
            "declaration": "median",
            "target_aware": True,
            "beliefs": True,
            "endgame": 4,
            "num_worlds": 500,
            "ansage_worlds": 2000,
            "decl_percentile": 48,
            "decl_offset": -1,
        })

    if c_available:
        for tl_name, tl_ms in [("1s", 1000), ("2s", 2000)]:
            configs.append({
                "name": f"ISMCTS {tl_name}",
                "player_type": "ismcts",
                "time_limit_ms": tl_ms,
                "beliefs": True,
                "endgame": 4,
                "ansage_worlds": 2000,
            })

    results = []
    names = {0: "PIMC", 1: "Heur1", 2: "Heur2", 3: "Heur3"}

    for cfg in configs:
        print(f"\n=== {cfg['name']} ({num_rounds} rounds, {cores} cores) ===", flush=True)
        rng = random.Random(42)

        if cfg.get("player_type") == "ismcts":
            player_config = {
                "player_type": "ismcts",
                "time_limit_ms": cfg.get("time_limit_ms", 2000),
                "use_belief_tracking": cfg.get("beliefs", True),
                "endgame_depth": cfg.get("endgame", 4),
                "ansage_worlds": cfg.get("ansage_worlds", 2000),
                "exploration": cfg.get("exploration", 0.7),
            }
        else:
            player_config = {
                "declaration": cfg["declaration"],
                "use_target_aware_opponents": cfg["target_aware"],
                "use_belief_tracking": cfg["beliefs"],
                "endgame_depth": cfg["endgame"],
                "num_worlds": cfg.get("num_worlds", 25),
                "ansage_worlds": cfg.get("ansage_worlds", 100),
            }
            if "decl_percentile" in cfg:
                player_config["decl_percentile"] = cfg["decl_percentile"]
            if "decl_offset" in cfg:
                player_config["decl_offset"] = cfg["decl_offset"]

        stats = arena_run(
            players={},
            num_rounds=num_rounds,
            player_names=names,
            rng=rng,
            parallel=True,
            player_config=player_config,
        )
        print(stats.summary(names), flush=True)
        results.append((cfg["name"], stats.avg_deviation(0), stats.perfect_rounds[0], stats.elapsed_seconds))

    # Print comparison table
    print("\n" + "=" * 65)
    print("COMPARISON TABLE")
    print("=" * 65)
    print(f"{'Configuration':<35} {'Avg Dev':>8} {'Perfect':>8} {'Time':>8}")
    print("-" * 65)
    for name, avg_dev, perfect, elapsed in results:
        print(f"{name:<35} {avg_dev:>8.1f} {perfect:>8d} {elapsed:>7.1f}s")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jass Differenzler Bot")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gui", action="store_true", help="Launch Pygame GUI")
    group.add_argument("--arena", type=int, metavar="N", help="Run N arena rounds")
    group.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    group.add_argument("--compare", nargs="?", type=int, const=200, metavar="N",
                       help="Compare old vs new models (default 200 rounds)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.gui:
        run_gui()
    elif args.arena is not None:
        run_arena(args.arena, verbose=args.verbose)
    elif args.benchmark:
        run_benchmark()
    elif args.compare is not None:
        run_compare(args.compare)


if __name__ == "__main__":
    main()

"""Evaluate RL bot against different opponent types in the arena.

Tests:
1. RL vs 3 Heuristic players
2. RL vs 3 Random players
3. RL vs 1 PIMC + 2 Heuristic
4. 4x RL self-play (sanity check)

Usage:
    python -m arena.eval_rl --checkpoint checkpoints_league/latest.pt --rounds 200
"""

import argparse
import os
import sys
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.game import play_round
from arena.players import RandomPlayer, HeuristicPlayer, PIMCPlayer
from arena.rl_player import RLPlayer
from arena.arena import ArenaStats, _accumulate


def run_eval(players, names, num_rounds, verbose=False):
    stats = ArenaStats()
    rng = random.Random(42)
    t0 = time.time()

    for i in range(num_rounds):
        result = play_round(players, rng=rng)
        _accumulate(stats, result)

        # Pass declarations to RL players
        for pid, p in players.items():
            if hasattr(p, 'notify_declarations'):
                decls = {j: result.declarations[j] for j in range(4)}
                p.notify_declarations(decls)

        if verbose and (i + 1) % 50 == 0:
            print(f"  Round {i+1}/{num_rounds}...")

    stats.elapsed_seconds = time.time() - t0
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_league/latest.pt")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--pimc-worlds", type=int, default=500)
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Rounds per test: {args.rounds}")
    print("=" * 70)

    # ── Test 1: RL vs 3 Heuristic ──
    print("\n[1] RL (seat 0) vs 3 Heuristic players")
    rl = RLPlayer(player_id=0, checkpoint_path=args.checkpoint)
    players = {
        0: rl,
        1: HeuristicPlayer(player_id=1, rng=random.Random(1)),
        2: HeuristicPlayer(player_id=2, rng=random.Random(2)),
        3: HeuristicPlayer(player_id=3, rng=random.Random(3)),
    }
    stats = run_eval(players, {0: "RL Bot", 1: "Heur1", 2: "Heur2", 3: "Heur3"}, args.rounds, verbose=True)
    print(stats.summary({0: "RL Bot", 1: "Heuristic 1", 2: "Heuristic 2", 3: "Heuristic 3"}))

    # ── Test 2: RL vs 3 Random ──
    print("\n[2] RL (seat 0) vs 3 Random players")
    rl2 = RLPlayer(player_id=0, checkpoint_path=args.checkpoint)
    players2 = {
        0: rl2,
        1: RandomPlayer(rng=random.Random(1)),
        2: RandomPlayer(rng=random.Random(2)),
        3: RandomPlayer(rng=random.Random(3)),
    }
    stats2 = run_eval(players2, {0: "RL Bot", 1: "Rand1", 2: "Rand2", 3: "Rand3"}, args.rounds, verbose=True)
    print(stats2.summary({0: "RL Bot", 1: "Random 1", 2: "Random 2", 3: "Random 3"}))

    # ── Test 3: RL vs PIMC + 2 Heuristic ──
    print(f"\n[3] RL vs PIMC ({args.pimc_worlds}w) + 2 Heuristic")
    rl3 = RLPlayer(player_id=0, checkpoint_path=args.checkpoint)
    pimc = PIMCPlayer(
        player_id=1,
        num_worlds=args.pimc_worlds,
        ansage_worlds=2000,
        declaration="median",
        use_target_aware_opponents=True,
        use_belief_tracking=True,
        endgame_depth=4,
        rng=random.Random(1),
    )
    players3 = {
        0: rl3,
        1: pimc,
        2: HeuristicPlayer(player_id=2, rng=random.Random(2)),
        3: HeuristicPlayer(player_id=3, rng=random.Random(3)),
    }
    stats3 = run_eval(players3, {0: "RL Bot", 1: "PIMC", 2: "Heur1", 3: "Heur2"}, args.rounds, verbose=True)
    print(stats3.summary({0: "RL Bot", 1: f"PIMC ({args.pimc_worlds}w)", 2: "Heuristic 1", 3: "Heuristic 2"}))

    # ── Test 4: 4x RL self-play ──
    print("\n[4] 4x RL self-play")
    players4 = {
        i: RLPlayer(player_id=i, checkpoint_path=args.checkpoint)
        for i in range(4)
    }
    stats4 = run_eval(players4, {i: f"RL-{i}" for i in range(4)}, args.rounds, verbose=True)
    print(stats4.summary({i: f"RL Player {i}" for i in range(4)}))

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  RL vs Heuristic:   RL={stats.avg_deviation(0):.1f}  Heur={stats.avg_deviation(1):.1f}")
    print(f"  RL vs Random:      RL={stats2.avg_deviation(0):.1f}  Rand={stats2.avg_deviation(1):.1f}")
    print(f"  RL vs PIMC:        RL={stats3.avg_deviation(0):.1f}  PIMC={stats3.avg_deviation(1):.1f}")
    print(f"  RL self-play:      avg={sum(stats4.avg_deviation(i) for i in range(4))/4:.1f}")
    print(f"\n  RL perfect rounds: {stats.perfect_rounds[0]}/{args.rounds} vs heur, "
          f"{stats2.perfect_rounds[0]}/{args.rounds} vs rand, "
          f"{stats3.perfect_rounds[0]}/{args.rounds} vs PIMC")


if __name__ == "__main__":
    main()

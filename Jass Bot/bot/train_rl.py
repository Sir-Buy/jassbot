"""RL Self-Play training for Jass Differenzler.

4 copies of the same network play each other. PPO updates the play policy.
Declaration net trained supervised on (hand_features, actual_score) from self-play.

Optimized: all trajectory storage in pre-allocated tensors — zero Python per-element loops.

Usage:
    python -m bot.train_rl --n-envs 4096 --epochs 10000
    python -m bot.train_rl --resume checkpoints_v2/latest.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.vec_env import VecJassEnv
from bot.rl_network import PlayNet, DeclNet
from bot.ppo import EpisodeBuffer, PPOTrainer


def cosine_schedule(epoch: int, total: int, start: float, end: float) -> float:
    progress = min(epoch / max(total - 1, 1), 1.0)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))


def collect_rollouts_fast(
    env: VecJassEnv,
    play_net: PlayNet,
    decl_net: DeclNet,
    device: torch.device,
    reward_mode: str = "dev",
) -> tuple[EpisodeBuffer, dict]:
    """Play one batch of games. All storage in pre-allocated tensors."""
    n = env.n

    play_net.eval()
    decl_net.eval()

    # ── Declaration phase ──
    decl_obs = env.reset()  # (N, 4, 72) numpy
    declarations = np.zeros((n, 4), dtype=np.int32)

    # Batch all 4 players at once: (N*4, 72)
    decl_flat = torch.tensor(
        decl_obs.reshape(n * 4, -1), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        pred = decl_net(decl_flat)  # (N*4,)
    declarations = pred.cpu().numpy().reshape(n, 4).round().clip(0, 157).astype(np.int32)
    env.set_declarations(declarations)

    # ── Pre-allocate trajectory storage ──
    # Max 9 steps per player per game. We store in (N, 4, 9, ...) tensors.
    obs_size = env.obs_play_size  # 204
    traj_obs = torch.zeros(n, 4, 9, obs_size, device=device)
    traj_legal = torch.zeros(n, 4, 9, 36, device=device)
    traj_actions = torch.zeros(n, 4, 9, dtype=torch.long, device=device)
    traj_log_probs = torch.zeros(n, 4, 9, device=device)
    traj_values = torch.zeros(n, 4, 9, device=device)
    step_counts = torch.zeros(n, 4, dtype=torch.long)  # how many steps each player took

    # ── Play phase ──
    obs, legal, cur_players = env.get_play_obs()

    for step in range(36):
        phases = env.get_phases()
        if (phases == 2).all():
            break

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        legal_t = torch.tensor(legal, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits, values = play_net(obs_t, legal_t)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        actions_np = actions.cpu().numpy().astype(np.int32)
        cur_np = cur_players  # (N,) int32

        # Vectorized storage using advanced indexing — no Python per-element loop
        active_mask = phases != 2
        active_idx = np.where(active_mask)[0]
        if len(active_idx) > 0:
            ai = torch.from_numpy(active_idx).long()
            pi = torch.from_numpy(cur_np[active_idx]).long()
            si = step_counts[ai, pi]
            valid = si < 9
            if valid.any():
                ai = ai[valid]
                pi = pi[valid]
                si = si[valid]
                traj_obs[ai, pi, si] = obs_t[ai]
                traj_legal[ai, pi, si] = legal_t[ai]
                traj_actions[ai, pi, si] = actions[ai]
                traj_log_probs[ai, pi, si] = log_probs[ai]
                traj_values[ai, pi, si] = values[ai]
                step_counts[ai, pi] += 1

        actions_np[~active_mask] = 0
        obs, legal, _, dones, cur_players = env.step(actions_np)

    # ── Build episodes from tensors ──
    points = env.get_points()  # (N, 4) numpy
    buffer = EpisodeBuffer()
    all_devs = []

    # Compute rewards based on mode
    if reward_mode == "rank":
        reward_matrix = env.get_rank_rewards(declarations, points)  # (N, 4)
    else:
        # Default: -|decl - actual|
        reward_matrix = -np.abs(declarations - points).astype(np.float32)

    # Convert to CPU for GAE computation
    traj_obs_cpu = traj_obs.cpu()
    traj_legal_cpu = traj_legal.cpu()
    traj_actions_cpu = traj_actions.cpu()
    traj_log_probs_cpu = traj_log_probs.cpu()
    traj_values_cpu = traj_values.cpu()

    for i in range(n):
        for p in range(4):
            sc = step_counts[i, p].item()
            if sc == 0:
                continue
            dev = abs(int(declarations[i, p]) - int(points[i, p]))
            all_devs.append(dev)

            buffer.episodes.append({
                "obs": traj_obs_cpu[i, p, :sc],
                "legal": traj_legal_cpu[i, p, :sc],
                "actions": traj_actions_cpu[i, p, :sc],
                "log_probs": traj_log_probs_cpu[i, p, :sc],
                "values": traj_values_cpu[i, p, :sc],
                "reward": float(reward_matrix[i, p]),
            })

    # Declaration training data — bulk
    decl_feat_t = torch.tensor(decl_obs.reshape(n * 4, -1), dtype=torch.float32)
    decl_score_t = torch.tensor(points.reshape(-1).astype(np.float32))
    buffer.decl_features.append(decl_feat_t)
    buffer.decl_targets.append(decl_score_t)

    perfect = sum(1 for d in all_devs if d == 0)
    stats = {
        "games": n,
        "mean_dev": np.mean(all_devs) if all_devs else 0,
        "median_dev": int(np.median(all_devs)) if all_devs else 0,
        "perfect_pct": 100 * perfect / max(len(all_devs), 1),
    }
    return buffer, stats


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    play_net = PlayNet(obs_size=204, hidden=args.hidden).to(device)
    decl_net = DeclNet(obs_size=72, hidden=256).to(device)

    pp = sum(p.numel() for p in play_net.parameters())
    dp = sum(p.numel() for p in decl_net.parameters())
    print(f"PlayNet: {pp:,} params | DeclNet: {dp:,} params")

    start_epoch = 0
    best_dev = float("inf")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        play_net.load_state_dict(ckpt["play_net"])
        decl_net.load_state_dict(ckpt["decl_net"])
        start_epoch = ckpt.get("epoch", 0)
        best_dev = ckpt.get("best_dev", float("inf"))
        print(f"Resumed from {args.resume} (epoch {start_epoch}, best={best_dev:.1f})")

    trainer = PPOTrainer(
        play_net, decl_net, device,
        lr=args.lr_start, decl_lr=args.decl_lr,
        clip_eps=0.2, entropy_coef=args.ent_start,
        ppo_epochs=args.ppo_epochs, batch_size=args.batch_size,
        kl_limit=args.kl_limit,
    )

    env = VecJassEnv(n_envs=args.n_envs, seed=42)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    total_epochs = args.epochs
    end_epoch = start_epoch + total_epochs

    print(f"\n{'='*90}")
    print(f"Training: {total_epochs} epochs x {args.n_envs} games = ~{total_epochs * args.n_envs:,} games")
    print(f"LR: {args.lr_start} -> {args.lr_end} (cosine) | Entropy: {args.ent_start} -> {args.ent_end} (cosine)")
    print(f"PPO: epochs={args.ppo_epochs} bs={args.batch_size} kl_limit={args.kl_limit} reward={args.reward_mode}")
    print(f"{'='*90}")

    for epoch in range(start_epoch, end_epoch):
        t0 = time.perf_counter()

        # Schedules
        progress_epoch = epoch - start_epoch
        cur_lr = cosine_schedule(progress_epoch, total_epochs, args.lr_start, args.lr_end)
        cur_ent = cosine_schedule(progress_epoch, total_epochs, args.ent_start, args.ent_end)
        cur_decl_lr = cosine_schedule(progress_epoch, total_epochs, args.decl_lr, args.decl_lr * 0.1)

        trainer.set_lr(cur_lr, cur_decl_lr)
        trainer.entropy_coef = cur_ent

        # Collect + update
        buffer, stats = collect_rollouts_fast(env, play_net, decl_net, device, args.reward_mode)
        t_collect = time.perf_counter() - t0

        play_m = trainer.update_play(buffer)
        decl_m = trainer.update_decl(buffer)
        t_total = time.perf_counter() - t0

        gps = stats["games"] / t_total
        dm = f"dmae={decl_m['decl_mae']:.1f}" if decl_m else ""
        ppo_used = play_m.get("ppo_epochs_used", args.ppo_epochs)

        print(f"E{epoch:>5d} | dev={stats['mean_dev']:.1f} med={stats['median_dev']} "
              f"perf={stats['perfect_pct']:.1f}% | "
              f"pl={play_m['policy_loss']:+.4f} vl={play_m['value_loss']:.1f} "
              f"ent={play_m['entropy']:.2f} kl={play_m['approx_kl']:.4f} "
              f"pe={ppo_used} | {dm} | "
              f"lr={cur_lr:.1e} c={t_collect:.1f}s {gps:.0f}g/s {t_total:.1f}s")

        if (epoch + 1) % args.save_every == 0 or epoch == end_epoch - 1:
            sd = {
                "epoch": epoch + 1, "play_net": play_net.state_dict(),
                "decl_net": decl_net.state_dict(), "best_dev": best_dev,
                "stats": stats, "args": vars(args),
            }
            torch.save(sd, ckpt_dir / "latest.pt")
            if stats["mean_dev"] < best_dev:
                best_dev = stats["mean_dev"]
                sd["best_dev"] = best_dev
                torch.save(sd, ckpt_dir / "best.pt")
                print(f"  >>> New best: {best_dev:.2f}")

        buffer.clear()

    print(f"\nDone. Best: {best_dev:.2f} avg dev over {total_epochs} epochs")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr-start", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--decl-lr", type=float, default=1e-3)
    p.add_argument("--ent-start", type=float, default=0.01)
    p.add_argument("--ent-end", type=float, default=0.001)
    p.add_argument("--kl-limit", type=float, default=0.015)
    p.add_argument("--ppo-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_v2")
    p.add_argument("--reward-mode", type=str, default="dev", choices=["dev", "rank"])
    p.add_argument("--resume", type=str, default=None)
    train(p.parse_args())

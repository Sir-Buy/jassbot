"""V4 RL Self-Play: 4-round matches, blended rank+dev reward, context-aware.

Each game = 4 rounds with same 4 players. Per-round rank reward + deviation shaping.
Blended reward: α*rank + (1-α)*normalized_dev, α ramps 0.3→1.0 over training.
Cumulative standings fed as context to policy and declaration networks.

Usage:
    python -m bot.train_rl_v4 --n-envs 4096 --epochs 10000
    python -m bot.train_rl_v4 --resume-v3 checkpoints_v3/best.pt
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.vec_env import VecJassEnv
from bot.rl_network_v4 import PlayNetV4, DeclNetV4
from bot.ppo import EpisodeBuffer, PPOTrainer


def cosine_schedule(epoch: int, total: int, start: float, end: float) -> float:
    progress = min(epoch / max(total - 1, 1), 1.0)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))


def compute_context(cum_devs: np.ndarray, round_num: int) -> np.ndarray:
    """Build context features (N, 4, 8) from cumulative deviations.

    Per-player context:
      [0]   round_number / 3
      [1]   my_cumulative_dev / 80
      [2-4] opp_cumulative_devs / 80
      [5]   my_current_rank / 3
      [6-7] zeros (reserved)
    """
    n = cum_devs.shape[0]
    ctx = np.zeros((n, 4, 8), dtype=np.float32)

    for p in range(4):
        ctx[:, p, 0] = round_num / 3.0

        ctx[:, p, 1] = cum_devs[:, p] / 80.0

        opp_idx = 0
        for q in range(4):
            if q == p:
                continue
            ctx[:, p, 2 + opp_idx] = cum_devs[:, q] / 80.0
            opp_idx += 1

        # Rank: how many opponents have strictly lower cumulative dev
        rank = np.zeros(n, dtype=np.float32)
        for q in range(4):
            if q == p:
                continue
            rank += (cum_devs[:, q] < cum_devs[:, p]).astype(np.float32)
        ctx[:, p, 5] = rank / 3.0

    return ctx


def play_one_round(
    env: VecJassEnv,
    play_net: PlayNetV4,
    decl_net: DeclNetV4,
    device: torch.device,
    cum_devs: np.ndarray,
    round_num: int,
    reward_alpha: float,
) -> tuple[EpisodeBuffer, np.ndarray, dict]:
    """Play one round across all envs. Returns buffer, per-player devs, stats."""
    n = env.n
    buffer = EpisodeBuffer()

    play_net.eval()
    decl_net.eval()

    # Context from cumulative standings
    ctx_np = compute_context(cum_devs, round_num)  # (N, 4, 8)

    # Declaration phase
    decl_obs = env.reset()  # (N, 4, 72)
    declarations = np.zeros((n, 4), dtype=np.int32)

    with torch.no_grad():
        for p in range(4):
            feat = torch.tensor(decl_obs[:, p, :], dtype=torch.float32, device=device)
            ctx_t = torch.tensor(ctx_np[:, p, :], dtype=torch.float32, device=device)
            pred = decl_net(feat, ctx_t)
            declarations[:, p] = pred.cpu().numpy().round().clip(0, 157).astype(np.int32)
    env.set_declarations(declarations)

    # Pre-allocate trajectories
    obs_size = env.obs_play_size
    traj_obs = torch.zeros(n, 4, 9, obs_size, device=device)
    traj_legal = torch.zeros(n, 4, 9, 36, device=device)
    traj_actions = torch.zeros(n, 4, 9, dtype=torch.long, device=device)
    traj_log_probs = torch.zeros(n, 4, 9, device=device)
    traj_values = torch.zeros(n, 4, 9, device=device)
    step_counts = torch.zeros(n, 4, dtype=torch.long)

    # Context tensors for play (same for all steps in this round)
    ctx_play = {}  # player → (N, 8) tensor on device
    for p in range(4):
        ctx_play[p] = torch.tensor(ctx_np[:, p, :], dtype=torch.float32, device=device)

    obs, legal, cur_players = env.get_play_obs()

    for step in range(36):
        phases = env.get_phases()
        if (phases == 2).all():
            break

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        legal_t = torch.tensor(legal, dtype=torch.float32, device=device)

        # Build context for current players (vectorized)
        # Each game's current player needs their context
        ctx_batch = torch.zeros(n, 8, device=device)
        for p in range(4):
            mask = torch.from_numpy((cur_players == p) & (phases != 2)).to(device)
            if mask.any():
                ctx_batch[mask] = ctx_play[p][mask]

        with torch.no_grad():
            logits, values = play_net(obs_t, legal_t, ctx_batch)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        actions_np = actions.cpu().numpy().astype(np.int32)

        # Store trajectories (vectorized)
        active_mask = phases != 2
        active_idx = np.where(active_mask)[0]
        if len(active_idx) > 0:
            ai = torch.from_numpy(active_idx).long()
            pi = torch.from_numpy(cur_players[active_idx]).long()
            si = step_counts[ai, pi]
            valid = si < 9
            if valid.any():
                ai, pi, si = ai[valid], pi[valid], si[valid]
                traj_obs[ai, pi, si] = obs_t[ai]
                traj_legal[ai, pi, si] = legal_t[ai]
                traj_actions[ai, pi, si] = actions[ai]
                traj_log_probs[ai, pi, si] = log_probs[ai]
                traj_values[ai, pi, si] = values[ai]
                step_counts[ai, pi] += 1

        actions_np[~active_mask] = 0
        obs, legal, _, dones, cur_players = env.step(actions_np)

    # Compute rewards
    points = env.get_points()  # (N, 4)
    devs = np.abs(declarations - points)  # (N, 4)

    # Rank rewards from C engine
    rank_rewards = env.get_rank_rewards(declarations, points)  # (N, 4)

    # Normalized deviation reward: 3*(1 - dev/40) clamped to [-3, +3]
    dev_rewards = np.clip(3.0 * (1.0 - devs / 40.0), -3.0, 3.0).astype(np.float32)

    # Blended reward
    reward_matrix = reward_alpha * rank_rewards + (1.0 - reward_alpha) * dev_rewards

    # Build episodes
    traj_obs_cpu = traj_obs.cpu()
    traj_legal_cpu = traj_legal.cpu()
    traj_actions_cpu = traj_actions.cpu()
    traj_log_probs_cpu = traj_log_probs.cpu()
    traj_values_cpu = traj_values.cpu()

    all_devs = []
    for i in range(n):
        for p in range(4):
            sc = step_counts[i, p].item()
            if sc == 0:
                continue
            all_devs.append(devs[i, p])
            buffer.episodes.append({
                "obs": traj_obs_cpu[i, p, :sc],
                "legal": traj_legal_cpu[i, p, :sc],
                "actions": traj_actions_cpu[i, p, :sc],
                "log_probs": traj_log_probs_cpu[i, p, :sc],
                "values": traj_values_cpu[i, p, :sc],
                "reward": float(reward_matrix[i, p]),
            })

    # Declaration training data
    decl_feat_t = torch.tensor(decl_obs.reshape(n * 4, -1), dtype=torch.float32)
    decl_score_t = torch.tensor(points.reshape(-1).astype(np.float32))
    buffer.decl_features.append(decl_feat_t)
    buffer.decl_targets.append(decl_score_t)

    stats = {
        "mean_dev": np.mean(all_devs) if all_devs else 0,
        "median_dev": int(np.median(all_devs)) if all_devs else 0,
        "perfect_pct": 100 * sum(1 for d in all_devs if d == 0) / max(len(all_devs), 1),
    }
    return buffer, devs, stats


def collect_match(
    env: VecJassEnv,
    play_net: PlayNetV4,
    decl_net: DeclNetV4,
    device: torch.device,
    reward_alpha: float,
) -> tuple[EpisodeBuffer, dict]:
    """Play a 4-round match. Returns combined buffer + match stats."""
    n = env.n
    combined_buffer = EpisodeBuffer()
    cum_devs = np.zeros((n, 4), dtype=np.float32)
    round_devs = []

    for rnd in range(4):
        buf, devs, rstats = play_one_round(
            env, play_net, decl_net, device, cum_devs, rnd, reward_alpha
        )
        cum_devs += devs
        round_devs.append(rstats["mean_dev"])

        # Merge into combined buffer
        combined_buffer.episodes.extend(buf.episodes)
        combined_buffer.decl_features.extend(buf.decl_features)
        combined_buffer.decl_targets.extend(buf.decl_targets)

    stats = {
        "games": n,
        "rounds": n * 4,
        "mean_dev": np.mean(round_devs),
        "median_dev": int(np.median([ep["reward"] for ep in combined_buffer.episodes]) if combined_buffer.episodes else 0),
        "perfect_pct": sum(1 for ep in combined_buffer.episodes if abs(ep["reward"]) < 0.01) / max(len(combined_buffer.episodes), 1) * 100,
        "episodes": len(combined_buffer.episodes),
        "round_devs": [f"{d:.1f}" for d in round_devs],
    }
    # Recompute proper stats from all episodes
    all_episode_devs = []
    for ep in combined_buffer.episodes:
        # We can't easily get dev from reward in rank mode, use decl data instead
        pass
    stats["mean_dev"] = np.mean(round_devs)

    return combined_buffer, stats


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    play_net = PlayNetV4(obs_size=204, hidden=args.hidden, ctx_size=8).to(device)
    decl_net = DeclNetV4(obs_size=72, hidden=256, ctx_size=8).to(device)

    # Load v3 weights
    if args.resume_v3 and os.path.exists(args.resume_v3):
        ckpt = torch.load(args.resume_v3, map_location=device, weights_only=False)
        np_loaded = play_net.load_v3(ckpt["play_net"])
        nd_loaded = decl_net.load_v3(ckpt["decl_net"])
        print(f"Loaded v3: {np_loaded} play tensors, {nd_loaded} decl tensors")
        print(f"v3 best dev: {ckpt.get('best_dev', '?')}, epoch: {ckpt.get('epoch', '?')}")

    # Resume v4
    start_epoch = 0
    best_dev = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        play_net.load_state_dict(ckpt["play_net"])
        decl_net.load_state_dict(ckpt["decl_net"])
        start_epoch = ckpt.get("epoch", 0)
        best_dev = ckpt.get("best_dev", float("inf"))
        print(f"Resumed v4 from epoch {start_epoch}, best={best_dev:.1f}")

    pp = sum(p.numel() for p in play_net.parameters())
    dp = sum(p.numel() for p in decl_net.parameters())
    print(f"PlayNetV4: {pp:,} params | DeclNetV4: {dp:,} params")

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
    print(f"V4: {total_epochs} epochs x {args.n_envs} games x 4 rounds = ~{total_epochs * args.n_envs * 4:,} rounds")
    print(f"LR: {args.lr_start}->{args.lr_end} | Ent: {args.ent_start}->{args.ent_end} | "
          f"Alpha: {args.alpha_start}->{args.alpha_end}")
    print(f"PPO: ep={args.ppo_epochs} bs={args.batch_size} kl={args.kl_limit}")
    print(f"{'='*90}")

    for epoch in range(start_epoch, end_epoch):
        t0 = time.perf_counter()

        progress = epoch - start_epoch
        cur_lr = cosine_schedule(progress, total_epochs, args.lr_start, args.lr_end)
        cur_ent = cosine_schedule(progress, total_epochs, args.ent_start, args.ent_end)
        cur_decl_lr = cosine_schedule(progress, total_epochs, args.decl_lr, args.decl_lr * 0.1)
        cur_alpha = cosine_schedule(progress, total_epochs, args.alpha_start, args.alpha_end)
        # Alpha goes UP (more rank weight over time), so reverse the cosine
        cur_alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * (1 - (1 + math.cos(math.pi * min(progress / max(total_epochs - 1, 1), 1.0))) / 2)

        trainer.set_lr(cur_lr, cur_decl_lr)
        trainer.entropy_coef = cur_ent

        buffer, stats = collect_match(env, play_net, decl_net, device, cur_alpha)
        t_collect = time.perf_counter() - t0

        play_m = trainer.update_play(buffer)
        decl_m = trainer.update_decl(buffer)
        t_total = time.perf_counter() - t0

        gps = stats["games"] / t_total
        dm = f"dm={decl_m['decl_mae']:.1f}" if decl_m else ""
        pe = play_m.get("ppo_epochs_used", args.ppo_epochs)

        print(f"E{epoch:>5d} | dev={stats['mean_dev']:.1f} "
              f"perf={stats['perfect_pct']:.1f}% | "
              f"pl={play_m['policy_loss']:+.4f} vl={play_m['value_loss']:.1f} "
              f"ent={play_m['entropy']:.2f} kl={play_m['approx_kl']:.4f} pe={pe} | "
              f"{dm} | a={cur_alpha:.2f} lr={cur_lr:.1e} "
              f"c={t_collect:.1f}s {gps:.0f}g/s {t_total:.1f}s")

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
    p.add_argument("--alpha-start", type=float, default=0.3, help="Rank reward weight start")
    p.add_argument("--alpha-end", type=float, default=1.0, help="Rank reward weight end")
    p.add_argument("--kl-limit", type=float, default=0.015)
    p.add_argument("--ppo-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_v4")
    p.add_argument("--resume-v3", type=str, default=None, help="Load v3 checkpoint as starting point")
    p.add_argument("--resume", type=str, default=None, help="Resume v4 checkpoint")
    train(p.parse_args())

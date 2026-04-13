"""V5 RL Training — single round, blended rank+dev reward, max throughput.

Loads v3 weights, continues training with:
- Blended reward: α*rank + (1-α)*normalized_dev, α: 0.3→1.0
- Cosine LR/entropy schedules
- KL early stopping
- Maximum GPU/CPU utilization: large batches, pinned memory, torch.compile

Usage:
    python -m bot.train_v5 --resume-v3 checkpoints_v3/v3_final.pt
    python -m bot.train_v5 --resume checkpoints_v5/latest.pt
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
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.vec_env import VecJassEnv


# ── Network (same as v3 PlayNet, keep it simple) ──

class PlayNet(nn.Module):
    def __init__(self, obs_size=204, hidden=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden // 2, 128), nn.ReLU(), nn.Linear(128, 36),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden // 2, 128), nn.ReLU(), nn.Linear(128, 1),
        )

    def forward(self, obs, legal_mask):
        x = self.trunk(obs)
        logits = self.policy_head(x) + (1.0 - legal_mask) * (-1e8)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class DeclNet(nn.Module):
    def __init__(self, obs_size=72, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Schedules ──

def cosine(epoch, total, start, end):
    p = min(epoch / max(total - 1, 1), 1.0)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * p))


def alpha_schedule(epoch, total, start, end):
    """Linear ramp from start to end."""
    p = min(epoch / max(total - 1, 1), 1.0)
    return start + (end - start) * p


# ── GAE (vectorized across fixed-length episodes) ──

def compute_gae_batched(values, rewards, gamma=1.0, lam=0.95):
    """Batched GAE for fixed-length episodes.

    values: (N, T)  — predicted values
    rewards: (N,)   — terminal reward per episode
    Returns advantages (N, T), returns (N, T)
    """
    N, T = values.shape
    advantages = torch.zeros_like(values)
    last_gae = torch.zeros(N, device=values.device)

    for t in reversed(range(T)):
        if t == T - 1:
            delta = rewards - values[:, t]
            next_nt = 0.0
        else:
            delta = gamma * values[:, t + 1] - values[:, t]
            next_nt = 1.0
        last_gae = delta + gamma * lam * next_nt * last_gae
        advantages[:, t] = last_gae

    return advantages, advantages + values


# ── Main training ──

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True

    N = args.n_envs

    play_net = PlayNet(204, args.hidden).to(device)
    decl_net = DeclNet(72, 256).to(device)

    # Load v3
    if args.resume_v3 and os.path.exists(args.resume_v3):
        ckpt = torch.load(args.resume_v3, map_location=device, weights_only=False)
        play_net.load_state_dict(ckpt["play_net"])
        decl_net.load_state_dict(ckpt["decl_net"])
        print(f"Loaded v3: epoch={ckpt.get('epoch')}, best_dev={ckpt.get('best_dev', '?'):.2f}")

    start_epoch = 0
    best_dev = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        play_net.load_state_dict(ckpt["play_net"])
        decl_net.load_state_dict(ckpt["decl_net"])
        start_epoch = ckpt.get("epoch", 0)
        best_dev = ckpt.get("best_dev", float("inf"))
        print(f"Resumed v5: epoch={start_epoch}, best={best_dev:.2f}")

    play_compiled = play_net
    decl_compiled = decl_net
    torch.set_float32_matmul_precision('high')

    pp = sum(p.numel() for p in play_net.parameters())
    dp = sum(p.numel() for p in decl_net.parameters())
    print(f"PlayNet: {pp:,} | DeclNet: {dp:,}")

    play_opt = torch.optim.Adam(play_net.parameters(), lr=args.lr_start)
    decl_opt = torch.optim.Adam(decl_net.parameters(), lr=args.decl_lr)

    env = VecJassEnv(n_envs=N, seed=42)
    obs_size = env.obs_play_size

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    # Log file for monitor
    log_path = ckpt_dir / "training.log"
    log_f = open(log_path, "w", buffering=1)  # line-buffered

    total = args.epochs
    end_epoch = start_epoch + total

    header = (
        f"Training: {total} epochs x {N} games = ~{total * N:,} games\n"
        f"LR: {args.lr_start}->{args.lr_end} | Ent: {args.ent_start}->{args.ent_end} | "
        f"Alpha: {args.alpha_start}->{args.alpha_end}\n"
        f"PPO: ep={args.ppo_epochs} bs={args.batch_size} kl={args.kl_limit}\n"
    )
    print(header)
    log_f.write(header)

    # Pre-allocate GPU tensors for trajectories
    traj_obs = torch.zeros(N, 4, 9, obs_size, device=device)
    traj_legal = torch.zeros(N, 4, 9, 36, device=device)
    traj_actions = torch.zeros(N, 4, 9, dtype=torch.long, device=device)
    traj_lp = torch.zeros(N, 4, 9, device=device)
    traj_val = torch.zeros(N, 4, 9, device=device)
    step_counts = torch.zeros(N, 4, dtype=torch.long)

    for epoch in range(start_epoch, end_epoch):
        t0 = time.perf_counter()
        prog = epoch - start_epoch

        cur_lr = cosine(prog, total, args.lr_start, args.lr_end)
        cur_ent = cosine(prog, total, args.ent_start, args.ent_end)
        cur_alpha = alpha_schedule(prog, total, args.alpha_start, args.alpha_end)
        cur_dlr = cosine(prog, total, args.decl_lr, args.decl_lr * 0.1)

        for pg in play_opt.param_groups: pg["lr"] = cur_lr
        for pg in decl_opt.param_groups: pg["lr"] = cur_dlr

        # ── Collect ──
        traj_obs.zero_(); traj_legal.zero_(); traj_actions.zero_()
        traj_lp.zero_(); traj_val.zero_(); step_counts.zero_()

        play_net.eval(); decl_net.eval()

        decl_obs = env.reset()
        decl_flat = torch.tensor(decl_obs.reshape(N * 4, -1), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = decl_compiled(decl_flat)
        declarations = pred.cpu().numpy().reshape(N, 4).round().clip(0, 157).astype(np.int32)
        env.set_declarations(declarations)

        obs, legal, cur_players = env.get_play_obs()

        for step in range(36):
            phases = env.get_phases()
            if (phases == 2).all():
                break

            obs_t = torch.as_tensor(obs, device=device)
            legal_t = torch.as_tensor(legal, device=device)

            with torch.no_grad():
                logits, values = play_compiled(obs_t, legal_t)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            actions_np = actions.cpu().numpy().astype(np.int32)
            cur_np = cur_players

            # Vectorized trajectory storage
            active = phases != 2
            aidx = np.where(active)[0]
            if len(aidx) > 0:
                ai = torch.from_numpy(aidx).long()
                pi = torch.from_numpy(cur_np[aidx]).long()
                si = step_counts[ai, pi]
                v = si < 9
                if v.any():
                    ai, pi, si = ai[v], pi[v], si[v]
                    traj_obs[ai, pi, si] = obs_t[ai]
                    traj_legal[ai, pi, si] = legal_t[ai]
                    traj_actions[ai, pi, si] = actions[ai]
                    traj_lp[ai, pi, si] = log_probs[ai]
                    traj_val[ai, pi, si] = values[ai]
                    step_counts[ai, pi] += 1

            actions_np[~active] = 0
            obs, legal, _, dones, cur_players = env.step(actions_np)

        t_collect = time.perf_counter() - t0

        # ── Rewards ──
        points = env.get_points()
        devs = np.abs(declarations - points)
        rank_rewards = env.get_rank_rewards(declarations, points)
        dev_rewards = np.clip(3.0 * (1.0 - devs / 40.0), -3.0, 3.0).astype(np.float32)
        reward_matrix = cur_alpha * rank_rewards + (1.0 - cur_alpha) * dev_rewards

        # ── Build flat episode tensors (all on GPU, no Python per-element loop) ──
        # Each player in each game has exactly step_counts[i,p] steps
        # Flatten: (N*4) episodes, each up to 9 steps
        sc = step_counts.reshape(N * 4)  # (N*4,)
        max_sc = 9
        n_eps = N * 4

        # Reshape trajectories: (N, 4, 9, ...) → (N*4, 9, ...)
        flat_obs = traj_obs.reshape(n_eps, max_sc, obs_size)
        flat_legal = traj_legal.reshape(n_eps, max_sc, 36)
        flat_actions = traj_actions.reshape(n_eps, max_sc)
        flat_lp = traj_lp.reshape(n_eps, max_sc)
        flat_val = traj_val.reshape(n_eps, max_sc)
        flat_rewards = torch.tensor(reward_matrix.reshape(-1), dtype=torch.float32, device=device)

        # GAE (vectorized)
        advantages, returns = compute_gae_batched(flat_val, flat_rewards)

        # Create valid-step mask: step < step_counts for each episode
        step_idx = torch.arange(max_sc, device=device).unsqueeze(0)  # (1, 9)
        valid_mask = step_idx < sc.unsqueeze(1).to(device)  # (N*4, 9)
        valid_flat = valid_mask.reshape(-1)  # (N*4*9,)

        # Flatten everything and select valid steps
        all_obs = flat_obs.reshape(-1, obs_size)[valid_flat]
        all_legal = flat_legal.reshape(-1, 36)[valid_flat]
        all_actions = flat_actions.reshape(-1)[valid_flat]
        all_lp = flat_lp.reshape(-1)[valid_flat]
        all_adv = advantages.reshape(-1)[valid_flat]
        all_ret = returns.reshape(-1)[valid_flat]

        total_steps = all_obs.shape[0]

        # Normalize advantages
        if total_steps > 1:
            all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

        # ── PPO update ──
        play_net.train()
        m_pl, m_vl, m_ent, m_kl = 0.0, 0.0, 0.0, 0.0
        n_updates = 0
        ppo_used = 0

        for ppo_ep in range(args.ppo_epochs):
            kl_break = False
            idx = torch.randperm(total_steps, device=device)
            for s in range(0, total_steps, args.batch_size):
                e = min(s + args.batch_size, total_steps)
                bi = idx[s:e]

                logits, vals = play_net(all_obs[bi], all_legal[bi])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(all_actions[bi])
                ent = dist.entropy().mean()

                ratio = torch.exp(new_lp - all_lp[bi])
                s1 = ratio * all_adv[bi]
                s2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * all_adv[bi]
                pl = -torch.min(s1, s2).mean()
                vl = F.mse_loss(vals, all_ret[bi])

                loss = pl + 0.5 * vl - cur_ent * ent
                play_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(play_net.parameters(), 0.5)
                play_opt.step()

                with torch.no_grad():
                    kl = (all_lp[bi] - new_lp).mean().item()

                m_pl += pl.item(); m_vl += vl.item()
                m_ent += ent.item(); m_kl += kl
                n_updates += 1

                if kl > args.kl_limit:
                    kl_break = True
                    break
            ppo_used = ppo_ep + 1
            if kl_break:
                break

        if n_updates > 0:
            m_pl /= n_updates; m_vl /= n_updates
            m_ent /= n_updates; m_kl /= n_updates

        # ── Declaration update ──
        decl_net.train()
        decl_feat = torch.tensor(decl_obs.reshape(N * 4, -1), dtype=torch.float32, device=device)
        decl_tgt = torch.tensor(points.reshape(-1).astype(np.float32), device=device)
        dm_loss = 0.0
        dm_n = 0
        didx = torch.randperm(N * 4, device=device)
        for s in range(0, N * 4, args.batch_size):
            e = min(s + args.batch_size, N * 4)
            bi = didx[s:e]
            pred = decl_net(decl_feat[bi])
            dl = F.l1_loss(pred, decl_tgt[bi])
            decl_opt.zero_grad(); dl.backward()
            nn.utils.clip_grad_norm_(decl_net.parameters(), 0.5)
            decl_opt.step()
            dm_loss += dl.item(); dm_n += 1
        dm = dm_loss / max(dm_n, 1)

        t_total = time.perf_counter() - t0
        gps = N / t_total
        mean_dev = devs.mean()
        med_dev = int(np.median(devs))
        perf = 100 * (devs == 0).sum() / (N * 4)

        line = (
            f"E{epoch:>5d} | dev={mean_dev:.1f} med={med_dev} "
            f"perf={perf:.1f}% | "
            f"pl={m_pl:+.4f} vl={m_vl:.1f} "
            f"ent={m_ent:.2f} kl={m_kl:.4f} pe={ppo_used} | "
            f"dmae={dm:.1f} | a={cur_alpha:.2f} lr={cur_lr:.1e} "
            f"c={t_collect:.1f}s {gps:.0f}g/s {t_total:.1f}s"
        )
        print(line)
        log_f.write(line + "\n")

        if (epoch + 1) % args.save_every == 0 or epoch == end_epoch - 1:
            sd = {
                "epoch": epoch + 1, "play_net": play_net.state_dict(),
                "decl_net": decl_net.state_dict(), "best_dev": best_dev,
            }
            torch.save(sd, ckpt_dir / "latest.pt")
            if mean_dev < best_dev:
                best_dev = mean_dev
                sd["best_dev"] = best_dev
                torch.save(sd, ckpt_dir / "best.pt")
                print(f"  >>> New best: {best_dev:.2f}")
                log_f.write(f"  >>> New best: {best_dev:.2f}\n")

    log_f.close()
    print(f"\nDone. Best: {best_dev:.2f}")


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
    p.add_argument("--alpha-start", type=float, default=0.3)
    p.add_argument("--alpha-end", type=float, default=1.0)
    p.add_argument("--kl-limit", type=float, default=0.015)
    p.add_argument("--ppo-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_v5")
    p.add_argument("--resume-v3", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    train(p.parse_args())

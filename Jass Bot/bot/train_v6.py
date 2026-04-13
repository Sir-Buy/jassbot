"""V6 RL Training — 4-round matches, rank reward, expert declarations, PlayNet only.

Key changes from v5:
- Expert declarations via C engine (no DeclNet)
- 4-round matches: sum deviation across rounds, rank players, reward by rank
- Rank reward: 1st=+3, 2nd=+1, 3rd=-1, 4th=-3
- Fresh PlayNet weights
- Profiling mode for first epoch
- CSV + PNG logging

Usage:
    python -m bot.train_v6                          # fresh start
    python -m bot.train_v6 --resume checkpoints_v6/latest.pt
    python -m bot.train_v6 --profile                # profile 1 epoch then exit
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from ctypes import POINTER, c_int, c_float
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent.parent))
from env.vec_env import VecJassEnv


# ── PlayNet ──

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


# ── Schedules ──

def cosine(epoch, total, start, end):
    p = min(epoch / max(total - 1, 1), 1.0)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * p))


# ── GAE ──

def compute_gae(values, rewards, gamma=1.0, lam=0.95):
    """values: (E, T), rewards: (E,). Returns advantages, returns both (E, T)."""
    E, T = values.shape
    advantages = torch.zeros_like(values)
    last_gae = torch.zeros(E, device=values.device)
    for t in reversed(range(T)):
        if t == T - 1:
            delta = rewards - values[:, t]
            next_val_term = 0.0
        else:
            delta = gamma * values[:, t + 1] - values[:, t]
            next_val_term = 1.0
        last_gae = delta + gamma * lam * next_val_term * last_gae
        advantages[:, t] = last_gae
    return advantages, advantages + values


# ── GPU util ──

def gpu_util_pct() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return -1


# ── Plotting ──

def save_plot(csv_path: str, png_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs, devs, pls, ents = [], [], [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                devs.append(float(row["avg_dev"]))
                pls.append(float(row["policy_loss"]))
                ents.append(float(row["entropy"]))
        if len(epochs) < 2:
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(epochs, devs, "b-", lw=0.8)
        axes[0].set_ylabel("Avg Deviation (per round)")
        axes[0].set_title("V6 Training — 4-Round Rank Reward, Expert Declarations")
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=5.6, color="r", ls="--", alpha=0.5, label="PIMC baseline (5.6)")
        axes[0].legend()
        axes[1].plot(epochs, pls, "g-", lw=0.8)
        axes[1].set_ylabel("Policy Loss")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(epochs, ents, "m-", lw=0.8)
        axes[2].set_ylabel("Entropy")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(png_path, dpi=100)
        plt.close()
    except Exception as e:
        print(f"  [plot error: {e}]")


# ── Collect one round ──

def collect_round(
    env: VecJassEnv,
    play_net: nn.Module,
    device: torch.device,
    N: int,
    obs_size: int,
    # Pre-allocated buffers (avoid re-alloc each round)
    obs_buf: torch.Tensor,    # (N, 4, 9, obs_size) on device
    legal_buf: torch.Tensor,  # (N, 4, 9, 36) on device
    act_buf: torch.Tensor,    # (N, 4, 9) long on device
    lp_buf: torch.Tensor,     # (N, 4, 9) on device
    val_buf: torch.Tensor,    # (N, 4, 9) on device
) -> tuple[np.ndarray, np.ndarray]:
    """Play one round. Fills pre-allocated buffers in-place.
    Returns (declarations, points) both (N, 4) int32."""

    obs_buf.zero_(); legal_buf.zero_(); act_buf.zero_()
    lp_buf.zero_(); val_buf.zero_()
    step_counts = np.zeros((N, 4), dtype=np.int32)

    env.reset()
    declarations = env.expert_declare()
    env.set_declarations(declarations)

    obs, legal, cur_players = env.get_play_obs()

    for step in range(36):
        phases = env.get_phases()
        if (phases == 2).all():
            break

        obs_t = torch.as_tensor(obs, device=device)
        legal_t = torch.as_tensor(legal, device=device)

        with torch.no_grad():
            logits, values = play_net(obs_t, legal_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        actions_np = actions.cpu().numpy().astype(np.int32)

        # Store trajectories
        active = phases != 2
        aidx = np.where(active)[0]
        if len(aidx) > 0:
            pi = cur_players[aidx]  # player index for each active game
            si = step_counts[aidx, pi]  # step index
            valid = si < 9
            if valid.any():
                ai_v = aidx[valid]
                pi_v = pi[valid]
                si_v = si[valid]
                ai_t = torch.from_numpy(ai_v).long()
                pi_t = torch.from_numpy(pi_v).long()
                si_t = torch.from_numpy(si_v).long()
                obs_buf[ai_t, pi_t, si_t] = obs_t[ai_t]
                legal_buf[ai_t, pi_t, si_t] = legal_t[ai_t]
                act_buf[ai_t, pi_t, si_t] = actions[ai_t]
                lp_buf[ai_t, pi_t, si_t] = log_probs[ai_t]
                val_buf[ai_t, pi_t, si_t] = values[ai_t]
                step_counts[ai_v, pi_v] += 1

        actions_np[~active] = 0
        obs, legal, _, dones, cur_players = env.step(actions_np)

    points = env.get_points()
    return declarations, points


# ── Main ──

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    N = args.n_envs
    NR = 4  # rounds per match
    OBS = 204
    STEPS_PER_ROUND = 9
    STEPS_PER_MATCH = STEPS_PER_ROUND * NR  # 36

    play_net = PlayNet(OBS, args.hidden).to(device)

    start_epoch = 0
    best_dev = float("inf")
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        play_net.load_state_dict(ckpt["play_net"])
        start_epoch = ckpt.get("epoch", 0)
        best_dev = ckpt.get("best_dev", float("inf"))
        print(f"Resumed v6: epoch={start_epoch}, best={best_dev:.2f}")

    pp = sum(p.numel() for p in play_net.parameters())
    print(f"PlayNet: {pp:,} params {'(fresh)' if not args.resume else ''}")

    optimizer = torch.optim.Adam(play_net.parameters(), lr=args.lr_start)
    env = VecJassEnv(n_envs=N, seed=42)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    csv_path = ckpt_dir / "training.csv"
    png_path = ckpt_dir / "training.png"

    # Handle resume: append to existing CSV or create new
    csv_new = not (args.resume and csv_path.exists())
    csv_file = open(csv_path, "w" if csv_new else "a", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    if csv_new:
        csv_writer.writerow([
            "epoch", "avg_dev", "median_dev", "perfect_pct",
            "policy_loss", "value_loss", "entropy", "kl",
            "games_per_sec", "gpu_util", "lr", "ent_coeff",
        ])

    log_path = ckpt_dir / "training.log"
    log_f = open(log_path, "w" if csv_new else "a", buffering=1)

    total = args.epochs
    end_epoch = start_epoch + total

    header = (
        f"V6 Training: {total} epochs x {N} matches x {NR} rounds = "
        f"~{total * N * NR:,} games\n"
        f"LR: {args.lr_start}->{args.lr_end} | Ent: {args.ent_start}->{args.ent_end}\n"
        f"PPO: ep={args.ppo_epochs} bs={args.batch_size} kl={args.kl_limit}\n"
        f"Reward: 4-round rank (1st=+3, 2nd=+1, 3rd=-1, 4th=-3)\n"
        f"Declaration: expert formula (C engine)\n"
    )
    print(header)
    log_f.write(header)

    # Pre-allocate per-round buffers (reused every round)
    rnd_obs = [torch.zeros(N, 4, 9, OBS, device=device) for _ in range(NR)]
    rnd_legal = [torch.zeros(N, 4, 9, 36, device=device) for _ in range(NR)]
    rnd_act = [torch.zeros(N, 4, 9, dtype=torch.long, device=device) for _ in range(NR)]
    rnd_lp = [torch.zeros(N, 4, 9, device=device) for _ in range(NR)]
    rnd_val = [torch.zeros(N, 4, 9, device=device) for _ in range(NR)]

    for epoch in range(start_epoch, end_epoch):
        t0 = time.perf_counter()
        prog = epoch - start_epoch

        cur_lr = cosine(prog, total, args.lr_start, args.lr_end)
        cur_ent = cosine(prog, total, args.ent_start, args.ent_end)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        is_profile = args.profile and epoch == start_epoch

        # ═══ COLLECT 4 ROUNDS ═══
        if is_profile:
            t_c = 0.0
            if device.type == "cuda":
                torch.cuda.synchronize()

        play_net.eval()
        total_devs = np.zeros((N, 4), dtype=np.int32)
        all_round_devs = []  # per-round devs for perfect% tracking

        for rnd in range(NR):
            if is_profile:
                tc0 = time.perf_counter()

            decls, pts = collect_round(
                env, play_net, device, N, OBS,
                rnd_obs[rnd], rnd_legal[rnd], rnd_act[rnd], rnd_lp[rnd], rnd_val[rnd],
            )

            if is_profile:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_c += time.perf_counter() - tc0

            round_devs = np.abs(decls - pts)
            total_devs += round_devs
            all_round_devs.append(round_devs)

        t_collect = time.perf_counter() - t0

        # ═══ COMPUTE 4-ROUND RANK REWARDS ═══
        dev_flat = np.ascontiguousarray(total_devs.reshape(-1).astype(np.int32))
        rank_rew_flat = np.zeros(N * 4, dtype=np.float32)
        env.lib.rl_batch_rank_rewards(
            dev_flat.ctypes.data_as(POINTER(c_int)), N,
            rank_rew_flat.ctypes.data_as(POINTER(c_float)),
        )
        match_rewards = torch.tensor(rank_rew_flat, dtype=torch.float32, device=device)  # (N*4,)

        # ═══ ASSEMBLE TRAJECTORIES — vectorized concat, no Python loops ═══
        if is_profile:
            tnp0 = time.perf_counter()

        # Each player plays exactly 9 steps per round → 36 per match
        # Stack rounds: (NR, N, 4, 9, ...) → cat on dim 2 → (N, 4, 36, ...)
        cat_obs = torch.cat([rnd_obs[r] for r in range(NR)], dim=2)      # (N,4,36,OBS)
        cat_legal = torch.cat([rnd_legal[r] for r in range(NR)], dim=2)  # (N,4,36,36)
        cat_act = torch.cat([rnd_act[r] for r in range(NR)], dim=2)      # (N,4,36)
        cat_lp = torch.cat([rnd_lp[r] for r in range(NR)], dim=2)        # (N,4,36)
        cat_val = torch.cat([rnd_val[r] for r in range(NR)], dim=2)      # (N,4,36)

        # Reshape to episodes: (N*4, 36, ...)
        n_eps = N * 4
        flat_obs = cat_obs.reshape(n_eps, STEPS_PER_MATCH, OBS)
        flat_legal = cat_legal.reshape(n_eps, STEPS_PER_MATCH, 36)
        flat_act = cat_act.reshape(n_eps, STEPS_PER_MATCH)
        flat_lp = cat_lp.reshape(n_eps, STEPS_PER_MATCH)
        flat_val = cat_val.reshape(n_eps, STEPS_PER_MATCH)

        # GAE with match-level rewards
        advantages, returns = compute_gae(flat_val, match_rewards)

        # Flatten all steps (all 36 are valid — 9 per round × 4 rounds)
        total_steps = n_eps * STEPS_PER_MATCH
        all_obs_f = flat_obs.reshape(total_steps, OBS)
        all_legal_f = flat_legal.reshape(total_steps, 36)
        all_act_f = flat_act.reshape(total_steps)
        all_lp_f = flat_lp.reshape(total_steps)
        all_adv = advantages.reshape(total_steps)
        all_ret = returns.reshape(total_steps)

        # Normalize advantages
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)

        if is_profile:
            tnp1 = time.perf_counter()
            t_np = tnp1 - tnp0

        # ═══ PPO UPDATE ═══
        if is_profile:
            tfwd0 = time.perf_counter()

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

                logits, vals = play_net(all_obs_f[bi], all_legal_f[bi])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(all_act_f[bi])
                ent = dist.entropy().mean()

                ratio = torch.exp(new_lp - all_lp_f[bi])
                surr1 = ratio * all_adv[bi]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * all_adv[bi]
                pl = -torch.min(surr1, surr2).mean()
                vl = F.mse_loss(vals, all_ret[bi])

                loss = pl + 0.5 * vl - cur_ent * ent
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(play_net.parameters(), 0.5)
                optimizer.step()

                with torch.no_grad():
                    kl = (all_lp_f[bi] - new_lp).mean().item()

                m_pl += pl.item()
                m_vl += vl.item()
                m_ent += ent.item()
                m_kl += kl
                n_updates += 1

                if kl > args.kl_limit:
                    kl_break = True
                    break
            ppo_used = ppo_ep + 1
            if kl_break:
                break

        if is_profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd = time.perf_counter() - tfwd0

        if n_updates > 0:
            m_pl /= n_updates
            m_vl /= n_updates
            m_ent /= n_updates
            m_kl /= n_updates

        # ═══ STATS ═══
        t_total = time.perf_counter() - t0
        gps = N * NR / t_total

        # Per-round deviation stats
        stacked_devs = np.stack(all_round_devs)  # (4, N, 4) = (rounds, matches, players)
        per_round_flat = stacked_devs.reshape(-1)  # all individual round-player deviations
        mean_dev = per_round_flat.mean()
        med_dev = int(np.median(per_round_flat))
        perf_pct = 100.0 * (per_round_flat == 0).sum() / len(per_round_flat)

        gpu_u = gpu_util_pct()

        line = (
            f"E{epoch:>5d} | dev={mean_dev:.1f} med={med_dev} "
            f"perf={perf_pct:.1f}% | "
            f"pl={m_pl:+.4f} vl={m_vl:.1f} "
            f"ent={m_ent:.2f} kl={m_kl:.4f} pe={ppo_used} | "
            f"gpu={gpu_u}% {gps:.0f}g/s {t_total:.1f}s"
        )
        print(line)
        log_f.write(line + "\n")

        csv_writer.writerow([
            epoch, f"{mean_dev:.2f}", med_dev, f"{perf_pct:.1f}",
            f"{m_pl:.6f}", f"{m_vl:.4f}", f"{m_ent:.4f}", f"{m_kl:.6f}",
            f"{gps:.0f}", gpu_u, f"{cur_lr:.2e}", f"{cur_ent:.4f}",
        ])

        # Profile report
        if is_profile:
            print(f"\n{'='*60}")
            print(f"PROFILE (1 epoch, {N} matches x {NR} rounds):")
            print(f"  C engine (collect):  {t_c:.3f}s ({100*t_c/t_total:.0f}%)")
            print(f"  Numpy/torch reshape: {t_np:.3f}s ({100*t_np/t_total:.0f}%)")
            print(f"  GPU (PPO update):    {t_fwd:.3f}s ({100*t_fwd/t_total:.0f}%)")
            print(f"  Other overhead:      {t_total-t_c-t_np-t_fwd:.3f}s")
            print(f"  Total:               {t_total:.3f}s")
            print(f"  Steps/epoch:         {total_steps:,}")
            print(f"  Games/sec:           {gps:.0f}")
            print(f"  GPU util:            {gpu_u}%")
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  GPU mem peak:        {mem:.2f} GB")
            print(f"{'='*60}\n")
            if args.profile:
                csv_file.close()
                log_f.close()
                return

        # Save
        if (epoch + 1) % args.save_every == 0 or epoch == end_epoch - 1:
            sd = {
                "epoch": epoch + 1,
                "play_net": play_net.state_dict(),
                "best_dev": best_dev,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(sd, ckpt_dir / "latest.pt")
            if mean_dev < best_dev:
                best_dev = mean_dev
                sd["best_dev"] = best_dev
                torch.save(sd, ckpt_dir / "best.pt")
                print(f"  >>> New best: {best_dev:.2f}")
                log_f.write(f"  >>> New best: {best_dev:.2f}\n")

        # Plot every 50 epochs
        if (epoch + 1) % 50 == 0:
            csv_file.flush()
            save_plot(str(csv_path), str(png_path))

    csv_file.close()
    log_f.close()
    save_plot(str(csv_path), str(png_path))
    print(f"\nDone. Best avg round dev: {best_dev:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr-start", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--ent-start", type=float, default=0.02)
    p.add_argument("--ent-end", type=float, default=0.005)
    p.add_argument("--kl-limit", type=float, default=0.015)
    p.add_argument("--ppo-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_v6")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--profile", action="store_true")
    train(p.parse_args())

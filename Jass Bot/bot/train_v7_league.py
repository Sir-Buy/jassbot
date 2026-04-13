"""V7 League Training — diverse opponents, single-round, winner-take-all.

Single round per game. Reward: 1st = +3, 2nd/3rd/4th = -1.
Seat 0: training agent. Seats 1-3: opponent pool.
LR warmup with decaying KL limit for stable transition from self-play.

Opponents: self(40%) frozen(15%) heur(15%) rand(5%) human(15%) aggro(5%) pass(5%)
Neural opponents use BATCHED GPU forward passes — zero per-game Python loops.

Usage:
    python -m bot.train_v7_league --resume-v6 checkpoints_v6/best.pt
    python -m bot.train_v7_league --resume checkpoints_v7/latest.pt
    python -m bot.train_v7_league --profile
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from ctypes import POINTER, c_int, c_float, c_void_p
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent.parent))
from env.vec_env import VecJassEnv


# ── PlayNet (same arch as v6) ──

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


# ── Opponent types ──
OPP_SELF = 0
OPP_FROZEN = 1
OPP_HEURISTIC = 2
OPP_RANDOM = 3
OPP_HUMAN_SIM = 4
OPP_AGGRESSIVE = 5
OPP_PASSIVE = 6

OPP_NAMES = ["self", "frozen", "heur", "rand", "human", "aggro", "pass"]
OPP_PROBS = [0.40, 0.15, 0.15, 0.05, 0.15, 0.05, 0.05]

# Map opponent type to C opp_mode for non-neural opponents
# C modes: 0=heuristic, 1=random, 2=aggressive(gap+100), 3=passive(gap-100), 4=human-sim
OPP_TO_C_MODE = {
    OPP_HEURISTIC: 0,
    OPP_RANDOM: 1,
    OPP_HUMAN_SIM: 4,
    OPP_AGGRESSIVE: 2,
    OPP_PASSIVE: 3,
}

NEURAL_OPPS = {OPP_SELF, OPP_FROZEN}


# ── Schedules ──

def cosine(epoch, total, start, end):
    p = min(epoch / max(total - 1, 1), 1.0)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * p))


# ── GAE ──

def compute_gae(values, rewards, gamma=1.0, lam=0.95):
    E, T = values.shape
    advantages = torch.zeros_like(values)
    last_gae = torch.zeros(E, device=values.device)
    for t in reversed(range(T)):
        if t == T - 1:
            delta = rewards - values[:, t]
            nv = 0.0
        else:
            delta = gamma * values[:, t + 1] - values[:, t]
            nv = 1.0
        last_gae = delta + gamma * lam * nv * last_gae
        advantages[:, t] = last_gae
    return advantages, advantages + values


def gpu_util_pct() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        )
        return int(r.stdout.strip().split("\n")[0])
    except Exception:
        return -1


def save_plot(csv_path: str, png_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs, devs, pls, ents, ranks = [], [], [], [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row["epoch"]))
                devs.append(float(row["avg_dev"]))
                pls.append(float(row["policy_loss"]))
                ents.append(float(row["entropy"]))
                ranks.append(float(row["mean_rank"]))
        if len(epochs) < 2:
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(epochs, devs, "b-", lw=1)
        axes[0].set_ylabel("Avg Deviation (seat 0)")
        axes[0].set_title("V7 League Training")
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=5.6, color="r", ls="--", alpha=0.5, label="PIMC (5.6)")
        axes[0].axhline(y=10.2, color="orange", ls="--", alpha=0.5, label="V6 plateau (10.2)")
        axes[0].legend()

        ax1r = axes[1].twinx()
        axes[1].plot(epochs, pls, "g-", lw=1, label="Policy Loss")
        ax1r.plot(epochs, ranks, "b-", lw=1, alpha=0.5, label="Mean Rank")
        axes[1].set_ylabel("Policy Loss")
        ax1r.set_ylabel("Mean Rank Reward")
        axes[1].legend(loc="upper left")
        ax1r.legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, ents, "m-", lw=1)
        axes[2].set_ylabel("Entropy")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(png_path, dpi=100)
        plt.close()
    except Exception as e:
        print(f"  [plot error: {e}]")


# ── Collect one round with league opponents (fully batched, zero Python per-game loops) ──

def collect_league_round(
    env: VecJassEnv,
    play_net: nn.Module,
    frozen_net: nn.Module | None,
    device: torch.device,
    N: int,
    OBS: int,
    SEAT: int,
    opp_types: np.ndarray,   # (N,) opponent type per game
    opp_c_modes: np.ndarray, # (N,) C mode for non-neural opps
    seed_offset: int,
    obs_buf: torch.Tensor,   # (N, 9, OBS)
    legal_buf: torch.Tensor, # (N, 9, 36)
    act_buf: torch.Tensor,   # (N, 9)
    lp_buf: torch.Tensor,    # (N, 9)
    val_buf: torch.Tensor,   # (N, 9)
) -> tuple[np.ndarray, np.ndarray]:
    """Play one round. Zero per-game Python loops."""

    obs_buf.zero_(); legal_buf.zero_(); act_buf.zero_()
    lp_buf.zero_(); val_buf.zero_()
    step_counts = np.zeros(N, dtype=np.int32)

    # ── Declarations ──
    env.reset()
    declarations = env.expert_declare()  # (N, 4) — expert for all

    # Override opponent declarations by type
    human_mask = opp_types == OPP_HUMAN_SIM
    if human_mask.any():
        noisy = env.expert_declare_noisy(noise_range=5, scale=1.0, seed=seed_offset + 1000)
        for p in range(1, 4):
            declarations[human_mask, p] = noisy[human_mask, p]

    aggro_mask = opp_types == OPP_AGGRESSIVE
    if aggro_mask.any():
        aggro = env.expert_declare_noisy(noise_range=0, scale=1.2, seed=0)
        for p in range(1, 4):
            declarations[aggro_mask, p] = aggro[aggro_mask, p]

    pass_mask = opp_types == OPP_PASSIVE
    if pass_mask.any():
        passive = env.expert_declare_noisy(noise_range=0, scale=0.7, seed=0)
        for p in range(1, 4):
            declarations[pass_mask, p] = passive[pass_mask, p]

    rand_mask = opp_types == OPP_RANDOM
    if rand_mask.any():
        rng = np.random.RandomState(seed_offset + 2000)
        for p in range(1, 4):
            declarations[rand_mask, p] = rng.randint(10, 81, size=rand_mask.sum())

    env.set_declarations(declarations)

    # Pre-compute per-game masks (bool arrays, no loops)
    is_self_opp = opp_types == OPP_SELF
    is_frozen_opp = opp_types == OPP_FROZEN
    is_neural_opp = is_self_opp | is_frozen_opp

    # ── Play loop: 36 steps (9 tricks × 4 players) ──
    # Each step: get obs → pick actions for ALL games → step ALL games
    for step in range(36):
        phases = env.get_phases()
        if (phases == 2).all():
            break

        obs_np, legal_np, cur_players = env.get_play_obs()
        active = phases != 2

        # Classify each game by what it needs this step
        is_seat0 = (cur_players == SEAT) & active
        is_opp = (cur_players != SEAT) & active
        needs_playnet = is_seat0 | (is_opp & is_self_opp)    # seat 0 + self opponents
        needs_frozen = is_opp & is_frozen_opp                  # frozen opponents
        needs_c = is_opp & ~is_neural_opp                      # C opponents

        actions_all = np.full(N, -1, dtype=np.int32)

        # ── GPU batch 1: play_net (seat 0 + self opponents) ──
        if needs_playnet.any():
            pn_idx = np.where(needs_playnet)[0]
            obs_t = torch.as_tensor(obs_np[pn_idx], device=device)
            legal_t = torch.as_tensor(legal_np[pn_idx], device=device)

            with torch.no_grad():
                logits, values = play_net(obs_t, legal_t)
                dist = Categorical(logits=logits)
                acts = dist.sample()
                lps = dist.log_prob(acts)

            acts_np = acts.cpu().numpy().astype(np.int32)
            actions_all[pn_idx] = acts_np

            # Store trajectory for seat 0 subset only
            seat0_in_batch = is_seat0[pn_idx]
            if seat0_in_batch.any():
                s0_local = np.where(seat0_in_batch)[0]
                s0_global = pn_idx[s0_local]
                si = step_counts[s0_global]
                valid = si < 9
                if valid.any():
                    s0_l_v = s0_local[valid]
                    s0_g_v = s0_global[valid]
                    si_v = si[valid]
                    ai_t = torch.from_numpy(s0_g_v).long()
                    si_t = torch.from_numpy(si_v).long()
                    obs_buf[ai_t, si_t] = obs_t[s0_l_v]
                    legal_buf[ai_t, si_t] = legal_t[s0_l_v]
                    act_buf[ai_t, si_t] = acts[s0_l_v]
                    lp_buf[ai_t, si_t] = lps[s0_l_v]
                    val_buf[ai_t, si_t] = values[s0_l_v]
                    step_counts[s0_g_v] += 1

        # ── GPU batch 2: frozen_net ──
        if needs_frozen.any() and frozen_net is not None:
            fn_idx = np.where(needs_frozen)[0]
            obs_f = torch.as_tensor(obs_np[fn_idx], device=device)
            legal_f = torch.as_tensor(legal_np[fn_idx], device=device)
            with torch.no_grad():
                logits_f, _ = frozen_net(obs_f, legal_f)
                dist_f = Categorical(logits=logits_f)
                acts_f = dist_f.sample()
            actions_all[fn_idx] = acts_f.cpu().numpy().astype(np.int32)

        # ── C batch: pick actions for C opponents ──
        if needs_c.any():
            c_actions = env.pick_opponents_c(opp_c_modes, SEAT)
            c_mask = needs_c & (c_actions >= 0)
            actions_all[c_mask] = c_actions[c_mask]

        # ── Step ALL games at once ──
        env.step_actions(actions_all)

    points = env.get_points()
    return declarations, points


# ── Main training ──

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    N = args.n_envs
    OBS = 204
    SEAT = 0

    play_net = PlayNet(OBS, args.hidden).to(device)
    frozen_net = PlayNet(OBS, args.hidden).to(device)

    start_epoch = 0
    best_dev = float("inf")

    # Load v6 checkpoint
    if args.resume_v6 and os.path.exists(args.resume_v6):
        ckpt = torch.load(args.resume_v6, map_location=device, weights_only=False)
        play_net.load_state_dict(ckpt["play_net"])
        frozen_net.load_state_dict(ckpt["play_net"])  # freeze v6 as first opponent
        best_dev = ckpt.get("best_dev", float("inf"))
        print(f"Loaded v6: best_dev={best_dev:.2f}")

    # Resume league
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        play_net.load_state_dict(ckpt["play_net"])
        start_epoch = ckpt.get("epoch", 0)
        best_dev = ckpt.get("best_dev", float("inf"))
        if "frozen_net" in ckpt:
            frozen_net.load_state_dict(ckpt["frozen_net"])
        print(f"Resumed v7: epoch={start_epoch}, best={best_dev:.2f}")

    frozen_net.eval()

    # Frozen checkpoint pool (state dicts on CPU)
    frozen_pool = [
        {k: v.cpu().clone() for k, v in frozen_net.state_dict().items()}
    ]

    pp = sum(p.numel() for p in play_net.parameters())
    print(f"PlayNet: {pp:,} params")

    optimizer = torch.optim.Adam(play_net.parameters(), lr=args.lr_start)
    env = VecJassEnv(n_envs=N, seed=42)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)

    csv_path = ckpt_dir / "training.csv"
    png_path = ckpt_dir / "training.png"
    csv_new = not (args.resume and csv_path.exists())
    csv_file = open(csv_path, "w" if csv_new else "a", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    if csv_new:
        csv_writer.writerow([
            "epoch", "avg_dev", "median_dev", "perfect_pct", "mean_rank",
            "policy_loss", "value_loss", "entropy", "kl",
            "games_per_sec", "gpu_util", "lr", "ent_coeff",
        ])

    log_path = ckpt_dir / "training.log"
    log_f = open(log_path, "w" if csv_new else "a", buffering=1)

    total = args.epochs
    end_epoch = start_epoch + total

    header = (
        f"V7 League: {total} epochs x {N} games (single round)\n"
        f"LR: {args.lr_start}->{args.lr_end} | Ent: {args.ent_start}->{args.ent_end}\n"
        f"PPO: ep={args.ppo_epochs} bs={args.batch_size} kl={args.kl_limit}\n"
        f"Opponents: self(40%) frozen(15%) heur(15%) rand(5%) human(15%) aggro(5%) pass(5%)\n"
        f"Reward: 1st=+3, rest=-1 | Seat 0 only | Frozen pool: {len(frozen_pool)}\n"
    )
    print(header)
    log_f.write(header)

    # Pre-allocate trajectory buffer for seat 0 (9 steps per round)
    traj_obs = torch.zeros(N, 9, OBS, device=device)
    traj_legal = torch.zeros(N, 9, 36, device=device)
    traj_act = torch.zeros(N, 9, dtype=torch.long, device=device)
    traj_lp = torch.zeros(N, 9, device=device)
    traj_val = torch.zeros(N, 9, device=device)

    for epoch in range(start_epoch, end_epoch):
        t0 = time.perf_counter()
        prog = epoch - start_epoch
        is_profile = args.profile and epoch == start_epoch

        cur_lr = cosine(prog, total, args.lr_start, args.lr_end)
        cur_ent = cosine(prog, total, args.ent_start, args.ent_end)
        # KL warmup: start high (0.05) for first 200 epochs, decay to target
        kl_warmup = 200
        if prog < kl_warmup:
            cur_kl = args.kl_limit + (0.05 - args.kl_limit) * (1.0 - prog / kl_warmup)
        else:
            cur_kl = args.kl_limit
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # ── Sample opponent types ──
        opp_types = np.random.choice(7, size=N, p=OPP_PROBS).astype(np.int32)
        opp_c_modes = np.zeros(N, dtype=np.int32)
        for ot, cm in OPP_TO_C_MODE.items():
            opp_c_modes[opp_types == ot] = cm

        # Load random frozen checkpoint
        if len(frozen_pool) > 0:
            fidx = np.random.randint(len(frozen_pool))
            frozen_net.load_state_dict(
                {k: v.to(device) for k, v in frozen_pool[fidx].items()}
            )
            frozen_net.eval()

        if is_profile:
            t_c = 0.0
            if device.type == "cuda":
                torch.cuda.synchronize()

        # ═══ COLLECT SINGLE ROUND ═══
        play_net.eval()

        if is_profile:
            tc0 = time.perf_counter()

        decls, pts = collect_league_round(
            env, play_net, frozen_net, device, N, OBS, SEAT,
            opp_types, opp_c_modes, seed_offset=epoch,
            obs_buf=traj_obs, legal_buf=traj_legal,
            act_buf=traj_act, lp_buf=traj_lp, val_buf=traj_val,
        )

        if is_profile:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_c = time.perf_counter() - tc0

        t_collect = time.perf_counter() - t0

        # ═══ WINNER-TAKE-ALL REWARD: 1st=+3, rest=-1 ═══
        devs_all = np.abs(decls - pts)  # (N, 4)
        # Rank: find who has lowest deviation
        rewards = np.full((N, 4), -1.0, dtype=np.float32)
        min_devs = devs_all.min(axis=1, keepdims=True)  # (N, 1)
        winners = devs_all == min_devs  # ties all count as winners
        n_winners = winners.sum(axis=1, keepdims=True).astype(np.float32)  # (N, 1)
        # Winners share the +3 pool, losers get -1
        # If k players tie for 1st: each gets (k*3 + (4-k)*(-1))/k = (4k-4)/k...
        # Simpler: winners get +3, losers get -1. If all tie, all get +3.
        rewards[winners] = 3.0
        seat0_rewards = rewards[:, SEAT]
        rewards_t = torch.tensor(seat0_rewards, dtype=torch.float32, device=device)

        # ═══ ASSEMBLE TRAJECTORIES (single round = 9 steps) ═══
        if is_profile:
            tnp0 = time.perf_counter()

        STEPS = 9
        total_steps = N * STEPS

        # GAE on (N, 9) trajectories
        advantages, returns = compute_gae(traj_val, rewards_t)

        all_obs_f = traj_obs.reshape(total_steps, OBS)
        all_legal_f = traj_legal.reshape(total_steps, 36)
        all_act_f = traj_act.reshape(total_steps)
        all_lp_f = traj_lp.reshape(total_steps)
        all_adv = advantages.reshape(total_steps)
        all_ret = returns.reshape(total_steps)

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

                if kl > cur_kl:
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
        gps = N / t_total

        seat0_devs = devs_all[:, SEAT]
        mean_dev = seat0_devs.mean()
        med_dev = int(np.median(seat0_devs))
        perf_pct = 100.0 * (seat0_devs == 0).sum() / N
        mean_rank = seat0_rewards.mean()
        win_pct = 100.0 * (seat0_rewards > 0).sum() / N

        # Opponent breakdown
        opp_counts = [int((opp_types == i).sum()) for i in range(7)]

        gpu_u = gpu_util_pct()

        line = (
            f"E{epoch:>5d} | dev={mean_dev:.1f} med={med_dev} "
            f"perf={perf_pct:.1f}% win={win_pct:.0f}% rank={mean_rank:+.2f} | "
            f"pl={m_pl:+.4f} vl={m_vl:.1f} "
            f"ent={m_ent:.2f} kl={m_kl:.4f}/{cur_kl:.3f} pe={ppo_used} | "
            f"gpu={gpu_u}% {gps:.0f}g/s {t_total:.1f}s"
        )
        print(line)
        log_f.write(line + "\n")

        csv_writer.writerow([
            epoch, f"{mean_dev:.2f}", med_dev, f"{perf_pct:.1f}", f"{mean_rank:.3f}",
            f"{m_pl:.6f}", f"{m_vl:.4f}", f"{m_ent:.4f}", f"{m_kl:.6f}",
            f"{gps:.0f}", gpu_u, f"{cur_lr:.2e}", f"{cur_ent:.4f}",
        ])

        # Profile
        if is_profile:
            print(f"\n{'='*60}")
            print(f"PROFILE (1 epoch, {N} games, single round):")
            print(f"  Collection (C+GPU): {t_c:.3f}s ({100*t_c/t_total:.0f}%)")
            print(f"  Reshape:            {t_np:.3f}s ({100*t_np/t_total:.0f}%)")
            print(f"  PPO update (GPU):   {t_fwd:.3f}s ({100*t_fwd/t_total:.0f}%)")
            print(f"  Other:              {t_total-t_c-t_np-t_fwd:.3f}s")
            print(f"  Total:              {t_total:.3f}s")
            print(f"  Steps/epoch:        {total_steps:,}")
            print(f"  Games/sec:          {gps:.0f}")
            print(f"  GPU util:           {gpu_u}%")
            print(f"  Opp mix: " + " ".join(f"{OPP_NAMES[i]}={opp_counts[i]}" for i in range(7)))
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated() / 1e9
                print(f"  GPU mem peak:       {mem:.2f} GB")
            print(f"{'='*60}\n")
            if args.profile:
                csv_file.close()
                log_f.close()
                return

        # ── Save frozen checkpoint ──
        if (epoch + 1) % args.snapshot_every == 0:
            snap = {k: v.cpu().clone() for k, v in play_net.state_dict().items()}
            frozen_pool.append(snap)
            if len(frozen_pool) > 10:
                frozen_pool.pop(1)  # keep first (v6) + last 9
            print(f"  [frozen pool: {len(frozen_pool)} checkpoints]")
            log_f.write(f"  [frozen pool: {len(frozen_pool)} checkpoints]\n")

        # ── Save training checkpoint ──
        if (epoch + 1) % args.save_every == 0 or epoch == end_epoch - 1:
            sd = {
                "epoch": epoch + 1,
                "play_net": play_net.state_dict(),
                "frozen_net": frozen_net.state_dict(),
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

        if (epoch + 1) % 50 == 0:
            csv_file.flush()
            save_plot(str(csv_path), str(png_path))

    csv_file.close()
    log_f.close()
    save_plot(str(csv_path), str(png_path))
    print(f"\nDone. Best: {best_dev:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr-start", type=float, default=1e-4)
    p.add_argument("--lr-end", type=float, default=1e-5)
    p.add_argument("--ent-start", type=float, default=0.02)
    p.add_argument("--ent-end", type=float, default=0.01)
    p.add_argument("--kl-limit", type=float, default=0.015)
    p.add_argument("--ppo-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--snapshot-every", type=int, default=500)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_v7")
    p.add_argument("--resume-v6", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--profile", action="store_true")
    train(p.parse_args())

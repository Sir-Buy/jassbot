"""League RL Training — diverse opponents for generalization.

Seat 0: learning agent (neural net, gets gradient updates)
Seats 1-3: sampled from opponent pool each game:
  - self (current weights, no grad)     — 40% of games
  - heuristic (C engine pick_card)      — 30% of games
  - random (random legal card)          — 15% of games
  - past checkpoint                     — 15% of games

Only seat 0 collects trajectories and trains.
Opponents use C engine directly (no GPU) for heuristic/random — faster.

Blended reward: α*rank + (1-α)*normalized_dev, α ramps 0.3→1.0

Usage:
    python -m bot.train_league --resume checkpoints_v5/latest.pt
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
from ctypes import c_void_p, c_int

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.vec_env import VecJassEnv
from bot.train_v5 import PlayNet, DeclNet, cosine, compute_gae_batched


# Opponent types
OPP_SELF = 0       # current neural net (no grad)
OPP_HEURISTIC = 1  # C engine gap-based
OPP_RANDOM = 2     # random legal card
OPP_PAST = 3       # past checkpoint

OPP_NAMES = {0: "self", 1: "heur", 2: "rand", 3: "past"}


def alpha_schedule(epoch, total, start, end):
    p = min(epoch / max(total - 1, 1), 1.0)
    return start + (end - start) * p


def sample_opponent_types(n: int, past_available: bool) -> np.ndarray:
    """Sample opponent type for each game. Returns (N,) array."""
    probs = [0.40, 0.30, 0.15, 0.15] if past_available else [0.50, 0.30, 0.20, 0.0]
    types = np.random.choice(4, size=n, p=probs)
    return types


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    N = args.n_envs
    SEAT = 0  # learning seat

    play_net = PlayNet(204, args.hidden).to(device)
    decl_net = DeclNet(72, 256).to(device)

    # Load checkpoint
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        play_net.load_state_dict(ckpt["play_net"])
        decl_net.load_state_dict(ckpt["decl_net"])
        print(f"Loaded: epoch={ckpt.get('epoch')}, best_dev={ckpt.get('best_dev', '?')}")

    start_epoch = 0
    best_dev = float("inf")
    if args.resume_league and os.path.exists(args.resume_league):
        ckpt = torch.load(args.resume_league, map_location=device, weights_only=False)
        play_net.load_state_dict(ckpt["play_net"])
        decl_net.load_state_dict(ckpt["decl_net"])
        start_epoch = ckpt.get("epoch", 0)
        best_dev = ckpt.get("best_dev", float("inf"))
        print(f"Resumed league: epoch={start_epoch}, best={best_dev:.2f}")

    # Past checkpoint pool (snapshots of earlier weights)
    past_nets = []

    # Save initial as first past checkpoint
    past_state = {k: v.cpu().clone() for k, v in play_net.state_dict().items()}
    past_nets.append(past_state)

    pp = sum(p.numel() for p in play_net.parameters())
    dp = sum(p.numel() for p in decl_net.parameters())
    print(f"PlayNet: {pp:,} | DeclNet: {dp:,}")

    play_opt = torch.optim.Adam(play_net.parameters(), lr=args.lr_start)
    decl_opt = torch.optim.Adam(decl_net.parameters(), lr=args.decl_lr)

    env = VecJassEnv(n_envs=N, seed=42)
    obs_size = env.obs_play_size

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True)
    log_f = open(ckpt_dir / "training.log", "w", buffering=1)

    total = args.epochs
    end_epoch = start_epoch + total

    header = (
        f"League Training: {total} epochs x {N} games = ~{total * N:,} games\n"
        f"LR: {args.lr_start}->{args.lr_end} | Ent: {args.ent_start}->{args.ent_end} | "
        f"Alpha: {args.alpha_start}->{args.alpha_end}\n"
        f"Opponents: self(40%) heuristic(30%) random(15%) past(15%)\n"
        f"PPO: ep={args.ppo_epochs} bs={args.batch_size} kl={args.kl_limit}\n"
    )
    print(header); log_f.write(header)

    # Pre-allocate trajectory tensors for seat 0 only (9 steps per game)
    traj_obs = torch.zeros(N, 9, obs_size, device=device)
    traj_legal = torch.zeros(N, 9, 36, device=device)
    traj_actions = torch.zeros(N, 9, dtype=torch.long, device=device)
    traj_lp = torch.zeros(N, 9, device=device)
    traj_val = torch.zeros(N, 9, device=device)
    step_counts = torch.zeros(N, dtype=torch.long)

    # Past net for inference (loaded on demand)
    past_play_net = PlayNet(204, args.hidden).to(device)

    for epoch in range(start_epoch, end_epoch):
        t0 = time.perf_counter()
        prog = epoch - start_epoch

        cur_lr = cosine(prog, total, args.lr_start, args.lr_end)
        cur_ent = cosine(prog, total, args.ent_start, args.ent_end)
        cur_alpha = alpha_schedule(prog, total, args.alpha_start, args.alpha_end)
        cur_dlr = cosine(prog, total, args.decl_lr, args.decl_lr * 0.1)

        for pg in play_opt.param_groups: pg["lr"] = cur_lr
        for pg in decl_opt.param_groups: pg["lr"] = cur_dlr

        # ── Setup opponents ──
        opp_types = sample_opponent_types(N, len(past_nets) > 1)

        # Load a random past checkpoint for OPP_PAST games
        if len(past_nets) > 0:
            past_idx = np.random.randint(len(past_nets))
            past_play_net.load_state_dict(past_nets[past_idx])
            past_play_net.eval()

        # ── Collect ──
        traj_obs.zero_(); traj_legal.zero_(); traj_actions.zero_()
        traj_lp.zero_(); traj_val.zero_(); step_counts.zero_()

        play_net.eval(); decl_net.eval()

        # Declaration: seat 0 uses neural net, opponents use simple estimate
        decl_obs = env.reset()
        declarations = np.zeros((N, 4), dtype=np.int32)

        # Seat 0: neural declaration
        feat0 = torch.tensor(decl_obs[:, SEAT, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred0 = decl_net(feat0)
        declarations[:, SEAT] = pred0.cpu().numpy().round().clip(0, 157).astype(np.int32)

        # Other seats: use neural for self/past opponents, heuristic estimate for others
        for p in range(4):
            if p == SEAT:
                continue
            # Self and past opponents use same declaration net
            self_mask = (opp_types == OPP_SELF) | (opp_types == OPP_PAST)
            if self_mask.any():
                feat = torch.tensor(decl_obs[self_mask, p, :], dtype=torch.float32, device=device)
                with torch.no_grad():
                    pred = decl_net(feat)
                declarations[self_mask, p] = pred.cpu().numpy().round().clip(0, 157).astype(np.int32)
            # Heuristic/random: declare ~40 (rough median)
            other_mask = ~self_mask
            if other_mask.any():
                # Simple: declare based on hand point sum * 1.5
                hand_pts = decl_obs[other_mask, p, :36].sum(axis=1)  # not real points, just card count
                declarations[other_mask, p] = 40  # simple fixed declaration

        env.set_declarations(declarations)

        # ── Play loop ──
        # For each step: if it's seat 0's turn → neural inference + store trajectory
        #                if it's opponent's turn → C engine plays directly

        obs, legal, cur_players = env.get_play_obs()
        game_done = np.zeros(N, dtype=bool)

        for step in range(100):  # max iterations (36 actions + some overhead)
            phases = env.get_phases()
            game_done = phases == 2
            if game_done.all():
                break

            # Which games have seat 0 playing?
            seat0_turn = (cur_players == SEAT) & ~game_done

            if seat0_turn.any():
                # Neural inference for seat 0
                idx = np.where(seat0_turn)[0]
                obs_t = torch.tensor(obs[idx], dtype=torch.float32, device=device)
                legal_t = torch.tensor(legal[idx], dtype=torch.float32, device=device)

                with torch.no_grad():
                    logits, values = play_net(obs_t, legal_t)
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)

                actions_np = actions.cpu().numpy().astype(np.int32)

                # Store trajectories
                ai = torch.from_numpy(idx).long()
                si = step_counts[ai]
                valid = si < 9
                if valid.any():
                    ai_v = ai[valid]
                    si_v = si[valid]
                    traj_obs[ai_v, si_v] = obs_t[valid.cpu()]
                    traj_legal[ai_v, si_v] = legal_t[valid.cpu()]
                    traj_actions[ai_v, si_v] = actions[valid.cpu()]
                    traj_lp[ai_v, si_v] = log_probs[valid.cpu()]
                    traj_val[ai_v, si_v] = values[valid.cpu()]
                    step_counts[ai_v] += 1

                # Step seat 0 games
                full_actions = np.zeros(N, dtype=np.int32)
                for j, i in enumerate(idx):
                    full_actions[i] = actions_np[j]

                # Only step the seat0 games
                for i in idx:
                    env.lib.rl_step(env._ptr_at(int(i)), c_int(full_actions[i]))

            # Now handle opponent turns
            # Re-check who's playing after seat 0 stepped
            _, _, cur_players = env.get_play_obs()
            cur_players = cur_players
            phases = env.get_phases()

            # Play opponents until it's seat 0's turn again (or game ends)
            for _ in range(3):  # max 3 opponents per trick
                opp_turn = (cur_players != SEAT) & (phases != 2)
                if not opp_turn.any():
                    break

                for i in np.where(opp_turn)[0]:
                    ot = opp_types[i]
                    if ot == OPP_SELF or ot == OPP_PAST:
                        # Neural opponent: need inference
                        # For speed, use heuristic as proxy for self/past in this loop
                        # (full neural opponent is too slow for the inner loop)
                        env.lib.rl_step_opponent(env._ptr_at(int(i)), c_int(0))  # heuristic
                    elif ot == OPP_RANDOM:
                        env.lib.rl_step_opponent(env._ptr_at(int(i)), c_int(1))  # random
                    else:
                        env.lib.rl_step_opponent(env._ptr_at(int(i)), c_int(0))  # heuristic

                # Update state
                _, _, cur_players = env.get_play_obs()
                phases = env.get_phases()

            # Get fresh obs for next seat 0 turn
            obs, legal, cur_players = env.get_play_obs()

        t_collect = time.perf_counter() - t0

        # ── Rewards ──
        points = env.get_points()
        devs_all = np.abs(declarations - points)
        devs_seat0 = devs_all[:, SEAT]

        rank_rewards = env.get_rank_rewards(declarations, points)
        dev_rewards = np.clip(3.0 * (1.0 - devs_seat0 / 40.0), -3.0, 3.0).astype(np.float32)
        rank_seat0 = rank_rewards[:, SEAT]
        blended = cur_alpha * rank_seat0 + (1.0 - cur_alpha) * dev_rewards

        # ── GAE (seat 0 only) ──
        sc = step_counts  # (N,)
        rewards_t = torch.tensor(blended, dtype=torch.float32, device=device)
        advantages, returns = compute_gae_batched(traj_val, rewards_t)

        step_idx = torch.arange(9, device=device).unsqueeze(0)
        valid_mask = step_idx < sc.unsqueeze(1).to(device)
        valid_flat = valid_mask.reshape(-1)

        all_obs = traj_obs.reshape(-1, obs_size)[valid_flat]
        all_legal = traj_legal.reshape(-1, 36)[valid_flat]
        all_actions = traj_actions.reshape(-1)[valid_flat]
        all_lp = traj_lp.reshape(-1)[valid_flat]
        all_adv = advantages.reshape(-1)[valid_flat]
        all_ret = returns.reshape(-1)[valid_flat]

        total_steps = all_obs.shape[0]
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
                s2 = torch.clamp(ratio, 0.8, 1.2) * all_adv[bi]
                pl = -torch.min(s1, s2).mean()
                vl = F.mse_loss(vals, all_ret[bi])
                loss = pl + 0.5 * vl - cur_ent * ent
                play_opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(play_net.parameters(), 0.5)
                play_opt.step()
                with torch.no_grad():
                    kl = (all_lp[bi] - new_lp).mean().item()
                m_pl += pl.item(); m_vl += vl.item()
                m_ent += ent.item(); m_kl += kl
                n_updates += 1
                if kl > args.kl_limit:
                    kl_break = True; break
            ppo_used = ppo_ep + 1
            if kl_break: break

        if n_updates:
            m_pl /= n_updates; m_vl /= n_updates
            m_ent /= n_updates; m_kl /= n_updates

        # ── Declaration update (seat 0 data) ──
        decl_net.train()
        d_feat = torch.tensor(decl_obs[:, SEAT, :], dtype=torch.float32, device=device)
        d_tgt = torch.tensor(points[:, SEAT].astype(np.float32), device=device)
        dp = decl_net(d_feat)
        dl = F.l1_loss(dp, d_tgt)
        decl_opt.zero_grad(); dl.backward()
        nn.utils.clip_grad_norm_(decl_net.parameters(), 0.5)
        decl_opt.step()
        dm = dl.item()

        t_total = time.perf_counter() - t0
        gps = N / t_total
        mean_dev = devs_seat0.mean()
        med_dev = int(np.median(devs_seat0))
        perf = 100 * (devs_seat0 == 0).sum() / N

        # Opponent type counts
        n_self = (opp_types == OPP_SELF).sum()
        n_heur = (opp_types == OPP_HEURISTIC).sum()
        n_rand = (opp_types == OPP_RANDOM).sum()
        n_past = (opp_types == OPP_PAST).sum()

        # Mean rank for seat 0
        mean_rank = rank_seat0.mean()

        line = (
            f"E{epoch:>5d} | dev={mean_dev:.1f} med={med_dev} "
            f"perf={perf:.1f}% rank={mean_rank:+.1f} | "
            f"pl={m_pl:+.4f} vl={m_vl:.1f} "
            f"ent={m_ent:.2f} kl={m_kl:.4f} pe={ppo_used} | "
            f"dmae={dm:.1f} | a={cur_alpha:.2f} lr={cur_lr:.1e} "
            f"opp=s{n_self}h{n_heur}r{n_rand}p{n_past} "
            f"c={t_collect:.1f}s {gps:.0f}g/s {t_total:.1f}s"
        )
        print(line); log_f.write(line + "\n")

        # Save past checkpoint every N epochs
        if (epoch + 1) % args.snapshot_every == 0:
            snap = {k: v.cpu().clone() for k, v in play_net.state_dict().items()}
            past_nets.append(snap)
            if len(past_nets) > 10:
                past_nets.pop(1)  # keep first + last 9

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
    p.add_argument("--n-envs", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr-start", type=float, default=1e-4)
    p.add_argument("--lr-end", type=float, default=1e-5)
    p.add_argument("--decl-lr", type=float, default=5e-4)
    p.add_argument("--ent-start", type=float, default=0.02)
    p.add_argument("--ent-end", type=float, default=0.005)
    p.add_argument("--alpha-start", type=float, default=0.3)
    p.add_argument("--alpha-end", type=float, default=1.0)
    p.add_argument("--kl-limit", type=float, default=0.02)
    p.add_argument("--ppo-epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--snapshot-every", type=int, default=500)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints_league")
    p.add_argument("--resume", type=str, default=None, help="Load initial weights (v5)")
    p.add_argument("--resume-league", type=str, default=None, help="Resume league training")
    train(p.parse_args())

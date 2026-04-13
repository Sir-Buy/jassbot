"""PPO training for Jass Differenzler self-play.

Key insight: each game produces 4 separate 9-step episodes (one per player).
GAE is computed per-player-per-game, then all episodes are merged for PPO update.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class EpisodeBuffer:
    """Stores complete per-player episodes from self-play."""

    def __init__(self):
        self.episodes = []  # list of episode dicts
        self.decl_features = []
        self.decl_targets = []

    def add_episode(self, obs, legal, actions, log_probs, values, reward):
        """Add one player's complete episode (9 steps).

        Args:
            obs: list of (obs_size,) tensors, len=9
            legal: list of (36,) tensors, len=9
            actions: list of scalar tensors, len=9
            log_probs: list of scalar tensors, len=9
            values: list of scalar tensors, len=9
            reward: float, the final reward for this player
        """
        self.episodes.append({
            "obs": torch.stack(obs),
            "legal": torch.stack(legal),
            "actions": torch.stack(actions),
            "log_probs": torch.stack(log_probs),
            "values": torch.stack(values),
            "reward": reward,
        })

    def add_decl_data(self, features: torch.Tensor, targets: torch.Tensor):
        self.decl_features.append(features)
        self.decl_targets.append(targets)

    def clear(self):
        self.__init__()

    def compute_all(self, gamma: float = 1.0, lam: float = 0.95):
        """Compute GAE for all episodes and merge into flat tensors.

        Batched: all episodes padded to same length, GAE computed in parallel.
        """
        n_eps = len(self.episodes)
        if n_eps == 0:
            return {k: torch.zeros(0) for k in
                    ["obs", "legal", "actions", "log_probs", "advantages", "returns"]}

        max_T = max(len(ep["obs"]) for ep in self.episodes)
        obs_dim = self.episodes[0]["obs"].shape[-1]

        # Pad all episodes to max_T
        all_values = torch.zeros(n_eps, max_T)
        all_rewards = torch.zeros(n_eps)  # terminal reward
        lengths = torch.zeros(n_eps, dtype=torch.long)

        flat_obs, flat_legal, flat_actions, flat_lp = [], [], [], []

        for i, ep in enumerate(self.episodes):
            T = len(ep["obs"])
            lengths[i] = T
            all_values[i, :T] = ep["values"].cpu()
            all_rewards[i] = ep["reward"]
            flat_obs.append(ep["obs"].cpu())
            flat_legal.append(ep["legal"].cpu())
            flat_actions.append(ep["actions"].cpu())
            flat_lp.append(ep["log_probs"].cpu())

        # Batched GAE: iterate max_T steps (usually 9), vectorized across episodes
        advantages = torch.zeros(n_eps, max_T)
        last_gae = torch.zeros(n_eps)

        for t in reversed(range(max_T)):
            mask = (t < lengths).float()  # which episodes have this step

            if t == max_T - 1:
                next_val = torch.zeros(n_eps)
            else:
                next_val = all_values[:, t + 1]

            # For terminal steps (t == lengths[i] - 1): delta = reward - V(t)
            is_terminal = (t == lengths - 1).float()
            delta = (is_terminal * (all_rewards - all_values[:, t])
                     + (1 - is_terminal) * (gamma * next_val - all_values[:, t]))

            next_non_terminal = 1.0 - is_terminal
            last_gae = (delta + gamma * lam * next_non_terminal * last_gae) * mask
            advantages[:, t] = last_gae

        returns = advantages + all_values

        # Flatten: extract valid steps only
        flat_adv, flat_ret = [], []
        for i in range(n_eps):
            T = lengths[i].item()
            flat_adv.append(advantages[i, :T])
            flat_ret.append(returns[i, :T])

        return {
            "obs": torch.cat(flat_obs),
            "legal": torch.cat(flat_legal),
            "actions": torch.cat(flat_actions),
            "log_probs": torch.cat(flat_lp),
            "advantages": torch.cat(flat_adv),
            "returns": torch.cat(flat_ret),
        }

    def get_decl_data(self, device):
        if not self.decl_features:
            return None
        return torch.cat(self.decl_features).to(device), torch.cat(self.decl_targets).to(device)


class PPOTrainer:
    """PPO trainer for Jass self-play.

    Supports scheduled entropy and KL early stopping.
    LR scheduling is handled externally via set_lr().
    """

    def __init__(
        self,
        play_net: nn.Module,
        decl_net: nn.Module,
        device: torch.device,
        lr: float = 3e-4,
        decl_lr: float = 1e-3,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 512,
        kl_limit: float = 0.015,
    ):
        self.play_net = play_net
        self.decl_net = decl_net
        self.device = device
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.kl_limit = kl_limit

        self.play_optimizer = torch.optim.Adam(play_net.parameters(), lr=lr)
        self.decl_optimizer = torch.optim.Adam(decl_net.parameters(), lr=decl_lr)

    def set_lr(self, play_lr: float, decl_lr: float | None = None):
        for pg in self.play_optimizer.param_groups:
            pg["lr"] = play_lr
        if decl_lr is not None:
            for pg in self.decl_optimizer.param_groups:
                pg["lr"] = decl_lr

    def update_play(self, buffer: EpisodeBuffer) -> dict:
        """PPO update on play network using per-player episodes."""
        if not buffer.episodes:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "approx_kl": 0}

        data = buffer.compute_all()
        obs = data["obs"].to(self.device)
        legal = data["legal"].to(self.device)
        actions = data["actions"].to(self.device)
        old_log_probs = data["log_probs"].to(self.device)
        advantages = data["advantages"].to(self.device)
        returns = data["returns"].to(self.device)

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_samples = len(obs)
        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "approx_kl": 0}
        n_updates = 0

        self.play_net.train()
        kl_breaked = False
        for ppo_ep in range(self.ppo_epochs):
            if kl_breaked:
                break
            indices = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                idx = indices[start:end]

                b_obs = obs[idx]
                b_legal = legal[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages[idx]
                b_ret = returns[idx]

                logits, values = self.play_net(b_obs, b_legal)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, b_ret)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.play_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.play_net.parameters(), self.max_grad_norm)
                self.play_optimizer.step()

                with torch.no_grad():
                    approx_kl = (b_old_lp - new_log_probs).mean().item()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += approx_kl
                n_updates += 1

                # KL early stopping
                if approx_kl > self.kl_limit:
                    kl_breaked = True
                    break

        for k in metrics:
            metrics[k] /= max(n_updates, 1)
        metrics["ppo_epochs_used"] = ppo_ep + 1
        return metrics

    def update_decl(self, buffer: EpisodeBuffer) -> dict | None:
        """Supervised update on declaration network."""
        data = buffer.get_decl_data(self.device)
        if data is None:
            return None
        features, targets = data
        if len(features) < 32:
            return None

        self.decl_net.train()
        total_loss = 0
        n_batches = 0

        indices = torch.randperm(len(features), device=self.device)
        for start in range(0, len(features), self.batch_size):
            end = min(start + self.batch_size, len(features))
            idx = indices[start:end]

            pred = self.decl_net(features[idx])
            loss = F.l1_loss(pred, targets[idx])

            self.decl_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.decl_net.parameters(), self.max_grad_norm)
            self.decl_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"decl_mae": total_loss / max(n_batches, 1), "decl_samples": len(features)}

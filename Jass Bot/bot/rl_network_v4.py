"""V4 Neural network for RL Jass agent — multi-round context aware.

Extends v3 PlayNet/DeclNet with a side context branch for cumulative standings.
Trunk weights are fully compatible with v3 — load and continue training.

Context features (8 floats):
  [0]   round_number / 3
  [1]   my_cumulative_dev / 80
  [2-4] opp_cumulative_devs / 80  (3 opponents)
  [5]   my_current_rank / 3  (0=1st, 1=4th)
  [6-7] reserved (zeros)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PlayNetV4(nn.Module):
    """Card play policy + value network with multi-round context.

    Trunk (204→512→512→256) is identical to v3 PlayNet.
    Context branch (8→32) merges at the head level.
    """

    def __init__(self, obs_size: int = 204, hidden: int = 512, ctx_size: int = 8):
        super().__init__()
        self.obs_size = obs_size
        self.ctx_size = ctx_size

        # Trunk — identical to v3
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )

        # Context branch
        self.ctx_net = nn.Sequential(
            nn.Linear(ctx_size, 32),
            nn.ReLU(),
        )

        # Heads take trunk (256) + context (32) = 288
        head_in = hidden // 2 + 32
        self.policy_head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )
        self.value_head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor, legal_mask: torch.Tensor,
                ctx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        trunk_out = self.trunk(obs)

        if ctx is not None:
            ctx_out = self.ctx_net(ctx)
            combined = torch.cat([trunk_out, ctx_out], dim=-1)
        else:
            # No context — pad with zeros (backward compatible)
            zeros = torch.zeros(trunk_out.shape[0], 32, device=trunk_out.device)
            combined = torch.cat([trunk_out, zeros], dim=-1)

        logits = self.policy_head(combined)
        logits = logits + (1.0 - legal_mask) * (-1e8)
        value = self.value_head(combined).squeeze(-1)
        return logits, value

    def load_v3(self, v3_state_dict: dict):
        """Load v3 PlayNet weights. Trunk loads directly, heads are zero-padded.

        v3 heads: Linear(256, 128). v4 heads: Linear(288, 128).
        First 256 columns get v3 weights, last 32 (context) stay zero.
        This means the network starts at v3 performance with context ignored.
        """
        own = self.state_dict()
        loaded = 0
        for k, v in v3_state_dict.items():
            if k in own:
                if own[k].shape == v.shape:
                    own[k] = v
                    loaded += 1
                elif own[k].dim() == 2 and v.dim() == 2 and own[k].shape[0] == v.shape[0]:
                    # Zero-pad: copy v3 weights into first columns
                    own[k][:, :v.shape[1]] = v
                    loaded += 1
                elif own[k].dim() == 1 and v.dim() == 1 and own[k].shape == v.shape:
                    own[k] = v
                    loaded += 1
        self.load_state_dict(own)
        return loaded


class DeclNetV4(nn.Module):
    """Declaration network with multi-round context.

    Main trunk (72→256→256→128→1) identical to v3.
    Context branch (8→16) added before final layer.
    """

    def __init__(self, obs_size: int = 72, hidden: int = 256, ctx_size: int = 8):
        super().__init__()

        # Main trunk — layers match v3 DeclNet
        self.layer1 = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(0.1),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
        )

        # Context branch
        self.ctx_net = nn.Sequential(
            nn.Linear(ctx_size, 16),
            nn.ReLU(),
        )

        # Final: trunk (128) + context (16) → 1
        self.final = nn.Linear(hidden // 2 + 16, 1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)

        if ctx is not None:
            ctx_out = self.ctx_net(ctx)
            h = torch.cat([h, ctx_out], dim=-1)
        else:
            zeros = torch.zeros(h.shape[0], 16, device=h.device)
            h = torch.cat([h, zeros], dim=-1)

        return self.final(h).squeeze(-1)

    def load_v3(self, v3_state_dict: dict):
        """Load v3 DeclNet weights. Map net.N.* → layerN.*, zero-pad final layer."""
        own = self.state_dict()
        loaded = 0

        mapping = {}
        for i in range(4):
            mapping[f"net.{i}"] = f"layer1.{i}"
        for i in range(4):
            mapping[f"net.{i+4}"] = f"layer2.{i}"
        for i in range(3):
            mapping[f"net.{i+8}"] = f"layer3.{i}"
        # net.11 → final (zero-padded: [1,128] → [1,144])
        mapping["net.11"] = "final"

        for old_k, v in v3_state_dict.items():
            parts = old_k.split(".")
            if len(parts) >= 2:
                prefix = f"{parts[0]}.{parts[1]}"
                suffix = ".".join(parts[2:])
                if prefix in mapping:
                    new_k = f"{mapping[prefix]}.{suffix}"
                    if new_k in own:
                        if own[new_k].shape == v.shape:
                            own[new_k] = v
                            loaded += 1
                        elif own[new_k].dim() == 2 and v.dim() == 2 and own[new_k].shape[0] == v.shape[0]:
                            own[new_k][:, :v.shape[1]] = v
                            loaded += 1

        self.load_state_dict(own)
        return loaded

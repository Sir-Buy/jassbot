"""Neural network for RL Jass agent.

Two separate networks:
- PlayNet: policy (36 card logits) + value head for card play
- DeclNet: predicts expected score for declaration

Both share nothing — different input sizes, different tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlayNet(nn.Module):
    """Card play policy + value network.

    Input: 204 floats (observation from C engine)
    Outputs:
        policy: 36 logits (masked to legal moves)
        value: scalar (expected negative deviation)
    """

    def __init__(self, obs_size: int = 204, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 36),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs: torch.Tensor, legal_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: (B, 204) observation
            legal_mask: (B, 36) binary mask of legal moves

        Returns:
            logits: (B, 36) masked policy logits
            value: (B,) state value
        """
        x = self.trunk(obs)
        logits = self.policy_head(x)
        # Mask illegal moves with large negative
        logits = logits + (1.0 - legal_mask) * (-1e8)
        value = self.value_head(x).squeeze(-1)
        return logits, value


class DeclNet(nn.Module):
    """Declaration prediction network.

    Input: 72 floats (hand features from C engine)
    Output: scalar (predicted score, used as declaration)
    """

    def __init__(self, obs_size: int = 72, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted score (B,)."""
        return self.net(x).squeeze(-1)

"""Neural network for Differenzler card play + value estimation."""

from __future__ import annotations

import torch
import torch.nn as nn

FEATURE_SIZE = 266


class DifferenzlerNet(nn.Module):
    def __init__(self, input_size: int = FEATURE_SIZE):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 36),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, legal_mask=None):
        trunk = self.trunk(x)
        logits = self.policy(trunk)
        if legal_mask is not None:
            logits = logits.float().masked_fill(~legal_mask.bool(), -1e9)
        value = self.value(trunk)
        return logits, value

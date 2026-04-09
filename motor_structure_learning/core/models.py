"""Model API for motor-structure-learning."""

import torch
import torch.nn as nn

from .ec import EntorhinalCortex
from .hc import Hippocampus


class CerebellumForwardModelContinuous(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, state, action, context):
        x = torch.cat([state, action, context], dim=-1)
        return self.net(x)

__all__ = [
    "EntorhinalCortex",
    "Hippocampus",
    "CerebellumForwardModelContinuous",
]

import torch
import torch.nn as nn


class EntorhinalCortex(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, latent_dim=16):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

    def forward(self, transitions, h0=None):
        h_seq, h_last = self.gru(transitions, h0)
        z = self.readout(h_seq)
        return z, h_last

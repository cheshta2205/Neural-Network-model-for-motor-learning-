import torch
import torch.nn as nn


class Hippocampus(nn.Module):
    def __init__(self, latent_dim=16, hc_dim=16):
        super().__init__()

        self.hc_dim = hc_dim
        self.alpha_raw = nn.Parameter(torch.tensor(-2.2))
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hc_dim),
            nn.Tanh(),
        )
        self.output_proj = nn.Linear(hc_dim, hc_dim)

    def forward(self, z_seq, c0=None):
        batch, seq_len, _ = z_seq.shape
        c = torch.zeros(batch, self.hc_dim, device=z_seq.device) if c0 is None else c0
        alpha = torch.sigmoid(self.alpha_raw)

        c_seq = []
        for t in range(seq_len):
            z_proj = self.input_proj(z_seq[:, t, :])
            c = (1 - alpha) * c + alpha * z_proj
            c_seq.append(c)

        c_seq = torch.stack(c_seq, dim=1)
        c_out = self.output_proj(c_seq)
        return c_out, c

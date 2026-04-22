"""Model API for motor-structure-learning."""

import torch
from torch.distributions import Normal
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
    
class MotorCortex(nn.Module):
    """Actor-Critic MC with shared GRU hidden state."""

    def __init__(self, latent_dim=32, ec_hidden=256, cb_proj_dim=32, mc_hidden=256,
                 action_std_min=-4.5, action_std_max=0.0):
        super().__init__()
        input_dim = 2 + 2 + latent_dim + cb_proj_dim

        self.ec_readout = nn.Linear(ec_hidden, latent_dim)
        self.cb_proj = nn.Linear(cb_proj_dim, cb_proj_dim, bias=False)
        self.cell = nn.GRUCell(input_dim, mc_hidden)
        self.mu_head = nn.Linear(mc_hidden, 2)
        self.log_std_head = nn.Linear(mc_hidden, 2)
        self.value_head = nn.Linear(mc_hidden, 1)
        self.action_std_min = action_std_min
        self.action_std_max = action_std_max

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.cell.hidden_size, device=device)

    def _forward_hidden(self, s_t, g_t, z_ec, z_cb, h_mc):
        x = torch.cat([s_t, g_t, z_ec, z_cb], dim=-1)
        h_new = self.cell(x, h_mc)
        mu = self.mu_head(h_new)
        log_std = torch.clamp(self.log_std_head(h_new), self.action_std_min, self.action_std_max)
        V = self.value_head(h_new)
        return mu, log_std, V, h_new

    def step(self, s_t, h_ec, g_t, z_cb_t, h_mc):
        z_ec = torch.tanh(self.ec_readout(h_ec))
        z_cb = self.cb_proj(z_cb_t)
        return self._forward_hidden(s_t, g_t, z_ec, z_cb, h_mc)

    def sample_action(self, s_t, h_ec, g_t, z_cb_t, h_mc):
        mu, log_std, V, h_new = self.step(s_t, h_ec, g_t, z_cb_t, h_mc)
        std = log_std.exp()
        dist = Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return a, log_prob, entropy, V, h_new

    def greedy_action(self, s_t, h_ec, g_t, z_cb_t, h_mc):
        mu, _, _, h_new = self.step(s_t, h_ec, g_t, z_cb_t, h_mc)
        return torch.tanh(mu), h_new

__all__ = [
    "EntorhinalCortex",
    "Hippocampus",
    "CerebellumForwardModelContinuous",
    "MotorCortex",
]

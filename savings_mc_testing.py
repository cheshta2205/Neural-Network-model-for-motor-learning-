"""
savings_mc_testing.py

Control experiment: train a fresh MC from scratch on diagonal vs anti-diagonal FF.
No ABA, no transfer — isolated training to test if anti-diagonal is inherently easier.

Two conditions:
  1. MC trained only on diagonal     (0.8,  0.8)
  2. MC trained only on anti-diagonal(-0.8, -0.8)

Figures saved to figs/ECtoCB/gridworld_policy/learning_MC_SAVINGS/mc_testing/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyperparameters ──────────────────────────────────────────────────────────
LATENT_DIM  = 16
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
MC_HIDDEN   = 256
MC_INPUT    = 36
CB_PROJ_DIM = 16

MAX_STEPS     = 50
N_TRAIN_EP    = 2000
GOAL_RADIUS   = 0.3
GRID_SIZE     = 4
LR_MC         = 1e-3
EVAL_INTERVAL = 1
N_CURVE_EP    = 1
N_SEEDS       = 5

FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/gridworld_policy/learning_MC_SAVINGS/mc_testing"
CKPT    = "checkpoints/ectocb_mse_shiftedv2.pt"

DIAG_CONTEXT     = [(0.8,  0.8)]
ANTIDIAG_CONTEXT = [(-0.8, -0.8)]


# ── Models ───────────────────────────────────────────────────────────────────
class EntorhinalCortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell    = nn.GRUCell(EC_INPUT, EC_HIDDEN)
        self.readout = nn.Sequential(nn.Linear(EC_HIDDEN, LATENT_DIM), nn.Tanh())

    def init_hidden(self, b, d):
        return torch.zeros(b, EC_HIDDEN, device=d)

    def step(self, x, h):
        h2 = self.cell(x, h)
        return self.readout(h2), h2


class CerebellumCB(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4 + LATENT_DIM, CB_HIDDEN), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU())
        self.out    = nn.Linear(CB_HIDDEN, 2)

    def forward(self, s, a, z):
        h = self.layer1(torch.cat([s, a, z], dim=-1))
        return self.out(self.layer2(h))

    def hidden(self, s, a, z):
        return self.layer2(self.layer1(torch.cat([s, a, z], dim=-1)))


class MotorCortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell       = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.readout    = nn.Linear(MC_HIDDEN, 2)
        W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
        self.register_buffer('cb_proj', W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, b, d):
        return torch.zeros(b, MC_HIDDEN, device=d)

    def step(self, s_t, h_ec, g, h, cb_hidden):
        z_ec  = torch.tanh(self.ec_readout(h_ec))
        z_cb  = cb_hidden @ self.cb_proj
        h2    = self.cell(torch.cat([s_t, z_ec, z_cb, g], dim=-1), h)
        return torch.tanh(self.readout(h2)), h2


# ── Environment ──────────────────────────────────────────────────────────────
def env_step_torch(pos, action, beta, gamma):
    n    = torch.norm(action) + 1e-8
    ah   = action / n
    ap   = torch.stack([-ah[1], ah[0]])
    return torch.clamp(pos + action + beta * ah + gamma * ap, 0, GRID_SIZE)


# ── Episode ──────────────────────────────────────────────────────────────────
def run_episode(ec, cb, mc, device, rng, contexts, h_ec):
    start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    while np.array_equal(start, goal):
        goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    beta, gamma = contexts[rng.integers(0, len(contexts))]

    g      = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos    = torch.tensor(start, dtype=torch.float32, device=device)
    h_mc   = mc.init_hidden(1, device)
    z_t    = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)
    losses = []

    for t in range(MAX_STEPS):
        if torch.norm(pos - g).item() < GOAL_RADIUS:
            break
        with torch.no_grad():
            cb_h = cb.hidden(pos.unsqueeze(0), torch.zeros(1, 2, device=device) if t == 0 else prev_a, z_t)
        a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g.unsqueeze(0), h_mc, cb_h)
        s_next    = env_step_torch(pos, a_t.squeeze(0), beta, gamma)
        losses.append(torch.sum((g - s_next) ** 2))
        with torch.no_grad():
            s_tm1     = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            eps       = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
            z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)
        prev_a = a_t.detach()
        pos    = s_next.detach()

    h_ec = h_ec.detach()
    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True), h_ec
    return torch.stack(losses).sum(), h_ec


# ── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate(ec, cb, mc, device, contexts, h_ec, seed=42):
    ec.eval(); cb.eval(); mc.eval()
    rng     = np.random.default_rng(seed)
    results = {'success': [], 'steps': [], 'norm_path': []}

    with torch.no_grad():
        for _ in range(N_CURVE_EP):
            start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
            goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
            while np.array_equal(start, goal):
                goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
            beta, gamma = contexts[rng.integers(0, len(contexts))]

            g      = torch.tensor(goal,  dtype=torch.float32, device=device).unsqueeze(0)
            pos    = torch.tensor(start, dtype=torch.float32, device=device)
            h_mc   = mc.init_hidden(1, device)
            z_t    = torch.zeros(1, LATENT_DIM, device=device)
            prev_a = torch.zeros(1, 2, device=device)
            success    = False
            steps      = 0
            total_dist = 0.0
            init_dist  = np.linalg.norm(start - goal)

            for t in range(MAX_STEPS):
                if np.linalg.norm(pos.cpu().numpy() - goal) < GOAL_RADIUS:
                    success = True; break
                total_dist += np.linalg.norm(pos.cpu().numpy() - goal)
                cb_h      = cb.hidden(pos.unsqueeze(0), torch.zeros(1, 2, device=device) if t == 0 else prev_a, z_t)
                a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g, h_mc, cb_h)
                s_next    = env_step_torch(pos, a_t.squeeze(0), beta, gamma)
                s_tm1     = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                eps       = s_next.unsqueeze(0) - cb(pos.unsqueeze(0), a_t, z_t)
                z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)
                prev_a = a_t; pos = s_next; steps += 1

            norm_path = (total_dist / steps / init_dist) if (steps > 0 and init_dist > 0) else 0.0
            results['success'].append(float(success))
            results['steps'].append(steps)
            results['norm_path'].append(norm_path)

    return results, h_ec


# ── Train ─────────────────────────────────────────────────────────────────────
def train_condition(ec, cb, device, contexts, label, seed):
    mc  = MotorCortex().to(device)
    opt = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)

    rng  = np.random.default_rng(seed)
    h_ec = ec.init_hidden(1, device)
    curve = {'success': [], 'steps': [], 'norm_path': [], 'ep': []}

    print(f'  {label} | seed={seed}')
    for ep in range(1, N_TRAIN_EP + 1):
        mc.train(); opt.zero_grad()
        loss, h_ec = run_episode(ec, cb, mc, device, rng, contexts, h_ec)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
        opt.step()

        if ep % EVAL_INTERVAL == 0:
            r, h_ec = evaluate(ec, cb, mc, device, contexts, h_ec, seed=ep)
            curve['success'].extend(r['success'])
            curve['steps'].extend(r['steps'])
            curve['norm_path'].extend(r['norm_path'])
            curve['ep'].extend([ep] * N_CURVE_EP)

    return curve


# ── Plot ──────────────────────────────────────────────────────────────────────
ROLLING_WIN = 30

def rolling_mean(x, w):
    x = np.array(x); r = np.zeros(len(x))
    for i in range(len(x)):
        r[i] = np.mean(x[max(0, i - w + 1):i + 1])
    return r


def plot_comparison(curve_diag, curve_anti, seed):
    metrics = [
        ('norm_path', 'Mean dist/step / init dist', False),
        ('success',   'Success rate',                True),
        ('steps',     'Steps to goal',               True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (metric, ylabel, smooth) in zip(axes, metrics):
        d = np.array(curve_diag[metric])
        a = np.array(curve_anti[metric])
        if smooth:
            d = rolling_mean(d, ROLLING_WIN)
            a = rolling_mean(a, ROLLING_WIN)
        x = np.arange(len(d))
        ax.plot(x, d, color='steelblue', lw=1.5, label='diagonal (0.8, 0.8)')
        ax.plot(x, a, color='tomato',    lw=1.5, label='anti-diagonal (-0.8, -0.8)')
        ax.set_xlabel('Trial'); ax.set_ylabel(ylabel)
        if metric == 'success': ax.set_ylim(0, 1.15)
        if smooth: ax.set_title(f'rolling mean={ROLLING_WIN}', fontsize=8)
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Fresh MC: diagonal vs anti-diagonal from scratch | seed={seed}', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, f'mc_testing_seed{seed}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    ckpt = torch.load(CKPT, map_location=device)
    ec   = EntorhinalCortex().to(device)
    cb   = CerebellumCB().to(device)
    ec.load_state_dict(ckpt['ec'])
    cb_sd = ckpt['cb']
    remap = {'net.0': 'layer1.0', 'net.2': 'layer2.0', 'net.4': 'out'}
    new_sd = {}
    for k, v in cb_sd.items():
        prefix  = '.'.join(k.split('.')[:2])
        new_sd[k.replace(prefix, remap[prefix]) if prefix in remap else k] = v
    cb.load_state_dict(new_sd)
    ec.eval(); cb.eval()
    print(f'Loaded EC + CB from {CKPT}')

    os.makedirs(FIG_DIR, exist_ok=True)

    for seed in range(N_SEEDS):
        print(f'\n=== Seed {seed} ===')
        curve_diag = train_condition(ec, cb, device, DIAG_CONTEXT,     'diagonal',     seed * 100)
        curve_anti = train_condition(ec, cb, device, ANTIDIAG_CONTEXT, 'anti-diagonal', seed * 100 + 1)
        np.save(os.path.join(FIG_DIR, f'curve_diag_seed{seed}.npy'),     curve_diag, allow_pickle=True)
        np.save(os.path.join(FIG_DIR, f'curve_antidiag_seed{seed}.npy'), curve_anti, allow_pickle=True)
        plot_comparison(curve_diag, curve_anti, seed)

    print('\nAll done.')


if __name__ == '__main__':
    main()

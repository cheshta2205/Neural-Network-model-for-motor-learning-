"""
learning_MC_CBonly_novel.py

MC trained on ONE novel diagonal FF (0.8, 0.8).
No EC — MC only receives CB random projection as context signal.

MC input: s_t(2) + z_cb(16) + g(2) = 20-D
CB input: z=0 (no EC context, CB runs with no context signal)

Question: can MC learn a novel FF using only CB's internal activity,
          with no online context inference from EC?

Figures saved to figs/ECtoCB/gridworld_policy/learning_MC_CBonly_novel/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
LATENT_DIM  = 16
CB_HIDDEN   = 256
MC_HIDDEN   = 256
MC_INPUT    = 20   # s_t(2) + z_cb(16) + g(2)
CB_PROJ_DIM = 16

MAX_STEPS      = 50
N_TRAIN_EP     = 15000
N_EVAL_EP      = 500
GOAL_RADIUS    = 0.3
GRID_SIZE      = 4
LR_MC          = 1e-3
EVAL_INTERVAL  = 100
N_CURVE_EP     = 100

FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/gridworld_policy/learning_MC_CBonly_novel"
CKPT    = "checkpoints/ectocb_mse_shiftedv2.pt"

# Train on ONE novel diagonal FF
TRAIN_CONTEXTS = [
    (0.8, 0.8),
]

FAMILIAR_CONTEXTS = [
    ( 0.8,  0.0),
    (-0.8,  0.0),
    ( 0.0,  0.8),
    ( 0.0, -0.8),
]

ALL_NOVEL_CONTEXTS = [
    ( 0.8,  0.8),
    (-0.8,  0.8),
    ( 0.8, -0.8),
    (-0.8, -0.8),
]


# ── Models ─────────────────────────────────────────────────────────────────────
class CerebellumCB(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4 + LATENT_DIM, CB_HIDDEN), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(CB_HIDDEN, CB_HIDDEN),       nn.ReLU())
        self.out    = nn.Linear(CB_HIDDEN, 2)

    def forward(self, s, a, z):
        h = self.layer1(torch.cat([s, a, z], dim=-1))
        h = self.layer2(h)
        return self.out(h)

    def hidden(self, s, a, z):
        h = self.layer1(torch.cat([s, a, z], dim=-1))
        return self.layer2(h)


class MotorCortex(nn.Module):
    """MC with no EC — only CB random projection as context."""
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.cell    = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.readout = nn.Linear(MC_HIDDEN, 2)

        W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
        self.register_buffer('cb_proj', W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, g, h, cb_hidden):
        z_cb  = cb_hidden @ self.cb_proj          # (1, 16)
        h_new = self.cell(torch.cat([s_t, z_cb, g], dim=-1), h)
        a_t   = torch.tanh(self.readout(h_new))
        return a_t, h_new


# ── Environment ────────────────────────────────────────────────────────────────
def env_step_torch(pos, action, beta, gamma):
    a_norm = torch.norm(action) + 1e-8
    a_hat  = action / a_norm
    a_perp = torch.stack([-a_hat[1], a_hat[0]])
    f_env  = beta * a_hat + gamma * a_perp
    return torch.clamp(pos + action + f_env, 0, GRID_SIZE)


# ── Training ───────────────────────────────────────────────────────────────────
def run_mc_episode(cb, mc, device, rng):
    start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    while np.array_equal(start, goal):
        goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    beta, gamma = TRAIN_CONTEXTS[rng.integers(0, len(TRAIN_CONTEXTS))]

    g_tensor = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos      = torch.tensor(start, dtype=torch.float32, device=device)
    h_mc     = mc.init_hidden(1, device)
    z_t      = torch.zeros(1, LATENT_DIM, device=device)   # CB always gets z=0
    prev_a   = torch.zeros(1, 2, device=device)

    step_losses = []

    for t in range(MAX_STEPS):
        if torch.norm(pos - g_tensor).item() < GOAL_RADIUS:
            break

        with torch.no_grad():
            cb_hidden = cb.hidden(pos.unsqueeze(0),
                                  torch.zeros(1, 2, device=device) if t == 0 else prev_a,
                                  z_t)

        a_t, h_mc = mc.step(pos.unsqueeze(0), g_tensor.unsqueeze(0), h_mc, cb_hidden)
        s_next    = env_step_torch(pos, a_t.squeeze(0), beta, gamma)

        step_losses.append(torch.sum((g_tensor - s_next) ** 2))

        prev_a = a_t.detach()
        pos    = s_next.detach()

    if not step_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(step_losses).sum()


def train_mc(cb, mc, device, seed=0):
    cb.eval(); mc.train()
    for p in cb.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    rng       = np.random.default_rng(seed)
    losses    = []

    curve = {
        'ep':                      [],
        'novel_success':           [], 'novel_steps':           [],
        'allnovel_success':        [], 'allnovel_steps':        [],
        'fam_success':             [], 'fam_steps':             [],
    }

    print(f"\nTraining MC (no EC, CB projection only) on novel FF (0.8,0.8) — {N_TRAIN_EP} episodes...")
    for ep in range(1, N_TRAIN_EP + 1):
        optimizer.zero_grad()
        loss = run_mc_episode(cb, mc, device, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if ep % EVAL_INTERVAL == 0:
            mc.eval()
            r_n  = evaluate(cb, mc, device, TRAIN_CONTEXTS,     seed=ep, n_ep=N_CURVE_EP)
            r_an = evaluate(cb, mc, device, ALL_NOVEL_CONTEXTS, seed=ep, n_ep=N_CURVE_EP)
            r_f  = evaluate(cb, mc, device, FAMILIAR_CONTEXTS,  seed=ep, n_ep=N_CURVE_EP)
            curve['ep'].append(ep)
            curve['novel_success'].append(np.mean(r_n['success']))
            curve['novel_steps'].append(np.mean(r_n['steps']))
            curve['allnovel_success'].append(np.mean(r_an['success']))
            curve['allnovel_steps'].append(np.mean(r_an['steps']))
            curve['fam_success'].append(np.mean(r_f['success']))
            curve['fam_steps'].append(np.mean(r_f['steps']))
            mc.train()

            if ep % 300 == 0:
                print(f"  Ep {ep:5d} | loss={np.mean(losses[-300:]):.3f} | "
                      f"trained_nov={curve['novel_success'][-1]:.2f} "
                      f"all_nov={curve['allnovel_success'][-1]:.2f} "
                      f"fam={curve['fam_success'][-1]:.2f}")

    return losses, curve


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(cb, mc, device, contexts, seed=42, n_ep=None):
    cb.eval(); mc.eval()
    rng     = np.random.default_rng(seed)
    results = {"success": [], "steps": []}
    n_ep    = n_ep if n_ep is not None else N_EVAL_EP

    for _ in range(n_ep):
        start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        while np.array_equal(start, goal):
            goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        beta, gamma = contexts[rng.integers(0, len(contexts))]

        g_tensor = torch.tensor(goal,  dtype=torch.float32, device=device).unsqueeze(0)
        pos      = torch.tensor(start, dtype=torch.float32, device=device)
        h_mc     = mc.init_hidden(1, device)
        z_t      = torch.zeros(1, LATENT_DIM, device=device)
        prev_a   = torch.zeros(1, 2, device=device)
        success  = False
        steps    = 0

        with torch.no_grad():
            for t in range(MAX_STEPS):
                if np.linalg.norm(pos.numpy() - goal) < GOAL_RADIUS:
                    success = True; break

                cb_hidden = cb.hidden(pos.unsqueeze(0),
                                      torch.zeros(1, 2, device=device) if t == 0 else prev_a,
                                      z_t)
                a_t, h_mc = mc.step(pos.unsqueeze(0), g_tensor, h_mc, cb_hidden)
                s_next    = env_step_torch(pos, a_t.squeeze(0), beta, gamma)

                prev_a = a_t
                pos    = s_next
                steps += 1

        results["success"].append(float(success))
        results["steps"].append(steps)

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_learning_curves(curve):
    os.makedirs(FIG_DIR, exist_ok=True)
    ep = curve['ep']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    styles = [
        ('novel',    'Trained novel (0.8,0.8)', 'steelblue', '-'),
        ('allnovel', 'All 4 novel (±0.8 diag)', 'purple',    '-'),
        ('fam',      'Familiar (±0.8 axis)',     'tomato',    '-'),
    ]
    for key, label, color, ls in styles:
        axes[0].plot(ep, curve[f'{key}_success'], color=color, ls=ls, lw=2, label=label)
        axes[1].plot(ep, curve[f'{key}_steps'],   color=color, ls=ls, lw=2, label=label)

    axes[0].set_ylabel('Success rate'); axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel('Mean steps to goal')
    for ax in axes:
        ax.set_xlabel('Training episode')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    fig.suptitle('MC (no EC) + CB projection — trained on novel FF (0.8, 0.8)', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'learning_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {out}')


def plot_training(losses):
    os.makedirs(FIG_DIR, exist_ok=True)
    smoothed = np.convolve(losses, np.ones(50) / 50, mode='valid')
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.plot(smoothed, color='steelblue', lw=2)
    ax.set_xlabel('Episode'); ax.set_ylabel('Loss')
    ax.set_title('MC (no EC) training loss — novel FF (0.8, 0.8)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'mc_training_loss.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {out}')


def plot_bar(results):
    os.makedirs(FIG_DIR, exist_ok=True)

    labels_x = ['Trained novel\n(0.8, 0.8)', 'All 4 novel\n(±0.8 diag)', 'Familiar\n(±0.8 axis)']
    keys     = ['trained_novel', 'all_novel', 'familiar']
    x        = np.arange(len(labels_x))

    success = [np.mean(results[k]['success']) for k in keys]
    steps   = [np.mean(results[k]['steps'])   for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, ylabel in zip(axes, [success, steps], ['Success rate', 'Mean steps to goal']):
        bars = ax.bar(labels_x, vals, color='steelblue', alpha=0.85, width=0.5)
        for bar, val in zip(bars, vals):
            fmt = f'{val:.2f}' if ylabel == 'Success rate' else f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.02 if ylabel == 'Success rate' else 0.5),
                    fmt, ha='center', fontsize=9)
        ax.set_ylabel(ylabel)
        if ylabel == 'Success rate':
            ax.set_ylim(0, 1.15)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('MC (no EC) + CB projection — trained on novel FF (0.8, 0.8)\n'
                 'Input: s_t + z_cb + g  (no EC)', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'bar_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {out}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    ckpt = torch.load(CKPT, map_location=device)
    cb   = CerebellumCB().to(device)
    cb_sd = ckpt['cb']
    remap = {'net.0': 'layer1.0', 'net.2': 'layer2.0', 'net.4': 'out'}
    new_sd = {}
    for k, v in cb_sd.items():
        prefix = '.'.join(k.split('.')[:2])
        new_key = k.replace(prefix, remap[prefix]) if prefix in remap else k
        new_sd[new_key] = v
    cb.load_state_dict(new_sd)
    cb.eval()
    print(f'Loaded frozen CB from {CKPT}')

    mc            = MotorCortex().to(device)
    losses, curve = train_mc(cb, mc, device)
    plot_training(losses)
    plot_learning_curves(curve)

    print('\nEvaluating...')
    conds = [
        ('trained_novel', TRAIN_CONTEXTS),
        ('all_novel',     ALL_NOVEL_CONTEXTS),
        ('familiar',      FAMILIAR_CONTEXTS),
    ]
    results = {}
    for key, ctx in conds:
        r = evaluate(cb, mc, device, ctx)
        results[key] = r
        print(f'  {key:20s} | success={np.mean(r["success"]):.3f}  steps={np.mean(r["steps"]):.1f}')

    plot_bar(results)
    print('\nDone.')


if __name__ == '__main__':
    main()

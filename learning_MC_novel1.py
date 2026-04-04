"""
learning_MC_novel1.py


MC trained directly on novel diagonal force fields (4 contexts)
MC input: s_t(2) + z'_ec(16) + z_cb(16) + g(2) = 36-D

Question1 : is can MC (with EC and CB projections in it) learn a single novel environment?
Question 2: Does EC play a role here? or is EC still needed when CB signal is available to MC.

Evaluate:
  - Trained novel FF (0.8, 0.8)        — EC intact vs EC lesion
Figures saved to figs/ECtoCB/gridworld_policy/learning_MC_novel1/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
LATENT_DIM  = 16
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
MC_HIDDEN   = 256
MC_INPUT    = 36   # s_t(2) + z'_ec(16) + z_cb(16) + g(2)
CB_PROJ_DIM = 16

MAX_STEPS      = 50
N_TRAIN_EP     = 15000
N_EVAL_EP      = 500
GOAL_RADIUS    = 0.3
GRID_SIZE      = 4
LR_MC          = 1e-3
EVAL_INTERVAL  = 100
N_CURVE_EP     = 100
GAMMA_DISCOUNT = 0.95

FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/gridworld_policy/learning_MC_novel4"
CKPT    = "checkpoints/ectocb_mse_shiftedv2.pt"

# Train on all 4 novel diagonal FFs
TRAIN_CONTEXTS = [
    ( 0.8,  0.8),
    (-0.8,  0.8),
    ( 0.8, -0.8),
    (-0.8, -0.8),
]

# All 4 diagonal novel FFs — for generalisation evaluation
ALL_NOVEL_CONTEXTS = [
    ( 0.8,  0.8),
    (-0.8,  0.8),
    ( 0.8, -0.8),
    (-0.8, -0.8),
]


# ── Models ─────────────────────────────────────────────────────────────────────
class EntorhinalCortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = EC_HIDDEN
        self.cell    = nn.GRUCell(EC_INPUT, EC_HIDDEN)
        self.readout = nn.Sequential(nn.Linear(EC_HIDDEN, LATENT_DIM), nn.Tanh())

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, EC_HIDDEN, device=device)

    def step(self, x, h):
        h_new = self.cell(x, h)
        return self.readout(h_new), h_new


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
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell       = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.readout    = nn.Linear(MC_HIDDEN, 2)

        W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
        self.register_buffer('cb_proj', W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, h_ec, g, h, cb_hidden):
        z_ec  = torch.tanh(self.ec_readout(h_ec))
        z_cb  = cb_hidden @ self.cb_proj
        h_new = self.cell(torch.cat([s_t, z_ec, z_cb, g], dim=-1), h)
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
def run_mc_episode(ec, cb, mc, device, rng):
    start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    while np.array_equal(start, goal):
        goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    beta, gamma = TRAIN_CONTEXTS[rng.integers(0, len(TRAIN_CONTEXTS))]

    g_tensor = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos      = torch.tensor(start, dtype=torch.float32, device=device)

    h_ec = ec.init_hidden(1, device)
    h_mc = mc.init_hidden(1, device)
    z_t  = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)

    step_losses = []

    for t in range(MAX_STEPS):
        if torch.norm(pos - g_tensor).item() < GOAL_RADIUS:
            break

        with torch.no_grad():
            cb_hidden = cb.hidden(pos.unsqueeze(0),
                                  torch.zeros(1, 2, device=device) if t == 0 else prev_a,
                                  z_t)

        a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g_tensor.unsqueeze(0), h_mc, cb_hidden)
        s_next    = env_step_torch(pos, a_t.squeeze(0), beta, gamma)

        step_losses.append(torch.sum((g_tensor - s_next) ** 2))

        with torch.no_grad():
            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
            z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()

    if not step_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(step_losses).sum()


def train_mc(ec, cb, mc, device, seed=0):
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    rng       = np.random.default_rng(seed)
    losses    = []

    curve = {
        'ep':                      [],
        'novel_ec_success':        [], 'novel_ec_steps':        [],
        'novel_lesion_success':    [], 'novel_lesion_steps':    [],
        'allnovel_ec_success':     [], 'allnovel_ec_steps':     [],
        'allnovel_lesion_success': [], 'allnovel_lesion_steps': [],
        'fam_ec_success':          [], 'fam_ec_steps':          [],
        'fam_lesion_success':      [], 'fam_lesion_steps':      [],
    }

    print(f"\nTraining MC on all 4 novel diagonal FFs (±0.8) — {N_TRAIN_EP} episodes...")
    for ep in range(1, N_TRAIN_EP + 1):
        optimizer.zero_grad()
        loss = run_mc_episode(ec, cb, mc, device, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if ep % EVAL_INTERVAL == 0:
            mc.eval()
            r_ne = evaluate(ec, cb, mc, device, TRAIN_CONTEXTS,     seed=ep, ec_lesion=False, n_ep=N_CURVE_EP)
            r_nl = evaluate(ec, cb, mc, device, TRAIN_CONTEXTS,     seed=ep, ec_lesion=True,  n_ep=N_CURVE_EP)
            r_ae = evaluate(ec, cb, mc, device, ALL_NOVEL_CONTEXTS, seed=ep, ec_lesion=False, n_ep=N_CURVE_EP)
            r_al = evaluate(ec, cb, mc, device, ALL_NOVEL_CONTEXTS, seed=ep, ec_lesion=True,  n_ep=N_CURVE_EP)
            r_fe = evaluate(ec, cb, mc, device, FAMILIAR_CONTEXTS,  seed=ep, ec_lesion=False, n_ep=N_CURVE_EP)
            r_fl = evaluate(ec, cb, mc, device, FAMILIAR_CONTEXTS,  seed=ep, ec_lesion=True,  n_ep=N_CURVE_EP)
            curve['ep'].append(ep)
            curve['novel_ec_success'].append(np.mean(r_ne['success']))
            curve['novel_ec_steps'].append(np.mean(r_ne['steps']))
            curve['novel_lesion_success'].append(np.mean(r_nl['success']))
            curve['novel_lesion_steps'].append(np.mean(r_nl['steps']))
            curve['allnovel_ec_success'].append(np.mean(r_ae['success']))
            curve['allnovel_ec_steps'].append(np.mean(r_ae['steps']))
            curve['allnovel_lesion_success'].append(np.mean(r_al['success']))
            curve['allnovel_lesion_steps'].append(np.mean(r_al['steps']))
            curve['fam_ec_success'].append(np.mean(r_fe['success']))
            curve['fam_ec_steps'].append(np.mean(r_fe['steps']))
            curve['fam_lesion_success'].append(np.mean(r_fl['success']))
            curve['fam_lesion_steps'].append(np.mean(r_fl['steps']))
            mc.train()

            if ep % 300 == 0:
                print(f"  Ep {ep:5d} | loss={np.mean(losses[-300:]):.3f} | "
                      f"trained_nov={curve['novel_ec_success'][-1]:.2f} "
                      f"all_nov={curve['allnovel_ec_success'][-1]:.2f} "
                      f"fam={curve['fam_ec_success'][-1]:.2f}")

    return losses, curve


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(ec, cb, mc, device, contexts, seed=42, ec_lesion=False, n_ep=None):
    ec.eval(); cb.eval(); mc.eval()
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
        h_ec     = torch.zeros(1, EC_HIDDEN, device=device)
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
                a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g_tensor, h_mc, cb_hidden)
                s_next    = env_step_torch(pos, a_t.squeeze(0), beta, gamma)

                if not ec_lesion:
                    s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                    eps   = s_next.unsqueeze(0) - cb(pos.unsqueeze(0), a_t, z_t)
                    z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

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

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    styles = [
        ('novel_ec',     'Trained novel (0.8,0.8) — EC intact',  'steelblue', '-'),
        ('novel_lesion', 'Trained novel (0.8,0.8) — EC lesion',  'steelblue', '--'),
        ('allnovel_ec',     'All 4 novel — EC intact',           'purple',    '-'),
        ('allnovel_lesion', 'All 4 novel — EC lesion',           'purple',    '--'),
        ('fam_ec',       'Familiar — EC intact',                 'tomato',    '-'),
        ('fam_lesion',   'Familiar — EC lesion',                 'tomato',    '--'),
    ]
    for key, label, color, ls in styles:
        axes[0].plot(ep, curve[f'{key}_success'], color=color, ls=ls, lw=2, label=label)
        axes[1].plot(ep, curve[f'{key}_steps'],   color=color, ls=ls, lw=2, label=label)

    axes[0].set_ylabel('Success rate'); axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel('Mean steps to goal')
    for ax in axes:
        ax.set_xlabel('Training episode')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=7)

    fig.suptitle('Learning curves — trained on all 4 novel diagonal FFs (±0.8)', fontsize=11)
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
    ax.set_title('MC training loss — all 4 novel diagonal FFs (±0.8)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'mc_training_loss.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {out}')


def plot_bar(results):
    os.makedirs(FIG_DIR, exist_ok=True)

    labels_x   = ['Trained novel\n(0.8, 0.8)', 'All 4 novel\n(±0.8 diag)', 'Familiar\n(±0.8 axis)']
    bar_labels = ['EC intact', 'EC lesion']
    colors     = ['steelblue', 'tomato']
    x          = np.arange(len(labels_x))
    w          = 0.35

    success = {
        'EC intact': [np.mean(results['trained_novel_ec']['success']),
                      np.mean(results['all_novel_ec']['success']),
                      np.mean(results['familiar_ec']['success'])],
        'EC lesion': [np.mean(results['trained_novel_lesion']['success']),
                      np.mean(results['all_novel_lesion']['success']),
                      np.mean(results['familiar_lesion']['success'])],
    }
    steps = {
        'EC intact': [np.mean(results['trained_novel_ec']['steps']),
                      np.mean(results['all_novel_ec']['steps']),
                      np.mean(results['familiar_ec']['steps'])],
        'EC lesion': [np.mean(results['trained_novel_lesion']['steps']),
                      np.mean(results['all_novel_lesion']['steps']),
                      np.mean(results['familiar_lesion']['steps'])],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylabel in zip(axes, [success, steps], ['Success rate', 'Mean steps to goal']):
        for i, (lbl, color) in enumerate(zip(bar_labels, colors)):
            vals = metric[lbl]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=lbl, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                fmt = f'{val:.2f}' if ylabel == 'Success rate' else f'{val:.1f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.02 if ylabel == 'Success rate' else 0.5),
                        fmt, ha='center', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels_x)
        ax.set_ylabel(ylabel)
        if ylabel == 'Success rate':
            ax.set_ylim(0, 1.15)
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('MC trained on all 4 novel diagonal FFs (±0.8)\n'
                 'Input: s_t + z\'_ec + z_cb + g  |  EC intact vs lesion',
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'bar_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {out}')


# ── Main ───────────────────────────────────────────────────────────────────────
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
        prefix = '.'.join(k.split('.')[:2])
        new_key = k.replace(prefix, remap[prefix]) if prefix in remap else k
        new_sd[new_key] = v
    cb.load_state_dict(new_sd)
    ec.eval(); cb.eval()
    print(f'Loaded frozen EC + CB from {CKPT}')

    mc            = MotorCortex().to(device)
    losses, curve = train_mc(ec, cb, mc, device)
    plot_training(losses)
    plot_learning_curves(curve)

    print('\nEvaluating...')
    conds = [
        ('trained_novel_ec',     TRAIN_CONTEXTS,     False),
        ('trained_novel_lesion', TRAIN_CONTEXTS,     True),
        ('all_novel_ec',         ALL_NOVEL_CONTEXTS, False),
        ('all_novel_lesion',     ALL_NOVEL_CONTEXTS, True),
        ('familiar_ec',          FAMILIAR_CONTEXTS,  False),
        ('familiar_lesion',      FAMILIAR_CONTEXTS,  True),
    ]
    results = {}
    for key, ctx, lesion in conds:
        r = evaluate(ec, cb, mc, device, ctx, ec_lesion=lesion)
        results[key] = r
        tag = 'EC intact' if not lesion else 'EC lesion'
        print(f'  {key:25s} | {tag:10s} | '
              f'success={np.mean(r["success"]):.3f}  steps={np.mean(r["steps"]):.1f}')

    plot_bar(results)
    print('\nDone.')


if __name__ == '__main__':
    main()

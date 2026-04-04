"""
debug_ff.py

Diagnostic script: run a trained MC on diagonal vs anti-diagonal FF
and log per-step mechanics to understand why anti-diagonal is easier.

For each FF, logs per step:
- ||a||          : action magnitude
- ||net disp||   : actual displacement magnitude
- dist_before    : distance to goal before step
- dist_after     : distance to goal after step
- overshot       : did the agent move further from goal?

Runs N_DIAG_EP episodes on diagonal and N_ANTI_EP on anti-diagonal
using the MC trained at end of A1 phase (saved from a prior run).
Since we don't save MC checkpoints, we retrain A1 briefly and evaluate.
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

MAX_STEPS   = 50
GOAL_RADIUS = 0.3
GRID_SIZE   = 4
LR_MC       = 1e-3
N_TRAIN_EP  = 2000   # train MC briefly before diagnosing
N_DIAG_EP   = 200    # episodes to diagnose

CKPT    = "checkpoints/ectocb_mse_shiftedv2.pt"
FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/gridworld_policy/learning_MC_SAVINGS/debug_ff"

DIAG_CONTEXT     = [(0.8,  0.8)]
ANTIDIAG_CONTEXT = [(-0.8, -0.8)]


# ── Models ───────────────────────────────────────────────────────────────────
class EntorhinalCortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell    = nn.GRUCell(EC_INPUT, EC_HIDDEN)
        self.readout = nn.Sequential(nn.Linear(EC_HIDDEN, LATENT_DIM), nn.Tanh())

    def init_hidden(self, b, d): return torch.zeros(b, EC_HIDDEN, device=d)

    def step(self, x, h):
        h2 = self.cell(x, h); return self.readout(h2), h2


class CerebellumCB(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(4 + LATENT_DIM, CB_HIDDEN), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(CB_HIDDEN, CB_HIDDEN), nn.ReLU())
        self.out    = nn.Linear(CB_HIDDEN, 2)

    def forward(self, s, a, z):
        return self.out(self.layer2(self.layer1(torch.cat([s, a, z], dim=-1))))

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

    def init_hidden(self, b, d): return torch.zeros(b, MC_HIDDEN, device=d)

    def step(self, s_t, h_ec, g, h, cb_hidden):
        z_ec = torch.tanh(self.ec_readout(h_ec))
        z_cb = cb_hidden @ self.cb_proj
        h2   = self.cell(torch.cat([s_t, z_ec, z_cb, g], dim=-1), h)
        return torch.tanh(self.readout(h2)), h2


# ── Environment ──────────────────────────────────────────────────────────────
def env_step_torch(pos, action, beta, gamma):
    n   = torch.norm(action) + 1e-8
    ah  = action / n
    ap  = torch.stack([-ah[1], ah[0]])
    return torch.clamp(pos + action + beta * ah + gamma * ap, 0, GRID_SIZE)


# ── Train briefly ─────────────────────────────────────────────────────────────
def train_mc(ec, cb, device, contexts, seed):
    mc  = MotorCortex().to(device)
    opt = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)
    rng  = np.random.default_rng(seed)
    h_ec = ec.init_hidden(1, device)

    for ep in range(N_TRAIN_EP):
        mc.train(); opt.zero_grad()
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
            if torch.norm(pos - g).item() < GOAL_RADIUS: break
            with torch.no_grad():
                cb_h = cb.hidden(pos.unsqueeze(0), torch.zeros(1,2,device=device) if t==0 else prev_a, z_t)
            a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g.unsqueeze(0), h_mc, cb_h)
            s_next    = env_step_torch(pos, a_t.squeeze(0), beta, gamma)
            losses.append(torch.sum((g - s_next) ** 2))
            with torch.no_grad():
                s_tm1     = torch.zeros(1,2,device=device) if t==0 else pos.unsqueeze(0)
                eps       = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
                z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)
            prev_a = a_t.detach(); pos = s_next.detach()
        if losses:
            torch.stack(losses).sum().backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            opt.step()
        h_ec = h_ec.detach()
    return mc


# ── Diagnose ─────────────────────────────────────────────────────────────────
def diagnose(ec, cb, mc, device, contexts, seed):
    ec.eval(); cb.eval(); mc.eval()
    rng  = np.random.default_rng(seed)
    h_ec = ec.init_hidden(1, device)

    logs = {
        'action_mag': [], 'net_disp': [],
        'dist_before': [], 'dist_after': [], 'overshot': [],
        'final_step_dist': [],   # dist to goal on the step before success/failure
        'success': [],           # per episode
        'success_by_dist': [],   # (dist_before_last_step, success) per episode
    }

    with torch.no_grad():
        for _ in range(N_DIAG_EP):
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

            success      = False
            last_dist    = None

            for t in range(MAX_STEPS):
                dist_before = np.linalg.norm(pos.cpu().numpy() - goal)
                last_dist   = dist_before

                if dist_before < GOAL_RADIUS:
                    success = True; break

                cb_h      = cb.hidden(pos.unsqueeze(0), torch.zeros(1,2,device=device) if t==0 else prev_a, z_t)
                a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g, h_mc, cb_h)
                s_next    = env_step_torch(pos, a_t.squeeze(0), beta, gamma)

                dist_after = np.linalg.norm(s_next.cpu().numpy() - goal)
                net_disp   = np.linalg.norm((s_next - pos).cpu().numpy())
                action_mag = np.linalg.norm(a_t.squeeze(0).cpu().numpy())

                logs['action_mag'].append(action_mag)
                logs['net_disp'].append(net_disp)
                logs['dist_before'].append(dist_before)
                logs['dist_after'].append(dist_after)
                logs['overshot'].append(float(dist_after > dist_before))

                s_tm1     = torch.zeros(1,2,device=device) if t==0 else pos.unsqueeze(0)
                eps       = s_next.unsqueeze(0) - cb(pos.unsqueeze(0), a_t, z_t)
                z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)
                prev_a = a_t; pos = s_next

            logs['success'].append(float(success))
            if last_dist is not None:
                logs['final_step_dist'].append(last_dist)
                logs['success_by_dist'].append((last_dist, float(success)))

    logs['success_by_dist'] = np.array(logs['success_by_dist'])
    return {k: np.array(v) if k != 'success_by_dist' else v for k, v in logs.items()}


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_diagnostics(logs_diag, logs_anti):
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── Row 1: per-step metrics ───────────────────────────────────────────────
    metrics = [
        ('action_mag',  '||action||',           False),
        ('net_disp',    '||net displacement||',  False),
        ('dist_before', 'dist to goal (before)', False),
        ('dist_after',  'dist to goal (after)',  False),
        ('overshot',    'overshot rate',         True),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))

    for ax, (key, xlabel, is_rate) in zip(axes[0], metrics):
        d = logs_diag[key]; a = logs_anti[key]
        if is_rate:
            ax.bar(['diagonal', 'anti-diag'], [d.mean(), a.mean()], color=['steelblue', 'tomato'])
            ax.set_ylabel('Rate')
        else:
            ax.hist(d, bins=40, alpha=0.6, color='steelblue', label='diagonal',  density=True)
            ax.hist(a, bins=40, alpha=0.6, color='tomato',    label='anti-diag', density=True)
            ax.axvline(d.mean(), color='steelblue', lw=2, ls='--', label=f'diag={d.mean():.2f}')
            ax.axvline(a.mean(), color='tomato',    lw=2, ls='--', label=f'anti={a.mean():.2f}')
            ax.legend(fontsize=7)
        ax.set_title(xlabel, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    # ── Row 2: final-step dist + success conditioned on proximity ─────────────
    # Panel 1: histogram of dist on last step before success/timeout
    ax = axes[1][0]
    ax.hist(logs_diag['final_step_dist'], bins=30, alpha=0.6, color='steelblue', density=True, label=f'diag mean={logs_diag["final_step_dist"].mean():.2f}')
    ax.hist(logs_anti['final_step_dist'], bins=30, alpha=0.6, color='tomato',    density=True, label=f'anti mean={logs_anti["final_step_dist"].mean():.2f}')
    ax.axvline(GOAL_RADIUS, color='black', lw=1.5, ls='--', label=f'goal radius={GOAL_RADIUS}')
    ax.set_title('Dist to goal at last step', fontsize=9)
    ax.legend(fontsize=7); ax.spines[['top', 'right']].set_visible(False)

    # Panel 2: overall success rate
    ax = axes[1][1]
    ax.bar(['diagonal', 'anti-diag'],
           [logs_diag['success'].mean(), logs_anti['success'].mean()],
           color=['steelblue', 'tomato'])
    ax.set_title('Overall success rate', fontsize=9)
    ax.set_ylim(0, 1); ax.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate([logs_diag['success'].mean(), logs_anti['success'].mean()]):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    # Panel 3: success rate by proximity bins
    ax = axes[1][2]
    bins = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    bin_labels = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-3', '3+']
    x = np.arange(len(bin_labels)); width = 0.35
    for logs, color, label in [(logs_diag, 'steelblue', 'diagonal'), (logs_anti, 'tomato', 'anti-diag')]:
        sbd = np.array(logs['success_by_dist'])
        rates = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (sbd[:, 0] >= lo) & (sbd[:, 0] < hi)
            rates.append(sbd[mask, 1].mean() if mask.sum() > 0 else np.nan)
        offset = -width/2 if label == 'diagonal' else width/2
        ax.bar(x + offset, rates, width, color=color, alpha=0.8, label=label)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels, fontsize=7)
    ax.set_title('Success rate by dist-to-goal\nat last step', fontsize=9)
    ax.set_ylabel('Success rate'); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7); ax.spines[['top', 'right']].set_visible(False)

    # Panels 4-5: empty
    for ax in axes[1][3:]: ax.axis('off')

    fig.suptitle(f'FF diagnostics: diagonal vs anti-diagonal | {N_DIAG_EP} episodes', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'debug_ff.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')

    # Print summary
    print('\n--- Summary ---')
    for key, label, _ in metrics:
        print(f'{label:30s}  diag={logs_diag[key].mean():.3f}  anti={logs_anti[key].mean():.3f}')
    print(f'{"success rate":30s}  diag={logs_diag["success"].mean():.3f}  anti={logs_anti["success"].mean():.3f}')
    print(f'{"final step dist":30s}  diag={logs_diag["final_step_dist"].mean():.3f}  anti={logs_anti["final_step_dist"].mean():.3f}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    ckpt = torch.load(CKPT, map_location=device)
    ec   = EntorhinalCortex().to(device)
    cb   = CerebellumCB().to(device)
    ec.load_state_dict(ckpt['ec'])
    cb_sd = ckpt['cb']; remap = {'net.0': 'layer1.0', 'net.2': 'layer2.0', 'net.4': 'out'}
    new_sd = {}
    for k, v in cb_sd.items():
        prefix = '.'.join(k.split('.')[:2])
        new_sd[k.replace(prefix, remap[prefix]) if prefix in remap else k] = v
    cb.load_state_dict(new_sd)
    ec.eval(); cb.eval()

    print('Training MC on diagonal...')
    mc_diag = train_mc(ec, cb, device, DIAG_CONTEXT, seed=42)
    print('Training MC on anti-diagonal...')
    mc_anti = train_mc(ec, cb, device, ANTIDIAG_CONTEXT, seed=42)

    print('Diagnosing diagonal...')
    logs_diag = diagnose(ec, cb, mc_diag, device, DIAG_CONTEXT, seed=99)
    print('Diagnosing anti-diagonal...')
    logs_anti = diagnose(ec, cb, mc_anti, device, ANTIDIAG_CONTEXT, seed=99)

    plot_diagnostics(logs_diag, logs_anti)
    print('Done.')


if __name__ == '__main__':
    main()

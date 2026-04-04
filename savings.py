"""
Savings paradigm — ABA training protocol

Phase A1: Train MC on diagonal FF     (0.8,  0.8)
Phase B:  Train MC on anti-diagonal FF(-0.8, -0.8)
Phase A2: Train MC on diagonal FF     (0.8,  0.8)

MC weights update throughout all phases (no freezing).
EC and CB stay frozen.

Savings = if A2 learning curve rises faster -- starts higher than A1.
Compared for EC intact vs EC lesion.

Figures saved to figs/ECtoCB/gridworld_policy/learning_MC_SAVINGS/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyperparameters ─────────────────────────────────────────────────────────────
LATENT_DIM  = 16
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
MC_HIDDEN   = 256
MC_INPUT    = 36   # s_t(2) + z'_ec(16) + z_cb(16) + g(2)
CB_PROJ_DIM = 16

MAX_STEPS      = 50
N_PHASE_EP     = 2000   # training episodes per phase
GOAL_RADIUS    = 0.3
GRID_SIZE      = 4
LR_MC          = 1e-3
EVAL_INTERVAL  = 1      # evaluate every N training episodes
N_CURVE_EP     = 1      # episodes per eval checkpoint

FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/gridworld_policy/learning_MC_SAVINGS"
CKPT    = "checkpoints/ectocb_mse_shiftedv2.pt"

A_CONTEXT = [(0.8,  0.8)]   # diagonal
B_CONTEXT = [(-0.8, -0.8)]  # anti-diagonal


# ── Models ────────
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


# ── Environment ──────────────────────────────────────────────────────────────────
def env_step_torch(pos, action, beta, gamma):
    a_norm = torch.norm(action) + 1e-8
    a_hat  = action / a_norm
    a_perp = torch.stack([-a_hat[1], a_hat[0]])
    f_env  = beta * a_hat + gamma * a_perp
    return torch.clamp(pos + action + f_env, 0, GRID_SIZE)


# ── Single training episode (h_ec carries in and out) ───────────────────────────
def run_episode(ec, cb, mc, device, rng, contexts, h_ec, ec_lesion=False):
    """Returns (loss, h_ec_out). h_ec carries across episodes."""
    start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    while np.array_equal(start, goal):
        goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    beta, gamma = contexts[rng.integers(0, len(contexts))]

    g_tensor = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos      = torch.tensor(start, dtype=torch.float32, device=device)

    h_mc   = mc.init_hidden(1, device)
    z_t    = torch.zeros(1, LATENT_DIM, device=device)
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
            if not ec_lesion:
                s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
                z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()

    h_ec = h_ec.detach()
    if not step_losses:
        return torch.tensor(0.0, device=device, requires_grad=True), h_ec
    return torch.stack(step_losses).sum(), h_ec


# ── Evaluate (MC frozen, h_ec carries in and out) ───────────────────────────────
def evaluate(ec, cb, mc, device, contexts, h_ec, ec_lesion=False, n_ep=None, seed=42):
    """Returns (results, h_ec_out). h_ec carries across eval episodes."""
    ec.eval(); cb.eval(); mc.eval()
    rng     = np.random.default_rng(seed)
    n_ep    = n_ep if n_ep is not None else N_CURVE_EP
    results = {"success": [], "steps": [], "norm_path": []}

    with torch.no_grad():
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
            success      = False
            steps        = 0
            init_dist    = np.linalg.norm(start - goal)
            total_dist   = 0.0

            for t in range(MAX_STEPS):
                if np.linalg.norm(pos.cpu().numpy() - goal) < GOAL_RADIUS:
                    success = True; break

                total_dist += np.linalg.norm(pos.cpu().numpy() - goal)

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

            norm_path = (total_dist / steps / init_dist) if (steps > 0 and init_dist > 0) else 0.0
            results["success"].append(float(success))
            results["steps"].append(steps)
            results["norm_path"].append(norm_path)

    return results, h_ec


# ── Train one phase (h_ec carries in and out across train + eval) ────────────────
def train_phase(ec, cb, mc, optimizer, device, contexts, phase_name, h_ec, ec_lesion=False, seed=0):
    """
    Train MC for N_PHASE_EP episodes. h_ec carries across training and eval episodes.
    Returns (curve, h_ec_out).
    """
    ec.eval(); cb.eval()
    rng = np.random.default_rng(seed)

    curve = {'success': [], 'steps': [], 'norm_path': [], 'ep': []}
    print(f"\n  Phase {phase_name} ({'EC lesion' if ec_lesion else 'EC intact'}) "
          f"— {N_PHASE_EP} episodes, context={contexts[0]}")

    for ep in range(1, N_PHASE_EP + 1):
        mc.train()
        optimizer.zero_grad()
        loss, h_ec = run_episode(ec, cb, mc, device, rng, contexts, h_ec, ec_lesion=ec_lesion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), max_norm=1.0)
        optimizer.step()

        if ep % EVAL_INTERVAL == 0:
            r, h_ec = evaluate(ec, cb, mc, device, contexts, h_ec,
                               ec_lesion=ec_lesion, n_ep=N_CURVE_EP, seed=ep)
            curve['success'].extend(r['success'])
            curve['steps'].extend(r['steps'])
            curve['norm_path'].extend(r['norm_path'])
            curve['ep'].extend([ep] * N_CURVE_EP)

            if ep % 500 == 0:
                print(f"    ep {ep:5d} | success={np.mean(curve['success'][-500:]):.2f} "
                      f"steps={np.mean(curve['steps'][-500:]):.1f}")

    return curve, h_ec


# ── Full ABA savings run ─────────────────────────────────────────────────────────
def run_aba(ec, cb, device, ec_lesion=False, seed=0):
    """
    Train MC from scratch through A1 → B → A2.
    Returns curves for all three phases.
    """
    mc        = MotorCortex().to(device)
    optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)

    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)

    # h_ec resets once at session start, then carries across all 3 phases
    h_ec = torch.zeros(1, EC_HIDDEN, device=device)

    curve_a1, h_ec = train_phase(ec, cb, mc, optimizer, device, A_CONTEXT, 'A1', h_ec, ec_lesion, seed)
    curve_b,  h_ec = train_phase(ec, cb, mc, optimizer, device, B_CONTEXT, 'B',  h_ec, ec_lesion, seed+1)
    curve_a2, h_ec = train_phase(ec, cb, mc, optimizer, device, A_CONTEXT, 'A2', h_ec, ec_lesion, seed+2)

    return curve_a1, curve_b, curve_a2


ROLLING_WIN = 20    # rolling mean window over test episodes (applied per phase)


def rolling_mean(x, w):
    """Rolling mean with min_periods=1 so phase start is not cut off."""
    x = np.array(x)
    result = np.zeros(len(x))
    for i in range(len(x)):
        start = max(0, i - w + 1)
        result[i] = np.mean(x[start:i + 1])
    return result


# ── Plotting ─────────────────────────────────────────────────────────────────────
def plot_savings(curves_ec):
    os.makedirs(FIG_DIR, exist_ok=True)

    phase_labels = ['A1: diagonal', 'B: anti-diagonal', 'A2: diagonal']
    colors_bg    = ['#d0e8f5', '#fde8d8', '#d0e8f5']

    fig, ax = plt.subplots(figsize=(10, 4))
    offset = 0

    for i, (label, bg) in enumerate(zip(phase_labels, colors_bg)):
        vals  = np.array(curves_ec[i]['norm_path'])
        n_pts = len(vals)
        x     = np.arange(offset, offset + n_pts)

        ax.axvspan(offset, offset + n_pts, color=bg, alpha=0.35, zorder=0)
        ax.axvline(offset, color='gray', lw=1, ls='--', zorder=1)
        ax.text(offset + n_pts / 2, 1.0, label, ha='center', fontsize=8,
                color='dimgray', transform=ax.get_xaxis_transform())
        ax.plot(x, vals, color='steelblue', lw=1, alpha=0.8,
                label='EC intact' if i == 0 else None)

        offset += n_pts

    ax.set_xlabel('Trial (test episode)')
    ax.set_ylabel('Normalised path cost\n(mean dist to goal per step / init dist)')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Savings paradigm: diagonal (A1) → anti-diagonal (B) → diagonal (A2)\n'
                 'EC intact | normalised path cost', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'savings_aba_normpath.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')



# ── Main ─────────────────────────────────────────────────────────────────────────
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

    print('\n=== EC intact ===')
    curves_ec = run_aba(ec, cb, device, ec_lesion=False, seed=0)

    os.makedirs(FIG_DIR, exist_ok=True)
    np.save(os.path.join(FIG_DIR, 'curves_ec.npy'), curves_ec, allow_pickle=True)
    print('Curves saved.')

    plot_savings(curves_ec)

    # Print summary: A1 vs A2 final performance
    a1_final = curves_ec[0]['norm_path'][-1]
    a2_start = curves_ec[2]['norm_path'][0]
    a2_final = curves_ec[2]['norm_path'][-1]
    print(f'\nEC intact:')
    print(f'  A1 final norm_path: {a1_final:.2f}')
    print(f'  A2 start norm_path: {a2_start:.2f}')
    print(f'  A2 final norm_path: {a2_final:.2f}')

    print('\nDone.')


if __name__ == '__main__':
    main()

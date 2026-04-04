"""
learning_MC_continuous.py

MC with continuous actions trained on familiar axis-aligned force fields (±0.8).

Loss: ||g - s_{t+1}||^2
      where s_{t+1} = env_step_torch(s_t, a_t, beta, gamma)
      and   a_t     = tanh(MC(s_t, z', g))  — continuous (dx, dy) in (-1, 1)^2

Gradient flows: loss -> s_{t+1} -> env_step_torch -> a_t -> MC weights.
No argmax, no discrete sampling, no REINFORCE needed.

EC and CB are frozen (loaded from checkpoint).
MC input: s_t (2-D), z'_ec (16-D), z_cb (16-D)(read from h_ec via mc.ec_readout), g (2-D)

Evaluate 2x2:
         | EC intact | EC lesion (z=0) |
Familiar |     A     |        B        |
Novel    |     C     |        D        |

Figures saved to figs/ECtoCB/gridworld_policy/learning_MC_continuous/
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
CB_PROJ_DIM = 16   # random projection of CB hidden activity

MAX_STEPS     = 50
N_TRAIN_EP    = 15000
N_EVAL_EP     = 500
GOAL_RADIUS   = 0.3
GRID_SIZE     = 4
LR_MC         = 1e-3
EVAL_INTERVAL = 100
N_CURVE_EP    = 100
GAMMA_DISCOUNT = 0.95
SWITCH_STEP    = 25   # step at which FF switches from novel to familiar in savings test

FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/gridworld_policy/learning_MC_CBproj"
CKPT    = "checkpoints/ectocb_mse_shiftedv2.pt"

# Familiar — axis-aligned ±0.8
CONTEXTS = [
    ( 0.8,  0.0),
    (-0.8,  0.0),
    ( 0.0,  0.8),
    ( 0.0, -0.8),
]

# Novel — diagonal ±0.8, never seen during EC+CB training
NOVEL_CONTEXTS = [
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
        """Return second hidden layer activations (256-D)."""
        h = self.layer1(torch.cat([s, a, z], dim=-1))
        return self.layer2(h)


class MotorCortex(nn.Module):
    """
    Continuous-action MC with EC z' + random CB projection as inputs.
    Output: tanh(linear(h)) in (-1, 1)^2 — a continuous (dx, dy) action.
    """
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell       = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.readout    = nn.Linear(MC_HIDDEN, 2)

        # fixed random projection CB_HIDDEN → CB_PROJ_DIM (not trained)
        W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
        self.register_buffer('cb_proj', W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, h_ec, g, h, cb_hidden):
        z_ec  = torch.tanh(self.ec_readout(h_ec))          # (1, 16)
        z_cb  = cb_hidden @ self.cb_proj                    # (1, 16) random projection
        h_new = self.cell(torch.cat([s_t, z_ec, z_cb, g], dim=-1), h)
        a_t   = torch.tanh(self.readout(h_new))
        return a_t, h_new


# ── Environment ────────────────────────────────────────────────────────────────
def env_step_torch(pos, action, beta, gamma):
    """
    Differentiable PyTorch env step.
    pos:    tensor (2,)
    action: tensor (2,)  continuous (dx, dy)
    returns tensor (2,)  next position
    """
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
    beta, gamma = CONTEXTS[rng.integers(0, len(CONTEXTS))]

    g_tensor = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos      = torch.tensor(start, dtype=torch.float32, device=device)

    h_ec = ec.init_hidden(1, device)
    h_mc = mc.init_hidden(1, device)
    z_t  = torch.zeros(1, LATENT_DIM, device=device)

    step_losses = []
    prev_a = torch.zeros(1, 2, device=device)   # a_{t-1} for EC input

    for t in range(MAX_STEPS):
        if torch.norm(pos - g_tensor).item() < GOAL_RADIUS:
            break

        # ── CB hidden activity → random projection ────────────────────────────
        with torch.no_grad():
            cb_hidden = cb.hidden(pos.unsqueeze(0),
                                  torch.zeros(1, 2, device=device) if t == 0 else prev_a,
                                  z_t)  # (1, 256)

        # ── MC forward — continuous action (differentiable) ───────────────────
        a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g_tensor.unsqueeze(0), h_mc, cb_hidden)
        # a_t: (1, 2) continuous action

        # ── Next state via differentiable env step ────────────────────────────
        s_next = env_step_torch(pos, a_t.squeeze(0), beta, gamma)  # (2,)

        # ── Loss: distance between real next state and goal ───────────────────
        step_losses.append(torch.sum((g_tensor - s_next) ** 2))

        # ── EC update (no grad needed) ────────────────────────────────────────
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
        'ep':                    [],
        'fam_ec_success':        [], 'fam_ec_steps':        [],
        'fam_lesion_success':    [], 'fam_lesion_steps':    [],
        'novel_ec_success':      [], 'novel_ec_steps':      [],
        'novel_lesion_success':  [], 'novel_lesion_steps':  [],
    }

    print(f"\nTraining MC (continuous actions) on familiar ±0.8 ({N_TRAIN_EP} episodes)...")
    for ep in range(1, N_TRAIN_EP + 1):
        optimizer.zero_grad()
        loss = run_mc_episode(ec, cb, mc, device, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if ep % EVAL_INTERVAL == 0:
            mc.eval()
            r_fe = evaluate(ec, cb, mc, device, CONTEXTS,       seed=ep, ec_lesion=False, n_ep=N_CURVE_EP)
            r_fl = evaluate(ec, cb, mc, device, CONTEXTS,       seed=ep, ec_lesion=True,  n_ep=N_CURVE_EP)
            r_ne = evaluate(ec, cb, mc, device, NOVEL_CONTEXTS, seed=ep, ec_lesion=False, n_ep=N_CURVE_EP)
            r_nl = evaluate(ec, cb, mc, device, NOVEL_CONTEXTS, seed=ep, ec_lesion=True,  n_ep=N_CURVE_EP)
            curve['ep'].append(ep)
            curve['fam_ec_success'].append(np.mean(r_fe['success']))
            curve['fam_ec_steps'].append(np.mean(r_fe['steps']))
            curve['fam_lesion_success'].append(np.mean(r_fl['success']))
            curve['fam_lesion_steps'].append(np.mean(r_fl['steps']))
            curve['novel_ec_success'].append(np.mean(r_ne['success']))
            curve['novel_ec_steps'].append(np.mean(r_ne['steps']))
            curve['novel_lesion_success'].append(np.mean(r_nl['success']))
            curve['novel_lesion_steps'].append(np.mean(r_nl['steps']))
            mc.train()

            if ep % 300 == 0:
                print(f"  Ep {ep:5d} | loss={np.mean(losses[-300:]):.3f} | "
                      f"fam_acc={curve['fam_ec_success'][-1]:.2f} "
                      f"nov_acc={curve['novel_ec_success'][-1]:.2f}")

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
        success  = False
        steps    = 0
        prev_a   = torch.zeros(1, 2, device=device)

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


# ── Savings evaluation ─────────────────────────────────────────────────────────
def evaluate_savings(ec, cb, mc, device, seed=42, n_ep=None):
    """
    Savings paradigm — single episode, FF switches mid-way:
      steps 0..SWITCH_STEP-1 : novel diagonal FF
      steps SWITCH_STEP..end : familiar axis-aligned FF

    Three conditions:
      EC intact  — h_ec carries over from novel phase into familiar phase
      EC lesion  — z'=0 throughout
      EC reset   — h_ec zeroed at switch point (control: no carry-over)

    Reports success rate and mean steps in the FAMILIAR phase only.
    """
    ec.eval(); cb.eval(); mc.eval()
    rng  = np.random.default_rng(seed)
    n_ep = n_ep if n_ep is not None else N_EVAL_EP

    results = {c: {'success': [], 'steps': []} for c in ['intact', 'lesion', 'reset']}

    for _ in range(n_ep):
        start         = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        goal          = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        while np.array_equal(start, goal):
            goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        novel_ctx  = NOVEL_CONTEXTS [rng.integers(0, len(NOVEL_CONTEXTS))]
        fam_ctx    = CONTEXTS       [rng.integers(0, len(CONTEXTS))]

        g_tensor = torch.tensor(goal,  dtype=torch.float32, device=device).unsqueeze(0)

        for cond in ['intact', 'lesion', 'reset']:
            pos    = torch.tensor(start, dtype=torch.float32, device=device)
            h_ec   = torch.zeros(1, EC_HIDDEN, device=device)
            h_mc   = mc.init_hidden(1, device)
            z_t    = torch.zeros(1, LATENT_DIM, device=device)
            prev_a = torch.zeros(1, 2, device=device)
            success = False
            steps   = 0

            with torch.no_grad():
                for t in range(MAX_STEPS):
                    # pick FF based on phase
                    beta, gamma = novel_ctx if t < SWITCH_STEP else fam_ctx

                    # reset EC at switch point for 'reset' condition
                    if cond == 'reset' and t == SWITCH_STEP:
                        h_ec = torch.zeros(1, EC_HIDDEN, device=device)
                        z_t  = torch.zeros(1, LATENT_DIM, device=device)

                    # check goal only in familiar phase
                    if t >= SWITCH_STEP:
                        if np.linalg.norm(pos.numpy() - goal) < GOAL_RADIUS:
                            success = True; break
                        steps += 1

                    cb_hidden = cb.hidden(pos.unsqueeze(0),
                                          torch.zeros(1, 2, device=device) if t == 0 else prev_a,
                                          z_t)
                    a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec, g_tensor, h_mc, cb_hidden)
                    s_next    = env_step_torch(pos, a_t.squeeze(0), beta, gamma)

                    if cond != 'lesion':
                        s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                        eps   = s_next.unsqueeze(0) - cb(pos.unsqueeze(0), a_t, z_t)
                        z_t, h_ec = ec.step(torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

                    prev_a = a_t
                    pos    = s_next

            results[cond]['success'].append(float(success))
            results[cond]['steps'].append(steps)

    return results


def plot_savings(results):
    os.makedirs(FIG_DIR, exist_ok=True)

    labels  = ['EC intact\n(h_ec carries over)', 'EC lesion\n(z\'=0)', 'EC reset\n(h_ec zeroed at switch)']
    keys    = ['intact', 'lesion', 'reset']
    colors  = ['steelblue', 'tomato', 'goldenrod']

    success = [np.mean(results[k]['success']) for k in keys]
    steps   = [np.mean(results[k]['steps'])   for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, vals, ylabel in zip(axes, [success, steps], ['Success rate (familiar phase)', 'Mean steps (familiar phase)']):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, vals):
            fmt = f'{val:.2f}' if 'Success' in ylabel else f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.02 if 'Success' in ylabel else 0.3),
                    fmt, ha='center', fontsize=9)
        ax.set_ylabel(ylabel)
        if 'Success' in ylabel:
            ax.set_ylim(0, 1.15)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'Savings test — novel FF (steps 0-{SWITCH_STEP-1}) → familiar FF (steps {SWITCH_STEP}-{MAX_STEPS})\n'
                 'Performance measured in familiar phase only', fontsize=10)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'savings.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {out}')


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_learning_curves(curve):
    os.makedirs(FIG_DIR, exist_ok=True)
    ep = curve['ep']

    styles = {
        ('fam',   'ec'):     dict(color='steelblue', ls='-',  label='Familiar — EC intact'),
        ('fam',   'lesion'): dict(color='steelblue', ls='--', label='Familiar — EC lesion'),
        ('novel', 'ec'):     dict(color='tomato',    ls='-',  label='Novel — EC intact'),
        ('novel', 'lesion'): dict(color='tomato',    ls='--', label='Novel — EC lesion'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (ctx, cond), kw in styles.items():
        axes[0].plot(ep, curve[f'{ctx}_{cond}_success'], lw=2, **kw)
        axes[1].plot(ep, curve[f'{ctx}_{cond}_steps'],   lw=2, **kw)

    axes[0].set_ylabel('Success rate'); axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel('Mean steps to goal')
    for ax in axes:
        ax.set_xlabel('Training episode')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    fig.suptitle('Learning curves — continuous actions, familiar vs novel × EC intact vs lesion', fontsize=11)
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
    ax.set_title('MC training loss — ||g − s_{t+1}||²  continuous actions, ±0.8 FF')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'mc_training_loss.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'Saved {out}')


def plot_2x2(results):
    os.makedirs(FIG_DIR, exist_ok=True)

    labels_x   = ['Familiar FF\n(±0.8)', 'Novel FF\n(±0.8 diagonal)']
    bar_labels = ['EC intact', 'EC lesion']
    colors     = ['steelblue', 'tomato']
    x          = np.arange(len(labels_x))
    w          = 0.35

    success = {
        'EC intact': [np.mean(results['familiar_ec']['success']),
                      np.mean(results['novel_ec']['success'])],
        'EC lesion': [np.mean(results['familiar_lesion']['success']),
                      np.mean(results['novel_lesion']['success'])],
    }
    steps = {
        'EC intact': [np.mean(results['familiar_ec']['steps']),
                      np.mean(results['novel_ec']['steps'])],
        'EC lesion': [np.mean(results['familiar_lesion']['steps']),
                      np.mean(results['novel_lesion']['steps'])],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, ylabel in zip(axes,
                                   [success, steps],
                                   ['Success rate', 'Mean steps to goal']):
        for i, (lbl, color) in enumerate(zip(bar_labels, colors)):
            vals = metric[lbl]
            bars = ax.bar(x + (i - 0.5) * w, vals, w,
                          label=lbl, color=color, alpha=0.85)
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

    fig.suptitle('EC lesion — continuous actions, ±0.8 FF\n'
                 'Loss: ||g − s_{t+1}||²  (real next state, not expectation)',
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, '2x2_EC_lesion.png')
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
    # remap checkpoint keys from nn.Sequential naming to split layer naming
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

    print('\nEvaluating 4 conditions...')
    results = {}
    conds = [
        ('familiar_ec',     CONTEXTS,       False),
        ('familiar_lesion', CONTEXTS,       True),
        ('novel_ec',        NOVEL_CONTEXTS, False),
        ('novel_lesion',    NOVEL_CONTEXTS, True),
    ]
    for key, ctx, lesion in conds:
        r = evaluate(ec, cb, mc, device, ctx, ec_lesion=lesion)
        results[key] = r
        tag = 'EC intact' if not lesion else 'EC lesion'
        ff  = 'familiar' if ctx is CONTEXTS else 'novel   '
        print(f'  {ff} | {tag:10s} | '
              f'success={np.mean(r["success"]):.3f}  steps={np.mean(r["steps"]):.1f}')

    plot_2x2(results)

    print('\nRunning savings test...')
    sav = evaluate_savings(ec, cb, mc, device)
    for cond, label in [('intact','EC intact'), ('lesion','EC lesion'), ('reset','EC reset')]:
        print(f'  {label:25s} | success={np.mean(sav[cond]["success"]):.3f}  steps={np.mean(sav[cond]["steps"]):.1f}')
    plot_savings(sav)
    print('\nDone.')


if __name__ == '__main__':
    main()

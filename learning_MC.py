"""
learning_MC.py

MC trained on familiar axis-aligned force fields (±0.4).

Loss: ||g - E[s_{t+1}]||^2
      where E[s_{t+1}] = sum_a pi(a) * env_step_torch(s_t, a, beta, gamma)

      Expected next position under MC's policy, computed by running the
      (pytorch) env_step for all 4 actions and weighting by pi.
      Gradient flows: loss -> E[s_{t+1}] -> pi -> MC weights.

EC and CB are frozen (loaded from checkpoint).
MC input: s_t, z' (read from h_ec via mc.ec_readout), g

Evaluate 2x2:
         | EC intact | EC lesion (z=0) |
Familiar |     A     |        B        |
Novel    |     C     |        D        |

Figures saved to figs/ECtoCB/gridworld_policy/learning_MC/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyperparameters 
LATENT_DIM  = 16
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
MC_HIDDEN   = 256
MC_INPUT    = 20

MAX_STEPS   = 50
N_TRAIN_EP  = 15000
N_EVAL_EP   = 500
GOAL_RADIUS = 0.3
GRID_SIZE   = 4
LR_MC       = 1e-3
EVAL_INTERVAL = 100   # evaluate every N training episodes
N_CURVE_EP    = 100   # episodes per curve evaluation point

FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/gridworld_policy/learning_MC_08_discount"
CKPT    = "checkpoints/ectocb_mse_shiftedv2.pt"

ACTIONS_NP = [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]
N_ACTIONS  = len(ACTIONS_NP)

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

GAMMA_DISCOUNT = 0.95   # discount factor for per-step loss


# ── Models
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
        self.net = nn.Sequential(
            nn.Linear(4 + LATENT_DIM, CB_HIDDEN), nn.ReLU(),
            nn.Linear(CB_HIDDEN, CB_HIDDEN),       nn.ReLU(),
            nn.Linear(CB_HIDDEN, 2),
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))


class MotorCortex(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell       = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.readout    = nn.Linear(MC_HIDDEN, N_ACTIONS)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, h_ec, g, h):
        z_prime = torch.tanh(self.ec_readout(h_ec))
        h_new   = self.cell(torch.cat([s_t, z_prime, g], dim=-1), h)
        pi      = torch.softmax(self.readout(h_new), dim=-1)
        return pi, h_new


# ── Environment ──
def env_step_numpy(x, y, dx, dy, beta, gamma):
    """Original numpy env step — used for real transitions during EC update."""
    a      = np.array([dx, dy], dtype=np.float32)
    a_norm = np.linalg.norm(a)
    a_hat  = a / a_norm if a_norm > 0 else np.zeros(2, dtype=np.float32)
    a_perp = np.array([-a_hat[1], a_hat[0]], dtype=np.float32)
    f_env  = beta * a_hat + gamma * a_perp
    return (float(np.clip(x + dx + f_env[0], 0, GRID_SIZE)),
            float(np.clip(y + dy + f_env[1], 0, GRID_SIZE)))


def env_step_torch(pos, action, beta, gamma):
    """
    PyTorch env step — used for the differentiable loss.

    pos:    tensor (2,)  current position
    action: tensor (2,)  dx, dy
    returns tensor (2,)  next position

    Gradient flows through this back to pi (MC's output).
    """
    a_norm = torch.norm(action) + 1e-8
    a_hat  = action / a_norm
    a_perp = torch.stack([-a_hat[1], a_hat[0]])
    f_env  = beta * a_hat + gamma * a_perp
    return torch.clamp(pos + action + f_env, 0, GRID_SIZE)


# ── Training 
def run_mc_episode(ec, cb, mc, device, rng):
    start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    while np.array_equal(start, goal):
        goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    beta, gamma = CONTEXTS[rng.integers(0, len(CONTEXTS))]

    g_tensor = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos      = torch.tensor(start, dtype=torch.float32, device=device)

    # all 4 action vectors as tensors — reused each step
    actions_torch = torch.tensor(ACTIONS_NP, dtype=torch.float32, device=device)  # (4,2)

    h_ec = ec.init_hidden(1, device)
    h_mc = mc.init_hidden(1, device)
    z_t  = torch.zeros(1, LATENT_DIM, device=device)

    step_losses = []

    for t in range(MAX_STEPS):
        if torch.norm(pos - g_tensor).item() < GOAL_RADIUS:
            break

        # ── MC forward (differentiable) 
        pi, h_mc = mc.step(pos.unsqueeze(0), h_ec, g_tensor.unsqueeze(0), h_mc)
        # pi: (1, 4) — action probabilities

        # ── Expected next state under pi ─────────────────────────────────────
        # Run env_step_torch for each of the 4 actions
        next_states = torch.stack([
            env_step_torch(pos, actions_torch[a], beta, gamma)
            for a in range(N_ACTIONS)
        ])  # (4, 2)

        # Weight each next state by its probability → expected next position
        E_next = (pi.squeeze(0).unsqueeze(1) * next_states).sum(0)  # (2,)

        # Loss: distance between expected next state and goal
        step_losses.append((GAMMA_DISCOUNT ** t) * torch.sum((g_tensor - E_next) ** 2))

        # ── Hard action for real env step + EC update (no grad needed) ───────
        with torch.no_grad():
            action_idx     = torch.argmax(pi.squeeze(0)).item()
            dx, dy         = ACTIONS_NP[action_idx]
            x_next, y_next = env_step_numpy(pos[0].item(), pos[1].item(),
                                            dx, dy, beta, gamma)
            pos_next = torch.tensor([x_next, y_next], dtype=torch.float32, device=device)

            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            a_tm1 = torch.zeros(1, 2, device=device) if t == 0 else \
                    torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
            a_t   = torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
            eps   = pos_next.unsqueeze(0) - cb(pos.unsqueeze(0), a_t, z_t)
            z_t, h_ec = ec.step(torch.cat([s_tm1, a_tm1, pos.unsqueeze(0), eps], dim=-1), h_ec)

        pos = pos_next

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

    # learning curve buffers: keyed by (context_set, ec_lesion)
    curve = {
        'ep':                    [],
        'fam_ec_success':        [], 'fam_ec_steps':        [],
        'fam_lesion_success':    [], 'fam_lesion_steps':    [],
        'novel_ec_success':      [], 'novel_ec_steps':      [],
        'novel_lesion_success':  [], 'novel_lesion_steps':  [],
    }

    print(f"\nTraining MC on familiar contexts ±0.4 ({N_TRAIN_EP} episodes)...")
    for ep in range(1, N_TRAIN_EP + 1):
        optimizer.zero_grad()
        loss = run_mc_episode(ec, cb, mc, device, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if ep % EVAL_INTERVAL == 0:
            mc.eval()
            r_fe  = evaluate(ec, cb, mc, device, CONTEXTS,       seed=ep, ec_lesion=False, n_ep=N_CURVE_EP)
            r_fl  = evaluate(ec, cb, mc, device, CONTEXTS,       seed=ep, ec_lesion=True,  n_ep=N_CURVE_EP)
            r_ne  = evaluate(ec, cb, mc, device, NOVEL_CONTEXTS, seed=ep, ec_lesion=False, n_ep=N_CURVE_EP)
            r_nl  = evaluate(ec, cb, mc, device, NOVEL_CONTEXTS, seed=ep, ec_lesion=True,  n_ep=N_CURVE_EP)
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
                print(f"  Ep {ep:4d} | loss={np.mean(losses[-300:]):.3f} | "
                      f"fam_acc={curve['fam_ec_success'][-1]:.2f} "
                      f"nov_acc={curve['novel_ec_success'][-1]:.2f}")

    return losses, curve


# ── Evaluation ─────
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

        g_tensor = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
        pos      = torch.tensor(start, dtype=torch.float32, device=device)
        h_ec     = torch.zeros(1, EC_HIDDEN, device=device)
        h_mc     = mc.init_hidden(1, device)
        z_t      = torch.zeros(1, LATENT_DIM, device=device)
        success  = False
        steps    = 0

        with torch.no_grad():
            for t in range(MAX_STEPS):
                if np.linalg.norm(pos.numpy() - goal) < GOAL_RADIUS:
                    success = True; break

                pi, h_mc   = mc.step(pos.unsqueeze(0), h_ec, g_tensor, h_mc)
                action_idx = torch.argmax(pi.squeeze(0)).item()
                dx, dy     = ACTIONS_NP[action_idx]
                x_next, y_next = env_step_numpy(pos[0].item(), pos[1].item(),
                                                dx, dy, beta, gamma)
                pos_next = torch.tensor([x_next, y_next], dtype=torch.float32, device=device)

                if not ec_lesion:
                    s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                    a_tm1 = torch.zeros(1, 2, device=device) if t == 0 else \
                            torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
                    a_t   = torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
                    eps   = pos_next.unsqueeze(0) - cb(pos.unsqueeze(0), a_t, z_t)
                    z_t, h_ec = ec.step(torch.cat([s_tm1, a_tm1, pos.unsqueeze(0), eps], dim=-1), h_ec)

                pos    = pos_next
                steps += 1

        results["success"].append(float(success))
        results["steps"].append(steps)

    return results


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_learning_curves(curve):
    os.makedirs(FIG_DIR, exist_ok=True)
    ep = curve['ep']

    styles = {
        ('fam',   'ec'):     dict(color='steelblue',  ls='-',  label='Familiar — EC intact'),
        ('fam',   'lesion'): dict(color='steelblue',  ls='--', label='Familiar — EC lesion'),
        ('novel', 'ec'):     dict(color='tomato',     ls='-',  label='Novel — EC intact'),
        ('novel', 'lesion'): dict(color='tomato',     ls='--', label='Novel — EC lesion'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for (ctx, cond), kw in styles.items():
        key_s = f'{ctx}_{cond}_success'
        key_t = f'{ctx}_{cond}_steps'
        axes[0].plot(ep, curve[key_s], lw=2, **kw)
        axes[1].plot(ep, curve[key_t], lw=2, **kw)

    axes[0].set_ylabel('Success rate');      axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel('Mean steps to goal')
    for ax in axes:
        ax.set_xlabel('Training episode')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8)

    fig.suptitle('Learning curves — familiar vs novel × EC intact vs lesion', fontsize=11)
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
    ax.set_title('MC training loss — ||g − E[s_{t+1}]||²  (±0.8 familiar FF)')
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

    fig.suptitle('EC lesion experiment — ±0.8 force fields\n'
                 'MC trained with ||g − E[s_{t+1}]||² loss (no teacher)',
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
    cb.load_state_dict(ckpt['cb'])
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
        r   = evaluate(ec, cb, mc, device, ctx, ec_lesion=lesion)
        results[key] = r
        tag = 'EC intact' if not lesion else 'EC lesion'
        ff  = 'familiar' if ctx is CONTEXTS else 'novel   '
        print(f'  {ff} | {tag:10s} | '
              f'success={np.mean(r["success"]):.3f}  steps={np.mean(r["steps"]):.1f}')

    plot_2x2(results)
    print('\nDone.')


if __name__ == '__main__':
    main()

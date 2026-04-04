"""
learning_component_2x2_strongF.py

Same 2x2 EC lesion experiment as learning_component_2x2.py but with stronger
force fields (±0.8) so MC cannot ignore context — weak ±0.4 fields allowed a
context-blind policy to succeed, strong fields force MC to use EC's z_t'.

Train:   CONTEXTS (familiar, axis-aligned, ±0.8)
Evaluate 2x2:
         | EC intact | EC lesion (z=0) |
Familiar |     A     |        B        |
Novel    |     C     |        D        |

Figures saved to figs/ECtoCB/gridworld_policy/2x2_EC_lesion_strongF/
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
MC_INPUT    = 20

MAX_STEPS   = 50
N_TRAIN_EP  = 3000
N_EVAL_EP   = 500
GOAL_RADIUS = 0.5
GRID_SIZE   = 4
LR_MC       = 1e-3

FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/gridworld_policy/2x2_EC_lesion_strongF"
CKPT    = "checkpoints/ectocb_mse_shiftedv2.pt"

ACTIONS   = [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]
N_ACTIONS = len(ACTIONS)

# Familiar — axis-aligned, STRONG (±0.8)
CONTEXTS = [
    ( 0.8,  0.0),
    (-0.8,  0.0),
    ( 0.0,  0.8),
    ( 0.0, -0.8),
]

# Novel — diagonal, STRONG (±0.8), never seen during EC+CB training
NOVEL_CONTEXTS = [
    ( 0.8,  0.8),
    (-0.8,  0.8),
    ( 0.8, -0.8),
    (-0.8, -0.8),
]


# ── Models ────────────────────────────────────────────────────────────────────
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


# ── Environment ───────────────────────────────────────────────────────────────
def env_step(x, y, dx, dy, beta, gamma):
    a      = np.array([dx, dy], dtype=np.float32)
    a_norm = np.linalg.norm(a)
    a_hat  = a / a_norm if a_norm > 0 else np.zeros(2, dtype=np.float32)
    a_perp = np.array([-a_hat[1], a_hat[0]], dtype=np.float32)
    f_env  = beta * a_hat + gamma * a_perp
    return (float(np.clip(x + dx + f_env[0], 0, GRID_SIZE)),
            float(np.clip(y + dy + f_env[1], 0, GRID_SIZE)))


# ── Training (BC on familiar strong contexts only) ────────────────────────────
def run_mc_episode(ec, cb, mc, device, rng):
    start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    while np.array_equal(start, goal):
        goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
    beta, gamma = CONTEXTS[rng.integers(0, len(CONTEXTS))]

    g_tensor    = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
    x, y        = float(start[0]), float(start[1])
    h_ec        = ec.init_hidden(1, device)
    h_mc        = mc.init_hidden(1, device)
    z_t         = torch.zeros(1, LATENT_DIM, device=device)
    step_losses = []

    for t in range(MAX_STEPS):
        s_np = np.array([x, y], dtype=np.float32)
        if np.linalg.norm(s_np - goal) < GOAL_RADIUS:
            break

        s_t    = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
        a_star = int(np.argmin([np.sum((np.array(env_step(x, y, dx, dy, beta, gamma)) - goal)**2)
                                for dx, dy in ACTIONS]))
        pi, h_mc = mc.step(s_t, h_ec, g_tensor, h_mc)
        step_losses.append(-torch.log(pi.squeeze(0)[a_star] + 1e-8))

        dx, dy         = ACTIONS[a_star]
        x_next, y_next = env_step(x, y, dx, dy, beta, gamma)

        with torch.no_grad():
            s_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[x, y]], dtype=torch.float32, device=device)
            a_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
            a_t       = torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
            s_next    = torch.tensor([[x_next, y_next]], dtype=torch.float32, device=device)
            eps       = s_next - cb(s_t, a_t, z_t)
            z_t, h_ec = ec.step(torch.cat([s_tm1, a_tm1, s_t, eps], dim=-1), h_ec)
        x, y = x_next, y_next

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

    print(f"\nTraining MC on familiar strong contexts ({N_TRAIN_EP} episodes)...")
    for ep in range(1, N_TRAIN_EP + 1):
        optimizer.zero_grad()
        loss = run_mc_episode(ec, cb, mc, device, rng)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if ep % 300 == 0:
            print(f"  Ep {ep:4d} | loss={np.mean(losses[-300:]):.2f}")

    return losses


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(ec, cb, mc, device, contexts, seed=42, ec_lesion=False):
    ec.eval(); cb.eval(); mc.eval()
    rng     = np.random.default_rng(seed)
    results = {"success": [], "steps": []}

    for _ in range(N_EVAL_EP):
        start = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        goal  = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        while np.array_equal(start, goal):
            goal = rng.integers(0, GRID_SIZE + 1, size=2).astype(float)
        beta, gamma = contexts[rng.integers(0, len(contexts))]

        g_tensor = torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
        x, y     = float(start[0]), float(start[1])
        h_ec     = torch.zeros(1, EC_HIDDEN, device=device)
        h_mc     = mc.init_hidden(1, device)
        z_t      = torch.zeros(1, LATENT_DIM, device=device)
        success  = False
        steps    = 0

        with torch.no_grad():
            for t in range(MAX_STEPS):
                if np.linalg.norm(np.array([x, y]) - goal) < GOAL_RADIUS:
                    success = True; break

                s_t        = torch.tensor([[x, y]], dtype=torch.float32, device=device)
                pi, h_mc   = mc.step(s_t, h_ec, g_tensor, h_mc)
                action_idx = torch.argmax(pi.squeeze(0)).item()
                dx, dy     = ACTIONS[action_idx]
                x_next, y_next = env_step(x, y, dx, dy, beta, gamma)

                if not ec_lesion:
                    s_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[x, y]], dtype=torch.float32, device=device)
                    a_tm1     = torch.zeros(1, 2, device=device) if t == 0 else torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
                    a_t       = torch.tensor([[dx, dy]], dtype=torch.float32, device=device)
                    s_next    = torch.tensor([[x_next, y_next]], dtype=torch.float32, device=device)
                    eps       = s_next - cb(s_t, a_t, z_t)
                    z_t, h_ec = ec.step(torch.cat([s_tm1, a_tm1, s_t, eps], dim=-1), h_ec)

                x, y   = x_next, y_next
                steps += 1

        results["success"].append(float(success))
        results["steps"].append(steps)

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_training(losses):
    os.makedirs(FIG_DIR, exist_ok=True)
    smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(smoothed, color="steelblue", lw=2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
    ax.set_title("MC training loss (familiar strong FF ±0.8, BC)")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "mc_training_loss.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


def plot_2x2(results):
    os.makedirs(FIG_DIR, exist_ok=True)

    labels_x   = ["Familiar FF\n(±0.8)", "Novel FF\n(±0.8 diagonal)"]
    bar_labels = ["EC intact", "EC lesion"]
    colors     = ["steelblue", "tomato"]
    x          = np.arange(len(labels_x))
    w          = 0.35

    success = {
        "EC intact": [np.mean(results["familiar_ec"]["success"]),
                      np.mean(results["novel_ec"]["success"])],
        "EC lesion": [np.mean(results["familiar_lesion"]["success"]),
                      np.mean(results["novel_lesion"]["success"])],
    }
    steps = {
        "EC intact": [np.mean(results["familiar_ec"]["steps"]),
                      np.mean(results["novel_ec"]["steps"])],
        "EC lesion": [np.mean(results["familiar_lesion"]["steps"]),
                      np.mean(results["novel_lesion"]["steps"])],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, ylabel in zip(axes,
                                   [success, steps],
                                   ["Success rate", "Mean steps to goal"]):
        for i, (lbl, color) in enumerate(zip(bar_labels, colors)):
            vals = metric[lbl]
            bars = ax.bar(x + (i - 0.5) * w, vals, w,
                          label=lbl, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                fmt = f"{val:.2f}" if ylabel == "Success rate" else f"{val:.1f}"
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.02 if ylabel == "Success rate" else 0.5),
                        fmt, ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels_x)
        ax.set_ylabel(ylabel)
        if ylabel == "Success rate":
            ax.set_ylim(0, 1.15)
        ax.legend()

    fig.suptitle("EC lesion experiment — strong force fields (±0.8)\n"
                 "MC trained on familiar FF only\n"
                 "EC lesion: h_ec = 0 throughout episode",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "2x2_EC_lesion_strongF.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(CKPT, map_location=device)
    ec   = EntorhinalCortex().to(device)
    cb   = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    cb.load_state_dict(ckpt["cb"])
    ec.eval(); cb.eval()
    print(f"Loaded frozen EC + CB from {CKPT}")

    mc = MotorCortex().to(device)
    losses = train_mc(ec, cb, mc, device)
    plot_training(losses)

    print("\nEvaluating 4 conditions...")
    results = {}
    conds = [
        ("familiar_ec",     CONTEXTS,       False),
        ("familiar_lesion", CONTEXTS,       True),
        ("novel_ec",        NOVEL_CONTEXTS, False),
        ("novel_lesion",    NOVEL_CONTEXTS, True),
    ]
    for key, ctx, lesion in conds:
        r = evaluate(ec, cb, mc, device, ctx, ec_lesion=lesion)
        results[key] = r
        tag = "EC intact" if not lesion else "EC lesion"
        ff  = "familiar" if ctx is CONTEXTS else "novel   "
        print(f"  {ff} | {tag:10s} | "
              f"success={np.mean(r['success']):.3f}  steps={np.mean(r['steps']):.1f}")

    plot_2x2(results)
    print("\nDone.")


if __name__ == "__main__":
    main()

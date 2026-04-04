"""
MC_savings_newcontexts.py
- new learning component (MC) - trained as a continous action selector. idea
is that EC infers context, and MC uses that info to select a.  

Motor Cortex (MC) trained on top of frozen EC + CB.
EC+CB checkpoint: ectocb_8ctx_seq80_z32.pt

Dynamics (velocity-based, matches EC+CB training):
    F       = β·R(θ)·a  +  γ·curl·V  +  g·b
    V_new   = V + F
    S_new   = clamp(S + dt·V_new, 0, GRID_SIZE)

MC architecture:
    Input : [s_t(2), z_ec(32), z_cb(32), g(2)] = 68 dims
    z_ec  = tanh(Linear(EC_hidden→32))   — EC context signal (frozen)
    z_cb  = CB_layer2 @ cb_proj          — fixed random projection of CB h2 (256→32)
    GRU(68→256) → Linear(256→2) → tanh  — continuous action in (-1,1)²

Loss : ||g - s_{t+1}||²
       s_{t+1} = real env step (differentiable)
       Gradient: loss → s_{t+1} → env_step_torch → a_t → MC weights
       EC + CB are fully frozen.

Action: continuous, bounded in (-1,1)² via tanh. No sampling, no REINFORCE.

Experiment: train MC on all 8 contexts, evaluate success rate + steps.
            Compare EC intact vs EC lesion (z_ec=0).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyperparameters ──────────────────────────────────────────────────────────────
LATENT_DIM  = 32
EC_HIDDEN   = 256
CB_HIDDEN   = 256
EC_INPUT    = 8
CB_PROJ_DIM = 32
MC_HIDDEN   = 256
MC_INPUT    = 2 + LATENT_DIM + CB_PROJ_DIM + 2   # s(2)+z_ec(32)+z_cb(32)+g(2) = 68

MAX_STEPS     = 50
N_TRAIN_EP    = 5000
N_EVAL_EP     = 500
GOAL_RADIUS   = 0.3
GRID_SIZE     = 4.0
DT            = 1.0
LR_MC         = 1e-3
EVAL_INTERVAL = 10
N_CURVE_EP    = 10

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "../../figs/ECtoCB/MC_policy/MC_4ctx_newcontexts"

# ── Contexts (must match EC+CB training) ────────────────────────────────────────
# (beta, theta, gamma, g, bx, by, label)
CONTEXTS = [
    ( 1.0, +np.pi/4,  0.0, 0.0, 0.0, 0.0, "C0: rot+"),
    ( 1.0, -np.pi/4,  0.0, 0.0, 0.0, 0.0, "C1: rot-"),
    ( 1.0,  0.0,     +0.4, 0.0, 0.0, 0.0, "C2: curl+"),
    ( 1.0,  0.0,     -0.4, 0.0, 0.0, 0.0, "C3: curl-"),
]


# ── Models ───────────────────────────────────────────────────────────────────────
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
        """Layer2 activations (256-D) → projected to z_cb for MC."""
        h = self.layer1(torch.cat([s, a, z], dim=-1))
        return self.layer2(h)


class MotorCortex(nn.Module):
    """
    GRU-based MC.
    Input per step: [s_t(2), z_ec(32), z_cb(32), g(2)] = 68 dims
    Output: tanh(Linear(h)) → continuous action in (-1,1)²
    z_cb = CB_layer2 @ cb_proj  — fixed random projection (not trained)
    """
    def __init__(self):
        super().__init__()
        self.hidden_dim = MC_HIDDEN
        self.ec_readout = nn.Linear(EC_HIDDEN, LATENT_DIM)
        self.cell       = nn.GRUCell(MC_INPUT, MC_HIDDEN)
        self.readout    = nn.Linear(MC_HIDDEN, 2)

        # fixed random column-normalised projection: CB_HIDDEN → CB_PROJ_DIM
        W = torch.randn(CB_HIDDEN, CB_PROJ_DIM)
        self.register_buffer("cb_proj", W / W.norm(dim=0, keepdim=True))

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, MC_HIDDEN, device=device)

    def step(self, s_t, h_ec, g, h_mc, cb_hidden):
        z_ec  = torch.tanh(self.ec_readout(h_ec))           # (1, 32)
        z_cb  = cb_hidden @ self.cb_proj                     # (1, 32)
        inp   = torch.cat([s_t, z_ec, z_cb, g], dim=-1)     # (1, 68)
        h_new = self.cell(inp, h_mc)
        a_t   = torch.tanh(self.readout(h_new))              # (1, 2) in (-1,1)²
        return a_t, h_new


# ── Differentiable env step ──────────────────────────────────────────────────────
def env_step_torch(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    """
    Velocity-based dynamics, differentiable w.r.t. action.
    pos, vel, action : tensors shape (2,)
    Returns pos_new (2,), vel_new (2,) detached.
    Gradient flows: loss → pos_new → action → MC weights.
    """
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R     = torch.stack([torch.stack([cos_t, -sin_t]),
                         torch.stack([sin_t,  cos_t])])
    curl  = torch.tensor([[0., -1.], [1., 0.]], device=device)
    b_vec = torch.tensor([float(bx), float(by)], device=device)

    F       = beta * (R @ action) + gamma * (curl @ vel) + g * b_vec
    vel_new = vel + F
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


# ── Single training episode ──────────────────────────────────────────────────────
def run_episode(ec, cb, mc, device, ctx_idx, rng, ec_lesion=False):
    beta, theta, gamma, g, bx, by, _ = CONTEXTS[ctx_idx]

    start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
    while np.linalg.norm(start - goal) < 0.5:
        goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

    g_t   = torch.tensor(goal,  dtype=torch.float32, device=device)
    pos   = torch.tensor(start, dtype=torch.float32, device=device)
    vel   = torch.zeros(2, device=device)
    h_ec  = ec.init_hidden(1, device)
    h_mc  = mc.init_hidden(1, device)
    z_t   = torch.zeros(1, LATENT_DIM, device=device)
    prev_a = torch.zeros(1, 2, device=device)
    step_losses = []

    for t in range(MAX_STEPS):
        if torch.norm(pos - g_t).item() < GOAL_RADIUS:
            break

        # CB hidden — no grad (z_cb is informational, not in loss graph)
        with torch.no_grad():
            cb_h = cb.hidden(pos.unsqueeze(0), prev_a, z_t)   # (1, 256)

        # MC selects action — DIFFERENTIABLE
        h_ec_in = torch.zeros(1, EC_HIDDEN, device=device) if ec_lesion else h_ec
        a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec_in, g_t.unsqueeze(0), h_mc, cb_h)

        # Real next state via differentiable env step
        # Gradient: loss → pos_new → action → MC
        s_next, vel_next = env_step_torch(
            pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device)

        # Loss: ||g - s_{t+1}||²
        step_losses.append(torch.sum((g_t - s_next) ** 2))

        # EC update — no grad, EC is frozen
        with torch.no_grad():
            s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
            eps   = s_next.detach().unsqueeze(0) - cb(pos.unsqueeze(0), a_t.detach(), z_t)
            if not ec_lesion:
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

        prev_a = a_t.detach()
        pos    = s_next.detach()
        vel    = vel_next

    if not step_losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(step_losses).sum()


# ── Evaluation ───────────────────────────────────────────────────────────────────
def evaluate(ec, cb, mc, device, ctx_indices, seed=42, ec_lesion=False, n_ep=None):
    ec.eval(); cb.eval(); mc.eval()
    rng     = np.random.default_rng(seed)
    results = {"success": [], "steps": []}
    n_ep    = n_ep or N_EVAL_EP

    for _ in range(n_ep):
        ctx_idx = ctx_indices[rng.integers(0, len(ctx_indices))]
        beta, theta, gamma, g, bx, by, _ = CONTEXTS[ctx_idx]

        start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
        goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
        while np.linalg.norm(start - goal) < 0.5:
            goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

        g_t   = torch.tensor(goal,  dtype=torch.float32, device=device).unsqueeze(0)
        pos   = torch.tensor(start, dtype=torch.float32, device=device)
        vel   = torch.zeros(2, device=device)
        h_ec  = ec.init_hidden(1, device)
        h_mc  = mc.init_hidden(1, device)
        z_t   = torch.zeros(1, LATENT_DIM, device=device)
        prev_a = torch.zeros(1, 2, device=device)
        success = False; steps = 0

        with torch.no_grad():
            for t in range(MAX_STEPS):
                if torch.norm(pos - g_t.squeeze()).item() < GOAL_RADIUS:
                    success = True; break

                cb_h = cb.hidden(pos.unsqueeze(0), prev_a, z_t)
                h_ec_in = torch.zeros(1, EC_HIDDEN, device=device) if ec_lesion else h_ec
                a_t, h_mc = mc.step(pos.unsqueeze(0), h_ec_in, g_t, h_mc, cb_h)

                s_next, vel_next = env_step_torch(
                    pos, vel, a_t.squeeze(0), beta, theta, gamma, g, bx, by, device)

                if not ec_lesion:
                    s_tm1 = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                    eps   = s_next.unsqueeze(0) - cb(pos.unsqueeze(0), a_t, z_t)
                    z_t, h_ec = ec.step(
                        torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec)

                prev_a = a_t; pos = s_next; vel = vel_next; steps += 1

        results["success"].append(float(success))
        results["steps"].append(steps)

    return results


# ── Training ─────────────────────────────────────────────────────────────────────
def train_mc(ec, cb, mc, device, seed=0):
    ec.eval(); cb.eval(); mc.train()
    for p in ec.parameters(): p.requires_grad_(False)
    for p in cb.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.Adam(mc.parameters(), lr=LR_MC)
    rng       = np.random.default_rng(seed)
    losses    = []
    curve     = {"ep": [], "success": [], "steps": [],
                 "lesion_success": [], "lesion_steps": []}

    print(f"\nTraining MC on rot+/rot-/curl+/curl- ({N_TRAIN_EP} episodes)...")
    for ep in range(1, N_TRAIN_EP + 1):
        ctx_idx = int(rng.integers(0, len(CONTEXTS)))
        optimizer.zero_grad()
        loss = run_episode(ec, cb, mc, device, ctx_idx, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if ep % EVAL_INTERVAL == 0:
            mc.eval()
            all_ctx = list(range(len(CONTEXTS)))
            r_ec  = evaluate(ec, cb, mc, device, all_ctx, seed=ep,
                             ec_lesion=False, n_ep=N_CURVE_EP)
            r_les = evaluate(ec, cb, mc, device, all_ctx, seed=ep,
                             ec_lesion=True,  n_ep=N_CURVE_EP)
            curve["ep"].append(ep)
            curve["success"].append(np.mean(r_ec["success"]))
            curve["steps"].append(np.mean(r_ec["steps"]))
            curve["lesion_success"].append(np.mean(r_les["success"]))
            curve["lesion_steps"].append(np.mean(r_les["steps"]))
            mc.train()

            if ep % 2000 == 0:
                print(f"  Ep {ep:5d} | loss={np.mean(losses[-200:]):.3f} | "
                      f"acc={curve['success'][-1]:.2f}  "
                      f"acc_lesion={curve['lesion_success'][-1]:.2f}")

    return losses, curve


# ── Plots ────────────────────────────────────────────────────────────────────────
def plot_all(losses, curve):
    os.makedirs(FIG_DIR, exist_ok=True)

    # training loss
    smoothed = np.convolve(losses, np.ones(100)/100, mode="valid")
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.plot(smoothed, color="steelblue", lw=2)
    ax.set_xlabel("Episode"); ax.set_ylabel("Loss ||g - s_{t+1}||²")
    ax.set_title("MC training loss")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/mc_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # learning curves
    ep = curve["ep"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, curve["success"],        color="steelblue", lw=2, label="EC intact")
    axes[0].plot(ep, curve["lesion_success"], color="tomato",    lw=2, label="EC lesion")
    axes[1].plot(ep, curve["steps"],          color="steelblue", lw=2, label="EC intact")
    axes[1].plot(ep, curve["lesion_steps"],   color="tomato",    lw=2, label="EC lesion")
    axes[0].set_ylabel("Success rate"); axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("Mean steps to goal")
    for ax in axes:
        ax.set_xlabel("Episode"); ax.legend()
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("MC learning — rot+/rot-/curl+/curl- | EC intact vs EC lesion")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/mc_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Figures saved to {FIG_DIR}/")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(CKPT, map_location=device)
    ec   = EntorhinalCortex().to(device)
    cb   = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])

    # remap CB weights from nn.Sequential to split layer architecture
    cb_sd = ckpt["cb"]
    remap = {"net.0": "layer1.0", "net.2": "layer2.0", "net.4": "out"}
    new_sd = {}
    for k, v in cb_sd.items():
        prefix = ".".join(k.split(".")[:2])
        new_sd[k.replace(prefix, remap[prefix]) if prefix in remap else k] = v
    cb.load_state_dict(new_sd)
    ec.eval(); cb.eval()
    print(f"Loaded frozen EC + CB from {CKPT}")

    os.makedirs(FIG_DIR, exist_ok=True)
    data_path = f"{FIG_DIR}/training_data.npz"

    if os.path.exists(data_path) and "--replot" in __import__("sys").argv:
        print(f"Loading saved data from {data_path} — skipping training.")
        d = np.load(data_path, allow_pickle=True)
        losses = list(d["losses"])
        curve  = d["curve"].item()
    else:
        mc = MotorCortex().to(device)
        losses, curve = train_mc(ec, cb, mc, device)
        torch.save(mc.state_dict(), f"{FIG_DIR}/mc_trained.pt")
        np.savez(data_path, losses=losses, curve=curve)
        print(f"Saved MC checkpoint to {FIG_DIR}/mc_trained.pt")

    plot_all(losses, curve)
    print("Done.")


if __name__ == "__main__":
    main()

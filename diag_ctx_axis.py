"""
diag_ctx_axis.py

Diagnostic: plot distribution of z_t @ ctx_axis (the scalar fed to HC)
for context A vs B episodes, under both:
  - random actions (training conditions for ctx_axis)
  - trained MC actions (actual inference conditions)

Also plots projection value over timesteps within an episode.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ── copy hyperparams from mc_svd_key_hc.py ───────────────────────────────────
LATENT_DIM  = 32
EC_HIDDEN   = 256
CB_HIDDEN   = 256
CB_PROJ_DIM = 32
MC_HIDDEN   = 256
EC_INPUT    = 8
MAX_STEPS   = 20
GRID_SIZE   = 4.0
DT          = 1.0
K_CONTEXT   = 1
N_CTX_EP    = 200

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/MC_EC_CB/MC_savings_norouting/svd_key_hebbian"

CTX_A = (1.0, np.pi / 3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B = (1.0, 0.0,      +0.4, 0.0, 0.0, 0.0, "B: curl+")


# ── Model definitions (minimal, matching mc_svd_key_hc.py) ───────────────────
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
            nn.Linear(CB_HIDDEN, 2)
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))


def env_step(pos, vel, action, beta, theta, gamma, g, bx, by, device):
    import math
    cos_t = torch.tensor(float(np.cos(theta)), dtype=torch.float32, device=device)
    sin_t = torch.tensor(float(np.sin(theta)), dtype=torch.float32, device=device)
    R    = torch.stack([torch.stack([cos_t, -sin_t]),
                        torch.stack([sin_t,  cos_t])])
    curl = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device)
    b_v  = torch.tensor([float(bx), float(by)], dtype=torch.float32, device=device)
    F    = beta * (R @ action) + gamma * (curl @ vel) + g * b_v
    vel_new = vel + F
    pos_new = torch.clamp(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new, vel_new.detach()


# ── Collect z_t sequences ─────────────────────────────────────────────────────
def collect_z(ec, cb, ctx, n_episodes, device, rng):
    """Collect z_t for each timestep under random actions."""
    beta, theta, gamma, g_coef, bx, by, _ = ctx
    all_z = []
    all_t = []

    with torch.no_grad():
        for _ in range(n_episodes):
            start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
            goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
            while np.linalg.norm(start - goal) < 0.5:
                goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

            pos    = torch.tensor(start, dtype=torch.float32, device=device)
            vel    = torch.zeros(2, dtype=torch.float32, device=device)
            h_ec   = ec.init_hidden(1, device)
            z_t    = torch.zeros(1, LATENT_DIM, device=device)
            prev_a = torch.zeros(1, 2, device=device)

            for t in range(MAX_STEPS):
                action = torch.tensor(
                    rng.uniform(-1, 1, size=2).astype(np.float32), device=device
                )
                s_next, vel_new = env_step(pos, vel, action, beta, theta, gamma, g_coef, bx, by, device)

                s_tm1   = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                cb_pred = cb(pos.unsqueeze(0), action.unsqueeze(0), z_t)
                eps     = s_next.unsqueeze(0) - cb_pred
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec
                )

                all_z.append(z_t.squeeze(0).cpu().numpy())
                all_t.append(t)

                prev_a = action.unsqueeze(0)
                pos    = s_next.detach()
                vel    = vel_new.detach()

    return np.array(all_z), np.array(all_t)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cpu")

    # Load checkpoint
    ckpt = torch.load(CKPT, map_location=device)
    ec = EntorhinalCortex().to(device)
    cb = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    cb.load_state_dict(ckpt["cb"])
    ec.eval(); cb.eval()

    rng = np.random.default_rng(42)

    # ── 1. Fit ctx_axis (same as mc_svd_key_hc.py) ───────────────────────────
    print("Collecting z_t with random actions...")
    zA, tA = collect_z(ec, cb, CTX_A, N_CTX_EP, device, rng)
    zB, tB = collect_z(ec, cb, CTX_B, N_CTX_EP, device, rng)

    all_z  = np.vstack([zA, zB])
    labels = np.array([0]*len(zA) + [1]*len(zB))
    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(all_z, labels)
    train_acc = clf.score(all_z, labels)
    print(f"  Logistic reg train accuracy: {train_acc:.3f}")

    W = clf.coef_   # (1, 32)
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    ctx_axis = Vt[0]  # (32,)

    proj_A = zA @ ctx_axis
    proj_B = zB @ ctx_axis

    # ── 2. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Context axis diagnostic  (train acc={train_acc:.3f})", fontsize=13)

    # Left: full distribution
    ax = axes[0]
    bins = np.linspace(min(proj_A.min(), proj_B.min()),
                       max(proj_A.max(), proj_B.max()), 60)
    ax.hist(proj_A, bins=bins, alpha=0.6, color="steelblue", label="A: rot+60", density=True)
    ax.hist(proj_B, bins=bins, alpha=0.6, color="tomato",    label="B: curl+",  density=True)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Distribution of z_t @ ctx_axis")
    ax.set_xlabel("Projected scalar")
    ax.set_ylabel("Density")
    ax.legend()

    # Right: mean ± std per timestep
    ax = axes[1]
    for z, t, color, label in [(zA, tA, "steelblue", "A: rot+60"),
                                (zB, tB, "tomato",    "B: curl+")]:
        proj = z @ ctx_axis
        means = np.array([proj[t == ti].mean() for ti in range(MAX_STEPS)])
        stds  = np.array([proj[t == ti].std()  for ti in range(MAX_STEPS)])
        ts = np.arange(MAX_STEPS)
        ax.plot(ts, means, color=color, label=label)
        ax.fill_between(ts, means - stds, means + stds, alpha=0.2, color=color)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_title("Projection over timestep within episode")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("z_t @ ctx_axis")
    ax.legend()

    plt.tight_layout()
    out = f"{FIG_DIR}/diag_ctx_axis.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == "__main__":
    main()

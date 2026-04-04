"""
diag_zec_phases.py

Check whether z_ec (and z_dg) in A2 phase actually resembles A-phase z_ec,
despite MC starting from B-policy in A2.

Method:
  Load seed 0 HC condition curves to get saved MC-like actions per phase.
  Re-run EC forward pass with:
    - A-phase actions (from trained MC at end of A)
    - B-phase actions (from trained MC at end of B)
    - A2-phase actions (B-policy applied to A-context = what happens at start of A2)
  Collect z_ec and z_dg at each timestep.
  PCA + overlap measure.

Since we don't have saved MC weights, we approximate with:
  - "A-like" actions: random actions biased toward -60° (rotation-correcting)
  - "B-like" actions: random actions biased toward 0° (curl-correcting)
  - This is a proxy — the key question is whether eps difference drives z_ec separation
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

LATENT_DIM  = 32
EC_HIDDEN   = 256
CB_HIDDEN   = 256
DG_DIM      = 128
DG_SPARSITY = 0.10
EC_INPUT    = 8
MAX_STEPS   = 20
GRID_SIZE   = 4.0
DT          = 1.0
N_EP        = 300

CKPT    = "checkpoints/ectocb_8ctx_seq80_z32.pt"
FIG_DIR = "/Users/cheshtabhatia/cereb_forward_model/figs/ECtoCB/MC_EC_CB/MC_savings_norouting/replay_hc"

CTX_A = (1.0, np.pi/3, 0.0, 0.0, 0.0, 0.0, "A: rot+60")
CTX_B = (1.0, 0.0,    +0.4, 0.0, 0.0, 0.0, "B: curl+")


class EntorhinalCortex(nn.Module):
    def __init__(self):
        super().__init__()
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


class DentateGyrus(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, DG_DIM, bias=False)
        rng = torch.Generator(); rng.manual_seed(seed)
        nn.init.normal_(self.fc.weight, std=1.0/LATENT_DIM**0.5, generator=rng)
        for p in self.parameters(): p.requires_grad_(False)
        self.k = max(1, int(DG_SPARSITY * DG_DIM))

    def forward(self, z):
        h = self.fc(z)
        topk_vals, _ = torch.topk(h, self.k, dim=-1)
        return torch.relu(h - topk_vals[..., -1:])


def env_step(pos, vel, action, beta, theta, gamma, g, bx, by, device):
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


def collect_zec_zdg(ec, cb, dg, ctx, action_mode, n_episodes, device, rng):
    """
    action_mode: 'random'   — uniform random actions
                 'toward_goal' — actions pointing toward goal (A-like corrective)
                 'perpendicular' — actions perpendicular to goal (B-like curl)
    Returns z_ec, z_dg arrays shape (n_ep * T, dim), timestep array
    """
    beta, theta, gamma, g_coef, bx, by, _ = ctx
    all_zec, all_zdg, all_t = [], [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            start = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
            goal  = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
            while np.linalg.norm(start - goal) < 0.5:
                goal = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)

            pos    = torch.tensor(start, dtype=torch.float32, device=device)
            vel    = torch.zeros(2, dtype=torch.float32, device=device)
            g_t    = torch.tensor(goal, dtype=torch.float32, device=device)
            h_ec   = ec.init_hidden(1, device)
            z_t    = torch.zeros(1, LATENT_DIM, device=device)
            prev_a = torch.zeros(1, 2, device=device)

            for t in range(MAX_STEPS):
                d = torch.norm(pos - g_t).item()
                if d < 0.1:
                    break

                # Generate action based on mode
                if action_mode == 'random':
                    a_np = rng.uniform(-1, 1, size=2).astype(np.float32)
                elif action_mode == 'toward_goal':
                    # Corrective action: points toward goal (A-policy-like at end of training)
                    gd = (g_t - pos).cpu().numpy()
                    gd = gd / (np.linalg.norm(gd) + 1e-8)
                    a_np = (gd + rng.normal(0, 0.3, 2)).astype(np.float32)
                    a_np = np.clip(a_np, -1, 1)
                elif action_mode == 'perpendicular':
                    # Perpendicular to goal direction (B-policy-like: curl compensation)
                    gd = (g_t - pos).cpu().numpy()
                    gd = gd / (np.linalg.norm(gd) + 1e-8)
                    perp = np.array([-gd[1], gd[0]])
                    a_np = (perp + rng.normal(0, 0.3, 2)).astype(np.float32)
                    a_np = np.clip(a_np, -1, 1)

                action = torch.tensor(a_np, device=device)
                s_next, vel_new = env_step(pos, vel, action, beta, theta, gamma, g_coef, bx, by, device)

                cb_pred = cb(pos.unsqueeze(0), action.unsqueeze(0), z_t)
                eps     = s_next.unsqueeze(0) - cb_pred
                s_tm1   = torch.zeros(1, 2, device=device) if t == 0 else pos.unsqueeze(0)
                z_t, h_ec = ec.step(
                    torch.cat([s_tm1, prev_a, pos.unsqueeze(0), eps], dim=-1), h_ec
                )
                z_dg = dg(z_t)

                all_zec.append(z_t.squeeze(0).cpu().numpy())
                all_zdg.append(z_dg.squeeze(0).cpu().numpy())
                all_t.append(t)

                prev_a = action.unsqueeze(0)
                pos    = s_next.detach()
                vel    = vel_new

    return np.array(all_zec), np.array(all_zdg), np.array(all_t)


def lda_accuracy(X, y):
    """Linear discriminant analysis accuracy (train=test, just checking separability)."""
    try:
        clf = LinearDiscriminantAnalysis()
        clf.fit(X, y)
        return clf.score(X, y)
    except:
        return float('nan')


def main():
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    device = torch.device("cpu")

    ckpt = torch.load(CKPT, map_location=device)
    ec = EntorhinalCortex().to(device)
    cb = CerebellumCB().to(device)
    ec.load_state_dict(ckpt["ec"])
    cb.load_state_dict(ckpt["cb"])
    ec.eval(); cb.eval()
    dg = DentateGyrus(seed=0).to(device)

    rng = np.random.default_rng(42)

    print("Collecting z_ec / z_dg for 3 conditions...")
    print("  A-context + toward-goal actions  (proxy for end-of-A MC)")
    zec_A_good, zdg_A_good, t_A_good = collect_zec_zdg(
        ec, cb, dg, CTX_A, 'toward_goal', N_EP, device, rng)

    print("  A-context + perpendicular actions (proxy for start-of-A2 MC with B-policy)")
    zec_A2_start, zdg_A2_start, t_A2 = collect_zec_zdg(
        ec, cb, dg, CTX_A, 'perpendicular', N_EP, device, rng)

    print("  B-context + toward-goal actions   (proxy for end-of-B MC)")
    zec_B_good, zdg_B_good, t_B_good = collect_zec_zdg(
        ec, cb, dg, CTX_B, 'toward_goal', N_EP, device, rng)

    print("  A-context + random actions        (baseline)")
    zec_A_rand, zdg_A_rand, t_A_rand = collect_zec_zdg(
        ec, cb, dg, CTX_A, 'random', N_EP, device, rng)

    # ── LDA separability ─────────────────────────────────────────────────────
    print("\nLDA separability (z_ec):")

    # Can we separate A-good from B-good? (should be high)
    X = np.vstack([zec_A_good, zec_B_good])
    y = np.array([0]*len(zec_A_good) + [1]*len(zec_B_good))
    print(f"  A(good actions) vs B(good actions): {lda_accuracy(X,y):.3f}")

    # Can we separate A-good from A2-start? (key question — are they different?)
    X = np.vstack([zec_A_good, zec_A2_start])
    y = np.array([0]*len(zec_A_good) + [1]*len(zec_A2_start))
    print(f"  A(good actions) vs A(perp actions): {lda_accuracy(X,y):.3f}  ← key: how diff is A2 start?")

    # Can we separate A2-start from B-good?
    X = np.vstack([zec_A2_start, zec_B_good])
    y = np.array([0]*len(zec_A2_start) + [1]*len(zec_B_good))
    print(f"  A(perp actions) vs B(good actions): {lda_accuracy(X,y):.3f}")

    print("\nLDA separability (z_dg):")
    X = np.vstack([zdg_A_good, zdg_B_good])
    y = np.array([0]*len(zdg_A_good) + [1]*len(zdg_B_good))
    print(f"  A(good actions) vs B(good actions): {lda_accuracy(X,y):.3f}")

    X = np.vstack([zdg_A_good, zdg_A2_start])
    y = np.array([0]*len(zdg_A_good) + [1]*len(zdg_A2_start))
    print(f"  A(good actions) vs A(perp actions): {lda_accuracy(X,y):.3f}  ← key")

    X = np.vstack([zdg_A2_start, zdg_B_good])
    y = np.array([0]*len(zdg_A2_start) + [1]*len(zdg_B_good))
    print(f"  A(perp actions) vs B(good actions): {lda_accuracy(X,y):.3f}")

    # ── PCA plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("z_ec and z_dg: does A2-start look like A or B?", fontsize=12)

    for ax, Z_list, labels, title in [
        (axes[0],
         [zec_A_good, zec_A2_start, zec_B_good],
         ["A (good MC)",  "A2-start (B-policy MC)", "B (good MC)"],
         "z_ec (PCA)"),
        (axes[1],
         [zdg_A_good, zdg_A2_start, zdg_B_good],
         ["A (good MC)", "A2-start (B-policy MC)", "B (good MC)"],
         "z_dg (PCA)"),
    ]:
        all_Z = np.vstack(Z_list)
        pca   = PCA(n_components=2)
        all_P = pca.fit_transform(all_Z)
        colors = ["steelblue", "gold", "tomato"]
        markers = ["o", "^", "s"]
        idx = 0
        for Z, lbl, col, mk in zip(Z_list, labels, colors, markers):
            n = len(Z)
            P = all_P[idx:idx+n]
            ax.scatter(P[::5, 0], P[::5, 1], c=col, label=lbl,
                      alpha=0.4, s=10, marker=mk)
            idx += n
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = f"{FIG_DIR}/diag_zec_phases.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")

    # ── Per-timestep mean cosine similarity ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Mean cosine similarity: A2-start vs A-good and B-good", fontsize=11)

    for ax, Z_A, Z_A2, Z_B, t_A, t_A2s, t_B, title in [
        (axes[0], zec_A_good, zec_A2_start, zec_B_good,
         t_A_good, t_A2, t_B_good, "z_ec"),
        (axes[1], zdg_A_good, zdg_A2_start, zdg_B_good,
         t_A_good, t_A2, t_B_good, "z_dg"),
    ]:
        # Mean vector per timestep for each condition
        ts = np.arange(MAX_STEPS)
        sim_A2_vs_A, sim_A2_vs_B = [], []
        for t in ts:
            mask_A  = t_A   == t
            mask_A2 = t_A2s == t
            mask_B  = t_B   == t
            if mask_A.sum() < 2 or mask_A2.sum() < 2 or mask_B.sum() < 2:
                sim_A2_vs_A.append(np.nan)
                sim_A2_vs_B.append(np.nan)
                continue
            mu_A  = Z_A[mask_A].mean(0)
            mu_A2 = Z_A2[mask_A2].mean(0)
            mu_B  = Z_B[mask_B].mean(0)
            cos_A  = np.dot(mu_A2, mu_A)  / (np.linalg.norm(mu_A2)*np.linalg.norm(mu_A)+1e-8)
            cos_B  = np.dot(mu_A2, mu_B)  / (np.linalg.norm(mu_A2)*np.linalg.norm(mu_B)+1e-8)
            sim_A2_vs_A.append(cos_A)
            sim_A2_vs_B.append(cos_B)

        ax.plot(ts, sim_A2_vs_A, color="steelblue", lw=2, label="A2-start vs A-good")
        ax.plot(ts, sim_A2_vs_B, color="tomato",    lw=2, label="A2-start vs B-good")
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("Timestep within episode"); ax.set_ylabel("Cosine similarity")
        ax.set_title(title); ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out2 = f"{FIG_DIR}/diag_zec_cosine.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()

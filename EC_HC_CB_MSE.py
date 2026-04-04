"""
EC_HC_CB_MSE.py

Full three-region circuit: EC → HC → CB, trained with shifted MSE prediction loss.

Architecture
------------
  EC (EntorhinalCortex) : GRU — observes sequence of transitions [s_t, a_t, s_{t+1}]
                          and infers latent context z_t at each timestep.

  HC (Hippocampus)      : Leaky integrator — receives z_t from EC and slowly
                          consolidates it into a stable context signal c_t.
                            c_t = (1 - alpha) * c_{t-1} + alpha * proj(z_t)
                          alpha is learnable (starts ~0.1, slow consolidation).

  CB (Cerebellum)       : MLP forward model — takes [s_t, a_t, c_t] and predicts s_{t+1}.

Shifted training (no shortcut)
-------------------------------
  z_t is produced from transitions 0..t  (includes s_{t+1}).
  HC consolidates z_t → c_t.
  CB uses  c_{t-1}  with [s_t, a_t] to predict s_{t+1}  (c never saw s_{t+1}).
  Gradient flows: CB loss → CB → HC → EC.

Data: 4 fixed contexts, each sequence contains exactly 2, switching at a random point.
  C0: (beta=+0.4, gamma=0.0) — force along action, positive
  C1: (beta=-0.4, gamma=0.0) — force along action, negative
  C2: (beta=0.0, gamma=+0.4) — force perpendicular, right
  C3: (beta=0.0, gamma=-0.4) — force perpendicular, left
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ec_module import EntorhinalCortex
from hc_module import Hippocampus


# ── Hyperparameters ──────────────────────────────────────────────────────────

SEQ_LEN    = 40
N_TRAIN    = 2000
N_VAL      = 200
BATCH_SIZE = 128
EPOCHS     = 100
LR         = 1e-3
LATENT_DIM = 16   # EC output dim (z_t)
HC_DIM     = 16   # HC output dim (c_t) — kept equal to LATENT_DIM
CB_HIDDEN  = 256

FIG_DIR    = "figs/EC_HC_CB"


# ── Cerebellar MLP ────────────────────────────────────────────────────────────

class CerebellumCB(nn.Module):
    """MLP: [s_t, a_t, c_t] → s_{t+1}.   input = 2 + 2 + HC_DIM dims."""
    def __init__(self, hc_dim=HC_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + hc_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, s, a, c):
        # s: (B, T, 2)  a: (B, T, 2)  c: (B, T, HC_DIM)
        return self.net(torch.cat([s, a, c], dim=-1))   # (B, T, 2)


# ── Data generation ───────────────────────────────────────────────────────────

def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0):
    """
    Each sequence has exactly 2 contexts chosen without replacement from 4.
    Switch point is sampled uniformly in [10, 30] so the model cannot
    memorise a fixed switch time.
    """
    rng  = np.random.default_rng(seed)
    acts = [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]

    all_trans, all_ctx = [], []

    for _ in range(n_seqs):
        x, y = float(rng.integers(0, 5)), float(rng.integers(0, 5))

        all_ctx_opts = [(0.4, 0.0, 0), (-0.4, 0.0, 1), (0.0, 0.4, 2), (0.0, -0.4, 3)]
        c_ids        = rng.choice(4, size=2, replace=False)
        ctx_first, ctx_second = all_ctx_opts[c_ids[0]], all_ctx_opts[c_ids[1]]
        switch = int(rng.integers(10, 31))

        seq_trans, seq_ctx = [], []

        for t in range(seq_len):
            beta, gamma, c_type = ctx_first if t < switch else ctx_second
            beta, gamma, c_type = float(beta), float(gamma), int(c_type)

            dx, dy = acts[rng.integers(0, 4)]
            a      = np.array([dx, dy], dtype=np.float32)
            a_norm = np.linalg.norm(a)
            a_hat  = a / a_norm if a_norm > 0 else np.zeros(2, dtype=np.float32)
            a_perp = np.array([-a_hat[1], a_hat[0]], dtype=np.float32)

            f_env  = beta * a_hat + gamma * a_perp
            x_next = float(np.clip(x + dx + f_env[0], 0, 4))
            y_next = float(np.clip(y + dy + f_env[1], 0, 4))

            seq_trans.append([x, y, dx, dy, x_next, y_next])
            seq_ctx.append([beta, gamma, c_type])
            x, y = x_next, y_next

        all_trans.append(seq_trans)
        all_ctx.append(seq_ctx)

    return (
        torch.tensor(np.array(all_trans), dtype=torch.float32),  # (N, T, 6)
        torch.tensor(np.array(all_ctx),   dtype=torch.float32),  # (N, T, 3)
    )


class TransitionDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions
    def __len__(self): return len(self.transitions)
    def __getitem__(self, idx): return self.transitions[idx]


# ── Evaluation + plots ────────────────────────────────────────────────────────

def evaluate_and_plot(ec, hc, cb, va_trans, va_ctx, history, device):
    ec.eval(); hc.eval(); cb.eval()

    all_z, all_c, all_ctype = [], [], []

    with torch.no_grad():
        for i in range(0, len(va_trans), BATCH_SIZE):
            trans = va_trans[i:i+BATCH_SIZE].to(device)
            ctx   = va_ctx[i:i+BATCH_SIZE]

            z, _  = ec(trans)                   # (B, T, LATENT_DIM)
            c, _  = hc(z)                       # (B, T, HC_DIM)

            B, T, _ = z.shape
            all_z.append(z.cpu().reshape(B * T, LATENT_DIM))
            all_c.append(c.cpu().reshape(B * T, HC_DIM))
            all_ctype.append(ctx[:, :, 2].reshape(B * T))

    all_z     = torch.cat(all_z).numpy()       # (N*T, LATENT_DIM)
    all_c     = torch.cat(all_c).numpy()       # (N*T, HC_DIM)
    all_ctype = torch.cat(all_ctype).numpy()   # (N*T,)

    # ── Scalar metrics for EC (z) and HC (c) ─────────────────────────────────
    def probe_metrics(rep, label=""):
        r_per_dim = [abs(stats.pearsonr(rep[:, d], all_ctype)[0]) for d in range(rep.shape[1])]
        r_max = max(r_per_dim)
        r2    = cross_val_score(LinearRegression(), rep, all_ctype,
                                cv=5, scoring="r2").mean()
        acc   = cross_val_score(LogisticRegression(max_iter=500), rep,
                                all_ctype.astype(int), cv=5, scoring="accuracy").mean()
        print(f"  {label}  Max |r|: {r_max:.4f} | Linear R²: {r2:.4f} | Logistic acc: {acc:.4f}")
        return r_per_dim, acc

    print("\n── Context encoding metrics ──────────────────────────────")
    print("  (chance accuracy = 0.25 for 4 contexts)")
    r_per_dim_z, _ = probe_metrics(all_z, "EC  z_t")
    r_per_dim_c, _ = probe_metrics(all_c, "HC  c_t")

    os.makedirs(FIG_DIR, exist_ok=True)

    colors   = {0: "steelblue", 1: "tomato", 2: "seagreen", 3: "orange"}
    c_labels = {0: "C0 (+beta)", 1: "C1 (-beta)", 2: "C2 (+gamma)", 3: "C3 (-gamma)"}

    # ── Fig 1: Training curve ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_mse"], label="train MSE", color="steelblue")
    ax.plot(history["val_mse"],   label="val MSE",   color="tomato")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title("Training curve — EC→HC→CB MSE (shifted)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/training.png")

    # ── Fig 2: PCA of z_t (EC) and c_t (HC) side by side ─────────────────────
    n_plot = min(2000, len(all_z))
    idx    = np.random.choice(len(all_z), n_plot, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, rep, title in zip(axes,
                               [all_z[idx], all_c[idx]],
                               ["EC z_t", "HC c_t"]):
        pca     = PCA(n_components=2)
        r2d     = pca.fit_transform(rep)
        ctx_sub = all_ctype[idx]
        for cc in range(4):
            mask = ctx_sub == cc
            if mask.sum() > 0:
                ax.scatter(r2d[mask, 0], r2d[mask, 1],
                           c=colors[cc], label=c_labels[cc], alpha=0.4, s=10)
        ax.set_title(f"PCA of {title} — coloured by context\nEC→HC→CB MSE (shifted)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/pca.png")

    # ── Fig 3: Per-dim |r| for EC and HC side by side ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 3))
    for ax, r_vals, title in zip(axes,
                                  [r_per_dim_z, r_per_dim_c],
                                  ["EC z_t", "HC c_t"]):
        ax.bar(range(len(r_vals)), r_vals, color="steelblue")
        ax.axhline(0.1, color="tomato", linestyle="--", label="|r|=0.1 ref")
        ax.set_xlabel("dimension"); ax.set_ylabel("|Pearson r| vs c_type")
        ax.set_title(f"Per-dim correlation — {title}\nEC→HC→CB MSE (shifted)")
        ax.set_xticks(range(len(r_vals))); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/perdim.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/perdim.png")

    # ── Fig 4: Decoding accuracy over time — EC vs HC ─────────────────────────
    N_val         = len(va_trans)
    all_z_seq     = all_z.reshape(N_val, SEQ_LEN, LATENT_DIM)
    all_c_seq     = all_c.reshape(N_val, SEQ_LEN, HC_DIM)
    all_ctype_seq = all_ctype.reshape(N_val, SEQ_LEN)

    acc_z_t, acc_c_t = [], []
    for t in range(SEQ_LEN):
        c_t_labels = all_ctype_seq[:, t].astype(int)
        acc_z_t.append(cross_val_score(LogisticRegression(max_iter=500),
                                       all_z_seq[:, t, :], c_t_labels,
                                       cv=5, scoring="accuracy").mean())
        acc_c_t.append(cross_val_score(LogisticRegression(max_iter=500),
                                       all_c_seq[:, t, :], c_t_labels,
                                       cv=5, scoring="accuracy").mean())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(acc_z_t, color="steelblue",  label="EC z_t (raw)")
    ax.plot(acc_c_t, color="darkorange", label="HC c_t (consolidated)", linewidth=2)
    ax.axhline(0.25, color="gray",  linestyle="--", label="chance (0.25)")
    ax.axvline(20,   color="tomato", linestyle="--", label="avg switch (t≈20)")
    ax.set_xlabel("Timestep t"); ax.set_ylabel("Context decode accuracy")
    ax.set_title("Context decodability over time — EC vs HC\nEC→HC→CB MSE (shifted)")
    ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/decodetime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/decodetime.png")

    # ── Fig 5: Confusion matrix on HC c_t ─────────────────────────────────────
    clf = LogisticRegression(max_iter=500)
    clf.fit(all_c, all_ctype.astype(int))
    cm  = confusion_matrix(all_ctype.astype(int), clf.predict(all_c))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["C0", "C1", "C2", "C3"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Context confusion matrix (HC c_t)\nEC→HC→CB MSE (shifted)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/confusion.png")

    # ── Fig 6: HC learned alpha over training ─────────────────────────────────
    if history.get("alpha"):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(history["alpha"], color="purple")
        ax.set_xlabel("Epoch"); ax.set_ylabel("HC alpha (leak rate)")
        ax.set_title("Learned HC leak rate over training\nEC→HC→CB MSE (shifted)")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{FIG_DIR}/hc_alpha.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {FIG_DIR}/hc_alpha.png")


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training sequences: {N_TRAIN} | Val: {N_VAL} | Epochs: {EPOCHS}")
    print(f"Architecture: EC (GRU, 256 hidden) → HC (leaky integrator) → CB (MLP, {CB_HIDDEN} hidden)")
    print(f"Shifted training: c_{{t-1}} + [s_t, a_t] → predicts s_{{t+1}}\n")

    tr_trans, _      = generate_sequences(N_TRAIN, seed=0)
    va_trans, va_ctx = generate_sequences(N_VAL,   seed=99)

    train_loader = DataLoader(TransitionDataset(tr_trans), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TransitionDataset(va_trans), batch_size=BATCH_SIZE)

    ec        = EntorhinalCortex(input_dim=6, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    hc        = Hippocampus(latent_dim=LATENT_DIM, hc_dim=HC_DIM).to(device)
    cb        = CerebellumCB(hc_dim=HC_DIM, hidden=CB_HIDDEN).to(device)
    optimizer = torch.optim.Adam(
        list(ec.parameters()) + list(hc.parameters()) + list(cb.parameters()), lr=LR
    )
    history = {"train_mse": [], "val_mse": [], "alpha": []}

    print("Training EC → HC → CB with shifted MSE loss...\n")

    for epoch in range(1, EPOCHS + 1):
        ec.train(); hc.train(); cb.train()
        total_mse = 0.0

        for trans in train_loader:
            trans = trans.to(device)            # (B, T, 6)

            # EC: infer z_t from full transition history up to t
            z, _  = ec(trans)                   # (B, T, LATENT_DIM)

            # HC: consolidate z_t → c_t via leaky integration
            c, _  = hc(z)                       # (B, T, HC_DIM)

            # Shift: c_{t-1} + [s_t, a_t] → predict s_{t+1}
            # c[:, :-1, :]   = c_0 .. c_{T-2}  (never saw s_{T-1} or beyond)
            # trans[:, 1:, ] = transitions 1 .. T-1
            c_prev   = c[:, :-1, :]             # (B, T-1, HC_DIM)
            s_curr   = trans[:, 1:, :2]         # s_{t+1}            (B, T-1, 2)
            a_curr   = trans[:, 1:, 2:4]        # a_{t+1}            (B, T-1, 2)
            s_target = trans[:, 1:, 4:6]        # s_{t+2}  target    (B, T-1, 2)

            s_pred = cb(s_curr, a_curr, c_prev) # (B, T-1, 2)
            loss   = F.mse_loss(s_pred, s_target)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_mse += loss.item() * len(trans)

        total_mse /= N_TRAIN
        history["train_mse"].append(total_mse)
        history["alpha"].append(torch.sigmoid(hc.alpha_raw).item())

        ec.eval(); hc.eval(); cb.eval()
        val_mse = 0.0
        with torch.no_grad():
            for trans in val_loader:
                trans    = trans.to(device)
                z, _     = ec(trans)
                c, _     = hc(z)
                c_prev   = c[:, :-1, :]
                s_curr   = trans[:, 1:, :2]
                a_curr   = trans[:, 1:, 2:4]
                s_target = trans[:, 1:, 4:6]
                s_pred   = cb(s_curr, a_curr, c_prev)
                val_mse += F.mse_loss(s_pred, s_target).item() * len(trans)
        val_mse /= N_VAL
        history["val_mse"].append(val_mse)

        alpha_now = history["alpha"][-1]
        print(f"Epoch {epoch:03d} | Train MSE: {total_mse:.4f} | Val MSE: {val_mse:.4f} "
              f"| HC alpha: {alpha_now:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"ec": ec.state_dict(), "hc": hc.state_dict(),
                "cb": cb.state_dict(), "history": history},
               "checkpoints/ec_hc_cb_mse.pt")
    print("\nSaved checkpoints/ec_hc_cb_mse.pt")

    print("\nEvaluating context representations in z_t (EC) and c_t (HC)...")
    evaluate_and_plot(ec, hc, cb, va_trans, va_ctx, history, device)


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"\nTotal training time: {time.time() - t0:.1f}s")

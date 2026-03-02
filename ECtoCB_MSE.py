"""
ECtoCB_MSE.py

EC → CB trained with only MSE prediction loss.
No context labels are ever given. EC must infer latent context from
the sequence of transitions [s_t, a_t, s_{t+1}] in order to help
CB predict s_{t+1} accurately.

Flow:
  [s_t, a_t, s_{t+1}]  →  input into EC (GRU)  →  z_t  (16-dim latent context)
  [s_t, a_t,    z_t  ]  → input into  CB (MLP)  →  predicted s_{t+1}
                                  ↓
                        MSE(predicted, actual s_{t+1})
                                  ↓
                  gradient flows back through CB → into EC

Data: 4 fixed contexts, each sequence switches between 2 of them.. and switches at a random
time step so model cannot memorise/cheat when the c swicthes
  C0: (beta=+0.4, gamma=0.0) — force along action, positive
  C1: (beta=-0.4, gamma=0.0) — force along action, negative
  C2: (beta=0.0, gamma=+0.4) — force perpendicular, right
  C3: (beta=0.0, gamma=-0.4) — force perpendicular, left
"""
#import needed packages
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


# tell the hyperparameters 
#train on 2k sequences, test n 200 and each seq has 40 time steps 
SEQ_LEN    = 40
N_TRAIN    = 2000
N_VAL      = 200
BATCH_SIZE = 128
EPOCHS     = 100
LR         = 1e-3
LATENT_DIM = 16  #say context has 16 latent Dimensions 
CB_HIDDEN  = 256


# Cerebellar feed fwd MLP

class CerebellumCB(nn.Module):
    """MLP: [s_t, a_t, z_t] → s_{t+1}.  input = 2+2+LATENT_DIM = 20 dims."""
    def __init__(self, latent_dim=LATENT_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),         nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, s, a, z):
        # s: (B, T, 2)  a: (B, T, 2)  z: (B, T, latent_dim)
        x = torch.cat([s, a, z], dim=-1)
        return self.net(x)   # (B, T, 2)


# simulate data --
def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0):
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


# evaluate and make figs 
def evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device, suffix):
    ec.eval(); cb.eval()
    all_z, all_ctype = [], []

    with torch.no_grad():
        for i in range(0, len(va_trans), BATCH_SIZE):
            trans = va_trans[i:i+BATCH_SIZE].to(device)
            z, _  = ec(trans)                       # (B, T, 16)
            ctx   = va_ctx[i:i+BATCH_SIZE]
            B, T, D = z.shape
            all_z.append(z.cpu().reshape(B * T, D))
            all_ctype.append(ctx[:, :, 2].reshape(B * T))

    all_z     = torch.cat(all_z).numpy()       # (N*T, 16)
    all_ctype = torch.cat(all_ctype).numpy()   # (N*T,)

    # Scalar metrics 
    r_per_dim = [abs(stats.pearsonr(all_z[:, d], all_ctype)[0]) for d in range(LATENT_DIM)]
    r_max     = max(r_per_dim)
    r2        = cross_val_score(LinearRegression(), all_z, all_ctype, cv=5, scoring="r2").mean()
    acc       = cross_val_score(LogisticRegression(max_iter=500), all_z,
                                all_ctype.astype(int), cv=5, scoring="accuracy").mean()
    print(f"\nMax |r| z_t vs c_type  : {r_max:.4f}")
    print(f"Linear probe R²         : {r2:.4f}  (1=perfect, 0=chance, <0=worse)")
    print(f"Logistic decode accuracy: {acc:.4f}  (chance=0.25 for 4 contexts)")

    os.makedirs("figs", exist_ok=True)

    colors   = {0: "steelblue", 1: "tomato", 2: "seagreen", 3: "orange"}
    c_labels = {0: "C0 (+beta)", 1: "C1 (-beta)", 2: "C2 (+gamma)", 3: "C3 (-gamma)"}
    title_tag = f"EC→CB MSE ({suffix})"

    # Fig 1: Training curve 
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(history["train_mse"], label="train MSE", color="steelblue")
    ax1.plot(history["val_mse"],   label="val MSE",   color="tomato")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE"); ax1.legend()
    ax1.set_title(f"Training curve — {title_tag}")
    plt.tight_layout()
    plt.savefig(f"figs/ectocb_training_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ectocb_training_{suffix}.png")

    # ── Fig 2: PCA coloured by context ────────────────────────────────────────
    n_plot  = min(2000, len(all_z))
    idx     = np.random.choice(len(all_z), n_plot, replace=False)
    z_sub   = all_z[idx];  ctx_sub = all_ctype[idx]
    pca     = PCA(n_components=2)
    z_2d    = pca.fit_transform(z_sub)

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(4):
        mask = ctx_sub == c
        if mask.sum() > 0:
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=colors[c], label=c_labels[c], alpha=0.4, s=10)
    ax.set_title(f"PCA of z_t — coloured by context\n{title_tag}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figs/ectocb_pca_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ectocb_pca_{suffix}.png")

    # ── Fig 3: Per-dim |r| bar chart 
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(LATENT_DIM), r_per_dim, color="steelblue")
    ax.axhline(0.1, color="tomato", linestyle="--", label="|r|=0.1 reference")
    ax.set_xlabel("z dimension"); ax.set_ylabel("|Pearson r| vs c_type")
    ax.set_title(f"Per-dim correlation with context\n{title_tag}")
    ax.set_xticks(range(LATENT_DIM)); ax.legend()
    plt.tight_layout()
    plt.savefig(f"figs/ectocb_perdim_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ectocb_perdim_{suffix}.png")

    # ── Fig 4: Decoding accuracy over time ────────────────────────────────────
    N_val         = len(va_trans)
    all_z_seq     = all_z.reshape(N_val, SEQ_LEN, LATENT_DIM)
    all_ctype_seq = all_ctype.reshape(N_val, SEQ_LEN)

    acc_over_t = []
    for t in range(SEQ_LEN):
        z_t = all_z_seq[:, t, :]
        c_t = all_ctype_seq[:, t].astype(int)
        a_t = cross_val_score(LogisticRegression(max_iter=500), z_t, c_t,
                              cv=5, scoring="accuracy").mean()
        acc_over_t.append(a_t)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(acc_over_t, color="steelblue", label="decode accuracy")
    ax.axhline(0.25, color="gray",  linestyle="--", label="chance (0.25)")
    ax.axvline(20,   color="tomato", linestyle="--", label="avg switch (t≈20)")
    ax.set_xlabel("Timestep t"); ax.set_ylabel("Context decode accuracy")
    ax.set_title(f"Context decodability over time\n{title_tag}")
    ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"figs/ectocb_decodetime_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ectocb_decodetime_{suffix}.png")

    # ── Fig 5: Confusion matrix ───────────────────────────────────────────────
    clf = LogisticRegression(max_iter=500)
    clf.fit(all_z, all_ctype.astype(int))
    cm  = confusion_matrix(all_ctype.astype(int), clf.predict(all_z))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["C0", "C1", "C2", "C3"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Context confusion matrix\n{title_tag}")
    plt.tight_layout()
    plt.savefig(f"figs/ectocb_confusion_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ectocb_confusion_{suffix}.png")


# ── Training 
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training sequences: {N_TRAIN} | Val: {N_VAL} | Epochs: {EPOCHS}")

    tr_trans, _      = generate_sequences(N_TRAIN, seed=0)
    va_trans, va_ctx = generate_sequences(N_VAL,   seed=99)

    train_loader = DataLoader(TransitionDataset(tr_trans), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TransitionDataset(va_trans), batch_size=BATCH_SIZE)

    ec        = EntorhinalCortex(input_dim=6, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    cb        = CerebellumCB(latent_dim=LATENT_DIM, hidden=CB_HIDDEN).to(device)
    optimizer = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)
    history   = {"train_mse": [], "val_mse": []}

    print("\nTraining EC → CB with MSE prediction loss...\n")

    for epoch in range(1, EPOCHS + 1):
        ec.train(); cb.train()
        total_mse = 0.0

        for trans in train_loader:
            trans = trans.to(device)
            s     = trans[:, :, :2]   # (B, T, 2) — current state
            a     = trans[:, :, 2:4]  # (B, T, 2) — action
            s_next= trans[:, :, 4:6]  # (B, T, 2) — next state (target)

            z, _      = ec(trans)              # (B, T, 16)
            s_pred    = cb(s, a, z)            # (B, T, 2)
            loss      = F.mse_loss(s_pred, s_next)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_mse += loss.item() * len(trans)

        total_mse /= N_TRAIN
        history["train_mse"].append(total_mse)

        ec.eval(); cb.eval(); val_mse = 0.0
        with torch.no_grad():
            for trans in val_loader:
                trans  = trans.to(device)
                s      = trans[:, :, :2]
                a      = trans[:, :, 2:4]
                s_next = trans[:, :, 4:6]
                z, _   = ec(trans)
                s_pred = cb(s, a, z)
                val_mse += F.mse_loss(s_pred, s_next).item() * len(trans)
        val_mse /= N_VAL
        history["val_mse"].append(val_mse)
        print(f"Epoch {epoch:03d} | Train MSE: {total_mse:.4f} | Val MSE: {val_mse:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "history": history},
               "checkpoints/ectocb_mse.pt")
    print("\nSaved checkpoints/ectocb_mse.pt")

    print("\nEvaluating context encoding in z_t...")
    evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device, suffix="unshifted")


# SECTION 2: Time-shifted EC → CB
#   EC uses transitions 0..t  to spit out  z_t
#   CB uses [s_{t+1}, a_{t+1}, z_t] as inputs and predicts s_{t+2}
#
#  now -- z_t has NOT seen s_{t+2}, so EC cannot cheat.(that it could in section 1)
#   Gradient from prediction error still flows back through CB → EC.


def train_shifted():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print("SECTION 2: Time-shifted EC → CB")
    print(f"{'='*60}")
    print(f"Device: {device} | n_train: {N_TRAIN} | epochs: {EPOCHS}")

    tr_trans, _      = generate_sequences(N_TRAIN, seed=0)
    va_trans, va_ctx = generate_sequences(N_VAL,   seed=99)

    train_loader = DataLoader(TransitionDataset(tr_trans), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TransitionDataset(va_trans), batch_size=BATCH_SIZE)

    ec        = EntorhinalCortex(input_dim=6, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    cb        = CerebellumCB(latent_dim=LATENT_DIM, hidden=CB_HIDDEN).to(device)
    optimizer = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)
    history   = {"train_mse": [], "val_mse": []}

    print("\nTraining (shifted: z_t predicts one step ahead)...\n")

    for epoch in range(1, EPOCHS + 1):
        ec.train(); cb.train()
        total_mse = 0.0

        for trans in train_loader:
            trans = trans.to(device)            # (B, T, 6)

            z, _ = ec(trans)                    # (B, T, 16)  — z[:,t,:] from trans 0..t

            # Shift: use z_t with [s_{t+1}, a_{t+1}] to predict s_{t+2}
            # z[:, :-1, :]   = z_0 .. z_{T-2}   (doesn't contain s_{T-1} or beyond)
            # trans[:, 1:, ] = transitions 1..T-1
            z_prev  = z[:, :-1, :]              # (B, T-1, 16)
            s_curr  = trans[:, 1:, :2]          # s_{t+1}      (B, T-1, 2)
            a_curr  = trans[:, 1:, 2:4]         # a_{t+1}      (B, T-1, 2)
            s_target= trans[:, 1:, 4:6]         # s_{t+2}      (B, T-1, 2)

            s_pred = cb(s_curr, a_curr, z_prev) # (B, T-1, 2)
            loss   = F.mse_loss(s_pred, s_target)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_mse += loss.item() * len(trans)

        total_mse /= N_TRAIN
        history["train_mse"].append(total_mse)

        ec.eval(); cb.eval(); val_mse = 0.0
        with torch.no_grad():
            for trans in val_loader:
                trans   = trans.to(device)
                z, _    = ec(trans)
                z_prev  = z[:, :-1, :]
                s_curr  = trans[:, 1:, :2]
                a_curr  = trans[:, 1:, 2:4]
                s_target= trans[:, 1:, 4:6]
                s_pred  = cb(s_curr, a_curr, z_prev)
                val_mse += F.mse_loss(s_pred, s_target).item() * len(trans)
        val_mse /= N_VAL
        history["val_mse"].append(val_mse)
        print(f"Epoch {epoch:03d} | Train MSE: {total_mse:.4f} | Val MSE: {val_mse:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "history": history},
               "checkpoints/ectocb_mse_shifted.pt")
    print("\nSaved checkpoints/ectocb_mse_shifted.pt")

    print("\nEvaluating context encoding in z_t...")
    evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device, suffix="shifted")


if __name__ == "__main__":
    t0 = time.time()

    # ── Section 1: EC sees full transition, predicts same-step s_{t+1} ────────
    # train()

    # ── Section 2: z_t predicts one step ahead — no shortcut ──────────────────
    train_shifted()

    print(f"\nTotal training time: {time.time() - t0:.1f}s")

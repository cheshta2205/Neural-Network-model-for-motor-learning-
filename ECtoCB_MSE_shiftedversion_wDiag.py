"""
ECtoCB_MSE_shiftedversion_wDiag.py

Same as ECtoCB_MSE_shiftedversion.py but trained on 8 contexts:
  Axis-aligned: (±0.4, 0), (0, ±0.4)         [same as original 4]
  Diagonal:     (±0.4, ±0.4)                  [4 new diagonal contexts]

Checkpoint saved to: checkpoints/ectocb_mse_shiftedv2_wDiag.pt   (new file)
Figures saved to:    ../../figs/ECtoCB/ECtoCB_MSE_shiftedversion_wDiag/  (new folder)

The original checkpoint checkpoints/ectocb_mse_shiftedv2.pt is NOT touched.
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


# ─ Hyperparameters
SEQ_LEN    = 40
N_TRAIN    = 2000
N_VAL      = 200
BATCH_SIZE = 128
EPOCHS     = 100
LR         = 1e-3
LATENT_DIM = 16
EC_HIDDEN  = 256
CB_HIDDEN  = 256
EC_INPUT   = 8
FIG_DIR    = "../../figs/ECtoCB/ECtoCB_MSE_shiftedversion_wDiag"
CKPT_OUT   = "checkpoints/ectocb_mse_shiftedv2_wDiag.pt"

# 8 contexts: 4 axis-aligned + 4 diagonal
ALL_CTX_OPTS = [
    ( 0.4,  0.0, 0),   # C0: +beta
    (-0.4,  0.0, 1),   # C1: -beta
    ( 0.0,  0.4, 2),   # C2: +gamma
    ( 0.0, -0.4, 3),   # C3: -gamma
    ( 0.4,  0.4, 4),   # C4: diag ++
    (-0.4,  0.4, 5),   # C5: diag -+
    ( 0.4, -0.4, 6),   # C6: diag +-
    (-0.4, -0.4, 7),   # C7: diag --
]
N_CONTEXTS = len(ALL_CTX_OPTS)


# EC: GRUCell

class EntorhinalCortex(nn.Module):
    def __init__(self, input_dim=EC_INPUT, hidden_dim=EC_HIDDEN, latent_dim=LATENT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell       = nn.GRUCell(input_dim, hidden_dim)
        self.readout    = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(self, x, h):
        h_new = self.cell(x, h)
        z     = self.readout(h_new)
        return z, h_new


# CB: MLP forward model

class CerebellumCB(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),         nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))


# Data generation — picks 2 of 8 contexts per sequence

def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0):
    rng  = np.random.default_rng(seed)
    acts = [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]

    all_trans, all_ctx = [], []

    for _ in range(n_seqs):
        x, y = float(rng.integers(0, 5)), float(rng.integers(0, 5))

        c_ids        = rng.choice(N_CONTEXTS, size=2, replace=False)
        ctx_first, ctx_second = ALL_CTX_OPTS[c_ids[0]], ALL_CTX_OPTS[c_ids[1]]
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
        torch.tensor(np.array(all_trans), dtype=torch.float32),
        torch.tensor(np.array(all_ctx),   dtype=torch.float32),
    )


class TransitionDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions
    def __len__(self): return len(self.transitions)
    def __getitem__(self, idx): return self.transitions[idx]


# Core sequential forward pass

def run_sequence(ec, cb, trans, device):
    B, T, _ = trans.shape
    zeros2   = torch.zeros(B, 2, device=device)

    h      = ec.init_hidden(B, device)
    z_t    = torch.zeros(B, LATENT_DIM, device=device)
    s_pred = zeros2.clone()

    z_seq      = []
    s_pred_seq = []

    for t in range(T):
        s_t = trans[:, t, :2]
        a_t = trans[:, t, 2:4]

        if t == 0:
            s_tm1 = zeros2
            a_tm1 = zeros2
            e_t   = zeros2
        else:
            s_tm1 = trans[:, t-1, :2]
            a_tm1 = trans[:, t-1, 2:4]
            e_t   = s_t - s_pred

        ec_input = torch.cat([s_tm1, a_tm1, s_t, e_t], dim=-1)
        z_t, h   = ec.step(ec_input, h)
        s_pred   = cb(s_t, a_t, z_t)

        z_seq.append(z_t)
        s_pred_seq.append(s_pred)

    z_seq      = torch.stack(z_seq,      dim=1)
    s_pred_seq = torch.stack(s_pred_seq, dim=1)

    return z_seq, s_pred_seq


# Evaluation + plots

def evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device):
    ec.eval(); cb.eval()
    all_z, all_ctype = [], []

    with torch.no_grad():
        for i in range(0, len(va_trans), BATCH_SIZE):
            trans = va_trans[i:i+BATCH_SIZE].to(device)
            ctx   = va_ctx[i:i+BATCH_SIZE]
            z, _  = run_sequence(ec, cb, trans, device)
            B, T, D = z.shape
            all_z.append(z.cpu().reshape(B * T, D))
            all_ctype.append(ctx[:, :, 2].reshape(B * T))

    all_z     = torch.cat(all_z).numpy()
    all_ctype = torch.cat(all_ctype).numpy()

    # Per-neuron logistic decode accuracy (8 contexts)
    C_onehot    = np.eye(N_CONTEXTS)[all_ctype.astype(int)]
    acc_per_dim = [
        cross_val_score(LogisticRegression(max_iter=500), all_z[:, d:d+1],
                        all_ctype.astype(int), cv=5, scoring="accuracy").mean()
        for d in range(LATENT_DIM)
    ]
    acc_max = max(acc_per_dim)

    r2  = cross_val_score(LinearRegression(), all_z, C_onehot, cv=5, scoring="r2").mean()
    acc = cross_val_score(LogisticRegression(max_iter=500), all_z,
                          all_ctype.astype(int), cv=5, scoring="accuracy").mean()
    chance = 1.0 / N_CONTEXTS
    print(f"\nMax per-neuron decode acc: {acc_max:.4f}  (chance={chance:.3f})")
    print(f"Linear probe R² (onehot) : {r2:.4f}  (1=perfect)")
    print(f"Logistic decode accuracy : {acc:.4f}  (chance={chance:.3f})")

    os.makedirs(FIG_DIR, exist_ok=True)

    colors = {
        0: "steelblue", 1: "tomato",    2: "seagreen",  3: "orange",
        4: "purple",    5: "brown",     6: "deeppink",  7: "gray",
    }
    c_labels = {
        0: "C0 (+β)",       1: "C1 (-β)",
        2: "C2 (+γ)",       3: "C3 (-γ)",
        4: "C4 (+β+γ)",     5: "C5 (-β+γ)",
        6: "C6 (+β-γ)",     7: "C7 (-β-γ)",
    }
    tag = "EC→CB MSE (8 contexts: axis-aligned + diagonal ±0.4)"

    # Fig 1: Training curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_mse"], label="train MSE", color="steelblue")
    ax.plot(history["val_mse"],   label="val MSE",   color="tomato")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title(f"Training curve — {tag}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/training.png")

    # Fig 2: PCA coloured by context
    n_plot = min(2000, len(all_z))
    idx    = np.random.choice(len(all_z), n_plot, replace=False)
    z_sub  = all_z[idx]; ctx_sub = all_ctype[idx]
    pca    = PCA(n_components=2)
    z_2d   = pca.fit_transform(z_sub)

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(N_CONTEXTS):
        mask = ctx_sub == c
        if mask.sum() > 0:
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=colors[c], label=c_labels[c], alpha=0.4, s=10)
    ax.set_title(f"PCA of z_t — coloured by context\n{tag}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/pca.png")

    # Fig 3: Per-dim decode accuracy bar chart
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(LATENT_DIM), acc_per_dim, color="steelblue")
    ax.axhline(chance, color="tomato", linestyle="--", label=f"chance ({chance:.3f})")
    ax.set_xlabel("z dimension"); ax.set_ylabel("Logistic decode accuracy (per neuron)")
    ax.set_title(f"Per-dim context decodability (8 contexts)\n{tag}")
    ax.set_xticks(range(LATENT_DIM)); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/perdim.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/perdim.png")

    # Fig 4: Decoding accuracy over time
    N_val         = len(va_trans)
    all_z_seq     = all_z.reshape(N_val, SEQ_LEN, LATENT_DIM)
    all_ctype_seq = all_ctype.reshape(N_val, SEQ_LEN)

    acc_over_t = []
    for t in range(SEQ_LEN):
        z_t = all_z_seq[:, t, :]
        c_t = all_ctype_seq[:, t].astype(int)
        acc_over_t.append(
            cross_val_score(LogisticRegression(max_iter=500), z_t, c_t,
                            cv=5, scoring="accuracy").mean()
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(acc_over_t, color="steelblue", label="decode accuracy")
    ax.axhline(chance, color="gray",  linestyle="--", label=f"chance ({chance:.3f})")
    ax.axvline(20,     color="tomato", linestyle="--", label="avg switch (t≈20)")
    ax.set_xlabel("Timestep t"); ax.set_ylabel("Context decode accuracy")
    ax.set_title(f"Context decodability over time\n{tag}")
    ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/decodetime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/decodetime.png")

    # Fig 5: Confusion matrix
    clf = LogisticRegression(max_iter=500)
    clf.fit(all_z, all_ctype.astype(int))
    cm  = confusion_matrix(all_ctype.astype(int), clf.predict(all_z))
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=[c_labels[i] for i in range(N_CONTEXTS)])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Context confusion matrix\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/confusion.png")

    # Fig 5b: Univariate vs joint decoder importance
    decoder_importance = np.linalg.norm(clf.coef_, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 3), sharey=False)
    axes[0].bar(range(LATENT_DIM), acc_per_dim, color="steelblue")
    axes[0].axhline(chance, color="tomato", linestyle="--", label=f"chance ({chance:.3f})")
    axes[0].set_xlabel("EC neuron"); axes[0].set_ylabel("Logistic decode accuracy (per neuron)")
    axes[0].set_title("Univariate context decodability\n(independent, one neuron at a time)")
    axes[0].set_xticks(range(LATENT_DIM)); axes[0].legend()

    axes[1].bar(range(LATENT_DIM), decoder_importance, color="seagreen")
    axes[1].set_xlabel("EC neuron"); axes[1].set_ylabel("||coef|| across context classes")
    axes[1].set_title("Linear decoder weight norm\n(joint — unique contribution per neuron)")
    axes[1].set_xticks(range(LATENT_DIM))

    fig.suptitle(f"Univariate vs joint context encoding per EC neuron\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/decoder_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/decoder_importance.png")

    # GLM + SVD: context subspace analysis
    W = clf.coef_                                   # (N_CONTEXTS, 16)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    n_axes = min(N_CONTEXTS, len(S))

    print(f"\nContext subspace singular values: {S.round(3)}")
    print(f"Variance explained by top axis  : {S[0]**2 / (S**2).sum() * 100:.1f}%")

    # Fig 6: Singular values
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(1, len(S) + 1), S, color="steelblue")
    ax.set_xlabel("Context axis"); ax.set_ylabel("Singular value")
    ax.set_title(f"Dimensionality of context subspace\n{tag}")
    ax.set_xticks(range(1, len(S) + 1))
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/glm_singular_values.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/glm_singular_values.png")

    # Fig 6b: U heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(U, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_axes))
    ax.set_xticklabels([f"axis{i+1}" for i in range(n_axes)], fontsize=8)
    ax.set_yticks(range(N_CONTEXTS))
    ax.set_yticklabels([c_labels[i] for i in range(N_CONTEXTS)], fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"U — context class loadings on SVD axes\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svd_U_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/svd_U_heatmap.png")

    # Fig 7: Top context axes
    show_axes = min(4, n_axes)
    fig, axes = plt.subplots(1, show_axes, figsize=(4 * show_axes, 3), sharey=False)
    if show_axes == 1: axes = [axes]
    for i, ax in enumerate(axes):
        ax.bar(range(LATENT_DIM), Vt[i], color="steelblue")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Context axis {i+1}\n({S[i]**2/(S**2).sum()*100:.0f}% var)")
        ax.set_xlabel("EC output neuron"); ax.set_xticks(range(LATENT_DIM))
    fig.suptitle(f"GLM context axes (from SVD of W)\n{tag}", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/glm_context_axes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/glm_context_axes.png")

    # Fig 8: CB weight alignment with context axes
    cb_W_z = cb.net[0].weight.detach().cpu().numpy()[:, 4:]   # (256, 16)
    alignment        = cb_W_z @ Vt[:show_axes].T              # (256, show_axes)
    cb_w_norm        = np.linalg.norm(cb_W_z, axis=1, keepdims=True) + 1e-8
    alignment_normed = np.linalg.norm(alignment, axis=1) / cb_w_norm[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(alignment_normed, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(alignment_normed.mean(), color="tomato", linestyle="--",
                    label=f"mean={alignment_normed.mean():.2f}")
    axes[0].set_xlabel("Fraction of z_t weight in context subspace")
    axes[0].set_ylabel("Number of CB neurons")
    axes[0].set_title("CB alignment with EC context subspace")
    axes[0].legend()

    axis_alignment = np.linalg.norm(alignment, axis=0)
    axes[1].bar(range(1, show_axes + 1), axis_alignment, color="steelblue")
    axes[1].set_xlabel("Context axis"); axes[1].set_ylabel("Total CB weight (L2)")
    axes[1].set_title("CB weight on each EC context axis")
    axes[1].set_xticks(range(1, show_axes + 1))

    fig.suptitle(f"Do CB's synapses read EC's context subspace?\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cb_context_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/cb_context_alignment.png")

    # Fig 9: Per-neuron ablation
    def cb_mse_with_z(z_seq_np):
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, N_val, BATCH_SIZE):
                trans    = va_trans[i:i+BATCH_SIZE].to(device)
                z        = torch.tensor(z_seq_np[i:i+BATCH_SIZE], dtype=torch.float32, device=device)
                s_curr   = trans[:, :, :2].reshape(-1, 2)
                a_curr   = trans[:, :, 2:4].reshape(-1, 2)
                s_target = trans[:, :, 4:6].reshape(-1, 2)
                z_flat   = z.reshape(-1, LATENT_DIM)
                total   += F.mse_loss(cb(s_curr, a_curr, z_flat), s_target,
                                      reduction='sum').item()
                count   += s_target.shape[0]
        return total / count

    mse_baseline = cb_mse_with_z(all_z_seq)
    print(f"\nAblation baseline MSE: {mse_baseline:.4f}")

    delta_mse = np.zeros(LATENT_DIM)
    for d in range(LATENT_DIM):
        z_abl = all_z_seq.copy()
        z_abl[:, :, d] = 0.0
        delta_mse[d] = cb_mse_with_z(z_abl) - mse_baseline
        print(f"  ablate z{d:2d}  Δmse={delta_mse[d]:+.4f}  (acc={acc_per_dim[d]:.2f})")

    sorted_dims = np.argsort(acc_per_dim)[::-1]
    prog_mse = [mse_baseline]
    for k in range(1, LATENT_DIM + 1):
        z_abl = all_z_seq.copy()
        z_abl[:, :, sorted_dims[:k]] = 0.0
        prog_mse.append(cb_mse_with_z(z_abl))

    sorted_idx = np.argsort(acc_per_dim)[::-1]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(LATENT_DIM), delta_mse[sorted_idx], color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(LATENT_DIM))
    ax.set_xticklabels(
        [f"z{sorted_idx[i]}\nacc={acc_per_dim[sorted_idx[i]]:.2f}" for i in range(LATENT_DIM)],
        fontsize=7)
    ax.set_xlabel("z neuron (sorted by per-neuron decode accuracy)")
    ax.set_ylabel("ΔMSE (ablated − baseline)")
    ax.set_title(f"Per-neuron ablation effect on CB MSE\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ablation_per_neuron.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/ablation_per_neuron.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(acc_per_dim, delta_mse, color="steelblue", edgecolors="white", s=60, zorder=3)
    for d in range(LATENT_DIM):
        ax.annotate(f"z{d}", (acc_per_dim[d], delta_mse[d]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Per-neuron decode accuracy (8 contexts)")
    ax.set_ylabel("ΔMSE when ablated")
    ax.set_title(f"Context encoding vs CB reliance per neuron\n{tag}")
    ax.axhline(0, color="gray", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ablation_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/ablation_scatter.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(LATENT_DIM + 1), prog_mse, color="steelblue", marker="o")
    ax.axhline(mse_baseline, color="gray", linestyle="--",
               label=f"baseline MSE = {mse_baseline:.4f}")
    ax.set_xlabel("Neurons ablated (highest decode accuracy first)")
    ax.set_ylabel("CB MSE")
    ax.set_title(f"Progressive ablation — high-context neurons first\n{tag}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ablation_progressive.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/ablation_progressive.png")

    # Fig 10: Context subspace ablation (SVD-based)
    Vt_k_all = Vt[:show_axes]
    ctx_prog_mse = [mse_baseline]
    for k in range(1, show_axes + 1):
        Vt_k  = Vt_k_all[:k]
        Pc_k  = Vt_k.T @ Vt_k
        z_abl = all_z_seq - all_z_seq @ Pc_k
        ctx_prog_mse.append(cb_mse_with_z(z_abl))
        print(f"  context subspace ablation top-{k} axes  MSE={ctx_prog_mse[-1]:.4f}  "
              f"ΔMSE={ctx_prog_mse[-1]-mse_baseline:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(show_axes + 1), ctx_prog_mse, color="seagreen", marker="o")
    axes[0].axhline(mse_baseline, color="gray", linestyle="--",
                    label=f"baseline = {mse_baseline:.4f}")
    axes[0].set_xlabel("Context axes removed (strongest first)")
    axes[0].set_ylabel("CB MSE")
    axes[0].set_title("Progressive context subspace ablation\n(SVD-based, removes full axes)")
    axes[0].set_xticks(range(show_axes + 1))
    axes[0].legend()

    full_abl_mse    = ctx_prog_mse[-1]
    best_neuron_mse = mse_baseline + max(delta_mse)
    axes[1].bar(["baseline", "best single\nneuron ablation", "full context\nsubspace ablation"],
                [mse_baseline, best_neuron_mse, full_abl_mse],
                color=["gray", "steelblue", "seagreen"])
    axes[1].set_ylabel("CB MSE")
    axes[1].set_title("Context subspace vs single neuron ablation")

    fig.suptitle(f"Does CB use the context subspace?\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ctx_subspace_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR}/ctx_subspace_ablation.png")


# ── Training
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Contexts: {N_CONTEXTS} (4 axis-aligned + 4 diagonal ±0.4)")
    print(f"Sequences: train={N_TRAIN} | val={N_VAL} | seq_len={SEQ_LEN} | epochs={EPOCHS}")
    print(f"EC input: [s_{{t-1}}, a_{{t-1}}, s_t, e_t] ({EC_INPUT} dims)")
    print(f"CB: [s_t, a_t, z_t] → s_{{t+1}}\n")

    tr_trans, _      = generate_sequences(N_TRAIN, seed=0)
    va_trans, va_ctx = generate_sequences(N_VAL,   seed=99)

    train_loader = DataLoader(TransitionDataset(tr_trans), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TransitionDataset(va_trans), batch_size=BATCH_SIZE)

    ec        = EntorhinalCortex(input_dim=EC_INPUT, hidden_dim=EC_HIDDEN, latent_dim=LATENT_DIM).to(device)
    cb        = CerebellumCB(latent_dim=LATENT_DIM, hidden=CB_HIDDEN).to(device)
    optimizer = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)
    history   = {"train_mse": [], "val_mse": []}

    print("Training EC → CB with prediction-error feedback into EC (8 contexts)...\n")

    for epoch in range(1, EPOCHS + 1):
        ec.train(); cb.train()
        total_mse = 0.0

        for trans in train_loader:
            trans  = trans.to(device)
            s_next = trans[:, :, 4:6]
            _, s_pred = run_sequence(ec, cb, trans, device)
            loss = F.mse_loss(s_pred, s_next)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_mse += loss.item() * len(trans)

        total_mse /= N_TRAIN
        history["train_mse"].append(total_mse)

        ec.eval(); cb.eval()
        val_mse = 0.0
        with torch.no_grad():
            for trans in val_loader:
                trans  = trans.to(device)
                s_next = trans[:, :, 4:6]
                _, s_pred = run_sequence(ec, cb, trans, device)
                val_mse += F.mse_loss(s_pred, s_next).item() * len(trans)
        val_mse /= N_VAL
        history["val_mse"].append(val_mse)

        print(f"Epoch {epoch:03d} | Train MSE: {total_mse:.4f} | Val MSE: {val_mse:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "history": history},
               CKPT_OUT)
    print(f"\nSaved {CKPT_OUT}")

    print("\nEvaluating context encoding in z_t...")
    evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device)


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"\nTotal training time: {time.time() - t0:.1f}s")

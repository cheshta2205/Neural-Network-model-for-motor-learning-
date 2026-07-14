"""
ECtoCB_train_MSE_12c.py
for paper FIG. 
EC → CB trained on velocity-based dynamics with 12 contexts (rotation, curl, drift).
Same as ECtoCB_newcontexts_train.py but without the scale contexts (C6, C7).

Dynamics -
--------
   5 axis do
   
    V_t     = V_{t-1} + F_total
    S_{t+1} = clamp(S_t + Δt·V_t,  0, GRID_SIZE)

where:
    β     = action scaling
    θ     = action rotation angle
    γ     = curl strength (acts on previous velocity, NOT action)
    g, b  = drift strength and direction

Velocity is NOT provided as explicit input to EC or CB.
EC infers it implicitly from consecutive states [s_{t-1}, s_t].

Architecture (unchanged from ECtoCB_MSE_shiftedversion.py)

    EC input : [s_{t-1}, a_{t-1}, s_t, ε_t]   — 8 dims
    EC output: z_t                              — LATENT_DIM
    CB input : [s_t, a_t, z_t]                 — 4 + LATENT_DIM dims
    CB output: s_{t+1} (predicted)             — 2 dims

Contexts (12, switching once per sequence)
-------------------------------------------
    C0: β=1.0, θ=+π/2, γ=0,    g=0             — rotation+
    C1: β=1.0, θ=+π/4, γ=0,    g=0             — rotation+
    C2: β=1.0, θ=-π/2, γ=0,    g=0             — rotation-
    C3: β=1.0, θ=-π/4, γ=0,    g=0             — rotation-
    C4: β=1.0, θ=0,    γ=+0.7, g=0             — curl+
    C5: β=1.0, θ=0,    γ=+0.4, g=0             — curl+
    C6: β=1.0, θ=0,    γ=-0.7, g=0             — curl-
    C7: β=1.0, θ=0,    γ=-0.4, g=0             — curl-
    C8: β=1.0, θ=0,    γ=0,    g=0.4, b=(+1.5,0) — drift++
    C9: β=1.0, θ=0,    γ=0,    g=0.4, b=(+1.0,0) — drift+
    C10: β=1.0, θ=0,   γ=0,    g=0.4, b=(-1.5,0) — drift--
    C11: β=1.0, θ=0,   γ=0,    g=0.4, b=(-1.0,0) — drift-

β=1.0 default. Positive + negative for each FF type.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from src.PlottingTools.plotting_functions import set_plt_params, make_axis_nice, make_3d_gif


# ── Contexts ───────────────────────────────────────────────────────────────────
CONTEXTS = [
    # (beta,  theta,      gamma,  g,    bx,   by,  c_id, label)
    (1.0,  +np.pi/2,  0.0,    0.0,  0.0,  0.0,  0,  "C0: rot++", 0),
    (1.0,  +np.pi/4,  0.0,    0.0,  0.0,  0.0,  1,  "C1: rot+", 0),
    (1.0,  -np.pi/2,  0.0,    0.0,  0.0,  0.0,  2,  "C2: rot--", 0),
    (1.0,  -np.pi/4,  0.0,    0.0,  0.0,  0.0,  3,  "C3: rot-", 0),
    (1.0,   0.0,     +0.7,    0.0,  0.0,  0.0,  4,  "C4: curl++", 1),
    (1.0,   0.0,     +0.4,    0.0,  0.0,  0.0,  5,  "C5: curl+", 1),
    (1.0,   0.0,     -0.7,    0.0,  0.0,  0.0,  6,  "C6: curl--", 1),
    (1.0,   0.0,     -0.4,    0.0,  0.0,  0.0,  7,  "C7: curl-", 1),
    (1.0,   0.0,      0.0,    0.4, +1.5,  0.0,  8,  "C8: drift++", 2),
    (1.0,   0.0,      0.0,    0.4, +1.0,  0.0,  9,  "C9: drift+", 2),
    (1.0,   0.0,      0.0,    0.4, -1.5,  0.0,  10,  "C10: drift--", 2),
    (1.0,   0.0,      0.0,    0.4, -1.0,  0.0,  11,  "C11: drift-", 2),
]

C_LABELS = {c[6]: c[7] for c in CONTEXTS}

COLORS = {0: "#1c96c5", 1: "#20a7db", 2: "#62c1e5", 3: "#a0d9ef",
        4: "#274e13", 5: "#38761d", 6: "#6aa84f", 7: "#93c47d",
        8: "#800080", 9: "#be29ec", 10: "#d896ff", 11: "#efbbff"}

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN    = 80
N_TRAIN    = 15000
N_VAL      = 200
BATCH_SIZE = 256
EPOCHS     = 50
LR         = 1e-3
LATENT_DIM = 64
EC_HIDDEN  = 256
CB_HIDDEN  = 256
EC_INPUT   = 8
GRID_SIZE  = 4
DT         = 1.0   # timestep

CKPT_PATH  = f"./checkpoints/ectocb_{len(CONTEXTS)}ctx_seq{SEQ_LEN}_z{LATENT_DIM}/"
FIG_DIR    = f"./figs/ectocb_{len(CONTEXTS)}ctx_seq{SEQ_LEN}_z{LATENT_DIM}.pt"
CKPT_NAME = f"ectocb_{len(CONTEXTS)}ctx_seq{SEQ_LEN}_z{LATENT_DIM}"


def seeded_ckpt_path(seed):
    """Return the CKPT_PATH with a `_seed{seed}` suffix before the extension."""

    base_path = Path(CKPT_PATH) / f"{CKPT_NAME}_seed{seed}.pt"
    return str(base_path)

# ── Environment step ───────────────────────────────────────────────────────────
def env_step(pos, vel, action, beta, theta, gamma, g, bx, by):
    """
    One environment step under velocity-based dynamics.

    F_total = β·R(θ)·a  +  γ·curl·V  +  g·b
    V_new   = V + F_total
    S_new   = clamp(S + dt·V_new, 0, GRID_SIZE)

    All inputs/outputs are numpy arrays.
    """
    # rotation matrix R(theta)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # curl matrix [[0,-1],[1,0]]
    curl = np.array([[0., -1.],
                     [1.,  0.]])

    # drift vector
    b = np.array([bx, by], dtype=np.float32)

    F = beta * (R @ action) + gamma * (curl @ vel) + g * b
    vel_new = vel + F
    pos_new = np.clip(pos + DT * vel_new, 0, GRID_SIZE)

    return pos_new.astype(np.float32), vel_new.astype(np.float32)


# ── Data generation ────────────────────────────────────────────────────────────
def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0):
    """
    Each sequence has exactly 2 contexts chosen without replacement.
    Switch point sampled uniformly in [40, 60].

    Actions are continuous uniform in [-1, 1]^2 (MC-like).

    Returns:
        transitions : (N, T, 6)  — [s_t(2), a_t(2), s_{t+1}(2)]
        ctx_labels  : (N, T, 3)  — [beta, gamma, c_id] for analysis
    """
    rng = np.random.default_rng(seed)

    all_trans, all_ctx = [], []

    for _ in range(n_seqs):
        # random start position, zero initial velocity
        pos = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
        vel = np.zeros(2, dtype=np.float32)

        # pick 2 contexts without replacement
        c_ids  = rng.choice(len(CONTEXTS), size=2, replace=False)
        ctx_a  = CONTEXTS[c_ids[0]]
        ctx_b  = CONTEXTS[c_ids[1]]
        switch = int(rng.integers(40, 61))

        seq_trans, seq_ctx = [], []

        for t in range(seq_len):
            beta, theta, gamma, g, bx, by, c_id, _, _ = (ctx_a if t < switch else ctx_b)

            # continuous random action in [-1, 1]^2
            action = rng.uniform(-1, 1, size=2).astype(np.float32)

            pos_next, vel_next = env_step(pos, vel, action,
                                          beta, theta, gamma, g, bx, by)

            seq_trans.append([pos[0], pos[1], action[0], action[1],
                               pos_next[0], pos_next[1]])
            seq_ctx.append([beta, gamma, float(c_id)])

            pos = pos_next
            vel = vel_next

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


# ── Models (unchanged architecture) ───────────────────────────────────────────
class EntorhinalCortex(nn.Module):
    """
    EC as a step-by-step GRU.
    Input per step : [s_{t-1}, a_{t-1}, s_t, e_t]  — 8 dims
    Output per step: z_t                             — LATENT_DIM

    Velocity is NOT an explicit input. EC infers it implicitly from
    consecutive states s_{t-1} and s_t in its input.
    """
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


class CerebellumCB(nn.Module):
    """MLP: [s_t, a_t, z_t] → s_{t+1}.  input = 4 + LATENT_DIM dims."""
    def __init__(self, latent_dim=LATENT_DIM, hidden=CB_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),         nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))


# ── Sequential forward pass (unchanged) ───────────────────────────────────────
def run_sequence(ec, cb, trans, device, return_representations=False):
    """
    Step through a batch of sequences feeding CB prediction error back into EC.

    At each step t:
      1. e_t = s_t - CB(s_{t-1}, a_{t-1}, z_{t-1})
      2. EC: z_t = EC([s_{t-1}, a_{t-1}, s_t, e_t])
      3. CB: s_pred_{t+1} = CB(s_t, a_t, z_t)

    trans : (B, T, 6)

    Returns z_seq (B,T,LATENT_DIM), s_pred_seq (B,T,2)
    """
    B, T, _ = trans.shape
    zeros2  = torch.zeros(B, 2, device=device)

    h      = ec.init_hidden(B, device)
    z_t    = torch.zeros(B, LATENT_DIM, device=device)

    s_pred = zeros2.clone()

    z_seq, s_pred_seq = [], []
    ec_seq, cb_1_seq, cb_2_seq = [], [], []


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

        cb_input = torch.cat([s_tm1, a_tm1, z_t], dim=-1)
        cb_h1    = cb.net[1](cb.net[0](cb_input))
        cb_h2    = cb.net[3](cb.net[2](cb_h1))

        s_pred   = cb(s_t, a_t, z_t)

        z_seq.append(z_t)
        ec_seq.append(h)
        cb_1_seq.append(cb_h1)
        cb_2_seq.append(cb_h2)

        s_pred_seq.append(s_pred)

    z_seq      = torch.stack(z_seq,      dim=1)
    ec_seq     = torch.stack(ec_seq,     dim=1)
    cb_1_seq   = torch.stack(cb_1_seq,   dim=1)
    cb_2_seq   = torch.stack(cb_2_seq,   dim=1)

    s_pred_seq = torch.stack(s_pred_seq, dim=1)

    representations = {
        'EC_hidden': ec_seq,
        'CB_hidden_1': cb_1_seq,
        'CB_hidden_2': cb_2_seq
    }

    if return_representations:
        return z_seq, s_pred_seq, representations

    return z_seq, s_pred_seq


# ── Evaluation + plots ─────────────────────────────────────────────────────────
def evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device):
    set_plt_params(style='fast', file_format='png')
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

    acc_per_dim = [
        cross_val_score(LogisticRegression(max_iter=500), all_z[:, d:d+1],
                        all_ctype.astype(int), cv=5, scoring="accuracy").mean()
        for d in range(LATENT_DIM)
    ]
    C_onehot = np.eye(len(CONTEXTS))[all_ctype.astype(int)]
    r2  = cross_val_score(LinearRegression(), all_z, C_onehot,
                          cv=5, scoring="r2").mean()
    acc = cross_val_score(LogisticRegression(max_iter=500), all_z,
                          all_ctype.astype(int), cv=5, scoring="accuracy").mean()

    print(f"\nMax per-neuron decode acc : {max(acc_per_dim):.4f}  (chance={1/len(CONTEXTS):.2f})")
    print(f"Linear probe R² (onehot)  : {r2:.4f}")
    print(f"Logistic decode accuracy  : {acc:.4f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    
    colors = {0: "#1c96c5", 1: "#20a7db", 2: "#62c1e5", 3: "#a0d9ef",
            4: "#274e13", 5: "#38761d", 6: "#6aa84f", 7: "#93c47d",
            8: "#800080", 9: "#be29ec", 10: "#d896ff", 11: "#efbbff"}


    tag     = "EC→CB | velocity dynamics | 12 contexts (no scale)"

    # Fig 1: training curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_mse"], label="train MSE", color="steelblue")
    ax.plot(history["val_mse"],   label="val MSE",   color="tomato")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.set_title(f"Training curve — {tag}")
    ax.legend(); plt.tight_layout()
    make_axis_nice(ax)
    plt.savefig(f"{FIG_DIR}/training", bbox_inches="tight")
    plt.close()

    # Fig 2: PCA coloured by context
    n_plot = min(2000, len(all_z))
    idx    = np.random.choice(len(all_z), n_plot, replace=False)
    z_sub  = all_z[idx]; ctx_sub = all_ctype[idx]
    pca    = PCA(n_components=2)
    z_2d   = pca.fit_transform(z_sub)

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(len(CONTEXTS)):
        mask = ctx_sub == c
        if mask.sum() > 0:
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=colors[c], label=C_LABELS[c], alpha=0.4, s=10)
    ax.set_title(f"PCA of z_t — coloured by context\n{tag}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(); plt.tight_layout()
    make_axis_nice(ax)
    plt.savefig(f"{FIG_DIR}/pca", bbox_inches="tight")
    plt.close()

    # Fig 3: per-dim decode accuracy
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(LATENT_DIM), acc_per_dim, color="steelblue")
    ax.axhline(1/len(CONTEXTS), color="tomato", linestyle="--",
               label=f"chance ({1/len(CONTEXTS):.2f})")
    ax.set_xlabel("z dimension"); ax.set_ylabel("Decode accuracy")
    ax.set_title(f"Per-dim context decodability\n{tag}")
    make_axis_nice(ax)
    ax.set_xticks(range(LATENT_DIM)); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/perdim", bbox_inches="tight")
    plt.close()

    # Fig 4: decodability over time
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
    ax.axhline(1/len(CONTEXTS), color="gray",   ls="--", label="chance")
    ax.axvline(50,              color="tomato",  ls="--", label="avg switch (t≈50)")
    ax.set_xlabel("Timestep t"); ax.set_ylabel("Context decode accuracy")
    ax.set_title(f"Context decodability over time\n{tag}")
    make_axis_nice(ax)
    ax.legend(); ax.set_ylim(0, 1); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/decodetime", bbox_inches="tight")
    plt.close()

    # Fig 5: confusion matrix — row-normalized + symmetrized (C + Cᵀ)/2
    # Raw C[i,j] = P(pred=j | true=i) is asymmetric (decoder bias + cluster
    # geometry). Symmetrizing collapses the two directional confusabilities
    # into a single C[i,j] = "how often i and j get mixed up either way",
    # which is what's needed if this matrix is to be read as a similarity /
    # turned into a distance downstream.
    clf = LogisticRegression(max_iter=500)
    clf.fit(all_z, all_ctype.astype(int))
    cm        = confusion_matrix(all_ctype.astype(int), clf.predict(all_z)).astype(float)
    row_sums  = cm.sum(axis=1, keepdims=True)
    cm_norm   = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    sym_conf  = 0.5 * (cm_norm + cm_norm.T)
    labels6   = [C_LABELS[i] for i in range(len(CONTEXTS))]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sym_conf, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(CONTEXTS))); ax.set_yticks(range(len(CONTEXTS)))
    ax.set_xticklabels(labels6, rotation=30, ha="right")
    ax.set_yticklabels(labels6)
    ax.set_xlabel("context"); ax.set_ylabel("context")
    for i in range(len(CONTEXTS)):
        for j in range(len(CONTEXTS)):
            ax.text(j, i, f"{sym_conf[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if sym_conf[i, j] < 0.5 else "black",
                    fontsize=8)
    ax.set_title(f"Symmetric confusion (C + Cᵀ)/2 — row-normalized\n{tag}")
    make_axis_nice(ax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion", bbox_inches="tight")
    plt.close()

    # ── Fig 5b: Univariate vs joint decoder importance ────────────────────────
    decoder_importance = np.linalg.norm(clf.coef_, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 3), sharey=False)
    axes[0].bar(range(LATENT_DIM), acc_per_dim, color="steelblue")
    axes[0].axhline(1/len(CONTEXTS), color="tomato", ls="--", label="chance")
    axes[0].set_xlabel("EC neuron"); axes[0].set_ylabel("Decode accuracy")
    make_axis_nice(axes[0])
    axes[0].set_title("Univariate context decodability"); axes[0].legend()
    axes[1].bar(range(LATENT_DIM), decoder_importance, color="seagreen")
    axes[1].set_xlabel("EC neuron"); axes[1].set_ylabel("||coef|| across context classes")
    axes[1].set_title("Joint decoder weight norm")
    fig.suptitle(f"Univariate vs joint context encoding\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/decoder_importance", bbox_inches="tight")
    plt.close()

    # ── Fig 6: SVD of context decoder ─────────────────────────────────────────
    W = clf.coef_                                    # (n_contexts, LATENT_DIM)
    U, S_sv, Vt = np.linalg.svd(W, full_matrices=False)
    n_axes = min(8, len(S_sv))

    print(f"\nContext subspace singular values: {S_sv.round(3)}")
    print(f"Variance explained by top axis  : {S_sv[0]**2/(S_sv**2).sum()*100:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(1, n_axes+1), S_sv[:n_axes], color="steelblue")
    ax.set_xlabel("Context axis"); ax.set_ylabel("Singular value")
    ax.set_title(f"Dimensionality of context subspace\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svd_singular_values", bbox_inches="tight")
    plt.close()

    # ── Fig 6b: U heatmap — do axes align with rotation vs curl? ──────────────
    print("\nU (context class loadings on SVD axes):")
    header = "  ".join([f"axis{i+1:>5}" for i in range(n_axes)])
    print(f"         {header}")
    for i, c in enumerate([C_LABELS[j] for j in range(len(CONTEXTS))]):
        vals = "  ".join([f"{U[i,k]:+.3f}" for k in range(n_axes)])
        print(f"  {c:14s}  {vals}")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(U, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_axes)); ax.set_xticklabels([f"axis{i+1}" for i in range(n_axes)])
    ax.set_yticks(range(len(CONTEXTS)))
    ax.set_yticklabels([C_LABELS[i] for i in range(len(CONTEXTS))])
    plt.colorbar(im, ax=ax)
    ax.set_title(f"U — context loadings on SVD axes\n(clean = rotation axis separate from curl axis)\n{tag}")
    make_axis_nice(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svd_U_heatmap", bbox_inches="tight")
    plt.close()

    # ── Fig 6c: Dot product of context axes? ──────────────

    fig, ax = plt.subplots(figsize=(3, 3))
    rank = 6
    W_tilde = U[:, :rank] @ np.diag(S_sv[:rank]) @ Vt[:rank]
    W_normed = W_tilde / (np.linalg.norm(W_tilde, axis=1, keepdims=True) + 1e-8)
    g = ax.imshow(W_normed @ W_normed.T, cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(g, ax=ax, pad=0.05, fraction=0.05,
                 label='Dot product')

    ax.set_yticks(range(len(CONTEXTS)))
    ax.set_yticklabels([C_LABELS[i] for i in range(len(CONTEXTS))])
    ax.set_xticks(range(len(CONTEXTS)))
    ax.set_xticklabels([C_LABELS[i] for i in range(len(CONTEXTS))], rotation=90)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/low_rank_classifier_dotproduct", bbox_inches="tight")
    plt.close()

    # ── Fig 6d: Plot data on top 3 feature dimensions? ──────────────
    fig = plt.figure(figsize=(10, 8))
    # make 3D scatter plot
    ax = fig.add_subplot(111, projection='3d')
    unique_contexts = np.unique(all_ctype)
    plot_percent = .3
    for i, c in enumerate(unique_contexts):
        mask = all_ctype == c
        zs = all_z[mask]
        points = np.random.choice(len(zs), int(len(zs) * plot_percent), replace=False)
        projections = zs @ Vt[:3].T
        ax.scatter(projections[points, 0], projections[points, 1], projections[points, 2],
                    color=COLORS[i], label=C_LABELS[i], alpha=0.5)
        
    ax.grid(False)
    ax.legend()

    make_3d_gif(elev_range=(0, 90), azim_range=(0, 360), fps=10, 
                ax=ax, fig=fig, save_path=f"{FIG_DIR}/z_3d_scatter.gif")

    plt.close()

    # ── Fig 7: Top context axes — which EC neurons contribute ─────────────────
    fig, axes = plt.subplots(1, n_axes, figsize=(4*n_axes, 3))
    if n_axes == 1: axes = [axes]
    for i, ax in enumerate(axes):
        ax.bar(range(LATENT_DIM), Vt[i], color="steelblue")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"Context axis {i+1}\n({S_sv[i]**2/(S_sv**2).sum()*100:.0f}% var)")
        ax.set_xlabel("EC neuron")
    fig.suptitle(f"Which EC neurons encode each context axis\n{tag}", y=1.02)
    make_axis_nice(fig)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svd_context_axes", bbox_inches="tight")
    plt.close()

    # ── Fig 8: CB weight alignment with context subspace ──────────────────────
    cb_W_z = cb.net[0].weight.detach().cpu().numpy()[:, 4:]   # (CB_HIDDEN, LATENT_DIM)
    alignment        = cb_W_z @ Vt[:n_axes].T                  # (CB_HIDDEN, n_axes)
    cb_w_norm        = np.linalg.norm(cb_W_z, axis=1, keepdims=True) + 1e-8
    alignment_normed = np.linalg.norm(alignment, axis=1) / cb_w_norm[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(alignment_normed, bins=30, color="steelblue", edgecolor="white")
    axes[0].axvline(alignment_normed.mean(), color="tomato", ls="--",
                    label=f"mean={alignment_normed.mean():.2f}")
    axes[0].set_xlabel("Fraction of z_t weight in context subspace")
    axes[0].set_ylabel("CB neurons"); axes[0].set_title("CB alignment with EC context subspace")
    axes[0].legend()
    axis_alignment = np.linalg.norm(alignment, axis=0)[:n_axes]
    axes[1].bar(range(1, n_axes+1), axis_alignment, color="steelblue")
    axes[1].set_xlabel("Context axis"); axes[1].set_ylabel("Total CB weight (L2)")
    axes[1].set_title("CB weight on each EC context axis")
    make_axis_nice(fig)
    fig.suptitle(f"Does CB read EC's context subspace?\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cb_context_alignment", bbox_inches="tight")
    plt.close()

    # ── Fig 9: Per-neuron ablation ─────────────────────────────────────────────
    N_val = len(va_trans)
    all_z_seq     = all_z.reshape(N_val, SEQ_LEN, LATENT_DIM)
    all_ctype_seq = all_ctype.reshape(N_val, SEQ_LEN)

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
                total   += F.mse_loss(cb(s_curr, a_curr, z_flat), s_target, reduction='sum').item()
                count   += s_target.shape[0]
        return total / count

    mse_baseline = cb_mse_with_z(all_z_seq)
    print(f"\nAblation baseline MSE: {mse_baseline:.4f}")

    delta_mse = np.zeros(LATENT_DIM)
    for d in range(LATENT_DIM):
        z_abl = all_z_seq.copy(); z_abl[:, :, d] = 0.0
        delta_mse[d] = cb_mse_with_z(z_abl) - mse_baseline
        print(f"  ablate z{d:2d}  Δmse={delta_mse[d]:+.4f}  (acc={acc_per_dim[d]:.2f})")

    sorted_dims = np.argsort(acc_per_dim)[::-1]
    prog_mse = [mse_baseline]
    for k in range(1, LATENT_DIM+1):
        z_abl = all_z_seq.copy(); z_abl[:, :, sorted_dims[:k]] = 0.0
        prog_mse.append(cb_mse_with_z(z_abl))

    sorted_idx = np.argsort(acc_per_dim)[::-1]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(range(LATENT_DIM), delta_mse[sorted_idx], color="steelblue")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(range(LATENT_DIM))
    ax.set_xticklabels([f"z{sorted_idx[i]}\n{acc_per_dim[sorted_idx[i]]:.2f}"
                        for i in range(LATENT_DIM)], fontsize=7)
    ax.set_xlabel("z neuron (sorted by decode acc)"); ax.set_ylabel("ΔMSE")
    ax.set_title(f"Per-neuron ablation effect on CB MSE\n{tag}")
    make_axis_nice(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ablation_per_neuron", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(LATENT_DIM+1), prog_mse, color="steelblue", marker="o")
    ax.axhline(mse_baseline, color="gray", ls="--", label=f"baseline={mse_baseline:.4f}")
    ax.set_xlabel("Neurons ablated (highest decode acc first)"); ax.set_ylabel("CB MSE")
    ax.set_title(f"Progressive ablation\n{tag}"); ax.legend()
    plt.tight_layout()
    make_axis_nice(ax)
    plt.savefig(f"{FIG_DIR}/ablation_progressive", bbox_inches="tight")
    plt.close()

    # ── Fig 10: Context subspace ablation ─────────────────────────────────────
    ctx_prog_mse = [mse_baseline]
    for k in range(1, n_axes+1):
        Vt_k  = Vt[:k]; Pc_k = Vt_k.T @ Vt_k
        z_abl = all_z_seq - all_z_seq @ Pc_k
        ctx_prog_mse.append(cb_mse_with_z(z_abl))
        print(f"  ctx subspace ablation top-{k}: MSE={ctx_prog_mse[-1]:.4f}  "
              f"ΔMSE={ctx_prog_mse[-1]-mse_baseline:+.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(n_axes+1), ctx_prog_mse, color="seagreen", marker="o")
    ax.axhline(mse_baseline, color="gray", ls="--", label=f"baseline={mse_baseline:.4f}")
    ax.set_xlabel("Context axes removed"); ax.set_ylabel("CB MSE")
    ax.set_title(f"Context subspace ablation\n{tag}"); ax.legend()
    make_axis_nice(ax)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ctx_subspace_ablation", bbox_inches="tight")
    plt.close()


# ── Training ───────────────────────────────────────────────────────────────────
def train(seed=0, skip_eval=True):
    # Seed model init + any stochastic loaders. Data seeds are kept fixed below
    # so every seeded run sees the same train/val sequences — only the model
    # initialization and SGD trajectory differ across seeds.
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckpt_path = seeded_ckpt_path(seed)

    exists_already = os.path.exists(ckpt_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  | seed: {seed}  | ckpt: {ckpt_path}")
    print(f"Sequences: train={N_TRAIN} | val={N_VAL} | seq_len={SEQ_LEN} | epochs={EPOCHS}")
    print(f"Dynamics: velocity-based | F = β·R(θ)·a + γ·curl·V + g·b")
    print(f"Contexts: {[c[7] for c in CONTEXTS]}")
    print(f"EC input: [s_{{t-1}}, a_{{t-1}}, s_t, e_t] — velocity inferred implicitly\n")

    tr_trans, _      = generate_sequences(N_TRAIN, seed=0)
    va_trans, va_ctx = generate_sequences(N_VAL,   seed=99)

    if exists_already:
        # load previously saved checkpoint
        checkpoint = torch.load(ckpt_path)
        ec = EntorhinalCortex().to(device)
        cb = CerebellumCB().to(device)
        ec.load_state_dict(checkpoint["ec"])
        cb.load_state_dict(checkpoint["cb"])
        history = checkpoint["history"]
        print("Loaded existing checkpoint")

    else:
        train_loader = DataLoader(TransitionDataset(tr_trans), batch_size=BATCH_SIZE, shuffle=True)

        ec        = EntorhinalCortex().to(device)
        cb        = CerebellumCB().to(device)
        optimizer = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)
        history   = {"train_mse": [], "val_mse": []}

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
                va_trans_dev = va_trans.to(device)
                s_next       = va_trans_dev[:, :, 4:6]
                _, s_pred    = run_sequence(ec, cb, va_trans_dev, device)
                val_mse      = F.mse_loss(s_pred, s_next).item()
            history["val_mse"].append(val_mse)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train MSE: {total_mse:.4f} | Val MSE: {val_mse:.4f}")

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "history": history,
                    "seed": seed}, ckpt_path)
        print(f"\nSaved {ckpt_path}")

    if skip_eval:
        print("(skipping evaluate_and_plot — --skip-eval set)")
    else:
        evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=0,
                        help="Model-init / training RNG seed. Checkpoint saved as "
                             "CKPT_PATH with `_seed{seed}` suffix.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluate_and_plot after training (useful when "
                             "running many seeds — figs would otherwise overwrite each other).")
    args = parser.parse_args()
    t0 = time.time()
    for seed in range(args.seeds):
        train(seed=seed, skip_eval=not seed==0)
    print(f"\nTotal training time: {time.time() - t0:.1f}s")

"""
ECtoCB_newcontexts_infoNCE.py

Same as ECtoCB_newcontexts_train.py but adds InfoNCE contrastive loss to EC training.

InfoNCE objective (temporal proximity — no context labels used):
    For each anchor timestep, sample:
        M_POS positives = timesteps within ±SMALL_DELTA (close → same context likely)
        K_NEG negatives = timesteps at least LARGE_DELTA away (far → different context likely)
    Each positive must outscore ALL negatives simultaneously.
    EC never sees context labels — must infer context from prediction errors alone.

Combined loss:
    L = L_MSE  +  LAMBDA_NCE * L_InfoNCE

Goal:
    z_ec clusters by context → thalamus router can separate them linearly.

All architecture, data generation, and evaluation code is unchanged.
New checkpoint: checkpoints/ectocb_8ctx_seq80_z32_infonce.pt
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN    = 80
N_TRAIN    = 2000
N_VAL      = 200
BATCH_SIZE = 128
EPOCHS     = 300
LR         = 1e-3
LATENT_DIM = 32
EC_HIDDEN  = 256
CB_HIDDEN  = 256
EC_INPUT   = 8
GRID_SIZE  = 4
DT         = 1.0

# InfoNCE hyperparameters
NCE_TAU     = 0.1    # temperature — lower = sharper contrastive signal
LAMBDA_NCE  = 0.1    # weight of InfoNCE loss relative to MSE
N_ANCHOR    = 4      # anchor timesteps sampled per sequence per step
SMALL_DELTA = 5      # positives: timesteps within ±SMALL_DELTA of anchor (close, same context likely)
LARGE_DELTA = 25     # negatives: timesteps at least LARGE_DELTA away (crosses switch boundary at t≈50)
K_NEG       = 8      # number of negative timesteps per anchor

CKPT_PATH  = "checkpoints/ectocb_8ctx_seq80_z32_infonce.pt"
FIG_DIR    = "../../figs/ECtoCB/ECtoCB_8ctx_seq80_z32_infonce"

# ── Contexts ───────────────────────────────────────────────────────────────────
CONTEXTS = [
    # (beta,  theta,      gamma,  g,    bx,   by,  c_id, label)
    (1.0,  +np.pi/4,  0.0,    0.0,  0.0,  0.0,  0,  "C0: rot+"),
    (1.0,  -np.pi/4,  0.0,    0.0,  0.0,  0.0,  1,  "C1: rot-"),
    (1.0,   0.0,     +0.4,    0.0,  0.0,  0.0,  2,  "C2: curl+"),
    (1.0,   0.0,     -0.4,    0.0,  0.0,  0.0,  3,  "C3: curl-"),
    (1.0,   0.0,      0.0,    0.4, +1.0,  0.0,  4,  "C4: drift+x"),
    (1.0,   0.0,      0.0,    0.4, -1.0,  0.0,  5,  "C5: drift-x"),
    (1.5,   0.0,      0.0,    0.0,  0.0,  0.0,  6,  "C6: scale_strong"),
    (-1.5,  0.0,      0.0,    0.0,  0.0,  0.0,  7,  "C7: scale_neg"),
]
C_LABELS = {c[6]: c[7] for c in CONTEXTS}


# ── Environment step ───────────────────────────────────────────────────────────
def env_step(pos, vel, action, beta, theta, gamma, g, bx, by):
    R    = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])
    curl = np.array([[0., -1.], [1.,  0.]])
    b    = np.array([bx, by], dtype=np.float32)
    F    = beta * (R @ action) + gamma * (curl @ vel) + g * b
    vel_new = vel + F
    pos_new = np.clip(pos + DT * vel_new, 0, GRID_SIZE)
    return pos_new.astype(np.float32), vel_new.astype(np.float32)


# ── Data generation ────────────────────────────────────────────────────────────
def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0):
    rng = np.random.default_rng(seed)
    all_trans, all_ctx = [], []

    for _ in range(n_seqs):
        pos    = rng.uniform(0, GRID_SIZE, size=2).astype(np.float32)
        vel    = np.zeros(2, dtype=np.float32)
        c_ids  = rng.choice(len(CONTEXTS), size=2, replace=False)
        ctx_a  = CONTEXTS[c_ids[0]]
        ctx_b  = CONTEXTS[c_ids[1]]
        switch = int(rng.integers(40, 61))

        seq_trans, seq_ctx = [], []
        for t in range(seq_len):
            beta, theta, gamma, g, bx, by, c_id, _ = (ctx_a if t < switch else ctx_b)
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            pos_next, vel_next = env_step(pos, vel, action, beta, theta, gamma, g, bx, by)
            seq_trans.append([pos[0], pos[1], action[0], action[1], pos_next[0], pos_next[1]])
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
    def __init__(self, transitions, ctx_labels):
        self.transitions = transitions
        self.ctx_labels  = ctx_labels
    def __len__(self): return len(self.transitions)
    def __getitem__(self, idx): return self.transitions[idx], self.ctx_labels[idx]


# ── Models (unchanged) ─────────────────────────────────────────────────────────
class EntorhinalCortex(nn.Module):
    def __init__(self, input_dim=EC_INPUT, hidden_dim=EC_HIDDEN, latent_dim=LATENT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell       = nn.GRUCell(input_dim, hidden_dim)
        self.readout    = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.Tanh())

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(self, x, h):
        h_new = self.cell(x, h)
        return self.readout(h_new), h_new


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


# ── Sequential forward pass (unchanged) ───────────────────────────────────────
def run_sequence(ec, cb, trans, device):
    B, T, _ = trans.shape
    zeros2  = torch.zeros(B, 2, device=device)
    h       = ec.init_hidden(B, device)
    z_t     = torch.zeros(B, LATENT_DIM, device=device)
    s_pred  = zeros2.clone()
    z_seq, s_pred_seq = [], []

    for t in range(T):
        s_t = trans[:, t, :2]
        a_t = trans[:, t, 2:4]
        if t == 0:
            s_tm1, a_tm1, e_t = zeros2, zeros2, zeros2
        else:
            s_tm1 = trans[:, t-1, :2]
            a_tm1 = trans[:, t-1, 2:4]
            e_t   = s_t - s_pred

        ec_input = torch.cat([s_tm1, a_tm1, s_t, e_t], dim=-1)
        z_t, h   = ec.step(ec_input, h)
        s_pred   = cb(s_t, a_t, z_t)
        z_seq.append(z_t)
        s_pred_seq.append(s_pred)

    return torch.stack(z_seq, dim=1), torch.stack(s_pred_seq, dim=1)


# ── InfoNCE loss ───────────────────────────────────────────────────────────────
def infonce_loss(z_seq, tau=NCE_TAU,
                 n_anchor=N_ANCHOR, small_delta=SMALL_DELTA,
                 large_delta=LARGE_DELTA, k_neg=K_NEG):
    """
    Temporal proximity InfoNCE — biologically motivated, no context labels needed.

    For each anchor timestep t_a:
        positives : M_POS timesteps within ±small_delta of t_a  (close → same context likely)
        negatives : k_neg timesteps at least large_delta away    (far   → different context likely)

    Loss per anchor = mean over positives of:
        -log( exp(sim(a, p_i)/τ) / (exp(sim(a, p_i)/τ) + Σ_k exp(sim(a, neg_k)/τ)) )

    Each positive must outscore ALL negatives — much stronger than binary triplet.

    z_seq : (B, T, D) — EC latent sequence, gradients flow back through this
    """
    B, T, D = z_seq.shape
    device  = z_seq.device

    # Valid anchor range: needs room for positives and negatives on both sides
    t_min = small_delta
    t_max = T - large_delta - k_neg
    if t_max <= t_min:
        return torch.tensor(0.0, device=device, requires_grad=True)

    total   = torch.tensor(0.0, device=device)
    n_valid = 0

    for _ in range(n_anchor):
        # Sample one anchor per sequence in the batch
        ta = torch.randint(t_min, t_max, (B,), device=device)   # (B,)

        # ── Positive timesteps: within ±small_delta, excluding anchor ─────────
        pos_offsets = [d for d in range(-small_delta, small_delta + 1) if d != 0]
        t_pos_list  = [torch.clamp(ta + d, 0, T-1) for d in pos_offsets]
        # z_pos: (B, M_POS, D)
        z_pos = torch.stack(
            [F.normalize(z_seq[torch.arange(B, device=device), tp], dim=-1)
             for tp in t_pos_list], dim=1
        )
        M_POS = len(pos_offsets)

        # ── Negative timesteps: at least large_delta away ─────────────────────
        neg_offsets = torch.randint(large_delta, T - t_max + 1,
                                    (B, k_neg), device=device)
        # Randomly assign sign (push negatives to either direction)
        signs = torch.randint(0, 2, (B, k_neg), device=device) * 2 - 1
        t_neg_raw = ta.unsqueeze(1) + signs * neg_offsets           # (B, k_neg)
        t_neg     = torch.clamp(t_neg_raw, 0, T-1)

        # z_neg: (B, k_neg, D)
        z_neg = torch.stack(
            [F.normalize(z_seq[torch.arange(B, device=device), t_neg[:, k]], dim=-1)
             for k in range(k_neg)], dim=1
        )

        # ── Anchor z ──────────────────────────────────────────────────────────
        z_a = F.normalize(z_seq[torch.arange(B, device=device), ta], dim=-1)  # (B, D)

        # ── InfoNCE: each positive vs all negatives ───────────────────────────
        # sim_pos: (B, M_POS)   sim_neg: (B, k_neg)
        sim_pos = torch.einsum('bd,bmd->bm', z_a, z_pos) / tau
        sim_neg = torch.einsum('bd,bkd->bk', z_a, z_neg) / tau

        # For each positive: loss = -log( exp(sim_pos_i) / (exp(sim_pos_i) + Σ exp(sim_neg_k)) )
        # logits: (B, M_POS, 1 + k_neg)  — positive is index 0 for each row
        sim_neg_exp = sim_neg.unsqueeze(1).expand(B, M_POS, k_neg)  # (B, M_POS, k_neg)
        logits = torch.cat([sim_pos.unsqueeze(-1), sim_neg_exp], dim=-1)  # (B, M_POS, 1+k_neg)
        labels = torch.zeros(B * M_POS, dtype=torch.long, device=device)

        loss_anchor = F.cross_entropy(logits.reshape(B * M_POS, -1), labels)
        total   = total + loss_anchor
        n_valid += 1

    return total / max(n_valid, 1)


# ── Evaluation + plots (unchanged except training curve) ──────────────────────
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

    acc_per_dim = [
        cross_val_score(LogisticRegression(max_iter=500), all_z[:, d:d+1],
                        all_ctype.astype(int), cv=5, scoring="accuracy").mean()
        for d in range(LATENT_DIM)
    ]
    C_onehot = np.eye(len(CONTEXTS))[all_ctype.astype(int)]
    r2  = cross_val_score(LinearRegression(), all_z, C_onehot, cv=5, scoring="r2").mean()
    acc = cross_val_score(LogisticRegression(max_iter=500), all_z,
                          all_ctype.astype(int), cv=5, scoring="accuracy").mean()

    print(f"\nMax per-neuron decode acc : {max(acc_per_dim):.4f}  (chance={1/len(CONTEXTS):.2f})")
    print(f"Linear probe R² (onehot)  : {r2:.4f}")
    print(f"Logistic decode accuracy  : {acc:.4f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    colors = {0: "steelblue", 1: "cornflowerblue",
              2: "seagreen",  3: "limegreen",
              4: "purple",    5: "violet",
              6: "tomato",    7: "orange"}
    tag = "EC→CB | InfoNCE + MSE | new contexts"

    # Fig 1: training curve — MSE + NCE loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_mse"], label="train MSE", color="steelblue")
    axes[0].plot(history["val_mse"],   label="val MSE",   color="tomato")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
    axes[0].set_title(f"MSE loss — {tag}")
    axes[0].legend(); axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(history["train_nce"], label="train InfoNCE", color="seagreen")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("InfoNCE loss")
    axes[1].set_title(f"InfoNCE loss — {tag}")
    axes[1].legend(); axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/training.png", dpi=150, bbox_inches="tight")
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
    plt.savefig(f"{FIG_DIR}/pca.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: per-dim decode accuracy
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(LATENT_DIM), acc_per_dim, color="steelblue")
    ax.axhline(1/len(CONTEXTS), color="tomato", linestyle="--",
               label=f"chance ({1/len(CONTEXTS):.2f})")
    ax.set_xlabel("z dimension"); ax.set_ylabel("Decode accuracy")
    ax.set_title(f"Per-dim context decodability\n{tag}")
    ax.set_xticks(range(LATENT_DIM)); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/perdim.png", dpi=150, bbox_inches="tight")
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
    ax.axhline(1/len(CONTEXTS), color="gray",  ls="--", label="chance")
    ax.axvline(50,              color="tomato", ls="--", label="avg switch (t≈50)")
    ax.set_xlabel("Timestep t"); ax.set_ylabel("Context decode accuracy")
    ax.set_title(f"Context decodability over time\n{tag}")
    ax.legend(); ax.set_ylim(0, 1); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/decodetime.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 5: confusion matrix
    clf = LogisticRegression(max_iter=500)
    clf.fit(all_z, all_ctype.astype(int))
    cm  = confusion_matrix(all_ctype.astype(int), clf.predict(all_z))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=[C_LABELS[i] for i in range(len(CONTEXTS))])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Context confusion matrix\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 5b: univariate vs joint decoder importance
    decoder_importance = np.linalg.norm(clf.coef_, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 3), sharey=False)
    axes[0].bar(range(LATENT_DIM), acc_per_dim, color="steelblue")
    axes[0].axhline(1/len(CONTEXTS), color="tomato", ls="--", label="chance")
    axes[0].set_xlabel("EC neuron"); axes[0].set_ylabel("Decode accuracy")
    axes[0].set_title("Univariate context decodability"); axes[0].legend()
    axes[1].bar(range(LATENT_DIM), decoder_importance, color="seagreen")
    axes[1].set_xlabel("EC neuron"); axes[1].set_ylabel("||coef|| across context classes")
    axes[1].set_title("Joint decoder weight norm")
    fig.suptitle(f"Univariate vs joint context encoding\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/decoder_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 6: SVD of context decoder
    W = clf.coef_
    U, S_sv, Vt = np.linalg.svd(W, full_matrices=False)
    n_axes = min(4, len(S_sv))

    print(f"\nContext subspace singular values: {S_sv.round(3)}")
    print(f"Variance explained by top axis  : {S_sv[0]**2/(S_sv**2).sum()*100:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(1, n_axes+1), S_sv[:n_axes], color="steelblue")
    ax.set_xlabel("Context axis"); ax.set_ylabel("Singular value")
    ax.set_title(f"Dimensionality of context subspace\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svd_singular_values.png", dpi=150, bbox_inches="tight")
    plt.close()

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
    ax.set_title(f"U — context loadings on SVD axes\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svd_U_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, n_axes, figsize=(4*n_axes, 3))
    if n_axes == 1: axes = [axes]
    for i, ax in enumerate(axes):
        ax.bar(range(LATENT_DIM), Vt[i], color="steelblue")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"Context axis {i+1}\n({S_sv[i]**2/(S_sv**2).sum()*100:.0f}% var)")
        ax.set_xlabel("EC neuron")
    fig.suptitle(f"Which EC neurons encode each context axis\n{tag}", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/svd_context_axes.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 8: CB alignment with context subspace
    cb_W_z = cb.net[0].weight.detach().cpu().numpy()[:, 4:]
    alignment        = cb_W_z @ Vt[:n_axes].T
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
    fig.suptitle(f"Does CB read EC's context subspace?\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cb_context_alignment.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 9: per-neuron ablation
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
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ablation_per_neuron.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(LATENT_DIM+1), prog_mse, color="steelblue", marker="o")
    ax.axhline(mse_baseline, color="gray", ls="--", label=f"baseline={mse_baseline:.4f}")
    ax.set_xlabel("Neurons ablated (highest decode acc first)"); ax.set_ylabel("CB MSE")
    ax.set_title(f"Progressive ablation\n{tag}"); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ablation_progressive.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 10: context subspace ablation
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
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/ctx_subspace_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Parameter encoding
    ctx_to_theta = {0: np.pi/4, 1: -np.pi/4, 2: 0.0,  3: 0.0,  4: 0.0,  5: 0.0,  6: 0.0,  7: 0.0}
    ctx_to_gamma = {0: 0.0,     1: 0.0,       2: 0.4,  3: -0.4, 4: 0.0,  5: 0.0,  6: 0.0,  7: 0.0}
    ctx_to_g     = {0: 0.0,     1: 0.0,       2: 0.0,  3: 0.0,  4: 0.4,  5: 0.4,  6: 0.0,  7: 0.0}
    ctx_to_beta  = {0: 1.0,     1: 1.0,       2: 1.0,  3: 1.0,  4: 1.0,  5: 1.0,  6: 1.5,  7: 0.5}
    ctx_to_type  = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}

    all_theta = np.array([ctx_to_theta[int(c)] for c in all_ctype])
    all_gamma = np.array([ctx_to_gamma[int(c)] for c in all_ctype])
    all_g     = np.array([ctx_to_g[int(c)]     for c in all_ctype])
    all_beta  = np.array([ctx_to_beta[int(c)]  for c in all_ctype])
    all_type  = np.array([ctx_to_type[int(c)]  for c in all_ctype])

    from sklearn.linear_model import Ridge
    r2_theta = cross_val_score(Ridge(), all_z, all_theta, cv=5, scoring="r2").mean()
    r2_gamma = cross_val_score(Ridge(), all_z, all_gamma, cv=5, scoring="r2").mean()
    r2_g     = cross_val_score(Ridge(), all_z, all_g,     cv=5, scoring="r2").mean()
    r2_beta  = cross_val_score(Ridge(), all_z, all_beta,  cv=5, scoring="r2").mean()
    acc_type = cross_val_score(LogisticRegression(max_iter=500), all_z, all_type,
                               cv=5, scoring="accuracy").mean()

    print(f"\n--- Parameter encoding in z_ec ---")
    print(f"  Decode θ (rotation angle)       R²  : {r2_theta:.3f}")
    print(f"  Decode γ (curl strength)        R²  : {r2_gamma:.3f}")
    print(f"  Decode g (drift strength)       R²  : {r2_g:.3f}")
    print(f"  Decode β (action scaling)       R²  : {r2_beta:.3f}")
    print(f"  Decode FF type (rot/curl/drift/scale): {acc_type:.3f}  (chance=0.50)")

    r2_theta_dim = [cross_val_score(Ridge(), all_z[:, d:d+1], all_theta,
                                    cv=5, scoring="r2").mean() for d in range(LATENT_DIM)]
    r2_gamma_dim = [cross_val_score(Ridge(), all_z[:, d:d+1], all_gamma,
                                    cv=5, scoring="r2").mean() for d in range(LATENT_DIM)]
    r2_g_dim     = [cross_val_score(Ridge(), all_z[:, d:d+1], all_g,
                                    cv=5, scoring="r2").mean() for d in range(LATENT_DIM)]
    r2_beta_dim  = [cross_val_score(Ridge(), all_z[:, d:d+1], all_beta,
                                    cv=5, scoring="r2").mean() for d in range(LATENT_DIM)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    for ax, r2_vals, label, color in zip(
        axes.flat,
        [r2_theta_dim, r2_gamma_dim, r2_g_dim, r2_beta_dim],
        ["θ (rotation angle)", "γ (curl strength)", "g (drift strength)", "β (action scaling)"],
        ["steelblue", "tomato", "seagreen", "orange"],
    ):
        ax.bar(range(LATENT_DIM), r2_vals, color=color)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("EC neuron"); ax.set_ylabel("R²")
        ax.set_title(f"Per-neuron R² for {label}\n{tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/param_encoding_all.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Context pair separability
    print("\n--- Context pair separability ---")
    for (ca, cb_id, label) in [
        (0, 1, "C0 vs C1 (rot+ vs rot-)"),
        (2, 3, "C2 vs C3 (curl+ vs curl-)"),
        (4, 5, "C4 vs C5 (drift+x vs drift-x)"),
        (6, 7, "C6 vs C7 (scale_strong vs scale_weak)"),
        (0, 2, "C0 vs C2 (rotation vs curl)"),
        (0, 4, "C0 vs C4 (rotation vs drift)"),
        (2, 4, "C2 vs C4 (curl vs drift)"),
    ]:
        mask     = np.isin(all_ctype.astype(int), [ca, cb_id])
        z_pair   = all_z[mask]
        c_pair   = all_ctype[mask].astype(int)
        acc_pair = cross_val_score(LogisticRegression(max_iter=500), z_pair, c_pair,
                                   cv=5, scoring="accuracy").mean()
        print(f"  {label}: decode acc = {acc_pair:.3f}")

    print(f"\nAll figures saved to {FIG_DIR}/")


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Sequences: train={N_TRAIN} | val={N_VAL} | seq_len={SEQ_LEN} | epochs={EPOCHS}")
    print(f"Loss: MSE + {LAMBDA_NCE} * InfoNCE (tau={NCE_TAU}, {N_ANCHOR} anchors, "
          f"{SMALL_DELTA*2} pos, {K_NEG} neg per anchor | temporal proximity, no labels)")
    print(f"Contexts: {[c[7] for c in CONTEXTS]}\n")

    tr_trans, tr_ctx = generate_sequences(N_TRAIN, seed=0)
    va_trans, va_ctx = generate_sequences(N_VAL,   seed=99)

    train_loader = DataLoader(
        TransitionDataset(tr_trans, tr_ctx), batch_size=BATCH_SIZE, shuffle=True
    )

    ec        = EntorhinalCortex().to(device)
    cb        = CerebellumCB().to(device)
    optimizer = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)
    history   = {"train_mse": [], "train_nce": [], "val_mse": []}

    for epoch in range(1, EPOCHS + 1):
        ec.train(); cb.train()
        total_mse = 0.0
        total_nce = 0.0

        for trans, _ in train_loader:
            trans  = trans.to(device)
            s_next = trans[:, :, 4:6]

            z_seq, s_pred = run_sequence(ec, cb, trans, device)

            loss_mse = F.mse_loss(s_pred, s_next)
            loss_nce = infonce_loss(z_seq)
            loss     = loss_mse + LAMBDA_NCE * loss_nce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_mse += loss_mse.item() * len(trans)
            total_nce += loss_nce.item() * len(trans)

        total_mse /= N_TRAIN
        total_nce /= N_TRAIN
        history["train_mse"].append(total_mse)
        history["train_nce"].append(total_nce)

        ec.eval(); cb.eval()
        with torch.no_grad():
            va_trans_dev = va_trans.to(device)
            s_next       = va_trans_dev[:, :, 4:6]
            _, s_pred    = run_sequence(ec, cb, va_trans_dev, device)
            val_mse      = F.mse_loss(s_pred, s_next).item()
        history["val_mse"].append(val_mse)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train MSE: {total_mse:.4f} | "
                  f"Train NCE: {total_nce:.4f} | Val MSE: {val_mse:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "history": history}, CKPT_PATH)
    print(f"\nSaved {CKPT_PATH}")

    evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device)


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"\nTotal training time: {time.time() - t0:.1f}s")

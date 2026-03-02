"""
ECtoCB_MSEandTIMEclustering.py

Combined loss: shifted MSE prediction + temporal clustering (InfoNCE).

Loss = MSE_shifted + lambda_nce * InfoNCE_posnegtime

  MSE_shifted  : z_t (from transitions 0..t) helps CB predict s_{t+2}
                 from [s_{t+1}, a_{t+1}]. No shortcut — z_t never saw s_{t+2}.

  InfoNCE_posnegtime : within-sequence triplet loss
                 positive = z_{t + small_delta}  (same context block)
                 negative = z from the OTHER context block (same sequence)
                 -- pushes same-context z_t to cluster together

Data: 4 fixed contexts, each sequence switches between 2 at a random point.
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


# ── Hyperparameters ──────────────────────────────────────────────────────────

SEQ_LEN     = 40
SMALL_DELTA = 3
N_TRAIN     = 2000
N_VAL       = 200
BATCH_SIZE  = 128
EPOCHS      = 100
LR          = 1e-3
LATENT_DIM  = 16
CB_HIDDEN   = 256
TEMPERATURE = 0.1
LAMBDA_NCE  = 0.5    # weight of InfoNCE relative to MSE


# ── Cerebellar MLP 
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


# ── Data generation 
def generate_sequences(n_seqs, seq_len=SEQ_LEN, seed=0):
    rng  = np.random.default_rng(seed)
    acts = [(-1., 0.), (1., 0.), (0., -1.), (0., 1.)]

    all_trans, all_ctx, all_switch = [], [], []

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
        all_switch.append(switch)

    return (
        torch.tensor(np.array(all_trans),  dtype=torch.float32),  # (N, T, 6)
        torch.tensor(np.array(all_ctx),    dtype=torch.float32),  # (N, T, 3)
        torch.tensor(np.array(all_switch), dtype=torch.long),     # (N,)
    )


class TransitionDataset(Dataset):
    def __init__(self, transitions, switches):
        self.transitions = transitions
        self.switches    = switches
    def __len__(self): return len(self.transitions)
    def __getitem__(self, idx): return self.transitions[idx], self.switches[idx]


# ── InfoNCE: within-sequence, pos=small delta same block, neg=other block ─────

def infonce_posnegtime(z, switch_points, small_delta=SMALL_DELTA, temperature=TEMPERATURE):
    """
    For each sequence:
      anchor : random timestep in block 1 or block 2
      positive: anchor + small_delta (same block)
      negative: random timestep from the OTHER block
    """
    B, T, D = z.shape

    t_anchor = torch.zeros(B, dtype=torch.long, device=z.device)
    t_pos    = torch.zeros(B, dtype=torch.long, device=z.device)
    t_neg    = torch.zeros(B, dtype=torch.long, device=z.device)

    for i in range(B):
        sw     = switch_points[i].item()
        b1_max = sw - small_delta - 1
        b2_max = T  - small_delta - 1

        if b1_max > 0 and torch.randint(0, 2, (1,)).item() == 0:
            t_anchor[i] = torch.randint(0, b1_max, (1,)).item()
            t_pos[i]    = t_anchor[i] + torch.randint(1, small_delta + 1, (1,)).item()
            t_neg[i]    = torch.randint(sw, T, (1,)).item()
        else:
            t_anchor[i] = torch.randint(sw, b2_max, (1,)).item()
            t_pos[i]    = t_anchor[i] + torch.randint(1, small_delta + 1, (1,)).item()
            t_neg[i]    = torch.randint(0, sw, (1,)).item()

    anchors   = F.normalize(z[torch.arange(B), t_anchor], dim=-1)
    positives = F.normalize(z[torch.arange(B), t_pos],    dim=-1)
    negatives = F.normalize(z[torch.arange(B), t_neg],    dim=-1)

    beta  = (anchors * positives).sum(dim=-1, keepdim=True) / temperature
    alpha = (anchors * negatives).sum(dim=-1, keepdim=True) / temperature

    logits = torch.cat([beta, alpha], dim=-1)
    labels = torch.zeros(B, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


# ── Evaluation + plots
def evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device):
    ec.eval(); cb.eval()
    all_z, all_ctype = [], []

    with torch.no_grad():
        for i in range(0, len(va_trans), BATCH_SIZE):
            trans = va_trans[i:i+BATCH_SIZE].to(device)
            z, _  = ec(trans)
            ctx   = va_ctx[i:i+BATCH_SIZE]
            B, T, D = z.shape
            all_z.append(z.cpu().reshape(B * T, D))
            all_ctype.append(ctx[:, :, 2].reshape(B * T))

    all_z     = torch.cat(all_z).numpy()
    all_ctype = torch.cat(all_ctype).numpy()

    r_per_dim = [abs(stats.pearsonr(all_z[:, d], all_ctype)[0]) for d in range(LATENT_DIM)]
    r_max     = max(r_per_dim)
    r2        = cross_val_score(LinearRegression(), all_z, all_ctype, cv=5, scoring="r2").mean()
    acc       = cross_val_score(LogisticRegression(max_iter=500), all_z,
                                all_ctype.astype(int), cv=5, scoring="accuracy").mean()
    print(f"\nMax |r| z_t vs c_type  : {r_max:.4f}")
    print(f"Linear probe R²         : {r2:.4f}")
    print(f"Logistic decode accuracy: {acc:.4f}  (chance=0.25)")

    os.makedirs("figs", exist_ok=True)
    colors   = {0: "steelblue", 1: "tomato", 2: "seagreen", 3: "orange"}
    c_labels = {0: "C0 (+beta)", 1: "C1 (-beta)", 2: "C2 (+gamma)", 3: "C3 (-gamma)"}
    tag      = f"EC→CB MSE+NCE (λ={LAMBDA_NCE})"

    # ── Fig 1: Training curves ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_mse"], label="train", color="steelblue")
    ax1.plot(history["val_mse"],   label="val",   color="tomato")
    ax1.set_title("MSE loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(history["train_nce"], label="train", color="steelblue")
    ax2.plot(history["val_nce"],   label="val",   color="tomato")
    ax2.set_title("InfoNCE loss"); ax2.set_xlabel("Epoch"); ax2.legend()
    fig.suptitle(tag)
    plt.tight_layout()
    plt.savefig("figs/mse_nce_training.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figs/mse_nce_training.png")

    # ── Fig 2: PCA ───────────────────────────────────────────────────────────
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
    ax.set_title(f"PCA of z_t — coloured by context\n{tag}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figs/mse_nce_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figs/mse_nce_pca.png")

    # ── Fig 3: Per-dim |r| ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(LATENT_DIM), r_per_dim, color="steelblue")
    ax.axhline(0.1, color="tomato", linestyle="--", label="|r|=0.1 reference")
    ax.set_xlabel("z dimension"); ax.set_ylabel("|Pearson r| vs c_type")
    ax.set_title(f"Per-dim correlation with context\n{tag}")
    ax.set_xticks(range(LATENT_DIM)); ax.legend()
    plt.tight_layout()
    plt.savefig("figs/mse_nce_perdim.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figs/mse_nce_perdim.png")

    # ── Fig 4: Decoding accuracy over time ───────────────────────────────────
    N_val         = len(va_trans)
    all_z_seq     = all_z.reshape(N_val, SEQ_LEN, LATENT_DIM)
    all_ctype_seq = all_ctype.reshape(N_val, SEQ_LEN)

    acc_over_t = []
    for t in range(SEQ_LEN):
        z_t = all_z_seq[:, t, :]
        c_t = all_ctype_seq[:, t].astype(int)
        acc_over_t.append(cross_val_score(LogisticRegression(max_iter=500), z_t, c_t,
                                          cv=5, scoring="accuracy").mean())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(acc_over_t, color="steelblue", label="decode accuracy")
    ax.axhline(0.25, color="gray",  linestyle="--", label="chance (0.25)")
    ax.axvline(20,   color="tomato", linestyle="--", label="avg switch (t≈20)")
    ax.set_xlabel("Timestep t"); ax.set_ylabel("Context decode accuracy")
    ax.set_title(f"Context decodability over time\n{tag}")
    ax.legend(); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figs/mse_nce_decodetime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figs/mse_nce_decodetime.png")

    # ── Fig 5: Confusion matrix ───────────────────────────────────────────────
    clf = LogisticRegression(max_iter=500)
    clf.fit(all_z, all_ctype.astype(int))
    cm  = confusion_matrix(all_ctype.astype(int), clf.predict(all_z))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["C0", "C1", "C2", "C3"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Context confusion matrix\n{tag}")
    plt.tight_layout()
    plt.savefig("figs/mse_nce_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved figs/mse_nce_confusion.png")


# ── Training ───────────────────────────────────────────────────────────────────

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training sequences: {N_TRAIN} | Val: {N_VAL} | Epochs: {EPOCHS}")
    print(f"LAMBDA_NCE: {LAMBDA_NCE} | TEMPERATURE: {TEMPERATURE} | SMALL_DELTA: {SMALL_DELTA}\n")

    tr_trans, _,      tr_switch = generate_sequences(N_TRAIN, seed=0)
    va_trans, va_ctx, va_switch = generate_sequences(N_VAL,   seed=99)

    train_loader = DataLoader(TransitionDataset(tr_trans, tr_switch),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TransitionDataset(va_trans, va_switch),
                              batch_size=BATCH_SIZE)

    ec        = EntorhinalCortex(input_dim=6, hidden_dim=256, latent_dim=LATENT_DIM).to(device)
    cb        = CerebellumCB(latent_dim=LATENT_DIM, hidden=CB_HIDDEN).to(device)
    optimizer = torch.optim.Adam(list(ec.parameters()) + list(cb.parameters()), lr=LR)
    history   = {"train_mse": [], "val_mse": [], "train_nce": [], "val_nce": []}

    for epoch in range(1, EPOCHS + 1):
        ec.train(); cb.train()
        total_mse, total_nce = 0.0, 0.0

        for trans, sw in train_loader:
            trans = trans.to(device)

            z, _ = ec(trans)                     # (B, T, 16)

            # ── Shifted MSE loss ──────────────────────────────────────────────
            z_prev   = z[:, :-1, :]              # z_0 .. z_{T-2}
            s_curr   = trans[:, 1:, :2]          # s_{t+1}
            a_curr   = trans[:, 1:, 2:4]         # a_{t+1}
            s_target = trans[:, 1:, 4:6]         # s_{t+2}
            mse_loss = F.mse_loss(cb(s_curr, a_curr, z_prev), s_target)

            # ── InfoNCE temporal clustering loss ─────────────────────────────
            nce_loss = infonce_posnegtime(z, sw)

            loss = mse_loss + LAMBDA_NCE * nce_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            total_mse += mse_loss.item() * len(trans)
            total_nce += nce_loss.item() * len(trans)

        total_mse /= N_TRAIN;  total_nce /= N_TRAIN
        history["train_mse"].append(total_mse)
        history["train_nce"].append(total_nce)

        ec.eval(); cb.eval()
        val_mse, val_nce = 0.0, 0.0
        with torch.no_grad():
            for trans, sw in val_loader:
                trans    = trans.to(device)
                z, _     = ec(trans)
                z_prev   = z[:, :-1, :]
                s_curr   = trans[:, 1:, :2]
                a_curr   = trans[:, 1:, 2:4]
                s_target = trans[:, 1:, 4:6]
                val_mse += F.mse_loss(cb(s_curr, a_curr, z_prev), s_target).item() * len(trans)
                val_nce += infonce_posnegtime(z, sw).item() * len(trans)
        val_mse /= N_VAL;  val_nce /= N_VAL
        history["val_mse"].append(val_mse)
        history["val_nce"].append(val_nce)

        print(f"Epoch {epoch:03d} | MSE: {total_mse:.4f} / {val_mse:.4f} | "
              f"NCE: {total_nce:.4f} / {val_nce:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"ec": ec.state_dict(), "cb": cb.state_dict(), "history": history},
               "checkpoints/ectocb_mse_nce.pt")
    print("\nSaved checkpoints/ectocb_mse_nce.pt")

    print("\nEvaluating context encoding in z_t...")
    evaluate_and_plot(ec, cb, va_trans, va_ctx, history, device)


if __name__ == "__main__":
    t0 = time.time()
    train()
    print(f"\nTotal training time: {time.time() - t0:.1f}s")

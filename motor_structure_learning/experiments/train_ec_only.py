"""
train_ec_only.py

Two versions of EC-only training -- run each section independently.

SECTION 1: train_crossseq()
  - positive : same sequence, small Δt (same context block)
  - negatives: all OTHER sequences at the SAME timestep (cross-sequence)
  - saves with suffix _crossseq

SECTION 2: train_posnegtime()
  - positive : same sequence, small Δt (same context block, switch-guaranteed)
  - negative : same sequence, from the OTHER context block (far in time)
  - saves with suffix _posnegtime

Data: 4 contexts, each sequence switches between 2 of them at a random point.
  C0: (beta=+0.4, gamma=0.0) -- force along action, positive
  C1: (beta=-0.4, gamma=0.0) -- force along action, negative
  C2: (beta=0.0, gamma=+0.4) -- force perpendicular, right
  C3: (beta=0.0, gamma=-0.4) -- force perpendicular, left
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from motor_structure_learning.core import EntorhinalCortex


SEQ_LEN = 40
SMALL_DELTA = 3
N_TRAIN = 2000
N_VAL = 200
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3
TEMPERATURE = 0.1


def generate_sequences(n_seqs, seq_len=40, seed=0):
    rng = np.random.default_rng(seed)
    acts = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]

    all_trans, all_c, all_switch = [], [], []

    for _ in range(n_seqs):
        x, y = float(rng.integers(0, 5)), float(rng.integers(0, 5))

        all_ctx = [(0.4, 0.0, 0), (-0.4, 0.0, 1), (0.0, 0.4, 2), (0.0, -0.4, 3)]
        c_ids = rng.choice(4, size=2, replace=False)
        ctx_first, ctx_second = all_ctx[c_ids[0]], all_ctx[c_ids[1]]

        switch = int(rng.integers(10, 31))

        seq_trans, seq_ctx = [], []

        for t in range(seq_len):
            beta, gamma, c_type = ctx_first if t < switch else ctx_second
            beta, gamma, c_type = float(beta), float(gamma), int(c_type)

            dx, dy = acts[rng.integers(0, 4)]
            a = np.array([dx, dy], dtype=np.float32)
            a_norm = np.linalg.norm(a)
            a_hat = a / a_norm if a_norm > 0 else np.zeros(2, dtype=np.float32)
            a_hat_perp = np.array([-a_hat[1], a_hat[0]], dtype=np.float32)

            f_env = beta * a_hat + gamma * a_hat_perp
            x_next = float(np.clip(x + dx + f_env[0], 0, 4))
            y_next = float(np.clip(y + dy + f_env[1], 0, 4))

            seq_trans.append([x, y, dx, dy, x_next, y_next])
            seq_ctx.append([beta, gamma, c_type])
            x, y = x_next, y_next

        all_trans.append(seq_trans)
        all_c.append(seq_ctx)
        all_switch.append(switch)

    return (
        torch.tensor(np.array(all_trans), dtype=torch.float32),
        torch.tensor(np.array(all_c), dtype=torch.float32),
        torch.tensor(np.array(all_switch), dtype=torch.long),
    )


class TransitionDataset(Dataset):
    def __init__(self, transitions, switches):
        self.transitions = transitions
        self.switches = switches

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx], self.switches[idx]


def _lazy_eval_imports():
    try:
        import matplotlib.pyplot as plt
        import scipy.stats as stats
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        from sklearn.model_selection import cross_val_score
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn and matplotlib are required to run EC-only training/evaluation. "
            "Install the project dependencies first."
        ) from exc

    return plt, stats, PCA, LinearRegression, LogisticRegression, confusion_matrix, ConfusionMatrixDisplay, cross_val_score


NEG_LABEL = {
    "crossseq": "negatives = other sequences (cross-seq)",
    "posnegtime": "negatives = other context block (within-seq)",
}


def evaluate_and_plot(ec, va_trans, va_ctx, va_switch, history, suffix, device, batch_size):
    plt, stats, PCA, LinearRegression, LogisticRegression, confusion_matrix, ConfusionMatrixDisplay, cross_val_score = _lazy_eval_imports()

    ec.eval()
    all_z, all_ctype = [], []

    with torch.no_grad():
        for i in range(0, len(va_trans), batch_size):
            trans = va_trans[i:i + batch_size].to(device)
            z, _ = ec(trans)
            ctx = va_ctx[i:i + batch_size]
            B, T, D = z.shape
            all_z.append(z.cpu().reshape(B * T, D))
            all_ctype.append(ctx[:, :, 2].reshape(B * T))

    all_z = torch.cat(all_z).numpy()
    all_ctype = torch.cat(all_ctype).numpy()

    neg_desc = NEG_LABEL.get(suffix, suffix)

    r_per_dim = [abs(stats.pearsonr(all_z[:, d], all_ctype)[0]) for d in range(16)]
    r_ctype = max(r_per_dim)
    r2_ctype = cross_val_score(LinearRegression(), all_z, all_ctype, cv=5, scoring="r2").mean()
    acc_ctype = cross_val_score(LogisticRegression(max_iter=500), all_z, all_ctype.astype(int), cv=5, scoring="accuracy").mean()
    print(f"Max |r| z_t vs c_type  : {r_ctype:.4f}")
    print(f"Linear probe R²         : {r2_ctype:.4f}  (1=perfect, 0=chance, <0=worse than mean)")
    print(f"Logistic decode accuracy: {acc_ctype:.4f}  (chance=0.25 for 4 contexts)")

    os.makedirs("figs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    colors = {0: "steelblue", 1: "tomato", 2: "seagreen", 3: "orange"}
    c_labels = {0: "C0 (+beta)", 1: "C1 (-beta)", 2: "C2 (+gamma)", 3: "C3 (-gamma)"}

    plt.figure(figsize=(7, 4))
    plt.plot(history["train_nce"], label="train")
    plt.plot(history["val_nce"], label="val")
    plt.title(f"EC InfoNCE loss\n{neg_desc}")
    plt.xlabel("Epoch")
    plt.ylabel("InfoNCE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/ec_only_training_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ec_only_training_{suffix}.png")

    n_plot = min(2000, len(all_z))
    idx = np.random.choice(len(all_z), n_plot, replace=False)
    z_sub = all_z[idx]
    ctx_sub = all_ctype[idx]
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z_sub)

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(4):
        mask = ctx_sub == c
        if mask.sum() > 0:
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=colors[c], label=c_labels[c], alpha=0.4, s=10)
    ax.set_title(f"PCA of z_t -- coloured by context\n{neg_desc}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figs/ec_only_pca_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ec_only_pca_{suffix}.png")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(16), r_per_dim, color="steelblue")
    ax.axhline(0.1, color="tomato", linestyle="--", label="|r|=0.1 reference")
    ax.set_xlabel("z dimension")
    ax.set_ylabel("|Pearson r| vs c_type")
    ax.set_title(f"Per-dim correlation with context\n{neg_desc}")
    ax.set_xticks(range(16))
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"figs/ec_only_perdim_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ec_only_perdim_{suffix}.png")

    N_val = len(va_trans)
    all_z_seq = all_z.reshape(N_val, SEQ_LEN, 16)
    all_ctype_seq = all_ctype.reshape(N_val, SEQ_LEN)

    acc_over_t = []
    for t in range(SEQ_LEN):
        z_t = all_z_seq[:, t, :]
        c_t = all_ctype_seq[:, t].astype(int)
        acc = cross_val_score(LogisticRegression(max_iter=500), z_t, c_t, cv=5, scoring="accuracy").mean()
        acc_over_t.append(acc)

    avg_sw = va_switch.float().mean().item()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(acc_over_t, color="steelblue", label="decode accuracy")
    ax.axhline(0.25, color="gray", linestyle="--", label="chance (0.25)")
    ax.axvline(avg_sw, color="tomato", linestyle="--", label=f"avg switch (t≈{avg_sw:.0f})")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Context decode accuracy")
    ax.set_title(f"Context decodability over time\n{neg_desc}")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"figs/ec_only_decodetime_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ec_only_decodetime_{suffix}.png")

    clf = LogisticRegression(max_iter=500)
    clf.fit(all_z, all_ctype.astype(int))
    cm = confusion_matrix(all_ctype.astype(int), clf.predict(all_z))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["C0", "C1", "C2", "C3"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"Context confusion matrix\n{neg_desc}")
    plt.tight_layout()
    plt.savefig(f"figs/ec_only_confusion_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figs/ec_only_confusion_{suffix}.png")


def infonce_crossseq(z, switch_points, small_delta=SMALL_DELTA, temperature=TEMPERATURE):
    B, T, D = z.shape

    t_anchor = torch.zeros(B, dtype=torch.long, device=z.device)
    for i in range(B):
        sw = switch_points[i].item()
        b1_max = sw - small_delta - 1
        b2_max = T - small_delta - 1
        if b1_max > 0 and torch.randint(0, 2, (1,)).item() == 0:
            t_anchor[i] = torch.randint(0, b1_max, (1,)).item()
        else:
            t_anchor[i] = torch.randint(sw, b2_max, (1,)).item()

    dt_pos = torch.randint(1, small_delta + 1, (B,), device=z.device)
    t_pos = t_anchor + dt_pos

    anchors = F.normalize(z[torch.arange(B), t_anchor], dim=-1)
    positives = F.normalize(z[torch.arange(B), t_pos], dim=-1)

    sim_pos = (anchors * positives).sum(dim=-1, keepdim=True) / temperature
    sim_neg = torch.mm(anchors, anchors.T) / temperature
    sim_neg = sim_neg.masked_fill(torch.eye(B, dtype=torch.bool, device=z.device), float("-inf"))

    logits = torch.cat([sim_pos, sim_neg], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


def train_crossseq():
    suffix = "crossseq"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 60}")
    print(f"SECTION 1: Cross-sequence negatives  (suffix: {suffix})")
    print(f"{'=' * 60}")
    print(f"Device: {device} | n_train: {N_TRAIN} | epochs: {EPOCHS} | batch: {BATCH_SIZE}\n")

    tr_trans, _, tr_switch = generate_sequences(N_TRAIN, seq_len=SEQ_LEN, seed=0)
    va_trans, va_ctx, va_switch = generate_sequences(N_VAL, seq_len=SEQ_LEN, seed=99)

    train_loader = DataLoader(TransitionDataset(tr_trans, tr_switch), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TransitionDataset(va_trans, va_switch), batch_size=BATCH_SIZE)

    ec = EntorhinalCortex(input_dim=6, hidden_dim=256, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(ec.parameters(), lr=LR)
    history = {"train_nce": [], "val_nce": []}

    for epoch in range(1, EPOCHS + 1):
        ec.train()
        total_nce = 0.0
        for trans, sw in train_loader:
            trans = trans.to(device)
            z, _ = ec(trans)
            loss = infonce_crossseq(z, sw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_nce += loss.item() * len(trans)
        total_nce /= N_TRAIN
        history["train_nce"].append(total_nce)

        ec.eval()
        val_nce = 0.0
        with torch.no_grad():
            for trans, sw in val_loader:
                trans = trans.to(device)
                z, _ = ec(trans)
                val_nce += infonce_crossseq(z, sw).item() * len(trans)
        val_nce /= N_VAL
        history["val_nce"].append(val_nce)
        print(f"Epoch {epoch:03d} | Train NCE: {total_nce:.4f} | Val NCE: {val_nce:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state_dict": ec.state_dict(), "history": history}, f"checkpoints/ec_only_{suffix}.pt")
    print(f"\nSaved checkpoints/ec_only_{suffix}.pt")
    print("\nEvaluating...")
    evaluate_and_plot(ec, va_trans, va_ctx, va_switch, history, suffix, device, BATCH_SIZE)


def infonce_posnegtime(z, switch_points, small_delta=SMALL_DELTA, temperature=TEMPERATURE):
    B, T, D = z.shape

    t_anchor = torch.zeros(B, dtype=torch.long, device=z.device)
    t_pos = torch.zeros(B, dtype=torch.long, device=z.device)
    t_neg = torch.zeros(B, dtype=torch.long, device=z.device)

    for i in range(B):
        sw = switch_points[i].item()
        b1_max = sw - small_delta - 1
        b2_max = T - small_delta - 1

        if b1_max > 0 and torch.randint(0, 2, (1,)).item() == 0:
            t_anchor[i] = torch.randint(0, b1_max, (1,)).item()
            t_pos[i] = t_anchor[i] + torch.randint(1, small_delta + 1, (1,)).item()
            t_neg[i] = torch.randint(sw, T, (1,)).item()
        else:
            t_anchor[i] = torch.randint(sw, b2_max, (1,)).item()
            t_pos[i] = t_anchor[i] + torch.randint(1, small_delta + 1, (1,)).item()
            t_neg[i] = torch.randint(0, sw, (1,)).item()

    anchors = F.normalize(z[torch.arange(B), t_anchor], dim=-1)
    positives = F.normalize(z[torch.arange(B), t_pos], dim=-1)
    negatives = F.normalize(z[torch.arange(B), t_neg], dim=-1)

    beta = (anchors * positives).sum(dim=-1, keepdim=True) / temperature
    alpha = (anchors * negatives).sum(dim=-1, keepdim=True) / temperature

    logits = torch.cat([beta, alpha], dim=-1)
    labels = torch.zeros(B, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


def train_posnegtime():
    suffix = "posnegtime"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'=' * 60}")
    print(f"SECTION 2: Pos/Neg by time (within-sequence)  (suffix: {suffix})")
    print(f"{'=' * 60}")
    print(f"Device: {device} | n_train: {N_TRAIN} | epochs: {EPOCHS} | batch: {BATCH_SIZE}\n")

    tr_trans, _, tr_switch = generate_sequences(N_TRAIN, seq_len=SEQ_LEN, seed=0)
    va_trans, va_ctx, va_switch = generate_sequences(N_VAL, seq_len=SEQ_LEN, seed=99)

    train_loader = DataLoader(TransitionDataset(tr_trans, tr_switch), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TransitionDataset(va_trans, va_switch), batch_size=BATCH_SIZE)

    ec = EntorhinalCortex(input_dim=6, hidden_dim=256, latent_dim=16).to(device)
    optimizer = torch.optim.Adam(ec.parameters(), lr=LR)
    history = {"train_nce": [], "val_nce": []}

    for epoch in range(1, EPOCHS + 1):
        ec.train()
        total_nce = 0.0
        for trans, sw in train_loader:
            trans = trans.to(device)
            z, _ = ec(trans)
            loss = infonce_posnegtime(z, sw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_nce += loss.item() * len(trans)
        total_nce /= N_TRAIN
        history["train_nce"].append(total_nce)

        ec.eval()
        val_nce = 0.0
        with torch.no_grad():
            for trans, sw in val_loader:
                trans = trans.to(device)
                z, _ = ec(trans)
                val_nce += infonce_posnegtime(z, sw).item() * len(trans)
        val_nce /= N_VAL
        history["val_nce"].append(val_nce)
        print(f"Epoch {epoch:03d} | Train NCE: {total_nce:.4f} | Val NCE: {val_nce:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state_dict": ec.state_dict()}, f"checkpoints/ec_only_{suffix}.pt")
    print(f"\nSaved checkpoints/ec_only_{suffix}.pt")
    print("\nEvaluating...")
    evaluate_and_plot(ec, va_trans, va_ctx, va_switch, history, suffix, device, BATCH_SIZE)


def plot_from_checkpoint(suffix):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = f"checkpoints/ec_only_{suffix}.pt"
    print(f"\nLoading {ckpt_path} ...")

    ec = EntorhinalCortex(input_dim=6, hidden_dim=256, latent_dim=16).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ec.load_state_dict(ckpt["model_state_dict"])
    history = ckpt.get("history", {"train_nce": [], "val_nce": []})

    va_trans, va_ctx, va_switch = generate_sequences(N_VAL, seq_len=SEQ_LEN, seed=99)
    evaluate_and_plot(ec, va_trans, va_ctx, va_switch, history, suffix, device, BATCH_SIZE)


def main():
    t0 = time.time()

    train_crossseq()
    train_posnegtime()

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
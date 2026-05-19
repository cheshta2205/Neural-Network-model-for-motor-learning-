"""
MLP_6C.py — nonlinear context decoder for the 6-context EC→CB model.

Motivation
----------
Earlier GLM analyses (LogisticRegression / Ridge on z_ec) showed context is
linearly decodable, but the directions carrying that context info capture only
a tiny fraction of z_ec variance. That pattern is consistent with mixed /
nonlinear context coding: the variance-dominant axes don't separate contexts
in a single linear cut.

This script does two things:
  1) Trains a multi-layer MLP classifier  z_ec → context_id  and compares its
     accuracy to the linear logistic baseline. If MLP ≫ linear → there is
     nonlinear context information that the GLM cannot read.
  2) Tests context geometry: for each context-pair (c_a, c_b), bins (s, a)
     and computes the per-bin centroid displacement
         Δ_bin = mean z_ec(c_b, s, a) - mean z_ec(c_a, s, a).
     If Δ_bin is roughly constant across bins → context = parallel translation
     in z_ec space (linear coding). If Δ_bin varies a lot with (s, a) →
     context warps state/action differently (mixed selectivity).

Architecture contrast with the earlier GLM probe
------------------------------------------------
  GLM (LogisticRegression):  z_ec ─W─► logits_6           one linear map
                              one weight matrix W ∈ ℝ^(6×32), no hidden layer
  MLP_6C:                    z_ec ─W1─► h1 ─ReLU─► W2 ─► h2 ─ReLU─► W3 ─► logits_6
                              two hidden ReLU layers (32 → 64 → 64 → 6)

The linear decoder is the special case of MLP_6C with depth 0 and no
nonlinearity. So any decode gain (MLP − linear) is exactly the nonlinear /
mixed-selectivity component of context coding in z_ec.
"""

import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA as SK_PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "train_mod", os.path.join(HERE, "ECtoCB_train_MSE_6c.py")
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

TAG = "EC→CB | velocity dynamics | 6 contexts | nonlinear decoder"
N_CTX = len(M.CONTEXTS)
COLORS = {0: "steelblue", 1: "cornflowerblue",
          2: "seagreen",  3: "limegreen",
          4: "purple",    5: "violet"}


# ── MLP decoder ───────────────────────────────────────────────────────────────
class MLP_6C(nn.Module):
    """Nonlinear classifier: z_ec (LATENT_DIM) → context logits (6)."""
    def __init__(self, in_dim=M.LATENT_DIM, hidden=64, n_classes=N_CTX, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(z_tr, c_tr, z_te, c_te, device,
              epochs=120, lr=1e-3, batch=512, weight_decay=1e-4):
    """Train MLP_6C on one fold, return test accuracy and the trained model."""
    mlp = MLP_6C().to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    z_tr_t = torch.tensor(z_tr, dtype=torch.float32, device=device)
    c_tr_t = torch.tensor(c_tr, dtype=torch.long,    device=device)
    z_te_t = torch.tensor(z_te, dtype=torch.float32, device=device)
    c_te_t = torch.tensor(c_te, dtype=torch.long,    device=device)
    N = len(z_tr_t)

    for _ in range(epochs):
        mlp.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            logits = mlp(z_tr_t[idx])
            loss = F.cross_entropy(logits, c_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    mlp.eval()
    with torch.no_grad():
        acc = (mlp(z_te_t).argmax(-1) == c_te_t).float().mean().item()
    return acc, mlp


# ── Data collection ───────────────────────────────────────────────────────────
def collect_z_state_action(ec, cb, va_trans, va_ctx, device):
    """Run val rollouts and return (z_ec, c_id, s_t, a_t) flat arrays."""
    all_z, all_c, all_s, all_a = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(va_trans), M.BATCH_SIZE):
            trans = va_trans[i:i+M.BATCH_SIZE].to(device)
            ctx   = va_ctx[i:i+M.BATCH_SIZE]
            z, _  = M.run_sequence(ec, cb, trans, device)
            B, T, D = z.shape
            all_z.append(z.cpu().reshape(B * T, D))
            all_c.append(ctx[:, :, 2].reshape(B * T))
            all_s.append(trans[:, :, :2].cpu().reshape(B * T, 2))
            all_a.append(trans[:, :, 2:4].cpu().reshape(B * T, 2))
    z = torch.cat(all_z).numpy()
    c = torch.cat(all_c).numpy().astype(int)
    s = torch.cat(all_s).numpy()
    a = torch.cat(all_a).numpy()
    return z, c, s, a


# ── Variance-regularized MLP (Elom's combined idea) ───────────────────────────
class MLP_6C_var(nn.Module):
    """
    Multilayer MLP context classifier with a reconstruction head.

    Architecture:
        z_ec ── Linear → ReLU → Linear → ReLU ── h (bottleneck, e.g. 8D)
                                                  ├── classifier → 6 logits
                                                  └── decoder    → ẑ_ec (32D)

    Loss = CE(logits, c)  +  λ · MSE(ẑ_ec, z_ec)

    The reconstruction head forces the bottleneck h to retain z_ec
    information — i.e. to capture variance from the input population.
    Sweeping λ traces out the tradeoff:
      λ=0      → free to use any (possibly low-variance) directions   (plain MLP)
      λ→large  → bottleneck must span the top variance directions of z_ec
    """
    def __init__(self, in_dim=M.LATENT_DIM, bottleneck=8, hidden=64,
                 n_classes=N_CTX, p_drop=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),       nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, bottleneck),   nn.ReLU(),
        )
        self.classifier = nn.Linear(bottleneck, n_classes)
        self.decoder    = nn.Linear(bottleneck, in_dim)

    def forward(self, x):
        h      = self.encoder(x)
        logits = self.classifier(h)
        x_hat  = self.decoder(h)
        return logits, x_hat, h


def train_mlp_var(z_tr, c_tr, z_te, c_te, device, lam=1.0,
                  epochs=120, lr=1e-3, batch=512, weight_decay=1e-4,
                  bottleneck=8):
    """Train MLP_6C_var with weighted CE + reconstruction loss."""
    mlp = MLP_6C_var(bottleneck=bottleneck).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    z_tr_t = torch.tensor(z_tr, dtype=torch.float32, device=device)
    c_tr_t = torch.tensor(c_tr, dtype=torch.long,    device=device)
    z_te_t = torch.tensor(z_te, dtype=torch.float32, device=device)
    c_te_t = torch.tensor(c_te, dtype=torch.long,    device=device)
    N = len(z_tr_t)

    for _ in range(epochs):
        mlp.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            logits, x_hat, _ = mlp(z_tr_t[idx])
            ce  = F.cross_entropy(logits, c_tr_t[idx])
            rec = F.mse_loss(x_hat, z_tr_t[idx])
            loss = ce + lam * rec
            opt.zero_grad(); loss.backward(); opt.step()

    mlp.eval()
    with torch.no_grad():
        logits, x_hat, _ = mlp(z_te_t)
        acc = (logits.argmax(-1) == c_te_t).float().mean().item()
        ss_res = ((z_te_t - x_hat) ** 2).sum().item()
        ss_tot = ((z_te_t - z_te_t.mean(0)) ** 2).sum().item()
        rec_r2 = 1.0 - ss_res / ss_tot
    return acc, rec_r2


# ── Per-timestep decode curve (linear vs MLP) ─────────────────────────────────
def decode_acc_over_time(z_seq, c_seq):
    """
    Mirror the existing GLM 'decodetime' protocol: at every timestep t, fit
    a fresh decoder on (z_t, c_t) across sequences with 5-fold CV. Returns
    one accuracy per timestep for both the linear probe and an MLP_6C-style
    sklearn MLPClassifier.

    z_seq : (N_val, T, LATENT_DIM)
    c_seq : (N_val, T) int
    """
    N_val, T, _ = z_seq.shape
    acc_lin = np.zeros(T)
    acc_mlp = np.zeros(T)
    for t in range(T):
        z_t = z_seq[:, t, :]
        c_t = c_seq[:, t].astype(int)
        # if any timestep has only one class present, skip (acc=nan handled)
        if len(np.unique(c_t)) < 2:
            acc_lin[t] = np.nan; acc_mlp[t] = np.nan; continue
        acc_lin[t] = cross_val_score(
            LogisticRegression(max_iter=500), z_t, c_t,
            cv=5, scoring="accuracy"
        ).mean()
        # MLPClassifier with 2 hidden layers — same family as MLP_6C, but
        # smaller hidden width to avoid overfitting on the 200 samples/timestep
        acc_mlp[t] = cross_val_score(
            MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=400,
                          alpha=1e-3, random_state=0),
            z_t, c_t, cv=5, scoring="accuracy"
        ).mean()
        if t % 10 == 0:
            print(f"  t={t:3d}  linear={acc_lin[t]:.3f}  MLP={acc_mlp[t]:.3f}")
    return acc_lin, acc_mlp


# ── Geometry: Δ(s,a) across context pairs ─────────────────────────────────────
def context_displacement_geometry(z, c, s, a, n_bins=4):
    """
    For every (c_a, c_b) context pair, bin (s, a) and compute the per-bin
    centroid displacement Δ_bin in z_ec space. Then report:
        ||mean(Δ)||             — the "average translation" magnitude
        std(Δ across bins)      — how (s,a)-dependent the displacement is
        cosine spread           — angular variation across bins
    A small std and tight cosine spread → near-parallel translation (linear).
    Larger values → state/action-dependent warping (mixed selectivity).
    """
    # bin states (2D) and actions (2D) into n_bins per axis
    s_edges = [np.quantile(s[:, d], np.linspace(0, 1, n_bins + 1)) for d in range(2)]
    a_edges = [np.quantile(a[:, d], np.linspace(0, 1, n_bins + 1)) for d in range(2)]

    def bin_idx(x, edges):
        return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)

    sb = np.stack([bin_idx(s[:, d], s_edges[d]) for d in range(2)], axis=1)
    ab = np.stack([bin_idx(a[:, d], a_edges[d]) for d in range(2)], axis=1)
    # combine 4 bin dims into a single bin id
    bin_id = sb[:, 0] + n_bins * (sb[:, 1] + n_bins * (ab[:, 0] + n_bins * ab[:, 1]))

    pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (0, 4), (2, 4)]
    rows = []
    for (ca, cb_id) in pairs:
        deltas = []
        for b in np.unique(bin_id):
            m_a = (c == ca)     & (bin_id == b)
            m_b = (c == cb_id)  & (bin_id == b)
            if m_a.sum() < 3 or m_b.sum() < 3:
                continue
            d = z[m_b].mean(0) - z[m_a].mean(0)
            deltas.append(d)
        if len(deltas) < 5:
            rows.append((ca, cb_id, np.nan, np.nan, np.nan, 0))
            continue
        D = np.stack(deltas)
        mu = D.mean(0)
        mu_norm = np.linalg.norm(mu) + 1e-12
        std_norm = D.std(0).mean()
        # cosine similarity of each Δ_bin to the mean Δ
        cos = (D @ mu) / (np.linalg.norm(D, axis=1) * mu_norm + 1e-12)
        rows.append((ca, cb_id, float(mu_norm), float(std_norm),
                     float(cos.mean()), len(deltas)))
    return rows


# ── Main analysis ─────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Discover seeded checkpoints (CKPT_PATH with `_seed{N}` suffix). If any
    # exist, average held-out MLP confusion matrices across all (seed, fold)
    # pairs to tighten the symmetric confusability used for the MDS geometry.
    base_path    = M.CKPT_PATH
    seeded_paths = sorted(glob.glob(base_path.replace(".pt", "_seed*.pt")))
    if seeded_paths:
        ckpt_paths = seeded_paths
        print(f"Found {len(ckpt_paths)} seeded checkpoints — confusion will be "
              f"averaged across them:")
        for p in ckpt_paths:
            print(f"  - {os.path.basename(p)}")
    else:
        ckpt_paths = [base_path]
        print(f"No seeded checkpoints found. Using single checkpoint {base_path}")

    skf            = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    mlp_confs_norm = []   # list of (N_CTX, N_CTX) row-normalized confusion matrices
    lin_accs_all, mlp_accs_all = [], []

    # Reference checkpoint (first seed) — its z/c/s/a/model feed all the
    # downstream per-checkpoint analyses (decode-over-time, geometry, plots).
    z = c = s = a = z_s = None
    ec = cb = None

    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        ckpt = torch.load(ckpt_path, map_location=device)
        ec_i = M.EntorhinalCortex().to(device); ec_i.load_state_dict(ckpt["ec"])
        cb_i = M.CerebellumCB().to(device);     cb_i.load_state_dict(ckpt["cb"])
        ec_i.eval(); cb_i.eval()
        print(f"\n=== ckpt {ckpt_idx+1}/{len(ckpt_paths)}: "
              f"{os.path.basename(ckpt_path)} ===")

        va_trans, va_ctx = M.generate_sequences(M.N_VAL, seed=99)
        z_i, c_i, s_i, a_i = collect_z_state_action(ec_i, cb_i, va_trans, va_ctx, device)
        scaler_i = StandardScaler().fit(z_i)
        z_s_i    = scaler_i.transform(z_i)

        for fold, (tr, te) in enumerate(skf.split(z_s_i, c_i)):
            lin = LogisticRegression(max_iter=500)
            lin.fit(z_s_i[tr], c_i[tr])
            lin_acc = lin.score(z_s_i[te], c_i[te])
            mlp_acc, mlp_model = train_mlp(z_s_i[tr], c_i[tr],
                                           z_s_i[te], c_i[te], device)
            with torch.no_grad():
                z_te_t = torch.tensor(z_s_i[te], dtype=torch.float32, device=device)
                mlp_preds = mlp_model(z_te_t).argmax(-1).cpu().numpy()
            C = confusion_matrix(c_i[te], mlp_preds,
                                 labels=list(range(N_CTX))).astype(float)
            row_sums = C.sum(axis=1, keepdims=True)
            C_norm   = np.divide(C, row_sums, out=np.zeros_like(C), where=row_sums > 0)
            mlp_confs_norm.append(C_norm)
            print(f"  fold {fold}: linear={lin_acc:.4f}  MLP={mlp_acc:.4f}  "
                  f"Δ={mlp_acc - lin_acc:+.4f}")
            lin_accs_all.append(lin_acc); mlp_accs_all.append(mlp_acc)

        # Keep the first checkpoint's tensors for the downstream analyses.
        if ckpt_idx == 0:
            z, c, s, a, z_s = z_i, c_i, s_i, a_i, z_s_i
            ec, cb          = ec_i, cb_i

    n_total = len(lin_accs_all)
    lin_mean = float(np.mean(lin_accs_all))
    lin_se   = float(np.std(lin_accs_all) / np.sqrt(n_total))
    mlp_mean = float(np.mean(mlp_accs_all))
    mlp_se   = float(np.std(mlp_accs_all) / np.sqrt(n_total))
    print(f"\n  Linear  : {lin_mean:.4f} ± {lin_se:.4f}  "
          f"({len(ckpt_paths)} ckpts × 5 folds = {n_total} estimates)")
    print(f"  MLP_6C  : {mlp_mean:.4f} ± {mlp_se:.4f}")
    print(f"  Δ (MLP − linear): {mlp_mean - lin_mean:+.4f}")
    print(f"  Chance  : {1/N_CTX:.4f}")

    print(f"\n(downstream per-checkpoint analyses use first ckpt: "
          f"{os.path.basename(ckpt_paths[0])})")

    # ── 1c. Variance-regularized MLP — λ sweep (Elom's combined idea) ──────────
    # Extended to large λ so decode actually descends to chance (1/N_CTX) when
    # the reconstruction loss dominates the classification loss completely.
    BOTTLENECK = 8
    print(f"\n--- Variance-regularized MLP (bottleneck={BOTTLENECK}, "
          f"CE + λ · MSE recon) ---")
    print("  λ=0      → plain MLP (free to use low-variance dims)")
    print("  λ→large  → bottleneck forced to also reconstruct z_ec;")
    print("            CE contribution vanishes ⇒ decode acc → chance\n")
    lambdas  = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    accs_lam = np.zeros(len(lambdas))
    r2_lam   = np.zeros(len(lambdas))
    for li, lam in enumerate(lambdas):
        fold_accs, fold_r2s = [], []
        for fold, (tr, te) in enumerate(skf.split(z_s, c)):
            acc_v, r2_v = train_mlp_var(
                z_s[tr], c[tr], z_s[te], c[te], device,
                lam=lam, epochs=120, bottleneck=BOTTLENECK,
            )
            fold_accs.append(acc_v); fold_r2s.append(r2_v)
        accs_lam[li] = float(np.mean(fold_accs))
        r2_lam[li]   = float(np.mean(fold_r2s))
        print(f"  λ={lam:9.2f}  acc={accs_lam[li]:.4f}  rec_R²={r2_lam[li]:.4f}")

    # ── 1c-PCA. Linear PCA-k baseline for the same x-axis (reconstruction R²)
    # PCA-k is the optimal rank-k linear autoencoder (Eckart–Young). Comparing
    # the bottleneck's R² to it on the same x-axis answers: "is the nonlinear
    # encoder packing more variance into k dims than the best linear projection
    # would?" If r2_bottleneck > r2_pca_k there's either real nonlinear
    # encoding or a bug — normalizing the x-axis by r2_pca_k puts PCA at x=1
    # so any excess is visually obvious.
    pca_r2_folds, pca_acc_folds = [], []
    for tr, te in skf.split(z_s, c):
        pca = SK_PCA(n_components=BOTTLENECK).fit(z_s[tr])
        z_te_proj  = pca.transform(z_s[te])
        z_te_recon = pca.inverse_transform(z_te_proj)
        ss_res = float(((z_s[te] - z_te_recon) ** 2).sum())
        ss_tot = float(((z_s[te] - z_s[te].mean(0)) ** 2).sum())
        pca_r2_folds.append(1.0 - ss_res / ss_tot)
        z_tr_proj = pca.transform(z_s[tr])
        clf_pca   = LogisticRegression(max_iter=500).fit(z_tr_proj, c[tr])
        pca_acc_folds.append(clf_pca.score(z_te_proj, c[te]))
    r2_pca  = float(np.mean(pca_r2_folds))
    acc_pca = float(np.mean(pca_acc_folds))
    print(f"\n  PCA-{BOTTLENECK} baseline (5-fold CV):")
    print(f"    reconstruction R² of z_ec : {r2_pca:.4f}")
    print(f"    decode accuracy           : {acc_pca:.4f}")

    # ── 1b. Per-timestep decode curve (linear vs MLP) ──────────────────────────
    print("\n--- Per-timestep decode accuracy (5-fold CV per t) ---")
    z_seq = z_s.reshape(M.N_VAL, M.SEQ_LEN, M.LATENT_DIM)
    c_seq = c.reshape(M.N_VAL, M.SEQ_LEN)
    acc_lin_t, acc_mlp_t = decode_acc_over_time(z_seq, c_seq)
    print(f"  linear mean over t: {np.nanmean(acc_lin_t):.3f}")
    print(f"  MLP    mean over t: {np.nanmean(acc_mlp_t):.3f}")

    # ── 2. Geometry: per-pair Δ(s,a) constancy ─────────────────────────────────
    print("\n--- Context-pair displacement geometry (small std / cos~1 ⇒ parallel) ---")
    print(f"{'pair':>14}  {'||mean Δ||':>10}  {'std(Δ)':>8}  {'mean cos':>9}  {'#bins':>5}")
    geom = context_displacement_geometry(z_s, c, s, a, n_bins=4)
    for ca, cb_id, mu_n, std_n, cos_mean, n_bins_used in geom:
        label = f"C{ca} vs C{cb_id}"
        if np.isnan(mu_n):
            print(f"{label:>14}  (too few matched bins)")
        else:
            print(f"{label:>14}  {mu_n:10.3f}  {std_n:8.3f}  {cos_mean:9.3f}  {n_bins_used:5d}")

    # ── 3. Plots ───────────────────────────────────────────────────────────────
    os.makedirs(M.FIG_DIR, exist_ok=True)

    # 3a. Decode comparison bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = np.arange(2)
    ax.bar(xs, [lin_mean, mlp_mean],
           yerr=[lin_se, mlp_se], color=["gray", "steelblue"], capsize=4)
    ax.axhline(1 / N_CTX, color="tomato", ls="--", label=f"chance ({1/N_CTX:.2f})")
    ax.set_xticks(xs); ax.set_xticklabels(["Linear (GLM)", "MLP_6C"])
    ax.set_ylabel("5-fold context decode accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Nonlinear vs linear context decoding\n{TAG}")
    ax.legend()
    for i, (m, se) in enumerate([(lin_mean, lin_se), (mlp_mean, mlp_se)]):
        ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    out = os.path.join(M.FIG_DIR, "mlp6c_vs_linear_decode.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nWrote {out}")

    # 3a-tris. Variance-regularized MLP: accuracy vs (normalized) reconstruction.
    # x-axis is R²_bottleneck / R²_PCA-k, so PCA-k sits exactly at x=1 and any
    # bottleneck point with x>1 would mean the nonlinear encoder is packing
    # more variance into k dims than the optimal linear rank-k projection —
    # flag as a possible bug.
    r2_lam_norm = r2_lam / r2_pca if r2_pca > 0 else r2_lam

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r2_lam_norm, accs_lam, "-", color="0.6", lw=1, zorder=1,
            label="MLP_6C_var sweep (one point per λ)")
    ax.scatter(r2_lam_norm, accs_lam, c=np.arange(len(lambdas)),
               cmap="viridis", s=80, zorder=2, edgecolor="black", lw=0.5)
    for lam, r2n, acc in zip(lambdas, r2_lam_norm, accs_lam):
        ax.annotate(f"λ={lam:g}", (r2n, acc),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    # PCA-k reference (at x=1 by construction)
    ax.scatter([1.0], [acc_pca], marker="*", s=240, color="crimson",
               edgecolor="black", lw=0.8, zorder=3,
               label=f"PCA-{BOTTLENECK} baseline (acc={acc_pca:.3f})")
    ax.axvline(1.0, color="crimson", ls=":", lw=0.8, zorder=0)
    ax.axhline(lin_mean, color="gray",   ls="--", label=f"GLM baseline ({lin_mean:.3f})")
    ax.axhline(mlp_mean, color="black",  ls="--", label=f"plain MLP_6C ({mlp_mean:.3f})")
    ax.axhline(1 / N_CTX, color="tomato", ls=":",
               label=f"chance ({1/N_CTX:.2f})")
    ax.set_xlabel(f"Reconstruction R² / R²(PCA-{BOTTLENECK} = "
                  f"{r2_pca:.3f})    [1.0 = matches PCA]")
    ax.set_ylabel("Held-out context decode accuracy (5-fold CV)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(-0.05, float(np.min(r2_lam_norm)) - 0.05),
                max(1.1,  float(np.max(r2_lam_norm)) + 0.05))
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Variance-regularized MLP: decode vs reconstruction tradeoff\n{TAG}")
    plt.tight_layout()
    out = os.path.join(M.FIG_DIR, "mlp6c_var_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nWrote {out}")

    # 3a-bis. Per-timestep decode curve — overlay linear vs MLP
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(acc_lin_t, color="gray",      label=f"Linear (GLM) — mean={np.nanmean(acc_lin_t):.3f}")
    ax.plot(acc_mlp_t, color="steelblue", label=f"MLP_6C — mean={np.nanmean(acc_mlp_t):.3f}")
    ax.axhline(1 / N_CTX, color="tomato", ls="--", label=f"chance ({1/N_CTX:.2f})")
    ax.axvline(50,        color="tomato", ls=":",  label="avg switch (t≈50)")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("5-fold CV context decode accuracy")
    ax.set_title(f"Per-timestep decoder accuracy — linear vs MLP_6C\n{TAG}")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(M.FIG_DIR, "mlp6c_decodetime.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")

    # 3b. Geometry summary
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"C{a_}-C{b_}" for a_, b_, *_ in geom]
    std_norms = [g[3] for g in geom]
    cos_means = [g[4] for g in geom]
    axes[0].bar(range(len(labels)), std_norms, color="steelblue")
    axes[0].set_xticks(range(len(labels))); axes[0].set_xticklabels(labels, rotation=30)
    axes[0].set_ylabel("std(Δ) across (s,a) bins")
    axes[0].set_title("How (s,a)-dependent is the context displacement?")
    axes[1].bar(range(len(labels)), cos_means, color="seagreen")
    axes[1].axhline(1.0, color="black", lw=0.5, ls="--")
    axes[1].set_xticks(range(len(labels))); axes[1].set_xticklabels(labels, rotation=30)
    axes[1].set_ylabel("mean cos(Δ_bin, mean Δ)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Δ direction stability across bins (1 = pure parallel translation)")
    fig.suptitle(f"Context displacement geometry in z_ec\n{TAG}")
    plt.tight_layout()
    out = os.path.join(M.FIG_DIR, "mlp6c_pair_geometry.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")

    # 3c. Symmetrized confusion → exponential distance → MDS geometry
    # Raw row-normalized confusion C[i,j] = P(pred=j | true=i) is asymmetric
    # (decoder bias + cluster geometry make C[i,j] ≠ C[j,i]). A distance metric
    # must satisfy d(i,j) = d(j,i), so first symmetrize:
    #     sym_conf = (C + Cᵀ) / 2
    # Then apply Elom's exponential distance (Gaussian-style bandwidth σ ≈
    # chance-level confusion — smaller σ → sharper separation):
    #     D = exp((1 − sym_conf)² / σ²) − 1
    # The −1 makes D(i,i) = 0. Tune σ if the off-diagonal values blow up to
    # numerical infinity or if MDS stress is too high.
    SIGMA = 0.025
    conf_mean = np.mean(np.stack(mlp_confs_norm, axis=0), axis=0)   # (N_CTX, N_CTX)
    sym_conf  = 0.5 * (conf_mean + conf_mean.T)
    dist      = np.exp((1.0 - sym_conf) ** 2 / SIGMA ** 2) - 1.0
    np.fill_diagonal(dist, 0.0)

    print(f"\n--- Symmetrized row-normalized confusion (C + Cᵀ)/2, "
          f"avg over {len(mlp_confs_norm)} folds × seeds ---")
    print(np.array2string(sym_conf, precision=3, suppress_small=True))
    print(f"\n--- Exponential distance D = exp((1−sym_conf)²/σ²) − 1  "
          f"(σ={SIGMA}) ---")
    print(np.array2string(dist, precision=3, suppress_small=True))

    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=0, n_init=8, normalized_stress="auto")
    coords = mds.fit_transform(dist)
    print(f"MDS stress: {mds.stress_:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 0: symmetric confusion
    im0 = axes[0].imshow(sym_conf, cmap="magma", vmin=0, vmax=1)
    axes[0].set_xticks(range(N_CTX)); axes[0].set_yticks(range(N_CTX))
    axes[0].set_xticklabels([f"C{i}" for i in range(N_CTX)])
    axes[0].set_yticklabels([f"C{i}" for i in range(N_CTX)])
    axes[0].set_xlabel("context"); axes[0].set_ylabel("context")
    for i in range(N_CTX):
        for j in range(N_CTX):
            axes[0].text(j, i, f"{sym_conf[i, j]:.2f}",
                         ha="center", va="center",
                         color="white" if sym_conf[i, j] < 0.5 else "black",
                         fontsize=8)
    axes[0].set_title(f"Symmetric confusion (C + Cᵀ)/2\n"
                      f"avg over {len(mlp_confs_norm)} folds × seeds")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 1: exponential distance matrix
    dist_max = float(np.max(dist))
    im1 = axes[1].imshow(dist, cmap="viridis")
    axes[1].set_xticks(range(N_CTX)); axes[1].set_yticks(range(N_CTX))
    axes[1].set_xticklabels([f"C{i}" for i in range(N_CTX)])
    axes[1].set_yticklabels([f"C{i}" for i in range(N_CTX)])
    axes[1].set_xlabel("context"); axes[1].set_ylabel("context")
    for i in range(N_CTX):
        for j in range(N_CTX):
            v = dist[i, j]
            txt = f"{v:.1e}" if v >= 1000 else f"{v:.2f}"
            axes[1].text(j, i, txt, ha="center", va="center",
                         color="white" if v < dist_max / 2 else "black",
                         fontsize=7)
    axes[1].set_title(f"Distance D = exp((1−sim)²/σ²) − 1  (σ={SIGMA})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 2: MDS embedding
    for i in range(N_CTX):
        axes[2].scatter(coords[i, 0], coords[i, 1],
                        s=180, color=COLORS[i], edgecolor="black", lw=0.7)
        axes[2].annotate(f"C{i}", (coords[i, 0], coords[i, 1]),
                         textcoords="offset points", xytext=(8, 6), fontsize=11)
    axes[2].set_xlabel("MDS dim 1"); axes[2].set_ylabel("MDS dim 2")
    axes[2].set_title(f"MDS on exponential distance  (stress={mds.stress_:.3f})")
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Context geometry from MLP confusability\n{TAG}")
    plt.tight_layout()
    out = os.path.join(M.FIG_DIR, "mlp6c_confusion_mds.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

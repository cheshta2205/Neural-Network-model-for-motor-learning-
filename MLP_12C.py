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

from typing import Optional
from pathlib import Path
from itertools import combinations
from sklearn.decomposition import PCA as SK_PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from src.PlottingTools.plotting_functions import set_plt_params, make_axis_nice, make_3d_gif

import pdb

import importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "train_mod", os.path.join(HERE, "ECtoCB_train_MSE_12c.py")
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

TAG = "EC→CB | velocity dynamics | 12 contexts | nonlinear decoder"
N_CTX = len(M.CONTEXTS)
COLORS = M.COLORS
C_LABELS = M.C_LABELS


# ── MLP decoder ───────────────────────────────────────────────────────────────
class MLP_12C(nn.Module):
    """Nonlinear classifier: z_ec (LATENT_DIM) → context logits (12)."""
    def __init__(self, in_dim=M.LATENT_DIM, hidden=64, n_classes=N_CTX):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(z_tr, c_tr, z_te, c_te, device,
              epochs=120, lr=1e-3, batch=512, weight_decay=1e-4):
    """Train MLP_6C on one fold, return test accuracy and the trained model."""
    mlp = MLP_12C(in_dim=z_tr.shape[-1]).to(device)
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
    all_ec, all_cb_1, all_cb_2 = [], [], []
    with torch.no_grad():
        for i in range(0, len(va_trans), M.BATCH_SIZE):
            trans = va_trans[i:i+M.BATCH_SIZE].to(device)
            ctx   = va_ctx[i:i+M.BATCH_SIZE]
            z, _, representations = M.run_sequence(ec, cb, trans, device, True)
            B, T, D = z.shape
            all_z.append(z.cpu().reshape(B * T, D))
            all_ec.append(representations['EC_hidden'].cpu().reshape(B * T, -1))
            all_cb_1.append(representations['CB_hidden_1'].cpu().reshape(B * T, -1))
            all_cb_2.append(representations['CB_hidden_2'].cpu().reshape(B * T, -1))
            all_c.append(ctx[:, :, 2].reshape(B * T))
            all_s.append(trans[:, :, :2].cpu().reshape(B * T, 2))
            all_a.append(trans[:, :, 2:4].cpu().reshape(B * T, 2))

    z = torch.cat(all_z).numpy()
    ec = torch.cat(all_ec).numpy()
    cb_1 = torch.cat(all_cb_1).numpy()
    cb_2 = torch.cat(all_cb_2).numpy()
    c = torch.cat(all_c).numpy().astype(int)
    s = torch.cat(all_s).numpy()
    a = torch.cat(all_a).numpy()
    return z, c, s, a, ec, cb_1, cb_2


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
    def __init__(self, in_dim=M.LATENT_DIM, bottleneck=8,
                 n_classes=N_CTX):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, bottleneck),       nn.SiLU(),
            nn.Linear(bottleneck, bottleneck),   nn.SiLU(),
        )
        self.classifier = nn.Linear(bottleneck, n_classes)
        self.decoder    = nn.Linear(bottleneck, in_dim)

    def forward(self, x):
        h      = self.encoder(x)
        logits = self.classifier(h)
        return logits, h


def train_mlp_var(z_tr, c_tr, z_te, c_te, device, lam=1.0,
                  epochs=120, lr=1e-3, batch=512, weight_decay=1e-4,
                  bottleneck=64, input_dim: int = 32):
    """Train MLP_6C_var with weighted CE + reconstruction loss."""
    mlp = MLP_6C_var(bottleneck=bottleneck, in_dim=input_dim).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)

    z_tr_t = torch.tensor(z_tr, dtype=torch.float32, device=device)
    c_tr_t = torch.tensor(c_tr, dtype=torch.long,    device=device)
    z_te_t = torch.tensor(z_te, dtype=torch.float32, device=device)
    c_te_t = torch.tensor(c_te, dtype=torch.long,    device=device)
    N = len(z_tr_t)

    train_covariance = torch.cov(z_tr_t.t())
    test_covariance = torch.cov(z_te_t.t())

    for _ in range(epochs):
        mlp.train()
        perm = torch.randperm(N, device=device)
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            logits, _ = mlp(z_tr_t[idx])
            ce  = F.cross_entropy(logits, c_tr_t[idx])

            mlp_weights = mlp.encoder[0].weight
            Q, _ = torch.linalg.qr(mlp_weights.t())
            rec_error = 1 - torch.trace(Q.T @ train_covariance @ Q) / torch.trace(train_covariance)            
            loss = ce + lam * rec_error
            opt.zero_grad(); loss.backward(); opt.step()

    mlp.eval()
    with torch.no_grad():
        logits, _ = mlp(z_te_t)
        acc = (logits.argmax(-1) == c_te_t).float().mean().item()
        mlp_weights = mlp.encoder[0].weight
        Q, _ = torch.linalg.qr(mlp_weights.t())
        U, S, V = torch.linalg.svd(test_covariance)
        rec_r2 = torch.trace(Q.T @ test_covariance @ Q) / torch.trace( U[:, :mlp_weights.shape[0]].T @ test_covariance @ U[:, :mlp_weights.shape[0]])     
    return acc, rec_r2


# ── Per-timestep decode curve (linear vs MLP) ─────────────────────────────────
def decode_acc_over_time_per_step(z_seq, c_seq, n_folds: int = 3):
    """
    Mirror the existing GLM 'decodetime' protocol: at every timestep t, fit
    a fresh decoder on (z_t, c_t) across sequences with n-fold CV. Returns
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
            cv=n_folds, scoring="accuracy"
        ).mean()
        # MLPClassifier with 2 hidden layers — same family as MLP_6C, but
        # smaller hidden width to avoid overfitting on the 200 samples/timestep
        acc_mlp[t] = cross_val_score(
            MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=500,
                          alpha=1e-3, random_state=0),
            z_t, c_t, cv=n_folds, scoring="accuracy"
        ).mean()
        if t % 10 == 0:
            print(f"  t={t:3d}  linear={acc_lin[t]:.3f}  MLP={acc_mlp[t]:.3f}")
    return acc_lin, acc_mlp

def decode_acc_over_time(z_seq, c_seq, n_folds: int = 5):
    """
    Fit a context decoder across all timesteps in the sequence, and evaluate its performance per timestep.

    z_seq : (N_val, T, LATENT_DIM)
    c_seq : (N_val, T) int
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_val, T, LATENT_DIM = z_seq.shape

    acc_lin = np.zeros((n_folds, T))
    acc_mlp = np.zeros((n_folds, T))

    for fold in range(n_folds):
        tr = np.random.choice(np.arange(N_val), size=int(0.8 * N_val), replace=False)
        te = np.setdiff1d(np.arange(N_val), tr)

        z_tr = z_seq[tr].reshape(-1, LATENT_DIM)
        c_tr = c_seq[tr].reshape(-1)
        z_te = z_seq[te].reshape(-1, LATENT_DIM)
        c_te = c_seq[te].reshape(-1)

        lin = LogisticRegression(max_iter=500)
        lin.fit(z_tr, c_tr)

        for t in range(T):
            acc_lin[fold, t] = lin.score(z_seq[te, t, :].reshape(-1, LATENT_DIM), c_seq[te, t])

        _, mlp_model = train_mlp(z_tr, c_tr, z_te, c_te, device)
        with torch.no_grad():
            z_te_t = torch.tensor(z_te, dtype=torch.float32, device=device)
            mlp_preds = mlp_model(z_te_t).argmax(-1).cpu().numpy()
            mlp_acc = (mlp_preds == c_te)
        
        mlp_acc = mlp_acc.reshape(-1, T)
        acc_mlp[fold, :] = mlp_acc.mean(axis=0)
        
    return acc_lin.mean(axis=0), acc_mlp.mean(axis=0)


# ── Geometry: Δ(s,a) across context pairs ─────────────────────────────────────
def context_displacement_geometry(z, c, s, a, n_bins=25):
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
    unique_contexts = np.unique(c)
    n_contexts = len(unique_contexts)

    rows = np.zeros((n_contexts, n_contexts, 4))
    # compute the typical distance between points in the latent space
    typical_distance = np.linalg.norm(np.std(z, axis=0))

    for ca, cb in combinations(unique_contexts, 2):
        a_idx = np.where(unique_contexts == ca)[0]
        b_idx = np.where(unique_contexts == cb)[0]
        deltas = []
        for b in np.unique(bin_id):
            m_a = (c == ca)     & (bin_id == b)
            m_b = (c == cb)  & (bin_id == b)
            if m_a.sum() < 3 or m_b.sum() < 3:
                continue
            d = z[m_b].mean(0) - z[m_a].mean(0)
            deltas.append(d)

        if len(deltas) < 5:
            rows[a_idx, b_idx] = [np.nan, np.nan, np.nan, 0]
            rows[b_idx, a_idx] = [np.nan, np.nan, np.nan, 0]
            continue
        D = np.stack(deltas) / typical_distance
        mu = D.mean(0)
        mu_norm = np.linalg.norm(mu) + 1e-12
        std_norm = D.std(0).mean()
        # cosine similarity of each Δ_bin to the mean Δ
        cos = (D @ mu) / (np.linalg.norm(D, axis=1) * mu_norm + 1e-12)
        # rows.append((ca, cb_id, float(mu_norm), float(std_norm),
        #              float(cos.mean()), len(deltas)))
        rows[a_idx, b_idx] = [float(mu_norm), float(std_norm), float(cos.mean()), len(deltas)]
        rows[b_idx, a_idx] = [float(mu_norm), float(std_norm), float(cos.mean()), len(deltas)]
    return rows


def plot_pca(representation, axes, context_labels, title,
            color_map: Optional[dict] = None, plot_frac: float = 0.3,
            base_dir: Optional[Path] = None):
    
    unique_contexts = np.unique(context_labels)

    if base_dir is None:
        base_dir = Path().cwd()

    if color_map is None:
        color_map = {}

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')

    for ctx in unique_contexts:
        mask = context_labels == ctx
        reps = representation[mask]
        idxs = np.random.choice(len(reps), size=int(len(reps) 
        * plot_frac), replace=False)
        reps = reps[idxs]
        projections = reps @ axes[:3].T

        ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2], 
                   color=[color_map.get(ctx, 'gray')], s=4, alpha=0.3)
    
    ax.axis('equal')
    ax.set_title(title)
    ax.grid(False)
    save_folder = base_dir / title 
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / f"3d_scatter.gif"

    make_3d_gif(ax=ax, fig=fig, save_path=save_path)
    plt.close(fig)

# ── Main analysis ─────────────────────────────────────────────────────────────
def main(cross_val_folds: int = 5):
    context_labels = M.C_LABELS
    set_plt_params(style='fast', file_format='svg')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # Discover seeded checkpoints (CKPT_PATH with `_seed{N}` suffix). If any
    # exist, average held-out MLP confusion matrices across all (seed, fold)
    # pairs to tighten the symmetric confusability used for the MDS geometry.
    base_dir    = Path(M.CKPT_PATH)

    ckpt_paths = [f for f in base_dir.iterdir() if f.is_file() and '_seed' in f.name]

    skf            = StratifiedKFold(n_splits=cross_val_folds, shuffle=True, random_state=0)
    mlp_confs_norm = []   # list of (N_CTX, N_CTX) row-normalized confusion matrices
    lin_accs_all, mlp_accs_all = [], []

    # Reference checkpoint (first seed) — its z/c/s/a/model feed all the
    # downstream per-checkpoint analyses (decode-over-time, geometry, plots).
    z = c = s = a = z_s = ec_r = cb_1_r = cb_2_r = ctx_ctrl = None
    ec = cb = None

    n_seeds = len(ckpt_paths)
    n_seqs = 500
    time_decoding = {
        'z_latent': {'linear': np.zeros((n_seeds, M.SEQ_LEN)), 'mlp': np.zeros((n_seeds, M.SEQ_LEN))},
        'ec': {'linear': np.zeros((n_seeds, M.SEQ_LEN)), 'mlp': np.zeros((n_seeds, M.SEQ_LEN))},
        'cb_1': {'linear': np.zeros((n_seeds, M.SEQ_LEN)), 'mlp': np.zeros((n_seeds, M.SEQ_LEN))},
        'cb_2': {'linear': np.zeros((n_seeds, M.SEQ_LEN)), 'mlp': np.zeros((n_seeds, M.SEQ_LEN))}
    }

    REP_COLORS = {
        'ec': '#c9076c',
        'z_latent': '#ee77a8',
        'cb_1': '#ffa6a6',
        'cb_2': '#ffdc44'
    }

    va_trans, va_ctx = M.generate_sequences(n_seqs, seed=99)

    weight_rank = len(M.C_LABELS) // 2

    context_weight_dot_products = {key: [] for key in ['z_latent', 'ec', 'cb_1', 'cb_2']}

    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        ckpt = torch.load(ckpt_path, map_location=device)
        ec_i = M.EntorhinalCortex().to(device); ec_i.load_state_dict(ckpt["ec"])
        cb_i = M.CerebellumCB().to(device);     cb_i.load_state_dict(ckpt["cb"])
        ec_i.eval(); cb_i.eval()
        print(f"\n=== ckpt {ckpt_idx+1}/{len(ckpt_paths)}: "
              f"{os.path.basename(ckpt_path)} ===")
        
        z_i, c_i, s_i, a_i, ec_i, cb_1_i, cb_2_i = collect_z_state_action(ec_i, cb_i, va_trans, va_ctx, device)

        scaler_i = StandardScaler().fit(z_i)
        scaler_ec = StandardScaler().fit(ec_i)
        scaler_cb_1 = StandardScaler().fit(cb_1_i)
        scaler_cb_2 = StandardScaler().fit(cb_2_i)

        z_s_i    = scaler_i.transform(z_i)
        ec_s_i   = scaler_ec.transform(ec_i)
        cb_1_s_i = scaler_cb_1.transform(cb_1_i)
        cb_2_s_i = scaler_cb_2.transform(cb_2_i)

        z_seq = z_s_i.reshape(n_seqs, M.SEQ_LEN, z_i.shape[-1])
        ec_seq = ec_s_i.reshape(n_seqs, M.SEQ_LEN, ec_i.shape[-1])
        cb_1_seq = cb_1_s_i.reshape(n_seqs, M.SEQ_LEN, cb_1_i.shape[-1])
        cb_2_seq = cb_2_s_i.reshape(n_seqs, M.SEQ_LEN, cb_2_i.shape[-1])
        c_seq = c_i.reshape(n_seqs, M.SEQ_LEN)

        for rep, label in zip([z_seq, ec_seq, cb_1_seq, cb_2_seq], ['z_latent', 'ec', 'cb_1', 'cb_2']):
            acc_lin_t, acc_mlp_t = decode_acc_over_time(rep, c_seq)
            time_decoding[label]['linear'][ckpt_idx] = acc_lin_t
            time_decoding[label]['mlp'][ckpt_idx] = acc_mlp_t

            rep_flattened = rep.reshape(n_seqs * M.SEQ_LEN, -1)

            clf = LogisticRegression(max_iter=500)
            clf.fit(rep_flattened, c_i)

            W = clf.coef_                                    # (n_contexts, LATENT_DIM)
            U, S_sv, Vt = np.linalg.svd(W, full_matrices=False)
            W_tilde = U[:, :weight_rank] @ np.diag(S_sv[:weight_rank]) @ Vt[:weight_rank]
            W_normed = W_tilde / (np.linalg.norm(W_tilde,
             axis=1, keepdims=True) + 1e-8)
            
            context_weight_dot_products[label].append(
                W_normed @ W_normed.T
            )

            if ckpt_idx == 0:
                plot_pca(rep_flattened, Vt, c_i, title=label,
                          color_map=COLORS, base_dir=Path(M.FIG_DIR))
            
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
            # divide each row by its diagonal element
            diags = np.diag(C)
            C_norm = C / diags[:, np.newaxis]
            mlp_confs_norm.append(C_norm)
            lin_accs_all.append(lin_acc); mlp_accs_all.append(mlp_acc)


        # Keep the first checkpoint's tensors for the downstream analyses.
        if ckpt_idx == 0:
            z, c, s, a, z_s, ec_r, cb_1_r, cb_2_r = z_i, c_i, s_i, a_i, z_s_i, ec_s_i, cb_1_s_i, cb_2_s_i
            ec, cb          = ec_i, cb_i
            permuted_idxs = np.random.permutation(np.arange(c.shape[0]))
            ctx_ctrl = c[permuted_idxs]

            BOTTLENECK = 32

            print(f"\n--- Variance-regularized MLP (bottleneck={BOTTLENECK}, "
                f"CE + λ · MSE recon) ---")
            print("  λ=0      → plain MLP (free to use low-variance dims)")
            print("  λ→large  → bottleneck forced to also reconstruct z_ec;")
            print("            CE contribution vanishes ⇒ decode acc → chance\n")
            lambdas  = np.logspace(-6, 7, 25)

            var_regularized_data = {
                'ec': {'representation': ec_r, 'r2_dec': np.zeros((cross_val_folds, lambdas.shape[0])),
                        'r2_rep': np.zeros((cross_val_folds, lambdas.shape[0])), 'acc_ctrl': np.zeros((cross_val_folds, lambdas.shape[0])),
                        'rep_ctrl': np.zeros((cross_val_folds, lambdas.shape[0])), 'var_ratio': None,
                        'color': '#c9076c'},
                'z_latent': {'representation': z_s, 'r2_dec': np.zeros((cross_val_folds, lambdas.shape[0])),
                            'r2_rep': np.zeros((cross_val_folds, lambdas.shape[0])), 'acc_ctrl': np.zeros((cross_val_folds, lambdas.shape[0])),
                            'rep_ctrl': np.zeros((cross_val_folds, lambdas.shape[0])), 'var_ratio': None, 'color': "#ee77a8"},
                'cb_1': {'representation': cb_1_r, 'r2_dec': np.zeros((cross_val_folds, lambdas.shape[0])),
                        'r2_rep': np.zeros((cross_val_folds, lambdas.shape[0])), 'acc_ctrl': np.zeros((cross_val_folds, lambdas.shape[0])),
                        'rep_ctrl': np.zeros((cross_val_folds, lambdas.shape[0])), 'var_ratio': None, 'color': '#ffa6a6'},
                'cb_2': {'representation': cb_2_r, 'r2_dec': np.zeros((cross_val_folds, lambdas.shape[0])),
                        'r2_rep': np.zeros((cross_val_folds, lambdas.shape[0])), 'acc_ctrl': np.zeros((cross_val_folds, lambdas.shape[0])),
                        'rep_ctrl': np.zeros((cross_val_folds, lambdas.shape[0])), 'var_ratio': None, 'color': '#ffdc44'},
            }

            for key, data in var_regularized_data.items():
                print(key)
                representation = data['representation']
                r2_dec = data['r2_dec']
                r2_rep = data['r2_rep']

                acc_ctrl = data['acc_ctrl']
                rep_ctrl = data['rep_ctrl']

                for fold, (tr, te) in enumerate(skf.split(representation, c)):
                    for li, lam in enumerate(lambdas):
                        acc_v, r2_v = train_mlp_var(
                            representation[tr], c[tr], representation[te], c[te], device,
                            lam=lam, epochs=120, bottleneck=BOTTLENECK, input_dim=representation.shape[1]
                        )

                        c_acc, c_r2_v = train_mlp_var(
                            representation[tr], ctx_ctrl[tr], representation[te], ctx_ctrl[te], device,
                            lam=lam, epochs=120, bottleneck=BOTTLENECK, input_dim=representation.shape[1]
                        )

                        acc_ctrl[fold, li] = c_acc
                        rep_ctrl[fold, li] = c_r2_v
                        r2_dec[fold, li] = acc_v
                        r2_rep[fold, li] = r2_v

                print(f"  {key} - R² (decoding): {np.mean(r2_dec, axis=0)}")
                print(f"  {key} - R² (representation): {np.mean(r2_rep, axis=0)}")

                print(f"  {key} - Accuracy (control): {np.mean(acc_ctrl, axis=0)}")
                print(f"  {key} - R² (control): {np.mean(rep_ctrl, axis=0)}")

                data['r2_dec'] = r2_dec
                data['r2_rep'] = r2_rep
                data['acc_ctrl'] = acc_ctrl
                data['rep_ctrl'] = rep_ctrl

            fig, ax = plt.subplots(figsize=(3, 3))

            for key, data in var_regularized_data.items():
                r2_dec = data['r2_dec']
                r2_rep = data['r2_rep']

                acc_ctrl = data['acc_ctrl']
                rep_ctrl = data['rep_ctrl']

                color = REP_COLORS[key]

                dec_mean = np.mean(r2_dec, axis=0)
                dec_se = np.std(r2_dec, axis=0) / np.sqrt(cross_val_folds)

                rep_mean = np.mean(r2_rep, axis=0)
                rep_se = np.std(r2_rep, axis=0) / np.sqrt(cross_val_folds)

                dec_mean_c = np.mean(acc_ctrl, axis=0)
                dec_se_c = np.std(acc_ctrl, axis=0) / np.sqrt(cross_val_folds)

                rep_mean_c = np.mean(rep_ctrl, axis=0)
                rep_se_c = np.std(rep_ctrl, axis=0) / np.sqrt(cross_val_folds)

                ax.errorbar(rep_mean, dec_mean, xerr=rep_se, yerr=dec_se, color=color, fmt='o',
                            label=key, markersize=4)
                idx = np.argsort(rep_mean)
                ax.plot(rep_mean[idx], dec_mean[idx], color=color, linestyle='-')

                ax.errorbar(rep_mean_c, dec_mean_c, xerr=rep_se_c, yerr=dec_se_c, color=color, fmt='s', markersize=4)
                idx = np.argsort(rep_mean_c)
                ax.plot(rep_mean_c[idx], dec_mean_c[idx], color=color, linestyle='--')

            ax.hlines(1/N_CTX, 0, 1, color="black", linestyle="--", linewidth=1)

            ax.set_xlabel('Representation Var. Explained')
            ax.set_ylabel('Context Decode Accuracy')
            ax.set_xlim(0.25, 1)
            ax.set_ylim(0, 1)
            ax.legend()
            make_axis_nice(ax)
            out = os.path.join(M.FIG_DIR, "mlp12c_var_regularized_decode")
            plt.savefig(out, bbox_inches="tight"); plt.close()
            print(f"\nWrote {out}")

    for key, data in context_weight_dot_products.items():
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))

        dot_products = np.stack(data, axis=0)
        dot_product_means = dot_products.mean(axis=0)

        dist = 1 - dot_product_means

        mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=0, n_init=8, normalized_stress="auto")
        coords = mds.fit_transform(dist)

        g = ax[0].imshow(dot_product_means, cmap='RdBu', vmin=-1, vmax=1)
        fig.colorbar(g, ax=ax[0], pad=0.05, fraction=0.05, label='Dot Product')
        fig.suptitle(f"Dot Product and MDS for {key}")

        for i in range(N_CTX):
            ax[1].scatter(coords[i, 0], coords[i, 1],
                            s=180, color=COLORS[i], edgecolor="black", lw=0.7)
            ax[1].annotate(f"{C_LABELS[i]}", (coords[i, 0], coords[i, 1]),
                            textcoords="offset points", xytext=(8, 6),
                              fontsize=6, color=COLORS[i])
        
        ax[0].set_xticks(range(len(C_LABELS)))
        ax[0].set_xticklabels(C_LABELS.values(), rotation=90)
        ax[0].set_yticks(range(len(C_LABELS)))
        ax[0].set_yticklabels(C_LABELS.values())

        ax[1].set_xlabel("MDS dim 1"); ax[1].set_ylabel("MDS dim 2")
        ax[1].set_aspect("equal", adjustable="datalim")
        ax[1].grid(alpha=0.3)
        make_axis_nice(ax[1])
        
        fig.tight_layout()
        fig_path = os.path.join(M.FIG_DIR, f"{key}_dot_product_mds")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close()

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
    make_axis_nice(ax)
    plt.tight_layout()
    out = os.path.join(M.FIG_DIR, "mlp12c_vs_linear_decode.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\nWrote {out}")

    # 3a-tris. Variance-regularized MLP: accuracy vs (normalized) reconstruction.
    # x-axis is R²_bottleneck / R²_PCA-k, so PCA-k sits exactly at x=1 and any
    # bottleneck point with x>1 would mean the nonlinear encoder is packing
    # more variance into k dims than the optimal linear rank-k projection —
    # flag as a possible bug.


    # 3a-bis. Per-timestep decode curve — overlay linear vs MLP
    fig, ax = plt.subplots(1, 2, figsize=(4, 2), sharey=True)

    for key, data in time_decoding.items():
        acc_lin_t_mean = data['linear'].mean(axis=0)
        acc_lin_t_se = data['linear'].std(axis=0) / np.sqrt(n_seeds)

        acc_mlp_t_mean = data['mlp'].mean(axis=0)
        acc_mlp_t_se = data['mlp'].std(axis=0) / np.sqrt(n_seeds)

        color = REP_COLORS[key]

        ax[0].set_title("MLP-Decoding")
        ax[0].plot(acc_mlp_t_mean, color=color, label=key)
        ax[0].fill_between(range(len(acc_mlp_t_mean)), acc_mlp_t_mean - acc_mlp_t_se, acc_mlp_t_mean + acc_mlp_t_se, color=color, alpha=0.2)
        ax[0].set_ylabel("Context Decode Accuracy")

        ax[1].set_title("Linear-Decoding")
        ax[1].plot(acc_lin_t_mean, color=color, label=key)
        ax[1].fill_between(range(len(acc_lin_t_mean)), acc_lin_t_mean - acc_lin_t_se, acc_lin_t_mean + acc_lin_t_se, color=color, alpha=0.2)

    for axis in ax:
        axis.set_ylim(0, 1)
        axis.set_xlabel("Timestep t")
        axis.set_xlim(0, M.SEQ_LEN)

    ax[0].legend()
    make_axis_nice(fig)

    plt.tight_layout()
    out = os.path.join(M.FIG_DIR, "mlp12c_decodetime")
    plt.savefig(out, bbox_inches="tight"); #plt.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
